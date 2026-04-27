"""OpenAI-compat HTTP wrapper around ``claude --print``.

The proxy is a subprocess shim. It receives an OpenAI-shaped chat-completions
request, flattens the messages into a single prompt, invokes the host's
``claude --print`` CLI in stream-json mode, parses the resulting JSONL stream
for the final ``result`` event, and returns an OpenAI-shaped response.

The LLM-side tool calls happen entirely inside ``claude --print`` against
whatever MCPs the user has configured in ``~/.claude.json``. The proxy never
sees the MCP traffic — that is the whole point of this design.
"""

from __future__ import annotations

import json
import os
import shutil
import socketserver
import struct
import subprocess
import threading
import time
import uuid
from collections.abc import Iterable
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, ClassVar, cast

import structlog

log = structlog.get_logger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3456
DEFAULT_MAX_CONCURRENT = 4
HEALTHZ_CACHE_SECONDS = 30
LOG_RETENTION_SECONDS = 7 * 24 * 60 * 60
# Keep below the Windows CreateProcess command-line cap (~32,767 chars) with
# headroom for the rest of argv and the env block. Larger prompts are piped
# via stdin instead of `-p`.
PROMPT_STDIN_THRESHOLD = 8 * 1024  # bytes


class ProxyError(Exception):
    """Raised when claude --print returns no parsable result event."""


class _State:
    """Process-global counters and bookkeeping shared across handler threads."""

    def __init__(self, max_concurrent: int, log_dir: Path) -> None:
        self.max_concurrent = max_concurrent
        self.semaphore = threading.Semaphore(max_concurrent)
        self.in_flight = 0
        self.in_flight_lock = threading.Lock()
        self.total_requests = 0
        self.last_error: str | None = None
        self.log_dir = log_dir
        self._healthz_cached_at: float = 0.0
        self._healthz_cached_ok: bool = False
        self._healthz_lock = threading.Lock()

    def healthz(self) -> bool:
        """Return True if ``claude --version`` succeeded recently (cached 30s)."""
        with self._healthz_lock:
            now = time.monotonic()
            if now - self._healthz_cached_at < HEALTHZ_CACHE_SECONDS:
                return self._healthz_cached_ok
            self._healthz_cached_ok = _claude_version_ok()
            self._healthz_cached_at = now
            return self._healthz_cached_ok


def _claude_version_ok() -> bool:
    exe = shutil.which("claude")
    if exe is None:
        return False
    try:
        result = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    return result.returncode == 0


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    """Collapse OpenAI-style messages into a single text prompt for claude --print.

    System messages are concatenated at the top. User/assistant turns follow,
    each prefixed with their role label. ``content`` may be a plain string or
    an OpenAI-style list of ``{"type": "text", "text": ...}`` parts.
    """
    system_parts: list[str] = []
    turns: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = _coerce_content(msg.get("content", ""))
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        label = "Assistant" if role == "assistant" else "User"
        turns.append(f"{label}: {content}")
    chunks: list[str] = []
    if system_parts:
        chunks.append("\n\n".join(system_parts))
    if turns:
        chunks.append("\n\n".join(turns))
    return "\n\n".join(chunks)


def _coerce_content(raw: Any) -> str:
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts: list[str] = []
        for item in cast(list[Any], raw):
            if isinstance(item, dict):
                d = cast(dict[str, Any], item)
                text_val = d.get("text")
                if isinstance(text_val, str):
                    parts.append(text_val)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(raw)


def _build_claude_argv(
    *,
    exe: str,
    model: str,
    prompt: str,
    use_stdin: bool,
) -> list[str]:
    argv = [
        exe,
        "--print",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--verbose",
        "--model",
        model,
        "--tools",
        "default",
        "--allowedTools",
        "*",
        "--permission-mode",
        "bypassPermissions",
    ]
    if not use_stdin:
        argv.extend(["-p", prompt])
    return argv


def _invoke_claude(
    *,
    model: str,
    prompt: str,
    request_id: str,
    log_dir: Path,
    timeout: float | None,
) -> tuple[str, dict[str, int]]:
    """Run ``claude --print`` and return (assembled_text, usage_dict).

    Streams stdout line-by-line so the operator can see tool calls,
    text starts, and message boundaries land in the proxy console as
    they happen — instead of one wall of output after the subprocess
    exits. The full raw stream is still buffered to a per-request
    JSONL file under ``log_dir`` for forensic replay.

    Raises ``ProxyError`` on missing/garbled stream-json or on subprocess
    failure that did not produce a ``result`` event.
    """
    exe = shutil.which("claude")
    if exe is None:
        raise ProxyError("claude CLI not found on PATH")

    use_stdin = len(prompt.encode("utf-8")) >= PROMPT_STDIN_THRESHOLD
    argv = _build_claude_argv(exe=exe, model=model, prompt=prompt, use_stdin=use_stdin)

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{request_id}.jsonl"
    short_id = request_id[:8]

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if use_stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if use_stdin:
        # stdin is closed before we start reading stdout to avoid the
        # classic deadlock where the child fills the stdout pipe while
        # we sit on stdin.
        try:
            assert proc.stdin is not None
            proc.stdin.write(prompt.encode("utf-8"))
            proc.stdin.close()
        except (OSError, BrokenPipeError) as exc:
            proc.kill()
            proc.communicate()
            raise ProxyError(f"failed to feed prompt over stdin: {exc}") from exc

    deadline = time.monotonic() + timeout if timeout is not None else None
    raw_lines: list[str] = []
    tool_calls = 0
    saw_first_text = False
    started = time.monotonic()

    assert proc.stdout is not None
    try:
        for line_bytes in proc.stdout:
            if deadline is not None and time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                raise ProxyError(f"claude --print timed out after {timeout}s")
            line_text = line_bytes.decode("utf-8", errors="replace")
            raw_lines.append(line_text)
            stripped = line_text.strip()
            if not stripped:
                continue
            try:
                event = cast(dict[str, Any], json.loads(stripped))
            except json.JSONDecodeError:
                # Non-JSON noise; just buffer and move on.
                continue
            new_tool_calls, saw_first_text = _log_stream_event(
                event,
                request_id_short=short_id,
                model=model,
                started_at=started,
                saw_first_text=saw_first_text,
            )
            tool_calls += new_tool_calls
    finally:
        # Drain stderr now that stdout is done. ``communicate`` returns
        # the leftover bytes; we don't need stdout because we already
        # streamed it.
        try:
            _, stderr_bytes = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_bytes = proc.communicate()

    stdout_text = "".join(raw_lines)
    stderr_text = stderr_bytes.decode("utf-8", errors="replace")

    try:
        log_path.write_text(stdout_text, encoding="utf-8")
    except OSError as exc:
        log.warning("proxy.log_write_failed", path=str(log_path), error=str(exc))

    if not stdout_text.strip():
        raise ProxyError(
            f"claude --print returned empty stdout (exit={proc.returncode}); "
            f"stderr={stderr_text.strip()[:500]}"
        )

    text, usage = _parse_stream_json(
        stdout_text, exit_code=proc.returncode, stderr=stderr_text
    )
    log.info(
        "proxy.response_sent",
        request_id=short_id,
        model=model,
        duration_s=round(time.monotonic() - started, 2),
        tool_calls=tool_calls,
        in_tokens=usage.get("input_tokens", 0),
        out_tokens=usage.get("output_tokens", 0),
        chars=len(text),
    )
    return text, usage


def _log_stream_event(
    event: dict[str, Any],
    *,
    request_id_short: str,
    model: str,
    started_at: float,
    saw_first_text: bool,
) -> tuple[int, bool]:
    """Emit user-visible log lines for the events that matter.

    Returns ``(tool_calls_added_in_this_event, saw_first_text_flag)``.
    Quiet on the noisy stuff (text deltas, message_delta usage). Loud
    on session init, tool calls, and the first text chunk per message.
    """
    new_tool_calls = 0
    etype = event.get("type")

    if etype == "system" and event.get("subtype") == "init":
        session_id = event.get("session_id", "")
        log.info(
            "proxy.session_init",
            request_id=request_id_short,
            model=model,
            session_id=str(session_id)[:8],
            tools=len(event.get("tools") or []),
        )
        return new_tool_calls, saw_first_text

    if etype == "stream_event":
        inner = event.get("event") or {}
        inner_type = inner.get("type") if isinstance(inner, dict) else None
        if inner_type == "content_block_start":
            block = inner.get("content_block") or {}
            block_type = block.get("type") if isinstance(block, dict) else None
            if block_type == "tool_use":
                name = block.get("name") or "?"
                tool_id = (block.get("id") or "")[:12]
                log.info(
                    "proxy.tool_call",
                    request_id=request_id_short,
                    tool=str(name),
                    tool_id=str(tool_id),
                    elapsed_s=round(time.monotonic() - started_at, 2),
                )
                new_tool_calls = 1
            elif block_type == "text" and not saw_first_text:
                # First text chunk of the message — log so the operator
                # knows the model is producing prose. Subsequent deltas
                # stay silent to avoid one log line per token.
                log.info(
                    "proxy.text_started",
                    request_id=request_id_short,
                    elapsed_s=round(time.monotonic() - started_at, 2),
                )
                saw_first_text = True
        elif inner_type == "message_stop":
            saw_first_text = False  # next message resets the gate

    elif etype == "user":
        # Tool result coming back from the host's MCP. Useful breadcrumb
        # so the operator sees the round-trip, but skip the payload —
        # tool results can be huge.
        log.info(
            "proxy.tool_result",
            request_id=request_id_short,
            elapsed_s=round(time.monotonic() - started_at, 2),
        )

    return new_tool_calls, saw_first_text


def _parse_stream_json(
    stdout_text: str,
    *,
    exit_code: int,
    stderr: str,
) -> tuple[str, dict[str, int]]:
    """Walk the stream-json transcript and return (text, usage)."""
    result_text: str | None = None
    usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
    saw_any_event = False

    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = cast(dict[str, Any], json.loads(line))
        except json.JSONDecodeError:
            # A truncated final line is the most common cause; keep walking.
            continue
        saw_any_event = True
        if event.get("type") != "result":
            continue
        raw_result = event.get("result")
        if isinstance(raw_result, str):
            result_text = raw_result
        raw_usage = event.get("usage")
        if isinstance(raw_usage, dict):
            u = cast(dict[str, Any], raw_usage)
            input_tokens = u.get("input_tokens", 0)
            output_tokens = u.get("output_tokens", 0)
            if isinstance(input_tokens, int):
                usage["input_tokens"] = input_tokens
            if isinstance(output_tokens, int):
                usage["output_tokens"] = output_tokens

    if result_text is None:
        snippet = stderr.strip()[:500] if stderr.strip() else stdout_text.strip()[-500:]
        if not saw_any_event:
            raise ProxyError(
                f"claude --print produced no JSON events (exit={exit_code}); snippet={snippet}"
            )
        raise ProxyError(
            f"claude --print stream missing result event (exit={exit_code}); snippet={snippet}"
        )
    return result_text, usage


def _prune_old_logs(log_dir: Path) -> None:
    if not log_dir.exists():
        return
    cutoff = time.time() - LOG_RETENTION_SECONDS
    for path in log_dir.iterdir():
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            continue


class ProxyHandler(BaseHTTPRequestHandler):
    """Per-connection request handler. One instance per request thread."""

    state: ClassVar[_State]
    request_timeout: ClassVar[float | None] = None

    server_version = "tasque-proxy/0.1"
    sys_version = ""

    def log_message(self, format: str, *args: Any) -> None:
        log.debug("proxy.http", line=format % args, addr=self.client_address)

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._handle_healthz()
            return
        if self.path == "/status":
            self._handle_status()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    def do_POST(self) -> None:
        if self.path == "/v1/chat/completions":
            self._handle_chat_completions()
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "not found"}})

    # ------------------------------------------------------------------ handlers

    def _handle_healthz(self) -> None:
        if self.state.healthz():
            self._send_json(HTTPStatus.OK, {"status": "ok"})
        else:
            self._send_json(
                HTTPStatus.SERVICE_UNAVAILABLE,
                {"status": "claude CLI unavailable"},
            )

    def _handle_status(self) -> None:
        with self.state.in_flight_lock:
            in_flight = self.state.in_flight
            total = self.state.total_requests
        self._send_json(
            HTTPStatus.OK,
            {
                "in_flight": in_flight,
                "capacity": self.state.max_concurrent,
                "total_requests": total,
                "last_error": self.state.last_error,
            },
        )

    def _handle_chat_completions(self) -> None:
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return

        messages_raw = body.get("messages")
        if not isinstance(messages_raw, list) or not messages_raw:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "messages must be a non-empty list"}},
            )
            return
        model = body.get("model")
        if not isinstance(model, str) or not model:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "model must be a non-empty string"}},
            )
            return

        messages_typed: list[dict[str, Any]] = []
        for item in cast(list[Any], messages_raw):
            if isinstance(item, dict):
                messages_typed.append(cast(dict[str, Any], item))
        prompt = _flatten_messages(messages_typed)
        request_id = uuid.uuid4().hex
        log.info(
            "proxy.request_received",
            request_id=request_id[:8],
            model=model,
            messages=len(messages_typed),
            prompt_kb=round(len(prompt.encode("utf-8")) / 1024, 1),
        )

        with self.state.semaphore:
            with self.state.in_flight_lock:
                self.state.in_flight += 1
                self.state.total_requests += 1
            try:
                text, usage = _invoke_claude(
                    model=model,
                    prompt=prompt,
                    request_id=request_id,
                    log_dir=self.state.log_dir,
                    timeout=self.request_timeout,
                )
            except ProxyError as exc:
                self.state.last_error = str(exc)
                log.warning(
                    "proxy.upstream_error",
                    request_id=request_id,
                    model=model,
                    error=str(exc),
                )
                self._send_json(
                    HTTPStatus.BAD_GATEWAY,
                    {"error": {"message": str(exc), "request_id": request_id}},
                )
                return
            except Exception as exc:
                self.state.last_error = repr(exc)
                log.exception("proxy.unhandled", request_id=request_id, model=model)
                self._send_json(
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                    {"error": {"message": "internal proxy error", "request_id": request_id}},
                )
                return
            finally:
                with self.state.in_flight_lock:
                    self.state.in_flight -= 1

        self._send_json(
            HTTPStatus.OK,
            _build_completion_response(
                request_id=request_id,
                model=model,
                text=text,
                usage=usage,
            ),
        )

    # ----------------------------------------------------------------- helpers

    def _read_json_body(self) -> dict[str, Any]:
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ValueError("missing Content-Length header")
        try:
            length = int(length_header)
        except ValueError as exc:
            raise ValueError("invalid Content-Length header") from exc
        if length < 0:
            raise ValueError("negative Content-Length")
        raw = self.rfile.read(length) if length else b""
        try:
            decoded = json.loads(raw.decode("utf-8")) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc
        if not isinstance(decoded, dict):
            raise ValueError("body must be a JSON object")
        return cast(dict[str, Any], decoded)

    def _send_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        try:
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, struct.error):
            # Client hung up before we finished — nothing to do.
            pass


def _build_completion_response(
    *,
    request_id: str,
    model: str,
    text: str,
    usage: dict[str, int],
) -> dict[str, Any]:
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return {
        "id": f"chatcmpl-{request_id}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
    }


class ThreadingHTTPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """``ThreadingMixIn`` over ``TCPServer`` — one thread per request."""

    daemon_threads = True
    allow_reuse_address = True


def make_server(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_concurrent: int | None = None,
    log_dir: Path | None = None,
    timeout: float | None = None,
) -> tuple[ThreadingHTTPServer, _State]:
    """Wire up the handler class with shared state and return (server, state)."""
    cap = max_concurrent if max_concurrent is not None else _env_max_concurrent()
    resolved_log_dir = log_dir if log_dir is not None else _env_log_dir()
    resolved_timeout = timeout if timeout is not None else _env_timeout()

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(resolved_log_dir)

    state = _State(max_concurrent=cap, log_dir=resolved_log_dir)

    handler_cls = type(
        "BoundProxyHandler",
        (ProxyHandler,),
        {"state": state, "request_timeout": resolved_timeout},
    )
    server = ThreadingHTTPServer((host, port), cast(Any, handler_cls))
    return server, state


def _env_max_concurrent() -> int:
    raw = os.environ.get("TASQUE_PROXY_MAX_CONCURRENT")
    if raw is None:
        return DEFAULT_MAX_CONCURRENT
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONCURRENT
    return max(1, value)


def _env_log_dir() -> Path:
    raw = os.environ.get("TASQUE_PROXY_LOG_DIR")
    return Path(raw) if raw else Path("data") / "proxy-logs"


def _env_timeout() -> float | None:
    raw = os.environ.get("TASQUE_PROXY_TIMEOUT")
    if raw is None:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    max_concurrent: int | None = None,
    log_dir: Path | None = None,
    timeout: float | None = None,
    iterations: Iterable[None] | None = None,  # tests can pass [None] for one-shot
) -> None:
    """Start the proxy and block until interrupted.

    ``iterations`` is exposed only for tests that want to handle a fixed
    number of requests; production callers leave it as ``None`` so the
    server runs forever via ``serve_forever``.
    """
    server, state = make_server(
        host=host,
        port=port,
        max_concurrent=max_concurrent,
        log_dir=log_dir,
        timeout=timeout,
    )
    log.info(
        "proxy.start",
        host=host,
        port=port,
        max_concurrent=state.max_concurrent,
        log_dir=str(state.log_dir),
        timeout=timeout,
    )
    try:
        if iterations is None:
            server.serve_forever()
        else:
            for _ in iterations:
                server.handle_request()
    except KeyboardInterrupt:
        log.info("proxy.stop", reason="keyboard_interrupt")
    finally:
        server.server_close()
