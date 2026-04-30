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

import contextlib
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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
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

# Stall watchdog: kill the ``claude --print`` subprocess if its stdout
# produces zero bytes for this long AND no idle-silence budget is in
# effect. Genuine hangs (model crashed mid-generate, deadlocked tool)
# get caught; legitimately silent stretches are protected by the
# subprocess calling ``mcp__tasque__claim_idle_silence`` first. Override
# via ``TASQUE_PROXY_STALL_SECONDS``; ``0`` (or negative) disables the
# kill entirely (the heartbeat log still fires).
DEFAULT_STALL_SECONDS = 300.0  # 5 minutes
DEFAULT_HEARTBEAT_SECONDS = 60.0  # idle-stretch log cadence
_STALL_POLL_INTERVAL = 1.0


def _resolve_stall_seconds() -> float:
    raw = os.environ.get("TASQUE_PROXY_STALL_SECONDS")
    if raw is None or raw.strip() == "":
        return DEFAULT_STALL_SECONDS
    try:
        return float(raw)
    except ValueError:
        log.warning(
            "proxy.stall_seconds_env_invalid",
            value=raw,
            falling_back_to=DEFAULT_STALL_SECONDS,
        )
        return DEFAULT_STALL_SECONDS


class ProxyError(Exception):
    """Raised when claude --print returns no parsable result event."""


class ProxyStallError(ProxyError):
    """Subprocess was killed by the stall watchdog (silent past threshold
    with no active idle-silence budget). Distinct from a wall-clock
    timeout: only fires when the LLM is producing nothing AND hasn't
    declared an upcoming silent stretch via ``claim_idle_silence``."""


class ProxyCancelledError(ProxyError):
    """Subprocess was killed via the explicit ``/v1/cancel/{id}`` endpoint."""


@dataclass
class _RequestRecord:
    """Live state for one in-flight ``claude --print`` invocation.

    Carried in :class:`_State.requests` while the request is running and
    removed when ``_invoke_claude`` returns. Both the watchdog thread
    and the HTTP handlers (``/v1/internal/idle_grant``, ``/v1/cancel``,
    ``/status``) read/write this through ``_State.requests_lock``.
    """

    request_id: str
    model: str
    started_at: float  # monotonic
    proc: subprocess.Popen[bytes]
    last_byte_at: float  # monotonic
    silence_budget_until: float | None = None  # monotonic, None if no budget
    silence_budget_reason: str | None = None
    silence_budget_seconds: float | None = None  # last grant size, for /status
    cancel_requested: bool = False
    stall_killed: bool = False
    cancelled: bool = False
    # Heartbeat tracking — the watchdog fires a log line every
    # ``DEFAULT_HEARTBEAT_SECONDS`` of silence so the operator sees the
    # subprocess go quiet before the watchdog kills.
    last_heartbeat_at: float = field(default_factory=time.monotonic)


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
        # Per-request registry. Keyed by request_id (uuid hex). The
        # lock guards both dict access and per-record field reads/writes
        # — records are dataclasses, not thread-safe on their own.
        self.requests: dict[str, _RequestRecord] = {}
        self.requests_lock = threading.Lock()
        # Public URL for the proxy's own internal endpoints, injected
        # into the ``claude --print`` subprocess env so the tasque MCP
        # can call ``/v1/internal/idle_grant``. Set by ``make_server``
        # once it knows the bound host:port.
        self.internal_base_url: str | None = None

    def healthz(self) -> bool:
        """Return True if ``claude --version`` succeeded recently (cached 30s)."""
        with self._healthz_lock:
            now = time.monotonic()
            if now - self._healthz_cached_at < HEALTHZ_CACHE_SECONDS:
                return self._healthz_cached_ok
            self._healthz_cached_ok = _claude_version_ok()
            self._healthz_cached_at = now
            return self._healthz_cached_ok

    def register_request(self, record: _RequestRecord) -> None:
        with self.requests_lock:
            self.requests[record.request_id] = record

    def unregister_request(self, request_id: str) -> None:
        with self.requests_lock:
            self.requests.pop(request_id, None)

    def get_request(self, request_id: str) -> _RequestRecord | None:
        with self.requests_lock:
            return self.requests.get(request_id)


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


def _extract_disallowed_tools(body: dict[str, Any]) -> list[str] | None:
    """Pull a per-request tool denylist out of an OpenAI-shaped body.

    Vendor extension: callers may pass ``disallowed_tools: list[str]`` at
    the top level of the request body, or nested under ``extra_body`` —
    both produce the same effect (forwarded to ``claude --print`` as
    ``--disallowedTools <comma-list>``). Used to gate user-action tools
    in contexts where the synchronous reply path has already executed
    the user's request (see the bucket-coach post-reply trigger).
    Returns ``None`` when nothing usable was passed.
    """
    raw = body.get("disallowed_tools")
    if raw is None:
        extra = body.get("extra_body")
        if isinstance(extra, dict):
            raw = cast(dict[str, Any], extra).get("disallowed_tools")
    if not isinstance(raw, list):
        return None
    cleaned = [str(t) for t in cast(list[Any], raw) if isinstance(t, str) and t.strip()]
    return cleaned or None


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
    disallowed_tools: list[str] | None = None,
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
    if disallowed_tools:
        argv.extend(["--disallowedTools", ",".join(disallowed_tools)])
    if not use_stdin:
        argv.extend(["-p", prompt])
    return argv


def _claude_cwd() -> Path:
    """Working directory for the ``claude --print`` subprocess.

    Project-scoped MCP servers in ``~/.claude.json`` are keyed by the
    invoking cwd. If the proxy's cwd doesn't match the tasque project
    root, ``claude --print`` only sees user-scoped MCPs — the worker
    LLM loses autopilot, slack, google-workspace, etc., and reports
    "no autopilot tool available" mid-chain. Pinning the cwd here makes
    the tool surface deterministic regardless of where ``tasque proxy``
    was launched from. Override with ``TASQUE_CLAUDE_CWD`` for
    non-standard layouts.
    """
    raw = os.environ.get("TASQUE_CLAUDE_CWD")
    if raw:
        return Path(raw)
    # src/tasque/proxy/server.py → project root is parents[3]
    return Path(__file__).resolve().parents[3]


def _invoke_claude(
    *,
    model: str,
    prompt: str,
    request_id: str,
    log_dir: Path,
    timeout: float | None,
    disallowed_tools: list[str] | None = None,
    state: _State | None = None,
    stall_seconds: float | None = None,
) -> tuple[str, dict[str, int]]:
    """Run ``claude --print`` and return (assembled_text, usage_dict).

    Streams stdout line-by-line so the operator can see tool calls,
    text starts, and message boundaries land in the proxy console as
    they happen — instead of one wall of output after the subprocess
    exits. The full raw stream is still buffered to a per-request
    JSONL file under ``log_dir`` for forensic replay.

    ``state`` is the per-process registry — when supplied, this run is
    registered so ``/v1/internal/idle_grant``, ``/v1/cancel``, and
    ``/status`` can address it by ``request_id``. ``stall_seconds`` is
    the silence-without-budget kill threshold (default
    :data:`DEFAULT_STALL_SECONDS`); pass ``0`` or negative to disable
    the kill (the heartbeat log still fires).

    Raises :class:`ProxyStallError` on stall kill, :class:`ProxyCancelledError`
    on explicit cancel, :class:`ProxyError` on timeout / missing /
    garbled stream-json / subprocess failure with no ``result`` event.
    """
    exe = shutil.which("claude")
    if exe is None:
        raise ProxyError("claude CLI not found on PATH")

    use_stdin = len(prompt.encode("utf-8")) >= PROMPT_STDIN_THRESHOLD
    argv = _build_claude_argv(
        exe=exe,
        model=model,
        prompt=prompt,
        use_stdin=use_stdin,
        disallowed_tools=disallowed_tools,
    )

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{request_id}.jsonl"
    short_id = request_id[:8]

    # Inject request id + internal URL so the tasque MCP can call
    # ``/v1/internal/idle_grant/{request_id}`` to declare upcoming
    # silent stretches without false-positive stall kills.
    env = os.environ.copy()
    env["TASQUE_PROXY_REQUEST_ID"] = request_id
    if state is not None and state.internal_base_url:
        env["TASQUE_PROXY_INTERNAL_URL"] = state.internal_base_url

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if use_stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(_claude_cwd()),
        env=env,
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
    effective_stall = (
        stall_seconds if stall_seconds is not None else _resolve_stall_seconds()
    )

    record = _RequestRecord(
        request_id=request_id,
        model=model,
        started_at=started,
        proc=proc,
        last_byte_at=started,
        last_heartbeat_at=started,
    )
    if state is not None:
        state.register_request(record)

    watchdog_done = threading.Event()

    def _watchdog() -> None:
        """Run while the subprocess is alive. Two responsibilities:

        - **Heartbeat**: every ``DEFAULT_HEARTBEAT_SECONDS`` of stdout
          silence, log a single ``proxy.idle_heartbeat`` line so the
          operator sees the subprocess gone quiet before any kill.
        - **Stall kill**: if silence exceeds ``effective_stall`` AND
          there's no active ``silence_budget_until`` AND the kill is
          enabled (``> 0``), kill the subprocess. The for-loop above
          then exits on EOF; the post-loop check converts that to a
          :class:`ProxyStallError`.

        Cancellation requests from ``/v1/cancel/{request_id}`` are also
        observed here — same kill path, different error class.
        """
        while not watchdog_done.is_set():
            if proc.poll() is not None:
                return
            now = time.monotonic()
            should_kill_stall = False
            should_kill_cancel = False
            if state is not None:
                with state.requests_lock:
                    rec = state.requests.get(request_id)
                    if rec is None:
                        return
                    if rec.cancel_requested and not rec.cancelled:
                        should_kill_cancel = True
                        rec.cancelled = True
                    elif effective_stall > 0:
                        idle = now - rec.last_byte_at
                        budget_until = rec.silence_budget_until
                        in_budget = budget_until is not None and now < budget_until
                        if idle > effective_stall and not in_budget:
                            should_kill_stall = True
                            rec.stall_killed = True
                        elif (
                            now - rec.last_heartbeat_at >= DEFAULT_HEARTBEAT_SECONDS
                            and idle >= DEFAULT_HEARTBEAT_SECONDS
                        ):
                            rec.last_heartbeat_at = now
                            log.info(
                                "proxy.idle_heartbeat",
                                request_id=short_id,
                                model=model,
                                idle_s=round(idle, 1),
                                budget_remaining_s=(
                                    round(budget_until - now, 1)
                                    if budget_until is not None and budget_until > now
                                    else None
                                ),
                                budget_reason=rec.silence_budget_reason,
                            )
            if should_kill_cancel:
                log.warning(
                    "proxy.cancel_kill",
                    request_id=short_id,
                    model=model,
                )
                with contextlib.suppress(Exception):
                    proc.kill()
                return
            if should_kill_stall:
                log.warning(
                    "proxy.stall_kill",
                    request_id=short_id,
                    model=model,
                    threshold_s=round(effective_stall, 1),
                )
                with contextlib.suppress(Exception):
                    proc.kill()
                return
            if watchdog_done.wait(timeout=_STALL_POLL_INTERVAL):
                return

    watchdog: threading.Thread | None = None
    if state is not None:
        watchdog = threading.Thread(
            target=_watchdog,
            name=f"proxy-watchdog-{short_id}",
            daemon=True,
        )
        watchdog.start()

    assert proc.stdout is not None
    try:
        for line_bytes in proc.stdout:
            if deadline is not None and time.monotonic() > deadline:
                proc.kill()
                proc.communicate()
                raise ProxyError(f"claude --print timed out after {timeout}s")
            now = time.monotonic()
            if state is not None:
                with state.requests_lock:
                    rec = state.requests.get(request_id)
                    if rec is not None:
                        rec.last_byte_at = now
                        rec.last_heartbeat_at = now
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
        watchdog_done.set()
        # Drain stderr now that stdout is done. ``communicate`` returns
        # the leftover bytes; we don't need stdout because we already
        # streamed it.
        try:
            _, stderr_bytes = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr_bytes = proc.communicate()
        if state is not None:
            state.unregister_request(request_id)
        if watchdog is not None:
            watchdog.join(timeout=2)

    # Watchdog may have killed the subprocess; raise the corresponding
    # error so callers can distinguish stall / cancel from other modes.
    if record.cancelled:
        raise ProxyCancelledError(
            f"claude --print cancelled via /v1/cancel (request_id={short_id})"
        )
    if record.stall_killed:
        raise ProxyStallError(
            f"claude --print killed by stall watchdog after "
            f"{round(effective_stall, 1)}s of stdout silence "
            f"with no idle-silence budget (request_id={short_id})"
        )

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
        "proxy.upstream_finished",
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
        if self.path.startswith("/v1/internal/idle_grant/"):
            request_id = self.path[len("/v1/internal/idle_grant/"):]
            self._handle_idle_grant(request_id)
            return
        if self.path.startswith("/v1/cancel/"):
            request_id = self.path[len("/v1/cancel/"):]
            self._handle_cancel(request_id)
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
        now_mono = time.monotonic()
        with self.state.requests_lock:
            request_view = [
                {
                    "request_id": r.request_id,
                    "model": r.model,
                    "elapsed_s": round(now_mono - r.started_at, 1),
                    "last_byte_age_s": round(now_mono - r.last_byte_at, 1),
                    "silence_budget_remaining_s": (
                        round(r.silence_budget_until - now_mono, 1)
                        if r.silence_budget_until is not None
                        and r.silence_budget_until > now_mono
                        else None
                    ),
                    "silence_budget_seconds": r.silence_budget_seconds,
                    "silence_budget_reason": r.silence_budget_reason,
                    "cancel_requested": r.cancel_requested,
                }
                for r in self.state.requests.values()
            ]
        self._send_json(
            HTTPStatus.OK,
            {
                "in_flight": in_flight,
                "capacity": self.state.max_concurrent,
                "total_requests": total,
                "last_error": self.state.last_error,
                "requests": request_view,
            },
        )

    def _handle_idle_grant(self, request_id: str) -> None:
        """Tasque-MCP-only endpoint: extend the silence budget for a
        running ``claude --print`` request.

        Body: ``{"seconds": <number>, "reason": "<short label>"}``. The
        budget is calculated from *now* — calling twice with the same
        ``seconds`` does not double it; the larger of (``now + seconds``,
        existing ``silence_budget_until``) wins. Returns 404 when the
        request_id is unknown (typical: subprocess already exited or the
        MCP guessed the wrong id), 400 on a malformed body.
        """
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": {"message": str(exc)}})
            return
        seconds_raw = body.get("seconds")
        try:
            seconds = float(seconds_raw)
        except (TypeError, ValueError):
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "seconds must be a positive number"}},
            )
            return
        if seconds <= 0:
            self._send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "seconds must be > 0"}},
            )
            return
        reason_raw = body.get("reason")
        reason = (
            str(reason_raw).strip()
            if isinstance(reason_raw, str) and reason_raw.strip()
            else None
        )
        now_mono = time.monotonic()
        new_until = now_mono + seconds
        with self.state.requests_lock:
            rec = self.state.requests.get(request_id)
            if rec is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": f"unknown request_id {request_id!r}"}},
                )
                return
            existing = rec.silence_budget_until
            chosen_until = (
                max(existing, new_until) if existing is not None else new_until
            )
            rec.silence_budget_until = chosen_until
            rec.silence_budget_seconds = seconds
            rec.silence_budget_reason = reason
            granted_remaining = chosen_until - now_mono
        granted_until_iso = (
            datetime.now(UTC) + timedelta(seconds=granted_remaining)
        ).isoformat()
        log.info(
            "proxy.idle_grant",
            request_id=request_id[:8],
            seconds=round(seconds, 1),
            remaining_s=round(granted_remaining, 1),
            reason=reason,
        )
        self._send_json(
            HTTPStatus.OK,
            {
                "ok": True,
                "request_id": request_id,
                "granted_until_iso": granted_until_iso,
                "remaining_s": round(granted_remaining, 1),
                "reason": reason,
            },
        )

    def _handle_cancel(self, request_id: str) -> None:
        """Operator-facing endpoint: kill a running ``claude --print``.

        Returns 404 when ``request_id`` is unknown (already finished or
        never started), 200 with ``{"ok": true}`` otherwise. The actual
        kill happens asynchronously in the watchdog within ~1s of this
        request returning; the corresponding ``/v1/chat/completions``
        call will then surface a :class:`ProxyCancelledError` as a 502.
        """
        with self.state.requests_lock:
            rec = self.state.requests.get(request_id)
            if rec is None:
                self._send_json(
                    HTTPStatus.NOT_FOUND,
                    {"error": {"message": f"unknown request_id {request_id!r}"}},
                )
                return
            rec.cancel_requested = True
        log.info("proxy.cancel_requested", request_id=request_id[:8])
        self._send_json(
            HTTPStatus.OK,
            {"ok": True, "request_id": request_id},
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
        disallowed_tools = _extract_disallowed_tools(body)
        request_id = uuid.uuid4().hex
        log.info(
            "proxy.request_received",
            request_id=request_id[:8],
            model=model,
            messages=len(messages_typed),
            prompt_kb=round(len(prompt.encode("utf-8")) / 1024, 1),
            disallowed_tools=len(disallowed_tools) if disallowed_tools else 0,
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
                    disallowed_tools=disallowed_tools,
                    state=self.state,
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
        log.info(
            "proxy.response_sent",
            request_id=request_id[:8],
            model=model,
            chars=len(text),
            in_tokens=usage.get("input_tokens", 0),
            out_tokens=usage.get("output_tokens", 0),
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
    # Pin the public-facing URL the tasque MCP uses to call the proxy's
    # internal endpoints (``/v1/internal/idle_grant/<request_id>``).
    # ``host=0.0.0.0`` would bind all interfaces but the MCP runs on
    # the same machine, so 127.0.0.1 is the right address to publish.
    bound_host, bound_port = server.server_address[0], server.server_address[1]
    advertised_host = "127.0.0.1" if bound_host in ("0.0.0.0", "::") else bound_host
    state.internal_base_url = f"http://{advertised_host}:{bound_port}"
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
        claude_cwd=str(_claude_cwd()),
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
