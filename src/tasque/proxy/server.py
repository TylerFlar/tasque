"""OpenAI-compat HTTP wrapper around a local model CLI.

The proxy is a subprocess shim. It receives an OpenAI-shaped chat-completions
request, flattens the messages into a single prompt, invokes the selected
upstream CLI (Claude or Codex), parses its JSONL stream, and returns an
OpenAI-shaped response.

The LLM-side tool calls happen entirely inside the upstream CLI against
whatever MCPs the user has configured for that host. The proxy never sees the
MCP traffic — that is the whole point of this design.
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
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import structlog

log = structlog.get_logger(__name__)

Upstream = Literal["claude", "codex"]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 3456
DEFAULT_MAX_CONCURRENT = 4
HEALTHZ_CACHE_SECONDS = 30
LOG_RETENTION_SECONDS = 7 * 24 * 60 * 60
# Keep below the Windows CreateProcess command-line cap (~32,767 chars) with
# headroom for the rest of argv and the env block. Larger prompts are piped
# via stdin instead of `-p`.
PROMPT_STDIN_THRESHOLD = 8 * 1024  # bytes

# Idle monitor: log long quiet stretches so operators can tell a request is
# still in-flight. It never kills or times out the upstream CLI.
IDLE_HEARTBEAT_SECONDS = 60.0
_IDLE_MONITOR_POLL_SECONDS = 1.0


class ProxyError(Exception):
    """Raised when the upstream CLI returns no parsable result event."""


@dataclass
class _RequestRecord:
    """Live state for one in-flight upstream CLI invocation.

    Carried in :class:`_State.requests` while the request is running and
    removed when ``_invoke_upstream`` returns. Both the idle monitor thread
    and the HTTP handlers read/write this through ``_State.requests_lock``.
    """

    request_id: str
    model: str
    upstream: Upstream
    started_at: float  # monotonic
    proc: subprocess.Popen[bytes]
    last_byte_at: float  # monotonic
    # Heartbeat tracking: the idle monitor emits an observability-only log line
    # every ``IDLE_HEARTBEAT_SECONDS`` of stdout silence.
    last_heartbeat_at: float = field(default_factory=time.monotonic)


class _State:
    """Process-global counters and bookkeeping shared across handler threads."""

    def __init__(
        self,
        max_concurrent: int,
        log_dir: Path,
        upstream: Upstream,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.upstream: Upstream = upstream
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

    def healthz(self) -> bool:
        """Return True if the selected upstream CLI responds (cached 30s)."""
        with self._healthz_lock:
            now = time.monotonic()
            if now - self._healthz_cached_at < HEALTHZ_CACHE_SECONDS:
                return self._healthz_cached_ok
            self._healthz_cached_ok = _upstream_version_ok(self.upstream)
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


def _env_upstream() -> Upstream:
    raw = os.environ.get("TASQUE_PROXY_UPSTREAM")
    if raw is None or not raw.strip():
        raise ProxyError("TASQUE_PROXY_UPSTREAM is required: set it to 'claude' or 'codex'")
    normalized = raw.strip().lower()
    if normalized in ("claude", "codex"):
        return normalized
    raise ProxyError(
        f"TASQUE_PROXY_UPSTREAM must be 'claude' or 'codex', got {raw!r}"
    )


def _upstream_executable(upstream: Upstream) -> str:
    return "codex" if upstream == "codex" else "claude"


def _upstream_display(upstream: Upstream) -> str:
    return "codex exec" if upstream == "codex" else "claude --print"


def _upstream_version_ok(upstream: Upstream) -> bool:
    exe = shutil.which(_upstream_executable(upstream))
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
    """Collapse OpenAI-style messages into a single text prompt.

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
    both produce the same effect. Claude receives
    ``--disallowedTools <comma-list>``; upstreams without a native deny
    flag receive an instruction in-prompt. Used to gate user-action
    tools in contexts that should only consolidate state, not start new
    user-visible work.
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


def _build_codex_argv(
    *,
    exe: str,
    model: str,
    cwd: Path,
    mcp_servers: dict[str, dict[str, Any]] | None = None,
) -> list[str]:
    argv = [
        exe,
        "exec",
    ]
    for override in _codex_mcp_config_overrides(mcp_servers or {}):
        argv.extend(["-c", override])
    argv.extend(
        [
            "--json",
            "--ephemeral",
            "--skip-git-repo-check",
            "--dangerously-bypass-approvals-and-sandbox",
            "--model",
            model,
            "--cd",
            str(cwd),
            "-",
        ]
    )
    return argv


def _json_toml(value: str | list[str]) -> str:
    """Render a TOML-compatible scalar or array for ``codex -c``."""
    return json.dumps(value)


def _inline_toml_table(value: dict[str, str]) -> str:
    pairs = ", ".join(
        f"{json.dumps(key)} = {json.dumps(val)}" for key, val in sorted(value.items())
    )
    return "{" + pairs + "}"


def _codex_mcp_config_overrides(
    mcp_servers: dict[str, dict[str, Any]],
) -> list[str]:
    overrides: list[str] = []
    for server_name in sorted(mcp_servers):
        server = mcp_servers[server_name]
        command = server.get("command")
        if not isinstance(command, str) or not command:
            continue
        raw_args = server.get("args", [])
        args = [str(arg) for arg in raw_args] if isinstance(raw_args, list) else []
        raw_env = server.get("env", {})
        env = (
            {str(k): str(v) for k, v in raw_env.items()}
            if isinstance(raw_env, dict)
            else {}
        )
        prefix = f"mcp_servers.{server_name}"
        overrides.append(f"{prefix}.command={_json_toml(command)}")
        overrides.append(f"{prefix}.args={_json_toml(args)}")
        if env:
            overrides.append(f"{prefix}.env={_inline_toml_table(env)}")
    return overrides


def _local_codex_mcp_servers() -> dict[str, dict[str, Any]]:
    """MCP servers tasque itself requires every Codex proxy turn to have.

    User/project Codex config still provides the broad personal tool catalog.
    These local registrations make the daemon's own result-submission MCP and
    the trading tool surface deterministic even when the mirrored user config
    is stale or incomplete.
    """
    root = _project_root()
    servers: dict[str, dict[str, Any]] = {
        "tasque": {
            "command": "uv",
            "args": [
                "run",
                "--directory",
                root.as_posix(),
                "tasque",
                "mcp",
            ],
        }
    }
    trading_project = root / "mcps" / "trading-mcp"
    if trading_project.exists():
        servers["trading-mcp"] = {
            "command": "uv",
            "args": [
                "run",
                "--project",
                trading_project.as_posix(),
                "trading-mcp-server",
            ],
        }
    return servers


def _prepend_disallowed_tool_instruction(prompt: str, disallowed_tools: list[str]) -> str:
    denylist = "\n".join(f"- {name}" for name in disallowed_tools)
    return (
        "System safety constraint for this turn: do not call any MCP/tool with "
        "one of these exact names if it is available.\n"
        f"{denylist}\n\n"
        f"{prompt}"
    )


def _project_root() -> Path:
    # src/tasque/proxy/server.py -> project root is parents[3]
    return Path(__file__).resolve().parents[3]


def _upstream_cwd(upstream: Upstream) -> Path:
    """Working directory for the upstream CLI subprocess.

    Project-scoped MCP servers in ``~/.claude.json`` are keyed by the
    invoking cwd. If the proxy's cwd doesn't match the tasque project
    root, ``claude --print`` only sees user-scoped MCPs — the worker
    LLM loses autopilot, slack, google-workspace, etc., and reports
    "no autopilot tool available" mid-chain. Pinning the cwd here makes
    the tool surface deterministic regardless of where ``tasque proxy``
    was launched from. Codex also receives the cwd as ``--cd`` so it
    starts in the same trusted project. Override with
    ``TASQUE_CLAUDE_CWD`` / ``TASQUE_CODEX_CWD`` or the generic
    ``TASQUE_PROXY_CWD`` for non-standard layouts.
    """
    provider_env = (
        os.environ.get("TASQUE_CODEX_CWD")
        if upstream == "codex"
        else os.environ.get("TASQUE_CLAUDE_CWD")
    )
    raw = provider_env or os.environ.get("TASQUE_PROXY_CWD")
    if raw:
        return Path(raw)
    return _project_root()


def _invoke_upstream(
    *,
    upstream: Upstream,
    model: str,
    prompt: str,
    request_id: str,
    log_dir: Path,
    disallowed_tools: list[str] | None = None,
    state: _State | None = None,
) -> tuple[str, dict[str, int]]:
    """Run the selected upstream CLI and return (assembled_text, usage_dict).

    Streams stdout line-by-line so the operator can see tool calls,
    text starts, and message boundaries land in the proxy console as
    they happen — instead of one wall of output after the subprocess
    exits. The full raw stream is still buffered to a per-request
    JSONL file under ``log_dir`` for forensic replay.

    ``state`` is the per-process registry. When supplied, this run appears in
    ``/status`` until the upstream process exits.

    Raises :class:`ProxyError` on missing / garbled stream-json or subprocess
    failure with no ``result`` event.
    """
    exe_name = _upstream_executable(upstream)
    upstream_label = _upstream_display(upstream)
    exe = shutil.which(exe_name)
    if exe is None:
        raise ProxyError(f"{exe_name} CLI not found on PATH")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{request_id}.jsonl"
    short_id = request_id[:8]

    upstream_cwd = _upstream_cwd(upstream)
    if upstream == "codex":
        use_stdin = True
        if disallowed_tools:
            prompt = _prepend_disallowed_tool_instruction(prompt, disallowed_tools)
        argv = _build_codex_argv(
            exe=exe,
            model=model,
            cwd=upstream_cwd,
            mcp_servers=_local_codex_mcp_servers(),
        )
    else:
        use_stdin = len(prompt.encode("utf-8")) >= PROMPT_STDIN_THRESHOLD
        argv = _build_claude_argv(
            exe=exe,
            model=model,
            prompt=prompt,
            use_stdin=use_stdin,
            disallowed_tools=disallowed_tools,
        )

    proc = subprocess.Popen(
        argv,
        stdin=subprocess.PIPE if use_stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(upstream_cwd),
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
            proc.wait()
            raise ProxyError(f"failed to feed prompt over stdin: {exc}") from exc

    raw_lines: list[str] = []
    stderr_chunks: list[bytes] = []
    tool_calls = 0
    saw_first_text = False
    started = time.monotonic()

    record = _RequestRecord(
        request_id=request_id,
        model=model,
        upstream=upstream,
        started_at=started,
        proc=proc,
        last_byte_at=started,
        last_heartbeat_at=started,
    )
    if state is not None:
        state.register_request(record)

    monitor_done = threading.Event()

    def _idle_monitor() -> None:
        """Log quiet in-flight requests without enforcing a deadline."""
        while not monitor_done.is_set():
            if proc.poll() is not None:
                return
            now = time.monotonic()
            if state is not None:
                with state.requests_lock:
                    rec = state.requests.get(request_id)
                    if rec is None:
                        return
                    idle = now - rec.last_byte_at
                    if (
                        now - rec.last_heartbeat_at >= IDLE_HEARTBEAT_SECONDS
                        and idle >= IDLE_HEARTBEAT_SECONDS
                    ):
                        rec.last_heartbeat_at = now
                        log.info(
                            "proxy.idle_heartbeat",
                            request_id=short_id,
                            model=model,
                            idle_s=round(idle, 1),
                        )
            if monitor_done.wait(timeout=_IDLE_MONITOR_POLL_SECONDS):
                return

    def _drain_stderr() -> None:
        stream = proc.stderr
        if stream is None:
            return
        while True:
            chunk = stream.read(8192)
            if not chunk:
                return
            stderr_chunks.append(chunk)

    monitor: threading.Thread | None = None
    if state is not None:
        monitor = threading.Thread(
            target=_idle_monitor,
            name=f"proxy-idle-monitor-{short_id}",
            daemon=True,
        )
        monitor.start()

    stderr_thread = threading.Thread(
        target=_drain_stderr,
        name=f"proxy-stderr-drain-{short_id}",
        daemon=True,
    )
    stderr_thread.start()

    assert proc.stdout is not None
    try:
        for line_bytes in proc.stdout:
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
        returncode = proc.wait()
        monitor_done.set()
        if state is not None:
            state.unregister_request(request_id)
        if monitor is not None:
            monitor.join(timeout=2)
        stderr_thread.join(timeout=2)

    stdout_text = "".join(raw_lines)
    stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")

    try:
        log_path.write_text(stdout_text, encoding="utf-8")
    except OSError as exc:
        log.warning("proxy.log_write_failed", path=str(log_path), error=str(exc))

    if not stdout_text.strip():
        raise ProxyError(
            f"{upstream_label} returned empty stdout (exit={returncode}); "
            f"stderr={stderr_text.strip()[:500]}"
        )
    else:
        text, usage = _parse_upstream_json(
            stdout_text,
            upstream=upstream,
            exit_code=returncode,
            stderr=stderr_text,
        )
    log.info(
        "proxy.upstream_finished",
        request_id=short_id,
        upstream=upstream,
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

    if etype == "thread.started":
        thread_id = event.get("thread_id", "")
        log.info(
            "proxy.session_init",
            request_id=request_id_short,
            model=model,
            session_id=str(thread_id)[:8],
            tools=None,
        )
        return new_tool_calls, saw_first_text

    if etype == "turn.started":
        log.info(
            "proxy.turn_started",
            request_id=request_id_short,
            model=model,
            elapsed_s=round(time.monotonic() - started_at, 2),
        )
        return new_tool_calls, saw_first_text

    if etype == "turn.completed":
        usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}
        _merge_usage_from_event(event, usage)
        log.info(
            "proxy.turn_completed",
            request_id=request_id_short,
            model=model,
            elapsed_s=round(time.monotonic() - started_at, 2),
            in_tokens=usage.get("input_tokens", 0),
            out_tokens=usage.get("output_tokens", 0),
        )
        return new_tool_calls, saw_first_text

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

    elif etype in ("item.started", "item.completed", "response_item"):
        item = event.get("item") if etype != "response_item" else event.get("payload")
        if isinstance(item, dict):
            item_type = item.get("type")
            if item_type in (
                "toolCall",
                "function_call",
                "mcpToolCall",
                "mcp_tool_call",
            ):
                name = (
                    item.get("name")
                    or item.get("toolName")
                    or item.get("tool")
                    or "?"
                )
                tool_id = str(item.get("id") or item.get("call_id") or "")[:12]
                server = item.get("server")
                status = item.get("status")
                if etype == "item.started":
                    log.info(
                        "proxy.tool_call",
                        request_id=request_id_short,
                        server=str(server) if server is not None else None,
                        tool=str(name),
                        tool_id=tool_id,
                        status=str(status) if status is not None else None,
                        elapsed_s=round(time.monotonic() - started_at, 2),
                    )
                    new_tool_calls = 1
                else:
                    log.info(
                        "proxy.tool_result",
                        request_id=request_id_short,
                        server=str(server) if server is not None else None,
                        tool=str(name),
                        tool_id=tool_id,
                        status=str(status) if status is not None else None,
                        elapsed_s=round(time.monotonic() - started_at, 2),
                    )
            elif item_type in (
                "agentMessage",
                "agent_message",
                "assistant_message",
                "message",
            ):
                text = _extract_text_field(cast(dict[str, Any], item))
                if text is not None:
                    log.info(
                        "proxy.agent_update",
                        request_id=request_id_short,
                        elapsed_s=round(time.monotonic() - started_at, 2),
                        chars=len(text),
                        preview=_preview_text(text),
                    )
                if not saw_first_text:
                    log.info(
                        "proxy.text_started",
                        request_id=request_id_short,
                        elapsed_s=round(time.monotonic() - started_at, 2),
                    )
                    saw_first_text = True

    return new_tool_calls, saw_first_text


def _preview_text(text: str, *, limit: int = 180) -> str:
    """Short, single-line preview for live proxy logs."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[: limit - 3] + "..."


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


def _parse_upstream_json(
    stdout_text: str,
    *,
    upstream: Upstream,
    exit_code: int,
    stderr: str,
) -> tuple[str, dict[str, int]]:
    if upstream == "codex":
        return _parse_codex_exec_json(stdout_text, exit_code=exit_code, stderr=stderr)
    return _parse_stream_json(stdout_text, exit_code=exit_code, stderr=stderr)


def _parse_codex_exec_json(
    stdout_text: str,
    *,
    exit_code: int,
    stderr: str,
) -> tuple[str, dict[str, int]]:
    """Walk ``codex exec --json`` output and return (text, usage).

    Codex JSONL has evolved across CLI builds, so this parser deliberately
    accepts the stable shapes we care about: completed agent-message items
    and usage objects, whether they arrive top-level or nested under
    ``item`` / ``payload`` / ``turn``.
    """
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
            continue
        saw_any_event = True
        _merge_usage_from_event(event, usage)
        text = _extract_codex_agent_text(event)
        if text is not None:
            result_text = text

    if result_text is None:
        snippet = stderr.strip()[:500] if stderr.strip() else stdout_text.strip()[-500:]
        if not saw_any_event:
            raise ProxyError(
                f"codex exec produced no JSON events (exit={exit_code}); snippet={snippet}"
            )
        raise ProxyError(
            f"codex exec stream missing agent message (exit={exit_code}); snippet={snippet}"
        )
    return result_text, usage


def _extract_codex_agent_text(event: dict[str, Any]) -> str | None:
    candidates: list[dict[str, Any]] = [event]
    for key in ("item", "payload", "message"):
        raw = event.get(key)
        if isinstance(raw, dict):
            candidates.append(cast(dict[str, Any], raw))
    turn = event.get("turn")
    if isinstance(turn, dict):
        candidates.append(cast(dict[str, Any], turn))

    for obj in candidates:
        obj_type = obj.get("type")
        if obj_type not in (
            "agentMessage",
            "agent_message",
            "assistant_message",
            "message",
            "final_message",
        ):
            continue
        text = _extract_text_field(obj)
        if text is not None:
            return text

    for obj in candidates:
        for key in ("last_agent_message", "final_response", "finalMessage", "lastMessage"):
            text = _coerce_optional_text(obj.get(key))
            if text is not None:
                return text
    return None


def _extract_text_field(obj: dict[str, Any]) -> str | None:
    for key in ("text", "content", "message", "output_text"):
        text = _coerce_optional_text(obj.get(key))
        if text is not None:
            return text
    return None


def _coerce_optional_text(raw: Any) -> str | None:
    if isinstance(raw, str):
        stripped = raw.strip()
        return stripped if stripped else None
    if isinstance(raw, list):
        text = _coerce_content(raw).strip()
        return text if text else None
    if isinstance(raw, dict):
        return _extract_text_field(cast(dict[str, Any], raw))
    return None


def _merge_usage_from_event(event: dict[str, Any], usage: dict[str, int]) -> None:
    for raw in (
        event.get("usage"),
        event.get("tokenUsage"),
        event.get("token_usage"),
    ):
        _merge_usage(raw, usage)

    for key in ("item", "payload", "turn"):
        nested = event.get(key)
        if not isinstance(nested, dict):
            continue
        nested_dict = cast(dict[str, Any], nested)
        for raw in (
            nested_dict.get("usage"),
            nested_dict.get("tokenUsage"),
            nested_dict.get("token_usage"),
        ):
            _merge_usage(raw, usage)


def _merge_usage(raw: Any, usage: dict[str, int]) -> None:
    if not isinstance(raw, dict):
        return
    raw_dict = cast(dict[str, Any], raw)
    nested_last = raw_dict.get("last")
    if isinstance(nested_last, dict):
        _merge_usage(nested_last, usage)

    input_tokens = _first_int(
        raw_dict,
        ("input_tokens", "prompt_tokens", "inputTokens", "promptTokens"),
    )
    output_tokens = _first_int(
        raw_dict,
        ("output_tokens", "completion_tokens", "outputTokens", "completionTokens"),
    )
    if input_tokens is not None:
        usage["input_tokens"] = input_tokens
    if output_tokens is not None:
        usage["output_tokens"] = output_tokens


def _first_int(raw: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    for key in keys:
        value = raw.get(key)
        if isinstance(value, int):
            return value
    return None


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
                {"status": f"{_upstream_executable(self.state.upstream)} CLI unavailable"},
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
                    "upstream": r.upstream,
                    "model": r.model,
                    "elapsed_s": round(now_mono - r.started_at, 1),
                    "last_byte_age_s": round(now_mono - r.last_byte_at, 1),
                }
                for r in self.state.requests.values()
            ]
        self._send_json(
            HTTPStatus.OK,
            {
                "in_flight": in_flight,
                "capacity": self.state.max_concurrent,
                "upstream": self.state.upstream,
                "total_requests": total,
                "last_error": self.state.last_error,
                "requests": request_view,
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
                text, usage = _invoke_upstream(
                    upstream=self.state.upstream,
                    model=model,
                    prompt=prompt,
                    request_id=request_id,
                    log_dir=self.state.log_dir,
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
    upstream: Upstream | None = None,
    max_concurrent: int | None = None,
    log_dir: Path | None = None,
) -> tuple[ThreadingHTTPServer, _State]:
    """Wire up the handler class with shared state and return (server, state)."""
    cap = max_concurrent if max_concurrent is not None else _env_max_concurrent()
    resolved_upstream = upstream if upstream is not None else _env_upstream()
    resolved_log_dir = log_dir if log_dir is not None else _env_log_dir()

    resolved_log_dir.mkdir(parents=True, exist_ok=True)
    _prune_old_logs(resolved_log_dir)

    state = _State(
        max_concurrent=cap,
        log_dir=resolved_log_dir,
        upstream=resolved_upstream,
    )

    handler_cls = type(
        "BoundProxyHandler",
        (ProxyHandler,),
        {"state": state},
    )
    server = ThreadingHTTPServer((host, port), cast(Any, handler_cls))
    return server, state


def _env_max_concurrent() -> int:
    raw = os.environ.get("TASQUE_PROXY_MAX_CONCURRENT")
    if raw is None:
        return DEFAULT_MAX_CONCURRENT
    try:
        value = int(raw)
    except ValueError as exc:
        raise ProxyError(
            f"TASQUE_PROXY_MAX_CONCURRENT must be an integer, got {raw!r}"
        ) from exc
    if value < 1:
        raise ProxyError(f"TASQUE_PROXY_MAX_CONCURRENT must be >= 1, got {value}")
    return value


def _env_log_dir() -> Path:
    raw = os.environ.get("TASQUE_PROXY_LOG_DIR")
    return Path(raw) if raw else Path("data") / "proxy-logs"


def serve(
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    upstream: Upstream | None = None,
    max_concurrent: int | None = None,
    log_dir: Path | None = None,
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
        upstream=upstream,
        max_concurrent=max_concurrent,
        log_dir=log_dir,
    )
    log.info(
        "proxy.start",
        host=host,
        port=port,
        upstream=state.upstream,
        max_concurrent=state.max_concurrent,
        log_dir=str(state.log_dir),
        upstream_cwd=str(_upstream_cwd(state.upstream)),
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
