"""Unit tests for the host-side LLM proxy.

`subprocess.Popen` and `shutil.which` are monkeypatched so the tests never
actually spawn `claude --print`. The server itself runs on a real OS-assigned
port in a background thread and is exercised via stdlib `urllib.request`.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from tasque.proxy import server as proxy_server

FIXTURE_STREAM = Path(__file__).parent / "fixtures" / "sample_stream.jsonl"


class _FakeStream:
    """Iterable byte-line reader satisfying the subset of ``BufferedReader``
    the proxy uses (line iteration). Sleeps once on first read so the
    concurrency test can still observe queue back-pressure.

    Watchdog tests use ``kill_event``: when set, the next ``__next__``
    breaks out of any pending wait and raises ``StopIteration`` to mirror
    real ``Popen.kill()`` closing stdout."""

    def __init__(
        self,
        payload: str,
        delay: float,
        *,
        kill_event: threading.Event | None = None,
        block_until_killed: bool = False,
    ) -> None:
        self._lines = [
            (line + "\n").encode("utf-8")
            for line in payload.splitlines()
        ]
        self._delay = delay
        self._slept = False
        self._kill_event = kill_event
        self._block_until_killed = block_until_killed

    def __iter__(self) -> _FakeStream:
        return self

    def __next__(self) -> bytes:
        if not self._slept and self._delay:
            time.sleep(self._delay)
            self._slept = True
        if not self._lines:
            if self._block_until_killed and self._kill_event is not None:
                # Stay silent until the watchdog (or test) signals kill.
                self._kill_event.wait()
                raise StopIteration
            raise StopIteration
        return self._lines.pop(0)


class _FakeStdin:
    def write(self, _data: bytes) -> int:
        return 0

    def close(self) -> None:
        pass


class FakePopen:
    """Stand-in for `subprocess.Popen` returning canned stream-json output.

    The proxy now streams ``proc.stdout`` line-by-line and then calls
    ``communicate`` only to drain stderr — so the fake exposes a
    line-iterable ``stdout`` and a ``communicate`` that returns no
    additional stdout.

    With ``block_until_killed=True``, the stdout stream stays silent
    after exhausting any canned lines until ``kill()`` is called or the
    external ``kill_event`` is set — used to test the stall watchdog
    and the cancel endpoint without timing on real subprocess behavior.
    """

    def __init__(
        self,
        *,
        stdout: str,
        stderr: str = "",
        returncode: int = 0,
        delay: float = 0.0,
        block_until_killed: bool = False,
    ) -> None:
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False
        self._kill_event = threading.Event()
        self.stdout = _FakeStream(
            stdout,
            delay,
            kill_event=self._kill_event,
            block_until_killed=block_until_killed,
        )
        self.stderr = None  # proxy reads stderr via communicate, not directly
        self.stdin = _FakeStdin()

    def communicate(
        self,
        input: bytes | None = None,
        timeout: float | None = None,
    ) -> tuple[bytes, bytes]:
        return b"", self._stderr.encode("utf-8")

    def kill(self) -> None:
        self.killed = True
        self._kill_event.set()

    def poll(self) -> int | None:
        # Mirror real Popen: ``None`` while running, exit code once
        # terminated. The fake "completes" once stdout is exhausted.
        if self.killed:
            return self.returncode
        return None


def _make_popen_factory(
    *,
    stdout: str,
    stderr: str = "",
    returncode: int = 0,
    delay: float = 0.0,
    block_until_killed: bool = False,
    sink: list[FakePopen] | None = None,
) -> Callable[..., FakePopen]:
    def factory(*_args: Any, **_kwargs: Any) -> FakePopen:
        fake = FakePopen(
            stdout=stdout,
            stderr=stderr,
            returncode=returncode,
            delay=delay,
            block_until_killed=block_until_killed,
        )
        if sink is not None:
            sink.append(fake)
        return fake

    return factory


@pytest.fixture
def fake_claude_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make `shutil.which("claude")` resolve to a fake absolute path."""
    monkeypatch.setattr(
        proxy_server.shutil,
        "which",
        lambda name: "/fake/bin/claude" if name == "claude" else None,
    )


@pytest.fixture
def running_proxy(
    tmp_path: Path,
) -> Iterator[Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]]]:
    """Spawn the proxy in a background thread; tear it down on exit.

    Yields a factory that creates one server per call so tests can use
    different concurrency caps. All servers spawned through the factory are
    cleaned up when the test finishes.
    """
    spawned: list[tuple[proxy_server.ThreadingHTTPServer, threading.Thread]] = []

    def factory(max_concurrent: int) -> tuple[proxy_server.ThreadingHTTPServer, int]:
        srv, _state = proxy_server.make_server(
            host="127.0.0.1",
            port=0,
            max_concurrent=max_concurrent,
            log_dir=tmp_path / "logs",
            timeout=None,
        )
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        spawned.append((srv, thread))
        port = srv.server_address[1]
        return srv, port

    try:
        yield factory
    finally:
        for srv, thread in spawned:
            srv.shutdown()
            srv.server_close()
            thread.join(timeout=2)


def _post_chat(port: int, *, model: str = "claude-haiku-4-5", content: str = "hi") -> dict[str, Any]:
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {"model": model, "messages": [{"role": "user", "content": content}]}
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return dict(json.loads(resp.read().decode("utf-8")))


def _get(port: int, path: str) -> tuple[int, dict[str, Any]]:
    try:
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}{path}", timeout=5
        ) as resp:
            return resp.status, dict(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        return exc.code, dict(json.loads(exc.read().decode("utf-8")))


# ---------------------------------------------------------------------------
# pure-function tests
# ---------------------------------------------------------------------------


def test_flatten_messages_orders_system_then_turns() -> None:
    out = proxy_server._flatten_messages(
        [
            {"role": "system", "content": "you are terse"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "yo"},
            {"role": "user", "content": "bye"},
        ]
    )
    assert out.startswith("you are terse")
    assert "User: hi" in out
    assert "Assistant: yo" in out
    assert out.endswith("User: bye")


def test_flatten_messages_handles_list_content() -> None:
    out = proxy_server._flatten_messages(
        [{"role": "user", "content": [{"type": "text", "text": "alpha"}, {"type": "text", "text": "beta"}]}]
    )
    assert "alpha" in out and "beta" in out


def test_parse_stream_json_extracts_result_and_usage() -> None:
    text, usage = proxy_server._parse_stream_json(
        FIXTURE_STREAM.read_text(encoding="utf-8"),
        exit_code=0,
        stderr="",
    )
    assert text == "hello"
    assert usage == {"input_tokens": 10, "output_tokens": 3}


def test_parse_stream_json_raises_when_no_result_event() -> None:
    transcript = '{"type":"system"}\n{"type":"assistant","message":{}}\n'
    with pytest.raises(proxy_server.ProxyError, match="missing result event"):
        proxy_server._parse_stream_json(transcript, exit_code=1, stderr="boom")


def test_parse_stream_json_raises_when_truncated_with_no_events() -> None:
    transcript = "not json at all\n"
    with pytest.raises(proxy_server.ProxyError, match="no JSON events"):
        proxy_server._parse_stream_json(transcript, exit_code=1, stderr="")


def test_build_claude_argv_omits_disallowed_when_none() -> None:
    argv = proxy_server._build_claude_argv(
        exe="/fake/bin/claude",
        model="claude-haiku-4-5",
        prompt="hi",
        use_stdin=False,
    )
    assert "--disallowedTools" not in argv


def test_build_claude_argv_appends_disallowed_tools_csv() -> None:
    argv = proxy_server._build_claude_argv(
        exe="/fake/bin/claude",
        model="claude-haiku-4-5",
        prompt="hi",
        use_stdin=False,
        disallowed_tools=["mcp__tasque__chain_fire_template", "mcp__tasque__job_create"],
    )
    flag_idx = argv.index("--disallowedTools")
    assert argv[flag_idx + 1] == (
        "mcp__tasque__chain_fire_template,mcp__tasque__job_create"
    )


def test_extract_disallowed_tools_reads_top_level_field() -> None:
    out = proxy_server._extract_disallowed_tools(
        {"disallowed_tools": ["mcp__tasque__chain_fire_template"]}
    )
    assert out == ["mcp__tasque__chain_fire_template"]


def test_extract_disallowed_tools_reads_extra_body_nested() -> None:
    out = proxy_server._extract_disallowed_tools(
        {"extra_body": {"disallowed_tools": ["mcp__tasque__job_create"]}}
    )
    assert out == ["mcp__tasque__job_create"]


def test_extract_disallowed_tools_returns_none_for_empty_or_garbage() -> None:
    assert proxy_server._extract_disallowed_tools({}) is None
    assert proxy_server._extract_disallowed_tools({"disallowed_tools": []}) is None
    assert proxy_server._extract_disallowed_tools({"disallowed_tools": "not-a-list"}) is None
    assert proxy_server._extract_disallowed_tools(
        {"disallowed_tools": ["", "  "]}
    ) is None


# ---------------------------------------------------------------------------
# integration tests (real HTTP server, mocked subprocess)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("fake_claude_path")
def test_chat_completions_returns_openai_shape(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout=FIXTURE_STREAM.read_text(encoding="utf-8")),
    )
    _, port = running_proxy(2)

    body = _post_chat(port)
    assert body["object"] == "chat.completion"
    assert body["model"] == "claude-haiku-4-5"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert body["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
    }


@pytest.mark.usefixtures("fake_claude_path")
def test_chat_completions_forwards_disallowed_tools_to_argv(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    """Per-request denylist threads the whole way: HTTP body → subprocess argv."""
    captured: dict[str, list[str]] = {}

    def factory(*args: Any, **_kwargs: Any) -> FakePopen:
        captured["argv"] = list(args[0])
        return FakePopen(stdout=FIXTURE_STREAM.read_text(encoding="utf-8"))

    monkeypatch.setattr(proxy_server.subprocess, "Popen", factory)
    _, port = running_proxy(2)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {
                "model": "claude-haiku-4-5",
                "messages": [{"role": "user", "content": "hi"}],
                "disallowed_tools": [
                    "mcp__tasque__chain_fire_template",
                    "mcp__tasque__job_create",
                ],
            }
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200

    argv = captured["argv"]
    flag_idx = argv.index("--disallowedTools")
    assert argv[flag_idx + 1] == (
        "mcp__tasque__chain_fire_template,mcp__tasque__job_create"
    )


@pytest.mark.usefixtures("fake_claude_path")
def test_concurrent_requests_queue_beyond_capacity(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    cap = 2
    delay = 0.4
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(
            stdout=FIXTURE_STREAM.read_text(encoding="utf-8"),
            delay=delay,
        ),
    )
    _, port = running_proxy(cap)

    n = cap * 2  # twice capacity → second batch must wait
    results: list[dict[str, Any]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def fire() -> None:
        try:
            body = _post_chat(port)
        except BaseException as exc:
            with lock:
                errors.append(exc)
        else:
            with lock:
                results.append(body)

    threads = [threading.Thread(target=fire) for _ in range(n)]
    started = time.monotonic()
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=15)
    elapsed = time.monotonic() - started

    assert not errors, f"unexpected errors: {errors!r}"
    assert len(results) == n
    assert all(r["choices"][0]["message"]["content"] == "hello" for r in results)
    # With cap=2 and delay=0.4s, n=4 requests need at least ~2 batches → 0.7s+.
    # If the semaphore were missing, all 4 would finish in ~0.4s.
    assert elapsed >= delay * 1.5, f"requests did not queue (elapsed={elapsed:.2f}s)"


@pytest.mark.usefixtures("fake_claude_path")
def test_malformed_stream_returns_502(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(
            stdout='{"type":"system"}\n{"type":"assistant"}\n',
            stderr="upstream rejected",
            returncode=1,
        ),
    )
    _, port = running_proxy(2)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=5)
    err = excinfo.value
    assert err.code == 502
    payload = json.loads(err.read().decode("utf-8"))
    assert "error" in payload
    assert "request_id" in payload["error"]
    assert "result event" in payload["error"]["message"]


@pytest.mark.usefixtures("fake_claude_path")
def test_healthz_ok_when_claude_version_succeeds(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    class _Result:
        returncode = 0

    monkeypatch.setattr(
        proxy_server.subprocess,
        "run",
        lambda *_a, **_kw: _Result(),
    )
    _, port = running_proxy(2)

    status, body = _get(port, "/healthz")
    assert status == 200
    assert body == {"status": "ok"}


def test_healthz_503_when_claude_missing(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    monkeypatch.setattr(proxy_server.shutil, "which", lambda _name: None)
    _, port = running_proxy(2)

    status, body = _get(port, "/healthz")
    assert status == 503
    assert body["status"].startswith("claude CLI")


@pytest.mark.usefixtures("fake_claude_path")
def test_status_endpoint_reports_counters(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout=FIXTURE_STREAM.read_text(encoding="utf-8")),
    )
    _, port = running_proxy(3)

    _post_chat(port)
    _post_chat(port)
    status, body = _get(port, "/status")
    assert status == 200
    assert body["capacity"] == 3
    assert body["total_requests"] == 2
    assert body["in_flight"] == 0


@pytest.mark.usefixtures("fake_claude_path")
def test_chat_completions_rejects_missing_messages(
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    _, port = running_proxy(2)
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps({"model": "claude-haiku-4-5"}).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=5)
    assert excinfo.value.code == 400


# ---------------------------------------------------------------------------
# stall watchdog + idle-grant + cancel endpoints
# ---------------------------------------------------------------------------


def _post_json(
    port: int,
    path: str,
    body: dict[str, Any] | None = None,
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(body or {}).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, dict(json.loads(resp.read().decode("utf-8")))
    except urllib.error.HTTPError as exc:
        return exc.code, dict(json.loads(exc.read().decode("utf-8")))


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return False


@pytest.mark.usefixtures("fake_claude_path")
def test_stall_kill_after_silence_with_no_budget(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    """Subprocess silent past the stall threshold AND no idle-silence
    budget ⇒ watchdog kills, request returns 502 with a stall message."""
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "0.5")
    fakes: list[FakePopen] = []
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout="", block_until_killed=True, sink=fakes),
    )
    _, port = running_proxy(2)

    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(req, timeout=10)
    assert excinfo.value.code == 502
    payload = json.loads(excinfo.value.read().decode("utf-8"))
    assert "stall watchdog" in payload["error"]["message"]
    # Watchdog must have actually called kill() on the fake.
    assert fakes and fakes[0].killed


@pytest.mark.usefixtures("fake_claude_path")
def test_idle_grant_suppresses_stall_kill(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    """A claim_idle_silence call extends the silence budget so the
    watchdog does NOT kill while the budget is in effect."""
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "0.5")
    fakes: list[FakePopen] = []
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout="", block_until_killed=True, sink=fakes),
    )
    _, port = running_proxy(2)

    request_body = json.dumps(
        {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
    ).encode("utf-8")
    chat_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=request_body,
        headers={"content-type": "application/json"},
        method="POST",
    )

    chat_result: dict[str, Any] = {}

    def _fire() -> None:
        try:
            with urllib.request.urlopen(chat_req, timeout=10) as resp:
                chat_result["status"] = resp.status
        except urllib.error.HTTPError as exc:
            chat_result["status"] = exc.code
            chat_result["body"] = exc.read().decode("utf-8")

    chat_thread = threading.Thread(target=_fire, daemon=True)
    chat_thread.start()

    # Wait until the request is registered, then grant a budget BEFORE
    # the stall threshold (0.5s) elapses.
    server, _ = running_proxy(2)  # reuse a registered fixture-spawned server
    del server  # silence linter; we use the port from the first running_proxy

    def _request_visible() -> bool:
        _, body = _get(port, "/status")
        return bool(body.get("requests"))

    assert _wait_until(_request_visible, timeout=2.0), "request never registered"
    _, status_body = _get(port, "/status")
    request_id = status_body["requests"][0]["request_id"]

    grant_status, grant_body = _post_json(
        port,
        f"/v1/internal/idle_grant/{request_id}",
        {"seconds": 5, "reason": "training task1"},
    )
    assert grant_status == 200
    assert grant_body["ok"] is True
    assert grant_body["reason"] == "training task1"
    assert grant_body["remaining_s"] >= 4.0

    # Wait past where stall would have fired without the budget.
    time.sleep(1.5)
    assert not (fakes and fakes[0].killed), (
        "watchdog killed despite active idle-silence budget"
    )

    # /status surfaces the budget so an operator can see the deferral.
    _, mid_status = _get(port, "/status")
    assert mid_status["requests"][0]["silence_budget_remaining_s"] is not None
    assert mid_status["requests"][0]["silence_budget_reason"] == "training task1"

    # Tear down: cancel so the chat thread doesn't hang the test.
    _post_json(port, f"/v1/cancel/{request_id}")
    chat_thread.join(timeout=5)
    assert chat_result.get("status") == 502


@pytest.mark.usefixtures("fake_claude_path")
def test_cancel_endpoint_kills_running_request(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    """``POST /v1/cancel/{request_id}`` kills the matching subprocess
    and the chat completion returns 502 with a cancelled message."""
    # High stall threshold so we know the kill path is /cancel, not stall.
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "60")
    fakes: list[FakePopen] = []
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout="", block_until_killed=True, sink=fakes),
    )
    _, port = running_proxy(2)

    chat_result: dict[str, Any] = {}
    chat_req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "hi"}]}
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )

    def _fire() -> None:
        try:
            with urllib.request.urlopen(chat_req, timeout=10) as resp:
                chat_result["status"] = resp.status
        except urllib.error.HTTPError as exc:
            chat_result["status"] = exc.code
            chat_result["body"] = exc.read().decode("utf-8")

    t = threading.Thread(target=_fire, daemon=True)
    t.start()

    def _registered() -> bool:
        _, body = _get(port, "/status")
        return bool(body.get("requests"))

    assert _wait_until(_registered, timeout=2.0)
    _, status_body = _get(port, "/status")
    request_id = status_body["requests"][0]["request_id"]

    cancel_status, cancel_body = _post_json(port, f"/v1/cancel/{request_id}")
    assert cancel_status == 200
    assert cancel_body["ok"] is True

    t.join(timeout=5)
    assert chat_result.get("status") == 502
    payload = json.loads(chat_result["body"])
    assert "cancelled" in payload["error"]["message"]


@pytest.mark.usefixtures("fake_claude_path")
def test_idle_grant_unknown_request_id_returns_404(
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    _, port = running_proxy(2)
    status, body = _post_json(
        port,
        "/v1/internal/idle_grant/does-not-exist",
        {"seconds": 30, "reason": "x"},
    )
    assert status == 404
    assert "unknown request_id" in body["error"]["message"]


@pytest.mark.usefixtures("fake_claude_path")
def test_cancel_unknown_request_id_returns_404(
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    _, port = running_proxy(2)
    status, body = _post_json(port, "/v1/cancel/does-not-exist")
    assert status == 404
    assert "unknown request_id" in body["error"]["message"]


@pytest.mark.usefixtures("fake_claude_path")
def test_idle_grant_rejects_non_positive_seconds(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "60")
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout="", block_until_killed=True),
    )
    _, port = running_proxy(2)

    chat_thread = threading.Thread(
        target=lambda: _post_json(
            port,
            "/v1/chat/completions",
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "x"}]},
        ),
        daemon=True,
    )
    chat_thread.start()

    def _registered() -> bool:
        _, body = _get(port, "/status")
        return bool(body.get("requests"))

    assert _wait_until(_registered, timeout=2.0)
    _, status_body = _get(port, "/status")
    request_id = status_body["requests"][0]["request_id"]

    bad, body = _post_json(
        port,
        f"/v1/internal/idle_grant/{request_id}",
        {"seconds": -5, "reason": "x"},
    )
    assert bad == 400
    assert "seconds" in body["error"]["message"]
    _post_json(port, f"/v1/cancel/{request_id}")
    chat_thread.join(timeout=5)


def test_resolve_stall_seconds_respects_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "12.5")
    assert proxy_server._resolve_stall_seconds() == 12.5


def test_resolve_stall_seconds_falls_back_on_garbage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "not-a-number")
    assert proxy_server._resolve_stall_seconds() == proxy_server.DEFAULT_STALL_SECONDS


def test_resolve_stall_seconds_default_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TASQUE_PROXY_STALL_SECONDS", raising=False)
    assert proxy_server._resolve_stall_seconds() == proxy_server.DEFAULT_STALL_SECONDS


@pytest.mark.usefixtures("fake_claude_path")
def test_status_endpoint_includes_in_flight_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
    running_proxy: Callable[[int], tuple[proxy_server.ThreadingHTTPServer, int]],
) -> None:
    """In-flight requests appear in /status with timing + budget fields."""
    monkeypatch.setenv("TASQUE_PROXY_STALL_SECONDS", "30")
    monkeypatch.setattr(
        proxy_server.subprocess,
        "Popen",
        _make_popen_factory(stdout="", block_until_killed=True),
    )
    _, port = running_proxy(2)

    chat_thread = threading.Thread(
        target=lambda: _post_json(
            port,
            "/v1/chat/completions",
            {"model": "claude-haiku-4-5", "messages": [{"role": "user", "content": "x"}]},
        ),
        daemon=True,
    )
    chat_thread.start()

    def _registered() -> bool:
        _, body = _get(port, "/status")
        return bool(body.get("requests"))

    assert _wait_until(_registered, timeout=2.0)
    _, body = _get(port, "/status")
    assert body["in_flight"] == 1
    assert len(body["requests"]) == 1
    rec = body["requests"][0]
    assert {"request_id", "model", "elapsed_s", "last_byte_age_s"} <= set(rec.keys())
    assert rec["model"] == "claude-haiku-4-5"
    assert rec["silence_budget_remaining_s"] is None  # not granted yet
    _post_json(port, f"/v1/cancel/{rec['request_id']}")
    chat_thread.join(timeout=5)
