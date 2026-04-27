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
    concurrency test can still observe queue back-pressure."""

    def __init__(self, payload: str, delay: float) -> None:
        self._lines = [
            (line + "\n").encode("utf-8")
            for line in payload.splitlines()
        ]
        self._delay = delay
        self._slept = False

    def __iter__(self) -> _FakeStream:
        return self

    def __next__(self) -> bytes:
        if not self._slept and self._delay:
            time.sleep(self._delay)
            self._slept = True
        if not self._lines:
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
    """

    def __init__(
        self,
        *,
        stdout: str,
        stderr: str = "",
        returncode: int = 0,
        delay: float = 0.0,
    ) -> None:
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False
        self.stdout = _FakeStream(stdout, delay)
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


def _make_popen_factory(
    *,
    stdout: str,
    stderr: str = "",
    returncode: int = 0,
    delay: float = 0.0,
) -> Callable[..., FakePopen]:
    def factory(*_args: Any, **_kwargs: Any) -> FakePopen:
        return FakePopen(stdout=stdout, stderr=stderr, returncode=returncode, delay=delay)

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
