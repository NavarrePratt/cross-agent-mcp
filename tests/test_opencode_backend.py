"""Unit tests for the OpenCode backend (mocked transport, no real server)."""

from __future__ import annotations

import httpx
import pytest

from cross_agent.backends.opencode import (
    OpenCodeBackend,
    OpenCodeConfig,
    _parse_model,
    _redact,
    _sanitize_directory,
)


def _msg_handler(text: str = "PONG", status: int = 200):
    """Handler that serves a single message response (ignores the agent field)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session" and request.method == "POST":
            return httpx.Response(200, json={"id": "ses_test_123"})
        if request.url.path.startswith("/session/") and request.url.path.endswith("/message"):
            return httpx.Response(
                status,
                json={
                    "info": {"agent": "build"},
                    "parts": [
                        {"type": "step-start"},
                        {"type": "text", "text": text},
                        {"type": "step-finish"},
                    ],
                },
            )
        # DELETE /session/{id} and POST /abort -> 200
        return httpx.Response(200, json=True)

    return handler


# --------------------------------------------------------------------- helpers


def test_sanitize_directory_strips_crlf_nul():
    assert _sanitize_directory("/safe/path") == "/safe/path"
    assert _sanitize_directory(None) is None
    assert _sanitize_directory("ev\r\0il\npath") == "evilpath"


def test_parse_model_splits_provider_and_id():
    assert _parse_model(None) is None
    assert _parse_model("wandb/zai-org/GLM-5.2") == {
        "providerID": "wandb",
        "modelID": "zai-org/GLM-5.2",
    }
    assert _parse_model("noprovider") == {"providerID": "noprovider", "modelID": "noprovider"}


def test_redact_scrubs_known_secret_prefixes():
    assert "[REDACTED]" in _redact("key=sk-" + "a" * 40)
    assert "[REDACTED]" in _redact("token ghp_" + "b" * 36)
    assert "[REDACTED]" in _redact("hf_" + "c" * 30)
    assert "[REDACTED]" in _redact("xoxb-" + "d" * 20)
    assert _redact("plain text with no secrets") == "plain text with no secrets"


# ------------------------------------------------------------------- prompt


async def test_prompt_creates_messages_deletes_session(make_backend, captured_requests):
    backend = make_backend(_msg_handler("hello"))
    text, sid = await backend.prompt("say hello")
    assert text == "hello"
    assert sid == "ses_test_123"

    methods = [(r.method, r.url.path) for r in captured_requests]
    assert ("POST", "/session") in methods
    assert ("POST", "/session/ses_test_123/message") in methods
    # stateless one-shot deletes the session after the response
    assert ("DELETE", "/session/ses_test_123") in methods


async def test_prompt_read_only_default_omits_agent(make_backend, captured_requests):
    """With the default empty agent, read_only=True does not send an agent field."""
    backend = make_backend(_msg_handler("ok"))
    await backend.prompt("review this", read_only=True)

    msg_req = next(
        r for r in captured_requests if r.url.path.endswith("/message") and r.method == "POST"
    )
    import json

    body = json.loads(msg_req.content)
    assert "agent" not in body


async def test_prompt_read_only_with_explicit_agent_sends_it(make_backend, captured_requests):
    """When config.agent is set, read_only=True sends that agent."""
    from cross_agent.backends.opencode import OpenCodeConfig

    backend = make_backend(
        _msg_handler("ok"),
        config=OpenCodeConfig(agent="my-reviewer"),
    )
    await backend.prompt("review this", read_only=True)

    msg_req = next(
        r for r in captured_requests if r.url.path.endswith("/message") and r.method == "POST"
    )
    import json

    body = json.loads(msg_req.content)
    assert body["agent"] == "my-reviewer"


async def test_prompt_read_only_false_omits_agent(make_backend, captured_requests):
    backend = make_backend(_msg_handler("ok"))
    await backend.prompt("do work", read_only=False)

    msg_req = next(
        r for r in captured_requests if r.url.path.endswith("/message") and r.method == "POST"
    )
    import json

    body = json.loads(msg_req.content)
    assert "agent" not in body


async def test_prompt_passes_model_object(make_backend, captured_requests):
    backend = make_backend(_msg_handler("ok"))
    await backend.prompt("hi", model="wandb/zai-org/GLM-5.2")

    msg_req = next(
        r for r in captured_requests if r.url.path.endswith("/message") and r.method == "POST"
    )
    import json

    body = json.loads(msg_req.content)
    assert body["model"] == {"providerID": "wandb", "modelID": "zai-org/GLM-5.2"}


async def test_prompt_directory_header_strips_crlf(make_backend, captured_requests):
    backend = make_backend(_msg_handler("ok"))
    await backend.prompt("hi", directory="evil\r\npath")

    # No request may carry a header with CRLF/NUL; any request that carries the
    # directory header at all must carry the sanitized value.
    saw_sanitized = False
    for r in captured_requests:
        value = r.headers.get("x-opencode-directory")
        if value is not None:
            assert "\r" not in value and "\n" not in value and "\0" not in value
            assert value == "evilpath"
            saw_sanitized = True
    assert saw_sanitized, "no request carried the directory header"


# ---------------------------------------------------------- read-only agent


async def test_prompt_no_warning_when_agent_empty(make_backend):
    """With the default empty agent, the response has no enforcement warning."""
    backend = make_backend(_msg_handler("clean response"))
    text, _ = await backend.prompt("hi", read_only=True)
    assert "read-only enforcement" not in text
    assert text == "clean response"


# ------------------------------------------------------------------ continue


async def test_continue_session_does_not_delete(make_backend, captured_requests):
    backend = make_backend(_msg_handler("more"))
    out = await backend.continue_session("ses_existing", "again")
    assert out == "more"

    methods = [(r.method, r.url.path) for r in captured_requests]
    assert ("POST", "/session/ses_existing/message") in methods
    # continue must NOT delete the session (it stays alive for further turns)
    delete = [m for m in methods if m[0] == "DELETE"]
    assert not delete


async def test_continue_session_passes_resume_session_id(make_backend, captured_requests):
    backend = make_backend(_msg_handler("ack"))
    await backend.continue_session("ses_xyz", "ack")
    # Only a message POST is made against the resumed session id (no new session).
    assert all(r.url.path != "/session" for r in captured_requests if r.method == "POST")


# --------------------------------------------------------------------- abort


async def test_abort_posts_abort_endpoint(make_backend, captured_requests):
    backend = make_backend(_msg_handler())
    await backend.abort("ses_run_1")
    assert ("POST", "/session/ses_run_1/abort") in [
        (r.method, r.url.path) for r in captured_requests
    ]


async def test_abort_treats_404_as_success(make_backend):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/abort"):
            return httpx.Response(404, json={"error": "not found"})
        return httpx.Response(200, json=True)

    backend = make_backend(handler)
    # Should not raise.
    await backend.abort("ses_gone")


# --------------------------------------------------------------- secret scrub


async def test_response_secrets_are_redacted(make_backend):
    backend = make_backend(_msg_handler("here is my key sk-" + "a" * 40))
    text, _ = await backend.prompt("show key")
    assert "sk-" not in text
    assert "[REDACTED]" in text


# ---------------------------------------------------------------- auto-start


async def test_autoserve_spawns_opencode_when_down(monkeypatch):
    """When the server is down and autoserve is on, opencode serve is spawned."""
    spawned: list[list[str]] = []

    class FakeProc:
        def __init__(self) -> None:
            self.returncode = None

        def poll(self) -> None:
            return None

        def terminate(self) -> None: ...
        def wait(self, timeout=None) -> None: ...
        def kill(self) -> None: ...

    def fake_popen(cmd, **kwargs):
        spawned.append(cmd)
        return FakeProc()

    monkeypatch.setattr("cross_agent.backends.opencode.subprocess.Popen", fake_popen)

    # Health probe fails forever, but _wait_healthy short-circuits to True.
    backend = OpenCodeBackend(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=True)
    )
    backend._is_healthy = monkeypatch_async(False)  # type: ignore[method-assign]
    backend._wait_healthy = monkeypatch_async(True)  # type: ignore[method-assign]

    await backend._ensure_server()
    assert backend._spawned is True
    assert spawned and spawned[0][:2] == ["opencode", "serve"]
    assert "--port" in spawned[0]


async def test_autoserve_disabled_raises_when_down(monkeypatch):
    backend = OpenCodeBackend(
        OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False)
    )
    backend._is_healthy = monkeypatch_async(False)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="not reachable"):
        await backend._ensure_server()


def monkeypatch_async(value):
    """Return an async function that resolves to ``value`` (for stubbing backend methods)."""

    async def _stub(*args, **kwargs):
        return value

    return _stub
