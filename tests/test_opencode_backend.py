"""Unit tests for the OpenCode backend (mocked transport, no real server)."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from cross_agent.backends.opencode import (
    HEALTH_TIMEOUT,
    OpenCodeBackend,
    OpenCodeConfig,
    PendingPermission,
    _matching_permission,
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
        # POST /abort -> 200
        return httpx.Response(200, json=True)

    return handler


def _msg_error_handler():
    """Serve the semantic error shape returned by opencode with HTTP 200."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/session" and request.method == "POST":
            return httpx.Response(200, json={"id": "ses_test_123"})
        if request.url.path.startswith("/session/") and request.url.path.endswith("/message"):
            return httpx.Response(
                200,
                json={
                    "info": {
                        "agent": "build",
                        "error": {
                            "name": "APIError",
                            "data": {
                                "message": "Missing bearer authentication in header",
                                "statusCode": 401,
                                "responseBody": "secret=sk-" + "x" * 40,
                            },
                        },
                    },
                    "parts": [],
                },
            )
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


def test_matching_permission_filters_session_and_malformed_records():
    payload = [
        {"sessionID": "ses_other", "permission": "bash", "patterns": ["git status"]},
        {"sessionID": "ses_target", "permission": "read", "patterns": "not-a-list"},
        {"sessionID": "ses_target", "permission": "external_directory", "patterns": ["/tmp"]},
    ]

    assert _matching_permission(payload, "ses_target") == PendingPermission(
        permission="external_directory",
        pattern_count=1,
    )
    assert _matching_permission(payload, "ses_missing") is None
    assert _matching_permission({"sessionID": "ses_target"}, "ses_target") is None


def test_matching_permission_bounds_name_and_counts_patterns_without_retaining_values():
    secret = "https://example.test/?token=top-secret-value"
    payload = [
        {
            "sessionID": "ses_target",
            "permission": "external\ndirectory" + "x" * 250,
            "patterns": [secret, "two", "three", "four", "five", "six"],
        }
    ]

    pending = _matching_permission(payload, "ses_target")
    assert pending is not None
    assert pending.permission.startswith("externaldirectory")
    assert pending.permission.endswith("...")
    assert len(pending.permission) == 203
    assert pending.pattern_count == 6
    assert secret not in repr(pending)


def test_config_from_env_reads_query_timeout(monkeypatch):
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "900")
    cfg = OpenCodeConfig.from_env()
    assert cfg.query_timeout == 900.0


def test_config_from_env_query_timeout_falls_back_on_invalid(monkeypatch):
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "not-a-number")
    cfg = OpenCodeConfig.from_env()
    assert cfg.query_timeout == 1200.0


def test_config_from_env_query_timeout_ignores_nonpositive(monkeypatch):
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "0")
    cfg = OpenCodeConfig.from_env()
    assert cfg.query_timeout == 1200.0


def test_config_from_env_query_timeout_rejects_inf(monkeypatch):
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "inf")
    cfg = OpenCodeConfig.from_env()
    assert cfg.query_timeout == 1200.0


# ------------------------------------------------------------------- prompt


async def test_prompt_creates_message_and_leaves_native_session(make_backend, captured_requests):
    backend = make_backend(_msg_handler("hello"))
    text, sid = await backend.prompt("say hello")
    assert text == "hello"
    assert sid == "ses_test_123"

    methods = [(r.method, r.url.path) for r in captured_requests]
    assert ("POST", "/session") in methods
    assert ("POST", "/session/ses_test_123/message") in methods
    # prompt returns a durable native session id; lifecycle stays with opencode.
    assert ("DELETE", "/session/ses_test_123") not in methods
    assert "ses_test_123" not in backend._tasks
    assert "ses_test_123" not in backend._session_locks
    assert "ses_test_123" not in backend._session_lock_users


async def test_prompt_failure_preserves_native_session_id(make_backend, captured_requests):
    backend = make_backend(_msg_handler("boom", status=500))
    with pytest.raises(RuntimeError, match="opencode session ses_test_123"):
        await backend.prompt("say hello")

    methods = [(r.method, r.url.path) for r in captured_requests]
    assert ("POST", "/session") in methods
    assert ("POST", "/session/ses_test_123/message") in methods
    assert ("DELETE", "/session/ses_test_123") not in methods
    assert "ses_test_123" not in backend._tasks
    assert "ses_test_123" not in backend._session_locks
    assert "ses_test_123" not in backend._session_lock_users


async def test_prompt_surfaces_embedded_response_error(make_backend):
    backend = make_backend(_msg_error_handler())

    with pytest.raises(RuntimeError) as exc_info:
        await backend.prompt("say hello")

    message = str(exc_info.value)
    assert "opencode session ses_test_123" in message
    assert "APIError (HTTP 401): Missing bearer authentication in header" in message
    assert "secret=" not in message


async def test_prompt_cancellation_aborts_native_session(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    message_started = asyncio.Event()
    abort_calls = 0

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal abort_calls
        if path == "/session":
            return httpx.Response(
                200,
                json={"id": "ses_cancel"},
                request=httpx.Request(method, "http://test/session"),
            )
        if path.endswith("/message"):
            message_started.set()
            await asyncio.Event().wait()
        if path.endswith("/abort"):
            abort_calls += 1
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    task = asyncio.create_task(backend.prompt("hello"))
    await message_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert abort_calls == 1
    assert "ses_cancel" not in backend._tasks
    assert "ses_cancel" not in backend._session_locks
    assert "ses_cancel" not in backend._session_lock_users


async def test_prompt_per_call_timeout_reaches_message_request(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    message_timeout = None

    async def fake_request(
        method,
        path,
        *,
        json=None,
        headers=None,
        timeout=httpx.USE_CLIENT_DEFAULT,
    ):
        nonlocal message_timeout
        if path == "/session":
            return httpx.Response(
                200,
                json={"id": "ses_prompt_timeout"},
                request=httpx.Request(method, "http://test/session"),
            )
        if path.endswith("/message"):
            message_timeout = timeout
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "ok"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.prompt("hello", timeout=22.0) == ("ok", "ses_prompt_timeout")
    assert isinstance(message_timeout, httpx.Timeout)
    assert message_timeout.read == 22.0
    assert message_timeout.connect == 5.0


async def test_prompt_message_timeout_wraps_error_and_cleans_tracking(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    message_timeouts = []
    abort_timeouts = []

    async def fake_request(
        method,
        path,
        *,
        json=None,
        headers=None,
        timeout=httpx.USE_CLIENT_DEFAULT,
    ):
        if path == "/session":
            return httpx.Response(
                200,
                json={"id": "ses_prompt_slow"},
                request=httpx.Request(method, "http://test/session"),
            )
        if path.endswith("/message"):
            message_timeouts.append(timeout)
            raise httpx.ReadTimeout("slow response")
        if path.endswith("/abort"):
            abort_timeouts.append(timeout)
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    with pytest.raises(
        RuntimeError,
        match="opencode session ses_prompt_slow was created but message failed: ReadTimeout",
    ):
        await backend.prompt("hello", timeout=33.0)
    assert len(message_timeouts) == 1
    assert isinstance(message_timeouts[0], httpx.Timeout)
    assert message_timeouts[0].read == 33.0
    assert len(abort_timeouts) == 1
    assert "ses_prompt_slow" not in backend._tasks
    assert "ses_prompt_slow" not in backend._session_locks
    assert "ses_prompt_slow" not in backend._session_lock_users


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


# ----------------------------------------------------- permission monitoring


async def test_matching_pending_permission_aborts_and_reports_session(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    message_started = asyncio.Event()
    message_cancelled = asyncio.Event()
    abort_calls = 0
    drain_calls = []
    secret = "https://example.test/?token=top-secret-value"

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal abort_calls
        if path == "/session":
            return httpx.Response(
                200,
                json={"id": "ses_permission"},
                request=httpx.Request(method, "http://test/session"),
            )
        if method == "POST" and path.endswith("/message"):
            message_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                message_cancelled.set()
                raise
        if method == "GET" and path == "/permission":
            await message_started.wait()
            assert headers == {"x-opencode-directory": "/tmp/work"}
            return httpx.Response(
                200,
                json=[
                    {
                        "sessionID": "ses_other",
                        "permission": "bash",
                        "patterns": ["git status"],
                    },
                    {
                        "sessionID": "ses_permission",
                        "permission": "external_directory",
                        "patterns": [f"/tmp/{secret}"],
                    },
                ],
                request=httpx.Request(method, "http://test/permission"),
            )
        if path.endswith("/abort"):
            abort_calls += 1
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        raise AssertionError(f"unexpected request: {method} {path}")

    backend._request = fake_request  # type: ignore[method-assign]
    original_drain = backend._drain_cancelled_tasks

    async def recording_drain(tasks):
        drain_calls.append(list(tasks))
        await original_drain(tasks)

    backend._drain_cancelled_tasks = recording_drain  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as exc_info:
        await backend.prompt("read it", directory="/tmp/work")

    error = str(exc_info.value)
    assert "opencode session ses_permission" in error
    assert "external_directory" in error
    assert "/tmp/work" in error
    assert "for 1 pattern" in error
    assert secret not in error
    assert "native session was aborted" in error
    assert abort_calls == 1
    assert len(drain_calls) == 1
    assert len(drain_calls[0]) == 2
    assert message_cancelled.is_set()
    assert "ses_permission" not in backend._tasks
    assert "ses_permission" not in backend._session_locks
    assert "ses_permission" not in backend._session_lock_users


@pytest.mark.parametrize(
    ("abort_status", "expected_error"),
    [
        (500, "native abort could not be confirmed"),
        (404, "native session was not found; abort was not confirmed"),
    ],
)
async def test_matching_pending_permission_reports_unconfirmed_native_abort(
    monkeypatch,
    abort_status,
    expected_error,
):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    message_started = asyncio.Event()
    message_cancelled = asyncio.Event()

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if method == "POST" and path.endswith("/message"):
            message_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                message_cancelled.set()
                raise
        if path.endswith("/abort"):
            return httpx.Response(
                abort_status,
                request=httpx.Request(method, f"http://test{path}"),
            )
        raise AssertionError(f"unexpected request: {method} {path}")

    async def matching_permission(session_id, *, directory):
        await message_started.wait()
        return PendingPermission(permission="question", pattern_count=2)

    backend._request = fake_request  # type: ignore[method-assign]
    backend._watch_pending_permission = matching_permission  # type: ignore[method-assign]

    with pytest.raises(RuntimeError) as exc_info:
        await backend.continue_session("ses_abort_failed", "hello")

    error = str(exc_info.value)
    assert "opencode session ses_abort_failed" in error
    assert "for 2 patterns" in error
    assert expected_error in error
    assert "the native session was aborted" not in error
    assert message_cancelled.is_set()


async def test_unrelated_permission_does_not_interrupt_message(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    permission_seen = asyncio.Event()

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if method == "GET" and path == "/permission":
            permission_seen.set()
            return httpx.Response(
                200,
                json=[
                    {
                        "sessionID": "ses_other",
                        "permission": "external_directory",
                        "patterns": ["/tmp"],
                    }
                ],
                request=httpx.Request(method, "http://test/permission"),
            )
        if method == "POST" and path.endswith("/message"):
            await permission_seen.wait()
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "ok"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        if path.endswith("/abort"):
            raise AssertionError("unrelated permission must not abort the session")
        raise AssertionError(f"unexpected request: {method} {path}")

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_target", "hello") == "ok"


@pytest.mark.parametrize("status_code", [404, 405])
async def test_unsupported_permission_endpoint_preserves_message_flow(monkeypatch, status_code):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    permission_checked = asyncio.Event()

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if method == "GET" and path == "/permission":
            permission_checked.set()
            return httpx.Response(
                status_code,
                request=httpx.Request(method, "http://test/permission"),
            )
        if method == "POST" and path.endswith("/message"):
            await permission_checked.wait()
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "fallback"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        raise AssertionError(f"unexpected request: {method} {path}")

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_old", "hello") == "fallback"


async def test_permission_watcher_retries_transient_failure(monkeypatch):
    monkeypatch.setattr("cross_agent.backends.opencode.PERMISSION_POLL_INTERVAL", 0)
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    message_started = asyncio.Event()
    permission_calls = 0
    abort_calls = 0

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal permission_calls, abort_calls
        if method == "POST" and path.endswith("/message"):
            message_started.set()
            await asyncio.Event().wait()
        if method == "GET" and path == "/permission":
            await message_started.wait()
            permission_calls += 1
            if permission_calls == 1:
                return httpx.Response(
                    503,
                    request=httpx.Request(method, "http://test/permission"),
                )
            return httpx.Response(
                200,
                json=[
                    {
                        "sessionID": "ses_retry",
                        "permission": "question",
                        "patterns": ["confirm"],
                    }
                ],
                request=httpx.Request(method, "http://test/permission"),
            )
        if path.endswith("/abort"):
            abort_calls += 1
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        raise AssertionError(f"unexpected request: {method} {path}")

    backend._request = fake_request  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="unattended permission 'question'"):
        await backend.continue_session("ses_retry", "hello", read_only=False)
    assert permission_calls == 2
    assert abort_calls == 1


async def test_successful_message_cancels_permission_watcher(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    permission_started = asyncio.Event()
    permission_cancelled = asyncio.Event()

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if method == "GET" and path == "/permission":
            permission_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                permission_cancelled.set()
                raise
        if method == "POST" and path.endswith("/message"):
            await permission_started.wait()
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "done"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        raise AssertionError(f"unexpected request: {method} {path}")

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_done", "hello") == "done"
    assert permission_cancelled.is_set()


async def test_completed_message_wins_permission_race(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    abort_calls = 0

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal abort_calls
        if method == "POST" and path.endswith("/message"):
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "completed"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        if path.endswith("/abort"):
            abort_calls += 1
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        raise AssertionError(f"unexpected request: {method} {path}")

    async def immediate_permission(session_id, *, directory):
        return PendingPermission(permission="question", pattern_count=1)

    backend._request = fake_request  # type: ignore[method-assign]
    backend._watch_pending_permission = immediate_permission  # type: ignore[method-assign]

    assert await backend.continue_session("ses_race", "hello") == "completed"
    assert abort_calls == 0


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


async def test_continue_session_surfaces_embedded_response_error(make_backend):
    backend = make_backend(_msg_error_handler())

    with pytest.raises(
        RuntimeError,
        match=r"APIError \(HTTP 401\): Missing bearer authentication in header",
    ):
        await backend.continue_session("ses_existing", "again")


async def test_continue_session_serializes_same_native_session(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    active = 0
    max_active = 0

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal active, max_active
        if method == "POST" and path.endswith("/message"):
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0.01)
            active -= 1
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": path}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    await asyncio.gather(
        backend.continue_session("ses_shared", "first"),
        backend.continue_session("ses_shared", "second"),
    )
    assert max_active == 1


async def test_continue_session_uses_config_default_timeout_when_none(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    captured_timeouts = []

    async def fake_request(
        method,
        path,
        *,
        json=None,
        headers=None,
        timeout=httpx.USE_CLIENT_DEFAULT,
    ):
        if path.endswith("/message"):
            captured_timeouts.append(timeout)
        return httpx.Response(
            200,
            json={"parts": [{"type": "text", "text": "ok"}]},
            request=httpx.Request(method, f"http://test{path}"),
        )

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_default_timeout", "hello") == "ok"
    assert len(captured_timeouts) == 1
    assert isinstance(captured_timeouts[0], httpx.Timeout)
    assert captured_timeouts[0].connect == 5.0
    assert captured_timeouts[0].read == 1200.0  # config default


async def test_continue_session_per_call_timeout_reaches_httpx(monkeypatch):
    """A per-call timeout=120 produces an httpx.Timeout(120) on the message request."""
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    captured_timeouts = []

    async def fake_request(
        method,
        path,
        *,
        json=None,
        headers=None,
        timeout=httpx.USE_CLIENT_DEFAULT,
    ):
        if path.endswith("/message"):
            captured_timeouts.append(timeout)
        return httpx.Response(
            200,
            json={"parts": [{"type": "text", "text": "ok"}]},
            request=httpx.Request(method, f"http://test{path}"),
        )

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_t", "hello", timeout=120.0) == "ok"
    assert len(captured_timeouts) == 1
    assert isinstance(captured_timeouts[0], httpx.Timeout)
    assert captured_timeouts[0].read == 120.0
    assert captured_timeouts[0].connect == 5.0


async def test_continue_session_timeout_zero_falls_back_to_default(monkeypatch):
    """Per-call timeout=0 falls back to the config default rather than timing out instantly."""
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    captured_timeouts = []

    async def fake_request(
        method,
        path,
        *,
        json=None,
        headers=None,
        timeout=httpx.USE_CLIENT_DEFAULT,
    ):
        if path.endswith("/message"):
            captured_timeouts.append(timeout)
        return httpx.Response(
            200,
            json={"parts": [{"type": "text", "text": "ok"}]},
            request=httpx.Request(method, f"http://test{path}"),
        )

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_zero", "hello", timeout=0) == "ok"
    assert len(captured_timeouts) == 1
    assert isinstance(captured_timeouts[0], httpx.Timeout)
    assert captured_timeouts[0].read == 1200.0  # fell back to default


async def test_continue_session_cleans_up_session_tracking(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        return httpx.Response(
            200,
            json={"parts": [{"type": "text", "text": "ok"}]},
            request=httpx.Request(method, f"http://test{path}"),
        )

    backend._request = fake_request  # type: ignore[method-assign]

    assert await backend.continue_session("ses_cleanup", "hello") == "ok"
    assert "ses_cleanup" not in backend._tasks
    assert "ses_cleanup" not in backend._session_locks
    assert "ses_cleanup" not in backend._session_lock_users


async def test_cancelled_send_aborts_before_queued_continue_starts(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    first_started = asyncio.Event()
    order: list[str] = []

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if method == "POST" and path.endswith("/message"):
            text = json["parts"][0]["text"]
            order.append(f"message:{text}")
            if text == "first":
                first_started.set()
                await asyncio.Event().wait()
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "ok"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        if path.endswith("/abort"):
            order.append("abort")
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    first = asyncio.create_task(backend.continue_session("ses_shared", "first"))
    await first_started.wait()
    second = asyncio.create_task(backend.continue_session("ses_shared", "second"))
    while len(backend._tasks.get("ses_shared", ())) < 2:
        await asyncio.sleep(0)

    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    assert await second == "ok"
    assert order == ["message:first", "abort", "message:second"]
    assert "ses_shared" not in backend._tasks
    assert "ses_shared" not in backend._session_locks
    assert "ses_shared" not in backend._session_lock_users


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


async def test_abort_cancels_running_task(make_backend, captured_requests):
    backend = make_backend(_msg_handler())
    started = asyncio.Event()

    async def blocker():
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(blocker())
    await started.wait()
    backend._tasks["ses_block"] = {task}

    await backend.abort("ses_block")
    assert task.cancelled() or task.done()
    assert ("POST", "/session/ses_block/abort") in [
        (r.method, r.url.path) for r in captured_requests
    ]


async def test_abort_ignores_native_http_error_and_untracks_cancelled_task(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    started = asyncio.Event()

    async def blocker():
        started.set()
        await asyncio.Event().wait()

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if path.endswith("/abort"):
            return httpx.Response(
                500,
                json={"error": "server error"},
                request=httpx.Request(method, f"http://test{path}"),
            )
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    task = asyncio.create_task(blocker())
    await started.wait()
    backend._tasks["ses_500"] = {task}
    backend._request = fake_request  # type: ignore[method-assign]

    await backend.abort("ses_500")
    assert task.cancelled()
    assert "ses_500" not in backend._tasks


async def test_abort_cancels_queued_same_session_task(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    backend._is_healthy = monkeypatch_async(False)  # type: ignore[method-assign]

    started_first = asyncio.Event()
    message_calls = 0

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal message_calls
        if method == "POST" and path.endswith("/message"):
            message_calls += 1
            if message_calls == 1:
                started_first.set()
                await asyncio.Event().wait()
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "ok"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    first = asyncio.create_task(backend.continue_session("ses_shared", "first"))
    await started_first.wait()
    second = asyncio.create_task(backend.continue_session("ses_shared", "second"))
    while len(backend._tasks.get("ses_shared", ())) < 2:
        await asyncio.sleep(0)

    await backend.abort("ses_shared")

    assert first.cancelled()
    assert second.cancelled()
    assert message_calls == 1
    assert "ses_shared" not in backend._tasks
    assert "ses_shared" not in backend._session_locks
    assert "ses_shared" not in backend._session_lock_users


async def test_abort_cancels_continue_waiting_for_autoserve(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=True))
    ensure_started = asyncio.Event()
    abort_calls = 0
    message_calls = 0

    async def fake_ensure_server():
        ensure_started.set()
        await asyncio.Event().wait()

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal abort_calls, message_calls
        if path.endswith("/abort"):
            abort_calls += 1
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        if path.endswith("/message"):
            message_calls += 1
            return httpx.Response(
                200,
                json={"parts": [{"type": "text", "text": "ok"}]},
                request=httpx.Request(method, f"http://test{path}"),
            )
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._ensure_server = fake_ensure_server  # type: ignore[method-assign]
    backend._request = fake_request  # type: ignore[method-assign]

    task = asyncio.create_task(backend.continue_session("ses_autoserve", "hello"))
    await ensure_started.wait()

    await backend.abort("ses_autoserve")

    with pytest.raises(asyncio.CancelledError):
        await task
    assert abort_calls >= 1
    assert message_calls == 0
    assert "ses_autoserve" not in backend._tasks
    assert "ses_autoserve" not in backend._session_locks
    assert "ses_autoserve" not in backend._session_lock_users


async def test_abort_does_not_autoserve_unknown_unhealthy_session(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=True))
    abort_attempted = False

    async def fail_ensure_server():
        raise AssertionError("abort should not autoserve")

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        nonlocal abort_attempted
        abort_attempted = path.endswith("/abort")
        raise httpx.ConnectError("server down")

    backend._ensure_server = fail_ensure_server  # type: ignore[method-assign]
    backend._request = fake_request  # type: ignore[method-assign]

    await backend.abort("ses_unknown")
    assert abort_attempted


async def test_timeout_attempts_bounded_native_abort(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._ensure_server = monkeypatch_async(None)  # type: ignore[method-assign]
    abort_timeouts = []

    async def fake_request(method, path, *, json=None, headers=None, timeout=None):
        if path.endswith("/message"):
            raise httpx.ReadTimeout("slow response")
        if path.endswith("/abort"):
            abort_timeouts.append(timeout)
            return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))
        return httpx.Response(200, json=True, request=httpx.Request(method, f"http://test{path}"))

    backend._request = fake_request  # type: ignore[method-assign]

    with pytest.raises(httpx.TimeoutException):
        await backend.continue_session("ses_slow", "hello")
    assert len(abort_timeouts) == 1
    assert isinstance(abort_timeouts[0], httpx.Timeout)
    assert abort_timeouts[0].connect == HEALTH_TIMEOUT
    assert "ses_slow" not in backend._tasks
    assert "ses_slow" not in backend._session_locks
    assert "ses_slow" not in backend._session_lock_users


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
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=True))
    backend._is_healthy = monkeypatch_async(False)  # type: ignore[method-assign]
    backend._wait_healthy = monkeypatch_async(True)  # type: ignore[method-assign]

    await backend._ensure_server()
    assert backend._spawned is True
    assert spawned and spawned[0][:2] == ["opencode", "serve"]
    assert "--port" in spawned[0]


async def test_autoserve_disabled_raises_when_down(monkeypatch):
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=False))
    backend._is_healthy = monkeypatch_async(False)  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="not reachable"):
        await backend._ensure_server()


def test_autoserve_base_url_requires_explicit_port():
    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1", autoserve=True))
    with pytest.raises(RuntimeError, match="explicit port"):
        backend._port_from_url()


async def test_ensure_server_lock_prevents_duplicate_spawns(monkeypatch):
    """Concurrent _ensure_server calls only spawn one opencode process."""
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

    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=True))
    # Server becomes healthy after it's spawned; before that health checks fail.
    healthy = [False]

    async def fake_is_healthy():
        return healthy[0]

    original_start = backend._start_server

    async def fake_start_server():
        await original_start()
        healthy[0] = True

    backend._is_healthy = fake_is_healthy  # type: ignore[method-assign]
    backend._start_server = fake_start_server  # type: ignore[method-assign]

    # Fire 5 concurrent _ensure_server calls; only one should spawn.
    await asyncio.gather(*[backend._ensure_server() for _ in range(5)])
    assert len(spawned) == 1


async def test_start_server_cleans_previous_spawn_before_replacing(monkeypatch):
    class FakeProc:
        def __init__(self) -> None:
            self.returncode = None
            self.terminated = False
            self.killed = False
            self.waits: list[float | None] = []

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = -15

        def wait(self, timeout=None):
            self.waits.append(timeout)
            return self.returncode

        def kill(self) -> None:
            self.killed = True
            self.returncode = -9

    procs: list[FakeProc] = []

    def fake_popen(cmd, **kwargs):
        proc = FakeProc()
        procs.append(proc)
        return proc

    monkeypatch.setattr("cross_agent.backends.opencode.subprocess.Popen", fake_popen)

    backend = OpenCodeBackend(OpenCodeConfig(base_url="http://127.0.0.1:4096", autoserve=True))
    await backend._start_server()
    first = procs[0]
    await backend._start_server()

    assert first.terminated
    assert not first.killed
    assert first.waits == [5]
    assert len(procs) == 2
    assert backend._proc is procs[1]
    assert backend._spawned is True


# -------------------------------------------------------------- url encoding


async def test_session_id_url_encoded_in_message_path(make_backend, captured_requests):
    """A session_id with slashes/? is URL-encoded so it can't retarget the path."""
    backend = make_backend(_msg_handler("encoded-ok"))
    response = await backend.continue_session("ses/evil?abort", "hello")

    # The message endpoint is reached (returns our text), proving the encoded
    # path doesn't get split by the slash or truncated by the ?.
    assert response == "encoded-ok"
    msg_req = next(r for r in captured_requests if r.method == "POST")
    assert b"ses%2Fevil%3Fabort" in msg_req.url.raw_path


async def test_session_id_url_encoded_in_abort_path(make_backend, captured_requests):
    backend = make_backend(_msg_handler())
    await backend.abort("ses/evil?abort")
    abort_req = next(
        r for r in captured_requests if r.method == "POST" and b"abort" in r.url.raw_path
    )
    assert b"ses%2Fevil%3Fabort" in abort_req.url.raw_path


def monkeypatch_async(value):
    """Return an async function that resolves to ``value`` (for stubbing backend methods)."""

    async def _stub(*args, **kwargs):
        return value

    return _stub
