"""Unit tests for the Claude backend (mocked claude_agent_sdk.query)."""

from __future__ import annotations

import asyncio

import pytest

claude_sdk = pytest.importorskip("claude_agent_sdk")

from claude_agent_sdk import AssistantMessage, ResultMessage, SystemMessage, TextBlock  # noqa: E402

from cross_agent.backends.claude import (  # noqa: E402
    READ_ONLY_ALLOWED,
    READ_ONLY_DISALLOWED,
    ClaudeBackend,
)


def _async_query(messages):
    """Build an async generator that replays ``messages`` and records its options."""

    captured = {}

    async def _query(*, prompt, options=None):
        captured["prompt"] = prompt
        captured["options"] = options
        for msg in messages:
            yield msg

    _query.captured = captured  # type: ignore[attr-defined]
    return _query


def _result(result="PONG", session_id="sid_abc", is_error=False, errors=None) -> ResultMessage:
    return ResultMessage(
        subtype="result",
        duration_ms=1,
        duration_api_ms=1,
        is_error=is_error,
        num_turns=1,
        session_id=session_id,
        result=result,
        errors=errors,
    )


def _init(session_id="sid_abc") -> SystemMessage:
    return SystemMessage(subtype="init", data={"session_id": session_id})


async def test_prompt_returns_result_and_session_id(monkeypatch):
    msgs = [
        _init(),
        AssistantMessage(content=[TextBlock(text="thinking")], model="m"),
        _result("PONG", "sid_abc"),
    ]
    q = _async_query(msgs)
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    text, sid = await backend.prompt("Reply with exactly: PONG")
    assert text == "PONG"
    assert sid == "sid_abc"
    assert q.captured["prompt"] == "Reply with exactly: PONG"


async def test_prompt_success_does_not_stop_native_session(monkeypatch):
    stopped: list[str] = []
    q = _async_query([_init("sid_success"), _result("ok", "sid_success")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()

    async def fake_cli_stop(session_id):
        stopped.append(session_id)

    backend._cli_stop = fake_cli_stop  # type: ignore[method-assign]

    assert await backend.prompt("hi") == ("ok", "sid_success")
    assert stopped == []
    assert "sid_success" not in backend._tasks


async def test_prompt_read_only_sets_tool_restrictions(monkeypatch):
    q = _async_query([_init(), _result("ok", "sid_ro")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    await backend.prompt("review", read_only=True)
    opts = q.captured["options"]
    expected_tools = ["Read", "Glob", "Grep", "LS", "WebSearch", "WebFetch"]
    assert READ_ONLY_ALLOWED == expected_tools
    assert opts.tools == expected_tools
    assert opts.allowed_tools == expected_tools
    assert opts.disallowed_tools == READ_ONLY_DISALLOWED
    assert opts.permission_mode == "dontAsk"
    assert opts.setting_sources == []
    assert opts.strict_mcp_config is True
    assert opts.mcp_servers == {}
    assert opts.skills == []
    assert opts.disallowed_tools == ["mcp__*"]


async def test_prompt_read_only_false_omits_tool_restrictions(monkeypatch):
    q = _async_query([_init(), _result("ok", "sid_rw")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    await backend.prompt("do work", read_only=False)
    opts = q.captured["options"]
    assert opts.tools is None
    assert not opts.allowed_tools
    assert not opts.disallowed_tools
    assert opts.permission_mode is None
    assert opts.setting_sources is None
    assert opts.strict_mcp_config is False
    assert opts.mcp_servers == {}
    assert opts.skills is None


async def test_prompt_passes_cwd_and_model(monkeypatch):
    q = _async_query([_init(), _result("ok", "sid_cwd")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend(model="claude-sonnet-4")
    await backend.prompt("hi", directory="/tmp/work", model="claude-opus-4")
    opts = q.captured["options"]
    assert str(opts.cwd).endswith("/tmp/work")
    assert opts.model == "claude-opus-4"  # explicit arg beats backend default


async def test_continue_sets_resume(monkeypatch):
    q = _async_query([_init("sid_resume"), _result("ack", "sid_resume")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    out = await backend.continue_session("sid_resume", "again")
    assert out == "ack"
    assert q.captured["options"].resume == "sid_resume"


async def test_continue_tracks_native_session_before_init(monkeypatch):
    started = asyncio.Event()
    release = asyncio.Event()

    async def blocking_query(*, prompt, options=None):
        started.set()
        await release.wait()
        yield _init("sid_resume")
        yield _result("ack", "sid_resume")

    monkeypatch.setattr("cross_agent.backends.claude.query", blocking_query)

    backend = ClaudeBackend()
    task = asyncio.create_task(backend.continue_session("sid_resume", "again"))
    await started.wait()
    assert "sid_resume" in backend._tasks

    release.set()
    assert await task == "ack"
    assert "sid_resume" not in backend._tasks
    assert "sid_resume" not in backend._session_locks
    assert "sid_resume" not in backend._session_lock_users


async def test_continue_serializes_same_native_session(monkeypatch):
    active = 0
    max_active = 0

    async def serial_query(*, prompt, options=None):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        try:
            await asyncio.sleep(0.01)
            yield _init("sid_resume")
            yield _result("ack", "sid_resume")
        finally:
            active -= 1

    monkeypatch.setattr("cross_agent.backends.claude.query", serial_query)

    backend = ClaudeBackend()
    await asyncio.gather(
        backend.continue_session("sid_resume", "first"),
        backend.continue_session("sid_resume", "second"),
    )

    assert max_active == 1


async def test_continue_abort_cancels_running_native_session(monkeypatch):
    started = asyncio.Event()
    stopped: list[str] = []

    async def blocking_query(*, prompt, options=None):
        started.set()
        await asyncio.Event().wait()
        yield _init("sid_resume")

    monkeypatch.setattr("cross_agent.backends.claude.query", blocking_query)

    backend = ClaudeBackend()

    async def fake_cli_stop(session_id):
        stopped.append(session_id)

    backend._cli_stop = fake_cli_stop  # type: ignore[method-assign]
    task = asyncio.create_task(backend.continue_session("sid_resume", "again"))
    await started.wait()

    await backend.abort("sid_resume")
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped == ["sid_resume"]


async def test_prompt_outer_cancellation_stops_discovered_native_session(monkeypatch):
    started = asyncio.Event()
    stopped: list[str] = []

    async def blocking_query(*, prompt, options=None):
        yield _init("sid_outer_cancel")
        started.set()
        await asyncio.Event().wait()
        yield _result("never", "sid_outer_cancel")

    async def fake_cli_stop(session_id):
        stopped.append(session_id)

    monkeypatch.setattr("cross_agent.backends.claude.query", blocking_query)

    backend = ClaudeBackend()
    backend._cli_stop = fake_cli_stop  # type: ignore[method-assign]
    task = asyncio.create_task(backend.prompt("hi"))
    await started.wait()
    assert "sid_outer_cancel" in backend._tasks

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert stopped == ["sid_outer_cancel"]
    assert "sid_outer_cancel" not in backend._tasks


async def test_prompt_error_surfaces_message(monkeypatch):
    q = _async_query([_init(), _result(is_error=True, errors=["rate limited"], session_id="sid_e")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    with pytest.raises(RuntimeError, match="rate limited"):
        await backend.prompt("hi")


async def test_prompt_falls_back_to_assistant_text_when_result_none(monkeypatch):
    msgs = [
        _init("sid_txt"),
        AssistantMessage(content=[TextBlock(text="streamed answer")], model="m"),
        _result(None, "sid_txt"),
    ]
    q = _async_query(msgs)
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    text, sid = await backend.prompt("hi")
    assert text == "streamed answer"
    assert sid == "sid_txt"


async def test_abort_cancels_running_task(monkeypatch):
    backend = ClaudeBackend()
    stopped: list[str] = []

    async def fake_cli_stop(session_id):
        stopped.append(session_id)

    backend._cli_stop = fake_cli_stop  # type: ignore[method-assign]

    # A long-running coroutine we can cancel mid-flight.
    started = asyncio.Event()

    async def blocker():
        started.set()
        try:
            await asyncio.Event().wait()  # never resolves
        except asyncio.CancelledError:
            raise

    task = asyncio.create_task(blocker())
    await started.wait()
    backend._tasks["sid_block"] = task

    await backend.abort("sid_block")
    assert task.cancelled() or task.done()
    assert stopped == ["sid_block"]


async def test_abort_falls_back_to_cli_stop_for_unknown_session(monkeypatch):
    backend = ClaudeBackend()
    calls: list[list[str]] = []

    class FakeProc:
        returncode = 0

        async def wait(self):
            return 0

    async def fake_create_subprocess_exec(*args, **kwargs):
        calls.append(list(args))
        return FakeProc()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    await backend.abort("sid_unknown_finished")
    assert calls and calls[0] == ["claude", "stop", "sid_unknown_finished"]


async def test_run_cleans_up_tasks_on_query_exception(monkeypatch):
    """If query() raises after registering a session_id, _tasks is cleaned up."""

    async def failing_query(*, prompt, options=None):
        yield SystemMessage(subtype="init", data={"session_id": "sid_fail"})
        raise RuntimeError("query exploded")

    monkeypatch.setattr("cross_agent.backends.claude.query", failing_query)

    backend = ClaudeBackend()
    with pytest.raises(RuntimeError, match="query exploded"):
        await backend.prompt("hi")
    assert "sid_fail" not in backend._tasks


async def test_cli_stop_kills_subprocess_on_timeout(monkeypatch):
    """When claude stop times out, the subprocess is killed and reaped."""
    backend = ClaudeBackend()

    killed = []
    waited = []

    class FakeProc:
        returncode = None

        async def wait(self):
            waited.append(True)
            return 0

        def kill(self):
            killed.append(True)

    proc = FakeProc()

    async def fake_create_subprocess_exec(*args, **kwargs):
        return proc

    async def fake_wait_for(coro, timeout):
        coro.close()  # clean up the never-awaited coroutine
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(asyncio, "wait_for", fake_wait_for)

    await backend._cli_stop("sid_timeout")
    assert killed, "subprocess was not killed on timeout"
    assert waited, "subprocess was not reaped after kill"


async def test_prompt_timeout_surfaces_as_error(monkeypatch):
    """A query that exceeds its timeout raises a TimeoutError surfaced by _run."""
    async def blocking_query(*, prompt, options=None):
        await asyncio.Event().wait()
        yield _init("sid_t")

    monkeypatch.setattr("cross_agent.backends.claude.query", blocking_query)

    backend = ClaudeBackend()
    with pytest.raises(asyncio.TimeoutError):
        await backend.prompt("hi", timeout=0.001)


async def test_prompt_timeout_stops_discovered_native_session(monkeypatch):
    stopped: list[str] = []

    async def blocking_query(*, prompt, options=None):
        yield _init("sid_prompt_timeout")
        await asyncio.Event().wait()
        yield _result("never", "sid_prompt_timeout")

    async def fake_cli_stop(session_id):
        stopped.append(session_id)

    monkeypatch.setattr("cross_agent.backends.claude.query", blocking_query)

    backend = ClaudeBackend()
    backend._cli_stop = fake_cli_stop  # type: ignore[method-assign]

    with pytest.raises(asyncio.TimeoutError):
        await backend.prompt("hi", timeout=0.01)
    assert stopped == ["sid_prompt_timeout"]


async def test_continue_timeout_stops_known_native_session(monkeypatch):
    backend = ClaudeBackend()
    stopped: list[str] = []

    async def blocking_query(*, prompt, options=None):
        await asyncio.Event().wait()
        yield _init("sid_resume")

    async def fake_cli_stop(session_id):
        stopped.append(session_id)

    monkeypatch.setattr("cross_agent.backends.claude.query", blocking_query)
    backend._cli_stop = fake_cli_stop  # type: ignore[method-assign]

    with pytest.raises(asyncio.TimeoutError):
        await backend.continue_session("sid_resume", "again", timeout=0.001)
    assert stopped == ["sid_resume"]
    assert "sid_resume" not in backend._tasks
    assert "sid_resume" not in backend._session_locks
    assert "sid_resume" not in backend._session_lock_users


async def test_query_timeout_env_var_sets_default(monkeypatch):
    """CROSS_AGENT_QUERY_TIMEOUT sets the Claude backend's default timeout."""
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "900")
    backend = ClaudeBackend()
    assert backend._query_timeout == 900.0


async def test_query_timeout_env_var_invalid_falls_back(monkeypatch):
    """Invalid CROSS_AGENT_QUERY_TIMEOUT falls back to the default."""
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "not-a-number")
    backend = ClaudeBackend()
    assert backend._query_timeout == 1200.0


async def test_query_timeout_env_var_zero_falls_back(monkeypatch):
    """Zero CROSS_AGENT_QUERY_TIMEOUT falls back to the default."""
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "0")
    backend = ClaudeBackend()
    assert backend._query_timeout == 1200.0


async def test_query_timeout_env_var_inf_falls_back(monkeypatch):
    """inf CROSS_AGENT_QUERY_TIMEOUT falls back to the default."""
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "inf")
    backend = ClaudeBackend()
    assert backend._query_timeout == 1200.0


async def test_prompt_timeout_none_uses_configured_default(monkeypatch):
    """When timeout=None, the configured _query_timeout is used, not an unbounded wait."""
    monkeypatch.setenv("CROSS_AGENT_QUERY_TIMEOUT", "42")
    q = _async_query([_init("sid_default_t"), _result("ok", "sid_default_t")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    captured_timeouts = []

    original_wait = asyncio.wait

    async def capturing_wait(fs, timeout=None):
        captured_timeouts.append(timeout)
        return await original_wait(fs, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait", capturing_wait)

    backend = ClaudeBackend()
    assert backend._query_timeout == 42.0
    await backend.prompt("hi")
    assert captured_timeouts == [42.0]


async def test_prompt_timeout_zero_falls_back_to_default(monkeypatch):
    """Per-call timeout=0 falls back to the configured default."""
    q = _async_query([_init("sid_zero_t"), _result("ok", "sid_zero_t")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    captured_timeouts = []

    original_wait = asyncio.wait

    async def capturing_wait(fs, timeout=None):
        captured_timeouts.append(timeout)
        return await original_wait(fs, timeout=timeout)

    monkeypatch.setattr(asyncio, "wait", capturing_wait)

    backend = ClaudeBackend()
    await backend.prompt("hi", timeout=0)
    assert captured_timeouts == [1200.0]  # default, not 0
