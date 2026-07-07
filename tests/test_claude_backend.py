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


async def test_prompt_read_only_sets_tool_restrictions(monkeypatch):
    q = _async_query([_init(), _result("ok", "sid_ro")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    await backend.prompt("review", read_only=True)
    opts = q.captured["options"]
    assert opts.allowed_tools == READ_ONLY_ALLOWED
    assert opts.disallowed_tools == READ_ONLY_DISALLOWED


async def test_prompt_read_only_false_omits_tool_restrictions(monkeypatch):
    q = _async_query([_init(), _result("ok", "sid_rw")])
    monkeypatch.setattr("cross_agent.backends.claude.query", q)

    backend = ClaudeBackend()
    await backend.prompt("do work", read_only=False)
    opts = q.captured["options"]
    assert not opts.allowed_tools
    assert not opts.disallowed_tools


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
