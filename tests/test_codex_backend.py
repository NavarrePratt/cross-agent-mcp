"""Unit tests for the Codex backend, driven against a fake ``codex`` executable."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from cross_agent.backends.codex import CodexBackend

FAKE_CODEX = """\
import json, os, sys, time

record = {"argv": sys.argv[1:], "stdin": sys.stdin.read()}
with open(os.environ["FAKE_CODEX_RECORD"], "a") as fh:
    fh.write(json.dumps(record) + "\\n")

def emit(event):
    print(json.dumps(event), flush=True)

scenario = os.environ.get("FAKE_CODEX_SCENARIO", "ok")
thread_id = sys.argv[sys.argv.index("resume") + 1] if "resume" in sys.argv else "thread-new"
emit({"type": "thread.started", "thread_id": thread_id})
emit({"type": "item.completed", "item": {"type": "error", "message": "config warning"}})
emit({"type": "turn.started"})
if scenario == "ok":
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": "looking"}})
    emit({"type": "item.completed", "item": {"type": "agent_message", "text": "final answer"}})
    emit({"type": "turn.completed", "usage": {}})
elif scenario == "turn_failed":
    emit({"type": "turn.failed", "error": {"message": "model overloaded"}})
    sys.exit(1)
elif scenario == "crash":
    print("fatal: not logged in", file=sys.stderr)
    sys.exit(2)
elif scenario == "hang":
    time.sleep(60)
"""


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    script = tmp_path / "codex"
    script.write_text(f"#!{sys.executable}\n{FAKE_CODEX}")
    script.chmod(0o755)
    record = tmp_path / "record.jsonl"
    monkeypatch.setenv("FAKE_CODEX_RECORD", str(record))
    for var in ("CROSS_AGENT_CODEX_MODEL", "CROSS_AGENT_CODEX_REASONING_EFFORT"):
        monkeypatch.delenv(var, raising=False)

    def calls() -> list[dict]:
        return [json.loads(line) for line in record.read_text().splitlines()]

    return str(script), calls


async def test_prompt_returns_final_message_and_thread_id(fake_codex):
    binary, calls = fake_codex
    backend = CodexBackend(binary=binary)

    text, sid = await backend.prompt("-review src/ please", directory="/repo")

    assert (text, sid) == ("final answer", "thread-new")
    [call] = calls()
    assert call["stdin"] == "-review src/ please"
    assert call["argv"][-1] == "-"
    assert "--cd=/repo" in call["argv"]


async def test_read_only_isolates_user_config_and_sandboxes(fake_codex):
    binary, calls = fake_codex
    backend = CodexBackend(binary=binary)

    await backend.prompt("hi")
    await backend.prompt("hi", read_only=False)

    read_only, writable = (call["argv"] for call in calls())
    # Without --ignore-user-config Codex loads the user's MCP servers, which can
    # include this server, and plugins with write access.
    assert "--ignore-user-config" in read_only
    assert "--sandbox=read-only" in read_only
    assert "--ignore-user-config" not in writable
    assert not any(arg.startswith("--sandbox") for arg in writable)


async def test_model_and_reasoning_effort_apply_to_prompt_and_continue(fake_codex, monkeypatch):
    binary, calls = fake_codex
    monkeypatch.setenv("CROSS_AGENT_CODEX_MODEL", "gpt-env")
    monkeypatch.setenv("CROSS_AGENT_CODEX_REASONING_EFFORT", "high")
    backend = CodexBackend(binary=binary)

    await backend.prompt("hi", model="gpt-override")
    await backend.continue_session("thread-1", "again")

    prompt_call, continue_call = (call["argv"] for call in calls())
    assert "--model=gpt-override" in prompt_call
    assert "--model=gpt-env" in continue_call
    assert 'model_reasoning_effort="high"' in prompt_call
    assert 'model_reasoning_effort="high"' in continue_call


async def test_continue_resumes_thread_read_only(fake_codex):
    binary, calls = fake_codex
    backend = CodexBackend(binary=binary)

    assert await backend.continue_session("thread-1", "follow up") == "final answer"

    [call] = calls()
    argv = call["argv"]
    assert argv[argv.index("resume") :] == ["resume", "thread-1", "-"]
    assert "--sandbox=read-only" in argv
    assert call["stdin"] == "follow up"


async def test_continue_rejects_flag_like_session_id(fake_codex):
    binary, _ = fake_codex
    backend = CodexBackend(binary=binary)

    with pytest.raises(ValueError, match="invalid codex session id"):
        await backend.continue_session("--dangerously-bypass-approvals-and-sandbox", "hi")


@pytest.mark.parametrize(
    ("scenario", "match"),
    [
        ("turn_failed", r"codex turn failed \(session thread-new\): model overloaded"),
        ("crash", r"exited with code 2 \(session thread-new\): fatal: not logged in"),
    ],
)
async def test_failures_raise_with_session_id(fake_codex, monkeypatch, scenario, match):
    binary, _ = fake_codex
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", scenario)
    backend = CodexBackend(binary=binary)

    with pytest.raises(RuntimeError, match=match):
        await backend.prompt("hi")


async def test_timeout_kills_process_and_names_session(fake_codex, monkeypatch):
    binary, _ = fake_codex
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "hang")
    backend = CodexBackend(binary=binary)

    with pytest.raises(TimeoutError, match=r"session thread-new"):
        await backend.prompt("hi", timeout=1)
    assert backend._procs == {}


async def test_abort_stops_running_prompt(fake_codex, monkeypatch):
    binary, _ = fake_codex
    monkeypatch.setenv("FAKE_CODEX_SCENARIO", "hang")
    backend = CodexBackend(binary=binary)

    task = asyncio.create_task(backend.prompt("hi"))
    for _ in range(100):
        if "thread-new" in backend._procs:
            break
        await asyncio.sleep(0.05)
    await backend.abort("thread-new")

    with pytest.raises(RuntimeError, match="exited with code"):
        await asyncio.wait_for(task, timeout=10)


async def test_abort_unknown_session_is_noop(fake_codex):
    binary, _ = fake_codex
    await CodexBackend(binary=binary).abort("missing")


def test_server_registers_codex_tools_only_when_cli_found(fake_codex, monkeypatch):
    from cross_agent.server import get_available_backends

    binary, _ = fake_codex
    monkeypatch.setenv("CROSS_AGENT_CODEX_BIN", binary)
    assert "codex" in [b.name for b in get_available_backends()]

    monkeypatch.setenv("CROSS_AGENT_CODEX_BIN", str(Path(binary).parent / "missing-codex"))
    assert "codex" not in [b.name for b in get_available_backends()]
