"""Tests for MCP server tool registration and backend detection."""

from __future__ import annotations

import sys

import pytest

from cross_agent.server import build_server, get_available_backends
from cross_agent.tools.prompt import _call_continue, _call_prompt


class LegacyBackend:
    name = "legacy"
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]]

    def __init__(self) -> None:
        self.calls = []

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
    ) -> tuple[str, str]:
        self.calls.append(("prompt", (text, read_only, directory, model), {}))
        return "legacy response", "legacy_session"

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
    ) -> str:
        self.calls.append(("continue", (session_id, text, read_only, directory), {}))
        return "legacy continued"

    async def abort(self, session_id: str) -> None:
        self.calls.append(("abort", (session_id,), {}))


class PositionalOnlyLegacyBackend:
    name = "positional_legacy"
    calls: list[tuple[str, tuple[object, ...]]]

    def __init__(self) -> None:
        self.calls = []

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        /,
    ) -> tuple[str, str]:
        self.calls.append(("prompt", (text, read_only, directory, model)))
        return "positional response", "positional_session"

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        /,
    ) -> str:
        self.calls.append(("continue", (session_id, text, read_only, directory)))
        return "positional continued"

    async def abort(self, session_id: str, /) -> None:
        self.calls.append(("abort", (session_id,)))


class ModernBackend:
    name = "modern"
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]]

    def __init__(self) -> None:
        self.calls = []

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        self.calls.append(
            ("prompt", (text, read_only, directory, model), {"timeout": timeout})
        )
        return "modern response", "modern_session"

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        self.calls.append(
            ("continue", (session_id, text, read_only, directory), {"timeout": timeout})
        )
        return "modern continued"

    async def abort(self, session_id: str) -> None:
        self.calls.append(("abort", (session_id,), {}))


def _tool_names(mcp) -> list[str]:
    import asyncio

    return [t.name for t in asyncio.run(mcp.list_tools())]


def test_opencode_tools_always_register():
    mcp = build_server()
    names = _tool_names(mcp)
    assert "opencode_prompt" in names
    assert "opencode_continue" in names
    assert "opencode_abort" in names


def test_three_tools_per_backend():
    mcp = build_server()
    names = _tool_names(mcp)
    # opencode is always present.
    oc = [n for n in names if n.startswith("opencode_")]
    assert oc == ["opencode_prompt", "opencode_continue", "opencode_abort"]


def test_get_available_backends_includes_opencode_always():
    backends = get_available_backends()
    names = [b.name for b in backends]
    assert "opencode" in names


def test_claude_backend_registered_when_sdk_installed():
    """If claude_agent_sdk is importable, claude_* tools register."""
    try:
        import claude_agent_sdk  # noqa: F401
    except ImportError:
        pytest.skip("claude-agent-sdk not installed in this environment")

    mcp = build_server()
    names = _tool_names(mcp)
    assert "claude_prompt" in names
    assert "claude_continue" in names
    assert "claude_abort" in names


def test_claude_backend_skipped_when_sdk_missing(monkeypatch):
    """When claude_agent_sdk cannot be imported, only opencode_* tools register."""

    # Force the SDK import inside get_available_backends to fail. Setting the
    # module to None in sys.modules makes ``import claude_agent_sdk`` raise
    # ImportError, simulating the optional dependency not being installed.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    backends = get_available_backends()
    names = [b.name for b in backends]
    assert names == ["opencode"]

    # build_server should also only register opencode_* tools.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)
    mcp = build_server()
    tool_names = _tool_names(mcp)
    assert not any(n.startswith("claude_") for n in tool_names)


def test_tool_functions_have_distinct_descriptions():
    mcp = build_server()
    names = _tool_names(mcp)
    # opencode and claude (if present) tools never collide.
    assert len(names) == len(set(names))


def test_prompt_and_continue_tool_schemas_expose_timeout():
    import asyncio

    mcp = build_server()
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    assert "timeout" in tools["opencode_prompt"].inputSchema["properties"]
    assert "timeout" in tools["opencode_continue"].inputSchema["properties"]


async def test_call_prompt_preserves_legacy_backend_without_timeout():
    backend = LegacyBackend()

    text, sid = await _call_prompt(
        backend,
        "hello",
        read_only=True,
        directory="/tmp/project",
        model="provider/model",
        timeout=None,
    )

    assert (text, sid) == ("legacy response", "legacy_session")
    assert backend.calls == [
        ("prompt", ("hello", True, "/tmp/project", "provider/model"), {})
    ]


async def test_call_continue_preserves_legacy_backend_without_timeout():
    backend = LegacyBackend()

    text = await _call_continue(
        backend,
        "legacy_session",
        "again",
        read_only=False,
        directory="/tmp/project",
        timeout=None,
    )

    assert text == "legacy continued"
    assert backend.calls == [
        ("continue", ("legacy_session", "again", False, "/tmp/project"), {})
    ]


async def test_call_prompt_rejects_timeout_for_legacy_backend():
    backend = LegacyBackend()

    with pytest.raises(RuntimeError, match="does not support per-call timeouts"):
        await _call_prompt(
            backend,
            "hello",
            read_only=True,
            directory=None,
            model=None,
            timeout=1.0,
        )


async def test_call_continue_rejects_timeout_for_legacy_backend():
    backend = LegacyBackend()

    with pytest.raises(RuntimeError, match="does not support per-call timeouts"):
        await _call_continue(
            backend,
            "legacy_session",
            "again",
            read_only=True,
            directory=None,
            timeout=1.0,
        )


async def test_call_helpers_pass_timeout_to_modern_backend():
    backend = ModernBackend()

    text, sid = await _call_prompt(
        backend,
        "hello",
        read_only=True,
        directory="/tmp/project",
        model="provider/model",
        timeout=2.5,
    )
    continued = await _call_continue(
        backend,
        "modern_session",
        "again",
        read_only=False,
        directory="/tmp/project",
        timeout=3.5,
    )

    assert (text, sid, continued) == (
        "modern response",
        "modern_session",
        "modern continued",
    )
    assert backend.calls == [
        ("prompt", ("hello", True, "/tmp/project", "provider/model"), {"timeout": 2.5}),
        ("continue", ("modern_session", "again", False, "/tmp/project"), {"timeout": 3.5}),
    ]


async def test_call_prompt_preserves_positional_only_legacy_backend_without_timeout():
    backend = PositionalOnlyLegacyBackend()

    text, sid = await _call_prompt(
        backend,
        "hello",
        read_only=True,
        directory="/tmp/project",
        model="provider/model",
        timeout=None,
    )

    assert (text, sid) == ("positional response", "positional_session")
    assert backend.calls == [
        ("prompt", ("hello", True, "/tmp/project", "provider/model"))
    ]


async def test_call_continue_preserves_positional_only_legacy_backend_without_timeout():
    backend = PositionalOnlyLegacyBackend()

    text = await _call_continue(
        backend,
        "positional_session",
        "again",
        read_only=False,
        directory="/tmp/project",
        timeout=None,
    )

    assert text == "positional continued"
    assert backend.calls == [
        ("continue", ("positional_session", "again", False, "/tmp/project"))
    ]
