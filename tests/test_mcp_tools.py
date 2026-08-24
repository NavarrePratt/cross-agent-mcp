"""Tests for MCP server tool registration and backend detection."""

from __future__ import annotations

import sys

import pytest
from mcp.server import MCPServer

from cross_agent.server import build_server, get_available_backends
from cross_agent.tools.prompt import _call_continue, _call_prompt, register_tools


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


class PositionalOnlyTimeoutBackend:
    name = "positional_timeout"
    calls: list[tuple[str, tuple[object, ...]]]

    def __init__(self) -> None:
        self.calls = []

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        /,
    ) -> tuple[str, str]:
        self.calls.append(("prompt", (text, read_only, directory, model, timeout)))
        return "positional timeout response", "positional_timeout_session"

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
        /,
    ) -> str:
        self.calls.append(("continue", (session_id, text, read_only, directory, timeout)))
        return "positional timeout continued"

    async def abort(self, session_id: str, /) -> None:
        self.calls.append(("abort", (session_id,)))


class KwargsBackend:
    name = "kwargs"
    calls: list[tuple[str, tuple[object, ...], dict[str, object]]]

    def __init__(self) -> None:
        self.calls = []

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        **kwargs: object,
    ) -> tuple[str, str]:
        self.calls.append(("prompt", (text, read_only, directory, model), kwargs))
        return "kwargs response", "kwargs_session"

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        **kwargs: object,
    ) -> str:
        self.calls.append(("continue", (session_id, text, read_only, directory), kwargs))
        return "kwargs continued"

    async def abort(self, session_id: str) -> None:
        self.calls.append(("abort", (session_id,), {}))


class FailingBackend(ModernBackend):
    name = "failing"

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        raise ValueError("prompt failed")

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        raise RuntimeError("continue failed")

    async def abort(self, session_id: str) -> None:
        raise OSError("abort failed")


def _tool_names(mcp) -> list[str]:
    import asyncio

    return [t.name for t in asyncio.run(mcp.list_tools())]


async def _call_tool_text(mcp: MCPServer, name: str, arguments: dict[str, object]) -> str:
    result = await mcp.call_tool(name, arguments)
    return "\n".join(item.text for item in result.content)


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
    import asyncio

    tools = asyncio.run(mcp.list_tools())
    names = [tool.name for tool in tools]
    # opencode and claude (if present) tools never collide.
    assert len(names) == len(set(names))
    for tool in tools:
        assert tool.description
    opencode_descriptions = [
        tool.description for tool in tools if tool.name.startswith("opencode_")
    ]
    assert len(opencode_descriptions) == len(set(opencode_descriptions))


def test_tool_schemas_expose_parameter_descriptions():
    import asyncio

    mcp = build_server()
    tools = {tool.name: tool for tool in asyncio.run(mcp.list_tools())}
    prompt_props = tools["opencode_prompt"].input_schema["properties"]
    continue_props = tools["opencode_continue"].input_schema["properties"]
    abort_props = tools["opencode_abort"].input_schema["properties"]

    for name in ("prompt", "read_only", "directory", "model", "timeout"):
        assert prompt_props[name]["description"]
    for name in ("session_id", "prompt", "read_only", "directory", "timeout"):
        assert continue_props[name]["description"]
    assert abort_props["session_id"]["description"]
    assert "1200" not in tools["opencode_prompt"].description
    assert "1200" not in tools["opencode_continue"].description


async def test_registered_tool_wrappers_return_public_output_contracts():
    backend = ModernBackend()
    mcp = MCPServer("contract")
    register_tools(mcp, backend)

    prompt = await _call_tool_text(
        mcp,
        "modern_prompt",
        {"prompt": "hello", "directory": "/tmp/project", "timeout": 2.5},
    )
    continued = await _call_tool_text(
        mcp,
        "modern_continue",
        {"session_id": "modern_session", "prompt": "again", "timeout": 3.5},
    )
    aborted = await _call_tool_text(
        mcp,
        "modern_abort",
        {"session_id": "modern_session"},
    )

    assert prompt == "Session: modern_session\n\nmodern response"
    assert continued == "modern continued"
    assert aborted == "Abort requested for session modern_session."


async def test_registered_tool_wrappers_return_error_text_contracts():
    backend = FailingBackend()
    mcp = MCPServer("errors")
    register_tools(mcp, backend)

    prompt = await _call_tool_text(mcp, "failing_prompt", {"prompt": "hello"})
    continued = await _call_tool_text(
        mcp,
        "failing_continue",
        {"session_id": "s", "prompt": "again"},
    )
    aborted = await _call_tool_text(mcp, "failing_abort", {"session_id": "s"})

    assert prompt == "Error: ValueError: prompt failed"
    assert continued == "Error: RuntimeError: continue failed"
    assert aborted == "Error: OSError: abort failed"


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


async def test_call_helpers_pass_positional_only_timeout():
    backend = PositionalOnlyTimeoutBackend()

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
        "positional_timeout_session",
        "again",
        read_only=False,
        directory="/tmp/project",
        timeout=3.5,
    )

    assert (text, sid, continued) == (
        "positional timeout response",
        "positional_timeout_session",
        "positional timeout continued",
    )
    assert backend.calls == [
        ("prompt", ("hello", True, "/tmp/project", "provider/model", 2.5)),
        ("continue", ("positional_timeout_session", "again", False, "/tmp/project", 3.5)),
    ]


async def test_call_helpers_pass_timeout_through_kwargs_backend():
    backend = KwargsBackend()

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
        "kwargs_session",
        "again",
        read_only=False,
        directory="/tmp/project",
        timeout=3.5,
    )

    assert (text, sid, continued) == (
        "kwargs response",
        "kwargs_session",
        "kwargs continued",
    )
    assert backend.calls == [
        ("prompt", ("hello", True, "/tmp/project", "provider/model"), {"timeout": 2.5}),
        ("continue", ("kwargs_session", "again", False, "/tmp/project"), {"timeout": 3.5}),
    ]
