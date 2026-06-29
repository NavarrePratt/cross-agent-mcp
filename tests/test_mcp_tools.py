"""Tests for MCP server tool registration and backend detection."""

from __future__ import annotations

import sys

import pytest

from cross_agent.server import build_server, get_available_backends


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

    # Force the local import inside get_available_backends to fail.
    monkeypatch.setitem(sys.modules, "cross_agent.backends.claude", None)
    backends = get_available_backends()
    names = [b.name for b in backends]
    assert names == ["opencode"]

    # build_server should also only register opencode_* tools.
    monkeypatch.setitem(sys.modules, "cross_agent.backends.claude", None)
    mcp = build_server()
    tool_names = _tool_names(mcp)
    assert not any(n.startswith("claude_") for n in tool_names)


def test_tool_functions_have_distinct_descriptions():
    mcp = build_server()
    names = _tool_names(mcp)
    # opencode and claude (if present) tools never collide.
    assert len(names) == len(set(names))
