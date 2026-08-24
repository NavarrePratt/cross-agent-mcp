"""MCPServer entry point: detect available backends and register their tools."""

from __future__ import annotations

import sys
from importlib.metadata import version

from mcp.server import MCPServer

from cross_agent.backends.base import Backend
from cross_agent.backends.opencode import OpenCodeBackend
from cross_agent.tools.prompt import register_tools

SERVER_NAME = "cross-agent"


def get_available_backends() -> list[Backend]:
    """Return the backends available in this install.

    The opencode backend is always available (httpx ships with the package). The
    Claude backend requires the optional ``claude-agent-sdk`` extra; if it is not
    installed we silently skip it so only ``opencode_*`` tools register.
    """
    backends: list[Backend] = [OpenCodeBackend()]
    try:
        import claude_agent_sdk  # noqa: F401  # local import keeps SDK optional
    except ImportError:
        return backends
    # Import the backend separately so real bugs in claude.py surface rather
    # than being masked as a missing optional dependency.
    from cross_agent.backends.claude import ClaudeBackend

    try:
        backends.append(ClaudeBackend())
    except Exception as exc:  # noqa: BLE001 - never let a backend ctor crash server startup
        print(
            f"cross-agent: Claude backend skipped: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
    return backends


def build_server() -> MCPServer:
    """Construct the MCP server with tools for every available backend."""
    mcp = MCPServer(
        SERVER_NAME,
        version=version("cross-agent-mcp"),
        instructions=(
            "Cross-family conversation server. Provides prompt, continue, "
            "and abort tools for each available backend (opencode for GLM-5.2, "
            "claude for Claude Code). Read-only is requested by default, with "
            "enforcement depending on backend configuration. Use the "
            "<backend>_prompt tool to start a backend-native conversation, "
            "<backend>_continue to resume a native session by id, and "
            "<backend>_abort to cancel a running session."
        ),
    )
    for backend in get_available_backends():
        register_tools(mcp, backend)
    return mcp


def main() -> None:
    """Stdio entry point for ``cross-agent-mcp`` (``[project.scripts]``)."""
    build_server().run(transport="stdio")


if __name__ == "__main__":
    main()
