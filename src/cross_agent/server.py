"""FastMCP entry point: detect available backends and register their tools."""

from __future__ import annotations

from collections.abc import Callable

from mcp.server.fastmcp import FastMCP

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
        from cross_agent.backends.claude import ClaudeBackend  # local import keeps SDK optional
    except ImportError:
        return backends
    try:
        backends.append(ClaudeBackend())
    except Exception:  # noqa: BLE001 - never let a backend ctor crash server startup
        pass
    return backends


def build_server() -> FastMCP:
    """Construct the FastMCP server with tools for every available backend."""
    mcp = FastMCP(SERVER_NAME)
    for backend in get_available_backends():
        register_tools(mcp, backend)
    return mcp


def main() -> None:
    """Stdio entry point for ``cross-agent-mcp`` (``[project.scripts]``)."""
    build_server().run(transport="stdio")


# Bind backend aclose on server shutdown so spawned opencode servers are cleaned up.
_original_atexit: Callable | None = None


if __name__ == "__main__":
    main()
