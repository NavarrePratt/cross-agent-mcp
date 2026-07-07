"""Tool factory: registers prompt/continue/abort tools for a backend on a FastMCP server."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from cross_agent.backends.base import Backend


def register_tools(mcp: FastMCP, backend: Backend) -> None:
    """Register ``<name>_prompt``, ``<name>_continue``, ``<name>_abort`` for one backend.

    Each tool surfaces backend errors as ``"Error: ..."`` text rather than raising,
    so an MCP client always gets a readable result.
    """
    prefix = backend.name

    def _prompt_desc() -> str:
        return (
            f"Send a one-shot prompt to {prefix} for adversarial review.\n\n"
            "Read-only by default: the backend can read files, search, and fetch, but "
            "cannot write or execute commands. Pass read_only=False only if you "
            "explicitly want write/exec access.\n\n"
            f"Returns the response text prefixed with the session id, which can be "
            f"passed to {prefix}_continue to resume the conversation."
        )

    def _continue_desc() -> str:
        return (
            f"Continue an existing {prefix} session by id.\n\n"
            f"Pass the session_id returned from {prefix}_prompt. Read-only by default, "
            "matching the one-shot tool."
        )

    def _abort_desc() -> str:
        return f"Abort a running {prefix} session by id. No-op if it already finished."

    async def _prompt(
        prompt: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
    ) -> str:
        try:
            text, sid = await backend.prompt(prompt, read_only, directory, model)
        except Exception as exc:  # noqa: BLE001 - surface as tool result, not a crash
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Session: {sid}\n\n{text}"

    async def _continue(
        session_id: str,
        prompt: str,
        read_only: bool = True,
        directory: str | None = None,
    ) -> str:
        try:
            return await backend.continue_session(session_id, prompt, read_only, directory)
        except Exception as exc:  # noqa: BLE001 - surface as tool result, not a crash
            return f"Error: {type(exc).__name__}: {exc}"

    async def _abort(session_id: str) -> str:
        try:
            await backend.abort(session_id)
        except Exception as exc:  # noqa: BLE001 - surface as tool result, not a crash
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Session {session_id} aborted."

    # FastMCP derives the MCP tool name from the explicit ``name`` kwarg, so the
    # closures do not need distinct ``__name__`` attributes.
    mcp.tool(name=f"{prefix}_prompt", description=_prompt_desc())(_prompt)
    mcp.tool(name=f"{prefix}_continue", description=_continue_desc())(_continue)
    mcp.tool(name=f"{prefix}_abort", description=_abort_desc())(_abort)


__all__ = ["register_tools"]
