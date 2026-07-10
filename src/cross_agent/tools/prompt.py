"""Tool factory: registers prompt/continue/abort tools for a backend on a FastMCP server."""

from __future__ import annotations

import inspect
from typing import Annotated, Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from cross_agent.backends.base import Backend

PromptArg = Annotated[str, Field(description="Prompt text to send to the backend agent.")]
SessionIdArg = Annotated[
    str,
    Field(
        description=(
            "Native backend session id returned by this server or copied from "
            "the backend harness."
        )
    ),
]
ReadOnlyArg = Annotated[
    bool,
    Field(
        description=(
            "Request read-only behavior. Claude enforces tool restrictions; opencode "
            "requires a configured read-only agent for hard enforcement."
        )
    ),
]
DirectoryArg = Annotated[
    str | None,
    Field(description="Working directory the backend should use for file reads and searches."),
]
ModelArg = Annotated[
    str | None,
    Field(description="Optional backend-native model override for this prompt call."),
]
TimeoutArg = Annotated[
    float | None,
    Field(
        description=(
            "Timeout in seconds for this MCP call. null uses CROSS_AGENT_QUERY_TIMEOUT; "
            "non-positive and non-finite values fall back to that configured default."
        )
    ),
]


def _timeout_passing_style(callable_obj: Any) -> Literal["keyword", "positional", "unsupported"]:
    """Return how ``callable_obj`` accepts the optional ``timeout`` argument."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return "keyword"
    timeout = signature.parameters.get("timeout")
    if timeout is not None:
        if timeout.kind is inspect.Parameter.POSITIONAL_ONLY:
            return "positional"
        return "keyword"
    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return "keyword"
    return "unsupported"


async def _call_prompt(
    backend: Backend,
    prompt: str,
    *,
    read_only: bool,
    directory: str | None,
    model: str | None,
    timeout: float | None,
) -> tuple[str, str]:
    timeout_style = _timeout_passing_style(backend.prompt)
    if timeout_style == "unsupported":
        if timeout is not None:
            raise RuntimeError(f"{backend.name} backend does not support per-call timeouts")
        return await backend.prompt(prompt, read_only, directory, model)

    if timeout_style == "positional":
        return await backend.prompt(prompt, read_only, directory, model, timeout)
    return await backend.prompt(prompt, read_only, directory, model, timeout=timeout)


async def _call_continue(
    backend: Backend,
    session_id: str,
    prompt: str,
    *,
    read_only: bool,
    directory: str | None,
    timeout: float | None,
) -> str:
    timeout_style = _timeout_passing_style(backend.continue_session)
    if timeout_style == "unsupported":
        if timeout is not None:
            raise RuntimeError(f"{backend.name} backend does not support per-call timeouts")
        return await backend.continue_session(session_id, prompt, read_only, directory)

    if timeout_style == "positional":
        return await backend.continue_session(session_id, prompt, read_only, directory, timeout)
    return await backend.continue_session(session_id, prompt, read_only, directory, timeout=timeout)


def register_tools(mcp: FastMCP, backend: Backend) -> None:
    """Register ``<name>_prompt``, ``<name>_continue``, ``<name>_abort`` for one backend.

    Each tool surfaces backend errors as ``"Error: ..."`` text rather than raising,
    so an MCP client always gets a readable result.
    """
    prefix = backend.name

    def _prompt_desc() -> str:
        return (
            f"Start a native {prefix} session for adversarial review.\n\n"
            "Read-only is requested by default. Enforcement depends on the backend "
            "and configuration: Claude restricts tools directly; opencode requires a "
            "configured read-only agent for hard enforcement. Pass read_only=False "
            "only if you explicitly want write/exec access.\n\n"
            f"Returns the response text prefixed with the native session id, which can be "
            f"passed to {prefix}_continue to resume the conversation.\n\n"
            "timeout (seconds) overrides the configured default for this MCP call; "
            "null uses CROSS_AGENT_QUERY_TIMEOUT."
        )

    def _continue_desc() -> str:
        return (
            f"Continue an existing native {prefix} session by id.\n\n"
            f"The session_id may come from {prefix}_prompt or directly from the "
            f"{prefix} harness. Read-only is requested by default, matching the "
            "prompt tool.\n\n"
            "timeout (seconds) overrides the configured default for this MCP call; "
            "null uses CROSS_AGENT_QUERY_TIMEOUT."
        )

    def _abort_desc() -> str:
        return (
            f"Request cancellation of a running {prefix} session by id. "
            "Best-effort for unknown or already-finished sessions."
        )

    async def _prompt(
        prompt: PromptArg,
        read_only: ReadOnlyArg = True,
        directory: DirectoryArg = None,
        model: ModelArg = None,
        timeout: TimeoutArg = None,
    ) -> str:
        try:
            text, sid = await _call_prompt(
                backend,
                prompt,
                read_only=read_only,
                directory=directory,
                model=model,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surface as tool result, not a crash
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Session: {sid}\n\n{text}"

    async def _continue(
        session_id: SessionIdArg,
        prompt: PromptArg,
        read_only: ReadOnlyArg = True,
        directory: DirectoryArg = None,
        timeout: TimeoutArg = None,
    ) -> str:
        try:
            return await _call_continue(
                backend,
                session_id,
                prompt,
                read_only=read_only,
                directory=directory,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001 - surface as tool result, not a crash
            return f"Error: {type(exc).__name__}: {exc}"

    async def _abort(session_id: SessionIdArg) -> str:
        try:
            await backend.abort(session_id)
        except Exception as exc:  # noqa: BLE001 - surface as tool result, not a crash
            return f"Error: {type(exc).__name__}: {exc}"
        return f"Abort requested for session {session_id}."

    # FastMCP derives the MCP tool name from the explicit ``name`` kwarg, so the
    # closures do not need distinct ``__name__`` attributes.
    mcp.tool(name=f"{prefix}_prompt", description=_prompt_desc())(_prompt)
    mcp.tool(name=f"{prefix}_continue", description=_continue_desc())(_continue)
    mcp.tool(name=f"{prefix}_abort", description=_abort_desc())(_abort)


__all__ = ["register_tools"]
