"""Tool factory: registers prompt/continue/abort tools for a backend on a FastMCP server."""

from __future__ import annotations

import inspect
from typing import Any

from mcp.server.fastmcp import FastMCP

from cross_agent.backends.base import Backend


def _accepts_kwarg(callable_obj: Any, name: str) -> bool:
    """Return whether ``callable_obj`` can accept keyword ``name``."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    if name in signature.parameters:
        return signature.parameters[name].kind is not inspect.Parameter.POSITIONAL_ONLY
    return any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


async def _call_prompt(
    backend: Backend,
    prompt: str,
    *,
    read_only: bool,
    directory: str | None,
    model: str | None,
    timeout: float | None,
) -> tuple[str, str]:
    if not _accepts_kwarg(backend.prompt, "timeout"):
        if timeout is not None:
            raise RuntimeError(f"{backend.name} backend does not support per-call timeouts")
        return await backend.prompt(prompt, read_only, directory, model)

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
    if not _accepts_kwarg(backend.continue_session, "timeout"):
        if timeout is not None:
            raise RuntimeError(f"{backend.name} backend does not support per-call timeouts")
        return await backend.continue_session(session_id, prompt, read_only, directory)

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
            "timeout (seconds) overrides the backend's default query timeout. Set it "
            "high for long reviews; the default is 1200s (20 minutes)."
        )

    def _continue_desc() -> str:
        return (
            f"Continue an existing native {prefix} session by id.\n\n"
            f"The session_id may come from {prefix}_prompt or directly from the "
            f"{prefix} harness. Read-only is requested by default, matching the "
            "prompt tool.\n\n"
            "timeout (seconds) overrides the backend's default query timeout."
        )

    def _abort_desc() -> str:
        return (
            f"Request cancellation of a running {prefix} session by id. "
            "No-op if it already finished."
        )

    async def _prompt(
        prompt: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
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
        session_id: str,
        prompt: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
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

    async def _abort(session_id: str) -> str:
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
