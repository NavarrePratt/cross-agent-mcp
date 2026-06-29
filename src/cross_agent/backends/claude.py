"""Claude Code backend: drives ``claude-agent-sdk`` (spawns the Claude CLI in-process).

Auth comes from the Claude CLI's stored enterprise OAuth - no ``ANTHROPIC_API_KEY``
is required. The SDK inherits the calling environment.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
from typing import Any

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    query,
)

# Read-only toolset: pre-approve safe operations and remove write/bash entirely so
# the model cannot mutate the working tree or shell out, even if it tries.
READ_ONLY_ALLOWED = ["Read", "Glob", "Grep", "List", "WebSearch", "WebFetch"]
READ_ONLY_DISALLOWED = ["Write", "Edit", "Bash"]


def _text_from_content(content: list[Any]) -> str:
    chunks: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if isinstance(text, str) and text:
            chunks.append(text)
    return "\n".join(chunks)


class ClaudeBackend:
    """Backend wrapping ``claude-agent-sdk`` for cross-family review."""

    def __init__(self, model: str | None = None) -> None:
        self._model = model or os.environ.get("CROSS_AGENT_CLAUDE_MODEL")
        # session_id -> running asyncio task, for abort support.
        self._tasks: dict[str, asyncio.Task[Any]] = {}

    @property
    def name(self) -> str:
        return "claude"

    # ----------------------------------------------------------------- options

    def _options(
        self,
        *,
        read_only: bool,
        directory: str | None,
        model: str | None,
        resume: str | None = None,
    ) -> ClaudeAgentOptions:
        kwargs: dict[str, Any] = {}
        if read_only:
            kwargs["allowed_tools"] = list(READ_ONLY_ALLOWED)
            kwargs["disallowed_tools"] = list(READ_ONLY_DISALLOWED)
        if directory:
            kwargs["cwd"] = directory
        chosen = model or self._model
        if chosen:
            kwargs["model"] = chosen
        if resume:
            kwargs["resume"] = resume
        return ClaudeAgentOptions(**kwargs)

    async def _run(self, text: str, options: ClaudeAgentOptions) -> tuple[str, str]:
        """Drive a ``query()`` to completion, tracking the task for abort."""
        task = asyncio.current_task()
        session_id: str | None = None
        assistant_chunks: list[str] = []
        result_text: str | None = None
        is_error = False
        errors: list[str] = []

        async for msg in query(prompt=text, options=options):
            if isinstance(msg, SystemMessage) and msg.subtype == "init":
                session_id = msg.data.get("session_id") or msg.data.get("sessionID") or session_id
                if task is not None and session_id is not None:
                    self._tasks[session_id] = task
            elif isinstance(msg, AssistantMessage):
                assistant_chunks.append(_text_from_content(msg.content))
            elif isinstance(msg, ResultMessage):
                session_id = msg.session_id or session_id
                result_text = msg.result
                is_error = msg.is_error
                if msg.errors:
                    errors = list(msg.errors)

        try:
            if is_error:
                detail = "; ".join(errors) if errors else "claude query reported an error"
                raise RuntimeError(f"claude query failed: {detail}")
            response = result_text if result_text is not None else "\n".join(
                c for c in assistant_chunks if c
            )
            if not response:
                response = "[claude returned no text response]"
            if session_id is None:
                raise RuntimeError("claude query did not return a session_id")
            return response, session_id
        finally:
            if session_id is not None:
                self._tasks.pop(session_id, None)

    # ----------------------------------------------------------------- Backend

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
    ) -> tuple[str, str]:
        options = self._options(read_only=read_only, directory=directory, model=model)
        return await self._run(text, options)

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
    ) -> str:
        options = self._options(
            read_only=read_only, directory=directory, model=None, resume=session_id
        )
        response, _ = await self._run(text, options)
        return response

    async def abort(self, session_id: str) -> None:
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            return
        # Session already finished or unknown: best-effort CLI stop.
        await self._cli_stop(session_id)

    async def _cli_stop(self, session_id: str) -> None:
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude", "stop", session_id,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (OSError, asyncio.TimeoutError):
            # Best-effort: a failed stop is not fatal for an already-finished session.
            pass
