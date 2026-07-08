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

from cross_agent.config import env_float, validate_timeout

# Read-only toolset: pre-approve safe operations and remove write/bash entirely so
# the model cannot mutate the working tree or shell out, even if it tries.
READ_ONLY_ALLOWED = ["Read", "Glob", "Grep", "LS", "WebSearch", "WebFetch"]
READ_ONLY_DISALLOWED = ["Write", "Edit", "MultiEdit", "NotebookEdit", "Bash"]

DEFAULT_QUERY_TIMEOUT = 600.0
TASK_DRAIN_TIMEOUT = 1.0


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
        self._query_timeout = env_float("CROSS_AGENT_QUERY_TIMEOUT", DEFAULT_QUERY_TIMEOUT)
        # session_id -> running asyncio task, for abort support.
        self._tasks: dict[str, asyncio.Task[Any]] = {}
        # Process-local serialization for MCP-originated calls to the same native
        # Claude session. Other Claude clients may still use the session concurrently.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_lock_users: dict[str, int] = {}

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

    def _session_lock(self, session_id: str) -> asyncio.Lock:
        lock = self._session_locks.get(session_id)
        if lock is None:
            lock = asyncio.Lock()
            self._session_locks[session_id] = lock
        return lock

    def _begin_session_use(self, session_id: str) -> asyncio.Lock:
        lock = self._session_lock(session_id)
        self._session_lock_users[session_id] = self._session_lock_users.get(session_id, 0) + 1
        return lock

    def _end_session_use(self, session_id: str) -> None:
        users = self._session_lock_users.get(session_id, 0) - 1
        if users > 0:
            self._session_lock_users[session_id] = users
            return
        self._session_lock_users.pop(session_id, None)
        self._session_locks.pop(session_id, None)

    def _untrack_task(self, session_ids: set[str], task: asyncio.Task[Any]) -> None:
        for sid in session_ids:
            if self._tasks.get(sid) is task:
                self._tasks.pop(sid, None)

    async def _drain_task(self, task: asyncio.Task[Any]) -> None:
        done, _ = await asyncio.wait({task}, timeout=TASK_DRAIN_TIMEOUT)
        if task not in done:
            return
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            pass

    async def _run(
        self,
        text: str,
        options: ClaudeAgentOptions,
        *,
        timeout: float | None = None,
        session_id_hint: str | None = None,
    ) -> tuple[str, str]:
        """Drive a ``query()`` to completion, tracking the task for abort.

        ``timeout`` overrides the configured default query timeout (seconds).
        The timeout wraps the entire query stream consumption.
        """
        effective_timeout = validate_timeout(timeout, self._query_timeout)
        known_session_ids: set[str] = {session_id_hint} if session_id_hint else set()
        task = asyncio.create_task(
            self._run_unbounded(text, options, known_session_ids=known_session_ids)
        )
        try:
            done, _ = await asyncio.wait({task}, timeout=effective_timeout)
        except asyncio.CancelledError:
            task.cancel()
            await self._stop_sessions(known_session_ids)
            await self._drain_task(task)
            self._untrack_task(known_session_ids, task)
            raise
        if task in done:
            return await task
        try:
            task.cancel()
            await self._stop_sessions(known_session_ids)
            await self._drain_task(task)
            self._untrack_task(known_session_ids, task)
            raise asyncio.TimeoutError()
        except asyncio.CancelledError:
            self._untrack_task(known_session_ids, task)
            raise

    async def _run_unbounded(
        self,
        text: str,
        options: ClaudeAgentOptions,
        *,
        known_session_ids: set[str],
    ) -> tuple[str, str]:
        """Drive a ``query()`` to completion, tracking the task for abort."""
        task = asyncio.current_task()
        session_id = next(iter(known_session_ids), None)
        if task is not None:
            for sid in known_session_ids:
                self._tasks[sid] = task
        assistant_chunks: list[str] = []
        result_text: str | None = None
        is_error = False
        errors: list[str] = []

        try:
            # The async for loop is inside the try block so the finally always
            # cleans up _tasks, even if query() raises after registering the task.
            async for msg in query(prompt=text, options=options):
                if isinstance(msg, SystemMessage) and msg.subtype == "init":
                    sid = msg.data.get("session_id") or msg.data.get("sessionID")
                    session_id = sid or session_id
                    # One slot per session_id: concurrent runs for the same
                    # session are a caller bug by Protocol contract.
                    if task is not None and session_id is not None:
                        known_session_ids.add(session_id)
                        self._tasks[session_id] = task
                elif isinstance(msg, AssistantMessage):
                    assistant_chunks.append(_text_from_content(msg.content))
                elif isinstance(msg, ResultMessage):
                    session_id = msg.session_id or session_id
                    if task is not None and session_id is not None:
                        known_session_ids.add(session_id)
                        self._tasks[session_id] = task
                    result_text = msg.result
                    is_error = msg.is_error
                    if msg.errors:
                        errors = list(msg.errors)

            if is_error:
                detail = "; ".join(errors) if errors else "claude query reported an error"
                raise RuntimeError(f"claude query failed: {detail}")
            response = (
                result_text
                if result_text is not None
                else "\n".join(c for c in assistant_chunks if c)
            )
            if not response:
                response = "[claude returned no text response]"
            if session_id is None:
                raise RuntimeError("claude query did not return a session_id")
            return response, session_id
        finally:
            if task is not None:
                self._untrack_task(known_session_ids, task)

    async def _stop_sessions(self, session_ids: set[str]) -> None:
        for session_id in session_ids:
            await self._cli_stop(session_id)

    # ----------------------------------------------------------------- Backend

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        options = self._options(read_only=read_only, directory=directory, model=model)
        return await self._run(text, options, timeout=timeout)

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        options = self._options(
            read_only=read_only, directory=directory, model=None, resume=session_id
        )
        lock = self._begin_session_use(session_id)
        try:
            async with lock:
                response, _ = await self._run(
                    text, options, timeout=timeout, session_id_hint=session_id
                )
            return response
        finally:
            self._end_session_use(session_id)

    async def abort(self, session_id: str) -> None:
        task = self._tasks.get(session_id)
        if task is not None and not task.done():
            task.cancel()
            await self._cli_stop(session_id)
            await self._drain_task(task)
            if self._tasks.get(session_id) is task:
                self._tasks.pop(session_id, None)
            return
        # Session already finished or unknown: best-effort CLI stop.
        await self._cli_stop(session_id)

    async def _cli_stop(self, session_id: str) -> None:
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(
                "claude",
                "stop",
                session_id,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            await asyncio.wait_for(proc.wait(), timeout=10.0)
        except (OSError, asyncio.TimeoutError):
            # Best-effort: a failed stop is not fatal for an already-finished session.
            if proc is not None and proc.returncode is None:
                proc.kill()
                await proc.wait()
