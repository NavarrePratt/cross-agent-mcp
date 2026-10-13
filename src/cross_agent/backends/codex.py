"""Codex backend: drives ``codex exec --json`` as a subprocess.

Codex CLI 0.154.0 removed ``codex mcp-server``, so this backend uses the
non-interactive exec CLI instead. Auth comes from the Codex CLI's stored login.
"""

from __future__ import annotations

import asyncio
import json
import os
from collections import deque

from cross_agent.config import env_float, validate_timeout

DEFAULT_QUERY_TIMEOUT = 1200.0
# Agent messages arrive as single JSONL lines and can exceed asyncio's 64 KiB default.
STREAM_LIMIT = 16 * 1024 * 1024
STDERR_TAIL_LINES = 20
TERMINATE_GRACE = 5.0


async def _terminate(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
        await asyncio.wait_for(proc.wait(), timeout=TERMINATE_GRACE)
    except ProcessLookupError:
        pass
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()


async def _collect_lines(stream: asyncio.StreamReader, tail: deque[str]) -> None:
    async for line in stream:
        tail.append(line.decode(errors="replace").rstrip())


class CodexBackend:
    """Backend wrapping the ``codex exec`` CLI for cross-family conversations."""

    def __init__(self, model: str | None = None, binary: str | None = None) -> None:
        self.binary = binary or os.environ.get("CROSS_AGENT_CODEX_BIN", "codex")
        self._model = model or os.environ.get("CROSS_AGENT_CODEX_MODEL")
        self._reasoning_effort = os.environ.get("CROSS_AGENT_CODEX_REASONING_EFFORT")
        self._query_timeout = env_float("CROSS_AGENT_QUERY_TIMEOUT", DEFAULT_QUERY_TIMEOUT)
        # session_id -> running ``codex exec`` process, for abort support.
        self._procs: dict[str, asyncio.subprocess.Process] = {}
        # Process-local serialization for MCP-originated calls to the same native
        # Codex thread. Other Codex clients may still use the thread concurrently.
        self._session_locks: dict[str, asyncio.Lock] = {}

    @property
    def name(self) -> str:
        return "codex"

    def _args(
        self,
        *,
        read_only: bool,
        directory: str | None,
        model: str | None,
        resume: str | None = None,
    ) -> list[str]:
        # Nobody is present to answer approval prompts in a headless call.
        args = [self.binary, "exec", "--json", "--skip-git-repo-check"]
        args += ["-c", 'approval_policy="never"']
        if read_only:
            # User config can enable MCP servers (including this one) and plugins. A
            # ``-c mcp_servers={}`` override merges into that table instead of
            # replacing it, so skip user config entirely.
            args += ["--ignore-user-config", "--sandbox=read-only"]
        if directory:
            args.append(f"--cd={directory}")
        chosen = model or self._model
        if chosen:
            args.append(f"--model={chosen}")
        if self._reasoning_effort:
            args += ["-c", f"model_reasoning_effort={json.dumps(self._reasoning_effort)}"]
        if resume is not None:
            if resume.startswith("-"):
                raise ValueError(f"invalid codex session id: {resume!r}")
            args += ["resume", resume]
        # Read the prompt from stdin so long or dash-prefixed prompts stay out of argv.
        args.append("-")
        return args

    async def _run(
        self,
        args: list[str],
        text: str,
        *,
        timeout: float | None,
        session_id_hint: str | None = None,
    ) -> tuple[str, str]:
        effective_timeout = validate_timeout(timeout, self._query_timeout)
        proc = await asyncio.create_subprocess_exec(
            *args,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=STREAM_LIMIT,
        )
        session_ids: set[str] = {session_id_hint} if session_id_hint else set()
        for sid in session_ids:
            self._procs[sid] = proc
        stderr_tail: deque[str] = deque(maxlen=STDERR_TAIL_LINES)
        stderr_task = asyncio.create_task(_collect_lines(proc.stderr, stderr_tail))
        try:
            proc.stdin.write(text.encode())
            await proc.stdin.drain()
            proc.stdin.close()
            try:
                messages, error, session_id = await asyncio.wait_for(
                    self._read_events(proc, session_ids, session_id_hint),
                    timeout=effective_timeout,
                )
            except asyncio.TimeoutError:
                sid = next(iter(session_ids), "unknown")
                raise TimeoutError(
                    f"codex exec timed out after {effective_timeout:g}s (session {sid})"
                ) from None
            await proc.wait()
            await stderr_task
        finally:
            await _terminate(proc)
            stderr_task.cancel()
            for sid in session_ids:
                if self._procs.get(sid) is proc:
                    self._procs.pop(sid, None)

        where = f" (session {session_id})" if session_id else ""
        if error:
            raise RuntimeError(f"codex turn failed{where}: {error}")
        if proc.returncode != 0:
            detail = "\n".join(stderr_tail) or "no stderr output"
            raise RuntimeError(f"codex exec exited with code {proc.returncode}{where}: {detail}")
        if session_id is None:
            raise RuntimeError("codex exec did not report a thread id")
        # Earlier agent messages are progress commentary; the last one is the answer.
        response = messages[-1] if messages else "[codex returned no text response]"
        return response, session_id

    async def _read_events(
        self,
        proc: asyncio.subprocess.Process,
        session_ids: set[str],
        session_id: str | None,
    ) -> tuple[list[str], str | None, str | None]:
        messages: list[str] = []
        error: str | None = None
        async for line in proc.stdout:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            etype = event.get("type")
            if etype == "thread.started":
                session_id = event.get("thread_id") or session_id
                if session_id:
                    session_ids.add(session_id)
                    self._procs[session_id] = proc
            elif etype == "item.completed":
                # ``error`` items are non-fatal warnings; fatal errors arrive as
                # ``turn.failed`` or top-level ``error`` events.
                item = event.get("item") or {}
                if item.get("type") == "agent_message" and item.get("text"):
                    messages.append(item["text"])
            elif etype == "turn.failed":
                error = (event.get("error") or {}).get("message") or "turn failed"
            elif etype == "error":
                error = event.get("message") or "codex reported an error"
        return messages, error, session_id

    # ----------------------------------------------------------------- Backend

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        args = self._args(read_only=read_only, directory=directory, model=model)
        return await self._run(args, text, timeout=timeout)

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        args = self._args(read_only=read_only, directory=directory, model=None, resume=session_id)
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            response, _ = await self._run(args, text, timeout=timeout, session_id_hint=session_id)
        return response

    async def abort(self, session_id: str) -> None:
        proc = self._procs.get(session_id)
        if proc is not None:
            await _terminate(proc)
