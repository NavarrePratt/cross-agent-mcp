"""OpenCode backend: talks to a local ``opencode serve`` HTTP API over httpx."""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urlsplit

import httpx

from cross_agent.config import env_float, validate_timeout

DEFAULT_PORT = 4096
DEFAULT_BASE_URL = f"http://127.0.0.1:{DEFAULT_PORT}"
# When non-empty, read-only prompts request this opencode agent (which must be
# defined in opencode.json with read-only permissions). Empty string = do not
# send an agent, falling back to opencode's default "build" agent. Read-only
# enforcement then relies on the caller's prompt, not opencode-level sandboxing.
DEFAULT_AGENT = ""
DEFAULT_QUERY_TIMEOUT = 600.0
HEALTH_TIMEOUT = 3.0
AUTOSERVE_WAIT = 10.0
AUTOSERVE_POLL = 0.5

# Secret patterns scrubbed from response text before returning to the MCP client.
# Mirrors the redactSecrets pattern from opencode-mcp to keep provider keys out
# of LLM-visible tool output.
_SECRET_PATTERNS = [
    # OpenAI / Anthropic / generic ``sk-`` keys
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    # GitHub personal access tokens (classic + fine-grained)
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{22,}"),
    # HuggingFace tokens
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    # Slack tokens
    re.compile(r"xox[abp]-[A-Za-z0-9\-]{10,}"),
    # Stripe / webhook signing secrets
    re.compile(r"whsec_[A-Za-z0-9]{20,}"),
    re.compile(r"sk_live_[A-Za-z0-9]{20,}"),
    # Long hex / base64 token blobs (64+ chars) - catches raw API keys
    re.compile(r"\b[A-Fa-f0-9]{64}\b"),
    re.compile(r"\b[A-Za-z0-9+/]{86}={0,2}\b"),
]

# Characters that must never reach an HTTP header value (CRLF / NUL injection).
_HEADER_STRIP = re.compile(r"[\r\n\0]")


@dataclass
class OpenCodeConfig:
    base_url: str = DEFAULT_BASE_URL
    autoserve: bool = True
    agent: str = DEFAULT_AGENT
    username: str | None = None
    password: str | None = None
    query_timeout: float = DEFAULT_QUERY_TIMEOUT

    @classmethod
    def from_env(cls) -> OpenCodeConfig:
        def env_bool(name: str, default: bool) -> bool:
            val = os.environ.get(name)
            if val is None:
                return default
            return val.strip().lower() in ("1", "true", "yes", "on")

        return cls(
            base_url=os.environ.get("CROSS_AGENT_OPENCODE_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            autoserve=env_bool("CROSS_AGENT_OPENCODE_AUTOSERVE", True),
            agent=os.environ.get("CROSS_AGENT_OPENCODE_AGENT", DEFAULT_AGENT),
            username=os.environ.get("OPENCODE_SERVER_USERNAME"),
            password=os.environ.get("OPENCODE_SERVER_PASSWORD"),
            query_timeout=env_float("CROSS_AGENT_QUERY_TIMEOUT", DEFAULT_QUERY_TIMEOUT),
        )


def _sanitize_directory(directory: str | None) -> str | None:
    """Strip NUL/CR/LF from a directory before it lands in an HTTP header."""
    if directory is None:
        return None
    cleaned = _HEADER_STRIP.sub("", directory)
    return cleaned or None


def _redact(text: str) -> str:
    """Scrub API-key-like patterns from ``text``."""
    redacted = text
    for pattern in _SECRET_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _parse_model(model: str | None) -> dict[str, str] | None:
    """Split a flat ``provider/model`` string into the opencode API model object.

    opencode expects ``{"providerID": ..., "modelID": ...}``. A flat string like
    ``wandb/zai-org/GLM-5.2`` splits on the first ``/`` so the modelID may itself
    contain slashes.
    """
    if not model:
        return None
    if "/" not in model:
        return {"providerID": model, "modelID": model}
    provider, _, model_id = model.partition("/")
    return {"providerID": provider, "modelID": model_id}


def _extract_text(parts: list[dict[str, Any]]) -> str:
    """Extract assistant text from an opencode message response.

    opencode message parts are a heterogeneous list: ``step-start``, ``text``,
    ``step-finish``, tool calls, etc. Only ``type: "text"`` parts carry the
    assistant's prose; concatenate them in order.
    """
    chunks: list[str] = []
    for part in parts or []:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                chunks.append(text)
    return "\n".join(chunks)


class OpenCodeBackend:
    """Backend that drives a local ``opencode serve`` instance over HTTP."""

    def __init__(self, config: OpenCodeConfig | None = None) -> None:
        self.config = config or OpenCodeConfig.from_env()
        self._proc: subprocess.Popen[bytes] | None = None
        self._spawned = False
        self._client: httpx.AsyncClient | None = None
        # Serializes the health-check-then-spawn in _ensure_server so concurrent
        # callers cannot each spawn their own ``opencode serve`` process.
        self._server_lock = asyncio.Lock()
        # Native opencode session id -> running MCP task, for abort support.
        self._tasks: dict[str, set[asyncio.Task[Any]]] = {}
        # Process-local serialization for MCP-originated calls to the same native
        # session. Other opencode clients may still use the session concurrently.
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._session_lock_users: dict[str, int] = {}

    @property
    def name(self) -> str:
        return "opencode"

    # ------------------------------------------------------------------ client

    def _auth(self) -> httpx.Auth | None:
        if self.config.username or self.config.password:
            return httpx.BasicAuth(self.config.username or "", self.config.password or "")
        return None

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                auth=self._auth(),
                headers={"content-type": "application/json"},
                timeout=httpx.Timeout(self.config.query_timeout, connect=5.0),
            )
        return self._client

    async def _ensure_server(self) -> None:
        """Health-check the opencode server; auto-start it if down and enabled."""
        if await self._is_healthy():
            return
        async with self._server_lock:
            # Double-checked locking: another caller may have started the server
            # while we were waiting for the lock.
            if await self._is_healthy():
                return
            if not self.config.autoserve:
                raise RuntimeError(
                    f"opencode server not reachable at {self.config.base_url} "
                    "and CROSS_AGENT_OPENCODE_AUTOSERVE is false."
                )
            await self._start_server()
            if not await self._wait_healthy():
                raise RuntimeError(
                    f"opencode server did not become healthy at {self.config.base_url} "
                    f"within {AUTOSERVE_WAIT:.0f}s."
                )

    async def _is_healthy(self) -> bool:
        client = await self._ensure_client()
        try:
            resp = await client.get("/global/health", timeout=HEALTH_TIMEOUT)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    async def _wait_healthy(self) -> bool:
        deadline = time.monotonic() + AUTOSERVE_WAIT
        while time.monotonic() < deadline:
            if await self._is_healthy():
                return True
            await asyncio.sleep(AUTOSERVE_POLL)
        return False

    async def _start_server(self) -> None:
        """Spawn ``opencode serve --port <port>`` as a subprocess we own."""
        port = self._port_from_url()
        cmd = ["opencode", "serve", "--port", str(port)]
        self._proc = subprocess.Popen(  # noqa: S603 - trusted local binary
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._spawned = True

    def _port_from_url(self) -> int:
        parsed = urlsplit(self.config.base_url)
        if parsed.port is not None:
            return parsed.port
        raise RuntimeError(
            "CROSS_AGENT_OPENCODE_BASE_URL must include an explicit port "
            "when CROSS_AGENT_OPENCODE_AUTOSERVE is true."
        )

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: Any = httpx.USE_CLIENT_DEFAULT,
    ) -> httpx.Response:
        client = await self._ensure_client()
        return await client.request(method, path, json=json, headers=headers, timeout=timeout)

    # ----------------------------------------------------------------- helpers

    def _dir_headers(self, directory: str | None) -> dict[str, str]:
        sanitized = _sanitize_directory(directory)
        return {"x-opencode-directory": sanitized} if sanitized else {}

    def _session_path(self, session_id: str, suffix: str = "") -> str:
        """Build a ``/session/<id>[/suffix]`` path with the id URL-encoded.

        Encoding with ``safe=""`` ensures a slash or ``?`` in the id cannot
        retarget the request to a different endpoint.
        """
        return f"/session/{quote(session_id, safe='')}{suffix}"

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

    def _track_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        self._tasks.setdefault(session_id, set()).add(task)

    def _untrack_task(self, session_id: str, task: asyncio.Task[Any]) -> None:
        tasks = self._tasks.get(session_id)
        if tasks is None:
            return
        tasks.discard(task)
        if not tasks:
            self._tasks.pop(session_id, None)

    async def _abort_native(self, session_id: str) -> None:
        resp = await self._request(
            "POST",
            self._session_path(session_id, "/abort"),
            timeout=httpx.Timeout(HEALTH_TIMEOUT, connect=HEALTH_TIMEOUT),
        )
        # 404 means the session already finished; treat as success.
        if resp.status_code not in (200, 404):
            resp.raise_for_status()

    def _cancel_tasks(self, session_id: str) -> list[asyncio.Task[Any]]:
        tasks = list(self._tasks.get(session_id, ()))
        for task in tasks:
            if not task.done():
                task.cancel()
        return tasks

    async def _drain_cancelled_tasks(self, tasks: list[asyncio.Task[Any]]) -> None:
        if not tasks:
            return
        done, _ = await asyncio.wait(tasks, timeout=HEALTH_TIMEOUT)
        for task in done:
            if task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not task.cancelled():
                        raise
                except Exception:
                    pass

    async def _send_message(
        self,
        session_id: str,
        text: str,
        *,
        read_only: bool,
        directory: str | None,
        model: str | None,
        timeout: float | None = None,
    ) -> str:
        """POST a message to a session and return the assistant text.

        When ``read_only`` is set and ``config.agent`` is non-empty, we request
        that opencode agent (expected to be configured with read-only permissions
        in opencode.json). When ``config.agent`` is empty, no agent is sent and
        read-only enforcement relies on the caller's prompt.
        """
        task = asyncio.current_task()
        if task is not None:
            self._track_task(session_id, task)
        lock = self._begin_session_use(session_id)
        try:
            await self._ensure_server()
            async with lock:
                return await self._send_message_locked(
                    session_id,
                    text,
                    read_only=read_only,
                    directory=directory,
                    model=model,
                    timeout=timeout,
                )
        except asyncio.CancelledError:
            try:
                await self._abort_native(session_id)
            except httpx.HTTPError:
                pass
            raise
        finally:
            if task is not None:
                self._untrack_task(session_id, task)
            self._end_session_use(session_id)

    async def _send_message_locked(
        self,
        session_id: str,
        text: str,
        *,
        read_only: bool,
        directory: str | None,
        model: str | None,
        timeout: float | None = None,
    ) -> str:
        """POST a message to a session with the caller holding the session lock."""
        body: dict[str, Any] = {"parts": [{"type": "text", "text": text}]}
        if read_only and self.config.agent:
            body["agent"] = self.config.agent
        model_obj = _parse_model(model)
        if model_obj:
            body["model"] = model_obj

        dir_headers = self._dir_headers(directory)
        msg_path = self._session_path(session_id, "/message")
        effective_timeout = validate_timeout(timeout, self.config.query_timeout)
        request_timeout = httpx.Timeout(effective_timeout, connect=5.0)
        try:
            resp = await self._request(
                "POST", msg_path, json=body, headers=dir_headers, timeout=request_timeout
            )
        except httpx.TimeoutException:
            try:
                await self._abort_native(session_id)
            except httpx.HTTPError:
                pass
            raise
        resp.raise_for_status()
        return _redact(_extract_text(resp.json().get("parts", [])))

    # ----------------------------------------------------------------- Backend

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        await self._ensure_server()
        create = await self._request(
            "POST", "/session", json={"title": text[:80]}, headers=self._dir_headers(directory)
        )
        create.raise_for_status()
        session_id = create.json()["id"]
        try:
            response = await self._send_message(
                session_id,
                text,
                read_only=read_only,
                directory=directory,
                model=model,
                timeout=timeout,
            )
        except Exception as exc:
            raise RuntimeError(
                f"opencode session {session_id} was created but message failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return response, session_id

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        return await self._send_message(
            session_id,
            text,
            read_only=read_only,
            directory=directory,
            model=None,
            timeout=timeout,
        )

    async def abort(self, session_id: str) -> None:
        tasks = self._cancel_tasks(session_id)
        try:
            await self._abort_native(session_id)
        except httpx.TransportError:
            pass
        await self._drain_cancelled_tasks(tasks)
        for task in tasks:
            self._untrack_task(session_id, task)

    # -------------------------------------------------------------- lifecycle

    async def aclose(self) -> None:
        """Tear down the HTTP client and any server subprocess we spawned."""
        try:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
        finally:
            # Subprocess cleanup runs even if client close raised.
            if self._spawned and self._proc is not None and self._proc.poll() is None:
                self._proc.terminate()
                try:
                    # Blocking, but only runs during shutdown - acceptable.
                    self._proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    try:
                        self._proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            self._proc = None
            self._spawned = False

    # convenience for tests/integration streaming (not part of the Protocol)
    async def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[Any]:  # pragma: no cover
        raise NotImplementedError
