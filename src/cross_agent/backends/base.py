from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """A coding-agent backend exposed through the cross-agent MCP server."""

    @property
    def name(self) -> str:
        """Backend identifier used as the tool name prefix (e.g. 'opencode', 'claude')."""
        ...

    async def prompt(
        self,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ) -> tuple[str, str]:
        """Start a backend-native session. Returns ``(response_text, session_id)``.

        The returned ``session_id`` is the native backend/harness id. Pass it to
        :meth:`continue_session` to resume the same native conversation.

        ``timeout`` overrides the backend's default query timeout (seconds).
        ``None`` means use the configured default (see each backend).
        """
        ...

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
        timeout: float | None = None,
    ) -> str:
        """Continue an existing backend-native session. Returns response text.

        The session may have been created by this MCP server or copied from the
        backend harness directly.
        """
        ...

    async def abort(self, session_id: str) -> None:
        """Abort a running session. No-op if the session already finished."""
        ...
