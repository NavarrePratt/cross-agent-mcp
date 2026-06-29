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
    ) -> tuple[str, str]:
        """One-shot prompt. Returns ``(response_text, session_id)``.

        Stateless by default: the caller may discard ``session_id``. Pass it to
        :meth:`continue_session` to resume the conversation.
        """
        ...

    async def continue_session(
        self,
        session_id: str,
        text: str,
        read_only: bool = True,
        directory: str | None = None,
    ) -> str:
        """Continue an existing session. Returns response text."""
        ...

    async def abort(self, session_id: str) -> None:
        """Abort a running session. No-op if the session already finished."""
        ...
