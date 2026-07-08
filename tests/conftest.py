"""Shared pytest configuration and fixtures."""

from __future__ import annotations

import httpx
import pytest

from cross_agent.backends.opencode import OpenCodeBackend, OpenCodeConfig


@pytest.fixture
def captured_requests() -> list[httpx.Request]:
    """Record every request the backend makes through its httpx client."""
    return []


@pytest.fixture
async def make_backend(captured_requests):
    """Build an OpenCodeBackend wired to an in-process MockTransport.

    Each test passes a handler that maps incoming (non-health) requests to
    responses. The transport also handles ``GET /global/health``: 200 when
    ``healthy=True`` (the default), or a connection error when ``healthy=False``
    so auto-start logic can be exercised. The backend is constructed with
    ``autoserve=False`` by default so it never spawns a real process unless a
    test explicitly opts in.
    """
    created: list[OpenCodeBackend] = []

    def _make(
        handler,
        *,
        base_url: str = "http://127.0.0.1:4096",
        config: OpenCodeConfig | None = None,
        healthy: bool = True,
    ) -> OpenCodeBackend:
        def wrapped(request: httpx.Request) -> httpx.Response:
            captured_requests.append(request)
            if request.url.path == "/global/health":
                if healthy:
                    return httpx.Response(200, json={"healthy": True})
                raise httpx.ConnectError("server down")
            return handler(request)

        cfg = config or OpenCodeConfig(base_url=base_url, autoserve=False)
        # Tests never spawn a real opencode process; force autoserve off even
        # when a custom config is passed in.
        cfg.autoserve = False
        backend = OpenCodeBackend(config=cfg)
        transport = httpx.MockTransport(wrapped)
        backend._client = httpx.AsyncClient(
            base_url=cfg.base_url,
            headers={"content-type": "application/json"},
            transport=transport,
            timeout=httpx.Timeout(cfg.query_timeout, connect=5.0),
        )
        created.append(backend)
        return backend

    yield _make
    for backend in created:
        await backend.aclose()
