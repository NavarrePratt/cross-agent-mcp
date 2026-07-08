"""Shared configuration helpers used by both backends."""

from __future__ import annotations

import math
import os


def env_float(name: str, default: float) -> float:
    """Parse a positive, finite float from env var ``name``.

    Falls back to ``default`` when the variable is unset, non-numeric, or
    non-positive/non-finite (zero, negative, ``inf``, ``nan``).
    """
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        seconds = float(val)
    except ValueError:
        return default
    if not math.isfinite(seconds) or seconds <= 0:
        return default
    return seconds


def validate_timeout(timeout: float | None, default: float) -> float:
    """Return a valid timeout (seconds) for a per-call override.

    ``None`` means "use the default". Non-positive or non-finite values
    also fall back to the default rather than causing an immediate timeout
    or an unbounded wait.
    """
    if timeout is None:
        return default
    if not math.isfinite(timeout) or timeout <= 0:
        return default
    return timeout
