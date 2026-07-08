"""Tests for shared configuration helpers."""

from __future__ import annotations

from cross_agent.config import env_float, validate_timeout


def test_env_float_parses_valid_value(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "42.5")
    assert env_float("TEST_FLOAT", 10.0) == 42.5


def test_env_float_falls_back_when_unset(monkeypatch):
    monkeypatch.delenv("TEST_FLOAT", raising=False)
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_env_float_falls_back_on_invalid(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "not-a-number")
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_env_float_rejects_zero(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "0")
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_env_float_rejects_negative(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "-5")
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_env_float_rejects_inf(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "inf")
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_env_float_rejects_nan(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "nan")
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_env_float_rejects_overflow(monkeypatch):
    monkeypatch.setenv("TEST_FLOAT", "1e309")
    assert env_float("TEST_FLOAT", 10.0) == 10.0


def test_validate_timeout_none_returns_default():
    assert validate_timeout(None, 600.0) == 600.0


def test_validate_timeout_valid_returns_override():
    assert validate_timeout(120.0, 600.0) == 120.0


def test_validate_timeout_zero_returns_default():
    assert validate_timeout(0, 600.0) == 600.0


def test_validate_timeout_negative_returns_default():
    assert validate_timeout(-1.0, 600.0) == 600.0


def test_validate_timeout_inf_returns_default():
    assert validate_timeout(float("inf"), 600.0) == 600.0


def test_validate_timeout_nan_returns_default():
    assert validate_timeout(float("nan"), 600.0) == 600.0
