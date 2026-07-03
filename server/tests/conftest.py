"""Shared fixtures for server tests.

Tests must be hermetic: no Postgres, no model checkpoint on disk beyond a
placeholder file. ``create_app`` validates that the model path exists and
eagerly loads the agent, so the fixture provides a stub file and patches
``load_agent`` before building the app.
"""

from __future__ import annotations

import pytest

from server.config import get_settings


@pytest.fixture
def app(monkeypatch, tmp_path):
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"stub checkpoint")
    monkeypatch.setenv("SHEEPSHEAD_MODEL_PATH", str(model_file))
    monkeypatch.setenv("SHEEPSHEAD_MODEL_LABEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
    monkeypatch.setenv("ENV", "development")
    get_settings.cache_clear()

    import server.app as app_module

    monkeypatch.setattr(app_module, "load_agent", lambda path: object())
    try:
        yield app_module.create_app()
    finally:
        get_settings.cache_clear()
