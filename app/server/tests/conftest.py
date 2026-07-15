"""Shared fixtures for the product/server test suite (the deployed FastAPI
app). Training/research tests for the sheepshead package live in
sheepshead/tests.

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
    from server.api.ratelimit import limiter
    from server.runtime.tables import tables

    monkeypatch.setattr(app_module, "load_agent", lambda path: object())
    # The limiter and TableManager are module-level state; reset so earlier
    # tests' requests and tables don't leak into this test.
    limiter.reset()
    tables.tables.clear()

    from server.api import auth

    auth.clear_cache()
    try:
        yield app_module.create_app()
    finally:
        get_settings.cache_clear()
