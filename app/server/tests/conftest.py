"""Shared fixtures for the product/server test suite (the deployed FastAPI
app). Training/research tests for the sheepshead package live in
sheepshead/tests.

Tests must be hermetic: no Postgres, no model checkpoint on disk beyond a
placeholder file. The exception is ``db_app``, for the opt-in tests that
run against TEST_DATABASE_URL. ``create_app`` validates that the model path exists and
eagerly loads the agent, so the fixture provides a stub file and patches
``load_agent`` before building the app.
"""

from __future__ import annotations

import os

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
    # Never send real email from tests, whatever the developer's .env says.
    monkeypatch.setenv("RESEND_API_KEY", "")
    get_settings.cache_clear()

    import server.app as app_module
    from server.api.ratelimit import limiter
    from server.runtime.manager import tables

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


class StubAgent:
    """Deterministic stand-in for PPOAgent: always the lowest valid action."""

    def act(self, state, valid_actions=None, player_id=None, deterministic=False):
        assert valid_actions is not None, "the server always passes valid actions"
        return (sorted(valid_actions)[0], None, None)

    def observe(self, *args, **kwargs):
        pass

    def reset_recurrent_state(self):
        pass


@pytest.fixture
async def db_app(app, monkeypatch):
    """The hermetic app fixture, but with a live pool wired to TEST_DATABASE_URL
    (httpx's ASGITransport does not run the lifespan, so do its DB work here)."""
    import server.app as app_module
    import server.runtime.dealing as dealing_module
    from server.services.persistence.pool import (
        close_pool,
        open_pool,
        set_db_state,
    )

    monkeypatch.setattr(dealing_module, "load_agent", lambda path: StubAgent())

    pool = await open_pool(os.environ["TEST_DATABASE_URL"])
    async with pool.acquire() as conn:
        ai_model_id = await app_module._upsert_ai_model(conn, "test-model")
        ai_player_id = await app_module._upsert_ai_player(conn, ai_model_id)
    set_db_state(pool, ai_player_id)
    try:
        yield app, pool
    finally:
        # Cancel background tasks owned by tables created in this test.
        from server.runtime.manager import tables

        for table in list(tables.tables.values()):
            for task in (table.ai_task, table.autoclose_task):
                if task and not task.done():
                    task.cancel()
        tables.tables.clear()
        await close_pool()
