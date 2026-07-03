"""Unit tests for the bearer-token auth dependency (no real DB)."""

from __future__ import annotations

import uuid

import pytest
from fastapi import HTTPException

from server.api import auth


class _Request:
    def __init__(self, authorization: str | None):
        self.headers = {}
        if authorization is not None:
            self.headers["authorization"] = authorization


@pytest.fixture(autouse=True)
def _clean_cache():
    auth.clear_cache()
    yield
    auth.clear_cache()


async def test_missing_header_is_401():
    with pytest.raises(HTTPException) as exc:
        await auth.current_player(_Request(None))
    assert exc.value.status_code == 401
    assert exc.value.detail == "missing_token"


async def test_non_bearer_header_is_401():
    with pytest.raises(HTTPException) as exc:
        await auth.current_player(_Request("Basic dXNlcjpwdw=="))
    assert exc.value.status_code == 401


async def test_unknown_token_is_401(monkeypatch):
    async def no_session(pool, token):
        return None

    monkeypatch.setattr(auth.sessions_db, "resolve_token", no_session)
    monkeypatch.setattr(auth, "get_db_pool", lambda: object())
    with pytest.raises(HTTPException) as exc:
        await auth.current_player(_Request("Bearer nope"))
    assert exc.value.status_code == 401
    assert exc.value.detail == "invalid_token"


async def test_valid_token_resolves_and_caches(monkeypatch):
    player_id = uuid.uuid4()
    calls = {"n": 0}

    async def one_session(pool, token):
        calls["n"] += 1
        return player_id

    monkeypatch.setattr(auth.sessions_db, "resolve_token", one_session)
    monkeypatch.setattr(auth, "get_db_pool", lambda: object())

    first = await auth.current_player(_Request("Bearer tok"))
    second = await auth.current_player(_Request("Bearer tok"))
    assert first.id == second.id == player_id
    assert calls["n"] == 1  # second hit served from cache

    assert await auth.optional_player(_Request(None)) is None
