"""The dev CORS regex must only match genuinely local origins, and LAN origins
must be reachable only when SHEEPSHEAD_CORS_ORIGINS opts them in."""

from __future__ import annotations

import re

import httpx
import pytest

from server.app import DEV_CORS_ORIGIN_REGEX, parse_cors_origins

LAN_ORIGIN = "http://andrews-macbook-pro.local:3000"


def build_app(monkeypatch, tmp_path, cors_origins: str):
    """Build a dev app with an explicit SHEEPSHEAD_CORS_ORIGINS value.

    The shared ``app`` fixture can't vary the setting, and it must be set
    explicitly here rather than inherited from the repo's .env.
    """
    from server.config import get_settings

    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"stub checkpoint")
    monkeypatch.setenv("SHEEPSHEAD_MODEL_PATH", str(model_file))
    monkeypatch.setenv("SHEEPSHEAD_MODEL_LABEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
    monkeypatch.setenv("ENV", "development")
    monkeypatch.setenv("SHEEPSHEAD_CORS_ORIGINS", cors_origins)
    get_settings.cache_clear()

    import server.app as app_module

    monkeypatch.setattr(app_module, "load_agent", lambda path: object())
    try:
        return app_module.create_app()
    finally:
        get_settings.cache_clear()


async def preflight(app, origin: str) -> httpx.Response:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.options(
            "/api/tables",
            headers={
                "Origin": origin,
                "Access-Control-Request-Method": "POST",
            },
        )


def test_dev_cors_regex_matches_local_origins():
    for origin in (
        "http://localhost:3000",
        "http://localhost",
        "https://localhost:3001",
        "http://127.0.0.1:3000",
    ):
        assert re.match(DEV_CORS_ORIGIN_REGEX, origin), origin


def test_dev_cors_regex_rejects_foreign_origins():
    for origin in (
        "http://evil.com:3000",
        "http://localhost.evil.com:3000",
        "http://evil.com/localhost:3000",
        "https://notlocalhost:3000",
    ):
        assert not re.match(DEV_CORS_ORIGIN_REGEX, origin), origin


def test_dev_cors_regex_rejects_lan_origins_by_default():
    """LAN access is opt-in via SHEEPSHEAD_CORS_ORIGINS, never on by default.

    Guards the whole point of the additive design: an unset setting must leave
    the dev posture exactly localhost-only, so a `.local` name an attacker can
    claim over mDNS is not trusted just because the dev server is running.
    """
    for origin in (
        "http://andrews-macbook-pro.local:3000",
        "http://192.168.182.197:3000",
        "http://andrews-macbook-pro:3000",
    ):
        assert not re.match(DEV_CORS_ORIGIN_REGEX, origin), origin
    assert parse_cors_origins("") == []


def test_parse_cors_origins_splits_and_strips():
    raw = "http://andrews-macbook-pro.local:3000, https://example.com ,"
    assert parse_cors_origins(raw) == [
        "http://andrews-macbook-pro.local:3000",
        "https://example.com",
    ]


@pytest.mark.asyncio
async def test_lan_origin_blocked_when_not_opted_in(monkeypatch, tmp_path):
    app = build_app(monkeypatch, tmp_path, cors_origins="")
    resp = await preflight(app, LAN_ORIGIN)
    assert "access-control-allow-origin" not in resp.headers


@pytest.mark.asyncio
async def test_lan_origin_allowed_when_opted_in(monkeypatch, tmp_path):
    app = build_app(monkeypatch, tmp_path, cors_origins=LAN_ORIGIN)
    resp = await preflight(app, LAN_ORIGIN)
    assert resp.headers.get("access-control-allow-origin") == LAN_ORIGIN


@pytest.mark.asyncio
async def test_localhost_still_allowed_when_opted_in(monkeypatch, tmp_path):
    """The extra origins are additive: naming a LAN host must not displace the
    localhost regex, or opting in would break the normal desktop workflow."""
    app = build_app(monkeypatch, tmp_path, cors_origins=LAN_ORIGIN)
    resp = await preflight(app, "http://localhost:3000")
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


@pytest.mark.asyncio
async def test_foreign_origin_blocked_when_lan_opted_in(monkeypatch, tmp_path):
    app = build_app(monkeypatch, tmp_path, cors_origins=LAN_ORIGIN)
    resp = await preflight(app, "http://evil.com")
    assert "access-control-allow-origin" not in resp.headers
