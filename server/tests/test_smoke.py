"""Smoke tests: every server module imports, and the app serves /health."""

from __future__ import annotations

import importlib
import pkgutil

import httpx
import pytest


def test_all_server_modules_import():
    import server

    failures = []
    for mod in pkgutil.walk_packages(server.__path__, prefix="server."):
        try:
            importlib.import_module(mod.name)
        except Exception as exc:  # noqa: BLE001 - collect all failures
            failures.append(f"{mod.name}: {exc!r}")
    assert not failures, "modules failed to import:\n" + "\n".join(failures)


@pytest.mark.asyncio
async def test_health_endpoint(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model"] == "test-model"
    # ASGITransport does not run the lifespan, so no pool is initialised and
    # the endpoint must degrade gracefully rather than 500.
    assert body["db"] is False
