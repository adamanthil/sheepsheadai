"""Per-IP rate limits on mutating endpoints."""

from __future__ import annotations

import httpx

from server.config import get_settings


async def test_create_table_rate_limited(app, monkeypatch):
    # Measure the rate limit alone, not the per-IP open-table cap.
    monkeypatch.setenv("SHEEPSHEAD_MAX_TABLES_PER_IP", "100")
    get_settings.cache_clear()
    transport = httpx.ASGITransport(app=app, client=("203.0.113.7", 12345))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        statuses = []
        for i in range(11):
            resp = await client.post("/api/tables", json={"name": f"t{i}"})
            statuses.append(resp.status_code)
    assert statuses[:10] == [200] * 10
    assert statuses[10] == 429
