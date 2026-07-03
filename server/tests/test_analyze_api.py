"""Request-validation tests for the analyze endpoint."""

from __future__ import annotations

import httpx


async def test_model_path_field_rejected(app):
    """modelPath was removed from the API: clients must not choose which
    file the server torch.loads. extra="forbid" turns it into a 422."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/analyze/simulate",
            json={"seed": 1, "modelPath": "/etc/passwd"},
        )
    assert resp.status_code == 422
