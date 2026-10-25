"""Deployment sampling: settings -> SamplingSpec -> table agent sampler, and
the (model, sampling) identity row in ai_player."""

from __future__ import annotations

import os
from typing import Any

import pytest

import server.runtime.dealing as dealing_module
from server.config import Settings
from server.runtime.dealing import build_table_agent
from server.services.ai_sampling import SamplingSpec
from sheepshead.agent.sampling import DeploySampling


class _StubAgent:
    deploy_sampling = None

    def act(self, *args, **kwargs):
        raise AssertionError("not called")


def _settings(**overrides: Any) -> Settings:
    return Settings(
        sheepshead_model_path="unused.pt",
        sheepshead_model_label="test",
        database_url="postgresql://unused",
        **overrides,
    )


def test_default_settings_are_argmax():
    spec = SamplingSpec.from_settings(_settings())
    assert spec.is_argmax
    assert spec.columns() == (0.0, 0.0, 0.0, 0.0, None)
    assert spec.sampler() is None
    assert spec.label() == "argmax"


def test_tempered_settings_build_a_sampler_with_fresh_seeds():
    spec = SamplingSpec.from_settings(
        _settings(sheepshead_temperature_play=0.5, sheepshead_logit_band=1.0)
    )
    assert not spec.is_argmax
    assert spec.columns() == (0.0, 0.0, 0.0, 0.5, 1.0)
    a, b = spec.sampler(), spec.sampler()
    assert isinstance(a, DeploySampling) and isinstance(b, DeploySampling)
    assert a.temperature == {"pick": 0.0, "partner": 0.0, "bury": 0.0, "play": 0.5}
    assert a.band == 1.0
    assert a.seed != b.seed  # two tables never share a uniform stream
    assert spec.label() == "play=0.5 band=1"


def test_negative_values_are_rejected():
    with pytest.raises(ValueError):
        SamplingSpec.from_settings(_settings(sheepshead_temperature_pick=-0.1))
    with pytest.raises(ValueError):
        SamplingSpec.from_settings(_settings(sheepshead_logit_band=-1.0))


def test_table_agent_gets_the_sampler_before_wrapping(monkeypatch):
    stub = _StubAgent()
    monkeypatch.setattr(dealing_module, "load_agent", lambda path: stub)
    agent = build_table_agent(_settings(), table_id="t1")
    assert agent is stub and stub.deploy_sampling is None

    agent = build_table_agent(
        _settings(sheepshead_temperature_play=0.75, sheepshead_convention_wrap="c1"),
        table_id="t2",
    )
    # The sampler sits on the raw agent under the convention wrapper, which
    # delegates attribute reads but not writes.
    assert agent is not stub
    assert isinstance(stub.deploy_sampling, DeploySampling)
    assert stub.deploy_sampling.temperature["play"] == 0.75
    assert agent.deploy_sampling is stub.deploy_sampling


@pytest.mark.skipif(
    "TEST_DATABASE_URL" not in os.environ, reason="needs TEST_DATABASE_URL"
)
async def test_ai_player_identity_is_model_plus_sampling():
    import server.app as app_module
    from server.services.persistence.pool import close_pool, open_pool

    pool = await open_pool(os.environ["TEST_DATABASE_URL"])
    try:
        async with pool.acquire() as conn:
            model_id = await app_module._upsert_ai_model(conn, "test-model-sampling")
            argmax = SamplingSpec.from_settings(_settings())
            warm = SamplingSpec.from_settings(
                _settings(sheepshead_temperature_play=0.5, sheepshead_logit_band=1.0)
            )
            a1 = await app_module._upsert_ai_player(conn, model_id, argmax)
            a2 = await app_module._upsert_ai_player(conn, model_id, argmax)
            w1 = await app_module._upsert_ai_player(conn, model_id, warm)
            w2 = await app_module._upsert_ai_player(conn, model_id, warm)
            assert a1 == a2 and w1 == w2 and a1 != w1
            row = await conn.fetchrow(
                "SELECT temperature_play, logit_band FROM ai_player WHERE ai_player_id = $1",
                w1,
            )
            assert row["temperature_play"] == 0.5 and row["logit_band"] == 1.0
            await conn.execute(
                "DELETE FROM ai_player WHERE ai_player_id = ANY($1::bigint[])", [a1, w1]
            )
            await conn.execute("DELETE FROM ai_model WHERE ai_model_id = $1", model_id)
    finally:
        await close_pool()
