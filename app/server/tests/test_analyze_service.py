"""Service-level tests for the analyze simulation."""

from __future__ import annotations

import pytest

from server.api.schemas import AnalyzeSimulateRequest
from server.config import get_settings


@pytest.fixture
def analyze_env(monkeypatch, tmp_path):
    """Hermetic settings for direct service calls (no app, no checkpoint):
    load_agent is patched per-test, but get_settings still needs a model
    path that exists."""
    model_file = tmp_path / "model.pt"
    model_file.write_bytes(b"stub checkpoint")
    monkeypatch.setenv("SHEEPSHEAD_MODEL_PATH", str(model_file))
    monkeypatch.setenv("SHEEPSHEAD_MODEL_LABEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql://test:test@127.0.0.1:1/test")
    monkeypatch.setenv("ENV", "development")
    get_settings.cache_clear()
    try:
        yield
    finally:
        get_settings.cache_clear()


def test_simulate_with_no_aux_critic(analyze_env, monkeypatch):
    """Architectures without auxiliary critic heads (e.g. "no-aux") must
    simulate cleanly with every aux-derived field left None, instead of
    crashing on the missing critic_adapter."""
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.analyze as analyze_mod

    agent = PPOAgent(len(ACTIONS), arch="no-aux")
    monkeypatch.setattr(analyze_mod, "load_agent", lambda path: agent)

    resp = analyze_mod.simulate_game(
        AnalyzeSimulateRequest(seed=7, deterministic=True)
    )

    assert resp.trace, "expected at least one simulated decision"
    for step in resp.trace:
        assert step.winProb is None
        assert step.expectedFinalReturn is None
        assert step.secretPartnerProb is None
        assert step.pointEstimates is None
        assert step.pointActuals is None
        assert step.trumpSeenMask is None
        assert step.unseenTrumpHigherThanHandProb is None
        assert step.unseenTrumpHigherThanHandActual is None
        # Non-aux essentials still populated
        assert step.probabilities
        assert step.valueEstimate == step.valueEstimate  # not NaN
