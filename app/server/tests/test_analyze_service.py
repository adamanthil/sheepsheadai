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
        assert step.observation.hand
        assert step.memoryNorm is not None

    # Memory drift: a None distance can only happen on a seat's first
    # decision (zero memory going in); afterwards the seat's own previous
    # encode guarantees non-zero memory, so a distance is always reported.
    seen_seats: set[int] = set()
    for step in resp.trace:
        if step.seat in seen_seats:
            assert step.memoryCosineDistance is not None
        seen_seats.add(step.seat)
    # The very first decision of the game always starts from zero memory.
    assert resp.trace[0].memoryCosineDistance is None

    # No aux heads -> no calibration rollup.
    assert resp.calibration is None


def test_simulate_calibration_summary(analyze_env, monkeypatch):
    """With aux heads, a finished game gets a calibration rollup covering
    every seat that acted, with sane metric ranges."""
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.analyze as analyze_mod

    agent = PPOAgent(len(ACTIONS), arch="full")
    monkeypatch.setattr(analyze_mod, "load_agent", lambda path: agent)

    resp = analyze_mod.simulate_game(
        AnalyzeSimulateRequest(seed=3, deterministic=True)
    )

    cal = resp.calibration
    assert cal is not None
    acting_seats = {step.seat for step in resp.trace}
    assert {s.seat for s in cal.seats} == acting_seats
    assert 0.0 <= cal.overallBrier <= 1.0
    assert cal.overallPointsMae >= 0.0
    assert cal.trumpMaskCount > 0
    assert cal.trumpMaskAccuracy is not None
    assert 0.0 <= cal.trumpMaskAccuracy <= 1.0
    for seat_cal in cal.seats:
        assert 0.0 <= seat_cal.brierScore <= 1.0
        assert 0.0 <= seat_cal.meanWinProb <= 1.0
        assert seat_cal.decisionCount > 0


def test_simulate_with_oracle_critic(analyze_env, monkeypatch):
    """An oracle-mode agent must report a privileged value on every
    decision; a limited agent must not (covered by the no-aux test's
    schema default)."""
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.analyze as analyze_mod

    agent = PPOAgent(len(ACTIONS), critic_mode="oracle")
    monkeypatch.setattr(analyze_mod, "load_agent", lambda path: agent)

    resp = analyze_mod.simulate_game(
        AnalyzeSimulateRequest(seed=11, deterministic=True)
    )

    assert resp.meta["hasOracle"] is True
    assert resp.meta["criticMode"] == "oracle"
    assert resp.trace
    for step in resp.trace:
        assert step.oracleValue is not None
        assert step.oracleValue == step.oracleValue  # not NaN


def test_simulate_with_perceiver_shared_v2(analyze_env, monkeypatch):
    """Token-readout critics source their aux features from a cross-attention
    readout, not critic_adapter(features) — the simulate path must go through
    the critic's _aux_features_single seam so perceiver archs get real
    trump-tracking numbers instead of the vestigial memory vector."""
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.analyze as analyze_mod

    agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2")
    monkeypatch.setattr(analyze_mod, "load_agent", lambda path: agent)

    resp = analyze_mod.simulate_game(
        AnalyzeSimulateRequest(seed=13, deterministic=True)
    )

    assert resp.trace
    for step in resp.trace:
        assert step.winProb is not None
        assert step.trumpSeenMask is not None
        assert step.unseenTrumpHigherThanHandProb is not None
        assert step.probabilities
    assert resp.calibration is not None


def test_pick_with_perceiver_shared_v2(analyze_env, monkeypatch):
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.pick_analysis as pick_mod
    from server.api.schemas import AnalyzePickRequest

    agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2")
    monkeypatch.setattr(pick_mod, "load_agent", lambda path: agent)

    resp = pick_mod.analyze_pick(AnalyzePickRequest(seat=2, seed=3))
    assert resp.decisions
    assert resp.decisions[0].seat == 2
    assert resp.decisions[0].phase == "pick"
    assert resp.decisions[0].winProb is not None


def test_model_info_with_perceiver_shared_v2(analyze_env, monkeypatch):
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.model_info as model_info_mod

    agent = PPOAgent(len(ACTIONS), arch="perceiver-shared-v2")
    monkeypatch.setattr(model_info_mod, "load_agent", lambda path: agent)
    model_info_mod._model_info.cache_clear()

    info = model_info_mod.get_model_info()
    assert info.arch == "perceiver-shared-v2"
    assert info.hasAuxHeads is True
    assert info.cardEmbeddings is not None
    assert len(info.cardEmbeddings.cards) == 33


def test_model_info_card_embeddings(analyze_env, monkeypatch):
    """The model-info payload must describe the full card table (32 cards +
    UNDER, pad row dropped) with consistent geometry shapes."""
    from sheepshead import ACTIONS, DECK
    from sheepshead.agent.ppo import PPOAgent

    import server.services.model_info as model_info_mod

    agent = PPOAgent(len(ACTIONS), arch="full")
    monkeypatch.setattr(model_info_mod, "load_agent", lambda path: agent)
    model_info_mod._model_info.cache_clear()

    info = model_info_mod.get_model_info()

    assert info.arch == "full"
    assert info.hasAuxHeads is True
    assert info.hasOracle is False
    emb = info.cardEmbeddings
    assert emb is not None
    assert emb.dims == 16
    assert len(emb.cards) == 33
    assert [e.card for e in emb.cards[:32]] == DECK
    assert emb.cards[32].card == "UNDER"
    assert len(emb.cosineSim) == 33 and len(emb.cosineSim[0]) == 33
    assert all(abs(emb.cosineSim[i][i] - 1.0) < 1e-5 for i in range(33))
    assert len(emb.pcaCoords) == 33 and len(emb.pcaCoords[0]) == 2
    assert len(emb.pcaExplainedVariance) == 2
    assert 0.0 < sum(emb.pcaExplainedVariance) <= 1.0 + 1e-6


def test_build_observation_decodes_state():
    """_build_observation must mirror the acting player's state dict:
    their own cards, relative trick order anchored on their seat, and
    hidden blind/bury for non-pickers."""
    from sheepshead import Game

    from server.runtime.seating import ANALYZE_SEAT_NAMES
    from server.services.analysis_common import build_observation

    game = Game(partner_selection_mode=1, seed=42)
    player = game.players[2]  # seat 3, pre-pick

    obs = build_observation(player.get_state_dict(), 3, ANALYZE_SEAT_NAMES)

    assert sorted(obs.hand) == sorted(player.hand)
    assert obs.blind == []  # not the picker: blind is hidden
    assert obs.bury == []
    assert obs.pickerPosition == 0
    assert obs.pickerRel == 0
    assert not obs.playStarted
    assert not obs.isLeaster
    assert obs.partnerMode == 1
    assert obs.calledCard is None
    assert [slot.relativePosition for slot in obs.trick] == [1, 2, 3, 4, 5]
    assert [slot.seat for slot in obs.trick] == [3, 4, 5, 1, 2]
    assert all(slot.card is None for slot in obs.trick)
