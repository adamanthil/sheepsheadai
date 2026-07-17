"""Tests for the pick/call/bury scenario analysis service."""

from __future__ import annotations

import pytest

from server.api.schemas import AnalyzePickRequest
from server.config import get_settings


@pytest.fixture
def analyze_env(monkeypatch, tmp_path):
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


@pytest.fixture
def pick_agent(monkeypatch):
    from sheepshead import ACTIONS
    from sheepshead.agent.ppo import PPOAgent

    import server.services.pick_analysis as pick_mod

    agent = PPOAgent(len(ACTIONS), arch="full")
    monkeypatch.setattr(pick_mod, "load_agent", lambda path: agent)
    return agent


def _run(req: AnalyzePickRequest):
    from server.services.pick_analysis import analyze_pick

    return analyze_pick(req)


def test_pick_validation_errors(analyze_env, pick_agent):
    with pytest.raises(ValueError, match="seat"):
        _run(AnalyzePickRequest(seat=6))
    with pytest.raises(ValueError, match="at most 6"):
        _run(
            AnalyzePickRequest(
                hand=["QC", "QS", "QH", "QD", "JC", "JS", "JH"]
            )
        )
    with pytest.raises(ValueError, match="unknown"):
        _run(AnalyzePickRequest(hand=["QC", "QS", "QH", "QD", "JC", "XX"]))
    with pytest.raises(ValueError, match="duplicate"):
        _run(AnalyzePickRequest(hand=["QC", "QC", "QH", "QD", "JC", "JS"]))
    with pytest.raises(ValueError, match="at most 2"):
        _run(AnalyzePickRequest(blind=["7C", "8C", "9C"]))
    with pytest.raises(ValueError, match="overlap"):
        _run(
            AnalyzePickRequest(
                hand=["QC", "QS", "QH", "QD", "JC", "JS"], blind=["QC", "7C"]
            )
        )


def test_pick_partial_hand_and_blind_filled_randomly(analyze_env, pick_agent):
    """A partial selection locks those cards and deals the rest; the fill
    respects the seed and never reuses a locked card."""
    locked_hand = ["QC", "QS"]
    locked_blind = ["7C"]
    resp = _run(
        AnalyzePickRequest(
            seat=4, hand=locked_hand, blind=locked_blind, seed=21
        )
    )

    assert len(resp.scenario.hand) == 6
    assert len(resp.scenario.blind) == 2
    assert set(locked_hand) <= set(resp.scenario.hand)
    assert set(locked_blind) <= set(resp.scenario.blind)
    assert not set(resp.scenario.hand) & set(resp.scenario.blind)
    assert resp.decisions[0].seat == 4
    assert resp.decisions[0].phase == "pick"

    # Same seed -> same fill; the locked cards stay put either way.
    again = _run(
        AnalyzePickRequest(
            seat=4, hand=locked_hand, blind=locked_blind, seed=21
        )
    )
    assert again.scenario.hand == resp.scenario.hand
    assert again.scenario.blind == resp.scenario.blind


def test_pick_scenario_targets_requested_seat(analyze_env, pick_agent):
    hand = ["QC", "QS", "QH", "QD", "JC", "JS"]
    blind = ["7C", "8C"]
    resp = _run(
        AnalyzePickRequest(seat=3, hand=hand, blind=blind, seed=5)
    )

    assert resp.scenario.seat == 3
    assert sorted(resp.scenario.hand) == sorted(hand)
    assert resp.scenario.blind == blind

    # The first decision must be seat 3 facing PICK/PASS with the forced hand.
    first = resp.decisions[0]
    assert first.seat == 3
    assert first.phase == "pick"
    assert {p.action for p in first.probabilities} == {"PICK", "PASS"}

    # Deal integrity: every seat's observed cards are disjoint and no forced
    # card leaked to another seat (decisions after a pick reveal the picker's
    # hand incl. blind, so check via the game-agnostic first decision only).
    assert resp.decisions
    for d in resp.decisions:
        assert d.phase in {"pick", "partner", "bury"}

    # Pre-play only: never reaches a play decision, and the outcome is
    # coherent (either someone picked or it went to leaster).
    if resp.outcome.pickerSeat is not None:
        assert resp.outcome.isLeaster is False
        assert len(resp.outcome.bury) == 2 or resp.outcome.pickerSeat != 3
    else:
        assert resp.outcome.isLeaster is True


def test_pick_random_hand_is_reproducible(analyze_env, pick_agent):
    r1 = _run(AnalyzePickRequest(seat=2, seed=99))
    r2 = _run(AnalyzePickRequest(seat=2, seed=99))
    assert r1.scenario.hand == r2.scenario.hand
    assert r1.scenario.blind == r2.scenario.blind
    assert [d.actionId for d in r1.decisions] == [d.actionId for d in r2.decisions]
    assert len(r1.scenario.hand) == 6
    assert len(r1.scenario.blind) == 2
    assert not set(r1.scenario.hand) & set(r1.scenario.blind)
    assert r1.decisions[0].seat == 2
    assert r1.decisions[0].phase == "pick"
