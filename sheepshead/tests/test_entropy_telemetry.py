"""Adaptive-entropy Phase 1 instrumentation invariants.

Two measurement paths must agree in definition (per-node H/ln(n_legal) over
the legal set, per head, forced moves excluded): the live theta_old update
telemetry (stats["head_entropy_norm"]) and the offline checkpoint probe
(analysis/entropy_probe.py). These tests pin the shared contract: values in
[0, 1], forced moves excluded, first-epoch-only measurement (theta_old under
grad accumulation), probe determinism and side-effect freedom.

Background: SAC-style target-entropy control (Haarnoja et al.,
arXiv:1812.05905; discrete: Christodoulou, arXiv:1910.07207) initialized
bumplessly from measured values (Astrom & Wittenmark, Adaptive Control)."""

import random

from sheepshead import ACTIONS, PARTNER_BY_JD
from sheepshead.agent.ppo import PPOAgent
from sheepshead.analysis.entropy_probe import HEADS, probe_agent
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.league import SELF_PLAY
from sheepshead.training.pfsp_runtime import play_population_game
from sheepshead.training.train_league_ppo import _Seat, store_events_by_seat

SEED = 20260728
ARCH = "perceiver-shared-v2"


def _agent():
    seed_all(SEED)
    return PPOAgent(len(ACTIONS), critic_mode="limited", arch=ARCH)


def _fill_buffer(agent, n_episodes=2):
    for i in range(n_episodes):
        opponents = [_Seat(agent, SELF_PLAY) for _ in range(4)]
        seed_all(SEED + 500 + i)
        _, events, _, _, _ = play_population_game(
            training_agent=agent,
            opponents=opponents,
            partner_mode=PARTNER_BY_JD,
            training_agent_position=1,
            reward_mode="terminal",
        )
        store_events_by_seat(agent, events)


def _assert_valid_norm_stats(hnorm, rows):
    assert set(hnorm) == set(HEADS)
    for head in HEADS:
        v = hnorm[head]
        if v is not None:
            assert 0.0 <= v <= 1.0, f"{head}: H_norm {v} outside [0, 1]"
            assert rows[head] > 0
        else:
            assert rows[head] == 0
    # Every episode has pick and play decisions with >= 2 legal actions.
    assert hnorm["pick"] is not None
    assert hnorm["play"] is not None


def test_update_stats_report_normalized_head_entropy():
    agent = _agent()
    _fill_buffer(agent)
    stats = agent.update(epochs=2, batch_size=8, grad_accum=True)
    _assert_valid_norm_stats(
        stats["head_entropy_norm"], stats["head_entropy_norm_rows"]
    )


def test_measurement_is_first_epoch_only(monkeypatch):
    """theta_old semantics: under grad accumulation no optimizer step lands
    inside the first epoch, so measuring on first-epoch minibatches only IS
    measuring the behavior policy. Pin the wiring: measure_norm_entropy is
    True for exactly the first epoch's minibatches (a strict prefix of the
    call sequence), and each buffer row is counted at most once (a
    per-epoch measurement would multiply the row total by the epoch
    count)."""
    agent = _agent()
    _fill_buffer(agent)
    n_action_rows = sum(1 for e in agent.events if e["kind"] == "action")

    flags = []
    orig = PPOAgent._update_minibatch

    def spy(self, *args, **kwargs):
        flags.append(bool(kwargs.get("measure_norm_entropy", False)))
        return orig(self, *args, **kwargs)

    monkeypatch.setattr(PPOAgent, "_update_minibatch", spy)
    stats = agent.update(epochs=3, batch_size=4, grad_accum=True)

    k = sum(flags)
    assert k >= 1
    assert flags == [True] * k + [False] * (len(flags) - k), (
        "measured minibatches must be exactly the first epoch's"
    )
    rows = stats["head_entropy_norm_rows"]
    assert 0 < sum(rows.values()) <= n_action_rows


def test_probe_bounds_determinism_and_no_side_effects():
    agent = _agent()
    rng_before = random.getstate()
    res1 = probe_agent(agent, n_games=3, seed=123)
    res2 = probe_agent(agent, n_games=3, seed=123)
    # Same seed => same deal panel and sampling stream. Device forwards are
    # not bit-deterministic on this platform (cf. the environment-stamped
    # bit-exact fixtures), so trajectories can diverge at near-tie nodes;
    # repeated readings on adequately-sampled heads must still agree to well
    # under the effect sizes the backfill interprets (~0.01). Heads with only
    # a handful of rows in 3 games (partner/bury) are excluded — one
    # divergent trajectory legitimately moves their mean.
    for head in HEADS:
        s1, s2 = res1["heads"][head], res2["heads"][head]
        if s1["rows"] >= 30 and s2["rows"] >= 30:
            assert abs(s1["mean"] - s2["mean"]) < 0.02, (
                f"{head}: {s1['mean']} vs {s2['mean']}"
            )
    assert res1["heads"]["play"]["rows"] >= 30  # the comparison is non-vacuous
    assert random.getstate() == rng_before  # global RNG restored
    hnorm = {h: res1["heads"][h]["mean"] for h in HEADS}
    rows = {h: res1["heads"][h]["rows"] for h in HEADS}
    _assert_valid_norm_stats(hnorm, rows)
    # A freshly initialized policy is near-uniform over legal actions: the
    # play head (many legal moves) must read high normalized entropy.
    assert hnorm["play"] > 0.5


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-v"]))
