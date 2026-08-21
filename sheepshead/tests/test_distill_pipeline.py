"""Phased-offline distillation pipeline (CE_Teacher_Design §17).

Corpus generation is tested with a scripted committee (deterministic
material targets, no real search); the trainer's channel alignment, loss
masking and the KD anchor's zero-gradient-at-init property are tested on
real generated corpora with the same agent that produced them.
"""

import argparse
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE
from sheepshead.agent.ppo import PPOAgent
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training import train_distill
from sheepshead.training.distill_corpus import (
    _W,
    play_corpus_game,
    schedule_p,
)
from sheepshead.training.train_distill import (
    SET_CODES,
    distill_losses,
    flat_channels,
    kd_kl,
    omega_weights,
    store_episodes,
)

ARCH = "perceiver-shared-v2"


class _ScriptedCommittee:
    """ISMCTSTeacher stand-in: deterministic committee replicates with a
    LARGE Q spread (always material under the real shrinkage) favoring the
    lowest-id valid action. Carries the engine config the target builder
    reads readout constants from."""

    def __init__(self, replicates=3):
        self.replicates = replicates
        self.config = SimpleNamespace(gumbel_c_visit=50.0, gumbel_c_scale=0.1)

    def search_committee(self, game, observer, forced_public, rngs, d_rollout=None):
        player = game.players[observer - 1]
        valid = sorted(player.get_valid_action_ids())
        out = []
        for rep in range(self.replicates):
            q = {a: 1.0 - 0.5 * i + 1e-4 * rep for i, a in enumerate(valid)}
            out.append(
                {
                    "ok": True,
                    "root_q": q,
                    "root_n": {a: 256.0 for a in valid},
                    "root_prior": {a: 1.0 / len(valid) for a in valid},
                }
            )
        return out


def _worker_state(agent, *, committee_act_ready=True, **overrides):
    args = {
        "seed": 7,
        "collect_oracle": overrides.pop("collect_oracle", False),
        "iters": 8,
        "replicates": 3,
        "d_rollout": 1,
        "shrink_nu": 4.0,
        "shrink_s2_global": 6.95e-4,
        "p_base": overrides.pop("p_base", 1.0),
        "boost_lead": 1.0,
        "boost_cs": 1.0,
        "p_min": overrides.pop("p_min", 1.0),
        "p_max": overrides.pop("p_max", 1.0),
    }
    args.update(overrides)
    _W.clear()
    _W.update({"agent": agent, "teacher": _ScriptedCommittee(), "args": args})
    return args


def _fresh_agent():
    seed_all(0)
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    agent.stash_action_probs = True
    return agent


def _generate_game(agent, game_idx=0, committee_act=False, **overrides):
    _worker_state(agent, **overrides)
    return play_corpus_game((game_idx, PARTNER_BY_CALLED_ACE, committee_act))


# --------------------------------------------------------------------------- #
# p-schedule and loss primitives
# --------------------------------------------------------------------------- #
def test_schedule_p_boosts_and_clip():
    kw = dict(p_base=0.10, boost_lead=1.25, boost_cs=1.5, p_min=0.05, p_max=0.25)
    assert schedule_p(False, False, **kw) == pytest.approx(0.10)
    assert schedule_p(True, False, **kw) == pytest.approx(0.125)
    assert schedule_p(True, True, **kw) == pytest.approx(0.1875)
    # clip both ways
    kw["p_base"] = 0.30
    assert schedule_p(True, True, **kw) == pytest.approx(0.25)
    kw["p_base"] = 0.01
    assert schedule_p(False, False, **kw) == pytest.approx(0.05)


def test_omega_weights_monotone_bounded():
    gaps = torch.tensor([0.0, 0.01, 0.03, 0.10, 1.0])
    w = omega_weights(gaps, beta=0.03, omega_max=float(np.e))
    assert torch.all(w[1:] >= w[:-1])
    assert w[0] == pytest.approx(1.0 / np.e)
    assert w[-1] == pytest.approx(1.0)  # clipped at omega_max
    assert torch.all(w <= 1.0) and torch.all(w > 0.0)


def test_kd_kl_zero_at_anchor_and_tau_grad():
    torch.manual_seed(0)
    logits = torch.randn(4, 12)
    logits[:, 8:] = -1e9  # illegal tail
    anchor = torch.softmax(logits, dim=-1)
    anchor = torch.where(anchor > 1e-8, anchor, torch.zeros_like(anchor))
    anchor = anchor / anchor.sum(dim=-1, keepdim=True)
    for tau in (1.0, 2.0):
        student = logits.clone().requires_grad_(True)
        kl = kd_kl(anchor, student, tau)
        assert float(kl.detach().abs().max()) < 1e-5
        kl.sum().backward()
        assert float(student.grad.abs().max()) < 1e-4
    # a perturbed student sees positive KL
    student = logits.clone()
    student[:, 0] += 1.0
    assert float(kd_kl(anchor, student, 1.0).min()) > 0.0


# --------------------------------------------------------------------------- #
# Corpus generation (scripted committee)
# --------------------------------------------------------------------------- #
def test_partition_assignment_and_schema():
    agent = _fresh_agent()
    res = _generate_game(agent, collect_oracle=True)
    assert len(res["episodes"]) == 5
    saw = {"override": 0, "retention": 0, "none": 0, "endorsed": 0}
    for ep in res["episodes"]:
        actions = [e for e in ep if e["kind"] == "action"]
        assert actions, "every seat stream has action rows"
        # terminal reward on the last action only; labels on every row
        rewards = [e["reward"] for e in actions]
        assert all(r == 0.0 for r in rewards[:-1])
        for ev in actions:
            saw[ev["distill_set"]] += 1
            assert "oracle_state" in ev
            head_is_play = ev["node_class"].startswith(("std|", "alone|", "leaster|"))
            if ev["distill_set"] == "override":
                assert head_is_play
                assert ev["has_search_target"]
                assert len(ev["search_target"]) == len(ev["valid_actions"])
                assert ev["anchor_probs"] is None
                assert ev["search_gap"] > 0.0
            elif ev["distill_set"] == "retention":
                # bidding heads and leaster play (alone play is searched)
                assert ev["anchor_probs"] is not None
                assert sum(ev["anchor_probs"]) == pytest.approx(1.0, abs=1e-6)
                assert not ev["has_search_target"]
            elif ev["distill_set"] == "none":
                assert len(ev["valid_actions"]) == 1 or head_is_play
                assert ev["anchor_probs"] is None
    # p=1 + scripted-material committee: every eligible non-leaster play
    # node (std AND alone) searched and material; bidding rows all retention
    assert saw["override"] > 0
    assert saw["retention"] > 0
    assert saw["endorsed"] == 0
    counts = res["counts"]
    play_cells = [c for c in counts if c.startswith(("std|", "alone|"))]
    assert sum(counts[c]["searched"] for c in play_cells) == saw["override"]
    assert all(
        counts[h]["searched"] == 0 for h in ("pick", "partner", "bury") if h in counts
    )
    assert res["telemetry"], "searched nodes emit telemetry rows"
    for row in res["telemetry"]:
        assert row["pair_diffs"] and len(row["top_pair"]) == 2


def test_committee_acting_takes_target_argmax():
    agent = _fresh_agent()
    acted_rows = []
    for game_idx in range(1, 8):  # skip leaster/alone-only draws
        res = _generate_game(agent, game_idx=game_idx, committee_act=True)
        acted_rows = [
            e
            for ep in res["episodes"]
            for e in ep
            if e["kind"] == "action" and e["distill_set"] == "override"
        ]
        if acted_rows:
            break
    assert acted_rows
    # The scripted committee's Q always favors the lowest-id valid action;
    # with committee acting the acted action must be the target argmax.
    for ev in acted_rows:
        acts = sorted(ev["valid_actions"])
        assert ev["action"] == acts[int(np.argmax(ev["search_target"]))]
    assert res["committee_acted"] >= 0


def test_unsearched_play_is_no_loss_not_retention():
    """The §16.9 addendum-3 invariant: eligible-but-unsearched play rows
    carry NO anchor (never merely-unasked anchoring)."""
    agent = _fresh_agent()
    play_rows = []
    for game_idx in range(2, 9):  # skip leaster/alone-only draws
        res = _generate_game(agent, game_idx=game_idx, p_base=0.0, p_min=0.0, p_max=0.0)
        play_rows = [
            e
            for ep in res["episodes"]
            for e in ep
            if e["kind"] == "action"
            and e["node_class"].startswith("std|")
            and len(e["valid_actions"]) >= 2
        ]
        if play_rows:
            break
    assert play_rows
    assert all(e["distill_set"] == "none" for e in play_rows)
    assert all(e["anchor_probs"] is None for e in play_rows)


# --------------------------------------------------------------------------- #
# Trainer: channel alignment, loss masking, zero-gradient anchors
# --------------------------------------------------------------------------- #
def _trainer_args(**overrides):
    args = argparse.Namespace(
        lambda_ce=1.0,
        lambda_end=0.5,
        lambda_ret=0.5,
        beta=0.03,
        omega_max=float(np.e),
        kd_tau=1.0,
        train_oracle=False,
        buffer_episodes=64,
        batch_segments=8,
    )
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _store_and_batch(agent, episodes):
    agent.reset_storage()
    store_episodes(agent, episodes)
    states, masks_t, kinds = agent._prepare_training_views()
    segments = agent._segments_from_events(kinds)
    minibatch = agent._build_minibatch_tensors(segments, states, masks_t, kinds)
    forward = agent._forward_vectorized(minibatch.states_seqs, minibatch.masks_bt)
    flat = agent._flatten_action_steps(minibatch, forward)
    dchan = flat_channels(agent, segments, kinds)
    return minibatch, forward, flat, dchan


def test_channel_alignment_and_loss_masking():
    agent = _fresh_agent()
    res = _generate_game(agent, game_idx=3)
    with torch.no_grad():
        minibatch, forward, flat, dchan = _store_and_batch(agent, res["episodes"])
        set_flat, gap_flat, anchor_flat = dchan

        src = [e for ep in res["episodes"] for e in ep if e["kind"] == "action"]
        # Row counts per partition match the source events exactly.
        for name, code in SET_CODES.items():
            assert int((set_flat == code).sum()) == sum(
                e["distill_set"] == name for e in src
            )
        # Alignment: override rows carry the dense target exactly where
        # has_search does; anchors normalize on anchor rows and are zero
        # elsewhere.
        ov = set_flat == SET_CODES["override"]
        assert torch.equal(ov, flat.has_search_flat > 0.5)
        assert torch.all(flat.search_target_flat[ov].sum(dim=-1) > 0.99)
        anchored = set_flat >= SET_CODES["endorsed"]
        assert torch.allclose(
            anchor_flat[anchored].sum(dim=-1),
            torch.ones(int(anchored.sum())),
            atol=1e-5,
        )
        assert float(anchor_flat[~anchored].abs().sum()) == 0.0
        # Gaps live only on override rows.
        assert torch.all(gap_flat[ov] > 0.0)
        assert float(gap_flat[~ov].abs().sum()) == 0.0

        total, stats = distill_losses(
            agent, minibatch, forward, flat, dchan, _trainer_args()
        )
        assert torch.isfinite(total)
        assert stats["override_rows"] == int(ov.sum())
        # Zero-gradient-at-init property (§17.3): the anchors were stashed
        # from THIS agent's own act() calls, and the replayed unroll
        # reproduces them — the retention/endorsed KL starts ~0.
        assert stats["retention_kl"] < 1e-3


def test_run_epoch_steps_and_updates_weights():
    agent = _fresh_agent()
    episodes = []
    overrides = 0
    for g in range(10, 20):  # collect until an override row exists
        res = _generate_game(agent, game_idx=g)
        episodes.extend(res["episodes"])
        overrides += sum(c["override"] for c in res["counts"].values())
        if len(episodes) >= 10 and overrides:
            break
    assert overrides
    ref = agent.encoder.card.weight.detach().clone()
    args = _trainer_args(batch_segments=4, buffer_episodes=16)
    stats, steps = train_distill.run_epoch(agent, episodes, args, train=True)
    assert steps > 0
    assert stats["override_rows"] > 0
    assert torch.isfinite(torch.tensor(stats["value_mse"]))
    assert not torch.equal(ref, agent.encoder.card.weight.detach())
    # Eval pass leaves weights untouched.
    ref2 = agent.encoder.card.weight.detach().clone()
    train_distill.run_epoch(agent, episodes, args, train=False)
    assert torch.equal(ref2, agent.encoder.card.weight.detach())


@pytest.mark.slow
def test_generator_end_to_end_real_search(tmp_path):
    """Smoke: the real ISMCTS committee through play_corpus_game at a tiny
    budget (uncertified; schema only)."""
    from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher

    agent = _fresh_agent()
    _worker_state(agent, p_base=0.3, p_min=0.0, p_max=0.3)
    _W["teacher"] = ISMCTSTeacher(
        agent, ISMCTSConfig(iters={h: 8 for h in ("pick", "partner", "bury", "play")})
    )
    res = play_corpus_game((0, PARTNER_BY_CALLED_ACE, False))
    assert len(res["episodes"]) == 5
    sets = {
        e["distill_set"] for ep in res["episodes"] for e in ep if e["kind"] == "action"
    }
    assert "retention" in sets


def teardown_function(_fn):
    _W.clear()
    random.seed()
