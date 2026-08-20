"""CE search teacher (CE_Teacher_Design §1-§2, 2026-08).

Target construction is unit-tested against synthetic committee replicate
tables (deterministic root_q / root_n / root_prior); the emission wrapper is
tested with a scripted committee; one smoke test runs the real ISMCTS
teacher end-to-end through play_population_game.
"""

import random

import numpy as np
import pytest

from sheepshead import ACTION_LOOKUP, ACTIONS, PARTNER_BY_CALLED_ACE, Game
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher, _minmax_unit
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.config import SearchConfig
from sheepshead.training.pfsp_runtime import (
    _attach_ce_search_target,
    build_ce_search_target,
    play_population_game,
)

ARCH = "perceiver-shared-v2"

C_VISIT = 50.0
C_SCALE = 0.1


def _to_first_play_node(game_seed=11):
    """Advance a game to the first PLAY decision with scripted bidding that
    forces a standard called-ace game (PICK when offered, never ALONE, never
    a leaster), returning (game, player, valid_actions)."""
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=game_seed)
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                valid_sorted = sorted(valid)
                names = {a: ACTION_LOOKUP[a] for a in valid_sorted}
                play = [a for a, n in names.items() if n.startswith("PLAY ")]
                if play and not game.is_leaster and not game.alone_called:
                    return game, player, valid_sorted
                pick = [a for a, n in names.items() if n == "PICK"]
                safe = [a for a, n in names.items() if "ALONE" not in n]
                player.act(pick[0] if pick else (safe[0] if safe else valid_sorted[0]))
                valid = player.get_valid_action_ids()
    raise AssertionError("no standard play node reached")


def _replicate(valid, qmap, n=256.0, prior=None):
    """Synthetic committee replicate in SearchResult shape (the target
    builder consumes ok / root_q / root_n / root_prior only)."""
    uniform = 1.0 / len(valid)
    return {
        "ok": True,
        "root_q": {a: float(qmap[a]) for a in valid},
        "root_n": {a: float(n) for a in valid},
        "root_prior": {a: float((prior or {}).get(a, uniform)) for a in valid},
    }


def _build(replicates, valid, s2_global=1.1e-4, nu=4.0):
    return build_ce_search_target(
        replicates,
        valid,
        shrink_nu=nu,
        shrink_s2_global=s2_global,
        gumbel_c_visit=C_VISIT,
        gumbel_c_scale=C_SCALE,
    )


class _ScriptedCommittee:
    """ISMCTSTeacher stand-in: search_committee returns the queued replicate
    list; carries the engine config the emission reads readout constants
    from."""

    def __init__(self, replicates):
        self.replicates = list(replicates)
        self.calls = 0
        self.config = ISMCTSConfig()

    def search_committee(self, game, observer, forced_public, rngs, d_rollout=None):
        self.calls += 1
        assert len(rngs) == len(self.replicates)
        return list(self.replicates)


def _run_emission(teacher, valid, game, player, cfg=None, live=None):
    transition = {"search_target": None, "has_search_target": False}
    diag = {
        "play": {
            "count": 0,
            "labeled": 0,
            "material": 0,
            "w_sum": 0.0,
            "spread_sum": 0.0,
            "kl_sum": 0.0,
            "kl_n": 0,
        }
    }
    cfg = cfg or SearchConfig(teacher_prob=1.0, teacher_replicates=3)
    _attach_ce_search_target(
        game, player, valid, transition, teacher, random.Random(3), cfg, [], diag, live
    )
    return transition, diag


# --------------------------------------------------------------------------
# Target construction (build_ce_search_target)
# --------------------------------------------------------------------------
class TestTargetConstruction:
    def test_flat_committee_shrinks_to_prior(self):
        # Identical Q everywhere: zero signal variance -> w = 0 -> the
        # target IS the pooled expert prior (abstention as the target's
        # fixed point; CE gradient ~ 0 when the student matches the prior).
        valid = [1, 2, 3, 4]
        prior = {1: 0.5, 2: 0.25, 3: 0.15, 4: 0.10}
        reps = [_replicate(valid, {a: 0.4 for a in valid}, prior=prior)] * 3
        target, info = _build(reps, valid)
        assert info["w"] == 0.0
        np.testing.assert_allclose(target, [prior[a] for a in sorted(valid)], rtol=1e-6)

    def test_within_noise_spread_shrinks_to_flat(self):
        # Q spread comparable to the replicate noise floor: James-Stein
        # shrinks the node to (near) zero tilt even though the raw tables
        # are not exactly flat.
        valid = [1, 2, 3]
        base = {1: 0.400, 2: 0.402, 3: 0.401}
        reps = [
            _replicate(valid, {a: q + d for a, q in base.items()})
            for d in (-0.01, 0.0, 0.01)
        ]
        target, info = _build(reps, valid)
        assert info["w"] == 0.0
        np.testing.assert_allclose(target, [1 / 3] * 3, rtol=1e-6)

    def test_separated_committee_preserves_direction(self):
        # A clear, replicate-stable gap tilts the target toward the better
        # action and away from the worst, past the (uniform) prior.
        valid = [1, 2, 3]
        qmap = {1: 0.20, 2: 0.60, 3: 0.40}
        reps = [_replicate(valid, qmap)] * 3
        target, info = _build(reps, valid, s2_global=1e-6)
        assert info["w"] > 0.9
        assert target[1] > target[2] > target[0]
        assert target[1] > 1 / 3 > target[0]

    def test_full_confidence_matches_pi_gumbel_readout(self):
        # Zero replicate noise + zero global noise -> w = 1 exactly, and the
        # target must reproduce the engine's pi_gumbel formula on the pooled
        # stats: softmax(log p_raw + (c_visit + max N) * c_scale * qhat).
        valid = [2, 5, 9]
        qmap = {2: 0.1, 5: 0.7, 9: 0.4}
        prior = {2: 0.6, 5: 0.1, 9: 0.3}
        n = 128.0
        reps = [_replicate(valid, qmap, n=n, prior=prior)] * 3
        target, info = _build(reps, valid, s2_global=0.0)
        assert info["w"] == pytest.approx(1.0)
        q = np.array([qmap[a] for a in sorted(valid)])
        p = np.array([prior[a] for a in sorted(valid)])
        logits = np.log(p) + (C_VISIT + n) * C_SCALE * _minmax_unit(q)
        expected = np.exp(logits - logits.max())
        expected /= expected.sum()
        np.testing.assert_allclose(target, expected, rtol=1e-5)

    def test_shrink_interpolates_between_prior_and_readout(self):
        # 0 < w < 1 must land strictly between the flat (prior) target and
        # the full readout on every action — the continuous-evidence
        # property the affine-invariance note in the builder protects.
        valid = [1, 2, 3]
        qmap = {1: 0.30, 2: 0.42, 3: 0.36}
        reps = [
            _replicate(valid, {a: q + d for a, q in qmap.items()})
            for d in (-0.02, 0.0, 0.02)
        ]
        target, info = _build(reps, valid)
        assert 0.0 < info["w"] < 1.0
        full, _ = _build([_replicate(valid, qmap)] * 3, valid, s2_global=0.0)
        assert 1 / 3 < target[1] < full[1]
        assert full[0] < target[0] < 1 / 3

    def test_variance_blend_math(self):
        # The hierarchical blend, checked end-to-end through w: with the
        # node variance forced to zero (identical replicates), the noise
        # term is nu*s2_global/(nu + R - 1)/R and w = 1 - noise/Var(q).
        valid = [1, 2]
        qmap = {1: 0.40, 2: 0.44}
        reps = [_replicate(valid, qmap)] * 3
        nu, s2g, r = 4.0, 2e-4, 3
        target, info = _build(reps, valid, s2_global=s2g, nu=nu)
        noise = (nu * s2g / (nu + r - 1)) / r
        var_q = np.var([0.40, 0.44])
        assert info["w"] == pytest.approx(1.0 - noise / var_q, rel=1e-9)

    def test_unusable_committee_returns_none(self):
        valid = [1, 2]
        bad = {"ok": False, "root_q": None, "root_n": {}, "root_prior": None}
        assert _build([bad] * 3, valid) is None
        # A single usable replicate cannot estimate replicate variance.
        one = _replicate(valid, {1: 0.4, 2: 0.6})
        assert _build([one, bad, bad], valid) is None

    def test_target_shape_and_normalization(self):
        valid = [7, 3, 12]  # deliberately unsorted input
        reps = [_replicate(sorted(valid), {3: 0.2, 7: 0.5, 12: 0.35})] * 3
        target, _ = _build(reps, valid)
        assert target.dtype == np.float32
        assert target.shape == (len(valid),)  # aligned to sorted(valid)
        assert target.sum() == pytest.approx(1.0, abs=1e-6)


# --------------------------------------------------------------------------
# Emission wrapper (_attach_ce_search_target)
# --------------------------------------------------------------------------
class TestEmission:
    def test_labels_eligible_node_and_counts_material(self):
        game, player, valid = _to_first_play_node()
        qmap = {a: 0.40 for a in valid}
        qmap[valid[1]] = 0.55
        teacher = _ScriptedCommittee([_replicate(valid, qmap)] * 3)
        tr, diag = _run_emission(teacher, valid, game, player)
        assert teacher.calls == 1  # ONE lockstep committee call, not R serial
        assert tr["has_search_target"] is True
        assert tr["search_target"].shape == (len(valid),)
        d = diag["play"]
        assert d["count"] == 1 and d["labeled"] == 1 and d["material"] == 1
        assert d["w_sum"] > 0.0 and d["spread_sum"] == pytest.approx(0.15)

    def test_flat_node_still_labeled_but_immaterial(self):
        # Class-blind, no emission gate: a within-noise node still ships a
        # target (= the expert prior), it just counts as non-material.
        game, player, valid = _to_first_play_node()
        teacher = _ScriptedCommittee([_replicate(valid, {a: 0.4 for a in valid})] * 3)
        tr, diag = _run_emission(teacher, valid, game, player)
        assert tr["has_search_target"] is True
        assert diag["play"]["material"] == 0
        np.testing.assert_allclose(tr["search_target"], 1.0 / len(valid), rtol=1e-6)

    def test_subsample_probability_gates_search(self):
        game, player, valid = _to_first_play_node()
        teacher = _ScriptedCommittee([])
        cfg = SearchConfig(teacher_prob=0.0, teacher_replicates=3)
        tr, diag = _run_emission(teacher, valid, game, player, cfg)
        assert teacher.calls == 0
        assert tr["has_search_target"] is False
        assert diag["play"]["count"] == 0

    def test_kl_telemetry_uses_live_policy(self):
        game, player, valid = _to_first_play_node()
        qmap = {a: 0.40 for a in valid}
        qmap[valid[1]] = 0.55
        teacher = _ScriptedCommittee([_replicate(valid, qmap)] * 3)
        live = np.zeros(len(ACTIONS))
        for a in valid:
            live[a - 1] = 1.0 / len(valid)
        tr, diag = _run_emission(teacher, valid, game, player, live=live)
        assert diag["play"]["kl_n"] == 1
        # Target is tilted away from the uniform live policy -> positive KL.
        assert diag["play"]["kl_sum"] > 0.0
        # Without live probs the label still ships; only telemetry is skipped.
        teacher2 = _ScriptedCommittee([_replicate(valid, qmap)] * 3)
        tr2, diag2 = _run_emission(teacher2, valid, game, player, live=None)
        assert tr2["has_search_target"] is True
        assert diag2["play"]["kl_n"] == 0

    def test_class_blind_serves_jack_of_diamonds_games(self):
        # The teacher covers BOTH partner modes (operator directive
        # 2026-08-11); with the cell taxonomy gone, any standard-play node
        # is eligible.
        from sheepshead import PARTNER_BY_JD

        game = Game(partner_selection_mode=PARTNER_BY_JD, seed=11)
        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    valid_sorted = sorted(valid)
                    names = {a: ACTION_LOOKUP[a] for a in valid_sorted}
                    play = [a for a, n in names.items() if n.startswith("PLAY ")]
                    if play and not game.is_leaster and not game.alone_called:
                        qmap = {a: 0.40 for a in valid_sorted}
                        qmap[valid_sorted[1]] = 0.55
                        teacher = _ScriptedCommittee(
                            [_replicate(valid_sorted, qmap)] * 3
                        )
                        tr, diag = _run_emission(teacher, valid_sorted, game, player)
                        assert tr["has_search_target"] is True
                        assert diag["play"]["labeled"] == 1
                        return
                    pick = [a for a, n in names.items() if n == "PICK"]
                    safe = [a for a, n in names.items() if "ALONE" not in n]
                    player.act(
                        pick[0] if pick else (safe[0] if safe else valid_sorted[0])
                    )
                    valid = player.get_valid_action_ids()
        raise AssertionError("no standard JD play node reached")


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------
def test_ce_teacher_end_to_end_smoke():
    seed_all(7)
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    # Stationary expert: the teacher wraps a separate frozen agent.
    expert = PPOAgent(len(ACTIONS), arch=ARCH)

    class _OpponentStub:
        def __init__(self, a):
            self.agent = a
            self.member_id = "stub"

    teacher = ISMCTSTeacher(
        expert,
        ISMCTSConfig(
            iters={"pick": 8, "partner": 8, "bury": 8, "play": 8}, batch_size=4
        ),
    )
    cfg = SearchConfig(teacher_prob=1.0, teacher_replicates=3, teacher_d_rollout=1)
    _, events, _, data, _ = play_population_game(
        training_agent=agent,
        opponents=[_OpponentStub(agent) for _ in range(4)],
        partner_mode=PARTNER_BY_CALLED_ACE,
        training_agent_position=1,
        reward_mode="terminal",
        teacher=teacher,
        determinization_rng=random.Random(5),
        search_config=cfg,
        game_seed=11,
    )
    transitions = [t for t in events if isinstance(t, dict) and "state" in t]
    assert transitions, "no training transitions collected"
    diag = data["search_diagnostics"]["play"]
    labeled = [t for t in transitions if t.get("has_search_target")]
    assert diag["labeled"] == len(labeled)
    assert labeled, "prob=1.0 teacher never labeled a play node"
    for t in labeled:
        # The target must survive event normalization: without it the CE
        # loss no-ops on the labeled row (the attempt-5a failure mode).
        target = t.get("search_target")
        assert target is not None
        assert len(target) == len(t["valid_actions"])
        assert float(np.sum(target)) == pytest.approx(1.0, abs=1e-5)
        assert np.all(np.asarray(target) >= 0.0)


def test_worker_protocol_serves_ce_teacher(tmp_path):
    # Parallel-collection path: an oracle-mode worker built from init_args
    # must load the payload's oracle head + gamma on weight refresh and run
    # committee searches through league_worker_play (in-process, no pool).
    import torch

    from sheepshead.training import league_worker as lw
    from sheepshead.training.league import SELF_PLAY

    seed_all(9)
    main_agent = PPOAgent(
        len(ACTIONS), arch=ARCH, critic_mode="oracle", oracle_aux_heads=True
    )
    main_agent.gamma = 1.0
    base = tmp_path / "weights"
    torch.save(
        {
            "encoder_state_dict": main_agent.encoder.state_dict(),
            "actor_state_dict": main_agent.actor.state_dict(),
            "critic_state_dict": main_agent.critic.state_dict(),
            "gamma": main_agent.gamma,
            "oracle_state_dict": main_agent.oracle_critic.state_dict(),
        },
        f"{base}_v1.pt",
    )
    lw.league_worker_init(
        {
            "arch": ARCH,
            "members_dir": str(tmp_path),
            "weight_path_base": str(base),
            "base_seed": 0,
            "critic_mode": "oracle",
            "oracle_aux_heads": True,
            "teacher": True,
            "teacher_prob": 1.0,
            "teacher_replicates": 3,  # test-speed committee
            "teacher_iters": 8,  # test-speed override
            "teacher_gamma": 1.0,
        }
    )
    # Which deals avoid leaster/ALONE (ineligible) is seed-sensitive: try a
    # few deals until the teacher fires.
    out = None
    for ep, seed in enumerate((11, 12, 13, 14, 15), start=1):
        job = lw.WorkerJob(
            episode=ep,
            partner_mode=PARTNER_BY_CALLED_ACE,
            training_position=1,
            opponent_ids=[SELF_PLAY] * 4,
            weight_version=1,
            game_seed=seed,
        )
        out = lw.league_worker_play(job)
        if out["training_data_single"]["search_diagnostics"]["play"]["count"]:
            break
    worker = lw.WORKER_STATE["agent"]
    assert worker.gamma == 1.0  # payload gamma reached the live worker agent
    assert worker.oracle_critic is not None
    ref = main_agent.oracle_critic.state_dict()
    got = worker.oracle_critic.state_dict()
    assert all(torch.equal(ref[k], got[k]) for k in ref)
    # Closed-loop expert (§15a): the teacher wraps the live worker agent, so
    # every weight refresh reaches the committee with at most one version lag.
    assert lw.WORKER_STATE["teacher"].agent is worker
    assert out["episode_events"], "worker produced no transitions"
    diag = out["training_data_single"]["search_diagnostics"]["play"]
    assert diag["count"] >= 1, "teacher never attempted despite prob=1.0"
