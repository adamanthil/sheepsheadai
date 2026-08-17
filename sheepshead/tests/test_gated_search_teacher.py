"""Resolved-pair search teacher (Search_Teacher_Design §12.7, 2026-08-16).

Gate mechanics are unit-tested against a scripted committee (deterministic
per-replicate root-Q tables); one smoke test runs the real ISMCTS teacher
end-to-end through play_population_game in gated mode.
"""

import random

import numpy as np
import pytest

from sheepshead import ACTION_LOOKUP, ACTIONS, PARTNER_BY_CALLED_ACE, Game
from sheepshead.agent.ppo import PPOAgent
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
from sheepshead.tests.ppo_test_helpers import seed_all
from sheepshead.training.config import SearchConfig
from sheepshead.training.pfsp_runtime import (
    _attach_gated_search_target,
    play_cell,
    play_population_game,
)

ARCH = "perceiver-shared-v2"


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


class _ScriptedTeacher:
    """Committee stub: returns the queued result per search() call."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def search(self, game, observer, forced_public, rng, d_rollout=None):
        self.calls += 1
        return self.results.pop(0)


def _qres(valid, qmap, ok=True):
    """Canned FROZEN-expert search result: a completed-Q table over the
    node's legal actions (the §12.7 gate consumes root_q only)."""
    return {
        "ok": ok,
        "root_q": {a: float(qmap[a]) for a in valid},
        "valid": list(valid),
    }


def _live_probs(valid, argmax_action):
    """LIVE-policy label-time distribution (the act() stash): 0.6 on the
    student's argmax, the rest spread over the other legal actions."""
    probs = np.zeros(len(ACTIONS))
    probs[argmax_action - 1] = 0.6
    for a in valid:
        if a != argmax_action:
            probs[a - 1] = 0.4 / (len(valid) - 1)
    return probs


def _run_gate(teacher, valid, game, player, cfg=None, live=None):
    transition = {"search_pairs": None, "has_search_target": False}
    diag = {
        "play": {
            "count": 0,
            "accepted": 0,
            "resolved_sum": 0.0,
            "satisfied_sum": 0.0,
            "pairs_sum": 0.0,
            "gap_sum": 0.0,
        }
    }
    cfg = cfg or SearchConfig(
        gate_node_prob=1.0,
        gate_replicates=3,
        gate_cells=frozenset({play_cell(game, player)}),
    )
    live = _live_probs(valid, valid[0]) if live is None else live
    _attach_gated_search_target(
        game, player, valid, transition, teacher, random.Random(3), cfg, [], diag, live
    )
    return transition, diag


def test_emits_resolved_pair_with_live_anchors():
    game, player, valid = _to_first_play_node()
    pol, alt = valid[0], valid[1]
    # alt clearly better than pol; every other action exactly tied with pol
    # (zero paired difference = sign-inconsistent = never resolved).
    qmap = {a: 0.40 for a in valid}
    qmap[alt] = 0.50
    teacher = _ScriptedTeacher([_qres(valid, qmap)] * 3)
    tr, diag = _run_gate(teacher, valid, game, player)
    assert teacher.calls == 3  # pair statistics need the full committee
    assert tr["has_search_target"] is True
    pairs = tr["search_pairs"]
    # alt beats pol AND every tied-with-pol action: n-1 resolved pairs, all
    # unsatisfied at label time (live argmax pol holds 0.6).
    assert len(pairs) == min(len(valid) - 1, SearchConfig().gate_max_pairs)
    n_other = len(valid) - 1
    for w, loser, anchor_w, _anchor_l in pairs:
        assert int(w) == alt and int(loser) != alt
        assert anchor_w == pytest.approx(np.log(0.4 / n_other), abs=1e-9)
    losers = {int(p[1]) for p in pairs}
    assert pol in losers
    pol_pair = next(p for p in pairs if int(p[1]) == pol)
    assert pol_pair[3] == pytest.approx(np.log(0.6), abs=1e-9)
    assert diag["play"]["count"] == 1 and diag["play"]["accepted"] == 1
    assert diag["play"]["resolved_sum"] == len(pairs)
    assert diag["play"]["satisfied_sum"] == 0


def test_abstains_on_replicate_inconsistent_sign():
    game, player, valid = _to_first_play_node()
    alt = valid[1]
    # The alt-vs-policy difference flips sign across replicates (seed noise at
    # a genuine near-tie): never resolved, no emission.
    hi = {a: 0.40 for a in valid}
    hi[alt] = 0.45
    lo = {a: 0.40 for a in valid}
    lo[alt] = 0.35
    teacher = _ScriptedTeacher([_qres(valid, hi), _qres(valid, lo), _qres(valid, hi)])
    tr, diag = _run_gate(teacher, valid, game, player)
    assert tr["has_search_target"] is False
    assert diag["play"]["resolved_sum"] == 0
    assert diag["play"]["accepted"] == 0


def test_eps_floor_blocks_sub_material_gaps():
    game, player, valid = _to_first_play_node()
    alt = valid[1]
    # Sign-consistent but the mean gap (0.005) is below gate_pair_eps: a
    # statistically-clean but sub-material preference is not taught.
    qmap = {a: 0.400 for a in valid}
    qmap[alt] = 0.405
    teacher = _ScriptedTeacher([_qres(valid, qmap)] * 3)
    tr, diag = _run_gate(teacher, valid, game, player)
    assert tr["has_search_target"] is False
    assert diag["play"]["resolved_sum"] == 0


def test_noise_scaled_threshold_blocks_unstable_gaps():
    game, player, valid = _to_first_play_node()
    alt = valid[1]
    # Sign-consistent, mean 0.02 >= eps, but replicate spread makes
    # z*s/sqrt(R) ~ 0.038 > 0.02: the committee cannot statistically
    # resolve the pair, so no constraint is emitted.
    base = {a: 0.40 for a in valid}
    reps = []
    for bump in (0.001, 0.001, 0.058):
        q = dict(base)
        q[alt] = 0.40 + bump
        reps.append(_qres(valid, q))
    teacher = _ScriptedTeacher(reps)
    tr, diag = _run_gate(teacher, valid, game, player)
    assert tr["has_search_target"] is False
    assert diag["play"]["resolved_sum"] == 0


def test_satisfied_pairs_counted_but_not_emitted():
    game, player, valid = _to_first_play_node()
    pol, alt = valid[0], valid[1]
    # The committee resolves pol > alt — but the live policy ALREADY orders
    # pol over alt by more than the margin (0.6 vs 0.4/(n-1)): counted as
    # satisfied (self-retirement telemetry), not emitted.
    qmap = {a: 0.40 for a in valid}
    qmap[pol] = 0.50
    qmap[alt] = 0.40 - 0.05  # keep alt distinctly below the tied pack too?
    # No: a distinct alt would resolve (tied-action, alt) pairs as well.
    # Keep everything except pol exactly tied so pol-vs-X are the only
    # resolved pairs — all satisfied.
    qmap[alt] = 0.40
    teacher = _ScriptedTeacher([_qres(valid, qmap)] * 3)
    tr, diag = _run_gate(teacher, valid, game, player)
    assert tr["has_search_target"] is False
    assert diag["play"]["resolved_sum"] == len(valid) - 1
    assert diag["play"]["satisfied_sum"] == len(valid) - 1
    assert diag["play"]["accepted"] == 0


def test_pair_cap_keeps_strongest_evidence():
    game, player, valid = _to_first_play_node()
    if len(valid) < 3:
        pytest.skip("need >= 3 legal actions for a multi-pair node")
    # Distinct descending Q with the LIVE argmax (valid[0]) Q-worst: every
    # pair is resolved and unsatisfied, so emission hits the cap when the
    # node is large enough.
    qmap = {a: 0.60 - 0.02 * i for i, a in enumerate(reversed(valid))}
    teacher = _ScriptedTeacher([_qres(valid, qmap)] * 3)
    tr, diag = _run_gate(teacher, valid, game, player)
    n = len(valid)
    expected = min(n * (n - 1) // 2, SearchConfig().gate_max_pairs)
    assert tr["has_search_target"] is True
    assert len(tr["search_pairs"]) == expected
    assert diag["play"]["pairs_sum"] == expected


def test_cell_filter_skips_search_entirely():
    game, player, valid = _to_first_play_node()
    teacher = _ScriptedTeacher([])
    cfg = SearchConfig(gate_node_prob=1.0, gate_cells=frozenset({"t4-picker-lead"}))
    tr, diag = _run_gate(teacher, valid, game, player, cfg)
    assert teacher.calls == 0
    assert tr["has_search_target"] is False
    assert diag["play"]["count"] == 0


def test_gate_serves_jack_of_diamonds_games():
    # The teacher covers BOTH partner modes (operator directive 2026-08-11):
    # eligibility must not filter on partner_mode_flag, and play_cell role
    # detection must work in JD mode (is_secret_partner checks the JD hand).
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
                    alt = valid_sorted[1]
                    qmap = {a: 0.40 for a in valid_sorted}
                    qmap[alt] = 0.50
                    teacher = _ScriptedTeacher([_qres(valid_sorted, qmap)] * 3)
                    tr, diag = _run_gate(teacher, valid_sorted, game, player)
                    assert tr["has_search_target"] is True
                    assert diag["play"]["accepted"] == 1
                    return
                pick = [a for a, n in names.items() if n == "PICK"]
                safe = [a for a, n in names.items() if "ALONE" not in n]
                player.act(pick[0] if pick else (safe[0] if safe else valid_sorted[0]))
                valid = player.get_valid_action_ids()
    raise AssertionError("no standard JD play node reached")


def test_worker_protocol_serves_gated_teacher(tmp_path):
    # Parallel-collection path: an oracle-mode worker built from init_args
    # must load the payload's oracle head + gamma on weight refresh and run
    # gated searches through _league_worker_play (in-process, no pool).
    import torch

    from sheepshead.training import train_league_ppo as tl

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
    # Full checkpoint for the frozen expert (§12.1 stationary teacher).
    teacher_ckpt = str(tmp_path / "teacher_resume.pt")
    main_agent.save(teacher_ckpt)
    tl._league_worker_init(
        {
            "arch": ARCH,
            "members_dir": str(tmp_path),
            "weight_path_base": str(base),
            "base_seed": 0,
            "critic_mode": "oracle",
            "oracle_aux_heads": True,
            "search_teacher": True,
            "search_prob": 1.0,
            "search_replicates": 3,  # test-speed committee
            "search_iters": 8,  # test-speed override
            "teacher_resume": teacher_ckpt,
            "teacher_oracle_init": None,
            "teacher_gamma": 1.0,
        }
    )
    # The frozen-expert build consumes torch RNG, so which deals avoid
    # leaster/ALONE (gate-ineligible) is seed-sensitive: try a few deals
    # until the gate fires.
    out = None
    for ep, seed in enumerate((11, 12, 13, 14, 15), start=1):
        job = tl._Job(
            episode=ep,
            partner_mode=PARTNER_BY_CALLED_ACE,
            training_position=1,
            opponent_ids=[tl.SELF_PLAY] * 4,
            weight_version=1,
            game_seed=seed,
        )
        out = tl._league_worker_play(job)
        if out["training_data_single"]["search_diagnostics"]["play"]["count"]:
            break
    worker = tl._LWORKER["agent"]
    assert worker.gamma == 1.0  # payload gamma reached the live worker agent
    assert worker.oracle_critic is not None
    ref = main_agent.oracle_critic.state_dict()
    got = worker.oracle_critic.state_dict()
    assert all(torch.equal(ref[k], got[k]) for k in ref)
    # Stationary expert: the teacher wraps its OWN frozen agent, not the
    # live worker agent that weight refreshes update.
    frozen = tl._LWORKER["teacher"].agent
    assert frozen is not worker, "teacher must not wrap the live agent"
    assert frozen.gamma == 1.0
    f_ref = frozen.actor.state_dict()
    assert all(torch.equal(main_agent.actor.state_dict()[k], f_ref[k]) for k in f_ref)
    assert out["episode_events"], "worker produced no transitions"
    diag = out["training_data_single"]["search_diagnostics"]["play"]
    assert diag["count"] >= 1, "gate never attempted despite prob=1.0"


def test_gated_mode_end_to_end_smoke():
    seed_all(7)
    agent = PPOAgent(len(ACTIONS), arch=ARCH)
    # Stationary expert (§12.1): the teacher wraps a separate frozen agent.
    expert = PPOAgent(len(ACTIONS), arch=ARCH)

    class _Seat:
        def __init__(self, a):
            self.agent = a
            self.member_id = "stub"

    teacher = ISMCTSTeacher(
        expert,
        ISMCTSConfig(
            iters={"pick": 8, "partner": 8, "bury": 8, "play": 8}, batch_size=4
        ),
    )
    cfg = SearchConfig(
        gate_node_prob=1.0,
        gate_replicates=3,
        gate_d_rollout=1,
        gate_cells=frozenset(
            f"t{t}-{r}-{k}"
            for t in range(5)
            for r in ("picker", "partner", "defender")
            for k in ("lead", "follow")
        ),
    )
    _, events, _, data, _ = play_population_game(
        training_agent=agent,
        opponents=[_Seat(agent) for _ in range(4)],
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
    emitted = sum(1 for t in transitions if t.get("has_search_target"))
    assert diag["accepted"] == emitted
    for t in transitions:
        if t.get("has_search_target"):
            # The pairs (indices + live anchors) must survive event
            # normalization: without them the margin loss no-ops and the
            # label only PG-masks its row (the attempt-5a failure, §10.3).
            pairs = t.get("search_pairs")
            assert pairs, "labeled row lost its pairs in normalization"
            assert len(pairs) <= SearchConfig().gate_max_pairs
            for w, loser, anchor_w, anchor_l in pairs:
                assert int(w) >= 1 and int(loser) >= 1 and int(w) != int(loser)
                assert anchor_w <= 0.0 and anchor_l <= 0.0


def test_margin_loss_gradient_support_and_saturation():
    """The §10.2 math, verified by autograd: while the hinge is active its
    logit gradient is exactly e_l - e_w (all other coordinates zero —
    softmax terms cancel), and once the winner out-ranks the loser by the
    margin the loss and gradient vanish. Contrast forward-KL, whose
    gradient (pi_theta - pi') touches every logit until the full target
    profile is matched (the attempt-3/4 flattening mechanism)."""
    import torch

    m = 0.3
    a_star, a_ref = 3, 7

    logits = torch.zeros(1, 10, requires_grad=True)
    logp = torch.log_softmax(logits, dim=-1)
    loss = (m + logp[0, a_ref] - logp[0, a_star]).clamp(min=0.0)
    loss.backward()
    g = logits.grad[0]
    assert g[a_ref] > 0 and g[a_star] < 0
    others = [i for i in range(10) if i not in (a_star, a_ref)]
    assert torch.allclose(g[others], torch.zeros(len(others)), atol=1e-7)
    assert abs(g[a_ref] - 1.0) < 1e-6 and abs(g[a_star] + 1.0) < 1e-6

    # Saturated: the winner already out-ranks the loser by more than the
    # margin.
    base = torch.zeros(1, 10)
    base[0, a_star] = 1.0
    logits2 = base.clone().requires_grad_(True)
    logp2 = torch.log_softmax(logits2, dim=-1)
    loss2 = (m + logp2[0, a_ref] - logp2[0, a_star]).clamp(min=0.0)
    assert loss2.item() == 0.0
    loss2.backward()
    assert torch.allclose(logits2.grad, torch.zeros_like(logits2))


def test_pair_gap_trust_region_gates_gradient():
    """The §12 gap trust region, verified by autograd. The region is on the
    PAIR GAP log pi(w) - log pi(l), not per leg: while the gap has
    improved less than delta over its label-time anchors the pair carries
    the exact two-logit hinge gradient (softmax terms cancel — support is
    {w, l} only), and past delta the WHOLE pair gates to zero. Per-leg
    clamping was rejected because zeroing one leg leaves the other leg's
    full softmax gradient (e_a - pi on every logit) — an entropy-injection
    direction; the gate preserves the pair or removes it entirely. Zeroing
    (not scaling) is what binds under Adam: second-moment normalization
    re-inflates any scaled-down coherent direction, but cannot step on an
    exactly-zero gradient."""
    import torch

    m, delta = 0.3, 0.2
    a_star, a_ref = 3, 7

    def gated_loss(logits, old_star, old_ref):
        logp = torch.log_softmax(logits, dim=-1)
        gap_gain = (logp[0, a_star] - logp[0, a_ref]).detach() - (old_star - old_ref)
        within = (gap_gain < delta).to(logp.dtype)
        return (m + logp[0, a_ref] - logp[0, a_star]).clamp(min=0.0) * within

    # Anchors at the current values: gap unmoved -> active pair with the
    # exact two-logit gradient (all other coordinates zero).
    logits = torch.zeros(1, 10, requires_grad=True)
    with torch.no_grad():
        lp0 = torch.log_softmax(logits, dim=-1)
    loss = gated_loss(logits, lp0[0, a_star], lp0[0, a_ref])
    assert loss.item() == pytest.approx(m, abs=1e-6)
    loss.backward()
    g = logits.grad[0]
    assert g[a_star] < 0 and g[a_ref] > 0
    others = [i for i in range(10) if i not in (a_star, a_ref)]
    assert torch.allclose(g[others], torch.zeros(len(others)), atol=1e-7)

    # Gap already improved by more than delta since label time: the pair
    # gates to zero — loss AND gradient — regardless of the hinge being
    # nominally unsatisfied. This update can no longer move the pair.
    logits2 = torch.zeros(1, 10, requires_grad=True)
    loss2 = gated_loss(
        logits2, lp0[0, a_star] - 0.5, lp0[0, a_ref] + 0.5
    )  # gap_gain = 1.0 > delta
    assert loss2.item() == 0.0
    loss2.backward()
    assert torch.allclose(logits2.grad, torch.zeros_like(logits2))
