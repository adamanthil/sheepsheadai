"""Agreement-gated search teacher (Search_Teacher_Design §9, 2026-08-11).

Gate mechanics are unit-tested against a scripted teacher (deterministic
committee votes); one smoke test runs the real ISMCTS teacher end-to-end
through play_population_game in gated mode.
"""

import random

import numpy as np

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


def _res(valid, pick, prior_top, ok=True):
    """Canned search result whose pi_gumbel argmax is ``pick`` and whose raw
    root prior argmax is ``prior_top``."""
    gum = np.zeros(len(ACTIONS))
    gum[pick - 1] = 0.7
    for a in valid:
        if a != pick:
            gum[a - 1] = 0.3 / (len(valid) - 1)
    prior = {a: (1.0 if a == prior_top else 0.0) for a in valid}
    return {"ok": ok, "pi_gumbel": gum, "root_prior": prior, "valid": list(valid)}


def _run_gate(teacher, valid, game, player, cfg=None):
    transition = {"search_target": None, "has_search_target": False}
    diag = {"play": {"count": 0, "accepted": 0, "ess_sum": 0.0, "entropy_sum": 0.0}}
    cfg = cfg or SearchConfig(
        mode="gated",
        gate_node_prob=1.0,
        gate_replicates=3,
        gate_agreement=2,
        gate_cells=frozenset({play_cell(game, player)}),
    )
    _attach_gated_search_target(
        game, player, valid, transition, teacher, random.Random(3), cfg, [], diag
    )
    return transition, diag


def test_emits_on_majority_nonpolicy_agreement():
    game, player, valid = _to_first_play_node()
    pol, alt = valid[0], valid[1]
    teacher = _ScriptedTeacher(
        [_res(valid, alt, pol), _res(valid, alt, pol), _res(valid, pol, pol)]
    )
    tr, diag = _run_gate(teacher, valid, game, player)
    assert teacher.calls == 2  # committee early-stop: majority decided
    assert tr["has_search_target"] is True
    target = np.asarray(tr["search_target"])
    assert abs(target.sum() - 1.0) < 1e-9
    # smoothed one-hot on the agreed action (the calibrated semantics)
    assert target.argmax() + 1 == alt
    assert abs(target[alt - 1] - 0.95) < 1e-9
    assert tr["search_ref_action"] == pol  # margin-loss ranking referent
    assert diag["play"]["count"] == 1 and diag["play"]["accepted"] == 1


def test_abstains_when_committee_backs_policy():
    game, player, valid = _to_first_play_node()
    pol = valid[0]
    teacher = _ScriptedTeacher([_res(valid, pol, pol)] * 3)
    tr, diag = _run_gate(teacher, valid, game, player)
    assert teacher.calls == 2  # early-stop applies to policy-backing majorities too
    assert tr["has_search_target"] is False
    assert diag["play"]["count"] == 1 and diag["play"]["accepted"] == 0


def test_abstains_on_split_committee():
    game, player, valid = _to_first_play_node()
    assert len(valid) >= 3, "need 3 distinct picks for a split committee"
    pol = valid[0]
    teacher = _ScriptedTeacher(
        [
            _res(valid, valid[0], pol),
            _res(valid, valid[1], pol),
            _res(valid, valid[2], pol),
        ]
    )
    tr, _ = _run_gate(teacher, valid, game, player)
    assert tr["has_search_target"] is False


def test_cell_filter_skips_search_entirely():
    game, player, valid = _to_first_play_node()
    teacher = _ScriptedTeacher([])
    cfg = SearchConfig(
        mode="gated", gate_node_prob=1.0, gate_cells=frozenset({"t4-picker-lead"})
    )
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
                    pol, alt = valid_sorted[0], valid_sorted[1]
                    teacher = _ScriptedTeacher(
                        [_res(valid_sorted, alt, pol)] * 2
                        + [_res(valid_sorted, pol, pol)]
                    )
                    tr, diag = _run_gate(teacher, valid_sorted, game, player)
                    assert teacher.calls == 2  # early-stop
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
            "search_iters": 8,  # test-speed override
        }
    )
    job = tl._Job(
        episode=1,
        partner_mode=PARTNER_BY_CALLED_ACE,
        training_position=1,
        opponent_ids=[tl.SELF_PLAY] * 4,
        weight_version=1,
        game_seed=11,
    )
    out = tl._league_worker_play(job)
    worker = tl._LWORKER["agent"]
    assert worker.gamma == 1.0  # payload gamma reached the worker teacher
    assert worker.oracle_critic is not None
    ref = main_agent.oracle_critic.state_dict()
    got = worker.oracle_critic.state_dict()
    assert all(torch.equal(ref[k], got[k]) for k in ref)
    assert out["episode_events"], "worker produced no transitions"
    diag = out["training_data_single"]["search_diagnostics"]["play"]
    assert diag["count"] >= 1, "gate never attempted despite prob=1.0"


def test_gated_mode_end_to_end_smoke():
    seed_all(7)
    agent = PPOAgent(len(ACTIONS), arch=ARCH)

    class _Seat:
        def __init__(self, a):
            self.agent = a
            self.member_id = "stub"

    teacher = ISMCTSTeacher(
        agent,
        ISMCTSConfig(
            iters={"pick": 8, "partner": 8, "bury": 8, "play": 8}, batch_size=4
        ),
    )
    cfg = SearchConfig(
        mode="gated",
        gate_node_prob=1.0,
        gate_replicates=2,
        gate_agreement=2,
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
            target = np.asarray(t["search_target"])
            assert target.shape == (len(ACTIONS),)
            assert abs(target.sum() - 1.0) < 1e-6


def test_margin_loss_gradient_support_and_saturation():
    """The §10.2 math, verified by autograd: while the hinge is active its
    logit gradient is exactly e_ref - e_star (all other coordinates zero —
    softmax terms cancel), and once a* out-ranks a_ref by the margin the
    loss and gradient vanish. Contrast forward-KL, whose gradient
    (pi_theta - pi') touches every logit until the full target profile is
    matched (the attempt-3/4 flattening mechanism)."""
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

    # Saturated: a* already out-ranks a_ref by more than the margin.
    base = torch.zeros(1, 10)
    base[0, a_star] = 1.0
    logits2 = base.clone().requires_grad_(True)
    logp2 = torch.log_softmax(logits2, dim=-1)
    loss2 = (m + logp2[0, a_ref] - logp2[0, a_star]).clamp(min=0.0)
    assert loss2.item() == 0.0
    loss2.backward()
    assert torch.allclose(logits2.grad, torch.zeros_like(logits2))
