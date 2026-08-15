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
    assert teacher.calls == 3
    assert tr["has_search_target"] is True
    target = np.asarray(tr["search_target"])
    assert abs(target.sum() - 1.0) < 1e-9
    # averaged mass concentrates on the agreed action
    assert target.argmax() + 1 == alt
    assert diag["play"]["count"] == 1 and diag["play"]["accepted"] == 1


def test_abstains_when_committee_backs_policy():
    game, player, valid = _to_first_play_node()
    pol = valid[0]
    teacher = _ScriptedTeacher([_res(valid, pol, pol)] * 3)
    tr, diag = _run_gate(teacher, valid, game, player)
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
                    assert teacher.calls == 3
                    assert tr["has_search_target"] is True
                    assert diag["play"]["accepted"] == 1
                    return
                pick = [a for a, n in names.items() if n == "PICK"]
                safe = [a for a, n in names.items() if "ALONE" not in n]
                player.act(pick[0] if pick else (safe[0] if safe else valid_sorted[0]))
                valid = player.get_valid_action_ids()
    raise AssertionError("no standard JD play node reached")


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
