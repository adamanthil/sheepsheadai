#!/usr/bin/env python3
"""Lockstep committee search (``ISMCTSTeacher.search_committee``) equivalence
and contract tests.

The committee driver merges R replicate searches' network rounds into shared
batches while swapping per-replicate transient state (rng / tree / Q-span /
readout accumulators) around every non-network step. The load-bearing claims,
each pinned here:

  1. R=1 lockstep is BIT-EQUAL to a serial ``search`` with the same rng —
     with one replicate the merged batches are the serial batches, so any
     divergence would mean the state swap itself changed semantics.
  2. R=3 lockstep matches per-seed serial searches up to merged-batch float
     tiling: identical pools/ESS (pool build is per-replicate serial),
     near-identical root Q, and consistent visit argmax away from ties.
  3. Replicates are independent (distinct rngs -> distinct visit tables) and
     the SearchResult contract holds per replicate.
  4. The committee path is deterministic given seeds.

Model-free (untrained agent) and tiny budgets, per the ismcts test convention.
"""

from __future__ import annotations

import random
import sys

import numpy as np
import pytest
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, Game
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher, is_private_action

pytestmark = pytest.mark.slow

SEED = 4242
HEADS = ("pick", "partner", "bury", "play")


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _fresh_agent():
    from sheepshead.agent.ppo import PPOAgent

    return PPOAgent(len(ACTIONS))


def _tiny_config(iters=48):
    return ISMCTSConfig(
        iters={h: iters for h in HEADS},
        det_max_tries=2000,
    )


def _to_play_node(game_seed=11):
    """Drive a game to the first standard-game PLAY decision with scripted
    bidding (PICK when offered, never ALONE), tracking the public action
    record the teacher's forced replay needs. Returns
    (game, observer, forced_public)."""
    game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=game_seed)
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                valid_sorted = sorted(valid)
                names = {a: ACTIONS[a - 1] for a in valid_sorted}
                play = [a for a, n in names.items() if n.startswith("PLAY ")]
                if play and not game.is_leaster and not game.alone_called:
                    return game, player.position, forced_public
                pick = [a for a, n in names.items() if n == "PICK"]
                safe = [a for a, n in names.items() if "ALONE" not in n]
                action = pick[0] if pick else (safe[0] if safe else valid_sorted[0])
                if not is_private_action(action):
                    forced_public.append((player.position, action))
                player.act(action)
                valid = player.get_valid_action_ids()
    raise AssertionError("no standard play node reached")


def test_committee_r1_bitexact_vs_serial():
    """One-replicate lockstep must reproduce serial search exactly: the
    batches are identical, so only a state-swap bug could differ."""
    _seed()
    torch.set_num_threads(1)
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, _tiny_config())
    game, observer, fp = _to_play_node()

    serial = teacher.search(game, observer, list(fp), random.Random(5), d_rollout=1)
    lock = teacher.search_committee(
        game, observer, list(fp), [random.Random(5)], d_rollout=1
    )[0]

    assert lock["valid"] == serial["valid"]
    assert lock["head"] == serial["head"]
    assert lock["ess"] == serial["ess"]
    assert lock["ok"] == serial["ok"]
    assert lock["n_iter"] == serial["n_iter"]
    assert lock["root_n"] == serial["root_n"]
    assert lock["root_q"] == serial["root_q"]
    assert lock["root_prior"] == serial["root_prior"]
    assert np.array_equal(lock["pi"], serial["pi"])
    if serial["pi_gumbel"] is None:
        assert lock["pi_gumbel"] is None
    else:
        l_gumbel, s_gumbel = lock["pi_gumbel"], serial["pi_gumbel"]
    assert l_gumbel is not None and s_gumbel is not None
    assert np.array_equal(l_gumbel, s_gumbel)


def test_committee_r3_matches_per_seed_serial():
    """Three-replicate lockstep vs three serial searches with the same seeds:
    pools/ESS identical (pool build is per-replicate serial and rng streams
    match); tree-phase outputs equal up to merged-batch float tiling."""
    _seed()
    torch.set_num_threads(1)
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, _tiny_config())
    game, observer, fp = _to_play_node()
    seeds = (11, 22, 33)

    serials = [
        teacher.search(game, observer, list(fp), random.Random(s), d_rollout=1)
        for s in seeds
    ]
    locks = teacher.search_committee(
        game, observer, list(fp), [random.Random(s) for s in seeds], d_rollout=1
    )

    for serial, lock in zip(serials, locks):
        assert lock["valid"] == serial["valid"]
        assert lock["ess"] == serial["ess"], "pool build diverged from serial"
        assert lock["n_iter"] == serial["n_iter"]
        for a in serial["valid"]:
            assert lock["root_q"][a] == pytest.approx(serial["root_q"][a], abs=5e-3), (
                f"root_q diverged at action {a}"
            )
        # Visit argmax must agree unless the top two are Q-near-tied (a
        # single tiling-flipped tie-break can reroute visits at a tie).
        s_n, l_n = serial["root_n"], lock["root_n"]
        s_top = max(s_n, key=lambda a: s_n[a])
        l_top = max(l_n, key=lambda a: l_n[a])
        if s_top != l_top:
            gap = abs(serial["root_q"][s_top] - serial["root_q"][l_top])
            assert gap < 5e-3, (
                f"visit argmax flipped across a non-tie (gap {gap:.4f}): "
                f"serial {s_top} vs lockstep {l_top}"
            )


def test_committee_replicates_independent_and_contract():
    """Distinct rngs must produce distinct searches (no accidental rng
    sharing), and every replicate honors the SearchResult contract."""
    _seed()
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, _tiny_config())
    game, observer, fp = _to_play_node()

    results = teacher.search_committee(
        game, observer, list(fp), [random.Random(s) for s in (1, 2, 3)], d_rollout=1
    )
    assert len(results) == 3
    tables = []
    for res in results:
        assert set(res.keys()) >= {
            "pi",
            "ess",
            "ok",
            "head",
            "n_iter",
            "valid",
            "root_n",
            "root_q",
            "root_prior",
            "pi_gumbel",
            "pi_rm",
        }
        assert res["head"] == "play"
        mass = res["pi"][[a - 1 for a in res["valid"]]].sum()
        assert mass == pytest.approx(1.0, abs=1e-5)
        assert sum(res["root_n"].values()) > 0
        tables.append(tuple(sorted(res["root_n"].items())))
    assert len(set(tables)) > 1, "replicates with distinct rngs were identical"


def test_committee_deterministic():
    """Same seeds -> identical committee output (within-process determinism,
    threads pinned)."""
    _seed()
    torch.set_num_threads(1)
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, _tiny_config(iters=32))
    game, observer, fp = _to_play_node()

    first = teacher.search_committee(
        game, observer, list(fp), [random.Random(s) for s in (7, 8)], d_rollout=1
    )
    second = teacher.search_committee(
        game, observer, list(fp), [random.Random(s) for s in (7, 8)], d_rollout=1
    )
    for a, b in zip(first, second):
        assert a["root_n"] == b["root_n"]
        assert a["root_q"] == b["root_q"]


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failures else 0)
