"""Paired-deal evaluation: play the same deal with the probe in every seat
against a fixed field, so the deal's luck cancels out of the edge.
"""

from typing import Dict

import numpy as np

from sheepshead.agent.observation import (
    last_trick_observation_for,
    observation_for,
)
from sheepshead.game import (
    Game,
    get_partner_selection_mode,
)


def play_paired_deal(deal_seed: int, mode, seat: int, probe, field) -> float:
    """One greedy deal; ``probe`` plays the probe seat, ``field`` the rest.
    Returns the probe seat's final score. ``probe`` and ``field`` may be the
    same agent instance (recurrent memory is keyed by player_id)."""
    game = Game(partner_selection_mode=mode, seed=deal_seed)
    probe.reset_recurrent_state()
    field.reset_recurrent_state()
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                ag = probe if player.position == seat else field
                a, _, _ = ag.act(
                    observation_for(player, ag),
                    valid,
                    player.position,
                    deterministic=True,
                )
                player.act(a)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for p in game.players:
                        ctrl = probe if p.position == seat else field
                        ctrl.observe(
                            last_trick_observation_for(p, ctrl), player_id=p.position
                        )
    return float(game.players[seat - 1].get_score())


def paired_edge(
    challenger,
    incumbent,
    field,
    n_deals: int,
    seed: int = 0,
    log_every: int = 100,
) -> Dict:
    """Paired duplicate-deal edge of ``challenger`` over ``incumbent`` against
    the same greedy ``field``: mean (challenger − incumbent) score from the
    same seat on the same deal. Returns edge/SE/win_frac/n. A fixed ``seed``
    fixes the deal set, so a series of calls (e.g. the trainer's anchored
    probe) is itself paired across calls."""
    import time as _time

    deltas, wins = [], 0.0
    t0 = _time.time()
    for d in range(n_deals):
        mode = get_partner_selection_mode(d)
        seat = (d % 5) + 1
        deal_seed = seed * 1_000_003 + d
        s_inc = play_paired_deal(deal_seed, mode, seat, incumbent, field)
        s_chal = play_paired_deal(deal_seed, mode, seat, challenger, field)
        delta = s_chal - s_inc
        deltas.append(delta)
        wins += 1.0 if delta > 0 else 0.5 if delta == 0 else 0.0
        if log_every and (d + 1) % log_every == 0:
            arr = np.array(deltas)
            print(
                f"  gate {d + 1}/{n_deals} ({_time.time() - t0:.0f}s)  "
                f"edge {arr.mean():+.3f} +/- {arr.std(ddof=1) / np.sqrt(len(arr)):.3f}",
                flush=True,
            )
    arr = np.array(deltas)
    n = len(arr)
    return {
        "edge": float(arr.mean()),
        "se": float(arr.std(ddof=1) / np.sqrt(n)),
        "win_frac": float(wins / n),
        "n_deals": n,
        "deviating_frac": float(np.mean(arr != 0.0)),
    }
