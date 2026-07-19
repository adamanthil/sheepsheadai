#!/usr/bin/env python3
"""Secret-partner trump-lead CONVENTION probe (the good behavior).

Mirror of trump_lead_probe.py, which measures the diagnosed DEFENDER
trump-lead leak (bad). This one measures the convention on the other side:
a SECRET partner — holds the called card / JD, not yet revealed — leading
trump when they have the lead on tricks 0-2 and a fail lead is legal.
Leading trump toward the picker is the standard partner convention; the
scripted conventions agent does it by design, so --scripted doubles as the
instrument's high-anchor self-check (defender probe's is the zero-anchor).

Motivation (2026-07-19): operator observed the stage-1 v2 league agent
reliably leading trump as secret partner near 2M episodes but not a few
hundred k earlier. This probe timestamps that switch across a checkpoint
ladder — representation/behavior gains the strength panels can't see.

Design matches trump_lead_probe.py: fixed CRN deal set, hero seated in all
5 seats per deal, ScriptedAgent field (lineage-free context), games
abandoned after trick 2, deterministic play.

Usage:
  PYTHONPATH=. python -m sheepshead.analysis.partner_trump_lead_probe \
    --ckpt <checkpoint.pt> [--deals 500] [--out-json out.json]
  PYTHONPATH=. python -m sheepshead.analysis.partner_trump_lead_probe --scripted
"""

from __future__ import annotations


import argparse
import json
import time

from sheepshead.analysis.trump_lead_probe import _is_secret_partner, _lead_options
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import (
    ACTION_LOOKUP,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP_SET,
    Game,
)

PROBE_SEED = 20260719  # fixed CRN deal set: results comparable forever
MAX_TRICK = 2  # partner conventions fire when the partner first gets the lead


def probe_agent(hero, n_deals: int, partner_mode: int, seed: int = PROBE_SEED) -> dict:
    """Count secret-partner lead opportunities/trump-leads for ``hero`` seated
    in all 5 seats per deal against a ScriptedAgent field. Deterministic."""
    field = ScriptedAgent()
    stats = {
        "opportunities": 0,
        "trump_leads": 0,
        "by_trick": {t: [0, 0] for t in range(MAX_TRICK + 1)},
        "hero_hands": 0,
    }

    for d in range(n_deals):
        deal_seed = seed * 1_000_003 + d
        for hero_seat in range(1, 6):
            game = Game(partner_selection_mode=partner_mode, seed=deal_seed)
            hero.reset_recurrent_state()
            field.reset_recurrent_state()
            stats["hero_hands"] += 1
            while not game.is_done() and game.current_trick <= MAX_TRICK:
                for player in game.players:
                    valid = player.get_valid_action_ids()
                    while valid:
                        is_hero = player.position == hero_seat
                        ag = hero if is_hero else field
                        record = None
                        if (
                            is_hero
                            and game.play_started
                            and not game.is_leaster
                            and game.current_trick <= MAX_TRICK
                            and game.leader == player.position
                            and game.cards_played == 0
                            and not player.is_picker
                            and game.partner != player.position
                            and _is_secret_partner(game, player)
                        ):
                            trumps, fails = _lead_options(player)
                            if trumps and fails:
                                record = game.current_trick
                        a, _, _ = ag.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=True,
                        )
                        if record is not None:
                            name = ACTION_LOOKUP[a]
                            led_trump = (
                                name.startswith("PLAY ")
                                and name.split(" ", 1)[1] in TRUMP_SET
                            )
                            stats["opportunities"] += 1
                            stats["by_trick"][record][0] += 1
                            if led_trump:
                                stats["trump_leads"] += 1
                                stats["by_trick"][record][1] += 1
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for p in game.players:
                                ctrl = hero if p.position == hero_seat else field
                                ctrl.observe(
                                    p.get_last_trick_state_dict(), player_id=p.position
                                )
                    if game.is_done() or game.current_trick > MAX_TRICK:
                        break

    opp = stats["opportunities"]
    stats["lead_rate"] = stats["trump_leads"] / opp if opp else 0.0
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Secret-partner trump-lead probe")
    ap.add_argument("--ckpt", default=None, help="PPO checkpoint to probe")
    ap.add_argument(
        "--scripted",
        action="store_true",
        help="probe the ScriptedAgent itself (convention high-anchor self-check)",
    )
    ap.add_argument("--deals", type=int, default=500)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.scripted:
        hero, label = ScriptedAgent(), "ScriptedAgent"
    elif args.ckpt:
        from sheepshead.agent.ppo import load_agent

        hero = load_agent(args.ckpt)
        label = args.ckpt
    else:
        ap.error("provide --ckpt or --scripted")

    results = {"probe_seed": args.seed, "deals": args.deals, "hero": label}
    for mode, mode_name in ((PARTNER_BY_JD, "jd"), (PARTNER_BY_CALLED_ACE, "called")):
        t0 = time.time()
        r = probe_agent(hero, args.deals, mode, seed=args.seed)
        results[mode_name] = r
        print(
            f"[{mode_name}] opportunities={r['opportunities']} "
            f"lead_rate={r['lead_rate']:.3f} "
            f"by_trick={ {t: v for t, v in r['by_trick'].items()} } "
            f"({time.time() - t0:.0f}s)",
            flush=True,
        )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
