#!/usr/bin/env python3
"""Static incidence probe for the diagnosed defender trump-lead leak.

The first scripted exploit probe from the run-review plan
(League_Run_Review_202607.md §5/§6 step 6b). The diagnosed flaw
(notebooks/defender_trump_lead_investigation.md): a DEFENDER — not picker,
not revealed partner, not secret partner, non-leaster — LEADS trump on trick
0 or 1 while a fail lead is legal. Hindsight-free cost: −0.19 leader game
score per occurrence (belief-pool MC, SE 0.08), concentrated in trump-rich
(3+ trump) hands.

Why incidence, not an exploiting opponent: the tell's *exploitation* EV was
pre-registered at ~0.01–0.05 pts/game at baseline rates — below any
affordable probe's resolution — while the leak's cost is mostly the
self-inflicted EV error of the lead itself. So the regression question "is
the known hole closed, and by how much?" reduces to: how often does the
policy still make the diagnosed lead in a fixed, lineage-free context?

Design:
  * Fixed CRN deal set (seeded); the hero is seated in all 5 seats per deal
    (duplicate replay), the conventions ScriptedAgent fills the field — a
    context that cannot share the RL lineage's blind spots.
  * Games are abandoned after trick 1 (only tricks 0–1 carry the diagnosis),
    so the probe is fast enough to run per-checkpoint.
  * Reported: opportunity count (hero leads t0/t1 as defender with both a
    trump and a fail lead legal), trump-lead count and rate, the trump-rich
    (3+ trump) split where the documented cost lives, and the implied score
    EV per 1000 hero-hands (rate × −0.19).
  * Self-check: --scripted probes the ScriptedAgent itself; its rate is 0 by
    construction, validating the instrument.

Usage:
  PYTHONPATH=. python analysis/trump_lead_probe.py --ckpt final_pfsp_swish_ppo.pt
  PYTHONPATH=. python analysis/trump_lead_probe.py --scripted   # instrument check
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import (
    ACTION_LOOKUP,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP,
    Game,
)

_TRUMP_SET = set(TRUMP)
PROBE_SEED = 20260702  # fixed CRN deal set: results comparable forever
LEAK_COST_SCORE = 0.19  # −score per occurrence (investigation §5, belief-pool MC)


def _is_secret_partner(game: Game, player) -> bool:
    if game.partner_mode_flag == PARTNER_BY_CALLED_ACE:
        return bool(game.called_card) and game.called_card in player.hand
    return not game.alone_called and "JD" in player.hand


def _lead_options(player) -> tuple[list[str], list[str]]:
    """(trump, fail) cards among the player's currently legal PLAY actions."""
    cards = [
        ACTION_LOOKUP[a].split(" ", 1)[1]
        for a in player.get_valid_action_ids()
        if ACTION_LOOKUP[a].startswith("PLAY ")
    ]
    return (
        [c for c in cards if c in _TRUMP_SET],
        [c for c in cards if c not in _TRUMP_SET],
    )


def probe_agent(hero, n_deals: int, partner_mode: int, seed: int = PROBE_SEED) -> dict:
    """Count diagnosed-lead opportunities/occurrences for ``hero`` seated in
    all 5 seats per deal against a ScriptedAgent field. Deterministic."""
    field = ScriptedAgent()
    stats = {
        "opportunities": 0,
        "trump_leads": 0,
        "opportunities_trump_rich": 0,  # 3+ trump in hand at the decision
        "trump_leads_trump_rich": 0,
        "by_trick": {0: [0, 0], 1: [0, 0]},  # trick -> [opportunities, leads]
        "hero_hands": 0,
    }

    for d in range(n_deals):
        deal_seed = seed * 1_000_003 + d
        for hero_seat in range(1, 6):
            game = Game(partner_selection_mode=partner_mode, seed=deal_seed)
            hero.reset_recurrent_state()
            field.reset_recurrent_state()
            stats["hero_hands"] += 1
            # Only tricks 0-1 carry the diagnosis; abandon the game after.
            while not game.is_done() and game.current_trick <= 1:
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
                            and game.current_trick <= 1
                            and game.leader == player.position
                            and game.cards_played == 0
                            and not (
                                player.is_picker
                                or player.is_partner
                                or game.partner == player.position
                                or _is_secret_partner(game, player)
                            )
                        ):
                            trumps, fails = _lead_options(player)
                            if trumps and fails:
                                record = (
                                    game.current_trick,
                                    sum(1 for c in player.hand if c in _TRUMP_SET),
                                )
                        a, _, _ = ag.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=True,
                        )
                        if record is not None:
                            trick, n_trump = record
                            name = ACTION_LOOKUP[a]
                            led_trump = (
                                name.startswith("PLAY ")
                                and name.split(" ", 1)[1] in _TRUMP_SET
                            )
                            stats["opportunities"] += 1
                            stats["by_trick"][trick][0] += 1
                            if n_trump >= 3:
                                stats["opportunities_trump_rich"] += 1
                            if led_trump:
                                stats["trump_leads"] += 1
                                stats["by_trick"][trick][1] += 1
                                if n_trump >= 3:
                                    stats["trump_leads_trump_rich"] += 1
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for p in game.players:
                                ctrl = hero if p.position == hero_seat else field
                                ctrl.observe(
                                    p.get_last_trick_state_dict(), player_id=p.position
                                )
                    if game.is_done() or game.current_trick > 1:
                        break

    opp = stats["opportunities"]
    leads = stats["trump_leads"]
    stats["lead_rate"] = leads / opp if opp else 0.0
    stats["lead_rate_trump_rich"] = (
        stats["trump_leads_trump_rich"] / stats["opportunities_trump_rich"]
        if stats["opportunities_trump_rich"]
        else 0.0
    )
    # Diagnosed-leak EV in score per 1000 hero-hands (documented cost/case).
    stats["implied_ev_per_1000_hands"] = (
        -LEAK_COST_SCORE * 1000.0 * leads / stats["hero_hands"]
    )
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Defender trump-lead incidence probe")
    ap.add_argument("--ckpt", default=None, help="PPO checkpoint to probe")
    ap.add_argument(
        "--scripted",
        action="store_true",
        help="probe the ScriptedAgent itself (instrument self-check; rate must be 0)",
    )
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    if args.scripted:
        hero, label = ScriptedAgent(), "ScriptedAgent"
    elif args.ckpt:
        from sheepshead.agent.ppo import load_agent

        # Arch-aware: constructs whatever architecture the checkpoint records.
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
            f"[{mode_name:>6}] opportunities {r['opportunities']:>5} | "
            f"trump leads {r['trump_leads']:>3} ({100 * r['lead_rate']:.2f}%) | "
            f"trump-rich {r['trump_leads_trump_rich']}/{r['opportunities_trump_rich']} "
            f"({100 * r['lead_rate_trump_rich']:.2f}%) | "
            f"implied EV {r['implied_ev_per_1000_hands']:+.2f} score/1000 hands | "
            f"{time.time() - t0:.0f}s",
            flush=True,
        )
    if args.out_json:
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"wrote {args.out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
