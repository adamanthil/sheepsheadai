#!/usr/bin/env python3
"""Static incidence probe for defender called-suit-lead adherence (convention C2).

Companion to ``trump_lead_probe.py`` (same CRN deal set, same hero-in-all-seats
duplicate replay against a conventions ScriptedAgent field), measuring the
second human convention from notebooks/Convention_Optimality_202607.md: a
DEFENDER holding a called-suit fail leads it while the called suit has not yet
been led. The picker is rule-guaranteed to follow, the secret partner must
surface the called card, so the lead identifies the partner and offers a void
defender the 11-point ace.

An ELIGIBLE node is a hero lead (non-leaster, non-alone, called-ace mode, hero
is a true defender) with a legal called-suit fail lead AND at least one legal
non-called-suit lead (a forced all-called-suit hand is not a decision) while
``game.was_called_suit_played`` is False. ADHERENT = the led card is a
called-suit fail. Under-call hands (picker void in the called suit by rule)
are tallied separately — the "picker must follow" premise does not hold there.

Games are abandoned once the called suit has been led (no further eligibility
is possible), keeping the probe fast enough to run per-checkpoint.

Self-checks: the ScriptedAgent leads the called suit through on trick 0 by
construction, so its trick-0 adherence is exactly 1.0 (see tests).

Usage:
  uv run python -m sheepshead.analysis.called_suit_probe --ckpt final_pfsp_swish_ppo.pt
  uv run python -m sheepshead.analysis.called_suit_probe --scripted
"""

from __future__ import annotations

import argparse
import json
import time

from sheepshead.analysis.trump_lead_probe import PROBE_SEED, _is_secret_partner
from sheepshead.scripted_agent import ScriptedAgent
from sheepshead import (
    ACTION_LOOKUP,
    FAIL,
    PARTNER_BY_CALLED_ACE,
    Game,
)

_FAIL_SET = set(FAIL)


def _called_suit_fail(card: str, called_card: str) -> bool:
    """True when ``card`` is a fail of the called card's suit (the suit letter
    is the last character for all fail cards; QC/JC etc. are trump)."""
    return card in _FAIL_SET and card[-1] == called_card[-1]


def _legal_lead_cards(player) -> list[str]:
    return [
        ACTION_LOOKUP[a].split(" ", 1)[1]
        for a in player.get_valid_action_ids()
        if ACTION_LOOKUP[a].startswith("PLAY ")
    ]


def probe_agent(hero, n_deals: int, seed: int = PROBE_SEED) -> dict:
    """Count C2-eligible hero leads and adherence, hero seated in all 5 seats
    per deal against a ScriptedAgent field. Called-ace mode; deterministic."""
    field = ScriptedAgent()
    stats = {
        "hero_hands": 0,
        "eligible": 0,
        "adherent": 0,
        "eligible_trick0": 0,
        "adherent_trick0": 0,
        "eligible_first_opp": 0,
        "adherent_first_opp": 0,
        "eligible_under": 0,
        "adherent_under": 0,
        # trick -> [eligible, adherent] (primary, i.e. non-under)
        "by_trick": {t: [0, 0] for t in range(6)},
        # (seat - picker) % 5 -> [eligible, adherent] (primary)
        "by_rel_pos": {k: [0, 0] for k in (1, 2, 3, 4)},
    }

    for d in range(n_deals):
        deal_seed = seed * 1_000_003 + d
        for hero_seat in range(1, 6):
            game = Game(
                partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=deal_seed
            )
            hero.reset_recurrent_state()
            field.reset_recurrent_state()
            stats["hero_hands"] += 1
            hero_had_opportunity = False
            abandoned = False
            while not game.is_done() and not abandoned:
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
                            and not game.alone_called
                            and game.called_card
                            and not game.was_called_suit_played
                            and game.leader == player.position
                            and game.cards_played == 0
                            and not (
                                player.is_picker
                                or player.is_partner
                                or game.partner == player.position
                                or _is_secret_partner(game, player)
                            )
                        ):
                            leads = _legal_lead_cards(player)
                            called_fails = [
                                c
                                for c in leads
                                if _called_suit_fail(c, game.called_card)
                            ]
                            if called_fails and len(called_fails) < len(leads):
                                record = (
                                    game.current_trick,
                                    (player.position - game.picker) % 5,
                                    bool(game.is_called_under),
                                    not hero_had_opportunity,
                                )
                                hero_had_opportunity = True
                        a, _, _ = ag.act(
                            player.get_state_dict(),
                            valid,
                            player.position,
                            deterministic=True,
                        )
                        if record is not None:
                            trick, rel_pos, under, first_opp = record
                            name = ACTION_LOOKUP[a]
                            adhered = name.startswith("PLAY ") and _called_suit_fail(
                                name.split(" ", 1)[1], game.called_card
                            )
                            if under:
                                stats["eligible_under"] += 1
                                stats["adherent_under"] += int(adhered)
                            else:
                                stats["eligible"] += 1
                                stats["adherent"] += int(adhered)
                                stats["by_trick"][trick][0] += 1
                                stats["by_trick"][trick][1] += int(adhered)
                                stats["by_rel_pos"][rel_pos][0] += 1
                                stats["by_rel_pos"][rel_pos][1] += int(adhered)
                                if trick == 0:
                                    stats["eligible_trick0"] += 1
                                    stats["adherent_trick0"] += int(adhered)
                                if first_opp:
                                    stats["eligible_first_opp"] += 1
                                    stats["adherent_first_opp"] += int(adhered)
                        player.act(a)
                        valid = player.get_valid_action_ids()
                        if game.was_trick_just_completed:
                            for p in game.players:
                                ctrl = hero if p.position == hero_seat else field
                                ctrl.observe(
                                    p.get_last_trick_state_dict(), player_id=p.position
                                )
                        # No further eligibility once the called suit has been
                        # led (or the hand has no called-ace structure at all).
                        if game.play_started and (
                            game.was_called_suit_played
                            or game.is_leaster
                            or game.alone_called
                            or not game.called_card
                        ):
                            abandoned = True
                            break
                    if game.is_done() or abandoned:
                        break

    for num, den, key in (
        ("adherent", "eligible", "adherence_rate"),
        ("adherent_trick0", "eligible_trick0", "adherence_rate_trick0"),
        ("adherent_first_opp", "eligible_first_opp", "adherence_rate_first_opp"),
        ("adherent_under", "eligible_under", "adherence_rate_under"),
    ):
        stats[key] = stats[num] / stats[den] if stats[den] else 0.0
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description="Defender called-suit-lead adherence probe")
    ap.add_argument("--ckpt", default=None, help="PPO checkpoint to probe")
    ap.add_argument(
        "--scripted",
        action="store_true",
        help="probe the ScriptedAgent itself (trick-0 adherence must be 1.0)",
    )
    ap.add_argument("--deals", type=int, default=2000)
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

    t0 = time.time()
    r = probe_agent(hero, args.deals, seed=args.seed)
    results = {"probe_seed": args.seed, "deals": args.deals, "hero": label, "called": r}
    print(
        f"[called] eligible {r['eligible']:>5} | "
        f"adherent {r['adherent']:>5} ({100 * r['adherence_rate']:.2f}%) | "
        f"trick0 {r['adherent_trick0']}/{r['eligible_trick0']} "
        f"({100 * r['adherence_rate_trick0']:.2f}%) | "
        f"first-opp {100 * r['adherence_rate_first_opp']:.2f}% | "
        f"under {r['adherent_under']}/{r['eligible_under']} | "
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
