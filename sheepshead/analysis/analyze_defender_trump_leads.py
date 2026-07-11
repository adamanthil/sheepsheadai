#!/usr/bin/env python3
"""Analyze when the trained agent leads trump as a *defender* (not picker/partner).

Drives all 5 seats with one agent (mirror self-play), maintaining per-seat
recurrent memory exactly like play.py. At every decision where a defender is on
lead, it records the policy's trump-lead probability plus a bag of contextual
features, then plays out a sampled action. Aggregates characterize *when* and
*why* trump leads happen, and whether they look forced, "boss" (guaranteed
trick win), or genuine bleed-out leads.
"""

from __future__ import annotations


# Repo-root imports work regardless of invocation directory.


import argparse
import random
from collections import Counter, defaultdict

import numpy as np

from sheepshead.agent.ppo import load_agent
from sheepshead import (
    ACTIONS,
    TRUMP,
    Game,
    get_card_points,
    get_card_suit,
)
from sheepshead.training.training_utils import (
    compute_seen_trump_mask,
    get_partner_selection_mode,
)


def trump_idx(card: str) -> int:
    return TRUMP.index(card) if card in TRUMP else 99


def unseen_higher_trumps(player, card: str) -> int:
    """How many *unseen* trumps outrank `card` (0 => leading it guarantees the trick now)."""
    if card not in TRUMP:
        return -1
    seen_mask = compute_seen_trump_mask(player)  # 1 = seen, indexed like TRUMP
    ci = TRUMP.index(card)
    return sum(1 for i in range(ci) if seen_mask[i] == 0)


def collect(model_path: str, num_games: int, seed: int, deterministic: bool):
    agent = load_agent(model_path)

    rng = random.Random(seed)
    records = []

    for g in range(num_games):
        partner_mode = get_partner_selection_mode(g)
        game = Game(partner_selection_mode=partner_mode)
        agent.reset_recurrent_state()

        while not game.is_done():
            for player in game.players:
                valid = player.get_valid_action_ids()
                while valid:
                    state = player.get_state_dict()
                    probs, _ = agent.get_action_probs_with_logits(
                        state, valid, player_id=player.position
                    )
                    p = probs[0].detach().cpu().numpy()

                    # Is this a defender lead decision?
                    is_lead = (
                        game.play_started
                        and game.cards_played == 0
                        and game.leader == player.position
                    )
                    is_defender = (
                        not game.is_leaster
                        and not player.is_picker
                        and not player.is_partner
                        and not player.is_secret_partner
                    )

                    if is_lead and is_defender:
                        records.append(
                            _record_lead(game, player, valid, p, deterministic, rng)
                        )

                    # Choose action (sample or argmax over valid set).
                    valid_idx = [a - 1 for a in valid]
                    if deterministic:
                        action = max(valid_idx, key=lambda i: p[i]) + 1
                    else:
                        mass = np.array([p[i] for i in valid_idx])
                        mass = mass / mass.sum() if mass.sum() > 0 else None
                        action = (
                            rng.choices(valid_idx, weights=mass)[0]
                            if mass is not None
                            else rng.choice(valid_idx)
                        ) + 1

                    player.act(action)
                    valid = player.get_valid_action_ids()

                    if game.was_trick_just_completed:
                        for seat in game.players:
                            agent.observe(
                                seat.get_last_trick_state_dict(),
                                player_id=seat.position,
                            )
    return records


def _record_lead(game, player, valid, p, deterministic, rng):
    hand = player.hand
    trump_in_hand = [c for c in hand if c in TRUMP]
    fail_in_hand = [c for c in hand if c not in TRUMP]

    # Valid play cards and trump-lead probability mass.
    play_cards = {}  # card -> prob
    for a in valid:
        name = ACTIONS[a - 1]
        if name.startswith("PLAY "):
            play_cards[name[5:]] = p[a - 1]
    p_trump = sum(pr for c, pr in play_cards.items() if c in TRUMP)

    # Chosen card (mirror the actual play selection logic).
    if deterministic:
        chosen = max(play_cards, key=lambda c: play_cards[c])
    else:
        cards = list(play_cards)
        w = np.array([play_cards[c] for c in cards])
        w = w / w.sum() if w.sum() > 0 else None
        chosen = (
            rng.choices(cards, weights=w)[0] if w is not None else rng.choice(cards)
        )

    picker_pos = game.picker
    # Order in which picker plays this trick relative to leader (0=leads, 4=plays last).
    picker_play_order = (picker_pos - player.position) % 5

    return {
        "trick": game.current_trick,
        "partner_mode": game.partner_mode_flag,
        "n_trump": len(trump_in_hand),
        "n_fail": len(fail_in_hand),
        "n_cards": len(hand),
        "has_fail_option": len(fail_in_hand) > 0,
        "best_trump_idx": min((trump_idx(c) for c in trump_in_hand), default=99),
        "has_QC": "QC" in hand,
        "has_QS": "QS" in hand,
        "n_queens": sum(1 for c in hand if c.startswith("Q")),
        "hand_points": sum(get_card_points(c) for c in hand),
        "n_fail_suits": len({get_card_suit(c) for c in fail_in_hand}),
        "called_suit_played": bool(game.was_called_suit_played),
        "picker_play_order": picker_play_order,
        "p_trump_lead": float(p_trump),
        "chosen": chosen,
        "chosen_is_trump": chosen in TRUMP,
        "chosen_trump_idx": trump_idx(chosen),
        "unseen_higher": unseen_higher_trumps(player, chosen)
        if chosen in TRUMP
        else -1,
        "hand": list(hand),
        "play_probs": play_cards,
    }


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def report(records):
    total = len(records)
    if not total:
        print("No defender-lead decisions captured.")
        return

    trump_leads = [r for r in records if r["chosen_is_trump"]]
    forced = [r for r in records if not r["has_fail_option"]]
    voluntary = [r for r in records if r["has_fail_option"]]
    vol_trump = [r for r in voluntary if r["chosen_is_trump"]]
    vol_fail = [r for r in voluntary if not r["chosen_is_trump"]]

    print("=" * 70)
    print("DEFENDER LEAD ANALYSIS")
    print("=" * 70)
    print(f"Total defender-lead decisions : {total}")
    print(
        f"  Trump led (any)             : {len(trump_leads)} ({pct(len(trump_leads), total):.1f}%)"
    )
    print(
        f"  Forced (hand all trump)     : {len(forced)} ({pct(len(forced), total):.1f}%)"
    )
    print(
        f"  Had a fail option           : {len(voluntary)} ({pct(len(voluntary), total):.1f}%)"
    )
    print(
        f"    -> still led trump (vol.) : {len(vol_trump)} ({pct(len(vol_trump), len(voluntary)):.1f}% of voluntary)"
    )
    print(
        f"  Mean P(trump lead) overall  : {np.mean([r['p_trump_lead'] for r in records]):.3f}"
    )
    print(
        f"  Mean P(trump lead) | fail option avail: {np.mean([r['p_trump_lead'] for r in voluntary]) if voluntary else 0:.3f}"
    )

    # Boss vs bleed among voluntary trump leads.
    if vol_trump:
        boss = [r for r in vol_trump if r["unseen_higher"] == 0]
        bleed = [r for r in vol_trump if r["unseen_higher"] > 0]
        print("\n--- Voluntary trump leads: boss vs bleed ---")
        print(
            f"  Boss (no unseen higher trump -> wins trick): {len(boss)} ({pct(len(boss), len(vol_trump)):.1f}%)"
        )
        print(
            f"  Bleed (a higher trump still out there)     : {len(bleed)} ({pct(len(bleed), len(vol_trump)):.1f}%)"
        )
        print(
            f"  Mean unseen-higher when leading trump      : {np.mean([r['unseen_higher'] for r in vol_trump]):.2f}"
        )

    # The genuinely-questionable bucket: early game, unforced, and a bleed
    # (a higher trump is still out there, so the lead can lose and burns a
    # defender's trump). Endgame boss leads are standard correct play.
    early_bleed = [r for r in vol_trump if r["n_cards"] >= 4 and r["unseen_higher"] > 0]
    print(
        "\n--- 'Questionable' early bleed leads (>=4 cards in hand, higher trump still out) ---"
    )
    print(
        f"  Count: {len(early_bleed)} "
        f"({pct(len(early_bleed), len(voluntary)):.2f}% of all voluntary defender leads)"
    )
    if early_bleed:
        dist = Counter(r["n_trump"] for r in early_bleed)
        print(
            f"  Mean trump in hand for these leads: {np.mean([r['n_trump'] for r in early_bleed]):.2f}"
        )
        print(
            f"  Mean fail in hand for these leads : {np.mean([r['n_fail'] for r in early_bleed]):.2f}"
        )
        print(
            "  n_trump distribution: "
            + ", ".join(f"{k}t:{dist[k]}" for k in sorted(dist))
        )
        trump_rich = sum(1 for r in early_bleed if r["n_trump"] >= 3)
        print(
            f"  From trump-rich hands (>=3 trump): {trump_rich}/{len(early_bleed)} "
            f"({pct(trump_rich, len(early_bleed)):.0f}%)"
        )

    # By hand size (cards remaining) -- endgame vs midgame.
    print("\n--- Voluntary trump-lead rate by cards-in-hand ---")
    by_size = defaultdict(lambda: [0, 0])
    for r in voluntary:
        by_size[r["n_cards"]][1] += 1
        if r["chosen_is_trump"]:
            by_size[r["n_cards"]][0] += 1
    for s in sorted(by_size, reverse=True):
        tl, tot = by_size[s]
        print(f"  {s} cards: {tl}/{tot} ({pct(tl, tot):.1f}%)")

    # By trick.
    print("\n--- Voluntary trump-lead rate by trick ---")
    by_trick = defaultdict(lambda: [0, 0])
    for r in voluntary:
        by_trick[r["trick"]][1] += 1
        if r["chosen_is_trump"]:
            by_trick[r["trick"]][0] += 1
    for t in sorted(by_trick):
        tl, tot = by_trick[t]
        print(f"  Trick {t}: {tl}/{tot} ({pct(tl, tot):.1f}%)")

    # Feature contrast: voluntary trump vs voluntary fail.
    print("\n--- Feature means: voluntary TRUMP-lead vs FAIL-lead ---")
    feats = [
        "n_trump",
        "n_fail",
        "n_fail_suits",
        "hand_points",
        "best_trump_idx",
        "picker_play_order",
        "trick",
    ]
    print(f"  {'feature':<18}{'trump-lead':>12}{'fail-lead':>12}")
    for f in feats:
        a = np.mean([r[f] for r in vol_trump]) if vol_trump else float("nan")
        b = np.mean([r[f] for r in vol_fail]) if vol_fail else float("nan")
        print(f"  {f:<18}{a:>12.2f}{b:>12.2f}")
    for f in ["has_QC", "has_QS", "called_suit_played"]:
        a = pct(sum(r[f] for r in vol_trump), len(vol_trump)) if vol_trump else 0
        b = pct(sum(r[f] for r in vol_fail), len(vol_fail)) if vol_fail else 0
        print(f"  {f:<18}{a:>11.1f}%{b:>11.1f}%")

    # Picker play-order effect on voluntary trump leads.
    print(
        "\n--- Voluntary trump-lead rate by picker play-order (0=picker leads next to me ... 4=picker plays last) ---"
    )
    by_order = defaultdict(lambda: [0, 0])
    for r in voluntary:
        by_order[r["picker_play_order"]][1] += 1
        if r["chosen_is_trump"]:
            by_order[r["picker_play_order"]][0] += 1
    for o in sorted(by_order):
        tl, tot = by_order[o]
        print(f"  order {o}: {tl}/{tot} ({pct(tl, tot):.1f}%)")

    _print_examples(vol_trump)


def _print_examples(vol_trump, n=8):
    print("\n" + "=" * 70)
    print("EXAMPLE VOLUNTARY TRUMP LEADS (highest P(trump), bleed cases first)")
    print("=" * 70)
    ranked = sorted(
        vol_trump,
        key=lambda r: (r["unseen_higher"] == 0, -r["p_trump_lead"]),
    )
    for r in ranked[:n]:
        kind = (
            "BOSS"
            if r["unseen_higher"] == 0
            else f"BLEED(+{r['unseen_higher']} higher out)"
        )
        probs = ", ".join(
            f"{c}:{p:.2f}"
            for c, p in sorted(r["play_probs"].items(), key=lambda x: -x[1])[:6]
        )
        print(
            f"\nTrick {r['trick']} mode={'JD' if r['partner_mode'] == 0 else 'CA'} "
            f"[{kind}] led {r['chosen']} P(trump)={r['p_trump_lead']:.2f}"
        )
        print(
            f"  hand: {' '.join(r['hand'])}  (trump={r['n_trump']} fail={r['n_fail']} pts={r['hand_points']})"
        )
        print(
            f"  picker_play_order={r['picker_play_order']} called_suit_played={r['called_suit_played']}"
        )
        print(f"  top play probs: {probs}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "-m",
        "--model",
        default="pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt",
    )
    ap.add_argument("-n", "--num-games", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--deterministic",
        action="store_true",
        help="Use argmax actions instead of sampling.",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    print(f"Loading {args.model} ...")
    records = collect(args.model, args.num_games, args.seed, args.deterministic)
    print(
        f"Played {args.num_games} games "
        f"({'deterministic' if args.deterministic else 'sampled'}).\n"
    )
    report(records)


if __name__ == "__main__":
    main()
