#!/usr/bin/env python3
"""
Compare lead distributions for a secret partner versus a defender.

This script samples random first-trick states where seat 1 has the lead and
either holds the secret partner card or a non-partner substitute. The only
difference between the paired states is the card swap (JD→JH or called ace→
another fail ace). For each pair we query the PPO policy and print the action
probabilities and logits side-by-side.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Dict, Sequence

from ppo import PPOAgent
from sheepshead import (
    ACTIONS,
    CALLED_ACES,
    DECK,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    Game,
    Player,
)


@dataclass
class Scenario:
    partner_game: Game
    defender_game: Game
    partner_player: Player
    defender_player: Player
    called_card: str | None
    picker_pos: int
    partner_card: str
    defender_swap: str


def _sample_layout(partner_mode: int, rng: random.Random) -> dict:
    """Return hands/blind metadata satisfying partner constraints."""
    attempts = 0
    while True:
        attempts += 1
        deck = DECK[:]
        rng.shuffle(deck)
        hands = [deck[i * 6 : (i + 1) * 6] for i in range(5)]
        blind = deck[30:32]

        if partner_mode == PARTNER_BY_JD:
            if "JD" not in hands[0] or "JH" in hands[0]:
                continue
            partner_card = "JD"
            defender_swap = "JH"
            called_card = None
        else:
            partner_candidates = [card for card in CALLED_ACES if card in hands[0]]
            if not partner_candidates:
                continue
            called_card = rng.choice(partner_candidates)
            replacement_pool = [
                card for card in CALLED_ACES if card != called_card and card not in hands[0]
            ]
            if not replacement_pool:
                continue
            partner_card = called_card
            defender_swap = rng.choice(replacement_pool)

        picker_pos = rng.choice([2, 3, 4, 5])
        return {
            "hands": hands,
            "blind": blind,
            "picker_pos": picker_pos,
            "partner_card": partner_card,
            "defender_swap": defender_swap,
            "called_card": called_card,
        }


def _build_game(
    partner_mode: int,
    hands: Sequence[Sequence[str]],
    blind: Sequence[str],
    picker_pos: int,
    called_card: str | None,
) -> Game:
    """Create a Game snapshot representing the first lead of trick 1."""
    game = Game(partner_selection_mode=partner_mode)
    game.players = []
    for idx, cards in enumerate(hands):
        player = Player(game, idx + 1, list(cards))
        game.players.append(player)

    game.blind = list(blind)
    game.bury = list(blind)
    game.picker = picker_pos
    game.partner = 0
    game.last_passed = (picker_pos - 1) or 5
    game.leader = 1
    game.leaders = [0] * 6
    game.leaders[0] = 1
    game.play_started = True
    game.current_trick = 0
    game.cards_played = 0
    game.current_suit = ""
    game.was_trick_just_completed = False
    game.last_player = 0
    game.was_called_suit_played = False
    game.is_called_under = False
    game.under_card = None
    game.called_card = called_card
    game.picker_chose_partner = True
    game.alone_called = False
    game.history = [[""] * 5 for _ in range(6)]
    game.trick_points = [0] * 6
    game.trick_winners = [0] * 6
    game.points_taken = [0] * 5
    return game


def build_scenario(partner_mode: int, rng: random.Random) -> Scenario:
    """Create paired partner and defender games with shared context."""
    layout = _sample_layout(partner_mode, rng)
    hands_partner = [list(hand) for hand in layout["hands"]]
    hands_defender = [list(hand) for hand in layout["hands"]]

    # Swap the partner card for the defender substitute.
    defender_hand = hands_defender[0]
    swap_index = defender_hand.index(layout["partner_card"])
    defender_hand[swap_index] = layout["defender_swap"]

    partner_game = _build_game(
        partner_mode,
        hands_partner,
        layout["blind"],
        layout["picker_pos"],
        layout["called_card"],
    )
    defender_game = _build_game(
        partner_mode,
        hands_defender,
        layout["blind"],
        layout["picker_pos"],
        layout["called_card"],
    )

    return Scenario(
        partner_game=partner_game,
        defender_game=defender_game,
        partner_player=partner_game.players[0],
        defender_player=defender_game.players[0],
        called_card=layout["called_card"],
        picker_pos=layout["picker_pos"],
        partner_card=layout["partner_card"],
        defender_swap=layout["defender_swap"],
    )


def evaluate_state(agent: PPOAgent, player: Player) -> Dict[str, Dict[str, float]]:
    """Return per-action probability/logit data for the player's lead."""
    state = player.get_state_dict()
    valid_actions = player.get_valid_action_ids()
    probs, logits = agent.get_action_probs_with_logits(state, valid_actions, player_id=None)
    probs_v = probs[0].detach().cpu().numpy()
    logits_v = logits[0].detach().cpu().numpy()

    metrics: Dict[str, Dict[str, float]] = {}
    for action_id in sorted(valid_actions):
        name = ACTIONS[action_id - 1]
        metrics[name] = {
            "prob": float(probs_v[action_id - 1]),
            "logit": float(logits_v[action_id - 1]),
        }
    return metrics


def format_hand(hand: Sequence[str]) -> str:
    return " ".join(hand)


def print_comparison(index: int, scenario: Scenario, partner_stats, defender_stats) -> Dict[str, float]:
    """Pretty-print a single scenario comparison and return summary stats."""
    title = f"Scenario {index}: {'Jack-of-Diamonds' if scenario.called_card is None else f'Called Ace ({scenario.called_card})'}"
    print(f"\n{'=' * len(title)}\n{title}\n{'=' * len(title)}")
    print(f"Picker seat: {scenario.picker_pos}")
    print(f"Partner card: {scenario.partner_card}  |  Defender swap: {scenario.defender_swap}")
    if scenario.called_card:
        print(f"Called card in play: {scenario.called_card}")
    print(f"Partner hand : {format_hand(scenario.partner_player.hand)}")
    print(f"Defender hand: {format_hand(scenario.defender_player.hand)}\n")

    header = (
        f"{'Action':<12} {'Partner%':>10} {'Defender%':>10} {'Δ%':>8} "
        f"{'PartnerLogit':>14} {'DefenderLogit':>14} {'ΔLogit':>10}"
    )
    print(header)
    print("-" * len(header))

    action_names = sorted(set(partner_stats) | set(defender_stats))
    overlap_diffs = []
    for name in action_names:
        partner = partner_stats.get(name)
        defender = defender_stats.get(name)
        p_prob = partner["prob"] if partner else None
        d_prob = defender["prob"] if defender else None
        p_logit = partner["logit"] if partner else None
        d_logit = defender["logit"] if defender else None

        if p_prob is not None and d_prob is not None:
            overlap_diffs.append(abs(p_prob - d_prob))

        def fmt(value: float | None, scale: float = 1.0) -> str:
            if value is None:
                return " " * 10 + "--"
            return f"{value * scale:10.3f}"

        delta_prob = (
            f"{(p_prob - d_prob) * 100:8.3f}" if p_prob is not None and d_prob is not None else " " * 8 + "--"
        )
        delta_logit = (
            f"{(p_logit - d_logit):10.3f}" if p_logit is not None and d_logit is not None else " " * 10 + "--"
        )

        print(
            f"{name:<12}"
            f"{fmt(p_prob, 100):>10}"
            f"{fmt(d_prob, 100):>10}"
            f"{delta_prob:>10}"
            f"{fmt(p_logit):>14}"
            f"{fmt(d_logit):>14}"
            f"{delta_logit:>10}"
        )

    abs_avg = sum(overlap_diffs) / len(overlap_diffs) if overlap_diffs else 0.0
    abs_max = max(overlap_diffs) if overlap_diffs else 0.0
    print(
        f"\nOverlapping actions: {len(overlap_diffs)} "
        f"| mean |Δprob|: {abs_avg * 100:5.2f} pp "
        f"| max |Δprob|: {abs_max * 100:5.2f} pp"
    )
    return {"mean_abs": abs_avg, "max_abs": abs_max}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare lead distributions for secret partners vs defenders."
    )
    parser.add_argument(
        "-m",
        "--model",
        default="final_swish_ppo.pt",
        help="Path to the trained PPO checkpoint.",
    )
    parser.add_argument(
        "--partner-mode",
        choices=["called-ace", "jd"],
        default="called-ace",
        help="Partner selection rules to evaluate.",
    )
    parser.add_argument(
        "-n",
        "--samples",
        type=int,
        default=5,
        help="Number of random paired scenarios to evaluate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    partner_mode = PARTNER_BY_JD if args.partner_mode == "jd" else PARTNER_BY_CALLED_ACE
    rng = random.Random(args.seed)

    agent = PPOAgent(len(ACTIONS), activation="swish")
    try:
        agent.load(args.model, load_optimizers=False)
    except FileNotFoundError as exc:
        print(f"Model not found: {exc}")
        return 1
    agent.reset_recurrent_state()

    summaries = []
    for idx in range(1, args.samples + 1):
        scenario = build_scenario(partner_mode, rng)
        partner_stats = evaluate_state(agent, scenario.partner_player)
        defender_stats = evaluate_state(agent, scenario.defender_player)
        summary = print_comparison(idx, scenario, partner_stats, defender_stats)
        summaries.append(summary)

    if summaries:
        mean_mean = sum(s["mean_abs"] for s in summaries) / len(summaries)
        max_max = max((s["max_abs"] for s in summaries), default=0.0)
        print(
            f"\nOverall mean |Δprob| across scenarios: {mean_mean * 100:5.2f} pp | "
            f"Max |Δprob| observed: {max_max * 100:5.2f} pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

