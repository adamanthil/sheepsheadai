#!/usr/bin/env python3
"""
Tournament evaluator for Sheepshead PPO snapshots using OpenSkill ratings.

Features:
- Discovers snapshot .pt files in an input directory
- Filters to snapshots with episode markers divisible by 100,000
- Loads each snapshot into a PPOAgent
- Runs an extensive multi-round tournament of 5-player games
- Uses deterministic deals: for each 5-agent group, generates a fixed number of unique
  deals (controlled by --deals-per-round). Each deal is reused across all 5 seat rotations,
  ensuring fair comparison of agent skill across positions with identical card distributions.
- Updates OpenSkill (Plackett-Luce) ratings from per-game scores
- Writes a CSV of agent stats (id, episodes, mu, sigma, games, avg_score)
- Generates a PNG plot of rating (mu) vs training episodes

Example:
  uv run python tournament_eval.py \
    --input-dir checkpoints_swish \
    --partner-mode called \
    --rounds 200 \
    --deals-per-round 10 \
    --out-csv tournament_results.csv \
    --out-plot tournament_ratings.png
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openskill.models import PlackettLuce
import matplotlib.pyplot as plt

from ppo import PPOAgent
from sheepshead import (
    Game,
    PARTNER_BY_JD,
    PARTNER_BY_CALLED_ACE,
    ACTIONS,
)


@dataclass
class EvalAgent:
    """Lightweight wrapper for a loaded snapshot and its rating/stats."""
    agent_id: str
    filepath: Path
    episodes: int
    agent: PPOAgent
    rating_mu: float
    rating_sigma: float
    games_played: int = 0
    total_score: float = 0.0

    def update_rating(self, mu: float, sigma: float) -> None:
        self.rating_mu = float(mu)
        self.rating_sigma = float(sigma)

    def add_game_result(self, score: float) -> None:
        self.games_played += 1
        self.total_score += float(score)

    @property
    def avg_score(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.total_score / self.games_played


def extract_episodes_from_name(path: Path) -> Optional[int]:
    """Extract episode count from filename.

    Attempts patterns such as:
      - *checkpoint_12345.pt
      - *_12345.pt
    Returns None if no integer suffix looks like an episode marker.
    """
    name = path.name

    # Prefer explicit "checkpoint_XXXXX" pattern
    m = re.search(r"checkpoint_(\d+)", name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None

    # Fallback: last integer group before extension
    m2 = re.search(r"(\d+)(?:\.pt)$", name)
    if m2:
        try:
            return int(m2.group(1))
        except ValueError:
            return None

    return None


def discover_snapshots(input_dir: Path) -> List[Path]:
    """Recursively find .pt snapshot files in the input directory."""
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    return [p for p in input_dir.rglob("*.pt") if p.is_file()]


def filter_snapshots_by_episode(paths: List[Path], divisible_by: int = 100_000) -> List[Tuple[Path, int]]:
    """Filter to snapshots whose extracted episode number is divisible by N.

    Returns a list of tuples: (path, episodes)
    """
    filtered: List[Tuple[Path, int]] = []
    for p in paths:
        episodes = extract_episodes_from_name(p)
        if episodes is None:
            continue
        if episodes % divisible_by == 0:
            filtered.append((p, episodes))

    # Sort by episodes ascending
    filtered.sort(key=lambda t: t[1])
    return filtered


def load_eval_agents(paths_with_eps: List[Tuple[Path, int]], activation: str = "swish") -> List[EvalAgent]:
    agents: List[EvalAgent] = []
    for path, episodes in paths_with_eps:
        try:
            agent = PPOAgent(len(ACTIONS), activation=activation)
            agent.load(str(path), load_optimizers=False)
            # Create default PL rating (mu defaults to ~25, sigma ~25/3)
            pl = PlackettLuce()
            rating = pl.rating()
            eval_agent = EvalAgent(
                agent_id=path.stem,
                filepath=path,
                episodes=episodes,
                agent=agent,
                rating_mu=float(rating.mu),
                rating_sigma=float(rating.sigma),
            )
            agents.append(eval_agent)
        except (OSError, RuntimeError) as e:
            print(f"Warning: failed to load snapshot '{path}': {e}")

    return agents


def update_ratings_pl(
    game_agents: List[EvalAgent],
    scores: List[float],
    picker_seat: int,
    partner_seat: int,
    is_leaster: bool,
) -> None:
    """Update Plackett-Luce ratings using team semantics.

    For leasters we rate each player individually (teams of one),
    otherwise we rate the picking team (picker + partner) vs defenders,
    providing aggregated scores to `model.rate`.
    """
    if len(game_agents) != 5 or len(scores) != 5:
        raise ValueError("Expected exactly 5 agents and 5 scores per game")

    model = PlackettLuce()

    if is_leaster:
        teams = [[model.rating(mu=a.rating_mu, sigma=a.rating_sigma)] for a in game_agents]
        new_teams = model.rate(teams, scores=scores)
        for i, a in enumerate(game_agents):
            new_rating = new_teams[i][0]
            a.update_rating(mu=float(new_rating.mu), sigma=float(new_rating.sigma))
        return

    picker_team_indices = []
    if picker_seat:
        picker_team_indices.append(picker_seat - 1)
    if partner_seat and partner_seat != picker_seat:
        picker_team_indices.append(partner_seat - 1)

    defender_indices = [i for i in range(5) if i not in picker_team_indices]

    team_picker = [model.rating(mu=game_agents[i].rating_mu, sigma=game_agents[i].rating_sigma) for i in picker_team_indices]
    team_def = [model.rating(mu=game_agents[i].rating_mu, sigma=game_agents[i].rating_sigma) for i in defender_indices]

    picker_score = sum(scores[i] for i in picker_team_indices)
    defender_score = sum(scores[i] for i in defender_indices)
    team_scores = [picker_score, defender_score]

    new_team_ratings = model.rate([team_picker, team_def], scores=team_scores)
    new_picker_ratings, new_def_ratings = new_team_ratings

    for idx, new_rating in zip(picker_team_indices, new_picker_ratings):
        game_agents[idx].update_rating(mu=float(new_rating.mu), sigma=float(new_rating.sigma))
    for idx, new_rating in zip(defender_indices, new_def_ratings):
        game_agents[idx].update_rating(mu=float(new_rating.mu), sigma=float(new_rating.sigma))


def play_evaluation_game(partner_mode: int, agents: List[EvalAgent], seed: int) -> Tuple[List[float], int, int, bool]:
    """Play one evaluation game among the given 5 agents; return final scores and team info.

    Args:
        partner_mode: Partner selection mode (PARTNER_BY_JD or PARTNER_BY_CALLED_ACE)
        agents: List of exactly 5 EvalAgent instances
        seed: Seed for deterministic deck shuffling. The same seed will produce identical
              card deals across different seat rotations.
    """
    if len(agents) != 5:
        raise ValueError("Exactly 5 agents required to play a game")

    # Initialize game and mapping
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    pos_to_agent: Dict[int, EvalAgent] = {pos: agent for pos, agent in enumerate(agents, start=1)}

    # Reset recurrent states
    for ea in agents:
        ea.agent.reset_recurrent_state()

    # Capture hand strength category at start (if needed by agent; we don't use here)
    # Main loop
    while not game.is_done():
        for player in game.players:
            current_eval_agent = pos_to_agent[player.position]
            valid_actions = player.get_valid_action_ids()

            while valid_actions:
                state = player.get_state_dict()
                action, _, _ = current_eval_agent.agent.act(
                    state, valid_actions, player.position, deterministic=True
                )
                player.act(action)
                valid_actions = player.get_valid_action_ids()

                # Propagate observation at end of trick for all agents
                if game.was_trick_just_completed:
                    for seat in game.players:
                        seat_agent = pos_to_agent[seat.position]
                        seat_agent.agent.observe(
                            seat.get_last_trick_state_dict(),
                            player_id=seat.position,
                        )

                # Optional strategic profiling omitted here; tournament only updates ratings

    final_scores = [p.get_score() for p in game.players]
    picker_seat = game.picker
    partner_seat = game.partner
    is_leaster = bool(getattr(game, "is_leaster", False))
    return final_scores, picker_seat, partner_seat, is_leaster


def run_tournament(
    agents: List[EvalAgent],
    partner_mode: int,
    rounds: int,
    seed: Optional[int] = None,
    deals_per_round: int = 10,
) -> None:
    """Run a multi-round tournament; update agents in place with ratings and stats.

    Uses deterministic deals: for each 5-agent group, generates a fixed number of
    unique deals (seeds). Each deal is reused across all 5 seat rotations to ensure
    fair comparison of agent skill across positions with identical card distributions.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if len(agents) < 5:
        print("Not enough agents to run a 5-player game. Exiting.")
        return

    # Create a deterministic RNG for generating deal seeds
    deal_rng = random.Random(seed) if seed is not None else random.Random()

    # For each round, shuffle and partition into 5-player groups
    for rnd in range(rounds):
        shuffled = agents[:]
        random.shuffle(shuffled)

        games_this_round = 0
        for start_idx in range(0, len(shuffled) - 4, 5):
            group = shuffled[start_idx:start_idx + 5]

            # Generate deterministic seeds for this group's deals
            # Each deal seed will be reused for all 5 seat rotations
            deal_seeds = [deal_rng.randint(0, 2**31 - 1) for _ in range(deals_per_round)]

            # For each deal, play all 5 seat rotations
            for deal_idx, deal_seed in enumerate(deal_seeds):
                for r in range(5):
                    order = [(i + r) % 5 for i in range(5)]
                    rotated_group = [group[idx] for idx in order]

                    # Use the same seed for all rotations of this deal
                    scores, picker_seat, partner_seat, is_leaster = play_evaluation_game(
                        partner_mode, rotated_group, seed=deal_seed
                    )

                    # Update ratings and stats for this game
                    update_ratings_pl(
                        rotated_group,
                        scores,
                        picker_seat=picker_seat,
                        partner_seat=partner_seat,
                        is_leaster=is_leaster,
                    )
                    for a, s in zip(rotated_group, scores):
                        a.add_game_result(s)

                    games_this_round += 1

        print(f"Round {rnd + 1}/{rounds}: completed {games_this_round} games ({deals_per_round} deals × 5 rotations per group)")


def write_csv(agents: List[EvalAgent], out_csv: Path) -> None:
    fieldnames = [
        "agent_id",
        "filepath",
        "episodes",
        "rating_mu",
        "rating_sigma",
        "games_played",
        "avg_score",
    ]
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for a in sorted(agents, key=lambda x: x.episodes):
            writer.writerow({
                "agent_id": a.agent_id,
                "filepath": str(a.filepath),
                "episodes": a.episodes,
                "rating_mu": f"{a.rating_mu:.4f}",
                "rating_sigma": f"{a.rating_sigma:.4f}",
                "games_played": a.games_played,
                "avg_score": f"{a.avg_score:.4f}",
            })
    print(f"Wrote CSV: {out_csv}")


def plot_ratings(agents: List[EvalAgent], out_plot: Path, title: str = "OpenSkill Rating vs Episodes") -> None:
    episodes = [a.episodes for a in agents]
    mus = [a.rating_mu for a in agents]
    sigmas = [a.rating_sigma for a in agents]

    # Sort by episodes
    order = np.argsort(episodes)
    episodes_sorted = np.array(episodes)[order]
    mus_sorted = np.array(mus)[order]
    sigmas_sorted = np.array(sigmas)[order]

    plt.figure(figsize=(10, 6))
    plt.plot(episodes_sorted, mus_sorted, marker="o", label="mu")
    # Optionally shade mu ± sigma
    plt.fill_between(
        episodes_sorted,
        mus_sorted - sigmas_sorted,
        mus_sorted + sigmas_sorted,
        color="blue",
        alpha=0.15,
        label="mu ± sigma",
    )
    plt.xlabel("Training Episodes")
    plt.ylabel("OpenSkill mu")
    plt.title(title)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()
    print(f"Wrote plot: {out_plot}")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sheepshead snapshot tournament evaluator")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing snapshot .pt files")
    parser.add_argument("--partner-mode", type=str, default="called", choices=["jd", "called", "0", "1"],
                        help="Partner selection mode for games: 'jd'(0) or 'called'(1)")
    parser.add_argument("--rounds", type=int, default=100, help="Number of tournament rounds (each partitions agents into 5-player games)")
    parser.add_argument("--deals-per-round", type=int, default=10, help="Number of unique deals per 5-agent group (same deal reused across all 5 seat rotations)")
    parser.add_argument("--activation", type=str, default="swish", help="Activation used by agents (for model init)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--out-csv", type=str, default="tournament_results.csv", help="Output CSV path")
    parser.add_argument("--out-plot", type=str, default="tournament_ratings.png", help="Output plot (PNG)")
    return parser.parse_args(argv)


def resolve_partner_mode(mode_str: str) -> int:
    if mode_str in ("0", "jd"):
        return PARTNER_BY_JD
    if mode_str in ("1", "called"):
        return PARTNER_BY_CALLED_ACE
    raise ValueError(f"Unknown partner mode: {mode_str}")


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    input_dir = Path(args.input_dir).resolve()
    partner_mode = resolve_partner_mode(args.partner_mode)
    activation = args.activation
    out_csv = Path(args.out_csv).resolve()
    out_plot = Path(args.out_plot).resolve()

    print(f"Discovering snapshots in {input_dir} ...")
    all_paths = discover_snapshots(input_dir)
    print(f"Found {len(all_paths)} *.pt files")

    print("Filtering to episode markers divisible by 100,000 ...")
    filtered = filter_snapshots_by_episode(all_paths, divisible_by=100_000)
    print(f"Eligible snapshots: {len(filtered)}")
    if len(filtered) < 5:
        print("Need at least 5 eligible snapshots to run a tournament. Exiting.")
        return 1

    print("Loading agents ...")
    agents = load_eval_agents(filtered, activation=activation)
    print(f"Loaded {len(agents)} agents")
    if len(agents) < 5:
        print("Need at least 5 loaded agents to run a tournament. Exiting.")
        return 1

    print(f"Running tournament: rounds={args.rounds}, partner_mode={args.partner_mode}, deals_per_round={args.deals_per_round}")
    print("Using deterministic deals: each deal reused across all 5 seat rotations for fair comparison")
    run_tournament(agents, partner_mode=partner_mode, rounds=args.rounds, seed=args.seed, deals_per_round=args.deals_per_round)

    print("Writing outputs ...")
    write_csv(agents, out_csv)
    plot_ratings(agents, out_plot)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
