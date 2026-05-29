#!/usr/bin/env python3
"""PFSP ISMCTS ExIt trainer (hybrid) — thin entry point.

Population-based training with terminal-only reward and ISMCTS soft-teacher
distillation (the PG-mask owns the policy update on confidently-searched
transitions; PPO owns the rest). No reward shaping or epsilon-floor controllers
— the search teacher replaces hand-tuned exploration. All machinery lives in
pfsp_runtime.py; hyperparameters in config.py.

Bidding-head search is gated off by default (only the PLAY head is searched);
see ISMCTS_Overview_And_Roadmap.md §4 (P4) for the pre-pick determinizer that
enables PICK-head search.
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from config import PFSPHyperparams, SearchConfig
from pfsp_runtime import run_pfsp_training


def main():
    parser = ArgumentParser(
        description="PFSP ISMCTS ExIt (hybrid) population training for Sheepshead"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=20_000_000,
        help="Number of training episodes (default: 20,000,000)",
    )
    parser.add_argument(
        "--update-interval",
        type=int,
        default=2048,
        help="Number of transitions between model updates",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5000,
        help="Number of episodes between checkpoints",
    )
    parser.add_argument(
        "--strategic-eval-interval",
        type=int,
        default=10000,
        help="Number of episodes between strategic evaluations",
    )
    parser.add_argument(
        "--population-add-interval",
        type=int,
        default=5000,
        help="Number of episodes between adding agents to population",
    )
    parser.add_argument(
        "--cross-eval-interval",
        type=int,
        default=20000,
        help="Number of episodes between cross-evaluation tournaments",
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Model file to resume training from"
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="swish",
        choices=["relu", "swish"],
        help="Activation function to use (default: swish)",
    )
    parser.add_argument(
        "--initial-checkpoints",
        nargs="+",
        default=None,
        help="Checkpoint patterns to initialize population from",
    )
    parser.add_argument(
        "--schedule-horizon-episodes",
        type=int,
        default=None,
        help="Episode horizon used for entropy schedules (defaults to --episodes)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="pfsp_exit",
        help="Run name; all artifacts go under runs/<run-name>/ (default: pfsp_exit)",
    )
    parser.add_argument(
        "--population-dir",
        type=str,
        default=None,
        help="Population directory (default: runs/<run-name>/population). Point at a seeded pool to reuse it.",
    )
    # Search knobs (see config.SearchConfig). Bidding heads gated off pending the
    # pre-pick determinizer (P4); only the PLAY head is searched by default.
    parser.add_argument(
        "--f-play",
        type=float,
        default=0.10,
        help="Per-decision probability of searching a PLAY decision (default: 0.10)",
    )
    parser.add_argument(
        "--t-full",
        type=int,
        default=1,
        help="Trick-indexed rollout-depth cutoff: full rollout for tricks 0..t_full (default: 1)",
    )
    parser.add_argument(
        "--d-short",
        type=int,
        default=2,
        help="Bootstrap rollout depth for tricks > t_full (default: 2)",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend("Agg")

    # ExIt hybrid: terminal-only reward + ISMCTS distillation, no shaping/controllers.
    search = SearchConfig(
        fracs={"pick": 0.0, "partner": 0.0, "bury": 0.0, "play": args.f_play},
        t_full=args.t_full,
        d_short=args.d_short,
    )
    hyperparams = PFSPHyperparams(reward_mode="terminal", search=search)

    run_pfsp_training(
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        save_interval=args.save_interval,
        strategic_eval_interval=args.strategic_eval_interval,
        population_add_interval=args.population_add_interval,
        cross_eval_interval=args.cross_eval_interval,
        resume_model=args.resume,
        activation=args.activation,
        initial_checkpoints=args.initial_checkpoints,
        schedule_horizon_episodes=args.schedule_horizon_episodes,
        hyperparams=hyperparams,
        run_name=args.run_name,
        population_dir=args.population_dir,
    )


if __name__ == "__main__":
    main()
