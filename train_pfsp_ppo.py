#!/usr/bin/env python3
"""PFSP PPO trainer (shaped baseline) — thin entry point.

Population-based PPO with reward shaping + adaptive entropy-bump exploration.
All machinery lives in pfsp_runtime.py; hyperparameters in config.py. This is the
behaviour-frozen baseline for comparison against the ISMCTS ExIt hybrid
(train_pfsp_exit.py).
"""

import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from config import PFSPHyperparams
from pfsp_runtime import run_pfsp_training


def main():
    parser = ArgumentParser(
        description="PFSP shaped-PPO population training for Sheepshead"
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
        help="Episode horizon used for entropy/reward-shaping schedules (defaults to --episodes)",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="pfsp_ppo",
        help="Run name; all artifacts go under runs/<run-name>/ (default: pfsp_ppo)",
    )
    parser.add_argument(
        "--population-dir",
        type=str,
        default=None,
        help="Population directory (default: runs/<run-name>/population). Point at a seeded pool to reuse it.",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend("Agg")

    # Shaped baseline: reward shaping + entropy bumps, no search.
    hyperparams = PFSPHyperparams(reward_mode="shaped", search=None)

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
