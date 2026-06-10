#!/usr/bin/env python3
"""PFSP ISMCTS Expert Iteration (ExIt) trainer hybrid — thin entry point.

Population-based training with terminal-only reward and (optional) ISMCTS
soft-teacher distillation. No reward shaping or epsilon-floor controllers. All
machinery lives in pfsp_runtime.py; hyperparameters in config.py.

Search-knob CLI defaults are read from ``config.SearchConfig`` (single source of
truth — the argparse defaults previously hardcoded stale values that silently
overrode config.py). All heads are searchable (P4); per-head fractions, the
searched-PPO weight, the distill ramp, the bidding-head KL anchor and the greedy
health probe are exposed below. See ISMCTS_Overview_And_Roadmap.md §4-5 and the
run-1/run-2 collapse post-mortems for why the anchor + ramp + greedy guard
default ON for this trainer.
"""

import random
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import PFSPHyperparams, SearchConfig
from pfsp_runtime import run_pfsp_training

# Single source of truth for search-knob defaults (config.SearchConfig).
_SEARCH_DEFAULTS = SearchConfig()


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
    # Search knobs — defaults come from config.SearchConfig (single source of
    # truth). All heads searchable as of P4 (the pre-pick determinizer).
    parser.add_argument(
        "--f-pick",
        type=float,
        default=_SEARCH_DEFAULTS.head_search_fractions["pick"],
        help="Per-decision probability of searching a PICK/PASS decision "
        f"(default: {_SEARCH_DEFAULTS.head_search_fractions['pick']})",
    )
    parser.add_argument(
        "--f-partner",
        type=float,
        default=_SEARCH_DEFAULTS.head_search_fractions["partner"],
        help="Per-decision probability of searching a PARTNER decision "
        f"(default: {_SEARCH_DEFAULTS.head_search_fractions['partner']})",
    )
    parser.add_argument(
        "--f-bury",
        type=float,
        default=_SEARCH_DEFAULTS.head_search_fractions["bury"],
        help="Per-decision probability of searching a BURY decision "
        f"(default: {_SEARCH_DEFAULTS.head_search_fractions['bury']})",
    )
    parser.add_argument(
        "--f-play",
        type=float,
        default=_SEARCH_DEFAULTS.head_search_fractions["play"],
        help="Per-decision probability of searching a PLAY decision "
        f"(default: {_SEARCH_DEFAULTS.head_search_fractions['play']})",
    )
    parser.add_argument(
        "--t-full",
        type=int,
        default=_SEARCH_DEFAULTS.t_full,
        help="Trick-indexed rollout-depth cutoff: full rollout for tricks 0..t_full "
        f"(default: {_SEARCH_DEFAULTS.t_full})",
    )
    parser.add_argument(
        "--d-short",
        type=int,
        default=_SEARCH_DEFAULTS.d_short,
        help="Bootstrap rollout depth for tricks > t_full "
        f"(default: {_SEARCH_DEFAULTS.d_short})",
    )
    parser.add_argument(
        "--searched-ppo-weight",
        type=float,
        default=_SEARCH_DEFAULTS.searched_ppo_weight,
        help="PG-mask vs additive-form A/B: weight on the PPO clip term for searched "
        "transitions (0.0=hard mask, 1.0=additive, 0<w<1=residual PPO). "
        "Distillation toward pi' is applied either way at search_distill_coeff. "
        f"(default: {_SEARCH_DEFAULTS.searched_ppo_weight})",
    )
    parser.add_argument(
        "--distill-ramp",
        type=int,
        default=_SEARCH_DEFAULTS.distill_ramp_episodes,
        help="Episodes over which search_distill_coeff ramps 0->configured value "
        "after run start (onset-shock guard; 0 disables) "
        f"(default: {_SEARCH_DEFAULTS.distill_ramp_episodes})",
    )
    # Bidding-head KL anchor (warm-start collapse guard). Both collapse runs
    # (distill-owned and PG-owned bidding) flattened the bidding heads to
    # always-PASS/ALONE; the anchor pins pick/partner/bury to the frozen
    # reference while the play head trains freely.
    parser.add_argument(
        "--anchor-coeff",
        type=float,
        default=1.0,
        help="Weight on KL(pi_ref || pi_theta) over bidding-head transitions "
        "(0 disables; default: 1.0)",
    )
    parser.add_argument(
        "--anchor-ref",
        type=str,
        default="final_pfsp_swish_ppo.pt",
        help="Frozen reference model for the bidding-head KL anchor "
        "(default: final_pfsp_swish_ppo.pt)",
    )
    # Greedy self-play health probe (collapse guard): argmax rates expose a
    # flattening collapse that stochastic training-time rates mask.
    parser.add_argument(
        "--greedy-eval-interval",
        type=int,
        default=5000,
        help="Episodes between greedy health probes (0 disables; default: 5000)",
    )
    parser.add_argument(
        "--greedy-eval-games",
        type=int,
        default=200,
        help="Greedy self-play games per health probe (default: 200)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Parallel game-generation workers (Lever 1). Default: auto "
        "(min(cpu_count-1, 8) for ExIt). 1 forces the sequential loop.",
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
        head_search_fractions={
            "pick": args.f_pick,
            "partner": args.f_partner,
            "bury": args.f_bury,
            "play": args.f_play,
        },
        t_full=args.t_full,
        d_short=args.d_short,
        searched_ppo_weight=args.searched_ppo_weight,
        distill_ramp_episodes=args.distill_ramp,
    )
    hyperparams = PFSPHyperparams(
        reward_mode="terminal",
        search=search,
        num_workers=args.num_workers,
        anchor_loss_coeff=args.anchor_coeff,
        anchor_ref_model=args.anchor_ref,
        greedy_eval_interval=args.greedy_eval_interval,
        greedy_eval_games=args.greedy_eval_games,
    )

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
