"""CLI argument parser for train_league_ppo.py.

Split out of train_league_ppo.py as pure code motion (Stage 1 of the
league-trainer maintainability refactor): build_arg_parser's body, grouped
into topical helper functions, each taking the parser and adding its
add_argument calls. The grouping is cosmetic only — build_arg_parser() calls
the groups in the exact order needed to preserve the historical --help
argument ordering.
"""

from __future__ import annotations

import argparse

from sheepshead.agent import architectures


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume", required=True, help="main agent checkpoint to start from"
    )
    parser.add_argument("--league-dir", required=True)
    parser.add_argument(
        "--migrate-from",
        default=None,
        help="legacy population dir (used once if league empty)",
    )
    parser.add_argument(
        "--seed-checkpoints",
        default=None,
        help="glob or dir of PPO checkpoints to seed an empty league as "
        "past_mains (e.g. the selfplay bootstrap snapshots that seeded the "
        "original pfsp run: 'runs/reference_selfplay_ppo/checkpoints/*.pt')",
    )
    parser.add_argument("--run-name", default="league_run")
    parser.add_argument(
        "--generations",
        type=int,
        default=3,
        help="Number of exploiter generations to run from the resume point. "
        "Boundaries are keyed to absolute episode (gen g ends at g*main-episodes), "
        "so the starting generation index is derived from the resumed episode.",
    )
    parser.add_argument("--main-episodes", type=int, default=1_000_000)
    parser.add_argument("--exploiter-episodes", type=int, default=50_000)
    parser.add_argument("--gate-deals", type=int, default=3000)
    parser.add_argument(
        "--screen-deals",
        type=int,
        default=200,
        help="paired deals per exploiter checkpoint for best-of-checkpoints "
        "selection before the full gate (0 = gate the final save only)",
    )
    parser.add_argument("--update-interval", type=int, default=16_384)
    parser.add_argument("--save-interval", type=int, default=50_000)
    parser.add_argument("--snapshot-interval", type=int, default=50_000)
    parser.add_argument("--greedy-eval-interval", type=int, default=50_000)
    parser.add_argument("--greedy-eval-games", type=int, default=200)
    parser.add_argument("--schedule-horizon", type=int, default=20_000_000)


def add_entropy_args(parser: argparse.ArgumentParser) -> None:
    # Entropy controller v2 (CE_Teacher_Design §4): ALWAYS ON for the league
    # trainer — the signed SAC-style target-entropy controller
    # (entropy_controller.py) owns the entropy coefficients; the LR schedule
    # keeps its clock and the legacy --entropy-mode selector is gone.
    # Targets initialize bumplessly from the first update's measurement
    # unless given explicitly; state persists in
    # <checkpoint-dir>/entropy_controller.json. Standalone runs hold targets
    # at their initial operating point; the orchestrator's outer loop
    # (run_extended_league --adaptive-entropy) steps them between
    # generations. Not a CLI flag: the exploiter's SimpleNamespace args
    # omits it, keeping best-response training on the plain schedule.
    parser.set_defaults(entropy_controller=True)
    for head in ("pick", "partner", "bury", "play"):
        parser.add_argument(
            f"--entropy-target-{head}",
            type=float,
            default=None,
            help=f"explicit initial H_norm target for the {head} head "
            "(default: bumpless from first measurement)",
        )
    parser.add_argument(
        "--entropy-play-floor",
        type=float,
        default=0.28,
        help="play-head target floor (mixed-equilibrium reserve, ~37%% of "
        "the retention run's 1.8M operating point)",
    )


def add_worker_inference_args(parser: argparse.ArgumentParser) -> None:
    """Throughput-only options for the worker processes. Both change results in
    the last bits, so both are opt-in and neither may be set for a run whose
    output is compared bit-exactly against another."""
    parser.add_argument(
        "--worker-device",
        default=None,
        metavar="DEVICE",
        help="device for worker inference (e.g. 'mps'). Default: the process "
        "default, which is CPU everywhere but CUDA. Workers only generate "
        "episodes; the update, the gates and greedy eval stay where they were",
    )
    parser.add_argument(
        "--worker-compile",
        nargs="?",
        const="default",
        default=None,
        metavar="MODE",
        help="torch.compile the encoder in workers (all four search call "
        "sites). Opt-in: output differs from eager by ~2.6e-08, so "
        "capture_search_goldens cannot pass against a run using it. Pays for "
        "itself only over thousands of committees — the first few per worker "
        "are slower. With --worker-device mps this is the measured 1.36x",
    )
    parser.add_argument(
        "--worker-compile-granularity",
        type=int,
        default=32,
        help="round encode batches up to a multiple of this so compilation "
        "sees ~14 shapes instead of ~93, at ~1.8%% wasted rows",
    )


def add_training_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--critic-mode",
        choices=["limited", "oracle"],
        default="oracle",
        help="'oracle' trains a privileged full-information critic as the GAE "
        "baseline (asymmetric actor-critic; see oracle.py). The actor, the "
        "limited critic, and all aux heads train identically in both modes.",
    )
    parser.add_argument("--anchor-coeff", type=float, default=0.0)
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=None,
        help="Override GAE lambda (default: agent's 0.95). Phase B of "
        "Learning_System_Redesign_202607 lowers this toward 0.8 once the "
        "stratified-EV gate shows trustworthy mid-game values.",
    )
    parser.add_argument(
        "--exploiter-patched-ema",
        type=float,
        default=0.35,
        help="retire an exploiter to past_main once its live outcome EMA vs "
        "the training agent falls below this (with enough samples) — the "
        "exploit is patched, stop paying its seat share (default: age-based "
        "retirement only)",
    )
    parser.add_argument(
        "--grad-accum",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="accumulate minibatch gradients (row-fraction scaled) and step "
        "once per epoch: the single full-buffer optimizer step of the "
        "validated low-temperature design, with activation memory bounded "
        "by --minibatch-episodes. (Historically also the fix for the ~40GB "
        "braided-segment OOM; post per-seat storage fix it is kept for the "
        "step-size semantics, not memory.)",
    )
    parser.add_argument(
        "--minibatch-episodes",
        type=int,
        default=128,
        help="episodes per forward/backward chunk (PPOAgent.update "
        "batch_size). Under --grad-accum this bounds peak activation "
        "memory only — the optimizer still steps once per epoch over the "
        "whole buffer; with --no-grad-accum it becomes the per-step "
        "minibatch size",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="discount factor. Default 1.0: undiscounted terminal returns "
        "— removes the ~7%% depth tilt against early nodes on this "
        "finite-horizon terminal-reward game (retention-run validated). "
        "The agent's historical value was 0.99",
    )
    parser.add_argument(
        "--oracle-aux-heads",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="build the oracle critic with deterministic aux heads "
        "(per-seat team membership + team points; offline-validated "
        "2026-07-24). Historical checkpoints without heads still load "
        "(heads start fresh)",
    )
    parser.add_argument(
        "--oracle-init",
        default=None,
        help="path to a pretrained oracle state_dict (e.g. from "
        "oracle_moe_offline pretrain) loaded into the oracle critic after "
        "--resume — the supervised warm start that removes the fresh-"
        "oracle burn-in window",
    )


def add_teacher_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--teacher",
        action="store_true",
        help="always-on CE search teacher on main-agent play decisions "
        "(CE_Teacher_Design): a --teacher-replicates lockstep ISMCTS "
        "committee at the calibrated budget (--teacher-iters, d=1, oracle "
        "leaves) labels a --teacher-prob subsample of eligible PLAY nodes "
        "with shrink-and-tilt CE targets (pi_gumbel on James-Stein-shrunk "
        "committee Q), distilled for --teacher-epochs supervised passes "
        "after each PPO update. No phases, no PG-mask; abstention lives "
        "in the target. Works with --num-workers > 1: weight payloads "
        "carry the oracle head + gamma so worker-side searches stay "
        "calibrated",
    )
    parser.add_argument(
        "--teacher-prob",
        type=float,
        default=0.1,
        help="subsample probability for eligible play nodes — the labeling "
        "budget knob (unbiased; class-blind per CE_Teacher_Design §2). "
        "Expected wall cost per episode ~ prob x eligible-nodes/game x "
        "one lockstep committee search",
    )
    parser.add_argument(
        "--teacher-replicates",
        type=int,
        default=3,
        help="committee size R: independent replicate searches run in "
        "lockstep (search_committee); replicate spread feeds the "
        "shrinkage noise model that flattens within-noise targets",
    )
    parser.add_argument(
        "--teacher-iters",
        type=int,
        default=1024,
        help="ISMCTS iterations per replicate (the E9-certified cheap "
        "budget; §12.8 refuted the heavy arm — replicates beat "
        "iterations)",
    )
    parser.add_argument(
        "--teacher-coeff",
        type=float,
        default=1.0,
        help="coefficient on the CE distillation loss (mean over labeled "
        "rows); safe near 1.0 because a conformed or within-noise target "
        "carries ~zero gradient by construction",
    )
    parser.add_argument(
        "--teacher-epochs",
        type=int,
        default=4,
        help="supervised CE passes over the update window's labeled rows "
        "after the PPO epochs (asymmetric epochs: the PG loss keeps its "
        "own tuning; a fixed supervised target needs no importance "
        "ratios). Labels are discarded with their update window",
    )


def add_guard_cert_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--adherence-guard-interval",
        type=int,
        default=50000,
        help="run the n=1000 fixed-seed convention adherence guard every N "
        "episodes (0 = off). §12.17: smaller probes masked an 8-point "
        "partner-trump regression for a full teacher run",
    )
    parser.add_argument(
        "--adherence-guard-games",
        type=int,
        default=1000,
        help="games per adherence guard probe",
    )
    parser.add_argument(
        "--guard-partner-floor",
        type=float,
        default=90.0,
        help="adherence guard HARD stop: halt the run (exit 3, checkpoint "
        "saved) if partner trump-lead %% falls below this floor "
        "(CE_Teacher_Design §3 two-tier protocol; recalibrate for other "
        "lineages — the 8M perceiver-shared-v2 seed reads 96.5 at n=1000)",
    )
    parser.add_argument(
        "--guard-partner-notify",
        type=float,
        default=93.5,
        help="adherence guard NOTIFY tier: print a warning (no stop) when "
        "partner trump-lead %% dips below this line — §12.20 showed "
        "teaching-time dips can be oscillation with a restoring force",
    )
    parser.add_argument(
        "--guard-t0-ceiling",
        type=float,
        default=5.0,
        help="adherence guard HARD stop: halt the run if defender t0 "
        "trump-lead %% rises above this ceiling (the v7 scramble "
        "signature; seed level is ~0.1%%)",
    )
    parser.add_argument(
        "--cert-seeds",
        type=int,
        default=3,
        help="boundary cert: number of distinct fixed deal seeds in the "
        "adherence battery (multi-seed: §12.22 — single reads are "
        "luck-of-phase)",
    )
    parser.add_argument(
        "--cert-games",
        type=int,
        default=1000,
        help="boundary cert: games per adherence-battery seed (n=1000: "
        "smaller probes cannot resolve <5-point convention deltas, §12.17)",
    )
    parser.add_argument(
        "--cert-partner-floor",
        type=float,
        default=93.5,
        help="boundary cert ABSOLUTE bar: across-seed mean partner "
        "trump-lead %% required for the candidate to become the next "
        "generation's frozen expert",
    )
    parser.add_argument(
        "--cert-t0-ceiling",
        type=float,
        default=5.0,
        help="boundary cert ABSOLUTE bar: across-seed mean t0 trump-lead %% ceiling",
    )
    parser.add_argument(
        "--cert-h2h-deals",
        type=int,
        default=1000,
        help="boundary cert: paired CRN deals vs the fixed cert anchor; "
        "the candidate fails if its edge is significantly negative",
    )
    parser.add_argument(
        "--cert-anchor-ckpt",
        default=None,
        help="FIXED h2h anchor for every boundary cert (default: --resume "
        "at launch). Absolute anchoring prevents boundary-to-boundary "
        "drift from ratcheting (CE_Teacher_Design §3) — with the live "
        "expert it is the only certification the teacher has (§15a)",
    )


def add_eval_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--seat-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="deal-paired collection: each sampled deal is played 5 times "
        "with the hero rotating through all seats against the same table "
        "(train-time duplicate instrument; equalizes role exposure per "
        "deal)",
    )
    parser.add_argument(
        "--gns-log",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="log the gradient noise scale each update (global + partner-"
        "lead stratum, units = action rows) to the progress CSV. One extra "
        "epoch-equivalent of compute per update; measurement only — the "
        "applied updates are bit-identical",
    )
    parser.add_argument(
        "--oracle-extra-epochs",
        type=int,
        default=4,
        help="extra oracle-regression-only epochs after each update "
        "(per-minibatch oracle optimizer steps). The oracle has its own "
        "encoder, so this touches no policy/limited-critic parameter — a "
        "step-count lever for the fresh-oracle transient at large "
        "--update-interval. 0 = historical behavior",
    )
    parser.add_argument("--anchor-ref", default=None)
    parser.add_argument(
        "--anchor-eval-ckpt",
        default="final_pfsp_swish_ppo.pt",
        help="frozen reference for the periodic anchored strength probe, the "
        "run's absolute-strength trend line ('' disables)",
    )
    parser.add_argument("--anchor-eval-interval", type=int, default=100_000)
    parser.add_argument("--anchor-eval-deals", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--arch",
        default="perceiver-shared-v2",
        choices=architectures.available_architectures(),
        help="Network architecture variant for the training agent, its "
        "snapshots, and the exploiter phase (see the architectures package)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--leaster-watchdog",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pick-entropy kick against the always-PASS/leaster collapse "
        "(seen re-entered from a trained policy in anchor-free stage-1 "
        "generations; see leaster_watchdog.py). Main phases only — the "
        "exploiter phase never runs it. Enable uniformly across the arms "
        "of any comparison.",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="League training (main/exploiter generations)"
    )
    add_run_args(parser)
    add_entropy_args(parser)
    add_worker_inference_args(parser)
    add_training_args(parser)
    add_teacher_args(parser)
    add_guard_cert_args(parser)
    add_eval_args(parser)
    return parser
