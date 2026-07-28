#!/usr/bin/env python3
"""Checkpoint entropy-trajectory backfill (adaptive-entropy Phase 1).

Sweeps ``entropy_probe.probe_agent`` over a run's checkpoint history to
reconstruct the per-head normalized-entropy (H/ln n_legal) trajectory that
the live telemetry (``head_entropy_norm`` in league_training_progress.csv)
would have recorded, and derives the two quantities the planned
target-entropy controller needs from a run that worked:

* ``suggested_targets`` — the final checkpoint's per-head mean H_norm, i.e.
  the operating point a fresh run's controller would be initialized at for
  a bumpless start (Astrom & Wittenmark, *Adaptive Control*, 2nd ed. 1995,
  ch. 9: switch a controller in at the measured operating point so the
  handoff itself changes nothing).
* ``organic_decline_per_gen`` — the OLS slope of mean H_norm vs episodes,
  scaled to one league generation (1M episodes): the entropy drift training
  produces on its own under the run's FIXED coefficients. A plateau-
  triggered target step must exceed this scale to be distinguishable from
  the null; the step magnitude itself is anchored to PBT's hyperparameter
  perturbation factors of 0.8/1.2x (Jaderberg et al., arXiv:1711.09846).

Controller background: SAC automatic temperature adjustment (Haarnoja et
al., arXiv:1812.05905), discrete form with fraction-of-max targets
(Christodoulou, arXiv:1910.07207). Rationale for measuring rather than
scheduling: the coefficient-to-entropy mapping drifts over training, and
entropy-cost schedules are environment-specific tuning levers (Andrychowicz
et al., "What Matters in On-Policy RL", arXiv:2006.05990; Jaderberg et al.
found non-obvious entropy-cost schedules when they were learned online).

For the writeup, each row also records the run's scheduled per-head entropy
coefficients at that episode (linear decay over --schedule-horizon; see
training/config.py PFSP_HYPERPARAMS) so coefficient and measured entropy
can be compared directly.

Usage (from repo root):

    uv run python -m sheepshead.analysis.entropy_backfill \
        --out runs/league_retention_pg/entropy_backfill.json
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import time
from pathlib import Path

import numpy as np

from sheepshead.analysis.entropy_probe import HEADS, PROBE_SEED, probe_agent
from sheepshead.training.config import PFSP_HYPERPARAMS

DEFAULT_GLOB = (
    "runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_*.pt"
)
EPISODES_PER_GEN = 1_000_000


def _episodes_from_name(path: str) -> int | None:
    m = re.search(r"checkpoint_(\d+)", Path(path).name)
    return int(m.group(1)) if m else None


def scheduled_coeffs(episode: int, horizon: int) -> dict:
    """The run's per-head entropy coefficients at ``episode`` (the linear
    schedule from train_league_ppo.apply_schedules)."""
    decay = 1.0 - min(1.0, episode / max(horizon, 1))
    hp = PFSP_HYPERPARAMS
    return {
        "pick": hp.entropy_pick_end
        + (hp.entropy_pick_start - hp.entropy_pick_end) * decay,
        "partner": hp.entropy_partner_end
        + (hp.entropy_partner_start - hp.entropy_partner_end) * decay,
        "bury": hp.entropy_bury_end
        + (hp.entropy_bury_start - hp.entropy_bury_end) * decay,
        "play": hp.entropy_play_end
        + (hp.entropy_play_start - hp.entropy_play_end) * decay,
    }


def derive(rows: list[dict]) -> dict:
    """Bumpless targets (last checkpoint) + organic per-generation decline
    (OLS slope of mean H_norm vs episodes) per head."""
    derived: dict = {"suggested_targets": {}, "organic_decline_per_gen": {}}
    measured = [r for r in rows if r["episodes"] is not None]
    if not measured:
        return derived
    last = max(measured, key=lambda r: r["episodes"])
    for head in HEADS:
        derived["suggested_targets"][head] = last["heads"][head]["mean"]
        pts = [
            (r["episodes"], r["heads"][head]["mean"])
            for r in measured
            if r["heads"][head]["mean"] is not None
        ]
        if len(pts) >= 3:
            x = np.array([p[0] for p in pts], dtype=np.float64)
            y = np.array([p[1] for p in pts], dtype=np.float64)
            slope = float(np.polyfit(x, y, 1)[0])
            derived["organic_decline_per_gen"][head] = slope * EPISODES_PER_GEN
        else:
            derived["organic_decline_per_gen"][head] = None
    derived["source_checkpoint"] = last["ckpt"]
    derived["source_episodes"] = last["episodes"]
    return derived


def _print_table(rows: list[dict]) -> None:
    print("\n" + "=" * 88)
    print(f"{'episodes':>10} | " + " | ".join(f"{h + ' Hn (n)':>18}" for h in HEADS))
    print("-" * 88)
    for r in rows:
        ep = f"{r['episodes']:,}" if r["episodes"] is not None else "-"
        cells = []
        for head in HEADS:
            s = r["heads"][head]
            cells.append(
                f"{s['mean']:.4f} ({s['rows']:>5})"
                if s["mean"] is not None
                else f"{'-':>14}"
            )
        print(f"{ep:>10} | " + " | ".join(f"{c:>18}" for c in cells))
    print("=" * 88)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--ckpts",
        nargs="*",
        default=None,
        help="Explicit checkpoint list (default: --glob expansion).",
    )
    ap.add_argument("--glob", default=DEFAULT_GLOB)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=PROBE_SEED)
    ap.add_argument(
        "--schedule-horizon",
        type=int,
        default=20_000_000,
        help="Horizon the swept run trained under (for the coefficient record).",
    )
    ap.add_argument("--out", default=None, help="Path for the JSON report.")
    args = ap.parse_args()

    from sheepshead.agent.ppo import load_agent

    ckpts = args.ckpts if args.ckpts else sorted(glob.glob(args.glob))
    if not ckpts:
        raise SystemExit(f"no checkpoints matched {args.glob}")
    ckpts = sorted(ckpts, key=lambda c: (_episodes_from_name(c) or 0, c))

    rows: list[dict] = []
    for ckpt in ckpts:
        t0 = time.time()
        agent = load_agent(ckpt)
        res = probe_agent(agent, n_games=args.games, seed=args.seed)
        episodes = _episodes_from_name(ckpt)
        rows.append(
            {
                "ckpt": ckpt,
                "episodes": episodes,
                "heads": res["heads"],
                "scheduled_coeffs": (
                    scheduled_coeffs(episodes, args.schedule_horizon)
                    if episodes is not None
                    else None
                ),
            }
        )
        del agent
        ep = f"{episodes:,}" if episodes is not None else "?"
        print(f"[{ep} eps] done ({time.time() - t0:.0f}s)", flush=True)

    _print_table(rows)
    derived = derive(rows)
    print("suggested bumpless targets (last ckpt per-head mean H_norm):")
    for head in HEADS:
        tgt = derived["suggested_targets"].get(head)
        dec = derived["organic_decline_per_gen"].get(head)
        tgt_s = f"{tgt:.4f}" if tgt is not None else "-"
        dec_s = f"{dec:+.4f}" if dec is not None else "-"
        print(f"  {head:>8}: target {tgt_s}   organic drift/gen {dec_s}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "probe_seed": args.seed,
                "games": args.games,
                "schedule_horizon": args.schedule_horizon,
                "ckpts": ckpts,
            },
            "rows": rows,
            "derived": derived,
        }
        out_path.write_text(json.dumps(payload, indent=2))
        print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
