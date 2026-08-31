#!/usr/bin/env python3
"""Zero-fine-tuning blind/bury observation ablation (picker EV probe).

Measures how much EV the trained agent loses when the blind/bury tokens are
removed from its observations with NO retraining — i.e. how much information
those tokens actually carry beyond what the GRU memory already holds. This is
the cheap go/no-go gate for the planned architecture change that drops
``blind_ids``/``bury_ids`` from the observation dict entirely (the picker is
the only seat that ever sees them populated — game.py builds them empty for
everyone else — so today's encoder re-injects the picker's blind/bury
knowledge every step instead of requiring the memory to carry it).

Instrument: duplicate-bridge h2h (league_progress_eval.h2h_duplicate design)
with an exactly-zero null on non-picker hands. The hero plays the SAME
weights as the field, but with blind/bury PAD-masked at the encoder input.
Because (a) pre-pick observations are identical under the mask, (b) play is
deterministic (argmax), and (c) the field is unablated, the two arms produce
bit-identical hands whenever the hero does not pick — the paired per-hand
diff is nonzero ONLY on hero-picker hands, where the ablation actually binds
(bury decision, called-card decision, picker play). We therefore play the
ablated arm only on hero-picker hands (--verify-identity replays every hand
and asserts the zero-null instead, for validation runs).

The masking is a pure input ablation: masked blind/bury tokens are excluded
from attention via key_padding_mask, so no weights are touched and no
architecture surgery is needed — checkpoints load unchanged.

Reported: overall duplicate edge (= pick-rate-weighted cost), the
picker-conditional per-hand EV diff, and the fraction of picker hands whose
outcome changed. Bootstrap is deal-clustered (bootstrap.py conventions).

Usage:
  uv run python -m sheepshead.analysis.blind_bury_ablation \
      --ckpt final_pfsp_swish_ppo.pt --deals-per-mode 1000 --workers 7 \
      --out-json runs/blind_bury_ablation_202608/prod30m.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from multiprocessing import get_context
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sheepshead import PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game

H2H_SEED = 42  # matches the h2h_duplicate deal-seed schedule
N_BOOT = 10_000

_W: Dict[str, Any] = {}  # per-worker state


# --------------------------------------------------------------------------- #
# Ablation wrapper
# --------------------------------------------------------------------------- #
_ZERO2 = np.zeros(2, dtype=np.uint8)


def ablate_blind_bury(agent: Any) -> Any:
    """Patch ``agent`` so its encoder sees PAD blind/bury in every observation.

    Instance-level patch of ``encoder.encode_batch`` (every inference path —
    act / observe / get_action_probs_with_logits — funnels through it). PAD
    ids produce all-False masks, so the tokens drop out of attention and
    pooling exactly as always-empty bags do for non-picker seats.
    """
    enc = agent.encoder
    orig = enc.encode_batch

    def encode_batch_ablated(
        batch: List[Dict[str, Any]],
        memory_in: Any = None,
        device: Any = None,
    ) -> Dict[str, Any]:
        stripped = []
        for s in batch:
            s2 = dict(s)
            s2["blind_ids"] = _ZERO2
            s2["bury_ids"] = _ZERO2
            stripped.append(s2)
        return orig(stripped, memory_in=memory_in, device=device)

    enc.encode_batch = encode_batch_ablated
    return agent


# --------------------------------------------------------------------------- #
# Game driver (mirrors rigorous_eval.play_hand stepping exactly)
# --------------------------------------------------------------------------- #
def _play_hand(
    hero_kind: str, mode: int, deal_seed: int, hero_seat: int
) -> Tuple[float, int, bool]:
    """One hand: hero at ``hero_seat`` (baseline or ablated weights-identical
    agent), baseline field elsewhere. Returns (hero_score, picker, is_leaster).
    """
    hero = _W["abl"] if hero_kind == "abl" else _W["base"]
    field = _W["base"]
    game = Game(partner_selection_mode=mode, seed=deal_seed)
    hero.reset_recurrent_state()
    field.reset_recurrent_state()

    while not game.is_done():
        for player in game.players:
            agent = hero if player.position == hero_seat else field
            valid = player.get_valid_action_ids()
            while valid:
                state = player.get_state_dict()
                action, _, _ = agent.act(
                    state, valid, player.position, deterministic=True
                )
                player.act(action)
                valid = player.get_valid_action_ids()
                if game.was_trick_just_completed:
                    for seat in game.players:
                        seat_agent = hero if seat.position == hero_seat else field
                        seat_agent.observe(
                            seat.get_last_trick_state_dict(),
                            player_id=seat.position,
                        )

    return (
        float(game.players[hero_seat - 1].get_score()),
        int(game.picker),
        bool(game.is_leaster),
    )


def _worker_init(ckpt: str, torch_threads: int) -> None:
    import torch

    torch.set_num_threads(torch_threads)
    from sheepshead.agent.ppo import load_agent

    _W["base"] = load_agent(ckpt)
    _W["abl"] = ablate_blind_bury(load_agent(ckpt))


def _run_deal(task: Tuple[int, int, int, bool]) -> Dict[str, Any]:
    """One CRN deal in one mode: hero in all 5 seats, both arms.

    The ablated arm is replayed only when the hero picked (the paired diff is
    provably zero otherwise); ``verify`` replays every hand and asserts that.
    """
    mode, deal_seed, deal_idx, verify = task
    t0 = time.time()
    cells = []
    for hero_seat in range(1, 6):
        s_base, picker, is_leaster = _play_hand("base", mode, deal_seed, hero_seat)
        hero_is_picker = picker == hero_seat
        if hero_is_picker or verify:
            s_abl, picker_a, leaster_a = _play_hand("abl", mode, deal_seed, hero_seat)
            # Pre-pick observations are identical under the mask, so the pick
            # sequence (and thus role assignment) must match across arms.
            if picker_a != picker or leaster_a != is_leaster:
                raise AssertionError(
                    f"role divergence: deal {deal_idx} seat {hero_seat} "
                    f"picker {picker}->{picker_a} leaster {is_leaster}->{leaster_a}"
                )
            if not hero_is_picker and s_abl != s_base:
                raise AssertionError(
                    f"nonzero null: deal {deal_idx} seat {hero_seat} "
                    f"{s_base} -> {s_abl} without hero picking"
                )
        else:
            s_abl = s_base
        cells.append(
            {
                "seat": hero_seat,
                "score_base": s_base,
                "score_abl": s_abl,
                "hero_is_picker": hero_is_picker,
                "is_leaster": is_leaster,
            }
        )
    return {
        "deal": deal_idx,
        "mode": "called" if mode == PARTNER_BY_CALLED_ACE else "jd",
        "cells": cells,
        "wall_s": time.time() - t0,
    }


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #
def _deal_clustered_picker_diff(
    deal_diffs: List[List[float]], rng: np.random.Generator, n_boot: int = N_BOOT
) -> Dict[str, float]:
    """Mean per-hand picker diff with a deal-clustered bootstrap interval.

    ``deal_diffs[d]`` holds the (score_abl - score_base) values of deal d's
    hero-picker cells (possibly empty). Deals are the resampling unit.
    """
    n = len(deal_diffs)
    sums = np.array([float(np.sum(d)) for d in deal_diffs])
    counts = np.array([float(len(d)) for d in deal_diffs])
    total = counts.sum()
    if total == 0:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0, "se": 0.0, "n_cells": 0}
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_counts = counts[idx].sum(axis=1)
    boot_means = np.divide(
        sums[idx].sum(axis=1),
        boot_counts,
        out=np.zeros(n_boot),
        where=boot_counts > 0,
    )
    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return {
        "mean": float(sums.sum() / total),
        "lo": float(lo),
        "hi": float(hi),
        "se": float(boot_means.std(ddof=1)),
        "n_cells": int(total),
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    from sheepshead.analysis.bootstrap import (
        bootstrap_deal_indices,
        bootstrap_interval,
        interval_to_dict,
    )

    out: Dict[str, Any] = {"modes": {}}
    pooled_deal_diff: List[float] = []
    pooled_picker_diffs: List[List[float]] = []
    pooled_counts = {"cells": 0, "picker": 0, "leaster": 0, "changed": 0}

    for mode_name in ("called", "jd"):
        rows = [r for r in results if r["mode"] == mode_name]
        rows.sort(key=lambda r: r["deal"])
        deal_diff = []
        picker_diffs = []
        counts = {"cells": 0, "picker": 0, "leaster": 0, "changed": 0}
        for r in rows:
            diffs = [c["score_abl"] - c["score_base"] for c in r["cells"]]
            deal_diff.append(float(np.mean(diffs)))
            pdiffs = [
                c["score_abl"] - c["score_base"]
                for c in r["cells"]
                if c["hero_is_picker"]
            ]
            picker_diffs.append(pdiffs)
            counts["cells"] += len(r["cells"])
            counts["picker"] += sum(c["hero_is_picker"] for c in r["cells"])
            counts["leaster"] += sum(c["is_leaster"] for c in r["cells"])
            counts["changed"] += sum(1 for d in pdiffs if d != 0.0)

        rng = np.random.default_rng(H2H_SEED)
        deal_diff_arr = np.array(deal_diff)
        boot_idx = bootstrap_deal_indices(len(deal_diff_arr), N_BOOT, rng)
        out["modes"][mode_name] = {
            "edge": interval_to_dict(bootstrap_interval(deal_diff_arr, boot_idx)),
            "picker_diff": _deal_clustered_picker_diff(picker_diffs, rng),
            "counts": counts,
            "n_deals": len(rows),
        }
        pooled_deal_diff.extend(deal_diff)
        pooled_picker_diffs.extend(picker_diffs)
        for k in pooled_counts:
            pooled_counts[k] += counts[k]

    rng = np.random.default_rng(H2H_SEED + 1)
    pooled_arr = np.array(pooled_deal_diff)
    boot_idx = bootstrap_deal_indices(len(pooled_arr), N_BOOT, rng)
    out["pooled"] = {
        "edge": interval_to_dict(bootstrap_interval(pooled_arr, boot_idx)),
        "picker_diff": _deal_clustered_picker_diff(pooled_picker_diffs, rng),
        "counts": pooled_counts,
        "n_deals": len(pooled_arr),
    }
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--deals-per-mode", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--torch-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=H2H_SEED)
    ap.add_argument(
        "--verify-identity",
        action="store_true",
        help="replay the ablated arm on EVERY hand and assert the zero null "
        "on non-picker hands (validation mode; ~5x slower)",
    )
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args(argv)

    seed_rng = random.Random(args.seed)
    deal_seeds = [seed_rng.randint(0, 2**31 - 1) for _ in range(args.deals_per_mode)]
    tasks = []
    for d, s in enumerate(deal_seeds):
        tasks.append((PARTNER_BY_CALLED_ACE, s, d, args.verify_identity))
        tasks.append((PARTNER_BY_JD, s, d, args.verify_identity))

    results: List[Dict[str, Any]] = []
    done = 0
    t_start = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(
        processes=args.workers,
        initializer=_worker_init,
        initargs=(args.ckpt, args.torch_threads),
    ) as pool:
        for res in pool.imap_unordered(_run_deal, tasks, chunksize=1):
            results.append(res)
            done += 1
            if done % 100 == 0 or done == len(tasks):
                rate = done / (time.time() - t_start)
                print(
                    f"  {done}/{len(tasks)} deal-modes "
                    f"({rate:.2f}/s, eta {(len(tasks) - done) / rate:.0f}s)",
                    flush=True,
                )

    summary = aggregate(results)
    summary["config"] = {
        "ckpt": os.path.abspath(args.ckpt),
        "deals_per_mode": args.deals_per_mode,
        "seed": args.seed,
        "verify_identity": args.verify_identity,
        "instrument": "duplicate_bridge_ablation",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    p = summary["pooled"]
    print(f"\n=== blind/bury ablation: {os.path.basename(args.ckpt)} ===")
    print(
        f"overall edge (abl - base): {p['edge']['mean']:+.4f} "
        f"[{p['edge']['lo']:+.4f}, {p['edge']['hi']:+.4f}] "
        f"se {p['edge']['se']:.4f}  (n_deals {p['n_deals']})"
    )
    pd = p["picker_diff"]
    print(
        f"picker-hand EV diff:       {pd['mean']:+.4f} "
        f"[{pd['lo']:+.4f}, {pd['hi']:+.4f}] se {pd['se']:.4f} "
        f"(n_picker {pd['n_cells']})"
    )
    c = p["counts"]
    print(
        f"hero-picker rate {c['picker'] / c['cells']:.3f}  "
        f"leaster rate {c['leaster'] / c['cells']:.3f}  "
        f"picker hands w/ changed outcome {c['changed']}/{c['picker']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
