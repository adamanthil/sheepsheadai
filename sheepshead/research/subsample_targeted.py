#!/usr/bin/env python3
"""Matched-composition control from a targeted corpus (CE_Teacher_Design
§20.12 addendum: the matched-volume control for a STALL read).

Takes the targeted shards a `train_policy_iteration target` stage wrote
and reproduces, post hoc, the row composition of a corpus generated under
a thinner p-schedule:

  games        keep the first N games (episodes are game-contiguous, five
               per game, in shard order) — the corpus-size difference;
  follow-keep  each TARGETED follow row (every searched follow row once
               Stage 2 has run) stays with this probability, otherwise it becomes "none" (no policy
               loss, value stream only) — exactly what the p-schedule does
               to a follow node it passes over. Lead rows and retention
               rows are untouched.

Targets, anchors and the value stream of the kept rows are unchanged, so
the only difference between a distill on the output and one on the input
is the number of rows carrying a policy loss. ``search_weight`` is
re-normalized to mean 1 over the remaining targeted rows, matching the
target stage's own normalization on a corpus of that composition.

Usage:
  uv run python -m sheepshead.research.subsample_targeted \\
      --targeted-dir runs/.../iter11/targeted --out-dir runs/.../ctrl/targeted \\
      --games 2000 --follow-keep 0.5 --seed 20260903
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch

from sheepshead.rl.corpus_rows import load_shard

EPISODES_PER_GAME = 5


def is_follow_row(event: dict) -> bool:
    return event.get("kind") == "action" and str(event.get("node_class", "")).endswith(
        "-follow"
    )


def subsample_episodes(
    episodes: list, follow_keep: float, rng: random.Random
) -> tuple[list, Counter, float]:
    """Demote targeted follow rows to "none" with probability 1 - follow_keep
    and re-normalize the targeted rows' weights. Mutates and returns
    ``episodes``, the counts and the weight mean before renormalization."""
    counts: Counter = Counter()
    mean = 1.0
    for ep in episodes:
        for ev in ep:
            if ev.get("kind") != "action" or ev.get("distill_set") != "targeted":
                continue
            if not is_follow_row(ev):
                counts["lead_targeted_kept"] += 1
                continue
            if rng.random() < follow_keep:
                counts["follow_targeted_kept"] += 1
            else:
                ev["distill_set"] = "none"
                counts["follow_targeted_dropped"] += 1
    weights = [
        float(ev["search_weight"])
        for ep in episodes
        for ev in ep
        if ev.get("kind") == "action"
        and ev.get("distill_set") == "targeted"
        and ev.get("search_weight") is not None
    ]
    if weights:
        mean = sum(weights) / len(weights)
        for ep in episodes:
            for ev in ep:
                if (
                    ev.get("kind") == "action"
                    and ev.get("distill_set") == "targeted"
                    and ev.get("search_weight") is not None
                ):
                    ev["search_weight"] = float(ev["search_weight"]) / mean
    return episodes, counts, mean


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--targeted-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--games", type=int, required=True, help="keep the first N games")
    p.add_argument("--follow-keep", type=float, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    names = sorted(
        f
        for f in os.listdir(args.targeted_dir)
        if f.startswith("corpus_") and f.endswith(".pt")
    )
    if not names:
        raise SystemExit(f"no corpus shards in {args.targeted_dir}")
    rng = random.Random(args.seed)
    budget = args.games * EPISODES_PER_GAME
    total: Counter = Counter()
    written = []
    for name in names:
        if budget <= 0:
            break
        shard = load_shard(os.path.join(args.targeted_dir, name))
        episodes = shard["episodes"][:budget]
        budget -= len(episodes)
        episodes, counts, weight_mean = subsample_episodes(
            episodes, args.follow_keep, rng
        )
        total.update(counts)
        shard["episodes"] = episodes
        torch.save(shard, os.path.join(args.out_dir, name))
        written.append(name)
        print(
            f"[subsample] {name}: {len(episodes)} episodes kept; target weight "
            f"mean before renormalization {weight_mean:.4f}",
            flush=True,
        )

    manifest = {}
    src_manifest = os.path.join(args.targeted_dir, "manifest.json")
    if os.path.exists(src_manifest):
        with open(src_manifest) as f:
            manifest = json.load(f)
    manifest.update(
        {
            "subsampled_from": os.path.abspath(args.targeted_dir),
            "subsample_games": args.games,
            "subsample_follow_keep": args.follow_keep,
            "subsample_seed": args.seed,
            "subsample_counts": dict(total),
            "shards": written,
        }
    )
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[subsample] wrote {len(written)} shards: {dict(total)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
