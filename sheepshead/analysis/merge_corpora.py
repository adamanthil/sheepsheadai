#!/usr/bin/env python3
"""Pool two or more schema-2 distillation corpora into one corpus directory
(CE_Teacher_Design §20.13: the iteration-2 corpus + its p = 1.0 top-up,
same theta_k, same acting policy, same anchors).

Shards are re-numbered in order and each corpus's game indices are offset
by the games already pooled, so the fit stage's split-by-game keeps
distinct deals apart (both corpora number their games from 0). Episodes
and their rows are copied untouched. The merged manifest is the first
corpus's with the per-class counters summed, ``games`` / ``kept_games`` /
``episodes`` / ``committee_acted_nodes`` summed, and ``merged_from``
listing the inputs; ``ckpt`` and ``row_schema`` must agree across inputs.

Usage:
  uv run python -m sheepshead.analysis.merge_corpora \\
      --corpus-dir runs/distill_corpus_iter2_202609 \\
      --corpus-dir runs/distill_corpus_iter2b_202609 \\
      --out-dir runs/distill_corpus_iter2_pooled_202609
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

SUMMED_KEYS = (
    "games",
    "kept_games",
    "episodes",
    "committee_act_games",
    "committee_acted_nodes",
)
MUST_MATCH = ("ckpt", "row_schema")


def shard_names(corpus_dir: str) -> list[str]:
    return sorted(
        f
        for f in os.listdir(corpus_dir)
        if f.startswith("corpus_") and f.endswith(".pt")
    )


def offset_games(shard: dict, offset: int) -> dict:
    """Return the shard with every game index shifted by ``offset``."""
    games = [{**g, "game": int(g["game"]) + offset} for g in shard["games"]]
    return {**shard, "games": games}


def merge_manifests(
    manifests: list[dict], inputs: list[str], shards: list[str]
) -> dict:
    merged = json.loads(json.dumps(manifests[0]))
    for key in MUST_MATCH:
        values = {json.dumps(m.get(key)) for m in manifests}
        if len(values) != 1:
            raise SystemExit(f"corpora disagree on {key}: {sorted(values)}")
    for key in SUMMED_KEYS:
        merged[key] = sum(int(m.get(key, 0)) for m in manifests)
    classes: dict[str, dict[str, int]] = {}
    for m in manifests:
        for cls, counters in m.get("classes", {}).items():
            slot = classes.setdefault(cls, {})
            for k, v in counters.items():
                slot[k] = slot.get(k, 0) + v
    merged["classes"] = classes
    merged["merged_from"] = [os.path.abspath(p) for p in inputs]
    merged["shards"] = shards
    merged.pop("gap_percentiles", None)
    return merged


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus-dir", action="append", required=True, dest="corpus_dirs")
    p.add_argument("--out-dir", required=True)
    args = p.parse_args(argv)
    if len(args.corpus_dirs) < 2:
        raise SystemExit("need at least two --corpus-dir inputs")

    os.makedirs(args.out_dir, exist_ok=True)
    manifests = []
    written: list[str] = []
    game_offset = 0
    for corpus_dir in args.corpus_dirs:
        with open(os.path.join(corpus_dir, "manifest.json")) as f:
            manifest = json.load(f)
        manifests.append(manifest)
        n_games = 0
        for name in shard_names(corpus_dir):
            shard = torch.load(
                os.path.join(corpus_dir, name), map_location="cpu", weights_only=False
            )
            out_name = f"corpus_{len(written):04d}.pt"
            torch.save(
                offset_games(shard, game_offset), os.path.join(args.out_dir, out_name)
            )
            written.append(out_name)
            n_games += len(shard["games"])
            print(
                f"[merge] {corpus_dir}/{name} -> {out_name} (+{len(shard['games'])} games)"
            )
        game_offset += n_games
    merged = merge_manifests(manifests, args.corpus_dirs, written)
    with open(os.path.join(args.out_dir, "manifest.json"), "w") as f:
        json.dump(merged, f, indent=2)
    print(
        f"[merge] wrote {len(written)} shards, {game_offset} games, "
        f"searched {sum(c.get('searched', 0) for c in merged['classes'].values())}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
