#!/usr/bin/env python3
"""Projection realization at the convention lead rows (CE_Teacher_Design
§20.8-§20.9): how far a checkpoint's policy moved toward the §20 targets
at trick-0/1 defender leads with a called-suit option, on the corpus's
TRAINING games and on its HELD-OUT games (the same game-level split the
distill stage used), for one or more checkpoints.

Reports per cell: rows, the policy's called-suit probability mass vs the
target's, the greedy argmax rate on the called suit vs the target's, and
the realized fraction of the target's mass shift relative to the first
checkpoint given (pass theta_k first).

Usage:
  uv run python -m sheepshead.analysis.lead_row_realization \\
      --targeted-dir runs/policy_iteration_202609/iter3/targeted \\
      --ckpt theta_k=runs/.../checkpoint_8000000.pt \\
      --ckpt arm3=runs/policy_iteration_202609/iter3/distill_epoch1.pt
"""

from __future__ import annotations

import argparse
import collections
import sys

import numpy as np

from sheepshead.agent.ppo import load_agent
from sheepshead.training import train_distill
from sheepshead.training.corpus_rows import (
    iter_row_batches,
    replayed_policy_rows,
    store_corpus_episodes,
)

CELLS = ("std|t0-defender-lead", "std|t1-defender-lead")


def _is_lead_row(e: dict) -> bool:
    return (
        e.get("kind") == "action"
        and bool(e.get("pi_target_source"))
        and bool(e.get("conv_cs_ids"))
        and e.get("node_class") in CELLS
    )


def measure(agent, episodes: list, buffer: int = 250) -> dict:
    """Per cell: n, policy called-suit mass, target mass, policy argmax
    rate, target argmax rate."""
    acc: dict = collections.defaultdict(
        lambda: {"n": 0, "mass": 0.0, "tmass": 0.0, "arg": 0, "targ": 0}
    )
    for start in range(0, len(episodes), buffer):
        chunk = episodes[start : start + buffer]
        agent.reset_storage()
        smap = store_corpus_episodes(agent, chunk)
        for rows in iter_row_batches(agent, batch_segments=32):
            probs = replayed_policy_rows(agent, rows)
            for r, ev_idx in enumerate(rows.event_indices):
                ref = smap[ev_idx]
                assert ref is not None
                src = chunk[ref[0]][ref[1]]
                if not _is_lead_row(src):
                    continue
                acts = sorted(src["valid_actions"])
                t = np.asarray(src["search_target"], dtype=np.float64)
                p = np.asarray([float(probs[r][a - 1]) for a in acts], dtype=np.float64)
                p = p / max(p.sum(), 1e-12)
                cs = set(src["conv_cs_ids"])
                idx = [i for i, a in enumerate(acts) if a in cs]
                a = acc[src["node_class"]]
                a["n"] += 1
                a["mass"] += float(p[idx].sum())
                a["tmass"] += float(t[idx].sum())
                a["arg"] += int(int(p.argmax()) in idx)
                a["targ"] += int(int(t.argmax()) in idx)
        agent.reset_storage()
    return acc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--targeted-dir", required=True)
    ap.add_argument(
        "--ckpt", action="append", required=True, help="name=path (first = reference)"
    )
    ap.add_argument("--holdout-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train-episodes", type=int, default=4000)
    args = ap.parse_args(argv)

    episodes = train_distill.load_shards(args.targeted_dir)
    train_eps, hold = train_distill.split_by_game(
        episodes, args.holdout_frac, args.seed
    )
    keep = lambda eps: [ep for ep in eps if any(_is_lead_row(e) for e in ep)]  # noqa: E731
    train_eps = keep(train_eps)[: args.max_train_episodes]
    hold = keep(hold)
    results = {}
    names = []
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        names.append(name)
        agent = load_agent(path)
        for net in (agent.encoder, agent.actor, agent.critic):
            for prm in net.parameters():
                prm.requires_grad_(False)
        results[name] = {
            "train": measure(agent, train_eps),
            "holdout": measure(agent, hold),
        }
    ref = names[0]
    for split in ("train", "holdout"):
        print(f"== {split} ==")
        for cell in CELLS:
            r0 = results[ref][split].get(cell)
            if not r0 or r0["n"] == 0:
                continue
            n = r0["n"]
            print(
                f"  {cell:22s} n={n:4d} target mass {r0['tmass'] / n:.3f} "
                f"argmax {100 * r0['targ'] / n:.1f}"
            )
            for name in names:
                a = results[name][split][cell]
                shift = (a["mass"] - r0["mass"]) / max(r0["tmass"] - r0["mass"], 1e-9)
                print(
                    f"    {name:14s} mass {a['mass'] / n:.3f} argmax {100 * a['arg'] / n:.1f}"
                    + ("" if name == ref else f"  realized {100 * shift:.0f}%")
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
