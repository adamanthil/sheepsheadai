#!/usr/bin/env python3
"""Model-push diagnostics for a finished policy-iteration target stage
(Arm_D_Review_202610 §8): the read Stage 2 now writes into
``target_report.json`` as ``model_diagnostics``, recomputed post hoc for
an iteration whose target stage predates it.

Reads the iteration directory's ``row_table.pt``, ``oof_predictions.pt``
(cross-fit only: an in-sample a_model has seen the labels it is judged
by), ``target_report.json`` (eta / kappa / tilt_max) and the targeted
shards' per-row ``pi_gamma`` and ``pi_v_post``. The saved row table holds
the RAW label noise; the working noise the blend used is recovered per
row as v_post / gamma.

Usage:
  uv run python -m sheepshead.research.target_diagnostics \\
      --iter-dir runs/202609_recall_rc/pi/iter1 [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

from sheepshead.rewards import RETURN_SCALE
from sheepshead.rl.corpus_rows import (
    OOF_FILE,
    TARGETED_SUBDIR,
    load_shard,
    shard_paths,
)
from sheepshead.rl.search_advantage import (
    RowTable,
    format_model_diagnostics,
    model_push_diagnostics,
    tilt_scale,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iter-dir", required=True, help="policy-iteration out dir")
    ap.add_argument("--json", default=None, help="write the diagnostics here")
    args = ap.parse_args(argv)

    oof_path = os.path.join(args.iter_dir, OOF_FILE)
    if not os.path.exists(oof_path):
        print(f"no {OOF_FILE} in {args.iter_dir}: needs a cross-fit iteration")
        return 1
    table = RowTable.load(os.path.join(args.iter_dir, "row_table.pt"))
    a_model = torch.load(oof_path, map_location="cpu", weights_only=False)["a_model"]
    with open(os.path.join(args.iter_dir, "target_report.json")) as f:
        rep = json.load(f)

    gamma = torch.zeros(len(table))
    v_post = torch.zeros(len(table))
    by_shard: dict[int, list[int]] = {}
    for r, (s, _, _) in enumerate(table.refs):
        by_shard.setdefault(s, []).append(r)
    paths = shard_paths(os.path.join(args.iter_dir, TARGETED_SUBDIR))
    for s, rows in by_shard.items():
        episodes = load_shard(paths[s])["episodes"]
        for r in rows:
            _, e, k = table.refs[r]
            ev = episodes[e][k]
            gamma[r] = float(ev["pi_gamma"])
            v_post[r] = float(ev["pi_v_post"])
    noise = torch.where(gamma > 0, v_post / gamma.clamp(min=1e-12), table.noise_var)

    diag = model_push_diagnostics(
        table,
        a_model,
        gamma,
        tilt_scale(v_post, kappa=float(rep["kappa"]), eta=rep.get("eta")),
        noise,
        tilt_max=float(rep["tilt_max"]),
        out_of_fold=True,
    )
    for line in format_model_diagnostics(diag, RETURN_SCALE):
        print(line)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(diag, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
