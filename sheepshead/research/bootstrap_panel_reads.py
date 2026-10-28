"""Panel reads for a run's bootstrap phase, under the orchestrator's
endpoint protocol.

The orchestrator reads PANEL-A and PANEL-B at every league generation and
policy iteration but never inside the bootstrap phase, so the skill
trajectory (``plot_program_skill``) starts at the seed. This fills the gap:
for each stage N it reads the three-checkpoint composite (N-100k, N-50k,
N), the same protocol as the league generations, on both panels, writing
``panel_boot<N>.npz`` / ``panelB_boot<N>.npz`` into the program directory
next to the orchestrator's own reads. Existing outputs are skipped, so the
run can be resumed; register the files in ``build_panel_reference``.

    uv run python -m sheepshead.research.bootstrap_panel_reads \\
        --run runs/202609_recall_rc --stages 150000 200000 250000 \\
        300000 350000 400000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from sheepshead.eval.league_progress_eval import (
    PANEL_SEED,
    eval_endpoint,
)

STEP = 50_000


def composite(ckpt_dir: Path, stage: int) -> list[str]:
    paths = [
        ckpt_dir / f"checkpoint_{n}.pt" for n in (stage - 2 * STEP, stage - STEP, stage)
    ]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"stage {stage}: missing {missing}")
    return [str(p) for p in paths]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--run", required=True, help="run directory (holds program/ and bootstrap/)"
    )
    ap.add_argument("--stages", type=int, nargs="+", required=True)
    ap.add_argument("--panels", nargs="+", default=["A", "B"], choices=["A", "B"])
    args = ap.parse_args(argv)
    run = Path(args.run)
    with (run / "program" / "config.json").open() as f:
        league = json.load(f)["league"]
    members = {"A": league["panel"], "B": league["panel_b"]}
    prefix = {"A": "panel", "B": "panelB"}
    deals = int(league.get("panel_deals", 3996))
    for stage in args.stages:
        ckpts = composite(run / "bootstrap" / "checkpoints", stage)
        for panel in args.panels:
            out = run / "program" / f"{prefix[panel]}_boot{stage}.npz"
            if out.exists():
                print(f"{out} exists, skipped", flush=True)
                continue
            t0 = time.time()
            print(f"PANEL-{panel} bootstrap {stage}: {deals} deals ...", flush=True)
            e = eval_endpoint(
                ckpts,
                n_deals=deals,
                seed=PANEL_SEED,
                panel_paths=tuple(members[panel]),
                out_npz=out,
            )
            print(
                f"PANEL-{panel} bootstrap {stage}: {e.score.mean:+.4f} "
                f"[{e.score.lo:+.4f}, {e.score.hi:+.4f}] in {(time.time() - t0) / 60:.0f} min",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
