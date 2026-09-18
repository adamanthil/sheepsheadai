"""Re-run the greedy health probe on saved checkpoints.

Writes ``greedy_health.csv``'s schema (train_ppo.GREEDY_CSV_HEADER), one
row per checkpoint, seeded exactly as the trainer seeds its in-run probe
(``seed = episode``), so the rows are comparable with the ones the trainer
wrote — and the conventions columns reproduce the trainer's when the probe
definition has not moved. Use it when a probe column is added or redefined
after a phase ran (Training_Program_Redesign §7.1, 09-16: the seen-trump
memory columns) to get one consistent series across the run.

    uv run python -m sheepshead.analysis.reprobe_checkpoints \
        runs/<run>/checkpoints/checkpoint_*.pt \
        --out runs/<run>/checkpoints/greedy_health_recall.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import time

from sheepshead.agent.ppo import load_agent
from sheepshead.training.train_ppo import (
    GREEDY_CSV_HEADER,
    aux_summary,
    episode_of,
    greedy_csv_row,
)
from sheepshead.training.training_utils import greedy_health_probe


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("checkpoints", nargs="+", help="checkpoint .pt files")
    ap.add_argument("--games", type=int, default=200, help="greedy games per probe")
    ap.add_argument("--out", required=True, help="CSV to write (overwritten)")
    args = ap.parse_args()

    paths = sorted(args.checkpoints, key=episode_of)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(GREEDY_CSV_HEADER)
        for path in paths:
            episode = episode_of(path)
            t0 = time.time()
            agent = load_agent(path)
            probe = greedy_health_probe(agent, n_games=args.games, seed=episode)
            writer.writerow(greedy_csv_row(episode, probe))
            f.flush()
            print(
                f"{os.path.basename(path)} (ep {episode:,}, {time.time() - t0:.0f}s): "
                f"PICK {probe['pick_rate']:.1f}%, leaster {probe['leaster_rate']:.1f}%, "
                f"called-suit lead {probe['called_suit_lead_rate']:.1f}%, "
                f"{aux_summary(probe)}",
                flush=True,
            )
    print(f"wrote {len(paths)} rows -> {args.out}")


if __name__ == "__main__":
    main()
