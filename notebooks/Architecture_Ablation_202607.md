# Architecture Ablation — Experiment Log (July 2026)

Companion to `Evaluation_Harnesses_202607.md` §5 (protocol) and
`architectures.py` (the registry + literature). This notebook is the record
of the actual experiment: exact commands, environment, output schema, and
the final results.

**Question:** what did each historical architecture addition buy, in
training speed (sample efficiency AND wall-clock) and in self-play-regime
skill ceiling? Each adjacent ladder rung removes exactly one addition;
`full-uninformed` is a factorial arm testing informed init in the presence
of the transformer.

## Design

- **6 architectures** (`architectures.py` registry): `full`,
  `full-uninformed`, `no-aux`, `no-transformer` (PooledMemoryEncoder — see
  §5 footnote 2 of the harnesses notebook), `no-transformer-uninformed`,
  `onehot-ff`.
- **3 seeds** {42, 1042, 2042} × **100k episodes** pure self-play per arm =
  **18 runs**. `--seed` fully determines a run (init, sampling, and deals);
  arms sharing a seed see identical deal sequences (CRN across arms).
- **In-training curve:** paired 300-deal CRN edges every 5k episodes vs
  three fixed yardsticks (ScriptedAgent / selfplay-100k reference /
  `final_pfsp_swish_ppo.pt`), frozen `ANCHOR_EVAL_SEED = 20260703`. Eval
  wall-clock excluded from training wall-clock.
- **Endpoints per run:** scripted paired probe (500 deals, seed 31) +
  trump-lead incidence probe (2000 deals, seed 20260702).
- **Endpoints per matrix:** PANEL-A gauntlet over all 18 finals, both
  partner modes, 1000 deals, seed 42 (frozen panel; MDE ≈ 0.07 score/hand).
- **Concurrency:** 8 simultaneous training subprocesses, 1 BLAS thread each
  (Apple M1 Max, 10 cores, 64 GB; game logic is Python-bound so process
  parallelism wins). Finished slot ⇒ next queued job starts automatically.

## Environment

- Machine: Apple M1 Max (10 cores), 64 GB, macOS (Darwin 23.6.0)
- Python 3.14 (`uv` venv), torch 2.11.0, CPU only
- Code: commit `94b7ab4` lineage — registry `3e89023`, trainer plumbing
  `da60615`, pooled-memory rung `978b244`, activation removal `b43f908`
- Launched: 2026-07-04

## Exact commands

Orchestrator (manages the queue, per-run probes, PANEL-A, aggregation;
resumable — rerunning it skips finished jobs):

```bash
mkdir -p runs/ablation_202607
PYTHONPATH=. nohup .venv/bin/python analysis/run_ablation_matrix.py \
    > runs/ablation_202607/orchestrator.out 2>&1 &
```

Which executes, per (arch, seed) job — exact argv also recorded in
`runs/ablation_202607/status/<run>.json`:

```bash
PYTHONPATH=. .venv/bin/python train_selfplay_ppo.py \
    --arch <ARCH> --seed <SEED> --episodes 100000 \
    --run-name ablate_<ARCH>_s<SEED> \
    --anchor-eval-interval 5000 --anchor-eval-deals 300 \
    --save-interval 25000 --strategic-eval-interval 1000000
# then, on the run's final checkpoint:
PYTHONPATH=. .venv/bin/python analysis/scripted_probe.py \
    --ckpt runs/ablate_<ARCH>_s<SEED>/final_<ARCH>.pt --deals 500 \
    --out-json runs/ablate_<ARCH>_s<SEED>/scripted_probe.json
PYTHONPATH=. .venv/bin/python analysis/trump_lead_probe.py \
    --ckpt runs/ablate_<ARCH>_s<SEED>/final_<ARCH>.pt --deals 2000 \
    --out-json runs/ablate_<ARCH>_s<SEED>/trump_lead_probe.json
```

And once all 18 runs finish (finals copied to
`runs/ablation_202607/finals/<arch>__s<seed>.pt`):

```bash
PYTHONPATH=. .venv/bin/python analysis/rigorous_eval.py \
    --candidates runs/ablation_202607/finals/*.pt \
    --anchors final_pfsp_swish_ppo.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt \
              runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt \
    --deals 1000 --partner-mode called --seed 42 \
    --out-csv runs/ablation_202607/panel_a_called.csv \
    --out-plot runs/ablation_202607/panel_a_called.png
# repeated with --partner-mode jd
PYTHONPATH=. .venv/bin/python analysis/aggregate_ablation.py   # CSVs + plots + table
```

## Outputs (all under `runs/ablation_202607/`)

| file | contents |
|---|---|
| `learning_curves.csv` | long format, one row per (run, eval point): `run, arch, seed, episode, train_wall_s, eval_wall_s, updates_done, transitions_done, edge_<yard>, se_<yard>` for yard ∈ {scripted, selfplay100k, final_pfsp} — **the graphing source of truth** |
| `results.csv` / `results_table.md` | one row per run: final edges, endpoint probes, PANEL-A scores, wall-clock, eps/s |
| `curves_{scripted,selfplay100k,final_pfsp}.png` | pre-built learning-curve plots (bold = per-arch seed mean, faint = seeds) |
| `panel_a_{called,jd}.csv/.png` | PANEL-A gauntlet over all finals |
| `orchestrator.log` | timestamped job lifecycle (start/end/rc/duration of every command) |
| `status/<run>.json` | per-run status, exact train argv, wall minutes |
| `runs/ablate_<arch>_s<seed>/` | per-run: `train.log`, `probes.log`, `anchored_eval.csv`, checkpoints at 25k/50k/75k, `final_<arch>.pt`, probe JSONs |

Plotting from the long CSV, e.g.:

```python
import pandas as pd
df = pd.read_csv("runs/ablation_202607/learning_curves.csv")
ax = None
for (arch,), g in df.groupby(["arch"]):
    m = g.groupby("episode")["edge_scripted"].mean()
    ax = m.plot(label=arch, ax=ax)
ax.axhline(0, ls="--", c="gray"); ax.legend(); ax.set_ylabel("edge vs scripted")
```

## Decision criteria (pre-registered)

- **Training speed:** median (across seeds) episodes AND train-wall-clock to
  fixed edge thresholds vs scripted / selfplay-100k. Two separate axes —
  cheap archs win s/episode; the open question is sample efficiency.
- **Ceiling (self-play regime):** mean final PANEL-A score/hand; deltas
  > 0.07 treated as real. Report per-seed spread and collapse count (edge
  regressing > 2 SE from its own peak). 100k-episode self-play measures
  *learnability*, not the league-regime ceiling (phase 2 = top-2 archs
  through `train_league_ppo.py --arch`).
- **Component verdicts:** adjacent-rung deltas (full−no-aux = aux heads;
  no-aux−no-transformer = transformer; no-transformer−…-uninformed =
  informed init; …-uninformed−onehot-ff = card-token pipeline;
  full−full-uninformed = informed init under the transformer).

## Status

- 2026-07-04: matrix launched (18 jobs, 8 concurrent). Results pending —
  this section and the results table below are filled in on completion.

## Results

*(pending — populated from `results_table.md` when the matrix completes)*
