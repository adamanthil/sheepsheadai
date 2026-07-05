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
- **Reward regime: SHAPED** (the trainer's historical bootstrap stack:
  hand-conditioned pick/pass nudges + per-trick intermediate rewards +
  final score/RETURN_SCALE with leaster bonus — `process_episode_rewards`
  + `update_intermediate_rewards_for_action`). Deliberate: identical
  shaping across arms keeps the comparison controlled; it matches the
  regime in which these components were historically adopted; and
  cold-start terminal-only at 100k episodes risks a floor effect (the
  terminal-reward league run warm-started from a shaped policy — cold
  terminal bootstrap is untested). **Pre-registered caveat:** dense shaping
  does some of the work the aux heads / critic do, so shaped-regime deltas
  for those rungs (esp. `full − no-aux`) are best read as a *lower bound*
  on their terminal-regime value; phase 2 (league trainer, terminal-only)
  is the regime check.

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

- 2026-07-04 (morning): matrix launched (18 jobs, 8 concurrent).
- 2026-07-04 19:40: **matrix complete — 18/18 runs ok, 0 failures, 19.8 h
  wall** (incl. PANEL-A over all finals, 136 min/mode). No collapse in any
  run. 100k-episode artifacts snapshotted as `*_100k.*`.
- 2026-07-04 evening: pre-registered extension rule applied — `full` and
  `no-aux` (top-2 still-climbing arms) resumed to 200k episodes, all seeds.

## Results — 100k episodes (18 runs)

Full per-run table: `runs/ablation_202607/results_table_100k.md` /
`results_100k.csv`; curves: `curves_*_100k.png`; raw curve data:
`learning_curves_100k.csv`.

### Per-arch endpoint summary (mean ± seed-std over 3 seeds)

PANEL-A = score/hand vs the frozen reference field (higher = stronger;
per-measurement SE ≈ 0.04, but **seed-to-seed spread dominates** — exactly
the Henderson-2018 point the 3-seed design anticipated).

| arch | PANEL-A called | PANEL-A jd | PANEL-A both | scripted edge (500) | train h/run | eps/s |
|---|---|---|---|---|---|---|
| `full` | −0.431 ± 0.071 | −0.394 ± 0.156 | **−0.413** | −0.097 ± 0.098 | 7.9 | 3.5 |
| `full-uninformed` | −0.566 ± 0.103 | −0.559 ± 0.063 | −0.562 | −0.249 ± 0.117 | 7.8 | 3.6 |
| `no-aux` | −0.463 ± 0.107 | −0.378 ± 0.052 | **−0.420** | −0.288 ± 0.064 | 7.4 | 3.8 |
| `no-transformer` | −0.511 ± 0.150 | −0.577 ± 0.184 | −0.544 | −0.295 ± 0.347 | 4.1 | 6.8 |
| `no-transformer-uninformed` | −0.654 ± 0.167 | −0.555 ± 0.159 | −0.605 | −0.461 ± 0.269 | 3.9 | 7.1 |
| `onehot-ff` | −0.423 ± 0.128 | −0.351 ± 0.070 | **−0.387** | −0.065 ± 0.113 | 1.6 | 17.4 |

### Adjacent-rung component deltas (PANEL-A both-modes mean; + = helps)

| component | comparison | delta | seed-level SE | verdict at 100k/shaped |
|---|---|---|---|---|
| aux heads | full − no-aux | +0.007 | 0.077 | **null** (pre-registered caveat: shaping supplies the dense signal aux heads would; lower bound on terminal-regime value) |
| transformer | no-aux − no-transformer | +0.124 | 0.102 | positive, ~1.2 SE; transformer arms still climbing while no-transformer flattened — likely understated |
| informed init (under transformer) | full − full-uninformed | **+0.150** | 0.075 | **clearest positive component result** (2 SE) |
| informed init (no-transformer) | no-transformer − …-uninformed | +0.060 | 0.132 | positive but noisy |
| card-token pipeline | …-uninformed − onehot-ff | **−0.218** | 0.106 | *negative at this scale* — see below |

### The onehot-ff surprise, and how to read it

`onehot-ff` matches `full` on every endpoint at 100k (PANEL-A −0.387 vs
−0.413; scripted −0.065 vs −0.097 — both within seed noise) while training
**5× faster wall-clock**. The curve shapes explain it (edge vs scripted,
seed means):

| arch | 20k | 60k | 100k | 80k→100k slope |
|---|---|---|---|---|
| `onehot-ff` | **−0.29** | −0.34 | −0.23 | +0.10 (≈ flat, < 1 SE) |
| `full` | −0.49 | −0.38 | −0.24 | **+0.24** (> 2 SE) |
| `no-aux` | −0.45 | −0.30 | −0.21 | **+0.31** (> 2 SE) |

Classic fast-start / early-plateau vs slow-start / still-climbing: the flat
one-hot rep with a wide first layer (it is actually the *largest* net,
1.13M params) fits the shaped bootstrap signal fastest, but stops
improving by ~20–60k; the token/transformer arms keep climbing through the
budget end. **At bootstrap scale the token pipeline has not yet paid for
itself — this is a horizon statement, not a ceiling statement.** The
ceiling question passes to the 200k extension below and to phase 2
(terminal-reward league regime). Two ladder confounds to keep in mind:
`onehot-ff` also swaps the actor (flat heads vs pointer) and its parameter
count is not matched.

Other observations:
- **No run collapsed** (0/18; collapse-count metric clean). Trump-lead
  incidence at 100k is scattered 0–6% with two 5%+ outliers
  (`full` s42 called 5.5%, `full-uninformed` s1042 called 5.9%).
- **Informed init matters more WITH the transformer** (+0.150) than
  without (+0.060) — attention amplifies good card geometry.
- Wall-clock ranking (h/100k, 1 thread): onehot 1.6 < no-transformer ~4 <
  full-family ~7.7. Sample-efficiency and wall-clock verdicts differ, as
  pre-registered.

### Extension (pre-registered rule)

Rule: top-2 arms +100k episodes if last-20k slope > 1 SE above zero.
Slopes (edge vs selfplay-100k, 80k→100k, seed mean): full **+0.236**
(fires), no-aux **+0.308** (fires), full-uninformed +0.156 (fires),
no-transformer-uninformed +0.148 (fires), no-transformer +0.020 (no),
onehot-ff +0.098 (no). Top-2 *still-climbing* arms = **full** and
**no-aux** (onehot-ff ranks high but is flat ⇒ not extended).

Extension commands (resumed in place; same run dirs, anchored_eval.csv
continues; entropy schedule refers to the new 200k horizon, identical
treatment for both extended arms):

```bash
PYTHONPATH=. .venv/bin/python train_selfplay_ppo.py \
    --arch <full|no-aux> --seed <S> --episodes 200000 \
    --resume runs/ablate_<A>_s<S>/<A>_checkpoint_100000.pt \
    --run-name ablate_<A>_s<S> \
    --anchor-eval-interval 5000 --anchor-eval-deals 300 \
    --save-interval 25000 --strategic-eval-interval 10000000
```

## Results — 200k extension (full, no-aux)

*(pending — filled when the extension completes)*
