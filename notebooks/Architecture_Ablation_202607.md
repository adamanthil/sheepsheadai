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
- 2026-07-05 05:20: **extension complete** (6/6 ok). Experiment closed;
  conclusions below.

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

Extension complete 2026-07-05 05:20 (6 runs resumed 100k→200k, ~9.6 h wall;
probes + PANEL-A on the 200k finals in `panel_a_ext_{called,jd}.csv`).
Note: `results.csv`'s `panel_a_*` columns still refer to the **100k**
finals; the 200k gauntlet values live in the `_ext_` CSVs and below.

### PANEL-A at 200k (mean ± seed-std)

| arch | called | jd | both | vs its own 100k | vs onehot-ff@100k (−0.387) |
|---|---|---|---|---|---|
| `full` | −0.249 ± 0.047 | −0.216 ± 0.039 | **−0.233** | +0.180 | **+0.154** |
| `no-aux` | −0.270 ± 0.091 | −0.284 ± 0.121 | **−0.277** | +0.143 | **+0.110** |

Both gaps over onehot-ff's (flat) 100k endpoint exceed the 0.07 MDE and the
seed spread. Trajectory (edge vs selfplay-100k, seed means): full
−0.13 → −0.03 → **+0.19** at 100k/150k/200k; no-aux −0.15 → −0.08 →
**+0.08** — both crossed *positive* (now beating their 100k-lineage
ancestor head-to-head) and are **still climbing at 200k** (180k→200k slope
+0.13 / +0.19). The scripted-probe endpoints also crossed zero for full
(+0.10/−0.24/+0.24 per seed).

Honest caveat: onehot-ff was not extended (pre-registered rule extends only
still-climbing arms; its 80k→100k slope was +0.098 < 1 SE), so 200k-vs-100k
is budget-unequal. Its flatness bounds the expected shortfall, but a 200k
onehot control (~3.3 h) is cheap if wanted.

Canary: `full` s42's called-mode trump-lead incidence jumped to **10.6%**
at 200k (other extended seeds 0–1.8%) — the documented defender
trump-lead mechanism appearing in a self-play arm; worth tracking into
phase 2.

## Conclusions

1. ~~The token/transformer stack is about ceiling, not start speed~~
   **AMENDED 2026-07-05 after the budget-equal control:** at the equal
   200k budget, `onehot-ff` (−0.247) is statistically indistinguishable
   from `full` (−0.233) — the original version of this conclusion compared
   200k arms against onehot's 100k endpoint and was a budget artifact.
   What stands: onehot reaches strength ~5× cheaper in wall-clock at every
   matched budget so far, and **the token stack's ceiling advantage is not
   demonstrated at ≤ 200k in the shaped self-play regime**. The ceiling
   question now rests entirely on the terminal-reward league regime
   (phase 2 tests full vs no-aux; an onehot league arm is the missing
   experiment — see the onehot-control section).
2. **Informed embedding init is the clearest single win** (+0.150 under
   the transformer, 2 SE) and it *synergizes* with attention (only +0.060
   without it). Keep it.
3. **Transformer reasoning: +0.124 at 100k and growing** (its arms were
   still climbing when the flat arms had stopped). Keep it.
4. **Aux heads: no measurable effect in the shaped self-play regime**
   (+0.007 at 100k; full−no-aux = +0.044 at 200k, within seed noise).
   Per the pre-registered caveat this is a lower bound — the terminal
   league regime (phase 2) is where their dense-signal role should show,
   if anywhere. Worth pairing with the oracle-critic comparison, which
   attacks the same credit-assignment problem from the value side.
5. **Practical**: `onehot-ff` is a legitimate fast prototyping baseline
   (17 eps/s); `no-transformer` (pooled memory) buys 2× wall-clock over
   full at a real strength cost.
6. **Next**: phase 2 — `full` and `no-aux` through
   `train_league_ppo.py --arch` (terminal reward, PFSP league) to settle
   the aux-head question and the true ceiling; optionally a 200k
   onehot-ff control run for the budget-equal comparison.

## Phase 2 + onehot control (launched 2026-07-05)

**Onehot 200k control** (budget-equal comparison; ~3 h): the three
`onehot-ff` arms resumed 100k→200k with the same command shape as the
full/no-aux extension, then probes + PANEL-A →
`runs/ablation_202607/panel_a_onehot200k_{called,jd}.csv`.
Runner: `runs/ablation_202607/onehot_control_200k.sh`.

*Update 2026-07-05 22:00: the first launch was paused at ~129k gen-1
episodes (operator prioritized the full-tokenread probe; attempt archived
at `runs/phase2_202607/aborted_gen1_20260705/`). Phase 2 restarts from
scratch as stage 1 of the chained watcher once the probe finishes —
everything below is unchanged except the dates.*

**Phase 2** (terminal-reward league regime; ~6 days): `full` vs `no-aux`,
one arm each, both seeded from their s42 200k self-play finals, run
concurrently with 4 workers each. Budget: **2 generations × 750k = 1.5M
episodes per arm** (the 5–10M "ideal" would be 3+ weeks; the repro-v1
control curve gives a PANEL-A reference at ~1M episodes). Recipe per arm —
generation 1 anchored, generation 2 released (the anchored-then-release
plan; identical treatment keeps the arms paired; the anchor is collapse
insurance for an anchor-free warm start that has failed twice before):

```bash
# gen 1 (bidding anchor to the arm's own resume checkpoint)
PYTHONPATH=. .venv/bin/python train_league_ppo.py \
    --arch <full|no-aux> --resume runs/ablate_<A>_s42/final_<A>.pt \
    --seed-checkpoints 'runs/reference_selfplay_ppo/checkpoints/*.pt' \
    --league-dir runs/phase2_<A>/league --run-name phase2_<A> \
    --generations 1 --main-episodes 750000 --anchor-coeff 1.0 \
    --num-workers 4 --seed 42 --schedule-horizon 20000000
# gen 2 (anchor-free, resumed at the generation boundary)
PYTHONPATH=. .venv/bin/python train_league_ppo.py \
    --arch <A> --resume runs/phase2_<A>/checkpoints/pfsp_<A>_checkpoint_750000.pt \
    --league-dir runs/phase2_<A>/league --run-name phase2_<A> \
    --generations 1 --main-episodes 750000 \
    --num-workers 4 --seed 42 --schedule-horizon 20000000
```

Exploiter phases fire at each generation boundary (default 50k episodes,
3000-deal gate) → an exploitability datapoint per arm per generation. Live
telemetry per arm: `runs/phase2_<A>/checkpoints/anchored_eval.csv` (300
paired deals vs `final_pfsp` every 100k eps), `greedy_health.csv`,
`exploitability.csv`. Endpoints: PANEL-A + trump-lead probe on the 1.5M
finals → `runs/phase2_202607/`. Runner: `runs/phase2_202607/phase2.sh`.

**What phase 2 decides:** (a) whether aux heads matter once the dense
shaped signal is gone (terminal reward) — the fair test the self-play arms
couldn't give; (b) whether the self-play ceiling ordering survives the
league regime; (c) per-arm exploitability.

**Arm 3: onehot-ff (queued 2026-07-05, runs after the full/no-aux arms
finish).** Added the day the 200k control landed (see "Results — onehot
200k control"): onehot-ff tied `full` at every matched self-play budget,
so the league/terminal regime is now the *decisive* onehot-vs-full test —
if the token/transformer stack has real value, this is where it must show.
Identical recipe, `--arch onehot-ff --resume
runs/ablate_onehot-ff_s42/final_onehot-ff.pt`, run solo on the whole
machine (still 4 workers for recipe identity). Runner:
`runs/phase2_202607/phase2_onehot.sh` (restartable; skips finished
generations), chained automatically by
`runs/size_sweep_202607/watch_and_launch.sh` between phase 2 and the
capacity sweep. It ends by re-running PANEL-A over **all three** phase-2
finals in one paired gauntlet →
`runs/phase2_202607/panel_a_all3_{called,jd}.csv` (supersedes the
2-candidate `panel_a_{called,jd}.csv` from phase2.sh; CRN makes them
consistent, the all3 files just put every arm in one table).
Pre-registered interpretation is in the script header: onehot within the
0.07 MDE of `phase2_full` ⇒ the token stack has no demonstrated value at
≤ 1.5M league episodes and the default architecture should be revisited on
cost grounds (~5× cheaper); `full` ahead by > 0.07 and > 2 SE ⇒ the
stack's edge lives in the terminal/league regime. Single seed per arm —
treat sub-0.07 orderings as noise.

## Phase 3: capacity sweep (queued, auto-launches after phase 2)

**Why:** the ladder validated the *components* of `full`, but every
dimension (d_token 64, 4 reasoning layers, and especially the 256-wide
trunk/memory/pools, which hold ~70% of the parameters) is untested
folklore. This sweep asks whether `full` is the right **size** — would a
wider/deeper net raise the ceiling, or a smaller one train faster at no
cost?

**Design:** six one-knob variants around `full` (registry
`_full_size_variant`; actor/critic widths follow the encoder's `d_model`,
parameterized in commit `4cd5505` with the default bit-identical):

| arch | knob | params (E+A+C) |
|---|---|---|
| `full-dmodel128` | trunk/memory/pool width /2 | 470,519 |
| `full-dtok32` | transformer width /2 | 783,031 |
| `full-layers2` | depth /2 | 936,663 |
| `full` (existing runs = center point) | — | 1,003,607 |
| `full-layers6` | depth ×1.5 | 1,070,551 |
| `full-dtok128` | transformer width ×2 | 1,739,671 |
| `full-dmodel512` | trunk/memory/pool width ×2 | 2,929,943 |

3 seeds {42, 1042, 2042} × **200k episodes** (not 100k — the 100k matrix
showed endpoint ranks can invert while arms are still climbing; 200k
matches the existing `full` extension runs, which serve as the CRN-paired
center point, same seeds ⇒ same deals). Shaped self-play, same protocol
and yardsticks as the main matrix. ~2 days on the free machine.

**Launch (automatic):** `runs/size_sweep_202607/watch_and_launch.sh` is
running detached; it waits for phase 2 to exit, then runs:

```bash
PYTHONPATH=. caffeinate -is .venv/bin/python analysis/run_ablation_matrix.py \
    --archs full-dtok32 full-dtok128 full-layers2 full-layers6 \
            full-dmodel128 full-dmodel512 \
    --episodes 200000 --out-dir runs/size_sweep_202607 --prefix ablate
```

and finishes by writing `runs/size_sweep_202607/report.md` via
`analysis/ablation_report.py` (which merges the existing full@200k rows
from `runs/ablation_202607/results_200k_panel.csv` — that file carries the
*corrected* 200k PANEL-A values for the extended arms; the main
`results.csv`'s panel columns for those rows are stale 100k values).

**How to read the report** (`report.md` — no further analysis needed):
- *Delta vs `full` table*: a variant is better/worse only if its delta
  exceeds ~2 seed-level SE **and** the 0.07 MDE. Expected outcomes:
  smaller variants win eps/s — the question is whether they lose PANEL-A;
  bigger variants must beat `full` by > 0.07 to justify their cost.
- *Slope table*: any still-climbing variant's endpoint understates it
  (the onehot-ff lesson). If `full-dmodel512` ranks ≈ `full` but is still
  climbing steeply while `full` flattens, size likely pays at longer
  horizons — the right follow-up is extending the climbing arm, not
  concluding "same".

### What this sweep does NOT probe: the pooling readout (open question)

Raised 2026-07-05 (operator): could the attention-pooling stage between
the token stack and the shared trunk be *throwing away* the rich
embeddings' advantage — explaining why onehot-ff ties full? The
architecture facts (encoder.py):

- After transformer reasoning over all 19 tokens (context, memory, 8 hand,
  5 trick, 2 blind, 2 bury; d_token 64), each bag is compressed by an
  `AttentionPool` with **4 learned queries × 4 heads (hardcoded,
  encoder.py:92-93)**: hand → 64 dims, trick → 64, blind → 32, bury → 32;
  concat with the 64-dim context token → 256 trunk features. Everything
  the pick/partner/call heads and the critic see passes through this.
- The recurrent memory is even tighter: `memory_gru` input is the **64-dim
  context token alone** (encoder.py:615) — all cross-trick history must
  squeeze through one token.
- The only unmediated token access is the pointer head (play/bury/under
  scores see post-reasoning hand tokens directly, ppo.py:129), but its
  situation-conditioning `Wg(feat)` still comes from the pooled trunk.

**Sweep coverage:** `full-dmodel128/512` scale the pool *output* widths
proportionally (`d_model//4`, `//8`) — the closest existing probe, but
conflated with trunk/GRU width by design (one-knob). `full-dtok32/128`
scale the pool inputs/queries. The query count (4) and the *structure*
(pooling vs direct token readout by the heads) are NOT probed.
Diagnostic reading: if `full-dmodel512` beats `full` (> 0.07), capacity at
the pooled bottleneck was binding and a readout redesign is the highest-
value next architecture experiment; if it doesn't, the bottleneck-width
story loses support (though the structural question stays open).

*Caveat added 2026-07-06 (operator):* every knob in this sweep is
confounded by the pool squeeze — in particular a **depth** null on `full`
(`full-layers6` ≈ `full`) does not mean depth is useless, only that extra
reasoning capacity doesn't survive the 4-query pools. If the perceiver
probe wins and the base changes, rerun the sweep on the perceiver
variants instead (registered; see the perceiver-probe playbook entry for
the exact stage-3 swap).

**Candidate variants** (registry makes each a contained `architectures.py`
entry): (1) `full-tokenread` — cross-attention readout: the actor gets
learned queries attending over all 19 post-reasoning tokens, fused with
the pooled features (Perceiver-style; the direct fix) — **implemented
2026-07-05 (commit a019b9f) and running as its own probe, see the
"full-tokenread probe" section below**. (2) wider pools — parameterize
`AttentionPool.n_queries` (4→8) — cheapest capacity fix inside the same
structure. (3) richer memory — feed the GRU the fused 256-dim features
instead of the 64-dim context token (the `PooledMemoryEncoder` seam
`_fuse_and_update_memory` already exists for exactly this kind of
rerouting). (2) and (3) are not implemented.

## full-tokenread probe (launched 2026-07-05 21:32 — runs FIRST)

**Priority call by the operator:** run this before everything else queued,
pausing phase 2 (its gen-1 arms were ~129k/750k episodes in; archived to
`runs/phase2_202607/aborted_gen1_20260705/` and restarted from scratch
afterward so the anchor-to-resume-checkpoint semantics stay exactly as
pre-registered).

**Design** (`full-tokenread`, commit a019b9f): `full` plus a
cross-attention readout in the actor — 4 learned queries × 4 heads
(mirroring `AttentionPool`'s internals) attend over all **19**
post-reasoning tokens; the flattened result is projected to 256 and fused
(`Linear(512→256)+SiLU`) with the adapted trunk features before every
head. Everything else — encoder weights, critic, aux heads, and the
**memory recurrence** (context token → `GRUCell` → 256-d state →
`memory_in_proj` → next step's 64-d memory token) — is identical to
`full`; the encoder variant adds zero parameters and its standard outputs
are byte-identical (test-pinned). Note the readout also attends to the
post-reasoning *memory token*, which the base architecture computes and
then discards. Params: 1,217,623 vs full 1,003,607 (+214k, actor only).

**What it tests:** whether the per-bag attention-pool squeeze (hand→64,
trick→64, blind→32, bury→32 dims) between the token stack and the heads is
what's holding `full` at onehot-ff's level — readout *structure* at fixed
trunk width. The sweep's `full-dmodel512` tests *width*; together they
triangulate the bottleneck question.

**Protocol:** 3 seeds {42, 1042, 2042} × 200k episodes, shaped self-play,
identical to the main matrix — CRN-paired with the existing full@200k runs
(same seeds ⇒ same deals). Runner: stage 0 of
`runs/size_sweep_202607/watch_and_launch.sh`; auto-report at
`runs/tokenread_202607/report.md` (delta-vs-`full` table merges the
existing full@200k rows).

**Pre-registered interpretation:** delta vs `full` (PANEL-A both-modes,
3 seeds) > +0.07 MDE and > 2 seed-level SE ⇒ the pooling readout is a real
bottleneck — prefer `full-tokenread` for future runs and consider variants
(2)/(3). Within ±0.07 ⇒ readout structure is not the binding constraint at
this budget/regime; the onehot tie is then more likely a regime property
(shaped ≤200k self-play doesn't reward fine-grained state) — the phase-2
league arms become the operative test. Worse by > 0.07 ⇒ the extra actor
capacity hurts at this budget (optimization, not representation). Same
caveats as the whole matrix: shaped rewards, 200k horizon, watch the slope
table before calling a plateau (the onehot lesson).

**Epistemics amendment (2026-07-06, before results):** this probe is
**one-sided** for the Perceiver-IO redesign question. A *win* is strong
evidence the pools discard policy-relevant information. A *tie/loss* is
ambiguous between four readings: pools fine / dual-path optimization
pathology / critic still trunk-bound (GAE advantages computed from pooled
features can't resolve distinctions the actor could represent) / regime
doesn't reward it. **In-flight evidence for the pathology reading:**
s1042's pick/partner/bury entropies hit exactly 0.000 (other runs of both
archs: 0.004-0.006) with its scripted edge flat at −0.45..−0.49 from
50k→150k, while `full` on the same seed and deals recovered −0.75 → −0.06
— premature deterministic lock-in of the bidding heads. One seed proves
nothing, but treat a tokenread null as NOT falsifying the redesign — the
`perceiver` probe below is the fair test. (If the additive design is ever
iterated: zero-init the readout projection so the new path starts inert.)

## perceiver probe (launched 2026-07-06 — the fair Perceiver-IO test)

**Why (operator + assistant, Jul-6):** the tokenread probe can't test the
redesign's expressive potential (see amendment above), and after Jul-7
nobody can build the clean architecture. So it was built and queued as
stage 0.5, before the phase-2 restart.

**Design** (`perceiver`, commit 365a7de): token-centric end to end.
Shared: embeddings + token MLPs + 4-layer transformer + recurrence. Gone:
all four per-bag attention pools and the fused 256-d feature trunk.
- **Actor** (`PerceiverActorNetwork`): 4 learned queries × 4 heads attend
  over the 19 post-reasoning tokens → project to 256 → the standard
  actor_adapter → the existing pick/partner/two-tower/pointer heads. It
  ignores trunk features entirely (test-pinned).
- **Critic** (`PerceiverCriticNetwork`): its own independent 4-query
  readout → the same deep value-trunk shape as before → value. No aux
  heads (aux was ~null under shaping; compare against `no-aux` as well
  as `full`).
- **Memory** (operator's design): the GRU input is the post-reasoning
  **memory token** — the transformer's own "what to remember" slot, which
  the old architecture computed and discarded (its GRU read the *context*
  token). Same GRUCell(64, 256), zero parameter change; the state
  re-enters next step via `memory_in_proj` as always.
- **Params: 873,678** — *smaller* than full (1,003,607): the two readouts
  (+~170k) cost less than the pools + trunk they replace (−~300k).

**Protocol:** identical to the tokenread probe — 3 seeds {42, 1042, 2042}
× 200k shaped self-play, CRN-paired with full@200k; auto-report at
`runs/perceiver_202607/report.md`.

**Pre-registered interpretation** (PANEL-A both-modes vs `full`, also read
the `no-aux` row since perceiver has no aux heads):
- **> +0.07 and > 2 SE ⇒ the redesign wins** — adopt `perceiver` as the
  base for subsequent experiments (operator's stated intent) and rebase
  phase 2 per the contingency below.
- **Within ±0.07 ⇒ capability advantage NOT demonstrated** at this
  budget/regime (remember: onehot also ties here — a tie is weak
  evidence). Adopting anyway is a simplicity/cost call (~13% fewer
  params), not an evidence call; the operator set "justified by evidence
  that it is more capable" as the bar, so a tie means DON'T adopt yet —
  the league regime or longer horizons would have to justify it.
- **< −0.07 ⇒ the redesign underperforms** at this budget; keep `full`,
  and note whether the failure is optimization (check per-seed entropy
  collapse à la tokenread s1042) or representation (all seeds uniformly
  worse, healthy entropies).
- Stability check either way: per-seed pick-head entropy in each
  `runs/ablate_perceiver_s*/train.log` (grep "Entropy - pick") — exactly
  0.000 early = the lock-in failure mode; flag that seed.

## Results — onehot 200k control

Completed 2026-07-05 18:36. PANEL-A both-modes per seed: s42 −0.251,
s1042 −0.307, s2042 −0.181 → **mean −0.247 ± 0.063**.

**This overturns the extension section's conclusion.** At the equal 200k
budget: full −0.233, onehot-ff −0.247, no-aux −0.277 — all within ~1
seed-level SE of each other. The "flat plateau" inference from onehot's
80k→100k slope (+0.098, < 1 SE) was wrong: its curve sat flat 100k→150k
(edge vs scripted −0.23 → −0.23) and then moved sharply 150k→200k
(→ −0.09, +0.14 PANEL-A overall). Slope-based plateau calls on 20k windows
are unreliable — a second lesson of the same kind as the 100k rank
inversion, one level up.

Pre-registered verdict (playbook rule): **the token-stack ceiling
advantage is NOT demonstrated at ≤ 200k in the shaped self-play regime.**
The earlier "full-family blows past onehot by 200k" compared unequal
budgets and is retracted (the Conclusions section below is amended
accordingly). All three arms are still climbing at 200k; the regimes where
the token stack should matter most (terminal reward, long-horizon league)
remain untested for onehot. **Follow-up QUEUED same day:** an onehot-ff
league arm mirroring phase 2 (arm 3 in the Phase 2 section above) runs
automatically after the full/no-aux arms finish and before the capacity
sweep — it settles onehot-vs-full where it counts.

## Results — full-tokenread probe

**Landed 2026-07-06 (matrix 17.2h, 3/3 seeds OK). Verdict: NULL by the
pre-registered rule — the readout did not beat `full` by > +0.07 and
> 2 SE, so the pooling bottleneck is NOT demonstrated in this regime.**
Per the one-sided-probe logic this is ambiguous, not exculpatory; the
`perceiver` probe (running next) is the fair test of the token-centric
design.

From `runs/tokenread_202607/report.md`:

| arch | PANEL-A called | PANEL-A jd | both | eps/s |
|---|---|---|---|---|
| full | −0.249 ± 0.047 | −0.216 ± 0.039 | −0.233 | 7.6 |
| full-tokenread | −0.194 ± 0.048 | −0.185 ± 0.142 | **−0.189** | 3.9 |

Delta vs full = **+0.043, seed-level SE 0.059** — inside the ±0.07 null
band, not > 2 SE. Tokenread cost ~2× the wall-clock per episode for a
within-noise gain: not adoptable on this evidence (and it was always a
probe, not a candidate — the clean end-state is `perceiver`).

Per-seed diagnostics (both-modes mean): s42 −0.159, s1042 −0.294,
s2042 −0.115.

- **s1042 = a full PASS-collapse (recovered late), not mere noise**: by
  episode ~1.4k pick/partner/bury entropies hit exactly 0.000 (PPO KL
  0.0000 — heads frozen) and the self-play leaster rate pinned at 100%
  (every seat passing every deal); it stayed in the all-leaster
  equilibrium for ~88% of the run (anchored edges flat ~−0.75 from
  25k–175k), then spontaneously escaped at ~180k (leaster 100%→1.9%),
  relearned bidding, and jumped (selfplay100k edge −0.71→−0.20 in the
  final window). Same attractor as the ExIt warm-start collapse. Its
  endpoint (−0.294, worst seed; drives the jd seed-std 0.142) is really
  "~20k episodes of non-leaster training." `full` on the identical
  seed/deals did NOT collapse — evidence the dual-path readout
  destabilizes early bidding gradients. Headline verdict still uses all
  three seeds as pre-registered; count this arm as 1 collapse per the
  protocol's collapse-reporting rule.
- **s2042 trump-lead canary hot**: 20.9% jd / 17.6% called at 200k (full
  s42 canary was 10.6%) — the leak got *worse* under the readout for that
  seed.
- **Still-climbing flag**: last-20k slope vs selfplay-100k = +0.201 ±
  0.120 (> 1 SE) — the 200k endpoint may understate tokenread. Noted, but
  no extension run: the perceiver probe answers the underlying question
  more cleanly than +100k of a deliberately-clunky probe arm would.

### Post-hoc contrast: tokenread vs onehot-ff at 200k (suggestive only)

Seed-paired PANEL-A both-modes means (onehot from
`panel_a_onehot200k_{called,jd}.csv`): s42 +0.093, s1042 +0.014,
s2042 +0.066 in tokenread's favor — **+0.057 ± 0.023 seed-SE, same sign
in all three seeds**: the first consistent separation any token
architecture has shown over the dense FF net (plain `full` vs onehot
remains a wash, −0.233 vs −0.247). NOT pre-registered, under the 0.07
MDE, n=3 — treat as suggestive. Its specific shape (tokens pay only when
a head reads them directly, not through pools) is exactly the perceiver
thesis; the perceiver probe and the phase-2 onehot league arm are the
decisive tests. Cost caveat: onehot trains several× faster than full,
tokenread ~2× slower — evidence-per-compute still favors onehot.

## The always-PASS collapse is UNIVERSAL in from-scratch self-play
## (leaster-rate scan, 2026-07-06)

Prompted by the operator observing the crash across arms (and live in
perceiver s1042), `analysis/leaster_scan.py` (committed) extracts the
rollout leaster-rate trajectory from every run log. Rerun any time with
`PYTHONPATH=. uv run python analysis/leaster_scan.py [runs/glob ...]`.

**Finding: every one of the 24 from-scratch shaped-reward self-play runs
collapsed to ~100% leaster (all seats always PASS) within the first
~3-4k episodes — entry is universal; architectures differ ONLY in escape
time.** Hard-collapse spans (leaster ≥ 90%):

| arch | s42 | s1042 | s2042 | worst % of budget |
|---|---|---|---|---|
| onehot-ff | 3k-5k | 3k-3k | 3k-3k | ~1% |
| full | 3k-7k | 3k-10k | 3k-10k | ~4% |
| full-uninformed | 3k-10k | 3k-11k | 4k-8k | ~9% |
| no-aux | 4k-12k | 3k-80k | 4k-20k | **39%** |
| no-transformer | 3k-78k | 3k-18k | 3k-67k | **76%** |
| no-transformer-uninformed | 3k-30k | 3k-61k | 2k-31k | **59%** |
| full-tokenread | 3k-32k | 3k-178k | 3k-13k | **88%** |
| perceiver (@~59k, running) | 3k-16k | 4k-**51k** (escaped) | 3k-23k | TBD |

Mechanism (consistent with every trace): at init play skill is terrible,
so picking has strongly negative EV → PASS dominates for every seat →
all-leaster equilibrium with bidding entropy frozen at ~0. Escape appears
driven by rare exploratory PICKs finally paying off once play skill
learned *inside leasters* transfers to trick play generally — which
explains why fast-learning-per-episode archs (onehot) escape almost
immediately and slow/unstable ones (tokenread, no-transformer) linger.

**Reinterpretation duties:**

1. **100k adjacent-rung deltas are partly escape-speed, not steady-state
   capability.** no-transformer spent 18-78% of its 100k budget in the
   collapse; the "+0.124 transformer" delta substantially measures
   "transformer escapes faster." Same for informed-init (+0.150).
   Escape speed is a real, valuable property — but it is a different
   claim, and a cheap stabilizer could erase those deltas.
2. **no-aux s1042 lost 80k of 200k** — the no-aux endpoint (−0.277) is
   partly collapse time, not aux-head value.
3. **perceiver s1042 lost its first ~51k but escaped with 145k to go**
   (vs tokenread s1042's escape at 178k with only 20k left). When the
   perceiver report lands, run the scanner and read the endpoint with
   the per-seed spans next to it.
4. onehot's 100-150k flat-then-jump was NOT leaster (it escaped at 5k) —
   separate phenomenon.

**Fix chosen and IMPLEMENTED (operator decision 2026-07-06): option (a),
the selective entropy kick** — `--leaster-watchdog` on
train_selfplay_ppo.py, **default OFF** (no completed or running run used
it; enable explicitly on future from-scratch runs). `LeasterWatchdog`:
when the rolling 3000-episode leaster rate crosses 90% it multiplies the
pick head's *scheduled* entropy coefficient ×10 each update until the
rate falls below 30% (hysteresis), then normal annealing resumes.
Rationale for firing at the crossing: the entropy bonus's gradient
vanishes as the head approaches determinism, so the kick must land while
pick entropy is still alive (~episode 3k), holding a probability floor so
the deep freeze never forms. Rewards untouched. Survey result: NO other
adaptive mechanism exists in either trainer today — both use open-loop
per-head entropy schedules from config.py only (the old trainer's
controllers/epsilon-floors were retired); ppo.py's per-head
`entropy_coeff_*` attributes made this a trainer-side-only change.
Engage/release events print to train.log (grep "watchdog"). If used in a
comparison, enable it for ALL arms. Alternatives considered:
(b) constant per-head entropy floor on pick/partner/bury;
(c) bidding-warmup curriculum (first N eps sample pick/pass from an
ε-mixture, bidding heads frozen, so play heads see picker/defender data
before bidding optimizes); (d) early scripted-opponent mixing (breaks the
equilibrium but leaves the pure-self-play regime); (e) oracle critic
(already built) should shorten escapes via lower-variance PICK
evaluation but won't prevent entry. Any fix should be flag-gated and
applied to ALL arms of a comparison equally; existing completed runs stay
comparable (all collapsed, same regime). The league/warm-start production
regime never enters this attractor — it is a from-scratch probe artifact.

## Results — perceiver probe

**Landed 2026-07-07 07:26 (matrix 15.4 h, 3/3 seeds OK). Verdict by the
pre-registered rule: the redesign UNDERPERFORMS (`< −0.07` branch) —
keep `full` as the base.**

From `runs/perceiver_202607/report.md`:

| arch | PANEL-A called | PANEL-A jd | both | eps/s | train h |
|---|---|---|---|---|---|
| full | −0.249 ± 0.047 | −0.216 ± 0.039 | −0.233 | 7.6 | 7.3 |
| no-aux | −0.270 ± 0.091 | −0.284 ± 0.121 | −0.277 | 7.6 | 7.3 |
| perceiver | −0.402 ± 0.122 | −0.438 ± 0.237 | **−0.420** | 4.4 | 12.8 |

Delta vs full = **−0.187, seed-level SE 0.106** (1.8 SE; point estimate
2.7× the 0.07 MDE, in the wrong direction). CRN-paired per-seed deltas
(both-modes): s42 **−0.181**, s1042 −0.003, s2042 **−0.377** — 0/3 seed
wins. It also loses to the aux-free comparator `no-aux` (−0.143). And it
is ~1.7× slower per episode than full (4.4 vs 7.6 eps/s) — the per-head
token readouts cost more than the pools they replaced, echoing
tokenread's 2× slowdown.

**Failure-mode classification (pre-registered check): NOT entropy
lock-in.** Collapse spans (leaster_scan): s42 3k–16k, s2042 3k–23k,
s1042 4k–51k — all three escaped with 150–180k episodes to recover, and
endpoint pick entropies are alive (0.003–0.009, same magnitude as
healthy full runs; no 0.000 freeze). Escape latency does not explain the
ranking either: s1042 collapsed the LONGEST yet finished best (paired
delta ≈ 0), while s2042 escaped at 23k and finished worst (−0.62 both).
High seed variance (jd seed-std 0.237, ~2× full's worst) with uniformly
negative paired deltas points at **optimization instability /
representation under this budget**, not the always-PASS trap. Last-20k
slope −0.067 ± 0.131 — not climbing, so no extension run.

**Caveats for the record** (why this is "not demonstrated at this
budget/regime", not "perceiver refuted"): watchdog-OFF regime, 200k
shaped self-play, n=3 seeds, and the readout shape (4 queries × 4 heads)
was never tuned — the registered perceiver-readq/readheads/rheads
variants would test whether the readout is the bottleneck. But by the
operator's own bar (adoption requires capability evidence), the tie/loss
branch applies: base stays `full`, phase 2 runs as designed (full vs
no-aux, launched 2026-07-07 07:27), capacity sweep stays full-based.

**Chain edit executed 2026-07-07** (the one-time watcher edit, tie/loss
branch): stage 3 = `--archs full full-dtok32 full-dtok128 full-layers2
full-layers6 full-dmodel128 full-dmodel512 --episodes 200000
--leaster-watchdog --prefix sweep`; stage 4 report `--baseline full`
with NO `--extra-results`. Two non-obvious details: (1) the sweep uses a
**fresh prefix `sweep`** — with `--prefix ablate` the orchestrator's
skip-guard would have found the completed watchdog-OFF
`runs/ablate_full_s*` probe runs and silently skipped retraining the
watchdog-on baseline; (2) stage 1 gained a **pgrep wait-guard** for
`phase2_202607/phase2.sh` — phase2.sh has no internal resume guards, and
the relaunched watcher would otherwise double-launch it on top of the
running arms.

## Results — phase 2

*(pending)*

## Results — capacity sweep

*(pending — `runs/size_sweep_202607/report.md` is generated automatically;
paste it here with a short verdict per the "How to read" rules above)*

---

# HANDOFF PLAYBOOK (written 2026-07-05)

Context: the operator loses access to the LLM that ran this experiment
after 2026-07-07. Everything below is written so a person (or a smaller
model) can finish the analysis mechanically. **Read the Design + Decision
criteria sections above first; every number needed is produced by
committed tools.**

## What is running / queued right now

| item | what | ETA | completion signal |
|---|---|---|---|
| onehot 200k control | 3 self-play resumes + probes + PANEL-A | DONE 2026-07-05 18:36 | `ONEHOT CONTROL COMPLETE` in `runs/ablation_202607/onehot_control.log` |
| stage 0: tokenread probe | `full-tokenread` 3×200k self-play + PANEL-A + auto-report | ~2026-07-06 late afternoon (measured ~3.7-4.3 eps/s train — the token readout makes the PPO update ~1.9× costlier than full's 7.4-8.0 eps/s) | `STAGE 0 (TOKENREAD) COMPLETE` in `runs/size_sweep_202607/watcher.log`; report at `runs/tokenread_202607/report.md` |
| stage 0.5: perceiver probe | `perceiver` 3×200k self-play + PANEL-A + auto-report | ~14-17 h after stage 0 (~2026-07-07 morning/noon) | `STAGE 0.5 (PERCEIVER) COMPLETE` in `runs/size_sweep_202607/watcher.log`; report at `runs/perceiver_202607/report.md` |
| stage 1: phase 2 arms 1-2 | `full` vs `no-aux` league runs, 2×750k eps each (restarted fresh after the pause) | ~6 days after stage 0.5 (~2026-07-13) | `PHASE2 COMPLETE` in `runs/phase2_202607/phase2.log` |
| stage 2: phase 2 arm 3 | `onehot-ff` league run, 2×750k eps, solo on machine | ~1.5-3 days after stage 1 (~2026-07-15; check `runs/phase2_onehot-ff/checkpoints/` for progress) | `PHASE2 ONEHOT ARM COMPLETE` in `runs/phase2_202607/phase2_onehot.log` |
| stages 3-4: capacity sweep | 6 size variants × 3 seeds × 200k + auto-report | ~2-3 days after stage 2 (~2026-07-18) | `SIZE SWEEP COMPLETE` in `runs/size_sweep_202607/watcher.log` |

The whole chain is driven by one detached watcher
(`runs/size_sweep_202607/watch_and_launch.sh`, rewritten + relaunched
2026-07-06 ~10:15): tokenread probe → perceiver probe → phase2.sh →
phase2_onehot.sh → capacity sweep → report. Progress: `tail
runs/size_sweep_202607/watcher.log`. The first phase-2 attempt (paused
2026-07-05 at ~129k eps for the tokenread probe) is archived at
`runs/phase2_202607/aborted_gen1_20260705/` and feeds nothing.

**If the machine reboots or something dies**, everything is resumable:
- the whole chain: `nohup zsh runs/size_sweep_202607/watch_and_launch.sh
  > /dev/null 2>&1 & disown` — completed stages are skipped (stages
  0/0.5/3 per job via the orchestrator; stage 1 via `PHASE2 COMPLETE`;
  stage 2 per generation). Only caveat: a league generation that died *mid-run*
  restarts from its last `--resume` point, i.e. the generation start; to
  salvage partial progress instead, edit the arm script's `--resume` to
  the newest `runs/phase2_<arch>/checkpoints/pfsp_<arch>_checkpoint_*.pt`
  and reduce `--main-episodes` accordingly.
- phase 2 arms 1-2 alone: `zsh runs/phase2_202607/phase2.sh` (same caveat).
- arm 3 alone: `zsh runs/phase2_202607/phase2_onehot.sh`.
- onehot control: `zsh runs/ablation_202607/onehot_control_200k.sh`.

## When the onehot control lands — RESOLVED 2026-07-05

Onehot@200k = −0.247 ± 0.063 ⇒ within 0.07 of full (−0.233): the plateau
reading was wrong; conclusion #1 amended (see the onehot-control results
section). The recommended next experiment is an **onehot-ff league arm**
mirroring phase 2 (exact commands in that section) — it settles
onehot-vs-full in the regime that matters.

## When the tokenread probe lands (~2026-07-07)

1. Open `runs/tokenread_202607/report.md` (auto-generated; merges the
   existing full@200k rows as the baseline).
2. Read the `full-tokenread` row of the delta-vs-`full` table and apply
   the pre-registered interpretation in the "full-tokenread probe"
   section verbatim (> +0.07 and > 2 SE ⇒ pooling bottleneck real; ±0.07
   ⇒ structure not binding here, league arms decide; < −0.07 ⇒ extra
   capacity hurts at this budget).
3. Check the slope table: if full-tokenread is still climbing > 1 SE
   while `full` was flat at 200k, the endpoint understates it (onehot
   lesson) — say so next to the verdict.
4. Paste report + verdict into "Results — full-tokenread probe".

**CONTINGENCY if the verdict is "bottleneck real" (tokenread > full by
> 0.07 and > 2 SE)** — operator decision (2026-07-05): subsequent
experiments should move to the winning architecture. Concretely, in cost
order:
- *Phase 2 will be only hours into gen 1* when the report lands (stage 1
  starts right after stage 0). Pausing it again is cheap: kill the
  phase2.sh shells + the two train_league_ppo processes, archive
  `runs/phase2_full` / `runs/phase2_no-aux` next to
  `aborted_gen1_20260705/`, and re-plan arms before relaunching. The
  natural redesign keeps the aux question paired on the new base:
  `full-tokenread` vs a `tokenread-no-aux` variant (one registry entry:
  `build_critic=_no_aux_critic, has_aux_heads=False` on the tokenread
  encoder/actor) — plus the onehot arm unchanged as the cost baseline.
  Warm-start ckpt for a tokenread league arm:
  `runs/ablate_full-tokenread_s42/final_full-tokenread.pt`.
- The capacity sweep's variants are `full`-based; if tokenread wins they
  answer "is `full` the right size" but not "is `full-tokenread` the
  right size" — still useful (width vs structure triangulation), keep or
  drop by taste/time.
- Next architecture step already sketched by the operator: feed the
  post-reasoning MEMORY token into the GRU (today the GRU input is the
  post-reasoning *context* token and the post-reasoning memory token is
  discarded). Zero new params (same 64-d input width): a
  TokenReadEncoder subclass overriding `_fuse_and_update_memory` to use
  `all_tokens[:, 1, :]` (or concat with context → GRUCell(128, 256),
  +small params) as the GRU input. Not implemented; one-knob it against
  full-tokenread.
- **Operator intent (2026-07-05), longer term:** `full-tokenread` is a
  *probe*, not the end-state design. Its dual output (pooled bags +
  context → 256-d trunk, PLUS raw tokens for head-side MHA) is
  deliberately clunky — additive so the win is attributable. If the
  probe proves the readout valuable, the operator wants a clean
  re-architecture of the encoder rather than shipping the dual-path
  structure: token-centric end to end (Perceiver-IO shape — encoder
  emits tokens + recurrent state; actor AND critic each own a readout;
  the pooled trunk disappears). Build it as a NEW registry arch and
  ladder it against full-tokenread; it will not be
  checkpoint-compatible with anything (fresh training only), and the
  memory write path then needs its own summary (learned query or the
  memory-token feed above) since fused trunk features no longer exist.

## When the perceiver probe lands (~2026-07-07)

**EXECUTED 2026-07-07 — verdict: perceiver UNDERPERFORMS (−0.187 vs
full); the tie/loss branch below was applied, including the one-time
watcher edit (see "Results — perceiver probe"). Nothing in this section
remains to be done.**

Same mechanical recipe as the tokenread probe, using
`runs/perceiver_202607/report.md` and the pre-registered interpretation
in the "perceiver probe" section (including the per-seed
`grep "Entropy - pick" runs/ablate_perceiver_s*/train.log` stability
check). **If perceiver wins** (> +0.07 and > 2 SE vs full): the
contingency below activates with `perceiver` (not tokenread) as the new
base — pause phase 2 early and rebase its arms. **`perceiver-aux` exists
for exactly this** (registered 2026-07-06 after the operator flagged that
dropping the aux heads was not intended): the critic's token readout
feeds both the value trunk and the full inherited aux stack
(985,751 params vs perceiver's 873,678 and full's 1,003,607); the running
probe itself is aux-FREE (restarting it to add aux would have pushed the
report past the operator's access window, and redefining `perceiver`
mid-flight would break its own checkpoint loads). The natural rebased
arm pair is `perceiver-aux` vs `perceiver` — it mirrors full-vs-no-aux
and answers the aux question on the new base under league terminal
reward. The operator decides the exact arm set. **Also rebase the capacity
sweep** (operator, 2026-07-06): a depth/width sweep on `full` is
confounded by the attention-pool squeeze — extra transformer capacity
still exits through 4-query pools, so a depth null on `full` would not
mean depth is useless on `perceiver`. Perceiver-based one-knob variants
are already registered (commit with `_perceiver_size_variant`; params
412k-2.48M). The swap is one edit in
`runs/size_sweep_202607/watch_and_launch.sh` stage 3:

```
--archs perceiver perceiver-dtok32 perceiver-dtok128 perceiver-layers2 \
        perceiver-layers6 perceiver-dmodel128 perceiver-dmodel512 \
        perceiver-readq2 perceiver-readq8 \
        perceiver-readheads2 perceiver-readheads8 \
        perceiver-rheads2 perceiver-rheads8 \
--leaster-watchdog
```

**The sweep runs `--leaster-watchdog` (operator decision, 2026-07-06):**
escape latency from the always-PASS trap is seed lottery, and for
capacity questions it is pure confound — the sweep should measure
learning slope, not collapse luck. Because the watchdog changes the
regime, the base arch is INCLUDED in `--archs` (first entry above, +3
runs ≈ +17h) and the report baseline is that in-sweep, watchdog-on run:
`--baseline perceiver` with NO `--extra-results` as the baseline source
(old probe rows are watchdog-off; cite them as context only). The
watchdog-on base vs the watchdog-off probe is itself a free read on what
the watchdog is worth.
The second line onward are the **attention-shape knobs** (operator,
2026-07-06): readout queries, readout heads, and transformer reasoning
heads were all an unexamined "4" chosen when the transformer was first
added. The four head-count variants are exactly param-matched to base
perceiver (873,678 — MHA params don't depend on num_heads), so any delta
there is pure structure; readq2/readq8 are 807,886 / 1,005,262. NOTE:
12 variants ≈ doubles sweep wall-clock (~6 days at 3-seeds-parallel,
~17h per variant batch). If that's too long, run capacity first (line 1)
and attention second (lines 2-4) as two sweeps — the report can merge
them via repeated `--extra-results`.
**Oracle critic readout — DONE unconditionally (operator decision,
2026-07-06):** the operator chose to rebuild `oracle.py` perceiver-style
without waiting for the probe verdict (no oracle training had happened, no
checkpoint compatibility existed, and worst case it is equivalent —
elegance worth it). `OracleCriticEncoder` now deletes the five pools +
fusion MLP (incl. `pool_opp`, which squeezed 32 opponent-hand token slots
into 64 dims); `OracleValueNetwork` reads all 51 post-reasoning tokens
through its own MHA readout (4 queries × 4 heads, same modules as
`PerceiverCriticNetwork`); the memory GRU is fed the post-reasoning
MEMORY token. 622,473 params (was 772,745; −150k = the pools; the readout
replaces the fusion MLP at identical size). All 15 oracle tests pass
(incl. dual-GAE update, gradient isolation, checkpoints) + 4 new readout
invariants. Any future oracle run uses this design; there are NO
pooled-oracle checkpoints anywhere. Oracle health check for the first
run stands: watch oracle value loss / explained variance (`ev_oracle` in
update stats) — full information makes the target nearly deterministic
given the policies, so persistent underfit means the VALUE net (not the
regime) needs attention.
**If it ties**, the operator's own bar applies: adoption requires
capability evidence, so a tie ⇒ keep `full` as default, let phase 2 run
as designed, and keep the full-based sweep — but STILL apply the
watchdog decision: stage 3 becomes
`--archs full full-dtok32 ... full-dmodel512 --leaster-watchdog` with
`--baseline full` and no watchdog-off `--extra-results` baseline.
**Either branch requires one watcher edit** (stage 3 lacks the flag and
the in-sweep baseline arch as written): kill ONLY the watcher shell
(`pgrep -fl watch_and_launch`), never the training processes, edit
`runs/size_sweep_202607/watch_and_launch.sh` stage 3, relaunch
(`nohup zsh runs/size_sweep_202607/watch_and_launch.sh > /dev/null 2>&1 &
disown`) — the per-stage skip guards make this safe; do it while stage
0.5/1/2 is still running, well before stage 3 launches (~Jul-15).
Paste report + verdict into "Results — perceiver probe".

## When phase 2 lands

1. Numbers: `runs/phase2_202607/panel_a_{called,jd}.csv` (arms 1-2, lands
   ~Jul-11) and `panel_a_all3_{called,jd}.csv` (all three arms in one
   paired table, lands with arm 3 ~Jul-13 — **prefer this one once it
   exists**), `scripted_probe_*.json`, `trump_lead_*.json`, per-arm
   `runs/phase2_<arch>/checkpoints/
   {anchored_eval,greedy_health,exploitability}.csv`.
2. Questions, in order:
   a. **Did either arm collapse?** greedy_health.csv: greedy PICK → 0% or
      leaster → 100% means dead run — say so, compare only pre-collapse
      trends, and distrust the endpoint.
   b. **Aux heads under terminal reward:** PANEL-A(phase2_full) −
      PANEL-A(phase2_no-aux), both-modes mean. Single seed each, so use
      the per-measurement SE from the CSVs (~0.04 each ⇒ difference SE
      ~0.055; also compare the anchored_eval.csv *trends*, which are
      CRN-paired between arms at each episode). Gap > ~0.11 (2 SE) =
      aux heads matter under terminal reward. Gap within noise = the
      aux-head null generalizes; simplifying to no-aux is defensible.
   c. **League vs self-play regime:** both arms should sit well above
      their 200k self-play starting points (−0.233 / −0.277). Compare
      also to the repro-v1 control at ~1M episodes (PANEL-A baselines in
      `runs/rigorous_baseline_202607/`, headline league-13.65M CA −0.120 /
      JD +0.038): phase-2 arms at 1.5M with the roster/exploiter fixes
      should be at least competitive with the v1 curve at matched episodes.
   d. **Exploitability:** `exploitability.csv` per arm (gate edge per
      generation; lower/declining = better).
   e. **Onehot vs full where it counts (arm 3, ~Jul-13):**
      PANEL-A(phase2_full) − PANEL-A(phase2_onehot-ff) from
      `panel_a_all3_*.csv`, both-modes mean. Pre-registered rule (also in
      the `phase2_onehot.sh` header): gap within 0.07 ⇒ token stack has NO
      demonstrated value at ≤ 1.5M league episodes — recommend revisiting
      the default architecture on cost grounds (~5× cheaper training);
      full ahead by > 0.07 and > 2 SE (~0.11) ⇒ the stack's edge is
      real but regime-dependent. Also sanity-check arm 3's
      `greedy_health.csv` for collapse exactly as in (a) — a collapsed arm
      decides nothing.
3. Paste the numbers into "Results — phase 2" above with the verdicts.

## When the capacity sweep lands

Open `runs/size_sweep_202607/report.md` (auto-generated) and apply the
"How to read the report" rules in the Phase 3 section. Paste into
"Results — capacity sweep".

## Continuing league generations after the architecture settles — stopping rule (pre-registered 2026-07-06)

Once the winning architecture is chosen, further league generations are gated
per-generation instead of running an arbitrary episode budget. At the end of
each generation `g` run BOTH instruments on the gen-final checkpoint:

1. **Frozen longitudinal yardstick — PANEL-A** (membership and seed 42 never
   change; this is the number comparable across the whole project):

   ```bash
   PYTHONPATH=. uv run python analysis/rigorous_eval.py \
       --candidates runs/<league_run>/finals/gen<g>.pt \
       --anchors final_pfsp_swish_ppo.pt <pfsp 15M> <pfsp 5M> <selfplay 100k> \
       --deals 1000 --seed 42 --partner-mode called \
       --out-csv runs/<league_run>/panel_gen<g>_called.csv
   # repeat with --partner-mode jd
   ```

2. **Moving local anchor — gen g vs gen g−1 head-to-head** (cheap, one
   pairing; this signal does NOT saturate as the agent surpasses the panel):

   ```bash
   PYTHONPATH=. uv run python analysis/rigorous_eval.py \
       --candidates runs/<league_run>/finals/gen<g>.pt \
       --anchors    runs/<league_run>/finals/gen<g-1>.pt \
       --deals 1000 --seed 42 --partner-mode called \
       --out-csv runs/<league_run>/h2h_gen<g>_called.csv
   # repeat with --partner-mode jd; report the both-modes mean edge
   ```

**Stopping rule (fixed in advance — do not adjust after seeing results):**
stop after generation `g` when BOTH hold for `g` and `g−1` (two consecutive
flat generations):

- (a) PANEL-A both-modes mean has not exceeded its previous best by ≥ 0.07
  (the 1000-deal MDE), AND
- (b) the gen-vs-previous head-to-head edge is < +0.07.

Rationale for two consecutive: the ablation showed single-window plateau
calls are unreliable (onehot-ff sat flat 100k–150k, then jumped). Rationale
for instrument (b): panel edge vs fixed policies is bounded by the
best-response value against them and compresses as the agent approaches/
passes parity with `final_pfsp` — a flat panel slope past that point can be
signal saturation, not learning stagnation. The head-to-head anchor measures
the local improvement gradient directly and keeps sensitivity. Panel
statistical power (SE at 1000 CRN deals) does NOT degrade with candidate
strength; what degrades is construct validity — gains against strong
adaptive opponents (the live league) may be worth ~0 against weaker frozen
panel members.

**Oracle-critic regime (`--critic-mode oracle`):** this rule applies
VERBATIM. Every instrument above (PANEL-A, head-to-head, exploiter gate,
scripted probe) evaluates the ACTOR playing partial-observation games; the
oracle critic exists only inside `update()` (ppo.py `_fill_oracle_values`)
and never runs at play time, so nothing about eval variance, MDE, CRN
pairing, or thresholds changes. Two regime-specific additions only:
(1) league snapshots are oracle-stripped automatically and `ppo.load_agent`
handles `critic_mode` — no special checkpoint handling; (2) watch the
oracle value loss in training logs as a HEALTH diagnostic (an underfit
oracle silently degrades to a noisy baseline — that shows up in training
stats, not in the panel). If the goal is to ATTRIBUTE a gain to the oracle
critic (vs merely benefiting from it), that requires a paired arm
(oracle vs limited, same seed, CRN anchored evals) — same design as the
phase-2 arms; a single long oracle run + this stopping rule measures the
combined result but cannot isolate the cause.

**Confirmatory eval (guard against stopping-rule selection bias):** the
stopping decision peeks at seed-42 deals repeatedly, so the FINAL reported
number for the chosen checkpoint must come from one fresh-deal run:
identical panel membership, `--seed 20260706`, 1000 deals, both modes.
Label it "confirmation" — the seed-42 run remains the longitudinal series.
Keep exploiter-gate edge (`exploitability.csv`) as the robustness check
neither instrument covers.

## If something needs rerunning from scratch

All tooling is committed: `analysis/run_ablation_matrix.py` (orchestrator;
`--smoke` for a 5-minute pipeline check), `analysis/aggregate_ablation.py`
(CSVs/plots/table), `analysis/ablation_report.py` (statistics/verdict
tables), `analysis/rigorous_eval.py` + `analysis/scripted_probe.py` +
`analysis/trump_lead_probe.py` (instruments — frozen seeds, see
`Evaluation_Harnesses_202607.md`). Checkpoints record their architecture;
**always load via `ppo.load_agent(path)`**, never construct `PPOAgent`
with a guessed arch.
