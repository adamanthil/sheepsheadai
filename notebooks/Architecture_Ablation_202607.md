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
remain untested for onehot — phase 2 covers only full/no-aux. **Optional
follow-up for the operator:** an onehot-ff league arm mirroring phase 2
(same commands with `--arch onehot-ff --resume
runs/ablate_onehot-ff_s42/final_onehot-ff.pt`) would settle onehot-vs-full
where it counts.

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
| onehot 200k control | 3 self-play resumes + probes + PANEL-A | hours (2026-07-05 evening) | `ONEHOT CONTROL COMPLETE` in `runs/ablation_202607/onehot_control.log` |
| phase 2 | `full` vs `no-aux` league runs, 2×750k eps each | ~2026-07-11 | `PHASE2 COMPLETE` in `runs/phase2_202607/phase2.log` |
| capacity sweep | auto-launches when phase 2 exits (watcher) | ~2 days after phase 2 | `SIZE SWEEP COMPLETE` in `runs/size_sweep_202607/watcher.log` |

**If the machine reboots or something dies**, everything is resumable:
- phase 2: rerun the exact gen-1/gen-2 commands in the Phase 2 section
  (skip gen 1 if `runs/phase2_<arch>/checkpoints/pfsp_<arch>_checkpoint_750000.pt`
  exists) or simply `zsh runs/phase2_202607/phase2.sh` again — completed
  invocations resume from their last league checkpoint via `--resume`
  (use the newest `pfsp_<arch>_checkpoint_*.pt`; edit the script's
  `--resume` accordingly).
- capacity sweep: `zsh runs/size_sweep_202607/watch_and_launch.sh`
  (skips finished jobs).
- onehot control: `zsh runs/ablation_202607/onehot_control_200k.sh`.

## When the onehot control lands — RESOLVED 2026-07-05

Onehot@200k = −0.247 ± 0.063 ⇒ within 0.07 of full (−0.233): the plateau
reading was wrong; conclusion #1 amended (see the onehot-control results
section). The recommended next experiment is an **onehot-ff league arm**
mirroring phase 2 (exact commands in that section) — it settles
onehot-vs-full in the regime that matters.

## When phase 2 lands

1. Numbers: `runs/phase2_202607/panel_a_{called,jd}.csv` (2 candidates:
   `phase2_full`, `phase2_no-aux`), `scripted_probe_*.json`,
   `trump_lead_*.json`, per-arm `runs/phase2_<arch>/checkpoints/
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
3. Paste the numbers into "Results — phase 2" above with the verdicts.

## When the capacity sweep lands

Open `runs/size_sweep_202607/report.md` (auto-generated) and apply the
"How to read the report" rules in the Phase 3 section. Paste into
"Results — capacity sweep".

## If something needs rerunning from scratch

All tooling is committed: `analysis/run_ablation_matrix.py` (orchestrator;
`--smoke` for a 5-minute pipeline check), `analysis/aggregate_ablation.py`
(CSVs/plots/table), `analysis/ablation_report.py` (statistics/verdict
tables), `analysis/rigorous_eval.py` + `analysis/scripted_probe.py` +
`analysis/trump_lead_probe.py` (instruments — frozen seeds, see
`Evaluation_Harnesses_202607.md`). Checkpoints record their architecture;
**always load via `ppo.load_agent(path)`**, never construct `PPOAgent`
with a guessed arch.
