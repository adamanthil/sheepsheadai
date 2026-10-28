# Architecture Ablation — Recall-Contract Ladder (October 2026)

Successor to `Architecture_Ablation_202607.md` (the July study), built for
the write-up: a clean, additive architecture ladder trained through the
**202609_recall_rc bootstrap phase exactly as the release trained it**,
from a one-hot MLP baseline to the released `perceiver-recall`.

**Question:** under the release's observation contract (no blind/bury
re-injection; Training_Program_Redesign_202609 §3.2) and its bootstrap
recipe, what does each major architectural addition buy in shaped
self-play learning — speed, 400k-episode skill, and convention learning?

---

# 1. Current state

| item | value |
|---|---|
| status | **RUNNING** — launched 2026-10-28 (§5 log) |
| driver | `python -m sheepshead.training.run_recall_ablation {train,eval,status}` |
| run code | detached worktree `/Volumes/Nargothrond/dev/sheepsheadai-ablation-202610` (own `.venv`; `runs/`, `models/` symlinked to the main checkout) at the commit in §5 |
| outputs | `runs/arch_ablation_202610/<arch>_s<seed>/` (checkpoints, `train.log`, `status.json`, `eval/`); driver log `runs/arch_ablation_202610/driver.log` |
| results | `notebooks/arch_ablation_202610.csv`, `notebooks/arch_ablation_202610_deltas.csv`, `notebooks/figures/arch_ablation_202610_panelA.png` — rebuilt by `python -m sheepshead.research.recall_ablation_report` (safe to run any time; unfinished cells are blank) |
| progress | `uv run python -m sheepshead.training.run_recall_ablation status` |

Order of work: all training (one run at a time, seed-major: the full
seed-42 ladder, then 1042, then 2042), then every evaluation read
(8 single-threaded processes), then the report. Both stages are
resumable — rerunning `train` skips finished cells and resumes an
interrupted cell from its newest checkpoint (recorded as a second
segment in its `status.json`, `resumed: true` in the CSV); rerunning
`eval` skips existing outputs.

---

# 2. Design

## 2.1 The ladder

Seven rungs; each adds one component to the rung below it (except
5→6, see §2.3). All rungs read exactly `observation.RECALL_KEYS` and
carry a recurrent memory (the only route by which the picker's blind and
bury reach later decisions).

| rung | arch | adds | informed init | aux heads | bilinear pointer | params |
|---|---|---|---|---|---|---|
| 1 | `onehot-recall` | baseline: flat one-hot → MLP + GRU memory, flat per-phase heads | — | — | — | 1,093,487 |
| 2 | `pooled-recall-uninformed-noaux` | card tokens + per-bag attention pools + pointer actor | — | — | — | 837,390 |
| 3 | `pooled-recall-noaux` | + informed card-embedding init | ✓ | — | — | 837,390 |
| 4 | `full-recall-noaux` | + transformer card reasoning (4 layers × 4 heads) | ✓ | — | — | 823,822 |
| 5 | `full-recall` | + aux critic heads (= the production 30M design under the recall observation) | ✓ | ✓ | — | 935,895 |
| 6 | `perceiver-recall-nobilinear` | + shared 16-query readout + memory-token memory driver | ✓ | ✓ | — | 1,099,863 |
| 7 | `perceiver-recall` | + bilinear state × card play pointer (**the release**) | ✓ | ✓ | ✓ | 1,120,343 |

Lineage: rungs 2–5 are the July ladder (`no-transformer-uninformed` →
`no-transformer` → `no-aux` → `full`) under the recall observation; rung
1 is `onehot-ff` minus the blind/bury one-hot sections (434 → 366
inputs). Registry entries: `sheepshead/agent/architectures/registry.py`
("Recall-contract ladder").

## 2.2 Recipe (identical for every cell)

The release's phase-0 command (`runs/202609_recall_rc/orchestrator.log`,
2026-09-16) plus the arch, seed and an explicit aux scale:

```bash
python -m sheepshead.training.train_ppo --phase bootstrap --arch <ARCH> \
    --run-name arch_ablation_202610/<ARCH>_s<SEED> --until 400000 \
    --save-interval 50000 --greedy-eval-interval 50000 --greedy-eval-games 200 \
    --num-workers 8 --seed <SEED> --aux-det-scale 1.0
```

Shaped reward, limited critic, all heads trainable, γ 0.99,
`BootstrapHyperparams` (4096-transition updates, 4 epochs × 256, the
self-play entropy schedule over 400k), leaster watchdog ON (default),
no worker device/compile flags (as the release). Seeds 42, 1042, 2042
(the July convention).

**Aux coefficients at 1.0×.** `04cd382` (09-28) baked 2.5× coefficients
for the four deterministic aux heads into `PPOAgent`; the release
bootstrap (09-16) predates the scale option (introduced 09-18) and
trained at 1.0×. `train_ppo --aux-det-scale` (default 2.5, unchanged
behavior elsewhere) restores the original values for this study.

**The release's bootstrap IS cell (perceiver-recall, 42).** Same recipe,
same seed, aux 1.0×; audited 2026-10-28: every commit touching the
bootstrap training path after 09-16 is either the aux change, the
bidding phase, the league entropy controller, the policy-iteration
corpus, import/exception refactors, or the opt-in deploy sampler. It is
read, not retrained (`run_recall_ablation.REUSED`).

## 2.3 Known confounds (state them in the write-up)

1. **Rung 5→6 changes two things** (shared readout replacing the bag
   pools, and the memory driver moving from the context token to the
   memory token) and adds ~164k parameters (the 16-query readout). The
   single-switch decomposition arm `shared-recall` (v2 readout, context
   driver) was dropped by operator choice (July: v2 ≈ full); it can be
   added later at ≈1.3 days for 3 seeds without disturbing anything here.
2. **Order dependence.** Ladder deltas are conditional on the rungs below.
   July measured informed init at +0.060 without the transformer vs
   +0.150 with it (screening grade); here it enters at rung 3, before
   the transformer, so its delta is expected to be the smaller one.
3. **Rung 1→2 bundles the pointer actor with card tokens** (the pointer
   scores cards by their tokens; there is nothing to point at in a
   one-hot vector).
4. **Parameter counts differ** (0.82M–1.12M). Not param-matched; the
   counts are reported alongside.
5. **Regime.** Shaped self-play bootstrap only — July's P3 rule made this
   regime screening-grade for adoption decisions; here no adoption rides
   on it, the bootstrap phase is the object of study.

## 2.4 Decision log

| date | decision | by |
|---|---|---|
| 10-28 | New ablation on the bootstrap phase, recall observation, release hyperparameters; MLP baseline and release as endpoints | operator |
| 10-28 | Bilinear pointer gets its own step (operator: it made optimal conventions much faster to learn) | operator |
| 10-28 | Additive ladder only (no leave-one-out arms); aux and informed init enter as ladder steps; `shared-recall` dropped | operator |
| 10-28 | Order follows the July ladder (informed init before transformer) | proposed, accepted with the ladder |
| 10-28 | Aux coefficients 1.0× (directly comparable to the release bootstrap); 3 seeds | operator |
| 10-28 | Release bootstrap reused as (perceiver-recall, 42) | follows from 1.0× |
| 10-28 | Training sequential at equal load; evaluation batched afterwards (8 procs) so wall-clock / eps/s are comparable across cells | implementation |
| 10-28 | Run from a detached worktree with its own venv (branch switches / `uv sync` in the main checkout cannot reach the live run) | implementation |

---

# 3. Instruments and pre-registered reads

## 3.1 Instruments

| instrument | spec | stages | per read |
|---|---|---|---|
| **PANEL-A** (headline) | `league_progress_eval.eval_endpoint`, panel `PANEL-A` by manifest key (30M, pfsp 15M, pfsp 5M, selfplay 100k), 3,996 CRN deals (seed 42), both modes, three-checkpoint composite (N−100k, N−50k, N) | 150k…400k every 50k (6 reads/cell) | ~40 min single-thread |
| **PANEL-B** | same, panel `PANEL-B` (30M, v2 release, iter-11 P1, v2 league 8M) | 400k | ~40 min |
| **Convention battery** | the program's `greedy_health_probe` × 4 seeds × 1,000 greedy self-play games (pick/alone/leaster rates, t0 trump lead, partner trump lead, called-suit lead, logit spread, aux-head reads where present) | every checkpoint, 50k…400k | ~4 min |
| run-level | wall-clock (sequential, equal load), eps/s, watchdog engage/release episodes (from `train.log`), parameter count | per cell | free |

PANEL identity changed with the model-artifacts work (10-27): reads are
by manifest key with design hashes PANEL-A `3e6443e4c5e9`, PANEL-B
`04046aaca26e`. The release's recorded bootstrap reads
(`panel_reference.csv`, old hashes) are **not** pairable and are
re-read here like every other cell.

## 3.2 Reads (pre-registered 2026-10-28, before any new cell trained)

- **Endpoint** = the 400k composite (checkpoints 300k/350k/400k) — the
  release's protocol, and July rule 1 (last three checkpoints).
- **Rung delta** = paired per-deal difference between rung k and rung
  k−1 on the same seed (same CRN deals), averaged over seeds; seed-level
  SE from the three per-seed deltas. Reported for every PANEL-A stage
  and the PANEL-B endpoint (`arch_ablation_202610_deltas.csv`).
- **Calling a delta real** (descriptive; no adoption decision rides on
  this study): |seed-mean delta| > 2 seed-level SE **and** all three
  seeds agree in sign. Anything else is reported as unresolved with its
  interval — not as "no effect".
- **Learning speed** is read off the PANEL-A curve (150k→400k) and the
  watchdog escape episodes; no slope claims from the in-trainer
  greedy-health rows (July rule 2).
- **Bilinear hypothesis** (operator, pre-registered): rung 6→7 shows up
  most clearly in the convention battery — partner trump lead and
  called-suit lead rates rising earlier/faster in rung 7 — possibly
  more than in the panel delta (panels dilute rare-situation skill,
  July §5.11 validity limits).

## 3.3 Throughput and budget (smoke 2026-10-28; 600 episodes, 8 workers)

| arch | eps/s (smoke) | est. 400k |
|---|---|---|
| onehot-recall | 36.7 | ~3.0 h |
| pooled-recall-* | 25.1–25.7 | ~4.4 h each |
| full-recall-noaux / full-recall | 12.6–12.7 | ~8.8 h each |
| perceiver-recall-nobilinear | 12.9 | ~8.6 h |
| perceiver-recall | (release, measured) 14.3 | 7.77 h |

≈ 46 h of training per seed → ≈ 129 h (≈ 5.4 days) for 20 new cells;
evaluation ≈ 147 panel reads + 168 convention reads on 8 processes
≈ 14–16 h. **Expected completion ≈ 2026-11-03/04.** The seed-42 ladder
completes first (≈ 1.6 days in).

---

# 4. Results

*(pending — filled from the report outputs as cells land)*

## 4.1 Endpoint table (400k composite; mean over seeds ± seed SE)

| rung | arch | PANEL-A | PANEL-B | Δ vs rung below (A) | wall h | eps/s | watchdog escape ep |
|---|---|---|---|---|---|---|---|
| 1 | onehot-recall | | | — | | | |
| 2 | pooled-recall-uninformed-noaux | | | | | | |
| 3 | pooled-recall-noaux | | | | | | |
| 4 | full-recall-noaux | | | | | | |
| 5 | full-recall | | | | | | |
| 6 | perceiver-recall-nobilinear | | | | | | |
| 7 | perceiver-recall | | | | | | |

## 4.2 Learning curves

`notebooks/figures/arch_ablation_202610_panelA.png`

## 4.3 Conventions

---

# 5. Log

- **2026-10-28** — Design settled (§2.4). Code: recall layouts for the
  one-hot and pooled encoders; six ladder registry entries + golden
  fixtures (structural; the existing 36 fixtures re-verified
  bit-identical on this machine against a pre-change capture, their
  stale macOS-14 manifest left as is); `PPOAgent.set_deterministic_aux_scale`
  + `train_ppo --aux-det-scale`; `training/run_recall_ablation.py`
  driver; `research/recall_ablation_report.py`. Smoke: all six new arms
  train through the 8-worker bootstrap (600 episodes, updates,
  checkpoints, greedy health). Eval path checked on the release cell
  (12-deal panel read to scratch; the 400k convention read is real and
  kept).
- **2026-10-28 01:12** — LAUNCHED at `f211f73` from the detached
  worktree via `nohup caffeinate -is zsh runs/arch_ablation_202610/launch.sh`
  (train → eval --jobs 8 → report; launcher output
  `runs/arch_ablation_202610/launch.out`, final marker
  `ABLATION COMPLETE`). First cell onehot-recall s42. The worktree's
  `.venv` was built with `uv sync --frozen --extra dev`; its training
  suite passed (505) before launch, and the aux-only commit `df815c5`
  passed the recall/golden tests on its own.
