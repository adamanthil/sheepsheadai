# Evaluation Harnesses — How to Measure a Checkpoint (July 2026)

Companion to `League_Run_Review_202607.md`. That notebook records *what we
measured and why*; this one is the operator's manual: which instrument to
reach for, the exact command, and the frozen constants that make results
comparable across runs and months. All commands run from the repo root with
the project venv.

**The golden rule: never change a frozen seed or the PANEL-A membership.**
Every harness here uses common-random-number (CRN) deal sets keyed by a
frozen seed, so any two measurements with the same seed are *paired* — their
difference cancels deal luck. Change the seed and you lose pairing with
every number recorded so far.

| instrument | question it answers | script | frozen constant |
|---|---|---|---|
| PANEL-A anchored gauntlet | how strong is this checkpoint, on the reference scale? | `sheepshead/analysis/rigorous_eval.py` | `--seed 42`, PANEL-A (sha256s below) |
| Scripted paired probe | where does it sit on the static, lineage-free scale? | `sheepshead/analysis/scripted_probe.py` | seed `31` |
| Trump-lead incidence probe | is the diagnosed defender trump-lead hole still open? | `sheepshead/analysis/trump_lead_probe.py` | `PROBE_SEED = 20260702` |
| In-training telemetry | is the *live run* healthy / making absolute progress? | written by `sheepshead/training/train_league_ppo.py` | `ANCHOR_EVAL_SEED = 20260701` |

---

## 1. PANEL-A anchored gauntlet (`sheepshead/analysis/rigorous_eval.py`)

**Use for:** any strength claim about a checkpoint, and all
checkpoint-vs-checkpoint comparisons between strong agents. Each candidate
("hero") replays every deal from all 5 seats (duplicate replay) inside a
frozen 4-model reference field; scoring is deal-level with block bootstrap,
and all candidates see identical deals, so candidate deltas are paired.

**PANEL-A (frozen reference field — never change; sha256 prefixes):**

| member | path | sha256 |
|---|---|---|
| 30M reference (top rung) | `final_pfsp_swish_ppo.pt` (repo root) | `cc644b7109d5896b` |
| pfsp mid rung | `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt` | `6951cd42c52a9a84` |
| pfsp early rung | `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt` | `c8ec0d1df24875c8` |
| selfplay common ancestor (floor) | `runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt` | `7f5426b68d3ebc6b` |

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.rigorous_eval \
  --candidates <ckpt1.pt> <ckpt2.pt> ... \
  --anchors final_pfsp_swish_ppo.pt \
            runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt \
            runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt \
            runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt \
  --deals 1000 --partner-mode called --seed 42 \
  --out-csv out_called.csv --out-plot out_called.png
# repeat with --partner-mode jd (always run both modes)
```

* **Power:** 1000 deals ⇒ MDE ≈ 0.07 score/hand (paired); ~531 deals
  resolve a 0.10 gap. Runtime is roughly linear in deals × candidates.
* **Keep `--seed 42`.** All recorded curves used it, so new runs are
  CRN-paired with them.
* **Recorded baselines** live in `runs/rigorous_baseline_202607/`
  (`baseline_{called,jd}.csv/png` — repro-league v1 at 1M/7M/13.65M;
  `reference_lineage_{called,jd}.csv/png` — pfsp lineage at matched
  episodes; `run_baseline.sh` reproduces them). Headline numbers:
  league-13.65M CA −0.120 / JD +0.038; ref-15M CA +0.120 / JD +0.111.
  **The bar for a fresh league run is the repro-v1 control curve.**
* The CLI only loads `.pt` checkpoints; the ScriptedAgent is deliberately
  *not* in PANEL-A (it's a different scale — §2).
* Legacy checkpoints (pre-`value_trunk`) print a critic-compat note on
  load; harmless here — greedy play uses the actor only.

## 2. Scripted paired probe (`sheepshead/analysis/scripted_probe.py`)

**Use for:** absolute placement on a static scale that cannot drift,
sanity-flooring a young run, and cross-run/cross-year comparability. Hero
and ScriptedAgent play the same seat on the same deal against an
all-scripted field; the paired delta is the edge (positive = hero better).
One call covers both partner modes (alternating by deal) and all seats.

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.scripted_probe --ckpt <ckpt.pt> --deals 500
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.scripted_probe --ckpt a.pt b.pt --out-json out.json
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.scripted_probe --self-check   # instrument check: edge must be exactly 0
```

* **Keep the default seed 31** — pairs with the recorded placements:
  selfplay-100k **−0.63 ± 0.24** vs scripted; league-13.65M **+0.34 ± 0.21**
  (150 deals each). Same-seed probes of different checkpoints are
  CRN-paired with each other, too.
* **Not a top-of-ladder yardstick.** The scripted agent is mid-ladder;
  strong-vs-strong comparisons belong in §1. Its other role is hosting
  exploit probes (§3) in a field that can't share the RL lineage's blind
  spots.

## 3. Trump-lead incidence probe (`sheepshead/analysis/trump_lead_probe.py`)

**Use for:** regression-testing the one *diagnosed* behavioral hole — a
defender leading trump on trick 0–1 with a fail lead legal (cost −0.19
score/case; `defender_trump_lead_investigation.md`). Measures incidence,
not exploitation (the tell's exploitation EV is below any affordable
probe's resolution). Hero sits in all 5 seats per CRN deal in a scripted
field; games abandon after trick 1, so it's fast (~3 min for 2000 deals).

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.trump_lead_probe --ckpt <ckpt.pt> --deals 2000 --out-json out.json
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.trump_lead_probe --scripted   # self-check: rate must be 0
```

* Frozen `PROBE_SEED = 20260702` (the default — don't override).
* **Recorded baselines** (`runs/rigorous_baseline_202607/trump_lead_probe_*.json`):
  30M reference JD 1.89% / CA 0.80%; league-13.65M JD 1.21% / CA 2.88%.
  Implied cost −0.25..−0.99 score/1000 hands — a *canary*, not a strength
  lever. Watch the trump-rich (3+ trump) split, where the documented cost
  concentrates.

## 4. In-training telemetry (written by `sheepshead/training/train_league_ppo.py`)

Three streams land in the run's checkpoint dir; together they answer "is
the live run healthy?" without any manual evaluation:

* **`anchored_eval.csv`** — every `--anchor-eval-interval` episodes (default
  100k), a 300-deal paired probe vs a frozen reference (default
  `final_pfsp_swish_ppo.pt`) on the fixed `ANCHOR_EVAL_SEED = 20260701`
  deal set. Probe-to-probe differences are paired, so the *trend* is the
  signal: it should climb toward 0 from below. Flat or falling while gates
  pass ⇒ self-referential progress only.
* **`greedy_health.csv`** — greedy (argmax) bid/behavior rates. Sampled
  rates in the episode log can mask collapse; greedy rates cannot.
  Collapse signature (seen at 150k–250k in the July restart): greedy PICK
  → 0%, leaster → 100%, play-spread ≲ 0.1. Any one of these warrants
  attention; all three together means the run is dead.
* **`gate_result.json` / exploitability gates** — paired
  challenger-vs-incumbent edges (3000 deals) at each generation gate, plus
  best-of-checkpoints exploiter screens. These measure *relative* progress
  and exploitability, and are meaningless if the greedy gates are failing.

## 5. Architecture ablation (self-play arms; `architectures.py` registry)

Six registered architectures form a reverse-historical ladder — each
adjacent pair isolates one addition (see `architectures.py` for the specs
and the literature on multi-seed requirements):

| arch | removes | params (E+A+C) | s/episode¹ |
|---|---|---|---|
| `full` | — (current) | 1,003,607 | 0.26 |
| `full-uninformed` | informed init (factorial arm) | 1,003,607 | 0.27 |
| `no-aux` | critic aux heads | 891,534 | 0.26 |
| `no-transformer` | + transformer reasoning² | 905,102 | 0.15 |
| `no-transformer-uninformed` | + informed init | 905,102 | 0.15 |
| `onehot-ff` | + card-token pipeline (legacy FF+GRU, flat heads) | 1,128,303 | 0.04 |

¹ Single-process, 4 torch threads, 2026 dev Mac; ~7.3 h per 100k-episode
full-family run.

² `no-transformer` = `PooledMemoryEncoder`: embeddings → pools → fused
features → GRU(256,256), heads consume the recurrent state (the
pre-transformer LSTM shape, ppo.py before 0729e11). Plain
`n_reasoning_layers=0` would leave the memory write-only (no attention to
mix it back into features) and conflate "no transformer" with "no memory".

**Protocol (approved 2026-07-03):** 6 archs × seeds {42, 1042, 2042} ×
100k episodes, pure self-play. Launch each arm:

```bash
nohup uv run python -m sheepshead.training.train_selfplay_ppo --arch <A> --seed <S> --episodes 100000 \
    --run-name ablate_<A>_s<S> > runs/ablate_<A>_s<S>.log 2>&1 &
```

* `--seed` fully determines the run (network init, action sampling, AND
  deals — the trainer seeds each episode's `Game`).
* **Curves:** `runs/<run>/anchored_eval.csv` every 5k episodes — paired
  300-deal CRN edges (`ANCHOR_EVAL_SEED = 20260703`) vs three fixed
  yardsticks: `edge_scripted` (ScriptedAgent; resolves early curve),
  `edge_selfplay100k` (strength-matched reference), `edge_final_pfsp`
  (absolute yardstick, saturated-negative early). `train_wall_s` excludes
  eval time; `transitions_done`/`updates_done` give the sample axis.
* **Endpoints:** each run's `final_<arch>.pt` through §1 PANEL-A (both
  modes), §2 scripted probe, §3 trump-lead probe. `sheepshead/analysis/` tools are
  arch-aware via `ppo.load_agent` (checkpoints record their `arch`;
  pre-registry checkpoints = `full`). Saved-model names carry the arch, not
  the activation — the relu option is gone, SiLU everywhere (old
  `*_swish*.pt` files on disk keep working, incl. a legacy fallback in
  `exploiter.latest_checkpoint`).
* **Decision criteria:** *training speed* = median (across seeds) episodes
  AND train-wall-clock to fixed edge thresholds vs scripted/100k (report
  both axes — cheaper archs win s/episode; the question is sample
  efficiency). *Ceiling (self-play regime)* = mean final PANEL-A pt/game;
  deltas > 0.07 (the 1000-deal MDE) treated as real; report per-seed
  spread + collapse count (edge regressing > 2 SE from its own peak).
  Component verdicts come from adjacent-rung deltas.
* Extension rule: top-2 arms get +100k episodes if the last-20k anchored
  edge slope is still > 1 SE above zero.
* **Phase 2 (the honest ceiling test):** top-2 archs through
  `sheepshead/training/train_league_ppo.py --arch <A>` for 5–10M episodes — self-play ceiling
  ≠ league ceiling.

Gotcha: aux-head consumers (`server/services/analyze.py`,
`visualizations/dump_forward_pass.py`) hard-fail on no-aux checkpoints by
design. League members/exploiters inherit the run's arch; mixed-arch
leagues work for play but exploiter warm-starts must stay same-arch.

## 6. Gotchas that have already bitten

* **`PYTHONPATH=.`** is required for everything under `sheepshead/analysis/`.
* All harnesses play **greedy/deterministic**; results say nothing about
  the sampled policy's entropy.
* **Zero-sum invariant:** any score-based tool can smoke-test itself by
  asserting per-deal scores sum to 0 across the 5 seats — this is how the
  leaster tie-break engine bug was caught (fixed f67a827).
* Piping test runs through `| tail` eats the exit code; check
  `${pipestatus}` or don't pipe.
* If a probe needs the **critic** (search, value baselines — not the greedy
  harnesses above), remember legacy 30M-era checkpoints load `value_trunk`
  randomly under `strict=False`; the loader now routes through
  `critic_adapter` and prints a note. Verify the note appears.
