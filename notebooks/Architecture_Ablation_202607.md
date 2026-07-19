# Architecture Ablation — Experiment Log (July 2026)

Companion to `Evaluation_Harnesses_202607.md` §5 (protocol),
`Extended_League_202607.md` (league orchestrator pre-registration), and
`architectures.py` (the registry + literature). This notebook is the record
of the experiment: current state, standing rules, exact commands, and the
full history with results.

**Question:** what did each historical architecture addition buy, in
training speed (sample efficiency AND wall-clock) and in skill ceiling —
and is the current architecture the right base going forward?

## How to read this notebook

Reorganized 2026-07-14 (was a 2000-line chronological accretion; nothing
was deleted, only reordered and de-duplicated):

- **§1 CURRENT STATE** — what is running, what to do when it lands, and
  the decision ledger. If you are picking this program up, start here and
  you can act without reading anything else first.
- **§2 STANDING RULES** — operator preferences (P1–P5), measurement rules
  (1–5), the league stopping rule, and a known-traps checklist. These
  apply to *everything*; read before running or interpreting anything.
- **§3 REFERENCE** — environment, architecture registry with param
  counts, instruments/yardsticks, standard command shapes, output file
  map, automation inventory.
- **§4 SETTLED FINDINGS** — what we now know, organized by question, each
  with its evidence grade (decisive / screening-only / mechanism).
- **§5 CHRONOLOGICAL LOG** — the full dated history with all tables and
  per-seed numbers. Verdicts that were later overturned are tagged
  **[SUPERSEDED]** with a pointer to what replaced them.
- **Appendix A** — superseded plans (phase 2, full-based capacity sweep,
  probe contingency playbooks), kept verbatim for the record.

Status tags used throughout: **[CURRENT]** = operative; **[SUPERSEDED →
§x]** = kept as history, do not act on; **[SCREENING]** = shaped-self-play
evidence, per rule 3 not settled fact.

Operator model access ended **2026-07-12 23:59** (historical constraint —
several automation choices below exist because everything after that date
had to be executable mechanically).

---

# 1. CURRENT STATE (updated 2026-07-14)

## 1.1 Where the program stands

One paragraph: the self-play screening program (stage 0) closed 2026-07-11.
Outcome: plain `perceiver` OUT (decisively worse than `full` at equal 400k
budget, t = −4.8); `onehot-ff` OUT (decisively worse at 400k, t = −6.99 —
its 100k/200k ties were budget artifacts); **`perceiver-shared-v2`
QUALIFIED** (within the 0.07 MDE of full@400k, gap 0.050). The decisive
stage-1 experiment — paired oracle-critic league arms, `full` vs
`perceiver-shared-v2` — launched 2026-07-11 20:31 and is running now
(§1.2). Adoption is settled by its endpoint panels per preference P1
(§2.1). After adoption, the winner continues into the extended-league
orchestrator with the exact command in §1.3.

Decision ledger (details in §4 / §5):

| date | decision | basis |
|---|---|---|
| 07-05 | `full` + `no-aux` extended to 200k; onehot not extended | pre-registered slope rule (later shown unreliable) |
| 07-05 | onehot-vs-full "tie" at 200k; league arm queued as decisive test | budget-equal control **[SUPERSEDED 07-11: onehot OUT]** |
| 07-06 | leaster watchdog built, default OFF; from-scratch collapse is universal | leaster-rate scan of all 24 runs |
| 07-06 | oracle critic rebuilt perceiver-style (no pooled-oracle ckpts exist) | operator call, no compat constraints |
| 07-07 | tokenread probe NULL; perceiver probe LOSS; keep `full` as base | pre-registered probe rules **[partially superseded by P1]** |
| 07-07 | program re-planned: P1–P5 + stage 0/1/2 decision tree | budget-artifact discovery |
| 07-07 | decomposition matrix displaced by 400k budget-equal extensions | operator: "why it lost" < "is it on track to win" |
| 07-08 | selfplay flush update removed; finals from pre-07-08 runs deprecated | flush-update finding (rule 5) |
| 07-09 | plain perceiver OUT at 400k (−0.259, t=−4.8, 3/3 seeds) | pre-registered budget-equal read |
| 07-09 | `perceiver-shared-v2` built (16q readout + normed proj) and launched | mechanism audits: channels + LN |
| 07-10 | onehot 200k tie STANDS on ckpt methodology; both ext2 arms → 400k | ckpt200k re-panels; operator order |
| 07-11 | onehot OUT (−0.140, t=−6.99); **v2 QUALIFIED** (−0.050, sub-MDE) | pre-registered ext2 reads |
| 07-11 | stage-1 generations 750k → **1M** (extended-league alignment) | calibration argument, §5.10 |
| 07-11 | stage 1 LAUNCHED: full vs perceiver-shared-v2, oracle league | §1.2 |

## 1.2 Stage 1 — paired oracle-critic league arms **[COMPLETE 2026-07-18; verdict §5.11]**

The regime that matters (P3). Two arms, identical protocol, CRN seed 42,
concurrent, 4 workers each.

- **Arms:** `full` vs `perceiver-shared-v2`.
- **Warm starts:** each arch's best pre-registered 400k self-play
  checkpoint by both-modes panel (strength-matrix row 400000):
  full_s2042 (+0.0097) and v2_s2042 (−0.0882) — both happen to be s2042,
  picked independently. Used as **renamed copies**
  `runs/league_arch_<ARCH>/warmstart_<ARCH>_400k.pt` (see the
  `checkpoint_` filename trap, §2.4). Rule-5 audit 2026-07-11:
  `ablate_full400_s2042/final_full.pt` tensor-DIFFERS from its 400k
  checkpoint (post-final flush — finals unusable for full); the v2 s2042
  final IS tensor-identical, but both arms use the renamed-copy route for
  symmetric treatment.
- **Budget:** 2 generations × 1,000,000 episodes per arm (gen 1 anchored,
  gen 2 released). 1M per generation is the calibrated extended-league
  number (amendment rationale in §5.10).
- **Launcher:** `runs/stage1_202607/stage1.sh`, started 2026-07-11 20:31
  via `nohup caffeinate -is zsh runs/stage1_202607/stage1.sh`. Per-arm
  logs `runs/stage1_202607/league_arch_<ARCH>.log`; phase markers in
  `runs/stage1_202607/stage1.log`; final marker `STAGE1 COMPLETE`.

Exact trainer commands (per arm; ARCH = full | perceiver-shared-v2):

```bash
# gen 1 (anchored, oracle critic, league seeded with reference selfplay ckpts)
.venv/bin/python -m sheepshead.training.train_league_ppo \
    --arch <ARCH> --critic-mode oracle \
    --resume runs/league_arch_<ARCH>/warmstart_<ARCH>_400k.pt \
    --seed-checkpoints 'runs/reference_selfplay_ppo/checkpoints/*.pt' \
    --league-dir runs/league_arch_<ARCH>/league --run-name league_arch_<ARCH> \
    --generations 1 --main-episodes 1000000 \
    --anchor-coeff 1.0 --anchor-ref runs/league_arch_<ARCH>/warmstart_<ARCH>_400k.pt \
    --num-workers 4 --seed 42 --schedule-horizon 20000000
# gen 2 (anchor-free, resumed at the absolute 1M boundary; league roster inherited)
.venv/bin/python -m sheepshead.training.train_league_ppo \
    --arch <ARCH> --critic-mode oracle \
    --resume runs/league_arch_<ARCH>/checkpoints/pfsp_<ARCH>_checkpoint_1000000.pt \
    --league-dir runs/league_arch_<ARCH>/league --run-name league_arch_<ARCH> \
    --generations 1 --main-episodes 1000000 \
    --num-workers 4 --seed 42 --schedule-horizon 20000000
```

All other trainer flags are at defaults, which are IDENTICAL to the
extended-league orchestrator's `trainer_cmd` values (exploiter 50k / gate
3000 / screen 200 / update 2048 / save 50k / snapshot 50k / greedy 50k×200
/ horizon 20M) — verified 2026-07-11, so a stage-1 generation is
indistinguishable on disk from an orchestrator-launched one.

After both arms finish, the script runs the endpoint instruments
automatically: trump-lead probe (2000 deals) + scripted probe (500 deals)
on each arm's 2M checkpoint, then ONE shared-CRN PANEL-A gauntlet per mode
over BOTH arms' last-3 checkpoints (1.9M/1.95M/2M, six candidates, four
PANEL-A anchors, 1000 deals, seed 42) →
`runs/stage1_202607/panel_stage1_{called,jd}.csv`.

**Progress snapshot 2026-07-14:** full at ep ~883k, v2 at ep ~927k of the
gen-1 1M — both ~3.3–3.4 eps/s (faster than the 4.7-day/gen estimate);
gen-1 boundary lands ~Jul-14/15, both generations + panels ~Jul-18.
Health: anchor_kl ≈ 0.01 (well-behaved), leaster 0.7–1.2%, exploiter seat
share 0.00 (exploiter phases fire at generation boundaries), `ev_oracle`
above `ev_limited` in both arms (oracle critic healthy; the warm-start
shock sim predicted no burn-in needed — confirmed).

**Run log, consolidated 2026-07-18 — training COMPLETE both arms;
panels running.** (Consolidates the 07-15…07-18 health checks;
extends the 07-14 snapshot above.)

- **Timeline:** gen-1 ends Jul-15 (v2 07:19, full 11:49); gen-2 ends
  Jul-18 (v2 07:55, full 14:16); all four rc=0. Endpoint probes done
  Jul-18 14:24; panels 14:24–17:17; `STAGE1 COMPLETE` logged
  Jul-18 17:17.
- **Gen 1 (anchored): clean in both arms.** anchor_kl ≈ 0.01, leaster
  0.7–1.5%, no warm-start shock; oracle critic healthy throughout both
  generations (ev_oracle 0.3–0.5 vs ev_limited ≈ 0). Anchored-eval
  300-deal probes (noisy, non-decision): full improved −0.32 → −0.087
  at 1M (last-3 mean −0.17); v2 flat ~−0.30 (last-3 mean −0.32).
- **Gen-1 exploiter gates: PASSED both arms** at near-identical gated
  edges — full +0.100 ± 0.037, v2 +0.111 ± 0.045 (3000 deals, ≥2 SE
  over the 0.1 floor); exploiters inserted, seat share 0.10–0.11
  through gen 2, so exploiter pressure was live (unlike the inert
  repro-league gens 1–11). Directional only: v2 looked the softer
  target (screen edges up to 0.475 vs full's 0.185; deviating_frac
  0.31 vs 0.21).
- **Gen 2 (anchor-free): full COLLAPSED into the leaster attractor;
  v2 stayed healthy.** Both arms wobbled right after anchor release
  (~1.05–1.1M). v2's transient resolved and it oscillated in a normal
  band thereafter (greedy PICK 21–41%; operator read 07-17: within the
  30M run's own training oscillation, ~30–40% final ideal; endpoint
  rows 1.9M/1.95M/2M = PICK 41.5/36.3/22.1%, leaster 3.5/4.5/25%).
  full's deepened into the §1.3 dead-arm criterion — first hit at 1.5M
  (greedy PICK 0.10%, leaster 99.5%), partial rebound 1.6–1.75M (PICK
  9–15%), then terminal relapse: **endpoint rows = PICK 1.0/0.6/0.0%,
  leaster 95/97/100%**. Classic ExIt PASS-collapse signature. Chronic
  ALONE>20% greedy-gate violations in both arms throughout (full
  16–27%, v2 23–36%) — behavioral quirk, not a stopper.
- **Endpoint probes @2M (Jul-18):** scripted (500 deals): v2
  **+0.150 ± 0.119** (above the sanity floor), full −0.116 ± 0.130
  (below it — collapsed policy). Trump-lead canary (2000 deals): clean
  in both modes for both arms (v2 zero leads in ~1800 defender
  opportunities per mode; full 1/~2100, though full's read is of a
  leaster-locked policy and carries little information).
- **Read-ahead of §1.3/§5.11** (final numbers await
  `panel_stage1_{called,jd}.csv`): full is a dead arm per rule 1 and
  decides nothing ⇒ the P1 rule adopts **perceiver-shared-v2 by
  default**; the panels' remaining job is quantifying v2's endpoint vs
  its −0.115 self-play@400k start (league-regime lift). The PRIMARY
  residual finding is the anchor-free gen-2 instability itself, and it
  is full-specific in this run (v2 survived the same transient).
  Arch-linked fragility vs seed luck is unresolved at n=1 seed;
  candidate mechanisms (07-17 discussion): aux-forced shared readout
  as representational ballast (supervised aux grads keep the
  actor-read vector grounded during policy degeneracy — discriminator
  = v2-noaux arm under anchor release), soft capacity reallocation vs
  hard bag-scoped pools under opponent shift, residual plasticity
  (v2 still out-sloping full at stage-0 end), normed readout scale
  bounding; cheap falsifier for seed luck = rerun full gen-2 from its
  1M boundary on a new seed. CONSEQUENCE for the extended-league
  continuation (anchor-free from gen 3 on): consider a guardrail
  before launch — leaster watchdog on the league trainer, a weak
  retained anchor, or a gen-boundary greedy-health gate.

- Warm-start shock: none observed (as predicted by the shock sim, §4.7).
- Oracle cost: expect ~1.5–2× wall clock vs limited (measured 2.7× update
  cost); knob if needed = oracle loss on 1 of 4 epochs.
- Instruments per generation: PANEL-A both modes (last-3-ckpt rule) +
  gen-vs-gen h2h + trump-lead probe.

## 1.3 When stage 1 lands: the mechanical playbook **[EXECUTED 07-18; results §5.11]**

1. Confirm `STAGE1 COMPLETE` in `runs/stage1_202607/stage1.log`; check
   both arms' `runs/league_arch_<ARCH>/checkpoints/greedy_health.csv` for
   collapse (greedy PICK → 0% or leaster → 100% = dead arm; a collapsed
   arm decides nothing).
2. Read `runs/stage1_202607/panel_stage1_{called,jd}.csv`. Endpoint per
   arm = mean over its 1.9M/1.95M/2M rows, both modes (rule 1).
3. **Adoption rule (P1): adopt `perceiver-shared-v2` unless `full` wins
   by > 0.07 AND > 2 SE** on those shared-CRN both-modes means.
4. Secondary reads: trump-lead probes (the defender trump-leak canary —
   self-play values at 400k were low but the leak has appeared in past
   runs at 10–20%), scripted probes (sanity floor), per-arm
   `exploitability.csv` (gate edge per generation, lower/declining =
   better), and both arms vs their −0.065 / −0.115 self-play@400k
   starting points (league-regime lift).
5. Paste the numbers + verdict into §5.11 (placeholder at the end of the
   log).

**Continuing the adopted arm with `run_extended_league` (post-adoption):**

```bash
nohup uv run extended-league \
    --arch <WINNER> \
    --resume runs/league_arch_<WINNER>/warmstart_<WINNER>_400k.pt \
    --seed-checkpoints 'runs/reference_selfplay_ppo/checkpoints/*.pt' \
    --run-name league_arch_<WINNER> \
    --anchor-coeff 1.0 \
    > runs/league_arch_<WINNER>_ext.log 2>&1 &
```

Why this resumes cleanly (verified against the orchestrator source
2026-07-11):

- `--run-name league_arch_<WINNER>` ⇒ league-dir defaults to
  `runs/league_arch_<WINNER>/league` (same as stage 1) and
  `ensure_trained(g)` finds each stage-1 generation's boundary checkpoint
  (`checkpoints/pfsp_<ARCH>_checkpoint_{1000000,2000000}.pt`) plus its
  `exploitability.csv` gate row, so generations 1–2 are treated as
  complete: the orchestrator runs their 4000-deal composite panels + h2h,
  applies the stop rule, and starts TRAINING at generation 3
  (resume-chained from the 2M boundary checkpoint).
- `--resume` = the same warm-start file: parses to episode 0 (no
  `checkpoint_` in the name), records the correct arch metadata, defines
  the generation-0 endpoint, and is the anchor-ref on the books (gen 1
  never retrains, so the anchor is historical).
- `--anchor-coeff 1.0` skips the calibration phase and records the
  coefficient stage 1 actually used.
- `--main-episodes` default (1,000,000) matches stage 1 — the point of
  the 07-11 amendment; boundary math `g × 1M` lines up exactly.
- `--num-workers` (default 8) is free to differ from stage 1's 4: it is
  not state-bearing, and the winner runs alone.
- Composite endpoints need boundary−100k/−50k checkpoints: guaranteed by
  save-interval 50k during stage 1 (default).
- The stop rule then governs: floor 4 generations, cap 12, two
  consecutive flats + fresh-deal confirmation to stop (per the
  Extended_League pre-registration; generation numbering continues
  3, 4, …).

The loser's run directory stays as a complete 2-generation record (its
own extended-league continuation remains possible later with the mirrored
command).

**Optional third arm / follow-up — the aux-contribution measurement (P2):**
`perceiver-shared-v2` vs `perceiver-shared-v2-noaux` (registered; generic
build/play/update/roundtrip tests cover it), same league protocol. This is
the definitive aux measurement: under the oracle critic the only surviving
aux channel is trunk-shaping for the actor. The phase-2 questions this
program displaced (full vs no-aux league; onehot league arm) are ANSWERED
or OBSOLETED; run the onehot league arm later only if the writeup wants
the token-stack-vs-flat story at league scale (`phase2_onehot.sh`,
unlaunched, Appendix A.1).

## 1.4 Stage 2 — size sweep, redesigned (P4) **[CURRENT, not started]**

On the ADOPTED base from stage 1, not before:

- **Coarse screen (self-play):** one-knob variants × 3 seeds × **400k**
  episodes, `--leaster-watchdog`, base arch included in-sweep, endpoints
  by rule 1. (400k, not 200k: bigger models are plausibly slower climbers
  — a 200k sweep would systematically favor small variants; this is
  rule 3 applied prospectively.)
- **Confirm (league):** top-1/2 variants + base through one oracle-league
  generation each before any switch.
- If the base is perceiver-shared-v2: size/attention-shape registry
  variants for the shared encoder DON'T EXIST YET (only `perceiver-*`
  variants do) — needed code: a `_perceiver_shared_size_variant` factory
  mirroring `_perceiver_size_variant` (ask the operator before building;
  ~1h with tests).
- Attention-shape knobs (readout queries/heads, reasoning heads) fold
  into the same sweep on the adopted base. Head-count variants are exactly
  param-matched (MHA params don't depend on num_heads) so any delta is
  pure structure.
- The watchdog-on baseline killed on 07-07 (`sweep_full_s*`) is rerun
  only as part of this stage; use prefix `sweep` (see the prefix trap,
  §2.4).

## 1.5 Writeup figure pipeline (P5) **[CURRENT]**

- **PANEL-A strength matrix:** `notebooks/panel_a_strength_matrix.csv` —
  wide format, one row per 25k snapshot, one column per arch+seed lineage
  (both-modes panel mean; blank where panels haven't run), plus
  `_edge_scripted` / `_edge_100k` supplementary columns (300-deal
  anchored edges — motivation-only, never publishable). Per-mode
  companions `panel_a_strength_matrix_{called,jd}.csv` carry the same
  panel columns separated by partner mode (no edge columns — the anchored
  edges aren't mode-separable). Regenerate after any backfill with
  `uv run python -m sheepshead.analysis.build_panel_matrix` — it globs
  `runs/perceiver_202607/diag/panel*.csv` and picks up new
  checkpoint-based panels automatically (finals-based rows are
  structurally excluded via the filepath pattern, rule 5).
- **Panel-grade learning curves** (the headline figure): every probe run
  and extension has 25k-interval checkpoints on disk. Batch
  `rigorous_eval` over all checkpoints of an arch (one call, many
  candidates, both modes, seed 42) → score-vs-episodes with CIs,
  CRN-comparable across archs. Backfillable any time for full / perceiver
  / onehot-ff / no-aux / tokenread / extensions from existing run dirs
  (~2–3h background per arch batch; entirely mechanical).
- **Efficiency table:** param counts (registry) + equal-load per-decision
  act bench + update-phase timing split. Method scripts `act_bench.py`,
  `readout_attention_audit.py`, `oracle_update_bench.py`,
  `oracle_shock_sim.py` lived in the assistant session scratchpad — they
  were flagged for committing under `sheepshead/analysis/diagnostics/`
  before 2026-07-12; **verify they were committed before relying on them**
  (the committed diagnostics are `aux_audit.py`,
  `readout_squeeze_audit.py`, `readout_rank_audit.py`, `leaster_scan.py`).
- **Mechanism figures:** readout-attention coverage by token group over
  training (audit script), pass-collapse escape table (leaster_scan).
- Standing caveat for the text: 300-deal anchored curves are
  motivation-only; every published number is a checkpoint panel.

## 1.6 Loose ends ledger

- Decomposition matrix (readout-actor / readout-critic × 3 seeds) PAUSED
  ~5.5h in on 07-07 (partial data in `runs/ablate_readout-*`);
  restartable via the orchestrator command in §5.6 — now a "why"
  question, interesting for the writeup, not on the critical path.
- `perceiver-ctxmem` + `perceiver-aux` registered, never launched (the
  memory-driver and per-network-aux questions — writeup material).
- Watchdog-on `sweep_full_s*` baseline killed ~3.8h in on 07-07; rerun
  only when stage 2 launches (prefix `sweep`).
- `runs/size_sweep_202607/watch_and_launch.sh.new` staged file is STALE
  (pre-dates the 07-07 re-planning); do not relaunch it.
- Phase-2 league arms (full/no-aux) killed 07-07 morning; superseded by
  stage 1. `phase2_onehot.sh` unlaunched (optional writeup arm).
- `perceiver-shared-v2-noaux` registered — the stage-1 aux arm (§1.3) is
  launchable as written.
- Stale "running" status JSONs exist in `runs/size_sweep_202607/status/`
  and `runs/decomp_202607/status/` — harmless (skip logic only honors
  "done").
- Scratchpad method scripts (see §1.5 efficiency table) — commit status
  unverified after the access window closed.

---

# 2. STANDING RULES (always apply)

## 2.1 Codified operator preferences (P1–P5) — set 2026-07-07

- **P1 — Architecture preference:** the operator prefers the
  perceiver/perceiver-shared family on theoretical and engineering
  grounds. **Ties go to the perceiver-variant.** This REPLACES the old
  "tie ⇒ keep full" bar: `full` is retained only if it beats the best
  perceiver-variant DECISIVELY (> 0.07 and > 2 SE, CRN-paired) in the
  decisive test (P3).
- **P2 — Aux heads ship.** The deployed agent includes the aux heads as a
  product feature (limited-info score/win/trump insight) regardless of
  their training benefit. Their LEARNING contribution is still a real
  question — measured in the target regime (league aux arm, §1.3), not
  assumed from the shaped-self-play null.
- **P3 — The oracle-critic league regime is decisive.** Shaped self-play
  results are SCREENING evidence only: they pick contenders and catch
  gross failures. Adoption is settled by paired league arms under
  `--critic-mode oracle`.
- **P4 — A size sweep still happens**, redesigned to fit P3 (§1.4).
- **P5 — Writeup instrumentation** (secondary to strength, still
  required): every architecture comparison that might appear in the
  writeup uses the comparable-instrument set (checkpoint panels,
  equal-load throughput bench, param counts) — never the 300-deal
  anchored curves.

## 2.2 Standing measurement rules (learned the hard way — see §4.2)

1. **Endpoints = mean of the last THREE checkpoint panels** (e.g.
   350k/375k/400k), both modes, CRN-paired per seed.
   Single-final-checkpoint verdicts are exposed to ±0.1 endpoint churn
   (observed in BOTH archs' s2042 and full s42).
2. **Curve/slope claims only from checkpoint panels** (1000 deals/mode).
   The anchored 300-deal curves have produced three wrong calls.
3. **Fixed-budget endpoint deltas between archs with different learning
   speeds are budget artifacts until a differential-slope check clears
   them** (the perceiver lesson). This retro-applies to the banked ladder
   verdicts (transformer +0.124, informed-init +0.150, aux null, onehot
   tie): treat as instrument-limited screening reads, NOT settled facts.
   Anything load-bearing gets re-decided in the league regime.
4. **Never compare eps/s across differently-loaded runs**; use the paired
   act-bench + update-timing method.
5. **Measure standard checkpoints, never `final_<arch>.pt` from selfplay
   runs started before 2026-07-08.** Those finals carry one out-of-spec
   flush update (§4.6). Finals from runs started after the fix equal the
   last threshold update's weights. The league trainer never had a flush.

## 2.3 League stopping rule (pre-registered 2026-07-06) **[CURRENT]**

Once the winning architecture is chosen, further league generations are
gated per-generation instead of running an arbitrary episode budget. (The
extended-league orchestrator implements this; the manual procedure is
recorded here.) At the end of each generation `g` run BOTH instruments on
the gen-final checkpoint:

1. **Frozen longitudinal yardstick — PANEL-A** (membership and seed 42
   never change; this is the number comparable across the whole project):

   ```bash
   PYTHONPATH=. uv run python -m sheepshead.analysis.rigorous_eval \
       --candidates runs/<league_run>/finals/gen<g>.pt \
       --anchors final_pfsp_swish_ppo.pt <pfsp 15M> <pfsp 5M> <selfplay 100k> \
       --deals 1000 --seed 42 --partner-mode called \
       --out-csv runs/<league_run>/panel_gen<g>_called.csv
   # repeat with --partner-mode jd
   ```

2. **Moving local anchor — gen g vs gen g−1 head-to-head** (cheap, one
   pairing; this signal does NOT saturate as the agent surpasses the
   panel):

   ```bash
   PYTHONPATH=. uv run python -m sheepshead.analysis.rigorous_eval \
       --candidates runs/<league_run>/finals/gen<g>.pt \
       --anchors    runs/<league_run>/finals/gen<g-1>.pt \
       --deals 1000 --seed 42 --partner-mode called \
       --out-csv runs/<league_run>/h2h_gen<g>_called.csv
   # repeat with --partner-mode jd; report the both-modes mean edge
   ```

**Stopping rule (fixed in advance — do not adjust after seeing results):**
stop after generation `g` when BOTH hold for `g` and `g−1` (two
consecutive flat generations):

- (a) PANEL-A both-modes mean has not exceeded its previous best by
  ≥ 0.07 (the 1000-deal MDE), AND
- (b) the gen-vs-previous head-to-head edge is < +0.07.

Rationale for two consecutive: single-window plateau calls are unreliable
(onehot-ff sat flat 100k–150k, then jumped). Rationale for instrument (b):
panel edge vs fixed policies is bounded by the best-response value against
them and compresses as the agent approaches/passes parity with
`final_pfsp` — a flat panel slope past that point can be signal
saturation, not learning stagnation. The head-to-head anchor measures the
local improvement gradient directly and keeps sensitivity. Panel
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
(1) league snapshots are oracle-stripped automatically and
`ppo.load_agent` handles `critic_mode` — no special checkpoint handling;
(2) watch the oracle value loss in training logs as a HEALTH diagnostic
(an underfit oracle silently degrades to a noisy baseline — that shows up
in training stats, not in the panel). If the goal is to ATTRIBUTE a gain
to the oracle critic (vs merely benefiting from it), that requires a
paired arm (oracle vs limited, same seed, CRN anchored evals); a single
long oracle run + this stopping rule measures the combined result but
cannot isolate the cause.

**Confirmatory eval (guard against stopping-rule selection bias):** the
stopping decision peeks at seed-42 deals repeatedly, so the FINAL reported
number for the chosen checkpoint must come from one fresh-deal run:
identical panel membership, `--seed 20260706`, 1000 deals, both modes.
Label it "confirmation" — the seed-42 run remains the longitudinal series.
Keep exploiter-gate edge (`exploitability.csv`) as the robustness check
neither instrument covers.

## 2.4 Known traps checklist

- **`checkpoint_` in a resume FILENAME parses as the absolute episode
  counter** (both `train_selfplay_ppo`, `train_league_ppo`, and the
  extended-league preflight). Resuming `<arch>_checkpoint_400000.pt` into
  a 1M generation silently shrinks it to 600k episodes. Warm starts that
  should count from zero must be RENAMED COPIES (e.g.
  `warmstart_<arch>_400k.pt`). This is deliberate trainer behavior for
  in-lineage resumes — the trap is only when re-basing a checkpoint as a
  new run's start.
- **`final_<arch>.pt` from selfplay runs started before 2026-07-08 are
  measurement-deprecated** (flush update, rule 5, §4.6). Always measure
  `*_checkpoint_N.pt`.
- **Never compare eps/s across differently-loaded runs** (rule 4). The
  probe-era "perceiver is 1.7× slower" was a machine-load confound;
  equal-load benches showed perceiver's encoder is actually the fastest.
- **Orchestrator prefix skip-guard:** `run_ablation_matrix` skips any
  (arch, seed) whose run dir for the given `--prefix` is already done.
  Re-running an arch under a NEW regime (e.g. watchdog-on) requires a
  fresh prefix (`--prefix sweep`), or the completed old-regime runs get
  silently reused.
- **Status JSONs saying "running" do not block a rerun** — skip logic
  only honors "done"; stale "running" files are harmless but confusing.
- **Always load checkpoints via `ppo.load_agent(path)`** — never
  construct `PPOAgent` with a guessed arch. Checkpoints record their
  architecture; legacy ckpts without the key load as `full`.
- **A league generation that dies MID-RUN restarts from its `--resume`
  point** (the generation start) under the runner scripts. To salvage
  partial progress instead, edit the arm script's `--resume` to the
  newest `checkpoints/pfsp_<arch>_checkpoint_*.pt` and reduce
  `--main-episodes` accordingly.
- **Python:** repo pins 3.14 and uses PEP 758 syntax — verify scripts
  with `uv run python` / `.venv/bin/python`, never the system python3.

---

# 3. REFERENCE

## 3.1 Environment & code lineage

- Machine: Apple M1 Max (10 cores), 64 GB, macOS (Darwin 23.6.0)
- Python 3.14 (`uv` venv), torch 2.11.0, CPU only
- Code: commit `94b7ab4` lineage — registry `3e89023`, trainer plumbing
  `da60615`, pooled-memory rung `978b244`, activation removal `b43f908`;
  tokenread `a019b9f`, perceiver `365a7de`, decomposition arms `885da0f`,
  perceiver-shared `e56825e`, rank audit `eab809c`. (Post-reorg the
  modules live under `sheepshead/`; all commands here use the `-m
  sheepshead.*` forms.)
- Matrix launched 2026-07-04; stage 1 launched 2026-07-11.
- Concurrency model: 8 simultaneous training subprocesses, 1 BLAS thread
  each (game logic is Python-bound so process parallelism wins); a
  finished slot starts the next queued job automatically.

## 3.2 Architecture registry (`architectures.py`)

**The original ladder** (each adjacent rung removes exactly one addition;
`full-uninformed` is a factorial arm testing informed init in the presence
of the transformer):

| arch | what it is | params (E+A+C) |
|---|---|---|
| `full` | CardReasoningEncoder + pointer actor + aux critic | 1,003,607 |
| `full-uninformed` | full + `use_informed_init=False` | 1,003,607 |
| `no-aux` | full, critic aux heads removed | — |
| `no-transformer` | PooledMemoryEncoder (0 reasoning layers, pooled memory) | — |
| `no-transformer-uninformed` | + uninformed init | — |
| `onehot-ff` | flat one-hot → MLP encoder + flat linear heads (largest net) | 1,130,000 (~1.13M) |

Adjacent deltas: full−no-aux = aux heads; no-aux−no-transformer =
transformer; no-transformer−…-uninformed = informed init;
…-uninformed−onehot-ff = card-token pipeline; full−full-uninformed =
informed init under the transformer. Ladder confounds: `onehot-ff` also
swaps the actor (flat heads vs pointer) and is not param-matched.

**Probe / redesign architectures:**

| arch | design | params |
|---|---|---|
| `full-tokenread` | full + additive cross-attention readout in the actor: 4 learned queries × 4 heads over all 19 post-reasoning tokens, flattened → 256, fused (`Linear(512→256)+SiLU`) with the adapted trunk before every head; also attends the post-reasoning memory token (which base full discards). Encoder standard outputs byte-identical to full (test-pinned) | 1,217,623 |
| `perceiver` | token-centric end to end: pools + fused trunk DELETED. Actor = 4q×4h readout over 19 tokens → 256 → standard adapter/heads (ignores trunk features, test-pinned). Critic = own independent 4q readout → deep value trunk, NO aux. Memory GRU driven by the post-reasoning MEMORY token (not context) | 873,678 |
| `perceiver-shared` (v1) | bag SCOPING alone: full's 4 bag pools + fusion → ONE shared 4q/4h readout over all 19 tokens; trunk sharing, aux forcing, pointer actor, context-token memory driver all KEPT | 903,063 |
| `perceiver-shared-v2` | v1 with 16 readout queries × 4 heads (64 attention distributions = channel parity with full) + Linear+LayerNorm projection (full's feature_proj convention). Context-token memory driver kept (operator decision 07-09; `memory_token_driver` encoder kwarg exists, default False) | 1,100,951 |
| `perceiver-shared-v2-noaux` | v2 + `_no_aux_critic` — the missing cell of the {pools\|shared}×{aux\|noaux} factorial | — |

**Decomposition arms** (registered + tested, commit 885da0f; each flips
ONE switch off the perceiver delta; readout hybrids 974,222 params,
ctxmem 873,678, perceiver-aux 985,751):

| arch | change isolated | CRN comparator |
|---|---|---|
| `readout-actor` | actor trunk: pooled fusion → token readout | `ablate_no-aux_s*` |
| `readout-critic` | critic trunk: pooled fusion → token readout | `ablate_no-aux_s*` |
| `perceiver-ctxmem` | memory-GRU driver: memory token → context token | `ablate_perceiver_s*` |
| `perceiver-aux` | aux stack restored on the perceiver critic readout | `ablate_perceiver_s*` |

**Size variants** (one-knob around `full`, factory
`_full_size_variant`; actor/critic widths follow the encoder's `d_model`,
default bit-identical, commit `4cd5505`):

| arch | knob | params (E+A+C) |
|---|---|---|
| `full-dmodel128` | trunk/memory/pool width /2 | 470,519 |
| `full-dtok32` | transformer width /2 | 783,031 |
| `full-layers2` | depth /2 | 936,663 |
| `full` (center) | — | 1,003,607 |
| `full-layers6` | depth ×1.5 | 1,070,551 |
| `full-dtok128` | transformer width ×2 | 1,739,671 |
| `full-dmodel512` | trunk/memory/pool width ×2 | 2,929,943 |

Perceiver-based one-knob variants are also registered
(`_perceiver_size_variant`, params 412k–2.48M), plus attention-shape
knobs `perceiver-readq{2,8}` (807,886 / 1,005,262) and
`perceiver-readheads{2,8}` / `perceiver-rheads{2,8}` (exactly
param-matched to base perceiver — MHA params don't depend on num_heads,
so any delta is pure structure). Shared-encoder (v2-based) variants DO
NOT exist yet (§1.4).

**Encoder structural facts relevant to the readout question**
(encoder.py, for the record):

- After transformer reasoning over all 19 tokens (context, memory, 8
  hand, 5 trick, 2 blind, 2 bury; d_token 64), each bag is compressed by
  an `AttentionPool` with 4 learned queries × 4 heads (hardcoded,
  encoder.py:92-93): hand → 64 dims, trick → 64, blind → 32, bury → 32;
  concat with the 64-dim context token → 256 trunk features. Everything
  the pick/partner/call heads and the critic see passes through this.
- The recurrent memory is tighter still: `memory_gru` input is the 64-dim
  context token alone (encoder.py:615) — all cross-trick history squeezes
  through one token. The post-reasoning MEMORY token is computed and
  discarded by `full` (tokenread/perceiver read or feed it instead).
- The only unmediated token access in `full` is the pointer head
  (play/bury/under scores see post-reasoning hand tokens directly,
  ppo.py:129), but its situation-conditioning `Wg(feat)` comes from the
  pooled trunk.
- Attention-channel accounting: full = 4 pools × 4q × 4h = 64 scoped
  distributions (softmax over ≤8 bag tokens each, no cross-bag
  competition, guaranteed output bandwidth); v1-shared = 16 distributions
  over all 19 competing tokens (defeated the anti-squeeze goal — hence
  v2's 16 queries).

**Oracle critic** (`--critic-mode oracle`, CTDE): rebuilt perceiver-style
2026-07-06 (operator decision, unconditional — no oracle checkpoints
existed). `OracleCriticEncoder` has no pools/fusion (deleted five pools
incl. `pool_opp` which squeezed 32 opponent-hand token slots into 64
dims); `OracleValueNetwork` reads all 51 post-reasoning tokens through its
own 4q×4h MHA readout; memory GRU fed the post-reasoning MEMORY token.
622,473 params (was 772,745). All 15 oracle tests + 4 readout invariants
pass. **There are NO pooled-oracle checkpoints anywhere.**

## 3.3 Instruments & yardsticks

| instrument | spec | role |
|---|---|---|
| **PANEL-A gauntlet** | `sheepshead.analysis.rigorous_eval`, frozen panel (`final_pfsp_swish_ppo.pt`, pfsp 15M, pfsp 5M, selfplay-100k), 1000 deals/mode, seed 42, CRN + deal bootstrap | THE longitudinal number; MDE ≈ 0.07 score/hand; per-measurement SE ≈ 0.04 |
| scripted probe | `sheepshead.analysis.scripted_probe`, 500 deals, seed 31 | sanity floor vs ScriptedAgent |
| trump-lead probe | `sheepshead.analysis.trump_lead_probe`, 2000 deals, seed 20260702 | defender trump-leak canary |
| anchored eval | in-trainer, 300 paired deals vs 3 yardsticks every 5k eps, `ANCHOR_EVAL_SEED = 20260703`; eval wall-clock excluded from train wall-clock | **motivation-only** (rule 2); 3 wrong slope calls on record |
| leaster scan | `sheepshead.analysis.leaster_scan [runs/glob ...]` | PASS-collapse spans from train logs |
| aux audit | `sheepshead.analysis.diagnostics.aux_audit`, 200 CRN self-play games/ckpt | aux-head prediction quality |
| squeeze audit | `sheepshead.analysis.diagnostics.readout_squeeze_audit`, 30 games/ckpt | readout channel redundancy/coverage |
| rank audit | `sheepshead.analysis.diagnostics.readout_rank_audit` | linear-map rank + feature covariance |
| panel matrix | `sheepshead.analysis.build_panel_matrix` | aggregates checkpoint panels → strength matrix CSVs |

Anchored-eval yardsticks: ScriptedAgent /
`runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt` /
`final_pfsp_swish_ppo.pt` (repo root). Seed convention: `--seed` fully
determines a selfplay run (init, sampling, deals); arms sharing a seed see
identical deal sequences (CRN across arms). Statistical convention:
deltas > 0.07 (MDE) AND > 2 seed-level SE = real; report per-seed spread
and collapse count (edge regressing > 2 SE from its own peak).

## 3.4 Standard command shapes

Self-play matrix (orchestrator; resumable — rerunning skips finished
jobs; `--smoke` = 5-minute pipeline check):

```bash
mkdir -p runs/ablation_202607
PYTHONPATH=. nohup .venv/bin/python -m sheepshead.training.run_ablation_matrix \
    > runs/ablation_202607/orchestrator.out 2>&1 &
# variants: --archs A B C --episodes N --out-dir runs/X --prefix P [--leaster-watchdog]
```

Per (arch, seed) job the orchestrator executes (exact argv also recorded
in `runs/<out>/status/<run>.json`):

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.training.train_selfplay_ppo \
    --arch <ARCH> --seed <SEED> --episodes 100000 \
    --run-name ablate_<ARCH>_s<SEED> \
    --anchor-eval-interval 5000 --anchor-eval-deals 300 \
    --save-interval 25000 --strategic-eval-interval 1000000
# then, on the run's final checkpoint:
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.scripted_probe \
    --ckpt <CKPT> --deals 500 --out-json <RUN>/scripted_probe.json
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.trump_lead_probe \
    --ckpt <CKPT> --deals 2000 --out-json <RUN>/trump_lead_probe.json
```

Extension resume (in place; same run dir, `anchored_eval.csv` continues;
entropy schedule re-anneals to the new horizon — identical treatment for
all extended arms of a comparison):

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.training.train_selfplay_ppo \
    --arch <A> --seed <S> --episodes <NEW_TOTAL> \
    --resume runs/<RUN>/<A>_checkpoint_<PREV>.pt \
    --run-name <RUN or NEW_RUN> \
    --anchor-eval-interval 5000 --anchor-eval-deals 300 \
    --save-interval 25000 --strategic-eval-interval 10000000
```

PANEL-A panel (any candidates; CRN ⇒ rows comparable across calls with
identical args; the instrument is byte-deterministic — verified 07-08):

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.analysis.rigorous_eval \
    --candidates <ckpt> [<ckpt> ...] \
    --anchors final_pfsp_swish_ppo.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt \
              runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt \
    --deals 1000 --partner-mode <called|jd> --seed 42 \
    --out-csv <out>.csv [--out-plot <out>.png]
```

Reports/aggregation: `sheepshead.analysis.aggregate_ablation` (CSVs +
plots + table over an out-dir); `sheepshead.analysis.ablation_report
--out-dir X --baseline <arch> [--extra-results r.csv ...]` (delta/slope
verdict tables; merges external result rows).

League arm + extended-league: see §1.2 / §1.3 (those command blocks are
the canonical shapes).

## 3.5 Output file map

Self-play matrix outputs (under the matrix `--out-dir`, e.g.
`runs/ablation_202607/`):

| file | contents |
|---|---|
| `learning_curves.csv` | long format, one row per (run, eval point): `run, arch, seed, episode, train_wall_s, eval_wall_s, updates_done, transitions_done, edge_<yard>, se_<yard>` for yard ∈ {scripted, selfplay100k, final_pfsp} — graphing source of truth |
| `results.csv` / `results_table.md` | one row per run: final edges, endpoint probes, PANEL-A scores, wall-clock, eps/s. **Caveat:** the original matrix's `panel_a_*` columns refer to the 100k finals; later budgets live in suffixed CSVs (`results_200k_panel.csv`, `panel_a_ext_*`, etc.) |
| `curves_{scripted,selfplay100k,final_pfsp}.png` | learning-curve plots (bold = per-arch seed mean, faint = seeds) |
| `panel_a_{called,jd}.csv/.png` | PANEL-A gauntlet over the finals |
| `orchestrator.log` | timestamped job lifecycle |
| `status/<run>.json` | per-run status, exact train argv, wall minutes |
| `runs/ablate_<arch>_s<seed>/` | per-run: `train.log`, `probes.log`, `anchored_eval.csv`, checkpoints every 25k, `final_<arch>.pt` (rule 5!), probe JSONs |

League arm outputs (under `runs/<run-name>/checkpoints/`):
`pfsp_<arch>_checkpoint_<N>.pt` every 50k, `exploitability.csv` (gate edge
per generation), `greedy_health.csv`, `anchored_eval.csv`.

Plotting from the long CSV:

```python
import pandas as pd
df = pd.read_csv("runs/ablation_202607/learning_curves.csv")
ax = None
for (arch,), g in df.groupby(["arch"]):
    m = g.groupby("episode")["edge_scripted"].mean()
    ax = m.plot(label=arch, ax=ax)
ax.axhline(0, ls="--", c="gray"); ax.legend(); ax.set_ylabel("edge vs scripted")
```

## 3.6 Key artifact locations (results referenced in §4/§5)

| artifact | location |
|---|---|
| 100k matrix results | `runs/ablation_202607/results_100k.csv`, `results_table_100k.md`, `learning_curves_100k.csv`, `curves_*_100k.png`, `panel_a_{called,jd}.csv` |
| 200k extension panels | `runs/ablation_202607/panel_a_ext_{called,jd}.csv`; corrected values `results_200k_panel.csv` |
| onehot 200k control | `runs/ablation_202607/panel_a_onehot200k_{called,jd}.csv`; ckpt-based `panel_a_onehot_ckpt200k_{called,jd}.csv` |
| tokenread probe | `runs/tokenread_202607/report.md` |
| perceiver probe | `runs/perceiver_202607/report.md` |
| perceiver diagnostics + all later panels | `runs/perceiver_202607/diag/` — 175k controls (`panel_a_full175k_*`), ckpt200k re-panels (`panel_a_ckpt200k_*`, `panel_a_shared_ckpt200k_*`), 300k mid-course (`panel_a_400ext_300k_*`), 400k endpoints (`panel_a_400ext_{350,375}k_*`, `panel_a_400k_*`), v2 rungs (`panel_v2_{150,175,200}k_*`), ext2 rungs (`panel_ext2_{300,350,375,400}k_*`), 150k rung (`panel_a_full_ckpt150k_*`), `aux_audit.csv` |
| perceiver-shared (v1) runs | `runs/decomp_202607b/` |
| 400k extensions | `runs/ablate_{full,perceiver,onehot-ff,perceiver-shared-v2}400_s{42,1042,2042}/` |
| stage 1 | `runs/stage1_202607/` + `runs/league_arch_{full,perceiver-shared-v2}/` |
| strength matrix | `notebooks/panel_a_strength_matrix{,_called,_jd}.csv` |
| repro-league baseline | `runs/rigorous_baseline_202607/` (league-13.65M headline: CA −0.120 / JD +0.038) |

## 3.7 Automation inventory (detached watchers; all restartable, all skip existing outputs)

Historical but listed so orphaned processes/logs make sense:

| script | job | done marker |
|---|---|---|
| `runs/size_sweep_202607/watch_and_launch.sh` | the probe→phase2→sweep chain (stages now superseded; do not relaunch without editing) | per-stage markers in `watcher.log` |
| `runs/perceiver_202607/diag/panel300k_watch.sh` | 300k mid-course panels for the 400k extensions | `MID-COURSE PANELS COMPLETE` |
| `runs/perceiver_202607/diag/panel400k_endpoint_watch.sh` | 350/375/400k endpoint panels (full + perceiver) | `ENDPOINT PANELS COMPLETE` |
| `runs/perceiver_202607/diag/ckpt200k_repanel.sh` | checkpoint-based 200k re-panels after the flush finding | chain end in its log |
| `runs/perceiver_202607/diag/panel_v2_endpoint_watch.sh` | v2 150/175/200k panels | `V2 ENDPOINT PANELS COMPLETE` |
| `runs/perceiver_202607/diag/panel_ext2_watch.sh` | ext2 (onehot + v2) 300–400k panels | `EXT2 PANELS COMPLETE` |
| `runs/stage1_202607/stage1.sh` | **[CURRENT]** stage-1 arms + endpoint instruments | `STAGE1 COMPLETE` |

---

# 4. SETTLED FINDINGS (by question)

## 4.1 Component ladder verdicts **[SCREENING — rule 3 applies]**

From the 100k matrix (18 runs) + 200k extensions, shaped self-play.
Adjacent-rung deltas (PANEL-A both-modes; + = component helps):

| component | comparison | delta @100k | verdict |
|---|---|---|---|
| informed embedding init (under transformer) | full − full-uninformed | **+0.150** (2 SE) | clearest single win; synergizes with attention |
| informed init (no transformer) | no-transformer − …-uninformed | +0.060 (noisy) | positive but weak — attention amplifies good card geometry |
| transformer reasoning | no-aux − no-transformer | +0.124 (~1.2 SE) | positive; transformer arms still climbing at cutoff |
| aux heads | full − no-aux | +0.007 @100k, +0.044 @200k | null under shaping (pre-registered caveat: dense shaping supplies the signal aux heads would — lower bound on terminal-regime value) |
| card-token pipeline | …-uninformed − onehot-ff | −0.218 @100k | see §4.3 — resolved in the token stack's favor at 400k |

Two standing contaminations of these numbers: (1) **escape-speed
confound** — every from-scratch arm spent its early budget in the
always-PASS collapse (§4.5), and slow-escaping archs (no-transformer
18–78% of budget, no-aux s1042 40%) have endpoints that partly measure
collapse time, so the transformer/informed-init deltas substantially
measure "escapes faster"; (2) **budget artifacts** (§4.2). Per rule 3,
anything load-bearing gets re-decided in the league regime.

Also settled at screening grade: no run of the original 18 collapsed
terminally (0/18 collapse count); wall-clock ranking onehot 1.6h <
no-transformer ~4h < full-family ~7.7h per 100k; `onehot-ff` is a
legitimate fast prototyping baseline (17 eps/s); `no-transformer` (pooled
memory) buys 2× wall-clock over full at a real strength cost.

**Reward-regime note (design decision, 07-04):** the matrix ran SHAPED
(the trainer's historical bootstrap stack: hand-conditioned pick/pass
nudges + per-trick intermediate rewards + final score/RETURN_SCALE with
leaster bonus). Deliberate: identical shaping across arms keeps the
comparison controlled; it matches the regime in which the components were
historically adopted; cold-start terminal-only at 100k risks a floor
effect. Pre-registered caveat: shaped-regime deltas for aux/critic rungs
are lower bounds on their terminal-regime value.

## 4.2 Methodology lessons: budget artifacts & slope instruments **[SETTLED]**

The week's central methodological finding, now codified as rules 1–3:

- **Budget artifacts are the default, not the exception.** Onehot-ff
  "tied" full at 100k and 200k, then lost decisively at 400k (its own
  200k→400k gain was +0.023 vs full's +0.149 — the tiny net hit its
  ceiling). Perceiver's 175k→200k differential climb (+0.159 vs full's
  +0.021 per 25k) justified a 400k extension — where the differential
  REVERSED (perceiver +0.005/100k, full +0.071/100k over 200k→300k) and
  full won decisively. Fixed-budget deltas between archs with different
  learning speeds mean little until a budget-equal converged read.
- **The 300-deal anchored curves produced three wrong slope calls**:
  (1) onehot's "flat plateau" at 100k (it jumped 150k→200k);
  (2) perceiver's "last-20k slope ≈ 0" at 200k (checkpoint panels showed
  2/3 seeds gaining ~+0.25 in the final 25k); (3) the general
  175k-vs-200k snapshot reads on perceiver-shared (trajectory volatility
  ±0.15–0.2/25k made single rungs meaningless). Only checkpoint panels
  (1000 deals/mode) can read curves.
- **Endpoint churn is regime-wide**: ±0.1 at single checkpoints (both
  archs' s2042 regressed late; perceiver s42 swung +0.23 in one 25k
  rung). Hence rule 1's last-3-checkpoint endpoints.
- `rigorous_eval` itself is byte-deterministic (a finals re-check
  reproduced an earlier panel EXACTLY to 4 decimals) — instrument noise
  was eliminated as an explanation early.

## 4.3 onehot-ff: fast start, real ceiling — OUT at 400k **[DECISIVE for screening]**

Arc: at 100k it matched `full` on every endpoint while training 5× faster
wall-clock (fast-start/early-plateau vs slow-start/still-climbing). The
200k budget-equal control kept the tie (−0.247 vs −0.233 finals-based;
−0.228 vs −0.214 on the clean checkpoint methodology, CRN per-seed deltas
−0.023/−0.064/+0.045 — mixed signs, tie stands). At 400k
(ext2, rule-1 endpoints): **onehot −0.205 vs full −0.065, Δ −0.140 ±
0.020 (t = −6.99, 3/3 seeds)** — the token-stack advantage IS demonstrated
in shaped self-play at converged budget; the earlier ties were budget
artifacts. The onehot league arm (`phase2_onehot.sh`) is optional writeup
context now, not a decisive missing experiment.

## 4.4 Perceiver family: probe loss → mechanism audits → v2 qualified **[DECISIVE for screening]**

The full arc, condensed (details §5.4–§5.9):

1. **full-tokenread probe (additive readout, actor only): NULL.**
   +0.043 vs full, inside the ±0.07 band, ~2× wall-clock cost. One-sided
   probe: a win would have proven the pooling bottleneck; a null is
   ambiguous. Post-hoc: tokenread beat onehot in all 3 seeds (+0.057 ±
   0.023) — first consistent token-arch separation, suggestive only.
2. **perceiver probe (pools deleted end-to-end): LOSS.** −0.187 vs full
   at 200k; NOT entropy lock-in (all seeds escaped the PASS trap with
   150–180k to spare, healthy endpoint entropies); at 400k budget-equal
   the loss was confirmed decisive: **−0.259 ± 0.054 (t = −4.8, 3/3
   seeds)** → plain perceiver OUT.
3. **Mechanism audits converge**: the deficit is
   **optimization/priors, NOT representational capacity** —
   - readout attention healthy (trick-token mass 0.32–0.39 vs 0.24
     uniform, no dead queries);
   - critic-baseline proxy comparable (adv-std/target-std 0.69–0.74 vs
     full 0.68–0.70);
   - aux-head prediction quality EQUAL across archs at matched budget
     (§4.8) — information present and decodable in both trunks; full
     converts it into better policy;
   - LayerNorm audit: all consumers have their own input LN; v1-shared's
     bare-Linear readout_proj only changes gradient geometry (consumer
     LNs divide grads by wandering feature std vs full's pinned 1.0) and
     lets raw scale drift (2.7→3.3 over 100k) — real but second-order;
   - channel accounting: v1-shared had 16 attention distributions
     competing over 19 tokens vs full's 64 scoped ones — the
     anti-squeeze goal was defeated by the 4q/4h choice;
   - squeeze audit: v1's 16 distributions highly redundant (mean pairwise
     cosine 0.61–0.70, coverage 0.53) — readout bandwidth binds;
   - rank audit: NO rank bottleneck anywhere — v2's dense 1024→256 map
     eff. rank ~172, σ_min 0.31; full's pools∘fusion composite eff. rank
     ~119, σ_min ≈ 0 (the MORE rank-deficient map wins anyway);
     on-distribution participation ratio ~10–13 of 256 dims for BOTH
     archs (also deflates the dmodel512 width hypothesis).
4. **perceiver-shared v1** (scoping change only, aux kept): trailed full
   by ~0.16 at 200k (ckpt-based −0.373 vs −0.214); three-rung matrix
   (150/175/200k means): full −0.269, v1 −0.337, v2 −0.347 — v2 ≈ v1 at
   200k (Δ −0.009; the 16q+LN corrections had no effect at that budget —
   bandwidth was not binding *there*).
5. **perceiver-shared-v2 at 400k (ext2): QUALIFIED.** Rule-1 endpoint
   −0.115 vs full −0.065 — Δ −0.050 ± 0.022 (t = −2.27, per-seed +0.002 /
   −0.061 / −0.092): within the 0.07 band, real but sub-MDE, not decisive
   under P1. v2's own 200k→400k gain was +0.200, out-slope-ing full's
   +0.149 — the shared-readout family kept converging where plain
   perceiver stalled (−0.324 at 400k). → stage-1 league pair full vs v2;
   league regime decides adoption (§1.2).

## 4.5 The always-PASS collapse is UNIVERSAL in from-scratch self-play **[SETTLED]**

Leaster-rate scan (07-06) of all 24 from-scratch shaped-reward self-play
runs: **every one collapsed to ~100% leaster (all seats always PASS)
within the first ~3–4k episodes — entry is universal; architectures
differ ONLY in escape time.** Hard-collapse spans (leaster ≥ 90%):

| arch | s42 | s1042 | s2042 | worst % of budget |
|---|---|---|---|---|
| onehot-ff | 3k-5k | 3k-3k | 3k-3k | ~1% |
| full | 3k-7k | 3k-10k | 3k-10k | ~4% |
| full-uninformed | 3k-10k | 3k-11k | 4k-8k | ~9% |
| no-aux | 4k-12k | 3k-80k | 4k-20k | **39%** |
| no-transformer | 3k-78k | 3k-18k | 3k-67k | **76%** |
| no-transformer-uninformed | 3k-30k | 3k-61k | 2k-31k | **59%** |
| full-tokenread | 3k-32k | 3k-178k | 3k-13k | **88%** |
| perceiver | 3k-16k | 4k-51k | 3k-23k | ~26% |
| perceiver-shared v1 | esc. 3k | esc. 6k | esc. 20k | (fast escapes) |
| perceiver-shared-v2 | — | — | — | all escaped by ~29k (rolling leaster 3.4% / 18.5% / 4.0%) |

Mechanism (consistent with every trace): at init play skill is terrible,
so picking has strongly negative EV → PASS dominates for every seat →
all-leaster equilibrium with bidding entropy frozen at ~0. Escape appears
driven by rare exploratory PICKs finally paying off once play skill
learned *inside leasters* transfers — which explains why
fast-learning-per-episode archs (onehot) escape almost immediately and
slow/unstable ones (tokenread, no-transformer) linger. tokenread s1042
was the extreme case: pick/partner/bury entropies exactly 0.000 by ep
~1.4k (PPO KL 0.0000 — heads frozen), 88% of the run in the equilibrium,
spontaneous escape at ~180k. Same attractor as the documented ExIt
warm-start collapse. onehot's 100–150k flat-then-jump was NOT leaster (it
escaped at 5k) — separate phenomenon.

**Fix (implemented 07-06, operator decision): the selective entropy
kick** — `--leaster-watchdog` on `train_selfplay_ppo`, **default OFF**
(no completed run used it; enable explicitly on future from-scratch
runs, and if used in a comparison, enable it for ALL arms).
`LeasterWatchdog`: when the rolling 3000-episode leaster rate crosses 90%
it multiplies the pick head's *scheduled* entropy coefficient ×10 each
update until the rate falls below 30% (hysteresis), then normal annealing
resumes. It must fire at the crossing because the entropy bonus's
gradient vanishes as the head approaches determinism — the kick lands
while pick entropy is still alive (~ep 3k), holding a probability floor
so the deep freeze never forms. Rewards untouched. Survey: NO other
adaptive mechanism exists in either trainer (open-loop per-head entropy
schedules from config.py only; old controllers/epsilon-floors retired).
Engage/release events print to train.log (grep "watchdog"). Alternatives
considered: (b) constant per-head entropy floor; (c) bidding-warmup
curriculum (ε-mixture pick/pass, bidding heads frozen); (d) early
scripted-opponent mixing; (e) oracle critic (shortens escapes via
lower-variance PICK evaluation but won't prevent entry).

**The league/warm-start production regime never enters this attractor —
it is a from-scratch probe artifact.** Existing completed runs stay
comparable (all collapsed, same regime).

## 4.6 Flush-update defect (found 07-08, fixed same day) **[SETTLED]**

`train_selfplay_ppo` ended every run with a bare `agent.update()` "flush"
before saving `final_<arch>.pt` — out-of-spec on two axes vs every
in-loop update (`epochs=4, batch_size=256` at a 2048-transition
threshold): **epochs=6** (ppo.py default; 50% more replay passes) and an
**arbitrary partial buffer** (observed 715 leftover transitions in a
smoke) so advantage normalization ran over a small unrepresentative
sample. PPO's clip only bounds ratios ON the sampled states;
encoder/trunk drift generalizes off-buffer. Operator ruling: updates must
always run under the specified hyperparameters — unsound regardless of
measured damage.

Fix: flush REMOVED (leftover events discarded with a log line);
`final_<arch>.pt` now equals the last threshold update's weights. League
trainer and exploiter audited — never had a flush (single update site,
always epochs=4/bs=256); stale comment in `exploiter.latest_checkpoint`
corrected. Verified by 12-episode smoke ("Discarding 715 leftover
buffered events (no flush update)").

Measured impact (RESOLUTION, §5.7): flush deltas mixed-sign and mostly
small (−0.008 / +0.045 / −0.117 across shared seeds; mean −0.03;
consistent with the 30M `final_swish.pt` analog's KL≈0). A single flush
CAN perturb at panel-noise scale (the −0.117) — reason enough for
removal — but there was no systematic damage; the alarming "175k→200k
regression" that triggered the investigation was genuine
checkpoint-level trajectory VOLATILITY, not the flush. Consequence =
standing rule 5. Live lineage stayed clean (all 400k extensions resumed
from `*_checkpoint_200000.pt`, verified via ps).

## 4.7 Oracle critic facts **[SETTLED]**

- Design: see §3.2 (perceiver-style, 622,473 params, no pooled-oracle
  checkpoints exist anywhere).
- **Cost** (paired update() bench on identical 7.2k-event buffers):
  oracle update ≈ **2.7×** limited; rollout unchanged (values fill
  batched at update time); end-to-end league estimate ~1.5–2×. Cheapest
  knob if needed: train the oracle loss on 1 of 4 epochs (its target is
  near-deterministic).
- **Warm-start shock sim** (200k full ckpt + random oracle, 6
  collect→update cycles, terminal reward): **mild** — a random oracle
  acts as a near-zero-constant baseline (≈ REINFORCE with λ-returns), not
  an adversarial one; KL fell 0.013→0.005, oracle value loss 0.40→0.02
  within 2 updates, ev_oracle ≥ ev_limited from the first update.
  Notable: `ev_limited` is ALSO poor at league warm start (−2.7 → −0.11
  over 6 updates) because the shaped→terminal return mismatch forces the
  limited critic to re-fit at every league gen 1 — the gen-1 bidding
  anchor already covers this window. **Verdict: no burn-in mechanism
  needed**; an ev-gated policy freeze would release immediately. (If
  belt-and-suspenders is ever wanted: a fixed ~10–20-update policy
  freeze is the right shape.)
- Health check for any oracle run: watch oracle value loss / explained
  variance (`ev_oracle` in update stats) — full information makes the
  target nearly deterministic given the policies, so persistent underfit
  means the VALUE net (not the regime) needs attention. Stage 1 confirms
  healthy (`ev_oracle` > `ev_limited` from day 1).

## 4.8 Aux heads: current state of knowledge

- Ship regardless (P2 — product feature).
- Training contribution under SHAPING: null (+0.007 @100k, +0.044 @200k,
  within seed noise) — pre-registered as a lower bound on terminal-regime
  value since dense shaping supplies the signal aux heads would.
- **Aux-head accuracy audit (07-08,
  `sheepshead/analysis/diagnostics/aux_audit.py`; 200 CRN games/ckpt,
  trainer-identical labels; results `runs/perceiver_202607/diag/aux_audit.csv`):**

  | metric | full@100k | shared@100k | full@200k | shared@200k | full@300k |
  |---|---|---|---|---|---|
  | partner AUC | 0.972 | 0.994 | 0.998 | 0.999 | 0.999 |
  | win AUC | 0.812 | 0.788 | 0.824 | 0.817 | 0.822 |
  | return corr | 0.540 | 0.536 | 0.591 | 0.578 | 0.584 |
  | points MAE (pts) | 2.37 | 1.87 | 0.93 | 0.97 | 0.87 |
  | seen-trump acc | 0.789 | 0.791 | 0.825 | 0.831 | 0.842 |
  | unseen-higher acc | 0.944 | 0.932 | 0.981 | 0.986 | 0.988 |

  Two findings, both AGAINST the "aux bearing fruit" hypothesis in its
  simple form: (1) aux quality is EQUAL across archs at matched budget
  (shared even slightly better at 100k on partner/points) — prediction
  quality cannot explain full's strength edge; the information is present
  and decodable in both trunks, full converts it into better policy
  (points at optimization/gradient-shaping/capacity-allocation, not
  missing information); (2) NO inflection at 200k→300k in full — every
  metric already saturated by 200k, so whatever resumed full's climb, it
  was not newly-reliable aux predictions.
- Caveat: equal accuracy does not imply equal GRADIENT value (aux losses
  could shape actor-visible features differently in the two
  constructions) — that residual question is exactly what the league aux
  arm (v2 vs v2-noaux, §1.3) measures.
- Trump-lead canary log (the documented defender trump-leak surfacing in
  self-play arms): full s42 called 10.6% at 200k (other seeds 0–1.8%);
  tokenread s2042 20.9% jd / 17.6% called at 200k; 100k matrix scattered
  0–6%. Track into the league arms (stage-1 probes cover it).

---

# 5. CHRONOLOGICAL LOG (full detail)

## 5.1 2026-07-04 — the 100k matrix (18 runs)

Launched morning; **complete 19:40 — 18/18 ok, 0 failures, 19.8h wall**
(incl. PANEL-A over all finals, 136 min/mode). No collapse in any run.
100k artifacts snapshotted as `*_100k.*`.

Per-arch endpoint summary (mean ± seed-std over 3 seeds; per-measurement
SE ≈ 0.04 but seed-to-seed spread dominates — the Henderson-2018 point
the 3-seed design anticipated):

| arch | PANEL-A called | PANEL-A jd | both | scripted edge (500) | train h/run | eps/s |
|---|---|---|---|---|---|---|
| `full` | −0.431 ± 0.071 | −0.394 ± 0.156 | **−0.413** | −0.097 ± 0.098 | 7.9 | 3.5 |
| `full-uninformed` | −0.566 ± 0.103 | −0.559 ± 0.063 | −0.562 | −0.249 ± 0.117 | 7.8 | 3.6 |
| `no-aux` | −0.463 ± 0.107 | −0.378 ± 0.052 | **−0.420** | −0.288 ± 0.064 | 7.4 | 3.8 |
| `no-transformer` | −0.511 ± 0.150 | −0.577 ± 0.184 | −0.544 | −0.295 ± 0.347 | 4.1 | 6.8 |
| `no-transformer-uninformed` | −0.654 ± 0.167 | −0.555 ± 0.159 | −0.605 | −0.461 ± 0.269 | 3.9 | 7.1 |
| `onehot-ff` | −0.423 ± 0.128 | −0.351 ± 0.070 | **−0.387** | −0.065 ± 0.113 | 1.6 | 17.4 |

**The onehot-ff surprise:** matched `full` on every endpoint while
training 5× faster wall-clock. Curve shapes (edge vs scripted, seed
means): onehot −0.29 → −0.34 → −0.23 at 20k/60k/100k (80k→100k slope
+0.10, < 1 SE); full −0.49 → −0.38 → −0.24 (+0.24, > 2 SE); no-aux −0.45
→ −0.30 → −0.21 (+0.31, > 2 SE). Read at the time as fast-start/
early-plateau (wide first layer fits the shaped bootstrap signal fastest)
vs slow-start/still-climbing — "a horizon statement, not a ceiling
statement." **[The plateau call was WRONG — §5.3; the ceiling call was
eventually RIGHT — §5.9.]**

**Extension rule applied (pre-registered: top-2 arms +100k if last-20k
slope > 1 SE above zero).** Slopes (edge vs selfplay-100k, 80k→100k, seed
mean): full **+0.236** (fires), no-aux **+0.308** (fires),
full-uninformed +0.156 (fires), no-transformer-uninformed +0.148 (fires),
no-transformer +0.020 (no), onehot-ff +0.098 (no). Top-2 still-climbing =
**full** and **no-aux** → resumed to 200k, all seeds (extension command
shape in §3.4; `--strategic-eval-interval 10000000`).

## 5.2 2026-07-05 — 200k extension results + conclusions

Extension complete 05:20 (6/6 ok, ~9.6h). Panels in
`panel_a_ext_{called,jd}.csv`.

| arch | called | jd | both | vs own 100k | vs onehot-ff@100k (−0.387) |
|---|---|---|---|---|---|
| `full` | −0.249 ± 0.047 | −0.216 ± 0.039 | **−0.233** | +0.180 | +0.154 |
| `no-aux` | −0.270 ± 0.091 | −0.284 ± 0.121 | **−0.277** | +0.143 | +0.110 |

Trajectory (edge vs selfplay-100k, seed means, 100k/150k/200k): full
−0.13 → −0.03 → **+0.19**; no-aux −0.15 → −0.08 → **+0.08** — both
crossed positive (beating their 100k-lineage ancestor head-to-head),
still climbing at 200k (180k→200k slope +0.13 / +0.19). Scripted
endpoints crossed zero for full (+0.10/−0.24/+0.24 per seed). Canary:
full s42 called-mode trump-lead 10.6% at 200k.

**[SUPERSEDED]** The "both gaps over onehot's 100k endpoint exceed the
MDE" comparison was budget-UNEQUAL (onehot wasn't extended); the same-day
onehot control (§5.3) overturned it.

**Conclusions as amended 07-05** (the numbered list the experiment closed
with; items 1/6 later superseded again by the 400k reads):

1. At equal 200k budget onehot-ff (−0.247) is statistically
   indistinguishable from full (−0.233) — the original "token stack
   blows past onehot" was a budget artifact. Token-stack ceiling
   advantage NOT demonstrated at ≤ 200k shaped self-play.
   **[Later demonstrated at 400k — §5.9.]**
2. Informed embedding init is the clearest single win (+0.150 under the
   transformer, 2 SE) and synergizes with attention (+0.060 without).
   Keep it.
3. Transformer reasoning: +0.124 at 100k and growing. Keep it.
4. Aux heads: no measurable effect in the shaped self-play regime;
   lower bound per the shaping caveat; terminal league regime is where
   their dense-signal role should show, if anywhere.
5. Practical: onehot-ff is a legitimate fast prototyping baseline
   (17 eps/s); no-transformer buys 2× wall-clock at a real strength cost.
6. Next: league arms to settle the aux question and the true ceiling.
   **[Became phase 2 → displaced by stage 1.]**

## 5.3 2026-07-05 — onehot 200k control; phase-2 launch & pause

**Onehot control completed 18:36** (3 resumes 100k→200k, runner
`runs/ablation_202607/onehot_control_200k.sh`). PANEL-A both-modes per
seed: s42 −0.251, s1042 −0.307, s2042 −0.181 → **mean −0.247 ± 0.063** —
within ~1 seed-SE of full (−0.233) and no-aux (−0.277). The "flat
plateau" inference from onehot's 80k→100k slope was wrong: its curve sat
flat 100k→150k (edge vs scripted −0.23 → −0.23) then moved sharply
150k→200k (→ −0.09; +0.14 PANEL-A overall). Slope-based plateau calls on
20k windows are unreliable — first of the slope-instrument failures
(§4.2). Follow-up queued same day: an onehot league arm (Appendix A.1,
never launched, eventually obsoleted).

**Phase 2** (full vs no-aux league arms, 2×750k each — full plan in
Appendix A.1) launched, then **paused at ~129k gen-1 episodes at 22:00**
(operator prioritized the tokenread probe; attempt archived at
`runs/phase2_202607/aborted_gen1_20260705/`, feeds nothing). Restart was
chained behind the probes via the watcher
(`runs/size_sweep_202607/watch_and_launch.sh`: tokenread → perceiver →
phase2 → phase2_onehot → capacity sweep); the phase-2 arms were
eventually killed 07-07 and superseded by stage 1.

## 5.4 2026-07-05/06 — full-tokenread probe

**Launched 07-05 21:32** (priority call; design and params in §3.2).
What it tests: whether the per-bag attention-pool squeeze between the
token stack and the heads is what holds full at onehot's level — readout
*structure* at fixed trunk width (the sweep's dmodel512 tests *width*;
together they triangulate). Protocol: 3 seeds × 200k shaped self-play,
CRN-paired with full@200k. Pre-registered: > +0.07 and > 2 SE ⇒ pooling
bottleneck real; ±0.07 ⇒ structure not binding here; < −0.07 ⇒ extra
actor capacity hurts at this budget.

**Epistemics amendment (07-06, before results):** the probe is
ONE-SIDED for the Perceiver-IO redesign question — a win is strong
evidence; a tie/loss is ambiguous between four readings (pools fine /
dual-path optimization pathology / critic still trunk-bound — GAE
advantages from pooled features can't resolve distinctions the actor
could represent / regime doesn't reward it). In-flight evidence for the
pathology reading: s1042's pick/partner/bury entropies hit exactly 0.000
with its scripted edge flat at −0.45..−0.49 from 50k→150k while full on
the same seed/deals recovered −0.75 → −0.06. (Design note for any future
additive readout: zero-init the readout projection so the new path
starts inert.)

**Landed 07-06 (matrix 17.2h, 3/3 ok). Verdict: NULL** — from
`runs/tokenread_202607/report.md`:

| arch | called | jd | both | eps/s |
|---|---|---|---|---|
| full | −0.249 ± 0.047 | −0.216 ± 0.039 | −0.233 | 7.6 |
| full-tokenread | −0.194 ± 0.048 | −0.185 ± 0.142 | **−0.189** | 3.9 |

Delta +0.043, seed-SE 0.059 — inside the null band, not > 2 SE, at ~2×
wall-clock cost. Per-seed (both-modes): s42 −0.159, s1042 −0.294, s2042
−0.115. Details:

- **s1042 = a full PASS-collapse (recovered late)**: entropies 0.000 by
  ep ~1.4k, all-leaster for ~88% of the run, spontaneous escape at ~180k
  (leaster 100%→1.9%), selfplay100k edge −0.71→−0.20 in the final
  window. Its endpoint is really "~20k episodes of non-leaster
  training". full on identical seed/deals did NOT collapse — evidence
  the dual-path readout destabilizes early bidding gradients. Counted as
  1 collapse per protocol; headline uses all three seeds as
  pre-registered.
- s2042 trump-lead canary hot: 20.9% jd / 17.6% called.
- Still-climbing flag: last-20k slope +0.201 ± 0.120 (> 1 SE) — endpoint
  may understate tokenread. No extension: the perceiver probe answers
  the underlying question more cleanly.

**Post-hoc contrast (suggestive only, not pre-registered):** tokenread
vs onehot at 200k, seed-paired: +0.093 / +0.014 / +0.066 → **+0.057 ±
0.023, same sign all seeds** — first consistent separation of any token
architecture over the dense FF net (plain full vs onehot remained a
wash). Its shape (tokens pay only when a head reads them directly, not
through pools) is exactly the perceiver thesis. Cost caveat:
evidence-per-compute still favored onehot.

## 5.5 2026-07-06 — perceiver probe launched; leaster scan; oracle rebuild

- **perceiver probe launched** (design/params §3.2; rationale: tokenread
  can't test the redesign's expressive potential, and after 07-07 nobody
  could build the clean architecture, so it was built and queued as
  stage 0.5 before the phase-2 restart). Protocol identical to tokenread
  (3×200k, CRN with full@200k). Pre-registered interpretation: win (>
  +0.07, > 2 SE) ⇒ adopt perceiver as base and rebase phase 2; tie ⇒
  don't adopt (capability-evidence bar) **[this tie bar was REVERSED
  07-07 by P1]**; loss (< −0.07) ⇒ keep full, classify failure as
  optimization (entropy lock-in à la tokenread s1042) vs representation
  (uniformly worse, healthy entropies) via
  `grep "Entropy - pick" runs/ablate_perceiver_s*/train.log`.
- **Leaster-rate scan** built and run → the universal PASS-collapse
  finding + watchdog fix (full detail in §4.5). Reinterpretation duties
  logged: 100k adjacent-rung deltas partly measure escape speed; no-aux
  s1042 lost 80k of 200k; perceiver s1042 lost its first ~51k but
  escaped with 145k to go.
- **Oracle critic rebuilt perceiver-style** (unconditional operator
  decision; details §4.7/§3.2). Cost + warm-start-shock measurements
  followed 07-07.
- **League stopping rule pre-registered** (now §2.3).
- Capacity-sweep caveat registered (operator): every knob in the
  full-based sweep is confounded by the pool squeeze — a depth null on
  full would not mean depth is useless, only that extra reasoning
  capacity doesn't survive the 4-query pools; if the perceiver family
  won, the sweep should rebase (Appendix A.2/A.3).

## 5.6 2026-07-07 — perceiver verdict; re-planning (P1–P5); decomposition → 400k extensions

**Perceiver probe landed 07:26 (15.4h, 3/3 ok). Verdict by the
pre-registered rule: UNDERPERFORMS (< −0.07 branch) — keep full as
base.** From `runs/perceiver_202607/report.md`:

| arch | called | jd | both | eps/s | train h |
|---|---|---|---|---|---|
| full | −0.249 ± 0.047 | −0.216 ± 0.039 | −0.233 | 7.6 | 7.3 |
| no-aux | −0.270 ± 0.091 | −0.284 ± 0.121 | −0.277 | 7.6 | 7.3 |
| perceiver | −0.402 ± 0.122 | −0.438 ± 0.237 | **−0.420** | 4.4 | 12.8 |

Delta vs full **−0.187, seed-SE 0.106** (1.8 SE; 2.7× MDE in the wrong
direction). CRN per-seed deltas: s42 −0.181, s1042 −0.003, s2042 −0.377 —
0/3 wins. Also loses to the aux-free comparator no-aux (−0.143).
Failure-mode classification (pre-registered check): **NOT entropy
lock-in** — all three seeds escaped the PASS trap with 150–180k to
recover, endpoint pick entropies alive (0.003–0.009); escape latency
doesn't explain the ranking (s1042 collapsed longest yet finished best;
s2042 escaped at 23k and finished worst, −0.62). High seed variance with
uniformly negative deltas ⇒ optimization instability / representation
under this budget. Caveats banked: watchdog-OFF, 200k shaped, n=3,
readout shape (4q×4h) never tuned. **[The 200k verdict stood only as a
fixed-budget statement — the 175k diagnostics below reopened it, and the
400k extension settled it (§5.8).]**

**Chain edit executed** (tie/loss branch of the watcher playbook): sweep
stage rewritten to watchdog-on full-based variants with fresh prefix
`sweep` (prefix trap, §2.4) and a pgrep wait-guard on phase2.sh (no
internal resume guards — double-launch risk). Both details preserved in
Appendix A.3.

**Loss decomposition (operator paused adoption to understand WHY).**
Diagnostics run the same morning (numbers now consolidated in §4.4):
readout-attention audit HEALTHY; critic-baseline proxy comparable;
**175k-checkpoint panels** — the pivotal one:

| seed | full 175k → 200k | perceiver 175k → 200k |
|---|---|---|
| s42 | −0.282 → −0.186 (+0.096) | −0.658 → −0.367 (+0.291) |
| s1042 | −0.328 → −0.270 (+0.057) | −0.498 → −0.273 (+0.225) |
| s2042 | −0.154 → −0.242 (−0.088) | −0.580 → −0.619 (−0.039) |
| mean | −0.255 → −0.233 (**+0.021**) | −0.579 → −0.420 (**+0.159**) |

The perceiver was climbing ~7× as fast at cutoff — the "last-20k slope
≈ 0" anchored-curve call was WRONG (third slope-instrument failure); the
budget-artifact reading was strongly supported. Also: the s2042
"endpoint cliff" was 300-deal noise (already bad at 175k). **Throughput
correction** (microbenchmark): the 4.4-vs-7.6 eps/s gap was mostly
machine load — equal-load per-decision act cost full 13.4 ms / tokenread
14.0 / perceiver 12.5 (perceiver's encoder fastest); perceiver's update
phase cheaper than full's (s42 sum 7.59h vs 8.59h) → rule 4. **Strongest
structural clue:** tokenread (readout ADDED, pooled trunk KEPT) ≈ full;
perceiver (pooled trunk DELETED both networks + memory driver changed)
lost 0.187 — the damage lives in the deletion, split across three
simultaneous changes → the decomposition arms (§3.2 table) were
registered, commit 885da0f, with pre-registered reads: readout-actor −
no-aux = actor-side cost; readout-critic − no-aux = critic-side cost;
both ≈ 0 with perceiver still low ⇒ interaction or memory driver;
perceiver-ctxmem − perceiver = memory-driver effect; perceiver-aux −
perceiver = aux-rescue effect; additivity check vs the observed −0.143
(perceiver − no-aux) — a large residual ⇒ the pooled trunk is
load-bearing as a UNIT (inductive bias). Launch/report commands:

```bash
PYTHONPATH=. nohup caffeinate -is .venv/bin/python -m sheepshead.training.run_ablation_matrix \
    --archs readout-actor readout-critic perceiver-ctxmem perceiver-aux \
    --episodes 200000 --out-dir runs/decomp_202607 --prefix ablate \
    >> runs/decomp_202607/orchestrator.out 2>&1 & disown
# report: -m sheepshead.analysis.ablation_report --out-dir runs/decomp_202607 \
#   --baseline no-aux --extra-results runs/ablation_202607/results_200k_panel.csv \
#   --extra-results runs/perceiver_202607/results.csv \
#   --extra-results runs/decomp_202607b/results.csv
```

**Displacement (~14:45):** the operator displaced the
readout-actor/readout-critic batch (~5.5h in, killed; partial data in
`runs/ablate_readout-*`; the orchestrator command restarts the matrix
fresh whenever wanted) with the **six 400k budget-equal extensions**
(`ablate_{perceiver,full}400_s{42,1042,2042}`, resumed from the 200k
checkpoints with optimizer state; "if perceiver is on track to surpass
full, *why it hasn't yet* is much less critical") — **plus
`perceiver-shared`** (3 seeds, `runs/decomp_202607b`, since 12:49 — the
adoption candidate: shared readout restores full-strength aux forcing;
per-network readouts reduce aux to indirect token shaping, operator's
observation). The watchdog-on `sweep_full_s*` baseline was killed ~12:45
(~3.8h in) to free slots. 300k mid-course panels automated
(`panel300k_watch.sh`); 400k endpoint read pre-registered: perceiver@400k
beating full@400k reverses the probe verdict; also compare own-arch
200k→400k gains (plateau confirmation for full). Extension resume
command shape:

```bash
for S in 42 1042 2042; do
  PYTHONPATH=. nohup .venv/bin/python -m sheepshead.training.train_selfplay_ppo \
    --arch perceiver --seed $S --episodes 400000 \
    --resume runs/ablate_perceiver_s$S/perceiver_checkpoint_200000.pt \
    --run-name ablate_perceiver400_s$S \
    --anchor-eval-interval 5000 --anchor-eval-deals 300 \
    --save-interval 25000 --strategic-eval-interval 4000000 \
    > runs/ablate_perceiver400_s$S.log 2>&1 &
done   # same for --arch full
```

(07-08 note: endpoint candidates repointed from `final_*.pt` to
`*_checkpoint_400000.pt` — the six extension processes predate the flush
fix, so their finals carry a flush update. Do not measure them.)

**Evening: the re-planning session** → PROGRAM STATE & DECISION TREE:
preferences P1–P5 and measurement rules 1–4 codified (§2.1/§2.2), stages
0/1/2 defined, phase 2 formally displaced. Loose-ends ledger opened
(now §1.6).

## 5.7 2026-07-08 — flush finding; RESOLUTION; 300k mid-course reversal; aux audit

- **Flush-update finding** (trigger: perceiver-shared's 175k→200k panel
  regression −0.232 → −0.404 with clean telemetry;
  `final_perceiver-shared.pt` ≠ `checkpoint_200000.pt` in every state
  dict). Defect, fix, and verdict in §4.6; rule 5 added. Re-panel chain
  `diag/ckpt200k_repanel.sh` queued with a pre-registered three-way
  split: ckpt200k ≈ 175k level + finals low ⇒ flush damage real;
  ckpt200k ALSO low ⇒ genuine late regression; finals re-check disagrees
  with original ⇒ eval anomaly (this branch was eliminated — the
  re-check reproduced EXACTLY to 4 decimals).
- **RESOLUTION (panels landed).** Called-mode, CRN-paired,
  perceiver-shared:

  | seed | 150k ckpt | 175k ckpt | 200k ckpt | 200k final | flush Δ |
  |---|---|---|---|---|---|
  | s42 | −0.458 | −0.295 | −0.277 | −0.285 | −0.008 |
  | s1042 | −0.472 | −0.357 | −0.500 | −0.455 | +0.045 |
  | s2042 | −0.325 | −0.147 | −0.354 | −0.470 | −0.117 |

  Verdict: **genuine checkpoint-level movement, flush mostly noise.**
  (1) The 175k→200k drop lives in the standard checkpoints for 2/3 seeds;
  combined with the 150k→175k jumps (+0.163/+0.178), the correct reading
  is trajectory VOLATILITY at ±0.15–0.2/25k — "all-seed regression" was
  an artifact of snapshotting a volatile curve at two points. Rule 1's
  raison d'être; rule-1 endpoints (called): s42 −0.343, s1042 −0.443,
  s2042 −0.275, mean ≈ −0.354. (2) Flush deltas mixed-sign, mean −0.03
  (§4.6). (3) Shared's 150k→200k slope ~FLAT (+0.181/−0.028/−0.029 per
  50k) — a 400k extension of shared lacked differential-slope
  justification *at that point* (volatility, not climb) **[granted later
  on the v2 arm, §5.9]**.
- **300k MID-COURSE READ — DIFFERENTIAL SLOPE REVERSED (18:36)**,
  `diag/panel_a_400ext_300k_{called,jd}.csv`:

  | seed | full called | perc called | full jd | perc jd |
  |---|---|---|---|---|
  | s42 | −0.316 | −0.401 | −0.142 | −0.377 |
  | s1042 | −0.139 | −0.324 | −0.042 | −0.220 |
  | s2042 | −0.173 | −0.592 | −0.158 | −0.574 |
  | mean | −0.210 | −0.439 | −0.114 | −0.390 |

  Both-modes: **full −0.162, perceiver −0.415** — gap −0.253 (~3.6×
  MDE), 0/6 paired seed-mode wins. 200k→300k movement: perceiver +0.005
  per 100k (**the climb STALLED** — the +0.159/25k at cutoff was late
  PASS-recovery catch-up that ran out); full +0.071 per 100k (**the
  "plateaued" baseline resumed climbing**). The differential that
  justified the extension reversed; the extension was doing its job
  either way (converting "budget artifact" into a converged answer).
- **Aux-head accuracy audit** run (results and the two anti-"aux-fruit"
  findings in §4.8) — motivated by the operator's hypothesis that full's
  200k→300k climb = aux heads bearing fruit. It isn't.

## 5.8 2026-07-09 — perceiver-shared-v2 built + launched; 400k budget-equal verdict

- **v2 registered + LAUNCHED ~00:13** after the mechanism review (LN
  audit / channel accounting / squeeze audit — findings in §4.4; design
  in §3.2; context-token memory driver KEPT, operator decision: strong
  game-start prior, worked in all past training, change fewer things — a
  memory-token-driver variant was built and reverted, kwarg remains).
  `perceiver-shared-v2-noaux` registered alongside (the missing factorial
  cell). 42/42 tests. Launch: 3×200k, watchdog-off ablate regime, CRN
  seeds, save-interval 25k. Pre-registered read (rule-1 endpoints vs
  full's ckpt-based): v2 ≥ full − 0.07 ⇒ family's adoption candidate
  restored, stage-1 pair full vs v2; v2 ≈ v1 ⇒ channels+LN were not
  binding, league arms decide with full as base. PASS-trap escape watch:
  confirmed escaped by ~29k (leaster 3.4%/18.5%/4.0%). Endpoint panels
  automated (`panel_v2_endpoint_watch.sh`).
- **Readout rank audit** (evening, commit eab809c; operator hypothesis:
  is v2's single 1024→256 projection a bottleneck vs full's multi-stage
  path?): NO, decisively — numbers in §4.4; convergent conclusion across
  all four mechanism audits: the family's deficit is optimization/priors,
  NOT representational capacity.
- **400k BUDGET-EQUAL VERDICT (all six endpoint panels landed).** Rule-1
  endpoints = per-seed mean over SIX panels (350k/375k/400k ×
  called/jd):

  | seed | full | perceiver | paired Δ |
  |---|---|---|---|
  | s42 | −0.094 | −0.283 | −0.190 |
  | s1042 | −0.075 | −0.297 | −0.222 |
  | s2042 | −0.026 | −0.390 | −0.364 |
  | mean | **−0.065** | **−0.324** | **−0.259 (SE 0.054, t = −4.8)** |

  **full DECISIVELY better at equal 400k budget** — 3.7× the bar, 4.8
  seed-SE, 3/3 CRN wins, both modes. Trajectories (both-modes means,
  300k/350k/375k/400k): full −0.162 / −0.118 / −0.043 / −0.034;
  perceiver −0.415 / −0.351 / −0.354 / −0.286. Full's 200k plateau was
  temporary (+0.149 own gain 200k→400k); perceiver gained similarly
  (+0.131) but from a formative-phase hole that never closed. Per-rung
  volatility remains a family signature (perc s42 swung +0.23 in the
  last 25k, winning its 400k-called rung — rule 1 smooths exactly this).
  Milestone: full@400k at −0.065 ⇒ plain 400k self-play roughly at
  parity with the frozen panel's weaker half. Consequences: plain
  perceiver OUT; the family's remaining shot = perceiver-shared-v2; if
  v2 also trailed decisively, stage-1 base = full with the league aux
  arm as the P2 measurement.

## 5.9 2026-07-10/11 — ckpt200k restatement; ext2 (onehot + v2 to 400k); stage-0 close

- **Checkpoint-based 200k table** (chain finished 07-08 21:58; restates
  all finals-based 200k numbers; seed means, both modes): full
  **−0.214** (−0.221 called / −0.208 jd), perceiver-shared **−0.373**
  (−0.377/−0.369; jd per-seed −0.317/−0.464/−0.327), perceiver
  **−0.455** (−0.437/−0.473). **onehot-ff added 07-10**: **−0.228**
  (−0.242/−0.214); CRN per-seed deltas vs full −0.023/−0.064/+0.045
  (mixed signs) — the 200k tie STOOD on the consistent methodology; the
  open question was whether onehot matched full's late acceleration.
- **Three-rung matrix completed 07-10** (150k rung landed): rule-1
  endpoints over 150/175/200k, both modes: full **−0.269**, shared v1
  **−0.337**, v2 **−0.347** — v2 ≈ v1 (Δ −0.009; the 16q+LN corrections
  had no effect at this budget — bandwidth was never binding *here*);
  full led the family by ~0.07 (≈ MDE) over the 150–200k window.
  Per-rung both-modes means: full −0.338/−0.254/−0.214; v1
  −0.406/−0.232/−0.373; v2 −0.423/−0.301/−0.315 (v1's 175k spike-up and
  200k drop illustrate the volatility — endpoint rule, not single rungs).
- **Second extension batch launched 07-10 ~09:15** (operator order):
  400k extensions for BOTH onehot-ff and perceiver-shared-v2
  (`ablate_{onehot-ff,perceiver-shared-v2}400_s{42,1042,2042}`, resumed
  from checkpoint_200000, post-flush-fix trainer so finals are clean).
  Rationale for the v2 arm: v2 ≈ v1 at 200k (Δ −0.006 ± 0.008 on the
  175k+200k rungs) meant the extension tested whether the shared-readout
  family CONVERGES to full at 2× budget, not the corrections themselves.
  Panels automated (`panel_ext2_watch.sh`, per-rung as checkpoints land).
  Pre-registered: v2 within 0.07 of full@400k ⇒ family restored for
  stage-1 league; onehot within 0.07 ⇒ token-stack advantage remains
  undemonstrated at ANY tested budget.
- **EXT2 VERDICTS (07-11, panels complete 13:08).** All six runs reached
  400k clean (onehot 07-10 ~19:00, v2 07-11 ~10:45). Rule-1 endpoints
  (350/375/400k, both modes, 3 seeds) with full400's CRN comparators:

  | arch | 300k mid | 400k endpoint | Δ vs full (CRN) | per-seed Δ |
  |---|---|---|---|---|
  | full | −0.162 | **−0.065** | — | — |
  | perceiver-shared-v2 | −0.224 | **−0.115** | −0.050 ± 0.022 (t=−2.27) | +0.002 / −0.061 / −0.092 |
  | onehot-ff | −0.224 | **−0.205** | −0.140 ± 0.020 (t=−6.99, 3/3) | −0.082 / −0.161 / −0.176 |

  **v2: WITHIN 0.07 ⇒ FAMILY RESTORED for stage 1** (own gain +0.200,
  out-slope-ing full's +0.149; residual −0.050 real but sub-MDE, not
  decisive under P1). **onehot: OUTSIDE 0.07 ⇒ token-stack advantage
  demonstrated at converged budget**; own gain only +0.023 — the tiny
  net hit its ceiling; the earlier ties retired as budget artifacts;
  onehot league arm downgraded to optional context.
- **Stage 0 CLOSED.** League pair = full vs perceiver-shared-v2,
  warm-started from their own 400k checkpoints (rule 5).

## 5.10 2026-07-11 — stage-1 alignment amendment + launch

**Amendment (pre-launch): `--main-episodes` 750k → 1,000,000.** Alignment
with the `run_extended_league.py` pre-registration
(Extended_League_202607.md) so the adopted arm resume-chains directly
into the orchestrator after stage 1. The 750k figure was a phase-2
compute carryover, never derived; 1M is the calibrated number — the
orchestrator's stop-rule constants are *per-generation* quantities (gain
≥ 0.035, paired slope ≥ 0.0175/gen, 4000-deal composite MDE ≈ 0.035) set
against the measured repro-league slope of ~0.015 score/hand per 1M
episodes. At 750k generations the slow-regime expected gain (~0.011/gen)
falls below the slope criterion and the rule false-stops; 1M is the
smallest generation whose real gain the instrument resolves (the PSRO
principle — generation length is set by what the evaluator can certify,
not by a fixed count; nothing in the league literature prescribes a
portable episode number). Changing the orchestrator down instead would
have meant re-registering every stop-rule constant.

Warm-start selection, rule-5 audit, renamed-copy mechanics, launch
commands, and the continuation recipe: §1.2/§1.3 (they are the CURRENT
sections). Launched 20:31; both arms verified healthy at launch (episode
0 parse correct, oracle ON, anchor_kl ≈ 0.005, ev_oracle ≥ ev_limited
immediately, ~2.3–2.6 eps/s initially, later ~3.3).

## 5.11 Results — stage 1 (2026-07-18, STAGE1 COMPLETE 17:17)

Shared-CRN PANEL-A gauntlets (1000 deals, seed 42, four PANEL-A
anchors) over both arms' last-3 checkpoints,
`runs/stage1_202607/panel_stage1_{called,jd}.csv`. Rule-1 endpoints
(mean of 1.9M/1.95M/2M rows; per-row SE ≈ 0.029–0.036):

| arm  | called | jd | both-modes |
|------|--------|-----|-----------|
| perceiver-shared-v2 | −0.178 | −0.054 | **−0.116** |
| full | −0.227 | −0.095 | **−0.161** |

**VERDICT (P1): ADOPT `perceiver-shared-v2`.** full does not win by
> 0.07 and > 2 SE — it *loses* by 0.045 both-modes — and per §1.3
rule 1 it is a dead arm regardless: its panel pick_rate is 0.02–1.1%
(called) / 0.02–0.24% (jd) across all three endpoint checkpoints
(greedy PICK 0.0–1.0%, leaster 95–100% in greedy health). The full
rows measure a never-picking policy whose play skill alone still
scores −0.16 both-modes; they carry no adoption information beyond
confirming the collapse.

Secondary reads (per §1.3 step 4):

- **Exploiter gates** (gen 1): both passed at near-identical edges
  (full +0.100 ± 0.037, v2 +0.111 ± 0.045); exploiters live at seat
  share 0.10–0.11 through gen 2.
- **Trump-lead probe @2M** (2000 deals): CLEAN both arms, both modes
  (v2 0 leads / ~1800 defender opportunities per mode; full 1/~2100)
  — no defender trump leak at deploy in the adopted arm.
- **Scripted probe @2M** (500 deals): v2 +0.150 ± 0.119 (above the
  sanity floor); full −0.116 ± 0.130 (below — collapsed).
- **League-regime lift vs the 400k warm start: FLAT on this
  instrument — but not yet a conclusion.** v2 both-modes endpoint
  −0.116 vs its stage-0 rule-1 endpoint −0.115 (same instrument).
  The difference carries ~±0.1 uncertainty (two noisy readings;
  jd rows swing −0.195 → +0.017 across 50k eps, the known
  perceiver-family ±0.2/50k volatility), and the panel has known
  blind spots (below). The decisive read is the direct paired h2h
  (2M ckpt vs warm start, 2000 deals, both modes — the
  pre-registered gen-vs-gen h2h instrument; launched 07-18 eve).
  full went −0.065 → −0.161 (net −0.096, all collapse damage).

**Panel validity limits (operator review 07-18).** PANEL-A stays the
headline absolute-strength instrument (outcome-grounded, CRN-paired,
frozen field, the final claim's currency) but is NOT a sufficient
progress signal alone: (1) rare-situation dilution — convention-scale
skills (e.g. v2's newly reliable secret-partner trump leads;
near-perfect aux heads by 2M) plausibly aggregate to +0.005–0.05,
below the 1000-deal MDE ≈ 0.07; (2) noise floor + checkpoint
volatility make short-window slope reads unreliable; (3) fixed-field
relativity — all four anchors are 30M-lineage, skills vs other styles
invisible; (4) `deterministic=True` play hides sub-argmax
distributional learning; (5) role-mix confound — score/hand
marginalizes over a pick/partner/defender mixture that shifted
sharply during the league phase (panel pick_rate ~50% shaped
warm start → 10–20% at 2M). ⇒ **Gen-boundary decision battery = 4
instruments:** anchored panel + paired h2h vs previous gen +
targeted convention/trump-lead probes + aux audit.

**Historical comparators are ALL confounded (operator, 07-18):**
original run ≤15M trained under heavy reward shaping + ad-hoc guards
(epsilon mixing, entropy bumps) — slope and smoothness both regime
artifacts; original >15M has logits entrenched by 15M shaped episodes
— oscillation dampened AND slope plasticity-limited; the repro-league
run was degraded per League_Run_Review_202607 (sliding-window roster,
exploiter pressure inert gens 1–11 — "effectively the PFSP-only
control trajectory"), a soft floor at best (~+0.027/1M climb after
7M). ⇒ no clean historical slope yardstick exists; **v2's own forward
instruments are primary.** This is the fresh terminal-only-agent
thesis in measurement terms — only a fresh run can exhibit
grounded-regime learning dynamics — and belongs in the writeup.

**Collapse reframed (07-18):** the original run never trained
unguarded; stage 1 ran with zero guards (no shaping, no epsilon
floor, no entropy bumps, anchor released). full's death is evidence
about the unguarded regime at least as much as the architecture, and
v2 surviving unguarded is a *stronger* stability result than the
original's smooth guarded history. Arch-linked vs seed-luck stays
unresolved at n=1 (hypotheses + falsifiers in §1.2 run log). Writeup
narrative: replace the ad-hoc stabilizer stack with one principled,
inert-when-healthy guard — LeasterWatchdog, extracted to
`sheepshead/training/leaster_watchdog.py` and wired into
train_league_ppo + run_extended_league (`--leaster-watchdog`,
default off; commit 369880c8).

**Path forward (agreed 07-18; modest single-machine budget):**

- **Phase A — retro-eval, RUNNING 07-18 eve** (eval-only): original
  lineage on the standard instrument — early rungs 0.5/1/2/3/4M
  (shaped-era absolute timeline + oscillation-under-damping context)
  and grounded rungs 20/25/30M (30M-as-candidate pins the
  "exceed 30M" target bar) →
  `runs/rigorous_baseline_202607/reference_lineage_{early,grounded}_*`;
  plus the v2 2M-vs-400k h2h →
  `runs/stage1_202607/h2h_v2_2M_vs_400k_*`.
- **Phase B — main run:** continue the v2 lineage via
  `run_extended_league` (gens 3+, same run-name; lineage remains
  "from scratch on the current repo": selfplay 0→400k + league
  0.4→2.4M), 8 workers, `--leaster-watchdog` ON (documented regime
  addition), save-interval 50k. **Stopping condition = the
  pre-registered learning-completion estimate, never an external
  target** (operator amendment 07-18): the orchestrator's
  `league_stopping` rule decides from the run's own trajectory only
  — a generation is flat when none of (A) gain vs previous BEST
  ≥ 0.035 with bootstrap lo > 0 on the 4000-deal composite,
  (B) h2h vs previous gen ≥ 0.05 at 2 SE, (C) paired 3-endpoint
  slope ≥ 0.0175/gen with lo > 0 fire; stop candidate = 2
  consecutive flats after the 4-gen floor; fresh-deal confirmation
  (seed 20260706) must agree, else the streak resets. The 12-gen
  cap is a BUDGET checkpoint (forces confirmation + operator
  review; extendable by a new pre-registration if signals still
  fire), not a completion claim. The convention/aux battery informs
  the operator review at confirmation; residual type-II risk
  (sub-MDE rare-situation learning) is accepted and documented in
  the validity-limits section above. Comparison vs the 30M final
  (CRN h2h vs final_pfsp + fresh-deal run, deals sized to the
  observed edge) is POST-HOC measurement of whatever agent the
  rule produces — it is never a stop input.
- **Phase C — reserve lever (convention holes + sample efficiency):**
  search distillation (ISMCTS offline-grade trump-lead edge +0.16 @
  4096 iters; blitz/crack/recrack plan now unblocked by the stage-1
  winner) targeted at the residual convention holes rather than more
  league episodes if the league slope stalls.
- **Deferred, non-blocking:** v2-noaux ~1M anchor-free stress test
  (aux-ballast stability discriminator + the P2 aux measurement at a
  quarter of league-arm cost); full gen-2 seed rerun (seed-luck
  falsifier). Both writeup-valuable.
- **Goal framing for the writeup: reproduce the CAPABILITY, not the
  recipe** — same-or-better endpoint under a cleaner, instrumented,
  guard-minimal methodology.

## 5.12 Phase A results + gen-adaptation hypothesis battery (2026-07-19)

Retro-eval batch landed 07-19 00:05 (`runs/rigorous_baseline_202607/
reference_lineage_early_{called,jd}.csv`, `runs/stage1_202607/
h2h_v2_2M_vs_400k_{called,jd}.csv`). Grounded-era extension
(20/25/30M rungs + 30M-as-candidate target bar) chained and running.

### H2H gate (pre-registered go/diagnose input): FLAT

v2 league 2M vs its own 400k warm start (2000 deals/mode, 5 seatings,
warmstart-only field): called **−0.023 ± 0.021**, jd **+0.045 ± 0.024**,
both-modes **+0.011 ± 0.016** score/hand. 2M league episodes (oracle
critic) added no measurable head-to-head strength — confirms the flat
panel lift by direct paired measurement. Gate verdict: **diagnose before
Phase B launch**. Note the 2M hero's pick rate in this field is 8.7–9.3%
(vs the warm start's ~53% greedy self-play pick) — role mix shifted
heavily toward defender (~75%).

### Original-run early lineage (context, NOT calibration — shaped ≤15M)

Both-modes score/hand vs PANEL-A (1000 deals/mode, single rungs, field
contains own lineage relatives):

| eps | 0.5M | 1M | 2M | 3M | 4M | 5M† | 15M† |
|---|---|---|---|---|---|---|---|
| score | −0.21 | −0.16 | +0.06 | −0.03 | +0.11 | +0.05 | +0.12 |

† earlier ladder rows, same instrument. Read: the successful run's
panel-visible climb was ~all in the first ~2M shaped episodes (~0.17/M);
2M→15M realized ≈ **0.005/M** with ±0.08 adjacent-rung oscillation —
*below the extended-league stop rule's slope_min (0.0175/M) and MDE
(0.035)*. The rule as pre-registered would have stopped the original run
around gen 4–6. Implication recorded: either the rule's floor encodes
unrealistic optimism, or post-formative league grind genuinely isn't
worth the compute without a stronger mechanism — the 30M target bar
discriminates.

**Grounded-era rungs + target bar (landed 07-19 01:20,
`reference_lineage_grounded_{called,jd}.csv`):** both-modes 20M
**+0.196**, 25M **+0.150**, 30M **+0.203** (±0.020 each; ckpt_30000000
stands in for the final — provenance: KL≈0, 98.6% argmax agreement).
Late-era slope 15M→30M ≈ **0.006/M** — the same slow grind as 2M→15M;
the post-formative rate held ~0.005–0.006/M for the entire 28M episodes,
oscillation ±0.05 between 5M-spaced rungs.

Full lineage curve (both-modes): −0.21 (0.5M) → −0.16 → +0.06 (2M) →
−0.03 → +0.11 (4M) → +0.05 (5M) → +0.12 (15M) → +0.20 (20M) → +0.15
(25M) → +0.20 (30M).

**Target-bar arithmetic** (with the fixed-field caveat: the reference
lineage is evaluated in a field containing its own relatives, which may
flatter it vs an outsider like v2): the goal bar on this instrument is
**+0.20**; v2's league-2M composite sits at **−0.116** — a gap of
~0.32 score/hand. At the historically realized post-formative rate that
is ~50–60M episodes of league grind; the gap is NOT closable at modest
budget by more-of-the-same league episodes unless the adaptation
hypothesis is right that v2's post-warmstart rate substantially exceeds
the reference's post-formative rate. Also noted: at matched total
episodes (~2.4M) the reference (shaped throughout) was at +0.06 vs v2's
−0.116 — the terminal-only path is behind at this point, which is the
cost the cleaner methodology has to make up.

### Operator hypothesis (recorded pre-battery)

Gen 1 = adjusting to terminal reward with bidding anchored; gen 2 =
adjusting to release; net ≈ 0 panel movement but much-improved aux
tracking (poor at 400k) and newly-reliable partner trump leads ⇒
training would resume producing gains in gens 3+. Also: with pick rate
down at 2M, holding total score constant implies improved defender play.

Telemetry forensics (free, from `league_training_progress.csv`):

- **Supports**: oracle critic EV climbed 0.12→0.37 across gen 1 then
  plateaued (gen 1 substantially = building the fresh critic; GAE noisy
  meanwhile). Gen-2 tail: leaster peaked 1.6–1.8M (10.9%) then declined
  (9.3%), pick recovering, lowest within-block variance of the gen —
  consistent with settling.
- **Revises**: gen-1 anchor_kl flat at ~0.011 nats (≪ 0.05 cap) — the
  anchor was never binding; bidding sat still voluntarily. The bidding
  renegotiation happened entirely in gen 2 (greedy pick 53%→low-20s
  after release). Two-phase adaptation is real but sequential:
  gen 1 = critic, gen 2 = bidding.

### Pre-registered battery (queued behind the extension; scratchpad
`hypothesis_battery.sh` → `runs/stage1_202607/hypothesis_battery.log`)

1. **Per-gen h2h decomposition** (stream A): 1M-vs-400k and 2M-vs-1M,
   2000 deals/mode, same instrument as the gate. Hypothesis predicts
   gen-1 edge ≈ 0/negative, gen-2 edge modestly positive, summing to
   +0.011. Both ≈ 0 = weak support; gen1 > 0 > gen2 inverts the story.
   → `h2h_v2_1M_vs_400k_*.csv`, `h2h_v2_2M_vs_1M_*.csv`.
2. **Aux precision ladder** (stream B): reworked `aux_audit.py`
   (2026-07-19) with precision-scaled, phase-stratified metrics —
   points MAE + P(err≤5)/P(err≤10) and trick-0/1 splits, seen-trump
   per-bit Brier + positive-class F1 by phase, unseen-higher AUC/Brier
   at tricks 0–1. Rationale: prior audit's AUC/acc saturate (partner
   head predicts OWN partner status = deterministic from own hand;
   late-game states dominate pooled metrics). Ladder: selfplay
   200k/400k + league 0.5M/1M/1.5M/2M. Hypothesis predicts material
   early-trick precision gains across the league phase.
   → `aux_ladder_v2.csv`.
3. **Partner-convention ladder** (stream B): new
   `partner_trump_lead_probe.py` — secret-partner trump-lead rate,
   tricks 0–2, scripted field, CRN seed 20260719; scripted self-check
   = high anchor (smoke: rate 1.0). Hypothesis predicts the switch
   turns on late in gen 2. → `partner_lead_ladder/*.json`.
4. **Role decomposition** (stream B): new `role_score_probe.py` —
   heroes {400k, 1M, 2M} vs constant 400k field, 600 deals/mode, long
   (deal,seat,role,score) table; decompose flat total into role-mix
   shift + within-role deltas (endogenous-role caveat recorded in the
   script docstring; paired same-role cells are the clean cut).
   Tests the defender-improvement corollary. → `role_scores_v2.csv`.

**What the battery cannot do**: verify "gains resume in gens 3+" — only
training tests that; the orchestrator's gen 3 (h2h + composite panel
built in, 4-gen floor, two-flat stop) is the designated forward test.
The battery sets the prior and decides how a flat gen 3–4 should be
read (falsified vs still-adapting).

*(Results to be appended here as each stream lands.)*

---

# Appendix A — SUPERSEDED PLANS (kept for the record; do not act on)

## A.1 Phase 2: full vs no-aux league + onehot league arm **[SUPERSEDED by stage 1, 07-07]**

Plan (launched 07-05, paused same day, killed 07-07): full vs no-aux, one
arm each, seeded from their s42 200k self-play finals, concurrent, 4
workers each; 2 generations × 750k = 1.5M episodes per arm (the 5–10M
"ideal" would be 3+ weeks; repro-v1 gave a PANEL-A reference at ~1M).
Gen 1 anchored to the arm's own resume checkpoint, gen 2 released
(anchored-then-release: collapse insurance for an anchor-free warm start
that had failed twice before). Recipe per arm:

```bash
PYTHONPATH=. .venv/bin/python -m sheepshead.training.train_league_ppo \
    --arch <full|no-aux> --resume runs/ablate_<A>_s42/final_<A>.pt \
    --seed-checkpoints 'runs/reference_selfplay_ppo/checkpoints/*.pt' \
    --league-dir runs/phase2_<A>/league --run-name phase2_<A> \
    --generations 1 --main-episodes 750000 --anchor-coeff 1.0 \
    --num-workers 4 --seed 42 --schedule-horizon 20000000
# gen 2: same minus --seed-checkpoints/--anchor-coeff,
#   --resume runs/phase2_<A>/checkpoints/pfsp_<A>_checkpoint_750000.pt
```

Exploiter phases at each generation boundary (50k eps, 3000-deal gate) →
an exploitability datapoint per arm per generation. Telemetry per arm:
`anchored_eval.csv` (300 paired deals vs final_pfsp every 100k),
`greedy_health.csv`, `exploitability.csv`. What it would have decided:
(a) aux heads under terminal reward; (b) whether the self-play ordering
survives the league regime; (c) per-arm exploitability. Runner:
`runs/phase2_202607/phase2.sh` (no internal resume guards — the
double-launch risk that motivated the watcher's pgrep guard).

**Arm 3, onehot-ff league** (queued 07-05 when the 200k tie landed;
never launched; obsoleted 07-11 when onehot lost at 400k): identical
recipe, `--arch onehot-ff --resume
runs/ablate_onehot-ff_s42/final_onehot-ff.pt`, solo on the machine
(still 4 workers for recipe identity). Runner
`runs/phase2_202607/phase2_onehot.sh` (restartable, skips finished
generations); ends by re-running PANEL-A over all three phase-2 finals in
one paired gauntlet → `panel_a_all3_{called,jd}.csv`. Pre-registered
interpretation (in the script header): onehot within 0.07 of phase2_full
⇒ token stack has no demonstrated value at ≤ 1.5M league episodes,
revisit the default on cost grounds (~5× cheaper); full ahead by > 0.07
and > 2 SE ⇒ the stack's edge lives in the terminal/league regime.
Single seed per arm — sub-0.07 orderings are noise.

**"When phase 2 lands" playbook** (for the record): check greedy_health
for collapse first (greedy PICK → 0% or leaster → 100% = dead run);
aux-verdict = PANEL-A(full) − PANEL-A(no-aux), single seed ⇒ difference
SE ~0.055 from the per-measurement SEs, gap > ~0.11 = aux matters under
terminal reward; regime check = both arms well above their −0.233/−0.277
starting points and at least competitive with the repro-v1 curve at
matched episodes (league-13.65M headline CA −0.120 / JD +0.038, panels
in `runs/rigorous_baseline_202607/`); exploitability per generation
declining.

## A.2 Phase 3: full-based capacity sweep **[SUPERSEDED by stage 2, 07-07]**

Design: the six one-knob `full-*` size variants (§3.2 table) × 3 seeds ×
200k shaped self-play, with the existing full@200k runs as the CRN-paired
center point (same seeds ⇒ same deals); auto-launched by
`watch_and_launch.sh` stage 3 after phase 2:

```bash
PYTHONPATH=. caffeinate -is .venv/bin/python -m sheepshead.training.run_ablation_matrix \
    --archs full-dtok32 full-dtok128 full-layers2 full-layers6 \
            full-dmodel128 full-dmodel512 \
    --episodes 200000 --out-dir runs/size_sweep_202607 --prefix ablate
```

finishing with `sheepshead/analysis/ablation_report.py` merging the
corrected full@200k rows from `results_200k_panel.csv` (the main
`results.csv` panel columns for those rows are stale 100k values). How to
read: a variant is better/worse only if its delta exceeds ~2 seed-SE AND
the 0.07 MDE; smaller variants win eps/s (question: do they lose
PANEL-A?); bigger variants must beat full by > 0.07 to justify cost; any
still-climbing variant's endpoint understates it — extend the climbing
arm rather than concluding "same".

**What the sweep did NOT probe — the pooling readout (the open question
that drove the whole probe program):** raised 07-05 by the operator —
could the attention-pool squeeze be throwing away the rich embeddings'
advantage, explaining why onehot ties full? (Encoder facts now in §3.2.)
Sweep coverage: dmodel128/512 scale the pool *outputs* (conflated with
trunk/GRU width by design); dtok32/128 scale the pool inputs/queries; the
query count (4) and the *structure* (pooling vs direct token readout)
were NOT probed. Diagnostic reading pre-registered: dmodel512 beating
full (> 0.07) ⇒ pooled-bottleneck capacity binding, readout redesign =
highest-value next experiment; a null ⇒ bottleneck-width story loses
support (structural question stays open). Caveat (07-06): every knob is
confounded by the pool squeeze — a depth null on full ≠ depth useless.
Candidate structural variants registered at the time: (1)
`full-tokenread` (built, probed, NULL); (2) wider pools
(`AttentionPool.n_queries` 4→8, never built); (3) richer memory (feed
the GRU the fused 256-d features via the `_fuse_and_update_memory` seam,
never built).

## A.3 Probe contingency playbooks **[EXECUTED or SUPERSEDED, 07-06/07]**

**Tokenread-wins branch (did not fire — verdict was NULL):** pause
phase 2 (hours into gen 1), archive, re-plan arms on the new base —
full-tokenread vs a `tokenread-no-aux` registry one-liner, onehot arm
unchanged as cost baseline; warm-start
`runs/ablate_full-tokenread_s42/final_full-tokenread.pt`. Sweep variants
stay full-based (width-vs-structure triangulation). Next step already
sketched: feed the post-reasoning MEMORY token into the GRU (zero new
params — TokenReadEncoder subclass overriding `_fuse_and_update_memory`
to use `all_tokens[:, 1, :]`, or concat with context → GRUCell(128, 256))
— one-knob it against full-tokenread. Operator long-term intent recorded
07-05: tokenread is a *probe*, deliberately clunky (additive so the win
is attributable); if it won, the clean end-state was a token-centric
re-architecture (Perceiver-IO shape) — which became `perceiver`.

**Perceiver-lands branch (EXECUTED 07-07, tie/loss path):** the win path
(rebase phase 2 on perceiver, `perceiver-aux` as the aux arm — registered
07-06 after the operator flagged that dropping aux was unintended; the
probe itself ran aux-free because restarting would have pushed past the
access window and redefining `perceiver` mid-flight would break its own
checkpoint loads) did not fire. The sweep-rebase edit it would have used
(perceiver-based variants + attention-shape knobs, ~12 variants ≈ 2× the
wall-clock, splittable into capacity-first + attention-second with
`--extra-results` merging):

```
--archs perceiver perceiver-dtok32 perceiver-dtok128 perceiver-layers2 \
        perceiver-layers6 perceiver-dmodel128 perceiver-dmodel512 \
        perceiver-readq2 perceiver-readq8 \
        perceiver-readheads2 perceiver-readheads8 \
        perceiver-rheads2 perceiver-rheads8 \
--leaster-watchdog
```

The watchdog decision (07-06, applied to ANY sweep): escape latency from
the PASS trap is seed lottery — pure confound for capacity questions;
because the watchdog changes the regime, the base arch is INCLUDED
in-sweep (`--baseline <base>` with NO watchdog-off `--extra-results`; old
probe rows cited as context only; the watchdog-on base vs watchdog-off
probe is a free read on what the watchdog is worth). Watcher-edit
mechanics (both branches): kill ONLY the watcher shell (`pgrep -fl
watch_and_launch`), never the training processes; edit stage 3; relaunch
`nohup zsh runs/size_sweep_202607/watch_and_launch.sh > /dev/null 2>&1 &
disown` — per-stage skip guards make this safe; do it before stage 3
launches.

## A.4 The July watcher chain **[HISTORICAL]**

One detached watcher (`runs/size_sweep_202607/watch_and_launch.sh`,
rewritten 07-06 ~10:15) drove: tokenread probe → perceiver probe →
phase2.sh → phase2_onehot.sh → capacity sweep → report; progress in
`watcher.log`, per-stage done markers (`STAGE 0 (TOKENREAD) COMPLETE`,
`STAGE 0.5 (PERCEIVER) COMPLETE`, `PHASE2 COMPLETE`, `PHASE2 ONEHOT ARM
COMPLETE`, `SIZE SWEEP COMPLETE`). Stages 1+ never ran (phase 2
displaced). Resume-from-reboot rules: rerun the watcher (completed
stages skip: stages 0/0.5/3 per job via the orchestrator; stage 1 via
its marker; stage 2 per generation); a league generation that died
mid-run restarts from its generation start unless `--resume` is
hand-edited to the newest boundary checkpoint with `--main-episodes`
reduced accordingly. The tokenread probe's measured train rate at the
time: ~3.7–4.3 eps/s vs full's 7.4–8.0 (the token readout makes the PPO
update ~1.9× costlier) — later corrected by the equal-load bench
(rule 4).
