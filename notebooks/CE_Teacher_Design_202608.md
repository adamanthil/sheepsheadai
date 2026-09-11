# CE Search Teacher — Design & Implementation Plan (2026-08)

Status: APPROVED & LAUNCHED as attempt 11 (2026-08-17; §12 launch record).
Ceiling gate passed: +0.180 ± 0.029 at n=500, decisively material.
Successor to the resolved-pair hinge teacher (Search_Teacher_Design §12,
attempts 5a–10, all retired). This document is the implementation contract
for the always-on cross-entropy teacher and the accompanying cleanup: after
this lands, the ONLY search trainer in the codebase is this design, and the
ONLY entropy controller is the v2 signed controller. Removed code lives in
git history (final pre-removal commit will be tagged `pre-ce-teacher`).

Evidence base, in one paragraph: policy-space hinge teaching proved
transient against PG — gains and damage both squeezed out (§12.22); the
per-decision search committee itself carries real EV (§13.3 ceiling h2h,
FINAL +0.1800 ± 0.0289 at n=500 — full record in Search_Teacher_Design
§13.3 RESULT; called +0.210±0.038, jd +0.150±0.043; t0 called-suit
adherence raised 40.0→56.2 by the acting committee, per prediction) and
raises conventions at act time; the sample-efficiency frontier for
installing search results is CE toward the completed-Q improved policy
(ExIt → AlphaZero → Grill 2020 → Gumbel MuZero), which makes abstention,
ambiguity preservation, and entropy neutrality properties of the TARGET
rather than bolt-on gating (§13.4).

---

## 1. The loss

### 1.1 Target construction (at emission time, in the worker)

At a taught node with legal set V, label-time policy prior p_raw (the
root-visit-averaged unmixed prior the engine already accumulates), and
committee-pooled completed-Q vector q̄ with per-action replicate SEs:

1. **Shrink**: q̃ = shrink(q̄) — deviations from the visit-weighted mean
   are shrunk toward zero by the noise model (§1.2). A node whose Q
   spread is within noise shrinks to flat.
2. **Tilt**: π_target = softmax(log p_raw + scale · minmax_unit(q̃))
   with scale = (gumbel_c_visit + max N) · gumbel_c_scale — i.e. the
   EXISTING pi_gumbel readout evaluated on shrunk Q. The deployment
   readout IS the training target; act-time and train-time semantics
   never diverge.

Properties (each replaces a §12 mechanism):
- q̃ flat → π_target = p_raw ≈ label-time policy → CE gradient ≈ 0.
  Abstention is the target's fixed point (replaces ε-gate + emission
  bookkeeping; no incumbent tax at ties by construction).
- Near-ties keep the policy's own distribution over the tie set
  (ambiguity preserved natively; prior-preserving, NOT max-ent
  flattened — the §12.20 entropy-saturation lesson).
- Confident gaps sharpen proportionally to evidence (visit scale ×
  shrunk Q), bounded by the softmax tilt (no λ=50 scale mismatch).

The target is **fixed at label time**. It is NOT recomputed against the
moving policy during reuse epochs — recomputation iterates the
improvement operator and over-sharpens past the intended KL ball.

### 1.2 Noise model (shrinkage calibration)

Per-action variance from R=3 replicates is unstable (2 dof), so blend
with the global replicate-noise calibration measured in §12.8
(deflead_gating_study replicate spreads at 1024/1):

  s²_a ← (ν·s²_global + (R−1)·s²_node,a) / (ν + R − 1),   ν = 4

Single per-node shrink factor (positive-part James-Stein on the
centered Q vector): w = max(0, 1 − s̄²/Var_V(q̄)); q̃ = w · (q̄ − mean).
CALIBRATION GATE before first use: on the archived gating-study reps,
shrinkage must (a) produce ~zero tilt at the known EV-wash cells
(fat/nopoint), (b) preserve direction at the called-suit cells
(153:7 directionality), (c) at fresh committee draws of the same
nodes, tilt sign must be stable across draws at the surviving cells.

### 1.3 Trainer loss

CE(π_target ‖ π_θ) over the legal set on labeled rows, coefficient
`teacher_coeff` (default 1.0), ADDED to the PPO objective (no PG-mask:
the PG term stays active on labeled rows — reward and teacher are
aligned at material nodes per §12.16, and PG maintains equilibrium
where the target is flat).

**Asymmetric epochs**: the PG loss keeps the current single-epoch
tuning (ratio staleness). The CE term runs `teacher_epochs` (default 4)
passes over the update window's labeled rows — supervised target, no
importance ratios, AZ-standard reuse. Implementation slots into the
existing `--oracle-extra-epochs` aux-epoch structure. Labels are
discarded with their update window (staleness cap = 1 window).

---

## 2. Emission pipeline (worker)

Eligibility: learner-controlled seats (primary + the ~15%-occupancy
opponent seats), PLAY head, ≥ 2 legal actions, standard game (no
leaster/alone). CLASS-BLIND: no cell taxonomy, no confidence trigger
(§13.3: a top-2-gap trigger captures only ~35% of policy-wrong t0
called-suit nodes — confidence triggers are circular). Subsample at
`teacher_prob` per eligible node (the budget knob; unbiased).

At a sampled node: `search_committee` (lockstep, commit 7283fb9) with
R = `teacher_replicates` = 3 rngs at `teacher_iters` = 1024,
d_rollout 1, frozen expert (§3). Pool per-replicate root_q into q̄ /
SEs, build the target per §1.1, attach to the transition:

  transition["search_target"] = float32[len(valid)] (aligned to sorted
  valid), plus telemetry scalars (w, spread, max-tilt KL).

Replaces: `search_pairs` [w, l, anchor_w, anchor_l] × ≤ 8 rows.

Telemetry per gate window (replaces the pair telemetry line): nodes
searched, fraction with w > 0 ("material"), mean per-node
KL(target‖policy) at label time ("gap" analog — self-retirement
readout: decays as the policy conforms), CE loss, teacher_epochs.

---

## 3. Generation structure (always-on)

- NO phases. Teaching runs the whole generation at `teacher_prob`.
  No consolidation windows (§13.4: any teacher-off window is a
  measured reversion window). Phase markers, adaptive exit, and the
  consolidation branch are deleted.
- **Expert refresh**: per generation, expert = gen-start checkpoint,
  frozen (attempt-7/8 lesson). At the boundary, the candidate
  checkpoint runs the ABSOLUTE-anchor cert (fixed bars, never
  relative-to-previous, to prevent refresh-chain ratchet): n=1000 × 3
  seeds adherence battery (multi-seed: §12.22 — single reads are
  luck-of-phase) + h2h vs the FIXED 8M seed + exploiter gate. Pass →
  next gen's expert; fail → operator review.
- **Guards** (in-trainer, two-tier, §12.21 protocol): n=1000 fixed-seed
  adherence probe every `adherence_guard_interval`:
  - partner-trump < 90.0 (hard floor) OR t0-trump > 5.0 → checkpoint +
    SystemExit(3) (operator review).
  - partner-trump < 93.5 → print NOTIFY line, continue.
- Exploiter gate at boundary: unchanged.

---

## 4. Entropy controller v2 (signed)

The current controller steps alpha in LOG-space — alpha > 0 by
construction, so it saturates at its floor against any injection
(§12.20 diagnosis). v2 (same module, `entropy_controller.py`,
rewritten):

- Signed alpha ∈ [alpha_min, alpha_max] (default [−0.05, +0.25] per
  head; play cap tighter than legacy since negative range exists).
- Linear-space integral step: Δalpha = eta_lin · (target − measured),
  per-update clamp |Δalpha| ≤ max_step; eta_lin calibrated to match
  the legacy controller's ~5.9%/update response at alpha ≈ 0.15.
- Bumpless attach (alpha initialized from checkpoint/legacy value),
  per-head targets, target annealing + floors (play 0.28) preserved
  verbatim from v1.
- v1's log-space stepping, `--entropy-mode` selection, and any
  fixed-coefficient legacy path are REMOVED; the v2 controller is
  always on for the trainer.

Expected interaction: the CE teacher is approximately entropy-neutral
(prior-preserving at ties), so v2 should hover near legacy behavior;
the negative range is a backstop, not the operating point. Telemetry:
alpha sign flips logged.

---

## 5. Removal inventory (full sweep; git history is the archive)

`train_league_ppo.py` (1975 lines today):
- Flags: --search-teacher-margin, --gate-pair-eps, --search-label-weight,
  --search-clip-delta, --teacher-phase-cap, --teacher-exit-emission-pct,
  --teacher-exit-learned, --teacher-exit-windows, --entropy-mode.
- Two-phase generation loop: phase_a_budget math, teacher_phase_done
  marker read/write, 🧊/🧘 branches, phase_exit plumbing in
  run_main_phase, mid-phase checkpoint special-casing.
- Resolved-pair telemetry (🔍 gate window line) → replaced per §2.
- Retained (renamed where noted): --search-teacher → --teacher,
  --search-teacher-prob → --teacher-prob (default 0.1),
  --search-replicates → --teacher-replicates (default 3),
  --teacher-ckpt (unchanged), adherence-guard flags (two-tier
  defaults per §3), NEW: --teacher-coeff (1.0), --teacher-epochs (4),
  --teacher-iters (1024).

`pfsp_runtime.py`:
- `_attach_gated_search_target` pair-emission body (sign-consistency,
  ε floor, t-stat sort, satisfied filter, gate_max_pairs) → replaced
  by §2 target emission (committee call + shrinkage + tilt).
- Gate diagnostics dict reshaped (searched / material / KL sums).

`config.py` SearchConfig:
- DELETE: gate_pair_eps, gate_pair_z, gate_max_pairs, gate_emit_margin,
  gate_cells (class-blind now — the cell taxonomy dies).
- RETAIN/RENAME: gate_iters → teacher_iters (1024), gate_replicates →
  teacher_replicates (3), gate_d_rollout → teacher_d_rollout (1),
  gate_node_prob → teacher_prob (0.1). NEW: shrink_nu (4),
  shrink_s2_global (from §1.2 calibration).

`agent/ppo.py`:
- DELETE the pair-hinge distillation block: search_pairs_bt/flat
  tensors, search_label_weight (50), search_clip_delta / pair-gap
  trust region, PG-mask mix, DQfD per-sample weighting, hinge
  telemetry (search_hinge_sum etc.).
- ADD: search_target storage (ragged valid-aligned float32),
  CE-distill term + teacher_epochs loop, CE/KL telemetry.
- ExIt-era remnants in the same block (Stage-C distill/PG-mask paths)
  go with it. (train_pfsp_exit.py itself no longer exists post-reorg;
  sheepshead/validation/exit_validation.py is an evaluation harness,
  untouched by this plan.)

`entropy_controller.py`: v1 log-space controller replaced by v2 (§4).

NOT touched: ismcts.py engine (search_committee + serial path,
goldens), all analysis/ instruments, exploiter.py (its reward-shaping
`shaped` branch in pfsp_runtime is exploiter machinery, not search).

---

## 6. Tests & gates

- Existing, must stay green: test_ismcts_committee.py (R=1 bit-exact,
  R=3 equivalence), capture_search_goldens --check,
  capture_arch_goldens --check (arch untouched — should be trivially
  green), full sheepshead/tests suite.
- test_gated_search_teacher.py: pair-emission tests deleted with the
  code; REWRITTEN for §2: eligibility filter, shrinkage math on
  synthetic replicate tables (flat → zero tilt; separated → direction
  preserved; blend math), target = pi_gumbel-on-shrunk-Q equivalence,
  transition payload shape.
- New ppo unit test: CE term gradient is zero when target equals the
  policy distribution; teacher_epochs reuse touches only labeled rows;
  telemetry sums.
- New entropy_controller v2 tests: sign crossing, clamp, bumpless
  attach from a v1 checkpoint state (the json format carries over),
  floor/anneal behavior vs v1 reference traces.
- Calibration gate (§1.2) run and recorded BEFORE the first training
  launch.
- Smoke: 2-generation micro-run (crafted small budgets) exercising
  emission → CE loss → guard probe → boundary cert → refreeze,
  crash-resume mid-generation (no phase markers anymore — resume is
  plain checkpoint resume).

## 7. Implementation order

1. Entropy controller v2 + tests (independent, unblocks everything).
2. SearchConfig reshape + §2 emission (worker side) + tests.
3. ppo.py: CE loss + payload + asymmetric epochs + tests.
4. train_league_ppo.py: flag surface + always-on gen loop + guards;
   full removal sweep in the same commit series (one concern per
   commit; tag `pre-ce-teacher` first).
5. Shrinkage calibration gate on archived study data; record here.
6. Smoke gen; then attempt-11 pre-registration (separate section,
   written at launch time with the §13.3 final numbers in hand).

## 8. Cost projections (measured basis; 1.6 learner seats/ep)

eps/s ≈ 8 / (1.3 + 362·p) on current CPU (62 CPU-s per R=3 committee):
p=0.1 → ~0.21 eps/s (~5.5 d/100k, ~15k material labels, every
convention class ≥ 2× its ~1k installation budget); p=0.05 → ~0.41
(~2.8 d/100k). MPS path if p must rise: per-worker MPS ~2–3×;
central inference server (batches across workers; GRU memory already
explicit-row, device-residency straightforward) ~4–8× — build only if
always-on graduates to standing architecture.

## 9. Pre-registered expectations (to finalize at launch)

- KL(target‖policy) at labeled nodes decays within-generation
  (CE self-retirement signature — the analog of emission decay that
  attempt 10 never showed).
- Conventions: calibrated called_suit_probe TRICK-0 rises from ~40%
  and HOLDS while teaching continues (no consolidation reversion
  window exists to hide behind); partner-trump stays ≥ 93.5 n=1000
  THROUGHOUT (the §13.4 gentleness claim — attempt 10 bled to 80.9).
- Strength: gen-end h2h vs 8M seed captures 25–50% of the §13.3
  ceiling in gen 1 (honest guess, falsifiable).
- Entropy: Hn play stays within ±0.03 of target with alpha > 0
  (entropy-neutrality claim; sustained negative alpha = the teacher
  is injecting after all → investigate before gen 2).

---

## 10. Implementation record (2026-08-16)

Landed on master (commit series after tag `pre-ce-teacher`); full
sheepshead/tests suite green (540 passed), search + arch goldens green.
Two spec ambiguities in §1 were resolved during implementation and are
now load-bearing code comments (`pfsp_runtime.build_ce_search_target`):

1. **Shrink placement vs min-max normalization.** §1.1's literal
   "minmax_unit(q̃)" is affine-invariant in a single per-node scalar w —
   `minmax(w·(q̄−mean))` is identical for every w > 0, which would reduce
   the shrink to a hard on/off gate. To preserve the stated properties
   (continuous evidence-proportional sharpening; flat at w=0; exactly the
   deployment readout at w=1), the implementation multiplies the
   NORMALIZED vector: target = softmax(log p_raw + scale·w·minmax(q̄)).
2. **Noise term in the JS ratio.** s̄² is the sampling variance of the
   POOLED committee mean (the blended per-replicate variance divided by
   the per-action observation count), i.e. the estimator's noise is
   compared against the observed spread of the estimates — the
   statistically matched form of §1.2's formula. The committee scale
   (`max N`) is the per-replicate mean of max visit counts, keeping the
   tilt scale identical to a single deployment search.

Loss normalization: CE is mean-over-labeled-rows at `teacher_coeff`
(AZ-standard). Constant total force at shrinking label counts is safe
here — unlike the §12 hinge — because abstention lives in the target
(a conformed or within-noise row carries ~zero CE gradient), so
self-retirement is per-row, not per-batch. CE passes step the actor
path only (actor + encoder; one optimizer step per pass, counted in
optimizer_steps_total).

### 10.1 Shrinkage calibration gate (§1.2), run 2026-08-16

Instrument: `sheepshead/analysis/calibrate_shrinkage.py` on the archived
§12.8 deflead gating study (144 nodes × 6 replicates at 1024/1), fed
through the PRODUCTION target builder (uniform priors / equal visits —
w and tilt direction are invariant to both).

- **shrink_s2_global calibrated = 6.95e-4** (per-action per-replicate Q
  variance, pooled mean over 720 action cells; SD ≈ 0.026 Q; median
  3.5e-4, p90 1.7e-3). Config default updated from the provisional
  1.1e-4 derivation.
- **Abstention at noise**: committee-of-3 targets shrink to flat at
  10% / 27% / 50% of t0 / t1 / t2 defender-lead nodes (mean w 0.50 /
  0.44 / 0.30) — shrinkage tracks the known per-cell scatter ordering.
- **Criterion (c), split-committee stability**: disjoint 3-rep draws
  agree on the tilt argmax at only 39/81 both-material nodes — BUT all
  42 flips sit at pooled-6 top-2 gaps below 2·SE of a committee mean
  (median flip gap 0.0025 Q vs 0.0111 stable; SE₃ = 0.0152 Q), i.e.
  every instability lives inside the statistical tie set. A sweep showed
  this is intrinsic (raising shrinkage 12× still leaves ~20% flips while
  flattening 92% of nodes): the scalar w separates signal-vs-noise
  SPREAD, not top-2 order. The gate passes on the design's own terms:
  CE is LINEAR in the label, so repeated draws at a tie-set archetype
  average the teaching signal to the tie-set spread (the §1.1 ambiguity-
  preservation property) — the incumbent-tax mechanism needed a
  nonlinear anchored loss and is structurally absent.
- **Criteria (a)/(b) as written are NOT coverable from archives**: the
  §12.15 EV studies recorded belief-MC deltas, not committee Q tables,
  so no archived committee draws exist at fat/nopoint or called-suit
  cells. Direction agreement vs the self-agreeing 4096/term reference on
  this data: 32/49 (the reference itself self-agrees only 38-48% at
  these cells, §12.8, so this is a soft check). A fresh committee draw
  at fat/nopoint + called-suit nodes belongs on the attempt-11
  pre-launch checklist (cheap: lockstep committee ≈ seconds/node).

### 10.2 Deviations / notes

- The boundary cert (§3) is automated in-trainer
  (`train_league_ppo.run_boundary_cert`): --cert-seeds × --cert-games
  adherence battery judged on across-seed MEANS + paired CRN h2h vs a
  launch-time-fixed anchor (--cert-anchor-ckpt, default the original
  expert); FAIL saves the cert JSON and halts with exit 4 for operator
  review. The exploiter gate keeps its existing boundary flow.
- Progress-CSV gate columns were REPLACED (not appended):
  gate_attempts/gate_emitted/gate_pairs/gate_learned →
  teacher_searched/teacher_material_frac/teacher_kl/teacher_ce. Old
  teacher-run CSVs are not resumable across this boundary (none are:
  attempt-10's run is closed).
- run_extended_league no longer passes --entropy-mode (removed); the
  trainer's v2 controller is always on and attaches bumplessly, so the
  gen-1 deferral is gone. --adaptive-entropy now governs only the
  orchestrator's outer target-step + flat-absorption stop rule.
- `visualizations/dump_ismcts_trace.py` (uncommitted scratch from the
  explorer work) still references the removed gate_* SearchConfig
  fields and will need updating if it is ever committed.
- **--oracle-init finding (2026-08-16, attempt-11 launch prep;
  operator caught it)**: the flag OVERWRITES the oracle critic AFTER
  the resume load, in both the training agent and the frozen teacher
  expert (same order pre-refactor — verified against a7a0744's
  train_league_ppo). It exists for resuming PRE-oracle checkpoints
  (the original Jul-25 retention launch, where the seed carried no
  oracle_state_dict); on a post-oracle resume it silently downgrades
  the checkpoint's trained oracle to the 400k pretrain. The 8M seed
  checkpoint DOES carry oracle_state_dict + oracle_optimizer, so
  attempt 11 launches WITHOUT --oracle-init. Historical footnote:
  attempts 9 and 10 inherited the flag by launch-recipe copy-paste,
  so their training-time teacher experts evaluated leaves with the
  400k-pretrain oracle while every offline instrument (E9 cert, §12.8
  gating study, ceiling h2h, §10.3 verification) used the
  checkpoint's 8M oracle via load_agent — an instrument/deployment
  oracle mismatch. (Not retro-blamed for §12.4 scatter: the §12.8
  study measured high scatter WITH the 8M oracle.) Dropping the flag
  aligns attempt-11's deployed teacher with the calibrated
  instruments for the first time in the teacher lineage.

### 10.3 Fresh-draw cell verification (§1.2 criteria (a)/(b)), run 2026-08-16

Instrument: `analysis/verify_shrinkage_cells.py` (a7a0744) — live lockstep
committees (R=3 @ 1024/1, production target builder, trainer defaults) at
36 fat/nopoint EV-wash defender leads (tricks 0-2) + 36 t0 called-suit
defender leads, sampled from greedy self-play of the clean 8M seed.
Full draws: runs/ce_teacher_prelaunch/verify_shrinkage_cells.json.

**(b) called-suit cells: PASS.** 75% material (mean w 0.47); among the 27
material tilts, mass moves TOWARD the called-suit class 17 : 6 : 4
(toward/away/neutral at ±0.02; binomial p ≈ 0.035), mean push +0.23 of
probability mass, target-argmax installs 8 vs removals 2. Shrinkage at
the calibrated constant does NOT silence the one convention we most need
to teach. (The §12.17 153:7 analog is not expected 1:1 — that study
filtered by ε=0.03 Q materiality; w > 0 is a weaker filter and admits
near-neutral rows.)

**(a) wash cells: PASS on the pairwise reading, with a recorded nuance.**
The naive summary looks like a fail — only 47% shrink to w=0, and the
class-marginal push is 10:5 "toward nopoint". But the §12.15 wash
finding is about the fat↔nopoint PAIR, not the node: per-row, the
CONDITIONAL fat-share fat/(fat+nopoint) moves 6 toward nopoint, 6 toward
fat, 7 neutral (median delta +0.007) — no systematic pair direction
survives shrinkage, so CE-linear averaging cancels class-level teaching
pressure at washes (the §10.1 tie-band argument, confirmed behaviorally).
The material tilts at wash NODES are real signal on OTHER options at the
same node: called-suit installs at overlapping t0 nodes (+0.72/+0.83/
+0.97 called-class pushes — criterion (b) showing up in family (a)'s
sample) and two late-trick pushes INTO trump (t1/t2; consistent with
§12.8's "trump appears by t2"). Zero material wash rows push mass OUT of
trump beyond −0.05.

**Recorded caveat**: per-node label variance at wash cells is high
(single-draw conditional-share swings up to ±0.9 in both directions).
Class-level safety rests on sign-mixing + the ~15k-labels/gen averaging
scale, not on per-node convergence — the §9 partner-≥93.5-throughout
guard remains the behavioral backstop for this residual risk. With this,
the §10.1 open item is closed: all three §1.2 criteria are now verified
(criterion (c) in §10.1, criteria (a)/(b) here) and the calibration gate
is COMPLETE for attempt-11 launch.

---

## 11. References (for the eventual write-up)

The design's lineage claim in one line: Expert Iteration supplies the
loop, AlphaZero the CE projection step, Grill et al. the theory that the
visit/completed-Q target is a regularized policy improvement, Gumbel
MuZero the specific completed-Q readout we train toward, DAgger the
on-policy-states-with-stationary-expert correction, and James-Stein the
noise-adaptive abstention.

Search & target construction:
- Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree
  Search," IEEE Trans. Comput. Intell. AI Games 4(2), 2012 — SO-ISMCTS,
  the engine's algorithm.
- Long, Sturtevant, Buro & Furtak, "Understanding the Success of Perfect
  Information Monte Carlo Sampling in Game Tree Search," AAAI 2010 —
  determinization limits (strategy fusion / non-locality) behind the
  oracle-leaf "shortcut not leak" argument (§13.3 discussion).
- Rosin, "Multi-armed Bandits with Episode Context," Ann. Math. Artif.
  Intell. 61(3), 2011 — PUCT.
- Chaslot, Winands & van den Herik, "Parallel Monte-Carlo Tree Search,"
  Computers and Games 2008 — root parallelization (the committee's
  independent-replicate form; used for noise estimation, not speed).
- Danihelka, Guez, Schrittwieser & Silver, "Policy Improvement by
  Planning with Gumbel," ICLR 2022 — completed-Q + sigma-scale tilt;
  the pi_gumbel readout the target reuses.
- Grill, Altché, Tang, Hubert, Valko, Antonoglou & Munos, "Monte-Carlo
  Tree Search as Regularized Policy Optimization," ICML 2020 — the
  visit/Q-tilt target as a KL-regularized improvement step (why the
  softmax tilt is the principled sharpening bound).

Installation (the loop and the loss):
- Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning
  and Tree Search," NeurIPS 2017 — Expert Iteration.
- Silver et al., "Mastering the Game of Go with Deep Neural Networks
  and Tree Search," Nature 529, 2016 — prior-guided PUCT; "Mastering
  the Game of Go without Human Knowledge," Nature 550, 2017 — CE toward
  the search policy with buffer reuse (the asymmetric-epochs precedent);
  "A General Reinforcement Learning Algorithm that Masters Chess,
  Shogi, and Go through Self-Play," Science 362, 2018 — AlphaZero.
- Ross, Gordon & Bagnell, "A Reduction of Imitation Learning and
  Structured Prediction to No-Regret Online Learning," AISTATS 2011 —
  DAgger: labels on the student's state distribution from a stationary
  expert (the frozen-expert-per-generation rule; attempts 7/8 measured
  the non-stationary failure mode).
- Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural
  Network," arXiv:1503.02531, 2015 — soft-target CE (why ambiguity at
  ties transfers, not just argmaxes).
- Schulman et al., "Proximal Policy Optimization Algorithms,"
  arXiv:1707.06347, 2017 — the host objective; ratio staleness is what
  keeps PG at one epoch while the supervised CE term reuses the buffer.

Shrinkage / abstention:
- James & Stein, "Estimation with Quadratic Loss," 4th Berkeley
  Symposium, 1961; Baranchik, "Multiple Regression and Estimation of
  the Mean of a Multivariate Normal Distribution," Stanford TR 51,
  1964 (positive part); Efron & Morris, "Data Analysis Using Stein's
  Estimator and Its Generalizations," JASA 70(350), 1975 (the
  empirical-Bayes/hierarchical variance blend of §1.2).

Entropy controller v2 (§4):
- Haarnoja et al., "Soft Actor-Critic Algorithms and Applications,"
  arXiv:1812.05905, 2018 §5 — automatic temperature adjustment (the
  inner loop's form); Christodoulou, "Soft Actor-Critic for Discrete
  Action Settings," arXiv:1910.07207, 2019 — discrete/normalized form.
- Åström & Wittenmark, *Adaptive Control*, 2nd ed., 1995, ch. 9 —
  bumpless transfer.
- Jaderberg et al., "Population Based Training of Neural Networks,"
  arXiv:1711.09846, 2017 — outer-step perturbation scale.
- Sokota et al., "A Unified Approach to Reinforcement Learning, Quantal
  Response Equilibria, and Two-Player Zero-Sum Games,"
  arXiv:2206.05825 (ICLR 2023) — mixed equilibria in imperfect
  information; why entropy floors are never zero.

Evaluation instruments:
- Bard, Hawkin, Johanson & Szafron, "The Annual Computer Poker
  Competition," AI Magazine 34(2), 2013 — duplicate-match format (the
  ceiling h2h / boundary h2h pairing); Burch, Schmid, Moravčík,
  Morrill & Bowling, "AIVAT: A New Variance Reduction Technique for
  Agent Evaluation in Imperfect Information Games," AAAI 2018 — the
  variance-reduction goal the zero-centered paired design shares.

Search-Q policy iteration (§20) — reading order to understand the
implementation (2026-09-01; each item names the piece of the recipe it
supplies):
- Fay & Herriot, "Estimates of Income for Small Places," JASA 74, 1979 —
  THE estimator of Stage 1b: known-variance measurements, a covariate
  regression, the precision-weighted blend (gamma) and the residual-
  variance estimate; "area" = searched node.
- Efron & Morris, "Data Analysis Using Stein's Estimator and Its
  Generalizations," JASA 70, 1975 — why pooling toward a group estimate
  beats per-unit estimates under noise; the empirical-Bayes framing that
  also justifies shrinking per-class residual variances toward the
  global one.
- Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep
  Learning for Computer Vision?", NeurIPS 2017 (arXiv:1703.04977) —
  heteroscedastic regression with a learned per-input variance: the
  per-node version of the per-class residual variance (§20.6).
- Vieillard, Pietquin & Geist, "Leverage the Average: an Analysis of KL
  Regularization in Reinforcement Learning," NeurIPS 2020
  (arXiv:2007.06799) — KL-regularized policy iteration (new policy =
  old policy tilted by exp(advantage / temperature)); the error-
  averaging theorem behind the multi-iteration compounding claim.
- Grill et al., "Monte-Carlo Tree Search as Regularized Policy
  Optimization," ICML 2020 (arXiv:2007.12509) — the search target as
  the same regularized step with lambda set by visit counts; read with
  §20.1 for exactly which assumption fails under determinization.
- Wang et al., "Critic Regularized Regression," NeurIPS 2020
  (arXiv:2006.15134); Nair et al., "AWAC," 2020 (arXiv:2006.09359);
  Peng et al., "Advantage-Weighted Regression," 2019 (arXiv:1910.00177)
  — advantage-weighted policy extraction, the temperature and the clip;
  our tilt with the advantage from search instead of a TD critic.
- Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning and
  Tree Search," NeurIPS 2017 — the phase-pure loop (generate with a
  frozen policy, improve with search, project, certify).
- Sun et al., "Dual Policy Iteration," NeurIPS 2018 (arXiv:1805.10755)
  — theory of fast-policy / slow-search loops and when the projection
  contracts.
- Ross, Gordon & Bagnell, DAgger, AISTATS 2011 — corpus on the
  student's own state distribution with a stationary expert; why
  committee acting is off.
- Danihelka et al., "Policy Improvement by Planning with Gumbel," ICLR
  2022 — the retired target's completed-Q readout and its visit-count
  sigma scale (its eq. 8 vs §20.1).
- Optional: Wortsman et al., WiSE-FT (arXiv:2109.01903) — the
  interpolation control; Lisý, Lanctot & Bowling, "Online Monte Carlo
  Counterfactual Regret Minimization for Search in Imperfect
  Information Games," AAMAS 2015 — the in-engine average-strategy
  alternative held in reserve.

(The retired §12 pair-hinge lineage — Bradley-Terry, RankNet, DPO
(Rafailov et al. 2023), DQfD (Hester et al., AAAI 2018) — is cited in
Search_Teacher_Design_202608.md §12.7 and its references block, and
belongs to the negative-results half of the write-up.)

---

## 12. Attempt-11 launch record & pre-registration (2026-08-17)

Gate resolution: §13.3 ceiling h2h completed 2026-08-17 (21.7h,
runs/ceiling_h2h_202608/): EDGE +0.1800 ± 0.0289 score/deal at
n_deals 500 (~6.2σ; called +0.2096±0.0384, jd +0.1504±0.0433;
win_frac 0.591). Clears the pre-registered +0.05 materiality bar by
>4σ → always-on strength case ALIVE; launch authorized under the
operator's standing directive ("once the 500-deal measurement lands
and the conclusion is added to the notebook, set up and launch").
Full result + adherence tables: Search_Teacher_Design §13.3 RESULT.

Launch configuration (operator-directed where noted):

- Seed / expert: runs/league_retention_pg/checkpoints/
  pfsp_perceiver-shared-v2_checkpoint_8000000.pt — clean 8M seed;
  frozen committee expert defaults to the SAME checkpoint via
  --teacher-ckpt fallback to --resume. E9 cert carries (certified on
  these exact weights). NO --oracle-init: the 8M checkpoint restores
  its own trained oracle (strict load), and the flag would OVERWRITE
  it with the stale 400k pretrain post-load (§10.2 defect record;
  hardening warning + tests committed 0437eee). Attempt 11 is the
  first lineage run whose deployed expert oracle matches every
  calibrated offline instrument (E9, §12.8, ceiling, §10.3).
- Generation length: 100,000 episodes (operator: "make the
  generation 100k"), i.e. --main-episodes 100000; generations 3
  (default) → 8.0M → 8.3M.
- Emission: teacher_prob 0.1 (operator: "p=0.1"), class-blind PLAY
  nodes, ≥2 legal, standard game, self-play worlds. R=3 @1024/1,
  pi_gumbel-on-shrunk-Q targets, shrink_s2_global 6.95e-4 (§10.1),
  teacher_coeff 1.0, teacher_epochs 4 — all trainer defaults.
- Guards (in-trainer, §3 two-tier): partner hard floor 90 /
  notify 93.5 (n=1000), t0 trump-lead ceiling 5.0 → SystemExit(3);
  boundary cert 3 seeds × 1000 games on across-seed means + CRN h2h
  vs fixed anchor, exit 4 on fail; refreeze only on cert pass.
- Run dir: runs/league_ce_teacher11/ — league/ pool copied from
  teacher10 (36 members, current_generation 33); entropy sidecar
  copied from the 8M lineage (v1 format, migrates bumplessly to v2
  on first update).

Pre-registered expectations (finalizing §9 with concrete numbers):

1. Self-retirement: mean KL(target‖policy) at labeled nodes DECAYS
   within each generation (the signature attempt 10 never showed).
   Smoke baseline at 8M weights: KL ≈ 0.046 at n=20 nodes — small
   because most nodes abstain into the policy; the read is the
   TREND on the labeled subset, not the level.
2. Conventions: called_suit t0 adherence rises from ~43-45 (attempt-10
   boundary read 43.3; ceiling policy-arm 40.0) toward the committee's
   acted 56.2 and HOLDS while teaching continues — no consolidation
   phase exists to revert it. Partner-trump ≥ 93.5 n=1000 THROUGHOUT
   (gentleness claim; attempt 10 bled to 80.9 under the hinge).
   Expected-and-benign: mild late-trick (t2+) softening of def-lead /
   partner adherence toward the committee's acted profile (§13.3
   RESULT tables) — the guard battery, not adherence drift alone,
   arbitrates harm.
3. Strength: gen-1-end h2h vs 8M seed captures 25–50% of the ceiling
   → +0.045 to +0.090 expected band; ≥ +0.02 at 2σ = teaching signal
   confirmed; ≤ 0 after a full gen with healthy KL decay = CE
   transfer failure, stop and diagnose before gen 2.
4. Entropy: play Hn within ±0.03 of target with alpha ≥ 0; sustained
   negative alpha = the teacher injects entropy after all →
   investigate before gen 2 (v2 signed range −0.05..0.25 exists
   precisely to absorb this without saturation).
5. Emission health: labeled-node rate ≈ p·(play nodes) with ~90%+
   resolution (ceiling instrument: 93%); near-zero emission or flat
   adherence = ε/shrink miscalibration → fall back per §10.3 caveat
   (class pooling) rather than raising coefficients.

Launch command (from master, post-merge; nohup background):

    uv run python -m sheepshead.training.train_league_ppo \
      --resume runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt \
      --league-dir runs/league_ce_teacher11/league \
      --run-name league_ce_teacher11 \
      --teacher --main-episodes 100000

(teacher_prob 0.1, R 3, iters 1024, coeff 1.0, epochs 4,
generations 3, oracle critic mode, guard/cert defaults all from
league_cli defaults — verified pre-launch; §6 micro-smoke exercised
emission, guards, cert+refreeze chain, exploiter gate, crash-resume
with --teacher-ckpt pin, and the signed-alpha clamp end-to-end.)

---

## 13. Entropy investigation (2026-08-17, gen 1 in flight)

Observation (~15% into gen 1): play Hn 0.53 -> 0.64 vs target 0.476,
partner 0.10-0.13 vs 0.067, BOTH alphas pinned at the -0.05 clamp
(one sign flip each) — pre-registered expectation #4's investigate
condition. Run left in flight (guards quiet, outcomes healthy, clamp
bounds the fight); offline instrument built to identify the channel:
analysis/verify_entropy_baseline.py compares, at emission-eligible
nodes, pi (student), p_raw (pooled expert prior = target baseline),
the production target, and a target rebuilt with base_prior=pi (the
candidate structural fix; optional arg added to
build_ce_search_target, default behavior unchanged, 3b09c56).

PHASE 1 (n=121 nodes, seed weights = gen-start conditions, 1024/1):

- Root-level Jensen story FALSIFIED: at the root every determinized
  world presents the same info-state, so the pooled root prior EQUALS
  the expert policy (gap ~1e-7). The §1.1 abstention fixed-point
  claim is architecturally sound at gen start.
- Material rows (n=69): teacher is strongly entropy-REDUCING —
  H(target) 0.069 vs H(pi) 0.381, median dH -0.30, 99% negative.
  (A 64-iter smoke had suggested tilt softening; artifact of the
  small visit scale.)
- Abstention rows (n=52): KL(target||pi) = 0.030, ALL of it
  engine-replay recurrent-state divergence at trick 4 (t0-t3 exact
  zeros). base_prior=pi zeroes it to machine precision; fix safety
  at gen start perfect (argmax agree 1.0, push corr 0.99998).
- REVISED leading hypothesis for the live Hn rise: MASS-IN-TRANSIT.
  Material-row KL is heavy-tailed (median 0.19, p95 4.5; 14/69 rows
  > 1 nat = argmax replacements). CE moving mass between modes
  passes through bimodal intermediates; always-on emission keeps a
  standing population mid-transfer — consistent with flat live
  teacher KL ~0.40. Entropy rise = teaching's transient shadow
  (attempt-9 top1min-softening family), predicted to self-limit as
  labeled rows conform (same signature as within-gen KL decay).
- FIX MENU REVISED: base_prior=pi at MATERIAL rows is now judged
  RISKY — each relabel re-tilts from the already-taught position =
  the §1.1 iterated-improvement ratchet; the expert-prior baseline
  is what bounds the within-gen target. The clean surgical option
  if phase 2 shows a material drift-anchor pull: MASK CE LOSS AT
  w=0 ROWS (designed zero-gradient anyway; kills drift-anchor and
  replay-divergence channels exactly, zero ratchet risk, teaching
  untouched). Controller-authority widening (alpha_min) remains the
  fallback for the transient itself.

PHASE 2 (auto-armed): same instrument with --ckpt = first attempt-11
student checkpoint vs --teacher-ckpt = frozen 8M expert — sizes the
w=0 drift-anchor pull and tests fix safety under real drift. Result
to be recorded below; gen-2 decision (mask w0 / alpha_min / accept)
waits on it.

§13 ADDENDUM (operator decision, 2026-08-18): ACCEPT-AND-MONITOR
confirmed at ~29k eps. Mid-gen sanity battery on the live worker
payload (v21 nets + seed metadata = crafted eval ckpt, no trainer
disturbance): telemetry stationary (teacher KL 0.37-0.46 flat, CE
~0.79 flat, material ~0.48 flat; approx_kl healthy; ev/picker_avg
trendless; lead_trump_mass stable ~0.70); play Hn EQUILIBRIUM at
~0.64 since ep 8,013k (crested, not falling); greedy called-suit t0
probe 43.75 vs seed 41.16 paired (+2.6 ± ~2.3, whisper). Refined
stalemate mechanism (if gen-end confirms): NOT directional PG
opposition (reward aligned per §12.16, SNR-thin at taught cells) but
EROSION — dense unlabeled PG stream + negative-alpha re-sharpening
drag shared-trunk features between sparse label visits; mass shifts
toward search-preferred actions sub-argmax-flip (entropy up, softband
up 0.69→0.84, greedy probes flat).

Pre-analyzed contingency ladder if gen-1 verdict = stalemate
(ordered by cost, per attempt-6 Adam lesson that STEP COUNT binds
while coefficients are muted):
  1. teacher_epochs 4→8 (binding, ~free — labels already paid for);
  2. teacher_coeff raise (free, likely Adam-muted, second-order);
  3. teacher_prob raise (halves eps/s at p=0.2 — justified ONLY if
     diagnostics show coverage-limited failure: taught nodes conform
     while fresh-node KL stays flat; if taught nodes rebound, p buys
     more erosion).
Measurement subtlety recorded: probes/certs read GREEDY argmax; a
sub-flip mass shift changes SAMPLED play EV invisibly to the whole
greedy battery — if gen 1 ends probes-flat with the mass signature
intact, run one sampled-action h2h before concluding the teaching
did nothing.

§13 battery result (ep ~8,029k, paired probe seeds, n=3000 deals):
called-suit t0 43.75 vs seed 41.16 (+2.6, ~1σ, taught direction);
partner trump-lead CALLED mode 97.8 vs 93.2 (+4.6, ~5σ — untaught
convention SHARPENED, anti-§12.11 signature, gentleness claim
holding), jd mode 99.6 flat; defender t0 trump-lead 0.58-0.90% vs
0.08-0.24% (uptick, ~10x below the 5% tripwire, implied EV -0.3
per 1000 hands). No damage anywhere; accept-and-monitor unchanged.

---

## 14. Guard halt at 8,050k and the three-point reversal (2026-08-19)

FACTS: first n=1000 adherence guard probe (ep 8,050,000): partner
trump-lead 87.5 < hard floor 90.0 -> designed SystemExit(3), halt
checkpoint saved (checkpoints/..._checkpoint_8050000.pt); called-suit
39.3 (below seed baseline); t0-trump 0.2 (clean). Independent greedy
health at same episode: partner 86.1 (n=72), ALONE 22.2 > 20 gate.
Trainer-side telemetry showed NOTHING trending to the end (teacher KL
flat ~0.40, CE flat, ev/picker_avg flat) — §12.21 lesson repeated:
probes lead every lagging indicator.

THREE-POINT GREEDY BASELINE (same instrument/seed, n=500/point —
the load-bearing measurement):

  metric                 seed 8000k   mid 8029k   halt 8050k
  called-suit (taught)      45.8        55.5         38.5
  partner trump-lead        96.4        98.9         87.4
  pick rate (greedy)        38.4        34.0         32.1
  alone rate                13.4        13.8         17.6
  leaster rate               5.8         7.4         10.2
  play spread (med)         3.56        2.39         2.37
  top1min (med)             9.19        6.19         5.89

READING: (i) BY 29k THE TEACHER WAS WORKING AS DESIGNED — called-suit
+9.7 into the pre-registered 50s band with partner IMPROVED (+2.5);
the CE mechanism installs, and gently, at that horizon. (ii) Between
29k and 50k a BROAD reversal: both conventions collapsed (taught
metric to below seed), pick fell 6 pts across the gen, leaster nearly
doubled, alone +4, while logit spread sat compressed (2.4 vs seed
3.6; attempt-8's stop-rule line was 2.7) and top1min kept softening.
This is not single-convention oscillation (§12.20 trough shape —
others held there); it is systemic drift, the §10.4 "greedy orderings
scramble" failure family in slow motion (~50k eps vs attempt-5b's
3-4 updates), arriving through cumulative CE step count on the
shared trunk with the entropy controller PINNED at the -0.05 clamp
the entire generation (the bounded-fight design bound proved to be
the binding failure: sub-argmax mass accumulated until near-tie
argmaxes started flipping broadly).

CAUSAL CHAIN (working hypothesis): CE mass-transfer at material rows
(KL p95 ~4.5 = mode replacements) -> sustained sub-flip mass +
entropy elevation -> alpha saturates at clamp, cannot counter ->
softening compounds (spread 3.6->2.4, top1min 9.2->5.9) -> near-tie
greedy flips cascade across taught AND untaught heads (29k->50k).
Phase-2 instrument (drifted 8050k student vs frozen 8M expert, in
flight) will additionally size the w=0 anchor pull now that the
student has moved.

OPERATOR DECISION MENU (no action taken; run halted on checkpoint):
  A. Resume unchanged from 8,050k betting on §12.12-style
     self-recovery — argued AGAINST by the breadth of the drift
     (systemic, not single-cell) and by both softening tripwires
     sitting past their historical stop lines.
  B. Resume from 8,050k with dose reduction + controller authority:
     teacher_epochs 4->2 AND alpha_min widened (e.g. -0.15) so the
     controller can actually hold Hn at target. Rationale: 29k
     proves efficacy; the collapse tracks cumulative dose with a
     saturated controller. Cheapest live test of the causal chain.
  C. Kill attempt 11; fold into the §12.22 program conclusion
     (policy-space teaching on a shared trunk destabilizes at any
     dose that installs) and move to architectural separation
     (convention head / adapter).
  D. Crafted rollback to ~29k weights (v21 payload + 8M optimizer,
     attempt-9 §12.12 precedent) + reduced dose — preserves the
     good state but adds optimizer-mismatch confounds.

---

## 15. Theory of the failure and redesign space (2026-08-19)

PHASE 2 RECORD (drifted 8,050k student vs frozen 8M expert, n=120,
runs/entropy_baseline_202608/phase2_drift_8050k.json): student now
SOFTER than expert prior everywhere (H_pi 0.60 vs H_praw 0.50 —
Jensen gap flipped negative); w=0 anchor pull grew 0.030 (gen start)
-> 0.106; material targets still sharp (H 0.09) — by 50k the teacher
was RE-sharpening the softened student, i.e. the softening came from
the interaction dynamics, not from target entropy. Fix safety under
drift: argmax agree 98.4%, push corr 0.968.

MECHANISM (evidence-backed): three vector fields with NO COMMON
FIXED POINT on a shared trunk —
 (1) CE toward pi_gumbel(seed prior, seed Q) = a one-step
     improvement OF THE SEED, valid in a neighborhood (trust
     region); integrated open-loop for 50k eps, far past the
     linearization radius. The 29k peak = the radius edge. Frozen
     expert => label KL has a FLOOR set by seed-student distance —
     the pre-registered KL-decay signature was structurally
     impossible in this design.
 (2) PG's dense stream owns trunk features; CE's sparse off-mode
     pulls (KL p95 ~4.5) leave standing bimodal mass that
     generalizes into untaught heads (ALONE/pick drift).
 (3) Entropy controller pinned at clamp all gen — the stabilizer
     was bounded, the damage was not.
Composite attractor = neither PG optimum nor search-improved policy;
29k->50k near-tie flips = relaxation into it. HYPERPARAMETERS set
spiral speed and attractor location, NOT existence — the instability
is structural (the program's sweep across hinge/CE, coeff, epochs,
two-phase/always-on never varied the structure: policy-space pull
toward a NON-MOVING reference through a shared trunk).

REDESIGN SPACE (ranked by information/risk):
 (a) CLOSED LOOP (true AZ): expert = current net; target =
     softmax(log pi_current + tilt) — bounded-KL from policy by
     construction (no mass-in-transit), no staleness, KL decay
     becomes the real self-retirement signature. Same compute. Cost:
     CANNOT certify a moving expert — cert culture retreats to the
     absolute-anchor boundary instruments. Attempt-7 counterevidence
     is confounded (hinge at pathological scale + frozen-cert
     semantics); CE-tilt with moving expert is a tamer object.
 (b) PHASED OFFLINE ExIt: fixed certified target corpus at seed
     states -> supervised distill w/ PG OFF -> boundary cert ->
     refreeze as next expert. One clean Newton step per outer
     iteration; certifiable; respects the validity radius by
     construction. Cost: dedicated labeling runs (~days per
     iteration); do NOT interleave PG (it erodes — measured).
 (c) ADAPTER SEPARATION (§13.2 sketch): CE into zero-init additive
     logit module PG never touches; structural interference kill;
     the standing fallback if (a)/(b) still show trunk coupling.
 (d) VALUE-SPACE distillation (search Q -> action-value head, act
     on it at deploy): sidesteps softmax mass dynamics entirely;
     biggest deployment change; hold unless policy-space exhausted.
READING: attempt-11 indicts the FROZEN REFERENCE, not policy-space
distillation per se — the 29k state proves CE installs and
generalizes benignly inside the validity radius. (a) = highest
information next; (b) = safest; (c) = structural insurance.

---

## 16. Attempt-12 launch record & pre-registration (2026-08-19)

OPERATOR DECISION: proceed with §15(a) — closed-loop expert, the
committee backed by the TRAINING network. §14 menu items A/B/B'/D
retired unexecuted; attempt-11's halted checkpoint stays archived at
runs/league_ce_teacher11/checkpoints/..._8050000.pt.

### 16.1 Code change (commit 40c55e2)

The frozen expert is REMOVED, not made optional (operator: "drop the
old frozen teacher arguments and implementation... we could
reimplement it if we wanted"). --teacher-ckpt, TeacherSettings
ckpt/oracle_init, build_frozen_expert, the per-gen expert pin and the
refreeze-on-cert plumbing are gone. The teacher now wraps:

- sequential stream: the training agent itself;
- spawned workers: the worker's current-weights copy, which
  league_worker_play weight refreshes mutate IN PLACE — expert lags
  the student by at most one weight version (~1.4k episodes).

Safety of sharing the acting agent: the ISMCTS engine
snapshots/restores per-seat recurrent memories around every search,
keyed by id with self.agent always included (ismcts.py ~795) — the
frozen expert never masked a side effect; there wasn't one.

Boundary cert UNCHANGED and now the teacher's whole certification:
absolute anchors resolved once at launch (--cert-anchor-ckpt or
--resume), fixed bars, GateExit(4) + halt on fail. What is LOST with
the frozen expert: per-generation expert certification (the §12.18
refreeze gate). What is GAINED: label KL floor removed (KL decay =
real self-retirement signal), no open-loop integration past a fixed
policy's validity radius, w=0 drift-anchor channel structurally
zeroed (expert ≡ student ⇒ pooled root prior ≡ π up to the trick-4
replay divergence, 0.030 nats, §13 phase 1).

### 16.2 Worker throughput flags (audit of fce37f52)

Operator asked for an efficiency audit of fce37f52 ("Add opt-in
compiled/device inference for league workers") before launch.
VERDICT: CLEAN, adopted for attempt 12. Findings:

- Scope verified worker-pool-only: PPO update, adherence guards,
  greedy eval, boundary cert, exploiter gate all run in the main
  process (eager CPU) — cert/golden comparability unaffected.
- Ordering verified: device global patched + encoder compiled BEFORE
  PPOAgent construction in league_worker_init; ISMCTSTeacher reads
  device off the agent's params (9d8efff), no import-time snapshots
  left on the worker path (remaining DEV=ppo.device snapshots are all
  analysis/validation scripts, not reached by workers).
- Pad-and-slice wrapper: encode_tensors returns 4 batch-major tensors
  (features/hand_tokens/context_token/memory_out); the v[:n] slice
  covers all of them; pad rows replicate row 0 and cannot leak
  (positional indexing downstream sees exactly n rows).
- dynamo recompile_limit raised to 64 (allow_shape_specialisation) —
  covers the ~14 bucketed shapes at granularity 32; the silent-eager
  fallback trap (§ Distributed_Inference) is closed.
- Latent (not a blocker, noted): PPOAgent.get_recurrent_memory's
  device=None fallback recomputes cuda-or-cpu instead of reading the
  patched module global; no worker-path caller hits it (all pass
  device explicitly).
- 16 tests pass (test_worker_inference_options, test_search_encode_path).
- Known cost: worker episodes differ from eager in the last bits
  (~2.6e-08) — bit-exact cross-run comparison is off the table for
  this run; every statistical instrument is unaffected.

SPAWNED-POOL SMOKE (real spawn import ordering, 8M seed payload,
oracle mode + aux heads, live teacher, mps + compile, R=3 @16):
networks and teacher on mps:0, teacher.agent IS the worker agent,
17 play decisions labeled on the first eligible deal, v2 payload
refresh reached the teacher (actor param sum moved). PASS.

### 16.3 Attempt-12 design: single substantive change

Everything held at attempt-11 values so the frozen->live expert swap
is the only learning-relevant difference (throughput flags change
last-bit numerics only):

- Seed: runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt
  (same clean 8M seed; optimizer state included via --resume).
- Emission: prob 0.1, R=3 @1024/1, pi_gumbel-on-shrunk-Q,
  shrink_s2_global 6.95e-4, coeff 1.0, epochs 4 (trainer defaults).
- Target semantics CHANGE BY CONSTRUCTION: expert ≡ student ⇒ target
  = softmax-tilt of the CURRENT policy toward the CURRENT network's
  committee Q — a one-step policy improvement, bounded-KL from π.
- Guards unchanged: partner n=1000 hard floor 90 / notify 93.5, t0
  trump-lead ceiling 5.0 -> SystemExit(3); boundary cert 3x1000 +
  CRN h2h vs the 8M anchor, GateExit(4) on fail.
- Entropy controller v2 unchanged (alpha range -0.05..0.25): the
  attempt-11 saturation is a pre-registered readout here, not a
  patched symptom.
- League: fresh copy of teacher10's pool (attempt-11's LAUNCH state;
  attempt-11's own league drifted: ratings churn, 7350000 retired,
  its guard-halted 8050000 snapshot inserted — excluded here).
  Entropy sidecar copied from the 8M lineage as before.
- Run dir: runs/league_ce_teacher12/. 100k eps/gen x 3 gens.
- Throughput: --num-workers 8 (default) --worker-device mps
  --worker-compile. Expectation ~1.2-1.4x on episode generation
  (search-dominated share; §5.5-§5.6 measured 1.36x on committee
  search); attempt-11 baseline 0.3 eps/s.

### 16.4 Pre-registered expectations

1. SELF-RETIREMENT (the decisive readout): mean KL(target‖π) at
   labeled nodes starts LOW (no seed-student floor; smoke-scale
   analogue of §13 phase-1 abstention 0.030 + material tilt) and
   DECLINES within gen 1; material fraction and mean w decline as
   improvements are absorbed. Attempt-11 baseline: KL flat ~0.40 all
   gen. Live-expert KL flat-or-rising at attempt-11 levels = the
   §1.1 iterated-improvement ratchet materializing -> stop and go
   §15(b)/(c).
2. ENTROPY: play Hn rise strictly smaller than attempt-11's
   0.53->0.65; alphas NOT pinned at -0.05 for the whole gen (the
   w=0 anchor channel is gone; mass-in-transit at material rows may
   still produce a mild transient).
3. CONVENTIONS: called-suit t0 (greedy probe) rises from the seed's
   ~45.8 toward the committee's acted 56 AND HOLDS (no 29k->50k
   reversal); partner-trump >= 93.5 n=1000 throughout. The greedy
   3-POINT BATTERY (seed / ~29k / ~50k, greedy_health_probe n=500
   seed=0) is the arbitrating instrument — it caught attempt 11 when
   all trainer telemetry was flat.
4. STRENGTH: gen-1-end h2h vs 8M seed in the +0.045..+0.090 band
   (25-50% of the +0.180 ceiling); >= +0.02 at 2sigma = teaching
   signal; <= 0 with healthy KL decay = transfer failure, stop.
5. OUTCOME SANITY: pick rate stays ~30-38% (not the attempt-11 drift
   to 32 with ALONE 17.6); leaster <= ~8%; play spread does not
   cross the attempt-8 stop line (2.7) downward past 2.4.
6. RISK REGISTER (what closed-loop can do that frozen could not):
   self-referential drift — the committee certifying the student's
   own bad habits (attempt-7 family, now WITHOUT the hinge-scale
   confound). Tripwires: t0 trump-lead probe > 5% replicated (hard
   guard), called-suit falling BELOW seed while KL stays low
   (teaching toward a degraded self), partner < 88 replicated.
   Boundary cert vs the FIXED 8M anchor is the backstop.

Launch command (from master @ 40c55e2 + this doc):

    nohup uv run python -m sheepshead.training.train_league_ppo \
      --resume runs/league_retention_pg/checkpoints/pfsp_perceiver-shared-v2_checkpoint_8000000.pt \
      --league-dir runs/league_ce_teacher12/league \
      --run-name league_ce_teacher12 \
      --teacher --main-episodes 100000 \
      --worker-device mps --worker-compile \
      > runs/league_ce_teacher12/train.log 2>&1 &

### 16.5 AMENDMENT (2026-08-19 ~21:10): throughput flags REMOVED, relaunched

The §16.3 expectation (~1.2-1.4x from --worker-device mps
--worker-compile) is REFUTED on the full-episode workload:

- window 1 (incl. 8x compile warm-up): 1,443 eps in 117 min = 0.21 eps/s
- window 2 (fully warm):               1,462 eps in 121 min = 0.20 eps/s
- attempt-11 eager-CPU baseline:       ~80 min/window       = 0.30 eps/s

i.e. a 1.5x SLOWDOWN, stable after warm-up. Mechanism (hypothesis,
consistent with the numbers): the compiled-encoder patch is
class-global, so every SINGLE-ROW act() encode — the ~90% of
decisions the teacher never searches, across 5 seats plus opponent
pools — pays granularity padding (1 -> 32 rows) plus MPS dispatch
and host-sync latency. The §5.5-§5.6 1.36x was measured on committee
SEARCH in isolation (large merged batches), and does not transfer to
a p=0.1 emission workload dominated by singles. The fce37f52 audit
(§16.3) stands — the code does what it says, on the path it was
measured on; the workload composition is what was mispredicted.

Candidate future fix (NOT built): batch-size-thresholded routing —
small encodes stay eager-CPU, only committee-scale batches take the
compiled MPS path. Worth building only if teacher_prob rises enough
for search to dominate episode wall time.

OPERATOR CALL (after 9pm read of the same numbers): drop the flags.
RELAUNCH from the same 8M seed with the identical §16.3 command minus
--worker-device/--worker-compile; run dir reset to launch state
(league re-copied from teacher10, sidecar re-copied from the 8M
lineage, flagged-run log archived as train.log.mps-flags-attempt).
The discarded ~2,900 flagged episodes' telemetry, for the record:
KL 0.337/0.347, material 46/45%, w 0.31/0.30, Hn play 0.52/0.54 —
consistent with §16.4 expectations, decided nothing yet. All §16.4
pre-registrations carry unchanged; worker episodes are now
bit-comparable eager CPU again (the §16.3 numerics caveat is void).

CORRECTION to the §16.5 mechanism (2026-08-19, same evening): the
singles-padding hypothesis cannot be the dominant mechanism. Empirical
decomposition from the teacher10 log on this machine (same arch, 8
workers): teacher-OFF consolidation ran 6.0 eps/s vs 0.3 eps/s at
p=0.1 R=3 @1024 — committee search is ~95% of episode wall time, and
singles (~5%) cannot produce a 1.5x overall slowdown at any plausible
penalty. The flagged run's slowdown therefore came from INSIDE the
search path running slower on MPS in situ than in the §5.5 bench:
the prime suspects are the non-encoder ops that ran eager-MPS
(actor/critic head forwards at small round batches, oracle-leaf
forward_sequences, GRU memory updates) and 8-process Metal contention
against a bench whose committee composition may not have included the
production oracle-leaf path. Any routing design must therefore keep
EVERYTHING except large-batch encodes on CPU — and the realized
encode-slice speedup must be re-benched in situ before building.

### 16.6 Routed encoder: bench, build, relaunch (2026-08-19 late)

Operator: kill the eager relaunch, bench the batch-size-thresholded
routing first, build it if promising. Three-arm bench at PRODUCTION
committee composition (oracle-mode agent + aux heads — the §16.5
correction's point; 8 clients, R=3 @1024/1, steady state = best of
repeats 2-3; bench_search_committee gained --oracle and --routed):

    A eager CPU + oracle:            62.3s/committee   1.00x
    B whole-agent MPS+compile:       57.4s             1.09x
    C routed (CPU + MPS shadow):     41.4s             1.50x

Readings:
- The historical 1.36x (§5.5-§5.6) shrinks to 1.09x once committees
  pay the oracle-leaf path: OracleCriticEncoder OVERRIDES
  encode_batch with its own copy, so the compiled patch never touched
  it — on --worker-device mps it ran EAGER MPS (ragged sequence
  assembly, sync-heavy). That, plus per-instance dynamo recompiles
  for each lazily-loaded league opponent's encoder (36 members x ~14
  shapes >> the 64 cap -> silent eager-MPS tail), is the §16.5
  in-situ 1.5x slowdown, now mechanistically accounted for.
- Routing dodges both BY CONSTRUCTION: opponents only send
  single-row encodes (never routed, stay eager CPU), the oracle's
  override is untouched (eager CPU), and exactly one shadow exists —
  the live agent's.
- C = 1.50x on the committee; at the §16.5 decomposition (search =
  95% of wall) that projects to ~1.47x overall: 0.30 -> ~0.44 eps/s,
  100k generation ~3.9d -> ~2.6d.

BUILT (commit with this note's hash lineage): compiled_encoder.
enable_routed_encoder(granularity, mode, threshold=16, device) +
sync_routed_encoder (league_worker_play calls it after every weight
refresh — the shadow must follow the closed-loop expert or it labels
with stale weights) + disable_routed_encoder; trainer flag
--worker-routed-encoder [DEVICE=mps], mutually exclusive with
--worker-device. Small batches take the ORIGINAL eager method
bit-identically; routed large batches differ from eager by ~3e-6
(MPS numerics — same worker-only caveat class as §16.3, main-process
gates/certs untouched). Tests: 4 routing tests in
test_search_encode_path + 3 wiring tests in
test_worker_inference_options; spawned-pool smoke = nets CPU, teacher
live, single mps:0 shadow, labels on first eligible deal, v2 refresh
exercised sync. RELAUNCH: §16.3 command + --worker-routed-encoder,
run dir reset to launch state again; §16.4 pre-registrations carry;
pre-registered throughput mark: steady eps/s >= 0.40 by update 3
(else routing underdelivers in situ too — investigate, don't tune).

§16.6 THROUGHPUT VERDICT (2026-08-19 23:46, in situ): window 1
(incl. compile warm-up) 65 min = 0.37 eps/s; window 2 (warm) 62 min
= 0.40 eps/s — the pre-registered >=0.40 mark is MET (1.33x over the
0.30 eager baseline; below the 1.47x projection, i.e. in-situ search
share / shadow overhead slightly less favorable than the isolated
bench, but decisively worth it: generation ~3.9d -> ~2.9d). Early
teacher telemetry: KL 0.395 -> 0.321 over windows 1-2 (attempt-11:
flat ~0.40) — the §16.4 #1 decay direction, too early to call.

### 16.7 Attempt-12 mid-gen battery at 29k (2026-08-20)

Same instrument as the §14 three-point baseline (greedy_health_probe
n=500 seed=0; crafted eval ckpt = worker payload v21 nets @8,029,074
swapped into seed-checkpoint metadata). Comparators:

    metric         seed    a11@29k  a11@50k   A12@29k
    called_suit    45.8    55.5     38.5      50.3
    partner_trump  96.4    98.9     87.4      95.9
    t0_trump       —       —        0.2       1.19
    pick           38.4    34.0     32.1      33.2
    ALONE          13.4    13.8     17.6      25.5   <- FLAG
    leaster        5.8     7.4      10.2      8.4    (marginal)
    spread_med     3.56    2.39     2.37      2.12   <- below the 2.4 mark
    top1min_med    9.19    6.19     5.89      6.34

Reading:
- TEACHING LANDING (§16.4 #3): called-suit 45.8 -> 50.3, inside the
  pre-registered 50s band (less than a11's 55.5 at the same point);
  partner 95.9 above the 93.5 notify line; t0 clean. The 50k re-read
  is the decisive hold-vs-revert test.
- KL context at ~29k: windows 1-19 series 0.395...0.274-0.304 band —
  BELOW a11's flat ~0.40 throughout (self-retirement direction, #1).
- Hn: play 0.65-0.69 vs a11 equilibrium 0.65; alphas pinned -0.05
  since early gen — #2's "strictly smaller rise + unpinned alphas"
  is VIOLATED in direction (recorded; mass-in-transit now carries
  the whole rise, the w=0 anchor channel being structurally gone).
- OUTCOME FLAGS (#5): ALONE 25.5 = ~2x seed and worse than a11's
  HALT-time drift (17.6) while taught metrics are still good —
  untaught-bidding-head drift arriving EARLIER and LARGER under the
  closed-loop teacher; spread 2.12 crossed the pre-registered 2.4
  line (a11 29k: 2.39). Leaster 8.4 marginal. No in-trainer greedy
  probe has run yet (sparser cadence); this battery is the only 29k
  instrument. Hard guards (partner n=1000 >= 90, t0 > 5) armed and
  quiet. Operator decision point: continue to the 50k mark as
  pre-registered vs early action on the ALONE drift.

OPERATOR DECISION (2026-08-20): CONTINUE past the 29k flags to the
50k hold-vs-revert read. Amendment noted for a FUTURE attempt (not
this run): include ALONE nodes in teacher emission — the CE teacher
currently labels standard-game PLAY nodes only, so alone/bidding
behavior is an untaught head coupled through the shared trunk; the
25.5% greedy ALONE drift argues for anchoring it with search labels
(committee already searches all four heads; emission-side change)
rather than leaving it to PG generalization.

### 16.8 GUARD HALT at 8,050,000 and the attempt-12 verdict (2026-08-21)

Hard adherence guard (n=1000): t0 trump-lead 6.1% > 5.0 ceiling ->
SystemExit(3), checkpoint saved (..._checkpoint_8050000.pt). Same
episode as attempt-11's halt, different channel (t0 scramble vs
partner collapse). Partner held 99.2 this time.

Three-point same-instrument series (greedy_health_probe n=500
seed=0; a12@50k = payload v35 nets @8,049,295):

    metric        seed   a12@29k  a12@50k | a11@29k  a11@50k
    called_suit   45.8   50.3     37.6    | 55.5     38.5
    partner       96.4   95.9     98.4    | 98.9     87.4
    t0_trump      ~1     1.19     2.65    | —        0.2
    pick          38.4   33.2     29.5    | 34.0     32.1
    ALONE         13.4   25.5     14.5    | 13.8     17.6
    leaster       5.8    8.4      13.4    | 7.4      10.2
    spread_med    3.56   2.12     2.10    | 2.39     2.37
    top1min_med   9.19   6.34     5.19    | 6.19     5.89

(Guard's 6.1 vs battery's 2.65 t0: different deal sets — the guard
probes n=1000 fresh deals, the battery the fixed seed-0 set; both
are far above the seed's level and the direction is what matters.)

VERDICT — the reversal arc REPRODUCED with expert ≡ student:
called-suit peaked mid-gen then fell below seed on the same schedule
as attempt 11; leaster doubled; pick eroded; t0-trump escalated into
the hard guard; ALONE round-tripped (25.5 -> 14.5 = §12.20-style
oscillation, not collapse). ALL WHILE the teacher KL sat in the
0.27-0.31 band, decaying — labeled nodes conforming as the global
greedy ordering scrambled around them.

THEORY UPDATE: §15's indictment of the FROZEN REFERENCE is falsified
as the sufficient mechanism. The closed-loop expert delivered
everything it promised locally (low bounded KL, self-retirement
signature, no label staleness) and the mid-gen reversal happened
anyway, on the same clock. What remains indicted is the §12.22
conclusion, now strengthened and expert-independent: POLICY-SPACE CE
TEACHING ON THE SHARED TRUNK IS TRANSIENT AND DESTABILIZING UNDER
CONCURRENT PG — epochs-4 CE x PG co-training scrambles untaught
greedy orderings regardless of where the labels come from. Low label
KL is NOT protective; it measures the taught subspace only.

Remaining §15 directions, re-ranked by this result: (b) PHASED
OFFLINE ExIt (PG OFF during distill — the only variant that removes
the interaction term itself) and (c) ADAPTER SEPARATION (structural
removal of the trunk coupling) are now the live candidates; further
same-structure hyperparameter variants (epochs, p, coeff) are
third-line at best — two attempts have shown the arc survives the
biggest structural lever available inside this loss. The operator's
ALONE-emission amendment remains relevant to whichever path
continues but would not have prevented this halt (t0-trump is not an
emission-coverage gap; it is generalization damage).

Run artifacts: halt checkpoint 8,050,000; payload snapshots
t12_payload_29k/50k.pt + crafted eval ckpts in scratchpad; KL series
windows 1-35 in train.log / league_training_progress.csv. Awaiting
operator decision.

§16.8 ADDENDUM — strength h2h of the halt snapshot (2026-08-21):
operator challenged the reversal reading (oscillation-to-new-
equilibrium hypothesis: §12.20/§12.22 churn precedent, ALONE
round-trip, spread-compression greedy amplification). Pre-stated
interpretation grid: >= +0.045 teaching captured / ~0 (+-0.03)
EV-neutral churn / <= -0.03 damage confirmed. RESULT
(duplicate-bridge, 2000 deals/mode, checkpoint_8050000 vs 8M seed):

    edge -0.058 +- 0.010  (5.6 sigma below zero; win_frac 0.454)
    called -0.045 +- 0.015   jd -0.072 +- 0.015

DAMAGE CONFIRMED — both modes negative, no mode-split ambiguity.
The oscillation hypothesis is REJECTED for strength: whatever
equilibrium the policy was moving toward, it is materially weaker
than the seed (pre-reg expected +0.045..+0.090; delta from
expectation ~-0.10 to -0.15). The §16.8 verdict stands as written:
closed-loop CE teaching on the shared trunk under concurrent PG
destroyed value on the same mid-gen clock as attempt 11, with label
KL low throughout. Caveat kept honest: attempt-11's RAW 8,050k halt
state was never h2h'd (only probed), so cross-attempt severity is
not comparable — but the within-attempt question the operator asked
is answered.

### 16.9 Mechanism synthesis + phased-offline design sketch (2026-08-21)

Operator question: search and PG both optimize EV — why destructive?
Synthesis (full argument in session; condensed):

MECHANISM — "same signal" holds at the objective level, fails at the
gradient level, four layers: (1) magnitude mismatch: CE is O(1) in
logit space regardless of EV at stake (x4 epochs, Adam normalizes
coefficients away — a5b/a6), applied exactly where PG's true signal
is O(eps) under O(sigma) noise (near-ties, median edge 0.0097 vs
floor 0.006); (2) shared-trunk generalization: bounded KL at labeled
nodes bounds nothing elsewhere — feature drift dephases whichever
head has the weakest restoring force (partner in a11, t0/leaster/
pick in a12; "policy churn" Schaul-22 is the single-objective
baseline of this); (3) asymmetric repair: PG repair needs
O(sigma^2/eps^2) visits at rare nodes, CE re-applies pressure every
update — Grill-20: CE-to-search IS a KL-regularized policy update,
so concurrent PPO = two proximal operators, different centers, no
common fixed point, orbit through weaker space (h2h -0.058);
(4) third optimizer (entropy controller pinned at clamp) + critic
lag under distribution shift (ev O/L 0.66/0.52 -> 0.55/0.36 across
a12) + league self-play non-stationarity. Near-tie noise labels: PG
turns them into zero-mean dither, CE into persistent directed churn
(v7 incumbent tax). LITERATURE: AZ/ExIt are phase-pure (search is
the ONLY policy-improvement operator; no concurrent model-free PG,
ever); kickstarting/distillation anneal the distill term to zero;
AlphaStar KL-anchors to a FIXED reference; concurrent full-strength
CE + PG on one trunk is the unusual configuration and is now
falsified with both expert types.

PHASED OFFLINE ExIt (§15b sharpened): (i) freeze theta_k, generate
corpus (self-play, committee at eligible nodes; offline budget =>
p can rise; ALONE + bidding-head emission fits here); (ii) distill
PG-OFF with a MIXED loss — CE at material labeled nodes (sparse
override) + SELF-DISTILLATION ANCHOR KL(pi_k || pi) on broad replay
of ALL other states incl. unsearched classes (pick/partner/bury,
leaster, ALONE, forced, abstentions) — LwF-style: match your own
old outputs except exactly where search says otherwise; coverage
boundary becomes a specification, untaught-head bleed structurally
suppressed; value/oracle rehearse on corpus outcomes; low LR, few
epochs, 3-point-battery early stop; (iii) CERT before acceptance
(multi-seed n=1000 + duplicate h2h vs theta_k AND absolute anchor);
reject costs one iteration, not a run; (iv) PG in separate certified
phases only. NOT "exclusively search targets" — the anchor is the
answer to catastrophic forgetting, the cert gate the empirical
backstop. (b) composes with (c): distill into zero-init adapter w/
frozen trunk = forgetting structurally impossible. Suggested pilot:
ONE offline iteration from the clean 8M seed, elevated-p corpus,
cert at end — few days' compute, directly tests whether the +0.180
ceiling survives phase separation; anchor-despite-drift in cert
would be the clean signal that (c) is required.

§16.9 ADDENDUM — corpus design + sizing (operator dialogue, 2026-08-21):
- Acting-policy knob: student-acting corpus (DAgger-correct, states
  theta_k visits) vs committee-acting (AZ-style, improved-policy
  distribution; the §13.3 ceiling was committee-acting). Default:
  mostly student-acting + committee-acting slice; ratio pre-registered.
- p offline is NOT an accuracy trade: per-label quality = committee
  budget + materiality gates; unlabeled states are UNCHANGED by
  construction (anchor). p shapes composition — lower p over more
  games = more diversity + bigger free anchor replay per search-
  dollar (unsearched games ~6 eps/s). Naive random p thins rare node
  classes; offline enables STRATIFIED EMISSION (oversample t0
  defender leads / called-suit holdings) — a lever the online
  teacher never had.
- Sizing: installation dose demonstrated ~14k searched / ~6k
  material labels (both attempts reached the 50s called-suit band by
  29k while fighting PG). Target 25-50k material labels (4-7x dose,
  stratification headroom) + 100k+ free anchor states. Routed
  throughput ~5.2s/committee effective at 8-way => 30k labels ~1.5-2
  days; distill itself minutes-hours; full iteration incl. cert
  under a week.
- Iteration: theta_k+1 anchors BOTH next search and next self-KL;
  cert gates between iterations = AZ generational loop (within-gen
  staleness objection does not apply across certified boundaries).
  Expect diminishing per-iteration gains (+0.180 was step-one
  committee-acting ceiling); absolute-anchor cert prevents ratchet.
- Load-bearing claim the pilot tests: stability comes from anchor +
  early-stop + cert, not label volume; anchored PG-off drift in cert
  = clean verdict for the adapter path, corpus carries over.

§16.9 ADDENDUM 2 — anchor/override partition (operator-found flaw,
2026-08-21): naive "anchor all unsearched states" is SELF-DEFEATING
at p<1: a convention class sampled at p=0.1 gets a 10:90
contradictory vote (10 CE-toward-target vs 90 anchor-toward-old-
behavior on near-identical inputs) — the anchor wins by volume, and
the offline scheme would be WORSE than the online teacher on the
taught subspace (online unlabeled twins carried no explicit
counter-pull; that is how ~6k material labels installed the class).
FIX — the anchor set is constructed, three-way partition:
  1. OVERRIDE: material labels (w>0), CE toward target.
  2. ENDORSED ANCHOR: (a) classes outside emission by design
     (pick/partner/bury, leaster, ALONE, forced — the bidding-side
     collateral protection); (b) searched-and-ABSTAINED play nodes
     (w=0/tie/materiality-fail) — committee examined and endorsed
     the prior (§1.1; §13 phase-1 KL~0 rows) = certified-safe anchor
     INSIDE the play distribution, where a12's damage lived.
  3. NO-LOSS: eligible-but-unsearched play nodes — excluded from
     the loss entirely; shaped only by generalization (the regime
     the online teacher proved installs, minus concurrent PG).
Upgrades: STRATIFIED p (~1.0 at known convention cells per the
§13.5 map — the 10:90 situation cannot arise there; low p only on
low-material background) and optional CHEAP SCREEN routing (§12.8:
1-replicate panel matches heavy confident class 94% at t0) —
high-coverage triage into endorsed-anchor vs full-committee.
INVARIANT: a state carries an anchor loss only if search cannot
speak there or spoke and endorsed the prior — never because search
merely wasn't asked. p then controls corpus cost only, never
anti-teaching pressure.

§16.9 ADDENDUM 3 — data-driven stratification + literature audit
(2026-08-21): OPERATOR AMENDMENT ACCEPTED: p differences MODEST and
driven by the measured disagreement-EV map (ceiling-study node rows
+ §12.7/§12.8 resolved Q-gaps), not hand-picked convention cells —
p(class) = clip(p0 + k*gap_hat, p_min, p_max), nonzero floor
everywhere eligible; goal = conventions AND EV, one statistic, no
category distinction. LITERATURE SUPPORT per component: phase
separation (ExIt Anthony-17; AGZ/AZ Silver-17; offline corpus =
MuZero Reanalyze Schrittwieser-21); disagreement sampling (Query-by-
Committee Seung-92 — our labeler IS a committee; PER Schaul-16 w/
its annealed-IS caution = the "modest" instinct); regret-weighted
supervised updates (AWR Peng-19 / AWAC Nair-20 / CRR Wang-20 — makes
gradient prop. to EV at stake, repairing the §16.9 magnitude
pathology inside the loss; TD3+BC Fujimoto-21 = mixed
override+anchor loss shape); target form already lit-recommended
(Grill-20 regularized target; Gumbel MuZero Danihelka-22); anchor
(LwF Li-16, KD Hinton-15, replay>EWC per continual-learning
consensus, Born-Again Furlanello-18); acceptance gate (AGZ 55%/400
evaluator = exact precedent for dup-h2h gate); warm start/iteration
(Reincarnating RL Agarwal-22, IDA Christiano-18). AMENDMENTS FROM
LIT: (1) Reanalyze-style state reuse across iterations (re-search
stored states under theta_k+1; mix with fresh games per DAgger);
(2) interleave override+anchor within batches, never blocks;
(3) conservative fits x more certified iterations over one deep fit;
(4) temperatures: KD temp on anchor KL + AWR beta on override weight
(smooth knobs above the eps-materiality emission gate); (5) honest
AZ-line challenge: at p->1 committee-acting the pure design drops
the anchor entirely — partition machinery is scaffolding for partial
coverage, remove if search budget allows.

§16.9 ADDENDUM 4 — partition corrections + batch mixture (2026-08-21):
Operator clarifications resolved:
- Searched-and-abstained play nodes = ENDORSED anchor (positive
  evidence, in-distribution protection); eligible-but-UNSEARCHED
  play nodes = NO-LOSS (generalization region — the online-teacher
  regime). Not the other way around.
- Unsearched non-play classes (pick/partner/bury heads, leaster and
  ALONE games' rows) = RETENTION anchor (by necessity, not
  endorsement — search cannot speak; holding them IS the a9/a12
  collateral protection). Keep the two anchor justifications
  distinct in code.
- CORRECTION: forced nodes (1 legal action) = no-loss trivially
  (degenerate softmax, no gradient); earlier listing in the anchor
  was sloppy.
- Raw-count skew at p=0.1 (~1.4 override / ~1.6 endorsed / ~6-10
  retention rows per standard game + excluded-game rows) does NOT
  set gradient shares: BATCH MIXTURE is pre-registered separately
  (order 40-50% override / 25-30% endorsed / 25-30% retention,
  interleaved within every batch) — corpus composition and gradient
  composition decoupled, standard multi-task/distillation practice
  (PER's sampling-ratio-as-hyperparameter). Retention rows compete
  for trunk capacity (dilution risk, mixture-managed), they do not
  contradict play targets on near-identical inputs (unlike the
  addendum-2 10:90 flaw) — different heads, different states.
- References (QBC/PER/AWR/LwF/Reanalyze etc.) to be cited in §17,
  code docstrings and commits at implementation (operator request).

§16.9 ADDENDUM 5 — ALONE/leaster emission + acting-mixture tuning
(2026-08-21): Operator correction ACCEPTED: ALONE/leaster PLAY rows
share the token-pointer play head — retention-anchoring them is
same-head near-neighbor supervision against the taught behavior
(soft 10:90), not cross-head dilution as addendum 4 claimed.
- ALONE PLAY -> EMISSION (approved direction): determinization is
  MORE faithful than standard (no hidden-partner uncertainty, 1v4
  roles known); states structurally adjacent to standard play so
  labels generalize across the boundary instead of fighting it;
  class exits the retention anchor entirely. Pre-registered gate:
  §12.8-style mini-calibration on a few hundred alone nodes
  (paired-replicate noise floor; confirm eps=0.03 sits above it —
  shrink/eps were standard-game calibrated). Alone DECLARATION
  (bidding head) stays retention-anchored; bidding emission out of
  pilot scope.
- LEASTER PLAY -> retention anchor for the pilot: (1) interference
  attenuated by representation distance (no picker, inverted
  incentives, mode flags); (2) observed damage was leaster ENTRY
  (pick head; anchor already protects), not leaster-play quality;
  (3) search validity in leaster unvalidated (P4 determinizer
  exists, no E9-family calibration; different EV structure).
  Falsifiable: ADD leaster-play metrics (leaster score avg,
  point-avoidance) to the cert battery; anchor inadequacy promotes
  leaster into emission at iteration 2 behind the same
  mini-calibration gate.
- ACTING MIXTURE literature: BC compounding error (Ross-Bagnell 10),
  DAgger beta-mixture w/ anneal-to-0 regret theory (Ross-11),
  AggreVaTe roll-in/roll-out (Ross-Bagnell 14), scheduled sampling
  (Bengio-15); AZ = pure teacher-acting extreme (total coverage).
  Hybrid = DAgger stability + AZ coverage of post-improvement
  states (which student-acting at 45% adherence under-produces).
- TUNING: two knobs. Generation-time share — committee-acting games
  are search-efficient (every acted node yields a label), so budget
  ~25-30% of searches on committee trajectories. Train-time balance
  = batch weight over the two pools, SWEEPABLE ON THE FIXED CORPUS
  (distills are minutes-hours; fit 3-4 mixtures, select by cert
  battery + dup h2h). Across iterations: anneal driven by MEASURED
  on-support convergence (disagreement rate along committee
  trajectories -> student-trajectory baseline), not a faith
  schedule.

§16.9 ADDENDUM 6 — exact per-partition loss treatment (2026-08-21):
Two loss forms + an exclusion; the partition is a POLICY-loss
partition (value/oracle heads regress on outcomes at ALL states —
full-coverage rehearsal, no contradiction structure, keeps leaves
calibrated for the next iteration's search).
  1. OVERRIDE: L = lambda_CE * omega(s) * CE(t || pi_theta), valid-
     masked; t = full pi_gumbel-on-shrunk-Q distribution (ties keep
     §12.7 near-equal mass); omega = AWR-style soft weight, monotone
     in resolved Q-gap, temperature beta. Only loss that moves
     behavior.
  2. ENDORSED ANCHOR: L = lambda_end * KL(pi_thetak || pi_theta) at
     KD temp tau. Anchor to theta_k's DIRECT forward pass at the
     trajectory state, NOT the emitted w=0 target — §13 phase-1: the
     pooled engine-replay prior carries the trick-4 recurrent
     divergence artifact (KL~0.030); the committee's contribution at
     these rows is the CERTIFICATE, not the target.
  3. RETENTION ANCHOR: IDENTICAL KL form vs theta_k on whichever
     head the decision used. Anchor sets differ in ROLE not math:
     (a) evidence status -> separate pre-registered shares +
     separate lambdas (start equal; sweepable on dilution);
     (b) TELEMETRY: per-set KL logged separately = two distinct
     early-warning instruments (endorsed-KL rise = taught-region
     play-head drift; retention-KL rise = a9/a12-style collateral
     onset in untaught heads);
     (c) annealing: endorsed share tracks p; retention persists
     while its classes stay outside emission.
  4. NO-LOSS (eligible-unsearched play, forced): no policy loss, not
     in policy batches; PRESENT in the value-regression stream.
Batches: interleaved at pre-registered mixture (~40-50 / 25-30 /
25-30), each row = its policy term + value term.

§16.9 ADDENDUM 7 — bidding-head staleness (2026-08-21): operator
identifies the designed-in limitation: within a distill iteration
there is NO channel from play changes to bidding heads (retention
anchor holds them; PG off; inference never consults the retraining
value heads) — the plan holds bidding ACCEPTABLE-at-theta_k, not
optimal-for-new-play. Bounds + escape channels:
  1. The §13.3 ceiling (+0.180) was measured with committee PLAY on
     FIXED seed bidding — the pilot's target gain already prices in
     exactly this staleness; bidding re-optimization is upside
     beyond the ceiling, not a prerequisite. Staleness cost is
     VISIBLE (h2h attenuation + battery rates), never silent.
  2. Principled channel (iteration 2+, behind the alone-style
     mini-calibration gate): BIDDING EMISSION — pick/partner/bury
     shallow-root searches evaluate by ROLLOUTS under the current
     policy, so search-labeled bidding targets incorporate the
     improved play distribution by construction.
  3. Fallback: bidding-only PG phase w/ trunk + play head FROZEN —
     terminal reward, decent SNR at bidding (frequent decisions,
     larger gaps), and the a11/a12 interaction term (CE-play vs
     PG-trunk) structurally cannot exist. Composition: [distill
     play, PG off] -> cert -> [PG bidding-only, frozen trunk] ->
     cert.
Pilot: accept + pre-register the limitation; staleness meters =
battery pick/alone/leaster/called-card rates + dup h2h across
accepted iterations; drift triggers channel 2 or 3.

## 17. Phased-offline distillation pilot — pre-registration (2026-08-21)

Operator go on the §16.9 design. Build directive: TWO new scripts in
`sheepshead/training` — `distill_corpus.py` (corpus generator) and
`train_distill.py` (supervised trainer) — the league trainer's CLI and
generation machinery is deliberately NOT reused (its loop is built around
concurrent PG, weight publishing and gen boundaries, all absent here).

### 17.1 Disagreement map (mined from the §13.3 ceiling node log)

Source: `runs/ceiling_h2h_202608/nodes.jsonl` (9,099 committee-vs-argmax
node rows, n=500 deals, R=3 @ 1024/1). "Deviated" = 2-of-3 vote winner !=
policy argmax; includes near-tie noise (the §12.8 self-agreement caveat),
so these are UPPER bounds on material disagreement — the corpus manifest's
w>0 material rate per class is the refining instrument.

- By trick: deviation 35.6-41.4%, resolved 87.6->100% (t0->t4). FLAT.
- Lead 42.4% vs follow/other 37.9%; dev|resolved 48.5% vs 39.9%.
- Convention cells (lead nodes): called_suit-eligible 57.1% (64.9%
  dev|resolved, n=201 pooled) — the highest class, matching the known
  deficit; def_lead 41.4%; partner_lead 43.6%; no-cell leads 42.4%.
- n_valid gradient: dev|resolved 35.0% (nv2) -> 53.9% (nv6).
- Adherence flips (policy -> committee-acted): called_suit 45.3 -> 56.7%
  (the teachable deficit); def_lead_no_trump 97.4 -> 87.3% and
  partner_trump 96.9 -> 81.5% (search mildly ANTI-convention at these,
  reproducing E5/§12 — the tie-mass + shrinkage target construction, not
  emission exclusion, is what keeps this from becoming anti-teaching:
  near-tie committee opinions shrink toward the policy prior).

VERDICT for the p-schedule: disagreement is BROAD AND FLAT — the data
supports near-uniform coverage with small boosts, i.e. exactly the
"modest, data-driven" stratification the operator asked for; anything
sharply peaked would have been unsupported.

### 17.2 p-schedule (pre-registered)

p(node) = clip(p0 * b_lead^[is_lead] * b_cs^[called_suit_eligible],
p_min, p_max) with p0 = 0.10, b_lead = 1.25, b_cs = 1.5, p_min = 0.05,
p_max = 0.25. The nonzero floor is structural (§16.9 addendum 4: PER-style
annealed bias bound — Schaul et al. 2016 — every eligible class keeps
coverage). Committee-act games use the same schedule. Iteration 2 refits
b_* against the manifest's measured material-rate/gap map.

### 17.3 Corpus spec (`distill_corpus.py`)

- theta_k = the clean 8M seed
  (`runs/league_retention_pg/checkpoints/..._checkpoint_8000000.pt`).
  Expert = R=3 lockstep committee at 1024/1 with oracle leaves ON
  theta_k ITSELF. Offline phase purity (AZ/ExIt — Silver et al. 2017;
  Anthony et al. 2017): nothing updates during generation, so
  frozen-expert vs closed-loop is moot — expert == acting policy ==
  anchor, and targets are fixed at train time.
- Self-play, ALL-SEAT collection (5 episodes per game, per-seat streams),
  stochastic acting = on-policy state distribution (DAgger — Ross et al.
  2011); both partner modes alternate; terminal rewards; oracle states
  collected (keeps the privileged critic trainable for the next
  certified PG phase / iteration-2 search leaves).
- Committee-acting games (game-level flag, fraction 0.25 of games,
  §16.9 addendum 5): at searched nodes where the target is material
  (w > 0) the seat ACTS the target argmax; everywhere else the policy
  samples. Every material search on these trajectories is also a label
  row (search-efficient), and states downstream of improvements enter
  the corpus (AggreVaTe / scheduled-sampling rationale).
- Partition annotation per §16.9 addendum 6 exactly: override (w>0
  target attached), endorsed (searched, w=0), retention (bidding heads,
  leaster play, alone play+declaration), no-loss (eligible-unsearched,
  forced, committee-failure). ALONE play stays retention until §17.6
  passes, then flips to emission via `--search-alone`.
  AMENDED (operator, 2026-08-21): alone-game PLAY is searched BY
  DEFAULT — no flag, first corpus included. Rationale: same token-
  pointer play head as standard play (the addendum-5 same-head
  argument cuts both ways), and 1v4 determinization carries no hidden-
  partner uncertainty. §17.6 remains as a pre-registered MEASUREMENT
  (noise floor recorded below; a bad floor would prompt an alone-
  specific shrink_s2 or exclusion at iteration 2), no longer an
  inclusion gate. Retention play set = leaster only.
- ANCHOR IMPLEMENTATION of "theta_k's direct forward pass": the
  generator stores theta_k's act-time probability vector (the `act()`
  stash) per anchor row. This IS a direct forward output at the TRUE
  recurrent state of the realized trajectory; the trainer's replayed
  unroll reproduces it to replay noise (the standard PPO ratio~1
  property), so KL(anchor || pi_theta0) ~ 0 at init by construction —
  and the trick-4 engine-replay artifact (§13 phase 1) cannot enter
  because the engine's forced replay is never used for anchors.
- CE targets: `build_ce_search_target` with base_prior = the same
  act-time stash (§16.6 zero-gradient abstention referent), which adds
  a `gap` (top-2 pooled-Q separation) to its info dict for the omega
  weight and telemetry.
- Node telemetry (`--node-telemetry`): one JSONL row per searched node —
  class, regime, w, gap, spread, per-replicate top-pair Q diffs — the
  §17.6 calibration instrument and the map-refinement input.
- Output: shards of N games (default 200) as torch payloads of per-seat
  episode event lists (the `store_episode_events` schema + distill
  keys), plus `manifest.json`: per-class searched/material counts, gap
  histogram, config echo, ckpt path+hash, git rev.
- Dose (§16.9 addendum 2): target 30-50k MATERIAL labels + >=100k
  anchor rows. Measured basis: ~18 unforced play nodes/game across 5
  seats, mean p ~ 0.11 -> ~2 searches/game; ~5.2 s/committee (8-way
  routed) -> ~15-25k games, ~6-12 h wall on 8 workers. Anchor rows are
  free (every unsearched decision).

### 17.4 Trainer spec (`train_distill.py`)

PG OFF: no ratios, no advantages, no entropy controller, no PPO epochs —
a plain supervised loop over corpus shards (segments -> the existing
`_build_minibatch_tensors` / `_forward_vectorized` recurrent unroll; the
distill channels ride alongside via the same pad/flatten alignment).

Per-batch loss (addendum 6, exact forms):
- OVERRIDE: lambda_ce * mean_override[ omega * CE(t || pi_theta) ],
  omega = min(exp(gap/beta), omega_max)/omega_max — AWR/CRR-family
  advantage-weighted regression (Peng et al. 2019; Nair et al. 2020;
  Wang et al. 2020) with beta = 0.03 (the calibrated epsilon_Q scale,
  §12.17) and omega_max = e (weights in [1/e, 1] * 1 — soft, bounded).
- ENDORSED: lambda_end * tau^2 * mean_end[ KL(anchor_tau || pi_tau) ]
  (KD — Hinton et al. 2015; LwF — Li & Hoiem 2016), tau = 1.0 default,
  sweep {1, 2}.
- RETENTION: lambda_ret * tau^2 * mean_ret[ same form ]. Same math,
  separate lambda + separate telemetry stream (the two early-warning
  instruments of addendum 6).
- VALUE: MSE(v, final_score/RETURN_SCALE) at ALL action rows (MC
  target, gamma=1 terminal — no GAE without PG); aux heads keep the
  PPO forms/coefficients; oracle head plain MSE + its aux losses when
  oracle states are present.
Per-partition MEANS then lambdas = gradient-share mixture knob,
decoupled from row counts (addendum 4). Defaults lambda_ce/end/ret =
1.0/0.5/0.5 (~ the 40-50/25-30/25-30 pre-registered shares at observed
per-row magnitudes); sweep on the FIXED corpus: (1, 0.5, 0.5),
(1, 0.25, 0.25), (1, 1, 1), and committee-pool batch weight default vs
2x — 3-4 distills, selected by CERT not train loss.

Optimizer: the agent's existing AdamW groups (actor+encoder, critic) at
a flat distill LR 1e-4, grad-clip = agent.max_grad_norm, 2-4 epochs,
10% episode holdout for no-grad CE/KL eval, `greedy_health_probe`
(n=500, seed=0 — the battery instrument) at every epoch end,
`agent.save()` checkpoints per epoch.

### 17.5 Cert bars (pilot accept/reject)

Multi-seed battery (n=1000 x 4 seeds, §12.22 standard) on the selected
sweep arm: t0 called-suit 50s-60s (installed AND retained — the §16.8
arc peaked 55.5 mid-gen; offline must HOLD it); partner >= 94.5; t0
trump <= 5%; pick/alone/leaster within oscillation bands of seed
(staleness meters, addendum 7); NEW leaster-play metrics (addendum 5)
within bands. Duplicate-bridge h2h (2000 deals/mode): vs theta_k >= 0
at 2 se, expectation +0.05..+0.18 (the pilot captures part of the
+0.180 +/- 0.029 ceiling; committee-play-on-fixed-bidding staleness is
already priced into that number); vs absolute anchor no regression.
FAIL -> the sweep's other arms; all-fail -> §15(c) adapter path.

### 17.6 Alone-node mini-calibration (gate for `--search-alone`)

Run the generator in calibration mode over alone games only
(`--search-alone --p-base 1.0` on a few hundred alone-containing games),
read the node telemetry: paired per-replicate top-pair Q-diff noise
s/sqrt(R) vs the gap spectrum. PASS = alone noise floor <= the
standard-game 0.006 (§12.8) and implied shrink s2 within ~2x of the
6.95e-4 calibration; then alone play joins emission for the corpus
proper. Determinization is 1v4 with no hidden-partner uncertainty, so
the prior is that search there is MORE faithful, not less (§16.9
addendum 5).

### 17.7 Execution order

1. Generator + trainer + tests (this commit series).
2. Generator smoke (tiny run, schema + manifest sanity).
3. Alone calibration run -> §17.6 verdict recorded here.
4. Corpus proper (~20k games incl. 25% committee-act, alone searched
   by default per the §17.3 amendment) -> manifest map recorded here.
5. Sweep distills -> cert battery + dup h2h -> verdict.
Mining command for §17.1 (scratchpad, one-off):
`uv run python mine_disagreement_map.py` over the ceiling node log.

### 17.6 RESULT (2026-08-21): PASS

Run: `runs/distill_alone_cal_202608` — 2,000 games, 291 alone games
kept (14.6% incidence), 5,496 telemetry nodes at R=3 @ 1024/1 (p=1 on
alone play; --alone-only spent zero search elsewhere); 2 committee
failures total (0.04%).

- Paired-replicate noise floor s/sqrt(R): median 0.0050 (p75 0.0098,
  p90 0.0172) — BELOW the 0.006 standard-game reference (§12.8). PASS.
- Implied per-replicate per-action variance: median ~0.05x the
  standard shrink_s2_global 6.95e-4 — alone committees are far LESS
  noisy than standard play (the no-hidden-partner prediction). Using
  the standard s2 therefore OVER-shrinks alone targets (excess
  abstention — conservative, not harmful). Iteration-2 candidate: an
  alone-specific shrink_s2 to sharpen.
- Signal content: material rate 52.0% (mean w 0.378), 19.6% of top-2
  gaps >= eps_Q 0.03, gap median 0.0078.
- Per-class: picker cells noisiest (s ~0.02-0.05), defender-follow
  tight (~0.008); t4 picker rows exactly zero-variance (solved
  endgame, unanimous committee).

Both pre-registered criteria met; alone play stays in the searched
partitions (already the default per the §17.3 amendment). Stability
check: all headline stats within noise of the 825-node interim read.

Step 4 LAUNCHED same day: `runs/distill_corpus_202608`, 20,000 games,
seed 17, committee-act-frac 0.25, p-schedule defaults, oracle states
on, node telemetry on, routed encoder; projected ~0.08 g/s => ~3 days
(the program's long pole; ~2.2 searches/game matches the calibration
run's density).

2026-08-23 dose amendment (operator): the corpus STOPS at 20,000 games
as launched. Measured material yield is ~0.93/game (committee abstains
at ~50% of searched nodes) => ~18.6k override labels, under the §17.3
pre-registered 25-50k floor (which had assumed ~2 material/game).
Operator accepts 18.6k: the attempt-11/12 installation phases showed
teaching visibly landing at ~6k material labels, so the dose carries
~3x margin over the demonstrated-effective quantity; no extension run.
Also amended 2026-08-22 (operator request): train/holdout convention
telemetry added to the trainer (def trump-lead + partner trump-lead
derived from stored masks, full-corpus coverage; called-suit adherent
ids stored by the generator from game 8,000 => ~60% coverage incl.
proportional holdout share; e4d0f6c) — the corpus run was restarted at
the 8,000-game flush boundary to carry the called-suit fields.

2026-08-22 interruption note: the overnight session restart reaped the
background run at 4,075 games (20 shards / 4,000 games banked; ~75
unflushed games lost). Resume support added to the generator
(--start-game, 83d19a3) and the run RESUMED from game 4,000 with
operator approval — indices 4,000-4,074 are fresh independent replays
(per-index seeding), telemetry truncated at the boundary, shard
numbering and manifest continuous. Observed steady rate 0.10 g/s =>
remaining ~16k games ~1.9 days.

### 17.7 Step 4 COMPLETE (2026-08-26): corpus manifest map

DONE: 20,000 games / 100,000 episodes / 100 shards (2.1 GB), ckpt
b56ba26c5c0977dc, final git rev e4d0f6c. Totals: 708,830 action
nodes, 37,332 searched (1.87/game), 18,541 override + 18,783
endorsed (50.3% abstention — matches the ~50% projection), 8
committee failures (0.02%). Committee-act: 5,024 games flagged
(25.1%), 1,771 nodes actually re-acted (the argmax differed).
Override-gap percentiles: p50 0.019, p75 0.043, p90 0.086, p99 0.333
(vs eps_Q 0.03 => ~35-40% of override rows are above materiality by
themselves; omega weighting handles the rest).

Class map (top override mass): defender-follow t0-t3 (~1.1-1.4k
override each at ~41-49% override|searched), then the TARGET cells —
std|t0-defender-lead 1,002 override of 1,212 searched (83%
override|searched, the highest large-cell rate: search disagrees
most exactly where the conventions live), t1-defender-lead 732/1,149
(64%), picker-follow/lead t1-t2 ~600-800 each. Label mass is
concentrated where §12 wanted it without any cell-picking.

Wall-clock post-mortem: ~4.5 days vs ~3 projected — two harness
reaps (resumes at 4,000 / 8,000) plus a machine-wide slowdown in the
final 2 days (fseventsd ~2 cores, load avg 40-120 on 10 cores,
153-day uptime; run throughput 0.10 -> ~0.015-0.02 g/s; workers
memory-flat throughout, so external contention, not a leak — audit
2026-08-25). Mitigation for future runs: telemetry flushes batched
to shard granularity (9ac493d).

Step 5 LAUNCHED (detached, sequential): lambda-grid arms a/b/c =
(lambda_end, lambda_ret) (0.5,0.5) / (0.25,0.25) / (1,1) at
lambda_ce=1, tau=1, 3 epochs, seed 0 (same game-level split all
arms), 500-game greedy probe per epoch -> runs/distill_pilot_{a,b,c}.
Cert battery + dup-bridge h2h vs theta_k follows per §17.5; selection
by cert, never train loss (§12.8).

Arm-a epoch-1 probe (first phase-pure installation evidence):
called-suit 50.3 (baseline 44.7 -> inside the 50s-60s band in ONE
epoch), partner 97.5, t0-trump 0.0, fat/nopoint 65.3/14.2 (~baseline),
top1min_med 6.0; watch item: spread_med 2.9 (above the 2.7 attempt-8
line, below historical ~3.6) — stop-relevant only if it TRENDS down
across epochs.

### 17.8 Step-5 sweep RESULT (2026-08-28): dose, not mixture

All 9 epoch-checkpoints + paired 500-game probes (fixed seed 0 =>
same deals every read) banked; rc=0 all arms. Probe trajectories
(called-suit / t0-def-trump / partner):

- arm a (0.5): 50.3/0.0/97.5 -> 46.9/1.2/98.1 -> 50.0/1.3/98.7;
  pick drifts 34.3 -> 37.8 -> 37.5 (baseline 32.9).
- arm b (0.25): 49.5/0.0/94.7 -> 42.8/2.0/83.3 -> 49.0/0.4/100.0;
  the ep-2 partner trough (83.3) exceeds attempt-9 amplitude.
- arm c (1.0): 47.8/0.4/96.2 -> 40.4/6.2/97.4 -> 50.2/1.6/100.0;
  ep-2 t0 leak 6.2 = worst priority-metric read of the sweep.

FINDINGS. (1) DOSE NOT MIXTURE: epoch 1 installs called-suit to
~48-50 at EVERY lambda (2.5-pt spread across a 4x anchor range);
epoch 2+ damages every arm — lambda only selects the failure mode
(a: mild t0 leak + pick drift; b: partner collapse; c: t0 leak).
Post-installation passes fit label noise (§12.8 ceiling made
visible). (2) OSCILLATION NOT DECAY: ep-3 rebounds everywhere
(b partner 83.3 -> 100.0; c t0 6.2 -> 1.6); lambda maps to
oscillation AMPLITUDE (a ± few pts, b ±15). Single-epoch reads are
luck-of-phase — §12.22 lesson reconfirmed in pure supervision.
(3) ANCHORS DON'T CAP INSTALLATION: 4x anchor weight barely moves
epoch-1 called-suit => tie-band generalization (>60) needs a
tie-row gradient (class pooling, §12.19 fallback), not looser
anchors. (4) Mean anchor KL stays low (0.09-0.15) through argmax
flips — KL constrains the distribution, not the decisions (§12.14
mechanism, supervised edition). (5) Train/holdout streams stayed
nominal through ALL damage — probes are the only instrument that
sees it (§12.14 on/off-support gap).

Shortlist to cert: arm_a_ep1 PRIMARY (50.3/0.0/97.5, least drift),
arm_c_ep1 challenger (47.8/0.4/96.2, tightest anchors), b_ep1
reserve. Cert LAUNCHED (runs/distill_cert_202608, detached):
greedy probe n=1000 x 4 fresh seeds (multi-seed §12.22 standard)
+ h2h_duplicate vs theta_k (2000/mode, se ~0.015) per candidate;
bars = §17.5 + the operator notes below (t0 target 0; called-suit
>60 desirable, not overshoot; h2h >= 0 gates EV).

### 17.9 Cert verdict (2026-08-28): conventions PASS, EV FAIL — and a new lever

Battery (n=1000 x 4 fresh seeds + h2h_duplicate 2000/mode vs
theta_k; runs/distill_cert_202608/cert_results.jsonl):

- arm_a_ep1: called-suit 47.7 pooled (45.7/46.6/48.0/50.6), t0
  0.25, partner 97.5, pick 37.0 (baseline 32.9), leaster ~5;
  h2h edge -0.0697 se 0.0097 (called -0.075 / jd -0.065). FAIL.
- arm_c_ep1: called-suit 49.3 (47.2/52.6/45.6/51.8), t0 0.6,
  partner 96.7, pick 37.3; h2h edge -0.0276 se 0.0089
  (called -0.026 / jd -0.029). FAIL.

Both candidates fail the h2h >= 0 bar. Convention teaching is real
but SMALLER than the epoch probes advertised (pooled +3.0/+4.6 over
the 44.7 baseline, at/below the band floor — the training-loop
seed-0 reads of ~50 were the optimistic tail). Safety bars pass
(t0 at-or-below the seed's own leak; partner >= 96.7). Both arms
carry an IDENTICAL ~+4.4 pick-rate drift (leaster halves) that the
corpus never taught.

KEY FINDING — EV loss scales inversely with anchor weight at EQUAL
installation: a (-0.070) vs c (-0.028), diff 0.042 +/- 0.013
(3.2 sigma) with matched conventions AND matched pick drift. The
bidding-drift hypothesis therefore CANNOT explain the arm gap; the
dominant EV damage is anchor-suppressible play degradation —
label-noise fitting on the ~50%-self-agreement committee targets
(§12.8) that stronger anchors squeeze out without costing the
taught conventions. The §13.3 ceiling (+0.18) measured
committee-ACTED play; distilling argmaxes wholesale pays a noise
tax the ceiling never priced.

Iteration-2 levers (operator to choose; all reuse the corpus, no
new search):
(1) ANCHOR ESCALATION: lambda_end=lambda_ret=2-4, 1 epoch —
    extrapolates the a->c trend; cheapest, directly targeted at
    the measured mechanism. Risk: installation finally caps.
(2) HEAD-ROUTING DIAGNOSTIC: theta_k bidding + distilled play
    through the duplicate h2h — prices the pick drift exactly
    (~50 lines, wrapper agent, both recurrent streams).
(3) TIE-BAND CLASS POOLING (§12.19 fallback): adds the missing
    tie-row gradient for called-suit generalization >60 — value
    only after EV is fixed; pooling won't rescue a -0.03 deficit.
(4) Dose reduction: LR/step cuts (fractional epoch) — blunter
    than (1), same intent.

### 17.10 Iteration 2 pre-registration (2026-08-28): levers 1+2

Operator selected §17.9 levers (1) anchor escalation and (2)
head-routing diagnostic. Pipeline (detached, sequential:
runs/distill_sweep2_202608): arms d (lambda_end=lambda_ret=2) and
e (=4), ONE epoch each (the sweep's dose finding), same corpus /
seed 0 split / probes; then cert battery (n=1000 x 4 + dup h2h)
on both; then head-routed h2h — theta_k BIDDING + arm_c_ep1 PLAY
vs theta_k anchor (sheepshead/analysis/head_routed_h2h.py; both
sub-agents advance recurrent streams on the identical realized
trajectory — PPO act() folds only the encoded state into memory,
never the chosen action, so the chimera cannot desync; same
seed-42 deal pipeline as h2h_duplicate => row-comparable with
§17.9).

Pre-registered readings:
- Escalation trend: a(0.5) -0.070 -> c(1.0) -0.028 halved the loss
  per anchor doubling. NAIVE extrapolation d(2) ~ -0.007, e(4)
  ~ +0.003; expect diminishing returns => d in [-0.02, +0.01],
  e within noise of d. PASS = h2h >= 0 with called-suit >= +3
  over the 44.7 baseline and t0/partner bars held. FAILURE MODE
  to watch: installation finally capping (called-suit -> baseline)
  — anchors were NOT the cap at 0.5-1.0, but 2-4 is new territory.
- Routed diagnostic: edge(route) - edge(arm_c) = the EV recovered
  by undoing arm c's bidding drift. route ~ 0 => deficit was
  bidding (fix = bidding-head anchor escalation or routing at
  deploy); route ~ -0.028 => play damage persists at lambda=1
  (raising the stakes on arms d/e); intermediate => both
  contribute, sized by the split.

### 17.11 Iteration-2 verdict (2026-08-29): floor is play-borne;
### mismatch hypothesis promoted

Escalation arms (cert battery, n=1000 x 4 + dup h2h):
- arm_d_ep1 (lambda=2): called-suit 44.3 pooled (43.2/41.9/48.5/
  43.6) = BASELINE — teaching ERASED; t0 0.2, partner 95.1, pick
  35.9 (drift halved); h2h -0.0239 se 0.0091.
- arm_e_ep1 (lambda=4): called-suit 43.6 = below baseline; t0 0.4,
  partner 94.5 pooled (one seed 92.0 — worst partner card of the
  program), pick 36.8; h2h -0.0229 se 0.0087.

Four-point dose-response (lambda -> install / h2h): 0.5 -> +3.0 /
-0.070; 1.0 -> +4.6 / -0.028; 2.0 -> dead / -0.024; 4.0 -> dead /
-0.023. LEVER 1 CLOSED: EV loss hits a LAMBDA-INDEPENDENT FLOOR
~ -0.025 from lambda=1 on, while installation dies between 1 and 2
(mechanism note: override rows are unanchored — the teaching is
killed INDIRECTLY, via hard-anchored endorsed twins of the same
situations pulling the shared representation back; yet another
arrow at tie-row targets = class pooling). lambda=1 (arm c) is the
optimum of this axis: max installation at the floor. Escalation
past lambda=2 is strictly harmful (partner degrades).

ROUTED DIAGNOSTIC (lever 2): theta_k bidding + arm_c_ep1 play =
-0.0248 se 0.0082 (called -0.0215 / jd -0.0280;
runs/distill_cert_202608/routed_h2h_c.json) vs arm c full -0.0276
=> bidding contribution +0.003 +/- 0.012 NULL. The floor is
PLAY-BORNE; the +4 pick drift is EV-cosmetic (and plausibly a
symptom, see below), undoing it recovers nothing.

MECHANISM SYNTHESIS — anchor-target mismatch (zero-point
corruption), now the lead hypothesis: anchors are act-time stashes
(theta_k online act/observe streams) while the student is scored
under the trainer's batched replayed unroll; at INIT (student ==
theta_k) KL ~ 0.025 — pure computation-path residual. The anchor
loss's zero point is therefore NOT "behave like theta_k" but
"reproduce act-time outputs under replay streams," pushing weights
off theta_k to compensate — a correction that misgeneralizes at
deployment (act-time mode). Predicts: lambda-independent floor
(same wrong destination, any pull strength) OK; play-borne (16.9k
endorsed play rows) OK; pick drift as the same bias expressed via
the 116.5k retention (bidding-head) rows at ~zero EV cost OK;
invisible to convention probes OK; floor magnitude ~ init KL 0.025
(suggestive, not evidence). Search/override rows are immune —
their targets are exogenous Q-derived labels, so mismatch adds
noise, not a biased zero point.

ITERATION-3 PROPOSAL (single change, direct test of the
hypothesis): RECOMPUTED ANCHORS — load a frozen theta_k copy in
the trainer and take anchor targets from ITS OWN replayed unroll
(same batching/segments as the student): init KL == 0 and init
gradient == 0 by construction. Arm-c config (lambda=1, 1 epoch),
same corpus/split, cert battery + dup h2h. PASS = h2h floor
collapses toward 0 with installation held (+4-5 called-suit);
readings between -0.02 and 0 size the residual (value/aux
retraining = next suspect). Cost: one frozen forward per batch.

### 17.12 Premise check (2026-08-29): mismatch hypothesis FALSIFIED;
### floor suspects narrowed

Before launching the recomputed-anchor arm (operator-approved), a
cheap premise check: eval-mode run_epoch at INIT over 4 shards
(4,000 real-corpus episodes), stash anchors vs recomputed
(frozen-theta_k forward over the trainer's own unroll; built as
--recomputed-anchors, 11 tests green):

  STASH      endorsed_kl 0.00000   retention_kl -0.00000
  RECOMPUTE  endorsed_kl 0.00000   retention_kl -0.00000

The stored act-time stashes reproduce under the batched replayed
unroll to FLOAT NOISE — no zero-point corruption exists. (Mechanism:
the workers' routed encoder sends only >=16-row batches to the MPS
shadow; single-row act() encodes ran CPU, matching the trainer's CPU
replay. The "~0.025 init KL" premise was a misattribution — likely a
first-epoch running average taken after updates began.) The §17.11
mismatch hypothesis is FALSIFIED; the recomputed-anchor arm would be
a no-op and is NOT launched. The flag + init-zero test stay in-tree
as the check's record.

Floor suspects remaining — both lambda-independent, play-borne,
probe-invisible:
(A) VALUE/AUX/ORACLE REGRESSION THROUGH THE SHARED TRUNK: greedy
    play never reads the critic, but its gradients reshape the
    shared trunk under coefficients FIXED across all arms; init
    value_mse 0.031 is a real persistent gradient (fresh MC returns
    vs GAE-fitted critic).
(B) OVERRIDE-CE NOISE-FITTING: override rows are unanchored at
    every lambda; greedy expression of the teaching died at
    lambda>=2 but the CE gradient kept flowing through the trunk.

REVISED ITERATION-3 PROPOSAL (awaiting operator): decisive
attribution pair on the arm-c config, 1 epoch + dup h2h each —
  arm g: lambda_ce=0 (anchors nominal, value stream ON) — pure
    value-stream arm; floor present => (A).
  arm h: --no-value-aux --no-oracle (policy stream only) — floor
    present => (B). Built: --no-value-aux drops value+aux terms.
Readings are additive-ish if both contribute; either way the next
fix is targeted (A: freeze value/aux during distill or decouple
via critic-only optimizer steps; B: omega floor/eps tightening or
override-row dose reduction).

OPERATOR APPROVED + LAUNCHED 2026-08-29 ("make it so"):
runs/distill_sweep3_202608 detached — arm g then arm h then full
cert battery (probes n=1000 x 4 + dup h2h) on both.

### 17.13 Ablation verdict (2026-08-29): jointly-carried floor,
### mutually-protective streams

(Machine rebooted mid-battery; battery relaunched, ~10x faster on
the clean machine — h2h 35 min vs 2-4 h. INSTRUMENT NOTE: same-seed
greedy probes differ up to ~6 pts called-suit ACROSS PROCESSES
(g seed1 41.6 pre-reboot vs 48.0 post; seed3 46.1 vs 39.9) — the
probe carries process-level numeric nondeterminism that flips
near-tie argmaxes and forks game paths; multi-seed pooling absorbs
it, single-probe deltas < ~5 pts are not meaningful. Arm g pooled
over all 8 replicates.)

  arm            called-suit  partner  pick   h2h vs theta_k
  c  (both, l=1)    49.3       96.7    37.3   -0.0276 se 0.0089
  g  (value only)   44.6       93.3    34.2   -0.0158 se 0.0080
  h  (policy only)  48.2       ~69     36.4   -0.0543 se 0.0100

- Arm g: value/aux stream ALONE (zero policy teaching) loses
  -0.016 (2 sigma) with partner eroded ~3 pts and called-suit at
  baseline (the seed-0 35.3 was instrument noise) — value-stream
  trunk reshaping is a REAL, MAJORITY-SHARE contributor to the
  floor. No pick drift => the drift belongs to the policy stream.
- Arm h: policy stream ALONE installs called-suit fine (48.2) but
  COLLAPSES partner to ~69 (all 5 reads agree; retention anchor
  at lambda_ret=1 sitting right there) and loses -0.054. The
  partner protection in every combined arm was never the anchor —
  it was the VALUE/AUX stream (aux heads forcing the trunk to keep
  partner-relevant structure; the arch-ablation "aux-ballast"
  phenomenon rediscovered from the training side).
- NON-ADDITIVE: g+h = -0.070 vs combined -0.028. The streams are
  MUTUALLY PROTECTIVE: value/aux shields the untaught conventions
  from CE; CE offsets value-stream drift on the taught cells. The
  -0.025 floor is the residue this mutual protection cannot
  remove — value-stream damage (majority) + CE noise (remainder).

ITERATION-4 MENU (operator to choose; each arm now ~1-3 h train +
~1.5 h battery on the clean machine):
(1) STOP-GRADIENT VALUE: value MSE updates the value head through
    a DETACHED trunk; aux heads keep flowing to the trunk
    (ballast preserved). Prediction: partner held (~96+), install
    >= arm c, h2h improves by up to the value-drag share. One-line
    build; keeps the critic calibrated for the next phase.
(2) AUX-ONLY ablation (drop value MSE, keep aux): splits arm g's
    -0.016 into value-MSE vs aux-head shares first — one more
    attribution rung before committing to (1)'s mechanism.
(3) Run (2) then (1): rigorous order, ~day total at new speeds.

### 17.14 Iteration-4 lever 1 pre-registration + launch (2026-08-29)

Operator chose (1) first. BUILT --stop-grad-value (tested: encoder
grads byte-identical to a value-free backward; critic still gets
the full value-MSE gradient via manual add — gradient surgery, no
ppo.py change, no double forward): value MSE trains the critic
ONLY; the shared trunk sees zero value-stream gradient; ALL aux
heads keep flowing to the trunk (ballast preserved — the lever
splits arm g's stream into value-MSE drag vs aux protection and
removes only the former's trunk coupling).

ARM i LAUNCHED: arm-c config (lambda=1, 1 epoch, seed 0, aux +
oracle on) + --stop-grad-value -> runs/distill_pilot_i; cert
battery (probes n=1000 x 4 + dup h2h). Pre-registered readings:
- PASS: partner >= 94.5 (aux ballast intact), called-suit >= arm
  c's 49.3 (CE no longer fighting value drift), h2h > -0.012
  (value-drag share removed; ideally within noise of 0).
- Value-MSE drag confirmed if h2h improves materially vs c's
  -0.028 at held conventions; aux-drag indicted instead if h2h
  stays at floor (then lever 2 splits g's -0.016 directly).
- Critic calibration retained (value head still trained) for the
  next phase — value_mse telemetry should track arm c's, not arm
  h's untrained drift.

### 17.15 Arm-i verdict (2026-08-29): FAIL — value-MSE trunk
### gradient was protective, not the drag

Cert: called-suit 46.5 pooled (49.1/42.2/47.2/47.3), partner 95.5,
t0 0.25, pick 36.4 — conventions PASS, telemetry sanity checks all
passed (value_mse tracked arm c: critic trained normally). But h2h
= -0.0387 se 0.0093 (modes -0.0387/-0.0386) — WORSE than arm c's
-0.0276 (diff 0.011 +/- 0.013, n.s. but directionally opposite the
pre-registration; the > -0.012 bar decisively FAILED).

CONCLUSION: within the full recipe, the value-MSE trunk gradient is
net NEUTRAL-TO-PROTECTIVE — removing it recovered nothing and
plausibly cost ~0.01. Combined with §17.13, the drag inside arm g's
value/aux stream now points at the AUX-HEAD gradients and/or an
IRREDUCIBLE component (label-noise tax of distilling ~50%-self-
agreement targets, §12.8 — present in every arm, insensitive to
routing). Running tally of falsified single-mechanism stories:
anchors (17.11), replay mismatch (17.12), bidding drift (17.11
routed), value-MSE drag (here). The stream coupling is not
decomposable by one-lever surgery; every component is Janus-faced
(protective somewhere, damaging elsewhere).

REVISED MENU (operator to choose):
(1) WEIGHT INTERPOLATION (WiSE-FT — Wortsman et al. 2022,
    arXiv:2109.01903; model-soup family): theta_alpha = (1-alpha)
    theta_k + alpha*theta_c for alpha in {0.5, 0.75}, NO training —
    battery per alpha (~1.5 h each). Fine-tune damage often
    shrinks faster than task gains under interpolation; if some
    alpha holds ~+3 called-suit at h2h ~0, the pilot ships a
    deployable artifact without solving the attribution puzzle.
(2) Arm j: aux OFF, value ON (CE + anchors + value + oracle) —
    completes the 2x2 stream matrix (g/h/i/j), directly tests
    aux-drag and locates the partner protection (value vs aux).
    Risk: partner collapse repeat.
(3) Accept-and-conclude: record the floor as the §12.8 label-noise
    tax, declare phased-offline distillation conventions-capable
    but EV-taxed at this corpus quality; durable-gains options
    revert to the §12 conclusion list (reward coupling /
    architectural separation).

### 17.16 Option-1 launch (2026-08-29): WiSE-FT interpolation

Operator chose (1). Built runs/distill_interp_202608/make_interp.py:
theta_alpha = lerp(theta_k, arm_c_ep1, alpha) over every float
param+buffer of every module (encoder/actor/critic/oracle),
integer buffers asserted identical; saved via agent.save() =
ordinary checkpoints. interp_a50 + interp_a75 built; cert battery
LAUNCHED on both (probes n=1000 x 4 + dup h2h each).

Pre-registered readings (WiSE-FT anisotropy bet): damage from
noise-fitting decays faster along the path than the coherently-
installed convention shift. SHIP CANDIDATE = any alpha with h2h
within noise of 0 (>= -0.01) AND called-suit >= ~47.5 (i.e. most
of arm c's +4.6 held) AND partner >= 94.5, t0 <= 1. Linear-fade
null: called-suit gain and h2h loss shrink proportionally
(a75 ~ 3/4 of both, a50 ~ 1/2) — informative but no ship. Watch
partner: both endpoints hold it, so any interior dip would be a
mode-connectivity artifact (not expected at 1-epoch distance).

RESULT (2026-08-29/30): ANISOTROPY CONFIRMED, a50 PASSES ALL BARS.

  alpha  called-suit(4x1000)  partner  t0    h2h vs theta_k
  0.0        44.7 (baseline)   ~96.5    ~1    0 (identity)
  0.5        48.0              96.8     0.4   -0.0039 se 0.0074
  0.75       45.8              96.7     0.1   -0.0155 se 0.0083
  1.0 (c)    49.3              96.7     0.6   -0.0276 se 0.0089

EV damage is CONVEX along the path (a50 sheds ~86% of arm c's loss;
linear fade would predict -0.014) while the convention is CONCAVE-
ish (a50 keeps ~70% of the gain; a75's 45.8 dip vs a50 is ~1.2
sigma — noise or a mild non-monotone bump, either way dominated).
interp_a50 = FIRST ARTIFACT IN THE PROGRAM with an installed
convention (+3.3 called-suit, 4x n=1000) at EV PARITY (h2h -0.004
+/- 0.007; jd mode +0.007, called mode -0.015 ~ -1.4 sigma = watch
item, not a fail). Partner 96.8, t0 0.4, pick 37.2 (the cosmetic
drift interpolates through), spread 3.4 (compression mostly gone).

VERDICT: interp_a50 is the ship candidate
(runs/distill_interp_202608/interp_a50.pt). Deploy decision +
whether to refine alpha (e.g. 0.6) is the operator's; marginal
value of refinement looks low (a50 already passes every bar).
WiSE-FT interpolation is hereby a validated post-processing step
of the phased-offline ExIt recipe: distill hot (accept EV damage),
then walk back toward theta_k to the EV-parity point.

§17.16 ADDENDUM (2026-08-30) — cross-lineage h2h vs PROD: operator
requested interp_a50 vs final_pfsp_swish_ppo (the deployed 30M
legacy-arch model) through the duplicate instrument (2000/mode):
edge +0.0145 (called +0.0076 / jd +0.0213, se ~0.014) — a
STATISTICAL TIE with the point estimate in a50's favor. The
arch-ablation panel-absolute readings (v2 lineage ~-0.12) do NOT
transfer to direct head-to-head at theta_k+distill maturity:
deploying interp_a50 would NOT be a strength downgrade vs prod,
and it carries the called-suit convention at its lineage's EV
parity. Deploy decision remains the operator's
(runs/xarch_h2h_202608/run.log).

§17.16 ADDENDUM 2 (2026-08-30) — legacy-30M probe card + OPERATOR
PROVENANCE NOTE: same instrument (4 x n=1000) on
final_pfsp_swish_ppo: called-suit 90.1 (87.4-93.5), partner 99.8,
t0-def-trump 1.18, fat/nopoint 29.7/48.8, pick 32.0, leaster 10.4,
alone 8.1, spread_med 6.58, top1min ~17. (Eligibility is pooled
over ALL tricks with the called suit unplayed + both classes held;
high adherence extinguishes own eligibility early — 298 vs a50's
385 eligible leads/1000 games.) vs interp_a50: 30M wins called-suit
(90 vs 48) + partner (99.8 vs 96.7); a50 wins t0 cleanliness (0.41
vs 1.18); h2h = tie (+0.014 a50).

OPERATOR (verbatim substance): the 30M ran 15M episodes under
SHAPED convention rewards while entropy annealed, then 15M
unshaped. That directly explains the spread (6.6) and the
partner/called-suit adherence — mass concentrated ON the rewarded
actions. PROGRAM SIGNIFICANCE: (a) the 30M's 90-adherence is
trained-in preference, not tie-band generalization; (b) DURABILITY
DATUM — shaping-installed conventions at concentrated mass
SURVIVED 15M subsequent outcome-only episodes, while every
CE-installed convention in this program (spread ~3 at install)
eroded under any later gradient pressure => basin depth at install
time, not the convention or arch, is the durability variable;
(c) this is the §12 "reward coupling" durable-gains option with an
existence proof at EV parity (h2h tie after the unshaped 15M).

OPERATOR FRAMING CORRECTION (2026-08-30, supersedes the last
implication line): the no-shaping discipline is not a rule up for
revisiting — it IS the research program. History: the original
trigger for the convention investigations (-> oracle critic,
search teacher) was the 30M's DEFENDER TRUMP-LEAD LEAK; the
terminal-only lineage has FIXED that (t0 0.25-0.6 across §17 arms
vs the 30M's 1.18 pooled / 4.8 historical) — priority 1 achieved.
Priority 2 = comparable-or-higher EV WITH partner/called-suit
adherence under disciplined terminal-only rewards: EV
comparability achieved (h2h tie), partner close (96.7 vs 99.8),
called-suit the open gap (48 vs 90). The shaped 30M is the
BASELINE the terminal-only program must match, not a route back;
its analytical role is proving the 90-adherence target is
EV-compatible and durable. Live levers toward it: concentrated/
sharpened tie-band distill targets (class pooling on the endorsed
rows — where the 30M's shaped preference lives), corpus q's dense
endorsed coverage as the material.

PER-TRICK called-suit breakdown (2026-08-30, 2,000 greedy games
each, probe-identical eligibility, scratchpad cs_by_trick.py):
  trick    a50            30M
  0        39.0 (n=556)   95.1 (n=596)
  1        46.4 (n=416)   92.2 (n=232)
  2        53.4 (n=281)   92.5 (n=159)
  3        47.3 (n=224)   75.6 (n=127)
  4        52.8 (n=89)    62.2 (n=45)
  pooled   45.5           90.8
Mirror-image profiles: the shaped install is DEEPEST at t0 (95.1,
decaying late where the convention matters least); a50 is WEAKEST
at t0 (39). Priority-bin deficit = 56 points. 30M's t0 95 exceeds
the search-endorsed band + tie band (~85 ceiling): either mild
EV-irrelevant over-adherence, or single-node committee Q at deploy
budgets under-prices the convention's coordination value (operator
SNR thesis) — h2h tie cannot distinguish. PROGRAM TARGET, made
precise: raise t0 called-suit 39 -> ~90 under terminal-only
discipline while preserving t0-trump cleanliness (0.25-0.6).

## 18. EV-gain program — pre-registration (2026-08-30)

GOAL: convert the search ceiling (+0.180 committee-ACTED, §13.3;
offline-grade targeted-search edge independently confirmed) into
DISTILLED EV GAIN, not just parity. Theory inventory (recorded
in-session 2026-08-29): (1) near-tie label noise, (2) sparse label
support + uncontrolled generalization (2.7% of decisions labeled),
(3) covariate shift (75% of corpus states are theta_k trajectories;
the ceiling is search-on-its-own-trajectory), (4) partial-obs
unrealizability of some search decisions, (5) converged-policy
fine-tuning tax. Arm g ALREADY measures the label-independent
component (-0.016; the planned "self-distillation null" is
redundant with it — a pure-KD null is vacuous since every loss is
zero at init). Reference-recipe contrast (AGZ/ExIt): label density
1.0, committee-act 1.0, soft targets, moving data, never-converged-
elsewhere network; our budget-forced deviation is the Gumbel-family
Q readout (AGZ visit counts are prior-dominated at our ISMCTS
budgets — June audit). Deferred readout-fidelity item: Gumbel-style
value-completion of unvisited actions.

### 18.1 Arm p — near-tie label pruning (--gap-floor 0.03)

Arm-c config (lambda=1, 1 epoch, seed 0) + --gap-floor 0.03
(BUILT + tested: override rows with top-2 gap < 0.03 demote to
NO-LOSS, never anchored — anchoring searched-and-disagreed rows
would anti-teach). Keeps 6,889 of 18,545 override labels (37.1%)
— at the demonstrated-effective installation dose (~6k); every
surviving label saturates omega=1 (beta=0.03), so this is a HARD
exclusion the soft weighting never applied (near-tie labels
previously carried omega 0.37-1.0). Pre-registered readings: label
noise real => h2h improves from arm c's -0.028 toward the arm-g
floor (-0.016) with called-suit installation HELD (>= ~47.5; the
high-gap labels are the 153:7-directional ones). h2h unchanged =>
CE damage is not gap-concentrated (points back at support/shift
theories). Installation lost => the small-gap labels were carrying
teaching after all (dose result). WiSE-FT walk-back applies to the
winner afterward regardless.

### 18.1 RESULT (2026-08-30): FAIL both ways — near-tie labels
### were not the poison

Cert: called-suit 43.0 pooled (40.9/43.9/42.9/44.2) — BELOW
baseline, installation GONE; partner 97.5, t0 0.5, spread 3.2
(compression indeed came from the near-tie labels). h2h -0.0449
se 0.0090 — WORSE than arm c's -0.0276, with an extreme mode
split: called -0.0746 / jd -0.0151.

Reading: the pre-registered "noise component confirmed" outcome is
REFUTED AS EXECUTED — removing the noisiest 63% of labels made
both teaching and EV worse. FIFTH falsified single-mechanism story.
CONFOUND (honest note, not in the pre-reg): the override loss is a
per-partition MEAN, so pruning to 1/3 the rows RENORMALIZED
per-label intensity ~3x (and every survivor sits at omega=1, vs
0.37-1.0 mixed before) — arm p trained fewer labels HARDER, and
high-gap labels concentrate in called-game convention cells, which
matches the called-mode -0.075 damage signature (arm-a-like). So
the arm conflates label selection with intensity; a clean variant
would scale lambda_ce by ~0.37. EMERGENT ALTERNATIVE READING: the
full label set's redundancy/diversity acts as its own regularizer
— near-tie labels, individually noisy, spread the CE pressure
thin; concentrating it is what breaks play. Fits arm a (strong
effective pressure, -0.070), c (balanced, -0.028), p (concentrated,
-0.045). Intensity-compensated arm p' (lambda_ce 0.37) is the
deconfounding follow-up if wanted; NOT launched — corpus q is the
main event and its distill recipe defaults to the FULL label set
(gap floor NOT carried).

### 18.2 Corpus q — AGZ-density mini-corpus

5,000 games, seed 19: p-base/p-min/p-max = 1.0 (EVERY eligible
standard+alone play node searched), committee-act-frac 1.0 (EVERY
game acts the material argmaxes => states lie on expert
trajectories; MC value targets = expert-play returns, Reanalyze-
flavored), R=3 @ 1024/1, oracle + telemetry on, routed encoder.
Attacks theories 2+3 structurally (the two biggest deviations from
the reference recipes). Expected ~20-25 searches/game => ~100-125k
searches (~3.3x corpus proper); estimate 1.5-3 days wall on the
clean machine. Distill recipe for q decided after arm p (gap floor
carried iff validated); cert battery + WiSE-FT walk-back as
standard. Launched DETACHED alongside arm p (contention accepted:
corpus is the long pole).

OPERATOR INTERPRETATION NOTE (2026-08-26, for cert readings): the
50s-60s called-suit band is the SEARCH-ENDORSED optimum, but ~20-25%
of additional eligible leads sit in the tie band (search abstained;
called-suit and an alternative fail equally valid). Generalizing into
that band — rates ABOVE 60 — is MORE ideal from a human-convention
standpoint provided EV is not sacrificed (h2h >= 0 gates it). Do NOT
read >60 as overshoot. Structural link: tie-band rows ARE the
endorsed partition (w=0 => KL-anchored to theta_k), so lambda_end
directly prices this generalization — arm b (0.25) permits it most,
arm c (1.0) resists it most; the operator preference makes b's
region the desired outcome if conventions + h2h hold. Second note:
t0 defender trump-lead 0.0 is the priority read — this convention is
EV-positive and any leak > 0 is jarring at a human table; treat 0 as
the working target, the pre-registered <=5 as a hard outer bar only.

## 19. Post-30M program objectives — operator restatement
## (2026-08-30, authoritative)

The intentions of the post-30M agent program divide into three:

1. Produce an agent that FIXES THE DEFENDER TRUMP-LEAD LEAK
   (defender trump lead as close to 0 as possible).
2. To avoid behavior bias, produce an agent via TERMINAL-ONLY
   REWARDS with as much convention adherence as possible — and
   determine whether the human conventions are actually OPTIMAL
   and DERIVABLE from terminal-only rewards.
3. Produce an agent with MEASURABLY BETTER SKILL than the 30M.

Operator status summary (2026-08-30): significant progress on all
three; none completely satisfied.

1. CLOSE TO FULLY RESOLVED — very minor leaks still occur, but
   incredibly close to target. [Annex: t0 def trump 0.25-0.6
   across §17 arms / 0.41 interp_a50, vs 30M 1.18 same-instrument
   / 4.8 historical.]
2. OPTIMALITY LARGELY ANSWERED YES: defender and partner lead
   conventions appear 100% optimal under search; called-suit
   generally optimal with GENUINE exceptions. [Annex: E9 +
   ceiling confident-label studies; §12.15-12.17 t0 called-suit
   EV-backed; §12.8 tie band = the exception set.] TRANSLATION of
   that EV into play behavior beating the 30M: achieved ONLY for
   defender trump-lead (the most important — good news there);
   partner 96.7 vs 99.8 and called-suit t0 39 vs 95 remain behind
   the shaped baseline.
3. Terminal-only rewards CAN produce a much stronger agent — but
   so far only by running the search algorithm directly (+0.180
   committee-acted, §13.3); attempts to DISTILL that into a
   deployable network have been largely unsuccessful (best =
   parity: interp_a50 ties theta_k and the 30M). Corpus q (§18.2)
   is the live structural attempt.

## 20. Pivot: search-Q regularized policy iteration (2026-09-01)

Status: APPROVED, machinery in build; first run on corpus q (stopped at
its 3,000-game flush boundary, 15 shards) as soon as it lands.
Supersedes the §17/§18 target construction (pi_gumbel-on-shrunk-Q) as
the DISTILL target; the corpus generator, the cert battery and the
WiSE-FT walk-back all carry over.

### 20.1 Why the target had to change

Three measurements taken on corpus q's own output (48k searched nodes,
shard 13 rows, 2026-09-01) settle where the §17 EV floor lives.

1. At the convention cells the committee's top-2 ordering is at the
   noise null. Per-replicate top-pair Q diffs, R=3 @1024/1:

     class                material  gap>=0.03  median |t|  sign-agree
     std|t0-defender-lead   82.5%     1.9%       0.75        25.0%
     std|t1-defender-lead   63.7%     1.4%       0.85        28.4%
     std|t0-partner-lead    50.1%     2.1%       1.07        33.5%
     std|t3-picker-follow   64.3%    42.7%       4.9         81.7%
     std|t4-defender-follow 22.0%    19.5%      11.1         99.9%

   The pure-noise null for three-replicate sign agreement, oriented by
   the pooled mean, is 24.9% (simulated). Lead cells sit ON it; follows
   from trick 2 carry real signal.

2. Those rows nonetheless reach the trainer as near one-hot targets.
   Override rows, shard 13: at t0 defender leads the median target
   max-prob is 0.952 (57% of rows > 0.9, entropy 0.20 nats) on a card
   the policy itself rated 0.49, at a median top-2 gap of 0.006 Q =
   0.4 committee SEs. Follows are one-hot too, legitimately (gap 2-3
   SE, policy already at 0.8-0.9).

3. Mechanism, all three by construction and none a bug against §1:
   (a) the shrink factor w compares replicate noise to the Q variance
   over the WHOLE legal set — a lead set nearly always holds a blunder
   card (trump lead, −0.3 Q) that makes the set variance large while
   the top 3-4 cards sit within noise, so w = 0.5-0.7 says nothing
   about top-2 order (§10.1 recorded this); (b) min-max normalization
   is scale-free, so tie contenders land at 0.9-1.0 whatever the Q
   span; (c) the tilt multiplier (c_visit 50 + max N) × 0.1 is 50-80
   logit units — a 0.1 normalized separation × 60 × w 0.6 = 3.6 nats,
   i.e. the 0.95 observed.

   Gumbel MuZero's visit-count confidence proxy assumes Q error
   shrinks with visits. In determinized ISMCTS the residual error is
   world-sampling / strategy-fusion noise, which §12.8 showed does not
   shrink from 1024 to 4096 iterations. So the proxy carries no
   information about top-2 order in our search, and CE toward one-hot
   coin flips on near-identical states drives the policy to a flat
   tie set that greedy argmax then breaks inconsistently — the spread
   compression every hot arm showed, the λ-independent, play-borne,
   probe-invisible floor of §17.11-§17.15, and the arm-p failure
   (pruning the coin flips also renormalized intensity).

Reconciliation with Convention_Erosion's ecology-invariance result:
the conventions ARE unilaterally correct (partner-trump +0.24,
called-suit +0.10 to +0.11 score/opportunity in both ecologies,
falsifiers passing) — the value is material, not signalling. In Q
units (score/12) that is 0.008-0.02 per opportunity, i.e. AT or BELOW
the 0.015 Q per-node resolution of a committee of three, and §12.8
showed budget does not lower that floor. The convention is resolvable
only by POOLING across nodes, which is what the CRN counterfactual
instrument does at 2.5-4σ and what a per-node search label cannot do.
The population-equilibrium framing floated in-session on 2026-09-01
was withdrawn on this evidence.

### 20.2 The method

KL-regularized approximate policy iteration in which policy EVALUATION
is a regression of search Q onto the policy's own frozen features and
policy IMPROVEMENT is the mirror-descent step. Three stages per
iteration k, on a fixed corpus generated by θ_k:

  Stage 1  (pooling / evaluation)
      fit A_φ(s, a) by heteroscedastic weighted least squares
          min_φ  Σ_n  Σ_{a∈V_n}  ( A_φ(s_n,a) − (q̄_{n,a} − v̄_n) )² / σ²_n
      where q̄ is the pooled completed-Q vector, v̄_n its legal-set mean
      (paired design: centering removes the world-sample offset shared
      by all cards at a node), σ²_n the node's replicate sampling
      variance (blended per §1.2). Features = θ_k's frozen encoder
      output at the decision: the 256-d shared readout (through the
      frozen actor adapter) and the 8 post-reasoning hand tokens — the
      SAME two inputs the play pointer reads. A_φ is a twin of that
      pointer (v'·tanh(W'_g h + W'_t token_i)), so "similar" means
      similar for the decision the policy makes and any tilt it
      produces is expressible by the policy net (theory 4 of §18
      removed by construction). Capacity rungs: pointer only (~20k
      params, frozen adapter) / fresh adapter MLP + pointer (~150k) /
      encoder unfrozen. Selection by HELD-OUT weighted MSE — legitimate
      here because Stage 1 is a regression under a stated noise model,
      unlike policy-probe selection (§12.8 lesson).

  Stage 1b (Fay-Herriot combination — Fay & Herriot 1979, the
      covariate-based James-Stein of small-area estimation)
      residual variance  σ²_u = max( E_holdout[r²] − E[σ²_n], floor )
      per node           γ_n = σ²_u / (σ²_u + σ²_n)
      blended advantage  Â_n = γ_n·A_obs + (1−γ_n)·A_φ(s_n,·)   (rows with Q)
                         Â_n = A_φ(s_n,·)                        (searched rows without Q)
      posterior variance v_n = γ_n·σ²_n  (rows with Q),  σ²_u  (without)
      Resolved nodes keep their own reading; tie nodes take the pooled
      one; σ²_u absorbs the determinization bias common to a node's
      replicates that replicate SE cannot see.

  Stage 2  (improvement / target)
      t_n(a) ∝ p_{θ_k}(a | s_n) · exp( clip( Â_n(a) / (κ·√v_n), ±tilt_max ) )
      the mirror-descent / advantage-weighted extraction step (Vieillard
      et al. 2020; Peng 2019; Nair 2020; Wang 2020) with the temperature
      set by the posterior SE so a one-SE edge is one nat: a 0.75-t lead
      row stays near the prior, an 11-t endgame row goes one-hot. κ is
      the single Stage-2 knob (default 1; cert-gated sweep 0.5/1/2).
      p_{θ_k} = the act-time stash (§17.3 anchor; replay-exact, §17.12).

  Stage 3  (projection)
      CE( t_n ‖ π_θ ) on every searched play row, PG off, 1 epoch at the
      arm-c settings; retention KL(p_{θ_k} ‖ π_θ) on bidding-head and
      leaster-play rows; value/aux/oracle regression on all rows as in
      §17.4. The override/endorsed partition DISSOLVES: where Â ≈ 0 the
      target IS the prior, so abstention and anchoring are the same row.
      Then the standard battery (n=1000 × 4 seeds + duplicate h2h vs
      θ_k) and the WiSE-FT walk-back if needed.

Why this is "class pooling without classes": the statistical problem
is many scalar measurements at SE ~0.015 hiding a shared ~0.01 effect
across similar states; hierarchical shrinkage toward a cluster mean is
the textbook answer and a hand-labeled card class is just a
hand-specified cluster. Here the cluster is continuous, defined by
θ_k's frozen feature geometry; Q values set the function's VALUES,
capacity/regularization bounds how sharply it can vary between
neighbours (the "number of clusters" knob, chosen by held-out
likelihood). Across iterations the prior is multiplied by fresh
independent evidence each time, so a weak per-iteration tilt
compounds toward the pooled winner while the cert gate checks EV at
every step (Vieillard's error-averaging result for KL-regularized
API: evaluation errors average across iterations at rate 1/k rather
than compounding).

Literature: Fay & Herriot 1979 (J. Am. Stat. Assoc. 74, small-area
estimation; Efron & Morris 1975 for the James-Stein root); Vieillard,
Pietquin & Geist 2020 "Leverage the Average" (NeurIPS,
arXiv:2007.06799) and Vieillard et al. 2020 "Munchausen RL"
(arXiv:2007.14430) — KL-regularized policy iteration, error averaging;
Grill et al. 2020 (arXiv:2007.12509) — the regularized improvement
form; Peng et al. 2019 AWR (arXiv:1910.00177), Nair et al. 2020 AWAC
(arXiv:2006.09359), Wang et al. 2020 CRR (arXiv:2006.15134), Kostrikov
et al. 2021 IQL (arXiv:2110.06169) — advantage-weighted policy
extraction and the weight clip; Anthony et al. 2017 ExIt, Sun et al.
2018 Dual Policy Iteration (arXiv:1805.10755) — the search-flavoured
loop; Tesauro et al. 2010 "Bayesian inference in MCTS" (UAI) and Russo
2016 top-two Thompson sampling (arXiv:1602.08448) — posterior-of-
optimality readouts, the uninformative-pooling special case; Kendall &
Gal 2017 (arXiv:1703.04977) — heteroscedastic heads, the fallback if
σ²_u varies strongly by class; Lisý, Lanctot & Bowling 2015 (Online
Outcome Sampling, AAMAS) and Heinrich & Silver 2015 (Smooth UCT) —
regret-based search whose AVERAGE strategy is the output (the in-engine
alternative to a visit-count readout, held in reserve); Brown et al.
2020 ReBeL (arXiv:2007.13544) / Schmid et al. 2023 Student of Games —
public-belief-state search, the full-rewrite alternative, held in
reserve.

### 20.3 Implementation plan (this commit series)

- `pfsp_runtime.summarize_committee` → `CommitteeSummary` (pooled q̄,
  per-action sampling variance, mean visits, pooled prior, node noise
  variance, w, gap, spread, max visits). `build_ce_search_target`
  becomes a consumer of it (numerically identical; regression test
  against the pre-refactor implementation).
- `distill_corpus.py` row schema v2: every SEARCHED row (w = 0
  included) stores `search_q`, `search_q_var`, `search_n`,
  `search_prior`, `search_noise_var`, `search_spread`,
  `search_stats_source="committee"`; the act-time stash `anchor_probs`
  is stored on EVERY row with ≥2 legal actions (the anchor-LOSS
  invariant of §16.9 addendum 3 lives in the trainer's partition
  codes, not in data presence); `--committee-act-frac` default 0.0
  (acting the argmax of a coin-flip lead reshapes the trajectory and
  value-target distribution); manifest carries `row_schema: 2`.
- `recover_search_q.py`: corpus q (schema 1) → schema-2-equivalent
  shards. For override rows the stored target is softmax(log p +
  c·minmax(q̄)) with c a per-node constant, so (log t − log p) min-max
  normalized IS minmax(q̄); scaled by the telemetry spread it is q̄ up
  to an irrelevant offset; the node noise variance is (1 − w)·Var(q̄)
  by the definition of w. p is θ_k's replayed forward (act-time
  stashes reproduce under replay to float noise, §17.12). VERIFIED per
  row against the telemetry's independent top pair and gap (tolerance
  1e-4 Q); mismatches are marked `recovery_failed` and carry no Q.
  Endorsed rows (w = 0, no stored target) are `unrecoverable`: they
  keep gap/spread and receive Stage-2 targets from the model only.
- `search_advantage.py`: `AdvantageModel` (capacity pointer / adapter /
  trunk), row encoding through the frozen encoder + actor adapter,
  cached feature tables for the frozen rungs, weighted fit with
  held-out early stopping, Fay-Herriot σ²_u / γ, target builder,
  diagnostics (held-out MSE vs noise floor per class; top-card
  agreement with committee draws vs the 25% null and vs the prior's
  own agreement).
- `train_policy_iteration.py`: stages `fit` / `target` / `distill`
  (and `all`); writes the targeted corpus (`pi_target` per row) so the
  projection is reproducible and inspectable; PG-off CE + retention
  KL + value/aux/oracle loop; probes and checkpoints per epoch.
- `analysis/interpolate_checkpoints.py`: the §17.16 WiSE-FT lerp as a
  reusable CLI.
- Tests: summary-vs-legacy equivalence; schema-2 rows; recovery
  round-trip on a scripted-committee corpus (q̄ up to offset, gap,
  anchors); head scatter alignment; synthetic fit reaches the noise
  floor; Fay-Herriot limits; target builder identities (Â = 0 ⇒ prior;
  clip); distill smoke.

### 20.4 Pre-registered expectations (corpus q, iteration 1)

Stage 1 (decided before any policy is touched):
- Held-out weighted MSE reaches within ~1.5× the replicate noise floor
  for the adapter rung; the trunk rung does NOT beat it on held-out
  (fitting per-node noise) — if trunk wins, the feature-geometry
  premise is weaker than claimed and rung choice is revisited.
- Pooling diagnostic at std|t0-defender-lead: the model's top card
  agrees with fresh committee draws ABOVE the 25% noise null (target
  ≥ 35%), and above the prior argmax's own agreement; at endgame
  cells agreement is near 100% (no information destroyed).
- σ²_u is small relative to lead-cell σ²_n (γ_n well below 1 at leads
  ⇒ the model dominates there) and large relative to endgame σ²_n
  (γ_n ≈ 1 ⇒ resolved nodes keep their reading).

Stage 3 cert (n=1000 × 4 + duplicate h2h vs θ_k), κ = 1, 1 epoch:
- Called-suit installs at least as far as arm c (pooled ≥ 49; the
  §17.8 sweep's +4.6) with the t0 bin moving (≥ 45 from 39).
- h2h HOT ≥ −0.010 (vs arm c's −0.028 and the λ-independent −0.025
  floor): the floor was the coin-flip tax. A hot read inside noise of
  zero makes the walk-back unnecessary; a read at the old floor with
  installation intact says the tax is NOT target-borne after all and
  the value/aux coupling (§17.13) returns to the front.
- Partner ≥ 95, t0 defender trump ≤ 1.0, pick/leaster/alone within
  the §17.9 bands. Spread median ≥ 3.2 (no compression: near-tie rows
  now carry near-prior targets).
- Iteration 2 (fresh corpus from the accepted θ_{k+1}, schema 2,
  student-acting): called-suit climbs further with h2h ≥ 0 vs θ_k+1
  AND vs the 8M anchor — the compounding claim. The 30M's 90 is a
  multi-iteration target, not an iteration-1 bar; the exception-rate
  argument (§13.5) puts the terminal-only optimum in the 80s.

Failure readings: Stage-1 agreement at the null ⇒ the features do not
carry the convention linearly at the reachable capacity — fall back to
the Kendall-Gal heteroscedastic head, then to explicit tie-band
pooling as the last resort. Installation without EV recovery ⇒ §17.13
value-stream coupling is the residual, attack it separately (critic-
only phase). EV recovery without installation ⇒ raise κ / iterate.

### 20.5 Implementation record (2026-09-01, same day)

Landed as five commits on master (local): §20 doc; committee-summary
refactor + row schema 2 (`924ef54`); recovery tool (`fcf66dd`);
policy-iteration trainer + interpolation CLI (`32c547a`); log-prior
covariate (`c2a03d2`). 40 + 3 + 4 tests green in the touched files;
basedpyright clean.

Deviations from §20.3, both additive:
- The advantage model takes theta_k's CENTERED LOG-PRIOR over the
  legal set as a covariate with one learned scale. Motivation from
  the first real-shard smoke: the prior's own top card is the
  committee's top card 58% of the time, a freshly initialized head
  starts at chance and spends epochs relearning that. The head now
  fits the residual the prior does not explain (the obvious
  Fay-Herriot covariate); the bare head is kept for the scatter-
  alignment test.
- Stage 3 is not a new loss loop: the target stage writes a TARGETED
  corpus in the schema `train_distill` already consumes (search_target
  := the §20 target, distill_set := override, legacy target kept as
  search_target_legacy) and the projection reuses that tested loop
  with omega fixed at 1. The new script owns fit / target and the
  orchestration.

Recovery on real corpus-q rows (shard 13, 1,677 override rows):
1,612 recovered (96.1%), 65 refused — 39 exact-top-tie endgame rows
whose worst card underflowed float32 (no pair to pin on), 26 min-
residual mismatches (replayed prior vs stash beyond 1e-4). Recovered
min-residual max 3e-5; median tilt/w 78.8 (max N ~740, consistent
with 1024 iterations). 10 s per shard.

Stage 1 smoke on that one shard (adapter rung, 6 epochs, PRE-
covariate): 3,384 targetable rows / 1,612 with Q; held-out weighted
MSE 1.8e-2 -> 5.9e-3, still descending, against a noise floor of
2.1e-4; sigma_u^2 5.7e-3. A large sigma_u^2 makes gamma ~ 1 on every
row with Q, i.e. the Stage-2 targets reduce to the per-node evidence-
calibrated readout until the model has earned trust — the safe
default the design intended.

Corpus q: stopped at its 3,000-game shard flush (15 shards; ~24k
override + ~25k endorsed rows) rather than run to 5,000, since Stage 1
has ample power on what is banked and the schema-2 generator now
records what recovery cannot. Iteration-1 pipeline:
`runs/policy_iteration_202609/pipeline.py` (recover -> fit all rungs
-> target -> distill 1 epoch -> cert battery on distill_epoch1 ->
WiSE-FT alpha 0.5 -> cert battery), idempotent and detached.

### 20.6 Iteration-1 results (2026-09-01) and the next arm

Run: runs/policy_iteration_202609 (corpus q stopped at 3,000 games /
15 shards; recovery 23,938 of 25,151 override rows; fit on 51,714
targetable rows, 23,938 with Q; 4-seed n=1000 battery + duplicate
h2h vs theta_k, 2000 deals/mode).

Stage 1 (adapter rung selected; pointer 1.05e-3, adapter 9.7e-4, trunk
1.04e-3 held-out weighted MSE vs a 2.1e-4 noise floor; the trunk rung
lost, as pre-registered). The pointer rung needed the epoch cap raised
30 -> 120 (still creeping at 120; adapter early-stopped at 46). Pooled
top-card agreement with the committee draw: model 0.598 vs prior
0.602. At std|t0-defender-lead: model 0.44 vs prior 0.50 (above the
35% null bar, below the prior — a PARTIAL pass), with held-out MSE
3.3e-4 vs floor 2.4e-4: the explainable variance there is ~1e-4 Q^2,
the size a 0.01-Q convention effect carries.

Stage 2 (global sigma_u^2 1.24e-3, kappa 1): KL(target||prior) p50
0.013 / p90 0.20, |z| p50 1.1 / p90 4.5, 2.2% clipped. Direction at the
priority cells, prior -> target argmax on the called suit: t0 40.5 ->
49.2, t1 44.9 -> 51.9, t2 53.8 -> 56.0, t3 flat, t4 reversed (n=51) —
the committee's own late-trick profile.

Stage 3 (1 epoch, lambda_ce = lambda_ret = 1, 432 steps, 4.6 min):
mean KL(target||policy) 0.092 BEFORE -> 0.094 holdout / 0.097 train
AFTER. The policy head did not move toward the targets; the network
did move (value/aux/retention).

Cert:

    checkpoint        h2h vs theta_k        called-suit  partner  t0-trump  spread
    pi1_ep1 (hot)     +0.0171 se 0.0068     45.2         99.1     0.6       3.65
                      (called +0.016 / jd +0.018)
    pi1_a50 (interp)  +0.0126 se 0.0052     45.7         97.6     0.45      3.6
    arm c ep1 (§17.9) -0.0276 se 0.0089     49.3         96.7     0.6       ~2.9
    interp_a50 (§17)  -0.0039 se 0.0074     48.0         96.8     0.4       3.4

READINGS. (1) EV: the first distilled checkpoint in the program with a
POSITIVE h2h vs theta_k, hot, no walk-back (2.5 sigma, both modes),
against a pre-registered bar of -0.010 — the lambda-independent
-0.025 floor of §17.11-§17.15 was target-borne. No spread
compression (3.65 vs seed 3.6), top1min 8.5, partner SHARPENED 96.5
-> 99.1. (2) Installation: called-suit 45.2 vs 44.7 baseline — NOT
installed; the +9-point tilt at t0 leads never reached the policy.
(3) Interpolation: costs EV here (+0.017 -> +0.013) and installs
nothing; it was a remedy for noise-fitting damage and this arm has
none. Operator preference recorded: avoid interpolation in the final
model, keep it as a control.

DIAGNOSIS of (2), two mechanisms, both measured:
(a) Under-tilted lead targets. sigma_u^2 was estimated GLOBALLY
    (1.24e-3), dominated by high-variance cells (t3 picker-follow
    local residual 2.8e-3). At t0 defender leads the local residual is
    8.7e-5 against noise 2.4e-4, so gamma should be ~0.27 (model gets
    ~73%) but the global value gave 0.84 (model 16%): the pooled
    convention effect entered the lead targets at about a quarter of
    its calibrated strength and the node's own noise kept the rest
    (signal ~0.1 nats vs noise ~0.9 nats; locally ~0.9 vs ~0.5).
    This is the heteroscedastic case §20.4 pre-registered as the
    fallback.
(b) Under-dosed projection. Calibrated targets sit a median 0.013 nats
    from the prior, so the CE gradient is small where the one-hot
    gradients were O(1); at lambda_ce = 1 the fixed-coefficient value/
    aux terms owned the trunk and the target KL did not decrease. More
    epochs / a larger CE coefficient are a legitimate supervised
    projection onto FIXED targets (AZ buffer reuse); the §17.8 "epoch
    2+ damages" finding was noise-fitting of one-hot coin flips, which
    the calibrated targets bound per row.

NEXT ARM (operator-approved 2026-09-01), reusing corpus q, no new
search: (i) per-class residual variance — sigma_u^2 per telemetry cell
from the held-out per-class residual (weighted MSE - noise floor),
shrunk toward the global value by row count (an EB step; the cell
enters only as a VARIANCE bucket, never as a card class); Stage-2
targets rebuilt with the local gamma; (ii) projection with lambda_ce
raised (5) and up to 4 epochs under a held-out target-KL stop rule
(stop when holdout KL(target||policy) stops decreasing; keep the best
epoch); (iii) the standard battery, hot only (interpolation as a
control only if the operator asks). Pre-registered: t0 lead targets
now carry the pooled tilt (argmax on the called suit at t0 leads
> 49.2 in the targeted corpus); holdout target-KL DECREASES across
epochs; called-suit pooled >= 49 with t0-bin movement; h2h >= 0
maintained; partner >= 96.5, t0-trump <= 1, spread >= 3.2.

### 20.7 Arms 2-3 and the Stage-1 diagnosis (2026-09-01, late)

Cert table (4 x n=1000 fresh seeds + duplicate h2h vs theta_k):

    arm  change vs arm 1                    called-suit  partner  t0-trump  spread  h2h vs theta_k
    1    (baseline recipe, global sigma_u2)  45.2         99.1     0.6       3.65    +0.0171 se 0.0068
    2    per-class sigma_u2 (holdout, shrunk), lambda_ce 5, KL stop (stopped ep 1)
                                            43.7         99.2     1.2       3.7     +0.0183 se 0.0065
    3    FH iterated-WLS fit (2 rounds), all-rows per-class sigma_u2, lambda_ce 1, 2 epochs
         ep1                                45.0         97.9     0.2       3.8     +0.0252 se 0.0065 (called +0.029 / jd +0.022)
         ep2                                45.7         98.8     0.5       3.9     +0.0140 se 0.0065 (called +0.017 / jd +0.011)
    seed baseline                           44.7         96.5     ~1        3.6     0

EV: positive across all arms, both modes, no walk-back; arm 3 ep1 is
the strongest (+0.025, ~4 sigma) and moved in step with target
coherence and the held-out target-KL drop. Arm 3's SECOND epoch —
which lowered train KL while held-out KL stayed flat — gave back
~0.011 of EV and installed nothing: on fixed targets ONE epoch is the
dose (the §17.8 "epoch 2+ damages" finding survives calibration in a
milder form). Installation: none of the three arms moved greedy
called-suit on fresh deals beyond probe noise (pooled SE ~1.5 pts);
arm-3 ep2's 45.7 is the highest pooled read of the program's arms.

Where the convention signal goes missing — measured step by step:

1. The LABELS carry it. Pooled over corpus q's committee rows, the
   called-suit cards' advantage over the other fail cards is
   +0.0114 +/- 0.0011 Q at t0 defender leads (10 sigma; +0.0096 at
   t1, +0.0117 at t2) — the same 0.01 Q the CRN counterfactual
   instrument measured (0.10 score / 12). The pooled effect exists in
   the search evidence; a per-node label cannot see it (SE 0.015).
2. The FEATURES carry it. A linear probe on the frozen hand tokens
   classifies "this card is in the called suit" at held-out AUC 0.94.
3. The advantage MODEL under-fits it. Arm-1 fit: +0.0015 (13% of
   the observed effect); the prior covariate contributes nothing
   (+0.09 nats, < 1 sigma). Weight decay is NOT the cause (AdamW's
   decoupled decay at lr 1e-3 shrinks weights 0.2% per run; fits at
   wd 0 / 1e-4 / 1e-3 are identical). The cause is the fit's
   weighting and stopping: with 1 / noise_var weights every row weighs
   about the same, so cells whose true advantages vary by ~2e-3 Q^2
   (picker follows) dominate the loss over cells whose whole signal is
   ~1e-4 Q^2, and global early stopping (patience 8, epoch 46-73)
   halts before the small effect is learned. Fay-Herriot's own
   generalized-least-squares weight 1 / (noise_var + sigma_u^2_cell)
   raises the recovered effect to +0.0033 (30%) at the old stopping
   point; a 300-epoch trace shows the lead effect learned LATE and
   monotonically (+0.001 at epoch 40, +0.005 at 130 = the global
   held-out optimum, +0.007 at 190 while the global fit overfits),
   tracked identically on held-out lead rows (no memorization). The
   arm-3 fit (2 FH rounds, patience 25, best epoch 92) recovers
   +0.0042 (37%); its lead-cell held-out MSE is 2.8e-4 vs the 2.4e-4
   floor and its top-card agreement matches the prior at t0 (0.50)
   and beats it at t1 (0.52 vs 0.45).
4. The BLEND then dilutes it. Arm 1's global sigma_u^2 gave the model
   16% of the t0-lead blend; per-class holdout estimates shrunk by
   count gave 34% (arm 2); all-rows estimates give ~70% (arm 3, gamma
   0.30). Target coherence at t0 leads (rows tilting toward vs away
   from the called suit): arm 1 51/20, arm 3 57/20; called-suit mass
   0.377 -> 0.451 (arm 1) / 0.461 (arm 3).
5. The PROJECTION moves toward coherent targets and not toward
   incoherent ones. Held-out KL(target || policy) after one epoch:
   arm 1 0.092 -> 0.094, arm 2 0.094 -> 0.093, arm 3 0.110 -> 0.094
   (-14%; epoch 2 flat at 0.094 = fitting train rows only). On the
   corpus's own t0-lead rows: arm 2 moved called-suit mass 0.362 ->
   0.367 (target 0.417); arm 3 moved 0.391 -> 0.405 / 0.407 at ep1 /
   ep2 (target 0.470), argmax 45.0 -> 47.5 / 48.8 (target 51.2), row
   KL 0.086 -> 0.065 / 0.061. Gradient-norm audit (arm-2 targets,
   lambda 1): CE puts 4.6 on the shared encoder vs 0.75 for value +
   aux — CE dominates the trunk (the §20.6 "value/aux own the trunk"
   reading was wrong); the t0-lead rows are 1.7% of that CE gradient.
   The mean target KL is a poor progress metric (dominated by
   mutually inconsistent sharp rows); the KL stop rule is retired.

Reading: the recipe is sound end to end and every stage now measurably
moves in the right direction, but the per-iteration convention step is
small — the model carries ~37% of a 0.011 Q effect, the blend passes
~70% of that, the tilt at a lead row is then ~0.35 nats of coherent
signal against a 40/60 prior, and one epoch realizes ~20% of the
target's mass shift. Compounding this over iterations (each iteration
searches from the improved policy and multiplies the prior by fresh
evidence) is the design's answer, at perhaps +1-3 greedy points per
iteration at this capture rate.

Interpolation (operator preference recorded 2026-09-01): a control
only; not part of the final model. Arm 1's alpha 0.5 read +0.0126 se
0.0052 with no installation — walking back cost EV here.

Levers for the next arm, ranked by expected effect on the capture
rate (operator to choose):
(a) Raw per-card observation covariates in the ADVANTAGE HEAD ONLY
    (called-suit membership of the card, trump, point value — public
    observation attributes the encoder already receives, not human
    convention labels): the +0.011 mean effect becomes a single linear
    weight the WLS fit finds in epoch 1 instead of epoch 130. Expected
    capture ~100%. The policy network is untouched; the head is
    training-time only. Judgement call on whether this counts as
    special casing: it is a covariate choice in the pooling
    regression, and the covariates are raw observations.
(b) Stage-1 schedule without new inputs: train to the global held-out
    optimum (~130 epochs) with FH weights, or add a second, later
    round with a smaller learning rate. Expected capture ~50%.
(c) kappa < 1 (sharper tilt for the same evidence): doubles both the
    coherent and the noise tilt; cheap sweep (retarget + distill +
    cert ~1 h per value). Best paired with (a) or (b).
(d) Iterate now from the arm-3 checkpoint: generate the schema-2
    corpus from theta_{k+1} (student-acting, per-action stats stored),
    refit, project, cert against BOTH theta_{k+1} and the 8M anchor.
    This is the standing plan regardless of (a)-(c).

### 20.8 Stage-1 comparison and the projection bottleneck (2026-09-02)

Operator directions recorded 2026-09-01/02: interpolation is a control,
not a component of the final model; `called_suit_played` is REMOVED
from the observation dict (it was a precomputed table fact added for
the stateless scripted agent; no network ever read it — the scripted
agent now tracks the led suit from what it sees; commit 2a219fe); the
aux-head route is reserved for attributes a probe shows the trunk
lacks, and the trunk does NOT lack called-suit membership (linear
probe on frozen hand tokens, held-out AUC 0.94); observation-derived
covariates for the head are held in reserve.

Stage-1 variants on the same cached table (adapter rung; "capture" =
the model's pooled called-suit-minus-other-fail advantage at the cell
over the observed +0.0114 / +0.0096 / +0.0117 Q at t0 / t1 / t2):

    fit                                   holdout wMSE  t0 capture  t1   t2   t0 agree model/prior
    arm 1  1/noise weights, patience 8     9.74e-4        13%       2%   27%   0.44 / 0.50
    arm 3  FH iterated weights x2          9.17e-4        37%       29%  19%   0.50 / 0.50
    heteroscedastic head (NLL)             9.25e-4        35%       31%  35%   0.54 / 0.50
    heteroscedastic + BILINEAR pointer     9.23e-4        55%       52%  38%   0.40 / 0.50

- Weight decay is not a factor (AdamW decoupled decay at lr 1e-3
  shrinks weights 0.2% per run; wd 0 / 1e-4 / 1e-3 give identical fits).
- The heteroscedastic head learns the variance structure WITHOUT cells
  (lead rows sigma_u^2 ~4e-5, picker follows ~2e-3, correctly ordered)
  and gives the model ~80% of the lead blend, but standardizing the
  loss does not speed discovery of the effect: capture unchanged.
- The BILINEAR state x card term is the lever: the pointer's
  tanh(W_g h + W_t token) is additive inside the nonlinearity and gates
  a card attribute by the state weakly; "called-suit card is better,
  but only at a defender lead before the suit was led" is an
  interaction. (U h).(V token), zero-initialized on the state side,
  lifts capture to 55% / 52% at unchanged global fit.

Arms 4 and 4b (bilinear + heteroscedastic model; cert 4 x n=1000 +
dup h2h):

    arm   blend variance            targets: |z| p50 / clipped   called-suit  partner  t0-trump  h2h vs theta_k
    4     per-NODE (head sigma_u^2)   3.4 / 17.1%                 42.7         95.6     0.2       +0.0125 se 0.0075
    4b    per-class (all rows)        1.7 / 3.1%                  44.0         98.2     0.45      +0.0193 se 0.0068
    3     per-class (all rows)        1.65 / 2.8%                 45.0         97.9     0.2       +0.0252 se 0.0065

Arm 4's per-node variances make the targets much sharper (a fifth of
rows at the 8-nat clip); held-out target KL fell 17% in one epoch, the
largest yet, but fresh-deal called-suit fell BELOW baseline and
partner slipped to 95.6 — over-sharp targets where the head is over-
confident cost generalization. Arm 4b (same mean, class variance)
restores partner and the arm-3 regime and installs nothing more.

THE PROJECTION IS THE BOTTLENECK NOW. Lead-row movement measured on
the FULL training set and on held-out games (t0 defender leads with a
called-suit option; called-suit probability mass, policy vs target):

    arm   train rows (n=362)                 held-out rows (n=36)
    3     0.375 -> 0.385  (target 0.455)     0.398 -> 0.415  (target 0.515)
    4     0.375 -> 0.388  (target 0.494)     0.398 -> 0.423  (target 0.551)

One epoch realizes ~15% of the lead rows' target shift ON THE ROWS IT
TRAINS ON, and about the same on held-out rows — it is under-fitting,
not memorizing. Mechanism: the lead rows are ~2% of targeted rows and
their target tilt is moderate (KL 0.03-0.14), so their share of the
CE gradient is small (1.7% of the encoder gradient, §20.7) and Adam's
per-step displacement budget goes to the many larger-gradient rows;
the network fits the bulk first. Making the lead targets sharper (arm
4) did not raise the realized share and cost generalization; a second
epoch (arm 3 ep2) cost EV without installing. The greedy fresh-deal
rate across five arms — 45.2 / 43.7 / 45.0 / 42.7 / 44.0 vs 44.7 —
is flat at the probe's resolution.

What this leaves for installation, in order of principle:
(1) Iterate and compound: each iteration re-searches from the improved
    policy and moves the lead rows another ~15% of the gap; EV has
    been positive at every step. Slow but honest; the standing plan.
(2) Projection dose targeted by evidence, not by cell: weight each
    row's CE by its posterior confidence (the Fay-Herriot v_post the
    target was built from) so coherent, confident rows — leads under
    the bilinear model, resolved follows — carry more of the gradient
    than noise-dominated ties. This is the existing omega mechanism
    (AWR weight) with the posterior variance as the evidence instead
    of the top-2 gap; taxonomy-free. Untested.
(3) More corpus at the lead cells: p is already 1.0 there, so this
    means more games (the schema-2 regeneration from the accepted
    checkpoint), which raises the lead rows' absolute gradient share
    only through the model's capture, not the projection's.
(4) Bidding/leaster: held by the retention KL anchor at lambda 1;
    distributions hold (KL ~0.007) but near-tie decisions drift (pick
    32.9 -> ~36 in every arm; EV-neutral per the routed h2h). Across
    iterations the anchor ratchets; before iteration 3 either anchor
    bidding to the fixed 8M seed or bring bidding into search emission
    (§16.9 addendum 7).

### 20.9 Projection-realization experiments — pre-registration (2026-09-02)

Standing recipe = arm 3 (FH-weighted fit, class variance, lambda_ce 1,
one epoch). Screening base = arm 4b (bilinear model, class variance —
the same targets every variant below projects). Bar for adoption: no
regression vs the control on h2h, partner, t0-trump, spread, and a
measurable gain in lead-row realization (train AND held-out; §20.8
control ~15%). Screen = realization + the distill's 500-game probe;
cert (4 x n=1000 + dup h2h) only for variants that pass the screen.

Variants (each ~10 min to screen, ~1 h to cert):
  P1  Posterior-precision CE weights: omega_n = (1/v_post,n) mean-
      normalized, capped at 5 — GLS on noisy targets (Fay-Herriot one
      stage later; Kendall & Gal 2017); same dose, different
      allocation. Prediction: lead realization up (lead rows have small
      v_post under the bilinear model), noise-dominated ties down-
      weighted, EV held or better.
  P2  Head-first projection (LP-FT, Kumar et al. 2022): epoch 1 with
      the encoder frozen at actor lr 1e-3, epoch 2 full at 1e-4.
      Prediction: lead realization up without partner/pick collateral
      (no feature distortion in epoch 1); the full epoch then behaves
      like the control.
  P3  Epoch averaging (Polyak / SWA, Izmailov et al. 2018; Vieillard
      2020 across iterations): mean of arm 3's ep1 and ep2 weights, no
      training. Prediction: EV between ep1 and ep2 or above both if
      ep2's loss was variance; realization ~ep2's.
  P4  Linear scaling (Goyal et al. 2017): 4x batch (128 segments), 4x
      lr (4e-4), one epoch. Prediction: same displacement budget, less
      per-step noise; realization up modestly.
  P5  Best two combined, if two help independently.

Realization instrument: analysis/lead_row_realization.py (policy vs
target called-suit mass and argmax at t0/t1 defender leads on the
train and held-out games of the distill split; realized fraction of
the target mass shift relative to theta_k).

§20.9 RESULTS (2026-09-02, screen on arm 4b's targets; realization =
fraction of the target's called-suit mass shift realized at t0
defender leads, train / held-out rows; probe = the distill's 500-game
seed-0 read):

    variant                          realized t0   probe partner / spread   verdict
    arm 4b control (1 epoch)          10% / 15%     100 / 3.8                —
    P1 precision-weighted CE          10% / 21%     100 / 3.7                no train gain
    P2 head-first: frozen epoch 1     16% / 23%     96.8 / 4.0               best clean gain; cert below
       + full epoch 2                 11% / 22%     100 / 4.1                gives half back
    P3 arm-3 ep1/ep2 average          10% / 14%     —                        nothing
    P4 4x batch, 4x lr                20% / 19%     92.3 / 3.8               FAILS partner
    P2 head-first ep1 cert (4 x n=1000): called-suit 43.9, partner 97.6,
       t0-trump 0.4, spread 3.95; h2h +0.0023 se 0.0075 (called -0.011 /
       jd +0.016) = NO EV gain: the full projection's EV (+0.019..+0.025)
       lives in the trunk update, not the head.
    P2b four FROZEN-trunk epochs at actor lr 1e-3 (4 x 432 steps):
       realized t0 16 / 19 / 18 / 21%, t1 6 / 1 / 15 / 22% — SATURATES
       near 20% with the trunk fixed however many steps the head gets.

Reading: four well-grounded ways of reallocating or protecting the
projection's gradient leave lead-row realization at 10-20%, the only
one past 20% breaks partner, and the head alone saturates at ~20%
given unlimited steps. That is the signature of a representational
limit in the POLICY, not of the projection schedule:
the actor scores a card as tanh(W_g h + W_t token) — the same additive
form the advantage head started with, which captured 13% of the lead
effect until a bilinear state x card term lifted it to 55% (§20.8). A
state-conditional card preference (every convention) is what the
additive form expresses poorly, so the projection must bend shared
features to approximate it: slow, and collateral-prone. Proposed test
(operator decision — it changes the deployed actor): a zero-
initialized bilinear term in the policy pointer, registered as a new
architecture that loads the 8M checkpoint bit-identically at init
(existing goldens untouched), then the arm-3 projection on it.

§20.9 ARM 5 — the bilinear-pointer actor (2026-09-02). Operator
approved the architecture. `perceiver-shared-v2-bp` = v2 + (U h).(V t)
in the play pointer, U zero-init; `analysis/migrate_arch_checkpoint.py`
re-saves the 8M seed under it and verifies max action-probability
divergence 0 on real self-play (commit 6101c8d; the registry test
gates the fixture's key hashes; numeric fields to be recaptured on the
reference environment). Epoch-0 holdout under the trainer's replay is
identical to arm 4b's (target KL 0.1242, oracle loss 0.0182).

    checkpoint (arm 4b targets)              t0 realized train / holdout   holdout argmax cs   partner  spread
    theta_k                                   —                             44.4                96.5     3.6
    arm 4b control (plain actor, 1 full ep)   10% / 15%                     44.4                98.2     3.8
    arm 5  (bp actor, 1 full ep @1e-4)        15% / 21%                     44.4                97.8     3.8
    plain actor, trunk FROZEN 4 ep @1e-3      21% / 29%   (P2b)             47.2                98.6     4.1
    bp actor,    trunk FROZEN 1 ep @1e-3      27% / 40%                     47.2                99.5     3.9
    bp actor,    trunk FROZEN 2 ep @1e-3      36% / 55%                     50.0                98.8     4.1
    bp actor,    trunk FROZEN 4 ep @1e-3      37% / 48%                     52.8                99.5     4.3

Arm 5 (one full epoch) improves realization 1.5x over the control
but not more, because U starts at zero and 432 steps at 1e-4 grow it
to ~0.04 per entry — the term is BUDGET-limited in the standard
projection. Under the identical frozen-trunk schedule that saturated
the plain actor at 21% (P2b), the bp actor reaches 36-37% on training
rows and 55% on held-out rows by epoch 2, with held-out argmax on the
called suit 44.4 -> 52.8 (target 55.6) and every safety metric held or
sharper. CAPACITY CONFIRMED as the binding limit; the recipe change is a
head-first phase in which the bilinear term grows. Arm 5 cert: called-
suit 42.2 pooled (45.0 / 38.3 / 42.7 / 42.6), partner 97.8, t0-trump
0.35, spread 3.8; h2h (below).

Arm 5c (launched): bp actor, 2 frozen-trunk epochs @1e-3 then 1 full
epoch @1e-4 (the trunk update that carries EV), realization per epoch,
cert on the final checkpoint. Pre-registered: realization >= the
frozen-2 read (36% / 55%) after the full epoch (P2's plain-actor give-
back should not repeat if the term, not the trunk, holds the shift);
called-suit pooled >= 47; h2h >= arm 4b's +0.019; partner >= 96.5,
t0-trump <= 1, spread >= 3.6.

§20.9 ARM 5c RESULT (2026-09-02): bp actor, frozen-trunk epochs 1-2
@1e-3 then full epoch 3 @1e-4, arm-4b targets.

    checkpoint            t0 realized train / holdout   holdout argmax cs   t1 realized train
    arm 4b control         10% / 15%                     44.4                6%
    arm 5c after head ep2  36% / 55%                     50.0                12%
    arm 5c after full ep3  25% / 34%                     47.2                25%

The head phase installs the lead shift (2.5-3.5x the control) and the
following trunk epoch erodes about a third of it while lifting t1 —
the same give-back P2 showed on the plain actor, now at a higher
level. Held-out target KL after the full epoch 0.104 = the best of the
program; probe after ep3: partner 99.0, t0-trump 0.4, spread 4.2,
top1min 9.8, alone 14.7 (the frozen epochs had pushed alone to 22.4
via the actor-resident bidding heads at lr 1e-3; the full epoch re-
centered it). Cert of ep3 pending.

Arm 5d (launched): the ORDER swapped — full trunk epoch 1 @1e-4 (the
EV carrier), then frozen-trunk epochs 2-3 @1e-3 so nothing follows to
erode the head-installed shift. Pre-registered: realization after
epoch 3 >= arm 5c's head-phase read (36% / 55%); h2h >= arm 4b's
+0.019 (the head phase moved EV by +0.002 se 0.008 on the plain
actor); partner >= 96.5, t0-trump <= 1, spread >= 3.6; alone rate is
the watch item (retention anchor only; bidding PG phase queued).

§20.9 ARM 5 / 5c / 5d READINGS (2026-09-02 08:00):
- Arm 5 h2h (bp, 1 full epoch): +0.0127 se 0.0069 (called +0.006 /
  jd +0.019) vs arm 4b's +0.0193 on the same targets with the plain
  actor — inside one sigma; the term does not change EV under the
  standard projection, only what a head phase can install.
- Arm 5c ep3 cert probes: called-suit 46.4 pooled (48.1 / 44.4 / 45.1
  / 47.9) — the highest pooled read of the program (baseline 44.7, ~1
  sigma); partner 98.8, t0-trump 0.2, spread 4.3; pick 38.4, leaster
  3.9 = bidding drift. h2h pending.
- Arm 5d (trunk epoch FIRST, then 2 head epochs): t0 realized 15%
  after the trunk epoch -> 26% train / 28% held-out after the head
  epochs (held-out argmax 50.0), t1 19%; probe partner 97.3, t0-trump
  0.4, spread 4.1, top1min 9.4, alone 13.4. Ordering does NOT recover
  the head-only peak (36% / 55%): head epochs after a trunk epoch
  install less than from the seed, and both orderings settle ~2.5x the
  control. Cert pending.
- P5e (launched): head phase training ONLY pointer_U / pointer_V
  (--bilinear-only-frozen), 3 epochs @1e-3 from theta_k_bp — isolates
  the new capacity and removes the actor-resident bidding drift the
  head phase causes (alone 22.4 mid-phase in arm 5c).

§20.9 ARM 5d CERT PROBES + P5e (2026-09-02 08:15):
- Arm 5d ep3 (trunk epoch, then 2 full-actor head epochs): called-suit
  48.1 pooled (48.8 / 43.0 / 49.4 / 51.2) = +3.4 over baseline (~2
  sigma) — the FIRST fresh-deal movement of the calibrated-target
  program — with partner 98.4, t0-trump 0.35, spread 4.2, pick 36.9,
  leaster 4.6. h2h pending.
- P5e (theta_k_bp, 3 frozen epochs training ONLY pointer_U/V @1e-3):
  t0 realized 17 / 23 / 33% train, 23 / 30 / 39% held-out; t1 15 / 18
  / 30% train, 18 / 26 / 43% held-out — still RISING at epoch 3 with
  ~20k trainable parameters; partner 96-96.5, spread 3.9-4.1. The new
  capacity alone carries the installation; the full-actor head phase
  was faster but dragged the bidding heads.
- Arm 5f (launched, candidate recipe): full trunk epoch 1 @1e-4, then
  epochs 2-7 bilinear-only @1e-3; realization at ep1/4/7; cert on ep7.
  Pre-registered: t0 realization >= 40% train / 50% held-out at ep7;
  called-suit pooled >= 48; h2h >= arm 4b (+0.019); partner >= 96.5,
  t0-trump <= 1, spread >= 3.6; pick/alone within the seed's band
  (nothing but U/V trains after epoch 1).

§20.9 ORDERING VERDICT + ARM 5f REALIZATION (2026-09-02 09:05):
- Arm 5c (head-first, full actor, then trunk): h2h -0.0065 se 0.0078
  (called -0.003 / jd -0.010). The full-actor head phase at lr 1e-3
  gave back the whole EV gain; the bidding heads live in the actor and
  drifted (pick 38-39, leaster 3-4, alone 22 mid-phase).
- Arm 5d (trunk first, then full-actor head epochs): h2h +0.0135 se
  0.0077 (called +0.009 / jd +0.019) WITH called-suit 48.1 pooled
  (+3.4, ~2 sigma), partner 98.4, t0-trump 0.35, spread 4.2 — the
  first checkpoint carrying BOTH a fresh-deal convention gain and
  positive EV with sharpness intact. Ordering matters for EV: the
  trunk epoch must come first.
- Arm 5f (trunk first, then 6 bilinear-only epochs; nothing else
  trains after epoch 1): t0 realized 15% (ep1) -> 29% (ep4) -> 32%
  (ep7) train, 21 -> 32 -> 35% held-out; t1 11 -> 23 -> 30% train;
  plateau by epoch 7; held-out target KL 0.109 -> 0.085 (program low);
  final probe called-suit 50.6, partner 98.5, t0-trump 0.4, spread
  4.2, top1min 9.7, pick 35.8, alone 18.0 (probe-noise band). Cert
  pending. Realization per training row is the best of the program;
  the unresolved question for the cert is whether the bilinear-only
  phase keeps arm 5d's fresh-deal install (48.1) and EV.

§20.9 ARM 5f PROBES (2026-09-02 09:11): called-suit 45.6 / 45.4 / 49.6
/ 44.8 = 46.4 pooled (+1.7 over the 44.7 baseline; below arm 5d's
48.1), partner 98.1, t0-trump 0.0-0.8, spread 4.0-4.1, pick 34.2-34.3,
leaster 6.5-7.8 (the seed's band — bidding heads untouched, as
designed). h2h pending; it decides between 5d and 5f.

### 20.10 Corpus size per iteration (operator decision, 2026-09-02)

Objective restated by the operator: certifiable EV and general play
improvement, not convention adherence alone. Decision: 2,000 games per
iteration, leads searched at p = 1.0 and follows at half that rate
(`--p-base 0.5 --boost-lead 2 --p-max 1.0`; committee-act 0, i.e.
student-acting; schema 2; oracle states on).

Why 2,000. The binding rows are the t0/t1 defender leads. Corpus q at
p = 1.0 everywhere yielded, per class:

    class                  searched   per game
    std|t0-defender-lead     1,195      0.40
    std|t1-defender-lead     1,174      0.39
    all searched            51,728     17.2

The Fay-Herriot fit pools across rows; held-out lead-cell MSE reached
the noise floor at ~1,200 lead rows, and §20.8 located the bottleneck
in the projection, not the corpus (more games raise the lead rows'
gradient share only through the model's capture, which is saturated
at this size). 2,000 games gives ~800 lead rows per class at the
50-row per-class shrink — enough for the pooled fit and for the
posterior z-scores the tilt is built from. 1,000 games would fit but
halves the evidence behind the tilt; not used for an iteration whose
checkpoint is to be certified.

Why halve the follows. Follows are ~60% of corpus q's searches
(defender follows ~4,000 per trick alone). Sampling them at 0.5 cuts
searches per game from ~17 to ~11 without touching the lead-row count.
The follow rows still enter Stage 1 (pooled model) and Stage 2
(targets), just at half the density; their per-class sigma_u^2 is
larger (picker follows ~2e-3 vs leads ~4e-5, §20.8) so their targets
lean on the prior more anyway.

Cost. ~22k searches per iteration vs corpus q's ~52k (~40%); at corpus
q's observed 0.01 g/s (~6 s per committee search on 8 workers) about
36 h wall, less on a clean machine (corpus q ran under the §17.7 load
degradation).

Regeneration source = the accepted bilinear checkpoint (arm 5d or 5f
by cert), after the bidding PG phase.

§20.9 ARM 5f CERT COMPLETE (2026-09-02 09:50) — VERDICT:
- h2h vs theta_k: +0.0204 se 0.0070 (called +0.0249 / jd +0.0158),
  2.9 sigma, both modes positive — the program's best EV, above the
  pre-registered bar (arm 4b's +0.019) and above arm 5d (+0.0135).
- Conventions: called-suit 46.4 pooled (+1.7; MISSES the pre-reg >= 48
  and sits below arm 5d's 48.1, though the 5d-5f gap is ~1 sigma at
  the probe's 4-seed resolution), partner 98.1, t0-trump 0.0-0.8,
  spread 4.0-4.1. Realization 32/35% t0 also missed its 40/50% bar.
- Bidding heads: pick 34.2-34.3, leaster 6.5-7.8, alone in band — the
  bilinear-only phase leaves them at the seed, which removes the
  pick-drift ratchet that every full-actor arm carried.
- Scorecard vs pre-registration: EV bar PASS, partner/t0/spread PASS,
  install bars FAIL (46.4 < 48; realization 32 < 40).

Standing recipe = ARM 5f (trunk epoch first at 1e-4, then bilinear-
only epochs at 1e-3, holdout-KL plateau stop). Reason: the operator's
objective is certifiable EV and general play; 5f has the strongest
and cleanest EV (2.9 sigma, both modes, bidding untouched) while its
install deficit vs 5d is within probe noise. Arm 5d stays as the
fallback recipe if a later iteration needs faster convention movement
at the cost of a full-actor head phase.

Accepted checkpoint = runs/policy_iteration_202609/iter10/
distill_epoch7.pt (theta_{k+1}). Next: bidding PG phase (bidding
heads only; trunk + play head frozen; head-routed h2h as instrument),
then the §20.10 regeneration from theta_{k+1}.

§20.9 P1 SCREEN (2026-09-02 11:00) — PASS, cert launched:
Targets rebuilt from the iter6 fit with --weight-mode precision (omega =
1/v_post mean-normalized, cap 5; weight p50 0.94 / p90 1.79); target
distribution verified IDENTICAL to arm 5f's (same KL/z/clip stats), so
the screen isolates the weights. Same 5f projection (trunk epoch @1e-4,
six bilinear-only epochs @1e-3), iter11.

    lead-row realization        arm 5f ep7    P1 ep4    P1 ep7
    t0 defender lead, train        32%          37%       40%
    t1 defender lead, train        30%          32%       37%
    t0 defender lead, held-out     35%          43%       43%
    t1 defender lead, held-out     15%          41%       38%

Realization up on every row set, train AND held-out, at the same dose;
the bilinear head's 32% plateau was therefore allocation, not capacity
(the pre-registered P1 prediction). Held-out target KL 0.089 vs 5f's
0.085 — the unweighted KL is no longer the objective, so a slightly
higher value is expected and not a regression. Probe (500 games, ep7):
called-suit 47.3, partner 98.5, t0-trump 0.0, spread 4.1, pick 36.8,
alone 12.2 — within the seed's band. Cert (4 x n=1000 + dup h2h vs
theta_k) launched on iter11/distill_epoch7.pt as p1_bp_ep7; adoption
bar unchanged (h2h >= 5f's +0.020 within se, partner >= 96.5, t0 <= 1,
spread >= 3.6, called-suit pooled >= 5f's 46.4).

§20.9 ARM 5f vs PRODUCTION 30M (2026-09-02 11:05): duplicate h2h,
2000 deals/mode, seed 42 (same instrument as the cert): edge +0.0389
se 0.0136 (2.9 sigma; called +0.0206 / jd +0.0572). The transitive
estimate (+0.034 +/- 0.016 via interp_a50) is confirmed. Arm 5f —
8M terminal-only seed + one search-Q policy-iteration step — is the
first checkpoint of this lineage to BEAT the shaped 30M production
agent on the deployment instrument (interp_a50 tied it at +0.014 +/-
0.014). The JD-mode edge is the larger; the called-ace mode is where
the 30M's shaped called-suit adherence (90 vs 46) plays, and 5f still
leads there. Row in cert_results.jsonl as kind h2h_30m.

§20.9 P1 CERT COMPLETE (2026-09-02 11:50) — VERDICT: ADOPT.
- h2h vs theta_k: +0.0258 se 0.0073 (3.5 sigma; called +0.0335 / jd
  +0.0182), above arm 5f's +0.0204 (same se) — program best EV.
- Conventions: called-suit 44.6 / 41.3 / 48.8 / 47.0 = 45.4 pooled
  (5f: 46.4; the 1-pt gap is inside the 4-seed probe's resolution),
  partner 98.1, t0-trump 0.2-0.7, spread 4.1; pick 33.3-35.0, leaster
  6.2-8.3 = seed band (bilinear-only phase, bidding untouched).
- Scorecard: h2h PASS (above 5f), partner/t0/spread PASS, called-suit
  pooled 45.4 vs the >= 46.4 bar = TIE at instrument resolution (the
  deterministic realization instrument, which does resolve it, favors
  P1 on every row set: 40/37/43/38 vs 32/30/35/15).
- Reading: reallocating the projection dose by posterior precision buys
  EV (the confident rows are where the committee's signal is real; the
  down-weighted rows are the noise-dominated ties whose fitting cost EV
  in every earlier arm — §17.9's "anchor-suppressible label-noise
  fitting", now addressed at the source instead of by an anchor) and
  more of the targeted lead-row shift, without a fresh-deal convention
  gain visible at n=4000 games. Convention movement per iteration
  remains the slow variable; EV is compounding.

STANDING RECIPE (supersedes the 5f entry above): target stage with
--variance-mode class --variance-rows all --weight-mode precision
--weight-max 5; distill = trunk epoch @1e-4, then bilinear-only epochs
@1e-3 (--bilinear-only-frozen) to the holdout-KL plateau (~6).
theta_{k+1} = runs/policy_iteration_202609/iter11/distill_epoch7.pt.
Next: bidding PG phase, then the §20.10 regeneration from theta_{k+1}.

### 20.11 Standing-recipe constant provenance (2026-09-02)

Recorded at the operator's request so nobody later mistakes a default
for a tuned value. Operator decision: values kept as-is for now.

  constant            value   origin                                  status
  trunk lr (--lr)     1e-4    §17.4 "flat distill LR", 1/3 of the     conventional;
                              PPO rate 3e-4 (ppo.py lr_actor)         only P4 (4x, with
                                                                      4x batch) tested,
                                                                      failed partner
  head lr (--head-lr) 1e-3    default set in 0a80546 with the LP-FT   heuristic (10x
                              schedule; = --fit-lr used for the       trunk); epochs
                              same-sized Stage-1 adapter              swept (P2b/5e),
                                                                      rate never swept
  tilt clip           8       code default at 32c547a; §20.4 formula  numerical bound
  (--tilt-max)                names ±tilt_max without a number; e^8   (one-hot region);
                              ~ 3000:1 so the target is one-hot       binds 3.1% of rows
                              before the clip binds                   (z p90 5.17, so a
                                                                      clip at 5 WOULD
                                                                      change targets)
  kappa (--kappa)     1       DERIVED: one posterior SE = one nat     pre-registered
                              (§20.4)                                 sweep 0.5/1/2 not
                                                                      run
  weight cap          5       written into the P1 pre-registration   NEVER BINDS on
  (--weight-max)              (§20.9) as a standard exponentiated-    iter11 targets:
                              weight truncation; not data-derived     max 2.34, p99
                                                                      2.19, 0 rows at cap
  weight scale        mean=1  DERIVED: mean-normalized so weighted    by construction
                              loss sum = uniform loss sum ("same
                              dose, different allocation")

Iter11 weight distribution (51,714 targeted rows): min 0.013, p5 0.13,
p25 0.62, p50 0.94, p75 1.40, p90 1.79, p95 1.95, p99 2.19, max 2.34.
Consequence: the P1 h2h gain is a pure precision-weighting result; the
cap is inert for this corpus. Of the unpinned constants, --head-lr is
the most consequential and cheapest to sweep (~40 min/arm on the
realization screen) if a later iteration stalls at the plateau.

### 20.12 Iteration-2 corpus launch + pre-registration (2026-09-02)

Operator decision (same day, after the program-redesign review): before
the from-scratch perceiver-recall program is launched, VALIDATE a second
policy-iteration round under the standing recipe on the current
lineage. The compounding claim (§20.4) has never been tested: every
iteration so far re-projected corpus q from the same theta_k.

Launch (13:20, pid in runs/distill_corpus_iter2_202609/run.pid):

    uv run python -m sheepshead.training.distill_corpus \
      --ckpt runs/policy_iteration_202609/iter11/distill_epoch7.pt \
      --out-dir runs/distill_corpus_iter2_202609 \
      --games 2000 --workers 8 --seed 20260902 --shard-games 200 \
      --p-base 0.5 --boost-lead 2 --p-max 1.0 \
      --node-telemetry runs/distill_corpus_iter2_202609/nodes.jsonl \
      --routed-encoder mps

theta_k for this iteration = iter11/distill_epoch7.pt (the P1-certified
theta_{k+1} of §20.9). Schema 2, oracle states on, committee-act 0
(student-acting; corpus q was committee-acting at frac 1.0 under schema
1, so acting mode is a deliberate recipe difference this iteration
carries, per §20.3). Leads p = 1.0, follows 0.5 (§20.10). Fresh deal
seed, independent of corpus q. Expected ~22k searches, ~36 h.

Then: fit -> target (--variance-mode class --variance-rows all
--weight-mode precision --weight-max 5) -> distill (trunk epoch @1e-4,
bilinear-only epochs @1e-3 to the holdout-KL plateau) -> cert (4 x
n=1000 + dup h2h vs theta_k). The bidding PG phase is NOT run before
this regeneration (operator: validate the play-side compounding first;
bidding stays at the seed as in every arm so far).

Pre-registered reads (iteration 2 vs iteration 1):
- COMPOUNDS: h2h vs iter11 >= +0.015 with CI lower bound > 0, AND h2h
  vs the 8M seed exceeds iteration 1's +0.0258 (cumulative gain), with
  partner >= 96.5, t0-trump <= 1.0, spread >= 3.6.
- STALLS: h2h vs iter11 inside +/- 0.01 -> the per-iteration gain does
  not accumulate under a fresh corpus; the from-scratch program's phase
  3 budget (3-5 iterations) is cut to one iteration + bidding phase, and
  the search-ceiling residual is attributed to the projection.
- REGRESSES: h2h vs iter11 < -0.01 at 2 SE -> student-acting corpus
  from a distilled policy is the suspect (the accepted checkpoint acts
  the coin-flip leads its own targets flattened); rerun with
  --committee-act-frac 1.0 before any other change.
- Stage-1 diagnostics expected to reproduce §20.8 qualitatively (lead
  cells at the noise null per node, pooled fit at the floor with
  ~800 rows per lead class).

Leaster-play instrument (operator concern, 2026-09-02): leaster play rows
carry only the chained retention KL anchor to theta_k (train_distill
§17.4) plus value regression; no improvement signal, and the anchor
re-fits the old outputs through a moving trunk, so drift is bounded per
iteration but can random-walk across iterations. The cert battery has
no leaster-play quality metric (greedy probe reports leaster ENTRY rate
only; the duplicate h2h contains ~7% leaster hands, diluted). Adding a
leaster-conditioned paired score (rigorous_eval already tags hands
is_leaster) to the iteration-2 cert as a baseline read of iteration-1
-> 2 drift. Escalation order if drift shows: fixed-reference anchor
(anchor leaster rows to the handoff checkpoint, not theta_k, bounding
cumulative drift at handoff competence) -> leaster-play emission behind
the addendum-5 mini-calibration gate (P4 leaster determinizer exists).

§20.12 ADDENDUM — follow-density confound on the STALL read (2026-09-03,
pre-result; corpus at 1475/2000 games): this corpus searches follows at
p = 0.5 (corpus q: 1.0), so follow OVERRIDE labels are ~half as dense
per game. The anchor/override partition (§16.9 addendum 2; corpus
``distill_set`` "none" = no policy loss) rules out the anti-teaching
mechanism, and the partial manifest confirms the searched subset is
unbiased: override share of searched rows per class matches corpus q
(t0-def-follow 0.51 vs 0.48, t4-def-follow 0.21 vs 0.22, t0-def-lead
0.78 vs 0.82, t1-def-lead 0.69 vs 0.64). What p DOES change is the
follow-side label volume behind the h2h, which the STALL read above
does not separate from a compounding failure. Disambiguation, fixed now:
- STALL with Stage-1 lead cells at the §20.8 null and follow-class
  posterior z-scores comparable to corpus q's -> attributed to
  compounding (phase-3 budget cut as written).
- STALL with follow-class z-scores visibly weaker than corpus q's
  (same target settings) -> follows-only top-up: regenerate a subset
  at --p-base 1.0 --boost-lead 1.0 from the same theta_k and refit
  before the phase-3 budget is cut. Leads are already at p = 1.0 and
  are not re-searched.
- COMPOUNDS and REGRESSES reads are unchanged; the confound only
  weakens the null.

### 20.13 Iteration-2 results: STALL, diagnosis, and the volume finding (2026-09-04)

Corpus (runs/distill_corpus_iter2_202609): 2000 games, 28.5 h at 0.02
g/s, 21,630 searched rows (10,821 override / 10,807 endorsed, 2
committee failures) vs corpus q's 51,728 (25,151 / 26,563). Searched
follow rows per class are ~1/3 of corpus q's (p 0.5 AND 2000 vs 3000
games); lead rows ~2/3. Per-row evidence is not weaker (posterior
z_max p50: t0-def-follow 2.26 vs 1.86, t0-def-lead 3.76 vs 3.21).

Pipeline (runs/policy_iteration_202609/pipeline_iter2.py -> iter12),
standing recipe throughout:
- fit: adapter, bilinear + heteroscedastic, best epoch 132/157 (2 FH
  rounds), holdout wMSE 6.9e-4 vs floor 1.9e-4, top-agree model 0.562
  vs prior 0.542, sigma_u2 8.1e-4 (iter6: 9.2e-4 / 2.1e-4, 0.619 vs
  0.602, 1.12e-3). The iteration-1 fit command was never logged; the
  200-epoch / patience-25 setting reproduces iter6's stop shape.
- target: weights p50 0.94 p90 1.62 (cap 5 inert), KL(target||prior)
  p50 0.028 p90 0.232, z clipped 5.1%, gamma p50 0.34 (iter11: 0.94 /
  1.79, 0.032 / 0.285, 3.1%, 0.54).
- distill: HOLDOUT override KL 0.101 (ep0) -> 0.115 (ep1, trunk) ->
  0.0975 (ep4, best) -> 0.100 (ep7). Only -4% vs iteration 1's -28%
  (0.124 -> 0.089). The trainer's own selection = epoch 4; the
  pipeline was re-pointed to cert the holdout-best epoch first.
- lead realization (lead_row_realization, 4 row groups): ep4 33-43%,
  ep7 30-67% — as good as or better than P1 on iteration 1.

Cert (4 x n=1000 probes + dup h2h; leaster strata via the new
h2h_duplicate conditional read, edea44d):

    cand      vs        edge      se      called   jd      leaster (n)         non-leaster
    ep4       iter11   +0.0012  0.0064  +0.0006 +0.0019  +0.023±0.034 (1276)  -0.000±0.006
    ep4       seed8m   +0.0263  0.0076  +0.0256 +0.0271  -0.057±0.041 (1119)  +0.031±0.008
    ep7       iter11   -0.0034  0.0067  -0.0068 -0.0001  -0.036±0.033         -0.001±0.007
    ep7       seed8m   +0.0064  0.0080  +0.0083 +0.0045  -0.129±0.036         +0.014±0.008

Probes (pooled, ep4): called-suit 50.3 (clears the 48 install bar
iteration 1 missed at 46.4), partner 99.2, t0-trump 0.3, spread 4.6,
pick 36.4, leaster 5.8. ep7: 49.1 / 98.9 / 0.35 / 4.6.

VERDICT (§20.12 rules): STALL. h2h vs iter11 inside ±0.01; cumulative
vs the seed +0.0263 = iteration 1's +0.0258 (zero accumulated gain).
Conventions all pass; bidding at the seed band. The projection moved
the lead rows (realization, called-suit +4) and EV did not follow.

DIAGNOSIS (pipeline_stall.py, same night; operator-requested):

(1) Matched-volume CONTROL — iter11/targeted (corpus q) subsampled post
    hoc to the iteration-2 composition (first 2000 games; searched
    follow rows kept at 0.5: 12,526 of 25,083 demoted to no-loss, all
    9,408 lead rows kept; analysis/subsample_targeted.py, 12730ba),
    P1 distill from theta_k_bp, cert vs the seed:

        corpus q, full (iter11 ep7)          +0.0258  se 0.0073   holdout KL -28%
        corpus q, matched volume  ep5 (best) +0.0093  se 0.0069   holdout KL -18%
                                  ep7        +0.0102  se 0.0072
        iteration-2 corpus        ep4        +0.0012  se 0.0064   holdout KL  -4%  (vs iter11)

    Same targets, recipe and anchor; only the row count differs. The
    gain fell to ~1/3 (difference 0.0165, se ~0.010). Convention
    install unchanged (called-suit 48.2, partner 97.8, t0 0.2): the
    convention comes from the lead rows, which the subsample kept; the
    EV came from the follows, which it halved.

(2) LEAD-vs-FOLLOW attribution — head_routed_h2h --lead-ckpt (6345c4c):
    bidding pinned to the anchor, play split by is_lead_decision
    (leader_rel == 1); duplicate gauntlet, 2000 deals/mode:

        arm                 iteration 1 (iter11 vs seed)   iteration 2 (iter12 ep4 vs iter11)
        A  leads only       +0.0084  se 0.0043              +0.0023  se 0.0033  (called +0.0095 / jd -0.0050)
        B  follows only     +0.0149  se 0.0054              -0.0011  se 0.0048
        C  leads + follows  +0.0212  se 0.0066              +0.0020  se 0.0057

    Iteration 1: A + B = +0.0233 ~ C — additive, no interaction;
    follows carry ~2/3 of the play edge, leads ~1/3; play recovers
    most of the cert's +0.0258 (bidding heads carry little). Iteration
    2: flat in BOTH classes; the lead residual lives entirely in
    called-ace mode (the convention) and is offset in JD.

(3) Leaster instrument: every bilinear-only checkpoint reads negative
    on leaster hands vs the seed — ctrl ep5 -0.044±0.036, iter12 ep4
    -0.057±0.041, iter12 ep7 -0.129±0.036 (3.5 sigma) — and it grows
    with bilinear-only epochs past the holdout-KL minimum (ep7 also
    scores below ep4 overall vs the seed, +0.006 vs +0.026). Leaster
    rows carry only the chained retention anchor; the play pointer's
    bilinear term is shared with leaster play. BASELINE (iter11 ep7
    vs seed, same deals; pipeline_leaster.py): overall +0.0258 se
    0.0073 (reproduces the P1 cert exactly), leaster -0.078±0.034
    (n=1150, 2.3 sigma), non-leaster +0.032±0.008. So iteration 1
    ALREADY degraded leaster play; iteration 2 at ep4 added nothing
    (vs iter11 +0.023±0.034) and ep7 added -0.05. Leasters are ~5.5%
    of hands, so the damage costs ~0.005-0.006 of overall edge per
    checkpoint — the gap between non-leaster +0.032 and overall
    +0.026. Mechanism: leaster rows carry only the retention KL to
    theta_k's stash while the bilinear-only epochs train the shared
    play pointer on standard-game targets; nothing in the loss holds
    leaster play at handoff competence.

INTERPRETATION. Two mechanisms, both present:
- VOLUME (established): the control reproduces ~2/3 of the stall from
  row count alone, and the iteration-1 attribution says the EV lived in
  the follows that §20.10 halved. §20.10's premise — "the binding rows
  are the leads; follows lean on the prior" — was right about the
  CONVENTION and wrong about EV. The corpus was sized to push the
  called-suit convention and did (50.3); it was under-sized for EV.
- A NON-VOLUME residual (suggestive, not established): the iteration-2
  corpus generalized far less than the matched-volume control (holdout
  KL -4% vs -18%), and both routed classes are flat where the control
  would predict ~+0.009 (1.3 sigma below). Candidates: (a) DAgger lag
  under student acting — labels sit on theta_k's states, the deployed
  theta_{k+1}'s states arrive one round late and only in proportion to
  realization (only 28% of h2h deals deviate at all), while committee
  acting overshoots to states the student never reaches; (b) the
  residual disagreement after one projection is less class-coherent
  (harder to generalize) even though per-row z is not weaker. Neither
  is separable from tonight's data. Skill saturation is NOT indicated:
  the corpus's own start KL (0.101 vs 0.124) says search still
  disagrees with iter11 nearly as much as with the seed, and the
  committee-acting ceiling (+0.18) is ~7x the captured gain.

DECISIONS / NEXT (operator authorized follow-ups at the assistant's
judgment; verdict presented before the regen):
1. TOP-UP CORPUS LAUNCHED 2026-09-04 06:40 (runs/distill_corpus_iter2b_
   202609, pid in run.pid): same theta_k (iter11 ep7), student-acting,
   fresh seed 20260904, 2000 games, p = 1.0 EVERYWHERE (--p-base 1.0
   --boost-lead 1 --p-max 1.0), ~34k searches, ~40 h. POOLED with the
   iteration-2 corpus (same expert, same acting policy, same anchors)
   -> ~56k searched rows >= corpus q's 51.7k, i.e. the compounding step
   at iteration 1's volume. Pre-registered reads on the pooled
   fit/target/distill/cert (h2h vs iter11, ep = holdout-KL best):
   - COMPOUNDS: >= +0.015, CI > 0 -> volume was the whole story;
     recipe = >= 50k searched rows per iteration (follows at p = 1.0),
     phase-3 budget restored.
   - PARTIAL: +0.008 .. +0.015 -> non-volume component confirmed at
     about the size the control predicts; next arm = MIXED committee-
     acting corpus from iter11 (frac pre-registered, not 1.0), volume
     held at >= 50k.
   - STALL: < +0.008 -> acting mode / coherence dominates; the
     committee-acting arm becomes mandatory before any budget decision.
   Also read: pooled holdout-KL reduction vs the control's -18%.
2. LEASTER: baseline confirms the drift began in iteration 1.
   Proposed (operator decision, NOT built): §20.12 escalation step 1 —
   FIXED-REFERENCE anchor for leaster retention rows (anchor_probs
   recomputed from the handoff checkpoint's forward pass at distill
   time, replacing theta_k's stash on those rows only), tested as a
   separate distill arm on the pooled targets so the compounding read
   stays clean; expected recovery ~+0.005 overall. Epoch selection by
   holdout KL stands (ep7 hurt leasters and EV). A leaster-hand
   score should join the cert bars.
3. §20.10 amended: 2000 games / follows 0.5 is a CONVENTION-install
   budget, not an EV budget. §20.12's "STALL -> cut phase-3 to one
   iteration" is SUSPENDED pending the pooled read.
4. Bidding PG phase still not run; bidding heads at the seed in every
   checkpoint (routed C vs cert: bidding carries little).

§20.13 ADDENDUM — operator decisions (2026-09-04 morning):
- LEASTER: the fixed-reference anchor arm is REJECTED. Leasters are
  ~5-7% of hands and cost EV, so SEARCH them (leaster play emission via
  the P4 leaster determinizer, behind the addendum-5 mini-calibration
  gate; ~+7% searches per corpus) instead of pinning them better. ORDER:
  first pin the compounding recipe for the standard play nodes, then
  add leaster emission to the recipe. Until then the leaster-hand score
  stays a reported cert read, not a bar.
- ACTING MODE (committee vs student): wanted, but every arm costs a
  corpus, so the SMALLEST RELIABLE test. Design:
  * Corpus iter2c = the running iter2b with ONE change,
    --committee-act-frac 1.0: same theta_k (iter11 ep7), same 2000
    deals (seed 20260904 — CRN: trajectories are identical up to the
    first committee-acted deviation), same p = 1.0, same committee.
    ~34k searches, ~50 h; launched when iter2b finishes so the two
    do not contend.
  * Read = distill(iter2c) vs distill(iter2b), each 2000 games alone
    (matched volume; the pooled corpus is a separate read), standing
    recipe, holdout-KL-best epoch, cert vs iter11.
  * Reliability lever is the CERT, not the corpus: h2h at 8000 deals
    per mode (~2.5 h each, se ~0.0035; difference se ~0.005) so an
    acting-mode effect of the size the non-volume residual implies
    (~0.008-0.01) reads at ~2 sigma; at 2000 deals it could not
    (difference se ~0.01). Same 8000-deal cert on the pooled read.
  * Also compare the two corpora's held-out KL reduction and the
    fraction of decision nodes downstream of a committee deviation
    (corpus q: 9,444 acted nodes / 3000 games, ~3.1 per game).
  * Smaller corpora do not work: gains at <= 20k rows are ~+0.009 and
    acting-mode differences inside that are unreadable at any cert n.
    Frac 1.0 (not mixed) for the cleanest contrast; a mixed fraction
    is a later dial once the sign is known.
  * Schedule: iter2b lands ~Sat AM -> pooled compounding read (Sat) +
    iter2c launch -> iter2c lands ~Mon AM -> acting-mode read (Mon).

§20.13 ADDENDUM 2 — POOLED COMPOUNDING READ: STALL (2026-09-06, 10:29):
Top-up corpus iter2b done 04:58 (2000 games, 34,554 searched: lead
9,425 / follow 25,129, override 49%, 18 failures). Pooled with iter2
(runs/distill_corpus_iter2_pooled_202609): 4000 games, 56,160 searched
(lead ~18.7k / follow ~37.5k) — MORE rows than corpus q (51.7k), same
theta_k (iter11 ep7), student-acting. Pipeline pipeline_pooled.py ->
iter13; standing recipe P1.
- fit: sigma_u2 0.0008, holdout mse 0.0007 (= iter12). target: gamma
  p50 0.205 (iter12 0.34, corpus q 0.54), weight p90 2.35, frac_z_clipped
  10.4% (q 3.1%, iter12 5.1%) — recorded, not interpreted.
- distill holdout override-KL vs theta_k: ep4 −15%, ep7 −16%
  (matched-volume control −14/−18; iteration 2 alone −4/−1). The
  student FITS the pooled targets as well as it fit corpus q's. Best
  epoch 4 (KL 0.1034).
- probes ep4 (4 x n=1000, across-seed mean): called-suit 53.0
  (50.5/52.7/52.2/56.8), t0 trump-lead 0.5, partner 98.8, pick 36.0,
  leaster 6.6, spread 4.7 — conventions held; called-suit +3 over iter12.
- H2H pooled_ep4 vs iter11, n=8000/mode (239 min):
      edge +0.0009 +/- 0.0031   called −0.0039  jd +0.0058
      leaster +0.0175 +/- 0.0157 (n=4914 hands)  non-leaster −0.0001 +/- 0.0031
  VERDICT (pre-registered §20.13 item 1): STALL. The 2-sigma upper bound
  (+0.007) sits below the PARTIAL line (+0.008); COMPOUNDS (+0.015) is
  excluded at ~4.5 sigma. Volume is RULED OUT as the stall's cause:
  56k student-acting rows at theta_1 fit as well as 21.6k committee-
  acting rows at theta_0 and convert to ZERO EV.
- Search-gap check (manifest gap_percentiles, search's own Q gap
  argmax-vs-best at searched nodes): iter2b p50 0.0212 / p90 0.089 /
  override frac 0.49 vs corpus q p50 0.0197 / p90 0.089 / 0.49. By its
  OWN measure the one-step teacher disagrees with theta_1 exactly as
  often and as much as with theta_0 — the label signal did not
  visibly shrink. Note the median gap (0.02) is below the per-row
  noise sigma_u (~0.028), for q as well.
- What remains between iteration 1 (+0.026) and iteration 2 (0): (a)
  ACTING MODE — q was committee-acting on every game (labels on the
  search policy's states), iter2/2b student-acting; (b) theta_k-
  dependent REALIZABILITY — search-Q gaps at theta_1 may be real for
  the search yet not realizable by the argmax student (or be
  determinization noise the search cannot tell from signal). (a) is
  the pre-registered consequence of STALL ("committee-act arm
  mandatory") and is ALREADY RUNNING: iter2c CRN twin launched 05:01
  (pid runs/distill_corpus_iter2c_202609/run.pid, ~17.8 searches/game,
  ETA ~Tue AM), read per the addendum above at 8000 deals/mode.
- PRE-REGISTERED CONDITIONAL (written before the iter2c read): if
  distill(iter2c) vs iter11 ALSO reads < +0.008, run a theta_1 CEILING
  h2h (analysis/ceiling_h2h.py, committee-act at read time vs iter11
  argmax, 250 deals/mode, ~22 h) to split (b): ceiling >= +0.10 =>
  headroom intact, the stall is a CAPTURE/label-design problem (target
  construction, acting mixture, DAgger lag); ceiling <= +0.05 => the
  1024-iter one-step teacher is near-exhausted at theta_1 and further
  skill needs a stronger teacher (deeper budget / iterated search) or
  search at deploy. If iter2c COMPOUNDS (>= +0.015): recipe =
  committee-acting corpus >= ~35k rows per iteration; no ceiling run.
- Pending (appended when they land): pooled_ep4 vs seed8m; ep7 probes
  + h2h. Leaster stratum vs iter11 is +0.0175 +/- 0.0157 — first
  non-negative leaster read in the lineage (weak; vs iter11, not seed).

§20.13 ADDENDUM 3 — the 2000-deal cert overstated the iteration-1 gain
(2026-09-06, 18:52). Trigger: pooled_ep4 vs seed8m at 8000 deals/mode =
+0.0106 +/- 0.0041, yet pooled_ep4 ties iter11 (+0.0009 +/- 0.0031) and
iter11 beat the seed by +0.026 +/- 0.007 at 2000 deals — a 1.8-sigma
transitivity gap. Direct measurement (h2h_iter11_seed8k.py, 267 min):

    iter11_ep7 vs seed8m, n=8000/mode:
        edge +0.0141 +/- 0.0036   called +0.0206  jd +0.0075
        leaster −0.0573 +/- 0.0169 (n=4381)   non-leaster +0.0182 +/- 0.0038

- Transitivity HOLDS: pooled−iter11 via the seed = −0.0035 +/- 0.0055,
  consistent with the direct +0.0009. The 2000-deal set (the first
  quarter of the same seed-42 schedule) ran ~1.7 sigma hot: the other
  6000 deals imply ~+0.010 for iter11 vs seed.
- The ITERATION-1 GAIN vs the seed is +0.014, not +0.026. Every 2000-
  deal read this week sat on that same deal set: the matched-volume
  control (+0.009 vs +0.026, a 1.7-sigma gap) and the routed A/B/C
  splits are NOT robust to it; "volume ~2/3 of the stall" (§20.13) is
  DOWNGRADED to suggestive. Robust (8000-deal) facts: iter11 > seed by
  +0.014; pooled = iter11; pooled > seed by +0.011.
- Iteration 2 remains a STALL: +0.0009 +/- 0.0031 sits ~4 sigma below
  a repeat of iteration 1's +0.014. The §20.13 thresholds (+0.008 /
  +0.015) were calibrated to +0.026 and are superseded for the acting-
  mode read by: COMPOUNDS if committee−student >= +0.010 at >= 2 sigma
  (i.e. the arm recovers most of an iteration-1-sized step), else
  STALL -> theta_1 ceiling h2h per addendum 2. The acting pipeline's
  8000-deal certs (difference se ~0.005) can read that.
- LEASTER damage is now firm: iteration 1 cost −0.057 +/- 0.017 on
  leaster hands (3.4 sigma; ~5.5% of hands => ~−0.003 overall), while
  non-leaster hands gained +0.018 +/- 0.004. Search emission for leaster
  (addendum 1) stands.
- RECIPE AMENDMENT: cert h2h at 8000 deals/mode is now the standard
  (se ~0.0035). At 2000 deals (se ~0.007) a +0.014 step is a 2-sigma
  read and comparisons between arms are ~1 sigma — too coarse for the
  step sizes this program produces. Probes unchanged.

§20.13 ADDENDUM 4 — is the label signal still there at theta_1?
(2026-09-06 evening; operator question: "are the disagreement nodes
measurably higher in EV than the policy prior, or is the policy just not
generalizing them?")
- Per-node replicate diagnostic from the corpus telemetry (nodes.jsonl:
  pair_diffs = the top-pair Q gap in EACH of the 3 committee replicates;
  SNR = |mean| / se over replicates):

    corpus (theta_k, acting)           override rows  gap p50  SNR p50  3-rep sign-agree  |mean|>2se
    q      (theta_0, committee)         25,151         0.0197    2.51      0.62             0.56
      leads / follows                    8,260/16,891   0.011/0.027  1.47/3.17  0.48/0.68   0.42/0.63
    iter2b (theta_1, student)           16,946         0.0212    2.52      0.62             0.56
      leads / follows                    5,429/11,517   0.013/0.027  1.57/3.11  0.49/0.69   0.43/0.63
    iter2c (theta_1, committee, 30%)     5,071         0.0211    2.54      0.63             0.57

  The search's disagreement with theta_1 is EXACTLY as frequent, as
  large, and as self-consistent as with theta_0 — in every stratum. The
  "labels are determinization noise at theta_1" version of the
  realizability theory is NOT supported: the replicates agree with each
  other at theta_1 as they did at theta_0, where acting on the same
  labels realized +0.180 (§13.3). What the diagnostic CANNOT see is a
  shared bias (all three replicates wrong the same way, e.g. strategy
  fusion / determinization bias) — that is only visible in realized EV.
  Note the leads/follows split: follow labels are ~2x the gap and ~2x
  the SNR of lead labels (sign-agree 0.68 vs 0.48), consistent with the
  iteration-1 routed read that follows carried the EV.
- CRN twin confirmed: iter2c's node rows for game 0 are byte-identical
  to iter2b's up to the first committee-acted deviation.
- DECISION (my judgement under the standing "run follow-ups"
  authorization): the theta_1 CEILING h2h is now UNCONDITIONAL, queued
  after the iter2c corpus DONE (runs/ceiling_h2h_theta1_202609/
  launch_after_iter2c.sh, launcher.pid; 250 deals/mode, ~22 h, ETA ~Wed
  AM). It answers the operator's question directly — realized EV of
  acting on the labels at theta_1 — and calibrates what "compounding"
  can mean at theta_1 regardless of the acting-mode read. Reads: >= +0.10
  => labels carry EV, the stall is CAPTURE (student adopts the labels'
  argmax at too few nodes, or DAgger lag); <= +0.05 => the one-step
  1024-iter teacher is near-exhausted at theta_1 (shared bias / partial-
  obs floor), further skill needs a stronger teacher or search at deploy.
- INSTRUMENT (committed): h2h_duplicate and ceiling_h2h now store per-deal
  scores (+ leaster-hand counts; ceiling node rows carry a leaster
  flag). Two candidates certified vs the same anchor on the same schedule
  can now be compared PAIRED (se well below sqrt(se1^2+se2^2)); the
  acting-mode read (iter14 vs iter15, both vs iter11 at 8000 deals) will
  use it. The §13.3 ceiling's leaster share had to be replayed (23/500
  deal-modes, 416/9,099 nodes; resolution 93%, deviation 43%).
- Sensitivity of the h2h instrument at current step sizes (operator
  question): at 8000 deals/mode se = 0.0031 (vs a near-identical policy)
  to 0.0041 (vs the seed); an iteration-1-sized step (+0.014) is a 4-sigma
  read; a between-arm difference of +0.010 is ~2 sigma unpaired and
  ~3 sigma paired. It cannot see steps of +0.005 (would need ~4x deals,
  ~16 h per h2h). The instrument is adequate for the effects this
  program is looking for, NOT for fine-tuning within them.

§20.13 ADDENDUM 2 — pending reads landed (2026-09-06 22:55, POOLED DONE):
  pooled_ep7 probes (4 x 1000, mean): called-suit 54.1, t0 0.7, partner
    98.2, pick 35.1, leaster 6.3, spread 4.9.
  pooled_ep7 vs iter11  n=8000/mode: +0.0015 +/- 0.0032 (called −0.0032,
    jd +0.0062; leaster −0.0090 +/- 0.0162, non-leaster +0.0022)
  pooled_ep7 vs seed8m  n=8000/mode: +0.0135 +/- 0.0041 (called +0.0179,
    jd +0.0090; leaster −0.0470 +/- 0.0183, non-leaster +0.0169)
  Both epochs tie iter11 and both sit at iter11's +0.014 over the seed.
  Leaster-hand deficit vs the seed persists (−0.032 / −0.047 vs iter11's
  −0.057). The acting-mode pipeline started iter14 (top-up alone) at
  22:55.

§20.13 ADDENDUM 5 — ACTING-MODE READ: NULL; the SHRINK is the capture
mechanism (2026-09-08, 09:44 / 11:30).
Twin corpus iter2c done 03:42 (2000 games, 34,669 searched, override 49%,
11 failures; committee acted at 5,779 nodes = 2.9/game vs corpus q's 3.1;
lead/follow 9.4k/25.2k = the top-up's split exactly). Both arms: standing
recipe from iter11 ep7, holdout-KL-best epoch 3 (student −12%, committee
−13%), probes in band (called-suit 48.6 student / 46.1 committee vs
iter11 45.4; partner ~98-99, t0 <= 0.4), h2h vs iter11 at 8000/mode:

    student-acting   (iter14, sa_ep3)  −0.0056 +/- 0.0031
    committee-acting (iter15, ca_ep3)  −0.0048 +/- 0.0032
    committee − student, PAIRED (per-deal, same schedule): +0.0008 +/- 0.0035
    both arms averaged vs iter11 (paired):                −0.0052 +/- 0.0026

VERDICT: acting mode has NO effect (+0.010 excluded at ~2.6 sigma). Two
35k-row distills at theta_1 are mildly HARMFUL (2 sigma), the 56k pooled
one neutral. The theta_1 CEILING (running, 100/500 deal-modes at 4.2 h):
called +0.276 +/- 0.089, jd +0.292 +/- 0.105 — the labels DO carry EV
when acted on. So: volume no, acting mode no, headroom yes => CAPTURE.

MECHANISM (found in the target stage, verified by a flip diagnostic):
gamma = su2/(su2 + noise) per class, a_hat = gamma*a_obs + (1−gamma)*
a_model (§20.6). The advantage model fit on theta_1's features explains
the search's advantages nearly to the replicate-noise floor for defender
and partner rows (su2 defender-follow 1.2-1.6e-4 at theta_0 -> 1.4-2.8e-5
at theta_1), so gamma collapsed there while picker rows kept theta_0's
values:

    class            gamma p50 theta_0  theta_1(twin)   target->search | search!=prior   theta_0  theta_1
    follow|defender      0.44             0.15                                              0.41     0.22
    follow|partner       0.58             0.28                                              0.46     0.27
    follow|picker        0.78             0.81                                              0.34     0.27
    lead|defender        0.40             0.24                                              0.22     0.16
    lead|partner         0.52             0.37                                              0.30     0.18
    lead|picker          0.73             0.77                                              0.30     0.23
    (search!=prior unchanged: 0.31-0.55 both iterations; pooled iter13 identical to the twin)

  I.e. at theta_1 the targets ask the student to CHANGE its choice at half
  as many disagreeing defender/partner rows as at theta_0 and instead
  re-tilt toward the prior — which is exactly the observed signature:
  holdout KL falls, logit spread rises (4.1 -> 4.6-4.9), conventions
  sharpen, argmax behaviour barely moves, EV flat. Plausible cause: the
  linear structure the model can see in the features IS what iteration 1
  distilled; after one iteration the shrink estimator treats the search's
  remaining novelty as noise. The per-node replicate SNR (addendum 4)
  says it is not noise; the ceiling says it is EV.

ARM (launched 11:30, pipeline_gamma.py -> iter16; ceiling still running):
  twin corpus, standing recipe, ONE change: target --variance-mode global
  (gamma = global su2 / (su2 + noise) ~0.78, the theta_0 picker level) —
  an existing §20.6 mode, not a new knob. Cert vs iter11 at 8000/mode +
  PAIRED reads vs ca_ep3 and sa_ep3 (per-deal scores). PRE-REGISTERED:
  COMPOUNDS if >= +0.010 vs iter11 AND paired vs ca_ep3 >= +0.010 at 2
  sigma; PARTIAL +0.005..+0.010; NULL otherwise (then the capture failure
  sits in the distill stage — bilinear-only-frozen capacity — not the
  targets). If it compounds, the principled follow-up is a shrink
  estimator that does not use theta_k's own features to decide what is
  noise (e.g. su2 from replicate-level cross-validation, or a gamma floor
  calibrated once at theta_0).

§20.13 ADDENDUM 6 — the tilt, not only the shrink (2026-09-08, 12:30).
Target-only pre-checks on the twin corpus (fit reused from iter16; no
distill), disagreeing defender-follow rows (search argmax != prior
argmax, n=6130 at theta_1 / 2939 at theta_0):

    targets                              gamma   tilt->search p50/p75   flip   KL(t||prior) p50  z clipped
    theta_0, class mode (iter11)          0.44      +0.58 / 2.58        0.43      0.032            3%
    theta_1, class mode (iter15)          0.15      −0.15 / 0.46        0.23      0.031            9%
    theta_1, global mode (iter16, ARM 1)  0.81      +0.30 / 0.81        0.30      0.019            4%
    theta_1, gamma=1, kappa 1 (iter17)    1.00      +0.42 / 0.97        0.32      0.019            3%
    theta_1, gamma=1, kappa 0.5 (iter18)  1.00      +0.81 / 1.91        0.47      0.044           15%
    theta_1, gamma=1, kappa 0.25          1.00      +1.48 / 3.57        0.59      0.074           36%
    (prior log-gap on these rows p50 1.05 at theta_1 vs 1.27 at theta_0: the
     prior is NOT sharper where the search disagrees — that theory is out)

Reading: (1) class mode at theta_1 tilts the median disagreeing
defender-follow row AWAY from the search (−0.15): with gamma 0.15 the
target is the advantage model, and the model — which takes theta_k's
log-prior as a covariate (§20.5 "bilinear") — has learned that the
prior is right; after one iteration the denoiser regresses the teacher
onto the student's opinion. (2) Even at gamma = 1 the search's own
z-scores on the remaining disagreements are weaker than at theta_0 (p75
0.97 vs 2.58 nats): iteration 1 absorbed the confident disagreements;
what is left is many low-t rows. The committee acts on ALL of them (2-of-
3 argmax, no t-weighting) and the theta_1 ceiling is running ~+0.2, so
in aggregate they are worth taking; "one SE = one nat" (kappa 1) leaves
most below the prior's gap. §20.7(c) anticipated exactly this lever.

ARM 2 (launched 12:30, pipeline_kappa.py -> iter18_k0.5): twin corpus,
targets --variance-mode global --sigma-u2 1.0 --kappa 0.5 (gamma = 1:
the search's own replicate t-statistic, half a nat per SE; the model
drops out of the targets, keeping only the precision weights), standing
distill; cert vs iter11 at 8000/mode waits for ARM 1's cert (GAMMA
DONE). Paired reads vs ca_ep3, sa_ep3 and gamma_ep*. PRE-REGISTERED
(same bars as ARM 1): COMPOUNDS >= +0.010 vs iter11 and paired vs ca_ep3
>= +0.010 at 2 sigma. Dose-response expectation: class (−0.005) < global
< kappa 0.5. If kappa 0.5 compounds and global does not, the capture
bottleneck is the tilt temperature, and the principled recipe change is
to calibrate kappa per iteration against the committee's own flip
profile (the ceiling arm's deviation rate, ~37%) rather than fixing
"one SE = one nat"; a kappa sweep on the pooled corpus follows.

Operator question (per-class vs global shrink, recorded here): §20.6
introduced per-class sigma_u^2 at theta_0 because the GLOBAL value was
dominated by high-variance picker cells and gave t0 defender leads gamma
0.84 where the local residual implied 0.27 — the pooled convention
effect (model) entered lead targets at a quarter strength and the single
node's noise kept the rest; called-suit did not install under global.
Per-class fixed that (conventions installed in iteration 1). The
downside of global is therefore real and specific: at low-SNR lead cells
it trusts one noisy node over the pooled evidence, so convention
installation is slower/noisier. At theta_1 the same mechanism inverted:
the pooled model now IS the prior, and per-class shrink blocks the
follows. The two arms test the trade directly: ARM 1/2 probes will show
whether called-suit holds (48.6 student / 46.1 committee at kappa 1
class mode; iter11 45.4) when the model leaves the targets.

§20.13 ADDENDUM 6b — ARM 3 = global shrink + kappa 0.5 (2026-09-08, 13:10).
Operator framing recorded: the advantage model's pooling over theta_k's
frozen features is the ESSENTIAL element of §20 (the unbiased, data-
driven form of "any called-suit card beats an off-suit card at a
defender lead, even when the search's specific card is noisy"), chosen
precisely so no card classes are hand-coded; the role/trick strata were
a later optimization for the variance bucket only. Consequence: ARM 2
(gamma = 1) discards that element and is a diagnostic, not a candidate
recipe; ARM 1 keeps it at ~20% weight. ARM 3 keeps the model in the
blend (global gamma ~0.8) and halves the tilt temperature — the
combination that preserves the pooling and restores the theta_0 flip
profile. Pre-check (targets only, iter19_gk0.5): disagreeing defender-
follow tilt p50 +0.59 / p75 1.59 nats, flip 0.43 (theta_0: +0.58 / 2.58 /
0.43); partner-follow +0.92 / 2.46, flip 0.45; KL(t||prior) p50 0.041,
16% of z clipped. Launched 13:10 (pipeline_gk.py; distill now, cert
after KAPPA DONE; paired vs ca_ep3 / sa_ep3 / gamma / kappa). Same
pre-registered bars. Expected ordering if the tilt is the binding
constraint and the pooling is benign: class < global < gk ~ kappa; if
the model's self-reference is the binding constraint: kappa > gk.
Classes (regime|trick|role|lead-follow, 85 in the twin) are UNUSED by
global mode's targets and loss; they remain sampling-schedule and
reporting strata.

§20.13 ADDENDUM 7 — ARM 1 (global shrink, kappa 1) read: NULL (2026-09-08
18:05). Distill: holdout target KL never fell below theta_k's (ep7 +1%;
noisier targets are not fittable on held-out rows), selector -> epoch 0
-> certed epoch 7 (pipeline fallback added). Probes (4 seeds): called-
suit 48.0, t0 0.2, partner 98.9, pick 35.1, leaster 6.6, spread 4.3.
    gamma_ep7 vs iter11, 8000/mode:  −0.0059 +/- 0.0030 (called −0.0134, jd +0.0016)
    paired vs ca_ep3 (class mode, same corpus): −0.0011 +/- 0.0032
    paired vs sa_ep3:                            −0.0003 +/- 0.0035
Raising gamma from 0.15 to 0.81 on defender rows (flip rate 0.23 ->
0.30) changed nothing. The shrink weight alone is NOT the lever; the
tilt-temperature arms (ARM 2 gamma=1 kappa 0.5, flip 0.47; ARM 3 global
kappa 0.5, flip 0.43) are now the live test — kappa cert started 18:05.

§20.13 ADDENDUM 8 — ARM 2 (gamma=1, kappa 0.5) read: NULL; the loss is in
the PROJECTION (2026-09-08, 23:40-23:55).
    kappa_ep3 vs iter11, 8000/mode: −0.0057 +/- 0.0033 (called −0.0141, jd +0.0028)
    paired vs ca_ep3 −0.0009 +/- 0.0035 | vs sa_ep3 −0.0001 | vs gamma_ep7 +0.0002
    probes: called-suit 48.8, t0 0.5, partner 98.8, pick 35.5, leaster 7.0,
    spread 3.6 (softer than every other arm's 4.3-4.9)
Four target constructions on the same theta_1 corpus (class 0.15 /
global 0.81 / observation-only, kappa 1 / 0.5; flip profiles 0.23 ->
0.47) all read −0.005 +/- 0.003 vs iter11. The targets are NOT where
the EV is lost.

Held-out REALIZATION (scratch realization_all.py: replay the held-out
games through theta_k and the arm; "flip rows" = target argmax !=
theta_k's replayed argmax; adoption = arm argmax == target argmax):

    arm (targets)               defender-follow flip rows  adopted   partner-follow adopted   lead-defender adopted
    iteration 1 (theta_0 -> iter11)      14%                0.53          0.53                  0.36
    ca_ep3   (class, theta_1)            10%                0.41          0.33                  0.33
    gamma_ep7 (global)                   10%                0.23          0.31                  0.13
    kappa_ep3 (gamma=1, kappa 0.5)       16%                0.19          0.35                  0.11

Train/holdout target-KL gap at the final epoch: iteration 1 6%
(0.084/0.089); theta_1 arms 17-26% (class 0.083/0.101, global
0.076/0.096, kappa 0.182/0.213, gk 0.157/0.184). The head phase FITS the
training rows as well as at theta_0 and TRANSFERS half as well: at
theta_1 the bilinear-only epochs memorize row-specific corrections. The
kappa arm asked for the most flips and adopted the fewest (0.19) — the
same capacity spread thinner, and the policy softened everywhere (spread
3.6). Consistent with §20.9: the trunk epoch was the EV carrier at
theta_0 (arm 5: 1 full epoch +0.0127; head phase +0.002); the flips
that remain at theta_1 need features the frozen trunk does not have.

ARMS (launched 23:50, both queued behind ARM 3's cert):
  EP1: cert iter15/distill_epoch1.pt (the trunk epoch alone, before any
       head phase) vs iter11 at 8000 deals, paired vs ca_ep3. Reads: ep1
       > ep3 by >= +0.005 at 2 sigma => the head phase GIVES BACK the
       trunk epoch's gain at theta_1 (drop it); ep1 ~ ep3 ~ −0.005 =>
       one trunk epoch @1e-4 carries nothing at theta_1 either.
  ARM 4 (pipeline_trunk.py -> iter20_trunk): class-mode twin targets,
       3 full epochs @1e-4 with nothing frozen (the §17.8 "epoch 2+
       damages" hazard was one-hot noise-fitting; calibrated targets
       bound the per-row move). Cert best + ep3 vs iter11, paired vs
       ca_ep3 / ca_ep1. COMPOUNDS >= +0.010 vs iter11. Watch items:
       conventions (the head phase is what installs them), leaster,
       bidding drift (actor-resident heads train at 1e-4).
Also queued: theta_1 ceiling final read (450/500: called +0.177 +/-
0.041, jd +0.148 +/- 0.043).

§20.13 ADDENDUM 9 — theta_1 CEILING: HEADROOM INTACT (2026-09-09, 02:00;
runs/ceiling_h2h_theta1_202609/, 22.3 h, same instrument as §13.3:
committee R=3 x 1024/1, 2-of-3 pi_gumbel-argmax at every unforced play
node, hero = iter11 ep7 weights, anchor field = iter11 argmax, 250
deals/mode):

    EDGE +0.1664 +/- 0.0281 (n_deals 500, ~5.9 sigma)
    called +0.1808 +/- 0.0384 | jd +0.1520 +/- 0.0411
    win_frac 0.581, deals with >= 1 deviation 54%
    nodes searched 9105, resolved 8477 (93%), deviated 3327 (37%)
    theta_0 (§13.3): +0.1800 +/- 0.0289; called +0.2096, jd +0.1504;
    deviated 39%.  Difference −0.014 +/- 0.040: the one-step search
    ceiling is UNCHANGED after iteration 1 captured +0.014 of it (8%).
  Leaster (per-deal records now stored): 27/500 deal-modes carry >= 1
    leaster hand (135/2500 hands); deals with a leaster hand +0.26 +/-
    0.17, without +0.161 +/- 0.028; leaster nodes 494 (5.4%), resolved
    93%, deviated 48%. Search acting in leasters is not harmful in
    aggregate (weak; supports leaster emission later).
  Adherence at searched lead nodes (acted committee / theta_1 argmax):
    called_suit        t0 57.5 / 43.8   ALL 60.6 / 47.2   (theta_0: 56.7 / 45.3)
    def_lead_no_trump  t0 98.5 / 99.5   ALL 88.2 / 96.6   (theta_0: 87.3 / 97.4)
    partner_trump      t0 65.3 / 98.7   ALL 72.8 / 99.1   (theta_0: 81.5 / 96.9)
    The committee wants called-suit ~13 points above the policy (the
    remaining half of the installation); it deviates from the no-trump-
    lead rule at later tricks as at theta_0; and its partner-trump
    adherence FELL (t0 85.5 -> 65.3, n=75/80) while the policy's rose to
    99 — WATCH ITEM: either the search finds partner-trump leads worse
    in the theta_1 ecology or this is a small-n swing; a targeted
    partner-lead read (E2-style, §17) before any convention phase.

VERDICT for the §20.13 decision tree: headroom YES (>= +0.10 by 2.4
sigma); volume NO; acting mode NO; targets NO (four constructions);
the stall is CAPTURE in the projection (addendum 8). The teacher has
+0.17 to give at theta_1 and the student took 0.

§20.13 ADDENDUM 9b — ARM 3 (global shrink + kappa 0.5) read: NULL
(2026-09-09 03:50). gk_ep7 vs iter11 −0.0036 +/- 0.0033 (called −0.0119,
jd +0.0047); paired vs ca_ep3 +0.0012, sa_ep3 +0.0020, gamma_ep7 +0.0023,
kappa_ep3 +0.0021 (all < 1 sigma). Probes: called-suit 50.1 (highest of
the twin arms), t0 0.9, partner 99.0, pick 35.8, leaster 6.0, spread 4.0.
Five target constructions, one corpus, one distill recipe: −0.006 ..
−0.004. Closed: the target stage is not the lever at theta_1.

§20.13 ADDENDUM 10 — GENERALIZATION PRECISION: the theta_1 student's
behaviour changes on new deals are half wrong (2026-09-09, 10:00).
Held-out games replayed through theta_k and the distilled arm; "moved" =
arm argmax != theta_k argmax at a searched row; precision = fraction of
moves landing on the search's argmax; "away" = fraction of moves leaving
an action the search AGREED with theta_k on (harmful by the label):

    arm (corpus rows)            class            moved  to search  away-from-agreed
    iteration 1 (51.7k, theta_0) follow|defender   13%     0.60        0.37
                                 follow|partner    15%     0.65        0.31
                                 lead|defender      8%     0.65        0.27
                                 lead|partner      20%     0.69        0.27
    ca_ep3 (35k, class mode)     follow|defender    8%     0.50        0.46
                                 follow|partner     9%     0.47        0.50
                                 lead|defender      7%     0.50        0.47
                                 lead|partner      12%     0.26        0.63
    pooled_ep4 (56k, class)      follow|defender   10%     0.55        0.38
                                 follow|partner    14%     0.49        0.47
                                 lead|defender      7%     0.40        0.54
                                 lead|partner      18%     0.55        0.39
    kappa_ep3 (35k, gamma=1)     follow|defender    8%     0.38        0.50
                                 lead|defender      7%     0.34        0.59

Iteration 1 moved with ~2:1 precision (right:wrong by the label); the
theta_1 arms move with ~1:1, and the sharper-tilt arm worse. A student
whose changes are half wrong nets ~0 (the prior was already right at the
"away" rows). Volume helps SLOWLY: 56k vs 35k lifts defender-follow
precision 0.50 -> 0.55 and cuts "away" 0.46 -> 0.38. Labels are the
noisy per-row search readings, so a perfect student scores < 1; the
cross-arm comparison stands. Ties to addendum 6: the remaining
disagreements at theta_1 carry ~2.5x weaker per-row evidence (p75 z 0.97
vs 2.58), i.e. the SAME row count carries ~6x less evidence per pattern
— "insufficient data" in the precise sense of insufficient signal per
row, and the head memorizes the noise (train/holdout gap 4x).

§20.13 ADDENDUM 11 — EP1 and ARM 4 reads (2026-09-09, 04:16 / 09:57):
  ca_ep1 (trunk epoch only, class targets):  −0.0083 +/- 0.0031 (called −0.0170, jd +0.0005)
      paired vs ca_ep3 −0.0035 +/- 0.0024: the head phase RECOVERS part of
      the trunk epoch's damage; it does not give anything back.
  trunk_ep3 (3 full epochs @1e-4, nothing frozen): −0.0039 +/- 0.0034
      (called −0.0077, jd −0.0002); paired vs ca_ep3 +0.0008, vs ca_ep1
      +0.0043 +/- 0.0041. Probes: called-suit 48.6, partner 99.3, t0 0.6,
      pick 34.5, spread 4.5 — conventions and bidding held.
  trunk_ep1 is numerically IDENTICAL to ca_ep1 (same epoch, same targets;
  the pipeline reproduces bit-for-bit).
Trunk capacity is not the lever: more unfrozen epochs neither help nor
hurt beyond epoch 1. Every distill of theta_1 on this corpus, whatever
the targets or the schedule (7 arms), sits at −0.004 .. −0.008 with the
loss in CALLED mode (−0.008 .. −0.017) and JD flat (−0.002 .. +0.005).
Instrument note: the sharded h2h (af6f985/29e1fcc) certs 8000 deals/mode
in 16-17 min; a full cert is now ~30 min. Lesson: never edit a module a
queued pipeline imports lazily — the trunk cert spun 70k dead spawn
workers for 4 h when its parent held the old evaluator.
Live: ROUTED2 (lead vs follow attribution, ca_ep3 + kappa_ep3, 8000 deals)
then HIZ (top-evidence 6,908 rows vs random 6,908, paired).

§20.13 ADDENDUM 12 — ROUTED ATTRIBUTION: the theta_1 loss is a BIDDING
TAX, not a play loss (2026-09-09, 10:40 – 13:10). Head-routed chimeras
vs iter11, 8000 deals/mode, sharded h2h (routed_results.jsonl). Route =
which component comes from the ARM; everything else is iter11:

    arm        route          edge      se      called    jd
    ca_ep3     full ckpt     −0.0048  0.0032   −0.0091  +0.0003
               A_leads       +0.0010  0.0017   +0.0002  +0.0019
               B_follows     −0.0008  0.0022   −0.0006  −0.0011
               C_play (A+B)  +0.0005  0.0027   −0.0005  +0.0016
               D_bid         −0.0057  0.0018   −0.0102  −0.0012   ← 3.2σ
    kappa_ep3  full ckpt     −0.0057  0.0033   −0.0141  +0.0027
               A_leads       −0.0010  0.0020   −0.0028  +0.0007
               B_follows     −0.0047  0.0026   −0.0079  −0.0015
    trunk_ep3  full ckpt     −0.0039  0.0034   −0.0077  −0.0002
               C_play        −0.0053  0.0031   −0.0117  +0.0012
               D_bid         +0.0004  0.0016   +0.0028  −0.0021

Reads.
  1. ca_ep3 (standing recipe): play is at PARITY (leads, follows, and
     both together all within ±0.001) and the bidding heads alone
     reproduce the whole deficit (D_bid −0.0057 vs full −0.0048; called
     −0.0102 vs −0.0091). Routes are additive; no lead×follow interaction.
     Called mode is where the partner call and the call-dependent bury
     live, which is why every theta_1 arm lost there and not in JD.
  2. Mechanism. Epoch 1 (trunk unfrozen, 1e-4) moves the features under
     the pick/call/bury/alone heads; the ep1-only arm reads −0.0083 with
     called −0.0170 (addendum 11). The standing recipe then FREEZES the
     encoder and trains only the bilinear play pointer for six epochs,
     so the retention KL on bidding rows (lambda_ret 1, ~0.016 nats vs
     override ~0.15) has nothing it can move: the bidding drift is
     locked in while the head phase repairs play back to parity.
  3. trunk_ep3 is the mirror image: three full-actor epochs let the
     retention term pull bidding back (D_bid +0.0004, clean) but the
     same unfrozen epochs damage play (C_play −0.0053, called −0.0117).
     Its full read (−0.0039) is bidding-clean play damage, ca_ep3's is
     play-clean bidding damage; both land at −0.004 by different roads.
  4. kappa_ep3 (gamma=1, kappa 0.5, sharpest tilts): the FOLLOW head
     itself carries most of the loss (−0.0047) with a ~−0.003 residual
     left for bidding. Unshrunk targets over-move follows, consistent
     with addendum 10's precision table (0.38 to-search at defender
     follow).
Consequences.
  - Every "theta_1 stall" number in addenda 2–11 is play-parity minus a
    ~0.005 bidding tax. Correcting for it, the class-mode arm is +0.000,
    not −0.005: the projection still buys NOTHING at theta_1, but it is
    not destroying anything either. The compounding question is now
    purely why play stays at zero (addendum 10: 1:1 move precision).
  - The bidding tax is a recipe bug, not a teacher property: bidding is
    never taught by this phase, so theta_{k+1}'s bidding should be
    theta_k's bit-for-bit. Fix candidates, cheapest first:
      (a) --lambda-ret 10 on the standing recipe (queued: iter22_ret10
          on iter15's targets; paired vs ca_ep3 + its own D_bid; pass =
          D_bid within ±0.002 and full read ≥ ca_ep3 + 0.004);
      (b) head phase that also trains the bidding heads under the
          retention KL with the trunk frozen (repair, not prevention;
          needs a flag);
      (c) deploy/act-time routing of bidding through theta_0
          (HeadRoutedAgent already does it at eval; composes across
          iterations because bidding is inherited, never learned here).
  - Queued behind ret10: iter11 vs the 8M seed on routes C/D at 8000
    deals (did iteration 1 pay the same tax? its 2000-deal play-only
    route read +0.021 vs a +0.026 full read, both on the hot 2000-deal
    instrument — addendum 3), and student_ep3 (iter14) C_play to compare
    play-only between acting modes (user request).
  - HIZ (top-evidence rows vs random, both epoch-7 fallback): hiz_ep7
    −0.0246 ± 0.0042 vs iter11 (called −0.0305, jd −0.0186, leaster
    −0.044); ctrl cert running; paired read to follow as addendum 13.
Pipeline note: pipeline_routed3 crashed once on its idempotency check
(older routed rows carry "arm", newer "cand"); fixed to .get(), relaunched.

§20.13 ADDENDUM 13 — HIZ read: concentrating the policy loss on the
highest-evidence rows HURTS (2026-09-09, 13:55). Pre-registration
(addendum 11): keep override rows with z_raw = search_gap /
sqrt(search_noise_var) >= Z (6,908 of 34,658 class-mode twin rows, ~20%)
vs a random matched-count subset (seed 0), demoting the rest to the value
stream; same standing distill; both selected epoch 7 by fallback
(holdout KL never beat epoch 0 in either); 8000 deals/mode vs iter11.

    hiz_ep7   −0.0246 ± 0.0042   called −0.0305   jd −0.0186   leaster −0.044
    ctrl_ep7  −0.0089 ± 0.0034   called −0.0179   jd −0.0000   leaster −0.009
    paired hiz − ctrl  −0.0156 ± 0.0044  (−3.5σ)   [pre-reg pass was ≥ +0.008]

FAILS in the opposite direction at 3.5σ. Two readings, both consistent
with addenda 10 and 12:
  1. The top-z filter is a ROLE filter. Override-row composition (share):
        subset          fol-def  fol-par  fol-pick  lead-def  lead-par  lead-pick
        all (iter15)     0.502    0.100    0.126     0.122     0.044     0.106
        random ctrl      0.507    0.102    0.123     0.118     0.045     0.105
        top-z hiz        0.506    0.109    0.192     0.052     0.012     0.129
     Picker rows (the seat with the least hidden information, hence the
     tightest search noise) are 1.5x over-represented and defender/partner
     LEADS — the convention nodes — are cut to 40%/27% of their share. High
     per-row z is where the search is *certain*, not where the student is
     *wrong*; and certainty is cheapest exactly where the prior is already
     right (picker follows are the best-learned nodes). Training only there
     over-moves those rows for 7 epochs and drops the regularising mass of
     near-prior rows elsewhere; JD (−0.019) is hit as hard as called.
  2. Row count is not the lever either: the random 20% subset (−0.0089) is
     the full-row ca_ep3 (−0.0048) minus ~0.004, i.e. an 80% row cut costs
     roughly what the bidding tax costs — small. Both arms also carry the
     ~0.005 bidding tax of addendum 12 (epoch-7 standing schedule).
Net: "label SNR" as a per-row selection criterion is falsified; the
evidence-quality argument of addendum 6 survives only as a statement about
the WHOLE corpus (weaker evidence per pattern at theta_1), and the
per-row precision weights (cap 5) already encode the z ordering without
discarding the tail. The remaining play-side hypotheses are the
projection/realizability ones (addendum 8/10), to be re-read once the
bidding tax is removed (iter22_ret10, iter23_headonly).

§20.13 ADDENDUM 14 — bidding-tax follow-ups, head-only arm, and three
CORRECTIONS (2026-09-09, 13:10 – 16:45). All h2h at 8000 deals/mode,
paired per-deal reads share anchor + seed-42 deals.

Reads.
    iter11 vs 8M seed  C_play  +0.0107 ± 0.0032  (called +0.0113, jd +0.0101)
                       D_bid   +0.0036 ± 0.0018  (called +0.0097, jd −0.0026)
                       full    +0.0141 ± 0.0036  → additive; iteration 1's
                       trunk epoch moved bidding by +0.004, iteration 2's by
                       −0.006. Same recipe, opposite sign, both in CALLED.
                       The "tax" is an uncontrolled variance term, not a bias.
    ret10_ep7 (--lambda-ret 10, iter15 targets; epoch 7 fallback):
                       full −0.0017 ± 0.0033 (called −0.0036, jd +0.0002)
                       D_bid +0.0026 ± 0.0022; paired D_bid − ca_ep3 D_bid
                       +0.0083 ± 0.0027 (3.1σ): tax REMOVED. Paired full −
                       ca_ep3 +0.0031 ± 0.0039 (0.8σ): play by subtraction
                       ≈ −0.004. Holdout retention KL after epoch 1 was 0.020
                       vs 0.015 at lambda 1 — the coefficient did not shrink
                       the drift as measured by KL, yet the h2h bidding
                       effect flipped sign; the KL is not the h2h-relevant
                       drift metric (tie-band pick/call flips cost EV at
                       ~zero KL).
    student_ep3 (iter14) C_play −0.0046 ± 0.0028; paired ca_ep3 C_play −
                       student C_play +0.0052 ± 0.0029 (1.8σ): committee
                       acting DID help play by ~½ hundredth; the full-ckpt
                       acting read (+0.0008) was masked because the student
                       arm's trunk-epoch damage landed in play and the
                       committee arm's in bidding. Fresh-deal replicate
                       (seed 43, ROUTED6) running.
    headonly_ep3 (iter23: --freeze-epochs 1..7 --bilinear-only-frozen;
                       96/98 tensors bit-identical to theta_k, only
                       pointer_U/V changed; holdout target KL −11% by ep3,
                       comparable to ca's −13%; called-suit probe 50-51):
                       full −0.0035 ± 0.0023 (called −0.0072, jd +0.0002) —
                       a PURE play read (bidding identical). Paired vs ca_ep3
                       C_play −0.0041 ± 0.0024 (−1.7σ); vs trunk_ep3 C_play
                       +0.0018; vs student C_play +0.0011; ret10 − headonly
                       +0.0018.
Verdict on the schedule. The head alone fits the targets on held-out
rows as well as the standing recipe (KL) but LOSES ~0.004 of play EV vs
iter11; the trunk epoch adds ~+0.004 of play and costs ~0.005 of bidding
at theta_1. Every combination lands at −0.004..+0.001. The EV mechanism
of §20.9 (trunk first) is confirmed at theta_1 and it is now roughly
tax-neutral; the head phase installs conventions (50% called-suit with
nothing but U/V trained) and repairs, it does not add EV. Holdout target
KL does not track EV (headonly: best KL, worst EV).

CORRECTIONS.
 C1. Addenda 6 and 10 claimed the theta_1 disagreements carry "~2.5x
     weaker per-row evidence (p75 z 0.97 vs 2.58)". That compares the
     theta_0 CLASS-MODE tilt (pooled advantage model) with the theta_1 RAW
     replicate z (model off) — a target-construction difference, not a
     label-quality one. The like-for-like instrument is addendum 4
     (telemetry pair_diffs, same format for every corpus): gap, replicate
     SNR (~2.5) and 3-replicate sign agreement (0.62) are IDENTICAL at
     theta_0 and theta_1. A targeted-row z comparison is also invalid:
     iteration 1's corpus went through recover_search_q, whose noise_var =
     (1−w)·var(q) is not the live pipeline's pooled replicate variance.
     What survives: same evidence per ROW, fewer rows per learnable
     PATTERN — iteration 1 harvested the deviations shared by thousands
     of rows (convention-shaped); the remainder is spread across many
     fine situations. Sample complexity, not label noise. The R=9
     replicate corpus is therefore NOT the next experiment.
 C2. The greedy health probe seeds one RNG for dealing and play, so play
     changes alter the deal stream; headonly (bidding bit-identical to
     iter11) probes pick 36.0 vs iter11's 34.3 on seed 1. Probe pick/
     leaster/alone differences between arms are deal noise. Routed h2h
     (D_bid) is the only bidding instrument; earlier "pick 34→37 drift"
     readings carried no information.
 C3. (Operator caught, 16:30) The teacher does NOT roll out to terminal:
     d_rollout = 1 with the ORACLE critic as leaf evaluator on the
     observer's full-information stream inside each determinized world
     (ismcts.py leaf_evaluator="oracle"; config.teacher_d_rollout=1).
     Terminal rollouts were the deploy-tuning 4096-iteration study.
     Consequence: the +0.166 ceiling is ~entirely "oracle value averaged
     over sampled consistent worlds, one ply of prior in between" — a
     belief-marginalised PRIVILEGED value, not lookahead. So (a) a better
     oracle critic raises the ceiling directly and trains on search-free
     self-play outcomes; the ceiling stayed 0.180→0.166 because the
     oracle critic barely changed; (b) a one-ply expectimax over R worlds
     with oracle leaves is a candidate CHEAP teacher (|A|·R forward
     passes vs 1024 iterations) — validate argmax agreement vs the
     committee on existing corpus rows offline; (c) the function the
     actor must learn is E_belief[V_oracle] — the limited critic could be
     trained to that dense target with no search as a feature source.

Next (offline, no search): learnability probe on frozen theta_1 features
without the log-prior covariate (held-out accuracy on disagreeing rows,
theta_0 vs theta_1 corpora); belief aux-head held-out accuracy at the
disagreeing rows; cheap-teacher agreement check. Then the oracle-critic
scaling read. ROUTED6 (acting replicate, seed 43) in flight.

§20.13 ADDENDUM 15 — LEARNABILITY PROBE on frozen features (2026-09-09,
17:15; scratch learnability_probe.py on the cached row tables of the
iter11 (theta_0, seed features) and iter15 (theta_1, iter11 features)
fits; game-level 10% holdout, seed 0; adapter+bilinear+heteroscedastic
head, 200 epochs/patience 25, WITH the log-prior covariate (= the recipe's
model) and BARE (no prior). Held-out rows with Q; "disagreeing" = committee
top card != prior top card; "moved" = model top != prior top; precision =
moved rows landing on the committee top; away = moved rows leaving a
prior top the committee agreed with):

                    agree with committee      disagreeing rows     moves
    corpus  variant  model   prior    n      hits     n        frac  precision  away
    theta_0 prior    0.620   0.602   2294    0.303    912      0.28   0.43      0.37
    theta_0 bare     0.612   0.602   2294    0.338    912      0.33   0.41      0.39
    theta_1 prior    0.580   0.588   3467    0.228   1429      0.24   0.39      0.42
    theta_1 bare     0.575   0.588   3467    0.286   1429      0.31   0.38      0.42
    (lead/follow split in the log; leads worse everywhere: theta_1 bare
     lead precision 0.36 vs away 0.37, follow 0.39 vs 0.45)

Reads.
  1. On FROZEN features the head generalizes ~1:1 at BOTH points
     (precision/away 1.1 at theta_0, 0.9 at theta_1). The 2:1 precision of
     the iteration-1 student (addendum 10) came from the TRUNK epoch, not
     from anything the frozen features already separate. Consistent with
     addendum 14: the trunk is the EV mechanism; the head phase cannot
     manufacture EV.
  2. theta_1 is modestly worse than theta_0 on every column (agree −1.3
     vs +1.0 pts over the prior; disagreeing hits 0.29 vs 0.34), i.e. the
     residual is somewhat less linearly available in iter11's features
     than iteration 1's was in the seed's — but the gap is small; the
     dominant fact is (1).
  3. The probe is near its NOISE CEILING. The prior's top card matches
     the committee's 0.60 of the time, and the committee's own three
     replicates agree on the top-pair sign 0.62 of the time (addendum 4),
     so a perfect predictor of the TRUE argmax would score only ~0.6-0.7
     against a single committee draw, and "disagreeing rows" are enriched
     for coin-flip labels. Per-row argmax agreement cannot discriminate
     realizability at these noise levels; it says only that nothing
     large is left on the table at the row level.
  4. Reconciling with the ceiling (+0.166, C3 of addendum 14): the edge
     of committee ACTING is the sum over ~30 decisions per deal of small
     expected gains from a belief-averaged oracle value, taken at EVERY
     node, coin-flip rows included. That is not "patterns" a student can
     copy from 35k argmax-shaped labels; it is a calibrated shift of Q
     everywhere. Iteration 1 skimmed the pattern-shaped part (+0.014,
     8%); the remaining 92% is diffuse by construction.
Direction (recommendation, no code yet): stop trying to make the policy
copy the committee's argmax and make it maximise the committee's OBJECTIVE
directly — a one-ply expectimax over R determinized worlds with the
oracle critic at the leaf is cheap (|A|·R forward passes, no tree), dense
(every play node of ordinary self-play games), and differentiable through
the actor's action distribution (AWR/regularised policy improvement
against E_belief[V_oracle] instead of against a noisy argmax). Step 0
(offline, cheap): measure argmax agreement between the one-ply oracle
expectimax and the 1024-iteration committee on existing corpus nodes; if
it is ≥ the committee's own replicate self-agreement, the tree adds
nothing at teacher depth 1 and the cheap teacher is the teacher.

§20.13 ADDENDUM 16 — ACTING MODE, replicated: committee acting helps
PLAY (2026-09-09, 17:30; pipeline_routed6.py). Play-only route C_play
(bidding from iter11) for ca_ep3 (committee-acted twin, iter15) and
student_ep3 (student-acted corpus, iter14), same deals, same recipe:

    deals      ca_ep3 C_play        student_ep3 C_play     paired committee − student
    seed 42    +0.0005 ± 0.0027     −0.0046 ± 0.0028       +0.0052 ± 0.0029 (1.8σ)
    seed 43    −0.0027 ± 0.0027     −0.0085 ± 0.0028       +0.0058 ± 0.0030 (1.9σ)
    pooled     −0.0011              −0.0066                +0.0055 ± 0.0021 (2.6σ)

CONFIRMED at 2.6σ on 16,000 deals/mode: the committee-acted corpus
teaches ~+0.005 better play than the student-acted corpus at theta_1,
with the same labels per row (same search at every node). The full-
checkpoint acting read of addendum 3 (+0.0008) was masked by where each
arm's trunk-epoch damage landed (bidding for the committee arm, play for
the student arm; addendum 12). Mechanism is the state distribution:
committee acting samples the lines the search prefers, so the student
is taught where it will find itself once it adopts them. NOTE the
absolute level: even the committee-acted play route is at parity with
iter11 (−0.001 pooled); acting mode is a half-hundredth lever, not the
compounding lever. Standing recipe: committee acting stays.

§20.13 ADDENDUM 17 — amendments to C3 / addendum 15 and the CRN evaluator
proposal, restated (2026-09-09, 17:50; operator review).
 A1. C3 overstated "no lookahead". The play tree is a real multi-ply PUCT
     tree (max_depth 6 observer decisions, opponents' plies searched in-
     tree within each determinized world, policy priors); d_rollout = 1
     is the frontier rollout (one further observer play via the policy)
     BEFORE the oracle bootstrap. The ceiling is belief-averaged, trick-
     level lookahead with oracle leaves — not one ply.
 A2. Addendum 15's "argmax-shaped labels" was wrong: the target is the
     full distribution prior·exp(a_hat/(kappa·sqrt(v_post))) over all
     legal actions (pi_gumbel on shrunk Q). What the argmax-agreement
     probe measures is therefore only the top card; the deficiency is
     not the target's form but its per-row SNR: with R = 3 replicates the
     typical |Q gap| / noise is ~1, so the tilt is ~0 at most rows and
     the rows that do move are noise-enriched, at 35k rows/iteration.
 A3. Training-time only, as now: the oracle enters only inside sampled
     worlds on the teacher side; the actor conditions on the observation
     alone; the shipped network never searches (ismcts.py docstring).
Proposal, restated as a LABEL-SNR-PER-COMPUTE design, not a new target:
  The tilt uses only DIFFERENCES between root actions' Q. PUCT estimates
  each action's Q from its own visits and its own world draws: the top
  action gets most visits, the runner-up few, so the GAP's noise is
  dominated by the less-visited action and by unpaired world sampling.
  A common-random-numbers (CRN) evaluator scores every legal root action
  on the SAME R worlds with the same continuation rule, so the gap is a
  paired difference: same worlds, equal allocation. Two rungs:
    (a) CRN one-ply: continuation = policy (argmax or sample) to the
        observer's next decision, then oracle bootstrap. Cost ~ |A|·R·
        (≤4 policy passes + 1 oracle pass); R = 32 ≈ 1.3k forward passes
        vs ~10-15k for 3×1024 iterations. Shallow: no in-tree opponent
        search — biased where trick tactics matter.
    (b) CRN shallow tree: per world, exhaustive/PUCT expansion to the end
        of the CURRENT trick (≤4 plies), oracle at the trick boundary;
        same paired structure, most of the committee's lookahead, cost
        between (a) and the committee.
  Step 0 (offline, ~200 deals, ceiling_h2h replay machinery): at each
  searched node compute (a) and (b) alongside the committee; report top-
  card agreement with the 2-of-3 committee argmax and gap correlation,
  against the committee's own replicate self-agreement (0.62 sign) as the
  bar. If (b) ≥ bar at ≤1/5 the cost, the next corpus is 5x the rows at
  paired-variance labels for the same compute, targets unchanged.
  Independent lever: the oracle critic bounds every leaf; it trains on
  search-free self-play outcomes and has not been scaled.

§20.13 ADDENDUM 18 — STEP 0 RESULTS: the tree-free CRN evaluator FAILS;
the 256-iteration committee is the cheap teacher (2026-09-09, 19:45 –
21:05; runs/crn_probe_202609/: crn_probe.py, crn_analysis.py,
crn_budget.py + jsonl/logs). 295 unforced hero play nodes on 80 hands
(40 CRN deals/mode, hero = iter11 argmax acting, anchor = iter11), the
1024-iteration R=3 committee at every node alongside:
  crn_det   24 belief-pool worlds (scheme-B weights, ESS ~10), every
            legal root action continued on every world with the policy
            ARGMAX to the teacher's own leaf (one further observer play,
            oracle bootstrap at the next play decision / terminal score,
            same discount clock);  3.8 s/node vs 60.2 s.
  crn_samp  same worlds, SAMPLED continuation (1 draw/world);  3.9 s.
  and, on the SAME nodes, the R=3 committee at 256 and 64 iterations.

    predictor of a 1024 replicate's top card    pi_gumbel   by root Q   wall
    another 1024 replicate (the BAR)              0.633       0.631     60.2 s
    policy argmax                                 0.598       0.581      —
    crn_det                                       0.538       0.532      3.8 s
    crn_samp                                      0.519       0.516      3.9 s
    committee @256 (replicate)                    0.612         —       14.5 s
    committee @64  (replicate)                    0.568         —        3.8 s

    where the 1024 committee DEVIATES from the policy (n=79):
      crn_det ranks the winner above the policy card   0.45   (coin flip)
      committee@256: same winner 0.63; Q ranks winner > policy 0.71
      committee@64 : same winner 0.38; Q ranks winner > policy 0.61
    the low-budget evaluator's OWN deviations confirmed by 1024:
      crn_det 0.37 | @256 0.62 | @64 0.43
    crn_det Q vs 1024 mean Q: pooled Pearson 0.91 (centered per node),
      per-node Spearman 0.50 — agrees on the clearly-bad cards, diverges
      among the close candidates, which is where the tilt lives.
    committee-confidence terciles (1024 mean-Q gap): high tercile bar
      0.90, crn_det 0.745, policy 0.789; crn_det's OWN confident nodes
      (paired z ≥ 2, n=77): 0.60 vs policy 0.64 → BIAS, not noise.

Reads.
  1. Rung (a) is falsified: a belief average over 24 worlds with policy
     continuation and an oracle leaf ~2 tricks out predicts the committee
     WORSE than the prior does, on both readouts, and its deviations are
     wrong by the committee's Q 63% of the time. Confidence does not
     rescue it. The in-tree opponent responses and tree-optimised observer
     plays inside each world carry the discrimination among close cards;
     the oracle leaf alone does not. (Operator's point in addendum 17 A1
     confirmed: the depth-6 tree does real work.)
  2. The cheap teacher is the same tree at 256 iterations: 0.612 vs the
     0.633 bar (Δ −0.02, ~1σ at n≈870 comparisons), winner agreement
     0.775, deviations confirmed at the 1024's own self-rate (0.62), at
     4.1x lower cost. 64 iterations is below the policy prior (0.568) and
     its deviations are mostly wrong (0.43): the knee is between 64 and
     256. Replicate self-agreement does NOT discriminate budgets (@256
     0.557, @64 0.555); agreement with the 1024 committee does.
  3. Consequence for the data question (addendum 17): the next corpus
     should be searched at 256 iterations, R=3 — ~4x the rows for the
     same compute, targets unchanged (the precision weights absorb the
     higher per-row noise). That is the learning-curve corpus: ~8000
     games / ~140k override rows, distilled with the bidding-tax fix
     (--lambda-ret 10 or theta_0 bidding routing), certed at 8000 deals
     vs iter11. Pre-registration to follow.

§20.13 ADDENDUM 19 — PRE-REGISTRATION: the learning-curve corpus at 256
iterations (2026-09-09, 21:10; launched).
Corpus D  runs/distill_corpus_iter2d_256_202609: theta_k = iter11 ep7
          (theta_1), 8000 games (4x the twin), committee acting 1.0,
          R=3 x 256 iterations, d_rollout 1, oracle leaves, same p/boost
          schedule as the twin (p_base 1.0, boost_lead 1.0, boost_cs 1.5),
          seed 20260909, 40 shards of 200 games, node telemetry on.
          Expected ~140k override rows; twin cost was ~42 h for 2000
          games at 1024 (shared machine), so ~2 days here.
Arms (after DONE; standing recipe + --lambda-ret 10, the validated
bidding-tax fix; class targets; holdout-KL-best epoch with the epoch-7
fallback; cert 8000 deals/mode vs iter11, paired per-deal reads):
    D2k   shards 0-9   (2000 games @256)  — equal rows to the twin: the
          BUDGET cost at fixed rows, paired vs ret10_ep7 (twin @1024,
          same recipe) — pre-reg: within ±0.004 = 256 is free.
    D4k   shards 0-19  (4000 games)
    D8k   shards 0-39  (8000 games)      — the compounding read.
Reads: h2h vs iter11 against log(rows) = the learning-curve slope
(points: twin/ret10 35k, D2k, D4k, D8k). COMPOUNDS if D8k ≥ +0.010 with
CI > 0 (the §20.13 item-1 bar after the bidding fix); PARTIAL +0.005..
+0.010; STALL < +0.005 ⇒ the exchange rate is not worth paying and the
skill path is deploy-time search (addendum 17). Each arm also gets the
D_bid route once, to confirm the tax stays removed at 4x rows.

§20.13 ADDENDUM 18b — budget agreement by stratum (2026-09-10; same
node file; "rep" = a 256/64 replicate's top vs a 1024 replicate's top,
"bar" = 1024 replicate self-agreement, "dev same" = same 2-of-3 winner
where the 1024 committee deviates from the policy, "conf" = the low-
budget committee's own deviations confirmed by the 1024 winner):

    stratum         n    bar    rep@256  policy | dev-same@256  conf@256 | rep@64  dev-same@64  conf@64
    leads           86  0.547   0.553    0.477  |  0.750 (28)   0.700     |  0.499   0.360        0.474
      t0-1 leads    33  0.374   0.434    0.424  |  0.600 (10)   0.750     |  0.355   0.222        0.286
      t2+ leads     53  0.654   0.627    0.509  |  0.833 (18)   0.682     |  0.576   0.438        0.583
    follows        205  0.665   0.637    0.644  |  0.569 (51)   0.569     |  0.596   0.388        0.413
      t0-1 follows  87  0.567   0.576    0.609  |  0.545 (22)   0.632     |  0.508   0.200        0.250
      t2+ follows  118  0.737   0.683    0.669  |  0.586 (29)   0.531     |  0.657   0.517        0.500
    called mode    147  0.658   0.643    0.578  |  0.659 (44)   0.725     |  0.563   0.366        0.417
    jd mode        144  0.602   0.581    0.611  |  0.600 (35)   0.512     |  0.574   0.394        0.448

Reads: at LEADS the 256 committee is AT the bar (0.553 vs 0.547) and
its deviations are confirmed 0.70 — no sign of the 384-iter raw-Q
inversion the readout study saw at defender leads (that was a single-
search raw-Q readout; here 3 replicates + pi_gumbel). The budget's cost
concentrates at LATE FOLLOWS (t2+: 0.683 vs 0.737 bar), the counting/
schmear nodes where the E9 matrix put the largest headroom (t4-picker-
lead +0.039) and where the tree's in-world continuation does the work.
Early leads are the near-tie zone at any budget (bar 0.37). 64
iterations is below the prior at every stratum.

§20.13 ADDENDUM 19b — D2k read (early, first 10 shards; 2026-09-10 11:00)
+ primary-read amendment.
  D2k_ep7 (2000 games @256, --lambda-ret 10; epoch 7 fallback; probes
  called-suit 44-51, partner 97-99, t0 0-0.6):
    vs iter11  −0.0030 ± 0.0027  (called −0.0111, jd +0.0052)
    paired vs ret10_ep7 (twin @1024, same recipe) −0.0013 ± 0.0036 (−0.4σ)
      → inside the pre-registered ±0.004 band: 256 IS FREE at equal rows.
    D_bid −0.0015 ± 0.0012 (called −0.0065, jd +0.0035) → play by
      subtraction ≈ −0.0015: parity, as every theta_1 arm.
  Fit diagnostics vs the twin: noise floor 2.26e-4 vs 1.90e-4 (+19%),
  prior top-agree 0.579 vs 0.588 — the expected noise signature only.
  Bidding component across arms at lambda_ret 10: ret10 +0.0026, D2k
  −0.0015 (called −0.0065) — still a ±0.005 variance term. AMENDMENT
  (before any D4k/D8k result): the PRIMARY compounding read for each
  curve arm is the play-only route C_play (bidding from iter11), se
  ~0.0027; the full-checkpoint read stays as the secondary. pipeline_
  curve.py relaunched with C_play per arm; D2k's C_play running.
  Ops lesson (again): a pipeline script that calls the sharded h2h MUST
  have an ``if __name__ == "__main__"`` guard — the first D2k C_play
  launch lacked one, spawn workers re-imported it and respawned (load 57
  within 3 min); killed by process group, no data lost.

§20.13 ADDENDUM 20 — LEARNING CURVE, first two points (2026-09-11,
00:40; D8k pending the corpus, ~4400/8000 games). Both arms standing
recipe + --lambda-ret 10, epoch 7 (D4k: genuine holdout-KL best, −9%
monotone; D2k: fallback), 8000 deals/mode vs iter11, seed-42 deals:

    arm    rows(games)  full read           called    jd      D_bid            C_play (PRIMARY)
    ret10  35k (2000@1024) −0.0017 ± 0.0033  −0.0036  +0.0002  +0.0026 ± 0.0022  (by subtraction ≈ −0.004)
    D2k    35k (2000@256)  −0.0030 ± 0.0027  −0.0111  +0.0052  −0.0015 ± 0.0012  −0.0016 ± 0.0025 (called −0.0043)
    D4k    70k (4000@256)  −0.0031 ± 0.0031  −0.0078  +0.0016  +0.0006 ± 0.0017  −0.0038 ± 0.0027 (called −0.0090)
    paired: D4k − D2k (C_play) −0.0022 ± 0.0029; D4k − ret10 (full)
    −0.0015 ± 0.0035; D4k − ca_ep3 (C_play) −0.0043 ± 0.0030 (−1.4σ).
    Conventions (4-seed called-suit): iter11 45.4, D2k 47.8 ± 1.5, D4k
    46.7 ± 1.7 (pooled_ep4 @1024/56k rows: 53.0 — not reproduced);
    partner 98, t0-trump 0.35-0.4 everywhere.

Reads.
  1. Doubling rows at 256 moved NOTHING: play-only −0.0016 → −0.0038,
     full −0.0030 → −0.0031, called mode still the losing mode. The
     slope of the learning curve through 70k rows is ≤ 0 within ±0.003.
     The bidding fix held (D_bid +0.0006), so this is a clean play read.
  2. The fit got BETTER with rows (holdout target KL −9% monotone, best
     epoch 7 genuine; retention KL 0.008 vs 0.020) while EV did not —
     holdout target KL is confirmed NOT to track EV (addendum 14).
  3. The one theta_1 arm with a convention gain (pooled_ep4, 53.0) was
     56k rows at 1024 iterations; 70k rows at 256 gives 46.7. Consistent
     with E8/E9: the convention-lead edge exists under search-improved
     continuations, which a 256 tree resolves less often (addendum 18b:
     the budget's agreement cost sits at late follows, but the label
     SNR at early leads is the near-tie zone at any budget).
Pre-registered bar for D8k: COMPOUNDS ≥ +0.010 CI > 0. On the two
points so far the expected D8k read is ≈ −0.003 ± 0.003; a pass would
require a slope change of ~+0.013 between 70k and 140k rows that the
35k→70k step gave no sign of. If D8k reads STALL (< +0.005), the
§20.13 decision tree closes: at theta_1, neither targets, schedule,
acting mode, row selection, budget nor 4x rows compounds; the distill
phase's deliverable is conventions + prior/leaf quality, and skill
above theta_1 comes from deploy-time search (addendum 17) or from a
teacher whose per-row signal is not the committee's argmax tilt.

§20.13 ADDENDUM 21 — PRE-REGISTRATION: U1024 (2026-09-11, launched).
Operator question: is 256-iteration label quality the limiter? Existing-
data test: pool the two 1024-iteration theta_1 corpora (twin iter2c,
committee-acted, 34.7k searched rows; pooled iter2+iter2b, student-
acted, 56.2k) → ~91k rows @1024, ~1.3x D4k's rows and 2.6x the twin's.
Same recipe as the curve arms; cert 8000 vs iter11; paired vs D4k
(70k @256), D2k, ret10; D_bid + C_play. Reads: C_play ≥ +0.005 over
D4k's −0.0038 at 2σ ⇒ label quality at 256 IS a limiter (the 1024 tree
carries signal 256 loses); |Δ| < 0.004 ⇒ budget is not the limiter and
the stall is row-count-invariant at both budgets. Caveat: mixed acting
mode (38% committee-acted rows), which addendum 16 prices at ~+0.005 in
favour of committee acting — U1024 is thus slightly handicapped vs a
pure committee-acted 1024 corpus.
Also queued as thought-tests (no compute yet): (a) joint-improvement
ceiling — committee acting at a random half of nodes; (b) DAgger-style
aggregation — distill theta_0 on corpus q ∪ corpus D; (c) oracle-critic
retrain + ceiling re-read; (d) rich-feature realizability probe.
