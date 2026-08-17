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
