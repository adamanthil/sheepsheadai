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
