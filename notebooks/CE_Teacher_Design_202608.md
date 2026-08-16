# CE Search Teacher — Design & Implementation Plan (2026-08)

Status: DESIGN APPROVED PENDING §13.3 ceiling result (operator, 2026-08-16).
Successor to the resolved-pair hinge teacher (Search_Teacher_Design §12,
attempts 5a–10, all retired). This document is the implementation contract
for the always-on cross-entropy teacher and the accompanying cleanup: after
this lands, the ONLY search trainer in the codebase is this design, and the
ONLY entropy controller is the v2 signed controller. Removed code lives in
git history (final pre-removal commit will be tagged `pre-ce-teacher`).

Evidence base, in one paragraph: policy-space hinge teaching proved
transient against PG — gains and damage both squeezed out (§12.22); the
per-decision search committee itself carries real EV (§13.3 ceiling h2h,
interim +0.13 ± 0.04 at n=205, final result to be recorded below) and
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
