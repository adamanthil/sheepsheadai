# Search Readout Comparison: counts vs top@Q vs RM vs Gumbel (July 2026)

**Status (2026-07-22 evening): COMPLETE — both cells landed; VERDICT: ADOPT
`pi_gumbel` as the contingency-lane distill target, with node-selective
application (partner-lead targeting or |Q-gap| gating) as a REQUIRED
constraint carried by ALL Q-based readouts. RM rejected (dominated). See
§Results / §Verdict.**
Parallel research for the Learning_System_Redesign contingency lane: it prices
the *distillation readout* the search teacher would use IF the batch+λ SNR arm
passes B0/B1 but fails B2 (role decoupling does not pin). Nothing here touches
the running arm (`runs/league_snr_batchlam`). Companion to
[Convention_Erosion_202607.md](Convention_Erosion_202607.md) (whose fix-stack
step 4 names "targeted τ≈0.5 top@Q distillation" — this study asks whether
that recipe's three patches should be replaced by one mechanism) and to
[Exit_Arms_202606.md](Exit_Arms_202606.md) (the June floor/prior-domination
audits this design is built on).

## Problem

The ISMCTS teacher's visit-count target `pi' ∝ N^(1/τ)` entangles two things
the June audits showed must be separated:

1. **Forced-exploration floor.** FPU + `root_explore_frac=0.25` + τ=1.0 puts
   ~8.5pp of target mass on actions the search's own Q calls bad (measured at
   96 iters; ~10pp at 384 — the floor is structural, iteration-independent).
   Distilling it *injects* the C1 trump-lead leak. τ=0.5 removes ~85% of the
   floor (1.2pp residual) and slowed the Arm-B′ leak ~7x without stopping it;
   τ=0.25 is near-argmax and re-concentrates onto the trump lead at the nodes
   where Q is noisy-wrong.
2. **Prior domination.** At low explore frac the visit counts echo the network
   prior (the ISMCTS-audit lesson) — fatal for re-ignition at eroded nodes
   where the prior is the problem (partner trump-lead mass 0.004 at the 2M
   checkpoint). The June patch is frac=1.0 + top@Q, which destroys the root
   prior for exploration and commits to a brittle hard target.

So the tentatively-sighted teacher recipe (τ=0.5, top@Q, frac=1.0) is three
patches for two pathologies. Two literature-backed alternatives construct the
target so exploration never enters it:

- **`pi_rm` — regret-matching root** (RM+ selection at the root only, PUCT
  interior; target = linearly-weighted average RM strategy). Exploration
  visits update Q but never the target. With fixed opponent models this is
  bandit-grade only (converges toward best response, not equilibrium; SM-MCTS
  convergence results do NOT transfer) — registered as an engineering choice.
  Lineage: Shafiei et al. 2009 (UCT over-determinism), Lisý/Kovařík/Lanctot/
  Bošanský 2013 (SM-MCTS w/ RM), Heinrich & Silver 2015 (Smooth UCT), Lisý
  et al. 2015 (OOS as the sound ISMCTS alternative). Deep literature pass
  deferred unless RM is adopted (operator decision 2026-07-22).
- **`pi_gumbel` — completed-Q readout** (Grill et al. 2020; Danihelka et al.
  2022): `pi_gumbel ∝ exp(log P_raw + (c_visit + max N)·c_scale·q̂)` with q̂
  min-max normalized over root actions and unvisited actions completed with
  the visit-weighted mean Q. Target mass is a function of the (unmixed) prior
  and estimated Q only; counts enter through the sharpness scale. Carries the
  Gumbel-MuZero policy-improvement pedigree (the property an ExIt teacher must
  provide). This adaptation is readout-only (PUCT search unchanged); full
  Gumbel root (top-k + sequential halving) is a follow-up if the readout wins.

Both inherit Q-estimation error identically to top@Q — ESS gates and
rollout-to-terminal stay load-bearing regardless of winner.

## Implementation (2026-07-22)

`sheepshead/ismcts.py`, flag-gated, default path bit-identical (15/15
`test_ismcts_exit_regression.py` pass):

- `ISMCTSConfig.root_selection = "puct" | "rm"`, `rm_gamma=0.10` (uniform mix
  in the RM *sampling* policy only), `gumbel_c_visit=50`, `gumbel_c_scale=0.1`
  (mctx defaults; min-max q̂ normalization per this codebase's convention).
- Every search now also returns `pi_gumbel` (always), `pi_rm` (rm mode only),
  and `root_prior` (mean UNMIXED root prior — `root_explore_frac` never touches
  it).
- RM updates: once per completed simulation from the root mean-Q table
  (min-max normalized), RM+ clipping, linear (CFR+-style) strategy averaging.

Instrument: `sheepshead/analysis/search_readout_comparison.py`. Node classes
from the Convention_Erosion scan (CP-eligible secret-partner leads AGREE /
DISAGREE + DEFENDER-MIRROR), same seeds (0–799), deterministic self-play scan
of the probed checkpoint. Per node:

- **Three search arms, CRN-paired** (same determinization RNG seed → identical
  belief pools; smoke run confirmed identical ESS across arms):
  A `puct/frac=0.25` (training default) → readouts `counts_t10`, `counts_t05`,
  `topq`, `gumbel`; B `puct/frac=1.0` (June audit recipe) → `topq_f100`,
  `counts_t05_f100`; C `rm/frac=0.25` → `rm`. All τ-sharpening applied offline
  to the same counts. 384 iters (June leak-audit budget), rollout-to-terminal
  (`d_rollout = 6 − trick`), ESS floor 4 (all three arms must pass).
- **Per-action paired value oracle**: every legal lead card gets R=30
  true-deal MC policy rollouts from the identical node snapshot (shared torch
  seed). The value of ANY readout π is then `V(π) = Σ_a π(a)·v(a)` (mixture)
  plus `v(argmax π)` (mode) at zero extra rollout cost. Deltas are paired vs
  the prior-argmax baseline per node.
- Baselines: `prior` (masked policy over lead cards), `prior_argmax`.

2×2-style cells: {2M league boundary (eroded; re-ignition class),
400k warmstart (conventions intact; sanity class)} × {partner nodes,
defender-mirror nodes}.

## Pre-registered predictions & decision table

Recorded before the full runs (only the 2-node smoke, whose numbers are
noise-grade, had been seen).

**P1 — Floor test** (defender-mirror `trumpMass`; June reference points:
τ=1.0 ≈ 0.085–0.10, τ=0.5 ≈ 0.012, prior ≈ 0.005):
`gumbel` and `rm` ≤ `counts_t05`; `gumbel` ≈ prior. A candidate whose
defender-mirror trump mass exceeds `counts_t05` at 2σ FAILS the floor test.

**P2 — Re-ignition test** (2M DISAGREE nodes, where prior trump mass ≈ 0.004
and the convention's rollout-grounded value is +0.24): `gumbel`, `rm`, `topq`
put substantial mass on the trump lead (modeTrumpRate materially above the
prior's ~0) at frac=0.25 — i.e., de-domination WITHOUT the frac=1.0 hammer.
`counts_*` at frac=0.25 echo the prior. Caveat accepted in advance: the June
flip-rate audit showed root Q-gaps near zero at policy-selected leak nodes;
if Q cannot see the +0.24 here either, ALL Q-based readouts fail P2 together
— that outcome indicts search Q-resolution at this budget, not a readout, and
the contingency lane needs a Q fix (more iters / grounded fields / oracle
leaves) before any readout choice matters.

**P3 — Value endpoint (primary)**: at partner nodes, paired
`Δv = V(readout) − v(prior_argmax)`, mixture and mode. Adoption bar: a
candidate must (a) match or beat `counts_t05` (the tentative recipe's soft
target) AND `topq_f100` (its hard target) on partner-node Δv, and (b) show
defender-mirror Δv not below 0 at 2σ (no control harm — the analog of the
targeted-search zero-harm gate).

**P4 — Near-tie behavior** (secondary, qualitative): `gumbel`/`rm` entropy
increases with small |root Q-gap|; sharpened counts do not track the gap.

| Outcome | Reading / action |
|---|---|
| `gumbel` passes P1–P3 | Adopt completed-Q readout as the contingency-lane target; recipe becomes "pi_gumbel, frac default, ESS gates" (τ and frac stop being load-bearing). Follow-up: full Gumbel root (seq. halving) priced separately. |
| `rm` passes, `gumbel` doesn't | Adopt RM root; trigger the deep literature pass (operator pre-commitment) before any training use. |
| Both pass, comparable | Prefer `gumbel` (stronger theoretical fit for an ExIt teacher: policy-improvement guarantee; readout-only = smaller code surface). |
| Only `topq_f100` passes | June recipe stands as-is; the patches were load-bearing for a reason. Record why the principled targets failed. |
| Nothing passes P2 (incl. topq) | Q-resolution is the binding constraint at this budget; readout question moot until search Q improves. Escalation: iters ladder / oracle-leaf variant, new pre-registration. |
| Falsifier-style anomaly (any readout with defender-mirror trump mass >> prior AND positive Δv there at 2σ) | Measurement suspect (rollout field can't price the leak) — stop, don't interpret; cf. the June "search certifies leaks" mechanism. |

Known biases, accepted at this rung: rollout values are the probed policy's
own continuations (rung-1 hindsight grade; same instrument across readouts and
CRN-paired, so *differential* readings are robust); fixed opponent models
(no readout is scored on balance/exploitability); one lineage, two
checkpoints; ESS gate selects against low-ESS nodes for all arms equally.

## Run log

Outputs under `runs/search_readout_202607/`. Budget: ~70s/node incl. all
three arms (measured); ~140 nodes @2M + ~200 @400k ⇒ ~7h total, niced,
alongside the live SNR arm (precedent: erosion rung-1 pipeline).

| Step | Command sketch | Status |
|---|---|---|
| smoke (2 nodes, tiny budget) | `search_readout_comparison --num-seeds 20 --iters 48 --rollouts 4` | DONE 2026-07-22 (mechanics + CRN verified; numbers discarded) |
| readout @ 2M | `search_readout_comparison --model .../pfsp_perceiver-shared-v2_checkpoint_2000000.pt --num-seeds 800 --rollouts 30 --iters 384 --out .../readout_2000k.json` | DONE 2026-07-22 15:41, exit 0 (n = 1 AGREE / 79 DISAGREE / 58 mirror; 3 skips) |
| readout @ 400k | same, `--model .../warmstart_perceiver-shared-v2_400k.pt` | DONE 2026-07-22 18:23, exit 0 (n = 80 / 58 / 60; 0 skips) |

## Results (2026-07-22)

Headline table — the two decision cells (Δv = paired readout − prior-argmax,
leader score; mode@T = mode-is-trump rate):

**2M DISAGREE (re-ignition class, n=79; prior trump mass 0.024):**

| readout | Δv(mix) | Δv(mode) | trumpMass | mode@T | H |
|---|---|---|---|---|---|
| counts_t05 (f.25) | +0.051 ± 0.024 | +0.057 | 0.046 | 0.01 | 0.65 |
| counts_t10 (f.25) | +0.087 ± 0.031 | +0.057 | 0.154 | 0.01 | 1.15 |
| counts_t05_f100 | +0.208 ± 0.057 | +0.284 | 0.495 | 0.72 | 1.42 |
| topq (f.25) | +0.330 ± 0.076 | +0.330 | 0.747 | 0.75 | 0 |
| topq_f100 | +0.269 ± 0.074 | +0.269 | 0.722 | 0.72 | 0 |
| **gumbel (f.25)** | **+0.296 ± 0.070** | **+0.306 ± 0.073** | 0.594 | 0.57 | 0.17 |
| rm (f.25) | +0.275 ± 0.077 | +0.275 | 0.637 | 0.67 | 0.45 |

**2M DEFENDER-MIRROR (floor/control, n=58; prior trump mass 0.021):**

| readout | Δv(mix) | trumpMass | mass at Q-gap≤0 nodes |
|---|---|---|---|
| counts_t05 (f.25) | −0.020 ± 0.019 | 0.028 | — |
| counts_t10 (f.25) | −0.054 ± 0.026 | 0.127 | **0.113 (floor)** |
| counts_t05_f100 | **−0.134 ± 0.043 (3.1σ)** | 0.397 | — |
| topq (f.25) | −0.105 ± 0.059 (1.8σ) | 0.345 | 0.000 |
| topq_f100 | −0.059 ± 0.060 | 0.259 | 0.000 |
| **gumbel (f.25)** | −0.050 ± 0.048 (1.0σ) | 0.210 | **0.000** |
| rm (f.25) | −0.086 ± 0.046 (1.9σ) | 0.225 | **0.166 (residual)** |

Findings:

1. **P2 re-ignition: PASSED by every Q-based readout at frac=0.25.** From
   prior trump mass 0.024, gumbel/rm/topq put 0.57–0.75 of mode weight on the
   convention lead and recover Δv ≈ +0.27..+0.33 — the full forced-trump
   value (erosion reference +0.239 ± 0.059) — WITHOUT the frac=1.0 hammer.
   `counts_*` at frac=0.25 echo the prior exactly as the June audit predicted
   (trump mass 0.046/0.154, Δv +0.05/+0.09). De-domination is a property of
   Q-based readouts, not of frac.
2. **P1 floor: the decomposition is the result.** Splitting defender-mirror
   trump mass by the sign of the node's root Q-gap: `gumbel` carries **0.000**
   mass at nodes where search Q says fail is better (0.004 at 400k) — the
   forced-exploration floor is *structurally eliminated*, as designed.
   `counts_t10` carries 0.113–0.157 there (the floor, reproducing June's
   ~10pp). `rm` carries 0.166 — a residual average-strategy floor (early
   uniform RM iterates weight into `strat_sum`), i.e. RM is WORSE than gumbel
   on the exact property it was proposed for. All remaining Q-based mirror
   mass sits at the 25–34% of defender nodes where search Q itself inverts
   (trump ≥ fail) — the June best-Q-is-trump inversion (30–44%) reproduced.
   That is a Q-RESOLUTION constraint shared by every Q-based readout
   including the incumbent top@Q recipe; it selects WHERE the teacher may be
   applied, not which readout.
3. **P3 value + control harm: gumbel is the best-behaved candidate.** It
   matches/beats the incumbent recipe endpoints at the decision cell (+0.296
   vs topq_f100 +0.269, vs counts_t05 +0.051) and is the only Q-based
   readout comfortably inside the no-control-harm gate (−0.050 ± 0.048,
   1.0σ; topq 1.8σ, rm 1.9σ borderline; **counts_t05_f100 FAILS at 3.1σ** —
   the June-recipe soft target is measurably harmful at control nodes).
4. **P4 near-tie: confirmed for gumbel/rm, absent for counts.** 400k:
   gumbel H 0.401 at |Q-gap|<0.01 vs 0.185 at ≥0.03; rm 0.540 vs 0.442;
   counts_t05 flat (1.21 vs 1.24). Evidence-scaled sharpness works.
5. **400k AGREE sanity (n=80): search does not improve an intact convention
   policy — mild cost.** All readouts Δv ≤ 0 vs the adhering argmax (gumbel
   −0.073 ± 0.045; topq −0.069 ± 0.048); Q inversions flip ~40% of AGREE
   modes to fail leads. Consistent with erosion's optimal-adherence ≈ 1.0.
   Consequence: the teacher should target eroded/DISAGREE-like nodes, not
   all partner leads on an intact policy (moot at 2M where adherence is
   0.007, i.e. everything is DISAGREE).
6. 400k DISAGREE replicates the 2M ordering at smaller magnitude (gumbel
   +0.206 ± 0.063, topq_f100 +0.234 ± 0.067, counts_t05 +0.080); 400k
   mirror Δv all null within noise.

## Verdict (2026-07-22)

**Decision-table row: "gumbel passes P1–P3" — ADOPT `pi_gumbel`** as the
contingency-lane distill target, replacing τ=0.5 + top@Q + frac=1.0 (recipe
becomes: `pi_gumbel`, frac default, ESS gates unchanged; τ and frac are no
longer load-bearing). P1 is read via the pre-registered decomposition: floor
mass 0.000; the Q-inversion mass is the anticipated "Q-resolution binding"
phenomenon and is carried equally by the incumbent recipe.

**Constraint attached to adoption (from findings 2+5):** at 384-iter budgets
the teacher must be node-selective — partner-lead/eroded-node targeting as
already sighted in the contingency plan, optionally tightened by a per-node
|root Q-gap| confidence gate. Indiscriminate all-lead-node application would
re-inject ~0.2 trump mass at the 25–34% of defender nodes where Q inverts.
This also further disconfirms cheap search-everywhere on the play head
(2026-07-22 discussion): the Q-inversion rate, not the readout, is the
binding constraint.

**RM: REJECTED** — dominated on all three axes (value +0.275 vs +0.296,
control harm 1.9σ vs 1.0σ, residual floor 0.166 vs 0.000). Per the operator
pre-commitment, the deep RM literature pass is therefore NOT triggered.

Next steps: (1) amend the Learning_System_Redesign contingency recipe to
`pi_gumbel` + node targeting (text amendment; code already emits it);
(2) OPTIONAL iters-ladder (384 → 96 → 48) on the 2M DISAGREE cell to price
cheap-budget gumbel for the bidding heads / broader-application question;
(3) if the teacher lane ever fires, the 50–100k fine-tune → stratified-EV →
decay-curve loop from Convention_Erosion remains the validation harness.
