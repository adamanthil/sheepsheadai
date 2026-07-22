# Search Readout Comparison: counts vs top@Q vs RM vs Gumbel (July 2026)

**Status (2026-07-22): PRE-REGISTERED, runs launched (see Run log).**
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
| readout @ 2M | `search_readout_comparison --model .../pfsp_perceiver-shared-v2_checkpoint_2000000.pt --num-seeds 800 --rollouts 30 --iters 384 --out .../readout_2000k.json` | LAUNCHED 2026-07-22 |
| readout @ 400k | same, `--model .../warmstart_perceiver-shared-v2_400k.pt` | QUEUED (sequential) |

## Results

*(appended as runs land; every completed instrument gets its numbers recorded
here even if null/awkward)*
