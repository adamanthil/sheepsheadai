# ExIt stabilization arms (June 2026)

Decision experiment: confirm what broke in the two collapsed warm-start runs and
gather the evidence needed to choose between **(a) population-grounded teacher
rollouts** (determinize/roll out opponent seats with frozen population policies)
and **(b) pure self-play tree search** (AlphaZero-style; drop PFSP grading) for
the next structural investment.

## Background (one paragraph)

Run 1 (May): bidding owned by distillation (f_bid=1.0, hard PG-mask) →
always-PASS collapse in ~45k episodes via the self-referential-teacher ratchet.
Run 2 (June 3–7): bidding owned by terminal PPO (f_bid=0, f_play=0.30,
additive) → same collapse, slower and *flattened*: sampled PICK stayed ~32% for
586k episodes while greedy PICK was 5.7%, ALONE 56%, leaster 74%, trick-0
defender trump-lead 48.6% (baseline 4.8%), h2h −1.38 pts/game vs the 30M field.
Both ownership regimes failed ⇒ the shared conditions are causal: (1) onset
shock (critic re-fit shaped→terminal + distill yank) makes picker EV genuinely
negative, (2) nothing guards the bidding heads through that window, (3) the
teacher models all seats as pi_theta and cannot see the real (population)
opponents, (4) the training population accretes degraded checkpoints. New
guards now committed: bidding-head KL anchor, distill-coeff ramp, periodic
greedy health probe (greedy_health.csv + gate warnings).

## Falsifier: teacher trump-lead audit (runs first, gates Arm B's reading)

`validation/teacher_trump_lead_audit.py` on the PRISTINE 30M model: at trick-0
defender-lead nodes (trump+fail in hand), compare the policy prior's trump-lead
mass vs the production teacher's pi' trump mass, plus root-Q (best trump vs best
fail) to separate the exploration floor (FPU + 25% root uniform mix) from
genuine teacher preference.

- Prior trump mass is ~0.003 (healthy). If pi' mass is an order of magnitude
  higher AND root-Q favors/ties trump, the self-play-rollout teacher *endorses*
  the leak it was built to fix (rollout opponents can't punish an
  information-revealing lead) → play distillation from this teacher is expected
  to hurt, and the population-grounded rollout is the indicated fix.
- If pi' mass elevation is explained by the exploration floor alone (root-Q
  clearly favors fail leads), the teacher's *values* are fine and the run-2 leak
  regression came from the degrading-policy feedback loop instead.

Preliminary n=5: prior 0.001 → pi' 0.074 (+3.4 SE). Full n=150 running.

## Arm A — can terminal PPO + anchor hold the baseline? (null hypothesis)

- Resume pristine `pfsp_swish_checkpoint_30000000.pt`, fresh copy of the
  reference population, run `exit_armA_anchor`, episodes 30.0M → 30.05M.
- `--f-pick 0 --f-partner 0 --f-bury 0 --f-play 0` (NO search at all),
  `--anchor-coeff 1.0` anchored to the exact resume checkpoint (KL=0 at start),
  `--schedule-horizon-episodes 20000000` (end-of-schedule LR 5e-5 / entropies),
  greedy probe every 5k episodes (200 games).
- Note run 2 = this regime WITHOUT the anchor (f_play also 0.30); its greedy
  PICK was certainly gone within ~20k episodes (picker_avg −3.4 by then).

Gates at 25k and 50k (greedy_health.csv):
- greedy PICK ≥ 25% (baseline 32.9%)
- greedy ALONE ≤ 12% (baseline 6.6%)
- greedy t0 trump-lead ≤ 8% (baseline 4.8%)
- training picker_avg trending toward ≥ 0 (it will dip during critic re-fit;
  the question is recovery, not the dip)
- At 50k: h2h vs `final_pfsp_swish_ppo.pt` ≥ −0.2 pts/game (1000 games).

PASS ⇒ terminal-mode PPO is viable with the anchor; the ExIt program continues
on this base. FAIL ⇒ even anchored terminal PPO can't hold the warm start —
the critic re-fit/onset shock needs its own fix (critic-only warmup phase)
before any search question matters.

## Arm B — does the current (self-referential) play teacher help or hurt?

Arm A config PLUS `--f-play 0.30 --searched-ppo-weight 1.0` and the distill
ramp (default 50k). Run `exit_armB_distill`, launched after Arm A's 25k gate
and the falsifier read. Same gates; additionally compare against Arm A:
greedy t0 trump-lead trajectory and 50k h2h.

- B ≥ A on gates and h2h, trump-lead flat ⇒ current teacher is at least
  neutral; population-grounding is an *upgrade*, not a precondition.
- B < A, trump-lead rising (and falsifier shows teacher endorses trump leads)
  ⇒ the self-play-rollout teacher is net harmful exactly as hypothesized.

## Decision matrix (the question this experiment answers)

| Falsifier | Arm A | Arm B | Conclusion |
|---|---|---|---|
| teacher endorses leak | holds | worse than A | **Population-grounded rollouts** are the precondition for any play distillation; build them. Pure self-play search is contraindicated (it doubles down on the self-model that's failing). |
| teacher clean (floor-only) | holds | ≥ A | Teacher values fine; collapse was onset+guards. Continue hybrid, population-grounding optional upgrade. |
| teacher endorses leak | holds | ≥ A | Mixed: leak bias real but dominated by PPO grading; population-grounding still right long-term, lower urgency. |
| any | fails | — | Fix the terminal-critic warm-start first (critic-only warmup); search questions premature. |

Pure self-play tree search (the AlphaZero alternative) is only attractive if
the falsifier comes back clean AND we were willing to drop PFSP grading; a
dirty falsifier is direct evidence that self-modeled opponents corrupt the
search values in hidden-info multiplayer, which pure self-play would amplify
(real games would *also* be graded by the self-model's blind spots).

## Follow-on (2026-06-10, after the decision)

- **B′ (target_tau=0.5) RUNNING** (`runs/exit_armBprime_tau05/`): Arm B config
  with the distill target sharpened to tau=0.5 — launched BEFORE the
  population-grounding commit, so it isolates the cheap fix on the (dirty)
  self-play teacher. Prediction from the audit: trump-lead drift mostly
  suppressed (pi' floor mass 8.5% → 1.2% at tau=0.5) but Q noise remains;
  watch the greedy probes vs Arm A's flat trajectory and Arm B's explosion.
- **Population-grounded teacher IMPLEMENTED** (`15fbea7`,
  `Population_Grounded_Teacher_Plan.md`): teacher models non-observer seats
  with the agents actually controlling them this game (pool-build belief
  weights + advance + rollouts); observer stays pi_theta. 17/17 regression;
  lockstep pool build loses no batching; leaf-parallel rounds group by
  controller. Acceptance audit (n=150, 4 strongest frozen population members)
  pending below.
- **Population-grounded acceptance audit (n=150, 4 strongest frozen members):
  NULL RESULT on the pristine model.** pi' trump mass 9.1% (self-play 8.5–9.5%),
  Q-gap −0.021±0.0073 (self-play −0.025/−0.027), best-Q-trump 41% (35–36%) —
  statistically indistinguishable. In hindsight the criterion was weak: the
  pristine pi_theta IS (essentially) a member of the population it was trained
  with, so the two rollout fields are nearly the same policy and a static
  audit on the healthy model cannot separate them. Two implications: (1) the
  trick-0 target poison on the HEALTHY model is target construction (the
  exploration floor at tau=1.0), not rollout-field identity — the tau=0.5 fix
  carries that load (B′ tests it end-to-end); (2) population grounding's
  claimed value is DYNAMIC — pinning the rollout field when pi_theta degrades
  (the run-1/run-2 ratchet; Diagnostic B showed the collapsed model's teacher
  follows its own degraded policy down). The sharper falsifier (running): the
  SAME trump audit on the COLLAPSED run-2 checkpoint both ways. Prediction:
  the self-play teacher (rollout field = collapsed, trump-leading policy)
  shows a much weaker/positive Q-gap than the population-grounded teacher
  (rollout field pinned at 30M strength) — i.e. grounding restores the
  punishment signal exactly where the ratchet needs it.

## Status log

- 2026-06-10: guards implemented (anchor / ramp / greedy probe), 16/16
  regression tests pass, falsifier preliminary n=5 dirty (+3.4 SE), Arm A
  launched (`runs/exit_armA_anchor/train.log`).
- 2026-06-10 (falsifier FULL, n=150, 358 games, ESS-aborts 18): prior trump
  mass 0.0127 → pi' 0.0951 (paired +0.0824, **+26.3 SE**); root-Q gap (best
  trump − best fail) −0.0268 ± 0.0064 with best-Q-is-trump at 36% of nodes;
  **argmax(pi') leads trump 1.3% — identical to the prior's argmax**; mean root
  ESS 17.7. Reading: the teacher's visit-count MODE is correct, but the soft
  target carries ~8pp of trump mass from the exploration machinery (FPU +
  root_explore_frac=0.25 + tau_target=1.0) that its own Q-values say is wrong
  — and Q is too weak/noisy (−0.027 on a [−1,1] scale, 36% per-node
  inversions) for PUCT to starve those visits. Forward-KL distillation
  faithfully teaches the floor ⇒ the run-2 leak regression (4.8%→48.6%) is
  explained as distill-injected mass compounding through the self-model loop.
  TWO separable fixes implied: (1) sharpen/de-floor the play distill target
  (cheap: lower tau_target for the target, or subtract forced-exploration
  visits), (2) population-grounded rollouts (structural: makes the punishment
  for information-revealing leads real, so Q sharpens and visits concentrate).
  Note this also retro-explains part of run-2's bidding flattening: every
  searched decision distilled a floor-smoothed target.
- 2026-06-10 (multi-tau replication, n=150 fresh nodes): main numbers replicate
  (prior 0.005 → pi' 0.085, +28 SE; Q-gap −0.025; argmax 0% both). Target
  re-sharpening on the SAME visit counts: **tau=1.0 → 8.5% trump mass,
  tau=0.5 → 1.2%, tau=0.25 → 0.03%** (prior 0.5%). So τ=0.5 removes ~85% of
  the injected floor while staying soft; τ=0.25 is near-argmax (risky at ESS
  ~18 / 96 iters — would confidently teach a noisy argmax). Cheap-fix
  candidate: distill at tau=0.5 (keep tau=1.0 inside PUCT), i.e. a separate
  target temperature, pending the Arm B read.
- 2026-06-10 (Arm A early read, 10k eps): NO onset crash — picker_avg +1.25
  flat through 9k (collapse runs were −2..−2.5 by here); greedy PICK 32.5%,
  ALONE 10%, trump-lead ~1% at 10k. The critic re-fit alone does NOT cause the
  onset shock; the play-distill yank was the driver. Anchor restraining force
  engaged (KL grows from 0.004 at start).
- 2026-06-10 (Arm A COMPLETE, 50k eps): **ALL GATES PASSED.** Final eval
  (`runs/exit_armA_anchor/final_eval.txt`, 800 games + 1000 h2h): h2h vs
  frozen baseline **+0.061 ± 0.081 pts/game = statistical parity** (run-2
  collapsed agent: −1.38 on the same harness). Greedy PICK 31.2% / ALONE 8.6%
  / leaster 11.1%; per-position pick P1=25%→P5=50% (baseline shape).
  Post-harness-fix re-measure (see 08dca0b): trick-0 trump-lead **2.3% greedy
  / 2.1% mass vs baseline 1.4% / 1.3%** — essentially NO leak drift (the
  11.3% first reported was the double-encode harness artifact). Conclusion:
  terminal PPO + anchor fully HOLDS the warm start. picker_avg final ~+1.22.
- 2026-06-10 (Arm B STOPPED at 30k — **GATE FAILED at every probe, hypothesis
  confirmed**): with the ONLY delta from Arm A being play distillation toward
  the current teacher (f_play=0.30 additive, ramp at ~60% strength by 30k),
  greedy trick-0 trump-lead exploded monotonically 9.3% → 29.2% → 43.8% →
  38.2% → 59.1% → **74.5%** across the six probes, while bidding stayed
  healthy under the anchor (PICK 30–35%, ALONE 6–10%). Endpoint eval
  (ckpt 30030000, `runs/exit_armB_distill/final_eval.txt`): h2h **−0.375 ±
  0.103**, trump mass 51.6% (vs Arm A 2.1%, baseline 1.3%). Killed per the
  pre-registered 25k no-go; remaining 20k episodes would only have burned
  compute. NOTE: chasing the probe-vs-eval greedy-rate discrepancy here
  uncovered the eval-harness double-encode bug (fixed in 08dca0b).
- **DECISION (matrix row 1): falsifier dirty + Arm A holds + Arm B
  catastrophically worse.** Population-grounded teacher rollouts are the
  precondition for play distillation; τ≈0.5 distill-target sharpening is the
  cheap complementary mitigation to test as B′; pure self-play tree search is
  contraindicated. The anchor + ramp + greedy guard stay on for all future
  warm-start runs.
- 2026-06-10 (Arm A, 35k eps): **25k GATE PASSED.** Greedy probes 5k–35k all
  in-gate (PICK 30.1–35.5%, ALONE 5.9–11.4%, leaster 7.5–13%, trump-lead
  0–6.1%). picker_avg RISING: +1.28 → +1.41 → +1.35, above the +1.32
  warm-start baseline — terminal PPO + anchor is mildly improving picker EV,
  not just holding it. Anchor KL 0.036 → 0.11: real PG pressure against the
  bidding anchor; constant coeff caps bidding-head movement (add decay later
  if Arm A's h2h says bidding is the binding constraint).
- 2026-06-10 (RATCHET FALSIFIER, collapsed run-2 ckpt 30585000, n=100 each):
  **prediction PARTIALLY failed — grounding neutralizes the ratchet push but
  does NOT restore a strong punishment signal.** Self-play teacher: prior
  trump mass 0.408 → pi' 0.414, paired delta **+0.0053 ± 0.0018 (+3.0 SE)**
  — the teacher actively amplifies the collapsed model's leak; Q-gap
  −0.0073 ± 0.0055 (n.s.), best-Q-trump 44%, ESS 67.7, 0 aborts.
  Population-grounded teacher: paired delta **+0.0011 ± 0.0022 (+0.5 SE,
  null)**; Q-gap −0.0105 ± 0.0059, best-Q-trump 39%, argmax correction
  50%→35%; but **ESS 17.3 with 35/135 ESS-aborts** — the bidding importance
  weights correctly reject determinizations inconsistent with population
  behavior once the observer is off-population (an honest extra guard:
  aborted searches don't distill; also a throughput tax exactly when pi_theta
  drifts). The predicted Q-gap separation did NOT materialize: 96 iters of
  search cannot strongly punish trump leads even with healthy rollout
  opponents — pi' ≈ prior on a collapsed model either way. Reading:
  **grounding is a stabilizer, not a rescue.** Its measured benefit is
  removing the teacher's leak-direction push (+3.0 SE → null) on a degraded
  model; the load-bearing fixes remain prevention (anchor + ramp + greedy
  gates) and target sharpening (τ=0.5). Arm C (grounded + τ=0.5 + guards)
  stays the belt-and-braces recipe but grounding should not be sold as able
  to pull a drifted policy back.
- 2026-06-10 (B′ early probes, τ=0.5 self-play teacher + guards): 5k: PICK
  26.9% / ALONE 5.4% / leaster 16.0% / trump-lead 1.16%; 10k: PICK 35.4% /
  ALONE 8.1% / leaster 7.5% / **trump-lead 0.0%** (Arm B same points: 9.3% →
  29.2%). picker_avg ~+1.40. τ=0.5 is so far removing the distill-injected
  floor exactly as the multi-tau audit predicted. (15k: 6.52%, 25k: 5.21% —
  in-gate but hovering above Arm A's 1-2%; watch the endpoint.)
- 2026-06-10 (B′ COMPLETE, 50k eps + endpoint eval): **h2h +0.137 ± 0.077 —
  the best endpoint of any arm** (Arm A +0.061 ± 0.081; both vs the frozen
  baseline field). Bidding healthy: PICK 31.4%, ALONE 11.9% (at the gate
  edge), leaster 10.5%, per-position P1=24%→P4=46%. Trump-lead **7.6% greedy /
  7.5% mass** (baseline 1.4/1.3, Arm A 2.3/2.1): the leak drifted ~5x but
  stayed bounded — final probe trajectory 2.4/3.3/6.5/4.5 after the single
  11.1% outlier at 30k. Reading: τ=0.5 play distillation BUYS measurable
  strength (+0.08 vs Arm A's point estimate, same harness) at the cost of a
  contained leak. The leak's self-play EV cost appears small (B′ wins h2h
  DESPITE it) — but it remains a tell a human expert would exploit, which
  matters for the actual goal. If a future run wants B′'s recipe, pair it
  with the leak tracer and consider τ between 0.5 and the floor-subtraction
  variant.
- 2026-06-10 (B′ 30k probe: **GATE BREACH, trump-lead 11.11%**, n=90, ~1 SE
  over the 8% line). Trajectory 1.16 → 0.0 → 6.52 → 4.04 → 5.21 → 11.11:
  noisy upward drift. Compared at 30k: Arm B 74.5%, B′ 11.1%, Arm A ~2% ⇒
  **τ=0.5 slows the leak injection ~7x but does not stop it** — consistent
  with the multi-tau audit (τ=0.5 removes ~85% of the floor, not 100%, and
  the residual still compounds through the self-model loop). Single marginal
  breach ≠ Arm B's every-probe failure, so B′ runs to its designed 50k
  endpoint for the full trajectory + endpoint eval; but the working
  conclusion is that target sharpening alone is mitigation, not cure, which
  further motivates the exploiter-league pivot
  (`Exploiter_League_Plan_202606.md`).
- 2026-06-10 (TEACHER VALUE-ADD PROBE, run "before", 500 paired duplicate
  deals, iters_play=96, `validation/teacher_value_add_probe.py`): **positive
  but sparse.** Paired delta (search − raw) **+0.042 ± 0.020 pts/deal
  (+2.1 SE)**. Search deviated from the raw argmax at only **39/3000 (1.3%)**
  of searched PLAY decisions — but conditional on deviating, the deal gained
  **+0.54 ± 0.25 points**. ESS-aborts 2.2%. Reading: at production budget the
  teacher's argmax beats the policy exactly where they disagree (the
  improvement operator EXISTS), but the signal is sparse — 1 corrected
  decision per ~77 searched — which explains why distilling the full soft
  target at τ=1.0 drowned it in floor mass (Arm B) while τ=0.5 might let it
  through (B′). Also the first direct evidence for deploy-time search value
  (+0.04 pts/game at 6 searches/game, 96 iters). A 384-iter arm (300 deals,
  seed 7) is running to read the iteration-scaling of deviation rate and
  conditional gain. Re-run after exploiter league = the "after" arm.
- 2026-06-10 (PAIRED ENDPOINT COMPARISON, 1500 duplicate deals each, greedy,
  baseline field — same harness as the value-add probe, so directly
  comparable): **all three endpoints statistically indistinguishable.**
  B′ vs Arm A +0.013 ± 0.036; B′ vs baseline +0.018 ± 0.034; Arm A vs
  baseline +0.005 ± 0.036. B′'s h2h +0.137 (stochastic rotating-seat
  harness) does NOT survive greedy paired measurement — if B′'s true greedy
  edge were 0.137 it would have shown at ~4 SE here. Either the h2h gain was
  noise, or it lives in the stochastic sampling distribution (plausible for
  forward-KL distillation: it reshapes the soft distribution while moving
  the argmax at only ~1% of decisions) — and deployment plays greedy or
  search-wrapped, so the soft-distribution gain is not deployment strength.
  CONSEQUENCE: **no training intervention has yet moved greedy strength**
  (50k B′ distillation: +0.018 ± 0.034), while deploy search measured on the
  SAME harness gives +0.042 (96 iters) and +0.103 (384 iters) per deal —
  i.e. 384-iter decision-time search is worth ~5x B′'s entire training gain.
  The "B′ recipe as engine" hypothesis loses its evidentiary support; the
  decision matrix re-centers on gen-0d/e (league engine) and the 768 probe
  (deploy-search ceiling). Harness note for all future strength claims:
  stochastic h2h and greedy paired measurements answer different questions;
  use the paired-greedy harness for deployment-strength claims.
  scaling is real and steep.** Paired delta **+0.103 ± 0.050 pts/deal**
  (2.5x the 96-iter +0.042); deviation rate 2.4% (vs 1.3%); conditional gain
  +0.74 ± 0.34 (vs +0.54 ± 0.25); ESS-aborts collapsed to 0.2% (vs 2.2% —
  more iterations also stabilize the determinization weights). So 4x compute
  bought ~2x deviations × ~1.4x per-deviation quality. Both arms are
  independently +2.1 SE positive; jointly, decision-time search beating the
  raw policy is now well-established on this model. Deploy framing: +0.10
  pts/game at ~6 searched decisions/game, ~4-6s/search uncontended at 384 —
  viable for human-paced play, and it COMPOSES with any training-side gain
  (it's an inference-time wrapper on whatever policy ships). The curve was
  still rising at 384; a 768+ arm is the obvious next read if deploy search
  becomes the strength path.
