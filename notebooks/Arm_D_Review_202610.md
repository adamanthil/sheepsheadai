# Review of the proposed arm D (fold-spread model-error term) — October 2026

Status: **DRAFT FOR COMMENT (2026-10-10), uncommitted.** A review of
CE_Teacher_Design §22.6 (`7be5f42`, "Live iteration 1 and the recipe
arms") and of its proposed next step, written for the agent running the
iteration-1 arm orchestrations. Requested: comment on each objection
(§2, §3), verify or refute the readings in §3, and update §4/§5 with the
decision.

Proposed next step under review (§22.6 "Reading"):

> The remaining gap points at the one term still missing from the
> posterior variance, the advantage model's own estimation error.
> Cross-fitting supplies an estimate for free, the spread of the fold
> models' predictions at each row, and adding it would make C less
> confident at exactly the pooled rows where it is over-sharp.

Proposed form: `v_post += (1 − gamma)^2 · Var_folds(a_model)`, with an
explicit stated sigma_u^2 minimum beside it.

---

## 1. What §22.6 gets right (no change requested)

- The shrinkage-prior arithmetic: with the local estimate
  E[r^2] − E[noise] negative and floored, the class sigma_u^2 is
  k·global/(n+k) with k = 50 — 3.45e-5 from the 1,177 held-out rows
  (baseline floor) vs 3.25e-6 from the 11,412 out-of-fold rows (arm A).
  The floor was an implicit prior set by the holdout fraction, not a
  measurement.
- Root cause: the centered noise model shrank every card's replicate
  variance two-thirds of the way (nu = 4 vs 2 df) toward ONE corpus-wide
  mean (1.459e-3); quiet classes' noise overstated up to ~25x, picker
  classes understated. Class pooling (`4ff2846`, `--noise-pool class`)
  fixes it; held-out and out-of-fold sigma_u^2 then agree within 1.5x —
  good evidence the estimator is now identified.
- B dropped (2% distill holdout buys nothing); 10% stays.
- Class-pooled noise model is correct and should merge.

## 2. Objection 1 — under cross-fitting the term is already in gamma

Finding #2 (v_post omits the model's estimation error, the Prasad-Rao
g2 analogue) was stated for the IN-SAMPLE fit. It does not carry over to
arm C. With out-of-fold predictions:

```
r = y − a_oof = (A − a_oof) + e
E[r^2] − sigma_n^2 = E[(A − a_oof)^2] = model misfit u + estimation error
```

The out-of-fold sigma_u^2 is the model's TOTAL prediction-error variance.
Classic Fay-Herriot needs a separate g2 term because its sigma_u^2 is the
variance of u alone with beta estimated on top; arm C's empirical
sigma_u^2 already absorbs the estimation error. Blending two independent
estimates with those error variances gives
`v = sigma_u^2 sigma_n^2 / (sigma_u^2 + sigma_n^2)` exactly.
**Adding `(1−gamma)^2·Var_folds` double-counts at the class level.**

What fold spread could legitimately add is WITHIN-class heterogeneity
(rows where the model extrapolates): rescale each row's sigma_u^2 by its
relative fold spread with the class mean held fixed — not an additive
term. Fold spread is also a weak estimator of that:

- fold models share 75% of their training data pairwise (K = 5), so
  the spread understates single-model sampling error by roughly 4x;
- 4 of the 5 fold models trained on the row itself, re-coupling the
  spread to y_n (the leak cross-fitting removed).

**Question for review:** does any part of the arm-C pipeline compute
sigma_u^2 in-sample (e.g. a fallback path, or classes thin enough to hit
the global)? If so, those rows are the only place the additive term is
justified.

## 3. Objection 2 — C's extra sharpness is not at the pooled rows

Per-class medians from `runs/202609_recall_rc/pi/iter1/target_report.json`
(baseline) vs `pi/iter1_ablations/crossfit_classpool/target_report.json`
(arm C), ordered by contribution to the KL increase (n·ΔKL p50):

| class | n | gamma base → C | KL p50 base → C | z_max p50 base → C |
|---|---|---|---|---|
| std t3-defender-follow | 10,586 | 0.10 → **0.80** | 0.062 → 0.090 | 1.64 → 1.68 |
| alone t3-defender-follow | 3,132 | 0.23 → **0.77** | 0.069 → 0.155 | 1.29 → 2.09 |
| alone t4-defender-follow | 3,606 | 0.22 → **0.92** | 0.036 → 0.110 | 0.44 → 0.93 |
| std t4-defender-follow | 11,312 | 0.10 → **0.93** | 0.041 → 0.059 | 0.77 → 0.95 |
| std t3-defender-lead | 2,826 | 0.26 → **0.65** | 0.049 → 0.117 | 1.03 → 1.51 |
| std t0-defender-lead | 3,032 | 0.26 → 0.34 | 0.070 → 0.133 | 2.65 → 3.72 |
| std t1-defender-lead | 3,361 | 0.22 → 0.34 | 0.059 → 0.109 | 2.45 → 3.09 |
| std t0-defender-follow | 11,362 | 0.09 → 0.22 | 0.080 → **0.079** | 2.11 → 1.89 |
| std t1-defender-follow | 10,361 | 0.10 → 0.33 | 0.063 → **0.062** | 2.00 → 1.71 |

Reading:

- The rows that got sharper are mostly those that became
  **search-trusted** (gamma 0.65-0.93): late-trick follows and leads,
  alone play.
- The heavily POOLED rows (early defender follows, gamma 0.22-0.33) are
  no sharper than the baseline.
- The proposed term enters as `(1−gamma)^2` ≈ 0.01 at gamma 0.9: it
  cannot reach where C's extra KL lives. Early defender leads (gamma
  0.34, KL ~2x) are the only plausible arm-D target.

**Alternative hypothesis: search bias the replicates cannot see.** At
high-gamma rows the target is essentially `y/sigma_n` — the node's own
search in its own replicate SEs. Replicate variance measures randomness
only, not systematic error (PIMC strategy fusion, the oracle leaves'
full-information optimism, fixed theta_k models of the other seats).
Late-trick nodes are quiet and well replicated, so the honest posterior
trusts them fully and passes on any consistent bias at full confidence.
The baseline's accidental gamma ≈ 0.1 may have been protecting against
search bias, not against model overconfidence.

**Questions for review:** (a) reproduce the table; (b) is the C −
baseline gap (−0.0049 ± 0.0029) concentrated in late-trick play or early
leads — is there any read (e.g. per-trick deviating-deal split in
`h2h_vs_baseline.json` per_deal) that can localize it without a new arm?

## 4. Proposed arms instead of D (same corpus, ~4 h each)

1. **C at matched dose (run first).** Choose kappa so C's KL p50 equals
   the baseline's 0.043 (likely kappa ≈ 1.3-1.5; calibrate offline in
   the target stage before distilling). Separates the AMOUNT of movement
   from WHERE it lands.
   - C(kappa) ≥ baseline → the loss was step size; the honest allocation
     is fine; adopt C with the calibrated kappa.
   - C(kappa) < baseline → the allocation is the problem → arm 2.
2. **C + search-bias allowance.** `sigma_n,eff^2 = sigma_n^2 + b^2`,
   one global or per-class b^2: an explicit, stated knob for search error
   the replicates can't see — the search-side counterpart of the
   sigma_u^2 floor. Calibrate b^2 against something real: re-search a
   sample of late-trick nodes at a much higher budget (and/or deeper
   rollout) and measure the mean squared difference from the 256-iter
   labels beyond replicate noise.
3. Arm D only in its corrected form (§2: per-row redistribution of
   sigma_u^2 by relative fold spread, class mean fixed), and only if
   early defender leads are shown to carry the gap.

## 5. Iteration-1 relaunch

C − baseline = −0.0049 ± 0.0029 (1.7 SE), but all three arms lost
paired. Recommendation: relaunch iteration 1 on master's recipe (known
+0.0166 vs theta_0); merge class pooling as an available, non-default
flag until an arm clears the baseline.

## Comments / decisions

_(for the orchestration agent and operator)_

**Orchestration agent, 2026-10-10 07:20 — AGREED on both objections; plan
switched.** Arm D (launched 07:11 as `crossfit_classpool_folderr`, code
`3f912a8`) was killed at 07:18 in its cross-fit stage, and the commit
was dropped from the branch (operator: no theoretical justification, no
extra complexity); the branch tip is again `4ff2846` (class pooling).
The measurement that motivated it survives as a one-off file
(`pi/iter1_ablations/crossfit_classpool/fold_spread.pt`).

§2 — correct, and the question has a clean answer: under `--cross-fit`
NO row's sigma_u^2 is in-sample. The target stage takes every class
value from out-of-fold residuals (`class_residual_variances` on the OOF
evaluation, shrunk toward the OOF global 8.5e-4), classes never seen
fall back to that same OOF global, and the heteroscedastic head's per-
row variance is not used (`node_variance=False`). So r = y − a_oof and
E[r^2] − sigma_n^2 is the model's total prediction-error variance,
estimation error included; the additive term has no justified rows. The
fold-spread measurement itself (arm C's five models, `fold_spread.pt`
in the arm-C dir) is consistent with that: 3.0e-5 at std t0 defender
follows vs a class sigma_u^2 of 4.5e-5 — i.e. most of the "misfit" at
the pooled cells is already estimation error that the OOF moment has
counted once.

§3 — table reproduced exactly from the two target reports (same nine
classes, same gammas/KLs/z to the printed digit); the full ranking by
n·ΔKL (total +2,876 nats over the table) is led by std t3 (+299),
alone t3 (+270), alone t4 (+269), std t4 defender follows (+204), std
t3 defender leads (+193), then std t0/t1 defender leads (+191/+168); the
only classes LESS sharp under C are the picker follows (gamma 0.60-0.76
→ 0.17-0.69, n·ΔKL −19 to −54), where class pooling raised the noise.
The early defender follows (gamma 0.22/0.33) are not sharper at all. So
C's extra movement sits at the search-trusted rows, and the proposed
term — (1−gamma)^2 ≈ 0.005-0.12 there — could not have reached it. (b)
`h2h_vs_baseline.json` carries per-deal scores only (no trick index),
so the gap cannot be localized to late tricks without a replay; not
attempted.

The search-bias hypothesis is the right frame for the high-gamma rows:
at gamma 0.9 the target is y/sigma_n with sigma_n ≈ 2e-5-6e-5 per label,
so a 256-iteration search's systematic error (strategy fusion, oracle-
leaf optimism, theta_k as the other seats) passes through at full
confidence; the replicates cannot see it because they share it. The
baseline's accidental gamma ≈ 0.1 at those cells was damping the
search, not the model.

§4 — adopted in the stated order. Matched-dose calibration first:
target-only reruns of arm C's cached fit/OOF at kappa 1.2 / 1.4 / 1.6
(`runs/202609_recall_rc/kappa_calib.sh`), pick the kappa whose KL p50
hits the baseline's 0.0427, then one full arm (fit/target/distill/cert +
paired h2h vs baseline and vs C). Arm 2 (search-bias allowance b^2)
only if the matched-dose arm still loses — its calibration needs a
higher-budget re-search of late-trick nodes, a day of corpus-style
compute, so it is not queued yet.

§5 — not followed yet: the operator asked this morning to settle the
recipe on this iteration rather than relaunch into bidding with it
open; the matched-dose arm is ~4 h. Relaunch is the operator's call
after that read. Agreed that class pooling merges regardless; whether
as default or opt-in depends on the matched-dose result.

**10:51 — matched-dose result (arm E, kappa 1.27, KL p50 0.0424):** cert
+0.0098 ± 0.0031 vs theta_0 (play-only +0.0102 ± 0.0029); paired vs
baseline −0.0054 ± 0.0029; paired vs C +0.0003 ± 0.0016. §4.1 second
branch: the loss is allocation, not step size → §4.2 (search-bias
allowance) is the next arm, calibration pending. §5 followed: iteration
1 relaunched on master's recipe.
