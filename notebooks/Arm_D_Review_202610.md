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

---

## 6. Second review (2026-10-10 evening) — after the null bias calibration

Status: **FOR COMMENT.** A review of CE_Teacher §22.6 third addendum
(`214f5a6`) and its candidate arms F/G/H, with a different reading of the
C/E gap and two proposed arms. Requested: verify §6.2 (script below),
comment on §6.3-§6.4, choose in §6.5.

### 6.1 The fundamental problem

The recipe sizes each update by how SURE the search is, not by how much
the decision is WORTH. Both Stage-2 knobs scale with posterior precision
1/v_post:

- tilt `z = a_hat / (kappa sqrt v_post)` — in value units the temperature
  is `kappa sqrt v_post`, which shrinks as confidence grows;
- CE weight `omega ∝ 1/v_post` (§20.9, cap 5).

Uncertainty is already accounted for once, by the Fay-Herriot shrinkage
(gamma). Using precision again in the tilt and again in the weight counts
it twice more. Precision is highest late in the hand, where few hidden
cards remain — exactly where the least is at stake. The baseline's
corpus-pooled noise made sigma_n^2 roughly constant within a role, which
turned both knobs, by accident, into a fixed value-unit temperature and
near-uniform weights. Fixing the noise model (correctly) exposed the
allocation.

### 6.2 Evidence (iteration-1 targets, baseline vs arm C)

Per-row targets, v_post, gamma and weights read from each arm's
`targeted/` shards, aligned on `row_table.pt`; a_hat reconstructed as
`kappa sqrt(v_post) · (log t − log p)` centered over the legal set (the
~5% clipped rows are slightly understated; no effect on the picture).
"Dose" = weight × KL(target ‖ prior), as a share of the total; "stake" =
the dose-weighted top centered advantage (C's a_hat). The fixed-eta rows
tilt C's a_hat by one global eta = 0.0097, which matches the baseline's
KL p50 of 0.0427.

| recipe | t0 | t1 | t2 | t3 | t4 | picker rows | stake |
|---|---|---|---|---|---|---|---|
| baseline (z, 1/v) | .21 | .17 | .18 | .22 | .23 | .04 | .017 |
| **C (z, 1/v)** | .13 | .09 | .10 | .19 | **.50** | .04 | .017 |
| C tilt, omega = 1 | .16 | .13 | .17 | .26 | .28 | .15 | .023 |
| fixed eta, 1/v weights | .14 | .11 | .13 | .22 | .42 | .08 | .021 |
| **fixed eta, omega = 1** | .16 | .15 | .20 | .27 | .21 | .32 | .030 |

By class (medians; tau = kappa sqrt v_post in value units; top = C's top
centered advantage):

| class | gamma base → C | tau base → C | top | weight base → C | dose share base → C |
|---|---|---|---|---|---|
| std t4-defender-follow | 0.10 → 0.93 | .0057 → .0039 | .0040 | 1.90 → 2.61 | .179 → .246 |
| alone t4-defender-follow | 0.22 → 0.92 | .0085 → .0035 | .0034 | 0.84 → 3.43 | .019 → .126 |
| std t0-defender-follow | 0.09 → 0.22 | .0056 → .0059 | .0096 | 1.92 → 1.27 | .169 → .079 |
| std t1-defender-follow | 0.10 → 0.33 | .0059 → .0070 | .0106 | 1.75 → 0.89 | .129 → .044 |
| std t2-defender-follow | 0.10 → 0.58 | .0059 → .0076 | .0110 | 1.73 → 0.76 | .140 → .047 |
| std t3-defender-follow | 0.10 → 0.80 | .0058 → .0067 | .0099 | 1.79 → 0.96 | .163 → .083 |

Readings:

- **C spends half its update on trick 4**, mostly two-card endgames whose
  median top advantage is ~0.004, against ~0.011 at earlier defender
  rows. Early defender follows (t0-t2) fall from 44% of the dose to 17%.
  Even with perfect targets C would spend its distillation epochs on
  near-worthless decisions.
- **The 1/v weights are the main driver**: removing them alone takes the
  t4 share from 0.50 to 0.28; a fixed eta with the weights kept still
  leaves 0.42.
- **This is why E did nothing** (E − C +0.0003 ± 0.0016): kappa scales the
  amount of movement uniformly, not where it lands. (The §4.1 second
  branch was right that the loss is allocation; §6 identifies which.)
- **The baseline's temperature really was fixed**: tau ≈ 0.0057 at every
  std defender-follow class t0-t4; under C it ranges 0.0035-0.022.

Scripts (reviewing session's scratchpad, uncommitted; run from the repo
root with `uv run python`): `/private/tmp/claude-501/-Volumes-Nargothrond-dev-sheepsheadai/e4e1ea6c-83aa-4867-b852-dd7afcbf8811/scratchpad/temp_alloc.py <out.pt>`
(extract, ~1 min) then `temp_an.py <out.pt>` (tables).

### 6.3 Comments on `214f5a6`

- **The null calibration stands** and is a good result: the class-pooled
  noise model predicts the 256-vs-2048 disagreement on fresh data at
  every trick. The stated caveat (a bigger budget shares the structural
  errors) is right.
- **The convexity argument is correct math, wrong frame.** For a
  KL-regularized step the Bayes-optimal target given the data is
  `p · exp(a_hat / eta)`: the expected objective E[sum pi A] − eta KL is
  linear in A, so the posterior mean passes through and no convexity
  correction is needed. The label noise only bites because eta =
  kappa sqrt v_post shrinks with v, amplifying near-tie noise at quiet
  nodes — §6.1 seen from another side. (The 35% argmax-moved figure is
  consistent with near-ties at d^2 ≈ noise and is harmless under a
  value-unit tilt, which barely moves near-ties.)
- **G (probability-of-best) would make this worse**: P(a = best |
  posterior) goes one-hot as v → 0 whatever the stakes, so the t4
  two-card rows would become the sharpest targets in the corpus.
- **F (model-only)** is a useful diagnostic, not a recipe; with gamma = 0,
  v_post = class sigma_u^2 still sets tilt and weight, so it confounds
  the source with the allocation.
- **H (minimum model weight)** is the baseline's accident as a knob;
  agreed it is the weakest.

### 6.4 Proposed arms (same corpus, ~4 h each)

1. **Arm V — principled form (run first).** Honest posterior mean a_hat
   (cross-fit 5 + class pooling, unchanged); target
   `t ∝ p · exp(clip(a_hat / eta, ±tilt_max))` with ONE global eta in
   value units; omega = 1. v_post enters only through gamma. eta set
   offline to match the baseline's KL p50 (≈ 0.0097 on C's a_hat) — a
   dose calibration, not a tuning against h2h. This is the standard
   KL-regularized policy-improvement step (MPO / AWR form). Its trick
   profile is close to the baseline's, but it moves ~1/3 of the dose to
   picker rows, a regime no tested recipe has visited.
2. **Arm W — one-factor bridge.** Arm C with omega = 1, nothing else
   changed. Isolates the weights as the cause.

Code: a `--tilt-scale {posterior,global}` (+ `--eta`) and
`--weights {precision,uniform}` pair in `stage_target` /
`build_tilt_target`; defaults keep the current recipe.

### 6.5 Decision rule

- **V matches the baseline** (paired within ~1 SE or better): adopt V
  (cross-fit + class pooling + value-unit tilt + uniform weights) as the
  recipe; relaunch.
- **W matches, V loses**: the extra picker dose is what hurts. Picker
  rows carry the largest advantages (0.03-0.09, alone picker the
  highest) and are where structural search bias the replicates and the
  budget probe cannot see is most plausible (strategy fusion, oracle-
  leaf optimism). Next measurement: an estimator outside the search
  family (oracle referee / exhaustive late-trick play) on picker rows.
- **Both lose**: the "model beats label" reading regains weight; arm F
  becomes the next diagnostic.

### Comments / decisions (§6)

_(for the orchestration agent and operator)_

**Orchestration agent, 2026-10-10 evening — AGREED; arm V building.**
§6.2 reproduced exactly from the reviewing session's extraction
(`temp_an.py` on its `alloc.pt`): dose shares by trick baseline .206/
.165/.182/.221/.225 (picker .036, stake .0171) vs C .125/.088/.097/.187/
.504 (picker .038, stake .0166); C with omega = 1 → t4 .283, picker
.153; global eta with omega = 1 → .164/.151/.204/.274/.207, picker .324,
stake .0302. Agreed on §6.1: the §20.9 design counted posterior precision
in gamma, in the tilt temperature and in the weight; the baseline's
corpus-pooled noise flattened the last two by accident (tau ≈ 0.0057 at
every std defender-follow class) and the honest noise model exposed the
allocation. Agreed on §6.3: for the KL-regularized step the posterior
mean passes through linearly, so the convexity argument in §22.6's third
addendum is the same defect seen from the SE-unit tilt and needs no
separate correction; G would sharpen the quiet rows further and is
dropped; F stays a diagnostic. Built on branch `value-unit-tilt`
(worktree, from master `214f5a6`): `--tilt-scale {posterior,global}` +
`--eta`, `--weights {precision,uniform}` (`ce_weights`), defaults = the
current recipe; eta calibrated target-only on C's cache to the baseline's
KL p50 0.0427, then arm V (`valueunit_uniform`) and the one-factor bridge
W (`classpool_uniform`) run sequentially from `iter1_arms_vw.sh`, each
paired vs the baseline and vs C. §6.5 decision rule adopted as written.

---

## 7. Third review (2026-10-10 night) — does V override a confident policy, and is the advantage model calibrated?

Status: **FOR COMMENT.** Operator's question, prompted by §6: the §20
design deferred to theta_k unless the search's evidence was significant
(the SE-unit tilt). Arm V removes that guard. How often does V override a
confident policy, and on whose evidence? Operator's correction on the
framing, adopted here: overriding a confident prior is sometimes exactly
the goal — the defender fail-lead and called-suit conventions were
confident wrong priors — and the advantage model exists to find such
pockets from signal too close to the noise floor to resolve per node,
without hand-defined classes. The standard is therefore not "never
override a confident prior" but "override only on an overwhelming
preponderance of the aggregate evidence." §7.3 tests the model against
that standard; §7.5 applies it to the t0 defender trump lead. All
numbers are from arm V's actual targets and the out-of-fold predictions
in `pi/iter1_ablations/valueunit_uniform/`.

### 7.1 Where deference lives under Fay-Herriot

In the §20 design, deference came from the SE-unit tilt: a card moved
only when the search's evidence was statistically significant. Under
Fay-Herriot the fallback for an unreliable node is the advantage MODEL,
not zero: a_hat → a_model as gamma → 0. The target therefore defers to
theta_k only insofar as the model does, and it overrides theta_k only as
well as the model's overrides are supported. V gives up the SE guard and
relies on a_hat being calibrated (§7.4).

### 7.2 Search-driven overrides of a confident policy: rare, well supported

"Flip" = the target's top card is not theta_k's top card. P(wrong) = the
honest posterior's probability that the flipped-to card is the worse one,
Phi(−gain / sqrt(2 v_post)) with V's a_hat and v_post (the same judge for
all three recipes).

| theta_k top-card prob | share of rows | flip rate base / C / V | V flips: P(wrong) | search label agrees | model agrees |
|---|---|---|---|---|---|
| < 0.5 | 17% | 44% / 48% / 46% | 0.23 | 94% | 67% |
| 0.5-0.8 | 42% | 23% / 25% / 23% | 0.17 | 94% | 68% |
| 0.8-0.95 | 18% | 4.5% / 6.3% / 5.2% | 0.04 | 98% | 68% |
| >= 0.95 | 23% | 1.1% / 1.7% / 1.4% | 0.02 | 99% | 43% |

- V overrides a confident theta_k at ~1-5% of those rows, near the
  baseline's rate. Those flips need a large gain (median 0.036-0.075).
- The honest posterior puts only 2-4% of those flips as wrong. Under the
  same judge the baseline's confident flips were 7-13% wrong: with its
  noise model off, the baseline was the less deferential recipe there.
- On average V's targets sharpen theta_k's top card rather than take
  mass from it.
- At p >= 0.95 the flips are search-driven: the model agrees with only
  43% of them.
- Caveat: P(wrong) cannot see structural search error. The budget probe
  (§22.6 third addendum) found no bias that a bigger budget removes, and
  that is the limit of what has been checked.

### 7.3 Model-driven overrides: the aggregate test

Out of fold, the fold model that predicts a row never saw that row's
label, so the model's choice is independent of the label's noise. For
every row where the OOF model's top card differs from theta_k's,
compare the edge the model claims for its card with the edge the search
labels show, averaged over the group. This is the "preponderance of
evidence" test run directly.

The OOF model's top card differs from theta_k's at 45,273 of 137,765
rows (33%).

| group | rows | model's claimed edge | search's edge (z) | when theta_k confident (p >= 0.8) |
|---|---|---|---|---|
| std defender lead t0 | 929 | .0033 | +.0001 (z +0.1) | — |
| std defender lead t1 | 1,405 | .0038 | +.0003 (z +0.7) | — |
| std defender lead t3 | 1,290 | .0072 | +.0017 (z +2.7) | −.0015 (z −0.5) |
| std defender follow | 16,982 | .0084 | +.0012 (z +6.5) | −.0042 (z −6.7) |
| alone defender follow | 5,204 | .0096 | +.0047 (z +14.5) | +.0014 (z +1.1) |
| std partner follow | 4,569 | .0100 | +.0001 (z +0.2) | −.0039 (z −3.9) |
| std partner lead | 2,290 | .0083 | −.0006 (z −1.6) | −.0053 (z −5.9) |
| std picker follow | 4,655 | .0181 | −.0031 (z −3.0) | −.0131 (z −6.1) |
| std picker lead | 3,095 | .0132 | −.0018 (z −2.2) | −.0078 (z −4.8) |
| alone picker follow | 834 | .0331 | −.0106 (z −2.2) | −.0240 (z −3.4) |

By the size of the model's claimed edge (all groups):

| model's claimed edge | rows | search's edge | theta_k confident |
|---|---|---|---|
| < .002 | 11,318 | −.0023 (z −10.7) | −.0117 (z −10.5) |
| .002-.0055 | 11,318 | −.0023 (z −9.1) | −.0103 (z −10.6) |
| .0055-.013 | 11,318 | −.0008 (z −2.6) | −.0080 (z −8.7) |
| .013-.024 | 6,791 | +.0023 (z +4.4) | −.0057 (z −4.3) |
| > .024 | 4,528 | +.0144 (z +12.4) | +.0030 (z +1.2) |

Readings:

- The search backs at most about a third of the model's claimed edge,
  and only for large claimed edges (> ~0.024) or at alone-defender
  follows. Calibration slope of the search's edge on the model's,
  per group: 0.03-0.69.
- Small claimed edges point the wrong way: about half the overrides
  claim < 0.0055, and there the search says theta_k's card is better.
- When theta_k is confident, the search sides with the policy in every
  group except alone-defender follows, by 0.004-0.024. By the
  operator's standard these overrides should not happen, and the model
  makes them.
- The convention classes (t0/t1 defender leads) carry no aggregate
  signal for the model's top-card overrides. That does not contradict
  the convention story: theta_0 already leads fail ~96% of the time, so
  those overrides are mostly one fail card over another. The
  convention's own contrast (trump vs fail), which this test does not
  isolate, is strongly supported and is carried by the model (§7.5).

### 7.4 The violated assumption, and its size in V

Fay-Herriot assumes the regression part is calibrated:
E[y − a_model | a_model] = 0. sigma_u^2 covers how far the model
scatters, not a systematic lean, so neither gamma nor any variance term
can absorb it. Here the model is over-confident in its own deviations
from theta_k (§7.3).

The same defect read at theta_k's top card (theta_k's choice is
independent of the label noise, so the mean OOF residual there is a clean
bias read): the residual y − a_model is positive in 28 of 29 classes
(z up to +11), about +0.0008 at defender rows and +0.005-0.009 at picker
rows (+0.0225 alone picker). A per-class regression
y ~ b1·a_model + b2·(centered log p) gives a positive, significant b2 in
every class checked: the model under-uses theta_k's prior, whose
covariate scale was learned at ~0.005.

Average tilt this bias takes off theta_k's top card, in nats:
(1 − gamma)·resid/eta for V, (1 − gamma)·resid/(kappa sqrt v_post) for
the baseline.

| group | baseline | arm V | V after per-class linear calibration |
|---|---|---|---|
| std defender | 0.13 | 0.06 | 0.03 |
| std partner | 0.09 | 0.08 | 0.04 |
| alone defender | 0.07 | 0.05 | 0.01 |
| std picker | 0.15 | 0.39 | 0.20 |
| alone picker | 0.18 | 0.97 | 0.47 |

- At defender rows V is better than the baseline: its higher gamma
  leans less on the model.
- At picker rows V is clearly worse: the fixed temperature lets the
  larger picker residuals through, and V also moves ~1/3 of the dose
  onto picker rows (vs ~4%).
- If V loses and W matches (§6.5 second branch), this is now the leading
  suspect, ahead of structural search bias.
- Linear calibration on log p removes 60-75% of the lean but leaves
  picker rows at 0.2-0.5 nats: the bias concentrates at the top card and
  is not linear in log p.

### 7.5 Case study: who lowered the t0 defender trump lead?

Operator goal: the t0 defender trump lead as close to 0 as possible. On
this lineage theta_0 (`checkpoint_8000000`) probed 3.4-4.1%; the
iteration-1 candidates probe 1.7% (baseline), 1.9% (C), 1.7% (E). Was
the drop the search's doing or the model's?

**Method.** The target is
`theta_0 prior · exp((gamma·y + (1 − gamma)·a_model) / tau)`, a sum in log
space of a search term (gamma·y) and a model term ((1 − gamma)·a_model).
Rebuilding each recipe's target from the two terms together reproduces
the stored target to the fourth decimal, so the target can be split
exactly into "search term only" and "model term only." For the baseline
the in-sample a_model is recovered from its stored target as
(a_hat − gamma·y)/(1 − gamma); C and V use the OOF predictions. Trump
cards from `game.TRUMP_SET`; rows restricted to those with both a trump
and a fail lead legal.

**Trump-lead mass at std t0 defender leads** (2,953 rows):

| | trump mass |
|---|---|
| theta_0 prior | 5.0% (argmax trump 4.1%, consistent with the 3.4-4.1% probe) |
| baseline: search term only | 4.0% |
| baseline: model term only | 2.6% |
| **baseline target (both)** | **2.4%** |
| arm C / arm V target | 2.4% / 2.6% |

Alone t0 defender leads (896 rows): 0.8% → 0.2% (baseline), same
pattern. Std t1 defender leads: 5.8% → 5.2%, again mostly the model term.

**Reading.**

- The model term carries most of the push (gamma ≈ 0.26 at these rows,
  so ~3/4 of the target's weight is on the model).
- But the model is not inventing the signal. Prior-weighted trump-minus-
  fail edge averaged over the rows: search labels −0.0335 (z −70), OOF
  model −0.0317 (z −94); alone: −0.051 vs −0.049. The model, scoring
  rows whose labels it never saw, reproduces the search's aggregate
  verdict almost 1:1.
- This is the design intent: each node's search is too noisy to resolve
  the convention alone (gamma 0.26), the aggregate evidence is
  overwhelming, and the model pools it into every row. The search
  supplies the evidence; the model amplifies it.
- V does not weaken this convention at the target level (2.6% vs 2.4%).

**Where the residual 2.4% sits** (sums of per-row trump mass; 148 = 5.0%):

| rows | prior mass | target mass | note |
|---|---|---|---|
| 206 rows where the label's best trump ≥ best fail | 27.8 | 27.1 | OOF model disagrees at 89% of them |
| 2,747 rows where the label prefers fail | 120.2 | 43.2 | mostly theta_0 at 10-50% trump |

By theta_0's trump mass: < 2% → 8.6 to 0.5 (edge −0.039); 2-10% → 33.7
to 6.7 (−0.026); 10-50% → 89.8 to 52.6 (−0.015); > 50% (25 rows) →
15.9 to 10.6 (−0.019).

- ~40% of the residual is at rows where the node's own 1,024-iteration
  search prefers the trump lead; the model votes no, but the label
  outweighs it there. Genuine exceptions or search artifacts — this
  corpus cannot tell. The earlier convention study (C1 residual trump
  leads mildly bad; fail-preference exceptions hindsight-suspect) leans
  toward artifacts.
- ~60% is at rows where the edge against trump is real but smaller; one
  KL-limited step only halves it. This is the per-iteration step limit,
  not a disagreement; further iterations continue it.
- Levers toward 0: iterate; and adjudicate the search-endorsed
  exceptions with an estimator outside the search family (oracle
  referee, or hindsight replay of those nodes).

### 7.6 Proposed fix: OOF-calibrated model deviations — REVISED

Original proposal (superseded in part, kept for the record): split the
model's prediction into theta_k's ordering plus the model's deviation,
and scale the deviation by an OOF calibration factor beta (mean search
edge over mean claimed edge), fitted per class or per claimed-edge bin.

**Why a scalar beta fails, from §7.5:**

- Per class: in std t0 defender leads the model's top-card overrides are
  mostly one fail card over another, aggregate support ratio +0.02
  (§7.3). A class beta would be ≈ 0 and would scale away the trump-vs-
  fail push along with them — yet that contrast is calibrated ~1:1
  (model −0.032 vs search −0.034).
- Per claimed-edge bin: the trump-vs-fail edge (~0.03) falls in the
  > 0.024 bin, ratio 0.34; it would be cut to a third.
- Selection effect in §7.3: the model's top card is chosen by the
  model's own predictions, so its claimed edge is inflated by its own
  errors (winner's curse). Part of the overrides' weak support is that
  selection, not a general miscalibration of the model. The trump-vs-
  fail contrast is defined by a fixed feature, not selected by the
  model, and is clean.

So the model is calibrated on large, consistent contrasts (trump vs
fail) and over-confident on small, selected ones (one fail card over
another, picker-row top cards). A calibration must separate the two,
which no single scalar per class or per edge bin can. Candidate forms,
not yet designed: calibrate along fixed contrasts (card type, suit
relation, point value) rather than along the model's own argmax; or a
deviation-size-dependent shrinkage fitted on contrasts not selected by
the model.

**Standing recommendation:** do NOT ship a calibration factor yet. Keep
the gate as a diagnostic only: per class, mean OOF residual at theta_k's
top card ≈ 0 (|z| < 3) and the aggregate support of the model's
overrides, reported in `target_report.json`. Any future calibration must
pass a regression test: the §7.5 decomposition on this corpus must keep
the t0 defender-lead target at or below the baseline's 2.4% trump mass
(and the alone 0.2%).

### 7.7 Checks

1. **Would it have kept the overrides that mattered?** PARTLY ANSWERED by
   §7.5 for the t0 trump lead on this corpus: the model carries the push
   and the search backs it 1:1. Still open: the called-suit convention,
   and an iteration where a convention was being installed rather than
   finished — e.g. the `rc_validate_v2` iteration whose lead schedule
   took called-suit to 56%. Those runs predate cross-fitting, so use
   their 10% held-out games (`evaluate_rows` on the fit holdout) and run
   the §7.5 decomposition plus the aggregate trump-vs-fail /
   called-suit-vs-other contrast.
2. **Is the search label leaning toward theta_k's card?** PUCT spends
   most visits under theta_k's top card; if that inflates its Q, "the
   search sides with the policy" is partly the search echoing the prior,
   which would bias both V and the §7.3 test against legitimate
   overrides. The bias probe did not save the prior per node; re-run it
   with theta_k's prior recorded (~4 h) and read the signed 256-vs-2048
   difference at theta_k's top card.
3. **The 206 search-endorsed t0 trump leads** (§7.5): adjudicate with an
   estimator outside the search family before deciding whether the
   residual should be pushed to 0 or is a genuine exception set.

### 7.8 Effect on the V/W read

None: V and W are read by §6.5 as written. If V loses at the picker
rows, the next arm is no longer "V with a calibrated model" directly —
the calibration must first be designed per §7.6 and pass its trump-lead
regression test.

Scripts (reviewing session's scratchpad, uncommitted; run from the repo
root with `uv run python`): `/private/tmp/claude-501/-Volumes-Nargothrond-dev-sheepsheadai/e4e1ea6c-83aa-4867-b852-dd7afcbf8811/scratchpad/temp_alloc_v.py <alloc_v.pt>`
(extract arm V), then `temp_defer.py <dir>` (§7.2, top-card residuals),
`temp_bias.py <dir>` (§7.4 tilt and calibration), `temp_override.py
<dir>` (§7.3), `temp_trump.py <dir>` and `temp_trump2.py <dir>` (§7.5),
where `<dir>` holds `alloc.pt` and `alloc_v.pt`.

### Comments / decisions (§7)

_(for the orchestration agent and operator)_

**Orchestration agent, 2026-10-10 23:00 — §7 verified where it could be;
agreed on the standing recommendation.** Reproduced from the reviewing
session's extracts (`temp_override.py`, `temp_trump.py` on `alloc.pt` +
`alloc_v.pt`): the §7.3 per-group and per-edge-bin tables to the printed
digit (e.g. std defender follow n=16,982 claimed .0084 vs search +.0012
z +6.5, confident-prior −.0042 z −6.7; edge < .0021 bin search −.0023
z −10.7), and the §7.5 decomposition for alone t0 (prior 0.82% → target
0.21%, rebuilt 0.21%; search −0.0511 z −67 vs OOF model −0.0489 z −111)
and std t1 defender leads (5.8% → 5.2%). §7.2 and §7.4 not re-run; the
method is the same extraction and nothing in them is load-bearing for a
decision tonight.

Comments. (1) §7.3's "preponderance" test is the right instrument but,
as §7.6 itself notes, it conditions on the model's own argmax, so the
claimed edge carries the winner's curse; the per-bin ratios (−2.4 at
claimed < 0.002, +0.34 at > 0.024) are therefore a lower bound on the
model's calibration for SELECTED contrasts and say nothing about fixed
ones — which §7.5 shows are calibrated ~1:1 (trump vs fail −0.032 vs
−0.034). Both can be true: a model can rank a fixed, well-supported
contrast correctly and still over-claim small differences among
near-equivalent cards, because the latter is where its own error
dominates the signal. That is the Fay-Herriot assumption E[y − a_model |
a_model] = 0 failing in the tails of a_model's deviation, not at its
centre. (2) §7.4's clean read — positive OOF residual at theta_k's top
card in 28/29 classes, z to +11, with a significant positive coefficient
on centered log p in every class — is the actionable finding: the fit
under-uses the prior covariate (prior_scale learned ≈ 0.005). The right
place to address it is the FIT (a per-class or deviation-size-aware
prior term, or a calibration layer fitted on fixed contrasts), not the
target stage, and not a scalar beta (§7.6's regression test is the right
gate). (3) Picker rows: V's residual lean 0.39-0.97 nats at picker rows
vs the baseline's 0.15-0.18 is the one place V is structurally more
exposed than the baseline, and V moved a third of the dose there. V's
paired match (+0.0007 ± 0.0029) and its +0.0064 ± 0.0027 over C say the
exposure did not cost on this corpus, but W's read (uniform weights
without the value-unit tilt → picker dose 15%, not 32%) will show
whether the picker dose helped, was neutral, or was carried by the
defender rows. (4) §7.5 corrects my own note of 22:00: at the TARGET
level V and the baseline push the t0 defender trump lead equally (2.6%
vs 2.4% trump mass from a 5.0% prior; alone 0.21% vs 0.21%), and under
the value-unit tilt the convention rows carry LESS dose than under the
baseline (t0 share .164 vs .206) with a larger temperature (0.0097 vs
0.0057), not more. The probe gap (V 2.6% vs baseline 1.7% of ~1,900
greedy leads, ≈ 2 SE) sits on the student side of the distill or in
probe noise, not in the targets.

Decisions. (a) §7.6 standing recommendation adopted: no calibration
factor; add the two diagnostics to `target_report.json` (per-class mean
OOF residual at theta_k's top card with z; aggregate support of the
model's top-card overrides, overall and where theta_k is confident) —
queued behind the operator's refactor, not built tonight. (b) §7.7 (2):
the bias probe will record theta_k's root prior per node so the signed
256-vs-2048 difference at theta_k's top card can be read; small change
to `search_bias_calibration.py`, queued. (c) §7.7 (3): the 206 search-
endorsed t0 trump leads go to an oracle referee / hindsight replay;
design open. (d) §7.7 (1) on `rc_validate_v2`'s called-suit iteration:
queued as an analysis read, no new compute. (e) Relaunch on V is the
operator's call and is CANCELLED for now (refactor first); V and W read
by §6.5 as written: V matched → V is the pinned recipe (`d9f4391`).

---

## 8. Fourth review (2026-10-13) — what does the model's push cost where the node's search does not back it?

Status: **MEASURED; recipe unchanged (operator, 2026-10-13).** Operator's
question: §7.5 is the override we want, where the model amplifies a
convention that each node's search can't resolve. This is the other
side. How often, and how far, does the model move θ_k's logits where
the node's own search doesn't support it, and does that cause harm we
can measure? Answered on iteration 1's production targets (arm V,
`pi/iter1`, 137,765 rows, out-of-fold a_model).

### 8.1 Method: split the target and judge each part by the node's label

The target is t = softmax(log p + clip((γy + (1 − γ)a_model)/η, ±8)).
Rebuild two partial targets from the same terms (the full rebuild
matches the stored targets exactly):

- t_s: the search term γy alone;
- t_m: the model term (1 − γ)a_model alone.

Value each move p → q as Σ(q − p)·y, the gain the node's own centered
search label credits it. All values below are game points per decision
(advantage × RETURN_SCALE 12).

- **model-only gain** = value of t_m. Clean: the OOF model never saw y,
  so this is unbiased for t_m's true search-Q value.
- **model claim** = t_m valued by a_model itself. The ratio gain/claim
  is the model's calibration on the moves it actually makes.
- **marginal** = value(t) − value(t_s): what adding the model term adds
  inside the real target. Both t and t_s depend on y, so their label
  values are biased up by fitting the noise. They are Stein-corrected,
  E[Σ d_a ε_a] = Σ_ab C_ab ∂d_a/∂y_b, assuming Gaussian label noise
  with covariance s2·(I − 11'/k) on the legal set and s2 the WORKING
  row noise the blend used.
- **extra movement** = KL(t ‖ t_s), how far the model moves the target
  beyond what the node's search would.
- **changed / vs evidence** = the target's top card is not t_s's; then
  additionally with the node's label favouring t_s's card by > 2 SE.

**Correction to the first pass.** The first pass (reported in the
session before this section was written) took s2 from `row_table.pt`,
which stores the RAW corpus noise. The working (class-pooled, centered)
noise is ~0.5× raw at defender and partner rows and 1.7–2.4× at picker
rows, so that pass over-corrected at the former and under-corrected at
the latter. It reported a harmful tail (top 1% of extra movement:
marginal −0.038, z −3.5, −52 points over the corpus) that does not
survive the correct noise (§8.3). The diagnostic now takes the working
noise explicitly (`model_push_diagnostics(..., noise_var)`; post hoc,
s2 = v_post / γ from the targeted shards).

### 8.2 By group

| group | rows | extra KL | model-only gain | claim | gain/claim | marginal | target | top card changed / vs > 2 SE |
|---|---|---|---|---|---|---|---|---|
| std def lead t0 | 3,032 | .042 | +.0220 | +.0255 | 0.86 | +.0152 | +.0114 | 12.8% / 0.00% |
| std def lead t1 | 3,361 | .036 | +.0161 | +.0199 | 0.81 | +.0104 | +.0065 | 13.4% / 0.03% |
| std def lead t2–t4 | 8,132 | .015 | +.0088 | +.0126 | 0.70 | +.0027 | +.0338 | 6.5% / 0.30% |
| std defender follow | 54,124 | .024 | +.0117 | +.0164 | 0.71 | +.0047 | +.0302 | 5.6% / 0.24% |
| std partner follow | 13,909 | .016 | +.0078 | +.0133 | 0.59 | +.0034 | +.0300 | 4.1% / 0.17% |
| std partner lead | 5,962 | .021 | +.0071 | +.0138 | 0.52 | +.0045 | +.0143 | 6.5% / 0.13% |
| std picker follow | 14,857 | .050 | +.0187 | +.0334 | 0.56 | +.0139 | +.0357 | 6.5% / 0.18% |
| std picker lead | 10,900 | .042 | +.0175 | +.0277 | 0.63 | +.0118 | +.0271 | 6.5% / 0.15% |
| alone def lead t0 | 955 | .022 | +.0075 | +.0107 | 0.70 | +.0060 | −.0069 | 15.7% / 0.00% |
| alone def lead t1 | 459 | .028 | +.0134 | +.0169 | 0.79 | +.0083 | +.0050 | 15.0% / 0.00% |
| alone defender follow | 15,064 | .020 | +.0154 | +.0184 | 0.84 | +.0056 | +.0374 | 5.4% / 0.12% |
| alone picker follow | 2,247 | .090 | +.0328 | +.0793 | 0.41 | +.1225 | +.0343 | 7.3% / 0.13% |
| alone picker lead | 3,767 | .031 | +.0145 | +.0217 | 0.67 | +.0427 | −.0045 | 3.2% / 0.11% |
| **all** | 137,765 | .028 | **+.0133 (z +64)** | +.0201 | 0.66 | **+.0094 (z +44)** | +.0286 | 6.1% / 0.18% |

By θ_k's top-card probability:

| θ_k top | rows | extra KL | model-only gain | marginal | changed / vs > 2 SE |
|---|---|---|---|---|---|
| < 0.5 | 23,878 | .040 | +.0190 | +.0115 | 15.1% / 0.15% |
| 0.5–0.8 | 57,707 | .035 | +.0156 | +.0122 | 7.3% / 0.19% |
| 0.8–0.95 | 24,742 | .028 | +.0144 | +.0121 | 1.9% / 0.27% |
| ≥ 0.95 | 31,438 | .007 | +.0038 (z +12) | +.0008 (z +2.3) | 0.3% / 0.13% |

### 8.3 The tail: rows the model moves most

By quantile of extra movement KL(t ‖ t_s):

| bin | rows | extra KL | model-only gain | claim | marginal | changed / vs > 2 SE |
|---|---|---|---|---|---|---|
| < q50 | 68,882 | .001 | +.0069 | +.0055 | +.0014 | 0.9% / 0.01% |
| q50–q80 | 41,330 | .012 | +.0135 | +.0155 | +.0092 | 6.5% / 0.09% |
| q80–q90 | 13,776 | .045 | +.0219 | +.0331 | +.0218 | 11.1% / 0.20% |
| q90–q95 | 6,888 | .089 | +.0311 | +.0552 | +.0333 | 15.5% / 0.49% |
| q95–q99 | 5,511 | .194 | +.0390 | +.0998 | +.0524 | 27.5% / 1.16% |
| ≥ q99 | 1,378 | .746 | +.0444 (z +5.6) | +.2591 | +.0030 (z +0.2) | 69.3% / 6.24% |

The top 1% splits by group into positive defender and partner rows
(marginal +0.03 to +0.09) and slightly negative picker rows (std picker
follow −0.032 on 435 rows, alone picker lead −0.055 on 69). It nets to
+4 points over the corpus, against +1,302 for the model term on all
rows. The top 0.5% reads −0.067 marginal on 689 rows: about −46 points.

### 8.4 Readings

- **The model's push helps in every group.** The clean model-only gain
  is positive in all 50 classes, at z ≥ 3 in 43 of them; the
  rest are small or late-trick picker cells. Nowhere does the model, on
  aggregate, move θ_k toward cards the search labels rate lower.
- **It over-claims by about a third,** and more at picker and partner
  rows (gain/claim 0.41–0.63) than at defender rows (0.70–0.86). This is
  the §7.3 finding in a cleaner form: §7.3 conditioned on the model's
  argmax (winner's curse); this read values the move the target
  actually makes. At the convention rows (std t0/t1 defender leads) the
  model is close to calibrated (0.86 / 0.81) and its marginal is the
  largest in the table.
- **Confident θ_k (≥ 0.95, 23% of rows): the model barely moves them,
  and does no harm.** Extra KL is 0.007; it changes the top card at 0.3%
  of rows and the marginal is ~0 (+0.0008). θ_k's top card loses less
  mass under the full target than under the search term alone (0.008
  vs 0.012): on confident rows the model is the conservative term.
- **Against the node's clear evidence:** the model reverses a > 2 SE
  label at 0.18% of rows (~250), concentrated in the extreme tail (6.2%
  of the top 1%).
- **The tail is where it stops paying.** Value per row rises with
  extra movement up to q99. Beyond it, the model-only push is still
  positive (+0.044) but claims 6× that, and adding it to an already
  large search tilt over-concentrates into the clip: marginal ~0 at the
  top 1% and negative at the top 0.5%, mostly at picker rows.
- **Size of any harm:** at most the top 0.5% tail, about −46 points over
  the corpus, about −0.001 per seat-hand on targets, before distillation
  dilutes it. That is well inside the paired h2h SE (0.0029), so no
  arm could detect it in EV. It is ~4% of the model term's +1,302.

### 8.5 How reliable is this?

- **Model-only gain: reliable** as a search-Q read. It is unbiased under
  out-of-fold prediction and needs no noise model.
- **Marginal / target / search-only values: noise-model dependent.** The
  Stein correction is 0.03–0.06 at defender and partner rows, against a
  raw search-only label gain of ~0.04. At picker rows it is 0.09–0.35,
  against raw gains of 0.09–0.14. That is why the search-only and target
  values go negative at some picker classes (alone t0 picker follow
  target −0.070, search-only −0.35). Read literally, tilting toward the
  node's own search label would hurt there, which is implausible. The
  likelier story is that the class-pooled working noise overstates
  those rows' true noise: the first pass with raw noise moved the tail
  read by 0.04 points per decision. Treat the picker-row marginal
  values, and so the tail's sign there, as unresolved within
  noise-model uncertainty. Defender and partner reads are robust in
  sign.
- **The judge is the search itself.** If PUCT echoes θ_k's prior (§7.7
  check 2), the labels favour θ_k's card. The model's departures then
  look worse than they are, so the negative tail is an upper bound in
  that direction. Structural search error is invisible to any of these
  reads.
- **Real EV: not measurable.** The candidate harm is ~1/3 of the paired
  h2h SE before distillation dilutes it. The offline search-judged read
  is the only practical instrument.

### 8.6 Decision and diagnostics

Recipe unchanged (operator, 2026-10-13): the model does what it was
designed to do, and the harm is at most barely measurable. §7's
standing item (a) is now built:

- **Stage 2** computes `model_push_diagnostics` (`search_advantage.py`)
  and writes it to `target_report.json` as `model_diagnostics`:
  - overall push stats; by θ_k top-card bin; by extra-movement quantile;
  - per class: the push stats, the §7.4 top-card residual (mean with z)
    and the §7.3 override support;
  - the Stein corrections, so the noise-model dependence is visible;
  - `a_model_out_of_fold`: false means model-only is biased up.
  The log prints the summary and per-class lines in game points. It is
  diagnostic only; nothing downstream reads it.
- **Post hoc:** `python -m sheepshead.analysis.target_diagnostics
  --iter-dir <pi/iterN>` recomputes it for an iteration whose Stage 2
  predates the change (iteration 1 above). It needs cross-fit OOF
  predictions.
- **Iteration 2's Stage 2** runs it automatically. To watch:
  - whether the q99 tail and the picker gain/claim ratios persist;
  - whether the convention classes' marginal shrinks as the convention
    completes;
  - the top-card residual gate. In iteration 1, 35 of 50 classes have
    |z| ≥ 3, all positive, the §7.4 lean (largest at alone
    picker rows: +0.16 to +0.57 points).

Scripts (scratchpad, this session): `temp_harm_w.py <iter-dir> <out.pt>`
(row-level split with the working noise) and `temp_harm_tail.py
<out.pt>` (tail by group); `temp_stein_check.py` (the size of the
correction by role). The committed CLI reproduces §8.2–§8.3.
