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

### 8.7 Iteration 2 (2026-10-13): the same read, one step later

Iteration 2's Stage 2 ran on `ab222c8`, so `pi/iter2/target_report.json`
carries `model_diagnostics` (134,226 rows, out-of-fold, η 0.0097, cross-
fit 5). These are the targets built from θ₁'s corpus, so they describe
the update iteration 2 distilled, not the distilled policy's behaviour.
Iteration 2's cert passed: h2h vs θ₁ +0.0118 ± 0.0027 (duplicate, 16,000
deals).

**Overall** (game points per decision):

| | iter 1 | iter 2 |
|---|---|---|
| KL(target ‖ prior), mean | 0.136 | 0.105 |
| model's extra movement KL(t ‖ t_s) | 0.028 | 0.021 |
| model-only gain | +0.0133 (z +64) | +0.0080 (z +52) |
| model claim / gain-to-claim | +0.0201 / 0.66 | +0.0136 / 0.59 |
| marginal (Stein, working noise) | +0.0094 (z +44) | +0.0074 (z +34) |
| marginal summed over the corpus | +1,302 pts | +993 pts |
| top card changed / vs > 2 SE label | 6.1% / 0.18% | 5.0% / 0.18% |
| top-card residual \|z\| ≥ 3 | 35 of 50 classes | 26 of 50 |
| override support (model's argmax) | +0.0054 (z +2.5) | −0.0224 (z −9.4) |
| … where θ_k ≥ 0.8 | −0.080 (z −10.8) | −0.122 (z −14.8) |

With η fixed, the step shrinks by itself as θ improves: less total
movement, a smaller model-only gain, a smaller marginal. Both stay
clearly positive.

**By group** (gain = model-only, marg = marginal, resid = top-card
residual):

| group | iter 1 KL / extra / gain / g÷c / marg / resid | iter 2 KL / extra / gain / g÷c / marg / resid |
|---|---|---|
| std def lead t0 | .109 / .042 / +.0220 / .86 / +.0152 / +.0001 | .067 / .028 / +.0112 / .74 / +.0089 / +.0040 |
| std def lead t1 | .099 / .036 / +.0161 / .81 / +.0104 / −.0033 | .071 / .026 / +.0085 / .67 / +.0062 / +.0106 |
| std def lead t2–t4 | .145 / .015 / +.0088 / .70 / +.0027 / +.0033 | .116 / .010 / +.0049 / .68 / +.0012 / +.0105 |
| std defender follow | .115 / .024 / +.0117 / .71 / +.0047 / +.0120 | .086 / .019 / +.0061 / .57 / +.0024 / +.0064 |
| std partner follow | .127 / .016 / +.0078 / .59 / +.0034 / +.0147 | .103 / .011 / +.0055 / .63 / +.0021 / +.0069 |
| std partner lead | .098 / .021 / +.0071 / .52 / +.0045 / +.0152 | .076 / .016 / +.0059 / .56 / +.0036 / +.0004 |
| std picker follow | .220 / .050 / +.0187 / .56 / +.0139 / +.0903 | .192 / .036 / +.0139 / .56 / +.0107 / +.0832 |
| std picker lead | .159 / .042 / +.0175 / .63 / +.0118 / +.0591 | .126 / .030 / +.0129 / .68 / +.0097 / +.0644 |
| alone def lead t0 | .081 / .022 / +.0075 / .70 / +.0060 / −.0008 | .085 / .017 / +.0056 / .65 / +.0052 / +.0036 |
| alone def lead t1 | .136 / .028 / +.0134 / .79 / +.0083 / +.0236 | .113 / .013 / +.0068 / .79 / +.0058 / +.0536 |
| alone def lead t2–t4 | .174 / .010 / +.0085 / .74 / +.0024 / +.0020 | .131 / .005 / +.0035 / .72 / +.0009 / +.0116 |
| alone defender follow | .134 / .020 / +.0154 / .84 / +.0056 / +.0109 | .082 / .014 / +.0077 / .73 / +.0030 / +.0034 |
| alone picker follow | .241 / .090 / +.0328 / .41 / +.1225 / +.4289 | .190 / .094 / +.0240 / .32 / +.1830 / +.5243 |
| alone picker lead | .121 / .031 / +.0145 / .67 / +.0427 / +.1752 | .104 / .029 / +.0099 / .55 / +.0478 / +.1545 |

**By θ_k's top-card probability:**

| θ_k top | iter 1 rows / extra / gain / marg / changed | iter 2 rows / extra / gain / marg / changed |
|---|---|---|
| < 0.5 | 23,878 / .040 / +.0190 / +.0115 / 15.1% | 19,150 / .032 / +.0098 / +.0072 / 14.2% |
| 0.5–0.8 | 57,707 / .035 / +.0156 / +.0122 / 7.3% | 52,965 / .028 / +.0094 / +.0106 / 6.8% |
| 0.8–0.95 | 24,742 / .028 / +.0144 / +.0121 / 1.9% | 26,786 / .022 / +.0106 / +.0100 / 1.2% |
| ≥ 0.95 | 31,438 / .007 / +.0038 / +.0008 / 0.3% | 35,325 / .005 / +.0029 / +.0007 / 0.3% |

On θ_k ≥ 0.95 rows, the top-card mass lost is 0.0016 under the target
vs 0.0063 under the search-only tilt: the model is again the
conservative term there. Only 41 rows lose > 0.2 of the top card
because of the model; their model-only value is +0.013 (z +0.5).

**By the model's extra movement:**

| bin | iter 1 extra / gain / claim / marg | iter 2 extra / gain / claim / marg |
|---|---|---|
| < q50 | .001 / +.0069 / +.0055 / +.0014 | .000 / +.0053 / +.0039 / +.0010 |
| q50–q80 | .012 / +.0135 / +.0155 / +.0092 | .009 / +.0087 / +.0104 / +.0079 |
| q80–q90 | .045 / +.0219 / +.0331 / +.0218 | .034 / +.0126 / +.0219 / +.0168 |
| q90–q95 | .089 / +.0311 / +.0552 / +.0333 | .069 / +.0168 / +.0363 / +.0276 |
| q95–q99 | .194 / +.0390 / +.0998 / +.0524 | .151 / +.0192 / +.0670 / +.0379 |
| ≥ q99 | .746 / +.0444 (z +5.6) / +.2591 / +.0030 | .548 / **−.0135 (z −2.0)** / +.1869 / −.0068 |
| ≥ q99.5 | — / — / — / −.067 | — / −.037 / — / −.081 |

**Iteration 2's top 1% by group** (1,343 rows):

| group | rows (share of group) | marginal | model-only | claim |
|---|---|---|---|---|
| std picker follow | 416 (2.8%) | −.073 | −.025 | +.208 |
| std defender follow | 396 (0.7%) | −.030 | −.002 | +.155 |
| std picker lead | 195 (1.7%) | −.030 | +.007 | +.170 |
| alone picker follow | 158 (9.8%) | +.291 | −.055 | +.277 |
| alone picker lead | 56 (1.8%) | −.093 | −.017 | +.102 |
| alone defender follow | 47 (0.4%) | −.058 | −.012 | +.154 |
| std partner follow | 45 (0.3%) | +.035 | +.051 | +.202 |
| std defender lead | 20 (0.1%) | −.034 | −.019 | +.141 |
| std partner lead | 10 (0.2%) | +.012 | +.020 | +.159 |

The tail sums to −9 points against +993 for the model term on all rows:
−0.00025 per seat-hand on the targets (36,000 seat-hands), before
distillation dilutes it. The +0.29 marginal at alone picker follows is
the noise correction at picker rows (§8.5), not a reliable read; the
model-only −0.055 is.

**Readings:**

- **The model still helps everywhere it matters.** The model-only gain
  is positive in every group, and the marginal is positive in every
  group and every θ_k bin. Both shrink as θ improves, as a fixed-η step
  should.
- **Convention rows: the convention is being finished, not lost.** At
  std t0 defender leads, total movement fell 0.109 → 0.067 and the
  model's marginal +0.0152 → +0.0089. t1 is the same.
- **Over-claiming grew slightly** at defender follows (gain/claim 0.71
  → 0.57) and alone picker follows (0.41 → 0.32), and the q99 claim
  stayed large (+0.19) while the real signal shrank.
- **New in iteration 2: the clean model-only gain turns negative in the
  extreme tail** (top 1%: −0.0135, z −2.0; top 0.5%: −0.037). In
  iteration 1 it was positive there (+0.044). As θ improves, the true
  signal shrinks while the model's largest claims stay large, so in
  the tail the selection effect has started to outweigh the signal. It
  sits mostly at picker follows and alone picker rows.
- **The override-support read also went negative** (−0.022, z −9.4).
  It conditions on the model's own argmax and carries the winner's
  curse (§7.6). The clean tail read above is the better evidence.
- **The §7.4 top-card lean eased in most classes** (26 of 50 at |z| ≥ 3,
  from 35) but not at alone picker rows (+0.15 to +0.52).

**Standing:** no recipe change at this size. The trigger to act is
iteration 3's tail: if the top-1% model-only gain falls further and the
negative region spreads into q95–q99, design a cap on the model's
extra movement. One natural form scales the model term down where it
pushes far past the search term. It must pass the §7.6 trump-lead
regression test (the t0 defender-lead target at or below 2.4% trump
mass).

Scripts: as §8, with `temp_harm_tail.py <out.pt> <iter-dir>` now
taking the iteration directory.

### 8.8 Reproducing §8

Every §8 table is reproducible from committed code (`ab222c8`,
`6a09552`, and the tail breakdown added after §8.7). The scratchpad
scripts named above were the first pass and are not needed.

**Source.** `model_push_diagnostics` (`search_advantage.py`) computes
everything. Stage 2 writes it to `<iter-dir>/target_report.json` under
`model_diagnostics` and prints a summary in `policy_iteration.log`
(`[model]` lines). Iteration 2 onward have it already.

**Recompute** (needed for iteration 1, whose Stage 2 predates it, or to
regenerate any iteration):

```
uv run python -m sheepshead.analysis.target_diagnostics \
    --iter-dir runs/202609_recall_rc/pi/iter1 --json /tmp/iter1_diag.json
```

It reads the iteration's `row_table.pt`, `oof_predictions.pt` (so it
needs a cross-fit iteration), `target_report.json` and the targeted
shards. It recovers the working noise as v_post / γ from the shards.
Do not use the table's `noise_var`, which is RAW (§8.1). ~25 s per
iteration. On iterations 1 and 2 it matches the tables above to the
printed digit.

**Where each table comes from** in the JSON (all values in advantage
units; × 12 = game points per decision):

| table | JSON key |
|---|---|
| §8.2 / §8.7 overall | `push` |
| §8.2 / §8.7 by group | `per_class[c].push`, rows-weighted over the group's classes |
| θ_k top-card bins | `push_by_prior_top` |
| top-card mass lost (θ_k ≥ 0.95) | `push_by_prior_top["[0.95,1.0]"].prior_top_lost` |
| §8.3 / §8.7 extra-movement bins | `push_by_extra_kl` (q99.5 is not a bin; see below) |
| top 1% by group | `push_top_extra_by_class`, rows-weighted over the group's classes |
| §8.5 correction sizes | `stein` in any push entry (raw label gain = value + stein) |
| §7.4 top-card residual | `per_class[c].top_card_residual`; gate list `top_card_residual_classes_abs_z_ge_3` |
| §7.3 override support | `override_support`, `override_support_prior_confident`, `per_class[c].override_support` |

Rolling classes up into §8's groups (e.g. "std picker follow" = all
`std|t*-picker-follow`; "def lead t0" kept by trick):

```
import json, collections
d = json.load(open("/tmp/iter1_diag.json"))
src = d["per_class"]  # or d["push_top_extra_by_class"] for the tail
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for c, v in src.items():
    p = v.get("push", v)
    mode, rest = c.split("|", 1)
    t, role, kind = rest.split("-")
    g = f"{mode} {role} {kind}" + (f" {t}" if (role, kind) == ("defender", "lead") else "")
    a = agg[g]; n = p.get("n", 0); a["n"] += n
    for f in ("model_gain", "marginal", "full"):
        a[f] += n * (p[f]["mean"] if n else 0)
    a["claim"] += n * p.get("model_claim", 0)
for g, a in sorted(agg.items()):
    print(g, int(a["n"]), *(f"{12 * a[f] / a['n']:+.4f}" for f in ("model_gain", "claim", "marginal")))
```

**Not in the JSON:**

- **The q99.5 cut** (§8.3 "−0.067 on 689 rows", §8.7 "−0.037 /
  −0.081"). To reproduce it, call `model_push_diagnostics` on a table
  restricted to the rows above the q99 cut, or change
  `EXTRA_KL_QUANTILES`.
- **The "41 / 82 rows where the model alone costs θ_k's top card
  > 0.2"** count. It needs the row-level split: rebuild t and t_s as the
  function does.
- **§8.1's raw-noise first pass.** Pass `table.noise_var_raw` as
  `noise_var` to reproduce it. It is kept only as the record of the
  mistake.

## 9. Fifth review (2026-10-16) — a cell where the model's push is wrong: the called-suit over-trump

Status: **ADOPTED (2026-10-16): one-sided head variance is the program
default (`8e59391`) and the arm replaced iteration 3's candidate
(§9.8).** The operator played
iteration 3's class-variance candidate (then `pi/iter3/distill_epoch10.pt`,
archived to `pi/iter3_ablations/class_baseline/distill_epoch10.pt`) and saw a
defender over-trump a fellow defender who had already trumped the
called suit. §8 measured the model's push in aggregate and found it
helps almost everywhere. This section is the counterexample: a cell
where the model term is wrong-signed and wins the blend. It also tests
general fixes for that, with no game-specific branching in the recipe
(operator constraint).

Units: Q is score / RETURN_SCALE (12); every "score" figure is game
score per decision (Q × 12). Scripts: `runs/202609_recall_rc/overtrump_probe/`
(§9.8).

### 9.1 The spot and the rule that makes it a blunder

Called-ace mode. When the called suit is led, the picker must follow
suit (it holds a called-suit card, or the under card on an under-call),
and the secret partner must play the called ace (`game.py` ~1255–1280).
So once a defender trumps a called-suit lead, **only another defender
can beat it**. Seat order relative to the picker gives no reason to
"protect" the trick. A following defender who is void in the suit
should throw points (schmear) or a low card, not spend a trump to take
the trick from a teammate.

**Decision definition** (used for every rate below): the called suit
was led; a defender is currently winning it with a trump; the defender
to act is void in the called suit; it holds at least one trump that
beats the winner AND at least one other legal play (a fail, or a lower
trump).

### 9.2 Incidence

**Iteration 3 candidate** (`scan_overtrump.py`: deterministic /analyze
playouts, called-ace mode, seeds 0–3999 = 3,022 standard called-ace
hands): 359 decisions in 344 tricks (330 tricks with one qualifying
defender, 13 with two, 1 with three), about 12 per 100 hands.

| | n | over-trumped |
|---|---|---|
| all decisions | 359 | **30.9% ± 2.4** (mean p(over) 0.30) |
| tricks with ≥ 1 over-trump | 344 | 31.4% |
| has a fail to play instead | 336 | 28.9% |
| … and the over-trump is not AD/10D (§9.3: those break even) | 336 | **22.6%** (clear blunders) |
| holding a 10/A fail (schmear) | 176 | 29.0% |
| 10D winning, holding a fail | 52 | 34.6% ± 6.6 |
| only trumps in hand (over- vs under-trump) | 23 | 60.9% |

By trick: t0 102 decisions (23% over-trumped), t1 65 (37%), t2 82
(33%), t3 62 (35%), t4 48 (31%). About 3.7 over-trumps per 100 standard
called-ace hands, about 2.5 of them clear blunders.

By what is winning: diamond 38%, jack 22%, queen 15%. By players still
to act after the decider: 0 → 22%, 1 → 33%, 2 → 44%, although nobody
behind can take the trick (§9.4).

**Corpus policies** (θ₀, θ₁, θ₂ act-time priors in the iteration 1–3
targeted corpora, `extract.py`, 961 decisions): argmax over-trump 42% /
37% / 37%. These are corpus nodes (committee-acted games), not fresh
deterministic deals, so they are only roughly comparable with the 30.9%.

Example seeds (/analyze, Called-Ace, Deterministic, iteration 3
checkpoint loaded): 1 step 9, 441 step 15, 639 step 28, 759 step 14,
1613 step 28 (10D winning, over-trumped with a jack holding 10♥). Full
list: `scan_iter3.jsonl`.

### 9.3 Value

**Search verdict** on the corpus decisions: policy-preferred
over-trump minus policy-preferred other play, from the corpus committee
labels. **Every follow node in this corpus is searched at 256
iterations × 3 replicates** (`--iters 256 --iters-schedule
t0-lead:1024,t1-lead:512`). §9.5 re-searches the iteration-3 nodes at
512.

| situation (corpus, θ₀–θ₂) | n | over − alt | search's top card is an over-trump |
|---|---|---|---|
| all decisions | 940 | −0.52 ± 0.02 | 16% |
| where the policy over-trumps | 364 | −0.13 ± 0.03 | 27% |
| … holding a schmear, not AD/10D, picker already played | 74 | **−0.50 ± 0.06** (30 nodes z < −2, none z > 2) | 7% |
| … no schmear, not AD/10D, picker already played | 62 | −0.20 ± 0.05 | 21% |
| … over-trump card is AD or 10D | 79 | +0.06 ± 0.06 (break-even: the trump carries 10–11 points itself) | 42% |
| … picker still to play | 135 | −0.02 ± 0.04 | 32% |
| 10D winning, Q/J over-trump, holds a fail (operator's case) | 116 | −0.53 ± 0.06 | 10% |

The "picker still to play" cell is within noise of zero. My first
reading called it a legitimate exception ("protect against the picker
over-trumping"). That is wrong under §9.1's rule: nothing is
protected. The measured near-tie there has no protection mechanism
behind it, and every cell is negative or within noise of zero.

**Win probability.** Absolute search Q is optimistic against realized
outcomes. On 25,685 called-ace defender follows (iteration-3 corpus, Q
of the action taken vs final score), the bias is +1.04 score at t0,
+0.97 t1, +0.79 t2, +0.61 t3, +0.33 t4. Scoring is double on the bump:
a defender win pays +2/+4/+6 and a loss costs −1/−2/−3, so Q = 0 is
about a ⅓ win chance, not ½. Calibrating Q → P(defenders win) by
binning (`absolute.py`): Q 0 → 0.19, +1 → 0.36, +1.5 → 0.50, +2 → 0.82.

| where the policy over-trumps | n | P(win) alternative → over-trump | Δ | sign flips win → loss |
|---|---|---|---|---|
| all | 364 | 0.54 → 0.51 | −3 pts | 2% |
| holding a schmear (not AD/10D) | 142 | 0.56 → 0.49 | −8 pts | 3% |
| operator's case (10D winning, Q/J over, has fail) | 53 | 0.56 → 0.48 | −7 pts | 6% |

Defenders are usually ahead when this happens (88% of nodes have a
positive alternative Q), so the over-trump mostly trims the margin
(the schmear points, the +4 at 90 points). In the operator's case it
costs the hand about 7 times in 100. Realized outcomes (confounded,
sanity only): defenders won 47% of hands where the corpus over-trump
was played vs 62% where it was not; Q-implied 0.51 vs 0.63.

### 9.4 Why distillation does not remove it

**Where search is confident, it works.** At the 444 corpus decisions
where search says over-trumping is worse at z < −2, the model agrees
86% of the time. θ_k already rarely over-trumps there (0.18), and the
target cuts that to 0.08.

**Where the policy over-trumps, the model is wrong-signed.** Model
+0.15 vs search −0.13 (schmear cell: model +0.09 vs search −0.50), and
γ ≈ 0.4 gives the model most of the blend. Target over-trump mass
barely moves: prior 0.37 → target 0.31 averaged over iterations 1–3.

**The class variance hides the cell's error.** Fay-Herriot terms on
iteration 3 (working noise, i.e. after `apply_noise_model`; `variance.py`):

| set | n | E[r²] | E[noise] | σ_u² (E[r²] − noise) | class σ_u² | γ used | γ with the cell's σ_u² |
|---|---|---|---|---|---|---|---|
| std t0–t4 defender follow | ~11k each | 2.6–2.8e-4 | 2.1e-5 – 2.0e-4 | 7.8e-5 – 2.5e-4 | 5.1e-5 – 2.1e-4 | 0.21–0.91 | — |
| over-trump cell | 303 | 7.8e-4 | 1.7e-4 | **6.1e-4** | 1.07e-4 | **0.41** | **0.79** |

The model's error in the cell is about 5.7× its class's. The class
estimate dilutes it across ~11,000 easy follows.

*Correction.* My first pass of this table read `noise_var` straight
from `row_table.pt`, which is the RAW corpus noise. That is §8.1's
mistake again; §8.8 records it. It gave γ 0.28 → 0.64. The values above
apply the noise model first and match the stored `pi_gamma` (0.41).

**Bleed from a look-alike.** The same decision on a non-called fail
lead (picker can trump if void, partner known):

| lead | picker behind me | policy covers | search: cover − alt | model |
|---|---|---|---|---|
| other fail suit | yes | 75% | −0.05 ± 0.06 (tie) | −0.04 |
| other fail suit | no | 49% | −0.26 | −0.10 |
| **called suit** | **yes** | **62%** | **−0.14 ± 0.04** | **+0.01** |
| called suit | no | 39% | −0.62 | −0.34 |

(diamond winning, Q/J over-trump; corpus θ₀–θ₂.) Covering in the
look-alike is a legitimate tie. The policy carries the same "picker
behind me" response into the called-suit lead, where it no longer
applies. The model reads the called-suit cell at the look-alike's
value. Bleed explains the picker-behind cell but not the
picker-already-played cell (39% cover at −0.62). The broader source
there is plausibly "a teammate's low trump is usually not safe" (unknown
partner, JD mode); that hypothesis is not measured.

**Not a representation gap.** Linear probes on the frozen θ₂ state
features the model reads (`probe_features.py`; 38,343 called-ace
follows, held out by game):

| label | rows | base rate | held-out AUC |
|---|---|---|---|
| called suit was led | defender follows | 0.17 | 0.95 |
| trick secured (called suit led, defender winning with a trump) | defender follows | 0.03 | 0.99 |
| secured, given the called suit was led | defender follows | 0.18 | 1.00 |

An aux head that classifies the led suit would add information the
trunk already encodes linearly. This matches CE_Teacher's earlier
probe (called-suit membership AUC 0.94), where the lever was the
bilinear state × card term, not more information.

### 9.5 More search budget is not the lever

The 359 iteration-3 decisions were regenerated as corpus rows by the
real `play_corpus_game`, with three patches: the deal is the /analyze
seed's, every seat's action is forced to the logged one, and search
fires only at the targets. Each was searched at 256 and at 512
iterations (3 replicates, same det_rng; θ₃ = `distill_epoch10` acting
and searching; `budget_corpus.py`). They were then targeted with
iteration 3's own pieces (θ₂ encoding, iteration 3's AdvantageModel,
class σ_u², noise pools, η; `budget_targets.py`). At 256, 9 nodes got no
usable committee summary (no target at all); at 512, 1 did. 350 are
compared.

| | 256 | 512 |
|---|---|---|
| search verdict, over-trump − alternative | −0.50 ± 0.04 | −0.52 ± 0.04 (corr 0.91) |
| replicate variance of centered Q | — | ×0.85 |
| row noise (working) | 1.64e-4 | 1.51e-4 (these nodes only) / 1.37e-4 (whole class) |

| target scenario | γ | over-trump mass, θ₃ prior → target |
|---|---|---|
| 256, as configured | 0.44 | 0.303 → 0.269 |
| 512 at these nodes only | 0.46 | 0.303 → 0.273 |
| 512 for the whole class | 0.48 | 0.303 → 0.272 |
| 256 + the cell's own σ_u² | 0.80 | 0.303 → 0.247 |
| 512 + the cell's own σ_u² | 0.83 | 0.303 → 0.248 |
| search only (γ = 1) | 1.00 | 0.303 → 0.236 |

- 256 is not biased here: 512 gives the same verdict.
- Doubling iterations cuts replicate variance by only 15%. Most of the
  spread between replicates does not shrink with iterations; presumably
  it is the sampled hidden-card worlds. γ moves 0.44 → 0.48 and the
  target does not move.
- The model-error variance is the lever: the cell's own σ_u² recovers
  most of the gap to the search-only target.
- **One iteration has a ceiling regardless.** Even search-only moves
  over-trump mass only 0.30 → 0.24. Search still ranks an over-trump
  first on ~23% of these nodes, η bounds the step, and distillation
  realizes only part of a target's shift on rare rows (CE_Teacher's
  "projection" finding). A fix lowers incidence over iterations, not
  in one.

### 9.6 Can the model learn it? Oversampling (diagnostic, not a recipe)

Iteration 3's cross-fit reproduced exactly (θ₂ features, centered
noise, 5 game folds, seed 0, 22 fixed epochs, adapter + heteroscedastic
+ bilinear), with the cell's training rows repeated k times
(`oversample_fit.py`). k = 1 reproduces the stage's out-of-fold wMSE
(4.3615e-04). Model (out-of-fold) vs 256-iteration search, over-trump −
alternative:

| | k = 1 | k = 5 | k = 20 | search |
|---|---|---|---|---|
| cell (303) | −0.32 | −0.43 | −0.50 | −0.55 |
| … where the policy over-trumps (109) | +0.13 | +0.05 | −0.03 | −0.17 |
| … holding a schmear (158) | −0.41 | −0.50 | −0.59 | −0.67 |
| control: other suit, picker behind (34) | −0.17 | −0.18 | −0.23 | −0.22 |
| control: other suit, picker played (62) | −0.29 | −0.39 | −0.36 | −0.42 |
| global out-of-fold wMSE | 4.3615e-04 | 4.3630e-04 | 4.3643e-04 | |

The model can learn it given more of its rows, at no measurable global
cost, and the legitimate look-alike is not over-corrected. What it
learns is mostly a cell-wide offset: correlation with search in the
policy-over-trump subset stays 0.21 → 0.26, partly because 256-iteration
labels are weakest exactly there. Note: the corpus already searches
every node (`p_base 1.0`; 133,368/133,368 rows have Q), so "search
these nodes more often" is not available. More rows come only from
more games or from weighting in the fit.

### 9.7 General fixes: blend variance and fit weighting

The heteroscedastic NLL head already finds the cell (out-of-fold, cell
median vs class median): t0 16×, t1 17×, t2 7×, t3 6×, t4 6×. Under the
NLL that inflated variance *down-weights* the cell's squared error, so
the fit learns "uncertain here" instead of fixing the mean.

Tested on iteration 3's full table (133,368 rows; baseline reproduces
the stored `pi_gamma` exactly; `head_variance_test.py`,
`variance_regression_test.py`). Every variant is one-sided:
σ_u²(row) = max(estimate, class value). The estimate may only raise the
model's stated error, never make the model more confident than its
class. That was arm 4's failure mode (§CE_Teacher, per-node head under
the posterior-SE temperature).

| variant | cell σ_u² (measured 6.1e-4) | cell γ | cell over-trump target mass (prior 0.355) | table γ | rows γ +0.2 | top card changed | §8 target value (all) |
|---|---|---|---|---|---|---|---|
| class (production) | 1.07e-4 | 0.41 | 0.290 | 0.550 | — | — | +0.0149 (z 32) |
| one-sided head | 2.9e-4 | 0.56 | 0.274 | 0.587 | 6.9% | 0.56% | +0.0151 (z 32) |
| one-sided regression, raw | 9.5e-4 | 0.67 | 0.269 | 0.614 | 12.6% | 0.90% | +0.0134 (z 28) |
| one-sided regression, isotonic | 7.2e-4 | 0.67 | 0.269 | 0.617 | 12.8% | 0.99% | +0.0140 (z 29) |

The regression is gradient-boosted trees on the frozen θ₂ state
features plus two generic row statistics (legal-set size, prior max),
with no class and no game-specific input. Its target is y = mean
squared out-of-fold residual − working noise. It is cross-fitted over
the same 5 game folds with a 400-row minimum leaf (the shrinkage). Raw,
it over-predicts deciles 8–9 by 1.4–1.6×. Cross-fitted isotonic
recalibration brings every decile within ~10–20% of measured.

Groups that move in the §8 target value (game points per decision):

| group | n | class | head | regression | isotonic |
|---|---|---|---|---|---|
| std picker follow | 15,010 | +0.0340 | +0.0332 | +0.0250 | +0.0293 |
| std picker lead | 11,572 | +0.0198 | +0.0199 | +0.0150 | +0.0173 |
| alone picker lead | 2,854 | −0.0122 | −0.0108 | −0.0179 | −0.0152 |
| std defender follow | 55,254 | +0.0156 | +0.0163 | +0.0161 | +0.0159 |
| std defender lead t0 | 3,338 | −0.0178 | −0.0187 | −0.0180 | −0.0192 |

Readings:

- **The regression finds the cell with no hint**, and calibrated it
  matches the measured error. It pushes γ highest there.
- **Its table cost sits where §8.5 says this read is unresolved.** The
  regressions lose 0.0009–0.0015 table-wide, almost all at picker rows.
  §8.5 found the class-pooled working noise likely *overstates* picker
  noise. That inflates the Stein correction, which penalizes
  search-heavier targets in exactly those rows. So the picker "cost"
  is not established as real harm. It is not established as artifact
  either.
- **The head is neutral table-wide and gets ~75% of the regression's
  cell movement** (0.290 → 0.274 vs → 0.269). γ saturates in σ_u², so
  most of the gain comes early.
- The §8 read values targets by the labels that more search weight
  moves toward, so here it can show harm, not prove improvement.

**Proposed end state** (operator discussion, 2026-10-16). Classes
currently do several jobs: model-error σ_u² for γ and its shrinkage,
the search-noise pool, Fay-Herriot fit weights, and reporting. Split
them:

- **Strata = design factors only:** regime (std/alone/leaster) ×
  search-budget tier (× bidding head). These are exact, free causes of
  variance. Search noise must never pool across budgets. They also
  serve as the floor and shrinkage parent for everything learned. Trick
  × role mattered partly as a budget proxy; with budget explicit it
  becomes an ordinary feature.
- **Learned, cross-fitted variance models within strata** for σ_u²
  (and potentially search noise). Fit on out-of-fold r² minus noise,
  with a minimum-leaf or ridge shrinkage toward the stratum, recalibrated
  (isotonic), and used one-sided against the stratum floor. A separate
  second-stage model is preferred to the joint NLL head, which is
  trained to excuse its own mean errors. The floor guards against blind
  spots a self-assessed error model cannot see.
- **Residual-driven fit weighting.** Upweight rows whose learned excess
  error is high, applied to the mean term so the variance head cannot
  absorb it. This is what fixes the model rather than routing around
  it. It is §9.6's oversampling generalized, and close to group DRO /
  JTT. Pooled per region, never per row (row residuals are mostly
  search noise). **The reference must be the error expected for that
  kind of row** (class-like), not the coarse stratum. The Fay-Herriot
  fit weights exist to stop high-variance picker cells (σ_u² ~2e-3)
  from dominating the low-variance lead cells (~1e-4) where conventions
  live (`fit_advantage_model`, `extra_var`). Weighting by absolute
  excess error would undo that. The diffuse relative form below did
  not work either; a targeted form (thresholded) is untested.
- **Existing classes stay as reporting labels only,** so per-group
  trends and pre-registered gates stay comparable across iterations.

Open item before the regression variants are usable: a per-group
noise-calibration check from paired replicates (corpus telemetry
`top_pair` / `pair_diffs`). It settles whether the picker cost is real
and whether the noise model also needs a learned, budget-stratified
form.

**Convention targets under the variants** (iteration 3, t0/t1 defender
leads): called-suit target mass 0.991 (t0) / 0.985–0.986 (t1) and
defender trump-lead mass 0.0173–0.0174 (t0) / 0.0533–0.0540 (t1) under
class, head and isotonic alike. γ at t0 leads is 0.375 / 0.402 / 0.433.
The installed conventions' targets do not move. Picker groups: γ rises
by 0.03–0.06, and mean target KL vs production is 0.002–0.013.
Watch item: where a convention is *not yet* installed and the model is
amplifying it against noisy search (§7.5), raising γ weakens that
push.

**Relative residual weighting, tested and rejected in this form**
(`oversample_crossfit reweight`). The weight per training row is
w = clip(σ̂²_row / σ²_class, 1, cap), with σ̂² from the isotonic
variance regression. It is leakage-free: for each outer fold, the
regression and its calibration are refit on the training folds only.
Weights act by stochastic duplication, with epochs scaled to the
baseline's optimizer steps.

| | baseline | cap 4 | cap 10 | oversample k = 20 (§9.6) | search |
|---|---|---|---|---|---|
| global out-of-fold wMSE | 4.3615e-04 | 4.3570e-04 | 4.4552e-04 (+2.1%) | 4.3643e-04 | |
| over-trump cell | −0.32 | −0.42 | −0.39 | −0.50 | −0.55 |
| … where the policy over-trumps | +0.13 | +0.09 | +0.05 | −0.03 | −0.17 |
| t0 defender trump lead | −0.48 | −0.39 | −0.38 | | −0.45 |
| t0 called-suit lead | +0.31 | +0.29 | +0.32 | | +0.39 |
| control: other suit, picker behind | −0.17 | −0.24 | −0.19 | −0.23 | −0.22 |

About 47% of rows get w > 1 (the upper half of each class's error
distribution), so the weighting is diffuse. The cell gets about half the
targeted oversampling's correction, and its hard subset stays
wrong-signed. Cap 10 worsens most follow classes by 5–12%. Cap 4 is
neutral globally but reshuffles classes: t0 defender lead −11% wMSE,
follows +1–4%. It also moves the t0 trump-lead *mean* off search, while
improving its correlation (0.41 → 0.48). Not carried into the ablation.

### 9.8 Status and next steps

- **Shipped (operator, 2026-10-16):** `policy_iteration
  --model-variance {class, head, one-sided-head}` (`6557172`; default
  `class` = the recipe through arm V; non-class modes require the
  cross-fit's out-of-fold head) and the orchestrator knob
  `policy_iteration.model_variance` (`f7a1004`, unset by default).
- **Ablation arm, launched 2026-10-16 16:56**
  (`runs/202609_recall_rc/iter3_arm_onesided.sh`, output
  `pi/iter3_ablations/onesided_head/`). It re-targets iteration 3 with
  `--model-variance one-sided-head` on iteration 3's own fit artifacts
  (copied, not refit), then re-distills from θ₂ with the orchestrator's
  stage flags and the same seed, so the targets are the only difference.
  Then the cert vs θ₂, a paired duplicate h2h vs `distill_epoch10`
  (8,000 deals/mode), and the 4,000-seed incidence scan. Residual
  weighting is not included (§9.7). The iteration-3 bidding phase is
  paused for it (SIGSTOP, 16:48, bidding episode ~55,700; PIDs in
  `runs/202609_recall_rc/paused_pids.txt`). Bands:
  - **Primary 1 (amended 2026-10-16, before any ablation result was
    read): paired frozen-state read.** Each of the 359 §9.2 decision
    nodes is replayed on its identical deterministic trajectory (the
    iteration-3 candidate's /analyze playout). Each candidate rebuilds
    its own recurrent memory over the same prefix and reports its
    policy at the node. Statistic: the mean per-node difference in
    over-trump probability (sum over over-trumping cards), ablation
    minus `distill_epoch10`. Pass if it is negative with a one-sided
    95% bootstrap CI (resampled by seed) excluding zero. Reported
    alongside: argmax over-trump rate per candidate on the same nodes
    (McNemar), and the clear-blunder (has a fail, not AD/10D) and
    schmear subsets.
    *Why amended:* the original criterion, the fresh-deal scan rate
    significantly below 30.9% (two-proportion test), needs a drop of
    ~5.6 points at n ≈ 360 per arm (new rate ≲ 25.3%). §9.5/§9.7 predict
    a target shift of only ~0.02 in the cell, so a realistic one-step
    drop is a few points, and that test would fail for lack of power,
    not of effect. Pairing on identical states removes the deal
    variance.
  - **Secondary: fresh-deal incidence** on the 4,000-seed scan vs
    30.9% ± 2.4 (two-proportion), reported with its CI; not a gate.
  - **Primary 2:** EV equivalence by TOST on the paired duplicate h2h
    vs `distill_epoch10` (8,000 deals/mode). **Margin ±0.0075 score
    per hand** (operator, 2026-10-16, set before the h2h was read).
    Pass if the 90% CI of the edge lies inside (−0.0075, +0.0075),
    i.e. both one-sided tests reject at α = 0.05. With the expected SE
    ≈ 0.0027, that holds for any point estimate within about ±0.003.
  - **Secondary:** the §8 read; the t0 trump-lead and called-suit-lead
    probes.
  - **Expectation:** a modest incidence drop, not a collapse (§9.5
    ceiling).
- The iteration-3 bidding phase writes no checkpoints. If the ablation
  needs exclusive CPU, pause the orchestrator tree (SIGSTOP / SIGCONT on
  resolved PIDs) rather than kill it, and only on operator command.

**Ablation result (2026-10-16 20:10): both primaries pass; the effect is
small.**

| band | read | verdict |
|---|---|---|
| Primary 1, paired frozen read (359 nodes, 321 seeds) | p(over) 0.3036 → 0.2973, Δ −0.0063, one-sided 95% upper −0.0034 | PASS |
| — clear-blunder subset (336) | 0.2241 → 0.2194, Δ −0.0047, upper −0.0018 | PASS |
| — schmear subset (176) | 0.2703 → 0.2624, Δ −0.0079, upper −0.0033 | PASS |
| — argmax over-trump, same nodes | 30.9% → 29.5% (8 stopped, 3 started; McNemar one-sided p 0.11) | n.s. |
| Primary 2, TOST vs `distill_epoch10` (16,000 deals) | edge +0.0022 ± 0.0016 (called +0.0013, JD +0.0030); 90% CI (−0.0004, +0.0048) inside ±0.0075 | PASS (equivalent) |
| Secondary, fresh-deal incidence (4,000 seeds) | 108/357 = 30.3% ± 2.4 vs 30.9% ± 2.4 | n.s., as powered (§9.8 amendment) |
| Secondary, cert vs θ₂ | PASS; h2h vs θ₂ +0.0059 ± 0.0025 (iteration 3: +0.0033 ± 0.0025) | — |
| Secondary, probes (4-seed means, arm vs iteration 3) | t0 trump lead 0.57% vs 0.99%; called-suit lead 94.6% vs 92.6%; partner trump lead 98.7% vs 97.9%; pick 33.4% vs 33.1%; leaster 9.1% vs 9.3% | no convention damage |

The §8 read was not rerun for the arm. The h2h vs iteration 3 shows the
arm deviating on 5.6% of deals, against 16% for iteration 3 vs θ₂: it
is a small perturbation of iteration 3, not a different policy.

**Lineage on the same frozen nodes.** The same read across the whole
policy-iteration phase (probe = each checkpoint, trajectory = iteration
3's) puts the arm in scale:

| checkpoint | p(over) | clear blunder | schmear | argmax over |
|---|---|---|---|---|
| θ₀ (league 8M) | 0.395 | 0.315 | 0.376 | 40.9% |
| iteration 1 distill = θ₁ | 0.377 | 0.295 | 0.352 | 40.1% |
| iteration 2 distill = θ₂ | 0.338 | 0.263 | 0.297 | 34.5% |
| iteration 3 distill | 0.304 | 0.224 | 0.270 | 30.9% |
| one-sided head arm | 0.297 | 0.219 | 0.262 | 29.5% |

Steps (all decisions, Δp with one-sided 95% upper; argmax stopped/started):
θ₀ → it1 −0.019 (−0.009; 15/12), it1 → it2 −0.039 (−0.030; 21/1,
p < 0.001), it2 → it3 −0.034 (−0.025; 20/7, p 0.010), it3 → arm −0.006
(−0.003; 8/3). θ₀ → it3 totals −0.092 (47/11, p < 0.001). The bidding
phase leaves play unchanged: θ₁ and θ₂ read identically to their
distills. Caveat: the states are those iteration 3's play reaches;
earlier checkpoints are forced onto them, so the read is paired on
states, not on how often each policy reaches them.

**Reading.** Policy iteration is already removing this leak, about 3–4
points of probability per iteration since iteration 2, and the
one-sided head adds about a sixth of one iteration's step on top. That
matches §9.5/§9.7: the cell's search labels prefer not over-trumping
only weakly, so raising its γ from 0.41 to 0.56 moves the target, and
the policy, a little. The arm's case for adoption therefore rests on
it being the more principled variance rule at equivalent EV (both
primaries pass, convention probes no worse), not on fixing this leak.
Adoption is the operator's call.

**Adopted (operator, 2026-10-16 20:28).** The orchestrator default is
now `model_variance: one-sided-head` (`8e59391`; the `policy_iteration`
module default stays `class`, so earlier runs reproduce). The arm
replaced iteration 3's candidate in place
(`runs/202609_recall_rc/iter3_swap_onesided.sh`, log beside it): the
paused orchestrator tree was stopped, which discarded the class
candidate's partial bidding phase (~55,700 of 200,000 episodes, no
checkpoint); iteration 3's class-variance target, distill epochs, cert
and the partial bidding run were archived to
`pi/iter3_ablations/class_baseline/`; the arm's were copied into
`pi/iter3/` with the two recorded candidate paths (`distill_best.json`,
`cert.json`) rewritten. The arm's cert was run against the same θ₂ with
the same flags, so it stands as iteration 3's cert (h2h vs θ₂ +0.0059 ±
0.0025). The fit artifacts are shared by both arms and stayed. The
relaunched orchestrator skipped iteration 3's corpus, distill and cert
and started a fresh bidding phase from the arm's candidate; iteration 4
runs on the new default. Paths in §9 that name the class candidate now
point to the archive.

**Reproducing §9.** Every table comes from a committed module, run as
`uv run python -m sheepshead.analysis.<module> …` from the repo root.
P = `runs/202609_recall_rc/overtrump_probe` (where the cached
artifacts live; the session's first-pass scripts are kept there too,
but nothing needs them).

| table | commands |
|---|---|
| §9.2 incidence | `scan_called_suit_overtrumps --model runs/202609_recall_rc/pi/iter3_ablations/class_baseline/distill_epoch10.pt --num-seeds 4000 --out P/scan_iter3.jsonl`; then `scan_called_suit_overtrumps --summary P/scan_iter3.jsonl` |
| §9.2 corpus rates, §9.3 value | `called_suit_overtrump extract --out P/rows.pkl`; then `called_suit_overtrump value --rows P/rows.pkl` |
| §9.3 win probability | `q_outcome_calibration extract --out P/calib.pkl`; then `q_outcome_calibration report --calib P/calib.pkl --rows P/rows.pkl` |
| §9.4 target movement | `called_suit_overtrump targets --rows P/rows.pkl` |
| §9.4 Fay-Herriot terms | `blend_variance_probe fh-terms --rows P/rows.pkl` |
| §9.4 look-alike | `called_suit_overtrump extract --all-fail --out P/rows_all_fail.pkl`; then `called_suit_overtrump bleed --rows P/rows_all_fail.pkl` |
| §9.4 probes | `feature_probes --iter-dir runs/202609_recall_rc/pi/iter3` |
| §9.5 | `budget_rows generate --scan P/scan_iter3.jsonl --ckpt runs/202609_recall_rc/pi/iter3_ablations/class_baseline/distill_epoch10.pt --budgets 256 512 --out-dir P`; then `budget_rows targets --scan P/scan_iter3.jsonl --rows-dir P --budgets 256 512` |
| §9.6 | `oversample_crossfit oversample --k 1 5 20 --rows P/rows.pkl --controls P/rows_all_fail.pkl --cache-dir P` |
| §9.7 variants | `blend_variance_probe variants --rows P/rows.pkl --regression-cache P/variance_regression_oof.pt --json P/variance_regression_diag.json` |
| §9.7 conventions | `blend_variance_probe conventions --regression-cache P/variance_regression_oof.pt` |
| §9.7 relative weighting | `oversample_crossfit reweight --caps 4 10 --rows P/rows.pkl --controls P/rows_all_fail.pkl --cache-dir P` |
| §9.8 frozen read | `frozen_node_policy_read read --scan P/scan_iter3.jsonl --driver runs/202609_recall_rc/pi/iter3_ablations/class_baseline/distill_epoch10.pt --probe NAME=CKPT … --out A/frozen_read_lineage.json`; then `frozen_node_policy_read overtrump --read A/frozen_read_lineage.json --base NAME --arm NAME` (A = `runs/202609_recall_rc/pi/iter3_ablations/onesided_head`; probes θ₀ `league/checkpoints/checkpoint_8000000.pt`, `pi/iter{1,2}/distill_epoch10.pt`, `pi/iter{1,2}/bidding/final.pt`, the archived class candidate, A/`distill_epoch10.pt`) |

The fits and regressions reuse cached results in P when present; delete
them to recompute.

### 9.9 Iteration 4 (2026-10-19): the leak on θ₄

Iteration 4 is the first full iteration under the one-sided head default
(corpus from θ₃, target and distill under the rule, bidding phase on the
candidate). Both §9.8 instruments were rerun on θ₄ =
`pi/iter4/bidding/final.pt` the evening it landed, during the bidding
cert and panels (`runs/202609_recall_rc/overtrump_probe/iter4/`).

**Fresh-deal incidence (§9.2 scan, 4,000 seeds from seed 0).**

| checkpoint | standard called games | decisions | over-trumped | p(over) | clear blunder | schmear |
|---|---|---|---|---|---|---|
| iteration 3 class candidate | 3,022 | 359 | 111 = 30.9% ± 2.4 | 0.30 | 22.6% | 29.0% ± 3.4 |
| one-sided head arm (= θ₃ play) | 3,030 | — | 108/357 = 30.3% ± 2.4 | — | — | — |
| θ₄ | 3,123 | 357 | 106 = 29.7% ± 2.4 | 0.30 | 20.3% | 22.7% ± 3.2 |

Not significant, as powered (§9.8 amendment): the scan can only see a
drop of ~5.6 points. The by-trick and by-position shape is unchanged
(trick 0 17.9%, trick 4 40.9%; 0 players after 20.1%, 2 after 46.1%);
the schmear subset fell from 29.0% to 22.7% (± 3.3 each).

**Paired frozen-state read (§9.8 primary, the same 359 nodes on the
iteration-3 class candidate's trajectory).**

| checkpoint | p(over) | clear blunder | schmear | argmax over |
|---|---|---|---|---|
| θ₀ (league 8M) | 0.395 | 0.315 | 0.376 | 40.9% |
| θ₁ | 0.377 | 0.295 | 0.352 | 40.1% |
| θ₂ | 0.338 | 0.263 | 0.297 | 34.5% |
| iteration 3 class candidate | 0.304 | 0.224 | 0.270 | 30.9% |
| θ₃ (one-sided head arm) | 0.297 | 0.219 | 0.262 | 29.5% |
| iteration 4 distill (d4) | 0.289 | 0.214 | 0.247 | 28.4% |
| θ₄ | 0.289 | 0.214 | 0.247 | 28.4% |

θ₃ → θ₄: Δp −0.0086 (one-sided 95% upper −0.0027) → PASS on all
decisions; clear-blunder subset −0.0058 (upper +0.0001, not
significant); schmear subset −0.0152 (upper −0.0072) → PASS. Argmax
over-trumps 29.5% → 28.4% (10 stopped, 6 started; McNemar one-sided
p 0.23). d4 → θ₄ is identical on every node (Δ 0, 0 stopped, 0 started):
the bidding phase leaves play unchanged, as for θ₁ and θ₂.

**Reading.** The leak keeps falling but the step is a quarter of the
iteration-2 and iteration-3 steps (−0.039, −0.034) and only slightly
larger than the one-sided head's own effect at iteration 3 (−0.006).
That is the same deceleration the cert gains show (+0.0198 → +0.0100 →
+0.0056 → +0.0046) and the fixed-η dose (KL p50 0.0427 → 0.0180): the
search labels prefer not over-trumping only weakly in this cell (§9.5),
so as the policy approaches what the labels support, each iteration
buys less. θ₀ → θ₄ totals −0.106 on p(over) and 40.9% → 28.4% on the
argmax rate. Against the §9.8 conclusion nothing changes: policy
iteration is removing the leak; the variance rule is a small, principled
assist; the remaining ~28% is where the labels themselves are close to
indifferent. A further step needs a better label (oracle referee or a
larger search budget at these nodes, §9.5), not more weight on the
model.

Reproduction: `runs/202609_recall_rc/overtrump_probe/iter4/probe_theta4.sh`
(the two commands are `scan_called_suit_overtrumps --model
runs/202609_recall_rc/pi/iter4/bidding/final.pt --num-seeds 4000 --out
…/scan_theta4.jsonl` then `--summary`, and `frozen_node_policy_read read
--scan runs/202609_recall_rc/overtrump_probe/scan_iter3.jsonl --driver
runs/202609_recall_rc/pi/iter3_ablations/class_baseline/distill_epoch10.pt
--probe theta3=runs/202609_recall_rc/pi/iter3/bidding/final.pt --probe
d4=runs/202609_recall_rc/pi/iter4/distill_epoch10.pt --probe
theta4=runs/202609_recall_rc/pi/iter4/bidding/final.pt --out
…/frozen_read_theta4.json`, then `overtrump --read … --base theta3 --arm
theta4`). Outputs: `scan_theta4_summary.txt`,
`frozen_overtrump_theta4.txt`, `probe.log`.

### 9.10 Hindsight referee (2026-10-20): the labels are right on average, weak where the policy still over-trumps

An estimator outside the search family (§9.8 "next steps"; CE_Teacher
§22.6 "oracle referee"), `sheepshead/analysis/overtrump_referee.py`.
Each §9.2 node is replayed on its frozen trajectory, the state and every
seat's recurrent memory are snapshotted, each legal card is played, and
θ₄ plays all five seats to the end of the actual deal. The node seat's
realized hand score is the card's value (score units, the h2h scale):
one deterministic playout per card, and the mean of 8 stochastic
playouts with common random numbers across a node's cards. Nobody in the
continuation sees hidden cards, so the read has no determinization
(strategy-fusion) or oracle-leaf term; its noise is the hidden-card draw
(per-node SD ≈ 1.3 vs 0.83 for the search label, whose replicate noise
is ≈ 0.17), so it means something only in aggregates: ±0.15 at 50 nodes,
±0.25 at 15. It values a card under θ₄'s own follow-up play, where the
search values it under per-world optimal follow-up.

Two node sets: the 359 iteration-3 nodes on the class candidate's
trajectory (joins the §9.5 search labels at 256/512 iterations; 272
joined) and θ₄'s own 357 scan nodes (§9.9) as a replication.

**The cell as a whole: referee and search agree; the search is, if
anything, slightly too hard on the over-trump.**

| over-trump − policy's alternative (score) | n | search 256 | search 512 | referee (stoch) |
|---|---|---|---|---|
| all joined nodes | 256 | −0.642 ± 0.052 | −0.672 ± 0.053 | −0.523 ± 0.082 |
| trick 0 | 89 | −0.785 ± 0.063 | −0.800 ± 0.060 | −0.822 ± 0.117 |
| where search endorses the over-trump | 44 | +0.741 | — | +0.739 ± 0.208 |
| where search rejects it | 228 | −0.882 | — | −0.727 ± 0.075 |
| search − referee, joined | 272 | | | −0.130 ± 0.069 |

The 44 search endorsements are real exceptions (+0.74 at 3.5 SE in
hindsight). Per-node agreement is only ~50% (one deal is one draw), the
means are not in doubt. There is no cell-wide label artifact to correct.

**θ₄'s remaining over-trumps are net neutral; the trick-0 ones are the
blunders.** Replicated on both node sets:

| referee over − alt, stochastic | iteration-3 nodes | θ₄'s own nodes |
|---|---|---|
| nodes where θ₄ over-trumps | +0.148 ± 0.120 (102) | −0.032 ± 0.122 (106) |
| nodes where θ₄ declines | −0.621 ± 0.072 (257) | −0.720 ± 0.073 (251) |
| θ₄'s over-trumps at trick 0 | −0.728 ± 0.237 (17) | −0.651 ± 0.221 (19) |
| θ₄'s over-trumps at tricks 2–4 | +0.44 to +0.57 (63) | +0.314 ± 0.154 (64) |

The policy has sorted the cell: it over-trumps where hindsight says it
pays (late, two players still to act) and declines where it costs
(trick 0, last to act −0.7). Its mass-weighted expected value from
over-trumping is +0.04 per decision. The residual blunder set is the
trick-0 over-trumps: ~17% of trick-0 decisions, ~5% of all, ≈ 0.6 per
100 called games, costing ≈ 0.7 each (pooled over both sets ≈ −0.69 ±
0.16). Mostly a defender holding a 10/A fail (11/17) over-trumping a
partner-defender's small trump with a jack, queen or AD instead of
schmearing; 7/17 last to act.

**At those nodes the label is the limiting factor.** On the 14 joined
residual trick-0 over-trumps the search label is −0.133 ± 0.128 at 256
and −0.038 ± 0.125 at 512 (endorsing the over-trump on 2/16), the
referee −0.580 ± 0.241: a gap of ≈ 0.45 ± 0.27 (1.7–2.2 SE, suggestive).
By the policy's over-trump mass: under 0.1 the two agree (search −1.00,
referee −0.81); in 0.5–0.9 the search says −0.21 and the referee −0.48
(trick 0: −0.28 vs −0.98, n 8).

**Mechanism: OPEN, and not prior domination.** The first reading (a
prior-dominated 256-iteration search under-exploring the alternative
where θ_k is confident) is refuted by the budget rows: 512 iterations
leave the label unchanged or weaker at exactly these nodes (corr 0.92
with 256 across all joined nodes). What remains is budget-invariant and
structural: determinized strategy fusion (per world the search knows
where the schmear can later be cashed), oracle-leaf optimism, or the
belief sampler conditioned on θ_k's style. The correlation of label
weakness with the policy's confidence then runs the other way: four
iterations of these labels taught the policy to be confident exactly
where the label calls the over-trump cheap. Discriminating tests, not
run: the committee with a uniform root prior at these nodes (isolates
any residual prior effect); a committee without oracle leaves (isolates
the leaf term).

**Standing.** No recipe change. The next measurement is more residual
trick-0 nodes, not more playouts: a 20,000-seed scan gives ≈ 85 of them
and brings the subset to ≈ ±0.13. If the gap holds, the principled lever
is a referee-calibrated class-level label bias (estimated on a node
sample per class, applied before the blend like the class-pooled noise
model), confined to the fixed-recipe validation run; a non-recipe
alternative is a less confident root prior for the committee at trick-0
follows, which attacks the mechanism if it is prior-related at all.

Reproduction: `runs/202609_recall_rc/overtrump_probe/referee/run_referee.sh`
(`overtrump_referee run --scan P/scan_iter3.jsonl --driver
runs/202609_recall_rc/pi/iter3_ablations/class_baseline/distill_epoch10.pt
--continuation runs/202609_recall_rc/pi/iter4/bidding/final.pt
--budget-rows P/budget_corpus_256.pt --samples 8 --out R/referee_theta4.json`
then `summary --read R/referee_theta4.json`; the replication uses
`P/iter4/scan_theta4.jsonl` with θ₄ as driver). Outputs
`referee_theta4.json`, `summary_theta4.txt`, `referee_theta4_ownnodes.json`,
`summary_theta4_ownnodes.txt`. The 512 comparison and the confidence
bands were read from the JSON with `search_q_by_node` on
`P/budget_corpus_512.pt`.

### 9.11 Trick 0 at scale (2026-10-20): 20,000 fresh seeds

`runs/202609_recall_rc/overtrump_probe/t0_20k/run.sh`: θ₄ scanned on
seeds 4000–23999 (disjoint from every earlier scan), the 463 trick-0
decisions labelled by the 256-iteration committee (`budget_rows
generate`, θ₄ acting and searching, iteration 4's corpus config) and
refereed (θ₄ driver and continuation, 8 CRN samples).

**Correction to the incidence levels.** The 0–3999 seed block that every
§9.2/§9.8/§9.9 scan used is unrepresentative. The scan is deterministic
(seeds 0–599 re-scan identically), so this is the deal sample, not the
tool:

| θ₄, seed block | all decisions | trick 0 |
|---|---|---|
| 0–3999 (§9.9) | 29.7% (106/357) | 17.9% (19/106) |
| 4000–7999 | 30.2% | 30.3% |
| 8000–11999 | 33.2% | 31.1% |
| 12000–15999 | 37.9% | 33.7% |
| 16000–19999 | 37.7% | 33.3% |
| 20000–23999 | 36.0% | 34.4% |
| 4000–23999 pooled | 35.2% ± 1.2 (592/1684) | 32.6% ± 2.2 (151/463) |

The paired lineage reads (§9.8, §9.9) compare checkpoints on identical
nodes and are unaffected; the absolute rates quoted there understate the
population rate, which for θ₄ is ≈ 35% of decisions and ≈ 33% at trick 0
(≈ 1 trick-0 over-trump per 100 called games).

**Referee at trick 0, 463 nodes (420 with search labels).**

| over-trump − policy's alternative (score) | n | referee (stoch) | referee (det) | search 256 | search − referee |
|---|---|---|---|---|---|
| all trick-0 nodes | 463 | −0.722 ± 0.053 | −0.680 ± 0.082 | −0.654 ± 0.029 | +0.114 ± 0.053 |
| θ₄ declines | 312 | −0.927 ± 0.066 | −0.907 ± 0.099 | −0.886 ± 0.028 | +0.078 ± 0.066 |
| θ₄ over-trumps | 151 | −0.299 ± 0.080 [90% −0.43, −0.17] | −0.212 ± 0.140 | −0.113 ± 0.039 | +0.198 ± 0.086 |
| … clear (not AD/10D) | 74 | −0.454 ± 0.111 | −0.473 ± 0.203 | −0.234 ± 0.058 | +0.228 ± 0.124 |
| … holding a schmear | 99 | −0.312 ± 0.104 | −0.343 ± 0.186 | −0.111 ± 0.050 | +0.203 ± 0.108 |
| … 0 / 1 / 2 players after | 72 / 50 / 29 | −0.36 / −0.46 / +0.14 | | −0.13 / −0.13 / −0.01 | |
| policy over-trump mass ≥ 0.9 | 102 | −0.212 ± 0.092 | | +0.015 ± 0.045 | +0.244 ± 0.097 |
| search endorses the over-trump | 51 | +0.037 ± 0.113 | | | |

Readings.

- **θ₄'s residual trick-0 over-trumps are blunders, at ≈ 0.30 per
  hand** (90% CI −0.43 to −0.17), smaller than the −0.65/−0.73 the two
  small samples suggested but no longer in doubt. Where another player
  still follows (2 after) they are neutral; last or second-to-last to
  act they cost 0.36–0.46. Over the 15,524 called games that is ≈ 0.003
  score per called game: small in EV, visible at the table.
- **The search label under-penalizes exactly there.** On the 126 joined
  residual nodes the label is −0.113 ± 0.039 against the referee's
  −0.299: a gap of +0.20 ± 0.09 (2.3 SE); where the policy puts ≥ 0.9 on
  the over-trump the label is +0.015 (indifferent) against −0.21 (gap
  +0.24 ± 0.10, 2.5 SE). On the 51 trick-0 nodes where the search ranks
  the over-trump first the referee reads +0.04 ± 0.11: unlike the
  all-trick endorsements of §9.10 (+0.74), the trick-0 endorsements are
  not real exceptions. Across all 420 trick-0 nodes the two agree to
  +0.11 ± 0.05, so the bias is concentrated where the policy already
  over-trumps.
- **The student also under-realizes the label.** θ₄ over-trumps on 80 of
  the 126 joined residual nodes where the label rejects it (mildly). A
  −0.11 label is a ≈ 1-nat tilt against a prior near 0.9; one
  η-bounded step moves such a row little, and §9.5's one-iteration
  ceiling applies.

**Mechanism: still open.** This set has no 512 rows; §9.10's finding
that the budget leaves the label unchanged at these nodes stands, so the
≈ 0.2 under-penalty is budget-invariant and structural (determinized
strategy fusion, oracle leaves, or θ_k-conditioned beliefs). The
discriminating variants (uniform root prior; no oracle leaves) remain
unbuilt.

**Standing.** Unchanged from §9.10: no recipe change mid-phase. The
measured quantities a fix would use are now in hand: a trick-0
defender-follow class bias of ≈ +0.2 score (search minus truth) at the
over-trump, concentrated where θ_k is confident, and a residual blunder
cost of ≈ 0.3 per occurrence at ≈ 1 per 100 called games.

Outputs: `t0_20k/scan_theta4_20k.jsonl`, `scan_t0.jsonl`,
`budget_corpus_256.pt`, `referee_t0.json`, `summary_t0.txt`, `run.log`.
The joined table above was read from `referee_t0.json` with
`overtrump_referee.search_q_by_node` on `budget_corpus_256.pt`.

**Addendum (2026-10-20): the AD/10D split.** The headline rate counts
every higher trump, point trumps included; the clear-blunder subset
excludes AD/10D (§9.3: their points land on the defence's trick either
way). θ₄'s 151 residual trick-0 over-trumps by the card played, referee
value of the played card against the policy's best alternative:

| θ₄'s trick-0 over-trumps | n | referee (stoch) |
|---|---|---|
| with a jack, queen or KD | 74 | −0.454 ± 0.111 |
| with AD or 10D | 77 | −0.149 ± 0.113 |
| … AD/10D when a 10/A fail could be schmeared instead | 39 | −0.349 ± 0.167 |
| … AD/10D when the alternative was a low card | 38 | +0.056 ± 0.147 |

Covering a teammate's trump with the AD when the alternative is a low
fail is break-even; doing it instead of schmearing a ten or ace costs
≈ 0.35; spending a jack or queen on a trick the defence already owns
costs ≈ 0.45. About half the trick-0 incidence is the genuinely bad
half. The teammate's winning card was the KD, 10D or AD in 105 of the
151 cases, so the typical residual case covers a king or point trump,
not a 7D.

### 9.12 Where this sits, and what was deferred (2026-10-20)

A record of the discussion after §9.11, for whoever picks the leak up
next. Nothing below was built; the operator deferred all of it.

**Candidate fixes, from most to least local.**

1. *One-cell label correction.* Subtract the measured search-minus-
   referee gap (+0.20 ± 0.09 score, §9.11) from the over-trump cards'
   labels at trick-0 defender follows, before the fit and the blend;
   add its SE² to the label noise. Rows are tagged from the stored state
   by the §9.2 node definition. Expected target effect: the residual
   nodes' gap goes from −0.11 to −0.31 score (≈ −1.0 → −2.7 nats at
   η 0.0097), enough to flip a 0.9 prior's top card; distillation
   realizes part per iteration. Runnable on iteration 4's existing
   corpus as an arm (~4 h) with the §9.8 protocol. Rejected as too
   idiosyncratic: one cell, one hand-written action feature.
2. *Class-level benchmarking against the referee (generic form).* Per
   node class, estimate the search label's bias at θ_k's top card
   relative to its best alternative on a random node sample by hindsight
   continuation, shrink across classes toward zero, apply before fit and
   blend. No hand-written rule; the §7.4 residual diagnostics already
   have this structure, and the trick-0 cell is exactly "θ_k confident,
   search mildly agrees, hindsight disagrees". Cost per iteration a few
   hours of playouts; most entries expected ≈ 0, which is itself an
   audit. The version worth pre-registering for a fixed-recipe
   validation run.
3. *Replace the search estimator.* Belief-based search (public-belief-
   state methods in the DeepStack/ReBeL line) or recursive imperfect-
   information search (Furtak & Buro 2013) avoid strategy fusion by
   construction. The operator's philosophical preference, deferred as a
   large change; trade-offs (compute per node, belief-model
   requirements, how the committee's replicate/noise machinery carries
   over) not yet studied.

**Literature placement.** The bias is the textbook one: strategy fusion
and non-locality in determinized search (Frank & Basin 1998; Long,
Sturtevant, Buro & Furtak 2010 on when PIMC works regardless); ISMCTS
(Cowling, Powley & Whitehouse 2012) reduces but does not remove it.
Bridge and Skat programs have shipped heuristic patches for fusion-
induced mistakes (the precedent for option 1, and why it is inelegant).
The referee is Tesauro & Galperin's (1996) rollout policy improvement
used as a check rather than as the label. Correcting a biased
low-variance estimator with an unbiased high-variance one is standard:
doubly-robust off-policy evaluation (Jiang & Li 2016; Thomas & Brunskill
2016), outcome-mixed value targets in AlphaZero-family training, and
*benchmarking* in small-area estimation (Pfeffermann 2013), which is the
step the Fay-Herriot literature prescribes after the shrinkage this
recipe already does. Option 2 is therefore the framework's own next
step; option 1 is the patch; option 3 is the structural fix.

**Deployment temperature and the leak.** The operator intends a
stochastic deployment with a tuned temperature. Re-tempering θ₄'s
policy at the 463 trick-0 nodes (§9.11; p ∝ p_greedy^(1/τ), referee
values from the same read):

| τ | expected trick-0 over-trump rate | expected hindsight value vs greedy (score) |
|---|---|---|
| 0 (greedy) | 32.6% | 0 |
| 0.5 | 32.2% | +0.000 ± 0.008 |
| 1.0 | 31.5% | −0.003 ± 0.011 |
| 1.5 | 30.9% | −0.013 ± 0.013 |
| 2.0 | 30.4% | −0.027 ± 0.014 |

The policy is sharp at these nodes (mass near 0.9 or near 0.05), so
temperature barely moves the incidence (−1 point at τ = 1, −2 at τ = 2)
and does not change its value; across all tricks on the §9.10 nodes the
rate is 28.4% greedy vs 28.9% at τ = 1. Stochastic deployment neither
fixes nor worsens the leak. (Decision-level read on greedy-reached
states; a sampled policy also reaches different states, second order
here.)

### 9.13 Test arm: one-cell label correction (pre-registered 2026-10-20, before any read)

Operator's framing: a test of the impact of §9.12 option 1, **not a
candidate recipe** (too idiosyncratic to adopt). Built with
`sheepshead/analysis/overtrump_label_bias.py` on iteration 4's own
corpus and row table (θ₃ encoding), so the only difference from the
iteration-4 class candidate d4 = `pi/iter4/distill_epoch10.pt` is the
label shift.

**Arm.** At rows that are §9.2 decisions at trick 0 (tagged from the
stored observation: called-ace game, called suit led, a teammate
defender winning with a trump, actor void in the called suit holding a
higher trump and another legal play), every over-trump card's label is
shifted by −0.198 score = −0.0165 advantage units (the §9.11 search −
referee gap on θ₄'s residual nodes; applied uniformly, AD/10D included).
86 rows tagged, all `std|t0-defender-follow`, 163 labels shifted (QS 28,
JC 22, JS 20, QD 17, JD 16, QH 15, JH 13, QC 12, AD 8, KD 6, 10D 3, 9D 2,
8D 1). Label variance unchanged (the correction's SE² would add ≈ 25% to
those rows' noise; omitted). Then `policy_iteration all` with iteration
4's flags (trunk 6, lr 3e-5, λ_ret 10, head 4, cross-fit 5, one-sided
head) into `pi/iter4_ablations/t0_label_bias/`, and the iteration-4 cert
vs θ₃.

**Pre-registered reads** (`t0_label_bias_run.sh`, log `arm.log`):

- *Primary 1:* paired frozen-state read on the 463 trick-0 nodes of the
  20k scan (driver θ₄), probes d4 and arm: mean per-node Δ p(over-trump)
  arm − d4 negative with the one-sided 95% bootstrap upper (by seed)
  below zero. Expected size small: 86 corrected rows in 132,692, one
  η-bounded step, and §9.5's one-iteration ceiling; a few points of
  probability would already be a clear read.
- *Primary 2:* TOST equivalence of the paired duplicate h2h arm vs d4,
  8,000 deals/mode, margin ±0.0075 (90% CI inside).
- *Secondary:* all-trick frozen read on the 359 iteration-3 nodes (the
  leak at tricks 1–4 should not move); the cert vs θ₃ passes with
  convention probes within ~1 pp of d4's; the expected hindsight value
  of each policy at the 463 nodes from the §9.11 referee's per-card
  values (arm − d4 ≥ 0 if the shift is buying real value).

**Interpretation rule.** Both primaries pass → the label is the binding
constraint in this cell and a class-level benchmarking step (§9.12
option 2) is worth designing for the validation run. Primary 1 fails →
the student does not realize a −0.2 label shift in one step at this
prior; the ceiling is distillation, not the label. Primary 2 fails → the
shift costs EV elsewhere (the model generalizing it), which would argue
against any label-side fix.

**Result (2026-10-20 12:51): both primaries pass; the effect is small and
confined, as pre-registered.** Stage: fit best epoch and target dose
unchanged from iteration 4 (KL p50 0.0181 vs 0.0180); cert vs θ₃ PASS,
h2h +0.0047 ± 0.0023, play-only +0.0049 ± 0.0022 (d4: +0.0040 / +0.0046);
probes t0 trump lead 0.0–1.4%, called-suit lead 93–94%, partner trump
lead 98–99%, logit spread 5.0–5.2 (no convention damage).

| read | verdict |
|---|---|
| Primary 1, paired frozen read, 463 trick-0 nodes | p(over) d4 0.3152 → arm 0.3015, Δ −0.0137, one-sided 95% upper −0.0109 → PASS |
| Primary 2, TOST h2h arm vs d4 (16,000 deals) | −0.0017 ± 0.0015; 90% CI (−0.0042, +0.0008) inside ±0.0075 → PASS (equivalent) |
| Secondary, hindsight value at the 463 nodes (referee per-card values × policy) | arm − d4 +0.0079 ± 0.0021 score per decision (z +3.8); 58% of nodes improved |
| Secondary, all-trick frozen read, 359 iteration-3 nodes | Δ −0.0062 (upper −0.0041); trick 0 −0.0159 ± 0.0037, tricks 1–4 pooled −0.0023 ± 0.0011 |
| Secondary, cert vs θ₃ and probes | PASS, within 1 pp of d4 |

Where the probability moved, by d4's over-trump mass at the node: < 0.1
−0.001; 0.1–0.5 −0.059 ± 0.010; 0.5–0.9 −0.066 ± 0.011; ≥ 0.9 −0.007 ±
0.002 (102 nodes). Split by card: jack/queen/KD −0.0063, AD/10D −0.0074
(the shift was uniform).

**Reading, per the interpretation rule.** Both primaries pass, so the
label is a binding constraint in this cell and the shift buys real
hindsight value at equivalent EV. But one corrected step moves the
mid-confidence nodes and barely touches the ≥ 0.9 band where most
residual blunders live: the §9.5 one-iteration ceiling, now seen from
the label side. A label-side fix would therefore act over iterations,
not at once, and at ≈ 1.4 points of trick-0 probability per step it
would take several iterations to matter at the table. The faint
generalization to tricks 1–4 (−0.002) is in the harmless direction. The
arm is NOT adopted (operator: a one-cell hand-written correction is not
a recipe); its value is the measurement: a referee-calibrated label does
what §9.11 predicted, and the class-level benchmarking form (§9.12
option 2) is the version that would be worth carrying into a
fixed-recipe validation run.

Artifacts: `pi/iter4_ablations/t0_label_bias/` (`label_bias.json`,
`arm.log`, `cert.json`, `frozen_t0_overtrump.txt`,
`frozen_alltricks_overtrump.txt`, `h2h_vs_d4.json`); script
`pi/iter4_ablations/t0_label_bias_run.sh`. The corpus was paused
(SIGSTOP on the orchestrator tree) 09:45–12:55 for the arm's compute.

### 9.14 Mechanism, corrected (2026-10-20): no strategy-fusion path in this engine

The operator challenged the §9.10/§9.12 "strategy fusion" candidate
("per world the search knows where the schmear can later be cashed"),
and the challenge is right. In this engine (`sheepshead/ismcts.py`,
SO-ISMCTS):

- Tree nodes are statistics-only and keyed by the OBSERVER'S OWN ACTION
  SEQUENCE. The observer's in-tree strategy is therefore one strategy
  shared across every determinized world (coarser than its information
  set, if anything: it does not even key on what others played between
  its actions, beyond the legal set via availability-PUCT). Nothing in
  the tree can choose a different continuation per world, which is what
  strategy fusion requires.
- Non-observer seats act in-tree and in rollouts by sampling their own
  controller from their own observation in the world; they never see the
  determinization's hidden cards.
- The oracle critic at the leaf sees the full world, but a critic trained
  to predict realized returns under information-limited play is an
  estimate of E[return | full state], whose belief average is
  E[return | information set]: a clairvoyant *solver* at the leaf would be
  optimistic, a calibrated *return predictor* is not.

So the per-world-continuation story has nothing to run on and is
withdrawn. The budget-invariant under-penalty at the residual trick-0
nodes (§9.10–§9.11) still needs a structural account; the channels that
DO exist in this engine, each testable with the `budget_rows` machinery
and a config change:

1. *Oracle-leaf calibration.* With `d_rollout = 1` the leaf value is
   almost entirely the privileged critic's bootstrap one observer ply
   past the tree. Over-trump states are rare and the critic was fitted on
   past policies' returns; a critic lean of ≈ 0.2 score at "defender who
   just over-trumped a teammate" vs "defender who schmeared" would
   reproduce the gap exactly. Test: the same nodes with
   `leaf_evaluator="limited"`, and with terminal rollouts (`d_rollout`
   past the hand's end, no bootstrap at all). If the gap closes under
   terminal rollouts, this is it.
2. *In-tree follow-up credit the student never realizes.* The root label
   values a line under the search's IMPROVED observer decisions for up to
   `max_depth = 6` of its own plies, but only the root decision is
   distilled; the student's follow-up stays θ_k-like. A line whose payoff
   depends on the observer's own later play (over-trump, then lead well
   from the won trick) is credited for follow-up skill the deployed
   policy does not have; a passive line (schmear) is not. Test: the same
   nodes at `max_depth = 1` (root-only tree, follow-up by θ_k rollout).
   If the over-trump's advantage shrinks, the label is right for an
   improving agent and wrong for the agent we distill.
3. *Belief pool ignores play inference.* Worlds are weighted by the
   bidding likelihood only; opponents' play CHOICES (a teammate trumping
   with the KD rather than a low trump) carry no likelihood, so the
   posterior over the remaining trump is wrong wherever play is
   informative. Direction unclear; the test is a play-likelihood
   reweighting of the pool (the SIR variant Stage B set aside).
4. *Node coarsening and world-selection bias.* Own-action keys blend
   heterogeneous follow situations into one node's Q, and PUCT's visit
   allocation makes a child's visited-world mix differ from the belief
   posterior (the ISMCTS non-locality analogue). Real, but no reason to
   expect a sign at this cell.

Channels 1 and 2 are cheap (a config flag each on ≈ 460 nodes, an hour
or two) and predict the observed sign; they are the next tests if the
leak is picked up again. §9.12 option 3 ("a fusion-free search") is
correspondingly re-labelled: what a belief-space search would buy here
is a better posterior (channel 3) and a non-determinized value, not a
cure for a fusion this engine does not have.

### 9.15 Channel tests (2026-10-20): leaf counters and engine variants at the trick-0 nodes

Tooling: `budget_rows generate --variant label:key=value` (engine variants
of the committee: `leaf_evaluator`, `d_rollout`, `max_depth_play`, `iters`)
and two in-process counter probes
(`overtrump_probe/leaf_probe*.py`), all on the 463 trick-0 nodes of the
20k scan with θ₄ acting and searching. The corpus was paused for the
compute (13:00–15:30).

**How simulations end at these nodes (256 iterations, 30 nodes × 3
replicates).** 46% of simulations reach a terminal world (28% with the
tree at the observer's last decision, depth 6; 18% at depth 5 plus the
one rollout ply); 54% bootstrap on the oracle critic after the rollout
ply, mostly from tree depths 2–4 (one to three observer plies still to
play). Split by ROOT card the shares are identical: over-trump 46.1%
terminal, alternatives 45.9%, same depth profile. So about half the root
value flows through the critic, symmetrically across the cards; a
selective shallow bootstrap of the disfavored line is not happening.
*Correction (§9.17):* symmetric by card CLASS only. Split by the card
the policy plays vs the other legal cards, the shares differ ≈ 3×
(62–67% vs 18–27% terminal) at both over-trump and decline nodes; the
class split pools nodes where the over-trump is preferred with nodes
where it is not, and the two cancel.

**Engine variants, over-trump − policy's alternative (score), 358 nodes
with all labels and the referee:**

| | referee | base 256 (oracle leaf, d_rollout 1, depth 6) | limited critic | terminal rollouts (no bootstrap) | root-only tree (depth 1) |
|---|---|---|---|---|---|
| all trick-0 nodes | −0.846 ± 0.063 | −0.713 ± 0.030 | −0.664 ± 0.054 | −0.745 ± 0.031 | −0.617 ± 0.026 |
| θ₄ over-trumps (98) | −0.392 ± 0.108 | −0.172 ± 0.043 | **+0.683 ± 0.062** | −0.251 ± 0.042 | −0.227 ± 0.034 |
| θ₄ declines (260) | −1.018 ± 0.074 | −0.918 ± 0.030 | −1.171 ± 0.038 | −0.932 ± 0.032 | −0.764 ± 0.028 |
| policy mass ≥ 0.9 (61) | −0.385 ± 0.128 | −0.034 ± 0.051 | +0.987 ± 0.058 | −0.115 ± 0.051 | −0.132 ± 0.039 |
| label − referee at θ₄'s over-trumps | — | +0.220 ± 0.106 | +1.075 ± 0.116 | +0.141 ± 0.104 | +0.164 ± 0.105 |

Cross-node correlation with the referee is 0.35–0.38 for every variant;
variants correlate 0.70–0.76 with each other.

Readings.

- **The limited critic is a prior echo.** With observation-only leaves
  the label follows the policy's confidence almost one-for-one (+0.99
  where the policy puts ≥ 0.9 on the over-trump, −1.24 where it puts
  < 0.1), over-valuing the residual over-trumps by a full point. The
  2026-08-10 decision to bootstrap on the privileged critic is strongly
  vindicated; the oracle leaf is not "the" culprit relative to the
  alternative.
- **Removing the bootstrap entirely closes part of the gap, not all.**
  Terminal rollouts move the residual-node label from −0.17 to −0.25
  (−0.08 ± 0.06 vs base, 1.3 SE) and the gap to the referee from +0.22
  to +0.14 ± 0.10. Channel 1 (oracle-leaf calibration) accounts for
  perhaps a third of the lean, with wide error bars.
- **Removing in-tree follow-up credit moves it a little, the same way.**
  The root-only tree reads −0.23 at the residual nodes (−0.06 ± 0.05 vs
  base) while softening the whole cell (−0.62 vs −0.71): channel 2 is
  consistent in sign and small.
- **Neither variant reaches the referee.** After terminal rollouts the
  residual gap is +0.14 ± 0.10 (1.4 SE), and the referee's own read at
  these 98 nodes is −0.39 ± 0.11. We are at the resolution limit of the
  instruments: the remaining difference is within noise of zero, within
  noise of the baseline gap, and could sit in the referee's frame (θ₄'s
  follow-up vs the search's improved follow-up), in channel 4, or in
  sampling.

**Standing.** The structural account is now bounded rather than
identified: the oracle leaf and in-tree follow-up credit each plausibly
carry a few hundredths of a score of lean at the residual trick-0
nodes, with terminal rollouts the single most effective change tested
(and the most expensive per simulation). Nothing here changes the §9.13
decision; the measured levers are recorded for a future estimator
redesign. Artifacts: `t0_20k/budget_corpus_{limited,terminal,depth1}.pt`,
`leaf_probe_256.json`, `leaf_probe_by_action_256.json`, `variants.log`.

### 9.16 Standing summary of the over-trump leak (2026-10-20)

What is established, from §9.9–§9.15 together:

- **Incidence (θ₄, fresh seeds).** 35% of over-trump opportunities, 33%
  at trick 0; at the table, 6% of trumped called-suit tricks (5% with a
  jack/queen/KD), ≈ 4.3 per 100 called games all-AI, ≈ 3.4 with one
  human who never over-trumps; ≈ one per 30-hand session, a fifth of them
  at trick 0.
- **Value.** By hindsight continuation, θ₄'s over-trumps are net neutral
  across tricks (+0.15 / −0.03 on two node sets) and cost ≈ 0.30 ± 0.08
  per hand at trick 0 (0.45 with a jack/queen/KD, 0.15 with AD/10D);
  ≈ 0.003 score per called game in total.
- **Label.** The 256-iteration committee under-penalizes the over-trump
  by ≈ +0.20 ± 0.09 at exactly the residual trick-0 nodes, budget-
  invariant (512 unchanged). Where it endorses an over-trump at trick 0
  the referee reads neutral; across all trick-0 nodes label and referee
  agree to 0.1.
- **How the committee values these nodes.** 46% of simulations reach a
  terminal world, 54% bootstrap on the oracle critic with one to three
  observer plies left. Symmetric by card class but NOT by preference
  (§9.17): the card the policy plays is valued mostly by terminal
  outcomes under a deep, maxed tree (62–67%), the alternatives mostly by
  the critic (73–82%), so the label differences two estimators. The limited
  critic as leaf would echo the prior (+1.08 lean); the oracle leaf
  carries ≈ a third of the +0.20 lean (terminal rollouts: +0.14 ± 0.10
  left); in-tree follow-up credit a little; strategy fusion is not a
  channel in this engine (§9.14). The rest is unresolved at the
  instruments' resolution.
- **Student.** A −0.2 label shift at the cell (one-cell arm, §9.13) moves
  trick-0 over-trump probability by −0.014 per iteration, buying
  +0.008 score per decision at the nodes at equivalent EV; the ≥ 0.9
  prior band barely moves in one step.

**Is there a recipe that would reliably resolve these blunders? No.**
Nothing tested removes them in one iteration, and nothing principled
removes them at all yet:

1. *Policy iteration as is* keeps lowering the leak ≈ 1 point of
   probability per iteration at this stage (−0.106 total θ₀ → θ₄), slowing
   as the label's lean becomes the floor. (Amended 2026-10-22: the
   iteration-5 candidate took a −4.2 point step, §9.18; −0.148 total.)
2. *Label-side correction* (one-cell: measured, works, hand-tuned,
   rejected; class-level referee benchmarking: principled, unbuilt) adds
   ≈ 1.4 points per iteration and cannot dislodge a 0.9 prior in one step.
3. *More expensive search at these nodes*: terminal rollouts (no
   bootstrap) are the single most effective engine change tested and
   remove ≈ a third of the lean; more iterations alone do nothing. Also
   acts over iterations.
4. *A different search estimator* (belief-space, non-determinized) would
   change the posterior and the value estimate; after §9.14 its expected
   benefit here is a better posterior and a bootstrap-free value, not a
   fusion fix, and it is unproven on this cell. The operator's
   philosophical preference; a large project.
5. *A deployment guard* (veto over-trumping a teammate's winning trump at
   trick 0 when a fail discard exists) is the only option that removes
   the table-visible case immediately; it is a hand-written rule outside
   the derivability objective.

The honest expectation is that options 1–3 lower the trick-0 incidence
over several iterations rather than eliminating it, and that the
residual after the label lean is removed is governed by the one-step
distillation ceiling (§9.5) and the referee-frame question (does the
label value the follow-up the deployed policy actually plays).

### 9.17 Leaf census by preference (2026-10-20): the label differences two estimators

The operator noted that at 256 iterations the tree adds a node every
time a visited node is revisited, so most simulations run deep and many
end at a terminal even from trick 0. Confirmed, with a finer split than
§9.15's. `overtrump_probe/leaf_probe_preferred.py` reruns production
searches (`budget_rows` generator, iteration-4 manifest: `d_rollout 1`,
256 iterations × 3 replicates, oracle leaf, `max_depth` 6; θ₄ acting
and searching) on the first 12 trick-0 nodes of the 20k scan in each
set, and logs each simulation's leaf depth and type, split by whether
the root card is the one the policy plays.

| node set (12 nodes × 768 sims) | all: terminal | card the policy plays: terminal | other legal cards: terminal |
|---|---|---|---|
| θ₄ over-trumps (preferred = the over-trump) | 48.5% | **66.7%** (5,774 sims) | 17.9% (3,442) |
| θ₄ declines (preferred = a fail or low trump) | 46.6% | **62.2%** (5,168) | 26.7% (4,048) |

The rule is mechanical: a leaf at observer tree depth ≥ 5 is always
terminal (the one rollout ply finishes the hand), shallower leaves always
bootstrap on the oracle critic. The preferred card's subtree reaches
depth 5–6 on two thirds of its simulations; the alternatives' leaves sit
mostly at depths 2–4. Per-node terminal shares range 33–62%.

**Why §9.15's class split saw no asymmetry.** It classified root cards
as over-trump vs not and pooled decline and over-trump nodes. The
asymmetry follows the policy's prior (PUCT concentrates visits, hence
tree depth, on the preferred card), not the card class, so the two node
sets cancel.

**What the label is.** At the residual nodes the label is
Q(over-trump) − Q(alternative) where the first term is mostly realized
terminal score along a deep tree whose observer decisions are maxed over
noisy statistics, and the second is mostly an oracle-critic bootstrap
two to four tricks from the end, with the network's own play in between.
Any difference between the two estimators (critic lean, max-of-noise
optimism, follow-up skill the student does not realize) lands in the
label with a sign set by the policy's preference. This is the concrete
form of §9.14 channels 1 and 2, and it explains why the lean is
concentrated where θ_k is confident and why a larger budget does not
help (512 deepens the preferred subtree further; the label moved
−0.133 → −0.038, more lenient, §9.10).

**What §9.15's variants already say about it.** Each removes one half
of the asymmetry: terminal rollouts put both branches on terminal
scores (−0.08 ± 0.06 vs base at the residual nodes), the root-only tree
removes the maxed follow-up (−0.06 ± 0.05). Both move the label toward
the referee, by about a third of the lean each, at 1.2–1.3 SE.

**Caveat against making this the whole story.** At decline nodes the
preferred card is the alternative, so a pure preference effect predicts
a label lean AGAINST the over-trump there. §9.15 shows the opposite:
base −0.918 vs referee −1.018 (lean +0.10 toward the over-trump, ≈ 1.0
SE). The preference-tracking mix is real and measured; its contribution
to the sign is not established.

**The one untested rung.** `max_depth 1` + terminal rollouts together
(`budget_rows generate --variant
depth1_terminal:iters=256,max_depth_play=1,d_rollout=99`, as in
`t0_20k/variants_run.sh`) makes the search estimate
Q^θ₄ on SAMPLED worlds: the referee's quantity, with only the belief
differing (§9.14 channel 3; the referee plays the true deal). Its gap to
the referee isolates the belief pool; its gap to base is the whole
estimator asymmetry. Same 463 nodes, ≈ the cost of the terminal variant.
Run it first if the leak is picked up again; an inexpensive companion is
a uniform root prior (spreads tree depth evenly across cards, equalizing
the estimator mix without changing either estimator).

Outputs: `overtrump_probe/leaf_probe_preferred_over.log`,
`leaf_probe_preferred_decline.log` (argv: node count, `over`|`decline`).

### 9.18 Iteration 5 candidate (2026-10-22): the largest step yet, on the flat iteration

Iteration 5's cert was the series' first null read (routed play-only
+0.0006 ± 0.0021 vs θ₄, full h2h +0.0001 ± 0.0023; stop-rule counter
1/2). The operator asked for the over-trump read on the candidate
before the bidding phase finished, so both §9.8 instruments were run on
d5 = `pi/iter5/distill_epoch10.pt` under the bidding PPO, niced
(`runs/202609_recall_rc/overtrump_probe/iter5/probe_d5_now.sh`). Since
the bidding phase has left play unchanged on every node at iterations
1, 2 and 4 (§9.9: d4 → θ₄ Δ 0), the d5 read is the θ₅ read for play.

**Paired frozen-state read (the §9.8 359 nodes).**

| checkpoint | p(over) | clear blunder | schmear | argmax over |
|---|---|---|---|---|
| θ₃ | 0.297 | 0.219 | 0.262 | 29.5% |
| θ₄ | 0.289 | 0.214 | 0.247 | 28.4% |
| d5 | **0.247** | **0.182** | **0.205** | **24.0%** |

θ₄ → d5: Δp −0.0415 (one-sided 95% upper −0.0321) → PASS; clear-blunder
−0.0322 (upper −0.0236) → PASS; schmear −0.0419 (upper −0.0289) → PASS.
Argmax over-trumps 28.4% → 24.0%: 16 stopped, 0 started (McNemar
one-sided p < 0.001). The step is four to five times the θ₃ → θ₄ step
(−0.0086) and larger than the iteration-2 and iteration-3 steps
(−0.039, −0.034); every flipped argmax went the right way.

**Fresh-deal incidence (§9.2 scan, 4,000 seeds).** 96/346 = 27.7% ± 2.4
over-trumped (θ₄ 29.7%), trick 0 15.2% ± 3.5 (θ₄ 17.9%), clear-blunder
share 18.9% (20.3%); p(over) by trick 0.15 / 0.29 / 0.31 / 0.43 / 0.36.
Same direction, under the scan's power as usual.

**Reading.** The iteration whose aggregate gain went flat is the
iteration where this cell moved most. That is what §8's model-push
read says should happen as the series converges: the targets keep
pulling the residual classes off the prior (iteration 5 model-push
q95–q99 marginal +0.0236, 32 of 50 classes with a |z| ≥ 3 residual)
after the mass of ordinary decisions has stopped moving. §9.16 item 1's
"≈ 1 point per iteration" was the θ₃ → θ₄ pace; θ₀ → d5 now totals
−0.148 on p(over) and 40.9% → 24.0% on the argmax rate. Whether the
step persists into θ₅ and iteration 6 is the next read; the
bidding-boundary rerun on `pi/iter5/bidding/final.pt` confirms d5 = θ₅.

Outputs: `overtrump_probe/iter5/scan_d5_summary.txt`,
`frozen_overtrump_d5.txt`, `probe.log`.


