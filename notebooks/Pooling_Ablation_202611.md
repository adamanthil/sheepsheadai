# Pooling Ablation: Borrowed-Strength vs Plain Search Distillation — November 2026

**Status (2026-11-23): COMPLETE. The blend shows no measurable benefit
over dose-matched plain search distillation on strength or on trick-0
convention adherence, for one iteration from θ₀. Results in §7.**

The write-up of the search phase
([Building an Expert Sheepshead AI with Reinforcement Learning](https://bender.io/2026/11/17/building-an-expert-sheepshead-ai-with-reinforcement-learning))
presents borrowed-strength search distillation (BSSD): an advantage
model pools noisy search labels over similar decisions, a Fay–Herriot
blend combines the pooled estimate with each node's own label, and the
network is distilled toward a KL-regularized tilt of the blend
(`CE_Teacher_Design_202608.md` §20–§22 is the development record). It
says raw Q values "could easily fit to the noise instead of the signal".
It never tests that against plain search distillation with everything
else held equal. This notebook is that test.

## 1. The criticism

1. **No clean ablation.** The final recipe was never run against plain
   search distillation on the same corpus, with the same improvement
   step, schedule and dose.
2. **Regression already pools.** A network trained by cross-entropy on
   noisy targets averages across similar states, because that is what
   regression does. The advantage model may be doing work the student
   would do for itself.

## 2. Where the evidence stood

The first point is correct. What exists is confounded or indirect:

- **§17 → §20 of the design record** (pi_gumbel one-hot targets, −0.028
  h2h, replaced by calibrated targets, +0.017) changed the target
  operator, the anchor and the dose together.
- **The label-only arms of §20.13 addenda 6–9** (gamma = 1) were read at
  θ₁ of the v2 lineage, under the SE-unit tilt, precision weights and one
  trunk epoch. Every target construction read null there; the trunk
  epochs that made the recipe compound came later.
- **§22.6 and `Arm_D_Review_202610.md` §8** rank recipes by how much
  weight the label gets and value the model's share of each target by
  the node's own label. Both favour pooling; neither trains a student on
  label-only targets.

### 2.1 The original case for the blend (added 2026-11-23, after §7)

What led to the recipe, traced through `CE_Teacher_Design_202608.md`
(CE) and `Arm_D_Review_202610.md` (AD). Each item says what was
measured and what kind of evidence it was.

1. **Per-node labels cannot order the top cards at lead rows** (CE
   §20.1, 2026-09-01; 48k searched nodes of corpus q). Three-replicate
   sign agreement on the top pair at trick-0 defender leads was 25.0%
   against a simulated pure-noise null of 24.9% (median |t| 0.75);
   trick-1 leads 28.4%. Follows from trick 2 carried signal (81.7% and
   99.9%). The convention's value, 0.008–0.02 Q per opportunity, sits
   at or below the committee's 0.015 Q per-node resolution, and more
   search budget did not lower that floor. *Evidence about the labels.*
2. **The previous target turned those coin flips into near one-hot
   targets** (CE §20.1). At trick-0 defender leads the median target
   max-probability was 0.952 on a card the policy rated 0.49, at a
   top-2 gap of 0.4 committee SEs, because the Gumbel-style tilt
   multiplier was 50–80 logit units on min-max-normalized Q. Every
   student trained on it lost: arm c −0.0276 ± 0.0089, with spread
   compressed to about 2.9 (CE §17.9). *Evidence against an
   uncalibrated tilt.*
3. **The pooled effect is in the labels and in the features** (CE
   §20.7). Pooled over committee rows, called-suit cards beat other
   fail cards by +0.0114 ± 0.0011 Q at trick-0 defender leads (10 SE),
   matching the CRN counterfactual instrument. A linear probe on the
   frozen hand tokens finds called-suit membership at held-out AUC
   0.94. From these two facts §20.1 concluded the convention "is
   resolvable only by pooling across nodes ... what a per-node search
   label cannot do". *Evidence that pooling is needed somewhere; it
   does not say the student cannot be the one to do it.*
4. **The first calibrated arm won** (CE §20.6): +0.0171 ± 0.0068 vs θ_k
   against −0.0276 for arm c, with no spread compression (3.65). This
   was read as "the floor was target-borne". The arm changed the tilt
   form (Gumbel min-max to an SE-unit mirror-descent step), the
   temperature, the anchor and the blend together, and was certified on
   2,000 deals per mode, which CE §20.13 addendum 3 later found
   overstated the gain. *Confounded; consistent with calibration alone
   being the fix.*
5. **More model in the blend made lead targets more coherent** (CE
   §20.7–§20.8). The model's share at trick-0 leads went 16% → 34% →
   about 70% across arms 1–3 and its capture of the pooled effect 13% →
   37% → 55% with the bilinear term. Rows tilting toward against away
   from the called suit went 51/20 → 57/20 and called-suit target mass
   0.377 → 0.451 / 0.461. The h2h reads were +0.0171, +0.0183, +0.0252,
   each ± 0.0065: not separable. None of the arms moved the greedy
   called-suit rate on fresh deals (45.2 / 43.7 / 45.0 against 44.7).
   *Target-side evidence only; the student's play did not follow.*
6. **Label-only was set aside by design** (CE §20.13 addendum 6b,
   2026-09-08). The operator framing recorded there calls the advantage
   model's pooling "the ESSENTIAL element of §20", so the gamma = 1 arm
   "discards that element and is a diagnostic, not a candidate recipe".
7. **The one direct comparison was a tie** (CE §20.13 addenda 7–9b).
   On the θ₁ twin corpus five target constructions, class blend, global
   blend, gamma = 1 at two temperatures, all read −0.006 to −0.004 vs
   θ₁; label-only against the class blend paired −0.0009 ± 0.0035. It
   was read as "the target stage is not the lever at θ₁", which was
   right: six trunk epochs were. It was not re-read as a question about
   the blend once the schedule was fixed.
8. **On the recall lineage, recipes with more model weight certified
   higher** (CE §22.6, 2026-10-09). The baseline, whose defender-follow
   gamma of 0.09 was an accident of a variance floor, read +0.0165
   play-only; arm A (gamma 0.01, no floor) +0.0055; arm C (class-pooled
   noise) +0.0106. The arms also differed in dose: target KL p50 0.043
   / 0.097 / 0.059, and 5% / 22% / 5% of rows at the clip. *Dose and
   blend moved together.* §7 separates them: the label-only arm at the
   recipe's η has KL p50 0.071 and plays 0.005 below the recipe, and
   the gap closes when the dose is matched.
9. **The model's push is credited by the node's own label** (AD §8,
   2026-10-13). Splitting each target into its search and model terms
   and valuing each by Σ(q − p)·y, the model-only gain was positive in
   all 50 classes, at 3 SE in 43, and close to calibrated at the
   convention rows. AD §7.5 attributes the fall in trick-0 defender
   trump leads to the model term. AD §8.5 states the limit itself:
   "Real EV: not measurable ... The offline search-judged read is the
   only practical instrument." §4 of this notebook is the same
   instrument.

In sum, the case rested on three things: a correct diagnosis that
single labels are noise at lead rows; a large win over an over-sharp
target that changed several things at once; and target-side reads
(coherence, capture, label-judged value) that improve as the model's
share rises. No item compares students trained on blended and
label-only targets with the tilt form, dose and schedule held equal.
The one near-comparison (item 7) tied.

The second point is half right, and the half is testable. Cross-entropy
is linear in the target, so the student fits E[t | s]. If the target
were linear in the label, that would be the pooled label and the
advantage model would be redundant. The target is not linear:

    t(a) ∝ p(a) · exp(clip(y(a) / η, ±8))

At η = 0.0097 the label noise is 1.3–1.4 nats SD at trick-0 and trick-1
defender rows (0.4–0.5 at trick 4). The student therefore fits the
average of an exponentiated noisy label, which is flatter than the tilt
of the average label and carries an attenuated signal. Pooling before
the tilt and pooling after it are different estimators. Against that,
the student pools over its own trainable features, which may be a
better notion of "similar" than a small head on θ_k's frozen ones, and
it sees every row ten times. Which effect is larger is an empirical
question.

## 3. Design

One iteration of the recall run (`runs/202609_recall_rc/pi/iter1`): the
8,000-game corpus generated by θ₀ = league `checkpoint_8000000`, 137,765
searched play rows, and iteration 1's own fit artifacts (advantage
model, out-of-fold predictions, row table) copied into every arm rather
than refit.

Identical in every arm: the tilt form and clip above, uniform CE
weights, six trunk epochs at 3e-5 then four head epochs, retention KL
×10, last-epoch candidate, and the certification battery. The arms
differ only in the advantage the tilt reads, selected by
`policy_iteration --blend`:

| arm | advantage | η | KL(t‖p) p50 / mean | rows at the clip |
|---|---|---|---|---|
| R_s0 | final recipe: Fay–Herriot blend (cross-fit 5, class-pooled noise, one-sided-head model variance) | 0.0097 | 0.0435 / 0.141 | 8.4% |
| P | the node's search label alone (`--blend search`, γ = 1) | 0.0097 | 0.0710 / 0.223 | 11.4% |
| P14 | label alone at R's dose | 0.0140 | 0.0436 / 0.147 | 6.6% |
| M | the pooled out-of-fold model alone (`--blend model`, γ = 0) | 0.0097 | 0.0346 / 0.105 | 6.7% |
| R_s1 | final recipe, distill seed 1 | 0.0097 | as R_s0 | |

Why each arm is there:

- **R_s0** is iteration 1 re-run under the final recipe. Iteration 1
  itself used class model variance; the final recipe (iteration 3 on)
  uses one-sided-head. That is the only difference.
- **P** is the literal one-flag ablation. Label-only targets move 60%
  further from the prior at the same η, so a loss could be blamed on
  dose.
- **P14** removes that objection: η = 0.014 matches R's median target
  KL. It is the headline contrast.
- **M** tells whether the node's own label adds anything once the
  pooled model is in hand (arm F of §22.6, never run).
- **R_s1** measures the run-to-run spread of a single distillation, the
  floor every contrast is read against.

"Plain" here means the same KL-regularized improvement step applied to
the raw committee Q. A visit-count target in the AlphaZero form is a
different operator and is not an arm; §17 is the nearest record of it.

**Primary read.** Direct duplicate h2h of each arm against R_s0, full
networks, 16,000 deals per partner mode on deal seed 20261122. Every
recipe decision to date was read on the seed-42 deals, which would
favour R slightly; these deals are fresh. Expected SE ≈ 0.002.

**Secondary reads.** The cert against θ₀ (play-only routed read on the
seed-42 deals), the convention probes, play-logit spread, and the
train and held-out target KL per distill epoch.

Code: `--blend` in `sheepshead/rl/search_advantage.py`
(`blend_advantages`) and `sheepshead/training/policy_iteration.py`.
Runner and ledger: `runs/202609_recall_rc/pi/iter1_ablations/pooling/`
(`run_pooling_ablation.sh`, `h2h_direct.py`, `ablation.log`).

## 4. Offline read of the targets

Before any training, each target set was scored by the search value its
move p → t captures, Σ(t − p)·y judged by the node's own label and
Stein-corrected for fitting that label's noise. Game points per
decision:

| rows | R | P | M | label noise (nats at η 0.0097) | γ p50 |
|---|---|---|---|---|---|
| all | +0.0288 | +0.0112 | +0.0206 | | 0.62 |
| std t0 defender lead | +0.0095 | −0.0252 | +0.0262 | 1.27 | 0.36 |
| std t1 defender lead | +0.0051 | −0.0222 | +0.0195 | 1.34 | 0.36 |
| std t0 defender follow | +0.0182 | +0.0096 | +0.0188 | 1.36 | 0.24 |
| std t1 defender follow | +0.0213 | +0.0144 | +0.0197 | 1.34 | 0.35 |
| std t2 defender follow | +0.0308 | +0.0325 | +0.0202 | 1.09 | 0.60 |
| std t3 defender follow | +0.0434 | +0.0459 | +0.0191 | 0.84 | 0.81 |
| std t4 defender follow | +0.0412 | +0.0420 | +0.0143 | 0.48 | 0.93 |
| std t1 picker follow | −0.0275 | −0.1130 | +0.0273 | 3.77 | 0.45 |

P's all-rows value does not move with η (+0.0111 to +0.0116 over 0.0097
to 0.04). Label-only targets lose where the noise is large relative to
η, the early tricks, and match the blend from trick 3 on, where the
blend trusts the search anyway. The model alone is best early and worst
late.

This scores a student that reproduces each row's target exactly. The
criticism is that the student does not, so the table is a lower bound
on P and decides nothing. The correction assumes Gaussian noise and is
least reliable on the picker rows, where a third of rows sit at the
clip.

## 5. Predictions

Recorded before the first arm finished distilling.

| arm | play-only vs θ₀ | arm − R_s0 (direct) | P(arm < R) |
|---|---|---|---|
| R_s0 | +0.019 (arm V read +0.0198 ± 0.0029) | — | — |
| R_s1 | +0.019 | 0.000, within ±0.004 | — |
| P | +0.009 (+0.003 to +0.015) | −0.010 (−0.016 to −0.004) | 0.90 |
| P14 | +0.012 (+0.006 to +0.018) | −0.007 (−0.013 to −0.001) | 0.80 |
| M | +0.014 (+0.008 to +0.019) | −0.005 (−0.010 to +0.001) | 0.75 |

- Plain distillation works. P and P14 both beat θ₀ clearly (P14 above
  θ₀ at 2 SE with probability 0.9), capturing half to two thirds of R's
  gain.
- Conventions do not separate the arms. The called-suit lead is already
  above 95% on this lineage, and a trick-0 defender trump lead is a
  −0.03 Q blunder that a single node resolves, so its rate falls in
  every arm.
- P has the widest train-to-held-out target-KL gap and the lowest
  play-logit spread.
- The loss sits mostly in called mode.

## 6. Readings fixed in advance

- **P14 − R within ±0.004 (2 SE).** Pooling adds nothing measurable
  beyond the student's own averaging. The criticism stands and the
  write-up's claim reduces to "equivalent, at the cost of five extra
  fits".
- **P14 − R ≤ −0.004.** The blend is worth that much per iteration at
  matched dose.
- **P well below P14.** The plain form needs its temperature retuned to
  the noise; part of the blend's gain is automatic dose allocation.
- **M ≈ R > P14.** The model is the whole effect and the node's own
  label adds nothing.
- **R > M and R > P14.** Both terms contribute, as designed.
- **|R_s1 − R_s0| > 0.004.** Single-seed contrasts of this size are not
  readable and every arm needs seeds before any conclusion.

Limits. One iteration from one θ₀: nothing here speaks to compounding,
where pooled and noisy targets could diverge further or converge. One
corpus size and one search budget: with less noise γ → 1 and the two
methods coincide, so the result is specific to this noise level. The
advantage model's fit is shared, so its cost and variance are not part
of the comparison.

## 7. Results

Run 2026-11-22 23:10 to 2026-11-23 14:44, the blend switch of `56469b9`. All arms use
the last-epoch candidate (epoch 10).

### 7.1 Strength

| arm | full h2h vs θ₀ | play-only vs θ₀ | bidding-only | arm − R_s0 (direct, fresh deals) | arm − R_s0 (paired play-only, cert deals) |
|---|---|---|---|---|---|
| R_s0 | +0.0218 ± 0.0031 | +0.0195 ± 0.0029 | +0.0027 ± 0.0013 | — | — |
| R_s1 | +0.0179 ± 0.0033 | +0.0191 ± 0.0031 | −0.0008 ± 0.0012 | −0.0014 ± 0.0021 | −0.0004 ± 0.0024 |
| P | +0.0214 ± 0.0034 | +0.0147 ± 0.0030 | +0.0062 ± 0.0019 | +0.0024 ± 0.0020 | −0.0048 ± 0.0022 |
| P14 | +0.0250 ± 0.0030 | +0.0221 ± 0.0028 | +0.0028 ± 0.0014 | −0.0009 ± 0.0016 | +0.0026 ± 0.0020 |
| M | +0.0125 ± 0.0031 | +0.0124 ± 0.0030 | −0.0003 ± 0.0010 | −0.0063 ± 0.0019 | −0.0071 ± 0.0024 |

The first three columns are the cert (8,000 deals per mode, seed 42).
The direct column is the full duplicate h2h on 16,000 fresh deals per
mode (seed 20261122). The paired column differences the cert's per-deal
play-only scores.

### 7.2 Conventions at trick 0

Seeded probes of `Convention_Optimality_202610.md` §2, 100,000 games
per mode. θ₀ reads 98.83% / 5.37% / 0.74% on the same deals.

| arm | called-suit lead (called ace) | defender trump lead (JD) | defender trump lead (called ace) |
|---|---|---|---|
| R_s0 | 99.84% (27,233 / 27,277) | 3.52% (1,639 / 46,518) | 0.13% (64 / 49,886) |
| R_s1 | 99.87% (26,594 / 26,628) | 4.76% (2,226 / 46,785) | 0.16% (80 / 50,128) |
| P | 99.84% (27,254 / 27,297) | 2.39% (1,121 / 46,871) | 0.03% (16 / 49,549) |
| P14 | 99.81% (27,170 / 27,223) | 3.24% (1,508 / 46,515) | 0.12% (58 / 49,794) |
| M | 99.89% (26,772 / 26,801) | 3.01% (1,403 / 46,670) | 0.10% (48 / 50,008) |

The called-suit rate is reported at trick 0 only. Later in the hand a
defender can be right not to lead the called suit, so the cert's
all-tricks rate (94–97% in every arm) is a health check, not a measure
of adherence.

### 7.3 Target fit and head sharpness

| arm | target KL p50 (dose) | held-out target KL | play-logit spread | cert |
|---|---|---|---|---|
| R_s0 | 0.0435 | 0.103 | 3.69 | PASS |
| R_s1 | 0.0435 | 0.102 | 3.66 | PASS |
| P | 0.0710 | 0.187 | 3.12 | FAIL (spread) |
| P14 | 0.0436 | 0.125 | 3.28 | FAIL (spread) |
| M | 0.0346 | 0.065 | 4.02 | PASS |

θ₀'s spread is 3.22 on the league probe. The 3.6 bar was set on the
D8k lineage, whose seed read 3.56–3.60, as a guard against compression
below the seed. P and P14 sit at θ₀'s value: they did not soften the
head, they did not sharpen it. The two spread failures are the
inherited bar, not the compression that preceded deterioration in the
earlier teacher attempts (`CE_Teacher_Design_202608.md`, attempt 11:
3.56 → 2.12).

### 7.4 Reading against §6

- **P14 − R within ±0.004: met.** Direct −0.0009 ± 0.0016; paired
  play-only +0.0026 ± 0.0020. The 95% upper bound on the blend's
  advantage at matched dose is about +0.004 on the direct read. The
  criticism stands for one iteration.
- **Noise floor: readable.** R_s1 − R_s0 is −0.0014 ± 0.0021 direct and
  −0.0004 ± 0.0024 paired, inside ±0.004.
- **M loses** by 0.006–0.007 on both reads. The model alone is a worse
  teacher than the raw labels; the "M ≈ R" reading is refuted.
- **P at the recipe's η is mixed.** Its play-only read is 0.005 below R
  and 0.007 below P14, so the plain form does want its temperature
  retuned. Its bidding-only read is the highest of any arm, and the
  full h2h ties R on both deal sets.
- **Conventions do not separate the arms.** Called-suit leads at trick 0
  land at 99.81–99.89% everywhere. On trump leads the two recipe seeds
  differ by 1.2 points in JD mode, as much as any arm differs from
  either, and no ablation arm is worse than both seeds.

### 7.5 Predictions against outcomes

| arm | predicted arm − R | observed (direct) |
|---|---|---|
| R_s1 | 0.000, within ±0.004 | −0.0014 ± 0.0021 |
| P | −0.010 | +0.0024 ± 0.0020 |
| P14 | −0.007 | −0.0009 ± 0.0016 |
| M | −0.005 | −0.0063 ± 0.0019 |

The predictions for the two label-only arms were wrong, and wrong for
the reason the criticism gives. The offline target read (§4) scored the
blended targets well above the label-only ones (+0.029 against +0.011
points per decision), and that did not transfer. Held-out target KL
orders by how much model is in the target (M < R < P14 < P): blended
targets are smoother and easier to fit, but the student gains nothing
from that. The advantage model is the play head on frozen θ₀ features;
the student is the same head with a trainable trunk, so it can do any
pooling the model can. The blend can only remove information from the
labels, and the M arm shows what removing all of it costs.

### 7.6 What this does not settle

- One iteration from θ₀. The published run compounded six; later
  iterations have smaller edges against the same label noise. A plain
  arm on the iteration-4 corpus (the last with a measurable gain,
  +0.0046 ± 0.0022) would test that regime, and needs a direct h2h of
  about 64,000 deals per mode to resolve a gain that size.
- One corpus size and search budget. With fewer games or fewer rollouts
  per node the student has less to average over.

The plain arm is established practice: a KL-regularized tilt of search
values, as in Grill et al. 2020, Gumbel MuZero (Danihelka et al. 2022)
and MPO. The blend is the novel part, and on this evidence it adds
nothing to it.

## 8. Iteration-4 follow-up (pre-registered 2026-11-23, RUNNING)

§7.6 leaves open whether the blend earns its place once edges are small
against the same label noise. Iteration 4 is the last iteration with a
measurable recipe gain (play-only +0.0046 ± 0.0022 vs θ₃; iterations 5
and 6 read +0.0006 and +0.0013), and the first run entirely under the
final recipe.

### 8.1 Design

Same construction as §3 on the iteration-4 corpus (8,000 games from θ₃
= `pi/iter3/bidding/final.pt`) with iteration 4's own fit artifacts.
Run directory `runs/202609_recall_rc/pi/iter4_ablations/pooling/`.

| arm | advantage the tilt reads | η | target KL p50 |
|---|---|---|---|
| R_s0 | Fay–Herriot blend (final recipe) | 0.0097 | 0.0180 |
| P149 | label alone (`--blend search`) | 0.0149 | 0.0180 (scan: 0.0145 → 0.0187, 0.0155 → 0.0170) |
| R_s1 | final recipe, distill seed 1 | 0.0097 | 0.0180 |

The model-only and same-η label arms are dropped: §7 showed the first
loses and the second is a temperature mismatch. Cert as before (8,000
deals per mode vs θ₃). The direct h2h against R_s0 is raised to 64,000
fresh deals per mode (seed 20261122), for an expected SE near 0.001,
because the whole gain to split is under 0.005.

### 8.2 Predictions

| arm | play-only vs θ₃ | arm − R_s0 (direct) |
|---|---|---|
| R_s0 | +0.005 (reproduces iteration 4 within 1 SE) | — |
| R_s1 | +0.005 | 0.000, within ±0.002 |
| P149 | +0.004 | −0.0005, within ±0.002 with probability 0.75 |

After §7 the expectation is a tie. The case for a gap is that the
label-only target leans on single noisy labels at exactly the rows
where the true edge is now smallest; the case against is unchanged,
the student pools for itself.

### 8.3 Readings fixed in advance

- **P149 − R_s0 within ±0.002.** The null extends to the low-signal
  regime. The blend adds nothing measurable at either end of the
  program.
- **P149 − R_s0 ≤ −0.002 and outside the R_s1 spread.** The blend is
  worth that much per iteration once edges are small, and §7's null is
  specific to the first iteration.
- **P149 − R_s0 ≥ +0.002.** The blend costs strength late, consistent
  with the model-only harm in the top rows recorded at iterations 5–6.
- **|R_s1 − R_s0| > 0.002.** The seed spread is as large as the effect
  sought; no conclusion without more seeds.

Limit, unchanged: θ₃ was itself built by three blended iterations, so
this compares single steps, not chains.
