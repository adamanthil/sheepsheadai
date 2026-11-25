# Acting-Mode Twin: Committee Acting vs Policy Sampling in the Corpus — November 2026

**Status (2026-11-25): COMPLETE. Policy sampling matched committee acting in one step (S − C = +0.0006 ± 0.0011, seed floor +0.0008 ± 0.0010); results in §5.**

The corpus generator plays θ_k's own stochastic self-play, except at
"resolved" searched nodes, where it plays the argmax of the legacy
pi_gumbel readout (`--committee-act-frac 1.0`, "committee acting").
Sampling the policy at every node would remove the resolved/unresolved
split, the pi_gumbel readout and its shrink constants from the
generator. This notebook tests whether that costs anything.

## 1. Where the evidence stood

The one prior estimate is `CE_Teacher_Design_202608.md` §20.13 addendum
16: committee acting +0.0055 ± 0.0021 play-only over student acting, on
twin corpora at θ₁ of the v2 lineage. It is not a clean read of
learning. Both arms were trained under the one-epoch schedule that
preceded the final recipe, and neither improved on θ₁: the
committee-acted student read −0.0011 and the student-acted one −0.0066.
The difference measures which arm was damaged less, and the addendum
notes the damage landed in different heads in the two arms. On that
estimate the 2.0 plan dropped the twin (2026-11-22); this reopens it.

On priors the effect should be small. Committee acting changes about
2.8 cards per game at iteration 1 (22,353 nodes over 8,000 games), and
one iteration moves the policy 0.02–0.04 nats, so θ_{k+1}'s state
distribution stays close to θ_k's.

## 2. Design

One step from θ₀ = `runs/202609_recall_rc/league/checkpoints/checkpoint_8000000.pt`,
the start point with the largest gain to split. Run directory
`runs/202609_recall_rc/pi/iter1_ablations/acting/`.

| arm | corpus | acting |
|---|---|---|
| C | games 0–1,999 of `pi/iter1/corpus`, re-sharded | committee argmax at resolved nodes |
| S | the same 2,000 game seeds regenerated, `--committee-act-frac 0.0` | policy sampling at every node |

Everything else in the generator is iteration 1's configuration (256
iterations, trick-0 leads 1,024, trick-1 leads 512, three replicates,
seed 20260903). Games are seeded by index, so the two corpora share
deals and play identically until the first committee deviation in each
game. Reproduction check: 16 games regenerated on today's code with
committee acting match the October corpus exactly in 15 (every action,
every Q to 1e-6); one game diverges, so generation is close to but not
bit-reproducible across the two dates.

Both arms are distilled label-only (`--blend search`, η 0.014, the
dose-matched form of `Pooling_Ablation_202611.md`), with the final
recipe's distillation and cert. η is fixed across arms; each arm's
target KL is reported.

**Schedule.** A quarter-size corpus gives a quarter of the optimizer
steps at the recipe's 6 + 4 epochs. The control arm is therefore run at
6 + 4 (C_e10) and at 24 + 16 epochs (C_e40, equal steps to the full
recipe) while the S corpus generates. The schedule with the higher
play-only read against θ₀ is used for both arms; a tie inside 1 SE
keeps 6 + 4. The choice is made on the control arm alone, before the S
corpus exists.

**Comparison.** Direct duplicate h2h, S against C, 64,000 fresh deals
per mode (seed 20261122), expected SE about 0.001. A second distill
seed of C gives the run-to-run noise floor on the same instrument.
Trick-0 convention probes on both candidates.

## 3. Predictions

| quantity | prediction |
|---|---|
| C play-only vs θ₀ | +0.010 (half the full-corpus +0.020) |
| S − C, direct | −0.001; inside ±0.003 with probability 0.7 |
| C seed spread | inside ±0.002 |

## 4. Readings fixed in advance

- **S − C within ±0.002.** Policy sampling costs nothing measurable in
  one step. Recommend dropping committee acting from the 2.0 generator,
  subject to the limit below.
- **S − C ≤ −0.003 and outside the seed spread.** Committee acting earns
  its place; the branch stays.
- **Between −0.003 and −0.002, or a seed spread above 0.002.**
  Inconclusive; a second step from each arm decides.
- **C below +0.005 against θ₀ at the chosen schedule.** A quarter corpus
  does not learn enough to split; the test needs more games.

Limits. One step: the mechanism claimed for committee acting, that it
teaches the student where it will be once it adopts the search's lines,
is a next-iteration effect, so a tie here is strong but not complete
evidence of equal compounding. One start point and one corpus size.

## 5. Results

### 5.1 Schedule choice (control arm only, 2026-11-24 21:40)

Cert against θ₀ (4 × 1,000 probe games, 8,000 duplicate deals per mode),
on the committee-acted 2,000 games, label-only target at η 0.014
(target KL p50 0.0436 for both, same corpus):

| arm | epochs (trunk + head) | play-only vs θ₀ | full h2h vs θ₀ | play logit spread |
|---|---|---|---|---|
| C_e10 | 6 + 4 | +0.0058 ± 0.0028 | +0.0066 ± 0.0031 | 3.27 |
| C_e40 | 24 + 16 | +0.0100 ± 0.0031 | +0.0111 ± 0.0033 | 3.49 |

The gap is +0.0042, larger than either read's SE, so the rule selects
**24 + 16** for both arms. The two certs are not independent draws
(same deals), and no paired SE was computed; treated as two
independent reads the gap is 1.0 combined SE. The choice follows the
rule as written and is not strong evidence that the longer schedule is
better. C_e40 lands on the predicted +0.010 and clears the +0.005 floor
of §4. Both "fail" the cert only on the inherited spread bar of 3.6
(θ₀ itself reads 3.22; see `Pooling_Ablation_202611.md` §7.3).

The S corpus was at 1,150 of 2,000 games when this was decided; no S
row had been read.

### 5.2 Comparison (2026-11-25)

The S corpus finished at 04:59 (2,000 games, 34,436 searched nodes,
16,374 of them resolved; with `--committee-act-frac 0.0` none was
acted on). All arms: label-only target at η 0.014, 24 + 16 epochs,
last-epoch candidate.

**Corpus and target.**

| arm | corpus | target rows | target KL p50 | held-out target KL |
|---|---|---|---|---|
| C_e40 | committee-acted | 34,387 | 0.0436 | 0.128 |
| C_e40_s1 | committee-acted, seed 1 | 34,387 | 0.0436 | 0.137 |
| S_e40 | policy-sampled | 34,348 | 0.0436 | 0.122 |

Row counts and dose are equal to within 0.1%, so no η adjustment was
needed.

**Strength.** Cert reads against θ₀ are 8,000 duplicate deals per
mode; the direct reads against C_e40 are 64,000 fresh deals per mode
(seed 20261122).

| arm | play-only vs θ₀ | bidding-only vs θ₀ | full vs θ₀ | direct vs C_e40 (called / JD) |
|---|---|---|---|---|
| C_e40 | +0.0100 ± 0.0031 | +0.0008 ± 0.0013 | +0.0111 ± 0.0033 | — |
| C_e40_s1 | +0.0133 ± 0.0030 | −0.0002 ± 0.0011 | +0.0137 ± 0.0032 | +0.0008 ± 0.0010 (+0.0022 / −0.0006) |
| S_e40 | +0.0139 ± 0.0030 | −0.0010 ± 0.0009 | +0.0129 ± 0.0031 | +0.0006 ± 0.0011 (+0.0013 / −0.0001) |

**Trick-0 conventions** (seeded probes, 100,000 games per mode; θ₀
reads 98.83% / 5.37% / 0.74%):

| arm | called-suit lead | defender trump lead, JD | defender trump lead, called ace |
|---|---|---|---|
| C_e40 | 99.79% ± 0.03 | 3.32% ± 0.08 | 0.10% ± 0.01 |
| C_e40_s1 | 99.35% ± 0.05 | 6.15% ± 0.11 | 0.59% ± 0.03 |
| S_e40 | 99.80% ± 0.03 | 7.21% ± 0.12 | 0.30% ± 0.02 |

Cert play logit spread: 3.49 / 3.46 / 3.42 (θ₀ 3.22).

### 5.3 Reading

- **Strength: the first reading of §4 applies.** S − C is +0.0006 ±
  0.0011, inside ±0.002, and indistinguishable from the control's own
  seed spread (+0.0008 ± 0.0010). The 95% interval is −0.0016 to
  +0.0028, so a one-step cost of committee-free acting larger than
  about 0.002 is excluded. The play-only reads against θ₀ agree
  (+0.0139 for S against +0.0100 and +0.0133 for the two C seeds).
- **Called-suit lead at trick 0** is installed by all three arms
  (99.35–99.80%, from θ₀'s 98.83%); S ties the better C seed.
- **JD defender trump lead is not settled at this corpus size.** The
  two C seeds read 3.32% and 6.15% on identical data, a 2.8-point
  spread, and S reads 7.21%. Two of the three arms are above θ₀'s
  5.37%. S is 1.1 points above the worse C seed, which is smaller than
  the seed spread, so the probe cannot attribute it to acting mode.
  The full-corpus recipe arms read 3.5% and 4.8% across seeds
  (`Pooling_Ablation_202611.md` §7.2), so this node is seed-sensitive
  even at 8,000 games, and more so at 2,000 with four times the
  passes.
- **Predictions.** C against θ₀ +0.010 predicted, +0.0100 and +0.0133
  observed. S − C −0.001 predicted, +0.0006 observed (inside the
  ±0.003 band). Seed spread inside ±0.002 predicted, +0.0008 observed.

### 5.4 Conclusion and limits

In one step from θ₀, sampling the policy at every node produces a
student as strong as acting on the committee readout at resolved
nodes, on the same 2,000 deals with the same search. The earlier
+0.0055 read for committee acting (§1) is not reproduced.

Not covered:

- **Later iterations.** The acting mode changes which states the
  corpus visits, and the argument for committee acting is that it
  puts the next iteration's search on improved lines. One step from
  θ₀ cannot show a difference that only appears when the two lineages
  diverge. A two-step chain per arm would test it.
- **Full corpus size and the recipe schedule.** This ran at 2,000
  games and 24 + 16 epochs, label-only. Nothing here suggests the
  acting mode interacts with corpus size, but it was not measured.
- **JD trump-lead at trick 0**, per above.

Recommendation: drop committee acting from the 2.0 corpus generator
(`--committee-act-frac 0.0`) and keep the iteration-over-iteration
cert as the guard: a policy-sampled lineage that stops compounding
where 1.0 did not would show there.
