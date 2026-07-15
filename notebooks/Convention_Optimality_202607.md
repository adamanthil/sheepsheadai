# Convention Optimality — Experiment Log (July 2026)

**Status (2026-07-15, corrected same day): E1, E1-b, E2 rung 2, E3 rung 2,
E5 all DONE — see Results. Headline: both conventions are genuinely good on
average AND largely internalized by the 30M model. C2's residual deviations
are cost-neutral (DISAGREE null). C1's residual trump leads are still mildly
bad (−0.13/case, REPRODUCING the June investigation — see Reconciliation)
but rare enough (~0.8%/game) to be economically nil. The 17.5% fail-pref
"exception" class is hindsight-suspect and awaits rung-3. Remaining,
post-Stage-1: E2 step 2 + E3 step 2 (rung-3 search — decides the exception
verdicts), E4 gauntlet, gen-end E1-b/critic-probe oracle retest.**

> Editorial note 2026-07-14: restructured into runbook form (TL;DR, state of
> play, exact commands per experiment). Pre-registered decision rules,
> definitions, and dated amendments are carried over UNCHANGED from the
> 2026-07-12 design.

## TL;DR — what this is and how to use it

We are testing whether two human Sheepshead conventions are (Q1) actually
optimal and (Q2) learnable under terminal-only reward:

* **C1 — defenders don't lead trump** (while holding a legal fail lead).
* **C2 — defenders lead the called suit through** (called-ace mode, while the
  called suit has not been led yet).

Five experiments: **E1** measures whether checkpoints *follow* the conventions
(done — they increasingly do); **E2** measures whether C2 is actually *good*
(counterfactual ladder); **E3** measures whether C1 has legitimate
*exceptions*; **E4** prices *forcing* the conventions at deploy; **E5** turns
the results into an "episodes to learn this" number.

To continue this work: read **State of play**, pick the next PENDING row, run
the exact commands in its Runbook section, check the falsifiers listed there,
then append a `## Results` section here (copy the E1/E2-pilot format) and
apply the decision rule quoted in that section. Do not change decision rules
or definitions; if something must be amended, add a dated note (see the two
existing amendments) — never edit history.

## State of play

| Step | What | Status | Where |
|---|---|---|---|
| Instruments | scanners, probes, ladder, wrapper, calculators | **DONE 07-13** (commits 61a151ef..a866eaf8) | see Instruments table |
| E1 | adherence across checkpoints | **DONE 07-13** — both conventions being learned; see Results | `adherence_sweep.{json,log}` |
| E1-b | adherence on Stage-1 intermediate ckpts (850k/900k) | **DONE 07-15** — no C1 spike in oracle arms; see Results | `adherence_sweep_stage1_intermediate.*` |
| E2 pilot | 300 seeds, rung 2 | **DONE 07-13** — sanity + falsifier pass | `cf_called_suit_pilot.{json,log}` |
| E2 full step 1 | 1200 seeds, rung 2 | **DONE 07-15** — AGREE +4.4σ, PARTNER −2.7σ, DISAGREE null (verdict: underpowered) | `cf_called_suit_r2.*` |
| E2 full step 2 | rungs 2b + 3 (4096-iter search subsample) | PENDING (post-Stage-1; de-biases AGREE/PARTNER magnitudes) | Runbook §E2 |
| E3 step 1 | unconditional rung 2 + exception report | **DONE 07-15** — pooled −0.49 (10σ pro-convention) BUT 17.5% exceptions | `cf_trump_unconditional.*`, `exception_report.json` |
| E3 step 2 | rung-3 re-runs on flagged exceptions | PENDING (post-Stage-1) — decides heuristic-with-exceptions verdict | Runbook §E3 |
| E4 | wrapper gauntlet (raw vs @c1 vs @c1c2) | PENDING (post-Stage-1); pre-read (corrected 07-15): expect ≈ 0, tiny positive at most (~+0.001/game) — a bound-check | Runbook §E4 |
| E5 | learnability numbers | **DONE 07-15** (probes + SNR) — limited critic partially sighted; oracle≈limited at 850k (retest gen end); raw signal not the binding constraint | Runbook §E5 |

All outputs go to `runs/convention_optimality_202607/` (gitignored — results
are recorded in this document). All commands run from the repo root with
`uv run` (never system python; repo pins 3.14) and `nice -n 19` for anything
heavy.

## Background

### The two conventions

* **C1 — Defenders don't lead trump.** A defender (not picker, not revealed or
  secret partner, non-leaster) never leads trump while holding a legal fail
  lead. This is the behavior the 30M lineage historically leaked (the trick-0/1
  "trump leak" investigation, `notebooks/defender_trump_lead_investigation.md`).
* **C2 — Defenders lead the called suit through.** In called-ace mode, a
  defender holding a called-suit fail leads it at the first opportunity
  (scripted agent: trick 0). Rationale: the picker is *guaranteed* to hold the
  called suit (rule-enforced: `get_callable_cards` requires a fail of the suit
  and the bury respects `get_playable_called_picker_cards`, so the picker never
  buries the last one), the secret partner must surface the called ace when the
  suit is led, so the lead (a) publicly identifies the partner immediately and
  (b) offers a void defender the chance to trump the 11-point ace.

### The two questions

* **Q1 (optimality):** Is each convention actually optimal in expectation — or
  a good-but-imperfect human heuristic with exception classes? If exceptions
  are real and material, an agent that deviates is *right* to deviate, and the
  product answer is education/explanations, not behavior change.
* **Q2 (learnability):** Under the terminal-only reward regime, is the
  per-decision advantage of the convention detectable by PPO at realistic
  sample budgets — and is it empirically being learned across our checkpoints?
  If Q1 says "optimal" and Q2 says "not learnable," the remedy is a deploy
  wrapper (E4) or a privileged/teacher signal, not more episodes.

Product stake: experienced humans playing at a table with an agent that leads
trump as a defender, or sits on the called suit, will read it as weak play
regardless of its measured strength.

### Prior evidence (do not re-derive)

* C1 on the *agent's mistake nodes* is settled: the 3-rung counterfactual
  ladder (`counterfactual_trump_leads.py`) found belief-MC −0.19 score for the
  trump lead, and `targeted_trump_lead_search.py` at offline compute (4096
  iters, rollout-to-terminal, frac=1.0 + top@Q) confirmed fixing those leads is
  worth +0.16 (2.4σ, n=101) with zero control harm. **But both condition on
  TRUMP-PREF nodes** — spots where the policy argmax already leads trump. That
  answers "are the agent's trump leads mistakes?" (yes), not "is *never*
  leading trump optimal?" (E3's job).
* C2 had never been measured before E1.
* Terminal-only learnability mechanism is documented: PPO-can't-learn-
  small-early-gaps (hidden-card variance swamps small early advantages; see
  deploy-search notes + oracle-critic motivation). E5 quantifies it for these
  two specific decisions instead of arguing it generally.
* `exit_validation.py` tracks `t0_trump_lead_rate` (historical figure 4.8%
  greedy) — superseded: E1 found this does not reproduce in current greedy
  contexts (see E1 reading #3); the CRN probe is canonical now.

## Definitions (shared across experiments; pre-registered)

* **C1-eligible node:** defender (per scanner definition in
  `scan_defender_trump_leads.py`) leads on trick 0 or 1, holding ≥1 legal trump
  lead AND ≥1 legal fail lead, non-leaster.
* **C2-eligible node:** called-ace mode, non-alone, defender leads, holds ≥1
  called-suit fail AND ≥1 legal non-called-suit lead (refined 2026-07-13 while
  building the instruments, before any results: a forced all-called-suit hand
  is not a decision and trivially inflates adherence),
  `was_called_suit_played == False`, non-leaster. Primary slice: trick 0.
  Secondary: first lead opportunity at any trick.
* **Convention action:** C1 — any fail lead (best-fail per branch search);
  C2 — a called-suit lead (card chosen by policy argmax among called-suit
  fails; card choice recorded as a secondary observable, it is not part of the
  convention under test).
* **Δ convention value:** paired difference (convention branch − comparison
  branch) from one snapshot, on the three outcomes the ladder already reports:
  defender card points, leader's RL game score, defender win rate.

## Instruments (built 2026-07-13, commits 61a151ef..a866eaf8)

| Piece | Where | Notes |
|---|---|---|
| C2 self-play scanner | `sheepshead/analysis/scan_called_suit_leads.py` | /analyze-reproducible nodes + policy margins |
| C2 CRN probe | `sheepshead/analysis/called_suit_probe.py` | any agent incl. scripted anchor; same deal set as trump probe |
| E1 sweep driver | `sheepshead/analysis/convention_adherence_sweep.py` | scripted anchor = hard instrument gate |
| E2 ladder | `sheepshead/analysis/counterfactual_called_suit_leads.py` | AGREE/DISAGREE/PARTNER groups |
| E3 exception report | `sheepshead/analysis/convention_exception_report.py` | over an unconditional `counterfactual_trump_leads` run |
| E4 wrapper | `sheepshead/agent/convention_wrapper.py` + `rigorous_eval` `model.pt@c1/@c2/@c1c2` specs | C1 tricks 0-1; C2 trick 0 (provable-eligibility scope) |
| E5 SNR calc | `sheepshead/analysis/convention_learnability.py` | raw-signal bound |
| E5 critic-gap probe | `sheepshead/analysis/critic_convention_probe.py` (07-14) | limited + oracle bootstrap gaps vs realized |

Tests: `tests/test_called_suit_probe.py`, `tests/test_convention_wrapper.py`
(scripted anchors are exact by construction; wrapped violators must measure 0).

Repair note: the July web hardening removed `modelPath` from
`AnalyzeSimulateRequest`, which had silently broken the trump-lead scanner /
ladder / targeted search; fixed via `scan.set_scan_model()` (commit 055f38d3)
before any new measurements were taken.

Falsifier amendment (2026-07-13, before results): the pre-registered
"zero-sum check" for E2 is vacuous — points and scores are zero-sum across
teams by engine construction — so the AGREE sanity group and the PARTNER
mirror carry the falsification load.

E4 scope note: the wrapper forces C2 at trick 0 only (the pre-registered
primary slice, and the only trick where "called suit unled" is provable from
the per-seat observation dict); C1 masking covers tricks 0-1 (the diagnosed
leak scope), not a blanket all-trick mask.

---

# Runbook

## E1 — Convention adherence audit  [DONE for the 30M lineage]

**Question:** do checkpoints follow the conventions, and is adherence rising
with training compute? (Empirical half of Q2.)

Already run for the reference lineage — see `## Results — E1`. Re-run as
Stage-1 oracle-critic generations land (pre-registered E5 hypothesis: their C2
adherence slope should be steeper than the limited-critic lineage at matched
compute):

```bash
nice -n 19 uv run python -m sheepshead.analysis.convention_adherence_sweep \
    --deals 1000 --both-modes \
    --ckpts <stage1 gen checkpoints, oldest first, both arms> \
    --out runs/convention_optimality_202607/adherence_sweep_stage1.json \
    | tee runs/convention_optimality_202607/adherence_sweep_stage1.log
```

**Check before trusting output:** the sweep aborts by itself if the scripted
anchor is off its by-construction rates (C1 exactly 0%, C2 trick-0 exactly
100%). ~7 min per checkpoint niced.

**Self-play cross-check** (per-node policy margins; feeds E5) for any single
checkpoint:

```bash
nice -n 19 uv run python -m sheepshead.analysis.scan_called_suit_leads \
    --num-seeds 600 --quiet --model <ckpt.pt> \
    --out runs/convention_optimality_202607/called_suit_scan_<label>.json
```

## E2 — Is leading the called suit actually good? (C2, Q1)  [PILOT DONE]

**Question:** at C2-eligible nodes, is the called-suit lead better than the
best alternative lead? Counterfactual ladder, Δ = convention − alternative.

**Decision rule (pre-registered, UNCHANGED):** convention *supported* if
Δ(convention − best alternative) > 0 at ≥2σ on rung 2 AND sign-consistent on
rungs 2b and 3; *refuted* if ≤ 0 at 2σ on rung 2 and 3 agrees; otherwise
*underpowered* — report the CI and stop (no p-hacking by slicing). Secondary
read: positional heterogeneity reported descriptively, not tested. The
decision cell is the **DISAGREE** group only.

**Falsifiers — check FIRST, stop if either fails:**

* AGREE group (policy already leads called suit): rung-2 Δ must be ≥ 0. If
  negative at 2σ the forcing machinery is broken, not the convention.
* PARTNER mirror (secret partner surfaces the called card): rung-2 Δscore must
  be ≤ 0. If positive at 2σ the method is rubber-stamping leads.

**Step 1 — rung 2 over the full case set** (~3-4 h niced; pilot power calc
says ≥1100 seeds for ~90 DISAGREE cases):

```bash
nice -n 19 uv run python -m sheepshead.analysis.counterfactual_called_suit_leads \
    --num-seeds 1200 --rollouts 50 --no-search --no-belief-mc \
    --max-cases-per-group 150 \
    --out runs/convention_optimality_202607/cf_called_suit_r2.json \
    | tee runs/convention_optimality_202607/cf_called_suit_r2.log
```

**Step 2 — rungs 2b + 3 on a subsample** (offline-grade search: 4096 iters,
rollout-to-terminal and root-explore-frac 1.0 are the script defaults; verdict
is read by top@Q). Time one case first with `--max-cases-per-group 5`; then:

```bash
nice -n 19 uv run python -m sheepshead.analysis.counterfactual_called_suit_leads \
    --num-seeds 1200 --rollouts 50 --iters 4096 \
    --max-cases-per-group 40 \
    --out runs/convention_optimality_202607/cf_called_suit_r3.json \
    | tee runs/convention_optimality_202607/cf_called_suit_r3.log
```

Same `--num-seeds` and default `--subsample-seed` in both steps ⇒ step-2 cases
are a strict subset of step-1 cases (same shuffle permutation), so the rungs
are comparable case-for-case.

**Record:** copy the printed group tables into a `## Results — E2 full`
section; state the decision-rule verdict for DISAGREE; note the ISMCTS top@Q
convention fraction.

## E3 — Does "never lead trump" have real exceptions? (C1, Q1)

**Question:** over ALL C1-eligible nodes (not just where the policy prefers
trump), what fraction genuinely favor the trump lead?

**Decision rule (pre-registered, UNCHANGED):** convention *optimal-as-a-rule*
if exception rate <5% and mean Δ > 0 at 2σ (Δ oriented fail-better-positive;
the report prints trump−fail, so that is mean printed-Δ < 0); *heuristic-with-
exceptions* if exception rate ≥5% with rung-3 agreement on sampled exceptions
— in which case E4's wrapper must use the learned exception classifier, not a
blanket mask.

**Step 1 — unconditional rung-2 run + report** (`--control-ratio 1e9` keeps
every FAIL-PREF node instead of subsampling; expect the TRUMP-PREF group to be
nearly empty at tricks 0-1 — E1 showed the early-trick leak has receded — the
run is dominated by FAIL-PREF nodes, which is the point):

```bash
nice -n 19 uv run python -m sheepshead.analysis.counterfactual_trump_leads \
    --num-seeds 1200 --max-trick 1 --rollouts 50 --control-ratio 1e9 \
    --no-search --no-belief-mc \
    --out runs/convention_optimality_202607/cf_trump_unconditional.json \
    | tee runs/convention_optimality_202607/cf_trump_unconditional.log

uv run python -m sheepshead.analysis.convention_exception_report \
    runs/convention_optimality_202607/cf_trump_unconditional.json \
    --out runs/convention_optimality_202607/exception_report.json
```

**Step 2 — rung-3 verification of flagged exceptions** (the report prints
"Top exceptions" with their seeds; re-run each seed with search + belief
enabled — one game per command):

```bash
# for each exception seed S:
nice -n 19 uv run python -m sheepshead.analysis.counterfactual_trump_leads \
    --start-seed S --num-seeds 1 --max-trick 1 --rollouts 50 \
    --control-ratio 1e9 --iters 4096 \
    --out runs/convention_optimality_202607/e3_exception_seed_S.json
```

An exception is *confirmed* when the ISMCTS top@Q at that node is the trump
lead. Report the confirmed-exception rate against the 5% threshold.

## E4 — What does forcing the conventions cost at deploy? (product)

**Question:** wrapped vs raw strength under the frozen CRN gauntlet.

**Decision rule (pre-registered, UNCHANGED):** if wrapped ≥ raw (CI excludes a
loss > MDE), the convention costs nothing at deploy — ship the wrapper for
human tables regardless of Q2's answer. If wrapped < raw, that quantifies what
enforcing human convention costs.

**Calibration from the pilot:** expected |effect| ≈ 0.02 score/game — BELOW
the 1000-deal MDE (~0.07). Read the result as a *bound* ("forcing conventions
costs less than X"), not a detection.

```bash
# Called-ace mode (both wrapper rules active):
nice -n 19 uv run python -m sheepshead.analysis.rigorous_eval \
    --candidates final_pfsp_swish_ppo.pt \
                 final_pfsp_swish_ppo.pt@c1 \
                 final_pfsp_swish_ppo.pt@c1c2 \
    --anchors final_pfsp_swish_ppo.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt \
              runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt \
    --deals 1000 --partner-mode called --seed 42 \
    --out-csv runs/convention_optimality_202607/e4_gauntlet_called.csv \
    --out-plot runs/convention_optimality_202607/e4_gauntlet_called.png \
    | tee runs/convention_optimality_202607/e4_gauntlet_called.log

# JD mode (C2 is inert there — drop the @c1c2 arm):
nice -n 19 uv run python -m sheepshead.analysis.rigorous_eval \
    --candidates final_pfsp_swish_ppo.pt final_pfsp_swish_ppo.pt@c1 \
    --anchors final_pfsp_swish_ppo.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_15000000.pt \
              runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_5000000.pt \
              runs/reference_selfplay_ppo/checkpoints/swish_checkpoint_100000.pt \
    --deals 1000 --partner-mode jd --seed 42 \
    --out-csv runs/convention_optimality_202607/e4_gauntlet_jd.csv \
    --out-plot runs/convention_optimality_202607/e4_gauntlet_jd.png \
    | tee runs/convention_optimality_202607/e4_gauntlet_jd.log
```

The anchors are frozen PANEL-A (`sheepshead/analysis/panels.py`) — do not
substitute. Read the pairwise `wrapped vs raw  d=… […, …]` lines (paired over
CRN deals), not the absolute ranking.

## E5 — Episodes-to-learn numbers (Q2, quantitative half)

**Decision rule (pre-registered, UNCHANGED):** *likely learnable* if N_detect
is < ~20% of a realistic budget AND the critic gap has the right sign;
*unlikely under terminal-only* if N_detect exceeds budgets or the critic is
blind to the gap — remedies in order: oracle critic (in flight via Stage 1),
V_oracle-baseline GAE, ISMCTS-teacher distillation, deploy wrapper (E4).

**Part 1 — SNR calculator** (after E2/E3 produce their JSONs):

```bash
# C2 decision class (DISAGREE). p = violations per game; pilot measured
# 25 cases / 300 games = 0.083 (re-derive from the full run's scan line):
uv run python -m sheepshead.analysis.convention_learnability \
    runs/convention_optimality_202607/cf_called_suit_r2.json \
    --group disagree --p 0.083 --budgets 1000000,30000000

# C2 all-eligible class (p auto-derived from the self-play scan JSON):
uv run python -m sheepshead.analysis.convention_learnability \
    runs/convention_optimality_202607/cf_called_suit_r2.json \
    --group disagree \
    --scan-json runs/convention_optimality_202607/called_suit_scan_final.json

# C1 (from the E3 run; trumpPref may be near-empty at tricks 0-1 — if so,
# report "class extinct at 30M" rather than forcing a number):
uv run python -m sheepshead.analysis.convention_learnability \
    runs/convention_optimality_202607/cf_trump_unconditional.json \
    --group trumpPref --p 0.01
```

Caveat is printed by the tool: this is a raw-signal bound (no GAE/critic
variance reduction, no credit assignment).

**Part 2 — critic-gap probe** (`critic_convention_probe.py`, built
2026-07-14, commit 735b852e). Per eligible node it compares the realized
rollout gap (convention − alternative) against the GAE-relevant bootstrap
readout `r_trick + γ·V(next decision)` for the limited critic AND — on
checkpoints trained with `--critic-mode oracle` — the oracle critic
(evaluated with the exact `_fill_oracle_values` event-stream protocol).
Stage-1 intermediate checkpoints already carry `oracle_state_dict`, so the
oracle contrast is runnable NOW; note the oracle head fresh-initialized at
the 400k warmstart, so its training age is (checkpoint episodes − 400k).

```bash
# Limited critic, 30M final, C2 nodes (~30-60 min niced at these settings):
nice -n 19 uv run python -m sheepshead.analysis.critic_convention_probe \
    --ckpt final_pfsp_swish_ppo.pt --convention c2 --nodes 80 --rollouts 25 \
    --out runs/convention_optimality_202607/critic_gap_final_c2.json

# Oracle vs limited on a Stage-1 arm (same nodes budget; prints both critics):
nice -n 19 uv run python -m sheepshead.analysis.critic_convention_probe \
    --ckpt runs/league_arch_full/checkpoints/pfsp_full_checkpoint_850000.pt \
    --convention c2 --nodes 80 --rollouts 25 \
    --out runs/convention_optimality_202607/critic_gap_stage1full_c2.json

# Matched-compute LIMITED comparator for the Stage-1 read (reference lineage
# near the same episode count):
nice -n 19 uv run python -m sheepshead.analysis.critic_convention_probe \
    --ckpt runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_1000000.pt \
    --convention c2 --nodes 80 --rollouts 25 \
    --out runs/convention_optimality_202607/critic_gap_pfsp1m_c2.json

# C1 variant: --convention c1 (fail vs trump at tricks 0-1 eligible nodes).
```

**Read:** critic "sees" the convention when its bootstrap gap matches the
realized gap's sign with material magnitude; bootstrap gap ≈ 0 with a nonzero
realized gap = critic-blind. Pre-registered hypothesis: oracle-critic arms
show a larger critic gap at C2 nodes and a steeper E1 adherence slope than
the limited-critic lineage at matched compute. Caveat: the probe's node set
is generated by each checkpoint's own policy (self-play), so cross-checkpoint
comparisons are same-class-different-nodes; the paired quantity is
critic-vs-realized within a checkpoint.

---

# Results

## Results — E1 adherence sweep (2026-07-13)

Run: `convention_adherence_sweep --deals 1000 --both-modes` (CRN probe deal set,
scripted field, greedy heroes; anchor gate passed) →
`runs/convention_optimality_202607/adherence_sweep.{json,log}`.

| checkpoint | episodes | C1 trump-lead% (n) | C1 rich% | C2 adh% | C2 t0% | C2 1st% | (n) |
|---|---|---|---|---|---|---|---|
| scripted-anchor | – | 0.00 (758) | 0.00 | 69.7 | **100.0** | 70.6 | (643) |
| selfplay_100000 | 100k | 0.00 (674) | 0.00 | 52.0 | 48.0 | 51.6 | (615) |
| pfsp_1000000 | 1M | **7.80** (744) | 9.52 | 44.6 | **32.3** | 42.1 | (660) |
| pfsp_5000000 | 5M | 0.82 (854) | 1.05 | 52.9 | 45.5 | 51.1 | (736) |
| pfsp_15000000 | 15M | 0.44 (686) | 1.52 | 74.1 | 75.2 | 75.0 | (614) |
| final_pfsp (30M) | 30M | 1.00 (798) | 2.28 | 86.3 | 94.0 | 87.4 | (672) |

Self-play cross-check (600 seeds, `scan_called_suit_leads`, final model):
C2 94.4% at trick 0 (n=162), 86.5% overall (n=342), position-uniform
(85–89% across picker+1..4); eligible ≈ 0.57 nodes/game (0.27 at trick 0)
→ `called_suit_scan_final.json` (per-node policy margins for E5).

**Readings (empirical half of Q2):**

1. **Both conventions ARE being learned under terminal-only reward.** C2
   trick-0 adherence climbs 48% → 94% over the lineage and is still rising at
   30M (15M→30M: +19 points); the C1 early-trick leak has receded to 1.0%
   (probe) and ~0 in self-play greedy at tricks 0-1. The premise that
   terminal-only reward cannot find these behaviors is **not supported** for
   the convention *directions* — the open question is the residual gap
   (94→100 on C2; 1%→0 and trump-rich 2.3%→0 on C1) and its cost, which is
   E2/E3's job.
2. **Learning is non-monotone**: both conventions dip at pfsp-1M (C1 spikes to
   7.8%, C2 t0 drops to 32%) before recovering — convention adherence emerges
   late, consistent with the small-early-gap mechanism slowing (not
   preventing) acquisition. Checkpoint-picking by strength alone could ship a
   convention-poor model.
3. The historical "trick-0 trump-lead 4.8%" baseline (exit_validation, PPO@30M)
   does not reproduce in either of today's greedy contexts (probe 1.0%,
   self-play scan 0/40-seed spot-check at tricks 0-1; the 11 remaining
   self-play trump leads all sat at tricks 2-4). Measurement context matters;
   the CRN probe is the canonical adherence instrument going forward.
4. Scripted anchor's C2 69.7% overall confirms its convention is trick-0-only;
   the 30M model exceeds the scripted agent's own overall adherence (86.3%)
   by continuing to lead the suit at later first opportunities.

**Implication for the E4 wrapper:** at 94% t0 adherence the wrapper's
behavioral delta is small; its gauntlet arm mainly prices the *remaining* 6%
(and C1's trump-rich 2.3%). A null result would mean the convention gap is
already economically irrelevant at deploy; the human-perception argument then
rests on the rarity of violations rather than strength.

## Results — E2 pilot (2026-07-13, rung 2 only, R=25, 300 seeds)

**PILOT — harness validation + power calibration only; excluded from decision
rules by pre-registration.** → `cf_called_suit_pilot.{json,log}`.

| group | n | true-deal MC Δscore (SE) | read |
|---|---|---|---|
| AGREE (sanity) | 120 | **+0.272 (0.108)** ≈ +2.5σ | Δ ≥ 0 holds — forcing machinery sound |
| DISAGREE (decision) | 25 | +0.253 (0.235) ≈ +1.1σ | right sign, underpowered (expected) |
| PARTNER mirror (falsifier) | 80 | **−0.286 (0.123)** ≈ −2.3σ | fires correctly: the method can say "no" |

Coherence check: PARTNER Δpts is +5.8 for the *defenders* while the partner's
own ΔleaderScore is −0.29 — surfacing the called card early helps the other
team, exactly the convention's logic. Descriptive: AGREE conv value decays
with seat distance from picker (+0.49 at picker+1 → +0.07 at picker+4).

**Power calibration for the full run:** DISAGREE yield ≈ 25 cases / 300 seeds;
per-case SE ≈ 1.17 score at R=25. For 2σ on Δ ≈ 0.25 we need ~90 cases →
**≥ 1100 seeds** (or R=50 + ~70 cases). Note the yield will *shrink* as
adherence rises in newer models — scan seeds accordingly.

Rough per-game stake implied by the pilot (30M model): Δ ≈ 0.25 ×
p(disagree) ≈ 0.083/game ⇒ ~0.02 score/game — consistent with expecting a
small E4 wrapper effect at deploy (MDE 0.07 at 1000 deals would NOT resolve
this; the wrapper arm is a bound, not a detection).

## Results — 07-15 throttled queue (E1-b, E2 full rung 2, E3 rung 2, critic probes, E5)

Queue `queue_202607_14.sh` ran 07-14 23:55 → 07-15 05:36, all 7 steps DONE,
thread-capped alongside Stage 1. Logs/JSONs per step in
`runs/convention_optimality_202607/`.

### E2 full rung 2 (1200 seeds, R=50) — the C2 decision cell

| group | n | true-deal MC Δscore (SE) | read |
|---|---|---|---|
| AGREE (sanity) | 150 | **+0.488 (0.110)** ≈ +4.4σ | convention genuinely valuable where followed |
| DISAGREE (decision) | 71 | **+0.092 (0.138)** ≈ +0.7σ | CI ≈ [−0.18, +0.37] |
| PARTNER (falsifier) | 150 | **−0.236 (0.089)** ≈ −2.7σ | fires correctly |

**Pre-registered verdict (Q1, C2, rung 2): UNDERPOWERED / null.** Δ > 0 at 2σ
not met; ≤ 0 at 2σ not met; per the rule we report the CI and stop. Two honest
notes: (1) the pilot's Δ ≈ +0.25 shrank to +0.09 at full n — the model's
residual deviations look approximately cost-neutral, not leak-like; detecting
+0.09 at 2σ would need ~650 cases ≈ 11k seeds. (2) DISAGREE yield was 71/1200
seeds (pilot extrapolation said ~100) — adherence keeps rising, shrinking the
class. Rungs 2b/3 (step 2) remain worthwhile mainly to de-bias AGREE/PARTNER
magnitudes, not to rescue the DISAGREE cell.

### E3 rung 2 (unconditional, 810 cases) + exception report — the C1 cell

Pooled Δ (trump − fail) = **−0.490 (SE 0.046)**: the fail lead is worth ~half
a score point on the average eligible node — the convention *direction* is
supported at >10σ. But:

* **Exception rate 17.5%** (142/810; Wilson 15.1–20.3%) — far above the
  pre-registered 5% line. Per the rule the provisional verdict is
  **heuristic-with-exceptions, PENDING rung-3 agreement** on sampled
  exceptions (E3 step 2; the per-seed re-run commands are in the Runbook, top
  candidates listed in `exception_report.json`).
* Exceptions are NOT concentrated in trump-rich hands (rates by trump count:
  1T 20.5%, 2T 12.5%, 3T 20.8%, 4T 24.5%) — the pre-registered "trump-heavy"
  hypothesis is not supported; many top exceptions are LOW-DIAMOND leads over
  a fail ten/ace.
* The 10 remaining TRUMP-PREF nodes (the residual "leak") score
  Δ(trump − fail) = **−0.130**: the surviving trump leads are still *worse*
  than the best fail on average — the residual C1 leak is real but tiny
  (mix: 6/10 cases ≈ 0, mean dragged by two clearly-bad leads — seeds 266 and
  679, the June investigation's canonical sweep cases).

  > **Correction 2026-07-15:** this bullet originally read the sign backwards
  > ("the model's surviving trump leads are *better*") because the E5 tool
  > labeled the raw trump−fail Δ as "convention advantage" (only true for the
  > C2 layout). Tool label fixed; see the Reconciliation subsection below.

**Emerging synthesis (corrected):** both conventions are genuinely good on
average and the 30M model has largely internalized them. Its residual
deviations split by convention: **C2** deviations are cost-neutral (DISAGREE
null) — plausibly justified; **C1** residual trump leads remain mildly bad
(−0.13/case, matching the June investigation) but are now so rare
(~0.8% of games) that the per-game stake is ~0.001 score. The separate 17.5%
"exception" class (fail-pref nodes where rung-2 says trump was better) is a
NEW claim, hindsight-suspect, awaiting rung-3. Caveat: all rung-2 until
step-2 search runs confirm.

**E4 pre-read (corrected):** the @c1 wrapper mask binds exactly on the
TRUMP-PREF class, where trump leads are mildly bad — so expect @c1 ≥ 0 but
utterly negligible (~+0.001/game, far below the 0.07 MDE). The @c1c2 arm
similarly prices a near-null. E4 is a bound-check, not a detection.

### Reconciliation with defender_trump_lead_investigation.md (2026-07-15)

Prompted by the obvious question: the June study found search-to-terminal
overruled the policy's trump leads back to fail (+0.16/case, 2.4σ, n=101) —
how can today's deviations be "justified"? Answer, after checking:

1. **For C1 there is NO tension — today's numbers reproduce June's.** The
   trump-pref node rate matches (June: 101/9600 seeds ≈ 1.05%/game; today:
   10/1200 ≈ 0.83%/game — same population, seeds 266/679 recur byte-for-byte),
   and the per-case cost matches (today rung-2 −0.13; June belief-MC −0.19,
   search-confirmed −0.16). The "justified exceptions" phrasing was a
   sign-misreading for this class (corrected above): the residual trump
   leads are still leak-flavored, just rare and cheap in aggregate.
2. **The 17.5% exception class is a different population** the June study
   never tested for optimality: FAIL-PREF nodes (the agent already follows
   the convention; hindsight rung-2 MC says trump would have scored better).
   June's evidence actually *tempers* this class: its offline-grade FAIL-PREF
   control (n=101, search to terminal) was FLAT — a value-seeking search
   found no gains flipping fail → trump. If ~17.5% of fail-pref nodes truly
   favored trump by material margins, that control should have moved. So
   expect rung-3 (E3 step 2) to deflate the exception rate substantially;
   the 5%-line verdict stays open until then.
3. The E1 reading #3 ("4.8% doesn't reproduce") stands for that specific
   exit_validation-era figure, but note the June *investigation's* node rate
   (~1%/game) reproduces exactly — the discrepancy is with the older 4.8%
   context, not with the investigation.

### E1-b — Stage-1 intermediate checkpoints (oracle-critic league arms)

| checkpoint | episodes | C1 % (rich%) | C2 t0% (overall) |
|---|---|---|---|
| stage-1 full (oracle) | 850k | **1.15** (2.84) | 35.4 (40.8) |
| stage-1 perceiver-v2 (oracle) | 900k | **0.45** (1.92) | 29.3 (41.4) |
| reference pfsp (limited) | 1M | **7.80** (9.52) | 32.3 (44.6) |

The reference lineage's C1 spike at 1M is **absent** in both oracle arms; C2
sits in the same dip region (no acceleration yet). Confounded comparison
(league + anchor + warmstart differ, not just critic mode) — suggestive only;
the clean read is at gen end.

### E5 — critic-gap probes + SNR numbers

Critic probes (C2 nodes, 80 nodes × R=25, critic reward units; read
sign-agree and r, not raw magnitude):

| ckpt | realized gap (SE) | limited boot gap | oracle boot gap | sign-agree lim/orc | r lim/orc |
|---|---|---|---|---|---|
| final 30M | +0.032 (0.016) | +0.019 | – | 68% / – | +0.39 / – |
| stage-1 full 850k | +0.000 (0.010) | +0.013 | +0.007 | 66% / 69% | +0.42 / +0.41 |
| pfsp-1M | +0.022 (0.010) | +0.007 | – | 64% / – | +0.31 / – |

The limited critic is **partially sighted** at C2 nodes (positive correlation,
~2/3 sign agreement, attenuated magnitude) — not blind. Oracle ≈ limited at
850k: the pre-registered "larger oracle gap" hypothesis is **not yet
supported**, with two outs recorded in advance of the gen-end retest: the
oracle head is only ~450k episodes old (fresh-init at the 400k warmstart), and
the realized gap at this checkpoint's own nodes is ≈ 0 (nothing to see).

SNR calculator (raw-signal bound): C2 DISAGREE N_detect ≈ 2.4k episodes
(but its Δ̄ ≈ 0 — detecting a null); C1 TRUMP-PREF ≈ 18k episodes for a real
+0.13 convention advantage (fail − trump) at the residual nodes. Both trivial
vs budgets: the raw signal was never the binding constraint at the current
margins. The learnability bottleneck story (slow non-monotone acquisition,
E1) is about credit assignment/variance along the way, not signal existence
at 30M — the E1-b gen-end slope comparison stays the operative test.

---

# Reference

## Scheduling & budgets

Stage 1 (two oracle-critic league trainers, 4 workers each) saturates the box
through ~Jul-21. Plan:

| Phase | What | Cost | When |
|---|---|---|---|
| ~~Now~~ DONE | E1 scanner + adherence sweep (niced) | hours, forward passes | done 07-13 |
| ~~Now~~ DONE | E2 **pilot**: 300 seeds, rung 2 only, R=25 (niced) | ~1 h | done 07-13 |
| Post-Stage-1 | E2 full (Runbook §E2), E3 (Runbook §E3) | ~day total | after Jul-21 |
| Post-Stage-1 | E4 gauntlet (Runbook §E4) | ~day | after Jul-21 |
| Free-riding | E1-b + E5 critic probes on Stage 1 checkpoints | minutes each | as gens land |

Pilots exist to validate eligibility rates, harness correctness (falsifiers),
and effect-size guesses for powering the full runs — **pilot results do not
count toward decision rules.**

## Threats to validity

* **Hindsight bias in true-deal MC** — bracketed by the 2/2b/3 ladder as in the
  original investigation; conclusions require rung consistency, not rung-2
  alone.
* **Search-continuation optimism** at rung 3 — same bracket, plus the
  targeted-search lesson: prior-dominated search (default frac) silently
  rubber-stamps the policy; use frac=1.0 + top@Q at offline budgets (these are
  the E2 script's defaults).
* **Selection effects from eligibility conditioning** — eligibility is defined
  by public state + own hand only (no policy-preference filter in E2/E3), so
  the node distribution is policy-dependent only through *reaching* the lead;
  reported per checkpoint if adherence differs.
* **Policy-as-rollout-model bias** — all rollouts are the 30M-lineage policy;
  if the population never punishes convention violations, MC under-estimates
  Δ (the exploiter-league lesson). Mitigation: rung 3's search continuation,
  plus a descriptive re-run of rung 2 with the scripted agent seated as the
  other four seats (convention-aware world) — reported as a sensitivity, not a
  primary result.
* **Critic-load bug class** — any rung-3 run must be on the fixed
  value_trunk→critic_adapter path (the loader prints
  "routing the value head through the trained critic_adapter" for legacy
  checkpoints — if that note is absent on a legacy model, STOP and check).

## Deliverables

1. This document updated in place with results per experiment (house style).
2. `runs/convention_optimality_202607/` — JSON per experiment, byte-reproducible
   `(seed, partnerMode, stepIndex)` cases for `/analyze` inspection.
3. A yes/no/underpowered verdict per (convention × question) cell:

|  | Q1 optimal? | Q2 learnable terminal-only? |
|---|---|---|
| C1 never-lead-trump | E3 rung-2: optimal-on-average (10σ) but **17.5% exceptions** → provisional heuristic-with-exceptions, rung-3 pending | E1 ✅ learned (slowly); E5: signal exists, critic partially sighted |
| C2 lead-called-suit | E2 rung-2: strongly valuable where followed (+0.49, 4.4σ); residual-deviation cell **null/underpowered** (CI −0.18..+0.37) | E1 ✅ learned (slowly); E5: same |

4. If Q1=yes & Q2=no for either: the E4 wrapper verdict decides whether the
   product ships convention enforcement while training-side fixes (oracle
   critic et al.) mature.
