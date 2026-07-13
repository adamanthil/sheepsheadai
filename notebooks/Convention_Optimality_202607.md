# Convention Optimality — Experiment Design (July 2026)

**Status: instruments built (2026-07-13); E1 running. E2/E3 pilots pending;
full runs + E4 gauntlet wait for Stage 1 (ETA ~Jul-21).**
Written 2026-07-12 while ablation Stage 1 owns the box; see
Scheduling section for what may run concurrently.

## Instruments (built 2026-07-13, commits 61a151ef..a866eaf8)

| Piece | Where | Notes |
|---|---|---|
| C2 self-play scanner | `sheepshead/analysis/scan_called_suit_leads.py` | /analyze-reproducible nodes + policy margins |
| C2 CRN probe | `sheepshead/analysis/called_suit_probe.py` | any agent incl. scripted anchor; same deal set as trump probe |
| E1 sweep driver | `sheepshead/analysis/convention_adherence_sweep.py` | scripted anchor = hard instrument gate |
| E2 ladder | `sheepshead/analysis/counterfactual_called_suit_leads.py` | AGREE/DISAGREE/PARTNER groups |
| E3 exception report | `sheepshead/analysis/convention_exception_report.py` | over an unconditional `counterfactual_trump_leads` run (`--control-ratio 1e9`) |
| E4 wrapper | `sheepshead/agent/convention_wrapper.py` + `rigorous_eval` `model.pt@c1/@c2/@c1c2` specs | C1 tricks 0-1; C2 trick 0 (provable-eligibility scope) |
| E5 SNR calc | `sheepshead/analysis/convention_learnability.py` | raw-signal bound |

Repair note: the July web hardening removed `modelPath` from
`AnalyzeSimulateRequest`, which had silently broken the trump-lead scanner /
ladder / targeted search; fixed via `scan.set_scan_model()` (commit 055f38d3)
before any new measurements were taken.

Falsifier amendment (2026-07-13, before results): the pre-registered
"zero-sum check" for E2 is vacuous — points and scores are zero-sum across
teams by engine construction — so the AGREE sanity group and the
PARTNER mirror carry the falsification load.

E4 scope note: the wrapper forces C2 at trick 0 only (the pre-registered
primary slice, and the only trick where "called suit unled" is provable from
the per-seat observation dict); C1 masking covers tricks 0-1 (the diagnosed
leak scope), not a blanket all-trick mask.

## Motivation

Two conventions are near-universal among experienced human Sheepshead players:

* **C1 — Defenders don't lead trump.** A defender (not picker, not revealed or
  secret partner, non-leaster) never leads trump while holding a legal fail
  lead. This is the exact behavior the 30M lineage leaks (the trick-0/1
  "trump leak" investigation, `notebooks/defender_trump_lead_investigation.md`).
* **C2 — Defenders lead the called suit through.** In called-ace mode, a
  defender holding a called-suit fail leads it at the first opportunity
  (scripted agent: trick 0). Rationale: the picker is *guaranteed* to hold the
  called suit (rule-enforced: `get_callable_cards` requires a fail of the suit
  and the bury respects `get_playable_called_picker_cards`, so the picker never
  buries the last one), the secret partner must surface the called ace when the
  suit is led, so the lead (a) publicly identifies the partner immediately and
  (b) offers a void defender the chance to trump the 11-point ace.

Two distinct questions, pre-registered separately because they have different
consequences:

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

## Prior evidence (what this design must not re-derive)

* C1 on the *agent's mistake nodes* is largely settled: the 3-rung
  counterfactual ladder (`counterfactual_trump_leads.py`) found belief-MC
  −0.19 score for the trump lead, and `targeted_trump_lead_search.py` at
  offline compute (4096 iters, rollout-to-terminal, low frac + top@Q) confirmed
  fixing those leads is worth +0.16 (2.4σ, n=101) with zero control harm.
  **But both condition on TRUMP-PREF nodes** — spots where the policy argmax
  already leads trump. That answers "are the agent's trump leads mistakes?"
  (yes), not "is *never* leading trump optimal?" (E3's job).
* C2 has never been measured — no scanner stat, no counterfactual.
* Terminal-only learnability mechanism is documented: PPO-can't-learn-
  small-early-gaps (hidden-card variance swamps small early advantages;
  see deploy-search notes + oracle-critic motivation). E5 quantifies it for
  these two specific decisions instead of arguing it generally.
* `exit_validation.py` already tracks `t0_trump_lead_rate` (PPO baseline 4.8%
  greedy).

## Definitions (shared across experiments)

* **C1-eligible node:** defender (per scanner definition in
  `scan_defender_trump_leads.py`) leads on trick 0 or 1, holding ≥1 legal trump
  lead AND ≥1 legal fail lead, non-leaster.
* **C2-eligible node:** called-ace mode, non-alone, defender leads, holds ≥1
  called-suit fail AND ≥1 legal non-called-suit lead (refined 2026-07-13 while
  building the instruments, before any results: a forced all-called-suit hand
  is not a decision and trivially inflates adherence),
  `was_called_suit_played == False`, non-leaster. Primary
  slice: trick 0. Secondary: first lead opportunity at any trick.
* **Convention action:** C1 — any fail lead (best-fail per branch search);
  C2 — a called-suit lead (card chosen by policy argmax among called-suit
  fails; card choice recorded as a secondary observable, it is not part of the
  convention under test).
* **Δ convention value:** paired difference (convention branch − comparison
  branch) from one snapshot, on the three outcomes the ladder already reports:
  defender card points, leader's RL game score, defender win rate.

## Experiments

### E1 — Convention adherence audit (cheap; runs now)

**What:** extend the scanner to report, per checkpoint, greedy-mode adherence:

* C1: defender trump-lead rate at trick 0 and trick 1 (existing stat).
* C2 (new): among C2-eligible nodes, fraction where the agent leads the called
  suit — at trick 0, and at first-opportunity-any-trick. Also positional split
  (seat relative to picker) and the card chosen.
* Denominators reported alongside rates (eligibility is rare-ish; we need to
  know n before trusting any rate).

**Checkpoints:** selfplay 100k, pfsp 5M / 15M / 30M (`final_pfsp_swish_ppo.pt`),
scripted agent (must read C1 = 0%, C2 = 100% by construction — harness sanity
check), and — free evidence as they land — Stage 1's oracle-critic league gens
for both arms.

**Deliverable:** adherence-vs-training-compute curves. This is the empirical
half of Q2: if Δ>0 in E2/E3 but adherence is flat across 100k→30M, terminal-only
reward is not finding it.

**Implementation:** new `sheepshead/analysis/scan_called_suit_leads.py`
mirroring `scan_defender_trump_leads.py` (same deterministic
`simulate_game` replay path so every case reproduces in `/analyze`), plus the
per-checkpoint sweep. ~forward passes only; niced, fine to run during Stage 1.

### E2 — Called-suit lead counterfactual ladder (C2 optimality)

**What:** port the 3-rung ladder to C2-eligible nodes. Branches from one
snapshot (game + per-seat recurrent memory), differing only in the led card:

1. convention: called-suit lead;
2. policy argmax lead (when ≠ convention);
3. best other-fail lead;
4. best trump lead (when legal).

Rungs, exactly as in `counterfactual_trump_leads.py`: (2) paired true-deal MC
(R=50 stochastic rollouts/branch — the workhorse), (2b) belief-pool MC
(hindsight bracket), (3) ISMCTS at offline budget (the targeted-search
confirmed recipe: iters-to-terminal, low root_explore_frac, top@Q) on a
subsample.

**Falsifiers / controls (pre-registered):**

* *Zero-sum check:* Δ computed from the picker team's perspective must be ≈ −Δ
  defenders (accounting for the score vector); a same-sign result means the
  measurement is broken.
* *Agree-group sanity:* on nodes where policy argmax already IS the called-suit
  lead, forced-convention vs next-best-alternative must show Δ ≥ 0 under the
  policy's own rollouts; if not, the forcing machinery is suspect.
* *Partner mirror:* on matched nodes where the leader is the SECRET PARTNER
  holding called-suit fails (convention: partner does NOT lead the suit except
  via the ace/under rules — engine restricts via
  `get_leadable_called_partner_cards`), the sign should reverse. This is the
  strongest "the method can find a No" probe.

**Decision rule (Q1, C2):** convention *supported* if Δ(convention − best
alternative) > 0 at ≥2σ on rung 2 AND sign-consistent on rungs 2b and 3;
*refuted* if ≤ 0 at 2σ on rung 2 and 3 agrees; otherwise *underpowered* —
report the CI and stop (no p-hacking by slicing). Secondary read: positional
heterogeneity (leader upstream vs downstream of picker) reported descriptively,
not tested.

### E3 — Unconditional trump-lead Δ distribution (C1 optimality)

**What:** the missing complement to the TRUMP-PREF studies. Sample C1-eligible
nodes *unconditionally* (not filtered on the policy's preference), estimate per
node Δ = (best-fail − best-trump) via rung 2, with rung 3 on a subsample
enriched for Δ<0 candidates.

**Read-outs:**

* Distribution of Δ (not just the mean): the convention claim is Δ ≥ 0
  *pointwise-ish*, which a mean cannot establish.
* **Exception rate:** fraction of nodes where trump-lead is better by > ε
  (ε = 0.05 score, ~the rigorous-eval MDE scale) with per-node MC support
  (n_rollouts CI excluding 0). Characterize exceptions by hand trump count,
  trump quality (queens held), seat vs picker, fail shape.
* Pre-registered hypothesis: exceptions exist but are rare (<5% of eligible
  nodes) and cluster on trump-heavy hands where the "fail option" is a bare
  10/ace (the classic human exception "lead trump when long and the picker sits
  behind you" is folklore we may actually confirm or kill).

**Decision rule (Q1, C1):** convention *optimal-as-a-rule* if exception rate
<5% and mean Δ > 0 at 2σ; *heuristic-with-exceptions* if exception rate ≥5%
with rung-3 agreement on sampled exceptions — in which case the trump-leak
investigation's framing shifts from "agent is wrong" to "agent is wrong at
THESE nodes, right at THOSE", and E4's wrapper must use the learned exception
classifier, not a blanket mask.

### E4 — Deploy-level intervention eval (what a human would experience)

**What:** a `ConventionWrapper` around the deploy agent that overrides only
lead decisions: (i) C1 — mask trump leads for a defender when fail is legal;
(ii) C2 — force the called-suit lead at C2-eligible nodes. Three arms vs raw
`final_pfsp_swish_ppo.pt`: raw, +C1, +C1+C2.

**Harness:** `rigorous_eval.py` anchored gauntlet vs PANEL-A, CRN paired deals,
1000 deals (MDE ≈ 0.07), both partner modes (matrix already split:
`panel_a_strength_matrix_{called,jd}.csv`).

**Decision rule:** if wrapped ≥ raw (CI excludes a loss > MDE), the convention
costs nothing at deploy — ship the wrapper for human tables regardless of Q2's
answer (product fix, no retrain). If wrapped < raw, quantifies exactly what
enforcing human convention costs, which is the honest answer to "is
conventional human play optimal" at the whole-policy level.

**Implementation:** small wrapper class + a `rigorous_eval` hook to accept it
as a panel entrant. Forward passes only, but 1000 deals × panel is moderate —
schedule after Stage 1 or heavily niced.

### E5 — Terminal-only learnability quantification (analysis over E1–E3 outputs)

**What:** turn the documented "PPO can't learn small early gaps" mechanism into
numbers for these two specific decisions. From rung-2 outputs at each eligible
node class:

* effect size Δ̄ (score units) and per-rollout terminal-return SD σ at the node;
* node visitation rate p under the *training* policy (E1 scanner over self-play
  episodes — same distribution the trainer sees);
* detectability: episodes to resolve the advantage sign,
  N ≈ (z·σ/Δ̄)² / p at z=2, vs actual budgets (1M eps/gen league, 30M lineage).

**Critic-side probe:** does the limited critic even represent the gap? Paired
value-head readout at the node for convention vs violation branch (extends
`critic_calibration.py`), compared against realized rollout outcomes. Then the
same probe on Stage 1's oracle-critic arms: the oracle critic *sees* the hands,
so C2's core premise (picker holds the suit; who has the ace) is directly
observable to it. **Pre-registered hypothesis:** oracle-critic arms show a
larger critic gap at C2 nodes and a steeper E1 adherence slope than the
limited-critic 30M lineage did at matched compute.

**Decision rule (Q2):** *likely learnable* if N_detect is < ~20% of a realistic
budget AND the critic gap has the right sign; *unlikely under terminal-only* if
N_detect exceeds budgets or the critic is blind to the gap — in which case the
remedies, in order of preference (per the deploy-search teacher plan): oracle
critic (already in flight via Stage 1), V_oracle-baseline GAE, ISMCTS-teacher
distillation at these node classes, deploy wrapper (E4) as the product
backstop.

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

## Scheduling & budgets

Stage 1 (two oracle-critic league trainers, 4 workers each) saturates the box
through ~Jul-21. Plan:

| Phase | What | Cost | When |
|---|---|---|---|
| Now | E1 scanner + adherence sweep (niced) | hours, forward passes | during Stage 1 |
| Now | E2/E3 **pilots**: n≈300 seeds, rung 2 only, R=25 (niced) | ~hours | during Stage 1 |
| Post-Stage-1 | E2/E3 full: n≥1000 eligible nodes rung 2; rung 2b + rung 3 subsamples (≥100 nodes, targeted-search recipe) | days | after Jul-21 |
| Post-Stage-1 | E4 gauntlet (3 arms × 1000 deals × panel) | ~day | after Jul-21 |
| Free-riding | E1 + E5 critic probes on Stage 1 checkpoints | minutes each | as gens land |

Pilots exist to validate eligibility rates (is n achievable?), harness
correctness (falsifiers fire the right way), and effect-size guesses for
powering the full runs — **pilot results do not count toward decision rules.**

## Threats to validity

* **Hindsight bias in true-deal MC** — bracketed by the 2/2b/3 ladder as in the
  original investigation; conclusions require rung consistency, not rung-2
  alone.
* **Search-continuation optimism** at rung 3 — same bracket, plus the
  targeted-search lesson: prior-dominated search (default frac) silently
  rubber-stamps the policy; use frac=1.0 + top@Q at offline budgets.
* **Selection effects from eligibility conditioning** — eligibility is defined
  by public state + own hand only (no policy-preference filter in E2/E3), so
  the node distribution is policy-dependent only through *reaching* the lead;
  reported per checkpoint if adherence differs.
* **Policy-as-rollout-model bias** — all rollouts are the 30M-lineage policy;
  if the population never punishes convention violations, MC under-estimates
  Δ (the exploiter-league lesson). Mitigation: rung 3's search continuation,
  plus a descriptive re-run of rung 2 with the scripted agent seated as the
  three other defenders/picker opponents (convention-aware world) — reported as
  a sensitivity, not a primary result.
* **Critic-load bug class** — any rung-3 run must be on the fixed
  value_trunk→critic_adapter path (verified once at pilot start with the
  r=0.13-noise regression check from the deploy-search notes).

## Deliverables

1. This document updated in place with results per experiment (house style).
2. `runs/convention_optimality_202607/` — JSON per experiment, byte-reproducible
   `(seed, partnerMode, stepIndex)` cases for `/analyze` inspection.
3. A yes/no/underpowered verdict per (convention × question) cell:

|  | Q1 optimal? | Q2 learnable terminal-only? |
|---|---|---|
| C1 never-lead-trump | E3 | E1 + E5 |
| C2 lead-called-suit | E2 | E1 + E5 |

4. If Q1=yes & Q2=no for either: the E4 wrapper verdict decides whether the
   product ships convention enforcement while training-side fixes (oracle
   critic et al.) mature.
