# Convention Optimality — Experiment Design (July 2026)

**Status: DESIGN / pre-registration. No experiments launched.**
Written 2026-07-12 while ablation Stage 1 owns the box (ETA ~Jul-21); see
Scheduling section for what may run concurrently.

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
