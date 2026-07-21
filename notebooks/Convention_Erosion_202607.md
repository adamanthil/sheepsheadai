# Convention Erosion Under Terminal-Reward League Play (July 2026)

**Status (2026-07-20 evening): rung-1 COMPLETE — all 4 value cells + decay
curve recorded; consolidated verdict below (§Verdict). All falsifiers
passed.**
Companion to [Convention_Optimality_202607.md](Convention_Optimality_202607.md)
(which asked *are the conventions good and learned* for the 30M lineage; this
notebook asks *why did the v2 league lineage lose them*), and to
[Extended_League_202607.md](Extended_League_202607.md) /
[Architecture_Ablation_202607.md](Architecture_Ablation_202607.md) §5.11–5.12
(the run whose flat panel motivated the question). Findings feed the
league-design review (teammate sampling / table composition), which is
**blocked on this verdict**.

## Background

The 2026-07-19 hypothesis battery (§5.12) measured the perceiver-shared-v2
league lineage losing the secret-partner trump-lead convention across the
league phase: behavior rate **0.89 (400k selfplay warmstart) → 0.06 (2M league)**
(`partner_trump_lead_probe`, scripted field, CRN seed 20260719). Operator
snapshot review indicates the **C2 defender called-suit-lead** convention
degraded as well. Meanwhile **defenders-lead-fail** (avoidance of the C1
trump-lead leak) *persisted* through the same regime.

Two things changed at the same boundary (selfplay → league, episode 0 of the
league run), so the erosion is confounded:

1. **Reward regime.** The selfplay warmstart trained with shaped rewards that
   *directly subsidize* the conventions (`sheepshead/training/reward_shaping.py`:
   partner trump lead **+0.08**, defender called-suit lead **+0.10 − 0.02·trick**,
   defender trump lead **−0.06** / fail lead **+0.03**). League training is
   terminal-reward only: every subsidy vanished at once.
2. **Partner distribution.** In selfplay the training agent's partner is always
   its current self. At league tables (`league.sample_table`,
   `self_play_share = 0.15` per seat) the endogenous partner (called-ace/JD
   holder) is a roster member ~85% of the time — a snapshot with its own,
   possibly drifted, conventions.

## Decision framing (operator, 2026-07-20)

Recorded before any rung-1 results. Priority order and stakes:

1. **Q1 (primary): is the league structure specifically degrading convention
   play, such that more self-play samples would rectify it?** Bears on league
   architecture — operator is willing to change it. The 2×2 value probes give
   the *necessary condition* (positive value in the 2M self-field ⇒ league
   tables deny realizable value); the *sufficient* answer — recovery actually
   happens — is a training-dynamics claim, so the *intervention arm* below is
   promoted from contingency to the Q1 decider.
2. **Q2: are the three conventions optimal AND learnable under terminal-only
   reward, in this lineage/ecology** (separate from the 30M optimality study)?
   Bears on the terminal-only reward decision — operator is resistant to
   changing it. "Optimal" must be ecology-indexed (coordination behaviors):
   the decomposition is (a) optimal when the table follows them (≈ 400k
   field), (b) optimal in the eroded ecology (2M field), (c) gradient-
   reachable/holdable. Terminal-only reward is threatened **only** by the cell
   "value positive in the right ecology but gradient can't find/hold it even
   with self-play-heavy tables" — and even there, escalations that preserve
   terminal-only reward (variance reduction, search-teacher distillation,
   deploy-time convention wrapper) come before shaping. Note Q2 is already
   answered YES for defenders-lead-fail (shaped-installed, subsidy withdrawn,
   persisted 2M terminal league episodes); the open cases are the two
   coordination-contingent conventions.

Deploy context: the operator wants to ship a convention-following agent, so
per-convention value/learnability data is product-relevant regardless of
verdict (train it in vs wrap it on).

**Intervention arm (Q1 decider, post-rung-1):** two short fine-tunes from the
SAME 2M checkpoint under terminal reward — (A) table-level self-play forced
high (needs a new trainer knob: the current `self_play_share = 0.15` is
per-seat, so an all-self table almost never occurs), (B) standard league
tables — tracking `convention_decay_curve` stats on their checkpoints.
A-recovers/B-doesn't ⇒ Q1 yes, league restructure indicated; neither recovers
while value probes say the value is there ⇒ Q2's hard cell (gradient can't
re-select the equilibrium); both recover ⇒ erosion was transient dynamics, no
structural indictment.

## Hypotheses

- **H1 — coordination-equilibrium collapse.** Convention value routes through
  partners responding; the league's mixed-partner tables deny that value, the
  gradient (correctly, locally) removes the behavior, and erosion anywhere in
  the roster cascades (the convention's value shrinks for everyone as the pool
  erodes).
- **H2 — subsidy withdrawal + maintenance-gradient failure.** The conventions
  were installed by shaping; under terminal reward their per-node advantage is
  real but too small/rare for PPO to hold against drift (the documented
  PPO-can't-learn-small-early-gaps mechanism), regardless of who the partner
  is. Frequency-weighted version: partner-lead nodes get ~⅓ the gradient
  traffic of defender-lead nodes (1 partner vs 3 defender seats), so the rare
  convention decays first.
- **H3 — conventions not terminal-optimal in this ecology.** The removal is
  correct learning, not erosion. (Reference against: E2 measured AGREE +0.49 @
  4.4σ for C2 — but in the *30M* convention-following ecology; no partner-lead
  value measurement exists in any ecology prior to this study.)

**Known discriminating fact** (recorded before the new runs): the persistence
of defenders-lead-fail — whose value is *unilateral* (no teammate
interpretation required; C1 residual deviation cost −0.13) — rules out the
naive form of H2 ("all unsupported shaped behavior decays"): a shaped-installed
lead behavior CAN be maintained by terminal reward alone. What died are the
*coordination-contingent* behaviors. That favors H1, but the frequency-weighted
H2 predicts the same asymmetry (defender leads: 3× seats, more nodes), so the
value probes below carry the decision.

## Instruments

**I1 — decay timeline** (`sheepshead/analysis/convention_decay_curve.py`, new):
behavior rates for `partner_trump`, `c2_called_suit`, and `defender_trump`
(control) in one pass per checkpoint — hero in all 5 seats per CRN deal
(seed 20260719, same node definitions as `partner_trump_lead_probe` /
C1-C2 scans), ScriptedAgent field, tricks 0–2, deterministic. Ladder = league
50k checkpoints (episodes 50k–2.5M) + warmstart at episode 0. League timeline
landmarks: gen 1 = 0–1M (bidding-head KL anchor only — play heads free from
episode 0), gen 2 = 1M–2M (unanchored), gen 3 = 2M+ (in progress).

**I2 — convention value, per ecology** (rung 1: deterministic + paired
true-deal MC, 50 rollouts/branch; policy-rollout-grounded, no critic, no
search):

- Partner convention: `sheepshead/analysis/counterfactual_partner_trump_leads.py`
  (new; reuses the C1 `counterfactual_trump_leads` primitives/CaseResult).
  Δ = best-trump-lead − best-fail-lead at secret-partner lead nodes, tricks
  0–2, called-ace mode. Groups AGREE / DISAGREE + **DEFENDER-MIRROR falsifier**
  (same forcing at defender leads; expect Δ ≤ 0 per C1). Pooled
  AGREE+DISAGREE (scan-mix reweighted) = the ecology's unconditional
  convention value.
- C2 convention: `sheepshead/analysis/counterfactual_called_suit_leads.py`
  (existing E2 instrument) run at rung 1 with `--model` overridden.

Each value probe is **self-play of the probed checkpoint in all 5 seats**, so
"ecology" = that checkpoint's own conventions everywhere. 2×2 design:
{partner-trump, C2} × {**400k warmstart** (conventions intact),
**2M league boundary** (conventions eroded)}.

## Pre-registered decision table

Primary endpoint per cell: pooled (AGREE+DISAGREE) true-deal-MC Δscore, 2σ.

| Outcome across ecologies | Reading |
|---|---|
| Value **> 0 in 400k field, ≈ 0/− in 2M field** | **H1**: equilibrium collapsed — the convention pays only where partners still hold it. Table-composition fix (higher table-level self-play share) is the lever, but re-establishing from 0.06 needs equilibrium re-selection, not just maintenance. |
| Value **> 0 in BOTH fields** (rate still fell) | **H2**: value exists even in the eroded ecology but the terminal gradient can't hold/find it. Lever = variance reduction (oracle critic, antithetic deals) or an equilibrium-selection mechanism; table composition alone won't fix it. |
| Value **≈ 0/− in both** | **H3** for this lineage/skill: removal was correct learning; the "erosion" framing is wrong and the league design needs no convention fix. |
| Falsifier fails (AGREE < 0 or DEFENDER-MIRROR > 0 at 2σ) | Measurement suspect — stop, do not interpret; escalate to rung 2 (belief-MC / ISMCTS) per the Convention_Optimality ladder. |

I1 shape evidence (secondary): a **cliff in gens 0–1** right after subsidy
withdrawal supports H2's withdrawal half; a **gradual slide through gens 1–2**
is more consistent with H1-style cascade. (Both hypotheses allow either shape;
I1 timestamps, I2 decides.)

## Run log

All runs niced alongside the live gen-3 trainer (precedent:
Convention_Optimality runbook probes). Outputs under
`runs/convention_erosion_202607/`.

| Step | Command sketch | Status |
|---|---|---|
| I1 decay curve | `convention_decay_curve --ckpt-dir .../checkpoints --extra 0=warmstart --deals 400` | QUEUED 2026-07-20 |
| I2 partner @ 2M | `counterfactual_partner_trump_leads --model ...checkpoint_2000000.pt --num-seeds 800 --rollouts 50 --no-search --no-belief-mc` | QUEUED 2026-07-20 |
| I2 partner @ 400k | same, `--model ...warmstart_perceiver-shared-v2_400k.pt` | QUEUED 2026-07-20 |
| I2 C2 @ 2M | `counterfactual_called_suit_leads --model ...checkpoint_2000000.pt --num-seeds 800 --rollouts 50 --no-search --no-belief-mc` | QUEUED 2026-07-20 |
| I2 C2 @ 400k | same, `--model` warmstart | QUEUED 2026-07-20 |

## Results

*(appended as runs land; nothing gets buried — every completed instrument gets
its numbers recorded here even if null/awkward)*

### I1 — decay curve (DONE 2026-07-20, exit 0; `decay_curve.csv`)

52 ladder entries (scripted anchor, warmstart@0, 50k→2.5M), called mode,
400 deals each. **The "erosion" framing is wrong: the curve is not a decay.**

| Stat (league ckpts 50k–2.5M) | mean ± sd | range | anchors (scripted / warmstart) |
|---|---|---|---|
| partner_trump | 0.19 ± 0.24 | 0.00 – **0.90** | 1.00 / 0.77 |
| defender_trump (control) | 0.09 ± 0.13 | 0.00 – 0.50 | 0.00 / 0.03 |
| c2_called_suit | **0.41 ± 0.06** | 0.21 – 0.50 | 0.77 / 0.48 |

Findings:

1. **Initial collapse is immediate**: 0.766 (warmstart) → 0.057 within the
   FIRST 50k league episodes (play heads were never anchored). Consistent
   with subsidy withdrawal (+0.08/lead removed) driving the initial removal.
2. **Then oscillation, not erosion**: ≥11 excursions above 0.35 across gens
   1–2, repeatedly reaching 0.55–0.90 (550k, 750k, 1.25M, 1.6–1.7M, 2.1M,
   2.2M) and repeatedly returning to ≈0 within 50–100k episodes. The
   battery's 0.89→0.06 was a two-endpoint sample of an oscillator — episode
   2,000,000 measures exactly 0.000; 1.65M measures 0.895.
3. **The oscillation is substantially a GLOBAL trump-leading mode**:
   corr(partner_trump, defender_trump) = **+0.75** across the ladder. Some
   high-partner states discriminate partner vs defender (1.25M: 0.73 vs
   0.10 — convention-like), others don't (750k: 0.54 vs 0.50 — the C1 leak
   returned wholesale). corr(defender_trump, c2) = −0.76: trump-lead spikes
   mechanically displace called-suit leads.
4. **C2 did NOT degrade.** Stable 0.41 ± 0.06 through the entire league
   phase, ≈ its warmstart level (0.48); the shaped regime itself never got
   C2 above ~0.48 (scripted anchor 0.77). The Q2 scope narrows: the open
   convention questions are partner-trump (unstable) and *why C2 sits at
   0.4*, not a C2 erosion.
5. The trainer's greedy trump-lead gate streak at 1.6–1.7M (dismissed as
   probe noise in the 07-18 health-gate demotion) coincides with the
   1.6–1.7M excursion here — those 200-game probes were detecting a real
   behavior excursion.

Reading vs hypotheses: H1's cascade predicts decay-to-floor and staying
there — contradicted by recurrent 0.9 excursions. The shape says **nothing
pins lead behavior under the league/terminal regime**: it either cycles
(population-dynamics story: trump-leading grows, gets punished, collapses,
regrows) or drifts on near-tied logits that the greedy probe amplifies.
Subsidy withdrawal explains the *first* 50k; the value probes must now
explain the oscillation: persistent positive partner-lead value would mean
the gradient repeatedly finds and then loses it (variance/instability);
value that itself flips sign with the ecology would mean the cycling is
locally rational.

Caveats: deterministic-argmax probe amplifies near-tie policy shifts; 50k
checkpoint spacing may alias faster cycles; called mode only; behavior only
(value pending). Warmstart partner rate reads 0.766 here vs 0.89 in the
battery probe (called-mode-only + eligibility differences; same seed).

### I2 — partner convention value @ 2M ecology (DONE 2026-07-20, exit 0; `cf_partner_trump_2000k.json`)

800 seeds → 1 AGREE / 140 DISAGREE / 601 defender-mirror; caps 120/60;
rung 1 (det + 50-rollout true-deal MC).

- **DISAGREE (n=120): forcing the trump lead beats the eroded policy's fail
  lead by +0.239 ± 0.059 (4.0σ)**, trump better at 64% of nodes; trick-0
  subset +0.171 ± 0.062; defender win rate −4.2 ± 1.7 pts. Pooled
  (scan-mix reweighted): **+0.236 ± 0.058**.
- **Falsifier PASSES**: defender-mirror −0.220 ± 0.082 (right sign, ≈ C1's
  −0.13 within noise). The machinery is not rubber-stamping trump leads.
- AGREE cell empty (n=1): the 2M policy essentially never trump-leads as
  partner in its own self-play (0.7% of eligible nodes) — consistent with
  the decay curve's 0.000 at exactly 2M.

Reading: **the partner convention has large positive value in the eroded
ecology itself** — the agent's own 2M partner responds well enough that the
convention pays +0.24/opportunity NOW. Combined with I1's oscillation, this
is the "gradient repeatedly finds and loses a positive-value behavior"
signature (H2/instability), NOT H3 (removal was not correct learning) and
not the strong form of H1 (the value is not gone in the current ecology —
at least not in the self-play field). Open before verdict: (a) the 400k
cell (AGREE-sanity guardrail + cross-ecology comparison); (b) H1's live
variant — value at *league-mixture* tables (mixed partners) is NOT measured
by the self-field 2×2; if it is negative there, the league gradient locally
opposes a behavior that pays in self-play, which the intervention arm (or a
mixture-field probe variant) would expose. Hindsight-bias caveat on the
magnitude (rung 2b/3 pending per inherited limitations); the sign at 4σ
with a passing falsifier is the load-bearing part.

### I2 — partner convention value @ 400k ecology (DONE 2026-07-20, exit 0; `cf_partner_trump_400k.json`)

800 seeds → 158 AGREE / 58 DISAGREE / 664 defender-mirror (adherence
158/216 = 73%, matching the decay curve's 0.766). Rung 1 as above.

- **AGREE sanity PASSES**: +0.212 ± 0.049 (the guardrail the 2M cell
  couldn't provide). Falsifier passes again: −0.251 ± 0.069.
- DISAGREE +0.307 ± 0.066 — even the convention-following policy's
  *exceptions* are bad at these nodes (if they were skilled exceptions,
  Δ ≤ 0). The optimal adherence rate at eligible nodes is near 1.0, not
  ~0.77 — the "subsidy-inflated overshoot" concern was backwards.
- Pooled: **+0.237 ± 0.040**.

### Partner-convention row verdict (rung 1)

| Ecology | behavior rate | pooled value |
|---|---|---|
| 400k (conventions intact) | 0.73–0.77 | **+0.237 ± 0.040** |
| 2M (eroded) | 0.00–0.007 | **+0.236 ± 0.058** |

**Value is ecology-invariant; behavior is not. This is the pre-registered
H2 cell**: the convention pays +0.24/opportunity in BOTH fields, the
gradient just doesn't hold it. Two sharpenings beyond the table:

1. **H1 is effectively dead for this convention, and so is the
   "coordination-contingent" theory of its value.** If the value routed
   through partners interpreting a signal, it should have shrunk in the
   eroded ecology. It didn't move (Δ of Δ ≈ 0.00 ± 0.07). The trump lead's
   value appears to be largely *material* (developing picker-team trump
   control), like the defender-fail-lead logic — which also removes the
   main reason to expect league-mixture tables to deny it (any pool member
   as partner is from this same lineage, whose both endpoints deliver
   +0.24).
2. **Gradient arithmetic explains the oscillation without any exotic
   mechanism**: +0.24 score = 0.02 in reward units (score/12) against
   advantage σ ≈ 0.12 — a 0.17σ signal — at nodes ~4.5× rarer than
   defender leads (58+158 vs 664 per 800 games). The I1 co-movement
   (r = +0.75 partner-vs-defender trump-leading) suggests the play head
   partially couples "lead trump" across roles: the coupled feature gets
   +0.24 pressure at rare partner nodes and −0.25 pressure at common
   defender nodes → net weakly negative with noise-driven cycling. The
   1.25M excursion (0.73 vs 0.10) shows the representation CAN decouple
   them; it just isn't pinned. Testable later via role-conditioned logit
   probes across the ladder.

Implication for Q1/Q2 (partner convention): terminal-only reward is NOT
indicted — the value is present and large; the failure is maintenance at
rare nodes under gradient noise. Levers, in order of fit: variance
reduction at early/rare nodes (phase-stratified EV diagnostic already
queued), targeted gradient traffic at partner-lead nodes (upsampling /
search-distillation at those nodes), THEN table composition (helps
self-partner consistency but the value data says partner identity was
not the binding issue). Rung-2b/3 escalation optional for magnitude
de-biasing; signs and the cross-ecology null are 4σ-grade at rung 1.

### I2 — C2 value @ 2M ecology (DONE 2026-07-20, exit 0; `cf_called_suit_2000k.json`)

800 seeds → 142 AGREE / 154 DISAGREE / 176 partner-mirror (adherence at
eligible nodes ≈ 48%, tricks 0–5; consistent with the decay curve's ~0.41
at tricks 0–2). Rung 1.

- AGREE sanity passes: +0.119 ± 0.050. Partner-mirror falsifier right
  sign: −0.079 ± 0.056.
- **DISAGREE +0.076 ± 0.044 (1.7σ)** — positive but BELOW the
  pre-registered 2σ support bar on its own; trick-0 subset +0.104 ± 0.052
  (2.0σ). Count-weighted pool ≈ **+0.10 ± 0.033**.
- Reading: C2's per-node value in this lineage (~+0.10) is well below the
  partner convention's (+0.24) and below the 30M-ecology E2 AGREE (+0.49,
  rung-2-biased). The 2M policy's ~50% adherence with mildly-costly
  exceptions looks like **stable under-adoption of a weak-signal
  convention**, not erosion — matching I1 (C2 flat at 0.41 throughout).
  Consistent with the SNR frame: a 0.10-score signal (~0.008 reward units)
  is even deeper under noise than the partner signal; the difference is
  C2's behavior isn't coupled to the destabilized trump-lead feature
  (I1: corr(defender_trump, c2) = −0.76 is displacement, not coupling),
  so it parks at partial adherence instead of oscillating.

### I2 — C2 value @ 400k ecology (DONE 2026-07-20, exit 0; `cf_called_suit_400k.json`)

800 seeds → 300 AGREE / 273 DISAGREE / 283 partner-mirror (adherence ≈ 52%).
AGREE +0.097 ± 0.058; **DISAGREE +0.130 ± 0.052 (2.5σ)**; falsifier passes
(−0.122 ± 0.057). Count-weighted pool ≈ **+0.113 ± 0.039**.

## Verdict (rung 1, 2026-07-20)

Complete 2×2 (pooled true-deal-MC value; adherence at eligible nodes):

| Convention | 400k value | 2M value | 400k adherence | 2M adherence |
|---|---|---|---|---|
| partner-trump | +0.237 ± 0.040 | +0.236 ± 0.058 | 0.73 | 0.007 (oscillating 0–0.9) |
| C2 called-suit | +0.113 ± 0.039 | +0.097 ± 0.033 | 0.52 | 0.48 |

All 4 falsifier cells negative; both populated AGREE-sanity cells positive.
Value is **ecology-invariant** for both conventions; behavior differs only
in stability (partner oscillates on the role-coupled trump-lead feature;
C2 parks at half-adherence).

**Q1 — is the league structure specifically degrading convention play,
rectifiable with more self-play? NO in its strong form.** The league's
partner mixture does not deny convention value (identical value at both
lineage endpoints ⇒ any pool partner realizes it). "Erosion" itself was
mischaracterized: partner-trump is an unpinned oscillator from episode 50k
onward (subsidy withdrawal explains only the first 50k), and C2 never
degraded. Consequence: the **intervention arm is demoted** from Q1-decider
to optional — the mechanism it would test (partner identity) has been
answered by the value data. League simplification (table-level self-play
share, uniform recency window, keep exploiter/HOF) proceeds on its own
merits as operational hygiene, not as the convention fix.

**Q2 — are the conventions optimal and learnable under terminal-only
reward? Optimal: YES** (positive terminal-grounded value in every cell;
partner at 4σ, C2 at ~2.5σ pooled — C2's DISAGREE cell alone straddles the
2σ bar, rung-2b/3 escalation optional for magnitude). **Learnable:
demonstrated transiently but not reliably MAINTAINED** at current SNR: the
signals (0.10–0.24 score/node ≈ 0.008–0.02 reward units) sit under a
measured playout-noise floor of σ ≈ 1.0 at nodes comprising a few % of
steps; partner-trump additionally suffers the suspected role-coupled
lead-trump feature (net-negative frequency-weighted gradient). Defenders-
lead-fail (3rd convention) remains the existence proof that terminal
reward CAN pin a lead behavior when the signal is unilateral and the
node is common. **Terminal-only reward is not indicted by any cell.**

**Decided next steps** (cheapest-first tree, per operator's decision
framing): (1) phase-stratified EV + limited-head sanity (queued task);
(2) NEW: role-coupling logit probe across the existing ladder (partner-
vs defender-node lead-trump logits — is the coupling parametric or
correlational?); (3) league simplification (operational, decided);
(4) contingent on 1+2: critic-training emphasis at early nodes / lead-node
upsampling / targeted τ≈0.5 top@Q distillation at harvested lead nodes
(June ExIt components validated; teacher sighted at offline budget for all
three conventions pending C2 magnitude de-bias); (5) E4-style wrapper
gauntlet on any deploy candidate (deploy goal; unpriced fallback).

## Critic diagnostics (verdict next-steps 1+2; DONE 2026-07-20 21:21)

Tools: `diagnostics/critic_stratified_ev.py` (3000 self-play episodes from
the 2M ckpt, both critic heads vs empirical discounted G, per stratum) and
`role_coupling_probe.py` (54-ckpt ladder, fixed scripted-replay node set,
trump-lead prob mass, partner vs defender nodes).

### Stratified EV (`critic_stratified_ev_2000k.json`)

| stratum | n | sd(G) | EV_lim | EV_ora | ~ceiling |
|---|---|---|---|---|---|
| all | 21,331 | 0.215 | 0.368 | 0.436 | — |
| pick | 2,037 | 0.211 | 0.091 | **0.140** | — |
| play_lead_t02 | 1,320 | 0.267 | 0.399 | **0.458** | **≈0.91** |
| …secret_partner | 152 | 0.204 | 0.128 | **0.187** | **≈0.85** |
| …defender | 554 | 0.337 | 0.467 | 0.498 | — |
| play_t3plus | 6,564 | 0.215 | 0.640 | 0.711 | — |

Ceiling estimate: measured playout noise at lead nodes (I2 probes,
σ ≈ 1.0 score = 0.083 reward units) ⇒ EV ceiling 1 − (0.083/sd(G))².

1. **Early-node headroom CONFIRMED, largest exactly at the target nodes.**
   The oracle realizes ~half its ceiling at early leads (0.458 vs ~0.91)
   and barely a fifth at secret-partner leads (0.187 vs ~0.85) — the
   worst stratum in the table is the one the convention lives at. Pooled
   EV (0.436) is carried by late play (0.711). Allocation, not
   information: at pick the oracle sees the entire deal yet scores 0.140.
2. **Limited-head sanity: undertrained, and field-dependent.** At pick,
   the limited head (0.091) barely beats a one-feature linear
   hand-strength regression (0.077). Pooled self-play EV_lim = 0.368 vs
   the trainer's logged ≈ 0.00 against the league mixture: the limited
   head collapses when opponent identity (hidden to it, varying per
   table) enters the return variance. Deploy/analyze implication: its
   value displays are only meaningful for self-play-like tables.

### Role coupling (`role_coupling.json`)

**corr(levels) = 0.806, corr(first differences) = 0.790** (n=54
checkpoints, identical node set, probability mass — no argmax
amplification). The partner-node and defender-node trump-lead masses move
together step-by-step across the ladder: **parametric coupling CONFIRMED**
per the pre-registered reading. Role-decoupled credit is required; per-role
variance fixes alone cannot pin the partner convention while defender
pressure pushes the shared feature down.

### Combined implication

Both branches of the decision tree fired: the fix stack for the partner
convention is **(a) critic allocation at early nodes** (phase-weighted
value loss, early-node critic replay — large confirmed headroom) **AND
(b) role-decoupled node-local credit** (lead-node upsampling or targeted
τ≈0.5 top@Q distillation — coupling is parametric). λ reduction (currently
0.95, ~70% MC at 7-decision horizons ⇒ GAE absorbs almost no downstream
noise today) is the harvest step once mid-game EV is trustworthy. All are
trainer changes: next-run or generation-boundary amendments, validated by
the 50–100k fine-tune → stratified-EV → decay-curve loop.

## Inherited limitations (Convention_Optimality table incomplete)

The original study's pending rows (recorded 2026-07-20, before rung-1
results) constrain interpretation here:

- **E2 step 2 (rung 2b/3 de-biasing) never ran** → the 30M reference
  magnitudes (AGREE +0.49) and every rung-1/2 number in this study carry an
  unquantified hindsight correction. Differential/sign readings are robust
  (same instrument both ecologies; shared bias cancels), but any value cell
  reading **≈ 0 is not concludable at rung 1** — escalate (2b/3) before it
  enters the decision table.
- **E3 step 2 (exception frontier) never ran** → the *optimal* adherence rate
  for each convention is unknown and is NOT 1.0 (ScriptedAgent's own C2 rate
  = 0.77; the 0.89 partner rate at 400k was subsidy-inflated and plausibly
  above terminal-optimal). Intervention-arm "recovery" is therefore defined
  as returning to the exception-aware band (≈ scripted-anchor region), not
  re-hitting the shaped-era rate.
- **E4 (wrapper gauntlet) never ran** → the deploy fallback ("wrap
  conventions on at deploy") is unpriced on any lineage. Given the deploy
  goal, an E4-style `rigorous_eval` gauntlet (raw vs `@c1c2`) on the eventual
  deploy candidate is part of this study's endgame regardless of verdict.

## Caveats

- Rung 1 rollouts are the probed policy's own continuations: an eroded policy
  may under-realize a convention's value downstream (it no longer knows the
  follow-ups). A positive-in-400k/zero-in-2M result is therefore H1 *or*
  "2M forgot the continuation play"; the rung-2 escalation (belief-MC +
  ISMCTS continuation) separates those if it matters for the league decision.
- I1's scripted field is lineage-free context, not the league ecology;
  it measures behavior, never value.
- The 2×2 compares two checkpoints of ONE lineage; the 30M-ecology E2 numbers
  are cross-lineage reference only.
- Subsample caps distort group mixes; pooled estimates reweight by scanned
  counts (recorded in each JSON).
