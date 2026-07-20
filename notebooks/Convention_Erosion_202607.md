# Convention Erosion Under Terminal-Reward League Play (July 2026)

**Status (2026-07-20): pre-registered; instruments built; rung-1 runs queued.**
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
