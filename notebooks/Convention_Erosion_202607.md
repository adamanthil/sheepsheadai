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
