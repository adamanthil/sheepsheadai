# Learning System Redesign — pre-registration (July 2026)

**Status (2026-07-20): design adopted by operator; validation phases not yet
run.** Successor to the extended-league design
([Extended_League_202607.md](Extended_League_202607.md)) incorporating the
convention-erosion findings
([Convention_Erosion_202607.md](Convention_Erosion_202607.md)). Recorded
before any implementation or validation runs.

## Operator decisions (2026-07-20)

1. **No selective distillation in v1.** Pure policy-gradient with noise
   reduction, testing whether improved SNR alone lets the policy distinguish
   partner vs defender lead strategy. The search-teacher lane (τ=0.5, top@Q,
   offline budget, ESS gates — June-validated components) is held as a
   **contingency** with an explicit trigger (§Contingency).
2. **High table-level self-play share: 50–80%** (OpenAI-Five-flavored).
   Rationale: the studied conventions turned out ecology-invariant, but
   collaboration-dependent strategies (defender coordination, ALONE play)
   need consistent partners; the remainder keeps league diversity for
   state-space coverage.
3. Terminal-only reward stays (unchanged constraint; no erosion-study cell
   indicted it).

## Evidence → design map

| Decision | Evidence |
|---|---|
| Keep oracle critic; fix allocation | stratified EV: early leads 0.458 vs ~0.91 ceiling; secret-partner leads 0.187 vs ~0.85; pick 0.140 with full-deal info (allocation, not information) |
| Forced-node hygiene + decision-content weights | 32.8% of action nodes are forced (100% trick 5); zero policy gradient but pollute loss denominator + adv-norm stats |
| λ schedule (0.95 → ~0.8, gated) | λ-return at 7-decision horizon ≈ 70% MC ⇒ GAE currently absorbs almost no playout noise (σ≈1.0 at lead nodes) |
| Self-play as engine; league demoted to insurance | league lift ≈ 0 over 2M eps; PFSP behaviorally uniform (1.38:1, EMA sd ≈ noise); convention values ecology-invariant |
| Keep window+HOF (state-space coverage) | search covers action-space only; documented state-coverage failures: ALONE/defender-collaboration hole in selfplay lineages, leaster attractor, trump-lead invasion cycles |
| Keep exploiters as audits (duplicate-bridge gate) | pressure inert historically; only global-exploitability instrument; se 0.017 vs 0.045 |
| Distillation deferred, not dropped | coupling is parametric (diff-corr 0.79) but representable (warmstart 0.77/0.03; 1.25M excursion 0.73/0.10); shaped era proves clean per-node signal at these frequencies pins behavior; open question is whether PG channel gets close enough to "clean" |

Baseline reference numbers (all 2026-07-20, v2 lineage): partner-trump value
+0.237/+0.236 (400k/2M), C2 +0.113/+0.097, defender-mirror −0.22..−0.25;
partner-trump mass @2M = 0.004; playout noise at lead nodes σ ≈ 1.0 score
(0.083 reward units); trainer pooled ev_oracle ≈ 0.38 / ev_limited ≈ 0.00
(league field) vs 0.436/0.368 (self field).

## The system

### Core (unchanged)
PPO, terminal-only reward (`final_score/12` at last action), oracle critic as
GAE baseline (`--critic-mode oracle`, exploiters inherit), aux heads on
(v2-noaux remains the ballast discriminator for a future ablation).

### Loss allocation (new; flags default to historical behavior, golden-gate
checked)
- **Policy loss + entropy + advantage normalization computed over
  decision nodes only** (|valid| > 1). Forced nodes stay in GAE chains and
  episode structure; they leave the denominators and normalization stats
  (removes a ~1.5× hidden gradient dilution; zero objective change on
  decisions — masked-softmax forced nodes already have zero gradient).
- **Value loss weighted by decision content**: w = 1 for |valid| > 1,
  w = 0.25 for forced nodes (kept as bootstrap anchors). Optional per-head
  multipliers (pick / early-lead upweight) reserved for Phase A tuning if
  stratified EV does not move.
- Theory note: state-dependent loss weights = interest-weighted objective
  (emphatic weightings; Imani et al. 2018); per-state fixed points unchanged;
  ratios/clipping intact; on-policy sampling untouched.

### λ schedule (Phase B, gated)
Start 0.95. Reduce toward 0.8 ONLY after stratified EV shows trustworthy
mid-game values (gate below). Mechanism: with accurate successor values,
lead-node advantages become ~1–2-trick innovations instead of ~70%-MC
returns — the largest available SNR multiplier inside the PG channel.

### Table composition (new)
- **Table-level self-play share `p_self_table` ∈ [0.5, 0.8]** (Phase A
  default 0.65): with prob `p_self_table` ALL four opponent seats are a
  frozen copy of the current agent; otherwise all four are drawn from the
  **uniform recency window** (sliding max_past_mains, no PFSP weighting, no
  exploitation EMA) with the existing `hof_floor_prob` HOF floor.
- Replaces the per-seat `self_play_share`/PFSP mechanism. EMA bookkeeping
  (`record_vs_training_outcome`, `exploitation_*`) removed from the sampling
  path (fields may persist for schema compat).
- Exploiters: unchanged cadence as **audits** — duplicate-bridge gate,
  recorded, never a stop-rule input; HOF promotion semantics unchanged.

### Stability scaffolding (carried forward)
Gen-1 bidding-head KL anchor on warm starts (Arm-A recipe), leaster
watchdog, one-shot health verdicts (gates warn, leaster-trend halts),
algorithm changes land only at generation boundaries.

### Standing instrumentation (per generation)
Panel endpoint + duplicate-bridge h2h + stopping rule (unchanged from
Extended_League amendments); `convention_decay_curve` (3 rates);
`role_coupling_probe` (diff-corr regression check); `critic_stratified_ev`
(EV-by-stratum trend); exploitability audit.

## Validation phases (pre-registered gates; all runs niced, from the
existing v2 2M checkpoint as testbed)

**Phase A — allocation + table composition** (~100k-episode fine-tune):
loss hygiene + decision-content weights + `p_self_table = 0.65`, λ = 0.95.
*LAUNCHED 2026-07-21 (run `runs/redesign_phaseA/`, commit 8bf7a56 +
λ-flag): resumed checkpoint_2000000, window = copy of the league roster,
gen boundary 2.1M then exploiter audit; gen-3 league run killed first
(artifacts kept). First updates: ~3.2 eps/s; ev O/L dipped to 0.19/−0.49
at start (new 65%-self field mix — critic re-adapting; watch).*
- GATE A1 (primary): stratified-EV early-node movement — `play_lead_t02`
  EV_ora ≥ 0.60 (from 0.458) and `pick` EV_ora ≥ 0.25 (from 0.140).
- GATE A2 (non-inferiority): duplicate-bridge h2h vs the 2M start ≥ −0.02.
- GATE A3 (health): no leaster-trend halt; greedy gates may warn.
- Exploratory (not gates): partner-rate ratchet behavior on the decay curve;
  coupling diff-corr trend.

**Phase B — λ harvest** (fine-tune continues or restarts from A's best):
λ 0.95 → 0.85 → 0.80, stepped.
- GATE B0 (precondition): `play_follow_t02` + `play_t3plus` EV_ora ≥ 0.60
  at Phase-A endpoint.
- GATE B1: `adv_std` at lead nodes drops materially (target ≥ 30% reduction)
  with h2h non-inferiority (as A2).
- GATE B2 (the point): partner trump-lead rate reaches AND HOLDS ≥ 0.5
  (exception-aware band; NOT the subsidy-era 0.89) across ≥ 150k episodes
  with defender trump-lead ≤ 0.10 — i.e., decoupled pinning, not a shared
  excursion.

**Phase C — campaign**: winning config, fresh start from the 400k selfplay
seed via the portable orchestrator (`run_extended_league.py` recipe:
gen-1 anchored, watchdog, stop rule as amended). The 2M-testbed lineage is
NOT the campaign start (reproducibility goal).

## Contingency: selective-distillation trigger

Activate the teacher lane (KataGo-style selective supervision at early lead
nodes; τ = 0.5, top@Q, frac = 1.0, rollout-to-terminal, ESS gates,
`seat_policies` grounding on window tables) IF Phase B completes its λ step
with B0/B1 passing but **B2 fails** — i.e., SNR demonstrably improved but
partner/defender lead strategy still fails to decouple and pin. Rationale
recorded in Convention_Erosion_202607 (distillation = zero-noise
role-conditional credit + off-policy-in-action re-ignition at mass 0.004).

## Implementation notes

- All loss/sampling changes behind config flags defaulting to historical
  behavior; `capture_arch_goldens --check` + bit-exact fixture suite before
  merge; new behavior activated per-run via CLI/PFSPHyperparams.
- Trainer CSV schemas append-only (stratified adv/EV columns may be added,
  never renamed).
- `league.py` sampling simplification keeps member JSON schema readable by
  old code (EMA fields dormant).
- The live gen-3 extended-league run is unaffected until a decision is made
  about it; no mid-run changes.
