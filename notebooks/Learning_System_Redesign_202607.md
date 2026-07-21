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
*ATTEMPT 1 (2026-07-21 early, run `runs/redesign_phaseA/`, commit
8bf7a56): resumed checkpoint_2000000, NO anchor (launch error — the
design's own scaffolding section requires the Arm-A bidding anchor on
warm starts). **COLLAPSED into the leaster attractor within 50k
episodes**: 2.025M pick 14%/leaster 21% (lineage-normal) → 2.05M pick
0%/leaster 72.5%, greedy gates firing on PICK < 15% AND play-head logit
spread < 0.5. Killed at ~2.05M; log kept. **Mechanism identified**: the
decision flag originally filtered the per-head ENTROPY means to decision
rows — but the entropy coefficients were tuned against the historical
all-rows (diluted) scale, so effective entropy pressure rose ~1.5× and
pushed the play head toward uniform (exactly the failing gate), dragging
picker EV down into the pass/leaster spiral; the missing anchor removed
the bidding-head brake. Head-balanced PG gradients were NOT amplified by
the flag (the total/count normalization cancels the dilution — verified
arithmetically). Fixes (commit 2ceb778): entropy stays at the all-rows
scale under the flag (+ regression test); anchor made mandatory for all
warm-started fine-tunes in this program, per the design's own rule.
Lesson recorded: the greedy-gate warnings + quarter-mark monitor caught
the collapse in one wall-clock hour — the scaffolding works when used.*

*ATTEMPT 2 (2026-07-21, run `runs/redesign_phaseA_r2/`): same config +
`--anchor-coeff 1.0` (ref = the 2M resume ckpt), fresh league-window
copy (attempt 1's collapsed 2.05M snapshot discarded with its run dir's
league).*
- GATE A1 (primary): stratified-EV early-node movement — `play_lead_t02`
  EV_ora ≥ 0.60 (from 0.458) and `pick` EV_ora ≥ 0.25 (from 0.140).
- GATE A2 (non-inferiority): duplicate-bridge h2h vs the 2M start ≥ −0.02.
- GATE A3 (health): no leaster-trend halt; greedy gates may warn.
- Exploratory (not gates): partner-rate ratchet behavior on the decay curve;
  coupling diff-corr trend.

### Attempt-2 results (2026-07-21; run completed 2.0M → 2.1M cleanly)

**GATE A1: FAIL** (`critic_stratified_ev_2100k.json`, matched instrument:
3000 self-play episodes, seed 20260720, vs the 2000k baseline probe).
EV_oracle by stratum, 2M → 2.1M:

| stratum | 2M | 2.1M | gate |
|---|---|---|---|
| play_lead_t02 | 0.458 | **0.368** | ≥ 0.60 FAIL |
| pick | 0.140 | **0.036** | ≥ 0.25 FAIL |
| play_lead_t02_secret_partner | 0.187 | **0.361** | — (near-doubled) |
| play_lead_t02_partner | 0.382 | 0.228 | — |
| play_lead_t02_defender | 0.498 | 0.438 | — |
| partner_call | 0.222 | 0.152 | — |
| bury | 0.242 | 0.159 | — |
| play_follow_t02 | 0.373 | 0.372 | — |
| play_t3plus | 0.711 | 0.693 | — |
| leaster | 0.225 | 0.166 | — |
| pooled | 0.436 | 0.384 | — |

Limited head dropped in every stratum as well (pooled 0.368 → 0.305);
trainer-pooled `ev_limited` went **negative** in late updates (−0.3..−0.6).
The single mover in the intended direction is the rarest and most
program-relevant stratum (secret-partner leads), consistent with the
value-loss decision weighting shifting critic capacity toward rare decision
nodes — but the broad EV regression says the 100k fine-tune left the critic
mid-transient (field shift from p_self_table 0.65 + reweighted value loss),
or worse, that the allocation change degrades the critic at this budget.
Per pre-registration: **A1 fail ⇒ no Phase B launch; stop for operator
review** (A2/A3 + exploratory probes still recorded below for the review).

**GATE A2: FAIL** (`h2h_duplicate_2100k_vs_2000k.json`; duplicate-bridge
instrument, 2×2000 deals, seed 42): edge **−0.300 ± 0.015** score/hand vs
the 2M start (gate: ≥ −0.02; called −0.287, jd −0.313 — modes agree).
A ~20σ strength regression in 100k episodes. Corroborated by the noisy
in-trainer anchored eval vs final_pfsp_swish_ppo: −0.23 ± 0.13 at the 2M
league checkpoint → −0.707 ± 0.16 at the Phase-A endpoint. Trainer-batch
pooled ev_oracle also dropped 0.38 → ~0.21 within the FIRST 10k episodes
and stayed flat all 100k (no recovery slope), with ev_limited going
negative (−0.3..−0.5). For calibration: the from-scratch oracle head took
~1.0–1.2M episodes to plateau (0.30 by ~600k), so the 100k window likely
could not complete any re-convergence transient — but the flat (not
recovering) EV plus the large strength drop reads as genuine disruption,
not a benign transient passing through.

**Exploratory behavior probes** (`decay_curve_r2.csv`,
`role_coupling_r2.json`; same instruments/seeds as the erosion study):
no partner ratchet — partner_trump rate 0.000 @2.05M → 0.013 @2.1M (low
phase of the known oscillation). Role coupling INTACT: partner/defender
node masses rose together ~10× between the two r2 checkpoints (0.0025 →
0.026 partner, 0.0067 → 0.079 defender) — a fresh SHARED excursion, echoed
behaviorally by greedy defender trump-lead 0.000 → 0.097 (above the
0.03–0.08 historical band). C2 dipped mildly (0.392 → 0.333, ~2σ below
the 0.41 ± 0.06 band; n=219, watch-only). Net: 100k of Phase-A config did
not decouple roles or start a ratchet — expected at this budget (the
design's mechanism for decoupling is Phase B λ-harvest on top of a
TRUSTED critic; A1 shows the critic is not yet trustworthy post-change).

**GATE A3: PASS with flag.** No leaster-watchdog halt; training leaster
stable 21–24%, pick 12–15%, picker_avg +1.07 → +1.21, anchor_kl
0.007–0.024 throughout. Flag: greedy ALONE rate exceeded the 20% warn gate
on both probes and is rising — 26.2% @2.05M → 31.7% @2.1M (lineage-normal
band 18–27%); greedy PICK 21.4%, leaster 27.5%, play-spread 0.84 at the
boundary (all normal). In-trainer anchored eval vs final_pfsp_swish_ppo:
−0.707 ± 0.164 (n=300; noisy instrument, recorded for continuity).

**Exploitability audit (gen-1 exploiter, 50k eps + duplicate-bridge gate,
3000 deals):** the exploiter **PASSED its gate — the first gate pass in
program history** — edge +0.106 ± 0.022 score/deal vs the frozen Phase-A
endpoint (win frac 0.587, 83.3% of deals perturbed; best screen ckpt
2140000). Historically exploiters were inert against healthy checkpoints
(League_Run_Review gens 1–11), so a passing exploiter is an independent
confirmation that the endpoint is degraded, consistent with A2.

### PHASE A VERDICT (2026-07-21): FAIL — stop for operator review

A1 FAIL (early-node EV regressed; sole gain: secret-partner ×1.9),
A2 FAIL (−0.300 ± 0.015 vs 2M start), A3 pass-with-flag (ALONE streak
26→32%), behavior probes: no ratchet, coupling intact, fresh coupled
defender-trump excursion, C2 mild dip. Exploiter audit: endpoint
exploitable (+0.106, first-ever gate pass). Per pre-registration, Phase B
is NOT launched. The 2M start checkpoint remains the lineage reference;
the Phase-A endpoint is not a candidate for anything.

Candidate mechanisms for the regression (not yet discriminated):
1. **Critic disruption from the reweighted value loss** — trainer-batch
   pooled ev_oracle fell 0.38 → ~0.21 within 10k eps and stayed flat
   (no recovery slope in 100k); ev_limited went negative (−0.3..−0.5).
   For scale: the from-scratch oracle took ~1.0–1.2M eps to plateau, so
   100k could not complete a re-convergence transient even if benign.
2. **Advantage-scale shift from decision-only normalization** — raw
   adv_std fell (all 0.119 → 0.086; pick 0.124 → 0.056 — pick rows are
   genuine decisions, so this is not the mechanical forced-zeroing
   effect), changing the effective policy step size.
3. **Opponent-diversity loss** (p_self_table 0.65) — least likely to
   produce −0.30 in 100k on its own, but plausibly compounds 1–2.

Discriminating experiments for review (cheap → expensive):
- **Offline critic-fit bake-off (no RL loop, zero risk):** frozen
  self-play dataset from the 2M ckpt; fit (a) current shared oracle,
  (b) decision-weighted variant, (c) per-phase expert heads (precedent:
  backgammon phase nets, NNUE material buckets, Suphx per-action
  models) to convergence; compare stratified EV. Directly measures the
  interference/allocation gap and tests the value-loss reweighting in
  isolation, with convergence-time constants as a bonus.
- **Single-change 100k arms:** table-composition-only (no decision
  weighting) and decision-weighting-only (historical PFSP field),
  each gated on A2 non-inferiority alone.
- Longer Phase A (300–500k, stratified probe every ~100k) only if a
  single-change arm looks healthy.

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
