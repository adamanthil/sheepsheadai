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
3000 deals):** the exploiter passed its gate — edge +0.106 ± 0.022
score/deal vs the frozen Phase-A endpoint (win frac 0.587, 83.3% of deals
perturbed; best screen ckpt 2140000). *Correction (operator, 2026-07-21):
an earlier draft called this "the first gate pass in program history" —
wrong; that record belongs to the old repro-run league (inert gens 1–11).
In THIS lineage the v2 gen-1 exploiter passed (+0.111 ± 0.045 vs the 1M
ckpt) and both `full`-arm exploiters passed.* Against the 2M start's own
gen-2 audit (+0.064 ± 0.042, fail), the Phase-A endpoint's +0.106 ± 0.022
is directionally worse but NOT significant (Δ ≈ +0.04 ± 0.05). The audit
is therefore only weakly consistent with degradation — A2 carries the
verdict on its own.

### PHASE A VERDICT (2026-07-21): FAIL — stop for operator review

A1 FAIL (early-node EV regressed; sole gain: secret-partner ×1.9),
A2 FAIL (−0.300 ± 0.015 vs 2M start), A3 pass-with-flag (ALONE streak
26→32%), behavior probes: no ratchet, coupling intact, fresh coupled
defender-trump excursion, C2 mild dip. Exploiter audit: gate pass
(+0.106; lineage-normal — see correction above, not additional evidence).
Per pre-registration, Phase B is NOT launched. The 2M start checkpoint remains the lineage reference;
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

### Post-Phase-A operator directives (2026-07-21)

1. **Decision-weighting machinery REVERTED** (commit af9614a): the
   `decision_weighting` flag and all loss-path machinery removed from
   PPOAgent and the trainer — a mostly-failed experiment is not worth its
   codebase complexity. Table-level sampling (`--table-self-play`) and the
   `--gae-lambda` override remain (tests moved to test_table_sampling.py).
   Goldens 34/34 bit-identical, fast suite green after removal.
2. **Offline oracle bake-off commissioned** (below) as the next
   discriminating experiment, replacing in-loop allocation probes.

### Offline oracle bake-off: shared vs per-phase experts (pre-registered
2026-07-21, before any full run; tool
`diagnostics/oracle_moe_offline.py`)

**Question:** how much of the early-node oracle EV gap is shared-capacity
interference (architecture-fixable) vs effective-sample starvation (not)?
Phase A showed in-loop allocation probes are expensive and confounded;
this measures the allocation question as supervised regression on frozen
data with zero RL-loop risk.

**Design:** 36,000 self-play episodes from the 2M league checkpoint
(stochastic acting, oracle observations, empirical discounted G — the
`critic_stratified_ev` semantics; seed 20260721), split 80/10/10
train/val/test by episode. Arms trained from scratch on identical data,
early-stopped on val MSE (patience 2, max 15 epochs, Adam 3e-4 = the
trainer's critic LR):
- `ref` — the 2M checkpoint's online-trained oracle head, eval-only
  (anchors the offline numbers to the online lineage).
- `shared` — one production-shape OracleValueNetwork.
- `moe` — five fresh OracleValueNetworks hard-routed by phase (operator
  spec): pick, partner-call, bury, play tricks 0–2, play tricks 3–5.
  Observable routing (head + trick), so per-phase heads, not learned-gate
  MoE; each expert consumes episode prefixes up to its last routed step.
  Capacity deliberately unmatched (5×): the production question is
  "beat the production critic on identical data", and oracle capacity is
  deploy-free. Precedent: backgammon phase nets (GNU BG/Snowie),
  Stockfish NNUE material buckets, Suphx per-action-type models.

**Endpoints (measurement study, not a gated phase):** per-stratum test EV
per arm, same strata as the stratified probe. Interpretation guide fixed
in advance: (i) `moe` ≳ closes half the shared-vs-ceiling gap at
pick/play_lead_t02 ⇒ interference is the dominant mechanism — justifies
wiring per-phase experts into the trainer as the next Phase-A variant;
(ii) `moe` ≈ `shared` at those strata ⇒ starvation/rarity dominates —
the search/expectation lane (contingency) moves up the queue;
(iii) `shared` (offline, converged) ≫ `ref` would additionally indicate
the ONLINE oracle is undertrained at the trainer's incidental budget,
independent of architecture. Secondary: val-MSE convergence curves
(epochs-to-best) per arm; per-expert n (partner/bury experts train on
~15–20% of episodes — their EVs carry that caveat).

### Offline oracle bake-off: RESULTS (2026-07-21; `runs/oracle_moe_offline/`)

Run exactly as pre-registered: 36,000 episodes generated (28,800 train /
3,600 val / 3,600 test; 26,421 test action rows), all three arms trained
and evaluated (`results.json`). A paired episode-level bootstrap (1,000
resamples; `bootstrap` subcommand added to the tool, output
`bootstrap.json`) supplies 95% CIs; arm deltas are paired on identical
test rows, so deal-sampling noise cancels and the delta CIs are tight.

Test EV per stratum (point [95% CI]); Δ = moe − shared (paired):

| stratum | n | ref (online 2M) | shared (offline) | moe (offline) | Δ moe−shared |
|---|---|---|---|---|---|
| all | 26,421 | 0.434 [.41,.46] | 0.338 [.30,.37] | 0.260 [.23,.29] | **−0.078 [−.09,−.06]** |
| pick | 3,143 | 0.126 [.10,.15] | 0.139 [.10,.18] | 0.001 [.00,.00] | **−0.138 [−.18,−.10]** |
| partner_call | 562 | 0.191 [.15,.23] | 0.194 [.15,.24] | 0.001 [.00,.00] | **−0.193 [−.24,−.15]** |
| bury | 1,116 | 0.222 [.18,.26] | 0.242 [.18,.30] | 0.083 [.01,.14] | **−0.159 [−.19,−.12]** |
| play_lead_t02 | 2,546 | 0.429 [.38,.47] | 0.374 [.31,.43] | 0.234 [.16,.31] | **−0.140 [−.17,−.11]** |
| … secret_partner | 366 | 0.284 [.21,.36] | 0.152 [.07,.21] | −0.135 [−.23,−.04] | **−0.286 [−.36,−.20]** |
| … partner | 1,369 | 0.368 [.31,.41] | 0.287 [.23,.34] | 0.136 [.07,.19] | **−0.151 [−.19,−.11]** |
| … defender | 811 | 0.515 [.45,.58] | 0.507 [.41,.60] | 0.423 [.31,.53] | **−0.083 [−.12,−.05]** |
| play_follow_t02 | 5,143 | 0.482 [.44,.53] | 0.328 [.30,.36] | 0.137 [.10,.18] | **−0.191 [−.22,−.16]** |
| play_t3plus | 7,689 | 0.749 [.71,.78] | 0.524 [.48,.56] | 0.563 [.53,.60] | +0.039 [+.02,+.06] |
| leaster | 6,222 | 0.176 [.14,.21] | 0.130 [.09,.16] | 0.167 [.14,.19] | +0.037 [+.02,+.06] |

**VERDICT: pre-registered outcome (ii), in amplified form.** The per-phase
experts did not close the shared-vs-ceiling gap at pick/play_lead_t02
(criterion (i) required ≈0.49 and ≈0.64; they scored 0.001 and 0.234) —
they lost to the single shared network at *every* minority stratum, with
all delta CIs excluding zero. The only strata where routing won are the
majority stratum (play_t3plus, whose expert got 86,400 routed rows and all
15 epochs) and leaster, both by a marginal +0.04. This is the mirror image
of the interference prediction, which said routing's gains should
concentrate at exactly the strata that lose the shared trunk's gradient
tug-of-war.

**Mechanism: cross-phase representation transfer outweighs interference at
this data scale.** The GRU encoder is causal, so the shared net's value at
a pick step uses exactly the same information the pick expert sees — the
comparison is information-matched by construction. What routing removes is
transfer: the shared trunk's features, learned mostly from the abundant
play rows, evidently transfer to pick/partner/bury value estimation
(shared 0.139 vs expert 0.001 on identical pick rows). Experts trade
interference relief for transfer loss, and transfer wins — despite moe
holding a deliberate 5× capacity advantage.

Caveats, none verdict-threatening:
- **Small-expert optimization stalls.** The pick expert collapsed to the
  stratum mean in epoch 1 (val MSE 0.0431 ≈ Var(G) = 0.042) and
  patience-2 stopped it at epoch 4; partner similar (val 0.122 ≈ Var
  0.134). These are stalls, not converged failures. But the play_t02
  expert trained healthily (86k rows, best at epoch 7 of 10, real val
  descent) and still lost at its own strata by −0.14 (lead) and −0.19
  (follow) — the decision-relevant comparison does not rest on the
  stalled experts. A declared re-run with higher patience/LR for the
  small experts is available but not decision-relevant.
- **Shared under-converged at cap.** Its best val MSE came at epoch 15
  (the max), so shared's numbers are lower bounds — which only widens
  the verdict.

**Secondary finding — interpretation (iii) resolved against
"undertrained online oracle", with a sharper twist.** Pooled, ref ≫
shared (−0.096 [−.120,−.068] shared−ref): 36k frozen episodes cannot
recreate the 2M-episode online head. But the deficit is entirely
concentrated in the play strata (play_t3plus −0.225, follow −0.154);
at pick (+0.013 [−.004,+.030]), partner_call (+0.003) and bury (+0.020)
the 15-epoch from-scratch fit already *matches* the online oracle. The
online head's low early-node EVs (~0.13–0.24) are therefore reproduced by
supervised regression on 230k rows — the early-node gap is a property of
the data (playout-noise floor + conditional-outcome spread), not of the
online training regime or its budget. "Train the oracle more/better" is
off the candidate list for the early nodes.

**Consequences (per the pre-registered guide):**
1. Per-phase experts are NOT wired into the trainer. If routing loses
   transfer with 230k rows and near-converged offline updates, it loses
   worse at the trainer's ~560 action rows per update.
2. Interference is not the dominant mechanism at the early strata;
   starvation + the noise floor is. The **search/expectation lane**
   (privileged-search teacher / expectation-based targets, the
   selective-distillation contingency below) **moves up the queue**.
3. The batch-scale arm (raise `--update-interval`, SNR-maintenance
   arithmetic recorded 2026-07-21 in conversation) remains the cheapest
   in-loop lever consistent with these results; not yet commissioned.

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

## Batch+λ SNR arm (pre-registered 2026-07-21, operator-approved;
LAUNCHED 2026-07-21 22:57 — `runs/league_snr_batchlam/`, historical
league path per operator decision, launch log
`runs/league_snr_batchlam_launch.log`)

**Hypothesis under test:** rare-node policy-gradient SNR is the binding
constraint on role differentiation (partner-vs-defender lead conventions
decoupling and pinning). This arm tests it at the strongest dose the
current levers compose to; a fail is therefore close to a falsification,
not an underdose (the reason the operator chose the composed arm over
batch-only at ~2 days / 1M episodes).

Dose arithmetic (2026-07-21; CORRECTED pre-launch by an empirical probe —
the trainer's transition counter counts hero ACTION rows only, ~7.05/ep
measured over 2,325 episodes, so update-interval 2048 ≈ **290 episodes**
per update, not the ~80 quoted in earlier conversation, and the
historical 256-episode minibatch cap DID bind mildly: 2 steps/epoch of
256+~34 episodes). Per-row SNR at partner-lead nodes ≈ Δ/σ = 0.24/1.0;
at 2048 (~12.5 partner-lead rows/update) an update is ~0.85σ. 8×
(16,384 ≈ 2,325 eps, ~100 partner-lead rows) ⇒ ~2.4σ; composed with
λ-harvest (σ 1.0 → ~0.6 via critic bootstrap) ⇒ ~4σ-equivalent. The
correction strengthens the falsification framing: the composed dose sits
well past the 2σ threshold. Values at
these nodes are already correct and ecology-invariant (Convention-Erosion
rung 1); the failure mode is noisy-overwrite oscillation, which per-step
averaging attacks directly. The bake-off (above) additionally certified
the critic's early-node EV as data-supported — λ bootstrapping from it is
as sound as it gets short of expectation-based targets.

**Design (single arm, matched-endpoint comparison):**
- Start: `runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt`
  — the SAME 400k selfplay seed as the v2 league, arch
  perceiver-shared-v2, critic-mode oracle, seed 42, leaster-watchdog on,
  all cadences as the v2 orchestrator invocation (main-episodes 1M,
  schedule-horizon 20M, workers 8).
- Changes vs that baseline (all flags, no code defaults touched):
  1. `--update-interval 16384` (hero action rows; ~2,325 episodes/update
     at the measured ~7.05 rows/ep).
  2. `--trainer-args "--minibatch-episodes 4096 …"` — keeps every
     optimizer step full-buffer (1024 as originally drafted would bind
     at ~2,325-episode buffers and reintroduce minibatching; noise
     between applied steps does not cancel: Adam renormalizes small
     noisy gradients and the PPO clip freezes early moves). Probe:
     full-size update = 41s / ~7–15 GB peak on 64 GB — both fine.
     **AMENDED 2026-07-22 after OOM incident (see below): now
     `--minibatch-episodes 128 --grad-accum`** — gradient accumulation
     applies the SAME full-buffer step once per epoch with per-forward
     memory bounded at 128 episodes. Step semantics of the design are
     preserved exactly; only activation memory changes.
  3. λ stays at the default 0.95 (= v2) for the first ~250k, then a
     DECLARED restart with `--gae-lambda 0.85` gated on: duplicate h2h
     vs the 400k seed ≥ −0.05 AND a recorded lead-node adv_std baseline.
     Post-step check: lead-node adv_std down ≥ 20% within 2 probes,
     else revert to 0.95 (λ-harvest inert ⇒ batch-only continuation).
     **AMENDED 2026-07-22 (declared at 100k, before the gate fires):**
     additional precondition — trainer pooled ev_ora ≥ 0.30 sustained
     over 3 consecutive updates. Mechanism: 8× batch means 8× fewer
     optimizer steps at matched episodes, and the FRESH oracle head's
     transient is step-count-limited, not sample-limited — observed
     ev_ora 0.00 at 100k vs the ~0.12 from-scratch reference, i.e. the
     ~1M-episode transient stretches toward ~a full generation.
     Stepping λ onto an immature critic would inject bootstrap bias
     exactly when the critic is least trustworthy (the bake-off
     certified the TRAINED head, not a mid-transient one). Expected λ
     step therefore lands late gen 1 or gen 2, not 250k. The h2h and
     adv_std conditions are unchanged.
  4. Exploiter re-entry amendment (operator, 2026-07-21; commits
     0db57fc/d647404): `--exploiter-full-table --exploiter-patched-ema
     0.35` in trainer-args. Gated exploiters re-enter sampling as WHOLE
     tables — one edge-weighted exploiter in all four opponent seats at
     the historical edge-scaled share (cap 0.30 × edge/0.30), so
     role/coordination exploits express against the hero regardless of
     seat assignment; expected exploiter seat mass unchanged vs per-seat
     mixing, only concentrated. Patched retirement: live outcome EMA
     < 0.35 with ≥ 200 samples demotes to past_main (checked at PPO-update
     cadence), so a patched exploit stops burning its frozen-edge share
     before the 3-generation age floor. Expression check pre-registered:
     if realized hero deficit on exploiter tables sits far below the gate
     edge, the all-exploiter field is muting the exploit (it was gated in
     a main-only field) ⇒ fall back to per-seat seating.
- Launch shape (orchestrator):
  `python -m sheepshead.training.run_extended_league --resume <400k seed>
  --run-name league_snr_batchlam --update-interval 16384 --critic-mode
  oracle --leaster-watchdog --seed 42 --trainer-args "--minibatch-episodes
  4096 --exploiter-full-table --exploiter-patched-ema 0.35"` (all other
  flags at defaults = the v2 invocation: main-episodes 1M, anchor-coeff
  1.0, panel A, min/max generations 4/12, workers 8, empty-league
  bootstrap identical to v2's `seed_checkpoints: null`).

**Incident 2026-07-22 — OOM at ~240k, root-caused and fixed.** The gen-1
trainer was SIGKILLed at ~240k episodes (≈18h in) and on every resume
(~2 min in, at the first update). Diagnosis (RSS tracing + faulthandler
stack at the spike): the first full-buffer update in oracle+anchor mode
peaks ~40 GB — the with-grad oracle forward (51 tokens/step) plus the
anchor reference forward over a max-length-padded minibatch whose
segment lengths turned heterogeneous once tables mixed (mostly
~35-event hero streams + occasional ~175-event self-table streams:
B×T_max jumped ~5×, from ~80k to ~400k padded steps — exactly at the
episode where 4-member mixed tables appeared, explaining the original
death location). The pre-launch memory probe missed it by testing
limited-mode/no-anchor/homogeneous lengths (14.2 GB). The user's
concurrent analysis job likely set the final tipping point at 18:03 but
the peak was marginal-to-fatal on 64 GB regardless. FIX: gradient
accumulation (`update(grad_accum=True)`) — row-fraction-scaled
minibatch backwards, ONE optimizer step per epoch: the design's
full-buffer step exactly, memory bounded by `--minibatch-episodes 128`.
Default-off, historical path bit-identical (test + 34/34 goldens).
Verified live: post-fix first update completed at Ep 201,430, 10 GB
peak, 4.5 eps/s. ~40k episodes lost to the 200k checkpoint on resume.

**500k kill probe (2026-07-23): PASS.** Duplicate-bridge h2h of the 500k
checkpoint vs the 400k seed: edge **−0.068 ± 0.018** (called −0.070 / jd
−0.066, 2,000 deals; `orchestrator/killprobe_500k_vs_seed.json`) — clear
of the ≤ −0.10 kill rule, and lineage-normal: v2's gen-1 ENDPOINT (1M)
measured −0.086 ± 0.013 vs this same seed on this same instrument, so
the arm at 500k is tracking the ordinary anchored-gen-1 dip, slightly
ahead of v2's pace. Run state at probe: ev_ora 0.14–0.16 (climbing),
ev_lim ≈ −0.02 (see limited-critic variance-composition note in
conversation record), leaster 0.3–0.5%, ALONE 25–32%, 4.4 eps/s.

**Oracle representation probe (2026-07-24): deterministic features NOT
at ceiling; trunk attenuates them.** Question (operator): would
deterministic aux heads (like the limited critic's) help the oracle, per
the 30M-era secret-partner-head precedent? Linear probes on the 850k
checkpoint's oracle at two taps — trunk input (post-readout 256-d) and
trunk output (which the LINEAR value head reads, so linear decodability
there is exactly the currency of value expressivity) — vs a random-init
control of the same architecture; 6,000 episodes from the frozen offline
dataset (policy-distribution caveat: dataset generated by v2 2M),
episode-split 80/20, labels computed exactly from full-info obs
(`scratchpad oracle_probe.py`, results JSON archived in conversation
record). Findings:

- Secret-partner seat (deterministic; injected into the encoder TWICE —
  context scalar + role-embedding on every opp-hand token): partner-
  present rows decode at **86.3%** at trunk input, degrading to
  **79.5%** at trunk output (trained). Random-init control: 70.8/65.4.
  So training added ~+15 pts over random but a *deterministic input
  feature* sits far from ceiling in the representation the value head
  reads — and the trunk compresses it AWAY (−6.8 pts through the trunk).
- Opponent trump counts: R² 0.84 (trunk in) → 0.73 (trunk out);
  random-init 0.82/0.79. Barely better than random features, attenuated
  by the trunk. Hero trump: ~0.99 (trivially preserved).
- Picker-team points-so-far (binding composite): R² 0.81/0.79 vs random
  0.76 — modest gain over random.

Reading: explicit injection does most of the linear work (random-init is
already high), value-loss gradients have NOT demanded crisp preservation
of role/trump structure through the bottleneck, and the trunk actively
attenuates it — the exact signature predicted by the "noisy scalar
regression under-provisions representation" mechanism from the 30M-era
policy-side precedent. Counterpoint kept honest: the value head needs
only a value-relevant 1-d projection, not full class decodability, so
sub-ceiling ≠ proof of harm. Consequence: deterministic aux heads on the
oracle (partner-seat 5-way + per-seat trump counts + team points,
attached at TRUNK OUTPUT so gradients force preservation through the
whole path) upgrade from speculative to grounded amendment candidate.
Next rung if pursued: offline test via the `oracle_moe_offline` harness
(add heads to the shared arm, retrain on the frozen dataset, paired-
bootstrap per-stratum EV vs shared baseline, partner/lead strata
primary); promote to trainer amendment only on an offline per-stratum EV
win, and only at a generation boundary.

**Oracle aux-head offline test (pre-registered 2026-07-24, launched
before results seen; harness commit 85d5730).** Operator design after
the representation probe: `shared_aux` arm = shared oracle + TWO
deterministic heads at the value-trunk output — per-seat picker-team
MEMBERSHIP (5-dim multi-label sigmoid; operator redesign 2026-07-24,
third pre-results amendment, prior launches killed within ~2 epochs:
supervise the full team split rather than classify the secret partner,
because the partition is the feature the value composes over — partner
identity is recoverable as the non-picker member, alone = picker-only,
and the pre-call window labels the true current team; this also unifies
the two heads around one membership concept, since team points is a
membership-weighted sum) and picker/defender team points-so-far
(2-dim, /120). Trump-counts head EXCLUDED by operator
choice: imposing a count-summary target could anchor the trump
representation to exactly the crude statistic we don't want; the net
should learn a richer remaining-trump-strength representation on its
own. Team-points format decision: loss MASKED on leaster and pre-pick
rows — leaster "teams" are 5 singletons, so the target degenerates to
`points_taken_rel`, which is already an explicit context input (identity
task, no binding value, and a 5-vs-2 format mismatch); alone hands kept
(picker team = picker alone, well-defined). Bury INCLUDED in the
picker-team total, as it stands at the timestamp (operator amendment
2026-07-24, before any results seen — first launch was killed ~1 epoch
in and restarted with the corrected label): the target should be the
quantity that determines who is/will be winning, and the bury's points
are the picker team's from the moment they're buried; the head's value
is forcing the transformer to assemble current team point state at
every timestamp, which translates directly to terminal score. Defender
total stays trick-based. Coefficients mirror the limited critic's
(partner 0.1, points 0.2); early stop selects on val value-MSE only
(heads are scaffolding, not the objective). Protocol otherwise
IDENTICAL to the shared arm (same dataset/splits/lr/batch/patience) so
the existing `shared.pt` is the paired baseline. Interpretation guide,
declared in advance:

- Primary: paired-bootstrap per-stratum EV, `shared_aux − shared`, at
  partner/lead strata (play_lead_t02 primary, partner_call secondary).
  CI > 0 there ⇒ representation-forcing works ⇒ trainer amendment
  candidate (gen boundary only). CI spanning zero at role strata with
  no majority-stratum harm ⇒ heads are inert offline ⇒ do NOT amend the
  trainer on speculation; the sub-ceiling probe result would then read
  as "attenuation is real but not value-binding at this data scale."
- Secondary: team-membership exact-set test accuracy ~ceiling (it had
  better be — supervised deterministic target); team-points MAE; no EV
  regression at play_t3plus (CI must not exclude zero from below).
- Caveat carried from the probe: offline-15-epochs ≠ online 850k-episode
  regime; an offline null does not rule out an online transient-speed
  benefit, but an offline win is necessary evidence before touching the
  trainer.

**Comparison protocol — matched-endpoint, NOT matched-machinery:** the
current league differs from the v2 run's (duplicate-bridge gate
instruments, this amendment), and v2's single seed makes trajectory
pairing illusory regardless. Comparisons are offline at matched episode
counts: duplicate h2h vs the 400k seed and vs v2 checkpoints
(1M/2M), stratified critic EV, role-coupling probe.

**Endpoints & rules:**
- Primary (the B2 criterion): partner trump-lead reaches AND HOLDS ≥ 0.5
  (exception-aware band) with defender ≤ 0.10 across ≥ 150k episodes —
  decoupled pinning, not a shared excursion — judged on the
  role-coupling-probe trajectory by 2M.
- Secondary: duplicate h2h vs v2 at matched episodes ≥ 0.00 − 0.02
  (non-inferiority: SNR machinery must not cost strength); oscillation
  half-life of convention excursions vs v2's telemetry.
- Kill rules: duplicate h2h vs the 400k seed ≤ −0.10 at the 500k probe;
  leaster-watchdog trip + failure to recover within 100k; greedy-health
  gate streaks (orchestrator default).
- Outcome mapping: pin ⇒ SNR hypothesis confirmed, campaign config found.
  Improved half-life without pinning ⇒ SNR necessary-not-sufficient ⇒
  selective-distillation contingency activates (its trigger condition —
  "SNR demonstrably improved but B2 fails" — is exactly this branch).
  No improvement ⇒ SNR falsified at 2σ dose ⇒ search/expectation lane.

**AMENDED OUTCOME MAPPING (2026-07-24, operator-confirmed, before any
endpoint read).** External review (Opus 5) identified a flaw in the
pre-registered dose arithmetic, verified against code and telemetry:

- The "8× ≈ 2.4σ" dose was a PER-UPDATE statement. Accumulated
  signal-to-noise over E episodes is √E·Δ/σ, invariant to batch size at
  fixed LR (checked robust under Adam normalization and PPO clip-bounded
  steps — E episodes contain √E·Δ/σ of signal, no batching extracts
  more). Batch is therefore NOT an acquisition-SNR lever; at fixed LR it
  is a displacement reducer.
- Verified: `apply_schedules` keys LR to EPISODE (train_league_ppo.py:250)
  — the arm walks the same LR decay with ~16× fewer optimizer steps
  than v2 at matched episodes (~1,550 vs ~25,000 at 900k; grad-accum =
  1 step/epoch, v2 = ~2 minibatch steps/epoch × 8× more updates; the
  review said 8× — the truth is worse). anchor_kl ~0.0045 vs v2
  ~0.008–0.012 at matched episodes confirms suppressed displacement
  (equilibrium vs anchor, not 16×, and deviating_frac 0.59 @500k shows
  real movement — but direction confirmed).
- Verified: λ gate cannot fire at the 1M boundary. ev_ora 0.084/0.176/
  0.212 @300k/600k/900k (v2: 0.26/0.40/0.39); concave trend crosses
  0.30 ≥ ~1.6M. Gen 1 delivered the batch half only.
- The frame the displacement math misses (and the arm's remaining
  design intent): ACQUISITION vs RETENTION. Stationary policy jitter
  scales with SGD temperature η·σ²/B — batch ×8 at fixed LR cuts
  equilibrium rare-node jitter ~8×, a real dose for the HOLD half of B2
  (the erosion/oscillation mechanism from convention-erosion rung-1),
  while under-dosing REACH (the new partner/defender differentiation).

New mapping, replacing the original where they conflict:
- Pin (reach AND hold) ⇒ STRONGER evidence than originally registered
  for the temperature/oscillation mechanism — linear theory says the
  arm should under-acquire, so pinning is informative, not expected.
- Hold-improvement without reach (oscillation half-life up, no
  decoupled pinning) ⇒ temperature mechanism supported on retention;
  acquisition starved ⇒ gen-3 decision is a corrected-displacement
  config (LR/epochs per GNS readout), NOT the search/expectation lane.
- B2 fail ⇒ CONFOUNDED (displacement starvation vs SNR falsity). Does
  NOT falsify SNR-as-binding-constraint and does NOT by itself activate
  the search/expectation lane.

**Aux-head offline RESULT (2026-07-24, results_aux.json +
bootstrap.json): pre-registered primary NULL; theory-predicted
substratum and global fit POSITIVE; harm checks pass.** Paired deltas
`shared_aux − shared` (95% CI):

- PRIMARY play_lead_t02 (pooled): +0.010 [−0.011, +0.032] — NULL.
  Secondary partner_call: −0.004 [−0.020, +0.011] — null.
- play_lead_t02_secret_partner: **+0.064 [+0.025, +0.103]** — POSITIVE
  (post-hoc subgroup by pre-registration standards, but it is the
  hidden-info-heaviest role stratum the probe motivation targeted: EV
  0.152 → 0.216, closing ~48% of the shared→ref gap there).
- Broad fit wins: all +0.025 [+0.011, +0.038]; play_follow_t02 +0.040
  [+0.011, +0.065]; play_t3plus +0.077 [+0.049, +0.103] (the
  no-regression harm check passes by improving). Best val MSE 0.0282
  vs shared 0.0296, both at the 15-epoch cap (protocol-matched).
- Costs: pick −0.013 [−0.023, −0.002], bury −0.028 [−0.053, −0.003] —
  small, marginal CIs.
- Secondary criteria: membership head 99.99% exact-set accuracy
  (ceiling — the representation IS forced through the trunk when
  supervised, confirming the probe's attenuation was demand-driven,
  not capacity-driven); team-points MAE 7.4 points.
- Notable: with heads, the 15-epoch/36k-episode offline oracle MATCHES
  the 2M-episode online ref at every early stratum (pick/partner_call/
  bury deltas ≈ 0.000); ref's remaining edge is confined to play
  strata where its data advantage lives.

Verdict per the pre-registered map: NOT the automatic-amendment win
(pooled-lead CI spans zero) — the trainer is not amended on this
alone. But the inert-branch reading is also excluded (global and
secret-partner-stratum CIs > 0). Status: **amendment candidate for the
next config**, where the composed inject+hold design makes the oracle
baseline's quality at partner-lead nodes directly load-bearing;
operator decision at a boundary, alongside the round-3 levers.

**Review round 2 (2026-07-24): dose arithmetic + endpoint measurement.**
Two further critiques, assessed against code and telemetry:

*Numerator (Δ = +0.237) — ACCEPTED in full.* The partner-lead gap is
max-over-trump-branches − max-over-fail-branches selected on the SAME
50 rollouts that score them (per-branch SE ≈ 0.14): winner's-curse
inflation, flagged in the original study but never de-biased (rung-2b
pre-registered, never ran). PRE-REGISTERED FIX, to run before any B2
interpretation: hold the branch selection FIXED as made by the June
data and re-evaluate the selected branches only, on fresh independent
rollouts (new seed, same nodes, same belief-MC machinery). This is a
pure evaluation-of-a-fixed-hypothesis — no selection on the new data —
and yields an unbiased estimate of the selected-branch gap at ~one
evaluation pass of compute. Every downstream σ-dose statement inherits
whatever correction results.

*Denominator (σ ≈ 1.0 score) — conclusion accepted, mechanism
corrected.* Verified: advantages ARE globally normalized before the
loss (ppo.py ~1552). But a global scalar divides the rare-node signal
and its within-node noise identically, so per-row resolvability
(σ_node/Δ)² is scale-invariant — normalization per se cannot halve the
dose. The live issue is different: 1.0 is the PLAYOUT σ, while the
optimizer resolves against the realized ADVANTAGE noise at lead rows —
never measured (shrunk by baseline EV, inflated by baseline error,
λ-bootstrap variance, within-stratum heterogeneity). Logged global
adv_std ≈ 0.15 reward ≈ 1.8 score says realized scatter overall is
~1.8× the playout assumption; if lead rows match, per-row SNR ≈ 0.13
and the dose halves. The committed GNS instrumentation measures
exactly this (B_noise at partner-lead rows = the aggregated
(σ/Δ)² question, answered from real gradients); stratified lead-row
adv_std joins the boundary baseline. Post-amendment the per-update
dose is demoted anyway; the σ question survives as the temperature
(hold-dose) calibration.

*Endpoint not measured — ACCEPTED; probe launched.* Verified: no
convention_decay_curve / role_coupling_probe output exists for the arm;
greedy_health tracks the DEFENDER t0 trump-lead only, 0.00 at all 18
probes (n ≈ 90 each). Context the critique lacked: v2 was ALSO
dead-flat 0.00 through 450k and only began oscillating at 500k
(0→27→13→0→33% …) — so the arm's flatness is discriminating only over
500k–900k, where v2 oscillated and the arm did not. Two readings:
temperature reduction holding the defender at the correct equilibrium
(B2 wants ≤ 0.10), or global pinning of the shared lead-trump feature
with the partner rate collapsed alongside (re-ignition regime, where
on-policy PG at action mass ~0.004 cannot relearn at any batch size —
the pi_gumbel search-readout finding is the contingency for exactly
that state). The partner_trump column of convention_decay_curve
distinguishes; launched over the arm's full 50k ladder + 400k seed
(400 CRN deals/ckpt, scripted field) → orchestrator/decay_curve.csv.

*Outcome map addendum (operator-confirmed):* new cell — **oscillation
eliminated but pinned at the WRONG equilibrium** (defender flat-zero
AND partner flat-zero): the hold mechanism works, the reach mechanism
is dead at this temperature, and the arm cannot deliver B2 from inside
the run; branch = search-distillation re-ignition contingency
(selective distill at lead nodes per pre-registration), not the
search/expectation lane and not a temperature increase alone.
Stability is not correctness.

**Review round 3 (2026-07-24): variance levers + direct SNR
measurement.** Assessment of three further suggestions:

- *Deal-paired/antithetic collection at train time — ACCEPTED as
  first-order; pre-registered for the NEXT config, not mid-arm.* The
  duplicate-bridge eval instrument (se 0.045→0.017, ≈7× variance) has
  no train-time analog. K-replaying each deal and subtracting the
  deal-level mean return is a valid baseline (the replays are
  independent of the gradient episode's actions given the deal ⇒
  unbiased) that removes deal-conditional variance INCLUDING the part
  the oracle critic misses — the oracle (ev ~0.4 plateau) is the
  LEARNED version of this control variate, and the empirical version
  composes with it. At fixed compute, K=2 halves unique deals but, if
  the eval variance split carries over (~6/7 deal-conditional), nets
  roughly 3× cleaner advantages per episode. Seat-rotated antithetic
  (hero in all 5 seats of one deal) is the exact train-time duplicate
  instrument. Costs: episode-generation restructuring, replay
  correlation reduces deal diversity per episode, and a mid-arm switch
  would confound the batch read — hence next-config. Cheap validation
  first: offline probe measuring realized lead-row advantage std with
  vs without deal-mean subtraction on K-replayed deals (no trainer
  change).
- *γ = 1.0 — ACCEPTED; pre-registered for the next config.* With
  purely terminal reward and ≤9-decision horizon, γ=0.99 shrinks
  early-node targets by 0.99^7 ≈ 0.93 — a systematic ~7% objective
  tilt against exactly the early nodes at issue, with no variance
  benefit at this horizon. Near-free to change (small critic re-fit
  transient) but touches the policy objective, so NOT at the gen-2
  boundary (kept measurement-only + oracle-side by design). Program-
  wide consistency required on flip: gamma also enters GAE recursion
  and every offline dataset built with agent.gamma.
- *Measure gradient SNR directly — ACCEPTED and IMPLEMENTED
  immediately* (extends the committed GNS instrument, which already
  isolates partner-lead rows with logits in hand): per update, at
  partner-lead rows, log sampled count, realized advantage mean/std
  (normalized units — the scale the loss consumes), and mean policy
  mass on trump-lead plays (legality-masked softmax over the 14
  PLAY-trump actions; the direct re-ignition-regime detector). CSV
  columns lead_rows/lead_adv_mean/lead_adv_std/lead_trump_mass.
  Concurred with the critique's process point: this instrument before
  launch would have caught both §2 errors; it is exactly the
  operator's standing cheap-gating-diagnostics-first preference and
  should have been step one.

**DECAY CURVE RESULT (2026-07-24, orchestrator/decay_curve.csv): the
arm is in the pinned-at-wrong-equilibrium branch.** 400 CRN deals per
checkpoint, called-ace mode, scripted field, seed prepended:

- partner_trump: **0.766 at the 400k seed → 0.012 by 50k**, then ~0.00–
  0.05 for 850k episodes (single excursion 0.124 @750k, back to 0.01).
- defender_trump: 0.00–0.065 throughout (B2's ≤ 0.10 satisfied — the
  greedy probe's flat zero was this, masking the partner collapse).
- c2_called_suit: stable ~0.45–0.53 across the whole run (control).

v2 comparison at matched episodes (convention_erosion decay curve):
v2 collapsed IDENTICALLY (0.766 → 0.057 @50k) — the collapse is
lineage-normal for an anchored league start from this seed (the anchor
protects bidding heads only; play conventions are exposed) — but v2
then re-ignited repeatedly at high temperature (0.21 @400k, 0.35
@500k, 0.46 @550k, 0.54 @750k) and lost it each time (the documented
oscillation). The arm never re-ignited: the low-temperature regime
suppressed the noise-driven excursions in BOTH directions, holding the
near-zero equilibrium the collapse left it in. Two corollaries: (1)
the reviewer's "PG cannot re-ignite at mass ~0.004" is too strong as
stated — v2's excursions prove noise+entropy CAN re-ignite from ~0 at
high temperature — but the excursions never stabilized, and the arm
removed exactly the noise that powered them; (2) the arm's collapse
happened in the FIRST 50k, before any batch/temperature property could
matter — the hold mechanism then preserved the wrong fixed point,
exactly as the amended outcome map's new cell describes. Stability
confirmed; correctness not achieved; B2 unreachable from inside the
run.

Per the operator-confirmed outcome map, this outcome's designated
branch is the **search-distillation re-ignition contingency**:
node-selective distillation at partner-lead nodes (pi_gumbel readout —
the instrument measured to re-ignite from a zero floor in the
search-readout study) to INJECT the convention, composed with the
low-temperature regime to HOLD it — the two mechanisms this program
has now separately validated (hold: this arm; inject: v2's excursions
show the ecology accepts the convention transiently; the oracle
counterfactuals say it is value-correct). Design decision at the gen-1
boundary is the operator's: continue gen 2 as pre-registered (mostly
confirmatory now), hold for the composed config, or relaunch gen 2
with instrumentation while the composed config is designed.

Gen-2 boundary package (pre-registered now, activated at the declared
boundary relaunch; all default-off / measurement-only before then):
optimizer-step telemetry column; gradient-noise-scale logging (global +
partner-lead stratum, from the per-minibatch gradients grad-accum
already computes) — the GNS readout, not guesswork, decides any
policy-LR/epoch correction; oracle-only extra epochs (supervised
regression on the oracle's own encoder — no PPO staleness cost, no
policy-side perturbation) to pull the λ-gate crossing forward. λ gate
itself unchanged.

## Reconstituted run: retention-first pure-PG (pre-registered 2026-07-24,
## operator-approved; the last pure-policy-gradient attempt before the
## search-teacher lane)

**Diagnosis this design answers** (decay-curve + review rounds 1–3): the
batch arm proved the low-temperature regime can HOLD an equilibrium for
850k episodes but was handed the wrong one — the seed's conventions
(partner trump-lead 0.766, defender 0.033: already B2-compliant) died in
the first ~21 updates, during the fresh-oracle burn-in (ev_ora ≈ 0) and
the shaped→terminal objective switch (verified: the 400k selfplay seed
was trained with intermediate trick rewards + leaster bonus; the league
trains terminal-only). Retention, not acquisition, is the task. The
operator chose supervised oracle pretraining + a fully UNANCHORED gen 1
over a play-head anchor: the anchor's reference forward is a real
throughput/memory cost, and with an accurate terminal baseline from
update 1 the collapse driver it would fight is largely gone; the
convention is terminal-optimal (+0.237 ± 0.040 at the seed, mechanism-
checked — branch selection is by policy logits, independent of the
evaluation rollouts, so no winner's curse; independent cross-checkpoint
reproduction +0.236 at 2M; fresh-population replication running), so
correct advantages defend it on merit.

**Δ validation RESULT (2026-07-24,
cf_partner_trump_400k_replication.json): fresh-population replication
+0.358 ± 0.039** (171 agree / 62 disagree, seeds 100000+) vs the
original +0.237 ± 0.040 — the estimate is NOT selection-inflated; the
fresh population reads HIGHER. The ~2σ between-population spread
exceeds nominal SE (node-population heterogeneity), so the working
figure is "≈ +0.24 to +0.36 score, robustly positive": the convention
the retention design protects is unambiguously terminal-optimal at the
seed.

**Design (all committed 1494902/d888062/7c413ca; flags default-off):**
1. Oracle SUPERVISED PRETRAINING: 40k frozen-seed self-play episodes
   (γ=1.0 terminal returns), official OracleValueNetwork with the two
   offline-validated aux heads (team membership 5-way multi-label +
   team points w/ bury; coefficients 0.1/0.2), trained to plateau;
   loaded via --oracle-init. Removes the burn-in window entirely.
2. Aux heads stay on ONLINE (--oracle-aux-heads): same losses in the
   oracle update path and the extra-epoch pass.
3. --oracle-extra-epochs 4: ~0.033 oracle steps/episode, restoring v2's
   oracle step rate at the 16384 interval.
4. Seat-rotated collection (--seat-rotation): each sampled deal played
   5×, hero in every seat, same table/cards — role-exposure
   equalization + train-time deal pairing. Instruments judge the
   realized variance benefit (the oracle already conditions on the
   deal).
5. γ = 1.0 (--gamma 1.0): kills the 0.99^7 ≈ 0.93 early-node tilt;
   consistent across dataset, pretraining, and trainer.
6. UNANCHORED gen 1+ (anchor-coeff 0). Bidding drift under the terminal
   objective is expected and partially correct; guarded by
   leaster-watchdog + greedy gates + the contingency below.
7. Low-temperature regime kept: --update-interval 16384,
   --minibatch-episodes 128, --grad-accum; λ gate as registered
   (pretrained oracle may satisfy ev_ora ≥ 0.30 early — follow the
   gate); exploiter full-table + patched-EMA amendments carried over.
8. Instrumentation from episode 0: --gns-log (GNS global + partner-lead,
   lead_adv_mean/std, lead_trump_mass), opt_steps; greedy probe now
   reports partner trump-lead (tricks 0–2) every 50k; decay-curve
   probes at gen boundaries. NOTE: rel-seat role-label bug fixed
   7c413ca (0-means-self misread) — historical partner/defender lead
   SUBSTRATA in offline studies were scrambled mixtures;
   secret_partner substrata were always clean.

**Tripwires & kill rules (pre-registered):**
- RETENTION: greedy partner trump-lead < 50% at BOTH the 50k and 100k
  probes ⇒ NEEDS REVIEW (retention failing; low temperature is holding
  the wrong thing again).
- MECHANISM DISCRIMINATOR: lead_adv_mean (normalized units) persistently
  negative across the first ~20 updates while lead_trump_mass falls ⇒
  the anti-convention force is systematic, not noise ⇒ stop early;
  search-teacher lane (do not burn 1M episodes).
- Bidding: leaster-watchdog trip or greedy PICK-gate streak ⇒ re-engage
  the bidding anchor at a declared restart (contingency, not a kill).
- Strength: duplicate h2h vs the 400k seed ≤ −0.10 at 500k (unchanged).
- B2 endpoint, comparison protocol, and outcome mapping otherwise as
  amended 2026-07-24 (retention framing: reach = keep what the seed
  has; hold = keep it through gen 2's ecology churn).

**LAUNCHED 2026-07-24 ~19:33 (`runs/league_retention_pg/`).** Oracle
pretraining result (oracle_init.report.json): held-out pooled EV
**0.508** — above the online oracle's ~0.40 all-time plateau and the
0.30 λ-gate threshold before the run begins; early strata pick 0.268 /
partner_call 0.398 / bury 0.402 / lead 0.477; membership head 99.99%,
team-points MAE 4.6 (γ=1.0 seed-policy data; EVs not directly
comparable to v2-2M numbers). Smoke test caught and fixed one
integration bug pre-launch (exploiter × headed-oracle checkpoints,
8a5d1b4). All banners verified at launch; first-update and 50k-probe
watchers armed.

**LAUNCH AMENDMENT (2026-07-24 ~21:15, before 5k episodes; relaunched
from scratch).** First-update telemetry showed ev_ora −0.65 → ~0.05:
the pretrained oracle was NOT expressing during the empty-league
bootstrap. Direct probe isolated the cause: the pretrained oracle
scores **EV 0.556 on hero-stream segments** (its training
distribution: 12-step, frozen distinct opponents) but **EV ≈ −0.9 on
the 60-step all-seat segments** that pure-self tables produce — the
bootstrap phase was recreating the burn-in window the design exists to
remove. Fix: **--seed-checkpoints = the 400k seed itself**, so the
league is non-empty from episode 0 and tables are mixed (hero +
frozen-seed opponents) — behaviorally identical to self-play, exactly
per the design premise, but structurally hero-stream data. Residual
p_self 0.15 self-tables stay initially OOD for the oracle; online
oracle training (8 passes/update) absorbs them. The ~4.7k episodes and
6 policy updates made against the garbage baseline were discarded with
the restart. Also recorded from the aborted run's instruments: GNS
global ≈ 7,200 rows / lead ≈ 230 rows at update 1 (single-update
estimates; the lead noise scale reading BELOW global is contrary to
the rare-node-noise assumption and worth tracking), lead_trump_mass
0.485 at the seed.

**BRAIDED-STORAGE BUG FOUND AND FIXED (2026-07-24, ea1914d; operator
directive — a sibling of the pre-30M interleaving bug).** Chasing the
oracle's non-expression exposed the root cause: the league path stored
ALL collecting seats' events as one temporally-interleaved list with a
single done flag, so every self-seat episode became ONE braided multi-
perspective recurrent segment. Consequences, all verified: (1)
train/act mismatch — act-time memories are per-player, the update
forward ran one memory across perspective switches, giving NON-UNIT
PPO ratios at theta_old for self-seat rows (every league run in the
lineage carried this silently; the selfplay trainer that produced the
seeds always stored per-player streams); (2) the pretrained oracle
scored EV −0.9 on braided segments vs 0.556 on coherent ones; (3) the
~175-event braided outliers drove the July OOM padding blowup. Fix:
store_events_by_seat groups by player_id, one coherent stream + done
flag per seat (segments now ≤ 10 actions + own-perspective observation
frames). Test coverage: per-seat segmentation, braided control,
hero-only byte-equivalence, and ratio-at-theta_old ≈ 1 on a self-table
episode (<1e-4; the coherence property the bug broke). Interim
workarounds (--self-play-share 0) retired; relaunched with DEFAULT
self-play share 0.15 + the 4-seed pool + full config. Two earlier
same-day relaunches (single-seed without-replacement fallback; p_self
composition) are recorded in the run log; all pre-fix data discarded.

**Braided-bug impact assessment (2026-07-24).** Exposure: league-path
training only (incl. exploiters); ~48% of episodes ⇒ **~67% of
training ROWS** braided at historical p_self 0.15 (braided episodes
carry more rows); 100% during empty-league bootstraps; **~90% of rows
under Phase A's p_self_table 0.65**. Clean: selfplay trainer (always
per-player), all offline diagnostics (frozen distinct opponents ⇒ hero
streams), all eval instruments (act-time per-player memories).
Consequences for the record:
- Phase A's gate-fail verdict is CONFOUNDED: the design change also
  raised braided rows 67%→90%, and the observed oracle disruption
  (0.38→0.21, flat) is what a 90% frame-hopping diet predicts
  independent of decision weighting. Design-vs-bug split unrecoverable
  from that run; verdict demoted to "failed under confound."
- The chronic ev_limited ≈ 0/negative pattern across all league runs
  is plausibly primarily this bug (critic TRAINED on braided
  update-forward features, EVALUATED on coherent rollout values). The
  earlier two-mechanism decline analysis gains braiding as the leading
  third candidate; the fixed run is the discriminating test.
- The arch-ablation "league lift ≈ zero over selfplay start" finding is
  now confounded with the league trainer's ~67% corrupted-context
  gradients (selfplay trainer was clean) — reinterpretation open.
- Batch-arm decay-curve facts stand (measurement-side); the 0–50k
  collapse window gains braided bootstrap gradients as a third
  co-factor. Aborted-launch GNS readings (7,200/230) VOID — measured
  on braided buffers.
- Unaffected: MoE bake-off, aux-head study, oracle pretraining, Δ
  counterfactuals, h2h/decay measurements. Transfer caveat: bake-off
  "ref" was a braided-trained head evaluated on clean streams.

**Reproduction commands (verbatim; final relaunch verified against the
live process command line).**

Oracle pretraining dataset (40k frozen-seed self-play episodes, γ=1.0
terminal returns; ~1.5h at 6 workers):

```bash
uv run python -m sheepshead.analysis.diagnostics.oracle_moe_offline generate \
  --ckpt runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt \
  --episodes 40000 --workers 6 --gamma 1.0 --seed 20260725 \
  --out runs/oracle_pretrain_400k/dataset.pt \
  > runs/oracle_pretrain_400k/generate.log 2>&1
```

Supervised pretraining of the official headed OracleValueNetwork (γ is
read from the dataset; aux coefficients default 0.1/0.2; early stop on
val value-MSE, best epoch 20 of 24):

```bash
uv run python -m sheepshead.analysis.diagnostics.oracle_moe_offline pretrain \
  --dataset runs/oracle_pretrain_400k/dataset.pt \
  --max-epochs 25 --patience 3 --seed 20260725 \
  --out runs/oracle_pretrain_400k/oracle_init.pt \
  > runs/oracle_pretrain_400k/pretrain.log 2>&1
```

Seed pool (4 copies of the 400k warmstart, because `sample_table`
samples members without replacement — one member can fill only one
seat):

```bash
mkdir -p runs/retention_seeds
for s in a b c d; do
  cp runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt \
     runs/retention_seeds/seed400k_$s.pt
done
```

Run launch (final relaunch, post-fix, default self-play share; the
glob in --seed-checkpoints is quoted — the orchestrator expands it):

```bash
nohup uv run python -m sheepshead.training.run_extended_league \
  --resume runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt \
  --seed-checkpoints "runs/retention_seeds/*.pt" \
  --run-name league_retention_pg \
  --critic-mode oracle --update-interval 16384 --num-workers 8 --seed 42 \
  --leaster-watchdog --anchor-coeff 0 \
  --trainer-args "--minibatch-episodes 128 --grad-accum --gamma 1.0 \
    --oracle-aux-heads --oracle-init runs/oracle_pretrain_400k/oracle_init.pt \
    --seat-rotation --gns-log --oracle-extra-epochs 4 \
    --exploiter-full-table --exploiter-patched-ema 0.35" \
  > runs/league_retention_pg_launch.log 2>&1 &
```

**First-update telemetry, fixed run (2026-07-25, updates 1–4 / ep
5,829).** The run is healthy; two headline readings and one prediction
miss:
- **ev_limited: −0.65 → −0.50 → −0.05 → +0.147.** First league run in
  the lineage where the limited critic crosses POSITIVE within four
  updates — early confirming evidence that the chronic ev_lim disease
  was the braided bug (the free discriminator identified in the impact
  assessment).
- **ev_oracle: 0.11 → 0.19 → 0.21 → 0.24**, climbing ~+0.03/update.
  NOT the predicted instant 0.4–0.5: the update-1 pre-training reading
  (0.11) shows a residual distribution gap between generate-path
  pretraining data and live trainer buffers, absorbed steadily by the
  online passes (8 + 4 extra/update). No burn-in disaster (pre-fix
  launches: −0.65/−0.86, pinned). WATCH: should cross ~0.4 by ~update
  10; a plateau below 0.30 blocks the λ-gate and warrants diagnosis.
- GNS global: 33k / 22k / 45k rows vs the 16,384-row update — B_noise
  ≈ 1.5–3× batch, so the batch is NOT oversized (noise-dominated
  regime; the hold-dose is not wasted compute). Replaces the VOID
  braided reading (7,200).
- GNS lead: BLANK by guard, and that is itself the measurement — the
  paired estimator cannot distinguish the mean gradient at ~350
  lead rows/update from zero (g2 ≤ 0 guard), consistent with
  lead_adv_mean ≈ 0.01 vs lead_adv_std ≈ 0.62 (normalized units):
  per-row SNR at partner-lead nodes ~1–2%. The convention's protection
  here is temperature, not signal — the design premise, now measured.
- Tripwire status: lead_adv_mean −0.07 → +0.007 → +0.114 (NOT
  persistently negative — mechanism discriminator quiet);
  lead_trump_mass 0.53–0.57, stable at seed level. leaster ≤0.4%,
  sampled pick ~20%, opt_steps 4/update as designed, anchor_kl blank
  (unanchored confirmed), 6.4 eps/s. 50k greedy probe (first
  partner_trump_lead_rate gate) expected ~2h in.

**Success reading:** partner ≥ 0.5 AND defender ≤ 0.10 held through 2M
with the ordinary strength trajectory ⇒ retention-first on-policy PG is
sufficient; the search teacher stays shelved. Retention holds through
gen 1 but breaks when exploiters/ecology churn arrives ⇒ hold-dose
insufficient against ecology pressure ⇒ selective distillation at lead
nodes composes next. Retention fails in the first 100k despite the
accurate baseline ⇒ the collapse force is systematic ⇒ search-teacher
lane with the mechanism identified.

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
