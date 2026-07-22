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
