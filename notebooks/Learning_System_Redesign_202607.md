# Learning System Redesign — program notebook (July 2026)

Successor to the extended-league design
([Extended_League_202607.md](Extended_League_202607.md)) incorporating the
convention-erosion findings
([Convention_Erosion_202607.md](Convention_Erosion_202607.md)).

> **Restructure note (2026-07-28):** reorganized from the original
> chronological append-only form into the program-arc structure below, for
> readability. No findings, numbers, dates, commits, or pre-registration
> terms were altered; superseded intermediate designs are condensed with
> their reasoning kept. The original chronological form — including the
> as-written pre-registration entries — is preserved verbatim in git
> history (through commit a4a40c9).

## 0. Current status (2026-07-28)

**Live run:** `runs/league_retention_pg` — retention-first pure-PG league
(§7), gen 3 of ≥4 running UNDER THE ADAPTIVE-ENTROPY CONTROLLER (activated
at the gen-2→3 boundary per operator GO, §7.10), the healthiest run in the
lineage: partner trump-lead convention SHARPENED to the 84–98% band (every
prior league arm lost it in the first 50k). ABSOLUTE panel trajectory
(PANEL-A mixed-anchor table): seed −0.077 → gen 1 +0.0425 → gen 2
**+0.1323 [+0.1019, +0.1629]**; per-generation gains +0.120 then +0.090
(both ≥5σ; mild deceleration, normal as the hero pulls away from a fixed
field). h2h gen2-vs-gen1 +0.101 ± 0.013. Both exploiter gates failed (the
main is not PPO-exploitable at budget; gen 2 −0.184 ± 0.016, worse for
the exploiter than gen 1).

**Root-cause context:** two league-path training bugs (braided storage +
last-actor-only terminal rewards, §6) were found and fixed 2026-07-24/25.
They retroactively explain the chronic ev_limited ≈ 0 disease, confound
the Phase-A fail (§3) and the historical "league lift ≈ zero" finding, and
are the reason this run behaves differently from everything before it.

**Next actions (armed):**
- Verify the bumpless-switch-on prediction across gen 3: no measurable
  behavior change at controller attach; α tracks the schedule's value;
  ent_norm/softband/approx_kl/lr_actor columns populate. First play-target
  step only on a flat boundary verdict (gen 3 was NOT flat-checked yet —
  first candidate is the gen-3→4 boundary).
- Gen-3 exploiter gate at 3M: first potential per-seat pressure test under
  the §7.7 seating amendment (gens 1–2 exploiters failed their gates, so
  the amendment is still untested by live pressure).
- B2 through gen-3 churn; C2 trajectory (recovered to 43–51% at 1.85–2.0M
  from the 40.9% dip at 1M; §7.9 watch bounds unchanged).
- Standing watch items and armed contingencies: appendix A.

## 1. Program timeline

| Date (2026) | Event | Outcome | § |
|---|---|---|---|
| 07-20 | Program pre-registered; design adopted by operator | — | 2 |
| 07-21 | Phase A attempt 1 | Collapsed ~50k (entropy-scale bug + missing anchor) | 3.1 |
| 07-21 | Phase A attempt 2 | A1 FAIL, A2 FAIL (−0.300); stop for review | 3.2–3.3 |
| 07-21 | Decision-weighting reverted (af9614a); bake-off commissioned | — | 3.4 |
| 07-21 | Offline oracle bake-off (shared vs per-phase experts) | Outcome (ii): transfer beats routing; search lane moves up | 4.1 |
| 07-21 | Batch+λ SNR arm launched (`runs/league_snr_batchlam`) | — | 5 |
| 07-22 | OOM at ~240k | Root-caused; grad-accum fix | 5.3 |
| 07-23 | Batch arm 500k kill probe | PASS (−0.068, lineage-normal) | 5.4 |
| 07-24 | Oracle representation probe; aux-head offline test | Primary null; secret-partner +0.064; amendment candidate | 4.2–4.3 |
| 07-24 | External reviews 1–3; outcome map amended | Batch ≠ acquisition lever; acquisition-vs-retention frame | 5.5 |
| 07-24 | Batch-arm decay curve | Pinned at WRONG equilibrium; B2 unreachable in-run | 5.6 |
| 07-24 | Retention-first run pre-registered + launched | — | 7 |
| 07-24 | **Braided-storage bug** found + fixed (ea1914d) | Reinterprets the whole league lineage | 6.1 |
| 07-25 | **Terminal-reward attachment bug** found + fixed (45570ff) | Completes the bug pair | 6.2 |
| 07-25 | Retention relaunch with both fixes | Pretrained oracle expresses from update 1 | 7.3 |
| 07-25 | λ-sweep probe; λ policy amendment | λ stays 0.95; node-selective λ 0.5–0.7 armed as contingency | 7.5 |
| 07-25 | 50k/100k retention gates | BOTH PASSED — falsification window survived | 7.6 |
| 07-25 | 500k kill probe | +0.0885 ± 0.0139 — kill rule INVERTED | 7.6 |
| 07-26 | Gen-1 exploiter gate | FAILED (exploiter loses; main not PPO-exploitable) | 7.7 |
| 07-26 | Gen-1 boundary + per-seat seating amendment applied | Panel +0.043 CI>0; h2h +0.078; continue | 7.8 |
| 07-27 | Retention config adopted as trainer/orchestrator defaults (51a81ac); C2 baseline | — | 7.9 |
| 07-28 | Adaptive entropy Phase 1 + backfill | Entropy-inflated-pick hypothesis REFUTED; play = only binding head | 8.3–8.4 |
| 07-28 | Adaptive entropy Phase 2 controller; operator GO; default-on | Activation rides gen-2→3 relaunch | 8.5–8.8 |
| 07-28 | Legacy args stripped (exploiter-full-table, self-play-share, table-self-play) | `sample_table` single-mode | 9 |
| 07-28 | ScriptedAgent C2 extended to all tricks (19e4028) | Instrument version boundary for scripted-field probes | 7.9 |
| 07-28 | Orchestrator crash at gen-2 endpoint eval | torch venv-swap under long-running process; no data lost | 7.10 |
| 07-28 | Gen-2 boundary: panel +0.1323 CI>0, h2h +0.101; exploiter gate FAILED (−0.184) | Continue; **adaptive entropy ACTIVATED for gen 3** | 7.10 |

## 2. Original pre-registration (2026-07-20)

Recorded before any implementation or validation runs; design adopted by
operator.

### 2.1 Operator decisions

1. **No selective distillation in v1.** Pure policy-gradient with noise
   reduction, testing whether improved SNR alone lets the policy
   distinguish partner vs defender lead strategy. The search-teacher lane
   (τ=0.5, top@Q, offline budget, ESS gates — June-validated components)
   is held as a **contingency** with an explicit trigger (§2.5).
2. **High table-level self-play share: 50–80%** (OpenAI-Five-flavored).
   Rationale: the studied conventions turned out ecology-invariant, but
   collaboration-dependent strategies (defender coordination, ALONE play)
   need consistent partners; the remainder keeps league diversity for
   state-space coverage. *(Retrospective: this mechanism was Phase A's
   `p_self_table`; reverted after Phase A and stripped from the codebase
   2026-07-28, §9.)*
3. Terminal-only reward stays (unchanged constraint; no erosion-study cell
   indicted it).

### 2.2 Evidence → design map

| Decision | Evidence |
|---|---|
| Keep oracle critic; fix allocation | stratified EV: early leads 0.458 vs ~0.91 ceiling; secret-partner leads 0.187 vs ~0.85; pick 0.140 with full-deal info (allocation, not information) |
| Forced-node hygiene + decision-content weights | 32.8% of action nodes are forced (100% trick 5); zero policy gradient but pollute loss denominator + adv-norm stats |
| λ schedule (0.95 → ~0.8, gated) | λ-return at 7-decision horizon ≈ 70% MC ⇒ GAE currently absorbs almost no playout noise (σ≈1.0 at lead nodes) |
| Self-play as engine; league demoted to insurance | league lift ≈ 0 over 2M eps; PFSP behaviorally uniform (1.38:1, EMA sd ≈ noise); convention values ecology-invariant |
| Keep window+HOF (state-space coverage) | search covers action-space only; documented state-coverage failures: ALONE/defender-collaboration hole in selfplay lineages, leaster attractor, trump-lead invasion cycles |
| Keep exploiters as audits (duplicate-bridge gate) | pressure inert historically; only global-exploitability instrument; se 0.017 vs 0.045 |
| Distillation deferred, not dropped | coupling is parametric (diff-corr 0.79) but representable (warmstart 0.77/0.03; 1.25M excursion 0.73/0.10); shaped era proves clean per-node signal at these frequencies pins behavior; open question is whether PG channel gets close enough to "clean" |

*(Retrospective note: the "league lift ≈ 0" and "pressure inert" rows are
now known to be confounded by the §6 bugs.)*

Baseline reference numbers (all 2026-07-20, v2 lineage): partner-trump
value +0.237/+0.236 (400k/2M), C2 +0.113/+0.097, defender-mirror
−0.22..−0.25; partner-trump mass @2M = 0.004; playout noise at lead nodes
σ ≈ 1.0 score (0.083 reward units); trainer pooled ev_oracle ≈ 0.38 /
ev_limited ≈ 0.00 (league field) vs 0.436/0.368 (self field).

### 2.3 The system as designed

**Core (unchanged):** PPO, terminal-only reward (`final_score/12` at last
action), oracle critic as GAE baseline (`--critic-mode oracle`, exploiters
inherit), aux heads on (v2-noaux remains the ballast discriminator for a
future ablation).

**Loss allocation (new; flags defaulted to historical behavior,
golden-gate checked; REVERTED after Phase A, §3.4):**
- Policy loss + entropy + advantage normalization computed over decision
  nodes only (|valid| > 1). Forced nodes stay in GAE chains and episode
  structure; they leave the denominators and normalization stats (removes
  a ~1.5× hidden gradient dilution; zero objective change on decisions —
  masked-softmax forced nodes already have zero gradient).
- Value loss weighted by decision content: w = 1 for |valid| > 1, w = 0.25
  for forced nodes (kept as bootstrap anchors). Optional per-head
  multipliers reserved for Phase A tuning if stratified EV does not move.
- Theory note: state-dependent loss weights = interest-weighted objective
  (emphatic weightings; Imani et al. 2018); per-state fixed points
  unchanged; ratios/clipping intact; on-policy sampling untouched.

**λ schedule (Phase B, gated):** start 0.95; reduce toward 0.8 ONLY after
stratified EV shows trustworthy mid-game values. Mechanism: with accurate
successor values, lead-node advantages become ~1–2-trick innovations
instead of ~70%-MC returns — the largest available SNR multiplier inside
the PG channel. *(Superseded for the retention run by the §7.5 λ policy.)*

**Table composition (new; REVERTED after Phase A, stripped §9):**
table-level self-play share `p_self_table` ∈ [0.5, 0.8] (Phase A default
0.65): with prob `p_self_table` ALL four opponent seats are a frozen copy
of the current agent; otherwise all four drawn from the uniform recency
window (no PFSP weighting, no exploitation EMA) with the existing
`hof_floor_prob` HOF floor. Exploiters unchanged as audits.

**Stability scaffolding (carried forward):** gen-1 bidding-head KL anchor
on warm starts (Arm-A recipe), leaster watchdog, one-shot health verdicts
(gates warn, leaster-trend halts), algorithm changes land only at
generation boundaries.

**Standing instrumentation (per generation):** panel endpoint +
duplicate-bridge h2h + stopping rule (unchanged from Extended_League
amendments); `convention_decay_curve` (3 rates); `role_coupling_probe`
(diff-corr regression check); `critic_stratified_ev` (EV-by-stratum
trend); exploitability audit.

### 2.4 Pre-registered validation phases and gates

All runs niced, from the existing v2 2M checkpoint as testbed.

**Phase A — allocation + table composition** (~100k-episode fine-tune):
loss hygiene + decision-content weights + `p_self_table = 0.65`, λ = 0.95.
- GATE A1 (primary): stratified-EV early-node movement — `play_lead_t02`
  EV_ora ≥ 0.60 (from 0.458) and `pick` EV_ora ≥ 0.25 (from 0.140).
- GATE A2 (non-inferiority): duplicate-bridge h2h vs the 2M start ≥ −0.02.
- GATE A3 (health): no leaster-trend halt; greedy gates may warn.
- Exploratory (not gates): partner-rate ratchet behavior on the decay
  curve; coupling diff-corr trend.

**Phase B — λ harvest** (fine-tune continues or restarts from A's best):
λ 0.95 → 0.85 → 0.80, stepped.
- GATE B0 (precondition): `play_follow_t02` + `play_t3plus` EV_ora ≥ 0.60
  at Phase-A endpoint.
- GATE B1: `adv_std` at lead nodes drops materially (target ≥ 30%
  reduction) with h2h non-inferiority (as A2).
- **GATE B2 (the point): partner trump-lead rate reaches AND HOLDS ≥ 0.5**
  (exception-aware band; NOT the subsidy-era 0.89) across ≥ 150k episodes
  **with defender trump-lead ≤ 0.10** — i.e., decoupled pinning, not a
  shared excursion. B2 is the program's standing convention endpoint,
  referenced throughout.

**Phase C — campaign:** winning config, fresh start from the 400k selfplay
seed via the portable orchestrator (`run_extended_league.py` recipe: gen-1
anchored, watchdog, stop rule as amended). The 2M-testbed lineage is NOT
the campaign start (reproducibility goal).

### 2.5 Contingency: selective-distillation trigger

Activate the teacher lane (KataGo-style selective supervision at early
lead nodes; τ = 0.5, top@Q, frac = 1.0, rollout-to-terminal, ESS gates,
`seat_policies` grounding on window tables) IF Phase B completes its λ
step with B0/B1 passing but **B2 fails** — i.e., SNR demonstrably improved
but partner/defender lead strategy still fails to decouple and pin.
Rationale recorded in Convention_Erosion_202607 (distillation = zero-noise
role-conditional credit + off-policy-in-action re-ignition at mass 0.004).

## 3. Phase A: allocation + table composition — FAILED (later demoted to "failed under confound")

### 3.1 Attempt 1 (2026-07-21, `runs/redesign_phaseA/`, commit 8bf7a56): collapse

Resumed checkpoint_2000000, NO anchor (launch error — the design's own
scaffolding section requires the Arm-A bidding anchor on warm starts).
**Collapsed into the leaster attractor within 50k episodes**: 2.025M pick
14%/leaster 21% (lineage-normal) → 2.05M pick 0%/leaster 72.5%, greedy
gates firing on PICK < 15% AND play-head logit spread < 0.5. Killed at
~2.05M; log kept.

**Mechanism identified:** the decision flag originally filtered the
per-head ENTROPY means to decision rows — but the entropy coefficients
were tuned against the historical all-rows (diluted) scale, so effective
entropy pressure rose ~1.5× and pushed the play head toward uniform
(exactly the failing gate), dragging picker EV down into the pass/leaster
spiral; the missing anchor removed the bidding-head brake. Head-balanced
PG gradients were NOT amplified by the flag (the total/count normalization
cancels the dilution — verified arithmetically). Fixes (commit 2ceb778):
entropy stays at the all-rows scale under the flag (+ regression test);
anchor made mandatory for all warm-started fine-tunes in this program.
Lesson recorded: the greedy-gate warnings + quarter-mark monitor caught
the collapse in one wall-clock hour — the scaffolding works when used.

### 3.2 Attempt 2 (2026-07-21, `runs/redesign_phaseA_r2/`): gate results

Same config + `--anchor-coeff 1.0` (ref = the 2M resume ckpt), fresh
league-window copy. Run completed 2.0M → 2.1M cleanly.

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
value-loss decision weighting shifting critic capacity toward rare
decision nodes — but the broad EV regression says the 100k fine-tune left
the critic mid-transient, or worse, that the allocation change degrades
the critic at this budget.

**GATE A2: FAIL** (`h2h_duplicate_2100k_vs_2000k.json`; duplicate-bridge
instrument, 2×2000 deals, seed 42): edge **−0.300 ± 0.015** score/hand vs
the 2M start (gate: ≥ −0.02; called −0.287, jd −0.313 — modes agree). A
~20σ strength regression in 100k episodes. Corroborated by the noisy
in-trainer anchored eval vs final_pfsp_swish_ppo: −0.23 ± 0.13 at the 2M
league checkpoint → −0.707 ± 0.164 (n=300) at the Phase-A endpoint.
Trainer-batch
pooled ev_oracle dropped 0.38 → ~0.21 within the FIRST 10k episodes and
stayed flat all 100k (no recovery slope), with ev_limited negative
(−0.3..−0.5). For calibration: the from-scratch oracle head took ~1.0–1.2M
episodes to plateau (0.30 by ~600k), so the 100k window likely could not
complete any re-convergence transient — but the flat (not recovering) EV
plus the large strength drop reads as genuine disruption, not a benign
transient passing through.

**GATE A3: PASS with flag.** No leaster-watchdog halt; training leaster
stable 21–24%, pick 12–15%, picker_avg +1.07 → +1.21, anchor_kl
0.007–0.024 throughout. Flag: greedy ALONE rate exceeded the 20% warn gate
on both probes and rising — 26.2% @2.05M → 31.7% @2.1M (lineage-normal
band 18–27%); greedy PICK 21.4%, leaster 27.5%, play-spread 0.84 at the
boundary (all normal).

**Exploratory behavior probes** (`decay_curve_r2.csv`,
`role_coupling_r2.json`; same instruments/seeds as the erosion study): no
partner ratchet — partner_trump rate 0.000 @2.05M → 0.013 @2.1M (low phase
of the known oscillation). Role coupling INTACT: partner/defender node
masses rose together ~10× between the two r2 checkpoints (0.0025 → 0.026
partner, 0.0067 → 0.079 defender) — a fresh SHARED excursion, echoed
behaviorally by greedy defender trump-lead 0.000 → 0.097 (above the
0.03–0.08 historical band). C2 dipped mildly (0.392 → 0.333, ~2σ below the
0.41 ± 0.06 band; n=219, watch-only). Net: 100k of Phase-A config did not
decouple roles or start a ratchet — expected at this budget.

**Exploitability audit** (gen-1 exploiter, 50k eps + duplicate-bridge
gate, 3000 deals): the exploiter passed its gate — edge +0.106 ± 0.022
score/deal vs the frozen Phase-A endpoint (win frac 0.587, 83.3% of deals
perturbed; best screen ckpt 2140000). *Correction (operator, 2026-07-21):
an earlier draft called this "the first gate pass in program history" —
wrong; that record belongs to the old repro-run league (inert gens 1–11).
In THIS lineage the v2 gen-1 exploiter passed (+0.111 ± 0.045 vs the 1M
ckpt) and both `full`-arm exploiters passed.* Against the 2M start's own
gen-2 audit (+0.064 ± 0.042, fail), the Phase-A endpoint's +0.106 ± 0.022
is directionally worse but NOT significant (Δ ≈ +0.04 ± 0.05) — only
weakly consistent with degradation; A2 carries the verdict on its own.

### 3.3 Verdict (2026-07-21) and candidate mechanisms

**PHASE A VERDICT: FAIL — stop for operator review.** A1 FAIL (early-node
EV regressed; sole gain: secret-partner ×1.9), A2 FAIL (−0.300 ± 0.015 vs
2M start), A3 pass-with-flag (ALONE streak 26→32%). Per pre-registration,
Phase B was NOT launched. The 2M start checkpoint remains the lineage
reference; the Phase-A endpoint is not a candidate for anything.

Candidate mechanisms recorded at the time (not discriminated):
1. **Critic disruption from the reweighted value loss** — ev_oracle fell
   0.38 → ~0.21 within 10k eps, flat all 100k; ev_limited negative.
2. **Advantage-scale shift from decision-only normalization** — raw
   adv_std fell (all 0.119 → 0.086; pick 0.124 → 0.056 — pick rows are
   genuine decisions, so this is not the mechanical forced-zeroing
   effect), changing the effective policy step size.
3. **Opponent-diversity loss** (p_self_table 0.65) — least likely to
   produce −0.30 in 100k alone, but plausibly compounds 1–2.

Discriminating experiments listed cheap → expensive: offline critic-fit
bake-off (run, §4.1); single-change 100k arms; longer Phase A. **CONFOUND,
established 2026-07-24 (§6.3): the design change also raised braided rows
~67% → ~90%, and the observed oracle disruption is what a 90%
frame-hopping diet predicts independent of decision weighting.
Design-vs-bug split unrecoverable from that run; verdict demoted to
"failed under confound."**

### 3.4 Post-Phase-A operator directives (2026-07-21)

1. **Decision-weighting machinery REVERTED** (commit af9614a): the
   `decision_weighting` flag and all loss-path machinery removed from
   PPOAgent and the trainer — a mostly-failed experiment is not worth its
   codebase complexity. Table-level sampling (`--table-self-play`) and the
   `--gae-lambda` override remained at the time (tests moved to
   test_table_sampling.py; table-self-play later stripped too, §9).
   Goldens 34/34 bit-identical, fast suite green.
2. **Offline oracle bake-off commissioned** (§4.1) as the next
   discriminating experiment, replacing in-loop allocation probes.

## 4. Offline oracle studies

### 4.1 Bake-off: shared vs per-phase experts (pre-registered + run 2026-07-21)

Tool `diagnostics/oracle_moe_offline.py`; results
`runs/oracle_moe_offline/`.

**Question:** how much of the early-node oracle EV gap is shared-capacity
interference (architecture-fixable) vs effective-sample starvation (not)?
Phase A showed in-loop allocation probes are expensive and confounded;
this measures the allocation question as supervised regression on frozen
data with zero RL-loop risk.

**Design (pre-registered):** 36,000 self-play episodes from the 2M league
checkpoint (stochastic acting, oracle observations, empirical discounted
G — the `critic_stratified_ev` semantics; seed 20260721), split 80/10/10
train/val/test by episode. Arms trained from scratch on identical data,
early-stopped on val MSE (patience 2, max 15 epochs, Adam 3e-4 = the
trainer's critic LR):
- `ref` — the 2M checkpoint's online-trained oracle head, eval-only.
- `shared` — one production-shape OracleValueNetwork.
- `moe` — five fresh OracleValueNetworks hard-routed by phase (operator
  spec): pick, partner-call, bury, play tricks 0–2, play tricks 3–5.
  Observable routing (head + trick); each expert consumes episode prefixes
  up to its last routed step. Capacity deliberately unmatched (5×): the
  production question is "beat the production critic on identical data",
  and oracle capacity is deploy-free. Precedent: backgammon phase nets
  (GNU BG/Snowie), Stockfish NNUE material buckets, Suphx per-action-type
  models.

**Interpretation guide fixed in advance:** (i) `moe` ≳ closes half the
shared-vs-ceiling gap at pick/play_lead_t02 ⇒ interference dominant ⇒ wire
per-phase experts into the trainer; (ii) `moe` ≈ `shared` there ⇒
starvation/rarity dominates ⇒ the search/expectation lane moves up the
queue; (iii) `shared` (offline, converged) ≫ `ref` ⇒ online oracle
undertrained independent of architecture. Secondary: val-MSE convergence
curves; per-expert n (partner/bury experts train on ~15–20% of episodes).

**Results** (`results.json`; paired episode-level bootstrap, 1,000
resamples, `bootstrap.json`; 28,800/3,600/3,600 split, 26,421 test rows).
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
experts lost to the single shared network at *every* minority stratum,
all delta CIs excluding zero; the only routing wins are the majority
stratum (play_t3plus) and leaster, both marginal +0.04. This is the mirror
image of the interference prediction.

**Mechanism: cross-phase representation transfer outweighs interference
at this data scale.** The GRU encoder is causal, so the comparison is
information-matched by construction. What routing removes is transfer:
the shared trunk's features, learned mostly from abundant play rows,
transfer to pick/partner/bury value estimation (shared 0.139 vs expert
0.001 on identical pick rows). Experts trade interference relief for
transfer loss, and transfer wins — despite moe's 5× capacity advantage.

Caveats, none verdict-threatening: the small experts stalled to stratum
means — pick collapsed in epoch 1 (val MSE 0.0431 ≈ Var(G) = 0.042,
patience-2 stopped at epoch 4), partner similar (val 0.122 ≈ Var 0.134);
stalls, not converged failures. But the play_t02 expert trained healthily
(86k rows, best at epoch 7 of 10, real val descent) and still lost at its
own strata by −0.14 (lead) and −0.19 (follow), so the decision-relevant
comparison does not rest on the stalled experts. `shared` was
under-converged at the 15-epoch cap, which only widens the verdict.

**Secondary finding — interpretation (iii) resolved with a twist.**
Pooled, ref ≫ shared (−0.096 [−.120,−.068] shared−ref), but the deficit is
entirely in the play strata (play_t3plus −0.225, follow −0.154); at pick
(+0.013), partner_call (+0.003) and bury (+0.020) the 15-epoch
from-scratch fit already *matches* the online oracle. The online head's
low early-node EVs (~0.13–0.24) are reproduced by supervised regression on
230k rows — **the early-node gap is a property of the data (playout-noise
floor + conditional-outcome spread), not of the online training regime**.
"Train the oracle more/better" is off the candidate list for early nodes.
*(Transfer caveat added 2026-07-24: `ref` was a braided-trained head
evaluated on clean streams, §6.3.)*

**Consequences (per the pre-registered guide):** (1) per-phase experts NOT
wired into the trainer; (2) starvation + noise floor dominate ⇒ the
search/expectation lane moves up the queue; (3) the batch-scale arm
(§5) remains the cheapest in-loop lever consistent with these results.

### 4.2 Oracle representation probe (2026-07-24)

Question (operator): would deterministic aux heads (like the limited
critic's) help the oracle, per the 30M-era secret-partner-head precedent?
Linear probes on the batch arm's 850k checkpoint at two taps — trunk input
(post-readout 256-d) and trunk output (which the LINEAR value head reads,
so linear decodability there is exactly the currency of value
expressivity) — vs a random-init control of the same architecture; 6,000
episodes from the frozen offline dataset (policy-distribution caveat:
dataset generated by v2 2M), episode-split 80/20, labels computed exactly
from full-info obs (scratchpad `oracle_probe.py`). Findings:

- Secret-partner seat (deterministic; injected into the encoder TWICE —
  context scalar + role-embedding on every opp-hand token):
  partner-present rows decode at **86.3%** at trunk input, degrading to
  **79.5%** at trunk output (trained). Random-init control: 70.8/65.4. So
  training added ~+15 pts over random but a *deterministic input feature*
  sits far from ceiling in the representation the value head reads — and
  the trunk compresses it AWAY (−6.8 pts through the trunk).
- Opponent trump counts: R² 0.84 (trunk in) → 0.73 (trunk out);
  random-init 0.82/0.79. Barely better than random features, attenuated by
  the trunk. Hero trump: ~0.99 (trivially preserved).
- Picker-team points-so-far (binding composite): R² 0.81/0.79 vs random
  0.76 — modest gain over random.

Reading: explicit injection does most of the linear work, value-loss
gradients have NOT demanded crisp preservation of role/trump structure
through the bottleneck, and the trunk actively attenuates it — the
signature predicted by the "noisy scalar regression under-provisions
representation" mechanism from the 30M-era policy-side precedent.
Counterpoint kept honest: the value head needs only a value-relevant 1-d
projection, not full class decodability, so sub-ceiling ≠ proof of harm.
Consequence: deterministic aux heads on the oracle upgrade from
speculative to grounded amendment candidate; offline test first (§4.3).

### 4.3 Aux-head offline test (pre-registered 2026-07-24, launched before results seen; harness commit 85d5730)

`shared_aux` arm = shared oracle + TWO deterministic heads at the
value-trunk output:
- **Per-seat picker-team MEMBERSHIP** (5-dim multi-label sigmoid; operator
  redesign 2026-07-24, third pre-results amendment, prior launches killed
  within ~2 epochs: supervise the full team split rather than classify the
  secret partner, because the partition is the feature the value composes
  over — partner identity is recoverable as the non-picker member, alone =
  picker-only, and the pre-call window labels the true current team; this
  also unifies the two heads around one membership concept).
- **Picker/defender team points-so-far** (2-dim, /120). Loss MASKED on
  leaster and pre-pick rows (leaster "teams" are 5 singletons — the target
  degenerates to `points_taken_rel`, already an explicit input; format
  mismatch); alone hands kept. **Bury INCLUDED in the picker-team total**,
  as it stands at the timestamp (operator amendment 2026-07-24, before any
  results seen): the target should be the quantity that determines who
  is/will be winning; the head's value is forcing the transformer to
  assemble current team point state at every timestamp.
- Trump-counts head EXCLUDED by operator choice: imposing a count-summary
  target could anchor the trump representation to exactly the crude
  statistic we don't want.

Coefficients mirror the limited critic's (partner 0.1, points 0.2); early
stop on val value-MSE only. Protocol otherwise IDENTICAL to the shared arm
(same dataset/splits/lr/batch/patience) so `shared.pt` is the paired
baseline. Interpretation guide declared in advance: primary =
paired-bootstrap per-stratum EV `shared_aux − shared` at partner/lead
strata (play_lead_t02 primary, partner_call secondary); CI > 0 ⇒ trainer
amendment candidate (gen boundary only); CI spanning zero at role strata
with no majority-stratum harm ⇒ heads inert offline, do NOT amend on
speculation. Secondary: membership accuracy ~ceiling; team-points MAE; no
play_t3plus regression. Caveat: offline-15-epochs ≠ online 850k regime;
an offline null does not rule out an online transient-speed benefit, but
an offline win is necessary evidence before touching the trainer.

**RESULT (2026-07-24, results_aux.json + bootstrap.json): pre-registered
primary NULL; theory-predicted substratum and global fit POSITIVE; harm
checks pass.** Paired deltas `shared_aux − shared` (95% CI):

- PRIMARY play_lead_t02 (pooled): +0.010 [−0.011, +0.032] — NULL.
  Secondary partner_call: −0.004 [−0.020, +0.011] — null.
- play_lead_t02_secret_partner: **+0.064 [+0.025, +0.103]** — POSITIVE
  (post-hoc subgroup by pre-registration standards, but the
  hidden-info-heaviest role stratum the probe motivation targeted: EV
  0.152 → 0.216, closing ~48% of the shared→ref gap there).
- Broad fit wins: all +0.025 [+0.011, +0.038]; play_follow_t02 +0.040
  [+0.011, +0.065]; play_t3plus +0.077 [+0.049, +0.103] (the
  no-regression harm check passes by improving). Best val MSE 0.0282 vs
  shared 0.0296, both at the 15-epoch cap.
- Costs: pick −0.013 [−0.023, −0.002], bury −0.028 [−0.053, −0.003] —
  small, marginal CIs.
- Membership head 99.99% exact-set accuracy (ceiling — the representation
  IS forced through the trunk when supervised, confirming the probe's
  attenuation was demand-driven, not capacity-driven); team-points MAE 7.4.
- Notable: with heads, the 15-epoch/36k-episode offline oracle MATCHES the
  2M-episode online ref at every early stratum; ref's remaining edge is
  confined to play strata where its data advantage lives.

Verdict per the pre-registered map: NOT the automatic-amendment win
(pooled-lead CI spans zero) — but the inert-branch reading is also
excluded (global and secret-partner CIs > 0). Status: **amendment
candidate for the next config** — and the heads were subsequently adopted
in the retention run's pretrained oracle (§7.1), where the oracle
baseline's quality at partner-lead nodes is directly load-bearing.

## 5. Batch+λ SNR arm — pinned at the wrong equilibrium

Pre-registered 2026-07-21, operator-approved; LAUNCHED 2026-07-21 22:57 —
`runs/league_snr_batchlam/`, historical league path, launch log
`runs/league_snr_batchlam_launch.log`.

### 5.1 Hypothesis and dose arithmetic (with corrections)

**Hypothesis under test:** rare-node policy-gradient SNR is the binding
constraint on role differentiation (partner-vs-defender lead conventions
decoupling and pinning). Tested at the strongest dose the current levers
compose to; a fail is close to a falsification, not an underdose (why the
operator chose the composed arm over batch-only at ~2 days / 1M episodes).

Original dose arithmetic (2026-07-21; corrected pre-launch by an empirical
probe — the trainer's transition counter counts hero ACTION rows only,
~7.05/ep measured over 2,325 episodes, so update-interval 2048 ≈ **290
episodes** per update, not the ~80 quoted earlier; the historical
256-episode minibatch cap DID bind mildly): per-row SNR at partner-lead
nodes ≈ Δ/σ = 0.24/1.0; at 2048 (~12.5 partner-lead rows/update) an update
is ~0.85σ; 8× (16,384 ≈ 2,325 eps, ~100 partner-lead rows) ⇒ ~2.4σ;
composed with λ-harvest (σ 1.0 → ~0.6) ⇒ ~4σ-equivalent. Values at these
nodes already correct and ecology-invariant (erosion rung 1); failure mode
is noisy-overwrite oscillation, which per-step averaging attacks directly.

**AMENDED OUTCOME MAPPING (2026-07-24, operator-confirmed, before any
endpoint read).** External review (Opus 5) identified a flaw, verified
against code and telemetry:
- The "8× ≈ 2.4σ" dose was a PER-UPDATE statement. Accumulated
  signal-to-noise over E episodes is √E·Δ/σ, invariant to batch size at
  fixed LR (robust under Adam normalization and PPO clip-bounded steps).
  **Batch is NOT an acquisition-SNR lever; at fixed LR it is a
  displacement reducer.**
- Verified: `apply_schedules` keys LR to EPISODE (train_league_ppo.py:250)
  — the arm walks the same
  LR decay with ~16× fewer optimizer steps than v2 at matched episodes
  (~1,550 vs ~25,000 at 900k; the review said 8× — the truth is worse).
  anchor_kl ~0.0045 vs v2 ~0.008–0.012 confirms suppressed displacement
  (deviating_frac 0.59 @500k shows real movement — but direction
  confirmed).
- Verified: the λ gate could not fire at the 1M boundary. ev_ora
  0.084/0.176/0.212 @300k/600k/900k (v2: 0.26/0.40/0.39); concave trend
  crosses 0.30 ≥ ~1.6M. Gen 1 delivered the batch half only.
- The frame the displacement math misses: **ACQUISITION vs RETENTION.**
  Stationary policy jitter scales with SGD temperature η·σ²/B — batch ×8
  at fixed LR cuts equilibrium rare-node jitter ~8×, a real dose for the
  HOLD half of B2, while under-dosing REACH.

New mapping, replacing the original where they conflict:
- Pin (reach AND hold) ⇒ STRONGER evidence than originally registered for
  the temperature/oscillation mechanism (linear theory says the arm should
  under-acquire, so pinning is informative, not expected).
- Hold-improvement without reach ⇒ temperature mechanism supported on
  retention; acquisition starved ⇒ gen-3 decision is a
  corrected-displacement config (LR/epochs per GNS readout), NOT the
  search/expectation lane.
- B2 fail ⇒ CONFOUNDED (displacement starvation vs SNR falsity); does NOT
  falsify SNR-as-binding-constraint, does NOT activate the
  search/expectation lane.
- *Addendum (review round 2, operator-confirmed):* **oscillation
  eliminated but pinned at the WRONG equilibrium** (defender flat-zero AND
  partner flat-zero) ⇒ hold works, reach dead at this temperature, B2
  unreachable from inside the run; branch = search-distillation
  re-ignition contingency (selective distill at lead nodes), not a
  temperature increase alone. Stability is not correctness.

### 5.2 Design and launch

Single arm, matched-endpoint comparison. Start:
`runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt`
— the SAME 400k selfplay seed as the v2 league, arch perceiver-shared-v2,
critic-mode oracle, seed 42, leaster-watchdog on, all cadences as the v2
orchestrator invocation (main-episodes 1M, schedule-horizon 20M, workers
8). Changes vs baseline (all flags):
1. `--update-interval 16384` (hero action rows; ~2,325 episodes/update).
2. `--minibatch-episodes 4096` — keeps every optimizer step full-buffer.
   **AMENDED 2026-07-22 after the OOM (§5.3): `--minibatch-episodes 128
   --grad-accum`** — gradient accumulation applies the SAME full-buffer
   step once per epoch with per-forward memory bounded at 128 episodes;
   step semantics preserved exactly.
3. λ stays 0.95 for the first ~250k, then a DECLARED restart with
   `--gae-lambda 0.85` gated on: duplicate h2h vs the 400k seed ≥ −0.05
   AND a recorded lead-node adv_std baseline; post-step check lead-node
   adv_std down ≥ 20% within 2 probes, else revert. **AMENDED 2026-07-22
   (declared at 100k, before the gate fires):** additional precondition —
   trainer pooled ev_ora ≥ 0.30 sustained over 3 consecutive updates
   (8× batch ⇒ 8× fewer optimizer steps; the fresh oracle's transient is
   step-count-limited: observed ev_ora 0.00 at 100k vs ~0.12 from-scratch
   reference; stepping λ onto an immature critic injects bootstrap bias
   exactly when least trustworthy).
4. Exploiter re-entry amendment (operator, 2026-07-21; commits
   0db57fc/d647404): `--exploiter-full-table --exploiter-patched-ema 0.35`
   — gated exploiters re-enter sampling as WHOLE tables at the historical
   edge-scaled share; patched retirement (live outcome EMA < 0.35 with
   ≥ 200 samples ⇒ demote to past_main) so a patched exploit stops burning
   its frozen-edge share. Expression check pre-registered: hero deficit on
   exploiter tables far below the gate edge ⇒ fall back to per-seat.
   *(Whole-table seating later retired for the retention run, §7.7, and
   stripped from the codebase, §9.)*

Launch shape (orchestrator): `python -m
sheepshead.training.run_extended_league --resume <400k seed> --run-name
league_snr_batchlam --update-interval 16384 --critic-mode oracle
--leaster-watchdog --seed 42 --trainer-args "--minibatch-episodes 4096
--exploiter-full-table --exploiter-patched-ema 0.35"` (all other flags at
defaults = the v2 invocation).

**Comparison protocol — matched-endpoint, NOT matched-machinery:**
comparisons offline at matched episode counts: duplicate h2h vs the 400k
seed and vs v2 checkpoints (1M/2M), stratified critic EV, role-coupling
probe. **Endpoints:** primary = B2 by 2M, judged on the
role-coupling-probe trajectory; secondary = duplicate h2h vs v2 at matched
episodes ≥ 0.00 − 0.02, oscillation half-life vs v2. **Kill rules:**
duplicate h2h vs the 400k seed ≤ −0.10 at the 500k probe;
leaster-watchdog trip + failure to recover within 100k; greedy-health gate
streaks.

### 5.3 OOM incident (2026-07-22) — root-caused and fixed

Gen-1 trainer SIGKILLed at ~240k episodes (≈18h in) and on every resume
(~2 min in, at the first update). Diagnosis (RSS tracing + faulthandler
stack at the spike): the first full-buffer update in oracle+anchor mode
peaks ~40 GB — the with-grad oracle forward (51 tokens/step) plus the
anchor reference forward over a max-length-padded minibatch whose segment
lengths turned heterogeneous once tables mixed (mostly ~35-event hero
streams + occasional ~175-event self-table streams: B×T_max jumped ~5×,
from ~80k to ~400k padded steps — exactly at the episode where 4-member
mixed tables appeared). The pre-launch memory probe missed it by testing
limited-mode/no-anchor/homogeneous lengths (14.2 GB). The user's
concurrent analysis job likely set the final tipping point but the peak
was marginal-to-fatal on 64 GB regardless. FIX: gradient accumulation
(`update(grad_accum=True)`) — row-fraction-scaled minibatch backwards, ONE
optimizer step per epoch: the design's full-buffer step exactly, memory
bounded by `--minibatch-episodes 128`. Default-off, historical path
bit-identical (test + 34/34 goldens). Verified live: post-fix first update
completed at Ep 201,430, 10 GB peak, 4.5 eps/s. ~40k episodes lost to the
200k checkpoint on resume. *(Retrospective §6.1: the ~175-event braided
outliers that drove the padding blowup were themselves the storage bug.)*

### 5.4 500k kill probe (2026-07-23): PASS

Duplicate-bridge h2h of the 500k checkpoint vs the 400k seed: edge
**−0.068 ± 0.018** (called −0.070 / jd −0.066, 2,000 deals;
`orchestrator/killprobe_500k_vs_seed.json`) — clear of the ≤ −0.10 kill
rule, and lineage-normal: v2's gen-1 ENDPOINT (1M) measured −0.086 ± 0.013
vs this same seed on this same instrument, so the arm at 500k tracked the
ordinary anchored-gen-1 dip, slightly ahead of v2's pace. Run state at
probe: ev_ora 0.14–0.16 (climbing), ev_lim ≈ −0.02, leaster 0.3–0.5%,
ALONE 25–32%, 4.4 eps/s.

### 5.5 External reviews, rounds 2–3 (2026-07-24)

Round 1 was the amended outcome mapping (§5.1). Rounds 2–3, assessed
against code and telemetry:

**Numerator (Δ = +0.237) — ACCEPTED in full.** The partner-lead gap is
max-over-trump-branches − max-over-fail-branches selected on the SAME 50
rollouts that score them (per-branch SE ≈ 0.14): winner's-curse inflation,
flagged in the original study but never de-biased. PRE-REGISTERED FIX, to
run before any B2 interpretation: hold the branch selection FIXED as made
by the June data and re-evaluate the selected branches only, on fresh
independent rollouts — a pure evaluation-of-a-fixed-hypothesis, unbiased,
~one evaluation pass of compute. *(Run for the retention design: §7.2 —
fresh-population replication +0.358 ± 0.039; not selection-inflated.)*

**Denominator (σ ≈ 1.0 score) — conclusion accepted, mechanism
corrected.** Advantages ARE globally normalized before the loss (ppo.py
~1552), but a
global scalar divides rare-node signal and within-node noise identically —
per-row resolvability (σ_node/Δ)² is scale-invariant; normalization per se
cannot halve the dose. The live issue: 1.0 is the PLAYOUT σ, while the
optimizer resolves against realized ADVANTAGE noise at lead rows — never
measured (shrunk by baseline EV, inflated by baseline error, λ-bootstrap
variance, within-stratum heterogeneity). Logged global adv_std ≈ 0.15
reward ≈ 1.8 score suggests per-row SNR ≈ 0.13 if lead rows match. The GNS
instrumentation measures exactly this; stratified lead-row adv_std joined
the boundary baseline.

**Endpoint not measured — ACCEPTED; probe launched.** No decay-curve /
role-coupling output existed for the arm; greedy_health tracks only the
DEFENDER t0 trump-lead, 0.00 at all 18 probes. Context the critique
lacked: v2 was ALSO dead-flat 0.00 through 450k and only began oscillating
at 500k — so the arm's flatness is discriminating only over 500k–900k.
The partner_trump column of convention_decay_curve distinguishes
(temperature holding the correct equilibrium vs global pinning with
partner collapsed alongside); launched over the arm's full 50k ladder +
seed → §5.6.

**Round 3 — variance levers + direct SNR measurement:**
- *Deal-paired/antithetic collection at train time — ACCEPTED as
  first-order; pre-registered for the NEXT config, not mid-arm.* K-replaying
  each deal and subtracting the deal-level mean return is a valid,
  unbiased baseline that removes deal-conditional variance INCLUDING the
  part the oracle critic misses; seat-rotated antithetic (hero in all 5
  seats of one deal) is the exact train-time analog of the
  duplicate-bridge instrument (which cut eval se 0.045→0.017, ≈7×
  variance). If the eval variance split carries over (~6/7
  deal-conditional), K=2 nets ~3× cleaner advantages per episode. Costs:
  episode-generation restructuring, replay correlation, mid-arm confound —
  hence next-config. *(Adopted as `--seat-rotation` in §7.1.)*
- *γ = 1.0 — ACCEPTED; pre-registered for the next config.* With purely
  terminal reward and ≤9-decision horizon, γ=0.99 shrinks early-node
  targets by 0.99^7 ≈ 0.93 — a systematic ~7% objective tilt against
  exactly the early nodes at issue, no variance benefit at this horizon.
  Program-wide consistency required on flip (GAE recursion, offline
  datasets). *(Adopted as `--gamma 1.0` in §7.1.)*
- *Measure gradient SNR directly — ACCEPTED and IMPLEMENTED immediately*
  (extends the committed GNS instrument): per update, at partner-lead
  rows: sampled count, realized advantage mean/std (normalized units),
  mean policy mass on trump-lead plays (legality-masked softmax over the
  14 PLAY-trump actions; the direct re-ignition-regime detector). CSV
  columns lead_rows/lead_adv_mean/lead_adv_std/lead_trump_mass. Process
  point concurred: this instrument before launch would have caught both
  §5.1 errors — exactly the operator's standing
  cheap-gating-diagnostics-first preference; should have been step one.

### 5.6 Decay-curve result (2026-07-24): pinned at the wrong equilibrium

`orchestrator/decay_curve.csv`: 400 CRN deals per checkpoint, called-ace
mode, scripted field, seed prepended:
- partner_trump: **0.766 at the 400k seed → 0.012 by 50k**, then
  ~0.00–0.05 for 850k episodes (single excursion 0.124 @750k, back to
  0.01).
- defender_trump: 0.00–0.065 throughout (B2's ≤ 0.10 satisfied — the
  greedy probe's flat zero was this, masking the partner collapse).
- c2_called_suit: stable ~0.45–0.53 across the whole run (control).

v2 comparison at matched episodes: v2 collapsed IDENTICALLY (0.766 → 0.057
@50k) — the collapse is lineage-normal for an anchored league start from
this seed (the anchor protects bidding heads only; play conventions are
exposed) — but v2 then re-ignited repeatedly at high temperature (0.21
@400k, 0.35 @500k, 0.46 @550k, 0.54 @750k) and lost it each time (the
documented oscillation). The arm never re-ignited: the low-temperature
regime suppressed the noise-driven excursions in BOTH directions, holding
the near-zero equilibrium the collapse left it in. Two corollaries: (1)
the reviewer's "PG cannot re-ignite at mass ~0.004" is too strong as
stated — v2's excursions prove noise+entropy CAN re-ignite from ~0 at high
temperature — but the excursions never stabilized, and the arm removed
exactly the noise that powered them; (2) the arm's collapse happened in
the FIRST 50k, before any batch/temperature property could matter — the
hold mechanism then preserved the wrong fixed point, exactly as the
amended outcome map's new cell describes. **Stability confirmed;
correctness not achieved; B2 unreachable from inside the run.**

Designated branch per the operator-confirmed outcome map: the
**search-distillation re-ignition contingency** — INJECT the convention
(pi_gumbel readout, measured to re-ignite from a zero floor in the
search-readout study) composed with the low-temperature regime to HOLD it;
the two mechanisms now separately validated (hold: this arm; inject: v2's
excursions show the ecology accepts the convention transiently; oracle
counterfactuals say it is value-correct). *What actually happened next:
the §6 bugs were found days later, adding braided bootstrap gradients as a
third co-factor in the 0–50k collapse window, and the operator chose the
retention-first design (§7) — protect the seed's conventions from update 1
rather than re-ignite them after death.*

A gen-2 boundary package was pre-registered for this arm (optimizer-step
telemetry, GNS logging global + partner-lead, oracle-only extra epochs to
pull the λ-gate crossing forward); it was absorbed into the retention-run
design (§7.1 items 3 and 8) when that superseded the arm. Batch-arm
decay-curve facts stand (measurement-side).

## 6. The two league-path training bugs (2026-07-24/25)

Both found while chasing the retention run's oracle non-expression; both
league-path only. Together they reinterpret most of the program's league
history. The selfplay trainer (which produced the seeds), all offline
diagnostics (frozen distinct opponents ⇒ hero streams), and all eval
instruments (act-time per-player memories) were never exposed.

### 6.1 Braided storage (fixed ea1914d, 2026-07-24; operator directive — a sibling of the pre-30M interleaving bug)

The league path stored ALL collecting seats' events as one
temporally-interleaved list with a single done flag, so every self-seat
episode became ONE braided multi-perspective recurrent segment.
Consequences, all verified:
1. Train/act mismatch — act-time memories are per-player, the update
   forward ran one memory across perspective switches, giving NON-UNIT PPO
   ratios at θ_old for self-seat rows (every league run in the lineage
   carried this silently).
2. The pretrained oracle scored EV −0.9 on braided segments vs 0.556 on
   coherent ones.
3. The ~175-event braided outliers drove the July OOM padding blowup
   (§5.3).

Fix: `store_events_by_seat` groups by player_id, one coherent stream +
done flag per seat (segments now ≤ 10 actions + own-perspective
observation frames). Test coverage: per-seat segmentation, braided
control, hero-only byte-equivalence, and ratio-at-θ_old ≈ 1 on a
self-table episode (<1e-4 — the coherence property the bug broke).

### 6.2 Terminal-reward attachment (fixed 45570ff, 2026-07-25; exposed by chasing the ev_oracle zero-shot gap)

After the storage fix, the pretrained oracle still under-expressed
(ev_ora 0.11 at update 1 vs the predicted 0.4–0.5). The operator asked why
live buffers differ from the pretraining distribution at all. Code-diffing
eliminated every candidate (same collection call, γ, sequence structure,
no dropout, no normalization); a decomposition probe rebuilt a live-like
buffer offline and ran oracle_init.pt zero-shot (scratchpad
`oracle_gap_probe.json`):
- dataset-test control EV 0.49 (harness valid); trainer-path frozen-only
  control EV 0.42, MSE 0.035 = held-out ⇒ trainer path fine.
- live-like buffer pooled EV −0.001, but **MSE uniform across ALL row
  classes (0.034–0.040 = held-out)** — the oracle predicts equally well
  everywhere; the EV collapse is in the TARGETS: hero rows on mixed tables
  sd_g 0.140 and self-seat rows 0.157 vs 0.262 on pure tables — the
  signature of zeroed returns.
- Root cause (`pfsp_runtime._finalize_rewards`): the merged multi-seat
  action list was fed to `process_terminal_rewards`, whose documented
  contract is ONE player's chronological transitions. The single terminal
  reward landed on the globally-last actor; every other collecting seat's
  stream was ALL-ZERO. Fix: group transitions by player position (no-op
  for hero-only episodes; also fixes shaped mode's identical contract
  violation). Tests: per-seat reward placement + γ=1 returns through
  storage matching each row's own score.

Combined historical mechanics: pre-storage-fix, the braid's single done
flag propagated that one reward backward through the WHOLE braid — every
seat in a self-containing episode trained toward the LAST ACTOR's return,
not its own (wrong-player targets on ~67% of rows at p_self 0.15, ~90%
under Phase A), on top of incoherent recurrent features.
Post-storage-fix only, non-last-actor seats trained toward zero returns
(~46% of rows).

### 6.3 Historical impact assessment

Exposure: league-path training only (incl. exploiters); ~48% of episodes ⇒
**~67% of training ROWS** braided at historical p_self 0.15 (braided
episodes carry more rows); 100% during empty-league bootstraps; **~90% of
rows under Phase A's p_self_table 0.65**. Consequences for the record:
- **Phase A's gate-fail verdict is CONFOUNDED** (§3.3): the design change
  also raised braided rows 67%→90%; the observed oracle disruption
  (0.38→0.21, flat) is what a 90% frame-hopping diet predicts independent
  of decision weighting. Design-vs-bug split unrecoverable; verdict
  demoted to "failed under confound."
- The **chronic ev_limited ≈ 0/negative pattern across all league runs**
  is plausibly primarily this bug pair (critic trained on braided
  update-forward features toward wrong-player/zeroed targets, evaluated on
  coherent rollout values). The fixed retention run was the discriminating
  test — and resolved it (§7.3: ev_lim positive from update 1).
- The arch-ablation **"league lift ≈ zero over selfplay start"** finding
  is confounded with ~67% corrupted-context gradients (selfplay trainer
  was clean) — reinterpretation open; the retention run's +0.089 at 500k
  (§7.6) is consistent with "league lift ≈ zero" having been the bugs.
- The old **"exploiter pressure inert"** history predates the fixes;
  exploiters trained on the corrupted path (also §7.7).
- Batch-arm decay-curve facts stand (measurement-side); the 0–50k collapse
  window gains braided bootstrap gradients as a third co-factor.
  Aborted-launch GNS readings (7,200/230) VOID — measured on braided
  buffers.
- Unaffected: MoE bake-off, aux-head study, oracle pretraining, Δ
  counterfactuals, h2h/decay measurements. Transfer caveat: bake-off "ref"
  was a braided-trained head evaluated on clean streams.

## 7. Retention-first pure-PG run (LIVE — `runs/league_retention_pg`)

Pre-registered 2026-07-24, operator-approved; the last
pure-policy-gradient attempt before the search-teacher lane.

### 7.1 Diagnosis and design

**Diagnosis this design answers** (decay-curve §5.6 + review rounds 1–3):
the batch arm proved the low-temperature regime can HOLD an equilibrium
for 850k episodes but was handed the wrong one — the seed's conventions
(partner trump-lead 0.766, defender 0.033: already B2-compliant) died in
the first ~21 updates, during the fresh-oracle burn-in (ev_ora ≈ 0) and
the shaped→terminal objective switch (verified: the 400k selfplay seed was
trained with intermediate trick rewards + leaster bonus; the league trains
terminal-only). **Retention, not acquisition, is the task.** The operator
chose supervised oracle pretraining + a fully UNANCHORED gen 1 over a
play-head anchor: the anchor's reference forward is a real
throughput/memory cost, and with an accurate terminal baseline from
update 1 the collapse driver it would fight is largely gone; the
convention is terminal-optimal (§7.2), so correct advantages defend it on
merit.

**Design (all committed 1494902/d888062/7c413ca; flags default-off at the
time — adopted as defaults 2026-07-27, §7.9):**
1. Oracle SUPERVISED PRETRAINING: 40k frozen-seed self-play episodes
   (γ=1.0 terminal returns), official OracleValueNetwork with the two
   offline-validated aux heads (§4.3: team membership 5-way multi-label +
   team points w/ bury; coefficients 0.1/0.2), trained to plateau; loaded
   via `--oracle-init`. Removes the burn-in window entirely.
2. Aux heads stay on ONLINE (`--oracle-aux-heads`): same losses in the
   oracle update path and the extra-epoch pass.
3. `--oracle-extra-epochs 4`: ~0.033 oracle steps/episode, restoring v2's
   oracle step rate at the 16384 interval.
4. Seat-rotated collection (`--seat-rotation`): each sampled deal played
   5×, hero in every seat, same table/cards — role-exposure equalization +
   train-time deal pairing (§5.5 round 3).
5. γ = 1.0 (`--gamma 1.0`): kills the 0.99^7 ≈ 0.93 early-node tilt;
   consistent across dataset, pretraining, and trainer.
6. UNANCHORED gen 1+ (anchor-coeff 0). Bidding drift under the terminal
   objective is expected and partially correct; guarded by
   leaster-watchdog + greedy gates + the bidding contingency (§7.6).
7. Low-temperature regime kept: `--update-interval 16384`,
   `--minibatch-episodes 128`, `--grad-accum`; λ gate as registered
   (pretrained oracle may satisfy ev_ora ≥ 0.30 early — follow the gate;
   later amended, §7.5); exploiter full-table + patched-EMA carried over
   (full-table later dropped, §7.7).
8. Instrumentation from episode 0: `--gns-log` (GNS global + partner-lead,
   lead_adv_mean/std, lead_trump_mass), opt_steps; greedy probe reports
   partner trump-lead (tricks 0–2) every 50k; decay-curve probes at gen
   boundaries. NOTE: rel-seat role-label bug fixed 7c413ca (0-means-self
   misread) — historical partner/defender lead SUBSTRATA in offline
   studies were scrambled mixtures; secret_partner substrata were always
   clean.

### 7.2 Δ validation (2026-07-24): the convention is terminal-optimal

Answering the §5.5 winner's-curse critique before relying on the number
(`cf_partner_trump_400k_replication.json`): fresh-population replication
**+0.358 ± 0.039** (171 agree / 62 disagree, seeds 100000+) vs the
original +0.237 ± 0.040 — the estimate is NOT selection-inflated; the
fresh population reads HIGHER. (Mechanism check: branch selection is by
policy logits, independent of the evaluation rollouts, so no winner's
curse; independent cross-checkpoint reproduction +0.236 at 2M.) The ~2σ
between-population spread exceeds nominal SE (node-population
heterogeneity), so the working figure is "≈ +0.24 to +0.36 score, robustly
positive": the convention the retention design protects is unambiguously
terminal-optimal at the seed.

### 7.3 Launch history (three launches, two bug fixes)

**Launch 1 (2026-07-24 ~19:33).** Oracle pretraining result
(`oracle_init.report.json`): held-out pooled EV **0.508** — above the
online oracle's ~0.40 all-time plateau and the 0.30 λ-gate threshold
before the run begins; early strata pick 0.268 / partner_call 0.398 / bury
0.402 / lead 0.477; membership head 99.99%, team-points MAE 4.6 (γ=1.0
seed-policy data; EVs not directly comparable to v2-2M numbers). Smoke
test caught one integration bug pre-launch (exploiter × headed-oracle
checkpoints, 8a5d1b4).

**Launch amendment (~21:15, before 5k episodes; relaunched).**
First-update telemetry showed ev_ora −0.65 → ~0.05: the pretrained oracle
was NOT expressing during the empty-league bootstrap. Direct probe: the
oracle scores EV 0.556 on hero-stream segments (its training
distribution) but EV ≈ −0.9 on the 60-step all-seat segments pure-self
tables produce — the bootstrap was recreating the burn-in window the
design exists to remove. Fix: **`--seed-checkpoints` = the 400k seed
itself** (4 copies, since `sample_table` samples without replacement), so
the league is non-empty from episode 0 and tables are mixed —
behaviorally identical to self-play, structurally hero-stream data.
Chasing WHY those braided segments existed at all led to the §6.1 storage
bug (fixed ea1914d); interim workarounds (--self-play-share 0) retired
and the run relaunched with default self-play share 0.15 + the 4-seed
pool. Two same-day relaunches are in the run log; all pre-fix data
discarded.

**Fixed-storage first-update telemetry (2026-07-25, updates 1–4):**
ev_limited −0.65 → +0.147 (first league run in the lineage to cross
positive within four updates; pre-fix launches −0.65/−0.86, pinned) but
ev_oracle only 0.11 → 0.24, NOT the predicted instant 0.4–0.5 — the residual gap led to the §6.2
terminal-reward bug. Reinterpretation on record: that ev_ora climb was
the oracle being graded against (and partly learning to predict) ~46%
zeroed targets; all pre-fix telemetry discarded with the final relaunch.
Also from the aborted-launch instruments (VOID, braided buffers): GNS
global ≈ 7,200 / lead ≈ 230 rows; lead_trump_mass 0.485 at the seed.

**Launch 3 — both fixes active (2026-07-25). Zero-shot prediction
CONFIRMED:**
- **ev_oracle 0.52 / 0.55 / 0.55 from update 1** (predicted 0.4–0.5 from
  the probe's trainer-path control). The live trainer distribution IS the
  pretraining distribution; the earlier 0.11 was entirely the
  zeroed-targets bug.
- **ev_limited 0.35 / 0.38 / 0.44 — positive from update 1**,
  unprecedented in the league lineage; the chronic ev_lim disease is
  conclusively attributed to the bug pair.
- GNS global 15.3k/16.5k rows ≈ the 16,384-row batch (B ≈ B_noise:
  critically sized). gns_lead now RESOLVES (458/792 rows updates 1–2; 16k
  at update 3 — single-update estimates are high-variance): with real
  targets the lead-stratum mean gradient is measurable.
- lead_adv_mean −0.14 → −0.011 → −0.008 (mechanism discriminator quiet),
  lead_trump_mass 0.50–0.58 (seed level), adv_std_pick 0.21 (up from 0.16
  under zeroed targets — real signal variance restored), leaster ≤0.3%,
  6.4 eps/s, opt_steps 4/update as designed. Gen-0 endpoint served from
  the new content-hash cache (2b0a77c): trainer handoff 49s after launch.

### 7.4 Reproduction commands (verbatim; final relaunch verified against the live process command line)

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
read from the dataset; aux coefficients default 0.1/0.2; early stop on val
value-MSE, best epoch 20 of 24):

```bash
uv run python -m sheepshead.analysis.diagnostics.oracle_moe_offline pretrain \
  --dataset runs/oracle_pretrain_400k/dataset.pt \
  --max-epochs 25 --patience 3 --seed 20260725 \
  --out runs/oracle_pretrain_400k/oracle_init.pt \
  > runs/oracle_pretrain_400k/pretrain.log 2>&1
```

Seed pool (4 copies of the 400k warmstart, because `sample_table` samples
members without replacement — one member can fill only one seat):

```bash
mkdir -p runs/retention_seeds
for s in a b c d; do
  cp runs/league_arch_perceiver-shared-v2/warmstart_perceiver-shared-v2_400k.pt \
     runs/retention_seeds/seed400k_$s.pt
done
```

Run launch (final relaunch, post-fix, default self-play share; the glob in
--seed-checkpoints is quoted — the orchestrator expands it):

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

*(Historical note: `--exploiter-full-table` was dropped at the gen-1
boundary relaunch (§7.8) and later removed from the CLI entirely (§9); as
of 2026-07-27 the rest of this configuration is the trainer/orchestrator
DEFAULT (§7.9), so a fresh reproduction needs almost none of these
flags.)*

### 7.5 λ policy and the λ-sweep contingency (2026-07-25, operator-approved)

**Operator decision: λ stays 0.95 for the entire retention run** unless
(a) health degrades, or (b) convention leaks appear that additional
training under λ=0.95 cannot close. The registered Phase-B λ schedule is
DECOUPLED from its gate: ev_ora ≥ 0.30 passes trivially from update 1 with
the pretrained oracle, so gate passage is a precondition, not a trigger.
Rationale: the unbiased variance levers (oracle baseline ~3×
advantage-variance cut, seat-rotation deal pairing, γ=1.0) now carry the
load the λ schedule was registered for; λ<1 is the only lever that pays in
bias ∝ critic error, and the critic is weakest exactly at early nodes
(pick EV 0.27 vs lead 0.48). Accumulated-SNR view: variance washes out as
√E; persistent bias does not.

**Probe (offline, no training impact;
`runs/oracle_pretrain_400k/lambda_sweep.{json,probe.py}`):** 4,992 fresh
seed-policy episodes (hero + 4 frozen-seed opponents, terminal, γ=1.0),
V_ora from oracle_init.pt via the trainer's own `_fill_oracle_values`;
per-episode A(λ) via the trainer's `_gae_1d` for λ ∈ {1.0, 0.95, 0.9, 0.7,
0.5, 0.0} — SAME rows across λ (paired); 755 partner-lead rows (385 trump
/ 370 fail). Metrics in raw advantage units; signal = on-policy trump−fail
lead contrast (λ=1 value +0.0208 ± 0.0073 independently reproduces the
+0.24–0.36-score counterfactual prior at /12 scale). Pre-registered
decision rule: adopt-candidate λ* = argmax SNR subject to signal(λ*) same
sign and within 1 bootstrap-σ of signal(1.0), and |pick-row bias| < 0.25 ×
adv_std_pick; node-selective λ (lead nodes only) preferred if λ*
qualifies.

| λ | sd_lead | signal | Δsignal vs λ=1 (paired) | SNR | bias_pick | corr_lead |
|------|--------|---------|------------------|-------|--------|------|
| 1.0 | 0.101 | +0.0208 | — | 0.206 | — | 1.000 |
| 0.95 | 0.091 | +0.0203 | −0.0005 ± 0.0009 | 0.222 | −0.013 | 0.997 |
| 0.9 | 0.083 | +0.0198 | −0.0010 ± 0.0017 | 0.238 | −0.021 | 0.988 |
| 0.7 | 0.062 | +0.0179 | −0.0029 ± 0.0040 | 0.288 | −0.036 | 0.894 |
| 0.5 | 0.052 | +0.0165 | −0.0043 ± 0.0053 | 0.316 | −0.040 | 0.753 |
| 0.0 | 0.045 | +0.0153 | −0.0055 ± 0.0068 | 0.338 | −0.043 | 0.453 |

Findings: (1) **the pretrained oracle DOES encode the convention edge** —
at λ=0 the pure one-step oracle judgment retains ~74% of the MC signal;
the 48-episode smoke's contrary hint did not replicate. (2) σ at lead
nodes halves λ=1→0 while signal falls only ~26%, so estimator SNR rises
monotonically, 1.64× at λ=0. (3) All λ pass the pre-registered
constraints (bias_pick under the 0.071 = 0.25×sd_pickhead bound), so the
formal adopt-candidate is λ* = 0.0. POWER CAVEAT,
recorded against overreach: the paired-Δ SE grows with λ distance (±0.0068
at λ=0), so the "within 1σ" test is weakest exactly where attenuation is
largest; point estimates suggest real ~15–26% signal loss below λ≈0.7.
**Conservative contingency candidate: λ ≈ 0.5–0.7, node-selective (lead
nodes only)** — 1.4–1.5× SNR with corr_lead 0.75–0.89 and pick bias ≤ 14%
of pick-head σ. (4) bias_pick grows monotonically negative (bootstrap
systematically deflates PICK advantages — consistent with the critic's
weak pick stratum, EV 0.27); within bound but the reason node-selective
adoption is preferred over global λ.

**Operator policy unchanged:** λ = 0.95 for this entire run; the sweep
arms the contingency only (fires on unclosable convention leaks).

### 7.6 Gen-1 probe log

**Pre-registered tripwires & kill rules:**
- RETENTION: greedy partner trump-lead < 50% at BOTH the 50k and 100k
  probes ⇒ NEEDS REVIEW (low temperature holding the wrong thing again).
- MECHANISM DISCRIMINATOR: lead_adv_mean persistently negative across the
  first ~20 updates while lead_trump_mass falls ⇒ the anti-convention
  force is systematic ⇒ stop early; search-teacher lane.
- Bidding: leaster-watchdog trip or greedy PICK-gate streak ⇒ re-engage
  the bidding anchor at a declared restart (contingency, not a kill).
- Strength: duplicate h2h vs the 400k seed ≤ −0.10 at 500k.
- B2 endpoint, comparison protocol, outcome mapping as amended 2026-07-24
  (retention framing: reach = keep what the seed has; hold = keep it
  through gen 2's ecology churn).

**Probe trajectory (greedy probes every 50k; % of relevant leads; ev
ranges over the surrounding updates):**

| ep | partner trump-lead | defender t0 | pick | leaster | ev_ora / ev_lim | note |
|---|---|---|---|---|---|---|
| seed | 76.6% | ~3.3% | 47.7% | 0.25% | — | baseline |
| 50k | **75.81%** (62) | 0.00% (94) | 41.7% | 2.0% | 0.55–0.59 / 0.45–0.51 | **GATE 1 PASS** |
| 100k | **80.95%** (63) | 0.00% (113) | 37.4% | 2.5% | 0.54–0.63 / 0.41–0.52 | **GATE 2 PASS — tripwire pair cleared** |
| 150k | 79.76% (84) | 0.00% | ~37% | 5.5% | — | |
| 200k | 62.07% (58) | 0.00% | ~37% | 5.5% | — | watch opened: dip ~2.3σ below plateau |
| 250k | 78.57% | 1.11% | — | 1.5% | — | dip resolved: probe noise |
| 300k | **84.29%** | 4.81% | 35.5–43.2% | 3.5% | 0.60–0.66 / 0.48–0.55 | both watches benign |
| 350k | 94.64% | 1.79% | — | 3.0% | — | sharpening |
| 400k | **97.59%** (83) | 9.00% (9/100) | 32.2% | 7.0% | 0.60–0.63 / 0.49–0.51 | play_logit_spread 0.81→0.91 |
| 450k | 86.7% | 0.91% | 41.6% | 1.0% | ~0.62 / ~0.51 | |
| 500k | 86.8% | 8.91% | 42.2% | 1.5% | — | **kill probe PASSED, see below** |
| 550–700k | 89.0 / 82.8 / 84.0 / 78.4% | 1.1–6.3% → 0.0% | 38–43% | 1.5–4.5% | 0.56–0.64 / 0.46–0.53 | ALONE 36.2% @650k / 34.0% @700k warnings |
| 900k/950k/1M | 93.7 / 93.2 / 80.8% | 0.0% @1M | — | — | — | **gen-1 B2 reading HELD** |

Notes carried from the probe entries:
- **50k/100k: the retention-first hypothesis survived its designated
  falsification window** — the first-100k collapse that killed every prior
  league arm (batch arm 0.012 @50k; v2 ~0) did not occur under correct
  per-seat targets + pretrained oracle + unanchored low-temperature PG.
- The 200k partner dip (62%, SE ≈ 6.4% at n=58) was called
  noise-vs-erosion at the time, with sampled lead_trump_mass flat
  (0.49–0.54) as counter-evidence for noise; 250k/300k confirmed noise.
- Greedy pick drift 47.7% → ~37% → low-40s oscillation. **Operator
  calibration (2026-07-25): the decline is HEALTHY — ~50% is far too
  high, ~30% near-optimal, down to ~20% defensible; concern threshold
  ~20%**, below which the formal greedy gate (min_pick 15%, warning-only)
  sits. Pick settled at 32.2% @400k — inside the optimal band.
- picker_avg climbed 1.10 → 1.53–1.58 (50k) → ~2.0 (200k), sustained
  1.72–1.87 through 700k.
- Defender t0 trump-lead bounced 0–9% under the B2 ≤10% bound (tiny
  counts, SE ≈ 2–3%); never crossed.
- Greedy leaster oscillated 1.5–7.0% with no trend (early-fire 8%,
  watchdog owns the halt; sampled 0.6–3.7%).
- ALONE exceeded this run's effective limit (32.7% = baseline 27.7 + 5) at
  650k/700k — greedy-gate WARNINGS on record (never halts); alone rising
  alongside picker_avg is consistent with growing picker confidence, not
  pathology.
- Watcher early-fire conditions as re-armed through gen 1: partner < 50%,
  pick < 20%, leaster > 8%, defender t0 > 10%.

**500k KILL PROBE — PASSED WITH STRENGTH GAIN (2026-07-25;
`runs/league_retention_pg/h2h_500k_vs_seed.json`).** Duplicate h2h, 500k checkpoint vs 400k seed,
4,000 deals: **edge +0.0885 ± 0.0139 score/hand (+6.4σ)**; called +0.0731
± 0.0191, JD +0.1040 ± 0.0202; win_frac 0.524. Kill rule (≤ −0.10) not
merely cleared — INVERTED. Historical contrast: the arch ablation found
v2's league lift over this same 400k selfplay start ≈ ZERO after 2M
episodes; the fixed trainer delivers +0.089 in 500k WHILE the partner
convention sharpened to the 84–98% band — first league configuration in
the lineage to show real strength lift over its selfplay seed, consistent
with "league lift ≈ zero" having been the bugs (§6.3).

### 7.7 Exploiter: seating amendment + gen-1 gate

**EXPLOITER-SEATING AMENDMENT (pre-registered 2026-07-26,
operator-approved; applies at the gen-2 boundary IF the gen-1 exploiter
passes its gate).** Drop `--exploiter-full-table` for gen 2 onward (keep
`--exploiter-patched-ema 0.35` — retirement logic is orthogonal);
exploiter pressure reverts to per-seat single-exploiter seating. Rationale
recorded BEFORE the gate result:
1. Seat rotation now supplies the role coverage whole-table was
   compensating for: the table is fixed per rotation group, so one
   exploiter seat meets the hero from all 5 relative offsets on the SAME
   deal (matched-pair role sweep). The July "per-seat pressure inert"
   finding predates rotation AND is confounded by both §6 bugs.
2. Certified-configuration match: the exploiter trains and gates as 1
   exploiter vs 4 mains. Whole-table deploys the MIRROR (1 main vs 4
   exploiter copies) — uncertified pressure dosed by a certified edge
   (units mismatch in exploiter_share heat scaling). Per-seat + rotation
   restores the certified configuration from every role.
3. Collusion artifact: 4 parameter-shared exploiter copies form an
   implicit coalition with no evaluation analog. For THIS run it would
   confound B2: retention breaking under 4-way anti-convention collusion ≠
   conventions not robust to realistic exploitation.
4. Literature: AlphaStar main-exploiters = 2p best responses into PFSP
   mixtures; PSRO deploys BRs per-seat from the meta-strategy; multi-seat
   poker practice = one learner seat + position rotation. No precedent for
   whole-table; nothing certified is lost (a 1-vs-4-trained exploiter
   cannot express coordinated multi-seat exploits anyway).

Mechanics: flag affects main-phase table sampling only ⇒ kill orchestrator
at the boundary, relaunch with amended --trainer-args; state/league/
endpoint caches persist, partial episodes under the old flag discarded by
the boundary restart. Readout: greedy partner probes remain the
exploitation instrument — a certified per-seat exploiter driving partner
trump-lead down is a legitimate B2-relevant finding.

**GEN-1 EXPLOITER GATE: FAILED (2026-07-26; exploitability.csv).** Edge
**−0.0279 ± 0.0166**, win_frac 0.460 over 3,000 gate deals — the 50k PPO
exploiter LOSES to the 1M main. First exploiter in the program trained on
the FIXED path (coherent streams, correct rewards), so unlike the old
inert exploiters this is a fair probe: the unanchored 1M main presents no
PPO-findable hole at this budget (echoes the 30M gen-0 "not
PPO-exploitable" finding; consistent with the +0.089 h2h strength gain
rather than degenerate drift). Consequence: no exploiter seated in gen 2
(x-share 0); per operator directive the seating amendment was applied at
the gen-1 boundary restart anyway (§7.8), so it is in force for any future
passed exploiter.

### 7.8 Gen-1 boundary results (2026-07-26)

Orchestrator gen-1 verdict: **panel +0.0425 [+0.0100, +0.0752]** (absolute
vs PANEL-A; the seed's baseline was −0.0771, so the gain vs gen 0 is
+0.1196 [+0.0855, +0.1533] — labeling corrected 2026-07-28, see §7.10; the
same instrument read ≈ 0 league lift after 2M episodes in the arch
ablation) and **h2h gen1-vs-gen0 +0.078 ± 0.013**;
flat=False, streak=0, continues (min_generations 4). End-of-gen probes:
partner 93.7/93.2/80.8% (900k/950k/1M), defender t0 0.0% at 1M ⇒ **gen-1
B2 reading HELD**.

Seating amendment applied at this boundary regardless of the gate failure:
orchestrator killed after h2h, relaunched 23:01 WITHOUT
`--exploiter-full-table` (patched-EMA kept). Resume machinery replayed gen
1 from artifacts in 1s (gen-0 cache + panel_gen1.npz + h2h_gen1.json +
exploitability row); gen 2 restarted from the 1M boundary checkpoint
(~6min of whole-table gen-2 episodes discarded; flag was dormant anyway —
no gated exploiter). Verified: live gen-2 trainer command line lacks the
flag.

### 7.9 Defaults adoption + C2 baseline (2026-07-27)

**Defaults adoption (51a81ac):** the retention-run configuration is now
the DEFAULT for train_league_ppo and run_extended_league (label args
excluded; `--no-*` opt-outs for the boolean flags; PPOAgent internals
untouched). grad-accum / minibatch-episodes reassessed on request: the
braided-segment OOM that motivated them is gone (T_max ~175 → ~40 post
per-seat fix, anchor off), but they are RETAINED as defaults because
grad-accum is now the mechanism of the validated low-temperature design —
one full-buffer optimizer step per epoch with peak memory bounded by
minibatch-episodes, bit-equivalent gradients (test_grad_accum). Their role
changed from memory workaround to step-size semantics.

**C2 (defender called-suit lead) baseline from the backfill sweep**
(`runs/league_retention_pg/c2_adherence_sweep.json`, 1000 CRN deals per
checkpoint): the scripted conventions agent reads 69.7% overall (t0 100%
by construction); the 400k seed reads **47.6%** — C2 was only ever
HALF-learned, unlike partner trump-lead (77%). Retention-run trajectory
46.9 / 46.3 / 48.9 / 47.6 / **40.9% @1M** (n=668, binomial SE ≈ 1.9% ⇒ the
1M reading is ~2.5σ below the seed — mild erosion signal or noise; the
live greedy column tracks it every 50k from 1.05M). C1 defender trump-lead
stays 0.1–4.8% across all checkpoints (convention intact). Note E2's +0.49
@4.4σ says agreeing with C2 is terminal-optimal on average — at 47%
adherence there is real unclaimed headroom; **C2 is an ACQUISITION metric
for this program, not a retention one.** Watch: C2 < ~35% sustained
(erosion below seed) or a climb toward the scripted 70% (the terminal
objective claiming the headroom).

**Amendment — ScriptedAgent C2 extended to all tricks (2026-07-28,
operator directive).** The scripted agent's called-suit-through rule was
trick-0-only — not by convention design but because "called suit not yet
led" was not derivable from the observation dict (the agent is stateless
across tricks). Fixed by adding the public table fact
`called_suit_played` to `get_state_dict` (invisible to RL encoders —
named-field selection; goldens 34/34 bit-identical; web WS schemas are
deliberately loose) and switching the rule to "lead the called suit
through whenever held and not yet led, any trick, all variants."
Calibrated-probe self-check now 100.0% on every split (pooled/trick-0/
first-opp/under; previously pooled 69.7%). Consequences for the record:
- The "scripted 70%" reads above were the TRICK-0-ONLY agent, not a
  structural ceiling — the watch item's upper reference is now ~100%.
- Instrument version boundary: the ScriptedAgent also seats the OPPONENT
  field in the CRN scripted-field instruments (called_suit_probe,
  trump_lead_probe, adherence sweep, decay curve, role-coupling), so
  pre/post-2026-07-28 measurements differ even at frozen seeds —
  adherent field members resolve the called suit sooner, shrinking
  later-trick eligibility (observed: eligible nodes per deal roughly
  halve). Historical numbers stand as recorded under the old agent;
  cross-boundary comparisons must rerun both sides.
- Same-instrument same-day reads for context (2,000 CRN deals, old
  field): retention 1.9M ckpt pooled 38.6% / trick-0 33.1%;
  final_pfsp_swish_ppo (30M, the deployed app model) pooled 85.6% /
  trick-0 93.7% — the C2 acquisition headroom between lineages is
  ~47 points.
- ConventionWrapper C2 (the E-study instrument) deliberately KEPT
  trick-0-only for comparability with recorded results; extending it is
  now possible but would be a new instrument version with its own
  pre-registration.

### 7.10 Gen-2 boundary + adaptive-entropy activation (2026-07-28)

**Crash incident (environment, not code).** The orchestrator died at
21:35:16 starting the gen-2 endpoint eval: `TypeError: Config() got an
unexpected keyword argument 'deprecated'` inside a lazy `torch._dynamo`
import, with garbled traceback line attributions. Root cause: commit
79374e7 (torch security upgrade) changed the lock; an `uv run` re-synced
the venv at Jul-27 00:21 — 80 minutes AFTER the orchestrator launched
(Jul-26 23:01). The long-running process carried old torch in memory over
the swapped install and survived all of gen-2 training (no poisoned
imports on that path); the endpoint eval's fresh Adam construction
triggered the first lazy `torch._dynamo` import, pulling new-version
files into the old process. Fresh processes import cleanly (torch 2.13.0
verified). No data lost: gen-2 training + exploiter phase were complete
and no boundary artifact was half-written — the crash landed exactly at
the pre-registered kill point. **Operational lesson: `uv run`/`uv sync`
swaps the venv under live long-running trainers; on lockfile changes,
either defer the sync or expect lazy-import deaths at phase boundaries.**

**Relaunch (2026-07-28 21:56, `runs/league_retention_pg_relaunch_gen3.log`)
= the pre-registered activation.** Same invocation minus the stripped
`--exploiter-full-table` (patched-EMA kept); adaptive entropy now active
by DEFAULT (§8.6). Resume replayed gen 1 from artifacts in <1s and reran
the gen-2 boundary evals.

**Gen-2 boundary results:**
- **Panel endpoint (ABSOLUTE, vs the PANEL-A mixed-anchor table):
  +0.1323 [+0.1019, +0.1629]**, from +0.0425 at gen 1 and −0.0771 at the
  seed. Per-generation gains: +0.1196 [+0.0855, +0.1533] (gen 1 vs seed)
  and +0.0898 [+0.0617, +0.1176] (gen 2 vs gen 1) — both far clear of
  zero (gain_p = 0.0000), mildly decelerating. *Correction (2026-07-28):
  the first write-up of this boundary (and the gen-1 entry's "panel
  +0.0425 vs gen 0" phrasing) misread panel_mean as a vs-previous-gen
  delta and called gen 2 "accelerating, 3× the gen-1 delta"; panel_mean
  is the absolute anchored edge, the deltas are the gain_vs_best column,
  and the gain sequence is +0.120 → +0.090.*
- **h2h gen2-vs-gen1 (duplicate-bridge, 2000 deals/mode): +0.101 ± 0.013.**
- Verdict: flat=False, streak=0 → continue (below min_generations 4). NO
  entropy-target step (correct per design — steps fire only on flat
  verdicts).
- **Gen-2 exploiter gate: FAILED, worse than gen 1** — edge −0.1841 ±
  0.0156, win_frac 0.420 over 3,000 gate deals (gen 1: −0.0279). The
  fixed-path exploiter loses MORE decisively to the 2M main than to the
  1M main: the unanchored main is hardening, consistent with the panel
  acceleration and not with degenerate drift. No exploiter seated in gen
  3; the §7.7 per-seat seating amendment remains untested by live
  pressure.
- **B2 through gen 2: HELD** — partner trump-lead 92.7/96.3/84.6/87.7% at
  1.85–2.0M, defender t0 0.0–1.1%. C2 43.3–51.4% over the last four
  probes (recovered from the 40.9% dip at 1M to at/above the 47.6% seed).
  Pick 41.6–46.7%, greedy leaster 0.5–2.5%, picker_avg 1.35–1.59.

**Gen-3 launch under the controller (23:30:46):** trainer command carries
the injected `--entropy-mode target --entropy-play-floor 0.28` (gen-1
deferral logic correct at g=3); trainer log shows "🎯 Entropy controller
fresh (bumpless targets pending)". Bumpless target adoption lands at the
first update (~16k episodes); the switch-on prediction (§8.8) is now
live-testable. New telemetry columns (ent_norm_*, softband_*, approx_kl,
lr_actor) populate from gen 3's first progress rows.

### 7.11 Success reading / outcome map

- Partner ≥ 0.5 AND defender ≤ 0.10 held through 2M with the ordinary
  strength trajectory ⇒ retention-first on-policy PG is sufficient; the
  search teacher stays shelved.
- Retention holds through gen 1 but breaks when exploiters/ecology churn
  arrives ⇒ hold-dose insufficient against ecology pressure ⇒ selective
  distillation at lead nodes composes next.
- Retention fails in the first 100k despite the accurate baseline ⇒ the
  collapse force is systematic ⇒ search-teacher lane with the mechanism
  identified. *(This branch is dead: gates 1–2 passed.)*

### 7.12 Mid-gen-3 C2 violation audit @ 2.8M (2026-07-30)

Operator-directed case inspection of the ~50% C2 plateau (probe band 43–51%
through 2.8M). Tooling added: `scan_called_suit_leads --violations-only`
(a37c6f3), `counterfactual_called_suit_leads --cases/--cases-limit` (bbc9152),
and pi_gumbel promoted to the primary search verdict across all three
counterfactual audits per the Search_Readout_Comparison adoption (2929531;
top@Q retained for continuity).

Scan (500 seeds via the /analyze path, self-play field, ckpt 2.8M;
`runs/called_suit_nodes_2800k.json`): pooled adherence 43.9% (141/321) —
matches the greedy probe. Structure: adherence RISES with trick (33% t1,
45% t2, 55% t3–4); violations concentrate at tricks 1–2 (127/180). Late
deviations look like judgment: 66% of late violators are trump-void (median
0 trump) vs 6% of early violators (median 2). Positional gradient survives
within tricks 1–2 (picker+1 52% → picker+4 30%, monotone; not a trick-mix
confound). Violation margins are near-ties: median chosen−conv prob gap
0.138, 37% within 0.10 — favorable for entropy-ladder conversion (§8).
Motif: violators overwhelmingly lead a fat fail (10/A) over a low
called-suit card — "cash points early" vs partner-identification tempo.

Counterfactual ladder on the 10 highest-margin tricks-1–2 violations
(4096 iters to terminal, 200 rollouts, 512 belief worlds, frac=1.0;
`runs/cf_called_suit_top10_2800k.json`), all ESS-valid, gumbel/topQ verdicts
agree 10/10 at this budget:

- **Search rejects the policy's actual lead in 10/10 cases** (argmax
  agreement 0%). The learned "lead fail points early" choice is never
  endorsed at these nodes.
- Replacement splits 5/5: half flip to the convention lead; the other half
  to a LOW fail (often same suit as the spurned 10 — e.g. 10H→9H/7H):
  probe the suit without donating the 10.
- Verdicts track grounded value: search-says-conv cases mean true-deal
  Δ +0.24 / belief Δ +0.08 vs +0.06 / −0.03 for search-says-alt.
- Group value deltas are small (true-deal +0.150 ± 0.163, belief
  +0.024 ± 0.047): at these extreme-margin nodes the conv-vs-best-alt edge
  is modest; the clear error is the specific point-cashing lead, not
  always the suit choice.

Reading: consistent with §6/§8 mechanism (weak per-node signal, near-tied
mass) and with the E-study's ~17.5% legitimate-exception rate — expected
C2 ceiling well below 100% but well above the current ~44%. No action
taken mid-generation; rung-3 stratified counterfactuals (relPos gradient)
remain the follow-up if C2 fails to climb once entropy steps land.

### 7.13 Gen-3 boundary (2026-07-31): first null h2h; slope carries the verdict

Gen-3 trainer (first generation under the active entropy controller) ran
2M->3M clean; exploiter gen-3 gate FAIL (main survives): best candidate
ckpt 3,030,000 edge -0.0258 +/- 0.0160 over 3000 duplicate-bridge deals.
Gate trajectory gen-1/2/3: -0.028 / -0.184 / -0.026 - the gen-2 "hardening"
did not repeat; exploiters are again near-even. Boundary (01:07):

- Panel ABSOLUTE +0.1618 [+0.1303,+0.1931] (called +0.116, jd +0.207).
- gain_vs_best (vs gen 2, paired) +0.0294 [+0.0011,+0.0572] p=0.043 ->
  gain_improving FALSE (first sub-threshold gain; prior gains +0.120,
  +0.090).
- h2h gen3-vs-gen2 (duplicate-bridge) -0.005 +/- 0.013, win frac 0.513 ->
  FIRST NULL h2h (prior: +0.078, +0.101).
- slope +0.0596 [+0.0446,+0.0748] climbing=True -> flat=False on the slope
  criterion ALONE; streak 0; below min_generations floor (4). NO entropy
  step (absorption requires flat).
- Entropy controller, first active generation: targets held (bumpless -
  play 0.7453 throughout); inner loop reconfigured coefficients to hold
  them: pick alpha 0.0455->0.0072, partner ->0.005 (ALPHA_MIN), bury
  0.0362->0.0020, play 0.0136->0.0269 (~2x legacy - the controller pushes
  UP against the organic -0.057/gen play-entropy drift, by design).
- Gen 4 launched 01:07:01; controller resumed from sidecar with held
  targets/alphas (log line confirms).

Reading: the deceleration is now sharp (h2h null; gain sub-threshold;
only the trailing 3-gen slope keeps flat=False). Interpretive caveat ON
THE RECORD: gen 3 is also the first generation where the controller
actively suppressed the organic play-entropy decline that accompanied
gens 1-2 progress - the hold itself may contribute to the plateau. That
confound is priced into the design: the response either way is the same
gated mechanism (a flat verdict steps the play target 0.745->0.629 and
resets the streak). Likeliest path: gen-4 boundary flat -> first ladder
step fires. Watch items for gen 4: h2h recovery or confirmed plateau;
exploiter gate (two near-even gates in a row would end the
gate-trajectory-hardening story); C2/fat-fail motif at boundary ckpts
(atlas E6 results due morning of 07-31).

### 7.14 Gen-4 boundary (2026-08-02): FIRST FLAT -> first entropy step fires

Gen-4 (targets held at 0.745 throughout; live Hn 0.76-0.78): exploiter
gate FAIL -0.0668 +/- 0.0164 (series -0.028/-0.184/-0.026/-0.067: bounded
bouncing, no exploitability trend). C2 at 4M scan: pooled 40.7%, trick-0
34.5%, core-proxy (early non-void) 39.9% - flat in band vs 2.8M, as
predicted while entropy holds. Boundary (01:17):

- Panel +0.1604 [+0.1288,+0.1916] (called +0.114, jd +0.207) - flat vs
  gen-3's +0.1618.
- gain_vs_best (vs gen 3) -0.0014 [-0.0287,+0.0258] p=0.918 - null.
- h2h gen4-vs-gen3 +0.001 +/- 0.013 (win 0.511) - second consecutive null.
- slope (gens 2-4) +0.0140 [-0.0000,+0.0280] climbing=False.
- All three criteria quiet -> **flat=True, the run's first** ->
  **ABSORBED by the first play-target step 0.745->0.629** (exactly the
  pre-registered geometry: 0.28+0.75*(0.745-0.28)); streak RESET to 0
  ("improvement signal within the last two generations" per stop rule).
- Sidecar verified stepped (play 0.6290); gen-5 launched 01:17:51 and
  resumed with the stepped target (log line confirms).

This boundary opens the run's key experiment (the hold-confound test from
7.13): gens 3-4 plateaued (h2h +0.078/+0.101 -> -0.005/+0.001) with the
controller suppressing organic play-entropy decline. Gen 5 trains with
0.116 of that entropy released in a controlled step. Pre-registered
readings of the gen-5 boundary:
- h2h gen5-vs-gen4 positive again => plateau was (at least partly)
  entropy-hold-limited; ladder mechanism validated; expect another flat
  ~gen 6-7 -> step 2 (0.629->0.542).
- h2h null again => plateau is not entropy-binding at this rung; flats
  keep stepping the ladder until floor (~0.39 after 5 steps), after which
  flats count toward stopping (auto-stop path).
Watch items: play Hn should transition sharply to ~0.63 in the first
~10-20 updates (controller gain sizing 8.5) - verify early; C2 core-cell
adherence (39.9% at 4M) is the predicted near-tie conversion metric;
play softband < 0.5 canary now live-relevant; B2 bounds; exploiter gate
under a sharper policy.

### 7.15 Gen-5 boundary (2026-08-03): entropy step VALIDATED - progress re-ignites

First generation after the 0.745->0.629 play-target step. TIMING
CORRECTION to 8.5's gain sizing: measured play entropy descended
0.740->~0.63 over ~the whole generation (603 updates), not 10-20 - alpha
moves fast but the plant (policy entropy) integrates at the policy
gradient's own pace. Gen 5 therefore trained while TRANSITIONING; gen 6
is the first generation wholly at 0.629.

Boundary (23:02): panel +0.2220 [+0.1930,+0.2505] - best of run by far
(called +0.190, jd +0.254; called-mode edge nearly doubled);
gain_vs_best (vs gen 3) +0.0602 [+0.0324,+0.0882] p=0.0000; h2h
gen5-vs-gen4 +0.107 +/- 0.014 (7.9 sigma - gen-1/2-scale progress);
slope climbing again; flat=False streak 0. Exploiter gate FAIL -0.1629
+/- 0.0146 (2nd-strongest main win; series -0.028/-0.184/-0.026/-0.067/
-0.163) - the SHARPER policy is HARDER to exploit: removed mixing was
regularization tail, not equilibrium content.

**Hold-confound verdict (7.13/7.14 pre-registered): the gens 3-4 plateau
WAS entropy-hold-limited.** h2h went null/null under the hold and +0.107
immediately after release. Ladder mechanism validated as a progress
lever; expect progress to continue then re-flatten -> step 2
(0.629->0.542) at some later boundary.

**C2 conversion prediction: NOT yet fired.** 5M scan: pooled 43.8%,
core-proxy (early non-void) 38.2%, trick-0 34.9% - all still in the
oscillation band despite the step and the big general gain. The near-tie
conversion at C2 lead nodes either needs a full generation AT the lower
target (gen 6 read) or is SNR-bound independent of entropy (the 7.12
contingency). Two more null reads at gens 6-7 with h2h progressing would
point at SNR and the node-selective distill contingency.

Picker stats at 5M: picker_avg +1.29-1.54 range late-gen (vs +1.22 gen
4), pick% 13-18. Watch into gen 6.

### 7.16 BUG (2026-08-04): limited points head FROZEN all run - oracle-aux variable shadowing

Operator observation: aux audit shows the 5.3M league agent's per-seat
points-prediction MAE at ~7-10 vs ~0.6-0.7 for the reference swish 5M
agent (deterministic bookkeeping task; order-of-magnitude regression),
seen-trump mask also worse.

Diagnosis (ppo.py `_update_minibatch`): with `critic_mode=oracle` +
`--oracle-aux-heads` (this run's config), the oracle aux block did
`membership_loss, points_loss = self.oracle_critic.aux_losses(...)` -
REUSING the name `points_loss`, which at that point held the limited
critic's points aux loss. total_loss (assembled after) therefore added
the ORACLE's points loss twice (effective oracle points coeff 0.4, not
0.2) and the LIMITED critic's points loss never entered total_loss: the
limited points head received ZERO gradient. Introduced 1494902
(2026-07-24, "Add official oracle aux heads"); the entire retention run
sits after it.

Decisive evidence: `points_head.{weight,bias}` byte-identical from the
50k to the 5.3M checkpoint; every other critic component moves (adapter
0.30, secret 0.61, return 0.51 max-|dW|). Only points_head is frozen -
it is the 400k-selfplay-warmstart-era head reading an adapter that has
since rotated under it (hence MAE ~7-10, worse than merely stale).
Telemetry gap: the trainer accumulates points_loss internally but
league_training_progress.csv never exported aux losses, so a flat
points_loss was invisible to run monitoring.

The seen-trump gap is a SEPARATE, milder effect: that head trains
(weights move), but the league path takes ~2.6x fewer optimizer steps
per episode than the reference run (Adam step 54,660 vs 144,780 at 5M
episodes - larger batches per update), and it is a cross-architecture
comparison (v2 shared-readout vs full pools). Supervised bookkeeping
converges with steps, not episodes. A knock-on from the bug is
plausible too: the adapter lost the points-task shaping signal that the
seen-trump head shares.

Fix (same day): oracle losses renamed `oracle_membership_loss`/
`oracle_points_loss` + regression test asserting the limited points
head moves under an oracle-aux update (verified test FAILS on pre-fix
code). Policy impact of the bug itself: no direct policy-gradient
corruption (aux only), but the shared encoder lost the points-tracking
auxiliary shaping for the whole run - possibly relevant to the C2
point-donation gap (hypothesis only, not evidenced).

DEPLOYMENT NOTE: the running gen-6 trainer keeps the OLD code (already
imported); the fix takes effect when the orchestrator spawns the gen-7
trainer at the next boundary (~Aug-5 late evening). At that point (a)
the limited points head resumes training from its stale state and its
gradient again shapes adapter+encoder (restores the pre-registered aux
design), (b) effective oracle points coeff drops 0.4->0.2 (intended
value).

OPERATOR DECISION (2026-08-04): let gen 7 pick up the fix - no pin, no
relaunch (training too expensive; trajectory mildly corrupted but
progress strong, a few generations should re-train the head in place).
Gen-7+ watch: points aux converging (aux_audit spot-check), no
h2h/panel regression at the changepoint.

REVISED (2026-08-04, ~30 min later): operator chose instead to RESTART
GEN 6 from the gen-5 boundary with the fix (gen 6 was <40% done; the
gen5->6 boundary is the natural changepoint). Executed 13:07: SIGSTOP
orchestrator -> SIGTERM gen-6 trainer (rc=-15) -> archived the 6
partial old-code checkpoints (5.05M-5.30M) + the 6 partial league
snapshots to runs/league_retention_pg/gen6_oldcode_partial/ -> SIGCONT;
the orchestrator's attempt-2 retry re-resolved _resume_for(6) to
checkpoint_5000000 and spawned a fresh trainer (new process = fixed
code). Verified: resumed at episode 5,000,000; telemetry trimmed
(220/6/3 stale rows); entropy controller resumed targets incl. play
0.6290. Accepted deviations, recorded: (a) the 6 gen-6 snapshot adds
had evicted ~6 lowest-skill past_mains at the 30-cap - files deleted,
not restorable, so restarted gen 6 opens with 24 past_mains and refills
at 50k cadence; (b) surviving members' ratings/exploit-EMAs carry 0.35M
episodes of abandoned-trajectory drift (self-correcting); (c) sidecar
alphas are mid-gen-6 values (controller re-adapts in a few updates;
targets unchanged). Gen 6 is therefore the FIRST generation with the
limited points head training AND wholly at play target 0.629 - the
gen-6 boundary reads h2h + C2 + points-aux recovery together (aux_audit
spot-check vs the frozen-era 3.47 t01 MAE). NOTE: this consumed gen 6's
trainer retry; a later gen-6 crash raises NeedsReview and halts the
orchestrator until manually resumed.

CONFIRMED (first fixed snapshot, 5.05M, ~55 updates): points_head
max|dW| 0.037 vs the boundary (old-code 5.05M: exactly 0); aux_audit
60-game t01 points MAE 0.93 (frozen-era 5.3M: 3.47; even 1M was 1.70),
pooled 2.20, P(err<=5) 0.855. One update-slice recovered most of the
gap toward the reference (t01 0.28); mature trunk + supervised head =
fast convergence, as expected. Aux-recovery watch item satisfied;
remaining gen-6 boundary reads are h2h and C2.

### 7.17 Gen-6 boundary (2026-08-06): consolidation gen; aux fully recovered; C2 null #1 at settled target

Context: gen 6 = the RESTARTED generation (7.16: aux fix live, roster
opened at 24 past_mains, first gen wholly at play target 0.629).

Boundary (15:26): panel +0.2063 [+0.1766,+0.2351] (called +0.168, jd
+0.244); gain_vs_best (vs gen 5) -0.0157 [-0.0397,+0.0082] p=0.20 -
first negative point estimate, statistically null; h2h gen6-vs-gen5
+0.018 +/- 0.011 (1.8 sigma; win_frac 0.507) - far below gen-5's
+0.107; slope +0.023 [+0.009,+0.037] still climbing (window includes
the gen-5 jump) -> flat=False streak 0. Exploiter gate -0.098 +/- 0.013
(series -0.028/-0.184/-0.026/-0.067/-0.163/-0.098: bounded bouncing).

Reading: CONSOLIDATION. The step-release gain was concentrated in gen 5;
gen 6 held it (panel overlap, h2h marginally positive) but did not
extend it. If gen 7 posts h2h ~null with a quiet slope, that is the
second flat -> step 2 (0.629->0.542) fires. Restart perturbations
(roster hole, re-engaged points-aux gradient into the shared encoder)
are a possible mild damper on this gen; not separable from plateau.

**Aux recovery COMPLETE in one generation** (60-game audit, 6M): points
t01 MAE 0.36 (frozen era 3.47; reference swish 5M 0.28), pooled 0.77
(ref 0.59), P(err<=5) 0.991, P(err<=10) 1.000, t45 1.26, seen Brier
0.0019. The order-of-magnitude deficit the operator caught is closed;
7.16 fix validated end-to-end.

CROSS-RUN COMPARISON vs 30M reference (final_pfsp_swish_ppo.pt,
2026-08-06, operator-requested). Strength: duplicate-bridge h2h, 6M
seated all seats vs all-30M table, 1000 CRN deals/mode: edge +0.011
+/- 0.019 (called +0.002+/-0.025, jd +0.019+/-0.029, win 50.1%) =
STATISTICAL PARITY at ~5x fewer episodes (6.4M incl. warmstart vs
30M). Conventions (matched-seed greedy probe + 500-seed sampled scan):
partner trump lead 100% both; defender fail lead ~96% (t0 trump-lead
4.0%) vs ~98% (2.2%) - both near the residual-leads-are-mildly-bad
optimum; C2 called-suit lead pooled 40.4%/trick0 31.0% (6M) vs
86.2%/93.4% (30M) - the ONE large behavioral divergence: 30M
OVER-adheres relative to the E6 optimal ceiling (~60-70%), league
UNDER-adheres; equal strength either way (consistent with E6's small
per-node stakes at C2 nodes). Panel +0.206 measures edge over the
mixed 4-anchor table, NOT superiority over the 30M itself.

CORRECTION (2026-08-06, operator): the initial read of this comparison
("C2 clearly learnable at this strength level, so the league's
non-acquisition is SNR/equilibrium-path not capability") was WRONG as
stated - the 30M's C2 came from ~15M episodes of explicit reward
shaping for called-suit leads, so its adherence is evidence that
shaping INSTALLS the convention, not that terminal-only rewards can
discover it. Whether the optimal rate is discernible from terminal
rewards alone is an OPEN question and this run is the experiment.
Coherent picture: shaped lineage overshoots the E6 optimum by
construction (paid to lead called suit -> 93% trick-0 incl.
search-refuted nodes); terminal-only lineage stalls under it (~40%,
where per-node SNR runs out); optimum ~60-70% sits between; NEITHER
signal found it. Sharpens the contingency: node-selective pi_gumbel
distill is the middle path - search-grounded rather than hand-shaped,
targeting the rate itself rather than the action unconditionally.

**C2: null read #1 at the settled target.** 500-seed scan @6M: pooled
40.4% (5M 43.8%), core-proxy 34.1% (38.2%), trick-0 31.0% (34.9%) -
all within the oscillation band, slightly down. The "gen 6 = cleaner
read" hope from 7.15 did not materialize as conversion. Per the 7.15
decision tree, one more null at gen 7 points at SNR and the
node-selective pi_gumbel distill contingency (target nodes from the
scanner; E6 atlas classifier defines the conv-correct cells). Caveat:
"h2h progressing" is only weakly met this gen (+0.018), so the
entropy-vs-SNR attribution stays open until gen 7.

### 7.18 Gen-7 boundary (2026-08-08): SECOND FLAT -> step 2; C2 null #2 = SNR verdict, distill contingency ARMED

Boundary (16:47): panel +0.2122 [+0.1825,+0.2410] (called +0.164, jd
+0.261); gain_vs_best (best = gen 5) -0.0098 p=0.42 null; h2h
gen7-vs-gen6 +0.019 +/- 0.011 (win 0.512) - second consecutive
marginal ~+0.019 read; slope -0.005 [-0.017,+0.007] QUIET (first
non-climbing window). All three criteria False -> raw flat=True ->
ABSORBED by entropy step 2: play 0.629->0.542. Per the 7.15 plant
timing, gen 8 trains transitioning, gen 9 is the first generation
wholly at 0.542. Exploiter gate -0.100 +/- 0.015 (series
-0.028/-0.184/-0.026/-0.067/-0.163/-0.098/-0.100; bounded bouncing).

Post-step-1 profile now complete: one surge generation (+0.107) then
two marginal ones (+0.018, +0.019). If step 2 repeats it, gen 8/9
posts a surge; if not, ladder value is diminishing and the at-floor
endgame (7.17 stop-rule arithmetic) approaches.

**C2: null #2 CONFIRMED - and directional.** Three consecutive
declining reads as the ladder tightened: pooled 43.8 -> 40.4 -> 37.8,
core-proxy 38.2 -> 34.1 -> 31.2, trick-0 34.9 -> 31.0 -> 26.3
(SE ~2.5%/read; ~2 sigma pooled decline over two gens). The
pre-registered 7.15 condition (gens 6-7 C2-null with h2h progressing)
is met, with the caveat that "progressing" is weak (+0.019@1.8 sigma
x2). VERDICT: terminal-reward PG at this SNR is not converging toward
the E6 optimum at C2 nodes - it is SHARPENING ONTO the anti-convention
mode (entropy reduction locks in the non-adherent argmax). The
node-selective pi_gumbel distill contingency is formally ARMED
(activation = operator decision; training-signal change).

Pre-registered prediction for the step-2 window: if the
sharpening-onto-wrong-mode mechanism is right, C2 keeps eroding at
0.542 (pooled ~mid-30s or below by gen 9); a REVERSAL at lower entropy
would falsify the SNR story and reopen the entropy-timing account.
Recommendation recorded: hold distill activation until the gen-8/9
step-2 read (a clean changepoint; if h2h re-surges the ladder still
has value and the distill arm can target the post-ladder policy).

CORRECTION (2026-08-08, operator-prompted): the "directional erosion /
sharpening onto the anti-convention mode" verdict above is
UNSUPPORTED - it aliased an oscillation. Generation-POOLED greedy
telemetry (20 probes/gen, n~1,500 leads/gen, SE ~1.3%): gen 5 41.8%,
gen 6 48.9%, gen 7 40.7%, within-gen range 30-60% (operator flagged
the 6.85M probe at 47.6% two reads before the 7M trough at 36.1%).
Gen 6 - wholly at the stepped 0.629 target - was the HIGHEST pooled
read of the run, directly contradicting the entropy-lock-in mechanism.
Single-checkpoint boundary scans (n~300 nodes) cannot distinguish
trend from oscillation phase at this amplitude; henceforth C2 trend
calls use generation-pooled telemetry, boundary scans only for
node-level structure (core-cell splits, seeds). What SURVIVES: three
generations without conversion toward the E6 optimum (the state
oscillates around ~42-45% pooled-greedy regardless of entropy), so
null #2 = "no conversion" stands, the SNR-as-no-gradient-pressure
account (7.12/erosion-study form: values near-tied, adherence
wanders) stands, and the distill contingency stays ARMED. What is
RETRACTED: the claim that the ladder is actively eroding C2, and the
gen-9 "mid-30s or below" erosion prediction - replaced by: gen-8/9
POOLED means outside 38-52% in either direction would be the first
real trend signal; inside that band = oscillation confirmed,
entropy-independent.

## 8. Adaptive entropy program (2026-07-28, operator-directed)

### 8.1 Motivation

The run trains under `--schedule-horizon 20_000_000`, calibrated for the
docstring's 6×5M regime. This run is 1M-episode generations with auto-stop
(min 4 gens; realistic total 4–10M), so the linear entropy decay never
leaves its high region: at 1.78M the pick coefficient sits at 0.046 (92%
of start) and even at 10M it would be 0.028 vs the 0.005 end value. Two
consequences as originally stated: (1) the entropy bonus biases learned
optima toward Boltzmann-flat wherever Q-gaps are small — *this premise was
REFUTED for pick by the backfill (§8.4): pick is near-deterministic per
node; the 42–45% greedy pick rate is confident behavior, not flattening*;
(2) the flat-streak stop rule cannot distinguish "converged" from
"entropy-limited", so the orchestrator could read a regularization ceiling
as convergence and halt early — *this survives, narrowed to the play head
(§8.4)*. Operator has no fixed budget; wants the highest-quality model,
with convergence driven by data rather than a clock.

### 8.2 Adopted design: two-loop control (operator-approved)

- **Inner loop:** per-head coefficient becomes a feedback controller
  holding measured normalized entropy at a target (SAC automatic
  temperature adjustment, Haarnoja et al. arXiv:1812.05905 §5; discrete
  form Christodoulou arXiv:1910.07207 — targets as fraction of max
  entropy). Coefficient floors + leaster watchdog stay as collapse
  insurance (the watchdog is already a bang-bang controller in the upward
  direction; its kick multiplies ON TOP of the controller's α).
- **Outer loop:** targets step down only at generation boundaries and only
  on a flat h2h verdict (CI-positive ⇒ hold). A flat generation triggers a
  step and RESETS the stop-rule streak; flatness only counts toward
  stopping once targets sit at floor. This removes the
  converged-vs-entropy-limited confound from the stop rule.
- Entropy anneals to a small floor, never zero: imperfect-info equilibria
  are genuinely mixed and entropy regularization has convergence support
  there (Sokota et al. arXiv:2206.05825 magnetic mirror descent;
  quantal-response/regularized equilibria). The legacy schedule's
  end-values (0.001–0.005) already encode this.
- Alternatives surveyed and rejected: meta-gradient self-tuning (Xu et al.
  arXiv:1805.09801; Zahavy et al. STACX arXiv:2002.12928) — most
  principled single-run option but requires differentiating through the
  update on a freshly-validated recurrent PPO path; PBT/PB2 (Jaderberg et
  al. arXiv:1711.09846; Parker-Holder et al. arXiv:2002.02518) —
  population cost infeasible on one workstation; KL-to-prior in place of
  entropy (AlphaStar, Vinyals et al. 2019,
  doi:10.1038/s41586-019-1724-z) — held as node-selective contingency
  only, since the retention run validated anchor-free training.

### 8.3 Phase 1: instrumentation + backfill tooling (commits 2e60bc3 + 38824e8 + 2945ad6; zero behavior change)

- Measurement: per-node policy entropy over the LEGAL action set,
  normalized by ln(n_legal), per head; forced moves (n_legal=1) excluded.
- Live half: measured at θ_old (first-epoch minibatches; under grad-accum
  no optimizer step lands inside epoch one — so live telemetry matches
  offline checkpoint probes, required for bumpless target derivation) in
  `_update_minibatch`; `stats["head_entropy_norm"]`; progress-CSV columns
  `ent_norm_pick/partner/bury/play` (append-only; ensure_csv_columns
  migrates); "Hn" field on the update log line. Goes live on the gen-2→3
  boundary restart.
- Offline half: `analysis/entropy_probe.py` (sampled self-play — the
  on-policy distribution, unlike the argmax greedy probe — CRN deal panel,
  side-effect-free) + `analysis/entropy_backfill.py` (checkpoint sweep →
  trajectory + derived quantities + per-row scheduled coefficients for the
  coefficient-vs-measured comparison). Tests:
  `tests/test_entropy_telemetry.py` (bounds, first-epoch-only wiring,
  probe reproducibility/side-effect freedom).
- Soft-band fraction (share of eligible nodes with H_norm > 0.3,
  `SOFTBAND_HNORM`) added to both instruments (`softband_*` CSV columns)
  as the boundary-band collapse canary the mean hides.

### 8.4 Backfill results (2026-07-28, 37 ckpts × 200 games, seed 20260728; `runs/league_retention_pg/entropy_backfill.json`)

Per-head mean H_norm (seed row = seed400k_a warm start; selected rows of
the 37-checkpoint sweep):

| ckpt | pick | partner | bury | play |
|---|---|---|---|---|
| seed400k | 0.053 | 0.166 | 0.256 | 0.882 |
| 200k | 0.052 | 0.099 | 0.162 | 0.818 |
| 600k | 0.064 | 0.108 | 0.161 | 0.815 |
| 1.0M | 0.050 | 0.055 | 0.170 | 0.758 |
| 1.4M | 0.048 | 0.099 | 0.170 | 0.763 |
| 1.8M | 0.046 | 0.127 | 0.163 | 0.754 |

Derived (bumpless targets = 1.8M values; organic drift per generation from
OLS over all 36 in-run checkpoints): pick 0.046 / −0.0003, partner 0.127 /
−0.0008, bury 0.163 / −0.0148, play **0.754 / −0.0569**.

**Headline — the entropy-inflated-pick hypothesis is REFUTED.** Pick reads
H_norm ≈ 0.05–0.07 flat across the entire run: at actual pick nodes the
policy is near-deterministic despite the 0.046–0.05 coefficient. The
42–45% greedy pick rate is confident behavior, not Boltzmann flattening;
the pick-rate drift toward the operator's ~30% optimum, if it happens,
must come from learning, not annealing. Partner and bury sit at low,
roughly trend-free levels (partner noisy 0.05–0.17 at n≈190/probe; bury
0.26→0.16 in the first 200k, flat since).

**Where entropy actually lives: the play head.** Clean monotone decline
0.88 → 0.75 with organic drift −0.057/gen under the fixed 0.0138–0.015
coefficient. The only head where (a) the regularizer is plausibly binding
and (b) a target-entropy controller has real leverage; also the head the
operator's original concern (sampling noise near convergence) applies to.

**Soft-band trajectory (same sweep; canary baselines for the hold
targets):**

| ckpt | pick | partner | bury | play |
|---|---|---|---|---|
| seed400k | 0.049 | 0.205 | 0.400 | 0.985 |
| 200k | 0.072 | 0.144 | 0.279 | 0.961 |
| 600k | 0.068 | 0.112 | 0.213 | 0.949 |
| 1.0M | 0.074 | 0.097 | 0.293 | 0.926 |
| 1.4M | 0.088 | 0.101 | 0.305 | 0.925 |
| 1.8M | 0.072 | 0.106 | 0.279 | 0.909 |

Readings: **pick's boundary band is alive and remarkably stable — 5–9% of
pick nodes genuinely mixed across the entire run** (a committed core plus
a thin soft boundary band; the structure the hold target exists to
protect — a collapse reads as a sustained fall toward 0 against this
steady baseline). Partner oscillates 5–21% (n≈190/probe — noisy; the 800k
dip to 0.047 recovered unaided). Bury declined 0.40→~0.28 with the early
sharpening then stabilized. Play 0.985→0.909: even at 1.8M, 9 in 10 play
nodes remain genuinely mixed — the play-target ladder has real room before
the band thins. **Canary thresholds** (informal, operator review if
breached sustained over ≥3 probes): pick softband < 0.03; play softband
< 0.5 while the ladder is above floor.

**Consequences for Phase 2:** the outer loop's first and main lever is the
PLAY target (a 0.25-of-gap step from 0.754 toward a ~0.28 floor moves
~0.12 — comfortably above the −0.057/gen organic drift). Pick/partner/bury
controllers exist to HOLD their measured operating points (collapse/drift
insurance), not to anneal; their coefficients may fall to near the floor
since measured entropy already sits at target without help. The stop-rule
confound is narrower than feared but real: a plateau cannot be
entropy-limited via pick/partner (not binding), but can be via play — the
plateau→step→reset-streak rule applies to the play target. (This
hold-vs-anneal split also answers the "keep pick entropy high until play
converges?" question: hold at the measured operating point — pick's thin
boundary band migrates via generalization from play improvements, not via
added entropy pressure.)

### 8.5 Phase 2 controller (commits 3117ca9 + 73dfdbd) + pre-registered hyperparameters

Code: `training/entropy_controller.py` (inner loop + JSON sidecar
persistence, `checkpoints/entropy_controller.json`), trainer
`--entropy-mode target`, orchestrator `--adaptive-entropy` with the
stop-rule absorption amendment (flat + targets above floor ⇒ play-target
step + streak reset, replay-idempotent via the generation record). Tests:
test_entropy_controller.py + extended test_entropy_telemetry.py;
end-to-end absorb ladder verified against a real sidecar (5 steps
0.754→0.392, then at-floor flats stand).

**The three controller hyperparameters, as derived from the backfill:**

1. **Starting targets — bumpless, measured, not chosen.** Each head adopts
   its first live measurement at switch-on (the controller's first act is
   to hold the status quo; Åström & Wittenmark, *Adaptive Control* 2e 1995
   ch. 9). α initialization is also bumpless (adopts the legacy schedule's
   value at the current episode). Per-head `--entropy-target-*` flags
   remain as expert overrides only.
2. **Inner gain η = 1.0 (log-space), |Δlog α| ≤ 0.1/update.** Derivation:
   the plant timescale is the backfill's organic play drift (0.057
   H_norm/gen ≈ 0.001/update); at η=1 a sustained error the size of one
   outer step (~0.12) moves α ~12%/update ⇒ settles in ~10–20 of a
   generation's ~61 updates — fast relative to the outer cadence, slow
   relative to per-update noise (SE ≈ 0.002–0.004 at 16k rows ⇒ ~0.3%
   α-jitter, two orders under the clamp). The gain is deliberately
   uncritical: targets move rarely and α has a whole generation to settle;
   the clamp bounds any transient.
3. **Outer step — geometric, retain 0.75, min_step 0.03, play only.**
   `target ← floor + 0.75·(target − floor)`; first step 0.754 → 0.635
   (Δ≈0.12 ≈ 2× organic drift ⇒ distinguishable from the null; PBT-scale
   perturbation, Jaderberg 0.8/1.2×). At most one step per boundary, only
   on a flat h2h, monotone (watchdog is the sole upward force). min_step
   0.03 (probe checkpoint-to-checkpoint noise ±0.02, rounded up) ends the
   ladder after 5 steps at ~0.39 — **HONEST NOTE: the effective terminal
   target is ~0.39, not the nominal 0.28 floor**; steps below noise are
   not worth a 1M-episode generation each. If post-ladder evidence favors
   going lower, lowering min_step late is a one-line pre-registered
   amendment. Floors: play 0.28 (mixed-equilibrium reserve;
   least-data-grounded number, protected by per-step h2h gates), hold
   heads floor-at-target.

**Coefficient bounds:** α ∈ [legacy end, 4× legacy start] per head — the
legacy schedule's "small, not zero" endpoint becomes the hard floor; the
cap leaves collapse-fighting headroom without runaway.

### 8.6 Activation semantics (final form: gen-1 deferral, default-on)

Two intermediate recipes were designed and superseded the same day; the
reasoning is kept because it shaped the final form:
- *Explicit fresh-run targets* (pass all four backfill values) — wrong
  twice over: bumpless-everything at a SEED would hold bury at 0.256 and
  partner at 0.166 against the organic early sharpening the validated run
  exhibited (bury 0.26→0.16, partner 0.17→0.10 in the first 200k; seed
  entropy levels are transients, not operating points), and importing the
  mature play target 0.754 would suppress the validated high-entropy early
  play phase (0.88→0.82) across the historically collapse-critical league
  transition.
- *`--entropy-targets-from <backfill.json>`* (hold heads from the mature
  operating point, play bumpless) — worked but required a backfill
  artifact; deleted (commit 5546cf7) when the deferral made it
  unnecessary.

**Final semantics (commit 5546cf7; default flip b97d880):**
- `--adaptive-entropy` (orchestrator) is the ONLY activation switch and is
  now **BooleanOptionalAction default TRUE** (`--no-adaptive-entropy`
  restores the pure legacy schedule). It injects `--entropy-mode target` +
  `--entropy-play-floor` into trainer commands **from generation 2
  onward**: gen 1 always runs the legacy schedule, so a seed's entropy
  transients play out exactly as in the validated retention run, and the
  controller attaches bumplessly at the gen-1 boundary's settled operating
  point — precisely the program the live run followed de facto, promoted
  to built-in behavior.
- Reproduction flow from scratch = selfplay seed → orchestrator, zero
  entropy-related flags. No seed probing, no backfill importation.
- Trainer `--entropy-mode target` standalone = hold-only experiment (inner
  loop, no ladder, unamended stop rule) — documented as such in its help.
- Lineage caveat: the validated setpoints/floor are for this arch/setup; a
  different lineage re-derives from its own backfill or relies on the
  bumpless attach with generation 1 as calibration. (Portability note:
  pick barely moved seed→league, 0.053→0.046 — plausibly a property of the
  game's pick structure, the most portable of the four numbers.)

### 8.7 LR: assessment — diagnostics only, not adopted (commit 236d0b0)

Question: should LR also be adaptive? Facts: actor+critic LR still ride
the 20M clock (1.5e-4 → 5e-5; ~1.41e-4 now, ending a realistic 4–10M run
at 1.3–1.0e-4 — i.e. nearly flat, which under Adam + PPO clip is a
defensible default; Andrychowicz et al. arXiv:2006.05990 rate LR decay
helpful but modest). approx_kl is computed every update but was never
logged; target_kl is None (the early-stop is dormant) — there is NO
trust-region feedback and NO recorded evidence that LR is mis-set.

Verdict: the justified adaptive scheme, when evidence supports one, is
KL-targeted LR (banded feedback holding per-update approx KL near a
target — rl_games/IsaacGym practice, kin of Schulman's adaptive-KL PPO),
the structural analog of the entropy controller; GNS (already logged) is
the complementary instrument (McCandlish eps_opt = eps_max/(1+B_noise/B):
B_noise growth across generations = measured case for decay). NOT adopted
now: (1) the evidence bar the entropy change met — a measured clock-binds
mismatch — is unmet for LR; the lineage-best run climbs under
effectively-flat LR; (2) the flat-boundary signal must stay single-lever
while the entropy activation's predictions are being tested. Sequencing:
entropy ladder → floor → plateaus persist ⇒ LR becomes the next
pre-registered lever (boundary halving w/ streak reset). Phase-1 analog
shipped instead: approx_kl + lr_actor progress-CSV columns (append-only,
ride the boundary restart); read a generation before any control decision.

### 8.8 Activation plan (operator GO, 2026-07-28)

Operator adopted the program for the live run at the gen 2→3 boundary. At
the boundary, once the boundary evals complete (watcher b5eg076cx fires on
"endpoint eval gen 2:"), kill and relaunch the orchestrator — before gen-3
training accumulates episodes under the schedule, so gen 3 is cleanly
single-regime. With the default flip, the standard relaunch command
activates the program with NO new flags. Bumpless predictions to falsify,
pre-registered: switch-on changes nothing measurable in gen 3; the first
target step happens only after a flat boundary verdict, and if the
entropy-limited hypothesis is right, post-step generations claw back
CI-positive h2h until the ladder floors. Guards unchanged: watchdog,
greedy gates, B2 bounds, soft-band canaries (§8.4). Expected at first
gen-3 update: "🎯 Entropy targets initialized (bumpless)" log line + the
new ent_norm/softband/approx_kl/lr_actor columns.

## 9. Config simplifications: legacy args stripped (2026-07-28, operator decisions)

All three removed outright (not deprecated), simplifying league-trainer
branching; `League.sample_table` now has exactly ONE mode — the validated
per-seat PFSP/self/exploiter mixture.

- **`--exploiter-full-table`** (+ LeagueConfig.exploiter_full_table + the
  whole-table branch in League.sample_table): the batch-λ-arm-era
  whole-table exploiter pressure deployed an uncertified
  1-main-vs-4-parameter-shared-copies mirror with no evaluation analog
  (the gate certifies 1-exploiter-vs-4-mains), and seat rotation now
  supplies the role coverage it compensated for (§7.7). This finalizes the
  Jul-26 seating amendment — the flag had already been dropped from the
  live relaunch; now the code path is gone. (b664e4c)
- **`--self-play-share`** (CLI override only): the per-seat SELF share
  remains a code-level knob, LeagueConfig.self_play_share = 0.15 — the
  validated value of the current pfsp mixture. Operator rationale:
  adjusting it is a legitimate future experiment, but that knob belongs
  inside the validated pfsp regime (a LeagueConfig change with its own
  pre-registration), not a per-run CLI surface. (b664e4c)
- **`--table-self-play`** (+ LeagueConfig.table_self_play_prob +
  League._sample_table_level — the arg the operator originally meant by
  "self-play-share"): the Phase-A-era table-level composition (pure-self
  tables with probability p, else uniform recency window, no PFSP/EMA
  weighting, exploiters audit-only) — the p_self_table 0.65 machinery of
  the failed Phase A design, kept after the revert as a knob; with the §6
  bugs fixed and the per-seat pfsp mixture validated by the retention run,
  the alternative composition regime was dead code. (a4a40c9)

The validated retention-run trainer invocation is unaffected (it passed
none of these). Tests: full-table + table-level sampling tests removed
with the branches; per-seat mixing + exploit-patched retirement tests
remain (TestExploiterSeating).

## 10. Implementation notes

- All loss/sampling changes behind config flags defaulting to historical
  behavior at introduction; `capture_arch_goldens --check` + bit-exact
  fixture suite before merge. (The validated retention config was later
  promoted to defaults, §7.9.)
- Trainer CSV schemas append-only (columns may be added, never renamed;
  `ensure_csv_columns` migrates on resume).
- `league.py` member JSON schema stays readable by old code (EMA fields
  dormant where unused).

## Appendix A: armed contingencies and standing watch items

**Contingencies (armed, not active):**
- **Search-teacher / selective-distillation lane** (§2.5): node-selective
  distillation at partner-lead nodes, pi_gumbel readout, composed with the
  low-temperature hold. Trigger: convention leaks additional training
  cannot close, or B2 failure per the amended outcome maps.
- **Node-selective λ ≈ 0.5–0.7 at lead nodes** (§7.5): fires only on
  unclosable convention leaks, per operator policy; λ otherwise 0.95.
- **Bidding anchor re-engagement** at a declared restart (§7.6): on
  PICK-gate streak or leaster-watchdog trip.
- **KL-targeted LR** (§8.7): sequenced strictly after the entropy ladder
  floors; read a generation of approx_kl/lr_actor first.
- **Oracle aux heads** are ALREADY adopted in the retention config; the
  offline-null-at-pooled-lead caveat (§4.3) stands on the record.

**Standing watch items:**
- B2 bounds: partner trump-lead ≥ 0.5 held, defender t0 ≤ 0.10.
- Greedy gates/tripwires: partner < 50%, pick < 20% (formal gate 15%,
  warning-only), defender t0 > 10%, greedy leaster > 8% (watchdog owns
  halts); ALONE warnings on record 650k/700k.
- C2 adherence (§7.9): < ~35% sustained = erosion; climb toward ~70% =
  headroom being claimed.
- Soft-band canaries (§8.4): pick softband < 0.03 sustained ≥3 probes ⇒
  operator review; play softband < 0.5 while ladder above floor.
- Entropy activation predictions (§8.8): bumpless switch-on ⇒ no
  measurable gen-3 change; targets step only on flat verdicts.
