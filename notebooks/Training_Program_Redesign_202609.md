# Release-Candidate Training Pipeline — end-to-end redesign (September 2026)

Status: **DESIGN APPROVED 2026-09-02 (operator); NOT YET BUILT.** Launch
is gated on the iteration-2 policy-iteration compounding test running on
the current lineage (CE_Teacher_Design §20.12). This notebook is the
design contract, the pre-registration, and the write-up scaffold for the
program's final run: a from-scratch training of the deployable agent on
the `perceiver-recall` architecture through a simplified, validated
pipeline that ends in search-Q policy iteration.

Predecessors (the evidence base; nothing here re-argues them):
[Architecture_Ablation_202607](Architecture_Ablation_202607.md),
[Learning_System_Redesign_202607](Learning_System_Redesign_202607.md),
[Convention_Erosion_202607](Convention_Erosion_202607.md),
[Convention_Optimality_202607](Convention_Optimality_202607.md),
[Search_Teacher_Design_202608](Search_Teacher_Design_202608.md),
[CE_Teacher_Design_202608](CE_Teacher_Design_202608.md),
[Blind_Bury_Ablation_202608](Blind_Bury_Ablation_202608.md).

---

## 0. Decision log

| Date (2026) | Decision | § |
|---|---|---|
| 09-02 | Restart the program from scratch on `perceiver-recall` to produce the final deployable artifact; simplify the pipeline end to end | 1, 3, 4 |
| 09-02 | Observation contract: `blind_ids`/`bury_ids` REMOVED from `get_state_dict`; legacy architectures read them through a separate `Player.get_picker_memory` interface merged per agent by `observation_for` (an earlier same-day amendment had kept the keys in the dict with encoder-side enforcement) | 3.2 |
| 09-02 | Skip the recall-vs-ctxmem architecture pilot; no strength bar on the bootstrap | 3.4, 5.3 |
| 09-02 | PG phase stop rule = marginal-value handoff (not plateau); one entropy step max; settled-checkpoint handoff | 5.1 |
| 09-02 | Exploiters dropped from the training loop (post-hoc audit only); PFSP kept | 4.3 |
| 09-02 | Shaped bootstrap KEPT (400k), run in the unified trainer; fallback = legacy self-play trainer | 4.1 |
| 09-02 | Bidding channel = frozen-trunk PG phase between iterations; bidding emission = contingency | 4.4 |
| 09-02 | Leaster play: measure drift first; fixed-reference anchor, then emission, if it drifts | 4.4, A |
| 09-02 | Validate iteration 2 of policy iteration on the CURRENT lineage before building/launching | 7 |
| 09-12 | §7.0 gate RESULT: the standard recipe STALLS at theta_1 on every axis (targets, schedule, acting mode, rows 35k→140k, budget 256/1024); the OPTIMISED recipe compounds small-positive (+0.003 ± 0.002 play per iteration; D8k_t6 +0.0038 ± 0.0019 at 2σ, replicated) — CE_Teacher §20.13 add. 24-27, §20.14 | 4.4, 7.0 |
| 09-12 | Phase 3 recipe pinned to CE_Teacher §20.14: 8000 committee-acted games, 256 iterations with the trick-indexed lead schedule (t0 leads 1024, t1 leads 512), six trunk epochs @3e-5 then bilinear-only head epochs, retention KL ×10, 8000-deal cert with head-routed play-only (compounding statistic) and bidding-only (guard) reads, NON-INFERIORITY adoption gate; single-network deployment (no head routing) | 4.4 |
| 09-12 | Committee acting RESTORED (act fraction 1.0; +0.0055 play vs student acting, add. 16) — reverses the 09-02 "decided 0" | 4.4 |
| 09-12 | No replay window: the previous generation's corpus, even re-anchored to the current prior, reads at the harm line (add. 30); one corpus per iteration | 4.4 |
| 09-12 | Bidding/value PG phase kept between iterations (play heads, adapter and encoder pinned); validation order on the v2 lineage: bidding phase on theta_2 FIRST, then iteration 3 from the adopted checkpoint | 4.4, 7.0 |
| 09-12 | `training-program-redesign` merged into master (sharded h2h kept; public helper names; `--iters-schedule`); validation run `rc_validate_v2` from `runs/policy_iteration_202609/iter29_d8k_t6/distill_epoch10.pt` | 6 |
| 09-13 | Bidding phase on θ₂ ADOPTED (+0.0054 ± 0.0036; leaster +0.024 ± 0.009; play network bit-identical) — first run of the between-iteration PG phase | 4.4, 7.0 |
| 09-15 | Iteration 3 under the RC pipeline COMPOUNDS: play-only +0.0066 ± 0.0024 (last epoch) / +0.0036 ± 0.0023 (KL-best epoch) over θ₂′; full network +0.0058 ± 0.0026; bidding route −0.0010 ± 0.0008; called-suit 45.6 → 56.3 under the lead schedule (CE_Teacher add. 31) | 4.4, 7.0 |
| 09-15 | Distill candidate = the LAST epoch of the schedule; the held-out-KL plateau rule retired (it stopped the head phase after one epoch and discarded +0.003 of certified play); fit defaults aligned to 200 epochs / patience 25 | 4.4 |
| 09-15 | Routed-chimera observation defect fixed (`needs_picker_memory` delegation); orchestrator log restructured with phase banners, sub-headers, stage/decision markers and per-phase timings; README training guide rewritten | 6 |
| 09-16 | Pre-launch config audit: final references + the v2 release, final h2h 8,000 deals; league workers on MPS + torch.compile (bit-exact, 1.36x); seen-trump recall added to the greedy probe (informational); goldens/export = manual step; `--dry-run` side-effect fix | 4.3, 4.5, 7.1 |
| 09-16 | `rc_validate_v2` COMPLETE: release = θ₃ + bidding phase; +0.0505 ± 0.0062 vs the 30M, +0.0135 ± 0.0046 vs iter11 P1 (8,000 deals/mode) — the program's strongest agent and the fresh run's bar; two consecutive bidding phases lowered the pick rate ~2 points each (watch in the fresh run) | 7.0 |
| 09-16 | PANEL-B added (tentative): strong-skill / cross-ecology yardstick (30M, v2 release, iter11 P1, v2 8M seed), recorded per generation and at the final, no gate; membership provisional until the v2 release's reads vs the 30M land | 5.3 |
| 09-16 | CE_Teacher_Design investigation CLOSED (§21); the v2 lineage ends at θ₃ = `rc_validate_v2/pi/iter1/distill_epoch7.pt` (+ its bidding phase); next: the fresh perceiver-recall run | 7 |
| 09-16 | `202609_recall_rc` LAUNCHED 13:25 (bootstrap escape at ~5k episodes, watchdog engaged once). Seen-trump probe REDEFINED mid-bootstrap: all seats, recall = must-remember trumps (earlier tricks + picker bury/blind) by trick, false-seen; the picker-only/blind-bury first cut mistook the recall problem. Bootstrap checkpoints re-probed offline; later phases record the new definition (informational either way) | 7.1 |
| 09-18 | Seen-trump memory expectation RECALIBRATED against the v2 retention lineage re-probed under the new definition: reliable recall is a league-scale outcome (v2 false-seen 12 → 1% between league 0.7M and 4.7M, gens 3–5), not a bootstrap one; perceiver-recall at league 700k a few points behind v2 on the harder task, same regime; trick 0 erratic for both | 7.1 |
| 09-18 | AUX-HEAD READINESS GATE added as the handoff's precondition (§5.4): the four deterministic heads (seen-trump, unseen-higher, known points, secret partner) must be essentially never wrong on the boundary battery before the league hands off — the league is the only phase that builds the trunk memory they read; deferral = continue, cap without readiness = NEEDS REVIEW. Bars from v2 at 7.7M. Probe extended to the other three heads; reads recorded per generation and at every PI cert | 4.3, 5.4, 7.1 |
| 09-18 | Deterministic aux-head loss coefficients ×2.5 (`aux_det_scale`, every stage that trains them; win/return unchanged) to shorten the memory transient; applied from `202609_recall_rc` league gen 2 (bootstrap and gen 1 ran at 1.0). Unweighted aux losses now logged per update. The running orchestrator (old code) is replaced at the gen-1 boundary by a resume under the new code | 4.3, 5.4 |
| 09-19 | `202609_recall_rc` GEN 1: h2h vs seed +0.2482 ± 0.0149 (v2 retention gen 1: +0.0781 ± 0.0134), PANEL-A +0.178 from a seed at −0.065 (v2: +0.043 from −0.077; gains +0.243 vs +0.120), PANEL-B −0.096, t0 defender trump lead 7.2% (seed) → 0.6% (v2 gen 1: 2.0%); aux readiness NOT READY on seen-trump alone (acc 90.6, false-seen 14.3); secret / points MAE / unseen-higher already at their bars. Gen 2 launched 08:42 at ×2.5. Operator theory "the bilinear play pointer is a big part of the gain" — three cheap reads (`runs/202609_recall_rc/recall_compare/`): (a) head-routed h2h vs the seed: bidding-only route +0.192 ± 0.014, play-only +0.071 ± 0.011 — the gain is ~¾ BIDDING (the shaped seed's over-eager picker, greedy PICK 56%, corrected by terminal reward), not play; (b) zeroing the bilinear term (U=0, exact removal) costs gen 1 −0.606 ± 0.014 and the SEED −0.707 ± 0.016 — the term became the play pathway inside the 400k bootstrap, so reliance says nothing about gen 1's gain. Verdict: NOT SUPPORTED as the main driver of gen 1; the surviving form (a bilinear scorer lets PG move PLAY faster than an additive one) is only answerable by the ablation arm (no-bilinear bootstrap + gen 1 under this pipeline, ~3 machine-days; a FUTURE OPTION, not queued). RESOLVED same day: the seeds bid alike (1,000-game greedy probes: v2 warm start PICK 48.9 / ALONE 30.5, ours 53.9 / 23.7; same leak 2.6 vs 2.9%, same memory 90.9 vs 88.8 acc) and v2's gen 1 routed vs its seed reads bidding-only +0.047 ± 0.010, play-only +0.044 ± 0.011 (`v2_gen1_routed_vs_seed.json`). So the gen-1 gap between the runs is a BIDDING-learning gap — ours extracted 4× more from the bidding head in one generation (+0.192 vs +0.047) from a seed of the same rates — while play moved similarly (+0.071 vs +0.044, 1.7 SE apart). Neither architecture change is a bidding-head change; candidates are the population (4 copies of the seed vs the July retention seeds), the memory-token GRU (feeds every head), and the unified trainer's bidding-side entropy/update settings — not disentangled | 4.3, 7.1 |
| 09-19 | League-construction audit of the v2 retention run vs `202609_recall_rc` (operator suspected the population): reconstructed from `runs/league_retention_pg_launch.log`, the retired trainer/config at `03127db^`, and `d15b506` (the code at the retention launch). IDENTICAL: the four retention seeds are byte-identical copies of the v2 400k warm start (md5 91cfe5cd97), seeded as 4 past_mains exactly as ours; `sample_table` mixture (self 0.15, PFSP weights 0.7/0.3/2.0/0.1/5.0, HOF floor 0.05, max 30 past_mains, quota 6); exploiter share 0.00 through gen 1; trainer values (LR 1.5e-4 vs 1.45e-4 interpolated; entropy 0.05/0.05/0.04/0.015 fixed vs annealed from the same starts over 20M, ≤5% apart at 1M; update 16,384; 128-episode minibatches with grad accumulation = 4.0 optimizer steps/update in BOTH (verified from `opt_steps`); 4 oracle extra epochs; γ 1.0, λ 0.95, clips, grad-norm 0.3, global advantage normalization, no target KL; terminal reward code unchanged; optimizer state resumed from the warm start in both). So the gen-1 gap is NOT league construction and NOT trainer settings. Remaining: the architecture (the memory-token GRU is the only change that feeds the bidding heads; bilinear pointer and recall constraint are play-side) and any update-path code change inside the 830-line `ppo.py` redesign diff not reflected in constants (not audited line by line). The future ablation arm should therefore be the ARCHITECTURE arm (v2 under this pipeline), not a population arm | 4.3, 7.1 |
| 09-19 | Bidding-gap follow-up (operator: pick has no recurrence in either network, so the GRU can only reach the later bid decisions; audit the PPO update path too). (a) PPO UPDATE AUDIT of `ppo.py` d15b506..HEAD (830 lines, read in full): NO substantive change to the update math — surrogate, clipping, entropy, global advantage normalization, value clipping, the 4-epoch grad-accumulation loop and the oracle extra epochs are the same logic; removed code was inert in the retention gen 1 (search-distillation loss + searched-row PG mask: no search targets; bidding-head KL anchor: coefficient 0.0; GNS diagnostic: measurement on its own forward passes); added code is measurement (normalized-entropy telemetry under no_grad, unweighted aux-loss stats, aux probe, action-prob finiteness check) or off in the league (head freezing = all; aux scale 1.0 in gen 1); gamma persisted in the checkpoint. Rollout path: reward attachment and seat rotation unchanged; the only change is the clean-observation helper for recall architectures (the recall constraint itself). Leaster watchdog engaged 0 times in both gen 1s. Pipeline EXONERATED. (b) PER-HEAD SPLIT of the bidding-only route (one gen-1 bidding head inside an otherwise all-seed agent vs the all-seed field, 2,000 deals/mode; `{ours,v2}_gen1_head_split_vs_seed.json`): ours pick-only +0.152 ± 0.011, partner-only +0.041 ± 0.008, bury-only +0.003 ± 0.007 (sum +0.196 ≈ the +0.192 route: additive); v2 pick-only +0.042 ± 0.007, partner-only −0.006 ± 0.007, bury-only +0.005 ± 0.005 (sum +0.041 ≈ +0.047). In BOTH runs the bidding gain is the PICK decision; ours moved it 3.6× further, and partner-call moved only in ours. Since pick is memory-free, the memory-token GRU is NOT the mechanism of the gap; whatever it is acts on a non-recurrent head. Greedy probes (1,000 games, `gen1s_greedy_probe_1000games.csv`): ours PICK 53.9 → 37.1 / ALONE 23.7 → 18.5 / partner trump lead 77.7 → 94.3 / called-suit 52.2 → 62.3; v2 PICK 48.9 → 42.3 / ALONE 30.5 → 35.7 / partner trump lead 90.3 → 80.2 / called-suit 54.2 → 50.0 — ours corrected the over-eager picker 2.5× as far in one generation and sharpened its conventions while v2's drifted. Training-time sampled pick share (18-20 %/seat) and picker averages track each other in both logs, so the difference is in the greedy policy, not the training mixture. The points-head gap at 1M (ours MAE 0.99 / exact 42 % vs v2 6.31 / 10 %, both seeds ≈ 1.0) is NOT a candidate: it is the oracle-aux name-shadowing bug fixed 08-04 (`45df670`) — the limited points head was byte-identical from 50k to 5.3M in the retention run (MAE 1.0 → 5.3 → 9.4, then 0.88 at 7.7M after the fix), the oracle's points loss was counted twice inside the oracle's OWN encoder (zero shared parameters), and the points label is `compute_known_points_rel` = points taken so far, all zeros at the pick node, so the points loss carries no hand-strength signal to the pick node in either run. Pick-node structure is IDENTICAL in the two architectures (operator's question, verified in the code): the pick head is the same `Linear(d_model, 2)` reading the same 16-query/4-head LayerNorm'd readout over the same transformer; at the pick node v2's blind/bury tokens are PAD and masked out of attention, the trick tokens are empty in both, and the recurrent state is the initial state in both (`observe()` runs only after completed tricks, pick is the first forward), so the memory token is a learned constant and the GRU-driver difference (context vs MEMORY token) never reaches pick. So the gap is a difference in WEIGHTS and their dynamics, not in structure: (i) the seeds' pick sharpness — same sampled-self-play entropy probe (200 games, seed 20260728): pick H_norm ours 0.083 vs v2 0.044 (gen 1: 0.080 vs 0.064; partner 0.062 vs 0.158, bury 0.19 vs 0.22, play 0.86 vs 0.87) — a 2-way head twice as sharp moves about half as fast per unit advantage (PG ∝ p(1−p)), a factor ≈2 of the 3.6; (ii) the advantage signal at pick from the oracle critic (same class and settings, different pretrained weights — not read); (iii) shared-trunk cross-talk from play-node gradients, which under the recall constraint must carry blind/bury through the memory token (direction unknown; only the architecture arm separates it). Still only separable by the architecture ablation arm (v2 under this pipeline), which remains a FUTURE OPTION | 4.3, 7.1 |
| 09-19 | Oracle-critic quality at BID nodes (operator: compare the two oracles per the suggestion). `analysis/diagnostics/critic_stratified_ev` on each run's own league checkpoints, 3,000 sampled self-play episodes each, seed 20260720 (`runs/202609_recall_rc/recall_compare/critic_ev_{ours,v2}_{50k,1m}.json`). Oracle EV — pick: ours 0.310 (50k) → 0.366 (1M) vs v2 0.396 → 0.367; partner-call: 0.353 → 0.412 vs 0.474 → 0.420; bury: 0.407 → 0.434 vs 0.489 → 0.452; all nodes: 0.529 → 0.575 vs 0.605 → 0.613. Limited critic at pick: ours 0.152 → 0.166 vs v2 0.242 → 0.157; linear hand-strength baseline at pick 0.099 / 0.098 vs 0.146 / 0.099 (the v2 warm start's early pick states carried more hand-strength-explainable return variance, consistent with its sharper, hand-driven pick head). VERDICT: the retention run's oracle was as good or BETTER at every bid stratum through gen 1 (better at 50k, equal at 1M), so a better advantage signal at pick is NOT why ours corrected pick faster — candidate (ii) eliminated. Left standing for the 3.6× pick-learning gap: the seeds' pick sharpness (H_norm 0.083 vs 0.044, ≈2× via p(1−p)) and shared-trunk dynamics under the recall constraint (architecture arm only); residual ≈1.8× unexplained. Each EV is on the run's own on-policy state distribution (the quantity that feeds its own GAE), not a common-deal comparison | 4.3, 7.1 |
| 09-19 | Did the old self-play script's entropy schedule make the v2 warm start's pick head sharper (operator's question)? NO — the schedule is the same: `BootstrapHyperparams` is `SelfPlayHyperparams` at `d15b506` verbatim (pick 0.08→0.05, partner 0.05→0.04, bury 0.04→0.03, play 0.05 flat, linear over the run's own horizon, LR 1e-4/1e-4, update 4096, 4 epochs); if anything v2's 200k→400k extension re-annealed from the midpoint, i.e. MORE integrated pick entropy bonus, not less. What differed is the ESCAPE from the all-PASS attractor: the v2 self-play run (`ablate_perceiver-shared-v2_s2042`, warm start = its 400k, md5 91cfe5cd97) ran with `Leaster watchdog: off`, its pick head saturated within ~10 updates (training-time pick entropy 0.032 → 0.000 by update ~30, JD pick rate 9% → 0.0%) and it escaped organically ~20k episodes later with the head already saturated on both sides of its boundary (pick entropy 0.001-0.012 raw for the rest of the 400k); ours engaged the watchdog at episode 1,876 (leaster 90%, pick entropy ×10 for 27 updates, released at 29%) and never saturated (training-time H_norm pick 0.10-0.20 across the whole bootstrap). Pick-node gradient mobility on the entropy probe's CRN panel (200 sampled self-play games, seed 20260728; mean p(1−p) of PICK at 2-way pick nodes = the per-node scale of the pick head's policy gradient; scratch `pick_mobility.py`): seed ours 0.0184 (soft nodes 0.1<p<0.9: 7.6 %, median H_norm 3.6e-4) vs v2 0.0092 (4.0 %, 1.2e-4) — a factor 2.0; gen 1: ours 0.0135 (5.5 %) vs v2 0.0104 (4.4 %), the league's fixed entropy bonus re-opening v2's head (its H_norm 0.044 → 0.064 over gen 1) so the mobility gap SHRINKS with generations. Accounting: pick-only edge ratio 3.6 = 2.0 (seed mobility) × 1.8 (residual), so the bootstrap difference explains about HALF of the pick-learning gap, and it is the watchdog-vs-collapse difference, not the schedule. The residual 1.8 remains the trunk under the recall constraint or noise — the architecture arm (v2 under THIS bootstrap, watchdog on) would separate the two and is the same arm already held as a future option | 4.3, 7.1 |

---

## 1. Objectives and constraints

Objectives (operator, 2026-08-30 restatement + this program):

1. **Validate the full training system end to end from scratch**,
   incorporating every learning of the last two months and the
   observation/recall restrictions the program was founded on.
2. **Highest skill of any agent in the program**: beat the current best
   (iter11 P1, CE_Teacher §20.9) and the production 30M on the
   deployment instrument. Expert-human comparison is not measurable
   in-program without human game data; the 30M and the current best are
   the operational bars.
3. **Highest convention adherence justifiable from terminal-only rewards
   plus search targets**, on the three instruments: defender trump lead
   (t0/t1), partner trump lead, defender called-suit lead.

Standing constraints (unchanged): terminal-only reward after the
bootstrap (no shaping is the research question); observations contain
nothing a human at the table could not see or remember — no precomputed
features, no history replay, no strategic hints; principled, general
mechanisms over hand-selected node classes; aux-head hints only for
attributes the trunk demonstrably lacks; every phase pre-registered and
certified before the next starts.

The recall constraint is the new element. Play history is already
carried only by the GRU memory[^drqn]; the picker's blind and bury were
re-injected into every observation, a documented violation
(Blind_Bury_Ablation §1). This program closes it structurally.

---

## 2. Evidence base: what each design choice rests on

| Choice | Evidence (notebook §) |
|---|---|
| Keep a shaped self-play bootstrap | Every from-scratch run enters the all-PASS/leaster attractor in 3–4k episodes; architectures differ only in escape time; watchdog fixes it (Arch_Ablation §4.5). No terminal-only from-scratch run has ever been attempted. The 400k seed's conventions were terminal-optimal (+0.24 to +0.36 score, Redesign §7.2), so shaping acted as a curriculum, not a bias, and the league re-derived pick rate (50% → 34%) under the terminal objective. |
| Oracle critic + supervised oracle pretraining | Oracle GAE baseline removes hidden-card variance (Redesign §2.2, §7.1); pretraining (held-out EV 0.508) made ev_oracle 0.52–0.55 from update 1 and removed the burn-in window that killed conventions in every earlier league arm (§7.3). |
| Seat rotation, γ = 1, 16k updates + grad-accum, λ = 0.95 | The validated retention config (§7.1, §7.5, §7.9): the only league lineage that held conventions AND climbed (+0.078/+0.101 h2h gens 1–2). |
| PFSP kept, exploiters dropped | Exploiter gates failed 8/8 (edges −0.028…−0.184; §7.10–§7.19), pressure inert historically (League_Run_Review F2). PFSP weights measured near-uniform (§2.2) — kept because it is validated and cheap, not because it is load-bearing. |
| Entropy: inner controller, at most ONE target step | Play is the only binding head (pick H_norm ≈ 0.05 flat; §8.4). Step 1 re-ignited h2h +0.107 (gen 5); step 2 gave null (gens 7–8). Transition checkpoints carry the deepest wrong-side lock-in (E7). |
| Marginal-value handoff to search | Per-gen h2h: +0.078, +0.101, −0.005, +0.001, +0.107, +0.018, +0.019, −0.005 (Redesign generations.csv). Search yields +0.026 per ~2.5-day iteration (CE_Teacher §20.9). E8: the defender-lead edge is ~zero under the policy's own continuations — PG cannot climb it at any SNR; E7: more PG deepens the wrong-side prior. |
| Search-Q policy iteration as the final operator | Committee ceiling +0.180 ± 0.029 (Search_Teacher §13.3); concurrent CE+PG falsified twice (CE_Teacher §14–§16); phase-pure P1 recipe +0.0258 ± 0.0073 vs seed, 5f +0.039 ± 0.014 vs the 30M (§20.9). |
| Bilinear play pointer | Additive pointer realized ~20% of lead-row targets; the bilinear term took the advantage twin from 13% to 55% capture (§20.9). |
| Drop blind/bury tokens | Masking costs the 8M seed −1.90/picker hand and the 30M −0.30 with zero fine-tuning (Blind_Bury §3): the policies consume the re-injection; a from-scratch run is the only clean version of the claim (§4.3). |

---

## 3. Architecture: `perceiver-recall`

### 3.1 Specification

Derived from `perceiver-shared-v2-bp` (Architecture_Ablation §5.11,
CE_Teacher §20.9). Four differences from that spec, everything else
identical (d_card 16 informed init[^informed], d_token 64, 4 reasoning
layers × 4 heads, d_model 256, 16-query/4-head LayerNorm'd shared
readout, aux critic, oracle critic).

1. **Token layout 19 → 15**: `[context, memory, hand×8, trick×5]`,
   four card-type ids. The base encoder gains an `observe_blind_bury`
   switch; legacy encoders keep the 19-token path bit-identically
   (golden gate), so every existing checkpoint remains loadable and the
   production 30M is unaffected. The oracle encoder keeps its 51-token
   full-information layout (blind/bury true cards for all seats + 32
   opponent-hand tokens) — CTDE is exempt by construction[^ctde].
2. **Memory driver = post-reasoning MEMORY token.** The GRUCell input
   is the transformer's own "what to remember" slot (the `perceiver`
   rung's design, encoders.PerceiverEncoder) rather than the context
   token. Zero parameter change. Rationale: under the recall constraint
   the memory token is the only path by which the blind (seen once via
   `hand_ids` during the bury phase) and the bury (the picker's own
   action, inferable as the hand difference) reach later decisions; the
   architecture should make the network learn to write what it must
   recall, and leave the context token as the current-situation summary.
   Caveat on record: this change was never measured in isolation (the
   `perceiver-ctxmem` decomposition arm was registered, never launched;
   Arch_Ablation §5.6); the operator chose it as a program requirement,
   not a learning optimization (§0).
3. **Bilinear state × card term in the play pointer** (actors
   `bilinear_pointer=True`; zero-init U, orthogonal V)[^pointer].
4. **Aux critic unchanged**; note that the seen-trump aux head becomes
   a genuine recall target for the picker (it must remember the blind
   to answer), which is the kind of aux hint the program allows.

Parameter count ≈ v2-bp minus the simple-bag MLP (~1k). Throughput:
attention cost scales with token count squared (15² / 19² ≈ 0.62), but
the encoder is a minority of wall time; expect ~10–20% on the encoder
forward, less end to end.

### 3.2 Observation contract (two interfaces; amended 2026-09-02)

The contract lives in `sheepshead/agent/observation.py`.
`Player.get_state_dict` is the observation: `RECALL_KEYS` (header
flags, called card, seats/roles, hand, trick on the table) and nothing
else — what a human at the table sees or is entitled to remember. The
picker's blind and bury are NOT in it. They live behind a second
interface, `Player.get_picker_memory` (`LEGACY_PICKER_MEMORY_KEYS`:
the picker's own blind and bury, zeros for everyone else), which
exists only so that architectures registered before this program stay
loadable and evaluable as they were trained. Every `ArchitectureSpec`
declares `legacy_picker_memory` (True for every entry registered before
this program, False for `perceiver-recall`); `PPOAgent.needs_picker_memory`
reads it, and `observation_for(player, agent)` is the one place the two
interfaces meet — it hands a legacy agent the merged dict and everyone
else the clean observation. The recall encoder is pinned by test to
exactly `RECALL_KEYS` with the 15-token layout, and every legacy encoder
(the token family and the one-hot baseline) raises when handed a dict
without the memory keys, so a call site that bypasses the helper fails
loudly rather than running a legacy model with its memory zeroed
(−0.30/picker hand for the 30M, Blind_Bury §3).

Why the second interface exists at all (the decision recorded in §0 was
to remove the keys outright; amended at build time, then restated in
this form 2026-09-02):

- Every evaluation anchor and the production 30M are legacy
  architectures. The review gates (§5.3) compare against PANEL-A and
  the 30M as they were measured; running them masked would shift the
  panel by ~0.05 and turn "beats the 30M" into "beats a handicapped
  30M".
- The h2h instruments seat a recall agent and a legacy agent at the
  same table, so both views must be producible for one game state;
  `observation_for` differentiates per agent at the table.

The oracle critic's view, `Player.get_oracle_state_dict`, is a third,
privileged interface: it carries the TRUE blind and bury for every seat
(the definition of the full-information view, built in rather than
borrowed from the picker-memory interface) plus the opponents' hands.
Only the centralized critic ever sees it.

Consumers audited 2026-09-02: `app/server/runtime/views.py` (the only
UI reader) reads the picker's blind/bury from the game object — a table
fact, not an agent observation; the raw observation dict never reaches
the web client. Every path that feeds an agent — the trainer's streams,
search (`ismcts.py`), the evaluation instruments, the app's inference
loop and analysis service, the golden capture — observes through
`observation_for`, so the deployed recall agent consumes `RECALL_KEYS`
by construction.

### 3.3 Registry and gates

New `ArchitectureSpec` entry; `capture_arch_goldens` recapture with the
environment stamp + skip guard (Maintainability §CI); compiled-encoder
and MPS worker paths checked against the 15-token layout; a test
asserting the recall encoder's declared key set excludes the two ids.

### 3.4 What is deliberately NOT gated

No strength comparison against easier-job architectures during the
bootstrap. The recall constraint makes early learning harder by design
(less information, a routing problem to solve); judging it by early
curves would penalize the requirement. Strength gates start where the
comparison is like-for-like (§5.3).

---

## 4. The pipeline

Five phases, one orchestrator, one config dataclass that doubles as the
pre-registration artifact. Everything below is phase-pure: exactly one
policy-improvement operator acts on the network at a time[^phasepure].

### 4.1 Phase 0 — shaped self-play bootstrap (400k hero episodes)

- Reward: the historical shaping (intermediate trick rewards, leaster
  bonus, running picker baseline)[^shaping]; limited critic; aux heads
  on; leaster watchdog ON (selective pick-entropy kick, inert when
  healthy; Arch_Ablation §4.5)[^watchdog].
- Runs in the **unified trainer** (§6) with an empty population: pure
  self tables, per-seat hero-stream storage, 8 workers, seat rotation.
  Bootstrap presets differ from the league's: 4k-episode updates,
  standard minibatching (4 epochs × 256), the self-play entropy schedule
  (config.SelfPlayHyperparams). Rationale: the from-scratch escape needs
  update density, not the league's low-temperature regime.
- Why kept (vs terminal-only from scratch): §2 row 1. A terminal-only
  from-scratch run remains a future ablation, not this run's risk.
- Health gates only: escape from the all-PASS attractor by ≲30k
  episodes; no divergence; no watchdog halt streak. NO strength bar
  (§3.4).
- Cost: legacy single-process trainer did 400k in 20.8 h (~5.3 eps/s,
  v2); the unified 8-worker path is expected at 6–8 h. Fallback if the
  smoke shows anything odd: `train_selfplay_ppo` as-is for this run.

### 4.2 Phase 1 — oracle supervised pretraining

Verbatim Redesign §7.4: 40k frozen-seed self-play episodes, γ = 1
terminal returns, official `OracleValueNetwork` with the two validated
aux heads (team membership 5-way multi-label, team points with bury;
coefficients 0.1/0.2), early stop on validation value-MSE. Moved out of
`analysis/diagnostics/oracle_moe_offline.py` into a training module
without the MoE arms. ~3 h.

Literature: asymmetric actor-critic / centralized critics[^ctde]; the
history-state value U(h, s) (Baisero & Amato) is why the oracle is
recurrent over the same event stream the actor sees; critic
pre-fitting before policy optimization is standard practice in
warm-started RL (e.g. value-head initialization before PPO in
Ziegler et al.[^ziegler]).

### 4.3 Phase 2 — terminal-only league policy gradient

The retention configuration is the ONLY mode (Redesign §7.9 defaults):
PPO[^ppo] with GAE λ = 0.95, γ = 1.0[^gae]; oracle GAE baseline; aux
heads on-line; seat-rotated collection (each deal 5×, hero in every
seat: role-exposure equalization + paired deals); update interval 16,384
episodes, 128-episode minibatches with gradient accumulation (one
full-buffer optimizer step per epoch, bit-equivalent gradients);
unanchored from gen 1; leaster watchdog; snapshots every 50k; 1M-episode
generations. Terminal reward = final_score / 12 at the last action.

Population (`League.sample_table`, per-seat mixture)[^pfsp]: 0.15
current-self, otherwise PFSP over past-main snapshots with a 0.05 HOF
floor. HOF promotion = every boundary snapshot (quota 6 by rating).
Exploiter role, seat-heat EMA, retirement clocks: DELETED (§2).

Entropy (§5.2): per-head target-entropy controller (SAC automatic
temperature, discrete/normalized form)[^sac]; gen 1 runs the fixed start
coefficients; at the gen-1 boundary each head adopts its measured H_norm
as its hold target (bumpless transfer[^bumpless]); at most one play-target
step (retain 0.75 toward the 0.28 floor) over the whole phase, fired by
the stop rule. Learning rate constant 1.5e-4 (the 20M-episode clock never
annealed within a real run; PPO under Adam tolerates flat LR[^andry]).

Removed outright: the online CE teacher and its guards/in-trainer cert
(falsified, CE_Teacher §14–§16), the gen-1 bidding anchor (validated 0),
in-trainer anchor eval, GNS logging, schedule horizon, all exploiter
flags, the outer entropy ladder and its sidecar absorption.

Boundary battery (per generation): duplicate-bridge h2h vs the previous
generation (2,000 deals/mode, all-seat CRN, deal-clustered
bootstrap)[^duplicate]; PANEL-A absolute endpoint (the 30M-lineage
yardstick, kept because it is available and absolute); the three
convention probes at n=1000 × 4 seeds; greedy health (pick/leaster/alone/
spread) and, from the same battery, the deterministic aux heads' reads
(seen-trump accuracy / recall / false-seen by trick, unseen-higher,
known points, secret partner — the handoff's readiness precondition,
§5.4, added 09-18); the E7 frozen-node logit ladder (scale-free
low-mass share, C2 mass); role-conditioned h2h (picker hands) as the
recall diagnostic. The four deterministic aux heads train at 2.5× their
base loss coefficients (§5.4). Cost ≈ 47 h/gen training + ~1 h evals
(retention run measured 44–63 h; `202609_recall_rc` gen 1: 5.0 eps/s ≈
55 h).

### 4.4 Phase 3 — search-Q regularized policy iteration, to convergence

The CE_Teacher §20 standing recipe (§20.9 P1 + §20.10 + §20.11), one
iteration = corpus → fit → target → distill → cert → bidding phase →
cert:

1. **Corpus** from frozen θ_k (amended 09-12, CE_Teacher §20.14): 8,000
   COMMITTEE-ACTING games (`--committee-act-frac 1.0`; the corpus samples
   the search-preferred lines, +0.0055 play vs student acting, add. 16),
   committee R = 3, `--iters 256` with the trick-indexed lead schedule
   `--iters-schedule t0-lead:1024,t1-lead:512` (256 does not resolve the
   lead conventions; t0 leads want 1024, t1 leads 512; leads are ~5.5% of
   searched nodes per trick so the schedule costs ~+22% over all-256, add.
   29/29b), d_rollout 1 with oracle leaves, every play node searched
   (p_base 1.0, boost_lead 1.0, boost_cs 1.5, p_min 0.05), schema 2,
   oracle states stored. ~55 h on the M1 Max (0.04-0.05 games/s).
2. **Stage 1 (evaluation)** — heteroscedastic weighted least squares of
   the centered committee Q onto θ_k's frozen features through a twin of
   the play pointer, with the centered log-prior as a covariate; held-out
   weighted MSE selects capacity. **Stage 1b** — Fay–Herriot blend:
   γ_n = σ²_u/(σ²_u + σ²_n), per-class residual variance, posterior
   variance v_n = γ_n σ²_n[^fh][^efron].
3. **Stage 2 (improvement)** — t_n(a) ∝ p_θk(a|s) · exp(clip(Â_n(a)/
   (κ √v_n), ±8)), κ = 1: the KL-regularized mirror-descent step with
   the posterior SE as temperature[^vieillard][^awr]; precision weights
   ω_n ∝ 1/v_n, mean-normalized, cap 5 (never binds; §20.11).
4. **Stage 3 (projection)** — PG off. Weighted CE on searched play rows;
   retention KL(p_θk ‖ π_θ) on bidding-head and leaster-play rows[^lwf]
   at λ_ret = 10 (halves the bidding drift; does not pin it); value/aux/
   oracle regression on all rows. Schedule (amended 09-12): SIX trunk
   epochs at 3e-5 (everything trains), then bilinear-only head epochs at
   1e-3 with the encoder frozen (4). The trunk dose is the change that
   made iteration 2 compound: on the same 140k-row targets, 1 epoch @1e-4
   read +0.0002, 3 @3e-5 +0.0024, 6 @3e-5 +0.0043 (play-only, vs θ_k).
   The candidate is the LAST epoch of the schedule (amended 09-15): the
   held-out target KL is logged per epoch as a fidelity check but selects
   nothing — it does not track EV, and the plateau rule it drove ended
   iteration 3's head phase after one epoch and discarded +0.0030 ±
   0.0023 of certified play (CE_Teacher add. 31)[^lpft]. The advantage
   fit runs 200 epochs with patience 25 (the recipe's values; the module
   defaulted to 120 / 4 until 09-15).
5. **Cert** (amended 09-12) — 4 × n=1000 convention/health probes +
   duplicate h2h vs θ_k at 8,000 deals/mode (sharded, ~17 min) + the
   head-routed reads: PLAY-ONLY (bidding from θ_k, play from the
   candidate) is the compounding statistic the stop rule reads — it
   strips the ±0.005 bidding variance the trunk epochs add — and
   BIDDING-ONLY is the drift guard (flag below −0.003 at 2 SE). Adoption
   gate = NON-INFERIORITY on the full checkpoint (edge + 2·SE ≥ 0) plus
   the convention guards (partner ≥ 96.5, t0 trump ≤ 1.0, spread ≥ 3.6):
   the per-iteration gain (~+0.003) sits inside the 8000-deal SE, so a
   positivity gate would reject every real step; compounding is judged
   at the program level by the slope of the play-only reads. WiSE-FT
   walk-back stays the operator's option[^wise].
6. **Bidding phase** — PG under terminal reward with the encoder, actor
   adapter, play pointer (which also carries bury) and play-under scalar
   FROZEN (`PPOAgent.set_trainable_heads("bidding")`); the pick, partner
   and call heads, limited critic and oracle train against the league
   population; 200k episodes; certified vs the candidate (non-inferiority
   on the same battery) and adopted if it passes. Deployment is the
   SINGLE network — head routing was a hedge against the λ_ret-1 drift
   and is not needed under the pinned recipe (bidding read +0.0014 /
   −0.0005 on the optimised arms; iteration 1's trunk epoch improved
   bidding by +0.0036).
7. Iterate: θ_{k+1} generates the next corpus — ONE generation of corpus
   per iteration (no replay window: corpus D ∪ the re-anchored iteration-1
   corpus read −0.0052 ± 0.0028 on the play route vs corpus D alone,
   CE_Teacher add. 30; the search advantages are continuation-dependent
   and do not transfer across generations). Stop when the play-only gain
   is below 2 SE for two consecutive iterations, or at the cap. Measured
   exchange rate on the v2 lineage: ≈ +0.003–0.004 play per ~55 h
   iteration against a deploy-time search ceiling of +0.166 over the same
   policy.

Code consolidation: the §17 partition trainer (`train_distill.py`:
override/endorsed/retention partition, AWR ω, KD temperature) merges
into the policy-iteration module with the standing recipe as defaults;
capacity/variance-mode/weight-mode sweep flags removed; schema-1 support
and `recover_search_q.py` deleted. `--committee-act-frac` was removed on
09-02 (decided 0) and RESTORED on 09-12 at 1.0 after the acting-mode
replicate (CE_Teacher add. 16); `--iters-schedule` added (add. 29b).

### 4.5 Phase 4 — final certification and release

Duplicate h2h vs the v2-lineage release (θ₃ after its bidding phase),
iter11 P1 and the 30M at 8,000 deals/mode each (amended 09-16: the
final bars — 30M positive at 2 SE, iter11 excluding −0.02 — are
underpowered at 2,000); the convention battery; ONE exploitability audit
(a best-response run + gate, `exploiter.py` retained as an analysis
tool). Golden capture for the release checkpoint
(`analysis/capture_arch_goldens`) and the export to the app's model path
are MANUAL steps after the program finishes (amended 09-16).

---

## 5. Stop rule and gates

### 5.1 PG → search handoff: marginal value, not plateau

Continue PG while the duplicate h2h gain over the previous generation is
≥ +0.02 with the bootstrap CI lower bound > 0. The bar is search's
measured yield at equal compute (+0.026 per ~2.5-day iteration vs ~2
days per generation). Applied to the retention run it passes gens 1, 2
and 5 and fails 3, 4, 6, 7, 8 — exactly the generations that paid.

- First failure → fire the single play-target entropy step (the gen-4→5
  precedent: +0.107 after the step).
- Second failure → hand off, after a fresh-deal confirmation (seed
  20260706) contradicts nothing. Minimum 3 generations, cap 8.
- Handoff checkpoint = the last generation wholly at a settled entropy
  target, never a transition generation: E7's two deepest wrong-side
  reads were the two transition checkpoints (low-mass share 0.226 at 5M,
  0.108 at 7.9M vs 0.371 settled at 7M).
- Power note: at 2,000 deals/mode the h2h SE ≈ 0.012, so a true +0.02
  gain passes the rule roughly half the time; the confirmation run is
  what keeps a noise miss from ending PG a generation early.

### 5.2 Conventions are guards, not triggers

Per generation: B2 hard bounds (partner trump lead ≥ 0.5 AND defender
t0 trump lead ≤ 0.10) halt for review; the E7 ladder, C2 pooled-greedy,
and the three headline probes are recorded. A convention-based early
stop would have discarded gens 2 and 5; search reverses the lock-in
either way. Pre-registered: the low-mass share at handoff predicts the
search phase's install cost (more locked-in prior → lower per-iteration
realization).

### 5.3 Review gates (operator review, not automatic kill)

| gate | instrument | v2-lineage reference | bar |
|---|---|---|---|
| gen-2 boundary | PANEL-A absolute endpoint | +0.132 | ≥ +0.06 (one MDE below) |
| (recorded) | PANEL-B absolute endpoint (tentative, 09-16): the 30M, the v2 release, iter11 P1, the v2 8M seed — a strong-skill, cross-ecology field where PANEL-A's weak anchors compress candidate differences; per generation and at the final | — | none; a bar, if any, is defined from this run |
| handoff | duplicate h2h vs the v2 8M seed | parity | CI lower bound > −0.02 |
| end of phase 3 | duplicate h2h vs iter11 P1; vs 30M | +0.026 vs seed; +0.039 vs 30M | CI excludes −0.02 vs P1; > 0 at 2 SE vs 30M |

A gate failure is diagnosed before any decision: the role-conditioned
h2h (picker hands) separates a recall-routing deficit from a general
one; the `perceiver-recall-ctxmem` twin becomes the first diagnostic
arm if the gen-2 gate fails.

### 5.4 Aux-head readiness: the handoff's precondition (added 09-18)

The league is the only phase that can BUILD the trunk memory the aux
heads read: policy iteration trains the six aux losses too, but on
8,000 games at 3e-5 (maintenance), and the bidding phase freezes the
encoder. Four of the six heads are exact functions of what the seat has
observed — the seen-trump mask (own history), unseen-trump-higher (that
mask + hand), known points per seat (trick history + own bury), secret
partner (the self label, hand + call) — so 100% is their true ceiling
and "essentially never wrong" is a fair bar. Win and return predict
outcomes with irreducible uncertainty and are never gated.

Rule (`stop_rules.aux_readiness`, `LeagueConfig.aux_bars`): a handoff
the marginal-value rule would make is DEFERRED (the generation reads
`continue`, reason "handoff deferred, aux heads not ready") until the
boundary battery's means (4 × 1,000 greedy games, every seat's play
nodes) clear every bar; at the generation cap without readiness the
program exits NEEDS REVIEW. The entropy step and plain continues are
untouched. Bars, from the v2 retention lineage's converged checkpoint
(league 7.7M) re-probed under the 09-16 probe definition (values in
`runs/202609_recall_rc/recall_compare/`): seen-trump accuracy ≥ 99.5%,
false-seen ≤ 0.5% overall and at each of tricks 1–5, recall ≥ 99% at
each of tricks 1–5; unseen-higher accuracy ≥ 99%; known points MAE
≤ 1.0 point; secret-partner accuracy ≥ 99.5%. Trick 0 (the picker's
bury alone) is never read. The reads are recorded per generation in
`generations.csv` (`seen_trump_acc`, `seen_trump_false_seen`,
`aux_unseen_higher_acc`, `aux_points_mae`, `aux_points_exact`,
`aux_secret_acc`, `aux_ready`) and the same probe runs at every policy
iteration cert, so a trunk-epoch regression of the memory is visible.

Measured basis (400 games per checkpoint, every seat's play nodes):

| league episodes | seen acc / false-seen | secret | unseen-higher | points exact / MAE |
|---|---|---|---|---|
| v2 0.7M | 92.8 / 12.2 | 99.5 | 99.9 | 10.4 / 5.31 |
| v2 2.7M | 95.8 / 6.1 | 99.9 | 99.9 | 7.6 / 8.35 |
| v2 4.7M | 99.4 / 0.9 | 100.0 | 100.0 | 5.5 / 9.40 |
| v2 7.7M | 99.8 / 0.2 | 100.0 | 100.0 | 45.6 / 0.88 |
| recall 0.7M | 90.2 / 16.7 | 99.9 | 99.8 | 42.2 / 1.02 |

Secret partner and unseen-higher are at ceiling from the first
checkpoint in both lineages; seen-trump is the memory head and the
binding bar. The KNOWN-POINTS head is the exception to "never wrong":
it is a regression under a smooth-L1 loss, so it is never integer-exact
(45.6% at v2's best), and in the v2 lineage it wandered to a 9-point
error while seen-trump converged, then recovered to 0.88 at the end.
Its bar is therefore the demonstrated converged MAE, which would have
held v2's handoff at 4.7M — the gate's protective function — rather
than an exactness no lineage has reached. OPEN: a "never wrong" points
head is a classification head over per-seat totals (0–120, or the
score-relevant thresholds); not changed mid-run.

Coefficients (same date): the four deterministic heads train at 2.5×
their base loss coefficients (`ProgramConfig.aux_det_scale`, the
trainers' `--aux-det-scale`; seen-trump 0.2 → 0.5, points 0.2 → 0.5,
secret 0.1 → 0.25, unseen-higher 0.1 → 0.25; win 0.05 and return 0.1
unchanged), in every stage that trains them. Both losses are
residual-driven, so a converged head contributes nothing at any
coefficient; the multiplier only shortens the transient in which the
trunk is pushed to carry the memory. Kept modest because the trunk is
shared and the play head's SNR at rare lead nodes is the league's
binding constraint. `202609_recall_rc` ran its bootstrap and league
gen 1 at 1.0, so gens 1 and 2 are not a single-condition series
against v2. Rejected: a standalone trunk memory pretrain (moves the
trunk under frozen policy heads → policy drift); learned uncertainty
weighting (Kendall & Gal) as more machinery than the static bump plus
the gate need — revisit if the bump proves insufficient. The
unweighted aux losses are now logged per update
(`training_progress.csv` `aux_loss_*`), so the effect is visible.

---

## 6. Code (built 2026-09-02 on branch `training-program-redesign`)

Modules in `sheepshead/training/`:

| module | role | provenance |
|---|---|---|
| `train_ppo.py` | the one PPO trainer: `--phase {bootstrap,league,bidding}` with phase presets (`PhaseSpec`), `--until` on an absolute episode clock, the target-entropy controller from generation 2, HOF promotion of every boundary snapshot | `train_league_ppo` stripped of teacher / exploiter / anchor / GNS / clock schedules; `train_selfplay_ppo` retired (the bootstrap is the same loop on an empty population with shaped rewards) |
| `config.py` | `BootstrapHyperparams`, `LeagueHyperparams` (constant LR, fixed gen-1 coefficients), `CommitteeConfig`, `LeagueConfig` | `PFSPHyperparams` / `SelfPlayHyperparams` / `SearchConfig` replaced |
| `league.py` | roster + per-seat PFSP/self sampling + HOF | exploiter role, seat heat, retirement clocks, legacy migration removed |
| `league_streams.py`, `league_worker.py` | episode streams and the worker pool, `reward_mode` threaded through | CE emission removed |
| `pfsp_runtime.py` | the game primitive, committee summary/tilt (corpus generation) | online CE emission removed |
| `entropy_controller.py`, `leaster_watchdog.py` | unchanged | — |
| `pretrain_oracle.py` | phase 1 | from `oracle_moe_offline.py` (MoE arms dropped) |
| `distill_corpus.py` | phase 3 corpus, schema 2 only | committee acting, alone-only calibration removed |
| `search_advantage.py` | Stage 1/1b/2 math (pointer/adapter rungs) | trunk rung removed |
| `policy_iteration.py` | fit / target / distill / cert with the standing recipe as defaults | `train_policy_iteration` + `train_distill` merged; sweep flags, §17 partition machinery, `recover_search_q` removed |
| `stop_rules.py` | the marginal-value handoff rule, settled-checkpoint rule, iteration stop | replaces `league_stopping` |
| `program_config.py` | the config tree = the pre-registration artifact (`smoke_config()` for the minutes-long check) | new |
| `run_training_program.py` | the resumable orchestrator over the five phases, review gates, reports | replaces `run_extended_league` + `league_reports` |

Also: `agent/observation.py` (the contract), `analysis/exploitability_audit.py`
(the post-hoc audit, from `exploiter.py`), `analysis/league_progress_eval.py`
(`h2h_duplicate` now returns the leaster-hand paired score).

PPOAgent changes: `set_trainable_heads("bidding")`, `observation_keys`; the
teacher CE passes, GNS diagnostic and bidding anchor are gone.

09-12 merge into master: `analysis/league_progress_eval.h2h_duplicate` is
the SHARDED evaluator (spawn workers, bit-identical to serial; per-deal
scores stored for paired reads, leaster read carries `hands` and `n`);
`analysis/head_routed_h2h.routed_h2h` (sharded) provides the cert's
play-only / bidding-only routes; `policy_iteration.cert` writes `routed`
and `compounding`; `program_config.ProgramConfig.start_phase` +
`PolicyIterationConfig.{theta_0, league_dir, bidding_first}` enter phase 3
from an external lineage (`run_bidding_phase` extracted).

Tests: `test_recall_architecture`, `test_stop_rules`, `test_program`,
rewritten `test_league_smoke` / `test_trainer_output_contracts` /
`test_distill_pipeline` / `test_policy_iteration`; the arch goldens were
recaptured with the new fixture and every legacy fixture verified
byte-identical against a pre-change capture. Full suite green; the program
smoke (`--smoke`) exercises every phase end to end.

## 7. Pre-registration

### 7.0 Gate before launch: iteration-2 compounding on the current lineage

Running since 2026-09-02 (CE_Teacher §20.12). COMPOUNDS ⇒ phase 3 as
written. STALLS (h2h vs iter11 inside ±0.01) ⇒ phase 3 budget = one
iteration + bidding phase, and the §4.4 stop rule is moot. REGRESSES ⇒
student-acting corpus suspect; committee-acting rerun before launch.

**RESULT (2026-09-12; CE_Teacher §20.13 addenda 2–30, §20.14).** The
recipe as written on 09-02 STALLED: every theta_1 arm read −0.005..+0.001
on the full checkpoint, and the loss was a bidding drift the frozen head
phase could not repair (add. 12). With the bidding fix (λ_ret 10) the
play was at parity at 35k, 70k, 91k and 140k rows and at 256 or 1024
iterations (add. 20–24). The lever was trunk OPTIMISATION at scale: six
trunk epochs at 3e-5 on 140k rows read +0.0043 ± 0.0028 play-only (pooled
two seeds +0.0038 ± 0.0019, 2σ; add. 25–27). Verdict: COMPOUNDS, small
(≈ +0.003/iteration ≈ 1/4 of iteration 1's +0.011). Phase 3 proceeds with
the §4.4 recipe as amended; the exchange rate (~55 h per +0.003) is the
operator's call per iteration. The v2-lineage validation run
(`rc_validate_v2`: bidding phase on θ_2 = D8k_t6, then iteration 3 under
the merged scripts) is the last check before the fresh perceiver-recall
launch.

**VALIDATION RESULT (2026-09-15; CE_Teacher add. 31, §21).** The pipeline
ran end to end from `start_phase: policy_iteration`: bidding phase on θ₂
(7.96 h) ADOPTED at +0.0054 ± 0.0036; corpus of 8,000 committee-acted
games under the lead schedule (60.6 h; 133k searched / 64.5k override
rows); fit + target + distill 2.1 h; cert 1.2 h. Iteration 3 COMPOUNDS:
play-only +0.0066 ± 0.0024 over θ₂′ on the last epoch (+0.0036 ± 0.0023
on the KL-best epoch 5), full single network +0.0058 ± 0.0026, bidding
route −0.0010 ± 0.0008 (inside the guard), every bar passed. Three
iterations of the lineage now read +0.0107 / +0.0038 / +0.0066 on the
play route, each vs its own θ_k. The lead schedule installed called suit:
45.6 → 56.3 in one corpus. Two pipeline defects surfaced and were fixed
(routed-chimera observation; fit defaults) and the candidate rule was
corrected (last epoch). The run then completed (bidding phase on θ₃
−0.0015 ± 0.0025, adopted; stop at the cap; final phase; PROGRAM
FINISHED 09-16 08:10). The RELEASE (`runs/rc_validate_v2/final/
release.pt`) at 8,000 deals/mode: **+0.0505 ± 0.0062 vs the production
30M** (jd +0.078, called +0.023), **+0.0135 ± 0.0046 vs iter11 P1**,
+0.0069 ± 0.0043 vs θ₂; conventions called-suit 60.3, partner 98.6, t0
trump 0.2. Both objective-2 bars are cleared by the old lineage's
release, which therefore sets the bar for the fresh run and joins the
final references and PANEL-B. The perceiver-recall launch is unblocked.

### 7.1 Per-phase expectations

- **Bootstrap.** Escape ≤ 30k; scripted-probe and PANEL-A curves
  recorded, no bar. Seen-trump memory at every seat's play nodes
  (greedy probe columns `seen_trump_acc` / `_recall` / `_recall_t0..t5`
  / `_false_seen` / `_false_seen_t0..t5`, redefined 09-16 evening — the
  first definition scored
  the picker alone and only its blind/bury trumps, which mistook the
  recall problem: play history is never re-shown (§3.2), so every trump
  played in an earlier trick is a memory item for every seat, and
  perceiver-recall adds the picker's bury/discarded blind to that set).
  `recall` = % of the must-remember trumps (seen, but neither in hand
  nor on the table) the aux head still reports seen; per trick, trick 0
  is the picker's bury/blind alone and later tricks add the played
  cards, so forgetting reads as decay across the row; `false_seen`
  (overall and per trick) guards against a head that says "seen" for
  everything — rising recall at rising false-seen is bias, rising recall
  at flat false-seen is memory. Expected (recalibrated 09-18 against
  the v2 retention lineage, re-probed under this definition —
  `runs/202609_recall_rc/recall_compare/`): the bootstrap ends in the
  partial-memory regime (acc ~89%, false-seen ~18%, rising with the
  trick index); reliable recall is a LEAGUE-scale outcome, not a
  bootstrap one — v2 read acc 93/94/96/99.3/99.8% and false-seen
  12/11/6/1.0/0.2% at league episodes 0.7/1.7/2.7/4.7/7.7M, i.e. the
  clean-up came between gens 3 and 5, in false-seen (recall was ~96%
  from the start, by bias). At league 700k perceiver-recall reads acc
  90.3 / false-seen 16.7 (t1–5 12/23/36/46/46) vs v2's 93.2 / 11.7
  (9/15/24/33/33): a few points behind on the harder task, same regime.
  Trick 0 (the picker's bury alone, ~100–200 cards per probe) is
  erratic for BOTH lineages (v2 0–67% with the bury in its
  observation), so read tricks 1–5. INFORMATIONAL ONLY — it is
  recorded every probe interval and never gates or changes anything.
  The bootstrap of `202609_recall_rc` ran under the first definition;
  its checkpoints were re-probed under the second
  (`analysis/reprobe_checkpoints.py` → `checkpoints/greedy_health_recall.csv`),
  and every later phase records the second.
- **League.** Gen-1 h2h ≥ +0.05 and gen-2 ≥ +0.05; B2 held from gen 1;
  panel ≥ +0.20 by gen 6 (v2: +0.206); C2 in the 38–52% band; handoff
  at gen 4–6. Aux readiness (§5.4, added 09-18): v2 at 1.0× cleared the
  seen-trump bars between league 2.7M and 4.7M (false-seen 6.2 → 1.0%),
  i.e. gens 3–5; at 2.5× on the harder recall task the expectation is
  readiness by gen 4, so the gate is expected NOT to defer the handoff.
  A deferral is informative (the handoff was about to fire on an
  unready trunk); readiness still missing at the cap is the failure
  reading (§7.2).
- **Policy iteration** (amended 09-12). Iteration 1 play-only vs θ_0
  ≥ +0.010; later iterations play-only ≥ 0 (non-inferior) with the pooled
  slope over iterations positive at 2σ; bidding route within ±0.003;
  called-suit ≥ 50 pooled by the end of phase 3 under the lead schedule
  (measured on the v2 lineage: 45.6 → 56.3 after one scheduled corpus,
  add. 31; the operator's target is ≥ 60); leaster paired score within
  noise of θ_k at every cert.
- **Final.** Bars per §5.3; conventions as EXPECTATIONS: defender t0
  trump ≤ 1%, partner ≥ 96%, called-suit pooled ≥ 50% (terminal-only
  optimum estimated 60–70%, E6; the 30M's 90% is shaped over-adherence);
  exploiter audit gate fails.

### 7.2 Failure readings

- Bootstrap fails to escape ⇒ watchdog gain / kick timing, not the
  architecture (all archs escaped with it).
- Gen-2 gate fails with picker-hand deficit ⇒ recall routing; run the
  ctxmem twin; consider a seen-trump/bury-points aux boost.
- Handoff h2h vs 8M seed below bar with conventions intact ⇒ the recall
  tax is a strength tax; proceed to search and re-read at the final gate.
- Iteration stalls at the first iteration ⇒ the §20.8 projection
  bottleneck reproduces; head-lr sweep (§20.11) before anything else.

### 7.3 Budget

| phase | estimate |
|---|---|
| bootstrap 400k | 6–8 h (unified) / 21 h (legacy) |
| oracle pretrain | ~3 h |
| league, 4–8 gens × ~48 h | 8–16 days |
| policy iteration, 1–5 iterations × ~3 days (measured: 60.6 h corpus + 2.1 h fit/distill + 1.2 h cert + 8.0 h bidding phase + 0.4 h its cert ≈ 72 h) | 3–15 days |
| final cert + audit | ~1 day |
| total | 2.5–5 weeks |

---

## 8. Novelty assessment (for the write-up)

Honest positioning, checked against the literature the program has
cited plus three targeted searches on 2026-09-02 (no systematic review).

**Genuinely novel (no precedent found):**

1. **Search-Q policy evaluation as small-area estimation over the
   policy's own frozen features** (Stage 1/1b). Treating each searched
   node as a Fay–Herriot "area" — a known-variance measurement (committee
   replicate SE) with a covariate regression (a twin of the play pointer
   on θ_k's frozen readout + hand tokens, log-prior covariate) — and
   using the posterior variance as the tilt temperature. Empirical-Bayes
   shrinkage of MCTS values exists in single-search settings (Tesauro et
   al. 2010[^tesauro]), and heteroscedastic regression heads are standard
   (Kendall & Gal[^kg]), but pooling search labels across states through
   the student's representation to estimate a shared advantage, with the
   improvement step tempered by the pooled posterior SE, appears to be
   new. It is the program's answer to a specific measured fact: per-node
   determinized-search labels at convention nodes sit at the noise null
   however much budget is spent (§20.1, §12.8).
2. **Recall-constrained observation as an architectural specification**
   enforced at the observation dict, with the memory token as the sole
   recall channel and a from-scratch training to prove it. Human-like
   play modeling exists (Jacob et al. 2022[^jacob]; Maia), but as
   imitation of humans, not as a restriction on machine recall under
   pure self-play.

**Uncommon compositions (each component has precedent; the composition
does not, as far as found):**

3. PPO league (AlphaStar-style PFSP[^pfsp], OpenAI-Five-style self
   share[^five]) → phase-pure offline expert iteration on the SAME
   network, with an explicit compute-marginal-value handoff rule.
   Classical approximate PI with rollouts (Tesauro & Galperin[^tg];
   Lagoudakis & Parr[^lp]; CBMPI[^scherrer]) and ExIt[^exit]/AlphaZero[^az]
   start from scratch or from supervised data; mixing PG with MCTS
   values inside one loop has been studied (Soemers et al.[^soemers]) and
   we measured its failure (CE_Teacher §16.9). Handing off by comparing
   the two operators' marginal yield per unit compute is, to our
   knowledge, undocumented.
4. **Head-partitioned alternating improvement operators**: search-CE on
   the play head, PG on the bidding heads with trunk and play head
   frozen, alternating under certification. Phasic Policy Gradient[^ppg]
   alternates phases with a behavior-cloning constraint on one network;
   ours partitions by decision type because the two heads have different
   SNR structure (§16.9 addendum 7).
5. Supervised oracle pretraining + seat-rotated paired deals + terminal
   reward as the league regime (each standard; the combination was what
   made retention work, Redesign §7).

**Not novel, and should be cited as such:** the CTDE oracle critic,
pointer/two-tower heads, Perceiver-style shared readout[^perceiver]
[^settrans], GRU memory for partial observability, SAC-style entropy
control, duplicate-bridge evaluation, WiSE-FT, LP-FT-style frozen-trunk
epochs, positive-part James–Stein shrinkage, DAgger state distribution.

---

## 9. References

[^drqn]: Hausknecht & Stone, "Deep Recurrent Q-Learning for Partially Observable MDPs," arXiv:1507.06527, 2015 — recurrent memory as the sole carrier of history under partial observability.
[^informed]: Architecture_Ablation §4.1/§4.3 — informed card-embedding init +0.150 PANEL-A under the transformer (2 SE).
[^ctde]: Pinto et al., "Asymmetric Actor Critic for Image-Based Robot Learning," arXiv:1710.06542, 2017; Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (MAPPO), arXiv:2103.01955, 2021; Foerster et al., COMA, arXiv:1705.08926, 2018; Baisero & Amato, "Unbiased Asymmetric Reinforcement Learning under Partial Observability," arXiv:2105.11674, 2022 (the history-state value U(h,s)); Li et al., Suphx, arXiv:2003.13590, 2020 (oracle guiding in a hidden-information card game); Berner et al., OpenAI Five, arXiv:1912.06680, 2019 (privileged value function).
[^pointer]: Vinyals, Fortunato & Jaitly, "Pointer Networks," arXiv:1506.03134, 2015 — hand-slot scoring; the bilinear term is the two-tower product the call scorer already used (CE_Teacher §20.9).
[^phasepure]: Anthony, Tian & Barber, "Thinking Fast and Slow with Deep Learning and Tree Search," NeurIPS 2017 (ExIt); Silver et al., AlphaZero, Science 362, 2018 (arXiv:1712.01815) — search is the only policy-improvement operator, never concurrent with model-free PG; Schaul et al., "The Phenomenon of Policy Churn," arXiv:2206.00730, 2022 — the single-objective baseline of the two-operator orbit measured in CE_Teacher §16.9.
[^shaping]: Ng, Harada & Russell, "Policy Invariance under Reward Transformations," ICML 1999 — our shaping is NOT potential-based, which is exactly why the phase switch to terminal reward exists and why the final agent is certified under the terminal objective only.
[^watchdog]: Architecture_Ablation §4.5; a bang-bang controller on the pick head's entropy coefficient, hysteresis 90%/30% rolling leaster rate.
[^ziegler]: Ziegler et al., "Fine-Tuning Language Models from Human Preferences," arXiv:1909.08593, 2019 — value head initialized before policy optimization; the general practice of critic pre-fitting on a frozen policy's returns.
[^ppo]: Schulman et al., "Proximal Policy Optimization Algorithms," arXiv:1707.06347, 2017.
[^gae]: Schulman et al., "High-Dimensional Continuous Control Using Generalized Advantage Estimation," arXiv:1506.02438, 2016; λ = 0.95 policy and the node-selective contingency: Learning_System_Redesign §7.5.
[^pfsp]: Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning," Nature 575, 2019, doi:10.1038/s41586-019-1724-z — prioritized fictitious self-play; Lanctot et al., PSRO, arXiv:1711.00832, 2017 — population-based best-response framing (the exploiter role we retire to an audit).
[^five]: Berner et al., OpenAI Five, arXiv:1912.06680, 2019 — 80/20 self vs past mixture; our 0.15 self share is the validated per-seat analog.
[^sac]: Haarnoja et al., "Soft Actor-Critic Algorithms and Applications," arXiv:1812.05905, 2018 §5; Christodoulou, "Soft Actor-Critic for Discrete Action Settings," arXiv:1910.07207, 2019; Sokota et al., arXiv:2206.05825, 2023 — mixed equilibria in imperfect information, why the floor is never zero.
[^bumpless]: Åström & Wittenmark, *Adaptive Control*, 2nd ed., 1995, ch. 9 — bumpless transfer; Learning_System_Redesign §8.5–§8.6.
[^andry]: Andrychowicz et al., "What Matters in On-Policy Reinforcement Learning?", arXiv:2006.05990, 2020 — LR decay helpful but modest; Learning_System_Redesign §8.7.
[^duplicate]: Bard, Hawkin, Johanson & Szafron, "The Annual Computer Poker Competition," AI Magazine 34(2), 2013 — duplicate-match format; Burch et al., AIVAT, AAAI 2018 — paired variance reduction; instrument: analysis/rigorous_eval.py + league_progress_eval.h2h_duplicate.
[^dagger]: Ross, Gordon & Bagnell, DAgger, AISTATS 2011 (arXiv:1011.0686) — labels on the student's own state distribution from a stationary expert; committee acting is off (CE_Teacher §20.3).
[^ismcts]: Cowling, Powley & Whitehouse, "Information Set Monte Carlo Tree Search," IEEE TCIAIG 4(2), 2012; Long et al., AAAI 2010 (determinization limits); committee form: Chaslot et al., root parallelization, CG 2008 — used for noise estimation; certified budget: Search_Teacher_Design §3–§8 (E9).
[^fh]: Fay & Herriot, "Estimates of Income for Small Places," JASA 74(366), 1979 — the Stage-1b estimator: known-variance measurements, covariate regression, precision-weighted blend, residual-variance estimate.
[^efron]: Efron & Morris, "Data Analysis Using Stein's Estimator and Its Generalizations," JASA 70(350), 1975; James & Stein 1961; Baranchik 1964 (positive part) — the shrinkage root and the per-class residual-variance pooling.
[^vieillard]: Vieillard, Pietquin & Geist, "Leverage the Average," NeurIPS 2020 (arXiv:2007.06799); Vieillard et al., Munchausen RL, arXiv:2007.14430 — KL-regularized policy iteration, evaluation errors average across iterations at rate 1/k (the compounding claim); Grill et al., "Monte-Carlo Tree Search as Regularized Policy Optimization," ICML 2020 (arXiv:2007.12509); Danihelka et al., "Policy Improvement by Planning with Gumbel," ICLR 2022 — the retired visit-count-tempered target and why its confidence proxy fails under determinization (§20.1).
[^awr]: Peng et al., AWR, arXiv:1910.00177, 2019; Nair et al., AWAC, arXiv:2006.09359, 2020; Wang et al., CRR, arXiv:2006.15134, 2020; Kostrikov et al., IQL, arXiv:2110.06169, 2021 — advantage-weighted extraction, temperature and weight clip.
[^lwf]: Li & Hoiem, "Learning without Forgetting," arXiv:1606.09282, 2016; Hinton, Vinyals & Dean, "Distilling the Knowledge in a Neural Network," arXiv:1503.02531, 2015 — the retention KL on heads search cannot speak to.
[^lpft]: Kumar et al., "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution," arXiv:2202.10054, 2022 (LP-FT) — precedent for separating trunk and head training epochs; our order is trunk-first then head-only, chosen empirically (CE_Teacher §20.9 arms 5d/5f/P1).
[^wise]: Wortsman et al., "Robust Fine-Tuning of Zero-Shot Models" (WiSE-FT), arXiv:2109.01903, 2022 — weight interpolation as the walk-back (CE_Teacher §17.16).
[^tesauro]: Tesauro, Rajan & Segal, "Bayesian Inference in Monte-Carlo Tree Search," UAI 2010 — posterior-of-optimality readouts within one search.
[^kg]: Kendall & Gal, "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NeurIPS 2017 (arXiv:1703.04977) — the per-node heteroscedastic fallback (CE_Teacher §20.4).
[^jacob]: Jacob et al., "Modeling Strong and Human-Like Gameplay with KL-Regularized Search," ICML 2022 (arXiv:2112.07544) — KL-regularized search toward a human-imitation prior; adjacent in method (regularized search targets), opposite in goal (imitating humans vs constraining machine recall).
[^tg]: Tesauro & Galperin, "On-line Policy Improvement using Monte-Carlo Search," NIPS 1996 — rollout-based policy improvement (backgammon), the ancestor of search-as-improvement-operator.
[^lp]: Lagoudakis & Parr, "Reinforcement Learning as Classification: Leveraging Modern Classifiers," ICML 2003 — approximate PI by classifying rollout-labeled actions (the Stage-3 CE projection in its classical form).
[^scherrer]: Scherrer et al., "Approximate Modified Policy Iteration and its Application to the Game of Tetris," JMLR 16, 2015 — classification-based MPI; error propagation across iterations.
[^exit]: Anthony, Tian & Barber, NeurIPS 2017; Sun et al., "Dual Policy Iteration," NeurIPS 2018 (arXiv:1805.10755) — fast-policy / slow-search loops and when the projection contracts.
[^az]: Silver et al., Nature 550, 2017 (AlphaGo Zero); Science 362, 2018 (AlphaZero); Schrittwieser et al., "Online and Offline Reinforcement Learning by Planning with a Learned Model" (MuZero Reanalyze), arXiv:2104.06294, 2021 — offline corpus re-search across iterations (the state-reuse amendment held in reserve, CE_Teacher §16.9 addendum 3).
[^soemers]: Soemers, Piette, Stephenson & Browne, "Learning Policies from Self-Play with Policy Gradients and MCTS Value Estimates," IEEE CoG 2019 (arXiv:1905.05809) — PG with search-derived values inside one loop; contrast with our measured two-operator failure and phase separation.
[^ppg]: Cobbe et al., "Phasic Policy Gradient," arXiv:2009.04416, 2020 — alternating policy and auxiliary phases with a behavior-cloning constraint; the nearest precedent for head-partitioned alternating operators.
[^perceiver]: Jaegle et al., "Perceiver IO," arXiv:2107.14795, 2021 — learned-query cross-attention readout over a token set.
[^settrans]: Lee et al., "Set Transformer," arXiv:1810.00825, 2019 — pooling by multi-head attention (the AttentionPool / shared-readout lineage).

Also load-bearing but internal: Schrittwieser et al. MuZero (Nature 588, 2020); Wu, KataGo, arXiv:1902.10565, 2019 (auxiliary targets shaping the trunk); Agarwal et al., "Reincarnating RL," arXiv:2206.01626, 2022 and Schmitt et al., "Kickstarting," arXiv:1803.03835, 2018 (warm-start and anneal-to-zero distillation, the precedent for phase-pure handoff rather than concurrent teaching); Brown et al., ReBeL, arXiv:2007.13544, 2020 and Schmid et al., "Student of Games," Science 382, 2023 (public-belief-state search, the full-rewrite alternative held in reserve); Schaul et al., PER, arXiv:1511.05952, 2016 (stratified emission's annealed-bias rationale).

---

## Appendix A — contingencies and open items

- **Leaster play under policy iteration** (operator concern 2026-09-02).
  Leaster rows carry only the chained retention KL to θ_k plus value
  regression: bounded drift per iteration, no improvement signal,
  possible random walk across iterations. Instrument added to the cert:
  leaster-conditioned paired score (rigorous_eval tags hands). Escalation:
  (1) fixed-reference anchor — anchor leaster rows to the handoff
  checkpoint, bounding cumulative drift at handoff competence
  (AlphaStar-style fixed KL reference); (2) leaster-play emission behind
  the addendum-5 mini-calibration gate (P4 leaster determinizer exists).
- **Bidding-node emission** if the frozen-trunk bidding phase yields
  nothing (§4.4 step 6).
- **`perceiver-recall-ctxmem` twin**: diagnostic arm only, on a gen-2
  gate failure.
- **Uniform-window population** in place of PFSP: measured near-
  equivalent, not in this run; a future simplification with its own
  pre-registration.
- **Terminal-only from-scratch bootstrap**: a future ablation of the
  shaped phase, not this run's risk.
- **Reanalyze-style corpus reuse across iterations** (re-search stored
  states under θ_{k+1}) if corpus cost becomes the binding constraint.
- **Human-expert comparison** requires human game data from the app;
  out of scope until it exists.
