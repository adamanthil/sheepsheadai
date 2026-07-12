# ISMCTS Soft-Teacher Refactor Plan

> **Note (2026-07-11):** file paths in this notebook predate the 2026-07 repo reorganization (core modules now live in `sheepshead/`, the hosted product under `app/`). Kept as-is for the historical record.

Status: planning (no implementation started). This document is the reference for
replacing the hand-crafted exploration/shaping machinery in PFSP training with a
single search-derived soft-target teacher.

---

## 1. The problem

### 1.1 The presenting symptom
The 30M PFSP agent (`pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt`)
leads trump as a *defender* on the first trick when a fail lead is available —
against human convention. Paired counterfactual rollouts on true deals
(`counterfactual_trump_leads.py`) showed this is genuinely EV-negative
(~-0.23 game score, ~-4 card points; n=86). It is rare (~1.6% of trick-0
defender leads) but it is a real mistake, and it *re-emerged* after the
defender-trump reward-shaping penalty annealed to zero (~12M: ~0%; 30M: ~1.6%).
Shaping was *suppressing* the action, not teaching the agent why it is wrong.

### 1.2 Why the critic cannot fix it
A frozen-encoder probe (`critic_probe.py`) showed value-prediction R^2 rises
monotonically by trick (0.19 -> 0.81) and that a deeper value head does no
better than a shallow one on the trick-0 leak states (R^2 ~ 0.08 for both). The
weakness is a **partial-observability ceiling**: at trick 0 the outcome is
dominated by the four hidden hands, which no value function over the observed
state can subtract. PPO's advantage `G(s,a) - V(s)` therefore carries the full
hidden-hand variance into the gradient, drowning the small (~0.23) EV gap. The
committed critic changes (deeper `value_trunk`, gamma 0.95->0.99,
value_loss_coeff 0.5->1.0) are kept as general value-stability improvements
(and as the search rollout-truncation value), **not** as a fix for this leak.

### 1.3 The deeper problem: a zoo of hand-crafted exploration patches
The leak is one symptom of a structural limit. On-policy PPO only learns from
actions it samples, so when the policy collapses into a local optimum it cannot
get a gradient toward the better action it never tries. The training loop
currently compensates with six families of hand-tuned, metric-triggered patches,
all solving the *same* failure:

| Mechanism | Trigger | What it forces | Code |
|---|---|---|---|
| pick/partner/bury entropy bumps | pick/alone/bury-quality rate out of band | inject entropy to escape a collapsed mode | train_pfsp.py 832-838, 1056-1074, 1217-1239, 1281-1311 |
| PASS-floor epsilon | pick rate high AND picker avg negative | force trying PASS | train_pfsp.py 1313-1338; ppo.py `_apply_head_epsilon_mix` |
| PICK-floor epsilon | pick rate very low | force trying PICK | train_pfsp.py 1340-1362 |
| partner CALL-uniform mixture | alone rate high AND picker avg poor | force trying CALL over ALONE | train_pfsp.py 1241-1279 |
| reward shaping schedules | scheduled anneal | inject a human prior, then remove (incl. the -0.06 defender-trump penalty) | train_pfsp.py 121-134, 656-680; training_utils.py `update_intermediate_rewards_for_action` |
| per-head entropy decay | scheduled | baseline exploration | train_pfsp.py 797-830 (KEEP) |

Every one is a blanket, metric-triggered approximation of "make the policy try
the better action it isn't trying." A search teacher provides the *principled*
version of the same signal: a directed, per-state, outcome-grounded target.

### 1.4 The fix, and the evidence it works (Gate 0)
An information-set-respecting **determinized teacher** averages over legal deals
of the hidden cards, cancelling the hidden-hand variance PPO cannot. Validated
offline (`gate0_determinizer.py`) on the trick-0 leak states (n=50):

- Direction: lowers trump-lead probability in 90% of states (mean dp = -0.47).
- Calibration: inference-corrected EV gap = -0.395 (SE 0.107), matching the
  true-deal oracle -0.379 (SE 0.147) essentially exactly; significantly negative.

**Critical, non-obvious finding — determinization MUST be inference-weighted.**
Naive uniform redeal is biased (it made trump leads look +0.09 *good*) because it
ignores the bidding: the real picker self-selected a trump-rich hand (4.58 trump
in its 8 vs 3.21 uniform; 3.50 vs 2.10 top Q/J). The fix, validated, is to
condition the redeal on the observed actions: rejection-sample worlds by the
policy's probability of the observed pick, then importance-weight the residual
(passers + call) likelihood. This lifted effective sample size from ~7/60 to
~17/40 and recovered the oracle. The production determinizer must implement this
inference (cf. Skat literature: Buro et al., "state evaluation, *inference*, and
search"; Long et al. AAAI 2010 on when PIMC succeeds; Cowling et al. 2012 on
ISMCTS).

### 1.5 Stage A result, and the mid-game weighting decision (validated)
The generalized determinizer (`Game.sample_determinization`, any decision point)
was validated offline (`stage_a_determinizer_check.py`, n=24/trick) against the
paired true-deal oracle at tricks 1-3. Findings:

- **Determinizer is correct.** 100% of sampled redeals are legal (partition /
  counts / play-revealed voids / called-ace placement / bury), and the forced
  replay reproduces the exact public history at all three tricks. (Three bugs
  found and fixed along the way: the bury sampler could bury a lone called-suit
  card; a seat that *led* the called suit was wrongly eligible to hold the called
  ace as secret partner; and the picker's CALL — especially an under-call —
  must be legality-checked against the determinized 8 even after the ace is
  revealed.)
- **Calibration: aggregate-unbiased but noisy.** Weighted determinized EV tracks
  the oracle in aggregate at all three tricks (within 2x combined SE), but it is
  a *weak* pass: the single-deal oracle is high-variance, mid-game ESS is low
  (~5-10/30), and per-state fidelity is hard to confirm. This is materially
  softer than the trick-0 Gate-0 result.

**Decision — three weighting schemes; we use the middle one (B).**
- **A Uniform:** no weighting; plays constrain only via voids.
- **B Bidding-weighted (chosen):** self-normalized importance weight by the
  policy likelihood of **pick + pass + call**; plays enter as **hard void
  constraints only**, never as soft weights.
- **C Full:** also weight every play's likelihood — **rejected**: the long
  product of per-play probabilities collapses ESS to ~1 (one world dominates).

Rationale: the dominant play-derived signal is "who could not follow," which B
captures *exactly* as a hard constraint; the residual "which legal card did they
choose" is second-order and is what destroys ESS in C. Keeping the **bidding**
weight is near-free (we must replay the bidding to rebuild recurrent memory
anyway, so the pick/call likelihood is one forward pass we already do) and stays
realistic — worlds where the determinized opponent would actually have
picked/called are upweighted, and pick/call decisions carry outsized leverage.
The bid is included via importance weight, **not** a rejection step (rejection on
P(pick) zero-skips states where every redealt picker looks weak). Note B helps
most at trick 0 (Gate 0); by mid-game uniform ≈ weighted because the voids
already pin down picker strength — so B is cheap insurance, not a mid-game
necessity. Full belief-search over play likelihoods (ReBeL-style) remains the
deferred heavy escalation if mid-game strength ever demands it.

---

## 2. Target architecture

`PPO backbone + small per-head baseline entropy + ONE search-distillation term (which owns the policy update on confidently-searched transitions; PPO owns the rest).`

- **Teacher:** single-observer ISMCTS over the training agent's information sets,
  with inference-weighted determinization, applied to **all heads**
  (pick / partner / bury / play).
- **Target:** soft, visit-count distribution `pi'(a) ∝ N(a)^(1/tau_target)`.
- **Loss:** one cross-entropy / forward-KL term `w_distill * CE(pi', pi_theta)` on
  searched transitions; gated by an effective-sample-size floor.
- **Division of labor (why keep PPO at all):** search is *expensive*, so it runs
  on only a fraction of decisions (`f_play ≈ 0.1`); PPO is what supplies the
  cheap, dense, outcome-grounded policy gradient on the ~85–90% of plays search
  never reaches, and an independent return-grounded anchor for a teacher that is
  only a *targeted corrector* of the collapse failure mode (§1.3), not a
  validated complete policy (its mid-game ESS is low and variable). On the
  transitions search *does* cover confidently, the policy gradient is masked off
  and the teacher owns the update (Stage C); the critic/value loss runs
  everywhere. Pure ExIt (drop PPO, distil only) is the *destination*, viable once
  search is cheap (batched/async, §5) or the teacher is well-calibrated mid-game;
  Phase-2 (value regressed to the search root) is the first step toward it.
- **On-policy preserved:** the agent still *acts* by sampling `pi_theta`; search
  is teacher-only, run on a fraction of decisions. Population opponents and
  saved snapshots never search, so **all search cost is training-time; deployment
  stays fast** (the ExIt bargain: amortize search into the network).
- **Deleted:** all six controller families above and all reward shaping —
  **including the per-trick trick-point reward, not just the hand-tuned nudges.**
  The episode return becomes exactly `final_score / 12` for every game mode; the
  per-head baseline entropy decay is the only signal that remains. Rationale: the
  trick-point reward is itself shaping (a raw, non-telescoping bonus), and its
  mode-dependence is what forces the hand-tuned `LEASTER_FINAL_REWARD_BONUS`
  (training_utils.py:369) — in a leaster every seat can only be *dinged*
  (`apply_leaster_trick_rewards`), a negative-sum band that biases the bidding EV
  against passing regardless of leaster win-likelihood, so a constant is bolted on
  to re-baseline it. Dropping the trick reward deletes both the shaping and the
  bonus in one move: `get_score()` already scores leasters correctly (+4 winner /
  −1 others, with the ≥1-trick qualification in `get_leaster_winner`), so the
  pick/pass decision is then driven purely by `E[final_score]` — i.e. directly by
  win-likelihood, the property we want, with zero hand-tuning. It also **aligns
  the critic with the search**: `ismcts.py` already bootstraps on terminal
  `get_score()/12` only, so a critic trained on shaped return is predicting a
  different quantity than the rollout assumes; terminal-only return closes that
  gap. If a post-refactor diagnostic shows the critic underfitting early-game
  without the dense reward, the *only* principled way to add density back is
  **potential-based shaping** (`F = γΦ(s′) − Φ(s)`, `Φ(terminal)=0`) — provably
  policy-invariant and mode-independent, so it can never reintroduce a leaster
  bias or need a correction constant. Start without it; add only if measured.

---

## 3. Locked design decisions

| Decision | Value |
|---|---|
| Search algorithm | SO-ISMCTS; statistics-only nodes; recurrent memory re-derived per iteration along the descended path |
| Selection | PUCT, `c_puct = 1.25`, network prior `P(a)`; availability counts at non-lead nodes, plain PUCT at lead nodes |
| Leaf evaluation | truncated rollout `d_rollout` plies then `value_trunk` V-bootstrap. **Stage B revised:** `d_rollout = 2` is too shallow early-game (trick-0 critic is in the partial-obs blind spot §1.2 → only a weak leak correction); roll deep early, shallow once critic R² ≳ 0.5 |
| Belief / search coupling | **Stage B addition:** belief enters by world-**resampling** (SIR ∝ `exp(log_w)`) feeding a **unit-weight** tree, not by weighting visit counts (which collapses the budget to ESS) |
| Exploration vs collapsed prior | **Stage B addition:** optimistic FPU (`fpu = 1.0`) + MuZero min-max Q-norm + uniform root mix (`root_explore_frac = 0.25`) — otherwise PUCT starves the near-zero-prior better action and the target echoes the leak |
| Determinization | sample hidden hands respecting voids + all hard constraints; self-normalized importance-weight worlds by the **bidding** likelihood (pick/pass/call) only; plays enter as **hard void constraints**, never as soft weights (see §1.5) |
| Heads searched | all; heavier on pick/partner/bury |
| Per-head search fraction | `f_pick = f_partner = f_bury = 1.0`, `f_play ≈ 0.10–0.15` |
| Iterations per search | `M_pick = 48, M_partner = 64, M_bury = 96, M_play = 96` |
| Target | visit-count `pi'(a) ∝ N(a)^(1/tau_target)`, `tau_target = 1.0` |
| Distillation | `w_distill = 1.0`, forward KL / cross-entropy, masked to searched transitions |
| ESS safeguard | `ESS_floor = 4`; below floor -> skip target (transition still trains via PG), log abort fraction |
| Acting policy | always `pi_theta` (search teacher-only); on-policy preserved |
| Reward / return | **terminal score only** (`final_score / 12`), identical for all modes; **no per-trick reward, no leaster bonus** — leaster EV is carried natively by `get_score()` + search/critic. PBRS is the only sanctioned fallback if density is later needed |

Phase-2 (deferred until after the refactor is validated): also regress V toward
the search root's backed-up value (AlphaZero-style virtuous loop).

---

## 4. Implementation stages

Stages A and B are standalone and **validated independently and offline** before
any training-loop change. Only once both are green do we do the single training
loop refactor (Stage C).

### Stage A — generalized inference-weighted determinizer ✅ DONE
- **Where:** `Game.sample_determinization(observer, rng)` in sheepshead.py
  (replaced `sample_trick0_determinization`; works at any decision point).
  Decomposed into `_determinization_context` (per-call invariants),
  `_sample_deal_attempt` (one shuffle), and helpers (`_play_revealed_voids`,
  `_draw_avoiding`, `_call_is_legal`, `_picker_discards_legal`).
- **Honours:** per-seat remaining-card counts; play-revealed voids (`T/C/S/H`);
  forced placement of already-played cards; called-ace placement (and the
  secret-partner can't have *led* the called suit); picker-8 call legality
  (incl. under-calls, even after the ace is revealed); legal under/bury discards.
- **Weighting:** scheme B (see §1.5) — bidding-likelihood SNIS, plays as hard
  void constraints.
- **Result:** legality 100% and replay reproduces the public history at tricks
  1-3; calibration aggregate-unbiased vs the paired oracle but a weak (noisy,
  low-ESS) pass. See §1.5. Validated via `stage_a_determinizer_check.py`
  (one-off, **not committed**).

### Stage B — SO-ISMCTS engine (`ismcts.py`, standalone) ✅ DONE
- **Where:** `ismcts.py` (committed): `ISMCTSTeacher.search(real_game, observer,
  forced_public, rng) -> {pi, ess, ok, head, n_iter, valid, root_n, root_q}` and
  `ISMCTSConfig`. Recurrent memory is re-derived each iteration by forced replay
  of the public record into the determinized world (Stage-A mechanism, extended
  forward through the in-tree descent and rollout). Nodes store statistics only.
- **Output:** `pi'(a) ∝ N(a)^(1/tau_target)`; `ok = ESS >= ESS_floor`.
- **All heads** via one engine: pick/partner/bury are shallow roots
  (`max_depth = 1`, degenerate to determinized rollout evaluation of each
  option); play is the deep tree (`max_depth = 6`).

**Three design corrections found necessary during offline validation (each was
a hard FAIL before the fix; all are now in `ismcts.py`):**

1. **Belief enters by world-RESAMPLING, not by weighting the tree counts.**
   Weighting each iteration's visit increment by its self-normalized bidding
   weight (the obvious reading of "importance-weight worlds") collapses the
   *effective* root visit budget to the ESS (~5–15, not `M`), so the visit-count
   target just echoes the policy. Fix: build a pool of `M` determinized worlds,
   then run `M` **unit-weight** tree iterations each sampling a world `∝
   exp(log_w)` (SIR). The bidding belief enters via sampling frequency; the tree
   gets the full `M` visits. (This is *world*-resampling within a state, not the
   state-skipping P(pick) rejection §1.5 rejected.)
2. **PUCT must be protected from the collapsed prior** (the disease itself). With
   the leaked policy assigning the better action a near-zero prior, vanilla PUCT
   never explores it (`N_fail = 0`) and the target reproduces the leak. Fixes:
   **optimistic first-play urgency** (`fpu = 1.0`, every legal action tried
   before any is revisited), **MuZero min-max Q-normalization** (the `score/12`
   action-value gaps are tiny next to prior-weighted exploration; normalizing to
   [0,1] makes `c_puct = 1.25` meaningful), and a small **uniform root-prior mix**
   (`root_explore_frac = 0.25`).
3. **Rollout depth must reach the observable outcome where the critic is blind.**
   The trick-0 leak lives exactly in the partial-obs ceiling the critic cannot
   subtract (§1.2). With `d_rollout = 2` + V-bootstrap the bootstrap lands at
   ~trick 2–3 and the correction is **weak** (mean dp = −0.06, lowers in 62%).
   With deep rollouts to (near) terminal it is **strong** (mean dp = −0.17,
   lowers in 75%), matching the Gate-0 direction. ⇒ **Stage C should roll deeper
   early-game** (the plan's flat `d_rollout = 2` relies on a mid-game critic R²
   that does not exist at trick 0); shallow + bootstrap is fine once R² ≳ 0.5.

- **Validation (offline) — `stage_b_ismcts_check.py`, one-off, NOT committed.**
  Two modes: `trick0` reproduces the Gate-0 leak correction (above results,
  n=16 trump-leaning seat-1 defender leads, M=96); `h2h` plays paired games with
  a focal seat acting on the teacher's argmax `pi'` over its play decisions, with
  an optional true-deal oracle EV check (teacher-argmax vs policy-argmax). Per
  the plan, STRENGTH (correct ranking) is the bar, not point-EV calibration.
- **ESS health:** trick-0 mean ESS ≈ 12–14/96; `ESS_floor = 4` skips ~25–30% of
  states (consistent with §6's scheme-B mid-game ESS warning).

### Stage C — single clean training-loop refactor
One decisive change to `ppo.py` + `train_pfsp.py` (+ shaping in
`training_utils.py`) that simultaneously adds the teacher and strips all
sample-adjustment / reward-shaping machinery. Done as one refactor (not
incremental deletions) because A and B give us prior confidence the teacher
works.

**Additions:**
- Buffer: extend the event dict in `store_episode_events` (ppo.py ~978) with
  optional `search_target: float32[action_size]` and `has_search_target: bool`.
  `compute_gae` unchanged.
- Loss: in `update()` (ppo.py ~1543), add
  `L_distill = w_distill * mean_over_searched( sum_a pi'(a) * (log pi'(a) - log pi_theta(a)) )`
  masked by `has_search_target`, using the same per-head masking the PPO term
  already applies.
- **Policy-gradient mask on confidently-searched transitions (Stage-B-informed).**
  On a transition where `has_search_target` is true (i.e. the teacher produced
  an `ok` target — root ESS ≥ `ESS_floor`), train the policy by **distillation
  only: drop the PPO policy-gradient (clip) term there.** Rationale: on exactly
  these states PPO's advantage `G−V` is dominated by hidden-hand variance and is
  blind to the small EV gaps the teacher corrects (§1.2), so the two objectives
  fight — PG pulls back toward the sampled (possibly leaked) action while the
  teacher pulls toward the searched one. Hand each state to its better teacher:
  search where we paid for it, PG everywhere else. **This is safe because the
  reliance is gated and the teacher is unbiased-in-expectation, not relied on
  per-state:**
    - *Gated:* the mask fires only on `ok` (ESS ≥ floor) transitions. Searched
      states that abort on `ESS_floor`, and all unsearched states, keep the PG
      term — so we never drop PPO's outcome grounding where the belief is thin.
    - *Unbiased, not precise:* distillation averages over the training
      distribution, so it needs the teacher to be unbiased in *expectation*
      (validated: Stage A scheme-B determinizer is aggregate-unbiased; Stage B's
      trick-0 result is unbiased+noisy, not biased — the one systematic bias,
      the shallow-bootstrap blind spot, is removed by deep-early rollouts below).
      The residual is variance, which distillation tolerates.
  - **Keep the PPO *value* (critic) loss on ALL transitions**, searched or not —
    only the *policy-gradient* term is masked off on `ok`-searched transitions.
  - **A/B this against the plain additive form** (`PPO_clip + w_distill·CE` on
    searched states) during the post-refactor run; promote whichever holds the
    guarded metrics better. If ESS 4–8 targets prove too noisy under the hard
    mask, the lever is raising `ESS_floor` (or, as a fallback, a small residual
    PG weight on searched states) — not abandoning the mask.
- **Search call with a depth schedule (Stage-B-informed).** In `play_episode`
  (train_pfsp.py ~183), for the training agent's decisions, with per-head
  probability `f_head`, run the ISMCTS teacher to produce `pi'`; store it (and
  `has_search_target = ok`). The agent still acts by sampling `pi_theta`.
  **Do NOT use a flat `d_rollout = 2`:** Stage B showed it is too shallow
  early-game — the trick-0 critic bootstrap lands in the partial-obs blind spot
  (§1.2) and the leak correction is weak (dp ≈ −0.06, 62%), whereas deep rollouts
  recover the Gate-0 direction (dp ≈ −0.17, 75%). Use a **trick-indexed rollout
  depth**: roll deep (to/near terminal) in the early tricks where the critic R²
  is low, and shorten to `d_rollout` plies + V-bootstrap once R² ≳ 0.5
  (mid-game). **Caveat — the earlier `d_rollout = max(2, 6 − current_trick)` form
  was wrong:** `d_rollout` counts the observer's remaining *play* plies before
  bootstrapping, and `6 − current_trick` is *exactly* the number of observer plies
  left to the end of the hand, so that formula rolls to terminal at every trick
  and **never exercises the critic bootstrap** (the `max(2, …)` floor only binds
  at tricks 4–5, where 2 still ≥ the plies remaining). It silently means "full
  rollout always." That is correct for *accuracy* (and a full sheepshead rollout
  is only ≤6 plies), but it forfeits the recurrent-re-encode savings the bootstrap
  exists to buy (§5's dominant cost is per-ply re-encoding), so the schedule must
  truncate to be worth having. Corrected, decoupled-from-length form:
  `d_rollout = (6 − current_trick) if current_trick <= T_full else d_short`
  — full rollout for the early tricks `0..T_full` (dodging the blind spot),
  then a genuinely *short* fixed depth `d_short` (≈2) that lands the bootstrap
  before terminal from trick `T_full+1` on. **Gate `T_full` on measurement, not a
  guess:** the "R² ≳ 0.5 by trick 2–3" claim is still unverified — use the
  existing `critic_calibration.py` / `critic_probe.py` probes to find the earliest
  trick at which the value head is trustworthy and set `T_full` to the last trick
  below it (default `T_full = 1` as a conservative placeholder until measured).
  `ismcts.ISMCTSConfig` exposes `d_rollout`; thread the per-call value through
  `ISMCTSTeacher.search`.
- **Raise search *coverage* (`f_head`) to backfill the removed dense reward.**
  With the per-trick reward gone, an *unsearched* play transition's policy
  gradient comes only from the terminal-score advantage via GAE — i.e. it leans
  entirely on the critic and is sparse/high-variance — whereas a *searched*
  transition gets a dense, low-variance, search-grounded target `pi'`. So raising
  `f_play` (currently `≈ 0.10–0.15`) directly replaces the dense signal we deleted
  with a *better* one (search-grounded, not hand-shaped). Two cautions: (1) this
  is a compute dial — each searched transition costs a full search, so it trades
  against throughput (§5); (2) keep the two knobs distinct — `f_head` is search
  *coverage* (how many transitions get a target), `M` (iterations/search) is target
  *resolution* (how sharp each target is), and raising `M` past the world-pool
  diversity is wasted (the target's quality is bounded by `min(M-resolution, ESS)`,
  so bump `M` only alongside the pool size). The critic still backstops every
  unsearched transition (value loss runs everywhere), so this is a mix dial, not a
  signal on/off. Set it by measurement: if the searched-subset `KL(pi' ||
  pi_theta)` stays large or `pi_theta` entropy drifts on the unsearched mass,
  raise `f_play` (and/or `M`) — a cheap diagnostic on the first refactor run
  before committing a higher budget to a full retrain.
- Logging in the update block: mean ESS, ESS-abort fraction, fraction of
  transitions PG-masked, mean `KL(pi' || pi_theta)` (the gap the teacher
  corrects), `pi'` entropy, search wall-time fraction.

**Deletions (all at once):**
- train_pfsp.py: entropy-bump scheduling and application (832-838, 1056-1074,
  1217-1239, 1281-1311); PASS-floor controller (1313-1338); PICK-floor controller
  (1340-1362); partner CALL-eps controller (1241-1279); all four
  `shaping_schedule_*` and their application (121-134, 656-680); the associated
  HyperParams fields (pick/partner/bury bump, *_floor_eps_*, partner_call_eps_*,
  shaping_schedule_*).
- ppo.py: `_apply_head_epsilon_mix` and its call sites (769-834, 890, 925, 1500);
  `set_partner_call_epsilon` / `set_pass_floor_epsilon` / `set_pick_floor_epsilon`
  (693-718) and their state in train_pfsp.py; the snapshot-time epsilon-disabling
  (train_pfsp.py 916-923) becomes unnecessary.
- training_utils.py: the shaping path in `update_intermediate_rewards_for_action`
  (incl. the -0.06 defender-trump penalty) **and the per-trick reward path**:
  `calculate_trick_reward`, `apply_trick_rewards`, `apply_leaster_trick_rewards`,
  `handle_trick_completion`'s reward application, and the `LEASTER_FINAL_REWARD_BONUS`
  branch in `process_episode_rewards` (lines 365–372). The episode return collapses
  to `final_score / 12` at the terminal step for every mode — keep only that.
- **Kept:** per-head baseline entropy decay (train_pfsp.py 797-830); LR schedules;
  PFSP population/sampling; all evaluation/logging not listed above.

**Post-refactor validation:**
- A training run with the clean loop; compare against the current agent.
- Monitor the metrics the deleted controllers used to guard — pick rate, ALONE
  rate, bury-quality rate, leaster rate, picker avg — to confirm the teacher
  holds them in band *without* the hand-crafted machinery.
- Track the trick-0 defender trump-lead rate (the original leak) and overall
  head-to-head strength vs the 30M checkpoint.

---

## 5. Cost and knobs
- Dominant new cost: recurrent re-encoding per search iteration. Levers:
  per-head search fraction `f`, iterations `M`, rollout depth `d_rollout`, and
  batching determinizations through the encoder.
- Critic's role: `value_trunk` is the rollout-truncation value; better mid-game
  calibration (R^2 ~0.5+ at trick 2–3) makes shallow rollouts cheap and adequate.
- Deployment cost: zero — snapshots and the shipped policy are the distilled
  network; search is training-time only.
- If in-loop search proves too slow at the throughput-tuning step, fallback is an
  offline/asynchronous target-generation pass decoupled from fast self-play (more
  plumbing; only if needed).

---

## 6. Risks and open items
- **Mid-game ESS is low (~5-10/30 from Stage A).** With scheme B the bidding
  weight gives only ~12-15% ESS ratio, so per-state mid-game targets are noisy
  and `ESS_floor=4` would skip ~25-40% of mid-game states. Mitigations: it
  matters less than feared (uniform ≈ weighted mid-game, since voids dominate),
  the soft target only needs a *ranking*, and Stage B is judged on strength not
  point-EV. If it bites, options are more iterations `M`, a better proposal
  (sequential/particle resampling), or simply dropping to scheme A mid-game.
- **Single-deal oracle is too noisy for a tight mid-game EV verdict** — hence the
  Stage B switch to head-to-head strength as the primary metric.
- **Determinizer rejection waste on under-calls:** to replay an under-call the
  determinized picker must be void in the called suit; we enforce this by
  resampling, which can raise the rejection rate on under-call states. Acceptable
  now; a suit-aware proposal would remove it if needed.
- **Strategy fusion / non-locality:** SO-ISMCTS with a network rollout policy is
  the pragmatic regime the trick-game literature supports; full MO-ISMCTS or
  belief-search (ReBeL-style) is a heavier escalation we are not taking now.
- **One-shot refactor risk:** removing six controllers at once means a regression
  cannot be isolated to a single removal; mitigated by independent A/B validation
  and the guarded-metric monitoring on the post-refactor run.
- **Analysis scripts** (`stage_a_determinizer_check.py`, `stage_b_ismcts_check.py`,
  `gate0_determinizer.py`, `gate0_inference_check.py`,
  `counterfactual_trump_leads.py`, etc.) are one-off and not slated for commit.
  `Game.sample_determinization` and its helpers in sheepshead.py, **and the
  `ismcts.py` engine (`ISMCTSTeacher` / `ISMCTSConfig`)**, ARE committed
  production code.
