# ISMCTS Soft-Teacher: Problem, Progress, and Roadmap

*Companion to the detailed technical plan in `ISMCTS_Teacher_Refactor_Plan.md`.
This file is the higher-level narrative: what we are solving, what is done, and
the agreed plan forward. Last updated 2026-05-29.*

---

## 1. The problem we are solving

The Sheepshead agent is trained by PPO self-play / PFSP. Two structural issues
motivated moving to an Expert-Iteration (ExIt) design with a search-derived
teacher:

1. **A partial-observability ceiling the critic cannot fix.** The 30M-step agent
   leaks information on the first trick — most visibly, defenders lead trump from
   hands where that is revealing/incorrect. The value head cannot subtract this
   bias because the relevant information is genuinely hidden at that node (see
   `project_trump_lead_investigation`). No amount of critic tuning closes it; a
   lookahead/search signal is the structural fix.

2. **Hand-tuned shaping and controllers as crutches.** The training loop carried
   substantial hand-crafted machinery: per-trick point rewards, bidding/lead
   shaping nudges, a leaster reward bonus, and adaptive exploration controllers
   (PASS-floor, PICK-floor, partner-CALL epsilon, entropy bumps). These encode a
   human prior and require constant tuning. We want to replace them with one
   principled, outcome-grounded signal.

**The thesis:** a single-observer ISMCTS soft-teacher produces a per-state,
outcome-grounded target `pi'(a)` over the agent's information set. Distilling
`pi'` into the policy (ExIt / AlphaZero-style) replaces the hand-tuned signals
with search, and corrects the partial-observability leak that the critic can't.
Deployment stays fast — search is training-time only; the shipped network is the
distilled policy.

---

## 2. What we have done

- **Stage A — inference-weighted determinizer** (`Game.sample_determinization`,
  committed). Samples a full deal consistent with the observer's information set
  at any *post-bidding* decision point: honours per-seat counts, play-revealed
  voids, forced plays, and called-ace constraints. Weighting is scheme B
  (bidding-likelihood SNIS; plays as hard void constraints). Validated:
  100% legality and replay reproduces the public record.

- **Stage B — the SO-ISMCTS engine** (`ismcts.py`, committed). `ISMCTSTeacher.
  search(real_game, observer, forced_public, rng)` returns `pi'`, ESS, and an
  `ok` (ESS ≥ floor) flag. Three non-obvious findings (each a hard failure before
  the fix), now baked in: belief enters by world-**resampling** (SIR), not by
  weighting visit counts; PUCT needs optimistic FPU + min-max Q-norm + a uniform
  root mix to escape the collapsed prior; and rollouts must reach the observable
  outcome early-game (deep-rollout-early), where the critic is blind. Offline
  validation reproduced the Gate-0 leak correction. (See
  `project_ismcts_stage_b`.)

- **Reward analysis.** We decided the episode return should be **terminal-only**
  (`final_score / 12`), deleting both the per-trick reward and the hand-tuned
  leaster bonus. The leaster bonus only existed to re-baseline the negative-sum
  trick-point shaping; with terminal-only return, `get_score()` already scores
  leasters correctly, so pass→leaster EV is win-likelihood-driven with zero
  hand-tuning. This also aligns the critic target with what the search already
  bootstraps. (See `ISMCTS_Teacher_Refactor_Plan.md` §2.)

- **Stage C — training-loop integration (first cut, smoke-validated).** Added
  the distillation loss + **PG-mask** (drop the PPO clip term on confident
  searched transitions; keep the value loss everywhere) to `ppo.py`; wired the
  teacher, `forced_public` tracking, a trick-indexed rollout-depth schedule, and
  the terminal-only reward into the play loop. End-to-end smoke + a leaster-guard
  test pass. (See `project_ismcts_stage_c`.) **This first cut was written inline
  in `train_pfsp.py`; it is being re-homed — see §4.**

- **Robustness realization (the reason for the refactor below).** Stripping
  bidding shaping *and* not searching the bidding head removes both the
  conditional crutch and its principled replacement on the most collapse-prone
  head (always-pick / always-pass). Entropy is only a state-independent pressure
  and cannot encode "pick with a strong hand, pass with a weak one." The correct
  fix is to **search the bidding head** (which also natively models the
  all-five-pass → leaster branch), not to re-add shaping.

---

## 3. Target architecture

Three training strategies, separated into thin entry points over shared
libraries:

| Strategy | Entry point | Exploration | Reward |
|---|---|---|---|
| Self-play PPO (legacy) | self-play trainer | shaping + entropy | shaped |
| PFSP PPO (baseline) | PFSP trainer | shaping + controllers | shaped |
| PFSP ExIt (hybrid, new) | ExIt trainer | ISMCTS distillation + PG-mask | terminal-only |

**Shared libraries:**
- `training_utils.py` — shared *pure* functions: reward fns (`process_episode_
  rewards` shaped + `process_terminal_rewards`), trick tracking, hand-strength,
  strategic-decision helpers, the **single** `RETURN_SCALE` constant,
  interpolation. (Today these are scattered / duplicated / inline.)
- `config.py` — unified hyperparameters: `PFSPHyperparams` gains a nested
  `search: SearchConfig | None` and `reward_mode ∈ {shaped, terminal}`. No
  config-like objects floating in the loop scripts.
- `pfsp_runtime.py` — the shared PFSP machinery: one parameterized
  `play_population_game(..., teacher=None, reward_mode=...)` and the training
  driver (population sampling, OpenSkill, checkpointing, logging, strategic
  eval). Both PFSP entry points are thin wrappers over it.
- `ppo.py` — the `search_target` / distillation / PG-mask additions are a
  **backward-compatible superset**: with no search targets, `distill_loss = 0`
  and nothing is masked, so the pure-PPO trainers are unaffected. Shared infra.

The frozen pure-PPO baseline lives in `pfsp_population_ppo` /
`pfsp_checkpoints_swish_ppo`. Stage C uses the default names
(`pfsp_population` seeded from the PPO population / `pfsp_checkpoints_swish`).

---

## 4. Plan forward

**Refactor (behavior-preserving; committed before any ExIt-specific work):**
- **P0** — restore `train_pfsp.py` to its committed (pure PFSP + shaped) state;
  the inline Stage C logic moves to the hybrid script.
- **P1** — extract shared pure functions + the single `RETURN_SCALE` into
  `training_utils.py`; add `process_terminal_rewards`; relocate
  `analyze_strategic_decisions` out of the self-play trainer (removes the
  cross-import); create `config.py` with the unified `PFSPHyperparams`.
- **P2** — extract `play_population_game` + the PFSP driver into
  `pfsp_runtime.py`; make `train_pfsp` a thin wrapper. **Verify behavior
  unchanged** (short run / checkpoint diff). **→ Commit here:** the self-play and
  PFSP trainers work identically to today, but refactored for the future.

**ExIt-specific work (after the commit):**
- **P3 (done)** — created the hybrid ExIt entry point on the shared runtime
  (`reward_mode = terminal`, teacher attached, no controllers); committed
  `15229d7`. Stage C smoke + leaster tests re-run green.
- **P4 (done)** — pre-pick **determinizer extension** (`Game._sample_prepick_deal`):
  with no picker, no plays (no voids), and no called card, a pre-pick info set is
  an unconstrained partition of the unseen 26 cards into the 4 hidden 6-card hands
  + the 2-card blind (passing is always legal, so passers carry no information) —
  one shuffle, no rejection, `bury=[]`/`under=None`. `sample_determinization`
  dispatches to it when `not picker and not is_leaster`; the existing forced-replay
  (`_build_world`) already drained the recorded passes and stopped at the observer,
  so the **PICK head is now searchable**. PARTNER/BURY were already covered by the
  post-pick determinizer (a picker exists). Leasters are also determinizable
  (`_sample_leaster_deal`) and now searched — see §5. Validated Stage-A-style
  (`stage_c_bidding_search_check.py`): pre-pick redeals legal (full-deck partition,
  counts, observer hand preserved, empty bury/under), and `teacher.search` returns
  a valid `pi'` on all three bidding heads with no `picker == 0` failure. Bidding
  `SearchConfig.fracs` defaults flipped to 1.0 (cheap shallow roots), play 0.10.

**Cross-cutting items (do not skip):**
- **Throughput is a prerequisite for a from-scratch run** (~8 s/episode at the
  default search budget × millions of episodes is infeasible). Plan §5: batch
  determinizations through the encoder, consider async/offline target generation,
  and/or a smaller per-search budget. Tune before committing to a long run.
- **`t_full` critic-calibration probe** (`critic_calibration.py`): set the
  rollout-depth-schedule cutoff from measured per-trick critic R², not a guess.
- **Validation protocol, defined up front:** trick-0 defender trump-lead rate
  (the original leak); head-to-head vs the frozen PPO baseline; bidding-health
  bands (pick / ALONE / leaster rates) held *without* controllers; distillation
  diagnostics (teacher_kl, ESS-abort fraction, pg_masked_fraction, pi′ entropy);
  and the **PG-mask vs additive-form A/B**.
- **A committed regression test** for the shared `pfsp_runtime` + distillation /
  PG-mask path (the current smokes are one-off and uncommitted).
- **Decide the dormant epsilon plumbing's fate** in `ppo.py` (excise vs keep)
  during the maintainability pass.
- **Periodic ExIt-vs-PPO cross-eval** so the comparison is actually measured.

---

## 5. Known scope boundaries / future work

- **Leaster play-state search IS in scope** (`Game._sample_leaster_deal`, P4
  follow-up). It was briefly excluded, but with the per-trick reward + leaster
  bonus gone the pass->leaster branch the bidding EV rides on is *only*
  win-likelihood-driven if the agent plays leasters well — which needs a teacher
  signal on leaster play decisions. A leaster has no picker / called card /
  bury / under, so determinizing it is a void-aware partition of the unseen pool
  into the 4 hidden hands + the 2-card face-down blind (forced plays + voids;
  no importance weighting — there were no bidding choices to weight). The leaster
  guard in the play loop was removed accordingly.
- **Phase-2 (regress V toward the search root's backed-up value)** is deferred
  until the ExIt loop is validated — the AlphaZero-style virtuous loop and the
  first step toward dropping PPO entirely.
- The one-off validation scripts (`stage_*`, `gate0_*`, `*_determinizer_check`,
  `critic_*`) are **not committed**; `ismcts.py`, `ppo.py`, `sheepshead.py`'s
  determinizer, and the trainers/shared-libs are the production code.
