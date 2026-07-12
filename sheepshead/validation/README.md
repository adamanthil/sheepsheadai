# Validation & probe scripts (ISMCTS ExIt refactor)

One-off analysis / smoke-test / probe scripts written while building the
SO-ISMCTS soft-teacher ExIt pipeline (the determinizer, the search engine, the
training integration, the throughput work, and the validation protocol).
Committed for reproducibility — they are NOT part of the production code path
(the `sheepshead` package — game engine, agent, ismcts, pfsp_runtime,
config, training_utils — is). The committed unit/regression test is
`tests/test_ismcts_exit_regression.py`; these are the heavier,
slower, evidence-gathering scripts behind the design decisions.

## Running

Run from the **repo root** (module invocation; the scripts
import repo modules like `sheepshead` and also cross-import each other):

```bash
uv run python -m sheepshead.validation.t_full_probe --games 3000
```

**Model arg:** several scripts default `-m` to
`pfsp_checkpoints_swish/pfsp_swish_checkpoint_30000000.pt`, which has been moved.
Point `-m` at an available checkpoint, e.g. `final_pfsp_swish_ppo.pt` (repo root)
or `runs/reference_pfsp_ppo/pfsp_checkpoints_swish/pfsp_swish_checkpoint_9995000.pt`.
The Stage C scripts already default to `final_pfsp_swish_ppo.pt`.

Findings for Gate 0 / Stage A / Stage B reflect the conclusions recorded in
`notebooks/ISMCTS_Teacher_Refactor_Plan.md` and the Stage B project notes; the Stage C,
throughput, t_full and validation findings were measured in the May 2026 session
that built them.

---

## Origin — the leak

**`counterfactual_trump_leads.py`** — Why the project exists. Measures the
counterfactual EV of a *defender* leading trump vs fail on the FIRST trick via
paired rollouts on the TRUE deal. Finding: the 30M PPO agent leads trump as a
trick-0 defender in spots where leading fail scores better — a partial-observability
leak the critic cannot close (the trick-0 critic is blind). Also provides the
`best_in_class` / `play_out` / `rollout` helpers reused by Stage A.

## Gate 0 — the trick-0-only determinizer

**`gate0_determinizer.py`** — Validate that a trick-0 determinized teacher
(redeal the hidden cards consistent with the observer's info set, roll out under
the policy) reproduces the true-deal oracle EV on the leak states. Finding:
determinization **must be inference-weighted** — plain (uniform) determinization
is biased because it includes worlds where the redealt picker would never have
picked; weighting by the bidding likelihood (PICK + passes + call) lifted
effective sample size (~7/60 → ~17/40) and recovered the oracle's sign/direction.

**`gate0_inference_check.py`** — Focused companion (imports `gate0_determinizer`)
isolating the inference-weighting contribution. Finding: confirms the weighting
is what removes the EV bias.

## Critic capacity & calibration probes

**`critic_probe.py`** — Step 0: is the weak trick-0 critic a *head-capacity* problem
(a deeper value trunk fixes it) or a *feature* problem (the frozen encoder doesn't
represent the distinction)? Trains fresh SHALLOW vs DEEP heads on frozen 30M
features. Finding: R² rises as cards are revealed and trick-0 stays low for both
depths → it is **partial observability** (a hidden-information ceiling), not head
capacity. Motivated the search-based correction over a bigger critic.

**`t_full_probe.py`** — Sets the rollout-depth-schedule cutoff `t_full` on evidence.
Same machinery as `critic_probe` but the target is the **ExIt terminal return**
`get_score()/12` that the rollout bootstraps toward. Finding (3000 games): per-trick
R² (fresh deep head) all-play **0.26→0.82**, defender-leads **0.04→0.61**, leasters
**≤0.21**. → `t_full=1`/`d_short=2` validated (bootstraps land at trick≥4, R²≥0.73;
trick-0 leak states always roll to terminal); leasters forced to terminal rollout.
See `notebooks/Validation_Baseline_Notes.md`.

## Stage A — generalized inference-weighted determinizer

**`stage_a_determinizer_check.py`** — Generalize Gate 0 to ANY decision point
(`Game.sample_determinization`). Asserts legality (full-deck partition, per-seat
counts, play-revealed voids, called-ace placement, bury rules) AND that forced
replay reproduces the recorded public history exactly, then checks the
inference-corrected EV vs the true-deal oracle at tricks 1–3. Finding: legality
PASS; weighted determinized EV is **aggregate-unbiased but noisy** (single-deal
oracle is high-variance, mid-game ESS is low). Established scheme-B (bidding-only
importance weighting; plays as hard void constraints).

## Stage B — SO-ISMCTS engine

**`stage_b_ismcts_check.py`** — Validate the search engine (`ismcts.py`) on the
leak states. Findings (Stage B notes): use **SIR** — sample worlds from the
belief pool rather than weighting tree visit counts (keeps the visit budget from
collapsing to the mid-game ESS); **FPU + min-max-Q normalization** are needed for
calibrated PUCT at the score/12 value scale; and **deep rollouts early-game**
recover the Gate-0 leak direction (dp ≈ −0.17, ~75%) where a shallow critic
bootstrap is weak (dp ≈ −0.06, ~62%) — the basis for the trick-indexed depth
schedule.

## Stage C — training-loop integration

**`stage_c_smoke.py`** — End-to-end Stage C path: `play_population_game`
(terminal reward + ISMCTS teacher) → `store_episode_events` → `update()`. Finding:
distillation + PG-mask fire on searched PLAY transitions (e.g. 44/51 searched,
pg_masked_fraction ~0.86 at high play frac).

**`stage_c_leaster_test.py`** — Leaster determinization + search (no picker).
Finding: leaster redeals legal (96/96 — partition, counts, voids, empty bury/under,
blind never played), `teacher.search` returns a valid pi' with ESS≥floor on
in-leaster play nodes.

**`stage_c_bidding_search_check.py`** — P4 bidding-head search. Finding: pre-pick
redeals legal (96/96), and `teacher.search` returns a valid pi' on pick / partner /
bury with no `picker==0` failure.

**`stage_c_driver_bidding_check.py`** — Driver wiring: `play_population_game` with
bidding fracs=1.0 lands search targets on every head (incl. leasters), distillation
+ PG-mask fire. Finding: PASS.

**`stage_c_batched_pool_check.py`** — Tier 1 guard: batched pool build vs the
sequential `_build_world` reference on identical deals. Finding: identical world
states; log_w / per-seat memory match to ~3e-5 (batch vs batch-1 matmul); **~16×**
faster pool build.

## Throughput

**`profile_throughput.py`** — cProfile of `[A]` pure game-play and `[B]` ISMCTS
search. Finding: search is **~95% transformer encoder at batch size 1** (~14
s/search pre-optimization), Game logic <5%; in a play search the rollout is only
~16% of encodes (tree descent + opponent advance + per-trick observes are ~84%).
Drove the Game hot-path cleanup (~2.9× game-play) and the Tier 1/2 batching
(~9–10× on search overall).

## Validation harness

**`exit_validation.py`** — The model-evaluable half of the validation protocol,
to compare a from-scratch ExIt agent vs the frozen PPO baseline: bidding-health
bands (PICK / ALONE / leaster rates, held WITHOUT epsilon controllers), trick-0
defender trump-lead rate (the leak), and head-to-head mean score. PPO baseline
recorded: PICK 32.9%, ALONE 6.6%, leaster 9.2%, trick-0 trump-lead **4.8%**;
self-h2h ≈0 (unbiased-harness sanity). Real use:
`exit_validation.py -m <exit>.pt -b final_pfsp_swish_ppo.pt`. See
`notebooks/Validation_Baseline_Notes.md`. NOTE: do NOT route h2h through
`play_population_game` (full training game ~2–3 s/game); this uses a bare
mixed-agent loop (~150 ms/game self-play, which cannot batch).

## Teacher audit (June 2026, post run-2 collapse)

**`teacher_trump_lead_audit.py`** — The Arm-B falsifier: at trick-0 defender-lead
nodes (trump+fail in hand) on the PRISTINE 30M model, compare the policy prior's
trump-lead mass vs the production teacher's pi' mass, plus root-Q (best trump vs
best fail) and the same visit counts re-sharpened at tau ∈ {1.0, 0.5, 0.25}.
Finding (n=150, two replications): prior mass ~0.005-0.013 → **pi' 0.085-0.095
(+26 to +28 SE paired)** — yet argmax(pi') leads trump 0-1.3% (= the prior's
argmax) and the root-Q gap is mildly CORRECT (best-fail > best-trump by ~0.025,
though inverted at ~35% of nodes). So the injected mass is the exploration
machinery (FPU=1.0 + root_explore_frac=0.25 + tau_target=1.0), which Q is too
weak/noisy to starve; forward-KL distillation then teaches the floor — the
mechanism behind the run-2 leak regression (4.8% → 48.6%). Re-sharpening the
same counts: tau=0.5 → 1.2% mass, tau=0.25 → 0.03% (near-argmax; risky at root
ESS ~18). Implications: (1) cheap fix — distill at a separate target
temperature ~0.5; (2) structural fix — population-grounded rollouts so the
punishment for information-revealing leads is real; (3) pure self-play search
is contraindicated (it amplifies the self-model blind spot the Q noise comes
from).

**Harness fix (2026-06-10):** `selfplay_metrics` previously called
`get_action_probs_with_logits` (which advances the recurrent memory) before
`act()` at trick-0 audit nodes, so the greedy action there was taken from a
double-encoded memory. Greedy trump-lead RATES measured before the fix
(baseline 4.8%, run-2 48.6%, Arm A 11.3%) carry that perturbation; the mass
metric and h2h were always clean. Re-measured post-fix (n=300 games): baseline
1.4% / mass 1.3%; Arm A@50k 2.3% / 2.1%; Arm B@30k 74.3% / 51.6% — now
consistent with the trainer's greedy_health_probe (single-encode,
deployment-faithful).

## Run-1 collapse post-mortem diagnostics (May 2026)

**`forced_pick_check.py`** — Diagnostic A: forced-pick EV vs a fixed field.
Forces the agent under test to be picker every deal (others forced to PASS), so
picker-play EV is measured with zero bidding-policy influence; paired vs the
frozen 30M reference on identical deals/seed. Findings on the run-1 collapse:
collapsed ckpt 30045000 delta −0.85±0.39 on pickable hands (str≥7) ⇒ play
degraded too, not just bidding. Trajectory sweep across ckpts 5k–45k: play
delta is WORST at 5k (−2.73, −6.2 SE) and self-heals to ~−0.85 ⇒ an immediate
warm-start ONSET SHOCK (not disuse atrophy), and the bidding head is the only
non-recovering failure.

**`teacher_pick_audit.py`** — Diagnostic B: teacher PICK-target calibration by
hand strength (pi'(PICK) and Q(PICK)−Q(PASS) at PICK/PASS roots, walking the
pick phase forcing PASS). Findings: the REFERENCE (pristine 30M) teacher is
well-calibrated — monotone pick gradient, Q-gap crosses 0 at strength ~7-8,
strong hands 75-78% pick. The COLLAPSED teacher lost it — whole curve shifted
down ~30-40pp pi' and ~0.2-0.3 Q; strong hands flipped to 36%/−0.13 ⇒ the
collapse is a SELF-REINFORCING ratchet (policy drifts → self-play rollouts
weaken → teacher follows), not a teacher that was wrong from episode 0. This
is the result that motivated the bidding KL-anchor and, together with
teacher_trump_lead_audit.py, the population-grounded-rollout decision.
