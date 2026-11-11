# Sheepshead AI 2.0 Plan (November 2026)

**Status (2026-11-09): PLANNING.** The design below is decided except where
marked. Of the 2.0 changes, only the observation contract in §1 is built. The
code is on branch `pick-seat-residual` (worktree `.claude/worktrees/pick-seat`),
unmerged and not pushed.

## Scope of 2.0

`sheepsheadai-2.0` is a full retrain from scratch on the architecture
`perceiver-recall-v2`, about 6 weeks of wall time. It combines four changes to
the model and the training recipe:

| # | Change | 1.0 today | 2.0 | Kind | Status | Detail |
|---|---|---|---|---|---|---|
| 1 | **Seat at the pick** | `leader_rel` is 0 before play, so the agent cannot see its position relative to the dealer | The seat reaches the trunk and the oracle from the deal on | Observation | Contract built | §1 |
| 2 | **History oracle** | Recurrent oracle that must learn to recall who played what | Feed-forward oracle with one token per card and its location, about 6× faster to fit | Architecture | Designed; not built | §2 |
| 3 | **Separate bury head** | BURY and UNDER share the play pointer, so they stop improving after the league | Their own pointer and bilinear term, trained in the bidding PPO phase | Architecture | Designed; not built | §3 |
| 4 | **Leaster search** | Leaster play is held to θ_k by retention and never improves in policy iteration | Leasters are searched like any other play | Training recipe | Designed; not built | §4 |

Everything else about the trunk and program is unchanged: the `perceiver-recall`
encoder and play pointer, and the bootstrap → oracle → league → policy iteration
phases.

| Program phase | What changes |
|---|---|
| Bootstrap | Seat is visible from episode 1; the bury head learns on its own |
| Oracle pretraining | Fits the history oracle (and a recurrent control for the gate) |
| League | History oracle is the GAE baseline; online gate over the first 200k episodes |
| Policy iteration, corpus and distill | Leasters are searched; retention covers bidding heads only |
| Policy iteration, bidding PPO | Bury trains alongside pick and partner |

Release version and architecture name are independent, as they are for 1.0, so
the architecture name carries no release number.

**Not in 2.0.**

- Blitz/crack/recrack: a separate future run
  (`Blitz_Crack_Recrack_Plan_202607.md`).
- An aux head locating the hidden partner in another seat: not indicated by the
  probe in the appendix.
- Terminal-only bootstrap, uniform-window population, searched bury nodes and
  belief-based search: tracked in `Training_Program_Redesign_202609.md` and
  `Arm_D_Review_202610.md`.
- Possible improvements, left out to keep the recipe simple: referee-corrected
  search labels (`Arm_D_Review_202610.md` §9.12 option 2) and weighting search
  budget toward called games (§9.20).

## Releases and sequence

| Release | Architecture | How it is produced | Purpose |
|---|---|---|---|
| `sheepsheadai-1.0-202610` (deployed) | `perceiver-recall` | `202609_recall_rc` program, θ₆ | Current model; seat-blind at PICK/PASS |
| `sheepsheadai-1.1` (candidate) | `perceiver-recall-pickseat` | Zero-init residual surgery on θ₆, then PPO (§6) | Seat-aware pick/pass analysis tool, without a retrain |
| `sheepsheadai-2.0` (planned) | `perceiver-recall-v2` | Full retrain from scratch | The four changes above |

1. Finish run 2, score the tool's seat accuracy, certify against 1.0 and run 1,
   and ship it as 1.1 unless it is clearly worse (§6).
2. Build the history oracle, bury head and leaster search, pass the pre-launch
   gates, and run the proposed acting-mode twin (§5).
3. Launch the `perceiver-recall-v2` retrain. The offline and online oracle gates
   run inside it, after bootstrap.

## 1. Seat at the pick

### Why

The seat relative to the dealer matters in human play, and `leader_rel` was
intended to carry it. Every model since `a803e3e` (2025-10-03) has been
seat-blind at PICK/PASS: that commit dropped the player's position and
`last_passed` from the observation and introduced `leader_rel` already gated to 0
before play. `picker_position` does not fill the gap, because it is 0 until
someone picks.

Measured on the deployed model (53,739 forced PICK/PASS pairs, argmax play,
earlier seats rejection-sampled to pass), the PICK − PASS score difference for
near-threshold hands (log-odds −1..1):

| Mode | Seat 1 | Seat 2 | Seat 3 | Seat 4 | Seat 5 |
|---|---|---|---|---|---|
| Called ace | −0.71 | −0.62 | −0.30 | +0.20 | +0.84 |
| JD | −1.17 | −1.00 | −0.47 | +0.87 | +2.00 |

Seat 5 − seat 1 is +2.0 to +3.1 points in every strength band. A crude cross-fit
rule that flips decisions by mode, band and seat gained about +0.01 points per
hand. That figure has no proper error bar and is likely a lower bound, but it is
the only estimate so far of what seat is worth at the pick.

### Observation contract (built, `5ee887a`)

`get_state_dict` is the current observation and reports `leader_rel` before play.
Architectures trained on something older declare what they need changed through
`ArchitectureSpec.legacy_observation`, applied in `observation_for` and
`oracle_observation_for`. The latest agent has no adapter.

| Contract | Picker memory | Actor sees pre-play leader | Oracle sees pre-play leader |
|---|---|---|---|
| `LEGACY_PICKER_MEMORY` | yes | no | no |
| `LEGACY_SEATLESS_RECALL` (includes θ₆) | no | no | no |
| `LEGACY_PICK_SEAT_SURGERY` | no | yes | no |
| Current (`None`) | no | yes | yes |

Every golden comparison stayed bit-identical through this change. A registry test
allowlists the architectures that may read the current observation, because a
bypassed gate cannot fail loudly.

### In the 2.0 architecture

`perceiver-recall-v2` uses the plain `perceiver-recall` builders under the current
observation, with no residual and no masking. It is registered in `5ee887a` as
`perceiver-recall-seated`, to be renamed when the oracle is built; no checkpoints
exist under that name. Because seat is available while the pick head still has
high entropy, the entropy limit seen in the 1.1 residual (§6) should not apply.
How much a retrain gains from seat alone is unmeasured.

## 2. History oracle

The oracle's GRU exists only to recall who played what, and in which order. With
every hand and that history given explicitly, a feed-forward network can represent
U(h, s) with no recall error.

Evidence is in `Oracle_History_Probe_202610.md`: about 6× the recurrent oracle's
training throughput at 57% of its parameters, and offline at least as good as a
fresh recurrent oracle (weak O3, one recurrent seed). The gates in §5 can still
reverse this decision.

**D1. Input layout.** 33 tokens: context plus one per deck card (357k parameters
at the probe's size). Each card token is
`MLP(card, location, seat, role, trick, order, under)`, followed by the same
transformer (4 layers, 4 heads, d_token 64), readout, value trunk and heads as the
production oracle.

| Location | Seat | Trick / order |
|---|---|---|
| hand | holder | none |
| blind (stays here all game in a leaster) | none | none |
| bury | picker | none |
| held under (declared, not yet played) | picker | none |
| current trick | player | trick, position from the leader |
| played in an earlier trick | player | trick, position from the leader |

Seats are relative to the acting player. `role` carries the picker and
secret-partner bits. `under` marks the called-under card wherever it is, including
after it is played face-down.

**D2. Context token.** The production oracle's fields, unchanged: the header
scalars, `points_taken_rel`, `secret_partner_rel`, and the called and under card
embeddings. Seat arrives through `leader_rel`. The pass order needs no input of
its own, since seat implies who has already passed.

**D3. Source of the table.** `Player.get_card_locations()` returns a (32, 5) uint8
table (location, seat, trick, order, under), computed at decision time from `hand`,
`game.blind`, `game.bury`, `game.under_card`, `game.history` and `game.leaders`,
with the precedence blind, bury, held under, hands, played.
`get_oracle_state_dict` adds it as `card_locations` for every caller; the recurrent
oracle ignores the key. A determinized ISMCTS world is a real `Game` replayed to
the root, and worlds whose `history` differs from the real game are dropped
(`ismcts.py:1332`), so worlds need no special handling.

**D4. Network.** `HistoryOracleEncoder` and `HistoryOracleValueNetwork` live in
`agent/oracle.py`. The encoder is built directly from the shared pieces (card
embedding with informed init, seat and role embeddings, context MLP,
`TransformerCardReasoning`), not by subclassing and deleting modules as the probe
does. The network keeps the existing interface: `forward_sequences`,
`forward_sequences_full`, `aux_losses`, `has_aux_heads`, `param_groups`. It
flattens non-empty rows, runs them feed-forward, and re-pads to (B, T). The probe
then imports the production classes.

**D5. Aux heads.** Keep both validated oracle heads (per-seat team membership and
team points with bury), at the same coefficients. `team_aux_labels` skips empty
rows.

**D6. Selection.** `ArchitectureSpec.build_oracle`, defaulting to the recurrent
network, replaces the direct construction at `ppo.py:293` and
`pretrain_oracle.py:293`; `pretrain_oracle` gains `--arch`. The architecture name
determines the layout, with no new checkpoint field. `perceiver-recall-v2` uses
the history oracle, and `perceiver-recall-v2-gru-oracle` is registered as the
control arm for the gates.

**D7. Checkpoint safety.** The two oracle loads use `strict=False` so headed and
headless checkpoints interchange, which would also accept a mismatched layout with
random weights. One helper replaces them, tolerating missing or unexpected keys
only under the aux heads (`team_membership`, `team_points`) and raising on anything
else.

**D8. Recurrent-only plumbing.** The network declares `uses_event_frames = False`.
Collection then stores `oracle_state=None` on trick-observation events, at
`pfsp_runtime.py`, `distill_corpus.py`, `traced_game.py`, the two research-probe
sites and `_OracleCapture.record_trick`. Decision rows keep their event index, under
one rule for the placeholders:

- `store_episode_events` keeps the field whenever it was provided
  (`if "oracle_state" in ev`). Dropping a `None` would remove the key, and consumers
  would then fail or skip. Limited-mode events never carry the key, so they are
  unchanged.
- Oracle states are required only on action events, at `_fill_oracle_values`
  (`ppo.py:930`) and the distill oracle step (`policy_iteration.py:895`). The network,
  `_build_oracle_minibatch`, `_oracle_extra_epochs` and `pretrain_oracle.py:148` accept
  `None` on observation events.
- A test confirms that a distill epoch changes the history oracle's weights. The check
  at `policy_iteration.py:895` skips oracle training without an error when a state is
  missing, so nothing else would catch this.

In ISMCTS, `_OracleCapture` records nothing, so `sim.oracle_seq + [state]` is just the leaf state and the per-world
prefix stream goes away. Recurrent architectures keep this path unchanged.

**D9. Start of training.** No distillation. The probe's distillation step was to
avoid discarding a 7.7M-episode critic in a running league, and 2.0 has none to
protect. It uses the program's existing oracle pretraining, `pretrain_oracle` on
the bootstrap policy's 40,000 episodes, passed as `--oracle-init`.

The speedup applies to oracle fitting and not to rollouts, so the wall-clock gain
on the whole program is unmeasured.

## 3. Separate bury head

**Today.** BURY and UNDER are scored by the play pointer. The bidding PPO phase
freezes the pointer to keep it disjoint from the play operator, and the corpus does
not search bury, so bury is fixed after the league apart from retention. The stake
is unmeasured at handoff; the generation-1 bury-only split was +0.003 ± 0.007.

**Design: a second pointer and bilinear term over hand tokens, for the bury phase
only.**

- **B1. Scope.** The head scores both BURY and UNDER, the two families the bidding
  group already treats as the bury head. `PLAY UNDER` stays with play.
- **B2. Code.** Extract the pointer into a `HandPointer` module used by the new
  head, `bury_pointer`, behind an actor flag. Legacy parameter names stay as they
  are. About 41k new parameters.
- **B3. Inputs.** The same `feat` and post-reasoning hand tokens as play, so bury
  shares the trunk's representation and only the scorer is separate.
- **B4. Training.** Ordinary PG in bootstrap and league. In the bidding phase it
  trains automatically, since `bury_pointer` does not match `PLAY_HEAD_PREFIXES`,
  and `BiddingHyperparams` already sets its 0.04 entropy for this case. Bury's
  normalized entropy is about 0.19, against pick's 0.06, so PG has signal. In
  distill it behaves like pick and partner: retention KL during trunk epochs, frozen
  during head epochs (the name filter matches only `pointer_U` and `pointer_V`).
- **B5. Composability.** The two improvement operators still touch disjoint
  parameters, which is the reason the pointer is frozen today.

**Alternatives rejected.** Unfreezing the shared pointer moves play during the
bidding phase. Conditioning one pointer on the phase still shares parameters.
Scoring bury pairs jointly adds outputs for a decision the sequential pointer
already expresses. Searching bury nodes is the noisiest option, as the decision
farthest from terminal, and can be added later on this head with no architecture
change.

## 4. Leaster search

**Today.** The corpus searches play nodes only when `not game.is_leaster`
(`distill_corpus.py:391`). Leaster plays go to the retention set, so distill holds
them to θ_k and nothing rewards improvement.

**Evidence.**

- *Headroom.* The θ₆ search-ceiling run already searched leasters. It deviated from
  the policy at 47.6% of 754 leaster nodes against 31.4% elsewhere, and leaster deals
  scored +0.439 ± 0.131 for the searching hero against +0.059 ± 0.024 on the rest
  (41 and 459 deals). Leasters were 8% of deals and about 40% of θ₆'s remaining
  ceiling. The ceiling is a best response to the current agent, so this bounds the
  gain rather than predicting it, and 41 deals is few.
- *Drift.* The per-iteration leaster-hand edges against θ_k were +0.0185, −0.0126,
  +0.0151, +0.0003, −0.0219, −0.0062, summing to −0.007 ± 0.020 per leaster hand: a
  random walk, not a measured decline. The divergence is real, though. The share of
  leaster deals that changed rose from 7.3% to 10.7% between iterations 2 and 5 while
  standard play's fell (`Arm_D_Review_202610.md` §9.18).

**Design.**

- **L1. Recipe.** Remove the leaster exclusion, and give leasters the same
  lead-feature branch as alone hands. Retention becomes bidding heads only. Every
  unforced play is searched at the recipe's p = 1.0, so no leaster row loses its
  anchor without gaining a target.
- **L2. One search, one value estimate.** Leasters use the same search algorithm,
  budget (256 iterations, t0 leads 1024, t1 leads 512, 3 replicates) and oracle at
  the leaves as every other node. There is no leaster-specific routing.
- **L3. Determinization.** Reused. `_sample_leaster_deal` handles voids and forced
  plays, and the forced replay weights each PASS, so "everyone passed" is evidence
  the world pool already uses. Leaster worlds are covered by the replay-equivalence
  tests.
- **L4. Node classes.** `play_cell` labels about 600 leaster nodes per corpus as
  "partner". Report every leaster seat under one role, so the class-variance fit does
  not get thin spurious cells.
- **L5. Cost.** The iteration-6 corpus had 22,860 leaster nodes against 217,140 std
  and alone nodes. Assuming the same unforced share, that is about +10% searched
  nodes: about +5 hours per 46-hour corpus and about +30 hours over six iterations.

## 5. Build order and gates

**Build order.** Rename the architecture; card-location table and its tests (D3);
oracle network (D4–D5); registry and load guard (D6–D7); plumbing (D8); bury head
(§3); leaster search (§4); gate tooling and final references (below). Build in a separate worktree from the head of
`pick-seat-residual` while run 2 imports from the current one.

**Before launch.**

| Gate | Test | Pass |
|---|---|---|
| Table correctness | Full games, both modes, including called under, alone and leaster | All 32 cards located at every decision; bit-equal to the probe's `build_card_locations` replay |
| No regression | `check_goldens` (the arch goldens exclude the oracle; the search goldens exercise the recurrent oracle through its leaves), plus θ₆'s oracle values over 400 full games | Every existing comparison bit-identical; θ₆'s oracle values differ by exactly 0 |
| Distill trains the oracle | One distill epoch on the history oracle (D8) | Oracle weights change |
| References present | Every path in `FinalConfig.references` at launch | All exist; a missing file is otherwise skipped with only a log line, at the end of the run |

**Proposed pre-work: acting-mode twin (not yet run or scheduled).** Should the corpus
always sample the policy at continuation, rather than play the committee's argmax on
resolved rows? Sampling makes the corpus plain DAgger and removes the acting branch from
`distill_corpus.py`. The case for committee acting is one comparison on an older recipe
(+0.0055 ± 0.0021 play-only, `CE_Teacher_Design_202608.md` §20.13 addendum 16; a
full-agent read of the same pair was +0.0008 ± 0.0035), and neither arm improved on its
parent there.

| | |
|---|---|
| Baseline | 1.0's iteration 1: `runs/202609_recall_rc/pi/iter1`, 8,000 committee-acted games from θ₀ (`league/checkpoints/checkpoint_8000000.pt`) |
| Twin | Same command and `--seed 20260903`, with `--committee-act-frac 0`, so games match until the first acted deviation; then the same fit, target, distill and cert |
| Read | Paired play-only, twin minus baseline, 16,000 deals/mode (seeds 42 and 43) |
| Decision | Lower bound above −0.005: 2.0 samples the policy and the acting branch is removed. Otherwise committee acting stays |
| Cost | About 2–3 days of machine time |

This is one step at the iteration with the most gain to lose. It does not show whether
more iterations would make up a loss; a second twin iteration would add about 5 days.

**Gate tooling (to build).** The oracle gates below need support the program does not
have: `stop_after` only accepts `iter{k}.{stage}`.

- Add pause points after oracle pretraining and at 200k episodes of league generation 1,
  so the history arm continues from there instead of restarting.
- Offline gate: a driver for the four fits on the run's own dataset, using the production
  classes and reporting the paired ΔMSE by stratum, adapted from
  `research/oracle_history_probe.py`.
- Online gate: run the `perceiver-recall-v2-gru-oracle` control arm from the same
  bootstrap checkpoint and seed. Whether the load path accepts that checkpoint under the
  control's architecture name is not yet checked.

**Final comparison.** `FinalConfig.references` becomes:

| Reference | Path | Why |
|---|---|---|
| 1.0 | `models/sheepsheadai-1.0-202610.pt` | The bar 2.0 has to beat |
| 1.1 | Its released file, expected `models/sheepsheadai-1.1-202611.pt` | The deployed model, once shipped |
| v2 release | `models/v2-release.pt` | Best agent from the earlier training pipeline |
| 30M | `models/pfsp-swish-30m.pt` | Long-running anchor |

iter11 P1 is dropped.

**Inside the run (thresholds pre-registered by the operator, 2026-11-09).**

| Gate | When | Test | Pass |
|---|---|---|---|
| Oracle, offline (about 5–6 hours) | Oracle pretraining, on 2.0's own bootstrap dataset | Fit a history and a recurrent oracle, seeds 42 and 43 (four fits), the probe's paired ΔMSE | Not worse by more than ΔEV 0.005 overall and in `pick`, `play_lead_t02_defender`, `play_t3plus` and `leaster` |
| Oracle, online (about 11 hours per arm) | First 200k league episodes of each arm, same bootstrap and seed | Compare the arms | History's `ev_oracle` within 0.01 of recurrent's over the last 100k; pick and play advantage σ no more than 2% higher |

The oracle choice applies to the whole model. Leasters are among the offline strata
because the history layout was worse than recurrent there in the October probe
(+0.002 MSE, z +3.5), and leaster search relies on those values. If the online gate
passes, the history arm's 200k episodes are the start of the run, so little time is
lost.

**Optional policy-iteration extension.** The aggregate stop rule stays as in 1.0, and
the iterate it stops on is the default release. After it fires, run up to two more
iterations as candidates. Adopt the later iterate only if called-suit over-trumping is
significantly lower and it is no more than 2 SE behind the stopped iterate on both
duplicate h2h and the PANEL-A ∪ PANEL-B composite. Use more than 1.0's 3,996 panel deals
(SE about 0.005) for this decision. Over-trumping was still falling 2–4 points per
iteration when 1.0 stopped, while the θ₃→θ₆ gain did not appear on the panels (h2h
+0.0068 ± 0.0036, composite −0.0016 ± 0.0057), so the extension must not be able to
replace the release without passing this check.

**Measured during the run, with no separate validation.**

- Leaster search: the leaster-hand cert edge after the first policy iteration, against
  the series in §4.
- Bury head: bury entropy and the picker's bury quality through bootstrap, because the
  head no longer inherits play's early gradients.

## 6. The 1.1 interim release: pick-seat residual

1.1 brings seat to the pick/pass analysis tool without a retrain. The tool's value
depends on reasoning about seat more than on absolute strength.

**Design.** Zero-initialized residuals fed the head's features, a seat embedding and
the partner mode (`architectures/pick_seat.py`), added to the PICK logit and the
limited critic's value, win and expected-return heads. The trunk keeps seeing
`leader_rel = 0` before play. The oracle is unchanged and seat-blind, which adds
variance to the GAE baseline at picks but no bias. Training freezes everything except
the actor's residual and the critics, so any gain is attributable to seat.

| | Run 1 (complete, fallback) | Run 2 (complete, ships as 1.1) |
|---|---|---|
| Episodes | 200,000 (about 10¼ hours) | 500,000 (about 24 hours at 5.2–5.4 eps/s) |
| Pick entropy coefficient | 0.08 fixed | 0.8 → 0.08, linear over 400k, then held for 100k (`53ad37d`) |
| Seat residuals | Pick, `V` | Pick, `V`, `Win %`, expected final return (`b9d5e4f`) |
| Start | `theta6_pickseat.pt` | `theta6_pickseat_v2.pt` |
| League copy | `league_snapshot` | `league_snapshot2` |
| Output | `runs/pick_seat_202611/run1/final.pt` | `runs/pick_seat_202611/run2/final.pt`, log `run2.log`, checkpoints every 50k |
| Certification vs 1.0 | PASS, +0.0018 ± 0.0015 | PASS, +0.0103 ± 0.0030 |

Both starting checkpoints reproduce θ₆ exactly: every shared tensor is identical,
including the oracle, with zero difference in action probabilities and values over 400
full games.

**Run 1 results.** The same 3,000 hands at each seat, earlier seats forced to pass
(their hands are not conditioned on having passed):

| | Seat 1 | Seat 2 | Seat 3 | Seat 4 | Seat 5 |
|---|---|---|---|---|---|
| θ₆ argmax PICK %, called / JD | 22.3 / 28.3 | same | same | same | same |
| Run 1 argmax PICK %, called / JD | 21.7 / 26.9 | 21.1 / 27.2 | 21.5 / 27.7 | 22.2 / 28.1 | 22.5 / 28.7 |
| Near-threshold log-odds (θ₆: −0.11) | −1.76 | −1.64 | −1.00 | −0.30 | 0.00 |
| Decisions flipped vs θ₆, called / JD (%) | 0.7 / 1.4 | 1.3 / 1.1 | 0.8 / 0.6 | 0.1 / 0.3 | 0.3 / 0.5 |

The direction matches §1, but few decisions moved. Only about 2% of hand-and-seat
combinations sit within ±1 log-odds of the threshold in θ₆. The pick head is nearly
deterministic, so PPO rarely samples the other action and gets little counterfactual
signal. That limit is what run 2's annealed entropy targets.

In the tool, on 1,500 hands: `V` varies by seat (mean +0.02, −0.18, −0.19, −0.03, +0.61
points; per-hand spread 0.87 points), and PICK probability moves more than 5 points
across seats on 7.9% of hands and more than 25 points on 2.9%. Run 1 has no seat input
to `Win %` or expected return, which run 2 adds.

Certification against 1.0 (`policy_iteration cert`, 8,000 deals/mode, routed reads
skipped because the play path is bit-identical; `runs/pick_seat_202611/cert/cert.json`):
edge **+0.0018 ± 0.0015** (called +0.0013 ± 0.0019, JD +0.0024 ± 0.0023, leaster hands
+0.0020 ± 0.0042), with 3.3% of deal-seats deviating. Non-inferior, and not
distinguishable from zero.

| Probe | Run 1 | θ₆ (same seeds) |
|---|---|---|
| PICK | 31.4% | 33.1% |
| Leaster | 10.3% | 8.4% |
| ALONE | 14.3% | 13.4% |
| Partner trump lead (bar 96.5) | 96.6% | 97.1% |
| t0 trump lead (bar 4.0) | 0.8% | 0.6% |
| Called-suit lead | 90.8% | 93.3% |

**Decision after run 2.** Early stop if pick entropy has not clearly risen by 150k
episodes. Otherwise:

1. Certify `run2/final.pt` against 1.0 with `policy_iteration cert` (8,000 deals/mode),
   and run a head-to-head against run 1's `final.pt`.
2. Score per-seat PICK log-odds and `V` against the forced PICK/PASS results of §1, with
   run 1 as the baseline, and check that `Win %` and expected return vary by seat in the
   tool.
3. Ship run 2 as 1.1 unless it is clearly worse: behind run 1 by more than 2 SE, or
   failing certification against 1.0.

**Run 2 results.** Run 2 ships as 1.1: it beats both 1.0 and run 1 by about 3.4 SE.

Training finished without incident. Greedy health over 200 games every 50k: PICK
28–36%, leaster 3.5–15.5%. The 15.5% at 150k fell back to 12.0%, then 5.0% (noise at
n = 200). The 500k check: PICK 33.6%, leaster 6.5%, t0 trump lead 0.0%, partner trump
lead 98.7%. Sampled pick rate held at 17–18% throughout.

Certification against 1.0 (`runs/pick_seat_202611/cert_run2/cert.json`, same settings
as run 1): edge **+0.0103 ± 0.0030** (called +0.0044, JD +0.0162; leaster hands
−0.0055 ± 0.0106), with 14.1% of deal-seats deviating, against 3.3% for run 1. Against
run 1 on the same instrument (`h2h_run2_vs_run1/cert.json`, bars not enforced): edge
**+0.0092 ± 0.0027** (called +0.0034, JD +0.0149).

| Probe (4 seeds) | Run 2 | Run 1 | θ₆ |
|---|---|---|---|
| PICK | 31.6% | 31.4% | 33.1% |
| Leaster | 6.6% | 10.3% | 8.4% |
| ALONE | 13.1% | 14.3% | 13.4% |
| Partner trump lead (bar 96.5) | 97.4% | 96.6% | 97.1% |
| t0 trump lead (bar 4.0) | 1.4% | 0.8% | 0.6% |
| Called-suit lead | 93.2% | 90.8% | 93.3% |

Pick decisions scored against §1's forced PICK/PASS outcomes (2,687 hands × 5 seats ×
4 reps, played out under θ₆, weighted back to natural frequency and to the chance that
earlier seats pass). Value of the pick decisions over always passing, ×1000 points per
pick decision: θ₆ +425.5, run 1 +434.5, run 2 +454.2. Run 2 − θ₆ **+28.7 ± 3.6**; run 2
− run 1 **+19.7 ± 3.5**.

| Seat | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| θ₆ PICK % (natural weights) | 27.0 | 30.5 | 36.0 | 44.5 | 61.9 |
| Run 2 PICK % | 22.1 | 25.3 | 32.7 | 54.1 | 75.6 |
| Measured PICK − PASS, θ₆ near-threshold hands (points) | −0.94 | −0.81 | −0.38 | +0.54 | +1.43 |
| Run 2 log-odds on those hands (run 1) | −3.84 (−1.64) | −3.32 (−1.55) | −1.73 (−0.92) | +2.86 (−0.22) | +5.08 (+0.08) |
| Decision value change vs θ₆ (×1000) | +39 | +29 | +7 | −12 | +108 |

The seat 5 gain comes from picking hands θ₆ passed into a leaster: from θ₆ log-odds −6
to −1, picking is worth +0.8 to +1.1 points there, and run 2 picks 98–100% of them.
Seat 4's −12 is almost all the weakest bin (−21 ± 21), so it is noise. The outcomes
were played with θ₆ in the earlier seats, which now pass more often, so this score
shows direction and calibration; the h2h is the deciding read.

In the tool (1,500 hands, earlier seats passing): `Win %` by seat 43.7, 39.3, 39.4,
41.9, 46.7 (per-hand spread 12.0 points, p90 18.1); expected return +0.04, −0.20,
−0.14, +0.03, +0.83 (spread 1.05); `V` +0.16, +0.01, −0.05, +0.10, +1.03 (spread 1.13,
run 1 0.87). PICK probability moves more than 5 points across seats on 29% of hands and
more than 25 points on 22% (run 1: 7.9% and 2.9%). In θ₆ and run 1, `Win %` and
expected return do not vary by seat.

**Release.** `runs/pick_seat_202611/run2/final.pt`, byte-identical, is
`sheepsheadai-1.1-202611.pt` (sha256 `f21ea6d9f13e`, 14,977,443 bytes, arch
`perceiver-recall-pickseat`). It is smaller than 1.0's file because the
optimizer state covers only the residual and critics. The manifest moves the
`deploy` role and 1.0's sampling (T 0.5 on pick, partner and play, bury 0, band
1.0) onto it. The temperature sweep was not rerun, because the play path is
unchanged. 1.0 stays pinned with the new `reference` role (`2c5d0d4`) for the
final comparison in §5. The app's pick analysis and game simulation run on the
file.

## Appendix: partner-seat probe (why no new aux head)

`Called_Suit_Schmear_202611.md` §7.2. Linear probe for "the next seat holds the called
card" on θ₆'s encoder output: 50,000 called-ace games, 31,487 positions, held out by
game, excluding positions where the next seat is the picker (the rules answer those).
The script is in session scratch only.

| Defenders left after the actor | Held-out n | True rate | Mean P, 2D outputs | Mean P, tokens |
|---|---|---|---|---|
| 0 (forced) | 2,169 | 1.000 | 0.991 | 0.995 |
| 1 | 4,219 | 0.519 | 0.505 | 0.503 |
| 2 | 1,418 | 0.344 | 0.328 | 0.320 |

At the 108 held-out schmear spots mean P is 0.987, and 1.000 at the 17 where θ₆'s top
choice is the A/10. By that notebook's criterion the trunk has the information and the
problem is the policy head or the training labels, so an aux head is not indicated. The
forced answer can also be read from the public count of defenders left, so this rules
out a missing input more than it shows the trunk reasons about the partner. The search
read at the WRONG nodes (§7.1 there) has not been run.

## Log

- **2026-11-09.** Seat audit and history. Residual built (`0859d0d`, `7466dfc`); run 1
  complete and certified. Observation contract committed (`5ee887a`). Partner-seat probe
  read. Run 2 launched (`b9d5e4f`, `53ad37d`). History oracle, bury head and leaster
  search designed; oracle gate thresholds pre-registered; architecture name set to
  `perceiver-recall-v2`. Notebook created.
- **2026-11-10.** Run 2 complete (500k). Certified against 1.0 at +0.0103 ± 0.0030 and
  beat run 1 by +0.0092 ± 0.0027; pick decisions score +0.029 ± 0.004 points per
  decision above θ₆ on the forced PICK/PASS hands. Ships as 1.1
  (`models/sheepsheadai-1.1-202611.pt`); manifest deploy entry moved to it.
