# ISMCTS ExIt — Throughput Profiling Notes (NOT committed)

> **Note (2026-07-11):** file paths in this notebook predate the 2026-07 repo reorganization (core modules now live in `sheepshead/`, the hosted product under `app/`). Kept as-is for the historical record.

Reference notes from the May 2026 throughput pass. All numbers are **CPU,
single process**, on the dev Mac (Darwin arm64). The real training box may be
CUDA — batching helps there too, but absolute times will differ.

## Bottleneck (what profiling found)

ISMCTS search was **~95% transformer encoder, called at batch size 1**
(~6300 single-state encodes/search, ~14 s/search before optimization). Game
logic was <5%. Surprises that flipped initial assumptions:

- In a **play** search, the **rollout is only ~16%** of encodes; the **tree
  descent + opponent advance + per-trick 5-seat observes are ~84%** (with
  `max_depth=6` the leaf is reached deep, so rollouts are short).
- The recurrence (GRU memory) is **not** the binding constraint. It forces a
  sequential dependency *along one trajectory*, but the slowness was running
  many *independent* trajectories serially at batch-1. Memory rides in the
  batch dimension (per-(sim,seat) `(B, 256)`); the GRU applies per-row.
- **MPS is slower than CPU** here: encode_batch bs=1 was 17.6 ms on MPS vs
  2.34 ms on CPU (kernel-launch overhead), and even bs=64 MPS (310 µs/state)
  > CPU bs=64 (127 µs/state). Ruled out as a lever on this machine.

### encode_batch microbenchmark (CPU, per-state cost)
| batch size | ms/call | µs/state |
|---|---|---|
| 1  | 2.34 | 2340 |
| 8  | 3.38 | 422 |
| 64 | 8.14 | 127 |

→ batching is ~18×/state at bs=64. This is the whole game.

## Optimizations landed

| Change | Commit | Win |
|---|---|---|
| Game hot paths: removed dead per-action `get_state_dict` in `Player.act()` (write-only `start_states`/`actions`); `get_card_suit`/`get_card_points` → O(1) dicts | `dfc481f` | pure game-play 0.142s→0.049s / 300 games (~2.9×) |
| Tier 1: batched pool build (`_build_worlds_batched`) — all M determinized worlds replay the identical public sequence in lockstep, batched encoder/actor | `7a991af` | ~16× on pool build (≈20% of search) |
| Regression test (Tier-2 guard) | `3b73a90` | — |
| Tier 2: leaf-parallel batched search (`_run_batched`/`_Sim`) — batch_size sims concurrent, every encode/actor/critic + trick-observe batched, virtual loss to diversify | `be32c5b` | **7.6× on play search** |

### Search timing (final_pfsp_swish_ppo.pt, play head, 96 iters)
- Sequential (B=1): **11157 ms/search**
- Batched (B=32): **1459 ms/search**  → 7.6×
- Combined w/ Tier 1: full search **~14 s → ~1.5 s/search (~9–10×)**
- Target fidelity: batched-vs-sequential `pi'` **TVD mean 0.040**, argmax agree
  5/5 — virtual loss barely perturbs the distillation target.

## Per-episode time in `train_pfsp_exit.py` (production settings)

Measured ~**2.0 s/episode** (CPU, single process) over 24 games: trained model,
default `ISMCTSConfig` (`batch_size=32`, iters 48/64/96/96), production fracs
(pick/partner/bury = 1.0, play = 0.10). `update()` amortizes to <0.1 s/ep.

Searched decisions per game (training agent is one of five seats):

| head | searches/game | tree depth | cost note |
|---|---|---|---|
| pick | 0.50 | shallow (`max_depth=1`) | but `d_rollout = 6−trick` rolls to terminal at trick 0 → ~full-game rollouts × 48 iters (NOT cheap) |
| partner | 0.12 | shallow | same trick-0 full rollout |
| bury | 0.25 | shallow | same |
| play | 0.42 | deep (`max_depth=6`) | ~1.46 s each |

≈0.6 s/game from play searches; most of the rest is bidding searches, whose
**hidden expense is the trick-0 full-depth rollout** (every bidding decision
pays it at `f=1.0`).

### Implications for a from-scratch run
- ~4× better than the pre-opt "~8 s/ep" note.
- Single-process CPU at 2 s/ep ≈ 23 days per **1M** episodes. ExIt should need
  far fewer episodes than the 30M PPO baseline (denser search target), but tens
  of millions on one CPU core is infeasible.

### Levers (rough bang-for-buck order)
1. **Parallelize game generation** across CPU cores (embarrassingly parallel,
   ~linear with cores). Biggest lever for a real run. **DONE — see below.**
2. **Trim the trick-0 bidding-rollout cost**: lower `t_full` (bootstrap sooner)
   or `f_pick`/`f_bury` below 1.0. The `t_full` critic-calibration probe sets
   this on evidence.
3. Larger `batch_size` / CUDA.
4. Fewer iters per search.

## Lever 1 — parallel game generation (DONE)

Synchronous learner + self-play worker pool in `run_pfsp_training`
(`pfsp_runtime.py`), opt-in via `--num-workers` (`PFSPHyperparams.num_workers`;
auto-defaults to `min(cpu_count-1, 8)` for ExIt/terminal, `1` for the shaped
baseline). `num_workers <= 1` runs the original in-process sequential loop unchanged.

Design: the learner owns the authoritative training agent + population + optimizer
and does the single gradient update; a pool of `spawn` workers (each
`torch.set_num_threads(1)`) generates games with **frozen, versioned weights**
(published to per-version files after each update; workers reload on a version bump)
and returns plain-data `GameResult`s. Opponents are sampled in the learner
(authoritative ratings/diversity), resolved by `agent_id` from a worker-local lazy
population cache. Opponent strategic profiling is **captured** in the worker
(`compute_action_profile_events` / `compute_trick_profile_samples` in `pfsp.py`) and
**replayed** onto the authoritative population by the learner — batch-lagged (applied
at result time, not in-game), exact otherwise. Windows are sized to the remaining
transition budget so updates fire near a window boundary (strictly-on-policy modulo a
small, bounded straddle the user accepted as "batch lag").

### Measured (CPU, dev Mac, production ExIt search config, fresh agent)
| workers | s/episode | note |
|---|---|---|
| 1 | 2.18 | matches the pre-Lever-1 ~2.0 s/ep baseline |
| 6 | 0.56 | **3.9× speedup** over 24 episodes |

The 3.9× (not 6×) is **pool-spawn-overhead-dominated at this tiny scale** (~10 s one-
time spawn of 6 torch-importing workers, amortized over only 24 games). On a real
multi-million-episode run the spawn cost is negligible and scaling approaches linear
in cores (bounded by core count / memory bandwidth). Reproduce:
`validation/parallel_selfplay_check.py --throughput`.

## How to reproduce
- `profile_throughput.py` (one-off, uncommitted): `[A]` pure-game cProfile,
  `[B]` ISMCTS search cProfile + ms/search.
- `stage_c_batched_pool_check.py`: batched-vs-sequential pool equivalence + speed.
- Per-episode timing: ad-hoc `play_population_game` loop with `ISMCTSConfig()` +
  `SearchConfig()` (see chat history / `project_throughput` memory).
