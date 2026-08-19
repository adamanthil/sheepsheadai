# ISMCTS ExIt — Throughput Profiling Notes

Measurement reference: how fast the search is, where its time goes, and how fast
each device and link is. Systems built on these numbers are documented
elsewhere — see **`Distributed_Inference_202608.md`** for the two-machine
inference split that addendum 2 below grew into.

> **Note (2026-07-11):** file paths in this notebook predate the 2026-07 repo reorganization (core modules now live in `sheepshead/`, the hosted product under `app/`). Kept as-is for the historical record. The original title claimed this file was uncommitted; it has been tracked in git since the reorg.

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
  *(Re-examined 2026-08-18 — see the addendum at the end of this file. The
  cause was not missing Metal kernels but host-synchronising control flow in
  our own actor, since fixed; the "ruled out on this machine" conclusion
  survives the fix, but for a different and better-quantified reason.)*

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

---

# Addendum (2026-08-18) — MPS re-examined for the CE search teacher

Revisits the one-line "**MPS is slower than CPU**" finding above (§Bottleneck),
prompted by a specific hypothesis: that MPS was slow because some op we use has
no Metal kernel and silently falls back to CPU, so every such op pays two device
transfers plus a pipeline stall. The question was whether switching
`train_league_ppo.py` to MPS is worth it **with the CE search teacher enabled**.

Answer up front: the fallback hypothesis is **refuted** — zero fallback ops. The
real cost was host-synchronising Python control flow in our own code, now fixed
(3 commits, all bit-exact). Even after the fix, **MPS is still not worth
switching to on this machine**: 8 M1 Max performance cores collectively
out-throughput the 32-core GPU on a model this small.

**Load caveat.** All of this was measured while an 8-worker
`league_ce_teacher11` generation was running. Ratios measured inside one process
(A/B of two implementations, or inference-vs-total within one search) are robust
to that; absolute wall-clock is not. Where a number is load-sensitive it is
flagged. CPU benchmarks use `torch.set_num_threads(1)` to match a league worker
— this matters enormously and an earlier default-threaded pass was badly
distorted by 10 torch threads fighting 8 busy workers.

## A1. Op-coverage audit — no CPU fallback anywhere

PyTorch registers an MPS backend fallback unconditionally but gates it at
runtime on `PYTORCH_ENABLE_MPS_FALLBACK`. Unset, a missing op **raises**; it
does not silently degrade. The repo never set it and never wired MPS at all
(`ppo.py` device is cuda-or-cpu).

Every `aten` op dispatched by three hot paths was traced through a
`TorchDispatchMode` and classified against the dispatcher table (native MPS
entry / backend-agnostic `Composite*` alias / CPU-only ⇒ fallback):

| path | distinct aten ops | dispatches | CPU-fallback ops |
|---|---|---|---|
| `act()` single-state rollout ×8 | 61 | 5,231 | **0** |
| batched inference B=64 | 49 | 1,913 | **0** |
| `update()` fwd+bwd+step | 118 | 63,915 | **0** |

Cross-checked empirically: all three paths run to completion on MPS with
`PYTORCH_ENABLE_MPS_FALLBACK` **unset**. `nn.MultiheadAttention`, `GRUCell`,
`Embedding`, `masked_fill`, `index_put_`, `scatter` are all covered. (A CPU-side
trace flags `_scaled_dot_product_flash_attention_for_cpu` as CPU-only; that is
an artefact of SDPA's device-dependent backend selection and does not appear in
the MPS trace.)

## A2. What was actually slow — host syncs in our own control flow

Attributed by stack frame, one `act()` call: **29** device-to-host round trips.

| count | site |
|---|---|
| 24 | `actors.py` `if valid_b/u/p.any():` — 8 hand slots × 3 action families |
| 2 | `ppo.py` `torch.distributions.Categorical` argument validation |
| 3 | `ppo.py` `action.item()`, `log_prob.item()`, `value.item()` |

`valid.any()` returns a 0-dim device tensor; a Python `if` on it calls
`__bool__`, which commits the command buffer and blocks. Marginal cost measured
at **~688 µs on MPS, ~0 on CPU** (−0.7 µs, i.e. noise) — which is exactly why
this was invisible before anyone tried an accelerator. The 24 in the actor are
**batch-size independent**, which is what pinned the batched forward to a flat
~52-60 ms floor from B=1 to B=256.

Three commits, each verified bitwise equal (probs, logits, and every gradient)
and each passing `capture_arch_goldens --check` 34/34 plus 570 tests:

| commit | change | effect |
|---|---|---|
| `0df8281` | per-slot masked loop → one sink-column `scatter` per family | 24 stalls → 0 |
| `0b0885e` | `act()` returns via one fused `torch.cat(...).tolist()` carrying a finiteness sentinel; `validate_args=False` | 5 stalls → 1 transfer |
| `03d151f` | five action-index maps → `register_buffer(..., persistent=False)` | −5 H2D copies per forward |

`persistent=False` is load-bearing: the actor loads with `strict=True`, so a
persistent buffer would add keys no existing checkpoint carries. Verified by a
bidirectional checkpoint round-trip across all 34 registered architectures.

Cumulative on MPS (load-sensitive absolutes, same-session before/after):

| path | before | after |
|---|---|---|
| `act()` single decision | 52.3 ms | **12.8 ms** |
| batched fwd B=1 | 52.3 | **9.9** |
| B=8 | 64.2 | **10.5** |
| B=64 | 63.9 | **10.9** |
| B=256 | 80.3 | **20.2** |

Host syncs per `act()` 29 → 0; per batched forward 24 → 0. On CPU all three are
neutral (the scatter A/B measured 0.91-1.15×), so the current CPU training path
gains nothing — the value is entirely on the accelerator path.

## A3. Where committee-search time goes

One **production-config** committee (1024 iters, R=3, `d_rollout=1`, live-run
v24 weights, single-threaded), instrumented at the encoder/actor/critic seam:

```
total wall            46.35 s     (§8 of the CE doc models ~62 CPU-s — same ballpark)
network inference     34.53 s     74.5%    2081 calls
everything else       11.81 s     25.5%    Python / game / tree
```

Inference time by batch size — this distribution is what decides the verdict:

| bucket | share of inference | note |
|---|---|---|
| B=2-8 | 0.2% | |
| B=9-32 | 0.5% | |
| **B=33-96** | **44.2%** | essentially all at B=96 = R × `batch_size` = 3 × 32 |
| B=97-256 | 5.1% | |
| **B>256** | **50.0%** | B=1024 pool build, then 480 / 320 / 160 |

## A4. CPU vs MPS at the batch sizes actually used

Single-threaded, `perceiver-shared-v2`, `encode_batch` only (96.5% of inference
time):

| B | CPU ms | MPS ms | µs/state CPU | µs/state MPS | MPS speedup |
|---|---|---|---|---|---|
| 1 | 1.18 | 16.24 | 1180 | 16236 | **0.07×** |
| 32 | 7.96 | 24.08 | 249 | 753 | 0.33× |
| **96** | 17.41 | 19.64 | 181 | 205 | **0.89×** |
| 160 | 32.95 | 30.22 | 206 | 189 | 1.09× |
| 320 | 68.04 | 38.48 | 213 | 120 | 1.77× |
| 480 | 111.16 | 46.06 | 232 | 96 | 2.41× |
| 1024 | 209.77 | 58.76 | 205 | 57 | 3.57× |

The May-2026 note above recorded CPU bs=1 2.34 ms / MPS bs=1 17.6 ms; this pass
gets 1.18 / 16.24. The MPS figures agreeing across 15 months and a torch upgrade
is a good sign the two passes are commensurable; CPU is faster now because
`perceiver-shared-v2` is a cheaper encoder than the `full` arch that note used.

**The crossover is at B≈160, and the dominant batch in the current search is
B=96 — below it.**

## A5. Throughput estimate for `train_league_ppo` + teacher

Time-weighting §A4 by the §A3 distribution: inference **1.34×**, hence search
**1.23×**, hence at `teacher_prob=0.1` (teacher = 96.5% of episode time) a
**single worker** would go 37.5 → ~31.9 s/episode, about **1.18×**.

That is not the operative number, because there are 8 workers and one GPU:

| configuration | aggregate inference throughput |
|---|---|
| current: 8 CPU workers | **~25,000-29,000 states/s** |
| MPS, best case (everything batched to B=1024) | ~17,400 states/s |
| MPS at the batch sizes used today | lower still |

Two independent routes agree on the CPU figure: 0.213 eps/s × ~113k
states/episode ≈ 24k/s, and 8 workers × 72% inference duty × ~200 µs/state ≈
29k/s.

**Both candidate architectures lose on this machine:**

- **Per-worker MPS** — 0.89× per worker before contention, then 8 workers
  serialise on one GPU. Roughly **0.05-0.15 eps/s** against today's 0.213.
- **Centralised inference server** (the deferred plan) — merges to B≈768 and
  removes the serialisation, but is capped by GPU throughput at 0.64-0.73× of
  the CPU aggregate: **~0.14-0.16 eps/s**. Still a regression.

### Confidence

The comparison is biased *toward* MPS, which is why the negative verdict is
fairly safe: CPU was measured on a box already saturated by 8 training workers,
while MPS had the GPU entirely to itself. Real 8-way MPS would be worse than
modelled. The softest input is states-per-episode (~113k), reconstructed from a
time histogram rather than counted directly; it would have to be wrong by ~2× to
change the conclusion, and the two independent routes agreeing argues against
that.

### What would flip it

1. **A CUDA box.** The header of these notes already flags that the real
   training box may be CUDA. There this likely flips and the inference-server
   design becomes worth building. **Re-run §A4 before writing MPS off there.**
2. **Raising `ISMCTSConfig.batch_size`** so the dominant batch moves from 96 to
   ≥512, where MPS gets its 2.4-3.6×. But that perturbs leaf-parallel
   virtual-loss diversity (a search-semantics change needing its own equivalence
   work), and even at B=1024 the GPU aggregate still sits below the CPU
   aggregate on this machine.
3. A materially larger model, where per-state work rises and kernel-launch
   overhead stops dominating.

**Decision: stay on CPU; keep the inference-server plan deferred.** The three
commits stand regardless — bit-exact and CPU-neutral, and they are what makes
MPS 0.89× at B=96 rather than the ~0.4× it would otherwise have been.

## A6. How to reproduce

The two instruments worth keeping are **committed** — §A1/A2 and §A4 can be
re-run directly, which is the point on a different box:

```
# A1/A2 — op coverage + host syncs (load-independent; --attribute names the
# source line behind each sync). Run the accelerator pass with the fallback
# enabled so gaps enumerate, then again without it: completing is the proof.
uv run python -m sheepshead.analysis.device_op_audit --device cpu
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  uv run python -m sheepshead.analysis.device_op_audit --device mps --attribute
uv run python -m sheepshead.analysis.device_op_audit --device mps

# A4 — per-device batch sweep + the aggregate-throughput table that actually
# decides the shared-accelerator question. --threads 1 (default) matches a
# league worker; benchmarking with default threads on a busy box is what
# produced the discarded first pass.
uv run python -m sheepshead.analysis.bench_inference_device
```

`bench_inference_device` splits each measurement into **marshal** (host-side
Python packing of the observation dicts) and **device** (everything after).
That distinction only matters off-box, but there it is decisive: marshalling is
a flat ~10 µs/state on the M1 Max and never amortizes with batch size, so a
remote accelerator fed pre-packed arrays should be charged the *device* column
only. Reading the *total* column for a remote device measures the remote host's
Python speed, not its GPU — which on an old CPU can be most of the number.

It also prints an aggregate states/s table comparing N concurrent CPU workers
against one accelerator, since per-call speedup is not the deciding number when
workers share one GPU. On this box it reads:

| B | cpu ×8 | mps ×1 |
|---|---|---|
| 1 | 5,266 | 84 |
| 96 | 40,507 | 5,421 |
| 1024 | 35,224 | 15,370 |

Two harnesses remain one-off and uncommitted, in the same spirit as
`profile_throughput.py` above:

- `sync_counterfactual.py` — per-sync cost microbenchmark, and a same-process
  A/B of the shipped vs sync-free actor scatter (its finding is now landed, so
  the A/B has no shipped counterpart left to compare against).
- `search_profile.py` — wraps `encode_batch` / actor / critic with timers and a
  batch-size histogram, then runs one `ISMCTSTeacher.search_committee` at
  production settings (§A3).

Op coverage and sync counts are dispatcher-level facts and are
load-independent; the timing sections should be re-run on a quiet machine
before being quoted as absolutes.

---
---

# Addendum 2 (2026-08-18) — device and link benchmarks for remote inference

Addendum 1 ruled MPS out **on this machine** and flagged that a CUDA box would
likely flip the verdict. One became available (PC with an **RTX 5060 8GB** and a
**Core i5-6500**, Skylake 2015), and the answer was yes — which grew into a
two-machine inference split with its own seam through the search engine.

**That system, its motivation, architecture, status and plan now live in
`Distributed_Inference_202608.md`.** What stays here is the benchmark data it
rests on: how fast each device is, how fast the link is, and the measurement
trap that nearly buried the result.

## B1. The marshal trap — read this before benchmarking any accelerator

The first `bench_inference_device --devices cuda` run on the PC read:

| B | 256 | 1024 | 2049 | 4096 |
|---|---|---|---|---|
| total µs/state | 76.0 | 52.5 | 50.7 | 52.0 |

Against a ~42 µs/state breakeven that is a clear no. It was wrong, and the shape
is the tell: per-state cost **flat** from B=1024 to B=4096, total time scaling
3.96× for a 4× batch increase. A small model on a GPU gets *cheaper* per state
as launch overhead amortizes; near-perfect linearity is a per-state **host**
cost.

That cost is `encode_batch`'s marshalling — ~19 Python operations per state, six
constructing a tensor. At B=4096 that is ~74,000 tensor constructions on one
Skylake core. Splitting the measurement:

| | B=1024 | B=2049 | B=4096 | B=8192 |
|---|---|---|---|---|
| i5-6500 marshal µs/state | 39.6 | 40.4 | 39.8 | 40.6 |
| **RTX 5060 device µs/state** | **10.0** | **6.3** | **7.0** | **9.0** |
| M1 Max marshal µs/state | 10.0 | — | 9.9 | — |
| M1 Max MPS device µs/state | 21.9 | — | 19.6 | — |

The original 52 µs/state was ~85% Skylake Python. `bench_inference_device` now
reports marshal and device separately: **on one box read `total us/st`; for a
remote accelerator fed pre-packed arrays read `device us/st`.** Reading the total
for a remote device measures that host's Python speed, not its GPU.

## B2. Device summary (per state, B≥1024)

| device | µs/state | note |
|---|---|---|
| RTX 5060 | ~7 | device only; host marshals |
| M1 Max GPU (MPS) | ~20 | device only |
| M1 Max CPU | 180-230 | single-threaded, as a league worker runs |
| i5-6500 | — | marshalling only, ~40 µs/state |

The CPU/MPS crossover on the M1 Max sits at **B≈160**, and the CE teacher's
dominant round is B=96 (`R × ISMCTSConfig.batch_size` = 3 × 32) — *below* it.
A sweep that stops at 64 or starts at 256 gives the opposite answer.

## B3. Link (direct point-to-point gigabit)

Baseline RTT **p50 0.354 ms** (min 0.157, p99 0.745). Sustained **0.72-0.87
Gbit/s**, ~90% of line rate. `t_wire` is the best per-state figure across round
sizes; the pre-measurement estimate from nominal link rate is in brackets, and
every one landed within ~20%.

| wire encoding | B/state | t_wire @1GbE |
|---|---|---|
| naive (int64 ids, fp32 memory both ways) | ~2768 | 25.7 µs *(22.1)* |
| packed (uint8 ids, fp16 memory both ways) | 1285 | 11.9 µs *(10.3)* |
| packed + memory resident on server | 261 | 2.6 µs *(2.1)* |

For contrast, the first attempt ran over a **WiFi 6 mesh**: 6 ms RTT at
0.08-0.21 Gbit/s, capping the byte-thriftiest encoding at 1.9×. The wire, not
the code, was the whole difference between that and 3.24×.

Tail caveat: percentiles need ≥30 samples per configuration. An earlier sweep
took 6 at 4032 states, so its p90/p99 reported little more than the worst
observation — one 355 ms outlier read as a 10% tail. Fixed, but worth
remembering: the search issues ~825 *sequential* rounds per committee, so tails
compound rather than average out.

## B4. How to reproduce

```
uv run python -m sheepshead.analysis.bench_inference_device
uv run python -m sheepshead.analysis.bench_lan_roundtrip --serve        # GPU box
uv run python -m sheepshead.analysis.bench_lan_roundtrip --connect <ip> # orchestrator
```

`--threads 1` (the default) matches a league worker. Benchmarking with the
default thread count on a box already running a generation oversubscribes the
cores and inflated the CPU column severalfold in a discarded first pass.
