# Distributed Inference for the CE Search Teacher — 2026-08

Status: **built and measured end to end; NOT viable as it stands.** Eight
concurrent clients against the RTX 5060 run at **0.68×** — slower than the CPU
path it replaces (§5.3). Cross-worker batching is built and correct but merges
only 1.4 requests per batch, for a queueing reason that no server-side change
fixes. The binding cost is ~17.7 ms of per-batch CUDA dispatch on the i5, with
the GPU 5% utilised, so the next move is cutting that (§6.0).

**Tried on the M1 Max first (§5.4): `torch.compile` gives 1.13× end-to-end at 8
workers — real, cheap, and far short of 3×.** It also found that the seam
carries only ~42% of committee time, not 74.5%, so the ceiling for offloading it
at all is ~1.7×. The same lever is still untested on the i5, where the dispatch
ratio is much worse and CUDA supports `reduce-overhead` properly.

What began as "is MPS worth it?" turned into a two-machine inference split, a
seam through the live trainer's search engine, and three bit-exact changes to
the actor and encoder. This notebook is the record for that system. Raw
device/link benchmark tables and the MPS verdict stay in
`Throughput_Profiling_Notes.md`; this file covers motivation, architecture,
what is built, what was measured, and what is left.

---

## 1. Why

At `teacher_prob=0.1` the league trainer runs at **0.30 eps/s** with 8 workers —
measured from `runs/league_ce_teacher11/train.log`, not modelled. Of a worker's
26.7 s episode, the CE teacher's committee search is the overwhelming majority,
and profiling a production-config committee found **74.5% of that is network
inference**, spread over batches of 96 to 1024 states.

So ~70% of training wall time is batched forward passes. That is the target.

> **Correction (2026-08-19).** This section originally used **0.213 eps/s** and
> **113,000 states/episode**, both taken from the CE doc's pre-launch model
> (§8: `eps/s ≈ 8/(1.3 + 362·p)`) and a reconstruction from a time histogram.
> Both were wrong, in opposite directions:
>
> | quantity | was | now | source |
> |---|---|---|---|
> | baseline | 0.213 eps/s | **0.30** | train.log, cumulative average |
> | committees/episode | 0.78 | **0.439** | log: 640 `searched` per 1459 episodes |
> | states/committee | ~145,000 | **76,501** | `bench_remote_search` |
> | states/episode | 113,000 | **33,558** | product of the two above |
> | rounds/episode | ~897 | **362** | 825 rounds/committee × 0.439 |
>
> The headline ceiling barely moved (3.24× → 3.29×) because a 41% higher
> baseline and a 3.4× lower state count push the ratio in opposite directions.
> But the *absolute* loads changed a lot, and two conclusions flipped — see the
> correction note in §6.3, and the warning in §7 about addendum 1.

Two prior conclusions bound the problem:

- **MPS is not the answer on this machine** — re-derived from measurement in
  §5.1 after the original argument's numbers were retracted. The binding cost is
  Metal kernel-launch overhead in the actor graph (~10 ms/round, flat in batch
  size), which no batching or transfer fusion reaches. Addendum 1 also found
  that MPS was being slowed by 24 batch-size-independent host syncs in our own
  actor, since fixed.
- **The Amdahl ceiling is ~3.2-4×.** 25.5% of committee-search time is Python
  tree/game work that stays on the orchestrator. No accelerator touches it.

A second machine changed the arithmetic: a PC with an **RTX 5060 8GB** — and a
**Core i5-6500** (Skylake, 4 cores, 2015) that is far too slow for anything CPU
bound. That asymmetry is the whole design constraint. It is a GPU on a stick.

## 2. Architecture

### 2.1 The seam

`ISMCTSTeacher._run_network_round` was already the right boundary: it takes an
ordered request list, contains no RNG, and returns plain arrays. Its only
coupling to search state is that it writes `memory_out` back into
`sim.mem[sim.seat-1]`, so the backend returns that alongside probabilities and
values and the caller performs the write-back, preserving ordering across
controller groups.

```
ISMCTSTeacher._run_network_round
    └── self.backend.evaluate(controller, states, memory_in, valid_lists, wants_critic)
            ├── LocalBackend   in-process forward (default)
            └── RemoteBackend  marshal → pack → TCP → unpack
```

### 2.2 Marshalling stays on the orchestrator

The load-bearing constraint, and the one that nearly sank the project when a
benchmark hid it.

`encode_batch` did two things with very different cost profiles: ~19 Python
operations per observation to pack dicts into tensors, then the forward. The
packing is **fixed per state and never amortizes** — ~10 µs/state on an M1 Max,
**~40 µs/state on the i5**. The GPU forward is ~7 µs/state.

Marshalling on the accelerator host would therefore cost ~40 µs/state to save
~7. `CardReasoningEncoder` is now split into `marshal_batch` (host) and
`encode_tensors` (device); the server consumes packed tensors and never sees an
observation dict.

### 2.3 Wire protocol

Byte count is the constraint, not CPU: the GPU does ~7 µs/state of work while
the link costs 2-26 µs/state depending on encoding. So the format is packed —
uint8 card ids, bit-packed action masks, and a configurable float dtype.

| direction | contents | fp16 | fp32 |
|---|---|---|---|
| up | observation + mask + `memory_in` | 565 B/state | 1108 B/state |
| down | `probs` + `value` + `memory_out` | 734 B/state | 1468 B/state |

Measured on a real search: 1331 B/state (fp16), 2576 B/state (fp32). **The GRU
memory is ~80% of fp32 traffic** — see §6.3.

### 2.4 Invariants

These are load-bearing, not stylistic:

- **`LocalBackend` is the default and is bit-exact.** `capture_search_goldens`
  pins torch op order and RNG draw order; a server-backed run batches across
  workers, changing GEMM tiling, so it can *never* satisfy that gate. Goldens
  are captured against the local path only.
- **Eval, CRN panels and goldens never move off CPU.**
- **The handshake compares weight fingerprints and refuses a mismatch.** Serving
  stale weights raises nowhere — it silently produces a plausible, wrong search
  target. A league run's `_league_worker_weights_v*.pt` rotate as it progresses,
  which is exactly how that would happen (observed: v24 → v29 during this work).

## 3. What is built

Bit-exact prerequisites, all gated on `capture_arch_goldens` **and**
`capture_search_goldens`:

| commit | change |
|---|---|
| `0df8281` | Actor logit scatter without per-slot host syncs (24 stalls → 0) |
| `0b0885e` | `act()` returns in one device-to-host transfer (5 → 1) |
| `03d151f` | Actor action-index maps as non-persistent buffers |
| `3060296` | **`encode_batch` split into `marshal_batch` + `encode_tensors`** |

The system:

| commit | change |
|---|---|
| `1e074cc` | `sheepshead/inference/`: protocol, `LocalBackend`/`RemoteBackend`, server, 9 tests; the `_run_network_round` seam |
| `aa3cd98` | Servers interruptible on Windows; `--bind`; startup prints reachable addresses |
| `34d14ea` | Connect failures diagnose refusal vs timeout instead of raising bare |
| `f0ba696` | Protocol v2: fused request/response blocks (13 transfers → 4 in, 3 D2H → 1) |
| (this change) | **Cross-worker batching** — see §6.1 |

Instruments (`sheepshead/analysis/`):

| tool | measures |
|---|---|
| `device_op_audit` | aten op coverage per backend, host syncs, transfers |
| `bench_inference_device` | per-device batch sweep, marshal vs device split |
| `bench_lan_roundtrip` | link RTT and throughput at realistic payload shapes |
| `bench_remote_search` | a real committee search, local vs remote |

## 4. Measured results

Full tables in `Throughput_Profiling_Notes.md`; the load-bearing numbers:

**Device** (per state, B≥1024): RTX 5060 **~7 µs**, M1 Max MPS ~20 µs, M1 Max
CPU ~180-230 µs single-threaded. i5-6500 marshalling ~40 µs.

**Link** (direct point-to-point gigabit): RTT p50 **0.354 ms**, sustained
**0.72-0.87 Gbit/s** ≈ 90% of line rate. Per state: packed 11.9 µs, naive 25.7,
memory-resident 2.6.

**End-to-end**, one production committee (1024 iters, R=3, 8M checkpoint),
single worker, no cross-worker batching:

| | local | remote fp32 |
|---|---|---|
| wall | 60.0 s | 50.7 s (**1.18×**) |
| rounds | 825 | 825 |
| mean batch | 93 states | 93 |
| client blocked on I/O | — | 13.44 s |
| server-side compute | — | 9.59 s |

Round count is **825**, not the ~1150 the model assumed — that figure counted
all `encode_batch` calls, not just blocking round-trips. Combined with the log's
0.439 committees/episode, that gives the corrected **362 rounds and 33,558
states per episode** used throughout (see the §1 correction).

### 4.1 Numerics: fp16 was the problem, CUDA is not

`pi_gumbel` divergence against the local search, same RNG:

| wire | rep 0 | rep 1 | rep 2 |
|---|---|---|---|
| fp16 | 3.3e-13 AGREE | 1.1e-04 AGREE | **1.0e+00 DIFFER** |
| fp32 | 2.2e-15 AGREE | 2.4e-11 AGREE | 7.8e-12 AGREE |

The fp16 pattern is bimodal — trees either stay locked or bifurcate entirely —
the signature of chaotic amplification rather than accumulating error: one
perturbed prior flips a PUCT selection and 825 rounds later the tree differs.
`pi_gumbel` saturates to one-hot at these budgets, so 1.0 is an outright label
flip.

**With fp32 the question closes.** CUDA-vs-CPU device numerics land ~nine orders
of magnitude below the 0.026 Q shrinkage noise floor, so `shrink_s2_global` does
not need refitting for the device. The cause was fp16 quantisation of the
recurrent state, fed back every round.

**fp32 is therefore required** until a per-field precision experiment says
otherwise, at the cost of doubling the payload.

#### 4.1.1 Merged batches diverge more than single ones — measured

The fp32 figures above are **single-client**. Batching changes batch
composition, hence GEMM tiling, hence the last bits of every reduction, so it
was never going to carry over unchanged. Measured on loopback with everything
else held fixed — same CPU, same weights, same RNG, fp32 wire, so tiling is the
*only* difference:

| comparison | max &#124;Δ pi_gumbel&#124; |
|---|---|
| single-client, CUDA vs local CPU | 2e-15 … 2e-11 |
| **merged batch (3-4 clients), CPU vs local CPU** | **1.6e-06 … 2.9e-06** |

Five orders of magnitude worse, and argmax still agrees everywhere. It is the
expected cost, not a defect — but four orders below the 0.026 Q shrinkage floor
rather than nine, so the margin becomes a thing with a size rather than a
formality.

**Measured on the real pair (2026-08-19), it is a non-issue**: 8 clients, fp32,
RTX 5060 gave **1.7e-16 / 9.1e-12 / 1.6e-11**, all argmax agreeing — as good as
the single-client figures. The reason is uncomfortable rather than reassuring:
the merge factor is only 1.4 (§5.3), so most rounds *are* batches of one. The
caveat is re-scoped, not retired. **If `S` ever drops far enough for batches to
actually fill, re-measure this.**

This is also why a server-backed run can never be golden-pinned (§2.4): the
divergence is a function of *who else happened to be in the batch*, which is not
reproducible even in principle.

## 5. Where the time goes now

Of 13.44 s blocked, **9.59 s was server-side compute** and only ~3.85 s was wire
plus client packing. Per round that is 11.6 ms of server time against a ~0.65 ms
GPU forward — **~94% is fixed per-round overhead** (unpack, dtype conversions,
ten small H2D transfers, three D2H syncs, pack) on a 2015 Skylake.

With the corrected state count that becomes decisive rather than merely
significant:

| server mode | µs/state | capacity | vs 0.30 |
|---|---|---|---|
| unbatched | 125 | 0.24 eps/s | **0.80× — a regression** |
| ~~batched 8× (~750-state rounds)~~ | ~~22~~ | ~~1.37 eps/s~~ | ~~above the Mac ceiling~~ |
| batched, **measured** 8 clients | **136** | **0.219 eps/s** | **0.73× — still a regression** |

**Unbatched, this would be slower than the current CPU trainer.** Not marginal —
an outright regression. Cross-worker batching was therefore treated as the thing
that makes the design viable at all.

**It is built, and it did not deliver (§5.3).** Eight real clients merged 1.4
requests per batch, not 8, for a reason that turns out to be arithmetic rather
than an implementation defect — so the second row above is struck and replaced
by measurement. The bottleneck is per-batch dispatch cost on the i5, not the
merge factor, and §6 is reordered accordingly.

## 5.1 MPS on the M1 Max: settled, and it loses (2026-08-19)

Addendum 1's verdict was flagged for re-derivation after the baseline
correction. It is now re-derived, and it holds — for a better reason than the
original.

A single-machine MPS design would be far simpler: no second box, no weight sync
over a wire, no network failure modes, no precision question. So it was worth
settling properly. Two candidate shapes:

**Per-worker MPS** (just change the device — the genuinely simple option). At the
production round size of B≈93 the GPU delivers **9,715 states/s** against a
demand of 33,222. Eight workers serialise on one GPU, so this is 3.4× short: a
wash with today at best.

**MPS behind a batching server.** Measured by pointing the existing server at
`--device mps` on loopback: **0.90×** against local, with 24.9-27.2 s of
server-side compute over 825 rounds. Loopback confirms the network was never
the issue — 0.3 s of 25.2 s.

The cost is **Metal kernel-launch overhead in the actor graph**, not data
movement. Measured decomposition:

| MPS device-only | B=93 | B=744 |
|---|---|---|
| encoder alone | 11.85 ms | 29.62 ms |
| encoder + actor | 21.31 ms | 39.98 ms |
| actor's contribution | 9.5 ms | **10.4 ms** |

The actor costs a flat ~10 ms at both batch sizes — it is launches, not work.
Fitting both points gives **18.6 ms fixed + 28.7 µs/state**:

| merged batch | µs/state | states/s | vs 33,222 demand |
|---|---|---|---|
| 744 (8 workers) | 53.7 | 18,609 | short 1.8× |
| 1488 | 41.2 | 24,267 | short 1.4× |
| 4096 | 33.2 | 30,093 | still short |

MPS does not clear even in the limit, and B=744 is what 8 workers actually
produce — larger merged batches would need ~20 workers, which the Mac has no
cores for. **A transfer-fusion attempt (protocol v2, `f0ba696`) did not move it**,
because the cost was never transfers; that hypothesis and its refutation are in
the commit message.

The RTX 5060 clears the same bar with room: ~10 ms of i5 Python plus 744 × 7 µs
≈ 15 ms per merged round ≈ 20 µs/state ≈ 49,000 states/s.

**Verdict: the two-machine design is the path.** MPS is simpler and loses on
throughput, and unlike the link or the transfers there is no cheap fix — it
would need the actor's op count cut or a working `torch.compile` MPS backend,
neither of which is a small change.

## 5.2 The non-teacher share: 1.3 s confirmed, and it is a serial update

The last modelled input in the chain. Production runs at **6.2 eps/s with the
teacher off** (operator, `league_retention_pg` generations), and the CE model's
`8/(1.3 + 362·p)` gives `8/1.3 = 6.15` at p=0. **The 1.3 s was right.**

A first attempt to measure it directly gave 0.199 s/episode and was wrong,
instructively. Timing one process generating episodes and calling `update()`
measures a *worker's* cost; production's 1.3 s is dominated by the **synchronous
learner**, where all 8 workers idle during the gradient update. Decomposed
against the 6.2 eps/s anchor, over a 1459-episode window:

| | per window | per episode-worker-equivalent |
|---|---|---|
| generation (parallel over 8 workers) | 21 s | 0.11 s |
| **PPO update (serial, workers idle)** | **215 s** | **1.18 s** |
| total | 235 s | 1.29 s |

So ~92% of the non-teacher share is a serial stall, not work that parallelism
helps. Measured update cost is also **superlinear** in window size — 0.156,
0.109, 0.146 s/episode at n = 100, 300, 600, with segment slopes 0.086 then
0.183 — so it must not be extrapolated; the production anchor is the number to
use.

Consequences, none of which move the headline:

- **Ceiling essentially unchanged: 0.99 eps/s = 3.29×** (it was 3.3× under the
  modelled value). Demand 33,138 states/s against 33,222. MPS still short at
  18,609; the 5060 still clears at ~49,000.
- **The serial update becomes the next bottleneck.** It is 4% of wall time today
  and **15% after offload**. If teacher work were free the ceiling would be
  6.2 eps/s — so there is real headroom, but the update is now a visible term
  rather than a rounding error, and it is the thing to attack after this.
- **`teacher_prob` interacts with it.** Raising p adds parallel work while the
  serial update stays fixed, so the update's share shrinks — the headroom from a
  faster accelerator converts into labels more efficiently than the naive model
  suggested.

## 5.3 Eight clients on the real pair: batching does not fill (2026-08-19)

The measurement §6.1 was waiting for. M1 Max uncontended, 8 client *processes*,
RTX 5060 server, gigabit point-to-point, fp32, 8M checkpoint, one production
committee each (1024 iters, R=3).

| | local, 8 workers | remote, 8 clients |
|---|---|---|
| wall, slowest client | 56.92 s | 83.36 s |
| aggregate | **10,751 states/s** | **7,341 states/s** |
| rounds | — | 6,602 |
| client blocked on I/O | — | 398.1 s of 642.5 s |
| merge factor | — | **1.32-1.45 req/batch** |

**0.68× — the remote path is slower than the CPU it replaces.** Batching did not
change the per-round economics either: **136 µs/state** against 125 unbatched,
where §5's model assumed 22.

(The local column independently confirms the §1 correction: 33,558 states/episode
× 0.30 eps/s = 10,067 states/s predicted, 10,751 measured.)

### 5.3.1 The merge factor is arithmetic, not a bug

For a greedy batcher the merge factor is simply how many requests arrive during
one service time. In a closed system with `N` synchronous clients:

```
k  =  N · S / (T + L)
```

where `S` is service time per batch, `T` client think time, `L` round-trip
latency. From the run: `L` = 398.1/6602 = **60.3 ms**, `T` = (80.31−49.8)/825 =
**37.0 ms**, `X` = 6602/83.36 = **79.2 req/s** (against `N/(T+L)` = 82 ✓).
Solving for `S` gives **17.7 ms/batch**; substituting back predicts
**k = 1.45** against 1.32-1.45 measured.

The structural consequence: **`k = N` requires `S = T + L`, which is impossible
because `L ≥ S`.** Eight synchronous clients cannot fill a batch of eight unless
the server is slower than an entire client cycle — and if batching ever does
make the server faster, it un-fills its own batches. It is self-limiting.

This is not greedy-vs-windowed. A 40 ms window models to about +20%, because in
a closed system the window is dead time that raises `L` and throttles arrivals.
The lever that does work is `N`: `k` scales linearly with client count, and
during the remote phase each client is only ~49% CPU-busy, so the Mac could host
more clients than it has cores. That is a trainer-side change, not a server one.

### 5.3.2 Everything except the i5 is idle

| station | load |
|---|---|
| RTX 5060 | 7,341 of ~140,000 states/s — **5%** |
| gigabit link | 19 MB/s of ~90 — 21% |
| M1 Max | 8 procs × 38% busy ≈ 3 of 10 cores |
| **i5 Python / CUDA dispatch** | **~17.7 ms per batch, near-saturated** |

17.7 ms against a 0.65 ms forward is kernel-*launch* overhead: hundreds of small
aten dispatches through Python on a 2015 Skylake. This is the same pathology
that killed MPS (§5.1, a flat ~10 ms in the actor graph), reappearing on the
other machine for the same reason — an op-count-heavy graph driven by a slow
interpreter.

It also reframes the contest. The Mac's local path scaled ~8× across 8 processes
(57 s for eight committees against 60 s for one); the remote path scaled 4.9×.
**One GIL versus ten cores** is what the server is actually up against, and no
amount of merging changes that ratio — merging only amortizes work *inside*
`run_batch`.

### 5.3.3 What this changes

- **Deployed today the remote path caps training at 7,341/33,558 = 0.219 eps/s
  against 0.30.** Consistent with the observed 0.68×.
- **§5's "batched 8× → 22 µs/state → 1.37 eps/s" row is refuted.** The failure
  was not in the batching implementation but in assuming 8 clients would produce
  a merge factor of 8.
- **The target moves from `k` to `S`.** Cutting per-batch dispatch cost helps
  unconditionally, at every merge factor, and it is the same fix on both
  machines. CUDA graphs / `torch.compile(mode="reduce-overhead")` is the direct
  attack: at `S` ≈ 3 ms the model gives `X` ≈ 190 req/s and a remote wall of
  ~35 s against local 57 s, i.e. ~1.6×. Static shapes are the obstacle (rounds
  are 96-1024 states), but padding to buckets is nearly free when the GPU sits
  at 5%.
- **Worth trying on the M1 Max first.** If the win is dispatch overhead, a
  compiled local forward needs no second machine, no weight sync and no wire —
  which was the simpler path all along.
- **2.5GbE stays unnecessary**, now for a second independent reason: the link
  ran at 21%.

## 5.4 torch.compile on the M1 Max: real but small (2026-08-19)

§6.0 said try it locally first. Done, on an idle machine (the CE generation had
finished), against the production committee.

**It compiles cleanly.** Encoder + actor go into **one graph with zero graph
breaks**, in 0.4-5 s. The single most encouraging fact here: no rewriting, no
`torch._dynamo` wrangling.

**`mode="reduce-overhead"` is a no-op on MPS.** It selects CUDA graphs, which
Metal has no equivalent of — measured 2.72 ms against `default`'s 2.73 ms. On
this machine the question is simply whether `torch.compile` helps at all.

Forward only, B=96, single process:

| device | eager | compiled | |
|---|---|---|---|
| MPS | 6.49 ms | 2.72 ms | **2.39×** |
| MPS, B=744 | 16.05 ms | 15.84 ms | 1.01× |
| CPU (1 thread) | 13.37 ms | 8.21 ms | 1.63× |

MPS gains only at small batches, which is the signature of dispatch overhead: by
B=744 it is compute-bound and compilation has nothing left to remove.

### 5.4.1 End to end at the production worker count

The micro-benchmark is seductive and wrong. Eight concurrent committees, one per
worker, is the shape that counts (mean of 8, idle machine):

| arm | committee wall | inference | share | vs eager |
|---|---|---|---|---|
| CPU eager | 61.2 s | 31.30 ms/round | 42.3% | — |
| **CPU compiled** | **54.2 s** | **24.49 ms/round** | 37.3% | **1.13×** |
| MPS compiled | 55.4 s | 23.84 ms/round | 35.6% | 1.10× |

A 2.4× forward becomes **1.13× of training throughput.** All arms produced
identical labels (argmax `[92, 92, 78]` across all three replicates).

Two things invert between one process and eight. Compiled CPU shows *no* gain
single-process (37.7 s against eager's 37.5 s) yet 1.13× at eight — fused
kernels move less memory, which only matters once cores contend. And MPS
compiled leads single-process (31.8 s against 37.5 s) but merely ties compiled
CPU at eight, while costing more setup (68 s against 59 s including per-process
compilation).

**So the MPS verdict stands** — compiled CPU is at least as good and simpler.

### 5.4.2 Two corrections this forced

- **§5.1's MPS numbers were taken under the live training run and are inflated
  ~3.3×.** Encoder+actor at B=93 measured 21.31 ms then, **6.49 ms idle**. The
  fit "18.6 ms fixed + 28.7 µs/state" should be 5.1 ms + 14.7. MPS dispatch is
  CPU-side work, so a saturated host slows it directly. The *verdict* survives;
  the numbers do not. Same failure mode as the 0.213 eps/s baseline — a number
  recorded without the conditions it was taken under.
- **§1's "74.5% of committee time is network inference" is right — but only
  ~33% of it is behind the seam.** Through `backend.evaluate` it is 42% of a
  committee; §5.5 attributes the rest and confirms the total at **75.3%**. So
  the seam alone is bounded at ~1.5×, and reaching the real ceiling means
  compiling the replay path too — which §5.5 does.

## 5.5 Compiling every encoder site: 1.41× (2026-08-19)

`encode_batch` is called from four places, and the seam is the smallest of the
three that matter. Attributed on a CPU committee, splitting host marshal from
device work:

| site | marshal | device | total | % wall | shapes |
|---|---|---|---|---|---|
| `_run_network_round` (the seam) | 0.87 s | 11.24 s | 12.11 s | **33.0%** | 44, B=96 dominant |
| `_observe_completers_merged` | 0.81 s | 9.15 s | 9.96 s | **27.1%** | 49, B=160/320/480 |
| `_encode_seat_batched` | 0.29 s | 3.33 s | 3.63 s | 9.9% | **1: B=1024** |
| `_observe_trick_lockstep` | 0.16 s | 1.79 s | 1.95 s | 5.3% | **1: B=1024** |
| **encoder total** | 2.13 s | 25.5 s | **27.6 s** | **75.3%** | |
| tree / game / actor / critic | | | 9.1 s | 24.7% | |

That vindicates §1's 74.5% and the "25.5% Python tree work" Amdahl figure — both
were right; §5.4.2's doubt was measuring only the seam. It also explains
`_observe_completers_merged`'s own docstring claim that the observes are "~30% of
committee runtime".

Since every site routes through `encode_batch`, patching *it* reaches all four at
once. Bucketing to multiples of 32 covers everything in **14 shapes / 8 compiled
graphs at 1.8% padding waste**. Eight concurrent committees, steady state
(committee 2 of 3):

| arm | committee wall | vs CPU eager |
|---|---|---|
| CPU eager | 62.4 s | — |
| CPU compiled | 58.5 s | 1.07× |
| MPS eager | 58.6 s | 1.07× |
| **MPS compiled** | **44.3 s** | **1.41×** |

Backing out the split: the 75.3% encoder share got **1.63× faster**, which is
exactly what the micro-benchmarks predicted, and 1.41× is what that becomes once
the 24.7% of tree work is carried along. **0.30 eps/s → ~0.42, on one machine,
with no wire and no weight sync.**

### 5.5.1 MPS eager was never as bad as recorded

`mps eager` runs its first committee in 50.6 s and its second in **33.7 s** —
Metal caches compiled shaders across calls. Every MPS figure in this project
before now was a first run, including §5.1's.

So the standing "MPS loses on the M1 Max" verdict was an artefact twice over:
measured under load *and* measured cold. In steady state MPS eager already edges
CPU eager (58.6 s against 62.4 s at 8 workers). The margin is small enough that
the verdict never mattered much on its own — but it mattered a great deal that
it discouraged trying `torch.compile` on MPS, which is where the 1.41× is.

**Measurement rule this earns: report steady state, and record machine load and
warm/cold status next to every timing.** Three separate conclusions in this
notebook have now been distorted by one or the other.

## 6. Planned work

### 6.0 Cut per-batch dispatch cost — measured locally at ~1.1×

**Superseded by §5.4 for the local path.** `torch.compile` works, needs no
bucketing scheme, and returns 1.13× end-to-end on CPU at 8 workers. Worth having
and cheap — but not the 3× this program was chasing.

The complexity is far lower than this section first assumed. It claimed rounds
"run 96-1024 states"; **measured, 87.4% of rounds are exactly B=96 and none
exceed it** (44 distinct shapes, but B=96 carries 90.7% of all states), because
`R × ISMCTSConfig.batch_size` bounds them. Padding every round up to 96 therefore
gives **one** compiled shape at a cost of 3.8% wasted rows — no buckets, and no
dynamic shapes, which are markedly worse (1.45-1.82× against 1.95-2.38×).

Productionising it would need:

- a `CompiledBackend` beside `LocalBackend`, padding to `R × batch_size`, with
  pad rows given an all-legal mask (an all-False row softmaxes to NaN and
  poisons the whole batch)
- the critic reusing the compiled graph's `encoded` dict — recomputing the
  encoder for it costs more than compilation saves, measured
- **opt-in, never the default.** Compiled output differs from eager by 2.6e-08
  on probs, so `capture_search_goldens` cannot pass against it. Same status as
  remote mode.

**Superseded again by §5.5**: compiling only the seam reaches 33% of a committee,
compiling every `encode_batch` site reaches 75.3% and returns **1.41× on MPS at
8 workers**. That is the local recommendation.

On the **remote** path the same lever is now available (`--compile`, default mode
`reduce-overhead`, `--pad-granularity`) and untested. The case is stronger there:
the i5 spends ~17 ms of dispatch against a 0.65 ms forward, a far worse ratio
than anything measured on the Mac, and CUDA actually does support
`reduce-overhead` (on MPS it is a no-op).

**Inductor needs Triton, and upstream PyTorch does not ship Triton for
Windows** — the GPU box raises `TritonMissing` on the first batch. Two ways past
it, and the second is arguably the better one anyway:

- `pip install triton-windows` (community port; must match the torch version).
- **`--compile-backend cudagraphs`.** No code generation at all: it captures the
  eager kernel sequence and replays it as one graph. That is aimed squarely at
  this server's problem, which is launch *count*, not kernel quality — ~17 ms of
  dispatch against a 0.65 ms forward. Inductor's fusion would be a bonus on top,
  not the main event.

A compile failure now **degrades to eager instead of taking the run down**. It
fails on the first batch — which is to say on every worker at once,
mid-generation — so the alternative is losing the generation to a missing
dependency. The fallback retries on the unpadded inputs, so the reply is
bit-identical to a plain eager server's.

Two things to expect when running it:

- **Compilation must amortize.** A short bench is meaningless — a 78-round
  loopback smoke test measured 644 ms/batch because nearly every batch compiled
  a new shape. Use `--clients 8` at full iters (~6,600 rounds).
- **Merged sizes vary more than a single client's**, being sums over however many
  clients arrived. `--pad-granularity 96` is the natural quantum here
  (`R × ISMCTSConfig.batch_size`), so one client rounds to 96 with zero waste,
  two to 192, and the shape count stays small — which matters because inductor
  codegen on a 2015 Skylake is not fast.

### 6.1 Cross-worker batching — BUILT, and it did not pay

**Built.** The server now accepts concurrent connections and merges whatever is
queued into a single forward. Its value is *not* mainly GPU efficiency: it
amortizes the fixed ~11 ms per round across 8× the states.

Shape:

- **One thread per connection** does socket I/O and the (cheap) decode; **one
  batcher thread** owns the model and runs every forward. Connection threads
  decode while the batcher computes, and no lock guards the model because only
  one thread ever touches it.
- **Greedy by default** (`--batch-window-ms 0`): take everything already queued
  and go. With synchronous clients that is self-synchronizing — while batch *k*
  computes, all *K* workers queue their next round, so batch *k+1* fills without
  anyone having waited. A window is available but costs its full value on
  *every* round when a single client is connected, which is the benchmark case.
- **Four transfers in, one out, regardless of client count.** `merge_requests`
  interleaves the field-major uint8 observation region so the merged batch keeps
  the layout that lets it cross in one copy; `response_block` returns a
  contiguous host array so each client's rows are a slice, not a second copy.
- A batch of one takes the same path — there is no separate unbatched branch to
  drift.
- Caps (`--max-batch-requests`, `--max-batch-states`) bound accelerator memory.
  A request that cannot share a forward (different `action_size` or wire dtype)
  is **returned to the queue**, never dropped: a dropped request strands a
  client forever on an Event only the batcher sets.
- Mixed `wants_critic` is handled by computing the critic if anyone asked and
  zeroing the column back out for the clients that did not, so a client cannot
  tell it was batched.

**Verified**: 17 protocol tests, both golden gates bit-identical (the local path
is untouched), and a 6-client loopback run whose every output matches
`LocalBackend` to <1e-5.

**Measured on the real pair, and the win did not materialise — see §5.3.** Eight
client processes merged **1.4** requests per batch, not 8, and end-to-end the
remote path ran at **0.68×**. The implementation is correct; the premise was
wrong. `k = N·S/(T+L)` caps the merge factor well below `N` for synchronous
clients, and the fixed cost that batching was meant to amortize is better
attacked directly.

Loopback on the loaded M1 Max had merged 1.6-2.6 — close to the real figure, and
in hindsight not the artefact of a saturated host it was written off as.

`bench_remote_search --clients N` is the instrument: N client *processes* (not
threads — GIL-serialised clients would stagger arrivals and flatter the merge
factor), barrier-synchronised so the local and remote phases are each genuinely
concurrent. The server prints merge factor, ms/batch, µs/state, queue wait and
**busy %**; the last two discriminate the two ways a merge factor can
disappoint. Busy near 100% with a long queue wait means the batcher is the wall
and a batch must get cheaper. Busy well under 100% with a short queue wait means
requests are not reaching the queue at all — the connection threads are behind,
and merging cannot help because the cost sits upstream of the merge point.

### 6.2 Weight sync — the largest production gap

The league publishes new weights every update. The fingerprint handshake would
reject the first one. Needs a coordinated reload: version-stamped weights, a
client-initiated swap, and a barrier so no round straddles a version change.
Until this exists the system cannot run a real generation.

### 6.3 Gigabit is enough — do not buy 2.5GbE

**Reversed 2026-08-19 by the corrected state count.** At 33,558 states/episode
and 2576 B/state (fp32), the link carries ~86 MB per episode. At the Mac-side
ceiling of 0.99 eps/s that is **0.66 Gbit/s against a measured 0.72-0.87
sustained** — inside budget, if not by a wide margin.

| configuration | t_eff | capacity | vs 0.30 |
|---|---|---|---|
| fp32, gigabit | 31.0 µs | 0.96 eps/s | 3.2× |
| fp32, 2.5GbE | 16.6 µs | 1.80 eps/s | (Mac-capped at 3.3×) |
| fp16, gigabit | 18.9 µs | 1.58 eps/s | (Mac-capped at 3.3×) |

fp32 on gigabit lands at 0.96 eps/s against a Mac ceiling of 0.99 — the two are
effectively tied, so faster networking buys nothing the orchestrator can use.
**2.5GbE is unnecessary.** The earlier recommendation to buy it came from the
inflated state count; with 3.4× less traffic than assumed, the link stopped
being the problem.

Memory-resident remains interesting for headroom and because it moots the
quantisation question — the GRU memory is 1024 B/state each way in fp32, ~80% of
traffic — but it is no longer needed to reach the ceiling, and it restructures
the search: `_run_network_round` writes `memory_out` back into client-side sim
state, so the client would have to send permutation indices against server-held
slots. Park it unless the link becomes binding again.

### 6.4 Smaller, cheap

- **Per-field precision experiment.** `--fp32` is all-or-nothing. If memory is
  the sensitive field (likely — fed back 825 times), fp16 probs+values saves
  ~10% of payload; if it is the other way round, fp16 memory saves 41%.
- **Fallback to local.** A dropped connection currently kills a generation.
- **Worker wiring** — routing league workers through `RemoteBackend`.
- **Client-side instrumentation.** ~40 s of the 50.7 s remote wall is
  orchestrator-side and only partly accounted for (tree work should be ~16 s).
  Could be client packing, could be the live training run's load — two local
  runs of identical work differed 4%. Worth measuring before it is modelled.

## 7. Risks

- **No golden pinning, ever.** Cross-worker batch composition changes GEMM
  tiling. The local backend stays default and stays bit-exact; that is the only
  path the goldens gate.
- **A remote dependency can kill a generation.** Needs §6.4's fallback.
- **The Amdahl ceiling is ~3.2-4×** regardless of hardware. It has never been
  approached: the measured figure is 0.68× (§5.3).
- **The merge factor cannot be fixed from the server side.** `k = N·S/(T+L)`
  bounds it below `N` for synchronous clients, so any future plan that assumes
  full batches is assuming something the client structure forbids.
- ~~**Addendum 1's MPS verdict rests on the same bad input and needs
  re-checking.**~~ **RESOLVED 2026-08-19 — see §5.1.** Re-derived from
  measurement: MPS loses, but because of Metal kernel-launch overhead in the
  actor graph (a flat ~10 ms/round), not the aggregate-throughput argument the
  original gave. Original note kept below for the record.
  <details><summary>original note</summary>

  `Throughput_Profiling_Notes.md` §A5 justified "MPS loses on the M1 Max" partly
  via `0.213 eps/s × 113k states/ep ≈ 24k states/s` for 8 CPU workers, and
  claimed a second route agreed at ~29k. Corrected, route 1 gives **10.1k
  states/s** — so the two routes never agreed; both were wrong and the errors
  happened to cancel. At 10.1k the M1 Max GPU may well *beat* 8 CPU cores. The
  original serialisation argument (8 workers, one GPU) is unaffected and may
  still carry the verdict, but it now needs the batched-server design to be
  compared honestly. **Do not treat addendum 1 as settled.**
  </details>
- **Production per-state inference cost is 2-3× the benchmark.** The corrected
  figures imply ~563 µs/state in production against 180-230 µs/state measured
  single-process by `bench_inference_device`. Eight workers plus a learner on
  ten cores is simply more contended than a benchmark process. This means the
  benchmarks *understate* the CPU cost, and so understate the offload win.
- ~~**Remaining unverified input: the 1.3 s non-teacher share.**~~ **CONFIRMED
  2026-08-19 — see §5.2.** Production runs at 6.2 eps/s with the teacher off,
  and `8/1.3 = 6.15`. The model was right. But it is ~92% *serial PPO update*,
  not per-worker work, which changes what it means — the update does not scale
  with workers and becomes the next bottleneck after inference is offloaded.
- **`0.30 eps/s` is printed to one decimal**, so the true rate is 0.25-0.349 —
  roughly ±16% on every ratio here.
- **Calibration is answered for fp32 only.** Any move back to fp16, or a change
  to what is quantised, reopens it. **And it is answered for single-client fp32
  only** — merged batches measured 1.6-2.9e-6 against 2e-15..2e-11 unbatched
  (§4.1.1). Still far under the 0.026 floor, but re-measure on the 5060, where
  device numerics and tiling changes compound.
- **The batcher thread is a single point of failure.** A forward that raises
  fails its whole batch's clients (deliberately — the alternative is stranding
  them), so one bad round disconnects up to `--max-batch-requests` workers at
  once. That is survivable only once §6.4's fallback-to-local exists.

## 8. Reproducing

```
# GPU box
uv run python -m sheepshead.inference.server --checkpoint <weights> \
    --device cuda --bind <p2p-ip>

# orchestrator
uv run python -m sheepshead.analysis.bench_lan_roundtrip --connect <p2p-ip>

# one client: latency and fidelity
uv run python -m sheepshead.analysis.bench_remote_search --checkpoint <weights> \
    --host <p2p-ip> --fp32

# eight clients: the batching win. Read the server's 'merge N req/batch' line.
uv run python -m sheepshead.analysis.bench_remote_search --checkpoint <weights> \
    --host <p2p-ip> --fp32 --clients 8
```

Both ends need **identical weights** — pass the same file, or omit `--checkpoint`
on both for a seeded fresh agent. Copy the file somewhere stable rather than
pointing at a live run's rotating `_league_worker_weights_v*.pt`.

Gates before any change to the search or encoder path:

```
uv run python -m sheepshead.analysis.capture_arch_goldens --check
uv run python -m sheepshead.analysis.capture_search_goldens --check
uv run pytest sheepshead/tests -q
```
