# Distributed Inference for the CE Search Teacher — 2026-08

Status: **prototype working and measured; not yet wired into training.**
Measured single-worker speedup 1.18×, modelled ceiling ~3.3×. **Cross-worker
batching is mandatory, not optional** — deployed as-is the design is a
regression (§5).

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

- **MPS is not the answer on this machine.** Eight M1 Max performance cores
  out-throughput the 32-core GPU on a model this small (~25-29k states/s vs
  ~17.4k best case). See `Throughput_Profiling_Notes.md` addendum 1 — including
  the finding that MPS was being slowed by 24 batch-size-independent host syncs
  in our own actor, since fixed.
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

## 5. Where the time goes now

Of 13.44 s blocked, **9.59 s was server-side compute** and only ~3.85 s was wire
plus client packing. Per round that is 11.6 ms of server time against a ~0.65 ms
GPU forward — **~94% is fixed per-round overhead** (unpack, dtype conversions,
ten small H2D transfers, three D2H syncs, pack) on a 2015 Skylake.

With the corrected state count that becomes decisive rather than merely
significant:

| server mode | µs/state | capacity | vs 0.30 |
|---|---|---|---|
| unbatched (today's prototype) | 125 | 0.24 eps/s | **0.80× — a regression** |
| batched 8× (~750-state rounds) | 22 | 1.37 eps/s | above the Mac ceiling |

**Deployed as-is, this would be slower than the current CPU trainer.** Not
marginal — an outright regression. Cross-worker batching is not an optimization
to schedule, it is the thing that makes the design viable at all, and everything
else in §6 is downstream of it.

## 6. Planned work

### 6.1 Cross-worker batching — the dominant lever

A server request queue with a batching window, merging 8 workers' rounds into
one ~750-state forward. Its value is *not* mainly GPU efficiency: it amortizes
the fixed ~11 ms per round across 8× the states. Free, and it must come before
any hardware decision.

Then re-measure. Only once that fixed cost is amortized does payload dominate.

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
- **The Amdahl ceiling is ~3.2-4×** regardless of hardware.
- **Addendum 1's MPS verdict rests on the same bad input and needs re-checking.**
  `Throughput_Profiling_Notes.md` §A5 justified "MPS loses on the M1 Max" partly
  via `0.213 eps/s × 113k states/ep ≈ 24k states/s` for 8 CPU workers, and
  claimed a second route agreed at ~29k. Corrected, route 1 gives **10.1k
  states/s** — so the two routes never agreed; both were wrong and the errors
  happened to cancel. At 10.1k the M1 Max GPU may well *beat* 8 CPU cores. The
  original serialisation argument (8 workers, one GPU) is unaffected and may
  still carry the verdict, but it now needs the batched-server design to be
  compared honestly. **Do not treat addendum 1 as settled.**
- **Production per-state inference cost is 2-3× the benchmark.** The corrected
  figures imply ~563 µs/state in production against 180-230 µs/state measured
  single-process by `bench_inference_device`. Eight workers plus a learner on
  ten cores is simply more contended than a benchmark process. This means the
  benchmarks *understate* the CPU cost, and so understate the offload win.
- **Remaining unverified input: the 1.3 s non-teacher share** per episode, still
  from the CE model. It sets the tree/inference split and hence the Mac-side
  ceiling. Worth measuring before the ceiling is quoted again.
- **`0.30 eps/s` is printed to one decimal**, so the true rate is 0.25-0.349 —
  roughly ±16% on every ratio here.
- **Calibration is answered for fp32 only.** Any move back to fp16, or a change
  to what is quantised, reopens it.

## 8. Reproducing

```
# GPU box
uv run python -m sheepshead.inference.server --checkpoint <weights> \
    --device cuda --bind <p2p-ip>

# orchestrator
uv run python -m sheepshead.analysis.bench_lan_roundtrip --connect <p2p-ip>
uv run python -m sheepshead.analysis.bench_remote_search --checkpoint <weights> \
    --host <p2p-ip> --fp32
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
