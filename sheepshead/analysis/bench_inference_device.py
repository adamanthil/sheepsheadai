#!/usr/bin/env python3
"""Per-device inference benchmark across the batch sizes search actually uses.

The go/no-go instrument for moving training inference to an accelerator (the
central-inference-server question). It times ``encode_batch`` -- which the
2026-08-18 profiling pass measured at 96.5% of a committee search's inference
time -- and optionally the full actor forward, on each requested device, in a
single process so the numbers are directly comparable.

Two details decide whether the output means anything:

  * ``--threads 1`` (the default) matches a league worker, which runs under
    torch.set_num_threads(1). Benchmarking with the default thread count on a
    box that is already running a training generation oversubscribes the cores
    and inflates the CPU column severalfold.

  * ``--batch-sizes`` should cover what the workload really dispatches, not
    round numbers. For the CE search teacher that is dominated by
    R * ISMCTSConfig.batch_size (= 96 at R=3) plus the larger pool-build
    batches; the CPU/MPS crossover sits between them, so a sweep that stops at
    64 or starts at 256 gives the opposite answer.

Interpreting the result: per-call speedup is not the deciding number when
workers share one accelerator. Compare the aggregate -- (worker count) /
(per-state CPU cost) against (1) / (per-state device cost) -- since N CPU
workers run concurrently while N MPS workers serialize on one GPU.

Usage:
    uv run python -m sheepshead.analysis.bench_inference_device
    uv run python -m sheepshead.analysis.bench_inference_device \
        --devices cpu mps --batch-sizes 1 32 96 160 320 480 1024
    uv run python -m sheepshead.analysis.bench_inference_device --with-actor

Results from the 2026-08-18 pass, and the aggregate-throughput argument built
on them, are in notebooks/Throughput_Profiling_Notes.md (addendum section A4).
"""

import argparse
import sys
import time

import numpy as np
import torch

from sheepshead import ACTION_IDS
from sheepshead.analysis.capture_arch_goldens import collect_probe_states

DEFAULT_BATCH_SIZES = (1, 32, 96, 160, 320, 480, 1024)

# The observation fields encode_batch marshals out of the state dicts, mirrored
# here so the host-side share of its cost can be timed on its own. Kept in sync
# with CardReasoningEncoder.encode_batch by hand; a drift only mis-attributes
# time between the marshal and device columns, it cannot change the total.
_HEADER_FIELDS = (
    "partner_mode",
    "is_leaster",
    "play_started",
    "current_trick",
    "alone_called",
    "called_under",
    "picker_rel",
    "partner_rel",
    "leader_rel",
    "picker_position",
)
_SCALAR_FIELDS = ("called_card_id", "picker_rel", "partner_rel")
_ID_FIELDS = (
    ("hand_ids", 8),
    ("blind_ids", 2),
    ("bury_ids", 2),
    ("trick_card_ids", 5),
    ("trick_is_picker", 5),
    ("trick_is_partner_known", 5),
)


def marshal_host(batch: list) -> dict:
    """The host-side half of encode_batch: turn a list of observation dicts into
    stacked CPU tensors, before any device transfer.

    This is one Python-level operation per state per field -- ~19 of them,
    six constructing a tensor -- so it costs a fixed amount per state and never
    amortizes with batch size. On a single machine it is invisible next to the
    forward. In a split design, where a fast host marshals and ships packed
    arrays to a remote accelerator, it stays on the host and must not be
    charged to the accelerator.
    """
    out = {}
    cols = [
        torch.as_tensor([int(s[key]) for s in batch], dtype=torch.float32).view(-1, 1)
        for key in _HEADER_FIELDS
    ]
    out["header"] = torch.cat(cols, dim=1)
    for key in _SCALAR_FIELDS:
        out[key] = torch.as_tensor([int(s[key]) for s in batch], dtype=torch.long)
    for key, width in _ID_FIELDS:
        stacked = torch.stack(
            [torch.as_tensor(s[key], dtype=torch.long) for s in batch], dim=0
        )
        out[key] = stacked.view(-1, width) if stacked.dim() == 1 else stacked
    return out


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def time_call(fn, device: torch.device, iters: int, warmup: int = 2) -> float:
    """Mean seconds per call, with the device drained before and after so that
    queued-but-unfinished work is not counted as free."""
    for _ in range(warmup):
        fn()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    synchronize(device)
    return (time.perf_counter() - start) / iters


def build_batch(states: list, batch: int) -> list:
    """A batch of ``batch`` observations, cycling the probe set as needed.

    Encoder cost is set by tensor shape, not content -- hands and tricks are
    padded to fixed widths and there is no data-dependent branching -- so
    cycling a small deterministic probe set measures the same thing as
    thousands of distinct states, reproducibly.
    """
    return [states[i % len(states)] for i in range(batch)]


def bench_device(
    agent,
    states: list,
    device: torch.device,
    batch_sizes: tuple,
    with_actor: bool,
    target_states: int,
) -> dict:
    agent.encoder.to(device)
    agent.actor.to(device)
    results = {}
    for batch in batch_sizes:
        observations = build_batch(states, batch)
        memory = torch.zeros((batch, agent.encoder.d_model), device=device)
        hand_ids = torch.as_tensor(
            np.stack([s["hand_ids"] for s in observations]),
            dtype=torch.long,
            device=device,
        )
        masks = torch.ones((batch, agent.action_size), dtype=torch.bool, device=device)

        def call():
            with torch.no_grad():
                encoded = agent.encoder.encode_batch(
                    observations, memory_in=memory, device=device
                )
                if with_actor:
                    agent.actor.forward_with_logits(
                        encoded, masks, hand_ids, agent.encoder.card
                    )

        iters = max(3, min(30, target_states // batch))
        total = time_call(call, device, iters)
        host = time_call(lambda obs=observations: marshal_host(obs), device, iters)
        results[batch] = {"total": total, "marshal": host, "device": total - host}
    return results


def print_table(per_device: dict, batch_sizes: tuple) -> None:
    for name, table in per_device.items():
        print(f"\n[{name}]")
        print(
            f"{'B':>6} | {'total ms':>9} {'marshal ms':>11} {'device ms':>10}"
            f" | {'total us/st':>12} {'device us/st':>13}"
        )
        print("-" * 72)
        for batch in batch_sizes:
            row = table[batch]
            print(
                f"{batch:>6} | {row['total'] * 1e3:9.2f} {row['marshal'] * 1e3:11.2f}"
                f" {row['device'] * 1e3:10.2f}"
                f" | {row['total'] / batch * 1e6:12.1f}"
                f" {row['device'] / batch * 1e6:13.1f}"
            )
    print(
        "\n  marshal = host-side Python packing of the observation dicts; it is"
        "\n  linear in B and never amortizes. Charge it to whichever machine"
        "\n  builds the batch: on one box read 'total us/st'; for a remote"
        "\n  accelerator fed pre-packed arrays read 'device us/st'."
    )


def print_aggregate(per_device: dict, batch_sizes: tuple, workers: int) -> None:
    """Aggregate states/s, which is what decides a shared-accelerator design:
    ``workers`` CPU processes run concurrently, but they would all serialize
    on one GPU."""
    if "cpu" not in per_device or len(per_device) < 2:
        return
    others = [n for n in per_device if n != "cpu"]
    print(f"\nAggregate throughput (states/s), assuming {workers} concurrent workers:")
    header = f"  {'B':>6} | {'cpu x' + str(workers):>14}"
    for name in others:
        header += f" | {name + ' x1':>14} | {name + ' x1 remote':>16}"
    print(header)
    for batch in batch_sizes:
        cpu_rate = workers * batch / per_device["cpu"][batch]["total"]
        row = f"  {batch:>6} | {cpu_rate:14,.0f}"
        for name in others:
            entry = per_device[name][batch]
            row += f" | {batch / entry['total']:14,.0f}"
            row += f" | {batch / entry['device']:16,.0f}"
        print(row)
    print(
        f"  x1        = that device doing everything, including marshalling.\n"
        f"  x1 remote = that device fed pre-packed arrays by a faster host\n"
        f"              (marshalling excluded -- the split-machine case).\n"
        f"  A device only pays off where its column beats cpu x{workers}."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="devices to compare (default: cpu plus mps/cuda if available)",
    )
    parser.add_argument("--arch", default="perceiver-shared-v2")
    parser.add_argument(
        "--batch-sizes", type=int, nargs="+", default=list(DEFAULT_BATCH_SIZES)
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="torch CPU threads; 1 matches a league worker (default)",
    )
    parser.add_argument(
        "--with-actor",
        action="store_true",
        help="also time the masked actor forward (encode_batch alone is ~96%% "
        "of committee-search inference time, so this is usually noise)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="concurrent worker count for the aggregate-throughput table",
    )
    parser.add_argument(
        "--target-states",
        type=int,
        default=6000,
        help="rough states per batch size; sets the iteration count",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)

    names = args.devices
    if names is None:
        names = ["cpu"]
        if torch.backends.mps.is_available():
            names.append("mps")
        if torch.cuda.is_available():
            names.append("cuda")

    from sheepshead.agent import ppo as ppo_module

    ppo_module.device = torch.device("cpu")  # build on CPU, move per device below
    from sheepshead.agent.ppo import PPOAgent
    from sheepshead.training.training_utils import set_all_seeds

    set_all_seeds(args.seed)
    agent = PPOAgent(action_size=len(ACTION_IDS), arch=args.arch)
    states = collect_probe_states()

    batch_sizes = tuple(args.batch_sizes)
    print(
        f"torch {torch.__version__}  arch={args.arch}  threads={torch.get_num_threads()}"
    )
    print(
        f"timing encode_batch{' + actor' if args.with_actor else ''} "
        f"over {len(states)} probe states (cycled)\n"
    )

    per_device = {}
    for name in names:
        device = torch.device(name)
        per_device[name] = bench_device(
            agent, states, device, batch_sizes, args.with_actor, args.target_states
        )

    print_table(per_device, batch_sizes)
    print_aggregate(per_device, batch_sizes, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
