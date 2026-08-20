#!/usr/bin/env python3
"""Backend op-coverage and host-sync audit for the agent's hot paths.

Answers two questions about running the agent on a non-CPU backend:

  * COVERAGE — does every aten op we dispatch have a native kernel on the
    target backend? Ops registered only for CPU/CUDA fall through to the
    backend's *fallback*, which copies every operand to host memory, runs on
    CPU, and copies back. On MPS that fallback is registered unconditionally
    but gated at runtime by PYTORCH_ENABLE_MPS_FALLBACK: with the variable
    unset an uncovered op raises NotImplementedError, with it set the op costs
    two device transfers plus a pipeline stall. So an audit needs both a
    dispatcher-table classification (this tool) and a run with the variable
    unset (which simply completing proves nothing fell back).

  * HOST SYNCS — how often does a path force a device-to-host round trip?
    ``aten::_local_scalar_dense`` is .item()/bool()/int() on a tensor,
    including the implicit bool() of a 0-d tensor used in a Python ``if``;
    ``aten::nonzero`` is boolean-mask indexing. These are not fallbacks, but
    they serialize the command queue exactly the same way, and they are
    invisible on CPU where a "transfer" is free. ``--attribute`` maps each one
    back to the source line that caused it.

Op coverage and sync counts are dispatcher-level facts, independent of machine
load -- which makes this tool usable on a busy box. For wall-clock, which is
not, see bench_inference_device.

Usage:
    uv run python -m sheepshead.analysis.device_op_audit --device cpu
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
        uv run python -m sheepshead.analysis.device_op_audit --device mps
    uv run python -m sheepshead.analysis.device_op_audit --device mps --attribute

Run the MPS pass with PYTORCH_ENABLE_MPS_FALLBACK=1 so that an uncovered op
enumerates in the report instead of aborting the run at the first one; then
re-run without it to confirm the paths complete, which is the direct proof
that nothing fell back.

Findings from the 2026-08-18 pass are recorded in
notebooks/Throughput_Profiling_Notes.md (addendum sections A1/A2).
"""

import argparse
import collections
import json
import os
import sys
import traceback

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from sheepshead import ACTION_IDS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD

# Ops that force a device-to-host round trip. Not exhaustive over aten, but
# these are the ones reachable from our forward/backward/update paths.
SYNC_OPS = frozenset(
    {
        "aten::_local_scalar_dense",  # .item()/float()/int()/bool() on a tensor
        "aten::nonzero",  # boolean-mask indexing, .nonzero()
        "aten::masked_select",
        "aten::_assert_scalar",
        "aten::item",
    }
)

_COPY_OPS = frozenset({"aten::_to_copy", "aten::copy_", "aten::to", "aten::copy"})

# Backend-agnostic alias keys. A kernel registered under any of these runs on
# every backend -- either by decomposing into other ops (which this tool
# classifies on their own) or by being written against a generic iterator.
_COMPOSITE_KEYS = frozenset(
    {
        "CompositeImplicitAutograd",
        "CompositeExplicitAutograd",
        "CompositeExplicitAutogradNonFunctional",
    }
)


def classify_backend_support(op_name: str, backend: str) -> str:
    """Return "native" | "composite" | "fallback" | "unknown" for an aten op.

    Note that torch._C._dispatch_has_computed_kernel_for_dispatch_key is NOT
    usable here: MPS registers a backend fallback unconditionally, so the
    computed table reports a kernel for every op. This reads the raw dispatch
    table instead and looks for a real backend entry or a Composite alias.
    """
    try:
        dump = torch._C._dispatch_dump(op_name)
    except RuntimeError:
        return "unknown"
    if not dump:
        return "unknown"
    keys = {line.split(":", 1)[0].strip() for line in dump.splitlines() if ":" in line}
    if any(k == backend or k.startswith(backend) for k in keys):
        return "native"
    if any(k.split("[")[0] in _COMPOSITE_KEYS for k in keys):
        return "composite"
    return "fallback"


def _tensors(obj):
    if isinstance(obj, torch.Tensor):
        yield obj
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            yield from _tensors(item)


def _device_str(x) -> str:
    return str(x.device) if isinstance(x, torch.Tensor) else "?"


class OpTracer(TorchDispatchMode):
    """Count every aten op dispatched inside the block, plus cross-device
    transfers and (optionally) the source line behind each host sync."""

    def __init__(self, attribute: bool = False, source_root: str = "sheepshead"):
        self.counts = collections.Counter()
        self.transfers = collections.Counter()  # (src, dst) -> n
        self.float64 = collections.Counter()
        self.attribution = collections.Counter()
        self.attribute = attribute
        self.source_root = os.path.join(source_root, "")

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func.name()
        self.counts[name] += 1

        if name in _COPY_OPS:
            self._record_transfer(name, args, kwargs)
        if self.attribute and name in SYNC_OPS:
            self.attribution[(name, self._caller())] += 1

        out = func(*args, **kwargs)

        for tensor in _tensors(out):
            if tensor.dtype in (torch.float64, torch.complex128):
                self.float64[f"{name}:{tensor.dtype}"] += 1
        return out

    def _record_transfer(self, name, args, kwargs) -> None:
        if not args:
            return
        src = _device_str(args[0])
        if name == "aten::copy_" and len(args) > 1:
            src, dst = _device_str(args[1]), _device_str(args[0])
        elif kwargs.get("device") is not None:
            dst = str(kwargs["device"])
        elif len(args) > 1 and isinstance(args[1], torch.Tensor):
            dst = _device_str(args[1])
        else:
            return
        if src != dst:
            self.transfers[(src, dst)] += 1

    def _caller(self) -> str:
        """Innermost repo frame, so a sync is attributed to our code rather
        than to whatever torch internal happened to materialize it."""
        frames = [
            f
            for f in traceback.extract_stack()[:-2]
            if self.source_root in f.filename and ".venv" not in f.filename
        ]
        if not frames:
            return "<outside the package>"
        frame = frames[-1]
        where = frame.filename.split(self.source_root, 1)[-1]
        return f"{self.source_root}{where}:{frame.lineno}  {(frame.line or '').strip()}"

    def sync_breakdown(self) -> dict:
        return {name: n for name, n in self.counts.items() if name in SYNC_OPS}

    def sync_total(self) -> int:
        return sum(self.sync_breakdown().values())


def collect_rollout_events(agent, episodes: int, seed: int) -> None:
    """Fill ``agent``'s event buffer with seeded self-play, so the update path
    has a realistic buffer to run over. Deals are seeded explicitly: an
    unseeded Game() draws from OS entropy and the audit stops being
    reproducible.

    LAYERING NOTE: the package docstring forbids analysis modules from
    importing ``pfsp_runtime``, and this is a deliberate exception (shared with
    diagnostics/critic_stratified_ev.py, which imports the same function, and
    with calibrate_shrinkage / called_suit_exceptions / verify_entropy_baseline
    / search_help_matrix, which import other pfsp_runtime primitives). The
    reason is fidelity: what this tool measures is which aten ops the update
    path dispatches, and a hand-rolled rollout loop would omit the optional aux
    labels (win / final_return / seen_trump_mask / unseen_trump_higher), whose
    absence silently removes the aux-head losses from the audited op set. An
    under-reporting coverage audit is worse than a layering exception."""
    from types import SimpleNamespace

    from sheepshead.training import pfsp_runtime

    opponents = [
        SimpleNamespace(agent=agent, metadata=SimpleNamespace(agent_id="self"))
        for _ in range(4)
    ]
    for episode in range(episodes):
        _, events, _, _, _ = pfsp_runtime.play_population_game(
            training_agent=agent,
            opponents=opponents,
            partner_mode=PARTNER_BY_CALLED_ACE if episode % 2 == 0 else PARTNER_BY_JD,
            training_agent_position=(episode % 5) + 1,
            reward_mode="terminal",
            game_seed=seed + episode,
        )
        agent.store_episode_events(events)


def action_states(agent) -> tuple[list, list]:
    """(state dicts, valid-action-id lists) for every action event in the
    agent's buffer."""
    states, valid_lists = [], []
    for event in agent.events:
        if event.get("kind") != "action":
            continue
        mask = event["mask"]
        if not isinstance(mask, torch.Tensor):
            mask = torch.as_tensor(mask)
        states.append(event["state"])
        valid_lists.append(
            [int(i) + 1 for i in torch.nonzero(mask.flatten()).flatten()]
        )
    return states, valid_lists


# ---------------------------------------------------------------------------
# The audited paths
# ---------------------------------------------------------------------------


def path_act(agent, states, valid_lists, device, batch: int, calls: int = 8) -> None:
    """Single-state inference -- the self-play rollout hot path."""
    for i in range(min(calls, len(states))):
        agent.act(states[i], valid_lists[i], player_id=(i % 5) + 1)


def path_batched_inference(agent, states, valid_lists, device, batch: int) -> None:
    """Batched encode + masked actor forward + critic -- the shape the ISMCTS
    committee runs (mirrors ismcts._encode_seat_batched / _masked_actor_probs)."""
    states, valid_lists = states[:batch], valid_lists[:batch]
    memory = torch.zeros((len(states), agent.encoder.d_model), device=device)
    with torch.no_grad():
        encoded = agent.encoder.encode_batch(states, memory_in=memory, device=device)
        masks = torch.stack(
            [agent.get_action_mask(v, agent.action_size) for v in valid_lists]
        ).to(device)
        hand_ids = torch.as_tensor(
            np.stack([s["hand_ids"] for s in states]), dtype=torch.long, device=device
        )
        agent.actor.forward_with_logits(encoded, masks, hand_ids, agent.encoder.card)
        agent.critic(encoded)


def path_update(agent, states, valid_lists, device, batch: int) -> None:
    """One PPO update: forward, backward, optimizer step."""
    agent.update(epochs=1, batch_size=32)


def audit_path(fn, agent, states, valid_lists, device, batch, backend, attribute):
    agent.clear_player_memories()
    tracer = OpTracer(attribute=attribute)
    try:
        with tracer:
            fn(agent, states, valid_lists, device, batch)
    except NotImplementedError as exc:
        return {"error": str(exc)}

    rows = [
        (name, n, classify_backend_support(name, backend))
        for name, n in tracer.counts.most_common()
    ]
    return {
        "distinct_ops": len(rows),
        "total_dispatches": sum(r[1] for r in rows),
        "fallback_ops": {r[0]: r[1] for r in rows if r[2] == "fallback"},
        "unknown_ops": [r[0] for r in rows if r[2] == "unknown"],
        "sync_ops": tracer.sync_breakdown(),
        "sync_total": tracer.sync_total(),
        "sync_attribution": {
            f"{op} @ {loc}": n for (op, loc), n in tracer.attribution.most_common()
        },
        "transfers": {f"{s}->{d}": n for (s, d), n in tracer.transfers.items()},
        "float64": dict(tracer.float64),
        "ops": {r[0]: {"calls": r[1], "kind": r[2]} for r in rows},
    }


def print_path_report(label: str, result: dict, top: int = 15) -> None:
    print("=" * 78)
    print(f"### {label}")
    if "error" in result:
        print("    RAISED NotImplementedError (no kernel on this backend):")
        print(f"    {result['error']}\n")
        return
    fallback = result["fallback_ops"]
    print(f"    distinct aten ops : {result['distinct_ops']}")
    print(f"    total dispatches  : {result['total_dispatches']}")
    print(
        f"    FALLBACK ops      : {len(fallback)}  ({sum(fallback.values())} calls)"
        + ("" if fallback else "   <-- clean")
    )
    for name, n in fallback.items():
        print(f"        !! {name:52s} x{n}")
    if result["unknown_ops"]:
        print(f"    unclassified      : {result['unknown_ops']}")
    print(f"    host syncs        : {result['sync_total']}   {result['sync_ops']}")
    for loc, n in result["sync_attribution"].items():
        print(f"        {n:4d}  {loc}")
    if result["transfers"]:
        print("    device transfers  :")
        for pair, n in sorted(result["transfers"].items(), key=lambda kv: -kv[1]):
            print(f"        {pair:28s} x{n}")
    if result["float64"]:
        print(f"    float64 producers : {result['float64']}")
    print("    top ops by call count:")
    for name, info in list(result["ops"].items())[:top]:
        flag = "  <== FALLBACK" if info["kind"] == "fallback" else ""
        print(f"        {info['calls']:7d}  {name:52s} [{info['kind']}]{flag}")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--arch", default="perceiver-shared-v2")
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--batch", type=int, default=64, help="batched-inference size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attribute",
        action="store_true",
        help="map each host sync back to the source line that caused it",
    )
    parser.add_argument("--threads", type=int, default=1, help="torch CPU threads")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    backend = {"cpu": "CPU", "mps": "MPS", "cuda": "CUDA"}[args.device]

    # PPOAgent places its networks on a module-level device global. Override it
    # before building the agent; the ISMCTS teacher then follows the networks.
    from sheepshead.agent import ppo as ppo_module

    ppo_module.device = device
    from sheepshead.agent.ppo import PPOAgent
    from sheepshead.training.training_utils import set_all_seeds

    print(f"torch {torch.__version__}  device={device}  arch={args.arch}")
    print(f"threads={torch.get_num_threads()}")
    print(
        f"PYTORCH_ENABLE_MPS_FALLBACK={os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')}"
    )
    print(
        f"MPS backend fallback registered: {torch._C._dispatch_has_backend_fallback('MPS')}"
    )
    print()

    set_all_seeds(args.seed)
    agent = PPOAgent(action_size=len(ACTION_IDS), arch=args.arch)
    collect_rollout_events(agent, args.episodes, args.seed)
    states, valid_lists = action_states(agent)
    print(f"collected {len(states)} action states, {len(agent.events)} events\n")

    paths = [
        ("act(single-state rollout)", path_act),
        (f"batched-inference(B={args.batch})", path_batched_inference),
        ("update(fwd+bwd+step)", path_update),
    ]
    results = {
        "device": str(device),
        "backend": backend,
        "arch": args.arch,
        "torch": torch.__version__,
        "paths": {},
    }
    for label, fn in paths:
        result = audit_path(
            fn,
            agent,
            states,
            valid_lists,
            device,
            args.batch,
            backend,
            args.attribute,
        )
        print_path_report(label, result)
        results["paths"][label] = result

    total_fallback = sum(
        len(p.get("fallback_ops", {})) for p in results["paths"].values()
    )
    print(
        f"VERDICT: {total_fallback} fallback op(s) across {len(paths)} paths on {backend}."
    )

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(results, handle, indent=2)
            handle.write("\n")
        print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
