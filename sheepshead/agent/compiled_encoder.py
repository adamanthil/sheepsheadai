"""Opt-in compiled encoder for the search path.

A global patch of ``CardReasoningEncoder.encode_batch``, and deliberately so.
The win is spread across four call sites in ``ismcts`` and the network round is
the largest but not the majority -- measured on a production committee,
``_run_network_round`` is 33.0% of wall, ``_observe_completers_merged`` 27.1%,
``_encode_seat_batched`` 9.9% and ``_observe_trick_lockstep`` 5.3%, for 75.3%
total. All four route through ``encode_batch``, so patching that one method
reaches every one of them; wrapping the round alone reaches a third of the work.

Measured effect, 8 concurrent committees at steady state on an M1 Max: 55.5s
CPU eager -> 40.7s MPS compiled, 1.36x. See
notebooks/Distributed_Inference_202608.md §5.5-§5.6.

**Never on by default.** Compiled output differs from eager by ~2.6e-08 on
probabilities, so ``capture_search_goldens`` cannot pass against it. This is an
opt-in throughput mode: fine for training, never for goldens, CRN panels or
eval.

Three properties are load-bearing rather than incidental:

* **Bucketed shapes.** ``dynamic=False`` specialises per input shape, and the
  search dispatches 44 distinct round sizes plus 49 observe sizes. Rounding up
  to a multiple of ``granularity`` collapses those to ~14 shapes at ~1.8%
  wasted rows. ``dynamic=True`` avoids recompiles but gives up most of the win
  (1.45-1.82x against 1.95-2.38x on the forward).
* **Steady state only.** The first committee pays compilation and is *slower*
  than eager. A training run does thousands; a short benchmark measures the
  compiler.
* **Enough shape slots.** dynamo compiles 8 variants per function by default and
  then silently falls back to eager, which is more shapes than bucketing leaves
  -- see ``allow_shape_specialisation``.
"""

import torch

from sheepshead.agent.encoder import CardReasoningEncoder

_ORIGINAL_ENCODE_BATCH = None

#: Bucketing still leaves more shapes than dynamo compiles by default.
_DEFAULT_SHAPE_BUDGET = 64


def allow_shape_specialisation(limit: int = _DEFAULT_SHAPE_BUDGET) -> None:
    """Raise dynamo's per-function recompile limit.

    It defaults to **8**, and on hitting it dynamo stops compiling and runs the
    function eager *without raising* -- it only logs. Bucketed search shapes
    exceed 8 (14 distinct at granularity 32), so the tail of the size
    distribution silently loses its compilation.

    Silently is the whole problem. The run still produces correct results, so
    nothing fails; only the throughput is worse than it looks like it should be,
    which is indistinguishable from the optimisation simply not helping much.
    """
    import torch._dynamo as dynamo

    for name in ("recompile_limit", "cache_size_limit"):
        if getattr(dynamo.config, name, limit) < limit:
            setattr(dynamo.config, name, limit)


def enable_compiled_encoder(granularity: int = 32, mode: str | None = None) -> None:
    """Route every ``encode_batch`` call through a compiled, shape-bucketed
    ``encode_tensors``. Idempotent.

    ``mode`` is passed to ``torch.compile``; ``None`` or ``"default"`` means
    plain inductor. ``"reduce-overhead"`` selects CUDA graphs on CUDA and is a
    no-op on MPS, where Metal has no equivalent.
    """
    global _ORIGINAL_ENCODE_BATCH
    if _ORIGINAL_ENCODE_BATCH is not None:
        return
    allow_shape_specialisation()
    _ORIGINAL_ENCODE_BATCH = CardReasoningEncoder.encode_batch
    granularity = max(1, int(granularity))
    kwargs = {} if mode in (None, "default") else {"mode": mode}
    graph = torch.compile(CardReasoningEncoder.encode_tensors, dynamic=False, **kwargs)

    def compiled_encode_batch(self, batch, memory_in=None, device=None):
        n = len(batch)
        marshalled = self.marshal_batch(batch, device)
        pad = -(-n // granularity) * granularity - n
        if pad:
            marshalled = {
                key: torch.cat([v, v[:1].repeat_interleave(pad, 0)], 0)
                for key, v in marshalled.items()
            }
            if memory_in is not None:
                memory_in = torch.cat(
                    [memory_in, memory_in[:1].repeat_interleave(pad, 0)], 0
                )
        out = graph(self, marshalled, memory_in=memory_in, device=device)
        # Slice the pad rows off before anything downstream can see them. The
        # caller indexes results positionally against its own state list.
        return {k: (v[:n] if torch.is_tensor(v) else v) for k, v in out.items()}

    CardReasoningEncoder.encode_batch = compiled_encode_batch


def disable_compiled_encoder() -> None:
    """Restore the eager method. Exists so a process can go back to a
    golden-comparable state without restarting."""
    global _ORIGINAL_ENCODE_BATCH
    if _ORIGINAL_ENCODE_BATCH is None:
        return
    CardReasoningEncoder.encode_batch = _ORIGINAL_ENCODE_BATCH
    _ORIGINAL_ENCODE_BATCH = None
