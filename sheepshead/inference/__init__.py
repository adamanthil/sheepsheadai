"""Remote inference for the ISMCTS search teacher (prototype).

Splits a search round across two machines: the orchestrator runs the game,
the tree and the Python-side marshalling, and an accelerator host runs the
network forward. The seam is ``ISMCTSTeacher._run_network_round``, which takes
an ordered request list, contains no RNG, and returns plain arrays.

Layering: this package may import the agent, and ``ismcts`` imports it. It must
never import ``ismcts`` back.

Backends are pluggable and the in-process one is the default. That is a hard
requirement, not a convenience: cross-worker batch composition changes GEMM
tiling, so a server-backed run can never be bit-reproducible, and the
capture_search_goldens gate only means anything against the local path.

See notebooks/Throughput_Profiling_Notes.md addendum 2 for the measurements
that motivate the design, in particular why marshalling must stay on the
orchestrator and why the wire byte count is the binding constraint.
"""

from sheepshead.inference.backend import (
    InferenceBackend,
    LocalBackend,
    RemoteBackend,
    RoundResult,
    masked_actor_probs,
)
from sheepshead.inference.protocol import (
    WireConfig,
    fingerprint_weights,
    pack_request,
    pack_response,
    unpack_request,
    unpack_response,
)

__all__ = [
    "InferenceBackend",
    "LocalBackend",
    "RemoteBackend",
    "RoundResult",
    "masked_actor_probs",
    "WireConfig",
    "fingerprint_weights",
    "pack_request",
    "pack_response",
    "unpack_request",
    "unpack_response",
]
