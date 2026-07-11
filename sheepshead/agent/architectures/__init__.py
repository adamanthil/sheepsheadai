"""Architecture registry for PPO network variants.

Each named ``ArchitectureSpec`` bundles the three network factories
(encoder / actor / critic) that ``PPOAgent`` uses to build itself. The
default ``full`` spec constructs byte-for-byte the same networks as the
pre-registry code, so existing checkpoints and training behavior are
unchanged; every other spec is an ablation rung or a future variant.

The registry exists to support controlled architecture ablations: each
adjacent rung of the ladder removes exactly one historical addition
(auxiliary heads, transformer card reasoning, informed embedding
initialization, the card-token pipeline itself), so paired training runs
measure that component's contribution directly. Multi-seed replication is
required for any conclusion — deep-RL comparisons are notoriously
seed-sensitive (Henderson et al. 2018, "Deep Reinforcement Learning that
Matters", arXiv:1709.06560), and controlled/equal-footing evaluation is the
difference between measuring architectures and measuring tuning effort
(Melis et al. 2018, arXiv:1707.05589).

Adding a variant = one ``register(ArchitectureSpec(...))`` call; the spec's
``name`` field is the registry key, and the trainers, checkpoint metadata
(``arch`` key), and ``ppo.load_agent`` pick it up by name. Workers and
subprocesses receive the architecture *name*, never the spec object.
"""

from .encoders import (
    PerceiverCtxMemEncoder,
    PerceiverEncoder,
    PooledMemoryEncoder,
    SharedReadoutEncoder,
    TokenReadEncoder,
)
from .onehot import (
    ONEHOT_STATE_DIM,
    FlatHeadActorNetwork,
    OneHotFeedForwardEncoder,
    build_onehot_state,
)
from .registry import (
    ARCHITECTURES,
    ActorMappings,
    ArchitectureSpec,
    available_architectures,
    get_spec,
    register,
)

__all__ = [
    "ARCHITECTURES",
    "ActorMappings",
    "ArchitectureSpec",
    "available_architectures",
    "get_spec",
    "register",
    "PooledMemoryEncoder",
    "TokenReadEncoder",
    "PerceiverEncoder",
    "PerceiverCtxMemEncoder",
    "SharedReadoutEncoder",
    "ONEHOT_STATE_DIM",
    "build_onehot_state",
    "OneHotFeedForwardEncoder",
    "FlatHeadActorNetwork",
]
