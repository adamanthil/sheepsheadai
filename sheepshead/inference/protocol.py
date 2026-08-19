"""Wire format for one remote search round.

Byte count is the binding constraint, not CPU: on the measured setup the GPU
does ~7 us/state of real work while the link costs 2-24 us/state depending on
encoding (addendum 2, B3). So the format is packed rather than convenient --
uint8 card ids, bit-packed action masks, and fp16 recurrent memory.

Per-state budget, "packed" mode:

    up    39 B observation + 14 B mask + 512 B memory (fp16 x 256)  = 565 B
    down  220 B probs (fp16 x 110) + 2 B value + 512 B memory_out   = 734 B

The memory is 91% of the traffic. A ``memory_resident`` mode that keeps the GRU
state on the server and ships permutation indices instead would cut this to
~39 B up / 222 B down, which on a gigabit link is the difference between ~2.3x
and ~3.2x end to end. It is not implemented here: ``_run_network_round`` writes
``memory_out`` back into client-side sim state, so eliding it is a restructuring
of the search rather than a change of wire format. The split is marked in
``pack_request``/``pack_response`` so a v2 can lift it out.

Frames are length-prefixed. Arrays are little-endian and written in a fixed
order, so both sides agree without a schema negotiation.
"""

import hashlib
import struct

import numpy as np
import torch

PROTOCOL_VERSION = 1
_MAGIC = b"SHPI"
_HEADER = struct.Struct("<4sHHIIH")  # magic, version, flags, n, action_size, dtype
FLAG_WANTS_CRITIC = 1 << 0

# Field order on the wire. Must match on both ends; the version guards drift.
_OBS_ID_FIELDS = (
    ("hand_ids", 8),
    ("blind_ids", 2),
    ("bury_ids", 2),
    ("trick_card_ids", 5),
    ("trick_is_picker", 5),
    ("trick_is_partner_known", 5),
)
_OBS_SCALAR_FIELDS = ("called_card_id", "picker_rel", "partner_rel")
_HEADER_WIDTH = 10


class WireConfig:
    """Payload precision. fp16 halves the memory and probability traffic, which
    is the dominant term; use fp32 when comparing search fidelity against the
    local backend, where the quantisation would otherwise be attributed to the
    remote path itself."""

    def __init__(self, half: bool = True):
        self.half = bool(half)

    @property
    def np_dtype(self):
        return np.float16 if self.half else np.float32

    @property
    def code(self) -> int:
        return 16 if self.half else 32

    @staticmethod
    def from_code(code: int) -> "WireConfig":
        return WireConfig(half=(code == 16))


def fingerprint_weights(module: torch.nn.Module) -> str:
    """Stable digest of a module's parameters.

    The client sends this at handshake and the server refuses a mismatch.
    Serving stale weights would not raise anywhere -- it would silently produce
    a plausible, wrong search target -- so this is the one check that has to
    exist before any of this is trusted with a training run.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _bitpack_masks(valid_lists, action_size: int) -> np.ndarray:
    """(n, ceil(A/8)) bit-packed legal-action masks. 14 B/state at A=110
    instead of 110 B as bytes or 440 B as a float tensor."""
    dense = np.zeros((len(valid_lists), action_size), dtype=bool)
    for row, valid in enumerate(valid_lists):
        for action_id in valid:
            dense[row, int(action_id) - 1] = True
    return np.packbits(dense, axis=1)


def _unpack_masks(packed: np.ndarray, action_size: int) -> np.ndarray:
    return np.unpackbits(packed, axis=1)[:, :action_size].astype(bool)


def pack_request(
    marshalled: dict,
    memory_in: torch.Tensor,
    valid_lists,
    action_size: int,
    wants_critic: bool,
    wire: WireConfig,
) -> bytes:
    """Serialize one round. ``marshalled`` is CardReasoningEncoder.marshal_batch
    output on CPU -- the orchestrator marshals, never the accelerator host."""
    n = int(memory_in.shape[0])
    flags = FLAG_WANTS_CRITIC if wants_critic else 0
    parts = [_HEADER.pack(_MAGIC, PROTOCOL_VERSION, flags, n, action_size, wire.code)]

    header = marshalled["header_scalar"].detach().cpu().numpy()
    parts.append(np.ascontiguousarray(header, dtype=np.float32).tobytes())
    for field in _OBS_SCALAR_FIELDS:
        col = marshalled[field].detach().cpu().numpy()
        parts.append(np.ascontiguousarray(col, dtype=np.uint8).tobytes())
    for field, _width in _OBS_ID_FIELDS:
        arr = marshalled[field].detach().cpu().numpy()
        parts.append(np.ascontiguousarray(arr, dtype=np.uint8).tobytes())

    parts.append(_bitpack_masks(valid_lists, action_size).tobytes())

    # --- memory: 91% of the payload; a memory-resident v2 elides this block.
    mem = memory_in.detach().cpu().numpy()
    parts.append(np.ascontiguousarray(mem, dtype=wire.np_dtype).tobytes())
    return b"".join(parts)


def unpack_request(buf: bytes, d_model: int) -> dict:
    """Inverse of pack_request. Returns tensors ready for encode_tensors --
    no observation dicts are reconstructed, which is the point."""
    magic, version, flags, n, action_size, dtype_code = _HEADER.unpack_from(buf, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad magic {magic!r}")
    if version != PROTOCOL_VERSION:
        raise ValueError(f"protocol version {version} != {PROTOCOL_VERSION}")
    wire = WireConfig.from_code(dtype_code)
    offset = _HEADER.size

    def take(count: int, dtype) -> np.ndarray:
        nonlocal offset
        itemsize = np.dtype(dtype).itemsize
        raw = np.frombuffer(buf, dtype=dtype, count=count, offset=offset)
        offset += count * itemsize
        return raw

    marshalled = {
        "header_scalar": torch.from_numpy(
            take(n * _HEADER_WIDTH, np.float32).reshape(n, _HEADER_WIDTH).copy()
        )
    }
    for field in _OBS_SCALAR_FIELDS:
        marshalled[field] = torch.from_numpy(take(n, np.uint8).copy()).long()
    for field, width in _OBS_ID_FIELDS:
        marshalled[field] = torch.from_numpy(
            take(n * width, np.uint8).reshape(n, width).copy()
        ).long()

    mask_bytes = (action_size + 7) // 8
    packed_masks = take(n * mask_bytes, np.uint8).reshape(n, mask_bytes)
    masks = _unpack_masks(packed_masks, action_size)

    mem = take(n * d_model, wire.np_dtype).reshape(n, d_model)
    memory_in = torch.from_numpy(mem.astype(np.float32).copy())

    return {
        "marshalled": marshalled,
        "memory_in": memory_in,
        "masks": torch.from_numpy(masks.copy()),
        "action_size": action_size,
        "wants_critic": bool(flags & FLAG_WANTS_CRITIC),
        "wire": wire,
    }


def pack_response(
    probs: np.ndarray,
    values: np.ndarray,
    memory_out: np.ndarray,
    wire: WireConfig,
) -> bytes:
    parts = [
        np.ascontiguousarray(probs, dtype=wire.np_dtype).tobytes(),
        np.ascontiguousarray(values, dtype=wire.np_dtype).tobytes(),
        # --- memory_out: elided by a memory-resident v2.
        np.ascontiguousarray(memory_out, dtype=wire.np_dtype).tobytes(),
    ]
    return b"".join(parts)


def unpack_response(
    buf: bytes, n: int, action_size: int, d_model: int, wire: WireConfig
) -> tuple:
    dtype = wire.np_dtype
    itemsize = np.dtype(dtype).itemsize
    offset = 0
    probs = (
        np.frombuffer(buf, dtype=dtype, count=n * action_size, offset=offset)
        .reshape(n, action_size)
        .astype(np.float32)
    )
    offset += n * action_size * itemsize
    values = np.frombuffer(buf, dtype=dtype, count=n, offset=offset).astype(np.float32)
    offset += n * itemsize
    memory_out = (
        np.frombuffer(buf, dtype=dtype, count=n * d_model, offset=offset)
        .reshape(n, d_model)
        .astype(np.float32)
    )
    return probs, values, memory_out


def request_nbytes(n: int, action_size: int, d_model: int, wire: WireConfig) -> int:
    """Exact request size, for the payload accounting in the benchmarks."""
    per_state_obs = _HEADER_WIDTH * 4 + len(_OBS_SCALAR_FIELDS)
    per_state_obs += sum(width for _f, width in _OBS_ID_FIELDS)
    mask_bytes = (action_size + 7) // 8
    mem_bytes = d_model * np.dtype(wire.np_dtype).itemsize
    return _HEADER.size + n * (per_state_obs + mask_bytes + mem_bytes)


def response_nbytes(n: int, action_size: int, d_model: int, wire: WireConfig) -> int:
    itemsize = np.dtype(wire.np_dtype).itemsize
    return n * (action_size + 1 + d_model) * itemsize
