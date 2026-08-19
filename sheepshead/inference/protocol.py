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
from typing import NamedTuple

import numpy as np
import torch

PROTOCOL_VERSION = 2  # v2: fused response block (probs | value | memory_out)
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
#: Total uint8 observation bytes per state; the whole region is contiguous on
#: the wire, which is what lets the server move it in one transfer.
_IDS_WIDTH = len(_OBS_SCALAR_FIELDS) + sum(w for _f, w in _OBS_ID_FIELDS)
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


def _unpack_bits(packed: torch.Tensor, action_size: int) -> torch.Tensor:
    """Bit-unpack masks wherever the packed tensor already lives, so the masks
    never make their own trip across the bus."""
    shifts = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    bits = (packed.unsqueeze(-1) >> shifts) & 1
    return bits.reshape(packed.shape[0], -1)[:, :action_size].bool()


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


def _ids_layout(n: int) -> list:
    """Byte spans of each observation field inside an ``n``-row ``ids_flat``.

    The region is *field-major*: all n ``called_card_id`` bytes, then all n
    ``picker_rel`` bytes, and so on. That layout is what lets the whole thing
    cross the bus in one transfer and be sliced for free on the far side -- but
    it also means merging several requests has to interleave at field
    granularity rather than concatenating whole regions end to end.

    Yields ``(field, width, start, stop)``; ``width`` is None for the scalar
    columns, which are (n,) rather than (n, width).
    """
    spans = []
    cursor = 0
    for field in _OBS_SCALAR_FIELDS:
        spans.append((field, None, cursor, cursor + n))
        cursor += n
    for field, width in _OBS_ID_FIELDS:
        spans.append((field, width, cursor, cursor + n * width))
        cursor += n * width
    return spans


class RawRequest(NamedTuple):
    """One decoded request, still on the host and still flat.

    Decoding deliberately stops short of reconstructing per-field tensors: the
    server merges several of these into a single forward, and the merge is
    cheapest while the observation bytes are still one contiguous field-major
    region. ``merge_requests`` is what turns a list of these into something
    ``encode_tensors`` can consume.
    """

    n: int
    action_size: int
    wants_critic: bool
    wire: WireConfig
    header: torch.Tensor  # (n, 10) float32
    ids_flat: torch.Tensor  # (n * _IDS_WIDTH,) uint8, field-major
    packed_masks: torch.Tensor  # (n, ceil(A/8)) uint8
    memory_in: torch.Tensor  # (n, d_model) float32

    @property
    def key(self) -> tuple:
        """Requests merge only within a key. The response block's column count
        and dtype have to agree across the whole batch for it to cross back in
        one copy, and both are fixed by these two fields."""
        return (self.action_size, self.wire.code)


def decode_request(buf: bytes, d_model: int) -> RawRequest:
    """Parse one request frame into host-side flat arrays. No transfers here --
    placing the batch on a device is ``merge_requests``' job, because it can do
    it once for a merged batch instead of once per client."""
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

    header = take(n * _HEADER_WIDTH, np.float32).reshape(n, _HEADER_WIDTH)
    ids_flat = take(n * _IDS_WIDTH, np.uint8)
    mask_bytes = (action_size + 7) // 8
    packed = take(n * mask_bytes, np.uint8).reshape(n, mask_bytes)
    mem = take(n * d_model, wire.np_dtype).reshape(n, d_model)

    return RawRequest(
        n=n,
        action_size=action_size,
        wants_critic=bool(flags & FLAG_WANTS_CRITIC),
        wire=wire,
        # .copy() throughout: np.frombuffer views immutable bytes, and torch
        # refuses to take ownership of a read-only array without warning.
        header=torch.from_numpy(header.copy()),
        ids_flat=torch.from_numpy(ids_flat.copy()),
        packed_masks=torch.from_numpy(packed.copy()),
        # astype already copies, and this is the largest array by an order of
        # magnitude -- a trailing .copy() would double the memcpy for nothing.
        memory_in=torch.from_numpy(mem.astype(np.float32)),
    )


def merge_requests(raws: list, device=None) -> dict:
    """Combine K decoded requests into one batch ready for ``encode_tensors``.

    Two things happen here, and both are about transfer count.

    The batch crosses to the accelerator in **four transfers regardless of K**:
    header, the whole uint8 observation region, the packed masks, and the
    recurrent memory. Naively moving each request separately would be four per
    client; reconstructing per-field tensors first would be ten per batch. On
    MPS every small copy carries command-buffer overhead -- the unfused original
    spent ~20 ms/round on transfer bookkeeping against a ~9.6 ms forward -- and
    CUDA hides this only partly.

    And merging at all is the point of the server rather than a refinement of
    it: ~94% of the measured per-round server cost was fixed overhead, so eight
    clients' rounds in one forward amortize it eightfold. See
    notebooks/Distributed_Inference_202608.md §5.

    Returns the same dict shape as a single-request unpack, plus ``rows`` --
    the per-request row counts the caller needs to split the results back up.
    """
    if not raws:
        raise ValueError("merge_requests needs at least one request")
    key = raws[0].key
    for other in raws[1:]:
        if other.key != key:
            raise ValueError(
                f"cannot merge requests with different (action_size, dtype): "
                f"{key} vs {other.key}"
            )
    action_size = raws[0].action_size
    n = sum(raw.n for raw in raws)

    def to_device(tensor):
        return tensor.to(device) if device is not None else tensor

    if len(raws) == 1:
        # Fast path, and not only an optimization: it keeps the single-client
        # cost identical to the pre-batching server, so a one-worker run is not
        # taxed for a feature it is not using.
        header = to_device(raws[0].header)
        ids_flat = to_device(raws[0].ids_flat)
        packed = to_device(raws[0].packed_masks)
        memory_in = to_device(raws[0].memory_in)
    else:
        header = to_device(torch.cat([raw.header for raw in raws], dim=0))
        # Interleave field-by-field so the merged region keeps the field-major
        # layout _ids_layout describes; concatenating whole regions would put
        # request 1's hand_ids where request 0's trick_card_ids belong.
        layouts = [_ids_layout(raw.n) for raw in raws]
        pieces = []
        for column in range(len(layouts[0])):
            for raw, layout in zip(raws, layouts):
                _field, _width, start, stop = layout[column]
                pieces.append(raw.ids_flat[start:stop])
        ids_flat = to_device(torch.cat(pieces))
        packed = to_device(torch.cat([raw.packed_masks for raw in raws], dim=0))
        memory_in = to_device(torch.cat([raw.memory_in for raw in raws], dim=0))

    # Slice the uint8 region into fields. Device-side and free -- no transfers.
    marshalled = {"header_scalar": header}
    for field, width, start, stop in _ids_layout(n):
        chunk = ids_flat[start:stop]
        marshalled[field] = (
            chunk.long() if width is None else chunk.view(n, width).long()
        )

    return {
        "marshalled": marshalled,
        "memory_in": memory_in,
        "masks": _unpack_bits(packed, action_size),
        "action_size": action_size,
        # Any client asking for the critic makes it one extra head on a forward
        # that is happening anyway; the server zeroes the column back out for
        # the clients that did not ask.
        "wants_critic": any(raw.wants_critic for raw in raws),
        "wire": raws[0].wire,
        "rows": [raw.n for raw in raws],
    }


def unpack_request(buf: bytes, d_model: int, device=None) -> dict:
    """Inverse of pack_request: the batch-of-one case of ``merge_requests``."""
    return merge_requests([decode_request(buf, d_model)], device=device)


def response_block(probs, values, memory_out, wire: WireConfig) -> np.ndarray:
    """One round's results as a writable, C-contiguous (n, A + 1 + d_model) host
    array.

    Accepts torch tensors or numpy arrays. Given device tensors it concatenates
    on the device and crosses the bus **once** -- three separate ``.cpu()``
    calls would be three full pipeline drains, which on MPS cost ~688 us each
    and dominated the server's per-round budget.

    Returning the array rather than bytes is what lets a batching server hand
    each client its own row range without a second copy per client: a row slice
    of a C-contiguous array is itself contiguous, so ``.tobytes()`` on it is
    exactly that client's response.

    The column layout (probs | value | memory_out) is why this is v2 of the
    protocol: v1 shipped three separate blocks, which cannot be fused.
    """
    if isinstance(probs, torch.Tensor):
        block = torch.cat(
            [
                probs,
                values.reshape(-1, 1),
                # --- memory_out: elided by a memory-resident v3.
                memory_out,
            ],
            dim=1,
        )
        dtype = torch.float16 if wire.half else torch.float32
        return block.to(dtype).cpu().numpy()

    block = np.concatenate(
        [
            np.asarray(probs, dtype=np.float32),
            np.asarray(values, dtype=np.float32).reshape(-1, 1),
            np.asarray(memory_out, dtype=np.float32),
        ],
        axis=1,
    )
    return np.ascontiguousarray(block, dtype=wire.np_dtype)


def pack_response(probs, values, memory_out, wire: WireConfig) -> bytes:
    """Serialize one round's results. See ``response_block``."""
    return response_block(probs, values, memory_out, wire).tobytes()


def unpack_response(
    buf: bytes, n: int, action_size: int, d_model: int, wire: WireConfig
) -> tuple:
    """Split the single (n, A + 1 + d_model) block written by pack_response."""
    width = action_size + 1 + d_model
    block = (
        np.frombuffer(buf, dtype=wire.np_dtype, count=n * width)
        .reshape(n, width)
        .astype(np.float32)
    )
    probs = block[:, :action_size]
    values = block[:, action_size]
    memory_out = block[:, action_size + 1 :]
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
