"""Inference backends for a search round.

``LocalBackend`` is the default and runs the forward in-process, exactly as
``_run_network_round`` always did. ``RemoteBackend`` ships the round to an
accelerator host over TCP.

The local path must stay bit-exact: capture_search_goldens pins torch op order
and RNG draw order, and a server-backed run can never satisfy it (cross-worker
batch composition changes GEMM tiling). Remote mode is therefore an opt-in
throughput mode, never the default and never the thing goldens are captured on.
"""

import socket
import struct
import time
from typing import NamedTuple, Protocol

import numpy as np
import torch

from sheepshead.inference.protocol import (
    WireConfig,
    fingerprint_weights,
    pack_request,
    unpack_response,
)

_FRAME = struct.Struct("!Q")


class RoundResult(NamedTuple):
    """One evaluated round: ``probs`` (n, A), ``values`` (n,), and the advanced
    recurrent state ``memory_out`` (n, d_model) that the caller writes back into
    its per-sim state."""

    probs: np.ndarray
    values: np.ndarray
    memory_out: torch.Tensor


class InferenceBackend(Protocol):
    def evaluate(
        self,
        controller,
        states: list,
        memory_in: torch.Tensor,
        valid_lists: list,
        wants_critic: bool,
    ) -> RoundResult: ...


class LocalBackend:
    """In-process forward. Byte-for-byte the original ``_run_network_round``
    body, moved but not changed."""

    name = "local"

    def __init__(self, device: torch.device | None = None):
        self.device = device
        self.rounds = 0
        self.states = 0

    def evaluate(self, controller, states, memory_in, valid_lists, wants_critic):
        self.rounds += 1
        self.states += len(states)
        encoded = controller.encoder.encode_batch(
            states, memory_in=memory_in, device=self.device
        )
        memory_out = encoded["memory_out"].detach()
        probs = masked_actor_probs(
            controller,
            encoded,
            states,
            valid_lists,
            controller.action_size,
            self.device,
        )
        values = np.zeros(len(states), dtype=np.float32)
        if wants_critic:
            with torch.no_grad():
                values = controller.critic(encoded).detach().view(-1).cpu().numpy()
        return RoundResult(probs.detach().cpu().numpy(), values, memory_out)


def masked_actor_probs(controller, encoded, states, valid_lists, action_size, device):
    """Post-mixture action probabilities (n, A) under ``controller`` for
    already-encoded states -- the shared mask/hand_ids/actor plumbing.

    Lives here rather than on ISMCTSTeacher because both the search's replay
    path and every inference backend need it, and a second copy would be free
    to drift from the one the search goldens pin. ``encoded`` must come from
    ``controller``'s own encoder.
    """
    masks = torch.stack(
        [controller.get_action_mask(valid, action_size) for valid in valid_lists]
    ).to(device)
    hand_ids = torch.as_tensor(
        np.stack([state["hand_ids"] for state in states]),
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        probs, _ = controller.actor.forward_with_logits(
            encoded, masks, hand_ids, controller.encoder.card
        )
    return probs


class RemoteBackend:
    """Ships each round to an accelerator host.

    Marshalling happens here, on the orchestrator, and only packed arrays go
    over the wire -- the measured reason being that the accelerator host in this
    setup packs observations at ~40 us/state against ~7 us/state of GPU work, so
    marshalling remotely would cost far more than the forward it accelerates.

    Records per-round latency and payload so the prototype can report what the
    throughput model has to assume.
    """

    name = "remote"

    def __init__(
        self, host: str, port: int, controller, half: bool = True, timeout: float = 60.0
    ):
        self.wire = WireConfig(half=half)
        self.rounds = 0
        self.states = 0
        self.latencies: list[float] = []
        self.bytes_up = 0
        self.bytes_down = 0
        self._sock = socket.create_connection((host, port), timeout=timeout)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._handshake(controller)

    def _handshake(self, controller) -> None:
        """Refuse to proceed unless the server holds identical weights. A stale
        server would raise nothing and silently emit a plausible, wrong search
        target -- the one failure mode that must be impossible."""
        fingerprints = "|".join(
            fingerprint_weights(net)
            for net in (controller.encoder, controller.actor, controller.critic)
        )
        self._send(fingerprints.encode())
        reply = self._recv().decode()
        if reply != "ok":
            raise RuntimeError(f"server rejected handshake: {reply}")

    def _send(self, payload: bytes) -> None:
        self._sock.sendall(_FRAME.pack(len(payload)) + payload)

    def _recv(self) -> bytes:
        (length,) = _FRAME.unpack(_recv_exact(self._sock, _FRAME.size))
        return _recv_exact(self._sock, length)

    def evaluate(self, controller, states, memory_in, valid_lists, wants_critic):
        marshalled = controller.encoder.marshal_batch(states)
        action_size = controller.action_size
        d_model = controller.encoder.d_model
        request = pack_request(
            marshalled, memory_in, valid_lists, action_size, wants_critic, self.wire
        )
        start = time.perf_counter()
        self._send(request)
        reply = self._recv()
        self.latencies.append(time.perf_counter() - start)
        self.rounds += 1
        self.states += len(states)
        self.bytes_up += len(request)
        self.bytes_down += len(reply)

        probs, values, memory_out = unpack_response(
            reply, len(states), action_size, d_model, self.wire
        )
        return RoundResult(probs, values, torch.from_numpy(memory_out))

    def close(self) -> None:
        try:
            self._sock.close()
        except OSError:
            pass

    def report(self) -> dict:
        lat = sorted(self.latencies)
        if not lat:
            return {"rounds": 0}
        return {
            "rounds": self.rounds,
            "states": self.states,
            "mean_batch": self.states / self.rounds,
            "p50_ms": lat[len(lat) // 2] * 1e3,
            "p90_ms": lat[min(len(lat) - 1, int(len(lat) * 0.9))] * 1e3,
            "p99_ms": lat[min(len(lat) - 1, int(len(lat) * 0.99))] * 1e3,
            "total_s": sum(lat),
            "mb_up": self.bytes_up / 1e6,
            "mb_down": self.bytes_down / 1e6,
            "us_per_state": sum(lat) / self.states * 1e6,
        }


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        chunk = sock.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)
