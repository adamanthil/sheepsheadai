#!/usr/bin/env python3
"""Accelerator-side inference server for the ISMCTS search teacher (prototype).

Holds one agent's weights on a GPU and answers packed search rounds. It never
sees an observation dict: the orchestrator marshals and ships tensors, because
on the measured setup this host packs observations at ~40 us/state against
~7 us/state of GPU work (addendum 2, B1).

Single-connection, single-threaded by design for the prototype. Batching across
several orchestrator workers is the eventual point -- one merged round of
8 x 126 states instead of eight separate ones -- but that needs a request queue
with a batching window, and adding it before per-round latency is measured
would be guessing at the parameter that matters.

Usage, on the GPU box:

    uv run python -m sheepshead.inference.server \
        --checkpoint runs/league_ce_teacher11/_league_worker_weights_v24.pt \
        --arch perceiver-shared-v2 --device cuda
"""

import argparse
import socket
import struct
import sys
import time

import torch

from sheepshead import ACTION_IDS
from sheepshead.inference.protocol import (
    fingerprint_weights,
    pack_response,
    unpack_request,
)

_FRAME = struct.Struct("!Q")


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


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_FRAME.pack(len(payload)) + payload)


def _recv(sock: socket.socket) -> bytes:
    (length,) = _FRAME.unpack(_recv_exact(sock, _FRAME.size))
    return _recv_exact(sock, length)


def build_agent(checkpoint: str | None, arch: str, device: torch.device, seed: int):
    """Load the served weights, or build a seeded fresh agent when no checkpoint
    is given.

    The fresh path exists for smoke tests: both ends construct the same
    architecture under the same seed, so the fingerprints match without either
    side reading a file. Real use should always pass --checkpoint, and note that
    a league run's ``_league_worker_weights_v*.pt`` files rotate as it
    progresses -- serving one while the client loads a newer one is exactly what
    the handshake exists to catch.
    """
    from sheepshead.agent import ppo as ppo_module

    ppo_module.device = device
    from sheepshead.agent.ppo import PPOAgent
    from sheepshead.training.training_utils import set_all_seeds

    set_all_seeds(seed)
    agent = PPOAgent(action_size=len(ACTION_IDS), arch=arch)
    if checkpoint:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        agent.load_network_states(payload, source=checkpoint)
    else:
        print(f"no --checkpoint: serving a fresh seed={seed} agent (smoke test only)")
    for net in (agent.encoder, agent.actor, agent.critic):
        net.to(device)
        net.eval()
    return agent


def serve_round(agent, request: bytes, device: torch.device) -> bytes:
    parsed = unpack_request(request, agent.encoder.d_model)
    marshalled = {k: v.to(device) for k, v in parsed["marshalled"].items()}
    memory_in = parsed["memory_in"].to(device)
    masks = parsed["masks"].to(device)

    with torch.no_grad():
        encoded = agent.encoder.encode_tensors(
            marshalled, memory_in=memory_in, device=device
        )
        probs, _ = agent.actor.forward_with_logits(
            encoded, masks, marshalled["hand_ids"], agent.encoder.card
        )
        memory_out = encoded["memory_out"]
        n = memory_in.shape[0]
        if parsed["wants_critic"]:
            values = agent.critic(encoded).view(-1)
        else:
            values = torch.zeros(n, device=device)

    return pack_response(
        probs.cpu().numpy(),
        values.cpu().numpy(),
        memory_out.cpu().numpy(),
        parsed["wire"],
    )


def serve(agent, port: int, device: torch.device) -> int:
    expected = "|".join(
        fingerprint_weights(net) for net in (agent.encoder, agent.actor, agent.critic)
    )
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", port))
    listener.listen(4)
    print(f"serving {device} on 0.0.0.0:{port}   weights {expected[:16]}...")
    while True:
        conn, addr = listener.accept()
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        print(f"connection from {addr[0]}:{addr[1]}")
        rounds = 0
        compute = 0.0
        try:
            offered = _recv(conn).decode()
            if offered != expected:
                _send(conn, b"weight fingerprint mismatch")
                print("  REJECTED: client weights differ from the served checkpoint")
                conn.close()
                continue
            _send(conn, b"ok")
            while True:
                request = _recv(conn)
                start = time.perf_counter()
                reply = serve_round(agent, request, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elif device.type == "mps":
                    torch.mps.synchronize()
                compute += time.perf_counter() - start
                _send(conn, reply)
                rounds += 1
        except (ConnectionError, OSError) as exc:
            print(f"  closed after {rounds} rounds ({type(exc).__name__})")
            if rounds:
                print(
                    f"  server-side compute total {compute:.2f}s over {rounds} rounds"
                )
        finally:
            conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--arch", default="perceiver-shared-v2")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=53018)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--seed", type=int, default=42, help="fresh-agent seed when no checkpoint"
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    agent = build_agent(args.checkpoint, args.arch, device, args.seed)
    try:
        return serve(agent, args.port, device)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
