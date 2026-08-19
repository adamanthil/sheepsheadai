#!/usr/bin/env python3
"""Accelerator-side inference server for the ISMCTS search teacher.

Holds one agent's weights on a GPU and answers packed search rounds. It never
sees an observation dict: the orchestrator marshals and ships tensors, because
on the measured setup this host packs observations at ~40 us/state against
~7 us/state of GPU work (addendum 2, B1).

**Rounds from several orchestrator workers are merged into one forward.** That
is the point of the server, not a refinement of it. Measured on the unbatched
prototype, ~94% of the per-round server cost was fixed overhead -- unpack, dtype
conversion, transfers, pack -- against a ~0.65 ms GPU forward, which works out
to ~125 us/state and would have run the trainer *slower* than the CPU path it
replaces. Merging eight workers' rounds amortizes that fixed cost eightfold and
is what makes the design viable at all
(notebooks/Distributed_Inference_202608.md §5, §6.1).

Threading: one thread per connection does the socket I/O and the (cheap) decode;
a single batcher thread owns the model and runs every forward. So connection
threads decode while the batcher computes, and no lock guards the model because
only one thread ever touches it.

Usage, on the GPU box:

    uv run python -m sheepshead.inference.server \
        --checkpoint runs/league_ce_teacher11/_league_worker_weights_v24.pt \
        --arch perceiver-shared-v2 --device cuda
"""

import argparse
import queue
import socket
import struct
import sys
import threading
import time

import torch

from sheepshead import ACTION_IDS
from sheepshead.inference.protocol import (
    decode_request,
    fingerprint_weights,
    merge_requests,
    response_block,
)

_FRAME = struct.Struct("!Q")


#: Blocking sockets swallow Ctrl-C on Windows: a signal handler only runs
#: between bytecode instructions, and WSAAccept/WSARecv do not return early the
#: way an EINTR-interrupted POSIX syscall does. Giving every blocking call a
#: short timeout hands control back to the interpreter often enough for the
#: KeyboardInterrupt to be raised.
_POLL_SECONDS = 0.5


def _recv_exact(sock: socket.socket, n: int, stop: threading.Event = None) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        if stop is not None and stop.is_set():
            raise ConnectionError("server shutting down")
        try:
            chunk = sock.recv(min(remaining, 1 << 20))
        except TimeoutError:
            # Not an error: keep waiting for the rest of the frame. Retrying
            # here rather than at the frame level is what keeps a timeout from
            # desynchronising a partially-read message.
            continue
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def local_addresses() -> list:
    """Best-effort IPv4 addresses of this host.

    Printed at startup because the common multi-adapter mistake is pointing the
    client at the wrong one -- a machine with WiFi for internet and a
    point-to-point link to the orchestrator has two, and only one of them is
    reachable from the other end.
    """
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except socket.gaierror:
        pass
    return sorted(found)


def _send(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_FRAME.pack(len(payload)) + payload)


def _recv(sock: socket.socket, stop: threading.Event = None) -> bytes:
    (length,) = _FRAME.unpack(_recv_exact(sock, _FRAME.size, stop))
    return _recv_exact(sock, length, stop)


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


def run_batch(agent, raws: list, device: torch.device) -> list:
    """Merge, forward, split. Returns one reply frame per request, in order.

    A batch of one takes exactly this path too -- there is no separate unbatched
    branch that could drift from the batched one. The transfer count is fixed at
    four in and one out no matter how many clients are merged, which is most of
    why merging pays: see ``merge_requests`` and ``response_block``.
    """
    merged = merge_requests(raws, device=device)
    marshalled = merged["marshalled"]
    action_size = merged["action_size"]

    with torch.no_grad():
        encoded = agent.encoder.encode_tensors(
            marshalled, memory_in=merged["memory_in"], device=device
        )
        probs, _ = agent.actor.forward_with_logits(
            encoded, merged["masks"], marshalled["hand_ids"], agent.encoder.card
        )
        memory_out = encoded["memory_out"]
        if merged["wants_critic"]:
            values = agent.critic(encoded).view(-1)
        else:
            values = torch.zeros(probs.shape[0], device=probs.device)

    # One device-to-host copy for the whole merged batch; the per-client split
    # below is row slicing on the host, which is free.
    block = response_block(probs, values, memory_out, merged["wire"])

    replies = []
    offset = 0
    for raw in raws:
        rows = block[offset : offset + raw.n]
        if merged["wants_critic"] and not raw.wants_critic:
            # Some other client in this batch asked for the critic. A client
            # that did not must still get what LocalBackend would have given it.
            rows[:, action_size] = 0
        replies.append(rows.tobytes())
        offset += raw.n
    return replies


def serve_round(agent, request: bytes, device: torch.device) -> bytes:
    """One request in, one reply out -- the batch-of-one case of ``run_batch``."""
    raw = decode_request(request, agent.encoder.d_model)
    return run_batch(agent, [raw], device)[0]


class _Pending:
    """One client's round, waiting on the batcher thread."""

    __slots__ = ("raw", "enqueued", "done", "reply", "error")

    def __init__(self, raw):
        self.raw = raw
        self.enqueued = time.perf_counter()
        self.done = threading.Event()
        self.reply = b""
        self.error = None


class _Stats:
    """Server-side accounting. ``requests / batches`` is the merge factor, which
    is the number this whole feature exists to raise."""

    def __init__(self):
        self._lock = threading.Lock()
        self.batches = 0
        self.requests = 0
        self.states = 0
        self.compute = 0.0
        self.waited = 0.0

    def record(self, requests: int, states: int, compute: float, waited: float) -> None:
        with self._lock:
            self.batches += 1
            self.requests += requests
            self.states += states
            self.compute += compute
            self.waited += waited

    def line(self) -> str:
        with self._lock:
            if not self.batches:
                return "no batches served"
            return (
                f"{self.batches} batches  "
                f"merge {self.requests / self.batches:.2f} req/batch  "
                f"{self.states / self.batches:.0f} states/batch  "
                f"{self.compute / self.batches * 1e3:.1f} ms/batch  "
                f"{self.compute / max(self.states, 1) * 1e6:.1f} us/state  "
                f"queue wait {self.waited / max(self.requests, 1) * 1e3:.1f} ms"
            )


class BatchPolicy:
    """When to stop collecting and run.

    Batching is **greedy by default** (``window == 0``): take everything already
    queued and go. With synchronous clients -- each worker blocks on its
    round-trip -- that is self-synchronizing, because while batch k computes all
    K workers queue their next round, so batch k+1 fills without anyone having
    waited. A nonzero window only helps if arrivals are adversarially staggered,
    and it costs its full value on *every* round when there is a single client,
    so it stays off unless asked for.
    """

    def __init__(
        self, window: float = 0.0, max_requests: int = 16, max_states: int = 8192
    ):
        self.window = float(window)
        self.max_requests = int(max_requests)
        self.max_states = int(max_states)


def _collect(work: queue.Queue, first: _Pending, policy: BatchPolicy) -> list:
    """Drain the queue into one batch, respecting the policy's caps."""
    batch = [first]
    states = first.raw.n
    key = first.raw.key
    deadline = time.perf_counter() + policy.window
    while len(batch) < policy.max_requests and states < policy.max_states:
        remaining = deadline - time.perf_counter()
        try:
            nxt = work.get(timeout=remaining) if remaining > 0 else work.get_nowait()
        except queue.Empty:
            break
        if nxt.raw.key != key:
            # Different action_size or wire dtype: it cannot share this forward.
            # Put it back rather than dropping it -- it heads the next batch.
            work.put(nxt)
            break
        batch.append(nxt)
        states += nxt.raw.n
    return batch


def _batcher(agent, device, work, stop, stats, policy, report_every) -> None:
    while not stop.is_set():
        try:
            first = work.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            continue
        batch = _collect(work, first, policy)

        start = time.perf_counter()
        try:
            replies = run_batch(agent, [p.raw for p in batch], device)
            for pending, reply in zip(batch, replies):
                pending.reply = reply
        except Exception as exc:
            # A failed forward must fail its clients, not strand them: every
            # waiter in this batch is blocked on an Event that only we set.
            for pending in batch:
                pending.error = exc
        compute = time.perf_counter() - start

        waited = sum(start - pending.enqueued for pending in batch)
        states = sum(pending.raw.n for pending in batch)
        for pending in batch:
            pending.done.set()
        stats.record(len(batch), states, compute, waited)
        if report_every and stats.batches % report_every == 0:
            print(f"  {stats.line()}", flush=True)


def _serve_connection(conn, addr, expected, work, d_model, stop) -> None:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    conn.settimeout(_POLL_SECONDS)
    who = f"{addr[0]}:{addr[1]}"
    rounds = 0
    try:
        offered = _recv(conn, stop).decode()
        if offered != expected:
            _send(conn, b"weight fingerprint mismatch")
            print(f"  REJECTED {who}: client weights differ from the served checkpoint")
            return
        _send(conn, b"ok")
        print(f"  {who} accepted", flush=True)
        while not stop.is_set():
            pending = _Pending(decode_request(_recv(conn, stop), d_model))
            work.put(pending)
            while not pending.done.wait(_POLL_SECONDS):
                if stop.is_set():
                    return
            if pending.error is not None:
                raise pending.error
            _send(conn, pending.reply)
            rounds += 1
    except (ConnectionError, OSError) as exc:
        print(f"  {who} closed after {rounds} rounds ({type(exc).__name__})")
    except Exception as exc:
        print(f"  {who} failed after {rounds} rounds: {exc!r}")
    finally:
        conn.close()


def serve(
    agent,
    port: int,
    device: torch.device,
    bind: str = "0.0.0.0",
    policy: BatchPolicy = None,
    report_every: int = 200,
) -> int:
    policy = policy or BatchPolicy()
    expected = "|".join(
        fingerprint_weights(net) for net in (agent.encoder, agent.actor, agent.critic)
    )
    stop = threading.Event()
    work: queue.Queue = queue.Queue()
    stats = _Stats()
    threading.Thread(
        target=_batcher,
        args=(agent, device, work, stop, stats, policy, report_every),
        name="batcher",
        daemon=True,
    ).start()

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind, port))
    listener.listen(32)
    listener.settimeout(_POLL_SECONDS)
    print(f"serving {device} on {bind}:{port}   weights {expected[:16]}...")
    if bind in ("0.0.0.0", ""):
        for address in local_addresses():
            print(f"  reachable at {address}:{port}")
        print("  (bound to every adapter; --bind <ip> to restrict)")
    print(
        f"  batching: up to {policy.max_requests} clients / {policy.max_states} "
        f"states per forward, window {policy.window * 1e3:.1f} ms"
    )
    print("  ctrl-c to stop")
    try:
        while True:
            try:
                conn, addr = listener.accept()
            except TimeoutError:
                continue  # see _POLL_SECONDS: keeps Ctrl-C responsive on Windows
            threading.Thread(
                target=_serve_connection,
                args=(conn, addr, expected, work, agent.encoder.d_model, stop),
                name=f"conn-{addr[1]}",
                daemon=True,
            ).start()
    finally:
        stop.set()
        listener.close()
        print(f"\n{stats.line()}", flush=True)
    return 0


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
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="interface to listen on; default every adapter. Set this to the "
        "point-to-point link's local IP to keep the server off a shared network",
    )
    parser.add_argument(
        "--batch-window-ms",
        type=float,
        default=0.0,
        help="how long to wait for more clients after the first request of a "
        "batch. 0 (default) is greedy: take whatever is queued and go, which "
        "with synchronous clients fills batches on its own. Nonzero adds this "
        "much latency to every round when only one client is connected",
    )
    parser.add_argument("--max-batch-requests", type=int, default=16)
    parser.add_argument("--max-batch-states", type=int, default=8192)
    parser.add_argument(
        "--report-every",
        type=int,
        default=200,
        help="print merge/throughput stats every N batches; 0 to stay quiet",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    agent = build_agent(args.checkpoint, args.arch, device, args.seed)
    policy = BatchPolicy(
        window=args.batch_window_ms / 1e3,
        max_requests=args.max_batch_requests,
        max_states=args.max_batch_states,
    )
    try:
        return serve(agent, args.port, device, args.bind, policy, args.report_every)
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
