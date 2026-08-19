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


class PeerClosed(ConnectionError):
    """A client hung up cleanly, between frames.

    At the socket layer a worker that finished its work looks exactly like one
    that died mid-transfer: both are a zero-byte read. The only thing telling
    them apart is whether any of the current frame had already arrived. Worth
    distinguishing, because the clean case is *every* worker at the end of
    *every* generation, and reporting that as an error is how an operator learns
    to stop reading the log.
    """


def _recv_exact(
    sock: socket.socket,
    n: int,
    stop: threading.Event = None,
    boundary: bool = False,
) -> bytes:
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
            if boundary and remaining == n:
                raise PeerClosed("peer closed between frames")
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
    # Only the length prefix sits on a frame boundary; an EOF anywhere after it
    # is a truncated message, not a clean goodbye.
    header = _recv_exact(sock, _FRAME.size, stop, boundary=True)
    return _recv_exact(sock, _FRAME.unpack(header)[0], stop)


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


class ServedModel:
    """The forward this server runs, optionally compiled.

    Compilation targets the cost that actually dominates here: ~17 ms of
    per-batch CUDA dispatch on a 2015 Skylake against a ~0.65 ms forward
    (notebooks/Distributed_Inference_202608.md §5.3). Hundreds of small aten
    ops driven through Python is exactly what inductor collapses, and unlike
    the merge factor it is not capped by queueing arithmetic.

    ``granularity`` rounds the batch up so a handful of compiled shapes serve
    every round size. Without it, ``dynamic=False`` recompiles per distinct
    merged batch -- and merged sizes are far more varied than a single client's,
    since they are sums over however many clients happened to arrive together.

    ``backend`` picks how. The default (inductor) needs Triton, which upstream
    PyTorch does not ship for Windows. ``cudagraphs`` needs no code generation
    at all -- it captures the eager kernel sequence and replays it as one graph
    -- which happens to be aimed squarely at this server's problem, since the
    cost here is launch count rather than kernel quality.
    """

    def __init__(
        self,
        agent,
        device,
        compile_mode=None,
        granularity: int = 1,
        backend: str | None = None,
    ):
        self.agent = agent
        self.device = device
        # Independent of compile_mode so the padding path is testable without
        # paying for a compile.
        self.granularity = max(1, int(granularity))
        self._forward_eager = self._forward
        self.forward = self._forward
        if compile_mode:
            # Merged batches take more distinct shapes than a single client's
            # do, and dynamo's default budget of 8 is silently exceeded.
            from sheepshead.inference.compiled import allow_shape_specialisation

            allow_shape_specialisation()
            if backend:
                kwargs = {"backend": backend}  # mode is an inductor-only knob
            else:
                kwargs = {} if compile_mode == "default" else {"mode": compile_mode}
            self.forward = torch.compile(self._forward, dynamic=False, **kwargs)

    def _fall_back(self, exc: Exception) -> None:
        """Degrade to eager rather than take the run down with us.

        Compilation fails for environmental reasons -- a missing Triton install,
        a graph CUDA cannot capture -- and it fails on the *first batch*, which
        is to say on eight workers at once, mid-generation. That must cost
        throughput, not the generation. Loud, once, then eager for good.
        """
        self.forward = self._forward_eager
        self.granularity = 1  # padding existed only to bound the shape count
        print(
            f"\n  COMPILE FAILED -- serving eager for the rest of this run.\n"
            f"  {type(exc).__name__}: {str(exc).splitlines()[0]}\n"
            f"  On Windows the inductor backend needs Triton, which upstream "
            f"PyTorch does not ship; try --compile-backend cudagraphs.\n",
            flush=True,
        )

    def _forward(self, marshalled, memory_in, masks, wants_critic: bool):
        encoded = self.agent.encoder.encode_tensors(
            marshalled, memory_in=memory_in, device=self.device
        )
        probs, _ = self.agent.actor.forward_with_logits(
            encoded, masks, marshalled["hand_ids"], self.agent.encoder.card
        )
        if wants_critic:
            values = self.agent.critic(encoded).view(-1)
        else:
            values = torch.zeros(probs.shape[0], device=probs.device)
        return probs, values, encoded["memory_out"]

    def __call__(self, merged: dict) -> tuple:
        exact = (merged["marshalled"], merged["memory_in"], merged["masks"])
        wants_critic = merged["wants_critic"]
        n = int(merged["memory_in"].shape[0])
        pad = -(-n // self.granularity) * self.granularity - n
        if pad:
            marshalled, memory_in, masks = exact
            padded = (
                {
                    key: torch.cat([v, v[:1].repeat_interleave(pad, 0)], 0)
                    for key, v in marshalled.items()
                },
                torch.cat([memory_in, memory_in[:1].repeat_interleave(pad, 0)], 0),
                # Pad rows get an all-legal mask. An all-False row softmaxes to
                # NaN, and one NaN row is enough to poison a client's reply.
                torch.cat([masks, torch.ones_like(masks[:1].expand(pad, -1))], 0),
            )
        else:
            padded = exact

        with torch.no_grad():
            try:
                probs, values, memory_out = self.forward(*padded, wants_critic)
            except Exception as exc:
                if self.forward is self._forward_eager:
                    raise  # eager failed too: a real bug, not a compile problem
                self._fall_back(exc)
                # Retry on the *unpadded* inputs, so this reply is bit-identical
                # to what a plain eager server would have sent. Padding existed
                # only to bound the compiled shape count.
                probs, values, memory_out = self._forward_eager(*exact, wants_critic)
        return probs[:n], values[:n], memory_out[:n]


def run_batch(model: ServedModel, raws: list, device: torch.device) -> list:
    """Merge, forward, split. Returns one reply frame per request, in order.

    A batch of one takes exactly this path too -- there is no separate unbatched
    branch that could drift from the batched one. The transfer count is fixed at
    four in and one out no matter how many clients are merged, which is most of
    why merging pays: see ``merge_requests`` and ``response_block``.
    """
    merged = merge_requests(raws, device=device)
    action_size = merged["action_size"]
    probs, values, memory_out = model(merged)

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
    return run_batch(ServedModel(agent, device), [raw], device)[0]


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
        self.started = None

    def record(
        self, requests: int, states: int, start: float, compute: float, waited: float
    ) -> None:
        with self._lock:
            if self.started is None:
                self.started = start
            self.batches += 1
            self.requests += requests
            self.states += states
            self.compute += compute
            self.waited += waited

    def line(self) -> str:
        with self._lock:
            if not self.batches:
                return "no batches served"
            elapsed = max(time.perf_counter() - self.started, 1e-9)
            return (
                f"{self.batches} batches  "
                f"merge {self.requests / self.batches:.2f} req/batch  "
                f"{self.states / self.batches:.0f} states/batch  "
                f"{self.compute / self.batches * 1e3:.1f} ms/batch  "
                f"{self.compute / max(self.states, 1) * 1e6:.1f} us/state  "
                f"queue wait {self.waited / max(self.requests, 1) * 1e3:.1f} ms  "
                # Decisive when the merge factor disappoints. Busy near 100% with
                # a long queue wait means the batcher itself is the wall, so the
                # fix is to make a batch cheaper. Busy well under 100% with a
                # short queue wait means requests are not reaching the queue --
                # the connection threads are behind, and merging cannot help
                # because the cost is upstream of the merge point.
                f"busy {100 * self.compute / elapsed:.0f}%"
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


def _batcher(model, device, work, stop, stats, policy, report_every) -> None:
    while not stop.is_set():
        try:
            first = work.get(timeout=_POLL_SECONDS)
        except queue.Empty:
            continue
        batch = _collect(work, first, policy)

        start = time.perf_counter()
        try:
            replies = run_batch(model, [p.raw for p in batch], device)
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
        stats.record(len(batch), states, start, compute, waited)
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
    except PeerClosed:
        print(f"  {who} done, {rounds} rounds", flush=True)
    except (ConnectionError, OSError) as exc:
        print(f"  {who} LOST after {rounds} rounds ({type(exc).__name__}: {exc})")
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
    model: "ServedModel" = None,
) -> int:
    policy = policy or BatchPolicy()
    model = model or ServedModel(agent, device)
    expected = "|".join(
        fingerprint_weights(net) for net in (agent.encoder, agent.actor, agent.critic)
    )
    stop = threading.Event()
    work: queue.Queue = queue.Queue()
    stats = _Stats()
    threading.Thread(
        target=_batcher,
        args=(model, device, work, stop, stats, policy, report_every),
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
        "--compile",
        nargs="?",
        const="reduce-overhead",
        default=None,
        metavar="MODE",
        help="compile the served forward (default mode reduce-overhead, which "
        "on CUDA means CUDA graphs; it is a no-op on MPS, where Metal has no "
        "equivalent). Pass 'default' for plain inductor. Off unless given: the "
        "first batch of each shape pays compilation",
    )
    parser.add_argument(
        "--compile-backend",
        default=None,
        metavar="BACKEND",
        help="torch.compile backend. Unset means inductor, which needs Triton "
        "-- upstream PyTorch does not ship it for Windows. 'cudagraphs' needs "
        "no code generation and targets exactly what dominates this server: it "
        "replays hundreds of kernel launches as a single graph. --compile MODE "
        "is inductor-specific and is ignored when this is set",
    )
    parser.add_argument(
        "--tf32",
        action="store_true",
        help="allow TF32 for float32 matmuls on Ampere and later. Real speedup, "
        "real precision loss (~10 mantissa bits). UNVALIDATED: fp16 on the wire "
        "flipped a search label, so check bench_remote_search's divergence "
        "output before trusting this",
    )
    parser.add_argument(
        "--pad-granularity",
        type=int,
        default=32,
        help="with --compile, round merged batches up to a multiple of this so "
        "a few compiled shapes cover every round size",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=200,
        help="print merge/throughput stats every N batches; 0 to stay quiet",
    )
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 enabled -- verify search divergence before trusting results")
    device = torch.device(args.device)
    agent = build_agent(args.checkpoint, args.arch, device, args.seed)
    model = ServedModel(
        agent,
        device,
        args.compile,
        args.pad_granularity if args.compile else 1,
        args.compile_backend,
    )
    if args.compile:
        how = args.compile_backend or f"inductor/{args.compile}"
        print(
            f"compiling the forward ({how}, pad to multiples of "
            f"{args.pad_granularity}); the first batch of each shape is slow, "
            f"and a failure degrades to eager rather than dropping clients"
        )
    policy = BatchPolicy(
        window=args.batch_window_ms / 1e3,
        max_requests=args.max_batch_requests,
        max_states=args.max_batch_states,
    )
    try:
        return serve(
            agent, args.port, device, args.bind, policy, args.report_every, model
        )
    except KeyboardInterrupt:
        print("\nstopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
