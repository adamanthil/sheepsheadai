#!/usr/bin/env python3
"""LAN round-trip probe for the remote-inference design.

Measures the one term the throughput model still derives rather than measures:
``t_wire``, the per-state cost of shipping a search round to a remote
accelerator and getting the answer back. Addendum 2 of
notebooks/Throughput_Profiling_Notes.md puts it at 2-22 us/state depending on
encoding and link, against ~7 us/state of actual GPU work -- so it is a
top-two term and a derived estimate is not good enough.

This deliberately does NOT import torch or sheepshead: it is a pure socket
probe, so it starts instantly and cannot be confounded by framework overhead.
It models the real payload shape rather than a symmetric ping, because the
workload is asymmetric -- a round ships observations plus recurrent memory up,
and gets memory, action probabilities and values back.

Per-state wire sizes (see addendum 2, B3):

  packed            up 39 B observation (uint8 ids)  + 512 B memory (fp16 x256)
                    down 512 B memory_out + 220 B probs (fp16 x110) + 2 B value
  memory-resident   up 39 B, down 222 B
                    (server holds the GRU memory; client sends permutations)
  naive             up ~280 B int64-stacked ids + 1024 B fp32 memory
                    down 1024 B memory + 440 B fp32 probs

Usage -- on the PC (the accelerator side):

    uv run python -m sheepshead.analysis.bench_lan_roundtrip --serve

Then on the Mac (the orchestrator side):

    uv run python -m sheepshead.analysis.bench_lan_roundtrip --connect 192.168.1.50

Add --states 126 1008 2016 4032 to sweep round sizes; 1008 is 8 workers'
worth of a typical 126-state round. Report the output back into addendum 2.
"""

import argparse
import socket
import statistics
import struct
import sys
import time

# (name, bytes up per state, bytes down per state)
ENCODINGS = (
    ("packed", 551, 734),
    ("memory-resident", 39, 222),
    ("naive", 1304, 1464),
)

DEFAULT_STATES = (126, 1008, 2016, 4032)
#: This probe and the inference server are different services on different
#: ports. Naming both here so the connect diagnostics can point at the mix-up,
#: which is the likeliest reason a connection to one of them fails.
DEFAULT_PORT = 53017
INFERENCE_PORT = 53018
_HEADER = struct.Struct("!Q")  # frame length prefix


#: Blocking sockets swallow Ctrl-C on Windows: a signal handler only runs
#: between bytecode instructions, and WSAAccept/WSARecv do not return early the
#: way an EINTR-interrupted POSIX syscall does. The server polls with a short
#: timeout so the interpreter gets a chance to raise KeyboardInterrupt. The
#: client does NOT retry -- there a timeout means the peer is gone and should
#: surface as an error.
_POLL_SECONDS = 0.5


def _recv_exact(sock: socket.socket, n: int, retry_on_timeout: bool = False) -> bytes:
    chunks = []
    remaining = n
    while remaining:
        try:
            chunk = sock.recv(min(remaining, 1 << 20))
        except TimeoutError:
            if retry_on_timeout:
                # Keep waiting for the rest of the frame. Retrying here rather
                # than at the frame level is what stops a timeout from
                # desynchronising a partially-read message.
                continue
            raise
        if not chunk:
            raise ConnectionError(f"peer closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_frame(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(_HEADER.pack(len(payload)) + payload)


def _recv_frame(sock: socket.socket, retry_on_timeout: bool = False) -> bytes:
    (length,) = _HEADER.unpack(
        _recv_exact(sock, _HEADER.size, retry_on_timeout=retry_on_timeout)
    )
    return _recv_exact(sock, length, retry_on_timeout=retry_on_timeout)


def local_addresses() -> list:
    """Best-effort IPv4 addresses of this host, printed so the operator can see
    which adapter to point the client at when several are up."""
    found = set()
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            found.add(info[4][0])
    except socket.gaierror:
        pass
    return sorted(found)


def _tune(sock: socket.socket) -> None:
    # Without TCP_NODELAY, Nagle batches the small request frame and adds tens
    # of milliseconds -- which would swamp the signal entirely.
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


def serve(port: int, bind: str = "0.0.0.0") -> int:
    """Echo server: read a frame, reply with a frame of the requested size.

    The reply size is carried in the first 8 bytes of the request, so the client
    controls the asymmetry and the server needs no configuration.
    """
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((bind, port))
    listener.listen(4)
    listener.settimeout(_POLL_SECONDS)
    print(f"listening on {bind}:{port}")
    if bind in ("0.0.0.0", ""):
        for address in local_addresses():
            print(f"  reachable at {address}:{port}")
        print("  (bound to every adapter; --bind <ip> to restrict)")
    print("  ctrl-c to stop")
    while True:
        try:
            conn, addr = listener.accept()
        except TimeoutError:
            continue  # see _POLL_SECONDS: keeps Ctrl-C responsive on Windows
        _tune(conn)
        conn.settimeout(_POLL_SECONDS)
        print(f"connection from {addr[0]}:{addr[1]}")
        blob = b"\0" * (1 << 22)
        try:
            while True:
                payload = _recv_frame(conn, retry_on_timeout=True)
                (reply_len,) = _HEADER.unpack(payload[: _HEADER.size])
                while len(blob) < reply_len:
                    blob += blob
                _send_frame(conn, blob[:reply_len])
        except (ConnectionError, OSError) as exc:
            print(f"  connection closed ({type(exc).__name__})")
        finally:
            conn.close()


def round_trip(sock: socket.socket, up: int, down: int, payload: bytes) -> float:
    request = _HEADER.pack(down) + payload[: max(0, up - _HEADER.size)]
    start = time.perf_counter()
    _send_frame(sock, request)
    reply = _recv_frame(sock)
    elapsed = time.perf_counter() - start
    if len(reply) != down:
        raise RuntimeError(f"expected {down} bytes back, got {len(reply)}")
    return elapsed


def measure(sock, up: int, down: int, iters: int, warmup: int = 5) -> dict:
    payload = b"\0" * max(up, 1)
    for _ in range(warmup):
        round_trip(sock, up, down, payload)
    samples = [round_trip(sock, up, down, payload) for _ in range(iters)]
    samples.sort()

    def pct(p: float) -> float:
        return samples[min(len(samples) - 1, int(len(samples) * p))]

    median = statistics.median(samples)
    return {
        "p50": median,
        "p90": pct(0.90),
        "p99": pct(0.99),
        "min": samples[0],
        "mbps": (up + down) / median / 1e6,
        "gbit": (up + down) * 8 / median / 1e9,
    }


def dial(host: str, port: int, connect_timeout: float = 8.0) -> socket.socket:
    """Connect, turning the two failure modes into the diagnosis they imply.

    The distinction is the whole diagnostic and is easy to miss: a refusal means
    the packet reached the host and nothing was listening, whereas a timeout
    means it was silently dropped -- so the two point at completely different
    causes, and a bare "timed out" sends people hunting for a routing problem
    that is not there.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(connect_timeout)
    try:
        sock.connect((host, port))
    except ConnectionRefusedError:
        raise SystemExit(
            f"connection refused at {host}:{port}\n"
            f"  The host is reachable and nothing is dropping packets -- there is\n"
            f"  simply no listener on that port. Start the probe server there:\n"
            f"    python -m sheepshead.analysis.bench_lan_roundtrip --serve "
            f"--port {port}\n"
            f"  Note this probe defaults to {DEFAULT_PORT}, which is NOT the\n"
            f"  inference server's port ({INFERENCE_PORT}) -- they are different\n"
            f"  services and both ends must agree."
        ) from None
    except TimeoutError:
        raise SystemExit(
            f"timed out connecting to {host}:{port}\n"
            f"  A timeout rather than 'connection refused' means the SYN was\n"
            f"  silently dropped, which is a firewall, not a missing route. If\n"
            f"  another port on this host answers, routing is fine. On Windows:\n"
            f"    New-NetFirewallRule -DisplayName sheepshead -Direction Inbound "
            f"-Protocol TCP -LocalPort {port} -Action Allow\n"
            f"    Set-NetConnectionProfile -InterfaceAlias <adapter> "
            f"-NetworkCategory Private\n"
            f"  Also confirm the port: this probe defaults to {DEFAULT_PORT}, the\n"
            f"  inference server to {INFERENCE_PORT}."
        ) from None
    sock.settimeout(None)
    return sock


def connect(host: str, port: int, states: tuple, iters: int) -> int:
    sock = dial(host, port)
    _tune(sock)
    print(f"connected to {host}:{port}\n")

    base = measure(sock, 64, 64, iters)
    print(
        f"baseline RTT (64 B each way): p50 {base['p50'] * 1e3:.3f} ms   "
        f"min {base['min'] * 1e3:.3f} ms   p99 {base['p99'] * 1e3:.3f} ms"
    )
    print(
        "  this is the fixed per-round cost; the model assumes ~0.5 ms and\n"
        "  charges it once per search round (~897 rounds/episode).\n"
    )

    results = {}
    for name, up_per, down_per in ENCODINGS:
        print(f"[{name}]  {up_per} B up + {down_per} B down per state")
        print(
            f"{'states':>8} | {'up KB':>8} {'down KB':>9} | {'p50 ms':>8} {'p90 ms':>8}"
            f" {'p99 ms':>8} | {'Gbit/s':>7} | {'us/state':>9}"
        )
        for n in states:
            up, down = up_per * n, down_per * n
            stat = measure(sock, up, down, max(5, iters // max(1, n // 126)))
            results[(name, n)] = stat
            print(
                f"{n:>8} | {up / 1e3:8.1f} {down / 1e3:9.1f} |"
                f" {stat['p50'] * 1e3:8.2f} {stat['p90'] * 1e3:8.2f}"
                f" {stat['p99'] * 1e3:8.2f} |"
                f" {stat['gbit']:7.2f} | {stat['p50'] / n * 1e6:9.1f}"
            )
        print()

    print("Model plug-in (addendum 2, B3):")
    print("  t_eff = t_wire + t_device, with t_device ~= 7 us/state on the 5060.")
    print(
        f"  {'encoding':>16} | {'t_wire':>9} | {'t_eff':>8} | {'capacity':>9} | vs 0.213"
    )
    for name, _, _ in ENCODINGS:
        best = min(
            (
                results[(name, n)]["p50"] / n * 1e6
                for n in states
                if (name, n) in results
            ),
            default=float("nan"),
        )
        t_eff = best + 7.0
        capacity = 1.0 / (113_000 * t_eff * 1e-6)
        capped = min(capacity, 0.69)  # M1 Max side ceiling at 8 workers
        print(
            f"  {name:>16} | {best:6.1f} us | {t_eff:5.1f} us |"
            f" {capped:6.2f} e/s | {capped / 0.213:5.2f}x"
            + ("  (Mac-capped)" if capacity > 0.69 else "")
        )
    sock.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--serve", action="store_true", help="run on the GPU box")
    mode.add_argument("--connect", metavar="HOST", help="run on the orchestrator")
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"probe port (default {DEFAULT_PORT}); the inference server is a "
        f"different service on {INFERENCE_PORT}",
    )
    parser.add_argument(
        "--bind",
        default="0.0.0.0",
        help="server: interface to listen on; default every adapter",
    )
    parser.add_argument(
        "--states",
        type=int,
        nargs="+",
        default=list(DEFAULT_STATES),
        help="states per round to sweep (1008 = 8 workers x a 126-state round)",
    )
    parser.add_argument("--iters", type=int, default=200)
    args = parser.parse_args()

    if args.serve:
        try:
            return serve(args.port, args.bind)
        except KeyboardInterrupt:
            print("\nstopped")
            return 0
    return connect(args.connect, args.port, tuple(args.states), args.iters)


if __name__ == "__main__":
    sys.exit(main())
