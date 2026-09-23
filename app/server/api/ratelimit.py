"""Shared per-IP rate limiter.

slowapi's in-memory storage is the right fit here: the server is pinned to a
single process (all game state is in-memory), so there is no cross-worker
state to synchronise. If the app ever runs multi-instance, swap the storage
for Redis via ``storage_uri``.

Behind the reverse proxy the client IP comes from X-Forwarded-For, which
uvicorn folds into ``request.client`` when run with ``--proxy-headers``.
"""

from __future__ import annotations

import ipaddress

from fastapi import Request
from slowapi import Limiter
from slowapi.util import get_remote_address


def client_key(host: str) -> str:
    """The per-client identity for a remote address. IPv6 is counted per
    /64: one subscriber is routinely handed a whole /64, so a per-address
    key would let a single host rotate through effectively unlimited keys.
    IPv4-mapped IPv6 counts as its IPv4 address."""
    try:
        addr = ipaddress.ip_address(host)
    except ValueError:
        return host  # not an IP (e.g. a test client's placeholder)
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped is not None:
            return str(addr.ipv4_mapped)
        return str(ipaddress.IPv6Network((addr, 64), strict=False))
    return str(addr)


def client_ip(request: Request) -> str:
    """The key every per-client limit is counted under (rate limits here,
    open-table caps in server.runtime.manager)."""
    return client_key(get_remote_address(request))


limiter = Limiter(key_func=client_ip)

# Budgets, per client IP. Mutating endpoints only; reads stay unlimited.
CREATE_JOIN = "10/minute"  # table creation / join: cheap to abuse, rare in play
HOST_ACTIONS = "30/minute"  # start, rules, seat, fill_ai, redeal, close, rename
GAME_ACTIONS = "120/minute"  # in-game moves; several humans can share a NAT
ANALYZE = "5/minute"  # each call runs a full torch simulation
ANALYZE_PICK = "20/minute"  # pre-play only (~10 inference steps); built for
# iterative lock-and-reroll use, so it gets a looser, separate budget
