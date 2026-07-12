"""Shared per-IP rate limiter.

slowapi's in-memory storage is the right fit here: the server is pinned to a
single process (all game state is in-memory), so there is no cross-worker
state to synchronise. If the app ever runs multi-instance, swap the storage
for Redis via ``storage_uri``.

Behind the reverse proxy the client IP comes from X-Forwarded-For, which
uvicorn folds into ``request.client`` when run with ``--proxy-headers``.
"""

from __future__ import annotations

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# Budgets, per client IP. Mutating endpoints only; reads stay unlimited.
CREATE_JOIN = "10/minute"  # table creation / join: cheap to abuse, rare in play
HOST_ACTIONS = "30/minute"  # start, rules, seat, fill_ai, redeal, close, rename
GAME_ACTIONS = "120/minute"  # in-game moves; several humans can share a NAT
ANALYZE = "5/minute"  # each call runs a full torch simulation
