"""Password hashing for accounts (argon2id).

Hashing is deliberately slow (tens of milliseconds of CPU), so every call
runs in a worker thread: on the event loop it would stall every table's
game and AI loop for the duration.
"""

from __future__ import annotations

import asyncio

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError

_hasher = PasswordHasher()

# Verified against when a login names no account, so an unknown login costs
# the same as a wrong password and response time doesn't reveal which.
_DUMMY_HASH = _hasher.hash("not-a-real-password")


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_hasher.hash, password)


def _verify(password_hash: str, password: str) -> bool:
    try:
        return _hasher.verify(password_hash, password)
    except VerificationError, InvalidHashError:
        return False


async def verify_password(password_hash: str | None, password: str) -> bool:
    """Check ``password``; a None hash (no such account) burns the same
    time as a real check and returns False."""
    if password_hash is None:
        await asyncio.to_thread(_verify, _DUMMY_HASH, password)
        return False
    return await asyncio.to_thread(_verify, password_hash, password)


def needs_rehash(password_hash: str) -> bool:
    """Whether a stored hash predates the current cost parameters."""
    return _hasher.check_needs_rehash(password_hash)
