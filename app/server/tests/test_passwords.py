"""argon2id password hashing and the unknown-account timing guard."""

from __future__ import annotations

from server.services import passwords


async def test_hash_round_trips_and_rejects_wrong_password():
    stored = await passwords.hash_password("correct horse")
    assert stored.startswith("$argon2id$")
    assert await passwords.verify_password(stored, "correct horse")
    assert not await passwords.verify_password(stored, "wrong horse")
    assert not passwords.needs_rehash(stored)


async def test_missing_or_corrupt_hash_never_verifies():
    assert not await passwords.verify_password(None, "anything")
    assert not await passwords.verify_password("not-a-hash", "anything")
