"""player.last_ip tracks the raw address a player was last seen from.

Opt-in like test_api_flow: set TEST_DATABASE_URL to a migrated database.
"""

from __future__ import annotations

import os
import uuid

import pytest

from server.api.ratelimit import remote_address
from server.services.persistence import players as players_db
from server.services.persistence.pool import open_pool

TEST_DB = os.environ.get("TEST_DATABASE_URL", "")

needs_db = pytest.mark.skipif(
    not TEST_DB, reason="TEST_DATABASE_URL not set (needs a migrated Postgres)"
)


def test_remote_address_is_exact_and_rejects_non_ips():
    # No /64 folding (that is client_key's job), but IPv4-mapped IPv6 is
    # reported as the IPv4 address it carries.
    assert remote_address("2001:db8::1234") == "2001:db8::1234"
    assert remote_address("::ffff:203.0.113.9") == "203.0.113.9"
    assert remote_address("203.0.113.9") == "203.0.113.9"
    assert remote_address("testclient") is None
    assert remote_address(None) is None


async def _last(pool, pid):
    row = await pool.fetchrow(
        "SELECT host(last_ip) AS ip, last_updated FROM player WHERE player_id = $1",
        pid,
    )
    return row["ip"], row["last_updated"]


@needs_db
async def test_ensure_player_records_and_updates_last_ip():
    pool = await open_pool(TEST_DB)
    try:
        pid = uuid.uuid4()
        await players_db.ensure_player(pool, pid, "203.0.113.9")
        ip, updated = await _last(pool, pid)
        assert ip == "203.0.113.9"

        # No address (e.g. a test client) never erases a recorded one.
        await players_db.ensure_player(pool, pid, None)
        assert (await _last(pool, pid))[0] == "203.0.113.9"

        await players_db.ensure_player(pool, pid, "2001:db8::1")
        ip, updated_after = await _last(pool, pid)
        assert ip == "2001:db8::1"
        # last_updated tracks the profile (name), not presence.
        assert updated_after == updated

        await players_db.touch_player_ip(pool, pid, "198.51.100.7")
        assert (await _last(pool, pid))[0] == "198.51.100.7"
    finally:
        await pool.close()
