"""Background jobs ride out expected failures and report the rest."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import cast

import asyncpg
import pytest

from server.runtime.tasks import spawn
from server.services.persistence import sessions


async def test_a_failing_task_is_logged(caplog):
    async def broken() -> None:
        raise ValueError("bug")

    with caplog.at_level(logging.ERROR):
        task = spawn(broken(), "broken-job")
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)  # done callbacks run on the next loop tick

    (record,) = [r for r in caplog.records if "broken-job" in r.getMessage()]
    assert record.exc_info is not None and record.exc_info[0] is ValueError


async def test_a_cancelled_task_is_not_reported(caplog):
    task = spawn(asyncio.sleep(60), "sleepy-job")
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0)
    assert not [r for r in caplog.records if "sleepy-job" in r.getMessage()]


class _Pool:
    def __init__(self, error: BaseException) -> None:
        self.error = error
        self.attempts = 0

    @asynccontextmanager
    async def acquire(self):
        self.attempts += 1
        raise self.error
        yield  # pragma: no cover


async def _run_purge_until_it_sleeps_twice(pool, monkeypatch) -> None:
    sleeps = 0

    async def fake_sleep(seconds: float) -> None:
        nonlocal sleeps
        sleeps += 1
        if sleeps == 2:
            raise asyncio.CancelledError

    monkeypatch.setattr(sessions.asyncio, "sleep", fake_sleep)
    with pytest.raises(asyncio.CancelledError):
        await sessions.run_identity_purge(cast(asyncpg.Pool, pool))


async def test_purge_rides_out_a_database_outage(monkeypatch):
    pool = _Pool(ConnectionRefusedError())
    await _run_purge_until_it_sleeps_twice(pool, monkeypatch)
    assert pool.attempts == 2  # kept going after the first failure


async def test_purge_stops_on_an_unexpected_error(monkeypatch):
    pool = _Pool(ValueError("bug"))
    with pytest.raises(ValueError):
        await sessions.run_identity_purge(cast(asyncpg.Pool, pool))
