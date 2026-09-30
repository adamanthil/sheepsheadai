"""A stand-in asyncpg pool that records every statement it is given."""

from __future__ import annotations

from contextlib import asynccontextmanager


class RecordingConn:
    def __init__(self, log: list) -> None:
        self.log = log

    async def execute(self, sql: str, *args) -> None:
        self.log.append((" ".join(sql.split()), args))

    async def executemany(self, sql: str, rows) -> None:
        self.log.append((" ".join(sql.split()), list(rows)))

    async def fetchval(self, sql: str, *args) -> int:
        self.log.append((" ".join(sql.split()), args))
        return 1

    @asynccontextmanager
    async def transaction(self):
        yield


class RecordingPool:
    def __init__(self) -> None:
        self.log: list = []

    @asynccontextmanager
    async def acquire(self):
        yield RecordingConn(self.log)
