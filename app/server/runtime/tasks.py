"""Background tasks whose failures are reported, not lost.

An exception escaping an asyncio task doesn't crash the process: the task
just ends, and the error surfaces (if ever) as "Task exception was never
retrieved" when the task is garbage-collected. spawn() logs any such
failure with its traceback as soon as the task ends, so an unexpected error
stops that one job loudly instead of silently.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Coroutine


def _report_failure(task: asyncio.Task) -> None:
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logging.error(
            "background task %s failed and stopped",
            task.get_name(),
            exc_info=exc,
        )


def spawn(coro: Coroutine[Any, Any, Any], name: str) -> asyncio.Task:
    """Start ``coro`` as a background task whose failure gets logged."""
    task = asyncio.create_task(coro, name=name)
    task.add_done_callback(_report_failure)
    return task
