"""Analysis steps in a worker thread share the live tables' inference slots."""

from __future__ import annotations

import asyncio

import anyio.to_thread

from server.services import ai_loader
from server.services.ai_loader import inference_turn_in_thread


async def test_worker_thread_step_waits_for_live_inference(monkeypatch):
    slots = asyncio.Semaphore(1)
    monkeypatch.setattr(ai_loader, "inference_limit", slots)
    await slots.acquire()  # a live table's AI is mid-move
    ran: list[bool] = []

    def analysis_step() -> None:
        with inference_turn_in_thread():
            ran.append(True)

    step = asyncio.create_task(anyio.to_thread.run_sync(analysis_step))
    await asyncio.sleep(0.05)
    assert ran == []  # queued behind the table

    slots.release()
    await step
    assert ran == [True]
    assert not slots.locked()  # and gave the slot back


def test_off_the_event_loop_it_does_nothing():
    with inference_turn_in_thread():
        pass
