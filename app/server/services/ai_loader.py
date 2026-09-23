from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Iterator

import anyio.from_thread

from sheepshead.agent import ppo
from sheepshead.agent.ppo import PPOAgent, device

# Global bound on concurrent torch inference across all tables so many
# simultaneous games can't oversubscribe the CPU (uvicorn runs 1 process).
inference_limit = asyncio.Semaphore(max(1, (os.cpu_count() or 4) // 2))


@contextmanager
def inference_turn_in_thread() -> Iterator[None]:
    """Hold an ``inference_limit`` slot for one model step taken in a worker
    thread the event loop started (a sync endpoint such as /analyze).

    Such steps then queue step by step alongside the live tables' AI, rather
    than a whole analysis competing with every table for the CPU at once.
    Off the event loop's worker threads (scripts, direct test calls) there
    is no live play to share with, and this does nothing.
    """
    try:
        anyio.from_thread.run(inference_limit.acquire)
    except RuntimeError:  # not an event-loop worker thread
        yield
        return
    try:
        yield
    finally:
        anyio.from_thread.run_sync(inference_limit.release)


@lru_cache(maxsize=2)
def _load_checkpoint(model_path: str, mtime: float) -> dict:
    """Read a checkpoint from disk once per (path, mtime).

    Tables each get their own PPOAgent (recurrent memory is keyed by seat, so
    sharing one agent would cross-contaminate games), but they all
    load_state_dict from this cached dict instead of re-reading the file.
    """
    logging.info("Reading AI checkpoint %s", model_path)
    import torch

    return torch.load(model_path, map_location=device)


def load_agent(model_path: str) -> PPOAgent:
    if not model_path:
        raise ValueError("model_path is required")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    checkpoint = _load_checkpoint(model_path, os.path.getmtime(model_path))
    # Arch-aware construction from checkpoint metadata (legacy checkpoints
    # without the "arch" key are the full architecture).
    agent = ppo.load_agent(model_path, checkpoint=checkpoint)
    agent.reset_recurrent_state()
    return agent
