from __future__ import annotations

import asyncio
import logging
import os
from functools import lru_cache

from ppo import PPOAgent, device
from sheepshead import ACTION_IDS

ACTION_SIZE = len(ACTION_IDS)

# Global bound on concurrent torch inference across all tables so many
# simultaneous games can't oversubscribe the CPU (uvicorn runs 1 process).
inference_limit = asyncio.Semaphore(max(1, (os.cpu_count() or 4) // 2))


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
    agent = PPOAgent(action_size=ACTION_SIZE)
    agent.load(model_path, load_optimizers=False, checkpoint=checkpoint)
    agent.reset_recurrent_state()
    return agent
