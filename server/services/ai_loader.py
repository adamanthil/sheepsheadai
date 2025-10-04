from __future__ import annotations

import os
import logging
from typing import Optional, Sequence

from ppo import PPOAgent
from sheepshead import ACTION_IDS


ACTION_SIZE = len(ACTION_IDS)


def load_agent(global_model_path: Optional[str] = None, candidate_paths: Optional[Sequence[str]] = None) -> Optional[PPOAgent]:
    """Load and return a PPOAgent using the first available checkpoint.

    If `global_model_path` is provided and exists, it is preferred. Otherwise
    try each of `candidate_paths` in order. Returns None if loading fails.
    """
    try:
        agent = PPOAgent(action_size=ACTION_SIZE)
    except Exception:
        logging.exception("failed to construct PPOAgent")
        return None

    try:
        if global_model_path and os.path.exists(global_model_path):
            agent.load(global_model_path, load_optimizers=False)
            logging.info("Loaded AI model from %s", global_model_path)
        else:
            paths = list(candidate_paths or [])
            for path in paths:
                if os.path.exists(path):
                    agent.load(path, load_optimizers=False)
                    logging.info("Loaded AI model from %s", path)
                    break
        agent.reset_recurrent_state()
        return agent
    except Exception:
        logging.exception("failed to load AI model")
        return None


