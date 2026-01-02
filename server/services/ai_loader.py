from __future__ import annotations

import os
import logging

from ppo import PPOAgent
from sheepshead import ACTION_IDS


ACTION_SIZE = len(ACTION_IDS)


def load_agent(model_path: str) -> PPOAgent:
    if not model_path:
        raise ValueError("model_path is required")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)

    agent = PPOAgent(action_size=ACTION_SIZE)
    agent.load(model_path, load_optimizers=False)
    logging.info("Loaded AI model from %s", model_path)
    agent.reset_recurrent_state()
    return agent


