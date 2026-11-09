"""The actor observation contract (Training_Program_Redesign §3.2).

``Player.get_state_dict`` is the only observation an actor ever sees: the
header flags, the called card, the seats and roles, the hand, and the trick
on the table — everything a human at the table can see or is entitled to
remember without bookkeeping (``RECALL_KEYS``). Play history reaches the
network only through its own recurrent memory (DRQN-style; Hausknecht &
Stone 2015) — nothing here replays it.

``get_state_dict`` is always the current observation, including
``leader_rel`` before play (the seat relative to the dealer). Architectures
trained on an older observation declare what they need changed through
``ArchitectureSpec.legacy_observation``; a current architecture declares
nothing and receives ``get_state_dict`` as it stands:

- ``Player.get_picker_memory`` is a separate interface: the picker's blind
  and bury (``LEGACY_PICKER_MEMORY_KEYS``). A human picker saw the blind
  once (it passes through ``hand_ids`` during the bury phase) and chose the
  bury; re-showing both each step is a recall violation
  (Blind_Bury_Ablation_202608 §1). Architectures registered before
  September 2026 were trained on observations that carried these keys.
- ``leader_rel`` was 0 before play for every architecture registered before
  ``perceiver-recall-seated``; those keep seeing 0, in the oracle's
  full-information observation as well.

``observation_for(player, agent)`` and ``oracle_observation_for(player,
agent)`` are the one place these meet, so the game loops, the evaluation
instruments that seat several kinds of agent at one table, and the app need
no knowledge of them. Legacy encoders fail loudly when the picker-memory
keys are missing, which catches any call site that bypasses the helper; a
bypass of the leader rule cannot fail loudly, so the architecture goldens
capture pick-phase states through these helpers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HEADER_KEYS: tuple[str, ...] = (
    "partner_mode",
    "is_leaster",
    "play_started",
    "current_trick",
    "alone_called",
    "called_card_id",
    "called_under",
    "picker_rel",
    "partner_rel",
    "leader_rel",
    "picker_position",
)
TABLE_KEYS: tuple[str, ...] = (
    "hand_ids",
    "trick_card_ids",
    "trick_is_picker",
    "trick_is_partner_known",
)
RECALL_KEYS: frozenset[str] = frozenset(HEADER_KEYS + TABLE_KEYS)
LEGACY_PICKER_MEMORY_KEYS: frozenset[str] = frozenset({"blind_ids", "bury_ids"})


def hide_pre_play_leader(state: dict) -> dict:
    """``state`` as an architecture trained on ``leader_rel = 0`` before
    play reads it."""
    if int(state["play_started"]):
        return state
    return {**state, "leader_rel": np.uint8(0)}


@dataclass(frozen=True)
class LegacyObservation:
    """How an architecture's observation differs from the current one."""

    picker_memory: bool
    hides_pre_play_leader: bool
    oracle_hides_pre_play_leader: bool = True

    def actor_state(self, state: dict) -> dict:
        """``state`` with the pre-play leader hidden if this contract hides
        it."""
        return hide_pre_play_leader(state) if self.hides_pre_play_leader else state


LEGACY_PICKER_MEMORY = LegacyObservation(picker_memory=True, hides_pre_play_leader=True)
LEGACY_SEATLESS_RECALL = LegacyObservation(
    picker_memory=False, hides_pre_play_leader=True
)
# The residual reads the seat; the oracle it inherited was trained without it.
LEGACY_PICK_SEAT_SURGERY = LegacyObservation(
    picker_memory=False, hides_pre_play_leader=False
)


def legacy_observation(agent) -> LegacyObservation | None:
    """``agent``'s observation contract, read off the agent
    (``PPOAgent.legacy_observation``); None means the current observation,
    which is also what anything without the attribute gets (scripted agents,
    random baselines). A property that raises is re-raised, never read as
    "current" (a legacy agent silently fed the current observation would
    encode zeros for its memory)."""
    try:
        return agent.legacy_observation
    except AttributeError:
        if any("legacy_observation" in vars(k) for k in type(agent).__mro__):
            raise
        return None


def actor_state_for(state: dict, agent) -> dict:
    """``state`` as ``agent`` reads it, for harnesses that hand one
    observation to several agents."""
    legacy = legacy_observation(agent)
    return state if legacy is None else legacy.actor_state(state)


def observation_for(player, agent, trick_index=None) -> dict:
    """The observation ``agent`` acts on at ``player``'s seat: ``get_state_dict``
    adapted by the agent's legacy contract, if it has one. ``trick_index`` is
    forwarded to ``get_state_dict`` (replay/search instruments observe past
    tricks)."""
    obs = player.get_state_dict(trick_index=trick_index)
    legacy = legacy_observation(agent)
    if legacy is None:
        return obs
    if legacy.picker_memory:
        obs.update(player.get_picker_memory())
    return legacy.actor_state(obs)


def oracle_observation_for(player, agent) -> dict:
    """The full-information observation for ``agent``'s oracle critic at
    ``player``'s decision: ``get_oracle_state_dict`` adapted by the agent's
    legacy contract, if it has one."""
    obs = player.get_oracle_state_dict()
    legacy = legacy_observation(agent)
    if legacy is None or not legacy.oracle_hides_pre_play_leader:
        return obs
    return hide_pre_play_leader(obs)


def last_trick_observation_for(player, agent) -> dict:
    """The completed-trick observation ``agent`` updates its memory on at
    ``player``'s seat (``Player.get_last_trick_state_dict``), merged with
    the picker memory only for an agent that reads it — the counterpart of
    ``observation_for`` for the end-of-trick ``observe`` calls."""
    obs = player.get_last_trick_state_dict()
    legacy = legacy_observation(agent)
    if legacy is not None and legacy.picker_memory:
        obs.update(player.get_picker_memory())
    return obs
