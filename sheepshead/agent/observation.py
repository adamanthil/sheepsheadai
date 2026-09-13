"""The actor observation contract (Training_Program_Redesign §3.2).

``Player.get_state_dict`` is the only observation an actor ever sees: the
header flags, the called card, the seats and roles, the hand, and the trick
on the table — everything a human at the table can see or is entitled to
remember without bookkeeping (``RECALL_KEYS``). Play history reaches the
network only through its own recurrent memory (DRQN-style; Hausknecht &
Stone 2015) — nothing here replays it.

``Player.get_picker_memory`` is a separate interface: the picker's blind
and bury (``LEGACY_PICKER_MEMORY_KEYS``). A human picker saw the blind once
(it passes through ``hand_ids`` during the bury phase) and chose the bury;
re-showing both each step is a recall violation
(Blind_Bury_Ablation_202608 §1). Architectures registered before September
2026 were trained on observations that carried these keys and stay loadable
through them (``ArchitectureSpec.legacy_picker_memory``); every
architecture from ``perceiver-recall`` on never sees them.

``observation_for(player, agent)`` is the one place the two interfaces
meet: it hands a legacy agent the merged dict and everyone else the clean
observation, so the game loops, the evaluation instruments that seat both
kinds of agent at one table, and the app need no knowledge of the keys.
Legacy encoders fail loudly when the keys are missing, which catches any
call site that bypasses the helper.
"""

from __future__ import annotations

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


def needs_picker_memory(agent) -> bool:
    """Whether ``agent`` consumes the legacy picker-memory interface. Read
    off the agent (``PPOAgent.needs_picker_memory``; wrappers delegate to
    the agent they wrap); anything without the attribute — scripted agents,
    random baselines — gets the clean observation. A property that raises
    is re-raised, never read as "no" (a legacy agent silently fed the
    clean observation would encode zeros for its memory)."""
    try:
        return bool(agent.needs_picker_memory)
    except AttributeError:
        if any("needs_picker_memory" in vars(k) for k in type(agent).__mro__):
            raise
        return False


def observation_for(player, agent, trick_index=None) -> dict:
    """The observation ``agent`` acts on at ``player``'s seat: the clean
    ``get_state_dict`` observation, merged with ``get_picker_memory`` only
    for a legacy agent. ``trick_index`` is forwarded to ``get_state_dict``
    (replay/search instruments observe past tricks)."""
    obs = player.get_state_dict(trick_index=trick_index)
    if needs_picker_memory(agent):
        obs.update(player.get_picker_memory())
    return obs


def last_trick_observation_for(player, agent) -> dict:
    """The completed-trick observation ``agent`` updates its memory on at
    ``player``'s seat (``Player.get_last_trick_state_dict``), merged with
    the picker memory only for a legacy agent — the counterpart of
    ``observation_for`` for the end-of-trick ``observe`` calls."""
    obs = player.get_last_trick_state_dict()
    if needs_picker_memory(agent):
        obs.update(player.get_picker_memory())
    return obs
