"""The actor observation contract (Training_Program_Redesign §3.2).

``Player.get_state_dict`` is the only observation an actor ever sees. Its
keys split into two sets:

``RECALL_KEYS``
    Everything a human at the table can see or is entitled to remember
    without bookkeeping: the header flags, the called card, the seats and
    roles, the hand, and the trick on the table. Play history reaches the
    network only through its own recurrent memory (DRQN-style; Hausknecht &
    Stone 2015) — nothing here replays it.

``LEGACY_PICKER_MEMORY_KEYS``
    The picker's blind and bury, re-injected into EVERY observation. A human
    picker saw the blind once (it passes through ``hand_ids`` during the
    bury phase) and chose the bury; re-showing both each step is a recall
    violation (Blind_Bury_Ablation_202608 §1). Architectures registered
    before September 2026 consume these keys and stay loadable through them;
    every architecture from ``perceiver-recall`` on must not — the registry
    records which is which (``ArchitectureSpec.legacy_picker_memory``) and
    the encoder's ``observation_keys()`` is the runtime truth.

Why the keys stay in the dict at all: the deployed 30M model and every
evaluation anchor are legacy architectures, and head-to-head instruments
seat a legacy agent and a recall agent at the same table, so both
observation variants have to be producible for one game state. The dict
therefore carries the superset and each ENCODER declares what it reads; a
test pins the recall encoder to ``RECALL_KEYS``.
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
