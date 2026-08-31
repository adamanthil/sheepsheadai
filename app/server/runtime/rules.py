"""House-rule readers for a table's ``rules`` dict.

Deliberately dependency-free so both the runtime and the persistence layer
can read a table's rules without either importing the other.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

ALL_PASS_LEASTERS = "leasters"
ALL_PASS_DOUBLERS = "doublers"


def all_pass_mode(rules: Optional[Mapping[str, Any]]) -> str:
    """What a table does when all five players pass.

    Anything unrecognised reads as leasters: that is the historical
    behaviour, so an old table (or a hand-written rules dict) keeps playing
    the way it always has.
    """
    mode = (rules or {}).get("allPassMode", ALL_PASS_LEASTERS)
    return ALL_PASS_DOUBLERS if mode == ALL_PASS_DOUBLERS else ALL_PASS_LEASTERS


def plays_doublers(rules: Optional[Mapping[str, Any]]) -> bool:
    return all_pass_mode(rules) == ALL_PASS_DOUBLERS
