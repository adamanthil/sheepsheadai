"""Deploy-time convention wrapper around a PPO agent (design E4).

Wraps any agent exposing ``act(state, valid_actions, player_id, deterministic)``
/ ``observe`` / ``reset_recurrent_state`` and overrides ONLY defender lead
decisions by restricting the valid-action set the inner agent sees (the policy
renormalizes over the restriction, so the wrapped agent still chooses *which*
card within the convention):

  * C1 — never lead trump: when a true defender leads on trick <= c1_max_trick
    with both a trump and a fail lead legal, trump leads are masked. Default
    scope is tricks 0-1, the diagnosed leak (unconditional masking at later
    tricks is NOT convention — see the E3 exception study).
  * C2 — lead the called suit through: at a true defender's trick-0 lead in
    called-ace mode (non-alone, non-under) holding a called-suit fail with an
    alternative, the lead is restricted to called-suit fails. Trick 0 only:
    that is the pre-registered primary slice, and it is the only trick where
    "called suit not yet led" is provable from the observation dict alone.

Everything else — non-lead plays, bidding, bury, leasters, picker/partner
seats — passes through untouched. All logic reads the per-seat observation
dict only (no game object access), so the wrapper is deployable anywhere the
raw agent is.

Evaluation hook: ``rigorous_eval`` accepts ``model.pt@c1`` / ``model.pt@c2`` /
``model.pt@c1c2`` specs to seat wrapped arms against raw anchors.
"""

from __future__ import annotations

from sheepshead import ACTIONS, DECK_IDS, FAIL, PARTNER_BY_CALLED_ACE, TRUMP

_TRUMP_SET = set(TRUMP)
_FAIL_SET = set(FAIL)
_JD_ID = DECK_IDS["JD"]
_ID_TO_CARD = {v: k for k, v in DECK_IDS.items()}


def _lead_card(action_id: int) -> str | None:
    name = ACTIONS[action_id - 1]
    if not name.startswith("PLAY "):
        return None
    card = name[5:]
    return card if card in _TRUMP_SET or card in _FAIL_SET else None


class ConventionWrapper:
    """Mask convention-violating defender leads; delegate everything else."""

    def __init__(self, agent, c1: bool = True, c2: bool = True, c1_max_trick: int = 1):
        self._agent = agent
        self.c1 = c1
        self.c2 = c2
        self.c1_max_trick = c1_max_trick

    # ------------------------------------------------------------------ #
    def act(self, state, valid_actions, player_id, deterministic: bool = True):
        restricted = self._restrict(state, list(valid_actions))
        return self._agent.act(
            state, restricted, player_id, deterministic=deterministic
        )

    def observe(self, state, player_id):
        return self._agent.observe(state, player_id=player_id)

    def reset_recurrent_state(self):
        return self._agent.reset_recurrent_state()

    def __getattr__(self, name):
        # Delegate anything else (probes, temperatures, ...) to the inner agent.
        return getattr(self._agent, name)

    # ------------------------------------------------------------------ #
    def _restrict(self, state, valid: list[int]) -> list[int]:
        if not int(state["play_started"]) or int(state["is_leaster"]):
            return valid
        if int(state["leader_rel"]) != 1:
            return valid
        if any(int(c) for c in state["trick_card_ids"]):
            return valid  # someone already played: not a lead
        # True defender only: not picker, not revealed partner, not secret partner.
        if int(state["picker_rel"]) == 1 or int(state["partner_rel"]) == 1:
            return valid
        alone = bool(int(state["alone_called"]))
        called_id = int(state["called_card_id"])
        hand_ids = {int(x) for x in state["hand_ids"] if int(x)}
        if int(state["partner_mode"]) == PARTNER_BY_CALLED_ACE:
            if not alone and called_id and called_id in hand_ids:
                return valid  # secret partner
        else:
            if not alone and _JD_ID in hand_ids:
                return valid  # secret partner (JD)

        plays = [(a, _lead_card(a)) for a in valid]
        plays = [(a, c) for a, c in plays if c is not None]
        if len(plays) < 2:
            return valid

        # C2: trick-0 called-suit lead (a fail lead, so C1-compatible).
        if (
            self.c2
            and int(state["current_trick"]) == 0
            and int(state["partner_mode"]) == PARTNER_BY_CALLED_ACE
            and not alone
            and called_id
            and not int(state["called_under"])
        ):
            called_suit = _ID_TO_CARD[called_id][-1]
            conv = [
                a for a, c in plays if c in _FAIL_SET and c[-1] == called_suit
            ]
            if conv and len(conv) < len(plays):
                return conv

        # C1: mask trump leads while a fail lead is legal (tricks <= max_trick).
        if self.c1 and int(state["current_trick"]) <= self.c1_max_trick:
            trump = [a for a, c in plays if c in _TRUMP_SET]
            fails = [a for a, c in plays if c in _FAIL_SET]
            if trump and fails:
                return [a for a in valid if a not in set(trump)]
        return valid


VALID_WRAPS = ("c1", "c2", "c1c2")


def parse_wrap_spec(spec: str):
    """Split a ``path[@c1|@c2|@c1c2]`` model spec into (path_str, wrap|None)."""
    if "@" not in spec:
        return spec, None
    path, wrap = spec.rsplit("@", 1)
    if wrap not in VALID_WRAPS:
        raise ValueError(f"unknown wrap spec '@{wrap}' in {spec!r}")
    return path, wrap


def wrap_agent(agent, wrap: str):
    if wrap not in VALID_WRAPS:
        raise ValueError(f"unknown convention wrap {wrap!r}; use one of {VALID_WRAPS}")
    return ConventionWrapper(agent, c1="c1" in wrap, c2="c2" in wrap)
