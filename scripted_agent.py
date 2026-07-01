#!/usr/bin/env python3
"""Rule-based Sheepshead agent encoding sound-amateur table conventions.

Role (run-review notebook §5, League_Run_Review_202607.md): a lineage-free
measurement instrument — the only agent in the project that cannot share the
RL family tree's blind spots. Used as:

  * a sanity floor / smoke test (any trained checkpoint that loses to it is
    broken in a way self-relative metrics cannot reveal);
  * a decorrelated opponent for ad-hoc paired probes (via
    ``training_utils.paired_edge``).

It is deliberately kept OUT of the frozen PANEL-A reference field so the
anchored strength scale never changes; if a panel wants it later, define a
new named panel alongside PANEL-A.

Scope is "solid conventions, no card counting": pick on trump density, call
the thinnest suit, bury fail points toward voids, picker leads trump /
defenders never lead trump, schmear the winning teammate, win cheap, duck
leasters. Stateless across tricks (observe() is a no-op), so one instance can
fill any number of seats.

Interface-compatible with PPOAgent for the greedy eval drivers:
``act(state_dict, valid_action_ids, player_id, deterministic) -> (id, 0, 0)``,
plus no-op ``observe`` / ``reset_recurrent_state``. Decisions are pure
functions of the structured observation dict (sheepshead.Player.get_state_dict)
and the valid-action set; any unexpected state falls back to the lowest legal
action id rather than raising mid-game.
"""

from __future__ import annotations

from sheepshead import (
    ACTION_LOOKUP,
    DECK,
    FAIL_POWER,
    TRUMP,
    TRUMP_POWER,
    UNDER_CARD_ID,
)

# Card points (standard Schafkopf/Sheepshead values).
_RANK_POINTS = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2, "9": 0, "8": 0, "7": 0}

_TRUMP_SET = set(TRUMP)


def _card(card_id: int) -> str | None:
    """Card string for a deck id (1..32); None for PAD(0)/UNDER(33)."""
    if 1 <= card_id <= len(DECK):
        return DECK[card_id - 1]
    return None


def _points(card: str | None) -> int:
    return _RANK_POINTS[card[:-1]] if card else 0


def _is_trump(card: str) -> bool:
    return card in _TRUMP_SET


def _fail_suit(card: str) -> str | None:
    """Fail suit letter, or None for trump."""
    return None if _is_trump(card) else card[-1]


def _power(card: str | None, led_suit: str | None) -> int:
    """Trick-taking power of a card given the led suit ('T' = trump led).

    Trump always outranks fail; a fail card only competes when it follows the
    led fail suit. UNDER / unknown cards can never win."""
    if card is None:
        return -1
    if _is_trump(card):
        return 100 + TRUMP_POWER[card]
    if led_suit is not None and led_suit != "T" and card[-1] == led_suit:
        return FAIL_POWER[card]
    return -1


def hand_strength(cards: list[str]) -> int:
    """Trump-density pick score: +3/queen, +2/jack, +1/other trump."""
    score = 0
    for c in cards:
        if c in _TRUMP_SET:
            score += 3 if c.startswith("Q") else 2 if c.startswith("J") else 1
    return score


class ScriptedAgent:
    """Deterministic conventions-following baseline (see module docstring).

    ``pick_threshold`` is the hand_strength() at/above which it picks
    (default 7 ~= a queen plus two supporting trump); ``alone_threshold``
    the score at/above which it goes alone."""

    def __init__(self, pick_threshold: int = 7, alone_threshold: int = 13):
        self.pick_threshold = pick_threshold
        self.alone_threshold = alone_threshold

    # ------------------------------------------------------------- interface
    def reset_recurrent_state(self) -> None:
        pass

    def observe(self, state, player_id=None, valid_actions=None) -> None:
        pass

    def act(self, state, valid_actions, player_id=None, deterministic=True):
        ids = sorted(valid_actions)
        try:
            action = self._decide(state, ids)
        except Exception:  # noqa: BLE001 - never crash a game; play legally
            action = ids[0]
        return action, 0.0, 0.0

    # -------------------------------------------------------------- decision
    def _decide(self, state, ids: list[int]) -> int:
        options = {ACTION_LOOKUP[i]: i for i in ids}
        hand = [c for c in (_card(cid) for cid in state["hand_ids"]) if c]
        strength = hand_strength(hand)

        if "PICK" in options:
            want = strength >= self.pick_threshold
            return options["PICK"] if want else options["PASS"]

        if "JD PARTNER" in options:
            if strength >= self.alone_threshold and "ALONE" in options:
                return options["ALONE"]
            return options["JD PARTNER"]

        calls = {a: i for a, i in options.items() if a.startswith("CALL ")}
        if calls:
            return self._choose_call(calls, options, hand, strength)

        unders = {a: i for a, i in options.items() if a.startswith("UNDER ")}
        if unders:
            return self._choose_discard(unders, hand)

        buries = {a: i for a, i in options.items() if a.startswith("BURY ")}
        if buries:
            return self._choose_discard(buries, hand)

        plays = {
            a.split(" ", 1)[1]: i for a, i in options.items() if a.startswith("PLAY ")
        }
        if plays:
            return self._choose_play(state, plays, hand)

        return ids[0]

    # ----------------------------------------------------------------- calls
    def _choose_call(self, calls, options, hand, strength) -> int:
        if strength >= self.alone_threshold and "ALONE" in options:
            return options["ALONE"]
        plain = {a: i for a, i in calls.items() if not a.endswith(" UNDER")}
        pool = plain or calls
        suit_len = lambda a: sum(  # noqa: E731
            1 for c in hand if not _is_trump(c) and c[-1] == a.split(" ")[1][-1]
        )
        # Thinnest callable suit: fewest of our fail cards blocking the
        # called ace's suit (and the closest thing to a future void).
        best = min(pool, key=lambda a: (suit_len(a), a))
        return pool[best]

    @staticmethod
    def _choose_discard(discards, hand) -> int:
        """BURY / UNDER card choice: dump fail points, shortest suit first,
        keep trump unless forced."""
        suit_count = {}
        for c in hand:
            if not _is_trump(c):
                suit_count[c[-1]] = suit_count.get(c[-1], 0) + 1

        def key(action):
            card = action.split(" ", 1)[1]
            if _is_trump(card):
                # Forced trump discard: lose the weakest trump last.
                return (1, TRUMP_POWER[card], 0)
            # Fail: highest points first, then from the shortest suit.
            return (0, -_points(card), suit_count.get(card[-1], 0))

        return discards[min(discards, key=key)]

    # ------------------------------------------------------------------ play
    def _choose_play(self, state, plays: dict[str, int], hand) -> int:
        trick_ids = [int(x) for x in state["trick_card_ids"]]
        leader_rel = int(state["leader_rel"])
        is_leaster = bool(state["is_leaster"])
        n_played = sum(1 for x in trick_ids if x)

        cards = sorted(plays)  # deterministic
        if n_played == 0:
            choice = (
                self._lead_leaster(cards) if is_leaster else self._lead(state, cards)
            )
            return plays[choice]

        # Current trick state: led suit + best card/seat so far.
        led_card = _card(trick_ids[leader_rel - 1])
        led_suit = (
            ("T" if led_card and _is_trump(led_card) else _fail_suit(led_card))
            if led_card
            else None
        )
        best_rel, best_pow = 0, -1
        trick_points = 0
        for k in range(5):
            rel = ((leader_rel - 1 + k) % 5) + 1
            cid = trick_ids[rel - 1]
            if not cid:
                continue
            card = _card(cid)
            trick_points += _points(card) if cid != UNDER_CARD_ID else 0
            p = _power(card, led_suit)
            if p > best_pow:
                best_pow, best_rel = p, rel

        if is_leaster:
            return plays[self._follow_leaster(cards, led_suit, best_pow)]

        winner_on_my_team = self._same_team(state, best_rel)
        last_to_act = n_played == 4
        winners = [c for c in cards if _power(c, led_suit) > best_pow]

        if winner_on_my_team:
            if last_to_act or best_pow >= 100 + TRUMP_POWER["JD"]:
                # Safe (or near-safe: queen-height trump) -> schmear.
                return plays[self._schmear(cards)]
            if winners and trick_points >= 10:
                return plays[min(winners, key=lambda c: _power(c, led_suit))]
            return plays[self._duck(cards)]

        if winners:
            worth_it = trick_points >= 10 or last_to_act or led_suit != "T"
            if worth_it:
                return plays[min(winners, key=lambda c: _power(c, led_suit))]
        return plays[self._duck(cards)]

    def _lead(self, state, cards: list[str]) -> str:
        on_picker_team = self._same_team(state, 0, leading=True)
        trumps = [c for c in cards if _is_trump(c)]
        fails = [c for c in cards if not _is_trump(c)]
        if on_picker_team:
            # Draw trump from the top; out of trump, cash fail aces.
            if trumps:
                return max(trumps, key=lambda c: TRUMP_POWER[c])
            return max(fails, key=lambda c: (_points(c), FAIL_POWER[c]))
        # Defender: NEVER lead trump while holding fail (the exact tell the
        # 30M lineage leaks). Called suit through first, then fail aces.
        if fails:
            called = _card(int(state["called_card_id"]))
            if called and int(state["current_trick"]) == 0:
                through = [c for c in fails if c[-1] == called[-1]]
                if through:
                    return max(through, key=lambda c: FAIL_POWER[c])
            aces = [c for c in fails if c.startswith("A")]
            if aces:
                return aces[0]
            return max(fails, key=lambda c: FAIL_POWER[c])
        return min(trumps, key=lambda c: TRUMP_POWER[c])

    @staticmethod
    def _lead_leaster(cards: list[str]) -> str:
        return min(
            cards, key=lambda c: (_points(c), _power(c, "T" if _is_trump(c) else c[-1]))
        )

    @staticmethod
    def _follow_leaster(cards: list[str], led_suit, best_pow: int) -> str:
        losers = [c for c in cards if _power(c, led_suit) < best_pow]
        if losers:
            # Duck with the biggest card that still loses (save low escapes).
            return max(losers, key=lambda c: (_power(c, led_suit), -_points(c)))
        return min(cards, key=lambda c: (_points(c), _power(c, led_suit)))

    @staticmethod
    def _schmear(cards: list[str]) -> str:
        return max(cards, key=lambda c: (_points(c), -_power(c, "T")))

    @staticmethod
    def _duck(cards: list[str]) -> str:
        return min(cards, key=lambda c: (_points(c), _power(c, "T")))

    def _same_team(self, state, rel: int, leading: bool = False) -> bool:
        """Is seat ``rel`` (or myself, when ``leading``) on my team?

        Uses public info plus own-hand partner knowledge (holding the called
        card / the JD in JD mode)."""
        picker_rel = int(state["picker_rel"])
        partner_rel = int(state["partner_rel"])
        hand = [c for c in (_card(cid) for cid in state["hand_ids"]) if c]
        called = _card(int(state["called_card_id"]))
        i_am_partner = (
            partner_rel == 1
            or (called is not None and called in hand)
            or (int(state["partner_mode"]) == 0 and "JD" in hand)
        )
        on_picker_team = picker_rel == 1 or i_am_partner
        if leading:
            return on_picker_team
        seat_is_picker_side = rel == picker_rel or (
            partner_rel != 0 and rel == partner_rel
        )
        return seat_is_picker_side == on_picker_team
