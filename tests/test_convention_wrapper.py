#!/usr/bin/env python3
"""ConventionWrapper test suite (sheepshead/agent/convention_wrapper.py).

The wrapper is product code (table agents wrap through it when
SHEEPSHEAD_CONVENTION_WRAP is set), so coverage is product-grade:

  * mask rules — the restriction must bind exactly when it should and never
    otherwise, verified on real engine states found by walking scripted
    self-play games (no hand-built observation dicts);
  * delegation & safety — the inner agent sees only the restricted set, the
    caller's list is never mutated, flags/attributes pass through, and the
    restriction is always a non-empty subset of the legal actions;
  * fuzz — full games with a wrapped random-legal agent in every seat stay
    legal end to end, and the mask demonstrably fires;
  * end-to-end instruments — wrapping deliberately convention-violating
    heroes drives their probe-measured violation rates to exactly zero.
"""

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sheepshead.agent.convention_wrapper import (
    ConventionWrapper,
    parse_wrap_spec,
    wrap_agent,
)
from sheepshead.analysis.called_suit_probe import _called_suit_fail, probe_agent
from sheepshead.analysis.trump_lead_probe import probe_agent as probe_trump
from sheepshead.scripted_agent import ScriptedAgent, _card
from sheepshead import (
    ACTIONS,
    FAIL,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP,
    Game,
)

_TRUMP = set(TRUMP)
_FAIL = set(FAIL)


# --------------------------------------------------------------------------
# Engine-state helpers: find real decision nodes by walking scripted games.
# --------------------------------------------------------------------------
def _decisions(partner_mode, max_seeds):
    """Yield (game, player, state, valid) at EVERY decision of scripted
    self-play games over seeds 0..max_seeds-1."""
    field = ScriptedAgent()
    for seed in range(max_seeds):
        game = Game(partner_selection_mode=partner_mode, seed=seed)
        field.reset_recurrent_state()
        while not game.is_done():
            actor = next(
                (p for p in game.players if p.get_valid_action_ids()), None
            )
            if actor is None:
                break
            valid = actor.get_valid_action_ids()
            yield game, actor, actor.get_state_dict(), list(valid)
            a, _, _ = field.act(
                actor.get_state_dict(), valid, actor.position, deterministic=True
            )
            actor.act(a)


def _leads(partner_mode, max_seeds):
    for game, p, state, valid in _decisions(partner_mode, max_seeds):
        if game.play_started and game.cards_played == 0 and game.leader == p.position:
            yield game, p, state, valid


def _true_defender(game, p):
    return not (
        p.is_picker
        or p.is_partner
        or game.partner == p.position
        or p.is_secret_partner
    )


def _lead_cards(valid):
    """[(action_id, card)] for real-card PLAY actions among valid ids."""
    out = []
    for a in valid:
        name = ACTIONS[a - 1]
        if name.startswith("PLAY "):
            c = name[5:]
            if c in _TRUMP or c in _FAIL:
                out.append((a, c))
    return out


def _c2_shape(game, p, valid):
    """Engine-truth mirror of the wrapper's C2 eligibility (any trick)."""
    if (
        game.is_leaster
        or game.alone_called
        or not game.called_card
        or game.is_called_under
        or game.was_called_suit_played
        or not _true_defender(game, p)
    ):
        return False
    plays = _lead_cards(valid)
    conv = [a for a, c in plays if _called_suit_fail(c, game.called_card)]
    return bool(conv) and len(conv) < len(plays)


def _conv_ids(game, valid):
    return [
        a
        for a, c in _lead_cards(valid)
        if _called_suit_fail(c, game.called_card)
    ]


def _collect(nodes_iter, predicate, n):
    """First n (game, p, state, valid) nodes matching predicate."""
    out = []
    for game, p, state, valid in nodes_iter:
        if predicate(game, p, valid):
            out.append((game, p, state, valid))
            if len(out) >= n:
                break
    return out


# --------------------------------------------------------------------------
# Mask rules
# --------------------------------------------------------------------------
class TestC2MaskRules(unittest.TestCase):
    def test_restricts_to_called_suit_fails_at_trick0(self):
        w = ConventionWrapper(ScriptedAgent(), c1=False, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 200),
            lambda g, p, v: g.current_trick == 0 and _c2_shape(g, p, v),
            8,
        )
        self.assertEqual(len(nodes), 8)
        for game, p, state, valid in nodes:
            restricted = w._restrict(state, list(valid))
            self.assertEqual(sorted(restricted), sorted(_conv_ids(game, valid)))
            self.assertTrue(set(restricted) <= set(valid))
            self.assertTrue(restricted)

    def test_inactive_after_trick0(self):
        # Later-trick eligibility is unknowable from the observation dict, so
        # the wrapper must not force there even when the game state qualifies.
        w = ConventionWrapper(ScriptedAgent(), c1=False, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 300),
            lambda g, p, v: g.current_trick >= 1 and _c2_shape(g, p, v),
            6,
        )
        self.assertGreaterEqual(len(nodes), 3)
        for _, _, state, valid in nodes:
            self.assertEqual(w._restrict(state, list(valid)), valid)

    def test_non_defender_seats_exempt(self):
        w = ConventionWrapper(ScriptedAgent(), c1=True, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 300),
            lambda g, p, v: not g.is_leaster and not _true_defender(g, p),
            8,
        )
        self.assertEqual(len(nodes), 8)
        for _, _, state, valid in nodes:
            self.assertEqual(w._restrict(state, list(valid)), valid)

    def test_alone_and_under_call_exempt_from_c2(self):
        w = ConventionWrapper(ScriptedAgent(), c1=False, c2=True)
        # Alone hands: no partner concept, no called-card obligation.
        alone = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 600),
            lambda g, p, v: g.alone_called and not g.is_leaster,
            3,
        )
        self.assertGreaterEqual(len(alone), 1)
        for _, _, state, valid in alone:
            self.assertEqual(w._restrict(state, list(valid)), valid)
        # Under calls: the picker is void by rule; the premise doesn't hold.
        under = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 800),
            lambda g, p, v: g.is_called_under
            and g.current_trick == 0
            and _true_defender(g, p)
            and not g.is_leaster,
            2,
        )
        self.assertGreaterEqual(len(under), 1)
        for game, p, state, valid in under:
            self.assertEqual(w._restrict(state, list(valid)), valid)

    def test_forced_hands_are_not_decisions(self):
        # All legal leads in the called suit -> nothing to mask.
        w = ConventionWrapper(ScriptedAgent(), c1=False, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 800),
            lambda g, p, v: g.current_trick == 0
            and not g.is_leaster
            and not g.alone_called
            and bool(g.called_card)
            and _true_defender(g, p)
            and len(_conv_ids(g, v)) == len(_lead_cards(v))
            and len(_lead_cards(v)) > 0,
            2,
        )
        for _, _, state, valid in nodes:
            self.assertEqual(w._restrict(state, list(valid)), valid)

    def test_leaster_passthrough(self):
        w = ConventionWrapper(ScriptedAgent(), c1=True, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 300),
            lambda g, p, v: g.is_leaster,
            5,
        )
        self.assertEqual(len(nodes), 5)
        for _, _, state, valid in nodes:
            self.assertEqual(w._restrict(state, list(valid)), valid)


class TestC1MaskRules(unittest.TestCase):
    def _c1_shape(self, g, p, v):
        if g.is_leaster or not _true_defender(g, p):
            return False
        plays = _lead_cards(v)
        return any(c in _TRUMP for _, c in plays) and any(
            c in _FAIL for _, c in plays
        )

    def test_masks_trump_leads_on_tricks_0_and_1(self):
        w = ConventionWrapper(ScriptedAgent(), c1=True, c2=False)
        for mode in (PARTNER_BY_CALLED_ACE, PARTNER_BY_JD):
            nodes = _collect(
                _leads(mode, 300),
                lambda g, p, v: g.current_trick <= 1 and self._c1_shape(g, p, v),
                8,
            )
            self.assertEqual(len(nodes), 8)
            for game, p, state, valid in nodes:
                restricted = w._restrict(state, list(valid))
                expected = [
                    a for a in valid if ACTIONS[a - 1][5:] not in _TRUMP
                ]
                self.assertEqual(restricted, expected)
                self.assertTrue(restricted)

    def test_respects_c1_max_trick(self):
        default = ConventionWrapper(ScriptedAgent(), c1=True, c2=False)
        extended = ConventionWrapper(
            ScriptedAgent(), c1=True, c2=False, c1_max_trick=5
        )
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 400),
            lambda g, p, v: g.current_trick >= 2 and self._c1_shape(g, p, v),
            6,
        )
        self.assertGreaterEqual(len(nodes), 3)
        for _, _, state, valid in nodes:
            self.assertEqual(default._restrict(state, list(valid)), valid)
            self.assertLess(
                len(extended._restrict(state, list(valid))), len(valid)
            )

    def test_no_choice_untouched(self):
        # All-trump or all-fail lead options: C1 has nothing to mask.
        w = ConventionWrapper(ScriptedAgent(), c1=True, c2=False)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 400),
            lambda g, p, v: g.current_trick <= 1
            and not g.is_leaster
            and _true_defender(g, p)
            and (
                all(c in _TRUMP for _, c in _lead_cards(v))
                or all(c in _FAIL for _, c in _lead_cards(v))
            ),
            5,
        )
        self.assertGreaterEqual(len(nodes), 2)
        for _, _, state, valid in nodes:
            self.assertEqual(w._restrict(state, list(valid)), valid)

    def test_jd_secret_partner_exempt(self):
        w = ConventionWrapper(ScriptedAgent(), c1=True, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_JD, 500),
            lambda g, p, v: p.is_secret_partner and self._c1_shape_ignoring_defender(g, p, v),
            3,
        )
        self.assertGreaterEqual(len(nodes), 1)
        for _, _, state, valid in nodes:
            self.assertEqual(w._restrict(state, list(valid)), valid)

    def _c1_shape_ignoring_defender(self, g, p, v):
        if g.is_leaster:
            return False
        plays = _lead_cards(v)
        return any(c in _TRUMP for _, c in plays) and any(
            c in _FAIL for _, c in plays
        )


class TestNonLeadPassthrough(unittest.TestCase):
    def test_bidding_bury_and_follows_untouched(self):
        w = ConventionWrapper(ScriptedAgent(), c1=True, c2=True)
        checked = 0
        for game, p, state, valid in _decisions(PARTNER_BY_CALLED_ACE, 40):
            is_lead = (
                game.play_started
                and game.cards_played == 0
                and game.leader == p.position
            )
            if is_lead:
                continue
            self.assertEqual(w._restrict(state, list(valid)), valid)
            checked += 1
            if checked >= 60:
                break
        self.assertGreaterEqual(checked, 60)


# --------------------------------------------------------------------------
# Delegation & safety
# --------------------------------------------------------------------------
class SpyAgent:
    """Records exactly what the wrapper passes through."""

    def __init__(self):
        self.seen_valid = None
        self.seen_deterministic = None
        self.observes = 0
        self.resets = 0
        self.custom_marker = "spy"

    def act(self, state, valid_actions, player_id, deterministic=True):
        self.seen_valid = list(valid_actions)
        self.seen_deterministic = deterministic
        return valid_actions[0], None, None

    def observe(self, state, player_id):
        self.observes += 1

    def reset_recurrent_state(self):
        self.resets += 1


class TestDelegation(unittest.TestCase):
    def test_act_passes_restriction_and_preserves_inputs(self):
        spy = SpyAgent()
        w = ConventionWrapper(spy, c1=False, c2=True)
        nodes = _collect(
            _leads(PARTNER_BY_CALLED_ACE, 200),
            lambda g, p, v: g.current_trick == 0 and _c2_shape(g, p, v),
            1,
        )
        self.assertEqual(len(nodes), 1)
        game, p, state, valid = nodes[0]
        original = list(valid)
        action, _, _ = w.act(state, valid, p.position, deterministic=False)
        # Inner agent saw exactly the convention set; caller's list untouched.
        self.assertEqual(sorted(spy.seen_valid), sorted(_conv_ids(game, valid)))
        self.assertIs(spy.seen_deterministic, False)
        self.assertEqual(valid, original)
        # Returned action is the inner agent's pick from the restricted set.
        self.assertIn(action, spy.seen_valid)

    def test_observe_reset_and_attribute_delegation(self):
        spy = SpyAgent()
        w = ConventionWrapper(spy)
        w.observe({}, player_id=3)
        w.reset_recurrent_state()
        self.assertEqual(spy.observes, 1)
        self.assertEqual(spy.resets, 1)
        self.assertEqual(w.custom_marker, "spy")


class RandomLegal:
    def __init__(self, seed):
        self.rng = random.Random(seed)

    def act(self, state, valid_actions, player_id, deterministic=True):
        return self.rng.choice(list(valid_actions)), None, None

    def observe(self, state, player_id):
        pass

    def reset_recurrent_state(self):
        pass


class TestFuzzLegality(unittest.TestCase):
    def test_wrapped_random_agent_plays_full_games_legally(self):
        # The strongest product invariant: whatever the mask does, every
        # returned action is legal, the restriction is never empty, and games
        # complete. Random inner agent maximizes state coverage.
        for mode in (PARTNER_BY_CALLED_ACE, PARTNER_BY_JD):
            w = ConventionWrapper(RandomLegal(seed=1234), c1=True, c2=True)
            shrunk = 0
            for seed in range(30):
                game = Game(partner_selection_mode=mode, seed=seed)
                while not game.is_done():
                    actor = next(
                        (p for p in game.players if p.get_valid_action_ids()),
                        None,
                    )
                    if actor is None:
                        break
                    valid = actor.get_valid_action_ids()
                    state = actor.get_state_dict()
                    restricted = w._restrict(state, list(valid))
                    self.assertTrue(restricted, "restriction must never be empty")
                    self.assertTrue(set(restricted) <= set(valid))
                    if len(restricted) < len(valid):
                        shrunk += 1
                    action, _, _ = w.act(state, valid, actor.position)
                    self.assertIn(action, valid)
                    actor.act(action)
                self.assertTrue(game.is_done())
            if mode == PARTNER_BY_CALLED_ACE:
                self.assertGreater(shrunk, 0, "mask never fired: fuzz is vacuous")


# --------------------------------------------------------------------------
# End-to-end through the measurement instruments
# --------------------------------------------------------------------------
class TrumpLeader(ScriptedAgent):
    def _lead(self, state, cards):
        trumps = [c for c in cards if c in _TRUMP]
        if trumps:
            return trumps[0]
        return super()._lead(state, cards)


class CalledSuitAvoider(ScriptedAgent):
    def _lead(self, state, cards):
        called = _card(int(state["called_card_id"]))
        if called:
            off = [c for c in cards if not _called_suit_fail(c, called)]
            if off:
                return off[0]
        return super()._lead(state, cards)


class TestEndToEnd(unittest.TestCase):
    def test_c1_mask_zeroes_a_trump_leader(self):
        raw = probe_trump(TrumpLeader(), n_deals=60, partner_mode=PARTNER_BY_CALLED_ACE)
        self.assertEqual(raw["trump_leads"], raw["opportunities"])  # violates freely
        wrapped = probe_trump(
            ConventionWrapper(TrumpLeader(), c1=True, c2=False),
            n_deals=60,
            partner_mode=PARTNER_BY_CALLED_ACE,
        )
        self.assertGreater(wrapped["opportunities"], 10)
        self.assertEqual(wrapped["trump_leads"], 0)

    def test_c2_force_makes_an_avoider_adherent_at_trick0(self):
        raw = probe_agent(CalledSuitAvoider(), n_deals=60)
        self.assertEqual(raw["adherent_trick0"], 0)  # violates freely
        wrapped = probe_agent(
            ConventionWrapper(CalledSuitAvoider(), c1=False, c2=True), n_deals=60
        )
        self.assertGreater(wrapped["eligible_trick0"], 5)
        self.assertEqual(wrapped["adherent_trick0"], wrapped["eligible_trick0"])
        # Trick-0 only: later-trick eligibility is unknowable from the state
        # dict, so the wrapper must NOT have forced anything there.
        self.assertEqual(
            wrapped["adherent"] - wrapped["adherent_trick0"],
            raw["adherent"] - raw["adherent_trick0"],
        )


class TestWrapSpecs(unittest.TestCase):
    def test_wrap_specs(self):
        self.assertEqual(parse_wrap_spec("m.pt"), ("m.pt", None))
        self.assertEqual(parse_wrap_spec("m.pt@c1c2"), ("m.pt", "c1c2"))
        with self.assertRaises(ValueError):
            parse_wrap_spec("m.pt@bogus")
        w = wrap_agent(ScriptedAgent(), "c1")
        self.assertTrue(w.c1)
        self.assertFalse(w.c2)
        # A typo'd deploy config must fail fast, not silently no-op.
        with self.assertRaises(ValueError):
            wrap_agent(ScriptedAgent(), "c3")


if __name__ == "__main__":
    unittest.main(verbosity=2)
