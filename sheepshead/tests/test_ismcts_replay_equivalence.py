"""De-risking regression tests for the forced-replay unification (Step 7b).

The ISMCTS world builders exist twice — the per-world sequential
``_build_world`` and the batched ``_build_worlds_lockstep`` — and are being
unified behind a single replay director. These tests were added BEFORE that
unification and pass against the dual implementation; they pin, at much wider
node coverage than ``test_batched_pool_matches_sequential``:

* executor equivalence at every root category the director must reproduce
  (pre-pick, partner, first bury, second-bury private root, mid-trick play,
  trick-boundary play, leaster play, both partner modes);
* the sequential drop semantics on an inconsistent deal (previously only the
  lockstep raise path was tested, and only via monkeypatch);
* the organic (non-monkeypatched) lockstep raise -> sequential fallback on a
  mixed good/corrupted pool, including the exact fail-counter mapping.
"""

from __future__ import annotations

import copy
import random

import numpy as np
import pytest
import torch

from sheepshead import ACTIONS, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD, Game
from sheepshead.ismcts import ISMCTSConfig, ISMCTSTeacher
from sheepshead.tests.test_ismcts_exit_regression import (
    _drive_to_second_bury,
    _head,
    _is_private,
)

# Runs real forced replays with network encodes (~1 min).
pytestmark = pytest.mark.slow

SEED = 77201


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _fresh_agent():
    from sheepshead.agent.ppo import PPOAgent

    return PPOAgent(len(ACTIONS))


def _drive_to_node(game, rng, want_head, min_public=0, trick_boundary=None):
    """Random-legal play until the first ``want_head`` decision satisfying the
    filters; returns (observer, forced_public) or None. ``trick_boundary``:
    True -> only accept a play decision leading a NEW trick after >= 1
    completed trick (cards_played is per-trick and 0 at the boundary);
    False -> only accept a mid-trick decision; None -> accept either.
    For "leaster" every seat passes."""
    pass_id = ACTIONS.index("PASS") + 1
    forced_public = []
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                if want_head == "leaster":
                    if game.is_leaster and any(
                        ACTIONS[a - 1].startswith("PLAY ") for a in valid
                    ):
                        return player.position, forced_public
                elif not game.is_leaster and _head(valid) == want_head:
                    at_boundary = game.cards_played == 0 and game.current_trick > 0
                    if len(forced_public) >= min_public and (
                        trick_boundary is None
                        or (want_head != "play")
                        or (at_boundary == trick_boundary)
                    ):
                        return player.position, forced_public
                if want_head == "leaster" and pass_id in valid:
                    action_id = pass_id
                else:
                    action_id = rng.choice(sorted(valid))
                if not _is_private(valid):
                    forced_public.append((player.position, action_id))
                player.act(action_id)
                valid = player.get_valid_action_ids()
    return None


# (label, want_head, partner_mode, kwargs for _drive_to_node)
_PANEL_SPECS = [
    ("prepick", "pick", PARTNER_BY_CALLED_ACE, {}),
    ("pick_after_passes", "pick", PARTNER_BY_JD, {"min_public": 2}),
    ("partner_jd", "partner", PARTNER_BY_JD, {}),
    ("partner_ca", "partner", PARTNER_BY_CALLED_ACE, {}),
    ("first_bury", "bury", PARTNER_BY_CALLED_ACE, {}),
    ("play_midtrick_ca", "play", PARTNER_BY_CALLED_ACE, {"trick_boundary": False}),
    ("play_boundary_jd", "play", PARTNER_BY_JD, {"trick_boundary": True}),
    ("leaster", "leaster", PARTNER_BY_CALLED_ACE, {}),
]
_NODES_PER_SPEC = 3
_WORLDS_PER_NODE = 4


def _collect_panel_nodes():
    """Deterministic (game, observer, forced_public, label) panel across the
    root categories, plus second-bury private roots."""
    nodes = []
    for label, want, mode, kwargs in _PANEL_SPECS:
        found = 0
        for game_seed in range(300):
            game = Game(partner_selection_mode=mode, seed=game_seed)
            out = _drive_to_node(game, random.Random(SEED + game_seed), want, **kwargs)
            if out is None:
                continue
            observer, forced_public = out
            nodes.append((game, observer, forced_public, label))
            found += 1
            if found >= _NODES_PER_SPEC:
                break
        assert found > 0, f"no node found for panel spec {label}"
    # Second-bury private roots (deterministic drive, JD mode).
    found = 0
    for game_seed in range(300):
        game = Game(partner_selection_mode=PARTNER_BY_JD, seed=game_seed)
        out = _drive_to_second_bury(game)
        if out is None or len(game.bury) != 1:
            continue
        observer, forced_public = out
        nodes.append((game, observer, forced_public, "second_bury"))
        found += 1
        if found >= _NODES_PER_SPEC:
            break
    assert found > 0, "no second-bury node found"
    return nodes


def assert_pools_equivalent(seq_pool, batched_pool, label=""):
    """Batched-vs-sequential pool equivalence at the tolerances pinned by
    test_batched_pool_matches_sequential (log_w 1e-2; memories 1e-2, with
    seats unset in the sparse sequential snapshot zero to 1e-6 batched)."""
    assert len(seq_pool) == len(batched_pool), f"{label}: pool size mismatch"
    for (gs, ms, lws), (gb, mb, lwb) in zip(seq_pool, batched_pool):
        for s in range(1, 6):
            assert sorted(gs.players[s - 1].initial_hand) == sorted(
                gb.players[s - 1].initial_hand
            ), f"{label}: seat {s} initial hand mismatch"
            assert gs.players[s - 1].hand == gb.players[s - 1].hand, (
                f"{label}: seat {s} hand mismatch"
            )
        assert gs.history == gb.history, f"{label}: history mismatch"
        assert gs.bury == gb.bury, f"{label}: bury mismatch"
        assert gs.under_card == gb.under_card, f"{label}: under_card mismatch"
        assert abs(lws - lwb) < 1e-2, f"{label}: log_w {lws} vs {lwb}"
        for s in range(1, 6):
            ms_s = ms.get(s)
            if ms_s is None:
                assert mb[s].abs().max().item() < 1e-6, (
                    f"{label}: seat {s} unset in seq but nonzero batched"
                )
            else:
                assert (ms_s - mb[s]).abs().max().item() < 1e-2, (
                    f"{label}: seat {s} memory mismatch"
                )


def test_replay_equivalence_panel():
    """Sequential and lockstep builds agree world-by-world at every root
    category (the control-flow surface the unified director must cover)."""
    _seed()
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, ISMCTSConfig(det_max_tries=2000))
    rng = random.Random(SEED)
    checked = 0
    for game, observer, forced_public, label in _collect_panel_nodes():
        try:
            deals = [
                game.sample_determinization(observer, rng)
                for _ in range(_WORLDS_PER_NODE)
            ]
        except RuntimeError:
            continue  # rare unsatisfiable node; panel breadth covers it
        seq_pool = teacher._build_pool_sequential(
            game, [copy.deepcopy(d) for d in deals], forced_public, observer
        )
        assert len(seq_pool) == _WORLDS_PER_NODE, (
            f"{label}: sequential build dropped a consistent deal"
        )
        batched_pool = teacher._build_worlds_lockstep(
            copy.deepcopy(game),
            [copy.deepcopy(d) for d in deals],
            forced_public,
            observer,
        )
        assert_pools_equivalent(seq_pool, batched_pool, label)
        # Private roots must have replayed the completed private actions.
        if label == "second_bury":
            for world, _, _ in batched_pool:
                assert world.bury == game.bury, (
                    "second-bury root did not replay the first bury"
                )
        checked += 1
    assert checked >= len(_PANEL_SPECS), f"only {checked} panel nodes checked"


def _find_corruptible_node_and_deal(agent, rng):
    """A play node plus (good_deal, corrupted_deal): the corrupted deal removes
    from a non-observer, non-picker seat the card it is recorded to play and
    swaps in a card from another such seat — so the forced replay must find the
    recorded play illegal in that world (FAIL_BAD_PUBLIC)."""
    for game_seed in range(300):
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=game_seed)
        out = _drive_to_node(
            game, random.Random(SEED + game_seed), "play", min_public=8
        )
        if out is None:
            continue
        observer, forced_public = out
        try:
            deal = game.sample_determinization(observer, rng)
        except RuntimeError:
            continue
        # Cards each seat is recorded to play (public record).
        plays_by_seat = {s: [] for s in range(1, 6)}
        for seat, action_id in forced_public:
            name = ACTIONS[action_id - 1]
            if name.startswith("PLAY "):
                plays_by_seat[seat].append(name[len("PLAY ") :])
        candidates = [
            s
            for s in range(1, 6)
            if s != observer and s != game.picker and plays_by_seat[s]
        ]
        for victim in candidates:
            card = plays_by_seat[victim][-1]
            if card not in deal["initial_hands"][victim]:
                continue  # e.g. UNDER token; pick another seat
            donors = [
                s
                for s in range(1, 6)
                if s not in (observer, game.picker, victim)
                and any(c not in plays_by_seat[s] for c in deal["initial_hands"][s])
            ]
            if not donors:
                continue
            donor = donors[0]
            swap_in = next(
                c for c in deal["initial_hands"][donor] if c not in plays_by_seat[donor]
            )
            corrupted = copy.deepcopy(deal)
            corrupted["initial_hands"][victim] = [
                swap_in if c == card else c for c in corrupted["initial_hands"][victim]
            ]
            corrupted["initial_hands"][donor] = [
                card if c == swap_in else c for c in corrupted["initial_hands"][donor]
            ]
            return game, observer, forced_public, deal, corrupted
    raise AssertionError("no corruptible play node found")


def test_sequential_drop_on_inconsistent_deal():
    """_build_world must DROP an inconsistent world — (None, None) plus a
    bad_public counter — never raise (the previously untested half of the
    raise-vs-drop contract)."""
    _seed()
    agent = _fresh_agent()
    rng = random.Random(SEED)
    game, observer, forced_public, good, corrupted = _find_corruptible_node_and_deal(
        agent, rng
    )

    teacher = ISMCTSTeacher(agent, ISMCTSConfig(det_max_tries=2000))
    world, log_w = teacher._build_world(
        game, copy.deepcopy(good), forced_public, observer
    )
    assert world is not None and log_w is not None, "good deal failed to build"
    assert np.isfinite(log_w), "good deal built with a non-finite log-weight"
    assert teacher.fail["bad_public"] == 0

    world, log_w = teacher._build_world(
        game, copy.deepcopy(corrupted), forced_public, observer
    )
    assert world is None and log_w is None, "corrupted deal was not dropped"
    assert teacher.fail["bad_public"] == 1, dict(teacher.fail)


def test_batched_fallback_on_organic_inconsistency():
    """A mixed good/corrupted pool must trigger the REAL (not monkeypatched)
    lockstep raise -> sequential fallback, keep exactly the consistent worlds,
    and produce the exact exception->counter mapping."""
    _seed()
    agent = _fresh_agent()
    rng = random.Random(SEED)
    game, observer, forced_public, good, corrupted = _find_corruptible_node_and_deal(
        agent, rng
    )

    teacher = ISMCTSTeacher(agent, ISMCTSConfig(det_max_tries=2000))
    pool = teacher._build_worlds_batched(
        copy.deepcopy(game),
        [copy.deepcopy(good), copy.deepcopy(corrupted)],
        forced_public,
        observer,
    )
    assert teacher.fail["batched_fallback"] == 1, dict(teacher.fail)
    # bad_public increments once in the lockstep attempt (raise) and once in
    # the sequential fallback (drop) — the mapping the unification must keep.
    assert teacher.fail["bad_public"] == 2, dict(teacher.fail)
    assert len(pool) == 1, f"fallback pool kept {len(pool)} worlds, wanted 1"
    world, memory_snapshot, log_w = pool[0]
    assert world.history == game.history, "surviving world history mismatch"
    assert np.isfinite(log_w)
    assert memory_snapshot, "surviving world lost its memory snapshot"


def _drain_director(real_game, deal, forced_public, observer):
    """Network-free run of the replay director: apply each yielded event to a
    fresh determinized world directly (no encodes, no weights); return the
    event trace and the finished world."""
    from collections import deque

    from sheepshead.ismcts import ISMCTSTeacher, _PrivateDecision, _replay_events

    world = Game(partner_selection_mode=real_game.partner_mode_flag)
    for seat in range(1, 6):
        hand = deal["initial_hands"][seat][:]
        world.players[seat - 1].hand = hand
        world.players[seat - 1].initial_hand = hand[:]
    world.blind = deal["blind"][:]
    det_bury = deque(deal["bury"])
    det_under = deal["under_card"]
    events = []
    for event in _replay_events(real_game, world, forced_public, observer):
        events.append(event)
        player = world.players[event.seat - 1]
        if isinstance(event, _PrivateDecision):
            action_id = ISMCTSTeacher._forced_private(
                player.get_valid_action_ids(), det_bury, det_under
            )
        else:
            action_id = event.action_id
        assert action_id in player.get_valid_action_ids()
        player.act(action_id)
    return world, events


def test_director_event_trace():
    """The director's event sequence is exactly the public record in order
    (with scheme-B weighted flags) interleaved with one _PrivateDecision per
    completed private action, terminating at the root (no raise)."""
    from sheepshead.ismcts import (
        _is_weighted_bidding_action,
        _PrivateDecision,
        _PublicAction,
    )

    _seed()
    rng = random.Random(SEED)
    cases = []
    for game_seed in range(300):  # mid-trick play node
        game = Game(partner_selection_mode=PARTNER_BY_CALLED_ACE, seed=game_seed)
        out = _drive_to_node(
            game, random.Random(SEED + game_seed), "play", trick_boundary=False
        )
        if out is not None:
            cases.append((game, *out))
            break
    for game_seed in range(300):  # second-bury private root
        game = Game(partner_selection_mode=PARTNER_BY_JD, seed=game_seed)
        out = _drive_to_second_bury(game)
        if out is not None and len(game.bury) == 1:
            cases.append((game, *out))
            break
    assert len(cases) == 2, "could not build both director trace cases"

    for game, observer, forced_public in cases:
        deal = game.sample_determinization(observer, rng)
        world, events = _drain_director(game, deal, forced_public, observer)

        public_events = [e for e in events if isinstance(e, _PublicAction)]
        assert [(e.seat, e.action_id) for e in public_events] == list(forced_public), (
            "public events must be the forced record, in order"
        )
        for e in public_events:
            assert e.weighted == _is_weighted_bidding_action(e.action_id)

        private_events = [e for e in events if isinstance(e, _PrivateDecision)]
        expected_private = len(game.bury) + (1 if game.under_card else 0)
        assert len(private_events) == expected_private, (
            f"{len(private_events)} private events, expected {expected_private}"
        )
        assert all(e.seat == game.picker for e in private_events), (
            "private decisions must belong to the picker"
        )
        assert world.history == game.history, "director replay history mismatch"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
