#!/usr/bin/env python3
"""Regression tests for the ISMCTS ExIt path (Game logic, determinizer,
batched pool build, search-output contract, terminal reward, distillation/PG-mask).

Committed and model-free: uses a freshly-initialized (untrained) PPOAgent and tiny
search budgets, so it is fast and deterministic and asserts STRUCTURE / INVARIANTS
/ MATH rather than learned behavior. These are the contracts that must survive the
batched-MCTS Tier-2 refactor (which reorders RNG/encodes): the batched pool build
must keep matching the sequential reference, the determinizer must stay legal, and
search must keep returning a valid distribution on the legal set.

Run directly (`python test_ismcts_exit_regression.py`) — prints PASS/FAIL per test
and exits non-zero on any failure. pytest-compatible (test_* functions) if added.
"""

from __future__ import annotations

import copy
import random
import sys
import time

import numpy as np
import torch

from sheepshead import (
    ACTIONS,
    DECK,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    UNDER_TOKEN,
    Game,
    get_card_points,
    get_card_suit,
)

SEED = 1234


def _seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def _fresh_agent():
    from ppo import PPOAgent

    return PPOAgent(len(ACTIONS), activation="swish")


def _is_private(valid):
    return any(
        ACTIONS[a - 1].startswith("BURY ") or ACTIONS[a - 1].startswith("UNDER ")
        for a in valid
    )


def _head(valid):
    names = [ACTIONS[a - 1] for a in valid]
    if any(n in ("PICK", "PASS") for n in names):
        return "pick"
    if any(n == "ALONE" or n == "JD PARTNER" or n.startswith("CALL ") for n in names):
        return "partner"
    if any(n.startswith("BURY ") or n.startswith("UNDER ") for n in names):
        return "bury"
    return "play"


# ---------------------------------------------------------------------------
# 1. Game-logic invariants (guard the card-lookup / dead-state perf changes)
# ---------------------------------------------------------------------------
def test_card_lookup_equivalence():
    def ref_suit(card):
        return "T" if card in __import__("sheepshead").TRUMP else card[-1]

    def ref_pts(card):
        if "A" in card:
            return 11
        if "10" in card:
            return 10
        if "K" in card:
            return 4
        if "Q" in card:
            return 3
        if "J" in card:
            return 2
        return 0

    for c in DECK + [UNDER_TOKEN]:
        assert get_card_suit(c) == ref_suit(c), f"suit mismatch {c}"
        assert get_card_points(c) == ref_pts(c), f"points mismatch {c}"
    assert sum(get_card_points(c) for c in DECK) == 120, "deck != 120 points"


def test_point_conservation():
    rng = random.Random(SEED)
    leasters = 0
    for g in range(800):
        mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
        game = Game(partner_selection_mode=mode)
        while not game.is_done():
            for p in game.players:
                v = p.get_valid_action_ids()
                while v:
                    p.act(rng.choice(tuple(v)))
                    v = p.get_valid_action_ids()
        bury_pts = sum(get_card_points(c) for c in game.bury)
        assert sum(game.points_taken) + bury_pts == 120, (
            f"game {g}: points not conserved (leaster={game.is_leaster})"
        )
        assert all(1 <= w <= 5 for w in game.trick_winners), "bad trick winner"
        _ = [game.players[i].get_score() for i in range(5)]
        leasters += 1 if game.is_leaster else 0
    assert leasters > 0, "no leasters sampled (need leaster coverage)"


# ---------------------------------------------------------------------------
# Node collection (no network needed — random legal actions)
# ---------------------------------------------------------------------------
def _drive_to_head(game, rng, want_head, force_pass_for_leaster=False):
    """Random-legal play until the first decision of want_head; return
    (observer, forced_public) at that node, or None. Public actions only in
    forced_public (matches pfsp_runtime / the teacher's replay contract)."""
    from sheepshead import ACTIONS as A

    fp = []
    pass_id = A.index("PASS") + 1
    while not game.is_done():
        for player in game.players:
            valid = player.get_valid_action_ids()
            while valid:
                if (
                    not game.is_leaster
                    and _head(valid) == want_head
                    and want_head != "leaster"
                ):
                    return player.position, list(fp)
                if game.is_leaster and want_head == "leaster":
                    # First leaster play decision for seat 1.
                    if player.position == 1 and any(
                        A[a - 1].startswith("PLAY ") for a in valid
                    ):
                        return player.position, list(fp)
                if force_pass_for_leaster and pass_id in valid:
                    a = pass_id
                else:
                    a = rng.choice(tuple(valid))
                if not _is_private(valid):
                    fp.append((player.position, a))
                player.act(a)
                valid = player.get_valid_action_ids()
    return None


# ---------------------------------------------------------------------------
# 2. Determinizer legality + replay-history reproduction (model-free)
# ---------------------------------------------------------------------------
def _played_by(game):
    pb = {s: [] for s in range(1, 6)}
    for trick in game.history:
        for s in range(1, 6):
            c = trick[s - 1]
            if c and c != UNDER_TOKEN:
                pb[s].append(c)
    return pb


def _assert_deal_legal(game, deal, observer, kind):
    ih, blind, bury, under = (
        deal["initial_hands"],
        deal["blind"],
        deal["bury"],
        deal["under_card"],
    )
    assert sorted([c for s in range(1, 6) for c in ih[s]] + list(blind)) == sorted(
        DECK
    ), f"{kind}: partition != full deck"
    for s in range(1, 6):
        assert len(ih[s]) == 6, f"{kind}: seat {s} not dealt 6"
    assert sorted(ih[observer]) == sorted(game.players[observer - 1].initial_hand), (
        f"{kind}: observer hand altered"
    )
    pb = _played_by(game)
    picker = game.picker
    eight = ih[picker] + list(blind) if picker else None
    for s in range(1, 6):
        pool = eight if (picker and s == picker) else ih[s]
        for c in pb[s]:
            assert c in pool, f"{kind}: seat {s} played {c} not in dealt cards"
    # Voids
    voids = game._play_revealed_voids()
    for s in range(1, 6):
        cur = set(eight if (picker and s == picker) else ih[s]) - set(pb[s])
        if picker and s == picker:
            cur -= set(bury)
            if under:
                cur.discard(under)
        for c in cur:
            assert get_card_suit(c) not in voids[s], (
                f"{kind}: seat {s} void violation {c}"
            )
    if kind == "pick":
        assert bury == [] and under is None, (
            f"{kind}: pre-pick must have empty bury/under"
        )
    if kind == "leaster":
        assert bury == [] and under is None, (
            f"{kind}: leaster must have empty bury/under"
        )
        assert len(blind) == 2, f"{kind}: leaster blind != 2"
        all_played = {c for cards in pb.values() for c in cards}
        for c in blind:
            assert c not in all_played, f"{kind}: leaster blind card {c} was played"


def _replay_reproduces_history(real_game, deal, forced_public, observer):
    """Structural replay (no network): a fresh Game with the determinized hands,
    replaying forced_public + forced bury/under, must reach the same node and
    reproduce the public history exactly."""
    from collections import deque

    from sheepshead import ACTION_IDS

    g = Game(partner_selection_mode=real_game.partner_mode_flag)
    for s in range(1, 6):
        h = deal["initial_hands"][s][:]
        g.players[s - 1].hand = h
        g.players[s - 1].initial_hand = h[:]
    g.blind = deal["blind"][:]
    pub = deque(forced_public)
    det_bury = deque(deal["bury"])
    det_under = deal["under_card"]
    guard = 0
    while True:
        guard += 1
        if guard > 6000:
            return False
        acted = False
        for player in g.players:
            valid = player.get_valid_action_ids()
            while valid:
                if not pub and player.position == observer:
                    return g.history == real_game.history
                if _is_private(valid):
                    is_under = any(ACTIONS[a - 1].startswith("UNDER ") for a in valid)
                    if is_under:
                        if det_under is None:
                            return False
                        aid = ACTION_IDS.get(f"UNDER {det_under}")
                    else:
                        if not det_bury:
                            return False
                        aid = ACTION_IDS.get(f"BURY {det_bury.popleft()}")
                    if aid not in valid:
                        return False
                    player.act(aid)
                else:
                    if not pub or pub[0][0] != player.position:
                        return False
                    _, aid = pub.popleft()
                    if aid not in valid:
                        return False
                    player.act(aid)
                acted = True
                valid = player.get_valid_action_ids()
        if not acted:
            return False


def test_determinizer_legality_and_replay():
    rng = random.Random(SEED)
    specs = [
        ("pick", False),
        ("partner", False),
        ("bury", False),
        ("play", False),
        ("leaster", True),
    ]
    for head, force_pass in specs:
        found = 0
        g = 0
        while found < 5 and g < 400:
            mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
            game = Game(partner_selection_mode=mode)
            out = _drive_to_head(game, rng, head, force_pass_for_leaster=force_pass)
            g += 1
            if out is None:
                continue
            observer, fp = out
            found += 1
            for _ in range(6):
                d = game.sample_determinization(observer, rng)
                _assert_deal_legal(game, d, observer, head)
                assert _replay_reproduces_history(game, d, fp, observer), (
                    f"{head}: replay did not reproduce history"
                )
        assert found > 0, f"no {head} nodes collected"


# ---------------------------------------------------------------------------
# 3. Batched pool build == sequential reference (the key Tier-2 guard)
# ---------------------------------------------------------------------------
def test_batched_pool_matches_sequential():
    from ismcts import ISMCTSConfig, ISMCTSTeacher

    _seed()
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, ISMCTSConfig(det_max_tries=2000))
    rng = random.Random(SEED)
    K = 16
    nodes = 0
    g = 0
    while nodes < 3 and g < 200:
        mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        out = _drive_to_head(game, rng, "play")
        g += 1
        if out is None:
            continue
        observer, fp = out
        nodes += 1
        deals = [game.sample_determinization(observer, rng) for _ in range(K)]

        seq = []
        for d in deals:
            world, lw = teacher._build_world(game, copy.deepcopy(d), fp, observer)
            assert world is not None, "sequential reference failed to build a world"
            mem = {pid: t.detach().clone() for pid, t in agent._player_memories.items()}
            seq.append((world, mem, lw))
        bat = teacher._build_worlds_batched(
            copy.deepcopy(game), [copy.deepcopy(d) for d in deals], fp, observer
        )
        assert len(seq) == len(bat) == K, "pool size mismatch"
        for (gs, ms, lws), (gb, mb, lwb) in zip(seq, bat):
            for s in range(1, 6):
                assert sorted(gs.players[s - 1].initial_hand) == sorted(
                    gb.players[s - 1].initial_hand
                )
                assert gs.players[s - 1].hand == gb.players[s - 1].hand
            assert gs.history == gb.history, "history mismatch batched vs sequential"
            assert abs(lws - lwb) < 1e-2, f"log_w mismatch {lws} vs {lwb}"
            # The sequential reference stores memory sparsely (only seats that
            # acted/observed); the batched build is dense (all 5, zeros for an
            # unacted seat). Both mean "zero memory" for an unset seat, so compare
            # with that equivalence.
            for s in range(1, 6):
                ms_s = ms.get(s)
                if ms_s is None:
                    assert mb[s].abs().max().item() < 1e-6, (
                        f"seat {s} unset in seq but nonzero batched"
                    )
                else:
                    assert (ms_s - mb[s]).abs().max().item() < 1e-2, "memory mismatch"
    assert nodes > 0, "no play nodes collected"


def test_batched_pool_fallback_on_inconsistency():
    """When the batched lockstep raises _ReplayInconsistency (rare: a redeal makes
    a recorded play illegal — void inference is not exhaustive), _build_worlds_batched
    must fall back to the per-world sequential build and still return a valid pool,
    not abort. Forced here by monkeypatching the lockstep to raise."""
    from ismcts import ISMCTSConfig, ISMCTSTeacher, _ReplayInconsistency

    _seed()
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(agent, ISMCTSConfig(det_max_tries=2000))
    rng = random.Random(SEED)
    K = 12
    g = 0
    built = False
    while not built and g < 200:
        mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
        game = Game(partner_selection_mode=mode)
        agent.reset_recurrent_state()
        out = _drive_to_head(game, rng, "play")
        g += 1
        if out is None:
            continue
        observer, fp = out
        deals = [game.sample_determinization(observer, rng) for _ in range(K)]

        def _raise(*a, **k):
            raise _ReplayInconsistency("forced for test")

        teacher._build_worlds_lockstep = _raise  # type: ignore[method-assign]
        pool = teacher._build_worlds_batched(
            copy.deepcopy(game), [copy.deepcopy(d) for d in deals], fp, observer
        )
        assert teacher.fail["batched_fallback"] >= 1, "fallback not taken"
        assert len(pool) == K, f"fallback pool {len(pool)} != {K} (consistent deals)"
        for world, mem, log_w in pool:
            for s in range(1, 6):
                assert len(world.players[s - 1].initial_hand) == 6
            assert world.history == game.history, "fallback world history mismatch"
            assert np.isfinite(log_w), "non-finite log_w"
        built = True
    assert built, "no play node collected for fallback test"


# ---------------------------------------------------------------------------
# 4. search() output contract (invariants that survive Tier-2 reordering)
# ---------------------------------------------------------------------------
def _assert_valid_pi(res):
    pi = np.asarray(res["pi"], dtype=np.float64)
    assert (pi >= -1e-9).all(), "negative mass"
    support = {a for a in range(1, len(pi) + 1) if pi[a - 1] > 0}
    assert support.issubset(set(res["valid"])), "mass off legal set"
    assert res["ess"] >= 0.0, "negative ESS"
    # ok implies a usable target: ESS cleared the floor AND a proper distribution
    # was produced. (ok can be False with ESS >= floor when there were no visit
    # counts, so the converse does not hold.)
    if res["ok"]:
        assert res["ess"] >= ISMCTS_FLOOR, "ok set but ESS below floor"
        assert abs(pi.sum() - 1.0) < 1e-4, f"ok target must sum to 1 (got {pi.sum()})"


ISMCTS_FLOOR = 1.0


def test_search_output_contract():
    from ismcts import ISMCTSConfig, ISMCTSTeacher

    _seed()
    agent = _fresh_agent()
    teacher = ISMCTSTeacher(
        agent,
        ISMCTSConfig(
            iters={"pick": 6, "partner": 6, "bury": 6, "play": 6},
            det_max_tries=400,
            ess_floor=ISMCTS_FLOOR,
        ),
    )
    rng = random.Random(SEED)
    for head, force_pass in [
        ("pick", False),
        ("partner", False),
        ("bury", False),
        ("play", False),
        ("leaster", True),
    ]:
        found = 0
        g = 0
        while found < 2 and g < 300:
            mode = PARTNER_BY_CALLED_ACE if g % 2 else PARTNER_BY_JD
            game = Game(partner_selection_mode=mode)
            agent.reset_recurrent_state()
            out = _drive_to_head(game, rng, head, force_pass_for_leaster=force_pass)
            g += 1
            if out is None:
                continue
            observer, fp = out
            found += 1
            res = teacher.search(game, observer, fp, rng, d_rollout=6)
            _assert_valid_pi(res)
            assert res["valid"] == sorted(
                game.players[observer - 1].get_valid_action_ids()
            )
        assert found > 0, f"no {head} nodes for search contract"


# ---------------------------------------------------------------------------
# 5. Terminal reward contract (pure)
# ---------------------------------------------------------------------------
def test_terminal_reward_contract():
    from training_utils import RETURN_SCALE, process_terminal_rewards

    class _P:
        def __init__(self, pos):
            self.position = pos

    pos = 3
    trans = [{"player": _P(pos)} for _ in range(5)]
    final_scores = [0, 0, 6.0, 0, 0]  # seat 3 scored +6
    out = list(process_terminal_rewards(trans, final_scores, is_leaster=False))
    rewards = [o["reward"] for o in out]
    assert rewards[:-1] == [0.0, 0.0, 0.0, 0.0], "non-terminal steps must be 0"
    assert abs(rewards[-1] - 6.0 / RETURN_SCALE) < 1e-12, (
        "terminal reward != final_score/RETURN_SCALE"
    )
    # Leaster parity: is_leaster is accepted and ignored (same contract).
    out_l = list(process_terminal_rewards(trans, final_scores, is_leaster=True))
    assert [o["reward"] for o in out_l] == rewards, "leaster reward contract differs"


# ---------------------------------------------------------------------------
# 6. Distillation + PG-mask path through play_population_game + update
# ---------------------------------------------------------------------------
def _make_pop_agent(agent, mode, i):
    from pfsp import AgentMetadata, PopulationAgent

    meta = AgentMetadata(
        agent_id=f"reg_opp_{i}",
        creation_time=time.time(),
        parent_id=None,
        training_episodes=0,
        partner_mode=mode,
        activation="swish",
    )
    return PopulationAgent(agent, meta)


def test_distill_pgmask_and_dormant():
    from config import SearchConfig
    from ismcts import ISMCTSConfig, ISMCTSTeacher
    from pfsp_runtime import play_population_game

    _seed()
    agent = _fresh_agent()
    mode = PARTNER_BY_JD
    opps = [_make_pop_agent(_fresh_agent(), mode, i) for i in range(4)]
    teacher = ISMCTSTeacher(
        agent,
        ISMCTSConfig(
            iters={"pick": 6, "partner": 6, "bury": 6, "play": 6},
            det_max_tries=300,
            ess_floor=0.5,
        ),
    )
    determinization_rng = random.Random(SEED)
    # High coverage so distillation reliably fires.
    sc = SearchConfig(
        head_search_fractions={"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 0.9}
    )

    searched = 0
    for gi in range(8):
        game, events, _, _, _ = play_population_game(
            training_agent=agent,
            opponents=opps,
            partner_mode=mode,
            training_agent_position=random.randint(1, 5),
            reward_mode="terminal",
            teacher=teacher,
            determinization_rng=determinization_rng,
            search_config=sc,
        )
        searched += sum(
            1 for e in events if e["kind"] == "action" and e.get("has_search_target")
        )
        agent.store_episode_events(events)
    assert searched > 0, "no search targets produced"
    stats = agent.update(epochs=2, batch_size=16)
    d = stats.get("distill", {})
    assert stats["num_transitions"] > 0
    assert d.get("teacher_kl", -1) >= 0.0, "teacher_kl must be >= 0"
    assert 0.0 < d.get("pg_masked_fraction", 0.0) <= 1.0, (
        "PG-mask fraction out of range"
    )
    assert "value" in stats["critic_losses"], "value loss must be computed"

    # Dormant control: shaped mode with no teacher -> distillation inactive,
    # confirming the PPO trainers are unaffected by the Stage C plumbing.
    _seed()
    agent2 = _fresh_agent()
    opps2 = [_make_pop_agent(_fresh_agent(), mode, i) for i in range(4)]
    for gi in range(6):
        _, events, _, _, _ = play_population_game(
            training_agent=agent2,
            opponents=opps2,
            partner_mode=mode,
            training_agent_position=random.randint(1, 5),
            reward_mode="shaped",
        )
        agent2.store_episode_events(events)
    stats2 = agent2.update(epochs=2, batch_size=16)
    d2 = stats2.get("distill", {})
    assert d2.get("pg_masked_fraction", 0.0) == 0.0, (
        "PG-mask must be dormant without search targets"
    )
    assert abs(d2.get("loss", 0.0)) < 1e-9, (
        "distill loss must be 0 without search targets"
    )


def _generate_searched_events(n_games=5):
    """Play terminal-mode games with a teacher and return the concatenated event
    stream (with search targets on a fraction of transitions)."""
    from config import SearchConfig
    from ismcts import ISMCTSConfig, ISMCTSTeacher
    from pfsp_runtime import play_population_game

    _seed()
    gen = _fresh_agent()
    mode = PARTNER_BY_JD
    opps = [_make_pop_agent(_fresh_agent(), mode, i) for i in range(4)]
    teacher = ISMCTSTeacher(
        gen,
        ISMCTSConfig(
            iters={"pick": 6, "partner": 6, "bury": 6, "play": 6},
            det_max_tries=300,
            ess_floor=0.5,
        ),
    )
    det_rng = random.Random(SEED)
    sc = SearchConfig(
        head_search_fractions={"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 0.9}
    )
    all_events, searched = [], 0
    for _ in range(n_games):
        _, events, _, _, _ = play_population_game(
            training_agent=gen, opponents=opps, partner_mode=mode,
            training_agent_position=random.randint(1, 5), reward_mode="terminal",
            teacher=teacher, determinization_rng=det_rng, search_config=sc,
        )
        searched += sum(
            1 for e in events if e["kind"] == "action" and e.get("has_search_target")
        )
        all_events.append(events)
    assert searched > 0, "no search targets generated"
    return all_events


def test_searched_pg_weight_ab():
    """The PG-mask vs additive-form A/B knob must reach the gradient: two agents
    identical except for searched_pg_weight (0.0 mask vs 1.0 additive), fed the SAME
    event stream, must diverge after one update (the additive PG term on searched
    transitions changes the policy update). Also: the default is the hard mask."""
    assert _fresh_agent().searched_pg_weight == 0.0, "default must be the hard mask (0.0)"

    event_streams = _generate_searched_events(n_games=5)

    def updated_agent(weight):
        _seed()
        a = _fresh_agent()  # identical init across the two calls (re-seeded)
        a.searched_pg_weight = weight
        for events in event_streams:
            a.store_episode_events(copy.deepcopy(events))
        a.update(epochs=1, batch_size=16)
        return a

    a_mask = updated_agent(0.0)
    a_add = updated_agent(1.0)
    max_diff = 0.0
    for p_mask, p_add in zip(a_mask.actor.parameters(), a_add.actor.parameters()):
        max_diff = max(max_diff, (p_mask - p_add).abs().max().item())
    assert max_diff > 1e-9, (
        f"searched_pg_weight had no effect on the actor update (max param diff {max_diff:.2e})"
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
TESTS = [
    test_card_lookup_equivalence,
    test_point_conservation,
    test_determinizer_legality_and_replay,
    test_batched_pool_matches_sequential,
    test_batched_pool_fallback_on_inconsistency,
    test_search_output_contract,
    test_terminal_reward_contract,
    test_distill_pgmask_and_dormant,
    test_searched_pg_weight_ab,
]


def main():
    failures = 0
    for t in TESTS:
        _seed()
        name = t.__name__
        t0 = time.perf_counter()
        try:
            t()
            print(f"PASS  {name}  ({time.perf_counter() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
