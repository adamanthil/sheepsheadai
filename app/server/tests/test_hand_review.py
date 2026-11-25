"""Hand review: replaying a recorded hand and comparing a seat's decisions
with the model's.

The hands here are played by an untrained model with the table AI's memory
schedule, so what the review should find is known in advance: a seat that
always took the model's first choice agrees with it everywhere, and a seat
made to do otherwise differs exactly there.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import pytest

from server.services.hand_review import (
    HandRecord,
    ReplayError,
    review_hand,
)
from sheepshead.agent import ppo
from sheepshead.agent.observation import last_trick_observation_for
from sheepshead.agent.ppo import PPOAgent
from sheepshead.agent.traced_game import find_actor, run_inference_step
from sheepshead.game import (
    ACTION_IDS,
    ACTION_LOOKUP,
    ACTIONS,
    UNDER_TOKEN,
    Game,
    Player,
)

# Given the actor and its legal actions ranked by the model, the action to
# take instead of the model's first choice, or None to take that.
Override = Callable[[Player, List[int]], Optional[int]]


@pytest.fixture(scope="module")
def agent() -> PPOAgent:
    return PPOAgent(len(ACTIONS), critic_mode="oracle")


def play_hand(
    agent: PPOAgent, seed: int, override: Optional[Override] = None
) -> HandRecord:
    """Play one hand with the model in every seat and return its record."""
    game = Game(seed=seed)
    agent.reset_recurrent_state()
    hands = {p.position: list(p.hand) for p in game.players}
    blind = list(game.blind)
    log: List[Tuple[int, str]] = []
    while not game.is_done():
        actor = find_actor(game)
        assert actor is not None
        ranked = [
            p.action_id
            for p in run_inference_step(agent, actor, ppo.device).probabilities()
        ]
        action = (override(actor, ranked) if override else None) or ranked[0]
        log.append((actor.position, ACTION_LOOKUP[action]))
        assert actor.act(action)
        if game.was_trick_just_completed and not game.is_done():
            for player in game.players:
                agent.observe(
                    last_trick_observation_for(player, agent),
                    player_id=player.position,
                )

    picker = next((seat for seat, action in log if action == "PICK"), None)
    called = next((a.split()[1] for _, a in log if a.startswith("CALL ")), None)
    under = next((a.split()[1] for _, a in log if a.startswith("UNDER ")), None)
    plays = [(seat, a.split(" ", 1)[1]) for seat, a in log if a.startswith("PLAY ")]
    plays = [(seat, under if card == UNDER_TOKEN else card) for seat, card in plays]
    assert all(card is not None for _, card in plays)
    return HandRecord(
        partner_mode=1,
        double_on_the_bump=True,
        doublers=False,
        hands=hands,
        blind=blind,
        picker=picker,
        alone=any(action == "ALONE" for _, action in log),
        called_card=called,
        under_card=under,
        bury=[a.split()[1] for _, a in log if a.startswith("BURY ")],
        tricks=[
            [(seat, str(card)) for seat, card in plays[i : i + 5]]
            for i in range(0, len(plays), 5)
        ],
    )


def seat_one_picks(actor: Player, ranked: List[int]) -> Optional[int]:
    pick = ACTION_IDS["PICK"]
    return pick if actor.position == 1 and pick in ranked else None


def test_a_hand_the_model_played_agrees_with_it_everywhere(agent):
    record = play_hand(agent, seed=11, override=seat_one_picks)

    review = review_hand(agent, record, seats=[2, 3, 4, 5])

    assert review.decisions
    assert all(d.agreed for d in review.decisions)
    # Nothing to compare where the seat made the model's choice.
    assert all(d.chosen is None and d.model is None for d in review.decisions)
    # The full-information estimate is read at every decision.
    assert all(d.oracle_score is not None for d in review.decisions)


def test_the_picker_is_reviewed_through_call_and_bury(agent):
    record = play_hand(agent, seed=11, override=seat_one_picks)
    assert record.picker == 1

    decisions = review_hand(agent, record, seats=[1]).decisions

    kinds = [d.kind for d in decisions]
    # One decision each to pick and to bury; the two buried cards are one.
    assert kinds.count("pick") == 1
    assert kinds.count("bury") == 1
    bury = next(d for d in decisions if d.kind == "bury")
    assert bury.action.startswith("BURY ") and len(bury.action.split()) == 3
    assert sorted(bury.action.split()[1:]) == sorted(record.bury)
    assert bury.agreed
    assert sum(p for _, p in bury.policy) == pytest.approx(1.0, abs=1e-4)
    assert [d.index for d in decisions] == list(range(len(decisions)))


def test_a_different_choice_is_compared_after_both_branches(agent):
    def last_choice_once(actor: Player, ranked: List[int]) -> Optional[int]:
        # Seat 3's first card with a real choice: the model's last choice.
        if actor.position != 3 or done or len(ranked) < 2:
            return None
        if not ACTION_LOOKUP[ranked[0]].startswith("PLAY "):
            return None
        done.append(True)
        return ranked[-1]

    done: List[bool] = []
    record = play_hand(agent, seed=5, override=last_choice_once)
    assert done, "the seed must give seat 3 a choice of cards"

    decisions = review_hand(agent, record, seats=[3]).decisions

    differing = [d for d in decisions if not d.agreed]
    assert len(differing) == 1
    (d,) = differing
    assert d.kind == "play" and not d.forced
    assert d.action == d.policy[-1][0] and d.model_action == d.policy[0][0]
    assert d.chosen is not None and d.model is not None


def test_a_pick_the_model_would_pass_is_read_over_every_blind(agent):
    passes: Dict[int, bool] = {}

    def note_model_passes(actor: Player, ranked: List[int]) -> Optional[int]:
        if actor.position == 1 and ACTION_IDS["PICK"] in ranked:
            passes[1] = ranked[0] == ACTION_IDS["PASS"]
        return seat_one_picks(actor, ranked)

    record = play_hand(agent, seed=11, override=note_model_passes)
    if not passes.get(1):
        pytest.skip("the untrained model picks this hand itself")

    pick = review_hand(agent, record, seats=[1]).decisions[0]

    assert pick.kind == "pick" and not pick.agreed
    assert pick.pick_all_blinds is not None
    assert pick.pick_actual_blind is not None
    assert pick.pass_actual_deal is not None
    # The comparison uses the all-blinds reading for the pick.
    assert pick.chosen == pick.pick_all_blinds
    assert pick.model == pick.pass_actual_deal


def test_a_record_the_engine_cannot_follow_is_refused(agent):
    record = play_hand(agent, seed=11, override=seat_one_picks)
    seat, card = record.tricks[0][0]
    # The leader plays a card it was never dealt.
    stranger = next(c for c in record.hands[seat % 5 + 1] if c not in record.bury)
    record.tricks[0][0] = (seat, stranger)

    with pytest.raises(ReplayError):
        review_hand(agent, record, seats=[seat])


def test_review_needs_the_first_choice_not_a_sampled_one(agent):
    record = play_hand(agent, seed=11)
    agent.deploy_sampling = object()
    try:
        with pytest.raises(ValueError, match="argmax"):
            review_hand(agent, record, seats=[1])
    finally:
        agent.deploy_sampling = None
