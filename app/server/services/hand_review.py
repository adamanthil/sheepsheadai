"""Review a finished hand: each decision a seat made, against the model.

The hand is replayed from its record with the table AI's memory fed on the
live schedule (runtime.ai_move): every seat's own decisions, and a
last-trick observation for every seat after each trick but the last. At
each reviewed decision the model's policy, win estimate and expected score
are read from that seat's own point of view.

Where the seat's choice differs from the model's first choice, the two are
compared by what the model thinks of the hand *after* each one:

- Each choice is applied to a copy of the hand as it was actually dealt,
  the model plays every other seat (argmax), and its estimate is read at
  the seat's next decision. A choice that ends the hand before the seat
  decides again is read as the final score instead.
- PICK is the exception: which blind turns up is pure luck, so it is read
  as the mean over every blind the seat could not rule out (all 325 pairs
  of the cards it does not hold), at the partner decision that follows.
  The reading with the blind that was really there is kept alongside.

The bury is one decision, not two: a picker chooses two cards, and the
record does not say which went down first. Its policy is the model's
probability of each pair (over both orders), the model's choice is the pair
it buries taking its first choice twice, and the comparison is made once
both cards are down.

Each reviewed decision also carries the full-information critic's estimate
of the hand at that point, where the model has one: what the hand was
worth with every card known, which moves at once when a play costs
something the seat cannot yet see.

Every forward pass takes one turn at the server's inference gate, so a
review running beside live tables never holds them up for longer than a
single step.

No database or web types here: the caller builds a HandRecord from
wherever the hand is stored.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from typing import Collection, Dict, List, Optional, Sequence, Tuple

import torch

from server.services.ai_loader import inference_turn_in_thread
from sheepshead.agent import ppo
from sheepshead.agent.observation import (
    last_trick_observation_for,
    observation_for,
    oracle_observation_for,
)
from sheepshead.agent.ppo import PPOAgent
from sheepshead.agent.traced_game import find_actor, run_inference_step
from sheepshead.game import ACTION_IDS, ACTION_LOOKUP, DECK, UNDER_TOKEN, Game, Player
from sheepshead.rewards import RETURN_SCALE

PICK = ACTION_IDS["PICK"]
PASS = ACTION_IDS["PASS"]
# Three bands for a difference in expected score, set from how often the
# two choices really ended in different scores when production hands were
# played out both ways (notebooks/Decision_Review_Design_202611.md, 11):
# under EVEN_BELOW the final score almost never moves; from FLAG_FROM it
# moves about four times in ten, nine in ten of those the predicted way;
# in between it is a real but small difference.
EVEN_BELOW = 0.25
FLAG_FROM = 1.0


class ReplayError(ValueError):
    """The stored record does not replay as a legal hand."""


@dataclass(frozen=True)
class HandRecord:
    """What is stored about a finished hand, seats numbered 1-5."""

    partner_mode: int
    double_on_the_bump: bool
    # An all-pass deal at this table is thrown in and redealt, not played
    # as a leaster.
    doublers: bool
    hands: Dict[int, List[str]]
    blind: List[str]
    picker: Optional[int]
    alone: bool
    called_card: Optional[str]
    under_card: Optional[str]
    bury: List[str]
    # Per trick, the (seat, card) plays in the order they were made.
    tricks: List[List[Tuple[int, str]]]


@dataclass(frozen=True)
class Readout:
    """The model's view of the hand from one seat."""

    win_prob: float
    # Expected final score in game points (the critic's value, unscaled).
    score: float
    # True when read off the finished hand rather than estimated.
    terminal: bool = False


@dataclass
class DecisionReview:
    seat: int
    # Order among this seat's decisions in the hand.
    index: int
    kind: str  # pick | call | under | bury | play
    trick: Optional[int]
    # The cards the seat held, and the (seat, card) plays already made to
    # the trick, when it decided.
    hand: List[str]
    table: List[Tuple[int, str]]
    # Engine action names, except the bury: "BURY <card> <card>", the pair
    # in deck order.
    action: str
    model_action: str
    # (action, probability) over the legal actions, most likely first.
    policy: List[Tuple[str, float]]
    forced: bool
    # The model's estimate at the decision, before anything is chosen.
    state: Readout
    # Set where the seat and the model differ (and for every pick decision
    # when asked): the estimate after the seat's choice and after the
    # model's. A deal thrown in and redealt (doublers) reads as a score of 0.
    chosen: Optional[Readout] = None
    model: Optional[Readout] = None
    # Pick decisions only, whenever they are compared: PICK over every
    # possible blind, PICK with the blind that was really there, and PASS.
    pick_all_blinds: Optional[Readout] = None
    pick_actual_blind: Optional[Readout] = None
    pass_actual_deal: Optional[Readout] = None
    # Expected score at the decision with every card known (the oracle
    # critic); None for a model without one.
    oracle_score: Optional[float] = None

    @property
    def agreed(self) -> bool:
        return self.action == self.model_action

    @property
    def difference(self) -> Optional[float]:
        """What the seat's choice gained (+) or cost (-) in expected score,
        to the one decimal it is shown at; None where there is nothing to
        compare."""
        if self.chosen is None or self.model is None or self.agreed:
            return None
        return round(self.chosen.score - self.model.score, 1)

    @property
    def standing(self) -> str:
        """forced | agreed | even | slight | costly | better."""
        if self.forced:
            return "forced"
        difference = self.difference
        if difference is None:
            return "agreed"
        if difference <= -FLAG_FROM:
            return "costly"
        if difference >= FLAG_FROM:
            return "better"
        return "even" if abs(difference) < EVEN_BELOW else "slight"


@dataclass
class HandReview:
    decisions: List[DecisionReview] = field(default_factory=list)
    # Model forward passes spent, for sizing the work.
    evaluations: int = 0


@dataclass(frozen=True)
class ReviewSummary:
    """One seat's reviewed hand in a few figures."""

    # Decisions where the seat had a choice, split by standing.
    decisions: int
    agreed: int
    even: int
    slight: int
    costly: int
    better: int
    # Expected score the seat's choices cost and gained, both >= 0; every
    # difference counts, so gained - cost is the hand's net.
    cost: float
    gained: float
    # Index of the decision that sums the hand up: the costliest, else the
    # best gain; None when no difference reaches FLAG_FROM.
    headline_index: Optional[int]


def summarize(decisions: Sequence[DecisionReview]) -> ReviewSummary:
    standings = [d.standing for d in decisions]
    differences = [(d.difference, d.index) for d in decisions if d.difference]
    worst = min(differences, default=None)
    best = max(differences, default=None)
    headline: Optional[int] = None
    if worst is not None and worst[0] <= -FLAG_FROM:
        headline = worst[1]
    elif best is not None and best[0] >= FLAG_FROM:
        headline = best[1]
    return ReviewSummary(
        decisions=sum(s != "forced" for s in standings),
        agreed=standings.count("agreed"),
        even=standings.count("even"),
        slight=standings.count("slight"),
        costly=standings.count("costly"),
        better=standings.count("better"),
        cost=round(-sum(d for d, _ in differences if d < 0), 1),
        gained=round(sum(d for d, _ in differences if d > 0), 1),
        headline_index=headline,
    )


def _build_game(record: HandRecord) -> Game:
    game = Game(
        double_on_the_bump=record.double_on_the_bump,
        partner_selection_mode=record.partner_mode,
    )
    game.blind = list(record.blind)
    for player in game.players:
        cards = list(record.hands[player.position])
        player.initial_hand = cards
        player.hand = cards[:]
    return game


def _step(game: Game, seat: int, action: str) -> int:
    action_id = ACTION_IDS[action]
    actor = find_actor(game)
    if actor is None or actor.position != seat:
        raise ReplayError(f"{action}: not seat {seat}'s turn")
    if not actor.act(action_id):
        raise ReplayError(f"{action}: illegal for seat {seat}")
    return action_id


def recorded_actions(record: HandRecord) -> List[Tuple[int, int]]:
    """The hand's (seat, action id) sequence, rebuilt from its record and
    checked against the engine.

    Passes are implied by the pick order. The bury is stored as a set, so
    its two cards are replayed in deck order (or the other way round where
    only that is legal)."""
    game = _build_game(record)
    out: List[Tuple[int, int]] = []

    def step(seat: int, action: str) -> None:
        out.append((seat, _step(game, seat, action)))

    for seat in range(1, (record.picker or 6)):
        step(seat, "PASS")
    if record.picker is not None:
        picker = record.picker
        step(picker, "PICK")
        if record.alone:
            step(picker, "ALONE")
        elif record.called_card is None:
            step(picker, "JD PARTNER")
        elif record.under_card is not None:
            step(picker, f"CALL {record.called_card} UNDER")
            step(picker, f"UNDER {record.under_card}")
        else:
            step(picker, f"CALL {record.called_card}")
        if len(record.bury) != 2:
            raise ReplayError("picked hand without a two-card bury")
        first, second = sorted(record.bury, key=DECK.index)
        legal = game.players[picker - 1].get_valid_actions()
        if f"BURY {first}" not in legal:
            first, second = second, first
        step(picker, f"BURY {first}")
        step(picker, f"BURY {second}")
    for trick in record.tricks:
        for seat, card in trick:
            legal = game.players[seat - 1].get_valid_actions()
            if f"PLAY {card}" not in legal and card == record.under_card:
                card = UNDER_TOKEN
            step(seat, f"PLAY {card}")
    return out


def _kind(action: str) -> str:
    if action in ("PICK", "PASS"):
        return "pick"
    if action.startswith("UNDER "):
        return "under"
    if action.startswith("BURY "):
        return "bury"
    if action.startswith("PLAY "):
        return "play"
    return "call"


def _settle_bury_order(
    game: Game,
    actions: List[Tuple[int, int]],
    position: int,
    policy: List[Tuple[int, float]],
) -> int:
    """The action to replay at ``position``, after putting a bury's two cards
    in the order the model likes better.

    The record does not say which card was buried first, and a seat should
    not be marked as differing from the model over an order nobody chose:
    at the first bury decision, the card the model rates higher goes first
    (when the other order is legal). ``actions`` is updated in place."""
    seat, action_id = actions[position]
    if not ACTION_LOOKUP[action_id].startswith("BURY ") or game.bury:
        return action_id
    _, second_id = actions[position + 1]
    prob = dict(policy)
    if prob.get(second_id, 0.0) <= prob[action_id]:
        return action_id
    trial = copy.deepcopy(game).players[seat - 1]
    if not (trial.act(second_id) and trial.act(action_id)):
        return action_id
    actions[position], actions[position + 1] = (seat, second_id), (seat, action_id)
    return second_id


def _bury_name(action_ids: Collection[int]) -> str:
    cards = sorted((ACTION_LOOKUP[a][5:] for a in action_ids), key=DECK.index)
    return "BURY " + " ".join(cards)


def _cards_on_table(game: Game) -> List[Tuple[int, str]]:
    """The plays made so far to the trick in progress, in order."""
    plays = game.history[game.current_trick]
    return [
        (seat, plays[seat - 1])
        for seat in (((game.leader - 1 + k) % 5) + 1 for k in range(game.cards_played))
    ]


class _Reviewer:
    def __init__(self, agent: PPOAgent, record: HandRecord) -> None:
        self.agent = agent
        self.record = record
        self.device = ppo.device
        self.evaluations = 0

    def observe_trick_end(self, game: Game) -> None:
        if not game.was_trick_just_completed or game.is_done():
            return
        for player in game.players:
            self.agent.observe(
                last_trick_observation_for(player, self.agent),
                player_id=player.position,
            )
            self.evaluations += 1

    def read(self, player: Player) -> Tuple[Readout, List[Tuple[int, float]]]:
        """The model's estimate and policy at ``player``'s decision. Advances
        that seat's memory, as its own decision does."""
        with inference_turn_in_thread():
            inference = run_inference_step(self.agent, player, self.device)
        self.evaluations += 1
        if inference.win_prob is None:
            raise ValueError("hand review needs a model with a win-probability head")
        readout = Readout(
            win_prob=inference.win_prob,
            score=float(inference.value.item()) * RETURN_SCALE,
        )
        policy = [(p.action_id, p.prob) for p in inference.probabilities()]
        return readout, policy

    def after(
        self, game: Game, seat: int, action_ids: Sequence[int]
    ) -> Optional[Readout]:
        """The model's estimate for ``seat`` once it has taken ``action_ids``
        (one action, or a bury's two) and the model has played the other
        seats up to its next decision."""
        memories = self.agent.snapshot_player_memories()
        try:
            return self._after(copy.deepcopy(game), seat, action_ids)
        finally:
            self.agent.restore_player_memories(memories)

    def _after(
        self, game: Game, seat: int, action_ids: Sequence[int]
    ) -> Optional[Readout]:
        player = game.players[seat - 1]
        for n, action_id in enumerate(action_ids):
            if n:
                # The seat's memory update at its second bury decision.
                self.agent.observe(observation_for(player, self.agent), player_id=seat)
                self.evaluations += 1
            player.act(action_id)
        self.observe_trick_end(game)
        while not game.is_done():
            if game.is_leaster and self.record.doublers:
                # Thrown in and redealt: nobody scores, nobody wins or loses.
                return Readout(win_prob=0.5, score=0.0, terminal=True)
            actor = find_actor(game)
            if actor is None:
                raise ReplayError("no seat to act in an unfinished hand")
            if actor.position == seat:
                return self.read(actor)[0]
            move, _, _ = self.agent.act(
                observation_for(actor, self.agent),
                valid_actions=actor.get_valid_action_ids(),
                player_id=actor.position,
                deterministic=True,
            )
            self.evaluations += 1
            actor.act(move)
            self.observe_trick_end(game)
        score = float(player.get_score())
        return Readout(win_prob=1.0 if score > 0 else 0.0, score=score, terminal=True)

    def pick_over_blinds(self, game: Game, seat: int) -> Readout:
        """PICK, averaged over every blind the seat cannot rule out."""
        held = set(game.players[seat - 1].hand)
        unseen = [card for card in DECK if card not in held]
        memories = self.agent.snapshot_player_memories()
        win = score = 0.0
        blinds = list(itertools.combinations(unseen, 2))
        try:
            for blind in blinds:
                branch = copy.deepcopy(game)
                branch.blind = list(blind)
                picker = branch.players[seat - 1]
                picker.act(PICK)
                readout, _ = self.read(picker)
                win += readout.win_prob
                score += readout.score
                self.agent.restore_player_memories(memories)
        finally:
            self.agent.restore_player_memories(memories)
        return Readout(win_prob=win / len(blinds), score=score / len(blinds))

    def bury_pairs(
        self, game: Game, seat: int, first: List[Tuple[int, float]]
    ) -> Tuple[List[Tuple[str, float]], List[int]]:
        """The model's probability of each pair of cards it could bury, most
        likely first, and the pair it buries (its first choice, twice).
        ``first`` is its policy at the first bury decision."""
        memories = self.agent.snapshot_player_memories()
        prob: Dict[str, float] = {}
        model_ids: List[int] = []
        for first_id, p_first in first:
            branch = copy.deepcopy(game)
            picker = branch.players[seat - 1]
            picker.act(first_id)
            _, second = self.read(picker)
            self.agent.restore_player_memories(memories)
            if not model_ids:
                model_ids = [first_id, second[0][0]]
            for second_id, p_second in second:
                name = _bury_name((first_id, second_id))
                prob[name] = prob.get(name, 0.0) + p_first * p_second
        return sorted(prob.items(), key=lambda item: -item[1]), model_ids

    def compare(
        self,
        game: Game,
        seat: int,
        review: DecisionReview,
        chosen_ids: Sequence[int],
        model_ids: Sequence[int],
    ) -> None:
        if review.kind != "pick":
            review.chosen = self.after(game, seat, chosen_ids)
            review.model = self.after(game, seat, model_ids)
            return
        review.pick_all_blinds = self.pick_over_blinds(game, seat)
        review.pick_actual_blind = self.after(game, seat, [PICK])
        review.pass_actual_deal = self.after(game, seat, [PASS])
        values = {PICK: review.pick_all_blinds, PASS: review.pass_actual_deal}
        review.chosen = values[chosen_ids[0]]
        review.model = values[model_ids[0]]


def review_hand(
    agent: PPOAgent,
    record: HandRecord,
    seats: Collection[int],
    skip: Collection[Tuple[int, int]] = (),
    every_pick: bool = False,
) -> HandReview:
    """Review the decisions ``seats`` made in ``record``.

    ``skip`` names decisions by (seat, position in the hand's action
    sequence) that are replayed but not reviewed, e.g. ones the AI made on
    the seat's behalf. ``every_pick`` compares PICK with PASS at every pick
    decision, not only where the seat and the model differ."""
    if agent.deploy_sampling is not None:
        raise ValueError("hand review needs an argmax agent (no deployment sampler)")
    actions = recorded_actions(record)
    agent.reset_recurrent_state()
    game = _build_game(record)
    reviewer = _Reviewer(agent, record)
    result = HandReview()
    counts: Dict[int, int] = {}
    # Per reviewed seat: its oracle observations in order (own decisions
    # and trick ends), and where each reviewed decision sits among them.
    oracle_events: Dict[int, list] = {seat: [] for seat in seats}
    oracle_marks: Dict[int, List[Tuple[int, DecisionReview]]] = {
        seat: [] for seat in seats
    }

    for position, (seat, action_id) in enumerate(actions):
        player = game.players[seat - 1]
        # A bury's second card belongs to the decision made at its first.
        second_bury = bool(game.bury) and ACTION_LOOKUP[action_id].startswith("BURY ")
        index = counts.get(seat, 0)
        if not second_bury:
            counts[seat] = index + 1
        if seat in seats and (seat, position) not in skip and not second_bury:
            readout, policy = reviewer.read(player)
            action_id = _settle_bury_order(game, actions, position, policy)
            kind = _kind(ACTION_LOOKUP[action_id])
            chosen_ids = [action_id]
            model_ids = [policy[0][0]]
            named = [(ACTION_LOOKUP[a], p) for a, p in policy]
            action = ACTION_LOOKUP[action_id]
            model_action = ACTION_LOOKUP[model_ids[0]]
            if kind == "bury":
                named, model_ids = reviewer.bury_pairs(game, seat, policy)
                chosen_ids = [action_id, actions[position + 1][1]]
                action = _bury_name(chosen_ids)
                model_action = _bury_name(model_ids)
            review = DecisionReview(
                seat=seat,
                index=index,
                kind=kind,
                trick=game.current_trick if kind == "play" else None,
                hand=list(player.hand),
                table=_cards_on_table(game) if kind == "play" else [],
                action=action,
                model_action=model_action,
                policy=named,
                forced=len(named) == 1,
                state=readout,
            )
            if not review.agreed or (every_pick and kind == "pick"):
                reviewer.compare(game, seat, review, chosen_ids, model_ids)
            result.decisions.append(review)
            oracle_marks[seat].append((len(oracle_events[seat]), review))
        else:
            # The memory update a seat's own decision makes (runtime.ai_move).
            agent.observe(observation_for(player, agent), player_id=seat)
            reviewer.evaluations += 1
        if seat in oracle_events:
            oracle_events[seat].append(oracle_observation_for(player, agent))
        player.act(action_id)
        if game.was_trick_just_completed and not game.is_done():
            for watcher, events in oracle_events.items():
                events.append(
                    game.players[watcher - 1].get_last_trick_oracle_state_dict()
                )
        reviewer.observe_trick_end(game)

    if agent.oracle_critic is not None:
        for seat, marks in oracle_marks.items():
            if not marks:
                continue
            with inference_turn_in_thread(), torch.no_grad():
                values = agent.oracle_critic.forward_sequences(
                    [oracle_events[seat]], device=ppo.device
                )[0]
            reviewer.evaluations += 1
            for at, review in marks:
                review.oracle_score = float(values[at]) * RETURN_SCALE

    result.evaluations = reviewer.evaluations
    return result
