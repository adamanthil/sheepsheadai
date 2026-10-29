"""Play one seeded game with an agent in every seat and record each
decision: the forward pass with its auxiliary-head readouts, the action
taken, and the actor's view of the table at that moment.

This is the engine behind the app's /analyze simulation
(server.services.analyze) and the research scanners, which hunt for
seeds that replay to the same decisions in the UI. Both must run this one
path, so it lives below the app and has no web types: the app wraps the
records in its response schemas.
"""

import random
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, Collection, Dict, List, Optional, Set

import numpy as np
import torch

from sheepshead.agent.observation import last_trick_observation_for, observation_for
from sheepshead.agent.ppo import PPOAgent
from sheepshead.agent.sampling import DeploySampling
from sheepshead.game import ACTION_LOOKUP, Game, Player
from sheepshead.rewards import (
    compute_any_unseen_trump_higher_than_hand,
    compute_known_points_rel,
    compute_seen_trump_mask,
    handle_trick_completion,
    update_intermediate_rewards_for_action,
)
from sheepshead.views import build_player_state

# The display names /analyze gives seats 1-5. Deliberately not the app's
# AI_NAME_POOL (server.runtime.occupants): changing these would alter
# user-visible analyze output.
SEAT_NAMES = ("Dan", "Kyle", "Trevor", "John", "Andrew")

# Wraps each model call; the app passes its inference concurrency slot.
InferenceTurn = Callable[[], AbstractContextManager[Any]]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_actor(game: Game) -> Optional[Player]:
    """The player whose turn it is (the one with valid actions), or None."""
    return next((p for p in game.players if p.get_valid_action_ids()), None)


def select_action_id(
    action_probs: torch.Tensor,
    deterministic: bool,
    valid_actions: Optional[Collection[int]] = None,
    sampling: Optional[DeploySampling] = None,
) -> int:
    """Pick a 1-indexed action from the policy: argmax when deterministic,
    else the deployment sampler when one is given (needs the legal set for
    its head group), else a sample from the raw distribution."""
    if deterministic:
        return int(torch.argmax(action_probs, dim=1).item()) + 1
    if sampling is not None and valid_actions is not None:
        log_probs = torch.log(action_probs[0].detach().cpu().clamp(min=1e-30)).numpy()
        return sampling.sample(log_probs, list(valid_actions)) + 1
    dist = torch.distributions.Categorical(action_probs)
    return int(dist.sample().item()) + 1


def memory_drift(
    memory_before: torch.Tensor, memory_after: torch.Tensor
) -> tuple[Optional[float], float]:
    """(cosine distance, norm) across one recurrent-memory update.

    Distance is None when the incoming memory is all zeros (a seat's first
    encode): there is no previous belief state to drift from."""
    norm = float(memory_after.detach().norm().item())
    if not bool(torch.any(memory_before != 0)):
        return None, norm
    distance = float(
        1.0
        - torch.nn.functional.cosine_similarity(
            memory_before.detach().flatten(),
            memory_after.detach().flatten(),
            dim=0,
        ).item()
    )
    return distance, norm


@dataclass
class ActionProbability:
    action_id: int
    action: str
    prob: float
    logit: float


@dataclass
class StepInference:
    """Encoder/actor/critic forward pass results and aux-head readouts for
    one acting player's turn. Aux fields are None for architectures without
    aux heads. Point vectors are in relative seat order (index 0 = the
    actor)."""

    state: Dict[str, Any]
    valid_actions: Set[int]
    encoder_out: Dict[str, Any]
    action_probs: torch.Tensor
    logits: torch.Tensor
    value: torch.Tensor
    win_prob: Optional[float]
    expected_final: Optional[float]
    secret_partner_prob: Optional[float]
    point_estimates_rel: Optional[List[float]]
    known_points_rel: Optional[List[float]]
    seen_trump_mask_probs: Optional[List[float]]
    seen_trump_mask_actual: Optional[List[bool]]
    unseen_trump_higher_than_hand_prob: Optional[float]
    unseen_trump_higher_than_hand_actual: Optional[bool]
    memory_cosine_distance: Optional[float]
    memory_norm: float

    def probabilities(self) -> List[ActionProbability]:
        """Per-valid-action probabilities and logits, sorted descending."""
        action_probs_np = self.action_probs.squeeze().cpu().numpy()
        logits_np = self.logits.squeeze().cpu().numpy()
        probabilities = [
            ActionProbability(
                action_id=valid_action_id,
                action=ACTION_LOOKUP[valid_action_id],
                prob=float(action_probs_np[valid_action_id - 1]),
                logit=float(logits_np[valid_action_id - 1]),
            )
            for valid_action_id in self.valid_actions
        ]
        probabilities.sort(key=lambda x: x.prob, reverse=True)
        return probabilities


def run_inference_step(
    agent: PPOAgent,
    actor_player: Player,
    device: torch.device,
    inference_turn: InferenceTurn = nullcontext,
) -> StepInference:
    """Run the encoder/actor/critic forward pass for the current actor and
    read the auxiliary heads (seen-trump mask, known-points estimate, win
    probability, etc). Advances the actor's recurrent memory."""
    with inference_turn():
        return _run_inference_step(agent, actor_player, device)


def _run_inference_step(
    agent: PPOAgent, actor_player: Player, device: torch.device
) -> StepInference:
    state = observation_for(actor_player, agent)
    valid_actions = actor_player.get_valid_action_ids()

    # Get or init memory for this player
    memory_in = agent.get_recurrent_memory(actor_player.position, device=device)

    # Encode dict state with memory
    encoder_out = agent.encoder.encode_batch(
        [state], memory_in=memory_in.unsqueeze(0), device=device
    )

    # Store updated memory
    memory_out = encoder_out["memory_out"][0]
    agent.set_recurrent_memory(actor_player.position, memory_out)

    memory_cosine_distance, memory_norm = memory_drift(memory_in, memory_out)

    with torch.no_grad():
        # Build mask and hand ids for actor
        action_mask_t = (
            agent.get_action_mask(valid_actions, agent.action_size)
            .unsqueeze(0)
            .to(device)
        )
        hand_ids_t = torch.as_tensor(
            state["hand_ids"], dtype=torch.long, device=device
        ).view(1, -1)

        # Use existing encoder_out for logits and probabilities
        action_probs, logits = agent.actor.forward_with_logits(
            encoder_out,
            action_mask_t,
            hand_ids_t,
            agent.encoder.card,
        )

        value = agent.critic(encoder_out)

        # Auxiliary critic heads via accessor. No-aux architecture variants
        # (e.g. "no-aux", "perceiver") have no adapter or aux heads at all,
        # so every aux-derived field stays None for them.
        win_prob: Optional[float] = None
        expected_final: Optional[float] = None
        secret_partner_prob: Optional[float] = None
        point_vector: Optional[List[float]] = None
        seen_trump_mask_probs: Optional[List[float]] = None
        unseen_trump_higher_than_hand_prob: Optional[float] = None
        if agent.critic.has_aux_heads:
            win_prob, expected_final, secret_partner_prob, point_vector = (
                agent.critic.aux_predictions(encoder_out)
            )
            # _aux_features_single is the critic's overridable seam for aux
            # features: adapter(features) on the pooled critics, but a token
            # readout on perceiver critics — reading critic_adapter(features)
            # directly would feed those archs the vestigial memory vector.
            aux_feat = agent.critic._aux_features_single(encoder_out)
            seen_trump_mask_logits = agent.critic.seen_trump_mask_logits(
                aux_feat, agent.encoder.card
            ).squeeze(0)
            seen_trump_mask_probs = (
                torch.sigmoid(seen_trump_mask_logits).detach().cpu().tolist()
            )
            unseen_trump_higher_than_hand_logit = (
                agent.critic.unseen_trump_higher_than_hand_logits(aux_feat).squeeze(0)
            )
            unseen_trump_higher_than_hand_prob = float(
                torch.sigmoid(unseen_trump_higher_than_hand_logit).item()
            )

    seen_trump_mask_actual: Optional[List[bool]] = None
    unseen_trump_higher_than_hand_actual: Optional[bool] = None
    if seen_trump_mask_probs is not None:
        seen_trump_mask_actual = [
            bool(seen) for seen in compute_seen_trump_mask(actor_player)
        ]
        unseen_trump_higher_than_hand_actual = bool(
            compute_any_unseen_trump_higher_than_hand(actor_player)
        )

    # Actuals exist only to be compared against the aux-head estimates, so
    # they are omitted whenever the estimates are.
    known_points_rel = (
        compute_known_points_rel(actor_player) if agent.critic.has_aux_heads else None
    )

    return StepInference(
        state=state,
        valid_actions=valid_actions,
        encoder_out=encoder_out,
        action_probs=action_probs,
        logits=logits,
        value=value,
        win_prob=win_prob,
        expected_final=expected_final,
        secret_partner_prob=secret_partner_prob,
        point_estimates_rel=point_vector,
        known_points_rel=known_points_rel,
        seen_trump_mask_probs=seen_trump_mask_probs,
        seen_trump_mask_actual=seen_trump_mask_actual,
        unseen_trump_higher_than_hand_prob=unseen_trump_higher_than_hand_prob,
        unseen_trump_higher_than_hand_actual=unseen_trump_higher_than_hand_actual,
        memory_cosine_distance=memory_cosine_distance,
        memory_norm=memory_norm,
    )


@dataclass
class Decision:
    """One recorded decision: who acted, what the policy saw and chose, and
    the actor's table view just before acting."""

    step_index: int
    seat: int
    action_id: int
    inference: StepInference
    view: Dict[str, Any]

    @property
    def seat_name(self) -> str:
        return SEAT_NAMES[self.seat - 1]

    @property
    def action(self) -> str:
        return ACTION_LOOKUP[self.action_id]

    @property
    def value_estimate(self) -> float:
        return float(self.inference.value.item())


@dataclass
class TrickObserve:
    """One seat's recurrent-memory update from a trick-completion observe."""

    after_step_index: int
    trick: int
    seat: int
    memory_cosine_distance: Optional[float]
    memory_norm: float


@dataclass
class TracedGame:
    game: Game
    decisions: List[Decision] = field(default_factory=list)
    observes: List[TrickObserve] = field(default_factory=list)
    # Trainer-shaped transitions (aligned with decisions) for replaying the
    # reward math over the episode.
    transitions: List[Dict[str, Any]] = field(default_factory=list)
    # Oracle-critic event streams per seat (chronological), collected the
    # same way the trainer does with collect_oracle=True, and each
    # decision's (seat, position) in them. Empty without an oracle critic.
    oracle_events: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict)
    oracle_decision_pos: List[tuple[int, int]] = field(default_factory=list)

    @property
    def final(self) -> Optional[Dict[str, Any]]:
        """The final-result payload of the table view, once the game is done."""
        if not self.game.is_done():
            return None
        return build_player_state(self.game.players[0])["view"].get("final")


def play_traced_game(
    agent: PPOAgent,
    *,
    seed: Optional[int],
    partner_mode: int,
    deterministic: bool,
    max_steps: int = 200,
    sampler: Optional[DeploySampling] = None,
    device: Optional[torch.device] = None,
    inference_turn: InferenceTurn = nullcontext,
) -> TracedGame:
    """Seed the RNGs, deal, and play the game to the end (or ``max_steps``
    decisions) with ``agent`` in all five seats.

    The deal follows from ``seed`` alone: Game keeps its own RNG, so the seed
    goes to it as well as to the global generators. The agent's recurrent
    state is reset first, so one agent can be reused across calls.
    """
    if seed is not None:
        set_seed(seed)
    agent.reset_recurrent_state()
    device = device or default_device()
    game = Game(partner_selection_mode=partner_mode, seed=seed)
    traced = TracedGame(game=game)
    has_oracle = getattr(agent, "oracle_critic", None) is not None
    current_trick_transitions: List[Dict[str, Any]] = []
    tricks_observed = 0
    step_index = 0

    while not game.is_done() and step_index < max_steps:
        actor_player = find_actor(game)
        if actor_player is None:
            # No valid actions found, game should be done
            break
        actor_seat = actor_player.position

        inference = run_inference_step(agent, actor_player, device, inference_turn)

        if has_oracle:
            # Must be captured at decision time: hands shrink as cards play.
            seq = traced.oracle_events.setdefault(actor_seat, [])
            traced.oracle_decision_pos.append((actor_seat, len(seq)))
            seq.append(actor_player.get_oracle_state_dict())

        action_id = select_action_id(
            inference.action_probs,
            deterministic,
            inference.valid_actions,
            sampler,
        )

        traced.decisions.append(
            Decision(
                step_index=step_index,
                seat=actor_seat,
                action_id=action_id,
                inference=inference,
                view=build_player_state(actor_player)["view"],
            )
        )

        # Build training-like transition for reward shaping
        transition = {
            "player": actor_player,
            "state": inference.state,
            "action": action_id,
            "log_prob": 0.0,
            "value": float(inference.value.item()),
            "valid_actions": set(inference.valid_actions),
            "intermediate_reward": 0.0,
        }

        # Apply shared intermediate shaping and trick tracking
        update_intermediate_rewards_for_action(
            game,
            actor_player,
            action_id,
            transition,
            current_trick_transitions,
        )

        traced.transitions.append(transition)

        actor_player.act(action_id)

        if handle_trick_completion(game, current_trick_transitions):
            # Propagate an observation for the just-completed trick to all seats
            for seat in game.players:
                memory_before = agent.get_recurrent_memory(seat.position, device=device)
                with inference_turn():
                    agent.observe(
                        last_trick_observation_for(seat, agent),
                        player_id=seat.position,
                    )
                memory_after = agent.get_recurrent_memory(seat.position, device=device)
                distance, norm = memory_drift(memory_before, memory_after)
                traced.observes.append(
                    TrickObserve(
                        after_step_index=step_index,
                        trick=tricks_observed,
                        seat=seat.position,
                        memory_cosine_distance=distance,
                        memory_norm=norm,
                    )
                )
                # Trainer protocol: no observation event after the final trick.
                if has_oracle and not game.is_done():
                    traced.oracle_events.setdefault(seat.position, []).append(
                        seat.get_last_trick_oracle_state_dict()
                    )
            tricks_observed += 1

        step_index += 1

    return traced
