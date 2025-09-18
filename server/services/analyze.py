from __future__ import annotations

import os
import random
from typing import Dict, List
import numpy as np
import torch

from sheepshead import Game, Player, ACTION_LOOKUP, DECK
from server.api.schemas import (
    AnalyzeSimulateRequest,
    AnalyzeSimulateResponse,
    AnalyzeActionDetail,
    AnalyzeProbability,
    AnalyzeGameSummary
)
from server.services.ai_loader import load_agent
from training_utils import (
    update_intermediate_rewards_for_action,
    handle_trick_completion,
    process_episode_rewards,
)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def build_player_state_for_analyze(player: Player) -> Dict[str, any]:
    """Build the per-seat state payload: vector and a readable view (simplified for analyze)."""
    from server.main import build_player_state
    return build_player_state(player)


def infer_phase_from_action_id(action_id: int, action_groups: Dict[str, List[int]]) -> str:
    """Infer the game phase from action ID using PPOAgent action groups."""
    # Convert to 0-indexed
    zero_indexed = action_id - 1

    for phase, indices in action_groups.items():
        if zero_indexed in indices:
            return phase

    return "unknown"


def simulate_game(req: AnalyzeSimulateRequest) -> AnalyzeSimulateResponse:
    """Simulate a full Sheepshead game and return detailed analysis trace."""

    # Set seed if provided
    if req.seed is not None:
        set_seed(req.seed)

    # Load agent
    global_model_path = os.environ.get("SHEEPSHEAD_MODEL_PATH")
    model_path = req.modelPath or global_model_path

    agent = load_agent(model_path)
    if agent is None:
        raise ValueError("Failed to load AI model")

    # agent.set_head_temperatures(partner=3.0)

    # Reset recurrent state before simulation
    agent.reset_recurrent_state()

    # Initialize game with default double_on_the_bump=True (model doesn't consider this anyway)
    partner_mode = 1 if req.partnerMode == 1 else 0  # Convert to Game's expected format
    game = Game(
        partner_selection_mode=partner_mode,
        double_on_the_bump=False
    )

    # Player display names
    players = ['Dan', 'Kyle', 'Trevor', 'John', 'Andrew']

    # Capture initial hands for summary
    initial_hands = {}
    for player in game.players:
        initial_hands[players[player.position - 1]] = sorted(list(player.hand), key=lambda card: DECK.index(card))

    # Capture blind for summary
    initial_blind = list(game.blind) if hasattr(game, 'blind') else []

    # Trace storage
    trace = []
    step_index = 0
    # For reward shaping and discounted return computation
    episode_transitions = []
    current_trick_transitions = []

    # Simulation loop
    while not game.is_done() and step_index < req.maxSteps:
        # Find the current actor seat
        actor_seat = None
        actor_player = None
        for player in game.players:
            valid_actions = player.get_valid_action_ids()
            if valid_actions:
                actor_seat = player.position
                actor_player = player
                break

        if actor_player is None:
            # No valid actions found, game should be done
            break

        # Get state and valid actions
        state = actor_player.get_state_vector()
        valid_actions = actor_player.get_valid_action_ids()

        # Get action mask and compute logits + probabilities
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            # Use the actor's previous hidden state so the critic's value reflects recurrent context
            prev_hidden = agent.actor._hidden_states.get(actor_player.position, None)
            action_probs, logits = agent.get_action_probs_with_logits(state, valid_actions, actor_player.position)

            value = agent.critic(state_tensor, hidden_in=prev_hidden)

        # Choose action
        if req.deterministic:
            action_id = torch.argmax(action_probs, dim=1).item() + 1  # Convert to 1-indexed
        else:
            dist = torch.distributions.Categorical(action_probs)
            action_id = dist.sample().item() + 1  # Convert to 1-indexed

        # Build probabilities list (only for valid actions, sorted descending)
        probabilities = []
        action_probs_np = action_probs.squeeze().cpu().numpy()
        logits_np = logits.squeeze().cpu().numpy()
        for valid_action_id in valid_actions:
            zero_indexed = valid_action_id - 1
            prob = float(action_probs_np[zero_indexed])
            logit = float(logits_np[zero_indexed])
            probabilities.append(AnalyzeProbability(
                actionId=valid_action_id,
                action=ACTION_LOOKUP[valid_action_id],
                prob=prob,
                logit=logit
            ))

        # Sort probabilities by probability descending
        probabilities.sort(key=lambda x: x.prob, reverse=True)

        # Get player state/view
        player_state = build_player_state_for_analyze(actor_player)

        # Infer phase
        phase = infer_phase_from_action_id(action_id, agent.action_groups)

        # Create action detail
        action_detail = AnalyzeActionDetail(
            stepIndex=step_index,
            seat=actor_seat,
            seatName=players[actor_seat - 1],  # Convert to 0-indexed for array access
            phase=phase,
            actionId=action_id,
            action=ACTION_LOOKUP[action_id],
            valueEstimate=float(value.item()),
            discountedReturn=None,
            validActionIds=valid_actions,
            probabilities=probabilities,
            view=player_state["view"],
            state=player_state["state"]  # Always include state vectors
        )

        trace.append(action_detail)

        # Build training-like transition for reward shaping
        transition = {
            'player': actor_player,
            'state': state,
            'action': action_id,
            'log_prob': 0.0,
            'value': float(value.item()),
            'valid_actions': set(valid_actions),
            'intermediate_reward': 0.0,
        }

        # Apply shared intermediate shaping and trick tracking
        update_intermediate_rewards_for_action(
            game,
            actor_player,
            action_id,
            transition,
            current_trick_transitions,
        )

        episode_transitions.append(transition)

        # Apply action
        actor_player.act(action_id)

        # Trick completion rewards, if any
        trick_completed = handle_trick_completion(game, current_trick_transitions)
        if trick_completed:
            # Propagate an observation for the just-completed trick to all seats
            for seat in game.players:
                agent.observe(seat.get_last_trick_state_vector(), player_id=seat.position)

        step_index += 1

    # Get final state and build summary if game is done
    final_payload = None
    game_summary = None
    if game.is_done():
        # Use any player to get the final state
        final_state = build_player_state_for_analyze(game.players[0])
        final_payload = final_state["view"].get("final")

        # Build game summary
        picker_seat = game.picker if hasattr(game, 'picker') and game.picker else 0
        partner_seat = game.partner if hasattr(game, 'partner') and game.partner else 0

        # If partner not revealed yet, try to find the secret partner
        if partner_seat == 0 and picker_seat > 0 and not game.alone_called:
            for player in game.players:
                if player.is_secret_partner:
                    partner_seat = player.position
                    break

        # Get bury cards (from the picker or game)
        bury_cards = []
        if picker_seat > 0:
            picker_player = game.players[picker_seat - 1]
            bury_cards = list(picker_player.bury) if hasattr(picker_player, 'bury') else []
        elif hasattr(game, 'bury'):
            bury_cards = list(game.bury)

        # Get point totals
        picker_points = 0
        defender_points = 0
        if final_payload and final_payload.get("mode") == "standard":
            picker_points = final_payload.get("picker_score", 0)
            defender_points = final_payload.get("defender_score", 0)

        # Get final scores
        scores = [p.get_score() for p in game.players] if final_payload else [0] * 5

        game_summary = AnalyzeGameSummary(
            hands=initial_hands,
            blind=initial_blind,
            picker=players[picker_seat - 1] if picker_seat > 0 else None,
            partner=players[partner_seat - 1] if partner_seat > 0 else None,
            bury=bury_cards,
            pickerPoints=picker_points,
            defenderPoints=defender_points,
            scores=scores
        )

    # Compute discounted returns per action if we have any transitions
    if episode_transitions:
        final_scores = [p.get_score() for p in game.players]
        last_transition_per_player: Dict[int, int] = {}
        for i, tr in enumerate(episode_transitions):
            last_transition_per_player[tr['player'].position] = i

        # Collect per-step rewards and done flags in order
        rewards = [0.0] * len(episode_transitions)
        dones = [False] * len(episode_transitions)
        for i, reward_data in enumerate(process_episode_rewards(
            episode_transitions,
            final_scores,
            last_transition_per_player,
            game.is_leaster,
        )):
            rewards[i] = float(reward_data['reward'])
            dones[i] = bool(reward_data['done'])

        # Group indices by player
        idxs_by_player: Dict[int, List[int]] = {}
        for i, tr in enumerate(episode_transitions):
            idxs_by_player.setdefault(tr['player'].position, []).append(i)

        gamma = getattr(agent, 'gamma', 0.95)
        discounted_by_index: Dict[int, float] = {}
        for _, idxs in idxs_by_player.items():
            ret = 0.0
            for idx in reversed(idxs):
                if dones[idx]:
                    ret = 0.0
                ret = rewards[idx] + gamma * ret
                discounted_by_index[idx] = ret

        # Attach discounted returns and per-step rewards back to trace elements (aligned by order)
        for i, action_detail in enumerate(trace):
            if i in discounted_by_index:
                action_detail.discountedReturn = float(discounted_by_index[i])
            # Always provide the raw per-step reward value for this action
            if i < len(rewards):
                action_detail.stepReward = float(rewards[i])

    # Build response
    response = AnalyzeSimulateResponse(
        meta={
            "partnerMode": req.partnerMode,
            "deterministic": req.deterministic,
            "seed": req.seed,
            "modelPath": model_path or "auto-selected"
        },
        actionLookup={k: v for k, v in ACTION_LOOKUP.items()},
        players=players,
        summary=game_summary,
        trace=trace,
        final=final_payload
    )

    return response
