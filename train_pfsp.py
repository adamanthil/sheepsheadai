#!/usr/bin/env python3
"""
Prioritized Fictitious Self-Play (PFSP) training for Sheepshead.

This script implements population-based training where a single agent trains against
a diverse population of opponents, ranked using OpenSkill ratings.
"""

import torch
import numpy as np
import random
import time
import os
import csv
from collections import deque
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import copy
from dataclasses import dataclass, field
from openskill.models import PlackettLuce

from ppo import PPOAgent
from pfsp import PFSPPopulation, create_initial_population_from_checkpoints
from training_utils import estimate_hand_strength_category
from sheepshead import (
    Game,
    ACTIONS,
    STATE_SIZE,
    ACTION_IDS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    TRUMP,
    ACTION_LOOKUP,
    UNDER_TOKEN,
    get_card_points,
    filter_by_suit,
)

from train_ppo import analyze_strategic_decisions
from training_utils import (
    process_episode_rewards,
    get_partner_selection_mode,
    save_training_plot,
    update_intermediate_rewards_for_action,
    handle_trick_completion,
)


@dataclass
class PFSPHyperparams:
    # Adaptive exploration for pick head
    low_pick_rate_threshold: float = 20.0  # percent
    high_pick_rate_threshold: float = 50.0  # percent
    # Pick-head temperature control
    pick_temp_base: float = 1.0
    pick_temp_max: float = 2.5
    pick_temp_step: float = 0.2
    pick_rate_hysteresis_down: float = 2.5  # percent below high threshold before cooling

    # Adaptive exploration for partner head (ALONE decision)
    high_alone_rate_threshold: float = 30.0  # percent
    # Partner-head temperature control
    # Base τ=1.0 (no change). When ALONE rate exceeds threshold, raise τ toward max
    # to soften partner selection and sample partner calls more often.
    partner_temp_base: float = 1.0
    partner_temp_max: float = 8.0
    partner_temp_step: float = 0.2
    alone_rate_hysteresis_down: float = 5.0  # percent below threshold before cooling

    # Adaptive exploration for bury head (bury decisions quality)
    # If bury_quality_rate drops below a threshold, temporarily bump bury entropy.
    low_bury_quality_threshold: float = 85.0  # percent
    bury_entropy_bump: float = 0.04          # added to base decayed bury entropy
    bury_entropy_bump_duration: int = 19000  # episodes

    # Entropy schedules (start -> end)
    entropy_pick_start: float = 0.05
    entropy_pick_end: float = 0.02
    entropy_partner_start: float = 0.05
    entropy_partner_end: float = 0.02
    entropy_bury_start: float = 0.04
    entropy_bury_end: float = 0.015
    entropy_play_start: float = 0.05
    entropy_play_end: float = 0.02

    # Shaped reward schedules (percent -> weight).
    shaping_schedule_pick: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    shaping_schedule_partner: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    shaping_schedule_bury: dict[int, float] = field(default_factory=lambda: {0: 1.0})
    shaping_schedule_play: dict[int, float] = field(default_factory=lambda: {0: 1.0})


DEFAULT_HYPERPARAMS = PFSPHyperparams()


def interpolated_weight(schedule: dict, progress_pct: float) -> float:
    """Linear interpolation of schedule weights by percent progress.

    schedule: mapping of percent (0-100) to weight.
    progress_pct: current percent progress in [0, 100].
    """
    if not schedule:
        return 1.0
    # Normalize and sort points by percent
    points = sorted((float(k), float(v)) for k, v in schedule.items())
    # Clamp to endpoints
    if progress_pct <= points[0][0]:
        return points[0][1]
    if progress_pct >= points[-1][0]:
        return points[-1][1]
    # Find segment and interpolate
    for (k0, v0), (k1, v1) in zip(points, points[1:]):
        if k0 <= progress_pct <= k1:
            if k1 == k0:
                return v1
            t = (progress_pct - k0) / (k1 - k0)
            return v0 + t * (v1 - v0)
    # Fallback (should not hit due to clamps)
    return points[-1][1]


def play_population_game(training_agent: PPOAgent,
                        opponents: list,
                        partner_mode: int,
                        training_agent_position: int = 1,
                        shaping_weights: dict | None = None) -> tuple:
    """Play a single game with the training agent and population opponents.

    Returns:
        tuple: (game, episode_transitions, final_scores, training_agent_data)
    """
    game = Game(partner_selection_mode=partner_mode)
    weights = shaping_weights or {"pick": 1.0, "partner": 1.0, "bury": 1.0, "play": 1.0}

    # Reset recurrent states for all agents
    training_agent.reset_recurrent_state()
    for opponent in opponents:
        opponent.agent.reset_recurrent_state()

    # Create position-to-agent mapping
    agents = [None] * 5
    agents[training_agent_position - 1] = training_agent

    opponent_positions = [i for i in range(5) if i != training_agent_position - 1]
    for i, opponent in enumerate(opponents[:4]):  # Take up to 4 opponents
        if i < len(opponent_positions):
            agents[opponent_positions[i]] = opponent.agent

    # Fill any remaining positions with **independent copies** of the training agent
    # so they do not share recurrent state. These filler seats are ignored in rating updates.
    for i in range(5):
        if agents[i] is None:
            agents[i] = copy.deepcopy(training_agent)
            agents[i].reset_recurrent_state()

    # Store transitions only for the training agent
    episode_transitions = []
    current_trick_transitions = []

    # Map positions to population opponents for profile updates
    pos_to_pop_agent = {}
    opp_positions = [i for i in range(1, 6) if i != training_agent_position]
    for i, opp in enumerate(opponents[:len(opp_positions)]):
        pos_to_pop_agent[opp_positions[i]] = opp

    # Hand strength categories captured once at start
    hand_strength_by_pos = {p.position: estimate_hand_strength_category(p.hand) for p in game.players}

    while not game.is_done():
        for player in game.players:
            current_agent = agents[player.position - 1]
            valid_actions = player.get_valid_action_ids()

            while valid_actions:
                state = player.get_state_vector()

                # Get action from appropriate agent
                if current_agent == training_agent:
                    action, log_prob, value = current_agent.act(state, valid_actions, player.position)

                    # Store transition for training agent
                    transition = {
                        'player': player,
                        'state': state,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy(),
                        'intermediate_reward': 0.0,
                    }
                    episode_transitions.append(transition)

                    # Shared intermediate reward shaping and trick tracking (for training agent only)
                    update_intermediate_rewards_for_action(
                        game,
                        player,
                        action,
                        transition,
                        current_trick_transitions,
                        pick_weight=weights["pick"],
                        partner_weight=weights["partner"],
                        bury_weight=weights["bury"],
                        play_weight=weights["play"],
                    )

                else:
                    # Opponent action (stochastic for diversity)
                    action, _, _ = current_agent.act(state, valid_actions, player.position, deterministic=False)

                player.act(action)

                # Handle trick completion; PFSP-specific observation propagation
                trick_completed = handle_trick_completion(
                    game, current_trick_transitions
                )
                if trick_completed:
                    # Emit observations for the completed trick using dedicated accessor
                    for seat in game.players:
                        seat_agent = agents[seat.position - 1]
                        if seat_agent == training_agent:
                            # Update training agent's recurrent hidden state and also store for unroll
                            training_agent.observe(seat.get_last_trick_state_vector(), player_id=seat.position)
                            training_agent.store_observation(seat.get_last_trick_state_vector(), player_id=seat.position)
                        else:
                            seat_agent.observe(seat.get_last_trick_state_vector(), seat.position)

                valid_actions = player.get_valid_action_ids()

                # --- Strategic profile updates for opponents only ---
                pop_agent = pos_to_pop_agent.get(player.position)
                if pop_agent:
                    action_name = ACTION_LOOKUP[action]
                    role = 'picker' if (player.is_picker or player.is_partner) else 'defender'

                    # Pick/pass profiling (only valid during picking phase)
                    if action_name in ("PICK", "PASS"):
                        pop_agent.update_strategic_profile_from_game({
                            'pick_decision': (action_name == "PICK"),
                            'hand_strength_category': hand_strength_by_pos[player.position],
                            'position': player.position,
                        })

                    # Bury profiling
                    if action_name.startswith("BURY "):
                        card = action_name.split()[-1]
                        pop_agent.update_strategic_profile_from_game({
                            'bury_decision': card,
                            'card_points': get_card_points(card),
                        })

                    # ALONE vs partner call profiling
                    if action_name == "ALONE":
                        pop_agent.update_strategic_profile_from_game({'alone_call': True})
                    elif action_name.startswith("CALL "):
                        pop_agent.update_strategic_profile_from_game({'alone_call': False, 'called_card': action_name.split()[1]})

                    # Lead trump (overall and early) and void behavior
                    if action_name.startswith("PLAY "):
                        card = action_name.split()[-1]
                        is_lead = (game.cards_played == 0)
                        is_early = is_lead and (game.current_trick <= 1)
                        if is_lead:
                            pop_agent.update_strategic_profile_from_game({
                                'trump_lead': (card in TRUMP or card == UNDER_TOKEN),
                                'role': role,
                                'is_early_lead': is_early,
                            })

                        # Void logic: only when following and can't follow suit
                        if not is_lead and game.current_suit:
                            can_follow = len(filter_by_suit(player.hand, game.current_suit)) > 0
                            if not can_follow:
                                played_trump = (card in TRUMP or card == UNDER_TOKEN)
                                pop_agent.update_strategic_profile_from_game({
                                    'void_event': True,
                                    'void_played_trump': played_trump,
                                    'schmeared_points': 0 if played_trump else get_card_points(card),
                                })

    final_scores = [player.get_score() for player in game.players]

    # Return training agent specific data
    training_agent_score = final_scores[training_agent_position - 1]
    was_picker = (game.picker == training_agent_position)

    training_agent_data = {
        'score': training_agent_score,
        'was_picker': was_picker,
        'position': training_agent_position
    }

    return game, episode_transitions, final_scores, training_agent_data


def train_pfsp(num_episodes: int = 500000,
               update_interval: int = 2048,
               save_interval: int = 5000,
               strategic_eval_interval: int = 10000,
               population_add_interval: int = 5000,
               cross_eval_interval: int = 20000,
               resume_model: str = None,
               activation: str = 'swish',
               initial_checkpoints: list = None,
               hyperparams: PFSPHyperparams = DEFAULT_HYPERPARAMS):
    """
    PFSP training with population-based opponents.
    """
    print("🚀 Starting PFSP (Population-Based) Training...")
    print("="*80)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Population add interval: {population_add_interval}")
    print(f"  Cross-evaluation interval: {cross_eval_interval}")
    print(f"  Activation function: {activation.upper()}")
    print("  Opponent strategy: POPULATION-BASED (PFSP)")
    print("  Population management: OpenSkill ratings + diversity")
    print("="*80)

    # Create training agent
    training_agent = PPOAgent(STATE_SIZE, len(ACTIONS),
                            lr_actor=5.0e-5,
                            lr_critic=1.5e-4,
                            activation=activation)

    # OpenSkill rating for the training agent
    rating_model = PlackettLuce()
    training_rating = rating_model.rating()
    training_agent_id = "training_agent"

    # Create population
    population = PFSPPopulation(
        max_population_jd=25,
        max_population_called_ace=25,
        population_dir="pfsp_population"
    )

    # Initialize population from checkpoints if provided
    if initial_checkpoints:
        create_initial_population_from_checkpoints(
            population,
            initial_checkpoints,
            activation=activation,
            max_agents_per_mode=10
        )

    # Load or create initial training agent
    start_episode = 0
    if resume_model:
        try:
            training_agent.load(resume_model)
            print(f"✅ Loaded training agent from {resume_model}")
            if 'checkpoint_' in resume_model:
                start_episode = int(resume_model.split('_')[-1].split('.')[0])
                print(f"📍 Resuming from episode {start_episode:,}")
        except Exception as e:
            print(f"❌ Could not load {resume_model}: {e}")
    else:
        print("🆕 Starting fresh training agent")

    # Initialize tracking variables
    picker_scores = deque(maxlen=3000)
    pick_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    pass_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    leaster_window = deque(maxlen=3000)
    alone_call_window = deque(maxlen=3000)
    called_ace_window = deque(maxlen=3000)
    called_under_window = deque(maxlen=3000)
    called_10_window = deque(maxlen=3000)
    team_point_differences = deque(maxlen=3000)
    picker_window = deque(maxlen=3000)

    training_data = {
        'episodes': [],
        'picker_avg': [],
        'called_pick_rate': [],
        'jd_pick_rate': [],
        'learning_rate': [],
        'time_elapsed': [],
        'pick_hand_correlation': [],
        'picker_trump_rate': [],
        'defender_trump_rate': [],
        'bury_quality_rate': [],
        'team_point_diff': [],
        'alone_rate': [],
        'leaster_rate': [],
        'strategic_episodes': [],
        'population_stats': []
    }

    # Create checkpoint directory
    checkpoint_dir = f'pfsp_checkpoints_{activation}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    # CSV log files for ongoing progress/metrics
    progress_csv = os.path.join(checkpoint_dir, 'pfsp_training_progress.csv')
    strategic_csv = os.path.join(checkpoint_dir, 'pfsp_strategic_metrics.csv')

    # If CSVs exist, preload them so training_data is continuous across resumes
    start_time_offset = 0.0
    if os.path.exists(progress_csv):
        with open(progress_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data['episodes'].append(int(row['episode']))
                training_data['picker_avg'].append(float(row['picker_avg']))
                training_data['called_pick_rate'].append(float(row['called_pick_rate']))
                training_data['jd_pick_rate'].append(float(row['jd_pick_rate']))
                training_data['learning_rate'].append(float(row['learning_rate']))
                training_data['time_elapsed'].append(float(row['time_elapsed']))
                training_data['team_point_diff'].append(float(row['team_point_diff']))
                # Optional historical series (may be missing in early CSVs)
                if 'alone_rate' in row and row['alone_rate'] != '':
                    training_data['alone_rate'].append(float(row['alone_rate']))
                if 'leaster_rate' in row and row['leaster_rate'] != '':
                    training_data['leaster_rate'].append(float(row['leaster_rate']))
        if training_data['time_elapsed']:
            start_time_offset = training_data['time_elapsed'][-1]

    if os.path.exists(strategic_csv):
        with open(strategic_csv, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                training_data['strategic_episodes'].append(int(row['episode']))
                training_data['pick_hand_correlation'].append(float(row['pick_hand_correlation']))
                training_data['picker_trump_rate'].append(float(row['picker_trump_rate']))
                training_data['defender_trump_rate'].append(float(row['defender_trump_rate']))
                training_data['bury_quality_rate'].append(float(row['bury_quality_rate']))

    start_time = time.time()
    game_count = 0
    # Track cumulative μ renormalisation shifts for logging
    cumulative_renorm = 0.0
    transitions_since_update = 0
    last_checkpoint_time = start_time

    bury_entropy_bump_until = 0

    # Apply initial partner/pick head temperatures (τ) controller state
    current_partner_temp = hyperparams.partner_temp_base
    current_pick_temp = hyperparams.pick_temp_base
    training_agent.set_head_temperatures(partner=current_partner_temp, pick=current_pick_temp)

    print(f"\n🎮 Beginning PFSP training... (target: {num_episodes:,} episodes)")
    print(population.get_population_summary())
    print("-" * 80)

    for episode in range(start_episode + 1, num_episodes + 1):
        # ------------------------------------------------------------------
        # Prevent numerical overflow in OpenSkill ratings by renormalising μ
        # ------------------------------------------------------------------
        MAX_ABS_MU = 350.0
        all_ratings = [training_rating] + [ag.rating for ag in population.jd_population] + [ag.rating for ag in population.called_ace_population]
        extreme_mu = max(abs(r.mu) for r in all_ratings) if all_ratings else 0
        if extreme_mu > MAX_ABS_MU:
            shift = extreme_mu - MAX_ABS_MU
            for r in all_ratings:
                r.mu -= np.sign(r.mu) * shift
            cumulative_renorm -= shift

        # Throttle log noise – print cumulative shift once every 2 000 episodes
        if episode % 2000 == 1:
            print(f"⚖️  Cumulative rating μ renorm Δ over last 2k eps: {cumulative_renorm:+.1f}  (|μ|max={extreme_mu:.1f})")
            cumulative_renorm = 0.0

        partner_mode = get_partner_selection_mode(episode)

        # Sample opponents from population based on current OpenSkill rating
        training_skill = training_rating.mu
        opponents = population.sample_opponents(
            partner_mode=partner_mode,
            n_opponents=4,
            training_agent_skill=training_skill,
            training_agent_id=training_agent_id,
            diversity_weight=0.35,
            curriculum_weight=0.4,
            exploitation_weight=0.1,
            uniform_mix=0.15,
        )

        # If no opponents available, use self-play as fallback
        if len(opponents) == 0:
            print(f"⚠️  No opponents available for {population._get_partner_mode_name(partner_mode)}, using self-play")
            opponents = []

        # Randomly select training agent position (1-5)
        training_position = random.randint(1, 5)

        # Compute per-episode shaping weights from schedules
        progress_pct = min(100.0, max(0.0, (episode / num_episodes) * 100.0))
        shaping_weights = {
            "pick": interpolated_weight(hyperparams.shaping_schedule_pick, progress_pct),
            "partner": interpolated_weight(hyperparams.shaping_schedule_partner, progress_pct),
            "bury": interpolated_weight(hyperparams.shaping_schedule_bury, progress_pct),
            "play": interpolated_weight(hyperparams.shaping_schedule_play, progress_pct),
        }

        # Play game
        game, episode_transitions, final_scores, training_data_single = play_population_game(
            training_agent=training_agent,
            opponents=opponents,
            partner_mode=partner_mode,
            training_agent_position=training_position,
            shaping_weights=shaping_weights
        )

        # Process rewards and store transitions
        # Build last_transition_per_player dictionary based on actual transitions
        last_transition_per_player = {}
        for i, transition in enumerate(episode_transitions):
            player_pos = transition['player'].position
            last_transition_per_player[player_pos] = i

        for reward_data in process_episode_rewards(
            episode_transitions,
            final_scores,
            last_transition_per_player,
            game.is_leaster
        ):
            transition = reward_data['transition']
            training_agent.store_transition(
                transition['state'],
                transition['action'],
                reward_data['reward'],
                transition['value'],
                transition['log_prob'],
                reward_data['done'],
                transition['valid_actions'],
                player_id=transition['player'].position,
            )
            transitions_since_update += 1

        # Update statistics
        picker_score = training_data_single['score'] if training_data_single['was_picker'] else 0
        if training_data_single['was_picker']:
            picker_scores.append(picker_score)

        # Track how often the training agent is the picker (unconditional)
        picker_window.append(1 if training_data_single['was_picker'] else 0)

        # ---------- OpenSkill rating update ----------
        position_to_agent = {}
        opponent_positions = [pos for pos in range(1, 6) if pos != training_position]
        for idx, opp in enumerate(opponents[:len(opponent_positions)]):
            position_to_agent[opponent_positions[idx]] = opp

        teams = []
        positions_order = list(range(1, 6))
        agents_in_order = []
        for pos in positions_order:
            if pos == training_position:
                teams.append([training_rating])
                agents_in_order.append(None)
            else:
                opp_agent = position_to_agent.get(pos)
                if opp_agent:
                    teams.append([opp_agent.rating])
                    agents_in_order.append(opp_agent)
                else:
                    # Self-play filler seat – share the training rating object so rating update is well-posed
                    teams.append([training_rating])
                    agents_in_order.append(None)

        scores_for_rank = final_scores
        score_rank_pairs = sorted(enumerate(scores_for_rank), key=lambda x: x[1], reverse=True)
        ranks = [0]*len(scores_for_rank)
        last_score = None
        current_rank = 0
        for i, (idx, sc) in enumerate(score_rank_pairs):
            if last_score is None or sc != last_score:
                current_rank = i
            ranks[idx] = current_rank
            last_score = sc

        new_teams = rating_model.rate(teams, ranks=ranks)

        for idx, pos in enumerate(positions_order):
            if pos == training_position:
                training_rating = new_teams[idx][0]
            else:
                opp_agent = position_to_agent.get(pos)
                if opp_agent:
                    opp_agent.update_rating(new_teams[idx][0])
                    was_picker_role = (game.picker == pos)
                    opp_agent.add_game_result(final_scores[pos-1], was_picker_role)
                    if final_scores[pos-1] > training_data_single['score']:
                        opp_agent.add_victory_against(training_agent_id)

        # Track team point differences
        if game.picker and not game.is_leaster:
            picker_team_points = game.get_final_picker_points()
            defender_team_points = game.get_final_defender_points()
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0
        team_point_differences.append(team_point_diff)

        # Track other statistics (similar to train_ppo.py)
        is_leaster_ep = 1 if game.is_leaster else 0
        leaster_window.append(is_leaster_ep)

        is_called_ace_ep = 1 if partner_mode == PARTNER_BY_CALLED_ACE else 0
        called_ace_window.append(is_called_ace_ep)

        if is_called_ace_ep and not is_leaster_ep:
            called_under_window.append(1 if game.is_called_under else 0)
            called_10_window.append(1 if (game.called_card and game.called_card.startswith("10")) else 0)
        elif is_called_ace_ep:
            called_under_window.append(0)
            called_10_window.append(0)

        # Track ALONE calls only for games with a picker (exclude leaster)
        if not game.is_leaster:
            alone_call_window.append(1 if game.alone_called else 0)

        # Count pick/pass decisions from transitions
        episode_picks = sum(1 for t in episode_transitions if t['action'] == ACTION_IDS["PICK"])
        episode_passes = sum(1 for t in episode_transitions if t['action'] == ACTION_IDS["PASS"])

        pick_decisions[partner_mode].append(episode_picks)
        pass_decisions[partner_mode].append(episode_passes)

        game_count += 1

        # Update model periodically
        if transitions_since_update >= update_interval:
            print(f"🔄 Updating model after {game_count} games... (Episode {episode:,})")

            # Entropy decay
            entropy_play_start, entropy_play_end = hyperparams.entropy_play_start, hyperparams.entropy_play_end
            entropy_pick_start, entropy_pick_end = hyperparams.entropy_pick_start, hyperparams.entropy_pick_end
            entropy_partner_start, entropy_partner_end = hyperparams.entropy_partner_start, hyperparams.entropy_partner_end
            entropy_bury_start, entropy_bury_end = hyperparams.entropy_bury_start, hyperparams.entropy_bury_end
            decay_fraction = min(episode / num_episodes, 1.0)
            training_agent.entropy_coeff_play = entropy_play_start + (entropy_play_end - entropy_play_start) * decay_fraction
            training_agent.entropy_coeff_pick = entropy_pick_start + (entropy_pick_end - entropy_pick_start) * decay_fraction
            training_agent.entropy_coeff_partner = entropy_partner_start + (entropy_partner_end - entropy_partner_start) * decay_fraction
            training_agent.entropy_coeff_bury = entropy_bury_start + (entropy_bury_end - entropy_bury_start) * decay_fraction


            # Apply temporary bump to bury entropy
            if episode <= bury_entropy_bump_until:
                training_agent.entropy_coeff_bury += hyperparams.bury_entropy_bump

            # Update
            update_stats = training_agent.update(epochs=4, batch_size=256)

            if update_stats:
                adv_stats = update_stats['advantage_stats']
                val_stats = update_stats['value_target_stats']
                num_transitions = update_stats['num_transitions']
                approx_kl = update_stats.get('approx_kl', None)
                early_stop = update_stats.get('early_stop', False)
                head_entropy = update_stats.get('head_entropy', {})
                pick_pass_adv = update_stats.get('pick_pass_adv', {})

                print(f"   Transitions: {num_transitions}")
                print(f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
                print(f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")
                if approx_kl is not None:
                    print(f"   PPO KL: {approx_kl:.4f}  Early stop: {early_stop}")
                # Instrumentation logs: head entropy and PICK/PASS advantages
                if head_entropy:
                    print(
                        f"   Entropy - pick: {head_entropy.get('pick', 0.0):.3f}, "
                        f"partner: {head_entropy.get('partner', 0.0):.3f}, "
                        f"bury: {head_entropy.get('bury', 0.0):.3f}, "
                        f"play: {head_entropy.get('play', 0.0):.3f}"
                    )
                if pick_pass_adv:
                    print(
                        f"   Adv(PICK): {pick_pass_adv.get('pick_mean', 0.0):+.3f} (n={pick_pass_adv.get('pick_count', 0)}), "
                        f"Adv(PASS): {pick_pass_adv.get('pass_mean', 0.0):+.3f} (n={pick_pass_adv.get('pass_count', 0)})"
                    )
                if 'timing' in update_stats:
                    t = update_stats['timing']
                    print(
                        f"   Timing - build: {t['build_s']:.3f}s, forward: {t['forward_s']:.3f}s, "
                        f"backward: {t['backward_s']:.3f}s, step: {t['step_s']:.3f}s, total: {t['total_update_s']:.3f}s, "
                        f"opt_steps: {t['optimizer_steps']}"
                    )

            game_count = 0
            transitions_since_update = 0

        # Add agent to population periodically
        if episode % population_add_interval == 0:
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                agent_id = population.add_agent(
                    agent=copy.deepcopy(training_agent),
                    partner_mode=mode,
                    training_episodes=episode,
                    parent_id=None,
                    activation=activation
                )
                print(f"👥 Added training agent snapshot to {population._get_partner_mode_name(mode)} population (ID: {agent_id})")

            # Log updated diversity stats after population change
            jd_div = population.get_diversity_stats(PARTNER_BY_JD)
            ca_div = population.get_diversity_stats(PARTNER_BY_CALLED_ACE)
            print("   Diversity (JD): avg=%.3f, spread=%.3f, clusters=%d, alone_rate_range=(%.2f, %.2f), pick_cv={weak: %.2f, med: %.2f, strong: %.2f}, coverage={early: %.2f, void: %.2f}" % (
                jd_div['avg_pairwise_diversity'], jd_div['diversity_spread'], jd_div['strategic_clusters'],
                jd_div['alone_rate_range'][0], jd_div['alone_rate_range'][1],
                jd_div['pick_rate_diversity']['weak'], jd_div['pick_rate_diversity']['medium'], jd_div['pick_rate_diversity']['strong'],
                jd_div['coverage']['early_leads'], jd_div['coverage']['void_events']))
            print("   Diversity (CA): avg=%.3f, spread=%.3f, clusters=%d, alone_rate_range=(%.2f, %.2f), pick_cv={weak: %.2f, med: %.2f, strong: %.2f}, coverage={early: %.2f, void: %.2f}" % (
                ca_div['avg_pairwise_diversity'], ca_div['diversity_spread'], ca_div['strategic_clusters'],
                ca_div['alone_rate_range'][0], ca_div['alone_rate_range'][1],
                ca_div['pick_rate_diversity']['weak'], ca_div['pick_rate_diversity']['medium'], ca_div['pick_rate_diversity']['strong'],
                ca_div['coverage']['early_leads'], ca_div['coverage']['void_events']))

        # Cross-evaluation periodically
        if episode % cross_eval_interval == 0:
            print(f"🏆 Running cross-evaluation tournaments... (Episode {episode:,})")
            for mode in [PARTNER_BY_JD, PARTNER_BY_CALLED_ACE]:
                eval_stats = population.run_cross_evaluation(mode, num_games=100, max_agents=25)
                if eval_stats:
                    print(f"   {population._get_partner_mode_name(mode)}: {eval_stats['games_played']} games, "
                          f"avg skill: {eval_stats['avg_skill_after']:.1f}")

        # Strategic evaluation
        if episode % strategic_eval_interval == 0:
            print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
            strategic_metrics = analyze_strategic_decisions(training_agent, num_samples=200)

            training_data['strategic_episodes'].append(episode)
            training_data['pick_hand_correlation'].append(strategic_metrics['pick_hand_correlation'])
            training_data['picker_trump_rate'].append(strategic_metrics['picker_trump_rate'])
            training_data['defender_trump_rate'].append(strategic_metrics['defender_trump_rate'])
            training_data['bury_quality_rate'].append(strategic_metrics['bury_quality_rate'])

            # Append strategic metrics row to CSV (episode-keyed)
            write_header = (not os.path.exists(strategic_csv)) or (os.path.getsize(strategic_csv) == 0)
            with open(strategic_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode','pick_hand_correlation','picker_trump_rate','defender_trump_rate','bury_quality_rate'])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    'episode': episode,
                    'pick_hand_correlation': strategic_metrics['pick_hand_correlation'],
                    'picker_trump_rate': strategic_metrics['picker_trump_rate'],
                    'defender_trump_rate': strategic_metrics['defender_trump_rate'],
                    'bury_quality_rate': strategic_metrics['bury_quality_rate'],
                })

            print(f"   Pick-Hand Correlation: {strategic_metrics['pick_hand_correlation']:.3f}")
            print(f"   Picker Trump Rate: {strategic_metrics['picker_trump_rate']:.1f}%")
            print(f"   Defender Trump Rate: {strategic_metrics['defender_trump_rate']:.1f}%")
            print(f"   Bury Quality Rate: {strategic_metrics['bury_quality_rate']:.1f}%")

            # --- Check bury quality rate and schedule bury-head entropy bump if too low ---
            current_bury_quality_rate = strategic_metrics['bury_quality_rate']
            if (
                current_bury_quality_rate < hyperparams.low_bury_quality_threshold and episode > bury_entropy_bump_until
            ):
                bury_entropy_bump_until = episode + hyperparams.bury_entropy_bump_duration
                print(f"   ⚠️  Low bury-quality rate detected ({current_bury_quality_rate:.1f}%). Increasing bury entropy by {hyperparams.bury_entropy_bump:.3f} for the next {hyperparams.bury_entropy_bump_duration:,} episodes.")
            if (
                current_bury_quality_rate >= hyperparams.low_bury_quality_threshold and
                episode > bury_entropy_bump_until and
                bury_entropy_bump_until != 0
            ):
                # Reset any expired bump marker to reduce log noise later
                bury_entropy_bump_until = 0

        # Progress reporting
        if episode % 1000 == 0:
            current_avg_picker_score = np.mean(picker_scores) if picker_scores else 0

            # Calculate pick rates
            total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
            total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
            total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
            total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
            current_called_pick_rate = (100 * total_called_picks / (total_called_picks + total_called_passes)) if (total_called_picks + total_called_passes) > 0 else 0
            current_jd_pick_rate = (100 * total_jd_picks / (total_jd_picks + total_jd_passes)) if (total_jd_picks + total_jd_passes) > 0 else 0
            current_team_diff = np.mean(team_point_differences) if team_point_differences else 0

            # Rolling window rates
            current_leaster_rate = (sum(leaster_window) / len(leaster_window)) * 100 if leaster_window else 0
            current_alone_rate = (sum(alone_call_window) / len(alone_call_window)) * 100 if alone_call_window else 0
            ca_denominator = sum(called_ace_window) or 1
            current_called_under_rate = (sum(called_under_window) / ca_denominator) * 100
            current_called_10s_rate = (sum(called_10_window) / ca_denominator) * 100
            current_picker_frequency = (sum(picker_window) / len(picker_window)) * 100 if picker_window else 0
            elapsed = start_time_offset + (time.time() - start_time)

            # Collect data for plotting
            training_data['episodes'].append(episode)
            training_data['picker_avg'].append(current_avg_picker_score)
            training_data['called_pick_rate'].append(current_called_pick_rate)
            training_data['jd_pick_rate'].append(current_jd_pick_rate)
            training_data['learning_rate'].append(training_agent.actor_optimizer.param_groups[0]['lr'])
            training_data['time_elapsed'].append(elapsed)
            training_data['team_point_diff'].append(current_team_diff)
            training_data['alone_rate'].append(current_alone_rate)
            training_data['leaster_rate'].append(current_leaster_rate)

            # Append progress row to CSV (create with header if new)
            write_header = (not os.path.exists(progress_csv)) or (os.path.getsize(progress_csv) == 0)
            with open(progress_csv, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['episode','picker_avg','called_pick_rate','jd_pick_rate','learning_rate','time_elapsed','team_point_diff','alone_rate','leaster_rate'])
                if write_header:
                    writer.writeheader()
                writer.writerow({
                    'episode': episode,
                    'picker_avg': current_avg_picker_score,
                    'called_pick_rate': current_called_pick_rate,
                    'jd_pick_rate': current_jd_pick_rate,
                    'learning_rate': training_agent.actor_optimizer.param_groups[0]['lr'],
                    'time_elapsed': elapsed,
                    'team_point_diff': current_team_diff,
                    'alone_rate': current_alone_rate,
                    'leaster_rate': current_leaster_rate,
                })

            # Population statistics
            jd_stats = population.get_population_stats(PARTNER_BY_JD)
            ca_stats = population.get_population_stats(PARTNER_BY_CALLED_ACE)
            training_data['population_stats'].append({
                'episode': episode,
                'jd_size': jd_stats['size'],
                'jd_avg_skill': jd_stats['avg_skill'],
                'ca_size': ca_stats['size'],
                'ca_avg_skill': ca_stats['avg_skill']
            })

            games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

            print(f"📊 Episode {episode:,}/{num_episodes:,} ({episode/num_episodes*100:.1f}%)")
            print("   " + "-" * 50)
            print(f"   Picker avg: {current_avg_picker_score:+.3f}")
            print(f"   Team point diff: {current_team_diff:+.1f}")
            print(f"   Called Ace Pick rate: {current_called_pick_rate:.1f}%")
            print(f"   JD Pick rate: {current_jd_pick_rate:.1f}%")
            print(f"   Picker Frequency: {current_picker_frequency:.2f}%")
            print("   " + "-" * 25)
            print(f"   Leaster Rate: {current_leaster_rate:.2f}%")
            print(f"   Alone Call Rate: {current_alone_rate:.2f}%")
            print(f"   Called Under Rate: {current_called_under_rate:.2f}%")
            print(f"   Called 10s Rate: {current_called_10s_rate:.2f}%")
            print("   " + "-" * 25)
            print(f"   Population JD: {jd_stats['size']} agents (avg skill: {jd_stats['avg_skill']:.1f})")
            print(f"   Population CA: {ca_stats['size']} agents (avg skill: {ca_stats['avg_skill']:.1f})")
            print("   " + "-" * 50)
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed/60:.1f} min")
            print("   " + "-" * 50)

            # --- Adaptive partner-head temperature control (τ) ---
            # If ALONE rate is too high, increase τ to soften the partner head;
            # cool it back toward base when rate drops sufficiently below threshold.
            desired_temp = current_partner_temp
            if current_alone_rate > hyperparams.high_alone_rate_threshold:
                desired_temp = min(current_partner_temp + hyperparams.partner_temp_step, hyperparams.partner_temp_max)
            elif current_alone_rate < (hyperparams.high_alone_rate_threshold - hyperparams.alone_rate_hysteresis_down):
                desired_temp = max(current_partner_temp - hyperparams.partner_temp_step, hyperparams.partner_temp_base)

            if abs(desired_temp - current_partner_temp) > 1e-6:
                current_partner_temp = desired_temp
                training_agent.set_head_temperatures(partner=current_partner_temp)
                print(f"   ⚠️  Partner head temperature τ adjusted to: {current_partner_temp:.2f}")

            # --- Adaptive pick-head temperature control (τ) ---
            # If pick rate is too high, increase τ_pick to soften the pick head;
            # cool back toward base when rate drops below band.
            overall_picks = total_called_picks + total_jd_picks
            overall_decisions = overall_picks + total_called_passes + total_jd_passes
            overall_pick_rate = (100 * overall_picks / overall_decisions) if overall_decisions > 0 else 0.0

            desired_pick_temp = current_pick_temp
            # Heat when rate too high or too low; cool when inside band with hysteresis
            if (
                overall_pick_rate > hyperparams.high_pick_rate_threshold or
                overall_pick_rate < hyperparams.low_pick_rate_threshold
            ):
                desired_pick_temp = min(current_pick_temp + hyperparams.pick_temp_step, hyperparams.pick_temp_max)
            elif (
                current_pick_temp > hyperparams.pick_temp_base and
                overall_pick_rate >= (hyperparams.low_pick_rate_threshold + hyperparams.pick_rate_hysteresis_down) and
                overall_pick_rate <= (hyperparams.high_pick_rate_threshold - hyperparams.pick_rate_hysteresis_down)
            ):
                desired_pick_temp = max(current_pick_temp - hyperparams.pick_temp_step, hyperparams.pick_temp_base)

            if abs(desired_pick_temp - current_pick_temp) > 1e-6:
                current_pick_temp = desired_pick_temp
                training_agent.set_head_temperatures(pick=current_pick_temp)
                print(f"   ⚠️  Pick head temperature τ adjusted to: {current_pick_temp:.2f}")


        # Save checkpoints
        if episode % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/pfsp_{activation}_checkpoint_{episode}.pt'
            training_agent.save(checkpoint_path)

            # Save population state
            population.save_population_state()

            # Save training plot
            if len(training_data['episodes']) > 10:
                plot_path = f'{checkpoint_dir}/pfsp_training_progress_{episode}.png'
                save_training_plot(training_data, plot_path)  # Reuse plotting function

            checkpoint_time = time.time()
            time_since_last = checkpoint_time - last_checkpoint_time
            last_checkpoint_time = checkpoint_time

            print(f"💾 Checkpoint saved at episode {episode:,}")
            print(f"   Time for last {save_interval:,} episodes: {time_since_last/60:.1f} min")
            remaining_episodes = num_episodes - episode
            if remaining_episodes > 0:
                estimated_time = remaining_episodes * (time_since_last / save_interval) / 60
                print(f"   Estimated time remaining: {estimated_time:.1f} min")

    # Final update and save
    if training_agent.events:
        print("🔄 Final model update...")
        final_update_stats = training_agent.update()

        if final_update_stats:
            adv_stats = final_update_stats['advantage_stats']
            val_stats = final_update_stats['value_target_stats']
            num_transitions = final_update_stats['num_transitions']

            print(f"   Final Transitions: {num_transitions}")
            print(f"   Final Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
            print(f"   Final Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")

    training_agent.save(f'final_pfsp_{activation}_ppo.pt')
    population.save_population_state()

    # Save final training plot
    if len(training_data['episodes']) > 0:
        save_training_plot(training_data, f'final_pfsp_{activation}_training.png')

    total_time = time.time() - start_time
    print("\n🎉 PFSP Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}")
    print(f"   Final team point difference: {np.mean(team_point_differences) if team_point_differences else 0:.1f}")

    # Final population summary
    print("\n" + population.get_population_summary())


def main():
    parser = ArgumentParser(description="PFSP population-based training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=500000,
                       help="Number of training episodes (default: 500,000)")
    parser.add_argument("--update-interval", type=int, default=2048,
                       help="Number of transitions between model updates")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Number of episodes between checkpoints")
    parser.add_argument("--strategic-eval-interval", type=int, default=10000,
                       help="Number of episodes between strategic evaluations")
    parser.add_argument("--population-add-interval", type=int, default=5000,
                       help="Number of episodes between adding agents to population")
    parser.add_argument("--cross-eval-interval", type=int, default=20000,
                       help="Number of episodes between cross-evaluation tournaments")
    parser.add_argument("--resume", type=str, default=None,
                       help="Model file to resume training from")
    parser.add_argument("--activation", type=str, default='swish', choices=['relu', 'swish'],
                       help="Activation function to use (default: swish)")
    parser.add_argument("--initial-checkpoints", nargs='+', default=None,
                       help="Checkpoint patterns to initialize population from")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend('Agg')

    train_pfsp(
        num_episodes=args.episodes,
        update_interval=args.update_interval,
        save_interval=args.save_interval,
        strategic_eval_interval=args.strategic_eval_interval,
        population_add_interval=args.population_add_interval,
        cross_eval_interval=args.cross_eval_interval,
        resume_model=args.resume,
        activation=args.activation,
        initial_checkpoints=args.initial_checkpoints
    )


if __name__ == "__main__":
    main()