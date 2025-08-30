#!/usr/bin/env python3
"""
Extended long-term PPO training for Sheepshead.
"""

import torch
import numpy as np
import random
import time
import os
from collections import deque
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from ppo import PPOAgent
from sheepshead import Game, ACTIONS, STATE_SIZE, ACTION_LOOKUP, ACTION_IDS, TRUMP, PARTNER_BY_CALLED_ACE, PARTNER_BY_JD
from training_utils import (
    process_episode_rewards,
    get_partner_selection_mode,
    save_training_plot,
    update_intermediate_rewards_for_action,
    handle_trick_completion,
)


def analyze_strategic_decisions(agent, num_samples=100):
    """Analyze strategic decision quality instead of random opponent evaluation."""

    # Trump leading analysis
    trump_leads = {'picker_team': 0, 'picker_total': 0, 'defender_team': 0, 'defender_total': 0}

    # Bury quality analysis
    bury_quality = {'good_burys': 0, 'bad_burys': 0, 'total_burys': 0}

    # Pick decision correlation with hand strength
    pick_decisions = []
    hand_strengths = []

    for episode in range(num_samples):
        game = Game(partner_selection_mode=get_partner_selection_mode(episode))

        # Analyze pick decisions
        initial_player = game.players[0]
        hand_strength = sum(3 if c[0] == 'Q' else 2 if c[0] == 'J' else 1 if c in TRUMP else 0 for c in initial_player.hand)

        state = torch.FloatTensor(initial_player.get_state_vector()).unsqueeze(0)
        # Build action mask for the current decision so probabilities reflect only valid actions
        initial_actions = initial_player.get_valid_action_ids()
        action_mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
        for valid_action in initial_actions:
            action_mask[valid_action - 1] = True
        with torch.no_grad():
            action_probs = agent.actor(state, action_mask.unsqueeze(0))

        pick_prob = action_probs[0, ACTION_IDS["PICK"] - 1].item()
        pick_decisions.append(pick_prob)
        hand_strengths.append(hand_strength)

        # Play full game to analyze trump leading and bury decisions
        while not game.is_done():
            for player in game.players:
                actions = player.get_valid_action_ids()

                if actions:
                    state = torch.FloatTensor(player.get_state_vector()).unsqueeze(0)
                    action_mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
                    for valid_action in actions:
                        action_mask[valid_action - 1] = True

                    with torch.no_grad():
                        action_probs = agent.actor(state, action_mask.unsqueeze(0))

                    action = torch.multinomial(action_probs, 1).item() + 1
                    action_name = ACTION_LOOKUP[action]

                    # Analyze trump leading
                    if "PLAY" in action_name and game.play_started and game.cards_played == 0:
                        card = action_name.split()[-1]
                        is_trump_lead = card in TRUMP
                        is_picker_team = (player.is_picker or player.is_partner)

                        if is_picker_team:
                            trump_leads['picker_total'] += 1
                            if is_trump_lead:
                                trump_leads['picker_team'] += 1
                        else:
                            trump_leads['defender_total'] += 1
                            if is_trump_lead:
                                trump_leads['defender_team'] += 1

                    # Analyze bury decisions
                    if "BURY" in action_name:
                        card = action_name.split()[-1]
                        bury_quality['total_burys'] += 1

                        # Good bury: fail
                        if card not in TRUMP:
                            bury_quality['good_burys'] += 1
                        else:
                            bury_quality['bad_burys'] += 1

                    player.act(action)

    # Calculate metrics
    pick_hand_correlation = np.corrcoef(hand_strengths, pick_decisions)[0, 1] if len(hand_strengths) > 1 else 0

    picker_trump_rate = trump_leads['picker_team'] / max(trump_leads['picker_total'], 1) * 100
    defender_trump_rate = trump_leads['defender_team'] / max(trump_leads['defender_total'], 1) * 100

    bury_quality_rate = bury_quality['good_burys'] / max(bury_quality['total_burys'], 1) * 100

    return {
        'pick_hand_correlation': pick_hand_correlation,
        'picker_trump_rate': picker_trump_rate,
        'defender_trump_rate': defender_trump_rate,
        'bury_quality_rate': bury_quality_rate
    }

def train_ppo(num_episodes=300000, update_interval=2048, save_interval=5000,
                            strategic_eval_interval=10000, resume_model=None, activation='swish'):
    """
    PPO training with strategic evaluation metrics.
    """
    print("🚀 Starting PPO training...")
    print("="*60)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Activation function: {activation.upper()}")
    print("="*60)

    # Create agent with optimized hyperparameters
    agent = PPOAgent(STATE_SIZE, len(ACTIONS),
                    lr_actor=5.0e-5,
                    lr_critic=1.5e-4,
                    activation=activation)

    # Resume from specified model or try to load best existing
    start_episode = 0
    if resume_model:
        try:
            agent.load(resume_model)
            print(f"✅ Loaded {resume_model} for continuation")
            # Try to extract episode number from filename
            if 'checkpoint_' in resume_model:
                start_episode = int(resume_model.split('_')[-1].split('.')[0])
                print(f"📍 Resuming from episode {start_episode:,}")
        except Exception as e:
            print(f"❌ Could not load {resume_model}: {e}")
    else:
        print("🆕 Starting fresh training")

    picker_scores = deque(maxlen=3000)
    pick_decisions = [deque(maxlen=3000), deque(maxlen=3000)]
    pass_decisions = [deque(maxlen=3000), deque(maxlen=3000)]

    leaster_window = deque(maxlen=3000)          # 1 ⇒ leaster, 0 ⇒ regular game
    alone_call_window = deque(maxlen=3000)       # 1 ⇒ ALONE called (non-leaster games)
    called_ace_window = deque(maxlen=3000)       # 1 ⇒ partner mode = Called-Ace, else 0
    called_under_window = deque(maxlen=3000)     # 1 ⇒ called-under occurred that game
    called_10_window = deque(maxlen=3000)        # 1 ⇒ called-10s occurred that game
    team_point_differences = deque(maxlen=3000)
    best_team_difference = float('inf')  # Lower is better (smaller point difference)
    current_avg_picker_score = float('-inf')

    training_data = {
        'episodes': [],
        'recent_avg': [],
        'overall_avg': [],
        'picker_avg': [],
        'called_pick_rate': [],
        'jd_pick_rate': [],
        'alone_rate': [],
        'leaster_rate': [],
        'learning_rate': [],
        'time_elapsed': [],
        'pick_hand_correlation': [],
        'picker_trump_rate': [],
        'defender_trump_rate': [],
        'bury_quality_rate': [],
        'team_point_diff': [],
        'strategic_episodes': []
    }

    # Running picker baseline for reward shaping

    # Create checkpoint directory with activation function suffix
    checkpoint_dir = f'checkpoints_{activation}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    game_count = 0
    last_checkpoint_time = start_time

    print(f"\n🎮 Beginning training... (target: {num_episodes:,} episodes)")
    print("-" * 60)

    transitions_since_update = 0
    for episode in range(start_episode + 1, num_episodes + 1):
        partner_mode = get_partner_selection_mode(episode)
        game = Game(partner_selection_mode=partner_mode)
        # Reset recurrent hidden states in the actor at the start of each game
        agent.reset_recurrent_state()
        episode_scores = []
        episode_picks = 0
        episode_passes = 0

        # Store all transitions for this episode independently for each player
        episode_transitions = {player.position: [] for player in game.players}

        # Track PLAY transitions for the current trick
        current_trick_transitions = []

        # Play full game with self-play
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    state = player.get_state_vector()
                    action, log_prob, value = agent.act(state, valid_actions, player.position)

                    transition = {
                        'player': player,
                        'state': state,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy(),
                        'intermediate_reward': 0.0,
                    }

                    action_name = ACTIONS[action - 1]

                    # Track pick/pass decisions for statistics
                    if action_name == "PICK":
                        episode_picks += 1
                    elif action_name == "PASS":
                        episode_passes += 1

                    episode_transitions[player.position].append(transition)

                    # Apply shared intermediate rewards and track trick transitions
                    update_intermediate_rewards_for_action(
                        game,
                        player,
                        action,
                        transition,
                        current_trick_transitions,
                    )

                    player.act(action)

                    # Trick resolution and observation frames
                    trick_completed = handle_trick_completion(
                        game, current_trick_transitions
                    )
                    if trick_completed:
                        # ------------------------------------------------
                        # Add post-trick observation frames for all seats
                        # (stored for training-time recurrent unroll)
                        # Also update the online recurrent hidden state
                        # ------------------------------------------------
                        for seat in game.players:
                            # Update online recurrent state
                            agent.observe(seat.get_last_trick_state_vector(), player_id=seat.position)
                            # Store for training-time unroll
                            episode_transitions[seat.position].append({
                                'kind': 'obs',
                                'player': seat,
                                'state': seat.get_state_vector(),
                            })

                    valid_actions = player.get_valid_action_ids()

        final_scores = [player.get_score() for player in game.players]
        episode_scores = final_scores[:]

        # ---------------------------------------------
        # Flatten per-player trajectories; compute rewards
        # ---------------------------------------------
        flat_all = []
        for pos in episode_transitions:
            flat_all.extend(episode_transitions[pos])

        # Build action-only list preserving per-player order
        flat_actions = [t for t in flat_all if t.get('kind', 'action') == 'action']

        # Compute last action index per player within action-only list
        last_transition_per_player = {}
        for idx, t in enumerate(flat_actions):
            ppos = t['player'].position
            last_transition_per_player[ppos] = idx

        # Compute rewards/done for action transitions
        reward_map = {}
        for reward_data in process_episode_rewards(
            flat_actions,
            final_scores,
            last_transition_per_player,
            game.is_leaster
        ):
            tr = reward_data['transition']
            reward_map[id(tr)] = (reward_data['reward'], reward_data['done'])

        # Store events in chronological per-player order (obs and actions interleaved)
        for ev in flat_all:
            if ev.get('kind') == 'obs':
                agent.store_observation(ev['state'], player_id=ev['player'].position)
            else:
                reward, done_flag = reward_map[id(ev)]
                agent.store_transition(
                    ev['state'],
                    ev['action'],
                    reward,
                    ev['value'],
                    ev['log_prob'],
                    done_flag,
                    ev['valid_actions'],
                    player_id=ev['player'].position,
                )
                transitions_since_update += 1

        # Track statistics
        picker_score = episode_scores[game.picker - 1] if game.picker else 0


        # Calculate team point difference (picker team points - defender team points)
        if game.picker and not game.is_leaster:
            picker_team_points = game.get_final_picker_points()
            defender_team_points = game.get_final_defender_points()
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0  # No team difference in leaster games

        # --------------------------------------------------
        # Append episode outcome to rolling windows
        # --------------------------------------------------
        is_leaster_ep = 1 if game.is_leaster else 0
        leaster_window.append(is_leaster_ep)

        is_called_ace_ep = 1 if partner_mode == PARTNER_BY_CALLED_ACE else 0
        called_ace_window.append(is_called_ace_ep)

        # Only meaningful for Called-Ace, non-leaster games
        if is_called_ace_ep and not is_leaster_ep:
            called_under_window.append(1 if game.is_called_under else 0)
            called_10_window.append(1 if (game.called_card and game.called_card.startswith("10")) else 0)
        elif is_called_ace_ep:
            called_under_window.append(0)
            called_10_window.append(0)

        picker_scores.append(picker_score)
        # ALONE tracking for games with a picker (exclude leaster)
        if not game.is_leaster:
            alone_call_window.append(1 if game.alone_called else 0)
        pick_decisions[get_partner_selection_mode(episode)].append(episode_picks)
        pass_decisions[get_partner_selection_mode(episode)].append(episode_passes)
        team_point_differences.append(team_point_diff)
        game_count += 1

        # Update model periodically by transition count (action transitions only)
        if transitions_since_update >= update_interval:
            print(f"🔄 Updating model after {transitions_since_update} transitions... (Episode {episode:,})")

            # Separate entropy decay schedules
            entropy_play_start, entropy_play_end = 0.05, 0.03
            entropy_pick_start, entropy_pick_end = 0.05, 0.03
            entropy_partner_start, entropy_partner_end = 0.05, 0.03
            entropy_bury_start, entropy_bury_end = 0.04, 0.02
            decay_fraction = min(episode / num_episodes, 1.0)
            agent.entropy_coeff_play = entropy_play_start + (entropy_play_end - entropy_play_start) * decay_fraction
            agent.entropy_coeff_pick = entropy_pick_start + (entropy_pick_end - entropy_pick_start) * decay_fraction
            agent.entropy_coeff_partner = entropy_partner_start + (entropy_partner_end - entropy_partner_start) * decay_fraction
            agent.entropy_coeff_bury = entropy_bury_start + (entropy_bury_end - entropy_bury_start) * decay_fraction

            update_stats = agent.update(epochs=4, batch_size=256)

            # Log advantage and value target statistics
            if update_stats:
                adv_stats = update_stats['advantage_stats']
                val_stats = update_stats['value_target_stats']
                num_transitions = update_stats['num_transitions']
                approx_kl = update_stats.get('approx_kl', None)
                early_stop = update_stats.get('early_stop', False)

                print(f"   Transitions: {num_transitions}")
                print(f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
                print(f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")
                if approx_kl is not None:
                    print(f"   PPO KL: {approx_kl:.4f}  Early stop: {early_stop}")
                if 'timing' in update_stats:
                    t = update_stats['timing']
                    print(
                        f"   Timing - build: {t['build_s']:.3f}s, forward: {t['forward_s']:.3f}s, "
                        f"backward: {t['backward_s']:.3f}s, step: {t['step_s']:.3f}s, total: {t['total_update_s']:.3f}s, "
                        f"opt_steps: {t['optimizer_steps']}"
                    )
                head_entropy = update_stats.get('head_entropy')
                if head_entropy:
                    print(
                        f"   Entropy - pick: {head_entropy.get('pick', 0.0):.3f}, "
                        f"partner: {head_entropy.get('partner', 0.0):.3f}, "
                        f"bury: {head_entropy.get('bury', 0.0):.3f}, "
                        f"play: {head_entropy.get('play', 0.0):.3f}"
                    )

            game_count = 0
            transitions_since_update = 0

                # Strategic evaluation at intervals
        if episode % strategic_eval_interval == 0:
            print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
            strategic_metrics = analyze_strategic_decisions(agent, num_samples=200)

            # Store strategic metrics
            training_data['strategic_episodes'].append(episode)
            training_data['pick_hand_correlation'].append(strategic_metrics['pick_hand_correlation'])
            training_data['picker_trump_rate'].append(strategic_metrics['picker_trump_rate'])
            training_data['defender_trump_rate'].append(strategic_metrics['defender_trump_rate'])
            training_data['bury_quality_rate'].append(strategic_metrics['bury_quality_rate'])

            print(f"   Pick-Hand Correlation: {strategic_metrics['pick_hand_correlation']:.3f}")
            print(f"   Picker Trump Rate: {strategic_metrics['picker_trump_rate']:.1f}%")
            print(f"   Defender Trump Rate: {strategic_metrics['defender_trump_rate']:.1f}%")
            print(f"   Bury Quality Rate: {strategic_metrics['bury_quality_rate']:.1f}%")

        # Progress reporting and data collection
        if episode % 1000 == 0:
            current_avg_picker_score = np.mean(picker_scores) if picker_scores else 0
            # Compute pick-rate over all individual decisions (weighting games by the number of decisions they contributed)
            total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
            total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
            total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
            total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
            current_called_pick_rate = (100 * total_called_picks / (total_called_picks + total_called_passes)) if (total_called_picks + total_called_passes) > 0 else 0
            current_jd_pick_rate = (100 * total_jd_picks / (total_jd_picks + total_jd_passes)) if (total_jd_picks + total_jd_passes) > 0 else 0
            current_team_diff = np.mean(team_point_differences) if team_point_differences else 0
            # --- Rolling-window rates ---
            current_leaster_rate = (sum(leaster_window) / len(leaster_window)) * 100 if leaster_window else 0
            current_alone_rate = (sum(alone_call_window) / len(alone_call_window)) * 100 if alone_call_window else 0

            ca_denominator = sum(called_ace_window) or 1  # avoid divide-by-zero
            current_called_under_rate = (sum(called_under_window) / ca_denominator) * 100
            current_called_10s_rate = (sum(called_10_window) / ca_denominator) * 100
            elapsed = time.time() - start_time

            # Collect data for plotting
            training_data['episodes'].append(episode)
            training_data['picker_avg'].append(current_avg_picker_score)
            training_data['called_pick_rate'].append(current_called_pick_rate)
            training_data['jd_pick_rate'].append(current_jd_pick_rate)
            training_data['learning_rate'].append(agent.actor_optimizer.param_groups[0]['lr'])
            training_data['time_elapsed'].append(elapsed)
            training_data['team_point_diff'].append(current_team_diff)
            training_data['alone_rate'].append(current_alone_rate)
            training_data['leaster_rate'].append(current_leaster_rate)

            # Strategic metrics are collected separately during strategic evaluation intervals
            # Don't try to collect them here as they're not always available

            # Calculate training speed
            games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

            print(f"📊 Episode {episode:,}/{num_episodes:,} ({episode/num_episodes*100:.1f}%)")
            print("   " + "-" * 40)
            print(f"   Picker avg: {current_avg_picker_score:+.3f}")
            print(f"   Team point diff: {current_team_diff:+.1f}")
            print(f"   Called Ace Pick rate: {current_called_pick_rate:.1f}%")
            print(f"   JD Pick rate: {current_jd_pick_rate:.1f}%")
            print("   " + "-" * 20)
            print(f"   Leaster Rate: {current_leaster_rate:.2f}%")
            print(f"   Alone Call Rate: {current_alone_rate:.2f}%")
            print(f"   Called Under Rate: {current_called_under_rate:.2f}%")
            print(f"   Called 10s Rate: {current_called_10s_rate:.2f}%")
            print("   " + "-" * 40)
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed/60:.1f} min")
            print("   " + "-" * 40)

            # Save best model based on team point difference (lower is better)
            # We want the absolute value to be as small as possible
            if current_team_diff < best_team_difference:
                best_team_difference = current_team_diff
                agent.save(f'best_{activation}_ppo.pt')
                print(f"   🏆 New best team point difference: {best_team_difference:.1f}! Model saved.")

        # Save regular checkpoints
        if episode % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/{activation}_checkpoint_{episode}.pt'
            agent.save(checkpoint_path)

            # Save enhanced training plot
            if len(training_data['episodes']) > 10:
                plot_path = f'{checkpoint_dir}/training_progress_{episode}.png'
                save_training_plot(training_data, plot_path)

            # Calculate time since last checkpoint
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
    if agent.events:
        print("🔄 Final model update...")
        final_update_stats = agent.update()

        # Log final advantage and value target statistics
        if final_update_stats:
            adv_stats = final_update_stats['advantage_stats']
            val_stats = final_update_stats['value_target_stats']
            num_transitions = final_update_stats['num_transitions']

            print(f"   Final Transitions: {num_transitions}")
            print(f"   Final Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
            print(f"   Final Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")

    agent.save(f'final_{activation}_ppo.pt')

    # Save final enhanced training plot
    if len(training_data['episodes']) > 0:
        save_training_plot(training_data, f'final_{activation}_training.png')

    total_time = time.time() - start_time
    print("\n🎉 Training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}")
    print(f"   Final team point difference: {np.mean(team_point_differences) if team_point_differences else 0:.1f}")
    print(f"   Best team point difference: {best_team_difference:.1f}")
    total_called_picks = sum(pick_decisions[PARTNER_BY_CALLED_ACE])
    total_called_passes = sum(pass_decisions[PARTNER_BY_CALLED_ACE])
    total_jd_picks = sum(pick_decisions[PARTNER_BY_JD])
    total_jd_passes = sum(pass_decisions[PARTNER_BY_JD])
    final_called_pick_rate = (100 * total_called_picks / (total_called_picks + total_called_passes)) if (total_called_picks + total_called_passes) > 0 else 0
    final_jd_pick_rate = (100 * total_jd_picks / (total_jd_picks + total_jd_passes)) if (total_jd_picks + total_jd_passes) > 0 else 0
    print(f"   Final called Ace Pick rate: {final_called_pick_rate:.1f}%")
    print(f"   Final JD Pick rate: {final_jd_pick_rate:.1f}%")
    print(f"   Training speed: {(num_episodes-start_episode)/(total_time/60):.1f} episodes/min")

def main():
    parser = ArgumentParser(description="PPO training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=20000,
                       help="Number of training episodes (default: 20,000)")
    parser.add_argument("--update-interval", type=int, default=4096,
                       help="Number of games between model updates")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Number of episodes between checkpoints")
    parser.add_argument("--strategic-eval-interval", type=int, default=10000,
                       help="Number of episodes between strategic evaluations")
    parser.add_argument("--resume", type=str, default=None,
                       help="Model file to resume from")
    parser.add_argument("--activation", type=str, default='swish', choices=['relu', 'swish'],
                       help="Activation function to use (default: swish)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend('Agg')

    train_ppo(
        args.episodes,
        args.update_interval,
        args.save_interval,
        args.strategic_eval_interval,
        args.resume,
        args.activation
    )

if __name__ == "__main__":
    main()