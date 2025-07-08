#!/usr/bin/env python3
"""
Extended long-term sparse reward PPO training for Sheepshead.
Builds on the successful sparse_long approach with strategic evaluation metrics.
"""

import torch
import numpy as np
import random
import time
import os
from collections import deque, defaultdict, Counter
from argparse import ArgumentParser
import matplotlib.pyplot as plt

from ppo import PPOAgent
from sheepshead import Game, ACTIONS, STATE_SIZE, ACTION_LOOKUP, ACTION_IDS, TRUMP

def analyze_strategic_decisions(agent, num_samples=100):
    """Analyze strategic decision quality instead of random opponent evaluation."""

    # Trump leading analysis
    trump_leads = {'picker_team': 0, 'picker_total': 0, 'defender_team': 0, 'defender_total': 0}

    # Bury quality analysis
    bury_quality = {'good_burys': 0, 'bad_burys': 0, 'total_burys': 0}

    # Pick decision correlation with hand strength
    pick_decisions = []
    hand_strengths = []

    for _ in range(num_samples):
        game = Game()

        # Analyze pick decisions
        initial_player = game.players[0]
        hand_strength = sum(3 if c[0] == 'Q' else 2 if c[0] == 'J' else 1 if c in TRUMP else 0 for c in initial_player.hand)

        state = torch.FloatTensor(initial_player.get_state_vector()).unsqueeze(0)
        with torch.no_grad():
            action_probs = agent.actor(state)

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

def save_extended_training_plot(training_data, save_path='extended_training_progress.png'):
    """Enhanced training plots with strategic metrics."""
    episodes = training_data['episodes']

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # Score progression
    ax1.plot(episodes, training_data['recent_avg'], label='Recent (200)', alpha=0.7)
    ax1.plot(episodes, training_data['overall_avg'], label='Overall (2000)', alpha=0.8)
    ax1.plot(episodes, training_data['picker_avg'], label='Picker Avg', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Picker Score')
    ax1.set_title('Score Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

        # Pick rate and correlation
    ax2.plot(episodes, training_data['pick_rate'], color='orange', alpha=0.8, label='Pick Rate')
    ax2_twin = ax2.twinx()
    if 'pick_hand_correlation' in training_data and len(training_data['pick_hand_correlation']) > 0:
        # Create episode list for strategic metrics (only available every strategic_eval_interval)
        strategic_episodes = episodes[::len(episodes)//len(training_data['pick_hand_correlation'])] if len(training_data['pick_hand_correlation']) > 0 else []
        if len(strategic_episodes) == len(training_data['pick_hand_correlation']):
            ax2_twin.plot(strategic_episodes, training_data['pick_hand_correlation'], color='green', alpha=0.8, label='Hand Correlation', marker='o')
            ax2_twin.set_ylabel('Hand Strength Correlation')
            ax2_twin.legend(loc='upper right')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Pick Rate (%)')
    ax2.set_title('Pick Strategy Quality')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Trump leading rates
    if 'picker_trump_rate' in training_data and len(training_data['picker_trump_rate']) > 0:
        strategic_episodes = episodes[::len(episodes)//len(training_data['picker_trump_rate'])] if len(training_data['picker_trump_rate']) > 0 else []
        if len(strategic_episodes) == len(training_data['picker_trump_rate']):
            ax3.plot(strategic_episodes, training_data['picker_trump_rate'], color='blue', alpha=0.8, label='Picker Team', marker='o')
            ax3.plot(strategic_episodes, training_data['defender_trump_rate'], color='red', alpha=0.8, label='Defender Team', marker='o')
            ax3.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='Picker Target')
            ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Defender Target')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Trump Lead Rate (%)')
            ax3.set_title('Trump Leading Strategy')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

    # Bury quality
    if 'bury_quality_rate' in training_data and len(training_data['bury_quality_rate']) > 0:
        strategic_episodes = episodes[::len(episodes)//len(training_data['bury_quality_rate'])] if len(training_data['bury_quality_rate']) > 0 else []
        if len(strategic_episodes) == len(training_data['bury_quality_rate']):
            ax4.plot(strategic_episodes, training_data['bury_quality_rate'], color='purple', alpha=0.8, marker='o')
            ax4.axhline(y=90, color='purple', linestyle='--', alpha=0.5, label='Good Target')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Good Bury Rate (%)')
            ax4.set_title('Bury Decision Quality')
            ax4.legend()
            ax4.grid(True, alpha=0.3)

    # Training efficiency
    games_per_min = [ep / (time_elapsed / 60) for ep, time_elapsed in zip(episodes, training_data['time_elapsed']) if time_elapsed > 0]
    ax5.plot(episodes[:len(games_per_min)], games_per_min, color='brown', alpha=0.8)
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Games per Minute')
    ax5.set_title('Training Efficiency')
    ax5.grid(True, alpha=0.3)

    # Team Point Difference
    if 'team_point_diff' in training_data and len(training_data['team_point_diff']) > 0:
        ax6.plot(episodes, training_data['team_point_diff'], color='red', alpha=0.8)
        ax6.axhline(y=0, color='black', linestyle='--', alpha=0.5, label='Perfect Balance')
        ax6.set_xlabel('Episode')
        ax6.set_ylabel('Team Point Difference')
        ax6.set_title('Team Point Difference (Picker - Defender)')
        ax6.legend()
        ax6.grid(True, alpha=0.3)


    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_sparse_ppo_extended(num_episodes=300000, update_interval=2048, save_interval=5000,
                            strategic_eval_interval=10000, resume_model=None, activation='relu'):
    """
    Extended sparse reward PPO training with strategic evaluation metrics.
    """
    print("🚀 Starting EXTENDED sparse reward PPO training...")
    print("="*60)
    print("EXTENDED TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print(f"  Strategic evaluation interval: {strategic_eval_interval}")
    print(f"  Activation function: {activation.upper()}")
    print("  Reward structure: SPARSE (final scores only)")
    print("  Opponent: SELF-PLAY")
    print("  Evaluation: STRATEGIC DECISION QUALITY + TEAM BALANCE")
    print("  Best model: LOWEST TEAM POINT DIFFERENCE")
    print("="*60)

    # Create agent with optimized hyperparameters
    agent = PPOAgent(STATE_SIZE, len(ACTIONS),
                    lr_actor=1.5e-4,  # Slightly reduced for fine-tuning
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
        # Try to load the best existing model
        for model_name in ['final_sparse_long_ppo.pth', 'best_sparse_long_ppo.pth']:
            try:
                agent.load(model_name)
                print(f"✅ Loaded existing {model_name} for continuation")
                break
            except:
                continue
        else:
            print("🆕 Starting fresh extended training")

    # Enhanced training tracking
    all_scores = deque(maxlen=3000)
    picker_scores = deque(maxlen=3000)
    recent_scores = deque(maxlen=300)
    pick_decisions = deque(maxlen=3000)
    pass_decisions = deque(maxlen=3000)
    team_point_differences = deque(maxlen=3000)
    best_team_difference = float('inf')  # Lower is better (smaller point difference)

    # Extended training data for plotting
    training_data = {
        'episodes': [],
        'recent_avg': [],
        'overall_avg': [],
        'picker_avg': [],
        'pick_rate': [],
        'learning_rate': [],
        'time_elapsed': [],
        'pick_hand_correlation': [],
        'picker_trump_rate': [],
        'defender_trump_rate': [],
        'bury_quality_rate': [],
        'team_point_diff': []
    }

    # Create checkpoint directory with activation function suffix
    checkpoint_dir = f'checkpoints_extended_{activation}'
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_time = time.time()
    game_count = 0
    last_checkpoint_time = start_time

    print(f"\n🎮 Beginning extended training... (target: {num_episodes:,} episodes)")
    print("-" * 60)

    for episode in range(start_episode + 1, num_episodes + 1):
        game = Game()
        episode_scores = []
        episode_picks = 0
        episode_passes = 0

        # Store all transitions for this episode
        episode_transitions = []

        # Play full game with self-play
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    state = player.get_state_vector()
                    action, log_prob, value = agent.act(state, valid_actions)

                    # Track pick/pass decisions
                    action_name = ACTIONS[action - 1]
                    if action_name == "PICK":
                        episode_picks += 1
                    elif action_name == "PASS":
                        episode_passes += 1

                    # Store transition (no intermediate reward)
                    episode_transitions.append({
                        'player': player,
                        'state': state,
                        'action': action,
                        'action_name': action_name,
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy()
                    })

                    # Execute action
                    player.act(action)
                    valid_actions = player.get_valid_action_ids()

        # Assign rewards only to last PLAY action of each player
        final_scores = [player.get_score() for player in game.players]
        episode_scores = final_scores[:]

        # Find the last PLAY action for each player
        last_play_indices = {}
        for i, transition in enumerate(episode_transitions):
            player_pos = transition['player'].position
            if "PLAY" in transition['action_name']:
                last_play_indices[player_pos] = i

        # Process transitions with sparse rewards
        for i, transition in enumerate(episode_transitions):
            player = transition['player']
            player_pos = player.position
            final_score = final_scores[player_pos - 1]

            # Give normalized reward to the last PLAY action of each player
            if i in last_play_indices.values():
                reward = final_score / 12
                done = True
            else:
                reward = 0.0
                done = False

            agent.store_transition(
                transition['state'],
                transition['action'],
                reward,
                transition['value'],
                transition['log_prob'],
                done,
                transition['valid_actions']
            )

        # Track statistics
        avg_score = np.mean(episode_scores)
        picker_score = episode_scores[game.picker - 1] if game.picker else 0
        pick_rate = episode_picks / (episode_picks + episode_passes) * 100 if (episode_picks + episode_passes) > 0 else 0

        # Calculate team point difference (picker team points - defender team points)
        if game.picker and not game.is_leaster:
            picker_team_points = game.get_final_picker_points()
            defender_team_points = game.get_final_defender_points()
            team_point_diff = abs(picker_team_points - defender_team_points)
        else:
            team_point_diff = 0  # No team difference in leaster games

        all_scores.append(avg_score)
        picker_scores.append(picker_score)
        recent_scores.append(avg_score)
        pick_decisions.append(episode_picks)
        pass_decisions.append(episode_passes)
        team_point_differences.append(team_point_diff)
        game_count += 1

        # Update model periodically
        if game_count >= update_interval:
            print(f"🔄 Updating model after {game_count} games... (Episode {episode:,})")
            update_stats = agent.update(epochs=8, batch_size=256)

            # Log advantage and value target statistics
            if update_stats:
                adv_stats = update_stats['advantage_stats']
                val_stats = update_stats['value_target_stats']
                num_transitions = update_stats['num_transitions']

                print(f"   Transitions: {num_transitions}")
                print(f"   Advantages - Mean: {adv_stats['mean']:+.3f}, Std: {adv_stats['std']:.3f}, Range: [{adv_stats['min']:+.3f}, {adv_stats['max']:+.3f}]")
                print(f"   Value Targets - Mean: {val_stats['mean']:+.3f}, Std: {val_stats['std']:.3f}, Range: [{val_stats['min']:+.3f}, {val_stats['max']:+.3f}]")

            game_count = 0

                # Strategic evaluation at intervals
        if episode % strategic_eval_interval == 0:
            print(f"🧠 Analyzing strategic decisions... (Episode {episode:,})")
            strategic_metrics = analyze_strategic_decisions(agent, num_samples=200)

            # Store strategic metrics
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
            current_avg = np.mean(all_scores) if all_scores else 0
            current_picker = np.mean(picker_scores) if picker_scores else 0
            recent_avg = np.mean(recent_scores) if recent_scores else 0
            current_pick_rate = np.mean([p/(p+pa)*100 for p, pa in zip(pick_decisions, pass_decisions) if p+pa > 0]) if pick_decisions else 0
            current_team_diff = np.mean(team_point_differences) if team_point_differences else 0
            elapsed = time.time() - start_time

            # Collect data for plotting
            training_data['episodes'].append(episode)
            training_data['recent_avg'].append(recent_avg)
            training_data['overall_avg'].append(current_avg)
            training_data['picker_avg'].append(current_picker)
            training_data['pick_rate'].append(current_pick_rate)
            training_data['learning_rate'].append(agent.actor_optimizer.param_groups[0]['lr'])
            training_data['time_elapsed'].append(elapsed)
            training_data['team_point_diff'].append(current_team_diff)

            # Strategic metrics are collected separately during strategic evaluation intervals
            # Don't try to collect them here as they're not always available

            # Calculate training speed
            games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

            print(f"📊 Episode {episode:,}/{num_episodes:,} ({episode/num_episodes*100:.1f}%)")
            print(f"   Recent avg (300): {recent_avg:+.3f}")
            print(f"   Overall avg (3000): {current_avg:+.3f}")
            print(f"   Picker avg: {current_picker:+.3f}")
            print(f"   Team point diff: {current_team_diff:+.1f}")
            print(f"   Pick rate: {current_pick_rate:.1f}%")
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed/60:.1f} min")
            print("   " + "-" * 40)

            # Save best model based on team point difference (lower is better)
            # We want the absolute value to be as small as possible
            if current_team_diff < best_team_difference:
                best_team_difference = current_team_diff
                agent.save(f'best_extended_{activation}_ppo.pth')
                print(f"   🏆 New best team point difference: {best_team_difference:.1f}! Model saved.")

        # Save regular checkpoints
        if episode % save_interval == 0:
            checkpoint_path = f'{checkpoint_dir}/extended_{activation}_checkpoint_{episode}.pth'
            agent.save(checkpoint_path)

            # Save enhanced training plot
            if len(training_data['episodes']) > 10:
                plot_path = f'{checkpoint_dir}/training_progress_{episode}.png'
                save_extended_training_plot(training_data, plot_path)

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
    if len(agent.states) > 0:
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

    agent.save(f'final_extended_{activation}_ppo.pth')

    # Save final enhanced training plot
    if len(training_data['episodes']) > 0:
        save_extended_training_plot(training_data, f'final_extended_{activation}_training.png')

    total_time = time.time() - start_time
    print(f"\n🎉 Extended sparse training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}")
    print(f"   Final team point difference: {np.mean(team_point_differences) if team_point_differences else 0:.1f}")
    print(f"   Best team point difference: {best_team_difference:.1f}")
    print(f"   Final pick rate: {np.mean([p/(p+pa)*100 for p, pa in zip(pick_decisions, pass_decisions) if p+pa > 0]):.1f}%")
    print(f"   Training speed: {(num_episodes-start_episode)/(total_time/60):.1f} episodes/min")

def main():
    parser = ArgumentParser(description="Extended sparse reward PPO training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=300000,
                       help="Number of training episodes (default: 300,000)")
    parser.add_argument("--update-interval", type=int, default=2048,
                       help="Number of games between model updates")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Number of episodes between checkpoints")
    parser.add_argument("--strategic-eval-interval", type=int, default=10000,
                       help="Number of episodes between strategic evaluations")
    parser.add_argument("--resume", type=str, default=None,
                       help="Model file to resume from")
    parser.add_argument("--activation", type=str, default='relu', choices=['relu', 'swish'],
                       help="Activation function to use (default: relu)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend('Agg')

    train_sparse_ppo_extended(
        args.episodes,
        args.update_interval,
        args.save_interval,
        args.strategic_eval_interval,
        args.resume,
        args.activation
    )

if __name__ == "__main__":
    main()