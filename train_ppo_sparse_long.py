#!/usr/bin/env python3
"""
Long-term sparse reward PPO training for Sheepshead with self-play.
Optimized for extended training runs with comprehensive monitoring.
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
from sheepshead import Game, ACTIONS, STATE_SIZE

def save_training_plot(training_data, save_path='sparse_training_progress.png'):
    """Save training progress plots"""
    episodes = training_data['episodes']

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Score progression
    ax1.plot(episodes, training_data['recent_avg'], label='Recent (100)', alpha=0.7)
    ax1.plot(episodes, training_data['overall_avg'], label='Overall (1000)', alpha=0.8)
    ax1.plot(episodes, training_data['picker_avg'], label='Picker Avg', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Score')
    ax1.set_title('Score Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pick rate progression
    ax2.plot(episodes, training_data['pick_rate'], color='orange', alpha=0.8)
    ax2.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Optimal (~20%)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Pick Rate (%)')
    ax2.set_title('Picking Strategy Evolution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Learning rate
    ax3.plot(episodes, training_data['learning_rate'], color='green', alpha=0.8)
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.set_yscale('log')
    ax3.grid(True, alpha=0.3)

    # Training efficiency (games per minute)
    if len(training_data['time_elapsed']) > 1:
        games_per_min = [ep / (time_elapsed / 60) for ep, time_elapsed in zip(episodes, training_data['time_elapsed']) if time_elapsed > 0]
        ax4.plot(episodes[:len(games_per_min)], games_per_min, color='purple', alpha=0.8)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Games per Minute')
        ax4.set_title('Training Efficiency')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def train_sparse_ppo_long(num_episodes=200000, update_interval=2048, save_interval=5000):
    """
    Long-term sparse reward PPO training with comprehensive monitoring.
    """
    print("🚀 Starting LONG-TERM sparse reward PPO training...")
    print("="*60)
    print("TRAINING CONFIGURATION:")
    print(f"  Episodes: {num_episodes:,}")
    print(f"  Update interval: {update_interval}")
    print(f"  Save interval: {save_interval}")
    print("  Reward structure: SPARSE (final scores only)")
    print("  Opponent: SELF-PLAY")
    print("="*60)

    # Create agent with optimized hyperparameters for long training
    agent = PPOAgent(STATE_SIZE, len(ACTIONS),
                    lr_actor=2e-4,   # Slightly higher for longer training
                    lr_critic=2e-4)

    # Try to load existing model
    try:
        agent.load('best_sparse_long_ppo.pth')
        print("✅ Loaded existing long sparse model for continuation")
    except:
        print("🆕 Starting fresh long-term training")

    # Training tracking with more comprehensive metrics
    all_scores = deque(maxlen=2000)  # Larger history for long training
    picker_scores = deque(maxlen=2000)
    recent_scores = deque(maxlen=200)  # Larger recent window
    pick_decisions = deque(maxlen=2000)
    pass_decisions = deque(maxlen=2000)
    best_avg_score = float('-inf')

    # Training data for plotting
    training_data = {
        'episodes': [],
        'recent_avg': [],
        'overall_avg': [],
        'picker_avg': [],
        'pick_rate': [],
        'learning_rate': [],
        'time_elapsed': []
    }

    # Create checkpoint directory
    os.makedirs('checkpoints_sparse_long', exist_ok=True)

    start_time = time.time()
    game_count = 0
    last_checkpoint_time = start_time

    print(f"\n🎮 Beginning training... (target: {num_episodes:,} episodes)")
    print("-" * 60)

    for episode in range(1, num_episodes + 1):
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
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy()
                    })

                    # Execute action
                    player.act(action)
                    valid_actions = player.get_valid_action_ids()

        # Assign ONLY final scores as rewards
        final_scores = [player.get_score() for player in game.players]
        episode_scores = final_scores[:]

        # Process transitions with sparse rewards
        for i, transition in enumerate(episode_transitions):
            player = transition['player']
            final_score = final_scores[player.position - 1]

            # Sparse reward: ONLY final score, normalized
            final_reward = final_score / 12  # Normalize to [-1, 1] range

            # Mark done only for last transition
            done = (i == len(episode_transitions) - 1)
            agent.store_transition(
                transition['state'],
                transition['action'],
                final_reward,
                transition['value'],
                transition['log_prob'],
                done,
                transition['valid_actions']
            )

        # Track statistics
        avg_score = np.mean(episode_scores)
        picker_score = episode_scores[game.picker - 1] if game.picker else 0
        pick_rate = episode_picks / (episode_picks + episode_passes) * 100 if (episode_picks + episode_passes) > 0 else 0

        all_scores.append(avg_score)
        picker_scores.append(picker_score)
        recent_scores.append(avg_score)
        pick_decisions.append(episode_picks)
        pass_decisions.append(episode_passes)
        game_count += 1

        # Update model periodically
        if game_count >= update_interval:
            print(f"🔄 Updating model after {game_count} games... (Episode {episode})")
            agent.update(epochs=8, batch_size=256)  # Optimized for long training
            game_count = 0

        # Progress reporting and data collection
        if episode % 1000 == 0:
            current_avg = np.mean(all_scores) if all_scores else 0
            current_picker = np.mean(picker_scores) if picker_scores else 0
            recent_avg = np.mean(recent_scores) if recent_scores else 0
            current_pick_rate = np.mean([p/(p+pa)*100 for p, pa in zip(pick_decisions, pass_decisions) if p+pa > 0]) if pick_decisions else 0
            elapsed = time.time() - start_time

            # Collect data for plotting
            training_data['episodes'].append(episode)
            training_data['recent_avg'].append(recent_avg)
            training_data['overall_avg'].append(current_avg)
            training_data['picker_avg'].append(current_picker)
            training_data['pick_rate'].append(current_pick_rate)
            training_data['learning_rate'].append(agent.actor_optimizer.param_groups[0]['lr'])
            training_data['time_elapsed'].append(elapsed)

            # Calculate training speed
            games_per_min = episode / (elapsed / 60) if elapsed > 0 else 0

            print(f"📊 Episode {episode:,}/{num_episodes:,} ({episode/num_episodes*100:.1f}%)")
            print(f"   Recent avg (200): {recent_avg:+.3f}")
            print(f"   Overall avg (2000): {current_avg:+.3f}")
            print(f"   Picker avg: {current_picker:+.3f}")
            print(f"   Pick rate: {current_pick_rate:.1f}%")
            print(f"   Training speed: {games_per_min:.1f} games/min")
            print(f"   Time elapsed: {elapsed/60:.1f} min")
            print("   " + "-" * 40)

            # Save best model based on picker performance (more meaningful than overall avg which is always 0)
            if current_picker > best_avg_score:
                best_avg_score = current_picker
                agent.save('best_sparse_long_ppo.pth')
                print(f"   🏆 New best picker avg: {best_avg_score:.3f}! Model saved.")

        # Save regular checkpoints
        if episode % save_interval == 0:
            checkpoint_path = f'checkpoints_sparse_long/sparse_long_checkpoint_{episode}.pth'
            agent.save(checkpoint_path)

            # Save training plot
            if len(training_data['episodes']) > 10:
                plot_path = f'checkpoints_sparse_long/training_progress_{episode}.png'
                save_training_plot(training_data, plot_path)

            # Calculate time since last checkpoint
            checkpoint_time = time.time()
            time_since_last = checkpoint_time - last_checkpoint_time
            last_checkpoint_time = checkpoint_time

            print(f"💾 Checkpoint saved at episode {episode:,}")
            print(f"   Time for last {save_interval:,} episodes: {time_since_last/60:.1f} min")
            print(f"   Estimated time remaining: {(num_episodes-episode)*(time_since_last/save_interval)/60:.1f} min")

    # Final update and save
    if len(agent.states) > 0:
        print("🔄 Final model update...")
        agent.update()

    agent.save('final_sparse_long_ppo.pth')

    # Save final training plot
    if len(training_data['episodes']) > 0:
        save_training_plot(training_data, 'final_sparse_long_training.png')

    total_time = time.time() - start_time
    print(f"\n🎉 Long-term sparse training completed!")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print(f"   Final picker average: {np.mean(picker_scores) if picker_scores else 0:.3f}")
    print(f"   Best picker average: {best_avg_score:.3f}")
    print(f"   Final pick rate: {np.mean([p/(p+pa)*100 for p, pa in zip(pick_decisions, pass_decisions) if p+pa > 0]):.1f}%")
    print(f"   Training speed: {num_episodes/(total_time/60):.1f} episodes/min")

def main():
    parser = ArgumentParser(description="Long-term sparse reward PPO training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=200000,
                       help="Number of training episodes (default: 200,000)")
    parser.add_argument("--update-interval", type=int, default=2048,
                       help="Number of games between model updates")
    parser.add_argument("--save-interval", type=int, default=5000,
                       help="Number of episodes between checkpoints")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Ensure matplotlib uses a non-interactive backend
    plt.switch_backend('Agg')

    train_sparse_ppo_long(args.episodes, args.update_interval, args.save_interval)

if __name__ == "__main__":
    main()