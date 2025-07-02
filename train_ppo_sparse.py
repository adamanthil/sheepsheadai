#!/usr/bin/env python3
"""
Sparse reward PPO training for Sheepshead - FINAL SCORES ONLY.
Eliminates all intermediate rewards and reward shaping to test bias-free learning.
"""

import torch
import numpy as np
import random
import time
from collections import deque
from argparse import ArgumentParser

from ppo import PPOAgent
from sheepshead import Game, ACTIONS, STATE_SIZE

def train_sparse_ppo(num_episodes=50000, update_interval=1024):
    """
    Sparse reward PPO training - ONLY final game scores, no intermediate rewards.
    """
    print("Starting sparse reward PPO training...")
    print("REWARD STRUCTURE: Final game scores ONLY - no intermediate rewards")
    print(f"Training for {num_episodes} episodes")
    print(f"Update interval: {update_interval} (larger for sparse rewards)")

        # Create agent with settings optimized for sparse rewards
    agent = PPOAgent(STATE_SIZE, len(ACTIONS),
                    lr_actor=1e-4,   # Lower learning rate for stability
                    lr_critic=1e-4)  # gamma and gae_lambda are hardcoded in PPOAgent

    # Try to load existing model
    try:
        agent.load('best_sparse_ppo.pth')
        print("Loaded existing sparse model for continuation")
    except:
        print("Starting fresh sparse training")

    # Training tracking
    all_scores = deque(maxlen=1000)
    picker_scores = deque(maxlen=1000)
    recent_scores = deque(maxlen=100)
    pick_decisions = deque(maxlen=1000)
    pass_decisions = deque(maxlen=1000)
    best_avg_score = float('-inf')

    start_time = time.time()
    game_count = 0

    for episode in range(1, num_episodes + 1):
        game = Game()
        episode_scores = []
        episode_picks = 0
        episode_passes = 0

        # Store all transitions for this episode
        episode_transitions = []

        # Play full game
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    state = player.get_state_vector()
                    action, log_prob, value = agent.act(state, valid_actions)

                    # Track pick/pass decisions for statistics
                    action_name = ACTIONS[action - 1]
                    if action_name == "PICK":
                        episode_picks += 1
                    elif action_name == "PASS":
                        episode_passes += 1

                    # Store transition with ZERO intermediate reward
                    episode_transitions.append({
                        'player': player,
                        'state': state,
                        'action': action,
                        'log_prob': log_prob,
                        'value': value,
                        'valid_actions': valid_actions.copy()
                    })

                    # Execute action (this adds intermediate rewards to player.rewards, but we ignore them)
                    player.act(action)
                    valid_actions = player.get_valid_action_ids()

        # After game is complete, assign ONLY final scores as rewards
        final_scores = [player.get_score() for player in game.players]
        episode_scores = final_scores[:]

                # Process all stored transitions and assign final rewards
        for i, transition in enumerate(episode_transitions):
            player = transition['player']
            final_score = final_scores[player.position - 1]

            # ONLY final score as reward - normalized
            final_reward = final_score / 12  # Normalize to roughly [-1, 1] range

            # NO other rewards, NO bonuses, NO penalties, NO reward shaping
            # Just the pure final game outcome

            # Only mark the last transition as done
            done = (i == len(episode_transitions) - 1)
            agent.store_transition(
                transition['state'],
                transition['action'],
                final_reward,  # ONLY final score
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

        # Update model periodically (larger batches for sparse rewards)
        if game_count >= update_interval:
            print(f"Updating model after {game_count} games...")
            agent.update(epochs=10, batch_size=128)  # More epochs for sparse rewards
            game_count = 0

        # Progress reporting
        if episode % 1000 == 0:  # Less frequent reporting for cleaner output
            current_avg = np.mean(all_scores) if all_scores else 0
            current_picker = np.mean(picker_scores) if picker_scores else 0
            recent_avg = np.mean(recent_scores) if recent_scores else 0
            current_pick_rate = np.mean([p/(p+pa)*100 for p, pa in zip(pick_decisions, pass_decisions) if p+pa > 0]) if pick_decisions else 0
            elapsed = time.time() - start_time

            print(f"Episode {episode}/{num_episodes}")
            print(f"  Recent avg (100): {recent_avg:.3f}")
            print(f"  Overall avg (1000): {current_avg:.3f}")
            print(f"  Picker avg: {current_picker:.3f}")
            print(f"  Pick rate: {current_pick_rate:.1f}%")
            print(f"  Time elapsed: {elapsed:.1f}s")

            # Learning rate info
            current_lr = agent.actor_optimizer.param_groups[0]['lr']
            print(f"  Learning rate: {current_lr:.2e}")
            print("-" * 50)

            # Save best model
            if current_avg > best_avg_score:
                best_avg_score = current_avg
                agent.save('best_sparse_ppo.pth')
                print("New best average score! Saving model...")

        # Save checkpoint
        if episode % 10000 == 0:
            agent.save(f'sparse_ppo_checkpoint_{episode}.pth')
            print(f"Saving checkpoint at episode {episode}")

    # Final update if needed
    if len(agent.states) > 0:
        print("Final model update...")
        agent.update()

    agent.save('final_sparse_ppo.pth')
    print("Sparse training completed!")

def main():
    parser = ArgumentParser(description="Sparse reward PPO training for Sheepshead")
    parser.add_argument("--episodes", type=int, default=50000,
                       help="Number of training episodes")
    parser.add_argument("--update-interval", type=int, default=1024,
                       help="Number of games between model updates (larger for sparse rewards)")

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    train_sparse_ppo(args.episodes, args.update_interval)

if __name__ == "__main__":
    main()