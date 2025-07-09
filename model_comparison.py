#!/usr/bin/env python3
"""
Model Comparison Script for Sheepshead

This script runs simulations of Sheepshead games with different models
playing against each other. It provides comprehensive statistical analysis
including mean, median, mode, standard deviation, and statistical significance tests.

Usage:
    python model_comparison.py --games 10000 --model1 path/to/model1.pth --model2 path/to/model2.pth
"""

import torch
import numpy as np
import random
import time
import json
import argparse
from collections import defaultdict, Counter
import scipy.stats as stats
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

from ppo import PPOAgent
from sheepshead import Game, ACTIONS, STATE_SIZE


@dataclass
class ModelConfig:
    """Configuration for a model in the simulation."""
    path: str
    name: str
    num_positions: int  # How many player positions this model controls
    activation: str = 'swish'


@dataclass
class GameResult:
    """Result of a single game."""
    game_id: int
    scores: List[int]  # Scores for positions 1-5
    picker_position: int
    is_leaster: bool
    picker_points: int
    defender_points: int


class ModelComparisonSimulator:
    """Simulator for comparing different Sheepshead models."""

    def __init__(self, model_configs: List[ModelConfig], num_games: int, seed: int = 42):
        """
        Initialize the simulator.

        Args:
            model_configs: List of model configurations
            num_games: Number of games to simulate
            seed: Random seed for reproducibility
        """
        self.model_configs = model_configs
        self.num_games = num_games
        self.seed = seed

        # Validate that total positions equals 5
        total_positions = sum(config.num_positions for config in model_configs)
        if total_positions != 5:
            raise ValueError(f"Total positions must equal 5, got {total_positions}")

        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Load models
        self.models = {}
        self.load_models()

        # Storage for results
        self.results: List[GameResult] = []
        self.model_scores: Dict[str, List[int]] = defaultdict(list)

        # Position rotation tracking
        self.position_rotation_cycle = 5  # Rotate every 5 games to ensure equal distribution

    def get_rotated_positions(self, game_id: int) -> Dict[str, List[int]]:
        """Get rotated position assignments for a given game."""
        rotation_offset = game_id % self.position_rotation_cycle

        # Start with positions 1-5
        positions = list(range(1, 6))

        # Rotate positions by offset
        rotated_positions = positions[rotation_offset:] + positions[:rotation_offset]

        # Assign positions to models
        position_assignments = {}
        pos_index = 0

        for config in self.model_configs:
            assigned_positions = []
            for _ in range(config.num_positions):
                assigned_positions.append(rotated_positions[pos_index])
                pos_index += 1
            position_assignments[config.name] = assigned_positions

        return position_assignments

    def load_models(self):
        """Load all models from their checkpoint files."""
        print("Loading models...")
        for config in self.model_configs:
            try:
                agent = PPOAgent(STATE_SIZE, len(ACTIONS), activation=config.activation)
                agent.load(config.path)
                self.models[config.name] = agent
                print(f"✅ Loaded {config.name} from {config.path}")
            except Exception as e:
                print(f"❌ Failed to load {config.name} from {config.path}: {e}")
                raise

    def run_simulation(self):
        """Run the full simulation of games."""
        print(f"\n🎮 Starting simulation of {self.num_games:,} games...")
        print("=" * 60)

        # Print model assignment info
        print("Model assignments (positions will rotate every 5 games):")
        for config in self.model_configs:
            print(f"  {config.name}: Controls {config.num_positions} position(s)")
        print()

        start_time = time.time()

        for game_id in range(self.num_games):
            if (game_id + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                games_per_sec = (game_id + 1) / elapsed
                remaining_games = self.num_games - (game_id + 1)
                eta = remaining_games / games_per_sec
                print(f"Game {game_id + 1:,}/{self.num_games:,} ({(game_id + 1)/self.num_games*100:.1f}%) "
                      f"- {games_per_sec:.1f} games/sec - ETA: {eta:.0f}s")

            # Get rotated positions for this game
            position_assignments = self.get_rotated_positions(game_id)

            result = self.play_game(game_id, position_assignments)
            self.results.append(result)

            # Track scores by model using the rotated positions
            for config in self.model_configs:
                model_positions = position_assignments[config.name]
                model_total_score = sum(result.scores[pos - 1] for pos in model_positions)
                self.model_scores[config.name].append(model_total_score)

        elapsed = time.time() - start_time
        print(f"\n✅ Simulation completed in {elapsed:.1f}s ({self.num_games/elapsed:.1f} games/sec)")

    def play_game(self, game_id: int, position_assignments: Dict[str, List[int]]) -> GameResult:
        """Play a single game and return the result."""
        game = Game()

        # Create model assignment map from position assignments
        position_to_model = {}
        for config in self.model_configs:
            model_positions = position_assignments[config.name]
            for pos in model_positions:
                position_to_model[pos] = self.models[config.name]

        # Play the game
        while not game.is_done():
            for player in game.players:
                valid_actions = player.get_valid_action_ids()

                while valid_actions:
                    # Get the model for this player position
                    model = position_to_model[player.position]

                    # Get deterministic action from model
                    state = player.get_state_vector()
                    action, _, _ = model.act(state, valid_actions, deterministic=True)

                    player.act(action)
                    valid_actions = player.get_valid_action_ids()

        # Collect results
        scores = [player.get_score() for player in game.players]
        picker_position = game.picker if game.picker else 0
        is_leaster = game.is_leaster
        picker_points = game.get_final_picker_points() if not is_leaster else 0
        defender_points = game.get_final_defender_points() if not is_leaster else 0

        return GameResult(
            game_id=game_id,
            scores=scores,
            picker_position=picker_position,
            is_leaster=is_leaster,
            picker_points=picker_points or 0,
            defender_points=defender_points or 0
        )

    def analyze_results(self):
        """Perform comprehensive statistical analysis of results."""
        print("\n📊 STATISTICAL ANALYSIS")
        print("=" * 60)

                # Basic statistics for each model
        summary_stats = {}
        for config in self.model_configs:
            scores = self.model_scores[config.name]

            mean_score = np.mean(scores)
            median_score = np.median(scores)
            std_score = np.std(scores)

            # Calculate mode (most frequent score)
            score_counts = Counter(scores)
            mode_score = score_counts.most_common(1)[0][0]
            mode_frequency = score_counts.most_common(1)[0][1]

            # Calculate percentiles
            percentiles = np.percentile(scores, [10, 25, 75, 90])

            summary_stats[config.name] = {
                'mean': mean_score,
                'median': median_score,
                'mode': mode_score,
                'mode_frequency': mode_frequency,
                'std': std_score,
                'min': min(scores),
                'max': max(scores),
                'p10': percentiles[0],
                'p25': percentiles[1],
                'p75': percentiles[2],
                'p90': percentiles[3],
                'num_games': len(scores)
            }

        # Print summary statistics
        print("Summary Statistics:")
        print("-" * 40)
        for model_name, model_stats in summary_stats.items():
            print(f"\n{model_name}:")
            print(f"  Mean Score:       {model_stats['mean']:+.3f}")
            print(f"  Median Score:     {model_stats['median']:+.3f}")
            print(f"  Mode Score:       {model_stats['mode']:+.0f} (appeared {model_stats['mode_frequency']} times)")
            print(f"  Std Deviation:    {model_stats['std']:.3f}")
            print(f"  Min Score:        {model_stats['min']:+.0f}")
            print(f"  Max Score:        {model_stats['max']:+.0f}")
            print(f"  10th Percentile:  {model_stats['p10']:+.1f}")
            print(f"  25th Percentile:  {model_stats['p25']:+.1f}")
            print(f"  75th Percentile:  {model_stats['p75']:+.1f}")
            print(f"  90th Percentile:  {model_stats['p90']:+.1f}")

        # Statistical significance tests
        print("\n\n🔬 STATISTICAL SIGNIFICANCE TESTS")
        print("=" * 60)

        model_names = list(summary_stats.keys())
        if len(model_names) >= 2:
            print("Pairwise Comparisons:")
            print("-" * 40)

            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    scores1 = self.model_scores[model1]
                    scores2 = self.model_scores[model2]

                    # Perform t-test
                    t_stat, t_p_value = stats.ttest_ind(scores1, scores2)

                    # Perform Mann-Whitney U test (non-parametric)
                    u_stat, u_p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')

                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt(((np.std(scores1) ** 2) + (np.std(scores2) ** 2)) / 2)
                    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std

                    print(f"\n{model1} vs {model2}:")
                    print(f"  Mean Difference:  {np.mean(scores1) - np.mean(scores2):+.3f}")
                    print(f"  T-test p-value:   {t_p_value:.6f} {'***' if t_p_value < 0.001 else '**' if t_p_value < 0.01 else '*' if t_p_value < 0.05 else ''}")
                    print(f"  Mann-Whitney p:   {u_p_value:.6f} {'***' if u_p_value < 0.001 else '**' if u_p_value < 0.01 else '*' if u_p_value < 0.05 else ''}")
                    print(f"  Effect Size (d):  {cohens_d:.3f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small' if abs(cohens_d) > 0.2 else 'negligible'})")

                    # Interpretation
                    if t_p_value < 0.001:
                        significance = "highly significant"
                    elif t_p_value < 0.01:
                        significance = "very significant"
                    elif t_p_value < 0.05:
                        significance = "significant"
                    else:
                        significance = "not significant"

                    better_model = model1 if np.mean(scores1) > np.mean(scores2) else model2
                    print(f"  Result: {significance} difference (p < 0.05), {better_model} performs better")

        return summary_stats

    def generate_plots(self, output_dir: str = "."):
        """Generate visualization plots."""
        print(f"\n📈 Generating plots in {output_dir}/...")

        # Set up matplotlib for non-interactive use
        plt.switch_backend('Agg')

        # 1. Score distribution plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Box plot
        score_data = [self.model_scores[config.name] for config in self.model_configs]
        model_names = [config.name for config in self.model_configs]

        ax1.boxplot(score_data, tick_labels=model_names)
        ax1.set_title('Score Distribution by Model')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)

        # Histogram
        for i, config in enumerate(self.model_configs):
            scores = self.model_scores[config.name]
            ax2.hist(scores, bins=30, alpha=0.7, label=config.name)

        ax2.set_title('Score Distribution Histogram')
        ax2.set_xlabel('Score')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/score_distributions.png', dpi=150, bbox_inches='tight')
        plt.close()

        # 2. Performance over time plot
        fig, ax = plt.subplots(figsize=(12, 6))

        window_size = max(100, self.num_games // 50)  # Adaptive window size

        for config in self.model_configs:
            scores = self.model_scores[config.name]
            # Calculate rolling average
            rolling_avg = []
            for i in range(len(scores)):
                start_idx = max(0, i - window_size + 1)
                rolling_avg.append(np.mean(scores[start_idx:i+1]))

            ax.plot(rolling_avg, label=config.name, alpha=0.8)

        ax.set_title(f'Performance Over Time (Rolling Average, Window={window_size})')
        ax.set_xlabel('Game Number')
        ax.set_ylabel('Average Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{output_dir}/performance_over_time.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("✅ Plots saved successfully")

    def save_results(self, filename: str = "simulation_results.json"):
        """Save detailed results to JSON file."""
        print(f"\n💾 Saving results to {filename}...")

        # Prepare data for JSON serialization
        results_data = {
            'simulation_config': {
                'num_games': self.num_games,
                'seed': self.seed,
                'position_rotation_cycle': self.position_rotation_cycle,
                'models': [
                    {
                        'name': config.name,
                        'path': config.path,
                        'num_positions': config.num_positions,
                        'activation': config.activation
                    }
                    for config in self.model_configs
                ]
            },
            'game_results': [
                {
                    'game_id': result.game_id,
                    'scores': result.scores,
                    'picker_position': result.picker_position,
                    'is_leaster': result.is_leaster,
                    'picker_points': result.picker_points,
                    'defender_points': result.defender_points,
                    'position_assignments': self.get_rotated_positions(result.game_id)
                }
                for result in self.results
            ],
            'model_scores': {
                name: scores for name, scores in self.model_scores.items()
            }
        }

        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)

        print(f"✅ Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare different Sheepshead models through simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare two models with 10k games (2 positions vs 3 positions)
  python model_comparison.py --games 10000 \\
    --model1-path checkpoints_swish/swish_checkpoint_1000000.pth \\
    --model1-positions 2 \\
    --model2-path checkpoints_swish/swish_checkpoint_500000.pth \\
    --model2-positions 3

  # Compare with custom names (3 positions vs 2 positions)
  python model_comparison.py --games 5000 \\
    --model1-path final_swish_ppo.pth --model1-name "Final Model" --model1-positions 3 \\
    --model2-path best_swish_ppo.pth --model2-name "Best Model" --model2-positions 2
        """
    )

    # General arguments
    parser.add_argument('--games', type=int, default=10000,
                       help='Number of games to simulate (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output-dir', type=str, default='.',
                       help='Output directory for plots and results (default: current directory)')

    # Model 1 arguments
    parser.add_argument('--model1-path', type=str, required=True,
                       help='Path to first model checkpoint')
    parser.add_argument('--model1-name', type=str, default='Model 1',
                       help='Name for first model (default: Model 1)')
    parser.add_argument('--model1-positions', type=int, required=True,
                       help='Number of positions for first model (1-4)')
    parser.add_argument('--model1-activation', type=str, default='swish', choices=['relu', 'swish'],
                       help='Activation function for first model (default: swish)')

    # Model 2 arguments
    parser.add_argument('--model2-path', type=str, required=True,
                       help='Path to second model checkpoint')
    parser.add_argument('--model2-name', type=str, default='Model 2',
                       help='Name for second model (default: Model 2)')
    parser.add_argument('--model2-positions', type=int, required=True,
                       help='Number of positions for second model (1-4)')
    parser.add_argument('--model2-activation', type=str, default='swish', choices=['relu', 'swish'],
                       help='Activation function for second model (default: swish)')

    args = parser.parse_args()

    # Validate position arguments
    if args.model1_positions + args.model2_positions != 5:
        parser.error(f"Total positions must equal 5, got {args.model1_positions + args.model2_positions}")

    if args.model1_positions < 1 or args.model1_positions > 4:
        parser.error("Model 1 positions must be between 1 and 4")

    if args.model2_positions < 1 or args.model2_positions > 4:
        parser.error("Model 2 positions must be between 1 and 4")

    # Create model configurations
    model_configs = [
        ModelConfig(
            path=args.model1_path,
            name=args.model1_name,
            num_positions=args.model1_positions,
            activation=args.model1_activation
        ),
        ModelConfig(
            path=args.model2_path,
            name=args.model2_name,
            num_positions=args.model2_positions,
            activation=args.model2_activation
        )
    ]

    # Run simulation
    simulator = ModelComparisonSimulator(model_configs, args.games, args.seed)
    simulator.run_simulation()

    # Analyze results
    stats_summary = simulator.analyze_results()

    # Generate plots
    simulator.generate_plots(args.output_dir)

    # Save results
    simulator.save_results(f"{args.output_dir}/simulation_results.json")

    print("\n🎉 Simulation complete!")
    print(f"Results saved in {args.output_dir}/")


if __name__ == "__main__":
    main()