#!/usr/bin/env python3
"""
Training utilities shared across training scripts.
"""

from typing import List, Dict
import matplotlib.pyplot as plt

from sheepshead import (
    TRUMP,
    ACTIONS,
    PARTNER_BY_CALLED_ACE,
    PARTNER_BY_JD,
    get_card_suit,
    get_cards_from_vector,
)


def estimate_hand_strength_score(cards: List[str]) -> int:
    """Return a simple strength score for a hand based on trump density.

    Scoring heuristic:
      - +3 per Queen
      - +2 per Jack
      - +1 per other trump
    """
    score = 0
    for card in cards:
        if card in TRUMP:
            if card.startswith("Q"):
                score += 3
            elif card.startswith("J"):
                score += 2
            else:
                score += 1
    return score


def estimate_hand_strength_category(cards: List[str]) -> str:
    """Categorize hand strength into 'weak' | 'medium' | 'strong'.

    Bins:
      - weak   ≤ 4
      - medium 5–7
      - strong ≥ 8
    """
    score = estimate_hand_strength_score(cards)
    if score <= 4:
        return "weak"
    if score <= 7:
        return "medium"
    return "strong"


def get_partner_selection_mode(episode: int) -> int:
    return PARTNER_BY_CALLED_ACE if (episode % 2 == 0) else PARTNER_BY_JD


def calculate_trick_reward(trick_points: int) -> float:
    """Intermediate reward for trick points (scaled)."""
    return trick_points / 720.0  # 720 = 60 * 12


def is_same_team_as_winner(player, winner_pos: int, game) -> bool:
    if game.is_leaster:
        return False
    player_picker_team = (player.is_picker or player.is_partner)
    winner_player = game.players[winner_pos - 1]
    winner_picker_team = (winner_player.is_picker or winner_player.is_partner)
    return player_picker_team == winner_picker_team


def apply_trick_rewards(trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float, game) -> None:
    for transition in trick_transitions:
        player = transition['player']
        if player.game.partner:  # Partner known
            reward_multiplier = 1.0 if is_same_team_as_winner(player, trick_winner_pos, game) else -1.0
        else:  # Partner unknown
            if player.position == trick_winner_pos:
                reward_multiplier = 1.0
            elif player.is_secret_partner:
                reward_multiplier = 1.0 if player.game.picker == trick_winner_pos else -1.0
            elif (not player.is_secret_partner) and (player.game.picker == trick_winner_pos):
                reward_multiplier = -1.0
            else:
                reward_multiplier = 0.0
        transition['intermediate_reward'] += trick_reward * reward_multiplier


def apply_leaster_trick_rewards(trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float) -> None:
    for transition in trick_transitions:
        player = transition['player']
        if player.position == trick_winner_pos:
            transition['intermediate_reward'] -= trick_reward


def update_intermediate_rewards_for_action(game, player, action, transition, current_trick_transitions):
    """Apply shared intermediate reward shaping and trick tracking.
    Uses game engine state to detect leads and trick phase; no counter needed.
    """
    action_name = ACTIONS[action - 1]
    state_vec = transition.get('state')

    # Hand-conditioned PICK/PASS shaping (small human-like nudges)
    if action_name in ("PICK", "PASS"):
        hand_cards = get_cards_from_vector(state_vec[16:48])
        score = estimate_hand_strength_score(hand_cards)
        if score <= 4:
            pick_bonus, pass_bonus = -0.06, +0.06
        elif score <= 7:
            pick_bonus, pass_bonus = +0.03, -0.03
        else:
            pick_bonus, pass_bonus = +0.08, -0.08
        transition['intermediate_reward'] += pick_bonus if action_name == "PICK" else pass_bonus

    # Bury penalty: discourage burying trump if not required
    if "BURY" in action_name:
        card = action_name[5:]
        # Derive allowed bury actions from this step's valid_actions
        valid_actions = transition.get('valid_actions', set())
        allowed_bury_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
        has_allowed_fail_bury = any(get_card_suit(c) != "T" for c in allowed_bury_cards)
        has_allowed_trump_bury = any(get_card_suit(c) == "T" for c in allowed_bury_cards)

        if card in TRUMP and has_allowed_fail_bury:
            transition['intermediate_reward'] += -0.06
        elif card not in TRUMP and has_allowed_trump_bury:
            # Small preference when both options exist
            transition['intermediate_reward'] += 0.01

    if "PLAY" in action_name:
        is_lead = (game.cards_played == 0) and (game.leader == player.position)
        if is_lead:
            card = action_name[5:]
            if (
                not game.is_leaster
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and card in TRUMP
            ):
                # Discourage defenders from leading trump
                transition['intermediate_reward'] += -0.05
            elif (
                game.called_card
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and not game.was_called_suit_played
                and game.called_suit == get_card_suit(card)
            ):
                # Encourage defenders to lead called suit
                transition['intermediate_reward'] += 0.05

        current_trick_transitions.append(transition)


def handle_trick_completion(game, current_trick_transitions):
    """If a trick has completed, apply trick-based rewards and reset tracking.

    Returns True if the trick just completed, else False.
    """
    if game.was_trick_just_completed:
        trick_points = game.trick_points[game.current_trick - 1]
        trick_winner = game.trick_winners[game.current_trick - 1]
        trick_reward = calculate_trick_reward(trick_points)

        if not game.is_leaster:
            apply_trick_rewards(current_trick_transitions, trick_winner, trick_reward, game)
        else:
            apply_leaster_trick_rewards(current_trick_transitions, trick_winner, trick_reward)

        # Reset tracking for next trick
        current_trick_transitions.clear()
        return True

    return False


def process_episode_rewards(episode_transitions, final_scores, last_transition_per_player, is_leaster):
    """Process and assign rewards to all transitions in the episode."""
    for i, transition in enumerate(episode_transitions):
        player = transition['player']
        player_pos = player.position
        final_score = final_scores[player_pos - 1]
        is_episode_done = (i == last_transition_per_player[player_pos])

        if is_leaster:
            # Downweight all leaster rewards.
            # Agent should dislike playing leasters most of the time (similar to human behavior).
            leaster_reward = (final_score - 2) / 12
            final_reward = leaster_reward if is_episode_done else 0.0
        else:
            final_reward = (final_score / 12) if is_episode_done else 0.0

        total_reward = final_reward + transition['intermediate_reward']
        yield {
            'transition': transition,
            'reward': total_reward,
            'done': is_episode_done
        }


def save_training_plot(training_data, save_path='training_progress.png'):
    """Enhanced training plots with strategic metrics."""
    episodes = training_data['episodes']

    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(15, 15))

    # Score progression
    ax1.plot(episodes, training_data['picker_avg'], label='Picker Avg', alpha=0.8)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Average Picker Score')
    ax1.set_title('Score Progression')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Pick rate and correlation
    ax2.plot(episodes, training_data['called_pick_rate'], color='orange', alpha=0.8, label='Called Ace Pick Rate')
    ax2.plot(episodes, training_data['jd_pick_rate'], color='purple', alpha=0.8, label='JD Pick Rate')
    ax2_twin = ax2.twinx()
    if 'pick_hand_correlation' in training_data and len(training_data['pick_hand_correlation']) > 0:
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



