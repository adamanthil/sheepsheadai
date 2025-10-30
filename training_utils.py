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
)


LEASTER_FINAL_REWARD_BONUS = 0.05
TRICK_POINT_RATIO = 360.0


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
    """Intermediate reward for trick points."""
    return trick_points / TRICK_POINT_RATIO


def is_same_team_as_winner(player, winner_pos: int, game) -> bool:
    if game.is_leaster:
        return False
    player_picker_team = (player.is_picker or player.is_partner or player.is_secret_partner)
    winner_player = game.players[winner_pos - 1]
    winner_picker_team = (winner_player.is_picker or winner_player.is_partner or winner_player.is_secret_partner)
    return player_picker_team == winner_picker_team


def apply_trick_rewards(trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float, game) -> None:
    for transition in trick_transitions:
        player = transition['player']
        reward_multiplier = 1.0 if is_same_team_as_winner(player, trick_winner_pos, game) else -1.0
        transition['intermediate_reward'] += trick_reward * reward_multiplier


def apply_leaster_trick_rewards(trick_transitions: List[Dict], trick_winner_pos: int, trick_reward: float) -> None:
    for transition in trick_transitions:
        player = transition['player']
        if player.position == trick_winner_pos:
            transition['intermediate_reward'] -= trick_reward


def update_intermediate_rewards_for_action(
    game,
    player,
    action,
    transition,
    current_trick_transitions,
    pick_weight=1.0, # Multiplier for pick head shaping
    partner_weight=1.0, # Multiplier for partner head shaping
    bury_weight=1.0, # Multiplier for bury head shaping
    play_weight=1.0, # Multiplier for play head shaping
):
    """Apply shared intermediate reward shaping and trick tracking.
    Uses game engine state to detect leads and trick phase; no counter needed.
    """
    action_name = ACTIONS[action - 1]

    # Hand-conditioned PICK/PASS shaping (small human-like nudges)
    if action_name in ("PICK", "PASS"):
        score = estimate_hand_strength_score(player.hand)
        if score <= 4:
            pick_bonus, pass_bonus = -0.15, +0.15
        elif score <= 6:
            pick_bonus, pass_bonus = 0, 0
        elif score <= 7:
            pick_bonus, pass_bonus = +0.05, -0.05
        elif score >= 8:
            pick_bonus, pass_bonus = +0.2, -0.2
        transition['intermediate_reward'] += pick_bonus if action_name == "PICK" else pass_bonus
        transition['intermediate_reward'] *= pick_weight

    # ALONE shaping: discourage going alone with weak hands
    elif action_name == "ALONE":
        score = estimate_hand_strength_score(player.hand)
        if score <= 8:
            transition['intermediate_reward'] += -0.2
        transition['intermediate_reward'] *= partner_weight

    # Bury penalty: discourage burying trump if not required
    elif "BURY" in action_name:
        card = action_name[5:]
        # Derive allowed bury actions from this step's valid_actions
        valid_actions = transition.get('valid_actions', set())
        allowed_bury_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
        has_allowed_fail_bury = any(get_card_suit(c) != "T" for c in allowed_bury_cards)
        has_allowed_trump_bury = any(get_card_suit(c) == "T" for c in allowed_bury_cards)

        if card in TRUMP and has_allowed_fail_bury:
            transition['intermediate_reward'] += -0.2
        elif card not in TRUMP and has_allowed_trump_bury:
            # Small preference when both options exist
            transition['intermediate_reward'] += 0.03
        elif not has_allowed_fail_bury and card.startswith("Q"):
            # Even if we have to bury trump, we should not bury queens.
            transition['intermediate_reward'] += -0.2
        elif not has_allowed_fail_bury and card.startswith("J"):
            # Even if we have to bury trump, unlikely we should bury jacks.
            transition['intermediate_reward'] += -0.1
        transition['intermediate_reward'] *= bury_weight

    elif "PLAY" in action_name:
        is_lead = (game.cards_played == 0) and (game.leader == player.position)
        if is_lead:
            valid_actions = transition.get('valid_actions', set())
            allowed_play_cards = [ACTIONS[id - 1][5:] for id in valid_actions]
            has_allowed_fail_play = any(get_card_suit(c) != "T" for c in allowed_play_cards)
            card = action_name[5:]
            if (
                not game.is_leaster
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and has_allowed_fail_play
                and card in TRUMP
            ):
                # Discourage defenders from leading trump
                transition['intermediate_reward'] += -0.15
            elif (
                game.called_card
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and not game.was_called_suit_played
                and game.called_suit == get_card_suit(card)
            ):
                # Encourage defenders to lead called suit (early)
                transition['intermediate_reward'] += 0.12 - (0.02 * game.current_trick)
            elif (
                not game.is_leaster
                and not player.is_picker
                and not player.is_partner
                and not player.is_secret_partner
                and card not in TRUMP
            ):
                # Nudge defenders toward leading fail
                transition['intermediate_reward'] += 0.05
            elif (
                not game.is_leaster
                and (player.is_partner or player.is_secret_partner)
                and card in TRUMP
            ):
                # Gentle nudge toward partners leading trump
                transition['intermediate_reward'] += 0.05
        transition['intermediate_reward'] *= play_weight

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
            # Increase final reward for leasters to compensate for negative trick rewards.
            # Agent should dislike playing leasters most of the time (similar to human behavior)
            # but without this the bias is a bit too strong.
            leaster_reward = final_score / 12 + LEASTER_FINAL_REWARD_BONUS
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
        corr = training_data['pick_hand_correlation']
        strat_eps = training_data.get('strategic_episodes', [])
        if strat_eps:
            # Align by trimming to common length from the tail
            n = min(len(corr), len(strat_eps))
            x = strat_eps[-n:]
            y = corr[-n:]
        else:
            # Fallback: align to tail of episodes
            x = episodes[-len(corr):]
            y = corr
        ax2_twin.plot(x, y, color='green', alpha=0.8, label='Hand Correlation', marker='o')
        ax2_twin.set_ylabel('Hand Strength Correlation')
        ax2_twin.legend(loc='upper right')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Pick Rate (%)')
    ax2.set_title('Pick Strategy Quality')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # Trump leading rates
    if 'picker_trump_rate' in training_data and len(training_data['picker_trump_rate']) > 0:
        picker_trump = training_data['picker_trump_rate']
        defender_trump = training_data.get('defender_trump_rate', [])
        strat_eps = training_data.get('strategic_episodes', [])
        if strat_eps:
            n = min(len(picker_trump), len(defender_trump), len(strat_eps))
            x = strat_eps[-n:]
            y1 = picker_trump[-n:]
            y2 = defender_trump[-n:]
        else:
            n = min(len(picker_trump), len(defender_trump))
            x = episodes[-n:]
            y1 = picker_trump[-n:]
            y2 = defender_trump[-n:]
        ax3.plot(x, y1, color='blue', alpha=0.8, label='Picker Team', marker='o')
        ax3.plot(x, y2, color='red', alpha=0.8, label='Defender Team', marker='o')
        ax3.axhline(y=60, color='blue', linestyle='--', alpha=0.5, label='Picker Target')
        ax3.axhline(y=20, color='red', linestyle='--', alpha=0.5, label='Defender Target')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Trump Lead Rate (%)')
        ax3.set_title('Trump Leading Strategy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Bury quality
    if 'bury_quality_rate' in training_data and len(training_data['bury_quality_rate']) > 0:
        bury = training_data['bury_quality_rate']
        strat_eps = training_data.get('strategic_episodes', [])
        if strat_eps:
            n = min(len(bury), len(strat_eps))
            x = strat_eps[-n:]
            y = bury[-n:]
        else:
            x = episodes[-len(bury):]
            y = bury
        ax4.plot(x, y, color='purple', alpha=0.8, marker='o')
        ax4.axhline(y=90, color='purple', linestyle='--', alpha=0.5, label='Good Target')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Good Bury Rate (%)')
        ax4.set_title('Bury Decision Quality')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Rates: Alone call and Leaster
    # When resuming training, historical CSVs may not contain these rate series.
    # Rather than require exact length matching with `episodes`, plot whatever
    # portion is available by aligning each series to the tail of the episodes
    # timeline. This keeps the figure renderable on resume while still showing
    # new data going forward.
    alone_rate = training_data.get('alone_rate', [])
    leaster_rate = training_data.get('leaster_rate', [])
    any_rate_plotted = False
    if alone_rate:
        ex = episodes[-len(alone_rate):]
        ax5.plot(ex, alone_rate, color='brown', alpha=0.8, label='Alone Call Rate')
        any_rate_plotted = True
    if leaster_rate:
        ex = episodes[-len(leaster_rate):]
        ax5.plot(ex, leaster_rate, color='gray', alpha=0.8, label='Leaster Rate')
        any_rate_plotted = True
    if any_rate_plotted:
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Rate (%)')
        ax5.set_title('Alone Call and Leaster Rates')
        ax5.legend()
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



