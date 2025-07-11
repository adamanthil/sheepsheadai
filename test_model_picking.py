#!/usr/bin/env python3
"""
Test the final trained model to see if it learned proper hand strength correlations.
"""

import torch
import numpy as np
from argparse import ArgumentParser
import random
import re

from ppo import PPOAgent
from sheepshead import Game, Player, ACTIONS, DECK, STATE_SIZE, ACTION_IDS, TRUMP, pretty_card_list

def calculate_display_width(text):
    """Calculate the visible width of text, excluding ANSI escape sequences."""
    # Remove ANSI escape sequences using regex
    ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
    clean_text = ansi_escape.sub('', text)
    return len(clean_text)

def pad_text_with_ansi(text, width):
    """Pad text to a specific width, accounting for ANSI escape sequences."""
    display_width = calculate_display_width(text)
    padding_needed = width - display_width
    if padding_needed > 0:
        return text + ' ' * padding_needed
    return text

def calculate_hand_strength(hand):
    """Calculate objective hand strength score."""
    strength = 0
    for card in hand:
        if card[0] == 'Q':  # Queens are very strong
            strength += 4
        elif card == 'JD':  # Jack of Diamonds should discourage picking
            strength -= 1
        elif card[0] == 'J':  # Jacks are strong
            strength += 2
        elif card in TRUMP:  # Other trump cards
            strength += 1
        elif card[0] == 'A':  # Aces are okay
            strength += 0.5
        elif card[0:2] == '10':  # Tens are okayish
            strength += 0.5
    return strength


def test_final_model(model_path, position, random_hands):
    """Test if the final model learned proper hand strength correlations."""

    print("🎯 TESTING FINAL TRAINED MODEL")
    print("="*50)

    # Create agent and load final model
    agent = PPOAgent(STATE_SIZE, len(ACTIONS), lr_actor=1e-3, lr_critic=1e-3)

    try:
        agent.load(model_path)
        print(f"✅ Loaded {model_path}")
    except FileNotFoundError:
        print("❌ No trained model found")
        return

    if random_hands:
        test_hands = [
            (random.sample(DECK, 6), None)
            for _ in range(100)
        ]
    else:
        test_hands = [
            # Very weak hands (should almost never pick)
            (["7C", "8C", "9C", "10C", "KC", "AC"], "Very Weak - All fail clubs"),
            (["7S", "8S", "9S", "KS", "AS", "7H"], "Weak - Mostly fail"),
            (["8H", "9H", "10H", "KH", "AH", "7C"], "Weak - No trump"),

            # Medium hands (borderline picks)
            (["JC", "8D", "9D", "KD", "AC", "10S"], "Tempting but no - Jack of Clubs + 3 other trump"),
            (["QD", "JH", "10D", "9D", "7S", "8C"], "Red death - 1 queen + 2 Jack + 2 little trump"),
            (["QS", "7D", "8D", "9D", "AS", "10H"], "Medium - Q spades + 3 little trump"),

            # Strong hands (should often pick)
            (["QC", "JD", "AD", "10D", "KD", "9H"], "Strong - Queen + jack + only 1 fail. Forced alone"),
            (["QS", "QH", "JC", "JS", "AC", "10S"], "Strong - 2 queens + 2 jacks"),
            (["QC", "QS", "QH", "JD", "JC", "JS"], "Extremely Strong - 3 queens + 3 jacks"),
        ]

    print("Testing hand strength vs pick probability:")
    print("-" * 60)

    pick_counts = 0
    hand_data = []

    for hand, description in test_hands:
        # Setup game state for pick decision
        game = Game()
        player = Player(game, position, hand)
        game.last_passed = position - 1

        # Get state and valid actions
        state = player.get_state_vector()
        valid_actions = player.get_valid_action_ids()

        # Get action probabilities
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
        for action in valid_actions:
            action_mask[action - 1] = True

        with torch.no_grad():
            action_probs = agent.actor(state_tensor, action_mask.unsqueeze(0))

        # Extract pick/pass probabilities
        pick_prob = action_probs[0][ACTION_IDS["PICK"] - 1].item()
        pass_prob = action_probs[0][ACTION_IDS["PASS"] - 1].item()
        pick_percentage = pick_prob / (pick_prob + pass_prob) * 100

        if pick_percentage > 50:
            pick_counts += 1

        # Calculate objective hand strength
        strength = calculate_hand_strength(hand)

        hand_data.append({
            "hand": hand,
            "description": description,
            "strength": strength,
            "pick_percentage": pick_percentage
        })

    for hand in sorted(hand_data, key=lambda x: x["strength"]):
        description = hand["description"] if hand["description"] else pretty_card_list(hand["hand"])
        padded_description = pad_text_with_ansi(description, 50)
        checkbox = "\033[92m✅\033[0m" if hand["pick_percentage"] > 50 else ""
        print(f"{padded_description} | Strength: {hand['strength']:4.1f} | Pick: {hand['pick_percentage']:5.1f}% {checkbox}")

    # Calculate correlation
    correlation = np.corrcoef([hand["strength"] for hand in hand_data], [hand["pick_percentage"] for hand in hand_data])[0, 1] if len(hand_data) > 1 else 0

    print("-" * 80)
    print(f"POSITION: {position}")
    print("📊 CORRELATION ANALYSIS:")
    print(f"Hand Strength vs Pick Probability: {correlation:.3f}")
    print()

    if correlation > 0.7:
        print("✅ EXCELLENT: Strong positive correlation with hand strength!")
    elif correlation > 0.5:
        print("⚠️ MODERATE: Positive correlation but could be stronger")
    elif correlation > 0:
        print("⚠️ NEUTRAL: Weak correlation - neither good nor bad")
    else:
        print("❌ BROKEN: Negative correlation")

    # Additional analysis
    print("\n📈 DETAILED ANALYSIS:")
    # Get indices of strongest and weakest hands

    # hand_strengths = [hand["strength"] for hand in hand_data]
    # print(hand_data)
    strongest_idx = int(np.argmax([hand["strength"] for hand in hand_data]))
    weakest_idx = int(np.argmin([hand["strength"] for hand in hand_data]))
    strongest_hand_pick = hand_data[strongest_idx]["pick_percentage"]
    weakest_hand_pick = hand_data[weakest_idx]["pick_percentage"]

    print(f"Strongest hand pick rate: {strongest_hand_pick:.1f}%")
    print(f"Weakest hand pick rate: {weakest_hand_pick:.1f}%")
    print(f"Difference: {strongest_hand_pick - weakest_hand_pick:+.1f}%")
    print(f"Pick rate: {(pick_counts/len(test_hands) * 100.0):.1f}%")

    if strongest_hand_pick > weakest_hand_pick:
        print("✅ GOOD: Prefers strong hands over weak hands")
    else:
        print("❌ BAD: Prefers weak hands over strong hands")

    return correlation > 0.2

if __name__ == "__main__":
    parser = ArgumentParser(description="Test picking of trained model")
    parser.add_argument("-m", "--model", type=str, default="final_swish_ppo.pth",
                       help="Path to the trained model")
    parser.add_argument("-p", "--position", type=int, default=1,
                       help="Position of the player to test")
    parser.add_argument("-r", "--random", action="store_true",
                       help="Whether to use random hands")
    args = parser.parse_args()

    test_final_model(args.model, args.position, args.random)
