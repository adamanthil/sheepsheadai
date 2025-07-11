#!/usr/bin/env python3
"""
Test the final trained model to see if it learned proper hand strength correlations.
"""

import torch
import numpy as np
from argparse import ArgumentParser

from ppo import PPOAgent
from sheepshead import Game, Player, ACTIONS, STATE_SIZE, ACTION_IDS, TRUMP

def calculate_hand_strength(hand):
    """Calculate objective hand strength score."""
    strength = 0
    for card in hand:
        if card[0] == 'Q':  # Queens are very strong
            strength += 4
        elif card[0] == 'J':  # Jacks are strong
            strength += 2
        elif card in TRUMP:  # Other trump cards
            strength += 1
        elif card[0] == 'A':  # Aces are okay
            strength += 0.5
        elif card[0:2] == '10':  # Tens are okayish
            strength += 0.5
    return strength

def test_final_model(model_path):
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

    # Test hands from very weak to very strong
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

    pick_probs = []
    hand_strengths = []

    for hand, description in test_hands:
        # Setup game state for pick decision
        game = Game()
        player = Player(game, 1, hand)  # Player 1's turn to pick
        game.last_passed = 0  # It's player 1's turn

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

        # Calculate objective hand strength
        strength = calculate_hand_strength(hand)

        pick_probs.append(pick_percentage)
        hand_strengths.append(strength)

        print(f"{description:60} | Strength: {strength:4.1f} | Pick: {pick_percentage:5.1f}%")

    # Calculate correlation
    correlation = np.corrcoef(hand_strengths, pick_probs)[0, 1] if len(hand_strengths) > 1 else 0

    print("-" * 80)
    print("📊 CORRELATION ANALYSIS:")
    print(f"Hand Strength vs Pick Probability: {correlation:.3f}")
    print()

    if correlation > 0.5:
        print("✅ EXCELLENT: Strong positive correlation with hand strength!")
    elif correlation > 0.2:
        print("⚠️ MODERATE: Positive correlation but could be stronger")
    elif correlation > -0.1:
        print("⚠️ NEUTRAL: Weak correlation - neither good nor bad")
    else:
        print("❌ BROKEN: Negative correlation")

    # Additional analysis
    print("\n📈 DETAILED ANALYSIS:")
    strongest_hand_pick = pick_probs[-1]  # Last hand is strongest
    weakest_hand_pick = pick_probs[0]    # First hand is weakest

    print(f"Strongest hand pick rate: {strongest_hand_pick:.1f}%")
    print(f"Weakest hand pick rate: {weakest_hand_pick:.1f}%")
    print(f"Difference: {strongest_hand_pick - weakest_hand_pick:+.1f}%")

    if strongest_hand_pick > weakest_hand_pick:
        print("✅ GOOD: Prefers strong hands over weak hands")
    else:
        print("❌ BAD: Prefers weak hands over strong hands")

    return correlation > 0.2

if __name__ == "__main__":
    parser = ArgumentParser(description="Test picking of trained model")
    parser.add_argument("--model", type=str, default="final_swish_ppo.pth",
                       help="Path to the trained model")
    args = parser.parse_args()

    test_final_model(args.model)