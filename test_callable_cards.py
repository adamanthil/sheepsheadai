#!/usr/bin/env python3

import random
import re
from sheepshead import DECK, get_callable_cards, pretty_card_list

def generate_random_hand(size=8):
    """Generate a random hand of specified size from the deck."""
    return random.sample(DECK, size)

def get_display_width(text):
    """Get the display width of text, excluding ANSI color codes."""
    # Remove ANSI color codes for width calculation
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', text)
    return len(clean_text)

def test_callable_cards():
    """Test get_callable_cards function with 100 random hands."""
    print("Testing get_callable_cards() function with 100 random hands")
    print("=" * 80)
    print("Format: Hand -> Callable Cards")
    print("=" * 80)

    for i in range(1000):
        hand = generate_random_hand(8)
        callable_cards = get_callable_cards(hand)

        # Format the output for easy scanning
        hand_str = pretty_card_list(hand)
        callable_str = pretty_card_list(callable_cards) if callable_cards else "None"

        # Calculate actual display width of hand string (excluding color codes)
        hand_width = get_display_width(hand_str)
        # Use 30 as target width, but adjust based on actual display width
        padding = max(30 - hand_width, 1)

        print(f"{i+1:3d}. {hand_str}{' ' * padding} -> {callable_str}")

if __name__ == "__main__":
    test_callable_cards()
