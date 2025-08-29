#!/usr/bin/env python3

import torch
from argparse import ArgumentParser

from ppo import PPOAgent
from sheepshead import Game, Player, ACTIONS, STATE_SIZE, ACTION_IDS, PLAY_ACTIONS, colorize_card, PARTNER_BY_JD, PARTNER_BY_CALLED_ACE


parser = ArgumentParser(
    prog = "Play Sheepshead",
    description = "Interactive Sheepshead AI player and evaluator",
)
parser.add_argument(
    "-m",
    "--model",
    default="final_swish_ppo.pt",
    help="PyTorch model file. Default: final_swish_ppo.pt"
)
parser.add_argument(
    "--partner-mode",
    choices=["called-ace", "jd"],
    default="called-ace",
    help="Partner selection mode: 'called-ace' (default) or 'jd' (jack of diamonds)"
)

args = parser.parse_args()

# Convert partner mode string to constant
partner_mode = PARTNER_BY_CALLED_ACE if args.partner_mode == "called-ace" else PARTNER_BY_JD

mode_name = "Called Ace" if partner_mode == PARTNER_BY_CALLED_ACE else "Jack of Diamonds"
instructions = f"""
{'-'*40}
Welcome to the interactive Sheepshead AI player and evaluator.
Partner Mode: {mode_name}

Commands:
  play - Play a sample game
  pick - Evaluate picking engine
  h    - Help
  q    - Quit

{'-'*40}
"""

def play(agent):
    game = Game(partner_selection_mode=partner_mode)

    players = ['Dan', 'Kyle', 'Trevor', 'John', 'Andrew']
    game.print_player_hands(players)
    print(f"{'-'*40}")

    card_count = 0
    while not game.is_done():
        for i, player in enumerate(game.players):
            valid_actions = player.get_valid_action_ids()

            # print(list(map(lambda i: ACTIONS[i - 1], valid_actions)))
            while valid_actions:
                state = player.get_state_vector()
                action, _, _ = agent.act(state, valid_actions, player.position, deterministic=True)
                action_str = ACTIONS[action - 1]

                if action_str in PLAY_ACTIONS:
                    card_count += 1

                # Colorize cards in action descriptions
                if "BURY" in action_str or "PLAY" in action_str:
                    card = action_str.split()[-1]  # Get the card from "BURY QC" or "PLAY QC"
                    colored_card = colorize_card(card)
                    action_str = action_str.replace(card, colored_card)

                print(f" -- {players[i]}: {action_str}")

                # Print a new line between tricks
                if card_count == 5 or action_str in ("ALONE", "JD PARTNER"):
                    print(f"{'-'*40}")
                    card_count = 0

                player.act(action)
                valid_actions = player.get_valid_action_ids()
                # At end of trick, propagate observation frames so the LSTM sees the completed trick
                if player.game.was_trick_just_completed:
                    for seat in player.game.players:
                        agent.observe(seat.get_last_trick_state_vector(), player_id=seat.position)
                # print(list(map(lambda i: ACTIONS[i - 1], valid_actions)))

    print(f"{'-'*40}")
    print(game)


def pick_evaluator(agent):

    while True:
        position = int(input("Enter position [1-5]: "))
        hand = input("Enter hand (e.g. 'QC QS JD 10H 8C 7S'): ")
        hand = hand.split(" ")

        game = Game(partner_selection_mode=partner_mode)
        player = Player(game, position, hand)
        game.last_passed = position - 1

        state = player.get_state_vector()
        valid_actions = player.get_valid_action_ids()

        # Get action probabilities from PPO actor network
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_mask = torch.zeros(len(ACTIONS), dtype=torch.bool)
        for action in valid_actions:
            action_mask[action - 1] = True

        with torch.no_grad():
            action_probs = agent.actor(state_tensor, action_mask.unsqueeze(0))

        # Extract pick/pass probabilities
        pick_prob = action_probs[0][ACTION_IDS["PICK"] - 1].item()
        pass_prob = action_probs[0][ACTION_IDS["PASS"] - 1].item()

        if pick_prob + pass_prob > 0:
            normalized_pick = pick_prob / (pick_prob + pass_prob)
            print("Pick with %.2f%% confidence\n" % (normalized_pick * 100))
        else:
            print("No pick/pass decision available\n")

        again = input("Again? (y/n): ")
        if again != "y":
            return


if __name__ == "__main__":

    agent = PPOAgent(STATE_SIZE, len(ACTIONS), activation='swish')
    agent.load(args.model)
    param_count = sum(p.numel() for p in agent.actor.parameters())
    print(f"Loaded model: {args.model} with {param_count:,} parameters")
    print(instructions)

    while True:
        choice = input("Enter Command: ")
        if choice == "q":
            exit()
        if choice == "h":
            print(instructions)
        if choice == "play":
            print("Playing a sample game!")
            play(agent)
        if choice == "pick":
            print("Pick evaluator!")
            print("Evaluate model weight of PICK action with specific hands.")
            pick_evaluator(agent)
        print()
