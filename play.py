#!/usr/bin/env python3

import torch
from argparse import ArgumentParser
from scipy.special import softmax

from dqn import Agent
from sheepshead import Game, Player, ACTIONS, DECK, STATE_SIZE, get_experience_str


parser = ArgumentParser(
    prog = "Play Sheepshead",
    description = "Interactive Sheepshead AI player and evaluator",
)
parser.add_argument(
    "-m",
    "--model",
    default="checkpoint.pth",
    help="PyTorch model state_dict file. Default: checkpoint.pth"
)

args = parser.parse_args()

instructions = f"""
{'-'*40}
Welcome to the interactive Sheepshead AI player and evaluator.

Commands:
  play - Play a sample game
  pick - Evaluate picking engine
  h    - Help
  q    - Quit

{'-'*40}
"""

def play(agent):
    game = Game()

    players = ['Dan', 'Kyle', 'Trevor', 'John', 'Andrew']
    game.print_player_hands(players)
    print(f"{'-'*40}")

    while not game.is_done():
        for i, player in enumerate(game.players):
            valid_actions = player.get_valid_action_ids()

            # print(list(map(lambda i: ACTIONS[i - 1], valid_actions)))
            while valid_actions:
                state = player.get_state_vector()
                action = agent.act(state, valid_actions)
                print(f" -- {players[i]}: {ACTIONS[action - 1]}")
                player.act(action)
                valid_actions = player.get_valid_action_ids()
                # print(list(map(lambda i: ACTIONS[i - 1], valid_actions)))

    print(f"{'-'*40}")
    print("Player 1 experiences:")
    print(f"{'-'*40}")
    for exp in game.players[0].get_experiences():
        print(get_experience_str(exp))
    print(game)


def pick_evaluator(agent):

    while True:
        position = int(input("Enter position [1-5]: "))
        hand = input("Enter hand (e.g. 'QC QS JD 10H 8C 7S'): ")
        hand = hand.split(" ")

        game = Game()
        player = Player(game, position, hand)
        game.last_passed = position - 1

        state = player.get_state_vector()
        valid_actions = player.get_valid_action_ids()

        weights = agent.get_action_weights(state, valid_actions)
        normalized = softmax([weights[0][0], weights[0][1]])
        print("Pick with %.2f confidence\n" % (normalized[0] * 100))

        again = input("Again? (y/n): ")
        if again != "y":
            return


if __name__ == "__main__":

    agent = Agent(
        state_size=STATE_SIZE,
        action_size=len(ACTIONS),
        # seed=0,
    )
    agent.qnetwork_local.load_state_dict(torch.load(args.model))
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
