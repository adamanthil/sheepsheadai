#!/usr/bin/env python3

import torch
from argparse import ArgumentParser

from dqn import Agent
from sheepshead import Game, ACTIONS, STATE_SIZE


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
  p - Play a sample game
  h - Help
  q - Quit

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
    print(game)

if __name__ == "__main__":

    agent = Agent(
        state_size=STATE_SIZE,
        action_size=len(ACTIONS),
        seed=0,
    )
    agent.qnetwork_local.load_state_dict(torch.load(args.model))
    print(instructions)

    while True:
        choice = input("Enter Command: ")
        if choice == "q":
            exit()
        if choice == "h":
            print(instructions)
        if choice == "p":
            print("Playing a sample game!")
            play(agent)
        print()
