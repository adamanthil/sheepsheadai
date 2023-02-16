import random
import torch
import numpy as np
from collections import deque

from dqn import Agent
from sheepshead import Game, ACTIONS


if __name__ == "__main__":
    num_games = int(1e7)
    load_checkpoint = False
    best_score = -12
    state_size = 45

    eps_start=1.0
    eps_end=0.02
    eps_decay=0.99995

    agent = Agent(
        state_size=state_size,
        action_size=len(ACTIONS),
        seed=0
    )

    if load_checkpoint:
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    picker_scores = []
    n_steps = 0

    eps = eps_start
    for e in range(num_games):
        game = Game()
        picker_score = 0
        while not game.is_done():

            # Vectors to store state, action, new state data for all 5 players
            states = np.zeros((5, state_size), dtype=np.int8)
            states_ = np.zeros((5, state_size), dtype=np.int8)
            actions = np.zeros(5, dtype=np.int8)

            for player in game.players:
                valid_actions = player.get_valid_action_ids()
                while valid_actions:
                    state = player.get_state_vector()
                    action = agent.act(states, valid_actions, eps)
                    player.act(action)
                    valid_actions = player.get_valid_action_ids()


        # Save final state and scores for each player this game
        for i, player in enumerate(game.players):

            # DEBUG
            # print(player.position, player.is_picker, player.is_partner, game.is_done(), player.get_state_vector())

            if player.is_picker:
                picker_score = player.get_score()

            experiences = player.get_experiences()
            for exp in experiences:
                agent.step(*exp)
                n_steps += 1

        picker_scores.append(picker_score)

        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        if e % 100 == 0:
            avg_score = np.mean(picker_scores[-1000:])
            batch_score = np.mean(picker_scores[-100:])
            print(
                f"episode {e}",
                "batch score avg 100 %.2f" % batch_score,
                "long score avg 1000 %.2f" % avg_score,
                "epsilon %.2f" % eps,
                f"steps {n_steps}",
            )

            if avg_score > best_score:
                print("avg picker score %.2f better than best score %.2f" % (avg_score, best_score))
                best_score = avg_score

                print('Saving network...')
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
