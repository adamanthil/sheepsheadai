import random
import torch
import numpy as np
from collections import deque

from dqn import Agent
from sheepshead import Game, ACTIONS, STATE_SIZE, get_experience_str


if __name__ == "__main__":
    num_games = int(1e7)
    load_checkpoint = False
    best_score = -12

    # eps_start=1.0
    eps_start=0.8
    eps_end=0.1
    eps_decay=0.9999995

    agents = []
    for i in range(5):
        agents.append(Agent(
            state_size=STATE_SIZE,
            action_size=len(ACTIONS),
            seed=i
        ))

    if load_checkpoint:
        print('Loading network from checkpoint.')
        for i in range(5):
            agents[i].qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

    picker_scores = []
    rewards = []
    n_steps = 0

    eps = eps_start
    for e in range(num_games):
        game = Game()
        picker_score = 0
        while not game.is_done():

            for i, player in enumerate(game.players):
                valid_actions = player.get_valid_action_ids()
                while valid_actions:
                    state = player.get_state_vector()

                    # Stabilize learning by only randomizing agent in 0 position
                    epsilon = eps if i == 0 else 0

                    action = agents[i].act(state, valid_actions, epsilon)
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
                agents[i].step(*exp)
                rewards.append(exp.reward)
                n_steps += 1

        picker_scores.append(picker_score)

        # Shuffle agents to generalize position
        if e % 10 == 0:
            random.shuffle(agents)

        eps = max(eps_end, eps_decay*eps) # decrease epsilon

        if e % 100 == 0:
            avg_score = np.mean(picker_scores[-1000:])
            batch_score = np.mean(picker_scores[-100:])
            avg_reward = np.mean(rewards[-10000:])

            # Clean up memory
            del picker_scores[:(len(picker_scores) - 1000)]
            del picker_scores[:(len(picker_scores) - 100)]
            del picker_scores[:(len(picker_scores) - 10000)]

            print(
                f"episode {e}",
                "batch score avg 100 %.2f" % batch_score,
                "long score avg 1000 %.2f" % avg_score,
                "best score %.2f" % best_score,
                "avg_reward %.2f" % avg_reward,
                "epsilon %.2f" % eps,
                f"steps {n_steps}",
            )

            if avg_score > best_score:
                print("avg picker score %.2f better than best score %.2f" % (avg_score, best_score))
                best_score = avg_score

                print('Saving network...')
                torch.save(agents[0].qnetwork_local.state_dict(), 'checkpoint.pth')

        if e and e % 200000 == 0:
            print("Saving 200k snapshot")
            torch.save(agents[0].qnetwork_local.state_dict(), f'checkpoint-{int(e/200000)}.pth')
