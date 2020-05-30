import numpy as np
from dqn import Agent
from sheepshead import Game, ACTIONS

if __name__ == "__main__":
	num_games = 1000
	load_checkpoint = False
	best_score = -12

	agent = Agent(
		gamma=0.99,
		epsilon=1.0,
		alpha=0.001,
		input_dims=(44),
		n_actions=len(ACTIONS),
		mem_size=25000,
		eps_min=0.02,
		batch_size=32,
		replace=1000,
		eps_dec=1e-5,
	)

	if load_checkpoint:
		agent.load_models()

	scores, eps_history = [], []
	n_steps = 0

	for i in range(num_games):
		score = 0
		# state = Game() - player ???
		# state = env.reset()
		game = Game()
		done = False
		while not done:

			for player in game.players:
				actions = player.get_valid_action_ids()

			# Fix for sheepshead
			action = agent.choose_action(state)
			state_, reward, done, info = env.step(action)

			n_steps += 1
			score += reward
			if not load_checkpoint:
				agent.store_transition(state, action, reward, state_, int(done))
				agent.learn()
			else:
				# Fix to step through
				env.render()

			state = state_
		scores.append(score)

		avg_score = np.mean(scores[-100:])
		print(
			"episode ", i,
			"score ", score,
			"average score %.2f" % avg_score,
			"epsilon score %.2f" % agent.epsilon,
			"steps", n_steps,
		)

		if avg_score > best_score:
			agent.save_models()
			print("avg score %.2f better than best score %.2f" % (avg_score, best_score))
			best_score = avg_score

		eps_history.append(agent.epsilon)

	# x = [i + 1 for i in range(num_games)]
	# plotLearning(x, scores, eps_history, filename)
