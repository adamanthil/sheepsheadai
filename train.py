import numpy as np
from dqn import Agent
from sheepshead import Game, ACTIONS


if __name__ == "__main__":
	num_games = 10000000
	load_checkpoint = False
	best_score = -12
	state_size = 45

	agent = Agent(
		gamma=0.99,
		epsilon=1.0,
		alpha=0.001,
		input_dims=(state_size,),
		n_actions=len(ACTIONS),
		mem_size=25000,
		eps_min=0.02,
		batch_size=128,
		replace=1000,
		eps_dec=1e-6,
	)

	if load_checkpoint:
		agent.load_models()

	picker_scores = []
	n_steps = 0

	for e in range(num_games):
		game = Game()
		picker_score = 0
		while not game.is_done():

			states = np.zeros((5, state_size), dtype=np.int8)
			states_ = np.zeros((5, state_size), dtype=np.int8)
			actions = np.zeros(5, dtype=np.int8)

			for i, player in enumerate(game.players):
				valid_actions = player.get_valid_action_ids()
				while valid_actions:
					# Save last action for this player if necessary
					if actions[i]:
						state_ = player.get_state_vector()
						reward = player.get_score()
						agent.store_transition(states[i], actions[i], reward, state_, int(game.is_done()))
						states[i] = state_
						n_steps += 1

					states[i] = player.get_state_vector()
					if len(states[i]) == 1:
						# DEBUG
						print(player.position, player.picker, player.partner, game.is_done(), states[i])
					actions[i] = agent.choose_action(states[i], valid_actions)
					player.act(actions[i])

					valid_actions = player.get_valid_action_ids()

					# new state for each agent should be the state when they next have actions
					# save it immediately if we still have actions
					if valid_actions:
						state_ = player.get_state_vector()
						reward = player.get_score()
						agent.store_transition(states[i], actions[i], reward, state_, int(game.is_done()))
						states[i] = state_
						n_steps += 1

		# Save final state and scores for each player this game
		for i, player in enumerate(game.players):
			state_ = player.get_state_vector()
			reward = player.get_score()
			if player.is_picker:
				picker_score = reward
			agent.store_transition(states[i], actions[i], reward, state_, int(game.is_done()))
			n_steps += 1

		agent.learn()

		picker_scores.append(picker_score)

		if e % 100 == 0:
			avg_score = np.mean(picker_scores[-1000:])
			batch_score = np.mean(picker_scores[-100:])
			print(
				f"episode {e}",
				"batch score avg 100 %.2f" % batch_score,
				"long score avg 1000 %.2f" % avg_score,
				"epsilon score %.2f" % agent.epsilon,
				f"steps {n_steps}",
			)

			if avg_score > best_score:
				agent.save_models()
				print("avg picker score %.2f better than best score %.2f" % (avg_score, best_score))
				best_score = avg_score
