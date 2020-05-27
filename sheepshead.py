import numpy as np
import random


TRUMP = [
	"QC", "QS", "QH", "QD", "JC", "JS", "JH", "JD", "AD", "10D", "KD", "9D", "8D", "7D"
]
FAIL = [
	"AC", "10C", "KC", "9C", "8C", "7C",
	"AS", "10S", "KS", "9S", "8S", "7S",
	"AH", "10H", "KH", "9H", "8H", "7H",
]
DECK = TRUMP + FAIL
DECK_IDS = {k: v + 1 for v, k in enumerate(DECK)}


ACTIONS = {
	1: "PICK",
	2: "ALONE",
}
# Add bury actions for all cards
ACTIONS.update({ (k + len(ACTIONS) + 1): f"BURY {v}" for k, v in enumerate(DECK)})

# Add under actions for all cards (called ace)
# ACTIONS.update({ (k + len(ACTIONS) + 1): f"UNDER {v}" for k, v in enumerate(DECK)})

# Add play actions for all cards
ACTIONS.update({ (k + len(ACTIONS) + 1): f"PLAY {v}" for k, v in enumerate(DECK)})


class Agent:
	def __init__(self, position, hand):
		self.position = position
		self.hand = hand
		self.picker = 0
		self.blind = []
		self.bury = []

		# Nested list of all cards played in game so far
		self.history = []

	def get_state(self):
		"""Integer vector of current game state.
		Values in order:
			[0] player position
			[1-6] card ID of each card in starting hand
			[7] position of picker
			[8-9] card ID of cards in blind (if known)
			[10-11] card ID of cards buried (if known)
			[12-41] card ID of each card played in game
		"""
		state = [self.position]
		state.extend([DECK_IDS[c] for c in self.hand])
		state.append(self.picker)
		state.extend([DECK_IDS[c] for c in self.blind] if self.blind else [0, 0])
		state.extend([DECK_IDS[c] for c in self.bury] if self.bury else [0, 0])

		state = np.array(state, dtype=np.uint8)

		history2d = np.zeros((6, 5))
		for i, hand in enumerate(self.history):
			history2d[i, :len(hand)] = np.array([DECK_IDS[c] for c in hand])
		state = np.hstack([state, history2d.flatten()])
		return state


hand = random.sample(DECK, 6)
print(hand)

agent = Agent(3, hand)
agent.get_state()
