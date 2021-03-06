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

ACTIONS = [
	"PICK",
	"PASS",
	"ALONE",
	# Opposite action from calling ALONE. Standard Jack of Diamonds partner.
	# Avoids lacking an action for state transition.
	"JD PARTNER",
]
# Add bury actions for all cards
BURY_ACTIONS = [f"BURY {c}" for c in DECK]
ACTIONS.extend(BURY_ACTIONS)

# Add play actions for all cards
PLAY_ACTIONS = [f"PLAY {c}" for c in DECK]
ACTIONS.extend(PLAY_ACTIONS)

# Define dics for quick action lookup
ACTION_LOOKUP = { i + 1: a for i, a in enumerate(ACTIONS)}
ACTION_IDS = { a: i + 1 for i, a in enumerate(ACTIONS)}

# Define deck variables for quick lookup
DECK_IDS = {k: v + 1 for v, k in enumerate(DECK)}

TRUMP_POWER = {k: len(TRUMP) - v for v, k in enumerate(TRUMP)}
FAIL_POWER = {k: len(FAIL) - v for v, k in enumerate(FAIL)}


def get_card_suit(card):
	return "T" if card in TRUMP else card[-1]


def get_trick_winner(trick, leader):
	power_list = []
	suit = get_card_suit(trick[leader - 1])
	for card in trick:
		try:
			power = 100 * TRUMP_POWER[card]
		except KeyError:
			power = FAIL_POWER[card] if get_card_suit(card) == suit else 0
		power_list.append(power)
	return power_list.index(max(power_list)) + 1


def get_card_points(card):
	if "A" in card:
		return 11
	if "10" in card:
		return 10
	if "K" in card:
		return 4
	if "Q" in card:
		return 3
	if "J" in card:
		return 2
	return 0


def filter_by_suit(hand, suit):
	return [c for c in hand if get_card_suit(c) == suit]


def get_trick_points(trick):
	points = 0
	for card in trick:
		points += get_card_points(card)
	return points


class Game:
	def __init__(self):
		self.deck = DECK[:]
		random.shuffle(self.deck)

		self.last_passed = 0
		self.picker = 0
		self.partner = 0
		self.blind = self.deck[30:]
		self.bury = []
		self.alone_called = False
		self.play_started = False
		self.players = []
		self.points_taken = [0, 0, 0, 0, 0]
		# Nested list of all cards played in game so far
		self.history = [
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
		]

		# Internal state variables
		self.last_player = 0 # Position of last player to play a card
		self.leader = 0 # Position of leader this hand
		self.cards_played = 0 # Number of cards played this trick
		self.current_suit = ""
		self.current_trick = 0

		# Setup players and deal
		for i in range(5):
			hand = self.deck[i * 6: i * 6 + 6]
			self.players.append(Player(self, i + 1, hand))

	def play_random(self):
		print("Playing Sheepshead Randomly!")
		for p in self.players:
			print(f"Player {p.position}: ", p.hand)
		print("Blind: ", self.blind)

		while not self.is_done():
			for player in self.players:
				actions = player.get_valid_action_ids()
				# print(f"PLAYER {player.position}")
				# print([ACTION_LOOKUP[a] for a in actions])
				while actions:
					action = random.sample(actions, 1)[0]

					print(f" -- Player {player.position}: {ACTION_LOOKUP[action]}")
					player.act(action)

					actions = player.get_valid_action_ids()

		print("Bury: ", self.bury)
		print("Points taken: ", self.points_taken)
		print(f"Game done! Picker score: {self.get_picker_points()}  Defenders score: {self.get_defender_points()}")
		scores = [p.get_score() for p in self.players]
		print(f"Scores: {scores}")

	def is_done(self):
		return self.last_passed == 5 or "" not in self.history[5]

	def get_picker_points(self):
		if self.is_done():
			points = self.points_taken[self.picker - 1]

			# Add partner points
			if not self.alone_called and self.picker != self.partner:
				points += self.points_taken[self.partner - 1]

			# Add bury
			if points:
				points += get_trick_points(self.bury)

			return points
		return False

	def get_defender_points(self):
		if self.is_done():
			points = 0
			for i, taken in enumerate(self.points_taken):
				if i + 1 != self.picker and i + 1 != self.partner:
					points += taken

			# Add bury if picker got no points
			if not self.get_picker_points():
				points += get_trick_points(self.bury)
			return points
		return False


class Player:
	def __init__(self, game, position, hand):
		self.game = game
		self.position = position
		self.initial_hand = hand
		self.hand = hand[:]

	@property
	def picker(self):
		return self.game.picker

	@property
	def last_passed(self):
		return self.game.last_passed

	@property
	def current_trick(self):
		return self.game.current_trick

	@property
	def blind(self):
		return self.game.blind

	@property
	def bury(self):
		return self.game.bury

	@property
	def alone_called(self):
		return self.game.alone_called

	@property
	def play_started(self):
		return self.game.play_started

	@property
	def last_player(self):
		return self.game.last_player

	@property
	def is_picker(self):
		return self.picker == self.position

	@property
	def is_partner(self):
		return self.game.partner == self.position

	def get_state_vector(self):
		"""Integer vector of current game state.
		Values in order:
			[0] player position
			[1-6] card ID of each card in hand
			[7] last position to pass on picking blind
			[8] position of picker
			[9] alone called (bool)
			[10] play has started (bool)
			[11-12] card ID of cards in blind (if known)
			[13-14] card ID of cards buried (if known)
			[15-44] card ID of each card played in game
		"""

		state = [self.position]
		state.extend(([DECK_IDS[c] for c in self.hand] + [0] * 6)[:6])
		state.append(self.last_passed)
		state.append(self.picker)
		state.append(self.alone_called)
		state.append(self.play_started)

		blind = (([DECK_IDS[c] for c in self.blind] if self.is_picker else []) + [0] * 2)[:2]
		bury = (([DECK_IDS[c] for c in self.bury] if self.is_picker else []) + [0] * 2)[:2]
		state.extend(blind)
		state.extend(bury)

		state = np.array(state, dtype=np.uint8)
		history = np.zeros((6, 5))
		for i, trick in enumerate(self.game.history):
		        history[i] = np.array([DECK_IDS[c] if c else 0 for c in trick])
		state = np.hstack([state, history.flatten()])
		return state

	def get_valid_actions(self):
		"""Get set of valid actions."""

		# No one yet picked, and not our turn
		if not self.picker:
			if self.last_passed != self.position - 1:
				return set()
			return set(["PICK", "PASS"])

		# We picked.
		if self.is_picker and not self.play_started:
			# Need to bury.
			if len(self.bury) != 2:
				return set([f"BURY {c}" for c in set(self.hand)])

			# Call ALONE or JD PARTNER to begin hand
			if self.bury:
				return set(["ALONE", "JD PARTNER"])

		# Exclude actions when waiting on bury
		if len(self.bury) != 2:
			return set()

		# Exclude actions when not our turn
		if self.play_started and self.last_player != self.position - 1:
			return set()

		actions = set()

		# Determine which cards are valid to play
		if self.play_started and self.last_player == self.position - 1:
			if not self.game.current_suit:
				# Entire hand is valid at start of trick
				actions.update([f"PLAY {c}" for c in self.hand])
			else:
				cards_in_suit = filter_by_suit(self.hand, self.game.current_suit)
				if cards_in_suit:
					actions.update([f"PLAY {c}" for c in cards_in_suit])
				else:
					actions.update([f"PLAY {c}" for c in self.hand])

		return actions

	def get_valid_action_ids(self):
		"""Get set of valid action IDs."""
		return set(ACTION_IDS[a] for a in self.get_valid_actions())

	def act(self, action_id):
		if action_id not in self.get_valid_action_ids():
			return False

		action = ACTION_LOOKUP[action_id]

		if action == "PICK":
			self.game.picker = self.position
			self.hand.extend(self.game.blind)

		if action == "PASS":
			self.game.last_passed = self.position

		if "BURY" in action:
			card = action[5:]
			self.game.bury.append(card)
			self.hand.remove(card)

		if action == "ALONE":
			self.game.alone_called = True
			self.game.partner = self.position

		if action in ("ALONE", "JD PARTNER"):
			self.game.play_started = True

		if "PLAY" in action:
			card = action[5:]

			# Set leader and suit if we are the first to play this trick
			if not self.game.last_player:
				self.game.current_suit = get_card_suit(card)
				self.game.leader = self.position

			self.game.history[self.current_trick][self.position - 1] = card
			self.hand.remove(card)
			self.game.last_player = self.position
			self.game.cards_played += 1

			if card == "JD" and not self.game.alone_called:
				self.game.partner = self.position

			if self.game.last_player == 5:
				self.game.last_player = 0

			if self.game.cards_played == 5:
				# Handle Jack of Diamonds in bury on final play
				if self.current_trick == 5 and "JD" in self.bury:
					self.game.partner = self.game.picker

				trick = self.game.history[self.current_trick]
				winner = get_trick_winner(trick, self.game.leader)
				self.game.points_taken[winner - 1] += get_trick_points(trick)
				# print("Trick points: %i" % get_trick_points(trick))
				# print("Winner %i" % get_trick_winner(trick, self.game.leader))

				# Next trick must start with winner
				self.game.last_player = winner - 1
				self.game.current_trick += 1
				self.game.current_suit = ""
				self.game.leader = 0
				self.game.cards_played = 0

		return True

	def get_score(self):
		if self.game.is_done() and self.game.picker:
			picker_points = self.game.get_picker_points()
			defender_points = self.game.get_defender_points()
			if picker_points + defender_points != 120:
				raise Exception(f"Points don't add up to 120! Picker: {picker_points} Defenders: {defender_points}")
			multiplier = 1
			if picker_points == 120:
				multiplier = 3
			elif picker_points > 90:
				multiplier = 2
			elif defender_points == 120:
				multiplier = -3
			elif defender_points >= 90:
				multiplier = -2
			elif defender_points >= 60:
				multiplier = -1

			if self.is_picker and self.is_partner:
				return 4 * multiplier
			if self.is_picker:
				return 2 * multiplier
			if self.is_partner:
				return multiplier
			return -1 * multiplier
		return 0


if __name__ == '__main__':
	game = Game()
	game.play_random()
