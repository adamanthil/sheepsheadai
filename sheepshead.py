import numpy as np
import random
from collections import deque
from utils import PlayerExperience


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

STATE_SIZE = 133


def get_card_suit(card):
	return "T" if card in TRUMP else card[-1]


def get_trick_winner(trick, suit):
	power_list = []
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

def get_deck_vector(cards):
	"""Returns a vector of 32 booleans in DECK order
	indicating whether the passed list contains each card."""

	vector = np.zeros(len(DECK), dtype=np.uint8)
	for card in cards:
		vector[DECK_IDS[card] - 1] = 1

	return vector

def get_cards_from_vector(vector):
	"""Reverses deck vector to retrieve list of human readable cards."""
	return [DECK[i] for i, exists in enumerate(vector) if exists]


def get_hand_pick_reward(hand):
	"""Uses a simple heuristic to return an integer reward of whether a hand is pickable."""
	num_queens = 0
	num_jacks = 0
	num_trump = 0

	for card in hand:
		if card[0] == "Q":
			num_queens += 1
		elif card[0] == "J":
			num_jacks += 1
		elif card[-1] == "D":
			num_trump += 1

	return num_queens * 3 + num_jacks * 2 + num_trump - 7


def get_state_str(state):
	"""Return a human readable state string.
	Values in order:
		[0] player position
		[1] last position to pass on picking blind
		[2] position of picker
		[3] position of partner (if known)
		[4] alone called (bool)
		[5] play has started (bool)
		[6] current trick
		[7-38] 	boolean values for cards in current hand in DECK order
				(e.g. if hand has QC, index 6 will have value 1 otherwise 0)
		[39-70]	boolean values for cards in blind (if known)
		[71-102] boolean values for cards buried (if known)
		[103-132] card ID of each card played in game in reverse trick order starting with current trick
	"""
	out = ""
	out += f"Player #: {int(state[0])}\n"

	if (state[5]):
		out += f"Game Phase: Play\n"
		out += f"Picker: {int(state[2])}\n"
		out += f"Partner: {int(state[3])}\n"
		out += f"Alone: {int(state[4])}\n"

		state_start = 103
		tricks = ""
		trick = ""
		trick_number = int(state[6]) + 1
		for i in range(30):
			if i % 5 == 0 and i != 0:
				tricks += f"Trick {trick_number}: {trick}\n"
				trick = ""
				trick_number -= 1
				if trick_number < 1:
					break

			card_index = state_start + i
			card_id = int(state[card_index])
			if card_id:
				trick += f"{DECK[card_id - 1]} "
			else:
				trick += "__ "


		# Include first trick if needed (end of the loop)
		if trick:
			tricks += f"Trick 1: {trick}\n"

		out += tricks

	else:
		out += f"Game Phase: Picking\n"
		blind = " ".join(get_cards_from_vector(state[39:71]))
		if blind:
			out += f"Blind: {blind}\n"

	hand = " ".join(get_cards_from_vector(state[7:39]))

	out += f"Hand: {hand}\n"

	return out



def get_experience_str(experience):
	"""Return a human readable string from an experience tuple.
	"""
	out = "--------- Experience ---------\n"
	out += "Starting state:\n"
	out += get_state_str(experience.state)
	out += "-----------------------------\n"
	out += f"Action: {ACTIONS[experience.action - 1]}\n"
	out += "-----------------------------\n"
	out += "Ending state:\n"
	out += get_state_str(experience.next_state)
	out += "-----------------------------\n"
	out += f"Reward: {experience.reward}\n"
	out += "-----------------------------\n\n"
	return out


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
		self.points_taken = [0, 0, 0, 0, 0] # Sum of all points taken for each player
		# Nested list of all cards played in game so far
		self.history = [
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
			["", "", "", "", ""],
		]
		self.trick_points = [0, 0, 0, 0, 0, 0] # Points of each trick
		self.trick_winners = [0, 0, 0, 0, 0, 0] # Player ID of each trick winner
		self.leaders = [0, 0, 0, 0, 0, 0] # Player ID of each trick leader

		# Internal state variables
		self.last_player = 0 # Position of last player to play a card
		self.leader = 0 # Position of leader this trick
		self.cards_played = 0 # Number of cards played this trick
		self.current_suit = ""
		self.current_trick = 0

		# Setup players and deal
		for i in range(5):
			hand = self.deck[i * 6: i * 6 + 6]
			self.players.append(Player(self, i + 1, hand))

	def play_random(self):
		print("Playing Sheepshead Randomly!")
		self.print_player_hands()

		while not self.is_done():
			for player in self.players:
				actions = player.get_valid_action_ids()
				# print(f"PLAYER {player.position}")
				# print([ACTION_LOOKUP[a] for a in actions])
				while actions:
					action = random.sample(list(actions), 1)[0]

					print(f" -- Player {player.position}: {ACTION_LOOKUP[action]}")
					player.act(action)

					actions = player.get_valid_action_ids()

		print("Bury: ", self.bury)
		print("Points taken: ", self.points_taken)
		print(f"Game done! Picker score: {self.get_final_picker_points()}  Defenders score: {self.get_final_defender_points()}")
		scores = [p.get_score() for p in self.players]
		print(f"Scores: {scores}")

	def print_player_hands(self, player_names=["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]):
		for p in self.players:
			print(f"{player_names[p.position - 1].ljust(8)}: ", p.hand)
		print("Blind: ", self.blind)

	def __str__(self):
		out = ""
		out += f"Picker: {self.picker} - Partner: {self.partner}\n"
		out += f"Picking hand: {self.get_picker().initial_hand}\n"
		out += f"Blind: {self.blind}\n"
		out += f"Bury: {self.bury}\n"
		out += f"Points taken: {self.points_taken}\n"
		out += f"Picker score: {self.get_final_picker_points()}  Defenders score: {self.get_final_defender_points()}\n"
		scores = [p.get_score() for p in self.players]
		out += (f"Scores: {scores}\n")

		return out

	def is_done(self):
		return self.last_passed == 5 or "" not in self.history[5]

	def get_picker(self):
		return self.players[self.picker - 1]

	def get_final_picker_points(self):
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

	def get_final_defender_points(self):
		if self.is_done():
			points = 0
			for i, taken in enumerate(self.points_taken):
				if i + 1 != self.picker and i + 1 != self.partner:
					points += taken

			# Add bury if picker got no points
			if not self.get_final_picker_points():
				points += get_trick_points(self.bury)
			return points
		return False


class Player:
	def __init__(self, game, position, hand):
		self.game = game
		self.position = position
		self.initial_hand = hand
		self.hand = hand[:]
		self.start_states = deque()
		self.actions = deque()
		self.rewards = deque()

	@property
	def picker(self):
		return self.game.picker

	@property
	def partner(self):
		return self.game.partner

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
			[1] last position to pass on picking blind
			[2] position of picker
			[3] position of partner (if known)
			[4] alone called (bool)
			[5] play has started (bool)
			[6] current trick
			[7-38] 	boolean values for cards in current hand in DECK order
					(e.g. if hand has QC, index 6 will have value 1 otherwise 0)
			[39-70]	boolean values for cards in blind (if known)
			[71-102] boolean values for cards buried (if known)
			[103-132] card ID of each card played in game in reverse trick order starting with current trick
					Each trick ordered clockwise starting with this player:
					Player 1:
						1 2 3 4 5
					Player 2:
						2 3 4 5 1
		"""

		state = [self.position]
		state.append(self.last_passed)
		state.append(self.picker)
		partner = self.partner if self.partner else "JD" in self.hand
		state.append(partner)
		state.append(self.alone_called)
		state.append(self.play_started)
		state.append(self.current_trick)
		state.extend(get_deck_vector(self.hand))
		state.extend(get_deck_vector(self.blind if self.is_picker else []))
		state.extend(get_deck_vector(self.bury if self.is_picker else []))

		state = np.array(state, dtype=np.uint8)
		history = np.zeros((6, 5))
		i = 0
		if self.play_started:
			for t in reversed(range(min(self.current_trick, 5) + 1)):
				trick = self.game.history[t]
				c = self.position - self.game.leaders[t]
				for j in range(5):
					card = DECK_IDS[trick[c]] if trick[c] else 0
					history[i][j] = card

					c = 0 if c == 4 else c + 1

				history[i] = np.array([DECK_IDS[c] if c else 0 for c in trick])
				i += 1
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

		self.start_states.append(self.get_state_vector())
		self.actions.append(action_id)

		action = ACTION_LOOKUP[action_id]

		if action == "PICK":
			self.game.picker = self.position
			reward = get_hand_pick_reward(self.hand)
			self.hand.extend(self.game.blind)
			# self.rewards.append(0)
			self.rewards.append(reward)

		if action == "PASS":
			self.game.last_passed = self.position
			self.rewards.append(0)

		if "BURY" in action:
			card = action[5:]
			self.game.bury.append(card)
			self.hand.remove(card)
			if get_card_suit(card) == "T":
				self.rewards.append(-1)
			else:
				self.rewards.append(get_card_points(card))

		if action == "ALONE":
			self.game.alone_called = True
			self.game.partner = self.position

		if action in ("ALONE", "JD PARTNER"):
			self.game.play_started = True
			self.game.leader = 1
			self.game.leaders[0] = 1
			self.rewards.append(0)

		if "PLAY" in action:
			card = action[5:]

			# Set suit lead if we are the first to play this trick
			if self.game.leader == self.game.last_player + 1:
				self.game.current_suit = get_card_suit(card)

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
				winner = get_trick_winner(trick, self.game.current_suit)
				winner_index = winner - 1
				trick_points = get_trick_points(trick)
				self.game.trick_points[self.current_trick] = trick_points
				self.game.trick_winners[self.current_trick] = winner
				self.game.points_taken[winner_index] += trick_points

				if self.current_trick < 5:
					self.game.leaders[self.current_trick + 1] = winner
				# print("Trick points: %i" % get_trick_points(trick))
				# print("Winner %i" % get_trick_winner(trick, self.game.current_suit))

				# Add point rewards for each player after trick ends
				for i, player in enumerate(self.game.players):
					if (
						# Player took trick themselves
						i == winner_index

						# Picking team and partner took trick
						or player.is_picker and self.game.partner == winner
						or player.is_partner and self.game.picker == winner

						# Defending team and partners took trick (must know partner)
						or self.game.partner and not player.is_picker and not player.is_partner
						and winner != self.game.picker and winner != self.game.picker
					):
						player.rewards.append(trick_points)
						# print(f"AWARD for {i}: {trick_points}")
					else:
						# print("NO REWARD THIS TRICK")
						player.rewards.append(0)

				# Next trick must start with winner
				self.game.leader = winner
				self.game.last_player = winner - 1
				self.game.current_suit = ""
				self.game.cards_played = 0

				if self.game.current_trick < 5:
					self.game.current_trick += 1

		return True

	def get_score(self):
		if self.game.is_done() and self.game.picker:
			picker_points = self.game.get_final_picker_points()
			defender_points = self.game.get_final_defender_points()
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

	def get_experiences(self):
		if self.game.is_done():
			experiences = deque()
			end_state = self.get_state_vector()
			done = 1

			# Generates PlayerExperience tuples in reverse order
			# End state for each agent is the state when they next have actions,
			# Which corresponds to the previous start state
			while self.start_states:
				start_state = self.start_states.pop()
				action = self.actions.pop()
				reward = self.rewards.pop()
				if done:
					score = self.get_score()
					reward = score * 100

					# For the picker, use discounted final score
					# to update picking reward
					if self.is_picker:
						picking_reward = self.rewards.popleft()
						picking_reward += score * 50
						self.rewards.appendleft(picking_reward)

				if ACTION_LOOKUP[action] == "ALONE":
					reward += score * 50

				experiences.appendleft(PlayerExperience(
					start_state,
					action,
					reward,
					end_state,
					done,
				))

				end_state = start_state
				done = 0

			return experiences

		raise Exception("Cannot get experiences when game is incomplete.")


if __name__ == '__main__':
	game = Game()
	game.play_random()
