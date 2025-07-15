import numpy as np
import random
from collections import deque


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

# Observation layout (minimal but complete):
#  Header (8)                         : position, last_passed, picker, partner, alone, is_leaster, play_started, current_trick
#  Hand                               : 32 one-hot
#  Blind (picker only, else zeros)    : 32 one-hot
#  Bury  (picker only, else zeros)    : 32 one-hot
#  Current trick (seat-attributed)    : 5 × (33 card one-hot + 2 role bits) = 175
#  Total                              : 8 + 32 + 32 + 32 + 175 = 279
STATE_SIZE = 279


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


def get_state_str(state):
    """Return a human readable state string.
    Values in order:
        [0] player position
        [1] last position to pass on picking blind (or 6 if leaster mode)
        [2] position of picker
        [3] position of partner (if known)
        [4] alone called (bool)
        [5] play has started (bool)
        [6] current trick
        [7-38]     boolean values for cards in current hand in DECK order
                (e.g. if hand has QC, index 6 will have value 1 otherwise 0)
        [39-70]    boolean values for cards in blind (if known)
        [71-102] boolean values for cards buried (if known)
        [103-132] card ID of each card played in game in reverse trick order starting with current trick
    """
    out = ""
    out += f"Player #: {int(state[0])}\n"

    # Check if this is leaster mode
    is_leaster = int(state[1]) == 6

    if (state[5]):
        if is_leaster:
            out += "Game Phase: Play (Leaster)\n"
        else:
            out += "Game Phase: Play\n"
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
        if is_leaster:
            out += "Game Phase: Leaster (all players passed)\n"
        else:
            out += "Game Phase: Picking\n"
        blind = " ".join(get_cards_from_vector(state[39:71]))
        if blind:
            out += f"Blind: {blind}\n"

    hand = " ".join(get_cards_from_vector(state[7:39]))

    out += f"Hand: {hand}\n"

    return out


def colorize_card(card):
    if card in TRUMP:
        # Trump cards in yellow
        return f"\033[93m{card}\033[0m"
    return card


def pretty_card_list(card_list):
    return " ".join([colorize_card(c) for c in card_list])


def get_monte_carlo_pick_score(hand):
    """Returns the expected score for a player picking from a hand
    using a monte-carlo simulation of 30 random games."""

    scores = []
    for _ in range(30):
        game = Game(picking_hand=hand, double_on_the_bump=False)
        game.play_random(verbose=False)
        scores.append(game.get_picker().get_score())
    return sum(scores) / len(scores)


class Game:
    def __init__(self, double_on_the_bump=True, picking_player=None, picking_hand=None):
        if picking_hand:
            # Remove picking_hand cards from DECK to form the deck
            self.deck = [card for card in DECK if card not in picking_hand]
            if picking_player is None:
                picking_player = random.randint(1, 5)
        else:
            self.deck = DECK[:]
        random.shuffle(self.deck)

        self.last_passed = picking_player - 1 if picking_player else 0
        self.picker = picking_player if picking_player else 0
        self.partner = 0
        self.blind = self.deck[(len(self.deck) - 2):]
        self.bury = []
        self.alone_called = False
        self.play_started = False
        self.is_leaster = False
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
        self.is_double_on_the_bump = double_on_the_bump

        # Internal state variables
        self.last_player = 0 # Position of last player to play a card
        self.leader = 0 # Position of leader this trick
        self.cards_played = 0 # Number of cards played this trick
        self.current_suit = ""
        self.current_trick = 0
        self.was_trick_just_completed = False

        # Setup players and deal
        if picking_hand is not None:
            # Remove blind from deck
            deal_deck = self.deck[:]
            deal_deck = [card for card in deal_deck if card not in self.blind]
            for i in range(5):
                pos = i + 1
                if pos == picking_player:
                    hand = picking_hand[:]
                    hand.extend(self.blind)
                else:
                    hand = random.sample(deal_deck, 6)
                    deal_deck = [card for card in deal_deck if card not in hand]
                self.players.append(Player(self, pos, hand))
        else:
            # Default: deal out first 30 cards in deck in order (last 2 are blind)
            for i in range(5):
                hand = self.deck[i * 6: i * 6 + 6]
                self.players.append(Player(self, i + 1, hand))

    def play_random(self, verbose=True):
        if verbose:
            print("Playing Sheepshead Randomly!")
            self.print_player_hands()

        while not self.is_done():
            for player in self.players:
                actions = player.get_valid_action_ids()
                # print(f"PLAYER {player.position}")
                # print([ACTION_LOOKUP[a] for a in actions])
                while actions:
                    action = random.sample(list(actions), 1)[0]

                    if verbose:
                        print(f" -- Player {player.position}: {ACTION_LOOKUP[action]}")
                    player.act(action)

                    actions = player.get_valid_action_ids()

        if self.is_leaster:
            if verbose:
                print("Leaster - all players passed!")
                print("Points taken: ", self.points_taken)
            winner = self.get_leaster_winner()
            if verbose:
                print(f"Leaster winner: Player {winner}")
        else:
            if verbose:
                print("Bury: ", self.bury)
                print("Points taken: ", self.points_taken)
                print(f"Game done! Picker score: {self.get_final_picker_points()}  Defenders score: {self.get_final_defender_points()}")

        if verbose:
            scores = [p.get_score() for p in self.players]
            print(f"Scores: {scores}")

    def print_player_hands(self, player_names=["Player 1", "Player 2", "Player 3", "Player 4", "Player 5"]):
        for p in self.players:
            print(f"{player_names[p.position - 1].ljust(8)}: {pretty_card_list(p.hand)}")
        print(f"Blind: {pretty_card_list(self.blind)}")

    def __str__(self):
        out = ""
        if self.is_leaster:
            out += "Leaster\n"
            out += f"Points taken: {self.points_taken}\n"
            if self.is_done():
                winner = self.get_leaster_winner()
                out += f"Leaster winner: Player {winner}\n"
        else:
            out += f"Picker: {self.picker} - Partner: {self.partner}\n"
            if self.picker:
                out += f"Picking hand: {pretty_card_list(self.get_picker().initial_hand)}\n"
            out += f"Blind: {pretty_card_list(self.blind)}\n"
            out += f"Bury: {pretty_card_list(self.bury)}\n"
            out += f"Points taken: {self.points_taken}\n"
            if self.is_done():
                out += f"Picker score: {self.get_final_picker_points()}  Defenders score: {self.get_final_defender_points()}\n"

        if self.is_done():
            scores = [p.get_score() for p in self.players]
            out += (f"Scores: {scores}\n")

        return out

    def is_done(self):
        # Game is done when all tricks are played
        return "" not in self.history[5]

    def get_picker(self):
        return self.players[self.picker - 1]

    def get_final_picker_points(self):
        if self.is_done() and not self.is_leaster:
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
        if self.is_done() and not self.is_leaster:
            points = 0
            for i, taken in enumerate(self.points_taken):
                if i + 1 != self.picker and i + 1 != self.partner:
                    points += taken

            # Add bury if picker got no points
            if not self.get_final_picker_points():
                points += get_trick_points(self.bury)
            return points
        return False

    def get_leaster_winner(self):
        """Returns the player position (1-5) of the winner of a leaster."""
        if self.is_done() and self.is_leaster:
            # Find players who took at least one trick
            qualified_players = []
            for i in range(5):
                player_pos = i + 1
                took_trick = any(self.trick_winners[j] == player_pos for j in range(6))
                if took_trick:
                    qualified_players.append((player_pos, self.points_taken[i]))

            # Find minimum points among qualified players
            min_points = min(points for _, points in qualified_players)
            winners = [player_pos for player_pos, points in qualified_players if points == min_points]

            # Return winner if unique, otherwise draw randomly for tie
            return winners[0] if len(winners) == 1 else random.choice(winners)
        return False


class Player:
    def __init__(self, game, position, hand):
        self.game = game
        self.position = position
        self.initial_hand = hand
        self.hand = hand[:]
        self.start_states = deque()
        self.actions = deque()

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
        """If this player has been revealed to be the partner"""
        return self.game.partner == self.position

    @property
    def is_secret_partner(self):
        """If this player is partner, regardless of whether it's been revealed"""
        return "JD" in self.hand

    def get_state_vector(self):
        """Return the integer observation vector for the *acting* player.

        Layout (indices in ascending order):

        Header (8 values)
        ----------------
        [0]   player position   (1-5)
        [1]   last position to *pass* during blind picking (5 ⇒ all passed → leaster)
        [2]   picker position (0 if not yet picked)
        [3]   partner position *if known* (0 if unknown)
        [4]   alone called                     (0/1)
        [5]   is_leaster flag                  (0/1)
        [6]   play_started flag                (0/1)
        [7]   current trick index              (0-5)

        Private zones (boolean one-hots)
        ---------------------------------
        [8-39]    hand cards   — 32 bits in DECK order
        [40-71]   blind cards  — 32 bits (picker only, else zeros)
        [72-103]  bury cards   — 32 bits (picker only, else zeros)

        Current trick block  (5 seats × 35 = 175 values)
        -------------------------------------------------
        For each *relative* seat starting with self (index 0) and proceeding
        clockwise (LHO, partner seat, RHO, across):

            33-way one-hot for the card that seat has played this trick
                • index 0  ⇒ "empty"  (seat has not yet played)
                • index 1-32 map to DECK order

            2 role flags appended:
                • is_picker         (0/1)
                • is_known_partner  (0/1)

        This yields 5 × (33 + 2) = 175 elements.

        Total length = 8 + 32 + 32 + 32 + 175 = 279 (see STATE_SIZE).
        """

        state = [self.position]
        state.append(self.last_passed) # 5 if leaster mode. Everyone passed.
        state.append(self.picker)
        partner = self.partner if self.partner else "JD" in self.hand
        state.append(partner)
        state.append(self.alone_called)
        state.append(self.game.is_leaster)
        state.append(self.play_started)
        state.append(self.current_trick)

        # One-hot vectors for hand / blind / bury
        state.extend(get_deck_vector(self.hand))
        state.extend(get_deck_vector(self.blind if self.is_picker else []))
        state.extend(get_deck_vector(self.bury if self.is_picker else []))

        # --- Current trick seat-attributed one-hot + role bits ---
        # Helper to build 33-length one-hot for a card id (0 means empty)
        def one_hot_33(card_id):
            vec = [0] * 33
            if 0 <= card_id < 33:
                vec[card_id] = 1
            return vec

        # Build for self (rel_seat = 0) then clockwise order
        trick = self.game.history[self.current_trick] if self.current_trick < len(self.game.history) else ["" for _ in range(5)]

        for rel_seat in range(5):
            abs_seat = ((self.position + rel_seat - 1) % 5) + 1  # 1-5
            card_str = trick[abs_seat - 1]
            card_id = DECK_IDS.get(card_str, 0) if card_str else 0

            # Card one-hot
            state.extend(one_hot_33(card_id))

            # Role bits
            is_picker = 1 if abs_seat == self.game.picker else 0
            is_partner_known = 1 if self.game.partner and abs_seat == self.game.partner else 0
            state.append(is_picker)
            state.append(is_partner_known)



        return np.array(state, dtype=np.uint8)

    def get_valid_actions(self):
        """Get set of valid actions."""

        # No one yet picked, and not our turn
        if not self.picker and not self.game.is_leaster:
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
        if not self.game.is_leaster and len(self.bury) != 2:
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
            self.hand.extend(self.game.blind)

        if action == "PASS":
            self.game.last_passed = self.position

            if self.game.last_passed == 5:
                # Enter leaster mode
                self.game.is_leaster = True
                self.game.play_started = True
                self.game.leader = 1
                self.game.leaders[0] = 1
                return False

        if "BURY" in action:
            card = action[5:]
            self.game.bury.append(card)
            self.hand.remove(card)

        if action == "ALONE":
            self.game.alone_called = True
            self.game.partner = self.position

        if action in ("ALONE", "JD PARTNER"):
            self.game.play_started = True
            self.game.leader = 1
            self.game.leaders[0] = 1

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

                # In leaster mode, give blind to winner of first trick
                if self.game.is_leaster and self.current_trick == 0:
                    trick_points += get_trick_points(self.game.blind)

                self.game.trick_points[self.current_trick] = trick_points
                self.game.trick_winners[self.current_trick] = winner
                self.game.points_taken[winner_index] += trick_points

                if self.current_trick < 5:
                    self.game.leaders[self.current_trick + 1] = winner
                # print("Trick points: %i" % get_trick_points(trick))
                # print("Winner %i" % get_trick_winner(trick, self.game.current_suit))

                # Next trick must start with winner
                self.game.leader = winner
                self.game.last_player = winner - 1
                self.game.current_suit = ""
                self.game.cards_played = 0
                self.game.was_trick_just_completed = True

                self.game.current_trick += 1
            elif self.game.was_trick_just_completed:
                self.game.was_trick_just_completed = False

        return True

    def get_score(self):
        if self.game.is_done():
            if self.game.is_leaster:
                winner = self.game.get_leaster_winner()
                if winner == self.position:
                    return 4
                return -1
            elif self.game.picker:
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

                if multiplier < 0 and self.game.is_double_on_the_bump:
                    multiplier *= 2

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
