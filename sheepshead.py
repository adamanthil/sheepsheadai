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

# ---------------------------------------------------------------------------
# Partner-selection modes
#   0 → Standard Jack-of-Diamonds partner
#   1 → Called-Ace / Called-10 partner
# ---------------------------------------------------------------------------
PARTNER_BY_JD = 0
PARTNER_BY_CALLED_ACE = 1
# Valid cards that may be called as partner (fail Aces or 10s)
CALLED_ACES = ["AC", "AS", "AH"]
CALLED_10S = ["10C", "10S", "10H"]
CALLED_PARTNER_CARDS = CALLED_ACES + CALLED_10S

# Special token for a face-down “under” card when using called-ace variant
UNDER_TOKEN = "UNDER"
UNDER_CARD_ID = 33  # 0 = empty, 1-32 = real cards, 33 = unknown under

ACTIONS = [
    "PICK",
    "PASS",
    "ALONE",
    # Opposite action from calling ALONE. Standard Jack of Diamonds partner.
    # Avoids lacking an action for state transition.
    "JD PARTNER",
]

# Add partner call actions
CALL_ACTIONS = [f"CALL {c}" for c in CALLED_PARTNER_CARDS]
UNDER_CALL_ACTIONS = [f"CALL {c} UNDER" for c in ["AC", "AS", "AH"]]
CALL_ACTIONS.extend(UNDER_CALL_ACTIONS)
ACTIONS.extend(CALL_ACTIONS)

# Add UNDER actions for placing a card as an under
UNDER_ACTIONS = [f"UNDER {c}" for c in DECK]
ACTIONS.extend(UNDER_ACTIONS)

# Add bury actions fior all cards
BURY_ACTIONS = [f"BURY {c}" for c in DECK]
ACTIONS.extend(BURY_ACTIONS)

# Add play actions for all cards
PLAY_ACTIONS = [f"PLAY {c}" for c in DECK]
PLAY_ACTIONS.append(f"PLAY {UNDER_TOKEN}")
ACTIONS.extend(PLAY_ACTIONS)

# Define dics for quick action lookup
ACTION_LOOKUP = { i + 1: a for i, a in enumerate(ACTIONS)}
ACTION_IDS = { a: i + 1 for i, a in enumerate(ACTIONS)}

# Define deck variables for quick lookup
DECK_IDS = {k: v + 1 for v, k in enumerate(DECK)}

TRUMP_POWER = {k: len(TRUMP) - v for v, k in enumerate(TRUMP)}
FAIL_POWER = {k: len(FAIL) - v for v, k in enumerate(FAIL)}


# Human-readable name for partner-selection modes
def get_partner_mode_name(partner_mode: int) -> str:
    return "Jack-of-Diamonds" if partner_mode == PARTNER_BY_JD else "Called-Ace"


def get_card_suit(card):
    return "T" if card in TRUMP else card[-1]


def get_trick_winner(trick, suit, is_called_10_suit=False):
    power_list = []
    for card in trick:
        try:
            power = 100 * TRUMP_POWER[card]
        except KeyError:
            if is_called_10_suit and card == f"10{suit}":
                # Called 10s always take the suit if not trumped
                power = len(FAIL_POWER) + 1
            else:
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


def get_leadable_called_partner_cards(hand, called_card):
    # Filter out other cards in called suit besides the ace
    return [c for c in hand if c == called_card or get_card_suit(c) != get_card_suit(called_card)]


def get_playable_called_picker_cards(hand, called_card):
    # Picker can't fail off called suit (or the called card if they called themselves)
    if len(filter_by_suit(hand, get_card_suit(called_card))) == 1:
        return [c for c in hand if get_card_suit(c) != get_card_suit(called_card)]
    return [c for c in hand if c != called_card]


def get_callable_cards(hand):
    callable_cards = []
    possible_under_suits = []
    for card in CALLED_ACES:
        if card in hand:
            continue

        suit = get_card_suit(card)
        fails_in_suit = filter_by_suit(hand, suit)
        if not fails_in_suit:
            possible_under_suits.append(suit)
            continue

        callable_cards.append(card)

    if callable_cards:
        return callable_cards

    if possible_under_suits:
        return [f"A{suit} UNDER" for suit in possible_under_suits]

    return CALLED_10S


def get_trick_points(trick):
    points = 0
    for card in trick:
        points += get_card_points(card)
    return points


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
    def __init__(
        self,
        double_on_the_bump=True,
        partner_selection_mode=PARTNER_BY_CALLED_ACE,
        picking_player=None,
        picking_hand=None,
        seed=None,
    ):
        # Create local RNG instance for reproducible randomness
        self.rng = random.Random(seed) if seed is not None else random.Random()

        self.partner_mode_flag = partner_selection_mode  # 0 = JD, 1 = Called Ace
        if picking_hand:
            # Remove picking_hand cards from DECK to form the deck
            self.deck = [card for card in DECK if card not in picking_hand]
            if picking_player is None:
                picking_player = self.rng.randint(1, 5)
        else:
            self.deck = DECK[:]
        self.rng.shuffle(self.deck)

        self.last_passed = picking_player - 1 if picking_player else 0
        self.picker = picking_player if picking_player else 0
        self.partner = 0
        self.blind = self.deck[(len(self.deck) - 2):]
        self.bury = []
        self.picker_chose_partner = False
        self.was_called_suit_played = False
        self.alone_called = False
        self.called_card = None
        self.is_called_under = False  # Whether picker used an under call
        self.under_card = None # The card that is to be played as an under
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
                    hand = self.rng.sample(deal_deck, 6)
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

        card_count = 0
        bury_count = 0
        while not self.is_done():
            for player in self.players:
                actions = player.get_valid_action_ids()
                # print(f"PLAYER {player.position}")
                # print([ACTION_LOOKUP[a] for a in actions])
                while actions:
                    action = self.rng.sample(list(actions), 1)[0]

                    action_str = ACTION_LOOKUP[action]
                    pretty_action_str = action_str

                    if action_str in PLAY_ACTIONS:
                        card_count += 1

                    # Colorize cards in action descriptions
                    if verbose and ("BURY" in action_str or "PLAY" in action_str or "UNDER" in action_str):
                        card = action_str.split()[-1]  # Get the card from "BURY QC" or "PLAY QC"
                        colored_card = colorize_card(card)
                        pretty_action_str = action_str.replace(card, colored_card)

                    if "BURY" in action_str:
                        bury_count += 1

                    if verbose:
                        print(f" -- Player {player.position}: {pretty_action_str}")

                    # Print a new line between tricks
                    if verbose and (card_count == 5 or action_str in BURY_ACTIONS and bury_count == 2):
                        print(f"{'-'*40}")
                        card_count = 0

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

    @property
    def called_suit(self):
        return get_card_suit(self.called_card) if self.called_card else None

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
            return winners[0] if len(winners) == 1 else self.rng.choice(winners)
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
        if self.game.alone_called or self.game.is_leaster:
            return False
        if self.game.partner_mode_flag == PARTNER_BY_JD:
            return "JD" in self.hand
        # Called-Ace variant
        return self.game.called_card in self.hand if self.game.called_card else False

    def get_state_dict(self, trick_index=None):
        """Return a structured observation dict for the acting player.

        All values are integers (np.uint8) or small fixed-size integer arrays.
        Card ids: 0 = PAD/none, 1..32 = real cards, 33 = UNDER.

        Relative seats are with respect to this player: 1=self, 2=LHO, 3=+2, 4=+3, 5=+4.
        Also includes absolute picker position (1..5, 0 if unknown) for pick-strength context.
        """

        def rel(seat: int | None) -> int:
            if not seat:
                return 0
            return ((seat - self.position) % 5) + 1

        def to_ids(cards, length):
            ids = [DECK_IDS[c] for c in cards]
            ids += [0] * (length - len(ids))
            return np.array(ids[:length], dtype=np.uint8)

        # Header scalars
        partner_mode = np.uint8(self.game.partner_mode_flag)
        is_leaster = np.uint8(1 if self.game.is_leaster else 0)
        play_started = np.uint8(1 if self.game.play_started else 0)
        cur_trick_idx = self.game.current_trick if trick_index is None else trick_index
        current_trick = np.uint8(cur_trick_idx)
        alone_called = np.uint8(1 if self.game.alone_called else 0)
        called_card_id = np.uint8(DECK_IDS[self.game.called_card]) if self.game.called_card else np.uint8(0)
        called_under = np.uint8(1 if getattr(self.game, 'is_called_under', False) else 0)
        picker_rel = np.uint8(rel(self.game.picker))
        partner_rel = np.uint8(rel(self.game.partner) if self.game.partner else 0)
        leader_rel = np.uint8(rel(self.game.leader) if self.game.play_started else 0)
        picker_position = np.uint8(self.game.picker if self.game.picker else 0)

        # Sets (fixed sizes)
        hand_ids = to_ids(self.hand, 8)
        blind_ids = to_ids(self.blind if self.is_picker else [], 2)
        bury_ids = to_ids(self.bury if self.is_picker else [], 2)

        # Trick arrays in relative seat order 1..5
        idx = cur_trick_idx
        trick = self.game.history[idx] if idx < len(self.game.history) else ["" for _ in range(5)]
        trick_card_ids = []
        trick_is_picker = []
        trick_is_partner_known = []
        for r in range(1, 6):
            abs_seat = ((self.position + r - 2) % 5) + 1
            card_str = trick[abs_seat - 1]
            if card_str == UNDER_TOKEN:
                cid = UNDER_CARD_ID
            elif card_str:
                cid = DECK_IDS[card_str]
            else:
                cid = 0
            trick_card_ids.append(cid)
            trick_is_picker.append(1 if abs_seat == self.game.picker else 0)
            trick_is_partner_known.append(1 if self.game.partner and abs_seat == self.game.partner else 0)

        obs = {
            'partner_mode': partner_mode,
            'is_leaster': is_leaster,
            'play_started': play_started,
            'current_trick': current_trick,
            'alone_called': alone_called,
            'called_card_id': called_card_id,
            'called_under': called_under,
            'picker_rel': picker_rel,
            'partner_rel': partner_rel,
            'leader_rel': leader_rel,
            'picker_position': picker_position,
            'hand_ids': np.array(hand_ids, dtype=np.uint8),
            'blind_ids': np.array(blind_ids, dtype=np.uint8),
            'bury_ids': np.array(bury_ids, dtype=np.uint8),
            'trick_card_ids': np.array(trick_card_ids, dtype=np.uint8),
            'trick_is_picker': np.array(trick_is_picker, dtype=np.uint8),
            'trick_is_partner_known': np.array(trick_is_partner_known, dtype=np.uint8),
        }
        return obs

    def get_last_trick_state_dict(self):
        last_idx = max(0, self.game.current_trick - 1)
        return self.get_state_dict(trick_index=last_idx)

    def get_valid_actions(self):
        """Get set of valid actions."""

        # No one yet picked, and not our turn
        if not self.picker and not self.game.is_leaster:
            if self.last_passed != self.position - 1:
                return set()
            return set(["PICK", "PASS"])

        # We picked.
        if self.is_picker and not self.play_started:
            if not self.game.picker_chose_partner:
                if self.game.partner_mode_flag == PARTNER_BY_JD:
                    return set(["ALONE", "JD PARTNER"])
                else:
                    actions = set(["ALONE"])
                    actions.update([f"CALL {c}" for c in get_callable_cards(self.hand)])
                    return actions

            if self.game.is_called_under and not self.game.under_card:
                return set([f"UNDER {c}" for c in self.hand])

            # Need to bury.
            if self.game.called_card:
                return set([f"BURY {c}" for c in get_playable_called_picker_cards(self.hand, self.game.called_card)])
            else:
                return set([f"BURY {c}" for c in self.hand])

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
                # Start of trick, generally we can play anything
                if not self.is_partner and self.game.called_card and self.is_secret_partner:
                    actions.update([f"PLAY {c}" for c in get_leadable_called_partner_cards(self.hand, self.game.called_card)])
                else:
                    actions.update([f"PLAY {c}" for c in self.hand])

                if self.game.is_called_under and self.is_picker and not self.game.was_called_suit_played:
                    actions.update([f"PLAY {UNDER_TOKEN}"])
            else:
                # Follow suit if possible
                cards_in_suit = filter_by_suit(self.hand, self.game.current_suit)
                if self.game.called_card and self.game.called_card in cards_in_suit:
                    # We must play called card when suit is led
                    actions.update([f"PLAY {self.game.called_card}"])
                elif cards_in_suit:
                    actions.update([f"PLAY {c}" for c in cards_in_suit])
                elif self.game.is_called_under and self.is_picker and not self.game.was_called_suit_played and self.game.current_suit == self.game.called_suit:
                    actions.update([f"PLAY {UNDER_TOKEN}"])
                else:
                    # Can't follow suit
                    if self.is_picker and self.game.called_card and not self.game.was_called_suit_played and self.game.current_trick < 5:
                        # Picker can't fail off called suit
                        actions.update([f"PLAY {c}" for c in get_playable_called_picker_cards(self.hand, self.game.called_card)])
                    elif self.game.called_card and not self.is_partner and self.is_secret_partner and self.game.current_trick < 5:
                        # Partner can't fail off called card
                        actions.update([f"PLAY {c}" for c in self.hand if c != self.game.called_card])
                    else:
                        actions.update([f"PLAY {c}" for c in self.hand])
                        if self.is_picker and self.game.is_called_under and not self.game.was_called_suit_played and self.game.current_trick == 5:
                            # Allow playing under on final trick even if called suit isn't led
                            actions.update([f"PLAY {UNDER_TOKEN}"])

        return actions

    def get_valid_action_ids(self):
        """Get set of valid action IDs."""
        return set(ACTION_IDS[a] for a in self.get_valid_actions())

    def act(self, action_id):
        if action_id not in self.get_valid_action_ids():
            return False

        self.start_states.append(self.get_state_dict())
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

        if "BURY" in action:
            card = action[5:]
            self.game.bury.append(card)
            self.hand.remove(card)

            if len(self.game.bury) == 2:
                self.game.play_started = True
                self.game.leader = 1
                self.game.leaders[0] = 1

        if action == "ALONE":
            self.game.alone_called = True
            self.game.partner = self.position

        if "CALL" in action:
            parts = action.split()
            called_card = parts[1]

            self.game.called_card = called_card
            self.game.picker_chose_partner = True

            if action in UNDER_CALL_ACTIONS:
                self.game.is_called_under = True

        if action in ("ALONE", "JD PARTNER"):
            self.game.picker_chose_partner = True

        if action in UNDER_ACTIONS:
            parts = action.split()
            under_card = parts[1]
            self.game.under_card = under_card
            self.hand.remove(under_card)

        if "PLAY" in action:
            card = action[5:]

            # Set suit lead if we are the first to play this trick
            if self.game.leader == self.game.last_player + 1:
                if card == UNDER_TOKEN:
                    self.game.current_suit = self.game.called_suit
                else:
                    self.game.current_suit = get_card_suit(card)

            self.game.history[self.current_trick][self.position - 1] = card
            self.game.last_player = self.position
            self.game.cards_played += 1
            if card != UNDER_TOKEN:
                self.hand.remove(card)

            # Reveal partner when the partner card is played
            if not self.game.alone_called:
                if self.game.partner_mode_flag == PARTNER_BY_JD and card == "JD":
                    self.game.partner = self.position
                elif self.game.partner_mode_flag == PARTNER_BY_CALLED_ACE and card == self.game.called_card:
                    self.game.partner = self.position

            if self.game.last_player == 5:
                self.game.last_player = 0

            if self.game.cards_played == 5:
                if self.current_trick == 5:
                    # Handle buried partner card on final play (JD only)
                    if self.game.partner_mode_flag == PARTNER_BY_JD and "JD" in self.bury:
                        self.game.partner = self.game.picker

                trick = self.game.history[self.current_trick]

                is_called_10_suit = (
                    self.game.called_card
                    and self.game.called_card in CALLED_10S
                    and not self.game.was_called_suit_played
                    and self.game.current_suit == self.game.called_suit
                )
                winner = get_trick_winner(
                    trick,
                    self.game.current_suit,
                    is_called_10_suit
                )
                winner_index = winner - 1
                trick_points = get_trick_points(trick)

                # In leaster mode, give blind to winner of first trick
                if self.game.is_leaster and self.current_trick == 0:
                    trick_points += get_trick_points(self.game.blind)

                # Add under points to trick if it was played
                if self.game.is_called_under and UNDER_TOKEN in trick:
                    trick_points += get_card_points(self.game.under_card)

                self.game.trick_points[self.current_trick] = trick_points
                self.game.trick_winners[self.current_trick] = winner
                self.game.points_taken[winner_index] += trick_points

                if self.game.called_card and not self.game.was_called_suit_played and self.game.called_suit == self.game.current_suit:
                    self.game.was_called_suit_played = True

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
