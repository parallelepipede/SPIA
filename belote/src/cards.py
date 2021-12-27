import collections
from enum import Enum, unique
from typing import List, Optional
from copy import deepcopy
from utils import *
import random

color_to_symbol = {
    'SPADES': '♠',
    'DIAMONDS': '♦',
    'HEARTS': '♥',
    'CLUBS': '♣',
}


class Color(Enum):
    """
    Class defining the different colors.
    """
    CLUBS, DIAMONDS, HEARTS, SPADES = range(4)

    def __str__(self):
        return color_to_symbol[self.name]


height_to_symbol = {
    'SEVEN': '7',
    'EIGHT': '8',
    'NINE': '9',
    'TEN': '10',
    'JACK': 'J',
    'QUEEN': 'Q',
    'KING': 'K',
    'ACE': 'A'
}


class Rank(Enum):
    """
    Class defining the height of each card.
    """
    SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = range(8)

    def __str__(self):
        return height_to_symbol[self.name]


class BasicPlayer:
    pass


class Table:
    pass


class Gamestate:
    pass


class Card:
    def __init__(self, color: Color, rank: Rank) -> None:
        """
        Constructor.
        :param color: color of the card
        :param rank: rank of the card
        """
        self.color = color
        self.rank = rank

    def __str__(self) -> str:
        return "(" + str(self.rank) + str(self.color) + ")"

    def __eq__(self, other):
        return self.color == other.color and self.rank == other.rank

    def __lt__(self, other):
        """
        Function used for comparing the cards, must be used ONLY to order cards in a hand.
        :param other:
        :return:
        """
        return (self.color.value * 10 + self.rank.value) < (other.color.value * 10 + other.rank.value)

    def higher_than(self, other, trump: Color) -> bool:
        """
        Functions determining if the first card is higher than the second one to know which one will take the ply.
        :param other: the other card used to do the comparison
        :param trump: the color of the trump
        :return: True if the card if higher than the argument, False else
        """
        if self.color == trump:
            if other.color == trump:
                values = [0, 1, 6, 4, 7, 2, 3, 5]
                return values[self.rank.value] > values[other.rank.value]
            else:
                return True
        elif other.color == trump:
            return False
        else:
            values = [0, 1, 2, 6, 3, 4, 5, 7]
            return values[self.rank.value] >= values[other.rank.value]


class ListCards:
    """
    Class used for inheritance.
    """

    def __init__(self, cards: List[Card] = []) -> None:
        self.cards = cards.copy()

    def __str__(self) -> str:
        return list_str(self.cards)


class Deck(ListCards):
    """
    Class used for the deck of card prior and during distribution.
    """

    def __init__(self, cards: List[Card] = []) -> None:
        ListCards.__init__(self, cards)
        self.topcard: Optional[Card] = None

    def draw(self) -> Card:
        """
        Method drawing the first card of the deck.
        :return: the first card of the deck
        """
        self.topcard = None
        return self.cards.pop()

    def shuffle(self) -> None:
        """
        Method shuffling the deck.
        """
        random.shuffle(self.cards)

    def reveal(self) -> None:
        """
        Method revealing the first card of the deck.
        """
        self.topcard = self.cards[-1]

    def __str__(self) -> str:
        return "{ \ncards: " + ListCards.__str__(self) + ", \ntopcard: " + str(self.topcard) + "}"


class Hand(ListCards):
    def __init__(self, cards: List[Card] = []) -> None:
        ListCards.__init__(self, cards)

    def __str__(self) -> str:
        return "{ \ncards: " + ListCards.__str__(self) + "}"

    def add(self, card: Card) -> None:
        """
        Method adding a card to the hand.
        :param card: the card to add
        """
        self.cards.append(card)
        self.cards.sort()

    def eval(self, topcard: Card, trump: Color) -> int:
        """
        Method evaluating the player's hand.
        :param topcard: the card to take
        :param trump: the trump
        :return: an integer representing the value of this hand if the player takes this trump
        """
        trump_list = [card for card in self.cards if card.color == trump]
        if topcard.color == trump:
            trump_list.append(topcard)
        trick = Trick(BasicPlayer("Val"))
        for card in trump_list:
            trick.add(card)
        v1 = trick.get_points(topcard.color)
        v2 = 0
        for card in self.cards:
            if card not in trump_list:
                v2 += 10 * (card.rank == Rank.ACE)
        return v1 + v2


class Trick(ListCards):
    def __init__(self, first_player: BasicPlayer) -> None:
        ListCards.__init__(self)
        self.first_player = first_player

    def add(self, card: Card) -> None:
        """
        Method adding a card to the trick.
        :param card: the card to add to the trick
        """
        self.cards.append(card)

    def get_winner(self, trump: Color) -> int:
        """
        Method giving the winner of the trick.
        :param trump: the color of the trump
        :return: the index (in this trick) of the player winning the trick
        """
        max_card = self.cards[0]
        for card in self.cards[1:]:
            if card.higher_than(max_card, trump):
                max_card = card
        return self.cards.index(max_card)

    def get_points(self, trump: Color) -> int:
        """
        Method calculating the points of the trick.
        :param trump: the color of the trump
        :return: the points in the trick
        """
        trump_value = [0, 0, 14, 10, 20, 3, 4, 11]
        value = [0, 0, 0, 10, 2, 3, 4, 11]
        return sum(
            trump_value[card.rank.value] if card.color == trump else value[card.rank.value] for card in self.cards)

    def get_legal_moves(self, hand: Hand, trump: Color) -> List[Card]:
        """
        Method returning the list of legal moves for a given hand and a given trump in a trick.
        :param hand: the hand of the player
        :param trump: the current trump
        :return: the list of playable cards
        """
        if len(self.cards) == 0:
            return hand.cards
        else:
            legal_moves = []
            pseudo_legal = []
            if self.cards[0].color == trump:
                for card in hand.cards:
                    if card.color == trump:
                        if card.higher_than(self.cards[self.get_winner(trump)], trump):
                            legal_moves.append(card)
                        else:
                            pseudo_legal.append(card)
                if len(legal_moves) != 0:
                    return legal_moves
                elif len(pseudo_legal) != 0:
                    return pseudo_legal
                else:
                    return hand.cards
            else:
                for card in hand.cards:
                    if card.color == self.cards[0].color:
                        legal_moves.append(card)
                if len(legal_moves) != 0:
                    return legal_moves
                else:
                    if self.cards[self.get_winner(trump)].color == trump:
                        for card in hand.cards:
                            if card.color == trump:
                                if card.higher_than(self.cards[self.get_winner(trump)], trump):
                                    legal_moves.append(card)
                                else:
                                    pseudo_legal.append(card)
                        if len(legal_moves) != 0:
                            return legal_moves
                        elif len(pseudo_legal) != 0:
                            return pseudo_legal
                        else:
                            return hand.cards
                    else:
                        for card in hand.cards:
                            if card.color == trump:
                                legal_moves.append(card)
                        if len(legal_moves) != 0:
                            return legal_moves
                        else:
                            return hand.cards


class BasicPlayer:
    def __init__(self, name: str) -> None:
        self.name = name
        self.hand: Hand = Hand()
        self.team: str = None

    def __str__(self) -> str:
        return "{ name: " + str(self.name) + ", \nhand: " + str(self.hand) + "}"

    def draw(self, deck: Deck) -> None:
        """
        Method drawing a card for the player and adding it to its hand.
        :param deck: the deck to draw from
        """
        card = deck.draw()
        self.hand.add(card)

    def take(self, topcard: Card, second_turn: bool = False) -> Optional[Color]:
        """
        Method determining if the player will take.
        :param topcard: the topcard of the deck
        :param second_turn: boolean telling if it's the second turn
        :return: the color of the trump or None
        """
        pass

    def play_card(self, trick: Trick, trump: Color) -> None:
        """
        Method playing a card
        :param trick:
        :param trump:
        :return:
        """
        pass

    def play_this_card(self, trick: Trick, card: Card):
        trick.add(card)
        self.hand.cards.remove(card)

    def take_this_color(self, trump: Color, state: Gamestate) -> None:
        state.atout = trump
        state.preneur = self


class SimpleBotPlayer(BasicPlayer):
    def take(self, topcard: Card, second_turn: bool = False) -> Optional[Color]:
        """
        Method using the eval function and an arbitrary value to know if the player will take.
        :param topcard: the top card of the deck
        :param second_turn: a boolean to know if we are at the first turn
        :return: a color if the player wants to take, else None
        """
        if second_turn:
            max = -1
            current_color = None
            for color in Color:
                val = self.hand.eval(topcard, color)
                if val > max:
                    max = val
                    current_color = color
            return current_color if max > 30 else None
        return topcard.color if self.hand.eval(topcard, topcard.color) > 30 else None

    def play_card(self, trick: Trick, trump: Color) -> None:
        """
        Method playing the last legal move possible.
        :param trick: the current trick
        :param trump: the current trump
        """
        legal_moves = trick.get_legal_moves(self.hand, trump)
        card = legal_moves[-1]
        self.hand.cards.remove(card)
        trick.add(card)


class HumanPlayer(BasicPlayer):
    def take(self, topcard: Card, second_turn: bool = False) -> Optional[Color]:
        """
        Method asking the human player by CLI if he wants to take, and at which color.
        :param topcard: the top card of the deck
        :param second_turn: a boolean to know if we are at the first turn
        :return: a color if the player wants to take, else None
        """
        print("*" * 100)
        print("*" * 100)
        print("This is your hand : {}".format(str(self.hand)))
        print("This is the topcard : {}".format(topcard))
        if second_turn:
            print("Do you want to take ? ({}, None)".format(
                ", ".join([color.name for color in Color if color != topcard.color])))
            inp = input().split()
            while len(inp) == 0 : 
                print("Please enter a valid value : {}, None)".format(
                ", ".join([color.name for color in Color if color != topcard.color])))
                inp = input().split()
            inp = inp[0]
            print("*" * 100)
            print("*" * 100)
            if inp.upper() == "HEARTS" and topcard.color != Color.HEARTS:
                return Color.HEARTS
            if inp.upper() == "SPADES" and topcard.color != Color.SPADES:
                return Color.SPADES
            if inp.upper() == "DIAMONDS" and topcard.color != Color.DIAMONDS:
                return Color.DIAMONDS
            if inp.upper() == "CLUBS" and topcard.color != Color.CLUBS:
                return Color.CLUBS
        else:
            print("This is the color  : {}".format(topcard.color.name))
            inp = input("Do you want to take ? (y/n) ").split()[0]
            print("*" * 100)
            print("*" * 100)
            if inp == "y":
                return topcard.color
            else:
                return None

    def play_card(self, trick: Trick, trump: Color) -> None:
        """
        Method asking the player which card he wants to play.
        :param trick: the current trick
        :param trump: the current trump
        """
        legal_moves = trick.get_legal_moves(self.hand, trump)
        print("*" * 100)
        print("*" * 100)
        print("This is the current trick : {}".format(trick))
        print("Those are your legal moves : {}".format(Hand(legal_moves)))
        print("*" * 100)
        print("*" * 100)
        inp = input("Enter your move :")
        try :
            inp = int(inp)
        except ValueError : 
            if len(self.hand.cards) == 0 : 
                print("Please enter number 0")
            else:
                print("Please enter a number between 0 and", len(self.hand.cards))
        card = legal_moves[inp]
        self.hand.cards.remove(card)
        trick.add(card)


class Team:
    def __init__(self, name: str, players: List[BasicPlayer]) -> None:
        self.name = name

        if len(players) != 2:
            raise Exception("Expected a team of 2 players, got {} instead".format(len(players)))
        self.players = players.copy()
        for player in players:
            player.team = name
        self.score = 0

    def __str__(self) -> str:
        return ("{ name: " + str(self.name) + ", players: " + list_str(self.players)
                + ", score: " + str(self.score) + "}")


class Table:
    def __init__(self, players: List[BasicPlayer], deck: Deck) -> None:
        self.players = players
        self.teams = [Team("Team NS", [players[0], players[2]]), Team("Team EO", [players[1], players[3]])]
        self.deck = deck
        self.past_tricks = []
        self.trick = None

    def __str__(self) -> str:
        return ("{ players: " + list_str(self.players) + ", \n"  # trick: " + str(self.trick) +
                                                         ", \ndeck: " + str(self.deck) + "}")


class Gamestate:
    def __init__(self, table: Table, first_taker: BasicPlayer) -> None:
        self.table = table
        self.ply = 0
        self.first_player = first_taker
        self.current_player = first_taker
        self.first_taker = first_taker
        self.atout: Optional[Color] = None
        self.preneur: Optional[BasicPlayer] = None
        self.taking_turn = 1

    @property
    def iter_players(self) -> List[BasicPlayer]:
        first_taker_index = self.table.players.index(self.first_taker)
        return list_rotate(self.table.players, first_taker_index)

    def __str__(self) -> str:
        return ("{ table: " + str(self.table)
                + ", \nfirst player: " + str(self.first_player.name)
                + ", \natout: " + str(self.atout)
                + ", \npreneur: " + (str(self.preneur.name) if self.preneur else "None") + "}")

    def __deepcopy__(self, memo):
        """²
        https://stackoverflow.com/questions/1500718/how-to-override-the-copy-deepcopy-operations-for-a-python-object
        :param memo:
        :return:
        """
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def mcts_get_legal_actions(self):
        # for player in self.iter_players:
        #     print(len(player.hand.cards))
        if self.atout:
            legal_moves = self.table.trick.get_legal_moves(self.current_player.hand, self.atout)
        else:
            if self.taking_turn == 1:
                legal_moves = [self.table.deck.topcard.color, None]
            elif self.taking_turn == 2:
                legal_moves = [color for color in Color if color != self.table.deck.topcard.color] + [None]
            else:
                raise Exception("taking_turn value error.")
        return legal_moves

    def mcts_move(self, action):
        # if the trump is defined, we play a trick
        if self.atout:
            # the current player has to play a card
            self.current_player.play_this_card(self.table.trick, action)
            # if its not the last card of the trick, the next player will play
            if len(self.table.trick.cards) < 4:
                self.current_player = self.table.players[(self.table.players.index(self.current_player) + 1) % 4]
            # it's the last card of the trick
            elif len(self.table.trick.cards) == 4:
                # finding who will be the next first player (the player who won the trick)
                play_order = list_rotate(self.table.players, self.table.players.index(self.first_player))
                self.first_player = play_order[self.table.trick.get_winner(self.atout)]
                # adding the points of the trick to the corresponding team
                if self.table.teams[0].name == self.first_player.team:
                    self.table.teams[0].score += self.table.trick.get_points(self.atout) + 10 * (self.ply == 7)
                else:
                    self.table.teams[1].score += self.table.trick.get_points(self.atout) + 10 * (self.ply == 7)
                # adding the finished trick to the list of past tricks
                self.table.past_tricks.append(self.table.trick)
                # going for the next ply
                self.ply += 1
                print(self.ply)
                self.table.trick = Trick(self.first_player)
                # the current player is the first player of the trick
                self.current_player = self.first_player
            return self
        else:
            # if we are during the first turn
            if self.taking_turn == 1:
                # if the player doesn't want to take
                if action is None:
                    # we pass to the next player
                    self.current_player = self.table.players[(self.table.players.index(self.current_player) + 1) % 4]
                    # if the next player is the first one, then we are at the second turn
                    if self.current_player == self.first_taker:
                        self.taking_turn = 2
                    return self
                else:
                    # here the player wants to take
                    self.current_player.take_this_color(action, self)
                    self.current_player.hand.add(self.table.deck.topcard)
                    self.taking_turn = None
                    # drawing the cards
                    for player in self.iter_players:
                        n_cards = 2 if player is self.preneur else 3
                        for j in range(n_cards):
                            player.draw(self.table.deck)
                    # setting the first player of the first ply
                    self.current_player = self.first_taker
                    # creating the ply accordingly
                    self.table.trick = Trick(self.current_player)
                    return self
            # we are at the second turn
            elif self.taking_turn == 2:
                if action is None:
                    # we pass to the next player
                    self.current_player = self.table.players[(self.table.players.index(self.current_player) + 1) % 4]
                    # if the next player is the first one, then we scrap the game
                    print("END OF THE GAME")
                    raise Exception("This is the end of the game because no one took.")
                else:
                    # here the player wants to take
                    self.current_player.take_this_color(action, self)
                    self.current_player.hand.add(self.table.deck.topcard)
                    self.taking_turn = None
                    # drawing the cards
                    for player in self.iter_players:
                        n_cards = 2 if player is self.preneur else 3
                        for j in range(n_cards):
                            player.draw(self.table.deck)
                    # setting the first player of the first ply
                    self.current_player = self.first_taker
                    # creating the ply accordingly
                    self.table.trick = Trick(self.current_player)
                    return self
            else:
                raise Exception("Error as there is no trump and the game isn't during a taking turn.")

    def is_game_over(self) -> bool:
        """
        Checks if the game is over
        """
        return self.ply == 8

    def game_result(self):
        print("Game result : {}".format(self.table.teams[0].score > self.table.teams[0].score))
        if self.table.teams[0].score > self.table.teams[0].score:
            return 1
        return -1
