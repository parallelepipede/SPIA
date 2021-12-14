import collections
from enum import Enum, unique
from typing import List, Optional

from src.utils import *
import random


class Color(Enum):
    """
    Class defining the different colors.
    """
    CLUBS, DIAMONDS, HEARTS, SPADES = range(4)


class Rank(Enum):
    """
    Class defining the height of each card.
    """
    SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = range(8)


class BasicPlayer:
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
        return "(" + str(self.color) + ", " + str(self.rank) + ")"

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
            inp = input("")
            print("*" * 100)
            print("*" * 100)
            if inp.upper() == "None":
                return None
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
            inp = input("Do you want to take ? (y/n)")
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
        card = legal_moves[int(inp)]
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
