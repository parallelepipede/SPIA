from typing import List, Optional
from cards import *
from table import Table
from utils import maybe


class Gamestate:
    def __init__(self, table: Table, first_player: BasicPlayer) -> None:
        self.table = table
        self.first_player = first_player
        self.atout: Optional[Color] = None
        self.preneur: Optional[BasicPlayer] = None

    @property
    def iter_players(self) -> List[BasicPlayer]:
        first_player_index = self.table.players.index(self.first_player)
        return list_rotate(self.table.players, first_player_index)

    def __str__(self) -> str:
        return ("{ table: " + str(self.table)
                + ", \nfirst player: " + str(self.first_player.name)
                + ", \natout: " + str(self.atout)
                + ", \npreneur: " + (str(self.preneur.name) if self.preneur else "None") + "}")

    # def mcts_get_legal_actions(self):


def init_deck() -> Deck:
    L: List[Card] = []
    for color in Color:
        for rank in Rank:
            L.append(Card(color, rank))
    return Deck(L)


def game_loop() -> None:
    deck = init_deck()
    deck.shuffle()
    table = Table(
        [HumanPlayer("Human_South"), SimpleBotPlayer("West"), SimpleBotPlayer("North"), SimpleBotPlayer("East")], deck)
    state = Gamestate(table, random.choice(table.players))

    print("Initial state:\n", state)
    print(80 * "-")

    # Drawing phase
    for i in range(2):  # Two draw turns
        for player in state.iter_players:
            n_card = 2 if i == 0 else 3
            for j in range(n_card):
                player.draw(deck)

    # Revealing top card of the deck
    table.deck.reveal()

    # Loop until someone takes the first card
    assert table.deck.topcard is not None
    for player in state.iter_players:
        if player.take(table.deck.topcard):
            state.atout = table.deck.topcard.color
            state.preneur = player
            player.draw(deck)
            break
    if not state.preneur:
        for player in state.iter_players:
            if player.take(table.deck.topcard, second_turn=True):
                state.atout = table.deck.topcard.color
                state.preneur = player
                player.draw(deck)
                break

    # Final distribution
    for player in state.iter_players:
        n_cards = 2 if player is state.preneur else 3
        for j in range(n_cards):
            player.draw(deck)

    print("Drawing state: ", state)
    print(80 * "-")

    trick_winner = state.first_player
    for ply in range(8):  # iterating over the plies
        # first player is defined by the table for the first ply, else it's the last play winner
        first_player: BasicPlayer = state.first_player if ply == 0 else trick_winner
        current_trick = Trick(first_player)
        # setting the order of the turn, starting from the first player
        play_order = list_rotate(table.players, table.players.index(first_player))
        for player in play_order:
            player.play_card(current_trick, state.atout)
        trick_winner: BasicPlayer = play_order[current_trick.get_winner(state.atout)]
        if table.teams[0].name == trick_winner.team:
            table.teams[0].score += current_trick.get_points(state.atout) + 10 * (ply == 7)
        else:
            table.teams[1].score += current_trick.get_points(state.atout) + 10 * (ply == 7)
        table.past_tricks.append(current_trick)
        print(80 * "-")
        print("Ply {} : {}".format(ply, state))
        print("Current trick : {}\n started by {}".format(current_trick, first_player))
        print(80 * "-")
        for team in table.teams:
            print(str(team))
    print(80 * "-")
    print("Final result")
    for team in table.teams:
        print(str(team))
