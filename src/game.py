from typing import List, Optional
from cards import *
from utils import maybe


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
            state.taking_turn = None
            break
    if not state.preneur:
        state.taking_turn = 2
        for player in state.iter_players:
            if player.take(table.deck.topcard, second_turn=True):
                state.atout = table.deck.topcard.color
                state.preneur = player
                player.draw(deck)
                state.taking_turn = None
                break
    if state.taking_turn is None:
        raise Exception("No one took.")

    # Final distribution
    for player in state.iter_players:
        n_cards = 2 if player is state.preneur else 3
        for j in range(n_cards):
            player.draw(deck)

    print("Drawing state: ", state)
    print(80 * "-")

    trick_winner = state.first_player
    for ply in range(8):  # iterating over the plies
        table.trick = Trick(state.first_player)
        # setting the order of the turn, starting from the first player
        play_order = list_rotate(table.players, table.players.index(state.first_player))
        for player in play_order:
            player.play_card(table.trick, state.atout)
        state.first_player = play_order[table.trick.get_winner(state.atout)]
        if table.teams[0].name == trick_winner.team:
            table.teams[0].score += table.trick.get_points(state.atout) + 10 * (ply == 7)
        else:
            table.teams[1].score += table.trick.get_points(state.atout) + 10 * (ply == 7)
        table.past_tricks.append(table.trick)
        print(80 * "-")
        print("Ply {} : {}".format(ply, state))
        print("Current trick : {}\n with a new first player {}".format(table.trick, state.first_player))
        print(80 * "-")
        for team in table.teams:
            print(str(team))
        table.past_tricks.append(table.trick)
    print(80 * "-")
    print("Final result")
    for team in table.teams:
        print(str(team))
