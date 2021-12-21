import time
from random import random
from typing import List

import numpy as np
from collections import defaultdict

from src.cards import *
from src.game import init_deck, play_sample_game


# class MCTSNode:
#     pass


class MCTSNode:
    def __init__(self, state: Gamestate, parent=None, parent_action=None):
        self.state: Gamestate = state
        self.parent: MCTSNode = parent
        self.parent_action = parent_action
        self.children: List[MCTSNode] = []
        self.number_of_visits: int = 0
        self._results: dict = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        # self._untried_actions = None
        # self._untried_actions = self.untried_actions()

    def is_leaf(self):
        return len(self.children) == 0

    def selection(self):
        # while not self.is_leaf():
        #     return self.children[0].selection()
        return self.select_best_child()

    def is_terminal(self) -> bool:
        return self.state.is_game_over()

    def expansion(self):
        if self.is_terminal():
            result = self.state.game_result()
            self.backpropagation(result)  # maybe
            pass
        # elif not self.is_terminal() and len(self._untried_actions) != 0:
        #     raise Exception("Error : node isn't terminal but there is no move possible.")
        else:
            for move in self.state.mcts_get_legal_actions():
                new_state = self.state.mcts_move(move)
                child = MCTSNode(new_state, parent=self, parent_action=move)
                self.children.append(child)
            return self.selection()

    def simulation(self):
        playout_result = self.rollout()  # play a simulation
        return playout_result

    def backpropagation(self, result):
        self.number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagation(result)

    def select_best_move(self):
        c_param = 0.1
        choices_weights = [(c._results[1] / c.number_of_visits) + c_param * np.sqrt(
            (2 * np.log(self.number_of_visits) / c.number_of_visits)) for c in self.children]
        best_child = self.children[np.argmax(choices_weights)]
        return best_child.parent_action

    def select_best_child(self):
        c_param = 0.1
        choices_weights = [(c._results[1] / c.number_of_visits) + c_param * np.sqrt(
            (2 * np.log(self.number_of_visits) / c.number_of_visits)) for c in self.children]
        best_child = self.children[np.argmax(choices_weights)]
        return best_child

    def rollout(self):
        gamestate = deepcopy(self.state)
        if not self.state.atout:
            assert gamestate.table.deck.topcard is not None
            for player in gamestate.iter_players:
                if player.take(gamestate.table.deck.topcard):
                    gamestate.atout = gamestate.table.deck.topcard.color
                    gamestate.preneur = player
                    player.draw(gamestate.table.deck)
                    gamestate.taking_turn = None
                    break
            if not gamestate.preneur:
                gamestate.taking_turn = 2
                for player in gamestate.iter_players:
                    if player.take(gamestate.table.deck.topcard, second_turn=True):
                        gamestate.atout = gamestate.table.deck.topcard.color
                        gamestate.preneur = player
                        player.draw(gamestate.table.deck)
                        gamestate.taking_turn = None
                        break
            if gamestate.taking_turn is None:
                raise Exception("No one took.")
        remaining_deck = Deck([card for card in init_deck().cards if
                               card not in self.state.table.trick + self.state.table.past_tricks
                               + self.state.current_player.hand.cards])
        remaining_deck.shuffle()
        # same game state
        # we use simple players to do a simulation
        gamestate.table.players = [SimpleBotPlayer("Bot" + str(i + 1)) for i in range(4)]
        # we shuffle the hands of the different players
        # @todo there should be constraints on what cards are in which hand to be correct
        for player in gamestate.table.players:
            for i in range(len(player.hand.cards)):
                player.draw(remaining_deck)
        result = play_sample_game(gamestate)
        return result


def f():
    deck = init_deck()
    deck.shuffle()
    table = Table(
        [HumanPlayer("Human_South"), SimpleBotPlayer("West"), SimpleBotPlayer("North"), SimpleBotPlayer("East")], deck)
    initial_state = Gamestate(table, random.choice(table.players))

    # Drawing phase
    for i in range(2):  # Two draw turns
        for player in initial_state.iter_players:
            n_card = 2 if i == 0 else 3
            for j in range(n_card):
                player.draw(deck)

    # for player in initial_state.iter_players:
    #     print(len(player.hand.cards))
    deck.reveal()

    root = MCTSNode(state=initial_state)
    move = look_for_best_move(root, time_limit=1)


def look_for_best_move(root: MCTSNode, time_limit: int = 1):
    initial_time = time.time()
    while time.time() - initial_time < time_limit:
        node = root
        while not node.is_leaf():
            node: MCTSNode = node.selection()
        # @todo make the code better
        if node.number_of_visits == 0:
            result = node.simulation()  # just call a rollout
            node.backpropagation(result)
        else:
            node.expansion()
            node: MCTSNode = node.selection()
            result = node.simulation()  # just call a rollout
            node.backpropagation(result)
    return root.select_best_child().parent_action
