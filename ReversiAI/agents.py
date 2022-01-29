# -*- coding: utf-8; mode: python -*-
# @todo replace the reversi class by a grid, all info are already contained in the grid btw (maybe faster computations)
# ENSICAEN
# École Nationale Supérieure d'Ingénieurs de Caen
# 6 Boulevard Maréchal Juin
# F-14050 Caen Cedex France
#
# Artificial Intelligence 2I1AE1

# @file agents.py
#
# @author Régis Clouard
# based on Mathias Broxvall's classes
import random
import re
import threading
import time
from collections import defaultdict
from loguru import logger
import numpy as np

from utils import Flag

from reversistate import ReversiState


class Strategy:
    """ This is a base class for implementing strategies. """

    def do_move(self, state):
        """Determines next move. Derived classes must implement this."""
        raise Exception("Invalid Strategy class, do_move() not implemented")


class ReversiRandomAI(Strategy):
    """ The naive version of the AI.

    This is a naive AI that just makes a random choice between
    the possible legal moves.
    """

    def __init__(self, maxply, evaluation_function):
        pass

    def do_move(self, state):
        moves = state.legal_moves()
        move = moves[random.randrange(0, len(moves))]
        state.move(move)


class ReversiGreedyAI(Strategy):
    """ The greedy version of the AI.
    
    The best move is the one that flips the most disks.
    """

    def __init__(self, maxply, evaluation_function):
        pass

    def do_move(self, state):
        """
        Checks every move and picks the one with
        the most pieces belonging to player.
        """
        best_move = None
        best_count = -1
        for move in state.legal_moves():
            count = state.get_flips(move)
            if count > best_count:
                best_count = count
                best_move = move
        state.move(best_move)


class ReversiMinimaxAI(Strategy):
    """ This is an implementation of the minimax search algorithm.

    Finds the best move in the game, looking ahead maxply moves.
    The best move is determined by an external evaluation function.
    """

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        self.evaluation_function = evaluation_function
        print("Max ply = ", self.maxply)

    def do_move(self, state):
        """
        Finds the best move in the game, looking ahead maxply moves.
        The best move is determined by an external evaluation function.
        """
        maxply = self.maxply
        best_v = - 1e30
        mov = 0
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            v = self.min_value(next_state, maxply)
            if v > best_v:
                best_v = v
                mov = move
        state.move(mov)

    def max_value(self, state, maxply):
        maxply -= 1
        if state.terminal_test() or maxply == 0:
            return self.evaluation_function.eval(state, state.get_player())
        v = - 1e30
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            v = max(v, self.min_value(next_state, maxply))
        return v

    def min_value(self, state, maxply):
        maxply -= 1
        if state.terminal_test() or maxply == 0:
            return self.evaluation_function.eval(state, 1 - state.get_player())
        v = 1e30
        for move in state.legal_moves():
            nextState = ReversiState(state)
            nextState.move(move)
            v = min(v, self.max_value(nextState, maxply))
        return v


class EvaluationStrategy:
    """ 
    This strategy pattern defines the evaluation function used to evaluate a situation.
    """

    def eval(self, state: ReversiState, player: int):
        """  Returns the evaluation for the player with regard to the state. """
        raise Exception("Invalid Strategy class, EvaluationStrategy::eval() not implemented")


class SimpleEvaluationFunction(EvaluationStrategy):
    """ Evaluation function based only on the score. """

    def eval(self, state, player):
        return state.score()[player] - state.score()[1 - player]


class BetterEvaluationFunction(EvaluationStrategy):
    """ Evaluation function based only on the score, bigger number if the state is terminal. """

    def eval(self, state, player):
        score = state.score()[player] - state.score()[1 - player]
        if state.terminal_test():
            return 100 * score
        return score


class RandomEvaluationFunction(EvaluationStrategy):
    """ Evaluation function based on a random number, except if the game is in a terminal state. """

    def eval(self, state, player):
        if state.terminal_test():
            return state.score()[player] - state.score()[1 - player]
        return random.random()


class OtherEvaluationFunction(EvaluationStrategy):
    """ Evaluation function based on the score if the game is finished, on the number of possible moves else. """

    def eval(self, state, player):
        score = state.score()[player] - state.score()[1 - player]
        if state.terminal_test():
            return 100 * score
        else:
            return len(state.legal_moves(player))  # return random.randint(0, 100) gives a similar result


class MyEvaluationFunction(EvaluationStrategy):
    """Evaluation function based on the position of the pawns."""

    def eval(self, state, player):
        if state.terminal_test():
            return (state.score()[player] - state.score()[1 - player]) * 100
        matrix = [[4, -3, 2, 2, 2, 2, -3, 4],
                  [-3, -4, -1, -1, -1, -1, -4, -3],
                  [2, -1, 1, 0, 0, 1, -1, 2],
                  [2, -1, 0, 1, 1, 0, -1, 2],
                  [2, -1, 0, 1, 1, 0, -1, 2],
                  [2, -1, 1, 0, 0, 1, -1, 2],
                  [-3, -4, -1, -1, -1, -1, -4, -3],
                  [4, -3, 2, 2, 2, 2, -3, 4]]
        sum = 0
        for i in range(8):
            for j in range(8):
                if state.grid[i][j] == player + 1:
                    sum += matrix[i][j]
                if state.grid[i][j] == 1 - player + 1:
                    sum -= matrix[i][j]
        return sum


class ReversiAlphaBetaAI(Strategy):
    """
    This is an implementation of the alpha-beta search algorithm.
    """

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        self.evaluation_function = evaluation_function
        print("Max ply =", self.maxply)

    def do_move(self, state):
        """
        Finds the best move in the game, looking ahead maxply moves.
        The best move is determined by an external evaluation function.
        """
        maxply = self.maxply
        best_v = - 1e30
        mov = 0
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            v = self.min_value(next_state, maxply, -1e30, 1e30)
            if v > best_v:
                best_v = v
                mov = move
        state.move(mov)

    def max_value(self, state, maxply, alpha, beta):
        if state.terminal_test() or maxply == 1:
            return self.evaluation_function.eval(state, state.get_player())
        v = - 1e30
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            v = max(v, self.min_value(next_state, maxply - 1, alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(self, state, maxply, alpha, beta):
        if state.terminal_test() or maxply == 1:
            return self.evaluation_function.eval(state, 1 - state.get_player())
        v = 1e30
        for move in state.legal_moves():
            nextState = ReversiState(state)
            nextState.move(move)
            v = min(v, self.max_value(nextState, maxply - 1, alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v


class ReversiNegaMaxAI(Strategy):
    """
    This is an implementation of the negamax search algorithm.
    """

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        self.evaluation_function = evaluation_function
        print("Max ply =", self.maxply)
        self.__iterative_dictionary = {}

    def do_move(self, state: ReversiState):
        maxply = self.maxply
        v, best_move = -1e30, None
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            new_v = -self.negamax(next_state, maxply - 1)
            if new_v > v:
                v, best_move = new_v, move
        state.move(best_move)

    def negamax(self, state: ReversiState, maxply: int):  # maxply is the same as depth
        if state.terminal_test() or maxply == 1:
            if state.terminal_test():
                return self.evaluation_function.eval(state, state.get_player())
            else:
                f = MyEvaluationFunction()
                return f.eval(state, state.get_player())
        v = - 1e30
        legal_moves = state.legal_moves()
        for move in legal_moves:
            next_state = ReversiState(state)
            next_state.move(move)
            v = max(v, -self.negamax(next_state, maxply - 1))
        return v


class ReversiNegaMaxPruningAI(Strategy):
    """
    This is an implementation of the negamax search algorithm with alpha-beta pruning.
    """

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        self.evaluation_function = evaluation_function
        print("Max ply =", self.maxply)
        self.__iterative_dictionary = {}

    def do_move(self, state: ReversiState):
        maxply = self.maxply
        v, best_move = -1e30, None
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            new_v = -self.negamax(next_state, maxply - 1, -1e30, 1e30)
            if new_v > v:
                v, best_move = new_v, move
        state.move(best_move)

    def negamax(self, state: ReversiState, maxply: int, alpha, beta):  # maxply is the same as depth
        v = - 1e30
        legal_moves = state.legal_moves()
        if len(legal_moves) == 0 or maxply == 1:
            return self.evaluation_function.eval(state, state.get_player())
        for move in legal_moves:
            next_state = ReversiState(state)
            next_state.move(move)
            v = max(v, -self.negamax(next_state, maxply - 1, -beta, -alpha))
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return alpha


class TranspositionTable(dict):
    def __hash(self, state):
        # TODO: Prefer the Zobrist hash?
        # @todo optimize with rotations
        return re.sub('\ |\[|\]|\,', '', str(state.grid))

    def get_value(self, state, default_value):
        key = self.__hash(state)
        if key in self:
            return self[key]
        else:
            return default_value

    def set_value(self, state: ReversiState, entry: dict):
        key = self.__hash(state)
        self[key] = entry


class ReversiNegaMaxPruningTTAI(Strategy):
    """This is an implementation of the negamax algorithm with alpha-beta pruning and transposition tables."""

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        self.evaluation_function = evaluation_function
        print("Max ply =", self.maxply)
        self.__transposition_table = TranspositionTable()

    def do_move(self, state: ReversiState):
        maxply = self.maxply
        best_move, value = None, -1e30
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            v = -self.negamax(next_state, maxply - 1, -1e30, 1e30)
            if v > value:
                best_move, value = move, v
        state.move(best_move)

    def negamax(self, state: ReversiState, maxply: int, alpha, beta):  # maxply is the same as depth
        already_defined = False
        alpha_orig = alpha
        tt_entry: TranspositionTable = self.__transposition_table.get_value(state, None)
        if tt_entry is not None and tt_entry["depth"] >= maxply:
            already_defined = True
            if tt_entry["flag"] == Flag.EXACT:
                return tt_entry["value"]
            elif tt_entry["flag"] == Flag.LOWERBOUND:
                alpha = max(alpha, tt_entry["value"])
            elif tt_entry["flag"] == Flag.UPPERBOUND:
                beta = min(beta, tt_entry["value"])
            if alpha >= beta:
                return tt_entry["value"]
        if maxply == 1 or state.terminal_test():
            return self.evaluation_function.eval(state, state.get_player())
        if tt_entry is None:
            tt_entry = {"depth": None, "value": None, "flag": None}
        value = -999
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            value = max(value, -self.negamax(next_state, maxply - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        tt_entry["value"] = value
        if value <= alpha_orig:
            tt_entry["flag"] = Flag.UPPERBOUND
        elif value >= beta:
            tt_entry["flag"] = Flag.LOWERBOUND
        else:
            tt_entry["flag"] = Flag.EXACT
        tt_entry["depth"] = maxply
        if already_defined:
            self.__transposition_table.set_value(state, tt_entry)
        return value


class ReversiIDTNegaMaxPruningTTAI(Strategy):
    """
    Iterative Deepening Threaded negamax with Transposition Table.
    This class has a time limit on each of its moves.
    """

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        self.evaluation_function = evaluation_function
        print("Max time to play = {} seconds".format(self.maxply))
        self.__transposition_table = TranspositionTable()

    def do_move(self, state: ReversiState):
        # logger.info("IDTNegaMaxPruningTT to play")
        max_time = self.maxply  # seconds
        depth = 0
        move = [state.legal_moves()[0]]
        stop_threads = False
        p = threading.Thread(target=self.launch_negamax, args=(state, depth, move, lambda: stop_threads))
        p.start()
        p.join(max_time)
        stop_threads = True
        state.move(move[0])

    def launch_negamax(self, state: ReversiState, depth: int, best_move, stop):
        value = -1e30
        while True:
            depth += 2
            for move in state.legal_moves():
                if stop() or depth > 64 - state.nMoves + 4:  # tricks from https://stackoverflow.com/questions/323972/is-there-any-way-to-kill-a-thread
                    logger.debug("Depth {} reached".format(depth))
                    return
                next_state = ReversiState(state)
                next_state.move(move)
                v = -self.negamax(next_state, depth - 1, -1e30, 1e30)
                if v > value:
                    best_move[0], value = move, v

    def negamax(self, state: ReversiState, maxply: int, alpha, beta):  # maxply is the same as depth
        already_defined = False
        alpha_orig = alpha
        tt_entry: TranspositionTable = self.__transposition_table.get_value(state, None)
        if tt_entry is not None and tt_entry["depth"] >= maxply:
            already_defined = True
            if tt_entry["flag"] == Flag.EXACT:
                return tt_entry["value"]
            elif tt_entry["flag"] == Flag.LOWERBOUND:
                alpha = max(alpha, tt_entry["value"])
            elif tt_entry["flag"] == Flag.UPPERBOUND:
                beta = min(beta, tt_entry["value"])
            if alpha >= beta:
                return tt_entry["value"]
        if maxply == 1 or state.terminal_test():
            return self.evaluation_function.eval(state, state.get_player())
        if tt_entry is None:
            tt_entry = {"depth": None, "value": None, "flag": None}
        value = -999
        for move in state.legal_moves():
            next_state = ReversiState(state)
            next_state.move(move)
            value = max(value, -self.negamax(next_state, maxply - 1, -beta, -alpha))
            alpha = max(alpha, value)
            if alpha >= beta:
                break
        tt_entry["value"] = value
        if value <= alpha_orig:
            tt_entry["flag"] = Flag.UPPERBOUND
        elif value >= beta:
            tt_entry["flag"] = Flag.LOWERBOUND
        else:
            tt_entry["flag"] = Flag.EXACT
        tt_entry["depth"] = maxply
        if already_defined:
            self.__transposition_table.set_value(state, tt_entry)
        # print(value)
        return value


def eval(state: ReversiState):
    """
    Heuristic function for the current state.
    """
    player = state.get_player()
    if state.terminal_test():
        return (state.score()[player] - state.score()[1 - player]) * 100
    matrix = [[4, -3, 2, 2, 2, 2, -3, 4],
              [-3, -4, -1, -1, -1, -1, -4, -3],
              [2, -1, 1, 0, 0, 1, -1, 2],
              [2, -1, 0, 1, 1, 0, -1, 2],
              [2, -1, 0, 1, 1, 0, -1, 2],
              [2, -1, 1, 0, 0, 1, -1, 2],
              [-3, -4, -1, -1, -1, -1, -4, -3],
              [4, -3, 2, 2, 2, 2, -3, 4]]
    score = {1: 0, 0: 0}
    for i in range(8):
        for j in range(8):
            p = state.grid[i][j]
            if p != 0:
                score[p - 1] += matrix[i][j]
    return score[player] - score[1 - player]


class ReversiMonteCarloTreeSearchAI(Strategy):
    """
    This is an implementation of the Monte-Carlo Tress Search algorithm.
    """

    def __init__(self, maxply, evaluation_function):
        self.maxply = 6
        self.evaluation_function = evaluation_function

    def do_move(self, state: ReversiState):
        root_node = MonteCarloTreeSearchNode(state)
        selected_node = root_node.best_action()
        state.move(selected_node.parent_action)


class ReversiMonteCarloTreeSearchTLAI(Strategy):
    """This is an implementation of the Monte-Carlo Tress Search algorithm with a time limit."""

    def __init__(self, maxply, evaluation_function):
        self.maxply = maxply
        print("Max time to play = {} seconds".format(self.maxply))
        self.evaluation_function = evaluation_function

    def do_move(self, state: ReversiState):
        # logger.info("MonteCarloTreeSearchTL to play")
        max_time = self.maxply  # seconds
        move = [None]
        stop_threads = False
        root_node = MonteCarloTreeSearchNode(state)
        p = threading.Thread(target=root_node.best_action, args=(move, lambda: stop_threads))
        p.start()
        p.join(max_time)
        stop_threads = True
        if move[0] is None:
            state.move(state.legal_moves()[0])
        else:
            state.move(move[0].parent_action)


class MonteCarloTreeSearchNode:
    def __init__(self, state, parent=None, parent_action=None, player=None):
        self.state: ReversiState = state
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        # depth is 0 if us, 1 else
        self.player = self.state.get_player()
        return

    def untried_actions(self):
        self._untried_actions = self.state.legal_moves()
        return self._untried_actions

    def q(self):
        wins = self._results[1]
        loses = self._results[-1]
        return wins - loses

    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        next_state = ReversiState(self.state)
        next_state.move(action)
        # next_state = self.state.move(action) # original line instead of the two previous ones
        child_node = MonteCarloTreeSearchNode(next_state, parent=self, parent_action=action,
                                              player=(self.player + 1) % 2)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.terminal_test()

    def rollout(self):
        current_rollout_state = ReversiState(self.state)
        while not current_rollout_state.terminal_test():
            possible_moves = current_rollout_state.legal_moves()
            action = self.rollout_policy(possible_moves)
            current_rollout_state.move(action)
        result = current_rollout_state.game_result()
        if result == self.player:
            result = 1
        elif result == -1:
            # then it's a draw
            result = 0
        else:
            result = -1
        return result

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(-1 * result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1):
        choices_weights = [(child.q() / child.n()) + c_param * np.sqrt((2 * np.log(self.n()) / child.n())) for child in
                           self.children]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, move=None, stop=None):
        if stop is None and move is None:
            simulation_no = 10000
            for _ in range(simulation_no):
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
            return self.best_child(c_param=0.)
        else:
            n_iterations = 0
            while True:
                if stop() or n_iterations >= 50000:
                    logger.debug("{} iterations done".format(n_iterations))
                    return
                if len(self.state.legal_moves()) == 1:
                    # logger.info("Only one move possible here")
                    move[0] = MonteCarloTreeSearchNode(state=self.state, parent_action=self.state.legal_moves()[0])
                    return
                v = self._tree_policy()
                reward = v.rollout()
                v.backpropagate(reward)
                n_iterations += 1
                if n_iterations % 100 == 0:
                    move[0] = self.best_child(c_param=0.)
