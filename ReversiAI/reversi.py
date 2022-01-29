#! /usr/bin/env python
# -*- coding: utf-8; mode: python -*-

# ENSICAEN
# École Nationale Supérieure d'Ingénieurs de Caen
# 6 Boulevard Maréchal Juin
# F-14050 Caen Cedex France
#
# Artificial Intelligence 2I1AE1

# @file reversi.py
#
# @author Régis Clouard
# based on Mathias Broxvall's classes 

## code to handle timeouts
import signal
import sys

from reversiframe import ReversiFrame
from reversistate import ReversiState
from agents import *


class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:

    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args):
        if not 'SIGALRM' in dir(signal):
            return self.function(*args)
        old = signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.timeout)
        try:
            result = self.function(*args)
        finally:
            signal.signal(signal.SIGALRM, old)
        signal.alarm(0)
        return result


class Reversi:
    def __init__(self, player1, player2, state=None, function=None):
        self.__player1 = player1
        self.__player2 = player2
        self.__heuristic_function = function
        self.__state = state

    def run(self):
        """
        Calls the selected player.
        """
        f = ReversiFrame(self.__player1, self.__player2, state=self.__state)
        f.mainloop()
        return f.state.score()[0] > f.state.score()[1]


def run_agents(player1, player2, timeout, mute, function=None, state=None):
    """ The real main. """
    reversi = Reversi(player1, player2, function=function, state=state)
    reversi.run()


def default(str):
    return str + ' [Default: %default]'


def read_command(argv):
    """ Processes the command used to run Reversi from the command line. """
    from optparse import OptionParser
    usageStr = """
    USAGE:      python reversi.py <options>
    EXAMPLES:   python reversi.py -1 ReversiGreedyAI -2 ReversiRandomAI
    """
    parser = OptionParser(usageStr)

    parser.add_option('-f', '--function', dest='function',
                      help='The heuristics to use', default='SimpleEvaluationFunction')
    parser.add_option('-n', '--moves', dest='moves',
                      help=default('Maximum number of look-ahead moves'), default=3)
    parser.add_option('-t', '--timeout', dest='timeout',
                      help=default('Maximum search time'), default=2000)
    parser.add_option('-m', '--mute', action='store_true', dest='mute',
                      help='Display only results', default=False)
    parser.add_option('-1', '--player1', dest='player1',
                      help=default('the player1'), default=None)
    parser.add_option('-2', '--player2', dest='player2',
                      help=default('the player2'), default=None)

    options, otherjunk = parser.parse_args(argv)

    if len(otherjunk) != 0:
        raise Exception('Command line input not understood: ' + str(otherjunk))
    args = dict()

    args['timeout'] = int(options.timeout)
    args['mute'] = options.mute

    # Choose a heuristic
    heuristics = None
    if options.function != None:
        try:
            module = __import__('agents')
            if options.function in dir(module):
                function = getattr(module, options.function)
                heuristics = function()
            else:
                raise Exception('Unknown heuristic: ' + options.function)
        except ImportError:
            raise Exception('No file agents.py')

    # Choose a Reversi player
    try:
        module = __import__('agents')
        if options.player1 == None:
            args['player1'] = None
        elif options.player1 in dir(module):
            player1 = getattr(module, options.player1)
            args['player1'] = player1(int(options.moves) * 2, heuristics)
        else:
            raise Exception('Unknown player: ' + options.player1)

        if options.player2 == None:
            args['player2'] = None
        elif options.player2 in dir(module):
            player2 = getattr(module, options.player2)
            args['player2'] = player2(int(options.moves) * 2, heuristics)
        else:
            raise Exception('Unknown player: ' + options.player2)
    except ImportError:
        raise Exception('No file agents.py')

    return args


if __name__ == '__main__':
    """ The main function called when reversi.py is run
    from the command line:

    > python reversi.py

    See the usage string for more details.

    > python reversi.py --help
    > python reversi.py -h """
    args = read_command(sys.argv[1:])  # Get game components based on input
    print("\n-------------------------------------------------------")
    for arg in sys.argv:
        print(arg, end=" ")
    print("\n-------------------------------------------------------")
    run_agents(**args)


def simulate_game(player1, player2, state):
    # random.seed(30)
    state = ReversiState(state)
    while state.legal_moves():
        if (state.ply - 1) % 2 == 0:
            player1.do_move(state)
        else:
            player2.do_move(state)
    return state.score()
