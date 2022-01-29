#! /usr/bin/env python
# -*- coding: utf-8; mode: python -*-

# ENSICAEN
# École Nationale Supérieure d'Ingénieurs de Caen
# 6 Boulevard Maréchal Juin
# F-14050 Caen Cedex France
#
# Artificial Intelligence 2I1AE1

# @file compete.py
#
# @author Régis Clouard
# based on Mathias Broxvall's classes 

import sys
import agents
from reversistate import ReversiState
import random


def compete(player1, player2, timeout, mute, function=None):
    """ Organizes a contest between 2 IA.

    This function performs 10 matches between the two given
    AI's and reports the total score. 
    The AI's take turn playing black/white to
    compensate for any irregularities 
    """
    score = [0, 0]
    draws = 0
    for i in range(0, 10):
        state = ReversiState(None)
        while state.legal_moves():
            if (state.ply - 1 + i) % 2 == 0:
                player1.do_move(state)
            else:
                player2.do_move(state)
        s = state.score()
        print("  Game finished: %d %d" % (s[i % 2], s[(i + 1) % 2]))

        if s[0] > s[1]:
            score[i % 2] += 1
        elif s[1] > s[0]:
            score[(1 + i) % 2] += 1
        else:
            draws += 1
    resulting_string = "Total scores: AI #1 (%s) %d wins vs. AI #2 (%s) %d wins. (%d draws)" % (
        type(player1).__name__, score[0], type(player2).__name__, score[1], draws)
    print(resulting_string)


def default(d):
    return d + ' [Default: %default]'


def read_command(argv):
    """ Processes the command used to run compete from the command line. """
    from optparse import OptionParser
    usage_str = """
    USAGE:      python compete.py <options>
    EXAMPLES:   python compete.py -1 ReversiGreedyAI -2 ReversiRandomAI
    """
    parser = OptionParser(usage_str)

    parser.add_option('-f', '--function', dest='function',
                      help='The heuristic to use', default='SimpleEvaluationFunction')
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
    if options.function is not None:
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
        if options.player1 is None:
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
    """ The main function called when compete.py is run
    from the command line:

    > python compete.py

    See the usage string for more details.

    > python compete.py --help
    > python compete.py -h """
    args = read_command(sys.argv[1:])  # Get game components based on input
    print("\n-------------------------------------------------------")
    for arg in sys.argv:
        print(arg, end=" ")
    print("\n-------------------------------------------------------")
    compete(**args)
