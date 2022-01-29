# -*- coding: utf-8; mode: python -*-

# ENSICAEN
# École Nationale Supérieure d'Ingénieurs de Caen
# 6 Boulevard Maréchal Juin
# F-14050 Caen Cedex France
#
# Artificial Intelligence 2I1AE1

# @file utils.py
#
# @author Régis Clouard
import random
import sys
import inspect
from enum import Enum

# from agents import Strategy


class Flag(Enum):
    """
    Class used for the flags in the transposition table.
    """
    EXACT, LOWERBOUND, UPPERBOUND = range(3)


def raiseNotDefined():
    fileName = inspect.stack()[1][1]
    line = inspect.stack()[1][2]
    method = inspect.stack()[1][3]

    print("*** Method not implemented: %s at line %s of %s" % (method, line, fileName))
    sys.exit(1)


def rotate(matrix):
    return list(zip(*matrix[::-1]))


