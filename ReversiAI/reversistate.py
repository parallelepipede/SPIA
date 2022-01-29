# -*- coding: utf-8; mode: python -*-

# ENSICAEN
# École Nationale Supérieure d'Ingénieurs de Caen
# 6 Boulevard Maréchal Juin
# F-14050 Caen Cedex France
#
# Artificial Intelligence 2I1AE1

# @file reversistate.py
#
# @author Régis Clouard
# based on Mathias Broxvall's classes


class ReversiState:
    """ The class for representing a state of the game.
    
    grid: array integers: 0 = empty, 1 = first player, 2 = second player
    ply: next player to make a move, 1 or 2
    nMoves: the total number of moves performed since the start
    """

    def __init__(self, clone):
        if clone:
            self.grid = [X[:] for X in clone.grid]
            self.ply = clone.ply
            self.nMoves = clone.nMoves
        else:
            self.grid = [[0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 2, 1, 0, 0, 0],
                         [0, 0, 0, 1, 2, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0]]
            self.ply = 1
            self.nMoves = 0

    def __repr__(self):
        return "\n".join(str(self.grid[i]) for i in range(len(self.grid)))

    def __getitem__(self, X):
        return self.grid[X]

    def flips(self, x, y, dx, dy, player):
        """ Returns how many pieces we would flip if start at
        (x, y) going in the direction (dx, dy)
        """
        count = 0
        for i in range(1, 8):
            x2 = x + i * dx
            y2 = y + i * dy
            if x2 < 0 or y2 < 0 or x2 >= 8 or y2 >= 8:
                return 0
            elif self.grid[y2][x2] == 0:
                return 0
            elif self.grid[y2][x2] == player:
                return count
            else:
                count = count + 1;
        return 0

    def get_player(self):
        return self.ply - 1

    def get_flips(self, position):
        """Returns the number of flippable pieces from the position."""
        x, y = position
        count = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cnt = self.flips(x, y, dx, dy, self.ply)
                    count += cnt;
        return count

    def is_legal(self, x, y, player):
        """ Decides if a move is legal. """
        if self.grid[y][x] != 0:
            return False
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    cnt = self.flips(x, y, dx, dy, player)
                    if cnt > 0:
                        return True
        return False

    def move(self, position):
        """ Performs a game move, and returns the number of flipped pieces.
        """
        x, y = position
        me = self.ply
        he = (self.ply + 2) % 2 + 1
        count = 0
        self.nMoves = self.nMoves + 1

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx != 0 or dy != 0:
                    if self.flips(x, y, dx, dy, self.ply) > 0:
                        for i in range(1, 8):
                            x2 = x + i * dx
                            y2 = y + i * dy
                            if x2 < 0 or y2 < 0 or x2 >= 8 or y2 >= 8:
                                break
                            elif self.grid[y2][x2] != he:
                                break
                            else:
                                self.grid[y2][x2] = me
                                count += 1
        self.grid[y][x] = me
        self.ply = he
        return count

    def score(self):
        """ Returns the current score as a list (ply1, ply2).
        """
        score = [0, 0]
        for x in range(8):
            for y in range(8):
                p = self.grid[y][x]
                if p != 0:
                    score[p - 1] += 1
        return score

    def game_result(self):
        score = [0, 0]
        for x in range(8):
            for y in range(8):
                p = self.grid[y][x]
                if p != 0:
                    score[p - 1] += 1
        if score[self.get_player()] > score[self.get_player() - 1]:
            return 1
        elif score[self.get_player()] < score[self.get_player() - 1]:
            return 0
        else:
            return -1

    def legal_moves(self, player=None):
        """ Returns a list of all legal moves. """
        if not player:
            player = self.ply
        moves = []
        for x in range(8):
            for y in range(8):
                if self.is_legal(x, y, player):
                    moves.append((x, y))
        return moves

    def terminal_test(self):
        """ Returns true if the game is over.
        """
        return self.legal_moves() == []

    # @property
    # def rotated_states(self):
    #     grid = self.grid[::]
    #     rotated_states = []
    #     for _ in range(4):
    #         grid = list(zip(*grid[::-1]))
    #         rotated_states.append(grid)
    #     return rotated_states
