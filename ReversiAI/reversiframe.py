# -*- coding: utf-8; mode: python -*-

# ENSICAEN
# École Nationale Supérieure d'Ingénieurs de Caen
# 6 Boulevard Maréchal Juin
# F-14050 Caen Cedex France
#
# Artificial Intelligence 2I1AE1

# @file reversiframe.py
#
# @author Régis Clouard
# based on Mathias Broxvall's classes 


import sys

if sys.version_info.major >= 3:
    import tkinter as Tkinter
else:
    import Tkinter
from math import *
import random
import time
from tkinter import font

from reversistate import ReversiState


class ReversiFrame(Tkinter.Frame):
    """ The game frame.
    
    The frame for showing the game board
    and interacting with the human player
    """

    def __init__(self, ai1, ai2, state=None):
        """
        Creates an instance of the game using the given AI's.
        Pass in None in place of an AI to use the human player instead
        """
        Tkinter.Frame.__init__(self, None)
        self.master.title('Reversi')
        self.canvas = Tkinter.Canvas(self, width=64 * 8, height=64 * 8 + 60, bg='white')
        self.canvas.pack(expand=1, anchor=Tkinter.CENTER)
        self.pack()
        self.tkraise()
        self.state = ReversiState(state)
        self.images = {}
        self.font = font.Font(size=18)
        self.is_human = [0, 0]
        if ai1 is None:
            self.is_human[0] = 1
        if ai2 is None:
            self.is_human[1] = 1
        self.ai = [ai1, ai2]
        for img in ['ply1', 'ply2', 'cell', 'cell-hi']:
            self.images[img] = Tkinter.PhotoImage(file='./images/' + img + '.gif')
        self.text_item = self.canvas.create_text(40, 8 * 64, anchor=Tkinter.NW, text='', font=self.font)

        self.canvas.bind("<Button-1>", self.mouse_click)
        self.draw()

        if not self.is_human[self.state.ply - 1]:
            self.after(100, self.run_AI)

    def mouse_click(self, event):
        """ Accepts mouseclicks and interprets them as the humans move,
        if legal and if it is his turn
        """
        x = int(event.x / 64)
        y = int(event.y / 64)
        if x >= 0 and x < 8 and y >= 0 and y < 8 and self.is_human[self.state.ply - 1] and self.state.is_legal(x, y,
                                                                                                               self.state.ply):
            self.state.move((x, y))
            self.draw()
            self.update()
            moves = self.state.legal_moves()
            if moves == []:
                self.announce_winner()
                return None
            if not self.is_human[self.state.ply - 1]:
                self.after(100, self.run_AI)

    def run_AI(self):
        """ Runs the next AI one step. """
        self.configure(cursor="watch")
        self.update()
        self.ai[self.state.ply - 1].do_move(self.state)
        self.draw()
        self.configure(cursor="")
        self.update()
        moves = self.state.legal_moves()
        if moves == []:
            self.announce_winner()
            return None
        if not self.is_human[self.state.ply - 1]:
            self.after(10, self.run_AI)

    def announce_winner(self):
        """ Prints who is the winner."""
        score = self.state.score()
        if score[0] > score[1]:
            text = "Black (player #1) is the winner!"
        elif score[1] > score[0]:
            text = "White (player #2) is the winner!"
        else:
            text = "The game is a draw"
        self.canvas.itemconfigure(self.text_item, text="%s\nScore: %d / %d" % (text, score[0], score[1]))

    def draw(self):
        """ Draws the gameboard to the screen. """
        moves = self.state.legal_moves()
        for y in range(len(self.state.grid)):
            for x in range(len(self.state.grid[0])):
                if self.state[y][x] == 1:
                    im = self.images['ply1']
                elif self.state[y][x] == 2:
                    im = self.images['ply2']
                elif (x, y) in moves:
                    im = self.images['cell-hi']
                else:
                    im = self.images['cell']
                self.canvas.create_image(x * 64, y * 64, image=im, anchor=Tkinter.NW)
        score = self.state.score()
        if self.state.ply == 1:
            player = 'Black'
        else:
            player = 'White'
        self.canvas.itemconfigure(self.text_item, text="%s's move\nScore: %d / %d" % (player, score[0], score[1]))
