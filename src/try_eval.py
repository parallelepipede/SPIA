from src.cards import *
from itertools import combinations

def all_decks():
    full_deck = [Card(i, j) for i in Color for j in Rank]
    return combinations(full_deck, 6)




def eval(deck, topcard: Card) -> int:
    trump_list = [card for card in deck if card.color == topcard.color] + [topcard]
    trick = Trick(Player("Val"))
    for card in trump_list:
        trick.add(card)
    v1 = trick.get_points(topcard.color)
    v2 = 0
    for card in deck:
        if card not in trump_list:
            v2 += 10 * (card.rank == Rank.ACE)
    return v1 + v2


evals = []
for hand in all_decks():
    evals.append(eval(hand[:-1], hand[-1]))
