from cards import *


class Table:
    def __init__(self, players: List[BasicPlayer], deck: Deck) -> None:
        self.players = players
        self.teams = [Team("Team NS", [players[0], players[2]]), Team("Team EO", [players[1], players[3]])]
        self.deck = deck
        self.past_tricks = []
        self.trick = None

    def __str__(self) -> str:
        return ("{ players: " + list_str(self.players) + ", \n"#trick: " + str(self.trick) +
                ", \ndeck: " + str(self.deck) + "}")

