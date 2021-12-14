from unittest import TestCase

from src.cards import *

random.seed(0)


def full_ordered_deck() -> Deck:
    return Deck([Card(i, j) for i in Color for j in Rank])


class TestCard(TestCase):
    def test_color(self):
        card = Card(Color.CLUBS, Rank.KING)
        assert card.color == Color.CLUBS
        assert card.color != Color.HEARTS
        assert card.color != Color.SPADES
        assert card.color != Color.DIAMONDS

    def test_rank(self):
        card = Card(Color.CLUBS, Rank.KING)
        assert card.rank == Rank.KING

    def test_higher_than_1(self):
        card1 = Card(Color.SPADES, Rank.ACE)
        card2 = Card(Color.SPADES, Rank.KING)
        trump = Color.CLUBS
        assert card1.higher_than(card2, trump)
        assert not card2.higher_than(card1, trump)

    def test_higher_than_2(self):
        card1 = Card(Color.SPADES, Rank.ACE)
        card2 = Card(Color.SPADES, Rank.KING)
        trump = Color.SPADES
        assert card1.higher_than(card2, trump)
        assert not card2.higher_than(card1, trump)

    def test_higher_than_3(self):
        card1 = Card(Color.SPADES, Rank.SEVEN)
        card2 = Card(Color.SPADES, Rank.JACK)
        trump = Color.CLUBS
        assert not card1.higher_than(card2, trump)
        assert card2.higher_than(card1, trump)

    def test_higher_than_4(self):
        card1 = Card(Color.SPADES, Rank.TEN)
        card2 = Card(Color.SPADES, Rank.JACK)
        trump = Color.SPADES
        assert not card1.higher_than(card2, trump)
        assert card2.higher_than(card1, trump)

    def test_higher_than_5(self):
        card1 = Card(Color.SPADES, Rank.TEN)
        card2 = Card(Color.HEARTS, Rank.JACK)
        trump = Color.HEARTS
        assert not card1.higher_than(card2, trump)
        assert card2.higher_than(card1, trump)

    def test_higher_than_6(self):
        card1 = Card(Color.SPADES, Rank.TEN)
        card2 = Card(Color.HEARTS, Rank.JACK)
        trump = Color.CLUBS
        assert card1.higher_than(card2, trump)
        assert not card2.higher_than(card1, trump)


class TestDeck(TestCase):
    def test_draw(self):
        self.deck = full_ordered_deck()
        drawn_card = self.deck.draw()
        assert drawn_card == Card(Color.SPADES, Rank.ACE)
        self.deck = None

    def test_shuffle(self):
        self.deck = full_ordered_deck()
        random.shuffle(self.deck.cards)
        assert self.deck.cards[0] == Card(Color.CLUBS, Rank.TEN)
        self.deck = None

    def test_reveal(self):
        self.deck = full_ordered_deck()
        self.deck.reveal()
        assert self.deck.topcard == Card(Color.SPADES, Rank.ACE)
        self.deck = None


class TestTrick(TestCase):
    def test_get_winner_1(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.TEN))
        trick.add(Card(Color.SPADES, Rank.ACE))
        trick.add(Card(Color.SPADES, Rank.JACK))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_winner(Color.SPADES) == 2

    def test_get_winner_2(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.TEN))
        trick.add(Card(Color.SPADES, Rank.ACE))
        trick.add(Card(Color.SPADES, Rank.JACK))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_winner(Color.CLUBS) == 1

    def test_get_winner_3(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.CLUBS, Rank.ACE))
        trick.add(Card(Color.SPADES, Rank.ACE))
        trick.add(Card(Color.HEARTS, Rank.JACK))
        trick.add(Card(Color.DIAMONDS, Rank.SEVEN))
        assert trick.get_winner(Color.HEARTS) == 2

    def test_get_winner_4(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.QUEEN))
        trick.add(Card(Color.SPADES, Rank.KING))
        trick.add(Card(Color.DIAMONDS, Rank.EIGHT))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_winner(Color.CLUBS) == 1

    def test_get_points_1(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.TEN))
        trick.add(Card(Color.SPADES, Rank.ACE))
        trick.add(Card(Color.SPADES, Rank.JACK))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_points(Color.SPADES) == 41

    def test_get_points_2(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.TEN))
        trick.add(Card(Color.SPADES, Rank.ACE))
        trick.add(Card(Color.SPADES, Rank.JACK))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_points(Color.CLUBS) == 23

    def test_get_points_3(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.NINE))
        trick.add(Card(Color.SPADES, Rank.ACE))
        trick.add(Card(Color.SPADES, Rank.JACK))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_points(Color.SPADES) == 45

    def test_get_points_4(self):
        trick = Trick(BasicPlayer("Test"))
        trick.add(Card(Color.SPADES, Rank.QUEEN))
        trick.add(Card(Color.SPADES, Rank.KING))
        trick.add(Card(Color.DIAMONDS, Rank.EIGHT))
        trick.add(Card(Color.SPADES, Rank.SEVEN))
        assert trick.get_points(Color.SPADES) == 7

    def test_get_legal_moves_1(self):
        # no card in the trick
        player = BasicPlayer("Test")
        trump = Color.DIAMONDS
        player.hand = Hand([Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN), Card(Color.CLUBS, Rank.ACE),
                            Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN), Card(Color.CLUBS, Rank.ACE),
                                Card(Color.SPADES, Rank.NINE)]
        trick = Trick(player)
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_2(self):
        # a card in the trick, must play good color
        player = BasicPlayer("Test")
        trump = Color.DIAMONDS
        player.hand = Hand([Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN), Card(Color.CLUBS, Rank.ACE),
                            Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.CLUBS, Rank.ACE)]
        trick = Trick(player)
        trick.add(Card(Color.CLUBS, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_3(self):
        # a trump in the trick, must play a trump
        player = BasicPlayer("Test")
        trump = Color.SPADES
        player.hand = Hand([Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN), Card(Color.CLUBS, Rank.ACE),
                            Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN),
                                Card(Color.SPADES, Rank.NINE)]
        trick = Trick(player)
        trick.add(Card(Color.SPADES, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_4(self):
        # a trump in the trick, must play a higher trump
        player = BasicPlayer("Test")
        trump = Color.SPADES
        player.hand = Hand([Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.SEVEN), Card(Color.CLUBS, Rank.ACE),
                            Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.NINE)]
        trick = Trick(player)
        trick.add(Card(Color.SPADES, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_5(self):
        # a card an a trump in the trick, must play a higher trump
        player = BasicPlayer("Test")
        trump = Color.SPADES
        player.hand = Hand(
            [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.SEVEN), Card(Color.DIAMONDS, Rank.ACE),
             Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.NINE)]
        trick = Trick(player)
        trick.add(Card(Color.CLUBS, Rank.EIGHT))
        trick.add(Card(Color.SPADES, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_6(self):
        # a card an a trump in the trick, must play whatever
        player = BasicPlayer("Test")
        trump = Color.HEARTS
        player.hand = Hand(
            [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.SEVEN), Card(Color.DIAMONDS, Rank.ACE),
             Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.SEVEN),
                                Card(Color.DIAMONDS, Rank.ACE), Card(Color.SPADES, Rank.NINE)]
        trick = Trick(player)
        trick.add(Card(Color.CLUBS, Rank.EIGHT))
        trick.add(Card(Color.HEARTS, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_7(self):
        # a card an a trump in the trick, must play the same color as the card
        player = BasicPlayer("Test")
        trump = Color.HEARTS
        player.hand = Hand([Card(Color.SPADES, Rank.ACE), Card(Color.CLUBS, Rank.SEVEN), Card(Color.DIAMONDS, Rank.ACE),
                            Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.CLUBS, Rank.SEVEN)]
        trick = Trick(player)
        trick.add(Card(Color.CLUBS, Rank.EIGHT))
        trick.add(Card(Color.HEARTS, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_8(self):
        # a card an a trump in the trick, must play a small trump
        player = BasicPlayer("Test")
        trump = Color.HEARTS
        player.hand = Hand(
            [Card(Color.SPADES, Rank.ACE), Card(Color.HEARTS, Rank.SEVEN), Card(Color.DIAMONDS, Rank.ACE),
             Card(Color.SPADES, Rank.NINE)])
        expected_legal_moves = [Card(Color.HEARTS, Rank.SEVEN)]
        trick = Trick(player)
        trick.add(Card(Color.CLUBS, Rank.EIGHT))
        trick.add(Card(Color.HEARTS, Rank.EIGHT))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)

    def test_get_legal_moves_9(self):
        # two trumps in the trick, everything is playable
        player = BasicPlayer("Test")
        trump = Color.HEARTS
        player.hand = Hand([Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.KING), Card(Color.DIAMONDS, Rank.TEN),
                            Card(Color.DIAMONDS, Rank.SEVEN)])
        expected_legal_moves = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.KING),
                                Card(Color.DIAMONDS, Rank.TEN), Card(Color.DIAMONDS, Rank.SEVEN)]
        trick = Trick(player)
        trick.add(Card(Color.HEARTS, Rank.QUEEN))
        trick.add(Card(Color.HEARTS, Rank.ACE))
        legal_moves = trick.get_legal_moves(player.hand, trump)
        assert equal_ignore_order(expected_legal_moves, legal_moves)


class TestPlayer(TestCase):
    def test_play_card_1(self):
        player = BasicPlayer("Test")
        player.hand = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN), Card(Color.CLUBS, Rank.ACE),
                       Card(Color.SPADES, Rank.NINE)]
        expected_card_played = Card(Color.SPADES, Rank.NINE)
        trick = Trick(player)
        player.play_card(trick, Color.CLUBS)
        assert trick.cards[0] == expected_card_played
        for card in player.hand:
            assert card != expected_card_played

    def test_play_card_2(self):
        player = BasicPlayer("Test")
        player.hand = [Card(Color.SPADES, Rank.ACE), Card(Color.SPADES, Rank.TEN), Card(Color.CLUBS, Rank.ACE),
                       Card(Color.SPADES, Rank.NINE)]
        expected_card_played = Card(Color.CLUBS, Rank.ACE)
        trick = Trick(player)
        trick.add(Card(Color.CLUBS, Rank.NINE))
        player.play_card(trick, Color.SPADES)
        assert trick.cards[1] == expected_card_played
        for card in player.hand:
            assert card != expected_card_played
