from scorecard import Scorecard
from player import Player
from random import randint
from dice_util import get_counts_from_dice, get_dice_from_counts


class Yahtzee:
    def __init__(self, players: list[Player]):
        self.num_players = len(players)
        self.players = players
        self.num_dice = 5
        self.num_sides = 6
        self.num_rolls = 3
        self.turn = 1
        self.player_turn = 0
        self.dice = [0] * self.num_sides
        self.rolls = 0
        self.score_cards = [Scorecard() for _ in range(self.no_of_players)]

    def play_turn(self):
        for player_turn, player in enumerate(self.players):
            self.player_turn = player_turn
            self.roll_dice()
            action = player.play()

        pass

    def roll_dice(self, dice_to_keep):
        dice = []
        if dice_to_keep:
            dice = get_dice_from_counts(dice_to_keep, self.num_sides)
        for i in range(self.num_dice - len(dice)):
            dice.append(randint(1, self.num_sides))

        self.rolls += 1
        self.dice = get_counts_from_dice(dice)
