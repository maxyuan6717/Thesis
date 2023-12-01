from scorecard import Scorecard
from player import Player
from random import randint
from dice_util import get_dice_from_counts


class Yahtzee:
    def __init__(self, players: list[Player], init_num_games: int = 1):
        self.num_dice = 5
        self.num_sides = 6
        self.num_rolls = 3

        self.players = players
        self.num_players = len(players)

        self.num_games = init_num_games
        self.games_played = 0
        self.games_won = [0.0] * self.num_players
        self.game_scores = [[] for _ in range(self.num_players)]

        self.turn = 1
        self.score_cards = [Scorecard() for _ in range(self.num_players)]

        self.player_turn = 0
        self.dice_combo = [0] * self.num_sides
        self.dice = [0] * self.num_dice
        self.rolls = 0

        self.tot_category_scores = [0 for _ in range(13)]

    def reset_turn(self):
        self.rolls = 0
        self.dice_combo = [0] * self.num_sides
        self.dice = [0] * self.num_dice

    def play_player_turn(self, player_turn, player):
        self.player_turn = player_turn
        self.reset_turn()
        self.roll_dice()
        while True:
            turn_state = (self.rolls, self.dice_combo)
            action = player.get_action(self)
            if isinstance(action, int):
                score = self.score_cards[player_turn].score(action, self.dice_combo)
                self.tot_category_scores[action] += score
                break
            else:
                self.roll_dice(action[1])

    def play_player_action(self, player_turn, action):
        if isinstance(action, int):
            return self.score_cards[player_turn].score(action, self.dice_combo)
        else:
            self.roll_dice(action[1])
            return 0

    def play_players_turn(self):
        for player_turn, player in enumerate(self.players):
            self.play_player_turn(player_turn, player)

    def roll_dice(self, dice_to_keep_combo: tuple[int, ...] = (0, 0, 0, 0, 0, 0)):
        num_remaining_dice = self.num_dice - sum(dice_to_keep_combo)
        new_dice_combo = list(dice_to_keep_combo)
        for _ in range(num_remaining_dice):
            new_dice_combo[randint(0, self.num_sides - 1)] += 1
        self.dice_combo = tuple(new_dice_combo)
        self.dice = get_dice_from_counts(self.dice_combo)
        self.rolls += 1

    def play_game(self):
        self.turn = 1
        self.score_cards = [Scorecard() for _ in range(self.num_players)]
        while self.turn <= 13:
            self.play_players_turn()
            self.turn += 1

        winning_score = 0
        winning_players = []
        for player_num in range(self.num_players):
            player_score = self.score_cards[player_num].get_final_score()
            self.game_scores[player_num].append(player_score)
            if player_score > winning_score:
                winning_score = player_score
                winning_players = [player_num]
            elif player_score == winning_score:
                winning_players.append(player_num)

        for player_num in winning_players:
            self.games_won[player_num] += 1.0 / len(winning_players)

        self.games_played += 1

    def play_games(self):
        while self.games_played < self.num_games:
            self.play_game()
            percentage = self.games_played / self.num_games * 100

            if percentage > 0 and percentage % 10 == 0:
                print(f"{int(percentage)}% done")

        for player_num in range(self.num_players):
            print(
                f"Player {player_num} won {self.games_won[player_num]} games with an average score of {sum(self.game_scores[player_num]) / self.num_games}"
            )
