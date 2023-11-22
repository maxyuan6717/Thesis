from game import Yahtzee
from player_solitaire_optimal import OptimalPlayer
from player_solitaire_greedy import GreedyPlayer
from player_random import RandomPlayer


def main():
    player1 = OptimalPlayer()
    # player2 = OptimalPlayer()
    player2 = GreedyPlayer()
    # player2 = RandomPlayer()
    game = Yahtzee([player1, player2], 1000)
    game.play_games()


if __name__ == "__main__":
    main()
