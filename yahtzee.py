from game import Yahtzee
from player_solitaire_optimal import OptimalPlayer
from player_solitaire_greedy import GreedyPlayer
from player_random import RandomPlayer

# from player_dqn import DQNPlayer
from player_dqn_updated import DQNPlayer


def main():
    # player1 = OptimalPlayer()
    player1 = GreedyPlayer()
    # player2 = OptimalPlayer()
    # player2 = GreedyPlayer()
    # player2 = RandomPlayer()
    player2 = DQNPlayer("./model_38_23_2023-11-30_18-24-17.pt")
    game = Yahtzee([player1, player2], 100)
    game.play_games()


if __name__ == "__main__":
    main()
