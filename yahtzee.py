from game import Yahtzee
from player import OptimalPlayer


def main():
    player1 = OptimalPlayer()
    player2 = OptimalPlayer()
    game = Yahtzee([player1, player2], 100000)
    game.play_games()


if __name__ == "__main__":
    main()
