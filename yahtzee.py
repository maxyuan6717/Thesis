from game import Yahtzee


def main():
    game = Yahtzee()

    while game.turn <= 13:
        game.play_turn()


if __name__ == "__main__":
    main()
