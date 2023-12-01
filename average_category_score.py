import pickle as pkl

from game import Yahtzee
from player_solitaire_optimal import OptimalPlayer


def main():
    player = OptimalPlayer()
    game = Yahtzee([player], 1000)
    game.play_games()

    print("Average category scores:")
    average_category_scores = [0 for _ in range(13)]
    for category, score in enumerate(game.tot_category_scores):
        average_category_scores[category] = score / game.num_games
        print(f"{category}: {score / game.num_games}")

    with open("average_category_scores.pkl", "wb") as f:
        pkl.dump(average_category_scores, f)


if __name__ == "__main__":
    main()
