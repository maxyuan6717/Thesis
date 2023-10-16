import pickle as pkl

dice_rolls = pkl.load(open("dice_rolls.pkl", "rb"))
dice_to_keep_combos = pkl.load(open("dice_to_keep_combos.pkl", "rb"))
scores = pkl.load(open("scores.pkl", "rb"))
dice_probabilities = pkl.load(open("dice_probabilities.pkl", "rb"))
game_values = pkl.load(open("game_values.pkl", "rb"))


def main():
    # print(dice_rolls)
    # print(dice_to_keep_combos)
    # print(scores)
    # print(scores[(4, 1, 0, 0, 0, 0)][12])
    # print(dice_probabilities)
    # print out the game values
    # for i in range(len(game_values)):
    #     print(i, game_values[i])
    # print(game_values[0:1000])
    print(game_values[0], "max", max(game_values))
    pass


if __name__ == "__main__":
    main()
