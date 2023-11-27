import pickle as pkl

dice_rolls = pkl.load(open("dice_rolls.pkl", "rb"))
dice_to_keep_combos = pkl.load(open("dice_to_keep_combos.pkl", "rb"))
scores = pkl.load(open("scores.pkl", "rb"))
dice_probabilities = pkl.load(open("dice_probabilities.pkl", "rb"))
game_values = pkl.load(open("game_values.pkl", "rb"))
rerolled_dice_combos = pkl.load(open("rerolled_dice_combos.pkl", "rb"))
reachable_game_states = pkl.load(open("reachable_game_states.pkl", "rb"))
unused_categories = pkl.load(open("unused_categories.pkl", "rb"))
turn_actions = pkl.load(open("turn_actions.pkl", "rb"))


def main():
    # print expected highest score of optimal 1-player game
    print(game_values[0][0])
    print(scores[(0, 0, 0, 0, 0, 0)])


if __name__ == "__main__":
    main()
