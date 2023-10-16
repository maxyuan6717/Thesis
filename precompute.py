from dice_util import (
    get_dice_rolls,
    get_dice_to_keep_combos,
    get_dice_probability,
)
from scoring_util import get_score
import pickle as pkl


def main():
    dice_rolls = {}
    dice_to_keep_combos = {}
    scores = {}

    dice_probabilities = {}
    for num_dice in range(1, 6):
        dice_rolls[num_dice] = list(get_dice_rolls(num_dice))
        for dice in dice_rolls[num_dice]:
            dice_probabilities[dice] = get_dice_probability(dice)
            if num_dice == 5:
                dice_to_keep_combos[dice] = list(get_dice_to_keep_combos(dice))
                scores[dice] = {}
                for category in range(13):
                    scores[dice][category] = get_score(category, dice)

    with open("dice_rolls.pkl", "wb") as f:
        pkl.dump(dice_rolls, f)
    with open("dice_to_keep_combos.pkl", "wb") as f:
        pkl.dump(dice_to_keep_combos, f)
    with open("scores.pkl", "wb") as f:
        pkl.dump(scores, f)
    with open("dice_probabilities.pkl", "wb") as f:
        pkl.dump(dice_probabilities, f)


if __name__ == "__main__":
    main()
