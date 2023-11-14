from dice_util import (
    get_dice_rolls,
    get_dice_to_keep_combos,
    get_dice_probability,
    combine_dice_counts,
)
from scoring_util import get_score
from solve import get_turn_states
from scorecard import UPPER_SCORE_THRESHOLD
import itertools as it
import pickle as pkl


def get_upper_category_states():
    for num_filled in range(7):
        for filled_categories in it.combinations(range(6), num_filled):
            mask = 0
            for category in filled_categories:
                mask ^= 1 << category
            yield mask


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

    rerolled_dice_combos = {}
    for turn in get_turn_states(5, 6, 3):
        roll, dice = turn

        if roll == 3:
            rerolled_dice_combos[(roll, dice)] = []
            continue

        for dice_to_keep_combo in dice_to_keep_combos[dice]:
            if dice_to_keep_combo == dice:
                continue
            rerolled_dice_combos[(roll, dice_to_keep_combo)] = []
            num_remaining_dice = 5 - sum(dice_to_keep_combo)
            for new_roll in dice_rolls[num_remaining_dice]:
                new_dice = combine_dice_counts(new_roll, dice_to_keep_combo)
                rerolled_dice_combos[(roll, dice_to_keep_combo)].append(
                    (dice_probabilities[new_roll], new_dice)
                )

    reachable_states = [
        [False for _mask in range(1 << 6)]
        for _upper_score in range(UPPER_SCORE_THRESHOLD + 1)
    ]

    for upper_score in range(UPPER_SCORE_THRESHOLD + 1):
        for upper_mask in get_upper_category_states():
            if upper_score == 0 or upper_score == UPPER_SCORE_THRESHOLD:
                reachable_states[upper_score][upper_mask] = True
                continue
            filled_categories = []
            for category in range(6):
                if upper_mask & (1 << category):
                    filled_categories.append(category)

            for filled_category in filled_categories:
                works = False
                for k in range(6):
                    prev_score = upper_score - k * (filled_category + 1)
                    if prev_score < 0:
                        break
                    prev_mask = upper_mask ^ (1 << filled_category)
                    if reachable_states[prev_score][prev_mask]:
                        works = True
                        break
                if works:
                    reachable_states[upper_score][upper_mask] = True
                    break

    reachable_game_states = [
        [False for _mask in range(1 << 13)]
        for _score in range(UPPER_SCORE_THRESHOLD + 1)
    ]
    unused_categories = [[] for _ in range(1 << 13)]
    valid = 0
    for mask in range(1 << 13):
        for upper_score in range(UPPER_SCORE_THRESHOLD + 1):
            bottomSixBits = mask & ((1 << 6) - 1)
            if reachable_states[upper_score][bottomSixBits]:
                reachable_game_states[upper_score][mask] = True
                valid += 1
        for category in range(13):
            if mask & (1 << category):
                continue
            unused_categories[mask].append(category)

    # print(1 - (valid / ((UPPER_SCORE_THRESHOLD + 1) * (1 << 13))), 1260 / 4096)

    with open("dice_rolls.pkl", "wb") as f:
        pkl.dump(dice_rolls, f)
    with open("dice_to_keep_combos.pkl", "wb") as f:
        pkl.dump(dice_to_keep_combos, f)
    with open("scores.pkl", "wb") as f:
        pkl.dump(scores, f)
    with open("dice_probabilities.pkl", "wb") as f:
        pkl.dump(dice_probabilities, f)
    with open("rerolled_dice_combos.pkl", "wb") as f:
        pkl.dump(rerolled_dice_combos, f)
    with open("reachable_game_states.pkl", "wb") as f:
        pkl.dump(reachable_game_states, f)
    with open("unused_categories.pkl", "wb") as f:
        pkl.dump(unused_categories, f)


if __name__ == "__main__":
    main()
