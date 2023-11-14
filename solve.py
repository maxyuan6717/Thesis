from precomputed import (
    dice_rolls,
    dice_to_keep_combos,
    scores,
    dice_probabilities,
    rerolled_dice_combos,
    reachable_game_states,
    unused_categories,
)
from scorecard import UPPER_SCORE_THRESHOLD, UPPER_SCORE_BONUS

from math import inf
import itertools as it
import pickle as pkl

num_states = 1 << 13
filled_mask = num_states - 1


def get_turn_states(num_dice=5, num_sides=6, num_rolls=3):
    for roll in range(num_rolls, 0, -1):
        for dice in dice_rolls[num_dice]:
            yield (roll, dice)


def solve_turn_states(
    mask: int,
    upper_score: int,
    game_values: list[list[float]],
    num_dice=5,
    num_sides=6,
    num_rolls=3,
):
    turn_actions = {}
    turn_values = {}

    for turn in get_turn_states(num_dice, num_sides, num_rolls):
        roll, dice = turn

        best_action = None
        best_value = -inf

        if roll < 3:
            for dice_to_keep_combo in dice_to_keep_combos[dice]:
                if dice_to_keep_combo == dice:
                    continue
                value = 0.0
                for dice_probability, new_dice in rerolled_dice_combos[
                    (roll, dice_to_keep_combo)
                ]:
                    value += dice_probability * turn_values[(roll + 1, new_dice)]

                if value > best_value:
                    best_value = value
                    best_action = (roll, dice_to_keep_combo)

        for category in unused_categories[mask]:
            new_mask = mask | (1 << category)
            try:
                value = (
                    scores[dice][category]
                    + game_values[new_mask][
                        min(
                            UPPER_SCORE_THRESHOLD,
                            int(upper_score + scores[dice][category]),
                        )
                        if category < 6
                        else int(upper_score)
                    ]
                )
                if value > best_value:
                    best_value = value
                    best_action = category
            except Exception as error:
                print("error:", new_mask, category, scores[dice][category], upper_score)
                print(error)
                raise error

        turn_actions[turn] = best_action
        turn_values[turn] = best_value

    return turn_actions, turn_values


def get_game_states():
    for num_filled in range(14):
        for filled_categories in it.combinations(range(13), num_filled):
            mask = filled_mask
            for category in filled_categories:
                mask ^= 1 << category
            yield mask


def solve_game_states(num_dice=5, num_sides=6, num_rolls=3):
    game_values = [
        [0.0 for _upper_score in range(UPPER_SCORE_THRESHOLD + 1)]
        for _mask in range(num_states)
    ]
    game_values[filled_mask][UPPER_SCORE_THRESHOLD] = UPPER_SCORE_BONUS
    turn_values = [
        [None for _upper_score in range(UPPER_SCORE_THRESHOLD + 1)]
        for _mask in range(num_states)
    ]
    turn_actions = [
        [None for _upper_score in range(UPPER_SCORE_THRESHOLD + 1)]
        for _mask in range(num_states)
    ]
    progress = 0.0
    for mask in get_game_states():
        for upper_score in range(UPPER_SCORE_THRESHOLD, -1, -1):
            if mask != filled_mask and reachable_game_states[upper_score][mask]:
                _turn_actions, _turn_values = solve_turn_states(
                    mask, upper_score, game_values, num_dice, num_sides, num_rolls
                )

                value = 0.0
                for dice in dice_rolls[num_dice]:
                    value += dice_probabilities[dice] * _turn_values[(1, dice)]

                game_values[mask][upper_score] = value
                # turn_values[mask][upper_score] = _turn_values
                # turn_actions[mask][upper_score] = _turn_actions

            progress += 1.0
            percentage = progress / (num_states * (UPPER_SCORE_THRESHOLD + 1)) * 100
            if (
                progress % (round((num_states * (UPPER_SCORE_THRESHOLD + 1)) / 100))
                == 0
            ):
                print(f"{int(percentage)}%")

    return game_values, turn_values, turn_actions


def main():
    game_values, turn_values, turn_actions = solve_game_states()
    with open("game_values.pkl", "wb") as f:
        pkl.dump(game_values, f)
    # with open("turn_values_35.pkl", "wb") as f:
    #     pkl.dump(turn_values, f)
    # with open("turn_actions_35.pkl", "wb") as f:
    #     pkl.dump(turn_actions, f)


if __name__ == "__main__":
    main()
