from dice_util import (
    combine_dice_counts,
)

from precomputed import (
    dice_rolls,
    dice_to_keep_combos,
    scores,
    dice_probabilities,
)
from math import inf
import itertools as it
import pickle as pkl

num_states = 1 << 13
filled_mask = num_states - 1


def get_turn_states(num_dice=5, num_sides=6, num_rolls=3):
    for roll in range(num_rolls, 0, -1):
        for dice in dice_rolls[num_dice]:
            yield (roll, dice)


def solve_turn_states(mask, game_values, num_dice=5, num_sides=6, num_rolls=3):
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
                num_remaining_dice = num_dice - sum(dice_to_keep_combo)
                for new_roll in dice_rolls[num_remaining_dice]:
                    new_dice = combine_dice_counts(new_roll, dice_to_keep_combo)
                    value += (
                        dice_probabilities[new_roll] * turn_values[(roll + 1, new_dice)]
                    )
                if value > best_value:
                    best_value = value
                    best_action = (roll, dice_to_keep_combo)

        for category in range(13):
            if mask & (1 << category):
                continue
            new_mask = mask | (1 << category)
            value = scores[dice][category] + game_values[new_mask]
            if value > best_value:
                best_value = value
                best_action = category

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
    game_values = [0.0] * num_states
    turn_values = [None] * num_states
    turn_actions = [None] * num_states
    progress = 0.0
    for mask in get_game_states():
        if mask == filled_mask:
            continue
        _turn_actions, _turn_values = solve_turn_states(
            mask, game_values, num_dice, num_sides, num_rolls
        )

        value = 0.0
        for dice in dice_rolls[num_dice]:
            value += dice_probabilities[dice] * _turn_values[(1, dice)]

        game_values[mask] = value
        turn_values[mask] = _turn_values
        turn_actions[mask] = _turn_actions
        progress += 1.0

        percentage = progress / num_states * 100
        if progress % (round(num_states / 100)) == 0:
            print(f"{int(percentage)}%")

    return game_values, turn_values, turn_actions


def main():
    game_values, turn_values, turn_actions = solve_game_states()
    with open("game_values.pkl", "wb") as f:
        pkl.dump(game_values, f)
    with open("turn_values.pkl", "wb") as f:
        pkl.dump(turn_values, f)
    with open("turn_actions.pkl", "wb") as f:
        pkl.dump(turn_actions, f)


if __name__ == "__main__":
    main()
