import itertools as it
from math import factorial, prod


def get_counts_from_dice(dices, num_sides=6):
    return tuple(dices.count(i) for i in range(1, num_sides + 1))


def get_dice_from_counts(counts, num_sides=6):
    dices = []
    for i in range(num_sides):
        dices.extend([i + 1] * counts[i])
    return dices


# get combinations of possible dice rolls
# order doesnt matter
def get_dice_rolls(num_dice=5, num_sides=6):
    dice_combos = list(
        it.combinations_with_replacement(range(1, num_sides + 1), num_dice)
    )
    for dice_combo in dice_combos:
        yield get_counts_from_dice(dice_combo)


def get_dice_to_keep_combos(counts):
    return it.product(*[range(count + 1) for count in counts])


def combine_dice_counts(counts1, counts2):
    return tuple(counts1[i] + counts2[i] for i in range(len(counts1)))


def get_dice_probability(counts, num_sides=6):
    num_dice = sum(counts)
    return (
        factorial(num_dice)
        / prod(factorial(count) for count in counts)
        / (num_sides**num_dice)
    )


def get_sum_of_counts(counts):
    return sum(count * (i + 1) for i, count in enumerate(counts))
