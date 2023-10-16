from dice_util import get_sum_of_counts

categories_to_index = {
    "ones": 0,
    "twos": 1,
    "threes": 2,
    "fours": 3,
    "fives": 4,
    "sixes": 5,
    "3 of a kind": 6,
    "4 of a kind": 7,
    "full house": 8,
    "small straight": 9,
    "large straight": 10,
    "yahtzee": 11,
    "chance": 12,
}


def get_score(category, counts):
    if category < 6:
        return 1.0 * (category + 1) * counts[category]
    elif category == 6:
        if any(count >= 3 for count in counts):
            return float(get_sum_of_counts(counts))
        else:
            return 0.0
    elif category == 7:
        if any(count >= 4 for count in counts):
            return float(get_sum_of_counts(counts))
        else:
            return 0.0
    elif category == 8:
        if 3 in counts and 2 in counts:
            return 25.0
        else:
            return 0.0
    elif category == 9:
        in_a_row = 0
        for count in counts:
            if count > 0:
                in_a_row += 1
                if in_a_row >= 4:
                    return 30.0
            else:
                in_a_row = 0
        return 0.0
    elif category == 10:
        in_a_row = 0
        for count in counts:
            if count > 0:
                in_a_row += 1
                if in_a_row >= 5:
                    return 40.0
            else:
                in_a_row = 0
        return 0.0
    elif category == 11:
        if 5 in counts:
            return 50.0
        else:
            return 0.0
    elif category == 12:
        return float(get_sum_of_counts(counts))
    else:
        raise ValueError("Invalid category")
