categories = [
    "ones",
    "twos",
    "threes",
    "fours",
    "fives",
    "sixes",
    "3 of a kind",
    "4 of a kind",
    "full house",
    "small straight",
    "large straight",
    "yahtzee",
    "chance",
]

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


class Scorecard:
    def __init__(self, init_bitmask=0):
        self.bitmask = init_bitmask
        self.total_score = 0

    def is_category_filled(self, category):
        return self.bitmask & (1 << category)

    def fill_category(self, category, score):
        self.bitmask |= 1 << category
        self.total_score += score

    def get_score(self):
        return self.total_score

    def get_bitmask(self):
        return self.bitmask

    def get_mask_string(self):
        return bin(self.bitmask)[2:].zfill(13)
