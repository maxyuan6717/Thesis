import pickle as pkl
import numpy as np

scores = pkl.load(open("scores.pkl", "rb"))

UPPER_SCORE_THRESHOLD = 63
UPPER_SCORE_BONUS = 35.0


class Scorecard:
    def __init__(self, init_bitmask=0):
        self.bitmask = init_bitmask
        self.upper_score = 0
        self.total_score = 0

    def score(self, category: int, dice: tuple[int, ...]):
        # make dice a tuple if it isn't already
        if not isinstance(dice, tuple):
            dice = tuple(dice)
        if self.bitmask & (1 << category):
            raise ValueError("Category already scored")
        self.bitmask |= 1 << category
        self.total_score += scores[dice][category]
        if category < 6:
            self.upper_score += scores[dice][category]

        return scores[dice][category]

    def get_final_score(self):
        return self.total_score + (
            UPPER_SCORE_BONUS if self.upper_score >= UPPER_SCORE_THRESHOLD else 0
        )

    def get_bitmask(self):
        return self.bitmask

    def get_bitmask_np_array(self):
        # return array of binary representation of bitmask
        return np.array([int(x) for x in bin(self.bitmask)[2:].zfill(13)])

    def get_upper_score(self):
        return int(
            min(
                self.upper_score,
                UPPER_SCORE_THRESHOLD,
            )
        )

    def get_mask_string(self):
        return bin(self.bitmask)[2:].zfill(13)
