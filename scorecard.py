import pickle as pkl

scores = pkl.load(open("scores.pkl", "rb"))

UPPER_SCORE_THRESHOLD = 63
UPPER_SCORE_BONUS = 35.0


class Scorecard:
    def __init__(self, init_bitmask=0):
        self.bitmask = init_bitmask
        self.upper_score = 0
        self.total_score = 0

    def score(self, category: int, dice: tuple[int, ...]):
        if self.bitmask & (1 << category):
            raise ValueError("Category already scored")
        self.bitmask |= 1 << category
        self.total_score += scores[dice][category]
        if category < 6:
            self.upper_score += scores[dice][category]

    def get_final_score(self):
        return self.total_score + (
            UPPER_SCORE_BONUS if self.upper_score >= UPPER_SCORE_THRESHOLD else 0
        )

    def get_bitmask(self):
        return self.bitmask

    def get_upper_score(self):
        return int(
            min(
                self.upper_score,
                UPPER_SCORE_THRESHOLD,
            )
        )

    def get_mask_string(self):
        return bin(self.bitmask)[2:].zfill(13)
