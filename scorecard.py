import pickle as pkl

scores = pkl.load(open("scores.pkl", "rb"))


class Scorecard:
    def __init__(self, init_bitmask=0):
        self.bitmask = init_bitmask
        self.total_score = 0

    def score(self, category: int, dice: tuple[int, ...]):
        if self.bitmask & (1 << category):
            raise ValueError("Category already scored")
        self.bitmask |= 1 << category
        self.total_score += scores[dice][category]

    def get_score(self):
        return self.total_score

    def get_bitmask(self):
        return self.bitmask

    def get_mask_string(self):
        return bin(self.bitmask)[2:].zfill(13)
