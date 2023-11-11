from abc import ABC, abstractmethod
import pickle as pkl

turn_actions = pkl.load(open("turn_actions.pkl", "rb"))


class Player(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, scorecard_mask: int, turn_state: tuple[int, tuple[int, ...]]):
        pass


class OptimalPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, scorecard_mask, turn_state):
        return turn_actions[scorecard_mask][turn_state]
