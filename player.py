from abc import ABC, abstractmethod
from scorecard import Scorecard


class Player(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, scorecard: Scorecard, turn_state: tuple[int, tuple[int, ...]]):
        pass
