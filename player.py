from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, game):
        pass
