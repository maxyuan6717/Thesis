from abc import ABC, abstractmethod


class Player(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def play(self, game):
        pass


class OptimalPlayer(Player):
    def __init__(self):
        super().__init__()

    def play(self, game):
        pass
