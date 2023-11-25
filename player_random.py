from player import Player
from precomputed import turn_actions
from random import randint

turn_actions_cache = [None for _mask in range(1 << 13)]


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()
        self.player_type = "random"

    def get_action(self, scorecard, turn_state):
        mask = scorecard.get_bitmask()

        actions = turn_actions[turn_state]

        filtered_actions = []
        for action in actions:
            if not isinstance(action, tuple):
                if mask & (1 << action):
                    continue
            filtered_actions.append(action)

        # pick random action
        num_actions = len(filtered_actions)
        action_index = randint(0, num_actions - 1)
        return filtered_actions[action_index]
