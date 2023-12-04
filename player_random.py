from player import Player
from precomputed import turn_actions
from random import randint


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()
        self.player_type = "random"

    def get_action(self, game):
        player_turn = game.player_turn
        turn_state = (game.rolls, game.dice_combo)
        scorecard = game.score_cards[player_turn]
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
