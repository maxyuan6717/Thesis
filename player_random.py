from player import Player
from solve import get_turn_states
from precomputed import dice_to_keep_combos, unused_categories
from random import randint

turn_actions_cache = [None for _mask in range(1 << 13)]


class RandomPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, scorecard, turn_state):
        mask = scorecard.get_bitmask()
        turn_actions = turn_actions_cache[mask]
        if turn_actions is None:
            turn_actions = {}
            for turn in get_turn_states():
                turn_actions[turn] = []
                roll, dice = turn
                if roll < 3:
                    for dice_to_keep_combo in dice_to_keep_combos[dice]:
                        if dice_to_keep_combo == dice:
                            continue
                        turn_actions[turn].append((roll, dice_to_keep_combo))
                for category in unused_categories[mask]:
                    turn_actions[turn].append(category)

            turn_actions_cache[mask] = turn_actions

        # pick random action
        num_actions = len(turn_actions[turn_state])
        action_index = randint(0, num_actions - 1)
        return turn_actions[turn_state][action_index]
