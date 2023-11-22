from player import Player
from precomputed import game_values
from solve import solve_turn_states
from scorecard import UPPER_SCORE_THRESHOLD

turn_actions_cache = [
    [None for _upper_score in range(UPPER_SCORE_THRESHOLD + 1)]
    for _mask in range(1 << 13)
]


class OptimalPlayer(Player):
    def __init__(self):
        super().__init__()

    def get_action(self, scorecard, turn_state):
        mask = scorecard.get_bitmask()
        upper_score = scorecard.get_upper_score()
        turn_actions = turn_actions_cache[mask][upper_score]
        if turn_actions is None:
            turn_actions, _ = solve_turn_states(
                mask,
                upper_score,
                game_values,
            )
            turn_actions_cache[mask][upper_score] = turn_actions

        return turn_actions[turn_state]
