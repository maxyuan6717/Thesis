# gymnasium env for yahtzee game
from gymnasium import Env, spaces
import numpy as np
from typing import Dict, List, Tuple
from game import Yahtzee
from player_solitaire_greedy import GreedyPlayer
from player_controlled import ControlledPlayer
from dice_util import get_dice_from_counts
from scorecard import UPPER_SCORE_THRESHOLD, UPPER_SCORE_BONUS

# actions from https://github.com/villebro/pyhtzee/blob/main/pyhtzee/utils.py

CATEGORY_ACTION_OFFSET = 31
action_to_dice_roll_map: Dict[int, Tuple[bool, bool, bool, bool, bool]] = {}
dice_roll_to_action_map: Dict[Tuple[bool, bool, bool, bool, bool], int] = {}
for dice1 in [1, 0]:
    for dice2 in [1, 0]:
        for dice3 in [1, 0]:
            for dice4 in [1, 0]:
                for dice5 in [1, 0]:
                    # make rolling all dice the first action, i.e. zero
                    key = 31 - (
                        dice5 * 2**0
                        + dice4 * 2**1
                        + dice3 * 2**2
                        + dice2 * 2**3
                        + dice1 * 2**4
                    )
                    value = (
                        bool(dice1),
                        bool(dice2),
                        bool(dice3),
                        bool(dice4),
                        bool(dice5),
                    )
                    # not rolling any dice is not a valid action
                    if key < 31:
                        action_to_dice_roll_map[key] = value
                        dice_roll_to_action_map[value] = key


def get_dice_one_hot_encoding(dice_value: int) -> np.ndarray:
    return np.array(
        [
            1 if dice_value == 1 else 0,
            1 if dice_value == 2 else 0,
            1 if dice_value == 3 else 0,
            1 if dice_value == 4 else 0,
            1 if dice_value == 5 else 0,
            1 if dice_value == 6 else 0,
        ]
    )


class YahtzeeEnv(Env):
    def __init__(self):
        super().__init__()
        self.opponent = GreedyPlayer()
        self.game = Yahtzee(
            [
                ControlledPlayer(),
                self.opponent,
            ]
        )
        self.game.roll_dice()

        # max number of actions is 31 for diff combinations of rerolling 5 dice
        # and 13 for diff categories to score
        self.action_space = spaces.Discrete(44)

        self.observation_space = spaces.Tuple(
            (
                spaces.MultiBinary(13),  # our scorecard
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # our upper score normalized by dividing by 63
                spaces.MultiBinary(13),  # opponent scorecard
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # their upper score normalized by dividing by 63
                spaces.Discrete(
                    375 * 2 + 1, start=-375
                ),  # score difference between us and opponent
                spaces.MultiBinary(3),  # one hot encoding of which roll we are on
                spaces.MultiBinary(6),  # one hot encoding of dice values for dice 1
                spaces.MultiBinary(6),  # one hot encoding of dice values for dice 2
                spaces.MultiBinary(6),  # one hot encoding of dice values for dice 3
                spaces.MultiBinary(6),  # one hot encoding of dice values for dice 4
                spaces.MultiBinary(6),  # one hot encoding of dice values for dice 5
                spaces.Discrete(6),  # number of dices with value 1
                spaces.Discrete(6),  # number of dices with value 2
                spaces.Discrete(6),  # number of dices with value 3
                spaces.Discrete(6),  # number of dices with value 4
                spaces.Discrete(6),  # number of dices with value 5
                spaces.Discrete(6),  # number of dices with value 6
            )
        )

    def get_observation_space(self):
        game = self.game

        return (
            game.score_cards[0].get_bitmask_np_array(),  # our scorecard,
            np.array(
                [float(game.score_cards[0].get_upper_score()) / 63],
                dtype=np.float32,
            ),  # our upper score normalized by dividing by 63
            game.score_cards[1].get_bitmask_np_array(),  # opponent scorecard,
            np.array(
                [float(game.score_cards[1].get_upper_score()) / 63],
                dtype=np.float32,
            ),  # their upper score normalized by dividing by 63
            game.score_cards[0].get_final_score()
            - game.score_cards[
                1
            ].get_final_score(),  # score difference between us and opponent
            np.array(
                [
                    1 if game.rolls == 1 else 0,
                    1 if game.rolls == 2 else 0,
                    1 if game.rolls == 3 else 0,
                ]
            ),  # one hot encoding of which roll we are on
            get_dice_one_hot_encoding(
                game.dice[0]
            ),  # one hot encoding of dice values for dice 1
            get_dice_one_hot_encoding(
                game.dice[1]
            ),  # one hot encoding of dice values for dice 2
            get_dice_one_hot_encoding(
                game.dice[2]
            ),  # one hot encoding of dice values for dice 3
            get_dice_one_hot_encoding(
                game.dice[3]
            ),  # one hot encoding of dice values for dice 4
            get_dice_one_hot_encoding(
                game.dice[4]
            ),  # one hot encoding of dice values for dice 5
            game.dice_combo[0],
            game.dice_combo[1],
            game.dice_combo[2],
            game.dice_combo[3],
            game.dice_combo[4],
            game.dice_combo[5],
        )

    def get_possible_actions(self):
        game = self.game
        possible_actions = []
        if game.rolls < 3:
            possible_actions.extend(list(range(CATEGORY_ACTION_OFFSET)))

        for category in range(13):
            mask = game.score_cards[0].get_bitmask()
            if mask & (1 << category):
                continue
            possible_actions.append(category + CATEGORY_ACTION_OFFSET)

        return possible_actions

    def sample_action(self):
        # somehow convert action to an int from 0 to 43
        possible_actions = self.get_possible_actions()

        return np.random.choice(possible_actions)

    def step(self, action: int):
        # make action for player
        # if end of player's turn, play opponent's turn
        possible_actions = self.get_possible_actions()
        debug_info = {}
        if not action in possible_actions:
            # print("invalid action", action)
            return self.get_observation_space(), 0, False, False, debug_info

        game = self.game
        action_to_play = action
        if action < CATEGORY_ACTION_OFFSET:
            dice_to_keep_combo = [0, 0, 0, 0, 0, 0]
            dice = game.dice
            mask = action_to_dice_roll_map[action]
            for i in range(5):
                if mask[i]:
                    dice_to_keep_combo[dice[i] - 1] += 1

            action_to_play = (game.rolls, tuple(dice_to_keep_combo))

        else:
            category = action - CATEGORY_ACTION_OFFSET
            action_to_play = category

        reward = game.play_player_action(0, action_to_play)
        if reward is None:
            # todo fix
            reward = 0

        if action >= CATEGORY_ACTION_OFFSET:
            game.play_player_turn(1, self.opponent)
            game.turn += 1
            game.roll_dice()

        if isinstance(action_to_play, int) and action_to_play < 6:
            scorecard = game.score_cards[0]
            upper_score = scorecard.get_upper_score()
            filled_upper_categories = scorecard.get_bitmask() & ((1 << 6) - 1) == (
                (1 << 6) - 1
            )
            if upper_score >= UPPER_SCORE_THRESHOLD and filled_upper_categories:
                reward += UPPER_SCORE_BONUS

        if game.turn > 13:
            print("game over")
        return self.get_observation_space(), reward, game.turn > 13, False, debug_info

    def reset(self, **kwargs):
        self.game = Yahtzee(
            [
                ControlledPlayer(),
                self.opponent,
            ]
        )
        self.game.roll_dice()

        return self.get_observation_space(), {}

    def render(self, mode="human", close=False):
        pass
