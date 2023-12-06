# gymnasium env for yahtzee game
from gymnasium import Env, spaces
import numpy as np
from typing import Dict, List, Tuple
from game import Yahtzee
from precomputed import scores
from player_solitaire_greedy import GreedyPlayer
from player_random import RandomPlayer
from player_controlled import ControlledPlayer
from dice_util import get_dice_from_counts
from scorecard import UPPER_SCORE_THRESHOLD, UPPER_SCORE_BONUS

scores[(0, 0, 0, 0, 0, 0)] = [0 for _ in range(13)]

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


def game_to_observation_space(game: Yahtzee, player_turn=0) -> Tuple:
    return (
        game.score_cards[player_turn].get_bitmask_np_array(),  # our scorecard,
        np.array(
            [float(game.score_cards[player_turn].get_upper_score()) / 63],
            dtype=np.float32,
        ),  # our upper score normalized by dividing by 63
        game.score_cards[
            game.num_players - 1 - player_turn
        ].get_bitmask_np_array(),  # opponent scorecard,
        np.array(
            [
                float(
                    game.score_cards[
                        game.num_players - 1 - player_turn
                    ].get_upper_score()
                )
                / 63
            ],
            dtype=np.float32,
        ),  # their upper score normalized by dividing by 63
        game.score_cards[player_turn].get_final_score()
        - game.score_cards[
            game.num_players - 1 - player_turn
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
        # np.array(
        #     [
        #         int(scores[game.dice_combo][0]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][1]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][2]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][3]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][4]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][5]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][6]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][7]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][8]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][9]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][10]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][11]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         int(scores[game.dice_combo][12]),
        #     ],
        #     dtype=np.int16,
        # ),
        # np.array(
        #     [
        #         UPPER_SCORE_BONUS
        #         if game.score_cards[player_turn].get_upper_score() >= UPPER_SCORE_THRESHOLD
        #         else 0
        #     ],
        #     dtype=np.int16,
        # ),
    )


def get_possible_actions(game: Yahtzee, player_turn=0):
    possible_actions = []
    if game.rolls < 3:
        possible_actions.extend(list(range(CATEGORY_ACTION_OFFSET)))

    for category in range(13):
        mask = game.score_cards[player_turn].get_bitmask()
        if mask & (1 << category):
            continue
        possible_actions.append(category + CATEGORY_ACTION_OFFSET)

    return possible_actions


def get_action_to_play(game: Yahtzee, action: int):
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
        action_to_play = int(category)

    return action_to_play


class YahtzeeEnv(Env):
    def __init__(self):
        super().__init__()
        # self.opponent = GreedyPlayer()
        self.opponent = RandomPlayer()
        self.game = Yahtzee(
            [
                ControlledPlayer(),
                self.opponent,
            ]
        )
        self.game.roll_dice()

        self.invalid_actions = 0

        self.reward_system = "old"
        self.punish_not_rolling = False

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
                # spaces.Box(low=-1, high=5, shape=(1,), dtype=np.int16),  # aces
                # spaces.Box(low=-1, high=10, shape=(1,), dtype=np.int16),  # twos
                # spaces.Box(low=-1, high=15, shape=(1,), dtype=np.int16),  # threes
                # spaces.Box(low=-1, high=20, shape=(1,), dtype=np.int16),  # fours
                # spaces.Box(low=-1, high=25, shape=(1,), dtype=np.int16),  # fives
                # spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # sixes
                # spaces.Box(
                #     low=-1, high=30, shape=(1,), dtype=np.int16
                # ),  # three of a kind
                # spaces.Box(
                #     low=-1, high=30, shape=(1,), dtype=np.int16
                # ),  # four of a kind
                # spaces.Box(low=-1, high=25, shape=(1,), dtype=np.int16),  # full house
                # spaces.Box(
                #     low=-1, high=30, shape=(1,), dtype=np.int16
                # ),  # small straight
                # spaces.Box(
                #     low=-1, high=40, shape=(1,), dtype=np.int16
                # ),  # large straight
                # spaces.Box(low=-1, high=30, shape=(1,), dtype=np.int16),  # chance
                # spaces.Box(low=-1, high=50, shape=(1,), dtype=np.int16),  # yahtzee
                # spaces.Box(low=-1, high=35, shape=(1,), dtype=np.int16),  # upper bonus
            )
        )

    def sample_action(self):
        possible_actions = get_possible_actions(self.game)

        return np.random.choice(possible_actions)

    def step(self, action: int):
        # make action for player
        # if end of player's turn, play opponent's turn
        game = self.game
        possible_actions = get_possible_actions(game)
        debug_info = {"model_score": 0}
        if not action in possible_actions:
            self.invalid_actions += 1
            reward = -10.0
            return game_to_observation_space(game), reward, False, False, debug_info

        action_to_play = get_action_to_play(game, action)

        additional_score = game.play_player_action(0, action_to_play)

        if action >= CATEGORY_ACTION_OFFSET:
            game.play_player_turn(1, self.opponent)
            game.turn += 1
            game.player_turn = 0
            game.reset_turn()
            game.roll_dice()

        model_score = game.score_cards[0].get_final_score()
        opponent_score = game.score_cards[1].get_final_score()

        # v1
        reward = (model_score - opponent_score) / 375.0

        # incorporate both additional_score and difference between model_score and opponent_score in reward
        # v2
        # reward = (additional_score + model_score - opponent_score + 375.0) / 750.0

        # v3
        # reward = model_score
        # if action < CATEGORY_ACTION_OFFSET:
        #     reward = 0.0

        debug_info["model_score"] = model_score
        debug_info["invalid_actions"] = self.invalid_actions

        if game.turn > 13:
            # print(
            #     model_score,
            #     opponent_score,
            # )
            if model_score > opponent_score:
                # print("-------------WE WONNNNNNNNNNNNN-------------")
                debug_info["won"] = True
                reward = 1.0
            elif model_score < opponent_score:
                reward = -1.0
                pass
            else:
                reward = 0.0
                pass

        return (
            game_to_observation_space(game),
            reward,
            game.turn > 13,
            False,
            debug_info,
        )

    def reset(self, **kwargs):
        model_final_score = self.game.score_cards[0].get_final_score()
        self.game = Yahtzee(
            [
                ControlledPlayer(),
                self.opponent,
            ]
        )
        self.game.roll_dice()
        print("Invalid Actions:", self.invalid_actions)
        self.invalid_actions = 0

        return game_to_observation_space(self.game), {
            "model_final_score": model_final_score
        }

    def render(self, mode="human", close=False):
        pass


def flatten_state(state):
    return np.array(
        [
            *state[0],
            *state[1],
            *state[2],
            *state[3],
            *[state[4]],
            *state[5],
            *state[6],
            *state[7],
            *state[8],
            *state[9],
            *state[10],
            *[state[11]],
            *[state[12]],
            *[state[13]],
            *[state[14]],
            *[state[15]],
            *[state[16]],
            # *state[17],
            # *state[18],
            # *state[19],
            # *state[20],
            # *state[21],
            # *state[22],
            # *state[23],
            # *state[24],
            # *state[25],
            # *state[26],
            # *state[27],
            # *state[28],
            # *state[29],
            # *state[30],
        ]
    )
