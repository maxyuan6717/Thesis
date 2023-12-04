from gymnasium import Env, spaces
import numpy as np
from typing import Tuple
from game import Yahtzee
from player_solitaire_greedy import GreedyPlayer
from player_solitaire_optimal import OptimalPlayer
from player_random import RandomPlayer
from player_controlled import ControlledPlayer
import pickle as pkl

average_category_scores = pkl.load(open("average_category_scores.pkl", "rb"))

CATEGORY_ACTION_OFFSET = 10


def game_to_observation(game: Yahtzee, player_turn=0) -> Tuple:
    player_scorecard = game.score_cards[player_turn]
    opponent_scorecard = game.score_cards[1 - player_turn]
    player_mask = player_scorecard.get_bitmask_np_array()
    opponent_mask = opponent_scorecard.get_bitmask_np_array()

    player_upper_score = float(player_scorecard.get_upper_score())
    opponent_upper_score = float(opponent_scorecard.get_upper_score())

    player_score = player_scorecard.get_final_score()
    opponent_score = opponent_scorecard.get_final_score()

    score_diff = player_score - opponent_score

    roll_one_hot_encoding = np.array(
        [
            1 if game.rolls == 1 else 0,
            1 if game.rolls == 2 else 0,
            1 if game.rolls == 3 else 0,
        ]
    )
    dice_combo = game.dice_combo

    num_ones = float(dice_combo[0])
    num_twos = float(dice_combo[1])
    num_threes = float(dice_combo[2])
    num_fours = float(dice_combo[3])
    num_fives = float(dice_combo[4])
    num_sixes = float(dice_combo[5])

    return (
        player_mask,
        np.array([player_upper_score / 63.0], dtype=np.float32),
        opponent_mask,
        np.array([opponent_upper_score / 63.0], dtype=np.float32),
        np.array([score_diff / 375.0], dtype=np.float32),
        roll_one_hot_encoding,
        np.array([num_ones / 5.0], dtype=np.float32),
        np.array([num_twos / 5.0], dtype=np.float32),
        np.array([num_threes / 5.0], dtype=np.float32),
        np.array([num_fours / 5.0], dtype=np.float32),
        np.array([num_fives / 5.0], dtype=np.float32),
        np.array([num_sixes / 5.0], dtype=np.float32),
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


def get_action_from_meta_action(game: Yahtzee, meta_action: int):
    if meta_action >= CATEGORY_ACTION_OFFSET:
        return meta_action - CATEGORY_ACTION_OFFSET

    roll = game.rolls
    dice_combo = game.dice_combo
    dice_to_keep_combo = [0 for _ in range(6)]
    if meta_action == 0:
        # going for ones
        dice_to_keep_combo[0] = dice_combo[0]
    elif meta_action == 1:
        # going for twos
        dice_to_keep_combo[1] = dice_combo[1]
    elif meta_action == 2:
        # going for threes
        dice_to_keep_combo[2] = dice_combo[2]
    elif meta_action == 3:
        # going for fours
        dice_to_keep_combo[3] = dice_combo[3]
    elif meta_action == 4:
        # going for fives
        dice_to_keep_combo[4] = dice_combo[4]
    elif meta_action == 5:
        # going for sixes
        dice_to_keep_combo[5] = dice_combo[5]
    elif meta_action == 6:
        # going for as many as possible of a specific dice value
        max_dice = 0
        max_dice_val = 0
        for dice_val, num_dice in enumerate(dice_combo):
            if num_dice >= max_dice:
                max_dice = num_dice
                max_dice_val = dice_val
        dice_to_keep_combo[max_dice_val] = max_dice
    elif meta_action == 7:
        # going for a straight
        max_straight = 0
        max_straight_start = 0
        cur_straight = 0
        for dice_val, num_dice in enumerate(dice_combo):
            if num_dice > 0:
                cur_straight += 1
            else:
                cur_straight = 0

            if cur_straight > max_straight:
                max_straight = cur_straight
                max_straight_start = dice_val - cur_straight + 1

        for i in range(max_straight):
            dice_to_keep_combo[max_straight_start + i] = 1
    elif meta_action == 8:
        # going for a full house
        # if we have a full house, we are done
        if 2 in dice_combo and 3 in dice_combo:
            dice_to_keep_combo = dice_combo
        else:
            # otherwise, keep the top 2 most common dice
            max_dice = 0
            max_dice_val = -1
            second_max_dice = 0
            second_max_dice_val = -1
            for dice_val, num_dice in enumerate(dice_combo):
                if num_dice >= max_dice:
                    second_max_dice = max_dice
                    second_max_dice_val = max_dice_val
                    max_dice = num_dice
                    max_dice_val = dice_val
                elif num_dice >= second_max_dice:
                    second_max_dice = num_dice
                    second_max_dice_val = dice_val

            if max_dice_val >= 0:
                dice_to_keep_combo[max_dice_val] = min(max_dice, 3)

            if second_max_dice_val >= 0:
                dice_to_keep_combo[second_max_dice_val] = min(second_max_dice, 2)
    elif meta_action == 9:
        # going for chance
        dice_to_keep_combo[3] = dice_combo[3]
        dice_to_keep_combo[4] = dice_combo[4]
        dice_to_keep_combo[5] = dice_combo[5]
    else:
        raise Exception("Invalid meta action")

    return (roll, tuple(dice_to_keep_combo))


class YahtzeeEnv(Env):
    def __init__(self):
        super().__init__()
        self.opponent = RandomPlayer()
        # self.opponent = GreedyPlayer()
        # self.opponent = OptimalPlayer()
        self.game = Yahtzee(
            [
                ControlledPlayer(),
                self.opponent,
            ]
        )
        self.game.roll_dice()

        # p1 or p2
        self.reward_system = "p1"

        self.punish_not_rolling = False

        self.invalid_actions = 0

        # 0-5 are going for specific dice values
        # 6 is going for as many of a specific dice value as possible
        # 7 is going for a straight
        # 8 is going for a full house
        # 9 is going for chance
        # 10-22 are going for specific categories
        self.action_space = spaces.Discrete(23)

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
                spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
                ),  # score difference between us and opponent normalized by dividing by 375
                spaces.MultiBinary(3),  # one hot encoding of which roll we are on
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # number of dices with value 1 normalized by dividing by 5
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # number of dices with value 2 normalized by dividing by 5
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # number of dices with value 3 normalized by dividing by 5
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # number of dices with value 4 normalized by dividing by 5
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # number of dices with value 5 normalized by dividing by 5
                spaces.Box(
                    low=0, high=1, shape=(1,), dtype=np.float32
                ),  # number of dices with value 6 normalized by dividing by 5
            )
        )

    def sample_action(self):
        possible_actions = get_possible_actions(self.game)

        return np.random.choice(possible_actions)

    def step(self, action: int):
        game = self.game
        possible_actions = get_possible_actions(game)
        debug_info = {"model_score": 0}

        if not action in possible_actions:
            self.invalid_actions += 1
            reward = -5.0
            return game_to_observation(game), reward, False, False, debug_info

        action_to_play = get_action_from_meta_action(game, action)

        additional_score = game.play_player_action(0, action_to_play)
        reward = 0.0

        # if we scored a category, we are done with our turn
        # play the opponent's turn
        if isinstance(action_to_play, int):
            game.play_player_turn(1, self.opponent)
            game.turn += 1
            game.roll_dice()

            average_category_score = average_category_scores[action_to_play]

            reward = (
                additional_score - average_category_score
            ) / average_category_score

            if self.punish_not_rolling and game.rolls < 3:
                # we want to discourage the model from ending its turn early
                # so we give it a negative reward for ending its turn early
                reward -= 0.5

        model_score = game.score_cards[0].get_final_score()
        opponent_score = game.score_cards[1].get_final_score()

        debug_info["model_score"] = model_score
        debug_info["invalid_actions"] = self.invalid_actions

        done = game.turn > 13

        if done:
            if self.reward_system == "p1":
                # only reward based on our score

                # normalize by expected optimal score
                reward = model_score / 245.871
            else:
                # reward based on if we won or not
                print(
                    model_score,
                    opponent_score,
                )
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

        return game_to_observation(game), reward, done, False, debug_info

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

        return game_to_observation(self.game), {"model_final_score": model_final_score}

    def render(self, mode="human", close=False):
        pass


def flatten_state(state):
    return np.array(
        [
            *state[0],
            *state[1],
            *state[2],
            *state[3],
            *state[4],
            *state[5],
            *state[6],
            *state[7],
            *state[8],
            *state[9],
            *state[10],
            *state[11],
        ]
    )
