from player import Player
from qlearn import DQN, flatten_state
from yahtzee_env_updated import (
    game_to_observation,
    get_possible_actions,
    get_action_from_meta_action,
)
import torch
import numpy as np


# model_path = "./model_38_23_2023-11-30_18-24-17.pt"


class DQNPlayer(Player):
    def __init__(self, model_path):
        super().__init__()
        self.player_type = "dqn"
        n_observations = int(model_path.split("_")[1])
        n_actions = int(model_path.split("_")[2])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = DQN(n_observations, n_actions).to(self.device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def get_action(self, game):
        player_turn = game.player_turn
        turn_state = (game.rolls, game.dice_combo)
        state = game_to_observation(game, player_turn)
        state = flatten_state(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(
            0
        )
        possible_actions = get_possible_actions(game, player_turn)

        valid_action = None

        for attempts in range(20):
            action = self.model(state).max(1).indices.view(1, 1).item()
            if action in possible_actions:
                valid_action = action
                break

        if valid_action is None:
            print("invalid action")
            valid_action = np.random.choice(possible_actions)

        action_to_play = get_action_from_meta_action(game, valid_action)

        print("playing:", action_to_play)

        return action_to_play
