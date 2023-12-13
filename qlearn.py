import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# from yahtzee_env import YahtzeeEnv, flatten_state

from yahtzee_env_updated import YahtzeeEnv, flatten_state


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()

        layer_sizes = [input_size, 256, 256, output_size]

        self.layer1 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.layer2 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.layer3 = nn.Linear(layer_sizes[2], layer_sizes[3])

    def forward(self, x):
        x = x.float()
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Agent:
    # gamma is the discount factor as mentioned in the previous section
    # epsilon is the starting value of epsilon
    # lr is the learning rate of the ``AdamW`` optimizer
    # input_size is the size of the flattened observation state
    # output_size is the number of actions
    # memory_size is the size of the replay buffer
    # batch_size is the number of transitions sampled from the replay buffer
    # eps_end is the final value of epsilon
    # eps_decay controls the rate of exponential decay of epsilon, higher means a slower decay
    # tau is the update rate of the target network

    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_size,
        output_size,
        memory_size=20000,
        batch_size=500,
        eps_end=0.1,
        eps_decay=0.9996,
        tau=0.005,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.tau = tau
        self.input_size = input_size
        self.action_space = [i for i in range(output_size)]

        self.memory_size = memory_size
        self.memory = ReplayMemory(memory_size)
        self.batch_size = batch_size

        self.device = device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.policy_net = DQN(input_size, output_size).to(self.device)
        self.target_net = DQN(input_size, output_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr, amsgrad=True)
        self.steps_done = 0

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            random_action = random.choice(self.action_space)
            return torch.tensor([[random_action]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.policy_net(observation).max(1).indices.view(1, 1)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.epsilon = (
            self.epsilon * self.eps_decay
            if self.epsilon > self.eps_end
            else self.eps_end
        )

    def update_target_net(self):
        # Soft update of the target network's weights to move them closer to the policy network's weights
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.tau + target_net_state_dict[key] * (1 - self.tau)
        self.target_net.load_state_dict(target_net_state_dict)


steps_done = 0
Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))
game_score_history = []
average_game_scores = []
last_100_game_scores = []
last_100_game_sum = 0


def full_plot(title, game_score_history, eps_history, invalid_action_history):
    fig, ax = plt.subplots()

    fig.subplots_adjust(right=0.75)
    twin1 = ax.twinx()
    twin2 = ax.twinx()
    twin2.spines.right.set_position(("axes", 1.2))

    x_data = range(100, len(average_game_scores) + 100)
    y_data = average_game_scores

    # show only 500 points
    filter_size = max(1, len(x_data) // 500)
    x_data = x_data[::filter_size]
    y_data = y_data[::filter_size]

    (p1,) = ax.plot(x_data, y_data, color="steelblue", label="100 Game Average Score")
    (p2,) = twin1.plot(eps_history, color="seagreen", label="Epsilon")
    (p3,) = twin2.plot(invalid_action_history, color="crimson", label="Invalid Actions")

    # add padding below title
    fig.suptitle(title, wrap=True, y=1.2)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Game Score")
    twin1.set_ylabel("Epsilon")
    twin2.set_ylabel("Invalid Actions")

    ax.yaxis.label.set_color(p1.get_color())
    twin1.yaxis.label.set_color(p2.get_color())
    twin2.yaxis.label.set_color(p3.get_color())

    tkw = dict(size=4, width=1.5)
    ax.tick_params(axis="y", colors=p1.get_color(), **tkw)
    twin1.tick_params(axis="y", colors=p2.get_color(), **tkw)
    twin2.tick_params(axis="y", colors=p3.get_color(), **tkw)
    ax.tick_params(axis="x", **tkw)

    # put legend on top of plot
    ax.legend(handles=[p1, p2, p3], bbox_to_anchor=(0.5, 1.25), loc="upper center")

    return fig


def update_game_score_plot(title):
    plt.clf()
    plt.title(title, wrap=True)
    plt.xlabel("Episode")
    plt.ylabel("Average Game Score")

    if len(game_score_history) > 100:
        x_data = range(100, len(average_game_scores) + 100)
        y_data = average_game_scores

        # show only 500 points
        filter_size = max(1, len(x_data) // 500)
        x_data = x_data[::filter_size]
        y_data = y_data[::filter_size]

        plt.plot(
            x_data,
            y_data,
            color="darkorange",
        )

    plt.pause(0.1)


def main():
    gym.register(
        id="Yahtzee-v0",
        entry_point=YahtzeeEnv,
        max_episode_steps=5000,
    )

    env = gym.make("Yahtzee-v0")
    env_fields = env.unwrapped.__dict__
    reward_system = env_fields["reward_system"]
    opponent_type = env_fields["opponent"].player_type
    punish_not_rolling = env_fields["punish_not_rolling"]
    fixed_score_goal = env_fields["fixed_score_goal"]

    # init code from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    # set up matplotlib
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display

    plt.ion()

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    state = flatten_state(state)

    n_observations = len(state)

    print("neural network dimensions", n_observations, n_actions)

    agent = Agent(
        gamma=0.99,
        epsilon=0.9,
        eps_end=0.1,
        eps_decay=0.9998,
        tau=0.005,
        lr=0.001,
        input_size=n_observations,
        output_size=n_actions,
        memory_size=20000,
        batch_size=100,
    )

    episode_durations = []
    invalid_action_history = []
    num_wins = 0
    num_games = 100000

    plot_title = f"DQN2 Training against {opponent_type}...gamma={agent.gamma}, lr={agent.lr}, eps_decay={agent.eps_decay}, mem_size={agent.memory_size}, batch_size={agent.batch_size}, reward={reward_system}, {f'score goal={fixed_score_goal}, ' if reward_system == 'fixed' else ''}punish early category={punish_not_rolling}"

    for i_episode in range(num_games):
        # Initialize the environment and get it's state
        state, info = env.reset()
        state = flatten_state(state)
        state = torch.tensor(state, dtype=torch.float32, device=agent.device).unsqueeze(
            0
        )
        for t in count():
            action = agent.choose_action(state)
            observation, reward, terminated, truncated, info = env.step(action.item())
            # print(observation)
            observation = flatten_state(observation)
            reward = torch.tensor([reward], device=agent.device)
            done = terminated or truncated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32, device=agent.device
                ).unsqueeze(0)

            # Store the transition in memory
            agent.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            agent.learn()

            # Soft update of the target network's weights
            agent.update_target_net()

            if done:
                episode_durations.append(t + 1)
                game_score_history.append(info["model_score"])
                invalid_action_history.append(info["invalid_actions"])

                global last_100_game_sum
                if len(last_100_game_scores) < 100:
                    last_100_game_scores.append(info["model_score"])
                    last_100_game_sum += info["model_score"]
                else:
                    last_100_game_sum -= last_100_game_scores[i_episode % 100]
                    last_100_game_scores[i_episode % 100] = info["model_score"]
                    last_100_game_sum += info["model_score"]

                if len(game_score_history) > 100:
                    average_game_scores.append(last_100_game_sum / 100)

                if "won" in info:
                    num_wins += 1

                break

        if i_episode % 100 == 0:
            update_game_score_plot(plot_title)

    print("Complete")
    print("won", num_wins)

    # save model with time stamp
    torch.save(
        agent.policy_net.state_dict(),
        f"models/model_{n_observations}_{n_actions}_opponent={opponent_type}_gamma={agent.gamma}_lr={agent.lr}_eps_decay={agent.eps_decay}_mem_size={agent.memory_size}_batch_size={agent.batch_size}_reward={reward_system}{f'_score_goal={fixed_score_goal}' if reward_system == 'fixed' else ''}_punish={punish_not_rolling}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_dqn2.pt",
    )

    # full plot of game score, epsilon, and invalid actions
    fig = full_plot(plot_title, game_score_history, [], invalid_action_history)
    # save the plot to file
    fig.savefig(
        f"figures/figure_{n_observations}_{n_actions}_opponent={opponent_type}_gamma={agent.gamma}_lr={agent.lr}_eps_decay={agent.eps_decay}_mem_size={agent.memory_size}_batch_size={agent.batch_size}_reward={reward_system}{f'_score_goal={fixed_score_goal}' if reward_system == 'fixed' else ''}_punish={punish_not_rolling}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_dqn2.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.ioff()
    plt.show()

    pass


if __name__ == "__main__":
    main()
