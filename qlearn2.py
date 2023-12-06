import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import time

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
from yahtzee_env_updated import YahtzeeEnv, flatten_state


class DQN(nn.Module):
    def __init__(self, lr, input_size, fc1_size, fc2_size, output_size):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.output_size = output_size

        self.fc1 = nn.Linear(self.input_size, self.fc1_size)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.fc3 = nn.Linear(self.fc2_size, self.output_size)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions


# inspired by https://www.youtube.com/watch?v=wc-FxNENg9U


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        input_size,
        batch_size,
        output_size,
        memory_size=20000,
        eps_end=0.01,
        eps_decay=0.9996,
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.lr = lr
        self.input_size = input_size
        self.action_space = [i for i in range(output_size)]
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.memory_counter = 0

        self.Q_eval = DQN(lr, input_size, 256, 256, output_size)

        self.memory = np.zeros((self.memory_size, input_size), dtype=np.float32)
        self.new_memory = np.zeros((self.memory_size, input_size), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        self.memory[index] = state
        self.new_memory[index] = new_state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done
        self.memory_counter += 1

    def choose_action(self, observation):
        rand = np.random.random()
        if rand < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = torch.tensor(observation, dtype=torch.float).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()

        return int(action)

    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_memory = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_memory, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = torch.tensor(
            self.memory[batch], dtype=torch.float32, device=self.Q_eval.device
        )
        new_state_batch = torch.tensor(
            self.new_memory[batch], dtype=torch.float32, device=self.Q_eval.device
        )
        reward_batch = torch.tensor(
            self.reward_memory[batch], dtype=torch.float32, device=self.Q_eval.device
        )
        terminal_batch = torch.tensor(
            self.terminal_memory[batch], device=self.Q_eval.device
        )

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = (
            self.epsilon * self.eps_decay
            if self.epsilon > self.eps_end
            else self.eps_end
        )


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

    (p1,) = ax.plot(x_data, y_data, color="steelblue", label="Game Score")
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

    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    state, info = env.reset()
    state = flatten_state(state)
    n_observations = len(state)

    agent = Agent(
        gamma=0.99,
        epsilon=0.9,
        # try .1 min
        eps_end=0.1,
        eps_decay=0.9998,
        lr=0.001,
        input_size=n_observations,
        memory_size=20000,
        batch_size=200,
        output_size=n_actions,
    )
    scores, eps_history, invalid_action_history = [], [], []
    num_wins = 0
    num_games = 50000

    # set up matplotlib
    is_ipython = "inline" in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plot_title = f"DQN1 Training against {opponent_type}...gamma={agent.gamma}, lr={agent.lr}, eps_decay={agent.eps_decay}, mem_size={agent.memory_size}, batch_size={agent.batch_size}, reward={reward_system}, punish early category={punish_not_rolling}"
    plt.ion()
    steps = 0

    print("neural network dimensions", n_observations, n_actions)

    for i in range(num_games):
        done = False
        score = 0
        state, info = env.reset()
        state = flatten_state(state)
        while not done:
            action = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(action)
            new_state = flatten_state(new_state)
            score += reward
            agent.store_transition(state, action, reward, new_state, done)
            state = new_state

            if steps == 0:
                agent.learn()

            steps = (steps + 1) % 4

            if done:
                game_score_history.append(info["model_score"])
                invalid_action_history.append(info["invalid_actions"])

                global last_100_game_sum
                if len(last_100_game_scores) < 100:
                    last_100_game_scores.append(info["model_score"])
                    last_100_game_sum += info["model_score"]
                else:
                    last_100_game_sum -= last_100_game_scores[i % 100]
                    last_100_game_scores[i % 100] = info["model_score"]
                    last_100_game_sum += info["model_score"]

                if len(game_score_history) > 100:
                    average_game_scores.append(last_100_game_sum / 100)

                if "won" in info:
                    num_wins += 1

        # temporary plot of game score
        if i % 100 == 0:
            update_game_score_plot(plot_title)
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        # print("episode:", i, "score:", score, "average score %.1f" % avg_score)

    torch.save(
        agent.Q_eval.state_dict(),
        f"models/model_{n_observations}_{n_actions}_opponent={opponent_type}_gamma={agent.gamma}_lr={agent.lr}_eps_decay={agent.eps_decay}_mem_size={agent.memory_size}_batch_size={agent.batch_size}_reward={reward_system}_punish={punish_not_rolling}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_dqn1.pt",
    )

    # full plot of game score, epsilon, and invalid actions
    fig = full_plot(plot_title, game_score_history, eps_history, invalid_action_history)
    # save the plot to file
    fig.savefig(
        f"figures/figure_{n_observations}_{n_actions}_opponent={opponent_type}_gamma={agent.gamma}_lr={agent.lr}_eps_decay={agent.eps_decay}_mem_size={agent.memory_size}_batch_size={agent.batch_size}_reward={reward_system}_punish={punish_not_rolling}_{time.strftime('%Y-%m-%d_%H-%M-%S')}_dqn1.png",
        bbox_inches="tight",
        dpi=300,
    )

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
