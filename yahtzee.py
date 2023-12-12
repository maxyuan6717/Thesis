from game import Yahtzee
from player_solitaire_optimal import OptimalPlayer
from player_solitaire_greedy import GreedyPlayer
from player_random import RandomPlayer

# from player_dqn import DQNPlayer
from player_dqn_updated import DQNPlayer

model_map = {
    "VS_RANDOM": "model_38_23_opponent=random_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_punish=False_2023-12-05_22-53-38_dqn2.pt",
    "VS_GREEDY": "model_38_23_opponent=greedy_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_punish=False_2023-12-05_23-42-48_dqn2.pt",
    "VS_OPTIMAL": "model_38_23_opponent=optimal_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_punish=False_2023-12-06_01-03-08_dqn2.pt",
    "FIXED": "model_38_23_opponent=random_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=fixed_score_goal=110_punish=False_2023-12-05_23-45-26_dqn2.pt",
    "FLEX": "model_38_23_opponent=random_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=flex_punish=False_2023-12-06_00-33-09_dqn2.pt",
    "VS_GREEDY_GRADIENT": "model_38_23_opponent=greedy_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_gradient_punish=False_2023-12-06_04-04-45_dqn2.pt",
    "VS_GREEDY_GRADIENT_LONG": "model_38_23_opponent=greedy_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_gradient_punish=False_2023-12-07_11-36-42_dqn2.pt",
    "VS_OPTIMAL_GRADIENT": "model_38_23_opponent=optimal_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_gradient_punish=False_2023-12-06_00-40-48_dqn2.pt",
    "VS_OPTIMAL_GRADIENT_LONG": "model_38_23_opponent=optimal_gamma=0.99_lr=0.001_eps_decay=0.9998_mem_size=20000_batch_size=100_reward=p2_gradient_punish=False_2023-12-07_12-10-10_dqn2.pt",
}

opponent_map = {
    "RANDOM": RandomPlayer,
    "GREEDY": GreedyPlayer,
    "OPTIMAL": OptimalPlayer,
}


def main():
    key = "VS_RANDOM"
    opponent = "RANDOM"
    print("PLAYING", key, "AGAINST", opponent)
    # player1 = OptimalPlayer()
    # player1 = GreedyPlayer()
    player1 = DQNPlayer(model_map[key])
    # player2 = OptimalPlayer()
    # player2 = GreedyPlayer()
    # player2 = RandomPlayer()
    player2 = opponent_map[opponent]()
    game = Yahtzee([player1, player2], 10000)
    game.play_games()


if __name__ == "__main__":
    main()
