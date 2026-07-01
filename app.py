"""
app.py
------
Main entry point: trains a PPO policy on the 6G resource-allocation
environment, plots training curves, then runs a deterministic head-to-head
evaluation of PPO vs. two baseline scheduling policies.

Run:
    python app.py
"""

import random
import numpy as np
import torch

from src.environment import Tele6GResourceEnv
from src.ppo_trainer import PPOTrainer
from src.baselines import random_policy, greedy_lowest_noise_policy
from src.evaluate import evaluate_policy, print_comparison_table
from src.plotting import plot_training_results


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SEED = 42
NUM_EPISODES = 2000       # 1000-5000 recommended; higher = longer runtime, smoother curves
FRAMES_PER_EPISODE = 30
EVAL_EPISODES = 50


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(SEED)

    env = Tele6GResourceEnv()
    state_dim = env.observation_space.shape[0]

    trainer = PPOTrainer(
        env=env,
        state_dim=state_dim,
        num_channels=env.num_channels,
        num_users=env.num_users,
        frames_per_episode=FRAMES_PER_EPISODE,
    )

    print("Initializing PPO training for AI-Native 6G Resource Allocation...")
    print(f"Episodes: {NUM_EPISODES} | Frames/episode: {FRAMES_PER_EPISODE}\n")
    history = trainer.train(num_episodes=NUM_EPISODES)

    plot_training_results(history, num_users=env.num_users)

    print(f"\nRunning final head-to-head evaluation "
          f"({EVAL_EPISODES} episodes, deterministic policies)...\n")

    ppo_act = lambda s: trainer.model.act(s, greedy=True)[0]
    results = {
        "PPO (learned)": evaluate_policy(env, ppo_act, num_episodes=EVAL_EPISODES, frames_per_episode=FRAMES_PER_EPISODE),
        "Random assignment": evaluate_policy(env, lambda s: random_policy(env, s), num_episodes=EVAL_EPISODES, frames_per_episode=FRAMES_PER_EPISODE),
        "Greedy lowest-noise": evaluate_policy(env, lambda s: greedy_lowest_noise_policy(env, s), num_episodes=EVAL_EPISODES, frames_per_episode=FRAMES_PER_EPISODE),
    }
    print_comparison_table(results, num_users=env.num_users)


if __name__ == "__main__":
    main()
