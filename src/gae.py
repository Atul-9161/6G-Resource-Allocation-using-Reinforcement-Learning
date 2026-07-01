"""
gae.py
------
Generalized Advantage Estimation (GAE), used by PPO to compute low-variance
advantage estimates from a rollout of rewards and value estimates.
"""

import numpy as np


def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    Args:
        rewards: list/array of per-step rewards, length T
        values:  list/array of critic value estimates V(s_t), length T
        dones:   list/array of 0/1 episode-termination flags, length T
        gamma:   discount factor
        lam:     GAE smoothing parameter

    Returns:
        advantages: np.ndarray, length T
        returns:    np.ndarray, length T  (advantages + values, i.e. the
                    target for the value function)
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else 0.0
        next_non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        last_gae = delta + gamma * lam * next_non_terminal * last_gae
        advantages[t] = last_gae
    returns = advantages + np.array(values, dtype=np.float32)
    return advantages, returns
