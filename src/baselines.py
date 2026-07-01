"""
baselines.py
------------
Simple non-learned scheduling policies used to benchmark the trained PPO
policy against.
"""

import numpy as np


def random_policy(env, state):
    """Every user picks an independent random channel."""
    return np.random.randint(0, env.num_channels, size=env.num_users)


def greedy_lowest_noise_policy(env, state):
    """Naive heuristic: every user grabs the least-noisy channel.

    This deliberately causes congestion (all users pile onto one channel,
    maximizing interference) so the learned policy has a meaningful,
    realistic baseline to beat -- this is a common failure mode of simple
    heuristics in real scheduling systems.
    """
    best_channel = int(np.argmin(env.channel_noise))
    return np.full(env.num_users, best_channel)
