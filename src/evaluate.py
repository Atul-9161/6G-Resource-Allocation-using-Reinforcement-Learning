"""
evaluate.py
-----------
Runs a fixed policy deterministically over a number of evaluation episodes
and reports average reward, fairness, QoS satisfaction, and SINR -- used to
produce a fair, apples-to-apples comparison between PPO and the baselines.
"""

import numpy as np


def evaluate_policy(env, act_fn, num_episodes=50, frames_per_episode=30):
    rewards, fairness, qos, sinr = [], [], [], []

    for _ in range(num_episodes):
        state, _ = env.reset()
        ep_reward = 0.0
        ep_fair, ep_qos, ep_sinr = [], [], []

        for _frame in range(frames_per_episode):
            action = act_fn(state)
            state, reward, _, _, info = env.step(action)
            ep_reward += reward
            ep_fair.append(info["fairness_index"])
            ep_qos.append(info["qos_satisfied"])
            ep_sinr.append(info["avg_sinr_db"])

        rewards.append(ep_reward)
        fairness.append(np.mean(ep_fair))
        qos.append(np.mean(ep_qos))
        sinr.append(np.mean(ep_sinr))

    return {
        "reward": float(np.mean(rewards)),
        "fairness": float(np.mean(fairness)),
        "qos": float(np.mean(qos)),
        "sinr": float(np.mean(sinr)),
    }


def print_comparison_table(results: dict, num_users: int):
    """results: {policy_name: stats_dict} as returned by evaluate_policy."""
    print(f"{'Policy':<28}{'Reward':>10}{'Fairness':>12}{'QoS/' + str(num_users):>10}{'SINR(dB)':>12}")
    for name, stats in results.items():
        print(
            f"{name:<28}{stats['reward']:>10.1f}{stats['fairness']:>12.3f}"
            f"{stats['qos']:>10.2f}{stats['sinr']:>12.1f}"
        )
