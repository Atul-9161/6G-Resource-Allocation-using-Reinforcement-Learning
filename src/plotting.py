"""
plotting.py
-----------
Generates the training-curve figure: raw + smoothed PPO reward, plus
wireless quality metrics (fairness index, QoS satisfaction rate) over
training episodes.
"""

import numpy as np
import matplotlib.pyplot as plt


def moving_average(values, window=20):
    values = np.array(values, dtype=np.float64)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, smoothed[0])
    return np.concatenate([pad, smoothed])


def plot_training_results(history, num_users, out_path="tele6g_training_results.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 9))

    reward_smooth = moving_average(history["reward"])
    axes[0].plot(history["reward"], color="blue", alpha=0.15, linewidth=1)
    axes[0].plot(reward_smooth, color="blue", linewidth=2, label="PPO reward (20-ep avg)")
    axes[0].set_title("PPO Training Reward Over Time")
    axes[0].set_xlabel("Training Episode")
    axes[0].set_ylabel("Total Episode Reward")
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    fairness_smooth = moving_average(history["fairness"])
    qos_smooth = moving_average(np.array(history["qos"]) / num_users)
    axes[1].plot(fairness_smooth, color="green", linewidth=2, label="Fairness index (Jain's, 20-ep avg)")
    axes[1].plot(qos_smooth, color="purple", linewidth=2, label="QoS satisfaction rate (20-ep avg)")
    axes[1].set_title("Wireless Quality Metrics Over Training")
    axes[1].set_xlabel("Training Episode")
    axes[1].set_ylabel("Score (0-1)")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nTraining curves saved to '{out_path}'")
