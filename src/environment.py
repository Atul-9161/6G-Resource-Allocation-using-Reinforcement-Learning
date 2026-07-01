"""
environment.py
---------------
Custom Gymnasium environment simulating a simplified 6G wireless cell.

4 mobile users (UEs) must each be assigned one of 3 frequency channels every
scheduling frame. Users sharing a channel interfere with one another
(co-channel interference, NOMA-style), which lowers their SINR and therefore
their achievable Shannon-capacity throughput. The reward balances total
throughput, fairness across users (Jain's Fairness Index), and how many
users met their QoS demand.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class Tele6GResourceEnv(gym.Env):
    def __init__(self, num_users=4, num_channels=3, max_power_watt=2.0, bandwidth=200e6):
        super().__init__()
        self.num_users = num_users
        self.num_channels = num_channels
        self.max_power_watt = max_power_watt
        self.bandwidth = bandwidth  # Hz, per channel

        # State: [demands] + [channel noise levels] + [user distances]
        state_dim = num_users + num_channels + num_users
        self.observation_space = spaces.Box(
            low=-100, high=500, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.MultiDiscrete([num_channels] * num_users)

        self.demands = None
        self.channel_noise = None
        self.distances = None
        self.state = None

    def _new_context(self):
        self.demands = np.random.uniform(30, 250, size=self.num_users)
        self.channel_noise = np.random.uniform(-90, -75, size=self.num_channels)
        self.distances = np.random.uniform(10, 100, size=self.num_users)
        self.state = np.concatenate(
            [self.demands, self.channel_noise, self.distances]
        ).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._new_context()
        return self.state, {}

    def step(self, action):
        action = np.asarray(action).reshape(-1)
        per_user_throughput = np.zeros(self.num_users)
        per_user_sinr_db = np.zeros(self.num_users)

        p_tx_dbm = 10 * np.log10(self.max_power_watt * 1000)
        path_loss_db = 20 * np.log10(self.distances) + 90.0
        p_rx_dbm = p_tx_dbm - path_loss_db
        p_rx_watts = 10 ** (p_rx_dbm / 10) / 1000

        for ch in range(self.num_channels):
            users_on_ch = np.where(action == ch)[0]
            if len(users_on_ch) == 0:
                continue
            noise_watts = 10 ** (self.channel_noise[ch] / 10) / 1000
            total_power_on_ch = p_rx_watts[users_on_ch].sum()

            for ue in users_on_ch:
                # Interference = received power of every OTHER user on this channel
                interference_watts = total_power_on_ch - p_rx_watts[ue]
                sinr = p_rx_watts[ue] / (noise_watts + interference_watts)
                per_user_sinr_db[ue] = 10 * np.log10(sinr + 1e-12)
                per_user_throughput[ue] = (self.bandwidth * np.log2(1 + sinr)) / 1e6

        qos_met = per_user_throughput >= self.demands
        qos_satisfied = int(qos_met.sum())
        total_throughput_mbps = float(per_user_throughput.sum())

        satisfaction_ratio = per_user_throughput / np.maximum(self.demands, 1e-6)
        fairness_index = (satisfaction_ratio.sum() ** 2) / (
            self.num_users * np.sum(satisfaction_ratio ** 2) + 1e-9
        )

        reward = (
            0.05 * total_throughput_mbps
            + 25.0 * fairness_index
            + 8.0 * qos_satisfied
        )

        info = {
            "fairness_index": fairness_index,
            "qos_satisfied": qos_satisfied,
            "avg_sinr_db": float(per_user_sinr_db.mean()),
            "total_throughput_mbps": total_throughput_mbps,
        }

        self._new_context()
        return self.state, reward, False, False, info
