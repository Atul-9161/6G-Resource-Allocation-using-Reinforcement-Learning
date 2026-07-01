"""
ppo_trainer.py
--------------
Proximal Policy Optimization (PPO) training loop.

Key design choice: this environment is a *contextual bandit* dressed as an
episodic MDP -- every frame presents a brand-new, independent random context
(demand/noise/distance), so the current action has zero influence on future
frames. Two things follow from that:

1. gamma=0.0 -- there's nothing to bootstrap across steps for, since future
   reward genuinely doesn't depend on the current action. Using gamma>0 here
   just injects noise into the advantage estimate for no benefit.
2. Updates are batched across many episodes (episodes_per_update) rather than
   updating on a single 30-step episode at a time. A single episode is too
   small/noisy a sample to compute a reliable policy gradient from; pooling
   several episodes per update dramatically reduces gradient variance.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .model import ActorCritic
from .gae import compute_gae


class PPOTrainer:
    def __init__(
        self,
        env,
        state_dim,
        num_channels,
        num_users,
        lr=1e-4,
        gamma=0.0,
        gae_lambda=0.95,
        clip_eps=0.2,
        ppo_epochs=4,
        value_coef=0.5,
        entropy_coef_start=0.02,
        entropy_coef_end=0.0,
        frames_per_episode=30,
        episodes_per_update=10,
        minibatch_size=64,
    ):
        self.env = env
        self.model = ActorCritic(input_dim=state_dim, num_channels=num_channels, num_users=num_users)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.ppo_epochs = ppo_epochs
        self.value_coef = value_coef
        self.entropy_coef_start = entropy_coef_start
        self.entropy_coef_end = entropy_coef_end
        self.frames_per_episode = frames_per_episode
        self.episodes_per_update = episodes_per_update
        self.minibatch_size = minibatch_size

    def _entropy_coef(self, update_idx, num_updates):
        progress = update_idx / max(1, num_updates - 1)
        return self.entropy_coef_start + progress * (self.entropy_coef_end - self.entropy_coef_start)

    def _collect_rollout(self, num_episodes):
        """Run several episodes back-to-back under the current policy,
        pooling all steps into one buffer. Returns the buffer plus a list
        of per-episode summary metrics (for logging/plotting)."""
        states, actions, log_probs, values, rewards, dones = [], [], [], [], [], []
        episode_metrics = []

        for _ in range(num_episodes):
            state, _ = self.env.reset()
            ep_reward = 0.0
            ep_fairness, ep_qos, ep_sinr = [], [], []

            for frame in range(self.frames_per_episode):
                action, log_prob, value = self.model.act(state, greedy=False)
                next_state, reward, _, _, info = self.env.step(action)

                states.append(state)
                actions.append(action)
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(1.0 if frame == self.frames_per_episode - 1 else 0.0)

                ep_reward += reward
                ep_fairness.append(info["fairness_index"])
                ep_qos.append(info["qos_satisfied"])
                ep_sinr.append(info["avg_sinr_db"])

                state = next_state

            episode_metrics.append({
                "reward": ep_reward,
                "fairness": float(np.mean(ep_fairness)),
                "qos": float(np.mean(ep_qos)),
                "sinr": float(np.mean(ep_sinr)),
            })

        buffer = {
            "states": states,
            "actions": actions,
            "log_probs": log_probs,
            "values": values,
            "rewards": rewards,
            "dones": dones,
        }
        return buffer, episode_metrics

    def _update(self, buffer, entropy_coef):
        advantages, returns = compute_gae(
            buffer["rewards"], buffer["values"], buffer["dones"], self.gamma, self.gae_lambda
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        states_t = torch.FloatTensor(np.array(buffer["states"]))
        actions_t = torch.LongTensor(np.array(buffer["actions"]))
        old_log_probs_t = torch.FloatTensor(buffer["log_probs"])
        advantages_t = torch.FloatTensor(advantages)
        returns_t = torch.FloatTensor(returns)

        n = states_t.shape[0]
        batch_size = min(self.minibatch_size, n)

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(n)
            for start in range(0, n, batch_size):
                idx = perm[start:start + batch_size]

                new_log_probs, new_values, entropy = self.model.evaluate_actions(
                    states_t[idx], actions_t[idx]
                )
                ratio = torch.exp(new_log_probs - old_log_probs_t[idx])

                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(new_values, returns_t[idx])
                entropy_loss = -entropy.mean()

                loss = policy_loss + self.value_coef * value_loss + entropy_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()

    def train(self, num_episodes, log_every=None, verbose=True):
        num_updates = max(1, num_episodes // self.episodes_per_update)
        log_every = log_every or max(1, num_updates // 20)
        history = {"reward": [], "fairness": [], "qos": [], "sinr": []}

        episodes_done = 0
        update_idx = 0
        while episodes_done < num_episodes:
            k = min(self.episodes_per_update, num_episodes - episodes_done)
            entropy_coef = self._entropy_coef(update_idx, num_updates)

            buffer, episode_metrics = self._collect_rollout(k)
            self._update(buffer, entropy_coef)

            for m in episode_metrics:
                for key in history:
                    history[key].append(m[key])

            episodes_done += k
            update_idx += 1

            if verbose and update_idx % log_every == 0:
                recent = episode_metrics[-1]
                print(
                    f"Episode {episodes_done}/{num_episodes} | "
                    f"Reward: {recent['reward']:.1f} | "
                    f"Fairness: {recent['fairness']:.3f} | "
                    f"Avg QoS met/frame: {recent['qos']:.2f}/{self.env.num_users} | "
                    f"Avg SINR: {recent['sinr']:.1f} dB"
                )

        return history

