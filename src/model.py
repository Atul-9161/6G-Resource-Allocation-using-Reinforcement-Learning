"""
model.py
--------
Actor-critic neural network for PPO. A shared backbone feeds:
  - one classification "head" per user (the actor), each choosing a channel
  - one scalar value head (the critic), estimating expected return
"""

import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, input_dim, num_channels, num_users, hidden_dim=128):
        super().__init__()
        self.num_users = num_users
        self.num_channels = num_channels

        self.shared_network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
        )
        self.actor_heads = nn.ModuleList(
            [nn.Linear(hidden_dim // 2, num_channels) for _ in range(num_users)]
        )
        self.critic_head = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        features = self.shared_network(x)
        logits = torch.stack([head(features) for head in self.actor_heads], dim=1)
        value = self.critic_head(features).squeeze(-1)
        return logits, value

    def act(self, state, greedy=False):
        """Sample (or greedily pick) an action for a single state.

        Returns (action, log_prob, value) as plain Python / numpy types.
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits, value = self.forward(state_t)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=2) if greedy else dist.sample()
        log_prob = dist.log_prob(action).sum(dim=1)
        return (
            action.squeeze(0).numpy(),
            log_prob.squeeze(0).item(),
            value.squeeze(0).item(),
        )

    def evaluate_actions(self, states, actions):
        """Used during PPO updates: recompute log-probs/values/entropy for a
        batch of previously-taken actions, under the *current* policy."""
        logits, values = self.forward(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        return log_probs, values, entropy
