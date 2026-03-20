"""Self-Imitation Learning (SIL) buffer and loss.

Based on Oh et al., "Self-Imitation Learning" (ICML 2018).
Stores high-return transitions and trains the policy to imitate them.

SIL loss = max(0, R - V(s)) * -log π(a|s)  (policy)
         + 0.5 * max(0, R - V(s))^2          (value)

Only transitions where R > V(s) are used — the agent only imitates
experiences that are better than its current value estimate.
"""

import torch
import numpy as np


class SILBuffer:
    """Fixed-size buffer storing high-return (obs, action, return) transitions."""

    def __init__(self, capacity, obs_dim, device):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, dtype=torch.long, device=device)
        self.returns = torch.zeros(capacity, device=device)
        self.size = 0
        self.ptr = 0  # circular pointer

    def add_batch(self, obs_batch, act_batch, ret_batch, threshold):
        """Add transitions with returns above threshold.

        Args:
            obs_batch: (N, obs_dim) observations
            act_batch: (N,) actions (long)
            ret_batch: (N,) returns
            threshold: only add transitions with return > threshold
        """
        mask = ret_batch > threshold
        if not mask.any():
            return 0

        good_obs = obs_batch[mask]
        good_act = act_batch[mask]
        good_ret = ret_batch[mask]
        n = good_obs.shape[0]

        if n == 0:
            return 0

        # If buffer is not full, just append
        if self.size < self.capacity:
            space = self.capacity - self.size
            n_add = min(n, space)
            self.obs[self.size:self.size + n_add] = good_obs[:n_add]
            self.actions[self.size:self.size + n_add] = good_act[:n_add]
            self.returns[self.size:self.size + n_add] = good_ret[:n_add]
            self.size += n_add
            # If we filled up and still have more, write circularly
            if n_add < n:
                remaining = n - n_add
                idx = torch.arange(remaining, device=self.device) % self.capacity
                self.obs[idx] = good_obs[n_add:]
                self.actions[idx] = good_act[n_add:]
                self.returns[idx] = good_ret[n_add:]
                self.ptr = (remaining % self.capacity)
        else:
            # Circular overwrite
            for i in range(n):
                self.obs[self.ptr] = good_obs[i]
                self.actions[self.ptr] = good_act[i]
                self.returns[self.ptr] = good_ret[i]
                self.ptr = (self.ptr + 1) % self.capacity

        return n

    def sample(self, batch_size):
        """Sample a random batch of transitions."""
        indices = torch.randint(0, self.size, (min(batch_size, self.size),), device=self.device)
        return self.obs[indices], self.actions[indices], self.returns[indices]

    def __len__(self):
        return self.size


def compute_sil_loss(agent, sil_buffer, sil_batch_size, device):
    """Compute SIL policy + value loss from buffer samples.

    Returns:
        sil_policy_loss: BC loss weighted by positive advantage
        sil_value_loss: MSE loss on positive advantages
        n_positive: number of transitions with positive advantage (for logging)
    """
    if len(sil_buffer) < sil_batch_size:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0

    obs, actions, returns = sil_buffer.sample(sil_batch_size)

    # Get current policy logprobs and values (feedforward, no recurrence)
    # We need to handle the LSTM agent in feedforward mode
    # Create dummy LSTM states (zeros) and dummy dones (all True to reset hidden state)
    batch_size = obs.shape[0]

    # Reshape to (1, batch, ...) to simulate single-step sequence
    obs_seq = obs.unsqueeze(0)  # (1, B, obs_dim)
    dones_seq = torch.ones(1, batch_size, device=device)  # all True = reset hidden

    # Zero-init hidden states
    num_layers = agent.rnn.num_layers
    hidden_size = agent.rnn.hidden_size
    if hasattr(agent.rnn, 'proj_size') and agent.rnn.proj_size > 0:
        h_size = agent.rnn.proj_size
    else:
        h_size = hidden_size

    h0 = torch.zeros(num_layers, batch_size, h_size, device=device)
    c0 = torch.zeros(num_layers, batch_size, hidden_size, device=device)
    lstm_state = (h0, c0)

    _, logprobs, _, values, _, _ = agent.get_action_and_value(
        obs_seq.reshape(-1, obs.shape[-1]),
        lstm_state,
        dones_seq.reshape(-1),
        actions,
    )

    values = values.squeeze(-1)

    # SIL advantage: only positive (return > value)
    advantages = (returns - values.detach()).clamp(min=0.0)

    # Count how many have positive advantage
    n_positive = (advantages > 0).sum().item()

    if n_positive == 0:
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device), 0

    # Policy loss: advantage-weighted negative log-likelihood
    sil_policy_loss = -(advantages * logprobs).mean()

    # Value loss: push value towards return (only for positive advantages)
    sil_value_loss = 0.5 * (advantages ** 2).mean()

    return sil_policy_loss, sil_value_loss, n_positive
