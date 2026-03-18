"""Gated Transformer-XL (GTrXL) for reinforcement learning.

Based on "Stabilizing Transformers for Reinforcement Learning" (Parisotto et al., 2020).
Key modifications over standard Transformer-XL:
  1. Pre-layer normalization (LayerNorm on inputs to sublayers)
  2. GRU gating (replaces residual connections for training stability)
  3. Memory mechanism (caches past hidden states across rollout segments)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GRUGate(nn.Module):
    """GRU-style gating to replace residual connections.

    gate = sigmoid(W_y @ y + W_x @ x + bias)
    output = gate * x + (1 - gate) * y

    With bias=2.0, gate ≈ 0.88, so output starts close to residual (x).
    """
    def __init__(self, d_model, bias_init=2.0):
        super().__init__()
        self.w_y = nn.Linear(d_model, d_model, bias=False)
        self.w_x = nn.Linear(d_model, d_model)
        nn.init.constant_(self.w_x.bias, bias_init)

    def forward(self, x, y):
        gate = torch.sigmoid(self.w_y(y) + self.w_x(x))
        return gate * x + (1 - gate) * y


class GTrXLLayer(nn.Module):
    """Single Gated Transformer-XL layer with pre-norm and GRU gating."""

    def __init__(self, d_model, nhead, d_ff, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.gate1 = GRUGate(d_model)
        self.gate2 = GRUGate(d_model)

    def forward(self, x, memory=None, attn_mask=None):
        """
        x: (seq_len, batch, d_model) - current segment
        memory: (mem_len, batch, d_model) or None - cached past states
        attn_mask: optional pre-computed mask (overrides default causal mask)

        Returns: output (seq_len, batch, d_model)
        """
        # Pre-norm + attention
        x_norm = self.norm1(x)
        if memory is not None and memory.size(0) > 0:
            mem_norm = self.norm1(memory)
            kv = torch.cat([mem_norm, x_norm], dim=0)
        else:
            kv = x_norm

        if attn_mask is None:
            # Default causal mask: prevent attending to future positions
            seq_len = x.size(0)
            total_len = kv.size(0)
            # mask[i,j] = True means position i CANNOT attend to position j
            attn_mask = torch.triu(
                torch.ones(seq_len, total_len, device=x.device, dtype=torch.bool),
                diagonal=total_len - seq_len + 1
            )

        attn_out, _ = self.attn(x_norm, kv, kv, attn_mask=attn_mask)
        x = self.gate1(x, attn_out)

        # Pre-norm + FFN
        ff_out = self.ff(self.norm2(x))
        x = self.gate2(x, ff_out)

        return x


class PPO_GTrXL_Agent(nn.Module):
    """PPO agent with Gated Transformer-XL memory."""

    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_size=256,
        num_heads=8,
        num_transformer_layers=3,
        memory_len=64,
        num_mlp_layers=2,
        d_ff=None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_len = memory_len
        self.num_transformer_layers = num_transformer_layers

        if d_ff is None:
            d_ff = hidden_size * 4

        # Input projection: obs → hidden_size
        proj_layers = [layer_init(nn.Linear(obs_dim, hidden_size)), nn.ReLU()]
        for _ in range(num_mlp_layers - 1):
            proj_layers.extend([layer_init(nn.Linear(hidden_size, hidden_size)), nn.ReLU()])
        self.input_proj = nn.Sequential(*proj_layers)

        # Positional encoding (sinusoidal, large enough for memory + segment)
        self.register_buffer('pos_enc', self._make_pos_encoding(memory_len + 256, hidden_size))

        # Transformer-XL layers
        self.layers = nn.ModuleList([
            GTrXLLayer(hidden_size, num_heads, d_ff)
            for _ in range(num_transformer_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_size)

        # Actor/Critic heads
        self.actor_head = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)
        self.critic_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)

    @staticmethod
    def _make_pos_encoding(max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (max_len, d_model)

    def init_memory(self, batch_size, device):
        """Initialize empty memory for each transformer layer."""
        return [torch.zeros(0, batch_size, self.hidden_size, device=device)
                for _ in range(self.num_transformer_layers)]

    def _add_pos_encoding(self, x, memory_offset=0):
        """Add sinusoidal positional encoding to sequence x."""
        seq_len = x.size(0)
        positions = torch.arange(memory_offset, memory_offset + seq_len, device=x.device)
        return x + self.pos_enc[positions].unsqueeze(1)

    def get_states(self, x, memory, done):
        """
        Process observations step-by-step with done-aware memory.

        x: (T*B, obs_dim) - flattened observations
        memory: list of (mem_len, B, hidden_size) per layer
        done: (T*B,) - flattened done flags

        Returns: (T*B, hidden_size), updated_memory
        """
        # Determine batch size from memory (always has batch dim, even when empty)
        batch_size = memory[0].size(1)

        # Project input and reshape to (T, B, H)
        h = self.input_proj(x)
        T = h.size(0) // batch_size
        h = h.reshape(T, batch_size, self.hidden_size)
        done = done.reshape(T, batch_size)

        # Process step-by-step (like LSTM) to handle episode boundaries
        outputs = []
        for t in range(T):
            # Reset memory for environments that are done
            d = done[t].float().view(1, -1, 1)  # (1, B, 1)
            memory = [m * (1 - d) for m in memory]

            # Current step embedding with positional encoding
            h_t = h[t:t+1]  # (1, B, H)
            mem_len = memory[0].size(0) if memory[0].size(0) > 0 else 0
            h_t = self._add_pos_encoding(h_t, memory_offset=mem_len)

            # Forward through transformer layers
            new_memory = []
            layer_input = h_t
            for i, layer in enumerate(self.layers):
                mem = memory[i] if memory[i].size(0) > 0 else None
                # Add positional encoding to memory too
                if mem is not None:
                    mem_with_pos = self._add_pos_encoding(mem, memory_offset=0)
                else:
                    mem_with_pos = None

                layer_output = layer(layer_input, memory=mem_with_pos)

                # Update memory: append current layer's INPUT (before this layer)
                if mem is not None:
                    new_mem = torch.cat([mem, layer_input.detach()], dim=0)
                else:
                    new_mem = layer_input.detach()
                if new_mem.size(0) > self.memory_len:
                    new_mem = new_mem[-self.memory_len:]
                new_memory.append(new_mem)

                layer_input = layer_output

            memory = new_memory
            out = self.final_norm(layer_output)  # (1, B, H)
            outputs.append(out)

        output = torch.cat(outputs, dim=0)  # (T, B, H)
        output = output.reshape(T * batch_size, self.hidden_size)
        return output, memory

    def _compute_train_mask(self, done, mem_len, T, batch_size, device):
        """Compute episode-aware causal attention mask for batched training.

        Returns mask of shape (B*nhead, T, mem_len+T) where True = block attention.
        """
        nhead = self.layers[0].nhead

        # episode_id[t, b] = cumulative count of done flags up to step t
        # done[t]=True means obs[t] starts a new episode
        episode_id = done.float().cumsum(dim=0)  # (T, B)
        episode_id_t = episode_id.permute(1, 0)  # (B, T)

        # Within-sequence: block future (causal) and cross-episode attention
        causal = torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1
        )  # (T, T)
        ep_mask = episode_id_t.unsqueeze(1) != episode_id_t.unsqueeze(2)  # (B, T_q, T_kv)
        seq_mask = causal.unsqueeze(0) | ep_mask  # (B, T, T)

        # Memory: block if query is in a different episode than the initial one (episode 0)
        if mem_len > 0:
            mem_mask = (episode_id_t.unsqueeze(2) != 0).expand(-1, -1, mem_len)  # (B, T, mem_len)
            full_mask = torch.cat([mem_mask, seq_mask], dim=2)  # (B, T, mem_len+T)
        else:
            full_mask = seq_mask  # (B, T, T)

        # Expand for multi-head attention: (B*nhead, T, S)
        full_mask = full_mask.unsqueeze(1).expand(-1, nhead, -1, -1)
        full_mask = full_mask.reshape(batch_size * nhead, T, mem_len + T)
        return full_mask

    def _compute_positions(self, done, mem_len, T, batch_size, device):
        """Compute per-environment positional indices accounting for done resets."""
        positions = torch.zeros(T, batch_size, device=device, dtype=torch.long)
        running_pos = torch.full((batch_size,), mem_len, device=device, dtype=torch.long)
        for t in range(T):
            running_pos = torch.where(done[t].bool(), torch.zeros_like(running_pos), running_pos)
            positions[t] = running_pos
            running_pos = running_pos + 1
        return positions  # (T, B)

    def get_states_train(self, x, memory, done):
        """Process all T steps in parallel for training (batched attention).

        Much faster than step-by-step get_states since it avoids the T-loop
        through transformer layers. Uses episode-aware attention masking.
        """
        batch_size = memory[0].size(1)
        h = self.input_proj(x)
        T = h.size(0) // batch_size
        h = h.reshape(T, batch_size, self.hidden_size)
        done = done.reshape(T, batch_size)
        mem_len = memory[0].size(0)

        # Compute episode-aware attention mask
        attn_mask = self._compute_train_mask(done, mem_len, T, batch_size, x.device)

        # Compute per-environment positional encoding
        positions = self._compute_positions(done, mem_len, T, batch_size, x.device)
        h = h + self.pos_enc[positions]  # (T, B, H)

        # Process through all transformer layers in parallel
        layer_input = h  # (T, B, H)
        for i, layer in enumerate(self.layers):
            mem = memory[i] if memory[i].size(0) > 0 else None
            if mem is not None:
                mem_with_pos = self._add_pos_encoding(mem, memory_offset=0)
            else:
                mem_with_pos = None
            layer_output = layer(layer_input, memory=mem_with_pos, attn_mask=attn_mask)
            layer_input = layer_output

        output = self.final_norm(layer_output)  # (T, B, H)
        output = output.reshape(T * batch_size, self.hidden_size)
        return output, memory  # memory unchanged during training

    def get_value(self, x, memory, done):
        hidden, _ = self.get_states(x, memory, done)
        return self.critic_head(hidden)

    def get_action_and_value(self, x, memory, done, action=None):
        if action is not None:
            # Training: use batched path (all T steps in parallel)
            hidden, memory = self.get_states_train(x, memory, done)
        else:
            # Rollout: use step-by-step path (correct memory updates)
            hidden, memory = self.get_states(x, memory, done)
        logits = self.actor_head(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(hidden), memory
