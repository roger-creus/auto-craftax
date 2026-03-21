import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class CraftaxObsEncoder(nn.Module):
    """Structured observation encoder for Craftax-Symbolic-v1.

    Splits the 8268-dim flat obs into:
      - Spatial map (9x11x83): processed with CNN
      - Player stats (51): processed with MLP
    Then combines into a single hidden_size vector.
    """

    MAP_H, MAP_W, MAP_C = 9, 11, 83
    MAP_DIM = MAP_H * MAP_W * MAP_C  # 8217
    STATS_DIM = 51

    def __init__(self, hidden_size, extra_stats_dim=0, use_bn=False):
        super().__init__()
        self.extra_stats_dim = extra_stats_dim
        actual_stats_dim = self.STATS_DIM + extra_stats_dim
        total_obs_dim = self.MAP_DIM + actual_stats_dim
        # Optional BatchNorm on raw observation (PQN paper: critical for sparse symbolic obs)
        self.input_bn = nn.BatchNorm1d(total_obs_dim) if use_bn else None
        # Spatial map encoder: 2-layer CNN + adaptive pool
        self.map_cnn = nn.Sequential(
            nn.Conv2d(self.MAP_C, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 4)),  # → (B, 64, 3, 4) = 768
        )
        map_flat_dim = 64 * 3 * 4
        self.map_proj = nn.Sequential(
            layer_init(nn.Linear(map_flat_dim, hidden_size)),
            nn.ReLU(),
        )

        # Player stats encoder: 2-layer MLP
        stats_hidden = max(hidden_size // 2, 64)
        self.stats_mlp = nn.Sequential(
            layer_init(nn.Linear(actual_stats_dim, stats_hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(stats_hidden, stats_hidden)),
            nn.ReLU(),
        )

        # Combine map + stats → hidden_size
        self.combine = nn.Sequential(
            layer_init(nn.Linear(hidden_size + stats_hidden, hidden_size)),
            nn.ReLU(),
        )

    def forward(self, obs):
        # Optional input BatchNorm (normalize sparse symbolic features)
        if self.input_bn is not None:
            orig_shape = obs.shape
            obs = self.input_bn(obs.reshape(-1, orig_shape[-1])).reshape(orig_shape)
        # Split observation
        map_flat = obs[..., :self.MAP_DIM]
        stats = obs[..., self.MAP_DIM:]

        # Encode map spatially
        batch_shape = map_flat.shape[:-1]
        map_2d = map_flat.reshape(-1, self.MAP_H, self.MAP_W, self.MAP_C)
        map_2d = map_2d.permute(0, 3, 1, 2)  # (B, C, H, W)
        map_feat = self.map_cnn(map_2d).flatten(1)
        map_emb = self.map_proj(map_feat)
        map_emb = map_emb.reshape(*batch_shape, -1)

        # Encode stats
        stats_emb = self.stats_mlp(stats)

        # Combine
        return self.combine(torch.cat([map_emb, stats_emb], dim=-1))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation_fn, use_ln=False):
        super().__init__()

        layers = []
        for _ in range(num_layers):
            layers.append(layer_init(nn.Linear(input_size, hidden_size)))
            if use_ln:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(activation_fn())
            input_size = hidden_size
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class ResidualMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, activation_fn, use_ln=False):
        super().__init__()
        self.mlp1 = MLP(input_size, hidden_size, num_layers, activation_fn, use_ln)
        self.mlp2 = MLP(hidden_size, hidden_size, num_layers, activation_fn, use_ln)
        self.need_proj = input_size != hidden_size
        if self.need_proj:
            self.proj = layer_init(nn.Linear(input_size, hidden_size))
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        y = self.mlp1(x)
        y2 = self.mlp2(y)
        res = self.proj(x)
        return y2 + res

class PPO_LSTM_Agent(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_size: int = 512,
        num_layers: int = 5,
        activation_fn: nn.Module = nn.Tanh,
        use_ln: bool = False,
        use_bn: bool = False,
        mlp_class=MLP,
        use_structured_obs: bool = False,
        use_popart: bool = False,
        use_gru: bool = False,
        extra_stats_dim: int = 0,
        separate_critic: bool = False,
        ac_layer_size: int = -1,
    ):
        super().__init__()

        self.mlp_class = mlp_class
        self.use_popart = use_popart
        self.use_gru = use_gru
        self.separate_critic = separate_critic

        # ac_layer_size controls post-RNN FC width independently from GRU hidden
        ac_size = ac_layer_size if ac_layer_size > 0 else hidden_size

        if use_structured_obs:
            pre_layers = max(num_layers // 2 - 1, 1)
            self.shared_pre_lstm = nn.Sequential(
                CraftaxObsEncoder(hidden_size, extra_stats_dim=extra_stats_dim, use_bn=use_bn),
                mlp_class(hidden_size, hidden_size, pre_layers, activation_fn, use_ln=use_ln),
            )
        else:
            self.shared_pre_lstm = mlp_class(obs_dim, hidden_size, num_layers // 2, activation_fn, use_ln=use_ln)

        if use_gru:
            self.rnn = nn.GRU(hidden_size, hidden_size)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size)
        for name, param in self.rnn.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)

        self.shared_post_lstm = mlp_class(hidden_size, ac_size, num_layers // 2, activation_fn, use_ln=use_ln)

        # Separate critic trunk: independent post-RNN MLP for value estimation
        if separate_critic:
            self.critic_post_lstm = mlp_class(hidden_size, ac_size, num_layers // 2, activation_fn, use_ln=use_ln)

        if use_popart:
            from src.models.gtrxl import PopArtLayer
            self.critic_head = PopArtLayer(ac_size, 1)
        else:
            self.critic_head = layer_init(nn.Linear(ac_size, 1), std=1.0)
        self.actor_head = layer_init(nn.Linear(ac_size, n_actions), std=0.01)

        # Store ac_size for head initialization
        self._ac_size = ac_size

        # Intrinsic value head for dual-value RND (separate V_int from V_ext)
        self.critic_head_int = None

        # Auxiliary prediction head (e.g., predict kill count on current floor)
        self.aux_head = None

    def init_intrinsic_value_head(self, hidden_size=None):
        """Initialize separate value head for intrinsic rewards (dual-value RND)."""
        size = self._ac_size if hasattr(self, '_ac_size') else (hidden_size or 512)
        self.critic_head_int = layer_init(nn.Linear(size, 1), std=1.0)

    def init_aux_head(self, hidden_size=None, n_targets=1):
        """Initialize auxiliary prediction head for kill count / combat state prediction."""
        size = self._ac_size if hasattr(self, '_ac_size') else (hidden_size or 512)
        self.aux_head = nn.Sequential(
            nn.Linear(size, 64),
            nn.ReLU(),
            nn.Linear(64, n_targets),
            nn.Sigmoid(),
        )

    def get_states(self, x, lstm_state, done):
        hidden = self.shared_pre_lstm(x)

        # RNN logic (LSTM or GRU)
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.rnn.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        if self.use_gru:
            gru_state = lstm_state[0]  # only use h, ignore dummy c
            for h, d in zip(hidden, done):
                h, gru_state = self.rnn(
                    h.unsqueeze(0),
                    (1.0 - d.float()).view(1, -1, 1) * gru_state,
                )
                new_hidden += [h]
            lstm_state = (gru_state, torch.zeros_like(gru_state))  # return (h, dummy)
        else:
            for h, d in zip(hidden, done):
                h, lstm_state = self.rnn(
                    h.unsqueeze(0),
                    (
                        (1.0 - d.float()).view(1, -1, 1) * lstm_state[0],
                        (1.0 - d.float()).view(1, -1, 1) * lstm_state[1],
                    ),
                )
                new_hidden += [h]
        rnn_out = torch.flatten(torch.cat(new_hidden), 0, 1)
        actor_hidden = self.shared_post_lstm(rnn_out)
        critic_hidden = self.critic_post_lstm(rnn_out) if self.separate_critic else actor_hidden
        return actor_hidden, critic_hidden, lstm_state

    def get_value(self, x, lstm_state, done, denormalize=False):
        _, critic_hidden, _ = self.get_states(x, lstm_state, done)
        value = self.critic_head(critic_hidden)
        if denormalize and self.use_popart:
            value = self.critic_head.denormalize(value)
        if self.critic_head_int is not None:
            value_int = self.critic_head_int(critic_hidden)
            return value, value_int
        return value

    def get_action_and_value(self, x, lstm_state, done, action=None, denormalize=False):
        actor_hidden, critic_hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor_head(actor_hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        value = self.critic_head(critic_hidden)
        if denormalize and self.use_popart:
            value = self.critic_head.denormalize(value)
        value_int = self.critic_head_int(critic_hidden) if self.critic_head_int is not None else None
        aux_pred = self.aux_head(actor_hidden) if self.aux_head is not None else None
        return action, probs.log_prob(action), probs.entropy(), value, lstm_state, aux_pred, value_int

    def sample_action(self, x, lstm_state, done):
        actor_hidden, _, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor_head(actor_hidden)
        probs = Categorical(logits=logits)
        return probs.sample(), lstm_state
    
    
class PPO_Agent(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_size: int = 512,
        num_layers: int = 5,
        activation_fn: nn.Module = nn.Tanh,
        use_ln: bool = False,
        use_structured_obs: bool = False,
    ):
        super().__init__()

        if use_structured_obs:
            self.critic_trunk = nn.Sequential(CraftaxObsEncoder(hidden_size), MLP(hidden_size, hidden_size, max(num_layers - 2, 1), activation_fn, use_ln=use_ln))
            self.actor_trunk = nn.Sequential(CraftaxObsEncoder(hidden_size), MLP(hidden_size, hidden_size, max(num_layers - 2, 1), activation_fn, use_ln=use_ln))
        else:
            self.critic_trunk = MLP(obs_dim, hidden_size, num_layers, activation_fn, use_ln=use_ln)
            self.actor_trunk = MLP(obs_dim, hidden_size, num_layers, activation_fn, use_ln=use_ln)
        self.critic_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.actor_head = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)

    def get_value(self, x):
        return self.critic_head(self.critic_trunk(x))

    def get_action_and_value(self, x, action=None):
        logits = self.actor_head(self.actor_trunk(x))
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.get_value(x)
    
    def sample_action(self, x):
        logits = self.actor_head(self.actor_trunk(x))
        probs = Categorical(logits=logits)
        return probs.sample()
    
    
class PQN_Agent(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_size: int = 512,
        num_layers: int = 5,
        activation_fn: nn.Module = nn.Tanh,
        use_ln: bool = True,
    ):
        super().__init__()
        
    
        self.q_net_trunk = MLP(obs_dim, hidden_size, num_layers, activation_fn, use_ln=use_ln)
        self.q_net_head = layer_init(nn.Linear(hidden_size, n_actions))

    def forward(self, x):
        return self.q_net_head(self.q_net_trunk(x))
    
    def sample_action(self, x):
        q_values = self.q_net_head(self.q_net_trunk(x))
        return torch.argmax(q_values, dim=-1)
    
class PQN_LSTM_Agent(nn.Module):
    def __init__(
        self,
        obs_dim,
        n_actions,
        hidden_size: int = 512,
        num_layers: int = 5,
        activation_fn: nn.Module = nn.Tanh,
        use_ln: bool = True,
    ):
        super().__init__()
        
    
        self.pre_lstm = MLP(obs_dim, hidden_size, num_layers // 2, activation_fn, use_ln=use_ln)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        self.post_lstm = MLP(hidden_size, hidden_size, num_layers // 2, activation_fn, use_ln=use_ln)
        self.q_net_head = layer_init(nn.Linear(hidden_size, n_actions))

    def get_states(self, x, lstm_state, done):
        hidden = self.pre_lstm(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size))
        done = done.reshape((-1, batch_size))
        new_hidden = []
        for h, d in zip(hidden, done):
            h, lstm_state = self.lstm(
                h.unsqueeze(0),
                (
                    (1.0 - d.float()).view(1, -1, 1) * lstm_state[0],
                    (1.0 - d.float()).view(1, -1, 1) * lstm_state[1],
                ),
            )
            new_hidden += [h]
        new_hidden = torch.flatten(torch.cat(new_hidden), 0, 1)
        new_hidden = self.post_lstm(new_hidden)
        return new_hidden, lstm_state

    def forward(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        return self.q_net_head(hidden), lstm_state
    
    def sample_action(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        q_values = self.q_net_head(hidden)
        return torch.argmax(q_values, dim=-1), lstm_state
    