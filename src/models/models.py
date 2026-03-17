import torch
import numpy as np
from torch import nn
from torch.distributions import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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
        mlp_class=MLP,
    ):
        super().__init__()
        
        self.mlp_class = mlp_class

        self.shared_pre_lstm = mlp_class(obs_dim, hidden_size, num_layers // 2, activation_fn, use_ln=use_ln)
        
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        for name, param in self.lstm.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, 1.0)
        
        self.shared_post_lstm = mlp_class(hidden_size, hidden_size, num_layers // 2, activation_fn, use_ln=use_ln)
        
        self.critic_head = layer_init(nn.Linear(hidden_size, 1), std=1.0)
        self.actor_head = layer_init(nn.Linear(hidden_size, n_actions), std=0.01)

    def get_states(self, x, lstm_state, done):
        hidden = self.shared_pre_lstm(x)

        # LSTM logic
        batch_size = lstm_state[0].shape[1]
        hidden = hidden.reshape((-1, batch_size, self.lstm.input_size)) # 16, 64, 512
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
        new_hidden = self.shared_post_lstm(new_hidden)
        return new_hidden, lstm_state

    def get_value(self, x, lstm_state, done):
        hidden, _ = self.get_states(x, lstm_state, done)
        return self.critic_head(hidden)

    def get_action_and_value(self, x, lstm_state, done, action=None):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor_head(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic_head(hidden), lstm_state
    
    def sample_action(self, x, lstm_state, done):
        hidden, lstm_state = self.get_states(x, lstm_state, done)
        logits = self.actor_head(hidden)
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
    ):
        super().__init__()
        
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
    