import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from kron_torch import Kron
from src.models.models import MLP, ResidualMLP


class RunningMeanStd:
    """Running mean and std tracker for observation normalization.
    Uses Welford's online algorithm for numerical stability."""
    def __init__(self, shape, device, eps=1e-8):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = eps

    def update(self, x):
        """Update running stats with a batch of observations. x shape: (batch, *shape)"""
        batch_mean = x.mean(dim=0)
        batch_var = x.var(dim=0, correction=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    def normalize(self, x, clip=10.0):
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -clip, clip)

def linear_schedule(start_e, end_e, duration, t):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)

def soft_value_from_q(q_values, alpha):
    return alpha * torch.logsumexp(q_values / alpha, dim=-1)

def get_optimizer_class(optimizer_name):
    if optimizer_name == "adam":
        return optim.Adam
    elif optimizer_name == "radam":
        return optim.RAdam
    elif optimizer_name == "kron":
        return Kron
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")

def get_activation_fn(activation_fn_name):
    if activation_fn_name == "tanh":
        return nn.Tanh
    elif activation_fn_name == "relu":
        return nn.ReLU
    elif activation_fn_name == "gelu":
        return nn.GELU
    elif activation_fn_name == "leaky_relu":
        return nn.LeakyReLU
    else:
        raise ValueError(f"Unknown activation function: {activation_fn_name}")
    
def get_mlp_class(mlp_class_name):
    if mlp_class_name == "mlp":
        return MLP
    elif mlp_class_name == "residual_mlp":
        return ResidualMLP
    else:
        raise ValueError(f"Unknown MLP class: {mlp_class_name}")