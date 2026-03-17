import torch.optim as optim
import torch.nn as nn
import torch
from kron_torch import Kron
from src.models.models import MLP, ResidualMLP

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