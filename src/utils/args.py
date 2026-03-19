import os
from dataclasses import dataclass

@dataclass
class PPO_Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    hypothesis_id: str = ""
    """hypothesis ID for results tracking"""
    experiment_id: str = ""
    """experiment ID for results tracking"""
    output_dir: str = ""
    """output directory for results CSV (usually bound via Singularity)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "skills-via-llm"
    """the wandb's project name"""
    wandb_entity: str = "glen-berseth"
    """the entity (team) of wandb's project"""
    num_logs: int = 1000
    """how many logs to save during training"""

    # Algorithm specific arguments
    env_id: str = "Craftax-Symbolic-v1"
    """the id of the environment"""
    total_timesteps: int = 1_000_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0002
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.8
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""
    
    # network architecture
    num_layers: int = 5
    """the number of layers in the MLP"""
    hidden_size: int = 512
    """the size of the hidden layers in the MLP"""
    activation_fn: str = "tanh"
    """the activation function to use in the MLP"""
    use_ln: bool = False
    """if toggled, use layer normalization in the MLP"""
    use_structured_obs: bool = False
    """if toggled, use CNN+MLP structured observation encoder instead of flat MLP"""
    use_popart: bool = False
    """if toggled, use PopArt adaptive value normalization"""
    use_gru: bool = False
    """if toggled, use GRU instead of LSTM in recurrent agent"""
    obs_norm: bool = False
    """if toggled, normalize observations using running mean/std"""
    obs_clip: float = 10.0
    """clip range for normalized observations"""
    ent_coef_end: float = -1.0
    """final entropy coefficient (-1 = no annealing, use ent_coef throughout)"""
    lr_schedule: str = "linear"
    """learning rate schedule: 'linear' (default) or 'cosine'"""
    optimizer: str = "adam"
    """the optimizer to use"""
    go_explore: bool = False
    """if toggled, use Go-Explore frontier checkpointing"""
    frontier_buffer_size: int = 128
    """max number of frontier states to store"""
    frontier_reset_prob: float = 0.25
    """probability of resetting a done env to a frontier state instead of fresh start"""
    mlp_class: str = "mlp"
    """the MLP class to use"""
    
    # BC
    bc_dataset_path: str = ""
    bc_batch_size: int = 32
    
    # hierarchical
    skill_length: int = 8

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


@dataclass
class GTrXL_Args(PPO_Args):
    """PPO args extended with GTrXL-specific architecture parameters."""
    trxl_layers: int = 3
    """number of transformer layers"""
    trxl_heads: int = 8
    """number of attention heads"""
    trxl_memory: int = -1
    """transformer memory length (-1 = use num_steps)"""
    trxl_mlp_layers: int = 2
    """number of MLP layers in input projection"""
    ent_coef_end: float = -1.0
    """final entropy coefficient (-1 = no annealing, use ent_coef throughout)"""
    use_symlog: bool = False
    """if toggled, use symlog two-hot distributional value head (DreamerV3-style)"""
    gamma_start: float = -1.0
    """initial gamma for discount annealing (-1 = no annealing, use gamma throughout)"""
    gamma_anneal_frac: float = 0.2
    """fraction of training over which to anneal gamma from gamma_start to gamma"""
    lr_warmup_frac: float = 0.0
    """fraction of training for LR warmup (0 = no warmup)"""


@dataclass
class PQN_Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    hypothesis_id: str = ""
    """hypothesis ID for results tracking"""
    experiment_id: str = ""
    """experiment ID for results tracking"""
    output_dir: str = ""
    """output directory for results CSV (usually bound via Singularity)"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "skills-via-llm"
    """the wandb's project name"""
    wandb_entity: str = "glen-berseth"
    """the entity (team) of wandb's project"""
    num_logs: int = 1000
    """how many logs to save during training"""

    # Algorithm specific arguments
    env_id: str = "Craftax-Symbolic-v1"
    """the id of the environment"""
    total_timesteps: int = 1_000_000_000
    """total timesteps of the experiments"""
    learning_rate: float = 0.0002
    """the learning rate of the optimizer"""
    num_envs: int = 1024
    """the number of parallel game environments"""
    num_steps: int = 64
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    num_minibatches: int = 8
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    max_grad_norm: float = 10.0
    """the maximum norm for the gradient clipping"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.005
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    q_lambda: float = 0.65
    """the lambda for the Q-Learning algorithm"""
    
    # network architecture
    num_layers: int = 5
    """the number of layers in the MLP"""
    hidden_size: int = 512
    """the size of the hidden layers in the MLP"""
    activation_fn: str = "leaky_relu"
    """the activation function to use in the MLP"""
    use_ln: bool = True
    """if toggled, use layer normalization in the MLP"""
    optimizer: str = "radam"
    """the optimizer to use"""
    
    # soft pqn parameters
    target_entropy_scale: float = 0.75
    """the scale of the target entropy for the alpha parameter"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""