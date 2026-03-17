from craftax.craftax_env import make_craftax_env_from_name
from src.env.wrappers import LogWrapper, OptimisticResetVecEnvWrapper, TorchWrapper, RecordEpisodeStatistics

def make_craftax_env(
    env_id: str = "Craftax-Symbolic-v1",
    num_envs: int = 1024,
    reset_ratio: int = 16,
    device: str = "cuda",
):
    env = make_craftax_env_from_name(env_id, auto_reset=False)
    env = LogWrapper(env)
    env = OptimisticResetVecEnvWrapper(env, num_envs=num_envs, reset_ratio=reset_ratio)
    env = TorchWrapper(env, device=device)
    env = RecordEpisodeStatistics(env)
    env.num_envs = num_envs
    env.single_observation_space = env.observation_space
    env.single_action_space = env.action_space
    return env