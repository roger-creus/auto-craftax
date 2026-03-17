import jax
import numpy as np
import gymnasium as gym
import torch
import jax.numpy as jnp
import chex

from flax import struct
from jax import tree_util
from typing import Union, Any
from functools import partial
from brax.io import torch as brax_torch

class TorchWrapper(gym.Env):
    def __init__(self, env, device="cuda"):
        self.env = env
        self.device = torch.device(device)
        self.default_params = getattr(env, "default_params", {})
        self.static_env_params = getattr(env, "static_env_params", None)
        self.metadata = { 'render.modes': ['human', 'rgb_array'] }

        obs_shape = env.observation_space(self.default_params).shape 
        self.observation_space = gym.spaces.Box(
            low=-1e6,
            high=1e6,
            shape=obs_shape,
            dtype=np.float32
        ) 
        self.action_space = gym.spaces.Discrete(env.action_space(self.default_params).n)

        self._state = None
        self._key = jax.random.PRNGKey(0)

        def _reset_fn(key):
            next_key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey)
            info = {}
            return obs, state, info, next_key

        def _step_fn(key, state, action):
            next_key, subkey = jax.random.split(key)
            obs, new_state, reward, done, info = env.step(rng=subkey, state=state, action=action)
            return obs, new_state, reward, done, info, next_key

        self._reset = jax.jit(_reset_fn)
        self._step = jax.jit(_step_fn)

    def seed(self, seed: int = 0):
        self._key = jax.random.PRNGKey(int(seed))

    @staticmethod
    def _block_pytree(pytree):
        def _maybe_block(x):
            try:
                return x.block_until_ready()
            except Exception:
                return x
        return tree_util.tree_map(_maybe_block, pytree)

    def _pytree_to_torch(self, pytree):
        return tree_util.tree_map(lambda x: brax_torch.jax_to_torch(x, device=self.device), pytree)

    def reset(self, seed: int | None = None, options=None):
        if seed is not None:
            self.seed(seed)

        obs_jax, state_jax, info_jax, next_key = self._reset(self._key)
        self._key = next_key
        self._state = state_jax

        obs_t = self._pytree_to_torch(obs_jax)
        info_t = self._pytree_to_torch(info_jax)

        return obs_t, info_t

    def step(self, action):
        action_jax = brax_torch.torch_to_jax(action)

        obs_jax, new_state_jax, reward_jax, done_jax, info_jax, next_key = self._step(self._key, self._state, action_jax)
        self._key = next_key
        self._state = new_state_jax

        obs_t = self._pytree_to_torch(obs_jax)
        reward_t = self._pytree_to_torch(reward_jax)
        done_t = self._pytree_to_torch(done_jax)
        info_t = self._pytree_to_torch(info_jax)

        return obs_t, reward_t, done_t.to(dtype=torch.bool), done_t.to(dtype=torch.bool), info_t
    
    def get_state(self):
        return self._state.env_state

class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None
        self.state = None

    def reset(self, seed: int | None = None, options=None):
        observations, info = super().reset(seed=seed, options=options)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations, info

    def step(self, action):
        observations, rewards, done, truncated, infos = super().step(action)
        self.episode_returns += rewards.cpu().numpy()
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - done.cpu().numpy()
        self.episode_lengths *= 1 - done.cpu().numpy().astype(np.int32)
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            done,
            truncated,
            infos,
        )
        
    def get_state(self):
        return self.env.get_state()

class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)
        
class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_util.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_util.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: Any
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key: chex.PRNGKey, params=None):
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(
        self,
        key: chex.PRNGKey,
        state,
        action: Union[int, float],
        params=None,
    ):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
            + new_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
            + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        return obs, state, reward, done, info