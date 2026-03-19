"""Go-Explore frontier checkpointing for Craftax.

When environments reach milestones (enter_dungeon, etc.), save their JAX state.
When environments get reset (done), optionally restore a frontier state instead of
starting from scratch. This avoids wasting episodes replaying long prerequisite chains.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from collections import defaultdict

# Floor-entry milestones (main progression barriers)
FRONTIER_MILESTONES = [
    'Achievements/enter_dungeon',
    'Achievements/enter_gnomish_mines',
    'Achievements/enter_sewers',
    'Achievements/enter_vault',
    'Achievements/enter_troll_mines',
    'Achievements/enter_fire_realm',
    'Achievements/enter_ice_realm',
    'Achievements/enter_graveyard',
    # Key equipment milestones
    'Achievements/make_diamond_sword',
    'Achievements/make_diamond_armour',
    'Achievements/enchant_sword',
    'Achievements/find_bow',
    'Achievements/collect_diamond',
]

# Priority weights: deeper milestones get sampled more often
MILESTONE_PRIORITY = {
    'Achievements/enter_dungeon': 1.0,
    'Achievements/find_bow': 1.5,
    'Achievements/collect_diamond': 2.0,
    'Achievements/make_diamond_sword': 2.5,
    'Achievements/make_diamond_armour': 2.5,
    'Achievements/enchant_sword': 3.0,
    'Achievements/enter_gnomish_mines': 4.0,
    'Achievements/enter_sewers': 5.0,
    'Achievements/enter_vault': 6.0,
    'Achievements/enter_troll_mines': 7.0,
    'Achievements/enter_fire_realm': 8.0,
    'Achievements/enter_ice_realm': 9.0,
    'Achievements/enter_graveyard': 10.0,
}


class FrontierBuffer:
    """Circular buffer storing environment states at milestone achievements."""

    def __init__(self, max_size=128, milestones=None):
        self.max_size = max_size
        self.milestones = milestones or FRONTIER_MILESTONES
        self.buffer = []  # List of (jax_env_state, raw_obs_tensor, milestone_name)
        self.milestone_counts = defaultdict(int)
        self._ptr = 0

    def add(self, env_state, raw_obs, milestone_name):
        """Add a frontier state. env_state is a single-env JAX pytree, raw_obs is a 1D torch tensor."""
        entry = (env_state, raw_obs.detach().cpu(), milestone_name)
        if len(self.buffer) < self.max_size:
            self.buffer.append(entry)
        else:
            self.buffer[self._ptr % self.max_size] = entry
        self._ptr += 1
        self.milestone_counts[milestone_name] += 1

    def sample(self, n):
        """Sample n states with priority toward deeper milestones."""
        if len(self.buffer) == 0:
            return []
        n = min(n, len(self.buffer))
        weights = np.array([
            MILESTONE_PRIORITY.get(entry[2], 1.0) for entry in self.buffer
        ], dtype=np.float32)
        weights /= weights.sum()
        indices = np.random.choice(len(self.buffer), size=n, replace=False, p=weights)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

    def stats(self):
        return dict(self.milestone_counts)


def extract_single_env_state(batched_env_state, idx):
    """Extract a single environment's state from the batched JAX state."""
    return jax.tree_map(lambda x: x[idx], batched_env_state)


def replace_env_states_batched(batched_env_state, env_indices, replacement_states):
    """Replace multiple env states in the batched state.

    Args:
        batched_env_state: JAX pytree with first dim = num_envs
        env_indices: list of int indices to replace
        replacement_states: list of single-env JAX pytrees

    Returns:
        Updated batched_env_state
    """
    if not env_indices:
        return batched_env_state

    # Stack replacements into a batch
    stacked = jax.tree_map(lambda *xs: jnp.stack(xs), *replacement_states)
    indices = jnp.array(env_indices)

    return jax.tree_map(
        lambda old, new: old.at[indices].set(new),
        batched_env_state,
        stacked,
    )


def detect_and_save_milestones(frontier_buffer, infos, prev_milestone_status,
                                next_done, batched_env_state, raw_obs, num_envs):
    """Detect new milestone achievements and save frontier states.

    Args:
        frontier_buffer: FrontierBuffer instance
        infos: dict from env step, contains 'Achievements/...' keys
        prev_milestone_status: dict mapping milestone_name -> torch.bool tensor (num_envs,)
        next_done: torch.bool tensor (num_envs,) — True for done envs
        batched_env_state: JAX pytree with first dim = num_envs (post-step, post-auto-reset)
        raw_obs: torch tensor (num_envs, obs_dim) — raw observations before normalization
        num_envs: int

    Returns:
        Updated prev_milestone_status, number of new checkpoints saved
    """
    saved_count = 0

    for milestone in frontier_buffer.milestones:
        if milestone not in infos:
            continue

        current = infos[milestone].bool()
        if milestone not in prev_milestone_status:
            prev_milestone_status[milestone] = torch.zeros(num_envs, dtype=torch.bool, device=current.device)

        prev = prev_milestone_status[milestone]
        # Detect transition: False -> True, only for non-done envs
        # (done envs have been auto-reset, their state is lost)
        newly_achieved = current & ~prev & ~next_done

        if newly_achieved.any():
            new_indices = torch.nonzero(newly_achieved, as_tuple=False).squeeze(-1).tolist()
            if isinstance(new_indices, int):
                new_indices = [new_indices]
            for idx in new_indices:
                single_state = extract_single_env_state(batched_env_state, idx)
                frontier_buffer.add(single_state, raw_obs[idx], milestone)
                saved_count += 1

        # Update tracking: set to current for all envs
        prev_milestone_status[milestone] = current.clone()

    # Reset tracking for done envs (they start fresh episodes)
    done_mask = next_done.bool()
    if done_mask.any():
        for milestone in frontier_buffer.milestones:
            if milestone in prev_milestone_status:
                prev_milestone_status[milestone][done_mask] = False

    return prev_milestone_status, saved_count


def apply_frontier_resets(envs, frontier_buffer, done_indices, next_obs, raw_obs,
                          frontier_reset_prob, obs_rms=None, obs_clip=10.0, device='cuda'):
    """Replace some done envs' auto-reset states with frontier states.

    Args:
        envs: the environment wrapper chain (RecordEpisodeStatistics > TorchWrapper > ...)
        frontier_buffer: FrontierBuffer with saved states
        done_indices: list of env indices that just got done (and auto-reset)
        next_obs: torch tensor (num_envs, obs_dim) — current obs (normalized if obs_rms)
        raw_obs: torch tensor (num_envs, obs_dim) — raw obs reference (for non-frontier envs)
        frontier_reset_prob: probability of using frontier reset for each done env
        obs_rms: RunningMeanStd or None
        obs_clip: float clip range for normalization
        device: torch device

    Returns:
        next_obs (modified in-place for frontier envs), number of frontier resets applied
    """
    if len(frontier_buffer) == 0 or len(done_indices) == 0:
        return next_obs, 0

    # Decide which done envs get frontier reset
    frontier_mask = np.random.random(len(done_indices)) < frontier_reset_prob
    frontier_indices = [done_indices[i] for i in range(len(done_indices)) if frontier_mask[i]]

    if not frontier_indices:
        return next_obs, 0

    # Sample frontier states
    sampled = frontier_buffer.sample(len(frontier_indices))
    if not sampled:
        return next_obs, 0

    # Limit to available samples
    n_resets = min(len(frontier_indices), len(sampled))
    frontier_indices = frontier_indices[:n_resets]
    sampled = sampled[:n_resets]

    # Extract replacement states and observations
    replacement_states = [s[0] for s in sampled]  # JAX env states
    replacement_obs_raw = [s[1] for s in sampled]  # Raw torch obs (CPU)

    # Replace env states in the JAX state
    log_state = envs.env._state  # TorchWrapper._state = LogEnvState
    new_env_state = replace_env_states_batched(
        log_state.env_state, frontier_indices, replacement_states
    )

    # Reset episode tracking for frontier envs
    new_episode_returns = log_state.episode_returns
    new_episode_lengths = log_state.episode_lengths
    for idx in frontier_indices:
        new_episode_returns = new_episode_returns.at[idx].set(0.0)
        new_episode_lengths = new_episode_lengths.at[idx].set(0)

    # Update the state
    envs.env._state = log_state.replace(
        env_state=new_env_state,
        episode_returns=new_episode_returns,
        episode_lengths=new_episode_lengths,
    )

    # Update RecordEpisodeStatistics tracking
    for idx in frontier_indices:
        envs.episode_returns[idx] = 0.0
        envs.episode_lengths[idx] = 0

    # Replace observations
    for i, idx in enumerate(frontier_indices):
        obs_raw = replacement_obs_raw[i].to(device)
        if obs_rms is not None:
            obs_normalized = obs_rms.normalize(obs_raw.unsqueeze(0), obs_clip).squeeze(0)
            next_obs[idx] = obs_normalized
        else:
            next_obs[idx] = obs_raw

    return next_obs, n_resets
