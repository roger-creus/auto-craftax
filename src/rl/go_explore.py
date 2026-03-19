"""Go-Explore frontier checkpointing for Craftax.

When environments reach milestones (enter_dungeon, etc.), save their JAX state.
When environments get reset (done), optionally restore a frontier state instead of
starting from scratch. This avoids wasting episodes replaying long prerequisite chains.

BUG FIX (2026-03-19): Craftax's log_achievements_to_info multiplies achievements by
done, so infos['Achievements/...'] is always 0 for non-done envs. We now read
achievements directly from the JAX env state (env_state.achievements) which has the
true per-step values regardless of done status.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
from collections import defaultdict

# Craftax Achievement enum indices (from craftax.craftax.constants.Achievement)
# Maps milestone name -> index in env_state.achievements array
MILESTONE_ACH_INDICES = {
    'enter_dungeon': 29,
    'enter_gnomish_mines': 28,
    'enter_sewers': 30,
    'enter_vault': 31,
    'enter_troll_mines': 32,
    'enter_fire_realm': 33,
    'enter_ice_realm': 34,
    'enter_graveyard': 35,
    'make_diamond_sword': 25,
    'make_diamond_armour': 27,
    'enchant_sword': 63,
    'find_bow': 52,
    'collect_diamond': 19,
}

# Priority weights: deeper milestones get sampled more often
MILESTONE_PRIORITY = {
    'enter_dungeon': 1.0,
    'find_bow': 1.5,
    'collect_diamond': 2.0,
    'make_diamond_sword': 2.5,
    'make_diamond_armour': 2.5,
    'enchant_sword': 3.0,
    'enter_gnomish_mines': 4.0,
    'enter_sewers': 5.0,
    'enter_vault': 6.0,
    'enter_troll_mines': 7.0,
    'enter_fire_realm': 8.0,
    'enter_ice_realm': 9.0,
    'enter_graveyard': 10.0,
}


class FrontierBuffer:
    """Circular buffer storing environment states at milestone achievements."""

    def __init__(self, max_size=128, milestones=None):
        self.max_size = max_size
        self.milestones = milestones or list(MILESTONE_ACH_INDICES.keys())
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
    """Replace multiple env states in the batched state."""
    if not env_indices:
        return batched_env_state
    stacked = jax.tree_map(lambda *xs: jnp.stack(xs), *replacement_states)
    indices = jnp.array(env_indices)
    return jax.tree_map(
        lambda old, new: old.at[indices].set(new),
        batched_env_state,
        stacked,
    )


def get_achievements_from_state(env_state, device):
    """Extract achievement tensor from JAX env state.

    Returns:
        torch.bool tensor of shape (num_envs, num_achievements)
    """
    ach_jax = env_state.achievements  # (num_envs, N) boolean JAX array
    return torch.as_tensor(np.asarray(ach_jax), device=device).bool()


def detect_and_save_milestones(frontier_buffer, achievements, prev_milestone_status,
                                next_done, batched_env_state, raw_obs, num_envs):
    """Detect new milestone achievements and save frontier states.

    Args:
        frontier_buffer: FrontierBuffer instance
        achievements: torch.bool tensor (num_envs, num_achievements) from JAX state
        prev_milestone_status: dict mapping milestone_name -> torch.bool tensor (num_envs,)
        next_done: torch.bool tensor (num_envs,) — True for done envs
        batched_env_state: JAX pytree with first dim = num_envs (post-step, post-auto-reset)
        raw_obs: torch tensor (num_envs, obs_dim) — raw observations
        num_envs: int

    Returns:
        Updated prev_milestone_status, number of new checkpoints saved
    """
    saved_count = 0
    num_ach = achievements.shape[1]

    for milestone_name, ach_idx in MILESTONE_ACH_INDICES.items():
        if ach_idx >= num_ach:
            continue

        current = achievements[:, ach_idx]
        if milestone_name not in prev_milestone_status:
            prev_milestone_status[milestone_name] = torch.zeros(num_envs, dtype=torch.bool, device=current.device)

        prev = prev_milestone_status[milestone_name]
        # Detect transition: False -> True, only for non-done envs
        # (done envs have been auto-reset, their state is garbage)
        newly_achieved = current & ~prev & ~next_done

        if newly_achieved.any():
            new_indices = torch.nonzero(newly_achieved, as_tuple=False).squeeze(-1).tolist()
            if isinstance(new_indices, int):
                new_indices = [new_indices]
            for idx in new_indices:
                single_state = extract_single_env_state(batched_env_state, idx)
                frontier_buffer.add(single_state, raw_obs[idx], milestone_name)
                saved_count += 1

        # Update tracking: set to current for all envs
        prev_milestone_status[milestone_name] = current.clone()

    # Reset tracking for done envs (they start fresh episodes)
    done_mask = next_done.bool()
    if done_mask.any():
        for milestone_name in MILESTONE_ACH_INDICES:
            if milestone_name in prev_milestone_status:
                prev_milestone_status[milestone_name][done_mask] = False

    return prev_milestone_status, saved_count


def apply_frontier_resets(envs, frontier_buffer, done_indices, next_obs, raw_obs,
                          frontier_reset_prob, obs_rms=None, obs_clip=10.0, device='cuda'):
    """Replace some done envs' auto-reset states with frontier states."""
    if len(frontier_buffer) == 0 or len(done_indices) == 0:
        return next_obs, 0

    frontier_mask = np.random.random(len(done_indices)) < frontier_reset_prob
    frontier_indices = [done_indices[i] for i in range(len(done_indices)) if frontier_mask[i]]

    if not frontier_indices:
        return next_obs, 0

    sampled = frontier_buffer.sample(len(frontier_indices))
    if not sampled:
        return next_obs, 0

    n_resets = min(len(frontier_indices), len(sampled))
    frontier_indices = frontier_indices[:n_resets]
    sampled = sampled[:n_resets]

    replacement_states = [s[0] for s in sampled]
    replacement_obs_raw = [s[1] for s in sampled]

    log_state = envs.env._state
    new_env_state = replace_env_states_batched(
        log_state.env_state, frontier_indices, replacement_states
    )

    new_episode_returns = log_state.episode_returns
    new_episode_lengths = log_state.episode_lengths
    for idx in frontier_indices:
        new_episode_returns = new_episode_returns.at[idx].set(0.0)
        new_episode_lengths = new_episode_lengths.at[idx].set(0)

    envs.env._state = log_state.replace(
        env_state=new_env_state,
        episode_returns=new_episode_returns,
        episode_lengths=new_episode_lengths,
    )

    for idx in frontier_indices:
        envs.episode_returns[idx] = 0.0
        envs.episode_lengths[idx] = 0

    for i, idx in enumerate(frontier_indices):
        obs_raw = replacement_obs_raw[i].to(device)
        if obs_rms is not None:
            obs_normalized = obs_rms.normalize(obs_raw.unsqueeze(0), obs_clip).squeeze(0)
            next_obs[idx] = obs_normalized
        else:
            next_obs[idx] = obs_raw

    return next_obs, n_resets
