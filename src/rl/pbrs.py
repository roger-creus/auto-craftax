"""Potential-Based Reward Shaping (PBRS) for Craftax.

Adds F(s, s') = gamma * phi(s') - phi(s) to the reward, where phi(s) is a
potential function based on achievement milestones. This provably preserves
the optimal policy (Ng et al., 1999) while providing denser reward signals
for milestone progression.

On episode termination, phi(s') = 0 since the agent enters a fresh state.

BUG FIX (2026-03-19): Craftax's log_achievements_to_info multiplies achievements by
done, so infos['Achievements/...'] is always 0 for non-done envs. We now read
achievements directly from the JAX env state via get_achievements_from_state().
"""

import torch

# Craftax Achievement enum indices -> potential values
# Higher values for deeper/harder milestones
ACHIEVEMENT_IDX_POTENTIALS = {
    18: 1.0,   # collect_iron
    19: 3.0,   # collect_diamond
    52: 2.0,   # find_bow
    21: 1.5,   # make_iron_sword
    25: 4.0,   # make_diamond_sword
    27: 4.0,   # make_diamond_armour
    63: 5.0,   # enchant_sword
    64: 5.0,   # enchant_armour
    29: 8.0,   # enter_dungeon
    28: 15.0,  # enter_gnomish_mines
    30: 22.0,  # enter_sewers
    31: 30.0,  # enter_vault
    32: 38.0,  # enter_troll_mines
    33: 46.0,  # enter_fire_realm
    34: 54.0,  # enter_ice_realm
    35: 62.0,  # enter_graveyard
}


def compute_potential_from_state(achievements, device='cuda'):
    """Compute phi(s) from raw achievement tensor (from JAX env state).

    Args:
        achievements: torch tensor (num_envs, num_achievements) from JAX state
        device: torch device

    Returns:
        phi: torch tensor (num_envs,) — potential value per env
    """
    num_ach = achievements.shape[1]
    phi = torch.zeros(achievements.shape[0], device=device)
    for ach_idx, value in ACHIEVEMENT_IDX_POTENTIALS.items():
        if ach_idx < num_ach:
            phi += achievements[:, ach_idx].float() * value
    return phi


def compute_pbrs_bonus(prev_phi, next_phi, next_done, gamma):
    """Compute PBRS bonus: gamma * phi(s') - phi(s).

    On termination, phi(s') = 0 (fresh reset state has no achievements).

    Args:
        prev_phi: torch tensor (num_envs,) — potential before step
        next_phi: torch tensor (num_envs,) — potential after step
        next_done: torch tensor (num_envs,) — True for terminated envs
        gamma: discount factor

    Returns:
        bonus: torch tensor (num_envs,) — shaping reward to add
    """
    # On done, the "next state" is a reset state with phi=0
    effective_next_phi = next_phi * (~next_done).float()
    bonus = gamma * effective_next_phi - prev_phi
    return bonus
