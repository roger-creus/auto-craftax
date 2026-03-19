"""Potential-Based Reward Shaping (PBRS) for Craftax.

Adds F(s, s') = gamma * phi(s') - phi(s) to the reward, where phi(s) is a
potential function based on achievement milestones. This provably preserves
the optimal policy (Ng et al., 1999) while providing denser reward signals
for milestone progression.

On episode termination, phi(s') = 0 since the agent enters a fresh state.
"""

import torch

# Achievement potential values — higher for deeper/harder milestones
ACHIEVEMENT_POTENTIALS = {
    # Equipment progression
    'Achievements/collect_iron': 1.0,
    'Achievements/collect_diamond': 3.0,
    'Achievements/find_bow': 2.0,
    'Achievements/make_iron_sword': 1.5,
    'Achievements/make_diamond_sword': 4.0,
    'Achievements/make_diamond_armour': 4.0,
    'Achievements/enchant_sword': 5.0,
    'Achievements/enchant_armour': 5.0,
    # Floor progression (main bottleneck)
    'Achievements/enter_dungeon': 8.0,
    'Achievements/enter_gnomish_mines': 15.0,
    'Achievements/enter_sewers': 22.0,
    'Achievements/enter_vault': 30.0,
    'Achievements/enter_troll_mines': 38.0,
    'Achievements/enter_fire_realm': 46.0,
    'Achievements/enter_ice_realm': 54.0,
    'Achievements/enter_graveyard': 62.0,
}


def compute_potential(infos, device='cuda'):
    """Compute phi(s) for all envs from current achievement status.

    Args:
        infos: dict from env step, contains 'Achievements/...' bool tensors (num_envs,)
        device: torch device

    Returns:
        phi: torch tensor (num_envs,) — potential value per env
    """
    phi = None
    for ach_name, value in ACHIEVEMENT_POTENTIALS.items():
        if ach_name in infos:
            ach_status = infos[ach_name].float()  # 0.0 or 1.0
            contribution = ach_status * value
            if phi is None:
                phi = contribution
            else:
                phi = phi + contribution
    if phi is None:
        return torch.zeros(1, device=device)
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
