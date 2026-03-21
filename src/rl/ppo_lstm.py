
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
import jax.numpy as jnp

from collections import deque

from src.env.env import make_craftax_env
from src.utils.logger import write_row_csv, make_training_csv_craftax_classic, make_training_csv_craftax, make_ppo_losses_csv, save_results_csv
from src.models.models import PPO_LSTM_Agent
from src.utils.args import PPO_Args
from src.utils.utils import get_optimizer_class, get_activation_fn, get_mlp_class, RunningMeanStd
from src.rl.go_explore import FrontierBuffer, detect_and_save_milestones, apply_frontier_resets, get_achievements_from_state
from src.rl.pbrs import compute_potential_from_state, compute_pbrs_bonus
from src.rl.sil import SILBuffer, compute_sil_loss

if __name__ == "__main__":
    import os
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = tyro.cli(PPO_Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.log_iterations = np.linspace(0, args.num_iterations, args.num_logs).astype(int)

    algo_name = "PPO-LSTM"
    run_name = f"algo:{algo_name}_env:{args.env_id}_seed:{args.seed}_{int(time.time())}"

    print(f"Running with num_envs={args.num_envs}, num_steps={args.num_steps}, minibatch_size={args.minibatch_size}, max_grad_norm={args.max_grad_norm}")

    # Logging
    os.makedirs(f"./runs/{run_name}", exist_ok=True)
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    if "Classic" in args.env_id:
        training_csv_writer = make_training_csv_craftax_classic(f"./runs/{run_name}")
    else:
        training_csv_writer = make_training_csv_craftax(f"./runs/{run_name}")

    losses_csv_writer = make_ppo_losses_csv(f"./runs/{run_name}")

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"using device {device}")

    # environment setup
    envs = make_craftax_env(
        env_id=args.env_id,
        num_envs=args.num_envs,
        device=device,
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    # observation normalization
    obs_rms = None
    if args.obs_norm:
        obs_rms = RunningMeanStd(envs.single_observation_space.shape, device)
        print(f"Observation normalization enabled (clip={args.obs_clip})")

    # Go-Explore frontier checkpointing
    frontier_buffer = None
    prev_milestone_status = {}
    frontier_total_saves = 0
    frontier_total_resets = 0
    if args.go_explore:
        frontier_buffer = FrontierBuffer(max_size=args.frontier_buffer_size)
        print(f"Go-Explore enabled: buffer_size={args.frontier_buffer_size}, reset_prob={args.frontier_reset_prob}")

    # PBRS state
    prev_phi = None
    if args.pbrs:
        prev_phi = torch.zeros(args.num_envs, device=device)
        print(f"PBRS enabled: potential-based reward shaping for milestone achievements")

    # Kill bonus state: track monsters_killed and player_level per env
    prev_floor_kills = None
    prev_player_level = None
    kill_bonus_total = 0.0
    floor_bonus_total = 0.0
    if args.kill_bonus:
        prev_floor_kills = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
        prev_player_level = torch.zeros(args.num_envs, device=device, dtype=torch.float32)
        print(f"Kill bonus enabled: kill_scale={args.kill_bonus_scale}, floor_scale={args.floor_bonus_scale}")

    # Observation augmentation: append kill count on current floor (normalized 0-1)
    extra_stats_dim = 0
    if args.obs_augment:
        extra_stats_dim += 1  # kill count on current floor / 8.0
        print(f"Observation augmentation enabled: +1 features (kill count)")
    if args.rle:
        extra_stats_dim += args.rle_dim  # z vector appended to obs for policy conditioning
        print(f"RLE conditioning: +{args.rle_dim} features (latent z vector)")

    def augment_obs(obs_tensor):
        """Append kill count and/or RLE z vector to observation."""
        parts = [obs_tensor]
        if args.obs_augment:
            env_state = envs.env._state.env_state
            pl_np = np.asarray(env_state.player_level).astype(int)
            mk = np.asarray(env_state.monsters_killed)  # (num_envs, num_floors)
            kills = mk[np.arange(args.num_envs), pl_np].clip(max=8).astype(np.float32) / 8.0
            kills_t = torch.as_tensor(kills, device=device).unsqueeze(-1)  # (num_envs, 1)
            parts.append(kills_t)
        if rle_z is not None:
            parts.append(rle_z)  # (num_envs, rle_dim)
        if len(parts) == 1:
            return obs_tensor
        return torch.cat(parts, dim=-1)

    agent = PPO_LSTM_Agent(
        obs_dim=np.array(envs.single_observation_space.shape).prod() + extra_stats_dim,
        n_actions=envs.single_action_space.n,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        activation_fn=get_activation_fn(args.activation_fn),
        use_ln=args.use_ln,
        mlp_class=get_mlp_class(args.mlp_class),
        use_structured_obs=args.use_structured_obs,
        use_popart=args.use_popart,
        use_gru=args.use_gru,
        extra_stats_dim=extra_stats_dim,
        separate_critic=args.separate_critic,
    ).to(device)

    # Auxiliary kill-count prediction head
    if args.aux_kill_pred:
        agent.init_aux_head(args.hidden_size, n_targets=1)
        agent.aux_head = agent.aux_head.to(device)
        print(f"Auxiliary kill prediction enabled: coef={args.aux_kill_coef}")

    # Reward normalization (running mean/std)
    reward_rms = None
    if args.reward_norm:
        reward_rms = RunningMeanStd((1,), device)
        print(f"Reward normalization enabled")

    # RND (Random Network Distillation) exploration
    rnd_target = None
    rnd_predictor = None
    rnd_optimizer = None
    rnd_reward_rms = None
    if args.rnd:
        obs_dim_flat = np.array(envs.single_observation_space.shape).prod() + extra_stats_dim
        # Target network: fixed random MLP
        rnd_target = nn.Sequential(
            nn.Linear(obs_dim_flat, args.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnd_hidden_dim, args.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnd_hidden_dim, args.rnd_output_dim),
        ).to(device)
        # Freeze target
        for p in rnd_target.parameters():
            p.requires_grad = False
        # Predictor network: trainable MLP (slightly larger)
        rnd_predictor = nn.Sequential(
            nn.Linear(obs_dim_flat, args.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnd_hidden_dim, args.rnd_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rnd_hidden_dim, args.rnd_output_dim),
        ).to(device)
        rnd_optimizer = torch.optim.Adam(rnd_predictor.parameters(), lr=args.rnd_lr)
        # Running stats for normalizing intrinsic rewards
        rnd_reward_rms = RunningMeanStd((1,), device)
        rnd_anneal_str = f" -> {args.rnd_coef_end}" if args.rnd_coef_end >= 0 else ""
        noveld_str = " (NovelD mode)" if args.rnd_noveld else ""
        print(f"RND exploration enabled: coef={args.rnd_coef}{rnd_anneal_str}, output_dim={args.rnd_output_dim}, hidden_dim={args.rnd_hidden_dim}{noveld_str}")

    # RLE (Random Latent Exploration) — simpler than RND, conditions policy on random z
    rle_phi = None
    rle_z = None
    rle_reward_rms = None
    if args.rle:
        obs_dim_flat = np.array(envs.single_observation_space.shape).prod()
        if args.obs_augment:
            obs_dim_flat += 1  # obs augment adds 1 feature before RLE z
        # Fixed random feature extractor: obs → rle_dim embedding
        rle_phi = nn.Sequential(
            nn.Linear(obs_dim_flat, args.rle_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rle_hidden_dim, args.rle_hidden_dim),
            nn.ReLU(),
            nn.Linear(args.rle_hidden_dim, args.rle_dim),
        ).to(device)
        for p in rle_phi.parameters():
            p.requires_grad = False
        # Initialize z vectors (unit sphere) for each env
        rle_z = torch.randn(args.num_envs, args.rle_dim, device=device)
        rle_z = rle_z / (rle_z.norm(dim=-1, keepdim=True) + 1e-8)
        # Running stats for normalizing intrinsic rewards
        rle_reward_rms = RunningMeanStd((1,), device)
        print(f"RLE exploration enabled: coef={args.rle_coef}, dim={args.rle_dim}, hidden_dim={args.rle_hidden_dim}")

    # Self-Imitation Learning (SIL) buffer
    sil_buffer = None
    sil_return_history = deque(maxlen=10000)  # track recent returns for percentile threshold
    if args.sil:
        obs_dim_sil = np.array(envs.single_observation_space.shape).prod() + extra_stats_dim
        sil_buffer = SILBuffer(args.sil_buffer_size, obs_dim_sil, device)
        print(f"SIL enabled: coef={args.sil_coef}, vf_coef={args.sil_vf_coef}, buffer={args.sil_buffer_size}, percentile={args.sil_percentile}, warmup={args.sil_warmup_frac}")

    if args.curriculum_kills:
        curriculum_floors = [int(f) for f in args.curriculum_target_floors.split(",")]
        print(f"Curriculum learning enabled: frac={args.curriculum_frac}, kills=[{args.curriculum_min_kills},{args.curriculum_max_kills}], end_frac={args.curriculum_end_frac}, target_floors={curriculum_floors}")

    print("-------------")
    print(agent)

    opt_kwargs = {"lr": args.learning_rate}
    if "adam" in args.optimizer:
        opt_kwargs["eps"] = 1e-5
    optimizer = get_optimizer_class(args.optimizer)(agent.parameters(), **opt_kwargs)
    print(optimizer)

    # storage setup
    obs_shape = (envs.single_observation_space.shape[0] + extra_stats_dim,)
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    aux_targets = torch.zeros((args.num_steps, args.num_envs)).to(device) if args.aux_kill_pred else None

    next_lstm_state = (
        torch.zeros(agent.rnn.num_layers, args.num_envs, agent.rnn.hidden_size).to(device),
        torch.zeros(agent.rnn.num_layers, args.num_envs, agent.rnn.hidden_size).to(device),
    )

    # start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    if obs_rms is not None:
        obs_rms.update(next_obs)
        next_obs = obs_rms.normalize(next_obs, args.obs_clip)
    next_obs = augment_obs(next_obs)
    next_done = torch.zeros(args.num_envs).to(device)
    prev_rnd_error = torch.zeros(args.num_envs, device=device) if (args.rnd and args.rnd_noveld) else None
    start_time = time.time()

    # stats tracking
    avg_ep_reward = deque(maxlen=25)
    avg_ep_length = deque(maxlen=25)
    avg_achievements = {}

    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())

        # annealing the rate if instructed to do so.
        if args.anneal_lr:
            progress = (iteration - 1.0) / args.num_iterations
            if args.lr_schedule == "cosine":
                import math
                frac = 0.5 * (1.0 + math.cos(math.pi * progress))
            else:
                frac = 1.0 - progress
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # entropy coefficient annealing
        if args.ent_coef_end >= 0:
            ent_frac = (iteration - 1.0) / args.num_iterations
            ent_coef_now = args.ent_coef + (args.ent_coef_end - args.ent_coef) * ent_frac
        else:
            ent_coef_now = args.ent_coef

        # RND coefficient annealing
        if args.rnd and args.rnd_coef_end >= 0:
            rnd_frac = (iteration - 1.0) / args.num_iterations
            rnd_coef_now = args.rnd_coef + (args.rnd_coef_end - args.rnd_coef) * rnd_frac
        else:
            rnd_coef_now = args.rnd_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action logic
            with torch.no_grad():
                action, logprob, _, value, next_lstm_state, _ = agent.get_action_and_value(next_obs, next_lstm_state, next_done, denormalize=True)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            # Keep raw obs reference for Go-Explore before normalization
            raw_next_obs = next_obs
            next_done = torch.logical_or(terminations, truncations)

            # Curriculum: pre-fill monsters_killed for a fraction of reset envs
            if args.curriculum_kills:
                done_mask = next_done.cpu().numpy().astype(bool)
                if done_mask.any():
                    progress = global_step / args.total_timesteps
                    if progress < args.curriculum_end_frac:
                        curr_frac = args.curriculum_frac * (1.0 - progress / args.curriculum_end_frac)
                    else:
                        curr_frac = 0.0
                    if curr_frac > 0:
                        done_idx = np.where(done_mask)[0]
                        n_curr = max(1, int(len(done_idx) * curr_frac))
                        curr_idx = np.random.choice(done_idx, size=min(n_curr, len(done_idx)), replace=False)
                        env_state = envs.env._state.env_state
                        mk = np.asarray(env_state.monsters_killed).copy()
                        for idx in curr_idx:
                            for floor_idx in curriculum_floors:
                                mk[idx, floor_idx] = np.random.randint(args.curriculum_min_kills, args.curriculum_max_kills + 1)
                        new_env_state = env_state.replace(monsters_killed=jnp.array(mk))
                        envs.env._state = envs.env._state.replace(env_state=new_env_state)

            if obs_rms is not None:
                obs_rms.update(next_obs)
                next_obs = obs_rms.normalize(next_obs, args.obs_clip)
            next_obs = augment_obs(next_obs)
            # Get achievements from JAX state (not from infos which are zeroed for non-done envs)
            achievements = None
            if prev_phi is not None or frontier_buffer is not None:
                achievements = get_achievements_from_state(envs.env._state.env_state, device)

            # PBRS: add potential-based shaping bonus
            if prev_phi is not None:
                next_phi = compute_potential_from_state(achievements, device=device)
                pbrs_bonus = compute_pbrs_bonus(prev_phi, next_phi, next_done, args.gamma)
                reward = reward.view(-1) + pbrs_bonus
                # Update prev_phi: reset to 0 for done envs (fresh episode)
                prev_phi = next_phi * (~next_done).float()

            # Kill bonus: direct reward for killing monsters and descending floors
            if prev_floor_kills is not None:
                env_state = envs.env._state.env_state
                player_level = torch.as_tensor(np.asarray(env_state.player_level), device=device).float()
                monsters_killed = np.asarray(env_state.monsters_killed)  # (num_envs, num_floors)
                # Get kills on current floor for each env
                pl_np = np.asarray(env_state.player_level).astype(int)
                current_kills = torch.as_tensor(
                    monsters_killed[np.arange(args.num_envs), pl_np],
                    device=device
                ).float().clamp(max=8.0)

                # Kill delta: reward for each new kill (capped at 8)
                kill_delta = (current_kills - prev_floor_kills).clamp(min=0.0)
                # Floor delta: reward for descending to new floor
                floor_delta = (player_level - prev_player_level).clamp(min=0.0)

                intrinsic = kill_delta * args.kill_bonus_scale + floor_delta * args.floor_bonus_scale
                reward = reward.view(-1) + intrinsic

                kill_bonus_total += kill_delta.sum().item()
                floor_bonus_total += floor_delta.sum().item()

                # Update tracking: reset for done envs
                prev_floor_kills = current_kills * (~next_done).float()
                prev_player_level = player_level * (~next_done).float()

            # RND intrinsic reward
            if rnd_target is not None:
                with torch.no_grad():
                    obs_flat = next_obs.view(args.num_envs, -1)
                    rnd_target_feat = rnd_target(obs_flat)
                    rnd_pred_feat = rnd_predictor(obs_flat)
                    rnd_error = (rnd_target_feat - rnd_pred_feat).pow(2).mean(dim=-1)  # (num_envs,)
                    # NovelD: bonus = max(error(s') - error(s), 0) — reward transitions to MORE novel states
                    if prev_rnd_error is not None:
                        noveld_raw = (rnd_error - prev_rnd_error).clamp(min=0.0)
                        # Update prev_rnd_error: reset to 0 for done envs
                        prev_rnd_error = rnd_error.clone()
                        prev_rnd_error[next_done.bool()] = 0.0
                        rnd_error_for_reward = noveld_raw
                    else:
                        rnd_error_for_reward = rnd_error
                    # Normalize by running stats
                    rnd_reward_rms.update(rnd_error_for_reward.unsqueeze(-1))
                    rnd_intrinsic = rnd_error_for_reward / (rnd_reward_rms.var.squeeze().sqrt() + 1e-8)
                    # Optionally mask: only apply RND in dungeon (floor > 0)
                    if args.rnd_dungeon_only:
                        env_state_rnd = envs.env._state.env_state
                        in_dungeon = torch.as_tensor(np.asarray(env_state_rnd.player_level), device=device).float() > 0
                        rnd_intrinsic = rnd_intrinsic * in_dungeon.float()
                    reward = reward.view(-1) + rnd_coef_now * rnd_intrinsic

            # RLE intrinsic reward: r_i = φ(s) · z, normalized by running std
            if rle_phi is not None:
                with torch.no_grad():
                    # Strip RLE z from obs to get raw obs for feature extraction
                    obs_for_phi = next_obs[..., :-args.rle_dim]
                    phi_s = rle_phi(obs_for_phi.view(args.num_envs, -1))  # (num_envs, rle_dim)
                    rle_reward = (phi_s * rle_z).sum(dim=-1)  # (num_envs,)
                    # Normalize by running stats
                    rle_reward_rms.update(rle_reward.unsqueeze(-1))
                    rle_intrinsic = rle_reward / (rle_reward_rms.var.squeeze().sqrt() + 1e-8)
                    reward = reward.view(-1) + args.rle_coef * rle_intrinsic

            # Reward normalization
            if reward_rms is not None:
                reward_flat = reward.view(-1)
                reward_rms.update(reward_flat.unsqueeze(-1))
                reward = reward_flat / (reward_rms.var.squeeze().sqrt() + 1e-8)

            rewards[step] = reward.view(-1)

            # Auxiliary kill prediction targets
            if aux_targets is not None:
                env_state_aux = envs.env._state.env_state
                pl_aux = np.asarray(env_state_aux.player_level).astype(int)
                mk_aux = np.asarray(env_state_aux.monsters_killed)
                kills_aux = mk_aux[np.arange(args.num_envs), pl_aux].clip(max=8).astype(np.float32) / 8.0
                aux_targets[step] = torch.as_tensor(kills_aux, device=device)

            done_indices = torch.nonzero(next_done, as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                for idx in done_indices.tolist():
                    avg_ep_reward.append(infos['r'][idx])
                    avg_ep_length.append(infos['l'][idx])
                    achievement_logs = {k: v[idx].item() for k, v in infos.items() if 'Achievements' in k}
                    for k, v in achievement_logs.items():
                        if k not in avg_achievements:
                            avg_achievements[k] = deque(maxlen=25)
                        avg_achievements[k].append(v)
                    print(f"global_step={global_step}, episodic_return={infos['r'][idx]}, episodic_length={infos['l'][idx]}, avg_reward={np.mean(avg_ep_reward)}")

                # RLE: resample z for done envs (new episode = new exploration goal)
                if rle_z is not None:
                    done_idx_list = done_indices.tolist()
                    if isinstance(done_idx_list, int):
                        done_idx_list = [done_idx_list]
                    new_z = torch.randn(len(done_idx_list), args.rle_dim, device=device)
                    new_z = new_z / (new_z.norm(dim=-1, keepdim=True) + 1e-8)
                    rle_z[done_idx_list] = new_z
                    # Update z portion of obs for done envs
                    next_obs[done_idx_list, -args.rle_dim:] = new_z

            # Go-Explore: detect milestones and apply frontier resets
            if frontier_buffer is not None:
                # Detect and save new milestone states (using achievements from JAX state)
                prev_milestone_status, n_saved = detect_and_save_milestones(
                    frontier_buffer, achievements, prev_milestone_status,
                    next_done, envs.env._state.env_state, raw_next_obs, args.num_envs
                )
                frontier_total_saves += n_saved

                # Apply frontier resets to done envs
                if done_indices.numel() > 0 and len(frontier_buffer) >= 8:
                    done_list = done_indices.tolist()
                    if isinstance(done_list, int):
                        done_list = [done_list]
                    next_obs, n_resets = apply_frontier_resets(
                        envs, frontier_buffer, done_list, next_obs, raw_next_obs,
                        args.frontier_reset_prob, obs_rms, args.obs_clip, device
                    )
                    frontier_total_resets += n_resets

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_lstm_state,
                next_done,
                denormalize=True,
            ).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam

            # VC-PPO: separate critic returns with higher lambda for less biased value targets
            if args.gae_lambda_critic >= 0 and args.gae_lambda_critic != args.gae_lambda:
                critic_returns = torch.zeros_like(rewards).to(device)
                lastgaelam_c = 0
                for t in reversed(range(args.num_steps)):
                    if t == args.num_steps - 1:
                        nextnonterminal = 1.0 - next_done.float()
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                    critic_returns[t] = lastgaelam_c = delta + args.gamma * args.gae_lambda_critic * nextnonterminal * lastgaelam_c
                returns = critic_returns + values
            else:
                returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_aux_targets = aux_targets.reshape(-1) if aux_targets is not None else None

        # SIL: populate buffer with high-return transitions
        if sil_buffer is not None:
            progress = global_step / args.total_timesteps
            if progress >= args.sil_warmup_frac:
                # Update return history for percentile computation
                sil_return_history.extend(b_returns.cpu().numpy().tolist())
                if len(sil_return_history) > 100:
                    threshold = float(np.percentile(list(sil_return_history), args.sil_percentile))
                else:
                    threshold = float(b_returns.median().item())
                n_added = sil_buffer.add_batch(
                    b_obs.detach(), b_actions.long().detach(), b_returns.detach(), threshold
                )

        # optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []

        # Update PopArt stats ONCE per iteration
        if args.use_popart:
            agent.critic_head.update_stats(b_returns.unsqueeze(-1))

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                _, newlogprob, entropy, newvalue, _, aux_pred = agent.get_action_and_value(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # value loss
                newvalue = newvalue.view(-1)
                if args.use_popart:
                    mb_returns_norm = agent.critic_head.normalize(b_returns[mb_inds].unsqueeze(-1)).squeeze(-1)
                    v_loss = 0.5 * ((newvalue - mb_returns_norm) ** 2).mean()
                elif args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef_now * entropy_loss + v_loss * args.vf_coef

                # Auxiliary kill prediction loss
                if aux_pred is not None and b_aux_targets is not None:
                    aux_loss = ((aux_pred.squeeze(-1) - b_aux_targets[mb_inds]) ** 2).mean()
                    loss = loss + args.aux_kill_coef * aux_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

                # RND predictor update (separate optimizer, doesn't affect PPO agent)
                if rnd_predictor is not None:
                    obs_flat_rnd = b_obs[mb_inds].view(-1, b_obs.shape[-1])
                    with torch.no_grad():
                        rnd_tgt = rnd_target(obs_flat_rnd)
                    rnd_pred = rnd_predictor(obs_flat_rnd)
                    rnd_loss = (rnd_tgt - rnd_pred).pow(2).mean()
                    rnd_optimizer.zero_grad()
                    rnd_loss.backward()
                    rnd_optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # SIL update: learn from high-return transitions in replay buffer
        sil_pg_loss_val = 0.0
        sil_vf_loss_val = 0.0
        sil_n_pos = 0
        if sil_buffer is not None and len(sil_buffer) >= args.sil_batch_size:
            progress = global_step / args.total_timesteps
            if progress >= args.sil_warmup_frac:
                # Progressive SIL: ramp up coefficient after warmup
                sil_progress = (progress - args.sil_warmup_frac) / (1.0 - args.sil_warmup_frac)
                sil_coef_now = args.sil_coef * min(1.0, sil_progress * 2.0)  # reach full coef at midpoint

                sil_pg, sil_vf, sil_n_pos = compute_sil_loss(
                    agent, sil_buffer, args.sil_batch_size, device
                )
                if sil_n_pos > 0:
                    sil_loss = sil_coef_now * sil_pg + args.sil_vf_coef * sil_vf
                    optimizer.zero_grad()
                    sil_loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()
                    sil_pg_loss_val = sil_pg.item()
                    sil_vf_loss_val = sil_vf.item()

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        extra = ""
        if frontier_buffer is not None:
            extra += f" | GoExplore: buffer={len(frontier_buffer)}, saves={frontier_total_saves}, resets={frontier_total_resets}"
        if prev_floor_kills is not None:
            extra += f" | KillBonus: kills={kill_bonus_total:.0f}, floors={floor_bonus_total:.0f}"
        if sil_buffer is not None:
            extra += f" | SIL: buf={len(sil_buffer)}, pos={sil_n_pos}, pg={sil_pg_loss_val:.4f}"
        print(f"SPS: {sps}{extra}")

        if iteration in args.log_iterations:
            # log training data to csv
            row_to_write = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                            int(global_step / (time.time() - start_time)),
                            global_step,
                            np.mean(avg_ep_reward),
                            np.mean(avg_ep_length)] + [np.mean(v) for v in avg_achievements.values()]
            write_row_csv(training_csv_writer, row_to_write)

            # log losses to csv
            row_to_write = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                    int(global_step / (time.time() - start_time)),
                                    global_step,
                                    optimizer.param_groups[0]["lr"],
                                    v_loss.item(),
                                    pg_loss.item(),
                                    entropy_loss.item(),
                                    old_approx_kl.item(),
                                    approx_kl.item(),
                                    np.mean(clipfracs),
                                    explained_var]
            write_row_csv(losses_csv_writer, row_to_write)

            if args.track:
                wandb.log({
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "losses/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/ep_reward": np.mean(avg_ep_reward),
                    "charts/ep_length": np.mean(avg_ep_length),
                    "charts/global_step": global_step,
                    "charts/sps": sps,
                    **{f"{k}": np.mean(v) for k, v in avg_achievements.items()},
                }, step=global_step)

    torch.save(agent.state_dict(), f"runs/{run_name}/agent_{iteration}.pt")
    save_results_csv(args, "PPO-LSTM", avg_ep_reward, avg_ep_length, avg_achievements, global_step, time.time() - start_time)
