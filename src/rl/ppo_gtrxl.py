"""PPO with Gated Transformer-XL (GTrXL) memory for Craftax.

Uses a Transformer-XL architecture with GRU gating for long-term memory,
replacing the LSTM in ppo_lstm.py. Based on "Stabilizing Transformers for
Reinforcement Learning" (Parisotto et al., 2020).
"""
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro

from collections import deque

from src.env.env import make_craftax_env
from src.utils.logger import write_row_csv, make_training_csv_craftax_classic, make_training_csv_craftax, make_ppo_losses_csv, save_results_csv
from src.models.gtrxl import PPO_GTrXL_Agent
from src.utils.args import GTrXL_Args

if __name__ == "__main__":
    import os
    os.environ["WANDB__SERVICE_WAIT"] = "300"

    args = tyro.cli(GTrXL_Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.log_iterations = np.linspace(0, args.num_iterations, args.num_logs).astype(int)

    algo_name = "PPO-GTrXL"
    run_name = f"algo:{algo_name}_env:{args.env_id}_seed:{args.seed}_{int(time.time())}"

    print(f"Running PPO-GTrXL with num_envs={args.num_envs}, num_steps={args.num_steps}, minibatch_size={args.minibatch_size}")

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

    # GTrXL architecture hyperparameters
    trxl_hidden = args.hidden_size
    trxl_heads = args.trxl_heads
    trxl_layers = args.trxl_layers
    trxl_memory = args.trxl_memory if args.trxl_memory > 0 else args.num_steps
    trxl_mlp_layers = args.trxl_mlp_layers

    agent = PPO_GTrXL_Agent(
        obs_dim=np.array(envs.single_observation_space.shape).prod(),
        n_actions=envs.single_action_space.n,
        hidden_size=trxl_hidden,
        num_heads=trxl_heads,
        num_transformer_layers=trxl_layers,
        memory_len=trxl_memory,
        num_mlp_layers=trxl_mlp_layers,
        use_structured_obs=args.use_structured_obs,
        use_popart=args.use_popart,
    ).to(device)
    print("-------------")
    print(agent)
    num_params = sum(p.numel() for p in agent.parameters())
    print(f"Total parameters: {num_params:,}")

    opt_kwargs = {"lr": args.learning_rate}
    if "adam" in args.optimizer:
        opt_kwargs["eps"] = 1e-5
    optimizer = torch.optim.Adam(agent.parameters(), **opt_kwargs)

    # storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize transformer memory
    next_memory = agent.init_memory(args.num_envs, device)

    # start the game
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    start_time = time.time()

    # stats tracking
    avg_ep_reward = deque(maxlen=25)
    avg_ep_length = deque(maxlen=25)
    avg_achievements = {}

    for iteration in range(1, args.num_iterations + 1):
        # Save initial memory state for PPO update
        initial_memory = [m.clone() for m in next_memory]

        # annealing the rate if instructed to do so.
        frac = 1.0 - (iteration - 1.0) / args.num_iterations
        if args.anneal_lr:
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # entropy coefficient annealing
        if args.ent_coef_end >= 0:
            ent_coef_now = args.ent_coef_end + frac * (args.ent_coef - args.ent_coef_end)
        else:
            ent_coef_now = args.ent_coef

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # action logic
            with torch.no_grad():
                action, logprob, _, value, next_memory = agent.get_action_and_value(
                    next_obs, next_memory, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations)
            rewards[step] = reward.view(-1)

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

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(
                next_obs,
                next_memory,
                next_done,
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
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # optimizing the policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []

        # Update PopArt stats ONCE per iteration (not per-minibatch)
        if args.use_popart:
            agent.critic_head.update_stats(b_returns.unsqueeze(-1))

        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                # Slice initial memory for this minibatch of environments
                mb_memory = [m[:, mbenvinds].contiguous() for m in initial_memory]

                _, newlogprob, entropy, newvalue, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mb_memory,
                    b_dones[mb_inds],
                    b_actions.long()[mb_inds],
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
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
                    # Normalize targets using stats updated once per iteration (above)
                    mb_returns_norm = agent.critic_head.normalize(b_returns[mb_inds].unsqueeze(-1)).squeeze(-1)
                    mb_values_norm = agent.critic_head.normalize(b_values[mb_inds].unsqueeze(-1)).squeeze(-1)
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

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")

        if iteration in args.log_iterations:
            row_to_write = [time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                            int(global_step / (time.time() - start_time)),
                            global_step,
                            np.mean(avg_ep_reward),
                            np.mean(avg_ep_length)] + [np.mean(v) for v in avg_achievements.values()]
            write_row_csv(training_csv_writer, row_to_write)

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
    save_results_csv(args, algo_name, avg_ep_reward, avg_ep_length, avg_achievements, global_step, time.time() - start_time)
