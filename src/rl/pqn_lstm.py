
import random
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from collections import deque
from src.env.env import make_craftax_env
from src.utils.logger import write_row_csv, make_training_csv_craftax_classic, make_training_csv_craftax, make_pqn_losses_csv, save_results_csv
from src.models.models import PQN_LSTM_Agent
from src.utils.args import PQN_Args
from src.utils.utils import get_optimizer_class, get_activation_fn, linear_schedule

if __name__ == "__main__":
    import os
    os.environ["WANDB__SERVICE_WAIT"] = "300"
    
    args = tyro.cli(PQN_Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    args.log_iterations = np.linspace(0, args.num_iterations, args.num_logs).astype(int)
    run_name = f"algo:PQN-LSTM_env:{args.env_id}_seed:{args.seed}_{int(time.time())}"
    print(f"Running with num_envs={args.num_envs}, num_steps={args.num_steps}, minibatch_size={args.minibatch_size}")
    
    args.max_grad_norm = 0.5
    print("Overriding max_grad_norm to 0.5 since we're using LSTM")
    
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
    
    losses_csv_writer = make_pqn_losses_csv(f"./runs/{run_name}")

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
    
    agent = PQN_LSTM_Agent(
        obs_dim=np.array(envs.single_observation_space.shape).prod(),
        n_actions=envs.single_action_space.n,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        activation_fn=get_activation_fn(args.activation_fn),
        use_ln=args.use_ln,
    ).to(device)
    print("-------------")
    print(agent)

    optimizer = get_optimizer_class(args.optimizer)(agent.parameters(), lr=args.learning_rate)

    # storage setup
    obs_shape = envs.single_observation_space.shape
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    next_lstm_state = (
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
        torch.zeros(agent.lstm.num_layers, args.num_envs, agent.lstm.hidden_size).to(device),
    )

    # start the game
    print("Resetting environments for the first time")
    global_step = 0
    next_obs, _ = envs.reset(seed=args.seed)
    next_done = torch.zeros(args.num_envs).to(device)
    start_time = time.time()
    
    # stats tracking
    avg_ep_reward = deque(maxlen=25)
    avg_ep_length = deque(maxlen=25)
    avg_achievements = {}
    for iteration in range(1, args.num_iterations + 1):
        initial_lstm_state = (next_lstm_state[0].clone(), next_lstm_state[1].clone())
        
        # annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            
            epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
            random_actions = torch.randint(0, envs.single_action_space.n, (args.num_envs,), device=device)

            # action logic
            with torch.no_grad():
                q_values, next_lstm_state = agent(next_obs, next_lstm_state, next_done)
                max_actions = torch.argmax(q_values, dim=1)
                values[step] = q_values[torch.arange(args.num_envs), max_actions].flatten()
                
            explore = torch.rand((args.num_envs,), device=device) < epsilon
            action = torch.where(explore, random_actions, max_actions)
            actions[step] = action

            # execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(action)
            next_done = torch.logical_or(terminations, truncations)
            rewards[step] = reward.view(-1)
            
            done_indices = torch.nonzero(next_done, as_tuple=False).squeeze(-1)
            if done_indices.numel() > 0:
                for idx in done_indices.tolist():
                    # record stats but not yet log them
                    avg_ep_reward.append(infos['r'][idx])
                    avg_ep_length.append(infos['l'][idx])
                    achievement_logs = {k: v[idx].item() for k, v in infos.items() if 'Achievements' in k}
                    for k, v in achievement_logs.items():
                        if k not in avg_achievements:
                            avg_achievements[k] = deque(maxlen=25)
                        avg_achievements[k].append(v)
                    print(f"global_step={global_step}, episodic_return={infos['r'][idx]}, episodic_length={infos['l'][idx]}, avg_reward={np.mean(avg_ep_reward)}")

        # Compute Q(lambda) targets
        with torch.no_grad():
            returns = torch.zeros_like(rewards).to(device)
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    next_value, _ = torch.max(agent(next_obs, next_lstm_state, next_done)[0], dim=-1)
                    nextnonterminal = 1.0 - next_done.float()
                    returns[t] = rewards[t] + args.gamma * next_value * nextnonterminal
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    next_value = values[t + 1]
                    returns[t] = (
                        rewards[t]
                        + args.gamma * (args.q_lambda * returns[t + 1] + (1 - args.q_lambda) * next_value) * nextnonterminal
                    )

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_returns = returns.reshape(-1)
        b_dones = dones.reshape(-1)

        # optimize the Q-network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(args.batch_size).reshape(args.num_steps, args.num_envs)
        for epoch in range(args.update_epochs):
            np.random.shuffle(envinds)
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()  # be really careful about the index

                old_val, _ = agent(
                    b_obs[mb_inds],
                    (initial_lstm_state[0][:, mbenvinds], initial_lstm_state[1][:, mbenvinds]),
                    b_dones[mb_inds],
                )
                old_val = old_val.gather(1, b_actions[mb_inds].unsqueeze(-1).long()).squeeze()
                
                loss = 0.5 * F.mse_loss(b_returns[mb_inds], old_val)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()
                
        sps = int(global_step / (time.time() - start_time))
        print(f"SPS: {sps}")
        
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
                                    loss.item()]
            write_row_csv(losses_csv_writer, row_to_write)
            
            if args.track:
                wandb.log({
                    "losses/td_loss": loss.item(),
                    "losses/learning_rate": optimizer.param_groups[0]["lr"],
                    "charts/ep_reward": np.mean(avg_ep_reward),
                    "charts/ep_length": np.mean(avg_ep_length),
                    "charts/global_step": global_step,
                    "charts/sps": sps,
                    **{f"{k}": np.mean(v) for k, v in avg_achievements.items()},
                }, step=global_step)
    
    torch.save(agent.state_dict(), f"runs/{run_name}/agent_{iteration}.pt")
    save_results_csv(args, "PQN-LSTM", avg_ep_reward, avg_ep_length, avg_achievements, global_step, time.time() - start_time)