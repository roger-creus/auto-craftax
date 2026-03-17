# Research Goal

## Objective
Achieve **state-of-the-art performance on Craftax-Symbolic-v1** with 1B environment steps. The ultimate goal is to **beat the full game** — progress through all 9 floors and defeat the necromancer boss on Floor 8. Performance is measured by game score (episode return) and game completion rate.

This is an engineering + research challenge. Whatever works is good. Novel ideas are welcome but the priority is **results** — push the score as high as possible by any means necessary. Do not stop after small improvements. Continue to hill climb aggressively until the game is beaten or you've exhausted all ideas.

## Environment
**Craftax-Symbolic-v1** — a procedurally generated dungeon crawling game with crafting, combat, resource management, and 9 increasingly difficult floors. See `tutorial.md` for a full walkthrough of game mechanics, floors, enemies, crafting, enchanting, and the final boss fight.

Key challenges:
- Long-horizon decision making (surviving across 9 floors)
- Resource management (health, hunger, thirst, energy, mana)
- Crafting and equipment progression (wood → stone → iron → diamond tools/armor)
- Combat with diverse enemy types requiring different damage types (physical, fire, ice)
- Exploration of procedurally generated maps
- Potion identification through trial and error

## Benchmark
- **Environment:** Craftax-Symbolic-v1
- **Training budget:** 1,000,000,000 (1B) environment steps — this is fixed, cannot change
- **Seeds:** 3 seeds (1, 2, 3) for statistical validity
- **Metric:** Episode return (game score). Also track: max floor reached, game completion rate, achievement percentages

ALL JOBS NEED TO USE AT LEAST 1 GPU.

## Baselines
Four algorithms are implemented in `src/rl/`:

1. **PPO** (`src/rl/ppo.py`) — standard PPO with MLP network
2. **PPO-LSTM** (`src/rl/ppo_lstm.py`) — PPO with LSTM memory
3. **PQN** (`src/rl/pqn.py`) — Parallelized Q-Network
4. **PQN-LSTM** (`src/rl/pqn_lstm.py`) — PQN with LSTM memory

**Phase 0 (FIRST!):** Make sure all 4 baselines actually run in the Singularity container. The code may be outdated — fix any issues. Run each baseline on seed 1 for a short test (e.g., 10M steps) to verify they work. Then run all 4 × 3 seeds for the full 1B steps.

Default parameters are in `src/utils/args.py` and are reasonable starting points. All algorithms use:
- `num_envs = 1024`, `num_steps = 64`, `lr = 0.0002`
- 5-layer MLP with hidden_size 512
- The LSTM variants add recurrence on top

**How to run:**
```bash
python src/rl/ppo.py --seed 1 --total-timesteps 1000000000
python src/rl/ppo_lstm.py --seed 1 --total-timesteps 1000000000
python src/rl/pqn.py --seed 1 --total-timesteps 1000000000
python src/rl/pqn_lstm.py --seed 1 --total-timesteps 1000000000
```

## What Counts as SOTA
- **Minimum:** Beat all 4 baselines by >30% in mean episode return across 3 seeds
- **Target:** Consistently reach Floor 7+ (troll mines) within 1B steps
- **Stretch:** Beat the full game (defeat the necromancer) within 1B steps
- **Do NOT stop** after beating baselines by a small margin. Keep going. The game has 9 floors — push deeper.

## Directions to Explore

You are free to try ANYTHING. Restructure code, install new libraries, change architectures, combine algorithms — whatever maximizes performance. These are suggestions, not limits — come up with your own ideas too.

### Architecture
- Transformer/attention-based memory (GTrXL) instead of LSTM
- Larger networks, deeper networks, residual connections
- Separate networks for different aspects (combat vs exploration vs resource management)
- State encoding improvements — the symbolic observation has rich structure, exploit it

### Algorithmic
- Combine PPO + PQN (actor-critic with Q-learning auxiliary)
- Intrinsic motivation / curiosity (RND, ICM or novel) — crucial for sparse-reward exploration
- Hierarchical RL — options/skills framework for multi-step behaviors (crafting sequences, combat routines)
- Population-based training — evolve hyperparameters across runs
- Learned Reward shaping
- Self-play or adversarial training

### Craftax-specific
- Curriculum learning — train on easier configurations first, then transfer
- Achievement-based rewards — Craftax has built-in achievements, use them as learning signals
- Memory and planning — the agent needs to remember what resources it has, what it needs to craft
- Multi-phase strategy — different policies for overworld vs dungeon vs boss
- Read `tutorial.md` to understand optimal play strategy and encode that knowledge

### Optimization
- Modern optimizers (Lion, Muon, schedule-free)
- Learning rate schedules tuned for 1B steps
- Gradient accumulation for effective larger batch sizes
- Mixed precision training for speed

### Search online
- Look up papers on Craftax, Crafter, and similar procedurally generated environments
- Check for existing SOTA results and methods
- Find implementation tricks from the Craftax community

## Code Structure
- `src/rl/` — RL algorithms (PPO, PQN, + LSTM variants)
- `src/models/` — neural network architectures
- `src/env/` — environment wrappers
- `src/utils/` — args, logging, utilities
- `tutorial.md` — full game walkthrough (READ THIS to understand the challenge)

You are free to modify ANY file, add new files, restructure the codebase, install new dependencies (rebuild container if needed). The journal should document what you changed and why at each iteration.

## Experiment Protocol

### Phase 0: Verify baselines work
1. Build and push the Singularity container
2. Run each of the 4 baselines for a short test (10M steps, 1 seed) to verify they work in the container
3. Fix any bugs (outdated imports, missing dependencies, etc.)
4. Record working commands in the journal

### Phase 1: Full baselines
1. Run all 4 baselines × 3 seeds × 1B steps
2. Record baseline results in the results bank
3. Analyze: which baseline is best? Which floor does each reach? Where does it get stuck?

### Phase 2: Iterate and improve
1. Analyze baseline failures — why does the agent get stuck? What floor? What kills it?
2. Implement improvements one at a time
3. Test each improvement for 100M steps first (quick validation)
4. If promising, run full 1B steps × 3 seeds
5. Combine winning components
6. **Keep pushing — don't stop at small gains. The game has 9 floors. Beat them all.**

### Phase 3: SOTA
1. Run best algorithm × 3 seeds × 1B steps
2. Ablation study if time permits
3. Document the full method and results

## Compute Budget
You have access to a LOT of compute across 4 clusters (rorqual, narval, nibi, fir) with H100 and A100 GPUs. Jobs schedule fast. Submit large batches in parallel. Use all clusters. Don't be conservative — the faster you iterate the faster you improve.

## Output Format
When recording results, include:
- Episode return (mean ± std across seeds)
- Max floor reached (mean and max across seeds)
- Game completion rate (if any)
- Achievement percentages (if available)
- Training wall-clock time
- Any notable observations (where the agent gets stuck, training instability, etc.)

## Sources
- `tutorial.md` — full game walkthrough (MUST READ)
- `src/utils/args.py` — default hyperparameters
- Craftax paper: https://arxiv.org/abs/2402.16801
- Craftax GitHub: https://github.com/MichaelTMatthews/Craftax
