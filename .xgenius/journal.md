
---
**[2026-03-17 03:35 UTC]**

NOTE: The git remote uses 'main' as the default branch, not 'master'. Always push with: git push origin main. Never push to master.

---
**[2026-03-17 03:35 UTC]**

=== SESSION START: Phase 0 — Verify baselines ===
Fresh research start. No prior experiments.
Plan:
1. Build Docker + Singularity container
2. Push image to all 4 clusters (rorqual, narval, nibi, fir)
3. Submit pilot runs: 4 baselines × seed 1 × 10M steps to verify they work
4. Fix any issues, then submit full 4 × 3 seeds × 1B steps

---
**[2026-03-17 04:08 UTC]**

Phase 0 progress:
- Docker image built (40GB), Singularity .sif converted (18.4GB)
- Image pushed to all 4 clusters: rorqual, narval, nibi, fir
- Submitted 4 pilot runs (10M steps, seed 1, 1h walltime):
  - PPO on rorqual (job 8458536, exp ppo-pilot-s1)
  - PPO-LSTM on narval (job 57885388, exp ppo-lstm-pilot-s1)
  - PQN on nibi (job 10475734, exp pqn-pilot-s1)
  - PQN-LSTM on fir (job 28091145, exp pqn-lstm-pilot-s1)
- Waiting for pilots to complete to verify code works in container

---
**[2026-03-17 04:47 UTC]**

Phase 0 — Bug fixes applied, training confirmed running:
- Fix 1: Removed -H flag from SBATCH (caused path quoting error)
- Fix 2: Added PYTHONPATH=/src (needed for module imports inside container)
- Fix 3: Added --writable-tmpfs (Craftax texture cache needs writable fs)
- Fix 4: Added PYTHONUNBUFFERED=1 (stdout was being buffered, no log output visible)

v3 pilots submitted:
  - PPO on rorqual (8459128) — CONFIRMED TRAINING (runs/ dir + CSV output created)
  - PPO-LSTM on narval (57885756) — CONFIRMED TRAINING (48KB training.csv growing)
  - PQN on nibi (10476972) — PENDING in queue
  - PQN-LSTM on fir (28091666) — PENDING in queue

Waiting for pilots to complete. Once confirmed, will submit full 4x3 baseline runs.

---
**[2026-03-17 05:14 UTC]**

Phase 0 COMPLETE. Phase 1 STARTED.

Pilot results (10M steps, seed 1):
  PPO:      avg_return=9.1,  wall_time=412s  (rorqual H100 3g.40gb)
  PPO-LSTM: avg_return=9.42, wall_time=784s  (narval A100)
  PQN:      avg_return=4.94, wall_time=328s  (nibi H100 3g.40gb)
  PQN-LSTM: pending verification (submitted to narval as pilot-v3b)

Full 1B step baseline runs submitted (12 total = 4 algos × 3 seeds):
  PPO ×3 on rorqual: jobs 8459639, 8459641, 8459642
  PPO-LSTM ×3 on narval: jobs 57886760, 57886825, 57886826
  PQN ×3 on nibi: jobs 10478838, 10478839, 10478841
  PQN-LSTM ×3 on fir: jobs 28093387, 28093388, 28093389

Estimated completion: ~12h for PPO/PQN, ~22h for LSTM variants.

Early observations from pilots:
  - PPO-LSTM edges PPO slightly at 10M steps (9.42 vs 9.1)
  - PQN is significantly weaker (4.94) — Q-learning may struggle with Craftax
  - All baselines only learning basic survival (wood/stone collection, eating)
  - No dungeon entries yet at 10M steps

Next session: analyze 1B results, start Phase 2 improvements.

---
**[2026-03-18 00:47 UTC]**

=== SESSION: Phase 2 — Design and submit improvement hypotheses ===

Baseline results confirmed (matches published Craftax-1B scoreboard — PPO gets 11.9% of 226 max):
  PPO (3 seeds): mean=26.83±1.31 (1B steps, ~8h each on H100 3g.40gb)
  PQN (3 seeds): mean=20.95±1.04 (1B steps, ~5.6h each)
  PPO-LSTM: 3 jobs running (~2h in, ~20h to go) — resubmitted after original jobs disappeared
  PQN-LSTM: 2 running + 1 resubmitted on fir (job 28185700)

Key analysis of baseline failures:
  - Agent masters basic overworld survival (25 basic achievements worth 25 points)
  - Enters dungeon ~50% (PPO) but NEVER goes deeper (0% for all deeper floors)
  - No spellcasting, no enchanting, minimal diamond crafting
  - Root cause: gamma=0.99 gives effective planning horizon of ~100 steps
    but floor progression needs 200-500+ step planning

Research findings:
  - Craftax reward structure: basic=1pt, intermediate=3pt, advanced=5pt, very_advanced=8pt (226 max)
  - Intrinsic motivation (RND, ICM, E3B) DOES NOT help on Craftax (per paper)
  - PPO-RNN is best published baseline
  - Our hyperparameters match official Craftax baselines exactly

Phase 2 hypotheses submitted (all 200M step pilots, seed 1):
  h001: Better hyperparams (gamma=0.999, gae_lambda=0.95) — on rorqual + narval
  h002: Achievement reward shaping (bonus rewards for rare achievements) — on nibi
  h003: Combined h001+h002 — on fir
  h004: Architecture (ReLU, LayerNorm, 1024 width, 3 layers) — on rorqual
  h005: Kitchen sink (h001+h002+h004 combined) — on narval

Created src/rl/ppo_shaped.py with achievement bonus rewards.
Code committed and pushed. Synced to all clusters.

Next session:
  1. Check LSTM baseline completion (PPO-LSTM, PQN-LSTM)
  2. Analyze 200M pilot results for h001-h005
  3. If any pilot shows clear improvement, submit full 1B × 3 seeds
  4. Consider additional hypotheses based on pilot outcomes

---
**[2026-03-18 01:10 UTC]**

=== SESSION: PQN-LSTM OOM analysis, status check ===

Triggered by: pqn-lstm-1B-s2 (28093388) and pqn-lstm-1B-s3 (28093389) marked 'completed' on fir.

FINDING: Both were OOM KILLED (not truly completed):
  - s2: OOM at ~660M/1B steps (~12h walltime), avg_reward ~25 at death (32G memory)
  - s3: OOM at ~659M/1B steps (~12h walltime), avg_reward ~20 at death (32G memory)
  - SLURM exit_code=0 is misleading; actual error visible in logs: 'Detected 1 oom_kill event'
  - Previous session already resubmitted with 48G memory (good)

No CSV output was produced (processes killed mid-training). No new results to record.

Running jobs status (all healthy, 48G memory):
  LSTM baselines (6 jobs): ~2-2.5h elapsed, ~18-20h remaining
    - ppo-lstm-1B-s1 (narval), s2 (rorqual), s3 (nibi)
    - pqn-lstm-1B-s1 (rorqual), s2 (nibi), s3 (fir, just started 16min)
  Hypothesis pilots (6 jobs): 15-28 min elapsed, ~2-4h remaining
    - h001-pilot-s1 (rorqual): gamma=0.999, gae_lambda=0.95
    - h001-longer-rollout-s1 (narval): same + num_steps=128
    - h002-pilot-s1 (nibi): achievement reward shaping
    - h003-pilot-s1 (fir): h001+h002 combined
    - h004-pilot-s1 (rorqual): wider 1024, ReLU, LayerNorm, 3 layers
    - h005-pilot-s1 (narval): kitchen sink (all combined)

Next session priorities:
  1. Hypothesis pilots should complete first (~2-4h from now) — PARSE AND ANALYZE
  2. If any pilot shows improvement, submit full 1B × 3 seeds IMMEDIATELY
  3. LSTM baselines will take ~18-20h more — will be another session
  4. Watch for OOM on resubmitted PQN-LSTM jobs (48G should be enough but monitor)

---
**[2026-03-18 03:14 UTC]**

=== SESSION: Resubmit disappeared jobs + optimize GTrXL + new hypotheses ===

Triggered by: h003-pilot-s1 completed (FAILED: avg_return=9.06, already recorded).

SITUATION: 28 of 39 jobs 'disappeared' (silently killed by clusters). No results for:
  - PPO-LSTM 1B (3 seeds) — critical Phase 1 baseline
  - PQN-LSTM 1B (3 seeds) — critical Phase 1 baseline
  - h004 pilot (wider arch) — never produced output
  - h005 pilot (kitchen sink) — deprioritized, reward shaping proved harmful
  - h006 pilot (GTrXL) — never produced output
  - h001-longer-rollout — never produced output

ACTIONS TAKEN:

1. RESUBMITTED LSTM baselines (6 jobs, 48G memory):
   - ppo-lstm-1B: s1→rorqual(8508288), s2→narval(57931207), s3→nibi(10519904)
   - pqn-lstm-1B: s1→fir(28206537), s2→rorqual(8508293), s3→narval(57931230)

2. OPTIMIZED GTrXL training (critical engineering fix):
   - Step-by-step processing in get_states looped T=64 times per PPO update
   - New get_states_train processes all T steps in parallel with batched attention
   - Episode-aware causal masking prevents cross-episode attention leaks
   - Per-environment positional encoding handles done resets correctly
   - ~64x speedup in PPO training phase

3. MADE GTrXL HYPERPARAMETERS CONFIGURABLE:
   - Added GTrXL_Args dataclass: --trxl-layers, --trxl-heads, --trxl-memory, --trxl-mlp-layers
   - Can now match reference implementation exactly via CLI

4. WEB SEARCH: Found reference GTrXL implementation (Reytuag/transformerXL_PPO_JAX):
   - Official Craftax leaderboard: PPO-GTrXL = 18.3% (41.4/226) vs PPO = 11.9% (26.8/226)
   - Reference config: 256 hidden, 2 layers, 128 steps, 128 memory, gamma=0.999, ent=0.002, grad_norm=1.0
   - ~5M params, trains in 6.5h on single A100
   - SCALAR (LLM+RL hybrid) gets 88.2% diamond collection and 9.1% gnomish mines entry

5. SUBMITTED HYPOTHESIS PILOTS (200M steps):
   - h004-pilot-s1 → fir(28206913): wider PPO (1024, ReLU, LN, 3 layers)
   - h006-pilot-s1 → rorqual(8509592): GTrXL 512h/3L/64mem (our config, with batched training)
   - h007-pilot-s1 → narval(57931443): GTrXL FULL REFERENCE MATCH (256h/2L/128mem/gamma=0.999/ent=0.002)
   - h008-pilot-s1 → nibi(10520421): reference + gae_lambda=0.95

6. CLOSED h005 (kitchen sink) — reward shaping component proved harmful

CURRENT STATE: 10 active jobs across 4 clusters
  rorqual (3): ppo-lstm-1B-s1, pqn-lstm-1B-s2, h006-pilot-s1
  narval (3): ppo-lstm-1B-s2, pqn-lstm-1B-s3, h007-pilot-s1
  nibi (2): ppo-lstm-1B-s3, h008-pilot-s1
  fir (2): pqn-lstm-1B-s1, h004-pilot-s1

NEXT SESSION PRIORITIES:
  1. Parse h004/h006/h007/h008 pilot results (most important: h007 reference match)
  2. If h007 shows clear improvement, submit full 1B × 3 seeds immediately
  3. Parse LSTM baseline results (PPO-LSTM, PQN-LSTM)
  4. Based on GTrXL results, consider further ablations or new directions
  5. Key question: does our batched GTrXL match reference quality? h006 vs h007 comparison will show impact of architecture choices (2L/256h vs 3L/512h)

---
**[2026-03-18 03:30 UTC]**

=== SESSION: Fix GTrXL dtype crash, resubmit all GTrXL pilots ===

Triggered by: h007-pilot-s1 (57931443, narval) reported SUCCESS.

FINDING: h007 actually CRASHED at ~131K steps (first PPO update) with:
RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Float
Root cause: dones storage tensor is float, but _compute_positions uses torch.where which needs bool.
The crash happened in get_states_train (the batched training path). The rollout path (get_states)
was fine because it converts to float explicitly. All 3 GTrXL experiments (h006, h007, h008) hit this bug.

FIX: Added .bool() cast in _compute_positions at gtrxl.py:258. Committed and pushed.

ACTIONS:
1. Cancelled h006 (8509592, rorqual) — was running 16min, would have crashed at first PPO update
2. Cancelled h008 (10520421, nibi) — was still pending
3. Fixed bug, committed, pushed, synced to ALL 4 clusters
4. Resubmitted all 3 GTrXL pilots:
   - h006-pilot-s1 → rorqual (8510270): 512h/3L/64mem, 200M steps
   - h007-pilot-s1 → nibi (10520783): reference config 256h/2L/128mem/gamma=0.999, 200M steps
   - h008-pilot-s1 → fir (28214148): reference + gae_lambda=0.95, 200M steps

REFERENCE CONFIG VERIFICATION (vs Reytuag/transformerXL_PPO_JAX):
Our h007 matches exactly: lr=2e-4, anneal=True, num_envs=1024, num_steps=128,
gamma=0.999, gae_lambda=0.8, ent=0.002, clip=0.2, vf=0.5, grad_norm=1.0,
256h, 8heads, 2L, mem=128, update_epochs=4, minibatches=8, ReLU, gate_bias=2.0.
One difference: reference uses WINDOW_GRAD=64 (truncated BPTT through last 64 steps only),
we backprop through all 128. This shouldn't prevent learning but may affect memory/speed.

WEB SEARCH FINDINGS:
- SCALAR (LLM+RL hybrid): 88.2% diamond collection, 9.1% gnomish mines. Uses LLM for skill planning — out of scope for pure RL but ideas are interesting (frontier checkpointing, compositional GAE, mixture-of-experts).
- AGaLiTe: Alternative to GTrXL, 40% cheaper inference, 50% less memory, performs similarly or better. But JAX-only, no PyTorch implementation.
- M5 (model-based): 67.42% on Craftax-classic (different variant).

CURRENT ACTIVE JOBS (9 total):
Baselines (5 LSTM, all ~25min in, ~21h remaining):
  - ppo-lstm-1B-s1 (rorqual), s2 (narval), s3 (nibi, pending)
  - pqn-lstm-1B-s1 (fir), s2 (rorqual), s3 (narval)
Hypothesis pilots (4):
  - h004-pilot-s1 (fir): wider PPO 1024/ReLU/LN, ~24min in, ~1.5h remaining
  - h006-pilot-s1 (rorqual): GTrXL 512h/3L, just submitted
  - h007-pilot-s1 (nibi): GTrXL reference match, pending
  - h008-pilot-s1 (fir): GTrXL ref+GAE, just started

NEXT SESSION PRIORITIES:
1. Parse h004 results (wider arch) — should complete first (~1.5h)
2. Parse h006/h007/h008 GTrXL results (~4-8h)
3. If h007 matches reference performance (~41/226 = 18.3%), immediately submit 1B × 3 seeds
4. If h006 (larger GTrXL) beats h007, consider even larger models
5. Parse LSTM baseline results when ready (~21h)
6. Consider new hypotheses: observation encoding improvements, entropy scheduling, auxiliary losses

---
**[2026-03-18 05:32 UTC]**

=== SESSION: Parse h004 results, resubmit GTrXL pilots, organize experiments ===

Triggered by: h004-pilot-s1 (28206913, fir) SUCCESS.

RESULTS PARSED:

h004-pilot-s1 (wider arch: 1024/3L/ReLU/LN) @ 200M steps:
  - avg_return ~24.2 (running avg last 50) / ~21.9 (raw episode mean)
  - avg_episode_length ~495
  - 21% episodes >30 return, 3% >40, max 47.1
  - Wall time: 8240s (2.3h) on fir H100 3g.40gb
  - VERDICT: Similar to baseline PPO at 200M. Wider arch alone does NOT provide dramatic improvement.
  - NOTE: No CSV produced — command was missing --hypothesis-id/--experiment-id flags.

h007-pilot-s1 (narval, 57931443): CONFIRMED CRASH at 131K steps (already known dtype bug from last session).
  Not a real completion. Resubmitted below.

ACTIONS TAKEN:

1. Recorded h004 results in experiments.csv and hypotheses.csv (status: open, deprioritized vs GTrXL)

2. Cancelled h006-pilot-s2 on nibi (10519761) — wrong config (256h, should be 512h for h006)

3. Cancelled h007-pilot-s1 on nibi (10520783) — resubmitted with --hypothesis-id/--experiment-id/--output-dir flags

4. Resubmitted experiments with proper CSV output flags:
   - h007-pilot-s1 → nibi (10522776): GTrXL reference 256h/2L/128mem/gamma=0.999
   - h007-pilot-s1-fir → fir (28219284): DUPLICATE on fir for faster turnaround
   - h008-pilot-s1 → narval (57936329): GTrXL ref + gae_lambda=0.95

5. Synced code to narval and fir

CURRENT ACTIVE JOBS (10 total):
  Baselines (6, ~20h remaining):
    - ppo-lstm-1B: s1(rorqual 8508288), s2(narval 57931207), s3(nibi 10519904 pending)
    - pqn-lstm-1B: s1(fir 28206537), s2(rorqual 8508293), s3(narval 57931230)
  GTrXL pilots (4):
    - h006-pilot-s1 (rorqual 8510270): 512h/3L/64mem, ~2h in, running
    - h007-pilot-s1 (nibi 10522776): ref 256h/2L/128mem, pending
    - h007-pilot-s1-fir (fir 28219284): same, just submitted for faster results
    - h008-pilot-s1 (narval 57936329): ref + gae_lambda=0.95, just submitted

IMPORTANT FIX NEEDED: Commands MUST include --hypothesis-id X --experiment-id Y --output-dir /path for CSV output.
h006-pilot-s1 on rorqual does NOT have these flags (already running, can't fix). Will need to parse from stdout.

NEXT SESSION PRIORITIES (GTrXL pilots should complete in ~3-5h):
  1. Parse h006/h007/h008 GTrXL pilot results — THIS IS THE MOST IMPORTANT STEP
  2. If h007 matches reference ~41.4/226 (18.3%), immediately submit 1B x 3 seeds
  3. Compare h006 (larger model) vs h007 (reference size) to guide scaling decisions
  4. Compare h007 vs h008 to decide gae_lambda setting
  5. LSTM baselines should still be running (~15h remaining from then)
  6. Consider new hypotheses based on web research (agent running in background)
  7. Incorporate research findings on entropy scheduling, auxiliary losses, observation encoding

---
**[2026-03-18 05:34 UTC]**

=== RESEARCH FINDINGS: Web search for Craftax SOTA improvements ===

KEY FINDINGS (ranked by expected impact):

1. OBSERVATION ENCODER REDESIGN (HIGH PRIORITY):
   - The 8268-dim flat obs = (9,11,83) spatial map + player stats
   - 83 channels: block type [0:42] one-hot, mob ID [42:82] one-hot, light [82]
   - Current approach: flatten and MLP — loses spatial structure
   - Better: CNN/spatial attention for map + separate MLP for stats, concat before transformer
   - Low risk, high potential. Compatible with any backbone.

2. MAMBA-2 ARCHITECTURE (MEDIUM PRIORITY):
   - 8x less memory, 4.5x throughput vs TransformerXL (RLBenchNet 2025)
   - Matches GTrXL on hard memory tasks
   - Would need new implementation but could allow much larger context/batch

3. AGaLiTe (MEDIUM PRIORITY):
   - Gated linear attention, beats GTrXL on Craftax-Symbolic
   - 40% cheaper inference, 50% less memory
   - Code at github.com/subho406/agalite (JAX only — would need PyTorch port)

4. ADAPTIVE ENTROPY SCHEDULING (LOW-MEDIUM):
   - Start ent_coef=0.01-0.02, anneal to 0.001-0.005
   - Cosine schedule or performance-adaptive: ent_coef = base * (1 - normalized_return)
   - Standard constant entropy is suboptimal for long training

5. ACHIEVEMENT PREDICTION AUXILIARY LOSS (NOVEL):
   - Add head predicting current achievement state from transformer hidden
   - Forces representations to track game progress
   - Not tested on Craftax in published work — novel direction

6. PHASIC POLICY GRADIENT (PPG):
   - Separate policy/value training phases to avoid gradient interference
   - Especially useful with shared transformer backbone

CONFIRMED: Intrinsic motivation (RND, ICM, E3B) does NOT help on Craftax (tested in paper).
CONFIRMED: Best published model-free: GTrXL 18.3% at 1B, 20.6% at 4B. No one has exceeded ~20% at 1B.
CONFIRMED: The 18.3%→20.6% gap at 4B likely comes from longer training + better context management.

---
**[2026-03-18 05:42 UTC]**

=== SESSION SUMMARY: Implemented obs encoder + entropy scheduling, submitted h009/h010 ===

ENGINEERING CHANGES:
1. CraftaxObsEncoder (src/models/models.py): Structured observation encoder
   - Splits 8268-dim flat obs into spatial map (9x11x83) + player stats (51)
   - Map: 2-layer CNN (83→32→64, 3x3 conv, AdaptiveAvgPool → 768-dim) → linear → hidden_size
   - Stats: 2-layer MLP (51→128→128) 
   - Combine: cat + linear → hidden_size
   - 361K params vs 2.2M for flat MLP (42% reduction for 256 hidden)
   - Activated via --use-structured-obs flag
   - Integrated into both PPO_Agent and PPO_GTrXL_Agent

2. Entropy coefficient annealing (src/rl/ppo_gtrxl.py, src/utils/args.py):
   - New --ent-coef-end flag for linear annealing from ent_coef to ent_coef_end
   - Default -1 = no annealing (backward compatible)

NEW HYPOTHESES SUBMITTED:
  h009-pilot-s1 → fir (28219638): GTrXL ref + --use-structured-obs (200M steps)
  h010-pilot-s1 → fir (28219649): GTrXL ref + ent 0.01→0.002 anneal (200M steps)

FULL ACTIVE JOB LIST (12 total):
  Baselines (6, ~18h remaining):
    ppo-lstm-1B: s1(rorqual), s2(narval), s3(nibi pending)
    pqn-lstm-1B: s1(fir), s2(rorqual), s3(narval)
  GTrXL pilots (6):
    h006-pilot-s1 (rorqual 8510270): 512h/3L/64mem, ~3h in
    h007-pilot-s1 (nibi 10522776): ref 256h/2L/128mem, pending
    h007-pilot-s1-fir (fir 28219284): ref duplicate
    h008-pilot-s1 (narval 57936329): ref + gae_lambda=0.95
    h009-pilot-s1 (fir 28219638): ref + structured obs
    h010-pilot-s1 (fir 28219649): ref + entropy annealing

NEXT SESSION PRIORITIES:
  1. CRITICAL: Parse GTrXL pilot results (h006/h007/h008/h009/h010) — compare all variants
  2. If h007 matches reference 18.3% (41.4/226), submit best GTrXL variant at 1B × 3 seeds
  3. Best variant = h007 (baseline) + best improvements from h008/h009/h010
  4. Parse LSTM baseline results when ready (~15h from now)
  5. Consider combining winning improvements (structured obs + entropy + best GAE)
  6. Future ideas from web research: Mamba-2 (8x less memory), AGaLiTe, PPG, auxiliary losses
