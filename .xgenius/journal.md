
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
