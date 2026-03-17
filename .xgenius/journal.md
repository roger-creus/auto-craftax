
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
