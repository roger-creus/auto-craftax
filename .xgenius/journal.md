
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
