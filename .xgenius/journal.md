
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
