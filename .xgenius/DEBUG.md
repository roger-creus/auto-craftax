# Debug Log

Errors and issues encountered during autonomous research.

## 2026-03-17 00:08 — SBATCH template: path quoting + missing PYTHONPATH
The `-H {{CODE_DIR_IN_CLUSTER}}` flag in the SBATCH template caused `slurmstepd: error: couldn't chdir to '/home/rogercc/'/scratch/rogercc/auto-craftax''`. Removed `-H` flag. Also, `from src.env.env import make_craftax_env` failed with `ModuleNotFoundError: No module named 'src'` because the container didn't have PYTHONPATH set. Fixed by adding `--env PYTHONPATH=/src`.

## 2026-03-17 00:24 — Read-only filesystem for Craftax texture cache
Craftax tries to write `texture_cache.pbz2` to its install directory on first import, but the Singularity container is read-only. Fixed by adding `--writable-tmpfs` to the apptainer exec command.

## 2026-03-17 00:44 — Python stdout buffering (cosmetic)
SLURM log files appeared empty despite training running. Python buffers stdout when not connected to a terminal. Fixed by adding `--env PYTHONUNBUFFERED=1` to apptainer.

## 2026-03-18 01:10 — PQN-LSTM 1B jobs OOM killed on fir (32G memory)
Jobs 28093388 (s2) and 28093389 (s3) ran for ~660M/1B steps (~12h) before being OOM killed with 32G memory.
SLURM reported exit_code=0 despite OOM kill — the actual error is in the log: `slurmstepd: error: Detected 1 oom_kill event`.
LSTM models apparently need more than 32G for 1B-step training. Previous session already resubmitted both with 48G memory:
- s2 → job 10514355 on nibi (48G)
- s3 → job 28185700 on fir (48G)
Partial data: s2 had avg_reward ~25 at 660M, s3 had avg_reward ~20 at 659M.

## 2026-03-17 00:08 — slurmstepd chdir warning (cosmetic, not fixed)
`slurmstepd: error: couldn't chdir to '/home/rogercc/'/scratch/rogercc/auto-craftax''` still appears as a warning even after removing `-H`. This is a SLURM startup warning, not a fatal error — the actual training runs fine. The quotes around the path are likely from how xgenius expands template variables. Not blocking.
