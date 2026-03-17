# Debug Log

Errors and issues encountered during autonomous research.

## 2026-03-17 00:08 — SBATCH template: path quoting + missing PYTHONPATH
The `-H {{CODE_DIR_IN_CLUSTER}}` flag in the SBATCH template caused `slurmstepd: error: couldn't chdir to '/home/rogercc/'/scratch/rogercc/auto-craftax''`. Removed `-H` flag. Also, `from src.env.env import make_craftax_env` failed with `ModuleNotFoundError: No module named 'src'` because the container didn't have PYTHONPATH set. Fixed by adding `--env PYTHONPATH=/src`.

## 2026-03-17 00:24 — Read-only filesystem for Craftax texture cache
Craftax tries to write `texture_cache.pbz2` to its install directory on first import, but the Singularity container is read-only. Fixed by adding `--writable-tmpfs` to the apptainer exec command.

## 2026-03-17 00:44 — Python stdout buffering (cosmetic)
SLURM log files appeared empty despite training running. Python buffers stdout when not connected to a terminal. Fixed by adding `--env PYTHONUNBUFFERED=1` to apptainer.

## 2026-03-17 00:08 — slurmstepd chdir warning (cosmetic, not fixed)
`slurmstepd: error: couldn't chdir to '/home/rogercc/'/scratch/rogercc/auto-craftax''` still appears as a warning even after removing `-H`. This is a SLURM startup warning, not a fatal error — the actual training runs fine. The quotes around the path are likely from how xgenius expands template variables. Not blocking.
