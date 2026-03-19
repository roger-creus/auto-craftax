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

## 2026-03-17 23:22 — GTrXL get_states_train crash: done tensor dtype
All GTrXL experiments (h006, h007, h008) crashed at the first PPO update with:
`RuntimeError: where expected condition to be a boolean tensor, but got a tensor with dtype Float`
in `_compute_positions` at `gtrxl.py:258`. Root cause: `dones` storage tensor is float (line 105 of ppo_gtrxl.py)
but `torch.where` expects bool. The rollout path (`get_states`) converts to float explicitly (line 182),
so it works fine. The training path (`get_states_train`) passed raw float dones to `_compute_positions`.
Fix: added `.bool()` cast in `_compute_positions`. Cancelled and resubmitted all 3 GTrXL pilots.

## 2026-03-18 14:05 — PPO_Args missing use_popart field (h020/h021 crash)
h020-pilot-s1 (fir, 28244784) and h021-pilot-s1 (rorqual, 8540190) both crashed immediately with:
`AttributeError: 'PPO_Args' object has no attribute 'use_popart'`
Root cause: `use_popart` was only defined in `GTrXL_Args` (subclass), not `PPO_Args` (base class).
The ppo_lstm.py code at line 80 references `args.use_popart` but ppo_lstm uses `PPO_Args`.
Fix: moved `use_popart: bool = False` from `GTrXL_Args` to `PPO_Args`. Committed as 4ca339b.
Both jobs resubmitted: h020→nibi (10532814), h021→rorqual (8541287).

## 2026-03-17 00:08 — slurmstepd chdir warning (cosmetic, not fixed)
`slurmstepd: error: couldn't chdir to '/home/rogercc/'/scratch/rogercc/auto-craftax''` still appears as a warning even after removing `-H`. This is a SLURM startup warning, not a fatal error — the actual training runs fine. The quotes around the path are likely from how xgenius expands template variables. Not blocking.

## 2026-03-19 08:15 — Go-Explore and PBRS completely non-functional due to Craftax achievement info bug
**Root cause:** Craftax's `log_achievements_to_info()` in `common.py` does `achievements = state.achievements * done * 100.0`, which means `infos['Achievements/...']` is always 0.0 for non-done environments. Both Go-Explore (milestone detection) and PBRS (potential computation) read from infos, so:
- Go-Explore: buffer=0, saves=0, resets=0 throughout entire training (h051 pilot was a no-op)
- PBRS: bonus always 0, no reward shaping effect (h054, h055 pilots were no-ops)

**Fix:** Read achievements directly from JAX env state (`env_state.achievements`) which has the true per-step boolean values. Added `get_achievements_from_state()` helper. Cancelled broken h054/h055 pilots, resubmitted h051/h054/h055 with fix.

**Impact:** All previous h051 pilot results were invalid (Go-Explore not working). h054/h055 were also invalid. This is the first time these mechanisms will actually function.
