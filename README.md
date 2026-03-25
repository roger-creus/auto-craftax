# auto-craftax

Autonomous research on [Craftax-Symbolic-v1](https://github.com/MichaelTMatthews/Craftax), driven by [xgenius](https://github.com/roger-creus/xgenius).

An AI agent (Claude Code) autonomously formulated hypotheses, wrote code, ran 1B-step experiments on SLURM clusters, and iterated — with no human intervention.

## Key files

- [`research_goal.md`](research_goal.md) — the research objective given to the agent
- [`.xgenius/journal.md`](.xgenius/journal.md) — the agent's full research journal
- [`report/report.html`](report/report.html) — auto-generated research report
- [`results/`](results/) — raw experiment results (CSV)
- [`xgenius.toml`](xgenius.toml) — cluster and safety configuration
