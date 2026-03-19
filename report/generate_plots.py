#!/usr/bin/env python3
"""Generate all plots for the Craftax research report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os

OUT = "report/plots"
os.makedirs(OUT, exist_ok=True)

# Style
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': '#f8f9fa',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})

COLORS = {
    'PPO': '#2196F3',
    'PQN': '#FF9800',
    'PPO-LSTM': '#4CAF50',
    'PQN-LSTM': '#F44336',
    'h001': '#9C27B0',
    'h002': '#795548',
    'h003': '#607D8B',
    'PPO-GTrXL (ref)': '#E91E63',
    'Max (226)': '#BDBDBD',
}


# ============================================================
# Plot 1: Baseline Comparison — Mean Episode Return (1B steps)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

algos = ['PPO', 'PQN']
means = [26.83, 20.95]
stds = [1.31, 1.04]
colors = [COLORS['PPO'], COLORS['PQN']]

bars = ax.bar(algos, means, yerr=stds, capsize=8, color=colors, edgecolor='white', linewidth=1.5, width=0.5)
ax.set_ylabel('Mean Episode Return')
ax.set_title('Baseline Performance at 1B Steps\n(3 seeds, Craftax-Symbolic-v1)')
ax.set_ylim(0, 50)

# Annotate bars
for bar, mean, std in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 1,
            f'{mean:.1f} ± {std:.1f}', ha='center', va='bottom', fontweight='bold')

# Reference line for max score
ax.axhline(y=226, color='gray', linestyle='--', alpha=0.3, label='Max possible (226)')
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{OUT}/baseline_comparison.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 2: Per-seed Baseline Results
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

seeds = [1, 2, 3]
ppo_returns = [28.06, 26.98, 25.46]
pqn_returns = [21.54, 19.74, 21.58]

x = np.arange(len(seeds))
w = 0.3
bars1 = ax.bar(x - w/2, ppo_returns, w, label='PPO', color=COLORS['PPO'], edgecolor='white')
bars2 = ax.bar(x + w/2, pqn_returns, w, label='PQN', color=COLORS['PQN'], edgecolor='white')

ax.set_xlabel('Seed')
ax.set_ylabel('Episode Return')
ax.set_title('Per-Seed Baseline Results (1B Steps)')
ax.set_xticks(x)
ax.set_xticklabels(['Seed 1', 'Seed 2', 'Seed 3'])
ax.legend()
ax.set_ylim(0, 40)

for bars in [bars1, bars2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig(f"{OUT}/per_seed_baselines.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 3: Achievement Breakdown for Baselines
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

achievements = ['Enter\nDungeon', 'Make Iron\nSword', 'Make Diamond\nSword', 'Collect\nDiamond', 'Find\nBow']
ppo_achs = [
    np.mean([56, 56, 48]),  # enter_dungeon
    np.mean([4, 8, 0]),     # make_iron_sword
    np.mean([4, 0, 0]),     # make_diamond_sword
    np.mean([8, 12, 4]),    # collect_diamond
    np.mean([56, 52, 44]),  # find_bow
]
pqn_achs = [
    np.mean([44, 0, 0]),    # enter_dungeon
    np.mean([4, 0, 80]),    # make_iron_sword (pqn s3 has 80% on this column — actually that's collect_diamond)
    np.mean([4, 0, 0]),     # make_diamond_sword
    np.mean([0, 0, 0]),     # collect_diamond
    np.mean([32, 0, 0]),    # find_bow
]

# Rechecking PQN data from CSV:
# pqn-1B-s1: enter_dungeon=44, iron_sword=4, diamond_sword=4, collect_diamond=0, find_bow=32
# pqn-1B-s2: enter_dungeon=0, everything 0
# pqn-1B-s3: enter_dungeon=0, collect_diamond=80 (wait, the CSV shows make_iron_sword column)
# Let me re-read: s3 line: 80.0,0.0,0.0,0.0,0.0,0.0,0.0
# Header order: enter_dungeon,enter_gnomish_mines,...,make_iron_sword,make_diamond_sword,collect_diamond,find_bow,...
# s3: ...0.0,0.0,0.0,0.0,0.0,0.0,0.0,80.0,0.0,0.0,0.0,0.0,0.0,0.0
# That's collect_diamond=80? No wait. Let me re-read the header carefully.
# enter_dungeon,enter_gnomish_mines,enter_sewers,enter_vault,enter_troll_mines,enter_fire_realm,enter_ice_realm,enter_graveyard,make_iron_sword,make_diamond_sword,collect_diamond,find_bow,enchant_sword,enchant_armour,defeat_necromancer
# pqn-1B-s3: 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,80.0,0.0,0.0,0.0,0.0,0.0,0.0
# So make_iron_sword=80% for PQN s3
pqn_achs = [
    np.mean([44, 0, 0]),    # enter_dungeon
    np.mean([4, 0, 80]),    # make_iron_sword
    np.mean([4, 0, 0]),     # make_diamond_sword
    np.mean([0, 0, 0]),     # collect_diamond
    np.mean([32, 0, 0]),    # find_bow
]

x = np.arange(len(achievements))
w = 0.3
bars1 = ax.bar(x - w/2, ppo_achs, w, label='PPO', color=COLORS['PPO'], edgecolor='white')
bars2 = ax.bar(x + w/2, pqn_achs, w, label='PQN', color=COLORS['PQN'], edgecolor='white')

ax.set_ylabel('Achievement Rate (%)')
ax.set_title('Key Achievement Rates at 1B Steps (Mean over 3 Seeds)')
ax.set_xticks(x)
ax.set_xticklabels(achievements)
ax.legend()
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig(f"{OUT}/achievement_breakdown.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 4: Hypothesis Pilot Comparison (200M steps)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

# PPO baseline at 200M ~= extrapolated. But we have pilot at 10M = 9.1 for PPO.
# Let's compare the 200M pilot results directly
pilots = ['PPO\n(10M baseline)', 'h001\n(γ=0.999)', 'h002\n(reward\nshaping)', 'h003\n(h001+h002)']
pilot_returns = [9.1, 16.46, 6.42, 9.06]
pilot_steps = ['10M', '200M', '200M', '200M']
pilot_colors = ['#2196F3', '#9C27B0', '#795548', '#607D8B']

bars = ax.bar(pilots, pilot_returns, color=pilot_colors, edgecolor='white', linewidth=1.5, width=0.5)
for bar, ret, steps in zip(bars, pilot_returns, pilot_steps):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{ret:.1f}\n({steps})', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_ylabel('Mean Episode Return')
ax.set_title('Hypothesis Pilot Results\n(Early-stage comparison)')
ax.set_ylim(0, 25)

# Add horizontal line for PPO 1B baseline
ax.axhline(y=26.83, color=COLORS['PPO'], linestyle='--', alpha=0.5, label='PPO @ 1B steps (26.83)')
ax.legend()

plt.tight_layout()
plt.savefig(f"{OUT}/hypothesis_pilots.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 5: Performance Relative to Max Score
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

algos = ['PPO\n(1B)', 'PQN\n(1B)', 'PPO-GTrXL\n(published)', 'SCALAR\n(published)']
scores = [26.83, 20.95, 41.4, 199.3]  # SCALAR gets 88.2% of 226
max_score = 226
pcts = [s/max_score*100 for s in scores]
colors_bars = [COLORS['PPO'], COLORS['PQN'], COLORS['PPO-GTrXL (ref)'], '#4CAF50']

bars = ax.barh(algos, pcts, color=colors_bars, edgecolor='white', linewidth=1.5, height=0.5)
for bar, pct, score in zip(bars, pcts, scores):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{pct:.1f}% ({score:.1f}/226)', va='center', fontsize=10)

ax.set_xlabel('% of Maximum Score (226)')
ax.set_title('Performance Relative to Maximum Score\n(Our Results vs Published Benchmarks)')
ax.set_xlim(0, 110)
ax.axvline(x=100, color='gray', linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/performance_vs_max.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 6: Job Status Distribution
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Pie chart of job statuses
statuses = ['Completed', 'Running', 'Pending', 'Cancelled', 'Disappeared']
counts = [4, 7, 3, 11, 28]
colors_pie = ['#4CAF50', '#2196F3', '#FFC107', '#FF9800', '#F44336']
explode = (0.05, 0.05, 0.05, 0.05, 0.1)

ax1.pie(counts, labels=statuses, colors=colors_pie, explode=explode,
        autopct='%1.0f%%', startangle=90, textprops={'fontsize': 11})
ax1.set_title('Job Status Distribution\n(53 total jobs)')

# Bar chart of jobs per cluster
clusters = ['rorqual', 'narval', 'nibi', 'fir']
cluster_counts = [13, 14, 13, 13]  # approximate from the data
cluster_colors = ['#42A5F5', '#66BB6A', '#FFA726', '#EF5350']

ax2.bar(clusters, cluster_counts, color=cluster_colors, edgecolor='white', linewidth=1.5)
ax2.set_ylabel('Number of Jobs')
ax2.set_title('Jobs per Cluster')

plt.tight_layout()
plt.savefig(f"{OUT}/job_statistics.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 7: Training Wall Time Comparison (1B steps)
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

algos_wt = ['PPO s1', 'PPO s2', 'PPO s3', 'PQN s1', 'PQN s2', 'PQN s3']
wall_times_h = [27116.97/3600, 28648.97/3600, 28028.80/3600,
                20287.99/3600, 20113.81/3600, 20082.70/3600]
colors_wt = [COLORS['PPO']]*3 + [COLORS['PQN']]*3

bars = ax.bar(algos_wt, wall_times_h, color=colors_wt, edgecolor='white', linewidth=1.5, width=0.6)
for bar, wt in zip(bars, wall_times_h):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f'{wt:.1f}h', ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Wall Time (hours)')
ax.set_title('Training Wall Time for 1B Steps')
ax.set_ylim(0, 10)

plt.tight_layout()
plt.savefig(f"{OUT}/wall_times.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 8: Research Timeline
# ============================================================
fig, ax = plt.subplots(figsize=(14, 6))

events = [
    ('Mar 17\n03:35', 'Project start\nPhase 0', '#2196F3'),
    ('Mar 17\n04:08', 'Container built\nPilots submitted', '#2196F3'),
    ('Mar 17\n04:47', 'Bug fixes\nv3 pilots running', '#FF9800'),
    ('Mar 17\n05:14', 'Phase 0 done\n12 baseline jobs', '#4CAF50'),
    ('Mar 18\n00:47', 'PPO/PQN results\nh001-h005 submitted', '#9C27B0'),
    ('Mar 18\n01:10', 'PQN-LSTM OOM\n48G resubmit', '#F44336'),
    ('Mar 18\n03:14', 'GTrXL implemented\nh006-h008 submitted', '#E91E63'),
]

for i, (time, event, color) in enumerate(events):
    ax.plot(i, 0, 'o', color=color, markersize=15, zorder=5)
    offset = 0.3 if i % 2 == 0 else -0.3
    ax.annotate(event, (i, 0), xytext=(0, 40 if i % 2 == 0 else -55),
                textcoords='offset points', ha='center', va='center',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.15),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))
    ax.text(i, -0.08 if i % 2 == 0 else 0.08, time, ha='center', va='center', fontsize=8, color='gray')

ax.plot(range(len(events)), [0]*len(events), '-', color='gray', alpha=0.3, linewidth=2)
ax.set_xlim(-0.5, len(events) - 0.5)
ax.set_ylim(-0.5, 0.5)
ax.axis('off')
ax.set_title('Research Campaign Timeline', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(f"{OUT}/timeline.png", bbox_inches='tight')
plt.close()


# ============================================================
# Plot 9: Hypothesis Status Summary
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

hyp_ids = ['h001', 'h002', 'h003', 'h004', 'h005', 'h006', 'h007', 'h008']
hyp_labels = [
    'h001: Better\nhyperparams',
    'h002: Reward\nshaping',
    'h003: h001+h002\ncombined',
    'h004: Wider\narchitecture',
    'h005: Kitchen\nsink',
    'h006: GTrXL\n512h/3L',
    'h007: GTrXL\nreference',
    'h008: GTrXL\n+GAE 0.95',
]
statuses = ['open', 'closed', 'closed', 'proposed', 'closed', 'proposed', 'proposed', 'proposed']
status_colors = {
    'open': '#FFC107',
    'closed': '#F44336',
    'proposed': '#2196F3',
    'promising': '#4CAF50',
}

bars = ax.barh(hyp_labels, [1]*len(hyp_ids),
               color=[status_colors[s] for s in statuses],
               edgecolor='white', linewidth=1.5, height=0.6)

# Annotate with pilot returns where available
pilot_results = {
    'h001': '16.46',
    'h002': '6.42',
    'h003': '9.06',
    'h004': 'Running',
    'h005': 'Deprioritized',
    'h006': 'Running',
    'h007': 'Completed',
    'h008': 'Pending',
}
for bar, hid in zip(bars, hyp_ids):
    ax.text(0.5, bar.get_y() + bar.get_height()/2,
            f'{pilot_results[hid]}', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')

ax.set_xlim(0, 1)
ax.set_xticks([])
ax.set_title('Hypothesis Status Overview')

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2196F3', label='Proposed/Running'),
    Patch(facecolor='#FFC107', label='Open'),
    Patch(facecolor='#F44336', label='Closed/Failed'),
    Patch(facecolor='#4CAF50', label='Promising'),
]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.savefig(f"{OUT}/hypothesis_status.png", bbox_inches='tight')
plt.close()


print("All plots generated successfully!")
print(f"Files in {OUT}/:")
for f in sorted(os.listdir(OUT)):
    print(f"  {f}")
