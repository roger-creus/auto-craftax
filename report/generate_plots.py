#!/usr/bin/env python3
"""Generate all plots for the Craftax research report."""
import csv
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
})

PLOT_DIR = '/home/roger/Desktop/auto-craftax/report/plots'
EXP_CSV = '/home/roger/Desktop/auto-craftax/results/experiments.csv'

# Read experiments
experiments = []
with open(EXP_CSV) as f:
    reader = csv.DictReader(f)
    for row in reader:
        experiments.append(row)

def safe_float(v, default=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return default

# ===================== PLOT 1: Baseline Comparison =====================
baseline_data = {}
for e in experiments:
    if e['hypothesis_id'] == 'baseline' and '1B' in e['experiment_id']:
        algo = e['algorithm']
        ret = safe_float(e['avg_episode_return'])
        if algo not in baseline_data:
            baseline_data[algo] = []
        baseline_data[algo].append(ret)

algos = ['PPO', 'PPO-LSTM', 'PQN', 'PQN-LSTM']
means = []
stds = []
for a in algos:
    vals = baseline_data.get(a, [0])
    means.append(np.mean(vals))
    stds.append(np.std(vals))

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']
bars = ax.bar(algos, means, yerr=stds, capsize=5, color=colors, edgecolor='black', linewidth=0.5)
ax.set_ylabel('Average Episode Return')
ax.set_title('Baseline Algorithm Comparison (1B Steps, 3 Seeds)')
for bar, m in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
            f'{m:.1f}', ha='center', va='bottom', fontweight='bold')
ax.set_ylim(0, max(means) * 1.25)
ax.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'baselines.png'))
plt.close()

# ===================== PLOT 2: Top Configurations at 1B =====================
h_1b = {}
for e in experiments:
    hid = e['hypothesis_id']
    ret = safe_float(e['avg_episode_return'])
    ts = safe_float(e['total_timesteps'])
    if ts >= 900000000 and ret > 0:
        if hid not in h_1b:
            h_1b[hid] = []
        h_1b[hid].append(ret)

hyp_labels = {
    'h007': 'GTrXL ref (h007)',
    'h009': 'GTrXL+StructObs (h009)',
    'h012': 'GTrXL+PopArt (h012)',
    'h021': 'LSTM+StructObs (h021)',
    'h023': 'LSTM+128steps (h023)',
    'h031': 'GRU+64steps (h031)',
    'h032': 'LSTM+EntAnneal (h032)',
    'h040': 'GRU+128+Grad1.0 (h040)',
    'h043': 'LSTM+128+Ent+Grad (h043)',
    'h044': 'GRU+128+Ent+Grad (h044)',
    'h069': 'GRU+Curriculum (h069)',
}

ppo_lstm_vals = [safe_float(e['avg_episode_return']) for e in experiments
                 if e['hypothesis_id'] == 'baseline' and '1B' in e['experiment_id'] and e['algorithm'] == 'PPO-LSTM']

top_configs = [('baseline_lstm', 'PPO-LSTM baseline', np.mean(ppo_lstm_vals), np.std(ppo_lstm_vals), len(ppo_lstm_vals))]
for hid, vals in sorted(h_1b.items()):
    if len(vals) >= 2 and hid in hyp_labels:
        top_configs.append((hid, hyp_labels[hid], np.mean(vals), np.std(vals), len(vals)))

top_configs.sort(key=lambda x: x[2])

fig, ax = plt.subplots(figsize=(12, 7))
labels = [x[1] for x in top_configs]
vals = [x[2] for x in top_configs]
errs = [x[3] for x in top_configs]
colors_bar = ['#DD8452' if 'baseline' in x[0] else ('#2ecc71' if x[2] > 38 else '#4C72B0') for x in top_configs]
bars = ax.barh(labels, vals, xerr=errs, capsize=3, color=colors_bar, edgecolor='black', linewidth=0.5)
ax.set_xlabel('Average Episode Return (1B Steps)')
ax.set_title('All Multi-Seed Configurations at 1B Steps')
for bar, v in zip(bars, vals):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2., f'{v:.1f}', va='center', fontweight='bold', fontsize=10)
ax.axvline(x=np.mean(ppo_lstm_vals), color='red', linestyle='--', alpha=0.7, label=f'PPO-LSTM baseline ({np.mean(ppo_lstm_vals):.1f})')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'top_configs_1b.png'))
plt.close()

# ===================== PLOT 3: Pilot Results Top 30 =====================
pilots = []
for e in experiments:
    hid = e['hypothesis_id']
    ts = safe_float(e['total_timesteps'])
    ret = safe_float(e['avg_episode_return'])
    if 100000000 <= ts <= 300000000 and ret > 0 and 'pilot' in e['experiment_id']:
        pilots.append((hid, ret, safe_float(e.get('enter_dungeon', 0))))

best_pilots = {}
for hid, ret, dung in pilots:
    if hid not in best_pilots or ret > best_pilots[hid][1]:
        best_pilots[hid] = (hid, ret, dung)

pilot_list = sorted(best_pilots.values(), key=lambda x: x[1], reverse=True)[:30]

fig, ax = plt.subplots(figsize=(14, 7))
hids = [p[0] for p in pilot_list]
rets = [p[1] for p in pilot_list]
colors_p = ['#2ecc71' if r > 30 else ('#e74c3c' if r < 20 else '#f39c12') for r in rets]
bars = ax.bar(range(len(hids)), rets, color=colors_p, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(hids)))
ax.set_xticklabels(hids, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Episode Return')
ax.set_title('Top 30 Pilot Results (200M Steps, Best Per Hypothesis)')
ax.axhline(y=30.86, color='blue', linestyle='--', alpha=0.7, label='h040 pilot (30.86)')
ax.axhline(y=33.54, color='green', linestyle='--', alpha=0.7, label='h096 pilot (33.54) BEST')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'pilot_results.png'))
plt.close()

# ===================== PLOT 4: Exploration Methods =====================
exploration = {
    'No exploration\n(h040)': 30.86,
    'RND 0.005\n(h096)': 33.54,
    'RND 0.007\n(h107)': 31.86,
    'RND 0.01\n(h085)': 30.86,
    'RND 0.1\n(h086)': 21.98,
    'RLE 0.01\n(h092)': 18.94,
    'RLE 0.1\n(h093)': 4.94,
    'SIL 0.1\n(h101)': 30.7,
    'SIL+RND\n(h102)': 23.58,
    'RND 0->0.02\n(h105)': 27.74,
    'RND 0.05->0.005\n(h098)': 27.54,
    'Dungeon RND\n(h100)': 25.3,
}

fig, ax = plt.subplots(figsize=(14, 6))
sorted_exp = sorted(exploration.items(), key=lambda x: x[1], reverse=True)
labels_e = [e[0] for e in sorted_exp]
vals_e = [e[1] for e in sorted_exp]
colors_e = ['#2ecc71' if v > 30 else ('#e74c3c' if v < 20 else '#f39c12') for v in vals_e]
bars = ax.barh(range(len(labels_e)), vals_e, color=colors_e, edgecolor='black', linewidth=0.5)
ax.set_yticks(range(len(labels_e)))
ax.set_yticklabels(labels_e, fontsize=9)
ax.set_xlabel('Average Episode Return (200M Pilot)')
ax.set_title('Exploration Methods Comparison (all on h040 base config)')
ax.axvline(x=30.86, color='blue', linestyle='--', alpha=0.5, label='h040 no exploration (30.86)')
ax.legend()
ax.grid(axis='x', alpha=0.3)
for bar, v in zip(bars, vals_e):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2.,
            f'{v:.1f}', va='center', fontsize=9, fontweight='bold')
plt.savefig(os.path.join(PLOT_DIR, 'exploration_methods.png'))
plt.close()

# ===================== PLOT 5: Performance Progression =====================
milestones = [
    ('PPO\n(baseline)', 26.83),
    ('PQN-LSTM\n(baseline)', 24.23),
    ('PPO-LSTM\n(baseline)', 33.88),
    ('GTrXL\n(h007)', 19.01),
    ('LSTM+Struct\n(h021)', 32.59),
    ('LSTM+128\n(h023)', 38.10),
    ('GRU+64\n(h031)', 32.90),
    ('GRU+128+Grad\n(h040)', 39.55),
    ('GRU+Ent+128\n(h044)', 39.0),
    ('RND 0.005*\n(h096)', 33.54),
]

fig, ax = plt.subplots(figsize=(13, 6))
x = range(len(milestones))
labels_m = [m[0] for m in milestones]
vals_m = [m[1] for m in milestones]
colors_m = ['#2ecc71' if v >= 38 else ('#4C72B0' if v >= 30 else '#e74c3c') for v in vals_m]
bars = ax.bar(x, vals_m, color=colors_m, edgecolor='black', linewidth=0.5)
for i, (l, v) in enumerate(milestones):
    ax.text(i, v + 0.4, f'{v:.1f}', ha='center', fontweight='bold', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(labels_m, rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Average Episode Return')
ax.set_title('Key Configurations: Performance at 1B Steps (or *200M Pilot)')
ax.axhline(y=33.88, color='red', linestyle='--', alpha=0.5, label='Best baseline (33.88)')
ax.axhline(y=44.04, color='green', linestyle='--', alpha=0.5, label='+30% target (44.04)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
ax.set_ylim(0, 50)
plt.savefig(os.path.join(PLOT_DIR, 'performance_progression.png'))
plt.close()

# ===================== PLOT 6: Achievement Comparison =====================
achievements = ['enter_dungeon', 'enter_gnomish_mines', 'make_iron_sword', 'make_diamond_sword',
                'collect_diamond', 'find_bow', 'enchant_sword', 'enchant_armour']
ach_labels = ['Dungeon', 'Gnomish\nMines', 'Iron\nSword', 'Diamond\nSword',
              'Collect\nDiamond', 'Find\nBow', 'Enchant\nSword', 'Enchant\nArmour']

def get_mean_ach(hyp_id, algo_filter=None):
    ach_vals = {a: [] for a in achievements}
    for e in experiments:
        if e['hypothesis_id'] == hyp_id and '1B' in e['experiment_id']:
            if algo_filter and e.get('algorithm', '') != algo_filter:
                continue
            ts = safe_float(e['total_timesteps'])
            if ts >= 900000000:
                for a in achievements:
                    ach_vals[a].append(safe_float(e.get(a, 0)))
    return [np.mean(ach_vals[a]) if ach_vals[a] else 0 for a in achievements]

baseline_ach = get_mean_ach('baseline', 'PPO-LSTM')
h040_ach = get_mean_ach('h040')

fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(ach_labels))
w = 0.35
b1 = ax.bar(x - w/2, baseline_ach, w, label='PPO-LSTM Baseline', color='#4C72B0', edgecolor='black', linewidth=0.5)
b2 = ax.bar(x + w/2, h040_ach, w, label='h040 (Best Config)', color='#2ecc71', edgecolor='black', linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(ach_labels, fontsize=10)
ax.set_ylabel('Achievement Rate (%)')
ax.set_title('Achievement Rates: PPO-LSTM Baseline vs Best Config (h040, 1B Steps)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar in list(b1) + list(b2):
    if bar.get_height() > 0:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                f'{bar.get_height():.0f}%', ha='center', va='bottom', fontsize=8)
plt.savefig(os.path.join(PLOT_DIR, 'achievements.png'))
plt.close()

# ===================== PLOT 7: Hypothesis Outcomes Pie =====================
from collections import Counter
hyp_status = Counter()
with open('/home/roger/Desktop/auto-craftax/results/hypotheses.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        status = row['status'].strip()
        hyp_status[status] += 1

fig, ax = plt.subplots(figsize=(7, 7))
labels_s = list(hyp_status.keys())
sizes = list(hyp_status.values())
colors_s = {'closed': '#e74c3c', 'promising': '#2ecc71', 'open': '#f39c12', 'proposed': '#3498db'}
c = [colors_s.get(l, '#95a5a6') for l in labels_s]
total_h = sum(sizes)
wedges, texts, autotexts = ax.pie(sizes, labels=[f'{l}\n({s})' for l, s in zip(labels_s, sizes)],
                                   autopct='%1.0f%%', colors=c, startangle=90,
                                   textprops={'fontsize': 11})
ax.set_title(f'Hypothesis Outcome Distribution ({total_h} Total)')
plt.savefig(os.path.join(PLOT_DIR, 'hypothesis_outcomes.png'))
plt.close()

# ===================== PLOT 8: Architecture Comparison =====================
arch_search = {
    'PPO (MLP)': 26.83,
    'PPO-LSTM (64 steps)': 33.88,
    'PQN (MLP)': 20.95,
    'PQN-LSTM (64 steps)': 24.23,
    'GTrXL (h007)': 19.01,
    'GTrXL+PopArt (h012)': 19.78,
    'LSTM+StructObs (h021)': 32.59,
    'GRU+64 (h031)': 32.90,
    'GRU+128+Grad (h040)': 39.55,
    'LSTM+128 (h023)': 38.10,
}

fig, ax = plt.subplots(figsize=(12, 6))
sorted_arch = sorted(arch_search.items(), key=lambda x: x[1])
labels_a = [a[0] for a in sorted_arch]
vals_a = [a[1] for a in sorted_arch]
colors_a = ['#e74c3c' if v < 25 else ('#f39c12' if v < 35 else '#2ecc71') for v in vals_a]
bars = ax.barh(labels_a, vals_a, color=colors_a, edgecolor='black', linewidth=0.5)
for bar, v in zip(bars, vals_a):
    ax.text(v + 0.3, bar.get_y() + bar.get_height()/2., f'{v:.1f}', va='center', fontsize=10, fontweight='bold')
ax.set_xlabel('Average Episode Return (1B Steps)')
ax.set_title('Architecture Comparison at 1B Steps')
ax.axvline(x=33.88, color='red', linestyle='--', alpha=0.5, label='PPO-LSTM baseline (33.88)')
ax.legend()
ax.grid(axis='x', alpha=0.3)
plt.savefig(os.path.join(PLOT_DIR, 'architecture_comparison.png'))
plt.close()

# ===================== PLOT 9: RND Coefficient Sweep =====================
rnd_sweep = [
    (0.0, 30.86, 'h040 (none)'),
    (0.005, 33.54, 'h096'),
    (0.007, 31.86, 'h107'),
    (0.01, 30.86, 'h085'),
    (0.1, 21.98, 'h086'),
]

fig, ax = plt.subplots(figsize=(9, 5))
x_rnd = [r[0] for r in rnd_sweep]
y_rnd = [r[1] for r in rnd_sweep]
lab_rnd = [r[2] for r in rnd_sweep]
ax.plot(x_rnd, y_rnd, 'o-', color='#2C3E50', linewidth=2, markersize=12, markerfacecolor='#E74C3C')
for xi, yi, li in zip(x_rnd, y_rnd, lab_rnd):
    ax.annotate(f'{yi:.1f}\n({li})', (xi, yi), textcoords="offset points", xytext=(0, 15),
                ha='center', fontweight='bold', fontsize=9)
ax.set_xlabel('RND Coefficient')
ax.set_ylabel('Average Episode Return (200M Pilot)')
ax.set_title('RND Intrinsic Reward Coefficient Sweep')
ax.grid(alpha=0.3)
ax.set_ylim(18, 38)
plt.savefig(os.path.join(PLOT_DIR, 'rnd_sweep.png'))
plt.close()

# ===================== PLOT 10: Failed Approaches =====================
failed = [
    ('h056\nKill Bonus', 2.26),
    ('h058\nKill+Obs', 4.46),
    ('h093\nRLE 0.1', 4.94),
    ('h002\nReward Shape', 6.42),
    ('h003\nGamma+Shape', 9.06),
    ('h079\nVC-PPO+Split', 17.78),
    ('h055\nGoExplore+PBRS', 17.53),
    ('h081\nEpochs=2', 18.50),
    ('h035\nHidden=768', 18.30),
    ('h092\nRLE 0.01', 18.94),
]

fig, ax = plt.subplots(figsize=(12, 5))
sorted_failed = sorted(failed, key=lambda x: x[1])
labels_f = [f[0] for f in sorted_failed]
vals_f = [f[1] for f in sorted_failed]
colors_f = ['#c0392b' if v < 10 else '#e67e22' for v in vals_f]
bars = ax.bar(range(len(labels_f)), vals_f, color=colors_f, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(labels_f)))
ax.set_xticklabels(labels_f, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Average Episode Return (200M Pilot)')
ax.set_title('Worst-Performing Approaches')
ax.axhline(y=30.86, color='blue', linestyle='--', alpha=0.5, label='h040 baseline pilot (30.86)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for bar, v in zip(bars, vals_f):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
            f'{v:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.savefig(os.path.join(PLOT_DIR, 'failed_approaches.png'))
plt.close()

print("All plots generated successfully!")
for f in sorted(os.listdir(PLOT_DIR)):
    print(f"  {f}")
