"""
jitter_diagnostic.py — show the calibrated jitter multiplier vs alpha, per filter and
noise mode, so spikes and grid-boundary clamps are visible. Reads the variable-jitter
data/stage2_results_w{W}_{mode}_variable.npz files (fixed jitter has nothing to diagnose).
A pick sitting on the
grid floor or ceiling is a boundary hit: the true optimum is outside the scanned range
and the value is untrustworthy. Produces figs/jitter_diagnostic.png.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.8  Changed: read variable-jitter results only (fixed jitter has nothing to diagnose)
# 2.7  Changed: read window from cfg, tag input/output by window (w{obs_every})
# 2.5  Changed: GRID ceiling 12 -> 30 to match stage2.py (geomspace 0.3..30, 9 pts)
# 2.2  created — calibrated jitter vs alpha, with boundary-hit flags
# ============================================================

# must match the grid used in stage2.py
GRID = np.geomspace(0.3, 30, 9)
FLOOR, CEIL = GRID[0], GRID[-1]
W = cfg.obs_every                                    # window set by run.py (L63_OBS_EVERY)

os.makedirs('figs', exist_ok=True)
modes = [m for m in ('fixed_R', 'fixed_snr') if os.path.exists(f'data/stage2_results_w{W}_{m}_variable.npz')]
if not modes:
    raise SystemExit(f'no data/stage2_results_w{W}_*_variable.npz found — run stage2.py (variable) first')

EN, PF = '#c0392b', '#2471a3'
fig, axes = plt.subplots(1, len(modes), figsize=(6.2 * len(modes), 4.4), squeeze=False)

for ax, mode in zip(axes[0], modes):
    d = np.load(f'data/stage2_results_w{W}_{mode}_variable.npz')
    a, en, pf = d['alphas'], d['en_mult'], d['pf_mult']

    for g in GRID:                                   # faint lines at each scannable jitter level
        ax.axhline(g, color='#bbb', lw=0.5, zorder=0)
    ax.axhline(FLOOR, color='#888', ls='--', lw=1)
    ax.axhline(CEIL, color='#888', ls='--', lw=1)
    ax.text(a[0], CEIL * 1.04, 'grid ceiling — boundary hits untrustworthy', fontsize=7.5, color='#888')
    ax.text(a[0], FLOOR * 0.78, 'grid floor', fontsize=7.5, color='#888')

    ax.plot(a, en, 'o-', color=EN, lw=2, label='EnKF')
    ax.plot(a, pf, 's-', color=PF, lw=2, label='PF')

    for series, col in ((en, EN), (pf, PF)):          # ring the boundary hits
        hit = np.isclose(series, FLOOR) | np.isclose(series, CEIL)
        ax.scatter(a[hit], series[hit], s=130, facecolors='none', edgecolors=col, linewidths=2, zorder=5)

    ax.set_yscale('log'); ax.set_yticks(GRID)
    ax.set_yticklabels([f'{g:.2g}' for g in GRID])
    ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('calibrated jitter (x base perturb_std)')
    ax.set_title(f'{mode}'); ax.legend(loc='center left')

fig.suptitle(f'Calibrated jitter vs alpha — window {W} (ringed = grid-boundary hit)', y=1.02)
fig.tight_layout()
fig.savefig(f'figs/jitter_diagnostic_w{W}.png', dpi=145, bbox_inches='tight')
print(f'saved figs/jitter_diagnostic_w{W}.png')
