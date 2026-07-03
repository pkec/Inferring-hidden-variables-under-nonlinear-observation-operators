"""
calibration_ratio.py — spread/RMSE ratio vs alpha for one window, across noise mode and
jitter strategy. 2x2 grid: columns = fixed_R / fixed_snr, rows = variable / fixed jitter.
Each panel overlays EnKF and PF, points labelled by the jitter multiple the calibrator chose
(constant down a 'fixed' panel by construction). ratio=1 is well calibrated; <1 under-dispersed
(overconfident), >1 over-dispersed. Reads the four
data/stage2_results_w{W}_{mode}_{jitter}.npz files. Per window.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.8  Changed: 2x2 panel (noise mode x variable/fixed jitter); reads jitter-tagged npz
# 2.7  Changed: read window from cfg, tag input/output by window (w{obs_every})
# 2.6  Fixed: skip stage2 results that predate the ratio fields instead of crashing
# 2.5  created — spread/RMSE vs alpha with jitter-multiplier labels
# ============================================================

os.makedirs('figs', exist_ok=True)
W = cfg.obs_every
modes = ['fixed_R', 'fixed_snr']
jitters = ['variable', 'fixed']
EN, PF = '#c0392b', '#2471a3'

fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)

for ri, jit in enumerate(jitters):
    for ci, mode in enumerate(modes):
        ax = axes[ri, ci]
        f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
        if not os.path.exists(f) or 'en_ratio' not in np.load(f).files:
            ax.text(0.5, 0.5, f'no data\n{mode} / {jit}', ha='center', va='center', transform=ax.transAxes)
            continue
        d = np.load(f); a = d['alphas']
        ax.axhspan(0.95, 1.05, color='green', alpha=0.10, label='target 1 ± 0.05')
        ax.axhline(1, ls='--', color='k', lw=1)
        ax.plot(a, d['en_ratio'], 'o-', color=EN, lw=2, label='EnKF')
        ax.plot(a, d['pf_ratio'], 's-', color=PF, lw=2, label='PF')
        for x, y, m in zip(a, d['en_ratio'], d['en_mult']):     # EnKF jitter labels above
            ax.annotate(f'x{m:.1f}', (x, y), textcoords='offset points', xytext=(0, 7),
                        ha='center', fontsize=6.5, color=EN)
        for x, y, m in zip(a, d['pf_ratio'], d['pf_mult']):     # PF jitter labels below
            ax.annotate(f'x{m:.1f}', (x, y), textcoords='offset points', xytext=(0, -12),
                        ha='center', fontsize=6.5, color=PF)
        ax.set_title(f'{mode}  —  {jit} jitter'); ax.grid(alpha=0.3)
        if ri == 1:
            ax.set_xlabel(r'$\alpha$')
        if ci == 0:
            ax.set_ylabel('spread / RMSE  (calibrated)')
        if ri == 0 and ci == 0:
            ax.legend(loc='upper left', fontsize=8)

fig.suptitle(f'Calibration quality vs nonlinearity — window {W}  (labels = chosen jitter x base)', y=1.0)
fig.tight_layout()
fig.savefig(f'figs/calibration_ratio_w{W}.png', dpi=145, bbox_inches='tight')
print(f'saved figs/calibration_ratio_w{W}.png')
