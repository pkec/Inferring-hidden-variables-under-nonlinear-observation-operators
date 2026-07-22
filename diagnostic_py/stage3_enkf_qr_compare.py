"""
enkf_qr_compare.py — dedicated head-to-head for the quadratic-regression EnKF: analysis RMSE of
the linear EnKF (before), the QR EnKF (after) and the PF (reference) vs alpha, so QR's absolute
performance and how much of the EnKF-vs-PF gap it closes are both visible. 2x2 grid: columns =
fixed_R / fixed_snr, rows = variable / fixed jitter. Reads the four
data/stage2_results_w{W}_{mode}_{jitter}.npz files (run stage2.py first). Per window.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs -> figs/diagnostic; modes/jitters default to the committed single config
#      (fixed_snr, variable) with a general len(jitters) x len(modes) grid, so the full 2x2
#      mode x jitter comparison is one edit away for the writeup;  Added: sys.path bootstrap
# 3.0  created — QR-EnKF vs linear-EnKF vs PF RMSE comparison (2x2 mode x jitter), per window
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
W = cfg.obs_every
modes = ['fixed_snr']              # committed; use ['fixed_R', 'fixed_snr'] for the writeup grid
jitters = ['variable']             # committed; use ['variable', 'fixed'] for the writeup grid
EN, QR, PF = '#c0392b', '#e67e22', '#2471a3'

fig, axes = plt.subplots(len(jitters), len(modes),
                         figsize=(6.5 * len(modes), 4.5 * len(jitters)), sharex=True, squeeze=False)

for ri, jit in enumerate(jitters):
    for ci, mode in enumerate(modes):
        ax = axes[ri, ci]
        f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
        if not os.path.exists(f) or 'rmse_qr' not in np.load(f).files:
            ax.text(0.5, 0.5, f'no QR data\n{mode} / {jit}', ha='center', va='center', transform=ax.transAxes)
            continue
        d = np.load(f); a = d['alphas']
        sd = lambda k: d[k] if k in d.files else np.zeros_like(a)   # seed std if present
        ax.errorbar(a, d['rmse_en'], yerr=sd('rmse_en_std'), fmt='o-', color=EN, lw=2, capsize=2, label='linear EnKF')
        ax.errorbar(a, d['rmse_qr'], yerr=sd('rmse_qr_std'), fmt='D-', color=QR, lw=2, capsize=2, label='quad-reg EnKF')
        ax.errorbar(a, d['rmse_pf'], yerr=sd('rmse_pf_std'), fmt='s-', color=PF, lw=2, capsize=2, label='PF (reference)')
        ax.set_title(f'{mode}  —  {jit} jitter'); ax.grid(alpha=0.3)
        if ri == 1:
            ax.set_xlabel(r'$\alpha$')
        if ci == 0:
            ax.set_ylabel('analysis RMSE')
        if ri == 0 and ci == 0:
            ax.legend(loc='upper left', fontsize=8)

fig.suptitle(f'Quadratic-regression vs linear EnKF — window {W}  (PF as reference)', y=1.0)
fig.tight_layout()
fig.savefig(f'figs/diagnostic/enkf_qr_compare_w{W}.png', dpi=145, bbox_inches='tight')
print(f'saved figs/diagnostic/enkf_qr_compare_w{W}.png')
