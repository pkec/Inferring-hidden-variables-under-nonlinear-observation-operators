"""
stage2_errors.py — consolidated Stage 2 error panel for one window. Three error sources
(Jensen bias, cross-covariance error, EnKF-vs-PF RMSE) laid out as 2 rows x 6 cols:
rows are jitter strategy (variable / fixed), columns are fixed_R's 3 metrics then fixed_snr's
3 metrics. Only the RMSE panel overlays two curves (EnKF vs PF). Reads the four
data/stage2_results_w{W}_{mode}_{jitter}.npz files (run stage2.py first). Per window.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.8  created — consolidated 2x6 error panel (jitter rows x noise-mode metric cols) per window
# ============================================================

os.makedirs('figs', exist_ok=True)
W = cfg.obs_every
modes = ['fixed_R', 'fixed_snr']
jitters = ['variable', 'fixed']

EN, PF, J, C = '#c0392b', '#2471a3', '#8e44ad', '#16a085'
fig, axes = plt.subplots(2, 6, figsize=(22, 7.6))

for ri, jit in enumerate(jitters):
    for mi, mode in enumerate(modes):
        f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
        c0 = mi * 3                                   # first column for this noise mode
        if not os.path.exists(f):
            for k in range(3):
                axes[ri, c0 + k].text(0.5, 0.5, 'no data', ha='center', va='center',
                                      transform=axes[ri, c0 + k].transAxes)
            continue
        d = np.load(f); a = d['alphas']

        axes[ri, c0].plot(a, d['jensen_norm'], 'o-', color=J, lw=2)
        axes[ri, c0].set_title('(a) Jensen bias')
        axes[ri, c0].set_ylabel(r'$\|E[h(x)]-h(E[x])\|$')

        axes[ri, c0 + 1].plot(a, d['crosscov_err'], 'o-', color=C, lw=2)
        axes[ri, c0 + 1].set_title('(b) Cross-cov error')

        axes[ri, c0 + 2].fill_between(a, d['rmse_pf'], d['rmse_en'], color=EN, alpha=0.12)
        axes[ri, c0 + 2].plot(a, d['rmse_en'], 'o-', color=EN, lw=2, label='EnKF')
        axes[ri, c0 + 2].plot(a, d['rmse_pf'], 's-', color=PF, lw=2, label='PF')
        axes[ri, c0 + 2].set_title('(c) RMSE'); axes[ri, c0 + 2].legend(fontsize=7)

        for k in range(3):
            axes[ri, c0 + k].set_xlabel(r'$\alpha$'); axes[ri, c0 + k].grid(alpha=0.3)

fig.text(0.27, 0.99, 'fixed_R', ha='center', fontsize=13, weight='bold')   # left-half group header
fig.text(0.73, 0.99, 'fixed_snr', ha='center', fontsize=13, weight='bold')  # right-half group header
fig.text(0.004, 0.74, 'variable jitter', va='center', rotation=90, fontsize=11)
fig.text(0.004, 0.28, 'fixed jitter (alpha=0)', va='center', rotation=90, fontsize=11)

fig.suptitle(rf'Stage 2 error sources — window {W},  $h_\alpha(x)=x+\alpha x^2$', y=1.03)
fig.tight_layout(rect=[0.015, 0, 1, 0.97])
fig.savefig(f'figs/stage2_errors_w{W}.png', dpi=140, bbox_inches='tight')
print(f'saved figs/stage2_errors_w{W}.png')
