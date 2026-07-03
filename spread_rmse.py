"""
spread_rmse.py — spread and RMSE plotted separately vs alpha for one window, so the calibration
ratio can't hide which term is moving (e.g. flat spread under a rising RMSE). Same four configs
as calibration_ratio (noise mode x jitter strategy), but each split into a spread panel (top)
and an RMSE panel (bottom), both with EnKF and PF. The pair shares a y-axis per column, so the
relative motion of spread vs RMSE is read straight off the plot. Spread is recovered exactly as
ratio x RMSE from the stage2 npz, so no re-run is needed. Reads
data/stage2_results_w{W}_{mode}_{jitter}.npz. Per window.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.11 created — spread and RMSE vs alpha split out of the calibration ratio, per window
# ============================================================

os.makedirs('figs', exist_ok=True)
W = cfg.obs_every
configs = [(m, j) for m in ('fixed_R', 'fixed_snr') for j in ('variable', 'fixed')]
EN, PF = '#c0392b', '#2471a3'

fig, axes = plt.subplots(2, 4, figsize=(18, 7.2), sharex=True, sharey='col')

for ci, (mode, jit) in enumerate(configs):
    asp, arm = axes[0, ci], axes[1, ci]
    asp.set_title(f'{mode}\n{jit} jitter')
    arm.set_xlabel(r'$\alpha$')
    f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
    if not os.path.exists(f) or 'en_ratio' not in np.load(f).files:
        for ax in (asp, arm):
            ax.text(0.5, 0.5, f'no data\n{mode} / {jit}', ha='center', va='center', transform=ax.transAxes)
        continue
    d = np.load(f); a = d['alphas']
    en_spread = d['en_ratio'] * d['rmse_en']            # spread = (spread/RMSE) x RMSE, exact
    pf_spread = d['pf_ratio'] * d['rmse_pf']

    asp.plot(a, en_spread, 'o-', color=EN, lw=2, label='EnKF')
    asp.plot(a, pf_spread, 's-', color=PF, lw=2, label='PF')
    asp.grid(alpha=0.3)

    arm.plot(a, d['rmse_en'], 'o-', color=EN, lw=2, label='EnKF')
    arm.plot(a, d['rmse_pf'], 's-', color=PF, lw=2, label='PF')
    arm.grid(alpha=0.3)

axes[0, 0].set_ylabel('spread'); axes[1, 0].set_ylabel('RMSE')
axes[0, 0].legend(loc='upper left', fontsize=8)

fig.suptitle(f'Spread and RMSE vs nonlinearity — window {W}  (y shared per column: spread above its RMSE)', y=1.0)
fig.tight_layout()
fig.savefig(f'figs/spread_rmse_w{W}.png', dpi=145, bbox_inches='tight')
print(f'saved figs/spread_rmse_w{W}.png')
