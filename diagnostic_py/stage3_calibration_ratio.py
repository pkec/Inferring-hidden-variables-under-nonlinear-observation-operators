"""
calibration_ratio.py — spread/RMSE ratio vs alpha for the current config, all four filters
(EnKF, PF, quad-reg EnKF, iterative EnKF). ratio=1 is well calibrated; <1 under-dispersed
(overconfident, e.g. a collapsed ensemble); >1 over-dispersed. A ratio far from 1 flags a filter
that isn't running healthily even where its RMSE looks acceptable. Reads
data/stage2_results_w{W}_{mode}_{jitter}.npz.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 3.8  Changed: single-config single panel (was 2x2 mode x jitter); added quad-reg and iterative
#               EnKF ratio lines alongside EnKF/PF to flag mis-calibration/collapse of each remedy
# 2.14 Added: +/-1 std (across seeds) shaded band on the ratio curves
# 2.5  created — spread/RMSE vs alpha
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run stage2.py first')

d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)

# (label, colour, marker, ratio key)
filters = [
    ('EnKF',           '#c0392b', 'o-', 'en_ratio'),
    ('PF (reference)', '#2471a3', 's-', 'pf_ratio'),
    ('quad-reg EnKF',  '#e67e22', 'D-', 'qr_ratio'),
    ('iterative EnKF', '#16a085', '^-', 'ie_ratio'),
]

fig, ax = plt.subplots(figsize=(8, 5))
ax.axhspan(0.95, 1.05, color='green', alpha=0.10, label='target 1 ± 0.05')
ax.axhline(1, ls='--', color='k', lw=1)
for label, col, mk, kr in filters:
    if kr not in d.files:
        continue
    ax.plot(a, d[kr], mk, color=col, lw=2, label=label)
    ax.fill_between(a, d[kr] - sd(kr + '_std'), d[kr] + sd(kr + '_std'), color=col, alpha=0.15)
ax.set_xlabel(r'$\alpha$'); ax.set_ylabel('spread / RMSE  (calibrated)')
ax.set_title(f'Calibration quality vs nonlinearity — window {W}, {mode}, {jit} jitter')
ax.grid(alpha=0.3); ax.legend(loc='best', fontsize=8)
fig.tight_layout()
fig.savefig(f'figs/diagnostic/calibration_ratio_w{W}.png', dpi=145, bbox_inches='tight')
print(f'saved figs/diagnostic/calibration_ratio_w{W}.png')
