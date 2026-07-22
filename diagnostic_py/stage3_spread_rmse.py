"""
spread_rmse.py — spread and RMSE plotted separately vs alpha for the current config, so the
calibration ratio can't hide which term is moving. All four filters (EnKF, PF, quad-reg EnKF,
iterative EnKF) on each panel: (a) spread, (b) RMSE. Lets you see whether a remedy's RMSE win
comes with a healthy spread or a collapsed one. Reads data/stage2_results_w{W}_{mode}_{jitter}.npz.
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
# 3.8  Changed: single-config 1x2 layout (was 2x4 over mode x jitter); added quad-reg and
#               iterative EnKF lines alongside EnKF/PF so every filter's spread & RMSE are visible
# 2.14 Changed: read the seed-averaged en_spread/pf_spread saved by stage2 (+/-1 std bands)
# 2.11 created — spread and RMSE vs alpha split out of the calibration ratio, per window
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run stage2.py first')

d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)

# (label, colour, marker, spread key, rmse key)
filters = [
    ('EnKF',           '#c0392b', 'o-', 'en_spread', 'rmse_en'),
    ('PF (reference)', '#2471a3', 's-', 'pf_spread', 'rmse_pf'),
    ('quad-reg EnKF',  '#e67e22', 'D-', 'qr_spread', 'rmse_qr'),
    ('iterative EnKF', '#16a085', '^-', 'ie_spread', 'rmse_ie'),
]

fig, (asp, arm) = plt.subplots(1, 2, figsize=(13, 4.6), sharex=True)
for label, col, mk, ks, kr in filters:
    if ks not in d.files:
        continue
    asp.plot(a, d[ks], mk, color=col, lw=2, label=label)
    asp.fill_between(a, d[ks] - sd(ks + '_std'), d[ks] + sd(ks + '_std'), color=col, alpha=0.15)
    arm.plot(a, d[kr], mk, color=col, lw=2, label=label)
    arm.fill_between(a, d[kr] - sd(kr + '_std'), d[kr] + sd(kr + '_std'), color=col, alpha=0.15)

asp.set_title('(a) spread'); asp.set_ylabel('spread'); asp.set_xlabel(r'$\alpha$')
asp.grid(alpha=0.3); asp.legend(fontsize=8)
arm.set_title('(b) RMSE'); arm.set_ylabel('RMSE'); arm.set_xlabel(r'$\alpha$'); arm.grid(alpha=0.3)
fig.suptitle(f'Spread and RMSE vs nonlinearity — window {W}, {mode}, {jit} jitter', y=1.02)
fig.tight_layout()
fig.savefig(f'figs/diagnostic/spread_rmse_w{W}.png', dpi=145, bbox_inches='tight')
print(f'saved figs/diagnostic/spread_rmse_w{W}.png')
