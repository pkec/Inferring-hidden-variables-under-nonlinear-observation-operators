"""
spread_rmse_timeseries.py — per-cycle spread and RMSE OVER TIME (not over alpha) for the standard
EnKF and the iterative EnKF, at a few alphas (0, 0.5, 1.0). Distinct from spread_rmse.py, which
plots spread/RMSE vs the nonlinearity alpha; this one fixes alpha and shows the raw per-cycle
trajectory, so transient events (wild mean steps, lobe transitions) are visible rather than
averaged away. Top row = spread per cycle, bottom row = RMSE per cycle; one column per alpha,
EnKF vs IEnKF overlaid. RMSE/spread here are the root-mean over the 3 state axes at each cycle.
Reads data/l63_twin.npz (run truth_obs.py first).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
from config import cfg, rk4_vec
from enkf import EnKF
from enkf_ienkf import EnKF_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 3.9  created — per-cycle spread & RMSE time series, EnKF vs IEnKF, across alpha
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
ALPHAS = [0.0, 0.5, 1.0]
FILTERS = [('EnKF', EnKF, '#c0392b'), ('IEnKF', EnKF_ienkf, '#2471a3')]

d = np.load('data/l63_twin.npz')
truth, oi = d['truth'], d['obs_idx']
on, osa, al = d['obs_nonlinear'], d['obs_std_alpha'], list(np.round(d['alphas'], 2))
t = oi * cfg.dt

def run(anafn, a):
    """One filter at nonlinearity a. Returns per-cycle RMSE and spread (root-mean over x,y,z)."""
    ai = al.index(round(a, 2))
    obs, ostd = on[ai], osa[ai]
    R = np.diag(ostd ** 2)
    h = lambda x, a=a: x + a * x**2
    rng = np.random.default_rng(cfg.seed)
    ens = np.tile(truth[0], (cfg.ensembleN, 1)) + rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.init_std
    rmse = np.zeros(len(oi)); spread = np.zeros(len(oi))
    for k in range(len(oi)):
        ens += rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ens = rk4_vec(ens, cfg.dt)
        ens, _, _ = anafn(ens, obs[k], R, h, rng)
        m = ens.mean(0)
        rmse[k]   = np.sqrt(((truth[oi[k]] - m) ** 2).mean())     # root-mean over 3 axes
        spread[k] = np.sqrt((ens.std(0) ** 2).mean())
    return rmse, spread

fig, axes = plt.subplots(2, len(ALPHAS), figsize=(5 * len(ALPHAS), 7), sharex=True)
for ci, a in enumerate(ALPHAS):
    asp, arm = axes[0, ci], axes[1, ci]
    for fname, fn, col in FILTERS:
        rmse, spread = run(fn, a)
        asp.plot(t, spread, color=col, lw=0.8, label=fname)
        arm.plot(t, rmse,   color=col, lw=0.8, label=fname)
    asp.set_title(rf'$\alpha$ = {a}'); asp.grid(alpha=0.3)
    arm.grid(alpha=0.3); arm.set_xlabel('time')
    if ci == 0:
        asp.set_ylabel('spread (per cycle)'); arm.set_ylabel('RMSE (per cycle)')
        asp.legend(fontsize=8, loc='upper right')

fig.suptitle(f'Per-cycle spread & RMSE over time — EnKF vs IEnKF, window {cfg.obs_every}, {cfg.noise_mode}', y=1.0)
fig.tight_layout()
fig.savefig(f'figs/diagnostic/spread_rmse_timeseries_w{cfg.obs_every}.png', dpi=140, bbox_inches='tight')
print(f'saved figs/diagnostic/spread_rmse_timeseries_w{cfg.obs_every}.png')
