"""
trace_collapse.py — check whether the EnKF analysis covariance collapses.
The trace of the ensemble covariance is the filter's total uncertainty. If it shrinks
toward zero each cycle and sits far below the observation error tr(R) — with the forecast
trace failing to re-grow between updates — the filter has become overconfident: it stops
believing observations and can lose the truth at regime changes (Lorenz lobe transitions),
which shows up as elevated RMSE. Linear h, one panel per window at the per-window baseline
jitter. Self-contained (builds its own truth, same recipe as truth_obs.py).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg, rk4, PERTURB_BY_WINDOW
from enkf import run_enkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.13 created — EnKF covariance-trace collapse diagnostic (forecast/analysis trace vs obs error & climatology)
# ============================================================

os.makedirs('figs', exist_ok=True)
windows = [5, 15]
h = lambda x: x                               # linear baseline

s = np.array([1.0, 1.0, 1.0])                 # truth, same recipe as truth_obs.py
for _ in range(500): s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps): truth[k + 1] = rk4(truth[k], cfg.dt)

tr_R = (cfg.obs_std ** 2).sum()               # observation-error trace
tr_clim = truth.var(axis=0).sum()             # climatological (no-skill) variance trace

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, W in zip(axes, windows):
    cfg.obs_every = W
    cfg.perturb_std = PERTURB_BY_WINDOW[W]
    obs_idx = np.arange(W, cfg.n_steps + 1, W)
    rng = np.random.default_rng(cfg.seed)
    obs = truth[obs_idx] + rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
    en = run_enkf(obs, truth, obs_idx, h, save_forecast=False)

    tr_f = (en['fc_spread'] ** 2).sum(1)      # forecast (prior) cov trace per cycle
    tr_a = en['spread'].sum(1)                # analysis cov trace per cycle (spread is variance)
    t = obs_idx * cfg.dt

    ax.semilogy(t, tr_f, color='#e67e22', lw=0.7, alpha=0.85, label='forecast trace')
    ax.semilogy(t, tr_a, color='#c0392b', lw=0.7, alpha=0.85, label='analysis trace')
    ax.axhline(tr_R, ls='--', color='#2471a3', lw=1.4, label=f'obs error tr(R) = {tr_R:.1f}')
    ax.axhline(tr_clim, ls='--', color='#555', lw=1.4, label=f'climatology = {tr_clim:.0f}')
    ax.set_xlabel('time'); ax.set_title(f'window {W}   ($\\Delta t$ = {W*cfg.dt:.2f})')
    ax.grid(alpha=0.3, which='both')

    rmse = np.sqrt(en['sqerror'].mean(0))
    skill = (rmse / truth.std(axis=0)).mean()              # RMSE relative to climatology
    cal = (np.sqrt(en['spread'].mean(0)) / rmse).mean()    # spread/RMSE: <1 overconfident, =1 calibrated
    print(f"window {W}:  forecast trace median {np.median(tr_f):.3g}   "
          f"analysis trace median {np.median(tr_a):.3g}   tr(R) {tr_R:.1f}   "
          f"analysis/tr(R) {np.median(tr_a)/tr_R:.2g}   "
          f"spread/RMSE {cal:.2f}   RMSE/climatology {skill:.2f}")

axes[0].set_ylabel('covariance trace (sum of variances)')
axes[0].legend(loc='lower right', fontsize=8)
fig.suptitle('EnKF covariance trace vs observation error and climatology (linear h)')
fig.tight_layout(); fig.savefig('figs/trace_collapse.png', dpi=145)
print('saved figs/trace_collapse.png')
