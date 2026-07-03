import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg, rk4, PERTURB_BY_WINDOW
from partfilt import run_pf
from enkf import run_enkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.10 Changed: self-contained + consolidated — each Stage 1 figure is one PNG with both
#               windows as panels (traces 3x2, rmse/spread 2x2, rank-hist 2x3); no per-window files
# 2.9  Changed: window-aware — figures tagged _w{obs_every}, titles show the window; ensure figs/ exists
# ============================================================

os.makedirs('figs', exist_ok=True)
windows = [5, 15]                          # each window gets its own column/row of panels
labels = ['x', 'y', 'z']
h = lambda x: x                            # linear observation operator (Stage 1 baseline)

# truth trajectory (same recipe as truth_obs.py; deterministic, shared across windows)
s = np.array([1.0, 1.0, 1.0])
for _ in range(500): s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps): truth[k + 1] = rk4(truth[k], cfg.dt)
t_truth = np.arange(cfg.n_steps + 1) * cfg.dt

def rmse(sq): return np.sqrt(sq.mean(0))
def spr(sp):  return np.sqrt(sp.mean(0))

# run both filters at each window's calibrated baseline jitter
res = {}
for W in windows:
    cfg.obs_every = W
    cfg.perturb_std = PERTURB_BY_WINDOW[W]
    obs_idx = np.arange(W, cfg.n_steps + 1, W)
    rng = np.random.default_rng(cfg.seed)
    obs = truth[obs_idx] + rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
    pf = run_pf(obs, truth, obs_idx, h, save_particles=True)
    en = run_enkf(obs, truth, obs_idx, h, save_forecast=False)
    res[W] = dict(pf=pf, en=en, obs=obs, t_obs=obs_idx * cfg.dt)

    pr, er = rmse(pf['sqerror']), rmse(en['sqerror'])
    ps, es = spr(pf['spread']),  spr(en['spread'])
    print(f"\nwindow {W}  (linear h)        x        y        z")
    print(f"{'PF RMSE/spread':>16}" + ''.join(f"{v:9.3f}" for v in pr / ps))
    print(f"{'EnKF RMSE/spread':>16}" + ''.join(f"{v:9.3f}" for v in er / es))
    print(f"{'EnKF excess %':>16}" + ''.join(f"{v:8.1f}%" for v in (er - pr) / pr * 100))


# ---- Figure 1: time traces (rows x,y,z  x  cols = window) ----
fig, axes = plt.subplots(3, 2, figsize=(14, 8), sharex='col')
for ci, W in enumerate(windows):
    r = res[W]
    for ri in range(3):
        ax = axes[ri, ci]
        ax.plot(t_truth, truth[:, ri], 'k-', lw=0.8, label='Truth')
        ax.plot(r['t_obs'], r['obs'][:, ri], 'r.', ms=2, alpha=0.3, label='Obs')
        ax.plot(r['t_obs'], r['pf']['en_mean'][:, ri], 'b-', lw=0.7, label='PF')
        ax.plot(r['t_obs'], r['en']['en_mean'][:, ri], 'g-', lw=0.7, label='EnKF')
        if ci == 0: ax.set_ylabel(labels[ri])
        if ri == 0: ax.set_title(f'window {W}')
        if ri == 0 and ci == 0: ax.legend(loc='upper right', fontsize=8)
    axes[2, ci].set_xlabel('Time')
fig.suptitle('Stage 1: PF vs EnKF, linear h')
fig.tight_layout(); fig.savefig('figs/stage1_traces.png', dpi=130)


# ---- Figure 2: RMSE & spread bars (rows RMSE/spread  x  cols = window) ----
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
x = np.arange(3); bw = 0.35
for ci, W in enumerate(windows):
    r = res[W]
    a0, a1 = axes[0, ci], axes[1, ci]
    a0.bar(x - bw/2, rmse(r['pf']['sqerror']), bw, label='PF')
    a0.bar(x + bw/2, rmse(r['en']['sqerror']), bw, label='EnKF')
    a0.set_xticks(x); a0.set_xticklabels(labels); a0.set_title(f'window {W} — RMSE')
    a1.bar(x - bw/2, spr(r['pf']['spread']), bw, label='PF')
    a1.bar(x + bw/2, spr(r['en']['spread']), bw, label='EnKF')
    a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_title(f'window {W} — spread')
    if ci == 0: a0.legend(); a1.legend()
fig.suptitle('Stage 1: RMSE & spread agreement, linear h')
fig.tight_layout(); fig.savefig('figs/stage1_rmse_spread.png', dpi=130)


# ---- Figure 3: PF rank histograms (rows = window  x  cols x,y,z) ----
n_bins = 25
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ri, W in enumerate(windows):
    particles = res[W]['pf']['particles_history']
    tao = res[W]['pf']['truth_at_obs']
    T = particles.shape[0]
    for v in range(3):
        ranks = np.array([np.searchsorted(np.sort(particles[t, :, v]), tao[t, v]) for t in range(T)])
        ax = axes[ri, v]
        ax.hist(ranks, bins=n_bins, edgecolor='black', alpha=0.7)
        ax.axhline(T / n_bins, color='red', ls='--', label='Uniform')
        ax.set_title(f'window {W} — {labels[v]}')
        ax.set_xlabel('Rank')
        if v == 0 and ri == 0: ax.legend()
    axes[ri, 0].set_ylabel(f'window {W}\ncount')
fig.suptitle('Stage 1: PF rank histograms, linear h')
fig.tight_layout(); fig.savefig('figs/stage1_rank_hist.png', dpi=130)

print("\nsaved figs/stage1_traces.png, figs/stage1_rmse_spread.png, figs/stage1_rank_hist.png")
