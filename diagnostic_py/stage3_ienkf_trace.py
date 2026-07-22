"""
ienkf_trace.py — trace-collapse check for the iterative EnKF against the standard EnKF, at a few
alphas (0, 0.5, 1.0). For each filter and alpha it plots, per assimilation cycle: the forecast
(prior) covariance trace, the analysis covariance trace, and the actual squared error to truth,
plus the observation-error trace tr(R). A healthy filter's analysis trace tracks the squared
error and sits below tr(R); wild-mean-step cycles show up as green squared-error spikes far above
the (modest) analysis trace. Self-contained apart from the twin in data/l63_twin.npz
(run truth_obs.py first). Rows = filter (EnKF / IEnKF), columns = alpha.
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
# 3.9  created — IEnKF vs EnKF covariance-trace / squared-error diagnostic across alpha
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
ALPHAS = [0.0, 0.5, 1.0]
FILTERS = [('EnKF', EnKF), ('IEnKF', EnKF_ienkf)]

d = np.load('data/l63_twin.npz')
truth, oi = d['truth'], d['obs_idx']
on, osa, al = d['obs_nonlinear'], d['obs_std_alpha'], list(np.round(d['alphas'], 2))
t = oi * cfg.dt

def run(anafn, a):
    """One filter over the window at nonlinearity a. Returns per-cycle forecast trace,
    analysis trace, squared error, and tr(R)."""
    ai = al.index(round(a, 2))
    obs, ostd = on[ai], osa[ai]
    R = np.diag(ostd ** 2)
    h = lambda x, a=a: x + a * x**2
    rng = np.random.default_rng(cfg.seed)
    ens = np.tile(truth[0], (cfg.ensembleN, 1)) + rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.init_std
    trf = np.zeros(len(oi)); tra = np.zeros(len(oi)); sqe = np.zeros(len(oi))
    for k in range(len(oi)):
        ens += rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ens = rk4_vec(ens, cfg.dt)
        trf[k] = (ens.std(0) ** 2).sum()                 # forecast (prior) trace
        ens, _, _ = anafn(ens, obs[k], R, h, rng)
        tra[k] = (ens.std(0) ** 2).sum()                 # analysis trace
        sqe[k] = ((truth[oi[k]] - ens.mean(0)) ** 2).sum()
    return trf, tra, sqe, (ostd ** 2).sum()

fig, axes = plt.subplots(len(FILTERS), len(ALPHAS), figsize=(5 * len(ALPHAS), 8), sharex=True)
for ri, (fname, fn) in enumerate(FILTERS):
    for ci, a in enumerate(ALPHAS):
        ax = axes[ri, ci]
        trf, tra, sqe, trR = run(fn, a)
        ax.semilogy(t, trf, color='#e67e22', lw=0.7, alpha=0.85, label='forecast trace')
        ax.semilogy(t, tra, color='#c0392b', lw=0.7, alpha=0.85, label='analysis trace')
        ax.semilogy(t, sqe, color='#16a085', lw=0.7, alpha=0.7, label='squared error')
        ax.axhline(trR, ls='--', color='#2471a3', lw=1.3, label=f'tr(R)={trR:.0f}')
        ax.grid(alpha=0.3, which='both')
        if ri == 0:
            ax.set_title(rf'$\alpha$ = {a}')
        if ci == 0:
            ax.set_ylabel(f'{fname}\ntrace / sq-error')
        if ri == len(FILTERS) - 1:
            ax.set_xlabel('time')
        ax.legend(fontsize=6.5, loc='upper right')
        cal = np.sqrt(np.median(tra)) / np.sqrt(np.median(sqe))
        print(f"{fname:5s} alpha={a}: median analysis trace {np.median(tra):7.3f}  "
              f"median sq-err {np.median(sqe):7.3f}  spread/err {cal:.2f}  "
              f"spikes>10x median {int((sqe > 10 * np.median(sqe)).sum())}")

fig.suptitle(f'IEnKF vs EnKF trace-collapse check — window {cfg.obs_every}, {cfg.noise_mode}', y=1.0)
fig.tight_layout()
fig.savefig(f'figs/diagnostic/ienkf_trace_w{cfg.obs_every}.png', dpi=140, bbox_inches='tight')
print(f'saved figs/diagnostic/ienkf_trace_w{cfg.obs_every}.png')
