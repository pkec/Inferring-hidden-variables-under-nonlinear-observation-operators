"""
bimodal_snapshot.py — ensemble distribution snapshots at chosen assimilation times, showing what
the EnKF and the IEnKF each do to the forecast cloud. Two modes, set by L63_SNAP_MODE:

  'independent' (default) — each filter propagates its OWN trajectory, so the panels show the state
      each filter actually reaches. Columns: EnKF forecast | EnKF analysis | IEnKF forecast |
      IEnKF analysis. Use this to explain spikes in a filter's own RMSE timeseries.
  'controlled' — both analyses are applied to one shared forecast cloud (propagated by the filter
      named in L63_SNAP_CARRIER), isolating the analysis step. Columns: forecast | EnKF | IEnKF.

Histograms are of the x-component. Marked on each panel: truth, that cloud's own ensemble mean, the
two states satisfying h(x) = d (the preimages of the x-observation), and the fold at
x = -1/(2*alpha), which is exactly the midpoint of those two states. Panels are titled by model
time t = obs_idx*dt so they line up with spread_rmse_timeseries.py and ienkf_trace.py. Snapshots
default to the times of the IEnKF's largest errors (independent) or the most bimodal forecasts
(controlled) unless L63_SNAP_TIMES / L63_SNAP_CYCLES is set. Reads data/l63_twin.npz.

Env: L63_SNAP_MODE (independent|controlled), L63_SNAP_ALPHA (default 1.0),
     L63_SNAP_TIMES (e.g. "28,85"), L63_SNAP_CYCLES (e.g. "10,24"), L63_SNAP_N (default 3),
     L63_SNAP_CARRIER (enkf|ienkf, controlled mode only).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
from config import cfg, rk4_vec
from enkf import EnKF
from enkf_ienkf import EnKF_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 3.14 Added:   L63_SNAP_MODE 'independent' (now the default) — each filter propagates its own
#               trajectory, 4 columns, so the trapped wrong-lobe state the IEnKF actually reaches
#               is visible; default snapshot times become the IEnKF's largest-error cycles
#      Changed: 'controlled' (shared forecast cloud) is now an opt-in mode
# 3.13 Added:   L63_SNAP_CARRIER (enkf|ienkf) — which filter propagates the trajectory
# 3.12 Changed: panel titles show model time t = obs_idx*dt instead of the cycle index
#      Added:   L63_SNAP_TIMES to pick snapshots by time (nearest cycle)
# 3.11 Added:   ensemble-mean line per panel
#      Changed: preimage lines relabelled h(x) = d
# 3.10 created — per-cycle ensemble snapshots: EnKF keeps a bimodal forecast bimodal, the IEnKF
#      locks onto a single preimage
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
MODE = os.environ.get('L63_SNAP_MODE', 'independent')
A_SNAP = float(os.environ.get('L63_SNAP_ALPHA', 1.0))
N_SNAP = int(os.environ.get('L63_SNAP_N', 3))
CARRIER = os.environ.get('L63_SNAP_CARRIER', 'enkf')

d = np.load('data/l63_twin.npz')
truth, oi = d['truth'], d['obs_idx']
on, osa, al = d['obs_nonlinear'], d['obs_std_alpha'], list(np.round(d['alphas'], 2))
ai = al.index(round(A_SNAP, 2))
obs, ostd = on[ai], osa[ai]
R = np.diag(ostd ** 2)
h = lambda x: x + A_SNAP * x**2
fold = -1.0 / (2 * A_SNAP)                        # h'(x)=0 here; also the midpoint of the preimages
t_obs = oi * cfg.dt

def run(fn):
    """Propagate one filter over the whole window; return its per-cycle forecast and analysis clouds."""
    rng = np.random.default_rng(cfg.seed)
    ens = np.tile(truth[0], (cfg.ensembleN, 1)) + rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.init_std
    fc, an = [], []
    for k in range(len(oi)):
        ens += rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ens = rk4_vec(ens, cfg.dt)
        fc.append(ens.copy())
        ens, _, _ = fn(ens, obs[k], R, h, rng)
        an.append(ens.copy())
    return fc, an

def bimodality(v):
    """Sarle's bimodality coefficient; > 5/9 ~ 0.555 indicates a bimodal sample."""
    n = len(v)
    g, k = skew(v), kurtosis(v)
    return (g**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))

def preimages(obs_x):
    """The two states x with h(x) = obs_x, i.e. roots of alpha x^2 + x - obs_x = 0."""
    disc = 1 + 4 * A_SNAP * obs_x
    return [] if disc < 0 else sorted([(-1 - np.sqrt(disc)) / (2 * A_SNAP),
                                       (-1 + np.sqrt(disc)) / (2 * A_SNAP)])

fc_e, an_e = run(EnKF)
if MODE == 'independent':
    fc_i, an_i = run(EnKF_ienkf)
    ie_err = np.array([abs(an_i[k][:, 0].mean() - truth[oi[k]][0]) for k in range(len(oi))])
    auto = np.argsort(ie_err)[-N_SNAP:]           # where the IEnKF's own error is worst
else:
    fc_c, _ = (fc_e, an_e) if CARRIER == 'enkf' else run(EnKF_ienkf)
    auto = np.argsort([bimodality(f[:, 0]) for f in fc_c])[-N_SNAP:]

if 'L63_SNAP_TIMES' in os.environ:
    cycles = [int(np.argmin(np.abs(t_obs - float(t)))) for t in os.environ['L63_SNAP_TIMES'].split(',')]
elif 'L63_SNAP_CYCLES' in os.environ:
    cycles = [int(c) for c in os.environ['L63_SNAP_CYCLES'].split(',')]
else:
    cycles = sorted(auto)

ncol = 4 if MODE == 'independent' else 3
fig, axes = plt.subplots(len(cycles), ncol, figsize=(4.6 * ncol, 3.2 * len(cycles)), squeeze=False)
for ri, k in enumerate(cycles):
    if MODE == 'independent':
        panels = [(fc_e[k], 'EnKF forecast', '#e67e22'), (an_e[k], 'EnKF analysis', '#c0392b'),
                  (fc_i[k], 'IEnKF forecast', '#f1c40f'), (an_i[k], 'IEnKF analysis', '#2471a3')]
    else:
        Af = fc_c[k]
        Ae, _, _ = EnKF(Af.copy(), obs[k], R, h, np.random.default_rng(99))
        Ai, _, _ = EnKF_ienkf(Af.copy(), obs[k], R, h, np.random.default_rng(99))
        panels = [(Af, f'{CARRIER} forecast', '#e67e22'), (Ae, 'EnKF analysis', '#c0392b'),
                  (Ai, 'IEnKF analysis', '#2471a3')]
    pre = preimages(obs[k][0])
    for ci, (cloud, name, col) in enumerate(panels):
        ax = axes[ri, ci]
        ax.hist(cloud[:, 0], bins=60, color=col, alpha=0.75, edgecolor='none')
        ax.axvline(truth[oi[k]][0], color='k', lw=1.6, label='truth')
        ax.axvline(cloud[:, 0].mean(), color='#555', ls='-.', lw=1.5)   # this cloud's own mean
        for p in pre:
            ax.axvline(p, color='#16a085', ls='--', lw=1.2)
        ax.axvline(fold, color='#8e44ad', ls=':', lw=1.4)
        ax.set_title(f't = {t_obs[k]:.2f} — {name}', fontsize=10)
        ax.set_xlabel('x'); ax.grid(alpha=0.3)
        if ci == 0:
            ax.set_ylabel(f'cycle {k}\nBC={bimodality(cloud[:, 0]):.2f}\ncount')
        if ri == 0 and ci == 0:
            ax.plot([], [], '-.', color='#555', label='ensemble mean')
            ax.plot([], [], '--', color='#16a085', label=r'$h(x)=d$')
            ax.plot([], [], ':', color='#8e44ad', label=f'fold x={fold:.2f}')
            ax.legend(fontsize=7)
    line = f"t={t_obs[k]:.2f} (cycle {k}): h(x)=d at {np.round(pre,2)}  truth {truth[oi[k]][0]:.2f}"
    for cloud, name, _ in panels:
        line += f"\n    {name:16s} mean {cloud[:,0].mean():8.2f}  std {cloud[:,0].std():6.2f}  BC {bimodality(cloud[:,0]):.2f}"
    print(line)

fig.suptitle(rf'$h_\alpha$, $\alpha$={A_SNAP} — {MODE} propagation '
             f'(window {cfg.obs_every}, {cfg.noise_mode})', y=1.0)
fig.tight_layout()
out = f'figs/diagnostic/bimodal_snapshot_w{cfg.obs_every}_a{A_SNAP}_{MODE}.png'
fig.savefig(out, dpi=140, bbox_inches='tight')
print(f'saved {out}')