"""
window_sweep.py — sweep the assimilation window at alpha=0 (linear h), averaged over seeds:
  figs/fig_window.png       : (a) EnKF vs PF RMSE,  (b) EnKF RMSE excess over PF (per x,y,z)
  figs/fig_pf_fragility.png : the PF reference collapses without per-window jitter re-tuning
  figs/fig_mechanism.png    : the forecast prior spreads and deforms away from Gaussian (xy, xz, yz)
Run from the project folder (same place as run.py). Needs the save_forecast option in enkf.py.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from config import cfg, rk4
from partfilt import run_pf
from enkf import run_enkf

os.makedirs('figs', exist_ok=True)

windows = [5, 10, 15, 20, 25]                  # obs_every values (window = obs_every * cfg.dt)
seeds = list(range(3))                          # noise realizations to average over (more = smoother, slower)
base = cfg.perturb_std.copy()                  # PF jitter as tuned for obs_every=5
jitter_mults = np.geomspace(0.3, 8, 18)        # two-sided: can lower OR raise the base jitter
snap_time = 2000                               # truth step at which to snapshot the prior ensemble
h = lambda x: x                                # alpha = 0, linear observation operator

# --- truth trajectory (same recipe as truth_obs.py; deterministic, so it is shared across seeds) ---
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):                           # spin up onto the attractor
    s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3))
truth[0] = s
for k in range(cfg.n_steps):
    truth[k + 1] = rk4(truth[k], cfg.dt)
clim = truth.std(axis=0).mean()                # spread of the attractor = no-skill (climatology) RMSE

# --- sweep over seeds x windows ---
W, S = len(windows), len(seeds)
en_rmse_s = np.zeros((S, W, 3))                # EnKF RMSE
pf_rmse_s = np.zeros((S, W, 3))               # calibrated-PF RMSE
pf_u_rmse_s = np.zeros((S, W, 3))            # untuned-PF (base jitter) RMSE
pf_u_ratio_s = np.zeros((S, W))              # untuned-PF RMSE/spread
pf_mult_s = np.zeros((S, W))                  # PF jitter multiple used
pf_ratio_s = np.zeros((S, W))                 # re-tuned PF RMSE/spread (the calibration we achieve)
fc_std_s = np.zeros((S, W))                   # prior spread (mean over steps and vars)
snaps = {}                                     # prior snapshots, kept from the first seed only

for si, seed in enumerate(seeds):
    for wi, oe in enumerate(windows):
        cfg.obs_every = oe                     # drives both obs spacing and the propagation loop
        obs_idx = np.arange(oe, cfg.n_steps + 1, oe)
        rng = np.random.default_rng(seed)      # single noise realization for this seed
        obs = truth[obs_idx] + rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std

        # EnKF on base jitter; grab the prior ensemble once (first seed) for the mechanism plot
        cfg.perturb_std = base
        save_fc = (si == 0 and oe in (5, 25))
        en = run_enkf(obs, truth, obs_idx, h, seed=seed, save_forecast=save_fc)
        en_rmse_s[si, wi] = np.sqrt(en['sqerror'].mean(0))
        fc_std_s[si, wi] = en['fc_spread'].mean()
        if save_fc:
            k = snap_time // oe - 1            # cycle whose forecast lands at truth step snap_time
            snaps[oe] = en['fc_history'][k]

        # PF jitter scan: m=1 is the untuned (collapsing) PF; pick the calibrated one as the reference
        # untuned reference (base jitter) for panel (a)
        cfg.perturb_std = base
        pfu = run_pf(obs, truth, obs_idx, h, seed=seed)
        ru = np.sqrt(pfu['sqerror'].mean(0)); su = np.sqrt(pfu['spread'].mean(0))
        pf_u_rmse_s[si, wi] = ru; pf_u_ratio_s[si, wi] = (ru / su).mean()

        # re-tune: walk jitter high -> low, stop at collapse onset, keep the tracking jitter closest to ratio 1
        best_m, best_r, best_ratio, best_gap = None, None, None, np.inf
        for m in sorted(jitter_mults, reverse=True):
            cfg.perturb_std = base * m
            pf = run_pf(obs, truth, obs_idx, h, seed=seed)
            r = np.sqrt(pf['sqerror'].mean(0)); sp = np.sqrt(pf['spread'].mean(0))
            if best_m is not None and r.mean() > 0.25 * clim:   # collapse onset; lowering jitter only worsens it
                break
            ratio = (r / sp).mean()
            if abs(ratio - 1) < best_gap:                       # tracking; keep closest to RMSE/spread = 1
                best_m, best_r, best_ratio, best_gap = m, r, ratio, abs(ratio - 1)
        pf_rmse_s[si, wi] = best_r
        pf_mult_s[si, wi] = best_m
        pf_ratio_s[si, wi] = best_ratio

# --- average over seeds ---
en_rmse = en_rmse_s.mean(0); pf_rmse = pf_rmse_s.mean(0); pf_u_rmse = pf_u_rmse_s.mean(0)
pf_u_ratio = pf_u_ratio_s.mean(0); pf_mult = pf_mult_s.mean(0); fc_std = fc_std_s.mean(0)
pf_ratio = pf_ratio_s.mean(0)                  # achieved re-tuned PF calibration
excess = ((en_rmse_s - pf_rmse_s) / pf_rmse_s * 100).mean(0)   # per-seed % excess, then averaged -> (W,3)

w = np.array(windows)
EN, PF, REF, UNT = '#c0392b', '#2471a3', '#7f8c8d', '#884ea0'
VARC = ['#8e44ad', '#16a085', '#e67e22']       # colours for x, y, z lines
labels = ['x', 'y', 'z']

# ---------- figure 1: RMSE (a) + EnKF excess over PF (b), in one png ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))

a1.plot(w, en_rmse.mean(1), 'o-', color=EN, lw=2, label='EnKF')
a1.plot(w, pf_rmse.mean(1), 's-', color=PF, lw=2, label='PF')
a1.set_yscale('log'); a1.set_ylim(0.2, 12); a1.set_xticks(w)
a1.set_xlabel('assimilation window  (steps between updates)'); a1.set_ylabel('RMSE')
a1.set_title('(a) RMSE (mean over x, y, z)', pad=30)
a1.secondary_xaxis('top', functions=(lambda v: v * cfg.dt, lambda v: v / cfg.dt)).set_xlabel('observation interval  Δt')
a1.legend(); a1.grid(alpha=0.3, which='both')

for j in range(3):
    a2.plot(w, excess[:, j], 'o-', lw=2, color=VARC[j], label=labels[j])
a2.set_xticks(w); a2.set_xlabel('assimilation window  (steps between updates)')
a2.set_ylabel('EnKF RMSE excess over PF  (%)'); a2.set_title('(b) EnKF excess over PF', pad=30)
a2.secondary_xaxis('top', functions=(lambda v: v * cfg.dt, lambda v: v / cfg.dt)).set_xlabel('observation interval  Δt')
a2.legend(); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('figs/fig_window.png', dpi=145)

# ---------- figure 2: PF fragility ----------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.plot(w, pf_u_rmse.mean(1), 'v-', color=UNT, lw=2, label='PF, base jitter (not re-tuned)')
a1.plot(w, pf_rmse.mean(1), 's-', color=PF, lw=2, label='PF, jitter re-tuned per window')
a1.axhline(clim, ls='--', color=REF, lw=1.3)
a1.text(w[0], clim * 1.06, f'climatological scale ≈ {clim:.1f}', fontsize=8.5, color=REF)
a1.set_yscale('log'); a1.set_ylim(0.2, 20); a1.set_xticks(w)
a1.set_xlabel('assimilation window  (steps between updates)'); a1.set_ylabel('PF RMSE  (mean over x, y, z)')
a1.set_title('(a) PF RMSE: base jitter vs re-tuned')
a1.legend(); a1.grid(alpha=0.3, which='both')

a2.axhspan(0.95, 1.05, color='green', alpha=0.12, label='target  1 ± 0.05')
a2.axhline(1, ls='--', color='k', lw=1)
a2.plot(w, pf_ratio, 's-', color=PF, lw=2, label='re-tuned PF')
for x, y, m in zip(w, pf_ratio, pf_mult):
    a2.annotate(f'×{m:.1f}', (x, y), textcoords='offset points', xytext=(0, 8), ha='center', fontsize=8.5, color=PF)
a2.set_xticks(w); a2.set_xlabel('assimilation window  (steps between updates)')
a2.set_ylabel('PF RMSE / spread  (re-tuned)')
a2.set_title('(b) Re-tuned PF calibration')
a2.legend(loc='best'); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig('figs/fig_pf_fragility.png', dpi=145)

# ---------- figure 3: mechanism (prior spread + xy/xz/yz scatters) ----------
planes = [(0, 1, 'x', 'y'), (0, 2, 'x', 'z'), (1, 2, 'y', 'z')]

def cloud(ax, e, i, j, show_legend=False):             # scatter a prior ensemble in one plane
    a, b = e[:, i] - e[:, i].mean(), e[:, j] - e[:, j].mean()    # centre on its own mean
    ax.scatter(a, b, s=4, alpha=0.35, color=EN, edgecolors='none')
    cov = np.cov(np.c_[a, b].T)                         # best-fit Gaussian to compare against
    val, vec = np.linalg.eigh(cov)
    ang = np.degrees(np.arctan2(vec[1, 0], vec[0, 0]))
    ax.add_patch(Ellipse((0, 0), 4 * np.sqrt(val[0]), 4 * np.sqrt(val[1]),
                 angle=ang, fill=False, color='k', lw=1.3, label='Gaussian fit (2σ)'))
    ax.set_xlim(-12, 12); ax.set_ylim(-12, 12); ax.set_aspect('equal')
    if show_legend:
        ax.legend(fontsize=8, loc='upper right')

fig = plt.figure(figsize=(12, 11))
gs = fig.add_gridspec(3, 3, height_ratios=[0.7, 1, 1])
ag = fig.add_subplot(gs[0, :])
ag.plot(w, fc_std, 'o-', color=EN, lw=2)
ag.set_xticks(w); ag.set_xlabel('assimilation window  (steps between updates)')
ag.set_ylabel('forecast-ensemble spread'); ag.set_title('(a) Prior spread grows with window'); ag.grid(alpha=0.3)

for col, (i, j, xl, yl) in enumerate(planes):
    a5 = fig.add_subplot(gs[1, col]); cloud(a5, snaps[5], i, j, show_legend=(col == 0))
    a25 = fig.add_subplot(gs[2, col]); cloud(a25, snaps[25], i, j)
    a5.set_title(f'window=5  ({xl}–{yl})', fontsize=9)
    a25.set_title(f'window=25  ({xl}–{yl})', fontsize=9)
    a5.set_xlabel(xl); a5.set_ylabel(yl); a25.set_xlabel(xl); a25.set_ylabel(yl)
fig.tight_layout(); fig.savefig('figs/fig_mechanism.png', dpi=145)

print('saved figs/fig_window.png, figs/fig_pf_fragility.png, figs/fig_mechanism.png')
print('PF jitter multiple (averaged over seeds):', dict(zip(windows, np.round(pf_mult, 2))))
print('re-tuned PF RMSE/spread (target 1 ± 0.05):', dict(zip(windows, np.round(pf_ratio, 3))))