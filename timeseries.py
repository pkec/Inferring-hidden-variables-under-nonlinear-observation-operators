import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg, rk4
from partfilt import run_pf
from enkf import run_enkf
from enkf_qr import run_enkf_qr
from enkf_ienkf import run_enkf_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Added:   os.makedirs for the output directory
#      Changed: figs output -> figs/results
# 3.15 Added:   two remedy figures at ALPHA = 0.5 — fig_timeseries_qr.png and
#               fig_timeseries_ienkf.png, each PF vs EnKF vs one remedy on the same
#               truth/obs; the alpha=0 figure is untouched
#      Added:   fixed-SNR handling for alpha > 0 (obs noise = 0.25*std(h_a(truth)) per
#               axis, same recipe as the sweep); obs marked as assimilation-time ticks
#               because observed values live in h-space and cannot overlay state panels
#      Added:   bottom row of the remedy figures carries both gaps (EnKF - PF and
#               remedy - PF) with a mean-excess label per filter
#      Changed: pf_calibrated takes the observation operator as an argument (default h)
#               so the PF can be re-calibrated under h_alpha
# ============================================================

os.makedirs('figs/results', exist_ok=True)

# truth: spin up, then the assimilation trajectory (same as the sweep)
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):
    s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps):
    truth[k + 1] = rk4(truth[k], cfg.dt)
t_truth = np.arange(cfg.n_steps + 1) * cfg.dt
clim = truth.std(axis=0).mean()       # no-skill scale, used to detect PF collapse

h = lambda x: x                       # linear observation operator (alpha = 0)
windows = [15]                 # dense / medium / sparse observation cadence
base = cfg.perturb_std.copy()
jitter_mults = np.geomspace(0.3, 8, 18)
t0, t1 = 10.0, 50.0                   # zoom window (time units) for a readable view
labels = ['x', 'y', 'z']
EN, PF, TRU, OBS, DIF = '#c0392b', '#2471a3', '#333333', '#e67e22', '#8e44ad'

def pf_calibrated(obs, obs_idx, hop=h):  # re-tune PF jitter per window, same rule as the sweep
    best = None
    for m in sorted(jitter_mults, reverse=True):
        cfg.perturb_std = base * m
        pf = run_pf(obs, truth, obs_idx, hop, seed=cfg.seed)
        r = np.sqrt(pf['sqerror'].mean(0)); sp = np.sqrt(pf['spread'].mean(0))
        if best is not None and r.mean() > 0.25 * clim:   # collapse onset; stop lowering jitter
            break
        gap = abs((r / sp).mean() - 1)
        if best is None or gap < best[1]:                 # keep the tracking jitter closest to ratio 1
            best = (pf, gap)
    return best[0]

fig, ax = plt.subplots(4, 3, figsize=(16.5, 9.5), sharex='col', sharey='row')

for c, oe in enumerate(windows):
    cfg.obs_every = oe
    obs_idx = np.arange(oe, cfg.n_steps + 1, oe)
    rng = np.random.default_rng(cfg.seed)
    obs = truth[obs_idx] + rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
    t_obs = obs_idx * cfg.dt
    cfg.perturb_std = base                                # EnKF uses base jitter (it does not collapse)
    en = run_enkf(obs, truth, obs_idx, h, seed=cfg.seed)
    pf = pf_calibrated(obs, obs_idx)                      # PF uses the per-window calibrated jitter

    win = (t_obs >= t0) & (t_obs <= t1)
    wint = (t_truth >= t0) & (t_truth <= t1)
    for r in range(3):
        a = ax[r, c]
        a.plot(t_truth[wint], truth[wint, r], '-', color=TRU, lw=1.3, label='truth', zorder=1)
        a.scatter(t_obs[win], obs[win, r], s=14, color=OBS, alpha=0.7, label='observations', zorder=2)
        a.plot(t_obs[win], en['en_mean'][win, r], '-', color=EN, lw=1.6, label='EnKF mean', zorder=3)
        a.plot(t_obs[win], pf['en_mean'][win, r], '-', color=PF, lw=1.6, label='PF mean', zorder=3)
        if c == 0:
            a.set_ylabel(labels[r])
        a.grid(alpha=0.3)

    e_en = np.sqrt(en['sqerror'].mean(1))                 # per-step RMS error over x,y,z
    e_pf = np.sqrt(pf['sqerror'].mean(1))
    diff = e_en - e_pf                                    # >0 means EnKF is worse than PF
    excess = ((np.sqrt(en['sqerror'].mean(0)) - np.sqrt(pf['sqerror'].mean(0)))
              / np.sqrt(pf['sqerror'].mean(0)) * 100).mean()
    ad = ax[3, c]
    ad.axhline(0, ls='--', color='#777', lw=1)
    ad.plot(t_obs[win], diff[win], '-', color=DIF, lw=1.4)
    ad.text(0.02, 0.92, f'mean EnKF excess {excess:.0f}%', transform=ad.transAxes,
            fontsize=8.5, va='top', color=DIF)
    ad.set_xlabel('time'); ad.grid(alpha=0.3)

ax[0, 0].legend(loc='upper left', fontsize=8, ncol=2)
ax[3, 0].set_ylabel('RMS error\nEnKF − PF')
for c, oe in enumerate(windows):
    ax[0, c].set_title(f'window {oe}   (Δt = {oe * cfg.dt:.2f})')
fig.tight_layout()
fig.savefig('figs/results/fig_timeseries.png', dpi=130)
print('saved figs/results/fig_timeseries.png')

# ---- remedy figures: PF vs EnKF vs one remedy, at alpha > 0 ----
ALPHA = 0.5                                     # remedies only separate from the EnKF for alpha > 0
h_a = lambda x: x + ALPHA * x ** 2
if cfg.noise_mode == 'fixed_snr':
    cfg.obs_std = 0.25 * h_a(truth).std(axis=0) # noise tracks h_a's output scale (sweep recipe)
QRC, IEC = '#27ae60', '#8e44ad'                 # QR-EnKF green, IEnKF purple

res = {}
for oe in windows:
    cfg.obs_every = oe
    obs_idx = np.arange(oe, cfg.n_steps + 1, oe)
    rng = np.random.default_rng(cfg.seed)
    obs = h_a(truth[obs_idx]) + rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
    cfg.perturb_std = base                      # all EnKF-family filters use base jitter
    en = run_enkf(obs, truth, obs_idx, h_a, seed=cfg.seed)
    qr = run_enkf_qr(obs, truth, obs_idx, h_a, seed=cfg.seed)
    ie = run_enkf_ienkf(obs, truth, obs_idx, h_a, seed=cfg.seed)
    pf = pf_calibrated(obs, obs_idx, h_a)       # PF re-calibrated under the curved operator
    res[oe] = (obs_idx * cfg.dt, en, pf, {'QR-EnKF': (qr, QRC), 'IEnKF': (ie, IEC)})

for rname, fname in [('QR-EnKF', 'fig_timeseries_qr.png'), ('IEnKF', 'fig_timeseries_ienkf.png')]:
    fig, ax = plt.subplots(4, 3, figsize=(16.5, 9.5), sharex='col', sharey='row')
    for c, oe in enumerate(windows):
        t_obs, en, pf, rem = res[oe]
        rmy, rcol = rem[rname]
        win = (t_obs >= t0) & (t_obs <= t1)
        wint = (t_truth >= t0) & (t_truth <= t1)
        for r in range(3):
            a = ax[r, c]
            a.plot(t_truth[wint], truth[wint, r], '-', color=TRU, lw=1.3, label='truth', zorder=1)
            a.vlines(t_obs[win], 0, 0.05, transform=a.get_xaxis_transform(),
                     color=OBS, alpha=0.6, lw=1, label='obs times')
            a.plot(t_obs[win], en['en_mean'][win, r], '-', color=EN, lw=1.6, label='EnKF mean', zorder=3)
            a.plot(t_obs[win], pf['en_mean'][win, r], '-', color=PF, lw=1.6, label='PF mean', zorder=3)
            a.plot(t_obs[win], rmy['en_mean'][win, r], '-', color=rcol, lw=1.6,
                   label=f'{rname} mean', zorder=4)
            if c == 0:
                a.set_ylabel(labels[r])
            a.grid(alpha=0.3)

        e_pf = np.sqrt(pf['sqerror'].mean(1))             # per-step RMS error over x,y,z
        pf_axis = np.sqrt(pf['sqerror'].mean(0))          # per-axis RMSE, denominator of the excess
        ad = ax[3, c]
        ad.axhline(0, ls='--', color='#777', lw=1)
        for flt, col, tag, ty in [(en, EN, 'EnKF', 0.92), (rmy, rcol, rname, 0.78)]:
            ad.plot(t_obs[win], (np.sqrt(flt['sqerror'].mean(1)) - e_pf)[win], '-', color=col, lw=1.2)
            exc = ((np.sqrt(flt['sqerror'].mean(0)) - pf_axis) / pf_axis * 100).mean()
            ad.text(0.02, ty, f'{tag} excess {exc:.0f}%', transform=ad.transAxes,
                    fontsize=8.5, va='top', color=col)
        ad.set_xlabel('time'); ad.grid(alpha=0.3)

    ax[0, 0].legend(loc='upper left', fontsize=8, ncol=2)
    ax[3, 0].set_ylabel('RMS error\nfilter − PF')
    for c, oe in enumerate(windows):
        ax[0, c].set_title(f'window {oe}   (Δt = {oe * cfg.dt:.2f},  α = {ALPHA})')
    fig.tight_layout()
    fig.savefig(f'figs/results/{fname}', dpi=130)
    print(f'saved figs/results/{fname}')