import numpy as np
import matplotlib.pyplot as plt
from config import cfg, rk4
from partfilt import run_pf
from enkf import run_enkf

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
windows = [5, 15, 25]                 # dense / medium / sparse observation cadence
base = cfg.perturb_std.copy()
jitter_mults = np.geomspace(0.3, 8, 18)
t0, t1 = 10.0, 30.0                   # zoom window (time units) for a readable view
labels = ['x', 'y', 'z']
EN, PF, TRU, OBS, DIF = '#c0392b', '#2471a3', '#333333', '#e67e22', '#8e44ad'

def pf_calibrated(obs, obs_idx):      # re-tune PF jitter per window, same rule as the sweep
    best = None
    for m in sorted(jitter_mults, reverse=True):
        cfg.perturb_std = base * m
        pf = run_pf(obs, truth, obs_idx, h, seed=cfg.seed)
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
fig.savefig('figs/fig_timeseries.png', dpi=130)
print('saved figs/fig_timeseries.png')