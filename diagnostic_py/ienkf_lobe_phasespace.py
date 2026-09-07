# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.43 Fixed:   the legend sat top-left, on top of the (a) letter and on the left wing of the
#      attractor. The merged figure now carries one legend above the row (both panes draw
#      the same three lines) and the standalone panes put theirs above the axes.
# 5.42 Added: (a)/(b) drawn into the merged figure, which makes it usable as the write-up
#      figure rather than only as a log view.
#      Removed: the suptitle. alpha, seed and window belong in the caption; the per-panel
#      t-range and RMS error stay, since they are what separates the two windows.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 13.5in figure shrank by 0.50 at \textwidth.
#      Added: one standalone PNG per pane, which the script did not emit before — the panel
#      titles carrying the t-range and RMS error move to the LaTeX subcaption, and the
#      combined 1x2 keeps them as the backup / log view.
# 6.8  created — phase-space view of the IEnKF wrong-lobe episode; companion to
#      fold_phasespace.py. Failure onset uses the same criterion as stage3_ienkf_seeds.py.
# ============================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg, rk4
from enkf_ienkf import run_enkf_ienkf

os.makedirs('figs/results', exist_ok=True)

fs.use()
FRAC = 0.48                    # two panels across the text width
ALPHA = 0.5
SEED = 1                       # the seed that sustains failure in stage3_ienkf_seeds.py
h_a = lambda x: x + ALPHA * x ** 2
FOLD = -1.0 / (2 * ALPHA)
FAIL_FRAC, HOLD = 0.25, 3      # failure criterion, matching stage3_ienkf_seeds.py
WIN = 30                       # cycles either side of the onset to plot
TRU, IEC = '#333333', '#8e44ad'

# --- truth and observations: same recipe as the sweep ---
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):
    s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps):
    truth[k + 1] = rk4(truth[k], cfg.dt)
clim = truth.std(axis=0).mean()

obs_idx = np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every)
t_obs = obs_idx * cfg.dt
obs_std = 0.25 * h_a(truth).std(axis=0) if cfg.noise_mode == 'fixed_snr' else cfg.obs_std
obs = h_a(truth[obs_idx]) + np.random.default_rng(cfg.seed).normal(
    0, 1, (len(obs_idx), 3)) * obs_std

res = run_enkf_ienkf(obs, truth, obs_idx, h_a, seed=SEED, obs_std=obs_std)
mean = res['en_mean']                                  # (n_obs, 3) IEnKF analysis mean
err = np.sqrt(res['sqerror']).mean(1)                  # (n_obs,) per component, then averaged
spread = np.sqrt(res['spread']).mean(1)                # (n_obs,) same, from the variance
tao = truth[obs_idx]

# --- locate the sustained-failure onset, and a healthy stretch well clear of it ---
bad = err > FAIL_FRAC * clim
onset = next((k for k in range(len(bad) - HOLD + 1) if bad[k:k + HOLD].all()), None)
if onset is None:
    raise SystemExit(f'seed {SEED} never sustained failure at alpha={ALPHA} — try another seed')
fail = slice(onset, min(onset + WIN, len(err)))
if onset >= 2 * WIN:                                  # room before the episode
    good = slice(onset - 2 * WIN, onset - WIN)
else:                                                 # onset too early — take a stretch after it
    start = min(onset + 2 * WIN, len(err) - WIN)
    good = slice(start, start + WIN)
if len(err[good]) == 0:
    raise SystemExit('run too short to show a healthy window alongside the failure')

PANES = [(good, 'tracking'), (fail, 'wrong_lobe')]


def draw(ax, sl, legend):
    ax.plot(tao[sl, 0], tao[sl, 2], '-', color=TRU, lw=1.6, alpha=0.8, label='truth')
    ax.plot(mean[sl, 0], mean[sl, 2], '--', color=IEC, lw=1.6, label='IEnKF mean')
    ax.axvline(FOLD, ls='--', color='#c0392b', lw=1.2, label=r'fold $x_1=-1/(2\alpha)$')
    ax.set_xlabel('$x_1$', **fs.LAB); ax.set_ylabel('$x_3$', **fs.LAB); ax.grid(alpha=0.3)
    if legend:
        # above the axes, not in them: the attractor fills all four corners of this view, so
        # there is no interior spot a legend can take without covering a wing
        ax.legend(**fs.leg_above(ncol=2))


# one standalone PNG per pane — these are the write-up figures; the (a)/(b) letters and the
# t-range and RMS error that used to sit in the panel titles move to the LaTeX subcaption.
for sl, stem in PANES:
    f1, a1 = plt.subplots(figsize=fs.size(FRAC, 0.85))
    draw(a1, sl, legend=(stem == 'tracking'))
    fs.save(f1, f'figs/results/ienkf_lobe_{stem}_a{ALPHA}_w{cfg.obs_every}.png')
    print(f'  {stem}: t = {t_obs[sl][0]:.1f}-{t_obs[sl][-1]:.1f}, '
          f'RMS error {err[sl].mean():.2f}')

# combined 1x2, kept as the backup / log view — titles retained here only
fig, axes = plt.subplots(1, 2, figsize=fs.size(1.0, 0.42), sharex=True, sharey=True)
for i, (sl, stem) in enumerate(PANES):
    draw(axes[i], sl, legend=False)      # both panes draw the same three lines
    fs.panel_letter(axes[i], i)          # no subcaptions on a merged figure to carry these
    axes[i].set_title(f'$t$ = {t_obs[sl][0]:.1f}--{t_obs[sl][-1]:.1f}, '
                      f'RMS error {err[sl].mean():.2f}')
axes[1].set_ylabel('')
fig.tight_layout()
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, **fs.leg_fig(ncol=3, y=1.03))   # clear of the panel titles, and of (a)
fs.save(fig, f'figs/results/ienkf_lobe_phasespace_a{ALPHA}_w{cfg.obs_every}.png', tight=False)

print(f'\nonset cycle {onset} (t={t_obs[onset]:.2f}), failure = err > {FAIL_FRAC}*clim '
      f'for {HOLD} cycles')
print(f'  healthy window  RMS error {err[good].mean():6.2f}   spread {spread[good].mean():.2f}')
print(f'  failure window  RMS error {err[fail].mean():6.2f}   spread {spread[fail].mean():.2f}')