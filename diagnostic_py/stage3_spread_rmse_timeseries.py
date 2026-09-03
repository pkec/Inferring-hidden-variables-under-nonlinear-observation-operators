
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg, rk4_vec
from enkf import EnKF
from enkf_ienkf import EnKF_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.45 Changed: legends moved above the axes, matching stage4_excess_gap_alpha — the model
#      figure for this family. Single panels use fs.leg_above; the merged 1x2 and the
#      backlog grid carry ONE figure-level legend centred above all panels rather than one
#      legend per panel. The old loc='upper right' legend sat on top of the spikes.
#      Changed: labels, ticks and legend text scale by LABEL_SCALE (see below), so the
#      exported PNG reads at the same size as stage4_excess_gap_alpha, which prints at
#      0.5\textwidth against this file's 0.48-per-panel/1.0-merged.
# 5.37 Added: (a)/(b) drawn into the per-alpha merged figure, and letters across the backlog
#      grid. The standalone panels still take their letters from the LaTeX subcaption.
# 5.27 Added: ie_spread_rmse_a{alpha}_w{W}.png — spread and RMSE side by side at ONE alpha,
#      which is how the write-up uses them. One image with one caption replaces two 0.42
#      subfigures with two subcaptions, and the panels widen from 7.1cm to ~8.5cm.
#      Changed: the two existing figures now save through fs.save, so DPI and layout are
#      set in one place rather than repeated per call.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 6.5in panels shrank by 0.49 at 0.48\textwidth.
#      Changed: aspect 0.62 rather than the figstyle default — these are time series, so they
#      want to be wide and short.
# 5.10 Changed: report styling to match Stage 1/2 — axis labels Arial 14pt, legends Arial 11pt
#      (AX/FS/LEG), the per-column alpha titles and the suptitle dropped; alpha is carried by
#      the filename on the split PNGs instead
#      Added: one PNG per panel (ie_{spread,rmse}_a{alpha}_w{W}.png); the 2 x n_alpha grid
#      stays as the backup / log view
#      Changed: results are computed once into a dict up front and both the split and the
#      combined figures read from it, so splitting does not re-propagate every filter
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 3.9  created — per-cycle spread & RMSE time series, EnKF vs IEnKF, across alpha
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
ALPHAS = [0.0, 1.0]
FILTERS = [('EnKF', EnKF, '#c0392b'), ('IEnKF', EnKF_ienkf, '#2471a3')]
fs.use()
AX, FS = fs.AX, fs.FS       # sizes come from figstyle; do not re-assert them here
FRAC = 0.48                 # two panels across the text width

# ---- label / legend sizing ---------------------------------------------------------------
# Model figure: stage4_excess_gap_alpha. It prints at 0.5\textwidth, so at a common 9pt its
# labels fill more of the exported PNG than these wider figures do. LABEL_SCALE multiplies
# every figstyle point size — labels, ticks and legend alike — so the two match on screen.
# Set it to 1.0 to go back to plain figstyle sizes (9pt/8pt/7.5pt on the page).
LABEL_SCALE = 1.25

def scaled(kw, s=None):

    s = LABEL_SCALE if s is None else s
    out = dict(kw)
    for k in ('fontsize', 'size'):
        if isinstance(out.get(k), (int, float)):
            out[k] = out[k] * s
    p = out.get('prop')
    if isinstance(p, dict):
        q = dict(p)
        if isinstance(q.get('size'), (int, float)):
            q['size'] = q['size'] * s
        out['prop'] = q
    elif p is not None and hasattr(p, 'get_size'):        # a FontProperties instance
        q = p.copy(); q.set_size(p.get_size() * s); out['prop'] = q
    return out

LAB = scaled(fs.LAB)
_tk = plt.rcParams['xtick.labelsize']                     # figstyle sets this in fs.use()
TICKSIZE = (_tk if isinstance(_tk, (int, float)) else 8) * LABEL_SCALE

def legend_above_fig(fig, ax_src, ncol):

    h, l = ax_src.get_legend_handles_labels()
    kw = scaled(fs.leg_above(ncol=ncol))
    for k in ('loc', 'bbox_to_anchor', 'bbox_transform', 'ncol'):   # re-anchored in fig coords
        kw.pop(k, None)
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=ncol, **kw)

d = np.load('data/l63_twin.npz')
truth, oi = d['truth'], d['obs_idx']
on, osa, al = d['obs_nonlinear'], d['obs_std_alpha'], list(np.round(d['alphas'], 2))
t = oi * cfg.dt

def run(anafn, a):

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

# run every filter once and keep the series; both the split and the combined figures read
# from this, so splitting costs no extra propagation
series = {(a, fname): run(fn, a) for a in ALPHAS for fname, fn, _ in FILTERS}

# (row stem, y-label, index into run()'s (rmse, spread) return)
ROWS = [('spread', 'spread (per cycle)', 1),
        ('rmse',   'RMSE (per cycle)',   0)]

def panel(ax, a, stem, ylab, vi):

    for fname, _, col in FILTERS:
        ax.plot(t, series[(a, fname)][vi], color=col, lw=0.8, label=fname)
    ax.set_ylabel(ylab, **LAB)
    ax.set_xlabel('Time', **LAB)
    ax.tick_params(labelsize=TICKSIZE)
    ax.grid(alpha=0.3)

# one standalone PNG per panel — alpha is in the filename, since the titles are gone
for a in ALPHAS:
    for stem, ylab, vi in ROWS:
        fig, ax = plt.subplots(figsize=fs.size(FRAC, fs.SERIES))
        panel(ax, a, stem, ylab, vi)
        ax.legend(**scaled(fs.leg_above(ncol=len(FILTERS))))
        fs.save(fig, f'figs/diagnostic/ie_{stem}_a{a}_w{cfg.obs_every}.png')

# per-alpha merged 1x2 — spread and RMSE SIDE BY SIDE at one alpha. This is the write-up
# figure: one image with one caption replaces two subfigures with two subcaptions, and the
# panels come out ~8.5cm wide instead of the 7.1cm two 0.42 subfigures allow.
for a in ALPHAS:
    fig, axes = plt.subplots(1, 2, figsize=fs.size(1.0, 0.34))
    for i, (stem, ylab, vi) in enumerate(ROWS):
        panel(axes[i], a, stem, ylab, vi)
        fs.panel_letter(axes[i], i)  # no subcaptions on a merged figure to carry these
    fig.tight_layout()
    legend_above_fig(fig, axes[0], ncol=len(FILTERS))   # both panels share the same two keys
    fs.save(fig, f'figs/diagnostic/ie_spread_rmse_a{a}_w{cfg.obs_every}.png')

# combined 2 x n_alpha grid, kept as the backup / log view
fig, axes = plt.subplots(2, len(ALPHAS), figsize=fs.size(1.0, fs.SERIES), sharex=True, squeeze=False)
for ci, a in enumerate(ALPHAS):
    for ri, (stem, ylab, vi) in enumerate(ROWS):
        ax = axes[ri, ci]
        panel(ax, a, stem, ylab, vi)
        fs.panel_letter(ax, ri * len(ALPHAS) + ci)
        if ci:                                   # only the left column carries the y-label
            ax.set_ylabel('')
        if ri == 0:                              # sharex: only the bottom row carries the x-label
            ax.set_xlabel('')
fig.tight_layout()
legend_above_fig(fig, axes[0, 0], ncol=len(FILTERS))
fs.save(fig, f'figs/diagnostic/ie_spread_rmse_timeseries_w{cfg.obs_every}.png')