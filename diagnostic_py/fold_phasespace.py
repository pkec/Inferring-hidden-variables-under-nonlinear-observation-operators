# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.38 Fixed: the (a) letter was hidden under the panel-(a) legend — both want the top-left
#      corner. The legend moves above the axes; the letter stays put, since a panel letter
#      in a predictable place is worth more than a legend in one.
#      Changed: legend labels drop the fold value ('fold, $\alpha$=0.1  ($x=$-5.0)' ->
#      '$\alpha$=0.1'), which is what made them too wide to sit anywhere but inside the axes.
#      The values now print to stdout with whether each fold falls inside the range of x
#      and of z — the three geometric claims the figure exists to support.
# 5.37 Added: (a)/(b) drawn into the merged figure. The two standalone PNGs still take their
#      letters from the LaTeX subcaption.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 6.5in panels shrank by 0.49 at 0.48\textwidth, taking
#      14pt labels to 6.9pt and the 8pt legend to 3.9pt.
#      Removed: a 1x2 subplots() call that was created, never drawn on and never saved.
# 6.8  created — attractor with the operator fold overlaid, for the Methods operator-family
#      section; companion to ienkf_lobe_phasespace.py.
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

os.makedirs('figs/results', exist_ok=True)
fs.use()
FRAC = 0.48                                       # two panels across the text width
ALPHAS = [0.1, 0.5, 1.0]
COL = ['#2471a3', '#e67e22', '#c0392b']           # one per alpha
ATT = '#999999'                                   # attractor trace

# --- truth trajectory: same recipe as the sweep, transient discarded ---
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):
    s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps):
    truth[k + 1] = rk4(truth[k], cfg.dt)

x, y, z = truth[:, 0], truth[:, 1], truth[:, 2]

folds = [-1.0 / (2 * a) for a in ALPHAS]          # vertex of h_alpha, negative for all alpha>0

def panel_attractor(ax):
    """(a) attractor in x-z, with the fold marked in both coordinates."""
    ax.plot(x, z, '-', color=ATT, lw=0.3, alpha=0.7, zorder=1)
    for a, f, c in zip(ALPHAS, folds, COL):
        ax.axvline(f, ls='--', color=c, lw=1.5, zorder=3, label=rf'$\alpha$={a}')
        ax.axhline(f, ls=':', color=c, lw=1.2, zorder=2)
    ax.axhline(z.min(), ls='-', color='#333', lw=1.0, zorder=4)
    ax.annotate(r'$x_{3\, min}$', xy=(x.min(), z.min()),
                xytext=(4, 6), textcoords='offset points', color='#333', **fs.ANN)
    ax.set_ylim(min(folds) - 2, z.max() + 3)
    ax.set_xlabel('$x_1$', **fs.LAB); ax.set_ylabel('$x_3$', **fs.LAB)
    ax.grid(alpha=0.3); ax.legend(**fs.leg_above(ncol=len(ALPHAS)))


def panel_operator(ax):
    """(b) the operator over the range of x, with the vertex marked per alpha."""
    xs = np.linspace(x.min() - 1, x.max() + 1, 600)
    for a, f, c in zip(ALPHAS, folds, COL):
        ax.plot(xs, xs + a * xs ** 2, '-', color=c, lw=1.8, label=rf'$\alpha$={a}')
        ax.plot(f, f + a * f ** 2, 'o', color=c, ms=5, zorder=5)
    ax.axhline(0, color='#333', lw=0.8)
    ax.set_xlabel('$x_1$', **fs.LAB); ax.set_ylabel(r'$h_\alpha(x_1)$', **fs.LAB)
    ax.grid(alpha=0.3); ax.legend(**fs.leg_above(ncol=2))


print(f'z on the attractor: [{z.min():.2f}, {z.max():.2f}]   '
      f'x: [{x.min():.2f}, {x.max():.2f}]')
for a, f in zip(ALPHAS, folds):
    print(f'  alpha={a:<4} fold x = -1/(2a) = {f:6.2f}   '
          f"{'inside' if x.min() < f < x.max() else 'outside'} the range of x, "
          f"{'below' if f < z.min() else 'inside'} the range of z")

PANELS = [('attractor', panel_attractor), ('operator', panel_operator)]

# one standalone PNG per panel — these are the write-up figures; the (a)/(b) letters come
# from the LaTeX subfigure/subcaption
for stem, draw in PANELS:
    fig, ax = plt.subplots(figsize=fs.size(FRAC, 0.80))
    draw(ax)
    fs.save(fig, f'figs/results/fold_phasespace_{stem}.png')

# combined 1x2, kept as the backup / log view
fig, axes = plt.subplots(1, 2, figsize=fs.size(1.0, 0.40))
for i, (_, draw) in enumerate(PANELS):
    draw(axes[i])
    fs.panel_letter(axes[i], i)      # no subcaptions on a merged figure to carry these
fs.save(fig, 'figs/results/fold_phasespace.png')