
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 29):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg, PERTURB_BASELINE_ALPHA

fs.use()

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.45 Changed: sizing, fonts and DPI now come from figstyle rather than a literal
#      (6.2 x n, 4.4) figsize and matplotlib defaults — this was the last figure still on
#      raw defaults, so its labels and ticks did not match any other figure in the report.
#      Added: a legend. EnKF/PF were labelled but ax.legend() was never called, so the red
#      and blue series went unexplained. It sits ABOVE the axes, as in
#      stage4_excess_gap_alpha — the model figure — via fs.leg_above; a two-mode figure gets
#      one shared legend above both panels instead of the same key twice.
#      Added: LABEL_SCALE, multiplying every figstyle point size so this reads at the same
#      size on screen as the model figure.
#      Fixed: the +/-1 std error bars promised by 2.14 are actually drawn. en_sd/pf_sd were
#      computed and then never passed to errorbar, so the picks have been plotted bare since
#      2.14 — set SHOW_ERRORBARS = False for the old behaviour.
#      Changed: the grid-ceiling annotation shortened to 'grid ceiling'; the untrustworthy-
#      boundary-hit warning moved to the caption, since the full sentence overran the frame
#      once the figure was exported at print width.
#      Added: (a)/(b) panel letters when both noise modes are present.
# 5.44 Added: the unscanned baseline alpha is marked — a shaded vertical band, hollow
#      markers and a note. error_sweep.py stopped scanning at alpha=0 in 5.44 and pins the
#      multiplier at 1.0 there, which is not one of the nine grid levels; unmarked, that
#      point looks like a scan result sitting off-grid, which is the opposite of what it is.
#      The alpha is read from the npz (baseline_alpha) rather than assumed, so a file written
#      under a different rule still plots honestly.
#      Changed: the boundary-hit ring test skips the baseline alpha. A pinned multiplier
#      cannot be a boundary hit, and ringing it would report a scan failure that never
#      happened.
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 2.14 Added:   +/-1 std (across seeds) error bars on the multiplier picks
#      Changed: the plotted multiplier is now the seed mean, so boundary rings flag the mean pick
# 2.8  Changed: read variable-jitter results only (fixed jitter has nothing to diagnose)
# 2.7  Changed: read window from cfg, tag input/output by window (w{obs_every})
# 2.5  Changed: GRID ceiling 12 -> 30 to match stage2.py (geomspace 0.3..30, 9 pts)
# 2.2  created — calibrated jitter vs alpha, with boundary-hit flags
# ============================================================

# must match the grid used in stage2.py
GRID = np.geomspace(0.3, 30, 9)
FLOOR, CEIL = GRID[0], GRID[-1]
W = cfg.obs_every                                    # window set by run.py (L63_OBS_EVERY)
SHOW_ERRORBARS = False                                # +/-1 std across seeds on each pick

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
ANN = scaled(getattr(fs, 'ANN', {'fontsize': 7.5}))       # the grid floor/ceiling notes
_tk = plt.rcParams['xtick.labelsize']                     # figstyle sets this in fs.use()
TICKSIZE = (_tk if isinstance(_tk, (int, float)) else 8) * LABEL_SCALE

os.makedirs('figs/diagnostic', exist_ok=True)
modes = [m for m in ('fixed_R', 'fixed_snr') if os.path.exists(f'data/stage2_results_w{W}_{m}_variable.npz')]
if not modes:
    raise SystemExit(f'no data/stage2_results_w{W}_*_variable.npz found — run stage2.py (variable) first')

EN, PF = '#c0392b', '#2471a3'
# one mode -> the model figure's own geometry; two -> full width, one shared legend above.
FIGSIZE = fs.size(0.5, 0.80) if len(modes) == 1 else fs.size(1.0, 0.40)
fig, axes = plt.subplots(1, len(modes), figsize=FIGSIZE, squeeze=False)

for pi, (ax, mode) in enumerate(zip(axes[0], modes)):
    d = np.load(f'data/stage2_results_w{W}_{mode}_variable.npz')
    a, en, pf = d['alphas'], d['en_mult'], d['pf_mult']
    en_sd = d['en_mult_std'] if 'en_mult_std' in d.files else np.zeros_like(a)   # seed spread of the pick
    pf_sd = d['pf_mult_std'] if 'pf_mult_std' in d.files else np.zeros_like(a)

    # Which alpha, if any, was NOT scanned. Read from the file so a pre-5.44 npz — where every
    # alpha WAS scanned — is plotted without a band it does not deserve.
    base_a = float(d['baseline_alpha']) if 'baseline_alpha' in d.files else None
    pinned = np.isclose(a, base_a) if base_a is not None else np.zeros_like(a, dtype=bool)

    for g in GRID:                                   # faint lines at each scannable jitter level
        ax.axhline(g, color='#bbb', lw=0.5, zorder=0)
    ax.axhline(FLOOR, color='#888', ls='--', lw=1)
    ax.axhline(CEIL, color='#888', ls='--', lw=1)
    # short forms only — the full 'boundary hits untrustworthy' sentence overran the frame at
    # print width and now lives in the caption (see the module docstring).
    ax.text(a[0], CEIL * 1.04, 'grid ceiling', color='#888', **ANN)
    ax.text(a[0], FLOOR * 0.78, 'grid floor', color='#888', **ANN)

    # yerr is the seed spread of the pick — computed since 2.14, passed since 5.45.
    ax.errorbar(a, en, yerr=en_sd if SHOW_ERRORBARS else None,
                fmt='o-', color=EN, lw=2, ms=4, capsize=2, label='EnKF')
    ax.errorbar(a, pf, yerr=pf_sd if SHOW_ERRORBARS else None,
                fmt='s-', color=PF, lw=2, ms=4, capsize=2, label='PF')

    for series, col in ((en, EN), (pf, PF)):          # ring the boundary hits
        # A pinned multiplier is not a scan result, so it cannot be a boundary hit — excluding
        # it stops the figure reporting a scan failure at an alpha where no scan ran.
        hit = (np.isclose(series, FLOOR) | np.isclose(series, CEIL)) & ~pinned
        ax.scatter(a[hit], series[hit], s=130, facecolors='none', edgecolors=col, linewidths=2, zorder=5)
        # hollow marker on the pinned point: same shape, no fill, so it reads as "given" not "found"
        ax.scatter(a[pinned], series[pinned], s=70, facecolors='white', edgecolors=col,
                   linewidths=1.6, zorder=6)

    ax.set_yscale('log'); ax.set_yticks(GRID)
    ax.set_yticklabels([f'{g:.2g}' for g in GRID])
    ax.tick_params(labelsize=TICKSIZE)
    ax.set_xlabel(r'$\alpha$', **LAB)
    ax.set_ylabel('multiplier' if pi == 0 else '', **LAB)
    if len(modes) > 1:
        fs.panel_letter(ax, pi)                       # name the modes in the caption

fig.tight_layout()

# Legend above the axes, as in stage4_excess_gap_alpha. With two panels the key is identical
# in both, so one figure-level legend replaces two.
if len(modes) == 1:
    axes[0, 0].legend(**scaled(fs.leg_above(ncol=2)))
else:
    h, l = axes[0, 0].get_legend_handles_labels()
    kw = scaled(fs.leg_above(ncol=2))
    for k in ('loc', 'bbox_to_anchor', 'bbox_transform', 'ncol'):   # re-anchored in fig coords
        kw.pop(k, None)
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, **kw)

out = f'figs/diagnostic/jitter_diagnostic_w{W}.png'
fig.savefig(out, dpi=fs.DPI, bbox_inches='tight')
print(f'saved {out}')
print(f'baseline alpha (unscanned, multiplier pinned at 1.0): {PERTURB_BASELINE_ALPHA}')