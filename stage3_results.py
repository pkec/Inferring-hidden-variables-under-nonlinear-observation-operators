import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.32 Added: (a)/(b)/(c) in the top-left of each panel of the merged figure, via
#      fs.panel_letter. The separate PNGs keep getting their letters from the LaTeX
#      subcaption, so they are left alone.
# 5.31 Changed: the y-label goes back on the y axis. fs.compact() now defaults to pos='y',
#      so the merge keeps most of the width gain (4.21cm of plot box per panel against
#      3.35cm for three separate subfigures) without the label reading as a heading.
# 5.29 Changed: the combined 1x3 becomes the write-up figure and the three separate PNGs
#      become the backlog. Measured plot-box width per panel: 3.35cm as three 0.32 subfigures,
#      4.59cm as one \textwidth figure. The gain is the y-labels moving above the axes via
#      fs.compact — a rotated label costs width equal to the font HEIGHT, not the string
#      length — plus matplotlib controlling the gutters rather than LaTeX \hfill.
#      Changed: the merged figure carries one figure-level legend instead of a per-panel one.
# 5.28 Fixed: the legend sat on top of the data in the excess panel. loc='best' has nowhere
#      to go there — three shaded bands fill the axes — so it is now placed above the axes
#      via fs.leg_above(), where it cannot occlude and does not compress the curves.
#      Changed: series labels 'quad-reg EnKF'/'iterative EnKF' -> 'QR-EnKF'/'IEnKF', the
#      abbreviations the write-up introduces and then uses throughout. Short enough to fit
#      three across a 0.32\textwidth panel, and consistent with the prose.
#      Changed: on the combined 1x3 only the first panel carries the key.
# 5.26 Changed: panel aspect 0.72 -> fs.XY (0.95). At 0.32\textwidth the panels were
#      5.4 x 3.9 cm with only 56% of the height left for data after the y-label,
#      ticks and x-label — the y-label was physically taller than the plot. Now
#      5.4 x 5.2 cm, 66% data. FRAC/ASP are one line if you want 0.48 / fs.WIDE.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 5.5in panels shrank by 0.39 at 0.32\textwidth.
#      Changed: the excess y-label drops ', mean over x, y, z' — see stage2_results.py.
#      Changed: bar-value labels on the excess-removed chart 9pt -> figstyle BAR (6.5pt), which
#      is the printed size rather than the pre-shrink size.
# 5.20 Changed: legends are Arial 11pt via LEG, matching Stage 4/5 (were 8pt and 9pt in the
#      default font, unreadable beside a 14pt axis label). Bar-value labels on the
#      excess-removed chart go 7pt -> 9pt Arial for the same reason — on that figure the
#      printed value IS the datum, since the bars are read off a categorical axis.
#      prop= is used rather than fontsize= because prop overrides it.
# 5.19 Changed: the excess panel and the excess-removed bar chart are now labelled as
#      per-component-then-averaged numbers, which is what error_sweep.py has written into
#      excess_pct / excess_qr_pct / excess_ie_pct since 5.16. No arithmetic changes here —
#      this script only plots what the sweep saved.
#      Added:   a guard against plotting a pre-5.16 npz silently; such a file has no
#               rmse_convention key and its excess keys are the POOLED ratios, so the figure
#               would carry the new label over the old numbers.
# 5.8  Changed: report styling to match Stage 1/2 — axis labels Arial 14pt, all titles and
#               suptitles dropped (Imperial reports caption externally; (a)/(b)/(c) come from
#               the LaTeX subfigure/subcaption)
#      Added:   each metric also saved as its own PNG (stage3_{metric}_{suffix}w{W}.png); the
#               1x3 combined panel is still written as the backup / log view
#      Added:   y-label on the cross-covariance panel, which previously relied on its title
# 5.4  Added: stage3_excess_removed_w{W}.png — the percentage-points-removed table as a grouped
#             bar chart (one group per alpha, one bar per remedy, signed value labels)
# 5.3  Added: per-alpha stdout table of percentage points of EnKF excess removed by each remedy
#             (excess_pct - excess_<remedy>_pct), the write-up number for "how much of the gap
#             the remedy closes" as opposed to each filter's own excess level
# 3.16 Changed: figs output -> figs/results
# 3.8  Changed: emit four PNGs (QR-vs-EnKF, IEnKF-vs-EnKF, QR-vs-IEnKF, all-three) instead of one
# 3.6  Changed: all three panels use shaded +/-1 std (across seeds) bands instead of errorbar caps
# 3.5  Added: iterative-EnKF line on all three metric panels; panels skip absent remedies
# 3.2  created — Stage 3 error panel (standard EnKF vs quad-reg EnKF, 2 lines per metric)
# ============================================================

os.makedirs('figs/results', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run error_sweep.py first')

fs.use()
AX, FS = fs.AX, fs.FS       # sizes come from figstyle; do not re-assert them here
# three panels across the text width. For roomier panels use 0.48 / fs.WIDE — they then
# wrap to two rows in LaTeX, costing ~0.29 page.
FRAC, ASP = 0.32, fs.XY

d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)   # seed std if present

# ---- which RMSE convention is in this file? -------------------------------------------
# error_sweep.py 5.16 made excess_pct per-component-then-averaged and started stamping
# rmse_convention into the npz. An OLDER file has no such key and its excess_pct is the
# pooled ratio (one mean over cycles AND components), which weights each component by its
# own squared error — in L63 that hands z ~84% of the answer. Plotting it under the new
# axis label would be wrong by tens of percentage points, so say so rather than draw it
# quietly.
CONV = str(d['rmse_convention']) if 'rmse_convention' in d.files else None
if CONV is None:
    print(f'WARNING: {f} predates error_sweep.py 5.16 (no rmse_convention key). Its\n'
          f'excess_pct is the POOLED RMSE ratio, not the per-component-then-averaged number\n'
          f'the write-up quotes. Rerun error_sweep.py before using these figures.')
else:
    print(f'RMSE convention: {CONV}')

# each filter: (label, colour, marker, jensen key, crosscov key, excess key)
ENKF = ('EnKF',           '#c0392b', 'o-', 'jensen_norm',    'crosscov_err',    'excess_pct')
QR   = ('QR-EnKF',        '#e67e22', 'D-', 'jensen_qr_norm', 'crosscov_qr_err', 'excess_qr_pct')
IE   = ('IEnKF',          '#2471a3', '^-', 'jensen_ie_norm', 'crosscov_ie_err', 'excess_ie_pct')

# (filename stem, index into the key triple, y-label) — same order as the old (a)/(b)/(c) panels
PANELS = [('jensen_bias',  0, r'$\|E[h(x)]-h(E[x])\|$'),
          ('crosscov_err', 1, r'$\mathcal{E}_{xh}$'),
          ('rmse_excess',  2, 'excess over PF (%)')]


def draw(ax, series, ki, ylab, legend=True, compact=False):
    if ki == 2:
        ax.axhline(0, ls='--', color='#777', lw=1)          # PF baseline (0% excess)
    for label, colr, mk, *keys in series:
        if keys[0] not in d.files:
            continue                                        # remedy absent from this npz
        key = keys[ki]
        ax.plot(a, d[key], mk, color=colr, lw=2, label=label)
        ax.fill_between(a, d[key] - sd(key + '_std'), d[key] + sd(key + '_std'),
                        color=colr, alpha=0.2)
    ax.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
    if compact:
        fs.compact(ax, ylab)          # thinned ticks, folded 10^k, tight label padding
    else:
        ax.set_ylabel(ylab, fontname=AX, fontsize=FS)
    ax.grid(alpha=0.3)
    if legend:
        ax.legend(**fs.leg_above(ncol=len([t for t in series if t[3] in d.files])))


def make(series, suffix):
    # one standalone PNG per metric — these are the write-up figures
    for stem, ki, ylab in PANELS:
        fig, ax = plt.subplots(figsize=fs.size(FRAC, ASP))
        draw(ax, series, ki, ylab)
        fig.tight_layout()
        out = f'figs/results/stage3_{stem}_{suffix}w{W}.png'
        fig.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
        print(f'saved {out}')

    # combined 1x3 — now the WRITE-UP figure rather than the backup. Three separate PNGs in
    # 0.32 subfigures give a 3.35cm plot box; this gives 4.59cm at the same page height, because
    # the y-labels move above the axes and matplotlib controls the gutters instead of LaTeX.
    fig, axes = plt.subplots(1, 3, figsize=fs.size(1.0, 0.32))
    for i, (stem, ki, ylab) in enumerate(PANELS):
        draw(axes[i], series, ki, ylab, legend=False, compact=True)
        fs.panel_letter(axes[i], i)     # no subcaptions on a merged figure to carry these
    h, l = axes[0].get_legend_handles_labels()
    if l:
        fig.legend(h, l, **fs.leg_fig(ncol=len(l)))
    fig.tight_layout(w_pad=0.4)
    fs.save(fig, f'figs/results/stage3_errors_{suffix}w{W}.png', tight=False)


make([ENKF, QR],     'qr_')
make([ENKF, IE],     'ie_')
make([QR,   IE],     'qrie_')
make([ENKF, QR, IE], '')

# ---- percentage points of the EnKF's excess that each remedy removes, per alpha ----
# Both excesses are already in % of PF RMSE, so their difference is in percentage points:
# a remedy at 18.2% against an EnKF at 25.4% has removed 7.2 pp. Positive = the remedy closed
# that much of the EnKF's shortfall against the PF; negative = it widened it.
present = [(lbl, col, ke) for lbl, col, _, _, _, ke in (QR, IE) if ke in d.files]
print('\npercentage points of EnKF excess removed  (+ = remedy closer to PF)')
print(f"{'alpha':>6}{'EnKF':>9}" + ''.join(f'{lbl:>15}{"removed":>9}' for lbl, _, _ in present))
for ai, al in enumerate(a):
    line = f"{al:6.1f}{d['excess_pct'][ai]:8.1f}%"
    for _, _, ke in present:
        line += f"{d[ke][ai]:14.1f}%{d['excess_pct'][ai] - d[ke][ai]:+9.1f}"
    print(line)

# ---- same quantity as a grouped bar chart ----
# Bars are placed on a categorical axis (one slot per alpha) rather than at the alpha value,
# so the group spacing stays even if the sweep grid is ever made non-uniform.
x = np.arange(len(a))
width = 0.8 / len(present)
fig, ax = plt.subplots(figsize=fs.size(0.72, 0.55))
ax.axhline(0, color='#444', lw=1)                        # 0 = remedy matched the standard EnKF
for i, (lbl, col, ke) in enumerate(present):
    gap = d['excess_pct'] - d[ke]
    bars = ax.bar(x + (i - (len(present) - 1) / 2) * width, gap, width, color=col, label=lbl)
    ax.bar_label(bars, fmt='%+.1f', fontsize=fs.BAR, fontname=AX, padding=2)
ax.set_xticks(x); ax.set_xticklabels([f'{v:g}' for v in a])
ax.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
ax.set_ylabel('excess removed vs EnKF (percentage points)', fontname=AX, fontsize=FS)
ax.grid(axis='y', alpha=0.3); ax.set_axisbelow(True); ax.legend(**fs.LEG)
fig.tight_layout()
out = f'figs/results/stage3_excess_removed_w{W}.png'
fig.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
print(f'saved {out}')