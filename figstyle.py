# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.32 Added: panel_letter() — (a)/(b)/(c) inside the top-left of a panel, for merged
#      figures where there are no LaTeX subcaptions to carry the letters.
# 5.31 Changed: compact() keeps the y-label rotated beside the axis by default; pos='top'
#      is now opt-in. A label in the title slot reads as a heading, not as an axis label.
#      The tightened labelpad/tick pad recover most of the difference: 4.21cm of plot box
#      against 4.59cm for pos='top', and 3.35cm for three separate 0.32 subfigures.
# 5.30 Added: VERSION, so a stale figstyle on the import path reports itself instead of
#      surfacing as 'module figstyle has no attribute leg_above' from inside a plot call.
# 5.29 Added: leg_fig() — one legend above a merged row.
#      Added: compact() — y-label above the axes instead of rotated beside it, plus thinned
#      ticks and a folded 10^k factor. A rotated label costs width equal to the FONT HEIGHT,
#      not the string length, so moving it above buys ~1.2cm of plot box per panel on a
#      three-across row (3.35cm -> 4.54cm measured).
# 5.28 Added: leg_above() — legend above the axes. loc='best' had nowhere to go on the
#      excess panel, where three shaded bands fill the axes, so it sat on the quad-reg line.
# 5.26 Added: named aspects (XY/WIDE/SERIES) and a taller XY default (0.95, was 0.72).
#      At frac=0.32 the old default left 56% of the height for data because the axis
#      furniture costs a fixed amount whatever the panel size — the narrower the panel,
#      the taller it has to be, not shorter. 0.95 takes that to 66%.
# 5.25 created — shared sizing/fonts, so "consistent across figures" is enforced in one place
#      rather than re-asserted as literals in each script.
# ============================================================

import re
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

VERSION = (5, 32)            # bumped whenever a script gains a dependency on a new helper.
                             # Scripts guard on this, so a stale copy on the path fails with
                             # a clear message instead of an AttributeError mid-figure.

TEXTWIDTH_IN = 6.69          # A4 (21cm) less the 2cm side margins = 17cm

AX = 'Arial'                 # matches the thesis body font (helvet)
FS = 9                       # axis labels
TICK = 8                     # tick labels
LEGEND = 7.5                 # legend entries
ANNOT = 7                    # in-axes annotation boxes
BAR = 6.5                    # bar-value labels, which sit in the tightest space

LAB = dict(fontname=AX, fontsize=FS)
LEG = dict(prop=dict(family=AX, size=LEGEND), framealpha=0.85,
           handlelength=1.4, borderpad=0.3, labelspacing=0.3)
ANN = dict(fontname=AX, fontsize=ANNOT)
DPI = 300                    # 1:1 export, so this is the real print resolution


def use():
    """Set the defaults that need no per-call argument. Call once at the top of a script.

    Explicit fontname=/fontsize= arguments still win, so a script can override a single
    label without opting out of the rest."""
    plt.rcParams.update({
        'font.family': AX,
        'axes.labelsize': FS,
        'xtick.labelsize': TICK,
        'ytick.labelsize': TICK,
        'legend.fontsize': LEGEND,
        'axes.titlesize': FS,
        'savefig.dpi': DPI,
    })


# Aspect (height/width) by figure kind. A narrow panel spends a FIXED amount of width and
# height on the y-label, ticks and x-label, so the narrower the panel the larger the share
# that furniture takes: at frac=0.32 an aspect of 0.72 leaves only 56% of the height for data,
# and the y-label ends up physically taller than the plot. Small panels need a TALLER aspect,
# not a shorter one.
XY = 0.95          # line/scatter panels, three across a text width
WIDE = 0.72        # the same panel with room to breathe, two across
SERIES = 0.62      # time series: wide and short by nature


def size(frac, aspect=XY):
    w = TEXTWIDTH_IN * frac
    return (w, w * aspect)


def compact(ax, label, pos='y', nbins=4, sci=True):
    ax.xaxis.set_major_locator(MaxNLocator(nbins))
    ax.yaxis.set_major_locator(MaxNLocator(nbins))
    if sci:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 3))
        ax.figure.canvas.draw()                   # the offset text only exists once drawn
        off = ax.yaxis.get_offset_text().get_text()
        if off:
            ax.yaxis.get_offset_text().set_visible(False)
            m = re.fullmatch(r'1e([+-]?\d+)', off)
            label = (f'{label} ($\\times 10^{{{int(m.group(1))}}}$)' if m
                     else f'{label} ({off})')
    if pos == 'top':
        ax.set_title(label, fontname=AX, fontsize=FS, pad=3)
    else:
        ax.set_ylabel(label, fontname=AX, fontsize=FS, labelpad=1.5)
        ax.tick_params(axis='y', pad=1.5)


def panel_letter(ax, i, inside=True, weight='bold'):
    lab = f'({chr(ord("a") + i)})'
    if inside:
        ax.text(0.03, 0.97, lab, transform=ax.transAxes, ha='left', va='top',
                fontname=AX, fontsize=FS, fontweight=weight,
                bbox=dict(boxstyle='square,pad=0.15', fc='white', ec='none', alpha=0.75))
    else:
        ax.text(0.0, 1.02, lab, transform=ax.transAxes, ha='left', va='bottom',
                fontname=AX, fontsize=FS, fontweight=weight)


def leg_above(ncol=3, **kw):
    d = dict(LEG)
    d.update(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=ncol,
             columnspacing=0.7, handletextpad=0.4, framealpha=0)
    d.update(kw)
    return d


def leg_fig(ncol=3, y=0.99):
    return dict(loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, framealpha=0,
                prop=dict(family=AX, size=LEGEND), columnspacing=1.2, handletextpad=0.4)


def finish(*axes):
    for ax in axes:
        for a in (ax if hasattr(ax, '__len__') else [ax]):
            a.tick_params(labelsize=TICK)


def save(fig, path, dpi=DPI, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {path}')