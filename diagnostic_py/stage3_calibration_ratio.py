import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 29):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.45 Changed: labels, ticks and legend text scale by LABEL_SCALE, so the exported PNG reads
#      at the same size as stage4_excess_gap_alpha — the model figure — which prints at
#      0.5\textwidth against this figure's 0.72.
#      Changed: legend ncol is chosen from the number of KEYS drawn, not the number of series.
#      The target band contributes a key too, so ncol=len(series) put 4 entries on the first
#      row and left the fifth stranded on a row of its own.
# 5.28 Changed: legend moved above the axes via fs.leg_above, and series labels shortened to
#      QR-EnKF / IEnKF to match the write-up and stage3_results.py.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 8in figure shrank by 0.60 at 0.72\textwidth, taking the
#      8pt legend to 4.8pt. Literal fontname/fontsize arguments replaced by figstyle.
# 5.3  Changed: two PNGs (EnKF vs QR vs PF, EnKF vs IEnKF vs PF) via a make() helper, so each
#               remedy is read against the EnKF/PF pair without the other remedy's line on top
#      Removed: the single four-filter panel (calibration_ratio_w{W}.png)
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 3.8  Changed: single-config single panel (was 2x2 mode x jitter); added quad-reg and iterative
#               EnKF ratio lines alongside EnKF/PF to flag mis-calibration/collapse of each remedy
# 2.14 Added: +/-1 std (across seeds) shaded band on the ratio curves
# 2.5  created — spread/RMSE vs alpha
# ============================================================

fs.use()

# ---- label / legend sizing ---------------------------------------------------------------
# Model figure: stage4_excess_gap_alpha. It prints at 0.5\textwidth, so at a common 9pt its
# labels fill more of the exported PNG than this 0.72 figure does. LABEL_SCALE multiplies
# every figstyle point size — labels, ticks and legend alike — so the two match on screen.
# Set it to 1.0 to go back to plain figstyle sizes (9pt/8pt/7.5pt on the page).
LABEL_SCALE = 1.4

def scaled(kw, s=LABEL_SCALE):
    out = dict(kw)
    if 'fontsize' in out:                                 # fs.LAB
        out['fontsize'] = out['fontsize'] * s
    if 'prop' in out:                                     # fs.LEG, via fs.leg_above
        out['prop'] = dict(out['prop'], size=out['prop']['size'] * s)
    return out

LAB = scaled(fs.LAB)
TICKSIZE = fs.TICK * LABEL_SCALE

os.makedirs('figs/diagnostic', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run error_sweep.py first')

d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)

# each filter: (label, colour, marker, ratio key)
ENKF = ('EnKF',            '#c0392b', 'o-', 'en_ratio')
PF   = ('PF (reference)',  '#16a085', 's-', 'pf_ratio')
QR   = ('QR-EnKF',         '#e67e22', 'D-', 'qr_ratio')
IE   = ('IEnKF',           '#2471a3', '^-', 'ie_ratio')

def make(series, suffix, title):
    fig, ax = plt.subplots(figsize=fs.size(0.72))
    ax.axhspan(0.95, 1.05, color='green', alpha=0.10, label='target 1 ± 0.05')
    ax.axhline(1, ls='--', color='k', lw=1)
    for label, col, mk, kr in series:
        if kr not in d.files:
            continue                                    # remedy absent from this npz
        ax.plot(a, d[kr], mk, color=col, lw=2, label=label)
        ax.fill_between(a, d[kr] - sd(kr + '_std'), d[kr] + sd(kr + '_std'), color=col, alpha=0.15)
    ax.set_xlabel(r'$\alpha$', **LAB); ax.set_ylabel('spread / RMSE', **LAB)
    ax.tick_params(labelsize=TICKSIZE)
    #ax.set_title(f'Calibration quality vs nonlinearity — {title}\n'
    #             f'window {W}, {mode}, {jit} jitter')
    ax.grid(alpha=0.3)
    # ncol from the KEYS actually drawn (the target band contributes one), split into two
    # roughly equal rows — 5 keys at ncol=5 wrapped 4+1, which reads as a mistake.
    nkeys = 1 + len([x for x in series if x[3] in d.files])
    ax.legend(**scaled(fs.leg_above(ncol=(nkeys + 1) // 2)))
    fig.tight_layout()
    out = f'figs/diagnostic/calibration_ratio_{suffix}w{W}.png'
    fig.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}')

make([ENKF, QR, PF], 'qr_', 'quad-reg EnKF')
make([ENKF, IE, PF], 'ie_', 'iterative EnKF')
make([ENKF, QR, PF, IE], '', 'All')