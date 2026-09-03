
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
#      fs.panel_letter. The separate PNGs keep taking their letters from the LaTeX subcaption.
#      Removed: three set_ylabel calls on the merged axes that fs.compact overwrites anyway.
# 5.31 Changed: the y-label goes back on the y axis. fs.compact() now defaults to pos='y',
#      so the merge keeps most of the width gain (4.21cm of plot box per panel against
#      3.35cm for three separate subfigures) without the label reading as a heading.
# 5.29 Changed: the combined 1x3 becomes the write-up figure and the three separate PNGs the
#      backlog — 4.59cm of plot box per panel against 3.35cm as three 0.32 subfigures, at the
#      same page height. See stage3_results.py 5.29.
# 5.26 Changed: panel aspect 0.72 -> fs.XY (0.95). At 0.32\textwidth the panels were
#      5.4 x 3.9 cm with only 56% of the height left for data after the y-label,
#      ticks and x-label — the y-label was physically taller than the plot. Now
#      5.4 x 5.2 cm, 66% data. FRAC/ASP are one line if you want 0.48 / fs.WIDE.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 5.5in panels shrank by 0.39 at 0.32\textwidth.
#      Changed: the excess y-label drops ', mean over x, y, z' — at 0.32\textwidth it did not
#      fit, and the averaging convention belongs in the caption. State it there.
# 5.19 Changed: the excess panel is now labelled as the per-component-then-averaged number,
#      which is what error_sweep.py has written into excess_pct since 5.16. No arithmetic
#      changes here — this script only plots what the sweep saved — but the label did.
#      Added:   a guard that refuses to plot an npz written before 5.16 silently. Such a file
#               has no rmse_convention key and its excess_pct is the POOLED ratio, so the
#               figure would carry the new label over the old number.
#      Added:   stage2_excess_components_w{W}.png — excess per state component with the
#               quoted mean overlaid, so the averaging the write-up claims is visible rather
#               than asserted. Skipped if excess_c is absent (pre-5.16 file).
#      Changed: Arial 14pt applied through AX/FS constants instead of repeated literals.
# 3.16 Changed: figs output -> figs/results
# 3.6  Changed: RMSE-excess panel uses a shaded +/-1 std band instead of errorbar caps,
#               matching the Jensen / cross-cov panels
# 3.2  Changed: single-config 1x3 layout (was 2x6 over jitter x noise mode); dropped the quadratic-
#               regression overlay so Stage 2 shows the standard EnKF only (QR -> stage3_errors.py)
# 3.0  Changed: metric (c) is now RMSE excess over PF (one raw % line) instead of the two-line
#               EnKF/PF RMSE panel
# 2.14 Added:   +/-1 std (across seeds) error bars on RMSE / bands on the Jensen & cross-cov panels
# 2.8  created — consolidated Stage 2 error panel per window
# ============================================================

os.makedirs('figs/results', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run stage2.py first')

EN, J, C = '#c0392b', '#c0392b', '#c0392b'
fs.use()
AX, FS = fs.AX, fs.FS       # sizes come from figstyle; do not re-assert them here
FRAC, ASP = 0.32, fs.XY     # three panels across the text width.
                            # For roomier panels use 0.48 / fs.WIDE — they then
                            # wrap to two rows in LaTeX, costing ~0.29 page.
COMP = [('x', '#8e44ad'), ('y', '#16a085'), ('z', '#e67e22')]
d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)   # seed std, 0 on pre-2.14 files

# ---- which RMSE convention is in this file? -------------------------------------------
# error_sweep.py 5.16 made excess_pct per-component-then-averaged and started stamping
# rmse_convention into the npz. An OLDER file has no such key and its excess_pct is the
# pooled ratio (one mean over cycles AND components), which weights each component by its
# own squared error — in L63 that hands z ~84% of the answer. Plotting it under the new
# axis label would be wrong by tens of percentage points, so say so rather than draw it
# quietly.
CONV = str(d['rmse_convention']) if 'rmse_convention' in d.files else None
if CONV is None:
    print('!' * 78)
    print(f'WARNING: {f} predates error_sweep.py 5.16 (no rmse_convention key).')
    print('Its excess_pct is the POOLED RMSE ratio, NOT the per-component-then-averaged')
    print('number the write-up quotes. Rerun error_sweep.py before using these figures.')
    print('!' * 78)
else:
    print(f'RMSE convention: {CONV}')

fig, axes = plt.subplots(1, 3, figsize=fs.size(1.0, 0.32))
YLAB = [r'$\|E[h(x)]-h(E[x])\|$', r'$\mathcal{E}_{xh}$', 'excess over PF (%)']

axes[0].plot(a, d['jensen_norm'], 'o-', color=J, lw=2)
axes[0].fill_between(a, d['jensen_norm'] - sd('jensen_norm_std'),
                     d['jensen_norm'] + sd('jensen_norm_std'), color=J, alpha=0.2)

axes[1].plot(a, d['crosscov_err'], 'o-', color=C, lw=2)
axes[1].fill_between(a, d['crosscov_err'] - sd('crosscov_err_std'),
                     d['crosscov_err'] + sd('crosscov_err_std'), color=C, alpha=0.2)

axes[2].axhline(0, ls='--', color='#777', lw=1)                          # PF baseline (0% excess)
axes[2].plot(a, d['excess_pct'], 'o-', color=EN, lw=2)
axes[2].fill_between(a, d['excess_pct'] - sd('excess_pct_std'),
                     d['excess_pct'] + sd('excess_pct_std'), color=EN, alpha=0.2)

for i, (ax, ylab) in enumerate(zip(axes, YLAB)):
    ax.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS); ax.grid(alpha=0.3)
    fs.compact(ax, ylab)              # thinned ticks, folded 10^k, tight label padding
    fs.panel_letter(ax, i)            # no subcaptions on a merged figure to carry these

#fig.suptitle(rf'Stage 2 error sources — window {W}, {mode}, {jit} jitter,  $h_\alpha(x)=x+\alpha x^2$', y=1.02)
fig.tight_layout(w_pad=0.4)
fs.save(fig, f'figs/results/stage2_errors_w{W}.png', tight=False)


# ---------------- (a) Jensen bias ----------------
fig1, ax1 = plt.subplots(figsize=fs.size(FRAC, ASP))
ax1.plot(a, d['jensen_norm'], 'o-', color=J, lw=2)
ax1.fill_between(a, d['jensen_norm'] - sd('jensen_norm_std'),
                     d['jensen_norm'] + sd('jensen_norm_std'), color=J, alpha=0.2)
ax1.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
ax1.set_ylabel(r'$\|E[h(x)]-h(E[x])\|$', fontname=AX, fontsize=FS)
ax1.grid(alpha=0.3)

fig1.tight_layout()
out1 = f'figs/results/stage2_jensen_bias_w{W}.png'
fig1.savefig(out1, dpi=fs.DPI, bbox_inches='tight')
plt.close(fig1)
print(f'saved {out1}')

# ---------------- (b) Cross-covariance error ----------------
fig2, ax2 = plt.subplots(figsize=fs.size(FRAC, ASP))
ax2.plot(a, d['crosscov_err'], 'o-', color=C, lw=2)
ax2.fill_between(a, d['crosscov_err'] - sd('crosscov_err_std'),
                     d['crosscov_err'] + sd('crosscov_err_std'), color=C, alpha=0.2)
ax2.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
ax2.set_ylabel(r'$\mathcal{E}_{xh}$', fontname=AX, fontsize=FS)  
ax2.grid(alpha=0.3)

fig2.tight_layout()
out2 = f'figs/results/stage2_crosscov_err_w{W}.png'
fig2.savefig(out2, dpi=fs.DPI, bbox_inches='tight')
plt.close(fig2)
print(f'saved {out2}')

# ---------------- (c) RMSE excess over PF ----------------
fig3, ax3 = plt.subplots(figsize=fs.size(FRAC, ASP))
ax3.axhline(0, ls='--', color='#777', lw=1)
ax3.plot(a, d['excess_pct'], 'o-', color=EN, lw=2)
ax3.fill_between(a, d['excess_pct'] - sd('excess_pct_std'),
                     d['excess_pct'] + sd('excess_pct_std'), color=EN, alpha=0.2)
ax3.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
ax3.set_ylabel('excess over PF (%)', fontname=AX, fontsize=FS)
ax3.grid(alpha=0.3)

fig3.tight_layout()
out3 = f'figs/results/stage2_rmse_excess_w{W}.png'
fig3.savefig(out3, dpi=fs.DPI, bbox_inches='tight')
plt.close(fig3)
print(f'saved {out3}')

# ---------------- excess per component, with the quoted mean overlaid ----------------
# The write-up says the excess is "averaged over the three axis components". This figure is
# that sentence: the three per-component excesses and, dashed, the mean of them, which is
# the single number excess_pct carries. The spread between the three is what a pooled RMSE
# would have collapsed — silently, and weighted toward whichever component has the largest
# errors.
if 'excess_c' in d.files:
    fig4, ax4 = plt.subplots(figsize=fs.size(FRAC, ASP))
    ax4.axhline(0, ls='--', color='#777', lw=1)
    for i, (lab, colr) in enumerate(COMP):
        ax4.plot(a, d['excess_c'][:, i], 'o-', color=colr, lw=1.8, label=lab)
    ax4.plot(a, d['excess_pct'], 'k--', lw=2.2, label='mean (quoted)')
    ax4.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
    ax4.set_ylabel('excess over PF (%)', fontname=AX, fontsize=FS)
    ax4.grid(alpha=0.3); ax4.legend(**fs.LEG)
    fig4.tight_layout()
    out4 = f'figs/results/stage2_excess_components_w{W}.png'
    fig4.savefig(out4, dpi=fs.DPI, bbox_inches='tight')
    plt.close(fig4)
    print(f'saved {out4}')
else:
    print('excess_c absent — per-component excess figure skipped (rerun error_sweep.py)')