# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.40 Removed: bar-value labels. The full table is in the appendix, and at 90 degrees the
#      labels were forcing vertical headroom on the log axis purely to clear themselves.
#      barf/bar_rot go with them.
# 5.39 Changed: the per-seed scatter is off by default (show_pts=False). The spread it showed
#      is not lost — the per-seed min-max behind each mean now prints, which matters at
#      n=20, N=30000 where the three seeds run 1.2 to 64.8 and the mean alone hides it.
#      Changed: the merged figure carries ONE legend for the row. The bar colours mean the
#      same N in both panels, so a per-panel key repeated itself; handles are taken from the
#      calibration panel, the only one that also has the target band.
# 5.37 Added: (a)/(b) drawn into the merged 1x2, which is now sized as a write-up figure at
#      \textwidth rather than a log view.
#      Changed: sizing and fonts come from figstyle, replacing the local W_IN/H_IN/FS_T set.
#      Fixed:   panel() took a parameter named `fs`, which shadowed the figstyle alias inside
#      the function — renamed lab_fs.
#      Changed: the legend moves above the axes; inside, it needed the ylim headroom added
#      in 5.24 purely to clear the tallest bar.
# 5.24 Changed: the ESS panel plots min ESS / N as a percentage (L96_ESS=raw restores the
#      counts). This reverses 6.7, which had switched back to raw counts; both are kept
#      because the write-up quotes raw counts ("reaches exactly one particle") while the
#      figure now carries the fraction. The output filename is unchanged either way.
#      Added: cross-check of the computed percentage against the sheet's own
#      'min ESS / N (%)' column. Averaging the ratio and taking the ratio of the averages
#      agree here because N is constant within a cell, so the choice is not a judgement call.
# 5.23 Changed: the sweep values are read from pf_stage_results_1.xlsx and seed-averaged here,
#      replacing the hand-pasted ess/ratio arrays. Those arrays matched NO block in the
#      spreadsheet — not stride 1, 2 or 4, not any single seed, not the free-text block — so
#      every number in the two figures moves. See the printed comparison table.
#      Added: the three per-seed values are scattered over each bar. At n=20, N=30000 the
#      per-seed min ESS runs 1.2 to 64.8, which a mean alone hides, and the seed spread is
#      the point at that corner of the sweep.
#      Added: cross-check against the sheet's own 'avg' rows; a mismatch is reported.
#      Added: thesis-sized export — 3.21in wide is 0.48\textwidth on A4 with 2cm margins, so
#      \includegraphics scales by 1.0 and the fonts print at the pt size set here. The 5.6in
#      panels shrink by 0.50 at 0.42\textwidth, taking 14pt labels to 7pt and the 9pt bar
#      labels to 4.5pt.
# 5.11 Changed: report styling to match Stage 1/2 — axis labels Arial 14pt, legends and bar
#      labels Arial 11/9pt (AX/FS/LEG), both panel titles and the suptitle dropped; the
#      (a)/(b) letters now come from the LaTeX subfigure/subcaption
#      Added:   each panel also saved as its own PNG (tmp_l96_{ess,calibration}.png); the
#               1x2 combined figure stays as the backup / log view
#      Changed: calibration band relabelled 'target 1 ± 0.1', matching the other calibration
#               figures (the band itself was already 0.9-1.1)
#      Fixed:   figs/diagnostic was never created, so the save failed on a clean checkout
# 6.8  Changed: labelled the two panels (a) and (b) so they can be referenced individually
#      in the write-up.
# 6.7  Changed: dropped the per-stride subplot loop (only p = n was ever populated),
#      merged the ESS and calibration panels into a single png, and switched the
#      ESS panel from "% of N" back to raw minimum ESS counts.
# 6.6  Changed: split into two separate figures — min ESS as % of N, and calibration ratio.
#      The RMSE-vs-climatology panel is dropped.
# 6.5  created — throwaway plot of the L96 PF sweep: min ESS, calibration ratio and RMSE
#      against climatology, for the three observation strides.
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')

os.makedirs('figs/diagnostic', exist_ok=True)
fs.use()

SRC = os.environ.get('L96_XLSX', 'pf_l96.xlsx')
SHEET = 'PF Results'
STRIDE = float(os.environ.get('L96_STRIDE', 1))   # 1 = every state observed, the p = n case
ESS_MODE = os.environ.get('L96_ESS', 'pct')       # 'pct' = min ESS / N (%), 'raw' = counts
ESS_KEY = 'ess_pct' if ESS_MODE == 'pct' else 'ess'

COL = ['#2471a3', '#e67e22', '#16a085']          # one per N
AX, FS = fs.AX, fs.FS     # sizes come from figstyle; do not re-assert them here
FRAC = 0.48               # each thesis panel is one of two across the text width


def load(src, sheet, stride):
    df = pd.read_excel(src, sheet_name=sheet)
    df = df.loc[:, ~df.columns.astype(str).str.startswith('Unnamed')]
    for c in ['stride', 'seed', 'n', 'N', 'min ESS', 'spread/RMSE', 'min ESS / N (%)']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    raw = df.dropna(subset=['stride', 'seed', 'n', 'N'])
    raw = raw[raw.stride == stride]
    if raw.empty:
        raise SystemExit(f'no per-seed rows at stride={stride} in {src}')

    ns = sorted(raw.n.unique().astype(int))
    Ns = sorted(raw.N.unique().astype(int))
    seeds = sorted(raw.seed.unique().astype(int))
    shape = (len(ns), len(Ns))
    keys = ('ess', 'ess_pct', 'ratio')
    mean = {k: np.full(shape, np.nan) for k in keys}
    pts = {k: np.full(shape + (len(seeds),), np.nan) for k in keys}
    for i, n in enumerate(ns):
        for j, N in enumerate(Ns):
            cell = raw[(raw.n == n) & (raw.N == N)]
            for key, colname in (('ess', 'min ESS'), ('ratio', 'spread/RMSE')):
                mean[key][i, j] = cell[colname].mean()
                for s_i, s in enumerate(seeds):
                    v = cell[cell.seed == s][colname]
                    if len(v):
                        pts[key][i, j, s_i] = v.iloc[0]
            # N is constant within a cell, so averaging the ratio and taking the ratio of
            # the averages are the same number here — no choice to defend.
            mean['ess_pct'][i, j] = mean['ess'][i, j] / N * 100
            pts['ess_pct'][i, j] = pts['ess'][i, j] / N * 100
            col_pct = cell['min ESS / N (%)']
            if col_pct.notna().any() and abs(col_pct.mean() - mean['ess_pct'][i, j]) > 1e-6:
                print(f"  MISMATCH ess_pct n={n} N={N}: computed {mean['ess_pct'][i, j]:.5f}, "
                      f"sheet column {col_pct.mean():.5f}")

    # cross-check against the sheet's own averages: they should agree, and if they do not
    # one of the two blocks is stale and the figure would be built on the wrong one.
    avg = df[(df.seed.isna()) & (df.stride == stride) & df.n.notna()]
    for i, n in enumerate(ns):
        for j, N in enumerate(Ns):
            row = avg[(avg.n == n) & (avg.N == N)]
            if not len(row):
                continue
            for key, colname in (('ess', 'min ESS'), ('ratio', 'spread/RMSE')):
                sheet_v = pd.to_numeric(row[colname], errors='coerce').iloc[0]
                if np.isfinite(sheet_v) and abs(sheet_v - mean[key][i, j]) > 1e-6:
                    print(f'  MISMATCH {key} n={n} N={N}: computed {mean[key][i, j]:.4f}, '
                          f"sheet 'avg' row {sheet_v:.4f}")
    return ns, Ns, seeds, mean, pts


print(f'reading {SRC} at stride={STRIDE:g}')
ns, Ns, seeds, mean, pts = load(SRC, SHEET, STRIDE)
print(f'  n {ns}   N {Ns}   seeds {seeds}')
for key, name in ((ESS_KEY, 'min ESS' + (' / $N_{PF}$ (%)' if ESS_KEY == 'ess_pct' else '')),
                  ('ratio', 'spread/RMSE')):
    print(f'  {name} (rows n, cols N):')
    print('   ', np.array2string(mean[key], precision=3).replace('\n', '\n    '))

print('  per-seed range behind each mean (min - max):')
for i, n_ in enumerate(ns):
    cells = '  '.join(f'N={N_}: {np.nanmin(pts[ESS_KEY][i, j]):.3g}-'
                      f'{np.nanmax(pts[ESS_KEY][i, j]):.3g}' for j, N_ in enumerate(Ns))
    print(f'    n={n_}:  {cells}')

x = np.arange(len(ns))
width = 0.26


def panel(ax, which, lab_fs=FS, show_pts=False, legend=True):
    v, p = (mean[which], pts[which]) if which.startswith('ess') else (mean['ratio'], pts['ratio'])
    for j, N in enumerate(Ns):
        off = x + (j - 1) * width
        ax.bar(off, v[:, j], width, color=COL[j], label=fr'$N_{{PF}} = {N:,}$')
        if show_pts:                      # n=3, so show the runs rather than an error bar
            jit = np.linspace(-0.28, 0.28, p.shape[2]) * width   # spread across the bar, so
            for i in range(len(ns)):                             # points miss the bar label
                ax.plot(off[i] + jit, p[i, j, :], 'o', ms=2.2,
                        mfc='none', mec='#222', mew=0.6, zorder=5,
                        label='per seed' if (j == 0 and i == 0) else None)
    if which.startswith('ess'):
        ax.set_yscale('log')                  # values span ~2 decades either way
        # plain '%', not '\%' — this is a matplotlib label, not LaTeX source
        ax.set_ylabel('minimum ESS / $N_{PF}$ (%)' if which == 'ess_pct' else 'minimum ESS',
                      fontname=AX, fontsize=lab_fs)
    else:
        ax.axhspan(0.95, 1.05, color='green', alpha=0.10, label='target 1 ± 0.05')
        ax.set_ylim(0, 1.25)
        ax.set_ylabel('spread / RMSE', fontname=AX, fontsize=lab_fs)
    ax.set_xticks(x); ax.set_xticklabels(ns)
    ax.set_xlabel('state dimension $n$', fontname=AX, fontsize=lab_fs)
    ax.grid(alpha=0.3, axis='y')
    if legend:
        ax.legend(**fs.leg_above(ncol=2))


# one standalone PNG per panel — diagnostic size, unchanged
for stem, key in (('ess', ESS_KEY), ('calibration', 'calibration')):
    fig, ax = plt.subplots(figsize=fs.size(FRAC))
    panel(ax, key)
    fs.save(fig, f'figs/diagnostic/tmp_l96_{stem}.png')

# thesis size — exported at the printed width so the fonts above are what appears on the page
for stem, key in (('ess', ESS_KEY), ('calibration', 'calibration')):
    fig, ax = plt.subplots(figsize=fs.size(FRAC))
    panel(ax, key)
    fs.save(fig, f'figs/diagnostic/tmp_l96_{stem}_thesis.png')

# merged 1x2 at \textwidth — one image, one caption, with the letters drawn in
fig, axes = plt.subplots(1, 2, figsize=fs.size(1.0, 0.36))
for i, key in enumerate((ESS_KEY, 'calibration')):
    panel(axes[i], key, legend=False)
    fs.panel_letter(axes[i], i)   # no subcaptions on a merged figure to carry these
# One legend for the row. The bar colours mean the same N in both panels, so a per-panel
# key said nothing twice. Handles come from the calibration panel, the only one that also
# carries the target band.
h_, l_ = axes[1].get_legend_handles_labels()
fig.legend(h_, l_, **fs.leg_fig(ncol=len(l_)))
fig.tight_layout(w_pad=0.4)
fs.save(fig, 'figs/diagnostic/tmp_l96_summary.png', tight=False)