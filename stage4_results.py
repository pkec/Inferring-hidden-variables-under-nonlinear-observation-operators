# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.45 Changed: thesis_trace_pair() drops the per-panel alpha titles and moves the legend
#      above the axes, matching stage4_excess_gap_alpha. The two alphas are already in the
#      filename and belong in the caption alongside the (a)/(b) letters, which is how every
#      other split figure in the report carries them; dropping the titles is also what frees
#      the strip above the panels for the legend. The old legend sat inside panel (a) at
#      'upper right', on top of the trace, and covered only that panel even though the key is
#      identical in both — one figure-level legend now serves the pair.
#      CAPTION MUST NAME THE ALPHAS: with the titles gone, (a) and (b) are the only thing
#      distinguishing the panels.
# 5.36 Added: thesis_rls_resid_pair() — the same merge as thesis_trace_pair(), but across
#      TWO MODES at a fixed alpha (default shadow/inject at 0.8) instead of two alphas at
#      one mode. Combines what were two separate stage4_rls_resid_{mode}_a0.8_s0_thesis.png
#      files into one PNG, labelled (a)/(b), reusing thesis_rls_resid()'s magnitude/running-
#      mean logic unchanged. Saved as stage4_rls_resid_pair_{modes}_a{alpha}_s{seed}_thesis.png
#      in figs/results.
# 5.35 Added: thesis_trace_pair() — two thesis_trace() panels (default alpha=0.0 and 0.8)
#      merged into ONE PNG, side by side and labelled (a)/(b), for direct inclusion in the
#      write-up instead of two separately-exported 0.48\textwidth PNGs stitched together by
#      a LaTeX subfigure pair (see claude/page_cuts_exact.md B3). Reuses the exact window,
#      correlation and legend logic of thesis_trace(); only the combined layout is new.
#      Emitted per mode as stage4_residual_trace_pair_{mode}_a{a0}_a{a1}_s{seed}_thesis.png
#      in figs/results, since it is the body figure rather than a diagnostic.
# 5.34 Added: thesis_rls_resid() — the RLS residual for ONE seed, cut at TMAX=150 and sized
#      for print. The existing figure puts all three seeds on one axis, and each seed's clock
#      restarts at 0, so that trace doubles back on itself twice. Fine as a log view under a
#      running mean; wrong to print.
#      Changed: stage4_excess_gap_alpha and stage4_excess_alpha sized through figstyle, with
#      the legend above the axes. The 8in and 9in exports shrank by 0.42 and 0.52 at their
#      include widths, taking 14pt labels to ~6-7pt.
#      Changed: excess-removed y-label shortened to 'excess removed (pp)'; the sign
#      convention belongs in the caption.
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching error_sweep.py 5.16. base_rmse / pf_rmse / corr_rmse all move, so every
#      excess figure in this file (stage4_excess_alpha, stage4_excess_gap_alpha) and the
#      printed relative-reduction table change with it.
# 5.9  Added: TMAX (=150) time cut on the per-cycle traces — the residual trace and the RLS
#      residual ran the full t=300 and were too compressed to read. Each seed's time axis
#      restarts at 0, so the cut halves every seed rather than dropping any; a seed wholly
#      past TMAX would be skipped. Set TMAX = None for the whole run.
#      Changed: the per-seed correlation box is now computed over the PLOTTED window, so the
#      annotated number describes what is shown rather than the full run
#      Changed: the RLS running mean is still smoothed over the FULL run and only then sliced,
#      so the mode='same' edge artefact stays outside the plotted range
#      Changed: legends are Arial 11pt via LEG (was 8-9pt default font); prop= is used rather
#      than fontsize= because prop overrides it
# 5.8  Changed: report styling to match Stage 1/2 — axis labels Arial 14pt via the AX/FS
#      constants, remaining titles dropped (the weight-trajectory suptitle, and the per-seed
#      correlation title on the residual trace, which is now an in-axes text box so the number
#      survives without a title)
#      Added: the two multi-panel figures also emit one PNG per panel — residual trace per
#      seed, weight trajectories per output component. The combined figures stay as backups.
# 4.36 Removed: the per-alpha excess-gap-over-time figure (stage4_excess_gap_a{alpha}.png)
#      and its rolling-window helpers, which nothing else used. The excess-removed view
#      survives aggregated over alpha in stage4_excess_gap_alpha.png.
# 4.35 Changed: the residual trace is split one panel per SEED and shows a single component
#      (TRACE_COMP), with the innovation twin-axis dropped — three concatenated seeds, three
#      stacked components and an overlay whose range dwarfed the residual made it
#      unreadable. Per-seed correlation is annotated. Matches the blind trace layout.
# 5.7  Changed: comments corrected — they described the blind modes as scored on a disjoint
#      seed set, which stopped being true when rls_blind.py moved to TEST_SEEDS = cfg.seeds.
#      The seen/blind split is by TRAINING population; scoring is shared. Behaviour unchanged.
# 4.34 Changed: the excess-vs-alpha figure draws ONE baseline per seed population (seen /
#      blind) instead of one per mode — within a population they are the same runs, so the
#      four dashed lines were redundant. Marker now encodes the population (circle = seen,
#      square = blind) and baselines are dashed. Added a check that baselines within a
#      population actually agree, which catches stale result files.
# 4.29 Added: unrecognised result modes are flagged as probably-stale instead of being
#      plotted silently — a stage4_blind_a{alpha}.npz left over from before the
#      blind_shadow/blind_inject split was appearing as a phantom 'blind' mode. Weight
#      files are now excluded by prefix rather than relying on the schema guard.
# 4.26 Fixed: the file was truncated mid-way through rolling_rmse, so the excess-gap-over-
#      time figure never ran; restored. Added colours for the blind_shadow / blind_inject
#      modes (unknown modes still fall back to the default cycle).
# 4.9  Removed: raw RMSE-vs-alpha figure — the excess-over-PF panel carries the same
#      information in the Stage 2/3 convention, so the absolute plot was redundant.
# 4.24 Added: stage4_excess_gap_alpha.png — excess removed vs alpha per mode, absolute
#      (percentage points) and relative (% of that mode's baseline), so training-set and
#      blind performance can be read against each other directly.
#      Changed: weight-trajectory figure skipped for frozen-weight modes (blind).
# 4.23 Fixed: baseline and PF are now stored PER MODE. They were keyed on alpha alone, so
#      with blind running a disjoint seed set the last mode read overwrote the baseline and
#      one mode's correction was scored against another mode's runs — which is why
#      stage4_results and stage4_blind disagreed. Each mode now plots its own baseline.
# 4.13 Added: innovation overlaid on the residual trace (twin axis), so residual spikes can
#      be checked against innovation spikes. Skipped for pre-4.13 files with no innov key.
# 4.12 Changed: per-run figures (residual trace, RLS residual, weight trajectories) moved
#      to figs/diagnostic/s4 to stop figs/results filling up; the excess time series is
#      replaced by a single excess-GAP panel (EnKF excess minus corrected excess, so
#      +8% means the correction removed 8 points of excess). Dropped the per-cycle error
#      panel and the absolute excess-over-PF panel.
# 4.10 Added: per-cycle error & excess time series per alpha (stage4_excess_timeseries_
#      a{alpha}.png) — EnKF, PF and every mode on one axis. Excess uses a rolling-window
#      RMSE before the ratio (a naive per-cycle ratio hit 27000% when the PF error dipped
#      near zero) and a symlog axis; panel (a) shows raw per-cycle error, no ratio.
# 4.7  Fixed: discovery failed on Windows — glob returns backslash separators, but the
#      regex hard-coded 'data/', so no result file ever matched and the script exited
#      with "no ... found" despite the files existing. Now matches the filename only.
# 4.2  created — mode-agnostic Stage 4 plotting. Globs the common-schema result
#      files; per-cycle figures (residual trace, RLS residual, weight trajectories)
#      and per-alpha sweeps (RMSE, excess over PF). New modes need no edits.
# ============================================================

import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from rls_core import rmse_percomp

os.makedirs('figs/results', exist_ok=True)
os.makedirs('figs/diagnostic/s4', exist_ok=True)     # per-run, per-alpha diagnostics live here
S4 = 'figs/diagnostic/s4'                            # keeps the many per-run PNGs out of figs/results

LABELS = ['$x_1$', '$x_2$', '$x_3$']
fs.use()
AX, FS = fs.AX, fs.FS     # sizes come from figstyle; do not re-assert them here
# component shown in the residual trace (0=x). The fold in h_alpha lives in x, so x is the
# informative one by default.
TRACE_COMP = 0
# physical time cut for the per-cycle traces. Each seed's time axis restarts at 0, so this
# halves every seed rather than dropping any. None = plot the whole run.
TMAX = 300
# time cut for the residual-trace panel only; the full TMAX window at 8.2cm wide reads as a
# band rather than a shape.
THESIS_TMAX = 60.0
THESIS_SEED = 0          # seed shown in the body; the rest stay in the per-seed PNGs
TRACE_PAIR_ALPHAS = (0.0, 0.8)   # the two alphas shown side by side in thesis_trace_pair()
RESID_PAIR_ALPHA = 0.8           # the fixed alpha for thesis_rls_resid_pair()
RESID_PAIR_MODES = ('shadow', 'inject')   # the two modes shown side by side, at RESID_PAIR_ALPHA
# one colour per mode; unknown modes fall back to the cycle so a new mode still plots
MODE_COLOUR = {'shadow': "#0bcd2c", 'inject': "#b73dcf",
               'blind_shadow': "#1B613A", 'blind_inject': "#920070"}
BASE, PFC, TRUEC = '#c0392b', '#16a085', '#333333'
# (innov is still saved in the schema; the trace no longer overlays it — see 4.35)
MODE_ORDER = ['shadow', 'inject', 'blind_shadow', 'blind_inject']

# ---- discover every result file: data/stage4_{mode}_a{alpha}.npz ----
# Modes the current pipeline produces. Anything else on disk is almost certainly a result
# file left over from an earlier configuration — it still parses and would be plotted
# silently alongside current results, so it is flagged loudly instead. (A file named
# stage4_blind_a0.4.npz, for instance, predates the split into blind_shadow/blind_inject.)
KNOWN_MODES = {'shadow', 'inject', 'blind_shadow', 'blind_inject'}
NON_RESULT = ('log', 'weights')             # prefixes that are inputs, not results
stale = []
runs = []                                   # list of dicts: mode, alpha, path, data
for path in sorted(glob.glob('data/stage4_*_a*.npz')):
    # glob returns backslash separators on Windows, forward slashes on posix; normalise
    # and match the filename only, so discovery works on both platforms.
    name = path.replace('\\', '/').split('/')[-1]
    m = re.match(r'stage4_(\w+)_a([\d.]+)\.npz$', name)
    if not m:
        continue
    mode, alpha = m.group(1), float(m.group(2))
    if mode.split('_')[0] in NON_RESULT:     # logs and weight files are inputs, not results
        continue
    d = np.load(path, allow_pickle=True)
    if 'pred' not in d.files:                # not a Stage 4 result (schema guard)
        continue
    if mode not in KNOWN_MODES:
        stale.append(path)
    runs.append(dict(mode=mode, alpha=alpha, path=path, d=d))

if not runs:
    raise SystemExit('no data/stage4_{mode}_a{alpha}.npz found — run rls_shadow.py / rls_inject.py first')

modes_found = {r['mode'] for r in runs}
modes = [m for m in MODE_ORDER if m in modes_found]

alphas_all = sorted({r['alpha'] for r in runs})
print(f'found modes {modes} at alphas {alphas_all}')
if stale:
    print(f'\nWARNING: result files with unrecognised modes are being plotted. These are\n'
          f'probably stale, written under an older configuration. Expected: '
          f'{sorted(KNOWN_MODES)}\n'
          + ''.join(f'  {p_}\n' for p_ in stale)
          + 'Delete them and rerun the learners, or they will be compared against current\n'
            'runs as if they were produced the same way.\n')

col = lambda mode: MODE_COLOUR.get(mode, None)          # None -> matplotlib default cycle
rmse = rmse_percomp        # per component, then averaged over x, y, z (5.18)


def legend_above_fig(fig, ax_src, ncol=2):
    h, l = ax_src.get_legend_handles_labels()
    kw = dict(fs.leg_above(ncol=ncol))
    for k in ('loc', 'bbox_to_anchor', 'bbox_transform', 'ncol'):
        kw.pop(k, None)
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.04), ncol=ncol, **kw)


def thesis_trace(mode, alpha, t, target, pred, row_seed, seed, legend=True):
    if row_seed is None:
        return
    sel = np.where(row_seed == seed)[0]
    if not len(sel):
        return
    tv = t[sel] - t[sel][0]                     # this seed's clock, from 0 either way
    keep = (tv <= THESIS_TMAX) if THESIS_TMAX else np.ones(len(sel), bool)
    idx, tw = sel[keep], tv[keep]
    if len(idx) < 2:
        return
    c_win = np.corrcoef(target[idx, TRACE_COMP], pred[idx, TRACE_COMP])[0, 1]
    c_all = np.corrcoef(target[sel, TRACE_COMP], pred[sel, TRACE_COMP])[0, 1]

    fig, ax = plt.subplots(figsize=fs.size(0.48, fs.SERIES))
    ax.plot(tw, target[idx, TRACE_COMP], '-', color=TRUEC, lw=0.8, label=r'true  $r_k$')
    ax.plot(tw, pred[idx, TRACE_COMP], '-', color=col(mode), lw=1.1,
            label=r'learnt  $w_{k-1}^{\top}u_k$')
    ax.set_xlabel('Time', **fs.LAB)
    ax.set_ylabel(LABELS[TRACE_COMP], **fs.LAB)
    ax.grid(alpha=0.3)
    ax.text(0.99, 0.03, f'corr {c_win:+.3f} shown / {c_all:+.3f} full', transform=ax.transAxes,
            ha='right', va='bottom', **fs.ANN,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbb', alpha=0.9))
    if legend:
        ax.legend(loc='upper right', ncol=2, **fs.LEG)
    fs.save(fig, f'{S4}/stage4_residual_trace_{mode}_a{alpha}_s{seed}_thesis.png')


def thesis_rls_resid(mode, alpha, t, resid, row_seed, seed, tmax=TMAX):
    sel = np.arange(len(t)) if row_seed is None else np.where(row_seed == seed)[0]
    if len(sel) < 2:
        return
    tv = t[sel] - t[sel][0]                       # this seed's clock, from 0
    mag = np.linalg.norm(resid[sel], axis=1)
    win = max(1, len(mag) // 50)
    # smoothed over this seed's WHOLE run, then cut, so the mode='same' edge artefact
    # stays outside the plotted window
    runmean = np.convolve(mag, np.ones(win) / win, mode='same')
    k = tv <= tmax if tmax else np.ones(len(sel), bool)
    fig, ax = plt.subplots(figsize=fs.size(0.72, fs.SERIES))
    ax.plot(tv[k], mag[k], '-', color=col(mode), lw=0.6, alpha=0.35,
            label=r'$\|r_k-w_{k-1}^{\top}u_k\|$')
    ax.plot(tv[k], runmean[k], '-', color=col(mode), lw=1.8,
            label=f'running mean ({win} cycles)')
    ax.set_xlabel('Time', **fs.LAB); ax.set_ylabel('RLS residual', **fs.LAB)
    ax.grid(alpha=0.3); ax.legend(**fs.leg_above(ncol=2))
    fs.save(fig, f'{S4}/stage4_rls_resid_{mode}_a{alpha}_s{seed}_thesis.png')


def thesis_trace_pair(mode, alphas, by_ma, seed=THESIS_SEED, tmax=THESIS_TMAX):
    panels = []
    for alpha in alphas:
        r = by_ma.get((mode, alpha))
        if r is None:
            print(f'  [thesis_trace_pair] {mode}: no run at alpha={alpha} — figure skipped')
            return
        d = r['d']
        t, target, pred = d['times'], d['target'], d['pred']
        row_seed = d['row_seed'] if 'row_seed' in d.files else None
        sel = np.arange(len(t)) if row_seed is None else np.where(row_seed == seed)[0]
        if not len(sel):
            print(f'  [thesis_trace_pair] {mode} a={alpha}: seed {seed} not found — figure skipped')
            return
        tv = t[sel] - t[sel][0]                     # this seed's clock, from 0 either way
        keep = (tv <= tmax) if tmax else np.ones(len(sel), bool)
        idx, tw = sel[keep], tv[keep]
        if len(idx) < 2:
            print(f'  [thesis_trace_pair] {mode} a={alpha}: <2 points in window — figure skipped')
            return
        c_win = np.corrcoef(target[idx, TRACE_COMP], pred[idx, TRACE_COMP])[0, 1]
        c_all = np.corrcoef(target[sel, TRACE_COMP], pred[sel, TRACE_COMP])[0, 1]
        panels.append((alpha, tw, target[idx, TRACE_COMP], pred[idx, TRACE_COMP], c_win, c_all))

    # same per-panel size thesis_trace() already uses, doubled in width for the second panel
    w0, h0 = fs.size(0.48, fs.SERIES)
    fig, axes = plt.subplots(1, len(panels), figsize=(2 * w0, h0))
    for i, (ax, (alpha, tw, tgt, prd, c_win, c_all)) in enumerate(zip(axes, panels)):
        ax.plot(tw, tgt, '-', color=TRUEC, lw=0.8, label=r'true  $r_k$')
        ax.plot(tw, prd, '-', color=col(mode), lw=1.1, label=r'learnt  $w_{k-1}^{\top}u_k$')
        ax.set_xlabel('Time', **fs.LAB)
        ax.set_ylabel(LABELS[TRACE_COMP], **fs.LAB)
        # no per-panel alpha title (5.45): the alphas are in the filename and belong in the
        # caption against (a)/(b), and the strip above the panels carries the legend instead
        ax.grid(alpha=0.3)
        ax.text(0.99, 0.03, f'corr {c_win:+.3f} shown / {c_all:+.3f} full', transform=ax.transAxes,
                ha='right', va='bottom', **fs.ANN,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='#bbb', alpha=0.9))
        fs.panel_letter(ax, i)
    fig.tight_layout()
    legend_above_fig(fig, axes[0], ncol=2)   # both panels carry the same two keys
    tag = '_'.join(f'a{a}' for a, *_ in panels)
    fs.save(fig, f'figs/results/stage4_residual_trace_pair_{mode}_{tag}_s{seed}_thesis.png')


def thesis_rls_resid_pair(alpha, modes_, by_ma, seed=THESIS_SEED, tmax=TMAX):
    panels = []
    for mode in modes_:
        r = by_ma.get((mode, alpha))
        if r is None:
            print(f'  [thesis_rls_resid_pair] no run for {mode} at alpha={alpha} — figure skipped')
            return
        d = r['d']
        t, resid = d['times'], d['rls_resid']
        row_seed = d['row_seed'] if 'row_seed' in d.files else None
        sel = np.arange(len(t)) if row_seed is None else np.where(row_seed == seed)[0]
        if len(sel) < 2:
            print(f'  [thesis_rls_resid_pair] {mode} a={alpha}: seed {seed} not found — figure skipped')
            return
        tv = t[sel] - t[sel][0]                       # this seed's clock, from 0
        mag = np.linalg.norm(resid[sel], axis=1)
        win = max(1, len(mag) // 50)
        # smoothed over this seed's WHOLE run, then cut — same order as thesis_rls_resid()
        runmean = np.convolve(mag, np.ones(win) / win, mode='same')
        k = tv <= tmax if tmax else np.ones(len(sel), bool)
        panels.append((mode, tv[k], mag[k], runmean[k], win))

    # same per-panel size thesis_trace_pair() uses, so the two "pair" figures match on the page
    w0, h0 = fs.size(0.48, fs.SERIES)
    fig, axes = plt.subplots(1, len(panels), figsize=(2 * w0, h0))
    for i, (ax, (mode, tk, magk, rmk, win)) in enumerate(zip(axes, panels)):
        ax.plot(tk, magk, '-', color=col(mode), lw=0.6, alpha=0.35,
                label=r'$\|r_k-w_{k-1}^{\top}u_k\|$')
        ax.plot(tk, rmk, '-', color=col(mode), lw=1.8, label=f'running mean ({win} cycles)')
        ax.set_xlabel('Time', **fs.LAB)
        ax.set_ylabel('RLS residual', **fs.LAB)
        ax.set_title(mode.replace('_', ' '), **fs.LAB)
        ax.grid(alpha=0.3)
        fs.panel_letter(ax, i)
        if i == 0:
            ax.legend(loc='upper right', **fs.LEG)
    fig.tight_layout()
    tag = '_'.join(modes_)
    fs.save(fig, f'figs/results/stage4_rls_resid_pair_{tag}_a{alpha}_s{seed}_thesis.png')


# --- per-run, per-cycle figures ---
for r in runs:
    d, mode, alpha = r['d'], r['mode'], r['alpha']
    t = d['times']                                        # (T,) physical time
    target, pred = d['target'], d['pred']                # (T,3) true and learnt residual
    resid = d['rls_resid']                               # (T,3) per-cycle miss
    w_hist = d['w_hist']                                 # (T,K,3) weight trajectory

    # ---- Figure: true residual vs learnt residual ----
    # One panel per SEED, one component. Three seeds concatenated onto a single axis with
    # all three components stacked and an innovation twin-axis was unreadable: ~2000 cycles
    # of a spiky series compressed into one width, and the innovation's range (up to ~1500
    # in z) dwarfing the residual it was drawn beside. Split by seed, one component, no twin
    # axis — matching the blind trace, which was legible for exactly these reasons.
    keep = np.ones(len(t), bool) if TMAX is None else (t <= TMAX)   # (T,) time cut, see TMAX
    row_seed = d['row_seed'] if 'row_seed' in d.files else None
    groups = ([(sd, np.where((row_seed == sd) & keep)[0]) for sd in np.unique(row_seed)]
              if row_seed is not None else [(None, np.where(keep)[0])])
    # A seed with nothing inside the window drops out — but that means its clock does NOT
    # restart at 0, so TMAX is cutting by position in a concatenation rather than by run time.
    # Say so loudly instead of silently emitting fewer panels than there are seeds.
    dropped = [sd for sd, idx in groups if not len(idx)]
    if dropped:
        print(f'  [{mode}] a={alpha}: seeds {dropped} lie entirely past TMAX={TMAX} — their '
              f'time axes do not restart at 0, so this cut is not "first {TMAX} of each run"')
    groups = [(sd, idx) for sd, idx in groups if len(idx)]
    fig, axes = plt.subplots(len(groups), 1, figsize=(13, 2.9 * len(groups)))
    axes = np.atleast_1d(axes)
    singles = [plt.subplots(figsize=(13, 3.4)) for _ in groups]   # one standalone PNG per seed
    for i, (sd, idx) in enumerate(groups):
        # each panel is drawn twice: onto the combined backup figure and onto its own figure
        for ax, xlab in ((axes[i], i == len(groups) - 1), (singles[i][1], True)):
            ax.plot(t[idx], target[idx, TRACE_COMP], '-', color=TRUEC, lw=1.0,
                    label=r'true residual  $r_k$')
            ax.plot(t[idx], pred[idx, TRACE_COMP], '-', color=col(mode), lw=1.4,
                    label=r'learnt  $w_{k-1}^{\top}u_k$')
            ax.set_ylabel((f'seed {sd}\n' if sd is not None else '') + LABELS[TRACE_COMP],
                          fontname=AX, fontsize=FS)
            ax.grid(alpha=0.3)
            if xlab:
                ax.set_xlabel('Time', fontname=AX, fontsize=FS)
            if sd is not None:
                # correlation over this seed alone: a pooled number hides a model that tracks
                # one run well and another badly. In-axes box, not a title (titles are dropped).
                # Computed over the PLOTTED window only, so it describes what the reader sees.
                c = np.corrcoef(target[idx, TRACE_COMP], pred[idx, TRACE_COMP])[0, 1]
                ax.text(0.99, 0.04, f'corr(true, learnt) = {c:+.3f}', transform=ax.transAxes,
                        ha='right', va='bottom', fontsize=11, fontname=AX,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#bbb', alpha=0.9))
    axes[0].legend(loc='upper right', ncol=2, **fs.LEG)
    fig.tight_layout()
    out = f'{S4}/stage4_residual_trace_{mode}_a{alpha}.png'
    fig.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig); print(f'saved {out}')

    for (sd, _), (figx, axs) in zip(groups, singles):   # figx, not fs — fs is figstyle
        axs.legend(loc='upper right', ncol=2, **fs.LEG)
        figx.tight_layout()
        tag = f'_s{sd}' if sd is not None else ''
        out = f'{S4}/stage4_residual_trace_{mode}_a{alpha}{tag}.png'
        figx.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(figx)
        print(f'saved {out}')
    # thesis panel: ONE seed, so it belongs outside the per-seed loop — inside, it re-rendered
    # the same PNG once per seed. Legend on alpha=0 only, since the two sit side by side
    # under a single caption.
    thesis_trace(mode, alpha, t, target, pred, row_seed, THESIS_SEED, legend=(alpha == 0.0))

    # ---- Figure: RLS residual over cycles (convergence) ----
    # magnitude across components + a running mean so the trend is readable
    mag = np.linalg.norm(resid, axis=1)                   # (T,) per-cycle miss magnitude
    win = max(1, len(mag) // 50)                          # running-mean window; int
    # smoothed over the FULL run, then sliced to the TMAX window: cutting first would put a
    # mode='same' edge artefact right at t=TMAX, inside the plotted range
    runmean = np.convolve(mag, np.ones(win) / win, mode='same')
    kt = np.where(keep)[0]                                 # rows inside the TMAX window
    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(t[kt], mag[kt], '-', color=col(mode), lw=0.7, alpha=0.4,
            label=r'$\|r_k-w_{k-1}^{\top}u_k\|$')
    ax.plot(t[kt], runmean[kt], '-', color=col(mode), lw=2, label=f'running mean ({win} cycles)')
    ax.set_xlabel('Time', fontname=AX, fontsize=FS)
    ax.set_ylabel('RLS residual', fontname=AX, fontsize=FS)
    ax.grid(alpha=0.3)
    ax.legend(**fs.LEG)
    #ax.set_title(rf'Stage 4 [{mode}] RLS residual per cycle — $\alpha$={alpha}')
    fig.tight_layout()
    out = f'{S4}/stage4_rls_resid_{mode}_a{alpha}.png'
    fig.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig); print(f'saved {out}')
    thesis_rls_resid(mode, alpha, t, resid, row_seed, THESIS_SEED)


# --- combined thesis trace-pair figure ---
# One merged PNG per mode with two alphas (default 0.0 and 0.8) side by side — the figure
# claude/page_cuts_exact.md B3 asks for in place of a stacked or LaTeX-stitched pair.
by_ma = {(r['mode'], r['alpha']): r for r in runs}
for m in modes:
    thesis_trace_pair(m, TRACE_PAIR_ALPHAS, by_ma, THESIS_SEED)
thesis_rls_resid_pair(RESID_PAIR_ALPHA, RESID_PAIR_MODES, by_ma, THESIS_SEED)


# --- across-alpha sweeps ---
# Baseline and PF are stored PER MODE, not per alpha. Every mode now scores on cfg.seeds —
# rls_blind.py trains on cfg.blind_seeds but sets TEST_SEEDS = cfg.seeds — so all four
# baselines should be the same runs and the per-mode keying is redundant on current data.
# It is kept as the guard it has become: a per-mode baseline is what lets the agreement
# check below detect a stale result file written under a different seed set or config.
base_rmse = {m: {} for m in modes}                        # mode -> alpha -> scalar
pf_rmse = {m: {} for m in modes}
corr_rmse = {m: {} for m in modes}
for r in runs:
    d, a, m = r['d'], r['alpha'], r['mode']
    base_rmse[m][a] = rmse(d['xa_base'], d['truth'])
    pf_rmse[m][a] = rmse(d['xpf_mean'], d['truth'])
    corr_rmse[m][a] = rmse(d['xa_corr'], d['truth'])

excess = lambda r_c, r_p: (r_c - r_p) / r_p * 100         # % excess over the PF floor


# ---- Figure: excess removed vs alpha, per mode (train vs test in one view) ----
# For each mode: gap = baseline excess - corrected excess, in percentage points. All modes
# are scored on the same runs, so the lines differ only in what the weights were fitted on:
# seen modes on those runs, blind modes on cfg.blind_seeds. Their separation is therefore
# the generalisation cost, read off a common baseline.
fig, ax = plt.subplots(figsize=fs.size(0.5, 0.80))
LEGEND_LABELS = {
    'shadow': 'online shadow',
    'inject': 'online inject',
    'blind_shadow': 'blind shadow',
    'blind_inject': 'blind inject',
}

for m in modes:
    xa = sorted(corr_rmse[m])
    be = np.array([excess(base_rmse[m][a], pf_rmse[m][a]) for a in xa])
    ce = np.array([excess(corr_rmse[m][a], pf_rmse[m][a]) for a in xa])
    ax.plot(xa, be - ce, 'o-', color=col(m), lw=1.8, ms=4,
            label=LEGEND_LABELS.get(m, m.replace('_', ' ')))

ax.axhline(0, ls='--', color='#777', lw=1.2)  # 0 = correction changed nothing
ax.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
ax.set_ylabel('excess removed (pp)', **fs.LAB)
ax.grid(alpha=0.3)
ax.legend(**fs.leg_above(ncol=2))
#ax.set_title('(a) absolute gap: baseline - corrected')

#fig.suptitle('Stage 4: how much RMSE excess the correction removes ' '(+ = helping; separation between modes = generalisation cost)', y=1.02)
fs.save(fig, 'figs/results/stage4_excess_gap_alpha.png')