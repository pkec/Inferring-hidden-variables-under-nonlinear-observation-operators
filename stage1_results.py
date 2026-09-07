import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import (cfg, rk4, PERTURB_BY_WINDOW, rmse_comp, spread_comp,
                    cal_ratio_comp, CAL_RATIO_CONVENTION, PERTURB_BASELINE_ALPHA)
from partfilt import run_pf
from enkf import run_enkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.44 Changed: RMSE, spread and the calibration ratio come from config.rmse_comp /
#      spread_comp / cal_ratio_comp instead of the local rmse()/spr() lambdas. Same
#      arithmetic, but Stage 1 is the reference the sweep is now pinned to at alpha=0, so it
#      must not carry its own copy of the definition.
#      Added: the 3-component MEAN ratio is printed beside the per-component values. The
#      write-up quotes a single number (PF 0.994, EnKF 0.956); this file only ever printed
#      three, so that number was being formed by hand off the terminal.
#      Added: data/stage1_results_w{W}.npz — per-seed RMSE, spread, ratio and excess, both
#      per component and as the equal-weight headline, plus the convention strings. The save
#      block existed but was commented out, so every Stage 1 number in the thesis had to be
#      transcribed from stdout. thesis_values.py reads this file.
#      NOTE: this run is the alpha=0 column of error_sweep.py and stage2_hbar_diagnostic.py
#      as of 5.44 — same truth, same observation draw, same obs_std, same baseline jitter,
#      no multiplier scan. If the three ever disagree there, one of them has drifted.
# 5.43 Added: merged 1x3 rank histogram (stage1_rank_hist.png) with (a)/(b)/(c) drawn in and
#      one legend for the row — the per-component PNGs are unchanged and still emitted.
#      Changed: the rank-histogram legend moved above the axes. Inside, it sat top-right on
#      a distribution that is heaviest at low ranks, and top-left is where the letter goes.
# 5.42 Added: (a)/(b)/(c) drawn into the merged bar figure. The three standalone PNGs still
#      take their letters from the LaTeX subcaption.
# 5.25 Changed: sizing and fonts now come from figstyle.py. Figures are exported at the
#      width they are PRINTED at, so \includegraphics scales by 1.0 and the point sizes
#      are what lands on the page — 9pt labels, 8pt ticks, 7.5pt legends, matching every
#      other report figure. The 11in and 15in figures shrank by 0.49 and 0.45.
#      Added: one standalone PNG per bar panel (stage1_{rmse,spread,ratio}_w{W}.png); the
#      1x3 grid stays as the backup / log view.
#      Fixed: the three bar panels had neither title nor y-label once the titles were
#      dropped in 4.39, so they could not be told apart. Each now carries its y-label.
# 4.41 Changed: rank histograms back to the single cfg.seed run — seeds no longer pooled
#      Changed: bar error bars removed; the ratio panel carries a 1 +/- 0.05 shaded target band
#               instead, matching stage3_calibration_ratio.py
# 4.40 Changed: RMSE and spread bars are 3-seed too, so all of Figure 2 shares one provenance.
#               Per-seed scalars are formed first and then averaged, the same order as
#               error_sweep.py (note the sweep also re-calibrates jitter per seed, so the
#               two agree in order of operations, not necessarily in value)
#      Changed: stdout reports 3-seed means and now prints raw RMSE (feeds the linear table)
#      Removed: the RMSE/spread stdout lines — inverse of both figures and of the write-up;
#               spread/RMSE is the only direction reported from here
# 4.39 Changed: calibration ratio is a third bar panel in stage1_rmse_spread.png, so that
#               figure reads RMSE | spread | ratio across
#      Removed: the calibration-ratio time series (stage1_calibration_ratio.png), the
#               cal_ratio() rolling helper and the ROLL window that fed it
# 4.38 Added: calibration ratio (spread/RMSE) for PF and EnKF, averaged over cfg.seeds.
#             A seed carries its own obs realisation as well as its own filter rng,
#             matching error_sweep.py — so this is the first Stage 1 metric not single-seed
#      Changed: every figure now sized from len(windows) — kills the blank axes that appeared
#               when windows = [15] against a hard-coded 2-column / 2-row grid
#      Changed: layout is horizontal — rows = window, cols = panel (x,y,z across for traces
#               and rank histograms; RMSE | spread | ratio across for the bars)
# 3.16 Changed: figs -> figs/results; windows defaults to [15] (headline commit; re-add [5, 15]
#      for the writeup window comparison)
# 2.10 Changed: self-contained + consolidated — each Stage 1 figure is one PNG with both
#               windows as panels (traces 3x2, rmse/spread 2x2, rank-hist 2x3); no per-window files
# 2.9  Changed: window-aware — figures tagged _w{obs_every}, titles show the window; ensure figs/results/ exists
# ============================================================

os.makedirs('figs/results', exist_ok=True)
fs.use()
windows = [15]                             # committed headline window; use [5, 15] for the writeup window comparison
nW = len(windows)                          # every figure grid is sized from this, so no blank axes
labels = ['$x_1$', '$x_2$', '$x_3$']
h = lambda x: x                            # linear observation operator (Stage 1 baseline)

# truth trajectory (same recipe as truth_obs.py; deterministic, shared across windows and seeds)
s = np.array([1.0, 1.0, 1.0])
for _ in range(500): s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps): truth[k + 1] = rk4(truth[k], cfg.dt)
t_truth = np.arange(cfg.n_steps + 1) * cfg.dt

res = {}
for W in windows:
    cfg.obs_every = W
    cfg.perturb_std = PERTURB_BY_WINDOW[W]
    obs_idx = np.arange(W, cfg.n_steps + 1, W)

    # the cfg.seed realisation, kept separately because Figure 1 plots one trajectory
    # (averaging chaotic state paths across seeds would smear them into nonsense)
    obs = truth[obs_idx] + np.random.default_rng(cfg.seed).normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
    pf = run_pf(obs, truth, obs_idx, h, save_particles=True)
    en = run_enkf(obs, truth, obs_idx, h, save_forecast=False)
    res[W] = dict(pf=pf, en=en, obs=obs, t_obs=obs_idx * cfg.dt)

    # every metric below is averaged over cfg.seeds. A seed is a full independent replicate —
    # its own observation realisation AND its own filter rng — matching error_sweep.py and
    # stage4_log.py. Truth is shared.
    pfs, ens = [], []
    for sd in cfg.seeds:
        o = truth[obs_idx] + np.random.default_rng(sd).normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
        pfs.append(pf if sd == cfg.seed else run_pf(o, truth, obs_idx, h, seed=sd))
        ens.append(en if sd == cfg.seed else run_enkf(o, truth, obs_idx, h, seed=sd, save_forecast=False))
    # per-seed scalars first, then the seed average — the order error_sweep.py uses
    # every per-seed scalar below comes from config.py's definitions, the same ones
    # error_sweep.py and stage2_hbar_diagnostic.py use, so the three files cannot drift
    res[W]['prs'] = np.array([rmse_comp(p)      for p in pfs])      # (S, 3) per-seed PF RMSE
    res[W]['pss'] = np.array([spread_comp(p)    for p in pfs])      # (S, 3) per-seed PF spread
    res[W]['prt'] = np.array([cal_ratio_comp(p) for p in pfs])      # (S, 3) per-seed PF ratio
    res[W]['ers'] = np.array([rmse_comp(e)      for e in ens])      # (S, 3) per-seed EnKF RMSE
    res[W]['ess'] = np.array([spread_comp(e)    for e in ens])      # (S, 3) per-seed EnKF spread
    res[W]['ert'] = np.array([cal_ratio_comp(e) for e in ens])      # (S, 3) per-seed EnKF ratio

    prs, pss, prt = res[W]['prs'], res[W]['pss'], res[W]['prt']
    ers, ess, ert = res[W]['ers'], res[W]['ess'], res[W]['ert']
    # excess is a RATIO: per component, per seed, then averaged. Same rule as error_sweep.py
    res[W]['exc'] = (ers - prs) / prs * 100                         # (S, 3)
    excess = res[W]['exc'].mean(0)                                  # per seed, then averaged
    # the headline is the mean over x, y, z of the per-seed value — config.cal_ratio, applied
    # to each seed and then seed-averaged. This is the single number the write-up quotes.
    sd = lambda v: v.std(0, ddof=1)                                 # sample SD across seeds
    print(f"\nwindow {W}  (linear h, {len(cfg.seeds)}-seed)      x            y            z"
          f"        mean(x,y,z)")
    print(f"{'PF RMSE':>16}" + ''.join(f"{m:9.3f}±{s:.3f}" for m, s in zip(prs.mean(0), sd(prs)))
          + f"{prs.mean():12.3f}")
    print(f"{'EnKF RMSE':>16}" + ''.join(f"{m:9.3f}±{s:.3f}" for m, s in zip(ers.mean(0), sd(ers)))
          + f"{ers.mean():12.3f}")
    print(f"{'PF spread/RMSE':>16}" + ''.join(f"{v:15.3f}" for v in prt.mean(0))
          + f"{prt.mean():12.3f}")
    print(f"{'EnKF spread/RMSE':>16}" + ''.join(f"{v:15.3f}" for v in ert.mean(0))
          + f"{ert.mean():12.3f}")
    print(f"{'EnKF excess %':>16}" + ''.join(f"{v:14.1f}%" for v in excess)
          + f"{excess.mean():11.1f}%")
    print(f"{'':16}per-seed calibration ratio (mean over x, y, z), seeds {list(cfg.seeds)}")
    print(f"{'  PF':>16}" + ''.join(f"{v:15.4f}" for v in prt.mean(1)))
    print(f"{'  EnKF':>16}" + ''.join(f"{v:15.4f}" for v in ert.mean(1)))


# ---- Figure 1: time traces, single seed, one file per component ----
T0, T1 = 10.0, 40.0                        # zoom window; the full run is unreadable at 4 lines
for ci in range(3):
    fig, axes = plt.subplots(nW, 1, figsize=fs.size(0.8, 0.45 * nW), squeeze=False)
    for ri, W in enumerate(windows):
        r = res[W]
        wt = (t_truth >= T0) & (t_truth <= T1)
        wo = (r['t_obs'] >= T0) & (r['t_obs'] <= T1)
        ax = axes[ri, 0]
        ax.plot(t_truth[wt], truth[wt, ci], 'k-', lw=1.0, label='Truth')
        ax.plot(r['t_obs'][wo], r['obs'][wo, ci], 'r.', ms=3, alpha=0.35, label='Obs')
        ax.plot(r['t_obs'][wo], r['pf']['en_mean'][wo, ci], 'b-', lw=1.0, label='PF')
        ax.plot(r['t_obs'][wo], r['en']['en_mean'][wo, ci], 'g-', lw=1.0, label='EnKF')
#        ax.set_title(f'window {W} — {labels[ci]}')
        ax.set_xlabel('Time', **fs.LAB); ax.set_ylabel(labels[ci], **fs.LAB); ax.grid(alpha=0.3)
        if ri == 0: ax.legend(loc='upper right', ncol=4, **fs.LEG)
#    fig.suptitle(f'Stage 1: PF vs EnKF, linear h (seed {cfg.seed})')
    fs.save(fig, f'figs/results/stage1_traces_{labels[ci]}.png')


# ---- Figure 2: RMSE, spread and calibration ratio (rows = window x 3 cols), bars = seed mean,
#      error bars = +/-1 sd across seeds ----
x = np.arange(3); bw = 0.35
# (filename stem, y-label, is-the-ratio-panel). The panels had neither title nor y-label
# once the titles were dropped, which left three bar charts that could not be told apart.
BARS = [('rmse',   'RMSE',        False),
        ('spread', 'spread',      False),
        ('ratio',  'spread / RMSE', True)]


def bar_panel(ax, W, stem, ylab, is_ratio, legend):
    r = res[W]
    vpf, ven = {'rmse':   (r['prs'], r['ers']),
                'spread': (r['pss'], r['ess']),
                'ratio':  (r['prt'], r['ert'])}[stem]   # config.cal_ratio_comp, per seed
    ax.bar(x - bw / 2, vpf.mean(0), bw, label='PF')
    ax.bar(x + bw / 2, ven.mean(0), bw, label='EnKF')
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylab, **fs.LAB)
    if is_ratio:
        ax.axhspan(0.95, 1.05, color='green', alpha=0.10, label='target 1 ± 0.05')
        ax.axhline(1, ls='--', color='k', lw=1)
    if legend or is_ratio:
        ax.legend(**fs.LEG)


# one standalone PNG per panel — these are the write-up figures
for W in windows:
    for stem, ylab, is_ratio in BARS:
        fig, ax = plt.subplots(figsize=fs.size(0.32))
        bar_panel(ax, W, stem, ylab, is_ratio, legend=(stem == 'rmse'))
        fs.save(fig, f'figs/results/stage1_{stem}_w{W}.png')

# combined, kept as the backup / log view
fig, axes = plt.subplots(nW, 3, figsize=fs.size(1.0, 0.30 * nW), squeeze=False)
for ri, W in enumerate(windows):
    for ci, (stem, ylab, is_ratio) in enumerate(BARS):
        bar_panel(axes[ri, ci], W, stem, ylab, is_ratio, legend=(ri == 0 and ci == 0))
        fs.panel_letter(axes[ri, ci], ri * 3 + ci)   # no subcaptions here to carry these
fs.save(fig, 'figs/results/stage1_rmse_spread.png')


# ---- Figure 3: PF rank histograms, single cfg.seed run ----
n_bins = 25
var_names = ['x', 'y', 'z']


def rank_panel(ax, W, v, legend):
    particles = res[W]['pf']['particles_history']
    tao = res[W]['pf']['truth_at_obs']
    T = particles.shape[0]                 # cycles = ranks counted
    ranks = np.array([np.searchsorted(np.sort(particles[t, :, v]), tao[t, v])
                      for t in range(T)])
    ax.hist(ranks, bins=n_bins, edgecolor='black', alpha=0.7)
    ax.axhline(T / n_bins, color='red', ls='--', label='Uniform')
    ax.set_xlabel('Rank', **fs.LAB); ax.set_ylabel('Count', **fs.LAB)
    if legend:
        ax.legend(**fs.leg_above(ncol=1))  # top-left is where the (a) letter goes


# one standalone PNG per component — these are the write-up figures
for W in windows:
    for v, var in enumerate(var_names):
        fig, ax = plt.subplots(figsize=fs.size(0.48, 0.75))
        rank_panel(ax, W, v, legend=True)
        fs.save(fig, f'figs/results/stage1_rank_hist_{var}.png')

# combined 1x3 (rows = window), kept as the backup / log view. Same treatment as the bar
# figure: letters drawn in, one legend for the row since all three panels are identical.
fig, axes = plt.subplots(nW, 3, figsize=fs.size(1.0, 0.34 * nW), squeeze=False)
for ri, W in enumerate(windows):
    for v in range(3):
        rank_panel(axes[ri, v], W, v, legend=False)
        fs.panel_letter(axes[ri, v], ri * 3 + v)
        if v:
            axes[ri, v].set_ylabel('')
fig.tight_layout()
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, **fs.leg_fig(ncol=1, y=1.0))
fs.save(fig, 'figs/results/stage1_rank_hist.png', tight=False)


# ---- save metrics to npz ----
# One file per window. Everything is saved PER SEED: the thesis sheet quotes seed columns,
# and no seed-averaged array can be un-averaged back into them. Each quantity appears twice,
# per component (S, 3) and as the equal-weight headline (S,), so nothing downstream has to
# re-decide which convention it is on.
#
# This is the alpha=0 reference. error_sweep.py and stage2_hbar_diagnostic.py run the same
# filter, on the same truth and observations, at the same baseline jitter for alpha=0, so
# their alpha=0 column must reproduce these numbers exactly.
os.makedirs('data', exist_ok=True)

for W in windows:
    r = res[W]
    mean_sd = lambda v, ax=0: (v.mean(ax), v.std(ax, ddof=1))       # seed mean, sample SD

    np.savez(
        f'data/stage1_results_w{W}.npz',
        seeds=np.array(cfg.seeds),
        obs_every=W,
        alpha=PERTURB_BASELINE_ALPHA,          # linear h; this IS the sweep's alpha=0 point
        perturb_std=PERTURB_BY_WINDOW[W],      # baseline jitter, no multiplier applied
        obs_std=cfg.obs_std,
        components=np.array(labels),
        rmse_convention='per-component then averaged (5.16)',
        cal_ratio_convention=CAL_RATIO_CONVENTION,

        # ---- per component, every seed: (S, 3) ----
        pf_rmse_c_by_seed=r['prs'], pf_spread_c_by_seed=r['pss'], pf_ratio_c_by_seed=r['prt'],
        en_rmse_c_by_seed=r['ers'], en_spread_c_by_seed=r['ess'], en_ratio_c_by_seed=r['ert'],
        excess_c_by_seed=r['exc'],

        # ---- equal-weight headline, every seed: (S,) ----
        pf_rmse_by_seed=r['prs'].mean(1), pf_spread_by_seed=r['pss'].mean(1),
        pf_ratio_by_seed=r['prt'].mean(1),
        en_rmse_by_seed=r['ers'].mean(1), en_spread_by_seed=r['ess'].mean(1),
        en_ratio_by_seed=r['ert'].mean(1),
        excess_by_seed=r['exc'].mean(1),

        # ---- seed mean and sample SD of the headline: scalars ----
        pf_rmse=r['prs'].mean(), pf_rmse_std=mean_sd(r['prs'].mean(1))[1],
        en_rmse=r['ers'].mean(), en_rmse_std=mean_sd(r['ers'].mean(1))[1],
        pf_ratio=r['prt'].mean(), pf_ratio_std=mean_sd(r['prt'].mean(1))[1],
        en_ratio=r['ert'].mean(), en_ratio_std=mean_sd(r['ert'].mean(1))[1],
        excess=r['exc'].mean(), excess_std=mean_sd(r['exc'].mean(1))[1],

        # ---- seed mean and sample SD per component: (3,) ----
        pf_rmse_c=r['prs'].mean(0), pf_rmse_c_std=r['prs'].std(0, ddof=1),
        en_rmse_c=r['ers'].mean(0), en_rmse_c_std=r['ers'].std(0, ddof=1),
        pf_ratio_c=r['prt'].mean(0), pf_ratio_c_std=r['prt'].std(0, ddof=1),
        en_ratio_c=r['ert'].mean(0), en_ratio_c_std=r['ert'].std(0, ddof=1),
        excess_c=r['exc'].mean(0), excess_c_std=r['exc'].std(0, ddof=1),
    )
    print(f'saved data/stage1_results_w{W}.npz')