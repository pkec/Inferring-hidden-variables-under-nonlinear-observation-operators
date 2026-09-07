# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.47 Added: an npz. Every statistic this script printed — straddle counts, failure onset,
#      error and spread during failure — existed only in terminal output, so the 4.3.3
#      numbers were being transcribed by hand and could not be re-checked.
#      Added: straddle and lock-on now counted as EPISODES as well as cycles. "N occasions
#      per run" is a count of episodes; a count of cycles is a different and much larger
#      number, and the two were not distinguished.
#      Added: RMSE and spread during the lock-on periods vs outside them, which is the
#      "the penalty is large and the filter is unaware" claim stated as two numbers.
#      Changed: statistics run over ALL of cfg.seeds. SEEDS was cfg.seeds[:2] for figure
#      legibility, which is a fine reason to plot two and no reason at all to compute
#      three-seed statistics from two. The figures still show two.
#      Changed: a second, per-seed observation set (see the docstring) is run for the
#      statistics, so a "per run" number here is the same kind of replicate as one from
#      error_sweep.py. The figures keep the shared realisation.
#      Fixed: the observation noise came from a local recipe, 0.25 * h_a(truth).std over the
#      WHOLE trajectory. error_sweep.py uses obs_std_alpha from the twin file, which is
#      cfg.obs_std / truth[obs_idx].std scaled by h(truth[obs_idx]).std — a different
#      number on all three components. This script now reads obs_std_alpha, so 4.3.3 and
#      4.3.2 describe the same experiment. Every number here moves.
# 5.10 Changed: report styling to match Stage 1/2 — axis labels Arial 14pt, legends Arial 11pt
#      (AX/FS/LEG), the three column titles and the suptitle dropped; which column a PNG is
#      comes from its filename instead
#      Added: one PNG per panel (…_{tag}_s{seed}_{trace,straddle,err_spread}_a…_w….png); the
#      seeds x 3 grid stays as the backup / log view
# 5.8  Added: terminal list of cycle times where the forecast straddle index exceeds
#      STRADDLE_THRESH, over the whole run rather than the plotted zoom window
#      Added: frac_left kept per seed, so a wrong-basin threshold (fraction past the fold,
#      which unlike the straddle index can exceed 0.5) needs no recompute
# 4.5  Added: project-root path insert so the script runs from diagnostic_py/
# 4.3  created — per-seed IEnKF bimodality diagnostic. Runs the IEnKF at every seed in
#      cfg.seeds on shared truth/obs; per-seed rows with x-trace + fold line, forecast
#      straddle index, RMS error and spread, plus failure-onset markers and a summary
#      of onset cycles so seeds can be compared for common-cycle failure.
# ============================================================

import os
import sys
import textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg, rk4
from enkf_ienkf import run_enkf_ienkf

os.makedirs('figs/diagnostic', exist_ok=True)
os.makedirs('data', exist_ok=True)

ALPHA = 0.5                                   # remedies only separate from the EnKF for alpha > 0
h_a = lambda x: x + ALPHA * x ** 2            # observation operator h_alpha
FOLD = -1.0 / (2 * ALPHA)                     # x where h_alpha turns over; scalar (= -1 at alpha=0.5)
STAT_SEEDS = list(cfg.seeds)                  # every seed: what the saved statistics use
FIG_SEEDS = STAT_SEEDS[:2]                    # seeds 0 and 1; seed 2 dropped for figure legibility
t0, t1 = 18.0, 38.0                           # zoom window in time units, as timeseries.py uses
IEC, TRU, FOLDC, MARK = '#8e44ad', '#333333', '#c0392b', '#e67e22'
AX, FS = 'Arial', 14                          # axis-label font, all report figures
LEG = dict(family=AX, size=FS)                # legend font; prop= overrides fontsize=

# --- truth: identical recipe to timeseries.py / the sweep, shared by every seed ---
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):
    s = rk4(s, cfg.dt)
truth = np.zeros((cfg.n_steps + 1, 3)); truth[0] = s
for k in range(cfg.n_steps):
    truth[k + 1] = rk4(truth[k], cfg.dt)
t_truth = np.arange(cfg.n_steps + 1) * cfg.dt
clim = truth.std(axis=0).mean()               # no-skill scale; used to define failure onset

obs_idx = np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every)
t_obs = obs_idx * cfg.dt                      # (n_cycles,) cycle times
n_obs = len(obs_idx)

# --- observation noise: obs_std_alpha from the twin file, the draw the sweep uses ---
# Two scripts describing the same alpha under different noise cannot be quoted side by side.
_tw = np.load('data/l63_twin.npz')
_ai = int(np.argmin(np.abs(_tw['alphas'] - ALPHA)))
if abs(float(_tw['alphas'][_ai]) - ALPHA) > 1e-9:
    raise SystemExit(f'ALPHA={ALPHA} is not on the twin grid {_tw["alphas"]} — '
                     f'rerun truth_obs.py or pick an alpha that is')
obs_std = _tw['obs_std_alpha'][_ai] if cfg.noise_mode == 'fixed_snr' else cfg.obs_std
h_truth = h_a(truth[obs_idx])                                    # noiseless h(truth) at obs times

# One observation realization shared across seeds, so any difference between rows is
# the FILTER's rng (initial ensemble, perturbed obs, jitter), not different data.
obs_shared = h_truth + np.random.default_rng(cfg.seed).normal(0, 1, (n_obs, 3)) * obs_std
# One realization PER SEED, drawn exactly as error_sweep.py draws it — a full replicate.
obs_replicate = {sd: h_truth + np.random.default_rng(sd).normal(0, 1, (n_obs, 3)) * obs_std
                 for sd in STAT_SEEDS}

FAIL_FRAC = 0.25                              # error above FAIL_FRAC*clim counts as failed
HOLD = 3                                      # cycles it must stay failed to call it an onset
STRADDLE_THRESH = 0.4                         # near the 0.5 ceiling: ensemble close to evenly split


def spans(mask):
    m = np.asarray(mask, dtype=bool)
    if not m.any():
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.flatnonzero(d == 1) + 1)
    stops = list(np.flatnonzero(d == -1) + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        stops = stops + [len(m)]
    return list(zip(starts, stops))


def sustained(bad, hold):
    out = np.zeros(len(bad), dtype=bool)
    for a_, b_ in spans(bad):
        if b_ - a_ >= hold:
            out[a_:b_] = True
    return out


def onset_cycle(err):
    bad = err > FAIL_FRAC * clim              # (n_obs,) boolean
    for k in range(len(bad) - HOLD + 1):
        if bad[k:k + HOLD].all():
            return k
    return None


def one_run(sd, obs):
    res = run_enkf_ienkf(obs, truth, obs_idx, h_a, seed=sd,
                         obs_std=obs_std, save_forecast=True)
    fc = res['fc_history']                                 # (n_obs, N, 3) forecast ensemble
    frac_left = (fc[:, :, 0] < FOLD).mean(axis=1)          # (n_obs,) fraction left of the fold
    straddle = np.minimum(frac_left, 1 - frac_left)        # (n_obs,) 0 = one side, 0.5 = split
    err = np.sqrt(res['sqerror']).mean(1)                  # (n_obs,) per component, then averaged
    spread = np.sqrt(res['spread']).mean(1)                # (n_obs,) same, from the variance
    lock = sustained(err > FAIL_FRAC * clim, HOLD)         # (n_obs,) inside a lock-on episode
    strad_hi = straddle > STRADDLE_THRESH                  # (n_obs,) near-evenly-split cycles
    return dict(res=res, straddle=straddle, frac_left=frac_left, err=err, spread=spread,
                lock=lock, strad_hi=strad_hi,
                onset=onset_cycle(err),
                n_strad_cycles=int(strad_hi.sum()),
                n_strad_episodes=len(spans(strad_hi)),
                n_lock_cycles=int(lock.sum()),
                n_lock_episodes=len(spans(lock)),
                # error and spread inside the lock-on periods vs outside them. A large gap in
                # err with almost none in spread is the "penalty is large and the filter is
                # unaware" claim; NaN where a seed never locked on, which is not zero.
                err_lock_max=float(err[lock].max()) if lock.any() else np.nan,
                err_lock_min=float(err[lock].min()) if lock.any() else np.nan,
                err_lock_med=float(np.median(err[lock])) if lock.any() else np.nan,
                err_free_med=float(np.median(err[~lock])) if (~lock).any() else np.nan,
                spread_lock_med=float(np.median(spread[lock])) if lock.any() else np.nan,
                spread_free_med=float(np.median(spread[~lock])) if (~lock).any() else np.nan)


# --- the figure set: one shared observation realisation, every seed ---
runs = {sd: one_run(sd, obs_shared) for sd in STAT_SEEDS}
# --- the statistics set: each seed its own observations, as error_sweep.py does ---
reps = {sd: one_run(sd, obs_replicate[sd]) for sd in STAT_SEEDS}

# --- figure: rows = seeds, cols = (x-trace | straddle | error & spread) ---
win = (t_obs >= t0) & (t_obs <= t1)
wint = (t_truth >= t0) & (t_truth <= t1)
COLS = ['trace', 'straddle', 'err_spread']     # column stems; also the split-PNG filenames


def panel(a, sd, ci, legend, ylabel_seed=True, leg_kw=None):
    R = runs[sd]
    onset = R['onset']
    t_onset = t_obs[onset] if onset is not None else None   # time of sustained failure

    if ci == 0:            # x-component trace vs truth, with the fold marked
        a.plot(t_truth[wint], truth[wint, 0], '-', color=TRU, lw=1.2, label='truth', zorder=1)
        a.plot(t_obs[win], R['res']['en_mean'][win, 0], '-', color=IEC, lw=1.5,
               label='IEnKF mean', zorder=3)
        a.axhline(FOLD, ls=':', color=FOLDC, lw=1.4, label=r'fold $x=-1/(2\alpha)$')
        a.set_ylabel(f'seed {sd}\nx' if ylabel_seed else 'x', fontname=AX, fontsize=FS)
    elif ci == 1:          # straddle index from the forecast ensemble
        a.plot(t_obs[win], R['straddle'][win], '-', color=IEC, lw=1.4)
        a.axhline(0.5, ls=':', color='#777', lw=1)          # perfectly split
        a.set_ylim(-0.02, 0.52)
        a.set_ylabel('straddle index', fontname=AX, fontsize=FS)
    else:                  # flat spread beside a big error = wrong basin, not collapse
        a.plot(t_obs[win], R['err'][win], '-', color=FOLDC, lw=1.4, label='RMS error')
        a.plot(t_obs[win], R['spread'][win], '-', color=IEC, lw=1.4, label='spread')
        a.axhline(FAIL_FRAC * clim, ls=':', color='#777', lw=1, label='failure threshold')
        a.set_ylabel('error / spread', fontname=AX, fontsize=FS)

    if t_onset is not None and t0 <= t_onset <= t1:
        a.axvline(t_onset, ls='--', color=MARK, lw=1.4, zorder=4)   # sustained-failure onset
    a.set_xlabel('Time', fontname=AX, fontsize=FS)
    a.grid(alpha=0.3)
    if legend and ci != 1:                                  # column 1 has nothing to label
        a.legend(**(leg_kw or dict(loc='upper left', prop=LEG)))


def figure(seed_list, tag):
    for sd in seed_list:
        for ci, stem in enumerate(COLS):
            fig, a = plt.subplots(figsize=(6.5, 3.6))
            panel(a, sd, ci, legend=True)
            fig.tight_layout()
            out = (f'figs/diagnostic/stage3_ienkf_seeds_{tag}_s{sd}_{stem}'
                   f'_a{ALPHA}_w{cfg.obs_every}.png')
            fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
            print(f'saved {out}')

    # combined grid, kept as the backup / log view
    fig, ax = plt.subplots(len(seed_list), 3, figsize=(16.5, 3.1 * len(seed_list)),
                           sharex='col', squeeze=False)
    for ri, sd in enumerate(seed_list):
        for ci in range(3):
            panel(ax[ri, ci], sd, ci, legend=(ri == 0))
            if ri != len(seed_list) - 1:        # sharex: only the bottom row keeps the x-label
                ax[ri, ci].set_xlabel('')
    fig.tight_layout()
    out = f'figs/diagnostic/stage3_ienkf_seeds_{tag}_a{ALPHA}_w{cfg.obs_every}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}')


def figure_stacked(sd, cols=(0, 1, 2), frac=0.75, panel_in=1.15):
    fig, ax = plt.subplots(len(cols), 1, sharex=True, squeeze=False,
                           figsize=(fs.TEXTWIDTH_IN * frac, panel_in * len(cols) + 0.45))
    ax = ax[:, 0]
    leg = fs.leg_above(ncol=3)
    for k, ci in enumerate(cols):
        panel(ax[k], sd, ci, legend=True, ylabel_seed=False, leg_kw=leg)
        fs.panel_letter(ax[k], k)                   # no subcaptions here to carry these
        if k != len(cols) - 1:
            ax[k].set_xlabel('')                    # sharex: only the bottom panel keeps it
    fig.tight_layout(h_pad=0.35)
    out = (f'figs/diagnostic/stage3_ienkf_seeds_stacked{len(cols)}_s{sd}'
           f'_a{ALPHA}_w{cfg.obs_every}.png')
    fig.savefig(out, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}')


#figure([FIG_SEEDS[1]], 'seed1')      # single seed, for the mechanism argument
#figure(FIG_SEEDS, 'allseeds')        # the plotted seeds, for the common-onset comparison
figure_stacked(FIG_SEEDS[1], cols=(0, 1))   # thesis figure with error/spread in the appendix

# --- summary: do the seeds fail at the same cycle? (shared-observation set) ---
print(f"\nalpha={ALPHA}  fold at x={FOLD}  failure = err > {FAIL_FRAC}*clim for {HOLD} cycles")
print(f"obs_std = {np.array2string(np.asarray(obs_std), precision=3)}  (from the twin file)")
print(f"\nSHARED observation realisation — the figure set, {len(STAT_SEEDS)} seeds")
print(f"{'seed':>6}{'onset cycle':>13}{'onset time':>12}{'mean straddle':>15}{'max straddle':>14}")
for sd in STAT_SEEDS:
    R = runs[sd]; o = R['onset']
    oc = 'never' if o is None else f'{o}'
    ot = '-' if o is None else f'{t_obs[o]:.2f}'
    print(f"{sd:>6}{oc:>13}{ot:>12}{R['straddle'].mean():>15.3f}{R['straddle'].max():>14.3f}")

onsets = [runs[sd]['onset'] for sd in STAT_SEEDS if runs[sd]['onset'] is not None]
if len(onsets) > 1:
    print(f"\nonset spread across seeds: min={min(onsets)} max={max(onsets)} "
          f"range={max(onsets) - min(onsets)} cycles")

elif len(onsets) == 1:
    print("\nonly one seed sustained failure — not a common-cycle failure.")
else:
    print("\nno seed sustained failure at this alpha/window.")

# --- cycles where the forecast ensemble sits close to evenly split across the fold ---
# The whole run, not the plotted window: the t0-t1 zoom is a legibility choice for the
# figure and would silently hide episodes outside it.
print(f"\ncycle times with straddle index > {STRADDLE_THRESH}"
      f"  (index = min(f,1-f), so it cannot exceed 0.5)")
for sd in STAT_SEEDS:
    k = np.flatnonzero(runs[sd]['strad_hi'])   # cycle indices into t_obs
    if k.size == 0:
        print(f'  seed {sd}: none'); continue
    print(f'  seed {sd}: {k.size} of {n_obs} cycles in '
          f'{runs[sd]["n_strad_episodes"]} episodes, '
          f'peak {runs[sd]["straddle"][k].max():.3f}')
    print(textwrap.fill(', '.join(f'{t_obs[i]:.2f}' for i in k),
                        width=96, initial_indent=' ' * 6, subsequent_indent=' ' * 6))

# --- save ---
per = lambda key, src: np.array([src[sd][key] for sd in STAT_SEEDS], dtype=float)
series = lambda key, src: np.array([src[sd][key] for sd in STAT_SEEDS])

out_path = f'data/stage3_ienkf_seeds_a{ALPHA:g}_w{cfg.obs_every}_{cfg.noise_mode}.npz'
np.savez(
    out_path,
    alpha=ALPHA, fold=FOLD, seeds=np.array(STAT_SEEDS), fig_seeds=np.array(FIG_SEEDS),
    obs_every=cfg.obs_every, noise_mode=cfg.noise_mode, obs_std=np.asarray(obs_std),
    n_cycles=n_obs, t_obs=t_obs, clim=clim,
    fail_frac=FAIL_FRAC, hold=HOLD, straddle_thresh=STRADDLE_THRESH,
    definitions=('episode = one contiguous run above the threshold; '
                 'lock-on = a run of cycles with RMS error above fail_frac*clim lasting at '
                 'least hold cycles; replicate = own observations per seed (quote these), '
                 'shared = one observation realisation for every seed (the figures)'),

    # ---- replicate set: per seed, the quotable numbers ----
    n_strad_episodes_by_seed=per('n_strad_episodes', reps),
    n_strad_cycles_by_seed=per('n_strad_cycles', reps),
    n_lock_episodes_by_seed=per('n_lock_episodes', reps),
    n_lock_cycles_by_seed=per('n_lock_cycles', reps),
    frac_lock_by_seed=per('n_lock_cycles', reps) / n_obs * 100,
    err_lock_min_by_seed=per('err_lock_min', reps),
    err_lock_max_by_seed=per('err_lock_max', reps),
    err_lock_med_by_seed=per('err_lock_med', reps),
    err_free_med_by_seed=per('err_free_med', reps),
    spread_lock_med_by_seed=per('spread_lock_med', reps),
    spread_free_med_by_seed=per('spread_free_med', reps),

    # ---- per-cycle series, replicate set, in case a later claim needs them ----
    err_by_seed=series('err', reps), spread_by_seed=series('spread', reps),
    straddle_by_seed=series('straddle', reps), frac_left_by_seed=series('frac_left', reps),
    lock_by_seed=series('lock', reps),

    # ---- shared set: what the figures show, kept so the figure can be reproduced ----
    shared_n_strad_episodes_by_seed=per('n_strad_episodes', runs),
    shared_n_strad_cycles_by_seed=per('n_strad_cycles', runs),
    shared_n_lock_episodes_by_seed=per('n_lock_episodes', runs),
    shared_err_by_seed=series('err', runs),
    shared_spread_by_seed=series('spread', runs),
    shared_straddle_by_seed=series('straddle', runs),
    shared_onset_by_seed=np.array([-1 if runs[sd]['onset'] is None else runs[sd]['onset']
                                   for sd in STAT_SEEDS]),
)
print(f'\nsaved {out_path}')