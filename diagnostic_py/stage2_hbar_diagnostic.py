import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from a subfolder: put the project root on the import path
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import (cfg, rk4_vec, cal_ratio, cal_ratio_comp, cal_gap,
                    rmse_comp, spread_comp,
                    CAL_RATIO_CONVENTION, PERTURB_BASELINE_ALPHA)
from enkf import run_enkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.44 Changed: the calibration ratio is config.cal_ratio, imported, not computed here.
#      The arithmetic this file already performed is what cal_ratio does, so ratio_en /
#      ratio_hb do not move — but there is now one definition instead of a local copy that
#      can drift from error_sweep.py's, which is how the alpha=0 disagreement started.
#      Fixed: the "NOT changed: ratio_en / ratio_hb remain the POOLED spread/RMSE" notes in
#      5.34 and 5.21 described the code incorrectly. Since 5.21 calibrate() and run_at() have
#      returned (sqrt(spread.mean(0)) / sqrt(sqerror.mean(0))).mean() — per component, then
#      averaged — not the pooled form error_sweep.py used. The npz was right; the note was
#      wrong, and it was the note that made the two files look reconciled when they were not.
#      Changed: the jitter scan now minimises |cal_ratio - 1| (config.cal_gap) instead of
#      |mean_i(RMSE_i/spread_i) - 1|, so the multiplier is chosen on the quantity that gets
#      reported. Picks move at some alpha.
#      Changed: at alpha = PERTURB_BASELINE_ALPHA (0.0) the scan does not run — the window
#      baseline is used as-is, matching error_sweep.py and stage1_results.py. The alpha=0
#      column of all three files is now the same run.
#      Added: cal_ratio_convention stamped into the npz, and the cache check refuses a file
#      written before this patch instead of plotting it under the new labels.
# 5.34 Changed: removed all legacy *_pooled quantities from the npz.
#      Added: every seed-level result is now saved explicitly as *_by_seed, with shape
#      (A, S) for headline quantities and (A, S, 3) for per-component quantities.
#      Changed: *_std now uses sample standard deviation (ddof=1), since cfg.seeds are
#      treated as a sample of the stochastic seed population.
#      NOT changed: ratio_en / ratio_hb remain the POOLED spread/RMSE within each seed,
#      exactly as error_sweep.py's en_ratio does. The seed-level ratios are then averaged
#      over seeds.
# 5.33 Changed: panel order is now penalty | calibration | spread.
# 5.32 Changed: sizing and fonts come from figstyle.py.
# 5.21 Changed: RMSE, spread and the penalty are per-component-then-averaged. The penalty
#      is a RATIO, so it is formed per component and only then averaged.
#      NOT changed: ratio_en / ratio_hb are still the POOLED spread/RMSE, exactly as
#      error_sweep.py's en_ratio is.
# 5.10 Changed: report styling to match Stage 1/2.
# 5.1x Added: per-cycle RMSE excess over PF diagnostic.
# 4.43 created — h(E[x])-anchored EnKF vs the standard E[h(x)] filter across alpha.
# ============================================================

os.makedirs('figs/results', exist_ok=True)
os.makedirs('data', exist_ok=True)

W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
fs.use()
AX, FS = fs.AX, fs.FS
FRAC, ASP = 0.32, fs.XY

out = f'data/stage2_hbar_w{W}_{mode}_{jit}.npz'


def run_hbar(obs, truth, obs_idx, h, seed=cfg.seed, N=cfg.ensembleN, obs_std=None):
    rng = np.random.default_rng(seed)
    obs_std = cfg.obs_std if obs_std is None else obs_std
    R = np.diag(obs_std ** 2)
    ens = np.tile(truth[0], (N, 1)) + rng.normal(0, 1, (N, 3)) * cfg.init_std

    n = len(obs)
    sqerror = np.zeros((n, 3)); spread = np.zeros((n, 3))
    jensen = np.zeros((n, 3)); shift = np.zeros((n, 3))
    tao = truth[obs_idx]

    for k in range(n):
        ens += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ens = rk4_vec(ens, cfg.dt)

        xbar = ens.mean(0)
        D = rng.multivariate_normal(obs[k], R, N)
        Y = h(ens)
        S = Y - Y.mean(0)
        delta = Y.mean(0) - h(xbar)                          # Jensen gap in observation space
        M = linalg.inv((N - 1) * R + S.T @ S) @ (S.T @ ens)
        ens = ens + (D - Y + delta) @ M                      # delta re-anchors the innovation

        jensen[k] = delta
        shift[k] = delta @ M
        mu = ens.mean(0)
        sqerror[k] = (tao[k] - mu) ** 2
        spread[k] = ((ens - mu) ** 2).mean(0)

    return dict(sqerror=sqerror, spread=spread, jensen=jensen, shift=shift)


data = np.load('data/l63_twin.npz')
truth = data['truth']; obs_idx = data['obs_idx']
obs_std_alpha = data['obs_std_alpha']; alphas = data['alphas']
truth_at_obs = truth[obs_idx]

seeds = cfg.seeds
S_ = len(seeds)
noise = [np.random.default_rng(s).normal(0, 1, (len(obs_idx), 3)) for s in seeds]

clim = truth.std(axis=0).mean()          # no-skill scale, used to reject diverged runs
base = cfg.perturb_std.copy()
jitter_mults = np.geomspace(0.3, 30, 9)


def calibrate(run_fn, obs, h, ostd, seed):
    best_cal, best_any, last = None, None, None

    for m in jitter_mults:
        cfg.perturb_std = base * m
        o = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd)
        last = o

        rm = rmse_comp(o).mean()          # equal-weight headline RMSE for this run
        if not np.isfinite(rm):
            continue

        if best_any is None or rm < best_any[1]:
            best_any = (o, rm, m)

        if rm < 0.25 * clim:
            gap = cal_gap(o)              # |calibration ratio - 1|, the quantity reported below
            if best_cal is None or gap < best_cal[1]:
                best_cal = (o, gap, m)

    cfg.perturb_std = base

    if best_any is None:
        print(f'  ! {run_fn.__name__} diverged at every jitter multiplier '
              f'(seed={seed}) — recording NaN')
        return last, np.nan, np.nan

    pick = best_cal if best_cal is not None else best_any
    return pick[0], pick[2], cal_ratio(pick[0])


def run_at(run_fn, obs, h, ostd, m, seed):
    """Single run at a preset jitter multiple."""
    cfg.perturb_std = base * m
    o = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd)
    cfg.perturb_std = base

    ratio = cal_ratio(o)
    if not np.isfinite(ratio):
        print(f'  ! {run_fn.__name__} diverged at the fixed jitter multiplier '
              f'm={m} (seed={seed}) — ratio recorded as NaN')

    return o, m, ratio


def tune(run_fn, obs, h, ostd, seed, alpha):
    if cfg.jitter_mode == 'fixed' or float(alpha) == PERTURB_BASELINE_ALPHA:
        return run_at(run_fn, obs, h, ostd, 1.0, seed)
    return calibrate(run_fn, obs, h, ostd, seed)


keys = ('rmse_en', 'rmse_hb', 'spread_en', 'spread_hb', 'ratio_en', 'ratio_hb',
        'penalty', 'jensen', 'shift', 'en_mult', 'hb_mult')

ckeys = ('rmse_en_c', 'rmse_hb_c', 'spread_en_c', 'spread_hb_c', 'penalty_c',
         'ratio_en_c', 'ratio_hb_c')

CONVENTION = 'RMSE, spread and penalty: per-component then averaged (5.21); '

if os.path.exists(out):
    print(f'{out} found — plotting from it (delete it to re-run the sweep)')
    d = np.load(out)

    if 'cal_ratio_convention' not in d.files:
        print(f'WARNING: {out} predates 5.44 (no cal_ratio_convention key). Its alpha=0\n'
              f'column came from a jitter scan rather than the window baseline, so it does\n'
              f'not match stage1_results.py or error_sweep.py. Delete it and rerun.')
    elif 'rmse_convention' not in d.files:
        print(f'WARNING: {out} predates the current convention. Delete it and rerun.')
    elif not all((k + '_by_seed') in d.files for k in keys):
        print(f'WARNING: {out} carries no per-seed arrays. Delete it and rerun.')
    else:
        print(f'RMSE convention: {str(d["rmse_convention"])}')

else:
    acc = {k: np.zeros((len(alphas), S_)) for k in keys}          # (A, S)
    accc = {k: np.zeros((len(alphas), S_, 3)) for k in ckeys}     # (A, S, 3)

    for ai, a in enumerate(alphas):
        h = lambda x, a=a: x + a * x**2
        h_truth = truth_at_obs + a * truth_at_obs**2

        for si, s in enumerate(seeds):
            obs = h_truth + noise[si] * obs_std_alpha[ai]

            en, em, er = tune(run_enkf, obs, h, obs_std_alpha[ai], s, a)
            hb, hm, hr = tune(run_hbar, obs, h, obs_std_alpha[ai], s, a)

            # per component first, via the shared definitions in config.py so this file
            # cannot drift from error_sweep.py again
            ren_c = rmse_comp(en); rhb_c = rmse_comp(hb)
            spen_c = spread_comp(en); sphb_c = spread_comp(hb)
            pen_c = (rhb_c - ren_c) / ren_c * 100       # a ratio, so per component then averaged

            for k_, v_ in (('rmse_en_c', ren_c), ('rmse_hb_c', rhb_c),
                           ('spread_en_c', spen_c), ('spread_hb_c', sphb_c),
                           ('penalty_c', pen_c),
                           ('ratio_en_c', cal_ratio_comp(en)),
                           ('ratio_hb_c', cal_ratio_comp(hb))):
                accc[k_][ai, si] = v_

            acc['rmse_en'][ai, si] = ren_c.mean()
            acc['rmse_hb'][ai, si] = rhb_c.mean()
            acc['penalty'][ai, si] = pen_c.mean()
            acc['spread_en'][ai, si] = spen_c.mean()
            acc['spread_hb'][ai, si] = sphb_c.mean()

            # er / hr are config.cal_ratio for this seed: mean over i in x,y,z of
            # spread_i / RMSE_i, with RMSE_i = sqrt(mean_k sqerror[k,i]). By construction
            # that equals ratio_*_c.mean(), which is asserted below.
            acc['ratio_en'][ai, si] = er
            acc['ratio_hb'][ai, si] = hr

            acc['en_mult'][ai, si] = em
            acc['hb_mult'][ai, si] = hm
            acc['jensen'][ai, si] = np.linalg.norm(hb['jensen'], axis=1).mean()
            acc['shift'][ai, si] = np.linalg.norm(hb['shift'], axis=1).mean()

        nd = int(np.isnan(acc['rmse_hb'][ai]).sum())
        print(f"alpha={a:4.1f}  "
              f"RMSE en={np.nanmean(acc['rmse_en'][ai]):.3f} "
              f"hbar={np.nanmean(acc['rmse_hb'][ai]):.3f}  "
              f"penalty={np.nanmean(acc['penalty'][ai]):+6.1f}%  "
              f"||delta||={np.nanmean(acc['jensen'][ai]):.3f}  "
              f"||K delta||={np.nanmean(acc['shift'][ai]):.3f}"
              + (f"  [{nd}/{S_} seeds diverged]" if nd else ''))

    # the headline ratio must be the mean of the per-component ratios. The two are built by
    # different routes — headline from tune(), components in the loop — so this catches any
    # future edit that reintroduces a second definition of the calibration ratio.
    for f_ in ('en', 'hb'):
        lhs = acc[f'ratio_{f_}']
        rhs = accc[f'ratio_{f_}_c'].mean(axis=2)
        bad = ~np.isclose(lhs, rhs, rtol=1e-12, atol=0, equal_nan=True)
        if bad.any():
            ai_, si_ = np.argwhere(bad)[0]
            raise SystemExit(f'ratio_{f_} disagrees with mean(ratio_{f_}_c) at '
                             f'alpha={alphas[ai_]}, seed={seeds[si_]}: '
                             f'{lhs[ai_, si_]!r} vs {rhs[ai_, si_]!r}. Two definitions of '
                             f'the calibration ratio are live again — see config.cal_ratio.')

    save = {
        'alphas': alphas,
        'seeds': np.array(seeds),
        'rmse_convention': CONVENTION,
        'cal_ratio_convention': CAL_RATIO_CONVENTION,
        'baseline_alpha': PERTURB_BASELINE_ALPHA,
        'n_diverged': np.isnan(acc['rmse_hb']).sum(1),
        **{k: np.nanmean(acc[k], axis=1) for k in keys},                     # seed means
        **{k + '_std': np.nanstd(acc[k], axis=1, ddof=1) for k in keys},     # sample SD over seeds
        **{k + '_by_seed': acc[k] for k in keys},
        **{k: np.nanmean(accc[k], axis=1) for k in ckeys},                   # (A, 3)
        **{k + '_std': np.nanstd(accc[k], axis=1, ddof=1) for k in ckeys},   # (A, 3)
        **{k + '_by_seed': accc[k] for k in ckeys},                          # (A, S, 3)
    }

    np.savez(out, **save)
    print(f'\nsaved {out}')
    d = np.load(out)


EN, HB = '#c0392b', '#8e44ad'
a = d['alphas']


def sd(k):
    return d[k] if k in d.files else np.zeros_like(a)


def band(ax, xs, y, s, col, label, mk='o-'):
    ax.plot(xs, y, mk, color=col, lw=2, label=label)
    ax.fill_between(xs, y - s, y + s, color=col, alpha=0.2)


PANELS = [('penalty', 'excess over standard EnKF (pp)'),
          ('calibration', 'spread / RMSE'),
          ('spread', 'spread')]
LEGEND_PANEL = 'calibration'


def panel(ax, stem, ylab, merged=False):
    if stem == 'spread':
        band(ax, a, d['spread_en'], sd('spread_en_std'), EN, r'EnKF, $\overline{h^f}$')
        band(ax, a, d['spread_hb'], sd('spread_hb_std'), HB, r'EnKF, $h(\overline{x^f})$')
        if not merged:
            ax.legend(**fs.LEG)

    elif stem == 'penalty':
        ax.axhline(0, ls='--', color='#777', lw=1)
        band(ax, a, d['penalty'], sd('penalty_std'), HB, None)

    else:
        ax.axhspan(0.95, 1.05, color='green', alpha=0.10, label='target 1 ± 0.05')
        ax.axhline(1, ls='--', color='k', lw=1)
        band(ax, a, d['ratio_en'], sd('ratio_en_std'), EN, r'EnKF, $\overline{h}$')
        band(ax, a, d['ratio_hb'], sd('ratio_hb_std'), HB, r'EnKF, $h(\overline{x})$')
        if not merged:
            ax.legend(**fs.LEG)

    ax.set_xlabel(r'$\alpha$', fontname=AX, fontsize=FS)
    ax.grid(alpha=0.3)

    if merged:
        fs.compact(ax, ylab)
    else:
        ax.set_ylabel(ylab, fontname=AX, fontsize=FS)


for stem, ylab in PANELS:
    fig, ax = plt.subplots(figsize=fs.size(FRAC, ASP))
    panel(ax, stem, ylab)
    fs.save(fig, f'figs/results/stage2_hbar_{stem}_w{W}.png')

fig, axes = plt.subplots(1, 3, figsize=fs.size(1.0, 0.32))
for i, (stem, ylab) in enumerate(PANELS):
    panel(axes[i], stem, ylab, merged=True)
    fs.panel_letter(axes[i], i)

li = [st for st, _ in PANELS].index(LEGEND_PANEL)      # one legend for the row
h_, l_ = axes[li].get_legend_handles_labels()
fig.legend(h_, l_, **fs.leg_fig(ncol=len(l_)))
fig.tight_layout(w_pad=0.4)
fs.save(fig, f'figs/results/stage2_hbar_w{W}.png', tight=False)