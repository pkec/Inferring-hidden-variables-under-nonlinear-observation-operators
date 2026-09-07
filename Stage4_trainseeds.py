# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
#      Added: merged 1x2 with (a)/(b) drawn in, sharing the y axis. The shared range was
#      already the point of the figure, and side by side it is read directly. One
#      figure-level legend over the row — both panels use the same seed colours.
#      Changed: sizing and fonts come from figstyle. The 7.5in exports shrank by 0.43 at
#      0.32\textwidth, taking 14pt labels to 6pt.
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching rls_blind.py — this script mirrors its scoring, so the two must move together
#      or the trainseeds figure would price the correction differently from the blind figure.
# 5.12 created — blind shadow performance against the number of training trajectories.
#      Trains on subsets of cfg.blind_seeds of size 1, 2, 3 and scores every model on
#      cfg.seeds, emitting one per-blind-seed excess-removed bar chart per training size
#      on a shared y-axis.
#      NOTE: train_shadow / apply_blind / blind_shadow below are MIRRORED from rls_blind.py
#      rather than imported, because that script does its work at module level and importing
#      it would run the whole blind test. They must be kept in sync — if rls_blind.py's
#      training changes and this copy does not, this figure stops describing the model the
#      rest of Stage 4 reports on.
# ============================================================

import os
import itertools
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg
from rls_core import (load_log, n_features, RunningStandardiser, RLS,
                      dyn_row, N_DYN, tail_mean_weights, rmse_percomp)

os.makedirs('figs/results', exist_ok=True)
fs.use()

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TRAIN_POOL = list(cfg.blind_seeds)          # weights are fitted on subsets of these
TEST_SEEDS = list(cfg.seeds)                # every model is scored on these, unchanged
if set(TRAIN_POOL) & set(TEST_SEEDS):
    raise SystemExit(f'training pool {TRAIN_POOL} overlaps the test seeds {TEST_SEEDS} '
                     f'— the test would not be blind')

# shadow's offline parameters, copied from rls_blind.py PARAMS['shadow']
LAM, GAMMA, TAIL_FRAC = 1.0, 0.7, 0.5
P0 = 1e3
USE_DELTA = False
SUBSETS = os.environ.get('L63_TS_SUBSETS', 'all')      # 'all' | 'prefix'

SEED_COL = ['#2471a3', '#e67e22', '#16a085', '#8e44ad', '#c0392b']   # one per blind seed

tw = np.load('data/l63_twin.npz')
truth, obs_idx = tw['truth'], tw['obs_idx']
truth_at_obs = truth[obs_idx]
rmse = rmse_percomp                         # per component, then averaged (5.18)
pth = lambda a, sd: f'data/stage4_log_a{a}_s{sd}_{cfg.noise_mode}.npz'


# ---- mirrored from rls_blind.py; see the NOTE in the changelog -------------------------
def train_shadow(paths, alpha):
    """Offline RLS across the training seeds, state carried forward. Returns (w, std)."""
    first = load_log(paths[0], alpha, use_delta=USE_DELTA)
    K = n_features(first['U_raw'].shape[1])
    std = RunningStandardiser(first['U_raw'].shape[1] + N_DYN)
    rls = RLS(K, lam=LAM, P0=P0)
    hist = []
    for p in paths:
        L = load_log(p, alpha, use_delta=USE_DELTA)
        T = L['U_raw'].shape[0]
        lag_pred = np.zeros(3)
        for t in range(T):
            u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]))
            lag_pred = rls.predict(u)            # own prediction feeds the next cycle
            rls.update(u, L['r'][t])             # target seen here, and only here
            hist.append(rls.w.copy())
    return tail_mean_weights(np.array(hist), TAIL_FRAC), std


def apply_blind(w, std, L):
    """Frozen weights on one run, without reading its target. std uses update=False so the
    standardiser cannot adapt to the test run."""
    T = L['U_raw'].shape[0]
    pred = np.zeros((T, 3))
    lag_pred = np.zeros(3)
    for t in range(T):
        u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]),
                          update=False)
        pred[t] = w.T @ u
        lag_pred = pred[t]                       # autoregressive, own output only
    return pred


# ---- train at each size, score every model on the same blind seeds ----------------------
def subsets_of_size(k):
    if SUBSETS == 'prefix':
        return [tuple(TRAIN_POOL[:k])]
    return list(itertools.combinations(TRAIN_POOL, k))


# k=2 dropped. The claim is 'one training run does not transfer, three do', so the figure
# only needs the two ends of the ladder; k=2 sits between them and says nothing extra.
# The printed table below still covers whatever SIZES holds, so restore the full range here
# if the middle point is ever wanted back.
SIZES = sorted({1, len(TRAIN_POOL)})
# removed[k][ai, si] -> list over training subsets of percentage points of excess removed
removed = {k: [[[] for _ in TEST_SEEDS] for _ in ALPHAS] for k in SIZES}

for k in SIZES:
    for combo in subsets_of_size(k):
        for ai, a in enumerate(ALPHAS):
            tr = [pth(a, sd) for sd in combo if os.path.exists(pth(a, sd))]
            te = [(sd, pth(a, sd)) for sd in TEST_SEEDS if os.path.exists(pth(a, sd))]
            if len(tr) != k or not te:
                print(f'k={k} {combo} alpha={a}: missing logs '
                      f'({len(tr)}/{k} train, {len(te)} test) — skipped')
                continue
            w, std = train_shadow(tr, a)
            for si, (sd, p) in enumerate(te):
                L = load_log(p, a, use_delta=USE_DELTA)
                raw = np.load(p)
                xa_corr = raw['xa_mean'] + GAMMA * apply_blind(w, std, L)
                r_b = rmse(raw['xa_mean'], truth_at_obs)
                r_c = rmse_percomp(xa_corr, truth_at_obs, nan=True)
                r_p = rmse(raw['xpf_mean'], truth_at_obs)
                # both excesses are % of the PF floor, so their difference is in
                # percentage points — the same convention as stage3_results.py
                removed[k][ai][si].append((r_b - r_p) / r_p * 100 - (r_c - r_p) / r_p * 100)
    print(f'k={k}: trained {len(subsets_of_size(k))} subset(s) of {TRAIN_POOL} '
          f'x {len(ALPHAS)} alphas')

mean = {k: np.array([[np.mean(v) if v else np.nan for v in row] for row in removed[k]])
        for k in SIZES}                                   # (n_alpha, n_seed)
std_ = {k: np.array([[np.std(v) if len(v) > 1 else 0.0 for v in row] for row in removed[k]])
        for k in SIZES}

if all(np.isnan(mean[k]).all() for k in SIZES):
    raise SystemExit('no results — run stage4_log.py for cfg.blind_seeds and cfg.seeds first')


# ---- one PNG per training size, on a SHARED y-axis -------------------------------------
# The shared range is the whole point: separately autoscaled bar charts would hide that the
# k=1 bars are small or negative and the k=3 bars large.
lo = min(np.nanmin(mean[k] - std_[k]) for k in SIZES)
hi = max(np.nanmax(mean[k] + std_[k]) for k in SIZES)
pad = 0.08 * max(hi - lo, 1e-9)
YLIM = (min(lo - pad, -pad), hi + pad)                    # always show the zero line

x = np.arange(len(ALPHAS))
width = 0.8 / max(len(TEST_SEEDS), 1)


def bars(ax, k, legend):
    ax.axhline(0, color='#333', lw=1.2)                   # 0 = correction changed nothing
    for si, sd in enumerate(TEST_SEEDS):
        off = (si - (len(TEST_SEEDS) - 1) / 2) * width
        ax.bar(x + off, mean[k][:, si], width, color=SEED_COL[si % len(SEED_COL)],
               yerr=std_[k][:, si] if np.any(std_[k][:, si]) else None,
               capsize=2, ecolor='#444', label=f'seed {sd}')
    ax.set_xticks(x); ax.set_xticklabels([f'{a:g}' for a in ALPHAS])
    ax.set_xlabel(r'$\alpha$', **fs.LAB)
    ax.set_ylabel('excess removed (pp)', **fs.LAB)
    ax.set_ylim(*YLIM)
    ax.grid(alpha=0.3, axis='y'); ax.set_axisbelow(True)
    if legend:
        ax.legend(**fs.leg_above(ncol=len(TEST_SEEDS)))


# one standalone PNG per training size
for k in SIZES:
    fig, ax = plt.subplots(figsize=fs.size(0.48))
    bars(ax, k, legend=True)
    fs.save(fig, f'figs/results/stage4_trainseeds_shadow_n{k}.png')

# merged — the write-up figure. The shared y-axis is the whole point, and side by side it
# is read directly rather than by flipping between two pages.
fig, axes = plt.subplots(1, len(SIZES), figsize=fs.size(1.0, 0.36), squeeze=False, sharey=True)
for i, k in enumerate(SIZES):
    bars(axes[0, i], k, legend=False)   # both panels carry the same seed colours
    fs.panel_letter(axes[0, i], i)      # no subcaptions on a merged figure to carry these
    if i:
        axes[0, i].set_ylabel('')
fig.tight_layout(w_pad=0.4)
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, **fs.leg_fig(ncol=len(TEST_SEEDS)))   # one legend centred over the row
fs.save(fig, 'figs/results/stage4_trainseeds_shadow.png', tight=False)


# ---- printed table: the numbers behind the three figures --------------------------------
print(f'\npercentage points of excess removed by the blind shadow correction '
      f'(+ = helped), subsets = {SUBSETS}')
print(f"{'k':>3}{'alpha':>7}" + ''.join(f'{f"seed {sd}":>12}' for sd in TEST_SEEDS)
      + f"{'mean':>10}{'negative':>10}")
for k in SIZES:
    for ai, a in enumerate(ALPHAS):
        row = mean[k][ai]
        neg = int(np.sum(row < 0))
        print(f'{k:>3}{a:>7.1f}' + ''.join(f'{v:>12.2f}' for v in row)
              + f'{np.nanmean(row):>10.2f}{neg:>7}/{len(row)}')
    print()

print('negative = blind seeds whose RMSE the correction made worse. If that count falls as k')
print('rises, the weights are generalising rather than fitting each training run.')