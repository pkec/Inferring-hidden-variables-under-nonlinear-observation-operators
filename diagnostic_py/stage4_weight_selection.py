# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.42 Fixed:   analysis RMSE was POOLED over cycles and components here while every other
#      Stage 4 script moved to per-component-then-averaged in 5.18. This figure is cited
#      beside rls_blind.py's, so it was pricing the same correction a different way. Now
#      rls_core.rmse_percomp — EVERY number in this script's figure and tables moves.
#      Changed: sizing and fonts come from figstyle. The 9in export shrank by 0.54 at
#      0.72\textwidth, taking the 8pt bar labels to about 4.3pt.
#      Single-panel figure, so no letters.
# 6.0  Fixed:   train/test populations were reversed against rls_blind.py (patch 4.39).
#               Weights are now fitted on cfg.blind_seeds and scored on cfg.seeds, so the
#               figure matches the blind results it is cited alongside.
#      Changed: single-panel figure — blind seeds only; the training panel is dropped.
#               Training reductions are still computed for the printed overfit column.
# 4.27 created — replaces stage4_weight_features.py. Feature-set sweep removed (the
#      feature set is now fixed in rls_core); compares only 'final' vs 'tail_mean' weight
#      selection, and emits a single grouped bar figure (the line plot was dropped).
# ============================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg
import rls_core
from rls_core import (load_log, n_features, RunningStandardiser, RLS, dyn_row,
                      tail_mean_weights, rmse_percomp)

os.makedirs('figs/diagnostic/s4', exist_ok=True)
fs.use()
S4 = 'figs/diagnostic/s4'

ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
TRAIN_SEEDS = list(cfg.blind_seeds)                  # weights fitted here — matches rls_blind.py
TEST_SEEDS = list(cfg.seeds)                         # scored here, never seen
LAM = 0.995                                          # match rls_shadow.py
GAMMA = 1.0                                          # match rls_shadow.py (post-hoc, no feedback)
P0 = 1e3
TAIL_FRAC = 0.5                                      # average the last 50% of the trajectory
WEIGHT_MODES = ['final', 'tail_mean']
MODE_COL = {'final': '#c0392b', 'tail_mean': '#2471a3'}

tw = np.load('data/l63_twin.npz')
truth = tw['truth']; obs_idx = tw['obs_idx']
truth_at_obs = truth[obs_idx]
rmse = rmse_percomp


def train(paths, alpha):
    first = load_log(paths[0][1], alpha)
    K = n_features(first['U_raw'].shape[1])
    std = RunningStandardiser(first['U_raw'].shape[1] + rls_core.N_DYN)
    rls = RLS(K, lam=LAM, P0=P0)
    hist = []                                        # weight snapshots across ALL seeds
    for _, p in paths:
        L = load_log(p, alpha)
        T = L['U_raw'].shape[0]
        lag_pred = np.zeros(3)
        for t in range(T):
            u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]))
            lag_pred = rls.predict(u)                # own output feeds forward
            rls.update(u, L['r'][t])
            hist.append(rls.w.copy())
    hist = np.array(hist)                            # (total_cycles, K, 3)
    return rls.w.copy(), tail_mean_weights(hist, TAIL_FRAC), std


def apply_frozen(w, std, L):
    T = L['U_raw'].shape[0]
    pred = np.zeros((T, 3))
    lag_pred = np.zeros(3)
    for t in range(T):
        u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]),
                          update=False)              # FROZEN: no adaptation to test data
        pred[t] = w.T @ u
        lag_pred = pred[t]
    return pred


def score(w, std, paths, alpha):
    b, c, p_, helped = [], [], [], 0
    for _, path in paths:
        L = load_log(path, alpha); raw = np.load(path)
        pr = apply_frozen(w, std, L)
        rb = rmse(raw['xa_mean'], truth_at_obs)
        rc = rmse(raw['xa_mean'] + GAMMA * pr, truth_at_obs)
        b.append(rb); c.append(rc); p_.append(rmse(raw['xpf_mean'], truth_at_obs))
        helped += rc < rb
    be = (np.mean(b) - np.mean(p_)) / np.mean(p_) * 100     # baseline excess over PF
    ce = (np.mean(c) - np.mean(p_)) / np.mean(p_) * 100     # corrected excess over PF
    return be, ce, ((be - ce) / be * 100 if be else 0.0), helped, len(paths)


res = {m: {} for m in WEIGHT_MODES}                  # wmode -> alpha -> dict
for a in ALPHAS:
    pth = lambda sd: f'data/stage4_log_a{a}_s{sd}_{cfg.noise_mode}.npz'
    tr = [(sd, pth(sd)) for sd in TRAIN_SEEDS if os.path.exists(pth(sd))]
    te = [(sd, pth(sd)) for sd in TEST_SEEDS if os.path.exists(pth(sd))]
    if not tr or not te:
        continue

    w_final, w_tail, std = train(tr, a)              # one pass gives both weight modes
    for wmode, w in [('final', w_final), ('tail_mean', w_tail)]:
        _, _, tred, _, _ = score(w, std, tr, a)              # seen data (overfit column only)
        bbe, bce, bred, helped, n = score(w, std, te, a)     # unseen data (decides)
        res[wmode][a] = dict(train_red=tred, blind_red=bred, blind_base=bbe,
                             blind_corr=bce, helped=helped, n=n)
    print(f'alpha={a}: blind reduction  final {res["final"][a]["blind_red"]:+6.2f}%   '
          f'tail_mean {res["tail_mean"][a]["blind_red"]:+6.2f}%   '
          f'(tail_mean {res["tail_mean"][a]["blind_red"] - res["final"][a]["blind_red"]:+.2f} pp)')

A = sorted(res['final'])
if not A:
    raise SystemExit('no logs found — run stage4_log.py first')

# --- figure: grouped bars on BLIND seeds, per alpha plus the mean ---
groups = [f'{a}' for a in A] + ['mean']
x = np.arange(len(groups)); width = 0.35
fig, ax = plt.subplots(figsize=fs.size(0.72, 0.58))
for i, wmode in enumerate(WEIGHT_MODES):
    vals = [res[wmode][a]['blind_red'] for a in A]
    vals.append(float(np.mean(vals)))                # trailing 'mean' group
    bars = ax.bar(x + (i - 0.5) * width, vals, width, color=MODE_COL[wmode], label=wmode)
    ax.bar_label(bars, fmt='%.1f', fontsize=fs.BAR, fontname=fs.AX, padding=2)
ax.axhline(0, color='#333', lw=1.2)
ax.set_xticks(x); ax.set_xticklabels(groups)
ax.set_xlabel(r'$\alpha$', **fs.LAB)
ax.set_ylabel('RMSE-excess reduction (%)', **fs.LAB)   # sign convention -> caption
ax.grid(alpha=0.3, axis='y')
ax.legend(title='frozen weights', title_fontsize=fs.LEGEND, **fs.leg_above(ncol=2))
nfeat = n_features(load_log(f'data/stage4_log_a{A[0]}_s{TRAIN_SEEDS[0]}_{cfg.noise_mode}.npz',
                            A[0])['U_raw'].shape[1])
#ax.set_title(rf'Stage 4: which weights to freeze — blind seeds, $\lambda$={LAM}, '
#             rf'tail={int(TAIL_FRAC*100)}%, {nfeat} features'
#             '\n'
#             rf'train {TRAIN_SEEDS} $\rightarrow$ blind {TEST_SEEDS}')
fs.save(fig, f'{S4}/stage4_weight_selection.png')

# --- numbers ---
print('\n' + '=' * 78)
print(f'Relative RMSE-excess reduction (+ = helps). {nfeat} features, gamma={GAMMA}.')
print('=' * 78)
print(f"{'weights':>11}{'alpha':>7}{'train%':>9}{'BLIND%':>9}{'overfit':>9}{'seeds+':>8}")
for wmode in WEIGHT_MODES:
    for a in A:
        R = res[wmode][a]
        print(f"{wmode:>11}{a:>7.1f}{R['train_red']:>9.2f}{R['blind_red']:>9.2f}"
              f"{R['train_red'] - R['blind_red']:>9.2f}{R['helped']:>5}/{R['n']}")

print('\n' + '-' * 50)
print(f"{'weights':>11}{'train%':>9}{'BLIND%':>9}{'overfit':>9}")
summ = {}
for wmode in WEIGHT_MODES:
    tr_m = np.mean([res[wmode][a]['train_red'] for a in A])
    bl_m = np.mean([res[wmode][a]['blind_red'] for a in A])
    summ[wmode] = bl_m
    print(f"{wmode:>11}{tr_m:>9.2f}{bl_m:>9.2f}{tr_m - bl_m:>9.2f}")
win = max(summ, key=summ.get)
print(f"\nbetter on blind data: {win} ({summ[win]:+.2f}% vs "
      f"{summ[min(summ, key=summ.get)]:+.2f}%, "
      f"{summ[win] - summ[min(summ, key=summ.get)]:+.2f} pp)")
print("'overfit' = training reduction minus blind reduction; large = fits its own data only.")