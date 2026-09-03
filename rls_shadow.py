
# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching error_sweep.py 5.16. Print-only here (nothing is saved), but the
#      per-seed and per-alpha lines are quoted, so they must not be on a different footing
#      from the figures.
# 4.33 Changed: online shadow parameters set from the sweep — LAM 0.995, GAMMA 1.0 -> 0.7.
# 4.32 Removed: weight saving. rls_blind.py now trains its own model with its own
#      parameters; a weight file written here would be unused and could go stale.
# 4.28 Changed: the saved weights were the converged-tail mean rather than the final
#      iterate (superseded by 4.32).
# 4.26 Changed: renamed rls_shadow.py -> rls_shadow.py; weights saved under mode='shadow'
#      so they cannot collide with the inject learner's model.
# 4.21 Added: saves the trained weights + standardiser statistics via save_weights, so
#      rls_blind.py and the inject filter can load exactly this model instead of
#      re-deriving it. One training run, one source of truth.
# 4.19 Added: GAMMA (default 1.0) — shadow previously applied the FULL prediction while
#      rls_blind.py applied 0.05 of it, so the two were not comparable. Both now default
#      to 1.0, and gamma stays a knob for sweeping.
# 4.18 Changed: feature vector now carries lag_xf plus the per-cycle dynamic block
#      (lag_pred, cycle fraction); the loop threads the model's own prediction forward.
# 4.13 Added: saves the per-cycle innovation (first feature block) so stage4_results can
#      overlay it on the residual trace.
# 4.9  Changed: trains across multiple seeds per alpha (RLS state and standardiser carry
#      forward between seeds) and sweeps ALPHAS; reads the seed-tagged log files. Feature
#      reconstruction moved to rls_core.load_log so the seed diagnostic shares it.
# 4.2  Changed: records the per-cycle weight trajectory (w_hist) and saves via
#      rls_core.save_stage4 so shadow/inject/blind share one schema for plotting.
# 4.1  Changed: stripped to shadow-only — removed the replayed 'inject' branch
#      (it changed nothing on a frozen log; real injection moved to rls_inject.py).
#      Now imports the shared feature builder / standardiser / RLS from rls_core.
# 4.0  created — RLS learner for the PF-EnKF residual.
# ============================================================

import os
import numpy as np
from config import cfg
from rls_core import (load_log, n_features, RunningStandardiser, RLS, save_stage4,
                      dyn_row, N_DYN, rmse_percomp)

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]                # nonlinearity strengths to learn; list
SEEDS = cfg.seeds                                  # seeds trained in order, weights carried forward
# ONLINE shadow parameters: this is the learning run, weights updating every cycle. The
# BLIND (frozen-weight) counterpart has its own lambda and gamma in rls_blind.py — they are
# separate objectives and need not agree. From diagnostic_py/stage4_sweep.py.
LAM = 0.995                                          # forgetting factor lambda in (0,1]; 1 -> batch OLS
# Correction strength. Shadow is post-hoc — the correction never feeds back, so nothing can
# compound and there is no stability reason to damp it; gamma here is purely an accuracy
# choice, taken from the sweep rather than set to 1 on principle.
GAMMA = 0.7
P0 = 1e3                                            # initial P scale; large = weak prior on weights
USE_DELTA = False                                  # True only when training spans several alpha
# NOTE: this script no longer saves a weight file. rls_blind.py trains its own model with
# its own parameters, because "best on the seeds we trained on" and "best once frozen and
# applied to an unseen run" are different objectives and need not share a lambda. The full
# weight trajectory is still saved inside the result npz as w_hist if it is wanted.

for ALPHA in ALPHAS:
    # gather this alpha's seed files, skipping any that were not logged
    paths = []
    for s in SEEDS:
        p = f'data/stage4_log_a{ALPHA}_s{s}_{cfg.noise_mode}.npz'
        if os.path.exists(p):
            paths.append((s, p))
    if not paths:
        print(f'alpha={ALPHA}: no log files found, skipping')
        continue

    # RLS state is created ONCE per alpha and carried across every seed
    first = load_log(paths[0][1], ALPHA, use_delta=USE_DELTA)
    K = n_features(first['U_raw'].shape[1])         # feature count incl. constant; int
    std = RunningStandardiser(first['U_raw'].shape[1] + N_DYN)   # sized for static + dynamic block
    rls = RLS(K, lam=LAM, P0=P0)                     # matrix RLS state, shared across seeds

    all_pred, all_r, all_resid, all_w, all_innov, all_seed = [], [], [], [], [], []
    all_xa, all_xpf, all_truth, all_times = [], [], [], []
    t_offset = 0.0                                   # so concatenated seeds have increasing time

    for s, path in paths:
        L = load_log(path, ALPHA, use_delta=USE_DELTA)
        U_raw, r = L['U_raw'], L['r']                # (T, k-1) features, (T, 3) target
        T = U_raw.shape[0]                           # cycles in this seed; int

        pred = np.zeros((T, 3))                      # per-cycle prediction w_{t-1}^T u_t
        resid = np.zeros((T, 3))                     # per-cycle miss r_t - w_{t-1}^T u_t
        w_hist = np.zeros((T, K, 3))                 # weight trajectory within this seed

        lag_pred = np.zeros(3)                       # model's own previous prediction; (3,)
        for t in range(T):
            # static block + dynamic block (lag_pred, cycle fraction)
            u = std.transform(np.concatenate([U_raw[t], dyn_row(lag_pred, t, T)]))   # (K,)
            pred[t] = rls.predict(u)                  # predict with OLD weights; (3,)
            resid[t] = rls.update(u, r[t])           # update weights; returns pre-update miss (3,)
            w_hist[t] = rls.w                         # (K,3) weights after this cycle
            lag_pred = pred[t]                        # feed this prediction forward

        all_innov.append(U_raw[:, :3])           # first feature block is the innovation d
        all_seed.append(np.full(T, s))           # which seed each cycle came from
        all_pred.append(pred); all_r.append(r); all_resid.append(resid); all_w.append(w_hist)
        all_xa.append(L['xa_mean']); all_xpf.append(L['xpf_mean']); all_truth.append(L['truth'])
        all_times.append(L['times'] + t_offset)      # offset so seeds lay end-to-end on the time axis
        t_offset = all_times[-1][-1] + (L['times'][1] - L['times'][0])

        rm_e = rmse_percomp(L['xa_mean'], L['truth'])
        rm_s = rmse_percomp(L['xa_mean'] + GAMMA * pred, L['truth'])
        print(f'  alpha={ALPHA} seed={s}: RMSE enkf={rm_e:.3f} shadow_est={rm_s:.3f} '
              f'({(rm_e - rm_s) / rm_e * 100:+.1f}% vs enkf)')

    # concatenate every seed into one record for this alpha
    pred = np.vstack(all_pred); r = np.vstack(all_r); resid = np.vstack(all_resid)
    w_hist = np.vstack(all_w); xa_mean = np.vstack(all_xa)
    xpf_mean = np.vstack(all_xpf); truth = np.vstack(all_truth); times = np.concatenate(all_times)

    xa_shadow = xa_mean + GAMMA * pred               # post-hoc corrected mean estimate
    rmse_enkf = rmse_percomp(xa_mean, truth)
    rmse_pf = rmse_percomp(xpf_mean, truth)
    rmse_shadow = rmse_percomp(xa_shadow, truth)

    out = save_stage4('shadow', ALPHA, times, pred, r, resid,
                      xa_corr=xa_shadow, xa_base=xa_mean, xpf_mean=xpf_mean, truth=truth,
                      w_hist=w_hist, sqerror_corr=(xa_shadow - truth) ** 2,
                      innov=np.concatenate(all_innov), row_seed=np.concatenate(all_seed))
    print(f'| alpha={ALPHA} [{len(paths)} seeds, {len(times)} cycles] lambda={LAM} | \n'
          f'RMSE enkf={rmse_enkf:.3f} shadow_est={rmse_shadow:.3f} pf={rmse_pf:.3f} -> {out}')
    print()