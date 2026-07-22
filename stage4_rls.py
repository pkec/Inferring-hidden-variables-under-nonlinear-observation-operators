"""
stage4_rls.py — Stage 4 SHADOW learner (offline).

Reads the logged run, learns the residual r_t = xpf_mean - xa_mean online with RLS,
records the per-cycle prediction and the RLS residual, and measures how much of the
EnKF-PF gap the learned correction WOULD close — without ever changing the filter.
This answers only one question: is the residual learnable at all?

It does NOT inject. Injection changes the trajectory (a corrected mean propagates
into the next forecast), so it cannot be replayed on a frozen log; it lives in the
live filter enkf_rls.py. Shadow is the safe first diagnostic before closing the loop.

Because nothing is injected, the "corrected" RMSE here is a POST-HOC lower-bound
estimate (xa_mean + pred vs truth on the fixed trajectory), not what a live
corrected filter would achieve — the live number comes from enkf_rls.py.
"""

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 4.1  Changed: stripped to shadow-only — removed the replayed 'inject' branch
#      (it changed nothing on a frozen log; real injection moved to enkf_rls.py).
#      Now imports the shared feature builder / standardiser / RLS from rls_core.
# 4.0  created — RLS learner for the PF-EnKF residual (features reconstructed from
#      the logged forecast ensemble; matrix RLS; shadow + replayed-inject modes).
# ============================================================

import numpy as np
from config import cfg
from rls_core import build_features, n_features, RunningStandardiser, RLS

ALPHA = 0.5                                       # must match the logged run; scalar
LAM = 1.0                                          # forgetting factor lambda in (0,1]; 1 -> batch OLS
P0 = 1e3                                            # initial P scale; large = weak prior on weights
USE_DELTA = False                                  # True only when training spans several alpha
h = lambda x: x + ALPHA * x**2                     # observation operator, for reconstructing hbar

data = np.load(f'data/stage4_log_a{ALPHA}_{cfg.noise_mode}.npz')
fc_enkf  = data['fc_enkf']        # (T, N, 3) EnKF forecast ensemble per cycle
xa_mean  = data['xa_mean']        # (T, 3) EnKF analysis mean (uncorrected run)
jensen   = data['jensen']         # (T, 3) measured E[h(x)]-h(E[x]) the filter used
xpf_mean = data['xpf_mean']       # (T, 3) PF posterior mean (the teacher / ground truth)
obs      = data['obs']            # (T, 3) observations
truth    = data['truth_at_obs']   # (T, 3) true state per cycle
T = fc_enkf.shape[0]              # number of cycles; int

# --- reconstruct forecast-side primitives from the logged forecast ensemble ---
xf_mean = fc_enkf.mean(1)                          # (T, 3) forecast mean E[x]
hbar = h(fc_enkf).mean(1)                          # (T, 3) mean predicted obs E[h(x)]
s2 = fc_enkf.var(1, ddof=1)                        # (T, 3) forecast variance per component
d = obs - hbar                                     # (T, 3) mean innovation y - E[h(x)]
delta = jensen                                     # (T, 3) measured Jensen bias
lag_d = np.vstack([np.zeros((1, 3)), d[:-1]])      # (T, 3) previous cycle's innovation; row 0 padded

r = xpf_mean - xa_mean                             # (T, 3) TARGET: residual the model learns

U_raw = build_features(d, s2, xf_mean, delta, lag_d, use_delta=USE_DELTA)   # (T, k-1)
K = n_features(U_raw.shape[1])                     # feature count incl. constant; int

std = RunningStandardiser(U_raw.shape[1])          # running z-scorer (adds the constant)
rls = RLS(K, lam=LAM, P0=P0)                        # matrix RLS state

pred = np.zeros((T, 3))                            # per-cycle prediction w_{t-1}^T u_t
rls_resid = np.zeros((T, 3))                       # per-cycle miss r_t - w_{t-1}^T u_t (convergence metric)

for t in range(T):
    u = std.transform(U_raw[t])                     # feature vector this cycle; (K,)
    pred[t] = rls.predict(u)                         # predict with OLD weights (honest online); (3,)
    rls_resid[t] = rls.update(u, r[t])              # update weights; returns the pre-update miss (3,)

# --- headline RMSEs ---
# xa_shadow is a POST-HOC estimate: it applies the full prediction to the frozen mean.
# It is NOT a live corrected filter (no propagation of the shift) — that is enkf_rls.py.
xa_shadow = xa_mean + pred                         # (T, 3) "what if we'd added the prediction"
rmse_enkf   = np.sqrt(((xa_mean   - truth) ** 2).mean())   # baseline EnKF
rmse_pf     = np.sqrt(((xpf_mean  - truth) ** 2).mean())   # PF floor
rmse_shadow = np.sqrt(((xa_shadow - truth) ** 2).mean())   # post-hoc lower-bound estimate

out = f'data/stage4_rls_a{ALPHA}_shadow.npz'
np.savez(
    out,
    pred=pred, target=r, rls_resid=rls_resid,
    xa_mean=xa_mean, xa_shadow=xa_shadow, xpf_mean=xpf_mean, truth=truth,
    w_final=rls.w, times=data['times'], alpha=ALPHA,
    rmse_enkf=rmse_enkf, rmse_pf=rmse_pf, rmse_shadow=rmse_shadow,
)
print(f"shadow lambda={LAM} | RMSE enkf={rmse_enkf:.3f} "
      f"shadow_est={rmse_shadow:.3f} pf={rmse_pf:.3f} -> {out}")
