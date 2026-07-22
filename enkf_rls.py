"""
enkf_rls.py — EnKF with a live RLS residual correction (Stage 4 inject experiment).

A modified copy of enkf.run_enkf. Everything about the filter is unchanged EXCEPT
that, after each analysis step, a learned correction is injected into the ensemble
and the RLS weights are updated online. Mirrors how enkf_qr.py / enkf_ienkf.py are
modified copies of enkf.py: same loop, one changed step.

Per cycle, in this exact order (the ordering the user specified):
  1. propagate + analysis update (standard EnKF)                -> corrected-so-far mean
  2. build features u from the CURRENT forecast ensemble
  3. predict correction with OLD weights:  y_hat = w^T u
  4. INJECT: shift every ensemble member by gamma * y_hat        (mean moves, spread kept)
  5. form target r_t = xpf_teacher[k] - (corrected analysis mean)
  6. UPDATE weights on (u, r_t)                                  (learn from how the old
                                                                  weights just landed)
  7. propagate the CORRECTED ensemble into the next cycle

Why old weights act before the update (step 3 before step 6): the model must act with
what it currently knows, then learn from the result — this is how the future blind
filter (frozen weights, no teacher) will behave, so training in this order keeps the
learned model faithful to deployment.

The PF is fixed ground truth: its per-cycle mean is read from the log (it is never
affected by the correction, so it needs no rerun). If the corrected EnKF drifts from
the PF, that drift is the honest verdict on the method — a working correction should
track the PF, not diverge from it.
"""

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 4.1  created — live RLS-corrected EnKF. Copy of enkf.run_enkf with an in-loop
#      inject-then-learn step (old weights inject, target formed against the
#      corrected mean, weights updated, corrected ensemble propagated). PF teacher
#      read from the Stage 4 log. Fixed conservative gamma. Uses shared rls_core.
# ============================================================

import numpy as np
from scipy import linalg
from config import cfg, rk4_vec
from rls_core import build_features, n_features, RunningStandardiser, RLS


def _analysis(Af, d_obs, Cdd, h, rng):
    """Standard stochastic EnKF analysis (identical to enkf.EnKF), returning the
    analysis ensemble plus the per-cycle primitives the corrector needs.
        Af    : (Nm, 3) forecast ensemble
        d_obs : (3,) observation vector
        Cdd   : (3, 3) observation-error covariance
    Returns Aa (Nm,3), hbar (3,), xf_mean (3,), s2 (3,), delta (3,).
    """
    Nm = Af.shape[0]
    psi_f_m = np.mean(Af, 0, keepdims=True)         # (1,3) forecast mean E[x]
    D = rng.multivariate_normal(d_obs, Cdd, Nm)     # (Nm,3) perturbed obs
    Y = h(Af)                                       # (Nm,3) per-member predicted obs h(x_i)
    Y_mean = np.mean(Y, axis=0, keepdims=True)      # (1,3) E[h(x)]
    S = Y - Y_mean                                  # (Nm,3) obs-space anomalies
    delta = (Y_mean - h(psi_f_m[0]))[0]             # (3,) measured Jensen bias E[h(x)]-h(E[x])
    C = (Nm - 1) * Cdd + S.T @ S                    # (3,3)
    Cinv = linalg.inv(C)
    Aa = Af + (D - Y) @ (Cinv @ (S.T @ Af))         # factored update (as enkf.py)
    if not np.isreal(Aa).all():                     # numerical guard (as enkf.py)
        Aa = Af
    hbar = Y_mean[0]                                # (3,)
    xf_mean = psi_f_m[0]                            # (3,)
    s2 = Af.var(0, ddof=1)                          # (3,) forecast variance
    return Aa, hbar, xf_mean, s2, delta


def run_enkf_rls(obs, truth, obs_idx, h, xpf_teacher,
                 gamma=0.5, lam=1.0, P0=1e3, use_delta=False,
                 seed=cfg.seed, N=cfg.ensembleN, obs_std=None):
    """Run the RLS-corrected EnKF over an assimilation window.
        obs         : (n_cycles, 3) observations
        h           : observation operator (applied per member)
        xpf_teacher : (n_cycles, 3) PF posterior mean per cycle — fixed ground-truth target
        gamma       : injection damping (fixed, conservative); scalar in (0,1]
        lam, P0     : RLS forgetting factor and initial P scale
        use_delta   : pass standalone delta as a feature (multi-alpha training only)
    Returns dict of per-cycle arrays (corrected filter), including the learned
    prediction, the post-injection residual target, and the RLS residual.
    """
    rng = np.random.default_rng(seed)               # fresh rng per run
    n_obs = len(obs)
    obs_std = cfg.obs_std if obs_std is None else obs_std
    R = np.diag(obs_std ** 2)                        # observation-error covariance

    ensemble = np.tile(truth[0], (N, 1)) \
        + rng.normal(0, 1, (N, 3)) * cfg.init_std
    truth_at_obs = truth[obs_idx]

    # RLS state (shared definitions so shadow and this filter learn the identical model)
    # feature count: 6 blocks x 3 (+1 constant), +3 if use_delta. Probe once to size K.
    n_raw = 3 * (7 if use_delta else 6)             # raw feature columns before the constant
    K = n_features(n_raw)                           # incl. constant; int
    std = RunningStandardiser(n_raw)                 # running z-scorer
    rls = RLS(K, lam=lam, P0=P0)                     # matrix RLS

    en_mean = np.zeros((n_obs, 3))                   # CORRECTED analysis mean per cycle
    sqerror = np.zeros((n_obs, 3))                   # (corrected mean - truth)^2
    spread  = np.zeros((n_obs, 3))                   # analysis spread (unchanged by the shift)
    pred    = np.zeros((n_obs, 3))                   # learned correction w_{k-1}^T u_k
    target  = np.zeros((n_obs, 3))                   # r_k = pf_teacher - corrected mean
    rls_resid = np.zeros((n_obs, 3))               # r_k - w_{k-1}^T u_k (convergence metric)
    d_prev = np.zeros(3)                             # previous cycle's innovation (for the lag feature)

    for k in range(n_obs):
        ensemble += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ensemble = rk4_vec(ensemble, cfg.dt)     # propagate the (possibly corrected) ensemble

        # 1. analysis update
        Af = ensemble
        Aa, hbar, xf_mean, s2, delta = _analysis(Af, obs[k], R, h, rng)

        # 2. features from THIS cycle's forecast (single-row build via the shared builder)
        d_k = obs[k] - hbar                          # (3,) mean innovation this cycle
        U_raw_k = build_features(d_k[None], s2[None], xf_mean[None],
                                 delta[None], d_prev[None], use_delta=use_delta)[0]  # (k-1,)
        u = std.transform(U_raw_k)                   # standardised + constant; (K,)

        # 3. predict correction with OLD weights
        y_hat = rls.predict(u)                        # (3,)
        pred[k] = y_hat

        # 4. INJECT: shift every member by gamma * y_hat (mean moves, spread preserved)
        Aa = Aa + gamma * y_hat                       # broadcast (3,) across (N,3)
        ensemble = Aa                                 # this corrected ensemble propagates onward

        corr_mean = ensemble.mean(0)                  # (3,) corrected analysis mean

        # 5. target against the CORRECTED mean (the remaining gap to the PF)
        r_k = xpf_teacher[k] - corr_mean              # (3,)
        target[k] = r_k

        # 6. update weights on how the old-weight correction actually landed
        rls_resid[k] = rls.update(u, r_k)             # returns pre-update miss (3,)

        # diagnostics
        en_mean[k] = corr_mean
        dev = ensemble - corr_mean
        sqerror[k] = (truth_at_obs[k] - corr_mean) ** 2
        spread[k]  = np.mean(dev ** 2, axis=0)
        d_prev = d_k                                  # carry innovation forward for next lag

    return dict(en_mean=en_mean, sqerror=sqerror, spread=spread,
                pred=pred, target=target, rls_resid=rls_resid,
                truth_at_obs=truth_at_obs, w_final=rls.w)


if __name__ == '__main__':
    ALPHA = 0.5
    GAMMA = 0.5                                       # fixed conservative injection strength
    h = lambda x: x + ALPHA * x**2

    log = np.load(f'data/stage4_log_a{ALPHA}_{cfg.noise_mode}.npz')
    truth = None
    tw = np.load('data/l63_twin.npz')                 # truth + obs_idx live in the twin file
    truth = tw['truth']; obs_idx = tw['obs_idx']
    obs = log['obs']; xpf = log['xpf_mean']

    # noise std for this alpha (same as logging)
    alphas = tw['alphas']; ai = int(np.argmin(np.abs(alphas - ALPHA)))
    ostd = tw['obs_std_alpha'][ai]

    res = run_enkf_rls(obs, truth, obs_idx, h, xpf, gamma=GAMMA, obs_std=ostd)

    rmse_corr = np.sqrt(res['sqerror'].mean())
    rmse_pf = np.sqrt(((xpf - truth[obs_idx]) ** 2).mean())
    rmse_enkf = np.sqrt(((log['xa_mean'] - truth[obs_idx]) ** 2).mean())
    print(f"alpha={ALPHA} gamma={GAMMA} | RMSE enkf={rmse_enkf:.3f} "
          f"corrected={rmse_corr:.3f} pf={rmse_pf:.3f}")

    out = f'data/stage4_inject_a{ALPHA}.npz'
    np.savez(out, **res, xpf_mean=xpf, truth=truth[obs_idx], alpha=ALPHA,
             times=log['times'], rmse_enkf=rmse_enkf, rmse_corr=rmse_corr, rmse_pf=rmse_pf)
    print(f'saved {out}')
