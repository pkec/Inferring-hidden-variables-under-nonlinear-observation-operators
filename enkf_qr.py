import numpy as np
from config import cfg, rk4_vec

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 4.41 Added: divergence guard — a non-finite or off-attractor ensemble stops the run and
#      is reported as diverged_at, with the tail NaN-filled, instead of letting the next
#      window's RK4 overflow and crash the analysis. At N=100 the QR variant hit this at
#      alpha=0.1 under high jitter; same guard everywhere so the three variants agree.
# 3.1  Changed: restructured to mirror enkf.py line-for-line; only the analysis update differs
#               (quadratic least-squares regression of state on [obs, obs^2] instead of the linear
#               factored gain). run_enkf_qr now matches run_enkf (save_forecast/fc_spread/fc_history)
# 3.0  created — quadratic-regression EnKF
# ============================================================


def EnKF_qr(Af, d, Cdd, h, rng):

    Nm = Af.shape[0]

    psi_f_m = np.mean(Af, 0, keepdims=True)         # (1, n_state) = E[x]
    D = rng.multivariate_normal(d, Cdd, Nm)         # (Nm, n_obs) perturbed obs

    Y = h(Af)                                       # (Nm, n_obs) = h(x_i), h elementwise
    Y_e_mean = h(psi_f_m[0])                         # h(E[x])
    Y_mean = np.mean(Y, axis=0, keepdims=True)       # E[h(x)]
    S = Y - Y_mean                                   # obs-space anomalies
    jensen = (Y_mean - Y_e_mean)[0]                  # E[h(x)] - h(E[x])

    Xa = Af - psi_f_m                                 # state-space anomalies
    Cxh = (Xa.T @ S) / (Nm - 1)                       # forecast cross-cov Cov(x, h(x))

    # --- quadratic regression update (the only change from EnKF) ---
    P = Y + (D - d)                                  # noisy predictions; the obs noise puts R into the fit
    mu, sd = P.mean(0), P.std(0)                     # standardize so z^2 (~2000) can't swamp x^2 (~400)
    Ptil = (P - mu) / sd                             # standardized predictions, one row per member
    dtil = (d - mu) / sd                             # the real obs, standardized the same way

    Phi   = np.column_stack([np.ones(Nm), Ptil, Ptil ** 2])   # design matrix [1, y, y^2] per member
    phi_d = np.concatenate([[1.0], dtil, dtil ** 2])          # same features evaluated at the real obs
    B, *_ = np.linalg.lstsq(Phi, Af, rcond=None)     # fit state on the quadratic features -> (n_feat, n_state)
    Aa = Af + (phi_d @ B) - (Phi @ B)                # shift each member by g(d) - g(its own prediction)

    if not np.isreal(Aa).all():
        print('Aa not real; returning forecast')
        return Af, jensen, Cxh
    return Aa, jensen, Cxh


def run_enkf_qr(obs, truth, obs_idx, h, seed=cfg.seed, N=cfg.ensembleN,
                obs_std=None, save_forecast=True):

    rng = np.random.default_rng(seed)              # fresh rng per run
    n_obs = len(obs)
    obs_std = cfg.obs_std if obs_std is None else obs_std
    R = np.diag(obs_std ** 2)                       # observation-error covariance for this run

    ensemble = np.tile(truth[0], (N, 1)) \
        + rng.normal(0, 1, (N, 3)) * cfg.init_std

    en_mean = np.zeros((n_obs, 3))
    sqerror = np.zeros((n_obs, 3))
    spread  = np.zeros((n_obs, 3))
    jensen  = np.zeros((n_obs, 3))
    cross   = np.zeros((n_obs, 3, 3))              # forecast cross-cov Cov(x, h(x)) per step
    fc_spread = np.zeros((n_obs, 3))               # prior spread, before the update
    fc_history = np.zeros((n_obs, N, 3)) if save_forecast else None
    truth_at_obs = truth[obs_idx]
    diverged_at = None                             # cycle where the run left the attractor, or None
    BLOWUP = 1e3                                   # |state| far beyond the L63 attractor (~|x|<50)

    for k in range(n_obs):
        ensemble += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        # overflow here is expected once the state has left the attractor; the guard below
        # catches it, so the warnings are noise rather than information
        with np.errstate(over='ignore', invalid='ignore'):
            for _ in range(cfg.obs_every):
                ensemble = rk4_vec(ensemble, cfg.dt)

        # Divergence guard, AFTER propagation and BEFORE the analysis: this is where a
        # blown-up state actually becomes inf/NaN. The Lorenz terms are quadratic (x*y, x*z),
        # so an ensemble that is merely large at the end of one cycle overflows inside the
        # next window's RK4 steps, and the analysis then fails on a non-finite matrix.
        if not np.isfinite(ensemble).all() or np.abs(ensemble).max() > BLOWUP:
            diverged_at = k
            break

        fc_spread[k] = ensemble.std(axis=0)        # measured on the prior, before assimilation
        if save_forecast:
            fc_history[k] = ensemble.copy()

        ensemble, js, cxh = EnKF_qr(ensemble, obs[k], R, h, rng)

        jensen[k] = js
        cross[k] = cxh
        en_mean[k] = np.mean(ensemble, axis=0)
        deviation = ensemble - en_mean[k]
        sqerror[k] = (truth_at_obs[k] - en_mean[k]) ** 2
        spread[k]  = np.mean(deviation**2, axis=0)

    if diverged_at is not None:                    # tail was never written; mark it unusable
        for arr in (en_mean, sqerror, spread, jensen, fc_spread):
            arr[diverged_at:] = np.nan
        cross[diverged_at:] = np.nan
        if save_forecast:
            fc_history[diverged_at:] = np.nan

    return dict(en_mean=en_mean, sqerror=sqerror, spread=spread,
                jensen=jensen, cross=cross, truth_at_obs=truth_at_obs,
                fc_spread=fc_spread, fc_history=fc_history,
                diverged_at=diverged_at)


if __name__ == '__main__':
    data = np.load('data/l63_twin.npz')
    truth = data['truth']
    obs_idx = data['obs_idx']
    obs_linear = data['obs_linear']
    obs_nonlinear = data['obs_nonlinear']
    obs_std_alpha = data['obs_std_alpha']
    alphas = data['alphas']

    # --- linear run (h(x) = x; should match the linear EnKF) ---
    h_lin = lambda x: x
    lin = run_enkf_qr(obs_linear, truth, obs_idx, h_lin)

    # --- nonlinear sweep ---
    en_mean_nl = np.zeros((len(alphas), len(obs_idx), 3))
    sqerror_nl = np.zeros_like(en_mean_nl)
    spread_nl  = np.zeros_like(en_mean_nl)
    jensen_nl  = np.zeros_like(en_mean_nl)
    for ai, a in enumerate(alphas):
        h_nl = lambda x, a=a: x + a * x**2
        res = run_enkf_qr(obs_nonlinear[ai], truth, obs_idx, h_nl, obs_std=obs_std_alpha[ai])
        en_mean_nl[ai] = res['en_mean']
        sqerror_nl[ai] = res['sqerror']
        spread_nl[ai]  = res['spread']
        jensen_nl[ai]  = res['jensen']

    np.savez('data/qr_results.npz',
             en_mean_linear=lin['en_mean'],
             sqerror_linear=lin['sqerror'],
             spread_linear=lin['spread'],
             jensen_linear=lin['jensen'],
             truth_at_obs=lin['truth_at_obs'],
             en_mean_nonlinear=en_mean_nl,
             sqerror_nonlinear=sqerror_nl,
             spread_nonlinear=spread_nl,
             jensen_nonlinear=jensen_nl,
             alphas=alphas)

    rmse = np.sqrt(np.mean(lin['sqerror'], axis=0))
    print(f"QR-EnKF linear RMSE  x={rmse[0]:.3f} y={rmse[1]:.3f} z={rmse[2]:.3f}")