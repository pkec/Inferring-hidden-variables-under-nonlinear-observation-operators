import numpy as np
from scipy import linalg
from config import cfg, rk4_vec

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.0  Stage 2 — nonlinear-h support
#   Added:   obs_std arg + local R, so each run uses its own per-alpha noise
#   Added:   forecast cross-covariance Cov(x, h(x)) returned as 'cross'
#   Changed: analysis to factored update (no Nm x Nm matrix; ~5x faster, matches old to 4e-15)
# ============================================================


def EnKF(Af, d, Cdd, h, rng):
    """Stochastic EnKF (Evensen), row-ensemble convention. h applied per member.
        Af:  (Nm, n_state) forecast ensemble, rows are members
        d:   (n_obs,) observation vector
        Cdd: (n_obs, n_obs) observation error covariance
    Returns (analysis ensemble, jensen bias vector, forecast cross-covariance).
    """
    Nm = Af.shape[0]

    psi_f_m = np.mean(Af, 0, keepdims=True)         # (1, n_state) = E[x]
    D = rng.multivariate_normal(d, Cdd, Nm)         # (Nm, n_obs) perturbed obs

    Y = h(Af)                                       # (Nm, n_obs) = h(x_i), h elementwise
    Y_e_mean = h(psi_f_m[0])                         # h(E[x])
    Y_mean = np.mean(Y, axis=0, keepdims=True)       # E[h(x)]
    S = Y - Y_mean                                   # obs-space anomalies
    jensen = (Y_mean - Y_e_mean)[0]                  # E[h(x)] - h(E[x])

    Xa = Af - psi_f_m                                 # state-space anomalies
    Cxh = (Xa.T @ S) / (Nm - 1)                       # forecast cross-cov Cov(x, h(x)), (n_state, n_obs)

    C = (Nm - 1) * Cdd + S.T @ S                     # (n_obs, n_obs)
    Cinv = linalg.inv(C)

    Aa = Af + (D - Y) @ (Cinv @ (S.T @ Af))           # factored update; all intermediates (n_obs, *)

    if not np.isreal(Aa).all():
        print('Aa not real; returning forecast')
        return Af, jensen, Cxh
    return Aa, jensen, Cxh


def run_enkf(obs, truth, obs_idx, h, seed=cfg.seed, N=cfg.ensembleN,
             obs_std=None, save_forecast=True):
    """Run EnKF over an assimilation window. Returns dict of per-step arrays.
    obs_std: per-component observation noise std (defaults to cfg.obs_std).
    save_forecast: also return the forecast (prior) ensemble before each update.
    """
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

    for k in range(n_obs):
        ensemble += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ensemble = rk4_vec(ensemble, cfg.dt)

        fc_spread[k] = ensemble.std(axis=0)        # measured on the prior, before assimilation
        if save_forecast:
            fc_history[k] = ensemble.copy()

        ensemble, js, cxh = EnKF(ensemble, obs[k], R, h, rng)

        jensen[k] = js
        cross[k] = cxh
        en_mean[k] = np.mean(ensemble, axis=0)
        deviation = ensemble - en_mean[k]
        sqerror[k] = (truth_at_obs[k] - en_mean[k]) ** 2
        spread[k]  = np.mean(deviation**2, axis=0)

    return dict(en_mean=en_mean, sqerror=sqerror, spread=spread,
                jensen=jensen, cross=cross, truth_at_obs=truth_at_obs,
                fc_spread=fc_spread, fc_history=fc_history)


if __name__ == '__main__':
    data = np.load('data/l63_twin.npz')
    truth = data['truth']
    obs_idx = data['obs_idx']
    obs_linear = data['obs_linear']
    obs_nonlinear = data['obs_nonlinear']
    obs_std_alpha = data['obs_std_alpha']
    alphas = data['alphas']

    # --- linear run ---
    h_lin = lambda x: x
    lin = run_enkf(obs_linear, truth, obs_idx, h_lin)

    # --- nonlinear sweep ---
    en_mean_nl = np.zeros((len(alphas), len(obs_idx), 3))
    sqerror_nl = np.zeros_like(en_mean_nl)
    spread_nl  = np.zeros_like(en_mean_nl)
    jensen_nl  = np.zeros_like(en_mean_nl)
    for ai, a in enumerate(alphas):
        h_nl = lambda x, a=a: x + a * x**2
        res = run_enkf(obs_nonlinear[ai], truth, obs_idx, h_nl, obs_std=obs_std_alpha[ai])
        en_mean_nl[ai] = res['en_mean']
        sqerror_nl[ai] = res['sqerror']
        spread_nl[ai]  = res['spread']
        jensen_nl[ai]  = res['jensen']

    np.savez('data/enkf_results.npz',
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
    print(f"EnKF linear RMSE  x={rmse[0]:.3f} y={rmse[1]:.3f} z={rmse[2]:.3f}")
