import numpy as np
from scipy import linalg
from config import cfg, rk4_vec


def EnKF(Af, d, Cdd, h, rng):
    """Stochastic EnKF (Evensen), row-ensemble convention. h applied per member.
        Af:  (Nm, n_state) forecast ensemble, rows are members
        d:   (n_obs,) observation vector
        Cdd: (n_obs, n_obs) observation error covariance
    Returns (analysis ensemble, jensen bias vector).
    """
    Nm = Af.shape[0]

    psi_f_m = np.mean(Af, 0, keepdims=True)         # (1, n_state) = E[x]
    D = rng.multivariate_normal(d, Cdd, Nm)         # (Nm, n_obs) perturbed obs

    Y = h(Af)                                       # (Nm, n_obs) = h(x_i), h elementwise
    Y_e_mean = h(psi_f_m[0])                         # h(E[x])
    Y_mean = np.mean(Y, axis=0, keepdims=True)       # E[h(x)]
    S = Y - Y_mean                                   # obs-space anomalies
    jensen = (Y_mean - Y_e_mean)[0]                  # E[h(x)] - h(E[x])

    C = (Nm - 1) * Cdd + S.T @ S                     # (n_obs, n_obs)
    Cinv = linalg.inv(C)

    X = S @ (Cinv @ (D - Y).T)                        # (Nm, Nm)
    Aa = Af + X.T @ Af                                # (Nm, n_state)

    if not np.isreal(Aa).all():
        print('Aa not real; returning forecast')
        return Af, jensen
    return Aa, jensen


def run_enkf(obs, truth, obs_idx, h, seed=cfg.seed, N=cfg.ensembleN):
    """Run EnKF over an assimilation window. Returns dict of per-step arrays."""
    rng = np.random.default_rng(seed)              # fresh rng per run
    n_obs = len(obs)

    ensemble = np.tile(truth[0], (N, 1)) \
        + rng.normal(0, 1, (N, 3)) * cfg.init_std

    en_mean = np.zeros((n_obs, 3))
    sqerror = np.zeros((n_obs, 3))
    spread  = np.zeros((n_obs, 3))
    jensen  = np.zeros((n_obs, 3))
    truth_at_obs = truth[obs_idx]

    for k in range(n_obs):
        ensemble += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ensemble = rk4_vec(ensemble, cfg.dt)

        ensemble, js = EnKF(ensemble, obs[k], cfg.R, h, rng)

        jensen[k] = js
        en_mean[k] = np.mean(ensemble, axis=0)
        deviation = ensemble - en_mean[k]
        sqerror[k] = (truth_at_obs[k] - en_mean[k]) ** 2
        spread[k]  = np.mean(deviation**2, axis=0)

    return dict(en_mean=en_mean, sqerror=sqerror, spread=spread,
                jensen=jensen, truth_at_obs=truth_at_obs)


if __name__ == '__main__':
    data = np.load('data/l63_twin.npz')
    truth = data['truth']
    obs_idx = data['obs_idx']
    obs_linear = data['obs_linear']
    obs_nonlinear = data['obs_nonlinear']
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
        res = run_enkf(obs_nonlinear[ai], truth, obs_idx, h_nl)
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
