import numpy as np
from config import cfg, rk4_vec

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.4  created — iterative EnKF (Gauss-Newton on the MAP cost). Mirrors enkf.py; the single
#      analysis update is replaced by a Gauss-Newton loop that re-linearises h at the current
#      iterate. Stochastic (perturbed-obs) anomaly update, as in enkf.py, not Sakov's square-root.
# ============================================================

GN_MAXITER = 20      # hard cap on Gauss-Newton iterations (Sakov/Bocquet use pmax=40)
GN_TOL     = 1e-3    # stop when ||w_new - w|| < GN_TOL  (Bocquet & Sakov use e2 = 1e-3)
GN_EPS     = 1e-3    # bundle probe: anomalies shrunk by this before regressing the slope


def EnKF_ienkf(Af, d, Cdd, h, rng):
    """Iterative EnKF: Gauss-Newton minimisation of J(x) = ||x-x_f||^2_Pf^-1 + ||d-h(x)||^2_R^-1.
        Af:  (Nm, n_state) forecast ensemble, rows are members
        d:   (n_obs,) observation vector
        Cdd: (n_obs, n_obs) observation error covariance
    Each iteration re-estimates the tangent-linear H by regressing h(E) on E over a shrunken
    probe cloud centred at the current iterate, then takes an exact linear-Kalman step on the
    linearised cost. Iteration 0 (w = x_f) reproduces the standard EnKF update of the mean.
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

    Xa = Af - psi_f_m                                 # state-space anomalies (the prior spread)
    Cxh = (Xa.T @ S) / (Nm - 1)                       # forecast cross-cov Cov(x, h(x))

    # --- Gauss-Newton loop on the mean (the only change from EnKF) ---
    x_f = psi_f_m[0]                                  # background mean; FIXED for every iteration
    Pf = (Xa.T @ Xa) / (Nm - 1)                       # background covariance; FIXED
    w = x_f.copy()                                    # current iterate of the analysis mean
    H = None
    for _ in range(GN_MAXITER):
        E = w + GN_EPS * Xa                           # probe cloud: prior anomalies, shrunk(Sakov's bundle), re-centred at w
        Sb = h(E) - h(E).mean(0)                      # obs anomalies of the probe
        Eb = E - E.mean(0)                            # state anomalies of the probe (= GN_EPS * Xa)
        H = np.linalg.lstsq(Eb, Sb, rcond=None)[0].T  # (n_obs, n_state) tangent-linear by regression

        HPf = H @ Pf
        K = HPf.T @ np.linalg.inv(HPf @ H.T + Cdd)    # (n_state, n_obs) gain from the FIXED prior
        d_tilde = d - h(w) + H @ (w - x_f)            # effective innovation, referred back to x_f
        w_new = x_f + K @ d_tilde                     # exact minimiser of the linearised cost
        step = np.linalg.norm(w_new - w)
        w = w_new
        if step < GN_TOL:
            break

    # --- anomaly update: stochastic, reusing the perturbed obs (as enkf.py does) ---
    eps = D - d                                       # per-member obs-noise draws
    eps = eps - eps.mean(0)                           # centre them so the mean is untouched
    Aa_anom = Xa - Xa @ (K @ H).T + eps @ K.T            # Cov -> (I-KH)Pf(I-KH)^T + K R K^T
    Aa = w + Aa_anom                                  # analysis = converged mean + updated anomalies

    if not np.isreal(Aa).all():
        print('Aa not real; returning forecast')
        return Af, jensen, Cxh
    return Aa, jensen, Cxh


def run_enkf_ienkf(obs, truth, obs_idx, h, seed=cfg.seed, N=cfg.ensembleN,
                   obs_std=None, save_forecast=True):
    """Run the iterative EnKF over an assimilation window. Returns dict of per-step arrays.
    Same loop as run_enkf; only the analysis step (EnKF_ienkf) differs.
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

        ensemble, js, cxh = EnKF_ienkf(ensemble, obs[k], R, h, rng)

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

    # --- linear run (h(x) = x; should match the linear EnKF) ---
    h_lin = lambda x: x
    lin = run_enkf_ienkf(obs_linear, truth, obs_idx, h_lin)

    # --- nonlinear sweep ---
    for ai, a in enumerate(alphas):
        h_nl = lambda x, a=a: x + a * x**2
        res = run_enkf_ienkf(obs_nonlinear[ai], truth, obs_idx, h_nl, obs_std=obs_std_alpha[ai])
        rmse = np.sqrt(res['sqerror'].mean(0))
        print(f"alpha {a:.1f}  IEnKF RMSE  x={rmse[0]:.3f} y={rmse[1]:.3f} z={rmse[2]:.3f}")

    rmse = np.sqrt(np.mean(lin['sqerror'], axis=0))
    print(f"IEnKF linear RMSE  x={rmse[0]:.3f} y={rmse[1]:.3f} z={rmse[2]:.3f}")
