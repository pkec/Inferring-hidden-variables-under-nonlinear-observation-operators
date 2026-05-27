import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from config import cfg, rk4

rng = np.random.default_rng(cfg.seed)


def EnKF(Af, d, Cdd, M):
    """Ensemble Kalman Filter (Evensen 2009 eq. 9.27), row-ensemble convention.
        Af:  (Nm, n_state) forecast ensemble, rows are members
        d:   (n_obs,) observation vector
        Cdd: (n_obs, n_obs) observation error covariance
        M:   (n_obs, n_state) observation operator
    """
    Nm = np.size(Af, 0)

    psi_f_m = np.mean(Af, 0, keepdims=True)    # (1, n_state)
    Psi_f = Af - psi_f_m                        # (Nm, n_state)

    D = rng.multivariate_normal(d, Cdd, Nm)     # (Nm, n_obs)

    Y = np.dot(Af, M.T)                         # (Nm, n_obs)
    S = np.dot(Psi_f, M.T)                      # (Nm, n_obs)

    C = (Nm - 1) * Cdd + np.dot(S.T, S)        # (n_obs, n_obs)
    Cinv = linalg.inv(C)

    X = np.dot(S, np.dot(Cinv, (D - Y).T))     # (Nm, Nm)
    Aa = Af + np.dot(X.T, Af)                  # (Nm, n_state)

    if np.isreal(Aa).all():
        return Aa
    else:
        print('Aa not real')
        return Af


if __name__ == '__main__':
    data = np.load('data/l63_twin.npz')
    obs = data['obs_linear']
    truth = data['truth']
    obs_idx = data['obs_idx']

    M = np.eye(3)

    ensemble = np.tile(truth[0], (cfg.ensembleN, 1)) \
        + rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.init_std

    en_mean = np.zeros((len(obs), 3))
    sqerror = np.zeros((len(obs), 3))
    spread = np.zeros((len(obs), 3))
    truth_at_obs = np.zeros((len(obs), 3))

    for k in range(len(obs)):
        for i in range(cfg.ensembleN):
            for j in range(cfg.obs_every):
                ensemble[i] = rk4(ensemble[i], cfg.dt)
            ensemble[i] += rng.normal(0, 1, 3) * cfg.perturb_std

        ensemble = EnKF(ensemble, obs[k], cfg.R, M)

        truth_at_obs[k] = truth[obs_idx[k]]
        en_mean[k] = np.mean(ensemble, axis=0)
        deviation = ensemble - en_mean[k]
        sqerror[k] = (truth_at_obs[k] - en_mean[k]) ** 2
        spread[k] = np.mean(deviation ** 2, axis=0)

    per_axis_rmse = np.sqrt(np.mean(sqerror, axis=0))
    per_axis_spread = np.sqrt(np.mean(spread, axis=0))
    rmse_spread_ratio = per_axis_rmse / per_axis_spread

    print(f"{'':>8}{'x':>10}{'y':>10}{'z':>10}")
    print(f"{'RMSE':>8}{per_axis_rmse[0]:10.4f}{per_axis_rmse[1]:10.4f}{per_axis_rmse[2]:10.4f}")
    print(f"{'Spread':>8}{per_axis_spread[0]:10.4f}{per_axis_spread[1]:10.4f}{per_axis_spread[2]:10.4f}")
    print(f"{'Ratio':>8}{rmse_spread_ratio[0]:10.4f}{rmse_spread_ratio[1]:10.4f}{rmse_spread_ratio[2]:10.4f}")

    time_obs = obs_idx * float(cfg.dt)
    time_truth = np.arange(len(truth)) * float(cfg.dt)
    labels = ['x', 'y', 'z']

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for i in range(3):
        axes[i].plot(time_truth, truth[:, i], 'k-', lw=0.8, label='Truth')
        axes[i].plot(time_obs, obs[:, i], 'r.', ms=3, alpha=0.4, label='Obs')
        axes[i].plot(time_obs, en_mean[:, i], 'b.', lw=0.5, label='EnKF mean')
        axes[i].set_ylabel(labels[i])
        if i == 0:
            axes[i].legend(loc='upper right', fontsize=9)

    axes[-1].set_xlabel('Time')
    fig.suptitle(f'EnKF (N={cfg.ensembleN})')
    plt.tight_layout()
    plt.show()
