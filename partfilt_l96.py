import numpy as np
from config_l96 import cfg, rk4_vec

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.2  Removed: mean ESS from the reported output - it reads ~N after a collapse, because
#               identical resampled particles carry identical weights
#      Added:   weighted posterior spread and the spread/RMSE ratio, which do detect collapse
# 5.1  Added:   roughening jitter before propagation (see config_l96 perturb_std)
# 5.0  Stage 5 - Lorenz-96 PF feasibility check
#   created
# ============================================================


def run_pf_l96(obs, truth, obs_idx, obs_sites, N, seed=cfg.seed):

    rng = np.random.default_rng(seed)
    n = truth.shape[1]
    R = cfg.obs_std ** 2
    n_obs = len(obs)

    particles = truth[0] + rng.normal(0, cfg.init_std, (N, n))
    weights = np.ones(N) / N

    en_mean = np.zeros((n_obs, n))
    sqerror = np.zeros((n_obs, n))
    spread = np.zeros((n_obs, n))
    ess = np.zeros(n_obs)
    truth_at_obs = truth[obs_idx]

    for k in range(n_obs):
        particles += rng.normal(0, cfg.perturb_std, (N, n))
        for _ in range(cfg.obs_every):
            particles = rk4_vec(particles, cfg.dt)

        Yp = particles[:, obs_sites]
        Yp = Yp + cfg.alpha * Yp ** 2
        innov = obs[k] - Yp

        # work in logs: in high dimension the raw likelihood underflows to exactly 0 for every particle
        loglik = -0.5 * np.sum(innov ** 2, axis=1) / R
        loglik -= loglik.max()                  # shift so the best particle has weight 1
        weights *= np.exp(loglik)
        weights /= weights.sum()

        ess[k] = 1.0 / np.sum(weights ** 2)     # N if weights are even, 1 if one particle holds all of it
        en_mean[k] = weights @ particles
        sqerror[k] = (truth_at_obs[k] - en_mean[k]) ** 2
        spread[k] = weights @ (particles - en_mean[k]) ** 2   # weighted posterior variance

        positions = (np.arange(N) + rng.random()) / N
        indices = np.searchsorted(np.cumsum(weights), positions)
        particles = particles[indices]
        weights = np.ones(N) / N

    return dict(en_mean=en_mean, sqerror=sqerror, spread=spread, ess=ess)


if __name__ == '__main__':
    data = np.load('data/l96_twin.npz')
    obs_idx = data['obs_idx']

    ess_min = np.zeros((len(cfg.n_list), len(cfg.N_list)))
    rmse = np.zeros_like(ess_min)
    spread = np.zeros_like(ess_min)

    for i, n in enumerate(cfg.n_list):
        truth = data[f'truth_{n}']
        obs = data[f'obs_{n}']
        obs_sites = data[f'obs_sites_{n}']
        clim = truth.std()                       # RMSE of a filter that just guesses the mean
        for j, N in enumerate(cfg.N_list):
            res = run_pf_l96(obs, truth, obs_idx, obs_sites, N)
            ess_min[i, j] = res['ess'].min()
            rmse[i, j] = np.sqrt(res['sqerror'].mean())
            spread[i, j] = np.sqrt(res['spread'].mean())
            print(f"n={n:3d}  N={N:6d}  min ESS={ess_min[i,j]:7.1f}  "
                  f"spread={spread[i,j]:.3f}  RMSE={rmse[i,j]:.3f}  "
                  f"spread/RMSE={spread[i,j]/rmse[i,j]:.2f}  clim={clim:.2f}")

    np.savez('data/l96_pf_ess.npz', ess_min=ess_min, rmse=rmse, spread=spread,
             n_list=cfg.n_list, N_list=cfg.N_list)