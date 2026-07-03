import numpy as np
from config import cfg, rk4_vec

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.0  Stage 2 — nonlinear-h support
#   Added:   obs_std arg + local R_diag, so each run uses its own per-alpha noise
#   Added:   forecast cross-covariance Cov(x, h(x)) returned as 'cross'
#            (equally-weighted prior, before the weight update; PF reference for stage2)
# ============================================================


def run_pf(obs, truth, obs_idx, h, seed=cfg.seed, N=cfg.ensembleN,
           obs_std=None, save_particles=False):
    """Bootstrap particle filter with systematic resampling.
        obs:     (n_obs, 3) observations
        h:       observation operator, applied to particles
        obs_std: per-component observation noise std (defaults to cfg.obs_std).
    Returns dict of per-step arrays.
    """
    rng = np.random.default_rng(seed)          # fresh rng per run -> reproducible
    obs_std = cfg.obs_std if obs_std is None else obs_std
    R_diag = obs_std ** 2
    n_obs = len(obs)

    particles = np.tile(truth[0], (N, 1)) \
        + rng.normal(0, 1, (N, 3)) * cfg.init_std
    weights = np.ones(N) / N

    en_mean = np.zeros((n_obs, 3))
    sqerror = np.zeros((n_obs, 3))
    spread  = np.zeros((n_obs, 3))
    cross   = np.zeros((n_obs, 3, 3))          # forecast cross-cov Cov(x, h(x)), the reference
    truth_at_obs = truth[obs_idx]
    particles_history = np.zeros((n_obs, N, 3)) if save_particles else None

    for k in range(n_obs):
        # perturb then propagate the whole ensemble at once
        particles += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            particles = rk4_vec(particles, cfg.dt)

        # forecast cross-cov on the equally-weighted prior particles (before the weight update)
        Yp = h(particles)
        Xa = particles - particles.mean(0)
        Sa = Yp - Yp.mean(0)
        cross[k] = (Xa.T @ Sa) / (N - 1)

        # weight update: innovation is in observation space, so apply h
        innovation = obs[k] - Yp                            # (N, 3)
        likelihood = np.exp(-0.5 * np.sum(innovation**2 / R_diag, axis=1))
        weights *= likelihood
        weights += 1e-300                                   # guard against all-zero
        weights /= np.sum(weights)

        # diagnostics use pre-resampling weighted particles
        en_mean[k] = np.average(particles, weights=weights, axis=0)
        deviation = particles - en_mean[k]
        sqerror[k] = (truth_at_obs[k] - en_mean[k]) ** 2
        spread[k]  = np.average(deviation**2, weights=weights, axis=0)
        if save_particles:
            particles_history[k] = particles.copy()

        # systematic resampling (seeded rng, was np.random.rand before)
        positions = (np.arange(N) + rng.random()) / N
        cumsum = np.cumsum(weights)
        indices = np.searchsorted(cumsum, positions)
        particles = particles[indices]
        weights = np.ones(N) / N

    return dict(en_mean=en_mean, sqerror=sqerror, spread=spread, cross=cross,
                truth_at_obs=truth_at_obs, particles_history=particles_history)


if __name__ == '__main__':
    data = np.load('data/l63_twin.npz')
    truth = data['truth']
    obs_idx = data['obs_idx']
    obs_linear = data['obs_linear']
    obs_nonlinear = data['obs_nonlinear']
    obs_std_alpha = data['obs_std_alpha']
    alphas = data['alphas']

    # --- linear run (h(x) = x) ---
    h_lin = lambda x: x
    lin = run_pf(obs_linear, truth, obs_idx, h_lin, save_particles=True)

    # --- nonlinear sweep ---
    en_mean_nl = np.zeros((len(alphas), len(obs_idx), 3))
    sqerror_nl = np.zeros_like(en_mean_nl)
    spread_nl  = np.zeros_like(en_mean_nl)
    for ai, a in enumerate(alphas):
        h_nl = lambda x, a=a: x + a * x**2
        res = run_pf(obs_nonlinear[ai], truth, obs_idx, h_nl, obs_std=obs_std_alpha[ai])
        en_mean_nl[ai] = res['en_mean']
        sqerror_nl[ai] = res['sqerror']
        spread_nl[ai]  = res['spread']

    np.savez('data/pf_results.npz',
             en_mean_linear=lin['en_mean'],
             sqerror_linear=lin['sqerror'],
             spread_linear=lin['spread'],
             particles_linear=lin['particles_history'],
             truth_at_obs=lin['truth_at_obs'],
             en_mean_nonlinear=en_mean_nl,
             sqerror_nonlinear=sqerror_nl,
             spread_nonlinear=spread_nl,
             alphas=alphas)

    rmse = np.sqrt(np.mean(lin['sqerror'], axis=0))
    print(f"PF linear RMSE  x={rmse[0]:.3f} y={rmse[1]:.3f} z={rmse[2]:.3f}")
