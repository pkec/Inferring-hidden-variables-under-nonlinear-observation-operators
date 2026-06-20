import numpy as np
from config import cfg, rk4

rng = np.random.default_rng(cfg.seed)

# spin up onto attractor
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):
    s = rk4(s, cfg.dt)

# truth trajectory
truth = np.zeros((cfg.n_steps + 1, 3))
truth[0] = s
for k in range(cfg.n_steps):
    truth[k+1] = rk4(truth[k], cfg.dt)

# observation times and a single noise realization (reused for linear + every alpha)
obs_idx = np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every)
obs_noise = rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std

# linear obs: y = x + noise
obs_linear = truth[obs_idx] + obs_noise

# nonlinear obs for each alpha: y = x + a*x^2 + noise (same noise across alphas)
obs_nonlinear = np.stack([
    truth[obs_idx] + a * truth[obs_idx]**2 + obs_noise
    for a in cfg.alpha
])  # (n_alphas, n_obs, 3)

np.savez('data/l63_twin.npz',
         truth=truth,
         obs_linear=obs_linear,
         obs_nonlinear=obs_nonlinear,
         alphas=cfg.alpha,
         obs_idx=obs_idx,
         dt=cfg.dt)

print(f"truth_obs: saved {len(obs_idx)} obs, {len(cfg.alpha)} alphas")
