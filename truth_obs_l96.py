import numpy as np
import os
from config_l96 import cfg, rk4_vec

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.0  Stage 5 - Lorenz-96 PF feasibility check
#   created
# ============================================================

os.makedirs('data', exist_ok=True)
rng = np.random.default_rng(cfg.seed)

out = {}
for n in cfg.n_list:
    # start at the unstable fixed point x_i = F and nudge one component off it
    x = np.full(n, cfg.F)
    x[0] += 0.01
    for _ in range(cfg.spinup):
        x = rk4_vec(x, cfg.dt)

    truth = np.zeros((cfg.n_steps + 1, n))
    truth[0] = x
    for k in range(cfg.n_steps):
        truth[k + 1] = rk4_vec(truth[k], cfg.dt)

    obs_idx = np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every)
    obs_sites = np.arange(0, n, cfg.obs_stride)

    h_truth = truth[np.ix_(obs_idx, obs_sites)]        # (n_obs, p) truth at observed sites
    h_truth = h_truth + cfg.alpha * h_truth ** 2
    obs = h_truth + rng.normal(0, cfg.obs_std, h_truth.shape)

    out[f'truth_{n}'] = truth
    out[f'obs_{n}'] = obs
    out[f'obs_sites_{n}'] = obs_sites
    print(f"n={n:3d}  p={len(obs_sites):3d}  truth std={truth.std():.2f}")

np.savez('data/l96_twin.npz', obs_idx=obs_idx, n_list=cfg.n_list,
         alpha=cfg.alpha, obs_std=cfg.obs_std, dt=cfg.dt, **out)

print(f"truth_obs_l96: saved {len(obs_idx)} obs cycles for n={cfg.n_list}")
