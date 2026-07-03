import numpy as np
import os
from config import cfg, rk4

os.makedirs('data', exist_ok=True)
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

# observation times and a single unit-normal noise draw (reused for linear + every alpha)
obs_idx = np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every)
obs_noise = rng.normal(0, 1, (len(obs_idx), 3))    # unit normal; scaled per alpha below

# linear obs: y = x + noise (uses the baseline obs_std directly)
obs_linear = truth[obs_idx] + obs_noise * cfg.obs_std

# ratio that reproduces obs_std at alpha=0; only used in fixed_snr mode
snr = cfg.obs_std / truth[obs_idx].std(axis=0)

# nonlinear obs for each alpha. fixed_R: constant sensor std. fixed_snr: std scales with std(h_alpha) to hold SNR.
obs_nonlinear = np.zeros((len(cfg.alpha), len(obs_idx), 3))
obs_std_alpha = np.zeros((len(cfg.alpha), 3))      # the actual noise std used at each alpha
for ai, a in enumerate(cfg.alpha):
    h_truth = truth[obs_idx] + a * truth[obs_idx]**2
    std_a = cfg.obs_std if cfg.noise_mode == 'fixed_R' else snr * h_truth.std(axis=0)
    obs_std_alpha[ai] = std_a
    obs_nonlinear[ai] = h_truth + obs_noise * std_a

np.savez('data/l63_twin.npz',
         truth=truth,
         obs_linear=obs_linear,
         obs_nonlinear=obs_nonlinear,
         obs_std_alpha=obs_std_alpha,
         alphas=cfg.alpha,
         obs_idx=obs_idx,
         dt=cfg.dt)

print(f"truth_obs: saved {len(obs_idx)} obs, {len(cfg.alpha)} alphas (noise_mode={cfg.noise_mode})")
