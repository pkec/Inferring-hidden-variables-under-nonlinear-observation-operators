import numpy as np
import matplotlib.pyplot as plt
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

# observations with per-axis noise
obs_idx = np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every)
obs = truth[obs_idx] + rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std

# plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(truth[:,0], truth[:,2], lw=0.5)
ax[0].set_xlabel('x'); ax[0].set_ylabel('z'); ax[0].set_title('attractor')

t = np.arange(cfg.n_steps + 1) * cfg.dt
ax[1].plot(t, truth[:,0], label='truth')
ax[1].scatter(obs_idx * cfg.dt, obs[:,0], s=8, c='r', label='obs')
ax[1].set_xlabel('time'); ax[1].set_ylabel('x'); ax[1].legend()
plt.tight_layout(); plt.show()

np.savez('data/l63_twin.npz', truth=truth, obs=obs, obs_idx=obs_idx, dt=cfg.dt)