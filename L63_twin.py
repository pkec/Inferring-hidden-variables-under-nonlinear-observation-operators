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
obs_noise = rng.normal(0, 1, (len(obs_idx), 3)) * cfg.obs_std
obs_linear = truth[obs_idx] + obs_noise

# nonlinear observations for each alpha: shape (n_alphas, n_obs, 3)
# same noise realization across alphas to isolate the effect of nonlinearity
obs_nonlinear = np.stack([
    truth[obs_idx] + a * truth[obs_idx]**2 + obs_noise
    for a in cfg.alpha
])  # shape: (len(cfg.alpha), len(obs_idx), 3)

# plots
t = np.arange(cfg.n_steps + 1) * cfg.dt
fig, (ax_lin, ax_nl) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

ax_lin.plot(t, truth[:,0], lw=0.8, label='truth')
ax_lin.scatter(obs_idx * cfg.dt, obs_linear[:,0], s=8, c='r', label='obs (linear)')
ax_lin.set_ylabel('x'); ax_lin.set_title('Linear observations')
ax_lin.legend(fontsize=8)

ax_nl.plot(t, truth[:,0], lw=0.8, label='truth')
for i, a in enumerate(cfg.alpha):
    if np.isclose(a, [0.5, 1.0]).any():
        ax_nl.scatter(obs_idx * cfg.dt, obs_nonlinear[i,:,0], s=4, label=f'α={a:.1f}')
ax_nl.set_xlabel('time'); ax_nl.set_ylabel('x'); ax_nl.set_title('Nonlinear observations')
ax_nl.legend(fontsize=8)

plt.tight_layout(); plt.show()

# 3D attractor 
fig3d = plt.figure(figsize=(8, 7))
ax3d = fig3d.add_subplot(111, projection='3d')
ax3d.plot(truth[:, 0], truth[:, 1], truth[:, 2], lw=0.3, alpha=0.8, color='steelblue')
ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
ax3d.set_title('Lorenz 63 attractor — linear observations')
ax3d.legend(fontsize=8)
plt.tight_layout(); plt.show()

np.savez('data/l63_twin.npz',
         truth=truth,
         obs_linear=obs_linear,
         obs_nonlinear=obs_nonlinear,
         alphas=cfg.alpha,
         obs_idx=obs_idx,
         dt=cfg.dt)