import numpy as np
import matplotlib.pyplot as plt

sigma, rho, beta = 10.0, 28.0, 8.0/3.0

def lorenz63(s):
    x, y, z = s
    return np.array([sigma*(y - x), x*(rho - z) - y, x*y - beta*z])

def rk4(s, dt):
    k1 = lorenz63(s)
    k2 = lorenz63(s + 0.5*dt*k1)
    k3 = lorenz63(s + 0.5*dt*k2)
    k4 = lorenz63(s + dt*k3)
    return s + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

# config
dt = 0.01
n_steps = 2000
obs_every = 10
obs_std = 2.0
seed = 0

rng = np.random.default_rng(seed)

# spin up so we start on the attractor, not off it
s = np.array([1.0, 1.0, 1.0])
for _ in range(500):
    s = rk4(s, dt)

# truth trajectory
truth = np.zeros((n_steps + 1, 3))
truth[0] = s
for k in range(n_steps):
    truth[k+1] = rk4(truth[k], dt)

# observations: identity h, additive Gaussian noise
obs_idx = np.arange(obs_every, n_steps + 1, obs_every)
obs = truth[obs_idx] + rng.normal(0, obs_std, (len(obs_idx), 3))

# sanity plots
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(truth[:,0], truth[:,2], lw=0.5)
ax[0].set_xlabel('x'); ax[0].set_ylabel('z'); ax[0].set_title('attractor')

t = np.arange(n_steps + 1) * dt
ax[1].plot(t, truth[:,0], label='truth')
ax[1].scatter(obs_idx * dt, obs[:,0], s=8, c='r', label='obs')
ax[1].set_xlabel('time'); ax[1].set_ylabel('x'); ax[1].legend()
plt.tight_layout(); plt.show()

# save so tomorrow doesn't regenerate
np.savez('data/l63_twin.npz', truth=truth, obs=obs, obs_idx=obs_idx, dt=dt)