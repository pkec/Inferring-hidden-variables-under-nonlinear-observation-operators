import numpy as np
import matplotlib.pyplot as plt
from config import cfg, rk4

data = np.load('data/l63_twin.npz')
rng = np.random.default_rng(cfg.seed)

obs = data['obs']
truth = data['truth']
obs_idx = data['obs_idx']

# per-axis initial spread
init_guess = truth[0, :]
init_ensemble = np.tile(init_guess, (cfg.ensembleN, 1)) \
    + rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.init_std

weightArray = np.ones(cfg.ensembleN) / cfg.ensembleN
pf_mean = np.zeros((len(obs), 3))
truthPF = np.zeros((len(obs), 3))
particles_history = np.zeros((len(obs), cfg.ensembleN, 3))

# precompute inverse obs covariance for likelihood
R_diag = cfg.obs_std ** 2


for k in range(len(obs)):
    for d in range(cfg.ensembleN):
        for j in range(cfg.obs_every):
            init_ensemble[d] = rk4(init_ensemble[d], cfg.dt)
        init_ensemble[d] += rng.normal(0, 1, 3) * cfg.perturb_std

    innovationPF = obs[k] - init_ensemble
    likelihoodPF = np.exp(-0.5 * np.sum(innovationPF**2 / R_diag, axis=1))
    weightArray *= likelihoodPF
    weightArray += 1e-300
    weightArray /= np.sum(weightArray)

    Neff = 1. / np.sum(weightArray**2)
    particles_history[k] = init_ensemble.copy()

    #if Neff < cfg.ensembleN / 2: (Diversity preserving) (SIR adaptive)
    positions = (np.arange(cfg.ensembleN) + np.random.rand()) / cfg.ensembleN
    cumsum = np.cumsum(weightArray)
    indices = np.searchsorted(cumsum, positions)
    init_ensemble = init_ensemble[indices]
    weightArray = np.ones(cfg.ensembleN) / cfg.ensembleN

    pf_mean[k] = np.average(init_ensemble, weights=weightArray, axis=0)
    truthPF[k] = truth[obs_idx[k]]

rmsePF = np.sqrt((1 / len(obs)) * np.sum((truthPF - pf_mean)**2))
print(rmsePF)

time_obs = obs_idx * float(cfg.dt)
time_truth = np.arange(len(truth)) * float(cfg.dt)
labels = ['x', 'y', 'z']

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
for i in range(3):
    axes[i].plot(time_truth, truth[:, i], 'k-', lw=0.8, label='Truth')
    axes[i].plot(time_obs, obs[:, i], 'r.', ms=3, alpha=0.4, label='Obs')
    axes[i].plot(time_obs, pf_mean[:, i], 'b.', lw=1.2, label='PF estimate')
    axes[i].set_ylabel(labels[i])
    if i == 0:
        axes[i].legend(loc='upper right', fontsize=9)

axes[-1].set_xlabel('Time')
fig.suptitle(f'Bootstrap PF (N={cfg.ensembleN})')
plt.tight_layout()
plt.show()


def rank_histogram(particles_history, truth_at_obs, var_idx=0, n_display_bins=20):
    T, N, _ = particles_history.shape
    ranks = np.zeros(T, dtype=int)
    for t in range(T):
        sorted_particles = np.sort(particles_history[t, :, var_idx])
        ranks[t] = np.searchsorted(sorted_particles, truth_at_obs[t, var_idx])

    expected = T / n_display_bins

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(ranks, bins=n_display_bins, edgecolor='black', alpha=0.7)
    ax.axhline(expected, color='red', ls='--', label='Uniform')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Count')
    ax.set_title(f'Rank Histogram — {["x","y","z"][var_idx]} component')
    ax.legend()
    plt.tight_layout()
    return fig

for v in range(3):
    rank_histogram(particles_history, truthPF, var_idx=v)
plt.show()