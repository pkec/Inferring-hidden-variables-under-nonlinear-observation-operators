import numpy as np
import matplotlib.pyplot as plt
from config import cfg

twin = np.load('data/l63_twin.npz')
pf   = np.load('data/pf_results.npz')
enkf = np.load('data/enkf_results.npz')

truth = twin['truth']
obs_idx = twin['obs_idx']
obs_linear = twin['obs_linear']
time_obs = obs_idx * float(cfg.dt)
time_truth = np.arange(len(truth)) * float(cfg.dt)
labels = ['x', 'y', 'z']


def rmse(sqerr):   # (n_obs, 3) -> (3,)
    return np.sqrt(np.mean(sqerr, axis=0))

def spread(sp):    # (n_obs, 3) -> (3,)
    return np.sqrt(np.mean(sp, axis=0))


# ---- summary table (linear h) ----
pf_rmse, en_rmse = rmse(pf['sqerror_linear']), rmse(enkf['sqerror_linear'])
pf_spr, en_spr   = spread(pf['spread_linear']), spread(enkf['spread_linear'])

pf_ratio = pf_rmse / pf_spr
en_ratio = en_rmse / en_spr

# PF treated as ground truth: ratio and % excess of EnKF over PF
ratio   = en_rmse / pf_rmse            # 1.0 = perfect agreement
pct_diff = (en_rmse - pf_rmse) / pf_rmse * 100   # how much worse EnKF is, %
 
print(f"\nLinear-h baseline — PF as ground truth")
print(f"{'':>12}{'x':>9}{'y':>9}{'z':>9}")
print(f"{'PF RMSE':>12}{pf_rmse[0]:9.3f}{pf_rmse[1]:9.3f}{pf_rmse[2]:9.3f}")
print(f"{'PF spread':>12}{pf_spr[0]:9.3f}{pf_spr[1]:9.3f}{pf_spr[2]:9.3f}")
print(f"{'RMSE/spread':>12}{pf_ratio[0]:9.3f}{pf_ratio[1]:9.3f}{pf_ratio[2]:9.3f}\n")

print(f"{'EnKF RMSE':>12}{en_rmse[0]:9.3f}{en_rmse[1]:9.3f}{en_rmse[2]:9.3f}")
print(f"{'EnKF spread':>12}{en_spr[0]:9.3f}{en_spr[1]:9.3f}{en_spr[2]:9.3f}")
print(f"{'RMSE/spread':>12}{en_ratio[0]:9.3f}{en_ratio[1]:9.3f}{en_ratio[2]:9.3f}\n")

print(f"{'RMSE ratio (PF baseline)':>12}{ratio[0]:9.3f}{ratio[1]:9.3f}{ratio[2]:9.3f}")
print(f"{'% excess':>12}{pct_diff[0]:8.1f}%{pct_diff[1]:8.1f}%{pct_diff[2]:8.1f}%")

# ---- Figure 1: time traces, both filters vs truth ----
fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
for i in range(3):
    axes[i].plot(time_truth, truth[:, i], 'k-', lw=0.8, label='Truth')
    axes[i].plot(time_obs, obs_linear[:, i], 'r.', ms=2, alpha=0.3, label='Obs')
    axes[i].plot(time_obs, pf['en_mean_linear'][:, i], 'b-', lw=0.7, label='PF')
    axes[i].plot(time_obs, enkf['en_mean_linear'][:, i], 'g-', lw=0.7, label='EnKF')
    axes[i].set_ylabel(labels[i])
    if i == 0:
        axes[i].legend(loc='upper right', fontsize=8)
axes[-1].set_xlabel('Time')
fig.suptitle('Stage 1: PF vs EnKF, linear h')
plt.tight_layout()
plt.savefig('figs/stage1_traces.png', dpi=130)


# ---- Figure 2: RMSE & spread bar comparison ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(3); w = 0.35
a1.bar(x - w/2, pf_rmse, w, label='PF')
a1.bar(x + w/2, en_rmse, w, label='EnKF')
a1.set_xticks(x); a1.set_xticklabels(labels); a1.set_title('RMSE'); a1.legend()
a2.bar(x - w/2, pf_spr, w, label='PF')
a2.bar(x + w/2, en_spr, w, label='EnKF')
a2.set_xticks(x); a2.set_xticklabels(labels); a2.set_title('Spread'); a2.legend()
fig.suptitle('Stage 1: RMSE & spread agreement, linear h')
plt.tight_layout()
plt.savefig('figs/stage1_rmse_spread.png', dpi=130)


# ---- Figure 3: PF rank histograms (linear h) ----
particles = pf['particles_linear']          # (n_obs, N, 3)
truth_at_obs = pf['truth_at_obs']
T = particles.shape[0]
n_bins = 25
fig, axs = plt.subplots(1, 3, figsize=(13, 4))
for v in range(3):
    ranks = np.array([
        np.searchsorted(np.sort(particles[t, :, v]), truth_at_obs[t, v])
        for t in range(T)
    ])
    axs[v].hist(ranks, bins=n_bins, edgecolor='black', alpha=0.7)
    axs[v].axhline(T / n_bins, color='red', ls='--', label='Uniform')
    axs[v].set_title(f'Rank histogram — {labels[v]}')
    axs[v].set_xlabel('Rank')
    if v == 0:
        axs[v].legend()
fig.suptitle('Stage 1: PF rank histograms, linear h')
plt.tight_layout()
plt.savefig('figs/stage1_rank_hist.png', dpi=130)

print("\nsaved figs/stage1_*.png")
