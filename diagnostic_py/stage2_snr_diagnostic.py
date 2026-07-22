"""
snr_diagnostic.py — show why fixed_R conflates curvature with information gain.
Operates on the truth trajectory only (no filtering), so it's cheap and noise-model-agnostic:
it just asks how informative an observation is at each alpha under each noise model.
Produces figs/diagnostic/snr_diagnostic.png.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/diagnostic (script is noise-model-agnostic by design — it
#      contrasts fixed_R vs fixed_snr, so nothing to commit here);  Added: sys.path bootstrap
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
data = np.load('data/l63_twin.npz')
truth = data['truth']; obs_idx = data['obs_idx']; alphas = data['alphas']

sig = np.array([(truth[obs_idx] + a * truth[obs_idx]**2).std(axis=0) for a in alphas])  # signal std per alpha
noise_R = np.tile(cfg.obs_std, (len(alphas), 1))                  # fixed_R: constant noise std
snr_ratio = cfg.obs_std / truth[obs_idx].std(axis=0)             # ratio reproducing obs_std at alpha=0
noise_snr = snr_ratio * sig                                       # fixed_snr: noise tracks the signal

snr_fixedR = (sig / noise_R).mean(axis=1)                         # SNR averaged over the 3 axes
snr_fixedsnr = (sig / noise_snr).mean(axis=1)

R, S = '#c0392b', '#2471a3'
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))

ax[0].plot(alphas, sig.mean(1), 'o-', color='#444', lw=2, label='signal std  (grows with curvature)')
ax[0].plot(alphas, noise_R.mean(1), 's-', color=R, lw=2, label='noise std — fixed_R')
ax[0].plot(alphas, noise_snr.mean(1), 'd-', color=S, lw=2, label='noise std — fixed_snr')
ax[0].set_xlabel(r'$\alpha$'); ax[0].set_ylabel('std (mean over x,y,z)')
ax[0].set_title('(a) signal vs noise scale'); ax[0].legend(fontsize=8)

ax[1].plot(alphas, snr_fixedR, 's-', color=R, lw=2, label='fixed_R')
ax[1].plot(alphas, snr_fixedsnr, 'd-', color=S, lw=2, label='fixed_snr')
ax[1].set_xlabel(r'$\alpha$'); ax[1].set_ylabel('effective SNR = signal std / noise std')
ax[1].set_title('(b) information content vs curvature'); ax[1].legend()
ax[1].annotate(f'{snr_fixedR[-1]/snr_fixedR[0]:.0f}x rise', xy=(1.0, snr_fixedR[-1]),
               xytext=(0.55, snr_fixedR[-1]*0.7), color=R,
               arrowprops=dict(arrowstyle='->', color=R))

for a in ax:
    a.grid(alpha=0.3)
fig.suptitle('Why fixed_R conflates two effects: curvature rises AND observations get more informative', y=1.02)
fig.tight_layout()
fig.savefig('figs/diagnostic/snr_diagnostic.png', dpi=145, bbox_inches='tight')
print(f"fixed_R SNR: {snr_fixedR[0]:.1f} -> {snr_fixedR[-1]:.1f}  ({snr_fixedR[-1]/snr_fixedR[0]:.1f}x)")
print(f"fixed_snr SNR: {snr_fixedsnr[0]:.1f} -> {snr_fixedsnr[-1]:.1f}  (flat by construction)")
print('saved figs/diagnostic/snr_diagnostic.png')
