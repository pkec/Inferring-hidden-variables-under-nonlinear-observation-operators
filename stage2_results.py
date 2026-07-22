"""
stage2_errors.py — Stage 2 error panel for the current config (window / noise mode / jitter read
from cfg). Three error sources vs alpha, standard EnKF only (no quadratic regression — that lives
in stage3_errors.py, kept separate so a Stage 2 slide has no extra lines): (a) linearisation
(Jensen) bias, (b) cross-covariance error vs the PF reference, (c) EnKF RMSE excess over PF.
Reads data/stage2_results_w{W}_{mode}_{jitter}.npz (run stage2.py first).
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/results
# 3.6  Changed: RMSE-excess panel uses a shaded +/-1 std band instead of errorbar caps,
#               matching the Jensen / cross-cov panels
# 3.2  Changed: single-config 1x3 layout (was 2x6 over jitter x noise mode); dropped the quadratic-
#               regression overlay so Stage 2 shows the standard EnKF only (QR -> stage3_errors.py)
# 3.0  Changed: metric (c) is now RMSE excess over PF (one raw % line) instead of the two-line
#               EnKF/PF RMSE panel
# 2.14 Added:   +/-1 std (across seeds) error bars on RMSE / bands on the Jensen & cross-cov panels
# 2.8  created — consolidated Stage 2 error panel per window
# ============================================================

os.makedirs('figs/results', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run stage2.py first')

EN, J, C = '#c0392b', '#8e44ad', '#16a085'
d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)   # seed std, 0 on pre-2.14 files

fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

axes[0].plot(a, d['jensen_norm'], 'o-', color=J, lw=2)
axes[0].fill_between(a, d['jensen_norm'] - sd('jensen_norm_std'),
                     d['jensen_norm'] + sd('jensen_norm_std'), color=J, alpha=0.2)
axes[0].set_title('(a) Jensen bias'); axes[0].set_ylabel(r'$\|E[h(x)]-h(E[x])\|$')

axes[1].plot(a, d['crosscov_err'], 'o-', color=C, lw=2)
axes[1].fill_between(a, d['crosscov_err'] - sd('crosscov_err_std'),
                     d['crosscov_err'] + sd('crosscov_err_std'), color=C, alpha=0.2)
axes[1].set_title('(b) Cross-cov error')

axes[2].axhline(0, ls='--', color='#777', lw=1)                          # PF baseline (0% excess)
axes[2].plot(a, d['excess_pct'], 'o-', color=EN, lw=2)
axes[2].fill_between(a, d['excess_pct'] - sd('excess_pct_std'),
                     d['excess_pct'] + sd('excess_pct_std'), color=EN, alpha=0.2)
axes[2].set_title('(c) RMSE excess over PF'); axes[2].set_ylabel('excess over PF (%)')

for ax in axes:
    ax.set_xlabel(r'$\alpha$'); ax.grid(alpha=0.3)

fig.suptitle(rf'Stage 2 error sources — window {W}, {mode}, {jit} jitter,  $h_\alpha(x)=x+\alpha x^2$', y=1.02)
fig.tight_layout()
fig.savefig(f'figs/results/stage2_errors_w{W}.png', dpi=140, bbox_inches='tight')
print(f'saved figs/results/stage2_errors_w{W}.png')