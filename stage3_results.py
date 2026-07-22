"""
stage3_errors.py — Stage 3 remedy panels for the current config (window / noise mode / jitter
read from cfg). Three error sources vs alpha — (a) Jensen bias, (b) cross-covariance error vs PF,
(c) RMSE excess over PF — for the standard EnKF and the Stage 3 remedies. Produces FOUR PNGs so
each pairwise comparison can go on its own slide:
  stage3_errors_qr_w{W}.png     EnKF vs quad-reg EnKF
  stage3_errors_ie_w{W}.png     EnKF vs iterative EnKF
  stage3_errors_qrie_w{W}.png   quad-reg EnKF vs iterative EnKF
  stage3_errors_w{W}.png        all three overlaid
Reads data/stage2_results_w{W}_{mode}_{jitter}.npz (run stage2.py first). A series whose fields
are absent from the npz is skipped.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from config import cfg

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/results
# 3.8  Changed: emit four PNGs (QR-vs-EnKF, IEnKF-vs-EnKF, QR-vs-IEnKF, all-three) instead of one
# 3.6  Changed: all three panels use shaded +/-1 std (across seeds) bands instead of errorbar caps
# 3.5  Added: iterative-EnKF line on all three metric panels; panels skip absent remedies
# 3.2  created — Stage 3 error panel (standard EnKF vs quad-reg EnKF, 2 lines per metric)
# ============================================================

os.makedirs('figs/results', exist_ok=True)
W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if not os.path.exists(f):
    raise SystemExit(f'{f} not found — run stage2.py first')

d = np.load(f); a = d['alphas']
sd = lambda k: d[k] if k in d.files else np.zeros_like(a)   # seed std if present

# each filter: (label, colour, marker, jensen key, crosscov key, excess key)
ENKF = ('EnKF',           '#c0392b', 'o-', 'jensen_norm',    'crosscov_err',    'excess_pct')
QR   = ('quad-reg EnKF',  '#e67e22', 'D-', 'jensen_qr_norm', 'crosscov_qr_err', 'excess_qr_pct')
IE   = ('iterative EnKF', '#2471a3', '^-', 'jensen_ie_norm', 'crosscov_ie_err', 'excess_ie_pct')

def make(series, suffix, title):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    axes[2].axhline(0, ls='--', color='#777', lw=1)                    # PF baseline (0% excess)
    for label, col, mk, kj, kc, ke in series:
        if kj not in d.files:
            continue                                                   # remedy absent from this npz
        for ax, key in zip(axes, (kj, kc, ke)):
            ax.plot(a, d[key], mk, color=col, lw=2, label=label)
            ax.fill_between(a, d[key] - sd(key + '_std'), d[key] + sd(key + '_std'), color=col, alpha=0.2)
    axes[0].set_title('(a) Jensen bias'); axes[0].set_ylabel(r'$\|E[h(x)]-h(E[x])\|$')
    axes[1].set_title('(b) Cross-cov error')
    axes[2].set_title('(c) RMSE excess over PF'); axes[2].set_ylabel('excess over PF (%)')
    for ax in axes:
        ax.set_xlabel(r'$\alpha$'); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle(rf'Stage 3: {title} — window {W}, {mode}, {jit} jitter', y=1.02)
    fig.tight_layout()
    out = f'figs/results/stage3_errors_{suffix}w{W}.png'
    fig.savefig(out, dpi=140, bbox_inches='tight'); plt.close(fig)
    print(f'saved {out}')

make([ENKF, QR],     'qr_',   'quad-reg vs standard EnKF')
make([ENKF, IE],     'ie_',   'iterative vs standard EnKF')
make([QR,   IE],     'qrie_', 'quad-reg vs iterative EnKF')
make([ENKF, QR, IE], '',      'nonlinear remedies vs standard EnKF')
