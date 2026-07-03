"""
stage2.py — quantify the three error sources a nonlinear h introduces, as functions of alpha:
  (1) linearisation (Jensen) bias  E[h(x)] - h(E[x])
  (2) cross-covariance error       || Cxh(EnKF) - Cxh(PF) ||_F   (PF = calibrated reference)
  (3) analysis RMSE                EnKF vs PF, both calibrated, and the EnKF excess over PF
Jitter strategy is set by cfg.jitter_mode: 'variable' tunes the jitter per alpha (spread/RMSE
closest to 1, falling back to the least-divergent run), 'fixed' uses the per-window baseline
jitter as-is (no multiplier) for all alpha — the baseline is already calibrated to a flat alpha=0
histogram, so the ratio is left to drift and the calibration loss is measured rather than tuned
away. Saves data/stage2_results_w{W}_{mode}_{jitter}.npz only — plotting lives in
stage2_errors.py / calibration_ratio.py / jitter_diagnostic.py. Run after truth_obs.py.
"""
import numpy as np
from config import cfg
from partfilt import run_pf
from enkf import run_enkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.9  Changed: 'fixed' jitter now uses the per-window baseline as-is (m=1.0), no alpha=0 scan
# 2.8  Added:   jitter_mode 'fixed' — clamp each filter's jitter at its alpha=0 value (run_at)
#      Changed: results tagged with jitter_mode; plotting moved out (npz only now)
# 2.7  Changed: results/figure filenames tagged with window (w{obs_every}) for the matrix run
# 2.5  Added:   per-alpha spread/RMSE ratio saved (en_ratio, pf_ratio) for calibration_ratio.py
#      Changed: jitter grid ceiling 12 -> 30 (geomspace 0.3..30, 9 pts) to reduce boundary clamping
# 2.0  created — Stage 2 three-error diagnostic (Jensen, cross-cov, RMSE) with per-alpha calibration
# ============================================================

data = np.load('data/l63_twin.npz')
truth = data['truth']
obs_idx = data['obs_idx']
obs_nl = data['obs_nonlinear']
obs_std_alpha = data['obs_std_alpha']
alphas = data['alphas']
A = len(alphas)

clim = truth.std(axis=0).mean()           # no-skill scale, used to flag divergence
base = cfg.perturb_std.copy()
jitter_mults = np.geomspace(0.3, 30, 9)   # two-sided: can lower OR raise the base jitter

def calibrate(run_fn, obs, h, ostd):
    """Per-alpha jitter tuning. Among runs that still track the truth, keep the one with
    spread/RMSE closest to 1; if none track, keep the lowest-RMSE (least divergent) run.
    Returns (output dict, chosen multiplier, achieved spread/RMSE ratio)."""
    best_cal, best_any = None, None
    for m in jitter_mults:
        cfg.perturb_std = base * m
        out = run_fn(obs, truth, obs_idx, h, seed=cfg.seed, obs_std=ostd)
        r = np.sqrt(out['sqerror'].mean(0)); sp = np.sqrt(out['spread'].mean(0))
        rm = r.mean()
        if best_any is None or rm < best_any[1]:
            best_any = (out, rm, m)
        if rm < 0.25 * clim:                          # tracking; eligible for calibration
            gap = abs((r / sp).mean() - 1)
            if best_cal is None or gap < best_cal[1]:
                best_cal = (out, gap, m)
    cfg.perturb_std = base
    pick = best_cal if best_cal is not None else best_any
    ratio = np.sqrt(pick[0]['spread'].mean()) / np.sqrt(pick[0]['sqerror'].mean())  # spread/RMSE at chosen jitter
    return pick[0], pick[2], ratio

def run_at(run_fn, obs, h, ostd, m):
    """Single run at a preset jitter multiple (no scan). Returns (output, m, spread/RMSE)."""
    cfg.perturb_std = base * m
    out = run_fn(obs, truth, obs_idx, h, seed=cfg.seed, obs_std=ostd)
    cfg.perturb_std = base
    ratio = np.sqrt(out['spread'].mean()) / np.sqrt(out['sqerror'].mean())
    return out, m, ratio

jensen_norm = np.zeros(A)        # mean over window of ||E[h(x)] - h(E[x])||
crosscov_err = np.zeros(A)       # mean over window of Frobenius ||Cxh_EnKF - Cxh_PF||
rmse_en = np.zeros(A)            # calibrated-EnKF analysis RMSE (mean over x,y,z)
rmse_pf = np.zeros(A)            # calibrated-PF analysis RMSE (the non-Gaussian reference)
en_mult = np.zeros(A); pf_mult = np.zeros(A)
en_ratio = np.zeros(A); pf_ratio = np.zeros(A)   # achieved spread/RMSE at the chosen jitter

for ai, a in enumerate(alphas):
    h = lambda x, a=a: x + a * x**2
    if cfg.jitter_mode == 'fixed':
        en, em, er = run_at(run_enkf, obs_nl[ai], h, obs_std_alpha[ai], 1.0)   # window baseline, as-is
        pf, pm, pr = run_at(run_pf,   obs_nl[ai], h, obs_std_alpha[ai], 1.0)
    else:
        en, em, er = calibrate(run_enkf, obs_nl[ai], h, obs_std_alpha[ai])
        pf, pm, pr = calibrate(run_pf,   obs_nl[ai], h, obs_std_alpha[ai])

    jensen_norm[ai] = np.linalg.norm(en['jensen'], axis=1).mean()
    crosscov_err[ai] = np.linalg.norm(en['cross'] - pf['cross'], axis=(1, 2)).mean()
    rmse_en[ai] = np.sqrt(en['sqerror'].mean())
    rmse_pf[ai] = np.sqrt(pf['sqerror'].mean())
    en_mult[ai] = em; pf_mult[ai] = pm
    en_ratio[ai] = er; pf_ratio[ai] = pr

excess_pct = (rmse_en - rmse_pf) / rmse_pf * 100

tag = f'w{cfg.obs_every}_{cfg.noise_mode}_{cfg.jitter_mode}'
np.savez(f'data/stage2_results_{tag}.npz',
         alphas=alphas, jensen_norm=jensen_norm, crosscov_err=crosscov_err,
         rmse_en=rmse_en, rmse_pf=rmse_pf, excess_pct=excess_pct,
         en_mult=en_mult, pf_mult=pf_mult, en_ratio=en_ratio, pf_ratio=pf_ratio)

print(f"{'alpha':>6}{'jensen':>10}{'crosscov':>10}{'RMSE_EnKF':>11}{'RMSE_PF':>10}{'excess%':>9}")
for ai, a in enumerate(alphas):
    print(f"{a:6.1f}{jensen_norm[ai]:10.3f}{crosscov_err[ai]:10.2f}"
          f"{rmse_en[ai]:11.3f}{rmse_pf[ai]:10.3f}{excess_pct[ai]:8.1f}%")

print(f'saved data/stage2_results_{tag}.npz')
