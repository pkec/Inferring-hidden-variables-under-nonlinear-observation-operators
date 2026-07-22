"""
stage4_log.py — Stage 4 logging harness.

Runs the EnKF and the bootstrap PF on the SAME observation sequence at a chosen
alpha, each under the project's own per-alpha jitter calibration (the same rule
error_sweep.py uses), and saves the per-cycle forecast ensembles + both filter
means to a .npz. This is the expensive PF-included pass; all feature building,
standardisation and RLS iteration happen offline in stage4_rls.py on these arrays,
so the filters never need re-running.

Design notes (why this shape):
  - enkf.py already returns everything needed EXCEPT hbar and the forecast mean,
    but it returns fc_history (the full forecast ensemble per cycle). hbar,
    forecast mean and forecast variance are reconstructed from fc_history offline
    in stage4_rls.py, so enkf.py is not touched.
  - jensen (E[h(x)] - h(E[x])) is already returned per cycle by run_enkf as the
    numerically MEASURED value the filter used; we log that, not alpha*s2.
  - the calibration below is copied inline (not imported) because error_sweep.py
    runs a full sweep on import and mutates cfg.perturb_std; a fresh process with
    the loop inlined keeps that side-effect out.

Run once per alpha (Stage 4 plan calls for 3 nonlinearity strengths).
"""

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 4.0  created — Stage 4 logging harness. Runs EnKF + PF on one obs sequence at a
#      chosen alpha under per-filter jitter calibration; saves fc_history for both
#      filters, both en_means, jensen (measured), truth, obs, times to
#      data/stage4_log_a{alpha}_{mode}.npz for offline RLS. No edits to enkf.py.
# ============================================================

import numpy as np
from config import cfg
from enkf import run_enkf
from partfilt import run_pf

ALPHA = 0.5                                     # nonlinearity strength for this log run; scalar. rerun per alpha
h = lambda x: x + ALPHA * x**2                  # observation operator h_alpha(x) = x + alpha x^2; elementwise
SEED = cfg.seed                                 # single seed for the logged run; int

# --- twin data (built by truth_obs.py; same obs every filter sees) ---
data = np.load('data/l63_twin.npz')
truth = data['truth']                           # (n_steps+1, 3) true trajectory
obs_idx = data['obs_idx']                        # (n_cycles,) indices into truth at obs times
alphas = data['alphas']                          # (n_alpha,) the swept grid
obs_std_alpha = data['obs_std_alpha']            # (n_alpha, 3) noise std used at each alpha
obs_nonlinear = data['obs_nonlinear']            # (n_alpha, n_cycles, 3) observations per alpha

ai = int(np.argmin(np.abs(alphas - ALPHA)))      # index of our alpha in the grid; int
obs = obs_nonlinear[ai]                          # (n_cycles, 3) the observation sequence for this alpha
ostd = obs_std_alpha[ai]                         # (3,) matching noise std

# --- per-filter jitter calibration (inlined from error_sweep.calibrate) ---
# Each filter gets the jitter multiple that best makes spread/RMSE ~ 1 while still
# tracking; PF and EnKF may land on different multiples — that is correct, each is
# individually calibrated. Obs are identical, so r_t compares filters on one world.
clim = truth.std(axis=0).mean()                  # no-skill scale; scalar, flags divergence
base = cfg.perturb_std.copy()                    # window baseline jitter; (3,)
jitter_mults = np.geomspace(0.3, 30, 9)          # two-sided multiplier grid; (9,)

def calibrate(run_fn):
    """Return the calibrated output dict for one filter (forecast ensemble saved)."""
    best_cal, best_any = None, None
    for m in jitter_mults:
        cfg.perturb_std = base * m               # (3,) trial jitter
        if run_fn is run_enkf:
            out = run_fn(obs, truth, obs_idx, h, seed=SEED, obs_std=ostd, save_forecast=True)
        else:
            out = run_fn(obs, truth, obs_idx, h, seed=SEED, obs_std=ostd, save_particles=True)
        r = np.sqrt(out['sqerror'].mean(0))      # (3,) per-axis RMSE at this jitter
        sp = np.sqrt(out['spread'].mean(0))      # (3,) per-axis spread
        rm = r.mean()                            # scalar mean RMSE
        if best_any is None or rm < best_any[1]:
            best_any = (out, rm)
        if rm < 0.25 * clim:                     # tracking; eligible for calibration
            gap = abs((r / sp).mean() - 1)       # distance of spread/RMSE from 1
            if best_cal is None or gap < best_cal[1]:
                best_cal = (out, gap)
    cfg.perturb_std = base                        # restore
    return (best_cal or best_any)[0]

en = calibrate(run_enkf)                          # EnKF output dict (fc_history, jensen, en_mean, ...)
pf = calibrate(run_pf)                            # PF output dict (particles_history, en_mean, ...)

# --- collect what stage4_rls.py needs ---
# run_pf saves the forecast ensemble under 'particles_history'; run_enkf under
# 'fc_history'. Both are (n_cycles, N, 3) forecast (pre-update) ensembles. We log
# the EnKF forecast ensemble (features are EnKF-observable quantities).
times = obs_idx * data['dt']                      # (n_cycles,) physical time per cycle

out = f'data/stage4_log_a{ALPHA}_{cfg.noise_mode}.npz'
np.savez(
    out,
    fc_enkf=en['fc_history'],                     # (n_cycles, N, 3) EnKF forecast ensemble
    xa_mean=en['en_mean'],                        # (n_cycles, 3) EnKF ANALYSIS mean (post-update)
    jensen=en['jensen'],                          # (n_cycles, 3) measured E[h(x)]-h(E[x]) the filter used
    xpf_mean=pf['en_mean'],                       # (n_cycles, 3) PF posterior (weighted) mean
    obs=obs,                                       # (n_cycles, 3) observations
    truth_at_obs=truth[obs_idx],                  # (n_cycles, 3) true state at each cycle
    times=times, alpha=ALPHA,
)
print(f"logged {len(obs_idx)} cycles at alpha={ALPHA} -> {out}")
