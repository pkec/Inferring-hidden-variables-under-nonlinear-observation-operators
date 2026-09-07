# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching error_sweep.py 5.16. Print-only; calibrate() already used the
#      per-component form (r = sqrt(sqerror.mean(0)), rm = r.mean()), so the logged runs
#      themselves are unchanged — only the reported RMSE and excess lines move.
# 4.41 Fixed: calibrate() treated a diverged run as a valid candidate — a non-finite RMSE
#      became best_any on the first multiplier and stuck (every later `rm < nan` is False).
#      Diverged multipliers are skipped; an all-diverged scan now fails loudly.
# 4.39 Changed: comment corrected — the +100 seeds are now the blind TRAINING population and
#      cfg.seeds is what the blind test is scored on. Both sets are still logged here
#      unconditionally, so switching between 1 and 3 training seeds needs no re-log.
# 4.18 Changed: logs 6 seeds — cfg.seeds for training plus a disjoint set (+100) reserved
#      for the blind test, so the held-out runs share no observation noise with training.
# 4.11 Added: the chosen jitter multipliers (en_mult / pf_mult) are saved, so the live
#      inject filter can run at the SAME calibration the log was produced under —
#      otherwise the corrected filter is compared against a calibrated baseline it
#      could never match (uncalibrated RMSE 3.9 vs calibrated 0.9).
# 4.9  Changed: now sweeps ALPHAS x SEEDS instead of one alpha/one seed. Each seed draws
#      its own observation-noise realisation (matching error_sweep.py) so seeds give
#      independent training data. Filenames carry the seed: stage4_log_a{a}_s{seed}_{mode}.npz.
# 4.0  created — Stage 4 logging harness (single alpha, single seed).
# ============================================================

import numpy as np
from config import cfg
from enkf import run_enkf
from partfilt import run_pf
from rls_core import rmse_percomp

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]              # nonlinearity strengths to log; list
# Both populations, always. rls_blind.py fits its frozen weights on cfg.blind_seeds (a subset
# of the +100 set) and scores the blind test on cfg.seeds, so all six are logged here and the
# choice of how many training seeds is made there. The two sets must stay disjoint.
SEEDS = list(cfg.seeds) + [s + 100 for s in cfg.seeds]   # e.g. [0,1,2] + [100,101,102]

# --- twin data (built by truth_obs.py; truth is shared by every run) ---
data = np.load('data/l63_twin.npz')
truth = data['truth']                            # (n_steps+1, 3) true trajectory
obs_idx = data['obs_idx']                        # (n_cycles,) indices into truth at obs times
alphas_grid = data['alphas']                     # (n_alpha,) the swept grid
obs_std_alpha = data['obs_std_alpha']            # (n_alpha, 3) noise std used at each alpha
truth_at_obs = truth[obs_idx]                    # (n_cycles, 3) true state at each cycle
times = obs_idx * data['dt']                     # (n_cycles,) physical time per cycle

clim = truth.std(axis=0).mean()                  # no-skill scale; scalar, flags divergence
base = cfg.perturb_std.copy()                    # window baseline jitter; (3,)
jitter_mults = np.geomspace(0.3, 30, 9)          # two-sided multiplier grid; (9,)

# one unit-normal noise draw per seed, reused across alpha (same recipe as error_sweep.py)
noise = {s: np.random.default_rng(s).normal(0, 1, (len(obs_idx), 3)) for s in SEEDS}


def calibrate(run_fn, obs, h, ostd, seed):
    best_cal, best_any = None, None
    for m in jitter_mults:
        cfg.perturb_std = base * m               # (3,) trial jitter
        if run_fn is run_enkf:
            out = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd, save_forecast=True)
        else:
            out = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd, save_particles=True)
        r = np.sqrt(out['sqerror'].mean(0))      # (3,) per-axis RMSE at this jitter
        sp = np.sqrt(out['spread'].mean(0))      # (3,) per-axis spread
        rm = r.mean()                            # scalar mean RMSE
        # a diverged multiplier is not a candidate: `rm < nan` is always False, so without
        # this skip the first NaN would stick as best_any for the rest of the scan
        if not np.isfinite(rm):
            continue
        if best_any is None or rm < best_any[1]:
            best_any = (out, rm, m)
        if rm < 0.25 * clim:                     # tracking; eligible for calibration
            gap = abs((r / sp).mean() - 1)       # distance of spread/RMSE from 1
            if best_cal is None or gap < best_cal[1]:
                best_cal = (out, gap, m)
    cfg.perturb_std = base                       # restore
    if best_cal is None and best_any is None:
        raise SystemExit(f'{run_fn.__name__} diverged at every jitter multiplier '
                         f'(seed={seed}) — no usable run to log')
    pick = best_cal or best_any
    return pick[0], pick[2]                      # (output dict, chosen jitter multiplier)


for a in ALPHAS:
    ai = int(np.argmin(np.abs(alphas_grid - a)))     # index of this alpha in the twin grid; int
    ostd = obs_std_alpha[ai]                          # (3,) noise std for this alpha
    h = lambda x, a=a: x + a * x**2                   # h_alpha; a bound so the closure is correct
    h_truth = truth_at_obs + a * truth_at_obs**2      # (n_cycles, 3) noiseless h(truth), shared across seeds

    for s in SEEDS:
        obs = h_truth + noise[s] * ostd               # (n_cycles, 3) this seed's observation realisation
        en, en_mult = calibrate(run_enkf, obs, h, ostd, s)     # EnKF dict (fc_history, jensen, en_mean, ...)
        pf, pf_mult = calibrate(run_pf, obs, h, ostd, s)       # PF dict (en_mean is the teacher)

        out = f'data/stage4_log_a{a}_s{s}_{cfg.noise_mode}.npz'
        np.savez(
            out,
            fc_enkf=en['fc_history'],                 # (n_cycles, N, 3) EnKF forecast ensemble
            xa_mean=en['en_mean'],                    # (n_cycles, 3) EnKF ANALYSIS mean (post-update)
            jensen=en['jensen'],                      # (n_cycles, 3) measured E[h(x)]-h(E[x])
            xpf_mean=pf['en_mean'],                   # (n_cycles, 3) PF posterior mean (teacher)
            obs=obs,                                  # (n_cycles, 3) observations
            truth_at_obs=truth_at_obs,                # (n_cycles, 3) true state at each cycle
            times=times, alpha=a, seed=s, en_mult=en_mult, pf_mult=pf_mult,
        )
        rmse_en = rmse_percomp(en['en_mean'], truth_at_obs)
        rmse_pf = rmse_percomp(pf['en_mean'], truth_at_obs)
        print(f'alpha={a}  seed={s}  RMSE enkf={rmse_en:.3f} pf={rmse_pf:.3f} '
              f'excess={(rmse_en - rmse_pf) / rmse_pf * 100:5.1f}%  -> {out}')

print(f'\nlogged {len(ALPHAS)} alphas x {len(SEEDS)} seeds = {len(ALPHAS) * len(SEEDS)} runs')