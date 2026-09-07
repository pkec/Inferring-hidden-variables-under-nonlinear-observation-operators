# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.23 Fixed: EVERY ensemble size produced identical results. run_enkf_qr's signature is
#      `N=cfg.ensembleN`, and a Python default is bound once at function-definition time, so
#      mutating cfg.ensembleN after import had no effect — all five sizes ran at whatever
#      cfg.ensembleN was at import. N is now passed explicitly, and the forecast ensemble's
#      member count is asserted against it so this cannot fail silently again.
# 5.16 created — extrapolation frequency and divergence count vs ensemble size, at fixed
#      alpha. Fills the gap behind the draft's "19.7% at N_e = 50 ... 6 of 9 runs diverged"
#      sentence, which previously had no script behind it in the project.
# ============================================================

import os
import numpy as np
from config import cfg
from enkf_qr import run_enkf_qr

ALPHA = float(os.environ.get('L63_EXT_ALPHA', 0.1))
SIZES = [int(v) for v in os.environ.get('L63_EXT_N', '50,100,200,500,1000').split(',')]
SEEDS = [int(v) for v in os.environ.get('L63_EXT_SEEDS', ','.join(map(str, range(9)))).split(',')]
FAIL_FRAC = 0.25                    # same tracking threshold error_sweep.py uses

os.makedirs('data', exist_ok=True)
W, MODE = cfg.obs_every, cfg.noise_mode

tw = np.load('data/l63_twin.npz')
truth, obs_idx = tw['truth'], tw['obs_idx']
alphas, obs_std_alpha = tw['alphas'], tw['obs_std_alpha']
ai = int(np.argmin(np.abs(alphas - ALPHA)))
if abs(float(alphas[ai]) - ALPHA) > 1e-9:
    print(f'! alpha={ALPHA} is not on the twin grid; using the nearest, {alphas[ai]:g}')
ALPHA = float(alphas[ai])
ostd = obs_std_alpha[ai]

truth_at_obs = truth[obs_idx]
clim = truth.std(axis=0).mean()                       # no-skill scale
h = lambda x: x + ALPHA * x ** 2
n_cycles = len(obs_idx)

# One observation realisation per seed, drawn exactly as error_sweep.py draws it, so a seed
# here is the same experiment as a seed there.
h_truth = truth_at_obs + ALPHA * truth_at_obs ** 2
obs_of = {s: h_truth + np.random.default_rng(s).normal(0, 1, (n_cycles, 3)) * ostd
          for s in SEEDS}

K, S = len(SIZES), len(SEEDS)
frac_any = np.full((K, S), np.nan)      # fraction of cycles with ANY component outside
frac_comp = np.full((K, S), np.nan)     # fraction of cycle-component pairs outside
rmse = np.full((K, S), np.nan)          # mean-over-components RMSE
diverged = np.zeros((K, S), dtype=bool)

base_N = cfg.ensembleN
print(f'alpha={ALPHA:g}  window={W}  mode={MODE}  sizes={SIZES}  {S} runs each')

for ki, N in enumerate(SIZES):
    # N is passed EXPLICITLY. run_enkf_qr declares `N=cfg.ensembleN`, and that default was
    # evaluated once when the module was imported — setting cfg.ensembleN here does not reach
    # it. cfg is still updated in case the filter reads it internally for anything else.
    cfg.ensembleN = N
    for si, s in enumerate(SEEDS):
        out = run_enkf_qr(obs_of[s], truth, obs_idx, h,
                          N=N, seed=s, obs_std=ostd, save_forecast=True)

        fc = out['fc_history']                         # (n_cycles, N, 3) forecast members
        if fc.shape[1] != N:                           # the 5.23 bug, caught rather than plotted
            cfg.ensembleN = base_N
            raise SystemExit(
                f'asked for N_e={N} but the forecast ensemble has {fc.shape[1]} members, so '
                f'every ensemble size would produce the same numbers. A `N=cfg.ensembleN` '
                f'default is bound once at import and never sees a later change to cfg.')
        Y = h(fc)                                      # (n_cycles, N, 3) predicted observations
        lo, hi = Y.min(axis=1), Y.max(axis=1)          # (n_cycles, 3) min-max envelope over members
        d = obs_of[s]                                  # (n_cycles, 3) observations
        outside = (d < lo) | (d > hi)                  # (n_cycles, 3) boolean
        frac_any[ki, si] = outside.any(axis=1).mean() * 100      # % of cycles
        frac_comp[ki, si] = outside.mean() * 100                 # % of cycle-component pairs

        r = np.sqrt(out['sqerror'].mean(0)).mean()     # per component, then averaged (5.16)
        rmse[ki, si] = r
        diverged[ki, si] = (not np.isfinite(r)) or (r >= FAIL_FRAC * clim)

    print(f'  N_e={N:>5}: extrapolated {np.nanmean(frac_any[ki]):5.1f}% of cycles, '
          f'diverged {int(diverged[ki].sum())}/{S} runs, '
          f'RMSE {np.nanmean(rmse[ki]):.3f}')

cfg.ensembleN = base_N

out_path = f'data/stage3_qr_extrapolation_a{ALPHA:g}_w{W}_{MODE}.npz'
np.savez(out_path,
         alpha=ALPHA, ensemble_sizes=np.array(SIZES), seeds=np.array(SEEDS),
         n_runs=S, n_cycles=n_cycles, fail_frac=FAIL_FRAC, clim=clim,
         frac_any=frac_any, frac_comp=frac_comp, rmse=rmse, diverged=diverged,
         frac_any_mean=np.nanmean(frac_any, axis=1),
         frac_any_std=np.nanstd(frac_any, axis=1),
         frac_comp_mean=np.nanmean(frac_comp, axis=1),
         rmse_mean=np.nanmean(rmse, axis=1),
         n_diverged=diverged.sum(axis=1),
         criterion='observation outside the min-max envelope of h over forecast members')
print(f'\nsaved {out_path}')

print(f'\n{"N_e":>7}{"extrap % (any comp)":>22}{"extrap % (per comp)":>22}'
      f'{"diverged":>11}{"RMSE":>9}')
for ki, N in enumerate(SIZES):
    print(f'{N:>7}{np.nanmean(frac_any[ki]):>19.1f}%{np.nanmean(frac_comp[ki]):>21.1f}%'
          f'{int(diverged[ki].sum()):>7}/{S}{np.nanmean(rmse[ki]):>9.3f}')
print('\nextrap % (any comp) is the figure the write-up quotes: the share of cycles where at '
      'least one component of the observation fell outside the ensemble envelope.')