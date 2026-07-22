"""
run.py — project pipeline, one click.
Stage 1 baseline (linear h) runs once. Stage 2/3 then run a single headline config:
window 15, fixed_snr noise, variable jitter (set via the env vars config.py reads,
so nothing edits config.py). Each script is a fresh process, so cfg state never leaks.
The Stage 3 quadratic-regression EnKF is computed inside error_sweep.py; Stage 2 and Stage 3
error panels are separate PNGs (stage2_results.py has no QR line; stage3_results.py overlays it).

Outputs:
  figs/results/stage1_traces.png / stage1_rmse_spread.png / stage1_rank_hist.png   (Stage 1)
  data/stage2_results_w15_fixed_snr_variable.npz
  figs/results/stage2_errors_w15.png   (1x3: Jensen, cross-cov, RMSE excess — standard EnKF)
  figs/results/stage3_errors_qr_w15.png    (EnKF vs quad-reg EnKF)
  figs/results/stage3_errors_ie_w15.png    (EnKF vs iterative EnKF)
  figs/results/stage3_errors_qrie_w15.png  (quad-reg vs iterative EnKF)
  figs/results/stage3_errors_w15.png       (all three overlaid)
  figs/diagnostic/calibration_ratio_w15.png   (spread/RMSE ratio, all four filters)
  figs/diagnostic/spread_rmse_w15.png         (spread & RMSE split, all four filters)
"""
import os
import subprocess

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.17 Changed: renamed for consistency — stage2.py -> error_sweep.py (compute backend),
#               stage2_errors.py -> stage2_results.py, stage3_errors.py -> stage3_results.py
# 3.16 Changed: script names/paths updated for the diagnostic_py reorg — result scripts stay
#               top-level (plots.py -> stage1_results.py), diagnostics run from diagnostic_py/;
#               output list re-pathed to figs/results (main) and figs/diagnostic (diagnostics)
# 3.8  Added: calibration_ratio.py + spread_rmse.py back into the single-config Stage 2/3 block
#             (now single-config, 4 filters) so QR/IEnKF calibration & spread/RMSE are checked
#             each run; stage3_errors.py now emits four PNGs (per-pair + all-three)
# 3.2  Changed: Stage 2/3 reduced to the single headline config (w15, fixed_snr, variable jitter);
#               added stage3_errors.py; moved calibration_ratio / spread_rmse / jitter_diagnostic /
#               enkf_qr_compare to the optional manual block (they are multi-config grid tools)
# 3.0  Added: enkf_qr_compare.py to the per-window plot block; QR-EnKF computed inside stage2.py
# 2.11 Added: spread_rmse.py to the per-window plot block (spread & RMSE split out of the ratio)
# 2.10 Changed: Stage 1 uses self-contained plots.py (one run, both windows consolidated)
# 2.8  Changed: matrix also sweeps jitter_mode; added stage2_errors.py
# 2.7  Changed: pipeline sweeps a window x noise_mode matrix via env overrides
# 2.1  created — full pipeline runner
# ============================================================

def run(script, **over):
    env = dict(os.environ, **{k: str(v) for k, v in over.items()})
    tag = '  '.join(f'{k}={v}' for k, v in over.items())
    print(f"\n=== {script}   {tag} ===")
    subprocess.run(['python', script], check=True, env=env)

# --- Stage 1 baseline (linear h): plots.py is self-contained (both windows in one figure each) ---
run('truth_obs.py')                       # twin for the SNR diagnostic (default window)
run('stage1_results.py')                  # consolidated Stage 1 figures: traces, rmse/spread, rank-hist  -> figs/results
run('diagnostic_py/stage2_snr_diagnostic.py')   # curvature/SNR diagnostic (twin-based, runs once)  -> figs/diagnostic

# --- Stage 2 / 3: single headline config (window 15, fixed_snr noise, variable jitter) ---
W, mode, jit = 15, 'fixed_snr', 'variable'
run('truth_obs.py',     L63_OBS_EVERY=W, L63_NOISE_MODE=mode)
run('error_sweep.py',   L63_OBS_EVERY=W, L63_NOISE_MODE=mode, L63_JITTER_MODE=jit)   # compute backend -> data/*.npz
run('stage2_results.py', L63_OBS_EVERY=W, L63_NOISE_MODE=mode, L63_JITTER_MODE=jit)  # Stage 2 png (no QR)
run('stage3_results.py', L63_OBS_EVERY=W, L63_NOISE_MODE=mode, L63_JITTER_MODE=jit)  # Stage 3: 4 PNGs
run('diagnostic_py/stage3_calibration_ratio.py', L63_OBS_EVERY=W, L63_NOISE_MODE=mode, L63_JITTER_MODE=jit)  # spread/RMSE ratio, 4 filters
run('diagnostic_py/stage3_spread_rmse.py',       L63_OBS_EVERY=W, L63_NOISE_MODE=mode, L63_JITTER_MODE=jit)  # spread & RMSE split, 4 filters

# optional analyses — multi-config grid tools / slower deep dives. Run by hand if wanted:
#   diagnostic_py/stage2_jitter_diagnostic.py   (calibrated-jitter vs alpha, boundary-hit flags; EnKF/PF only)
#   diagnostic_py/stage3_enkf_qr_compare.py     QR vs linear EnKF vs PF absolute RMSE (mode x jitter grid)
#   timeseries.py                               EnKF vs PF/remedy traces (alpha=0 and alpha=0.5)  -> figs/results
#   diagnostic_py/stage2_window_sweep.py        window sweep over seeds + PF fragility + non-Gaussian mechanism