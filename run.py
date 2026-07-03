"""
run.py — full project pipeline, one click.
Stage 1 baseline runs once, then a Stage 2 matrix sweeps window x noise_mode
({5, 15} x {fixed_R, fixed_snr}). Each combination is driven by env vars that
config.py reads (L63_OBS_EVERY / L63_NOISE_MODE), so nothing edits config.py.
Each script is a fresh process, so cfg state never leaks between runs.

Outputs:
  figs/stage1_traces.png / stage1_rmse_spread.png / stage1_rank_hist.png
                                                 (Stage 1, both windows consolidated per figure)
  data/stage2_results_w{W}_{mode}_{jitter}.npz   (8 files: 2 windows x 2 modes x 2 jitters)
  figs/stage2_errors_w{W}.png        (2x6: jitter rows x noise-mode metric cols)
  figs/calibration_ratio_w{W}.png    (2x2: noise mode x jitter strategy)
  figs/spread_rmse_w{W}.png          (2x4: spread row + RMSE row, the ratio's two terms)
  figs/jitter_diagnostic_w{W}.png    (variable jitter only)
"""
import os
import subprocess

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.11 Added: spread_rmse.py to the per-window plot block (spread & RMSE split out of the ratio)
# 2.10 Changed: Stage 1 uses self-contained plots.py (one run, both windows consolidated);
#               dropped the per-window partfilt/enkf/plots loop
# 2.9  Changed: Stage 1 now runs per window (w5 & w15); stage1 figures tagged by window
# 2.8  Changed: matrix also sweeps jitter_mode (variable/fixed) -> 8 stage2 runs;
#               added stage2_errors.py; jitter_diagnostic stays variable-only
# 2.7  Changed: pipeline now sweeps a window x noise_mode matrix in one click
#               (windows 5 & 15, modes fixed_R & fixed_snr) via env overrides
# 2.5  Added: calibration_ratio.py to the pipeline
# 2.4  Added: jitter_diagnostic.py to the pipeline
# 2.1  created — full pipeline runner
# ============================================================

windows = [5, 15]
modes = ['fixed_R', 'fixed_snr']
jitters = ['variable', 'fixed']

def run(script, **over):
    env = dict(os.environ, **{k: str(v) for k, v in over.items()})
    tag = '  '.join(f'{k}={v}' for k, v in over.items())
    print(f"\n=== {script}   {tag} ===")
    subprocess.run(['python', script], check=True, env=env)

# --- Stage 1 baseline (linear h): plots.py is self-contained (both windows in one figure each) ---
run('truth_obs.py')                       # twin for snr_diagnostic (default window)
run('plots.py')                           # consolidated Stage 1 figures: traces, rmse/spread, rank-hist
run('snr_diagnostic.py')                  # curvature/SNR diagnostic (twin-based, runs once)

# --- Stage 2 matrix: window x noise_mode x jitter_mode (8 stage2 runs) ---
for w in windows:
    for m in modes:
        run('truth_obs.py', L63_OBS_EVERY=w, L63_NOISE_MODE=m)   # twin (jitter-independent), reused below
        for j in jitters:
            run('stage2.py', L63_OBS_EVERY=w, L63_NOISE_MODE=m, L63_JITTER_MODE=j)
    # all four configs for this window are on disk -> consolidated per-window plots
    run('stage2_errors.py',     L63_OBS_EVERY=w)
    run('calibration_ratio.py', L63_OBS_EVERY=w)
    run('spread_rmse.py',       L63_OBS_EVERY=w)
    run('jitter_diagnostic.py', L63_OBS_EVERY=w)   # variable jitter only (handled inside)

# optional deeper analyses — slower, each regenerates its own truth. Run by hand if wanted:
#   timeseries.py     EnKF vs PF traces at three windows (alpha=0)
#   window_sweep.py   window sweep over seeds + PF fragility + non-Gaussian mechanism
