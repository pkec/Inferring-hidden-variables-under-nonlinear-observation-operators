import os
import numpy as np
from dataclasses import dataclass, field

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.47 Added: fc_var_comp — per-component FORECAST variance, from out['fc_spread']. The
#      delta = alpha * s^2 argument is about the prior ensemble, but the only spread saved
#      anywhere was the ANALYSIS spread, which the update has already contracted. Keeping
#      the two named apart here stops one being quoted for the other.
# 5.44 Added: rmse_comp / spread_comp / cal_ratio — the ONE definition of the calibration
#      ratio for the whole project, plus CAL_RATIO_CONVENTION to stamp into every npz.
#      Three scripts had three different formulas for "spread/RMSE" (stage1 per component,
#      error_sweep pooled over cycles AND components, stage2_hbar per component then
#      averaged), so the same filter reported three different numbers at alpha=0. The
#      definition now lives here, beside rk4, and every script imports it.
#      Added: PERTURB_BASELINE_ALPHA — the alpha at which the per-window baseline jitter
#      was tuned (0.0). Scripts read it instead of hard-coding "if a == 0", so the rule
#      "baseline at the tuning point, scanned multipliers away from it" is stated once.
# 4.40 Added: blind_seeds (default [100,101,102]) + env override L63_BLIND_SEEDS — the runs
#      the blind RLS fits its frozen weights on; set to a single seed to train on one
#      trajectory
# 4.38 Changed: ensembleN 1000 -> 300 and now means the EnKF family only; new particleN=1000
#      for the PF. One shared size made the two filters look equally expensive.
# 2.14 Added: seeds list (default [0,1,2]) + env override L63_SEEDS, for Stage 2 seed-averaging
# 2.9  Changed: per-window baselines named W5_PERTURB_STD / W15_PERTURB_STD; verified each
#               gives a flat alpha=0 PF rank histogram (seed-avg spread/RMSE ~1.0) so 'fixed'
#               jitter uses them as-is
# 2.8  Added: jitter_mode toggle ('variable' per-alpha / 'fixed' clamp at alpha=0)
#             + env override L63_JITTER_MODE (driven by run.py matrix)
# 2.7  Added: env overrides L63_OBS_EVERY / L63_NOISE_MODE (used by run.py matrix);
#             perturb_std now auto-set per window via PERTURB_BY_WINDOW (w5 0.25%, w15 1%)
# 2.4  Changed: perturb_std retuned 0.25% -> 1% of obs_std for obs_every=15
# 2.3  Changed: obs_every 5 -> 15 (sparser obs, wider assimilation window)
# 2.1  Added TOGGLE REFERENCE comment block (below)
# 2.0  Stage 2 — nonlinear-h support
#   Added:   noise_mode toggle ('fixed_R' physical default / 'fixed_snr' control)
#   Changed: alpha grid now starts at 0.0 so the linear baseline is in-sweep
# ============================================================

# ----------------------------------------------------------------------------
# TOGGLE REFERENCE — the knobs you'll actually change between runs
#
#   noise_mode   'fixed_R'   : constant sensor std across alpha (physical default)
#                'fixed_snr' : noise scales with std(h_alpha) to hold SNR fixed,
#                              isolating curvature (use for the headline Stage 2 fig).
#
#   jitter_mode  'variable' : per-alpha jitter calibration (scan multiples of the window
#                             baseline for spread/RMSE -> 1 at each alpha) for alpha > 0;
#                             at alpha = 0 the baseline is used as-is, because that is the
#                             alpha it was tuned at — see PERTURB_BASELINE_ALPHA.
#                'fixed'    : use the window baseline jitter as-is (no multiplier) for all
#                             alpha — lets the ratio drift so the calibration loss is measured.
#
#   alpha        nonlinearity sweep grid; alpha=0 is the linear baseline h(x)=x.
#   obs_every    steps between observations (assimilation window). 5 = dense, larger = harder.
#                Auto-selects perturb_std via PERTURB_BY_WINDOW below.
#   n_steps      truth trajectory length. Longer = smoother statistics, slower.
#   ensembleN    EnKF-family ensemble size (EnKF, QR-EnKF, IEnKF, rls_inject).
#   particleN    bootstrap PF particle count; independent of ensembleN.
#   seed         RNG seed. Vary it to seed-average (error bars on the Stage 2 figure).
#   obs_std      per-axis sensor noise std; sets R and the effective SNR at alpha=0.
#   perturb_std  base ensemble jitter, set automatically from obs_every (PERTURB_BY_WINDOW);
#                error_sweep.py / stage2_window_sweep.py scan multiples of it to calibrate
#                spread/RMSE ~ 1.
#
#   run.py sweeps window x noise_mode in one click via env vars L63_OBS_EVERY /
#   L63_NOISE_MODE, which override the dataclass defaults below.
# ----------------------------------------------------------------------------

@dataclass
class L63Config:
    # Lorenz parameters
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    # nonlinearity strength; alpha=0 is the linear baseline h(x)=x
    alpha: np.ndarray = field(default_factory=lambda: np.round(np.arange(0.0, 1.01, 0.1), 2))

    # time stepping
    dt: float = 0.01
    n_steps: int = 10000

    # observation settings
    obs_every: int = 15

    # noise model: 'fixed_R' = fixed absolute sensor std (physical default);
    # 'fixed_snr' = noise std scaled with std(h_alpha(truth)) to hold SNR constant (curvature-isolating control)
    noise_mode: str = 'fixed_snr'

    # jitter strategy: 'variable' = per-alpha calibration; 'fixed' = clamp at the alpha=0 value
    jitter_mode: str = 'variable'

    # --- per-axis noise (10% of attractor range) ---
    # x ~ [-20, 20] range 40 → std 4.0
    # y ~ [-25, 25] range 50 → std 5.0
    # z ~ [  5, 48] range 43 → std 4.3
    obs_std: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.5, 2.15]))
    init_std: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.5, 2.15]))

    perturb_std: np.ndarray = field(default_factory=lambda: np.array([0.02, 0.025, 0.0215]))  # set per window below

    # observation error covariance (diagonal), derived from obs_std
    @property
    def R(self):
        return np.diag(self.obs_std ** 2)

    # reproducibility
    seed: int = 0                                              # single-seed default (Stage 1, diagnostics)
    seeds: list = field(default_factory=lambda: [0, 1, 2])     # Stage 2 seed-averaging set
    blind_seeds: list = field(default_factory=lambda: [100, 101, 102])  # blind RLS training set

    # ensemble sizes, set independently. The EnKF family needs far fewer members than the
    # PF on a 3-variable state; running both at 1000 hid that, and every cost figure came
    # out as if the PF were free.
    ensembleN: int = 300      # EnKF, QR-EnKF, IEnKF, rls_inject
    particleN: int = 1000     # bootstrap PF (the benchmark)

cfg = L63Config()

# per-window baseline ensemble jitter — each tuned so the alpha=0 PF rank histogram is flat
# (seed-averaged spread/RMSE ~ 1.0: w5 -> 1.00, w15 -> 0.99). 'fixed' jitter mode uses these
# directly with no multiplier; 'variable' mode scans multiples of the matching baseline.
W5_PERTURB_STD  = np.array([0.005, 0.00625, 0.005375])   # window 5  (0.25% of obs_std)
W15_PERTURB_STD = np.array([0.02,  0.025,   0.0215])     # window 15 (1% of obs_std)
PERTURB_BY_WINDOW = {5: W5_PERTURB_STD, 15: W15_PERTURB_STD}

# The alpha these baselines were tuned at. At this alpha the sweep uses the baseline as-is
# (multiplier 1.0, no scan), because scanning would re-tune a filter that is already
# calibrated here — and would then disagree with Stage 1, which has no scan at all. Away
# from it the operator curves, the baseline is no longer the right jitter, and the scan runs.
PERTURB_BASELINE_ALPHA = 0.0

# run.py drives the (window x noise_mode) matrix through these env vars; fall back to the defaults above
if 'L63_OBS_EVERY' in os.environ:
    cfg.obs_every = int(os.environ['L63_OBS_EVERY'])
if 'L63_NOISE_MODE' in os.environ:
    cfg.noise_mode = os.environ['L63_NOISE_MODE']
if 'L63_JITTER_MODE' in os.environ:
    cfg.jitter_mode = os.environ['L63_JITTER_MODE']
if 'L63_SEEDS' in os.environ:
    cfg.seeds = [int(x) for x in os.environ['L63_SEEDS'].split(',')]   # e.g. L63_SEEDS=0,1,2
if 'L63_BLIND_SEEDS' in os.environ:
    cfg.blind_seeds = [int(x) for x in os.environ['L63_BLIND_SEEDS'].split(',')]  # e.g. =100
if cfg.obs_every in PERTURB_BY_WINDOW:
    cfg.perturb_std = PERTURB_BY_WINDOW[cfg.obs_every]


# --- shared model functions ---

def lorenz63(s):
    x, y, z = s
    return np.array([
        cfg.sigma * (y - x),
        x * (cfg.rho - z) - y,
        x * y - cfg.beta * z
    ])

def rk4(s, dt):
    k1 = lorenz63(s)
    k2 = lorenz63(s + 0.5 * dt * k1)
    k3 = lorenz63(s + 0.5 * dt * k2)
    k4 = lorenz63(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# --- vectorized versions: act on a whole ensemble (N, 3) at once ---

def lorenz63_vec(S):
    x, y, z = S[:, 0], S[:, 1], S[:, 2]
    return np.stack([
        cfg.sigma * (y - x),
        x * (cfg.rho - z) - y,
        x * y - cfg.beta * z
    ], axis=1)

def rk4_vec(S, dt):
    k1 = lorenz63_vec(S)
    k2 = lorenz63_vec(S + 0.5 * dt * k1)
    k3 = lorenz63_vec(S + 0.5 * dt * k2)
    k4 = lorenz63_vec(S + dt * k3)
    return S + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)


# --- shared metric definitions ---

CAL_RATIO_CONVENTION = (
    'calibration ratio = mean over x, y, z of (spread_i / RMSE_i), with '
    'RMSE_i = sqrt(mean_k sqerror[k,i]) and spread_i = sqrt(mean_k spread[k,i]); '
    'formed within a single seed, then averaged over seeds. No pooled variant is saved.'
)


def rmse_comp(out):
    return np.sqrt(np.asarray(out['sqerror']).mean(0))


def spread_comp(out):
    return np.sqrt(np.asarray(out['spread']).mean(0))       # out['spread'] is VARIANCE


def cal_ratio_comp(out):
    return spread_comp(out) / rmse_comp(out)


def cal_ratio(out):
    return float(cal_ratio_comp(out).mean())


def fc_var_comp(out):
    fc = out.get('fc_spread') if hasattr(out, 'get') else None
    if fc is None:
        return None
    return np.nanmean(np.asarray(fc) ** 2, axis=0)          # out['fc_spread'] is STD


def cal_gap(out):
    return abs(cal_ratio(out) - 1.0)