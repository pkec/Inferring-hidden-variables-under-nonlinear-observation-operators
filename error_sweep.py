"""
stage2.py — quantify the three error sources a nonlinear h introduces, as functions of alpha:
  (1) linearisation (Jensen) bias  E[h(x)] - h(E[x])
  (2) cross-covariance error       || Cxh(EnKF) - Cxh(PF) ||_F   (PF = calibrated reference)
  (3) analysis RMSE                EnKF vs PF, both calibrated, and the EnKF excess over PF
The quadratic-regression EnKF (enkf_qr.run_enkf_qr) is run alongside as the Stage 3 remedy; its
RMSE/excess AND its Jensen bias + cross-cov error are saved so stage3_errors.py can overlay it on
the standard EnKF. Jitter strategy is set by cfg.jitter_mode: 'variable' tunes the jitter per alpha
(spread/RMSE closest to 1, falling back to the least-divergent run), 'fixed' uses the per-window
baseline jitter as-is (no multiplier) for all alpha. Saves data/stage2_results_w{W}_{mode}_{jitter}.npz
only — plotting lives in stage2_errors.py / stage3_errors.py / calibration_ratio.py /
jitter_diagnostic.py. Run after truth_obs.py.
"""
import numpy as np
from config import cfg
from partfilt import run_pf
from enkf import run_enkf
from enkf_qr import run_enkf_qr
from enkf_ienkf import run_enkf_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.5  Added: iterative EnKF (enkf_ienkf) run alongside EnKF/QR/PF under the same jitter
#             calibration; saves rmse_ie, excess_ie_pct, jensen_ie_norm, crosscov_ie_err,
#             ie_mult, ie_ratio, ie_spread (+ _std) for the Stage 3 three-way overlay
# 3.2  Added: QR Jensen bias + QR cross-cov error saved (jensen_qr_norm / crosscov_qr_err, +std),
#             so stage3_errors.py can show 2 lines (EnKF vs quad-reg EnKF) on every metric
# 3.0  Added: quadratic-regression EnKF run alongside EnKF/PF (same jitter calibration), saving
#             rmse_qr, excess_qr_pct, qr_mult, qr_ratio, qr_spread (+ _std) for the Stage 3 overlay
# 2.14 Changed: every metric now averaged over cfg.seeds (default 3). Each seed draws its own obs
#               noise realization (reused across alpha) and its own filter RNG; truth is shared.
#               Saves seed mean under the old field names + a _std field per metric, plus
#               en_spread/pf_spread (+std). calibrate()/run_at() take an explicit seed.
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
obs_std_alpha = data['obs_std_alpha']         # per-alpha noise std (seed-independent), reused below
alphas = data['alphas']
A = len(alphas)
truth_at_obs = truth[obs_idx]
seeds = cfg.seeds
S = len(seeds)
noise = [np.random.default_rng(s).normal(0, 1, (len(obs_idx), 3)) for s in seeds]  # one draw per seed

clim = truth.std(axis=0).mean()           # no-skill scale, used to flag divergence
base = cfg.perturb_std.copy()
jitter_mults = np.geomspace(0.3, 30, 9)   # two-sided: can lower OR raise the base jitter

def calibrate(run_fn, obs, h, ostd, seed):
    """Per-alpha jitter tuning. Among runs that still track the truth, keep the one with
    spread/RMSE closest to 1; if none track, keep the lowest-RMSE (least divergent) run.
    Returns (output dict, chosen multiplier, achieved spread/RMSE ratio)."""
    best_cal, best_any = None, None
    for m in jitter_mults:
        cfg.perturb_std = base * m
        out = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd)
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

def run_at(run_fn, obs, h, ostd, m, seed):
    """Single run at a preset jitter multiple (no scan). Returns (output, m, spread/RMSE)."""
    cfg.perturb_std = base * m
    out = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd)
    cfg.perturb_std = base
    ratio = np.sqrt(out['spread'].mean()) / np.sqrt(out['sqerror'].mean())
    return out, m, ratio

# one (A x S) table per metric; averaged over seeds at the end
keys = ('jensen', 'crosscov', 'jensen_qr', 'crosscov_qr', 'jensen_ie', 'crosscov_ie',
        'rmse_en', 'rmse_pf', 'rmse_qr', 'rmse_ie',
        'en_mult', 'pf_mult', 'qr_mult', 'ie_mult',
        'en_ratio', 'pf_ratio', 'qr_ratio', 'ie_ratio',
        'en_spread', 'pf_spread', 'qr_spread', 'ie_spread',
        'excess', 'excess_qr', 'excess_ie')
acc = {k: np.zeros((A, S)) for k in keys}

for ai, a in enumerate(alphas):
    h = lambda x, a=a: x + a * x**2
    h_truth = truth_at_obs + a * truth_at_obs**2          # noiseless h(truth), shared across seeds
    for si, s in enumerate(seeds):
        obs = h_truth + noise[si] * obs_std_alpha[ai]     # this seed's observation realization
        if cfg.jitter_mode == 'fixed':
            en, em, er = run_at(run_enkf,       obs, h, obs_std_alpha[ai], 1.0, s)   # baseline, as-is
            qr, qm, qrr = run_at(run_enkf_qr,    obs, h, obs_std_alpha[ai], 1.0, s)
            ie, im, ir = run_at(run_enkf_ienkf, obs, h, obs_std_alpha[ai], 1.0, s)
            pf, pm, pr = run_at(run_pf,         obs, h, obs_std_alpha[ai], 1.0, s)
        else:
            en, em, er = calibrate(run_enkf,       obs, h, obs_std_alpha[ai], s)
            qr, qm, qrr = calibrate(run_enkf_qr,    obs, h, obs_std_alpha[ai], s)
            ie, im, ir = calibrate(run_enkf_ienkf, obs, h, obs_std_alpha[ai], s)
            pf, pm, pr = calibrate(run_pf,         obs, h, obs_std_alpha[ai], s)

        ren = np.sqrt(en['sqerror'].mean()); rpf = np.sqrt(pf['sqerror'].mean())
        rqr = np.sqrt(qr['sqerror'].mean()); rie = np.sqrt(ie['sqerror'].mean())
        acc['jensen'][ai, si]      = np.linalg.norm(en['jensen'], axis=1).mean()
        acc['crosscov'][ai, si]    = np.linalg.norm(en['cross'] - pf['cross'], axis=(1, 2)).mean()
        acc['jensen_qr'][ai, si]   = np.linalg.norm(qr['jensen'], axis=1).mean()
        acc['crosscov_qr'][ai, si] = np.linalg.norm(qr['cross'] - pf['cross'], axis=(1, 2)).mean()
        acc['jensen_ie'][ai, si]   = np.linalg.norm(ie['jensen'], axis=1).mean()
        acc['crosscov_ie'][ai, si] = np.linalg.norm(ie['cross'] - pf['cross'], axis=(1, 2)).mean()
        acc['rmse_en'][ai, si]  = ren
        acc['rmse_pf'][ai, si]  = rpf
        acc['rmse_qr'][ai, si]  = rqr
        acc['rmse_ie'][ai, si]  = rie
        acc['excess'][ai, si]    = (ren - rpf) / rpf * 100
        acc['excess_qr'][ai, si] = (rqr - rpf) / rpf * 100
        acc['excess_ie'][ai, si] = (rie - rpf) / rpf * 100
        acc['en_mult'][ai, si]  = em; acc['pf_mult'][ai, si] = pm
        acc['qr_mult'][ai, si]  = qm; acc['ie_mult'][ai, si] = im
        acc['en_ratio'][ai, si] = er; acc['pf_ratio'][ai, si] = pr
        acc['qr_ratio'][ai, si] = qrr; acc['ie_ratio'][ai, si] = ir
        acc['en_spread'][ai, si] = er * ren               # spread = (spread/RMSE) x RMSE
        acc['pf_spread'][ai, si] = pr * rpf
        acc['qr_spread'][ai, si] = qrr * rqr
        acc['ie_spread'][ai, si] = ir * rie

mean = {k: acc[k].mean(1) for k in keys}                  # seed mean per alpha
sd = {k: acc[k].std(1) for k in keys}                     # seed spread per alpha

jensen_norm, crosscov_err = mean['jensen'], mean['crosscov']
rmse_en, rmse_pf, rmse_qr = mean['rmse_en'], mean['rmse_pf'], mean['rmse_qr']
excess_pct, excess_qr_pct = mean['excess'], mean['excess_qr']
en_mult, pf_mult = mean['en_mult'], mean['pf_mult']
en_ratio, pf_ratio = mean['en_ratio'], mean['pf_ratio']

tag = f'w{cfg.obs_every}_{cfg.noise_mode}_{cfg.jitter_mode}'
np.savez(f'data/stage2_results_{tag}.npz',
         alphas=alphas, seeds=np.array(seeds),
         jensen_norm=jensen_norm, crosscov_err=crosscov_err,
         jensen_qr_norm=mean['jensen_qr'], crosscov_qr_err=mean['crosscov_qr'],
         jensen_ie_norm=mean['jensen_ie'], crosscov_ie_err=mean['crosscov_ie'],
         rmse_en=rmse_en, rmse_pf=rmse_pf, rmse_qr=rmse_qr, rmse_ie=mean['rmse_ie'],
         excess_pct=excess_pct, excess_qr_pct=excess_qr_pct, excess_ie_pct=mean['excess_ie'],
         en_mult=en_mult, pf_mult=pf_mult, qr_mult=mean['qr_mult'], ie_mult=mean['ie_mult'],
         en_ratio=en_ratio, pf_ratio=pf_ratio, qr_ratio=mean['qr_ratio'], ie_ratio=mean['ie_ratio'],
         en_spread=mean['en_spread'], pf_spread=mean['pf_spread'],
         qr_spread=mean['qr_spread'], ie_spread=mean['ie_spread'],
         jensen_norm_std=sd['jensen'], crosscov_err_std=sd['crosscov'],
         jensen_qr_norm_std=sd['jensen_qr'], crosscov_qr_err_std=sd['crosscov_qr'],
         jensen_ie_norm_std=sd['jensen_ie'], crosscov_ie_err_std=sd['crosscov_ie'],
         rmse_en_std=sd['rmse_en'], rmse_pf_std=sd['rmse_pf'],
         rmse_qr_std=sd['rmse_qr'], rmse_ie_std=sd['rmse_ie'],
         excess_pct_std=sd['excess'], excess_qr_pct_std=sd['excess_qr'],
         excess_ie_pct_std=sd['excess_ie'],
         en_mult_std=sd['en_mult'], pf_mult_std=sd['pf_mult'],
         qr_mult_std=sd['qr_mult'], ie_mult_std=sd['ie_mult'],
         en_ratio_std=sd['en_ratio'], pf_ratio_std=sd['pf_ratio'],
         qr_ratio_std=sd['qr_ratio'], ie_ratio_std=sd['ie_ratio'],
         en_spread_std=sd['en_spread'], pf_spread_std=sd['pf_spread'],
         qr_spread_std=sd['qr_spread'], ie_spread_std=sd['ie_spread'])

print(f"seeds {seeds}  (mean +/- std across seeds)")
print(f"{'alpha':>6}{'jsEn':>8}{'jsQR':>8}{'jsIE':>8}{'xcEn':>9}{'xcQR':>9}{'xcIE':>9}"
      f"{'RMSE_En':>9}{'RMSE_QR':>9}{'RMSE_IE':>9}{'RMSE_PF':>9}{'exc%':>7}{'excQR%':>8}{'excIE%':>8}")
for ai, a in enumerate(alphas):
    print(f"{a:6.1f}{jensen_norm[ai]:8.3f}{mean['jensen_qr'][ai]:8.3f}{mean['jensen_ie'][ai]:8.3f}"
          f"{crosscov_err[ai]:9.2f}{mean['crosscov_qr'][ai]:9.2f}{mean['crosscov_ie'][ai]:9.2f}"
          f"{rmse_en[ai]:9.3f}{rmse_qr[ai]:9.3f}{mean['rmse_ie'][ai]:9.3f}{rmse_pf[ai]:9.3f}"
          f"{excess_pct[ai]:6.1f}%{excess_qr_pct[ai]:7.1f}%{mean['excess_ie'][ai]:7.1f}%")

print(f'saved data/stage2_results_{tag}.npz')
