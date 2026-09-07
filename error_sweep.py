import numpy as np
from config import (cfg, cal_ratio, cal_ratio_comp, cal_gap, rmse_comp, spread_comp,
                    fc_var_comp, CAL_RATIO_CONVENTION, PERTURB_BASELINE_ALPHA)
from partfilt import run_pf
from enkf import run_enkf
from enkf_qr import run_enkf_qr
from enkf_ienkf import run_enkf_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.47 Added: FORECAST variance saved per alpha, per seed, per component (en_fcvar, qr_fcvar,
#      ie_fcvar). enkf.py has always returned fc_spread — the prior spread measured after
#      propagation and before the update — and this file has always thrown it away, so any
#      claim about the ensemble contracting had only the ANALYSIS spread to lean on. Those
#      are different quantities: the analysis spread is the forecast spread after the update
#      has shrunk it. The delta = alpha * s^2 identity is about the forecast one.
#      NOTE: run_pf returns no fc_spread, so there is no pf_fcvar. A filter that does not
#      return fc_spread records NaN and says so once, rather than failing the sweep.
# 5.44 Changed: the calibration ratio is now config.cal_ratio — mean over components of
#      spread_i/RMSE_i — instead of the pooled sqrt(mean spread)/sqrt(mean sqerror) this
#      file used. The pooled form divides one number averaged over cycles AND components by
#      another, which weights each axis by its own error magnitude; in L63 that made the
#      quoted ratio close to a statement about z alone, and it disagreed with both
#      stage1_results.py and stage2_hbar_diagnostic.py on the SAME alpha=0 run.
#      Changed: the jitter scan now minimises |cal_ratio - 1| (config.cal_gap). It used to
#      minimise |mean_i(RMSE_i/spread_i) - 1| — the reciprocal, averaged in the other order —
#      so the multiplier was chosen to optimise a quantity the sweep never reported. Picks
#      move at some alpha, and every alpha > 0 number moves with them.
#      Changed: at alpha = PERTURB_BASELINE_ALPHA (0.0) the jitter scan does NOT run; the
#      window baseline is used as-is, multiplier 1.0. That is the alpha the baseline was
#      tuned at, so scanning re-tuned an already-calibrated filter and produced a different
#      alpha=0 column from Stage 1, which never scans. The alpha=0 column is now bit-for-bit
#      the Stage 1 run: same truth, same obs draw, same obs_std, same jitter.
#      Removed: every *_pooled key. They recorded the pre-5.16 definitions and now match no
#      script; keeping them invited exactly the mix-up this patch is fixing.
#      Added: *_by_seed arrays for every headline and per-component quantity, so the thesis
#      sheet can report seed 0 / seed 1 / seed 2 and their sample SD without a rerun.
#      Changed: *_std is the SAMPLE standard deviation (ddof=1), matching
#      stage2_hbar_diagnostic.py and the STDEV.S in the thesis sheet. It was the population
#      SD (ddof=0), so every seed band in Stage 2/3 widens by sqrt(3/2) ~ 1.22x.
#      Added: an invariant check that headline ratio == mean of the per-component ratios,
#      so a future edit that reintroduces a second definition fails loudly here.
# 5.16 Changed: RMSE and excess are now per-component-then-averaged, not pooled. Previously
#      ren = sqrt(en['sqerror'].mean()) collapsed cycles AND components in one mean, so the
#      headline was a pooled RMSE whose implied component weights are proportional to each
#      component's squared error — in L63 that gives z roughly 84% of the answer. Now
#      RMSE_i = sqrt(mean_k e_{k,i}^2) per component and the headline is their mean, so the
#      three axes weigh equally. Excess is a per-component ratio averaged the same way, which
#      is NOT the ratio of the averaged RMSEs.
#      Added:   per-component arrays saved (rmse_*_c, excess_*_c, spread_*_c, ratio_*_c, all
#               shape (A,3)), so any other aggregation is recoverable without a rerun.
#      NOTE:    every excess/RMSE number in the write-up changes with this patch. The Jensen
#               and cross-covariance metrics do NOT — they were already norms over components.
# 4.41 Fixed: calibrate() skipped nothing when a run diverged — a non-finite RMSE became
#      best_any on the first multiplier and stuck, because every later `rm < nan` is False.
#      Diverged multipliers are now skipped, and an all-diverged scan fails with a message
#      naming the filter and seed instead of returning a NaN run.
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
    best_cal, best_any = None, None
    for m in jitter_mults:
        cfg.perturb_std = base * m
        out = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd)
        rm = rmse_comp(out).mean()                    # equal-weight headline RMSE for this run
        # a diverged multiplier is not a candidate: `rm < nan` is always False, so without
        # this skip the first NaN would stick as best_any for the rest of the scan
        if not np.isfinite(rm):
            continue
        if best_any is None or rm < best_any[1]:
            best_any = (out, rm, m)
        if rm < 0.25 * clim:                          # tracking; eligible for calibration
            gap = cal_gap(out)                        # |calibration ratio - 1|
            if best_cal is None or gap < best_cal[1]:
                best_cal = (out, gap, m)
    cfg.perturb_std = base
    if best_cal is None and best_any is None:
        raise SystemExit(f'{run_fn.__name__} diverged at every jitter multiplier '
                         f'(seed={seed}) — no usable run to calibrate on')
    pick = best_cal if best_cal is not None else best_any
    return pick[0], pick[2], cal_ratio(pick[0])


def run_at(run_fn, obs, h, ostd, m, seed):
    cfg.perturb_std = base * m
    out = run_fn(obs, truth, obs_idx, h, seed=seed, obs_std=ostd)
    cfg.perturb_std = base
    ratio = cal_ratio(out)
    if not np.isfinite(ratio):
        print(f'  ! {run_fn.__name__} diverged at the fixed jitter multiplier m={m} '
              f'(seed={seed}) — ratio recorded as NaN')
    return out, m, ratio


def tune(run_fn, obs, h, ostd, seed, alpha):
    if cfg.jitter_mode == 'fixed' or float(alpha) == PERTURB_BASELINE_ALPHA:
        return run_at(run_fn, obs, h, ostd, 1.0, seed)
    return calibrate(run_fn, obs, h, ostd, seed)


# one (A x S) table per scalar metric; averaged over seeds at the end
keys = ('jensen', 'crosscov', 'jensen_qr', 'crosscov_qr', 'jensen_ie', 'crosscov_ie',
        'rmse_en', 'rmse_pf', 'rmse_qr', 'rmse_ie',
        'en_mult', 'pf_mult', 'qr_mult', 'ie_mult',
        'en_ratio', 'pf_ratio', 'qr_ratio', 'ie_ratio',
        'en_spread', 'pf_spread', 'qr_spread', 'ie_spread',
        'en_fcvar', 'qr_fcvar', 'ie_fcvar',
        'excess', 'excess_qr', 'excess_ie')
acc = {k: np.zeros((A, S)) for k in keys}

# (A x S x 3) tables — the per-component values behind every headline above
ckeys = ('rmse_en_c', 'rmse_pf_c', 'rmse_qr_c', 'rmse_ie_c',
         'excess_c', 'excess_qr_c', 'excess_ie_c',
         'en_spread_c', 'pf_spread_c', 'qr_spread_c', 'ie_spread_c',
         'en_ratio_c', 'pf_ratio_c', 'qr_ratio_c', 'ie_ratio_c',
         'en_fcvar_c', 'qr_fcvar_c', 'ie_fcvar_c')
accc = {k: np.zeros((A, S, 3)) for k in ckeys}

_no_fc = set()          # filters that returned no fc_spread; warned about once each


def fcvar(out, name):
    v = fc_var_comp(out)
    if v is None:
        if name not in _no_fc:
            _no_fc.add(name)
            print(f'  ! {name} returns no fc_spread — forecast variance recorded as NaN. '
                  f'Add the same fc_spread bookkeeping enkf.py has to make it available.')
        return np.full(3, np.nan)
    return v


for ai, a in enumerate(alphas):
    h = lambda x, a=a: x + a * x**2
    h_truth = truth_at_obs + a * truth_at_obs**2          # noiseless h(truth), shared across seeds
    for si, s in enumerate(seeds):
        obs = h_truth + noise[si] * obs_std_alpha[ai]     # this seed's observation realization
        en, em, er = tune(run_enkf,       obs, h, obs_std_alpha[ai], s, a)
        qr, qm, qrr = tune(run_enkf_qr,   obs, h, obs_std_alpha[ai], s, a)
        ie, im, ir = tune(run_enkf_ienkf, obs, h, obs_std_alpha[ai], s, a)
        pf, pm, pr = tune(run_pf,         obs, h, obs_std_alpha[ai], s, a)

        # ---- per-component RMSE (3,), then the equal-weight headline ----
        ren_c, rpf_c = rmse_comp(en), rmse_comp(pf)
        rqr_c, rie_c = rmse_comp(qr), rmse_comp(ie)
        exc_c    = (ren_c - rpf_c) / rpf_c * 100
        exc_qr_c = (rqr_c - rpf_c) / rpf_c * 100
        exc_ie_c = (rie_c - rpf_c) / rpf_c * 100
        spen_c, sppf_c = spread_comp(en), spread_comp(pf)
        spqr_c, spie_c = spread_comp(qr), spread_comp(ie)
        # FORECAST variance — the prior, before the update contracts it. Not the same thing
        # as the analysis spread above, and it is the one delta = alpha * s^2 refers to.
        fven_c, fvqr_c, fvie_c = fcvar(en, 'EnKF'), fcvar(qr, 'QR-EnKF'), fcvar(ie, 'IEnKF')

        for k, v in (('rmse_en_c', ren_c), ('rmse_pf_c', rpf_c),
                     ('rmse_qr_c', rqr_c), ('rmse_ie_c', rie_c),
                     ('excess_c', exc_c), ('excess_qr_c', exc_qr_c), ('excess_ie_c', exc_ie_c),
                     ('en_spread_c', spen_c), ('pf_spread_c', sppf_c),
                     ('qr_spread_c', spqr_c), ('ie_spread_c', spie_c),
                     ('en_ratio_c', cal_ratio_comp(en)), ('pf_ratio_c', cal_ratio_comp(pf)),
                     ('qr_ratio_c', cal_ratio_comp(qr)), ('ie_ratio_c', cal_ratio_comp(ie)),
                     ('en_fcvar_c', fven_c), ('qr_fcvar_c', fvqr_c), ('ie_fcvar_c', fvie_c)):
            accc[k][ai, si] = v

        acc['rmse_en'][ai, si] = ren_c.mean(); acc['rmse_pf'][ai, si] = rpf_c.mean()
        acc['rmse_qr'][ai, si] = rqr_c.mean(); acc['rmse_ie'][ai, si] = rie_c.mean()
        acc['excess'][ai, si]    = exc_c.mean()
        acc['excess_qr'][ai, si] = exc_qr_c.mean()
        acc['excess_ie'][ai, si] = exc_ie_c.mean()
        acc['en_spread'][ai, si] = spen_c.mean(); acc['pf_spread'][ai, si] = sppf_c.mean()
        acc['qr_spread'][ai, si] = spqr_c.mean(); acc['ie_spread'][ai, si] = spie_c.mean()
        acc['en_fcvar'][ai, si] = fven_c.mean(); acc['qr_fcvar'][ai, si] = fvqr_c.mean()
        acc['ie_fcvar'][ai, si] = fvie_c.mean()

        # ---- unchanged: both are already norms over components, not RMSEs ----
        acc['jensen'][ai, si]      = np.linalg.norm(en['jensen'], axis=1).mean()
        acc['crosscov'][ai, si]    = np.linalg.norm(en['cross'] - pf['cross'], axis=(1, 2)).mean()
        acc['jensen_qr'][ai, si]   = np.linalg.norm(qr['jensen'], axis=1).mean()
        acc['crosscov_qr'][ai, si] = np.linalg.norm(qr['cross'] - pf['cross'], axis=(1, 2)).mean()
        acc['jensen_ie'][ai, si]   = np.linalg.norm(ie['jensen'], axis=1).mean()
        acc['crosscov_ie'][ai, si] = np.linalg.norm(ie['cross'] - pf['cross'], axis=(1, 2)).mean()

        acc['en_mult'][ai, si]  = em; acc['pf_mult'][ai, si] = pm
        acc['qr_mult'][ai, si]  = qm; acc['ie_mult'][ai, si] = im
        acc['en_ratio'][ai, si] = er; acc['pf_ratio'][ai, si] = pr
        acc['qr_ratio'][ai, si] = qrr; acc['ie_ratio'][ai, si] = ir


for f in ('en', 'pf', 'qr', 'ie'):
    lhs, rhs = acc[f'{f}_ratio'], accc[f'{f}_ratio_c'].mean(axis=2)
    bad = ~np.isclose(lhs, rhs, rtol=1e-12, atol=0, equal_nan=True)
    if bad.any():
        ai, si = np.argwhere(bad)[0]
        raise SystemExit(
            f'{f}_ratio disagrees with mean({f}_ratio_c) at alpha={alphas[ai]}, seed={seeds[si]}: '
            f'{lhs[ai, si]!r} vs {rhs[ai, si]!r}. Two definitions of the calibration ratio are '
            f'live again — see config.cal_ratio.')

# Seed statistics. _std is the SAMPLE SD (ddof=1): cfg.seeds are a sample of the seed
# population, not the population, and the thesis sheet uses STDEV.S. Was ddof=0 before 5.44.
mean = {k: acc[k].mean(1) for k in keys}                  # seed mean per alpha
sd = {k: acc[k].std(1, ddof=1) for k in keys}             # seed sample SD per alpha
meanc = {k: accc[k].mean(1) for k in ckeys}               # (A,3) seed mean, per component
sdc = {k: accc[k].std(1, ddof=1) for k in ckeys}

jensen_norm, crosscov_err = mean['jensen'], mean['crosscov']
rmse_en, rmse_pf, rmse_qr = mean['rmse_en'], mean['rmse_pf'], mean['rmse_qr']
excess_pct, excess_qr_pct = mean['excess'], mean['excess_qr']
en_mult, pf_mult = mean['en_mult'], mean['pf_mult']
en_ratio, pf_ratio = mean['en_ratio'], mean['pf_ratio']

tag = f'w{cfg.obs_every}_{cfg.noise_mode}_{cfg.jitter_mode}'
np.savez(f'data/stage2_results_{tag}.npz',
         alphas=alphas, seeds=np.array(seeds),
         rmse_convention='per-component then averaged (5.16)',
         cal_ratio_convention=CAL_RATIO_CONVENTION,
         baseline_alpha=PERTURB_BASELINE_ALPHA,
         jensen_norm=jensen_norm, crosscov_err=crosscov_err,
         jensen_qr_norm=mean['jensen_qr'], crosscov_qr_err=mean['crosscov_qr'],
         jensen_ie_norm=mean['jensen_ie'], crosscov_ie_err=mean['crosscov_ie'],
         rmse_en=rmse_en, rmse_pf=rmse_pf, rmse_qr=rmse_qr, rmse_ie=mean['rmse_ie'],
         excess_pct=excess_pct, excess_qr_pct=excess_qr_pct, excess_ie_pct=mean['excess_ie'],
         en_mult=en_mult, pf_mult=pf_mult, qr_mult=mean['qr_mult'], ie_mult=mean['ie_mult'],
         en_ratio=en_ratio, pf_ratio=pf_ratio, qr_ratio=mean['qr_ratio'], ie_ratio=mean['ie_ratio'],
         en_spread=mean['en_spread'], pf_spread=mean['pf_spread'],
         qr_spread=mean['qr_spread'], ie_spread=mean['ie_spread'],
         # FORECAST variance (prior, pre-update). No pf_fcvar: run_pf returns no fc_spread.
         en_fcvar=mean['en_fcvar'], qr_fcvar=mean['qr_fcvar'], ie_fcvar=mean['ie_fcvar'],
         en_fcvar_std=sd['en_fcvar'], qr_fcvar_std=sd['qr_fcvar'], ie_fcvar_std=sd['ie_fcvar'],
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
         qr_spread_std=sd['qr_spread'], ie_spread_std=sd['ie_spread'],
         # per-component arrays (A,3): every headline above is the row mean of one of these
         **{k: meanc[k] for k in ckeys},
         **{k + '_std': sdc[k] for k in ckeys},
         # every individual seed, (A,S) for headlines and (A,S,3) per component. The thesis
         # sheet quotes seed columns, and no aggregate can be un-averaged back into them.
         **{k + '_by_seed': acc[k] for k in keys},
         **{k + '_by_seed': accc[k] for k in ckeys})

print(f"seeds {seeds}  (mean +/- sample SD across seeds, ddof=1)")
print("RMSE/excess convention: per component, then averaged over x, y, z")
print(f"calibration ratio: {CAL_RATIO_CONVENTION}")
print(f"{'alpha':>6}{'jsEn':>8}{'jsQR':>8}{'jsIE':>8}{'xcEn':>9}{'xcQR':>9}{'xcIE':>9}"
      f"{'RMSE_En':>9}{'RMSE_QR':>9}{'RMSE_IE':>9}{'RMSE_PF':>9}{'exc%':>7}{'excQR%':>8}{'excIE%':>8}")
for ai, a in enumerate(alphas):
    print(f"{a:6.1f}{jensen_norm[ai]:8.3f}{mean['jensen_qr'][ai]:8.3f}{mean['jensen_ie'][ai]:8.3f}"
          f"{crosscov_err[ai]:9.2f}{mean['crosscov_qr'][ai]:9.2f}{mean['crosscov_ie'][ai]:9.2f}"
          f"{rmse_en[ai]:9.3f}{rmse_qr[ai]:9.3f}{mean['rmse_ie'][ai]:9.3f}{rmse_pf[ai]:9.3f}"
          f"{excess_pct[ai]:6.1f}%{excess_qr_pct[ai]:7.1f}%{mean['excess_ie'][ai]:7.1f}%")

# Calibration ratio and the jitter it was achieved at. The alpha=0 row is flagged because it
# is no longer a scan result: it is the window baseline, which is what stage1_results.py runs,
# so the two files must agree there to the last digit. If they do not, something upstream
# (truth recipe, obs draw, obs_std) has drifted between the two scripts.
print("\ncalibration ratio (spread/RMSE, mean over x, y, z) and jitter multiplier")
print(f"{'alpha':>6}{'EnKF':>9}{'x':>7}{'PF':>9}{'x':>7}{'QR':>9}{'x':>7}{'IEnKF':>9}{'x':>7}   jitter")
for ai, a in enumerate(alphas):
    src = 'BASELINE (= stage1_results.py)' if float(a) == PERTURB_BASELINE_ALPHA \
        or cfg.jitter_mode == 'fixed' else 'scanned'
    print(f"{a:6.1f}{en_ratio[ai]:9.3f}{en_mult[ai]:7.2f}{pf_ratio[ai]:9.3f}{pf_mult[ai]:7.2f}"
          f"{mean['qr_ratio'][ai]:9.3f}{mean['qr_mult'][ai]:7.2f}"
          f"{mean['ie_ratio'][ai]:9.3f}{mean['ie_mult'][ai]:7.2f}   {src}")


if np.isfinite(mean['en_fcvar']).any():
    print("\nforecast variance (prior, before the update) and contraction vs the EnKF")
    print(f"{'alpha':>6}{'EnKF':>11}{'QR':>11}{'IEnKF':>11}{'QR fall %':>11}{'IE fall %':>11}")
    for ai, a in enumerate(alphas):
        e, q, i_ = mean['en_fcvar'][ai], mean['qr_fcvar'][ai], mean['ie_fcvar'][ai]
        with np.errstate(divide='ignore', invalid='ignore'):
            print(f"{a:6.1f}{e:11.4f}{q:11.4f}{i_:11.4f}"
                  f"{(e - q) / e * 100:11.1f}{(e - i_) / e * 100:11.1f}")

print("\nper-seed EnKF calibration ratio (what the thesis sheet quotes)")
print(f"{'alpha':>6}" + ''.join(f"{'seed ' + str(s):>10}" for s in seeds) + f"{'mean':>10}{'sd':>9}")
for ai, a in enumerate(alphas):
    print(f"{a:6.1f}" + ''.join(f"{v:10.4f}" for v in acc['en_ratio'][ai])
          + f"{en_ratio[ai]:10.4f}{sd['en_ratio'][ai]:9.4f}")

print(f'\nsaved data/stage2_results_{tag}.npz')