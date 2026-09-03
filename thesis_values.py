

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.44 created — reads every thesis-quoted number out of the persisted npz files and writes
#      them into the sheet layout used for the write-up: seed columns plus live =AVERAGE and
#      =STDEV.S. Exists because the numbers were being transcribed from terminal output by
#      hand, which is where the three different alpha=0 calibration ratios went unnoticed.
#      Reads the *_by_seed arrays added to error_sweep.py, stage2_hbar_diagnostic.py and
#      stage1_results.py in the same patch — no aggregate can be un-averaged back into seed
#      columns, so those arrays are what makes this script possible.
# ============================================================

import os
import re
import glob
import numpy as np
from config import cfg
from rls_core import rmse_percomp          # the one per-component RMSE for stage 4/5 arrays

try:
    from openpyxl import Workbook
    from openpyxl.styles import Font, Alignment, PatternFill
    HAVE_XLSX = True
except ModuleNotFoundError:                # CSV fallback below; the numbers are unaffected
    HAVE_XLSX = False

W, MODE, JIT = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
LAB = ['x', 'y', 'z']

STAGE1 = f'data/stage1_results_w{W}.npz'
SWEEP = f'data/stage2_results_w{W}_{MODE}_{JIT}.npz'
HBAR = f'data/stage2_hbar_w{W}_{MODE}_{JIT}.npz'
OUT = 'thesis_values.xlsx'

# stage4_results.py has annotated its trace figure with the correlation over t <= 150 since
# patch 5.9; before that it used the whole run. Set this to 150.0 if the thesis quotes the
# number printed on the current figure. None = whole run, which is what the values already
# in the hand-built sheet are.
TRACE_TMAX = None
TRACE_NOTE = 'whole run' if TRACE_TMAX is None else f't <= {TRACE_TMAX:g}'

# Alpha threshold for the "once alpha >= ..." rows, as used in the write-up.
ALPHA_HI = 0.3
BLIND_HI = 0.6


# ---------------------------------------------------------------------------------------
# row results
# ---------------------------------------------------------------------------------------
def ok(values, unit, source, how, seeds=None):

    v = np.asarray(values, dtype=float)
    return dict(values=v, unit=unit, source=source, how=how,
                seeds=list(seeds) if seeds is not None else None)


def missing(what, how=''):

    return dict(values=None, unit='', source=f'MISSING: {what}', how=how, seeds=None)


_cache = {}


def load(path):
    if path not in _cache:
        _cache[path] = np.load(path, allow_pickle=True) if os.path.exists(path) else None
    return _cache[path]


def need(path):
    d = load(path)
    if d is None:
        raise FileNotFoundError(path)
    return d


def alpha_index(d, a):

    al = np.asarray(d['alphas'], dtype=float)
    i = int(np.argmin(np.abs(al - a)))
    if abs(float(al[i]) - a) > 1e-9:
        raise KeyError(f'alpha={a:g} is not on the grid {np.array2string(al, precision=2)}')
    return i


def by_seed(path, *keys):

    d = need(path)
    arrs = []
    for k in keys:
        kk = f'{k}_by_seed'
        if kk not in d.files:
            raise KeyError(f'{path}:{kk} (rerun the script that writes it — pre-5.44 files '
                           f'have no per-seed arrays)')
        arrs.append(np.asarray(d[kk], dtype=float))
    return d, arrs


# ---------------------------------------------------------------------------------------
# section 4.1 — stage 1, linear h. This is also the alpha=0 column of the sweep since 5.44.
# ---------------------------------------------------------------------------------------
def S1(key, unit, what):
    def f():
        d = need(STAGE1)
        k = f'{key}_by_seed'
        if k not in d.files:
            return missing(f'{STAGE1}:{k}')
        return ok(d[k], unit, f'{STAGE1}:{k}',
                  f'{what}; linear h, baseline jitter, per seed', list(d['seeds']))
    return f


# ---------------------------------------------------------------------------------------
# sections 4.2 / 4.3 — the alpha sweep
# ---------------------------------------------------------------------------------------
def SW(key, alpha, unit, what):
    def f():
        d, (v,) = by_seed(SWEEP, key)
        i = alpha_index(d, alpha)
        return ok(v[i], unit, f'{SWEEP}:{key}_by_seed[alpha={alpha:g}]',
                  f'{what} at alpha={alpha:g}, per seed', list(d['seeds']))
    return f


def HB(key, alpha, unit, what):
    def f():
        d, (v,) = by_seed(HBAR, key)
        i = alpha_index(d, alpha)
        return ok(v[i], unit, f'{HBAR}:{key}_by_seed[alpha={alpha:g}]',
                  f'{what} at alpha={alpha:g}, per seed', list(d['seeds']))
    return f


def SW_MEAN_GE(key, thr, unit, what):

    def f():
        d, (v,) = by_seed(SWEEP, key)
        m = np.asarray(d['alphas'], dtype=float) >= thr
        return ok(np.nanmean(v[m], axis=0), unit, f'{SWEEP}:{key}_by_seed',
                  f'{what}: mean over alpha >= {thr:g} within each seed, then the seed columns',
                  list(d['seeds']))
    return f


def SW_DIFF(key_a, key_b, alpha, unit, what):
    def f():
        d, (va, vb) = by_seed(SWEEP, key_a, key_b)
        i = alpha_index(d, alpha)
        return ok(va[i] - vb[i], unit,
                  f'{SWEEP}:{key_a}_by_seed - {key_b}_by_seed [alpha={alpha:g}]',
                  f'{what} at alpha={alpha:g}, formed inside each seed', list(d['seeds']))
    return f


def SW_DIFF_ARGMAX(key_a, key_b, unit, what):

    def f():
        d, (va, vb) = by_seed(SWEEP, key_a, key_b)
        diff = va - vb
        i = int(np.nanargmax(np.nanmean(diff, axis=1)))
        a = float(d['alphas'][i])
        return ok(diff[i], unit, f'{SWEEP}:{key_a}_by_seed - {key_b}_by_seed [alpha={a:g}]',
                  f'{what}; alpha={a:g} is the argmax of the seed-mean curve, and the seed '
                  f'columns are the per-seed values there', list(d['seeds']))
    return f


def SW_COUNT_ABOVE(key_a, key_b, label_a, label_b):

    def f():
        d, (va, vb) = by_seed(SWEEP, key_a, key_b)
        al = np.asarray(d['alphas'], dtype=float)
        seeds = list(d['seeds'])
        with np.errstate(invalid='ignore'):
            above = va > vb                              # (A, S) boolean; NaN -> False
        bad = ~(np.isfinite(va) & np.isfinite(vb))       # (A, S) either side non-finite
        counts = above.sum(axis=0).astype(float)         # (S,)

        # the same count read off the seed-mean curves — the figure's version
        mc = np.nanmean(va, axis=1) > np.nanmean(vb, axis=1)
        where = '; '.join(
            f'seed {s}: ' + (', '.join(f'{a:g}' for a in al[above[:, i]]) or 'none')
            for i, s in enumerate(seeds))
        note = (f'count of alpha where {label_a} > {label_b}, within each seed. '
                f'On the seed-MEAN curves, which is what the figure plots, the count is '
                f'{int(mc.sum())} of {len(al)}, at alpha = '
                f'{", ".join(f"{a:g}" for a in al[mc]) or "none"}. Per seed: {where}')
        if bad.any():
            note += (f'. {int(bad.sum())} of {bad.size} (alpha, seed) points were non-finite '
                     f'and are counted as NOT above')
        return ok(counts, f'alphas (of {len(al)})',
                  f'{SWEEP}:{key_a}_by_seed > {key_b}_by_seed', note, seeds)
    return f


def SW_DIFF_MEAN(key_a, key_b, unit, what):
    def f():
        d, (va, vb) = by_seed(SWEEP, key_a, key_b)
        return ok(np.nanmean(va - vb, axis=0), unit,
                  f'{SWEEP}:{key_a}_by_seed - {key_b}_by_seed',
                  f'{what}: mean over the whole alpha grid within each seed', list(d['seeds']))
    return f


def SW_FALL_FROM_BASELINE(key, thr, unit, what):

    def f():
        d, (v,) = by_seed(SWEEP, key)
        al = np.asarray(d['alphas'], dtype=float)
        i0 = alpha_index(d, 0.0)
        m = al >= thr
        with np.errstate(divide='ignore', invalid='ignore'):
            fall = (v[i0] - np.nanmean(v[m], axis=0)) / v[i0] * 100
        return ok(fall, unit, f'{SWEEP}:{key}_by_seed',
                  f'{what}: fall from the alpha=0 value to the mean over alpha >= {thr:g}, '
                  f'within each seed, as a percentage of the alpha=0 value',
                  list(d['seeds']))
    return f


def SW_REDUCTION(key_remedy, key_enkf, unit, what, thr=None):

    def f():
        d, (vr, ve) = by_seed(SWEEP, key_remedy, key_enkf)
        al = np.asarray(d['alphas'], dtype=float)
        m = al > 0 if thr is None else al >= thr
        with np.errstate(divide='ignore', invalid='ignore'):
            red = (ve - vr) / ve * 100
        scope = ('mean over alpha > 0 within each seed (alpha=0 excluded, the bias is '
                 'identically zero there)' if thr is None else
                 f'mean over alpha >= {thr:g} within each seed')
        return ok(np.nanmean(red[m], axis=0), unit,
                  f'{SWEEP}:({key_enkf}_by_seed - {key_remedy}_by_seed)/{key_enkf}_by_seed',
                  f'{what}: reduction vs the standard EnKF at the same alpha, {scope}',
                  list(d['seeds']))
    return f


# ---------------------------------------------------------------------------------------
# section 4.3.3 — the IEnKF wrong-lobe diagnostic
# ---------------------------------------------------------------------------------------
IENKF3_ALPHA = 0.5          # the alpha Fig. 4.6 is drawn at
IENKF3 = f'data/stage3_ienkf_seeds_a{IENKF3_ALPHA:g}_w{W}_{MODE}.npz'


def IE3(key, unit, what):

    def f():
        d = load(IENKF3)
        if d is None:
            return missing(f'{IENKF3} (run stage3_ienkf_seeds.py)')
        k = f'{key}_by_seed'
        if k not in d.files:
            return missing(f'{IENKF3}:{k} (rerun stage3_ienkf_seeds.py — pre-5.47 runs '
                           f'saved nothing at all)')
        return ok(d[k], unit, f'{IENKF3}:{k}',
                  f'{what} at alpha={IENKF3_ALPHA:g}, per seed, own observations per seed',
                  list(np.asarray(d['seeds'], dtype=int)))
    return f


# ---------------------------------------------------------------------------------------
# section 4.3.1 — QR extrapolation frequency vs ensemble size
# ---------------------------------------------------------------------------------------
def EXT(N, alpha=0.1):

    def f():
        pat = f'data/stage3_qr_extrapolation_a{alpha:g}_w{W}_{MODE}.npz'
        d = load(pat)
        if d is None:
            return missing(f'{pat} (run stage3_qr_extrapolation.py)')
        sizes = list(np.asarray(d['ensemble_sizes'], dtype=int))
        if N not in sizes:
            return missing(f'{pat}: N_e={N} not in {sizes}')
        ki = sizes.index(N)
        return ok(d['frac_any'][ki], '% of cycles', f'{pat}:frac_any[N_e={N}]',
                  f'share of cycles where at least one component of the observation fell '
                  f'outside the min-max envelope of h over forecast members, N_e={N}, '
                  f'alpha={alpha:g}, per run', list(np.asarray(d['seeds'], dtype=int)))
    return f


# ---------------------------------------------------------------------------------------
# section 4.4 — stage 4 RLS modes
# ---------------------------------------------------------------------------------------
def s4_paths(mode):
    found = []
    for p in sorted(glob.glob(f'data/stage4_{mode}_a*.npz')):
        m = re.match(rf'stage4_{re.escape(mode)}_a([\d.]+)\.npz$',
                     p.replace('\\', '/').split('/')[-1])
        if not m:
            continue
        d = np.load(p, allow_pickle=True)
        if 'pred' not in d.files:            # not a result file (schema guard)
            continue
        found.append((float(m.group(1)), p))
    return sorted(found)


def s4_at(mode, alpha):
    for a, p in s4_paths(mode):
        if abs(a - alpha) < 1e-9:
            return p
    return None


# Which script writes each mode's result file, so a MISSING row says what to run.
S4_WRITER = {'shadow': 'rls_shadow.py', 'inject': 'rls_inject.py',
             'blind_shadow': 'rls_blind.py', 'blind_inject': 'rls_blind.py'}


def s4_name(mode, alpha):
    return f'data/stage4_{mode}_a{float(alpha)}.npz'


def s4_groups(d):

    if 'row_seed' not in d.files:
        return [(None, np.arange(len(d['times'])))]
    rs = np.asarray(d['row_seed'])
    return [(int(s), np.where(rs == s)[0]) for s in np.unique(rs)]


def s4_excess(d, idx):

    r_b = rmse_percomp(d['xa_base'][idx], d['truth'][idx])
    r_c = rmse_percomp(d['xa_corr'][idx], d['truth'][idx])
    r_p = rmse_percomp(d['xpf_mean'][idx], d['truth'][idx])
    return (r_b - r_p) / r_p * 100, (r_c - r_p) / r_p * 100


def S4_CORR(mode, alpha, comp):
    def f():
        p = s4_at(mode, alpha)
        if p is None:
            return missing(f'{s4_name(mode, alpha)} (run {S4_WRITER.get(mode, "the stage 4 script")})')
        d = np.load(p, allow_pickle=True)
        t = d['times']
        vals, seeds = [], []
        for sd_, idx in s4_groups(d):
            sel = idx if TRACE_TMAX is None else idx[t[idx] <= TRACE_TMAX]
            if len(sel) < 2:
                return missing(f'{p}: fewer than 2 cycles in the {TRACE_NOTE} window')
            vals.append(np.corrcoef(d['target'][sel, comp], d['pred'][sel, comp])[0, 1])
            seeds.append(sd_)
        return ok(vals, '', f'{p}:target,pred',
                  f'corr(true residual, learnt prediction) in {LAB[comp]}, {TRACE_NOTE}, '
                  f'per seed', seeds)
    return f


def S4_REMOVED(mode, alpha, relative):

    def f():
        p = s4_at(mode, alpha)
        if p is None:
            return missing(f'{s4_name(mode, alpha)} (run {S4_WRITER.get(mode, "the stage 4 script")})')
        d = np.load(p, allow_pickle=True)
        vals, seeds = [], []
        for sd_, idx in s4_groups(d):
            eb, ec = s4_excess(d, idx)
            vals.append((eb - ec) / eb * 100 if relative else eb - ec)
            seeds.append(sd_)
        unit = '% of baseline excess' if relative else 'pp'
        how = ('(baseline excess - corrected excess) / baseline excess, per seed'
               if relative else 'baseline excess over PF minus corrected excess over PF, per seed')
        return ok(vals, unit, f'{p}:xa_base,xa_corr,xpf_mean,truth',
                  f'{how}; RMSE per component then averaged (rls_core.rmse_percomp)', seeds)
    return f


def S4_REMOVED_AGG(mode, thr, agg, agg_name, relative):

    def f():
        paths = [(a, p) for a, p in s4_paths(mode) if a >= thr - 1e-9]
        if not paths:
            return missing(f'data/stage4_{mode}_a*.npz with alpha >= {thr:g}')
        per_seed, alphas_used = {}, []
        for a, p in paths:
            d = np.load(p, allow_pickle=True)
            alphas_used.append(a)
            for sd_, idx in s4_groups(d):
                eb, ec = s4_excess(d, idx)
                v = (eb - ec) / eb * 100 if relative else eb - ec
                per_seed.setdefault(sd_, []).append((v, a))     # keep the alpha with the value
        seeds = sorted(per_seed, key=lambda s: (s is None, s))
        picks = [agg(per_seed[s], key=lambda va: va[0]) for s in seeds]   # (value, alpha) per seed
        unit = '% of baseline excess' if relative else 'pp'
        where = ', '.join(f'seed {s} at alpha={a:g}' for s, (_, a) in zip(seeds, picks))
        return ok([v for v, _ in picks], unit,
                  f'data/stage4_{mode}_a{{{",".join(f"{a:g}" for a in alphas_used)}}}.npz',
                  f'{agg_name} over alpha >= {thr:g} of the excess removed ({unit}), taken '
                  f'within each seed, so each column may come from a different alpha: {where}',
                  seeds)
    return f


# ---------------------------------------------------------------------------------------
# section 4.5 — Lorenz-96 PF. No per-seed source exists in the project.
# ---------------------------------------------------------------------------------------
def L96(what):
    def f():
        return missing(
            'no per-seed source. partfilt_l96.py\'s __main__ runs a single cfg.seed and '
            'saves data/l96_pf_ess.npz with no seed axis, so the three seed columns cannot '
            'be read from it. Add a cfg.seeds loop there, or fill this row by hand.',
            what)
    return f


# ---------------------------------------------------------------------------------------
# THE SHEET. Order and wording follow the hand-built workbook line for line; where a label
# was ambiguous about units, the unit column now says which one it is.
# ---------------------------------------------------------------------------------------
ROWS = [
    ('4.1', 'pf calibration ratio (alpha = 0)',
     S1('pf_ratio', 'spread/RMSE', 'PF calibration ratio')),
    ('4.1', 'EnKF calibration ratio (alpha = 0)',
     S1('en_ratio', 'spread/RMSE', 'EnKF calibration ratio')),
    ('4.1', 'EnKF excess RMSE (alpha = 0)',
     S1('excess', '%', 'EnKF RMSE excess over the PF')),

    ('4.2.1', 'EnKF Jensen bias (alpha = 1)',
     SW('jensen', 1.0, '', 'mean over cycles of ||E[h(x)] - h(E[x])||')),
    ('4.2.1', 'EnKF Cross Cov error (alpha = 0)',
     SW('crosscov', 0.0, '', 'mean over cycles of ||Cxh(EnKF) - Cxh(PF)||_F')),
    ('4.2.1', 'EnKF Cross Cov error (alpha = 1)',
     SW('crosscov', 1.0, '', 'mean over cycles of ||Cxh(EnKF) - Cxh(PF)||_F')),
    ('4.2.1', 'EnKF RMSE excess (alpha = 0)',
     SW('excess', 0.0, '%', 'EnKF RMSE excess over the PF')),
    ('4.2.1', 'EnKF RMSE excess (alpha = 0.8)',
     SW('excess', 0.8, '%', 'EnKF RMSE excess over the PF')),
    ('4.2.1', 'EnKF RMSE excess (alpha = 1)',
     SW('excess', 1.0, '%', 'EnKF RMSE excess over the PF')),

    ('4.2.2', 'EnKF vs wrong EnKF excess RMSE gap (alpha = 0.2)',
     HB('penalty', 0.2, 'pp', 'h(E[x]) RMSE penalty over the standard E[h(x)] EnKF')),

    ('4.2.3', 'EnKF calibration ratio (alpha = 0)',
     SW('en_ratio', 0.0, 'spread/RMSE', 'EnKF calibration ratio')),
    ('4.2.3', 'EnKF calibration ratio (alpha = 1)',
     SW('en_ratio', 1.0, 'spread/RMSE', 'EnKF calibration ratio')),
    ('4.2.3', 'EnKF wrong update calibration ratio (alpha = 1)',
     HB('ratio_hb', 1.0, 'spread/RMSE', 'h(E[x])-anchored EnKF calibration ratio')),

    ('4.3.1', 'QR-EnKF excess removed - max over the sweep',
     SW_DIFF_ARGMAX('excess', 'excess_qr', 'pp',
                    'EnKF excess minus QR-EnKF excess')),
    ('4.3.1', 'QR-EnKF excess removed vs EnKF - mean over the a sweep',
     SW_DIFF_MEAN('excess', 'excess_qr', 'pp', 'EnKF excess minus QR-EnKF excess')),
    ('4.3.1', 'QR-EnKF Jensen bias reduction as % of EnKF',
     SW_REDUCTION('jensen_qr', 'jensen', '%', 'QR-EnKF Jensen bias')),
    ('4.3.1', 'QR-EnKF cross-cov error reduction as % of EnKF',
     SW_REDUCTION('crosscov_qr', 'crosscov', '%', 'QR-EnKF cross-covariance error')),
    ('4.3.1', 'extrapolation number Ne = 50 a=0.1 for all', EXT(50)),
    ('4.3.1', 'extrapolation number Ne = 100', EXT(100)),
    ('4.3.1', 'extrapolation number Ne = 200', EXT(200)),
    ('4.3.1', 'extrapolation number Ne = 500', EXT(500)),
    ('4.3.1', 'extrapolation number Ne = 1000', EXT(1000)),

    ('4.3.2', 'IEnKF Jensen bias reduction as % of EnKF',
     SW_REDUCTION('jensen_ie', 'jensen', '%', 'IEnKF Jensen bias')),
    ('4.3.2', 'IEnKF cross-cov error reduction as % of EnKF',
     SW_REDUCTION('crosscov_ie', 'crosscov', '%', 'IEnKF cross-covariance error')),
    ('4.3.2', f'IEnKF average calibration ratio (after a >= {ALPHA_HI:g})',
     SW_MEAN_GE('ie_ratio', ALPHA_HI, 'spread/RMSE', 'IEnKF calibration ratio')),
    ('4.3.2', f'EnKF calibration ratio at avg once a >= {ALPHA_HI:g}',
     SW_MEAN_GE('en_ratio', ALPHA_HI, 'spread/RMSE', 'EnKF calibration ratio')),
    # The contraction behind the IEnKF's apparent gain. delta = alpha * s^2 is formed on the
    # FORECAST ensemble, so a filter whose forecast variance has collapsed reports a smaller
    # Jensen bias without having handled the operator any better. Two readings of "falls by
    # N% over the same range" — against the EnKF at the same alpha, or against its own
    # alpha=0 value. Both are emitted; the sentence has to mean one of them.
    ('4.3.2', f'IEnKF forecast variance below EnKF (mean over a >= {ALPHA_HI:g})',
     SW_REDUCTION('ie_fcvar', 'en_fcvar', '%', 'IEnKF forecast variance', thr=ALPHA_HI)),
    ('4.3.2', f'IEnKF forecast variance fall from its own a=0 value (a >= {ALPHA_HI:g})',
     SW_FALL_FROM_BASELINE('ie_fcvar', ALPHA_HI, '%', 'IEnKF forecast variance')),

    ('4.3.2', 'IEnKF excess above EnKF - count of alpha tested',
     SW_COUNT_ABOVE('excess_ie', 'excess', 'IEnKF RMSE excess over PF',
                    'standard EnKF RMSE excess over PF')),
    ('4.3.2', 'IEnKF RMSE excess increased pp a = 0.5',
     SW_DIFF('excess_ie', 'excess', 0.5, 'pp',
             'IEnKF excess minus EnKF excess (positive = IEnKF worse)')),
    ('4.3.2', 'IEnKF RMSE excess reduced pp a = 0.8',
     SW_DIFF('excess', 'excess_ie', 0.8, 'pp',
             'EnKF excess minus IEnKF excess (positive = IEnKF better)')),

    # 4.3.3 — the wrong-lobe episodes. "reaches X - Y" is the typical level inside an
    # episode and the peak; the free-running median is what makes those two mean something.
    # "occasions per run" is a count of EPISODES; the cycle count is an order of magnitude
    # larger and is emitted beside it so the sentence can say which it means.
    ('4.3.3', 'IEnKF RMSE during lock-on - median',
     IE3('err_lock_med', '', 'median per-cycle RMS error inside the lock-on episodes')),
    ('4.3.3', 'IEnKF RMSE during lock-on - peak',
     IE3('err_lock_max', '', 'largest per-cycle RMS error inside the lock-on episodes')),
    ('4.3.3', 'IEnKF RMSE outside lock-on - median',
     IE3('err_free_med', '', 'median per-cycle RMS error outside the lock-on episodes')),
    ('4.3.3', 'IEnKF spread during lock-on - median',
     IE3('spread_lock_med', '', 'median per-cycle spread inside the lock-on episodes')),
    ('4.3.3', 'IEnKF spread outside lock-on - median',
     IE3('spread_free_med', '', 'median per-cycle spread outside the lock-on episodes')),
    ('4.3.3', 'straddle index > 0.4 - occasions per run',
     IE3('n_strad_episodes', 'episodes', 'contiguous episodes with straddle index above the '
         'threshold')),
    ('4.3.3', 'straddle index > 0.4 - cycles per run',
     IE3('n_strad_cycles', 'cycles', 'cycles with straddle index above the threshold')),
    ('4.3.3', 'lock-on - episodes per run',
     IE3('n_lock_episodes', 'episodes', 'sustained wrong-lobe episodes')),
    ('4.3.3', 'lock-on - % of the run spent locked on',
     IE3('frac_lock', '% of cycles', 'share of cycles inside a lock-on episode')),

    ('4.3.4', 'QR-EnKF excess removed - max over the sweep',
     SW_DIFF_ARGMAX('excess', 'excess_qr', 'pp', 'EnKF excess minus QR-EnKF excess')),

    ('4.4.1', 'shadow x corr at a = 0', S4_CORR('shadow', 0.0, 0)),
    ('4.4.1', 'shadow x corr at a = 0.8', S4_CORR('shadow', 0.8, 0)),
    ('4.4.1', 'shadow removal excess a = 0.8', S4_REMOVED('shadow', 0.8, relative=False)),
    ('4.4.1', 'shadow removal excess a = 0', S4_REMOVED('shadow', 0.0, relative=False)),
    ('4.4.1', 'shadow z corr at a = 0.8', S4_CORR('shadow', 0.8, 2)),

    ('4.4.3', f'blind shadow min removal a >= {BLIND_HI:g}',
     S4_REMOVED_AGG('blind_shadow', BLIND_HI, min, 'minimum', relative=False)),
    ('4.4.3', f'blind shadow max removal a >= {BLIND_HI:g}',
     S4_REMOVED_AGG('blind_shadow', BLIND_HI, max, 'maximum', relative=False)),
    ('4.4.3', 'blind shadow removal a = 0',
     S4_REMOVED('blind_shadow', 0.0, relative=False)),

    ('4.4.6', f'blind shadow max removal a >= {BLIND_HI:g}',
     S4_REMOVED_AGG('blind_shadow', BLIND_HI, max, 'maximum', relative=False)),
    ('4.4.6', 'QR-EnKF excess removed - max over the sweep',
     SW_DIFF_ARGMAX('excess', 'excess_qr', 'pp', 'EnKF excess minus QR-EnKF excess')),
    ('4.4.6', 'shadow x corr at a = 0.8', S4_CORR('shadow', 0.8, 0)),

    ('4.5', 'max min ess / N %', L96('maximum over the sweep of min(ESS)/N')),
    ('4.5', 'numbers of runs that sit cal ratio < 0.05',
     L96('count of (n, N) runs with calibration ratio below 0.05')),
    ('4.5', 'notable 0.71 calibration ratio', L96('the run quoted at ratio 0.71')),

    ('5.1', 'EnKF RMSE excess (alpha = 0.8)',
     SW('excess', 0.8, '%', 'EnKF RMSE excess over the PF')),
    ('5.1', 'EnKF vs wrong EnKF excess RMSE gap (alpha = 0.2)',
     HB('penalty', 0.2, 'pp', 'h(E[x]) RMSE penalty over the standard EnKF')),
    ('5.1', 'QR-EnKF excess removed - max over the sweep',
     SW_DIFF_ARGMAX('excess', 'excess_qr', 'pp', 'EnKF excess minus QR-EnKF excess')),
    ('5.1', f'blind shadow max removal a >= {BLIND_HI:g}',
     S4_REMOVED_AGG('blind_shadow', BLIND_HI, max, 'maximum', relative=False)),
    ('5.1', 'shadow x corr at a = 0', S4_CORR('shadow', 0.0, 0)),
    ('5.1', 'shadow x corr at a = 0.8', S4_CORR('shadow', 0.8, 0)),
]


# ---------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------
rows = []
for section, label, fn in ROWS:
    try:
        r = fn()
    except (FileNotFoundError, KeyError) as e:       # a missing file or key is a data gap,
        r = missing(str(e).strip("'"))               # not a reason to abandon the sheet
    except Exception as e:                           # anything else is a bug worth naming
        r = missing(f'{type(e).__name__}: {e}')
    r['section'], r['label'] = section, label
    rows.append(r)

filled = [r for r in rows if r['values'] is not None]
gaps = [r for r in rows if r['values'] is None]

# The seed block is exactly as wide as cfg.seeds — normally three columns, so mean and std
# land in H and I and the C:I block matches the hand-built sheet cell for cell. One
# experiment (the QR extrapolation sweep) runs its own nine seeds; those rows do NOT widen
# the sheet. They get a literal mean and SD instead, with every run listed in the notes
# column, because pushing H and I sideways would break every other row's layout.
SEED_LABELS = [str(s) for s in cfg.seeds]
N_SEED_COLS = len(SEED_LABELS)
odd = [r for r in filled if len(r['values']) != N_SEED_COLS]

print(f'window {W}, noise_mode {MODE}, jitter_mode {JIT}')
for p in (STAGE1, SWEEP, HBAR):
    print(f'  {"found  " if load(p) is not None else "MISSING"}  {p}')
print(f'\n{len(filled)} of {len(rows)} rows filled'
      + (f', {len(odd)} of them from a different seed set (literal mean/SD)' if odd else ''))
print()

hdr = ['section', 'numbers required']
hdr += [f'seed {s}' for s in SEED_LABELS]
hdr += ['mean', 'std', 'unit', 'seeds used', 'source', 'how it was computed']

FIRST_COL = 3                                        # column C, matching the hand-built sheet
FIRST_ROW = 3
SEED_COL = FIRST_COL + 2
MEAN_COL = SEED_COL + N_SEED_COLS
STD_COL = MEAN_COL + 1


def stats(v):
    return float(np.mean(v)), (float(np.std(v, ddof=1)) if len(v) > 1 else float('nan'))


def col_letter(i):
    s = ''
    while i:
        i, rem = divmod(i - 1, 26)
        s = chr(65 + rem) + s
    return s


if HAVE_XLSX:
    ARIAL = 'Arial'
    HEAD = Font(name=ARIAL, bold=True, color='FFFFFF')
    BODY = Font(name=ARIAL)
    GAPF = PatternFill('solid', fgColor='FDE9D9')
    HFILL = PatternFill('solid', fgColor='2471A3')

    wb = Workbook()
    ws = wb.active
    ws.title = 'values'

    for j, h in enumerate(hdr):
        c = ws.cell(FIRST_ROW, FIRST_COL + j, h)
        c.font = HEAD
        c.fill = HFILL
        c.alignment = Alignment(horizontal='center')

    for i, r in enumerate(rows):
        rr = FIRST_ROW + 1 + i
        ws.cell(rr, FIRST_COL, r['section']).font = BODY
        ws.cell(rr, FIRST_COL + 1, r['label']).font = BODY
        if r['values'] is None:
            for j in range(len(hdr)):
                ws.cell(rr, FIRST_COL + j).fill = GAPF
            ws.cell(rr, MEAN_COL, 'n/a').font = BODY
            ws.cell(rr, STD_COL, 'n/a').font = BODY
        elif len(r['values']) == N_SEED_COLS:
            for k, v in enumerate(r['values']):
                c = ws.cell(rr, SEED_COL + k, None if not np.isfinite(v) else float(v))
                c.font = BODY
                c.number_format = '0.000000'
            lo = col_letter(SEED_COL)
            hi = col_letter(SEED_COL + N_SEED_COLS - 1)
            ws.cell(rr, MEAN_COL, f'=AVERAGE({lo}{rr}:{hi}{rr})').font = BODY
            ws.cell(rr, STD_COL, f'=_xlfn.STDEV.S({lo}{rr}:{hi}{rr})').font = BODY
            ws.cell(rr, MEAN_COL).number_format = '0.000000'
            ws.cell(rr, STD_COL).number_format = '0.000000'
        else:
            # different seed set — literal values, so the seed columns stay meaningful
            m, s_ = stats(r['values'])
            ws.cell(rr, MEAN_COL, m).font = BODY
            ws.cell(rr, STD_COL, s_).font = BODY
            ws.cell(rr, MEAN_COL).number_format = '0.000000'
            ws.cell(rr, STD_COL).number_format = '0.000000'
            r = dict(r, how=(r['how'] + f'. {len(r["values"])} runs, not the {N_SEED_COLS} '
                             f'thesis seeds, so mean and SD are literal here: '
                             + ', '.join(f'{v:.4f}' for v in r['values'])))
        ws.cell(rr, STD_COL + 1, r['unit'] or '-').font = BODY
        ws.cell(rr, STD_COL + 2,
                '-' if r['seeds'] is None else ', '.join(str(s) for s in r['seeds'])).font = BODY
        ws.cell(rr, STD_COL + 3, r['source']).font = BODY
        ws.cell(rr, STD_COL + 4, r['how']).font = BODY

    ws.freeze_panes = ws.cell(FIRST_ROW + 1, SEED_COL)
    for j, wdt in enumerate([9, 46] + [14] * N_SEED_COLS + [14, 12, 20, 16, 58, 96]):
        ws.column_dimensions[col_letter(FIRST_COL + j)].width = wdt

    nt = wb.create_sheet('notes')
    notes = [
        ('Layout', 'Columns C:I match the hand-built sheet exactly, so that block can be '
                   'copied straight across. Everything from J rightwards is provenance.'),
        ('mean / std', 'Live formulas over the seed cells: AVERAGE and STDEV.S, the sample '
                       'standard deviation, matching the hand-built sheet.'),
        ('Order of averaging',
         'Where a row averages over alpha as well as over seeds, the ALPHA average is taken '
         'inside each seed first and the seed cells hold those numbers. The sheet then '
         'averages across seeds. Doing it the other way gives the same mean but a spread '
         'that means nothing.'),
        ('Calibration ratio',
         'Every ratio in this sheet is config.cal_ratio: the mean over x, y, z of '
         'spread_i / RMSE_i, formed inside one seed. Since patch 5.44 stage1_results.py, '
         'error_sweep.py and stage2_hbar_diagnostic.py all use that definition and all use '
         'the baseline jitter at alpha = 0, so their alpha = 0 numbers are the same run. '
         'The 4.1 and 4.2.3 alpha = 0 rows should therefore agree exactly; if they do not, '
         'one of the npz files predates the patch.'),
        ('Units', 'Excess figures are in PERCENT here (63.6 means 63.6%). Gaps between two '
                  'excesses, including every "removal" row, are in percentage POINTS: '
                  'baseline excess over the PF minus corrected excess over the PF. The unit '
                  'column says which, per row. To read a removal row as a share of the '
                  'baseline excess instead, pass relative=True at its call site in ROWS — '
                  'but do not mix the two units inside one table.'),
        ('Correlation window',
         f'Stage 4 correlations are computed over the {TRACE_NOTE}. stage4_results.py has '
         f'annotated its figure with the t <= 150 value since patch 5.9, so if the thesis '
         f'quotes the number printed on the current figure, set TRACE_TMAX = 150.0 at the '
         f'top of this script and rerun.'),
        ('Section 4.5', 'Left empty on purpose. partfilt_l96.py runs one seed in its '
                        '__main__ and l96_pf_ess.npz has no seed axis, so there is nothing '
                        'per-seed to read. Add a cfg.seeds loop there to fill these rows.'),
    ]
    nt['A1'] = 'thesis_values.py — how to read this workbook'
    nt['A1'].font = Font(name=ARIAL, bold=True)
    for i, (k, v) in enumerate(notes, start=3):
        nt.cell(i, 1, k).font = Font(name=ARIAL, bold=True)
        nt.cell(i, 2, v).alignment = Alignment(wrap_text=True, vertical='top')
    nt.column_dimensions['A'].width = 24
    nt.column_dimensions['B'].width = 120
    nt['A' + str(len(notes) + 5)] = (
        f'sources: {STAGE1} | {SWEEP} | {HBAR} | '
        f'data/stage3_qr_extrapolation_a0.1_w{W}_{MODE}.npz | data/stage4_{{mode}}_a{{alpha}}.npz')

    wb.save(OUT)
    print(f'saved {OUT}')
else:
    import csv
    with open('thesis_values.csv', 'w', newline='', encoding='utf-8') as fh:
        w_ = csv.writer(fh)
        w_.writerow(hdr)
        for r in rows:
            v = list(r['values']) if r['values'] is not None else []
            how = r['how']
            if len(v) == N_SEED_COLS:
                cells, (mean, std) = v, stats(v)
            elif v:                                   # different seed set: no seed columns
                cells = [''] * N_SEED_COLS
                mean, std = stats(v)
                how += (f'. {len(v)} runs, not the {N_SEED_COLS} thesis seeds: '
                        + ', '.join(f'{x:.4f}' for x in v))
            else:
                cells, mean, std = [''] * N_SEED_COLS, 'n/a', 'n/a'
            w_.writerow([r['section'], r['label'], *cells, mean, std, r['unit'] or '-',
                         '-' if r['seeds'] is None else ' '.join(str(s) for s in r['seeds']),
                         r['source'], how])
    print('openpyxl not installed — wrote thesis_values.csv (no live formulas).')
    print('For the workbook with AVERAGE / STDEV.S formulas:  pip install openpyxl')

# ---- terminal view, so the numbers are checkable without opening the file ----
print(f'\n{"section":>8}  {"quantity":<50}{"mean":>12}{"sd":>11}  unit')
for r in rows:
    if r['values'] is None:
        print(f'{r["section"]:>8}  {r["label"]:<50}{"n/a":>12}{"":>11}  {r["source"]}')
        continue
    v = r['values']
    sd_ = np.std(v, ddof=1) if len(v) > 1 else float('nan')
    print(f'{r["section"]:>8}  {r["label"]:<50}{np.mean(v):12.4f}{sd_:11.4f}  {r["unit"]}')

if gaps:
    print(f'\n{len(gaps)} row(s) could not be filled:')
    for r in gaps:
        print(f'  {r["section"]} {r["label"]}\n      {r["source"]}')



COUNT_CHECKS = [
    ('IEnKF', 'excess_ie', 'EnKF', 'excess'),
]

for label_a, key_a, label_b, key_b in COUNT_CHECKS:
    print(f'\n{"=" * 78}\ncount check: is the {label_a} RMSE excess above the {label_b}\'s?\n'
          f'{"=" * 78}')
    try:
        d, (va, vb) = by_seed(SWEEP, key_a, key_b)
    except (FileNotFoundError, KeyError) as e:
        print(f'  cannot check: {str(e).strip(chr(39))}')
        continue
    al = np.asarray(d['alphas'], dtype=float)
    seeds = list(d['seeds'])
    ma, mb = np.nanmean(va, axis=1), np.nanmean(vb, axis=1)
    with np.errstate(invalid='ignore'):
        above = va > vb
    sd_margin = np.nanstd(va - vb, axis=1, ddof=1) if len(seeds) > 1 else np.zeros(len(al))

    print(f'{"alpha":>6}{label_a + " %":>12}{label_b + " %":>12}{"margin pp":>12}'
          f'{"sd pp":>9}  seeds above   verdict')
    for i, a in enumerate(al):
        marg = ma[i] - mb[i]
        n = int(above[i].sum())
        # "clear" only when the seed-mean margin is bigger than its own spread across seeds
        # AND every seed agrees; anything else is a crossing the sentence should not lean on
        if n == len(seeds) and marg > sd_margin[i]:
            verdict = 'above, clear'
        elif n == 0 and -marg > sd_margin[i]:
            verdict = 'below, clear'
        elif n in (0, len(seeds)):
            verdict = 'ALL SEEDS AGREE BUT WITHIN THE SEED SPREAD'
        else:
            verdict = f'SEEDS DISAGREE ({n}/{len(seeds)})'
        print(f'{a:6.1f}{ma[i]:12.1f}{mb[i]:12.1f}{marg:12.1f}{sd_margin[i]:9.1f}'
              f'{n:>8}/{len(seeds)}   {verdict}')

    per_seed = above.sum(axis=0)
    mean_curve = int((ma > mb).sum())
    print(f'\n  per seed        : ' + ', '.join(f'seed {s} -> {c} of {len(al)}'
                                                for s, c in zip(seeds, per_seed)))
    m_, s_ = stats(per_seed.astype(float))
    print(f'  mean +/- SD     : {m_:.2f} +/- {s_:.2f} of {len(al)}')
    print(f'  seed-mean curve : {mean_curve} of {len(al)}   <- the count a reader gets from '
          f'the figure, which plots the seed means')
    if m_ != mean_curve:
        print(f'  NOTE: the two counts differ.')