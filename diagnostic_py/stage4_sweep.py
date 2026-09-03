# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.49 Changed: RMSE comes from rls_core.rmse_percomp — per component, then averaged over
#      x, y, z 
# 4.31 Clarified: lambda is inert at test time in the blind modes (frozen weights are never
#      updated) but still changes the result via training — heatmap axes now say so, and the
#      inert lam passed to the frozen run is flagged. No behaviour change.
# 4.30 Changed: rewritten against the current pipeline. Adds blind_shadow and blind_inject
#      evaluation on the disjoint test seeds; mirrors the learners (state chained across
#      seeds, dynamic feature block, tail-mean frozen weights); reports relative excess
#      reduction against each population's own baseline alongside the rolling gap;
#      NaN-safe on diverged runs. The previous version predated the rename, the dynamic
#      block and the tail-mean default, and crashed on feature-vector size.
# 4.16 Changed: gamma grid re-spaced with fine resolution below 0.1.
# 4.15 created — combined lambda x gamma sweep with per-seed consistency reporting,
#      CSV + text output and (gamma x lambda) heatmaps.
# ============================================================

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from config import cfg
import rls_core
from rls_core import (load_log, n_features, RunningStandardiser, RLS, dyn_row,
                      tail_mean_weights, rmse_percomp)
from rls_inject import run_enkf_rls

OUT = 'figs/diagnostic/s4'
os.makedirs(OUT, exist_ok=True)

ALPHAS = [0.2, 0.4, 0.6, 0.8, 1.0]
LAMBDAS = [0.95, 0.97, 0.98, 0.99, 0.995, 1.0]
GAMMAS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
TRAIN_SEEDS = list(cfg.seeds)                                # models are fitted on these
TEST_SEEDS = [s + 100 for s in cfg.seeds]                    # blind modes scored on these
MODES = ['shadow', 'blind_shadow', 'inject', 'blind_inject']
#MODES = ['shadow', 'blind_shadow']                          # trim for a cheaper pass
#MODES = ['inject', 'blind_inject']
ROLL = 25                                                    # rolling window for the gap, cycles
P0 = 1e3
TAIL_FRAC = 0.5                                              # frozen weights = tail mean
USE_DELTA = False

# Every RMSE below is per component then averaged (rls_core.rmse_percomp, Stage 4/5
# convention since 5.18). Nothing in this file pools cycles and components into one mean.
CONVENTION = ('RMSE per component then averaged over x, y, z; the rolling gap is formed '
              'the same way, per component within the window and then averaged.')

tw = np.load('data/l63_twin.npz')
truth = tw['truth']; obs_idx = tw['obs_idx']
alphas_grid = tw['alphas']; truth_at_obs = truth[obs_idx]


def rolling_rmse(a, b, w):
    """Rolling RMSE over a centred window: per component, then averaged. (T,3) -> (T,)."""
    se = (np.asarray(a) - np.asarray(b)) ** 2
    n = len(se); out = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - w // 2), min(n, i + w // 2 + 1)
        out[i] = np.sqrt(np.nanmean(se[lo:hi], axis=0)).mean()
    return out


def evaluate(xa_base, xa_corr, xpf, tr):
    r_b, r_p = rmse_percomp(xa_base, tr), rmse_percomp(xpf, tr)
    r_c = rmse_percomp(xa_corr, tr, nan=True)        # the corrected run is the one that diverges
    eb = (r_b - r_p) / r_p * 100                     # baseline excess over PF, %
    ec = (r_c - r_p) / r_p * 100                     # corrected excess over PF, %
    R_pf = rolling_rmse(xpf, tr, ROLL)
    R_b = rolling_rmse(xa_base, tr, ROLL)
    R_c = rolling_rmse(xa_corr, tr, ROLL)
    gap = (R_b - R_c) / R_pf * 100                   # (T,) excess removed per cycle, pp
    fin = np.isfinite(gap)                           # diverged cycles are skipped, not scored as 0
    return dict(rmse_base=r_b, rmse_corr=r_c, rmse_pf=r_p,
                excess_base_pct=eb, excess_corr_pct=ec,
                rel_red_pct=((eb - ec) / eb * 100) if eb else 0.0,   # meaningless if eb < 0
                mean_gap_pct=float(np.nanmean(gap)),
                help_pct=float((gap[fin] > 0).mean() * 100) if fin.any() else float('nan'),
                contribution='+' if r_c < r_b else '-')


def train_shadow(paths, alpha, lam):
    first = load_log(paths[0][1], alpha, use_delta=USE_DELTA)
    K = n_features(first['U_raw'].shape[1])
    std = RunningStandardiser(first['U_raw'].shape[1] + rls_core.N_DYN)
    rls = RLS(K, lam=lam, P0=P0)
    preds, hist = {}, []
    for sd, p in paths:
        L = load_log(p, alpha, use_delta=USE_DELTA)
        T = L['U_raw'].shape[0]
        pred = np.zeros((T, 3)); lag_pred = np.zeros(3)
        for t in range(T):
            u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]))
            pred[t] = rls.predict(u)                 # predict with OLD weights
            rls.update(u, L['r'][t])                 # then learn
            lag_pred = pred[t]
            hist.append(rls.w.copy())
        preds[sd] = pred
    return preds, tail_mean_weights(np.array(hist), TAIL_FRAC), std


def apply_frozen(w, std, L):
    T = L['U_raw'].shape[0]
    pred = np.zeros((T, 3)); lag_pred = np.zeros(3)
    for t in range(T):
        u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]),
                          update=False)
        pred[t] = w.T @ u
        lag_pred = pred[t]
    return pred


def live_run(alpha, raw, ostd, gamma, lam, seed, w=None, std=None, P=None, freeze=False):
    h = lambda x: x + alpha * x ** 2
    base = cfg.perturb_std.copy()
    # jitter comes from the log, not from a scan here: this run is scored against that
    # log's xa_mean, so it has to be the same filter. stage4_log.py owns the calibration.
    if 'en_mult' in raw.files:
        cfg.perturb_std = base * float(raw['en_mult'])
    try:
        res = run_enkf_rls(raw['obs'], truth, obs_idx, h, raw['xpf_mean'], gamma=gamma,
                           lam=lam, P0=P0, use_delta=USE_DELTA, seed=seed, obs_std=ostd,
                           w_init=w, std_init=std, P_init=P, freeze=freeze)
    except Exception:
        cfg.perturb_std = base
        return None
    cfg.perturb_std = base
    return res


# ---- CSV written incrementally so an interrupted sweep keeps its results ----
FIELDS = ['mode', 'alpha', 'lambda', 'gamma', 'seed', 'rmse_base', 'rmse_corr', 'rmse_pf',
          'excess_base_pct', 'excess_corr_pct', 'rel_red_pct', 'mean_gap_pct', 'help_pct',
          'contribution', 'diverged']
per_seed_path = f'{OUT}/stage4_sweep_perseed.csv'
fh = open(per_seed_path, 'w', newline='')
writer = csv.DictWriter(fh, fieldnames=FIELDS)
writer.writeheader()
rows = []

live_modes = [m for m in MODES if 'inject' in m]
n_live = len(ALPHAS) * len(LAMBDAS) * len(GAMMAS) * len(TRAIN_SEEDS) * len(live_modes)
print(f'sweep: {len(ALPHAS)} alphas x {len(LAMBDAS)} lambdas x {len(GAMMAS)} gammas '
      f'x {len(TRAIN_SEEDS)} seeds, modes {MODES}')
print(f'  live filter runs: {n_live}  (shadow modes are post-hoc and effectively free)')
print(f'  convention: {CONVENTION}')
t_start = time.time(); done = 0


def emit(mode, a, lam, g, sd, met, diverged=0):
    """Write one per-seed row to CSV and the in-memory table."""
    row = dict(mode=mode, alpha=a, **{'lambda': lam}, gamma=g, seed=sd, diverged=diverged)
    if diverged:
        row.update({k: '' for k in FIELDS if k not in row}, contribution='DIV')
    else:
        row.update({k: round(v, 4) if isinstance(v, float) else v for k, v in met.items()})
    writer.writerow(row); rows.append(row)


for a in ALPHAS:
    ai = int(np.argmin(np.abs(alphas_grid - a)))
    ostd = tw['obs_std_alpha'][ai]
    pth = lambda sd: f'data/stage4_log_a{a}_s{sd}_{cfg.noise_mode}.npz'
    tr_paths = [(sd, pth(sd)) for sd in TRAIN_SEEDS if os.path.exists(pth(sd))]
    te_paths = [(sd, pth(sd)) for sd in TEST_SEEDS if os.path.exists(pth(sd))]
    if not tr_paths:
        print(f'alpha={a}: no training logs — run stage4_log.py first'); continue
    raw_tr = {sd: np.load(p) for sd, p in tr_paths}
    raw_te = {sd: np.load(p) for sd, p in te_paths}

    for lam in LAMBDAS:
        # ---- shadow family: one training pass serves every gamma ----
        if 'shadow' in MODES or 'blind_shadow' in MODES:
            preds, w_tail, std_s = train_shadow(tr_paths, a, lam)
            blind_preds = {sd: apply_frozen(w_tail, std_s, load_log(p, a, use_delta=USE_DELTA))
                           for sd, p in te_paths}
            for g in GAMMAS:
                if 'shadow' in MODES:
                    for sd, raw in raw_tr.items():
                        met = evaluate(raw['xa_mean'], raw['xa_mean'] + g * preds[sd],
                                       raw['xpf_mean'], truth_at_obs)
                        emit('shadow', a, lam, g, sd, met)
                if 'blind_shadow' in MODES:
                    for sd, raw in raw_te.items():
                        met = evaluate(raw['xa_mean'], raw['xa_mean'] + g * blind_preds[sd],
                                       raw['xpf_mean'], truth_at_obs)
                        emit('blind_shadow', a, lam, g, sd, met)
            fh.flush()

        # ---- inject family: a live run per gamma, then the blind test on those weights ----
        if 'inject' in MODES or 'blind_inject' in MODES:
            for g in GAMMAS:
                w_c = std_c = P_c = None
                hist = []
                for sd, raw in raw_tr.items():       # chain state across training seeds
                    res = live_run(a, raw, ostd, g, lam, sd, w=w_c, std=std_c, P=P_c)
                    if res is None or res['diverged_at'] is not None:
                        if 'inject' in MODES:
                            emit('inject', a, lam, g, sd, None, diverged=1)
                        if res is None:
                            continue
                    else:
                        if 'inject' in MODES:
                            met = evaluate(raw['xa_mean'], res['en_mean'], raw['xpf_mean'],
                                           truth_at_obs)
                            emit('inject', a, lam, g, sd, met)
                    w_c, std_c, P_c = res['rls'].w, res['std'], res['rls'].P
                    hist.append(res['w_hist'])
                    done += 1

                # blind_inject uses the weights THIS gamma produced — inject learns against a
                # target that moves with the correction, so its model is gamma-specific.
                if 'blind_inject' in MODES and hist:
                    try:
                        w_frozen = tail_mean_weights(np.concatenate(hist), TAIL_FRAC)
                    except ValueError:
                        w_frozen = None
                    for sd, raw in raw_te.items():
                        if w_frozen is None:
                            emit('blind_inject', a, lam, g, sd, None, diverged=1); continue
                        # lam is passed but INERT here: freeze=True never calls
                        # rls.update(), so the forgetting factor is never applied. It
                        # reached this configuration through the weights, during training.
                        res = live_run(a, raw, ostd, g, lam, sd, w=w_frozen, std=std_c,
                                       freeze=True)
                        if res is None or res['diverged_at'] is not None:
                            emit('blind_inject', a, lam, g, sd, None, diverged=1)
                        else:
                            met = evaluate(raw['xa_mean'], res['en_mean'], raw['xpf_mean'],
                                           truth_at_obs)
                            emit('blind_inject', a, lam, g, sd, met)
                        done += 1
                fh.flush()

        el = time.time() - t_start
        eta = el / max(done, 1) * (n_live - done) if done else 0
        print(f'  alpha={a} lambda={lam} done  [live {done}/{n_live}]  '
              f'elapsed {el/60:.1f}m  eta {eta/60:.1f}m')

fh.close()
print(f'\nsaved {per_seed_path}')

# --- seed-averaged table ---
key = lambda r: (r['mode'], r['alpha'], r['lambda'], r['gamma'])
AFIELDS = ['mode', 'alpha', 'lambda', 'gamma', 'n_seeds', 'n_helped', 'n_diverged',
           'rel_red_pct', 'mean_gap_pct', 'help_pct', 'excess_base_pct', 'excess_corr_pct']
averaged = []
for k in sorted({key(r) for r in rows}):
    grp = [r for r in rows if key(r) == k]
    ok = [r for r in grp if not r['diverged']]
    mean_of = lambda f: round(float(np.mean([r[f] for r in ok])), 2) if ok else ''
    averaged.append(dict(mode=k[0], alpha=k[1], **{'lambda': k[2]}, gamma=k[3],
                         n_seeds=len(grp),
                         n_helped=sum(r['contribution'] == '+' for r in grp),
                         n_diverged=sum(r['diverged'] for r in grp),
                         rel_red_pct=mean_of('rel_red_pct'),
                         mean_gap_pct=mean_of('mean_gap_pct'),
                         help_pct=mean_of('help_pct'),
                         excess_base_pct=mean_of('excess_base_pct'),
                         excess_corr_pct=mean_of('excess_corr_pct')))
avg_path = f'{OUT}/stage4_sweep_averaged.csv'
with open(avg_path, 'w', newline='') as f:
    w_ = csv.DictWriter(f, fieldnames=AFIELDS); w_.writeheader(); w_.writerows(averaged)
print(f'saved {avg_path}')

# --- formatted text report ---
txt_path = f'{OUT}/stage4_sweep_report.txt'
with open(txt_path, 'w') as f:
    f.write('Stage 4 sweep — alpha x lambda x gamma, per seed and seed-averaged\n')
    f.write(f'train seeds={TRAIN_SEEDS}  blind seeds={TEST_SEEDS}  rolling window={ROLL}\n')
    f.write(f'convention: {CONVENTION}\n')
    f.write('rel red   = relative RMSE-excess reduction against that run\'s own baseline\n')
    f.write('mean gap  = excess removed in percentage points (rolling)\n')
    f.write('helping % = fraction of finite cycles with a positive gap\n')
    f.write('=' * 100 + '\n\n')
    for k in sorted({key(r) for r in rows}):
        grp = [r for r in rows if key(r) == k]
        mode, a, lam, g = k
        for r in grp:
            if r['diverged']:
                f.write(f"seed {r['seed']} gamma {g} lambda {lam} alpha={a} [{mode}]: DIVERGED\n")
            else:
                f.write(f"seed {r['seed']} gamma {g} lambda {lam} alpha={a} [{mode}]: "
                        f"rel red {r['rel_red_pct']:+.1f}% | mean gap {r['mean_gap_pct']:+.1f} pp | "
                        f"helping in {r['help_pct']:.0f}% of cycles | "
                        f"contribution {r['contribution']}\n")
        A = next(x for x in averaged if (x['mode'], x['alpha'], x['lambda'], x['gamma']) == k)
        rr = f"{A['rel_red_pct']:+.1f}%" if A['rel_red_pct'] != '' else 'n/a'
        mg = f"{A['mean_gap_pct']:+.1f} pp" if A['mean_gap_pct'] != '' else 'n/a'
        hp = f"{A['help_pct']:.0f}%" if A['help_pct'] != '' else 'n/a'
        f.write(f"  --> gamma {g} lambda {lam} alpha={a} [{mode}]: helped "
                f"{A['n_helped']}/{A['n_seeds']} seeds | rel red {rr} | mean gap {mg} | "
                f"helping in {hp} cycles"
                + (f" | {A['n_diverged']} diverged" if A['n_diverged'] else '') + '\n\n')
print(f'saved {txt_path}')

# --- heatmaps: (gamma x lambda) per mode and alpha ---
for mode in MODES:
    n_pop = len(TEST_SEEDS) if mode.startswith('blind') else len(TRAIN_SEEDS)
    for a in ALPHAS:
        sub = [x for x in averaged if x['mode'] == mode and x['alpha'] == a]
        if not sub:
            continue
        R = np.full((len(GAMMAS), len(LAMBDAS)), np.nan)   # relative reduction
        S = np.full((len(GAMMAS), len(LAMBDAS)), np.nan)   # seeds helped
        for x in sub:
            i, j = GAMMAS.index(x['gamma']), LAMBDAS.index(x['lambda'])
            if x['rel_red_pct'] != '':
                R[i, j] = x['rel_red_pct']
            S[i, j] = x['n_helped']
        for M, name, cmapn, lab in [
                (R, 'relred', 'RdBu_r', 'relative excess reduction (%)  + = helping'),
                (S, 'seeds', 'viridis', f'seeds helped (of {n_pop})')]:
            fig, ax = plt.subplots(figsize=(7.5, 6))
            if name == 'relred' and np.isfinite(M).any():
                v = np.nanmax(np.abs(M)); vmin, vmax = -v, v      # symmetric about 0
            else:
                vmin, vmax = 0, n_pop
            im = ax.imshow(M, aspect='auto', origin='lower', cmap=cmapn, vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(LAMBDAS))); ax.set_xticklabels(LAMBDAS, fontsize=8)
            ax.set_yticks(range(len(GAMMAS))); ax.set_yticklabels(GAMMAS, fontsize=8)
            # For the blind modes lambda acted during TRAINING only — at test time the
            # weights are frozen and never updated, so lambda is inert there. It still
            # changes the result, because it changed which weights got frozen.
            lam_lab = (r'$\lambda$ used in TRAINING' if mode.startswith('blind')
                       else r'$\lambda$ (forgetting)')
            gam_lab = (r'$\gamma$ (train + test)' if mode == 'blind_inject'
                       else r'$\gamma$ (injection)')
            ax.set_xlabel(lam_lab); ax.set_ylabel(gam_lab)
            for i in range(len(GAMMAS)):
                for j in range(len(LAMBDAS)):
                    txt = 'DIV' if np.isnan(M[i, j]) else f'{M[i, j]:.0f}'
                    ax.text(j, i, txt, ha='center', va='center', fontsize=6.5, color='k')
            fig.colorbar(im, ax=ax, label=lab)
            ax.set_title(rf'{mode} — $\alpha$={a}:  {lab}')
            fig.tight_layout()
            fig.savefig(f'{OUT}/stage4_sweep_{name}_{mode}_a{a}.png', dpi=140,
                        bbox_inches='tight')
            plt.close(fig)
print(f'saved heatmaps to {OUT}')

# --- recommended configuration ---
print('\nbest configuration by relative excess reduction (ties -> seeds helped):')
for mode in MODES:
    for a in ALPHAS:
        sub = [x for x in averaged if x['mode'] == mode and x['alpha'] == a
               and x['rel_red_pct'] != '']
        if not sub:
            continue
        b = max(sub, key=lambda x: (x['rel_red_pct'], x['n_helped']))
        print(f"  [{mode:>13}] alpha={a}: gamma={b['gamma']:<5} lambda={b['lambda']:<6} -> "
              f"rel red {b['rel_red_pct']:+6.1f}% | {b['n_helped']}/{b['n_seeds']} seeds | "
              f"helping in {b['help_pct']:.0f}% cycles")

# The single recommendation is taken from the BLIND modes only: those are the ones scored on
# seeds the weights never saw, so they are the only ones that say anything about deployment.
for blind_mode in [m for m in MODES if m.startswith('blind')]:
    sub = [x for x in averaged if x['mode'] == blind_mode and x['rel_red_pct'] != '']
    if not sub:
        continue
    best = {}
    for x in sub:                                   # average over alpha per (lambda, gamma)
        best.setdefault((x['lambda'], x['gamma']), []).append(x['rel_red_pct'])
    lam_g, sc = max(((k, float(np.mean(v))) for k, v in best.items()), key=lambda kv: kv[1])
    print(f'\nRECOMMENDED for {blind_mode} (mean over alpha): '
          f'lambda={lam_g[0]}, gamma={lam_g[1]}  -> {sc:+.1f}% mean relative reduction')