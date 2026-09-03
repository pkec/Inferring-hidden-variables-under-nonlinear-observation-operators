

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.34 Changed: sizing and fonts come from figstyle. The 7.5in and 14in exports shrank by
#      0.43 and 0.48 at their include widths, taking 14pt labels to ~6-7pt.
#      Changed: the merged 1x2 is now the write-up figure, with (a)/(b) drawn in — one
#      caption instead of two subcaptions, and wider panels.
#      Changed: legend labels drop the seed lists ('baseline, train [100, 101, 102]' ->
#      'train, uncorrected'). They did not fit a 0.48 panel and belong in the caption; the
#      populations are already printed to stdout.
# 5.10 Changed: report styling to match Stage 1/2 — axis labels Arial 14pt, legends Arial 11pt
#      (AX/FS/LEG), all titles and suptitles dropped. The per-seed trace titles carried
#      "RMSE base -> blind (PF)"; those numbers are already printed per seed above, so nothing
#      is lost from the record.
#      Added: both multi-panel figures also emit one PNG per panel — the generalisation figure
#      as _own_baseline / _per_seed, the blind trace as one PNG per seed. Combined versions
#      stay as backups.
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching error_sweep.py 5.16. This one is NOT print-only: r_b, r_c and r_p
#      feed exc_b/exc_c, the generalisation figures and the saved blind_* result files, so
#      every blind number in section 4.4 moves with this change.
# 4.40 Added: training population is cfg.blind_seeds (env L63_BLIND_SEEDS), so the number of
#      training trajectories is selectable — '100' for one, '100,101,102' for three. Weight
#      files carry that count (stage4_weights_blind_{source}_n{k}_a{alpha}.npz) so models
#      trained on different numbers of seeds no longer overwrite each other.
# 4.39 Changed: train and test populations swapped — the frozen weights are fitted on the
#      +100 runs and the blind test is scored on cfg.seeds, the same runs shadow and inject
#      report on, so all four modes share one baseline.
#      Added:   guard rejecting overlapping train/test seeds (the test would not be blind).
#      Removed: unused SEEDS.
#      Fixed:   stale docstring line saying the weights are loaded from rls_shadow.py's saved
#               model — 4.32 made this script train its own.
# 4.37 Changed: train_inject moved to rls_inject.train_inject_chain and imported, so the
#      trajectory diagnostic can build the same frozen model without importing this script.
# 4.33 Changed: offline parameters set from the sweep — shadow (lambda 1.0, gamma 0.7),
#      inject (lambda 0.995, gamma 0.05). Both differ from their online counterparts.
# 4.32 Changed: this script now TRAINS its own models with its own parameters (PARAMS, one
#      entry per blind mode) instead of loading whatever rls_shadow.py / rls_inject.py
#      saved. Those learners optimise performance on the seeds they trained on, which is a
#      different objective from "which frozen weights transfer best" — lambda especially
#      need not agree. Weights are saved under mode='blind_shadow' / 'blind_inject'.
# 4.27 Changed: gamma is per source model (shadow 1.0, inject 0.5) so each blind test uses
#      the same correction strength its training run did; a single shared gamma was
#      measuring a different correction from the one learned.
# 4.26 Changed: renamed rls_blind.py -> rls_blind.py and now runs TWO blind experiments —
#      the shadow model (post-hoc, isolates generalisation) and the inject model (live
#      frozen filter, the deployable case). Results save as modes 'blind_shadow' and
#      'blind_inject'; figures and summaries are emitted per source model.
# 4.24 Fixed: panel (a) drew ONE baseline (the blind seeds') while plotting the training
#      seeds' corrected line against it, and computed the training excess using the BLIND
#      seeds' PF floor. Each population now carries its own baseline and its own PF.
# 4.23 Fixed: panel (b) iterated SEEDS (train + test) so the training seeds appeared in the
#      legend with no bars; now iterates TEST_SEEDS. Dropped stale 'leave-one-out' and
#      'trained on all seeds' wording — training is cfg.seeds, blind is the +100 set — and
#      removed 'held-out' from the labels.
# 4.22 Fixed: generalisation cost was the DIFFERENCE OF ABSOLUTE excess levels between the
#      training seeds and the blind seeds — two populations with different intrinsic
#      difficulty (baselines can differ ~2x), so the number conflated generalisation with
#      seed difficulty and sometimes came out negative. Now each set is scored as a
#      relative reduction against its own baseline.
# 4.21 Changed: loads the trained model from rls_shadow.py (save_weights) instead of
#      re-training internally, and refuses to run if the stored training seeds overlap the
#      blind test seeds. Local training kept as a fallback.
# 4.19 Changed: GAMMA default 0.05 -> 1.0 to match rls_shadow.py, so blind and shadow numbers
#      are measured on the same footing.
# 4.18 Changed: trains on cfg.seeds and tests on a DISJOINT set of fresh seeds (+100)
#      instead of leave-one-out, so all training data is used and the blind runs share no
#      observation noise with it. Adds a standalone excess-vs-alpha figure and the three
#      new features.
# 4.17 created — blind-filter test for the shadow correction. Leave-one-out over seeds with
#      frozen weights and frozen feature statistics; the held-out run's residual and PF are
#      never read. Saves as mode='blind' in the common schema and emits a blind-vs-shadow
#      comparison figure.
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg
from rls_core import (load_log, n_features, RunningStandardiser, RLS, save_stage4,
                      dyn_row, N_DYN, save_weights, tail_mean_weights, rmse_percomp)
from rls_inject import run_enkf_rls, train_inject_chain

os.makedirs('figs/results', exist_ok=True)
os.makedirs('figs/diagnostic/s4', exist_ok=True)
S4 = 'figs/diagnostic/s4'

ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
# Frozen weights are fitted on TRAIN_SEEDS and scored on TEST_SEEDS. Choose how many
# trajectories the weights see via cfg.blind_seeds, or L63_BLIND_SEEDS=100 / 100,101,102.
# The test population never changes, so a 1-seed and a 3-seed model are scored on the same runs.
TRAIN_SEEDS = list(cfg.blind_seeds)                 # weights are fitted on these
TEST_SEEDS = list(cfg.seeds)                        # scored here — the shadow/inject population
if set(TRAIN_SEEDS) & set(TEST_SEEDS):
    raise SystemExit(f'blind training seeds {TRAIN_SEEDS} overlap the test seeds '
                     f'{TEST_SEEDS} — the test would not be blind')

PARAMS = {
    'shadow': dict(lam=1.0, gamma=0.7, tail_frac=0.5),
    'inject': dict(lam=0.995, gamma=0.05, tail_frac=0.5),
}
P0 = 1e3
USE_DELTA = False
BASE, PFC, SHA, BLI = '#c0392b', '#16a085', '#e67e22', '#8e44ad'
fs.use()
AX, FS = fs.AX, fs.FS     # sizes come from figstyle; do not re-assert them here

tw = np.load('data/l63_twin.npz')
truth = tw['truth']; obs_idx = tw['obs_idx']
truth_at_obs = truth[obs_idx]
rmse = rmse_percomp                     # per component, then averaged over x, y, z (5.18)


def train_shadow(paths, alpha, lam, tail_frac):

    first = load_log(paths[0][1], alpha, use_delta=USE_DELTA)
    K = n_features(first['U_raw'].shape[1])
    std = RunningStandardiser(first['U_raw'].shape[1] + N_DYN)
    rls = RLS(K, lam=lam, P0=P0)
    hist = []
    for _, p in paths:
        L = load_log(p, alpha, use_delta=USE_DELTA)
        T = L['U_raw'].shape[0]
        lag_pred = np.zeros(3)
        for t in range(T):
            u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]))
            lag_pred = rls.predict(u)                # own prediction feeds the next cycle
            rls.update(u, L['r'][t])                 # target seen here, and only here
            hist.append(rls.w.copy())
    return tail_mean_weights(np.array(hist), tail_frac), std


def apply_blind(w, std, L):

    T = L['U_raw'].shape[0]
    pred = np.zeros((T, 3))
    lag_pred = np.zeros(3)
    for t in range(T):
        u = std.transform(np.concatenate([L['U_raw'][t], dyn_row(lag_pred, t, T)]),
                          update=False)                   # FROZEN: no adaptation to test data
        pred[t] = w.T @ u
        lag_pred = pred[t]                                # autoregressive, own output only
    return pred



SOURCE_MODES = ['shadow', 'inject']
all_results = {}                                    # source mode -> alpha -> records


def blind_shadow(w, std, L, raw, gamma):
    pred = apply_blind(w, std, L)                   # target never read
    return raw['xa_mean'] + gamma * pred, pred


def blind_inject(w, std, a, raw, ostd, gamma):

    h = lambda x: x + a * x ** 2
    base = cfg.perturb_std.copy()                   # match the logged jitter calibration
    if 'en_mult' in raw.files:
        cfg.perturb_std = base * float(raw['en_mult'])
    res = run_enkf_rls(raw['obs'], truth, obs_idx, h, raw['xpf_mean'], gamma=gamma,
                       seed=int(raw['seed']) if 'seed' in raw.files else 0,
                       obs_std=ostd, w_init=w, std_init=std, freeze=True)
    cfg.perturb_std = base
    return res['en_mean'], res['pred'], res['target'], res['diverged_at']


for source in SOURCE_MODES:
    results = {}
    P = PARAMS[source]
    print(f'\n{"=" * 70}\nBLIND TEST of the {source.upper()} model'
          f"   lambda={P['lam']}  gamma={P['gamma']}  tail={int(P['tail_frac'] * 100)}%"
          f'\n{"=" * 70}')
    for a in ALPHAS:
        pth = lambda sd: f'data/stage4_log_a{a}_s{sd}_{cfg.noise_mode}.npz'
        tr_paths = [(sd, pth(sd)) for sd in TRAIN_SEEDS if os.path.exists(pth(sd))]
        te_paths = [(sd, pth(sd)) for sd in TEST_SEEDS if os.path.exists(pth(sd))]
        if not tr_paths or not te_paths:
            print(f'alpha={a}: need training seeds {TRAIN_SEEDS} and blind seeds {TEST_SEEDS} '
                  f'(found {len(tr_paths)} / {len(te_paths)}) — rerun stage4_log.py')
            continue

        ai = int(np.argmin(np.abs(tw['alphas'] - a)))
        ostd = tw['obs_std_alpha'][ai]


        if source == 'shadow':
            w, std = train_shadow(tr_paths, a, P['lam'], P['tail_frac'])
        else:
            w, std = train_inject_chain(tr_paths, a, P['lam'], P['gamma'], ostd,
                                        truth, obs_idx, P['tail_frac'])
        if w is None:
            print(f'alpha={a}: {source} training diverged everywhere, skipping'); continue
        wpath = save_weights(a, w, std, [sd for sd, _ in tr_paths], P['lam'],
                             mode=f'blind_{source}_n{len(tr_paths)}')   # n = training seeds
        print(f"alpha={a}: trained {source} model on seeds {[sd for sd, _ in tr_paths]} "
              f"(lambda={P['lam']}) -> {wpath}")

        recs = []
        for held, hpath in te_paths:
            L_test = load_log(hpath, a, use_delta=USE_DELTA)
            raw = np.load(hpath)
            xa_base = raw['xa_mean']
            if source == 'shadow':
                xa_corr, pred = blind_shadow(w, std, L_test, raw, P['gamma'])
                target, div = L_test['r'], None
            else:
                xa_corr, pred, target, div = blind_inject(w, std, a, raw, ostd, P['gamma'])
            r_b = rmse(xa_base, truth_at_obs)
            r_c = rmse_percomp(xa_corr, truth_at_obs, nan=True)   # blind inject can diverge
            r_p = rmse(raw['xpf_mean'], truth_at_obs)
            recs.append(dict(seed=held, pred=pred, target=target,
                             innov=L_test['U_raw'][:, :3],   # first block is the innovation
                             xa_base=xa_base, xa_corr=xa_corr, xpf=raw['xpf_mean'],
                             times=raw['times'], r_b=r_b, r_c=r_c, r_p=r_p,
                             exc_b=(r_b - r_p) / r_p * 100, exc_c=(r_c - r_p) / r_p * 100))
            tag = '' if div is None else f'  DIVERGED at cycle {div}'
            print(f'alpha={a} blind seed {held}: RMSE base={r_b:.3f} blind={r_c:.3f} '
                  f'pf={r_p:.3f}  contribution {"+" if r_c < r_b else "-"}{tag}')

        # non-blind reference: the SAME weights applied back to the training seeds.
        nb_b, nb_c = [], []
        for sd, p_ in tr_paths:
            L = load_log(p_, a, use_delta=USE_DELTA)
            raw = np.load(p_)
            if source == 'shadow':
                xc, _ = blind_shadow(w, std, L, raw, P['gamma'])
            else:
                xc, _, _, _ = blind_inject(w, std, a, raw, ostd, P['gamma'])
            nb_b.append(rmse(raw['xa_mean'], truth_at_obs))
            nb_c.append(rmse_percomp(xc, truth_at_obs, nan=True))
        pf_train = float(np.mean([rmse(np.load(p_)['xpf_mean'], truth_at_obs)
                                  for _, p_ in tr_paths]))
        results[a] = dict(recs=recs, nonblind=(float(np.mean(nb_b)), float(np.mean(nb_c))),
                          pf_train=pf_train)

        helped = sum(r['r_c'] < r['r_b'] for r in recs)
        print(f'  alpha={a}: blind {source} correction helped {helped}/{len(recs)} seeds | '
              f'mean excess {np.mean([r["exc_b"] for r in recs]):.1f}% -> '
              f'{np.mean([r["exc_c"] for r in recs]):.1f}%\n')

    if results:
        all_results[source] = results

if not all_results:
    raise SystemExit('no results — run stage4_log.py, rls_shadow.py and rls_inject.py first')

# ---- save each blind experiment under its own mode so stage4_results.py plots both ----
for source, results in all_results.items():
    for a, R in results.items():
        recs = R['recs']
        cat = lambda k: np.concatenate([r[k] for r in recs], axis=0)
        tr_all = np.tile(truth_at_obs, (len(recs), 1))
        out = save_stage4(f'blind_{source}', a, np.concatenate([r['times'] for r in recs]),
                          cat('pred'), cat('target'), cat('target') - cat('pred'),
                          xa_corr=cat('xa_corr'), xa_base=cat('xa_base'),
                          xpf_mean=cat('xpf'), truth=tr_all,
                          w_hist=np.zeros((cat('pred').shape[0], 1, 3)),  # frozen: no trajectory
                          sqerror_corr=(cat('xa_corr') - tr_all) ** 2,
                          innov=cat('innov'),
                          row_seed=np.concatenate([np.full(len(r['times']), r['seed'])
                                                   for r in recs]))
        print(f'saved {out}')


# ================= figures, one set per source model =================
for source, results in all_results.items():
    A = sorted(results)
    tb = [(results[a]['nonblind'][0] - results[a]['pf_train']) / results[a]['pf_train'] * 100
          for a in A]                                    # TRAINING seeds, uncorrected
    tc = [(results[a]['nonblind'][1] - results[a]['pf_train']) / results[a]['pf_train'] * 100
          for a in A]                                    # TRAINING seeds, corrected
    eb = [np.mean([r['exc_b'] for r in results[a]['recs']]) for a in A]   # BLIND, uncorrected
    ec = [np.mean([r['exc_c'] for r in results[a]['recs']]) for a in A]   # BLIND, corrected

    # (a) each population against its OWN baseline: the two seed sets are different runs
    # with different intrinsic difficulty, so a shared reference line would be misleading.
    def own_baseline(ax):
        # Seed lists were in these labels; at 0.48\textwidth they did not fit, and they
        # belong in the caption anyway — TRAIN_SEEDS and TEST_SEEDS are printed above.
        ax.plot(A, tb, 'o--', color=SHA, lw=1.4, ms=4, alpha=0.6, label='train, uncorrected')
        ax.plot(A, tc, 's-', color=SHA, lw=1.8, ms=4, label='train, corrected')
        ax.plot(A, eb, 'o--', color=BLI, lw=1.4, ms=4, alpha=0.6, label='blind, uncorrected')
        ax.plot(A, ec, '^-', color=BLI, lw=1.8, ms=4, label='blind, corrected')
        ax.axhline(0, ls=':', color='#777', lw=1)
        ax.set_xlabel(r'$\alpha$', **fs.LAB)
        ax.set_ylabel('excess over PF (%)', **fs.LAB)
        ax.grid(alpha=0.3); ax.legend(**fs.leg_above(ncol=2))

    # (b) per blind seed
    def per_seed(ax):
        width = 0.8 / max(len(TEST_SEEDS), 1)
        for i, sd in enumerate(TEST_SEEDS):
            vals, xs = [], []
            for j, a in enumerate(A):
                rec = [r for r in results[a]['recs'] if r['seed'] == sd]
                if rec:
                    xs.append(j + (i - len(TEST_SEEDS) / 2) * width + width / 2)
                    vals.append(rec[0]['exc_b'] - rec[0]['exc_c'])   # excess removed
            ax.bar(xs, vals, width, label=f'seed {sd}')
        ax.axhline(0, color='#333', lw=1.2)
        ax.set_xticks(range(len(A))); ax.set_xticklabels(A)
        ax.set_xlabel(r'$\alpha$', **fs.LAB)
        ax.set_ylabel('excess removed (pp)', **fs.LAB)   # sign convention -> caption
        ax.grid(alpha=0.3, axis='y'); ax.legend(**fs.leg_above(ncol=len(TEST_SEEDS)))

    PANES = [('own_baseline', own_baseline), ('per_seed', per_seed)]
    for stem, drawfn in PANES:                       # standalone, each read alone
        fig, ax = plt.subplots(figsize=fs.size(0.48))
        drawfn(ax)
        fs.save(fig, f'figs/results/stage4_blind_{source}_{stem}.png')

    # merged 1x2 — the write-up figure. One image with one caption instead of two
    # subfigures with two subcaptions, and matplotlib controls the gutter.
    fig, axes = plt.subplots(1, 2, figsize=fs.size(1.0, 0.42))
    for i, (stem, drawfn) in enumerate(PANES):
        drawfn(axes[i])
        fs.panel_letter(axes[i], i)   # no subcaptions here to carry them
    fig.tight_layout(w_pad=0.4)
    fs.save(fig, f'figs/results/stage4_blind_{source}_generalisation.png', tight=False)

    # per-alpha blind prediction vs true residual, one panel per seed plus a per-seed PNG
    for a in A:
        recs = results[a]['recs']
        fig, axes = plt.subplots(len(recs), 1, figsize=(13, 2.8 * len(recs)))
        axes = np.atleast_1d(axes)
        singles = [plt.subplots(figsize=(13, 3.4)) for _ in recs]
        for i, r in enumerate(recs):
            # each panel drawn twice: onto the combined backup and onto its own figure
            for ax, xlab in ((axes[i], i == len(recs) - 1), (singles[i][1], True)):
                ax.plot(r['times'], r['target'][:, 0], '-', color='#333333', lw=1.0,
                        label='true residual $r_t$ (x) — never learned from')
                ax.plot(r['times'], r['pred'][:, 0], '-', color=BLI, lw=1.2,
                        label=r'blind prediction $w_{\rm frozen}^{\top}u_t$')
                ax.set_ylabel(f'seed {r["seed"]}\nx', **fs.LAB)
                ax.grid(alpha=0.3)
                if xlab:
                    ax.set_xlabel('Time', **fs.LAB)
        axes[0].legend(**fs.LEG)
        fig.tight_layout()
        o = f'{S4}/stage4_blind_{source}_trace_a{a}.png'
        fig.savefig(o, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
        print(f'saved {o}')

        for r, (figx, axs) in zip(recs, singles):      # figx, not fs — fs is figstyle
            axs.legend(**fs.LEG)
            figx.tight_layout()
            o = f'{S4}/stage4_blind_{source}_trace_a{a}_s{r["seed"]}.png'
            figx.savefig(o, dpi=fs.DPI, bbox_inches='tight'); plt.close(figx)
            print(f'saved {o}')

# ================= summary =================
for source, results in all_results.items():
    A = sorted(results)
    print(f'\n[{source} model] blind vs optimistic — relative RMSE-excess reduction, '
          f"each set against its own baseline:")
    print(f"{'alpha':>6}{'train base%':>13}{'train red%':>12}{'blind base%':>13}"
          f"{'blind red%':>12}{'cost':>8}{'seeds+':>8}")
    for a in A:
        recs = results[a]['recs']
        rp_tr = results[a]['pf_train']
        tbe = (results[a]['nonblind'][0] - rp_tr) / rp_tr * 100
        tce = (results[a]['nonblind'][1] - rp_tr) / rp_tr * 100
        bbe = np.mean([r['exc_b'] for r in recs]); bce = np.mean([r['exc_c'] for r in recs])
        tred = (tbe - tce) / tbe * 100 if tbe else 0.0
        bred = (bbe - bce) / bbe * 100 if bbe else 0.0
        helped = sum(r['r_c'] < r['r_b'] for r in recs)
        print(f'{a:>6.1f}{tbe:>13.1f}{tred:>12.1f}{bbe:>13.1f}{bred:>12.1f}'
              f'{tred - bred:>8.1f}{helped:>4}/{len(recs)}')
print('\ncost = reduction on seen data minus reduction on unseen data, both relative to')
print('their own baselines. 0 = generalises perfectly.')