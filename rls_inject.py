

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching error_sweep.py 5.16. Print-only here. The corrected RMSE keeps its
#      NaN-safe form (nan=True) because a live inject run can diverge and NaN-fill its tail.
# 4.37 Added: train_inject_chain() — the seed-chained training used to build a frozen
#      inject model, moved here from rls_blind.py so the trajectory diagnostic can import it
#      without executing that script (rls_blind has no __main__ guard).
# 4.33 Changed: online inject parameters set from the sweep — GAMMA 0.5, LAM 0.995 -> 0.99.
# 4.32 Removed: weight saving (rls_blind.py trains its own model with its own parameters).
# 4.28 Changed: w_hist is NaN-filled past a divergence so those unwritten rows cannot drag
#      a tail-mean of the weights toward zero.
# 4.27 Changed: GAMMA 0.05 -> 0.5 (provisional, pending the gamma x lambda sweep).
# 4.26 Changed: renamed rls_inject.py -> rls_inject.py. RLS state (weights, covariance and
#      standardiser) now CARRIES FORWARD across seeds as it does in rls_shadow.py, so one
#      trained model emerges per alpha instead of three independent replicates; that model
#      is saved via save_weights(mode='inject') for rls_blind.py. run_enkf_rls returns the
#      live rls/std objects so the caller can chain seeds.
# 4.21 Added: w_init / std_init (warm start from a saved model) and freeze=True, which
#      never updates the weights and never reads xpf_teacher — the deployable blind live
#      filter: fixed weights, no PF, corrections still fed back into the forecast.
# 4.20 Changed: static feature count derived from the rls_core toggles rather than
#      hard-coded, so reverting to 19 features needs no edit here.
# 4.18 Changed: feature vector extended with lag_xf and the dynamic block (lag_pred, cycle
#      fraction); the loop threads the previous forecast mean and its own prediction forward.
# 4.14 Added: divergence guard — non-finite prediction or an ensemble leaving the attractor
#      stops the run and returns diverged_at instead of letting the RK4 overflow to NaN and
#      crash scipy's inv(). Diverged tails are NaN-filled so partial runs cannot be mistaken
#      for valid RMSE. Also reports the RLS windup counter.
# 4.13 Added: per-cycle innovation recorded and saved. Note this is the innovation the
#      CORRECTED filter saw, which diverges from the logged one once injection shifts the
#      trajectory — that is the point, not a discrepancy.
# 4.11 Changed: __main__ now sweeps ALPHAS x SEEDS (matching stage4_log.py / rls_shadow.py)
#      instead of a single hard-coded alpha; reads the per-seed log filenames, runs each
#      seed as an independent replicate with fresh weights, and concatenates seeds into one
#      inject result file per alpha. Also fixes a stale single-alpha log path.
# 4.2  Added: per-cycle weight trajectory (w_hist) returned; __main__ saves via
#      rls_core.save_stage4 so shadow/inject/blind share one schema for plotting.
# 4.1  created — live RLS-corrected EnKF. Copy of enkf.run_enkf with an in-loop
#      inject-then-learn step (old weights inject, target formed against the
#      corrected mean, weights updated, corrected ensemble propagated). PF teacher
#      read from the Stage 4 log. Fixed conservative gamma. Uses shared rls_core.
# ============================================================

import os
import numpy as np
from scipy import linalg
from config import cfg, rk4_vec
from rls_core import (build_features, n_features, RunningStandardiser, RLS, save_stage4,
                      dyn_row, N_DYN, USE_LAG_XF, load_weights, tail_mean_weights,
                      rmse_percomp)


def _analysis(Af, d_obs, Cdd, h, rng):

    Nm = Af.shape[0]
    psi_f_m = np.mean(Af, 0, keepdims=True)         # (1,3) forecast mean E[x]
    D = rng.multivariate_normal(d_obs, Cdd, Nm)     # (Nm,3) perturbed obs
    Y = h(Af)                                       # (Nm,3) per-member predicted obs h(x_i)
    Y_mean = np.mean(Y, axis=0, keepdims=True)      # (1,3) E[h(x)]
    S = Y - Y_mean                                  # (Nm,3) obs-space anomalies
    delta = (Y_mean - h(psi_f_m[0]))[0]             # (3,) measured Jensen bias E[h(x)]-h(E[x])
    C = (Nm - 1) * Cdd + S.T @ S                    # (3,3)
    Cinv = linalg.inv(C)
    Aa = Af + (D - Y) @ (Cinv @ (S.T @ Af))         # factored update (as enkf.py)
    if not np.isreal(Aa).all():                     # numerical guard (as enkf.py)
        Aa = Af
    hbar = Y_mean[0]                                # (3,)
    xf_mean = psi_f_m[0]                            # (3,)
    s2 = Af.var(0, ddof=1)                          # (3,) forecast variance
    return Aa, hbar, xf_mean, s2, delta


def run_enkf_rls(obs, truth, obs_idx, h, xpf_teacher,
                 gamma=0.5, lam=1.0, P0=1e3, use_delta=False,
                 seed=cfg.seed, N=cfg.ensembleN, obs_std=None,
                 w_init=None, std_init=None, P_init=None, freeze=False):

    rng = np.random.default_rng(seed)               # fresh rng per run
    n_obs = len(obs)
    obs_std = cfg.obs_std if obs_std is None else obs_std
    R = np.diag(obs_std ** 2)                        # observation-error covariance

    ensemble = np.tile(truth[0], (N, 1)) \
        + rng.normal(0, 1, (N, 3)) * cfg.init_std
    truth_at_obs = truth[obs_idx]

    # RLS state (shared definitions so shadow and this filter learn the identical model)
    # feature count: 6 blocks x 3 (+1 constant), +3 if use_delta. Probe once to size K.
    n_blocks = 6 + (1 if USE_LAG_XF else 0) + (1 if use_delta else 0)   # 3-wide blocks
    n_raw = 3 * n_blocks                            # static feature columns
    K = n_features(n_raw)                           # static + dynamic + constant; int
    std = std_init if std_init is not None else RunningStandardiser(n_raw + N_DYN)
    rls = RLS(K, lam=lam, P0=P0)                     # matrix RLS
    if w_init is not None:
        rls.w = w_init.copy()                        # warm start from a trained model
    if P_init is not None:
        # carry the covariance too when chaining seeds. Restarting P at P0 each seed would
        # tell the estimator it is uncertain again and let the first few cycles of every
        # seed overwrite what earlier seeds taught it.
        rls.P = P_init.copy()

    en_mean = np.zeros((n_obs, 3))                   # CORRECTED analysis mean per cycle
    sqerror = np.zeros((n_obs, 3))                   # (corrected mean - truth)^2
    spread  = np.zeros((n_obs, 3))                   # analysis spread (unchanged by the shift)
    pred    = np.zeros((n_obs, 3))                   # learned correction w_{k-1}^T u_k
    target  = np.zeros((n_obs, 3))                   # r_k = pf_teacher - corrected mean
    rls_resid = np.zeros((n_obs, 3))               # r_k - w_{k-1}^T u_k (convergence metric)
    w_hist  = np.zeros((n_obs, K, 3))                # weight trajectory, snapshot after each update
    innov   = np.zeros((n_obs, 3))                   # per-cycle mean innovation the corrected filter saw
    diverged_at = None                               # cycle index where the run blew up, or None
    BLOWUP = 1e3                                     # |state| far beyond the L63 attractor (~|x|<50);
                                                     # generous, but low enough that the next
                                                     # window's quadratic terms cannot overflow
    d_prev = np.zeros(3)                             # previous cycle's innovation (for the lag feature)
    xf_prev = np.zeros(3)                            # previous cycle's forecast mean (lag_xf)
    lag_pred = np.zeros(3)                           # model's own previous prediction (autoregressive)

    for k in range(n_obs):
        ensemble += rng.normal(0, 1, (N, 3)) * cfg.perturb_std
        # overflow here is expected when a correction has already blown the state up; the
        # guard below catches it, so the warnings are noise rather than information
        with np.errstate(over='ignore', invalid='ignore'):
            for _ in range(cfg.obs_every):
                ensemble = rk4_vec(ensemble, cfg.dt)  # propagate the (possibly corrected) ensemble

        # Divergence guard, placed AFTER propagation and BEFORE the analysis: this is where
        # a blown-up state actually becomes inf/NaN. The Lorenz terms are quadratic (x*y,
        # x*z), so an ensemble that looks merely large at the end of a cycle can overflow
        # within the next window's RK4 steps and then crash inv() on a non-finite matrix.
        if not np.isfinite(ensemble).all() or np.abs(ensemble).max() > BLOWUP:
            diverged_at = k
            break

        # 1. analysis update
        Af = ensemble
        Aa, hbar, xf_mean, s2, delta = _analysis(Af, obs[k], R, h, rng)

        # 2. features from THIS cycle's forecast (single-row build via the shared builder)
        d_k = obs[k] - hbar                          # (3,) mean innovation this cycle
        innov[k] = d_k
        U_raw_k = build_features(d_k[None], s2[None], xf_mean[None], delta[None],
                                 d_prev[None], lag_xf=xf_prev[None],
                                 use_delta=use_delta)[0]                     # (k_static,)
        # frozen: score against TRAINING statistics and leave them untouched, exactly as the
        # offline blind test does, so the live and offline blind models are identical
        u = std.transform(np.concatenate([U_raw_k, dyn_row(lag_pred, k, n_obs)]),
                          update=not freeze)                                   # (K,)

        # 3. predict correction with OLD weights
        y_hat = rls.predict(u)                        # (3,)
        pred[k] = y_hat

        # 4. INJECT: shift every member by gamma * y_hat (mean moves, spread preserved).
        # Guard the prediction first: once weights blow up, y_hat can be inf/NaN and would
        # poison the ensemble irrecoverably.
        if not np.isfinite(y_hat).all():
            diverged_at = k
            break
        Aa = Aa + gamma * y_hat                       # broadcast (3,) across (N,3)
        ensemble = Aa                                 # this corrected ensemble propagates onward

        corr_mean = ensemble.mean(0)                  # (3,) corrected analysis mean

        # 5. target against the CORRECTED mean (the remaining gap to the PF).
        # In freeze mode the teacher is never consulted — that is what makes it blind.
        if not freeze and xpf_teacher is not None:
            r_k = xpf_teacher[k] - corr_mean          # (3,)
            target[k] = r_k
            # 6. update weights on how the old-weight correction actually landed
            rls_resid[k] = rls.update(u, r_k)         # returns pre-update miss (3,)
        elif xpf_teacher is not None:
            target[k] = xpf_teacher[k] - corr_mean    # recorded for diagnostics only, not learned from
        w_hist[k] = rls.w                             # (K,3) weights after this cycle

        # diagnostics
        en_mean[k] = corr_mean
        dev = ensemble - corr_mean
        sqerror[k] = (truth_at_obs[k] - corr_mean) ** 2
        spread[k]  = np.mean(dev ** 2, axis=0)
        d_prev = d_k                                  # carry innovation forward for next lag
        xf_prev = xf_mean                             # carry forecast mean forward (lag_xf)
        lag_pred = y_hat                              # carry this prediction forward (autoregressive)

        # Divergence guard: a bad correction compounds through the forecast, and once the
        # state leaves the attractor by orders of magnitude the Lorenz RK4 overflows and
        # every downstream array becomes NaN. Stop here and report instead of crashing.
        if not np.isfinite(ensemble).all() or np.abs(ensemble).max() > BLOWUP:
            diverged_at = k
            break

    if diverged_at is not None:                      # tail is meaningless; mark it
        en_mean[diverged_at:] = np.nan
        sqerror[diverged_at:] = np.nan
        spread[diverged_at:] = np.nan
        # w_hist too: those rows were never written and would otherwise pull a tail-mean
        # of the weights toward zero.
        w_hist[diverged_at:] = np.nan

    # rls and std are returned so the caller can carry the learned state into the next
    # seed (rls_shadow.py does the same); they are live objects, not copies.
    return dict(diverged_at=diverged_at, wound_up=rls.wound_up,
                en_mean=en_mean, sqerror=sqerror, spread=spread,
                pred=pred, target=target, rls_resid=rls_resid,
                truth_at_obs=truth_at_obs, w_hist=w_hist, w_final=rls.w, innov=innov,
                rls=rls, std=std)


def train_inject_chain(paths, alpha, lam, gamma, ostd, truth, obs_idx, tail_frac=0.5):

    h = lambda x: x + alpha * x ** 2
    w_c = std_c = P_c = None
    hist = []
    for sd, p_ in paths:
        raw = np.load(p_)
        base = cfg.perturb_std.copy()                # match the logged jitter calibration
        if 'en_mult' in raw.files:
            cfg.perturb_std = base * float(raw['en_mult'])
        res = run_enkf_rls(raw['obs'], truth, obs_idx, h, raw['xpf_mean'], gamma=gamma,
                           lam=lam, use_delta=False, seed=sd, obs_std=ostd,
                           w_init=w_c, std_init=std_c, P_init=P_c)
        cfg.perturb_std = base
        w_c, std_c, P_c = res['rls'].w, res['std'], res['rls'].P
        hist.append(res['w_hist'])
        if res['diverged_at'] is not None:
            print(f"    warning: inject training diverged on seed {sd} at cycle "
                  f"{res['diverged_at']}")
    try:
        return tail_mean_weights(np.concatenate(hist), tail_frac), std_c
    except ValueError:
        return None, None


if __name__ == '__main__':
    ALPHAS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]                # nonlinearity strengths to sweep; list
    SEEDS = cfg.seeds                                  # seeds trained in order, state carried forward
    # ONLINE inject parameters: the learning run, weights updating every cycle. The BLIND
    # (frozen-weight) counterpart has its own lambda and gamma in rls_blind.py. From
    # diagnostic_py/stage4_sweep.py.
    GAMMA = 0.35                                        # injection strength
    LAM = 0.99                                         # forgetting factor
    P0 = 1e3
    USE_DELTA = False
    # NOTE: no weight file is written here — rls_blind.py trains its own model with its own
    # parameters (see the note in rls_shadow.py). w_hist is still saved in the result npz.

    tw = np.load('data/l63_twin.npz')                  # truth + obs_idx shared by every run
    truth = tw['truth']; obs_idx = tw['obs_idx']
    alphas_grid = tw['alphas']; truth_at_obs = truth[obs_idx]

    print(f"inject sweep: alphas={ALPHAS} seeds={SEEDS} gamma={GAMMA} lambda={LAM}\n")
    for a in ALPHAS:
        ai = int(np.argmin(np.abs(alphas_grid - a)))   # index of this alpha in the twin grid
        ostd = tw['obs_std_alpha'][ai]                  # (3,) noise std for this alpha
        h = lambda x, a=a: x + a * x**2                 # h_alpha; a bound so the closure is correct

        # Seeds are visited in order and the RLS state (weights, covariance, standardiser)
        # CARRIES FORWARD, exactly as rls_shadow.py does. Each seed is an independent
        # observation-noise realisation of the same relationship, so this is more training
        # data rather than a repeat — and it leaves ONE trained model per alpha to save.
        per_seed = []
        w_carry = std_carry = P_carry = None
        for sd in SEEDS:
            path = f'data/stage4_log_a{a}_s{sd}_{cfg.noise_mode}.npz'
            if not os.path.exists(path):
                print(f'  alpha={a} seed={sd}: {path} not found — run stage4_log.py first')
                continue
            log = np.load(path)
            obs = log['obs']                            # (T,3) this seed's observations
            xpf = log['xpf_mean']                       # (T,3) PF teacher, fixed ground truth

            # Run at the SAME jitter the log was calibrated with. Without this the
            # corrected filter runs at default jitter while the baseline it is compared
            # against was calibrated — an unfair comparison the correction cannot win.
            base = cfg.perturb_std.copy()               # (3,) window baseline jitter
            cfg.perturb_std = base * float(log['en_mult']) if 'en_mult' in log.files else base
            res = run_enkf_rls(obs, truth, obs_idx, h, xpf, gamma=GAMMA, lam=LAM, P0=P0,
                               use_delta=USE_DELTA, seed=sd, obs_std=ostd,
                               w_init=w_carry, std_init=std_carry, P_init=P_carry)
            cfg.perturb_std = base                      # restore
            w_carry, std_carry = res['rls'].w, res['std']    # chain into the next seed
            P_carry = res['rls'].P

            # sqerror is already (a-truth)^2, so take the per-component root directly
            rc = float(np.sqrt(np.nanmean(res['sqerror'], axis=0)).mean())   # NaN-safe if diverged
            rb = rmse_percomp(log['xa_mean'], truth_at_obs)               # uncorrected EnKF
            rp = rmse_percomp(xpf, truth_at_obs)                          # PF floor
            per_seed.append(dict(res=res, log=log, xpf=xpf, rc=rc, rb=rb, rp=rp, seed=sd))
            div = '' if res['diverged_at'] is None else f"  DIVERGED at cycle {res['diverged_at']}"
            print(f'  alpha={a} seed={sd}: RMSE enkf={rb:.3f} inject_est={rc:.3f} '
                  f'({(rb - rc) / rb * 100:+.1f}% vs enkf){div}')

        if not per_seed:
            continue

        # concatenate seeds into one record per alpha, matching the shadow convention.
        # Times are offset so the seeds lay end-to-end instead of overlapping on the axis.
        cat = lambda k: np.concatenate([p['res'][k] for p in per_seed], axis=0)
        times, t_offset = [], 0.0
        for p_ in per_seed:
            tt = p_['log']['times']
            times.append(tt + t_offset)
            t_offset = times[-1][-1] + (tt[1] - tt[0])
        times = np.concatenate(times)

        xa_base = np.concatenate([p['log']['xa_mean'] for p in per_seed])
        xa_corr = cat('en_mean')
        xpf_all = np.concatenate([p['xpf'] for p in per_seed])
        truth_all = np.tile(truth_at_obs, (len(per_seed), 1))

        rmse_enkf = rmse_percomp(xa_base, truth_all)
        rmse_pf = rmse_percomp(xpf_all, truth_all)
        rmse_inj = rmse_percomp(xa_corr, truth_all, nan=True)   # NaN-safe: inject can diverge

        out = save_stage4('inject', a, times, cat('pred'), cat('target'), cat('rls_resid'),
                          xa_corr=xa_corr, xa_base=xa_base, xpf_mean=xpf_all, truth=truth_all,
                          w_hist=cat('w_hist'), sqerror_corr=cat('sqerror'), innov=cat('innov'),
                          row_seed=np.concatenate([np.full(len(p_['log']['times']), p_['seed'])
                                                   for p_ in per_seed]))
        print(f'| alpha={a} [{len(per_seed)} seeds, {len(times)} cycles] lambda={LAM} | \n'
              f'RMSE enkf={rmse_enkf:.3f} inject_est={rmse_inj:.3f} pf={rmse_pf:.3f} -> {out}')
        print()