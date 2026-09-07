# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.18 Added: rmse_percomp() — the ONE definition of analysis RMSE for Stages 4 and 5: per
#      component first, then averaged over x, y, z, matching error_sweep.py since 5.16. Every
#      Stage 4/5 script now imports it instead of writing its own
#      sqrt(((a-b)**2).mean()), which pooled cycles and components in a single mean and so
#      weighted each component by its own squared error — in L63 that hands z ~84% of the
#      answer. Seven copies of that lambda existed; now there is one.
# 4.35 Added: optional row_seed in the common schema, so plots can split a concatenated
#      multi-seed result back into per-seed panels.
# 4.28 Added: tail_mean_weights() — shared definition of the converged-tail average, so
#      the learners and the diagnostic freeze weights the same way.
# 4.27 Changed: USE_DYN enabled (with USE_LAG_XF already on) -> 26 features, per the
#      weight/feature diagnostic. See the toggle comment for the honest margins.
# 4.26 Changed: weight files are mode-tagged (stage4_weights_{mode}_a{alpha}.npz) so the
#      shadow and inject learners no longer overwrite each other — they train against
#      different targets and are different models.
# 4.21 Added: save_weights / load_weights — a trained model is persisted as weights PLUS
#      the standardiser statistics (applying w against different feature stats is a
#      different model). Loading checks the feature-set toggles and refuses a mismatch.
# 4.20 Changed: reverted to the 19-feature baseline; lag_xf and the dynamic block sit
#      behind USE_LAG_XF / USE_DYN.
# 4.17 Added: RunningStandardiser.transform(update=False) to freeze the feature statistics
#      for a blind test, so held-out data is scored against TRAINING stats only.
# 4.14 Added: covariance-windup guard in RLS — trace(P) capped at p_max, without which
#      lambda<1 inflates P unboundedly in unexcited directions until the gain explodes.
# 4.1  created — shared feature builder + running standardiser + matrix RLS state,
#      factored out of rls_shadow.py so rls_inject.py (live inject) and the shadow
#      learner share one model definition.
#      NOTE: entries 4.14-4.21 were reconstructed after they were lost from this block;
#      the code they describe was already present.
# ============================================================

import numpy as np


def rmse_percomp(a, b, nan=False):
    se = (np.asarray(a) - np.asarray(b)) ** 2                    # (T, 3) squared error
    per = np.sqrt(np.nanmean(se, axis=0) if nan else se.mean(axis=0))   # (3,) per-component
    return float(np.nanmean(per) if nan else per.mean())


# ---- feature-set toggles -------------------------------------------------------------
# Both additions are ON, giving 26 features. An earlier reading said they destroyed
# generalisation, but that was measured on a 300-cycle test harness (~16 samples per
# feature) where any linear model memorises. Re-measured on the real logs (~105 samples
# per feature) with diagnostic_py/stage4_weight_features.py, the picture is much flatter:
# on the blind seeds with tail-mean weights, base 9.48%, +lag_xf 9.73%, +dyn 9.43%,
# +both 9.60%. The extras are worth at most a few tenths of a point and +dyn alone scored
# BELOW base, so treat this as a preference, not a demonstrated gain — the large, robust
# effect in that diagnostic was the weight-averaging, not the feature set.
USE_LAG_XF = True      # previous forecast mean, static block (+3 features)
USE_DYN = True         # lag_pred (3) + cycle fraction (1), dynamic block (+4 features)

N_DYN = 4 if USE_DYN else 0     # dynamic features appended per cycle


def dyn_row(lag_pred, t, T):
    if not USE_DYN:
        return np.empty(0)                       # disabled: contributes nothing to the vector
    return np.concatenate([lag_pred, [t / max(T - 1, 1)]])


def build_features(d, s2, xf_mean, delta, lag_d, lag_xf=None, use_delta=False):
    blocks = [d, s2, xf_mean, d**2, d * delta, lag_d]
    if lag_xf is not None and USE_LAG_XF:
        blocks.append(lag_xf)
    if use_delta:
        blocks.append(delta)
    return np.hstack(blocks)                        # (T, k_static)


def n_features(n_raw_cols):
    return n_raw_cols + N_DYN + 1


class RunningStandardiser:
    """Welford running mean/std z-scoring"""

    def __init__(self, n_raw_cols):
        self.n = 0                                   # cycles seen; int
        self.mean = np.zeros(n_raw_cols)             # running feature mean; (k-1,)
        self.M2 = np.zeros(n_raw_cols)               # running sum of squared devs; (k-1,)

    def transform(self, u_row, update=True):
        if update:
            self.n += 1
            dev = u_row - self.mean                  # deviation before mean update; (k-1,)
            self.mean += dev / self.n                # Welford mean
            self.M2 += dev * (u_row - self.mean)     # Welford M2
        std = np.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-8   # running std; +eps guards cycle 0
        z = (u_row - self.mean) / std                # standardised row; (k-1,)
        return np.concatenate([[1.0], z])            # prepend constant -> (K,)


def load_log(path, alpha, use_delta=False):
    d = np.load(path)
    h = lambda x: x + alpha * x ** 2               # observation operator for this alpha
    fc = d['fc_enkf']                              # (T, N, 3) forecast ensemble per cycle
    xa_mean = d['xa_mean']                         # (T, 3) analysis mean
    xpf_mean = d['xpf_mean']                       # (T, 3) PF mean (teacher)
    obs = d['obs']                                 # (T, 3) observations

    xf_mean = fc.mean(1)                           # (T, 3) forecast mean E[x]
    hbar = h(fc).mean(1)                           # (T, 3) mean predicted obs E[h(x)]
    s2 = fc.var(1, ddof=1)                         # (T, 3) forecast variance
    dd = obs - hbar                                # (T, 3) mean innovation
    delta = d['jensen']                            # (T, 3) measured Jensen bias
    lag_d = np.vstack([np.zeros((1, 3)), dd[:-1]])        # (T, 3) previous innovation, row 0 padded
    lag_xf = np.vstack([np.zeros((1, 3)), xf_mean[:-1]])  # (T, 3) previous forecast mean, row 0 padded

    return dict(
        U_raw=build_features(dd, s2, xf_mean, delta, lag_d, lag_xf=lag_xf, use_delta=use_delta),
        r=xpf_mean - xa_mean, xa_mean=xa_mean, xpf_mean=xpf_mean,
        truth=d['truth_at_obs'], times=d['times'],
    )


def tail_mean_weights(w_hist, frac=0.5):
    ok = np.isfinite(w_hist).all(axis=(1, 2))        # (T,) rows actually written
    if not ok.any():
        raise ValueError('weight trajectory is entirely non-finite; the run diverged '
                         'immediately, so there is no model to freeze.')
    live = w_hist[ok]                                 # keep only the cycles that ran
    tail = live[int((1 - frac) * len(live)):]
    return tail.mean(0)


def save_weights(alpha, w, std, seeds, lam, mode='shadow', path=None):
    out = path or f'data/stage4_weights_{mode}_a{alpha}.npz'
    np.savez(out, w=w, alpha=alpha, lam=lam, seeds=np.array(seeds),
             std_n=std.n, std_mean=std.mean, std_M2=std.M2,
             n_static=len(std.mean) - N_DYN, K=w.shape[0], mode=mode,
             use_lag_xf=USE_LAG_XF, use_dyn=USE_DYN)
    return out


def load_weights(alpha, mode='shadow', path=None):
    d = np.load(path or f'data/stage4_weights_{mode}_a{alpha}.npz')
    if bool(d['use_lag_xf']) != USE_LAG_XF or bool(d['use_dyn']) != USE_DYN:
        raise ValueError(
            f"feature-set mismatch: weights were trained with USE_LAG_XF={bool(d['use_lag_xf'])}, "
            f"USE_DYN={bool(d['use_dyn'])} but rls_core currently has {USE_LAG_XF}, {USE_DYN}. "
            f"Retrain (rls_shadow.py) or restore the toggles.")
    std = RunningStandardiser(int(d['n_static']) + N_DYN)
    std.n, std.mean, std.M2 = int(d['std_n']), d['std_mean'].copy(), d['std_M2'].copy()
    meta = dict(alpha=float(d['alpha']), lam=float(d['lam']), seeds=list(d['seeds']),
                K=int(d['K']), mode=str(d['mode']) if 'mode' in d.files else 'shadow')
    return d['w'].copy(), std, meta


def save_stage4(mode, alpha, times, pred, target, rls_resid,
                xa_corr, xa_base, xpf_mean, truth, w_hist, sqerror_corr, innov=None,
                row_seed=None):
    out = f'data/stage4_{mode}_a{alpha}.npz'
    np.savez(out, mode=mode, alpha=alpha, times=times,
             pred=pred, target=target, rls_resid=rls_resid,
             xa_corr=xa_corr, xa_base=xa_base, xpf_mean=xpf_mean, truth=truth,
             w_hist=w_hist, sqerror_corr=sqerror_corr,
             **({} if innov is None else dict(innov=innov)),
             **({} if row_seed is None else dict(row_seed=np.asarray(row_seed))))
    return out


class RLS:

    def __init__(self, K, lam=1.0, P0=1e3, p_max=1e8):
        self.w = np.zeros((K, 3))                    # weights; column c maps features -> r component c
        self.P = np.eye(K) * P0                      # inverse-covariance proxy (K,K); large P0 = weak prior
        self.lam = lam                               # forgetting factor in (0,1]; 1 -> batch OLS
        self.p_max = p_max                           # cap on trace(P); guards covariance windup
        self.wound_up = 0                            # how many times the cap was hit; diagnostic

    def predict(self, u):
        """Residual prediction from CURRENT weights. u:(K,) -> (3,)."""
        return self.w.T @ u

    def update(self, u, r_t):
        """One RLS step given feature vector u:(K,) and target r_t:(3,).
        Returns the pre-update miss err = r_t - w_old^T u  (the slide-4 per-cycle metric).
        """
        y_hat = self.w.T @ u                         # prediction with old weights; (3,)
        err = r_t - y_hat                            # miss; (3,)  == (r_t - w_{t-1} u_t)
        Pu = self.P @ u                              # (K,)
        kgain = Pu / (self.lam + u @ Pu)             # gain vector; (K,)  == g = u/(u^2 + lambda sum)
        self.w = self.w + np.outer(kgain, err)       # w_t = w_{t-1} + g * err ; (K,3)
        self.P = (self.P - np.outer(kgain, Pu)) / self.lam   # /lambda inflates P so old data fades

        # COVARIANCE WINDUP GUARD. With lambda < 1 the /lambda above inflates P every step.
        # Directions the features never excite receive no new information to balance that,
        # so P grows without bound (measured: ||P|| 4e3 -> 9e16 over 600 cycles at
        # lambda=0.95), the gain explodes, and the weights follow. Downstream that destroys
        # the ensemble and the Lorenz integration overflows to NaN. Capping trace(P) is the
        # standard remedy (covariance limiting); it bounds how confident-in-reverse the
        # estimator can become without changing the update direction.
        if self.p_max is not None:
            tr = np.trace(self.P)
            if tr > self.p_max:
                self.P *= self.p_max / tr            # rescale, preserving the direction
                self.wound_up += 1
        return err