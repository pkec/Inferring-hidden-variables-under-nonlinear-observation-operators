"""
rls_core.py — shared RLS machinery for Stage 4.

One source of truth for (a) how a feature vector is built from filter-observable
primitives, and (b) how the RLS weights update per cycle. Both the offline shadow
learner (stage4_rls.py) and the live corrected filter (enkf_rls.py) import this, so
the two experiments are guaranteed to use the identical model — otherwise "shadow
proves learnability, inject deploys it" would compare two different learners.

Nothing here runs a filter or reads data; it is pure per-cycle math.
"""

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 4.1  created — shared feature builder + running standardiser + matrix RLS state,
#      factored out of stage4_rls.py so enkf_rls.py (live inject) and the shadow
#      learner share one model definition.
# ============================================================

import numpy as np


def build_features(d, s2, xf_mean, delta, lag_d, use_delta=False):
    """Assemble the raw (pre-standardisation, pre-constant) feature matrix.
        d       : (T, 3) mean innovation y - E[h(x)]
        s2      : (T, 3) forecast variance per component
        xf_mean : (T, 3) forecast mean E[x]
        delta   : (T, 3) measured Jensen bias E[h(x)] - h(E[x])
        lag_d   : (T, 3) previous cycle's innovation (row 0 zero-padded)
        use_delta: include standalone delta (only when training spans several alpha;
                   within one alpha delta == alpha*s2 and the fit goes singular)
    Returns U_raw of shape (T, k-1); the constant column is added later, per cycle,
    by RunningStandardiser (so it is never standardised).
    Feature order and rationale:
        d        primary signal — residual scales with the pull
        s2       regime uncertainty (forecast spread)
        xf_mean  location — lobe / distance to the fold at x = -1/(2 alpha)
        d**2     leading curvature term — quadratic in the innovation (Taylor)
        d*delta  interaction — correction grows when pull large AND operator curved
        lag_d    one-cycle memory — cross-cycle error propagation
    """
    blocks = [d, s2, xf_mean, d**2, d * delta, lag_d]
    if use_delta:
        blocks.append(delta)
    return np.hstack(blocks)                        # (T, k-1)


def n_features(n_raw_cols):
    """Total feature count including the constant column. n_raw_cols = U_raw.shape[1]."""
    return n_raw_cols + 1


class RunningStandardiser:
    """Welford running mean/std z-scoring, then prepend a constant.

    Standardises each feature against statistics from cycles seen SO FAR only, so no
    future information leaks into the current cycle (matters for the live filter,
    where cycle t must not peek at t+1). d ~ O(1) and s2 ~ O(10) live on different
    scales; without this, P becomes ill-conditioned.
    """

    def __init__(self, n_raw_cols):
        self.n = 0                                   # cycles seen; int
        self.mean = np.zeros(n_raw_cols)             # running feature mean; (k-1,)
        self.M2 = np.zeros(n_raw_cols)               # running sum of squared devs; (k-1,)

    def transform(self, u_row):
        """z-score u_row (k-1,) against stats so far, update stats, prepend 1. Returns (K,)."""
        self.n += 1
        dev = u_row - self.mean                      # deviation before mean update; (k-1,)
        self.mean += dev / self.n                    # Welford mean
        self.M2 += dev * (u_row - self.mean)         # Welford M2
        std = np.sqrt(self.M2 / max(self.n - 1, 1)) + 1e-8   # running std; +eps guards cycle 0
        z = (u_row - self.mean) / std                # standardised row; (k-1,)
        return np.concatenate([[1.0], z])            # prepend constant -> (K,)


class RLS:
    """Matrix recursive least squares. Three output components share one feature
    vector, so w is (K, 3): three regressions on one design.

    The scalar slides are the 1-D case of this: u is a vector not a scalar, so the
    gain g = u / sum(u^2) becomes kgain = P u / (lambda + u^T P u), and the scalar
    1/sum(u^2) becomes the matrix P. lambda = 1 reproduces batch OLS exactly.
    """

    def __init__(self, K, lam=1.0, P0=1e3):
        self.w = np.zeros((K, 3))                    # weights; column c maps features -> r component c
        self.P = np.eye(K) * P0                      # inverse-covariance proxy (K,K); large P0 = weak prior
        self.lam = lam                               # forgetting factor in (0,1]; 1 -> batch OLS

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
        return err
