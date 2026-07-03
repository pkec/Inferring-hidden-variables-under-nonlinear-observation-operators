"""
enkf_linear_test.py — sanity check that the EnKF analysis is correct.
On a linear-Gaussian problem the Kalman filter is the exact Bayesian answer, so a
correct stochastic EnKF must converge to it as the ensemble size N grows. Two checks,
both reusing the project's own EnKF() analysis function:
  A) one analysis step vs the exact KF update, error vs N (should fall ~1/sqrt(N))
  B) a full multi-cycle linear filter vs the exact KF (RMSE and covariance trace match)
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from enkf import EnKF

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 2.12 created — EnKF correctness test vs the exact Kalman filter on a linear-Gaussian system
# ============================================================

os.makedirs('figs', exist_ok=True)
rng = np.random.default_rng(0)

# --- Test A: one analysis step, EnKF vs exact KF, error vs N ---
H = np.array([[1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]])              # observe x and z only (partial -> real gain test)
h = lambda X: X @ H.T                         # operator the EnKF expects: (N,3) -> (N,2)
R = np.diag([0.5, 0.8])                       # observation-error covariance
m_f = np.array([1.0, -2.0, 0.5])             # prior mean
P_f = np.array([[2.0, 0.3, 0.1],
                [0.3, 1.5, -0.2],
                [0.1, -0.2, 1.0]])           # prior covariance (SPD)
d = np.array([1.4, 0.2])                     # one observation

K = P_f @ H.T @ np.linalg.inv(H @ P_f @ H.T + R)   # exact Kalman gain
m_a = m_f + K @ (d - H @ m_f)                       # exact posterior mean
P_a = (np.eye(3) - K @ H) @ P_f                     # exact posterior covariance

Ns = [50, 200, 1000, 5000, 20000]
reps = 20                                     # average over ensemble draws to smooth the curve
mean_err = np.zeros(len(Ns))
cov_err = np.zeros(len(Ns))
for i, N in enumerate(Ns):
    me = np.zeros(reps); ce = np.zeros(reps)
    for r in range(reps):
        Af = rng.multivariate_normal(m_f, P_f, N)      # draw prior ensemble
        Aa, _, _ = EnKF(Af, d, R, h, rng)              # the code under test
        me[r] = np.linalg.norm(Aa.mean(0) - m_a)
        ce[r] = np.linalg.norm(np.cov(Aa.T) - P_a)     # Frobenius
    mean_err[i] = me.mean(); cov_err[i] = ce.mean()

slope_mean = np.polyfit(np.log(Ns), np.log(mean_err), 1)[0]   # ~ -0.5 confirms MC convergence
slope_cov = np.polyfit(np.log(Ns), np.log(cov_err), 1)[0]

print("Test A — single EnKF analysis vs exact KF (error should fall ~1/sqrt(N))")
print(f"{'N':>8}{'mean err':>12}{'cov err':>12}")
for N, me, ce in zip(Ns, mean_err, cov_err):
    print(f"{N:8d}{me:12.4f}{ce:12.4f}")
print(f"log-log slope: mean {slope_mean:.2f}, cov {slope_cov:.2f}   (expect ~ -0.5)\n")

# --- Test B: full linear filter over many cycles, EnKF vs exact KF ---
M = np.array([[0.90, 0.10, 0.00],
              [-0.10, 0.90, 0.05],
              [0.00, -0.05, 0.92]])           # stable linear dynamics (spectral radius < 1)
Q = 0.1 * np.eye(3)                            # process noise (plays the role of inflation)
Hb = np.eye(3); hb = lambda X: X @ Hb.T        # observe everything (cleanest end-to-end check)
Rb = 0.5 * np.eye(3)
T, burn, N = 3000, 500, 1000
Qsqrt = np.linalg.cholesky(Q)

xt = np.zeros(3)                               # truth
mk = np.zeros(3); Pk = np.eye(3)              # KF mean / covariance
ens = rng.multivariate_normal(mk, Pk, N)       # EnKF ensemble
rmse_kf = np.zeros(T); rmse_en = np.zeros(T)

for k in range(T):
    xt = M @ xt + Qsqrt @ rng.standard_normal(3)              # advance truth
    obs = Hb @ xt + np.sqrt(np.diag(Rb)) * rng.standard_normal(3)

    mk = M @ mk; Pk = M @ Pk @ M.T + Q                        # exact KF forecast
    Kk = Pk @ Hb.T @ np.linalg.inv(Hb @ Pk @ Hb.T + Rb)
    mk = mk + Kk @ (obs - Hb @ mk); Pk = (np.eye(3) - Kk @ Hb) @ Pk

    ens = ens @ M.T + rng.standard_normal((N, 3)) @ Qsqrt.T   # EnKF forecast + process noise
    ens, _, _ = EnKF(ens, obs, Rb, hb, rng)                   # the code under test

    rmse_kf[k] = np.sqrt(((mk - xt) ** 2).mean())
    rmse_en[k] = np.sqrt(((ens.mean(0) - xt) ** 2).mean())

a_kf, a_en = rmse_kf[burn:].mean(), rmse_en[burn:].mean()
print("Test B — full linear filter over cycles")
print(f"time-avg analysis RMSE:  KF {a_kf:.4f}   EnKF {a_en:.4f}   diff {100*(a_en-a_kf)/a_kf:+.1f}%")
print(f"steady-state cov trace:  KF {np.trace(Pk):.4f}   EnKF {np.trace(np.cov(ens.T)):.4f}\n")

# --- figure: Test A convergence ---
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.loglog(Ns, mean_err, 'o-', color='#c0392b', label='analysis mean error')
ax.loglog(Ns, cov_err, 's-', color='#2471a3', label='analysis cov error (Frobenius)')
ref = mean_err[0] * (np.array(Ns) / Ns[0]) ** -0.5
ax.loglog(Ns, ref, 'k--', lw=1, label=r'$N^{-1/2}$ reference')
ax.set_xlabel('ensemble size N'); ax.set_ylabel('error vs exact Kalman filter')
ax.set_title('EnKF analysis converges to the Kalman filter')
ax.legend(); ax.grid(alpha=0.3, which='both')
fig.tight_layout(); fig.savefig('figs/enkf_linear_test.png', dpi=145)
print('saved figs/enkf_linear_test.png')
