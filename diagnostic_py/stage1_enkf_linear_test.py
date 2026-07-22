"""
enkf_linear_test.py — sanity check that the EnKF is correct, on a fully linear system.
Same shape as run_enkf (start the ensemble around truth[0], then each cycle: add jitter,
propagate, run the EnKF analysis). The only two things swapped versus the Lorenz pipeline are
the dynamics (linear x -> a*x + b instead of rk4) and the observation operator (linear h(x)=x).
On a linear-Gaussian problem the Kalman filter is the exact answer, so an exact Kalman filter is
run in parallel on the same observations; the EnKF mean and covariance must track it cycle by cycle.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
from config import cfg
from enkf import EnKF

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 2.15 Changed: rewritten to mirror run_enkf's loop (init around truth[0] -> per-cycle jitter ->
#               propagate -> EnKF analysis), compared cycle-by-cycle to an exact Kalman filter
#      Added:   linear dynamics x->a*x+b and linear h(x)=x; KF process noise Q from perturb_std
#      Removed: the old single-step 1/sqrt(N) convergence test (structure was unlike run_enkf)
# 2.12 created — EnKF correctness test vs the exact Kalman filter on a linear-Gaussian system
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
rng = np.random.default_rng(cfg.seed)

# --- the only things that differ from the Lorenz pipeline ---
a = 0.9                                   # linear dynamics x_{t+1} = a*x_t + b (per component)
b = np.array([0.5, -0.3, 1.0])            # each component settles toward b/(1-a) = [5, -3, 10]
h = lambda x: x                           # linear observation operator (observe all 3, as the baseline)
H = np.eye(3)                             # same operator, as the matrix the KF needs

N = cfg.ensembleN                         # 1000, same as the pipeline
n_steps = 200
R = np.diag(cfg.obs_std ** 2)             # sensor-noise covariance (reused from config)
Q = np.diag(cfg.perturb_std ** 2)         # the per-cycle jitter, written as a process-noise covariance

# --- truth trajectory + noisy observations (mirrors truth_obs.py, but linear dynamics) ---
truth = np.zeros((n_steps + 1, 3))
truth[0] = np.array([12.0, -8.0, 20.0])   # start away from the fixed point so there is a transient
for k in range(n_steps):
    truth[k + 1] = a * truth[k] + b
obs = truth[1:] + rng.normal(0, 1, (n_steps, 3)) * cfg.obs_std   # one noisy reading per step

# --- EnKF ensemble and exact KF, both initialised the same way as run_enkf ---
ensemble = np.tile(truth[0], (N, 1)) + rng.normal(0, 1, (N, 3)) * cfg.init_std
m = truth[0].copy()                       # KF mean  = ensemble mean
P = np.diag(cfg.init_std ** 2)            # KF cov   = ensemble spread

en_mean = np.zeros((n_steps, 3)); en_tr = np.zeros(n_steps)   # EnKF mean and cov-trace per cycle
kf_mean = np.zeros((n_steps, 3)); kf_tr = np.zeros(n_steps)   # KF   mean and cov-trace per cycle

for k in range(n_steps):
    # forecast — EnKF: jitter then linear push (exactly the run_enkf order)
    ensemble += rng.normal(0, 1, (N, 3)) * cfg.perturb_std    # process noise, like run_enkf
    ensemble = a * ensemble + b                               # linear dynamics (replaces rk4_vec)
    # forecast — KF: same map on the mean; jitter is added before the push so it scales by a^2 too
    m = a * m + b
    P = a**2 * (P + Q)                                        # = M(P+Q)M^T with M = a*I

    # analysis — EnKF: the code under test ; KF: the exact update it must match
    ensemble, _, _ = EnKF(ensemble, obs[k], R, h, rng)
    K = P @ H.T @ np.linalg.inv(H @ P @ H.T + R)              # exact Kalman gain
    m = m + K @ (obs[k] - H @ m)
    P = (np.eye(3) - K @ H) @ P

    en_mean[k] = ensemble.mean(0); en_tr[k] = np.cov(ensemble.T).trace()
    kf_mean[k] = m;                kf_tr[k] = P.trace()

# --- how well the EnKF matched the exact KF ---
truth_at = truth[1:]
print(f"mean |EnKF - KF|  (avg over steps & components): {np.abs(en_mean - kf_mean).mean():.4f}")
print(f"cov-trace |EnKF - KF|  (avg over steps):         {np.abs(en_tr - kf_tr).mean():.4f}")
print(f"RMSE vs truth   EnKF {np.sqrt(((en_mean-truth_at)**2).mean()):.4f}   "
      f"KF {np.sqrt(((kf_mean-truth_at)**2).mean()):.4f}")
print(f"final cov trace  EnKF {en_tr[-1]:.3f}   KF {kf_tr[-1]:.3f}")

# --- figure: EnKF and KF should sit on top of each other ---
t = np.arange(n_steps)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
a1.plot(t, truth_at[:, 0], 'k-', lw=1.2, label='truth (x)')
a1.scatter(t, obs[:, 0], s=8, color='#e67e22', alpha=0.4, label='obs')
a1.plot(t, kf_mean[:, 0], '-', color='#2471a3', lw=2, label='KF mean')
a1.plot(t, en_mean[:, 0], '--', color='#c0392b', lw=1.6, label='EnKF mean')
a1.set_xlabel('cycle'); a1.set_ylabel('x'); a1.set_title('(a) analysis mean: EnKF vs exact KF')
a1.legend(fontsize=8); a1.grid(alpha=0.3)

a2.plot(t, kf_tr, '-', color='#2471a3', lw=2, label='KF')
a2.plot(t, en_tr, '--', color='#c0392b', lw=1.6, label='EnKF')
a2.set_xlabel('cycle'); a2.set_ylabel('covariance trace'); a2.set_title('(b) analysis uncertainty')
a2.legend(fontsize=8); a2.grid(alpha=0.3)
fig.suptitle('EnKF on a fully linear system tracks the exact Kalman filter')
fig.tight_layout(); fig.savefig('figs/diagnostic/enkf_linear_test.png', dpi=145)
print('saved figs/diagnostic/enkf_linear_test.png')