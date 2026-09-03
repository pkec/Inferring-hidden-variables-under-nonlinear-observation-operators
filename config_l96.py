import numpy as np
from types import SimpleNamespace

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.1  Added:   perturb_std - roughening jitter. Resampling a deterministic model only
#               removes distinct particles, so diversity ratchets down and never recovers.
# 5.0  Stage 5 - Lorenz-96 PF feasibility check
#   created
# ============================================================

cfg = SimpleNamespace(
    F=8.0,              # forcing; F=8 is the standard chaotic setting
    n_list=[20, 30, 40],   # state dimensions to sweep
    N_list=[1000, 10000, 30000],  # particle counts to sweep
    dt=0.01,
    obs_every=5,        # model steps between observations -> dt_obs = 0.05
    n_cycles=200,
    obs_stride=1,       # 1 = observe every variable, so obs dimension p = n
    obs_std=1.0,
    alpha=0.0,          # nonlinearity in h(x) = x + alpha*x^2
    init_std=1.0,
    perturb_std=0.05,   # jitter added each cycle so resampled duplicates do not stay identical
    spinup=2000,
    seed=3,
)
cfg.n_steps = cfg.n_cycles * cfg.obs_every


def l96(x, F):
    """dx_i/dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F, cyclic in i."""
    # roll acts on the last axis, so this works for a single state (n,) and an ensemble (N, n)
    return (np.roll(x, -1, axis=-1) - np.roll(x, 2, axis=-1)) * np.roll(x, 1, axis=-1) - x + F


def rk4_vec(x, dt, F=cfg.F):
    k1 = l96(x, F)
    k2 = l96(x + 0.5 * dt * k1, F)
    k3 = l96(x + 0.5 * dt * k2, F)
    k4 = l96(x + dt * k3, F)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)