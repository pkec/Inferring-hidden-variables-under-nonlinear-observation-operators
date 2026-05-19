import numpy as np
from dataclasses import dataclass, field

@dataclass
class L63Config:
    # Lorenz parameters
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0

    # time stepping
    dt: float = 0.01
    n_steps: int = 8000

    # observation settings
    obs_every: int = 5

    # --- per-axis noise (10% of attractor range) ---
    # x ~ [-20, 20] range 40 → std 4.0
    # y ~ [-25, 25] range 50 → std 5.0
    # z ~ [  5, 48] range 43 → std 4.3
    obs_std: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.5, 2.15]))
    init_std: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.5, 2.15]))


    perturb_std: np.ndarray = field(default_factory=lambda: np.array([0.01, 0.0125, 0.01075])) #0.5% of obs_std
    # observation error covariance (diagonal), derived from obs_std
    @property
    def R(self):
        return np.diag(self.obs_std ** 2)

    # reproducibility
    seed: int = 0

    # particle filter
    ensembleN: int = 800

cfg = L63Config()


# --- shared model functions ---

def lorenz63(s):
    x, y, z = s
    return np.array([
        cfg.sigma * (y - x),
        x * (cfg.rho - z) - y,
        x * y - cfg.beta * z
    ])

def rk4(s, dt):
    k1 = lorenz63(s)
    k2 = lorenz63(s + 0.5 * dt * k1)
    k3 = lorenz63(s + 0.5 * dt * k2)
    k4 = lorenz63(s + dt * k3)
    return s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)