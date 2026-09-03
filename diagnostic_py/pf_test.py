import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

mu_p, sigma_p = -0.5, 0.5
a, b = 1.0, 0.0
sigma_obs = 0.5
R = sigma_obs**2
y_obs = 0.5

N = 10000

#Gaussian
x = rng.normal(mu_p, sigma_p, size=N)

#Bimodal
#mu1, mu2 = -1, 2
#sigma1, sigma2 = 0.5, 0.5
#mix = 0.5  # probability of picking component 1

#mask = rng.uniform(size=N) < mix
#x = np.where(mask, rng.normal(mu1, sigma1, N), rng.normal(mu2, sigma2, N))


#Linear obs op
#y_pred = a * x + b

#Nonlinear obs op
alpha = 1
y_pred = x + alpha * x**2


innov = y_obs - y_pred
log_w = -0.5 * innov**2 / R
log_w -= log_w.max()
w = np.exp(log_w)
w /= w.sum()

mu_post = np.sum(w * x)
var_post = np.sum(w * (x - mu_post)**2)

# systematic resampling
positions = (rng.uniform() + np.arange(N)) / N
cumw = np.cumsum(w)
idx = np.searchsorted(cumw, positions)
x_rs = x[idx]

print(f"PF  posterior (weighted):  mean = {mu_post:.4f},  var = {var_post:.4f}")
print(f"PF  posterior (resampled): mean = {x_rs.mean():.4f},  var = {x_rs.var():.4f}")
print(f"unique particles after resampling: {len(np.unique(x_rs))} / {N}")

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

ax1.hist(x, bins=50, density=True, color='k', alpha=0.7)
ax1.set_ylabel('density'); ax1.set_title('Prior')

ax2.bar(x, w, width=0.04, color='C1')
ax2.axvline(y_obs, color='b', ls='--', alpha=0.5, label=f'y_obs = {y_obs}')
ax2.set_ylabel('weight'); ax2.set_title('Posterior (weighted particles)')
ax2.legend()

ax3.hist(x_rs, bins=50, density=True, color='C2', alpha=0.7)
ax3.axvline(y_obs, color='b', ls='--', alpha=0.5)
ax3.set_ylabel('density'); ax3.set_xlabel('x')
ax3.set_title('Posterior (after resampling, equal weights)')

plt.tight_layout(); plt.show()