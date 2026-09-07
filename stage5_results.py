import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg
from rls_core import rmse_percomp

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.41 Changed: a marker labels away from its NEAREST NEIGHBOUR rather than inward from the
#      axes edge. Inward is where the cluster is, so EnKF pointed at IEnKF and BS at BI.
#      Changed: the PF-floor and EnKF-baseline text sits below its dashed line, not above,
#      where it was running into the EnKF marker.
# 5.36 Changed: crowded marker labels stagger in HEIGHT above the marker rather than
#      alternating above/below. A label under its own marker reads as belonging to whatever
#      sits beneath it, which is exactly wrong in a cluster.
# 5.35 Changed: sizing and fonts come from figstyle. The 9.5in export shrank by 0.51 at
#      0.72\textwidth, taking 14pt labels to 7.1pt and the 8/8.5pt reference-line and key
#      text to about 4.2pt. Markers 11 -> 7 and mew 2 -> 1.5 to match the smaller canvas.
#      Single-axis figure, so no merged/separate split and no panel letters.
# 5.18 Changed: analysis RMSE is per-component-then-averaged via rls_core.rmse_percomp,
#      matching error_sweep.py 5.16. The stage-3 series (rmse_pf/en/qr/ie) already arrive in
#      that convention from the sweep npz, so before this patch the y-axis mixed a
#      per-component RMSE for the four classical filters with a pooled one for the four RLS
#      modes — the Pareto plot was comparing points measured two different ways.
# 5.8  Changed: report styling to match Stage 1/2 — axis labels Arial 14pt via the AX/FS
#      constants, title dropped (Imperial reports caption externally). The n/p/T/N_e/N_pf line
#      the title carried moves into the key box, so the parameters stay on the figure.
#      Single-axis figure, so there is nothing to split into separate PNGs.
# 5.7  Changed: markers labelled OS / OI / BS / BI with a key box top right, so the four
#      RLS variants fit beside each other; reference-line labels moved to the left edge to
#      clear it, and the hollow-marker note moved out of the title into the key
# 5.6  Changed: y-axis is analysis RMSE, not percentage points of EnKF excess removed. Under
#      the old axis the EnKF sat at 0 and the PF at the EnKF's full excess by construction,
#      so two markers carried no measurement and a level was plotted against a difference.
#      Excess over the PF is kept in the printout.
#      Removed: the two-baseline caveat. rls_blind.py scores on cfg.seeds (it only TRAINS on
#      cfg.blind_seeds), and stage4_log.py uses error_sweep.py's calibration and noise recipe,
#      so both stages read one reference pair. Replaced by a numerical check of that claim.
# 5.5  created — cost (dominant c_a only, Table tab:cost) against percentage points of EnKF
#      excess removed, one marker per filter, plus a printed cost derivation table and the
#      blind-vs-live and blind-vs-PF break-even run counts
# ============================================================

os.makedirs('figs/results', exist_ok=True)
ALPHA = float(os.environ.get('L63_COST_ALPHA', 0.6))
fs.use()
FRAC = 0.72                                             # \includegraphics width in the thesis

# ---------------- parameters (Table tab:params) ----------------
# Derived from cfg wherever cfg is the source of truth, so the cost model cannot drift from
# the code it is pricing. Only n_it is measured rather than configured.
n = 3                                                   # state dimension (L63)
p = 3                                                   # observation dimension (all components observed)
T = cfg.obs_every                                       # model steps per assimilation window
n_c = len(np.arange(cfg.obs_every, cfg.n_steps + 1, cfg.obs_every))   # assimilation cycles per run
N_e = cfg.ensembleN                                     # EnKF-family members
N_pf = cfg.particleN                                    # particles
m_q = 2 * p + 1                                         # QR-EnKF design-matrix columns [1, y, y^2]
K = 8 * n + 2                                           # RLS feature count (rls_core.n_features)
n_it = float(os.environ.get('L63_N_IT', 3))             # mean Gauss-Newton iterations (cap 20)
n_tr = int(os.environ.get('L63_N_TR', 3))               # training runs charged to the blind modes
r = int(os.environ.get('L63_R', 3))                     # deployment runs priced by C(r)

LOGN = np.log2(N_pf)   # the log N_pf of systematic resampling; base 2 = binary-search comparisons

# ---------------- dominant c_a, one line per Table tab:cost row ----------------
# c_a = N(T*s(n) + a(n,p)) + b(n,p) with s(n) = n. Each line below is the *dominant c_a at
# p = n* column read straight off the table, not a re-derivation from its a and b columns.
ca_en = N_e * (T * n + n**2) + n**3                     # EnKF
ca_pf = N_pf * (T * n + LOGN)                           # PF
ca_qr = N_e * (T * n + m_q**2) + m_q**3                 # QR-EnKF
A_en = N_e * n**2 + n**3                                # analysis-only part of ca_en
ca_ie = N_e * T * n + n_it * A_en                       # IEnKF
ca_sh = ca_en + ca_pf                                   # RLS shadow: EnKF and PF both propagate
ca_inj = ca_en + ca_pf                                  # live inject: same two ensembles
# Blind modes deploy the EnKF alone; the PF is only present while training.
ca_bsh = ca_en
ca_binj = ca_en

# (label, N description, symbolic c_a, substituted c_a, c_a value, C0 symbolic, C0 value, colour)
FILTERS = [
    ('PF',           f'N_pf={N_pf}',
     'N_pf*(T*n + log2 N_pf)',            f'{N_pf}*({T}*{n} + {LOGN:.2f})',
     ca_pf, '0', 0.0, '#16a085'),
    ('EnKF',         f'N_e={N_e}',
     'N_e*(T*n + n^2) + n^3',             f'{N_e}*({T}*{n} + {n}^2) + {n}^3',
     ca_en, '0', 0.0, '#c0392b'),
    ('QR-EnKF',      f'N_e={N_e}',
     'N_e*(T*n + m_q^2) + m_q^3',         f'{N_e}*({T}*{n} + {m_q}^2) + {m_q}^3',
     ca_qr, '0', 0.0, '#e67e22'),
    ('IEnKF',        f'N_e={N_e}',
     'N_e*T*n + n_it*(N_e*n^2 + n^3)',    f'{N_e}*{T}*{n} + {n_it:g}*({N_e}*{n}^2 + {n}^3)',
     ca_ie, '0', 0.0, '#2471a3'),
    ('RLS shadow',   f'N_e={N_e}, N_pf={N_pf}',
     'ca_en + ca_pf',                     f'{ca_en:.0f} + {ca_pf:.0f}',
     ca_sh, '0', 0.0, '#0bcd2c'),
    ('Live inject',  f'N_e={N_e}, N_pf={N_pf}',
     'ca_en + ca_pf',                     f'{ca_en:.0f} + {ca_pf:.0f}',
     ca_inj, '0', 0.0, '#b73dcf'),
    ('Blind shadow', f'N_e={N_e}',
     'ca_en',                             f'{ca_en:.0f}',
     ca_bsh, 'n_tr*n_c*ca_sh', n_tr * n_c * ca_sh, '#1B613A'),
    ('Blind inject', f'N_e={N_e}',
     'ca_en',                             f'{ca_en:.0f}',
     ca_binj, 'n_tr*n_c*ca_inj', n_tr * n_c * ca_inj, '#920070'),
]

cost = {lbl: C0 + r * n_c * ca for lbl, _, _, _, ca, _, C0, _ in FILTERS}   # C(r), flops

# ---------------- printed derivation, so the structure can be checked ----------------
print(f'\nparameters: n={n} p={p} T={T} n_c={n_c} N_e={N_e} N_pf={N_pf} '
      f'm_q={m_q} K={K} n_it={n_it:g} n_tr={n_tr} r={r}')
print('cost model: C(r) = C0 + r*n_c*c_a, dominant c_a only, all O() constants = 1\n')
for lbl, Nd, sym, sub, ca, C0sym, C0, _ in FILTERS:
    print(f'{lbl}   [{Nd}]')
    print(f'    c_a  = {sym:<34} = {sub:<34} = {ca:,.0f}')
    print(f'    C0   = {C0sym:<34} = {C0:,.0f}')
    print(f'    C(r) = C0 + {r}*{n_c}*c_a{"":<20} = {cost[lbl]:,.0f}\n')

# Break-even run counts. Blind pays n_tr training runs up front to deploy without the PF;
# these are the r at which that trade turns positive against the two things it replaces.
# n_c cancels: n_tr*n_c*ca_sh + r*n_c*ca_en = r*n_c*ca_rival  ->  r = n_tr*ca_sh/(ca_rival - ca_en)
for rival, ca_rival in (('live shadow/inject', ca_sh), ('the PF', ca_pf)):
    gain = ca_rival - ca_en                    # flops per cycle saved by dropping the PF
    if gain > 0:
        print(f'break-even vs {rival}: r = n_tr*ca_sh/(ca_rival - ca_en) '
              f'= {n_tr * ca_sh / gain:.1f} runs')

rmse_of, provenance = {}, {}
ref = {}                                                   # shared EnKF baseline and PF floor
rmse = rmse_percomp                                        # per component, then averaged (5.18)

W, mode, jit = cfg.obs_every, cfg.noise_mode, cfg.jitter_mode
f3 = f'data/stage2_results_w{W}_{mode}_{jit}.npz'
if os.path.exists(f3):
    d3 = np.load(f3)
    ai = int(np.argmin(np.abs(d3['alphas'] - ALPHA)))
    for lbl, key in (('PF', 'rmse_pf'), ('EnKF', 'rmse_en'),
                     ('QR-EnKF', 'rmse_qr'), ('IEnKF', 'rmse_ie')):
        if key in d3.files:
            rmse_of[lbl] = float(d3[key][ai]); provenance[lbl] = 'stage 3'
    ref = {'EnKF': rmse_of.get('EnKF'), 'PF': rmse_of.get('PF')}
else:
    print(f'\n{f3} not found — PF / EnKF / QR-EnKF / IEnKF skipped')

MODE_LABEL = {'shadow': 'RLS shadow', 'inject': 'Live inject',
              'blind_shadow': 'Blind shadow', 'blind_inject': 'Blind inject'}
base4, pf4 = [], []                                        # stage 4's own view of the reference pair
for path in sorted(glob.glob('data/stage4_*_a*.npz')):
    name = path.replace('\\', '/').split('/')[-1]          # glob gives backslashes on Windows
    m = re.match(r'stage4_(\w+)_a([\d.]+)\.npz$', name)
    if not m or m.group(1) not in MODE_LABEL or abs(float(m.group(2)) - ALPHA) > 1e-9:
        continue
    d4 = np.load(path, allow_pickle=True)
    if 'pred' not in d4.files:                             # not a result file (schema guard)
        continue
    rmse_of[MODE_LABEL[m.group(1)]] = rmse(d4['xa_corr'], d4['truth'])
    provenance[MODE_LABEL[m.group(1)]] = 'stage 4'
    base4.append(rmse(d4['xa_base'], d4['truth']))
    pf4.append(rmse(d4['xpf_mean'], d4['truth']))

# Both sides should be the same runs. A few tenths of a percent is still expected: stage 3
# averages per-seed RMSEs, while stage 4 concatenates the seeds and takes one RMSE over the
# lot. Since 5.18 both use the same per-component-then-averaged rule, so the aggregation
# convention no longer contributes to the gap — only the seed-averaging order does.
TOL = 2.0                                                  # % relative difference tolerated
if base4 and ref.get('EnKF'):
    print('\nreference pair, stage 3 vs stage 4 (should be the same runs):')
    for nm, got, want in (('EnKF baseline', float(np.mean(base4)), ref['EnKF']),
                          ('PF floor', float(np.mean(pf4)), ref['PF'])):
        rel = abs(got - want) / want * 100
        print(f'  {nm:<14} stage 4 = {got:.4f}   stage 3 = {want:.4f}   ({rel:.2f}%)'
              + ('' if rel < TOL else '   <-- MISMATCH, not a common axis'))

print(f'\nanalysis RMSE at alpha={ALPHA}  (lower is better)')
for lbl, _, _, _, _, _, _, _ in FILTERS:
    v = rmse_of.get(lbl)
    if v is None:
        print(f'  {lbl:<14}{"n/a":>10}   [-]'); continue
    # excess over the PF floor kept in the printout: it is Chapter 4's vocabulary, and the
    # figure no longer shows it
    exc = (v - ref['PF']) / ref['PF'] * 100 if ref.get('PF') else float('nan')
    print(f'  {lbl:<14}{v:10.4f}   [{provenance[lbl]}]   excess over PF {exc:+6.1f}%')

# ---------------- figure ----------------
SHORT = {'RLS shadow': ('OS', 'online shadow'), 'Live inject': ('OI', 'online inject'),
         'Blind shadow': ('BS', 'blind shadow'), 'Blind inject': ('BI', 'blind inject')}

fig, ax = plt.subplots(figsize=fs.size(FRAC, 0.68))
for lvl, colr, nm in ((ref.get('PF'), '#16a085', 'PF floor'),
                      (ref.get('EnKF'), '#c0392b', 'EnKF baseline')):
    if lvl:
        ax.axhline(lvl, ls='--', color=colr, lw=1)
        ax.text(0.005, lvl, f' {nm}', transform=ax.get_yaxis_transform(),   # left edge: the
                ha='left', va='top', color=colr, **fs.ANN)                  # key box is right
plotted = []
for lbl, _, _, _, _, _, C0, colr in FILTERS:
    if lbl in rmse_of:
        plotted.append((lbl, rmse_of[lbl], cost[lbl], colr, C0))
if not plotted:
    raise SystemExit('no accuracy data found at this alpha — run error_sweep.py / rls_*.py')
cs = [c for _, _, c, _, _ in plotted]
xlo, xhi = min(cs) / 1.6, max(cs) * 1.6                    # log-space padding for the labels
mid = np.sqrt(xlo * xhi)
# The blind modes deploy the EnKF alone (ca_bsh = ca_binj = ca_en), so EnKF / BS / BI differ
# in C(r) only by the constant C0 and land on almost the same x. Alternating the label offset
# above/below separates them; without it the text overlaps whatever the data happens to be.
xs_log = np.log10([c for _, _, c, _, _ in plotted])
span = max(xs_log.max() - xs_log.min(), 1e-9)
order = np.argsort(xs_log)
dy, level, prev = {}, 0, None
for i in order:
    close = prev is not None and (xs_log[i] - prev) < 0.08 * span
    # Stagger the HEIGHT, not the side: every label sits above its marker, and neighbours
    # that would otherwise collide are pushed to a second level instead of being flipped
    # underneath. A label below its marker reads as belonging to whatever is beneath it.
    level = (level + 1) % 2 if close else 0
    dy[i] = 6 + 1 * level
    prev = xs_log[i]

place_left = {}
for i in range(len(plotted)):
    others = [j for j in range(len(plotted)) if j != i]
    if others:
        nb = min(others, key=lambda j: abs(xs_log[j] - xs_log[i]))
        place_left[i] = xs_log[nb] > xs_log[i]     # neighbour to the right -> label leftward
    else:
        place_left[i] = xs_log[i] > np.log10(mid)

for i, (lbl, y, c, colr, C0) in enumerate(plotted):
    # hollow marker = the filter carries a training charge C0 that r never amortises away
    ax.plot(c, y, 'o', ms=7, color=colr, mfc='none' if C0 > 0 else colr, mew=1.5)
    left = place_left[i]
    ax.annotate(SHORT.get(lbl, (lbl,))[0], (c, y), textcoords='offset points',
                xytext=(-7 if left else 7, dy[i]),
                ha='right' if left else 'left', va='bottom',
                color=colr, **fs.LAB)
ax.set_xscale('log'); ax.set_xlim(xlo, xhi); ax.margins(y=0.16)
ax.set_xlabel(f'C(r) for r={r} runs  (dominant terms only)', **fs.LAB)
ax.set_ylabel(f'analysis RMSE at $\\alpha$={ALPHA}', **fs.LAB)
ax.grid(alpha=0.3, which='both'); ax.set_axisbelow(True)

# key box carries the marker abbreviations and, since the title is gone, the parameter set
key = [f'{s} = {full}' for lbl, (s, full) in SHORT.items() if lbl in rmse_of]
# Put the key in whichever corner holds the fewest markers, rather than pinning it top-right
# where it collided with the online-mode cluster.
ys = np.array([y for _, y, _, _, _ in plotted], float)
xn = (xs_log - xs_log.min()) / span
yn = (ys - ys.min()) / max(ys.max() - ys.min(), 1e-9)
# Clearance, not a quadrant count: a quadrant can hold one marker sitting right in the
# corner and still win on count, which is how the key ended up on top of the PF point.
# Pick the corner whose nearest marker is furthest away.
CORNERS = [(0.985, 0.975, 'right', 'top'), (0.015, 0.975, 'left', 'top'),
           (0.985, 0.025, 'right', 'bottom'), (0.015, 0.025, 'left', 'bottom')]
kx, ky, kha, kva = max(CORNERS, key=lambda c: np.hypot(xn - c[0], yn - c[1]).min())
ax.text(kx, ky, '\n'.join(key), transform=ax.transAxes, ha=kha, va=kva,
        zorder=5, linespacing=1.35, **fs.ANN,
        bbox=dict(boxstyle='round,pad=0.35', fc='white', ec='#bbb', alpha=0.92))
fs.save(fig, f'figs/results/stage5_cost_benefit_a{ALPHA}.png')