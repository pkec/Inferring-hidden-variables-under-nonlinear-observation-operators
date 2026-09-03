
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # run from diagnostic_py/: put the project root on the import path
import figstyle as fs
if getattr(fs, 'VERSION', (0, 0)) < (5, 32):      # stale copy on the path?
    raise SystemExit(f'figstyle.py at {fs.__file__} is out of date — replace it with '
                     f'the current version and delete any __pycache__ beside it')
from config import cfg, rk4_vec
from enkf import EnKF
from enkf_ienkf import EnKF_ienkf

# ============================================================
# CHANGELOG  (newest first; version = stage.patch)
# 5.37 Added: (a)/(b)/(c) drawn into the stacked figures.
#      Changed: sizing and fonts come from figstyle, matching every other report figure.
#      Fixed:   draw() took a parameter named `fs`, which shadowed the figstyle alias inside
#      the function — renamed lab_fs. The legend in stacked() moves above the axes, where it
#      cannot sit on the histogram it is labelling.
# 5.16 Added: controlled_trio() / most_bimodal() / controlled_summary(), and the controlled
#      stacked figure now also comes out of an 'independent' run, so one invocation gives both
#      views instead of needing L63_SNAP_MODE set twice.
#      Added: the controlled figure is emitted at the most bimodal carrier forecast as well as
#      at the selected cycles — a controlled comparison only says anything where the forecast
#      actually straddles the fold, and the independent-mode cycles are picked by the IEnKF's
#      own error instead, which is a different question.
#      Added: terminal table of forecast BC, truth, and the three means per controlled cycle,
#      so the comparison is readable without opening the PNGs.
#      Changed: the existing controlled branch now builds its trio through controlled_trio()
#      rather than inline; same clouds, same colours, same order.
# 5.15 Added: stacked() — panels in one column sharing the x axis, sized 1:1 for the thesis at
#      width=0.75\textwidth. The split PNGs are 5.2in wide and were being printed at 5.44cm
#      (0.32\textwidth), a scale of 0.41, so the 14pt labels landed at 5.8pt and the 11pt
#      legend at 4.5pt. Sizing the export to the printed width fixes that at source.
#      Changed: draw() gained fs/leg_kw/bins; defaults unchanged, so the existing grids and
#      split PNGs are unaffected. The stacked figure uses 30 bins — at cfg.ensembleN=300 the
#      60-bin default leaves ~5 counts per bin, which reads as noise rather than as shape.
# 5.10 Changed: report styling to match Stage 1/2 — axis labels Arial 14pt, legends Arial 11pt
#      (AX/FS/LEG), all panel titles and suptitles dropped. Those titles carried
#      "t = … — EnKF forecast", the ONLY thing identifying a column: on the split PNGs it
#      moves into the filename, but on the combined grids the column order is now documented
#      only in the module docstring above.
#      Added: one PNG per histogram panel; the grids stay as the backup / log view
#      below it in the list. Left as-is rather than renumbering history.
# 6.9  Changed: L63_SNAP_N default 2 -> 1, so the figure is a single row.
# 5.9  Changed: L63_SNAP_N default 3 -> 2, so the figure is two rows
#      Added:   L63_SNAP_PIN (default 19.5) — a time forced into the automatic selection,
#               displacing the weakest auto pick. L63_SNAP_TIMES/CYCLES still override both.
# 3.16 Changed: figs output -> figs/diagnostic;  Added: sys.path bootstrap so project modules import when run from diagnostic_py/
# 3.14 Added:   L63_SNAP_MODE 'independent' (now the default) — each filter propagates its own
#               trajectory, 4 columns, so the trapped wrong-lobe state the IEnKF actually reaches
#               is visible; default snapshot times become the IEnKF's largest-error cycles
#      Changed: 'controlled' (shared forecast cloud) is now an opt-in mode
# 3.13 Added:   L63_SNAP_CARRIER (enkf|ienkf) — which filter propagates the trajectory
# 3.12 Changed: panel titles show model time t = obs_idx*dt instead of the cycle index
#      Added:   L63_SNAP_TIMES to pick snapshots by time (nearest cycle)
# 3.11 Added:   ensemble-mean line per panel
#      Changed: preimage lines relabelled h(x) = d
# 3.10 created — per-cycle ensemble snapshots: EnKF keeps a bimodal forecast bimodal, the IEnKF
#      locks onto a single preimage
# ============================================================

os.makedirs('figs/diagnostic', exist_ok=True)
MODE = os.environ.get('L63_SNAP_MODE', 'independent')
A_SNAP = float(os.environ.get('L63_SNAP_ALPHA', 1.0))
N_SNAP = int(os.environ.get('L63_SNAP_N', 1))
PIN_TIME = os.environ.get('L63_SNAP_PIN', '19.5')   # always shown; set empty to auto-pick both
CARRIER = os.environ.get('L63_SNAP_CARRIER', 'enkf')
fs.use()
AX, FS = fs.AX, fs.FS     # sizes come from figstyle; do not re-assert them here
slug = lambda s: s.lower().replace(' ', '_')        # panel name -> filename fragment

d = np.load('data/l63_twin.npz')
truth, oi = d['truth'], d['obs_idx']
on, osa, al = d['obs_nonlinear'], d['obs_std_alpha'], list(np.round(d['alphas'], 2))
ai = al.index(round(A_SNAP, 2))
obs, ostd = on[ai], osa[ai]
R = np.diag(ostd ** 2)
h = lambda x: x + A_SNAP * x**2
fold = -1.0 / (2 * A_SNAP)                        # h'(x)=0 here; also the midpoint of the preimages
t_obs = oi * cfg.dt

def run(fn):
    rng = np.random.default_rng(cfg.seed)
    ens = np.tile(truth[0], (cfg.ensembleN, 1)) + rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.init_std
    fc, an = [], []
    for k in range(len(oi)):
        ens += rng.normal(0, 1, (cfg.ensembleN, 3)) * cfg.perturb_std
        for _ in range(cfg.obs_every):
            ens = rk4_vec(ens, cfg.dt)
        fc.append(ens.copy())
        ens, _, _ = fn(ens, obs[k], R, h, rng)
        an.append(ens.copy())
    return fc, an

def bimodality(v):
    n = len(v)
    g, k = skew(v), kurtosis(v)
    return (g**2 + 1) / (k + 3 * (n - 1)**2 / ((n - 2) * (n - 3)))

def preimages(obs_x):
    disc = 1 + 4 * A_SNAP * obs_x
    return [] if disc < 0 else sorted([(-1 - np.sqrt(disc)) / (2 * A_SNAP),
                                       (-1 + np.sqrt(disc)) / (2 * A_SNAP)])

fc_e, an_e = run(EnKF)
if MODE == 'independent':
    fc_i, an_i = run(EnKF_ienkf)
    ie_err = np.array([abs(an_i[k][:, 0].mean() - truth[oi[k]][0]) for k in range(len(oi))])
    auto = np.argsort(ie_err)[-N_SNAP:]           # where the IEnKF's own error is worst
else:
    fc_c, _ = (fc_e, an_e) if CARRIER == 'enkf' else run(EnKF_ienkf)
    auto = np.argsort([bimodality(f[:, 0]) for f in fc_c])[-N_SNAP:]

if 'L63_SNAP_TIMES' in os.environ:
    cycles = [int(np.argmin(np.abs(t_obs - float(t)))) for t in os.environ['L63_SNAP_TIMES'].split(',')]
elif 'L63_SNAP_CYCLES' in os.environ:
    cycles = [int(c) for c in os.environ['L63_SNAP_CYCLES'].split(',')]
else:
    sel = list(auto)                                # ascending by score; sel[0] is the weakest pick
    if PIN_TIME:
        pin = int(np.argmin(np.abs(t_obs - float(PIN_TIME))))   # nearest cycle to the pinned time
        if pin not in sel:
            sel = [pin] + sel[1:]                   # displace the weakest, keeping N_SNAP rows
    cycles = sorted(sel)

def draw(ax, cloud, name, col, k, pre, show_legend=False, lab_fs=FS, leg_kw=None, bins=60):
    ax.hist(cloud[:, 0], bins=bins, color=col, alpha=0.75, edgecolor='none')
    ax.axvline(truth[oi[k]][0], color='k', lw=1.6, label='truth')
    ax.axvline(cloud[:, 0].mean(), color='#555', ls='-.', lw=1.5)
    for p in pre:
        ax.axvline(p, color='#16a085', ls='--', lw=1.2)
    ax.axvline(fold, color='#8e44ad', ls=':', lw=1.4)
    ax.set_xlabel('$x_1$', fontname=AX, fontsize=lab_fs); ax.grid(alpha=0.3)
    if show_legend:
        ax.plot([], [], '-.', color='#555', label='ensemble mean')
        ax.plot([], [], '--', color='#16a085', label=r'$h(x)=$measurement')
        ax.plot([], [], ':', color='#8e44ad', label=f'fold $x_1$={fold:.2f}')
        ax.legend(**(leg_kw or fs.LEG))


def single(cloud, name, col, k, pre, stem):

    fig, ax = plt.subplots(figsize=fs.size(0.48, 0.72))
    draw(ax, cloud, name, col, k, pre, show_legend=True)
    ax.set_ylabel('count', **fs.LAB)
    fig.tight_layout()
    o = (f'figs/diagnostic/bimodal_snapshot_{stem}_t{t_obs[k]:.2f}_{slug(name)}'
         f'_w{cfg.obs_every}_a{A_SNAP}.png')
    fig.savefig(o, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
    print(f'saved {o}')


def most_bimodal(fc, n=1):

    return sorted(int(i) for i in np.argsort([bimodality(f[:, 0]) for f in fc])[-n:])


def controlled_trio(k, carrier_fc, carrier_name):

    Af = carrier_fc[k]
    Ae, _, _ = EnKF(Af.copy(), obs[k], R, h, np.random.default_rng(99))
    Ai, _, _ = EnKF_ienkf(Af.copy(), obs[k], R, h, np.random.default_rng(99))
    return [(f'{carrier_name} forecast', '#e67e22', Af),
            ('EnKF analysis', '#c0392b', Ae),
            ('IEnKF analysis', '#2471a3', Ai)]


def controlled_summary(ctrl):

    print(f"\ncontrolled comparison — one forecast cloud, both analyses  (alpha={A_SNAP})")
    print(f"{'t':>8}{'BC(fc)':>9}{'truth':>9}{'fc mean':>10}{'EnKF':>9}{'IEnKF':>9}"
          f"{'preimages':>20}")
    for k, trio in sorted(ctrl.items()):
        pre = preimages(obs[k][0])
        ps = ', '.join(f'{p:.2f}' for p in pre) if pre else '-'
        print(f"{t_obs[k]:>8.2f}{bimodality(trio[0][2][:, 0]):>9.3f}{truth[oi[k]][0]:>9.2f}"
              f"{trio[0][2][:, 0].mean():>10.2f}{trio[1][2][:, 0].mean():>9.2f}"
              f"{trio[2][2][:, 0].mean():>9.2f}{ps:>20}")
    print("BC > 0.555 indicates a bimodal forecast; compare each analysis mean against the two"
          " preimages to see which lobe it committed to.")


def stacked(panels, k, pre, stem, frac=0.75, panel_in=1.35, bins=30):

    w = fs.TEXTWIDTH_IN * frac
    fig, ax = plt.subplots(len(panels), 1, sharex=True, squeeze=False,
                           figsize=(w, panel_in * len(panels) + 0.5))
    ax = ax[:, 0]
    leg = fs.leg_above(ncol=2)
    for i, (name, col, cloud) in enumerate(panels):
        draw(ax[i], cloud, name, col, k, pre, show_legend=(i == 0), leg_kw=leg, bins=bins)
        ax[i].set_ylabel('count', **fs.LAB)
        fs.panel_letter(ax[i], i)               # no subcaptions here to carry these
        if i != len(panels) - 1:
            ax[i].set_xlabel('')                # sharex: only the bottom panel keeps it
    fig.tight_layout(h_pad=0.35)
    o = (f'figs/diagnostic/bimodal_snapshot_stacked_{stem}_t{t_obs[k]:.2f}'
         f'_w{cfg.obs_every}_a{A_SNAP}.png')
    fig.savefig(o, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
    print(f'saved {o}')


if MODE == 'independent':
    EN = [('EnKF forecast', '#e67e22', fc_e), ('EnKF analysis', '#c0392b', an_e)]
    IE = [('IEnKF forecast', '#f1c40f', fc_i), ('IEnKF analysis', '#2471a3', an_i)]

    # --- figure 1: IEnKF alone ---
    fig, axes = plt.subplots(len(cycles), 2, figsize=(9.2, 3.2 * len(cycles)), squeeze=False)
    for ri, k in enumerate(cycles):
        pre = preimages(obs[k][0])
        for ci, (name, col, seq) in enumerate(IE):
            draw(axes[ri, ci], seq[k], name, col, k, pre, ri == 0 and ci == 0)
            single(seq[k], name, col, k, pre, 'ienkf')
        axes[ri, 0].set_ylabel(f'cycle {k}\nBC={bimodality(fc_i[k][:, 0]):.2f}\ncount',
                               **fs.LAB)
    fig.tight_layout()
    o1 = f'figs/diagnostic/bimodal_snapshot_ienkf_w{cfg.obs_every}_a{A_SNAP}.png'
    fig.savefig(o1, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)

    # --- thesis figure: IEnKF forecast above its analysis, one shared x axis ---
    for k in cycles:
        stacked([(n, c, s[k]) for n, c, s in IE], k, preimages(obs[k][0]), 'ienkf')

    # --- thesis figure, controlled variant: one forecast cloud, both analyses beneath it ---
    # Emitted from this run too, so the like-for-like comparison needs no second invocation.
    carrier_fc = fc_i if CARRIER == 'ienkf' else fc_e
    ctrl_cycles = sorted(set(cycles) | set(most_bimodal(carrier_fc)))
    ctrl = {k: controlled_trio(k, carrier_fc, CARRIER) for k in ctrl_cycles}
    for k, trio in ctrl.items():
        stacked(trio, k, preimages(obs[k][0]), 'controlled')
    controlled_summary(ctrl)

    # --- figure 2: IEnKF above, EnKF below, one block per cycle ---
    # Only the EnKF panels are split here; the IEnKF ones are the same clouds figure 1 already
    # emitted, so re-saving them under a second stem would just duplicate the PNGs.
    fig, axes = plt.subplots(2 * len(cycles), 2, figsize=(9.2, 3.2 * 2 * len(cycles)), squeeze=False)
    for ri, k in enumerate(cycles):
        pre = preimages(obs[k][0])
        for bi, block in enumerate([IE, EN]):                 # IEnKF row first, EnKF beneath
            r = 2 * ri + bi
            for ci, (name, col, seq) in enumerate(block):
                draw(axes[r, ci], seq[k], name, col, k, pre, r == 0 and ci == 0)
                if block is EN:
                    single(seq[k], name, col, k, pre, 'enkf')
            axes[r, 0].set_ylabel(f'cycle {k}\nBC={bimodality(block[0][2][k][:, 0]):.2f}\ncount',
                                  **fs.LAB)
    fig.tight_layout()
    o2 = f'figs/diagnostic/bimodal_snapshot_both_w{cfg.obs_every}_a{A_SNAP}.png'
    fig.savefig(o2, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
    outs = [o1, o2]

else:                                                          # controlled mode
    ctrl = {}
    fig, axes = plt.subplots(len(cycles), 3, figsize=(13.8, 3.2 * len(cycles)), squeeze=False)
    for ri, k in enumerate(cycles):
        pre = preimages(obs[k][0])
        trio = controlled_trio(k, fc_c, CARRIER)
        ctrl[k] = trio
        for ci, (name, col, cloud) in enumerate(trio):
            draw(axes[ri, ci], cloud, name, col, k, pre, ri == 0 and ci == 0)
            single(cloud, name, col, k, pre, 'controlled')
        stacked(trio, k, pre, 'controlled')
        axes[ri, 0].set_ylabel(f'cycle {k}\nBC={bimodality(trio[0][2][:, 0]):.2f}\ncount',
                               **fs.LAB)
    fig.tight_layout()
    o = f'figs/diagnostic/bimodal_snapshot_w{cfg.obs_every}_a{A_SNAP}_controlled.png'
    fig.savefig(o, dpi=fs.DPI, bbox_inches='tight'); plt.close(fig)
    controlled_summary(ctrl)
    outs = [o]

for o in outs:
    print(f'saved {o}')