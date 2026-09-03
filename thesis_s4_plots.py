"""
Combine the Stage 4 shadow residual-trace thesis panels for alpha=0 and alpha=0.8.

Creates one two-panel figure:
    (a) alpha = 0.0
    (b) alpha = 0.8

The panel labels (a)/(b) are placed at the top-left of each axes.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import figstyle as fs


# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
S4 = "figs/diagnostic/s4"

ALPHAS = [0.0, 0.8]
THESIS_SEED = 0
THESIS_TMAX = 60.0
TRACE_COMP = 0

LABELS = ["x", "y", "z"]

TRUEC = "#333333"
SHADOWC = "#e67e22"

AX, FS = fs.AX, fs.FS
fs.use()


# ---------------------------------------------------------------------
# Load one shadow result
# ---------------------------------------------------------------------
def load_shadow(alpha):
    path = f"data/stage4_shadow_a{alpha}.npz"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing result file: {path}")

    d = np.load(path, allow_pickle=True)

    required = {"times", "target", "pred"}
    missing = required.difference(d.files)
    if missing:
        raise KeyError(f"{path} is missing: {sorted(missing)}")

    return d


# ---------------------------------------------------------------------
# Draw one thesis panel
# ---------------------------------------------------------------------
def draw_panel(ax, alpha, panel_label):
    d = load_shadow(alpha)

    t = d["times"]
    target = d["target"]
    pred = d["pred"]

    if "row_seed" not in d.files:
        raise KeyError(
            f"stage4_shadow_a{alpha}.npz has no 'row_seed'; "
            "cannot select THESIS_SEED={THESIS_SEED}"
        )

    row_seed = d["row_seed"]
    sel = np.where(row_seed == THESIS_SEED)[0]

    if len(sel) < 2:
        raise ValueError(
            f"No usable rows found for seed {THESIS_SEED} at alpha={alpha}"
        )

    # Restart this seed's clock at zero, matching thesis_trace() in
    # stage4_results.py.
    tv = t[sel] - t[sel][0]
    keep = tv <= THESIS_TMAX if THESIS_TMAX else np.ones(len(sel), dtype=bool)

    idx = sel[keep]
    tw = tv[keep]

    if len(idx) < 2:
        raise ValueError(
            f"Fewer than two points remain for seed {THESIS_SEED}, "
            f"alpha={alpha}, THESIS_TMAX={THESIS_TMAX}"
        )

    c_win = np.corrcoef(
        target[idx, TRACE_COMP],
        pred[idx, TRACE_COMP],
    )[0, 1]

    c_all = np.corrcoef(
        target[sel, TRACE_COMP],
        pred[sel, TRACE_COMP],
    )[0, 1]

    ax.plot(
        tw,
        target[idx, TRACE_COMP],
        "-",
        color=TRUEC,
        lw=0.8,
        label=r"true  $r_k$",
    )

    ax.plot(
        tw,
        pred[idx, TRACE_COMP],
        "-",
        color=SHADOWC,
        lw=1.1,
        label=r"learnt  $w_{k-1}^{\top}u_k$",
    )

    ax.set_xlabel("Time", **fs.LAB)
    ax.set_ylabel(LABELS[TRACE_COMP], **fs.LAB)
    ax.grid(alpha=0.3)

    # Requested panel label: (a), (b), ... at the top-left.
    ax.text(
        0.02,
        0.96,
        panel_label,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=14,
        fontname=AX,
        va="top",
        ha="left",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec="none",
            alpha=0.7,
        ),
    )

    # Alpha identifier at top-right.
    ax.text(
        0.98,
        0.96,
        rf"$\alpha={alpha:g}$",
        transform=ax.transAxes,
        fontname=AX,
        fontsize=FS,
        va="top",
        ha="right",
        bbox=dict(
            boxstyle="round,pad=0.2",
            fc="white",
            ec="none",
            alpha=0.7,
        ),
    )

    # Keep the correlation annotation from obscuring the traces.
    ax.text(
        0.99,
        0.03,
        f"corr {c_win:+.3f} shown / {c_all:+.3f} full",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        **fs.ANN,
        bbox=dict(
            boxstyle="round,pad=0.25",
            fc="white",
            ec="#bbb",
            alpha=0.9,
        ),
    )


# ---------------------------------------------------------------------
# Combined figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1,
    2,
    figsize=fs.size(0.98, fs.SERIES),
    sharey=True,
)

for i, (ax, alpha) in enumerate(zip(axes, ALPHAS)):
    draw_panel(
        ax,
        alpha,
        panel_label=f"({chr(97 + i)})",
    )

# One shared legend for the two panels.
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    **fs.LEG,
)

fig.tight_layout(rect=(0, 0, 1, 0.90))

os.makedirs(S4, exist_ok=True)

out = f"{S4}/stage4_residual_trace_shadow_thesis_combined.png"
fs.save(fig, out)

print(f"saved {out}")
