# Keep window up to TMAX (60)
TMAX = 60

...

# Create a 2-column layout:
# (a) (b)
#   (c)
n_groups = len(groups)
ncols = 2
nrows = int(np.ceil(n_groups / ncols))

fig = plt.figure(figsize=(8, 2.5 * nrows))
gs = fig.add_gridspec(nrows, ncols)

axes = []

for i in range(n_groups):
    if i < 2:
        # (a), (b) on the first row
        ax = fig.add_subplot(gs[0, i])
    else:
        # (c) centred on the second row
        ax = fig.add_subplot(gs[1, :])

    axes.append(ax)

for i, (sd, idx) in enumerate(groups):
    ax = axes[i]

    # Reset clock for this seed (from 0)
    tv = t[idx] - t[idx][0]

    ax.plot(
        tv, target[idx, TRACE_COMP], '-',
        color=TRUEC, lw=1.0, label=r'true  $r_k$'
    )
    ax.plot(
        tv, pred[idx, TRACE_COMP], '-',
        color=MODE_COLOUR, lw=1.4,
        label=r'learnt  $w_{k-1}^{\top}u_k$'
    )

    ax.set_ylabel(
        LABELS[TRACE_COMP],
        fontname=AX,
        fontsize=FS_LAB
    )

    # Only bottom panel gets the x-axis label
    if i == 2 or (i == n_groups - 1 and n_groups < 3):
        ax.set_xlabel('Time', fontname=AX, fontsize=FS_LAB)

    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', labelsize=FS_TICK)

    # Panel label
    panel_label = f'({chr(97 + i)})'
    ax.text(
        0.02, 0.96, panel_label,
        transform=ax.transAxes,
        fontweight='bold',
        fontsize=14,
        fontname=AX,
        va='top',
        ha='left',
        bbox=dict(
            boxstyle='round,pad=0.2',
            fc='white',
            ec='none',
            alpha=0.7
        )
    )

    # Correlation
    c_win = np.corrcoef(
        target[idx, TRACE_COMP],
        pred[idx, TRACE_COMP]
    )[0, 1]

    ax.text(
        0.99, 0.04,
        f'corr {c_win:+.3f}',
        transform=ax.transAxes,
        ha='right',
        va='bottom',
        fontname=AX,
        fontsize=FS_ANN,
        bbox=dict(
            boxstyle='round,pad=0.25',
            fc='white',
            ec='#bbb',
            alpha=0.9
        )
    )

    # Legend only on top-left panel
    if i == 0:
        ax.legend(
            loc='upper right',
            ncol=2,
            prop={'family': AX, 'size': FS_LEG}
        )

fig.tight_layout()
