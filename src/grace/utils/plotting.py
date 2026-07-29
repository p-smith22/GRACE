# Import packages:
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# Set settings:
matplotlib.use("Agg")

# State and control colors for a consistent look:
STATE_COLOR = "#2c6fbb"
CONTROL_COLOR = "#c0392b"

# Plot a trajectory, its states, and its controls as individual subplots:
def plotting(system, trajectory, control, dt=None, obstacles=None, R=None,
             pos_idx=(0, 1), state_names=None, control_names=None,
             title="States and controls", save=None, ncols=None, show_traj=True):

    # Default the time step from the system:
    dt = system.dt if dt is None else dt

    # Reshape the trajectory and control:
    Z = np.asarray(trajectory)
    nu = system.nu
    U = np.asarray(control).reshape(-1, nu)
    nx = Z.shape[1]
    pi = list(pos_idx)

    # Build the time axes for states and piecewise controls:
    t = np.arange(len(Z)) * dt
    tu = (np.arange(len(U)) + 0.5) * dt

    # Lay out the panel grid, with an optional trajectory panel first:
    panels = nx + nu + (1 if show_traj else 0)
    ncols = ncols or (4 if panels >= 4 else panels)
    nrows = int(np.ceil(panels / ncols))
    fig = plt.figure(figsize=(3.6 * ncols, 2.7 * nrows))
    gs = fig.add_gridspec(nrows, ncols)
    off = 0

    # Draw the optional trajectory panel with obstacles:
    if show_traj:
        axt = fig.add_subplot(gs[0, 0])
        off = 1
        if obstacles is not None and R is not None:
            th = np.linspace(0, 2 * np.pi, 60)
            for o in obstacles:
                o = np.asarray(o, float)
                axt.fill(o[0] + R * np.cos(th), o[1] + R * np.sin(th), color="crimson", alpha=0.15)
                axt.plot(o[0] + R * np.cos(th), o[1] + R * np.sin(th), "crimson", lw=1.2)
        axt.plot(Z[:, pi[0]], Z[:, pi[1]], lw=2, color="seagreen")
        axt.scatter([Z[0, pi[0]]], [Z[0, pi[1]]], c="k", s=45, zorder=5)
        axt.scatter([Z[-1, pi[0]]], [Z[-1, pi[1]]], c="seagreen", marker="*", s=120, zorder=5)
        axt.set_title("trajectory", fontsize=10)
        axt.set_aspect("equal")
        axt.grid(alpha=0.3)
        axt.set_xlabel(state_names[pi[0]] if state_names else f"x{pi[0]}")
        axt.set_ylabel(state_names[pi[1]] if state_names else f"x{pi[1]}")

    # Draw each state and control in its own panel, sharing the time axis:
    axes = []
    first_time_ax = None
    for n in range(nx + nu):
        slot = off + n
        r, c = divmod(slot, ncols)
        ax = fig.add_subplot(gs[r, c], sharex=first_time_ax)
        if first_time_ax is None:
            first_time_ax = ax
        axes.append(ax)

        # States on the left family, controls on the right family:
        if n < nx:
            ax.plot(t, Z[:, n], lw=1.8, color=STATE_COLOR)
            ax.set_title(state_names[n] if state_names else f"x{n}", fontsize=10)
        else:
            j = n - nx
            ax.step(tu, U[:, j], where="mid", lw=1.8, color=CONTROL_COLOR)
            ax.set_title(control_names[j] if control_names else f"u{j}", fontsize=10)
        ax.grid(alpha=0.3)
        ax.axhline(0, color="gray", lw=0.6, alpha=0.5)

    # Label the lowest time panel in each column:
    for c in range(ncols):
        lowest = None
        for n in range(nx + nu):
            slot = off + n
            r, cc = divmod(slot, ncols)
            if cc == c:
                lowest = axes[n]
        if lowest is not None:
            lowest.set_xlabel("t")

    # Add a small legend distinguishing states from controls:
    from matplotlib.lines import Line2D
    fig.legend([Line2D([0], [0], color=STATE_COLOR, lw=2),
                Line2D([0], [0], color=CONTROL_COLOR, lw=2)],
               ["state", "control"], loc="upper right", fontsize=9, frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    # Save or return the figure:
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
        return save
    return fig


# Overlay the nominal, open-loop, and closed-loop trajectories under disturbance:
def closed_loop(nominal, open_loop, closed, dt=1.0, comp_idx=0, comp_name=None,
                title="Closed-loop tracking", save=None):

    # Build the time axis:
    t = np.arange(len(nominal)) * dt

    # Overlay the three trajectories for one state component:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(t, np.asarray(nominal)[:, comp_idx], "k--", lw=1.5, label="nominal")
    ax.plot(t, np.asarray(open_loop)[:, comp_idx], "firebrick", lw=2, label="open-loop (disturbed)")
    ax.plot(t, np.asarray(closed)[:, comp_idx], "seagreen", lw=2, label="closed-loop (disturbed)")
    ax.set_xlabel("t")
    ax.set_ylabel(comp_name or f"state[{comp_idx}]")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    # Save or return the figure:
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
        return save
    return fig


# Draw the four-metric comparison panel: time, cost, endpoint error, stationarity:
def compare_panel(metrics, labels=("GRACE", "optimizer"), title="GRACE vs optimizer",
                  save=None, colors=("seagreen", "firebrick")):

    # Each panel is one metric, with error and stationarity on a log scale:
    panels = [("time_ms", "compute time (ms)", False),
              ("cost", "control cost", False),
              ("endpoint_err", "endpoint error", True),
              ("stationarity", "stationarity", True)]
    fig, ax = plt.subplots(1, 4, figsize=(16, 4))
    for i, (key, label, logy) in enumerate(panels):
        vals = metrics[key]
        bars = ax[i].bar(labels, vals, color=colors, alpha=0.85)

        # Use a log scale for the small-magnitude metrics:
        if logy:
            ax[i].set_yscale("log")

        # Annotate each bar with its value:
        for b, v in zip(bars, vals):
            ax[i].text(b.get_x() + b.get_width() / 2, v,
                       (f"{v:.2e}" if logy else f"{v:.1f}"),
                       ha="center", va="bottom", fontsize=9)
        ax[i].set_title(label)
        ax[i].grid(alpha=0.3, axis="y")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()

    # Save or return the figure:
    if save:
        fig.savefig(save, dpi=110, bbox_inches="tight")
        return save
    return fig
