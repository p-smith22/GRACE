# Import packages:
import numpy as np
import casadi as ca
import os
import grace
import time
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, Circle
from matplotlib.transforms import Affine2D


# Shared implementation -- power-limited drive force, quadratic drag, bicycle steering:
def car(x, u):
    m = 1500.0
    P = 8000.0
    cd = 0.4
    L = 2.5
    delta_max = 0.6

    v = x[3]

    # Saturated steering angle:
    delta = delta_max * ca.tanh(u[1] / delta_max)

    F_drive = P * u[0] / (ca.fabs(v) + 1.0)
    F_drag = cd * v * ca.fabs(v)

    return ca.vertcat(
        v * ca.cos(x[2]),
        v * ca.sin(x[2]),
        v / L * ca.tan(delta),
        (F_drive - F_drag) / m,
    )


# Animate obstacle-avoidance trajectory:
def animate_car_obstacle(
    Z,
    U,
    dt,
    obstacles,
    R_obs,
    target,
    save="car_obstacle.gif",
):

    # ------------------------------------------------------------
    # Convert data
    # ------------------------------------------------------------

    Z = np.asarray(Z, dtype=float)
    U = np.asarray(U, dtype=float)

    # Standardize Z -> (N+1, nx)
    if Z.shape[0] == 4:
        Z = Z.T

    # Number of control intervals:
    N = Z.shape[0] - 1

    # Standardize U -> (N, 2)
    if U.ndim == 1:
        U = U.reshape(N, 2)

    elif U.shape == (2, N):
        U = U.T

    elif U.shape != (N, 2):
        raise ValueError(
            f"Unexpected control shape {U.shape}; "
            f"expected ({2*N},), ({N}, 2), or (2, {N})"
        )

    # ------------------------------------------------------------
    # Time
    # ------------------------------------------------------------

    t_state = np.arange(N + 1) * dt
    t_control = np.arange(N) * dt

    duration = t_state[-1]

    # Animation settings:
    fps = 30
    playback_scale = 1.5

    n_frames = max(
        int(duration * fps * playback_scale),
        N + 1,
    )

    # THIS MUST EXIST BEFORE THE INTERPOLATION BELOW:
    t_anim = np.linspace(
        0.0,
        duration,
        n_frames,
    )

    # ------------------------------------------------------------
    # Interpolate states for smooth animation
    # ------------------------------------------------------------

    Z_anim = np.zeros(
        (n_frames, Z.shape[1])
    )

    for j in range(Z.shape[1]):
        Z_anim[:, j] = np.interp(
            t_anim,
            t_state,
            Z[:, j],
        )

    # ------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------

    throttle = U[:, 0]
    steer_raw = U[:, 1]

    # Actual steering angle entering the dynamics:
    delta_max = 0.6
    steer = delta_max * np.tanh(
        steer_raw / delta_max
    )

    # Controls have N values while states have N+1 values,
    # so hold the final control through the last state time:
    t_control_ext = np.append(
        t_control,
        duration,
    )

    throttle_ext = np.append(
        throttle,
        throttle[-1],
    )

    steer_ext = np.append(
        steer,
        steer[-1],
    )

    # Interpolate controls onto animation frames:
    throttle_anim = np.interp(
        t_anim,
        t_control_ext,
        throttle_ext,
    )

    steer_anim = np.interp(
        t_anim,
        t_control_ext,
        steer_ext,
    )

    # ------------------------------------------------------------
    # Figure
    # ------------------------------------------------------------

    fig = plt.figure(
        figsize=(10.5, 4.7),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.55, 1.0],
    )

    ax_car = fig.add_subplot(gs[:, 0])
    ax_throttle = fig.add_subplot(gs[0, 1])
    ax_steer = fig.add_subplot(gs[1, 1])

    # ============================================================
    # LEFT: CAR + TRAJECTORY
    # ============================================================

    x_path = Z[:, 0]
    y_path = Z[:, 1]

    # Determine axis bounds:
    obs_x = [o[0] for o in obstacles]
    obs_y = [o[1] for o in obstacles]

    all_x = np.concatenate([
        x_path,
        np.asarray(obs_x),
        [target[0]],
    ])

    all_y = np.concatenate([
        y_path,
        np.asarray(obs_y),
        [target[1]],
    ])

    pad_x = 3.0
    pad_y = 3.0

    ax_car.set_xlim(
        np.min(all_x) - R_obs - pad_x,
        np.max(all_x) + R_obs + pad_x,
    )

    ax_car.set_ylim(
        np.min(all_y) - R_obs - pad_y,
        np.max(all_y) + R_obs + pad_y,
    )

    ax_car.set_aspect("equal", adjustable="box")

    ax_car.set_xlabel(r"$x$ [m]")
    ax_car.set_ylabel(r"$y$ [m]")
    ax_car.set_title("GRACE Obstacle Avoidance")

    ax_car.grid(alpha=0.15)

    # ------------------------------------------------------------
    # Obstacles
    # ------------------------------------------------------------

    for obs in obstacles:

        obstacle = Circle(
            (obs[0], obs[1]),
            R_obs,
            fill=True,
            alpha=0.15,
            linewidth=2.0,
        )

        ax_car.add_patch(obstacle)

        boundary = Circle(
            (obs[0], obs[1]),
            R_obs,
            fill=False,
            linestyle="--",
            linewidth=1.5,
        )

        ax_car.add_patch(boundary)

    # ------------------------------------------------------------
    # Target
    # ------------------------------------------------------------

    ax_car.plot(
        target[0],
        target[1],
        marker="*",
        markersize=14,
        linestyle="none",
        label="Target",
    )

    # Full optimized trajectory faintly in background:
    ax_car.plot(
        x_path,
        y_path,
        linestyle="--",
        linewidth=1.2,
        alpha=0.20,
    )

    # Actual traveled portion:
    path_line, = ax_car.plot(
        [],
        [],
        linewidth=2.5,
    )

    # ------------------------------------------------------------
    # Car body
    # ------------------------------------------------------------

    car_length = 2.5
    car_width = 1.3

    car_patch = Rectangle(
        (
            -car_length / 2,
            -car_width / 2,
        ),
        car_length,
        car_width,
        fill=False,
        linewidth=2.3,
    )

    ax_car.add_patch(car_patch)

    # Heading indicator:
    heading_line, = ax_car.plot(
        [],
        [],
        linewidth=2.0,
    )

    # Current position:
    center_point, = ax_car.plot(
        [],
        [],
        "o",
        markersize=5,
    )

    # Status:
    status_text = ax_car.text(
        0.03,
        0.96,
        "",
        transform=ax_car.transAxes,
        va="top",
        ha="left",
        fontsize=10,
    )

    # ============================================================
    # RIGHT TOP: THROTTLE
    # ============================================================

    ax_throttle.set_title("Control Input")
    ax_throttle.set_ylabel("Throttle")

    ax_throttle.set_xlim(
        0.0,
        duration,
    )

    throttle_pad = 0.08

    ax_throttle.set_ylim(
        min(-0.05, np.min(throttle) - throttle_pad),
        max(1.05, np.max(throttle) + throttle_pad),
    )

    ax_throttle.grid(alpha=0.2)

    # Full solution faintly:
    ax_throttle.plot(
        t_control_ext,
        throttle_ext,
        linewidth=1.2,
        alpha=0.18,
    )

    # Animated solution:
    throttle_line, = ax_throttle.plot(
        [],
        [],
        linewidth=2.3,
    )

    throttle_point, = ax_throttle.plot(
        [],
        [],
        "o",
        markersize=6,
    )

    throttle_cursor = ax_throttle.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.45,
    )

    # ============================================================
    # RIGHT BOTTOM: STEERING
    # ============================================================

    ax_steer.set_xlabel("Time [s]")
    ax_steer.set_ylabel(r"Steering $\delta$ [rad]")

    ax_steer.set_xlim(
        0.0,
        duration,
    )

    steer_limit = max(
        delta_max,
        np.max(np.abs(steer)) * 1.1,
    )

    ax_steer.set_ylim(
        -1.1 * steer_limit,
        1.1 * steer_limit,
    )

    ax_steer.axhline(
        0.0,
        linewidth=0.8,
        alpha=0.35,
    )

    ax_steer.grid(alpha=0.2)

    # Steering limits:
    ax_steer.axhline(
        delta_max,
        linestyle=":",
        linewidth=1.0,
        alpha=0.35,
    )

    ax_steer.axhline(
        -delta_max,
        linestyle=":",
        linewidth=1.0,
        alpha=0.35,
    )

    # Full solution faintly:
    ax_steer.plot(
        t_control_ext,
        steer_ext,
        linewidth=1.2,
        alpha=0.18,
    )

    # Animated solution:
    steer_line, = ax_steer.plot(
        [],
        [],
        linewidth=2.3,
    )

    steer_point, = ax_steer.plot(
        [],
        [],
        "o",
        markersize=6,
    )

    steer_cursor = ax_steer.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        alpha=0.45,
    )

    # ------------------------------------------------------------
    # Figure cleanup
    # ------------------------------------------------------------

    for ax in (
        ax_car,
        ax_throttle,
        ax_steer,
    ):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # ------------------------------------------------------------
    # Animation update
    # ------------------------------------------------------------

    def update(frame):

        x = Z_anim[frame, 0]
        y = Z_anim[frame, 1]
        theta = Z_anim[frame, 2]
        v = Z_anim[frame, 3]

        t = t_anim[frame]

        # --------------------------------------------------------
        # Car transformation
        # --------------------------------------------------------

        transform = (
            Affine2D()
            .rotate(theta)
            .translate(x, y)
            + ax_car.transData
        )

        car_patch.set_transform(transform)

        # Heading line:
        nose_x = x + 0.75 * car_length * np.cos(theta)
        nose_y = y + 0.75 * car_length * np.sin(theta)

        heading_line.set_data(
            [x, nose_x],
            [y, nose_y],
        )

        center_point.set_data(
            [x],
            [y],
        )

        # Traveled path:
        path_line.set_data(
            Z_anim[: frame + 1, 0],
            Z_anim[: frame + 1, 1],
        )

        # Status:
        status_text.set_text(
            rf"$t={t:.1f}$ s"
            + "\n"
            + rf"$v={v:.1f}$ m/s"
        )

        # --------------------------------------------------------
        # Throttle history
        # --------------------------------------------------------

        throttle_line.set_data(
            t_anim[: frame + 1],
            throttle_anim[: frame + 1],
        )

        throttle_point.set_data(
            [t],
            [throttle_anim[frame]],
        )

        throttle_cursor.set_xdata(
            [t, t]
        )

        # --------------------------------------------------------
        # Steering history
        # --------------------------------------------------------

        steer_line.set_data(
            t_anim[: frame + 1],
            steer_anim[: frame + 1],
        )

        steer_point.set_data(
            [t],
            [steer_anim[frame]],
        )

        steer_cursor.set_xdata(
            [t, t]
        )

        return (
            car_patch,
            heading_line,
            center_point,
            path_line,
            status_text,
            throttle_line,
            throttle_point,
            throttle_cursor,
            steer_line,
            steer_point,
            steer_cursor,
        )

    # ------------------------------------------------------------
    # Save animation
    # ------------------------------------------------------------

    animation = FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000 / fps,
        blit=False,
    )

    animation.save(
        save,
        writer=PillowWriter(fps=fps),
        dpi=130,
    )

    plt.close(fig)

    print(f"Saved animation to: {save}")


# Main run function:
def main():

    # Define job name:
    job_name = "car"

    # Make directory:
    os.makedirs(
        f"figures/{job_name}",
        exist_ok=True,
    )

    # Define trajectory:
    N, dt = 60, 0.1

    # Build system and engine:
    system = grace.build_cached(
        car,
        nx=4,
        nu=2,
        N=N,
        z0=[0, 0, 0, 5.0],
        dt=dt,
        job=job_name,
    )

    engine = grace.GRACE(system)

    # ============================================================
    # SIMPLE TRAJECTORY
    # ============================================================

    target = np.array([
        30.0,
        8.0,
        0.0,
        6.0,
    ])

    start = time.time()

    U = engine.shooting.lambda_shoot(
        target
    )

    Z = system.rollout(U)

    print(
        f"SIMPLE CASE ({time.time() - start:.2f}s): "
        f"{engine.utils.diagnostics(U, target)}"
    )

    # ============================================================
    # OBSTACLE AVOIDANCE
    # ============================================================

    target = np.array([
        30.0,
        0.0,
        0.0,
        6.0,
    ])

    # Define obstacle:
    obstacles = [
        [16.0, 0.01]
    ]

    R_obs = 6.0

    # Keep-out zone and actuator limits:
    constraints = [
        (
            lambda z, u, o=o:
            R_obs ** 2
            - (
                (z[0] - o[0]) ** 2
                + (z[1] - o[1]) ** 2
            )
        )
        for o in obstacles
    ] + [
        lambda z, u: u[0] - 1.0,
        lambda z, u: 0.0 - u[0],
    ]

    # Solve:
    start = time.time()

    U_obs = engine.shooting.lambda_shoot(
        target,
        constraints=constraints,
        outer=60,
        inner=30,
    )

    Z_obs = system.rollout(U_obs)

    print(
        f"OBSTACLE CASE ({time.time() - start:.2f}s): "
        f"{engine.utils.diagnostics(U_obs, target, constraints)}"
    )

    # ------------------------------------------------------------
    # Presentation GIF
    # ------------------------------------------------------------

    animate_car_obstacle(
        Z_obs,
        U_obs,
        dt,
        obstacles,
        R_obs,
        target,
        save=f"figures/{job_name}/obstacle_avoidance.gif",
    )


# Run on call:
if __name__ == "__main__":
    main()