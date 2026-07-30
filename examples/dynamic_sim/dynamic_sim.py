# Import packages:
import os
import sys
import queue
import threading
import numpy as np
import casadi as ca
import pygame
import grace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# --- Thrust-vectored vehicle: position, velocity; controls are accelerations ---
def thruster(x, u):
    drag = 0.12
    return ca.vertcat(x[2], x[3],
                      u[0] - drag * x[2],
                      u[1] - drag * x[3])


# --- Arena and view configuration ---
WORLD_W, WORLD_H = 46.0, 30.0            # metres
PALETTE_W = 118                          # obstacle palette strip
PLOT_W = 300                             # telemetry panel on the right
SCREEN_W, SCREEN_H = 1400, 820
VIEW_W = SCREEN_W - PALETTE_W - PLOT_W
SCALE = min(VIEW_W / WORLD_W, SCREEN_H / WORLD_H)

N_HORIZON = 70                           # planning nodes
DT = 0.12                                # planning timestep (s)
FPS = 60
CTRL_SUBSTEPS = 8                        # frames per plan step (slows playback)
PREDICT_STEPS = 12                       # plan steps to look ahead when re-planning
A_MAX = 2.0                              # thrust limit (m/s^2)
VEH_R = 0.5                              # drawn radius (m)

PALETTE_SIZES = [1.5, 2.5, 3.5]          # obstacle radii (m)

BG = (14, 16, 21)
GRID = (28, 33, 41)
INK = (232, 238, 246)
MUTED = (120, 132, 150)
ACCENT = (60, 210, 150)
FUTURE = (44, 120, 96)
TRAIL = (38, 90, 74)
DANGER = (222, 74, 68)
DANGER_FILL = (58, 24, 24)
PANEL = (19, 22, 29)
PLOT_A = (90, 170, 255)
PLOT_B = (255, 170, 70)


def to_screen(p):
    return (int(PALETTE_W + p[0] * SCALE),
            int(SCREEN_H - p[1] * SCALE))


def to_world(p):
    return np.array([(p[0] - PALETTE_W) / SCALE,
                     (SCREEN_H - p[1]) / SCALE])


def in_arena(pos):
    return PALETTE_W <= pos[0] < PALETTE_W + VIEW_W


class _NullFont:
    def render(self, text, antialias, color):
        return pygame.Surface((0, 0), pygame.SRCALPHA)


def _load_font(size, bold=False):
    try:
        if not pygame.font.get_init():
            pygame.font.init()
        try:
            return pygame.font.SysFont("monospace", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)
    except Exception:
        return _NullFont()


def lqr_gains(system, Z, U):
    """Finite-horizon LQR about a planned segment, used to track it online."""
    nx, nu, N = system.nx, system.nu, system.N
    Q = np.diag([14.0, 14.0, 2.0, 2.0])
    Qf = Q * 18.0
    R = np.eye(nu) * 0.05
    if getattr(system, "step_jac_all", None) is not None:
        A, B = system.step_jac_all(Z, U)
    else:
        A = [system.step_jac(Z[k], U[k * nu:(k + 1) * nu])[0] for k in range(N)]
        B = [system.step_jac(Z[k], U[k * nu:(k + 1) * nu])[1] for k in range(N)]
    P = Qf.copy()
    K = [None] * N
    for k in range(N - 1, -1, -1):
        Ak, Bk = np.asarray(A[k]), np.asarray(B[k])
        S = R + Bk.T @ P @ Bk
        K[k] = np.linalg.solve(S, Bk.T @ P @ Ak)
        P = Q + Ak.T @ P @ Ak - Ak.T @ P @ Bk @ K[k]
    return K


class Segment:
    """A planned trajectory to one waypoint: states, controls, LQR gains."""

    def __init__(self, Z, U, K, goal):
        self.Z, self.U, self.K = Z, U, K
        self.goal = goal
        self.terminal = Z[-1].copy()


class Planner:
    """Mission planner: one GRACE solve per waypoint, on a worker thread."""

    def __init__(self):
        self.system = grace.build_cached(thruster, nx=4, nu=2, N=N_HORIZON,
                                  z0=[4, 15, 0, 0], dt=DT, pos_idx=(0, 1),
                                  job="dynamic_sim")
        self.engine = grace.GRACE(self.system)
        self.requests = queue.Queue()
        self.results = queue.Queue()
        self.busy = False
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            token, start_state, goal, obstacles = self.requests.get()
            self.system.z0 = np.asarray(start_state, float)
            goal = np.asarray(goal, float)
            target = np.array([goal[0], goal[1], 0.0, 0.0])   # arrive at rest
            try:
                if obstacles:
                    # The solver takes a single radius, so plan against the largest
                    # obstacle applied to every centre (conservative but simple):
                    radius = max(o[2] for o in obstacles) + VEH_R
                    centres = [[o[0], o[1]] for o in obstacles]
                    U = self.engine.shooting.lambda_shoot(
                        target, obstacles=centres, R=radius, pos_idx=(0, 1),
                        u_lo=[-A_MAX, -A_MAX], u_hi=[A_MAX, A_MAX])
                else:
                    U = self.engine.shooting.lambda_shoot(
                        target, u_lo=[-A_MAX, -A_MAX], u_hi=[A_MAX, A_MAX])
                Z = np.asarray(self.system.rollout(U))
                ok = not getattr(self.system, "_obstacle_infeasible", False)
                K = lqr_gains(self.system, Z, U)
                self.results.put((token, Segment(Z, U, K, goal), ok))
            except Exception:
                self.results.put((token, None, False))

    def request(self, token, start_state, goal, obstacles):
        self.busy = True
        self.requests.put((token, np.array(start_state, float),
                           np.array(goal, float), list(obstacles)))

    def poll(self):
        try:
            token, seg, ok = self.results.get_nowait()
            self.busy = False
            return token, seg, ok
        except queue.Empty:
            return None


class Sim:
    def __init__(self):
        self.planner = Planner()
        self.state = np.array([4.0, 15.0, 0.0, 0.0])
        self.step_np = self.planner.system.step_np

        self.waypoints = []          # goals not yet planned
        self.segments = []           # planned segments waiting to be flown
        self.active = None           # segment currently tracked
        self.active_k = 0
        self.sub = 0

        self.obstacles = []          # (x, y, r)
        self.trail = []
        self.speed_hist = []
        self.thrust_hist = []
        self.dragging = None
        self.pending_token = None
        self.next_token = 0
        self.splice = False          # next result replaces the active segment
        self.status = "click to set a target  |  drag obstacles from the palette"

    # --- planning ----------------------------------------------------------
    def _last_terminal(self):
        if self.segments:
            return self.segments[-1].terminal
        if self.active is not None:
            return self.active.terminal
        return self.state

    def _kick(self):
        if self.planner.busy or not self.waypoints:
            return
        self.next_token += 1
        self.pending_token = self.next_token
        self.planner.request(self.next_token, self._last_terminal(),
                             self.waypoints[0], self.obstacles)
        self.status = "planning..."

    def _replan(self):
        # The world changed.  The vehicle CANNOT stop and hover while a solve runs, so
        # keep flying the active segment and re-plan from where the vehicle will be by
        # the time the new trajectory is ready (PREDICT_STEPS ahead on the current plan).
        # The new segment is spliced in the moment it arrives, so motion is continuous.
        remaining = [s.goal for s in self.segments] + list(self.waypoints)
        if self.active is not None:
            remaining = [self.active.goal] + remaining
        self.segments = []
        self.waypoints = remaining
        self.pending_token = None
        self._kick_predicted()

    def _kick_predicted(self):
        # Plan the next segment from the vehicle's PREDICTED state rather than its
        # current one, so the solution is still valid when it lands:
        if self.planner.busy or not self.waypoints:
            return
        if self.active is not None:
            k = min(self.active_k + PREDICT_STEPS, self.planner.system.N - 1)
            start = self.active.Z[k].copy()
        else:
            start = self.state.copy()
        self.next_token += 1
        self.pending_token = self.next_token
        self.splice = True
        self.planner.request(self.next_token, start, self.waypoints[0], self.obstacles)
        self.status = "re-planning while flying..."

    # --- input -------------------------------------------------------------
    def obstacle_at(self, world):
        for i, (ox, oy, orad) in enumerate(self.obstacles):
            if np.hypot(world[0] - ox, world[1] - oy) <= orad:
                return i
        return None

    def on_mouse_down(self, event):
        if event.button == 1:
            if event.pos[0] < PALETTE_W:
                for i, r in enumerate(PALETTE_SIZES):
                    if abs(event.pos[1] - (140 + i * 165)) < 70:
                        self.dragging = {"radius": r, "index": None}
                        return
                return
            if not in_arena(event.pos):
                return
            world = to_world(event.pos)
            hit = self.obstacle_at(world)
            if hit is not None:
                self.dragging = {"radius": self.obstacles[hit][2], "index": hit}
                return
            self.waypoints.append(world)
            self.status = f"{len(self.waypoints)} target(s) queued"
            self._kick()

        elif event.button == 3 and in_arena(event.pos):
            world = to_world(event.pos)
            hit = self.obstacle_at(world)
            if hit is not None:
                self.obstacles.pop(hit)
                self.status = "obstacle removed -- replanning"
                self._replan()
                return
            if self.waypoints:
                d = [float(np.linalg.norm(w - world)) for w in self.waypoints]
                if min(d) < 1.5:
                    self.waypoints.pop(int(np.argmin(d)))
                    self.status = "target removed"

    def on_mouse_up(self, event):
        if event.button != 1 or self.dragging is None:
            return
        drop = self.dragging
        self.dragging = None
        if not in_arena(event.pos):
            if drop["index"] is not None:
                self.obstacles.pop(drop["index"])
                self.status = "obstacle removed -- replanning"
                self._replan()
            return
        world = to_world(event.pos)
        if drop["index"] is None:
            self.obstacles.append((world[0], world[1], drop["radius"]))
            self.status = "obstacle placed -- replanning around it"
        else:
            self.obstacles[drop["index"]] = (world[0], world[1], drop["radius"])
            self.status = "obstacle moved -- replanning"
        self._replan()

    # --- simulation --------------------------------------------------------
    def update(self):
        done = self.planner.poll()
        if done is not None:
            token, seg, ok = done
            if token == self.pending_token and seg is not None:
                if self.waypoints:
                    self.waypoints.pop(0)
                if self.splice:
                    # This solve replaced the trajectory being flown: swap it in now so
                    # the vehicle starts avoiding immediately, without ever stopping.
                    # It was planned from a PREDICTED state, so begin tracking at the
                    # point on the new plan closest to where the vehicle actually is --
                    # starting at index 0 would command a large catch-up and overshoot.
                    d = np.linalg.norm(seg.Z[:, :2] - self.state[:2], axis=1)
                    self.active = seg
                    self.active_k = int(np.argmin(d))
                    self.sub = 0
                    self.splice = False
                else:
                    self.segments.append(seg)
                self.pending_token = None
                if not ok:
                    self.status = "tight fit -- flying best feasible trajectory"
            self._kick()

        if self.active is None and self.segments:
            self.active = self.segments.pop(0)
            self.active_k = 0
            self.sub = 0
            self.status = "flying GRACE trajectory"

        # Advance the physics only every CTRL_SUBSTEPS frames.  One physics step
        # consumes exactly one plan step (both advance dt), which keeps the vehicle on
        # its trajectory; the extra frames simply slow wall-clock playback so there is
        # time to drop obstacles and watch the vehicle react.
        thrust = self.thrust_hist[-1] if self.thrust_hist else 0.0
        self.sub += 1
        if self.sub >= CTRL_SUBSTEPS:
            self.sub = 0
            if self.active is not None:
                Z, U, K = self.active.Z, self.active.U, self.active.K
                nu = self.planner.system.nu
                k = min(self.active_k, self.planner.system.N - 1)
                u = U[k * nu:(k + 1) * nu] - K[k] @ (self.state - Z[k])
                u = np.clip(u, -A_MAX, A_MAX)
                thrust = float(np.linalg.norm(u))
                self.state = np.asarray(self.step_np(self.state, u), float)
                self.active_k += 1
                if self.active_k >= self.planner.system.N:
                    self.active = None
                    if not self.segments and not self.waypoints:
                        self.status = "arrived -- click a new target"

        self.trail.append(self.state[:2].copy())
        if len(self.trail) > 1500:
            self.trail.pop(0)
        self.speed_hist.append(float(np.hypot(self.state[2], self.state[3])))
        self.thrust_hist.append(thrust)
        if len(self.speed_hist) > 260:
            self.speed_hist.pop(0)
            self.thrust_hist.pop(0)

        self._kick()

    # --- drawing -----------------------------------------------------------
    def draw(self, screen, font, big):
        screen.fill(BG)

        for gx in range(int(WORLD_W) + 1):
            if gx % 5 == 0:
                pygame.draw.line(screen, GRID, to_screen((gx, 0)), to_screen((gx, WORLD_H)))
        for gy in range(int(WORLD_H) + 1):
            if gy % 5 == 0:
                pygame.draw.line(screen, GRID, to_screen((0, gy)), to_screen((WORLD_W, gy)))

        if len(self.trail) > 1:
            pygame.draw.lines(screen, TRAIL, False, [to_screen(p) for p in self.trail], 2)

        for (ox, oy, orad) in self.obstacles:
            c = to_screen((ox, oy))
            pygame.draw.circle(screen, DANGER_FILL, c, int(orad * SCALE))
            pygame.draw.circle(screen, DANGER, c, int(orad * SCALE), 2)

        for seg in self.segments:
            pts = [to_screen(p) for p in seg.Z[:, :2]]
            if len(pts) > 1:
                pygame.draw.lines(screen, FUTURE, False, pts, 2)
        if self.active is not None:
            pts = [to_screen(p) for p in self.active.Z[:, :2]]
            if len(pts) > 1:
                pygame.draw.lines(screen, ACCENT, False, pts, 3)

        planned = ([self.active.goal] if self.active is not None else []) \
            + [s.goal for s in self.segments] + list(self.waypoints)
        for i, w in enumerate(planned):
            wx, wy = to_screen(w)
            colour = ACCENT if i == 0 else MUTED
            pygame.draw.circle(screen, colour, (wx, wy), 10, 2)
            pygame.draw.line(screen, colour, (wx - 14, wy), (wx + 14, wy), 1)
            pygame.draw.line(screen, colour, (wx, wy - 14), (wx, wy + 14), 1)
            screen.blit(font.render(str(i + 1), True, colour), (wx + 14, wy - 18))

        # vehicle with a thrust-direction velocity vector
        px, py = to_screen(self.state[:2])
        pygame.draw.circle(screen, INK, (px, py), max(4, int(VEH_R * SCALE)))
        v = self.state[2:4]
        if np.linalg.norm(v) > 0.2:
            tip = to_screen(self.state[:2] + v * 0.35)
            pygame.draw.line(screen, ACCENT, (px, py), tip, 2)

        # palette
        pygame.draw.rect(screen, PANEL, (0, 0, PALETTE_W, SCREEN_H))
        pygame.draw.line(screen, GRID, (PALETTE_W, 0), (PALETTE_W, SCREEN_H))
        screen.blit(font.render("obstacles", True, MUTED), (14, 66))
        for i, r in enumerate(PALETTE_SIZES):
            cy = 140 + i * 165
            rr = int(r * SCALE)
            pygame.draw.circle(screen, DANGER_FILL, (PALETTE_W // 2, cy), rr)
            pygame.draw.circle(screen, DANGER, (PALETTE_W // 2, cy), rr, 2)
            screen.blit(font.render(f"{r:.1f} m", True, MUTED),
                        (PALETTE_W // 2 - 22, cy + rr + 8))

        if self.dragging is not None:
            mx, my = pygame.mouse.get_pos()
            rad = int(self.dragging["radius"] * SCALE)
            pygame.draw.circle(screen, DANGER_FILL, (mx, my), rad)
            pygame.draw.circle(screen, DANGER, (mx, my), rad, 2)

        self._draw_telemetry(screen, font, big)

        screen.blit(big.render("GRACE -- online obstacle avoidance", True, INK),
                    (PALETTE_W + 16, 14))
        screen.blit(font.render(self.status, True, MUTED), (PALETTE_W + 16, 40))
        screen.blit(font.render(
            "left click: target   drag palette: obstacle   right click: remove   "
            "c: clear   r: reset   esc: quit", True, GRID),
            (PALETTE_W + 16, SCREEN_H - 24))
        if self.planner.busy:
            screen.blit(font.render("solving...", True, PLOT_B),
                        (PALETTE_W + VIEW_W - 110, 14))

    def _draw_telemetry(self, screen, font, big):
        x0 = PALETTE_W + VIEW_W
        pygame.draw.rect(screen, PANEL, (x0, 0, PLOT_W, SCREEN_H))
        pygame.draw.line(screen, GRID, (x0, 0), (x0, SCREEN_H))
        screen.blit(big.render("telemetry", True, INK), (x0 + 16, 16))

        def plot(top, h, data, colour, label, vmax):
            pygame.draw.rect(screen, GRID, (x0 + 16, top, PLOT_W - 34, h), 1)
            screen.blit(font.render(label, True, MUTED), (x0 + 18, top - 18))
            if len(data) > 1:
                n = len(data)
                pts = []
                for i, v in enumerate(data):
                    px = x0 + 17 + int(i / max(n - 1, 1) * (PLOT_W - 36))
                    py = top + h - int(min(v / vmax, 1.0) * (h - 3)) - 2
                    pts.append((px, py))
                pygame.draw.lines(screen, colour, False, pts, 2)
            cur = data[-1] if data else 0.0
            screen.blit(font.render(f"{cur:5.2f}", True, colour),
                        (x0 + PLOT_W - 76, top + 4))

        plot(90, 120, self.speed_hist, PLOT_A, "speed (m/s)", 12.0)
        plot(250, 120, self.thrust_hist, PLOT_B, "thrust (m/s^2)", A_MAX * 1.3)

        # status block
        y = 420
        rows = [
            ("position", f"{self.state[0]:5.1f}, {self.state[1]:5.1f} m"),
            ("velocity", f"{np.hypot(self.state[2], self.state[3]):5.2f} m/s"),
            ("obstacles", f"{len(self.obstacles)}"),
            ("targets queued", f"{len(self.waypoints) + len(self.segments)}"),
        ]
        clr = None
        if self.obstacles:
            clr = min(np.hypot(self.state[0] - o[0], self.state[1] - o[1]) - o[2]
                      for o in self.obstacles)
            rows.append(("nearest obstacle", f"{clr:5.2f} m"))
        for i, (k, v) in enumerate(rows):
            screen.blit(font.render(k, True, MUTED), (x0 + 18, y + i * 24))
            screen.blit(font.render(v, True, INK), (x0 + 168, y + i * 24))

        if clr is not None:
            col = DANGER if clr < 0 else ACCENT
            msg = "CLEAR" if clr >= 0 else "CONTACT"
            screen.blit(font.render(msg, True, col), (x0 + 18, y + len(rows) * 24 + 14))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("GRACE -- online obstacle avoidance")
    clock = pygame.time.Clock()
    font = _load_font(14)
    big = _load_font(18, bold=True)

    sim = Sim()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_c:
                    sim.obstacles.clear()
                    sim.status = "obstacles cleared -- replanning"
                    sim._replan()
                elif event.key == pygame.K_r:
                    sim.state = np.array([4.0, 15.0, 0.0, 0.0])
                    sim.waypoints.clear()
                    sim.segments.clear()
                    sim.active = None
                    sim.trail.clear()
                    sim.pending_token = None
                    sim.status = "reset"
            elif event.type == pygame.MOUSEBUTTONDOWN:
                sim.on_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                sim.on_mouse_up(event)

        sim.update()
        sim.draw(screen, font, big)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()