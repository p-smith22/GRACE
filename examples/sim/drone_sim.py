# ============================================================================
# drone_sim.py -- interactive real-time demo of GRACE
# ============================================================================
# The pygame window IS the simulator.  Every path the drone flies is a GRACE
# minimum-effort solve, tracked in closed loop by a finite-horizon LQR:
#   * left click in the world    -> queue a waypoint (click several to build a route)
#   * drag from the left palette -> drop an obstacle into the world
#   * drag an existing obstacle  -> move it (drop it on the palette to delete)
#   * right click an obstacle    -> remove it; right click a waypoint to unqueue it
# The drone flies the queue continuously and replans whenever the world changes.
# Planning runs on a worker thread so the simulation never stalls on a solve.
# ============================================================================

import os
import sys
import queue
import threading

import numpy as np
import casadi as ca
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grace


# --- Planar quadrotor: position, velocity, attitude, body rate ---
def quadrotor(x, u):
    g = 9.81
    return ca.vertcat(x[2], x[3],
                      -u[0] * ca.sin(x[4]),
                      u[0] * ca.cos(x[4]) - g,
                      x[5], u[1])


# --- World and view configuration ---
WORLD_W, WORLD_H = 12.0, 8.0             # metres
PALETTE_W = 132                          # pixels reserved for the obstacle palette
SCREEN_W, SCREEN_H = 1080, 640           # pixels
SCALE = (SCREEN_W - PALETTE_W) / WORLD_W
N_HORIZON = 40                           # planning nodes
DT = 0.08                                # planning timestep (s)
FPS = 50
DRONE_R = 0.22                           # drawn half-span (m)
ARRIVE_TOL = 0.3                         # metres, when a waypoint counts as reached

# Obstacle sizes offered by the palette (world radius in metres):
PALETTE_SIZES = [0.45, 0.7, 1.0]

BG = (248, 249, 251)
GRID = (232, 234, 238)
INK = (32, 42, 56)
MUTED = (120, 130, 145)
ACCENT = (40, 150, 110)
DANGER = (200, 60, 70)
DANGER_FILL = (250, 226, 228)
PANEL = (240, 242, 246)


def to_screen(p):
    # World metres -> screen pixels (y flipped so up is up, offset past the palette):
    return int(PALETTE_W + p[0] * SCALE), int(SCREEN_H - p[1] * SCALE)


def to_world(p):
    return np.array([(p[0] - PALETTE_W) / SCALE, (SCREEN_H - p[1]) / SCALE])


def in_world(pos):
    return pos[0] >= PALETTE_W



class _NullFont:
    """Stand-in used when pygame.font is unavailable (some Python/pygame builds).

    It renders an empty surface so every screen.blit(...) call still works and the
    simulator runs normally, just without the text labels.
    """

    def render(self, text, antialias, color):
        return pygame.Surface((0, 0), pygame.SRCALPHA)


def _load_font(size, bold=False):
    # pygame.font can fail to import on some builds; fall back rather than crash.
    try:
        if not pygame.font.get_init():
            pygame.font.init()
        try:
            return pygame.font.SysFont("monospace", size, bold=bold)
        except Exception:
            return pygame.font.Font(None, size)
    except Exception:
        return _NullFont()


class Planner:
    """Runs GRACE solves on a worker thread so the render loop never blocks."""

    def __init__(self):
        self.system = grace.build(quadrotor, nx=6, nu=2, N=N_HORIZON,
                                  z0=[0, 0, 0, 0, 0, 0], dt=DT,
                                  pos_idx=(0, 1), job="drone_sim")
        self.engine = grace.GRACE(self.system)
        self.requests = queue.Queue()
        self.results = queue.Queue()
        self.busy = False
        self._seq = 0
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while True:
            seq, state, goal, obstacles = self.requests.get()

            # Re-anchor the system at the drone's current state and solve to the goal:
            self.system.z0 = np.asarray(state, float)
            target = np.array([goal[0], goal[1], 0.0, 0.0, 0.0, 0.0])
            try:
                if obstacles:
                    # The solver takes a single radius, so plan against the largest one
                    # and treat every centre at that radius (conservative but simple):
                    radius = max(o[2] for o in obstacles)
                    centres = [[o[0], o[1]] for o in obstacles]
                    U = self.engine.shooting.lambda_shoot(
                        target, obstacles=centres, R=radius, pos_idx=(0, 1))
                else:
                    U = self.engine.shooting.lambda_shoot(target)
                Z = np.asarray(self.system.rollout(U))
                ok = not getattr(self.system, "_obstacle_infeasible", False)
                self.results.put((seq, Z, U, ok))
            except Exception:
                self.results.put((seq, None, None, False))

    def request(self, state, goal, obstacles):
        # Only one solve in flight at a time; later requests supersede earlier ones.
        if self.busy:
            return
        self.busy = True
        self._seq += 1
        self.requests.put((self._seq, np.array(state, float),
                           np.array(goal, float), list(obstacles)))

    def poll(self):
        try:
            seq, Z, U, ok = self.results.get_nowait()
            self.busy = False
            if seq != self._seq:          # stale result from a superseded request
                return None
            return Z, U, ok
        except queue.Empty:
            return None


def lqr_gains(system, Z, U):
    """Finite-horizon LQR about the planned trajectory, used to track it."""
    nx, nu, N = system.nx, system.nu, system.N
    Q = np.diag([12.0, 12.0, 2.0, 2.0, 1.0, 0.4])
    Qf = Q * 12.0
    R = np.eye(nu) * 0.06
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


class Sim:
    def __init__(self):
        self.planner = Planner()
        self.state = np.array([1.0, 1.5, 0.0, 0.0, 0.0, 0.0])
        self.waypoints = []              # queued goals, flown in order
        self.obstacles = []              # (x, y, r) in world units
        self.plan = None                 # (Z, U, K) currently being tracked
        self.step_i = 0
        self.dragging = None             # {"radius": r, "index": i or None}
        self.status = "click to add waypoints  |  drag obstacles from the palette"
        self.replan = False              # the world changed, replan when possible

    # --- goals -------------------------------------------------------------
    def current_goal(self):
        return self.waypoints[0] if self.waypoints else None

    def obstacle_at(self, world_pos):
        for i, (ox, oy, orad) in enumerate(self.obstacles):
            if np.hypot(world_pos[0] - ox, world_pos[1] - oy) <= orad:
                return i
        return None

    # --- input -------------------------------------------------------------
    def on_mouse_down(self, event):
        if event.button == 1:
            if not in_world(event.pos):
                # Palette: start dragging a new obstacle of the size clicked.
                for i, r in enumerate(PALETTE_SIZES):
                    if abs(event.pos[1] - (120 + i * 130)) < 55:
                        self.dragging = {"radius": r, "index": None}
                        return
                return
            world = to_world(event.pos)
            hit = self.obstacle_at(world)
            if hit is not None:
                # Grab an existing obstacle to move it.
                self.dragging = {"radius": self.obstacles[hit][2], "index": hit}
                return
            self.waypoints.append(world)
            self.status = f"{len(self.waypoints)} waypoint(s) queued"
            self.replan = True

        elif event.button == 3 and in_world(event.pos):
            world = to_world(event.pos)
            hit = self.obstacle_at(world)
            if hit is not None:
                self.obstacles.pop(hit)
                self.status = "obstacle removed"
                self.replan = True
                return
            if self.waypoints:
                d = [float(np.linalg.norm(w - world)) for w in self.waypoints]
                if min(d) < 0.6:
                    self.waypoints.pop(int(np.argmin(d)))
                    self.plan = None
                    self.status = "waypoint removed"
                    self.replan = True

    def on_mouse_up(self, event):
        if event.button != 1 or self.dragging is None:
            return
        drop = self.dragging
        self.dragging = None
        if not in_world(event.pos):
            # Dropped on the palette: delete it if it was an existing obstacle.
            if drop["index"] is not None:
                self.obstacles.pop(drop["index"])
                self.status = "obstacle removed"
                self.replan = True
            return
        world = to_world(event.pos)
        if drop["index"] is None:
            self.obstacles.append((world[0], world[1], drop["radius"]))
            self.status = f"{len(self.obstacles)} obstacle(s) placed"
        else:
            self.obstacles[drop["index"]] = (world[0], world[1], drop["radius"])
            self.status = "obstacle moved"
        self.replan = True

    # --- simulation --------------------------------------------------------
    def update(self):
        # Collect a finished solve:
        done = self.planner.poll()
        if done is not None:
            Z, U, ok = done
            if Z is None:
                self.status = "solve failed"
            elif not ok:
                self.status = "infeasible -- move the goal or shrink the obstacle"
                self.plan = None
            else:
                self.plan = (Z, U, lqr_gains(self.planner.system, Z, U))
                self.step_i = 0
                self.status = "flying GRACE trajectory"

        # Ask for a plan when the world changed, or we have a goal but nothing to fly:
        if self.current_goal() is not None and (self.replan or self.plan is None):
            if not self.planner.busy:
                self.planner.request(self.state, self.current_goal(), self.obstacles)
                self.replan = False

        # Track the current plan with LQR feedback:
        if self.plan is not None:
            Z, U, K = self.plan
            nu = self.planner.system.nu
            k = min(self.step_i, self.planner.system.N - 1)
            u = U[k * nu:(k + 1) * nu] - K[k] @ (self.state - Z[k])
            self.state = np.asarray(self.planner.system.step_np(self.state, u), float)
            self.step_i += 1

            goal = self.current_goal()
            reached = (goal is not None
                       and np.linalg.norm(self.state[:2] - goal) < ARRIVE_TOL)
            if reached or self.step_i >= self.planner.system.N:
                if reached and self.waypoints:
                    self.waypoints.pop(0)
                self.plan = None
                if not self.waypoints:
                    self.status = "idle -- click to add waypoints"
                else:
                    self.status = "next waypoint" if reached else "replanning"

    # --- drawing -----------------------------------------------------------
    def draw(self, screen, font, big):
        screen.fill(BG)
        for gx in range(int(WORLD_W) + 1):
            pygame.draw.line(screen, GRID, to_screen((gx, 0)), to_screen((gx, WORLD_H)))
        for gy in range(int(WORLD_H) + 1):
            pygame.draw.line(screen, GRID, to_screen((0, gy)), to_screen((WORLD_W, gy)))

        for (ox, oy, orad) in self.obstacles:
            c = to_screen((ox, oy))
            pygame.draw.circle(screen, DANGER_FILL, c, int(orad * SCALE))
            pygame.draw.circle(screen, DANGER, c, int(orad * SCALE), 2)

        if self.plan is not None:
            pts = [to_screen(p) for p in self.plan[0][:, :2]]
            if len(pts) > 1:
                pygame.draw.lines(screen, ACCENT, False, pts, 3)

        for i, w in enumerate(self.waypoints):
            wx, wy = to_screen(w)
            colour = ACCENT if i == 0 else MUTED
            pygame.draw.circle(screen, colour, (wx, wy), 10, 2)
            screen.blit(font.render(str(i + 1), True, colour), (wx + 12, wy - 10))

        px, py = to_screen(self.state[:2])
        phi = float(self.state[4])
        arm = DRONE_R * SCALE
        dx, dy = arm * np.cos(phi), -arm * np.sin(phi)
        pygame.draw.line(screen, INK, (px - dx, py - dy), (px + dx, py + dy), 5)
        pygame.draw.circle(screen, INK, (int(px - dx), int(py - dy)), 5)
        pygame.draw.circle(screen, INK, (int(px + dx), int(py + dy)), 5)

        pygame.draw.rect(screen, PANEL, (0, 0, PALETTE_W, SCREEN_H))
        pygame.draw.line(screen, GRID, (PALETTE_W, 0), (PALETTE_W, SCREEN_H))
        screen.blit(font.render("obstacles", True, MUTED), (18, 60))
        for i, r in enumerate(PALETTE_SIZES):
            cy = 120 + i * 130
            pygame.draw.circle(screen, DANGER_FILL, (PALETTE_W // 2, cy), int(r * SCALE))
            pygame.draw.circle(screen, DANGER, (PALETTE_W // 2, cy), int(r * SCALE), 2)
            screen.blit(font.render(f"{r:.2f} m", True, MUTED),
                        (PALETTE_W // 2 - 24, cy + 46))

        if self.dragging is not None:
            mx, my = pygame.mouse.get_pos()
            rad = int(self.dragging["radius"] * SCALE)
            pygame.draw.circle(screen, DANGER_FILL, (mx, my), rad)
            pygame.draw.circle(screen, DANGER, (mx, my), rad, 2)

        screen.blit(big.render("GRACE interactive drone", True, INK), (PALETTE_W + 16, 12))
        screen.blit(font.render(self.status, True, MUTED), (PALETTE_W + 16, 40))
        screen.blit(font.render(
            "left click: waypoint   drag palette: obstacle   right click: remove   "
            "c: clear   r: reset   esc: quit", True, MUTED),
            (PALETTE_W + 16, SCREEN_H - 26))
        if self.planner.busy:
            screen.blit(font.render("solving...", True, (200, 120, 40)),
                        (SCREEN_W - 108, 14))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("GRACE -- interactive drone")
    clock = pygame.time.Clock()
    font = _load_font(15)
    big = _load_font(19, bold=True)

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
                    sim.status = "obstacles cleared"
                    sim.replan = True
                elif event.key == pygame.K_r:
                    sim.state = np.array([1.0, 1.5, 0.0, 0.0, 0.0, 0.0])
                    sim.waypoints.clear()
                    sim.plan = None
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