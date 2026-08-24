# Import packages:
import os
import sys
import time
import queue
import threading
import numpy as np
import casadi as ca
import pygame

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import grace
from grace.utils.diagnostics import diagnostics


# --- The aircraft.  Unlike a quadcopter this cannot hover, cannot stop and
# --- cannot translate sideways: every avoidance has to be flown as a
# --- coordinated turn, and the only way to move laterally is to bank, yaw
# --- through the turn and roll back out.  That is what makes it a real test of
# --- the planner rather than a demonstration of thrust vectoring.
from Aircraft_Dynamics import f as aircraft, X0

# --- World and view configuration ---
# The aircraft covers about fifteen metres a second and never stops, so the
# world is a long corridor rather than a box:
WORLD = np.array([260.0, 120.0, 60.0])
PANEL_W = 330
SCREEN_W, SCREEN_H = 1500, 860
VIEW_W = SCREEN_W - PANEL_W

# Trim state: straight and level at fifteen metres a second.  Zero deflection
# holds it, so the natural starting guess for every solve is zero:
CRUISE = float(np.linalg.norm(X0[:3]))

# Node count is fixed; how long the manoeuvre takes is set by dt.  Each distinct
# timestep needs its own compiled system, so the estimate is snapped to one of a
# few prepared horizons rather than used directly:
N_NODES = 60
HORIZON_TIMES = [8.0, 13.0, 16.0]
FPS = 60
VEH_R = 3.0

# Control surface travel, in degrees, and the bank the planner is allowed to
# use.  These are ordinary constraints: nothing in the solver marks them as
# attitude or actuator limits, they take the same path as a keep-out:
SURFACE_LIMITS = np.array([20.0, 15.0, 10.0])
BANK_MAX = np.deg2rad(50.0)
PITCH_MAX = np.deg2rad(25.0)

# States, for readability.  The aircraft carries body velocities first, then
# rates, then Euler angles, then position:
IX, IY, IZ = 9, 10, 11
IPHI, ITHE, IPSI = 6, 7, 8

# Cross-section planes for the keep-out cylinders, named by the two position
# states each is measured in:
PLANES = {"vertical (x,y)": (IX, IY), "horizontal (x,z)": (IX, IZ),
          "horizontal (y,z)": (IY, IZ)}
PLANE_ORDER = list(PLANES.keys())

BG = (12, 14, 19)
GRID2 = (22, 27, 35)
GRID = (30, 36, 46)
INK = (233, 239, 247)
MUTED = (116, 128, 146)
ACCENT = (70, 200, 255)
FUTURE = (38, 105, 135)
TRAIL = (150, 90, 200)
DANGER = (224, 78, 70)
PANEL = (17, 20, 27)
FIELD = (28, 33, 43)
FIELD_ON = (44, 54, 70)
OK = (120, 210, 140)
WARN = (220, 200, 90)


# --- isometric camera -------------------------------------------------------
class Camera:

    def __init__(self):
        self.az = np.deg2rad(38.0)
        self.el = np.deg2rad(24.0)
        self.zoom = 4.0
        self.centre = np.array([WORLD[0] * 0.4, 0.0, X0[IZ]])

    def project(self, p):

        # Rotate into camera axes and drop the depth component.  A plain
        # axonometric projection keeps parallel lines parallel, which is what
        # makes the keep-out cylinders readable from any angle:
        d = np.asarray(p, float) - self.centre
        ca_, sa = np.cos(self.az), np.sin(self.az)
        ce, se = np.cos(self.el), np.sin(self.el)
        x = d[0] * ca_ - d[1] * sa
        y = d[0] * sa + d[1] * ca_
        return (int(VIEW_W / 2 + x * self.zoom),
                int(SCREEN_H / 2 + (y * se - d[2] * ce) * self.zoom))

    def depth(self, p):
        d = np.asarray(p, float) - self.centre
        return d[0] * np.sin(self.az) + d[1] * np.cos(self.az)


# --- text entry -------------------------------------------------------------
class Field:

    def __init__(self, x, y, w, label, value=""):
        self.rect = pygame.Rect(x, y, w, 24)
        self.label = label
        self.text = str(value)
        self.active = False

    def click(self, pos):
        self.active = self.rect.collidepoint(pos)
        return self.active

    def key(self, event):
        if not self.active:
            return
        if event.key == pygame.K_BACKSPACE:
            self.text = self.text[:-1]
        elif event.unicode and (event.unicode.isdigit() or event.unicode in "-."):
            self.text += event.unicode

    def value(self, default=0.0):
        try:
            return float(self.text.strip())
        except ValueError:
            return default

    def draw(self, screen, font):
        pygame.draw.rect(screen, FIELD_ON if self.active else FIELD, self.rect,
                         border_radius=3)
        pygame.draw.rect(screen, GRID, self.rect, 1, border_radius=3)
        screen.blit(font.render(self.label, True, MUTED),
                    (self.rect.x, self.rect.y - 16))
        screen.blit(font.render(self.text, True, INK),
                    (self.rect.x + 6, self.rect.y + 4))


class Button:

    def __init__(self, x, y, w, h, label, colour=ACCENT):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.colour = colour

    def hit(self, pos):
        return self.rect.collidepoint(pos)

    def draw(self, screen, font):
        pygame.draw.rect(screen, FIELD, self.rect, border_radius=4)
        pygame.draw.rect(screen, self.colour, self.rect, 1, border_radius=4)
        t = font.render(self.label, True, self.colour)
        screen.blit(t, (self.rect.centerx - t.get_width() // 2,
                        self.rect.centery - t.get_height() // 2))


def _font(size, bold=False):
    if not pygame.font.get_init():
        pygame.font.init()
    try:
        return pygame.font.SysFont("monospace", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)


class Segment:

    # A segment carries the system it was planned on.  Horizons use different
    # timesteps, so stepping one segment with another's physics flies it at the
    # wrong rate and the tracker spends the flight fighting it:
    def __init__(self, Z, U, K, goal, dt, step_np, report):
        self.Z, self.U, self.K, self.goal, self.dt = Z, U, K, goal, dt
        self.step_np = step_np
        self.report = report
        self.terminal = Z[-1].copy()
        self.N = len(Z) - 1

        # Frames of playback per plan step, so wall-clock time matches the dt
        # this segment was planned at:
        self.substeps = max(1, int(round(dt * FPS)))


# Mission planner: one GRACE solve per waypoint, on a worker thread:
class Planner:

    def __init__(self):
        self.levels = {}
        self.U_trim = np.zeros(N_NODES * 3)

        # Compile every horizon now, with progress, rather than on the worker
        # thread where the first request of each would look like a hang:
        for i, t_horizon in enumerate(HORIZON_TIMES):
            print("[sim] preparing horizon %.0f s (%d of %d)"
                  % (t_horizon, i + 1, len(HORIZON_TIMES)), flush=True)
            self.level(t_horizon / N_NODES)

        # Tracking weights.  Attitude is held tightly because the aircraft flies
        # by pointing itself, and the surfaces are cheap by comparison:
        self.Q = np.diag([5., 5., 5., 20., 20., 20., 200., 200., 50.,
                          10., 10., 10.])
        self.R = 0.05 * np.eye(3)
        self.Qf = self.Q * 100.0

        self.requests = queue.Queue()
        self.results = queue.Queue()
        self.busy = False
        self.solve_ema = 8.0
        self.last_horizon = 0.0
        threading.Thread(target=self._worker, daemon=True).start()

    # Fetch, building on first use, the system for one timestep:
    def level(self, dt):
        key = round(dt, 5)
        if key not in self.levels:

            # Downrange is left out of the target set on the free-flight solve
            # and put back for obstacles: the aircraft cannot choose how far it
            # travels in a given time, only where it is laterally and how it is
            # pointing when it gets there:
            tix = [3, 4, 5, 6, 7, 8, 9, 10, 11]
            system = grace.build_cached(aircraft, nx=12, nu=3, N=N_NODES,
                                        z0=list(X0), dt=key, substeps=2,
                                        target_idx=tix,
                                        job=f"aircraft_sim_N{N_NODES}_dt{int(key * 1e5)}")
            self.levels[key] = dict(system=system, engine=grace.GRACE(system),
                                    dt=key)
        return self.levels[key]

    # Every limit the aircraft flies under, as expressions g(z, u) <= 0:
    def limits(self):
        cons = []
        for i, b in enumerate(SURFACE_LIMITS):
            cons.append(lambda z, u, i=i, b=b: u[i] - b)
            cons.append(lambda z, u, i=i, b=b: -b - u[i])
        cons.append(lambda z, u: z[IPHI] - BANK_MAX)
        cons.append(lambda z, u: -BANK_MAX - z[IPHI])
        cons.append(lambda z, u: z[ITHE] - PITCH_MAX)
        cons.append(lambda z, u: -PITCH_MAX - z[ITHE])
        return cons

    def _solve_at(self, lvl, start_state, goal, obstacles, U_warm=None):
        system, engine = lvl["system"], lvl["engine"]
        system.z0 = np.asarray(start_state, float)

        # Where the aircraft ends up downrange is set by how long it flies, not
        # by the request, so the target takes its downrange from the free
        # response and only the lateral offset and altitude are commanded:
        free = np.asarray(system.rollout(self.U_trim))
        target = X0.copy()
        target[IX] = free[-1, IX]
        target[IY] = goal[0]
        target[IZ] = goal[1]

        cons = list(self.limits())
        for o in obstacles:
            a, b = PLANES[o["plane"]]
            c = np.asarray(o["centre"], float)
            r = o["radius"] + VEH_R
            cons.append(lambda z, u, a=a, b=b, c=c, r=r:
                        r ** 2 - ((z[a] - c[0]) ** 2 + (z[b] - c[1]) ** 2))

        # Warm start from the trajectory already being flown when there is one.
        # A replan is a small change to a good plan, and starting again from
        # trim throws that away:
        need = system.N * system.nu
        U_start = self.U_trim
        if U_warm is not None:
            warm = np.asarray(U_warm, float).flatten()
            if warm.size >= need:
                U_start = warm[:need]
            elif warm.size:
                pad = np.tile(warm[-system.nu:],
                              -(-(need - warm.size) // system.nu))
                U_start = np.concatenate([warm, pad])[:need]

        U = engine.shooting.lambda_shoot(target, constraints=cons, U0=U_start)
        Z = np.asarray(system.rollout(U))
        K, _ = engine.tracking.lqr_gains(U, self.Q, self.R, Qf=self.Qf)
        report = diagnostics(system, U, target, cons)

        # Accept only a plan that is feasible and reaches the target.  The
        # solver's own flag is not enough on its own: a solve can settle
        # somewhere feasible but well short of the endpoint, and flying that
        # looks like the aircraft ignoring the request:
        ok = (report["worst_violation"] < 1e-3
              and report["endpoint_error"] < 1e-2)
        return U, Z, K, ok, report

    def _worker(self):
        while True:
            token, start_state, goal, obstacles, U_warm = self.requests.get()
            t0 = time.time()
            try:

                # Step up the horizons until the request comes back good.  An
                # aircraft cannot turn in less than a certain distance, so a
                # tight avoidance simply needs more time -- there is no
                # iteration count that substitutes for it:
                seg = None
                for t_horizon in HORIZON_TIMES:
                    lvl = self.level(t_horizon / N_NODES)
                    U, Z, K, ok, report = self._solve_at(
                        lvl, start_state, goal, obstacles, U_warm)
                    seg = Segment(Z, U, K, goal, lvl["dt"],
                                  lvl["system"].step_np, report)
                    self.last_horizon = lvl["dt"] * N_NODES
                    if ok:
                        break
                self.results.put((token, seg, True, time.time() - t0))

            except Exception as exc:

                # A solve that raises is a bug or a genuinely impossible
                # request, and reporting both as unreachable hides the first
                # behind the second, so say which one happened:
                import traceback
                print("[sim] solve failed:", repr(exc))
                traceback.print_exc()
                self.results.put((token, None, False, time.time() - t0))

    def request(self, token, start_state, goal, obstacles, U_warm=None):
        self.busy = True
        self.requests.put((token, np.array(start_state, float),
                           np.array(goal, float), list(obstacles),
                           None if U_warm is None else np.asarray(U_warm, float)))

    def poll(self):
        try:
            token, seg, ok, dur = self.results.get_nowait()
            self.busy = False
            self.solve_ema = max(0.7 * self.solve_ema + 0.3 * dur, dur)
            return token, seg, ok
        except queue.Empty:
            return None


class Sim:

    def __init__(self):
        self.planner = Planner()
        self.cam = Camera()
        self.state = X0.copy()

        self.waypoints = []
        self.segments = []
        self.active = None
        self.active_k = 0
        self.sub = 0

        self.obstacles = []
        self.trail = []
        self.pending_token = None
        self.next_token = 0
        self.splice_token = None
        self.replan_queued = False
        self.dragging_view = False
        self.status = "set a lateral offset and altitude, then Add Waypoint"

        x0 = VIEW_W + 20
        self.f_y = Field(x0, 60, 120, "lateral offset y", "30")
        self.f_z = Field(x0 + 140, 60, 120, "altitude z", f"{X0[IZ]:.0f}")
        self.b_target = Button(x0, 96, 268, 26, "Add Waypoint")
        self.f_oa = Field(x0, 190, 80, "centre a", "90")
        self.f_ob = Field(x0 + 96, 190, 80, "centre b", "10")
        self.f_or = Field(x0 + 192, 190, 80, "radius", "25")
        self.b_plane = Button(x0, 226, 268, 26, PLANE_ORDER[0], MUTED)
        self.plane_i = 0
        self.b_obs = Button(x0, 258, 268, 26, "Add Cylinder")
        self.b_clear = Button(x0, 300, 130, 26, "Clear Cylinders", DANGER)
        self.b_reset = Button(x0 + 138, 300, 130, 26, "Reset", DANGER)
        self.fields = [self.f_y, self.f_z, self.f_oa, self.f_ob, self.f_or]

    # --- planning ----------------------------------------------------------
    def _last_terminal(self):
        if self.segments:
            return self.segments[-1].terminal
        if self.active is not None:
            return self.active.terminal
        return self.state

    def _last_controls(self):
        if self.segments:
            return self.segments[-1].U
        if self.active is not None:
            return self.active.U
        return None

    def _predict_steps(self):
        substeps = self.active.substeps if self.active is not None else 8
        n = int(np.ceil(self.planner.solve_ema / (substeps / float(FPS)))) + 2
        cap = self.active.N - 1 if self.active is not None else 2
        return int(np.clip(n, 2, max(cap, 2)))

    def _kick(self):
        if self.planner.busy or not self.waypoints:
            return
        self.next_token += 1
        self.pending_token = self.next_token
        self.planner.request(self.next_token, self._last_terminal(),
                             self.waypoints[0], self.obstacles,
                             self._last_controls())
        self.status = "planning..."

    def _replan(self):
        remaining = [s.goal for s in self.segments] + list(self.waypoints)
        if self.active is not None:
            remaining = [self.active.goal] + remaining
        self.segments = []
        self.waypoints = remaining
        self.pending_token = None
        if self.planner.busy:
            self.replan_queued = True
            self.status = "world changed -- re-planning when the solver frees up"
            return
        self._kick_predicted()

    def _kick_predicted(self):
        self.replan_queued = False
        if self.planner.busy or not self.waypoints:
            return

        # Re-plan from where the aircraft will be when the solve lands, and hand
        # over the controls it would have flown from that point.  It cannot stop
        # and wait, so planning from where it is now would be planning for a
        # place it has already left:
        U_warm = None
        if self.active is not None:
            k = min(self.active_k + self._predict_steps(), self.active.N - 1)
            start = self.active.Z[k].copy()
            U_warm = self.active.U[k * 3:]
        else:
            start = self.state.copy()
        self.next_token += 1
        self.pending_token = self.next_token
        self.splice_token = self.next_token
        self.planner.request(self.next_token, start, self.waypoints[0],
                             self.obstacles, U_warm)
        self.status = "re-planning while flying..."

    # --- input -------------------------------------------------------------
    def on_mouse_down(self, event):
        if event.pos[0] < VIEW_W:
            if event.button == 1:
                self.dragging_view = True
                self.drag_from = event.pos
            elif event.button in (4, 5):
                self.cam.zoom *= 1.12 if event.button == 4 else 1 / 1.12
                self.cam.zoom = float(np.clip(self.cam.zoom, 1.0, 30.0))
            return
        for f in self.fields:
            f.click(event.pos)
        if self.b_plane.hit(event.pos):
            self.plane_i = (self.plane_i + 1) % len(PLANE_ORDER)
            self.b_plane.label = PLANE_ORDER[self.plane_i]
        elif self.b_target.hit(event.pos):
            self.waypoints.append(np.array([self.f_y.value(), self.f_z.value()]))
            self.status = f"{len(self.waypoints)} waypoint(s) queued"
            self._kick()
        elif self.b_obs.hit(event.pos):
            self.obstacles.append(dict(
                centre=np.array([self.f_oa.value(), self.f_ob.value()]),
                radius=max(self.f_or.value(10.0), 1.0),
                plane=PLANE_ORDER[self.plane_i]))
            self.status = "cylinder added -- replanning"
            self._replan()
        elif self.b_clear.hit(event.pos):
            self.obstacles.clear()
            self.status = "cylinders cleared -- replanning"
            self._replan()
        elif self.b_reset.hit(event.pos):
            self.state = X0.copy()
            self.waypoints.clear()
            self.segments.clear()
            self.active = None
            self.trail.clear()
            self.pending_token = None
            self.splice_token = None
            self.replan_queued = False
            self.status = "reset"

    def on_mouse_up(self, event):
        self.dragging_view = False

    def on_mouse_motion(self, event):
        if not self.dragging_view:
            return
        dx = event.pos[0] - self.drag_from[0]
        dy = event.pos[1] - self.drag_from[1]
        self.drag_from = event.pos
        self.cam.az += dx * 0.006
        self.cam.el = float(np.clip(self.cam.el + dy * 0.005,
                                    np.deg2rad(5), np.deg2rad(85)))

    def on_key(self, event):
        for f in self.fields:
            f.key(event)

    # --- simulation --------------------------------------------------------
    def update(self):
        done = self.planner.poll()
        if done is not None:
            token, seg, ok = done
            fresh = token == self.pending_token
            if fresh and seg is not None:
                if self.waypoints:
                    self.waypoints.pop(0)
                if token == self.splice_token:

                    # Start tracking at the point on the new plan closest to
                    # where the aircraft actually is, since it was planned from
                    # a predicted state and will not match exactly:
                    d = np.linalg.norm(seg.Z[:, IX:IZ + 1]
                                       - self.state[IX:IZ + 1], axis=1)
                    self.active = seg
                    self.active_k = int(np.argmin(d))
                    self.sub = 0
                else:
                    self.segments.append(seg)
                self.splice_token = None
                self.pending_token = None
            elif fresh:
                if self.waypoints:
                    self.waypoints.pop(0)
                self.pending_token = None
                self.splice_token = None
                self.status = "waypoint unreachable -- skipped"
            if self.replan_queued:
                self._kick_predicted()
            else:
                self._kick()

        if self.active is None and self.segments:
            self.active = self.segments.pop(0)
            self.active_k = 0
            self.sub = 0
            self.status = "flying GRACE trajectory"

        self.sub += 1
        if self.active is not None and self.sub >= self.active.substeps:
            self.sub = 0
            Z, U, K = self.active.Z, self.active.U, self.active.K
            k = min(self.active_k, self.active.N - 1)
            u = U[k * 3:(k + 1) * 3] - K[k] @ (self.state - Z[k])
            self.state = np.asarray(self.active.step_np(self.state, u), float)
            self.active_k += 1
            if self.active_k >= self.active.N:
                self.active = None
                if not self.segments and not self.waypoints:
                    self.status = "waypoint reached -- add another"

        self.trail.append(self.state[IX:IZ + 1].copy())
        if len(self.trail) > 4000:
            self.trail.pop(0)

    # --- drawing -----------------------------------------------------------
    def draw(self, screen, font, big):
        screen.fill(BG)
        self._draw_ground(screen)
        for o in sorted(self.obstacles, key=lambda o: -self._obs_depth(o)):
            self._draw_cylinder(screen, o)
        for seg in self.segments:
            self._polyline(screen, seg.Z[:, IX:IZ + 1], FUTURE, 2)
        if self.active is not None:
            self._polyline(screen, self.active.Z[:, IX:IZ + 1], ACCENT, 3)
        if len(self.trail) > 1:
            self._polyline(screen, np.array(self.trail), TRAIL, 2)
        self._draw_aircraft(screen)
        self._draw_panel(screen, font, big)

    def _polyline(self, screen, pts, colour, w):
        sp = [self.cam.project(p) for p in pts]
        if len(sp) > 1:
            pygame.draw.lines(screen, colour, False, sp, w)

    def _draw_ground(self, screen):
        for gx in range(0, int(WORLD[0]) + 1, 20):
            pygame.draw.line(screen, GRID2,
                             self.cam.project([gx, -WORLD[1] / 2, 0]),
                             self.cam.project([gx, WORLD[1] / 2, 0]))
        for gy in range(int(-WORLD[1] / 2), int(WORLD[1] / 2) + 1, 20):
            pygame.draw.line(screen, GRID2, self.cam.project([0, gy, 0]),
                             self.cam.project([WORLD[0], gy, 0]))

        # The trim altitude, so a climb or descent is visible against it:
        for gx in range(0, int(WORLD[0]) + 1, 40):
            pygame.draw.line(screen, GRID, self.cam.project([gx, 0, 0]),
                             self.cam.project([gx, 0, X0[IZ]]), 1)

    def _obs_depth(self, o):
        cols = PLANES[o["plane"]]
        c = np.array([WORLD[0] / 2, 0.0, X0[IZ]])
        c[cols[0] - IX] = o["centre"][0]
        c[cols[1] - IX] = o["centre"][1]
        return self.cam.depth(c)

    def _draw_cylinder(self, screen, o):

        # A cylinder is infinite along the axis its plane does not name, so it
        # is drawn as two rings at the world bounds joined by rails:
        cols = [c - IX for c in PLANES[o["plane"]]]
        axis = [i for i in (0, 1, 2) if i not in cols][0]
        lo = 0.0 if axis != 1 else -WORLD[1] / 2
        hi = WORLD[axis] if axis != 1 else WORLD[1] / 2
        th = np.linspace(0, 2 * np.pi, 24)
        rings = []
        for end in (lo, hi):
            pts = []
            for t in th:
                p = np.zeros(3)
                p[cols[0]] = o["centre"][0] + o["radius"] * np.cos(t)
                p[cols[1]] = o["centre"][1] + o["radius"] * np.sin(t)
                p[axis] = end
                pts.append(self.cam.project(p))
            rings.append(pts)
            pygame.draw.lines(screen, DANGER, True, pts, 2)
        for i in range(0, len(th), 4):
            pygame.draw.line(screen, (90, 40, 40), rings[0][i], rings[1][i], 1)

    def _draw_aircraft(self, screen):
        pos = self.state[IX:IZ + 1]
        phi, theta, psi = self.state[IPHI], self.state[ITHE], self.state[IPSI]
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        # Body to world rotation, so the wings show real bank and pitch:
        R = np.array([
            [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
            [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
            [-sth, sph * cth, cph * cth]])

        span, chord = 3.0, 4.0
        nose = pos + R @ np.array([chord, 0.0, 0.0])
        tail = pos + R @ np.array([-chord * 0.6, 0.0, 0.0])
        left = pos + R @ np.array([0.0, -span, 0.0])
        right = pos + R @ np.array([0.0, span, 0.0])
        fin = pos + R @ np.array([-chord * 0.6, 0.0, -span * 0.5])

        pygame.draw.line(screen, GRID, self.cam.project(pos),
                         self.cam.project([pos[0], pos[1], 0]), 1)
        pygame.draw.line(screen, INK, self.cam.project(tail),
                         self.cam.project(nose), 2)
        pygame.draw.line(screen, ACCENT, self.cam.project(left),
                         self.cam.project(right), 2)
        pygame.draw.line(screen, INK, self.cam.project(tail),
                         self.cam.project(fin), 2)

    def _draw_panel(self, screen, font, big):
        x0 = VIEW_W
        pygame.draw.rect(screen, PANEL, (x0, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, GRID, (x0, 0), (x0, SCREEN_H))
        screen.blit(big.render("GRACE 6DOF aircraft", True, INK), (x0 + 20, 16))
        for f in self.fields:
            f.draw(screen, font)
        for b in [self.b_target, self.b_plane, self.b_obs, self.b_clear,
                  self.b_reset]:
            b.draw(screen, font)
        screen.blit(font.render("cylinder cross-section plane:", True, MUTED),
                    (x0 + 20, 210))

        V = float(np.linalg.norm(self.state[:3]))
        alpha = np.rad2deg(np.arctan2(self.state[2], self.state[0]))
        beta = np.rad2deg(np.arcsin(np.clip(self.state[1] / max(V, 1e-6),
                                            -1.0, 1.0)))
        y = 350
        rows = [
            ("downrange", f"{self.state[IX]:7.1f} m"),
            ("lateral / alt", f"{self.state[IY]:6.1f} {self.state[IZ]:6.1f} m"),
            ("airspeed", f"{V:7.2f} m/s"),
            ("alpha / beta", f"{alpha:+6.1f} {beta:+6.1f} deg"),
            ("bank / pitch", f"{np.rad2deg(self.state[IPHI]):+6.1f} "
                             f"{np.rad2deg(self.state[ITHE]):+6.1f} deg"),
            ("heading", f"{np.rad2deg(self.state[IPSI]):+7.1f} deg"),
            ("cylinders", f"{len(self.obstacles)}"),
            ("solve time", f"{self.planner.solve_ema:6.2f} s"),
            ("horizon used", f"{self.planner.last_horizon:6.1f} s"),
        ]

        # Solve quality of the trajectory being flown.  Stationarity near zero
        # means the plan is optimal, not merely clear of the cylinders:
        if self.active is not None:
            rep = self.active.report
            rows.append(("stationarity", f"{rep['stationarity']:8.2e}"))
            rows.append(("endpoint err", f"{rep['endpoint_error']:8.2e}"))
            if np.isfinite(rep["worst_violation"]):
                rows.append(("worst violation", f"{rep['worst_violation']:+8.2e}"))

        clr = None
        if self.obstacles:
            clr = min(np.linalg.norm(self.state[list(PLANES[o["plane"]])]
                                     - o["centre"]) - o["radius"] - VEH_R
                      for o in self.obstacles)
            rows.append(("nearest cylinder", f"{clr:7.2f} m"))

        for i, (k, v) in enumerate(rows):
            screen.blit(font.render(k, True, MUTED), (x0 + 20, y + i * 22))
            screen.blit(font.render(v, True, INK), (x0 + 180, y + i * 22))

        base = y + len(rows) * 22 + 12
        if clr is not None:
            screen.blit(font.render("CLEAR" if clr >= 0 else "CONTACT", True,
                                    OK if clr >= 0 else DANGER), (x0 + 20, base))
        if self.active is not None:
            st = self.active.report["stationarity"]
            if np.isfinite(st):
                verdict, colour = (("OPTIMAL", OK) if st < 1e-1 else
                                   ("NEAR OPTIMAL", WARN) if st < 5e-1 else
                                   ("NOT CONVERGED", DANGER))
                screen.blit(font.render(verdict, True, colour), (x0 + 150, base))

        screen.blit(font.render(self.status, True, MUTED), (x0 + 20, SCREEN_H - 74))
        screen.blit(font.render("drag left pane to orbit", True, GRID),
                    (x0 + 20, SCREEN_H - 50))
        screen.blit(font.render("scroll to zoom, esc to quit", True, GRID),
                    (x0 + 20, SCREEN_H - 32))
        if self.planner.busy:
            screen.blit(font.render("solving...", True, WARN), (x0 + 220, 20))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("GRACE -- 6DOF aircraft with cylinder keep-outs")
    clock = pygame.time.Clock()
    font, big = _font(14), _font(19, bold=True)

    print("[sim] each horizon compiles once on first use", flush=True)
    sim = Sim()
    print("[sim] ready", flush=True)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                else:
                    sim.on_key(event)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                sim.on_mouse_down(event)
            elif event.type == pygame.MOUSEBUTTONUP:
                sim.on_mouse_up(event)
            elif event.type == pygame.MOUSEMOTION:
                sim.on_mouse_motion(event)

        sim.update()
        sim.draw(screen, font, big)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()