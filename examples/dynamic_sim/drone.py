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

# Solve quality comes from the package diagnostics rather than a copy here.  The
# stationarity it reports asks whether ANY valid multiplier set balances the cost
# gradient, not whether the solver's own does: an augmented Lagrangian can hold a
# constraint by penalty with its multiplier decayed to zero, which certifies
# nothing while the trajectory is in fact optimal:
from grace.utils.diagnostics import diagnostics


# --- Full 12-state quadcopter.  Thrust points along the body z axis only, so the
# --- vehicle translates by rolling and pitching: position is reached through
# --- torque -> body rate -> attitude -> thrust direction.
def quadcopter(x, u, m=1.0, Ixx=0.011, Iyy=0.011, Izz=0.021, drag=0.10, g=9.81):
    vx, vy, vz = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    p, q, r = x[9], x[10], x[11]
    T, tau_p, tau_q, tau_r = u[0], u[1], u[2], u[3]
    cph, sph = ca.cos(phi), ca.sin(phi)
    cth, sth = ca.cos(theta), ca.sin(theta)
    cps, sps = ca.cos(psi), ca.sin(psi)
    ex = cph * sth * cps + sph * sps
    ey = cph * sth * sps - sph * cps
    ez = cph * cth
    cth_safe = ca.fmax(ca.fabs(cth), 0.2) * ca.sign(cth + 1e-9)
    tth = sth / cth_safe
    return ca.vertcat(
        vx, vy, vz,
        (T / m) * ex - drag * vx,
        (T / m) * ey - drag * vy,
        (T / m) * ez - g - drag * vz,
        p + sph * tth * q + cph * tth * r,
        cph * q - sph * r,
        (sph / cth_safe) * q + (cph / cth_safe) * r,
        (tau_p + (Iyy - Izz) * q * r) / Ixx,
        (tau_q + (Izz - Ixx) * p * r) / Iyy,
        (tau_r + (Ixx - Iyy) * p * q) / Izz,
    )


# --- World and view configuration ---
WORLD = np.array([46.0, 30.0, 22.0])     # metres, x by y by z
PANEL_W = 330
SCREEN_W, SCREEN_H = 1500, 860
VIEW_W = SCREEN_W - PANEL_W

# Node count is fixed, because it sets the size of the solve.  How long the
# manoeuvre takes is set by dt, chosen per request from the distance to fly:
N_NODES = 90
FPS = 60
HOVER_T = 9.81
VEH_R = 0.4

# Cruise speed used to estimate how long a hop should take.  The estimate is
# then snapped to one of a few fixed horizons rather than used directly: each
# distinct timestep needs its own compiled system, so a continuously varying dt
# would compile a new one for nearly every target and the first solve at each
# would stall for minutes:
CRUISE = 4.0
HORIZON_TIMES = [3.0, 6.0, 10.0, 15.0, 22.0, 30.0]

# Attitude limits.  Without them the solve throws the airframe into whatever
# tilt closes the endpoint fastest, which looks nothing like flight:
TILT_MAX = np.deg2rad(30.0)

# Yaw is weighted far above the other channels.  Left at parity the minimum
# effort solution spins the airframe through hundreds of degrees on the way,
# because yaw is nearly free and buys a little translation; at this weight the
# heading holds to well under a degree:
YAW_WEIGHT = 1000.0

# Cross-section planes.  Each obstacle is a cylinder infinite along the axis it
# does not name, so a vertical tower is measured in x,y and a horizontal pipe in
# x,z or y,z:
PLANES = {"vertical (x,y)": (0, 1), "horizontal (x,z)": (0, 2), "horizontal (y,z)": (1, 2)}
PLANE_ORDER = list(PLANES.keys())

BG = (12, 14, 19)
GRID = (30, 36, 46)
GRID2 = (22, 27, 35)
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


# --- isometric camera -------------------------------------------------------
class Camera:

    def __init__(self):
        self.az = np.deg2rad(38.0)
        self.el = np.deg2rad(26.0)
        self.zoom = 15.0
        self.centre = WORLD / 2.0

    def project(self, p):

        # Rotate the world into camera axes, then drop the depth component.  A
        # plain axonometric projection keeps parallel lines parallel, which is
        # what makes the obstacle cylinders readable from any angle:
        d = np.asarray(p, float) - self.centre
        ca_, sa = np.cos(self.az), np.sin(self.az)
        ce, se = np.cos(self.el), np.sin(self.el)
        x = d[0] * ca_ - d[1] * sa
        y = d[0] * sa + d[1] * ca_
        u = x
        v = y * se - d[2] * ce
        return (int(VIEW_W / 2 + u * self.zoom), int(SCREEN_H / 2 + v * self.zoom))

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
        elif event.unicode and (event.unicode.isdigit() or event.unicode in "-.,"):
            self.text += event.unicode

    def value(self, default=0.0):
        try:
            return float(self.text.strip())
        except ValueError:
            return default

    def draw(self, screen, font):
        pygame.draw.rect(screen, FIELD_ON if self.active else FIELD, self.rect, border_radius=3)
        pygame.draw.rect(screen, GRID, self.rect, 1, border_radius=3)
        screen.blit(font.render(self.label, True, MUTED), (self.rect.x, self.rect.y - 16))
        screen.blit(font.render(self.text, True, INK), (self.rect.x + 6, self.rect.y + 4))


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


def hover_state(x, y, z):
    return [x, y, z, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def _font(size, bold=False):
    if not pygame.font.get_init():
        pygame.font.init()
    try:
        return pygame.font.SysFont("monospace", size, bold=bold)
    except Exception:
        return pygame.font.Font(None, size)


# Derive per-channel control weights from the endpoint Jacobian:
def channel_weights(system, U_hover):

    # The endpoint is thousands of times more sensitive to torque than to
    # thrust, because a small torque against a small inertia tilts the whole
    # thrust vector.  A plain minimum-norm solve dumps everything into the
    # torque channels and the vehicle tumbles.  Weighting each channel by the
    # square of its own endpoint sensitivity makes all four contribute
    # comparably, and the scale comes from the problem rather than a guess:
    _, Co = system.endpoint_jac(U_hover)
    Co = np.asarray(Co).reshape(system.m, system.N, system.nu)
    sens = np.array([np.linalg.norm(Co[:, :, j]) for j in range(system.nu)])
    w = (sens / sens.min()) ** 2
    w = w / w.min()

    # Thrust is the cheapest channel by endpoint sensitivity, which makes
    # altitude nearly free and lets the solve lob the vehicle 14 m up and back
    # on a level transit.  Lifting it keeps the flight level for the same tilt
    # and the same endpoint accuracy:
    w[0] *= 100.0
    w[3] *= YAW_WEIGHT
    return w


class Segment:

    # A segment carries the system it was planned on.  Horizons use different
    # timesteps, so stepping one segment with another's physics flies it at the
    # wrong rate and the tracker spends the flight fighting it:
    def __init__(self, Z, U, K, goal, dt, step_np, stat=float("nan"),
                 worst=float("-inf"), end_err=float("nan")):
        self.Z, self.U, self.K, self.goal, self.dt = Z, U, K, goal, dt
        self.step_np = step_np

        # Diagnostics from the solve that produced this segment, so the panel
        # can report whether the trajectory being flown is actually optimal
        # rather than merely feasible:
        self.stat = stat
        self.worst = worst
        self.end_err = end_err
        self.terminal = Z[-1].copy()
        self.N = len(Z) - 1

        # Frames of playback per plan step, so wall-clock time matches the dt
        # this segment was planned at:
        self.substeps = max(1, int(round(dt * FPS)))


# Mission planner: one GRACE solve per waypoint, on a worker thread:
class Planner:

    def __init__(self):

        # Systems are built on demand, one per timestep asked for, and cached
        # from then on.  A longer flight is a larger dt rather than a bigger
        # solve, so the cost of a request does not grow with its distance:
        self.levels = {}
        self.U_hover = np.tile(np.array([HOVER_T, 0.0, 0.0, 0.0]), N_NODES)

        # Compile every horizon now, with progress, rather than on the worker
        # thread where the first request of each would look like a hang:
        for i, t_horizon in enumerate(HORIZON_TIMES):
            print("[sim] preparing horizon %.0f s (%d of %d)"
                  % (t_horizon, i + 1, len(HORIZON_TIMES)), flush=True)
            self.level(t_horizon / N_NODES)

        # Tracking weights.  R carries the same channel scaling as the solve or
        # the torque gains swamp the thrust gain, and Qf lifts the terminal
        # weight so the gains do not decay to nothing over the last few steps:
        self.Q = np.diag([200., 200., 200., 20., 20., 20., 5., 5., 5., 1., 1., 1.])
        self.Qf = self.Q * 100.0

        self.requests = queue.Queue()
        self.results = queue.Queue()
        self.busy = False
        self.solve_ema = 3.0
        self.last_horizon = 0.0
        threading.Thread(target=self._worker, daemon=True).start()

    # Fetch, building on first use, the system for one timestep:
    def level(self, dt):
        key = round(dt, 4)
        if key not in self.levels:
            system = grace.build_cached(
                quadcopter, nx=12, nu=4, N=N_NODES, z0=hover_state(4, 15, 10),
                dt=key, job=f"quad_sim_dt{int(key * 1e4)}")
            self.levels[key] = dict(system=system, engine=grace.GRACE(system),
                                    dt=key,
                                    weights=channel_weights(system, self.U_hover))
        return self.levels[key]

    def _solve_at(self, lvl, start_state, goal, obstacles, U_warm=None):
        system, engine = lvl["system"], lvl["engine"]
        system.z0 = np.asarray(start_state, float)
        target = np.array(hover_state(goal[0], goal[1], goal[2]), float)
        W, U_hover = lvl["weights"], self.U_hover

        # Roll and pitch limits, as ordinary constraints.  Nothing marks them as
        # attitude: the solver treats them exactly as it treats a keep-out:
        tilt = [lambda z, u: z[6] - TILT_MAX, lambda z, u: -TILT_MAX - z[6],
                lambda z, u: z[7] - TILT_MAX, lambda z, u: -TILT_MAX - z[7]]

        # Solve the obstacle-free problem first.  It is cheap and gives the
        # augmented-Lagrangian rounds a trajectory that already reaches the
        # target, so they only have to bend it around the cylinders:
        # Warm start from the trajectory already being flown when there is one.
        # A replan is a small change to a good plan, and throwing it away to
        # start from hover costs an order of magnitude in optimality: measured
        # on a mid-flight obstacle, warm gives a certified 1.6e-05 where hover
        # gives 1.2e-01 on the same problem.  Sizes are matched to this horizon,
        # since a replan may land on a different one:
        U_start = U_hover
        if U_warm is not None:
            warm = np.asarray(U_warm, float).flatten()
            need = system.N * system.nu
            if warm.size >= need:
                U_start = warm[:need]
            else:
                U_start = np.concatenate(
                    [warm, np.tile(warm[-system.nu:], -(-(need - warm.size)
                                                        // system.nu))])[:need]

        U = engine.shooting.lambda_shoot(target, R_weights=W, U0=U_start)

        # Each cylinder becomes an expression g(z, u) <= 0 measured in the two
        # position states its cross-section plane names.  The centre, radius and
        # plane are closed over so every lambda keeps its own.  Nothing marks
        # these as obstacles: they and the tilt limits take the same path:
        cons = []
        for o in obstacles:
            a, b = PLANES[o["plane"]]
            c = np.asarray(o["centre"], float)
            r = o["radius"] + VEH_R
            cons.append(lambda z, u, a=a, b=b, c=c, r=r:
                        r ** 2 - ((z[a] - c[0]) ** 2 + (z[b] - c[1]) ** 2))

        U = engine.shooting.lambda_shoot(
            target, constraints=cons + tilt, R_weights=W, U0=U)

        Z = np.asarray(system.rollout(U))
        ok = not getattr(system, "_infeasible", False)
        K, _ = engine.tracking.lqr_gains(U, self.Q, self.R_for(lvl), Qf=self.Qf)

        # Stationarity: at an optimum the cost gradient is cancelled by the
        # endpoint and active-constraint multipliers, so what is left over says
        # how far this is from optimal rather than just feasible.  The active
        # threshold cannot be tiny, since a row the solve is holding sits a hair
        # inside its limit and dropping it makes a converged answer look wild:
        report = diagnostics(system, U, target, cons + tilt, R_weights=W)
        return (U, Z, K, ok, report["stationarity"], report["worst_violation"],
                report["endpoint_error"])

    def R_for(self, lvl):
        return 0.01 * np.diag(lvl["weights"])

    def _worker(self):
        while True:
            token, start_state, goal, obstacles, U_warm = self.requests.get()
            t0 = time.time()
            try:

                # Walk up the horizon ladder until the request comes back
                # feasible.  A short hop is solved at the shortest horizon and a
                # manoeuvre that has to squeeze past several cylinders is given
                # the extra time it needs, without the caller choosing N:
                # Estimate how long the hop should take from the distance to
                # fly, snap it up to the nearest prepared horizon, and step to
                # longer ones only when a request comes back infeasible.  A
                # short hop is flown briskly and an awkward one is given the
                # time it needs, with no compile at run time either way:
                span = float(np.linalg.norm(np.asarray(goal, float)
                                            - np.asarray(start_state, float)[:3]))
                want = span / CRUISE
                start_i = next((i for i, t in enumerate(HORIZON_TIMES) if t >= want),
                               len(HORIZON_TIMES) - 1)

                seg = None
                for t_horizon in HORIZON_TIMES[start_i:]:
                    lvl = self.level(t_horizon / N_NODES)
                    U, Z, K, ok, stat, worst, end_err = self._solve_at(
                        lvl, start_state, goal, obstacles, U_warm)
                    seg = Segment(Z, U, K, goal, lvl["dt"], lvl["system"].step_np,
                                  stat, worst, end_err)
                    self.last_horizon = lvl["dt"] * N_NODES
                    ok = (ok and worst < 1e-3 and end_err < 1e-3)
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
        self.state = np.array(hover_state(4, 15, 10), float)

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
        self.status = "enter a target on the right and press Add Target"

        # Control panel widgets:
        x0 = VIEW_W + 20
        self.f_tx = Field(x0, 60, 80, "target x", "34")
        self.f_ty = Field(x0 + 96, 60, 80, "target y", "22")
        self.f_tz = Field(x0 + 192, 60, 80, "target z", "10")
        self.b_target = Button(x0, 96, 268, 26, "Add Target")
        self.f_oa = Field(x0, 190, 80, "centre a", "19")
        self.f_ob = Field(x0 + 96, 190, 80, "centre b", "15")
        self.f_or = Field(x0 + 192, 190, 80, "radius", "3")
        self.b_plane = Button(x0, 226, 268, 26, PLANE_ORDER[0], MUTED)
        self.plane_i = 0
        self.b_obs = Button(x0, 258, 268, 26, "Add Cylinder")
        self.b_clear = Button(x0, 300, 130, 26, "Clear Cylinders", DANGER)
        self.b_reset = Button(x0 + 138, 300, 130, 26, "Reset", DANGER)
        self.fields = [self.f_tx, self.f_ty, self.f_tz, self.f_oa, self.f_ob, self.f_or]

    # --- planning ----------------------------------------------------------
    def _last_terminal(self):
        if self.segments:
            return self.segments[-1].terminal
        if self.active is not None:
            return self.active.terminal
        return self.state

    # Controls of whatever segment the next one will follow, if any:
    def _last_controls(self):
        if self.segments:
            return self.segments[-1].U
        if self.active is not None:
            return self.active.U
        return None

    def _predict_steps(self):
        substeps = self.active.substeps if self.active is not None else 4
        step_wall = substeps / float(FPS)
        n = int(np.ceil(self.planner.solve_ema / step_wall)) + 2
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
        # Re-plan from where the vehicle will be when the solve lands, and hand
        # over the controls it would have flown from that point.  Those are a
        # good starting guess for the new plan, which is the difference between
        # a certified optimum and a merely feasible trajectory:
        U_warm = None
        if self.active is not None:
            k = min(self.active_k + self._predict_steps(), self.active.N - 1)
            start = self.active.Z[k].copy()
            U_warm = self.active.U[k * 4:]
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
                self.cam.zoom = float(np.clip(self.cam.zoom, 4.0, 60.0))
            return
        for f in self.fields:
            f.click(event.pos)
        if self.b_plane.hit(event.pos):
            self.plane_i = (self.plane_i + 1) % len(PLANE_ORDER)
            self.b_plane.label = PLANE_ORDER[self.plane_i]
        elif self.b_target.hit(event.pos):
            w = np.array([self.f_tx.value(), self.f_ty.value(), self.f_tz.value()])
            self.waypoints.append(w)
            self.status = f"{len(self.waypoints)} target(s) queued"
            self._kick()
        elif self.b_obs.hit(event.pos):
            self.obstacles.append(dict(centre=np.array([self.f_oa.value(), self.f_ob.value()]),
                                       radius=max(self.f_or.value(1.0), 0.2),
                                       plane=PLANE_ORDER[self.plane_i]))
            self.status = "cylinder added -- replanning"
            self._replan()
        elif self.b_clear.hit(event.pos):
            self.obstacles.clear()
            self.status = "cylinders cleared -- replanning"
            self._replan()
        elif self.b_reset.hit(event.pos):
            self.state = np.array(hover_state(4, 15, 10), float)
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
        self.cam.el = float(np.clip(self.cam.el + dy * 0.005, np.deg2rad(5), np.deg2rad(85)))

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
                    d = np.linalg.norm(seg.Z[:, :3] - self.state[:3], axis=1)
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
                self.status = "target unreachable -- skipped"
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
            if self.active is not None:
                Z, U, K = self.active.Z, self.active.U, self.active.K
                k = min(self.active_k, self.active.N - 1)
                u = U[k * 4:(k + 1) * 4] - K[k] @ (self.state - Z[k])
                self.state = np.asarray(self.active.step_np(self.state, u), float)
                self.active_k += 1
                if self.active_k >= self.active.N:
                    self.active = None
                    if not self.segments and not self.waypoints:
                        self.status = "arrived -- enter another target"

        self.trail.append(self.state[:3].copy())
        if len(self.trail) > 2000:
            self.trail.pop(0)

    # --- drawing -----------------------------------------------------------
    def draw(self, screen, font, big):
        screen.fill(BG)
        self._draw_ground(screen)
        for o in sorted(self.obstacles, key=lambda o: -self._obs_depth(o)):
            self._draw_cylinder(screen, o)
        for seg in self.segments:
            self._polyline(screen, seg.Z[:, :3], FUTURE, 2)
        if self.active is not None:
            self._polyline(screen, self.active.Z[:, :3], ACCENT, 3)
        if len(self.trail) > 1:
            self._polyline(screen, np.array(self.trail), TRAIL, 2)
        for i, w in enumerate(([self.active.goal] if self.active is not None else [])
                              + [s.goal for s in self.segments] + list(self.waypoints)):
            self._marker(screen, w, ACCENT if i == 0 else MUTED, font, str(i + 1))
        self._draw_vehicle(screen)
        self._draw_panel(screen, font, big)

    def _polyline(self, screen, pts, colour, w):
        sp = [self.cam.project(p) for p in pts]
        if len(sp) > 1:
            pygame.draw.lines(screen, colour, False, sp, w)

    def _draw_ground(self, screen):
        for gx in range(0, int(WORLD[0]) + 1, 5):
            pygame.draw.line(screen, GRID2, self.cam.project([gx, 0, 0]),
                             self.cam.project([gx, WORLD[1], 0]))
        for gy in range(0, int(WORLD[1]) + 1, 5):
            pygame.draw.line(screen, GRID2, self.cam.project([0, gy, 0]),
                             self.cam.project([WORLD[0], gy, 0]))
        for a, b, c in [([0, 0, 0], [WORLD[0], 0, 0], (200, 90, 90)),
                        ([0, 0, 0], [0, WORLD[1], 0], (90, 200, 90)),
                        ([0, 0, 0], [0, 0, WORLD[2]], (90, 140, 220))]:
            pygame.draw.line(screen, c, self.cam.project(a), self.cam.project(b), 2)

    def _obs_depth(self, o):
        c = self._obs_centre3(o)
        return self.cam.depth(c)

    def _obs_centre3(self, o):
        cols = PLANES[o["plane"]]
        c = WORLD / 2.0
        c[cols[0]] = o["centre"][0]
        c[cols[1]] = o["centre"][1]
        return c

    def _draw_cylinder(self, screen, o):

        # A cylinder is infinite along the axis its plane does not name, so it
        # is drawn as two rings at the world bounds joined by rails:
        cols = PLANES[o["plane"]]
        axis = [i for i in (0, 1, 2) if i not in cols][0]
        th = np.linspace(0, 2 * np.pi, 28)
        rings = []
        for end in (0.0, WORLD[axis]):
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

    def _marker(self, screen, w, colour, font, label):
        s = self.cam.project(w)
        pygame.draw.circle(screen, colour, s, 8, 2)
        pygame.draw.line(screen, colour, (s[0] - 12, s[1]), (s[0] + 12, s[1]), 1)
        pygame.draw.line(screen, colour, (s[0], s[1] - 12), (s[0], s[1] + 12), 1)
        pygame.draw.line(screen, GRID, s, self.cam.project([w[0], w[1], 0]), 1)
        screen.blit(font.render(label, True, colour), (s[0] + 12, s[1] - 16))

    def _draw_vehicle(self, screen):
        pos = self.state[:3]
        phi, theta, psi = self.state[6], self.state[7], self.state[8]
        cph, sph = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cps, sps = np.cos(psi), np.sin(psi)

        # Body to world rotation, so the arms show real roll and pitch:
        R = np.array([
            [cth * cps, sph * sth * cps - cph * sps, cph * sth * cps + sph * sps],
            [cth * sps, sph * sth * sps + cph * cps, cph * sth * sps - sph * cps],
            [-sth,      sph * cth,                   cph * cth]])
        arm = 1.1
        centre = self.cam.project(pos)
        pygame.draw.line(screen, GRID, centre, self.cam.project([pos[0], pos[1], 0]), 1)
        for i, (dx, dy) in enumerate([(1, 0), (0, 1), (-1, 0), (0, -1)]):
            tip = pos + R @ np.array([dx * arm, dy * arm, 0.0])
            st = self.cam.project(tip)
            pygame.draw.line(screen, INK, centre, st, 2)
            pygame.draw.circle(screen, ACCENT if i == 0 else INK, st, 4)
        nose = pos + R @ np.array([1.9 * arm, 0.0, 0.0])
        pygame.draw.line(screen, (255, 175, 75), centre, self.cam.project(nose), 2)

    def _draw_panel(self, screen, font, big):
        x0 = VIEW_W
        pygame.draw.rect(screen, PANEL, (x0, 0, PANEL_W, SCREEN_H))
        pygame.draw.line(screen, GRID, (x0, 0), (x0, SCREEN_H))
        screen.blit(big.render("GRACE quadcopter", True, INK), (x0 + 20, 16))
        for f in self.fields:
            f.draw(screen, font)
        for b in [self.b_target, self.b_plane, self.b_obs, self.b_clear, self.b_reset]:
            b.draw(screen, font)
        screen.blit(font.render("cylinder cross-section plane:", True, MUTED), (x0 + 20, 210))

        y = 350
        rows = [
            ("position", f"{self.state[0]:5.1f} {self.state[1]:5.1f} {self.state[2]:5.1f}"),
            ("speed", f"{np.linalg.norm(self.state[3:6]):5.2f} m/s"),
            ("roll/pitch", f"{np.rad2deg(self.state[6]):+5.1f} {np.rad2deg(self.state[7]):+5.1f} deg"),
            ("yaw", f"{np.rad2deg(self.state[8]):+6.1f} deg"),
            ("cylinders", f"{len(self.obstacles)}"),
            ("queued", f"{len(self.waypoints) + len(self.segments)}"),
            ("solve time", f"{self.planner.solve_ema:5.2f} s"),
            ("horizon used", f"{self.planner.last_horizon:4.1f} s"),
        ]

        # Solve quality of the trajectory currently being flown.  Stationarity
        # near zero means the plan is optimal, not just collision free:
        if self.active is not None:
            rows.append(("stationarity", f"{self.active.stat:8.2e}"))
            rows.append(("endpoint err", f"{self.active.end_err:8.2e}"))
            if self.active.worst > float("-inf"):
                rows.append(("worst violation", f"{self.active.worst:+8.2e}"))
        clr = None
        if self.obstacles:
            clr = min(np.linalg.norm(self.state[list(PLANES[o["plane"]])] - o["centre"])
                      - o["radius"] - VEH_R for o in self.obstacles)
            rows.append(("nearest cylinder", f"{clr:5.2f} m"))
        for i, (k, v) in enumerate(rows):
            screen.blit(font.render(k, True, MUTED), (x0 + 20, y + i * 22))
            screen.blit(font.render(v, True, INK), (x0 + 170, y + i * 22))
        if clr is not None:
            screen.blit(font.render("CLEAR" if clr >= 0 else "CONTACT", True,
                                    OK if clr >= 0 else DANGER), (x0 + 20, y + len(rows) * 22 + 12))

        # Say plainly whether the plan being flown is optimal, near it, or only
        # feasible, so a suboptimal solve is visible rather than something to be
        # inferred from an exponent:
        if self.active is not None and np.isfinite(self.active.stat):
            st = self.active.stat
            verdict, colour = (("OPTIMAL", OK) if st < 1e-2 else
                               ("NEAR OPTIMAL", (220, 200, 90)) if st < 1e-1 else
                               ("NOT CONVERGED", DANGER))
            screen.blit(font.render(verdict, True, colour),
                        (x0 + 150, y + len(rows) * 22 + 12))

        screen.blit(font.render(self.status, True, MUTED), (x0 + 20, SCREEN_H - 74))
        screen.blit(font.render("drag left pane to orbit", True, GRID), (x0 + 20, SCREEN_H - 50))
        screen.blit(font.render("scroll to zoom, esc to quit", True, GRID), (x0 + 20, SCREEN_H - 32))
        if self.planner.busy:
            screen.blit(font.render("solving...", True, (255, 175, 75)), (x0 + 200, 20))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("GRACE -- 3D quadcopter with cylinder keep-outs")
    clock = pygame.time.Clock()
    font, big = _font(14), _font(19, bold=True)

    print("[sim] ready; each new horizon compiles once on first use")
    sim = Sim()
    print("[sim] ready")

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