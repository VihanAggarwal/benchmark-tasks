"""Offline tests for scoring.py (and the rigstate hooks, the offline push model and the demo in task.py): synthetic
object states and synthetic arm snapshots fed through rigstate.offline_feed, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import math
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, both_seated, cart_low, cart_on_runner, centre_off, end_centres, finger_zone, hands_off, on_bed,
    point_box_distance, rotation, seat_of, tilt_deg, valid, yaw_err_deg, zone_gap, zone_radius, CART_HALF,
)

HZ = 20.0
DT = 1.0 / HZ
TOP = L.TOP
TIPS = 0.013  # pad tips below the TCP
OPEN, CLOSED = 1.0, 0.0
GRIPPED = L.RUNNER_SIZE[0] / L.FULL_GAP  # rigstate opening with the fingers closed on a 40 mm runner
Q0 = np.array([0.0, 0.0, 0.0, 1.0])
READY_L = np.array([0.323, 0.305, TOP + 0.26])
READY_R = np.array([0.323, -0.305, TOP + 0.26])
PUSH_Z = TOP + 0.060
FRONT = 0.008  # closed fingers' outer face ahead of the TCP
L_PARK = np.array([0.230, 0.200, TOP + 0.150])
L_END = np.array([0.220, 0.110, TOP + 0.150])
R_PARK = np.array([0.200, -0.250, TOP + 0.150])
HC = L.CART_HALF[2]
HR = L.RUNNER_HALF[2]


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def box(centre, quat, half, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c = np.array(centre, float)
    ext = np.abs(rotation(quat)) @ np.array(half, float)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def support_low(cy):
    """Height of the cart's underside above the table when slid along the path to centre y ``cy``: on the dock
    while its rear is on the dock's main block, on the seated runners until its rear is past them, then on the bed."""
    rear = cy - L.CART_HALF[1]
    if rear < -L.SEAT_Y:
        return L.DOCK_TOP
    if rear < 0.080:
        return L.RUNNER_TOP_SEATED
    return L.BED_TOP


def cart(x=L.CART_X, y=L.CART_Y, low=None, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """The cart with its underside ``low`` above the table (default: resting on the path's surfaces)."""
    low = support_low(y) if low is None else low
    return box((x, y, TOP + low + HC), quat, L.CART_HALF, vel, spin)


def tipped_cart(y, deg, low=None):
    """The cart turned ``deg`` about its front bottom edge (its rear lifting), the edge staying put."""
    low = support_low(y) if low is None else low
    pivot = np.array([L.CART_X, y + L.CART_HALF[1], TOP + low])
    q = about("x", -deg)
    centre = pivot + rotation(q) @ np.array([0.0, -L.CART_HALF[1], HC])
    return box(centre, q, L.CART_HALF)


def runner(x, y, bottom=0.0, yaw=0.0, quat=None, vel=(0, 0, 0)):
    """A runner lying flat with its underside ``bottom`` above the table; yaw 0 = its length along y."""
    q = about("z", yaw) if quat is None else quat
    return box((x, y, TOP + bottom + HR), q, L.RUNNER_HALF, vel)


def seated(key, dx=0.0, dy=0.0, bottom=L.SEAT_FLOOR, yaw=0.0):
    return runner(L.SEATED_X[key] + dx, dy, bottom, yaw)


RA0 = runner(*L.RUNNERS["runner_a"])
RB0 = runner(*L.RUNNERS["runner_b"])
SA, SB = seated("a"), seated("b")
TARGET = cart(L.TARGET_X, L.TARGET_Y)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


AWAY = dict(left_tcp=L_END, right_tcp=R_PARK)  # both grippers clear at the end


# -- a trial as a film ------------------------------------------------------------------------------------------
SCALARS = ("cx", "cy", "cl", "cyaw", "ax", "ay", "az", "ayaw", "bx", "by", "bz", "byaw", "Lo", "Ro")
VECTORS = ("L", "R")
START = {"cx": L.CART_X, "cy": L.CART_Y, "cl": 0.0, "cyaw": 0.0,
         "ax": L.RUNNERS["runner_a"][0], "ay": L.RUNNERS["runner_a"][1], "az": 0.0, "ayaw": 0.0,
         "bx": L.RUNNERS["runner_b"][0], "by": L.RUNNERS["runner_b"][1], "bz": 0.0, "byaw": 0.0,
         "Lo": OPEN, "Ro": OPEN, "L": READY_L, "R": READY_R}


class Film:
    """A trial as a list of frames: the cart (centre x, y, a lift over the path's surface, yaw), both runners
    (centre x, y, underside height, yaw), both TCPs and openings. ``go`` moves linearly to new values over n
    control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = dict(START)
        self.cur.update({k: np.array(v, float) if k in VECTORS else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VECTORS else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            self.frames.append({k: (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
                                if k in VECTORS else (1 - f) * a[k] + f * b[k] for k in a})
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)


def states(fr):
    c = cart(fr["cx"], fr["cy"], support_low(fr["cy"]) + fr["cl"], about("z", fr["cyaw"]))
    a = runner(fr["ax"], fr["ay"], fr["az"], fr["ayaw"])
    b = runner(fr["bx"], fr["by"], fr["bz"], fr["byaw"])
    return c, a, b


def with_vel(s, prev):
    if prev is None:
        return s
    return SimpleNamespace(**{**vars(s), "lin_vel": (s.pos - prev) / DT})


def play(film, tracker=None, arms=True):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r, prev = None, (None, None, None)
    for fr in film.frames:
        objs = states(fr)
        objs_v = [with_vel(s, p) for s, p in zip(objs, prev)]
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(*objs_v, a)
        prev = tuple(s.pos for s in objs)
    return r, t


def lay(f, keys=("b", "a")):
    """The left arm lays the runners into their seats, one at a time (B first, as the demo does)."""
    for k in keys:
        x0, y0 = f.cur[k + "x"], f.cur[k + "y"]
        sx = L.SEATED_X[k]
        f.go(20, L=[x0, y0, TOP + 0.10], Lo=OPEN)
        f.go(15, L=[x0, y0, TOP + 0.017]).go(10, Lo=GRIPPED)
        f.go(15, **{k + "z": 0.058}, L=[x0, y0, TOP + 0.075])
        f.go(40, **{k + "x": sx, k + "y": 0.0}, L=[sx, 0.0, TOP + 0.075])
        f.go(30, **{k + "z": L.SEAT_FLOOR}, L=[sx, 0.0, TOP + L.SEAT_FLOOR + 0.017])
        f.go(10, Lo=OPEN).go(10, L=[sx, 0.0, TOP + 0.075])
    return f.go(20, L=L_PARK, Lo=CLOSED)


def right_push(f, to=0.050, n=160):
    """The right arm pushes the cart from behind until its centre is at y ``to``, backs off and parks."""
    y = f.cur["cy"]
    f.go(10, R=[L.CART_X, y - 0.05 - FRONT, PUSH_Z], Ro=CLOSED)
    f.go(n, cy=to, R=[L.CART_X, to - 0.05 - FRONT, PUSH_Z])
    f.go(10, R=[L.CART_X, to - 0.05 - FRONT - 0.04, PUSH_Z]).go(10, R=[L.CART_X, to - 0.098, TOP + 0.12])
    return f.go(20, R=R_PARK).go(10, Ro=OPEN)


def left_push(f, to=L.TARGET_Y, n=140, let_go=True):
    """The left arm comes down behind the cart, pushes it to centre y ``to``, backs off, moves clear, opens."""
    y = f.cur["cy"]
    f.go(20, L=[L.CART_X, -0.040, TOP + 0.08]).go(10, L=[L.CART_X, -0.040, PUSH_Z])
    f.go(10, L=[L.CART_X, y - 0.05 - FRONT, PUSH_Z])
    f.go(n, cy=to, L=[L.CART_X, to - 0.05 - FRONT, PUSH_Z])
    if let_go:
        f.go(10, L=[L.CART_X, to - 0.098, PUSH_Z]).go(10, L=[L.CART_X, to - 0.098, TOP + 0.11])
        f.go(20, L=L_END).go(10, Lo=OPEN)
    return f


def procedure(to=L.TARGET_Y, settle=30, **kw):
    """The intended trial: runners into their seats, the right arm pushes the cart past the gap, the left arm
    pushes it the rest of the way, both arms clear; still for ``settle`` steps."""
    f = Film().hold(10)
    f.go(20, R=[L.CART_X, L.CART_Y - 0.05 - FRONT - 0.004, PUSH_Z], Ro=CLOSED)
    lay(f)
    right_push(f)
    left_push(f, to, **kw)
    return f.hold(settle)


# -- the tests --------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_layout_steps_down_along_the_path(self):
        self.assertGreater(L.DOCK_TOP, L.RUNNER_TOP_SEATED)
        self.assertGreater(L.RUNNER_TOP_SEATED, L.BED_TOP)
        self.assertAlmostEqual(L.DOCK_TOP - L.RUNNER_TOP_SEATED, 0.0005, places=9)
        self.assertAlmostEqual(L.RUNNER_TOP_SEATED - L.BED_TOP, 0.0005, places=9)
        self.assertLess(L.RIB_TOP, L.RUNNER_TOP_SEATED - 0.005)  # the rib never touches the cart
        # a runner fits each pair of seats with 5 mm of play at each side and each end
        for x0, x1 in L.SEATS.values():
            self.assertAlmostEqual((x1 - x0) - L.RUNNER_SIZE[0], 0.010, places=9)
        self.assertAlmostEqual(2 * L.SEAT_Y - L.RUNNER_SIZE[1], 0.010, places=9)
        # the cart overlaps each centred runner by 30 mm and its centre lies between them
        self.assertAlmostEqual((L.SEATED_X["a"] + 0.020) - (L.CART_X - 0.050), 0.030, places=9)
        self.assertAlmostEqual((L.CART_X + 0.050) - (L.SEATED_X["b"] - 0.020), 0.030, places=9)

    def test_start_layout(self):
        self.assertAlmostEqual(cart_low(cart()), L.DOCK_TOP, places=9)
        self.assertIsNone(seat_of(RA0))
        self.assertIsNone(seat_of(RB0))
        self.assertTrue(Tracker.at_start(cart(), RA0, RB0))
        self.assertFalse(on_bed(cart()))

    def test_seated(self):
        self.assertEqual(seat_of(SA), "a")
        self.assertEqual(seat_of(SB), "b")
        self.assertTrue(both_seated(SA, SB))
        self.assertTrue(both_seated(SB, SA))  # either runner in either pair
        for dx in (-0.005, 0.005):
            self.assertEqual(seat_of(seated("a", dx=dx)), "a")  # against a seat's side
        for dy in (-0.005, 0.005):
            self.assertEqual(seat_of(seated("b", dy=dy)), "b")  # against a seat's end
        self.assertIsNone(seat_of(seated("a", dx=0.0065)))  # past the seat's side: up on the rib
        self.assertIsNone(seat_of(seated("a", dy=0.0065)))  # past the seat's end
        self.assertIsNone(seat_of(seated("a", bottom=L.SEAT_FLOOR + 0.0025)))  # held 2.5 mm up
        self.assertEqual(seat_of(seated("a", bottom=L.SEAT_FLOOR + 0.0015)), "a")
        self.assertEqual(seat_of(seated("a", yaw=3.0)), "a")  # turned a little in its seats
        self.assertIsNone(seat_of(seated("a", yaw=8.0)))  # its corners out of the seats
        self.assertEqual(seat_of(seated("b", yaw=180.0)), "b")  # either end on the dock
        # flat, either face up; on edge or tilted is not seated
        c = np.array([L.SEATED_X["a"], 0.0, TOP + L.SEAT_FLOOR + HR])
        self.assertEqual(seat_of(box(c, about("y", 180), L.RUNNER_HALF)), "a")
        self.assertIsNone(seat_of(box(c, about("x", 6), L.RUNNER_HALF)))
        self.assertIsNone(seat_of(box(c, about("y", 90), L.RUNNER_HALF)))
        # rolled about its length (its lowest edge on the floors): 4 degrees is flat enough, 6 is not
        for deg, want in ((4.0, "a"), (6.0, None)):
            q = about("y", deg)
            ext = (np.abs(rotation(q)) @ np.array(L.RUNNER_HALF))[2]
            self.assertEqual(seat_of(box((L.SEATED_X["a"], 0.0, TOP + L.SEAT_FLOOR + ext), q, L.RUNNER_HALF)), want)
        # standing on edge in the gap on the table, top flush with the dock: not seated
        self.assertIsNone(seat_of(box((0.300, 0.0, TOP + 0.020), about("y", 90), L.RUNNER_HALF)))
        # two runners in the same pair (impossible in the simulator) are not both seated
        self.assertFalse(both_seated(SA, seated("a", dx=0.004)))
        # lying across the gap on the dock and bed tops: not seated
        self.assertIsNone(seat_of(runner(0.245, 0.0, L.DOCK_TOP)))

    def test_on_bed_and_target(self):
        self.assertTrue(on_bed(TARGET))
        self.assertAlmostEqual(centre_off(TARGET), 0.0, places=9)
        self.assertFalse(on_bed(cart(y=0.130)))  # its rear still over the seats
        self.assertTrue(on_bed(cart(y=0.135)))
        self.assertFalse(on_bed(cart(y=0.251)))  # its front past the bed's far edge
        self.assertTrue(on_bed(cart(x=0.360, y=0.210)))
        self.assertFalse(on_bed(cart(x=0.361, y=0.210)))  # its side past the bed's x 0.410 edge
        self.assertFalse(on_bed(cart(y=0.210, low=L.BED_TOP + 0.0025)))
        self.assertAlmostEqual(yaw_err_deg(about("z", 93)), 3.0, places=6)
        self.assertAlmostEqual(yaw_err_deg(about("z", -4)), 4.0, places=6)

    def test_cart_on_runner(self):
        self.assertFalse(cart_on_runner(cart(), SA))
        self.assertTrue(cart_on_runner(cart(y=0.0), SA))
        self.assertTrue(cart_on_runner(cart(y=-0.124), SA))  # its front 6 mm over the runner's end
        self.assertFalse(cart_on_runner(cart(y=-0.134), SA))
        self.assertFalse(cart_on_runner(cart(y=0.0, low=0.044), SA))  # 4.5 mm above the runner
        self.assertFalse(cart_on_runner(cart(y=0.0), runner(0.150, 0.0)))  # a runner lying on the table

    def test_tilt_is_folded(self):
        self.assertAlmostEqual(tilt_deg(about("y", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 176)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)
        self.assertAlmostEqual(cart_low(tipped_cart(-0.2, 11)), L.DOCK_TOP, places=9)

    def test_finger_zone_and_distance(self):
        # the real finger: 10 mm thick and 40 mm wide over its lowest 38 mm (review lesson): 60 mm open, 22 closed
        self.assertAlmostEqual(zone_radius(SimpleNamespace(opening=1.0)), math.hypot(L.FULL_GAP / 2 + 0.010, 0.020))
        self.assertAlmostEqual(zone_radius(SimpleNamespace(opening=0.0)), math.hypot(0.010, 0.020))
        self.assertGreater(zone_radius(SimpleNamespace(opening=1.0)), 0.060)
        self.assertGreater(zone_radius(SimpleNamespace(opening=0.0)), 0.022)
        a = SimpleNamespace(tcp=np.array([0.3, 0.0, TOP + 0.2]), tips=np.array([0.3, 0.0, TOP + 0.187]),
                            opening=0.0)
        pts, _ = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], TOP + 0.187)
        self.assertAlmostEqual(pts[-1][2], TOP + 0.187 + L.FINGER_LEN)
        self.assertAlmostEqual(point_box_distance([L.CART_X, L.CART_Y, TOP + 0.1], cart(), CART_HALF),
                               0.1 - L.DOCK_TOP - L.CART_SIZE[2], places=9)
        self.assertAlmostEqual(zone_gap(a, cart(y=0.0), CART_HALF), 0.187 - 0.0695 - zone_radius(a), places=9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        t = Tracker(HZ)
        r = t.update(cart(), RA0, RB0, snap())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)
        self.assertEqual(t.breaches, {})
        self.assertIsNotNone(t.both_seated_step)
        self.assertLess(t.both_seated_step, t.handover_step)
        self.assertAlmostEqual(t.min_low, L.BED_TOP, places=9)
        self.assertAlmostEqual(t.max_low, L.DOCK_TOP, places=9)

    def test_score_follows_the_relay(self):
        f = Film().hold(5)
        _, t = play(f)
        self.assertEqual(t.last.score, 0.0)
        lay(f)
        r, _ = play(f)
        self.assertEqual(r.score, 0.25)  # both runners seated
        right_push(f, to=0.030)
        self.assertEqual(play(f)[0].score, 0.25)  # the cart's centre short of y +0.040
        f = Film().hold(5)
        lay(f)
        right_push(f)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)  # the right arm's leg done: centre at y +0.050
        right_push(Film(), to=0.130)
        f2 = Film().hold(5)
        lay(f2)
        right_push(f2, to=0.130)  # as far as the right arm could ever push it: not yet on the bed
        self.assertEqual(play(f2)[0].score, 0.5)

    def test_on_bed_tiers(self):
        for to, want in ((L.TARGET_Y, 1.0), (L.TARGET_Y + 0.014, 1.0), (L.TARGET_Y - 0.014, 1.0),
                         (L.TARGET_Y + 0.016, 0.75), (L.TARGET_Y - 0.039, 0.75), (L.TARGET_Y + 0.039, 0.75),
                         (L.TARGET_Y - 0.041, 0.5), (L.TARGET_Y - 0.076, 0.5)):
            with self.subTest(to=to):
                r, _ = play(procedure(to))
                self.assertEqual(r.score, want, (to, r.details["centre_off_mm"], r.details["on_bed"]))

    def test_the_cart_must_be_square(self):
        f = procedure()
        f.go(10, cyaw=4.5).hold(30)
        self.assertEqual(play(f)[0].score, 1.0)
        f = procedure()
        f.go(10, cyaw=6.0).hold(30)
        self.assertEqual(play(f)[0].score, 0.75)

    def test_a_gripper_left_at_the_cart_scores_075(self):
        f = procedure(let_go=False)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["hands_off"])

    def test_hands_off_is_30_mm(self):
        a = snap(left_tcp=L_END, right_tcp=R_PARK)
        self.assertTrue(hands_off(a, TARGET, SA, SB))
        # an open gripper straight above the cart: its zone (60 mm) 30 mm clear once the tips are 90 mm over it
        top = TOP + L.BED_TOP + L.CART_SIZE[2]
        r0 = zone_radius(SimpleNamespace(opening=1.0))
        for up, want in ((r0 + 0.0305, True), (r0 + 0.0295, False)):
            a = snap(left_tcp=[L.TARGET_X, L.TARGET_Y, top + up + TIPS], right_tcp=R_PARK)
            self.assertEqual(hands_off(a, TARGET, SA, SB), want)
        # near a runner counts as well
        a = snap(left_tcp=L_END, right_tcp=[L.SEATED_X["a"], 0.0, TOP + L.RUNNER_TOP_SEATED + TIPS + 0.04],
                 right_open=CLOSED)
        self.assertFalse(hands_off(a, TARGET, SA, SB))

    def test_a_gripper_closed_well_clear_counts_as_let_go(self):
        a = snap(left_tcp=L_END, left_open=CLOSED, right_tcp=R_PARK, right_open=CLOSED)
        self.assertTrue(hands_off(a, TARGET, SA, SB))

    def test_two_runners_in_one_pair_score_nothing(self):
        # Not possible in the simulator (they would overlap), but the scorer must not count one pair twice.
        t = Tracker(HZ)
        r = t.update(TARGET, SA, seated("a", dx=0.004), snap(**AWAY))
        self.assertEqual(r.score, 0.0)
        self.assertEqual((r.details["seat_a"], r.details["seat_b"]), ("a", "a"))

    def test_a_runner_out_of_its_seat_at_the_end(self):
        f = procedure()
        f.go(10, ax=L.SEATED_X["a"] - 0.040).hold(30)  # knocked out of its seats after the cart crossed
        self.assertEqual(play(f)[0].score, 0.0)
        f = procedure()
        f.go(10, bz=L.SEAT_FLOOR + 0.004).hold(30)  # lifted out
        self.assertEqual(play(f)[0].score, 0.0)

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)
        moving = SimpleNamespace(**{**vars(TARGET), "lin_vel": np.array([0.0, 0.02, 0.0])})
        r = t.update(moving, SA, SB, snap(**AWAY))
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r = t.update(TARGET, SA, SB, snap(**AWAY))
        self.assertTrue(r.success)
        r = t.update(TARGET, SA, SimpleNamespace(**{**vars(SB), "ang_vel": np.array([0.6, 0, 0])}), snap(**AWAY))
        self.assertFalse(r.success)

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        a = snap(**AWAY)
        before = (t.steps, dict(t.breaches), t.min_low, dict(t.max_shift))
        for _ in range(3):
            self.assertEqual(t.result(TARGET, SA, SB, a).score, r.score)
        self.assertEqual((t.steps, dict(t.breaches), t.min_low, dict(t.max_shift)), before)


class ZeroConditions(unittest.TestCase):
    def test_cart_pushed_at_an_open_gap_drops(self):
        f = Film().hold(5)
        f.go(60, cy=-0.080)  # its centre past the dock's seats with nothing there: it falls in
        f.go(5, cl=-0.030).hold(5)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("dropped", t.breaches)
        lay(f)  # laying the runners afterwards does not undo it
        f.go(5, cl=0.0, cy=-0.200).hold(20)
        self.assertEqual(play(f)[0].score, 0.0)

    def test_drop_limit_is_35_mm(self):
        t = Tracker(HZ)
        t.update(cart(low=0.0351), RA0, RB0, snap())
        self.assertEqual(t.breaches, {})
        t.update(cart(low=0.0349), RA0, RB0, snap())
        self.assertIn("dropped", t.breaches)

    def test_lifted(self):
        t = Tracker(HZ)
        t.update(cart(low=L.DOCK_TOP + 0.0045), RA0, RB0, snap())
        self.assertEqual(t.breaches, {})
        f = procedure()
        f.frames = f.frames[:400] + [dict(f.frames[400], cl=0.006)] + f.frames[400:]
        r, t = play(f)
        self.assertIn("lifted", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_tipped(self):
        t = Tracker(HZ)
        t.update(tipped_cart(-0.2, 9.5), RA0, RB0, snap())
        self.assertEqual(t.breaches, {})
        r = t.update(tipped_cart(-0.2, 10.5), RA0, RB0, snap())
        self.assertIn("tipped", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_a_runner_sliding_under_the_cart(self):
        # Runner A set against the dock seat's end, then slid along its seats toward the bed seat's end: it stays
        # seated (5 mm of play each way, 1 mm tolerance), so only the 10 mm rule decides.
        for shift, broken in ((0.0095, False), (0.0105, True)):
            with self.subTest(shift=shift):
                f = Film().hold(5)
                lay(f)
                f.go(5, ay=-0.005)
                right_push(f, to=0.0)  # the cart on both runners
                f.go(10, ay=-0.005 + shift)
                f.go(10, ay=-0.005)  # and back again: the record stays
                right_push(f, to=0.050)
                left_push(f).hold(30)
                r, t = play(f)
                self.assertEqual("runner_slid" in t.breaches, broken)
                self.assertNotIn("open_gap", t.breaches)
                self.assertEqual(r.score, 0.0 if broken else 1.0)
                self.assertGreaterEqual(t.max_shift["runner_a"], shift - 1e-9)
        # slid further, out of its seats, with the cart over the gap: both rules
        f = Film().hold(5)
        lay(f)
        right_push(f, to=0.0)
        f.go(10, ay=0.012)
        _, t = play(f)
        self.assertEqual(set(t.breaches), {"runner_slid", "open_gap"})

    def test_a_runner_turning_under_the_cart(self):
        f = Film().hold(5)
        lay(f)
        right_push(f, to=0.0)
        f.go(10, byaw=8.0)  # its ends swing 11 mm
        _, t = play(f)
        self.assertIn("runner_slid", t.breaches)

    def test_moving_a_runner_with_the_cart_elsewhere_is_no_breach(self):
        f = Film().hold(5)
        lay(f)
        f.go(10, bx=L.SEATED_X["b"] + 0.004, by=0.004).go(10, bx=L.SEATED_X["b"], by=0.0)  # adjusted in its seats
        right_push(f)
        left_push(f).hold(30)
        r, t = play(f)
        self.assertEqual(t.breaches, {})
        self.assertEqual(r.score, 1.0)

    def test_a_runner_moved_while_the_cart_is_off_it(self):
        # Each time the cart comes onto a runner, the runner's movement is measured from where it is then.
        f = Film().hold(5)
        lay(f)
        right_push(f, to=0.0)
        f.go(40, cy=-0.150, R=[L.CART_X, -0.300, PUSH_Z])  # the cart drawn back onto the dock, off the runners
        f.go(10, ay=0.012)  # runner A moved while nothing is on it
        f.go(60, cy=0.0)  # and the cart pushed back onto it
        r, t = play(f)
        self.assertNotIn("runner_slid", t.breaches)
        self.assertLess(t.max_shift["runner_a"], 0.001)

    def test_anything_off_the_table(self):
        f = procedure(settle=5)
        f.go(5, by=0.70, bz=-0.3).hold(5)
        r, t = play(f)
        self.assertIn("off_table", t.breaches)
        self.assertEqual(r.score, 0.0)
        t = Tracker(HZ)
        t.update(cart(), RA0, runner(0.15, 0.29, bottom=-0.011), snap())
        self.assertIn("off_table", t.breaches)

    def test_a_breach_is_never_undone(self):
        t = Tracker(HZ)
        t.update(cart(low=0.050), RA0, RB0, snap())
        r, _ = play(procedure(), t)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.breaches["lifted"], 1)


class Exploits(unittest.TestCase):
    def test_runners_on_edge_in_the_gap(self):
        # Two runners stood on edge on the table in the gap fill it flush with the dock: the cart can cross, but the
        # runners are not in their seats, so it scores nothing, and seating them once the cart is across does not
        # help: the cart came over the gap before both were seated.
        t = Tracker(HZ)
        ra = box((0.300, -0.020, TOP + 0.020), about("y", 90), L.RUNNER_HALF)
        rb = box((0.350, 0.020, TOP + 0.020), about("y", 90), L.RUNNER_HALF)
        r = None
        for y in np.linspace(L.CART_Y, L.TARGET_Y, 200):
            r = t.update(cart(y=y, low=L.DOCK_TOP if y < 0 else L.BED_TOP), ra, rb, snap(**AWAY))
        self.assertEqual(set(t.breaches), {"open_gap"})
        self.assertTrue(on_bed(cart(y=L.TARGET_Y)))
        self.assertEqual(r.score, 0.0)
        for _ in range(30):
            r = t.update(TARGET, SA, SB, snap(**AWAY))
        self.assertEqual(r.score, 0.0)

    def test_crossing_before_both_runners_are_seated(self):
        # The cart held up by the grippers and carried over the open gap (never dropping, never 5 mm up), the
        # runners laid afterwards: the bridge was never used, 0.
        f = Film().hold(5)
        f.go(10, R=[L.CART_X, L.CART_Y - 0.058, PUSH_Z], Ro=CLOSED)
        f.go(120, cy=L.TARGET_Y, cl=0.001, R=[L.CART_X, L.TARGET_Y - 0.058, PUSH_Z])
        f.go(5, cl=0.0).go(20, R=R_PARK)
        lay(f).go(20, L=L_END, Lo=OPEN).hold(30)
        r, t = play(f)
        self.assertNotIn("dropped", t.breaches)
        self.assertNotIn("lifted", t.breaches)
        self.assertIn("open_gap", t.breaches)
        self.assertEqual(r.score, 0.0)
        # one runner seated, the cart taken over it (held level), the other runner seated after it crossed: 0
        f = Film().hold(5)
        lay(f, keys=("a",))
        right_push(f, to=L.TARGET_Y)
        lay(f, keys=("b",)).go(20, L=L_END, Lo=OPEN).hold(30)
        r, t = play(f)
        self.assertIn("open_gap", t.breaches)
        self.assertEqual(r.score, 0.0)
        # the cart's front up to the gap's edge before the runners are in is fine (1 mm tolerance); further is not
        for front, broken in ((-L.GAP_Y + 0.0009, False), (-L.GAP_Y + 0.0011, True)):
            with self.subTest(front=front):
                t = Tracker(HZ)
                t.update(cart(y=front - L.CART_HALF[1], low=L.DOCK_TOP), RA0, RB0, snap())
                self.assertEqual("open_gap" in t.breaches, broken)
        # and a runner taken out of its seats while the cart is over the gap (not yet on it) breaks it too
        f = Film().hold(5)
        lay(f)
        right_push(f, to=-0.088)  # the cart's front 2 mm over the gap, short of the runners' load
        f.go(10, az=L.SEAT_FLOOR + 0.010)
        _, t = play(f)
        self.assertIn("open_gap", t.breaches)
        self.assertNotIn("runner_slid", t.breaches)

    def test_carrying_the_cart_over_the_gap(self):
        f = Film().hold(5)
        lay(f)
        f.go(10, cl=0.010).go(80, cy=L.TARGET_Y).go(10, cl=0.0).hold(30)  # lifted 10 mm and carried across
        r, t = play(f)
        self.assertIn("lifted", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_a_runner_used_as_a_sled(self):
        # The cart dragged across on a runner that slides with it: the runner moves under the load.
        f = Film().hold(5)
        lay(f)
        right_push(f, to=-0.080)
        f.go(60, cy=0.020, ay=0.080)
        _, t = play(f)
        self.assertIn("runner_slid", t.breaches)

    def test_pushed_only_partway_onto_the_bed(self):
        r, _ = play(procedure(to=0.150))  # on the bed, 60 mm short
        self.assertEqual(r.score, 0.5)

    def test_one_arm_lays_the_runners_only(self):
        f = Film().hold(5)
        lay(f).go(20, L=L_END, Lo=OPEN).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.success)


class Arms(unittest.TestCase):
    def test_arms_unavailable_taken_as_met_and_counted(self):
        f = procedure()
        r, t = play(f, arms=False)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(f.frames))
        self.assertIsNone(r.details["hands_off"])

    def test_hands_off_applies_when_the_arms_come_back(self):
        r, t = play(procedure(let_go=False), arms=False)
        self.assertEqual(r.score, 1.0)
        f = procedure(let_go=False)
        last = f.frames[-1]
        r = t.update(*states(last), snap(last["L"], last["Lo"], last["R"], last["Ro"]))
        self.assertEqual(r.score, 0.75)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0]), tips=np.zeros(3), opening=1.0),
               "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3), opening=1.0)}
        t.update(cart(), RA0, RB0, bad)
        t.update(cart(), RA0, RB0, {"left": bad["right"]})
        t.update(cart(), RA0, RB0, {"left": bad["right"], "right": SimpleNamespace(tcp=np.zeros(3),
                                                                                   tips=np.zeros(3),
                                                                                   opening=float("inf"))})
        self.assertEqual(t.arms_unavailable, 3)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=[0.1, 0.2, TOP + 0.3], left_open=0.5)
        self.assertAlmostEqual(a["left"].opening, 0.5)
        np.testing.assert_allclose(a["left"].tips, [0.1, 0.2, TOP + 0.3 - TIPS])
        r = Tracker(HZ).update(cart(), RA0, RB0, a)
        self.assertTrue(r.details["arms_available"])
        self.assertEqual(r.details["opening"]["left"], 0.5)


class Robustness(unittest.TestCase):
    def test_turned_over_cart_and_runners_score_the_same(self):
        f = procedure()
        r, t = play(f)
        flipped = box(TARGET.pos, about("x", 180), L.CART_HALF)
        ra = box(SA.pos, about("y", 180), L.RUNNER_HALF)
        rb = box(SB.pos, qmul(about("z", 180), about("x", 180)), L.RUNNER_HALF)
        self.assertEqual(t.result(flipped, ra, rb, snap(**AWAY)).score, 1.0)
        self.assertAlmostEqual(cart_low(flipped), L.BED_TOP, places=9)
        self.assertAlmostEqual(float(np.max(np.abs(end_centres(rb) - end_centres(SB)))), 0.0, places=9)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        f = procedure(settle=5)
        frames = list(f.frames)
        f.frames = frames[:-25]
        _, t = play(f)
        before = (dict(t.breaches), dict(t.max_shift), t.min_low, t.max_low, t.prev["cart"][0].tolist(), t.steps)
        nan_pos = SimpleNamespace(**{**vars(TARGET), "pos": np.array([np.nan, 0.0, 0.0])})
        inf_vel = SimpleNamespace(**{**vars(SA), "lin_vel": np.array([np.inf, 0.0, 0.0])})
        zero_q = SimpleNamespace(**{**vars(SB), "quat_xyzw": np.zeros(4)})
        short = SimpleNamespace(**{**vars(TARGET), "pos": np.array([0.3, 0.2])})
        for c, a, b in ((nan_pos, SA, SB), (TARGET, inf_vel, SB), (TARGET, SA, zero_q), (short, SA, SB),
                        (None, SA, SB), (TARGET, SimpleNamespace(pos=[0, 0, 0]), SB)):
            r = t.update(c, a, b, snap(**AWAY))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
            self.assertEqual(t.result(c, a, b, snap(**AWAY)).score, 0.0)
        self.assertEqual(t.bad_steps, 6)
        self.assertEqual((dict(t.breaches), dict(t.max_shift), t.min_low, t.max_low, t.prev["cart"][0].tolist()),
                         before[:5])
        rest = Film()
        rest.frames = frames[-25:]
        r, _ = play(rest, t)  # the next good states are scored as before
        self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker(HZ)
        for _ in range(300):
            objs = []
            for half in (L.CART_HALF, L.RUNNER_HALF, L.RUNNER_HALF):
                q = rng.normal(size=4)
                objs.append(box(rng.uniform([-0.5, -1, TOP - 0.5], [1, 1, TOP + 1]), q / np.linalg.norm(q), half,
                                rng.normal(size=3), rng.normal(size=3)))
            a = snap(rng.uniform(-1, 1, 3), rng.uniform(0, 1), rng.uniform(-1, 1, 3), rng.uniform(0, 1))
            r = t.update(*objs, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
        self.assertEqual(t.update(cart(), RA0, RB0, None).score, 0.0)


class Reset(unittest.TestCase):
    def broken(self):
        f = Film().hold(5)
        f.go(60, cy=-0.080).go(5, cl=-0.030)  # dropped into the gap
        lay(f)
        _, t = play(f)
        self.assertIn("dropped", t.breaches)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.broken()
        r = t.update(cart(), RA0, RB0, snap())  # everything back in one step: a new episode
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.steps, 1)
        self.assertIsNone(t.both_seated_step)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_reset_after_a_full_trial(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        t.update(cart(), RA0, RB0, snap())
        self.assertIsNone(t.handover_step)
        self.assertEqual(t.result(cart(), RA0, RB0, snap()).score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.broken()
        t.update(cart(L.CART_X + 0.02, L.CART_Y - 0.02), runner(0.17, 0.27), runner(0.14, 0.48), snap())
        self.assertEqual(t.breaches, {})

    def test_gradual_return_does_not_reset(self):
        t = self.broken()
        f = Film(cy=-0.080, cl=-0.030, ax=L.SEATED_X["a"], ay=0.0, az=L.SEAT_FLOOR, bx=L.SEATED_X["b"], by=0.0,
                 bz=L.SEAT_FLOOR)
        f.go(5, cl=0.0).go(60, cy=L.CART_Y)
        f.go(100, ax=L.RUNNERS["runner_a"][0], ay=L.RUNNERS["runner_a"][1], az=0.0,
             bx=L.RUNNERS["runner_b"][0], by=L.RUNNERS["runner_b"][1], bz=0.0).hold(10)
        play(f, t)
        self.assertIn("dropped", t.breaches)  # back at the start, but gradually: the same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.broken()
        t.update(cart(y=0.2), RA0, RB0, snap())  # knocked far, not to the start
        self.assertIn("dropped", t.breaches)

    def test_a_drop_onto_the_start_is_not_a_teleport(self):
        # A runner dropped from 0.3 m straight onto its starting place: fast, but its velocity explains it.
        t = self.broken()
        z, v = 0.3, 0.0
        t.update(cart(), RA0, runner(*L.RUNNERS["runner_b"], bottom=z), snap())
        while z > 0:
            v += 9.81 * DT
            z = max(0.0, z - v * DT)
            t.update(cart(), RA0, runner(*L.RUNNERS["runner_b"], bottom=z, vel=(0, 0, -v if z > 0 else 0)), snap())
        self.assertIn("dropped", t.breaches)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.broken()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual(t.breaches, {})
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


class PushModel(unittest.TestCase):
    """offline_push.py, the offline checker's stand-in for pushing (the simulator never uses it)."""

    def model(self, runners=(SA, SB)):
        from offline_push import CartPush
        m = CartPush(("runner_a", "runner_b"))
        m.reset()
        m._runner_hook("runner_a")(runners[0])
        m._runner_hook("runner_b")(runners[1])
        return m

    def push_to(self, m, y_tcp_end, steps=200, z=PUSH_Z, side="right"):
        s = None
        for y in np.linspace(L.CART_Y - 0.05 - FRONT - 0.004, y_tcp_end, steps):
            tcp = [L.CART_X, y, z]
            if side == "right":
                snap(left_tcp=L_PARK, left_open=CLOSED, right_tcp=tcp, right_open=CLOSED)
            else:
                snap(left_tcp=tcp, left_open=CLOSED, right_tcp=R_PARK, right_open=CLOSED)
            s = m.cart_state(cart())
        return s

    def test_a_push_from_behind_moves_the_cart_over_seated_runners(self):
        m = self.model()
        s = self.push_to(m, 0.0045)
        self.assertFalse(m.fallen)
        self.assertAlmostEqual(float(s.pos[1]), 0.0045 + FRONT + 0.05, places=6)
        self.assertAlmostEqual(cart_low(s), L.RUNNER_TOP_SEATED, places=6)
        self.assertEqual(m.warnings, [])

    def test_no_runners_it_falls_in(self):
        m = self.model((RA0, RB0))
        s = self.push_to(m, 0.0045)
        self.assertTrue(m.fallen)
        self.assertLess(cart_low(s), L.CART_DROP)
        self.assertGreater(tilt_deg(s.quat_xyzw), L.CART_TIP_DEG)
        self.assertLess(float(s.pos[1]), -L.SEAT_Y + 0.05 + 0.003)  # as its centre passed the dock's seats

    def test_one_runner_cannot_carry_it(self):
        for runners in ((SA, RB0), (RA0, SB)):
            with self.subTest(seated=[seat_of(r) for r in runners]):
                m = self.model(runners)
                self.assertTrue(m.support(L.CART_X, -0.040)[1])  # its rear still on the dock: carried
                self.assertFalse(m.support(L.CART_X, -0.030)[1])  # its rear off the dock, on one runner: tips
                self.assertFalse(m.support(L.CART_X, 0.0)[1])
        m = self.model()
        for y in np.linspace(L.CART_Y, 0.25, 60):
            self.assertTrue(m.support(L.CART_X, y)[1], y)  # both: carried the whole way

    def test_coming_down_onto_the_cart_is_recorded(self):
        m = self.model()
        snap(right_tcp=[L.CART_X, L.CART_Y, TOP + 0.12], right_open=CLOSED)
        m.cart_state(cart())
        snap(right_tcp=[L.CART_X, L.CART_Y + 0.01, TOP + 0.06], right_open=CLOSED)
        m.cart_state(cart())
        self.assertTrue(any(w["issue"] == "came down onto the cart" for w in m.warnings))

    def test_no_arms_no_push(self):
        m = self.model()
        rigstate.offline_reset()
        s = m.cart_state(cart())
        np.testing.assert_allclose(s.pos, cart().pos)


@unittest.skipIf(_offline_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, its hooks, its rigstate reset hook, the
    demo, and the offline checker on the whole folder."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir() / "yam_stub"))
        cls.task = importlib.import_module("task")

    def objects(self, c=None, a=RA0, b=RB0):
        return {"cart": c if c is not None else cart(), "runner_a": a, "runner_b": b}

    def test_objects(self):
        objs = self.task.OBJECTS
        names = [o.name for o in objs]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in objs if o.mass is not None], ["runner_a", "runner_b", "cart"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        # the spawned loose objects are the starting states the scorer expects
        spec = {o.name: o for o in objs}
        np.testing.assert_allclose(spec["cart"].pos, cart().pos)
        np.testing.assert_allclose(spec["runner_a"].pos, RA0.pos)
        np.testing.assert_allclose(spec["runner_b"].pos, RB0.pos)
        # the fixed blocks: every block's top under the cart's path is the dock's, the bed's or below a runner's
        for o in objs:
            if o.mass is not None or o.name.startswith("mark"):
                continue
            top = o.pos[2] + o.size[2] / 2 - TOP
            x0, x1 = o.pos[0] - o.size[0] / 2, o.pos[0] + o.size[0] / 2
            y0, y1 = o.pos[1] - o.size[1] / 2, o.pos[1] + o.size[1] / 2
            self.assertGreaterEqual(x0, L.X0 - 1e-9)
            self.assertLessEqual(x1, L.X1 + 1e-9)
            under_path = x1 > L.CART_X - 0.05 and x0 < L.CART_X + 0.05
            if under_path and y1 <= 0:
                self.assertIn(round(top, 4), (L.DOCK_TOP, L.SEAT_FLOOR, L.RIB_TOP), o.name)
            elif under_path:
                self.assertIn(round(top, 4), (L.BED_TOP, L.SEAT_FLOOR, L.RIB_TOP), o.name)
            if under_path and round(top, 4) in (L.DOCK_TOP, L.BED_TOP):  # no seam across the path at its level
                self.assertTrue(x0 <= L.CART_X - 0.05 and x1 >= L.CART_X + 0.05, o.name)
        # each seat pair: floors and rib between the cheeks, the gap open to the table
        for o in objs:
            if o.mass is None and not o.name.startswith("mark"):
                y0, y1 = o.pos[1] - o.size[1] / 2, o.pos[1] + o.size[1] / 2
                self.assertTrue(y1 <= -L.GAP_Y + 1e-9 or y0 >= L.GAP_Y - 1e-9, o.name)

    def test_offline_specs_carry_the_push_hooks(self):
        spec = {o.name: o for o in self.task.OBJECTS}
        self.assertTrue(callable(getattr(spec["cart"], "offline_state", None)))
        self.assertTrue(callable(getattr(spec["runner_a"], "offline_state", None)))
        centre, size, yaw = spec["cart"].offline_box()
        self.assertEqual(tuple(size), L.CART_SIZE)
        names = [o.name for o in self.task.OBJECTS]
        self.assertLess(names.index("runner_b"), names.index("cart"))  # the push model reads the runners first

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects(cart(low=0.050))))  # lifted
        self.assertTrue(tr.breaches)
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.task._PUSH.pose = (0.0, 0.0, 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker and push model start over
        self.assertEqual(tr.breaches, {})
        self.assertEqual(tr.steps, 0)
        self.assertIsNone(self.task._PUSH.pose)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects(a=SA, b=SB)))
        self.assertEqual(self.task.score(self.objects(a=SA, b=SB)), 0.25)
        self.assertFalse(self.task.success({"cart": cart()}))  # missing runners: no crash, scores 0
        self.assertEqual(self.task.score({"cart": cart()}), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        span = {a: sum(w.duration for w in plan[a]) for a in plan}
        t_of = {a: [sum(w.duration for w in plan[a][:i + 1]) for i in range(len(plan[a]))] for a in plan}

        def when(arm, start):
            i = next(i for i, w in enumerate(plan[arm]) if w.label.startswith(start))
            return t_of[arm][i] - plan[arm][i].duration, t_of[arm][i]

        # the right arm pushes only once both runners are in and the left arm is parked out of the way
        seated = when("left", "out of the cart's way")[1]
        self.assertGreaterEqual(when("right", "push the cart")[0], seated - 1e-9)
        # the left arm comes over only once the right arm has backed off and risen
        self.assertGreaterEqual(when("left", "over the cart to behind")[0], when("right", "up")[1] - 1e-9)
        self.assertLess(span["right"], span["left"])
        # every open or close takes 1.5 s or more, and every final descent of 2 cm or more to a grip or a seat too
        for wps in (left, right):
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
        for w in left:
            if w.label.startswith("slowly down"):
                self.assertGreaterEqual(w.duration, 1.5)
        # pushes: fingers closed, yaw 0, at the push height, along y only
        for arm, label in (("right", "push the cart"), ("left", "push the cart")):
            w = next(w for w in plan[arm] if w.label.startswith(label))
            self.assertEqual((w.finger, w.yaw), (self.task.CLOSED, 0.0))
            self.assertAlmostEqual(w.pos[0], L.CART_X)
            self.assertAlmostEqual(w.pos[2], self.task.PUSH_Z)
        self.assertAlmostEqual(self.task.L_END_Y, L.TARGET_Y - 0.05 - FRONT + self.task.TRAIL_L)
        self.assertLess(self.task.R_END_Y, 0.075)  # the right arm's reach at the push height
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_offline_checker_predicts_one_and_the_push_stays_clean(self):
        """tools/offline/check_task.py --strict --one-arm on this folder (no reach table, for time), and its
        playback's push model: no gripper ever came down onto the cart. Run in-process with sys.modules and sys.path
        restored afterwards (the checker reloads the task's helper modules)."""
        saved_modules, saved_path = dict(sys.modules), list(sys.path)
        sys.path.insert(0, str(_offline_dir()))
        try:
            import check_task
            rep, _, code = check_task.check(HERE, strict=True, one_arm=True, reach=False)
            self.assertEqual(code, 0, rep.fails)
            self.assertEqual(rep.data["playback"]["proxy"]["predicted_task_score"], 1.0)
            self.assertEqual(rep.data["one_arm"], {"left_only": 0.25, "right_only": 0.0})
            task = check_task.load_task(HERE / "task.py", HERE)
            plans = check_task.get_plans(task, check_task.Report())
            solids = [check_task.G.solid_from_spec(o, task.table_top) for o in task.objects]
            out = check_task.Playback(task, plans, solids, hz=HZ, hold_s=3.0).run()
            self.assertEqual(out["proxy"]["predicted_task_score"], 1.0)
            tr = task.module._TRACKER
            self.assertEqual(task.module._PUSH.warnings, [])
            self.assertEqual(tr.breaches, {})
            self.assertGreaterEqual(tr.min_low, L.BED_TOP - 1e-9)
            self.assertLess(tr.both_seated_step, tr.handover_step)
            check_task._forget_task_modules()
        finally:
            for name in [n for n in sys.modules if n not in saved_modules]:
                del sys.modules[name]
            sys.modules.update(saved_modules)
            sys.path[:] = saved_path


if __name__ == "__main__":
    unittest.main()
