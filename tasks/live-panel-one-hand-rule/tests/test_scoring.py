"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic fuse states and synthetic arm snapshots
fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import math
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
    Tracker, clear_of_panel, in_panel, in_socket, in_tray, over_panel, panel_gap, park_margin, parked, rotation,
    seated, tilt_deg, valid, zone_box_gap, zone_radius, ZONE_HI, ZONE_LO,
)

HZ = 20.0
DT = 1.0 / HZ
HALF = L.FUSE_HEIGHT / 2
TIPS = 0.013  # pad tips below the TCP
GRIP_DEPTH = 0.046  # TCP above a held fuse's bottom (as in the demo)
HELD = GRIP_DEPTH - HALF  # a held fuse's centre is this far below the TCP
OPEN, GRIP = 1.0, 2 * L.FUSE_RADIUS / 0.0939  # rigstate openings: fully open; closed on a fuse (stopped at 20 mm)

R_START = np.array([L.RED_X, L.RED_Y, L.SOCKET_FLOOR + HALF])
G_START = np.array([L.CUP_X, L.CUP_Y, L.TOP + HALF])
SEATED = R_START.copy()
ON_PAD = np.array([L.HANDOFF_X, L.HANDOFF_Y, L.HANDOFF_TOP + HALF])
IN_TRAY = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + HALF])
BESIDE = np.array([0.16, 0.12, L.TOP + HALF])  # on the table, outside the panel, not in the tray
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
PARK_L = np.array([L.PARK_X, L.PARK_Y["left"], L.TOP + 0.05])
PARK_R = np.array([L.PARK_X, L.PARK_Y["right"], L.TOP + 0.05])
WAIT_L = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + 0.14])
CARRY = L.TOP + 0.12
HOVER = L.TOP + 0.12
OVER_SOCKET = np.array([L.SOCKET_X, L.SOCKET_Y, HOVER])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def cyl(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like fuse (20 x 50 mm cylinder): centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    axis = rotation(quat)[:, 2]
    ext = np.abs(axis) * HALF + L.FUSE_RADIUS * np.sqrt(np.clip(1 - axis ** 2, 0, 1))
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def arm(tcp, opening=OPEN, tips=None):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp, tcp - [0, 0, TIPS] if tips is None else np.array(tips, float), float(opening))


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("r", "rq", "g", "gq", "L", "Lo", "R", "Ro")
VEC = ("r", "g", "L", "R")


class Film:
    """A trial as a list of frames: both fuses' centres and orientations, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"r": R_START, "rq": Q0, "g": G_START, "gq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in ("rq", "gq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        if prev is None:
            vr = vg = np.zeros(3)
        else:
            vr, vg = (fr["r"] - prev["r"]) / DT, (fr["g"] - prev["g"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(cyl(fr["r"], fr["rq"], vr), cyl(fr["g"], fr["gq"], vg), a)
        prev = fr
    return r, t


def _xy(p, z):
    return [float(p[0]), float(p[1]), z]


def right_stages(f, park=PARK_R):
    """The right arm takes the green spare out of its cup, sets it on the hand-off pad and goes to ``park``."""
    grip = G_START + [0, 0, HELD]
    f.go(30, R=grip + [0, 0, 0.074]).go(30, R=grip).go(30, Ro=GRIP)
    f.go(30, R=_xy(G_START, CARRY), g=_xy(G_START, CARRY - HELD))
    f.go(40, R=_xy(ON_PAD, CARRY), g=_xy(ON_PAD, CARRY - HELD))
    f.go(30, R=ON_PAD + [0, 0, HELD + 0.002], g=ON_PAD + [0, 0, 0.002]).go(2, g=ON_PAD).go(30, Ro=OPEN)
    f.go(24, R=_xy(ON_PAD, HOVER)).go(40, R=_xy(park, HOVER)).go(24, R=park)
    return f


def left_pulls_red(f, to=IN_TRAY):
    """The left arm goes into the panel, pulls the red fuse out and sets it down at ``to`` (on the table)."""
    grip = R_START + [0, 0, HELD]
    f.go(40, L=grip + [0, 0, 0.07]).go(30, L=grip).go(30, Lo=GRIP)
    f.go(30, L=_xy(R_START, CARRY), r=_xy(R_START, CARRY - HELD))
    f.go(40, L=_xy(to, CARRY), r=_xy(to, CARRY - HELD))
    f.go(20, L=to + [0, 0, HELD + 0.005], r=to + [0, 0, 0.005]).go(2, r=to).go(30, Lo=OPEN)
    f.go(20, L=_xy(to, HOVER))
    return f


def left_takes_spare(f):
    """The left arm picks the spare off the hand-off pad and lifts it."""
    grip = ON_PAD + [0, 0, HELD]
    f.go(40, L=grip + [0, 0, 0.07]).go(30, L=grip).go(30, Lo=GRIP)
    f.go(30, L=_xy(ON_PAD, CARRY), g=_xy(ON_PAD, CARRY - HELD))
    return f


def left_sets_spare(f, at=SEATED, gq=Q0, let_go=True):
    """The left arm carries the spare (already held) to ``at`` and sets it down there."""
    f.go(40, L=_xy(at, CARRY), g=_xy(at, CARRY - HELD))
    f.go(40, L=at + [0, 0, HELD + 0.002], g=at + [0, 0, 0.002]).go(2, g=at, gq=gq)
    if let_go:
        f.go(30, Lo=OPEN).go(24, L=_xy(at, HOVER))
    return f


def procedure(red_to=IN_TRAY, green_at=SEATED, gq=Q0, leave=True, let_go=True, settle=25):
    """The whole job in the demo's order: stage the spare and park the right arm, pull the red fuse, seat the spare,
    leave the panel; then hold still."""
    f = right_stages(Film())
    left_pulls_red(f, red_to)
    left_takes_spare(f)
    left_sets_spare(f, green_at, gq, let_go)
    if leave:
        f.go(30, L=WAIT_L)
    return f.hold(settle)


# -- geometry ---------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_plain_fuse(self):
        self.assertAlmostEqual(tilt_deg(Q0), 0.0, places=6)
        self.assertAlmostEqual(tilt_deg(about("x", 180)), 0.0, places=6)  # upside down: the ends are alike
        self.assertAlmostEqual(tilt_deg(about("y", 170)), 10.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 90.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 70)), 0.0, places=6)  # turning about its own axis

    def test_finger_zone_radius(self):
        self.assertAlmostEqual(zone_radius(arm(PARK_R, OPEN)), math.hypot(0.0939 / 2 + 0.008, 0.0135), places=6)
        self.assertAlmostEqual(zone_radius(arm(PARK_R, OPEN)), 0.0566, places=4)  # fully open
        self.assertAlmostEqual(zone_radius(arm(PARK_R, 0.213)), 0.0225, places=3)  # closed on a 20 mm fuse
        self.assertAlmostEqual(zone_radius(arm(PARK_R, 0.0)), math.hypot(0.008, 0.0135), places=6)

    def test_zone_box_gap(self):
        lo, hi = np.zeros(3), np.ones(3)
        down = lambda tcp, o=OPEN: arm(tcp, o)  # noqa: E731 - a straight-down gripper
        r = zone_radius(down([0, 0, 0]))
        self.assertAlmostEqual(zone_box_gap(down([0.5, 0.5, 1.2]), lo, hi), 1.2 - TIPS - 1.0, places=6)  # flat end
        self.assertLess(zone_box_gap(down([0.5, 0.5, 1.0]), lo, hi), 0.0)  # tips inside
        self.assertAlmostEqual(zone_box_gap(down([1.1, 0.5, 0.5]), lo, hi), 0.1 - r, places=6)  # beside a face
        self.assertLess(zone_box_gap(down([1.0 + r - 0.002, 0.5, 0.5]), lo, hi), 0.0)  # fingers over the face
        self.assertGreater(zone_box_gap(down([1.0 + r + 0.002, 0.5, 0.5]), lo, hi), 0.0)
        d = 1.0 + (r + 0.003) / math.sqrt(2)  # beside a vertical edge, diagonally
        self.assertGreater(zone_box_gap(down([d, d, 0.5]), lo, hi), 0.0)
        side = arm([1.3, 0.5, 0.5], OPEN, tips=[1.3 - TIPS, 0.5, 0.5])  # pointing at a face, level
        self.assertAlmostEqual(zone_box_gap(side, lo, hi), 0.3 - TIPS, places=6)

    def test_gripper_in_panel(self):
        c = np.array([L.PANEL_X, L.PANEL_Y, 0.0])
        r_open, r_shut = zone_radius(arm(c, OPEN)), zone_radius(arm(c, GRIP))
        self.assertTrue(in_panel(arm(c + [0, 0, L.TOP + 0.05])))
        self.assertTrue(in_panel(arm(c + [0.075, -0.075, L.TOP + 0.10])))  # near a corner
        edge = L.PANEL_HALF  # from the panel's centre to its front edge
        # open fingers hang over the outline although the gripper's axis is outside it: inside
        self.assertTrue(in_panel(arm(c + [edge + r_open - 0.005, 0, L.TOP + 0.05], OPEN)))
        self.assertFalse(in_panel(arm(c + [edge + r_open + 0.005, 0, L.TOP + 0.05], OPEN)))
        # the same place with the fingers closed on a fuse: outside
        self.assertFalse(in_panel(arm(c + [edge + r_open - 0.005, 0, L.TOP + 0.05], GRIP)))
        self.assertTrue(in_panel(arm(c + [edge + r_shut - 0.003, 0, L.TOP + 0.05], GRIP)))
        self.assertFalse(in_panel(arm(c + [0, 0, L.ZONE_TOP_Z + TIPS + 0.002])))  # above the live zone
        self.assertTrue(in_panel(arm(c + [0, 0, L.ZONE_TOP_Z + TIPS - 0.002])))  # only the finger tips in it
        self.assertEqual(panel_gap(arm(c + [0, 0, L.TOP + 0.05])), 0.0)
        self.assertAlmostEqual(panel_gap(arm([L.PANEL_X, L.PANEL_Y - L.PANEL_HALF - 0.09, L.TOP + 0.05])),
                               0.09 - r_open, places=6)
        self.assertTrue(np.allclose(ZONE_LO[:2], [0.22, 0.07]) and np.allclose(ZONE_HI[:2], [0.38, 0.23]))

    def test_demo_places_clear_the_panel(self):
        # the right gripper fully open over the hand-off pad's centre, and the left waiting over the tray, are
        # outside the panel with room to spare (the pad is 3.5 cm from the panel's corner)
        self.assertGreater(panel_gap(arm(ON_PAD + [0, 0, 0.05], OPEN)), 0.02)
        self.assertGreater(panel_gap(arm(WAIT_L, OPEN)), 0.04)
        self.assertGreater(panel_gap(arm(PARK_R, OPEN)), 0.15)

    def test_parked(self):
        self.assertTrue(parked(arm(PARK_R), "right"))
        self.assertTrue(parked(arm(PARK_R + [0.045, -0.045, 0.045]), "right"))  # inside, near the zone's edges
        self.assertFalse(parked(arm(PARK_R + [0.055, 0, 0]), "right"))
        self.assertFalse(parked(arm(PARK_R + [0, 0, 0.06]), "right"))  # 11 cm up: hovering, not parked
        self.assertFalse(parked(arm(PARK_R), "left"))  # the other arm's square does not count
        self.assertTrue(parked(arm(PARK_L), "left"))
        self.assertAlmostEqual(park_margin(arm(PARK_R), "right"), 0.05)

    def test_seated_and_in_socket(self):
        self.assertTrue(seated(cyl(SEATED)))
        self.assertTrue(seated(cyl(SEATED, about("y", 180))))
        self.assertTrue(seated(cyl(SEATED + [0.005, -0.005, 0])))  # against a corner of the 30 mm opening
        self.assertTrue(seated(cyl(SEATED + [0, 0, 0.0025])))
        self.assertFalse(seated(cyl(SEATED + [0, 0, 0.004])))  # held 4 mm up
        self.assertFalse(seated(cyl(SEATED + [0, 0, L.SOCKET_HEIGHT])))  # standing on the walls
        self.assertFalse(seated(cyl(SEATED + [0.012, 0, 0])))
        tilted = cyl(SEATED, about("x", 12))
        tilted.pos = tilted.pos + [0, 0, float(L.SOCKET_FLOOR - tilted.low[2])]
        tilted.low, tilted.high = tilted.low + [0, 0, L.SOCKET_FLOOR - tilted.low[2]], tilted.high
        self.assertFalse(seated(tilted))
        self.assertTrue(in_socket(cyl(R_START)))
        self.assertTrue(in_socket(cyl(R_START + [0, 0, 0.02])))  # half pulled: still in
        self.assertFalse(in_socket(cyl(R_START + [0, 0, L.SOCKET_HEIGHT])))  # its bottom at the rim: out
        self.assertFalse(in_socket(cyl(BESIDE)))

    def test_tray_and_panel_outline(self):
        self.assertTrue(in_tray(cyl(IN_TRAY)))
        self.assertTrue(in_tray(cyl(IN_TRAY + [0.035, -0.035, 0])))
        lying = cyl(IN_TRAY + [0, 0, L.FUSE_RADIUS - HALF], about("x", 90))
        self.assertTrue(in_tray(lying))
        self.assertFalse(in_tray(cyl(IN_TRAY + [0, 0, 0.03])))  # held above it
        self.assertFalse(in_tray(cyl(IN_TRAY + [0.045, 0, L.TRAY_HEIGHT])))  # on the rim
        self.assertTrue(clear_of_panel(cyl(BESIDE)) and clear_of_panel(cyl(IN_TRAY)) and clear_of_panel(cyl(ON_PAD)))
        self.assertFalse(clear_of_panel(cyl([L.PANEL_X - L.PANEL_HALF - 0.005, 0.15, L.TOP + HALF])))  # on the edge
        self.assertFalse(clear_of_panel(cyl(R_START)))
        self.assertTrue(over_panel(cyl(SEATED)) and over_panel(cyl(SEATED + [0, 0, 0.09])))
        self.assertFalse(over_panel(cyl(SEATED + [0, 0, 0.15])))  # above the live zone
        self.assertFalse(over_panel(cyl(ON_PAD)))

    def test_layout_reach_split(self):
        # what the task's two-arm need rests on (straight-down reach from tools/offline/yamkin.py, checked there):
        # the spare's cup is on the right arm's side, the socket and the tray on the left's, the pad on the midline
        self.assertLess(L.CUP_Y, -0.30)
        self.assertGreater(L.SOCKET_Y, 0.12)
        self.assertGreater(L.TRAY_Y, 0.25)
        self.assertEqual(L.HANDOFF_Y, 0.0)
        pad_hi = L.HANDOFF_Y + L.HANDOFF_SIZE[1] / 2
        self.assertGreater(ZONE_LO[1] - pad_hi, 0.03)  # the pad is well outside the panel


# -- tiers -------------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(5))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIsNone(t.panel_hand)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.panel_hand, "left")
        self.assertFalse(t.breach)
        self.assertEqual(t.steps_inside["right"], 0)
        self.assertGreater(t.min_panel_gap["right"], 0.02)  # its open fingers over the hand-off pad
        self.assertGreaterEqual(t.min_park_margin, 0.0)
        self.assertTrue(r.details["green_seated"] and r.details["red_in_tray"] and r.details["grippers_out"])

    def test_red_beside_the_panel_scores_075(self):
        r, _ = play(procedure(red_to=BESIDE))
        self.assertEqual(r.score, 0.75)
        self.assertTrue(r.details["red_clear_of_panel"])

    def test_gripper_left_in_the_panel_scores_075(self):
        r, _ = play(procedure(leave=False))  # opened, but still 12 cm over the socket: inside
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(leave=False, let_go=False))  # still holding the seated spare
        self.assertEqual(r.score, 0.75)

    def test_red_left_inside_the_panel_scores_half(self):
        r, _ = play(procedure(red_to=np.array([L.PANEL_X + 0.05, L.PANEL_Y + 0.05, L.PLATE_TOP + HALF])))
        self.assertEqual(r.score, 0.5)

    def test_green_not_seated_scores_half(self):
        on_rim = SEATED + [0.02, 0, L.SOCKET_HEIGHT]  # standing on the socket's walls
        r, _ = play(procedure(green_at=on_rim))
        self.assertEqual(r.score, 0.5)
        beside = np.array([L.PANEL_X - 0.05, L.PANEL_Y - 0.05, L.PLATE_TOP + HALF])  # on the plate, not in the socket
        r, _ = play(procedure(green_at=beside))
        self.assertEqual(r.score, 0.5)

    def test_green_held_over_the_socket_scores_half(self):
        f = right_stages(Film())
        left_pulls_red(f)
        left_takes_spare(f)
        f.go(40, L=OVER_SOCKET, g=OVER_SOCKET - [0, 0, HELD]).hold(25)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_red_in_the_tray_only_scores_quarter(self):
        f = right_stages(Film())
        left_pulls_red(f)
        f.go(30, L=WAIT_L).hold(25)
        r, _ = play(f)
        self.assertEqual(r.score, 0.25)

    def test_red_set_aside_but_not_in_the_tray_scores_zero(self):
        f = right_stages(Film())
        left_pulls_red(f, BESIDE)
        f.go(30, L=WAIT_L).hold(25)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_red_pulled_and_put_back_scores_zero(self):
        f = right_stages(Film())
        left_pulls_red(f, R_START)
        f.go(30, L=WAIT_L).hold(25)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIsNotNone(t.red_out_step)

    def test_fuse_off_the_table_scores_zero(self):
        f = right_stages(Film())
        left_pulls_red(f)
        left_takes_spare(f)
        left_sets_spare(f)
        f.go(30, L=WAIT_L).hold(5)
        f.go(10, r=[L.TRAY_X, 0.70, L.TOP + 0.02]).go(10, r=[L.TRAY_X, 0.70, L.TOP - 0.30]).hold(25)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("fuse_off_table", t.breaches)
        f = Film().go(20, g=[0.25, -0.70, L.TOP + 0.02]).go(10, g=[0.25, -0.70, L.TOP - 0.5]).hold(5)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("fuse_off_table", t.breaches)

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        f.frames[-1] = dict(f.frames[-1])
        r, _ = play(f)
        t = Tracker(HZ)
        play(f, t)
        a = snap(WAIT_L, OPEN, PARK_R, OPEN)
        r = t.update(cyl(IN_TRAY, vel=(0.05, 0, 0)), cyl(SEATED), a)  # the red fuse still rolling
        self.assertEqual((r.score, r.success), (1.0, False))
        r = t.update(cyl(IN_TRAY), cyl(SEATED, spin=(0, 0, 1.0)), a)  # the green fuse spinning
        self.assertEqual((r.score, r.success), (1.0, False))
        r = t.update(cyl(IN_TRAY), cyl(SEATED), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        _, t = play(procedure())
        a = snap(WAIT_L, OPEN, PARK_R, OPEN)
        s1 = t.result(cyl(IN_TRAY), cyl(SEATED), a).score
        s2 = t.result(cyl(IN_TRAY), cyl(SEATED), a).score
        self.assertEqual((s1, s2), (1.0, 1.0))
        self.assertEqual(t.steps, len(procedure().frames))


# -- the one-hand rule -------------------------------------------------------------------------------------------
class Rule(unittest.TestCase):
    def test_second_gripper_inside_later_scores_zero(self):
        f = procedure(settle=5)
        f.go(30, R=[0.25, 0.09, L.TOP + 0.05]).go(30, R=PARK_R).hold(25)  # the right gripper dips into a corner
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(list(t.breaches), ["second_hand_in_panel"])
        self.assertGreater(t.steps_inside["right"], 0)

    def test_open_fingers_over_the_edge_count_as_inside(self):
        # After the job the open right gripper comes to 4 cm behind the panel's back edge: its axis (TCP and tip
        # point) stays outside the outline, but its open fingers (57 mm from the axis) reach over it.
        f = procedure(settle=5)
        near = np.array([ZONE_LO[0] - 0.04, L.PANEL_Y - 0.05, L.TOP + 0.08])
        f.go(30, R=near).go(30, R=PARK_R).hold(25)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(list(t.breaches), ["second_hand_in_panel"])
        f = procedure(settle=5)  # the same place with the fingers closed: 24 mm clear of the outline, no breach
        f.go(20, Ro=0.0).go(30, R=near).go(30, R=PARK_R).go(20, Ro=OPEN).hold(25)
        r, t = play(f)
        self.assertEqual((r.score, t.breach), (1.0, False))

    def test_right_inside_first_makes_it_the_panel_hand(self):
        f = Film().go(20, R=[0.25, 0.09, L.TOP + 0.05]).go(20, R=PARK_R)  # the right gripper goes in first
        right_stages(f)
        left_pulls_red(f)
        r, t = play(f)
        self.assertEqual(t.panel_hand, "right")
        self.assertIn("other_not_parked", t.breaches)  # the left arm was at its ready pose, not parked
        self.assertIn("second_hand_in_panel", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_both_in_on_the_same_step_scores_zero(self):
        f = Film(L=[0.30, 0.30, L.TOP + 0.05], R=[0.25, 0.0, L.TOP + 0.05])  # both just outside
        f.hold(3)
        self.assertIsNone(play(f)[1].panel_hand)
        f.go(1, L=[0.30, 0.20, L.TOP + 0.05], R=[0.25, 0.09, L.TOP + 0.05]).hold(5)  # both in at once
        r, t = play(f)
        self.assertEqual(t.panel_hand, "both")
        self.assertIn("second_hand_in_panel", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_left_in_while_the_right_is_not_parked(self):
        f = Film()
        left_pulls_red(f)  # the right arm still at its ready pose
        r, t = play(f)
        self.assertEqual(list(t.breaches), ["other_not_parked"])
        self.assertEqual(r.score, 0.0)
        self.assertLess(t.min_park_margin, 0.0)

    def test_hovering_over_the_square_is_not_parked(self):
        f = right_stages(Film(), park=PARK_R + [0, 0, 0.08])  # 13 cm up over its square
        left_pulls_red(f)
        r, t = play(f)
        self.assertIn("other_not_parked", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_parked_somewhere_else_is_not_parked(self):
        away = np.array([0.30, -0.30, L.TOP + 0.05])  # low and far from the panel, but not on its square
        f = right_stages(Film(), park=away)
        left_pulls_red(f)
        r, t = play(f)
        self.assertIn("other_not_parked", t.breaches)

    def test_right_leaves_its_square_while_the_left_is_inside(self):
        f = right_stages(Film())
        left_pulls_red(f)
        left_takes_spare(f)
        f.go(40, L=_xy(SEATED, CARRY), g=_xy(SEATED, CARRY - HELD))  # the left is inside again ...
        f.go(10, R=PARK_R + [0, 0, 0.07])  # ... and the right lifts out of its square
        f.go(10, R=PARK_R)
        left_sets_spare(f)
        f.go(30, L=WAIT_L).hold(25)
        r, t = play(f)
        self.assertEqual(list(t.breaches), ["other_not_parked"])
        self.assertEqual(r.score, 0.0)

    def test_right_may_move_while_the_left_is_outside(self):
        f = right_stages(Film())
        left_pulls_red(f)  # ends over the tray, outside the panel
        f.go(30, R=[0.25, -0.15, HOVER]).go(30, R=PARK_R)  # the right leaves its square and comes back
        left_takes_spare(f)
        left_sets_spare(f)
        f.go(30, L=WAIT_L).hold(25)
        r, t = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertFalse(t.breach)

    def test_left_high_over_the_panel_is_not_inside(self):
        high = np.array([L.PANEL_X, L.PANEL_Y, L.ZONE_TOP_Z + TIPS + 0.01])  # finger tips 1 cm above the live zone
        r, t = play(Film().go(30, L=high).hold(10))  # the right arm at its ready pose: no breach
        self.assertIsNone(t.panel_hand)
        self.assertFalse(t.breach)

    def test_a_breach_is_never_undone(self):
        f = Film()
        f.go(20, L=OVER_SOCKET).go(20, L=WAIT_L)  # in while the right is unparked, then out again
        f2 = right_stages(f)
        left_pulls_red(f2)
        left_takes_spare(f2)
        left_sets_spare(f2)
        f2.go(30, L=WAIT_L).hold(25)
        r, t = play(f2)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.breaches["other_not_parked"], t.first_entry_step)
        self.assertLess(t.first_entry_step, 21)  # flagged on the first excursion, kept through the rest


# -- exploits -----------------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_right_holding_the_spare_at_the_edge_scores_zero(self):
        # The right arm holds the spare just outside the panel's edge, ready for the left to take, while the left
        # works inside: helpful, and outside the panel, but not parked.
        f = Film()
        grip = G_START + [0, 0, HELD]
        f.go(30, R=grip + [0, 0, 0.074]).go(30, R=grip).go(30, Ro=GRIP)
        edge = np.array([L.PANEL_X - L.PANEL_HALF - 0.025, L.PANEL_Y - L.PANEL_HALF - 0.015, CARRY])
        f.go(40, R=edge, g=edge - [0, 0, HELD])
        left_pulls_red(f)
        f.go(30, L=_xy(edge, CARRY + 0.03))  # and takes it from the right gripper outside the panel ...
        r, t = play(f)
        self.assertEqual(t.panel_hand, "left")
        self.assertEqual(t.steps_inside["right"], 0)
        self.assertEqual(list(t.breaches), ["other_not_parked"])
        self.assertEqual(r.score, 0.0)

    def test_left_does_it_alone_with_the_right_idle_scores_zero(self):
        # DEMO_IDLE_ARM=right: the right arm stays at its ready pose, which is not its parking square.
        f = Film()
        left_pulls_red(f)
        f.go(30, L=WAIT_L).hold(25)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("other_not_parked", t.breaches)

    def test_left_alone_with_the_right_parked_is_capped_at_quarter(self):
        # Even with the right arm parked first, the left arm cannot reach the spare (straight-down grips): 0.25.
        f = Film().go(40, R=_xy(PARK_R, HOVER)).go(20, R=PARK_R)
        left_pulls_red(f)
        f.go(30, L=WAIT_L).hold(25)
        r, _ = play(f)
        self.assertEqual(r.score, 0.25)

    def test_right_alone_scores_zero(self):
        # DEMO_IDLE_ARM=left: the right arm stages the spare and parks; nobody touches the red fuse.
        r, t = play(right_stages(Film()).hold(25))
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(t.panel_hand)

    def test_sweeping_the_red_fuse_off_the_table_scores_zero(self):
        f = right_stages(Film())
        left_pulls_red(f, np.array([0.10, 0.62, L.TOP + HALF]))  # set on the table's left edge ...
        f.go(10, r=[0.10, 0.70, L.TOP - 0.2])  # ... and off it
        left_takes_spare(f)
        left_sets_spare(f)
        f.go(30, L=WAIT_L).hold(25)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("fuse_off_table", t.breaches)

    def test_green_seated_and_red_dropped_in_the_tray_from_high_up(self):
        # Dropping the red fuse into the tray from 10 cm counts if it lands in the tray (no rule against it).
        f = right_stages(Film())
        grip = R_START + [0, 0, HELD]
        f.go(40, L=grip + [0, 0, 0.07]).go(30, L=grip).go(30, Lo=GRIP)
        f.go(30, L=_xy(R_START, CARRY), r=_xy(R_START, CARRY - HELD))
        f.go(40, L=_xy(IN_TRAY, CARRY), r=_xy(IN_TRAY, CARRY - HELD))
        f.go(10, Lo=OPEN).go(4, r=IN_TRAY, rq=about("x", 90)).go(1, r=IN_TRAY + [0, 0, L.FUSE_RADIUS - HALF])
        left_takes_spare(f)
        left_sets_spare(f)
        f.go(30, L=WAIT_L).hold(25)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)


# -- the arms' state --------------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'grippers out' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(t.panel_hand)
        self.assertIsNone(r.details["grippers_out"])
        r, _ = play(procedure(leave=False), arms=False)  # even a gripper left inside cannot be seen
        self.assertEqual(r.score, 1.0)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        r = t.update(cyl(R_START), cyl(G_START), snap(left_tcp=OVER_SOCKET))  # right at its ready pose
        self.assertIn("other_not_parked", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(cyl(R_START), cyl(G_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(cyl(R_START), cyl(G_START), {"left": arm(OVER_SOCKET)})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)
        self.assertIsNone(t.panel_hand)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=PARK_R, right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, PARK_R - [0, 0, TIPS]))
        t = Tracker(HZ)
        r = t.update(cyl(R_START), cyl(G_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"] and r.details["right_parked"])
        self.assertFalse(r.details["left_parked"])


# -- robustness ---------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_either_end_down(self):
        r, _ = play(procedure(gq=about("y", 180)))
        self.assertEqual(r.score, 1.0)

    def test_a_few_mm_of_sag(self):
        # the spare seated against the opening's side, the parked TCP 3 cm off the square's centre and 3 cm higher
        r, t = play(procedure(green_at=SEATED + [0.004, -0.003, 0.001]))
        self.assertEqual(r.score, 1.0)
        f = right_stages(Film(), park=PARK_R + [0.03, 0.03, 0.03])
        left_pulls_red(f)
        left_takes_spare(f)
        left_sets_spare(f)
        f.go(30, L=WAIT_L).hold(25)
        r, t = play(f)
        self.assertEqual(r.score, 1.0)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = cyl(SEATED)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=SEATED)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.breaches), t.panel_hand, dict(t.steps_inside), t.min_park_margin, t.arms_unavailable)
            a = snap(WAIT_L, OPEN, PARK_R, OPEN)
            for red, green in ((cyl(IN_TRAY), bad), (bad, cyl(SEATED)), (bad, bad)):
                r = t.update(red, green, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(red, green, a).score, 0.0)
            after = (dict(t.breaches), t.panel_hand, dict(t.steps_inside), t.min_park_margin, t.arms_unavailable)
            self.assertEqual(after, before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [cyl((5.0, -3.0, -1.0)), cyl((0.3, 0.0, 0.0), about("x", 120)), cyl((-0.2, 0.6, L.TOP), about("y", 90)),
               cyl(SEATED, about("x", 89.9)), cyl((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        a = snap()
        for red, green in zip(odd + odd[::-1], odd[::-1] + odd):
            r = t.update(red, green, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
        r = t.update(cyl(R_START), cyl(G_START), None)
        self.assertEqual(r.score, 0.0)


# -- resets -------------------------------------------------------------------------------------------------------
class Reset(unittest.TestCase):
    def breached(self):
        f = Film()
        left_pulls_red(f)  # the right arm never parked
        _, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(cyl(R_START), cyl(G_START), snap())  # both fuses jump back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertIsNone(t.panel_hand)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(cyl(R_START + [0.004, -0.003, 0]), cyl(G_START + [-0.03, 0.02, 0]), snap())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur = t.prev
        f = Film(r=cur[0][0], g=G_START, L=WAIT_L)
        f.go(60, r=_xy(cur[0][0], CARRY)).go(60, r=_xy(R_START, CARRY)).go(60, r=R_START + [0, 0, 0.0005])
        f.go(2, r=R_START).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back in the socket, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(cyl(BESIDE), cyl(G_START + [0, 0.2, 0]), snap())  # knocked far, not back to the start
        self.assertTrue(t.breach)

    def test_a_dropped_fuse_is_not_a_teleport(self):
        # Free fall from 0.3 m into the socket: fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = R_START + [0, 0, 0.3]
        t.update(cyl(p), cyl(G_START), snap())
        t._flag("other_not_parked")
        z, v = p[2], 0.0
        while z > R_START[2]:
            v += 9.81 * DT
            z = max(R_START[2], z - v * DT)
            t.update(cyl((p[0], p[1], z), vel=(0, 0, -v if z > R_START[2] else 0)), cyl(G_START), snap())
        self.assertTrue(t.breach)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.breached()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertFalse(t.breach)
            self.assertIsNone(t.panel_hand)
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, red=R_START, green=G_START):
        return {"red_fuse": cyl(red), "green_fuse": cyl(green)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o.name for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(loose), ["green_fuse", "red_fuse"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(left_tcp=OVER_SOCKET)  # the left gripper in the panel, the right at its ready pose
        self.assertFalse(self.task.success(self.objects()))
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertFalse(tr.breach)
        rigstate.offline_reset()
        snap(WAIT_L, OPEN, PARK_R, OPEN)
        self.task.success(self.objects(IN_TRAY, SEATED))
        self.assertEqual(self.task.score(self.objects(IN_TRAY, SEATED)), 1.0)

    def test_trace_survives_a_bad_path(self):
        import os
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

    def test_demo_switches_and_timing(self):
        import os
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        # the left arm heads into the panel only after the right arm is down in its parking square
        right_parked = sum(w.duration for w in plan["right"])
        left_in_start = sum(w.duration for w in plan["left"][:2])
        self.assertIn("parking square", plan["right"][-1].label)
        self.assertIn("into the panel", plan["left"][2].label)
        self.assertGreaterEqual(left_in_start, right_parked + 0.5 - 1e-9)
        # the right arm's whole plan stays out of the panel's live zone
        for w in plan["right"]:
            self.assertFalse(in_panel(arm(w.pos)), w.label)
        self.assertTrue(parked(arm(plan["right"][-1].pos), "right"))
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
