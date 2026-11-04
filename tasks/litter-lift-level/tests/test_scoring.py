"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
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
    LITTER_HALF, Tracker, axis_tilt_deg, casualty_on, clearance, end_diff, end_heights,
    face_tilt_deg, flat_on_trolley, grippers_holding, heading_err_deg, in_the_air, let_go, lifted, low, lowest,
    on_trolley, resting_on_trolley, rotation, valid, zone_gap,
)

HZ = 20.0
DT = 1.0 / HZ
LH = L.LITTER_SIZE[2] / 2  # half the litter's 20 mm thickness
CH = L.CASUALTY_SIZE[2] / 2
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.004  # tips this far above the held litter's bottom
HELD = TIPS + GRIP_UP - LH  # the TCP is this far above a held litter's centre (7 mm)
RIDE = LH + CH  # the casualty's centre this far above the litter's centre when it lies on it (30 mm)
OPEN, GRIP = 1.0, L.LITTER_SIZE[0] / L.FULL_GAP  # rigstate openings: fully open, closed on the 40 mm litter

L_START = np.array([L.LITTER_X, L.LITTER_Y, L.TOP + LH])
C_START = L_START + [0, 0, RIDE]
L_ON_TROLLEY = np.array([L.TROLLEY_X, L.TROLLEY_Y, L.TROLLEY_TOP + LH])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
HY = L.HANDLE_Y
CARRY_Z = L.TOP + 0.080  # the TCP: the litter's underside 63 mm up
ENTRY_Z = L.TROLLEY_TOP + 0.002 + GRIP_UP + TIPS + 0.020
SET_Z = L.TROLLEY_TOP + 0.002 + GRIP_UP + TIPS
UP_Z = L.TOP + 0.10


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.LITTER_SIZE):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def litter(center, quat=Q0, **kw):
    return box(center, quat, size=L.LITTER_SIZE, **kw)


def casualty(center, quat=Q0, **kw):
    return box(center, quat, size=L.CASUALTY_SIZE, **kw)


def tilted_litter(low_end_z, high_end_z, x=L.LITTER_X, y=L.LITTER_Y):
    """A litter whose right (-y) end's bottom edge is at ``low_end_z`` and left (+y) end's at ``high_end_z``: the
    long axis tilted about x by the ends' difference over its 500 mm."""
    ang = math.degrees(math.asin((high_end_z - low_end_z) / L.LITTER_SIZE[1]))
    q = about("x", ang)  # a positive turn about x raises the +y end
    r = rotation(q)
    # the bottom edge of the -y end is at body (0, -half_y, -half_z)
    edge = r @ np.array([0.0, -LITTER_HALF[1], -LITTER_HALF[2]])
    centre = np.array([x, y, low_end_z]) - edge
    centre[0], centre[1] = x, y
    return litter(centre, q)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("l", "lq", "c", "cq", "L", "Lo", "R", "Ro")
VEC = ("l", "c", "L", "R")


class Film:
    """A trial as a list of frames: the litter's and casualty's centres and orientations, both TCPs and openings.
    ``go`` moves linearly to new values over n control steps; velocities come from the frame-to-frame motion. By
    default the casualty rides on the litter (``c`` follows ``l``) unless ``c`` is given."""

    def __init__(self, **start):
        self.cur = {"l": L_START, "lq": Q0, "c": C_START, "cq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, ride=True, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        if ride and "l" in to and "c" not in to:
            b["c"] = b["l"] + [0, 0, RIDE]
        if ride and "lq" in to and "cq" not in to:
            b["cq"] = b["lq"]
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in ("lq", "cq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def copy(self):
        out = Film()
        out.frames, out.cur = list(self.frames), dict(self.cur)
        return out


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        if prev is None:
            vl = vc = np.zeros(3)
        else:
            vl, vc = (fr["l"] - prev["l"]) / DT, (fr["c"] - prev["c"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(litter(fr["l"], fr["lq"], vel=vl), casualty(fr["c"], fr["cq"], vel=vc), a)
        prev = fr
    return r, t


# -- the intended trial, piece by piece (both arms on one clock, no sag) -------------------------------------------
def both_grip(f):
    """Both arms come over the handles, down beside them, and close on the litter (about 5 s)."""
    gl, gr = L_START + [0, HY, HELD], L_START + [0, -HY, HELD]
    f.go(30, L=gl + [0, 0, 0.10], R=gr + [0, 0, 0.10]).go(30, L=gl, R=gr).go(10).go(30, Lo=GRIP, Ro=GRIP).go(10)
    return f


def both_lift(f, dz=CARRY_Z - (L.TOP + LH + HELD)):
    """Both arms lift the litter straight up by ``dz`` (2 s)."""
    l = f.cur["l"] + [0, 0, dz]
    f.go(40, l=l, L=l + [0, HY, HELD], R=l + [0, -HY, HELD])
    return f


def both_carry(f, x=L.TROLLEY_X, y=L.TROLLEY_Y):
    """Both arms carry the litter level to (x, y) at the carry height (3 s)."""
    l = np.array([x, y, f.cur["l"][2]])
    f.go(60, l=l, L=l + [0, HY, HELD], R=l + [0, -HY, HELD])
    return f


def both_lower(f, rest_z=L.TROLLEY_TOP):
    """Both arms lower the litter onto the surface at ``rest_z``: to 2 cm above, settle, the last 2 cm slowly."""
    x, y = f.cur["l"][:2]
    entry = np.array([x, y, rest_z + 0.002 + LH + 0.020])
    setd = np.array([x, y, rest_z + 0.002 + LH])
    f.go(20, l=entry, L=entry + [0, HY, HELD], R=entry + [0, -HY, HELD]).go(10)
    f.go(30, l=setd, L=setd + [0, HY, HELD], R=setd + [0, -HY, HELD]).go(10)
    return f


def both_release(f, rest_z=L.TROLLEY_TOP, open_to=OPEN, up=True):
    """Both grippers open (the litter drops its last 2 mm onto the surface) and go up."""
    x, y = f.cur["l"][:2]
    down = np.array([x, y, rest_z + LH])
    f.go(6, Lo=0.6, Ro=0.6).go(2, l=down).go(20, Lo=open_to, Ro=open_to)
    if up:
        f.go(20, L=[x, HY, UP_Z], R=[x, -HY, UP_Z])
    return f


def procedure(at=(L.TROLLEY_X, L.TROLLEY_Y), open_to=OPEN, settle=70):
    """The intended trial: both arms grip, lift together, carry level over the trolley, lower together, let go, go
    up; all still for ``settle`` steps."""
    f = Film().hold(10)
    both_grip(f)
    both_lift(f)
    both_carry(f, *at)
    both_lower(f)
    both_release(f, open_to=open_to)
    return f.hold(settle)


# -- geometry ----------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_tilts_and_heading(self):
        self.assertAlmostEqual(axis_tilt_deg(about("x", 6)), 6.0, places=4)  # one end up
        self.assertAlmostEqual(axis_tilt_deg(about("y", 6)), 0.0, places=4)  # a roll is not an axis tilt
        self.assertAlmostEqual(face_tilt_deg(about("y", 6)), 6.0, places=4)  # but it is a face tilt
        self.assertAlmostEqual(face_tilt_deg(about("x", 6)), 6.0, places=4)
        self.assertAlmostEqual(face_tilt_deg(about("x", 180)), 0.0, places=4)  # upside down: the same board
        self.assertAlmostEqual(axis_tilt_deg(about("z", 30)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(heading_err_deg(about("z", 30)), 30.0, places=4)
        self.assertAlmostEqual(heading_err_deg(about("z", 180)), 0.0, places=4)  # end for end: along y still
        self.assertAlmostEqual(heading_err_deg(about("z", 90)), 90.0, places=4)
        self.assertAlmostEqual(heading_err_deg(about("x", 8)), 0.0, places=4)

    def test_end_heights(self):
        a, b = end_heights(litter(L_START))
        self.assertAlmostEqual(a, L.TOP, places=9)
        self.assertAlmostEqual(b, L.TOP, places=9)
        lit = tilted_litter(L.TOP, L.TOP + 0.050)  # right end on the table, left end 50 mm up
        left, right = end_heights(lit)
        self.assertAlmostEqual(right, L.TOP, places=6)
        self.assertAlmostEqual(left, L.TOP + 0.050, places=6)
        self.assertAlmostEqual(end_diff(lit), 0.050, places=6)
        self.assertAlmostEqual(axis_tilt_deg(lit.quat_xyzw), math.degrees(math.asin(0.1)), places=4)
        self.assertAlmostEqual(lowest(lit), L.TOP - 0.040 * math.sin(math.radians(0)), places=3)
        # rolled 5 degrees about its long axis: both ends' bottom edges stay level with each other
        rolled = litter(L_START + [0, 0, 0.05], about("y", 5))
        self.assertLess(end_diff(rolled), 1e-9)
        self.assertGreater(face_tilt_deg(rolled.quat_xyzw), 4.9)

    def test_surfaces_and_clearance(self):
        self.assertAlmostEqual(clearance(litter(L_START)), 0.0, places=9)
        self.assertAlmostEqual(clearance(litter(L_ON_TROLLEY)), 0.0, places=9)
        self.assertAlmostEqual(clearance(litter(L_ON_TROLLEY + [0, 0, 0.02])), 0.02, places=9)
        self.assertTrue(resting_on_trolley(litter(L_ON_TROLLEY)))
        self.assertTrue(resting_on_trolley(litter(L_ON_TROLLEY + [0.059, 0.17, 0.0025])))
        self.assertFalse(resting_on_trolley(litter(L_ON_TROLLEY + [0, 0, 0.004])))  # held 4 mm above it
        self.assertFalse(resting_on_trolley(litter(L_ON_TROLLEY + [0.061, 0, 0])))  # centre past its edge
        self.assertTrue(lifted(litter(L_START + [0, 0, 0.010])))
        self.assertFalse(lifted(litter(L_START + [0, 0, 0.009])))
        self.assertFalse(lifted(litter(L_ON_TROLLEY)))  # on the trolley is not lifted
        self.assertTrue(in_the_air(litter(L_ON_TROLLEY + [0, 0, 0.010])))
        self.assertFalse(in_the_air(litter(L_ON_TROLLEY + [0, 0, 0.009])))
        # low (dragging band): on the table; on the trolley; skimming the trolley with the centre still over the
        # table but the front edge over the trolley; not while carried 26 mm over the trolley
        self.assertTrue(low(litter(L_START)))
        self.assertTrue(low(litter(L_ON_TROLLEY)))
        self.assertTrue(low(litter([0.31, 0, L.TROLLEY_TOP + LH + 0.005])))
        self.assertFalse(low(litter([0.31, 0, L.TROLLEY_TOP + LH + 0.011])))
        self.assertFalse(low(litter(L_ON_TROLLEY + [0, 0, 0.026])))
        self.assertFalse(low(litter(L_START + [0, 0, 0.056])))

    def test_flat_on_trolley_tolerances(self):
        tight, loose = (L.CENTRE_X_FULL, L.CENTRE_Y_FULL), (L.CENTRE_LOOSE, L.CENTRE_LOOSE)
        self.assertTrue(flat_on_trolley(litter(L_ON_TROLLEY), *tight))
        self.assertTrue(flat_on_trolley(litter(L_ON_TROLLEY + [0.014, 0.019, 0]), *tight))
        self.assertFalse(flat_on_trolley(litter(L_ON_TROLLEY + [0.016, 0, 0]), *tight))
        self.assertFalse(flat_on_trolley(litter(L_ON_TROLLEY + [0, 0.021, 0]), *tight))
        self.assertTrue(flat_on_trolley(litter(L_ON_TROLLEY + [0.039, -0.039, 0]), *loose))
        self.assertFalse(flat_on_trolley(litter(L_ON_TROLLEY + [0.041, 0, 0]), *loose))
        self.assertFalse(flat_on_trolley(litter(L_ON_TROLLEY, about("y", 3.5)), *loose))  # rolled 3.5 deg
        self.assertFalse(flat_on_trolley(litter(L_ON_TROLLEY, about("z", 12)), *loose))  # turned 12 deg
        self.assertTrue(flat_on_trolley(litter(L_ON_TROLLEY, about("z", 8)), *tight))
        self.assertTrue(flat_on_trolley(litter(L_ON_TROLLEY, about("z", 180)), *tight))  # end for end
        self.assertTrue(flat_on_trolley(litter(L_ON_TROLLEY, about("x", 180)), *tight))  # upside down: a board
        self.assertTrue(on_trolley(litter(L_ON_TROLLEY, about("z", 12))))  # turned: still on it
        self.assertTrue(on_trolley(litter(L_ON_TROLLEY + [0, 0, 0.0015], about("y", 8))))
        self.assertFalse(on_trolley(litter(L_ON_TROLLEY + [0, 0, 0.005], about("y", 12))))

    def test_casualty_on(self):
        lit = litter(L_START)
        self.assertTrue(casualty_on(casualty(C_START), lit))
        self.assertTrue(casualty_on(casualty(C_START + [0.02, 0.02, 0.005]), lit))  # 28 mm off, 5 mm up
        self.assertFalse(casualty_on(casualty(C_START + [0.0, 0.031, 0]), lit))
        self.assertFalse(casualty_on(casualty(C_START + [0, 0, 0.007]), lit))  # lifted off it
        self.assertFalse(casualty_on(casualty(C_START, about("x", 11)), lit))
        self.assertTrue(casualty_on(casualty(C_START, about("x", 8)), lit))
        self.assertFalse(casualty_on(casualty(C_START), litter(L_START + [0, 0, 0.05])))  # litter lifted from under
        # the whole pair tilted 8 degrees together: still on
        q = about("x", 8)
        self.assertTrue(casualty_on(casualty(L_START + rotation(q) @ [0, 0, RIDE], q), litter(L_START, q)))

    def test_finger_zone_and_let_go(self):
        lit = litter(L_START)
        lo, hi = lit.low, lit.high
        held = snap(left_tcp=L_START + [0, HY, HELD], left_open=GRIP)["left"]
        self.assertLess(zone_gap(held, lo, hi), 0.0)
        self.assertFalse(let_go(held, lo, hi))
        open_there = snap(left_tcp=L_START + [0, HY, HELD], left_open=OPEN)["left"]
        self.assertTrue(let_go(open_there, lo, hi))  # open round it: let go
        away = snap(left_tcp=L_START + [0, HY, 0.10], left_open=GRIP)["left"]  # closed 10 cm above it
        self.assertGreater(zone_gap(away, lo, hi), L.LET_GO_GAP)
        self.assertTrue(let_go(away, lo, hi))
        a = snap(left_tcp=L_START + [0, HY, HELD], left_open=GRIP, right_tcp=L_START + [0, -HY, HELD],
                 right_open=GRIP)
        self.assertTrue(grippers_holding(a, lit))
        a = snap(left_tcp=L_START + [0, HY, HELD], left_open=GRIP)  # the right at its ready pose
        self.assertFalse(grippers_holding(a, lit))


# -- the tiers ---------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertFalse(t.carried_level)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.arms_unavailable, 0)
        self.assertTrue(t.carried_level)
        self.assertLess(t.max_end_diff, 1e-6)
        self.assertLess(t.max_tilt_up, 1e-6)
        self.assertLess(t.drag, L.DRAG_STEP)
        self.assertLess(t.setdown, L.SETDOWN_MAX)
        self.assertTrue(r.details["handled_evenly"])
        self.assertTrue(r.details["grippers_let_go"])

    def test_just_inside_the_centre_limits(self):
        r, _ = play(procedure(at=(L.TROLLEY_X + 0.014, L.TROLLEY_Y - 0.019)))
        self.assertEqual(r.score, 1.0)

    def test_off_centre_scores_075(self):
        r, _ = play(procedure(at=(L.TROLLEY_X + 0.025, L.TROLLEY_Y)))
        self.assertEqual((r.score, r.success), (0.75, False))
        r, _ = play(procedure(at=(L.TROLLEY_X, L.TROLLEY_Y + 0.035)))
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(at=(L.TROLLEY_X - 0.039, L.TROLLEY_Y + 0.039)))
        self.assertEqual(r.score, 0.75)

    def test_far_off_centre_scores_half(self):
        r, t = play(procedure(at=(L.TROLLEY_X + 0.045, L.TROLLEY_Y)))  # 45 mm off: 25 mm of it past the edge
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)
        r, _ = play(procedure(at=(L.TROLLEY_X, L.TROLLEY_Y + 0.10)))  # 10 cm along the trolley
        self.assertEqual(r.score, 0.5)

    def test_gripper_still_holding_scores_075(self):
        r, _ = play(procedure(open_to=GRIP))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(r.details["grippers_let_go"])
        r, _ = play(procedure(open_to=0.85))  # 80 mm apart, still round the litter: not let go
        self.assertEqual(r.score, 0.75)

    def test_a_gripper_closed_away_from_the_litter_has_let_go(self):
        f = procedure(settle=5)
        f.go(20, Lo=0.0, Ro=0.0).hold(70)  # both close again, 10 cm up: holding nothing
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertTrue(r.details["grippers_let_go"])

    def test_lifted_unevenly_scores_075(self):
        # the left end goes up 30 mm before the right follows; the rest done right
        f = Film().hold(10)
        both_grip(f)
        f.go(20, l=L_START + [0, 0, 0.015], lq=about("x", -math.degrees(math.asin(0.030 / 0.5))),
             L=L_START + [0, HY, HELD + 0.030], R=L_START + [0, -HY, HELD])
        f.go(20, l=L_START + [0, 0, 0.030], lq=Q0, L=L_START + [0, HY, HELD + 0.030],
             R=L_START + [0, -HY, HELD + 0.030])
        both_lift(f, CARRY_Z - (L.TOP + LH + HELD) - 0.030)
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, t = play(f.hold(70))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        self.assertGreater(t.max_end_diff, L.END_DIFF_FULL)
        self.assertLessEqual(t.max_end_diff, L.END_DIFF_MAX)

    def test_tilted_while_carried_scores_075(self):
        # carried with the long axis tilted 4 degrees for a moment: over 500 mm that is 35 mm between the ends,
        # past the 20 mm for 1.0 but short of the 40 mm breach (a 5 degree tilt is already 44 mm, so on this litter
        # the end rule always trips before the 5 degree tilt limit; the tilt record is checked on its own below)
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(10, lq=about("x", 4)).go(10, lq=Q0)
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, t = play(f.hold(70))
        self.assertEqual(r.score, 0.75)
        self.assertGreater(t.max_end_diff, L.END_DIFF_FULL)
        self.assertLess(t.max_tilt_up, L.TILT_FULL)
        self.assertFalse(t.breach)

    def test_tilt_record_alone_caps_at_075(self):
        t = Tracker(HZ)
        play(procedure(settle=0), t)
        t.max_tilt_up = 6.0  # as if the litter had tilted 6 degrees while up: the tilt record is its own limit
        #                      (on this 500 mm litter the ends would also have been 52 mm apart, a breach)
        f = Film(l=L_ON_TROLLEY, c=L_ON_TROLLEY + [0, 0, RIDE], L=[L.TROLLEY_X, HY, UP_Z],
                 R=[L.TROLLEY_X, -HY, UP_Z]).hold(5)
        r, _ = play(f, t)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["handled_evenly"])

    def test_dropped_onto_the_trolley_scores_075(self):
        # let go 15 mm above the trolley: it falls, lands flat and centred; the set-down speed shows the drop
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        both_carry(f)
        high = np.array([L.TROLLEY_X, L.TROLLEY_Y, L.TROLLEY_TOP + LH + 0.015])
        f.go(20, l=high, L=high + [0, HY, HELD], R=high + [0, -HY, HELD]).go(10)
        f.go(6, Lo=0.6, Ro=0.6).go(1, l=L_ON_TROLLEY).go(20, Lo=OPEN, Ro=OPEN)
        f.go(20, L=[L.TROLLEY_X, HY, UP_Z], R=[L.TROLLEY_X, -HY, UP_Z]).hold(70)
        r, t = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertGreater(t.setdown, L.SETDOWN_MAX)
        self.assertFalse(t.breach)

    def test_turned_on_the_trolley_scores_half(self):
        f = procedure(settle=5)
        f.go(10, lq=about("z", 15)).hold(60)  # turned 15 degrees across the trolley's axis
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_rolled_on_the_trolley_edge_scores_half(self):
        # one long edge on the trolley, rolled 5 degrees (the other edge 3.5 mm up): on it, not flat
        f = procedure(settle=5)
        f.go(10, l=L_ON_TROLLEY + [0, 0, 0.0017], lq=about("y", 5)).hold(60)
        r, _ = play(f)
        self.assertTrue(r.details["on_trolley"])
        self.assertFalse(r.details["flat_on_trolley_loose"])
        self.assertEqual(r.score, 0.5)

    def test_carried_level_then_set_back_on_the_table_scores_quarter(self):
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        both_carry(f, L.LITTER_X + 0.04, 0.0)  # moved 4 cm forward, still over the table
        both_lower(f, rest_z=L.TOP)
        both_release(f, rest_z=L.TOP)
        r, t = play(f.hold(70))
        self.assertEqual(r.score, 0.25)
        self.assertTrue(t.carried_level)
        self.assertFalse(t.breach)

    def test_still_held_in_the_air_scores_quarter(self):
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        both_carry(f)
        r, _ = play(f.hold(40))
        self.assertEqual((r.score, r.success), (0.25, False))
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        both_carry(f)
        both_lower(f)
        r, _ = play(f.hold(5))  # resting on the trolley 2 mm up, still gripped: 0.75 (on it, within 3 mm)
        self.assertEqual(r.score, 0.75)

    def test_lifted_but_never_level_at_30mm_scores_zero(self):
        # lifted only 20 mm (under the 30 mm 'carried' height) and put back
        f = Film().hold(10)
        both_grip(f)
        both_lift(f, 0.020)
        f.hold(20)
        both_lift(f, -0.020)
        both_release(f, rest_z=L.TOP)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.carried_level)
        self.assertFalse(t.breach)

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(3, l=L_ON_TROLLEY + [0.002, 0, 0]).go(3, l=L_ON_TROLLEY))  # still creeping
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=[L.TROLLEY_X, HY, UP_Z], right_tcp=[L.TROLLEY_X, -HY, UP_Z])
        r = tr.update(litter(L_ON_TROLLEY, vel=(0.02, 0, 0)), casualty(L_ON_TROLLEY + [0, 0, RIDE]), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(litter(L_ON_TROLLEY), casualty(L_ON_TROLLEY + [0, 0, RIDE], spin=(0, 0, 1.0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(litter(L_ON_TROLLEY), casualty(L_ON_TROLLEY + [0, 0, RIDE]), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        last = (litter(L_ON_TROLLEY), casualty(L_ON_TROLLEY + [0, 0, RIDE]),
                snap(left_tcp=[L.TROLLEY_X, HY, UP_Z], right_tcp=[L.TROLLEY_X, -HY, UP_Z]))
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))
        json.dumps(r.details, allow_nan=False)


# -- the rules, each broken once --------------------------------------------------------------------------------
class Breaches(unittest.TestCase):
    """Each rule broken once scores 0 for the rest of the trial, whatever follows."""

    def assert_zero(self, f, key):
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIn(key, t.breaches)
        return t

    def test_dragged_along_the_table(self):
        # slid 3 cm along the table toward the trolley while gripped, then done right
        f = Film().hold(10)
        both_grip(f)
        to = L_START + [0.03, 0, 0]
        f.go(30, l=to, L=to + [0, HY, HELD], R=to + [0, -HY, HELD])
        both_lift(f)
        both_carry(f)
        both_lower(f)
        both_release(f)
        t = self.assert_zero(f.hold(70), "dragged")
        self.assertGreater(t.drag, L.DRAG_MAX)

    def test_a_small_shift_is_not_dragging(self):
        # the litter shifts 8 mm while low (the arms' sag pulls it as they take the load), then is lifted
        f = Film().hold(10)
        both_grip(f)
        to = L_START + [-0.008, 0, 0]
        f.go(10, l=to, L=to + [0, HY, HELD], R=to + [0, -HY, HELD])
        both_lift(f)
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, t = play(f.hold(70))
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(t.drag, 0.008, places=3)

    def test_contact_jitter_does_not_add_up(self):
        t = Tracker(HZ)
        rng = np.random.default_rng(1)
        a = snap()
        for _ in range(600):  # 30 s of 0.2 mm shivers on the table
            j = rng.uniform(-0.0002, 0.0002, 2)
            t.update(litter(L_START + [j[0], j[1], 0]), casualty(C_START + [j[0], j[1], 0]), a)
        self.assertLess(t.drag, 0.001)
        self.assertFalse(t.breach)

    def test_skimmed_onto_the_trolley(self):
        # lifted only 8 mm and slid forward onto the trolley: dragging, whichever surface it is over
        f = Film().hold(10)
        both_grip(f)
        both_lift(f, 0.008)
        up8 = f.cur["l"]
        over = np.array([0.30, 0, L.TROLLEY_TOP + LH + 0.005])
        f.go(30, l=over, L=over + [0, HY, HELD], R=over + [0, -HY, HELD])
        f.go(30, l=L_ON_TROLLEY + [0, 0, 0.005], L=L_ON_TROLLEY + [0, HY, HELD + 0.005],
             R=L_ON_TROLLEY + [0, -HY, HELD + 0.005])
        both_release(f)
        t = self.assert_zero(f.hold(70), "dragged")
        self.assertGreater(float(np.linalg.norm(over[:2] - up8[:2])), L.DRAG_MAX)

    def test_dragged_on_the_trolley_after_setting_down(self):
        f = procedure(settle=5)
        f.go(20, l=L_ON_TROLLEY + [0, 0.02, 0]).hold(60)  # shoved 2 cm along the trolley afterwards
        self.assert_zero(f, "dragged")

    def test_lifted_at_one_end(self):
        # the left end lifted 50 mm while the right stays on the table; then everything done right
        f = Film().hold(10)
        both_grip(f)
        ang = math.degrees(math.asin(0.050 / 0.5))
        f.go(20, l=L_START + [0, 0, 0.025], lq=about("x", -ang), L=L_START + [0, HY, HELD + 0.050],
             R=L_START + [0, -HY, HELD])
        f.go(20, l=L_START + [0, 0, 0.050], lq=Q0, L=L_START + [0, HY, HELD + 0.050],
             R=L_START + [0, -HY, HELD + 0.050])
        both_lift(f, CARRY_Z - (L.TOP + LH + HELD) - 0.050)
        both_carry(f)
        both_lower(f)
        both_release(f)
        t = self.assert_zero(f.hold(70), "lifted_at_one_end")
        self.assertGreater(t.max_end_diff, L.END_DIFF_MAX)
        self.assertNotIn("litter_tipped", t.breaches)  # 5.7 degrees: under the 10 degree tip

    def test_lowered_at_one_end(self):
        # carried level, but the right end is set on the trolley 45 mm before the left
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        both_carry(f)
        ang = math.degrees(math.asin(0.045 / 0.5))
        f.go(30, l=L_ON_TROLLEY + [0, 0, 0.0225], lq=about("x", -ang), L=L_ON_TROLLEY + [0, HY, HELD + 0.045],
             R=L_ON_TROLLEY + [0, -HY, HELD])
        f.go(20, l=L_ON_TROLLEY, lq=Q0, L=L_ON_TROLLEY + [0, HY, HELD], R=L_ON_TROLLEY + [0, -HY, HELD])
        both_release(f)
        self.assert_zero(f.hold(70), "lifted_at_one_end")

    def test_litter_tipped(self):
        # the ends within 40 mm is impossible past 4.6 degrees over 500 mm, so a tip over 10 degrees is also an
        # end-difference breach; the tip rule stands on its own for a turned litter, where the end rule also holds.
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(10, lq=about("x", 12)).go(10, lq=Q0)
        both_carry(f)
        both_lower(f)
        both_release(f)
        t = self.assert_zero(f.hold(70), "litter_tipped")
        self.assertGreater(t.max_tilt, L.TILT_MAX)

    def test_casualty_slides_off(self):
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(20, c=f.cur["c"] + [0, 0.04, 0], ride=False)  # slid 4 cm along the litter
        f.go(20, c=f.cur["c"] - [0, 0.04, 0], ride=False)
        both_carry(f)
        both_lower(f)
        both_release(f)
        self.assert_zero(f.hold(70), "casualty_off")

    def test_a_one_step_hop_of_the_casualty_is_not_leaving(self):
        # a contact transient: the casualty 7 mm up off the litter for one step, back the next
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(1, c=f.cur["c"] + [0, 0, 0.007], ride=False).go(1, c=f.cur["c"] - [0, 0, 0.007], ride=False)
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, t = play(f.hold(70))
        self.assertEqual(r.score, 1.0)
        self.assertNotIn("casualty_off", t.breaches)
        # off for 4 steps: not yet; off for 5: it left
        for n, want in ((4, False), (5, True)):
            f = Film().hold(10)
            both_grip(f)
            both_lift(f)
            f.go(1, c=f.cur["c"] + [0, 0, 0.007], ride=False).hold(n - 1).go(1, c=f.cur["c"] - [0, 0, 0.007], ride=False)
            _, t = play(f.hold(5))
            self.assertEqual("casualty_off" in t.breaches, want, n)

    def test_rolled_about_its_long_axis_while_carried(self):
        # a 12 degree roll about the long axis keeps the ends level, but the casualty's tilt (12 degrees, with it)
        # takes it off the litter: 0 by the casualty rule, not by the litter's tilt
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(10, lq=about("y", 12)).go(10, lq=Q0)
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, t = play(f.hold(70))
        self.assertEqual(r.score, 0.0)
        self.assertIn("casualty_off", t.breaches)
        self.assertNotIn("litter_tipped", t.breaches)
        self.assertLess(t.max_end_diff, 1e-6)

    def test_casualty_tipped(self):
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(10, cq=about("y", 20), ride=False).go(10, cq=Q0, ride=False)
        both_carry(f)
        both_lower(f)
        both_release(f)
        t = self.assert_zero(f.hold(70), "casualty_tipped")
        self.assertIn("casualty_off", t.breaches)

    def test_casualty_lifted_off_the_litter(self):
        # the casualty picked up off the litter (say by one arm) and put back: it left the litter
        f = Film().hold(10).go(20, c=C_START + [0, 0, 0.05], ride=False).go(20, c=C_START, ride=False)
        f2 = Film()
        f2.frames = f.frames + procedure().frames[1:]
        self.assert_zero(f2, "casualty_off")

    def test_one_gripper_while_in_the_air(self):
        # both grip and lift; the right lets go in the air and comes back: the left held it alone for a moment
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(30, Ro=OPEN).go(30, Ro=GRIP)
        both_carry(f)
        both_lower(f)
        both_release(f)
        t = self.assert_zero(f.hold(70), "one_gripper")
        self.assertEqual(t.breaches["one_gripper"], min(t.breaches.values()))

    def test_one_gripper_skimming_the_trolley_edge(self):
        # held by one gripper 15 mm up over the table with its front over the trolley's deck, 5 mm above it:
        # 'low' by the trolley, so not in the air, and the drag rule (not the gripper rule) is what catches a slide
        self.assertFalse(in_the_air(litter([0.31, 0, L.TROLLEY_TOP + LH + 0.005])))
        self.assertTrue(in_the_air(litter([0.31, 0, L.TROLLEY_TOP + LH + 0.011])))
        self.assertTrue(in_the_air(litter([0.25, 0, L.TOP + LH + 0.011])))

    def test_both_grippers_must_hold_only_in_the_air(self):
        # on the table or on the trolley the grippers may do as they like: the rule is for the carry
        f = Film().hold(10)
        both_grip(f)
        f.go(20, Ro=OPEN).go(20, Ro=GRIP)  # the right opens and closes again on the table
        both_lift(f)
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, t = play(f.hold(70))
        self.assertEqual(r.score, 1.0)
        self.assertNotIn("one_gripper", t.breaches)

    def test_litter_off_the_table(self):
        f = Film().hold(5).go(30, l=[0.30, -0.80, L.TOP - 0.40]).hold(20)  # swept off the table's edge
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["off_table"])
        f = Film().hold(5).go(30, c=[0.30, 0.80, L.TOP - 0.40], ride=False).hold(20)  # the casualty off
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["off_table"])

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5)
        both_grip(f)
        to = L_START + [0.03, 0, 0]
        f.go(30, l=to, L=to + [0, HY, HELD], R=to + [0, -HY, HELD])  # dragged 3 cm
        f.go(30, l=L_START, L=L_START + [0, HY, HELD], R=L_START + [0, -HY, HELD])  # and back
        f.go(20, Lo=OPEN, Ro=OPEN).go(20, L=READY_L, R=READY_R)
        f2 = Film()
        f2.frames = f.frames + procedure().frames[1:]
        r, t = play(f2)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(r.details["first_breach_step"], t.breaches["dragged"])


# -- unintended strategies ---------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_doing_nothing_scores_zero(self):
        r, _ = play(Film().hold(400))
        self.assertEqual(r.score, 0.0)

    def test_one_arm_lifts_its_end_and_the_litter_pivots(self):
        # the left arm alone: the handle goes up to the carry height, the litter pivots in the gripper (the far
        # end stays on the table), then the arm carries its end forward and sets it on the trolley
        f = Film().hold(10)
        gl = L_START + [0, HY, HELD]
        f.go(30, L=gl + [0, 0, 0.10]).go(30, L=gl).go(10).go(30, Lo=GRIP).go(10)
        up = CARRY_Z - gl[2]
        ang = math.degrees(math.asin(up / 0.5))
        tip = tilted_litter(L.TOP, L.TOP + up)
        f.go(40, l=tip.pos, lq=tip.quat_xyzw, L=gl + [0, 0, up], c=tip.pos + rotation(tip.quat_xyzw) @ [0, 0, RIDE],
             cq=tip.quat_xyzw, ride=False)
        r, t = play(f.hold(40))
        self.assertEqual(r.score, 0.0)
        self.assertIn("lifted_at_one_end", t.breaches)
        self.assertGreater(t.max_end_diff, L.END_DIFF_MAX)
        self.assertGreater(ang, L.TILT_FULL)

    def test_one_arm_carries_the_whole_litter_level(self):
        # were the grasp to hold the litter rigidly, one arm could carry it level and set it centred on the
        # trolley (no sag): every end-state condition met, but it was in the air with one gripper only
        f = Film().hold(10)
        gl = L_START + [0, HY, HELD]
        f.go(30, L=gl + [0, 0, 0.10]).go(30, L=gl).go(10).go(30, Lo=GRIP).go(10)
        l = L_START + [0, 0, CARRY_Z - gl[2]]
        f.go(40, l=l, L=l + [0, HY, HELD])
        l = np.array([L.TROLLEY_X, 0, l[2]])
        f.go(60, l=l, L=l + [0, HY, HELD])
        entry = np.array([L.TROLLEY_X, 0, L.TROLLEY_TOP + 0.002 + LH + 0.020])
        setd = np.array([L.TROLLEY_X, 0, L.TROLLEY_TOP + 0.002 + LH])
        f.go(20, l=entry, L=entry + [0, HY, HELD]).go(10).go(30, l=setd, L=setd + [0, HY, HELD]).go(10)
        f.go(6, Lo=0.6).go(2, l=L_ON_TROLLEY).go(20, Lo=OPEN).go(20, L=[L.TROLLEY_X, HY, UP_Z]).hold(70)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(sorted(t.breaches), ["one_gripper"])
        self.assertTrue(r.details["flat_on_trolley"])  # the end state alone would have been 1.0
        r, _ = play(f, arms=False)  # the arms unreadable: the one-gripper rule cannot be applied
        self.assertEqual(r.score, 1.0)

    def test_pushing_the_litter_onto_the_trolley(self):
        # shoved along the table (casualty riding) up against the trolley's side, then up over its edge
        f = Film().hold(10)
        f.go(40, l=L_START + [0.06, 0, 0])
        f.go(20, l=[0.30, 0, L.TROLLEY_TOP + LH], lq=about("y", -8))
        f.go(30, l=L_ON_TROLLEY, lq=Q0)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dragged", t.breaches)

    def test_dragging_one_end_round_while_the_other_is_lifted(self):
        f = Film().hold(10)
        both_grip(f)
        ang = math.degrees(math.asin(0.030 / 0.5))
        tip = tilted_litter(L.TOP, L.TOP + 0.030)
        f.go(20, l=tip.pos, lq=tip.quat_xyzw, L=L_START + [0, HY, HELD + 0.030], R=L_START + [0, -HY, HELD])
        to = tip.pos + [0.05, 0, 0]
        f.go(30, l=to, L=to + [0, HY, HELD + 0.030], R=to + [0, -HY, HELD])  # the low end dragged 5 cm
        r, t = play(f.hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertIn("dragged", t.breaches)
        self.assertLess(ang, L.TILT_MAX)

    def test_tossing_the_litter_onto_the_trolley(self):
        # lifted level and flung: it lands on the trolley flat and centred, the casualty beside it
        f = Film().hold(10)
        both_grip(f)
        both_lift(f, 0.040)
        f.go(10, Lo=OPEN, Ro=OPEN)
        f.go(4, l=L_ON_TROLLEY, c=[L.TROLLEY_X + 0.05, 0, L.TOP + CH], ride=False)
        f.go(20, L=READY_L, R=READY_R)
        r, t = play(f.hold(60))
        self.assertEqual(r.score, 0.0)
        self.assertIn("one_gripper", t.breaches)
        self.assertIn("casualty_off", t.breaches)

    def test_carrying_the_casualty_separately(self):
        # the casualty lifted off and set on the trolley by hand, the empty litter dragged under it: 0
        f = Film().hold(10)
        f.go(30, c=C_START + [0, 0, 0.08], ride=False).go(40, c=[L.TROLLEY_X, 0, L.TROLLEY_TOP + 0.08 + CH], ride=False)
        f.go(30, c=[L.TROLLEY_X, 0, L.TROLLEY_TOP + CH], ride=False)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("casualty_off", t.breaches)

    def test_setting_the_litter_on_the_trolley_upside_down_is_fine(self):
        # a plain board: turned over on the way (both ends together) it is still flat on the trolley
        f = Film().hold(10)
        both_grip(f)
        both_lift(f)
        f.go(10, lq=about("z", 180))
        both_carry(f)
        both_lower(f)
        both_release(f)
        r, _ = play(f.hold(70))
        self.assertEqual(r.score, 1.0)


# -- the arms -----------------------------------------------------------------------------------------------------
class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'both let go' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        self.assertIsNone(r.details["grippers_holding"])
        r, _ = play(procedure(open_to=GRIP), arms=False)  # even a closed gripper cannot be seen
        self.assertEqual(r.score, 1.0)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        f = Film().hold(5)
        both_grip(f)
        both_lift(f)
        play(f, t, arms=False)
        self.assertEqual(t.breaches, {})
        up = f.cur["l"]
        r = t.update(litter(up), casualty(up + [0, 0, RIDE]), snap(left_tcp=up + [0, HY, HELD], left_open=GRIP))
        self.assertIn("one_gripper", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, len(f.frames))

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(litter(L_START), casualty(C_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(litter(L_START), casualty(C_START), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(litter(L_START), casualty(C_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


# -- odd states --------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = litter(L_ON_TROLLEY)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=L_ON_TROLLEY)
        frames = procedure().frames
        good_l, good_c = litter(L_ON_TROLLEY), casualty(L_ON_TROLLEY + [0, 0, RIDE])
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-60]
            _, t = play(f)
            before = (dict(t.breaches), t.max_end_diff, t.drag, t.setdown, t.carried_level, t.arms_unavailable)
            a = snap(left_tcp=L_ON_TROLLEY + [0, HY, 0.1], right_tcp=L_ON_TROLLEY + [0, -HY, 0.1])
            for lit, cas in ((good_l, bad), (bad, good_c), (bad, bad)):
                r = t.update(lit, cas, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(t.result(lit, cas, a).score, 0.0)
            self.assertEqual((dict(t.breaches), t.max_end_diff, t.drag, t.setdown, t.carried_level,
                              t.arms_unavailable), before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-60:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [litter((5.0, -3.0, -1.0)), litter((0.3, 0.0, 0.0), about("x", 120)),
               litter((-0.2, 0.6, L.TOP), about("y", 90)), litter(L_ON_TROLLEY, about("x", 89.9)),
               litter((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0)), litter(L_START, about("z", 45))]
        t = Tracker(HZ)
        a = snap()
        for lit, cas in zip(odd + odd[::-1], odd[::-1] + odd):
            r = t.update(lit, casualty(cas.pos, cas.quat_xyzw), a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
            json.dumps(r.details, allow_nan=False)
        r = t.update(litter(L_START), casualty(C_START), None)
        self.assertIn(r.score, (0.0, 0.25))


# -- resets --------------------------------------------------------------------------------------------------------
class Reset(unittest.TestCase):
    def breached(self):
        """Dragged 3 cm (a breach), then lifted and carried over the trolley, still held there."""
        f = Film().hold(5)
        both_grip(f)
        to = L_START + [0.03, 0, 0]
        f.go(30, l=to, L=to + [0, HY, HELD], R=to + [0, -HY, HELD])
        both_lift(f)
        both_carry(f)
        _, t = play(f.hold(5))
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(litter(L_START), casualty(C_START), snap())  # everything jumps back in one step: new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(litter(L_START + [0.02, -0.02, 0]), casualty(C_START + [0.02, -0.02, 0]), snap())
        self.assertFalse(t.breach)

    def test_teleport_after_the_demo_resets(self):
        _, t = play(procedure())
        self.assertTrue(t.carried_level)
        t.update(litter(L_START), casualty(C_START), snap())
        self.assertEqual(t.steps, 1)
        self.assertFalse(t.carried_level)
        self.assertEqual(t.drag, 0.0)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur = t.prev[0][0]
        f = Film(l=cur, c=cur + [0, 0, RIDE], L=cur + [0, HY, HELD], Lo=GRIP, R=cur + [0, -HY, HELD], Ro=GRIP)
        back = np.array([L.LITTER_X, L.LITTER_Y, cur[2]])
        f.go(60, l=back, L=back + [0, HY, HELD], R=back + [0, -HY, HELD])
        f.go(60, l=L_START + [0, 0, 0.0005], L=L_START + [0, HY, HELD], R=L_START + [0, -HY, HELD])
        f.go(2, l=L_START).go(20, Lo=OPEN, Ro=OPEN).go(20, L=READY_L, R=READY_R).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(litter(L_START + [0, 0.2, 0]), casualty(C_START + [0, 0.2, 0]), snap())
        self.assertTrue(t.breach)

    def test_a_dropped_litter_is_not_a_teleport(self):
        # free fall from 0.3 m back onto its mark: fast, but its velocity explains it
        t = Tracker(HZ)
        play(Film().hold(5), t)
        t._flag("dragged")
        p = L_START + [0, 0, 0.3]
        t.update(litter(p), casualty(p + [0, 0, RIDE]), snap())
        z, v = p[2], 0.0
        while z > L_START[2]:
            v += 9.81 * DT
            z = max(L_START[2], z - v * DT)
            vz = -v if z > L_START[2] else 0
            t.update(litter((p[0], p[1], z), vel=(0, 0, vz)), casualty((p[0], p[1], z + RIDE), vel=(0, 0, vz)),
                     snap())
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
            self.assertEqual(t.steps, 0)
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


# -- task.py through the offline stub ---------------------------------------------------------------------------
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

    def objects(self, l=L_START, c=None):
        c = l + [0, 0, RIDE] if c is None else c
        return {"litter": litter(l), "casualty": casualty(c), "trolley": box(
            (L.TROLLEY_X, L.TROLLEY_Y, L.TOP + L.TROLLEY_SIZE[2] / 2), size=L.TROLLEY_SIZE)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(set(names), {"litter", "casualty", "trolley"})
        by = {o.name: o for o in self.task.OBJECTS}
        self.assertIsNone(by["trolley"].mass)  # fixed
        self.assertEqual(by["litter"].mass, L.LITTER_MASS)
        self.assertEqual(by["casualty"].mass, L.CASUALTY_MASS)
        self.assertAlmostEqual(by["casualty"].pos[2] - by["casualty"].size[2] / 2,
                               by["litter"].pos[2] + by["litter"].size[2] / 2)  # it lies on the litter
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 3.0)
        self.assertIn("litter", self.task.INSTRUCTION)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        snap(left_tcp=L_START + [0, 0.05, HY], left_open=GRIP)
        self.assertFalse(self.task.success(self.objects(L_START + [0, 0, 0.05])))  # one gripper, in the air
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(self.objects(L_START + [0, 0, 0.05])), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertFalse(tr.breach)

    def test_hooks_score_the_end_state(self):
        rigstate.offline_reset()
        snap(left_tcp=[L.TROLLEY_X, HY, UP_Z], right_tcp=[L.TROLLEY_X, -HY, UP_Z])
        self.task._TRACKER.carried_level = True
        self.assertEqual(self.task.score(self.objects(L_ON_TROLLEY)), 1.0)
        self.assertTrue(self.task.success(self.objects(L_ON_TROLLEY)))
        rigstate.offline_reset()

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

    def test_trace_writes_json_lines(self):
        import tempfile
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                rigstate.offline_reset()
                snap()
                self.task.success(self.objects())
                rows = [json.loads(line) for line in path.read_text().splitlines()]
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old
        self.assertEqual(len(rows), 1)
        self.assertIn("end_diff_mm", rows[0])
        self.assertIn("left_tcp", rows[0])

    def test_demo_switches_and_timing(self):
        for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
            os.environ.pop(k, None)
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        self.assertEqual(len(left), len(right))
        # both bearers on one clock: the same durations, finger targets, x and z, mirrored y and yaw
        for a, b in zip(left, right):
            self.assertEqual(a.duration, b.duration)
            self.assertEqual(a.finger, b.finger)
            self.assertAlmostEqual(a.pos[0], b.pos[0])
            self.assertAlmostEqual(a.pos[2], b.pos[2])
            self.assertAlmostEqual(a.pos[1], -b.pos[1])
            self.assertAlmostEqual(a.yaw, -b.yaw)
        total = sum(w.duration for w in left)
        self.assertAlmostEqual(total, 17.5)
        self.assertLess((total + 3.0) * HZ, L.RUN_LIMIT_STEPS)
        # the release (1.5 s) plus the rise (1.0 s) is shorter than the success hold
        self.assertIn("let go", left[-2].label)
        self.assertLess(left[-2].duration + left[-1].duration, self.task.SUCCESS_HOLD_S)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(set(self.task.demo()), {"left"})
        finally:
            os.environ.pop("DEMO_IDLE_ARM", None)


if __name__ == "__main__":
    unittest.main()
