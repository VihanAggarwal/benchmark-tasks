"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic part states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
    Tracker, cell_gap, centre_in_cell, finger_zone, grippers_let_go, in_cell, let_go, pad_pressed, pad_released,
    part_in_tray, part_lifted, rotation, tilt_deg, valid, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
H = L.PART_SIZE[2] / 2  # half the part's 30 mm
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.004  # tips this far above a held part's bottom
HELD = TIPS + GRIP_UP - H  # a held part's centre is this far below the TCP (2 mm)
OPEN = 1.0  # rigstate opening: fully open
GRIP = (L.PART_SIZE[1] - 0.008) / L.FULL_GAP  # closed on the 30 mm part (22 mm gap): 0.234
CLOSED = 0.008 / L.FULL_GAP  # the narrowest commanded gap, 16 mm: 0.085

P_START = np.array([L.PART_X, L.PART_Y, L.TOP + H])
IN_TRAY = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + H])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
PAD_HOVER = np.array([L.PAD_X, L.PAD_Y, L.TOP + 0.10])
PRESS = np.array([L.PAD_X, L.PAD_Y, L.PAD_TOP_Z + 0.002 + TIPS])  # tips 2 mm above the pad's top
WAIT_R = np.array([L.PART_X, -0.17, L.TOP + 0.12])
HOVER_R = L.TOP + 0.12
PART_GRIP = P_START + [0, 0, HELD]  # the right TCP closing on the part where it lies
TRAY_GRIP = IN_TRAY + [0, 0, HELD + 0.002]  # ... and letting it go 2 mm above the tray's floor


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.PART_SIZE):
    """An ObjectState-like box: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(size, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def arm(tcp, opening=OPEN):
    tcp = np.array(tcp, float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0.0, 0.0, TIPS]), opening=float(opening))


KEYS = ("p", "pq", "L", "Lo", "R", "Ro")


class Film:
    """A trial as a list of frames: the part's centre and orientation, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; the part's velocity comes from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"p": P_START, "pq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in ("p", "L", "R") else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in ("p", "L", "R") else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k == "pq":
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
        v = np.zeros(3) if prev is None else (fr["p"] - prev["p"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(box(fr["p"], fr["pq"], v), a)
        prev = fr
    return r, t


# -- the intended procedure, in pieces -------------------------------------------------------------------------
def left_presses(f):
    """The left arm closes its fingers over the pad and presses it (about 5 s)."""
    f.go(30, L=PAD_HOVER).go(30, Lo=CLOSED).go(20, L=PRESS + [0, 0, 0.02]).go(30, L=PRESS)
    return f


def left_releases(f):
    f.go(20, L=PAD_HOVER)
    return f


def right_recovers(f, to=IN_TRAY, open_to=OPEN, drop=True, up=True):
    """The right arm goes in over the part, grips it, lifts it straight up, carries it to ``to`` (the tray by
    default), sets it down and opens to ``open_to``, then goes up (about 15 s)."""
    f.go(30, R=WAIT_R).go(30, R=[*P_START[:2], HOVER_R]).go(20, R=PART_GRIP + [0, 0, 0.03]).go(30, R=PART_GRIP)
    f.go(30, Ro=GRIP)
    f.go(30, R=[*P_START[:2], HOVER_R], p=[*P_START[:2], HOVER_R - HELD])
    f.go(40, R=[*to[:2], HOVER_R], p=[*to[:2], HOVER_R - HELD])
    set_tcp = to + [0, 0, HELD + 0.002]
    f.go(20, R=set_tcp + [0, 0, 0.02], p=set_tcp + [0, 0, 0.02 - HELD]).go(30, R=set_tcp, p=set_tcp - [0, 0, HELD])
    if drop:
        f.go(8, Ro=0.4).go(2, p=to).go(22, Ro=open_to)
    else:
        f.go(30, Ro=open_to)
    if up:
        f.go(20, R=[*to[:2], HOVER_R])
    return f


def procedure(to=IN_TRAY, open_to=OPEN, release=True, settle=30, drop=True, retreat=False):
    """The intended trial: the left presses the pad, the right recovers the part to ``to``, the left releases the
    pad (if ``release``), everything still for ``settle`` steps. ``retreat``: the right goes back to its waiting
    place outside the cell before the pad is released (needed when ``to`` is within reach of the cell)."""
    f = Film().hold(10)
    left_presses(f)
    f.hold(20)
    right_recovers(f, to, open_to, drop)
    if retreat:
        f.go(20, R=WAIT_R)
    if release:
        left_releases(f)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_cube(self):
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 0.0, places=4)  # on another face: the same cube
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("x", 80)), 10.0, places=4)

    def test_finger_zone_covers_the_pads(self):
        a = arm((0.2, 0.2, 0.9), OPEN)
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)  # 56.6 mm
        pts, r = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], 0.9 - TIPS)  # from the tips ...
        self.assertAlmostEqual(pts[-1][2], 0.9 - TIPS + L.FINGER_LEN)  # ... 38 mm back up the gripper
        closed = arm((0.2, 0.2, 0.9), 0.0)
        self.assertAlmostEqual(zone_radius(closed), math.hypot(0.008, 0.0135), places=6)

    def test_in_cell(self):
        # open fingers at the part: inside; at the waiting place 8 cm to the right of the footprint: outside
        self.assertTrue(in_cell(arm((L.PART_X, L.PART_Y, HOVER_R))))
        self.assertFalse(in_cell(arm(WAIT_R)))
        self.assertAlmostEqual(cell_gap(arm(WAIT_R)), 0.17 - 0.09 - zone_radius(arm(WAIT_R)), places=6)
        # just over the front stripe: the open zone reaches 57 mm past the axis into the footprint
        self.assertTrue(in_cell(arm((L.STRIPE_X, 0.0, L.TOP + 0.05))))
        self.assertTrue(in_cell(arm((0.21, 0.0, L.TOP + 0.05))))  # axis 5 cm before the boundary, open: inside
        self.assertFalse(in_cell(arm((0.21, 0.0, L.TOP + 0.05), 0.0)))  # closed (zone 16 mm): outside
        self.assertFalse(in_cell(arm((0.20, 0.0, L.TOP + 0.05))))  # open, 6 cm before it (zone 56.6 mm): outside
        # height: the zone runs 38 mm up from the tips; above 30 cm it is outside the volume
        self.assertTrue(in_cell(arm((L.PART_X, L.PART_Y, L.TOP + 0.30 + TIPS - 0.001))))  # tips 1 mm under 30 cm
        self.assertFalse(in_cell(arm((L.PART_X, L.PART_Y, L.TOP + 0.30 + TIPS + 0.0566))))  # a zone radius over it
        # the ready poses are outside
        self.assertFalse(in_cell(arm(READY_L)))
        self.assertFalse(in_cell(arm(READY_R)))

    def test_pad_pressed_and_released(self):
        self.assertTrue(pad_pressed(arm(PRESS, CLOSED)))
        self.assertTrue(pad_pressed(arm(PRESS, 0.5)))
        self.assertFalse(pad_pressed(arm(PRESS, 0.51)))  # fingers too open
        self.assertFalse(pad_pressed(arm(PRESS, OPEN)))
        self.assertTrue(pad_pressed(arm(PRESS + [0.015, 0, 0.007], CLOSED)))  # 15 mm off, tips 9 mm up
        self.assertFalse(pad_pressed(arm(PRESS + [0.017, 0, 0], CLOSED)))  # 17 mm off the axis
        self.assertFalse(pad_pressed(arm(PRESS + [0, 0, 0.009], CLOSED)))  # tips 11 mm above the top
        self.assertTrue(pad_pressed(arm(PRESS - [0, 0, 0.0065], CLOSED)))  # tips 4.5 mm 'into' the pad (sag)
        self.assertFalse(pad_pressed(arm(PRESS - [0, 0, 0.008], CLOSED)))  # 6 mm into it: not a press
        self.assertFalse(pad_released(arm(PRESS, CLOSED)))
        self.assertFalse(pad_released(arm(PRESS + [0, 0, 0.027], CLOSED)))  # tips 29 mm up: neither
        self.assertTrue(pad_released(arm(PRESS + [0, 0, 0.029], CLOSED)))  # 31 mm up
        self.assertTrue(pad_released(arm(PRESS + [0.031, 0, 0], CLOSED)))  # 31 mm off the axis
        self.assertTrue(pad_released(arm(PAD_HOVER, CLOSED)))

    def test_part_places(self):
        self.assertTrue(centre_in_cell(box(P_START)))
        self.assertTrue(centre_in_cell(box(P_START + [0, 0, 0.1])))  # held up inside the footprint: still in
        self.assertFalse(centre_in_cell(box((0.255, 0.0, L.TOP + H))))  # centre on the stripe: out
        self.assertFalse(centre_in_cell(box(IN_TRAY)))
        self.assertTrue(part_in_tray(box(IN_TRAY)))
        self.assertTrue(part_in_tray(box(IN_TRAY + [0.03, -0.06, 0.002])))  # off centre, 2 mm up
        self.assertFalse(part_in_tray(box(IN_TRAY + [0, 0, 0.004])))  # held 4 mm up
        self.assertFalse(part_in_tray(box(IN_TRAY + [0.045, 0, 0])))  # centre over the wall
        self.assertFalse(part_in_tray(box((L.TRAY_X, L.TRAY_Y - 0.08, L.TRAY_TOP_Z + H))))  # on the wall
        self.assertFalse(part_lifted(box(P_START + [0, 0, 0.019])))
        self.assertTrue(part_lifted(box(P_START + [0, 0, 0.020])))

    def test_let_go(self):
        lo, hi = box(IN_TRAY).low, box(IN_TRAY).high
        self.assertTrue(let_go(arm(TRAY_GRIP, OPEN), lo, hi))
        self.assertTrue(let_go(arm(TRAY_GRIP, 0.9), lo, hi))
        self.assertFalse(let_go(arm(TRAY_GRIP, 0.85), lo, hi))  # 80 mm apart, still round the part
        self.assertFalse(let_go(arm(TRAY_GRIP, GRIP), lo, hi))
        self.assertTrue(let_go(arm(TRAY_GRIP + [0, 0, 0.10], GRIP), lo, hi))  # closed 10 cm above it: not holding
        arms = {"left": arm(PRESS, CLOSED), "right": arm(TRAY_GRIP + [0, 0, 0.1], OPEN)}
        self.assertTrue(grippers_let_go(arms, box(IN_TRAY)))
        arms["left"] = arm(TRAY_GRIP, GRIP)  # the left could not reach, but if it held the part it would count
        self.assertFalse(grippers_let_go(arms, box(IN_TRAY)))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.breach_steps, 0)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertEqual(t.entries, 1)
        self.assertGreater(t.in_cell_steps, 100)
        self.assertTrue(r.details["pad_released"])
        self.assertTrue(r.details["grippers_let_go"])
        self.assertGreater(r.details["pressed_idle_before_s"], 0.5)
        self.assertGreater(r.details["pressed_idle_after_s"], 0.5)

    def test_pad_still_pressed_scores_075(self):
        r, t = play(procedure(release=False))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        self.assertTrue(r.details["part_in_tray"])
        self.assertFalse(r.details["pad_released"])
        # the left lifted only 2 cm: tips 22 mm over the pad, neither pressed nor released: still 0.75
        f = procedure(release=False, settle=0).go(20, L=PRESS + [0, 0, 0.02]).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["pad_pressed"])
        self.assertFalse(r.details["pad_released"])

    def test_gripper_still_holding_the_part_in_the_tray_scores_075(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        right_recovers(f, open_to=GRIP, drop=False, up=False)  # set down 2 mm up, the fingers never open
        left_releases(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        self.assertFalse(r.details["grippers_let_go"])
        self.assertTrue(r.details["part_in_tray"])

    def test_part_out_of_the_cell_but_not_in_the_tray_scores_half(self):
        beside = np.array([0.20, -0.08, L.TOP + H])  # on the table between the cell and the tray
        r, t = play(procedure(to=beside))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertFalse(t.breach)
        # still held in the air over the tray: out of the cell, not resting in it
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        right_recovers(f)
        r, _ = play(f, until=len(f.frames) - 100)
        self.assertFalse(r.details["part_in_tray"])
        self.assertEqual(r.score, 0.5)
        # on the tray's wall: 0.5
        r, _ = play(procedure(to=np.array([L.TRAY_X, L.TRAY_Y - 0.073, L.TRAY_TOP_Z + H])))
        self.assertEqual(r.score, 0.5)

    def test_lifted_with_the_pad_pressed_but_still_in_the_cell_scores_quarter(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        f.go(30, R=WAIT_R).go(30, R=[*P_START[:2], HOVER_R]).go(30, R=PART_GRIP).go(30, Ro=GRIP)
        f.go(30, R=[*P_START[:2], HOVER_R], p=[*P_START[:2], HOVER_R - HELD])
        r, t = play(f.hold(20))  # held up inside the cell
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.breach)
        self.assertTrue(t.lifted_pressed)
        f.go(30, R=PART_GRIP, p=P_START).go(30, Ro=OPEN).go(30, R=[*P_START[:2], HOVER_R]).go(30, R=WAIT_R)
        left_releases(f)
        r, t = play(f.hold(20))  # put back down where it was: still 0.25
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.breach)
        self.assertEqual(t.entries, 1)

    def test_lifted_only_19mm_is_not_lifted(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        f.go(30, R=WAIT_R).go(30, R=[*P_START[:2], HOVER_R]).go(30, R=PART_GRIP).go(30, Ro=GRIP)
        f.go(20, R=PART_GRIP + [0, 0, 0.019], p=P_START + [0, 0, 0.019]).hold(10)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_part_off_the_table_scores_zero(self):
        f = procedure(settle=5)
        f.go(30, p=[0.30, -0.70, L.TOP - 0.40]).hold(10)  # swept off the table's edge afterwards
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertTrue(r.details["part_off_table"])

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(3, p=IN_TRAY + [0.004, 0, 0]).go(3, p=IN_TRAY))  # still sliding in the tray
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))  # stopped
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=PAD_HOVER, left_open=CLOSED, right_tcp=TRAY_GRIP + [0, 0, 0.1])
        r = tr.update(box(IN_TRAY, vel=(0.02, 0, 0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(IN_TRAY, spin=(0, 0, 1.0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(IN_TRAY), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        a = snap(left_tcp=PAD_HOVER, left_open=CLOSED, right_tcp=TRAY_GRIP + [0, 0, 0.1])
        self.assertEqual(t.result(box(IN_TRAY), a).score, 1.0)
        self.assertEqual(t.result(box(IN_TRAY), a).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))

    def test_the_part_may_land_on_any_face(self):
        f = procedure(settle=0).go(1, pq=about("x", 90)).go(1, pq=about("y", 180)).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))


class Breaches(unittest.TestCase):
    """A gripper in the cell with the pad not pressed on the same step: 0 for the rest of the trial."""

    def assert_breach(self, f, arms=("right",)):
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertTrue(t.breach)
        self.assertEqual(t.breach_arms, set(arms))
        self.assertEqual(r.details["first_breach_step"], t.first_breach_step)
        return t

    def test_reaching_in_before_the_pad_is_pressed(self):
        f = Film().hold(10)
        f.go(30, R=WAIT_R).go(30, R=[*P_START[:2], HOVER_R]).go(30, R=WAIT_R)  # in and straight out again
        left_presses(f)
        f.hold(20)
        right_recovers(f)
        left_releases(f)
        t = self.assert_breach(f.hold(30))
        self.assertLess(t.first_breach_step, 60)
        self.assertEqual(t.entries, 2)

    def test_pad_released_while_the_gripper_is_still_inside(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        f.go(30, R=WAIT_R).go(30, R=[*P_START[:2], HOVER_R]).go(30, R=PART_GRIP).go(30, Ro=GRIP)
        f.go(30, R=[*P_START[:2], HOVER_R], p=[*P_START[:2], HOVER_R - HELD])
        left_releases(f)  # too early: the right is still over the cell floor
        f.go(40, R=[*IN_TRAY[:2], HOVER_R], p=[*IN_TRAY[:2], HOVER_R - HELD])
        f.go(30, R=TRAY_GRIP, p=IN_TRAY + [0, 0, 0.002]).go(8, Ro=0.4).go(2, p=IN_TRAY).go(22, Ro=OPEN)
        f.go(20, R=[*IN_TRAY[:2], HOVER_R])
        t = self.assert_breach(f.hold(30))
        self.assertGreater(t.breach_steps, 10)

    def test_pad_let_up_for_one_step_is_a_breach(self):
        f = procedure(settle=0)
        n = next(i for i, fr in enumerate(f.frames) if np.allclose(fr["R"], PART_GRIP) and fr["Ro"] == GRIP)
        f.frames[n] = dict(f.frames[n], L=PRESS + [0, 0, 0.012])  # the tips 14 mm up: not pressed, this step only
        t = self.assert_breach(f.hold(30))
        self.assertEqual(t.breach_steps, 1)
        self.assertEqual(t.first_breach_step, n + 1)

    def test_pad_pressed_with_open_fingers_does_not_count(self):
        f = Film().hold(10)
        f.go(30, L=PAD_HOVER).go(50, L=PRESS)  # down on the pad, fingers wide open (not 'closed')
        f.hold(20)
        right_recovers(f)
        left_releases(f)
        self.assert_breach(f.hold(30))

    def test_pad_pressed_off_centre_does_not_count(self):
        f = Film().hold(10)
        f.go(30, L=PAD_HOVER).go(30, Lo=CLOSED).go(50, L=PRESS + [0.0, 0.02, 0])  # 20 mm off the axis
        f.hold(20)
        right_recovers(f)
        left_releases(f)
        self.assert_breach(f.hold(30))

    def test_left_gripper_entering_the_cell_is_a_breach_too(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        f.go(30, L=[*P_START[:2], HOVER_R])  # the left leaves the pad for the part: not pressed once it is off
        t = self.assert_breach(f.hold(10), arms=("left",))
        self.assertGreater(t.breach_steps, 0)

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5).go(20, R=[0.30, -0.05, HOVER_R]).go(20, R=READY_R)  # a quick look inside
        g = procedure(settle=60)
        f.frames = f.frames + g.frames[1:]
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(r.details["breach"], True)
        self.assertLess(t.first_breach_step, 45)


class Exploits(unittest.TestCase):
    def test_pressing_the_pad_and_doing_nothing_scores_zero(self):
        f = Film().hold(10)
        left_presses(f)
        r, t = play(f.hold(200))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        self.assertGreaterEqual(r.details["pressed_idle_before_s"], 10.0)

    def test_right_arm_alone_scores_zero(self):
        f = Film().hold(10)
        right_recovers(f)  # the pad never pressed: a breach the moment the fingers cross the front boundary
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIn("right", t.breach_arms)
        self.assertTrue(r.details["part_in_tray"])  # the job was done, but unsafely

    def test_left_arm_alone_scores_zero(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(200)
        left_releases(f)
        r, t = play(f.hold(30))  # pressed, waited, released: nothing recovered
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)
        # the left presses, then goes for the part itself: a breach the moment it leaves the pad for the cell
        f = Film().hold(10)
        left_presses(f)
        f.go(30, L=[*P_START[:2], HOVER_R]).go(30, L=PART_GRIP).go(30, Lo=GRIP)
        f.go(30, L=[*P_START[:2], HOVER_R], p=[*P_START[:2], HOVER_R - HELD])
        aside = np.array([0.20, 0.0, L.TOP + H])
        f.go(30, L=[*aside[:2], HOVER_R], p=[*aside[:2], HOVER_R - HELD]).go(30, L=aside + [0, 0, HELD], p=aside)
        f.go(30, Lo=OPEN).go(20, L=PAD_HOVER)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(t.breach_arms, {"left"})

    def test_pushing_the_part_out_of_the_front_with_the_pad_up_scores_zero(self):
        # the open fingers come down behind the part (inside the footprint) and shove it out over the stripe
        f = Film().hold(10)
        f.go(30, R=[L.PART_X + 0.04, L.PART_Y, HOVER_R]).go(30, R=[L.PART_X + 0.04, L.PART_Y, L.TOP + 0.02])
        f.go(40, R=[0.20, L.PART_Y, L.TOP + 0.02], p=[0.17, L.PART_Y, L.TOP + H])
        f.go(20, R=READY_R)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertTrue(t.breach)
        self.assertFalse(r.details["part_centre_in_cell"])  # out of the cell, but it was a breach

    def test_pushing_from_outside_the_stripe_still_puts_the_fingers_inside(self):
        # The gripper's axis stays just before the stripe (x 0.25) while its open fingers reach the part: the zone
        # is in the footprint (the fingers reach 57 mm past the axis), so the push is a breach without the pad.
        f = Film().hold(10)
        f.go(30, R=[0.25, L.PART_Y, HOVER_R]).go(30, R=[0.25, L.PART_Y, L.TOP + 0.02]).go(20, R=READY_R)
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(t.breach)

    def test_pushing_the_part_out_with_the_pad_pressed_scores_half(self):
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        f.go(30, R=[L.PART_X + 0.04, L.PART_Y, HOVER_R]).go(30, R=[L.PART_X + 0.04, L.PART_Y, L.TOP + 0.02])
        f.go(40, R=[0.20, L.PART_Y, L.TOP + 0.02], p=[0.17, L.PART_Y, L.TOP + H])
        f.go(20, R=WAIT_R)
        left_releases(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertFalse(t.breach)

    def test_part_set_down_just_outside_the_stripe_scores_half(self):
        outside = np.array([0.235, 0.02, L.TOP + H])  # its far face 5 mm short of the stripe's near edge
        r, t = play(procedure(to=outside, retreat=True))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertFalse(t.breach)
        # the same without the right gripper backing off before the pad is released: its open fingers, hovering
        # 2.5 cm before the stripe, still reach into the footprint: a breach (0)
        r, t = play(procedure(to=outside))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(t.breach)
        # set down with its centre still over the floor (on the stripe's far edge): 0.25 (lifted, still in)
        r, _ = play(procedure(to=np.array([0.262, 0.02, L.TOP + H]), retreat=True))
        self.assertEqual(r.score, 0.25)

    def test_part_knocked_out_by_a_sweep_above_the_cell(self):
        # A gripper 31 cm up over the cell is outside the volume; it cannot touch the part on the floor, so a
        # 'sweep' there changes nothing (0). If the part moved anyway it would mean the fingers were inside.
        f = Film().hold(10)
        f.go(30, R=[L.PART_X, L.PART_Y, L.TOP + 0.30 + TIPS + 0.06]).go(30, R=[L.PART_X, -0.2, L.TOP + 0.30 + TIPS + 0.06])
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)

    def test_part_dropped_from_above_the_tray_still_counts_if_it_lands_in(self):
        # Letting go 2 cm up is not an exploit: landing flat in the tray and resting is a recovery.
        f = Film().hold(10)
        left_presses(f)
        f.hold(20)
        right_recovers(f, to=IN_TRAY + [0, 0, 0.02])
        f.go(2, p=IN_TRAY)
        left_releases(f)
        r, _ = play(f.hold(30))
        self.assertEqual((r.score, r.success), (1.0, True))


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: pressed, released and let go taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        self.assertIsNone(r.details["pad_pressed"])
        self.assertFalse(t.breach)
        r, t = play(Film().hold(5).go(30, R=[*P_START[:2], HOVER_R]).hold(5), arms=False)  # an unseen entry
        self.assertFalse(t.breach)
        self.assertEqual(r.score, 0.0)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        r = t.update(box(P_START), snap(right_tcp=[*P_START[:2], HOVER_R]))
        self.assertTrue(t.breach)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(box(P_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(box(P_START), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(box(P_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])
        self.assertEqual(r.details["in_cell"], {"left": False, "right": False})


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = box(IN_TRAY)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=IN_TRAY)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (t.breach_steps, t.lifted_pressed, t.entries, t.in_cell_steps, t.arms_unavailable)
            a = snap(left_tcp=PAD_HOVER, left_open=CLOSED, right_tcp=TRAY_GRIP + [0, 0, 0.1])
            for _ in range(3):
                r = t.update(bad, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(bad, a).score, 0.0)
                self.assertFalse(r.details["valid_state"])
            self.assertEqual((t.breach_steps, t.lifted_pressed, t.entries, t.in_cell_steps, t.arms_unavailable), before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [box((5.0, -3.0, -1.0)), box((0.3, 0.0, 0.0), about("x", 120)), box((-0.2, 0.6, L.TOP), about("y", 90)),
               box(IN_TRAY, about("x", 89.9)), box((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        a = snap()
        for s in odd + odd[::-1]:
            r = t.update(s, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertEqual(r.success, r.score == 1.0)
        r = t.update(box(P_START), None)
        self.assertIn(r.score, (0.0, 0.25))


class Reset(unittest.TestCase):
    def breached(self):
        f = Film().hold(5).go(20, R=[0.30, -0.05, HOVER_R])  # a look inside with the pad up
        g = procedure(settle=5)
        f.frames = f.frames + g.frames[1:330]  # the part held up in the cell
        _, t = play(f)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        self.assertGreater(float(np.linalg.norm(t.prev[0] - P_START)), 0.08)
        r = t.update(box(P_START), snap())  # the part jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(box(P_START + [0.02, -0.02, 0]), snap())  # a seeded reset puts it near, not on, its mark
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur = t.prev[0]
        f = Film(p=cur, R=[*cur[:2], HOVER_R], L=PRESS, Lo=CLOSED)
        f.go(40, p=[cur[0], cur[1], HOVER_R - HELD], R=[*cur[:2], HOVER_R])
        f.go(60, p=[*P_START[:2], HOVER_R - HELD], R=[*P_START[:2], HOVER_R])
        f.go(60, p=P_START + [0, 0, 0.0005], R=PART_GRIP + [0, 0, 0.0005]).go(2, p=P_START).hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(box((0.30, 0.20, L.TOP + H)), snap())  # knocked far, not to the start
        self.assertTrue(t.breach)

    def test_a_dropped_part_is_not_a_teleport(self):
        # Free fall from 0.3 m onto its mark: fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = P_START + [0, 0, 0.3]
        t.update(box(p), snap())
        t.breach_steps = 1
        z, v = p[2], 0.0
        while z > P_START[2]:
            v += 9.81 * DT
            z = max(P_START[2], z - v * DT)
            t.update(box((p[0], p[1], z), vel=(0, 0, -v if z > P_START[2] else 0)), snap())
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


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def sample_plan(plan, hz=HZ, ready=None):
    """The scripted plan sampled on one clock: per arm, (tcp, opening) per control step, interpolating linearly
    between waypoints from the ready pose, finger joint values turned into rigstate openings. Both arms run for
    the longer plan's length; a finished arm holds its last target."""
    ready = ready or {"left": READY_L, "right": READY_R}
    total = max((sum(w.duration for w in wps) for wps in plan.values()), default=0.0)
    n = int(round(total * hz))
    out = {}
    for side in ("left", "right"):
        wps = plan.get(side, [])
        pos, fin = np.array(ready[side], float), -0.04695
        segs = []
        for w in wps:
            segs.append((pos, fin, np.array(w.pos, float), w.finger, w.duration))
            pos, fin = np.array(w.pos, float), w.finger
        rows = []
        t0 = 0.0
        for i in range(n + 1):
            t = i / hz
            cur = (pos, fin)
            acc = 0.0
            for p0, f0, p1, f1, d in segs:
                if t <= acc + d:
                    u = (t - acc) / d if d > 0 else 1.0
                    cur = ((1 - u) * p0 + u * p1, (1 - u) * f0 + u * f1)
                    break
                acc += d
            rows.append((cur[0], float(np.clip(cur[1] / -0.04695, 0.0, 1.0))))
        out[side] = rows
    return out, n


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, p=P_START):
        return {"part": box(p)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("part", names)
        self.assertIn("enabling_pad", names)
        self.assertEqual(sum(n.startswith("cell_wall") for n in names), 3)
        self.assertEqual(sum(n.startswith("tray_wall") for n in names), 4)
        self.assertIn("front_stripe", names)
        loose = [o for o in self.task.OBJECTS if o.mass is not None]
        self.assertEqual([o.name for o in loose], ["part"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(right_tcp=[*P_START[:2], HOVER_R])  # the right gripper in the cell, the pad up
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

    def test_demo_switches_and_timing(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        left, right = plan["left"], plan["right"]
        # the right enters the cell only after the left's press is complete, plus a margin
        pressed = sum(w.duration for w in left[:4])
        self.assertIn("press the pad", left[3].label)
        self.assertIn("into the cell", right[2].label)
        enter = sum(w.duration for w in right[:2])
        self.assertGreaterEqual(enter, pressed + 0.5 - 1e-9)
        # the left lifts off the pad only after the right's TCP is over the tray, plus a margin
        self.assertIn("over the tray", right[9].label)
        at_tray = sum(w.duration for w in right[:10])
        self.assertIn("release", left[5].label)
        release = sum(w.duration for w in left[:5])
        self.assertGreaterEqual(release, at_tray + 0.5 - 1e-9)
        # the pad is released before the right lets go of the part (success needs both)
        let_go_end = sum(w.duration for w in right[:15])
        self.assertIn("let go", right[14].label)
        self.assertLess(release + left[5].duration, let_go_end)
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def run_plan(self, plan, settle_s=3.0):
        """Play the plan through the scorer with a kinematic proxy: the TCPs follow their targets exactly, the part
        follows the right TCP while the fingers are closed round it where it stands (or wherever it rests) and
        drops straight down onto the surface below when they open. Returns (last result, tracker, success step)."""
        rows, n = sample_plan(plan)
        n += int(settle_s * HZ)
        t = Tracker(HZ)
        p = P_START.copy()
        held = False
        prev = p.copy()
        hold_steps, done = 0, None
        r = None
        for i in range(n + 1):
            j = min(i, len(rows["left"]) - 1)
            (ltcp, lo), (rtcp, ro) = rows["left"][j], rows["right"][j]
            gap = ro * L.FULL_GAP
            at_part = np.hypot(*(rtcp[:2] - p[:2])) < 0.006 and (p[2] - H) <= rtcp[2] - TIPS <= (p[2] + H)
            if not held and at_part and gap <= L.PART_SIZE[1] + 0.0005:
                held = True
            if held and gap > L.PART_SIZE[1] + 0.002:
                held = False
            if held:
                p = rtcp - np.array([0.0, 0.0, HELD])
            else:
                floor = L.TOP + H  # the table (the cell's and the tray's floors are the table)
                p = np.array([p[0], p[1], floor])
            v = (p - prev) / DT
            prev = p.copy()
            r = t.update(box(p, vel=v), snap(ltcp, lo, rtcp, ro))
            hold_steps = hold_steps + 1 if r.success else 0
            if done is None and hold_steps >= int(self.task.SUCCESS_HOLD_S * HZ):
                done = i
        return r, t, done

    def test_demo_plan_scores_one_in_the_kinematic_proxy(self):
        plan = self.task.demo()
        total = sum(w.duration for w in plan["right"])
        r, t, done = self.run_plan(plan)
        self.assertEqual((r.score, r.success), (1.0, True), r.details)
        self.assertEqual(t.breach_steps, 0)
        self.assertEqual(t.entries, 1)
        self.assertIsNotNone(done)
        self.assertLess(done, total * HZ)  # success holds before the plan ends (during the last 'up')
        self.assertLess(total + 3.0, 60.0)
        self.assertTrue(r.details["pad_released"])
        self.assertTrue(r.details["grippers_let_go"])
        # the pad is pressed for some seconds before the entry and after the exit: the margins
        self.assertGreaterEqual(r.details["pressed_idle_before_s"], 1.0)
        self.assertGreaterEqual(r.details["pressed_idle_after_s"], 1.0)

    def test_one_arm_demos_score_zero(self):
        plan = self.task.demo()
        r, t, _ = self.run_plan({"right": plan["right"]})  # DEMO_IDLE_ARM=left: the right alone
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.breach_arms, {"right"})
        self.assertTrue(r.details["part_in_tray"])
        r, t, _ = self.run_plan({"left": plan["left"]})  # DEMO_IDLE_ARM=right: the left alone
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)
        self.assertTrue(r.details["part_centre_in_cell"])
        self.assertFalse(t.lifted_pressed)

    def test_demo_plan_without_the_pad_hold_margin_would_breach(self):
        # A sanity check of the proxy itself: the right's plan started 2 s earlier is a breach.
        plan = self.task.demo()
        right = list(plan["right"])
        w = right[1]
        right[1] = type(w)(w.duration - 2.0, w.pos, w.finger, w.label, w.yaw)
        r, t, _ = self.run_plan({"left": plan["left"], "right": right})
        self.assertEqual(r.score, 0.0)
        self.assertTrue(t.breach)


if __name__ == "__main__":
    unittest.main()
