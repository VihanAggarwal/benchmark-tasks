"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic object states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
    Tracker, dressing_in_bin, finger_zone, lowest_bare_corner, pad_at_bin, pad_in_tray, pad_on_bare_table,
    pad_on_mark, pads_meet, rotation, tilt_deg, valid, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
H = L.PAD_SIZE[2] / 2  # both pads: half their 16 mm thickness
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.004  # tips this far above a held pad's bottom
HELD = TIPS + GRIP_UP - H  # a held pad's centre is this far below the TCP
OPEN, GRIP = 1.0, (L.PAD_SIZE[0] - 0.008) / L.FULL_GAP  # rigstate openings: fully open, closed on a 50 mm pad

D_START = np.array([L.DRESSING_X, L.DRESSING_Y, L.MARK_TOP + H])
P_START = np.array([L.TRAY_X, L.TRAY_Y, L.TOP + H])
IN_BIN = np.array([L.BIN_X, L.BIN_Y, L.TOP + H])
ON_MARK = np.array([L.MARK_X, L.MARK_Y, L.MARK_TOP + H])
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
CARRY = L.TOP + 0.10
UP_R = np.array([L.BIN_X, L.BIN_Y, L.TOP + 0.13])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0), size=L.PAD_SIZE):
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


KEYS = ("d", "dq", "p", "pq", "L", "Lo", "R", "Ro")


class Film:
    """A trial as a list of frames: both pads' centres and orientations, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"d": D_START, "dq": Q0, "p": P_START, "pq": Q0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in ("d", "p", "L", "R") else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in ("d", "p", "L", "R") else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            fr = {}
            for k in KEYS:
                if k in ("dq", "pq"):
                    fr[k] = b[k] if f >= 1.0 or np.allclose(a[k], b[k]) else (a[k] if f < 0.5 else b[k])
                else:
                    fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
            self.frames.append(fr)
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def __add__(self, other):
        out = Film()
        out.frames = self.frames + other.frames
        out.cur = dict(other.cur)
        return out


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        if prev is None:
            vd = vp = np.zeros(3)
        else:
            vd, vp = (fr["d"] - prev["d"]) / DT, (fr["p"] - prev["p"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(box(fr["d"], fr["dq"], vd), box(fr["p"], fr["pq"], vp), a)
        prev = fr
    return r, t


def right_bins_dressing(f):
    """The right arm takes the used dressing off the wound and lets it drop into the bin (about 9 s)."""
    grip = D_START + np.array([0, 0, HELD])
    f.go(30, R=grip + [0, 0, 0.08]).go(30, R=grip).go(30, Ro=GRIP)
    f.go(24, R=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
    f.go(30, R=[L.BIN_X, L.BIN_Y, CARRY], d=[L.BIN_X, L.BIN_Y, CARRY - HELD])
    rel = L.BIN_TOP_Z + 0.012 + TIPS + GRIP_UP
    f.go(20, R=[L.BIN_X, L.BIN_Y, rel], d=[L.BIN_X, L.BIN_Y, rel - HELD])
    f.go(8, Ro=0.6).go(3, d=IN_BIN).go(20, Ro=OPEN).go(20, R=UP_R)
    return f


def left_picks_pad(f):
    grip = P_START + np.array([0, 0, HELD])
    f.go(30, L=grip + [0, 0, 0.08]).go(30, L=grip).go(30, Lo=GRIP)
    f.go(24, L=[*grip[:2], CARRY], p=[*P_START[:2], CARRY - HELD])
    return f


def left_places_pad(f, at=(L.MARK_X, L.MARK_Y), open_to=OPEN, pq=Q0):
    x, y = at
    place = L.MARK_TOP + 0.002 + TIPS + GRIP_UP
    f.go(36, L=[x, y, CARRY], p=[x, y, CARRY - HELD])
    f.go(30, L=[x, y, place], p=[x, y, place - HELD])
    f.go(6, Lo=0.6).go(2, p=[x, y, L.MARK_TOP + H], pq=pq).go(20, Lo=open_to)
    return f


def procedure(at=(L.MARK_X, L.MARK_Y), open_to=OPEN, settle=30, pq=Q0):
    """The intended trial, in sequence: the right arm bins the used dressing, then the left arm takes the clean pad
    from the tray and sets it on the wound mark at ``at``, opening to ``open_to``; all still for ``settle`` steps."""
    f = Film().hold(10)
    right_bins_dressing(f)
    left_picks_pad(f)
    left_places_pad(f, at, open_to, pq)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_plain_pad(self):
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 180)), 0.0, places=4)  # upside down: the same pad
        self.assertAlmostEqual(tilt_deg(about("x", 170)), 10.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 90.0, places=4)

    def test_finger_zone_covers_the_pads(self):
        a = snap(left_tcp=(0.2, 0.2, 0.9), left_open=OPEN)["left"]
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)  # 56.6 mm
        pts, r = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], 0.9 - TIPS)  # from the tips ...
        self.assertAlmostEqual(pts[-1][2], 0.9 - TIPS + L.FINGER_LEN)  # ... 38 mm back up the gripper
        # an open gripper 50 mm to the side of a box, at its height: its outer pad is already past the box's face
        lo, hi = np.array([0.25, 0.15, 0.88]), np.array([0.30, 0.20, 0.90])
        self.assertLess(zone_gap(a, lo, hi), 0.0)
        closed = snap(left_tcp=(0.2, 0.2, 0.9), left_open=0.0)["left"]
        self.assertAlmostEqual(zone_radius(closed), math.hypot(0.008, 0.0135), places=6)
        self.assertAlmostEqual(zone_gap(closed, lo, hi), 0.05 - zone_radius(closed), places=6)

    def test_bare_table(self):
        self.assertFalse(pad_on_bare_table(box(P_START)))  # in the tray: the tray's floor
        self.assertFalse(pad_on_bare_table(box(ON_MARK)))  # flat on the wound mark
        self.assertTrue(pad_on_bare_table(box((0.20, 0.0, L.TOP + H))))  # on the table between them
        self.assertFalse(pad_on_bare_table(box((0.20, 0.0, L.TOP + H + 0.002))))  # held 2 mm above it
        self.assertTrue(pad_on_bare_table(box((0.20, 0.0, L.TOP - 0.3))))  # fallen off the table
        # 20 mm off centre and flat on the mark: its overhanging edge is 2 mm up in the air, not on the table
        over = box((L.MARK_X + 0.020, L.MARK_Y, L.MARK_TOP + H))
        self.assertFalse(pad_on_bare_table(over))
        self.assertAlmostEqual(lowest_bare_corner(over), L.MARK_SIZE[2], places=6)
        # the same, tipped so the overhanging edge rests on the table beside the mark: touching the bare table
        tipped = box((L.MARK_X + 0.020, L.MARK_Y, L.TOP + 0.0089), about("y", 2.3))
        self.assertTrue(pad_on_bare_table(tipped))
        # leaning on the outside of the tray's front wall, one edge on the table outside the tray
        lean = box((L.TRAY_X + L.TRAY_INNER_X / 2 + 0.006 + 0.022, L.TRAY_Y, L.TOP + 0.012), about("y", -20))
        self.assertTrue(pad_on_bare_table(lean))
        # in the bin (the bin's floor is the table): bare table and the bin
        self.assertTrue(pad_on_bare_table(box(IN_BIN)) and pad_at_bin(box(IN_BIN)))

    def test_bin_and_tray(self):
        self.assertTrue(dressing_in_bin(box(IN_BIN)))
        self.assertTrue(dressing_in_bin(box((L.BIN_X + 0.015, L.BIN_Y - 0.01, L.TOP + 0.02), about("x", 40))))
        self.assertFalse(dressing_in_bin(box((L.BIN_X, L.BIN_Y, L.BIN_TOP_Z + 0.02))))  # held over it
        self.assertFalse(dressing_in_bin(box((L.BIN_X + 0.049, L.BIN_Y, L.BIN_TOP_Z + H))))  # on the rim
        self.assertFalse(dressing_in_bin(box(D_START)))
        self.assertTrue(pad_in_tray(box(P_START)))
        self.assertFalse(pad_in_tray(box(P_START + [0, 0, 0.02])))  # lifted
        self.assertTrue(pad_at_bin(box((L.BIN_X, L.BIN_Y - 0.06, L.BIN_TOP_Z + H))))  # resting on a bin wall

    def test_pads_meet(self):
        self.assertFalse(pads_meet(box(D_START), box(P_START)))
        self.assertTrue(pads_meet(box(D_START), box(D_START + [0, 0, 0.016])))  # stacked on it
        self.assertTrue(pads_meet(box(D_START), box(D_START + [0.03, 0, 0.12])))  # carried over it, high up
        self.assertTrue(pads_meet(box(D_START), box(D_START + [0.052, 0, 0])))  # side by side, 2 mm apart
        self.assertFalse(pads_meet(box(D_START), box(D_START + [0.06, 0, 0])))  # 10 mm apart

    def test_on_mark_tolerances(self):
        self.assertTrue(pad_on_mark(box(ON_MARK), L.CENTRE_TIGHT))
        self.assertTrue(pad_on_mark(box(ON_MARK + [0.007, -0.007, 0]), L.CENTRE_TIGHT))  # 9.9 mm off
        self.assertFalse(pad_on_mark(box(ON_MARK + [0.011, 0, 0]), L.CENTRE_TIGHT))
        self.assertTrue(pad_on_mark(box(ON_MARK + [0.024, 0, 0]), L.CENTRE_LOOSE))
        self.assertFalse(pad_on_mark(box(ON_MARK + [0.026, 0, 0]), L.CENTRE_LOOSE))
        self.assertFalse(pad_on_mark(box(ON_MARK + [0, 0, 0.004]), L.CENTRE_TIGHT))  # held 4 mm above it
        self.assertFalse(pad_on_mark(box(ON_MARK, about("x", 6)), L.CENTRE_LOOSE))  # tilted
        self.assertTrue(pad_on_mark(box(ON_MARK, about("x", 180)), L.CENTRE_TIGHT))  # either face down


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.breach)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.arms_unavailable, 0)
        self.assertGreater(t.min_left_dressing, 0.10)  # the clean hand never near the used dressing
        self.assertGreater(t.min_right_pad, 0.10)  # the dirty hand never near the clean pad

    def test_off_centre_scores_075(self):
        r, _ = play(procedure(at=(L.MARK_X + 0.015, L.MARK_Y)))
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(at=(L.MARK_X + 0.0085, L.MARK_Y - 0.0045)))  # 9.6 mm off
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(at=(L.MARK_X - 0.017, L.MARK_Y + 0.017)))  # 24 mm off diagonally
        self.assertEqual(r.score, 0.75)

    def test_gripper_still_closed_scores_075(self):
        r, _ = play(procedure(open_to=GRIP))
        self.assertEqual((r.score, r.success), (0.75, False))
        r, _ = play(procedure(open_to=0.85))  # 80 mm apart, still round the pad: not let go
        self.assertEqual(r.score, 0.75)

    def test_a_gripper_closed_away_from_its_item_has_let_go(self):
        # Closing a gripper again once it is clear of everything changes nothing that matters: still 1.0.
        f = procedure()
        f.go(20, Ro=0.0).hold(25)  # the right gripper closes up over the bin, 10 cm above the dressing
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertTrue(r.details["grippers_let_go"])
        self.assertFalse(r.details["grippers_open"])
        f = procedure(settle=0)
        f.go(20, L=ON_MARK + [0, 0, 0.10]).go(20, Lo=0.0).hold(25)  # the left goes up 10 cm, then closes
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_right_gripper_still_holding_the_dressing_in_the_bin_scores_075(self):
        # The dressing let down into the bin but the right gripper still closed on it there: not put in.
        f = Film().hold(10)
        grip = D_START + [0, 0, HELD]
        f.go(30, R=grip).go(30, Ro=GRIP).go(24, R=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
        f.go(30, R=[L.BIN_X, L.BIN_Y, CARRY], d=[L.BIN_X, L.BIN_Y, CARRY - HELD])
        f.go(30, R=IN_BIN + [0, 0, HELD], d=IN_BIN)  # down inside the bin, still gripped
        left_picks_pad(f)
        left_places_pad(f)
        r, t = play(f.hold(30))
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(t.breach)
        self.assertFalse(r.details["grippers_let_go"])

    def test_far_off_centre_scores_half(self):
        r, t = play(procedure(at=(L.MARK_X + 0.030, L.MARK_Y)))  # flat, 30 mm off: 20 mm hangs over the edge
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_tilted_on_the_mark_scores_half(self):
        r, t = play(procedure(pq=about("x", 7)))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_dressing_binned_pad_untouched_scores_half(self):
        r, t = play(right_bins_dressing(Film().hold(10)).hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.breach)

    def test_dressing_binned_pad_held_in_the_air_scores_half(self):
        f = right_bins_dressing(Film().hold(10))
        r, _ = play(left_picks_pad(f).hold(30))
        self.assertEqual(r.score, 0.5)

    def test_dressing_set_aside_scores_quarter(self):
        aside = np.array([0.24, -0.08, L.TOP + H])  # on the table, clear of the wound mark
        f = Film().hold(5)
        grip = D_START + [0, 0, HELD]
        f.go(30, R=grip).go(30, Ro=GRIP).go(20, R=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
        f.go(20, R=[*aside[:2], CARRY], d=[*aside[:2], CARRY - HELD]).go(20, R=aside + [0, 0, HELD], d=aside)
        f.go(20, Ro=OPEN).go(20, R=UP_R).hold(20)
        r, t = play(f)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(t.breach)
        r, _ = play(f, until=110)  # still held in the air beside the wound: off the wound too
        self.assertEqual(r.score, 0.25)

    def test_dressing_barely_moved_scores_zero(self):
        f = Film().hold(5).go(10, d=D_START + [0.03, 0, 0]).hold(20)  # still partly over the wound mark
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_dressing_off_the_table_scores_zero(self):
        f = Film().hold(5).go(30, d=[0.30, -0.70, L.TOP - 0.40]).hold(20)  # swept off the table's edge
        f.go(1, R=READY_R)
        r, t = play(left_places_pad(left_picks_pad(f.hold(5))).hold(30))
        # the pad came over the mark with the dressing not in the bin: a breach as well
        self.assertEqual(r.score, 0.0)
        f = Film().hold(5).go(30, d=[0.30, -0.70, L.TOP - 0.40]).hold(20)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.breach)  # 0 from the dressing's end place alone

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(3, p=ON_MARK + [0.004, 0, 0]).go(3, p=ON_MARK))  # still sliding on the mark
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))  # stopped
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        moving = box(ON_MARK, vel=(0.02, 0, 0))
        r = tr.update(box(IN_BIN), moving, snap(left_tcp=ON_MARK + [0, 0, 0.1], right_tcp=UP_R))
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(IN_BIN), box(ON_MARK, spin=(0, 0, 1.0)), snap(left_tcp=ON_MARK + [0, 0, 0.1],
                                                                          right_tcp=UP_R))
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(box(IN_BIN, vel=(0, 0, -0.05)), box(ON_MARK), snap(left_tcp=ON_MARK + [0, 0, 0.1],
                                                                          right_tcp=UP_R))
        self.assertEqual((r.score, r.success), (1.0, False))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        last = (box(IN_BIN), box(ON_MARK), snap(left_tcp=ON_MARK + [0, 0, 0.1], right_tcp=UP_R))
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.result(*last).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))


class Breaches(unittest.TestCase):
    """Each rule broken once scores 0 for the rest of the trial, whatever follows."""

    def assert_zero(self, f, key):
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn(key, t.breaches)
        return t

    def test_left_gripper_touching_the_dressing(self):
        f = Film().hold(5).go(30, L=D_START + [0, 0, 0.06]).go(20, L=D_START + [0, 0, 0.03])  # open, over it
        f.go(30, L=READY_L)
        f2 = Film()
        f2.frames = f.frames + procedure().frames[1:]
        self.assert_zero(f2, "left_touched_dressing")

    def test_left_open_fingers_beside_the_dressing(self):
        # the axis 70 mm from the dressing's centre (45 mm from its side), low: the open pads reach to 56.6 mm
        f = Film().hold(5).go(30, L=D_START + [0.07, 0, 0.004]).go(20, L=READY_L).hold(5)
        t = self.assert_zero(f, "left_touched_dressing")
        self.assertLess(t.min_left_dressing, 0.0)
        # the same with the fingers closed: 45 mm off, clear (zone 15.7 mm)
        f = Film().hold(5).go(10, Lo=0.0).go(30, L=D_START + [0.07, 0, 0.004]).go(20, L=READY_L).hold(5)
        r, t = play(f)
        self.assertNotIn("left_touched_dressing", t.breaches)

    def test_right_gripper_near_the_clean_pad(self):
        # open, its axis 110 mm from the pad's centre (85 mm from its side), the tips 15 mm above the table:
        # the zone's edge 28 mm from the pad
        f = Film().hold(5).go(30, R=P_START + [0, -0.11, 0.02]).go(20, R=READY_R)
        f2 = Film()
        f2.frames = f.frames + procedure().frames[1:]
        t = self.assert_zero(f2, "right_near_pad")
        self.assertLess(t.min_right_pad, L.KEEP_OUT)
        # 115 mm off (zone 56.6 mm + 30 mm + the pad's 25 mm half-width = 111.6 mm): clear
        f = Film().hold(5).go(30, R=P_START + [0, -0.115, 0.0]).go(20, R=READY_R)
        _, t = play(f)
        self.assertNotIn("right_near_pad", t.breaches)

    def test_pad_on_the_dressing(self):
        f = Film().hold(5)
        left_picks_pad(f)
        left_places_pad(f, at=(L.MARK_X, L.MARK_Y))  # the left arm alone: onto the used dressing
        t = self.assert_zero(f.hold(20), "pad_met_dressing")
        self.assertIn("pad_before_bin", t.breaches)

    def test_pad_over_the_mark_before_the_dressing_is_binned(self):
        f = Film().hold(10)
        grip = D_START + [0, 0, HELD]
        f.go(30, R=grip).go(30, Ro=GRIP).go(24, R=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
        f.go(30, R=[L.BIN_X, L.BIN_Y, CARRY], d=[L.BIN_X, L.BIN_Y, CARRY - HELD])  # over the bin, not yet in it
        left_picks_pad(f)
        left_places_pad(f)
        f.go(10, d=IN_BIN, Ro=OPEN).hold(30)  # dropped in only afterwards
        t = self.assert_zero(f, "pad_before_bin")
        self.assertNotIn("pad_met_dressing", t.breaches)

    def test_pad_slid_along_the_table(self):
        f = Film().hold(5)
        right_bins_dressing(f)
        grip = P_START + [0, 0, HELD]
        f.go(30, L=grip).go(30, Lo=GRIP)
        f.go(10, L=grip + [0, -0.06, 0], p=P_START + [0, -0.06, 0])  # dragged out over the wall... onto the table
        f.go(40, L=[L.MARK_X, L.MARK_Y + 0.06, grip[2]], p=[L.MARK_X, L.MARK_Y + 0.06, L.TOP + H])
        f.go(20, L=[L.MARK_X, L.MARK_Y, grip[2] + 0.002], p=ON_MARK).go(20, Lo=OPEN).hold(30)
        self.assert_zero(f, "pad_on_bare_table")

    def test_pad_set_down_on_the_table_then_placed(self):
        f = Film().hold(5)
        right_bins_dressing(f)
        left_picks_pad(f)
        spot = np.array([0.24, 0.10, L.TOP + H])
        f.go(20, L=spot + [0, 0, HELD + 0.05], p=spot + [0, 0, 0.05]).go(20, L=spot + [0, 0, HELD], p=spot)
        f.go(20, Lo=OPEN).go(20, Lo=GRIP)  # put down to change grip, picked up again
        f.go(20, L=spot + [0, 0, HELD + 0.08], p=spot + [0, 0, 0.08])
        left_places_pad(f)
        self.assert_zero(f.hold(30), "pad_on_bare_table")

    def test_pad_touching_the_bin(self):
        f = procedure(settle=5)
        # afterwards the left arm (it cannot reach so far) or a shove puts the pad on the bin's wall: still 0
        f.go(20, p=[L.BIN_X, L.BIN_Y + 0.06, L.BIN_TOP_Z + H]).hold(20)
        self.assert_zero(f, "pad_at_bin")

    def test_a_breach_is_never_undone(self):
        f = Film().hold(5).go(20, R=P_START + [0, -0.09, 0.0]).go(20, R=READY_R)  # the dirty hand by the pad
        f2 = Film()
        f2.frames = f.frames + procedure(settle=60).frames[1:]
        r, t = play(f2)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(t.breaches["right_near_pad"], min(t.breaches.values()))
        self.assertEqual(r.details["first_breach_step"], t.breaches["right_near_pad"])


class Exploits(unittest.TestCase):
    def test_left_hand_does_everything_scores_zero(self):
        # The left (clean) gripper lifts the used dressing off and sets it aside, then places the clean pad.
        f = Film().hold(5)
        grip = D_START + [0, 0, HELD]
        aside = np.array([0.20, 0.08, L.TOP + H])
        f.go(30, L=grip).go(30, Lo=GRIP).go(20, L=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
        f.go(20, L=[*aside[:2], CARRY], d=[*aside[:2], CARRY - HELD]).go(20, L=aside + [0, 0, HELD], d=aside)
        f.go(20, Lo=OPEN)
        left_picks_pad(f)
        left_places_pad(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("left_touched_dressing", t.breaches)

    def test_right_hand_places_the_pad_scores_zero(self):
        # The right (dirty) gripper bins the dressing, then the left hands it the clean pad... or it takes it.
        f = Film().hold(5)
        right_bins_dressing(f)
        left_picks_pad(f)
        mid = np.array([0.30, 0.08, CARRY])
        f.go(30, L=mid, p=mid - [0, 0, HELD])
        f.go(30, R=mid + [0.0, -0.002, 0.0]).go(20, Ro=GRIP).go(20, Lo=OPEN)  # handed over to the right
        f.go(20, L=READY_L).go(30, R=[L.MARK_X, L.MARK_Y, CARRY], p=[L.MARK_X, L.MARK_Y, CARRY - HELD])
        f.go(20, R=[L.MARK_X, L.MARK_Y, L.MARK_TOP + 0.019], p=ON_MARK + [0, 0, 0.002]).go(2, p=ON_MARK)
        f.go(20, Ro=OPEN).go(20, R=UP_R).hold(30)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("right_near_pad", t.breaches)

    def test_carrying_the_dressing_over_the_clean_tray_scores_zero(self):
        # The used dressing is carried across the sterile tray (a short way round), the rest done right.
        f = Film().hold(5)
        grip = D_START + [0, 0, HELD]
        f.go(30, R=grip).go(30, Ro=GRIP).go(20, R=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
        f.go(20, R=[L.TRAY_X, 0.15, CARRY + 0.05], d=[L.TRAY_X, 0.15, CARRY + 0.05 - HELD])
        f.go(20, R=[L.TRAY_X, 0.21, CARRY + 0.05], d=[L.TRAY_X, 0.21, CARRY + 0.05 - HELD])  # over the clean pad
        f.go(40, R=[L.BIN_X, L.BIN_Y, CARRY], d=[L.BIN_X, L.BIN_Y, CARRY - HELD])
        f.go(10, d=IN_BIN, Ro=OPEN).go(20, R=UP_R)
        left_picks_pad(f)
        left_places_pad(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("pad_met_dressing", t.breaches)

    def test_pushing_the_pad_off_the_dressing_onto_the_mark(self):
        # Shove the used dressing off the wound with the clean pad (held by the left), then set the pad there.
        f = Film().hold(5)
        left_picks_pad(f)
        low = L.MARK_TOP + H + 0.001
        f.go(30, L=[L.MARK_X, 0.07, low + HELD], p=[L.MARK_X, 0.07, low])
        f.go(20, L=[L.MARK_X, 0.0505, low + HELD], p=[L.MARK_X, 0.0505, low])  # up against the dressing
        f.go(30, L=[L.MARK_X, 0.0, low + HELD], p=[L.MARK_X, 0.0, low], d=[L.MARK_X, -0.0505, L.TOP + H])
        f.go(20, Lo=OPEN).hold(30)
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("pad_met_dressing", t.breaches)

    def test_dropping_the_pad_from_high_up_lands_fine_only_if_it_lands_flat(self):
        # Not an exploit: a pad dropped from 3 cm that lands flat and centred is a placement; one that bounces
        # off the edge onto the table is a breach.
        f = Film().hold(5)
        right_bins_dressing(f)
        left_picks_pad(f)
        f.go(30, L=[L.MARK_X, L.MARK_Y, CARRY], p=[L.MARK_X, L.MARK_Y, CARRY - HELD])
        f.go(20, L=[L.MARK_X, L.MARK_Y, L.MARK_TOP + 0.03 + HELD], p=[L.MARK_X, L.MARK_Y, L.MARK_TOP + 0.03])
        g = Film()
        g.frames, g.cur = list(f.frames), dict(f.cur)
        f.go(10, Lo=OPEN).go(3, p=ON_MARK).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        g.go(10, Lo=OPEN).go(3, p=ON_MARK).go(4, p=[L.MARK_X + 0.06, L.MARK_Y, L.TOP + H]).hold(30)
        r, t = play(g)
        self.assertEqual(r.score, 0.0)
        self.assertIn("pad_on_bare_table", t.breaches)

    def test_dressing_on_the_bin_rim_then_pad_placed_scores_zero(self):
        f = Film().hold(5)
        grip = D_START + [0, 0, HELD]
        rim = np.array([L.BIN_X + 0.049, L.BIN_Y, L.BIN_TOP_Z + H])  # balanced on the front wall
        f.go(30, R=grip).go(30, Ro=GRIP).go(20, R=[*grip[:2], CARRY], d=[*D_START[:2], CARRY - HELD])
        f.go(30, R=[*rim[:2], CARRY], d=[*rim[:2], CARRY - HELD]).go(20, R=rim + [0, 0, HELD], d=rim)
        f.go(20, Ro=OPEN).go(20, R=UP_R)
        r, t = play(f.hold(10))
        self.assertEqual(r.score, 0.25)  # off the wound, not in the bin
        left_picks_pad(f)
        left_places_pad(f)
        r, t = play(f.hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("pad_before_bin", t.breaches)


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'both open' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        f = procedure(open_to=GRIP)  # even a closed gripper cannot be seen
        r, _ = play(f, arms=False)
        self.assertEqual(r.score, 1.0)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        r = t.update(box(D_START), box(P_START), snap(right_tcp=P_START + [0, -0.05, 0.02]))
        self.assertIn("right_near_pad", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(box(D_START), box(P_START), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(box(D_START), box(P_START), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(box(D_START), box(P_START), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_either_face_down(self):
        r, _ = play(procedure(pq=about("y", 180)))
        self.assertEqual(r.score, 1.0)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = box(ON_MARK)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=ON_MARK)
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.breaches), t.min_left_dressing, t.min_right_pad, t.arms_unavailable)
            a = snap(left_tcp=ON_MARK + [0, 0, 0.1], right_tcp=UP_R)
            for d, p in ((box(IN_BIN), bad), (bad, box(ON_MARK)), (bad, bad)):
                r = t.update(d, p, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(d, p, a).score, 0.0)
            self.assertEqual((dict(t.breaches), t.min_left_dressing, t.min_right_pad, t.arms_unavailable), before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [box((5.0, -3.0, -1.0)), box((0.3, 0.0, 0.0), about("x", 120)), box((-0.2, 0.6, L.TOP), about("y", 90)),
               box(ON_MARK, about("x", 89.9)), box((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        a = snap()
        for d, p in zip(odd + odd[::-1], odd[::-1] + odd):
            r = t.update(d, p, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
        r = t.update(box(D_START), box(P_START), None)
        self.assertIn(r.score, (0.0, 0.25))


class Reset(unittest.TestCase):
    def breached(self):
        f = Film().hold(5).go(20, R=P_START + [0, -0.06, 0.02])  # the dirty hand by the clean pad
        f2 = Film()
        f2.frames = f.frames + procedure(settle=5).frames[1:200]
        _, t = play(f2)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(box(D_START), box(P_START), snap())  # everything jumps back in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        # A seeded reset (object_jitter, +-3 cm) puts things back near, not exactly on, their marks.
        t = self.breached()
        t.update(box(D_START + [0.02, -0.02, 0]), box(P_START + [-0.03, 0.01, 0]), snap())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur = t.prev
        f = Film(d=cur[0][0], p=cur[1][0], R=UP_R, L=READY_L)
        f.go(60, d=[cur[0][0][0], cur[0][0][1], CARRY]).go(60, d=[*D_START[:2], CARRY]).go(60, d=D_START + [0, 0, 0.0005])
        f.go(2, d=D_START).go(60, p=[*cur[1][0][:2], CARRY]).go(60, p=[*P_START[:2], CARRY]).go(60, p=P_START)
        f.hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(box((0.25, 0.0, L.TOP + H)), box(P_START + [0, 0.2, 0]), snap())  # knocked far, not to the start
        self.assertTrue(t.breach)

    def test_a_dropped_pad_is_not_a_teleport(self):
        # Free fall from 0.3 m onto the tray: fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = P_START + [0, 0, 0.3]
        t.update(box(D_START), box(p), snap())
        t._flag("right_near_pad")
        z, v = p[2], 0.0
        while z > P_START[2]:
            v += 9.81 * DT
            z = max(P_START[2], z - v * DT)
            t.update(box(D_START), box((p[0], p[1], z), vel=(0, 0, -v if z > P_START[2] else 0)), snap())
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


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its hooks, its rigstate reset hook, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def objects(self, d=D_START, p=P_START):
        return {"used_dressing": box(d), "clean_pad": box(p)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("used_dressing", names)
        self.assertIn("clean_pad", names)
        self.assertEqual(self.task.SCORE_OVER, "end")

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(right_tcp=P_START + [0, -0.05, 0.02])  # the dirty hand by the clean pad
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
        # the left arm brings the clean pad toward the mark only once the right gripper has let the dressing go
        right_release_end = sum(w.duration for w in plan["right"][:9])
        left_carry_start = sum(w.duration for w in plan["left"][:6])
        self.assertIn("let it drop", plan["right"][8].label)
        self.assertIn("carry it over", plan["left"][6].label)
        self.assertGreaterEqual(left_carry_start, right_release_end - 1e-9)
        for key, want in (("DEMO_IDLE_ARM", {"right"}), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
