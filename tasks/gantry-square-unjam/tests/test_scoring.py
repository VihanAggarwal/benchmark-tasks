"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic beam states and synthetic arm
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
    Tracker, end_centres, end_of, ends_held, finger_zone, lead, off_rails, point_beam_distance, popped, rise,
    rotation, signed_lead, tilt_deg, travel, valid, zone_gap, zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
H = L.BEAM_SIZE[2] / 2
Z0 = L.TOP + H  # the beam's centre height lying on the table
GRIP_Z = L.TOP + 0.018
OPEN, GRIP = 1.0, L.BEAM_SIZE[0] / L.FULL_GAP  # rigstate openings: fully open, closed on the 40 mm beam
GY = 0.200  # each arm grips 200 mm from the beam's centre
LX0 = L.BEAM_X0 - L.START_LEAD / 2  # 0.440: the left end's centre at the start
RX0 = L.BEAM_X0 + L.START_LEAD / 2  # 0.448: the right end's centre
T = L.TARGET_X
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
UP = L.TOP + 0.15


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


def state(center, quat, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like beam: centre, orientation, velocities, world bounding box."""
    c = np.array(center, float)
    ext = np.abs(rotation(quat)) @ (np.array(L.BEAM_SIZE, float) / 2)
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def beam(lx, rx, y=0.0, lift=0.0, extra=None, vel=(0, 0, 0), spin=(0, 0, 0)):
    """The beam lying flat between the rails with its left end's centre at x ``lx`` and its right end's at ``rx``
    (raised ``lift``, then turned by the quaternion ``extra`` about its centre, if given)."""
    yaw = math.asin(np.clip((rx - lx) / L.BEAM_LEN, -1, 1))
    q = about("z", math.degrees(yaw))
    if extra is not None:
        q = qmul(extra, q)
    return state(((lx + rx) / 2, y, Z0 + lift), q, vel, spin)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


KEYS = ("lx", "rx", "lift", "L", "Lo", "R", "Ro")
VEC = ("L", "R")


class Film:
    """A trial as a list of frames: both end positions (and a lift), both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; the beam's velocity comes from the frame-to-frame motion."""

    def __init__(self, **start):
        self.cur = {"lx": LX0, "rx": RX0, "lift": 0.0, "L": READY_L, "Lo": OPEN, "R": READY_R, "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in VEC else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in VEC else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            self.frames.append({k: (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
                                if k in VEC else (1 - f) * a[k] + f * b[k] for k in KEYS})
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def both(self, n, lx, rx, **more):
        """Move both ends, each gripper going with its own end."""
        return self.go(n, lx=lx, rx=rx, L=[lx, GY, GRIP_Z], R=[rx, -GY, GRIP_Z], **more)

    def __add__(self, other):
        out = Film()
        out.frames = self.frames + other.frames[1:]
        out.cur = dict(other.cur)
        return out


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    frames = film.frames[:until] if until else film.frames
    prev = None
    for fr in frames:
        b0 = beam(fr["lx"], fr["rx"], lift=fr["lift"])
        v = np.zeros(3) if prev is None else (b0.pos - prev) / DT
        b = beam(fr["lx"], fr["rx"], lift=fr["lift"], vel=v)
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(b, a)
        prev = b0.pos
    return r, t


def grip(f):
    """Both grippers down onto their ends and closed (about 3.5 s)."""
    f.go(30, L=[LX0, GY, GRIP_Z + 0.08], R=[RX0, -GY, GRIP_Z + 0.08]).go(20, L=[LX0, GY, GRIP_Z],
                                                                         R=[RX0, -GY, GRIP_Z])
    return f.go(20, Lo=GRIP, Ro=GRIP)


def square(f):
    """The right arm pulls the lagging right end level with the left one (2 s)."""
    return f.go(40, rx=LX0, R=[LX0, -GY, GRIP_Z])


def release(f, open_to=OPEN, up=UP):
    """Both grippers open and go straight up."""
    f.go(30, Lo=open_to, Ro=open_to)
    return f.go(30, L=[*f.cur["L"][:2], up], R=[*f.cur["R"][:2], up])


def procedure(lx=T, rx=None, open_to=OPEN, up=UP, settle=30):
    """The intended trial: grip both ends, square the beam, pull it to the marks with both ends together, let go,
    hands up; still for ``settle`` steps."""
    rx = lx if rx is None else rx
    f = grip(Film().hold(10))
    square(f).hold(10)
    f.both(180, lx, rx).hold(10)
    return release(f, open_to, up).hold(settle)


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        b = beam(LX0, RX0)
        self.assertAlmostEqual(signed_lead(b), 0.008, places=9)
        left, right = end_centres(b)
        self.assertAlmostEqual(left[0], LX0, places=9)
        self.assertAlmostEqual(right[0], RX0, places=9)
        self.assertGreater(left[1], 0)
        self.assertAlmostEqual(b.pos[0], L.BEAM_X0, places=9)
        self.assertAlmostEqual(math.atan2(rotation(b.quat_xyzw)[1, 0], rotation(b.quat_xyzw)[0, 0]), L.BEAM_YAW0)
        self.assertAlmostEqual(rise(b), 0.0, places=9)
        self.assertFalse(popped(b) or off_rails(b))
        # the yawed beam fits between the rails: its corners 0.22 mm from their inner faces
        self.assertLess(float(np.max(np.abs(b.high[1]))), L.RAIL_INNER_Y)

    def test_lead_is_signed_and_symmetric(self):
        self.assertAlmostEqual(signed_lead(beam(0.30, 0.31)), 0.010, places=9)
        self.assertAlmostEqual(signed_lead(beam(0.31, 0.30)), -0.010, places=9)
        self.assertAlmostEqual(lead(beam(0.31, 0.30)), 0.010, places=9)
        # turned over (about its length or across it), it is the same beam with the same lead
        for q in (about("y", 180), about("x", 180)):
            b = beam(0.30, 0.31, extra=q)  # turned over in the world: its ends swap round (mirrored yaw)
            self.assertAlmostEqual(lead(b), 0.010, places=9)
            self.assertAlmostEqual(signed_lead(b), -0.010, places=9)
            self.assertAlmostEqual(tilt_deg(b.quat_xyzw), 0.0, places=6)
            left, right = end_centres(b)
            self.assertAlmostEqual(left[0], 0.31, places=9)  # 'left' is whichever end is on the left
            self.assertGreater(left[1], right[1])

    def test_tilt_is_folded(self):
        self.assertAlmostEqual(tilt_deg(about("y", 4)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 176)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)

    def test_popped(self):
        self.assertFalse(popped(beam(0.3, 0.3, lift=0.004)))  # pushed up against the lips: not out
        self.assertFalse(popped(beam(0.3, 0.3, lift=0.0049)))
        self.assertTrue(popped(beam(0.3, 0.3, lift=0.0051)))  # past the lips
        self.assertFalse(popped(beam(0.3, 0.3, extra=about("y", 4.9))))
        self.assertTrue(popped(beam(0.3, 0.3, extra=about("y", 5.1))))  # rolled up onto an edge
        self.assertTrue(popped(beam(0.3, 0.3, lift=0.02, extra=about("x", 3))))  # one end lifted
        # one end raised 4 mm against its lip, the other on the table: the centre is 2 mm up, the lowest point is
        # not (rise counts the lowest corner)
        a = math.asin(0.004 / L.BEAM_LEN)
        b = beam(0.3, 0.3, lift=L.END_Y * math.sin(a) + H * (math.cos(a) - 1), extra=about("x", math.degrees(a)))
        self.assertAlmostEqual(rise(b), 0.0, places=9)
        self.assertAlmostEqual(b.pos[2] - Z0, 0.002, places=6)
        self.assertFalse(popped(b))
        # rolled 4.5 degrees onto its front bottom edge: its centre 1.5 mm up, its lowest point on the table
        e = math.radians(4.5)
        c = beam(0.3, 0.3, lift=L.BEAM_HALF[0] * math.sin(e) + H * (math.cos(e) - 1), extra=about("y", 4.5))
        self.assertAlmostEqual(rise(c), 0.0, places=9)
        self.assertGreater(c.pos[2] - Z0, 0.0015)
        self.assertFalse(popped(c))

    def test_off_rails(self):
        self.assertFalse(off_rails(beam(0.171, 0.171)))
        self.assertTrue(off_rails(beam(0.169, 0.172)))  # the left end past the rails' near end
        self.assertTrue(off_rails(beam(0.499, 0.501)))  # the right end past their far end
        self.assertFalse(off_rails(beam(0.3, 0.3, y=0.009)))
        self.assertTrue(off_rails(beam(0.3, 0.3, y=-0.011)))  # 11 mm sideways: out from between them

    def test_travel(self):
        self.assertAlmostEqual(travel(beam(LX0, RX0)), 0.0, places=9)
        self.assertAlmostEqual(travel(beam(T, T)), 1.0, places=9)

    def test_finger_zone_and_distance(self):
        b = beam(T, T)
        self.assertAlmostEqual(point_beam_distance((T, 0.1, Z0 + H + 0.05), b), 0.05, places=9)
        self.assertAlmostEqual(point_beam_distance((T, 0.0, Z0), b), 0.0, places=9)  # inside
        self.assertAlmostEqual(point_beam_distance((T, L.END_Y + 0.03, Z0), b), 0.03, places=9)  # past an end
        a = snap(left_tcp=(T, GY, UP))["left"]
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.008, 0.0135), places=6)  # 56.6 mm
        pts, r = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], UP - TIPS)
        # open, its tips 107 mm over the beam's top: the zone 50.4 mm clear
        self.assertAlmostEqual(zone_gap(a, b), (UP - TIPS - (L.TOP + 0.03)) - zone_radius(a), places=6)
        closed = snap(left_tcp=(T, GY, GRIP_Z), left_open=GRIP)["left"]
        self.assertLess(zone_gap(closed, b), 0.0)  # closed on the beam


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = play(Film().hold(40))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(t.squared)
        self.assertEqual(t.breaches, {})

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.breaches, {})
        self.assertTrue(t.squared_first)
        self.assertLess(t.squared_travel, 0.005)
        self.assertLessEqual(t.max_lead_after, L.SQUARE_LEAD)
        self.assertEqual(t.arms_unavailable, 0)
        self.assertTrue(r.details["hands_off"])

    def test_end_tolerances(self):
        r, _ = play(procedure(lx=T + 0.0049))
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(lx=T - 0.0049))
        self.assertEqual(r.score, 1.0)
        r, _ = play(procedure(lx=T + 0.006))  # 6 mm short of the marks
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(lx=T - 0.0099))
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(lx=T + 0.012))  # 12 mm short: the centre still within 15 mm
        self.assertEqual(r.score, 0.5)
        r, _ = play(procedure(lx=T - 0.0149))  # pulled 14.9 mm past
        self.assertEqual(r.score, 0.5)
        r, t = play(procedure(lx=T + 0.020))  # 20 mm short: moved 91 % of the way, square all along
        self.assertEqual(r.score, 0.25)
        self.assertEqual(t.breaches, {})

    def test_final_lead(self):
        r, _ = play(procedure(lx=T - 0.001, rx=T + 0.0019))  # 2.9 mm out of square at the end
        self.assertEqual(r.score, 0.75)
        r, _ = play(procedure(lx=T - 0.001, rx=T + 0.0009))  # 1.9 mm
        self.assertEqual(r.score, 1.0)

    def test_lead_during_the_move(self):
        def with_wobble(peak):
            f = grip(Film().hold(10))
            square(f).hold(10)
            mid = (LX0 + T) / 2
            f.both(90, mid, mid + peak).both(90, T, T).hold(10)  # the right end lagged ``peak`` mid-way
            return release(f).hold(30)

        r, t = play(with_wobble(0.0039))
        self.assertEqual(r.score, 1.0)
        r, t = play(with_wobble(0.005))
        self.assertEqual(r.score, 0.75)
        self.assertAlmostEqual(t.max_lead_after, 0.005, places=6)
        r, t = play(with_wobble(-0.0069))  # the right end led by 6.9 mm
        self.assertEqual(r.score, 0.75)
        r, t = play(with_wobble(0.009))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(t.breaches, {})

    def test_grippers_left_on_the_beam_score_075(self):
        r, _ = play(procedure(open_to=GRIP, up=GRIP_Z))  # still gripping both ends
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertFalse(r.details["hands_off"])
        r, _ = play(procedure(up=GRIP_Z + 0.04))  # opened but only 40 mm up: the open fingers still round it
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertLess(r.details["left_zone_mm"], 30)

    def test_hands_off_is_30_mm(self):
        # open and straight up over the beam: the zone (56.6 mm round the axis) clears the top by 30 mm once the
        # TCP is 129.6 mm over the table
        r, _ = play(procedure(up=L.TOP + 0.120))  # 20 mm clear
        self.assertEqual((r.score, r.success), (0.75, False))
        self.assertAlmostEqual(r.details["left_zone_mm"], 20.4, places=1)
        r, _ = play(procedure(up=L.TOP + 0.130))  # 30.4 mm clear
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_a_gripper_closed_well_clear_counts_as_let_go(self):
        f = procedure(settle=5)
        f.go(20, Lo=0.0, Ro=0.0).hold(25)  # both close again up in the air, 105 mm over the beam
        r, _ = play(f)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_squared_and_moved_part_way(self):
        f = grip(Film().hold(10))
        square(f).hold(10)
        x = L.BEAM_X0 - 0.31 * L.TRAVEL_SPAN
        r, _ = play(release(f.both(80, x, x)).hold(20))
        self.assertEqual(r.score, 0.25)
        f = grip(Film().hold(10))
        square(f).hold(10)
        x = L.BEAM_X0 - 0.28 * L.TRAVEL_SPAN
        r, _ = play(release(f.both(80, x, x)).hold(20))
        self.assertEqual(r.score, 0.0)

    def test_squared_only_scores_zero(self):
        r, t = play(release(square(grip(Film().hold(10)))).hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertTrue(t.squared)

    def test_never_squared_scores_zero(self):
        # pulled to the marks by both ends together but still 8 mm out of square: never squared
        f = grip(Film().hold(10))
        f.both(180, T - 0.004, T + 0.004).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.squared)
        self.assertEqual(t.breaches, {})
        # brought to within 2.5 mm of square and pulled so: not squared (the latch needs 2 mm)
        f = grip(Film().hold(10))
        f.go(40, rx=LX0 + 0.0025, R=[LX0 + 0.0025, -GY, GRIP_Z])
        f.both(180, T - 0.00125, T + 0.00125).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(t.squared)
        self.assertAlmostEqual(r.details["lead_mm"], 2.5, places=1)

    def test_squared_only_at_the_end_scores_half(self):
        # dragged out of square all the way, then straightened on the marks: not squared first
        f = grip(Film().hold(10))
        f.both(180, T - 0.004, T + 0.004).hold(10)
        f.both(30, T, T).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertTrue(t.squared)
        self.assertFalse(t.squared_first)

    def test_squared_within_the_first_25_mm_counts_as_first(self):
        f = grip(Film().hold(10))
        f.both(20, LX0 - 0.018, RX0 - 0.018)  # both ends pulled 18 mm out of square first
        f.both(30, LX0 - 0.018, LX0 - 0.018)  # then squared: the centre has moved 22 mm
        f.both(160, T, T).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 1.0)
        self.assertLess(t.squared_travel, L.SQUARE_BY)
        f = grip(Film().hold(10))
        f.both(30, LX0 - 0.024, RX0 - 0.024)
        f.both(30, LX0 - 0.024, LX0 - 0.024)  # 28 mm: too late
        f.both(160, T, T).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 0.5)

    def test_success_needs_stillness(self):
        f = procedure(settle=0)
        r, _ = play(f.go(2, lx=T + 0.002, rx=T + 0.002))  # still sliding (20 mm/s)
        self.assertEqual((r.score, r.success), (1.0, False))
        r, _ = play(f.hold(1))
        self.assertEqual((r.score, r.success), (1.0, True))
        tr = Tracker(HZ)
        play(procedure(settle=0), tr)
        a = snap(left_tcp=(T, GY, UP), right_tcp=(T, -GY, UP))
        r = tr.update(beam(T, T, vel=(0.02, 0, 0)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(beam(T, T, spin=(0, 0, 0.8)), a)
        self.assertEqual((r.score, r.success), (1.0, False))
        r = tr.update(beam(T, T), a)
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        a = snap(left_tcp=(T, GY, UP), right_tcp=(T, -GY, UP))
        self.assertEqual(t.result(beam(T, T), a).score, 1.0)
        self.assertEqual(t.result(beam(T, T), a).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))


class ZeroConditions(unittest.TestCase):
    """Each is permanent: once it happens the trial scores 0, whatever follows."""

    def assert_zero(self, f, key):
        r, t = play(f)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertIn(key, t.breaches)
        return t

    def test_pulling_the_leading_end_first_racks(self):
        f = grip(Film().hold(10))
        f.go(30, lx=LX0 - 0.0045, L=[LX0 - 0.0045, GY, GRIP_Z])  # the left end 4.5 mm on: 12.5 mm of lead
        f.go(30, lx=LX0, L=[LX0, GY, GRIP_Z])  # put back
        f2 = f + procedure()
        t = self.assert_zero(f2, "racked")
        self.assertGreater(t.max_lead, L.LEAD_MAX)

    def test_lead_limit_is_12_mm(self):
        f = grip(Film().hold(10))
        f.go(30, rx=LX0 + 0.0119).go(30, rx=LX0)  # 11.9 mm, then squared
        f.both(180, T, T).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 1.0)
        self.assertNotIn("racked", t.breaches)
        f = grip(Film().hold(10))
        f.go(30, rx=LX0 + 0.0121).go(30, rx=LX0)
        f.both(180, T, T).hold(10)
        self.assert_zero(release(f).hold(30), "racked")

    def test_one_arm_pull_racks(self):
        # the right end alone pulled on after squaring: the far end trails until it racks
        f = grip(Film().hold(10))
        square(f).hold(10)
        f.go(40, rx=LX0 - 0.016, lx=LX0 - 0.002, R=[LX0 - 0.016, -GY, GRIP_Z])
        self.assert_zero(f.hold(10), "racked")

    def test_popped(self):
        f = procedure(settle=0)
        f.go(2, lift=0.007).go(2, lift=0.0).hold(30)  # jumped out from under the lips and dropped back
        self.assert_zero(f, "popped")

    def test_tilted(self):
        f = grip(Film().hold(10))
        square(f).hold(10)
        tr = Tracker(HZ)
        play(f, tr)
        a = snap(left_tcp=(LX0, GY, GRIP_Z), left_open=GRIP, right_tcp=(LX0, -GY, GRIP_Z), right_open=GRIP)
        tr.update(beam(LX0, LX0, extra=about("y", 6)), a)  # rolled onto its front edge
        self.assertIn("popped", tr.breaches)
        r, _ = play(procedure(), tr)
        self.assertEqual(r.score, 0.0)

    def test_off_the_rails(self):
        f = procedure(settle=0)
        f.both(40, L.RAIL_X0 - 0.006, L.RAIL_X0 - 0.006).both(40, T, T)  # pulled out the front, pushed back
        f.go(30, L=[T, GY, UP], R=[T, -GY, UP]).hold(30)
        self.assert_zero(f, "off_rails")

    def test_a_breach_is_never_undone(self):
        f = grip(Film().hold(10))
        f.go(20, lx=LX0 - 0.006).go(20, lx=LX0)  # racked to 14 mm at once
        f2 = f + procedure(settle=60)
        r, t = play(f2)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(r.details["first_breach_step"], t.breaches["racked"])
        self.assertTrue(t.squared)


class Exploits(unittest.TestCase):
    def test_pull_one_end_at_a_time(self):
        # 'walking' the beam in 15 mm steps, one end at a time: racks at once
        f = grip(Film().hold(10))
        square(f)
        x = LX0
        for _ in range(3):
            f.go(20, rx=x - 0.015, R=[x - 0.015, -GY, GRIP_Z])
            f.go(20, lx=x - 0.030, L=[x - 0.030, GY, GRIP_Z])
            x -= 0.030
        r, t = play(f)
        self.assertEqual(r.score, 0.0)
        self.assertIn("racked", t.breaches)

    def test_walking_in_small_steps_is_scored_by_its_lead(self):
        # Each arm in turn moves its own end while the other holds: a two-arm way of keeping it square, scored
        # by the largest lead like any other: 5 mm steps 0.75, 3.5 mm steps 1.0.
        def walk(step):
            f = grip(Film().hold(10))
            square(f)
            x = LX0
            while x - 2 * step >= T - 1e-9:
                f.go(8, rx=x - step, R=[x - step, -GY, GRIP_Z])
                f.go(8, lx=x - 2 * step, L=[x - 2 * step, GY, GRIP_Z])
                f.go(8, rx=x - 2 * step, R=[x - 2 * step, -GY, GRIP_Z])
                x -= 2 * step
            f.both(20, T, T).hold(10)
            return play(release(f).hold(30))

        r, t = walk(0.005)
        self.assertEqual(t.breaches, {})
        self.assertAlmostEqual(t.max_lead_after, 0.005, places=6)
        self.assertEqual(r.score, 0.75)
        r, t = walk(0.0035)
        self.assertEqual(r.score, 1.0)

    def test_dragged_out_of_square_then_straightened(self):
        # the lagging end never read: pulled 8 mm out of square to the marks, squared there: 0.5, never more
        f = grip(Film().hold(10))
        f.both(180, T - 0.003, T + 0.005).hold(10)
        f.both(20, T + 0.001, T + 0.001).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(t.squared_first)

    def test_lifting_it_over_the_lips(self):
        f = grip(Film().hold(10))
        square(f)
        f.go(20, lift=0.03).both(60, T, T).go(20, lift=0.0)  # carried over instead of slid: through the lips
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 0.0)
        self.assertIn("popped", t.breaches)

    def test_pushed_back_square_then_pulled(self):
        # Squaring by pushing the leading left end back is squaring too (the scorer reads only the lead).
        f = grip(Film().hold(10))
        f.go(40, lx=RX0, L=[RX0, GY, GRIP_Z]).hold(10)
        f.both(180, T, T).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 1.0)
        self.assertLess(t.squared_travel, 0.0)


class BothEnds(unittest.TestCase):
    """'By both ends together': the centre may come at most 20 mm (net) toward the marks without a gripper at
    each end (finger-tip point within 10 mm of the beam, 130 mm or more from its middle toward that end)."""

    @staticmethod
    def middle_pull(arms=True):
        # Tilted 10 deg, the left arm reaches the beam's middle at the start: it twists the beam square about its
        # middle and pulls it straight to the marks while the right arm stays at its ready pose.
        f = Film().hold(10)
        f.go(30, L=[L.BEAM_X0, 0.0, GRIP_Z + 0.08]).go(20, L=[L.BEAM_X0, 0.0, GRIP_Z]).go(20, Lo=GRIP)
        f.go(40, lx=L.BEAM_X0, rx=L.BEAM_X0)  # turned square about its middle
        f.go(180, lx=T, rx=T, L=[T, 0.0, GRIP_Z]).hold(10)
        f.go(30, Lo=OPEN).go(30, L=[T, 0.0, UP]).hold(30)
        return f

    def test_one_arm_from_the_middle_scores_zero(self):
        r, t = play(self.middle_pull())
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(sorted(t.breaches), ["unheld"])  # square all the way, on the marks, let go: only this
        self.assertTrue(t.squared_first)
        self.assertAlmostEqual(t.unheld_travel, L.TRAVEL_SPAN, places=6)
        self.assertFalse(r.details["ends_held"])
        self.assertEqual(r.details["unheld_travel_mm"], 224.0)
        # it breaches the step the centre has come more than 20 mm
        _, t = play(self.middle_pull(), until=10 + 30 + 20 + 20 + 40 + 1 + 16)
        self.assertNotIn("unheld", t.breaches)
        _, t = play(self.middle_pull(), until=10 + 30 + 20 + 20 + 40 + 1 + 18)
        self.assertIn("unheld", t.breaches)

    def test_unreadable_arms_are_taken_as_holding(self):
        r, t = play(self.middle_pull(), arms=False)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.unheld_travel, 0.0)
        self.assertEqual(t.arms_unavailable, len(self.middle_pull().frames))

    def test_the_allowance_is_20_mm(self):
        def let_go_and_drag(d):
            # half-way, the left gripper opens and goes up, the right drags the beam on by d, the left grips again
            f = grip(Film().hold(10))
            square(f).hold(10)
            x = LX0 - 0.10
            f.both(80, x, x).go(20, Lo=OPEN).go(20, L=[x, GY, GRIP_Z + 0.05])
            f.go(20, lx=x - d, rx=x - d, R=[x - d, -GY, GRIP_Z])
            f.go(20, L=[x - d, GY, GRIP_Z]).go(20, Lo=GRIP)
            f.both(80, T, T).hold(10)
            return play(release(f).hold(30))

        r, t = let_go_and_drag(0.019)
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(t.unheld_travel, 0.019, places=6)
        r, t = let_go_and_drag(0.021)
        self.assertEqual(r.score, 0.0)
        self.assertIn("unheld", t.breaches)

    def test_resting_jitter_does_not_add_up(self):
        f = Film()
        for _ in range(300):  # the beam shivers 0.5 mm back and forth for 30 s with nobody touching it
            f.go(1, lx=LX0 - 0.0005, rx=RX0 - 0.0005).go(1, lx=LX0, rx=RX0)
        r, t = play(f + procedure())
        self.assertEqual(r.score, 1.0)
        self.assertLess(abs(t.unheld_travel), 1e-9)

    def test_squaring_with_one_arm_first_is_fine(self):
        # the right arm grips and squares its end while the left is still at its ready pose: the centre moves
        # 4 mm without a gripper at each end
        f = Film().hold(10)
        f.go(30, R=[RX0, -GY, GRIP_Z + 0.08]).go(20, R=[RX0, -GY, GRIP_Z]).go(20, Ro=GRIP)
        square(f).hold(10)
        f.go(30, L=[LX0, GY, GRIP_Z + 0.08]).go(20, L=[LX0, GY, GRIP_Z]).go(20, Lo=GRIP)
        f.both(180, T, T).hold(10)
        r, t = play(release(f).hold(30))
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(t.unheld_travel, L.START_LEAD / 2, places=6)

    def test_a_gripper_must_be_out_at_its_end(self):
        def pull_at(gy):
            f = Film().hold(10)
            f.go(30, L=[LX0, gy, GRIP_Z + 0.08], R=[RX0, -gy, GRIP_Z + 0.08])
            f.go(20, L=[LX0, gy, GRIP_Z], R=[RX0, -gy, GRIP_Z]).go(20, Lo=GRIP, Ro=GRIP)
            f.go(40, rx=LX0, R=[LX0, -gy, GRIP_Z]).hold(10)
            f.go(180, lx=T, rx=T, L=[T, gy, GRIP_Z], R=[T, -gy, GRIP_Z]).hold(10)
            f.go(30, Lo=OPEN, Ro=OPEN).go(30, L=[T, gy, UP], R=[T, -gy, UP]).hold(30)
            return play(f)

        r, t = pull_at(0.135)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.unheld_travel, 0.0)
        r, t = pull_at(0.125)  # 125 mm from the middle: not out at the ends
        self.assertEqual(r.score, 0.0)
        self.assertIn("unheld", t.breaches)

    def test_end_of(self):
        b = beam(T, T)
        top = L.TOP + L.BEAM_SIZE[2]
        for tips_z, want in ((top + 0.009, 1), (top + 0.011, 0)):
            a = rigstate.Arm(np.array([T, 0.2, tips_z + TIPS]), np.array([T, 0.2, tips_z]), 1.0)
            self.assertEqual(end_of(a, b), want)
        a = rigstate.Arm(np.array([T, -0.2, GRIP_Z]), np.array([T, -0.2, GRIP_Z - TIPS]), GRIP)
        self.assertEqual(end_of(a, b), -1)
        # turned over end for end, the beam's left end is still the one at +y
        flipped = beam(T, T, extra=about("x", 180))
        self.assertEqual(end_of(a, flipped), -1)
        # measured along the beam's own length: turned 30 degrees, a point 140 mm out along it is at an end
        turned = state((T, 0.0, Z0), about("z", 30))
        p = np.array([T, 0.0, L.TOP + 0.005]) + 0.140 * rotation(turned.quat_xyzw)[:, 1]
        self.assertEqual(end_of(rigstate.Arm(p + [0, 0, TIPS], p, GRIP), turned), 1)
        self.assertTrue(ends_held(snap(left_tcp=(T, GY, GRIP_Z), right_tcp=(T, -GY, GRIP_Z)), b))
        self.assertFalse(ends_held(snap(left_tcp=(T, GY, GRIP_Z), right_tcp=(T, 0.15, GRIP_Z)), b))


class Arms(unittest.TestCase):
    def test_arms_unavailable_taken_as_met_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        r, _ = play(procedure(open_to=GRIP, up=GRIP_Z), arms=False)  # even still gripping, it cannot be seen
        self.assertEqual(r.score, 1.0)

    def test_hands_off_applies_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(procedure(), t, arms=False)
        r = t.update(beam(T, T), snap(left_tcp=(T, GY, GRIP_Z), left_open=GRIP))
        self.assertEqual(r.score, 0.75)
        self.assertEqual(t.arms_unavailable, len(procedure().frames))

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY_R, READY_R - [0, 0, TIPS], float("nan"))}
        r = t.update(beam(LX0, RX0), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        t.update(beam(LX0, RX0), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        r = Tracker(HZ).update(beam(LX0, RX0), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_turned_over_beam_scores_the_same(self):
        f = procedure(settle=0)
        tr = Tracker(HZ)
        play(f, tr)
        a = snap(left_tcp=(T, GY, UP), right_tcp=(T, -GY, UP))
        for q in (about("y", 180), about("x", 180)):
            r = tr.update(beam(T, T, extra=q), a)
            self.assertEqual((r.score, r.success), (1.0, True))

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([T, 0.0, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]),
                              high=np.zeros(3))
        zero_q = beam(T, T)
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=np.array([T, 0.0, Z0]))
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (dict(t.breaches), t.squared_step, t.max_lead, t.max_lead_after, t.unheld_travel,
                      t.prev[0].tolist())
            a = snap(left_tcp=(T, GY, UP), right_tcp=(T, -GY, UP))
            for _ in range(3):
                r = t.update(bad, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(bad, a).score, 0.0)
            self.assertEqual((dict(t.breaches), t.squared_step, t.max_lead, t.max_lead_after, t.unheld_travel,
                              t.prev[0].tolist()), before)
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [state((5.0, -3.0, -1.0), about("z", 0)), state((0.3, 0.0, 0.0), about("x", 120)),
               state((-0.2, 0.6, L.TOP), about("y", 90)), state((0.3, 0.0, Z0), about("x", 89.9)),
               state((0.3, 0.0, 2.0), about("z", 30), vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        a = snap()
        for b in odd + odd[::-1]:
            r = t.update(b, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
        r = t.update(beam(LX0, RX0), None)
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def breached(self):
        f = grip(Film().hold(10))
        f.go(20, lx=LX0 - 0.006).go(40, lx=LX0 - 0.12, rx=RX0 - 0.12)  # racked, then dragged 12 cm on
        _, t = play(f)
        self.assertTrue(t.breaches)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(beam(LX0, RX0), snap())  # the beam jumps back in one step: a new episode
        self.assertEqual(t.breaches, {})
        self.assertEqual(t.steps, 1)
        self.assertFalse(t.squared)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        f.frames = f.frames[1:]
        r, _ = play(f, t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_reset_after_a_full_trial(self):
        # A finished trial (beam on the marks) put back by the simulator: the latch starts over too.
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        t.update(beam(LX0, RX0), snap())
        self.assertFalse(t.squared)
        self.assertEqual(t.result(beam(LX0, RX0), snap()).score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        t.update(beam(LX0 + 0.02, RX0 + 0.02, y=0.008), snap())  # a seeded reset's jitter
        self.assertEqual(t.breaches, {})

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        f = Film(lx=LX0 - 0.12, rx=RX0 - 0.12)
        f.go(120, lx=LX0, rx=RX0).hold(10)
        play(f, t)
        self.assertTrue(t.breaches)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        t.update(beam(0.25, 0.25), snap())  # knocked far, not to the start
        self.assertTrue(t.breaches)

    def test_a_drop_onto_the_start_is_not_a_teleport(self):
        # Free fall from 0.3 m straight onto its starting place: fast, but its velocity explains it.
        t = self.breached()
        z, v = 0.3, 0.0
        t.update(beam(LX0, RX0, lift=z), snap())
        while z > 0:
            v += 9.81 * DT
            z = max(0.0, z - v * DT)
            t.update(beam(LX0, RX0, lift=z, vel=(0, 0, -v if z > 0 else 0)), snap())
        self.assertTrue(t.breaches)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.breached()
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

    def objects(self, b=None):
        return {"beam": b if b is not None else beam(LX0, RX0)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertIn("beam", names)
        self.assertEqual([o.name for o in self.task.OBJECTS if o.mass is not None], ["beam"])
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertAlmostEqual(self.task.BEAM.yaw, L.BEAM_YAW0)
        # the spawned beam is the starting state the scorer expects
        self.assertTrue(Tracker.at_start(beam(LX0, RX0)))

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap()
        self.assertFalse(self.task.success(self.objects(beam(LX0 - 0.006, RX0))))  # racked
        self.assertTrue(tr.breaches)
        self.assertEqual(self.task.score(self.objects()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.breaches, {})
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)

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
        span = {a: sum(w.duration for w in plan[a]) for a in plan}
        self.assertAlmostEqual(span["left"], span["right"])
        # the two arms pull in step after squaring; the right arm, which trails more (GPU), a few mm further
        pull = {a: [w for w in plan[a] if w.label.startswith("pull toward")][0] for a in plan}
        self.assertAlmostEqual(pull["left"].pos[0] - self.task.RIGHT_PULL_BIAS, pull["right"].pos[0])
        self.assertLess(self.task.RIGHT_PULL_BIAS, 0.008)
        self.assertAlmostEqual(pull["left"].pos[1], -pull["right"].pos[1])
        t_pull = {a: sum(w.duration for w in plan[a][:plan[a].index(pull[a])]) for a in plan}
        self.assertAlmostEqual(t_pull["left"], t_pull["right"])
        # the grips start on the beam's centreline, turned with it
        self.assertAlmostEqual(plan["left"][0].pos[0], L.BEAM_X0 - 0.2 * math.tan(L.BEAM_YAW0))
        self.assertAlmostEqual(plan["right"][0].yaw, math.pi / 2 + L.BEAM_YAW0)
        # every close and open takes 1.5 s, and the last 22 mm of the descent too
        for a in plan:
            for prev, w in zip(plan[a], plan[a][1:]):
                if prev.finger != w.finger or (prev.pos[2] > w.pos[2] and w.pos[2] < L.TOP + 0.03):
                    self.assertGreaterEqual(w.duration, 1.5)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
