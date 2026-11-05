"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic sock states and synthetic arm
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
    Tracker, analyse_ring, angular_span, convex_hull, covered_deg, finger_clearance, finger_zone, let_go, outline,
    outline_gap, placed, point_outline_distance, rotation, spill_clearance, tilt_deg, valid, winding, zone_gap,
    zone_radius,
)

HZ = 20.0
DT = 1.0 / HZ
H = L.SOCK_H / 2  # half a sock's 30 mm height
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.006  # tips this far above a held sock's bottom
HELD = TIPS + GRIP_UP - H  # a held sock's centre is this far below the TCP (4 mm)
OPEN, PART, GRIP = 1.0, 0.050 / L.FULL_GAP, (L.SOCK_W - 0.008) / L.FULL_GAP  # rigstate openings
READY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
CARRY = L.TOP + 0.079  # TCP height while carrying (sock bottom 60 mm up)
PLACE = L.TOP + 0.002 + H + HELD  # TCP height as a sock is let go 2 mm above the table
UP = L.TOP + 0.100
PLACE_OF = {"sock_1": "left", "sock_2": "back", "sock_3": "right", "sock_4": "front"}
ARM_OF = {"sock_1": "left", "sock_2": "left", "sock_3": "right", "sock_4": "right"}
C = np.array([L.SPILL_X, L.SPILL_Y])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    """Quaternion product a * b (xyzw): turn by b, then by a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


Q0 = np.array([0.0, 0.0, 0.0, 1.0])
START = {n: np.array([x, y, L.TRAY_TOP_Z + H]) for n, (x, y) in L.SOCK_START.items()}
RING = {p: (np.array([x, y, L.TOP + H]), about("z", math.degrees(yaw))) for p, (x, y, yaw) in L.RING.items()}


def box(center, quat=Q0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """An ObjectState-like sock: centre, orientation, velocities, world bounding box."""
    c, half = np.array(center, float), np.array(L.SOCK_SIZE, float) / 2
    ext = np.abs(rotation(quat)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=np.array(quat, float), lin_vel=np.array(vel, float),
                           ang_vel=np.array(spin, float), low=c - ext, high=c + ext)


def at_ring(place, shift=(0, 0), quat=None, lift=0.0):
    """A sock lying in its ring place, shifted (x, y) and/or lifted (m)."""
    p, q = RING[place]
    return box(p + np.array([shift[0], shift[1], lift]), q if quat is None else quat)


def ring_socks(**shift):
    """All four socks in the demo's ring; ``sock_n=(dx, dy)`` moves one."""
    return {n: at_ring(PLACE_OF[n], shift.get(n, (0, 0))) for n in L.SOCKS}


def start_socks():
    return {n: box(START[n]) for n in L.SOCKS}


def snap(left_tcp=None, left_open=OPEN, right_tcp=None, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = READY[side] if tcp is None else np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


class Film:
    """A trial as a list of frames: the four socks' centres and orientations, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    def __init__(self, socks=None):
        socks = socks or {n: (START[n], Q0) for n in L.SOCKS}
        self.cur = {"pos": {n: np.array(p, float) for n, (p, _) in socks.items()},
                    "quat": {n: np.array(q, float) for n, (_, q) in socks.items()},
                    "left": READY["left"].copy(), "left_o": OPEN, "right": READY["right"].copy(), "right_o": OPEN}
        self.frames = [self._copy(self.cur)]

    @staticmethod
    def _copy(fr):
        return {"pos": dict(fr["pos"]), "quat": dict(fr["quat"]), "left": fr["left"], "left_o": fr["left_o"],
                "right": fr["right"], "right_o": fr["right_o"]}

    def go(self, n, pos=None, quat=None, **arms):
        a = self._copy(self.cur)
        b = self._copy(self.cur)
        b["pos"].update({k: np.array(v, float) for k, v in (pos or {}).items()})
        b["quat"].update({k: np.array(v, float) for k, v in (quat or {}).items()})
        for k, v in arms.items():
            b[k] = np.array(v, float) if k in ("left", "right") else float(v)
        for i in range(1, n + 1):
            f = i / n
            fr = self._copy(a)
            for k in L.SOCKS:
                fr["pos"][k] = (1 - f) * a["pos"][k] + f * b["pos"][k]
                q = (1 - f) * a["quat"][k] + f * (b["quat"][k] if a["quat"][k] @ b["quat"][k] >= 0 else -b["quat"][k])
                fr["quat"][k] = q / np.linalg.norm(q)
            for k in ("left", "right"):
                fr[k] = (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
                fr[k + "_o"] = (1 - f) * a[k + "_o"] + f * b[k + "_o"]
            self.frames.append(fr)
        self.cur = self._copy(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)


def play(film, tracker=None, arms=True, frames=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    prev = None
    for fr in (film.frames if frames is None else frames):
        socks = {}
        for n in L.SOCKS:
            v = np.zeros(3) if prev is None else (fr["pos"][n] - prev["pos"][n]) / DT
            socks[n] = box(fr["pos"][n], fr["quat"][n], v)
        a = snap(fr["left"], fr["left_o"], fr["right"], fr["right_o"]) if arms else None
        r = t.update(socks, a)
        prev = fr
    return r, t


def lay(f, sock, at_shift=(0, 0), open_to=PART, grip_off=L.GRIP_OFF, tcp_shift=(0, 0)):
    """One arm takes ``sock`` off its tray, carries it (turning it a quarter turn about the grip first if its place
    lies along y) and lays it in its ring place, shifted by ``at_shift``; lets go to ``open_to`` and goes up."""
    arm, place = ARM_OF[sock], PLACE_OF[sock]
    o = arm + "_o"
    gx, gy = L.RING_GRIP[place]
    s0 = START[sock]
    g0 = np.array([gx, s0[1], s0[2] + HELD])
    f.go(20, **{arm: g0 + [0, 0, 0.05], o: PART}).go(16, **{arm: g0}).go(20, **{o: GRIP})
    lift = CARRY - g0[2]
    f.go(16, pos={sock: s0 + [0, 0, lift]}, **{arm: g0 + [0, 0, lift]})
    target, q_end = RING[place]
    target = target + np.array([*at_shift, 0.0])
    if abs(L.RING[place][2]) > 0.1:  # turn a quarter turn about the grip, in 10 small steps
        sign = -1.0 if arm == "left" else 1.0
        off = (s0 - g0)[:2]
        for k in range(1, 11):
            a = math.radians(9 * k) * sign
            rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            c = g0[:2] + rot @ off
            f.go(2, pos={sock: [c[0], c[1], CARRY - HELD]}, quat={sock: about("z", math.degrees(a))})
        q_end = about("z", 90 * sign)
    tcp = np.array([target[0] + (gx - RING[place][0][0]) + tcp_shift[0],
                    target[1] + (gy - RING[place][0][1]) + tcp_shift[1]])
    f.go(30, pos={sock: [target[0], target[1], CARRY - HELD]}, quat={sock: q_end}, **{arm: [*tcp, CARRY]})
    f.go(30, pos={sock: [target[0], target[1], PLACE - HELD]}, **{arm: [*tcp, PLACE]})
    f.go(6, **{o: (GRIP + PART) / 2}).go(2, pos={sock: target}).go(14, **{o: open_to})
    f.go(20, **{arm: [*tcp, UP]})
    return f


def procedure(settle=30, open_to=PART, **shifts):
    """The intended trial: the left arm lays sock_1 (left side) and sock_2 (back), the right arm sock_3 (right side)
    and sock_4 (front); ``sock_n=(dx, dy)`` lays that sock off its ring place. All still for ``settle`` steps."""
    f = Film().hold(10)
    for sock in ("sock_1", "sock_3", "sock_2", "sock_4"):
        lay(f, sock, shifts.get(sock, (0, 0)), open_to)
    return f.hold(settle)


class Geometry(unittest.TestCase):
    def test_tilt_is_folded_for_a_square_bar(self):
        self.assertAlmostEqual(tilt_deg(about("x", 90)), 0.0, places=4)  # rolled onto another long face
        self.assertAlmostEqual(tilt_deg(about("x", 180)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("x", 4)), 4.0, places=4)  # rolled 4 degrees: on an edge, not flat
        self.assertAlmostEqual(tilt_deg(about("x", 86)), 4.0, places=4)
        self.assertAlmostEqual(tilt_deg(about("y", 4)), 4.0, places=4)  # one end up 4 degrees
        self.assertAlmostEqual(tilt_deg(about("y", 90)), 90.0, places=4)  # standing on end
        self.assertAlmostEqual(tilt_deg(about("z", 40)), 0.0, places=4)  # yaw is not tilt
        self.assertAlmostEqual(tilt_deg(qmul(about("z", 90), about("y", 3))), 3.0, places=4)

    def test_outline_follows_the_real_turned_sock(self):
        flat = outline(box((0.3, 0.2, L.TOP + H)))
        self.assertEqual(len(flat), 4)
        self.assertTrue(np.allclose(flat.min(axis=0), (0.2, 0.185)))
        turned = outline(box((0.3, 0.2, L.TOP + H), about("z", 45)))
        self.assertEqual(len(turned), 4)
        area = 0.5 * abs(sum(turned[i - 1][0] * turned[i][1] - turned[i][0] * turned[i - 1][1] for i in range(4)))
        self.assertAlmostEqual(area, L.SOCK_LEN * L.SOCK_W, places=8)  # not its far larger bounding box
        tilted = outline(box((0.3, 0.2, L.TOP + 0.05), about("y", 30)))
        self.assertGreater(len(tilted), 3)
        self.assertAlmostEqual(np.ptp(tilted[:, 0]), 0.2 * math.cos(math.radians(30)) + 0.03 * 0.5, places=6)

    def test_distances(self):
        sq = convex_hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5), (1, 1)])
        self.assertEqual(len(sq), 4)
        self.assertEqual(point_outline_distance(sq, (0.5, 0.2))[0], 0.0)
        self.assertAlmostEqual(point_outline_distance(sq, (2, 0.5))[0], 1.0)
        self.assertAlmostEqual(point_outline_distance(sq, (2, 2))[0], math.sqrt(2))
        sq2 = sq + np.array([1.5, 0.25])
        d, pa, pb = outline_gap(sq, sq2)
        self.assertAlmostEqual(d, 0.5)
        self.assertAlmostEqual(np.linalg.norm(pa - pb), 0.5)
        self.assertEqual(outline_gap(sq, sq + [0.5, 0.5])[0], 0.0)  # overlapping
        cross = convex_hull([(0.4, -1), (0.6, -1), (0.6, 2), (0.4, 2)])  # crosses the square, no vertex inside
        d, pa, pb = outline_gap(sq, cross)
        self.assertEqual(d, 0.0)
        self.assertTrue(np.allclose(pa, pb))

    def test_demo_ring_geometry(self):
        socks = ring_socks()
        for n, s in socks.items():
            self.assertAlmostEqual(spill_clearance(s), L.RING_A, places=6)  # inner face 87.5 mm from the centre
            self.assertTrue(placed(s))
        ring = analyse_ring({n: outline(s) for n, s in socks.items()})
        self.assertTrue(ring.tight)
        self.assertEqual(ring.joints_mm, [5.0, 5.0, 5.0, 5.0])
        self.assertEqual(ring.group, 4)
        self.assertAlmostEqual(ring.span_deg, 360.0, places=3)
        three = analyse_ring({n: outline(socks[n]) for n in ("sock_1", "sock_2", "sock_3")})
        self.assertFalse(three.one_loose)
        self.assertGreater(three.span_deg, 270.0)  # three sides: about 277 degrees
        two = analyse_ring({n: outline(socks[n]) for n in ("sock_1", "sock_2")})
        self.assertLess(covered_deg([angular_span(outline(socks[n])) for n in ("sock_1", "sock_2")]), 190.0)
        self.assertEqual(two.span_deg, 0.0)  # only groups of three or more count

    def test_winding(self):
        sq = [C + d for d in ((0.1, 0.1), (-0.1, 0.1), (-0.1, -0.1), (0.1, -0.1))]
        self.assertAlmostEqual(winding(sq), 1.0)
        self.assertAlmostEqual(winding(sq[::-1]), -1.0)
        self.assertAlmostEqual(winding([p + (0.3, 0.0) for p in sq]), 0.0)

    def test_finger_zone(self):
        a = snap(left_tcp=(0.2, 0.2, 0.9), left_open=OPEN)["left"]
        self.assertAlmostEqual(zone_radius(a), math.hypot(L.FULL_GAP / 2 + 0.010, 0.020), places=6)  # 60.4 mm
        pts, r = finger_zone(a)
        self.assertAlmostEqual(pts[0][2], 0.9 - TIPS)
        self.assertAlmostEqual(pts[-1][2], 0.9 - TIPS + L.FINGER_LEN)
        # at a ring grip point (117.5 mm from the spill's centre): 50 mm open keeps 12 mm out, fully open is in
        gx, gy = L.RING_GRIP["left"]
        part = snap(left_tcp=(gx, gy, PLACE), left_open=PART)["left"]
        self.assertAlmostEqual(finger_clearance(part), 0.1175 - math.hypot(0.025 + 0.010, 0.020), places=4)
        self.assertGreater(finger_clearance(part), L.ZONE_R + 0.012)
        full = snap(left_tcp=(gx, gy, PLACE), left_open=OPEN)["left"]
        self.assertLess(finger_clearance(full), L.ZONE_R)

    def test_let_go(self):
        gx, gy = L.RING_GRIP["left"]
        sock = at_ring("left")
        self.assertFalse(let_go(snap(left_tcp=(gx, gy, PLACE), left_open=GRIP)["left"], [sock]))
        self.assertFalse(let_go(snap(left_tcp=(gx, gy, PLACE), left_open=PART)["left"], [sock]))  # still round it
        up = snap(left_tcp=(gx, gy, UP), left_open=PART)["left"]
        self.assertGreater(zone_gap(up, sock), L.LET_GO_GAP + 0.006)  # 17 mm clear: let go
        self.assertTrue(let_go(up, [sock]))
        self.assertTrue(let_go(snap(left_tcp=(gx, gy, PLACE), left_open=0.95)["left"], [sock]))  # open
        self.assertTrue(let_go(snap(left_tcp=(gx, gy + 0.2, PLACE), left_open=0.0)["left"], [sock]))  # closed, away

    def test_placed(self):
        self.assertTrue(placed(at_ring("left")))
        self.assertFalse(placed(box(START["sock_1"])))  # on its tray, 8 mm up
        self.assertFalse(placed(at_ring("left", lift=0.004)))  # held 4 mm above the table
        self.assertTrue(placed(at_ring("left", lift=0.0025)))
        self.assertFalse(placed(at_ring("left", lift=-0.005)))  # sunk 5 mm into the table (a glitch): not lying on it
        self.assertFalse(placed(at_ring("left", (0, 0.034))))  # inner face 121.5 mm out: over 60 mm from the edge
        self.assertTrue(placed(at_ring("left", (0, 0.032))))
        self.assertTrue(placed(at_ring("left", quat=about("x", 90))))  # rolled onto another long face
        rolled = box(RING["left"][0] + [0, 0, 0.006], about("x", 45))  # balanced on an edge
        self.assertFalse(placed(rolled))
        p, q = RING["left"]
        tipped = box(p, qmul(about("y", 6), q))
        tipped.low[2] = L.TOP  # one end on the table, the other propped up: 6 degrees
        self.assertFalse(placed(tipped))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        t = Tracker(HZ)
        r = t.update(start_socks(), snap())
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(r.details["n_placed"], 0)

    def test_full_procedure_scores_one(self):
        r, t = play(procedure())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertFalse(t.breach)
        self.assertEqual(r.details["ring_joints_mm"], [5.0, 5.0, 5.0, 5.0])
        self.assertGreater(t.min_sock_clear, L.ZONE_R + 0.020)
        self.assertGreater(min(t.min_finger_clear.values()), L.ZONE_R + 0.012)

    def test_success_needs_stillness(self):
        f = procedure()
        last = f.frames[-1]
        creep = Film()
        creep.frames, creep.cur = [last], Film._copy(last)
        creep.go(5, pos={"sock_1": last["pos"]["sock_1"] + [0.003, 0, 0]})  # 12 mm/s along its length
        t = Tracker(HZ)
        r, _ = play(f, t)
        self.assertTrue(r.success)
        r, _ = play(creep, t, frames=creep.frames[1:])
        self.assertEqual(r.score, 1.0)  # still a closed ring (that joint now 8 mm) ...
        self.assertFalse(r.success)  # ... but not still
        self.assertNotIn("sock_1", r.details["resting"])
        r, _ = play(creep.hold(5), t, frames=creep.frames[-5:])
        self.assertTrue(r.success)

    def test_one_joint_loose_scores_075(self):
        r, _ = play(procedure(sock_4=(0.0, -0.012)))  # front sock 12 mm along: its end 17 mm from the left sock
        self.assertEqual(r.score, 0.75)
        self.assertEqual(sorted(r.details["ring_joints_mm"]), [5.0, 5.0, 5.0, 17.0])
        self.assertTrue(r.details["ring_one_loose_joint"])

    def test_joint_just_inside_10mm_scores_one(self):
        r, _ = play(procedure(sock_4=(0.0, -0.0045)))  # 9.5 mm
        self.assertEqual(r.score, 1.0)

    def test_two_loose_joints_score_half(self):
        r, _ = play(procedure(sock_4=(0.012, -0.012)))  # out and along: 17 mm at both its joints
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["group_size"], 4)

    def test_a_joint_over_25mm_scores_half(self):
        r, _ = play(procedure(sock_4=(0.0, -0.030)))  # 35 mm gap: an open ring, three sides and more in place
        self.assertEqual(r.score, 0.5)
        self.assertGreater(r.details["group_span_deg"], 300.0)

    def test_gripper_still_holding_the_last_sock_scores_075(self):
        f = Film().hold(10)
        for sock in ("sock_1", "sock_3", "sock_2"):
            lay(f, sock)
        lay(f, "sock_4", open_to=GRIP)  # the right gripper closes again round it (or never opens)
        f.frames = f.frames[:-20]  # ... and stays down there
        f.cur = Film._copy(f.frames[-1])
        f.hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["grippers_let_go"])

    def test_a_gripper_closed_away_from_the_socks_has_let_go(self):
        f = procedure(settle=5)
        f.go(10, right_o=0.0).go(20, right=[0.30, -0.25, L.TOP + 0.15]).hold(25)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_three_sides_score_half(self):
        f = Film().hold(10)
        for sock in ("sock_1", "sock_3", "sock_2"):
            lay(f, sock)
        r, _ = play(f.hold(30))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(r.details["n_placed"], 3)
        self.assertGreater(r.details["group_span_deg"], L.CHAIN_SPAN_DEG)

    def test_three_socks_that_do_not_wrap_far_enough_score_quarter(self):
        # Left side and back laid, the third sock doubled up outside the left side: a group of three, but they
        # cover only about 187 degrees round the spill, not three sides.
        socks = ring_socks()
        socks["sock_3"] = at_ring("left", (0.0, L.SOCK_W))
        socks["sock_4"] = box(START["sock_4"])
        t = Tracker(HZ)
        r = t.update(socks, snap())
        self.assertEqual(r.details["n_placed"], 3)
        self.assertEqual(r.details["group_size"], 3)
        self.assertLess(r.details["group_span_deg"], 200.0)
        self.assertEqual(r.score, 0.25)

    def test_one_arm_alone_scores_quarter(self):
        for socks in (("sock_1", "sock_2"), ("sock_3", "sock_4")):
            f = Film().hold(10)
            for sock in socks:
                lay(f, sock)
            r, _ = play(f.hold(30))
            self.assertEqual(r.score, 0.25)
            self.assertEqual(r.details["n_placed"], 2)

    def test_one_sock_scores_zero(self):
        r, _ = play(lay(Film().hold(10), "sock_1").hold(30))
        self.assertEqual(r.score, 0.0)

    def test_a_sock_too_far_out_is_not_placed(self):
        r, _ = play(procedure(sock_1=(0.0, 0.045)))  # left side 72.5 mm from the spill's edge
        self.assertEqual(r.details["n_placed"], 3)
        self.assertNotIn("sock_1", r.details["placed"])
        self.assertEqual(r.score, 0.5)  # the other three still wrap three sides

    def test_score_is_idempotent(self):
        r, t = play(procedure())
        socks = ring_socks()
        a = snap(right_tcp=(*L.RING_GRIP["front"], UP), right_open=PART,
                 left_tcp=(*L.RING_GRIP["back"], UP), left_open=PART)
        for _ in range(3):
            self.assertEqual(t.result(socks, a).score, 1.0)
        self.assertEqual(t.steps, len(procedure().frames))


class Breaches(unittest.TestCase):
    def test_sock_inner_face_inside_the_keep_out(self):
        t = Tracker(HZ)
        t.update(start_socks(), snap())
        socks = start_socks()
        socks["sock_1"] = box((L.RING["left"][0], L.SPILL_Y + L.ZONE_R + H - 0.001, L.TOP + H))  # inner face 64 mm
        r = t.update(socks, snap())
        self.assertEqual(r.score, 0.0)
        self.assertIn("sock_1_entered", t.breaches)

    def test_sock_just_outside_the_keep_out_is_fine(self):
        t = Tracker(HZ)
        socks = start_socks()
        socks["sock_1"] = box((L.RING["left"][0], L.SPILL_Y + L.ZONE_R + H + 0.001, L.TOP + H))  # 66 mm
        t.update(socks, snap())
        self.assertFalse(t.breach)

    def test_sock_carried_over_the_spill_high_up(self):
        f = Film().hold(5)
        s0 = START["sock_1"]
        f.go(20, pos={"sock_1": s0 + [0, 0, 0.25]}).go(30, pos={"sock_1": [s0[0], -0.05, s0[2] + 0.25]})
        r, t = play(f)
        self.assertIn("sock_1_entered", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_corner_of_a_turned_sock_counts(self):
        # Carried turned 45 degrees with its centre 130 mm out: its bounding box reaches to 49 mm from the spill's
        # centre, the sock itself only to 77 mm: no breach. Swung in closer, its real side crosses the line.
        t = Tracker(HZ)
        socks = start_socks()
        socks["sock_1"] = box((L.SPILL_X, L.SPILL_Y + 0.130, L.TOP + 0.05), about("z", 45))
        lo, hi = socks["sock_1"].low[:2], socks["sock_1"].high[:2]
        bbox_clear = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - C, C - hi))))
        self.assertAlmostEqual(bbox_clear, 0.130 - 0.115 * math.sqrt(0.5), places=6)  # 48.7 mm: would breach
        self.assertAlmostEqual(spill_clearance(socks["sock_1"]), 0.130 * math.sqrt(0.5) - 0.015, places=6)
        t.update(socks, snap())
        self.assertFalse(t.breach)
        socks["sock_1"] = box((L.SPILL_X + 0.03, L.SPILL_Y + 0.07, L.TOP + 0.05), about("z", 30))
        t.update(socks, snap())
        self.assertEqual(sorted(t.breaches), ["sock_1_entered"])

    def test_fingers_fully_open_at_the_ring(self):
        t = Tracker(HZ)
        t.update(ring_socks(), snap(left_tcp=(*L.RING_GRIP["left"], UP), left_open=PART))
        self.assertFalse(t.breach)
        t.update(ring_socks(), snap(left_tcp=(*L.RING_GRIP["left"], UP), left_open=OPEN))
        self.assertIn("left_fingers_entered", t.breaches)

    def test_fingers_over_the_spill_high_up(self):
        t = Tracker(HZ)
        r = t.update(start_socks(), snap(right_tcp=(L.SPILL_X + 0.02, -0.03, L.TOP + 0.3), right_open=0.0))
        self.assertIn("right_fingers_entered", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_laying_a_sock_with_the_gripper_beside_the_spill(self):
        # Gripping the left sock level with the spill's centre instead of toward its corner end: at 102.5 mm out,
        # fingers opened to 50 mm reach to 62.2 mm from the centre: inside the 65 mm circle.
        f = Film().hold(5)
        lay(f, "sock_1", tcp_shift=(-L.GRIP_OFF - L._ALONG, 0.0))
        r, t = play(f)
        self.assertIn("left_fingers_entered", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertLess(t.min_finger_clear["left"], 0.1025 - 0.040 + 0.0005)

    def test_a_breach_is_never_undone(self):
        f = procedure(settle=5)
        bad = Film()
        bad.frames = [f.frames[0]]
        bad.cur = Film._copy(f.frames[0])
        bad.go(10, right=[L.SPILL_X, L.SPILL_Y, L.TOP + 0.2]).go(10, right=READY["right"])  # a sweep over it
        g = Film()
        g.frames = bad.frames + procedure().frames[1:]
        r, t = play(g)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertIn("right_fingers_entered", t.breaches)
        self.assertTrue(r.details["ring_closed"])  # the ring is perfect, the breach stands


class Exploits(unittest.TestCase):
    def test_one_arm_reaching_across_the_spill_scores_zero(self):
        # The left arm fetches the right side's sock from the middle of the table by reaching straight over the
        # spill (as if it could take it off the right tray).
        f = Film().hold(5)
        f.go(30, left=[L.SPILL_X, 0.12, CARRY], left_o=PART)
        f.go(30, left=[L.SPILL_X, -0.12, CARRY])  # straight over the spill
        r, t = play(f)
        self.assertIn("left_fingers_entered", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_sliding_a_sock_across_the_spill_scores_zero(self):
        f = Film().hold(5)
        f.go(30, pos={"sock_1": [L.RING["left"][0], 0.10, L.TOP + H]})
        f.go(30, pos={"sock_1": [L.RING["left"][0], -0.10, L.TOP + H]})  # pushed over the spill to the other side
        r, t = play(f)
        self.assertIn("sock_1_entered", t.breaches)
        self.assertEqual(r.score, 0.0)

    def test_dropping_a_sock_on_the_spill_then_moving_it_out_scores_zero(self):
        f = procedure(settle=5)
        last = f.frames[-1]
        g = Film()
        g.frames, g.cur = [last], Film._copy(last)
        g.go(10, pos={"sock_4": [L.SPILL_X, L.SPILL_Y, L.TOP + L.SPILL_H + H]})
        g.go(10, pos={"sock_4": RING["front"][0]}).hold(30)
        t = Tracker(HZ)
        play(f, t)
        r, _ = play(g, t, frames=g.frames[1:])
        self.assertEqual(r.score, 0.0)
        self.assertIn("sock_4_entered", t.breaches)

    def test_socks_in_a_straight_row_do_not_wrap(self):
        # Four socks end to end in a row along the spill's left side, as close in as allowed: a wall, not a ring
        # (only the middle two are within 60 mm of the edge).
        socks = {n: box((L.SPILL_X - 0.3075 + 0.205 * i, L.SPILL_Y + 0.085, L.TOP + H)) for i, n in enumerate(L.SOCKS)}
        t = Tracker(HZ)
        r = t.update(socks, snap())
        self.assertFalse(t.breach)
        self.assertEqual(r.details["n_placed"], 2)
        self.assertEqual(r.score, 0.25)

    def test_two_pairs_side_by_side_do_not_wrap(self):
        # All four placed within 60 mm of the edge, but as two doubled walls left and right: no ring.
        y = {"sock_1": 0.1025, "sock_2": 0.1325, "sock_3": -0.1025, "sock_4": -0.1325}
        socks = {n: box((L.SPILL_X, y[n], L.TOP + H)) for n in L.SOCKS}
        t = Tracker(HZ)
        r = t.update(socks, snap())
        self.assertEqual(r.details["n_placed"], 4)
        self.assertEqual(r.details["group_size"], 2)
        self.assertFalse(r.details["ring_closed"])
        self.assertEqual(r.score, 0.25)

    def test_socks_stacked_or_on_end_are_not_placed(self):
        socks = ring_socks()
        socks["sock_4"] = at_ring("front", lift=L.SOCK_H)  # lying on top of the others' ends instead of the table
        socks["sock_2"] = box((RING["back"][0][0], RING["back"][0][1], L.TOP + L.SOCK_LEN / 2), about("y", 90))
        t = Tracker(HZ)
        r = t.update(socks, snap())
        self.assertEqual(r.details["n_placed"], 2)
        self.assertEqual(r.score, 0.25)

    def test_overlapping_ends_tilt_a_sock(self):
        # The last sock laid on top of a neighbour's end: tilted about 9 degrees, it is not flat, so not placed.
        socks = ring_socks(sock_4=(0.0, 0.010))
        p, q = RING["front"]
        socks["sock_4"] = box(p + [0, 0.010, 0.015], qmul(about("x", -9), q))
        t = Tracker(HZ)
        r = t.update(socks, snap())
        self.assertNotIn("sock_4", r.details["placed"])
        self.assertEqual(r.score, 0.5)

    def test_three_socks_closing_a_ring_with_the_fourth_elsewhere_score_half(self):
        # Three socks can just close a ring round the spill (joints 9, 9 and 24 mm, each about 66 mm from its
        # centre); with the fourth placed outside it, all four are placed but they do not together close a ring.
        p = [(0.2971, 0.0790, 2.9223), (0.2017, -0.0197, 4.9873), (0.3360, -0.0584, 7.0236)]
        socks = {n: box((x, y, L.TOP + H), about("z", math.degrees(yaw))) for n, (x, y, yaw) in zip(L.SOCKS, p)}
        three = analyse_ring({n: outline(s) for n, s in socks.items()})
        self.assertTrue(three.one_loose)  # the three alone: a ring with one loose joint
        socks["sock_4"] = box((L.SPILL_X + 0.099, 0.099, L.TOP + H), about("z", 165))
        t = Tracker(HZ)
        r = t.update(socks, snap())
        self.assertFalse(t.breach)
        self.assertEqual(r.details["n_placed"], 4)
        self.assertFalse(r.details["ring_closed"] or r.details["ring_one_loose_joint"])
        self.assertEqual(r.score, 0.5)  # three or more linked socks wrapping the spill

    def test_two_corners_apart_are_not_three_quarters(self):
        # All four in their ring places but the right side and the back moved 26 mm out together: two L-shaped
        # pairs, each joined at its corner, with two joints of more than 25 mm between them. Not one group of
        # three or more, so 0.25 (moved 19 mm out, with 24 mm joints, it is one group of four: 0.5).
        far = {n: (-0.026, -0.026) for n in ("sock_2", "sock_3")}
        r, _ = play(procedure(**far))
        self.assertEqual(r.details["n_placed"], 4)
        self.assertEqual(r.details["group_size"], 2)
        self.assertEqual(r.score, 0.25)
        near = {n: (-0.019, -0.019) for n in ("sock_2", "sock_3")}
        r, _ = play(procedure(**near))
        self.assertEqual(r.details["group_size"], 4)
        self.assertEqual(r.score, 0.5)

    def test_a_closed_ring_round_something_else_is_not_a_ring(self):
        # The same closed pinwheel moved 20 cm to the left: its loop does not go round the spill.
        moved = {n: outline(box(s.pos + [0, 0.20, 0], s.quat_xyzw)) for n, s in ring_socks().items()}
        ring = analyse_ring(moved)
        self.assertFalse(ring.tight or ring.one_loose)
        self.assertEqual(ring.group, 4)
        self.assertLess(ring.span_deg, 180.0)


class Arms(unittest.TestCase):
    def test_arms_unavailable_rules_kept_and_counted(self):
        r, t = play(procedure(), arms=False)
        self.assertEqual(r.score, 1.0)  # no gripper reading: 'both let go' taken as met
        self.assertEqual(t.arms_unavailable, len(procedure().frames))
        self.assertIsNone(r.details["opening"])
        r, _ = play(procedure(open_to=GRIP), arms=False)  # even a closed gripper cannot be seen
        self.assertEqual(r.score, 1.0)

    def test_arm_rules_resume_when_the_arms_come_back(self):
        t = Tracker(HZ)
        play(Film().hold(5), t, arms=False)
        r = t.update(start_socks(), snap(left_tcp=(L.SPILL_X, 0.05, L.TOP + 0.1), left_open=PART))
        self.assertIn("left_fingers_entered", t.breaches)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.arms_unavailable, 6)

    def test_non_finite_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": rigstate.Arm(np.full(3, np.nan), np.full(3, np.nan), 1.0),
               "right": rigstate.Arm(READY["right"], READY["right"] - [0, 0, TIPS], float("nan"))}
        r = t.update(start_socks(), bad)
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.0)
        r = t.update(start_socks(), {"left": bad["left"]})  # one arm missing
        self.assertEqual(t.arms_unavailable, 2)
        self.assertFalse(t.breach)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        self.assertAlmostEqual(a["left"].opening, 0.25)
        self.assertTrue(np.allclose(a["right"].tips, (0.1, -0.2, 0.9 - TIPS)))
        t = Tracker(HZ)
        r = t.update(start_socks(), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_any_long_face_down(self):
        for deg in (90, 180, 270):
            socks = {n: at_ring(PLACE_OF[n], quat=qmul(RING[PLACE_OF[n]][1], about("x", deg))) for n in L.SOCKS}
            t = Tracker(HZ)
            r = t.update(socks, snap())
            self.assertEqual(r.score, 1.0, deg)

    def test_a_few_mm_of_sag_is_tolerated(self):
        # Every sock 5 mm off the same way (both arms sagging alike): the joints are unchanged, the socks still
        # 82 mm from the spill's centre.
        r, _ = play(procedure(**{n: (0.004, -0.003) for n in L.SOCKS}))
        self.assertEqual(r.score, 1.0)
        # Each sock 2 mm off on its own, every joint opened by up to 4 mm (9 mm): still closed.
        r, _ = play(procedure(sock_1=(0.002, 0.002), sock_2=(-0.002, -0.002), sock_3=(-0.002, -0.002),
                              sock_4=(0.002, 0.002)))
        self.assertEqual(r.score, 1.0)
        self.assertLessEqual(max(r.details["ring_joints_mm"]), 9.0)

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([0.3, 0.0, np.inf]), quat_xyzw=Q0, lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]), high=np.zeros(3))
        zero_q = at_ring("front")
        zero_q.quat_xyzw = np.zeros(4)
        missing = SimpleNamespace(pos=RING["front"][0])
        frames = procedure().frames
        for bad in (nan, inf, zero_q, missing, None):
            self.assertFalse(valid(bad))
            _, t = play(Film(), frames=frames[:-25])
            before = (dict(t.breaches), t.min_sock_clear, dict(t.min_finger_clear), t.arms_unavailable)
            a = snap(left_tcp=(*L.RING_GRIP["back"], UP), left_open=PART)
            for socks in ({**ring_socks(), "sock_4": bad}, {**ring_socks(), "sock_1": bad},
                          {n: bad for n in L.SOCKS}):
                if bad is None:
                    socks = {k: v for k, v in socks.items() if v is not None}  # a sock missing from the dict
                r = t.update(socks, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(socks, a).score, 0.0)
            self.assertEqual((dict(t.breaches), t.min_sock_clear, dict(t.min_finger_clear), t.arms_unavailable),
                             before)
            self.assertEqual(t.bad_steps, 3)
            r, _ = play(Film(), t, frames=frames[-25:])  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [box((5.0, -3.0, -1.0)), box((0.3, 0.0, 0.0), about("x", 120)), box((-0.2, 0.6, L.TOP), about("y", 90)),
               box(RING["left"][0], about("x", 44.9)), box((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0)),
               box(C.tolist() + [L.TOP + H])]
        t = Tracker(HZ)
        a = snap()
        for i in range(len(odd)):
            socks = {n: odd[(i + k) % len(odd)] for k, n in enumerate(L.SOCKS)}
            r = t.update(socks, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertFalse(r.success)
        r = t.update(start_socks(), None)
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def breached(self):
        f = Film().hold(5).go(20, left=[L.SPILL_X, 0.03, L.TOP + 0.15])  # fingers over the spill
        g = Film()
        g.frames = f.frames + procedure(settle=5).frames[1:200]
        _, t = play(g)
        self.assertTrue(t.breach)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.breached()
        r = t.update(start_socks(), snap())  # every sock jumps back onto its tray in one step: a new episode
        self.assertFalse(t.breach)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        f = procedure()
        r, _ = play(f, t, frames=f.frames[1:])  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.breached()
        socks = {n: box(START[n] + [0.02 * (-1) ** i, -0.02, 0]) for i, n in enumerate(L.SOCKS)}
        t.update(socks, snap())
        self.assertFalse(t.breach)

    def test_gradual_return_does_not_reset(self):
        t = self.breached()
        cur = {n: p for n, (p, _) in zip(L.SOCKS, t.prev)}
        f = Film({n: (cur[n], Q0) for n in L.SOCKS})
        for n in L.SOCKS:
            f.go(40, pos={n: [cur[n][0], cur[n][1], CARRY]}).go(40, pos={n: [*START[n][:2], CARRY]})
            f.go(40, pos={n: START[n]})
        f.hold(10)
        play(f, t)
        self.assertTrue(t.breach)  # back at the start, but it got there gradually: same episode

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.breached()
        socks = start_socks()
        socks["sock_2"] = box((0.30, 0.45, L.TOP + H))  # knocked far, not to the start
        t.update(socks, snap())
        self.assertTrue(t.breach)

    def test_a_dropped_sock_is_not_a_teleport(self):
        # Free fall from 0.3 m back onto its tray: fast, but its velocity explains it.
        t = Tracker(HZ)
        play(Film().hold(5), t)
        p = START["sock_1"] + [0, 0, 0.3]
        socks = start_socks()
        socks["sock_1"] = box(p)
        t.update(socks, snap())
        t._flag("left_fingers_entered")
        z, v = p[2], 0.0
        while z > START["sock_1"][2]:
            v += 9.81 * DT
            z = max(START["sock_1"][2], z - v * DT)
            socks["sock_1"] = box((p[0], p[1], z), vel=(0, 0, -v if z > START["sock_1"][2] else 0))
            t.update(socks, snap())
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

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in (*L.SOCKS, "spill", "left_tray", "right_tray"):
            self.assertIn(n, names)
        self.assertEqual(self.task.SCORE_OVER, "end")
        masses = {o.name: o.mass for o in self.task.OBJECTS}
        self.assertIsNone(masses["spill"])
        self.assertIsNone(masses["left_tray"])
        self.assertEqual(masses["sock_3"], L.SOCK_MASS)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        snap(right_tcp=(L.SPILL_X, -0.02, L.TOP + 0.2))  # fingers over the spill
        self.assertFalse(self.task.success(start_socks()))
        self.assertTrue(tr.breach)
        self.assertEqual(self.task.score(start_socks()), 0.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertFalse(tr.breach)
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(start_socks()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(start_socks()))
        self.assertEqual(self.task.score(start_socks()), 0.0)
        self.assertFalse(tr.breach)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            snap()
            self.assertFalse(self.task.success(start_socks()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches(self):
        keep = {k: os.environ.get(k) for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
        try:
            for k in keep:
                os.environ.pop(k, None)
            plan = self.task.demo()
            self.assertEqual(sorted(plan), ["left", "right"])
            self.assertAlmostEqual(sum(w.duration for w in plan["left"]), sum(w.duration for w in plan["right"]))
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(sorted(self.task.demo()), ["right"])
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(self.task.demo(), {})
        finally:
            for k, v in keep.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    def test_demo_fingers_never_open_wide_near_the_ring(self):
        # Every waypoint within 15 cm of the spill's centre keeps the fingers at 50 mm or less.
        for arm, wps in self.task.demo().items():
            for w in wps:
                if math.hypot(w.pos[0] - L.SPILL_X, w.pos[1] - L.SPILL_Y) < 0.15:
                    self.assertGreaterEqual(w.finger, self.task.PART - 1e-9, (arm, w.label))


if __name__ == "__main__":
    unittest.main()
