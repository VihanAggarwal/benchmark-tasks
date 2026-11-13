"""Offline tests for scoring.py, offline_cordlock.py and task.py: synthetic lock states (mechkit AssemblyStates from the
task's own spec), synthetic cord states (cablekit CableStates) and synthetic arm snapshots (rigstate.Arm); no simulator.
python3 -m unittest discover -s tests -v"""

import json
import math
import os
import pathlib
import re
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir():
            return d
    return None


OFFLINE = _offline_dir()
if OFFLINE is not None:
    for _p in (OFFLINE, OFFLINE / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))  # mechkit, cablekit and task.py import yam_arena: the offline stand-in

import cablekit  # noqa: E402
import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_cordlock as OC  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
import tiltgrip  # noqa: E402

WIN = S.WINDOW  # 20 steps
TOP = L.TOP
LOCK0 = np.array([L.LOCK_ORIGIN[0], L.LOCK_ORIGIN[1], TOP])
HERE_LOCK = np.array([0.302, 0.040, TOP])  # where the demo sets the lock down
AWAY = {a: rigstate.Arm(np.array([0.30, s * 0.20, TOP + 0.12]), np.array([0.30, s * 0.20, TOP + 0.107]), 0.32)
        for a, s in (("left", 1.0), ("right", -1.0))}
MM = 0.001


# -- states --------------------------------------------------------------------------------------------------------


def lock(origin=HERE_LOCK, q=L.PINCH_Q, yaw=0.0, source="sim"):
    return mk.state_from_poses(T.LOCK_SPEC, (*origin, *mk.yaw_quat(yaw)), joints={S.PLUNGER: q}, source=source)


def cord(line):
    line = np.asarray(line, dtype=float)
    pts = line[1:-1]
    sd = line[0] - line[1]
    ed = line[-1] - line[-2]
    return cablekit.CableState(pos=pts.mean(axis=0), quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]), lin_vel=np.zeros(3),
                               ang_vel=np.zeros(3), low=line.min(axis=0), high=line.max(axis=0), points=pts,
                               start_dir=sd / np.linalg.norm(sd), end_dir=ed / np.linalg.norm(ed),
                               seg_len=L.CORD_SEGMENT)


def stations():
    """Arc lengths of the 18 line points from end 0: 0, 10, 30, ..., 310, 320 mm."""
    n = int(round(L.CORD_LEN / L.CORD_SEGMENT))
    return np.array([0.0, *[(i + 0.5) * L.CORD_SEGMENT for i in range(n)], L.CORD_LEN])


def path_line(pieces, start, heading):
    """Sample a unit-speed path in a horizontal plane at the 18 stations. ``pieces``: ("s", length) straight or
    ("a", radius, turn) arc (turn > 0 to the left); ``start`` (x, y, z) at end 0, ``heading`` rad."""
    out = []
    for s in stations():
        p = np.array(start, dtype=float)
        h = heading
        left = s
        for piece in pieces:
            if left <= 0:
                break
            if piece[0] == "s":
                d = min(left, piece[1])
                p = p + d * np.array([math.cos(h), math.sin(h), 0.0])
                left -= piece[1]
            else:
                r, turn = piece[1], piece[2]
                arc = abs(turn) * r
                d = min(left, arc)
                a = d / r * (1 if turn > 0 else -1)
                c = p + r * np.array([-math.sin(h), math.cos(h), 0.0]) * (1 if turn > 0 else -1)
                rel = p - c
                rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
                p = c + rot @ rel
                h += a
                left -= arc
        if left > 1e-12:  # past the last piece: straight on
            p = p + left * np.array([math.cos(h), math.sin(h), 0.0])
        out.append(p)
    return np.array(out)


def through_line(out_mm=48.5, origin=HERE_LOCK, x_local=L.HOLE_CX, z_local=L.HOLE_CZ, reverse=False):
    """A straight cord through the lock along y: end 0 ``out_mm`` past the far face (or end 1, with ``reverse``)."""
    end0 = origin + np.array([x_local, L.HALF_DEPTH + out_mm * MM, z_local])
    line = path_line([("s", 1.0)], end0, -math.pi / 2)
    return line[::-1].copy() if reverse else line


def end_in_line(depth_mm, origin=HERE_LOCK, x_local=L.HOLE_CX, z_local=L.HOLE_CZ):
    """End 0 ``depth_mm`` into the lock through the near face, the rest of the cord on the near side."""
    end0 = origin + np.array([x_local, -L.HALF_DEPTH + depth_mm * MM, z_local])
    return path_line([("s", 1.0)], end0, -math.pi / 2)


def start_states():
    return {S.LOCK: lock(LOCK0, 0.0), S.CORD: cord(OC.start_line())}


def prime(tracker):
    """One step with the button pressed flush and the cord still at its start (Pressed latches), as the demo does."""
    tracker.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(OC.start_line())}, AWAY)
    return tracker


def primed():
    return prime(S.Tracker())


def run(tracker, states, arms=AWAY, n=WIN + 1):
    res = None
    for _ in range(n):
        res = tracker.update(states, arms)
    return res


def holding_lock_arms():
    """The left gripper squeezing the lock where the demo sets it down (a 34 mm gap)."""
    tcp = HERE_LOCK + np.array([0.0, 0.0, 0.018])
    left = rigstate.Arm(tcp, tcp + 0.013 * tiltgrip.approach(T.YAW["left"], T.ROLL["left"]), 34.0 / 93.9)
    return {"left": left, "right": AWAY["right"]}


def holding_cord_arms(line):
    """The right gripper closed on the cord 81 mm from end 0, the cord's centre 12 mm below the TCP."""
    p = np.asarray(line)[5]  # 90 mm along the cord from end 0
    tcp = p + np.array([0.0, 0.0, 0.012])
    right = rigstate.Arm(tcp, tcp + 0.013 * tiltgrip.approach(T.YAW["right"], T.ROLL["right"]), 7.9 / 93.9)
    return {"left": AWAY["left"], "right": right}


# -- geometry recomputed ---------------------------------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_passage(self):
        self.assertAlmostEqual(L.passage_width(0.0), 0.0044, places=9)
        self.assertAlmostEqual(L.passage_width(L.TRAVEL), 0.0114, places=9)
        self.assertAlmostEqual(L.passage_width(0.0035), 0.0079, places=9)
        self.assertAlmostEqual(L.PINCH_Q, 0.0035, places=9)
        self.assertAlmostEqual(L.passage_x(L.TRAVEL)[0], L.FACE_HOLE_X[0], places=9)  # holes in line at flush
        self.assertAlmostEqual(L.PLUNGER_HOLE_Z[1] - L.PLUNGER_HOLE_Z[0], 0.013, places=9)
        per_side = (L.passage_width(L.TRAVEL) - L.CORD_D) / 2
        self.assertAlmostEqual(per_side, 0.00175, places=9)
        self.assertAlmostEqual((0.013 - L.CORD_D) / 2, 0.00255, places=9)
        # face hole: the catalogue 11.4 x 16.5 mm, its centre 15 mm from the closed end
        self.assertAlmostEqual(L.FACE_HOLE_X[1] - L.FACE_HOLE_X[0], 0.0114, places=9)
        self.assertAlmostEqual(L.FACE_HOLE_Z[1] - L.FACE_HOLE_Z[0], 0.0165, places=9)
        self.assertAlmostEqual(L.HOLE_CX - (-0.017), 0.015, places=9)

    def test_forces(self):
        self.assertAlmostEqual(L.SPRING * L.TRAVEL, 10.5, places=6)
        self.assertAlmostEqual(L.SPRING * L.PINCH_Q, 5.25, places=6)
        self.assertAlmostEqual(L.SPRING * L.PLUNGER_HELD, 1.5, places=6)
        weight = L.LOCK_MASS * 9.81
        self.assertAlmostEqual(weight, 0.0863, places=3)
        hold = 2 * 0.5 * L.SPRING * L.PLUNGER_HELD  # two contact faces at the pair's 0.5
        self.assertAlmostEqual(hold, 1.5, places=6)
        self.assertEqual(int(hold / weight), 17)
        # the demo's squeeze: 26 mm across the 34 mm body, 5 N/mm a finger
        self.assertAlmostEqual(5000.0 * (0.034 - 0.026) / 2, 20.0, places=6)
        self.assertGreater(5000.0 * (0.034 - 0.026) / 2, L.SPRING * L.TRAVEL)
        self.assertEqual(OC.squeeze_travel(0.026), L.TRAVEL)
        self.assertEqual(OC.squeeze_travel(0.060), 0.0)

    def test_masses_from_volumes(self):
        def volume(table):
            return sum(s[0] * s[1] * s[2] for _, _, s in table)
        body = volume(L.BODY_SHAPES) * L.DELRIN
        plunger = volume(L.PLUNGER_SHAPES) * L.DELRIN
        self.assertAlmostEqual(body, L.BODY_MASS, delta=0.00005)
        self.assertAlmostEqual(plunger, L.PLUNGER_MASS, delta=0.00005)
        self.assertAlmostEqual(L.LOCK_MASS, 0.0088, places=6)
        self.assertAlmostEqual(L.CORD_MASS / L.CORD_LEN, 0.046, delta=0.0005)

    def test_lock_extent(self):
        lo, hi = T.LOCK_SPEC.local_bounds()
        self.assertTrue(np.allclose(hi - lo, (0.041, 0.013, 0.020), atol=1e-9))
        lo7, hi7 = T.LOCK_SPEC.local_bounds({S.PLUNGER: L.TRAVEL})
        self.assertAlmostEqual(hi7[0] - lo7[0], 0.034, places=9)  # flush
        st = lock(LOCK0, 0.0)
        self.assertAlmostEqual(float(st.low[0]), 0.285, places=9)
        self.assertAlmostEqual(float(st.high[0]), 0.326, places=9)
        self.assertAlmostEqual(float(st.low[2]), TOP, places=9)

    def test_cord(self):
        c = T.CORD_SPEC
        self.assertEqual(c.segments, 16)
        self.assertAlmostEqual(c.bend_stiffness * c.seg_len, 1e-3, places=12)  # EI
        w = L.CORD_MASS / L.CORD_LEN * 9.81
        self.assertAlmostEqual(w * 0.06 ** 4 / (8 * 1e-3) * 1000, 0.73, delta=0.01)  # 60 mm droops 0.7 mm
        self.assertAlmostEqual(math.pi ** 2 * 1e-3 / (4 * 0.053 ** 2), 0.88, delta=0.005)  # 53 mm buckles at 0.88 N
        free = T.GRIP_BACK - 0.0054
        self.assertAlmostEqual(free, 0.0756, places=9)
        self.assertAlmostEqual(w * free ** 4 / (8 * 1e-3) * 1000, 1.84, delta=0.01)
        self.assertAlmostEqual(math.pi ** 2 * 1e-3 / (4 * free ** 2), 0.43, delta=0.005)
        self.assertEqual(c.collision_sides, 16)
        self.assertIsInstance(c, cablekit.Cable)
        line = OC.start_line()
        self.assertTrue(np.allclose(line[0], (0.300, -0.060, TOP + L.CORD_R)))
        self.assertTrue(np.allclose(line[-1], (0.300, -0.380, TOP + L.CORD_R)))

    def test_tilt_fit_angle(self):
        """A straight 20 mm length passes the 11.4 mm passage within about 14 degrees of the hole's axis."""
        def width(t):
            return L.CORD_D / math.cos(t) + L.DEPTH * math.tan(t)
        self.assertLess(width(math.radians(14.0)), 0.0114)
        self.assertGreater(width(math.radians(15.0)), 0.0114)

    def test_demo_numbers(self):
        self.assertAlmostEqual(T.PUSH_START_Y + T.GRIP_BACK, 0.007, places=9)  # end 26.5 mm short of the near face
        far_out = T.PUSH_END_Y + T.GRIP_BACK - (T.HOLD_LOCK[1] + L.HALF_DEPTH)
        self.assertAlmostEqual(far_out, 0.0485, places=9)
        self.assertAlmostEqual(T.PICK_LOCK[0] - 0.0035, T.HOLD_LOCK[0] + 0.0, delta=0.0001)  # squeezed: centred
        self.assertAlmostEqual(T.PUSH_Z - (T.CORD_PICK_Z - L.CORD_R), 0.01045, places=9)  # cord at 10.45 mm
        self.assertAlmostEqual(tiltgrip.approach(T.YAW["left"], T.ROLL["left"])[1], -0.479, places=3)
        self.assertAlmostEqual(tiltgrip.approach(T.YAW["right"], T.ROLL["right"])[1], 0.565, places=3)


# -- the cord's geometry against the lock ---------------------------------------------------------------------------


class Crossing(unittest.TestCase):
    def local(self, line, origin=HERE_LOCK):
        return S.to_local(line, origin, np.eye(3))

    def test_through_straight(self):
        out = S.through_out(self.local(through_line(45.0)))
        self.assertAlmostEqual(out, 0.045, places=9)

    def test_near_side_short(self):
        line = through_line(320.0 - 13.0 - 39.0)  # 39 mm left on the near side
        self.assertAlmostEqual(S.through_out(self.local(line)), 0.039, places=9)

    def test_partial_threading_has_no_out(self):
        loc = self.local(end_in_line(7.0))  # 0.5 mm past the mid-plane
        self.assertEqual(S.through_out(loc), -math.inf)
        self.assertAlmostEqual(S.depth_in(loc), 0.007, places=9)

    def test_outside_crossing(self):
        line = through_line(45.0, x_local=0.030)  # past the lock's side: crossing outside the window
        self.assertIsNone(S.through_out(self.local(line)))
        self.assertEqual(S.depth_in(self.local(line)), -math.inf)

    def test_doubled_cord(self):
        end0 = HERE_LOCK + np.array([L.HOLE_CX + 0.003, -0.031, L.HOLE_CZ])  # the return leg 6 mm over, also inside
        line = path_line([("s", 0.050), ("a", 0.003, math.pi), ("s", 0.300)], end0, math.pi / 2)
        loc = self.local(line)
        self.assertEqual(sum(ok for _, _, ok in S.crossings(loc)), 2)
        self.assertIsNone(S.through_out(loc))

    def test_curled_tail_measured_along(self):
        end0 = HERE_LOCK + np.array([L.HOLE_CX, L.HALF_DEPTH + 0.045, L.HOLE_CZ])
        a = (L.CORD_LEN - 0.078 - math.pi * 0.015) / 2
        line = path_line([("s", 0.078), ("s", a), ("a", 0.015, math.pi), ("s", a)], end0, -math.pi / 2)
        loc = self.local(line)
        self.assertAlmostEqual(float(loc[-1][1]), -0.0265, delta=0.0005)  # end 1 is 20 mm from the entry face
        self.assertGreater(S.through_out(loc), 0.040)  # but 230 mm along the cord
        self.assertAlmostEqual(S.through_out(loc), 0.045, delta=1e-6)

    def test_far_end_wrapped_round_outside(self):
        end0 = HERE_LOCK + np.array([L.HOLE_CX + 0.030, -0.010, L.HOLE_CZ])
        line = path_line([("s", 0.0265), ("a", 0.015, math.pi), ("s", 1.0)], end0, math.pi / 2)
        loc = self.local(line)
        inside = [ok for _, _, ok in S.crossings(loc)]
        self.assertEqual(sum(inside), 1)
        self.assertEqual(len(inside), 2)
        along = 0.010 + math.pi * 0.015 + 0.0265  # 83.6 mm; the polyline's chords read the arc a little short
        self.assertLess(S.through_out(loc), along)
        self.assertAlmostEqual(S.through_out(loc), along, delta=0.004)

    def test_line_distance(self):
        line = through_line(45.0)
        d = S.line_distance(line)
        p = line[3] + np.array([0.0, 0.0, 0.010])
        self.assertAlmostEqual(float(d(p)[0]), 0.010 - L.CORD_R, places=6)
        self.assertEqual(float(d(line[3])[0]), 0.0)


# -- the tracker: tiers, near misses, exploits ----------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def setUp(self):
        self.t = primed()

    def score(self, states, arms=AWAY, n=WIN + 1):
        return run(self.t, states, arms, n).score

    def test_nothing(self):
        self.assertEqual(self.score(start_states()), 0.0)

    def test_success(self):
        st = {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}
        res = run(self.t, st, n=WIN)
        self.assertFalse(res.success)  # 20 states: one still interval short
        self.assertEqual(res.score, 0.75)
        res = self.t.update(st, AWAY)
        self.assertTrue(res.success)
        self.assertEqual(res.score, 1.0)

    def test_either_end(self):
        st = {S.LOCK: lock(), S.CORD: cord(through_line(48.5, reverse=True))}
        self.assertEqual(self.score(st), 1.0)

    def test_depth_near_miss(self):
        self.assertEqual(self.score({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(4.9))}), 0.0)
        self.t.reset()
        prime(self.t)
        self.assertEqual(self.score({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(5.0))}), 0.25)

    def test_end_outside_outline(self):
        st = {S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(6.0, x_local=0.010))}
        self.assertEqual(self.score(st), 0.0)

    def test_partial_threading_released(self):
        st = {S.LOCK: lock(), S.CORD: cord(end_in_line(7.0))}
        self.assertEqual(self.score(st), 0.25)

    def test_out_near_misses(self):
        for out, want in ((9.9, 0.25), (10.0, 0.5), (39.9, 0.5), (40.0, 1.0)):
            t = primed()
            t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(6.0))}, AWAY)  # the first stage on the way
            res = run(t, {S.LOCK: lock(), S.CORD: cord(through_line(out))})
            self.assertEqual(res.score, want, out)

    def test_near_side_39(self):
        st = {S.LOCK: lock(), S.CORD: cord(through_line(320.0 - 13.0 - 39.0))}
        self.assertEqual(self.score(st), 0.5)

    def test_plunger_near_miss(self):
        self.assertEqual(self.score({S.LOCK: lock(q=0.0009), S.CORD: cord(through_line(48.5))}), 0.75)
        self.t.reset()
        prime(self.t)
        self.assertEqual(self.score({S.LOCK: lock(q=0.0010), S.CORD: cord(through_line(48.5))}), 1.0)

    def test_support_near_miss(self):
        for lift, want in ((0.0021, 0.75), (0.0020, 1.0)):
            t = primed()
            o = HERE_LOCK + np.array([0.0, 0.0, lift])
            res = run(t, {S.LOCK: lock(o), S.CORD: cord(through_line(48.5, origin=o))})
            self.assertEqual(res.score, want, lift)

    def test_moving(self):
        for total, want in ((0.0021, 0.75), (0.0019, 1.0)):
            t = primed()
            res = None
            for k in range(WIN + 1):
                o = HERE_LOCK + np.array([total * k / WIN, 0.0, 0.0])
                res = t.update({S.LOCK: lock(o), S.CORD: cord(through_line(48.5, origin=o))}, AWAY)
            self.assertEqual(res.score, want, total)

    def test_lock_held(self):
        st = {S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(through_line(48.5))}
        self.assertEqual(self.score(st, holding_lock_arms()), 0.25)

    def test_cord_held(self):
        line = through_line(48.5)
        st = {S.LOCK: lock(), S.CORD: cord(line)}
        self.assertEqual(self.score(st, holding_cord_arms(line)), 0.25)

    def test_hover_over_lock(self):
        tcp = HERE_LOCK + np.array([0.0, 0.0, 0.045])  # pad tips 32 mm above the table, over the lock
        hover = {"left": rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.64), "right": AWAY["right"]}
        st = {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}
        self.assertEqual(self.score(st, hover), 0.25)

    def test_zone_near_miss(self):
        st = {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}
        dist = S.line_distance(through_line(48.5))
        for clear, want in ((0.029, 0.25), (0.031, 1.0)):
            top = HERE_LOCK + np.array([L.HOLE_CX, -0.150, L.HOLE_CZ + L.CORD_R])  # on the cord's tail, open 0.32
            radius = rigstate.zone_radius(0.32)
            tcp = top + np.array([0.0, 0.0, clear + radius + 0.013])
            right = rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, 0.013]), 0.32)
            arms = {"left": AWAY["left"], "right": right}
            gap = rigstate.hands_gap(arms, dist)
            self.assertAlmostEqual(gap, clear, delta=0.0002)
            t = primed()
            t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(6.0))}, AWAY)
            self.assertEqual(run(t, st, arms).score, want, clear)

    def test_unreadable_arms(self):
        st = {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}
        bad_left = rigstate.Arm(np.array([np.nan, 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 0.3)
        forms = [None, {}, {"left": AWAY["left"]}, {"left": bad_left, "right": AWAY["right"]}, "arms"]
        for arms in forms:
            t = primed()
            t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(6.0))}, AWAY)
            res = run(t, st, arms)
            self.assertEqual(res.score, 0.25, repr(arms)[:40])
            self.assertFalse(res.success)

    def test_distance_raises(self):
        class Broken:
            def __init__(self, s):
                self._s = s

            def __getattr__(self, name):
                if name == "distance":
                    raise RuntimeError("no distance")
                return getattr(self._s, name)

        st = {S.LOCK: Broken(lock()), S.CORD: cord(through_line(48.5))}
        res = run(self.t, st)
        self.assertFalse(res.details["released_lock"])
        self.assertLess(res.score, 0.5)

    def test_pulled_back(self):
        run(self.t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}, AWAY, WIN + 5)
        self.assertEqual(self.t.last.score, 1.0)
        res = run(self.t, {S.LOCK: lock(), S.CORD: cord(through_line(30.0))})
        self.assertEqual(res.score, 0.5)  # the end state now; the 1.0 was not latched

    def test_cord_slides_out(self):
        run(self.t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))})
        res = run(self.t, {S.LOCK: lock(q=0.0), S.CORD: cord(OC.start_line())})
        self.assertEqual(self.t.resets, 0)
        self.assertEqual(res.score, 0.25)  # only the latched first stage remains

    def test_lock_set_down_by_the_cord(self):
        """Let go in the air while threaded, set down, all released: 1.0 (the instruction states the end state)."""
        o = HERE_LOCK + np.array([0.0, 0.0, 0.050])
        self.t.update({S.LOCK: lock(o), S.CORD: cord(through_line(48.5, origin=o))}, AWAY)
        self.assertEqual(self.score({S.LOCK: lock(), S.CORD: cord(through_line(48.5))}), 1.0)

    def test_unreadable_states(self):
        good = {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}
        run(self.t, good, n=WIN - 3)
        no_points = cablekit.CableState(pos=np.zeros(3), quat_xyzw=np.array([0, 0, 0, 1.0]), lin_vel=np.zeros(3),
                                        ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))
        nan_line = through_line(48.5)
        nan_line[4, 1] = np.nan
        plain = mk.state_from_poses(T.LOCK_SPEC, (*HERE_LOCK, 0, 0, 0, 1), joints={S.PLUNGER: 0.0035})
        no_joint = type("S", (), {"origin": HERE_LOCK, "quat_xyzw": np.array([0, 0, 0, 1.0]), "joints": {},
                                   "low": HERE_LOCK})()
        for bad in ({S.LOCK: plain, S.CORD: no_points}, {S.LOCK: plain, S.CORD: cord(nan_line)},
                    {S.LOCK: no_joint, S.CORD: cord(through_line(48.5))}, {S.CORD: cord(through_line(48.5))}, {}):
            res = self.t.update(bad, AWAY)
            self.assertFalse(res.details["valid_state"])
            self.assertFalse(res.success)
            self.assertEqual(res.score, 0.25)
        res = run(self.t, good, n=WIN)  # the windows start again
        self.assertFalse(res.success)
        self.assertTrue(self.t.update(good, AWAY).success)

    def test_teleport_reset(self):
        run(self.t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))})
        self.assertEqual(self.t.last.score, 1.0)
        res = self.t.update(start_states(), AWAY)  # the lock jumps 90 mm back onto its start: a new episode
        self.assertEqual(self.t.resets, 1)
        self.assertEqual(res.score, 0.0)

    def test_turned_lock(self):
        """The scorer works in the lock's frame: a lock turned 30 degrees, threaded along its own axis."""
        yaw = math.radians(30.0)
        st_lock = lock(yaw=yaw)
        r = st_lock.rotation()
        line = S.to_local(through_line(48.5), HERE_LOCK, np.eye(3)) @ r.T + HERE_LOCK
        self.assertEqual(self.score({S.LOCK: st_lock, S.CORD: cord(line)}), 1.0)


class Pressed(unittest.TestCase):
    """The instruction's "press in the button": every tier needs the plunger 5 mm in before a cord end is 5 mm in."""

    def test_wedged_without_press(self):
        """A straight end wedged into the 4.4 mm slot pushes the plunger 3.5 mm in at most: no tier, all the way to a
        cord through, let go, on the table and still."""
        t = S.Tracker()
        t.update({S.LOCK: lock(q=0.0033), S.CORD: cord(end_in_line(5.0))}, AWAY)
        res = run(t, {S.LOCK: lock(q=L.PINCH_Q), S.CORD: cord(through_line(48.5))})
        self.assertFalse(res.details["pressed"])
        self.assertTrue(res.details["end_state_1s"])
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_press_near_miss(self):
        for q, want in ((0.0049, 0.0), (0.0050, 1.0)):
            t = S.Tracker()
            t.update({S.LOCK: lock(q=q), S.CORD: cord(OC.start_line())}, AWAY)
            self.assertEqual(run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}).score, want, q)

    def test_press_with_the_end_short_of_5_mm(self):
        t = S.Tracker()
        t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(4.9))}, AWAY)
        self.assertTrue(t.pressed)
        self.assertEqual(run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}).score, 1.0)

    def test_press_once_the_cord_is_in_does_not_count(self):
        for line in (end_in_line(5.0), through_line(20.0)):
            t = S.Tracker()
            t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(line)}, AWAY)
            self.assertFalse(t.pressed)
            self.assertEqual(run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}).score, 0.0)

    def test_first_stage_needs_the_press(self):
        t = S.Tracker()
        self.assertEqual(run(t, {S.LOCK: lock(q=L.PINCH_Q), S.CORD: cord(end_in_line(7.0))}).score, 0.0)
        t = primed()
        self.assertEqual(run(t, {S.LOCK: lock(q=L.PINCH_Q), S.CORD: cord(end_in_line(7.0))}).score, 0.25)

    def test_press_release_then_wedge_is_capped(self):
        """Press, let the button spring back, then wedge the cord in (3.5 mm in) and on through: 0.25 at most."""
        t = primed()  # the press latches
        t.update({S.LOCK: lock(q=0.0), S.CORD: cord(OC.start_line())}, AWAY)  # let go: the button springs out
        for d in (5.0, 7.0):  # the end wedged 5 mm in, the plunger only 3.5 mm in
            res = t.update({S.LOCK: lock(q=L.PINCH_Q), S.CORD: cord(end_in_line(d))}, AWAY)
        self.assertTrue(res.details["press_lost"])
        res = run(t, {S.LOCK: lock(q=L.PINCH_Q), S.CORD: cord(through_line(48.5))})
        self.assertTrue(res.details["end_state_1s"])
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.success)

    def test_press_held_throughout(self):
        t = primed()
        for d in (2.0, 5.0, 6.0):
            res = t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(d))}, AWAY)
            self.assertFalse(res.details["press_lost"])
        t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(through_line(20.0))}, AWAY)
        res = run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))})
        self.assertFalse(res.details["press_lost"])
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_press_dropped_mid_insertion(self):
        t = primed()
        t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(5.0))}, AWAY)
        res = t.update({S.LOCK: lock(q=0.0049), S.CORD: cord(end_in_line(6.0))}, AWAY)  # 4.9 mm in: dropped
        self.assertTrue(res.details["press_lost"])
        t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(6.4))}, AWAY)  # pressed again: stays lost
        res = run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))})
        self.assertTrue(res.details["press_lost"])
        self.assertEqual(res.score, 0.25)

    def test_press_dropped_after_through_is_fine(self):
        """The window ends when the cord is through: the plunger then rests on the cord (3.5 mm in)."""
        t = primed()
        t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(end_in_line(5.0))}, AWAY)
        t.update({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(through_line(1.0))}, AWAY)
        res = run(t, {S.LOCK: lock(q=L.PINCH_Q), S.CORD: cord(through_line(48.5))})
        self.assertFalse(res.details["press_lost"])
        self.assertEqual(res.score, 1.0)

    def test_press_lost_clears_on_reset(self):
        t = primed()
        t.update({S.LOCK: lock(q=0.0), S.CORD: cord(end_in_line(5.0))}, AWAY)
        self.assertTrue(t.press_lost)
        t.update(start_states(), AWAY)
        self.assertFalse(t.press_lost)

    def test_press_survives_and_resets(self):
        t = primed()
        run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))})
        self.assertEqual(t.last.score, 1.0)
        t.update(start_states(), AWAY)  # teleported back to the start: a new trial
        self.assertFalse(t.pressed)
        self.assertEqual(run(t, {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}).score, 0.0)


# -- the hooks, the offline kinematics and the demo -----------------------------------------------------------------


def play(plan: dict):
    """The demo through the stand-in ArmPlan and IK at 20 Hz, as tools/offline/check_task.py plays it, feeding
    rigstate and the task's success(); returns (last score, first step success held 1 s, rows)."""
    import yamkin as K
    from yam_arena import demo as D

    rigstate.offline_reset()
    mk.offline_reset()
    bases = K.BASES["station"]
    plans = {a: D.ArmPlan(list(w), bases[a]) for a, w in plan.items()}
    end = max((p.duration for p in plans.values()), default=0.0)
    steps = int(math.floor((end + 3.0) * 20 + 1e-9)) + 1
    objects = {S.LOCK: lock(LOCK0, 0.0, source="offline"), S.CORD: None}
    held, streak, score = None, 0, 0.0
    for k in range(steps):
        t = (k + 1) / 20
        snap = {}
        for a in ("left", "right"):
            if a in plans:
                act, _ = plans[a].action7(t)
                q, opening = act[:6], float(act[6])
            else:
                q, opening = np.array(K.READY, dtype=float), 1.0
            p, r = K.tcp_pose(q, bases[a])
            snap[a] = {"tcp": p, "tips": K.pad_tip(p, r), "opening": opening}
        rigstate.offline_feed(snap)
        ok = T.success(dict(objects))
        score = T.score(dict(objects))
        streak = streak + 1 if ok else 0
        if held is None and streak >= 20:
            held = k + 1
    return score, held


class Hooks(unittest.TestCase):
    def tearDown(self):
        for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
            os.environ.pop(k, None)

    def test_start_layout(self):
        rigstate.offline_reset()
        mk.offline_reset()
        rigstate.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": 1.0} for a in AWAY})
        objects = {S.LOCK: lock(LOCK0, 0.0, source="offline"), S.CORD: None}
        self.assertFalse(T.success(objects))
        self.assertEqual(T.score(objects), 0.0)

    def test_sim_states_untouched(self):
        """With states read from a simulator (source "sim"), the offline kinematics never run."""
        rigstate.offline_reset()
        mk.offline_reset()
        rigstate.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": 0.32} for a in AWAY})
        st = {S.LOCK: lock(), S.CORD: cord(through_line(48.5))}
        self.assertFalse(T.no_simulator(st))
        T.success({S.LOCK: lock(q=L.TRAVEL), S.CORD: cord(OC.start_line())})  # the press, before the cord goes in
        for _ in range(WIN + 1):
            T.success(dict(st))
        self.assertEqual(T.score(st), 1.0)
        self.assertEqual(T.OFFLINE.lock_holder, None)

    def test_demo(self):
        score, held = play(T.demo())
        self.assertEqual(score, 1.0)
        self.assertIsNotNone(held)
        self.assertLess(held, 600)

    def test_one_arm_idle(self):
        for arm in ("left", "right"):
            os.environ["DEMO_IDLE_ARM"] = arm
            score, held = play(T.demo())
            self.assertEqual(score, 0.0, arm)
            self.assertIsNone(held)

    def test_hold_only(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})

    def test_variants(self):
        want = {"naive": 0.0, "short": 0.25, "keep_lock": 0.25, "keep_cord": 0.25, "early_lock": 0.0,
                "in_place": 1.0, "slide_right": 0.0, "slide_left": 0.0}
        for v, w in want.items():
            os.environ["DEMO_VARIANT"] = v
            score, _ = play(T.demo())
            self.assertEqual(score, w, v)
        os.environ["DEMO_VARIANT"] = "nonsense"
        with self.assertRaises(ValueError):
            T.demo()

    def test_demo_within_run_limit(self):
        plan = T.demo()
        end = max(sum(w.duration for w in v) for v in plan.values())
        self.assertLessEqual(end * 20, 900)
        for arm, wps in plan.items():
            for w in wps:
                self.assertTrue(hasattr(w.yaw, "rotation"), (arm, w.label))  # every waypoint tilted
            # every open or close takes 1.5 s
            prev = None
            for w in wps:
                if prev is not None and abs(w.finger - prev) > 0.005:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                prev = w.finger


class Offline(unittest.TestCase):
    def test_blocked_unpressed(self):
        rig = OC.OfflineRig()
        rig.origin = HERE_LOCK.copy()
        rig.line = end_in_line(-1.0)
        rig.cord_holder = "right"
        rig.cord_rel = rig.line - np.array([0.3, 0.0, TOP])
        rig.line[:, 1] += 0.010  # pushed 10 mm on: blocked, the loose lock is shoved away
        rig._passage()
        self.assertFalse(rig.threaded)
        self.assertGreater(rig.origin[1], HERE_LOCK[1])

    def test_fits_only_open(self):
        rig = OC.OfflineRig()
        tip = np.array([L.HOLE_CX, -L.HALF_DEPTH, 0.0105])
        rig.q = 0.0
        self.assertFalse(rig.fits(tip))
        rig.q = 0.0035  # 7.9 mm wide, centred 1.75 mm toward the button from the hole's centre
        self.assertFalse(rig.fits(tip))
        self.assertTrue(rig.fits(np.array([sum(L.passage_x(0.0035)) / 2, -L.HALF_DEPTH, 0.0105])))
        rig.q = L.TRAVEL
        self.assertTrue(rig.fits(tip))
        self.assertFalse(rig.fits(np.array([L.HOLE_CX, -L.HALF_DEPTH, L.CORD_R])))  # lying on the table


# -- kit files and text ---------------------------------------------------------------------------------------------


class Files(unittest.TestCase):
    def test_kit_unchanged(self):
        kit = None
        for p in HERE.parents:
            if (p / "tools" / "kit" / "mechkit.py").is_file():
                kit = p / "tools" / "kit"
                break
        if kit is None:
            self.skipTest("tools/kit not found (a copy of the task folder on its own)")
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual((HERE / name).read_bytes(), (kit / name).read_bytes(), name)
        self.assertEqual(mk.VERSION, "2")

    def test_task_json(self):
        d = json.loads((HERE / "task.json").read_text())
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "run_limit": 60,
                  "fidelity": 490, "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertEqual(d["category"], "Societal relatability")
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertRegex(d["scores"], r"^Dream-1 run: [0-9.]+; demo: 1\.0$")
        self.assertIn("900 steps", d["run_limit"])

    def test_words(self):
        texts = {n: (HERE / n).read_text() for n in ("task.json", "README.md")}
        banned = [r"\bscaled?\b", r"\bmodel of\b", r"\bmodels?\b", r"\bmock", r"\banalog", r"\btrainer\b",
                  r"stand-in", r"stands for", r"\brepresents?\b", r"\bTBD\b(?! after the GPU runs)", r"<[a-z ]+>",
                  r"\bmeasured later\b", r"\bto be measured\b", r"\bbought\b", r"\bgate\b"]
        for name, text in texts.items():
            for pat in banned:
                self.assertIsNone(re.search(pat, text, flags=re.IGNORECASE), (name, pat))
        for path in HERE.rglob("*"):
            if path.is_file() and path.suffix in (".py", ".md", ".json") and path.name != "cablekit.py":
                text = path.read_text()
                self.assertNotIn(chr(0x2014), text, path.name)
                self.assertNotIn(chr(0x2013), text, path.name)
        self.assertEqual(texts["task.json"].count("our estimates"), 1)


if __name__ == "__main__":
    unittest.main()
