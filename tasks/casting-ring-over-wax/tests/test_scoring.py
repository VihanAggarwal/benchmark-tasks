"""Offline tests for scoring.py, offline_model.py and the hooks in task.py: synthetic ring states and synthetic arm
snapshots, no simulator; the last tests play the demo and its variants through our offline checker when it is there.

    python3 -m unittest discover -s tests -v
"""

import contextlib
import importlib
import io
import json
import math
import os
import pathlib
import subprocess
import sys
import types
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import offline_model as M  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = 20.0
DT = 1.0 / HZ
TIPS = 0.013  # pad tips below the TCP
FX, FY = L.FORMER_X, L.FORMER_Y
BASE = L.BASE_SIZE[2]  # 0.010: the base's top above the table
STEP = BASE + L.STEP_HEIGHT  # 0.016
CROWN_TOP = L.CROWN_TOP - L.TOP  # 0.036
OPEN = 1.0
ON_RING = L.RING_OUTER / L.FULL_GAP  # the opening when the fingers stop on a ring's 60 mm outside
READY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
C = L.RIGHT_RING


def quat(yaw_deg=0.0, tilt_deg=0.0, flip=False, tilt_axis="x"):
    """Rz(yaw) @ R_axis(tilt) [@ Rx(180)] as a quaternion (x, y, z, w)."""
    def q_axis(axis, deg):
        h = math.radians(deg) / 2
        v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
        return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])

    def mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                         w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])

    q = mul(q_axis("z", yaw_deg), q_axis(tilt_axis, tilt_deg))
    return mul(q, q_axis("x", 180.0)) if flip else q


def ring(name, x, y, z0, yaw=0.0, tilt=0.0, flip=False, vel=(0, 0, 0), tilt_axis="x"):
    """A ring state: its lower end's centre at (x, y, TOP + z0), turned and tilted; the world bounding box of its
    real outer box, as the simulator reports it."""
    h = L.ring_height(name)
    q = quat(yaw, tilt, flip, tilt_axis)
    r = S.rotation(q)
    up = r[:, 2] if r[2, 2] >= 0 else -r[:, 2]
    c = np.array([x, y, L.TOP + z0]) + up * h / 2
    half = np.array([L.RING_OUTER_HALF, L.RING_OUTER_HALF, h / 2])
    ext = np.abs(r) @ half
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3), low=c - ext,
                           high=c + ext)


def start(name):
    _, _, x, y = L.RINGS[name]
    return ring(name, x, y, 0.0)


def frame_of(closing, axis=(0.0, 0.0, 1.0)):
    """The gripper's frame (columns: closing axis x, y, approach z from the tool point to the tips) for a gripper
    whose axis points ``axis`` up from the tips and whose fingers close along ``closing``."""
    z = -np.array(axis, float) / np.linalg.norm(axis)
    x = np.array(closing, float)
    x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def gripper(tips, opening=OPEN, axis=(0.0, 0.0, 1.0), closing=None):
    """An arm as rigstate reports it: tip point, TCP 13 mm up the gripper's axis, opening 0..1; with ``closing`` (the
    direction the fingers close along) also its turn, as task.py adds it from gripturn's reading."""
    tips = np.array(tips, float)
    a = np.array(axis, float) / np.linalg.norm(axis)
    arm = rigstate.Arm(tcp=tips + TIPS * a, tips=tips, opening=float(opening))
    if closing is None:
        return arm
    return S.Gripper(arm.tcp, arm.tips, arm.opening, frame_of(closing, axis))


def quat_of(r, order="wxyz"):
    """A unit quaternion of the rotation matrix ``r`` (Shepperd's method), in Isaac Lab's (w, x, y, z) order or as
    (x, y, z, w)."""
    t = float(np.trace(r))
    if t > 0:
        k = 2 * math.sqrt(1 + t)
        w, x, y, z = k / 4, (r[2, 1] - r[1, 2]) / k, (r[0, 2] - r[2, 0]) / k, (r[1, 0] - r[0, 1]) / k
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        k = 2 * math.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        w, x, y, z = (r[2, 1] - r[1, 2]) / k, k / 4, (r[0, 1] + r[1, 0]) / k, (r[0, 2] + r[2, 0]) / k
    elif r[1, 1] > r[2, 2]:
        k = 2 * math.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        w, x, y, z = (r[0, 2] - r[2, 0]) / k, (r[0, 1] + r[1, 0]) / k, k / 4, (r[1, 2] + r[2, 1]) / k
    else:
        k = 2 * math.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
        w, x, y, z = (r[1, 0] - r[0, 1]) / k, (r[0, 2] + r[2, 0]) / k, (r[1, 2] + r[2, 1]) / k, k / 4
    q = np.array([w, x, y, z]) if order == "wxyz" else np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def link_pose(arm, order="wxyz"):
    """The `gripper` link's (position, quaternion) that puts the tool point where ``arm`` (a Gripper) has it."""
    return (np.asarray(arm.tcp) - arm.rot[:, 2] * L.TCP_OFFSET, quat_of(arm.rot, order))


def ready(side):
    return gripper(READY[side] - [0, 0, TIPS])


def arms(left=None, right=None):
    return {"left": left or ready("left"), "right": right or ready("right")}


class Trial:
    """A trial as frames: the three rings (lower-end centre, yaw, tilt, flip) and the two grippers (tip point,
    opening). ``go`` moves numbers linearly to new values over n control steps; velocities come from the motion."""

    def __init__(self):
        self.r = {n: {"x": x, "y": y, "z": 0.0, "yaw": 0.0, "tilt": 0.0, "flip": False}
                  for n, (_, _, x, y) in L.RINGS.items()}
        self.a = {s: {"tips": READY[s] - [0, 0, TIPS], "open": OPEN, "closing": None} for s in ("left", "right")}
        self.frames = []
        self.snap()

    def snap(self):
        rings_ = {n: ring(n, p["x"], p["y"], p["z"], p["yaw"], p["tilt"], p["flip"]) for n, p in self.r.items()}
        if self.frames:
            for n, s in rings_.items():
                s.lin_vel = (s.pos - self.frames[-1][0][n].pos) / DT
        arms_ = {s: gripper(p["tips"], p["open"], closing=p["closing"]) for s, p in self.a.items()}
        self.frames.append((rings_, arms_))

    def go(self, n, rings=None, **sides):
        rings = rings or {}
        a_r = {k: dict(v) for k, v in self.r.items()}
        a_a = {k: {"tips": np.array(v["tips"], float), "open": v["open"]} for k, v in self.a.items()}
        for side, to in sides.items():
            if "closing" in to:  # the gripper's turn (None: not known), from this step on
                self.a[side]["closing"] = to["closing"]
        for i in range(1, n + 1):
            f = i / n
            for name, to in rings.items():
                for k, v in to.items():
                    if k == "flip":
                        self.r[name][k] = v
                    else:
                        self.r[name][k] = (1 - f) * a_r[name][k] + f * v
            for side, to in sides.items():
                if "tips" in to:
                    self.a[side]["tips"] = (1 - f) * a_a[side]["tips"] + f * np.array(to["tips"], float)
                if "open" in to:
                    self.a[side]["open"] = (1 - f) * a_a[side]["open"] + f * to["open"]
            self.snap()
        return self

    def hold(self, n):
        for _ in range(n):
            self.snap()
        return self


def tip_on(name, p, down=0.016):
    """The tip point of a gripper holding ring ``name`` (lower end at p = (x, y, z0)) across its outside, the pads
    ``down`` below its top."""
    return np.array([p["x"], p["y"], L.TOP + p["z"] + L.ring_height(name) - down])


def carry(t, name=C, to=(FX, FY), bottom=BASE + 0.001, land=BASE, release=True, settle=60, side="left",
          tilt=0.0, closing=None, grip=ON_RING):
    """The intended handling, in synthetic form: grip ``name`` where it stands (across its outside, fingers closing
    along ``closing`` if the turn is known, stopping at ``grip``), lift it 9 cm, carry it over ``to``, lower it to
    10 mm over the crown's top, then slowly to ``bottom``; with ``release`` let go (it lands with its lower end at
    ``land``, tilted ``tilt``) and go up."""
    p = t.r[name]
    t.go(20, **{side: {"tips": tip_on(name, p) + [0, 0, 0.06], "closing": closing}})
    t.go(20, **{side: {"tips": tip_on(name, p)}})
    t.go(20, **{side: {"open": grip}})
    for z, n in ((0.09, 20),):
        q = dict(p, z=z)
        t.go(n, rings={name: {"z": z}}, **{side: {"tips": tip_on(name, q)}})
    x, y = to
    q = dict(p, x=x, y=y)
    t.go(30, rings={name: {"x": x, "y": y}}, **{side: {"tips": tip_on(name, q)}})
    for z, n in ((CROWN_TOP + 0.010, 20), (bottom, 40)):
        q = dict(p, z=z)
        t.go(n, rings={name: {"z": z}}, **{side: {"tips": tip_on(name, q)}})
    t.hold(10)
    if not release:
        return t.hold(settle)
    t.go(1, rings={name: {"z": land, "tilt": tilt}})
    t.go(30, **{side: {"open": OPEN}})
    q = dict(p, z=0.09)
    t.go(20, **{side: {"tips": tip_on(name, q)}})
    return t.hold(settle)


def play(trial, tracker=None, use_arms=True):
    t = tracker or Tracker(HZ)
    r = None
    for rings_, arms_ in trial.frames:
        r = t.update(rings_, arms_ if use_arms else None)
    return r, t


def score_of(trial, **kw):
    return play(trial, **kw)[0].score


def seated_rings(name=C, z0=BASE, **kw):
    rings_ = {n: start(n) for n in L.RING_NAMES}
    rings_[name] = ring(name, FX, FY, z0, **kw)
    return rings_


def settle_on(rings_, n=15, arms_=None, tracker=None):
    """Feed the same state n times (still), then return the last result."""
    t = tracker or Tracker(HZ)
    r = None
    for _ in range(n):
        r = t.update(rings_, arms_ or arms())
    return r, t


class Geometry(unittest.TestCase):
    def test_only_the_right_ring_leaves_5_to_8_mm(self):
        self.assertAlmostEqual(L.cover("ring_c"), 0.006, places=9)
        self.assertAlmostEqual(L.cover("ring_b"), -0.002, places=9)
        self.assertAlmostEqual(L.cover("ring_a"), 0.014, places=9)
        self.assertEqual([n for n in L.RINGS if L.COVER_MIN <= L.cover(n) <= L.COVER_MAX], [C])

    def test_tilt_is_folded(self):
        self.assertAlmostEqual(S.tilt_deg(quat(30, 4)), 4.0, places=4)
        self.assertAlmostEqual(S.tilt_deg(quat(0, 0, flip=True)), 0.0, places=4)
        self.assertAlmostEqual(S.tilt_deg(quat(0, 170)), 10.0, places=4)
        self.assertAlmostEqual(S.tilt_deg(quat(0, 90)), 90.0, places=4)

    def test_upside_down_is_the_same_tube(self):
        a, b = ring(C, FX, FY, BASE), ring(C, FX, FY, BASE, flip=True)
        self.assertTrue(np.allclose(a.pos, b.pos))
        self.assertTrue(np.allclose(np.sort(S.corner_heights(a, 0.032)), np.sort(S.corner_heights(b, 0.032))))
        self.assertTrue(S.encloses(b, 0.032))

    def test_centred_ring_clears_the_wax(self):
        s = ring(C, FX, FY, BASE)
        self.assertAlmostEqual(S.wall_wax_gap(s, 0.032), 0.014, places=6)  # +y wall to the crown's +y face
        self.assertFalse(S.wall_touches(s, 0.032))
        self.assertTrue(S.encloses(s, 0.032))
        self.assertTrue(S.on_base(s, 0.032))
        self.assertIsNone(S.wall_wax_gap(start(C), 0.032))  # far away: not even computed

    def test_wall_gap_boundary_minus_y(self):
        crown_face = L.WAX_Y + L.CROWN_SIZE[1] / 2  # 0.092
        for gap, touch in ((0.0004, True), (0.0006, False)):
            y = crown_face + gap - L.RING_INNER_HALF
            s = ring(C, FX, y, BASE)
            self.assertAlmostEqual(S.wall_wax_gap(s, 0.032), gap, places=7)
            self.assertEqual(S.wall_touches(s, 0.032), touch)

    def test_wall_gap_boundary_plus_x(self):
        crown_face = L.WAX_X - L.CROWN_SIZE[0] / 2  # 0.295: the -x wall closing on the crown's -x face
        for gap, touch in ((0.0004, True), (0.0006, False)):
            s = ring(C, crown_face - gap + L.RING_INNER_HALF, FY, BASE)
            self.assertEqual(S.wall_touches(s, 0.032), touch)

    def test_sprue_post_counts(self):
        # ring_b standing low (synthetic): its -y wall's top (24 mm) stays 2 mm under the crown, its inner face
        # 0.3 mm / 0.7 mm from the 4 x 4 mm post round the sprue
        post = L.WAX_Y + L.SPRUE_RADIUS
        for gap, touch in ((0.0003, True), (0.0007, False)):
            s = ring("ring_b", FX, post + gap + L.RING_WALL + L.RING_INNER_HALF, 0.0)
            self.assertEqual(S.wall_touches(s, 0.024), touch)

    def test_tilted_wall_is_judged_as_it_is(self):
        # ring_c 13 mm off in -y: upright its +y wall clears the crown by 1 mm; leaning 10 degrees towards -y, the
        # +y wall's top comes in over the crown
        y = FY - 0.013
        self.assertFalse(S.wall_touches(ring(C, FX, y, BASE), 0.032))
        self.assertTrue(S.wall_touches(ring(C, FX, y, BASE, tilt=10.0), 0.032))

    def test_encloses(self):
        self.assertTrue(S.encloses(ring(C, FX, FY, 0.030), 0.032))  # held over it, bottom 30 mm up
        self.assertFalse(S.encloses(ring(C, FX, FY, CROWN_TOP + 0.001), 0.032))  # above the crown's top
        self.assertFalse(S.encloses(ring(C, FX + 0.030, FY, BASE), 0.032))  # the crown not in the opening
        self.assertFalse(S.encloses(ring(C, FX, FY - 0.016, BASE), 0.032))  # its +y corners outside the 26 mm half

    def test_enclose_tilt_boundary(self):
        for tilt, ok in ((10.0, True), (10.1, False)):
            s = ring(C, FX, L.WAX_Y, 0.020, tilt=tilt)
            self.assertEqual(S.encloses(s, 0.032), ok)
            self.assertFalse(S.wall_touches(s, 0.032))

    def test_corner_band(self):
        for z0, ok in ((0.0120, True), (0.0121, False), (0.0090, True), (0.0089, False), (STEP, False)):
            self.assertEqual(S.on_base(ring(C, FX, FY, z0), 0.032), ok, z0)

    def test_level_means_under_three_degrees(self):
        # tilted about x, the lower corners spread 60 mm x sin(tilt): 2.85 degrees fits the 3 mm band, 3.0 not
        for tilt, ok in ((2.85, True), (3.0, False)):
            spread = L.RING_OUTER * math.sin(math.radians(tilt))
            s = ring(C, FX, FY, 0.009 + spread / 2 + 1e-6, tilt=tilt)
            self.assertEqual(S.on_base(s, 0.032), ok, tilt)

    def test_finger_zone_radii(self):
        # turn not known: the finger box grown 1 mm, swept round the axis
        for gap in (0.060, L.FULL_GAP):
            g = gripper((0.1, 0.1, 0.9), gap / L.FULL_GAP)
            inner, outer = S.zone_radii(g)
            self.assertAlmostEqual(inner, gap / 2 - 0.001)
            self.assertAlmostEqual(outer, math.hypot(gap / 2 + 0.011, 0.021))
            mid = np.array([0.1, 0.1, 0.9 + 0.03])
            for r, inside in ((inner - 1e-4, False), (inner + 1e-4, True), (outer - 1e-4, True), (outer + 1e-4, False)):
                for d in ((1, 0, 0), (0.6, 0.8, 0)):  # any direction round the axis
                    self.assertEqual(S.zone_margin(g, [mid + r * np.array(d)]) <= 0, inside, (gap, r, d))
        g = gripper((0.1, 0.1, 0.9), 0.0)  # closed: a solid round zone
        self.assertLess(S.zone_margin(g, [[0.1, 0.1, 0.9]]), 0)
        for dz, inside in ((-0.0009, True), (-0.0011, False), (0.0709, True), (0.0711, False)):
            self.assertEqual(S.zone_margin(g, [[0.105, 0.1, 0.9 + dz]]) <= 0, inside, dz)

    def test_posed_finger_boxes(self):
        # turn known, fingers closing along x, 30 mm apart: each finger 15..25 mm out along x, +-20 mm along y,
        # 0..70 mm up from the tips, grown 1 mm
        tips = np.array([0.1, 0.1, 0.9])
        g = gripper(tips, 0.030 / L.FULL_GAP, closing=(1, 0, 0))
        mid = tips + [0, 0, 0.030]
        cases = [((0.0139, 0, 0), False), ((0.0141, 0, 0), True), ((0.0259, 0, 0), True),
                 ((0.0261, 0, 0), False), ((-0.020, 0, 0), True), ((0.020, 0.0209, 0), True),
                 ((0.020, 0.0211, 0), False), ((0, 0.020, 0), False), ((0.020, 0, -0.0309), True),
                 ((0.020, 0, -0.0311), False), ((0.020, 0, 0.0409), True), ((0.020, 0, 0.0411), False)]
        for d, inside in cases:
            self.assertEqual(S.zone_margin(g, [mid + d]) <= 0, inside, d)
        self.assertAlmostEqual(S.zone_margin(g, [mid + [0.026, 0, 0]]), 0.0, places=9)
        # the same gripper turned a quarter turn: the fingers are along y now
        g = gripper(tips, 0.030 / L.FULL_GAP, closing=(0, 1, 0))
        self.assertGreater(S.zone_margin(g, [mid + [0.020, 0, 0]]), 0)
        self.assertLess(S.zone_margin(g, [mid + [0, 0.020, 0]]), 0)

    def test_posed_boxes_lie_inside_the_turn_free_zone(self):
        # the fallback is never more lenient: a point in a posed finger box is in the swept zone too
        rng = np.random.default_rng(3)
        for _ in range(300):
            axis = rng.normal(size=3)
            g = gripper(rng.uniform(-0.1, 0.1, 3), rng.uniform(0, 1), axis=axis, closing=rng.normal(size=3))
            pts = g.tips + rng.uniform(-0.09, 0.09, (50, 3))
            posed = [S.zone_margin(g, [q]) <= 0 for q in pts]
            free = [S.sweep_margin(g, [q]) <= 0 for q in pts]
            self.assertTrue(all(f for p_, f in zip(posed, free) if p_))

    def test_gripper_turn_from_the_link_pose(self):
        g = gripper([0.3, 0.1, 0.8], 0.5, closing=(math.cos(0.4), math.sin(0.4), 0))
        arm = rigstate.Arm(g.tcp, g.tips, g.opening)
        for order in ("wxyz", "xyzw"):
            r = S.gripper_turn(arm, [link_pose(g, order)])
            self.assertTrue(np.allclose(r, g.rot, atol=1e-9), order)
        self.assertTrue(np.allclose(S.with_turn({"left": arm}, {"left": [link_pose(g)]})["left"].rot, g.rot))
        pos, q = link_pose(g)
        self.assertIsNone(S.gripper_turn(arm, [(pos + [0.004, 0, 0], q)]))  # 4 mm off: another link
        self.assertIsNotNone(S.gripper_turn(arm, [(pos + [0.0025, 0, 0], q)]))
        tilted = gripper([0.3, 0.1, 0.8], 0.5, axis=(math.sin(0.07), 0, math.cos(0.07)), closing=(0, 1, 0))
        _, q_t = link_pose(tilted)  # its axis 4 degrees off rigstate's
        self.assertIsNone(S.gripper_turn(arm, [(g.tcp - tilted.rot[:, 2] * L.TCP_OFFSET, q_t)]))
        other = gripper([0.3, 0.1, 0.8], 0.5, closing=(0, 1, 0))
        self.assertIsNone(S.gripper_turn(arm, [link_pose(g), link_pose(other)]))  # readings that disagree
        for bad in (None, [], [(pos, np.zeros(4))], [(pos, np.full(4, np.nan))], [(pos[:2], q)], [("x", q)]):
            self.assertIsNone(S.gripper_turn(arm, bad))
        self.assertIsNone(S.gripper_turn(rigstate.Arm(np.zeros(3), np.zeros(3), 1.0), [link_pose(g)]))
        self.assertIsNone(S.with_turn(None, None))
        self.assertIsNone(S.turn_of(S.with_turn({"left": arm}, None)["left"]))
        self.assertIsNone(S.turn_of(SimpleNamespace(rot=np.full((3, 3), np.nan))))
        self.assertIsNone(S.turn_of(SimpleNamespace(rot=2 * np.eye(3))))

    def test_finger_zone_follows_a_tilted_gripper(self):
        g = gripper((0.1, 0.1, 0.9), 0.0, axis=(1.0, 0.0, 0.0))  # lying on its side, fingers pointing to -x
        self.assertLess(S.zone_margin(g, [[0.15, 0.1, 0.9]]), 0)  # 50 mm up its axis
        self.assertGreater(S.zone_margin(g, [[0.1, 0.1, 0.95]]), 0)  # 50 mm above the tips: off its axis

    def test_wax_samples_cover_the_surfaces(self):
        pts = S.WAX_POINTS
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        self.assertAlmostEqual(lo[2], L.SPRUE_BOTTOM)
        self.assertAlmostEqual(hi[2], L.CROWN_TOP)
        self.assertAlmostEqual(hi[1], L.WAX_Y + L.CROWN_SIZE[1] / 2)
        crown = pts[pts[:, 2] >= L.CROWN_BOTTOM - 1e-12]
        for k in range(3):  # 2 mm spacing or finer on every face
            v = np.unique(np.round(crown[:, k], 9))
            self.assertLessEqual(np.diff(v).max(), L.WAX_SAMPLE + 1e-9)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Trial().hold(20))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["tier"], "none")

    def test_the_demo_handling_scores_one(self):
        for closing in (None, (0, 1, 0)):  # the turn not known (offline) and known (fingers along y, as the demo)
            r, t = play(carry(Trial(), closing=closing))
            self.assertEqual(r.score, 1.0)
            self.assertTrue(r.success)
            self.assertFalse(t.touched)
            self.assertGreater(t.min_wall_gap[C], 0.013)
            self.assertGreater(min(t.min_zone.values()), 0.010)
            self.assertEqual(t.turn_unknown["left"], t.steps if closing is None else 1)  # 1: the ready frame

    def test_held_seated_scores_half(self):
        r, _ = play(carry(Trial(), release=False))
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["rings"][C]["held"])

    def test_on_the_step_edge_scores_half(self):
        # released 10 mm off in +y: one wall on the step's edge, the other on the base, tilted about 6 degrees
        tilt = -math.degrees(math.asin(0.006 / 0.056))
        r, t = play(carry(Trial(), to=(FX, FY + 0.010), bottom=STEP + 0.001, land=STEP, tilt=tilt))
        self.assertFalse(t.touched)
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["rings"][C]["on_base"])
        r, _ = play(carry(Trial(), to=(FX, FY + 0.010), bottom=STEP + 0.001, land=STEP))  # level on the step
        self.assertEqual(r.score, 0.5)

    def test_hovering_over_the_wax_scores_half(self):
        r, _ = play(carry(Trial(), bottom=0.030, release=False))
        self.assertEqual(r.score, 0.5)

    def test_wrong_ring_seated_scores_quarter(self):
        for name in ("ring_a", "ring_b"):
            r, t = play(carry(Trial(), name=name))
            self.assertFalse(t.touched, name)
            self.assertEqual(r.score, 0.25, name)
            self.assertEqual(r.details["tier"], "wrong_ring_seated")

    def test_wrong_ring_held_over_the_wax_scores_zero(self):
        r, _ = play(carry(Trial(), name="ring_b", release=False))
        self.assertEqual(r.score, 0.0)

    def test_set_down_beside_the_former_scores_zero(self):
        r, _ = play(carry(Trial(), to=(FX + 0.110, FY), bottom=0.001, land=0.0))
        self.assertEqual(r.score, 0.0)

    def test_every_listed_tier_is_reachable(self):
        got = {score_of(Trial().hold(20)), score_of(carry(Trial())), score_of(carry(Trial(), release=False)),
               score_of(carry(Trial(), name="ring_b"))}
        self.assertEqual(sorted(got), list(S.TIERS))

    def test_hold_gap_boundary(self):
        rings_ = seated_rings()
        tips = [FX, FY, L.TOP + BASE + 0.032 - 0.016]
        diagonal = math.hypot(L.RING_OUTER, L.RING_OUTER)  # 84.9 mm: a grip across the corners
        for gap, held in ((0.087, True), (diagonal, True), (0.089, False), (L.FULL_GAP, False)):
            r, t = settle_on(rings_, arms_=arms(left=gripper(tips, gap / L.FULL_GAP)))
            self.assertFalse(t.touched)
            self.assertEqual(r.details["rings"][C]["held"], held, gap)
            self.assertEqual(r.score, 0.5 if held else 1.0)

    def test_hold_radius_and_height(self):
        rings_ = seated_rings()
        top = L.TOP + BASE + 0.032
        for off, z, held in ((0.044, top - 0.002, True), (0.046, top - 0.002, False), (0.0, top + 0.001, False),
                             (0.0, top - 0.001, True)):
            r, t = settle_on(rings_, arms_=arms(left=gripper([FX + off, FY, z], ON_RING)))
            self.assertFalse(t.touched)
            self.assertEqual(r.details["rings"][C]["held"], held, (off, z))

    def test_on_former_centre_boundary(self):
        # synthetic: ring_b level with its lower end 4.5 mm up (under the edge rule's 5 mm), centre 44 / 46 mm off
        for d, score in ((0.044, 0.0), (0.046, 1.0)):
            rings_ = seated_rings()
            rings_["ring_b"] = ring("ring_b", FX + d, FY, 0.0045)
            r, t = settle_on(rings_)
            self.assertFalse(t.touched)
            self.assertEqual(r.score, score, d)
            self.assertEqual(r.details["tier"], "two_rings_on_former" if score == 0 else "right_ring_seated")

    def test_on_former_edge_boundaries(self):
        # ring_b level, centre 70 mm and more off: its lower end's near edge just over / just past the base's
        # outline (44 mm from the axis), at 5 mm and more up
        for edge, z0, score in ((0.0439, 0.010, 0.0), (0.0441, 0.010, 1.0), (0.0439, 0.0051, 0.0),
                                (0.0439, 0.0049, 1.0)):
            rings_ = seated_rings()
            rings_["ring_b"] = ring("ring_b", FX + edge + L.RING_OUTER_HALF, FY + 0.050, z0)
            r, t = settle_on(rings_)
            self.assertEqual(r.score, score, (edge, z0))
        # perched on top of seated ring_c, centre 46 mm off: on the former
        rings_ = seated_rings()
        rings_["ring_b"] = ring("ring_b", FX + 0.046, FY, BASE + 0.032)
        r, t = settle_on(rings_)
        self.assertEqual(sorted(r.details["rings_on_former"]), ["ring_b", C])
        self.assertEqual(r.score, 0.0)

    def test_still_is_judged_by_path(self):
        rings_ = seated_rings()
        r, t = settle_on(rings_, n=10)  # only 9 steps of history: not yet judged still
        self.assertEqual(r.score, 0.5)
        r = t.update(rings_, arms())
        self.assertEqual(r.score, 1.0)
        # creeping 0.25 mm a step (2.5 mm in 0.5 s): not still, whatever velocity it reports
        t = Tracker(HZ)
        for k in range(20):
            rr = seated_rings(C, BASE)
            rr[C] = ring(C, FX + 0.00025 * k - 0.0025, FY, BASE)
            r = t.update(rr, arms())
        self.assertFalse(r.details["rings"][C]["still"])
        self.assertEqual(r.score, 0.5)
        # jitter of +-0.05 mm with a chattering 40 mm/s velocity: still
        t, rng = Tracker(HZ), np.random.default_rng(1)
        for k in range(30):
            rr = seated_rings()
            rr[C] = ring(C, FX + rng.uniform(-5e-5, 5e-5), FY + rng.uniform(-5e-5, 5e-5), BASE, vel=(0.04, 0, 0))
            r = t.update(rr, arms())
        self.assertEqual(r.score, 1.0)

    def test_arms_unreadable(self):
        rings_ = seated_rings()
        r, t = settle_on(rings_, n=99)
        r = t.update(rings_, None)  # 1 step of 100: 1%, allowed, but the ring counts as held on it
        self.assertEqual(t.arms_unavailable, 1)
        self.assertEqual(r.score, 0.5)
        r = t.update(rings_, arms())
        self.assertEqual(r.score, 1.0)
        r = t.update(rings_, None)  # 2 of 101: over 1%
        r = t.update(rings_, arms())
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["tier"], "arms_unreadable")
        _, t = play(carry(Trial()), use_arms=False)
        self.assertEqual(t.result(*carry(Trial()).frames[-1]).score, 0.0)

    def test_score_is_idempotent(self):
        tr = carry(Trial())
        r, t = play(tr)
        a = t.result(*tr.frames[-1])
        b = t.result(*tr.frames[-1])
        self.assertEqual((a.score, a.details), (b.score, b.details))
        self.assertEqual(t.steps, len(tr.frames))


class Exploits(unittest.TestCase):
    def test_pushing_it_across_the_base_touches_the_wax(self):
        # slid along the table to the base's edge, then up and across the base to the axis without lifting
        t = Trial()
        t.go(40, rings={C: {"x": FX, "y": FY - 0.080}}, left={"tips": [FX, FY - 0.080 - 0.040, L.TOP + 0.015]})
        t.go(5, rings={C: {"z": BASE, "y": FY - 0.074}})
        t.go(40, rings={C: {"y": FY}}, left={"tips": [FX, FY - 0.035, L.TOP + 0.020]})
        r, tr = play(t.hold(30))
        self.assertTrue(tr.touched)
        self.assertEqual(tr.touch["cause"], C)
        self.assertEqual(r.score, 0.0)

    def test_dropped_onto_the_crown_is_latched(self):
        # set down 16 mm off in -y: its +y wall comes down on the crown; lifted off and seated properly: still 0
        t = carry(Trial(), to=(FX, FY - 0.016), bottom=CROWN_TOP, release=False, settle=5)
        p = t.r[C]
        t.go(20, rings={C: {"z": 0.09}}, left={"tips": tip_on(C, dict(p, z=0.09))})
        t.go(10, rings={C: {"y": FY}}, left={"tips": tip_on(C, dict(p, y=FY, z=0.09))})
        t.go(40, rings={C: {"z": BASE + 0.001}}, left={"tips": tip_on(C, dict(p, y=FY, z=BASE + 0.001))})
        t.go(1, rings={C: {"z": BASE}}).go(30, left={"open": OPEN})
        t.go(20, left={"tips": tip_on(C, dict(p, y=FY, z=0.09))}).hold(60)
        r, tr = play(t)
        self.assertTrue(tr.touched)
        self.assertTrue(r.details["rings"][C]["seated"])
        self.assertEqual(r.score, 0.0)

    def test_seat_the_short_ring_then_stack_the_right_one_on_it(self):
        t = carry(Trial(), name="ring_b")
        t = carry(t, name=C, bottom=BASE + 0.024 + 0.001, land=BASE + 0.024)
        r, tr = play(t)
        self.assertFalse(tr.touched)
        self.assertTrue(r.details["rings"][C]["encloses"])  # it would be 0.5 on its own
        self.assertEqual(sorted(r.details["rings_on_former"]), ["ring_b", C])
        self.assertEqual(r.score, 0.0)

    def test_holding_instead_of_placing(self):
        self.assertEqual(score_of(carry(Trial(), release=False)), 0.5)

    def test_open_gripper_lowered_beside_the_wax(self):
        # fully open, its axis 25 mm to the +y side of the crown's centre, tips down at the step: a finger is on
        # the crown at some turn
        t = Trial().go(30, left={"tips": [L.WAX_X, L.WAX_Y + 0.025, L.TOP + 0.020]}).hold(5)
        r, tr = play(carry(t))
        self.assertTrue(tr.touched)
        self.assertEqual(tr.touch["cause"], "left gripper")
        self.assertEqual(r.score, 0.0)

    def test_open_fingers_straddling_the_wax_do_not_touch_it(self):
        t = Trial().go(30, left={"tips": [L.WAX_X, L.WAX_Y, L.TOP + 0.020]}).hold(5)
        _, tr = play(t)
        self.assertFalse(tr.touched)  # the crown sits between the open fingers, 47 mm each side

    def test_closed_gripper_poking_the_crown(self):
        t = Trial().go(30, left={"tips": [L.WAX_X + 0.012, L.WAX_Y, L.TOP + 0.030], "open": 0.0}).hold(5)
        _, tr = play(t)
        self.assertTrue(tr.touched)

    def test_idle_right_arm_reaching_in(self):
        t = Trial().go(30, right={"tips": [L.WAX_X, L.WAX_Y - 0.030, L.TOP + 0.030], "open": 0.3}).hold(5)
        _, tr = play(t)
        self.assertTrue(tr.touched)
        self.assertEqual(tr.touch["cause"], "right gripper")

    def test_upside_down_right_ring_seated(self):
        t = carry(Trial(), release=False, settle=0)
        t.go(1, rings={C: {"z": BASE, "flip": True}}).go(30, left={"open": OPEN})
        t.go(20, left={"tips": [FX, FY, L.TOP + 0.12]}).hold(60)
        r, _ = play(t)
        self.assertEqual(r.score, 1.0)

    def test_leaning_onto_the_crown(self):
        # resting tilted 20 degrees with a wall across the crown's top
        rings_ = seated_rings()
        rings_[C] = ring(C, FX, FY - 0.020, BASE, tilt=20.0)
        r, t = settle_on(rings_)
        self.assertTrue(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_grip_across_the_corners_never_released(self):
        # E1: ring_c gripped across two corners (fingers stop 84.9 mm apart), set on the base and held there:
        # still held, 0.5 (with 70 mm as the limit this was 1.0, and the trial ended with the ring still gripped)
        diagonal = math.hypot(L.RING_OUTER, L.RING_OUTER) / L.FULL_GAP
        for bottom, land in ((BASE, BASE), (BASE + 0.002, BASE + 0.002)):  # E1b: hovering 2 mm over the base
            for closing in (None, (1, 1, 0)):
                t = carry(Trial(), bottom=bottom, release=False, settle=80, closing=closing, grip=diagonal)
                r, tr = play(t)
                self.assertFalse(tr.touched)
                self.assertTrue(r.details["rings"][C]["held"])
                self.assertEqual(r.score, 0.5, (bottom, closing))

    def test_grip_off_the_flats_still_closed(self):
        # E9: ring_c on the base with a gripper still closed on it 15 degrees off its flats (73.5 mm)
        w = L.RING_OUTER * (math.cos(math.radians(15)) + math.sin(math.radians(15)))
        tips = [FX, FY, L.TOP + BASE + 0.032 - 0.016]
        for closing in (None, (math.sin(math.radians(15)), math.cos(math.radians(15)), 0)):
            r, t = settle_on(seated_rings(), arms_=arms(left=gripper(tips, w / L.FULL_GAP, closing=closing)))
            self.assertFalse(t.touched)
            self.assertTrue(r.details["rings"][C]["held"])
            self.assertEqual(r.score, 0.5)

    def test_pinch_at_a_walls_end(self):
        # E3: a closed pinch on the -x wall at its end, the tip point 41 mm from the axis: held
        for along in (0.026, 0.030):
            tips = [FX - 0.028, FY + along, L.TOP + BASE + 0.032 - 0.016]
            g = gripper(tips, 0.004 / L.FULL_GAP, closing=(1, 0, 0))
            r, t = settle_on(seated_rings(), arms_=arms(left=g))
            self.assertFalse(t.touched)
            self.assertTrue(r.details["rings"][C]["held"], along)
            self.assertEqual(r.score, 0.5)

    def single_wall_pinch(self, dx, dy, closing, open_to=0.016):
        """E2: ring_c pinched by one wall (the tip point on the wall's centre line, 16 mm below the ring's top,
        fingers 4 mm apart), carried centred over the wax, set on the base, opened to ``open_to``, lifted."""
        t = Trial()
        p = t.r[C]

        def tips_at(pp):
            return np.array([pp["x"] + dx, pp["y"] + dy, L.TOP + pp["z"] + L.ring_height(C) - 0.016])

        t.go(20, left={"tips": tips_at(p) + [0, 0, 0.06], "closing": closing}).go(20, left={"tips": tips_at(p)})
        t.go(20, left={"open": 0.004 / L.FULL_GAP})
        t.go(20, rings={C: {"z": 0.09}}, left={"tips": tips_at(dict(p, z=0.09))})
        t.go(30, rings={C: {"x": FX, "y": FY}}, left={"tips": tips_at(dict(p, x=FX, y=FY, z=0.09))})
        for z, n in ((CROWN_TOP + 0.010, 20), (BASE + 0.001, 40)):
            t.go(n, rings={C: {"z": z}}, left={"tips": tips_at(dict(p, x=FX, y=FY, z=z))})
        t.go(1, rings={C: {"z": BASE}})
        t.go(20, left={"open": open_to / L.FULL_GAP})
        t.go(30, left={"tips": tips_at(dict(p, x=FX, y=FY, z=0.09))}).hold(60)
        return play(t)

    def test_single_wall_pinch_with_the_turn_known(self):
        # the inner finger's face toward the axis is 10 mm in from the wall's inner face less the opening's growth:
        # opened to 16 mm, 10 mm from the axis, 5 mm clear of the crown's x faces and 10 mm of its -y face (no
        # touch, 1.0); on the +y wall, opened only to 8 mm, 14 mm from the axis, 2 mm clear of the crown's +y face
        walls = {"-x": (-0.028, 0.0, (1, 0, 0), 0.016), "+x": (0.028, 0.0, (1, 0, 0), 0.016),
                 "-y": (0.0, -0.028, (0, 1, 0), 0.016), "+y": (0.0, 0.028, (0, 1, 0), 0.008)}
        for wall, (dx, dy, closing, open_to) in walls.items():
            r, tr = self.single_wall_pinch(dx, dy, closing, open_to)
            self.assertFalse(tr.touched, wall)
            self.assertGreater(tr.min_zone["left"], 0.0009, wall)
            self.assertEqual(r.score, 1.0, wall)
        # the same pinch with the turn unknown (the fallback): a finger could be on the crown at some turn
        r, tr = self.single_wall_pinch(-0.028, 0.0, None)
        self.assertTrue(tr.touched)
        self.assertEqual(r.score, 0.0)

    def test_pinch_opened_into_the_crown(self):
        # the +y wall pinch opened to 16 mm: the inner finger moves to 10 mm from the axis, 2 mm into the crown
        # (whose +y face is 12 mm from the axis): a real touch, 0
        r, tr = self.single_wall_pinch(0.0, 0.028, (0, 1, 0), 0.016)
        self.assertTrue(tr.touched)
        self.assertEqual(tr.touch["cause"], "left gripper")
        self.assertEqual(r.score, 0.0)

    def test_closed_gripper_beside_the_crown(self):
        # E6: closed, its axis 20 mm from the crown's +x face, tips at crown height: fingers closing along x reach
        # 11 mm (no touch); turned a quarter, their 21 mm half width reaches the face (touch); turn unknown: touch
        tips = [L.WAX_X + 0.005 + 0.020, L.WAX_Y, L.TOP + 0.030]
        for closing, touched in (((1, 0, 0), False), ((0, 1, 0), True), (None, True)):
            t = Trial().go(30, left={"tips": tips, "open": 0.0, "closing": closing}).hold(5)
            _, tr = play(t)
            self.assertEqual(tr.touched, touched, closing)

    def test_second_ring_leaning_on_the_base(self):
        # E4: ring_b leaning with its -x wall on the base's margin (lower corners 9.9 mm up, 36 mm off the axis),
        # its other wall on the table, centre 68 mm off, beside seated ring_c: two rings on the former, 0
        tilt = math.degrees(math.atan(0.010 / 0.060))
        rings_ = seated_rings()
        rings_["ring_b"] = ring("ring_b", FX + 0.066, FY, 0.005, tilt=tilt, tilt_axis="y")
        z = {round(float(c[0] - FX), 3): round(float(c[2] - L.TOP), 4) for c in S.bottom_corners(rings_["ring_b"], 0.024)}
        self.assertAlmostEqual(z[0.036], 0.0099, places=3)
        r, t = settle_on(rings_)
        self.assertFalse(t.touched)
        self.assertEqual(sorted(r.details["rings_on_former"]), ["ring_b", C])
        self.assertEqual(r.score, 0.0)
        # standing on the table against the base's side instead: not on the former
        rings_["ring_b"] = ring("ring_b", FX + L.BASE_HALF + L.RING_OUTER_HALF, FY, 0.0)
        r, _ = settle_on(rings_)
        self.assertEqual(r.score, 1.0)

    def test_wrong_ring_then_right_ring(self):
        # ring_b seated first (0.25), lifted off and put back on its start, then ring_c seated: 1.0
        t = carry(Trial(), name="ring_b")
        p = t.r["ring_b"]
        _, _, sx, sy = L.RINGS["ring_b"]
        t.go(20, left={"tips": tip_on("ring_b", p)}).go(20, left={"open": ON_RING})
        t.go(20, rings={"ring_b": {"z": 0.09}}, left={"tips": tip_on("ring_b", dict(p, z=0.09))})
        t.go(30, rings={"ring_b": {"x": sx, "y": sy}}, left={"tips": tip_on("ring_b", dict(p, x=sx, y=sy, z=0.09))})
        t.go(30, rings={"ring_b": {"z": 0.0}}, left={"tips": tip_on("ring_b", dict(p, x=sx, y=sy, z=0.0))})
        t.go(20, left={"open": OPEN}).go(10, left={"tips": tip_on("ring_b", dict(p, x=sx, y=sy, z=0.08))})
        r, _ = play(carry(t))
        self.assertEqual(r.score, 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        good = seated_rings()
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([FX, FY, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.zeros(3), high=np.zeros(3))
        zero_q = SimpleNamespace(**{**vars(good[C]), "quat_xyzw": np.zeros(4)})
        no_attr = SimpleNamespace(pos=good[C].pos)
        for bad in (nan, inf, zero_q, no_attr, None):
            self.assertFalse(S.valid(bad))
            r, t = settle_on(good)
            self.assertEqual(r.score, 1.0)
            before = (t.touched, t.good_steps, t.arms_unavailable, [list(map(tuple, h)) for h in t.hist.values()])
            rr = dict(good, **{C: bad})
            r = t.update(rr, arms())
            self.assertEqual((r.score, r.success), (0.0, False))
            self.assertEqual(t.result(rr, arms()).score, 0.0)
            self.assertEqual((t.touched, t.good_steps, t.arms_unavailable,
                              [list(map(tuple, h)) for h in t.hist.values()]), before)
            r = t.update(good, arms())
            self.assertEqual(r.score, 1.0)  # the next good state is scored as before
        missing = {n: s for n, s in good.items() if n != C}
        self.assertEqual(Tracker(HZ).update(missing, arms()).score, 0.0)
        self.assertEqual(Tracker(HZ).update(None, arms()).score, 0.0)

    def test_bad_arm_readings(self):
        rings_ = seated_rings()
        bad = [None, {}, {"left": ready("left")}, {"left": ready("left"), "right": SimpleNamespace(tcp=[0, 0, 0])},
               {"left": gripper([np.nan, 0, 0]), "right": ready("right")},
               {"left": rigstate.Arm(np.zeros(3), np.zeros(3), float("nan")), "right": ready("right")}]
        for a in bad:
            self.assertFalse(S.arms_valid(a))
            t = Tracker(HZ)
            r = t.update(rings_, a)
            self.assertEqual(t.arms_unavailable, 1)
            self.assertIn(r.score, S.TIERS)

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(7)
        t = Tracker(HZ)
        for _ in range(300):
            rr = {}
            for n in L.RING_NAMES:
                rr[n] = ring(n, rng.uniform(-0.5, 1.0), rng.uniform(-0.7, 0.7), rng.uniform(-0.2, 0.3),
                             yaw=rng.uniform(-180, 180), tilt=rng.uniform(0, 180), tilt_axis=str(rng.choice(["x", "y"])))
            g = [gripper(rng.uniform(0.0, 1.0, 3) + [0, 0, 0.7], rng.uniform(-0.5, 1.5), axis=rng.normal(size=3))
                 for _ in range(2)]
            r = t.update(rr, {"left": g[0], "right": g[1]})
            self.assertIn(r.score, S.TIERS)
            json.dumps(r.details, default=str)

    def test_states_without_mechkit_fields(self):
        # a plain ObjectState (no origin, no source) is scored the same way
        rings_ = seated_rings()
        r, _ = settle_on(rings_)
        self.assertEqual(r.score, 1.0)


class Reset(unittest.TestCase):
    def touched_trial(self):
        t = carry(Trial(), to=(FX, FY - 0.016), bottom=CROWN_TOP, release=False, settle=5)
        _, tr = play(t)
        self.assertTrue(tr.touched)
        return t, tr

    def test_teleport_back_to_start_resets(self):
        _, tr = self.touched_trial()
        r = tr.update({n: start(n) for n in L.RING_NAMES}, arms())
        self.assertFalse(tr.touched)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)
        r, _ = play(carry(Trial()), tr)  # the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_jittered_teleport_resets(self):
        _, tr = self.touched_trial()
        jit = {"ring_a": (0.03, -0.03), "ring_b": (-0.02, 0.025), "ring_c": (0.028, 0.03)}
        rr = {n: ring(n, L.RINGS[n][2] + dx, L.RINGS[n][3] + dy, 0.0, yaw=0.0) for n, (dx, dy) in jit.items()}
        tr.update(rr, arms())
        self.assertFalse(tr.touched)

    def test_gradual_return_is_not_a_reset(self):
        t, tr = self.touched_trial()
        _, _, sx, sy = L.RINGS[C]
        p = t.r[C]
        t.frames = t.frames[-1:]
        t.go(20, rings={C: {"z": 0.09}}, left={"tips": tip_on(C, dict(p, z=0.09))})
        t.go(40, rings={C: {"x": sx, "y": sy}}, left={"tips": tip_on(C, dict(p, x=sx, y=sy, z=0.09))})
        t.go(30, rings={C: {"z": 0.0}}, left={"tips": tip_on(C, dict(p, x=sx, y=sy, z=0.0))})
        t.go(20, left={"open": OPEN}).hold(10)
        play(t, tr)
        self.assertTrue(tr.touched)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        _, tr = self.touched_trial()
        rr = {n: start(n) for n in L.RING_NAMES}
        rr[C] = ring(C, 0.45, -0.2, 0.0)
        n = tr.steps
        tr.update(rr, arms())
        self.assertTrue(tr.touched)
        self.assertEqual(tr.steps, n + 1)


class Model(unittest.TestCase):
    """offline_model: a released ring slides down round the wax onto what is under its walls."""

    def checker_states(self, where=None):
        out = {}
        for n in L.RING_NAMES:
            h, _, x, y = L.RINGS[n]
            x, y, z0 = (where or {}).get(n, (x, y, 0.0))
            s = ring(n, x, y, z0)
            s.origin = s.pos - [0, 0, h / 2]
            s.source = "offline"
            out[n] = s
        return out

    def release_at(self, x, y, z0=0.011, name=C):
        m = M.RingModel(HZ)
        tips = [x, y, L.TOP + z0 + L.ring_height(name) - 0.016]
        held = self.checker_states({name: (x, y, z0)})
        m.step(held, arms(left=gripper(tips, 0.052 / L.FULL_GAP)))  # gripped where the checker carries it
        m.step(held, arms(left=gripper(tips, 0.052 / L.FULL_GAP)))
        self.assertEqual(m.rings[name]["holder"], "left")
        dropped = self.checker_states({name: (x, y, L.SPRUE_TOP - L.TOP)})  # the checker's box lands on the sprue
        out = None
        for _ in range(20):
            out = m.step(dropped, arms(left=gripper(tips, OPEN)))
        self.assertIsNone(m.rings[name]["holder"])
        return out[0][name]

    def test_released_round_the_wax_lands_on_the_base(self):
        s = self.release_at(FX, FY)
        self.assertAlmostEqual(float(S.corner_heights(s, 0.032).min()), BASE, places=9)
        self.assertTrue(np.allclose(s.lin_vel, 0.0))

    def test_off_centre_lands_on_the_step(self):
        s = self.release_at(FX, FY + 0.010, z0=0.017)
        self.assertAlmostEqual(float(S.corner_heights(s, 0.032).min()), STEP, places=9)

    def test_over_the_crown_lands_on_it(self):
        s = self.release_at(FX, FY - 0.020, z0=0.037)
        self.assertAlmostEqual(float(S.corner_heights(s, 0.032).min()), CROWN_TOP, places=9)

    def test_on_the_table(self):
        s = self.release_at(0.20, -0.10, z0=0.005)
        self.assertAlmostEqual(float(S.corner_heights(s, 0.032).min()), 0.0, places=9)

    def test_onto_another_ring(self):
        m = M.RingModel(HZ)
        m.step(self.checker_states(), arms())
        m.rings["ring_b"]["origin"] = np.array([FX, FY, L.BASE_TOP])  # seated
        self.assertAlmostEqual(m.support(C, FX, FY, 0.0, L.TOP + 0.06)[0], L.BASE_TOP + 0.024)

    def test_falls_with_its_real_speed(self):
        m = M.RingModel(HZ)
        tips = [0.2, -0.1, L.TOP + 0.05 + 0.016]
        m.step(self.checker_states({C: (0.2, -0.1, 0.05)}), arms(left=gripper(tips, 0.5)))
        speeds = [-float(m.step(self.checker_states({C: (0.2, -0.1, 0.0)}), arms(left=gripper(tips, OPEN)))[0][C]
                         .lin_vel[2]) for _ in range(5)]
        self.assertGreater(max(speeds), 0.5)

    def test_same_step_twice_is_cached(self):
        m = M.RingModel(HZ)
        a = arms()
        st = self.checker_states()
        self.assertIs(m.step(st, a), m.step(st, a))
        self.assertEqual(m.steps, 1)

    def test_holder_opening_is_the_rings_width(self):
        m = M.RingModel(HZ)
        tips = [FX, FY, L.TOP + 0.03]
        _, seen = m.step(self.checker_states({C: (FX, FY, 0.011)}), arms(left=gripper(tips, 0.052 / L.FULL_GAP)))
        self.assertAlmostEqual(seen["left"].opening, ON_RING)


class _Scene(dict):
    """A stand-in for the env's scene: robots by name, and the env origins."""

    def __init__(self, robots, origin):
        super().__init__(robots)
        self.env_origins = np.array([origin], float)


class GripTurn(unittest.TestCase):
    """gripturn.py against a stand-in embodiment: it wraps _task_step and reset and reads each arm's gripper link
    before the step runs; the scorer recovers the turn from what it read."""

    ORIGIN = (1.0, -2.0, 0.0)

    def setUp(self):
        import gripturn
        self.gt = gripturn
        self.saved_mod = sys.modules.get("yam_arena.inspect_embodiment")
        self.saved_state = dict(gripturn._STATE)
        gripturn._STATE.update(bodies=None, error=None, installed=False)
        self.grips = {"left": gripper([0.30, 0.10, 0.80], 0.6, closing=(0.6, 0.8, 0)),
                      "right": gripper([0.35, -0.20, 0.85], 1.0, axis=(0.2, 0, 1), closing=(0, 1, 0))}

    def tearDown(self):
        if self.saved_mod is None:
            sys.modules.pop("yam_arena.inspect_embodiment", None)
        else:
            sys.modules["yam_arena.inspect_embodiment"] = self.saved_mod
        self.gt._STATE.clear()
        self.gt._STATE.update(self.saved_state)

    def embodiment(self, names=("base", "link5", "gripper", "tip_left"), layout="pos_quat", order="wxyz"):
        robots = {}
        for side, g in self.grips.items():
            n = len(names)
            pos, quat = np.zeros((1, n, 3)), np.tile([0.0, 0.0, 0.0, 1.0], (1, n, 1))
            if "gripper" in names:
                k = names.index("gripper")
                p_, q_ = link_pose(g, order)
                pos[0, k], quat[0, k] = p_ + self.ORIGIN, q_
            data = SimpleNamespace()
            if layout == "pos_quat":
                data.body_link_pos_w, data.body_link_quat_w = pos, quat
            else:
                data.body_link_pose_w = np.concatenate([pos, quat], axis=2)
            robots[f"{side}_robot"] = SimpleNamespace(body_names=list(names), data=data)
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene=_Scene(robots, self.ORIGIN)))
        seen = []
        gt = self.gt

        class YamArenaEmbodiment:
            def __init__(self):
                self._env = env

            def _task_step(self, action):
                seen.append(gt.bodies())  # what the scorer would see in this step's hooks
                return "stepped"

            def reset(self):
                return "reset"

        module = types.ModuleType("yam_arena.inspect_embodiment")
        module.YamArenaEmbodiment = YamArenaEmbodiment
        sys.modules["yam_arena.inspect_embodiment"] = module
        return YamArenaEmbodiment, seen

    def turns(self, bodies):
        arms_ = {s: rigstate.Arm(g.tcp, g.tips, g.opening) for s, g in self.grips.items()}
        return S.with_turn(arms_, bodies)

    def test_reads_the_link_before_the_step(self):
        for layout, order in (("pos_quat", "wxyz"), ("pose7", "xyzw")):
            self.gt._STATE.update(bodies=None, error=None, installed=False)
            cls, seen = self.embodiment(layout=layout, order=order)
            self.assertTrue(self.gt.install())
            self.assertTrue(self.gt.install())  # idempotent
            emb = cls()
            self.assertEqual(emb._task_step(None), "stepped")
            self.assertIsNotNone(seen[-1])
            got = self.turns(seen[-1])
            for side, g in self.grips.items():
                self.assertTrue(np.allclose(got[side].rot, g.rot, atol=1e-9), (layout, side))
            self.assertEqual(emb.reset(), "reset")
            self.assertIsNone(self.gt.error())
        step = cls._task_step
        self.assertTrue(getattr(step, "_gripturn_wrapped", False))
        self.assertFalse(getattr(step.__wrapped__, "_gripturn_wrapped", False))  # wrapped once

    def test_a_missing_link_leaves_the_turn_unknown(self):
        cls, seen = self.embodiment(names=("base", "link5", "tip_left"))
        self.assertTrue(self.gt.install())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(cls()._task_step(None), "stepped")  # the step still runs
            cls()._task_step(None)
        self.assertEqual(err.getvalue().count("[gripturn]"), 1)  # said once
        self.assertIsNone(seen[-1])
        self.assertIn("gripper", self.gt.error())
        got = self.turns(self.gt.bodies())
        self.assertIsNone(S.turn_of(got["left"]))

    def test_offline_nothing_is_hooked(self):
        sys.modules.pop("yam_arena.inspect_embodiment", None)
        self.assertFalse(self.gt.install())
        self.assertIsNone(self.gt.bodies())


def _stub_dir():
    for p in HERE.parents:
        stub = p / "tools" / "offline" / "yam_stub"
        if stub.is_dir():
            return stub
    return None


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hooks, demo and text."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def sim_states(self, where=None, source="sim"):
        out = {}
        for spec in self.task.RINGS:
            x, y, z = (where or {}).get(spec.name, spec.pos)
            out[spec.name] = self.mk.state_from_poses(spec, (x, y, z, 0, 0, 0, 1), source=source)
        return out

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(names, ["former_base", "former_step", "sprue", "wax_crown", "ring_a", "ring_b", "ring_c"])
        for o in self.task.OBJECTS[:4]:
            self.assertIsNone(o.mass)
        for spec in self.task.RINGS:
            h, m, x, y = L.RINGS[spec.name]
            self.assertEqual(spec.kind, "rigid")
            self.assertAlmostEqual(spec.mass, m)
            self.assertLessEqual(spec.mass, 0.3)
            vol = sum(s.volume() for s in spec.shapes)
            self.assertAlmostEqual(vol * 7900, m, delta=0.006)  # steel of that volume
            lo, hi = spec.world_bounds()
            self.assertTrue(np.allclose(lo, [x - 0.03, y - 0.03, L.TOP]))
            self.assertTrue(np.allclose(hi, [x + 0.03, y + 0.03, L.TOP + h]))
            st = self.sim_states()[spec.name]
            self.assertTrue(np.allclose(st.pos, start(spec.name).pos))  # its centre is its geometric centre
            self.assertTrue(np.allclose(S.corner_heights(st, h), 0.0))
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 2.5)

    def feed(self, left=None, right=None):
        snap = {}
        for side, a in (("left", left or ready("left")), ("right", right or ready("right"))):
            snap[side] = {"tcp": a.tcp, "tips": a.tips, "opening": a.opening}
        rigstate.offline_feed(snap)

    def test_hooks_on_sim_states(self):
        rigstate.offline_reset()
        seated = self.sim_states({C: (FX, FY, L.BASE_TOP)})
        for _ in range(10):
            self.feed()
            ok = self.task.success(seated)
        self.assertFalse(ok)  # 'still' needs 0.5 s of positions: 11 samples, this is the 10th
        self.assertEqual(self.task.score(seated), 0.5)
        self.feed()
        self.assertTrue(self.task.success(seated))
        self.assertEqual(self.task.score(seated), 1.0)
        rigstate.offline_reset()  # a reset clears the trial
        self.assertEqual(self.task._TRACKER.steps, 0)

    def test_mechkit_reset_signal(self):
        rigstate.offline_reset()
        self.feed()
        self.task.success(self.sim_states())
        self.assertEqual(self.task._TRACKER.steps, 1)
        self.mk.offline_reset()
        self.feed()
        self.task.success(self.sim_states())
        self.assertEqual(self.task._TRACKER.steps, 1)

    def test_offline_states_use_the_model(self):
        rigstate.offline_reset()
        st = self.sim_states(source="offline")
        self.feed()
        self.task.success(st)
        self.assertEqual(self.task._MODEL.steps, 1)
        self.task.score(st)  # the same step: the model is not stepped again
        self.assertEqual(self.task._MODEL.steps, 1)

    def test_trace_rows(self):
        path = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"casting_ring_trace_{os.getpid()}.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(path)
            rigstate.offline_reset()
            self.feed()
            self.task.success(self.sim_states())
            row = json.loads(path.read_text().splitlines()[-1])
            for k in ("tier", "touched", "rings", "arms", "arms_unavailable", "left_tips"):
                self.assertIn(k, row)
            for k in ("tilt_deg", "low_corner_mm", "high_corner_mm", "axis_offset_mm", "wall_wax_gap_mm", "held",
                      "encloses", "seated"):
                self.assertIn(k, row["rings"][C])
            os.environ["TASK_TRACE"] = "/nonexistent-dir/trace.jsonl"  # unwritable: ignored
            self.feed()
            self.task.success(self.sim_states())
        finally:
            os.environ.pop("TASK_TRACE", None)
            if path.exists():
                path.unlink()

    def test_turn_reaches_the_scorer_and_the_trace(self):
        import gripturn
        path = pathlib.Path(os.environ.get("TMPDIR", "/tmp")) / f"casting_ring_turn_{os.getpid()}.jsonl"
        saved = dict(gripturn._STATE)
        try:
            os.environ["TASK_TRACE"] = str(path)
            rigstate.offline_reset()
            g = gripper([FX, FY + 0.2, L.TOP + 0.1], 1.0, closing=(0, 1, 0))
            self.feed(left=g)
            gripturn._STATE["bodies"] = {"left": [link_pose(g)], "right": []}
            self.task.success(self.sim_states())
            row = json.loads(path.read_text().splitlines()[-1])
            self.assertTrue(row["arms"]["left"]["turn_known"])
            self.assertFalse(row["arms"]["right"]["turn_known"])
            self.assertEqual(row["arms_turn_unknown"], {"left": 0, "right": 1})
            self.assertTrue(np.allclose(np.abs(row["left_closing_axis"]), [0, 1, 0], atol=1e-4))
            self.assertNotIn("right_closing_axis", row)
        finally:
            os.environ.pop("TASK_TRACE", None)
            gripturn._STATE.clear()
            gripturn._STATE.update(saved)
            if path.exists():
                path.unlink()

    def test_demo_switches(self):
        keys = ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM")
        saved = {k: os.environ.pop(k, None) for k in keys}
        try:
            plan = self.task.demo()
            self.assertEqual(sorted(plan), ["left"])
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(self.task.demo(), {})
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(sorted(self.task.demo()), ["left"])
            os.environ.pop("DEMO_IDLE_ARM")
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(self.task.demo(), {})
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_demo_numbers(self):
        plan = self.task.ring_plan()
        by = {w.label: w for w in plan}
        grip = by["grip it across its outside"]
        self.assertAlmostEqual(grip.finger, -(0.060 - 0.008) / 2)
        self.assertAlmostEqual(grip.pos[2] - L.TOP, 0.029)
        seat = by["lower it round the wax, slowly, to 1 mm over the base"]
        self.assertAlmostEqual(seat.pos[2] - L.TOP, 0.040)
        self.assertGreaterEqual(seat.duration, 2.0)
        self.assertAlmostEqual(by["down: its bottom 10 mm over the crown's top"].pos[2] - L.TOP, 0.075)
        self.assertGreaterEqual(by["let go"].duration, 1.5)
        self.assertGreaterEqual(grip.duration, 1.5)
        self.assertAlmostEqual(sum(w.duration for w in plan), 21.8)
        for w in plan:
            self.assertEqual(w.yaw, 0.0)

    def test_instruction_is_the_same_everywhere(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        self.assertEqual(readme.count(f'--instruction "{self.task.INSTRUCTION}"'), 2)
        self.assertLessEqual(len(self.task.INSTRUCTION), 490)

    def test_text_numbers(self):
        spec = json.loads((HERE / "task.json").read_text())
        text = spec["setup"] + " " + spec["scoring"]
        for s in ("88 x 88", "36 mm across", "6 mm tall", "4 mm", "10 x 12", "36 mm", "60 x 60", "52 x 52",
                  "0.28 kg", "0.17 kg", "0.23 kg", "0.5 mm", "10 mm thick", "40 mm wide and 70 mm long",
                  "grown 1 mm", "sqrt((half the gap + 11 mm)^2 + (21 mm)^2)", "71 mm", "under 88 mm apart",
                  "within 45 mm of the ring's axis", "within 45 mm of the former's axis", "every 5 mm",
                  "5 mm or more above the table", "9 to 12 mm", "10 degrees", "1%", "2 mm", "clear the step by 8 mm",
                  "Possible scores: 0, 0.25, 0.5, 1.0"):
            self.assertIn(s, text, s)
        self.assertAlmostEqual(2 * L.STEP_RADIUS, 0.036)
        self.assertEqual((L.HOLD_GAP, L.HOLD_RADIUS, L.ON_FORMER_Z, L.EDGE_SAMPLE), (0.088, 0.045, 0.005, 0.005))
        # one colour name per part, the same in the instruction, the setup, the README and the code's colours
        readme = (HERE / "README.md").read_text()
        self.assertIn("blue wax crown", spec["instruction"])
        self.assertIn("blue inlay wax block", spec["setup"])
        self.assertIn("grey stainless-steel casting rings", spec["setup"])
        for t in (spec["setup"], spec["context"], readme, self.task.__doc__):
            self.assertNotIn("blue-green", t)
        r, g, b = L.WAX_BLUE
        self.assertTrue(b > 2 * g and b > 2 * r)
        self.assertEqual(spec["run_limit"], "1,600 steps (80 s at 20 Hz)")
        self.assertIn("demo: 1.0", spec["scores"])
        self.assertNotIn("media", spec)


_RUNNER = r'''
import json, sys
sys.path.insert(0, sys.argv[1])
import check_task as C, scenegeom as G, yamkin as K
from pathlib import Path
task_dir = Path(sys.argv[2]); name = sys.argv[3]
task = C.load_task(task_dir / "task.py", task_dir)
import layout as L
ring_plan = task.demo.__globals__["ring_plan"]  # the loaded task file's own plan builder
fx, fy = L.FORMER_X, L.FORMER_Y
plans = {
    "ring_b": ring_plan("ring_b"),
    "ring_a": ring_plan("ring_a"),
    "step_edge": ring_plan(place=(fx, fy + 0.010)),
    "held": ring_plan(release=False),
    "offset_touch": ring_plan(place=(fx, fy - 0.014)),
}
top = K.TABLE_TOP["station"]
solids = [G.solid_from_spec(o, top) for o in task.objects]
out = C.Playback(task, {"left": plans[name]}, solids, hz=20.0, hold_s=3.0).run()["proxy"]
print(json.dumps({"score": out["predicted_task_score"], "final": out["final_hooks"],
                  "events": [e["event"] for e in out["events"]]}))
'''


@unittest.skipIf(_checker() is None or _stub_dir() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The demo and its tier variants played through our offline checker's kinematics (rings moved by the checker's
    grasp proxy, released rings by offline_model.py)."""

    def env(self, **extra):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for k in ("TASK_TRACE", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY"):
            env.pop(k, None)
        env.update(extra)
        return env

    def checker(self, **extra):
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--json", "-"], capture_output=True,
                              text=True, env=self.env(**extra), timeout=900)
        return proc, json.loads(proc.stdout)

    def test_demo_predicts_one(self):
        proc, rep = self.checker()
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 420)
        self.assertLess(rep["playback"]["arms"]["left"]["worst_ik_mm"], 0.5)
        self.assertEqual(rep["playback"]["proxy"]["final_hooks"], {"success": True, "score": 1.0})

    def test_each_arm_alone(self):
        _, rep = self.checker(DEMO_IDLE_ARM="left")
        self.assertEqual(rep["playback"]["proxy"]["predicted_task_score"], 0.0)
        _, rep = self.checker(DEMO_IDLE_ARM="right")
        self.assertEqual(rep["playback"]["proxy"]["predicted_task_score"], 1.0)

    def variant(self, name):
        proc = subprocess.run([sys.executable, "-c", _RUNNER, str(_checker().parent), str(HERE), name],
                              capture_output=True, text=True, env=self.env(), timeout=900)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        return json.loads(proc.stdout.strip().splitlines()[-1])

    def test_tier_variants(self):
        for name, want in (("ring_b", 0.25), ("ring_a", 0.25), ("step_edge", 0.5), ("held", 0.5),
                           ("offset_touch", 0.0)):
            out = self.variant(name)
            self.assertEqual(out["score"], want, (name, out))


if __name__ == "__main__":
    unittest.main()
