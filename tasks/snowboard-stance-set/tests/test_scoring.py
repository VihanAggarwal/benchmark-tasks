"""Offline tests for scoring.py and task.py: synthetic binding states and synthetic arm snapshots (rigstate.Arm or
rigstate.offline_feed), no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import re
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _offline_dir():
    d = _stub_dir()
    return None if d is None else d.parent


if _stub_dir() is not None and str(_stub_dir()) not in sys.path:
    sys.path.insert(0, str(_stub_dir()))  # mechkit and task.py import yam_arena.task_file: the offline stand-in

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    NAMES, Tracker, hands_gap, over_grid, part_distance, pose, read_arms, tier, tip_distance, valid, wrap_deg,
    zone_gap, zone_radius,
)

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
TOP = L.BOARD_TOP
FRONT = (L.LINE_X, L.STANCE_W / 2, L.FRONT_TARGET_DEG)  # the finished stance
BACK = (L.LINE_X, -L.STANCE_W / 2, L.BACK_TARGET_DEG)
READY_TIPS = {"left": (0.323, 0.305, L.TOP + 0.247), "right": (0.323, -0.305, L.TOP + 0.247)}
OPEN70 = -0.035 / rigstate.FINGER_OPEN  # rigstate's opening with the pads 70 mm apart (the demo's): 0.745


def quat(angle_deg, tilt_deg=0.0, tilt_axis="v"):
    """Quaternion (xyzw) of a binding turned to ``angle_deg`` and then tilted ``tilt_deg`` about its own v (or u)
    axis."""
    a, t = math.radians(angle_deg), math.radians(tilt_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    if tilt_axis == "v":
        rt = np.array([[math.cos(t), 0, math.sin(t)], [0, 1, 0], [-math.sin(t), 0, math.cos(t)]])
    else:
        rt = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    r = rz @ rt
    w = math.sqrt(max(1e-12, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    if w > 1e-3:
        return np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], 4 * w * w]) / (4 * w)
    # half turns: fall back to an axis-angle construction
    ang = math.acos(max(-1.0, min(1.0, (np.trace(r) - 1) / 2)))
    vals, vecs = np.linalg.eig(r)
    axis = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    axis /= np.linalg.norm(axis)
    return np.array([*(axis * math.sin(ang / 2)), math.cos(ang / 2)])


def B(x, y, angle, z=TOP, tilt=0.0, tilt_axis="v"):
    """A binding state: its frame's origin (the baseplate's underside centre) and orientation."""
    o = np.array([x, y, z], float)
    return SimpleNamespace(origin=o, quat_xyzw=quat(angle, tilt, tilt_axis), pos=o + [0, 0, 0.03],
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=o - 0.1, high=o + 0.1)


def arms_at(left=READY_TIPS["left"], right=READY_TIPS["right"], opening=1.0):
    """rigstate-like arms pointing straight down, their finger-tip points at ``left`` and ``right`` (the TCP 13 mm
    above), both open ``opening`` (0 closed .. 1 open, 94 mm)."""
    out = {}
    for side, tips in (("left", left), ("right", right)):
        t = np.array(tips, float)
        out[side] = rigstate.Arm(tcp=t + [0, 0, 0.013], tips=t, opening=opening)
    return out


AWAY = arms_at()


def strap_point(b):
    """The world point at the middle of a binding's ankle-strap bar (where a gripper holds it)."""
    p = pose(b)
    return p.origin + p.rot @ np.array(L.STRAP[0])


def local_point(b, uvz):
    """A point given in a binding's own frame (u heel to toe, v across, z up from its underside), in the world."""
    p = pose(b)
    return p.origin + p.rot @ np.array(uvz, float)


def run(frames, tracker=None):
    """Feed (nose, tail, arms) frames to a tracker; return the last result and the tracker."""
    t = tracker or Tracker(HZ)
    r = None
    for nose, tail, arms in frames:
        r = t.update(nose, tail, arms)
    return r, t


def still(nose, tail, arms=AWAY, n=WIN + 2):
    return [(nose, tail, arms)] * n


def end_score(nose, tail, arms=AWAY):
    return run(still(nose, tail, arms))[0]


NOSE0 = B(*L.NOSE_START)
TAIL0 = B(*L.TAIL_START)


def glide(a, b, n, tail_or_nose="nose", other=None, arms=AWAY):
    """Frames moving one binding linearly (origin and angle) from state tuple a to b over n steps."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        s = B(*[(1 - f) * u + f * v for u, v in zip(a, b)])
        out.append((s, other, arms) if tail_or_nose == "nose" else (other, s, arms))
    return out


class Geometry(unittest.TestCase):
    def test_wrap(self):
        self.assertEqual(wrap_deg(180.0), 180.0)
        self.assertEqual(wrap_deg(-180.0), 180.0)
        self.assertAlmostEqual(wrap_deg(195.0), -165.0)
        self.assertAlmostEqual(wrap_deg(-15.0 - 180.0), 165.0)

    def test_pose_heading_tilt(self):
        p = pose(B(0.3, 0.2, 15.0))
        self.assertAlmostEqual(p.heading, 15.0, places=6)
        self.assertAlmostEqual(p.tilt, 0.0, places=6)
        self.assertAlmostEqual(pose(B(0.3, 0.2, 15.0, tilt=4.0)).tilt, 4.0, places=5)
        self.assertAlmostEqual(pose(B(0.3, 0.2, 15.0, tilt=180.0, tilt_axis="u")).tilt, 180.0, places=4)

    def test_part_distance(self):
        b = pose(B(0.3, 0.2, 0.0))
        self.assertEqual(part_distance(b.origin + [0, 0, 0.005], b), 0.0)  # inside the baseplate
        # 30 mm above the strap bar's top, over its centre
        self.assertAlmostEqual(part_distance(b.origin + [L.STRAP_U, 0, L.STRAP_TOP + 0.030], b), 0.030, places=9)
        # beside the highback: 30 mm behind its heel face (u -0.100), at mid-height
        self.assertAlmostEqual(part_distance(b.origin + [-0.130, 0, 0.05], b), 0.030, places=9)
        # the disc counts as its bounding box: 1 mm above the corner of its 90 mm square
        self.assertAlmostEqual(part_distance(b.origin + [0.044, 0.044, 0.019], b), 0.001, places=9)

    def test_tip_distance_needs_readable_arms(self):
        b = pose(B(*FRONT))
        self.assertIsNone(tip_distance(None, b))
        bad = arms_at(left=(np.nan, 0, 0))
        self.assertIsNone(tip_distance(bad, b))
        self.assertIsNone(tip_distance({"left": AWAY["left"]}, b))
        self.assertGreater(tip_distance(AWAY, b), 0.1)
        # 'let go' needs the whole snapshot: a missing or non-finite TCP or opening makes the arms unreadable
        for bad in ({"left": SimpleNamespace(tips=AWAY["left"].tips, opening=1.0), "right": AWAY["right"]},
                    {"left": rigstate.Arm(tcp=AWAY["left"].tcp, tips=AWAY["left"].tips, opening=math.nan),
                     "right": AWAY["right"]},
                    {"left": rigstate.Arm(tcp=np.array([np.inf, 0, 0]), tips=AWAY["left"].tips, opening=1.0),
                     "right": AWAY["right"]}, None, {}):
            self.assertIsNone(read_arms(bad))
            self.assertIsNone(hands_gap(bad, b))
        self.assertGreater(hands_gap(AWAY, b), 0.1)

    def test_finger_zone_radius(self):
        """Half the finger gap plus the 8 mm pad, and half the 27 mm pad width across: the numbers in the texts."""
        self.assertAlmostEqual(zone_radius(1.0) * 1000, 56.6, delta=0.05)  # fully open, 94 mm
        self.assertAlmostEqual(zone_radius(OPEN70) * 1000, 45.1, delta=0.05)  # the demo's 70 mm
        self.assertAlmostEqual(zone_radius(0.0) * 1000, 15.7, delta=0.05)  # closed
        self.assertEqual([round((L.LET_GO + zone_radius(o)) * 1000) for o in (1.0, OPEN70, 0.0)], [87, 75, 46])
        self.assertEqual(zone_radius(1.7), zone_radius(1.0))  # clipped to 0..1
        self.assertEqual(zone_radius(-0.2), zone_radius(0.0))

    def test_finger_zone_runs_up_the_fingers(self):
        """The zone is the 38 mm segment from the tip point toward the TCP, swept by the radius: a gripper reaching
        down past the highback's top, tips beside it, is measured from the fingers, not only from the tips."""
        b = pose(B(0.3, 0.2, 0.0))
        tips = b.origin + [-0.100 - 0.045, 0, 0.095 - 0.030]  # 45 mm behind the highback, 30 mm below its top
        arm = (tips + [0, 0, 0.013], tips, 0.0)  # closed: radius 15.7 mm
        self.assertAlmostEqual(zone_gap(arm, b) + zone_radius(0.0), 0.045, places=9)
        arm_side = (tips + [0.013, 0, 0], tips, 0.0)  # the same tips, the gripper on its side, wrist toward it
        self.assertLess(zone_gap(arm_side, b), zone_gap(arm, b))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r = end_score(NOSE0, TAIL0)
        self.assertEqual((r.score, r.success), (0.0, False))
        r, _ = run([(NOSE0, TAIL0, AWAY)])
        self.assertEqual(r.score, 0.0)

    def test_finished_stance_scores_one(self):
        r = end_score(B(*FRONT), B(*BACK))
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertEqual(r.details["W_mm"], 400.0)
        self.assertEqual(r.details["C_mm"], 0.0)
        self.assertEqual(r.details["binding_nose"]["role"], "front")

    def test_tier_function_boundaries(self):
        mm = 0.001
        for w, want in ((389.9, 0.75), (390.0, 1.0), (410.0, 1.0), (410.1, 0.75), (379.9, 0.5), (380.0, 0.75),
                        (420.0, 0.75), (420.1, 0.5)):
            self.assertEqual(tier(2, w * mm, 0.0), want, w)
        for c, want in ((10.0, 1.0), (10.1, 0.75), (-10.0, 1.0), (-10.1, 0.75), (20.0, 0.75), (20.1, 0.5),
                        (-20.1, 0.5)):
            self.assertEqual(tier(2, 0.400, c * mm), want, c)
        self.assertEqual((tier(0, 0.4, 0.0), tier(1, 0.4, 0.0)), (0.0, 0.25))

    def test_width_and_centre_boundaries_from_states(self):
        for w, want in ((389.9, 0.75), (390.0, 1.0), (410.0, 1.0), (410.1, 0.75), (379.9, 0.5), (380.0, 0.75),
                        (420.0, 0.75), (420.1, 0.5)):
            h = w / 2000.0
            self.assertEqual(end_score(B(L.LINE_X, h, 15), B(L.LINE_X, -h, -15)).score, want, w)
        for c, want in ((10.0, 1.0), (10.1, 0.75), (20.0, 0.75), (20.1, 0.5)):
            d = c / 1000.0
            self.assertEqual(end_score(B(L.LINE_X, 0.2 + d, 15), B(L.LINE_X, -0.2 + d, -15)).score, want, c)

    def _one(self, front, label):
        """Score with the back binding finished and the front binding as given: 1.0 when it is set, else 0.25."""
        return end_score(front, B(*BACK)).score

    def test_angle_boundary(self):
        for err, want in ((3.0, 1.0), (-3.0, 1.0), (3.1, 0.25), (-3.1, 0.25)):
            self.assertEqual(self._one(B(L.LINE_X, 0.2, 15 + err), "angle"), want, err)

    def test_centring_boundary(self):
        for off, want in ((10.0, 1.0), (-10.0, 1.0), (10.1, 0.25), (-10.1, 0.25)):
            self.assertEqual(self._one(B(L.LINE_X + off / 1000, 0.2, 15), "x"), want, off)

    def test_tilt_boundary(self):
        for t, want in ((5.0, 1.0), (5.1, 0.25)):
            for axis in ("u", "v"):
                self.assertEqual(self._one(B(L.LINE_X, 0.2, 15, tilt=t, tilt_axis=axis), "tilt"), want, (t, axis))

    def test_underside_height_boundary(self):
        for dz, want in ((3.0, 1.0), (-3.0, 1.0), (3.1, 0.25), (-3.1, 0.25), (0.5, 1.0)):
            self.assertEqual(self._one(B(L.LINE_X, 0.2, 15, z=TOP + dz / 1000), "z"), want, dz)

    def test_let_go_boundary(self):
        """A gripper straight above the bar: its tip point must be 30 mm plus the finger radius above the bar's top
        (86.6 mm fully open, 75.1 mm at 70 mm, 45.7 mm closed)."""
        front = B(*FRONT)
        top = strap_point(front) + [0, 0, L.STRAP[1][2] / 2]  # the strap bar's top, over its centre
        for opening in (1.0, OPEN70, 0.0):
            need = L.LET_GO + zone_radius(opening)
            for d, want in ((need - 0.0001, 0.25), (need, 1.0)):
                arms = arms_at(left=top + [0, 0, d], opening=opening)
                self.assertEqual(end_score(front, B(*BACK), arms).score, want, (opening, d))
            # the other arm counts too
            arms = arms_at(right=top + [0, 0, need - 0.0001], opening=opening)
            self.assertEqual(end_score(front, B(*BACK), arms).score, 0.25, opening)
        r = end_score(front, B(*BACK), arms_at(left=top + [0, 0, L.LET_GO + zone_radius(1.0)]))
        self.assertAlmostEqual(r.details["binding_nose"]["zone_mm"], 30.0, places=2)

    def test_grid_boundary(self):
        """The disc centre over an insert grid: |y| 140 to 260 mm (the other binding finished; both set but far from
        the stance is 0.5, one set 0.25)."""
        for y, want in ((0.140, 0.5), (0.1399, 0.25), (0.260, 0.5), (0.2601, 0.25)):
            self.assertEqual(self._one(B(L.LINE_X, y, 15), "grid"), want, y)
            self.assertEqual(end_score(B(*FRONT), B(L.LINE_X, -y, -15)).score, want, -y)
        self.assertTrue(over_grid(pose(B(*FRONT))) and over_grid(pose(B(*BACK))))
        self.assertFalse(over_grid(pose(NOSE0)) or over_grid(pose(TAIL0)) or over_grid(pose(B(L.LINE_X, 0, 15))))

    def test_still_path_boundary(self):
        for step_mm, want in ((0.2, 1.0), (0.21, 0.25)):  # 10 steps in the window: 2.0 mm and 2.1 mm of path
            frames = []
            for i in range(WIN + 1):
                y = 0.2 + (i % 2) * step_mm / 1000  # shuffling back and forth: the path adds up, the spread does not
                frames.append((B(L.LINE_X, y, 15), B(*BACK), AWAY))
            r, _ = run(frames)
            self.assertEqual(r.score, want, step_mm)

    def test_still_turn_boundary(self):
        for step_deg, want in ((0.1, 1.0), (0.11, 0.25)):  # 1.0 deg and 1.1 deg of turning in the window
            frames = [(B(L.LINE_X, 0.2, 15 + (i % 2) * step_deg), B(*BACK), AWAY) for i in range(WIN + 1)]
            self.assertEqual(run(frames)[0].score, want, step_deg)

    def test_stillness_needs_half_a_second(self):
        frames = still(B(*FRONT), B(*BACK), n=WIN)
        self.assertEqual(run(frames)[0].score, 0.0)  # 10 samples = 9 steps: not yet 0.5 s
        self.assertEqual(run(still(B(*FRONT), B(*BACK), n=WIN + 1))[0].score, 1.0)

    def test_every_tier_is_reachable_by_a_state_sequence(self):
        """From the parked start, a sequence of states reaching each listed score."""
        start = still(NOSE0, TAIL0)
        seqs = {
            0.0: start + still(B(L.LINE_X, 0.2, -15), B(L.LINE_X, -0.2, 15)),  # toes turned in
            0.25: start + glide(L.NOSE_START, FRONT, 40, other=TAIL0) + still(B(*FRONT), TAIL0),  # one arm's job
            0.5: start + glide(L.NOSE_START, (L.LINE_X, 0.24, 15), 40, other=TAIL0)
            + glide(L.TAIL_START, (L.LINE_X, -0.24, -15), 40, "tail", B(L.LINE_X, 0.24, 15))
            + still(B(L.LINE_X, 0.24, 15), B(L.LINE_X, -0.24, -15)),  # both on the outer inserts: W 480
            0.75: start + still(B(L.LINE_X, 0.1925, 15), B(L.LINE_X, -0.1925, -15)),  # W 385 mm
            1.0: start + glide(L.NOSE_START, FRONT, 40, other=TAIL0)
            + glide(L.TAIL_START, BACK, 40, "tail", B(*FRONT)) + still(B(*FRONT), B(*BACK)),
        }
        for want, frames in seqs.items():
            r, _ = run(frames)
            self.assertEqual(r.score, want, want)
        self.assertEqual(run(seqs[0.5])[0].details["W_mm"], 480.0)
        # the other 0.5 example in the texts: W 400 mm but 40 mm off the cross
        r = end_score(B(L.LINE_X, 0.24, 15), B(L.LINE_X, -0.16, -15))
        self.assertEqual((r.score, r.details["W_mm"], r.details["C_mm"]), (0.5, 400.0, 40.0))

    def test_scores_are_only_the_listed_ones(self):
        rng = np.random.default_rng(3)
        seen = set()
        for _ in range(400):
            ys = rng.uniform(-0.5, 0.5, 2)
            ang = rng.choice([15.0, -15.0, 90.0, -90.0, 165.0], 2)
            r = end_score(B(L.LINE_X, ys[0], ang[0]), B(L.LINE_X, ys[1], ang[1]))
            seen.add(r.score)
        self.assertTrue(seen <= {0.0, 0.25, 0.5, 0.75, 1.0})


class RankRule(unittest.TestCase):
    """Front and back come from where the bindings are; each has its own target angle."""

    def test_forward_stance_scores_quarter(self):  # +15 / +15
        self.assertEqual(end_score(B(L.LINE_X, 0.2, 15), B(L.LINE_X, -0.2, 15)).score, 0.25)

    def test_toes_turned_in_scores_zero(self):  # pigeon-toed: -15 / +15
        self.assertEqual(end_score(B(L.LINE_X, 0.2, -15), B(L.LINE_X, -0.2, 15)).score, 0.0)

    def test_heel_forward_scores_quarter(self):  # the front binding turned round: highback on the far edge
        r = end_score(B(L.LINE_X, 0.2, 15 - 180), B(*BACK))
        self.assertEqual(r.score, 0.25)
        self.assertAlmostEqual(abs(r.details["binding_nose"]["angle_err_deg"]), 180.0, places=4)

    def test_both_on_the_nose_half_sets_only_the_one_over_the_grid(self):
        r = end_score(B(L.LINE_X, 0.40, 15), B(L.LINE_X, 0.20, -15))
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.details["binding_nose"]["over_grid"])
        self.assertTrue(r.details["binding_tail"]["set"])

    def test_nose_binding_carried_to_the_tail_spot_scores_zero(self):
        # One arm (tilted) carries its own binding to y -0.20 at -15 while the tail binding stays parked: the moved
        # one is then the FRONT binding (larger y than -0.445), whose target is +15.
        r = end_score(B(L.LINE_X, -0.2, -15), TAIL0)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["binding_nose"]["role"], "front")

    def test_swapped_bindings_score_the_same(self):  # identical parts: which one went where does not matter
        self.assertEqual(end_score(B(*BACK), B(*FRONT)).score, 1.0)

    def test_equal_y_has_one_front(self):
        r = end_score(B(L.LINE_X, 0.0, 15), B(L.LINE_X, 0.0, -15))
        self.assertEqual({r.details[n]["role"] for n in NAMES}, {"front", "back"})
        self.assertEqual(r.score, 0.0)  # at the cross: neither is over an insert grid


class Exploits(unittest.TestCase):
    def test_holding_both_in_place_scores_zero(self):
        f, b = B(*FRONT), B(*BACK)
        held = arms_at(left=strap_point(f), right=strap_point(b))
        r = end_score(f, b, held)
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertFalse(r.details["binding_nose"]["let_go"])

    def test_resting_fingers_on_a_binding_is_not_let_go(self):
        f, b = B(*FRONT), B(*BACK)
        on_top = strap_point(f) + [0, 0, L.STRAP[1][2] / 2 + 0.005]  # tips 5 mm above the strap bar
        self.assertEqual(end_score(f, b, arms_at(left=on_top)).score, 0.25)

    def test_opening_at_the_release_pose_and_stopping_is_not_let_go(self):
        """The instruction ends 'let go of both and lift clear': opening the grippers where they let go (the tip
        points inside the strap bar), or lifting them only a little, scores 0; the demo's end (tips 92 mm over the
        bar) scores 1.0."""
        f, b = B(*FRONT), B(*BACK)
        lt, rt = local_point(f, (0, 0, 0.032)), local_point(b, (0, 0, 0.032))  # the release pose's tip points
        for lift, want in ((0.0, 0.0), (0.020, 0.0), (0.023 + 0.030, 0.0),  # 30 mm over the bar: not clear yet
                           (0.023 + 0.0750, 0.0), (0.023 + 0.0751, 1.0),  # the rule: 75.07 mm over the bar
                           (0.023 + 0.092, 1.0)):  # the demo's end: 92 mm over it
            r = end_score(f, b, arms_at(left=lt + [0, 0, lift], right=rt + [0, 0, lift], opening=OPEN70))
            self.assertEqual(r.score, want, lift)
        r = end_score(f, b, arms_at(left=lt, right=rt, opening=OPEN70))
        self.assertEqual(r.details["binding_nose"]["tip_mm"], 0.0)  # the tip point is inside the bar

    def test_an_open_finger_resting_on_the_highback_is_not_let_go(self):
        """An open gripper beside the binding with its heel-side pad resting on the highback's top: its tip point is
        45 mm from every part (more than 30), but its fingers touch the binding."""
        f, b = B(*FRONT), B(*BACK)
        tips = local_point(f, (-0.045, 0.0, 0.095))
        r = end_score(f, b, arms_at(left=tips, opening=1.0))
        self.assertEqual((r.score, r.success), (0.25, False))
        self.assertAlmostEqual(r.details["binding_nose"]["tip_mm"], 45.0, places=1)
        self.assertLess(r.details["binding_nose"]["zone_mm"], 0.0)
        self.assertFalse(r.details["binding_nose"]["let_go"])

    def test_single_pads_on_the_baseplate_edges_are_not_let_go(self):
        """Each open gripper with one pad on a baseplate's edge (toe end of the front one, side of the back one):
        tip points 37 mm from the parts, fingers on them. Neither binding is let go."""
        f, b = B(*FRONT), B(*BACK)
        left, right = local_point(f, (0.137, 0.0, 0.015)), local_point(b, (0.0, 0.102, 0.015))
        r = end_score(f, b, arms_at(left=left, right=right, opening=1.0))
        self.assertEqual(r.score, 0.0)
        self.assertAlmostEqual(r.details["binding_nose"]["tip_mm"], 37.0, delta=0.5)
        self.assertAlmostEqual(r.details["binding_tail"]["tip_mm"], 37.0, delta=0.5)

    def test_unreadable_arms_never_count_as_let_go(self):
        self.assertEqual(end_score(B(*FRONT), B(*BACK), None).score, 0.0)
        nan = arms_at(left=(np.nan, np.nan, np.nan))
        self.assertEqual(end_score(B(*FRONT), B(*BACK), nan).score, 0.0)

    def test_turning_both_in_place_scores_zero(self):
        """Turned to their angles where they lie, on the board's ends: neither disc is over an insert grid."""
        r = end_score(B(L.LINE_X, 0.445, 15), B(L.LINE_X, -0.445, -15))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["binding_nose"]["over_grid"] or r.details["binding_tail"]["over_grid"])
        self.assertTrue(r.details["binding_nose"]["angled"] and r.details["binding_tail"]["angled"])

    def test_one_placed_and_the_other_turned_in_place_scores_quarter(self):
        self.assertEqual(end_score(B(*FRONT), B(L.LINE_X, -0.445, -15)).score, 0.25)
        self.assertEqual(end_score(B(L.LINE_X, 0.445, 15), B(*BACK)).score, 0.25)

    def test_dragging_into_place_is_scored_by_the_end_pose_only(self):
        # Slid along the board without lifting (z never leaves the board): the end state is all that counts.
        dragged = (still(NOSE0, TAIL0) + glide(L.NOSE_START, FRONT, 60, other=TAIL0)
                   + glide(L.TAIL_START, BACK, 60, "tail", B(*FRONT)) + still(B(*FRONT), B(*BACK)))
        self.assertEqual(run(dragged)[0].score, 1.0)
        # ... and a drag that leaves it 12 mm off the centre line is scored as it lies
        off = still(B(L.LINE_X + 0.012, 0.2, 15), B(*BACK))
        self.assertEqual(run(dragged + off)[0].score, 0.25)

    def test_knocked_off_the_board_or_stacked_or_upside_down(self):
        back = B(*BACK)
        for bad in (B(L.LINE_X, 0.2, 15, z=L.TOP),  # fallen onto the table
                    B(L.LINE_X, 0.2, 15, z=TOP + 0.015),  # lying on the other binding's baseplate
                    B(L.LINE_X, 0.2, 15, z=TOP + 0.095, tilt=180, tilt_axis="u"),  # upside down
                    B(L.LINE_X, 0.2, 15, z=TOP + 0.06, tilt=90, tilt_axis="v")):  # standing on its heel
            self.assertEqual(end_score(bad, back).score, 0.25)

    def test_a_threshold_cannot_be_gamed_by_the_centre(self):
        # discs exactly 400 mm apart but both shifted 15 mm toward the nose: only 0.75
        self.assertEqual(end_score(B(L.LINE_X, 0.215, 15), B(L.LINE_X, -0.185, -15)).score, 0.75)

    def test_still_moving_is_not_set(self):
        frames = [(B(L.LINE_X, 0.2 + 0.0004 * i, 15), B(*BACK), AWAY) for i in range(WIN + 5)]
        self.assertEqual(run(frames)[0].score, 0.25)  # creeping 8 mm/s


class Robustness(unittest.TestCase):
    BAD = None

    @classmethod
    def setUpClass(cls):
        nan = SimpleNamespace(origin=np.full(3, np.nan), quat_xyzw=np.array([0, 0, 0, 1.0]))
        inf = SimpleNamespace(origin=np.array([0.3, 0.2, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]))
        nanq = SimpleNamespace(origin=np.array([0.3, 0.2, TOP]), quat_xyzw=np.array([np.nan, 0, 0, 1.0]))
        zero_q = SimpleNamespace(origin=np.array([0.3, 0.2, TOP]), quat_xyzw=np.zeros(4))
        missing = SimpleNamespace(pos=np.array([0.3, 0.2, TOP + 0.03]), quat_xyzw=np.array([0, 0, 0, 1.0]))
        short = SimpleNamespace(origin=np.array([0.3, 0.2]), quat_xyzw=np.array([0, 0, 0, 1.0]))
        cls.BAD = (nan, inf, nanq, zero_q, missing, short, None)

    def test_bad_states_score_zero_and_leave_the_tracker_alone(self):
        for bad in self.BAD:
            self.assertFalse(valid(bad))
            _, t = run(still(B(*FRONT), B(*BACK)))
            hist, prev, steps = list(t.hist), t.prev, t.steps
            for nose, tail in ((bad, B(*BACK)), (B(*FRONT), bad), (bad, bad)):
                r = t.update(nose, tail, AWAY)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(t.result(nose, tail, AWAY).score, 0.0)
            self.assertEqual(t.prev, prev)
            self.assertEqual(len(t.hist), len(hist))
            self.assertTrue(all(a is b for a, b in zip(t.hist, hist)))
            self.assertEqual((t.steps, t.bad_steps), (steps + 3, 3))
            r = t.update(B(*FRONT), B(*BACK), AWAY)  # the next good state is scored as before
            self.assertEqual(r.score, 1.0)

    def test_score_is_idempotent(self):
        _, t = run(still(B(*FRONT), B(*BACK)))
        a = [t.result(B(*FRONT), B(*BACK), AWAY).score for _ in range(5)]
        self.assertEqual(a, [1.0] * 5)
        self.assertEqual(t.steps, WIN + 2)

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(11)
        t = Tracker(HZ)
        for _ in range(500):
            q = rng.normal(size=4)
            s1 = SimpleNamespace(origin=rng.uniform(-2, 2, 3), quat_xyzw=q)
            s2 = B(*rng.uniform(-1, 1, 2), rng.uniform(-400, 400), z=rng.uniform(0, 2), tilt=rng.uniform(0, 180))
            arms = arms_at(left=rng.uniform(-1, 1, 3), right=rng.uniform(-1, 1, 3)) if rng.random() < 0.8 else None
            r = t.update(s1, s2, arms)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details)


class Reset(unittest.TestCase):
    def mid_trial(self):
        """A tracker part-way through: the nose binding set down at its spot."""
        frames = still(NOSE0, TAIL0) + glide(L.NOSE_START, FRONT, 40, other=TAIL0) + still(B(*FRONT), TAIL0)
        r, t = run(frames)
        self.assertEqual(r.score, 0.25)
        return t

    def test_exact_teleport_back_to_start_resets(self):
        t = self.mid_trial()
        r = t.update(NOSE0, TAIL0, AWAY)
        self.assertEqual((t.steps, t.resets, len(t.hist)), (1, 1, 1))
        self.assertEqual(r.score, 0.0)

    def test_jittered_teleport_resets(self):
        t = self.mid_trial()
        t.update(B(L.LINE_X + 0.003, 0.445 - 0.002, 91.5), B(L.LINE_X - 0.002, -0.445 + 0.003, -88.6), AWAY)
        self.assertEqual((t.steps, t.resets), (1, 1))

    def test_gradual_return_is_not_a_reset(self):
        t = self.mid_trial()
        steps = t.steps
        run(glide(FRONT, L.NOSE_START, 60, other=TAIL0), t)
        self.assertEqual(t.resets, 0)
        self.assertEqual(t.steps, steps + 60)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.mid_trial()
        steps = t.steps
        t.update(B(0.25, 0.0, 40), TAIL0, AWAY)  # knocked far in one step, not onto the start
        t.update(B(L.LINE_X, 0.30, 90), TAIL0, AWAY)  # a jump near, but not onto, the start
        self.assertEqual((t.resets, t.steps), (0, steps + 2))

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.mid_trial()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual((t.steps, len(t.hist)), (0, 0))
        finally:
            rigstate._RESET_CALLBACKS.clear()
            rigstate._RESET_CALLBACKS.update(saved)


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: objects, hooks, the offline proxy, the demo, texts."""

    @classmethod
    def setUpClass(cls):
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def state(self, spec, x, y, angle, z=TOP):
        return self.mk.state_from_poses(spec, (x, y, z, *self.mk.yaw_quat(math.radians(angle))), source="spec")

    def objects(self, nose=L.NOSE_START, tail=L.TAIL_START):
        return {"binding_nose": self.state(self.task.BINDING_NOSE, *nose),
                "binding_tail": self.state(self.task.BINDING_TAIL, *tail)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sum(n.startswith("insert_") for n in names), 16)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        for b in (self.task.BINDING_NOSE, self.task.BINDING_TAIL):
            self.assertEqual(b.mass, L.BINDING_MASS)
            self.assertEqual(b.kind, "rigid")
            self.assertEqual(tuple(round(v, 6) for v in b.size), (0.2, 0.13, 0.095))

    def test_centre_of_mass_under_the_strap_bar(self):
        com = self.task.BINDING_NOSE.center_of_mass()
        self.assertAlmostEqual(com[0], -0.0147, delta=0.0005)  # 14.7 mm behind the bar's centre, toward the heel
        self.assertLessEqual(abs(com[0] - L.STRAP_U), L.STRAP[1][0] / 2)  # still under the 30 mm bar
        self.assertAlmostEqual(com[1], 0.0, places=9)
        self.assertAlmostEqual(com[2], 0.022, delta=0.001)

    def test_start_layout(self):
        o = self.objects()
        nose, tail = pose(o["binding_nose"]), pose(o["binding_tail"])
        self.assertAlmostEqual(nose.origin[1] + 0.100, L.BOARD_Y1 - 0.005, places=9)  # toe 5 mm from the nose
        self.assertAlmostEqual(tail.origin[1] - 0.100, L.BOARD_Y0 + 0.005, places=9)
        self.assertAlmostEqual(nose.heading, 90.0, places=6)
        self.assertAlmostEqual(tail.heading, -90.0, places=6)
        lo, hi = o["binding_nose"].low, o["binding_nose"].high
        self.assertGreater(lo[1], max(L.DOT_YS_NOSE) + L.DOT_R)  # nothing but the strips under a parked binding
        self.assertAlmostEqual(lo[2], TOP, places=9)
        self.assertGreaterEqual(lo[0], L.BOARD_X0)
        self.assertLessEqual(hi[0], L.BOARD_X1)

    def test_offline_proxy_round_trips_the_origin(self):
        tf = importlib.import_module("yam_arena.task_file")
        for spec in (self.task.BINDING_NOSE, self.task.BINDING_TAIL):
            centre, size, yaw = spec.offline_box()
            self.assertEqual(size, L.COLUMN_SIZE)
            self.assertAlmostEqual(yaw, spec.yaw)
            for c, a in ((np.array(centre), yaw), (np.array([0.2807, 0.1948, TOP + 0.0275]), math.radians(15))):
                q = self.mk.yaw_quat(a)
                plain = tf.ObjectState(c, q, np.zeros(3), np.zeros(3), c - 0.05, c + 0.05)
                s = spec.offline_state(plain)
                back = s.origin + self.mk.quat_to_matrix(q) @ np.array(L.COLUMN_CENTER)
                np.testing.assert_allclose(back, c, atol=1e-12)
                self.assertEqual(s.source, "offline")
            s = spec.offline_state(tf.ObjectState(np.array(centre), self.mk.yaw_quat(yaw), np.zeros(3), np.zeros(3),
                                                  np.zeros(3), np.zeros(3)))
            np.testing.assert_allclose(s.origin, spec.pos, atol=1e-12)
        # the place point puts the disc on its target
        s = self.task.BINDING_NOSE.offline_state(tf.ObjectState(
            np.array([*self.task.place_point("left")[:2], TOP + 0.0275]), self.mk.yaw_quat(math.radians(15)),
            np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)))
        np.testing.assert_allclose(s.origin, [L.LINE_X, 0.2, TOP], atol=1e-12)

    def test_hooks_and_reset_signals(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.mk.offline_reset()
        rigstate.offline_feed({s: {"tcp": np.array(p) + [0, 0, 0.013], "tips": p, "opening": 1.0}
                               for s, p in READY_TIPS.items()})
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        self.assertGreaterEqual(tr.steps, 1)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.steps, 1)
        self.mk.offline_reset()  # mechkit's episode counter: also a new trial
        self.task.success(self.objects())
        self.assertEqual(tr.steps, 1)
        fin = self.objects(FRONT, BACK)
        for _ in range(WIN + 1):
            rigstate.offline_feed({s: {"tcp": np.array(p) + [0, 0, 0.013], "tips": p, "opening": 1.0}
                                   for s, p in READY_TIPS.items()})
            ok = self.task.success(fin)
        self.assertTrue(ok)
        self.assertEqual(self.task.score(fin), 1.0)
        self.assertEqual(self.task.score(fin), 1.0)  # idempotent

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_numbers(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side in ("left", "right"):
            self.assertAlmostEqual(sum(w.duration for w in plan[side]), 21.4, places=9)
            self.assertAlmostEqual(min(w.pos[2] for w in plan[side]) - L.TOP, 0.053, places=9)
        self.assertAlmostEqual(self.task.GRIP, -0.011, places=12)
        self.assertAlmostEqual(self.task.OPEN70, -0.035, places=12)
        self.assertAlmostEqual(self.task.PLACE_H, 0.055, places=12)
        x, y, yaw = self.task.place_point("left")  # the strap bar is over the disc: the TCP goes to the disc target
        self.assertAlmostEqual(x, 0.300, places=9)
        self.assertAlmostEqual(y, 0.200, places=9)
        self.assertAlmostEqual(math.degrees(yaw), -75.0, places=9)
        x, y, yaw = self.task.place_point("right")
        self.assertAlmostEqual(y, -0.200, places=9)
        self.assertAlmostEqual(math.degrees(yaw), 75.0, places=9)
        # the end: straight up to 17 cm with the fingers 70 mm apart; the tips 92 mm over the strap bar, so the finger
        # zone (45.1 mm round the axis) is 46.9 mm from the bar, 30 mm needed
        self.assertAlmostEqual(self.task.CLEAR_H, 0.170, places=12)
        self.assertAlmostEqual(self.task.OPEN70 / rigstate.FINGER_OPEN, OPEN70, places=12)
        for side, target in (("left", FRONT), ("right", BACK)):
            last = plan[side][-1]
            self.assertAlmostEqual(last.pos[2] - L.TOP, 0.170, places=9)
            self.assertEqual(last.finger, self.task.OPEN70)
            tcp = np.array(last.pos, float)
            tips = tcp - [0, 0, self.task.PAD_TIPS]
            b = pose(B(*target))
            self.assertAlmostEqual(part_distance(tips, b) * 1000, 92.0, places=6)
            gap = zone_gap((tcp, tips, OPEN70), b)
            self.assertAlmostEqual(gap * 1000, 46.9, delta=0.05)
            self.assertGreaterEqual(gap, L.LET_GO + 0.015)
            r = end_score(B(*FRONT), B(*BACK), arms_at(**{side: tips}, opening=OPEN70))
            self.assertEqual(r.score, 1.0)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_is_identical_everywhere(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        cmds = re.findall(r'--instruction "([^"]*)"', readme)
        self.assertGreaterEqual(len(cmds), 2)
        self.assertTrue(all(c == self.task.INSTRUCTION for c in cmds))

    def test_task_json_numbers(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertIn("demo: 1.0", meta["scores"])
        self.assertIn("1,500 steps (75 s at 20 Hz)", meta["run_limit"])
        self.assertIn("Possible scores: 0, 0.25, 0.5, 0.75, 1.0", meta["scoring"])
        readme = (HERE / "README.md").read_text()
        # the rule's numbers, as the code computes them, in task.json and in the README
        r = [round(zone_radius(o) * 1000) for o in (1.0, OPEN70, 0.0)]
        need = [round((L.LET_GO + zone_radius(o)) * 1000) for o in (1.0, OPEN70, 0.0)]
        g0, g1 = (round(v * 1000) for v in L.GRID_Y)
        for phrase in (f"{r[0]} mm fully open, {r[1]} mm at a 70 mm gap, {r[2]} mm closed",
                       f"{need[0]}, {need[1]} or {need[2]} mm", f"|y| {g0} to {g1} mm",
                       f"{round(L.FINGER_LEN * 1000)} mm", f"{round(L.LET_GO * 1000)} mm"):
            self.assertIn(phrase, meta["scoring"], phrase)
            self.assertIn(phrase, readme, phrase)
        self.assertNotIn("media", meta)
        for text in meta.values():
            self.assertNotIn(chr(0x2014), str(text))  # no em or en dashes in the task text
            self.assertNotIn(chr(0x2013), str(text))


@unittest.skipIf(_offline_dir() is None, "needs tools/offline (yamkin)")
class Clearances(unittest.TestCase):
    """The offline checker sees each binding only as its strap column, so it does not check the open fingers against
    the highback and the baseplate. This repeats that check with the checker's own models (finger pads as 8 x 20 x 38
    mm boxes beside the gap, link capsules) at the pick and the place: pads 20 mm from the bar's faces and 47 mm from
    the highback with the fingers open 70 mm, the gripper body 27 mm from the highback (the right arm: the housing
    sits 27 mm to one side of the closing axis, toward the tail binding's highback; the left arm's is 45 mm)."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_offline_dir()))
        cls.K = importlib.import_module("yamkin")
        cls.task = importlib.import_module("task")

    def parts(self, origin, angle):
        a = math.radians(angle)
        r = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
        return {n: (np.asarray(origin) + r @ np.array(c), r, np.array(s) / 2) for n, c, s in L.PART_BOXES}

    @staticmethod
    def pt_box(p, box):
        c, r, h = box
        q = np.abs(r.T @ (p - c)) - h
        return float(np.linalg.norm(np.maximum(q, 0)) + min(q.max(), 0))

    def pads(self, p, r, gap):
        mid = (-0.025 + 0.013) / 2
        return [(p + r[:, 0] * s * (gap / 2 + 0.004) + r[:, 2] * mid, r, np.array([0.004, 0.010, 0.019]))
                for s in (-1, 1)]

    def box_box(self, a, b, n=9):
        c, r, h = a
        g = np.linspace(-1, 1, n)
        return min(self.pt_box(c + r @ (h * np.array([i, j, k])), b) for i in g for j in g for k in g)

    def gaps(self, side, tcp_xy, h, yaw_deg, origin_xy, angle, gap):
        K = self.K
        base = K.BASES["station"][side]
        ok, q, _, _ = K.reachable((tcp_xy[0], tcp_xy[1], L.TOP + h), math.radians(yaw_deg), base)
        self.assertTrue(ok)
        p, r = K.tcp_pose(q, base)
        caps = K.arm().capsules(q, base, -gap / 2, links=("link3", "link4", "link5", "gripper"))
        parts = self.parts((*origin_xy, TOP), angle)
        out = {}
        for n, box in parts.items():
            body = min(min(self.pt_box(a + (b - a) * t, box) for t in np.linspace(0, 1, 30)) - rad
                       for _, a, b, rad in caps)
            pad = min(self.box_box(pd, box) for pd in self.pads(p, r, gap))
            out[n] = (body * 1000, pad * 1000)
        return out

    def test_pick_and_place_clearances(self):
        body = {}
        for side in ("left", "right"):
            x, y, a = L.NOSE_START if side == "left" else L.TAIL_START
            pick = (x + math.cos(math.radians(a)) * L.STRAP_U, y + math.sin(math.radians(a)) * L.STRAP_U)
            px, py, yaw = self.task.place_point(side)
            target = FRONT if side == "left" else BACK
            for where, g in (("pick", self.gaps(side, pick, self.task.GRIP_H, 0.0, (x, y), a, 0.070)),
                             ("place", self.gaps(side, (px, py), self.task.PLACE_H - 0.002, math.degrees(yaw),
                                                 target[:2], target[2], 0.070))):
                self.assertAlmostEqual(g["strap"][1], 20.0, delta=0.5, msg=(side, where))  # open pads to the bar
                self.assertAlmostEqual(g["highback"][1], 47.0, delta=0.5, msg=(side, where))  # open pads, highback
                self.assertGreater(g["baseplate"][1], 14.0, msg=(side, where))  # pad tips over the baseplate
                self.assertGreater(g["strap"][0], 23.0, msg=(side, where))  # gripper body to the bar's top
                body[side, where] = g["highback"][0]  # gripper body to the highback
        self.assertAlmostEqual(min(body.values()), 27.4, delta=0.5)
        self.assertAlmostEqual(body["right", "pick"], 27.4, delta=0.5)
        self.assertAlmostEqual(body["left", "pick"], 45.2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
