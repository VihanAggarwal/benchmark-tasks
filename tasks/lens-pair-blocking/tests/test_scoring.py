"""Offline tests for scoring.py (and the rigstate hooks in task.py): synthetic block states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

import importlib
import itertools
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, angle_to_line_deg, corners, low_above_lens, on_lens, target_distance, tilt_deg, valid,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 control steps
SETTLE = WINDOW + 2  # enough still steps for the 0.5 s window to fill
AWAY = {"left": np.array([0.323, 0.305, L.TOP + 0.26]), "right": np.array([0.323, -0.305, L.TOP + 0.26])}
REST_Z = L.TOP + L.MARK_TOP_H + L.BLOCK_HALF[2]  # a flat block's centre resting on the marks (0.1 mm proud)
SUMS = {0.0, 0.25, 0.5, 0.75, 1.0}


def qmul(a, b):
    x1, y1, z1, w1 = a
    x2, y2, z2, w2 = b
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a world axis."""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    h = math.radians(deg) / 2
    return np.array([*(math.sin(h) * a), math.cos(h)])


def blk(center, yaw_deg=0.0, pre=None, vel=(0, 0, 0)):
    """An ObjectState-like block: ``pre`` (a body-frame quaternion, e.g. a tilt or a flip) then a yaw of
    ``yaw_deg`` about the vertical; the world bounding box of its corners."""
    q = about((0, 0, 1), yaw_deg)
    if pre is not None:
        q = qmul(q, pre)
    s = SimpleNamespace(pos=np.array(center, float), quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                        low=np.zeros(3), high=np.zeros(3))
    c = corners(s)
    s.low, s.high = c.min(axis=0), c.max(axis=0)
    return s


def heading_yaw(side, dyaw=0.0):
    """Block yaw that puts its 24 mm (body y) axis along lens ``side``'s line, turned ``dyaw`` deg more."""
    return L.line_heading_deg(side) - 90.0 + dyaw


def placed(side, along=None, across=0.0, dyaw=0.0, lift=0.0, pre=None):
    """A block flat on lens ``side``: its centre ``along`` (default 8 mm) temporal of the OC along the line and
    ``across`` to the left of it, its long side turned ``dyaw`` from the line, ``lift`` above resting on the marks."""
    along = L.DECENTRATION if along is None else along
    (ox, oy), (ux, uy) = L.OC[side], L.temporal(side)
    x, y = ox + along * ux - across * uy, oy + along * uy + across * ux
    return blk((x, y, REST_Z + lift), heading_yaw(side, dyaw), pre)


def at(side, x, y, dyaw=0.0):
    """A block flat on lens ``side``'s marks at (x, y), its long side ``dyaw`` from the line."""
    return blk((x, y, REST_Z), heading_yaw(side, dyaw))


def start_blocks():
    return {n: blk((*L.BLOCK_START[n], L.TOP + L.BLOCK_HALF[2])) for n in L.BLOCKS}


def both(left=None, right=None):
    """The two blocks: block_left on the left lens's target and block_right on the right's unless given."""
    return {"block_left": placed("left") if left is None else left,
            "block_right": placed("right") if right is None else right}


def snap(left=None, right=None, lo=1.0, ro=1.0):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does; return rigstate.arms()."""
    feed = {}
    for side, tcp, op in (("left", left, lo), ("right", right, ro)):
        tcp = AWAY[side] if tcp is None else np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, 0.013]), "opening": op}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def hold(blocks, n=SETTLE, t=None, arms="away"):
    """Feed the same blocks n steps (arms away and open unless given); return the last result and the tracker."""
    t = t or Tracker(HZ)
    r = None
    for _ in range(n):
        a = snap() if isinstance(arms, str) else arms
        r = t.update(blocks, a)
    return r, t


def score_of(blocks, **kw):
    return hold(blocks, **kw)[0].score


def lens(result, side):
    return result.details["lenses"][side]


class Geometry(unittest.TestCase):
    def test_targets_and_marks(self):
        self.assertAlmostEqual(L.DECENTRATION, (54 + 20 - 58) / 2 / 1000, places=12)
        self.assertAlmostEqual(L.TARGET["left"][0], 0.2966, delta=5e-5)
        self.assertAlmostEqual(L.TARGET["left"][1], 0.2473, delta=5e-5)
        self.assertAlmostEqual(L.TARGET["right"][0], 0.2960, delta=5e-5)
        self.assertAlmostEqual(L.TARGET["right"][1], -0.2469, delta=5e-5)
        self.assertAlmostEqual(L.NASAL_XY["left"][0], 0.3110, delta=5e-5)
        self.assertAlmostEqual(L.NASAL_XY["left"][1], 0.2164, delta=5e-5)
        self.assertAlmostEqual(L.NASAL_XY["right"][0], 0.3130, delta=5e-5)
        self.assertAlmostEqual(L.NASAL_XY["right"][1], -0.2175, delta=5e-5)
        self.assertEqual(L.line_heading_deg("left"), 115.0)
        self.assertEqual(L.line_heading_deg("right"), 60.0)
        for side in L.SIDES:  # the nasal end points toward the midline, the temporal end away from it
            self.assertLess(abs(L.NASAL_XY[side][1]), abs(L.OC[side][1]))
            self.assertGreater(abs(L.TARGET[side][1]), abs(L.OC[side][1]))
        self.assertAlmostEqual(L.MARK_TOP_H - L.LENS_TOP_H, 0.0001, places=9)

    def test_a_placed_block_stays_on_the_lens_and_covers_the_dot(self):
        for side in L.SIDES:
            s = placed(side)
            c = corners(s)
            ox, oy = L.OC[side]
            far = max(math.hypot(x - ox, y - oy) for x, y, _ in c)
            self.assertAlmostEqual(far, math.hypot(0.008 + 0.012, 0.009), delta=1e-6)  # 21.9 mm, lens radius 35
            self.assertAlmostEqual(target_distance(s, side), 0.0, places=9)
            self.assertAlmostEqual(angle_to_line_deg(s, side), 0.0, places=6)
            self.assertAlmostEqual(low_above_lens(s), 0.0001, places=9)
            self.assertTrue(on_lens(s, side))
            self.assertFalse(on_lens(s, "right" if side == "left" else "left"))
            # it spans 4 mm nasal to 20 mm temporal of the OC along the line: the 8 mm dot is under it
            self.assertAlmostEqual(L.DECENTRATION - L.BLOCK_HALF[1], -L.DOT_R, places=9)

    def test_reference_habits(self):
        """The errors the task is built to catch, with the README's numbers."""
        for side, unturned in (("left", 25.0), ("right", 30.0)):
            oc = placed(side, along=0.0)
            self.assertAlmostEqual(target_distance(oc, side) * 1000, 8.0, places=6)
            nasal = placed(side, along=-L.DECENTRATION)
            self.assertAlmostEqual(target_distance(nasal, side) * 1000, 16.0, places=6)
            self.assertAlmostEqual(angle_to_line_deg(blk((*L.TARGET[side], REST_Z), 0.0), side), unturned, places=6)
        ox, oy = L.OC["left"]
        straight_l = at("left", ox, oy + 0.008)
        self.assertAlmostEqual(target_distance(straight_l, "left") * 1000, 3.46, delta=0.01)
        ox, oy = L.OC["right"]
        straight_r = at("right", ox, oy - 0.008)
        self.assertAlmostEqual(target_distance(straight_r, "right") * 1000, 4.14, delta=0.01)
        r, _ = hold(both(placed("left", along=0.0), placed("right", along=0.0)))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(lens(r, "left")["why"], "off target")
        self.assertEqual(score_of(both(placed("left", along=-L.DECENTRATION), placed("right", along=-L.DECENTRATION))),
                         0.0)
        unturned = both(blk((*L.TARGET["left"], REST_Z), 0.0), blk((*L.TARGET["right"], REST_Z), 0.0))
        self.assertEqual(score_of(unturned), 0.0)
        self.assertEqual(score_of(both(straight_l, straight_r)), 0.0)  # 3.46 and 4.14 mm: over 3 mm on both
        for side in L.SIDES:  # 8 mm from the dot's temporal or nasal edge instead of its centre: 4 mm off
            for along in (L.DECENTRATION + L.DOT_R, L.DECENTRATION - L.DOT_R):
                self.assertAlmostEqual(target_distance(placed(side, along=along), side) * 1000, 4.0, places=6)

    def test_folding(self):
        """A block turned half a turn, or either 18 x 24 face down, is the same block."""
        s = placed("left", dyaw=180.0)
        self.assertAlmostEqual(angle_to_line_deg(s, "left"), 0.0, places=6)
        flip = placed("right", pre=about((1, 0, 0), 180.0))
        self.assertAlmostEqual(tilt_deg(flip), 0.0, places=6)
        self.assertAlmostEqual(angle_to_line_deg(flip, "right"), 0.0, places=6)
        self.assertAlmostEqual(angle_to_line_deg(placed("left", dyaw=95.0), "left"), 85.0, places=6)
        self.assertAlmostEqual(angle_to_line_deg(placed("left", dyaw=-4.0), "left"), 4.0, places=6)
        self.assertEqual(score_of(both(s, flip)), 1.0)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = hold(start_blocks())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        for side in L.SIDES:
            self.assertEqual(lens(r, side)["why"], "no block")

    def test_every_lens_tier_and_every_sum(self):
        tier = {0.0: lambda s: placed(s, along=0.0), 0.25: lambda s: placed(s, across=0.0025), 0.5: placed}
        seen = set()
        for a, b in itertools.product(tier, repeat=2):
            r, _ = hold(both(tier[a]("left"), tier[b]("right")))
            self.assertEqual(lens(r, "left")["lens_score"], a)
            self.assertEqual(lens(r, "right")["lens_score"], b)
            self.assertEqual(r.score, a + b)
            self.assertEqual(r.success, a + b == 1.0)
            seen.add(r.score)
        self.assertEqual(seen, SUMS)

    def test_one_lens_alone(self):
        far = blk((0.45, 0.10, L.TOP + L.BLOCK_HALF[2]))  # the other block left on the table
        self.assertEqual(score_of(both(right=far)), 0.5)
        self.assertEqual(score_of(both(left=far)), 0.5)
        self.assertEqual(score_of(start_blocks() | {"block_left": placed("left")}), 0.5)

    def test_d_band_edges(self):
        for side in L.SIDES:
            for off, want in ((0.0019, 0.5), (0.0021, 0.25), (0.0029, 0.25), (0.0031, 0.0)):
                for along, across in ((L.DECENTRATION + off, 0.0), (L.DECENTRATION, -off)):
                    s = placed(side, along=along, across=across)
                    r, _ = hold({**both(), f"block_{side}": s})
                    self.assertEqual(lens(r, side)["lens_score"], want, (side, off, along, across))

    def test_a_band_edges(self):
        for side in L.SIDES:
            for dyaw, want in ((2.9, 0.5), (-3.1, 0.25), (5.9, 0.25), (-6.1, 0.0)):
                r, _ = hold({**both(), f"block_{side}": placed(side, dyaw=dyaw)})
                self.assertEqual(lens(r, side)["lens_score"], want, (side, dyaw))

    def test_both_bands_must_hold(self):
        self.assertEqual(lens(hold(both(placed("left", across=0.0019, dyaw=3.1)))[0], "left")["lens_score"], 0.25)
        self.assertEqual(lens(hold(both(placed("left", across=0.0031, dyaw=0.5)))[0], "left")["lens_score"], 0.0)
        self.assertEqual(lens(hold(both(placed("left", across=0.0005, dyaw=6.1)))[0], "left")["lens_score"], 0.0)


class Rules(unittest.TestCase):
    def test_flat_within_3_deg(self):
        for axis in ((1, 0, 0), (0, 1, 0)):
            ok = placed("left", lift=0.0004, pre=about(axis, 2.9))
            bad = placed("left", lift=0.0004, pre=about(axis, 3.1))
            self.assertEqual(lens(hold(both(ok))[0], "left")["lens_score"], 0.5)
            r, _ = hold(both(bad))
            self.assertEqual(lens(r, "left")["lens_score"], 0.0)
            self.assertIn("flat", lens(r, "left")["why"])

    def test_rocking_on_the_ink_line(self):
        """The marks stand 0.1 mm proud, so a placed block rests on the 4 mm line under its long axis and can rock
        sideways until its edge, 7 mm beyond the line's edge, touches the lens: atan(0.1 / 7), 0.8 deg. Rocked over
        and resting, it is flat and seated (0.5); rocking to and fro on every step (0.8 deg a step, 8 deg in 0.5 s)
        is not still (0), and it counts again once it has rested 0.5 s."""
        rock = math.degrees(math.atan2(L.MARK_TOP_H - L.LENS_TOP_H, 0.009 - L.LINE_SIZE[1] / 2))
        self.assertAlmostEqual(rock, 0.82, delta=0.005)
        for side in L.SIDES:
            s = placed(side, pre=about((0, 1, 0), rock))  # about the block's 24 mm axis, along the line
            s.pos = s.pos + [0, 0, -low_above_lens(s)]  # its lowest edge on the lens top
            self.assertAlmostEqual(tilt_deg(s), rock, places=6)
            self.assertAlmostEqual(angle_to_line_deg(s, side), 0.0, places=6)
            r, _ = hold({**both(), f"block_{side}": s})
            self.assertEqual(lens(r, side)["lens_score"], 0.5, side)
            t = Tracker(HZ)
            for j in range(SETTLE):
                r = t.update({**both(), f"block_{side}": s if j % 2 else placed(side)}, snap())
            self.assertEqual(lens(r, side)["lens_score"], 0.0, side)
            self.assertEqual(lens(r, side)["why"], "still")
            self.assertGreater(lens(r, side)["turn_deg"], 2.0)
            r, _ = hold({**both(), f"block_{side}": s}, n=WINDOW + 1, t=t)
            self.assertEqual(lens(r, side)["lens_score"], 0.5, side)

    def test_on_its_side_or_on_end(self):
        for pre in (about((1, 0, 0), 90.0), about((0, 1, 0), 90.0)):
            s = placed("right", pre=pre)
            s.pos = s.pos + [0, 0, -low_above_lens(s) + 0.0001]
            r, _ = hold(both(right=blk(s.pos, heading_yaw("right"), pre)))
            self.assertEqual(r.score, 0.5)
            self.assertIn("flat", lens(r, "right")["why"])

    def test_seated_within_2_mm(self):
        self.assertEqual(lens(hold(both(placed("left", lift=0.0018)))[0], "left")["lens_score"], 0.5)
        r, _ = hold(both(placed("left", lift=0.0021)))
        self.assertEqual(lens(r, "left")["lens_score"], 0.0)
        self.assertIn("seated", lens(r, "left")["why"])

    def test_tilted_on_the_lens_edge_or_half_on_the_pad(self):
        """Hanging off the lens's edge with one end on the pad: tilted and low, 0."""
        side = "left"
        (ox, oy), (ux, uy) = L.OC[side], L.temporal(side)
        tilt = math.degrees(math.atan2(L.LENS_T, 0.024))  # resting on the lens edge and the pad: 9.5 deg
        c = (ox + 0.030 * ux, oy + 0.030 * uy, L.TOP + L.LENS_Z0 + 0.010)
        s = blk(c, heading_yaw(side), about((1, 0, 0), tilt))
        r, _ = hold(both(s))
        self.assertEqual(lens(r, side)["lens_score"], 0.0)
        self.assertIn("flat", lens(r, side)["why"])
        # flat on the pad beside the lens (centre 36 mm from the OC): not on the lens at all
        s = blk((ox + 0.036 * ux, oy + 0.036 * uy, L.TOP + L.PAD_SIZE[2] + L.BLOCK_HALF[2]), heading_yaw(side))
        r, _ = hold(both(s))
        self.assertEqual(lens(r, side)["why"], "no block")

    def test_on_lens_radius_and_height(self):
        side = "right"
        (ox, oy), (ux, uy) = L.OC[side], L.temporal(side)
        self.assertTrue(on_lens(at(side, ox + 0.0349 * ux, oy + 0.0349 * uy), side))
        self.assertFalse(on_lens(at(side, ox + 0.0351 * ux, oy + 0.0351 * uy), side))
        s = placed(side)
        s.pos = s.pos + [0, 0, 0.059 - (REST_Z - L.LENS_TOP_Z)]
        self.assertTrue(on_lens(s, side))
        s.pos = s.pos + [0, 0, 0.002]
        self.assertFalse(on_lens(s, side))

    def test_two_blocks_on_one_lens(self):
        side_by_side = both(placed("left", across=0.0), placed("left", across=0.020))
        r, _ = hold(side_by_side)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(lens(r, "left")["why"], "two blocks")
        self.assertEqual(lens(r, "right")["why"], "no block")
        stacked = placed("left", lift=L.BLOCK_SIZE[2])
        r, _ = hold(both(placed("left"), stacked))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(lens(r, "left")["n_on"], 2)

    def test_a_block_held_over_a_blocked_lens(self):
        """One block placed, the other held 50 mm over the same lens: two blocks on it, 0. Held 61 mm up it is
        not on the lens (but it is not on the other one either)."""
        hover = placed("right", lift=0.050)
        r, _ = hold(both(placed("right"), hover))
        self.assertEqual(lens(r, "right")["why"], "two blocks")
        hover = placed("right", lift=0.061 - (REST_Z - L.LENS_TOP_Z))  # its centre 61 mm above the lens top
        self.assertAlmostEqual(hover.pos[2] - L.LENS_TOP_Z, 0.061)
        r, _ = hold(both(placed("right"), hover))
        self.assertEqual(r.score, 0.5)
        self.assertEqual(lens(r, "right")["n_on"], 1)
        self.assertEqual(lens(r, "left")["why"], "no block")

    def test_still_by_path_over_half_a_second(self):
        t = Tracker(HZ)
        blocks = both()
        r = t.update(blocks, snap())
        self.assertEqual(r.score, 0.0)  # just arrived: not still until 0.5 s of poses are seen
        self.assertIn("still", lens(r, "left")["why"])
        for _ in range(WINDOW - 1):
            r = t.update(blocks, snap())
        self.assertEqual(r.score, 0.0)
        r = t.update(blocks, snap())
        self.assertEqual(r.score, 1.0)
        # a chattering velocity reading is ignored
        for _ in range(SETTLE):
            r = t.update(both(blk(placed("left").pos, heading_yaw("left"), vel=(0.04, -0.03, 0.02))), snap())
        self.assertEqual(r.score, 1.0)

    def test_still_edges(self):
        """A slide of 0.19 mm a step (1.9 mm in 0.5 s) is still, 0.21 mm (2.1 mm) is not; a turn of 0.19 deg a step
        (1.9 deg) is still, 0.21 deg (2.1 deg) is not."""
        for per, want in ((0.00019, 1.0), (0.00021, 0.5)):
            t = Tracker(HZ)
            for j in range(SETTLE):
                r = t.update(both(placed("left", along=L.DECENTRATION - 0.001 + per * j)), snap())
            self.assertEqual(r.score, want, per)
            self.assertAlmostEqual(lens(r, "left")["path_mm"], per * WINDOW * 1000, delta=0.01)
        for per, want in ((0.19, 1.0), (0.21, 0.5)):
            t = Tracker(HZ)
            for j in range(SETTLE):
                r = t.update(both(right=placed("right", dyaw=-1.0 + per * j)), snap())
            self.assertEqual(r.score, want, per)
            self.assertAlmostEqual(lens(r, "right")["turn_deg"], per * WINDOW, delta=0.01)

    def test_still_moving_at_the_end(self):
        """Dropped onto the lens, it counts only once its last 0.5 s are still."""
        t = Tracker(HZ)
        for lift in np.linspace(0.010, 0.0, 4):
            r = t.update(both(placed("left", lift=float(lift))), snap())
        for _ in range(WINDOW - 4):
            r = t.update(both(), snap())
        self.assertEqual(lens(r, "left")["lens_score"], 0.0)
        self.assertIn("still", lens(r, "left")["why"])
        r, _ = hold(both(), n=4, t=t)
        self.assertEqual(r.score, 1.0)

    def test_let_go_40_mm_and_half_open(self):
        s = placed("left")
        c = s.pos
        for arm in ("left", "right"):
            for dist, op, ok in ((0.039, 0.49, False), (0.041, 0.0, True), (0.020, 0.51, True), (0.011, 0.49, False)):
                kw = {arm: c + [0, 0, dist], ("lo" if arm == "left" else "ro"): op}
                r, _ = hold(both(s), arms=snap(**kw))
                self.assertEqual(lens(r, "left")["lens_score"], 0.5 if ok else 0.0, (arm, dist, op))
                if not ok:
                    self.assertIn("let_go", lens(r, "left")["why"])
                    self.assertTrue(lens(r, "left")["held"])
        # 39 mm sideways counts as much as up
        r, _ = hold(both(s), arms=snap(left=c + [0.039, 0, 0], lo=0.1))
        self.assertEqual(lens(r, "left")["lens_score"], 0.0)

    def test_half_open_cannot_hold_a_block(self):
        """Half open, the pads (47 mm apart) are wider than the block's longest diagonal (34 mm): a gripper the
        let-go rule passes cannot be holding the block."""
        gap = L.LET_GO_OPENING * 0.0939
        self.assertAlmostEqual(gap, 0.047, delta=0.0002)
        self.assertGreater(gap, math.sqrt(sum(v * v for v in L.BLOCK_SIZE)) + 0.010)  # 34 mm + 10 mm

    def test_unreadable_arms_give_no_credit(self):
        r, t = hold(both(), arms=None)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(t.arms_unavailable, SETTLE)
        broken = {"left": SimpleNamespace(tcp=np.array([np.nan, 0, 0]), opening=1.0),
                  "right": SimpleNamespace(tcp=AWAY["right"], opening=1.0)}
        self.assertEqual(hold(both(), arms=broken)[0].score, 0.0)
        self.assertEqual(hold(both(), arms={"left": None})[0].score, 0.0)
        r, _ = hold(both(), n=1, t=t)  # readable again: back to the end state's score
        self.assertEqual(r.score, 1.0)

    def test_every_outcome_is_one_of_the_listed_scores(self):
        rng = np.random.default_rng(7)
        t = Tracker(HZ)
        for _ in range(300):
            blocks = {}
            for n in L.BLOCKS:
                side = rng.choice(L.SIDES)
                s = placed(side, across=rng.uniform(-0.006, 0.006), dyaw=rng.uniform(-8, 8),
                           along=L.DECENTRATION + rng.uniform(-0.006, 0.006))
                blocks[n] = s
            r = t.update(blocks, snap(lo=rng.uniform(0, 1), ro=rng.uniform(0, 1)))
            self.assertIn(r.score, SUMS)
            self.assertEqual(r.success, r.score == 1.0)
            for side in L.SIDES:
                self.assertIn(lens(r, side)["lens_score"], (0.0, 0.25, 0.5))


class Exploits(unittest.TestCase):
    def test_holding_the_blocks_in_place(self):
        """Both blocks perfectly placed but still gripped (fingers on the 18 mm side, TCP 10 mm over the centre)."""
        blocks = both()
        arms = snap(left=blocks["block_left"].pos + [0, 0, 0.010], right=blocks["block_right"].pos + [0, 0, 0.010],
                    lo=0.106, ro=0.106)
        r, _ = hold(blocks, arms=arms)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        # one let go, the other still held: only the let-go lens counts
        arms = snap(left=blocks["block_left"].pos + [0, 0, 0.010], lo=0.106)
        self.assertEqual(hold(blocks, arms=arms)[0].score, 0.5)

    def test_pressing_down_with_a_closed_gripper(self):
        blocks = both()
        tip_on_top = blocks["block_right"].pos + [0.0, 0.0, L.BLOCK_HALF[2] + 0.013]  # pad tips on its top face
        r, _ = hold(blocks, arms=snap(right=tip_on_top, ro=0.0))
        self.assertEqual(r.score, 0.5)
        self.assertIn("let_go", lens(r, "right")["why"])

    def test_wrong_side_choices(self):
        """Blocking on the OC (8 mm off), on the nasal side (16 mm off) or unturned (25 or 30 deg off) scores 0,
        however exactly it is done; a block with its short side on the line (90 deg) scores 0."""
        for side in L.SIDES:
            for s in (placed(side, along=0.0), placed(side, along=-L.DECENTRATION), placed(side, dyaw=90.0),
                      blk((*L.TARGET[side], REST_Z), 0.0)):
                r, _ = hold({**both(), f"block_{side}": s})
                self.assertEqual(lens(r, side)["lens_score"], 0.0)
                self.assertEqual(r.score, 0.5)

    def test_8_mm_straight_out_along_y(self):
        """Measuring 8 mm along y instead of along the turned line, the block turned onto the line: 3.5 mm off on
        the left lens and 4.1 mm off on the right, 0 on both (the loose band is 3 mm, so the same mistake costs the
        same on both lenses whatever their lines' angles)."""
        (lx, ly), (rx, ry) = L.OC["left"], L.OC["right"]
        r, _ = hold(both(at("left", lx, ly + 0.008), at("right", rx, ry - 0.008)))
        self.assertEqual(lens(r, "left")["d_mm"], 3.46)
        self.assertEqual(lens(r, "right")["d_mm"], 4.14)
        for side in L.SIDES:
            self.assertEqual(lens(r, side)["lens_score"], 0.0)
            self.assertEqual(lens(r, side)["why"], "off target")
        self.assertEqual(r.score, 0.0)

    def test_measured_from_the_dots_edge(self):
        """The dot is 8 mm across. Measuring the 8 mm from its temporal edge (12 mm from the OC) or its nasal edge
        (4 mm from the OC) instead of its centre lands 4 mm off: 0 on both lenses. The instruction names the dot's
        centre as the optical centre; the loose band's edge is 3 mm (11 mm from the OC)."""
        for along in (L.DECENTRATION + L.DOT_R, L.DECENTRATION - L.DOT_R):
            r, _ = hold(both(placed("left", along=along), placed("right", along=along)))
            self.assertEqual(r.score, 0.0, along)
            for side in L.SIDES:
                self.assertEqual(lens(r, side)["d_mm"], 4.0)
                self.assertEqual(lens(r, side)["why"], "off target")
        for along, want in ((0.0109, 0.5), (0.0111, 0.0), (0.0121, 0.0)):
            self.assertEqual(score_of(both(placed("left", along=along), placed("right", along=along))), want, along)

    def test_the_same_mistake_scores_alike_on_both_lenses(self):
        """Every reference mistake earns the same on the left lens (line turned 25 deg) as on the right (30 deg)."""
        mistakes = {
            "on the OC": lambda side: placed(side, along=0.0),
            "nasal side": lambda side: placed(side, along=-L.DECENTRATION),
            "not turned": lambda side: blk((*L.TARGET[side], REST_Z), 0.0),
            "short side on the line": lambda side: placed(side, dyaw=90.0),
            "8 mm along y": lambda side: at(side, L.OC[side][0], L.OC[side][1] + (0.008 if side == "left" else -0.008)),
            "dot's temporal edge": lambda side: placed(side, along=L.DECENTRATION + L.DOT_R),
            "dot's nasal edge": lambda side: placed(side, along=L.DECENTRATION - L.DOT_R),
            "turned the wrong way": lambda side: placed(side, dyaw=-2 * (L.line_heading_deg(side) - 90.0)),
        }
        for name, make in mistakes.items():
            r, _ = hold(both(make("left"), make("right")))
            self.assertEqual(lens(r, "left")["lens_score"], lens(r, "right")["lens_score"], name)
            self.assertEqual(r.score, 0.0, name)

    def test_swapped_blocks_count(self):
        """The blocks are identical: either one on either lens counts."""
        self.assertEqual(score_of({"block_left": placed("right"), "block_right": placed("left")}), 1.0)

    def test_slid_into_place_counts_once_still(self):
        """Set down off target and slid along the lens to the target: the end state counts (the scorer allows
        sliding), once it has been still for 0.5 s."""
        t = Tracker(HZ)
        for k in range(20):
            r = t.update(both(placed("left", along=0.0 + 0.0004 * k)), snap())
        self.assertEqual(lens(r, "left")["lens_score"], 0.0)
        r, _ = hold(both(), t=t)
        self.assertEqual(r.score, 1.0)

    def test_knocked_off_after_placing(self):
        """Placed, then knocked off the lens onto its pad: 0 for that lens at the end."""
        r, t = hold(both())
        self.assertTrue(r.success)
        (ox, oy), (ux, uy) = L.OC["left"], L.temporal("left")
        on_pad = blk((ox + 0.045 * ux, oy + 0.045 * uy, L.TOP + L.PAD_SIZE[2] + L.BLOCK_HALF[2]), 33.0)
        r, _ = hold(both(on_pad), t=t)
        self.assertEqual(r.score, 0.5)
        self.assertEqual(lens(r, "left")["why"], "no block")

    def test_half_open_gripper_resting_on_the_block(self):
        """Both grippers parked with their TCPs 10 mm over the blocks: opening 0.49 counts as holding (0), 0.50 as
        let go (1.0); pads 47 mm apart cannot hold a block whose longest diagonal is 34 mm."""
        blocks = both()
        tl, tr = blocks["block_left"].pos + [0, 0, 0.010], blocks["block_right"].pos + [0, 0, 0.010]
        self.assertEqual(hold(blocks, arms=snap(tl, tr, 0.49, 0.49))[0].score, 0.0)
        self.assertEqual(hold(blocks, arms=snap(tl, tr, 0.50, 0.50))[0].score, 1.0)

    def test_second_block_held_just_inside_or_outside_60_mm(self):
        """The spare block held closed over the blocked left lens: 59.5 mm up it is on the lens (two blocks, 0);
        60.5 mm up it is not, and the left lens counts (0.5) but the empty right lens earns nothing."""
        for up, want, why in ((0.0595, 0.0, "two blocks"), (0.0605, 0.5, None)):
            spare = blk((*L.OC["left"], L.LENS_TOP_Z + up), heading_yaw("left"))
            r, _ = hold({"block_left": placed("left"), "block_right": spare},
                        arms=snap(left=spare.pos + [0, 0, 0.010], lo=0.19))
            self.assertEqual(r.score, want, up)
            self.assertEqual(lens(r, "left").get("why"), why)
            self.assertEqual(lens(r, "right")["why"], "no block")

    def test_creeping_under_the_stillness_limit_is_scored_where_it_ends(self):
        """A block creeping 0.19 mm a step (1.9 mm in 0.5 s, under the 2 mm path limit) counts as still, but it is
        scored where it is at the last step: 2.7 mm off after 15 steps, 0.25 for that lens."""
        t = Tracker(HZ)
        for i in range(15):
            r = t.update(both(placed("left", along=L.DECENTRATION + 0.00019 * i)), snap())
        self.assertTrue(lens(r, "left")["still"])
        self.assertEqual(lens(r, "left")["d_mm"], 2.66)
        self.assertEqual(lens(r, "left")["lens_score"], 0.25)
        self.assertEqual(r.score, 0.75)

    def test_arms_unreadable_on_the_last_step_only(self):
        """Fifteen good steps, then one on which the arms cannot be read: that step scores 0 (no evidence of let-go),
        so a trial cannot be ended or scored on a blind step."""
        r, t = hold(both(), n=15)
        self.assertEqual(r.score, 1.0)
        r = t.update(both(), None)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_swapped_and_flipped(self):
        """Each block on the other's lens, upside down and turned half a turn: the same blocks, 1.0."""
        fl = placed("left", pre=about((0, 1, 0), 180.0))
        fr = placed("right", pre=about((1, 0, 0), 180.0), dyaw=180.0)
        self.assertEqual(score_of({"block_left": fr, "block_right": fl}), 1.0)

    def test_hovering_just_above_the_lens(self):
        """Held 2.1 mm above the lens with the gripper's TCP 40 mm away is not seated (and in a real trial a block
        cannot float)."""
        r, _ = hold(both(placed("left", lift=0.0021)))
        self.assertIn("seated", lens(r, "left")["why"])


class Robustness(unittest.TestCase):
    def test_invalid_states_score_zero_and_leave_the_tracker(self):
        r, t = hold(both())
        self.assertEqual(r.score, 1.0)
        before = {n: [(p.copy(), q.copy()) for p, q in h] for n, h in t.hist.items()}
        for bad in ("nan", "inf", "zero_quat", "missing", "short_pos", "no_attr"):
            blocks = both()
            s = blocks["block_left"]
            if bad == "nan":
                s.pos = np.array([np.nan, 0.0, 0.0])
            elif bad == "inf":
                s.lin_vel = np.array([np.inf, 0.0, 0.0])
            elif bad == "zero_quat":
                s.quat_xyzw = np.zeros(4)
            elif bad == "missing":
                del blocks["block_left"]
            elif bad == "short_pos":
                s.pos = np.array([0.1, 0.2])
            else:
                blocks["block_left"] = SimpleNamespace(pos=s.pos)
            r = t.update(blocks, snap())
            self.assertEqual(r.score, 0.0, bad)
            self.assertFalse(r.success)
            self.assertEqual(t.result(blocks, snap()).score, 0.0)
            for n, h in t.hist.items():
                self.assertEqual(len(h), len(before[n]))
                self.assertTrue(all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
                                    for a, b in zip(h, before[n])))
        self.assertEqual(t.bad_steps, 6)
        self.assertEqual(t.update(both(), snap()).score, 1.0)  # the next good step carries on where it was
        self.assertFalse(valid(SimpleNamespace(pos=[0, 0, 0])))

    def test_none_objects(self):
        self.assertEqual(Tracker(HZ).update({n: None for n in L.BLOCKS}, snap()).score, 0.0)
        self.assertEqual(Tracker(HZ).update(None, snap()).score, 0.0)
        self.assertEqual(Tracker(HZ).result(None, None).score, 0.0)

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(3)
        t = Tracker(HZ)
        for _ in range(300):
            blocks = {}
            for n in L.BLOCKS:
                q = rng.normal(size=4)
                c = (rng.uniform(0.15, 0.4), rng.uniform(-0.3, 0.3), L.TOP + rng.uniform(0.0, 0.08))
                s = blk(c, 0.0)
                s.quat_xyzw = q / np.linalg.norm(q)
                blocks[n] = s
            arms = snap(left=rng.uniform(-1, 1, 3), right=rng.uniform(-1, 1, 3), lo=rng.uniform(), ro=rng.uniform())
            r = t.update(blocks, arms)
            self.assertIn(r.score, SUMS)
            self.assertEqual(r.success, r.score == 1.0)

    def test_result_is_idempotent(self):
        r, t = hold(both())
        steps, hist = t.steps, {n: len(h) for n, h in t.hist.items()}
        a, b = t.result(both(), snap()), t.result(both(), snap())
        self.assertEqual(a.score, b.score)
        self.assertEqual(a.details["lenses"], b.details["lenses"])
        self.assertEqual(t.steps, steps)
        self.assertEqual({n: len(h) for n, h in t.hist.items()}, hist)


class Reset(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_episode(self):
        r, t = hold(both())
        self.assertTrue(r.success)
        r = t.update(start_blocks(), snap())  # both blocks back on their marks in one step
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(all(len(h) == 1 for h in t.hist.values()))

    def test_jittered_teleport_reset(self):
        r, t = hold(both())
        blocks = start_blocks()
        blocks["block_left"] = blk((L.BLOCK_START["block_left"][0] + 0.03, L.BLOCK_START["block_left"][1] - 0.02,
                                    L.TOP + L.BLOCK_HALF[2]), 7.0)
        blocks["block_right"] = blk((L.BLOCK_START["block_right"][0] - 0.025, L.BLOCK_START["block_right"][1] + 0.03,
                                     L.TOP + L.BLOCK_HALF[2]), -12.0)
        t.update(blocks, snap())
        self.assertEqual(t.steps, 1)

    def test_one_block_moved_then_reset(self):
        r, t = hold(start_blocks() | {"block_left": placed("left")})
        self.assertEqual(r.score, 0.5)
        t.update(start_blocks(), snap())
        self.assertEqual(t.steps, 1)

    def test_gradual_return_is_not_a_reset(self):
        r, t = hold(both())
        a, b = np.array(placed("left").pos), np.array([*L.BLOCK_START["block_left"], L.TOP + L.BLOCK_HALF[2]])
        c, d = np.array(placed("right").pos), np.array([*L.BLOCK_START["block_right"], L.TOP + L.BLOCK_HALF[2]])
        steps = t.steps
        for k in range(1, 31):
            f = k / 30
            t.update({"block_left": blk(a + f * (b - a), 0.0), "block_right": blk(c + f * (d - c), 0.0)}, snap())
        self.assertEqual(t.steps, steps + 30)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        r, t = hold(both())
        steps = t.steps
        t.update(both(left=blk((0.45, 0.10, L.TOP + L.BLOCK_HALF[2]), 0.0)), snap())  # thrown, not back on its mark
        self.assertEqual(t.steps, steps + 1)
        t.update(start_blocks() | {"block_right": blk((0.45, -0.1, L.TOP + 0.008))}, snap())  # one of two back
        self.assertEqual(t.steps, steps + 2)

    def test_no_reset_without_a_jump(self):
        r, t = hold(start_blocks())
        steps = t.steps
        t.update(start_blocks(), snap())
        self.assertEqual(t.steps, steps + 1)

    def test_rigstate_reset_hook(self):
        saved = dict(rigstate._RESET_CALLBACKS)
        try:
            t = Tracker(HZ)
            rigstate.on_reset(t.reset)
            hold(both(), t=t)
            ep = rigstate.episode()
            rigstate.offline_reset()
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertEqual(t.steps, 0)
            self.assertIsNone(rigstate.arms())
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
    """task.py loaded against the offline yam_arena stand-in: objects, hooks, the reset hook, trace, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        cls.json = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()

    def test_objects(self):
        by = {o.name: o for o in self.task.OBJECTS}
        self.assertEqual(len(by), len(self.task.OBJECTS))
        self.assertEqual(len(by), 12)
        self.assertEqual({n for n, o in by.items() if o.mass is not None}, set(L.BLOCKS))
        for n in L.BLOCKS:
            o = by[n]
            self.assertEqual(tuple(o.size), L.BLOCK_SIZE)
            self.assertEqual(o.mass, L.BLOCK_MASS)
            self.assertAlmostEqual(o.pos[2] - o.size[2] / 2, L.TOP)
            self.assertEqual(tuple(o.pos[:2]), L.BLOCK_START[n])
            self.assertEqual(o.yaw, 0.0)
        for side in L.SIDES:
            pad, lens_, dot = by[f"pad_{side}"], by[f"lens_{side}"], by[f"oc_dot_{side}"]
            line, nasal = by[f"line_{side}"], by[f"nasal_{side}"]
            self.assertAlmostEqual(pad.pos[2] + pad.size[2] / 2, lens_.pos[2] - lens_.height / 2)  # lens on its pad
            self.assertAlmostEqual(lens_.pos[2] + lens_.height / 2, L.LENS_TOP_Z)
            self.assertEqual(lens_.radius, L.LENS_R)
            for m in (line, dot, nasal):
                top = m.pos[2] + (m.size[2] if hasattr(m, "size") else m.height) / 2
                self.assertAlmostEqual(top - L.LENS_TOP_Z, 0.0001, places=9)  # 0.1 mm proud
            self.assertAlmostEqual(math.degrees(line.yaw), L.line_heading_deg(side))
            self.assertAlmostEqual(math.degrees(nasal.yaw), L.line_heading_deg(side))
            self.assertEqual(tuple(dot.pos[:2]), L.OC[side])
            self.assertEqual(tuple(line.pos[:2]), L.OC[side])
            self.assertAlmostEqual(math.dist(nasal.pos[:2], L.OC[side]), L.NASAL_OFF)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertEqual(self.task.SCENE, "station")

    def test_instruction_matches_task_json_and_readme(self):
        ins = self.task.INSTRUCTION
        self.assertEqual(ins, self.json["instruction"])
        self.assertLessEqual(len(ins), 490)
        self.assertEqual(self.readme.count(f'--instruction "{ins}"'), 2)  # demo and Dream-1 commands
        self.assertIn("(54 + 20 - 58) / 2 = 8 mm", ins)
        self.assertIn("8 mm from the centre of the black dot (the optical centre)", ins)  # not from its edge
        self.assertNotIn(chr(0x2014), ins + json.dumps(self.json) + self.readme)  # no em-dashes
        self.assertIn("Possible scores: 0, 0.25, 0.5, 0.75 and 1.0", self.json["scoring"])
        self.assertNotIn("media", self.json)
        steps = int(re.match(r"([\d,]+) steps", self.json["run_limit"]).group(1).replace(",", ""))
        self.assertEqual(steps, 1400)
        demo_steps = sum(w.duration for w in self.task.demo()["left"]) * HZ
        self.assertTrue(3 * demo_steps <= steps <= 6 * demo_steps)

    def test_numbers_in_the_texts(self):
        text = self.json["scoring"] + self.json["setup"] + self.readme
        for s in ("35 mm", "60 mm", "3 degrees", "2 mm", "0.5 s", "40 mm", "half open", "3 mm", "6 degrees",
                  "8 mm", "16 mm", "3.5 mm", "4.1 mm", "4 mm off"):
            self.assertIn(s, text, s)
        for t in (self.json["scoring"], self.readme):  # the loose band as the code has it, never the old 4 mm
            self.assertIn("d at most 3 mm and a at most 6 degrees", t)
            self.assertNotIn("d at most 4 mm", t)
        self.assertAlmostEqual(L.D_LOOSE, 0.003)
        self.assertNotIn("tools/", self.readme)  # blind review: no paths outside this folder
        self.assertIn("18 x 24 x 16 mm", self.json["setup"])
        self.assertIn("10 g", self.json["setup"])

    def test_hooks_and_reset_hook(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(both()))  # arms None right after a reset: no crash, no credit
        self.assertEqual(tr.arms_unavailable, 1)
        for _ in range(SETTLE):
            snap()
            ok = self.task.success(both())
        self.assertTrue(ok)
        self.assertEqual(self.task.score(both()), 1.0)
        self.assertEqual(self.task.score(both()), 1.0)  # score() changes nothing
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        snap()
        self.assertEqual(self.task.score(both()), 0.0)  # no history yet: not still
        self.assertFalse(self.task.success({}))  # missing objects: no crash
        self.assertEqual(self.task.score({}), 0.0)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            try:
                os.environ["TASK_TRACE"] = str(path)
                rigstate.offline_reset()
                for _ in range(SETTLE):
                    snap()
                    self.task.success(both())
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(len(rows), SETTLE)
                last = rows[-1]
                self.assertEqual(last["score"], 1.0)
                for key in ("score", "success", "lenses", "left_tcp", "right_tcp", "left_opening", "right_opening",
                            "block_pose", "arms_unavailable", "episode", "step"):
                    self.assertIn(key, last)
                for key in ("n_on", "d_mm", "a_deg", "tilt_deg", "low_mm", "still", "held", "lens_score", "path_mm",
                            "turn_deg", "tcp_mm"):
                    self.assertIn(key, last["lenses"]["left"])
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                snap()
                self.task.success(both())  # must not raise
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, wps in plan.items():
            self.assertEqual(len(wps), 12)
            self.assertAlmostEqual(sum(w.duration for w in wps), 18.2)
            labels = [w.label for w in wps]
            close, release = labels.index("grip it across its 18 mm side"), labels.index(
                "let go: it drops 1 mm onto the lens")
            self.assertEqual(wps[close].finger, self.task.GRIP)
            self.assertEqual(wps[release].finger, self.task.OPEN)
            self.assertGreaterEqual(wps[close].duration, 1.5)
            self.assertGreaterEqual(wps[release].duration, 1.5)
            for i in (close, release):
                self.assertEqual(wps[i - 1].label, "settle")
                self.assertTrue(0.5 <= wps[i - 1].duration <= 0.8)
            self.assertAlmostEqual(self.task.GRIP, -(L.BLOCK_SIZE[0] - 0.008) / 2)
            pick, place = wps[close], wps[release]
            self.assertEqual(tuple(pick.pos[:2]), L.BLOCK_START[f"block_{side}"])
            self.assertAlmostEqual(pick.pos[2] - L.TOP, L.BLOCK_SIZE[2] + 0.002)
            self.assertAlmostEqual(place.pos[0], L.TARGET[side][0] + self.task.PLACE_TRIM[side][0])
            self.assertAlmostEqual(place.pos[1], L.TARGET[side][1] + self.task.PLACE_TRIM[side][1])
            # the block's bottom 1.0 mm above the marks when it is let go
            self.assertAlmostEqual(place.pos[2] - (pick.pos[2] - L.TOP) - (L.TOP + L.MARK_TOP_H), 0.001)
            # turning the gripper by (line heading - 90 deg) lays the block's 24 mm side on the line; the GPU yaw
            # trim is added on top
            self.assertAlmostEqual(math.degrees(place.yaw - pick.yaw),
                                   L.line_heading_deg(side) - 90.0 + self.task.PLACE_YAW_TRIM_DEG[side])
            self.assertAlmostEqual(math.degrees(pick.yaw), -90.0 if side == "left" else 90.0)
            descent = labels.index("lower it the last 29 mm slowly")
            self.assertGreaterEqual(wps[descent].duration, 1.5)
        # the trims measured on the first GPU run (block centre minus target, long side to the line)
        self.assertEqual(self.task.PLACE_TRIM, {"left": (0.0006, -0.0012), "right": (0.0007, 0.0014)})
        self.assertEqual(self.task.PLACE_YAW_TRIM_DEG, {"left": -0.2, "right": -0.6})
        for side in L.SIDES:  # the offline check sees each trim as the block's whole error: keep it in the 0.5 band
            self.assertLessEqual(math.hypot(*self.task.PLACE_TRIM[side]), 0.0016)
            self.assertLessEqual(abs(self.task.PLACE_YAW_TRIM_DEG[side]), 2.0)

    def test_demo_switches(self):
        for key, value, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                                 ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = value
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    @unittest.skipIf(_stub_dir() is None or not (_stub_dir().parent / "check_task.py").is_file(),
                     "needs tools/offline/check_task.py")
    def test_offline_playback_demo_and_each_arm_alone(self):
        """The offline checker's kinematic playback (in its own process): the demo 1.0, each arm alone 0.5."""
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for k in ("DEMO_IDLE_ARM", "DEMO_HOLD_ONLY", "TASK_TRACE"):
            env.pop(k, None)
        out = subprocess.run([sys.executable, str(_stub_dir().parent / "check_task.py"), str(HERE), "--one-arm",
                              "--no-reach", "--json", "-", "--quiet"], capture_output=True, text=True, env=env,
                             timeout=600)
        data, _ = json.JSONDecoder().raw_decode(out.stdout[out.stdout.index("{"):])
        self.assertEqual(data["playback"]["proxy"]["predicted_task_score"], 1.0)
        self.assertEqual(data["one_arm"], {"left_only": 0.5, "right_only": 0.5})


if __name__ == "__main__":
    unittest.main()
