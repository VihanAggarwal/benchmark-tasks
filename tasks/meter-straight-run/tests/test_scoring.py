"""Offline tests for scoring.py and task.py: synthetic part states and synthetic arm snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import itertools
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

import layout as L  # noqa: E402
import rigstate  # noqa: E402
from scoring import (  # noqa: E402
    Tracker, ends, evaluate, flat_deg, heading_err_deg, laid, lowest, point_box_distance, pose, rotation,
    segment_box_distance, tier, valid,
)

HZ = 20.0
DT = 1.0 / HZ
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
ZC = L.TOP + L.HALF_W  # a part's centre height lying on the table
TIPS = 0.013  # pad tips below the TCP
PARK_L = np.array([0.24, 0.34, L.TOP + 0.127])  # tips of the parked arms (TCP 0.14 m up)
PARK_R = np.array([0.24, -0.34, L.TOP + 0.127])


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll): a part turned yaw about the vertical after pitching its long
    axis (local x) and rolling about it."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def part(name, centre, q, vel=(0.0, 0.0, 0.0)):
    """A part's state as the simulator reports it: a straight as an ObjectState-like box, the meter as mechkit's
    AssemblyState (origin = its body's centre; pos is its outline box's centre, which the scorer must not use)."""
    c = np.asarray(centre, dtype=float)
    q = np.asarray(q, dtype=float)
    if name == "meter":
        return SimpleNamespace(origin=c, pos=c + rotation(q) @ np.array([0.0, 0.0, 0.0075]), quat_xyzw=q,
                               lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), low=c - 0.06, high=c + 0.06)
    half = np.array([L.LENGTH[name] / 2, L.HALF_W, L.HALF_W])
    ext = np.abs(rotation(q)) @ half
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.asarray(vel, float), ang_vel=np.zeros(3), low=c - ext,
                           high=c + ext)


def start():
    s = {n: part(n, (x, y, ZC), quat(0.0)) for n, (x, y, _, _) in L.PIPE_START.items()}
    s["meter"] = part("meter", (*L.METER_START, ZC), quat(0.0))
    return s


def run(order=("pipe_75", "pipe_250", "meter", "pipe_125"), arrow="down", joints=(0.0, 0.0, 0.0, 0.0), shift=0.0):
    """The four parts laid in the channel in ``order`` from the elbow, with the given joints (elbow joint, then each
    part-to-part joint, m), the meter's arrow toward the valve ('down') or the elbow ('up'), the whole run moved
    ``shift`` toward the valve."""
    s, y = {}, L.ELBOW_FACE_Y - shift
    for n, j in zip(order, joints):
        y -= j
        c = (L.RUN_X, y - L.LENGTH[n] / 2, ZC)
        yaw = 90.0 if n != "meter" else (-90.0 if arrow == "down" else 90.0)
        s[n] = part(n, c, quat(yaw))
        y -= L.LENGTH[n]
    return s


def arms(left=PARK_L, right=PARK_R):
    """A rigstate snapshot with the finger tips where given."""
    return {side: rigstate.Arm(tcp=np.asarray(t, float) + [0, 0, TIPS], tips=np.asarray(t, float), opening=1.0)
            for side, t in (("left", left), ("right", right))}


def moved(states, name, **kw):
    """A copy of ``states`` with one part replaced: centre ``c`` and/or orientation ``q``."""
    s = dict(states)
    old = s[name]
    c = kw.get("c", getattr(old, "origin" if name == "meter" else "pos"))
    s[name] = part(name, c, kw.get("q", old.quat_xyzw), kw.get("vel", (0.0, 0.0, 0.0)))
    return s


def poses(states):
    return {n: pose(n, states[n]) for n in L.PARTS}


def score_of(states, a=None):
    return evaluate(poses(states), arms() if a is None else a)[0]


def settled(states, a=None, n=WINDOW + 1, tr=None):
    tr = tr or Tracker(HZ)
    r = None
    for _ in range(n):
        r = tr.update(states, arms() if a is None else a)
    return tr, r


def expected(order, arrow):
    """Independent of the scorer: 1.0 exactly when the 250 is upstream of the meter and the 125 downstream of it
    (the four orders listed in task.json), the arrow toward the valve; 0.25 backwards; 0.5 otherwise."""
    if arrow == "up":
        return 0.25
    i = order.index("meter")
    return 1.0 if order.index("pipe_250") < i < order.index("pipe_125") else 0.5


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        s = start()
        self.assertTrue(all(valid(n, s[n]) for n in L.PARTS))
        self.assertTrue(Tracker.at_start(poses(s)))
        self.assertEqual(score_of(s), 0.0)
        self.assertIn("not_along_the_run", evaluate(poses(s), arms())[1])

    def test_demo_layout_numbers(self):
        p = poses(run())
        self.assertAlmostEqual(ends("pipe_75", *p["pipe_75"])[0], 0.290, places=9)
        self.assertAlmostEqual(ends("pipe_250", *p["pipe_250"])[1], -0.035, places=9)
        self.assertAlmostEqual(ends("meter", *p["meter"])[1], -0.135, places=9)
        self.assertAlmostEqual(ends("pipe_125", *p["pipe_125"])[1], -0.260, places=9)
        score, why, d = evaluate(p, arms())
        self.assertEqual((score, why), (1.0, ""))
        self.assertEqual(d["U_mm"], 325.0)
        self.assertEqual(d["D_mm"], 155.0)
        self.assertEqual(d["valve_gap_mm"], 30.0)
        self.assertEqual(d["order"], ["pipe_75", "pipe_250", "meter", "pipe_125"])

    def test_straights_lie_on_any_long_face_and_either_way_round(self):
        for roll in (0.0, 90.0, 180.0, 270.0):
            for yaw in (90.0, -90.0):
                s = moved(run(), "pipe_250", q=quat(yaw, 0.0, roll))
                self.assertLess(flat_deg("pipe_250", rotation(s["pipe_250"].quat_xyzw)), 1e-6)
                self.assertEqual(score_of(s), 1.0, (roll, yaw))

    def test_flat_boundary(self):
        r4, r6 = rotation(quat(90.0, 0.0, 4.0)), rotation(quat(90.0, 0.0, 6.0))
        self.assertAlmostEqual(flat_deg("pipe_75", r4), 4.0, places=6)
        self.assertTrue(laid("pipe_75", np.array([L.RUN_X, 0.25, ZC]), r4)[0])
        self.assertFalse(laid("pipe_75", np.array([L.RUN_X, 0.25, ZC]), r6)[0])
        self.assertAlmostEqual(flat_deg("pipe_75", rotation(quat(90.0, 0.0, 45.0))), 45.0, places=6)

    def test_meter_register_up(self):
        self.assertAlmostEqual(flat_deg("meter", rotation(quat(-90.0, 0.0, 90.0))), 90.0, places=6)
        self.assertAlmostEqual(flat_deg("meter", rotation(quat(-90.0, 0.0, 180.0))), 180.0, places=6)
        self.assertAlmostEqual(flat_deg("meter", rotation(quat(-90.0, 0.0, 4.0))), 4.0, places=6)

    def test_heading_boundary(self):
        self.assertAlmostEqual(heading_err_deg(rotation(quat(94.0))), 4.0, places=6)
        self.assertAlmostEqual(heading_err_deg(rotation(quat(-84.0))), 6.0, places=6)
        self.assertEqual(heading_err_deg(rotation(quat(0.0, 90.0))), 90.0)  # standing on an end

    def test_lowest_point_reads_the_meter_body_not_its_outline_centre(self):
        m = run()["meter"]
        c, r = pose("meter", m)
        self.assertAlmostEqual(lowest("meter", c, r), 0.0, places=9)
        np.testing.assert_allclose(c, [L.RUN_X, -0.085, ZC])  # the origin, not pos (which is 7.5 mm higher)

    def test_point_box_distance(self):
        c, r = pose("pipe_125", run()["pipe_125"])
        self.assertEqual(point_box_distance(c, c, r, "pipe_125"), 0.0)
        self.assertAlmostEqual(point_box_distance(c + [0, 0, L.HALF_W + 0.03], c, r, "pipe_125"), 0.03, places=9)
        self.assertAlmostEqual(point_box_distance(c + [0, 0.0625 + 0.01, 0], c, r, "pipe_125"), 0.01, places=9)
        cm, rm = pose("meter", run()["meter"])
        self.assertAlmostEqual(point_box_distance(cm + [0, 0, 0.0275 + 0.03], cm, rm, "meter"), 0.03, places=9)


class Orders(unittest.TestCase):
    def test_all_24_orders_both_ways(self):
        n1 = 0
        for order in itertools.permutations(L.PARTS):
            for arrow in ("down", "up"):
                got = score_of(run(order, arrow))
                self.assertEqual(got, expected(order, arrow), (order, arrow))
                n1 += got == 1.0
        self.assertEqual(n1, 4)

    def test_valid_orders_hold_at_the_worst_joints(self):
        for order in (("pipe_75", "pipe_250", "meter", "pipe_125"), ("pipe_250", "pipe_75", "meter", "pipe_125"),
                      ("pipe_250", "meter", "pipe_75", "pipe_125"), ("pipe_250", "meter", "pipe_125", "pipe_75")):
            for j in (0.005, -0.002):
                s = run(order, joints=(j,) * 4)
                self.assertEqual(score_of(s), 1.0, (order, j))

    def test_the_trap_order_is_half_even_with_every_joint_squeezed(self):
        trap = ("pipe_250", "pipe_125", "meter", "pipe_75")  # 375 mm before the meter, only 105 mm after
        for j in (0.0, -0.002, 0.005):
            score, why, d = evaluate(poses(run(trap, joints=(j,) * 4)), arms())
            self.assertEqual((score, why), (0.5, "downstream_short"), j)
            self.assertLess(d["D_mm"], 120.0)

    def test_250_downstream_is_half(self):
        s = run(("pipe_75", "pipe_125", "meter", "pipe_250"), joints=(0.005,) * 4)
        score, why, d = evaluate(poses(s), arms())
        self.assertEqual((score, why), (0.5, "upstream_short"))
        self.assertLessEqual(d["U_mm"], 215.0)

    def test_tier_boundaries(self):
        self.assertEqual(tier(0.245, 0.120, True), (1.0, ""))
        self.assertEqual(tier(0.2449, 0.200, True)[0], 0.5)
        self.assertEqual(tier(0.300, 0.1199, True)[0], 0.5)
        self.assertEqual(tier(0.300, 0.200, False)[0], 0.25)
        self.assertEqual(tier(0.100, 0.100, False)[0], 0.25)


class Joints(unittest.TestCase):
    def test_five_mm_is_closed_six_is_open(self):
        for k in range(4):
            j = [0.0] * 4
            j[k] = 0.005
            self.assertEqual(score_of(run(joints=j)), 1.0, k)
            j[k] = 0.006
            score, why, _ = evaluate(poses(run(joints=j)), arms())
            self.assertEqual((score, why), (0.0, "joint_open"), k)

    def test_minus_two_mm_is_closed_minus_three_overlaps(self):
        for k in range(4):
            j = [0.0] * 4
            j[k] = -0.002
            self.assertEqual(score_of(run(joints=j)), 1.0, k)
            j[k] = -0.003
            self.assertEqual(evaluate(poses(run(joints=j)), arms())[:2], (0.0, "joint_overlap"), k)

    def test_run_butted_against_the_valve_instead(self):
        s = run(shift=0.030)  # everything slid to the valve: the 30 mm gap is at the elbow
        self.assertEqual(evaluate(poses(s), arms())[:2], (0.0, "joint_open"))
        self.assertEqual(score_of(run(shift=0.005)), 1.0)

    def test_valve_gap_is_not_checked(self):
        s = run(joints=(0.0, 0.005, 0.005, 0.005))  # the gap at the valve shrinks to 15 mm: still fine
        self.assertEqual(score_of(s), 1.0)


class ZeroConditions(unittest.TestCase):
    def test_a_part_on_a_rail(self):
        s = moved(run(), "pipe_125", c=(L.RUN_X + 0.016, -0.1975, ZC + 0.008))  # bottom 8 mm up, on the far rail
        score, why, _ = evaluate(poses(s), arms())
        self.assertEqual(score, 0.0)
        self.assertTrue(why.startswith("pipe_125_"), why)

    def test_centre_line_boundary(self):
        self.assertEqual(score_of(moved(run(), "pipe_125", c=(L.RUN_X + 0.0079, -0.1975, ZC))), 1.0)
        self.assertEqual(evaluate(poses(moved(run(), "pipe_125", c=(L.RUN_X + 0.0081, -0.1975, ZC))), arms())[1],
                         "pipe_125_off_the_centre_line")

    def test_bottom_boundary(self):
        self.assertEqual(score_of(moved(run(), "pipe_125", c=(L.RUN_X, -0.1975, ZC + 0.0029))), 1.0)
        self.assertEqual(evaluate(poses(moved(run(), "pipe_125", c=(L.RUN_X, -0.1975, ZC + 0.0031))), arms())[1],
                         "pipe_125_not_on_the_table")

    def test_a_stacked_part(self):
        s = run(("pipe_75", "pipe_250", "meter", "pipe_125"))
        s = moved(s, "pipe_75", c=(L.RUN_X, 0.100, ZC + L.PIPE_W))  # lying on the 250
        self.assertEqual(score_of(s), 0.0)

    def test_meter_rolled_on_its_side_or_upside_down(self):
        for roll in (90.0, 180.0, -90.0):
            s = moved(run(), "meter", q=quat(-90.0, 0.0, roll))
            score, why, _ = evaluate(poses(s), arms())
            self.assertEqual((score, why), (0.0, "meter_meter_not_register_up"), roll)

    def test_a_part_turned_across_the_run(self):
        s = moved(run(), "pipe_75", q=quat(84.0))
        self.assertEqual(evaluate(poses(s), arms())[1], "pipe_75_not_along_the_run")
        self.assertEqual(score_of(moved(run(), "pipe_75", q=quat(86.0))), 1.0)

    def test_last_part_still_gripped(self):
        c = pose("pipe_125", run()["pipe_125"])[0]
        score, why, d = evaluate(poses(run()), arms(right=c + [0, 0, 0.0005]))
        self.assertEqual((score, why), (0.0, "gripper_not_released"))
        self.assertEqual(d["tips_clear_mm"]["right"], 0.0)

    def test_gripper_hovering_above_a_part_boundary(self):
        self.assertAlmostEqual(L.RELEASE, 0.065)
        c = pose("pipe_125", run()["pipe_125"])[0]
        top = c + [0, 0, L.HALF_W]
        score, why, _ = evaluate(poses(run()), arms(right=top + [0, 0, 0.0649]))
        self.assertEqual((score, why), (0.0, "gripper_not_released"))
        self.assertEqual(score_of(run(), arms(right=top + [0, 0, 0.0651])), 1.0)
        self.assertEqual(score_of(run(), arms(left=top + [0, 0, 0.0649])), 0.0)  # either gripper, any part
        self.assertEqual(score_of(run(), arms(left=top + [0, 0, 0.0651])), 1.0)

    def test_gripper_above_the_meter_register(self):
        cm, _ = pose("meter", run()["meter"])
        self.assertEqual(score_of(run(), arms(left=cm + [0, 0, 0.0275 + 0.0649])), 0.0)
        self.assertEqual(score_of(run(), arms(left=cm + [0, 0, 0.0275 + 0.0651])), 1.0)

    def test_release_radius_covers_an_open_finger(self):
        # An open finger's outer corner (57 mm out along the closing axis, 20 mm across) is 60.4 mm from the finger
        # line; the rule's 65 mm is beyond it, and the demo's parked finger tips (123 and 138 mm) are well beyond.
        self.assertGreater(L.RELEASE, math.hypot(*L.OPEN_FINGER_REACH))
        self.assertAlmostEqual(math.hypot(*L.OPEN_FINGER_REACH), 0.0604, places=4)
        d = evaluate(poses(run()), arms())[2]["tips_clear_mm"]
        self.assertEqual((d["left"], d["right"]), (123.1, 138.1))

    def test_three_parts_only(self):
        s = run()
        s["pipe_125"] = part("pipe_125", (0.15, -0.42, ZC), quat(0.0))  # left where it started
        self.assertEqual(score_of(s), 0.0)


class Exploits(unittest.TestCase):
    def test_a_part_laid_on_the_table_past_the_elbow(self):
        # The 75 lies along the line x 0.300 beyond the elbow (y 0.35-0.425): laid, but the 'elbow joint' would be
        # -135 mm. The run from the 250 on is butted against the elbow.
        s = run(("pipe_250", "meter", "pipe_125", "pipe_75"))
        s = moved(s, "pipe_75", c=(L.RUN_X, 0.3875, ZC))
        self.assertTrue(laid("pipe_75", *pose("pipe_75", s["pipe_75"]))[0])
        score, why, d = evaluate(poses(s), arms())
        self.assertEqual(score, 0.0)
        self.assertIn(why, ("joint_open", "joint_overlap"))
        self.assertLess(d["joints_mm"][0], -100.0)  # the 'elbow joint' is far below -2 mm

    def test_a_part_laid_past_the_valve(self):
        s = run(("pipe_75", "pipe_250", "meter", "pipe_125"))
        s = moved(s, "pipe_125", c=(L.RUN_X, -0.40, ZC))  # beyond the valve's far side
        self.assertEqual(score_of(s), 0.0)

    def test_holding_the_last_part_in_place_instead_of_letting_go(self):
        c = pose("pipe_125", run()["pipe_125"])[0]
        tr, r = settled(run(), arms(right=c + [0, 0, 0.0]))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_spreading_the_run_to_lengthen_the_downstream_run(self):
        # The trap order with its joints opened to make D 125 mm: the joints are then too wide.
        trap = ("pipe_250", "pipe_125", "meter", "pipe_75")
        self.assertEqual(evaluate(poses(run(trap, joints=(0.0, 0.007, 0.007, 0.0))), arms())[:2],
                         (0.0, "joint_open"))

    def test_the_meter_set_on_top_of_the_250_to_save_length(self):
        s = run(("pipe_75", "pipe_250", "meter", "pipe_125"))
        s = moved(s, "meter", c=(L.RUN_X, 0.10, ZC + L.PIPE_W))
        self.assertEqual(score_of(s), 0.0)

    def test_the_meter_backwards_scores_quarter(self):
        self.assertEqual(evaluate(poses(run(arrow="up")), arms())[:2], (0.25, "meter_backwards"))

    def test_an_open_finger_resting_on_top_of_a_part(self):
        # Fingers closing along x, the tips point 30.1 mm beside the 125 and 1 mm above its top: the open finger on
        # the arms' side (47 mm from that point) lies on the 125. Under a 30 mm rule this scored 1.0.
        tips = (L.RUN_X + L.HALF_W + 0.0301, -0.1975, L.TOP + 0.026)
        score, why, d = evaluate(poses(run()), arms(right=tips))
        self.assertEqual((score, why), (0.0, "gripper_not_released"))
        self.assertEqual(d["tips_clear_mm"]["right"], 30.1)

    def test_the_back_of_an_open_finger_pushing_a_parts_end(self):
        # Fingers closing along y, the tips point 53.5 mm past the 125's outlet end: the back of the finger nearer
        # the run (47-53.4 mm out) presses that end toward the elbow.
        tips = (0.300, -0.260 - 0.0535, L.TOP + 0.010)
        score, why, d = evaluate(poses(run()), arms(right=tips))
        self.assertEqual((score, why), (0.0, "gripper_not_released"))
        self.assertEqual(d["tips_clear_mm"]["right"], 53.5)

    def test_an_open_finger_on_the_meter(self):
        cm, _ = pose("meter", run()["meter"])
        tips = (L.RUN_X + L.HALF_W + 0.0301, cm[1], L.TOP + 0.040)
        self.assertEqual(evaluate(poses(run()), arms(left=tips))[:2], (0.0, "gripper_not_released"))

    def test_the_outer_corner_of_an_open_finger_against_a_part(self):
        # The tips point 57 mm out from the 125's side and 20 mm past its outlet end, 2 mm above the table: the outer
        # corner of the finger's 40 mm wide section touches the 125's end corner. 60.4 mm: not let go; 65.1: let go.
        tips = np.array([L.RUN_X + L.HALF_W + 0.057, -0.260 - 0.020, L.TOP + 0.002])
        score, why, d = evaluate(poses(run()), arms(right=tips))
        self.assertEqual((score, why), (0.0, "gripper_not_released"))
        self.assertEqual(d["tips_clear_mm"]["right"], 60.4)
        far = np.array([L.RUN_X + L.HALF_W + 0.057, -0.260 - 0.020 - 0.0329, L.TOP + 0.002])
        self.assertEqual(score_of(run(), arms(right=far)), 1.0)

    def test_a_tilted_gripper_with_its_fingers_beside_a_part(self):
        # The gripper lying on its side, pointing away from the run (toward -y), its tips point 70 mm past the 125's
        # outlet end: the tips alone are clear, but its fingers run 38 mm back toward the 125 and end 32 mm from it.
        tips = np.array([L.RUN_X, -0.260 - 0.070, L.TOP + 0.0125])
        tilted = {"left": arms()["left"], "right": rigstate.Arm(tcp=tips + [0, 0.013, 0], tips=tips, opening=1.0)}
        score, why, d = evaluate(poses(run()), tilted)
        self.assertEqual((score, why), (0.0, "gripper_not_released"))
        self.assertEqual(d["tips_clear_mm"]["right"], 32.0)
        # The same gripper pointing toward the run instead: its fingers run back away from it, the tips decide.
        toward = {"left": arms()["left"], "right": rigstate.Arm(tcp=tips - [0, 0.013, 0], tips=tips, opening=1.0)}
        self.assertEqual(score_of(run(), toward), 1.0)

    def test_a_cocked_short_part(self):
        # The 75 turned 9 degrees in the 37 mm channel (it fits up to about 9.4): not along the run.
        s = moved(run(), "pipe_75", q=quat(99.0))
        self.assertEqual(score_of(s), 0.0)


class Tracking(unittest.TestCase):
    def test_success_needs_half_a_second_at_rest(self):
        tr = Tracker(HZ)
        for k in range(WINDOW):
            r = tr.update(run(), arms())
            self.assertEqual(r.score, 1.0)
            self.assertFalse(r.success, k)
        r = tr.update(run(), arms())
        self.assertTrue(r.success)
        self.assertEqual(r.details["path_mm"], 0.0)

    def test_a_creeping_part_is_not_at_rest(self):
        tr = Tracker(HZ)
        settled(run(), tr=tr)
        r = None
        for k in range(1, 12):  # 0.25 mm a step: 2.5 mm in 0.5 s
            r = tr.update(moved(run(), "pipe_125", c=(L.RUN_X, -0.1975 - 0.00025 * k, ZC)), arms())
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        tr2 = Tracker(HZ)
        settled(run(), tr=tr2)
        for k in range(1, 12):  # 0.15 mm a step: 1.5 mm in 0.5 s
            r = tr2.update(moved(run(), "pipe_125", c=(L.RUN_X, -0.1975 - 0.00015 * k, ZC)), arms())
        self.assertTrue(r.success)

    def test_a_jittering_part_adds_up_its_path(self):
        tr = Tracker(HZ)
        r = None
        for k in range(30):  # 0.3 mm back and forth: velocity would say nothing, the path says 3 mm
            r = tr.update(moved(run(), "pipe_75", c=(L.RUN_X, 0.2525 - 0.0003 * (k % 2), ZC)), arms())
        self.assertFalse(r.success)
        self.assertGreater(r.details["path_mm"], 2.0)

    def test_velocity_chatter_does_not_matter(self):
        s = {n: SimpleNamespace(**{**vars(v), "lin_vel": np.array([0.04, -0.03, 0.02])}) for n, v in run().items()}
        _, r = settled(s)
        self.assertTrue(r.success)

    def test_score_is_idempotent(self):
        tr, r = settled(run())
        for _ in range(3):
            self.assertEqual(tr.result(run(), arms()).score, 1.0)
        self.assertEqual(tr.steps, WINDOW + 1)
        self.assertEqual(tr.update(run(), arms()).score, r.score)

    def test_unreadable_grippers_have_not_let_go_and_are_counted(self):
        # The arms cannot be read for 15 steps while the right gripper may still be clamped on the 125: 0, never a
        # success; a 'let go' that was not seen does not count.
        tr = Tracker(HZ)
        r = None
        for _ in range(15):
            r = tr.update(run(), None)
        self.assertEqual((r.score, r.success, r.details["reason"]), (0.0, False, "gripper_not_readable"))
        self.assertEqual(tr.arms_unavailable, 15)
        self.assertEqual(tr.result(run(), None).score, 0.0)  # score() reads the same
        for _ in range(WINDOW + 1):  # read again, both parked: 1.0 and a success
            r = tr.update(run(), arms())
        self.assertTrue(r.success)
        self.assertEqual(tr.arms_unavailable, 15)

    def test_each_gripper_is_judged_on_its_own(self):
        # One gripper unreadable must not excuse the other: the right still on the 125 while the left reads NaN.
        c = pose("pipe_125", run()["pipe_125"])[0]
        nan = rigstate.Arm(tcp=np.array([np.nan, 0, 0]), tips=np.array([np.nan, 0, 0]), opening=1.0)
        score, why, d = evaluate(poses(run()), {"left": nan, "right": arms(right=c)["right"]})
        self.assertEqual(score, 0.0)
        self.assertIn(why, ("gripper_not_readable", "gripper_not_released"))
        self.assertEqual(d["tips_clear_mm"]["right"], 0.0)
        self.assertIsNone(d["tips_clear_mm"]["left"])
        # Unreadable in any other way, with the other gripper parked: still 0, and nothing crashes.
        for bad in ({"left": nan, "right": arms()["right"]}, {"right": arms()["right"]}, {"left": None, "right": None},
                    {"left": SimpleNamespace(tips=PARK_L), "right": arms()["right"]},  # no TCP
                    {"left": SimpleNamespace(tcp=PARK_L, tips="far"), "right": arms()["right"]},
                    [arms()["left"], arms()["right"]], "arms", 0):
            score, why, d = evaluate(poses(run()), bad)
            self.assertEqual((score, why), (0.0, "gripper_not_readable"), bad)
            self.assertFalse(d["arms_available"])
        tr = Tracker(HZ)
        tr.update(run(), {"left": nan, "right": arms()["right"]})
        self.assertEqual(tr.arms_unavailable, 1)

    def test_segment_box_distance(self):
        c, r = pose("pipe_125", run()["pipe_125"])
        top = c + [0, 0, L.HALF_W]
        self.assertAlmostEqual(segment_box_distance(top + [0, 0, 0.05], top + [0, 0, 0.09], c, r, "pipe_125"), 0.05,
                               places=7)
        # a segment passing over the part, its ends far to either side: the middle decides
        self.assertAlmostEqual(segment_box_distance(top + [0.3, 0, 0.02], top + [-0.3, 0, 0.02], c, r, "pipe_125"),
                               0.02, places=7)
        self.assertEqual(segment_box_distance(c - [0.1, 0, 0], c + [0.1, 0, 0], c, r, "pipe_125"), 0.0)

    def test_offline_feed_reaches_the_scorer(self):
        rigstate.offline_feed({"left": {"tcp": PARK_L + [0, 0, TIPS], "tips": PARK_L, "opening": 1.0},
                               "right": {"tcp": PARK_R + [0, 0, TIPS], "tips": PARK_R, "opening": 1.0}})
        self.assertEqual(score_of(run(), rigstate.arms()), 1.0)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(run())
        hist = [h.copy() for h in tr.history]
        bads = [
            moved(run(), "pipe_250", c=(np.nan, 0.09, ZC)),
            moved(run(), "pipe_75", c=(np.inf, 0.25, ZC)),
            moved(run(), "pipe_125", q=np.zeros(4)),
            moved(run(), "meter", q=(np.nan, 0, 0, 1)),
        ]
        no_origin = run()
        no_origin["meter"] = SimpleNamespace(pos=np.zeros(3), quat_xyzw=quat(-90.0))  # not built as an assembly
        missing = run()
        del missing["pipe_75"]
        for s in (*bads, no_origin, missing, None, {n: None for n in L.PARTS}):
            r = tr.update(s, arms())
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertFalse(r.details["valid_state"])
        self.assertEqual(tr.bad_steps, 8)
        for a, b in zip(hist, tr.history):
            np.testing.assert_array_equal(a, b)
        self.assertTrue(tr.update(run(), arms()).success)  # and it carries on as before

    def test_random_states_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker(HZ)
        for _ in range(400):
            s = {n: part(n, rng.uniform([0.0, -0.6, L.TOP - 0.1], [0.6, 0.6, L.TOP + 0.3]), rng.normal(size=4))
                 for n in L.PARTS}
            a = arms(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 3))
            r = tr.update(s, a)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class Reset(unittest.TestCase):
    def played(self):
        tr, _ = settled(run(), n=40)
        self.assertEqual(tr.steps, 40)
        return tr

    def test_teleport_back_to_start_resets(self):
        tr = self.played()
        r = tr.update(start(), arms())
        self.assertEqual(tr.steps, 1)
        self.assertEqual(len(tr.history), 1)
        self.assertEqual(r.score, 0.0)

    def test_teleport_to_a_jittered_start_resets(self):
        tr = self.played()
        s = start()
        for k, n in enumerate(L.PARTS):
            c = getattr(s[n], "origin" if n == "meter" else "pos") + [0.03 * (-1) ** k, -0.02, 0.0]
            s = moved(s, n, c=c, q=quat(3.0 * (-1) ** k))
        tr.update(s, arms())
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr = self.played()
        a, b = run(), start()
        for k in range(1, 61):
            f = k / 60
            s = {}
            for n in L.PARTS:
                ca, cb = pose(n, a[n])[0], pose(n, b[n])[0]
                s[n] = part(n, ca + (cb - ca) * f, b[n].quat_xyzw, vel=(cb - ca) * HZ / 60)
            tr.update(s, arms())
        self.assertEqual(tr.steps, 100)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr = self.played()
        s = moved(run(), "pipe_250", c=(0.15, 0.30, ZC))  # knocked far in one step, not to its start
        tr.update(s, arms())
        self.assertEqual(tr.steps, 41)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        tr = self.played()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(tr.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertEqual(tr.steps, 0)
            self.assertEqual(len(tr.history), 0)
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
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, reset hook and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def feed_parked(self):
        rigstate.offline_feed({"left": {"tcp": PARK_L + [0, 0, TIPS], "tips": PARK_L, "opening": 1.0},
                               "right": {"tcp": PARK_R + [0, 0, TIPS], "tips": PARK_R, "opening": 1.0}})

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in L.PARTS:
            self.assertIn(n, names)
        meter = next(o for o in self.task.OBJECTS if o.name == "meter")
        self.assertAlmostEqual(meter.mass, L.METER_MASS)
        np.testing.assert_allclose(meter.size, (0.100, 0.025, 0.040), atol=1e-12)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")

    def test_offline_meter_state_reads_the_body_centre(self):
        meter = next(o for o in self.task.OBJECTS if o.name == "meter")
        lo, hi = meter.local_bounds()
        box_centre = np.array([0.30, -0.085, L.TOP + (hi[2] - lo[2]) / 2])
        st = meter.offline_state(SimpleNamespace(pos=box_centre, quat_xyzw=quat(-90.0), lin_vel=np.zeros(3),
                                                 ang_vel=np.zeros(3), low=box_centre - 0.05,
                                                 high=box_centre + 0.05))
        np.testing.assert_allclose(st.origin, [0.30, -0.085, ZC], atol=1e-12)

    def test_hooks_and_reset_hook(self):
        rigstate.offline_reset()
        self.assertFalse(self.task.success(start()))
        self.assertEqual(self.task.score(start()), 0.0)
        self.feed_parked()
        for _ in range(WINDOW + 1):
            ok = self.task.success(run())
        self.assertTrue(ok)
        self.assertEqual(self.task.score(run()), 1.0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(self.task._TRACKER.steps, 0)
        self.assertFalse(self.task.success(run()))  # arms None right after a reset: no crash, not yet still

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed_parked()
            self.assertFalse(self.task.success(start()))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_sequence(self):
        """Each arm comes to the channel only once the other has let go and risen (or is rising) out of the way."""
        plan = self.task.demo()

        def end(arm, label):
            t = 0.0
            for w in plan[arm]:
                t += w.duration
                if w.label.startswith(label):
                    return t
            raise KeyError(label)

        def start_of(arm, label):
            return end(arm, label) - next(w.duration for w in plan[arm] if w.label.startswith(label))

        self.assertGreaterEqual(start_of("right", "carry pipe_250"), end("left", "let go of pipe_75") - 1e-9)
        self.assertGreaterEqual(start_of("left", "carry meter"), end("right", "straight up") - 1e-9)
        self.assertGreaterEqual(start_of("right", "carry pipe_125"), end("left", "let go of meter") + 1.0 - 1e-9)
        for arm in ("left", "right"):
            for w in plan[arm]:
                if w.label.startswith(("grip", "let go")):
                    self.assertGreaterEqual(w.duration, 1.5)
        # the meter is turned clockwise seen from above (yaw 0 -> -pi/2), so its arrow ends toward the valve
        turn = next(w for w in plan["left"] if w.label.startswith("turn meter"))
        self.assertAlmostEqual(turn.yaw, -math.pi / 2)

    def test_instruction_is_the_same_everywhere(self):
        text = self.task.INSTRUCTION
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], text)
        readme = (HERE / "README.md").read_text()
        cmds = re.findall(r'--instruction "([^"]*)"', readme)
        self.assertEqual(len(cmds), 2)
        for c in cmds:
            self.assertEqual(c, text)
        self.assertLessEqual(len(text), 490)
        # It asks for what the scorer checks: let go AND move the grippers clear (65 mm), not just open them.
        self.assertTrue(text.endswith("Let go and move clear."))

    def test_start_layout_in_view_and_picked_where_it_lies(self):
        # Setup image: every part inside |y| 43.25 cm (the arms' start pose covers the table outboard of about 45 cm);
        # each short pipe lies 22.5 mm short of its side's long part, and the demo picks every part where it lies.
        sides = {}
        for n in L.PARTS:
            x, y = L.METER_START if n == "meter" else L.PIPE_START[n][:2]
            self.assertLessEqual(abs(y) + L.HALF_W, 0.4325 + 1e-9, n)
            sides.setdefault(y > 0, []).append((x - L.LENGTH[n] / 2, x + L.LENGTH[n] / 2))
        for spans in sides.values():
            (_, near_end), (far_start, _) = sorted(spans)
            self.assertAlmostEqual(far_start - near_end, 0.0225)
        plan = self.task.demo()
        for arm, n, dx in (("left", "pipe_75", 0.0), ("right", "pipe_250", -self.task.GRIP_250_OFF),
                           ("right", "pipe_125", 0.0)):
            grip = next(w for w in plan[arm] if w.label == f"grip {n}")
            np.testing.assert_allclose(grip.pos[:2], (L.PIPE_START[n][0] + dx, L.PIPE_START[n][1]), atol=1e-12)

    def test_meter_grip_keeps_the_pads_off_its_inlet_end(self):
        # The 27 mm wide pads end 11.5 mm short of the meter's inlet end, more than the 8 mm they can sit off the
        # commanded point, so neither catches the 250's end corner while the meter is slid to butt.
        margin = (L.METER_GRIP_LOCAL - L.PAD_WIDTH / 2) - (-L.METER_LEN / 2)
        self.assertAlmostEqual(margin, 0.0115)
        self.assertGreater(margin, L.PAD_OFFSET)
        self.assertAlmostEqual((L.PIPE_W - L.REGISTER[1][1]) / 2, 0.0045)  # pads 4.5 mm outside the register
        plan = self.task.demo()["left"]
        grip = next(w for w in plan if w.label == "grip meter")
        self.assertAlmostEqual(grip.pos[0], L.METER_START[0] + L.METER_GRIP_LOCAL)
        self.assertAlmostEqual(grip.pos[0], 0.235)
        # At contact the meter's inlet end meets the 250's outlet end (y -0.035); it is set down 12 mm short of that.
        contact = self.task.CONTACT["meter"]
        self.assertAlmostEqual(contact, -0.060)
        self.assertAlmostEqual(contact + L.METER_LEN / 2 + L.METER_GRIP_LOCAL, L.ELBOW_FACE_Y - 0.075 - 0.250)
        k = next(i for i, w in enumerate(plan) if w.label == "carry meter")
        down = next(w for w in plan[k:] if w.label == "slowly down into the channel")
        self.assertAlmostEqual(down.pos[1], -0.072)

    def test_numbers_agree(self):
        """Code, task.json and README, number for number."""
        spec = json.loads((HERE / "task.json").read_text())
        readme = " ".join((HERE / "README.md").read_text().split())
        plan = self.task.demo()
        steps = round(max(sum(w.duration for w in p) for p in plan.values()) * HZ)
        self.assertEqual(steps, 1076)
        self.assertIn("takes 53.8 s (1,076 control steps)", readme)
        self.assertEqual(spec["run_limit"], "4,000 steps (200 s at 20 Hz)")
        self.assertIn(f"{4000 / steps:.1f} times the demo", readme)  # 3.7
        self.assertEqual(readme.count("--max-steps 4000"), 2)
        for n in (f"{L.RELEASE * 1000:.0f} mm", f"{L.FINGER_LEN * 1000:.0f} mm", f"{L.U_MIN * 1000:.0f} mm",
                  f"{L.D_MIN * 1000:.0f} mm", f"{L.JOINT_MIN * 1000:.0f} mm", f"+{L.JOINT_MAX * 1000:.0f} mm",
                  f"{L.CENTRE_TOL * 1000:.0f} mm", f"{L.BOTTOM_TOL * 1000:.0f} mm", f"{L.FLAT_DEG:.0f} degrees",
                  f"{L.STILL_PATH * 1000:.0f} mm", f"{L.STILL_WINDOW_S} s", f"{self.task.SUCCESS_HOLD_S:.0f} s",
                  "60.4 mm"):
            self.assertIn(n, spec["scoring"], n)
            self.assertIn(n, readme, n)
        self.assertAlmostEqual(math.hypot(*L.OPEN_FINGER_REACH) * 1000, 60.4, places=1)

    def test_setup_and_context_wording(self):
        spec = json.loads((HERE / "task.json").read_text())
        for m in ("painted-aluminium lever", "painted-aluminium register block", "plastic strips", "white plastic tile",
                  "hollow brass body", "grey aluminium strips", "square aluminium tubes painted copper"):
            self.assertIn(m, spec["setup"], m)  # every object's material is given
        # butted straights inside U or D (the 1.0 orders 75-250-meter-125 and 250-meter-75-125) are straight pipe
        self.assertIn("Butted straights of the same bore count as straight pipe", spec["context"])
        self.assertNotIn("other fitting inside those lengths", spec["context"])


if __name__ == "__main__":
    unittest.main()
