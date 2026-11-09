"""Offline tests for scoring.py and task.py: synthetic propeller states, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import Tracker, placement, points, tilt_deg, valid  # noqa: E402

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
MOTOR_TOP_W = L.TOP + L.MOTOR_TOP  # world z of every motor's top face
MAT_TOP_W = L.TOP + L.MAT_TOP
FL, FR, RL, RR = "front_left", "front_right", "rear_left", "rear_right"
CW_PROPS = [n for n in L.PROPS if L.HAND[n] == "CW"]
CCW_PROPS = [n for n in L.PROPS if L.HAND[n] == "CCW"]


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll)."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def prop(x, y, foot_z, yaw=0.0, pitch=0.0, roll=0.0):
    """A propeller as mechkit reports it: ``origin`` is the centre of the hub's bottom face, local +z up the hub,
    local +x along the blades."""
    return SimpleNamespace(origin=np.array([x, y, foot_z], dtype=float), quat_xyzw=quat(yaw, pitch, roll),
                           lin_vel=np.zeros(3), ang_vel=np.zeros(3), source="test")


def on_motor(motor, dx=0.0, dy=0.0, dz=0.0, yaw=0.0, tilt=0.0):
    """Upright on a motor's top: the hub's bottom-face centre at (dx, dy) off the motor's axis, dz above its top."""
    x, y = L.MOTOR_XY[motor]
    return prop(x + dx, y + dy, MOTOR_TOP_W + dz, yaw, pitch=tilt)


def upside_down_on(motor, dx=0.0, dz=0.0, yaw=0.0):
    """Upside down (blades uppermost), the hub's top face on the motor's top."""
    x, y = L.MOTOR_XY[motor]
    return prop(x + dx, y, MOTOR_TOP_W + L.HUB_H + dz, yaw, roll=180.0)


def on_mat(name, dx=0.0, dy=0.0, yaw=0.0):
    x, y = L.PROP_START[name]
    return prop(x + dx, y + dy, MAT_TOP_W, yaw)


START = {n: on_mat(n) for n in L.PROPS}
# the finished layout as the demo leaves it: each motor's own hand, the two spares still on the mats
DONE_PLAN = {"prop_1": FL, "prop_2": RL, "prop_4": FR, "prop_5": RR}


def state(**kw):
    s = {n: (on_motor(DONE_PLAN[n]) if n in DONE_PLAN else on_mat(n)) for n in L.PROPS}
    s.update(kw)
    return s


def only(**fitted):
    """Props named in ``fitted`` on those motors (or a given state), every other one on its mat."""
    s = dict(START)
    for n, m in fitted.items():
        s[n] = on_motor(m) if isinstance(m, str) else m
    return s


def feed(tr, states):
    r = None
    for st in states:
        r = tr.update(st)
    return r


def settled(st, n=WINDOW + 1):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n)


def scored(st):
    return settled(st)[1].score


def detail(st):
    return settled(st)[1].details


class Geometry(unittest.TestCase):
    def test_x_quad_motor_directions(self):
        self.assertEqual(L.MOTOR_DIR, {FL: "CW", FR: "CCW", RL: "CCW", RR: "CW"})
        # diagonal motors turn the same way, neighbouring motors opposite ways
        self.assertEqual(L.MOTOR_DIR[FL], L.MOTOR_DIR[RR])
        self.assertEqual(L.MOTOR_DIR[FR], L.MOTOR_DIR[RL])
        for a, b in ((FL, FR), (FL, RL), (RR, FR), (RR, RL)):
            self.assertNotEqual(L.MOTOR_DIR[a], L.MOTOR_DIR[b])
        self.assertEqual(L.MOTOR_XY[FL], (0.38, 0.10))
        self.assertAlmostEqual(L.MOTOR_XY[RR][0], 0.22)
        self.assertAlmostEqual(L.MOTOR_XY[RR][1], -0.10)
        self.assertAlmostEqual(L.MOTOR_TOP, 0.038)

    def test_turning_vector_is_the_motor_direction(self):
        for m in L.MOTOR_XY:
            u, t = L.outboard(m), L.turning(m)
            self.assertAlmostEqual(u[0] * t[0] + u[1] * t[1], 0.0)  # tangent
            cross = u[0] * t[1] - u[1] * t[0]  # > 0: counter-clockwise seen from above
            self.assertAlmostEqual(cross, -1.0 if L.MOTOR_DIR[m] == "CW" else 1.0)

    def test_leading_edges_lead_in_the_propellers_own_direction(self):
        for hand, omega in (("CW", -1.0), ("CCW", 1.0)):
            for (ex, ey, _), sign in zip(L.edge_centers(hand), (1.0, -1.0)):
                self.assertAlmostEqual(ex, sign * L.EDGE_X)
                vel = np.cross([0.0, 0.0, omega], [ex, 0.0, 0.0])  # the blade's centre line, turning
                self.assertGreater(float(vel[1] * ey), 0.0)  # the strip is on the side that moves first
                vel_wrong = np.cross([0.0, 0.0, -omega], [ex, 0.0, 0.0])
                self.assertLess(float(vel_wrong[1] * ey), 0.0)

    def test_no_turn_about_the_vertical_changes_the_hand(self):
        """The chirality seen from above (blade direction x strip offset) keeps its sign at every yaw; mirror images
        have opposite signs."""
        for hand, want in (("CW", -1.0), ("CCW", 1.0)):
            for yaw in range(0, 360, 15):
                r = _rot(quat(yaw))
                for (ex, ey, ez) in L.edge_centers(hand):
                    blade = r @ np.array([math.copysign(1.0, ex), 0.0, 0.0])
                    offset = r @ np.array([0.0, ey, 0.0])
                    self.assertAlmostEqual(math.copysign(1.0, blade[0] * offset[1] - blade[1] * offset[0]), want)

    def test_arrows_turn_the_motors_way(self):
        for m in L.MOTOR_XY:
            mx, my = L.MOTOR_XY[m]
            parts = L.arrow_parts(m)
            self.assertEqual([p[0] for p in parts], ["arc1", "arc2", "arc3", "barb1", "barb2"])
            angs = [math.atan2(c[1] - my, c[0] - mx) for _, c, _, _, _ in parts[:3]]
            steps = [math.remainder(b - a, 2 * math.pi) for a, b in zip(angs, angs[1:])]
            for s in steps:  # from the trailing end to the head: decreasing angle = clockwise seen from above
                self.assertLess(s, 0.0) if L.MOTOR_DIR[m] == "CW" else self.assertGreater(s, 0.0)
            # the middle plate on the arm's outward line, the arc 32 mm from the axis
            u = L.outboard(m)
            mid = parts[1][1]
            r_mid = math.hypot(mid[0] - mx, mid[1] - my)
            self.assertAlmostEqual(((mid[0] - mx) * u[0] + (mid[1] - my) * u[1]) / r_mid, 1.0, places=6)
            self.assertAlmostEqual(r_mid, L.ARROW_R * math.cos(math.radians(L.ARC_DEG / L.ARC_SEGMENTS / 2)))
            # the head: both barbs lie behind the tip, against the direction of turn there
            a_end = angs[-1] + steps[-1] / 2
            tip = np.array([mx + L.ARROW_R * math.cos(a_end), my + L.ARROW_R * math.sin(a_end)])
            ahead = np.array([-math.sin(a_end), math.cos(a_end)]) * (1.0 if L.MOTOR_DIR[m] == "CCW" else -1.0)
            for _, c, _, _, _ in parts[3:]:
                self.assertLess(float((np.array(c) - tip) @ ahead), -0.003)

    def test_arrows_clear_of_the_frame_and_the_propellers_reach(self):
        for m in L.MOTOR_XY:
            mx, my = L.MOTOR_XY[m]
            for _, (cx, cy), yaw, size, z0 in L.arrow_parts(m):
                c, s = math.cos(yaw), math.sin(yaw)
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        px = cx + c * sx * size[0] / 2 - s * sy * size[1] / 2
                        py = cy + s * sx * size[0] / 2 + c * sy * size[1] / 2
                        # beyond the motor and the end of its arm (14 mm out, 8 mm half-width)
                        self.assertGreater(math.hypot(px - mx, py - my), math.hypot(L.MOTOR_R, L.ARM_W / 2) + 0.005)
                self.assertLessEqual(z0 + size[2], 0.004)

    def test_mats_and_starting_marks(self):
        self.assertEqual(sorted(CW_PROPS), ["prop_1", "prop_5", "prop_6"])
        self.assertEqual(sorted(CCW_PROPS), ["prop_2", "prop_3", "prop_4"])
        for side, sign in (("left", 1.0), ("right", -1.0)):
            names = [n for n in L.PROPS if math.copysign(1.0, L.PROP_START[n][1]) == sign]
            self.assertEqual(len(names), 3)
            hands = sorted(L.HAND[n] for n in names)
            self.assertIn("CW", hands)
            self.assertIn("CCW", hands)
            # each mat's third (spare) propeller has the hand the other side's rear motor must not have
            far_rear = RR if side == "left" else RL
            spare_hand = max(set(hands), key=hands.count)
            self.assertNotEqual(spare_hand, L.MOTOR_DIR[far_rear])
            ys = sorted(L.PROP_START[n][1] for n in names)
            for a, b in zip(ys, ys[1:]):
                self.assertAlmostEqual(b - a, 0.055)
            for n in names:
                x, y = L.PROP_START[n]
                # every blade and hub on its own mat
                self.assertLessEqual(abs(x - L.MAT_X) + L.BLADE_OUT, L.MAT_SIZE[0] / 2)
                self.assertLessEqual(abs(abs(y) - L.MAT_Y) + L.HUB_R, L.MAT_SIZE[1] / 2)
                # nothing in the arms' start-pose shadow of the screenshot (|y| > 0.45 m with x < 0.2 m)
                self.assertLess(abs(y) + L.BLADE_W, 0.45)
        self.assertLess(L.MAT_Y + L.MAT_SIZE[1] / 2, 0.45)

    def test_fitted_propellers_never_touch_each_other(self):
        xy = list(L.MOTOR_XY.values())
        gaps = [math.dist(a, b) for i, a in enumerate(xy) for b in xy[i + 1:]]
        self.assertGreater(min(gaps), 2 * L.BLADE_OUT + 0.030)  # at any yaw, 34 mm between neighbouring tips

    def test_points_and_placement_measures(self):
        st = on_motor(FL, dx=0.003, dy=-0.004, dz=0.001, yaw=33.0)
        foot, top, tip_a, tip_b = points(st)
        np.testing.assert_allclose(top - foot, [0.0, 0.0, L.HUB_H], atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(tip_a[:2] - tip_b[:2])), 2 * L.BLADE_OUT)
        p = placement(st)
        self.assertEqual((p["motor"], p["way"]), (FL, "up"))
        self.assertAlmostEqual(p["centre_off_mm"], 5.0)
        self.assertAlmostEqual(p["seat_mm"], 1.0)
        down = placement(upside_down_on(RR))
        self.assertEqual((down["motor"], down["way"]), (RR, "down"))
        self.assertAlmostEqual(tilt_deg(upside_down_on(RR)), 180.0)
        mat = placement(START["prop_1"])
        self.assertIsNone(mat["motor"])
        self.assertEqual(mat["nearest"], RL)


class Tiers(unittest.TestCase):
    def test_finished_layout_scores_one(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["motors_fitted"], 4)
        self.assertEqual(r.details["fitted"], {FL: ["prop_1"], FR: ["prop_4"], RL: ["prop_2"], RR: ["prop_5"]})
        self.assertEqual(r.details["zero_reason"], "")

    def test_each_fitted_motor_adds_a_quarter(self):
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(scored(only(prop_1=FL)), 0.25)
        self.assertEqual(scored(only(prop_1=FL, prop_2=RL)), 0.5)
        self.assertEqual(scored(only(prop_1=FL, prop_2=RL, prop_4=FR)), 0.75)
        self.assertEqual(scored(only(prop_1=FL, prop_2=RL, prop_4=FR, prop_5=RR)), 1.0)
        self.assertEqual(scored(only(prop_5=RR, prop_4=FR)), 0.5)  # the right arm's half

    def test_any_propeller_of_the_right_hand_counts(self):
        # spares and the other mat's propellers are the same parts: any CW on a CW motor, any CCW on a CCW one
        self.assertEqual(scored(only(prop_6=FL, prop_3=RL, prop_2=FR, prop_1=RR)), 1.0)
        self.assertEqual(scored(only(prop_5=FL, prop_4=RL)), 0.5)

    def test_any_yaw_counts_the_same(self):
        for yaw in (0.0, 37.0, 90.0, 180.0, 271.0):
            s = state(prop_1=on_motor(FL, yaw=yaw), prop_4=on_motor(FR, yaw=-yaw))
            self.assertEqual(scored(s), 1.0, yaw)

    def test_spares_may_go_anywhere_off_the_motors(self):
        plate = prop(L.CENTER[0], L.CENTER[1], L.TOP + L.PLATE_Z0 + L.PLATE_SIZE[2], yaw=45.0)  # on the frame
        table = prop(0.15, 0.0, L.TOP, yaw=90.0)
        offt = prop(0.30, 0.0, L.TOP - 0.40)  # fallen off the table
        for spare3, spare6 in ((plate, table), (table, offt), (on_mat("prop_6"), on_mat("prop_3"))):
            self.assertEqual(scored(state(prop_3=spare3, prop_6=spare6)), 1.0)

    def test_every_tier_is_reachable(self):
        seen = {scored(only(**dict(list(DONE_PLAN.items())[:k]))) for k in range(5)}
        seen.add(scored(state(prop_3=on_motor(RR), prop_5=on_mat("prop_5"))))
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})

    def test_score_is_idempotent(self):
        tr, r = settled(state())
        again = [tr.result(state()).score for _ in range(3)]
        self.assertEqual(again, [1.0, 1.0, 1.0])
        self.assertEqual(tr.steps, WINDOW + 1)

    def test_random_states_map_to_exactly_one_tier(self):
        rng = random.Random(7)
        motors = list(L.MOTOR_XY)
        for _ in range(300):
            s = dict(START)
            for n in L.PROPS:
                k = rng.random()
                if k < 0.5:
                    s[n] = on_motor(rng.choice(motors), dx=rng.uniform(-0.008, 0.008), dz=rng.uniform(-0.002, 0.005),
                                    yaw=rng.uniform(0, 360), tilt=rng.uniform(0, 12))
                elif k < 0.6:
                    s[n] = upside_down_on(rng.choice(motors), dx=rng.uniform(-0.008, 0.008))
            r = settled(s)[1]
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            wrong = bool(r.details["wrong"] or r.details["upside_down"])
            if wrong:
                self.assertEqual(r.score, 0.0)
            else:
                self.assertAlmostEqual(r.score, 0.25 * r.details["motors_fitted"])


class Edges(unittest.TestCase):
    def test_centring_edge(self):
        self.assertEqual(scored(state(prop_1=on_motor(FL, dx=0.0059))), 1.0)
        self.assertEqual(scored(state(prop_1=on_motor(FL, dx=-0.0042, dy=0.0042))), 1.0)  # 5.9 mm diagonally
        self.assertEqual(scored(state(prop_1=on_motor(FL, dx=0.0061))), 0.75)
        self.assertEqual(scored(state(prop_1=on_motor(FL, dy=-0.0061))), 0.75)

    def test_seat_edge(self):
        self.assertEqual(scored(state(prop_2=on_motor(RL, dz=0.0029))), 1.0)  # held just above: counts
        self.assertEqual(scored(state(prop_2=on_motor(RL, dz=0.0031))), 0.75)
        self.assertEqual(scored(state(prop_2=on_motor(RL, dz=-0.0029))), 1.0)  # contact sink
        self.assertEqual(scored(state(prop_2=on_motor(RL, dz=-0.0031))), 0.75)

    def test_upright_edge(self):
        self.assertEqual(scored(state(prop_5=on_motor(RR, tilt=7.9))), 1.0)
        self.assertEqual(scored(state(prop_5=on_motor(RR, tilt=8.1))), 0.75)

    def test_wrong_hand_beyond_the_edges_is_on_no_motor(self):
        # a wrong-hand propeller counts (and zeroes) only when it stands on a motor by the same three numbers
        self.assertEqual(scored(only(prop_1=FL, prop_3=on_motor(RR, dx=0.0061))), 0.25)
        self.assertEqual(scored(only(prop_1=FL, prop_3=on_motor(RR, dz=0.0031))), 0.25)
        self.assertEqual(scored(only(prop_1=FL, prop_3=on_motor(RR, tilt=8.1))), 0.25)
        self.assertEqual(scored(only(prop_1=FL, prop_3=on_motor(RR, dx=0.0059))), 0.0)

    def test_upside_down_edges(self):
        self.assertEqual(placement(prop(0.38, 0.10, MOTOR_TOP_W + L.HUB_H, roll=172.1))["way"], "down")
        self.assertIsNone(placement(prop(0.38, 0.10, MOTOR_TOP_W + L.HUB_H, roll=171.9))["way"])
        self.assertEqual(scored(state(prop_3=upside_down_on(FL, dz=0.0031), prop_1=on_mat("prop_1"))), 0.75)


class ZeroConditions(unittest.TestCase):
    def test_a_wrong_hand_propeller_on_any_motor_scores_zero(self):
        wrong_for = {FL: "prop_2", FR: "prop_1", RL: "prop_1", RR: "prop_4"}
        for motor, n in wrong_for.items():
            s = state()
            keep = [p for p, m in DONE_PLAN.items() if m == motor][0]
            s[keep] = on_mat(keep)
            s[n] = on_motor(motor)  # (if it came off its own motor, that one is left empty)
            r = settled(s)[1]
            self.assertEqual(r.score, 0.0, motor)
            self.assertEqual(r.details["zero_reason"], "wrong_hand_on_a_motor")

    def test_wrong_hand_with_three_right_ones_still_zero(self):
        s = state(prop_5=on_mat("prop_5"), prop_3=on_motor(RR))
        self.assertEqual(scored(s), 0.0)
        self.assertEqual(detail(s)["wrong"], ["prop_3@rear_right"])

    def test_wrong_hand_zeroes_even_while_moving(self):
        tr = Tracker(HZ)
        feed(tr, [only(prop_1=FL)] * (WINDOW + 1))
        for k in range(5):  # a wrong propeller being set down, still moving within the bands
            r = tr.update(only(prop_1=FL, prop_3=on_motor(RR, dz=0.0025 - 0.0005 * k, yaw=10.0 * k)))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["props"]["prop_3"]["status"], "wrong_hand")

    def test_upside_down_scores_zero_even_of_the_right_hand(self):
        r = settled(state(prop_1=upside_down_on(FL)))[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "upside_down_on_a_motor")
        self.assertEqual(r.details["upside_down"], ["prop_1@front_left"])

    def test_the_one_arm_trap_spare_on_the_far_rear_motor(self):
        # the left arm alone: its CW and CCW on its own motors, then its spare (CCW) on the rear right (CW) motor
        self.assertEqual(scored(only(prop_1=FL, prop_2=RL)), 0.5)
        self.assertEqual(scored(only(prop_1=FL, prop_2=RL, prop_3=RR)), 0.0)
        self.assertEqual(scored(only(prop_5=RR, prop_4=FR, prop_6=RL)), 0.0)  # the right arm's mirror image


class Exploits(unittest.TestCase):
    def test_flipping_a_propeller_does_not_change_its_hand(self):
        """Upside down, a CCW propeller seen from below looks like a CW one; on a CW motor it pushes air the wrong
        way: 0, not a fitted motor."""
        s = state(prop_1=on_mat("prop_1"), prop_3=upside_down_on(FL))
        self.assertEqual(scored(s), 0.0)

    def test_held_just_above_the_motor_is_not_fitted(self):
        self.assertEqual(scored(state(prop_4=on_motor(FR, dz=0.010))), 0.75)
        self.assertEqual(scored(state(prop_4=on_motor(FR, dz=0.050))), 0.75)

    def test_leaning_on_the_motor_or_lying_on_the_frame(self):
        x, y = L.MOTOR_XY[RL]
        lean = prop(x + 0.012, y, MOTOR_TOP_W - 0.004, pitch=25.0)
        on_arm = prop(x - 0.03, y - 0.02, L.TOP + L.ARM_H, yaw=45.0)
        for p in (lean, on_arm):
            self.assertEqual(scored(state(prop_2=p)), 0.75)

    def test_a_wrong_propeller_stacked_on_a_right_one(self):
        s = state(prop_3=prop(*L.MOTOR_XY[RR], MOTOR_TOP_W + L.HUB_H))  # resting on the fitted hub's top
        r = settled(s)[1]
        self.assertEqual(r.score, 1.0)  # a propeller on another propeller stands on no motor, and adds nothing
        self.assertEqual(r.details["props"]["prop_3"]["status"], "not_on_a_motor")

    def test_two_on_one_motor_count_once(self):
        s = only(prop_1=FL, prop_6=on_motor(FL, dx=0.002))
        self.assertEqual(scored(s), 0.25)

    def test_spinning_in_place_is_not_at_rest(self):
        tr = Tracker(HZ)
        r = feed(tr, [state(prop_1=on_motor(FL, yaw=4.0 * k)) for k in range(WINDOW + 5)])  # tips move 4.4 mm a step
        self.assertEqual(r.score, 0.75)
        self.assertEqual(r.details["props"]["prop_1"]["status"], "still_moving")


class Stillness(unittest.TestCase):
    def test_the_window_is_half_a_second(self):
        self.assertEqual(Tracker(20.0).window, 10)
        tr = Tracker(HZ)
        # 3 mm of settling 0.45 s (9 steps) ago is still inside the window; 0.55 s ago it is not
        seq = [state(prop_2=on_motor(RL, dz=0.0029, dx=0.003))] + [state()] * 9
        self.assertEqual(feed(tr, [state()] * WINDOW + seq).score, 0.75)
        self.assertEqual(feed(tr, [state()] * 2).score, 1.0)

    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        for _ in range(WINDOW):
            self.assertEqual(tr.update(state()).score, 0.0)
        self.assertEqual(tr.update(state()).score, 1.0)

    def test_jitter_within_2_mm_is_rest(self):
        rng = random.Random(3)
        tr = Tracker(HZ)
        r = feed(tr, [state(prop_2=on_motor(RL, dx=rng.uniform(-0.0009, 0.0009), dz=rng.uniform(0, 0.0009)))
                      for _ in range(WINDOW + 5)])
        self.assertEqual(r.score, 1.0)

    def test_a_moving_propeller_is_not_fitted(self):
        tr = Tracker(HZ)
        seq = [state(prop_5=on_motor(RR, dz=0.0025 - 0.0005 * k)) for k in range(6)]  # settling 2.5 mm in 0.25 s
        r = feed(tr, [state(prop_5=on_motor(RR, dz=0.0025))] * WINDOW + seq)
        self.assertEqual(r.score, 0.75)  # 2.5 mm from where it is now, within the last 0.5 s
        r = feed(tr, [state(prop_5=on_motor(RR))] * WINDOW)
        self.assertEqual(r.score, 1.0)  # at rest once it has been still for 0.5 s
        tr2 = Tracker(HZ)
        far = [state(prop_5=on_motor(RR, dz=0.0029, dx=0.0005 * k)) for k in range(WINDOW + 1)]  # 5 mm creep
        r2 = feed(tr2, far)
        self.assertEqual(r2.score, 0.75)
        self.assertEqual(r2.details["props"]["prop_5"]["status"], "still_moving")


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        n = tr.steps
        for bad in (prop(np.nan, 0.1, MOTOR_TOP_W), prop(0.38, np.inf, MOTOR_TOP_W)):
            r = tr.update(state(prop_1=bad))
            self.assertEqual(r.score, 0.0)
            self.assertFalse(r.success)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
        nanq = state()
        nanq["prop_2"] = SimpleNamespace(origin=nanq["prop_2"].origin, quat_xyzw=np.array([np.nan, 0, 0, 1.0]))
        self.assertEqual(tr.update(nanq).score, 0.0)
        self.assertEqual(tr.steps, n + 3)
        self.assertEqual(tr.update(state()).score, 1.0)  # the history was kept

    def test_zero_quaternion_missing_origin_or_object(self):
        zq = state()
        zq["prop_4"] = SimpleNamespace(origin=zq["prop_4"].origin, quat_xyzw=np.zeros(4))
        self.assertFalse(valid(zq["prop_4"]))
        box = SimpleNamespace(pos=np.array([0.38, -0.1, 0.80]), quat_xyzw=quat())  # a plain box, no origin
        self.assertFalse(valid(box))
        missing = state()
        missing.pop("prop_6")
        for s in (zq, state(prop_4=box), missing):
            self.assertEqual(Tracker(HZ).update(s).score, 0.0)

    def test_unnormalised_quaternion(self):
        s = state()
        s["prop_1"] = SimpleNamespace(origin=s["prop_1"].origin, quat_xyzw=3.0 * quat(20.0))
        self.assertEqual(scored(s), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = random.Random(11)
        tr = Tracker(HZ)
        for _ in range(200):
            s = {n: prop(rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(0, 2), rng.uniform(-180, 180),
                         rng.uniform(-180, 180), rng.uniform(-180, 180)) for n in L.PROPS}
            r = tr.update(s)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, r = settled(state())
        self.assertEqual(r.score, 1.0)
        r = tr.update(START)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)

    def test_reset_then_a_new_trial(self):
        tr, _ = settled(state())
        tr.update(START)
        r = feed(tr, [only(prop_1=FL)] * (WINDOW + 1))
        self.assertEqual(r.score, 0.25)
        self.assertEqual(tr.steps, WINDOW + 2)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(only(prop_1=FL))
        n = tr.steps
        a, b = np.array(L.MOTOR_XY[FL]), np.array(L.PROP_START["prop_1"])
        for k in range(1, 11):  # carried back down onto its mark, 2.6 cm a step
            p = a + (b - a) * k / 10
            z = MAT_TOP_W + 0.05 * (1 - k / 10)
            tr.update(only(prop_1=prop(p[0], p[1], z)))
        self.assertEqual(tr.steps, n + 10)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(state())
        n = tr.steps
        tr.update(state(prop_1=prop(0.10, 0.0, L.TOP)))
        self.assertEqual(tr.steps, n + 1)

    def test_explicit_reset(self):
        tr, _ = settled(state())
        tr.reset()
        self.assertEqual(tr.steps, 0)
        self.assertEqual(tr.update(state()).score, 0.0)  # no history yet


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stand-in (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, mechkit's reset counter, the
    demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        sys.path.insert(0, str(_stub_dir().parent))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit
        cls.specs = {o.name: o for o in cls.task.PROPS}

    def sim(self, st):
        """Synthetic states turned into what mechkit would report from the simulator."""
        return {n: self.mk.state_from_poses(self.specs[n], (*st[n].origin, *st[n].quat_xyzw), source="sim")
                for n in L.PROPS}

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        loose = [o for o in t.OBJECTS if o.mass is not None]
        self.assertEqual(sorted(o.name for o in loose), sorted(L.PROPS))
        for o in loose:
            self.assertLessEqual(o.mass, 0.30)
        self.assertEqual(t.SCORE_OVER, "end")
        self.assertEqual(t.SUCCESS_HOLD_S, 2.5)
        by = {o.name: o for o in t.OBJECTS}
        for m, (x, y) in L.MOTOR_XY.items():
            c = by[f"motor_{m}"]
            self.assertAlmostEqual(c.pos[2] + c.height / 2, MOTOR_TOP_W)
            self.assertAlmostEqual(c.pos[2] - c.height / 2, L.TOP + L.ARM_H)  # standing on its arm
            self.assertEqual((c.pos[0], c.pos[1]), (x, y))
            arm = by[f"arm_{m}"]
            self.assertAlmostEqual(arm.pos[2] - arm.size[2] / 2, L.TOP)
            for part, *_ in L.arrow_parts(m):
                self.assertEqual(by[f"arrow_{m}_{part}"].color, L.ARROW_RED)
        self.assertAlmostEqual(by["mat_left"].pos[2] + by["mat_left"].size[2] / 2, MAT_TOP_W)
        # nothing in the arms' start-pose shadow of the setup screenshot (|y| > 0.45 m with x < 0.2 m)
        for o in t.OBJECTS:
            if hasattr(o, "offline_box"):
                c, size, yaw = o.offline_box()
            elif hasattr(o, "size"):
                c, size, yaw = o.pos, o.size, o.yaw
            else:
                c, size, yaw = o.pos, (2 * o.radius, 2 * o.radius, o.height), 0.0
            half = abs(math.cos(yaw)) * size[1] / 2 + abs(math.sin(yaw)) * size[0] / 2
            self.assertLess(abs(c[1]) + half, 0.45, o.name)

    def test_scorer_geometry_matches_mechkit(self):
        for n, spec in self.specs.items():
            lo, hi = spec.local_bounds()
            np.testing.assert_allclose(lo, [-L.BLADE_OUT, -L.HUB_R, 0.0], atol=1e-12)
            np.testing.assert_allclose(hi, [L.BLADE_OUT, L.HUB_R, L.HUB_H], atol=1e-12)
            com = spec.center_of_mass()
            np.testing.assert_allclose(com[:2], [0.0, 0.0], atol=1e-12)  # on the hub's axis: it sits level
            self.assertLess(com[2], 0.012)
            strips = [s for s in spec.shapes if s.color is not None and tuple(s.color) == L.EDGE_WHITE]
            self.assertEqual(len(strips), 2)
            for s, want in zip(strips, L.edge_centers(L.HAND[n])):
                np.testing.assert_allclose(s.center, want, atol=1e-12)
            # the strip on the +x blade is on its -y side for a CW propeller
            plus = [s for s in strips if s.center[0] > 0][0]
            self.assertEqual(plus.center[1] < 0, L.HAND[n] == "CW")
            rest = spec.rest_state()
            np.testing.assert_allclose(rest.origin, [*L.PROP_START[n], MAT_TOP_W], atol=1e-12)
            self.assertEqual(spec.mass, L.PROP_MASS)
        rest = {n: s.rest_state() for n, s in self.specs.items()}
        self.assertTrue(Tracker.at_start(rest))
        sim = self.sim(state())
        np.testing.assert_allclose(sim["prop_1"].low[2], MOTOR_TOP_W, atol=1e-9)
        self.assertEqual(scored(sim), 1.0)
        self.assertEqual(scored(self.sim(upside_down_on_state())), 0.0)
        np.testing.assert_allclose(self.sim(upside_down_on_state())["prop_1"].low[2], MOTOR_TOP_W, atol=1e-9)

    def test_offline_state_path(self):
        spec = self.specs["prop_2"]
        box = SimpleNamespace(pos=np.array(spec.offline_box()[0]), quat_xyzw=quat(), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3))
        st = spec.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [*L.PROP_START["prop_2"], MAT_TOP_W], atol=1e-12)
        self.assertTrue(valid(st))

    def test_hooks_and_mechkit_reset(self):
        tr = self.task._TRACKER
        self.mk.offline_reset()
        start = self.sim(START)
        self.assertFalse(self.task.success(start))
        self.assertEqual(self.task.score(start), 0.0)
        done = self.sim(state())
        for _ in range(WINDOW):
            self.task.success(done)
        self.assertTrue(self.task.success(done))
        self.assertEqual(self.task.score(done), 1.0)
        self.assertEqual(self.task.score(done), 1.0)  # score() does not advance it
        n = tr.steps
        self.mk.offline_reset()  # mechkit's episode counter: a new trial
        self.assertFalse(self.task.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(tr.steps, n)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.mk.offline_reset()
                self.task.success(self.sim(state(prop_3=on_motor(RR, dx=0.003))))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("episode", "step", "score", "success", "zero_reason", "motors_fitted", "wrong",
                          "upside_down", "fitted", "props", "origins", "quats", "source"):
                    self.assertIn(k, row)
                p3 = row["props"]["prop_3"]
                for k in ("hand", "motor", "status", "nearest", "centre_off_mm", "seat_mm", "tilt_deg", "at_rest",
                          "moved_mm"):
                    self.assertIn(k, p3)
                self.assertEqual(p3["status"], "wrong_hand")
                self.assertAlmostEqual(p3["centre_off_mm"], 3.0)
                self.assertEqual(row["source"], "sim")
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(state())))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    @staticmethod
    def _timeline(wps):
        t, out = 0.0, []
        for w in wps:
            out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        pick_z = L.TOP + L.MAT_TOP + t.GRIP_UP
        place_z = L.TOP + L.MOTOR_TOP + t.DROP + t.GRIP_UP
        for arm, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger and t.GRIP in (prev.finger, w.finger):  # every grip and release: 1.5 s,
                    # standing still (opening wider between two propellers happens on the way)
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                    self.assertEqual(prev.pos, w.pos, (arm, w.label))
                if prev.pos[2] > w.pos[2] and min(abs(w.pos[2] - z) for z in (pick_z, place_z)) < 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))  # slow last 30 mm
                    self.assertLessEqual(prev.pos[2] - w.pos[2], t.APPROACH + 1e-9)
            for w in wps:
                self.assertEqual(w.yaw, 0.0)
        # each arm fits its own side with its own mat's propellers, each of the motor's hand; the spares stay put
        moved = []
        for arm, pairs in t.PLAN.items():
            sign = 1.0 if arm == "left" else -1.0
            for n, m in pairs:
                self.assertEqual(L.HAND[n], L.MOTOR_DIR[m])
                self.assertEqual(math.copysign(1.0, L.PROP_START[n][1]), sign)
                self.assertEqual(math.copysign(1.0, L.MOTOR_XY[m][1]), sign)
                moved.append(n)
        self.assertEqual(sorted(moved), sorted(DONE_PLAN))
        self.assertEqual(sorted(set(L.MOTOR_XY)), sorted(m for pairs in t.PLAN.values() for _, m in pairs))
        # pad tips (13 mm below the TCP) 6 mm above the strips' tops at every grip; the fingers close across the hub
        self.assertAlmostEqual(t.GRIP_UP - 0.013 - (L.BLADE_H + L.EDGE_H), 0.006)
        self.assertLessEqual(t.GRIP_UP + 0.009, L.HUB_H + 0.001)
        self.assertAlmostEqual(-2 * t.GRIP, 2 * L.HUB_R - 0.008)
        # the open fingers (8 mm thick) clear the next hub on the mat
        self.assertGreater(0.055 - L.HUB_R - (-t.PICK_OPEN + 0.008), 0.010)
        # each propeller is let go with its hub's bottom 2 mm above the motor's top: inside the 3 mm seat band, so it
        # counts once still while held, and the 2.5 s success hold outlasts the 0.5 s settle and the 1.5 s release
        self.assertAlmostEqual(t.DROP, 0.002)
        self.assertLess(t.DROP, L.SEAT_TOL)
        # every grip and every place at its height
        for arm, wps in plan.items():
            for w in wps:
                if w.label.startswith("grip "):
                    self.assertAlmostEqual(w.pos[2], pick_z)
                if w.label.startswith("slowly set"):
                    self.assertAlmostEqual(w.pos[2], place_z)
            # carried 22 mm over a fitted propeller's hub top
            carry = [w for w in wps if w.label.startswith("carry")]
            for w in carry:
                self.assertAlmostEqual(w.pos[2] - t.GRIP_UP - (MOTOR_TOP_W + L.HUB_H), 0.022)
        # the two arms on one clock: whenever both are low over the frame, their TCPs are >= 0.25 m apart
        tl = {arm: self._timeline(wps) for arm, wps in plan.items()}
        self.assertEqual([round(b, 6) for _, b, _ in tl["left"]], [round(b, 6) for _, b, _ in tl["right"]])
        for (_, _, wl), (_, _, wr) in zip(tl["left"], tl["right"]):
            self.assertGreaterEqual(math.dist(wl.pos[:2], wr.pos[:2]), 0.25)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_one_arm_outcomes_of_the_demo(self):
        left = only(**{n: m for n, m in self.task.PLAN["left"]})
        right = only(**{n: m for n, m in self.task.PLAN["right"]})
        self.assertEqual(scored(left), 0.5)
        self.assertEqual(scored(right), 0.5)

    def test_instruction_states_the_rule(self):
        ins = self.task.INSTRUCTION
        for s in ("each of the drone's four motors", "upright", "centred", "hub up", "red arrow", "which way that motor turns",
                  "leading edges", "raised white strips", "lead in that direction", "spares"):
            self.assertIn(s, ins)
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], ins)
        self.assertIn(f"--max-steps {int(meta['run_limit'].split()[0].replace(',', ''))}",
                      (HERE / "README.md").read_text())
        self.assertIn(ins, (HERE / "README.md").read_text())


def upside_down_on_state():
    return state(prop_1=upside_down_on(FL))


if __name__ == "__main__":
    unittest.main()
