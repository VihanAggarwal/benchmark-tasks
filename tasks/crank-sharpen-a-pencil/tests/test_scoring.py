"""Offline tests for scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's own specs)
and synthetic arm snapshots (rigstate.Arm), no simulator.  python3 -m unittest discover -s tests -v"""

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
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline package

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

TWO_PI = 2 * math.pi
HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 20 steps
BORE_Z = L.TOP + L.AXIS_Z
AWAY = {a: rigstate.Arm(np.array([0.30, s * 0.30, L.TOP + 0.25]), np.array([0.30, s * 0.30, L.TOP + 0.237]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}


# -- states --------------------------------------------------------------------------------------------------------


def sharpener(slide=0.0, front=0.0, crank=0.0, knob=0.0, w=0.0):
    origin, quat = T.SHARPENER.spawn_pose()
    return mk.state_from_poses(T.SHARPENER, (*origin, *quat), source="offline",
                               joints={"slide": slide, "front": front, "crank": crank, "knob": knob},
                               joint_vel={"crank": w})


def pencil_at(centre, r=np.eye(3)):
    return mk.state_from_poses(T.PENCIL, (*centre, *mk.matrix_to_quat(r)), source="offline")


def pencil_in_bore(end_y, dx=0.0, dz=0.0):
    """The pencil on the bore line with its front end at world ``end_y`` (offset dx, dz from the axis)."""
    return pencil_at((L.SH_X + dx, end_y + L.HALF, BORE_Z + dz))


def pencil_on_rest(dx=0.0, dz=0.0, dy=0.0):
    return pencil_at((L.REST_XY[0] + dx, L.REST_XY[1] + dy, L.TOP + L.SEAT_Z + dz))


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def pressed_end(depth, feeler=0.004):
    """The front end's y with the feeler pushed in ``feeler`` at cut depth ``depth``."""
    return L.FRONT_Y0 - depth - feeler


def step_cut(cutter, th, end_y=None, feeler=0.004, w=3.0, dx=0.0, dz=0.0, pencil=None):
    """One control step of the cut rule with the crank at ``th`` and the pencil pressed (default) at the cut."""
    p = pencil if pencil is not None else pencil_in_bore(pressed_end(cutter.depth, feeler) if end_y is None
                                                         else end_y, dx, dz)
    return cutter.update(sharpener(slide=cutter.depth, front=feeler, crank=S.wrap_to_pi(th), w=w), p)


def turn(cutter, turns, start=0.0, steps_per_turn=36, **kw):
    """Turn the crank ``turns`` (negative: backward) from unwrapped angle ``start`` in small steps; returns the end
    angle."""
    n = max(1, int(round(abs(turns) * steps_per_turn)))
    th = start
    for k in range(1, n + 1):
        th = start + TWO_PI * turns * k / n
        step_cut(cutter, th, **kw)
    return th


def arms_holding(tcp, opening=0.11):
    tcp = np.asarray(tcp, dtype=float)
    a = rigstate.Arm(tcp, tcp + np.array([0.0, 0.0, -0.013]), opening)
    return {"left": a, "right": AWAY["right"]}


def run_tracker(tracker, cutter, pencil, arms, steps, sharp=None):
    states = {"pencil": pencil, "sharpener": sharp if sharp is not None else sharpener(slide=cutter.depth),
              "rest": T.REST.rest_state()}
    res = None
    for _ in range(steps):
        res = tracker.update(states, arms, cutter)
    return res


def full_cut():
    c = S.Cutter()
    step_cut(c, 0.0)
    turn(c, 16.5)
    return c


# -- geometry ------------------------------------------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_bore_fits_the_pencil_with_one_mm_a_side(self):
        self.assertAlmostEqual(L.BORE_APOTHEM - L.PENCIL_R, 0.001, places=6)
        self.assertAlmostEqual(L.RING_SPAN, 0.025, places=6)
        # tilt the rings allow: (12.3 - 10.3) / 25 rad, about 4.6 degrees
        self.assertAlmostEqual(math.degrees(2 * (L.BORE_APOTHEM - L.PENCIL_R) / L.RING_SPAN), 4.58, places=1)

    def test_ring_inner_faces_are_an_octagon_at_the_apothem(self):
        rings = [s for s in T.SHARPENER.shapes if s.kind == "box" and abs(s.center[1] - 0.036) < 1e-6]
        self.assertEqual(len(rings), 8)
        for s in rings:
            r = np.asarray(mk.quat_to_matrix(s.quat) if s.quat else np.eye(3))
            radial = r[:, 2]
            c = np.array(s.center) - np.array([L.AXIS_X, s.center[1], L.AXIS_Z])
            inner = float(c @ radial) - s.size[2] / 2
            self.assertAlmostEqual(inner, L.BORE_APOTHEM, places=6)

    def test_lead_in_faces_run_from_the_face_to_the_ring(self):
        lead = [s for s in T.SHARPENER.shapes if s.kind == "box" and s.size[2] == L.LEAD_T]
        self.assertEqual(len(lead), 8)
        for s in lead:
            r = mk.quat_to_matrix(s.quat)
            c = np.array(s.center)
            for sy in (-1.0, 1.0):  # the inner face's two ends (along the slant)
                p = c + r[:, 1] * sy * s.size[1] / 2 - r[:, 2] * s.size[2] / 2
                rad = math.hypot(p[0] - L.AXIS_X, p[2] - L.AXIS_Z)
                a = math.atan2(p[0] - L.AXIS_X, p[2] - L.AXIS_Z)
                apo = rad * math.cos(a - round(a / (math.pi / 4)) * math.pi / 4)
                if p[1] > 0.039:
                    self.assertAlmostEqual(p[1], L.DIAL_LEAD_Y[1], places=5)
                    self.assertAlmostEqual(apo, L.BORE_APOTHEM + 0.003, places=4)
                else:
                    self.assertAlmostEqual(p[1], L.DIAL_LEAD_Y[0], places=5)
                    self.assertAlmostEqual(apo, L.BORE_APOTHEM, places=4)

    def test_sharpener_body_is_the_ks_envelope(self):
        self.assertAlmostEqual(L.BODY_Y[1] - L.BODY_Y[0], 0.121, places=6)
        self.assertAlmostEqual(L.BODY_X[1] - L.BODY_X[0], 0.070, places=6)
        self.assertAlmostEqual(L.BODY_Z[1] - L.BODY_Z[0], 0.108, places=6)
        self.assertAlmostEqual(L.AXIS_Z - L.BODY_Z[0], 0.080, places=6)  # bore 80 mm above the sharpener's base
        self.assertTrue(T.SHARPENER.fixed_base)

    def test_masses_agree_with_sizes(self):
        vol = math.pi * L.PENCIL_R ** 2 * L.PENCIL_LEN * 1e6  # cm3
        self.assertAlmostEqual(vol, 13.8, delta=0.1)
        self.assertAlmostEqual(L.PENCIL_MASS, 0.009, places=4)
        arm = L.CRANK_ARM[1]
        steel = 7.85 * (arm[0] * arm[1] * arm[2] + math.pi * L.HUB_R ** 2 * L.HUB_LEN) * 1e6 / 1000
        self.assertAlmostEqual(steel, L.CRANK_MASS, delta=0.001)
        knob = 0.9 * math.pi * L.KNOB_R ** 2 * L.KNOB_LEN * 1e6 / 1000
        self.assertAlmostEqual(knob, L.KNOB_MASS, delta=0.0005)
        block = L.BLOCK[1][0] * L.BLOCK[1][1] * L.BLOCK[1][2] * 1e6 * 0.5 / 1000
        self.assertAlmostEqual(block, L.BLOCK_MASS_NOTE, delta=0.001)

    def test_crank_drive_is_near_critical(self):
        arm_c = L.CRANK_ARM[0][2] - L.AXIS_Z
        sz = L.CRANK_ARM[1]
        inertia = (L.CRANK_MASS - 0.0158) * (arm_c ** 2 + (sz[2] ** 2 + sz[0] ** 2) / 12) \
            + L.KNOB_MASS * L.CRANK_R ** 2 + 0.5 * 0.0158 * L.HUB_R ** 2
        self.assertAlmostEqual(inertia, L.CRANK_INERTIA, delta=0.3e-5)
        critical = 2 * math.sqrt(L.CRANK_K * inertia)
        self.assertGreater(L.CRANK_DAMP / critical, 0.8)
        self.assertLess(L.CRANK_DAMP / critical, 1.3)
        # a crank let go mid-cut springs back at most drag / K before the next step's target follows it
        self.assertLessEqual(L.T_CUT / L.CRANK_K, 0.076)

    def test_drag_numbers(self):
        knob_force = (L.T_CUT + L.BEARING_FRICTION) / L.CRANK_R
        self.assertAlmostEqual(knob_force, 4.44, places=2)
        removed = 2 / 3 * math.pi * (L.PENCIL_R * 1000) ** 2 * L.CUT_DEPTH * 1000 / 1000  # cm3
        self.assertAlmostEqual(removed, 1.33, places=2)
        angle = L.TURNS_TO_POINT * TWO_PI
        lo, hi = removed * 10 / angle, removed * 20 / angle  # 10 to 20 J/cm3
        self.assertTrue(lo <= L.T_CUT <= hi, (lo, hi))
        self.assertAlmostEqual(L.TURNS_TO_POINT, 16.0)

    def test_rest_seat_height(self):
        self.assertAlmostEqual(L.SEAT_Z, 0.0303, places=4)
        # the pencil in a 90 deg notch touches both faces 7 mm deep: its underside stays above the apex
        self.assertGreater(L.SEAT_Z - L.PENCIL_R, L.V_APEX_Z)
        self.assertLess(L.SEAT_Z - L.PENCIL_R * math.sqrt(0.5), L.V_TOP_Z)  # the contact lines are inside the notch

    def test_v_notch_faces(self):
        """Each cheek's face runs from the apex (x 0, z 23 mm) to the top edge (x +-7, z 30 mm)."""
        cheeks = [s for s in T.REST.shapes if s.quat is not None]
        self.assertEqual(len(cheeks), 4)
        for s in cheeks:
            r = mk.quat_to_matrix(s.quat)
            c = np.array(s.center)
            ends = sorted((c + r[:, 0] * sx * s.size[0] / 2 + r[:, 2] * s.size[2] / 2 for sx in (-1, 1)),
                          key=lambda p: p[2])
            self.assertAlmostEqual(ends[0][0], 0.0, places=6)
            self.assertAlmostEqual(ends[0][2], L.V_APEX_Z, places=6)
            self.assertAlmostEqual(abs(ends[1][0]), L.V_HALF_W, places=6)
            self.assertAlmostEqual(ends[1][2], L.V_TOP_Z, places=6)

    def test_offline_boxes(self):
        c, size, _ = T.REST.offline_box()
        self.assertAlmostEqual(c[2] + size[2] / 2, L.TOP + L.SEAT_Z - L.PENCIL_R, places=6)
        c, size, _ = T.SHARPENER.offline_box()
        self.assertAlmostEqual(c[1] + size[1] / 2, L.BODY_Y[1], places=6)
        self.assertAlmostEqual(c[0], L.SH_X, places=6)

    def test_crank_turns_forward_away_from_the_robot(self):
        s = sharpener(crank=math.pi / 2)
        knob = s.link_to_world("knob", (L.AXIS_X, L.KNOB_CENTRE_Y, L.KNOB_PIN[2]))
        self.assertAlmostEqual(knob[0], L.SH_X + L.CRANK_R, places=6)
        self.assertAlmostEqual(knob[2], BORE_Z, places=6)

    def test_knob_and_pads_clear_the_crank_arm(self):
        pad_inner_y = T.KNOB_GRIP_Y + 0.0135  # pads about 27 mm wide round the TCP at its height
        arm_outer_y = L.CRANK_ARM[0][1] - L.CRANK_ARM[1][1] / 2
        self.assertGreaterEqual(arm_outer_y - pad_inner_y, 0.020)
        self.assertGreaterEqual(T.KNOB_GRIP_Y - 0.0135, L.KNOB_PIN[1] - L.KNOB_LEN)  # pads on the knob
        lowest = L.AXIS_Z - L.CRANK_R - L.KNOB_R
        self.assertGreater(lowest, L.BLOCK[1][2] + 0.02)  # the knob clears the offcut by 25 mm

    def test_slide_and_feeler(self):
        self.assertAlmostEqual(L.PRESS * L.FEELER_K, 2.0)
        far = L.SLIDE_ROD[0][1] - L.SLIDE_ROD[1][1] / 2 - L.SLIDE_TRAVEL
        self.assertGreater(far, L.TUNNEL_Y0 + 0.02)  # the rod's far end stays 21 mm clear of the crank end wall
        self.assertLess(L.FRONT_Y0 - L.CUT_DEPTH - L.FEELER_TRAVEL, L.PAST_RINGS_Y)
        self.assertGreater(L.SLIDE_DAMP, 2 * math.sqrt(L.SLIDE_K * L.SLIDE_MASS))
        self.assertGreater(L.FEELER_DAMP, 2 * math.sqrt(L.FEELER_K * L.FEELER_MASS))

    def test_kit_limits(self):
        for j in T.SHARPENER.joints:
            if j.kind == "revolute":
                self.assertLessEqual(max(-j.lower, j.upper), TWO_PI)
        self.assertLess(L.WRAP_AT + 0.2 + L.T_CUT / L.CRANK_K, L.CRANK_LIMIT)

    def test_kit_copies_are_unchanged(self):
        for name in ("mechkit.py", "rigstate.py"):
            kit = HERE.parents[1] / "tools" / "kit" / name
            if kit.exists():
                self.assertEqual((HERE / name).read_bytes(), kit.read_bytes(), name)


# -- the cut -------------------------------------------------------------------------------------------------------


class Cut(unittest.TestCase):
    def test_nothing_done(self):
        c = S.Cutter()
        for _ in range(50):
            c.update(sharpener(), pencil_on_rest())
        self.assertEqual(c.depth, 0.0)

    def test_pressed_without_cranking(self):
        c = S.Cutter()
        for _ in range(200):
            step_cut(c, 0.0, w=0.0)
        self.assertEqual(c.depth, 0.0)
        self.assertTrue(c.pressed)

    def test_crank_without_pencil(self):
        c = S.Cutter()
        th = 0.0
        for k in range(36 * 3):
            th = TWO_PI * k / 36
            c.update(sharpener(crank=S.wrap_to_pi(th), w=3.0), pencil_on_rest())
        self.assertEqual(c.depth, 0.0)
        self.assertEqual(c.drag, 0.0)

    def test_one_pressed_forward_turn(self):
        c = S.Cutter()
        step_cut(c, 0.0)
        turn(c, 1.0)
        self.assertAlmostEqual(c.depth, L.CUT_PER_TURN, places=6)

    def test_press_thresholds(self):
        for kw in ({"feeler": 0.0019}, {"end_y": None, "feeler": 0.004, "dx": 0.0016},
                   {"end_y": None, "feeler": 0.004, "dz": -0.0016}):
            c = S.Cutter()
            step_cut(c, 0.0, **kw)
            turn(c, 1.0, **kw)
            self.assertEqual(c.depth, 0.0, kw)
        c = S.Cutter()  # the end 1.1 mm off the disc's face (feeler pushed in by something else)
        step_cut(c, 0.0, end_y=pressed_end(0.0) + 0.0011)
        turn(c, 1.0, end_y=pressed_end(0.0) + 0.0011)
        self.assertEqual(c.depth, 0.0)
        c = S.Cutter()  # just inside every threshold
        kw = {"feeler": 0.002, "dx": 0.0014}
        step_cut(c, 0.0, **kw)
        turn(c, 1.0, **kw)
        self.assertAlmostEqual(c.depth, L.CUT_PER_TURN, places=6)

    def test_end_must_be_past_both_rings(self):
        c = S.Cutter()
        # an end at the carriage ring's outer face, 2 mm off the disc: not through the rings
        sharp = sharpener(slide=0.0, front=0.004)
        p = pencil_in_bore(L.PAST_RINGS_Y + 0.0001)
        for k in range(40):
            c.update(sharpener(front=0.004, crank=S.wrap_to_pi(TWO_PI * k / 36), w=3.0), p)
        self.assertEqual(c.depth, 0.0)
        self.assertIsNotNone(sharp)

    def test_backward_turns_add_nothing(self):
        c = S.Cutter()
        step_cut(c, 0.0)
        turn(c, -3.0, w=-3.0)
        self.assertEqual(c.depth, 0.0)
        self.assertEqual(c.drag, 0.0)

    def test_rocking_counts_net_forward_travel_only(self):
        c = S.Cutter()
        step_cut(c, 0.0)
        th = 0.0
        for target in (1.0, 0.0, 1.0):
            n = 20
            start = th
            for k in range(1, n + 1):
                th = start + (target - start) * k / n
                step_cut(c, th)
        self.assertAlmostEqual(c.depth, L.CUT_PER_TURN * 1.0 / TWO_PI, places=7)

    def test_wrap_counts_the_short_way(self):
        c = S.Cutter()
        step_cut(c, math.pi - 0.05)
        step_cut(c, -math.pi + 0.05)  # +0.1 rad across the wrap
        self.assertAlmostEqual(c.th_un, 0.1, places=6)

    def test_backlog_is_forgiven_after_one_turn(self):
        c = S.Cutter()
        step_cut(c, 0.0)
        end = turn(c, -3.0, w=-3.0)
        end = turn(c, 16.0, start=end)
        self.assertAlmostEqual(c.depth, 0.0225, places=6)  # 15 of the 16 turns cut: one turn of backlog
        turn(c, 1.0, start=end)
        self.assertAlmostEqual(c.depth, L.CUT_DEPTH, places=6)

    def test_cut_stops_at_the_point(self):
        c = full_cut()
        self.assertAlmostEqual(c.depth, L.CUT_DEPTH, places=9)
        turn(c, 3.0, start=c.th_un)
        self.assertAlmostEqual(c.depth, L.CUT_DEPTH, places=9)
        self.assertEqual(c.drag, 0.0)

    def test_depth_never_shrinks(self):
        c = S.Cutter()
        step_cut(c, 0.0)
        end = turn(c, 5.0)
        d = c.depth
        for k in range(30):
            c.update(sharpener(slide=d, crank=S.wrap_to_pi(end + k * 0.1), w=2.0), pencil_on_rest())
        self.assertEqual(c.depth, d)

    def test_broken_states_change_nothing(self):
        c = S.Cutter()
        step_cut(c, 0.0)
        turn(c, 2.0)
        d = c.depth
        nan_pencil = pencil_at((float("nan"), 0.1, BORE_Z))
        for bad in ((None, pencil_in_bore(0.0)), (sharpener(), None), (sharpener(), nan_pencil),
                    (sharpener(crank=float("nan")), pencil_in_bore(0.0)), (object(), object())):
            info = c.update(*bad)
            self.assertFalse(info["valid"])
            self.assertFalse(c.pressed)
            self.assertEqual(c.drag, 0.0)
        self.assertEqual(c.depth, d)

    def test_either_end_can_go_in(self):
        c = S.Cutter()
        flipped = np.diag([-1.0, -1.0, 1.0])  # the pencil turned end for end
        p = pencil_at((L.SH_X, pressed_end(0.0) + L.HALF, BORE_Z), flipped)
        c.update(sharpener(front=0.004), p)
        self.assertTrue(c.pressed)

    def test_tilted_pencil_jammed_in_the_bore(self):
        """A pencil tilted 5 deg in the bore with its end 1.6 mm off the axis: nothing is cut."""
        c = S.Cutter()
        r = _rx(math.radians(5.0))
        end = np.array([L.SH_X, pressed_end(0.0), BORE_Z + 0.0016])
        p = pencil_at(end + r @ np.array([0.0, L.HALF, 0.0]), r)
        e, _ = S.pencil_ends(p)
        self.assertAlmostEqual(float(e[2]), BORE_Z + 0.0016, places=6)
        for k in range(40):
            c.update(sharpener(front=0.004, crank=S.wrap_to_pi(TWO_PI * k / 36), w=3.0), p)
        self.assertEqual(c.depth, 0.0)


def stiff_hold(push_n, turns=17, lead=None, steps_per_turn=36):
    """Quasi-static model of a pencil held STILL by a stiff grip (it does not move along the bore) while the crank
    turns: the feeler (1,000 N/m, 5 mm travel) and the slide (50,000 N/m round the cut depth) in series. ``push_n``
    is the force at the start; returns the cut depth after ``turns``."""
    c = S.Cutter()
    if lead is None:
        if push_n <= L.FEELER_K * L.FEELER_TRAVEL:
            x = push_n / L.FEELER_K * (1 + L.FEELER_K / L.SLIDE_K)
        else:
            x = L.FEELER_TRAVEL + push_n / L.SLIDE_K
    else:
        x = lead
    end_y = L.FRONT_Y0 - x  # where the still end sits: x past the free face
    th = 0.0
    for k in range(int(turns * steps_per_turn) + 1):
        th = TWO_PI * k / steps_per_turn
        travel = L.FRONT_Y0 - end_y - c.depth  # what the feeler and the slide take up together
        if travel <= 0:
            qf, qs = 0.0, c.depth
        else:
            qf = travel / (1 + L.FEELER_K / L.SLIDE_K)
            if qf > L.FEELER_TRAVEL:
                qf = L.FEELER_TRAVEL
            qs = L.FRONT_Y0 - end_y - qf
        c.update(sharpener(slide=qs, front=qf, crank=S.wrap_to_pi(th), w=3.0), pencil_in_bore(end_y))
    return c.depth


class StillPencil(unittest.TestCase):
    def test_push_and_hold_with_the_demo_press(self):
        d = stiff_hold(4.0)
        self.assertAlmostEqual(d, 0.002, delta=0.0003)
        self.assertEqual(S.tier_of_depth(d), 0.0)

    def test_hard_push_and_hold(self):
        for f in (20.0, 50.0):
            d = stiff_hold(f)
            self.assertLess(d, 0.0040)
            self.assertEqual(S.tier_of_depth(d), 0.0, f)

    def test_pushing_150_n_would_be_needed(self):
        self.assertLess(stiff_hold(140.0), 0.006)


class Drag(unittest.TestCase):
    def _pressed(self, w, depth=0.0):
        c = S.Cutter()
        c.depth = depth
        c.update(sharpener(slide=depth, front=0.004, w=w), pencil_in_bore(pressed_end(depth)))
        return c

    def test_full_drag_above_saturation(self):
        c = self._pressed(3.49)
        self.assertAlmostEqual(c.drag, L.T_CUT)
        t = S.crank_target(0.5, 3.49, c.drag)
        self.assertAlmostEqual(t, 0.5 + 3.49 * L.DT / 2 + (L.CRANK_DAMP * 3.49 - L.T_CUT) / L.CRANK_K)
        # mean torque over the step at steady speed: K (target - mean q) - D w = -drag
        mean_q = 0.5 + 3.49 * L.DT / 2
        self.assertAlmostEqual(L.CRANK_K * (t - mean_q) - L.CRANK_DAMP * 3.49, -L.T_CUT)

    def test_drag_rises_with_speed(self):
        self.assertAlmostEqual(self._pressed(0.25).drag, L.T_CUT / 2)
        self.assertAlmostEqual(self._pressed(0.5).drag, L.T_CUT)

    def test_no_drag_stopped_backward_unpressed_or_done(self):
        for w in (0.0, -2.0):
            self.assertEqual(self._pressed(w).drag, 0.0)
        c = S.Cutter()
        c.update(sharpener(w=3.0), pencil_on_rest())
        self.assertEqual(c.drag, 0.0)
        self.assertEqual(self._pressed(3.0, depth=L.CUT_DEPTH).drag, 0.0)
        self.assertAlmostEqual(S.crank_target(1.0, 0.0, 0.0), 1.0)

    def test_targets_stay_in_the_kit_bound(self):
        for th in (-math.pi - 0.2, math.pi + 0.2):
            for w in (-6.0, 6.0):
                self.assertLessEqual(abs(S.crank_target(th, w, L.T_CUT)), L.CRANK_LIMIT)
        self.assertEqual(S.crank_target(100.0, 0.0, 0.0), L.CRANK_LIMIT)

    def test_wrap_shift(self):
        self.assertEqual(S.wrap_shift(0.0), 0.0)
        self.assertEqual(S.wrap_shift(math.pi), 0.0)
        self.assertAlmostEqual(S.wrap_shift(math.pi + 0.01), -TWO_PI)
        self.assertAlmostEqual(S.wrap_shift(-math.pi - 0.01), TWO_PI)


# -- the scorer ----------------------------------------------------------------------------------------------------


class Tiers(unittest.TestCase):
    def test_depth_tiers(self):
        for d, tier in ((0.0, 0.0), (0.0059, 0.0), (0.0060, 0.25), (0.0119, 0.25), (0.0120, 0.5), (0.0239, 0.5),
                        (0.0240, 0.75)):
            self.assertEqual(S.tier_of_depth(d), tier, d)

    def test_doing_nothing_scores_zero(self):
        t, c = S.Tracker(), S.Cutter()
        res = run_tracker(t, c, pencil_on_rest(), AWAY, 60)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.success)

    def test_every_tier_is_reachable(self):
        for turns, tier in ((4.0, 0.25), (8.0, 0.5), (16.0, 0.75)):
            c = S.Cutter()
            step_cut(c, 0.0)
            turn(c, turns)
            res = run_tracker(S.Tracker(), c, pencil_in_bore(pressed_end(c.depth)), arms_holding(
                (L.SH_X, pressed_end(c.depth) + T.TCP_TO_END, BORE_Z)), 30)
            self.assertEqual(res.score, tier, turns)

    def test_full_score(self):
        c = full_cut()
        t = S.Tracker()
        res = run_tracker(t, c, pencil_on_rest(), AWAY, WIN - 1)
        self.assertEqual(res.score, 0.75)  # released and still for less than 1 s
        res = run_tracker(t, c, pencil_on_rest(), AWAY, 2)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def _top(self, pencil=None, arms="away", steps=WIN + 5):
        c = full_cut()
        return run_tracker(S.Tracker(), c, pencil if pencil is not None else pencil_on_rest(),
                           AWAY if isinstance(arms, str) and arms == "away" else arms, steps)

    def test_each_top_condition_broken_alone(self):
        self.assertEqual(self._top().score, 1.0)
        cases = {
            "notch 2.1 mm high": pencil_on_rest(dz=0.0021),
            "notch 3.1 mm sideways": pencil_on_rest(dx=0.0031),
            "one end off the rest": pencil_on_rest(dy=0.09),  # its end inside the far block's span only
        }
        for label, p in cases.items():
            self.assertEqual(self._top(p).score, 0.75, label)
        self.assertEqual(self._top(pencil_on_rest(dz=0.0019, dx=0.0029)).score, 1.0)

    def test_one_end_on_a_notch_the_other_on_the_table(self):
        tilt = math.atan2(L.SEAT_Z - L.PENCIL_R, L.HALF)  # resting on the near block, the far end on the table
        r = _rx(-tilt)
        p = pencil_at((L.REST_XY[0], L.V_BLOCK_YS[0] + 0.02, L.TOP + L.PENCIL_R + (L.SEAT_Z - L.PENCIL_R) / 2), r)
        self.assertEqual(self._top(p).score, 0.75)

    def test_not_clear_of_the_sharpener(self):
        # the pencil 9 mm from the sharpener's dial face, lying along y at the bore height
        p = pencil_at((L.SH_X, L.BODY_Y[1] + 0.009 + L.HALF, BORE_Z))
        self.assertLess(S.clearance_to_sharpener(p), L.CLEAR)
        self.assertEqual(self._top(p).score, 0.75)
        p = pencil_at((L.SH_X, L.BODY_Y[1] + 0.0101 + L.HALF, BORE_Z))
        self.assertGreaterEqual(S.clearance_to_sharpener(p), L.CLEAR)

    def test_left_in_the_sharpener(self):
        self.assertEqual(self._top(pencil_in_bore(pressed_end(L.CUT_DEPTH))).score, 0.75)

    def test_held_or_hovering_over_the_rest(self):
        hover = pencil_on_rest(dz=0.015)
        tcp = (L.REST_XY[0], L.REST_XY[1] + T.GRIP_BACK, L.TOP + L.SEAT_Z + 0.015)
        self.assertEqual(self._top(hover, arms_holding(tcp)).score, 0.75)
        held = (L.REST_XY[0], L.REST_XY[1] + T.GRIP_BACK, L.TOP + L.SEAT_Z)
        self.assertEqual(self._top(pencil_on_rest(), arms_holding(held)).score, 0.75)

    def test_fingers_resting_on_the_pencil(self):
        tips = np.array([L.REST_XY[0], L.REST_XY[1] + 0.02, L.TOP + L.SEAT_Z + L.PENCIL_R + 0.001])
        a = rigstate.Arm(tips + np.array([0, 0, 0.013]), tips, 0.32)
        self.assertEqual(self._top(arms={"left": a, "right": AWAY["right"]}).score, 0.75)

    def test_finger_zone_29_and_31_mm(self):
        p = pencil_on_rest()
        r_open = rigstate.zone_radius(1.0)
        top_y = L.TOP + L.SEAT_Z + L.PENCIL_R
        for gap, expect in ((0.029, 0.75), (0.031, 1.0)):
            tips = np.array([L.REST_XY[0], L.REST_XY[1], top_y + r_open + gap])
            a = rigstate.Arm(tips + np.array([0, 0, 0.013]), tips, 1.0)
            self.assertEqual(self._top(p, {"left": a, "right": AWAY["right"]}).score, expect, gap)

    def test_unreadable_arms_never_release(self):
        bad = rigstate.Arm(np.array([float("nan"), 0.0, 1.0]), np.array([0.0, 0.0, 1.0]), 1.0)
        for arms in (None, {}, {"left": AWAY["left"]}, {"left": AWAY["left"], "right": bad},
                     {"left": AWAY["left"], "right": "not an arm"}):
            self.assertEqual(self._top(arms=arms).score, 0.75, arms)

    def test_released_less_than_a_second(self):
        c = full_cut()
        t = S.Tracker()
        held = arms_holding((L.REST_XY[0], L.REST_XY[1] + T.GRIP_BACK, L.TOP + L.SEAT_Z))
        run_tracker(t, c, pencil_on_rest(), held, 30)
        res = run_tracker(t, c, pencil_on_rest(), AWAY, WIN - 1)
        self.assertEqual(res.score, 0.75)

    def test_moving_pencil(self):
        c = full_cut()
        t = S.Tracker()
        states = {"sharpener": sharpener(slide=c.depth), "rest": T.REST.rest_state()}
        res = None
        for k in range(WIN + 5):  # rolling 0.11 mm a step along x: 2.2 mm a second
            states["pencil"] = pencil_on_rest(dx=-0.0012 + 0.00011 * k)
            res = t.update(states, AWAY, c)
        self.assertEqual(res.score, 0.75)

    def test_put_back_then_picked_up_again(self):
        c = full_cut()
        t = S.Tracker()
        self.assertEqual(run_tracker(t, c, pencil_on_rest(), AWAY, WIN + 2).score, 1.0)
        tcp = (L.REST_XY[0], L.REST_XY[1] + T.GRIP_BACK, L.TOP + 0.08)
        self.assertEqual(run_tracker(t, c, pencil_on_rest(dz=0.05), arms_holding(tcp), 1).score, 0.75)

    def test_broken_state_scores_zero_and_restarts_windows(self):
        c = full_cut()
        t = S.Tracker()
        run_tracker(t, c, pencil_on_rest(), AWAY, WIN + 2)
        res = t.update({"pencil": None, "sharpener": sharpener()}, AWAY, c)
        self.assertEqual(res.score, 0.0)
        res = t.update({"pencil": pencil_at((float("nan"), 0, 0)), "sharpener": sharpener()}, AWAY, c)
        self.assertEqual(res.score, 0.0)
        res = run_tracker(t, c, pencil_on_rest(), AWAY, WIN - 1)
        self.assertEqual(res.score, 0.75)
        res = run_tracker(t, c, pencil_on_rest(), AWAY, 2)
        self.assertEqual(res.score, 1.0)

    def test_drop_into_the_notches_is_allowed(self):
        c = full_cut()
        t = S.Tracker()
        for dz in (0.03, 0.015, 0.0):
            run_tracker(t, c, pencil_on_rest(dz=dz), AWAY, 1)
        self.assertEqual(run_tracker(t, c, pencil_on_rest(), AWAY, WIN + 1).score, 1.0)


class Resets(unittest.TestCase):
    def test_tracker_reset(self):
        c = full_cut()
        t = S.Tracker()
        run_tracker(t, c, pencil_on_rest(), AWAY, WIN + 2)
        t.reset()
        c.reset()
        self.assertEqual(run_tracker(t, c, pencil_on_rest(), AWAY, WIN + 2).score, 0.0)

    def test_teleport_back_to_start_resets_the_cut_too(self):
        c = full_cut()
        t = S.Tracker()
        run_tracker(t, c, pencil_in_bore(pressed_end(c.depth)), AWAY, 3)
        res = run_tracker(t, c, pencil_on_rest(), AWAY, 1)  # jumps back to its start pose in one step
        self.assertEqual(t.resets, 1)
        self.assertEqual(c.depth, 0.0)
        self.assertEqual(res.score, 0.0)

    def test_task_success_resets_on_a_new_mechkit_episode(self):
        mk.offline_reset()
        states = {"pencil": pencil_on_rest(), "sharpener": sharpener(), "rest": T.REST.rest_state()}
        T.success(states)  # in step with mechkit's episode
        T.CUTTER.depth = L.CUT_DEPTH
        T._TRACKER.steps = 99
        mk.offline_reset()
        T.success(states)
        self.assertEqual(T.CUTTER.depth, 0.0)
        self.assertEqual(T._TRACKER.steps, 1)


# -- the offline reconstruction of the joints ------------------------------------------------------------------------


class OfflineMechanism(unittest.TestCase):
    def _right(self, phi, opening=0.13):
        x, y, z = T.knob_point(phi)
        tcp = np.array([x, y, z])
        return {"left": AWAY["left"], "right": rigstate.Arm(tcp, tcp + np.array([0, 0, -0.013]), opening)}

    def test_crank_follows_a_closed_right_gripper_on_the_knob(self):
        o = T.OfflineSharpener()
        c = S.Cutter()
        for k in range(1, 20):
            s = o.state(pencil_on_rest(), self._right(0.1 * k), c)
        self.assertAlmostEqual(s.joints["crank"], 1.9, places=6)
        self.assertGreater(s.joint_vel["crank"], 0.0)

    def test_open_or_distant_gripper_leaves_it(self):
        o = T.OfflineSharpener()
        c = S.Cutter()
        o.state(pencil_on_rest(), self._right(0.0), c)
        s = o.state(pencil_on_rest(), self._right(0.1, opening=0.5), c)
        self.assertEqual(s.joints["crank"], 0.0)
        s = o.state(pencil_on_rest(), self._right(0.4), c)  # 18 mm away: not on the knob
        self.assertEqual(s.joints["crank"], 0.0)

    def test_crank_follows_the_demo_command_outside_the_knob_circle(self):
        o = T.OfflineSharpener()
        c = S.Cutter()
        for k in range(0, T.ORBIT_STEPS + 1):
            tcp = np.array(T.orbit_point(k))
            arms = {"left": AWAY["left"], "right": rigstate.Arm(tcp, tcp + np.array([0, 0, -0.013]), 0.13)}
            s = o.state(pencil_on_rest(), arms, c)
            self.assertAlmostEqual(S.wrap_to_pi(s.joints["crank"] - T.orbit_angle(k)), 0.0, places=6)

    def test_feeler_from_the_pencil(self):
        o = T.OfflineSharpener()
        c = S.Cutter()
        s = o.state(pencil_in_bore(L.FRONT_Y0 - 0.003), AWAY, c)
        self.assertAlmostEqual(s.joints["front"], 0.003, places=6)
        s = o.state(pencil_in_bore(L.FRONT_Y0 - 0.026), AWAY, c)
        self.assertAlmostEqual(s.joints["front"], L.FEELER_TRAVEL, places=6)
        s = o.state(pencil_on_rest(), AWAY, c)
        self.assertEqual(s.joints["front"], 0.0)


# -- the demo --------------------------------------------------------------------------------------------------------


class Demo(unittest.TestCase):
    def setUp(self):
        self.env = {k: os.environ.pop(k) for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT")
                    if k in os.environ}

    def tearDown(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
            os.environ.pop(k, None)
        os.environ.update(self.env)

    def test_orbit_angle(self):
        self.assertEqual(T.orbit_angle(0), 0.0)
        self.assertAlmostEqual(T.orbit_angle(T.ORBIT_STEPS), T.TURNS * TWO_PI, places=9)
        prev = 0.0
        for k in range(1, T.ORBIT_STEPS + 1):
            a = T.orbit_angle(k)
            self.assertGreaterEqual(a, prev)
            self.assertLessEqual(a - prev, TWO_PI / T.STEPS_PER_TURN + 1e-9)
            prev = a

    def test_orbit_radius_blends_in_and_out(self):
        self.assertAlmostEqual(T.orbit_radius(0), L.CRANK_R)
        self.assertAlmostEqual(T.orbit_radius(T.ORBIT_STEPS), L.CRANK_R)
        self.assertAlmostEqual(T.orbit_radius(T.ORBIT_STEPS // 2), L.CRANK_R + T.ORBIT_PULL)
        self.assertAlmostEqual(T.ORBIT_PULL, 0.015)
        prev = np.array(T.orbit_point(0))
        self.assertTrue(np.allclose(prev, T.knob_point(0.0)))  # starts on the gripped knob
        for k in range(1, T.ORBIT_STEPS + 1):
            p = np.array(T.orbit_point(k))
            # no jump: at most 10.5 mm a step (0.21 m/s at full speed on the 60 mm circle), the radius 0.75 mm a step
            self.assertLess(np.linalg.norm(p - prev), 0.0106)
            self.assertLessEqual(abs(T.orbit_radius(k) - T.orbit_radius(k - 1)), T.ORBIT_PULL / T.RAMP_STEPS + 1e-12)
            prev = p
        self.assertTrue(np.allclose(prev, T.knob_point(0.0)))  # ends on the knob at the top
        back = [T.orbit_point(k, -1.0) for k in (0, T.ORBIT_STEPS // 2, T.ORBIT_STEPS)]
        self.assertTrue(np.allclose(back[0], T.knob_point(0.0)) and np.allclose(back[2], T.knob_point(0.0)))

    def test_compliant_arm_carries_the_crank(self):
        """Static model: the right arm as joint springs (80 N m/rad j1 to j3, 10 j4 to j6, the yam_arena gains), the
        gripper's rotation about x and z locked to the knob by the crank, free about the knob's axis. The pull it can
        put on the knob, maximised over the command's lead angle, is the torque it can turn the crank with. At the
        weakest point (about 150 deg) the 45 mm circle gives 0.235 N m, 18 % over the 0.2 N m load (a 20 % softer
        arm stalls it); the demo's 60 mm circle gives 0.37 N m, and 0.29 N m with a 20 % softer arm."""
        if OFFLINE is None:
            self.skipTest("tools/offline not found")
        import yamkin as K
        arm = K.arm()
        base = K.BASES["station"]["right"]
        rot = K.matrix_to_quat_xyzw(K.grip_rotation(T.YAW_R))
        load = L.BEARING_FRICTION + L.T_CUT

        def weakest(radius, soft):
            kq = np.diag([80.0, 80.0, 80.0, 10.0, 10.0, 10.0]) * soft
            q = np.array(K.READY, dtype=float)
            worst = None
            for th in np.radians(np.arange(0.0, 360.0, 6.0)):
                q, err, _ = arm.solve(T.knob_point(th), rot, q_init=q, base_pos=base, iters=200)
                self.assertLess(err, 1e-3)
                _, _, jl, ja = arm.tcp_pose_jacobian(q, base)
                j = np.vstack([jl, ja])
                k6 = np.linalg.inv(j @ np.linalg.inv(kq) @ j.T)
                lin, ry = [0, 1, 2], [4]
                keff = k6[np.ix_(lin, lin)] - k6[np.ix_(lin, ry)] @ np.linalg.inv(k6[np.ix_(ry, ry)]) @ k6[np.ix_(ry, lin)]
                x = np.array(T.knob_point(th))
                t = np.array([math.cos(th), 0.0, -math.sin(th)])
                best = max(L.CRANK_R * float(t @ keff @ (np.array(T.knob_point(th + d, radius)) - x))
                           for d in np.radians(np.arange(0.0, 181.0, 3.0)))
                worst = best if worst is None else min(worst, best)
            return worst

        on_knob = weakest(L.CRANK_R, 1.0)
        self.assertGreater(on_knob, load)
        self.assertLess(on_knob, 1.2 * load)  # why the demo pulls on a larger circle
        self.assertLess(weakest(L.CRANK_R, 0.8), load)
        self.assertGreater(weakest(L.CRANK_R + T.ORBIT_PULL, 1.0), 1.75 * load)
        self.assertGreater(weakest(L.CRANK_R + T.ORBIT_PULL, 0.8), 1.4 * load)

    def test_feed_follows_the_cut_then_holds(self):
        self.assertAlmostEqual(T.feed_end_y(0.0), L.FRONT_Y0 - T.LEAD)
        self.assertAlmostEqual(T.feed_end_y(8.0) - T.feed_end_y(0.0), -0.012)
        self.assertEqual(T.feed_end_y(16.0), T.feed_end_y(17.0))
        # the deepest command leaves the left pads 13 mm off the dial face even if the pencil went there
        deepest_tcp = T.end_to_tcp_y(T.feed_end_y(16.0))
        self.assertGreater(deepest_tcp - 0.010 - L.BODY_Y[1], 0.012)
        # the pencil cannot go deeper than the stop: there the pads are 34 mm off the face
        stop_tcp = T.end_to_tcp_y(L.FRONT_Y0 - L.CUT_DEPTH - L.FEELER_TRAVEL)
        self.assertGreater(stop_tcp - 0.010 - L.BODY_Y[1], 0.033)

    def test_press_with_a_compliant_arm(self):
        """The feeler reading the demo's lead gives with an arm of 120 to 400 N/m along y (static model ~200)."""
        for k_arm, low in ((120.0, 0.0026), (200.0, 0.0037), (400.0, L.FEELER_TRAVEL)):
            qf = min(L.FEELER_TRAVEL, T.LEAD / (1 + L.FEELER_K / k_arm))
            self.assertGreaterEqual(qf, low - 1e-4)
            self.assertGreaterEqual(qf, L.PRESS)

    def test_plans(self):
        plan = T.demo()
        self.assertEqual(set(plan), {"left", "right"})
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        del os.environ["DEMO_HOLD_ONLY"]
        for idle in ("left", "right"):
            os.environ["DEMO_IDLE_ARM"] = idle
            self.assertEqual(set(T.demo()), {"left", "right"} - {idle})
        del os.environ["DEMO_IDLE_ARM"]
        for v in ("still", "backward", "let_go", "no_layback", "table"):
            os.environ["DEMO_VARIANT"] = v
            self.assertEqual(set(T.demo()), {"left", "right"})

    def test_timing_and_run_limit(self):
        plan = T.demo()
        left = sum(w.duration for w in plan["left"])
        right = sum(w.duration for w in plan["right"])
        press = sum(w.duration for w in T._left_before(""))
        self.assertAlmostEqual(press, T.LEFT_PRESS_S, places=6)
        self.assertAlmostEqual(T.ORBIT_STEPS * T.STEP_S, 31.6, places=6)
        steps = round(max(left, right) * L.CONTROL_HZ)
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        self.assertLessEqual(steps + 60 + 40, run_limit)  # the policy's 3 s end hold and 2 s more to spare
        self.assertEqual(run_limit, 1200)

    def test_open_and_close_are_slow_and_open_loop(self):
        for arm, wps in T.demo().items():
            prev = None
            for w in wps:
                if prev is not None and abs(w.finger - prev) > 0.004:
                    self.assertGreaterEqual(w.duration, 1.5, (arm, w.label))
                prev = w.finger
        src = (HERE / "task.py").read_text()
        body = src[src.index("# -- the scripted demo"):]
        self.assertNotIn("CUTTER", body)  # the demo reads no simulator state
        self.assertNotIn("rigstate.arms", body)

    def test_grips(self):
        self.assertAlmostEqual(-2 * T.GRIP_PENCIL, 2 * L.PENCIL_R - 0.008)
        self.assertAlmostEqual(-2 * T.GRIP_KNOB, 2 * L.KNOB_R - 0.008)
        self.assertGreater(-2 * T.OPEN_PENCIL - 2 * L.PENCIL_R, 0.019)
        self.assertGreater(-2 * T.OPEN_KNOB - 2 * L.KNOB_R, 0.015)
        # the left pads (20 mm wide round the TCP) between the two V blocks
        self.assertGreater(L.V_BLOCK_YS[1] - L.V_BLOCK[1] / 2 - (T.PICK[1] + 0.010), 0.006)

    def test_release_height_clears_the_zone(self):
        tips_z = L.TOP + T.LIFT_H - 0.013
        gap = tips_z - (L.TOP + L.SEAT_Z + L.PENCIL_R) - rigstate.zone_radius(-2 * T.OPEN_PENCIL / rigstate.FULL_GAP)
        self.assertGreater(gap, L.LET_GO + 0.015)


# -- the text --------------------------------------------------------------------------------------------------------


class Text(unittest.TestCase):
    LIMITS = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
              "reset_notes": 580}

    def setUp(self):
        self.spec = json.loads((HERE / "task.json").read_text())

    def test_fields_and_limits(self):
        for k, n in self.LIMITS.items():
            self.assertLessEqual(len(self.spec[k]), n, k)
        self.assertEqual(self.spec["category"], "Societal relatability")
        self.assertEqual(self.spec["instruction"], T.INSTRUCTION)
        self.assertRegex(self.spec["scores"], r"^Dream-1 run: [0-9.]+; demo: 1\.0$")

    def test_no_dashes_or_banned_words(self):
        text = "\n".join([*(str(v) for v in self.spec.values()), (HERE / "README.md").read_text()])
        for ch in ("\u2014", "\u2013"):
            self.assertNotIn(ch, text)
        for w in (r"\bscale\b", r"\bscaled\b", r"model of", r"analogue", r"\bmock", r"trainer", r"stand-in",
                  r"stands for", r"represents", r"\bTBD\b(?! after the GPU)", r"placeholder", r"to be measured"):
            self.assertIsNone(re.search(w, text, re.I), w)

    def test_numbers_agree(self):
        joined = " ".join(str(v) for v in self.spec.values())
        for s in ("1.5 mm", "24 mm", "16 turns", "0.15 N m", "0.05 N m", "12.3 mm", "166 mm", "10.3 mm", "45 mm"):
            self.assertIn(s, joined, s)


if __name__ == "__main__":
    unittest.main()
