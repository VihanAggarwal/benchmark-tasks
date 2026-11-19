"""Offline tests for layout.py, parts.py, scoring.py, offline_model.py and task.py: synthetic states (mechkit
AssemblyStates from the task's own specs), synthetic arm snapshots (rigstate.Arm) and the demo replayed through the
offline stand-in, no simulator. Where tools/offline is found, the arms' collision capsules check the plan too.
    python3 -m unittest discover -s tests -v"""

import hashlib
import json
import math
import os
import pathlib
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


OFFLINE_TOOLS = _offline_dir()
if OFFLINE_TOOLS is not None:
    for _p in (OFFLINE_TOOLS, OFFLINE_TOOLS / "yam_stub"):
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_model as OM  # noqa: E402
import parts as P  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
from yam_arena import demo as D  # noqa: E402

HZ = L.CONTROL_HZ
WIN = L.WIN_STEPS
BASES = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}
AWAY = {a: rigstate.Arm(np.array([0.25, s * 0.32, L.TOP + 0.18]), np.array([0.25, s * 0.32, L.TOP + 0.167]), 0.4)
        for a, s in (("left", 1.0), ("right", -1.0))}
PLACED = P.placed_origin()
DASHES = (chr(0x2014), chr(0x2013))  # em and en dashes: none anywhere
SHOWN_AT = (P.TH_ORIGIN[0] + 0.012, P.TH_ORIGIN[1], P.TH_ORIGIN[2])  # laid back near its start, display up (12 mm
# off it: a synthetic jump straight back onto the start pose reads as a reset)


def states(theta=0.0, th=P.TH_ORIGIN, yaw=0.0, button=0.0, doll_pose=None):
    return {P.DOLL: P.doll_state(theta, doll_pose), P.THERMO: P.thermo_state(th, yaw, button),
            P.MAT: P.MAT_SPEC.rest_state()}


def run(tr, st, n, arms=AWAY):
    res = None
    for _ in range(n):
        res = tr.update(st, arms)
    return res


def press(tr, theta=0.0, th=P.TH_ORIGIN, hold_s=0.5):
    """A press of the On/Off button (full travel for ``hold_s``, then released)."""
    run(tr, states(theta, th, button=L.BUTTON_TRAVEL), int(round(hold_s * HZ)))
    return run(tr, states(theta, th), 1)


def reading(tr):
    """Switched on, then the tip on the sensor under the lowered arm for 11 s and more."""
    press(tr)
    run(tr, states(), int(L.READY_S * HZ) + 2)
    rest = P.support_angle(P.thermo_state(PLACED), clear=OM.REST_GAP)
    return run(tr, states(rest, PLACED), int(L.PEAK_S * HZ) + 4)


def full_trial(off=True, arm_back=True, shown=True):
    tr = S.Tracker()
    reading(tr)
    theta = 0.0 if arm_back else math.radians(30.0)
    if shown:
        run(tr, states(theta, SHOWN_AT), int(L.SHOWN_S * HZ) + 4)
    if off:
        press(tr, theta, SHOWN_AT)
    return run(tr, states(theta, SHOWN_AT), WIN + 4), tr


def replay(idle=None, hold_s=3.0, skip=None, on_step=None):
    """The demo's commanded motion through the offline stand-in and the scorer, as task.success plays it offline:
    (final score, step at which success first held, the model, the tracker). ``skip`` (first, last): from the first
    waypoint whose label contains ``first`` through the next whose label contains ``last``, the arm stays put.
    ``on_step(t, arms, yaws, model)``: called after each step."""
    plans = T.plans()
    if idle:
        plans.pop(idle, None)
    if skip:
        first, last = skip
        for ws in plans.values():
            on = False
            for i, w in enumerate(ws):
                on = on or (first in w.label and i > 0)
                if on:
                    ws[i] = D.Waypoint(w.duration, ws[i - 1].pos, ws[i - 1].finger, w.label + " (skipped)", w.yaw)
                    if last in w.label:
                        break
    ap = {s: D.ArmPlan(w, BASES[s]) for s, w in plans.items()}
    total = max(p.duration for p in ap.values()) + hold_s
    model, tr = OM.OfflineModel(), S.Tracker()
    start = {s: D.ArmPlan([], BASES[s]).start_pos for s in BASES}
    success_at, res = None, None
    for k in range(1, int(round(total * HZ)) + 1):
        t = k / HZ
        arms, yaws = {}, {}
        for s in BASES:
            if s in ap:
                pos, _rot, finger, _label = ap[s].target(t)
                yaws[s] = plans[s][0].yaw  # each arm's plan keeps one gripper heading (task._Plan)
            else:
                pos, finger = start[s], rigstate.FINGER_OPEN
            pos = np.asarray(pos, dtype=float)
            arms[s] = rigstate.Arm(pos, pos - np.array([0.0, 0.0, 0.013]),
                                   float(np.clip(finger / rigstate.FINGER_OPEN, 0.0, 1.0)))
        model.step(arms, tr.unit.on)
        res = tr.update(model.states(), arms)
        if res.success and success_at is None:
            success_at = k
        if on_step is not None:
            on_step(t, arms, yaws, model)
    return res.score, success_at, model, tr


# -- the kit and the objects ------------------------------------------------------------------------------------------
class Kit(unittest.TestCase):
    def test_kit_files_are_unchanged_copies(self):
        kit = HERE.parents[1] / "tools" / "kit"
        if not kit.is_dir():
            self.skipTest("tools/kit not found")
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                             hashlib.sha256((kit / name).read_bytes()).hexdigest(), name)

    def test_objects(self):
        self.assertEqual([o.name for o in T.OBJECTS], [P.MAT, P.DOLL, P.THERMO])
        self.assertEqual([j.name for j in P.DOLL_SPEC.joints], [P.ARM_LINK])
        self.assertEqual([j.name for j in P.THERMO_SPEC.joints], ["button", "display"])
        for spec, m in ((P.THERMO_SPEC, 0.015), (P.DOLL_SPEC, L.DOLL_MASS + L.ARM_MASS)):
            self.assertAlmostEqual(sum(spec.body_mass(b) for b in spec.body_names), m, places=6)
        self.assertEqual((T.SCORE_OVER, T.SCENE), ("end", "station"))

    def test_task_json(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertIn("demo: 1.0", d["scores"])
        self.assertNotIn("pending", d["scores"])
        self.assertIn("3200 steps", d["run_limit"])
        text = json.dumps(d)
        for s in DASHES:
            self.assertNotIn(s, text)
        for n in ("y 90", "28 mm from the tail", "22 deg or less", "within 0.5 mm", "460 x 600", "more than 45 deg",
                  "more than 30 mm", "3.5 mm from the armpit sensor film"):
            self.assertIn(n, text, n)
        self.assertNotIn("within 2 mm", text)
        self.assertEqual(d["instruction"], T.INSTRUCTION)  # the README quotes the same text
        self.assertIn(d["instruction"], " ".join((HERE / "README.md").read_text().split()))

    def test_no_dashes_in_the_files(self):
        for f in list(HERE.glob("*.py")) + [HERE / "README.md", HERE / "task.json"] + list(HERE.glob("tests/*.py")):
            text = f.read_text()
            for dash in DASHES:
                self.assertNotIn(dash, text, f.name)


# -- the device's rules -----------------------------------------------------------------------------------------------
class Device(unittest.TestCase):
    def step(self, u, n, travel=0.0, closed=False):
        for _ in range(n):
            u.update(travel, closed)

    def test_a_press_switches_on_and_off(self):
        u = S.Unit()
        self.step(u, 10)
        self.assertFalse(u.on)
        self.step(u, 6, L.BUTTON_TRAVEL)
        self.assertFalse(u.on)  # switches on release
        self.step(u, 1)
        self.assertTrue(u.on)
        self.step(u, 6, L.BUTTON_TRAVEL)
        self.step(u, 1)
        self.assertFalse(u.on)
        self.assertEqual(u.last_off_by_button(), u.events[-1][0])

    def test_a_light_touch_does_nothing(self):
        u = S.Unit()
        self.step(u, 10, 0.0002)
        self.step(u, 1)
        self.assertFalse(u.on)

    def test_a_5_s_hold_opens_the_setting_and_does_not_switch(self):
        u = S.Unit()
        self.step(u, int(L.HOLD_SETTING_S * HZ) + 2, L.BUTTON_TRAVEL)
        self.step(u, 1)
        self.assertFalse(u.on)
        self.assertTrue(u.setting_opened)

    def test_ready_after_2_s(self):
        u = S.Unit()
        self.step(u, 4, L.BUTTON_TRAVEL)
        self.step(u, 1)
        self.assertTrue(u.on and not u.ready)
        self.step(u, int(L.READY_S * HZ))
        self.assertTrue(u.ready)

    def test_reading_needs_11_s_unbroken(self):
        u = S.Unit()
        self.step(u, 4, L.BUTTON_TRAVEL)
        self.step(u, 1 + int(L.READY_S * HZ))
        self.step(u, int(L.PEAK_S * HZ) - 2, closed=True)
        self.step(u, 1, closed=False)  # a break restarts the count
        self.step(u, int(L.PEAK_S * HZ) - 2, closed=True)
        self.assertFalse(u.reading)
        self.step(u, 3, closed=True)
        self.assertTrue(u.reading)
        self.step(u, 40, closed=False)  # it holds the result
        self.assertTrue(u.reading)

    def test_no_count_before_ready(self):
        u = S.Unit()
        self.step(u, 4, L.BUTTON_TRAVEL)
        self.step(u, 1)
        self.step(u, int(L.PEAK_S * HZ), closed=True)
        self.assertFalse(u.reading)

    def test_auto_off(self):
        u = S.Unit()
        self.step(u, 4, L.BUTTON_TRAVEL)
        self.step(u, int(L.AUTO_OFF_S * HZ) - 2)
        self.assertTrue(u.on)
        self.step(u, 4)
        self.assertFalse(u.on)
        self.assertIsNone(u.last_off_by_button())  # off by itself
        u = S.Unit()
        self.step(u, 4, L.BUTTON_TRAVEL)
        self.step(u, 1 + int(L.READY_S * HZ))
        self.step(u, int(L.PEAK_S * HZ) + 1, closed=True)
        self.assertTrue(u.reading)
        self.step(u, int(L.AUTO_OFF_S * HZ) - 4)
        self.assertTrue(u.on)  # 90 s counted from the reading
        self.step(u, 6)
        self.assertFalse(u.on)

    def test_display_hook_follows_the_unit(self):
        T._TRACKER.reset()
        T._TRACKER.unit.on = True
        try:
            T._display({})  # offline: no environment, a no-op that must not raise
        finally:
            T._TRACKER.reset()


# -- geometry ----------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_start_layout(self):
        r = S.read(states())
        g = S.geometry(r)
        self.assertTrue(g["over_mat"])
        self.assertLess(abs(g["lowest"] - r.mat_top), 1e-6)
        self.assertGreater(g["tip_gap"], 0.15)
        self.assertGreater(g["arm_gap"], 0.05)
        self.assertGreater(P.thermo_to_torso(P.doll_state(0.0), P.thermo_state(P.TH_ORIGIN)), 0.03)
        self.assertLess(g["up_deg"], 1e-6)

    def test_the_armpit_is_closed_until_the_arm_is_lifted(self):
        # The upper arm's inner face is 3.5 mm from the film, under the 4 mm tip: the tip cannot get in with the arm
        # down (the placed thermometer goes through the arm at 0 deg).
        gap = L.FSR_C[0] - L.FSR_T - (L.UPPER_ARM_X + L.UPPER_ARM_R)
        self.assertLess(gap, 2 * L.TIP_R)
        self.assertLess(P.thermo_to_arm(P.doll_state(0.0), P.thermo_state(PLACED)), 1e-9)

    def test_placed_tip_touches_the_film(self):
        th = P.thermo_state(PLACED)
        self.assertLessEqual(P.tip_to_fsr(P.doll_state(0.0), th), L.TOUCH_M)
        self.assertGreater(P.thermo_to_torso(P.doll_state(0.0), th), -1e-9)

    def test_the_arm_rests_on_the_thermometer_closed(self):
        th = P.thermo_state(PLACED)
        rest = P.support_angle(th, clear=OM.REST_GAP)
        self.assertGreater(math.degrees(rest), L.ARM_BACK_DEG)
        self.assertLess(math.degrees(rest), L.CLOSED_MAX_DEG)
        r = S.read(states(rest, PLACED))
        self.assertTrue(S.contact(r, S.geometry(r)))

    def test_the_doll_lies_flat_on_its_back(self):
        # GPU 19 Nov: the diaper's rounded part stuck 2 mm under the base plane, so the doll lay on that one line
        # (origin 2.4 mm over the mat) and rocked; with the arm resting on the thermometer it rolled about 3 deg,
        # the armpit rose off the tip (gap 3.3 mm) and the arm sank to 4 deg: Closed broke after 0.8 s.
        lows = [float(np.min(P._surface_points([s], 0.002)[:, 2])) for s in P.DOLL_SHAPES]
        self.assertGreaterEqual(min(lows), -1e-9)
        pts = P._surface_points(P.DOLL_SHAPES, 0.004)
        floor = pts[pts[:, 2] < 0.0005]
        self.assertGreater(float(np.ptp(floor[:, 0])), 0.100)  # on its back across the torso, not on a line
        self.assertGreater(float(np.ptp(floor[:, 1])), 0.400)  # and from head to feet

    def test_closed_holds_with_the_thermometer_tipped_nose_down(self):
        # The arm lands on the taper (ahead of the flat underside's front edge, 84 mm from the tail) and tips the
        # 15 g stick nose down until its front boxes meet the mat: 7.5 deg, tail up about 11 mm (GPU, both runs).
        # With the doll flat, the tip stays on the film and the arm still rests well up on it.
        for pivot_x, y, yaw_deg in ((PLACED[0] + 0.084, PLACED[1], 0.0), (0.2057, 0.0889, 2.4)):
            a, yaw = math.radians(7.5), math.radians(yaw_deg)
            origin = (pivot_x - 0.084 * math.cos(a) * math.cos(yaw), y - 0.084 * math.cos(a) * math.sin(yaw),
                      L.TOP + L.MAT_H + 0.084 * math.sin(a))
            qz = (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
            qy = (0.0, math.sin(a / 2), 0.0, math.cos(a / 2))
            q = (qz[3] * qy[0] - qz[2] * qy[1], qz[3] * qy[1] + qz[2] * qy[0], qz[2] * qy[3], qz[3] * qy[3])
            th = mk.state_from_poses(P.THERMO_SPEC, (*origin, *q), joints={"button": 0.0, "display": 0.0})
            self.assertAlmostEqual(float(np.min(th.to_world(P.TH_POINTS)[:, 2])), L.TOP + L.MAT_H, delta=0.0006)
            rest = P.support_angle(th, clear=OM.REST_GAP)
            self.assertGreater(math.degrees(rest), 11.0)
            self.assertLess(math.degrees(rest), L.CLOSED_MAX_DEG)
            st = {P.DOLL: P.doll_state(rest), P.THERMO: th, P.MAT: P.MAT_SPEC.rest_state()}
            r = S.read(st)
            g = S.geometry(r)
            self.assertLessEqual(g["tip_gap"], 0.0015)
            self.assertLessEqual(g["arm_gap"], L.TOUCH_M)
            # on the line of the film the tip touches it (Closed); turned 2.4 deg it stops 1.0 to 1.3 mm short of the
            # film, which is not contact under the 0.5 mm rule (run 2 of the review gate: no credit without contact)
            self.assertEqual(S.contact(r, g), yaw_deg == 0.0, (yaw_deg, g["tip_gap"]))

    def test_the_slide_path_is_clear_with_the_arm_up(self):
        doll = P.doll_state(math.radians(T.LIFT_DEG))
        for dx in np.linspace(-T.SLIDE, 0.0, 10):
            th = P.thermo_state((PLACED[0] + dx, PLACED[1], PLACED[2]))
            self.assertGreater(P.thermo_to_arm(doll, th), 0.008)

    def test_shown_pose_away_from_arm_and_armpit(self):
        r = S.read(states(0.0, SHOWN_AT))
        g = S.geometry(r)
        self.assertGreater(g["tip_gap"], L.SHOWN_TIP_M)
        self.assertGreater(g["arm_gap"], L.SHOWN_ARM_M)

    def test_hand_kinematics_match_mechkit(self):
        for deg in (0.0, 15.0, 40.0, 90.0):
            th = math.radians(deg)
            doll = P.doll_state(th)
            got = doll.link_to_world(P.ARM_LINK, P.HAND_LOCAL)
            np.testing.assert_allclose(got, P.hand_at(th), atol=1e-6)
            self.assertAlmostEqual(P.angle_of(P.hand_at(th)), th, places=6)

    def test_the_arm_falls_back_below_about_70_deg(self):
        lever = np.linalg.norm((P.HAND_C - P.AXIS_WORLD)[1:]) * 0.6  # the arm's centre of mass, est 60 % out
        torque = L.ARM_MASS * 9.81 * lever
        self.assertGreater(torque, L.SHOULDER_FRICTION)  # flat: it falls
        hold = math.degrees(math.acos(L.SHOULDER_FRICTION / torque))
        self.assertTrue(60.0 < hold < 80.0, hold)


@unittest.skipIf(OFFLINE_TOOLS is None, "tools/offline not found")
class Reach(unittest.TestCase):
    """The left gripper's housing clears the doll at the pinch, and the two arms' capsules stay apart all through."""

    @classmethod
    def setUpClass(cls):
        import yamkin as K
        cls.K = K

    def solve(self, side, p, yaw, q0=None):
        K = self.K
        base = K.BASES["station"][side]
        r = K.solve(np.asarray(p, dtype=float), K.matrix_to_quat_xyzw(K.grip_rotation(yaw)),
                    q_init=np.array(K.READY if q0 is None else q0, dtype=float), base_pos=base)
        q = r[0] if isinstance(r, tuple) else r
        pos, _ = K.tcp_pose(q, base)
        return q, base, float(np.linalg.norm(pos - np.asarray(p)))

    def test_housing_clears_the_doll_at_the_pinch(self):
        K = self.K
        for deg in (0.0, math.degrees(T.THETA_REST), T.LIFT_DEG):
            th = math.radians(deg)
            q, base, err = self.solve(T.HAND_ARM, T._hand_tcp(th), T.YAW_HAND)
            self.assertLess(err, 0.001)
            doll = P.doll_state(th)
            for name, c0, c1, rr in K.arm().capsules(q, base, T.GRIP_HAND):
                if name in K.FINGER_LINKS:
                    continue
                pts = np.array([c0 + (c1 - c0) * s for s in np.linspace(0.0, 1.0, 25)])
                self.assertGreater(float(np.min(doll.distance(pts))) - rr, 0.005, (deg, name))

    def test_arms_apart_through_the_plan(self):
        K = self.K
        plans = T.plans()
        ap = {s: D.ArmPlan(w, BASES[s]) for s, w in plans.items()}
        end = max(p.duration for p in ap.values())
        q = {s: np.array(K.READY, dtype=float) for s in ap}
        worst = (1.0, None)
        for t in np.arange(0.0, end, 0.1):
            caps = {}
            for s in ap:
                pos, _rot, finger, label = ap[s].target(float(t))
                w = next(w for w in plans[s] if w.label == label)
                q[s], base, _ = self.solve(s, pos, w.yaw, q[s])
                caps[s] = K.arm().capsules(q[s], base, finger)
            c = K.capsule_clearance(caps["left"], caps["right"])[0]
            if c < worst[0]:
                worst = (c, round(float(t), 1))
        self.assertGreater(worst[0], 0.0, worst)


# -- the score tiers ---------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_tier_table(self):
        self.assertEqual(S.tier("", True, True, True, True, True, True, True), 1.0)
        self.assertEqual(S.tier("", True, True, True, False, True, True, True), 0.75)
        self.assertEqual(S.tier("", True, True, True, True, False, True, True), 0.75)
        self.assertEqual(S.tier("", True, True, True, True, True, False, True), 0.75)
        self.assertEqual(S.tier("", True, False, True, True, True, True, True), 0.5)
        self.assertEqual(S.tier("", True, True, False, True, True, True, True), 0.5)
        self.assertEqual(S.tier("", False, False, False, False, True, True, True), 0.25)
        self.assertEqual(S.tier("", False, False, True, False, True, True, False), 0.0)
        self.assertEqual(S.tier("moved", True, True, True, True, True, True, True), 0.0)

    def test_doing_nothing_scores_zero(self):
        tr = S.Tracker()
        res = run(tr, states(), 200)
        self.assertEqual(res.score, 0.0)

    def test_full(self):
        res, _ = full_trial()
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_left_on_is_three_quarters(self):
        res, _ = full_trial(off=False)
        self.assertEqual(res.score, 0.75)

    def test_arm_not_back_is_half(self):
        res, _ = full_trial(arm_back=False)
        self.assertEqual(res.score, 0.5)

    def test_off_before_shown_is_half(self):
        res, _ = full_trial(shown=False)
        self.assertEqual(res.score, 0.5)

    def test_closed_without_a_reading_is_a_quarter(self):
        tr = S.Tracker()
        press(tr)
        run(tr, states(), int(L.READY_S * HZ) + 2)
        rest = P.support_angle(P.thermo_state(PLACED), clear=OM.REST_GAP)
        res = run(tr, states(rest, PLACED), 40)
        self.assertEqual(res.score, 0.25)
        res = run(tr, states(0.0, SHOWN_AT), 40)
        self.assertEqual(res.score, 0.25)

    def test_tip_in_with_the_arm_up_is_not_closed(self):
        tr = S.Tracker()
        press(tr)
        run(tr, states(), int(L.READY_S * HZ) + 2)
        res = run(tr, states(math.radians(40.0), PLACED), int(L.PEAK_S * HZ) + 4)
        self.assertFalse(res.details["closed_ever"])
        self.assertEqual(res.score, 0.0)

    def test_no_reading_without_contact(self):
        # Run 2 of the review gate: no credit for a tip near the skin. The tip 1.5 mm short of the film (or the arm
        # 1.5 mm over the stick) for the whole count is never Closed; a 1 s lapse in a 0.4 mm contact restarts it.
        short = (PLACED[0] - 0.0015, PLACED[1], PLACED[2])
        rest_short = P.support_angle(P.thermo_state(short), clear=OM.REST_GAP)
        rest = P.support_angle(P.thermo_state(PLACED), clear=OM.REST_GAP)
        hover = P.support_angle(P.thermo_state(PLACED), clear=0.0015)
        for theta, th in ((rest_short, short), (hover, PLACED)):
            tr = S.Tracker()
            press(tr)
            run(tr, states(), int(L.READY_S * HZ) + 2)
            res = run(tr, states(theta, th), int(L.PEAK_S * HZ) + 4)
            self.assertFalse(res.details["closed_ever"], res.details)
            self.assertEqual(res.score, 0.0)
        near = (PLACED[0] - 0.0004, PLACED[1], PLACED[2])
        tr = S.Tracker()
        press(tr)
        run(tr, states(), int(L.READY_S * HZ) + 2)
        run(tr, states(rest, near), int(8.0 * HZ))
        run(tr, states(rest_short, short), int(1.0 * HZ))
        res = run(tr, states(rest, near), int(8.0 * HZ))
        self.assertTrue(res.details["closed"])
        self.assertFalse(res.details["reading"])
        res = run(tr, states(rest, near), int(3.5 * HZ))
        self.assertTrue(res.details["reading"])

    def test_not_switched_on_scores_zero(self):
        tr = S.Tracker()
        rest = P.support_angle(P.thermo_state(PLACED), clear=OM.REST_GAP)
        res = run(tr, states(rest, PLACED), int(L.PEAK_S * HZ) + 40)
        self.assertEqual(res.score, 0.0)

    def test_shown_needs_the_hands_off_and_display_up(self):
        tr = S.Tracker()
        reading(tr)
        near = dict(AWAY)
        top = np.array([SHOWN_AT[0] + 0.045, SHOWN_AT[1], L.TOP + L.MAT_H + 0.030])
        near["right"] = rigstate.Arm(top + (0.0, 0.0, 0.013), top, 0.5)
        res = run(tr, states(0.0, SHOWN_AT), int(L.SHOWN_S * HZ) + 10, near)
        self.assertFalse(res.details["shown"])
        flipped = {P.DOLL: P.doll_state(0.0), P.MAT: P.MAT_SPEC.rest_state(),
                   P.THERMO: mk.state_from_poses(P.THERMO_SPEC, (*SHOWN_AT[:2], SHOWN_AT[2] + 0.013, 1.0, 0.0, 0.0,
                                                                 0.0), joints={"button": 0.0, "display": 0.0})}
        res = run(tr, flipped, int(L.SHOWN_S * HZ) + 10)
        self.assertFalse(res.details["shown"])
        res = run(tr, states(0.0, SHOWN_AT), int(L.SHOWN_S * HZ) + 2)
        self.assertTrue(res.details["shown"])

    def test_off_by_itself_is_not_off(self):
        tr = S.Tracker()
        reading(tr)
        run(tr, states(0.0, SHOWN_AT), int(L.SHOWN_S * HZ) + 4)
        res = run(tr, states(0.0, SHOWN_AT), int(L.AUTO_OFF_S * HZ) + 10)
        self.assertFalse(tr.unit.on)
        self.assertEqual(res.score, 0.75)

    def test_repositioning_the_doll_on_the_mat_is_not_zero(self):
        # Run 2 of the review gate: a benign shift on the mat must not zero a good run. Slid 25 mm and turned 12 deg
        # on the mat, then back: still 1.0.
        res, tr = full_trial()
        self.assertEqual(res.score, 1.0)
        for pose in ((P.DOLL_ORIGIN[0] + 0.025, P.DOLL_ORIGIN[1], P.DOLL_ORIGIN[2], 0.0, 0.0, 0.0, 1.0),
                     (*P.DOLL_ORIGIN, *mk.yaw_quat(math.radians(12.0)))):
            run(tr, states(0.0, SHOWN_AT, doll_pose=pose), 2)
        res = run(tr, states(0.0, SHOWN_AT), WIN + 4)
        self.assertEqual(res.details["zero"], "")
        self.assertEqual(res.score, 1.0)

    def test_unsafe_handling_latches_zero(self):
        a = math.radians(50.0)
        cases = {"lifted": (P.DOLL_ORIGIN[0], P.DOLL_ORIGIN[1], P.DOLL_ORIGIN[2] + 0.035, 0.0, 0.0, 0.0, 1.0),
                 "rolled": (*P.DOLL_ORIGIN, 0.0, math.sin(a / 2), 0.0, math.cos(a / 2)),
                 "off the mat": (0.560, P.DOLL_ORIGIN[1], P.DOLL_ORIGIN[2], 0.0, 0.0, 0.0, 1.0)}
        for why, pose in cases.items():
            res, tr = full_trial()
            self.assertEqual(res.score, 1.0)
            run(tr, states(0.0, SHOWN_AT, doll_pose=pose), 2)
            res = run(tr, states(0.0, SHOWN_AT), WIN + 4)
            self.assertEqual(res.score, 0.0, why)
            self.assertIn(why.split()[0], res.details["zero"], why)
        # within the limits: lifted 25 mm, rolled 40 deg
        b = math.radians(40.0)
        for pose in ((P.DOLL_ORIGIN[0], P.DOLL_ORIGIN[1], P.DOLL_ORIGIN[2] + 0.025, 0.0, 0.0, 0.0, 1.0),
                     (*P.DOLL_ORIGIN, 0.0, math.sin(b / 2), 0.0, math.cos(b / 2))):
            tr = S.Tracker()
            run(tr, states(doll_pose=pose), 2)
            self.assertEqual(run(tr, states(), 2).details["zero"], "")

    def test_thermometer_off_the_mat_latches_zero(self):
        tr = S.Tracker()
        reading(tr)
        run(tr, states(th=(0.040, -0.140, L.TOP)), 2)  # on the table beside the mat
        res = run(tr, states(0.0, SHOWN_AT), int(L.SHOWN_S * HZ) + WIN + 10)
        self.assertEqual(res.score, 0.0)


class FailClosed(unittest.TestCase):
    def test_unreadable_parts(self):
        tr = S.Tracker()
        res = tr.update({P.DOLL: None, P.THERMO: None, P.MAT: None}, AWAY)
        self.assertEqual(res.score, 0.0)
        self.assertIn("unreadable", res.details)

    def test_unreadable_arms_are_never_hands_off(self):
        res, tr = full_trial()
        res = run(tr, states(0.0, SHOWN_AT), WIN + 2, arms=None)
        self.assertFalse(res.details["hands_off"])
        self.assertLess(res.score, 1.0)


class Resets(unittest.TestCase):
    def test_teleport_back_to_the_start_is_a_new_trial(self):
        tr = S.Tracker()
        reading(tr)
        run(tr, states(math.radians(30.0), (0.200, -0.100, L.TOP + L.MAT_H)), 2)
        res = run(tr, states(), 2)
        self.assertEqual(tr.resets, 1)
        self.assertFalse(res.details["reading"])

    def test_a_natural_state_is_not_a_reset(self):
        res, tr = full_trial()
        self.assertEqual(tr.resets, 0)


# -- the demo ----------------------------------------------------------------------------------------------------------
class Demo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full = replay()
        cls.left_idle = replay(idle="left")
        cls.right_idle = replay(idle="right")

    def test_demo_scores_one(self):
        score, at, model, tr = self.full
        self.assertEqual(score, 1.0)
        self.assertIsNotNone(at)
        self.assertLess(at, 3200 * 0.8)  # inside the run limit with room to spare
        self.assertEqual(tr.unit.readings, 1)
        self.assertEqual([e[1] for e in tr.unit.events], ["on", "off"])
        self.assertFalse(tr.unit.setting_opened)
        self.assertEqual(tr.resets, 0)

    def test_open_fingers_never_sweep_the_thermometer(self):
        """While no gripper carries it, an open finger pad (27 mm wide, 8 mm thick, 38 mm tall from its tip) never
        cuts into the thermometer's body (GPU 19 Nov: the right's first way out swept its tail and turned it)."""
        pts = P.TH_POINTS[::2]
        open_level = abs(T.OPEN / rigstate.FINGER_OPEN) - 0.01
        hits = []

        def watch(t, arms, yaws, model):
            if model.carry is not None:
                return
            body = model.thermo().to_world(pts)
            for side, a in arms.items():
                if side not in yaws or a.opening < open_level:
                    continue
                yaw = yaws[side]
                close = np.array([-math.sin(yaw), math.cos(yaw)])
                width = np.array([math.cos(yaw), math.sin(yaw)])
                rel = body[:, :2] - a.tcp[:2]
                along, across = rel @ close, rel @ width
                half_gap = a.opening * rigstate.FULL_GAP / 2
                inside = ((np.abs(along) >= half_gap - 0.001) & (np.abs(along) <= half_gap + rigstate.FINGER_THICK
                                                                   + 0.001)
                          & (np.abs(across) <= rigstate.FINGER_HALF_WIDTH + 0.001)
                          & (body[:, 2] >= a.tips[2] - 0.001) & (body[:, 2] <= a.tips[2] + rigstate.FINGER_LEN))
                if inside.any():
                    hits.append((round(t, 2), side))

        replay(on_step=watch)
        self.assertEqual(hits, [], hits[:5])

    def test_one_arm(self):
        self.assertEqual(self.left_idle[0], 0.0)
        self.assertEqual(self.right_idle[0], 0.0)

    def test_without_the_off_press(self):
        score, _at, _m, tr = replay(skip=("fingers closed (off)", "release: up off"))
        self.assertEqual(score, 0.75)
        self.assertTrue(tr.unit.on)

    def test_without_sliding_it_in(self):
        score, _at, _m, tr = replay(skip=("slide it in under the arm", "hold it there"))
        self.assertEqual(score, 0.0)
        self.assertEqual(tr.unit.readings, 0)

    def test_without_switching_on(self):
        score, _at, _m, tr = replay(skip=("press the button (on)", "release: up off"))
        self.assertEqual(score, 0.0)

    def test_demo_env(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(list(T.demo()), ["right"])
        finally:
            del os.environ["DEMO_IDLE_ARM"]


if __name__ == "__main__":
    unittest.main()
