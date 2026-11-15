"""Offline tests for layout.py, scoring.py, offline_stay.py and task.py: synthetic window states (mechkit
AssemblyStates from the task's own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v"""

import hashlib
import json
import math
import os
import pathlib
import re
import subprocess
import sys
import tempfile
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
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline stand-in

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import offline_stay as OS  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402
from yam_arena import demo as D  # noqa: E402

try:
    import yamkin as K  # noqa: E402
except ImportError:  # pragma: no cover
    K = None

SW = T.WINDOW_SPEC
WIN = int(round(S.STILL_WINDOW_S * L.CONTROL_HZ))
BASE_POSE = (*L.BASE_ORIGIN, *mk.yaw_quat(L.BASE_YAW))
AWAY = {a: rigstate.Arm(np.array([0.10, s * 0.305, L.TOP + 0.30]), np.array([0.10, s * 0.305, L.TOP + 0.287]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
META = json.loads((HERE / "task.json").read_text())
README = (HERE / "README.md").read_text()
RUN_LIMIT_STEPS = 800


def states(q=(0.0, 0.0, 0.0)):
    sw = mk.state_from_poses(SW, BASE_POSE, joints={S.SASH: q[0], S.SWIVEL: q[1], S.STAY: q[2]}, source="offline")
    return {S.WINDOW: sw}


def seated(k):
    """Joints with hole k (1..4) on pin A, seated."""
    return L.q_for(L.HOLE_S[k - 1], L.WEB_BOT_H)


def feed(tracker, st, arms, n):
    out = None
    for _ in range(n):
        out = tracker.update(st, arms)
    return out


def holding(q):
    """The right gripper on the ball (pads 14 mm apart), the left away."""
    d = L.disc_world(q)
    tips = d - np.array([0.0, 0.0, 0.004])
    return {"left": AWAY["left"], "right": rigstate.Arm(tips + np.array([0.0, 0.0, 0.013]), tips, 0.15)}


def surface(state, bodies, n=10):
    """Points (world) on the surface of every shape of ``bodies`` of an assembly state."""
    spec, out = state.spec, []
    u = np.linspace(-0.5, 0.5, n)
    for body in bodies:
        r, p = state.link_pose(body)
        off = spec.body_offset(body)
        for s in spec.body_shapes(body):
            if s.kind == "box":
                size = np.asarray(s.size, dtype=float)
                pts = []
                for k in range(3):
                    a, b = [i for i in range(3) if i != k]
                    uu, vv = np.meshgrid(u, u)
                    for sign in (-0.5, 0.5):
                        q = np.zeros((uu.size, 3))
                        q[:, a], q[:, b], q[:, k] = uu.ravel() * size[a], vv.ravel() * size[b], sign * size[k]
                        pts.append(q)
                local = np.vstack(pts)
            else:
                k = {"X": 0, "Y": 1, "Z": 2}[s.axis]
                a, b = [i for i in range(3) if i != k]
                ang = np.linspace(0, 2 * math.pi, 3 * n, endpoint=False)
                local = []
                for h in np.linspace(-0.5, 0.5, n):
                    q = np.zeros((ang.size, 3))
                    q[:, a], q[:, b], q[:, k] = s.radius * np.cos(ang), s.radius * np.sin(ang), h * s.height
                    local.append(q)
                local = np.vstack(local)
            base = (local @ s.rotation().T) + np.asarray(s.center, dtype=float) - off
            out.append(base @ r.T + p)
    return np.vstack(out)


def pad_points(tcp, tips, gap, yaw, n=6):
    """Points filling the two finger pads (8 mm thick, 27 mm wide, 38 mm long from the tips) of a straight-down
    gripper whose pads close along (-sin yaw, cos yaw) with ``gap`` between them."""
    c = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    a = np.asarray(tips, dtype=float) - np.asarray(tcp, dtype=float)
    a = a / np.linalg.norm(a)
    w = np.cross(a, c)
    out = []
    for side in (1.0, -1.0):
        for t in np.linspace(gap / 2, gap / 2 + 0.008, 3):
            for v in np.linspace(-0.0135, 0.0135, n):
                for h in np.linspace(0.0, 0.038, n):
                    out.append(np.asarray(tips) + side * t * c + v * w - h * a)
    return np.array(out)


class Geometry(unittest.TestCase):
    def test_kit_files_are_unchanged_copies(self):
        d = HERE.parents[1] / "tools" / "kit"
        if not (d / "mechkit.py").is_file():
            self.skipTest("the repository's kit folder is not next to this task")
        for name in ("mechkit.py", "rigstate.py"):
            self.assertEqual(hashlib.sha256((HERE / name).read_bytes()).hexdigest(),
                             hashlib.sha256((d / name).read_bytes()).hexdigest(), name)

    def test_figures_from_the_makers_cad(self):
        mm = lambda v: round(v * 1000, 2)  # noqa: E731
        self.assertEqual([mm(s) for s in L.HOLE_S], [64.52, 105.02, 145.52, 186.02])
        self.assertEqual({mm(b - a) for a, b in zip(L.HOLE_S, L.HOLE_S[1:])}, {40.5})
        self.assertAlmostEqual(mm(L.SWIVEL_OFF + L.TAIL_END), 278.61, places=1)  # 278.64 overall on the drawing
        self.assertEqual((mm(2 * L.HOLE_R), mm(L.PIN_TOP_H), mm(2 * L.BALL_R), mm(L.S_LIFT)), (7.2, 22.5, 14.0, 20.02))
        self.assertEqual((L.PIN_PLATE[0], L.PIN_PLATE[1]), (0.015, 0.040))
        self.assertAlmostEqual(L.pin_r(L.PIN_SEAT_H), L.HOLE_R, places=5)  # the hole sits where it meets the taper
        self.assertAlmostEqual(mm(2 * L.CONE_R0), 7.9, places=5)
        self.assertGreater(L.GROOVE_W / 2, L.pin_r(L.LEG_BOT_H) + 0.0001)  # the legs straddle the seated pin
        self.assertGreater(L.LEG_BOT_H, L.COLLAR_H)  # ... above its collar

    def test_start_is_on_the_second_hole_and_stays_open_wider_on_later_holes(self):
        self.assertAlmostEqual(math.degrees(L.THETA_START), 13.46, places=1)
        self.assertEqual(round(math.degrees(L.theta_for(L.HOLE_S[2])), 1), 22.4)
        self.assertEqual(round(math.degrees(L.theta_for(L.HOLE_S[3])), 1), 30.8)
        r = S.read(states()[S.WINDOW])
        self.assertEqual(S.hole_on_pin(r, "A"), 2)
        self.assertIsNone(S.hole_on_pin(r, "B"))
        shut = L.q_for(L.HOLE_S[0], L.WEB_BOT_H)
        self.assertAlmostEqual(shut[0], -L.THETA_START, places=6)
        r = S.read(states(shut)[S.WINDOW])
        self.assertEqual((S.hole_on_pin(r, "A"), S.hole_on_pin(r, "B")), (1, 4))  # shut: both pins in

    def test_kinematics_match_the_kit(self):
        rng = np.random.default_rng(3)
        rows = ((0.1, 0.01, 0.0), (0.26, 0.02, 0.005), (0.03, 0.0, -0.006))
        pts = np.array([L.bar_point(s, h, n) for s, h, n in rows])
        for _ in range(25):
            q = (rng.uniform(L.SASH_LOWER, L.SASH_UPPER), rng.uniform(-0.5, 0.8), rng.uniform(L.LIFT_LOWER, 0.4))
            sw = states(q)[S.WINDOW]
            self.assertTrue(np.allclose(sw.link_to_world(S.STAY, pts), L.world_of_bar(q, pts), atol=1e-9))
            s, n, h = L.pin_in_bar(q, "A")
            back = L.world_of_bar(q, L.bar_point(s, L.WEB_BOT_H, n))[0]
            self.assertTrue(np.allclose(back[:2], L.PINS["A"], atol=1e-9))
        q = L.q_for(0.15, L.PIN_TOP_H + 0.003)
        s, n, h = L.pin_in_bar(q, "A")
        self.assertAlmostEqual(n, 0.0, places=9)
        self.assertAlmostEqual(h, L.PIN_TOP_H + 0.003, places=7)

    def test_joints_and_objects(self):
        self.assertEqual([j.name for j in SW.joints], ["sash", "swivel", "stay"])
        self.assertEqual([j.axis for j in SW.joints], ["Z", "Z", "X"])
        self.assertEqual([j.parent for j in SW.joints], ["base", "sash", "swivel"])
        self.assertIsNone(SW.mass)
        self.assertIsNone(T.BOARD_SPEC.mass)
        self.assertEqual([o.name for o in T.OBJECTS], ["board", "window"])
        self.assertAlmostEqual(T.SASH_MASS, 3.1, delta=0.1)
        self.assertEqual(SW.contact_offset, mk.TIGHT_FIT["contact_offset"])
        self.assertEqual(T.BOARD_SPEC.contact_offset, mk.TIGHT_FIT["contact_offset"])

    def test_start_pose_is_free_of_overlap(self):
        sw = states()[S.WINDOW]
        board = T.BOARD_SPEC.rest_state()
        bar = surface(sw, (S.STAY, S.SWIVEL), n=8)
        self.assertGreater(float(board.distance(bar).min()), 1e-6)  # nothing of the stay touches a pin or the board
        gap = board.distance(surface(sw, (S.SASH,), n=8))
        self.assertGreater(float(gap.min()), 0.0025)  # the sash clears the board and the pins
        # seated: the hole's ring sits on the taper within 0.2 mm
        r = S.read(sw)
        self.assertAlmostEqual(r.holes[1, 2] - L.Z0, L.PIN_SEAT_H + L.SEAT_GAP, places=6)

    def test_sash_swings_clear_of_the_board_and_pins_at_every_setting(self):
        board = T.BOARD_SPEC.rest_state()
        for k in (1, 2, 3, 4):
            sw = states(seated(k))[S.WINDOW]
            self.assertGreater(float(board.distance(surface(sw, (S.SASH,), n=6)).min()), 0.0025, k)


class Scoring(unittest.TestCase):
    def test_start_scores_zero(self):
        res = feed(S.Tracker(), states(), AWAY, WIN + 2)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["hole"], 2)

    def test_last_hole_let_go_and_still_scores_one(self):
        t = S.Tracker()
        res = feed(t, states(seated(4)), AWAY, WIN)
        self.assertEqual(res.score, 0.25)  # not yet 20 steps
        res = feed(t, states(seated(4)), AWAY, 1)
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_last_hole_still_held_scores_a_quarter(self):
        q = seated(4)
        res = feed(S.Tracker(), states(q), holding(q), WIN + 2)
        self.assertEqual((res.score, res.details["released"]), (0.25, False))

    def test_middle_hole_let_go_scores_a_half_and_held_a_quarter(self):
        q = seated(3)
        self.assertEqual(feed(S.Tracker(), states(q), AWAY, WIN + 2).score, 0.5)
        self.assertEqual(feed(S.Tracker(), states(q), holding(q), WIN + 2).score, 0.25)

    def test_resting_on_the_pin_top_or_off_it_scores_zero(self):
        for rho in (L.HOLE_S[3] - 0.012, L.HOLE_S[3] + 0.015, L.HOLE_S[2] - 0.020):
            q = L.q_for(rho, L.PIN_TOP_H)
            res = feed(S.Tracker(), states(q), AWAY, WIN + 2)
            self.assertEqual((res.score, res.details["hole"]), (0.0, None), rho)

    def test_hole_over_the_pin_but_lifted_is_not_on_it(self):
        q = L.q_for(L.HOLE_S[3], L.PIN_TOP_H - 0.004)  # pin tip 0.25 mm below the bar's top: not through
        self.assertIsNone(S.hole_on_pin(S.read(states(q)[S.WINDOW]), "A"))
        q = L.q_for(L.HOLE_S[3], L.PIN_TOP_H - 0.0045)  # 0.25 mm proud
        self.assertEqual(S.hole_on_pin(S.read(states(q)[S.WINDOW]), "A"), 4)

    def test_hole_off_centre_by_more_than_two_millimetres_is_not_on(self):
        q = L.q_for(L.HOLE_S[3] + 0.0025, L.WEB_BOT_H)
        self.assertIsNone(S.hole_on_pin(S.read(states(q)[S.WINDOW]), "A"))
        q = L.q_for(L.HOLE_S[3] + 0.0015, L.WEB_BOT_H)
        self.assertEqual(S.hole_on_pin(S.read(states(q)[S.WINDOW]), "A"), 4)

    def test_moving_is_not_still(self):
        t = S.Tracker()
        q = list(seated(4))
        res = None
        for k in range(WIN + 5):
            q[0] = seated(4)[0] + (0.0003 if k % 2 else 0.0)  # the sash's free corner shakes 0.1 mm
            q[0] += 0.004 * k / WIN  # and creeps 1.6 mm a second
            res = t.update(states(tuple(q)), AWAY)
        self.assertEqual(res.score, 0.25)
        self.assertFalse(res.details["still_20"])

    def test_glass_latch(self):
        t = S.Tracker()
        c, size = L.PANE_BOX_SHUT
        centre = L.HINGE_XY + L.rz(L.THETA_START) @ (c[:2] - L.HINGE_XY)
        p = np.array([centre[0], centre[1], c[2]])
        poke = {"left": AWAY["left"], "right": rigstate.Arm(p + np.array([0, 0, 0.013]), p, 0.0)}
        self.assertEqual(t.update(states(), poke).details["glass"], True)
        res = feed(t, states(seated(4)), AWAY, WIN + 5)
        self.assertEqual(res.score, 0.0)  # latched
        t2 = S.Tracker()
        res = feed(t2, states(seated(4)), None, 3)  # unreadable arms never latch, never release
        self.assertFalse(res.details["glass"])
        self.assertFalse(res.details["released"])

    def test_unreadable_state_starts_the_window_again(self):
        t = S.Tracker()
        feed(t, states(seated(4)), AWAY, WIN + 1)
        res = t.update({S.WINDOW: None}, AWAY)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(feed(t, states(seated(4)), AWAY, WIN).score, 0.25)
        self.assertEqual(feed(t, states(seated(4)), AWAY, 1).score, 1.0)

    def test_reset_detection(self):
        t = S.Tracker()
        feed(t, states(seated(4)), AWAY, 30)
        t.update(states(), AWAY)
        self.assertEqual((t.resets, t.steps), (1, 1))


class Offline(unittest.TestCase):
    def test_floor_kinds(self):
        self.assertEqual(OS.floor_for(L.HOLE_S[2], 0.0)[0], "hole")
        self.assertEqual(OS.floor_for(L.HOLE_S[2] + 0.02, 0.0)[0], "top")
        self.assertEqual(OS.floor_for(L.HOLE_S[2] + 0.02, 0.003)[0], "leg")
        self.assertEqual(OS.floor_for(L.HOLE_S[2] + 0.02, 0.012)[0], "none")

    def test_free_bar_falls_into_a_hole_or_onto_the_pin(self):
        o = OS.OfflineStay()
        o.captured = None
        o.q = L.q_for(L.HOLE_S[2], L.PIN_TOP_H + 0.003)
        o._fall()
        self.assertEqual(o.captured, 2)
        self.assertAlmostEqual(L.pin_in_bar(o.q, "A")[2], L.PIN_SEAT_H, places=6)
        o = OS.OfflineStay()
        o.captured = None
        o.q = L.q_for(L.HOLE_S[2] + 0.015, L.PIN_TOP_H + 0.003)
        o._fall()
        self.assertIsNone(o.captured)
        self.assertAlmostEqual(L.pin_in_bar(o.q, "A")[2], L.PIN_TOP_H, places=6)


def _run(variant, idle=""):
    plan = D.ArmPlan(T.plans(variant)["right"], K.BASES["station"]["right"])
    off = OS.OfflineStay()
    rows = []
    for k in range(int((plan.duration + 1.0) * L.CONTROL_HZ)):
        t = (k + 1) / L.CONTROL_HZ
        pos, rot, finger, label = plan.target(t)
        arm = rigstate.Arm(pos, pos + K.PAD_TIP_PAST_TCP * rot[:, 2], float(np.clip(finger / K.FINGER_OPEN, 0, 1)))
        if idle == "right":
            arm = AWAY["right"]
        off.step({"left": AWAY["left"], "right": arm})
        rows.append((t, label, pos, arm, off.q, off.captured, off.holder))
    return plan, rows


class Demo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if K is None:
            raise unittest.SkipTest("tools/offline is not next to this task")

    def test_every_waypoint_solves_for_the_right_arm(self):
        base = K.BASES["station"]["right"]
        for variant in ("", *T.VARIANTS):
            wps = T.plans(variant)["right"]
            plan = D.ArmPlan(wps, base)
            for k in range(int(plan.duration * L.CONTROL_HZ) + 1):
                act, _ = plan.action7(k / L.CONTROL_HZ)
                self.assertGreaterEqual(float(K.joint_margins(act[:6]).min()), 0.0, (variant, k))
            self.assertLess(plan.worst_ik_mm, 0.5, variant)
            self.assertLess(plan.worst_ik_deg, 0.5, variant)
            self.assertLessEqual(int(math.ceil((plan.duration + 3.0) * L.CONTROL_HZ)), RUN_LIMIT_STEPS, variant)
            self.assertEqual(set(T.plans(variant)), {"right"})  # one arm
            self.assertGreaterEqual(min(w.pos[2] for w in wps) - L.TOP, 0.060)  # the TCP stays 60 mm up or more

    def test_demo_has_the_six_moves_and_the_hand_follows_the_bar(self):
        wps = T.plans("")["right"]
        labels = [w.label for w in wps]
        for word in ("grip the ball", "lift", "carry", "lower the bar onto the pin", "push on", "down onto the seat",
                     "open"):
            self.assertTrue(any(word in x for x in labels), word)
        self.assertLessEqual(sum(1 for x in labels if x.startswith("grip")), 1)  # one grasp
        carry = [w.yaw for w in wps if "carry" in w.label]
        self.assertAlmostEqual(math.degrees(carry[-1] - wps[0].yaw), 23.6, delta=0.5)  # the hand turns with the bar
        self.assertTrue(all(b >= a for a, b in zip(carry, carry[1:])))

    def test_carry_keeps_the_bar_clear_and_the_slide_starts_on_solid_web(self):
        for rho in np.linspace(L.HOLE_S[1], T.RHO_LOWER, 9):
            q = T.q_at(rho, T.CLEAR)
            s, n, h = L.pin_in_bar(q, "A")
            self.assertAlmostEqual(h - L.PIN_TOP_H, T.CLEAR, places=6)
            self.assertGreater(h - (L.BAR_D - L.WEB_T) - L.PIN_TOP_H, 0.002)  # even a leg clears the pin by 2 mm
        s, n, h = L.pin_in_bar(T.q_at(T.RHO_LOWER, -T.PRESS), "A")
        self.assertGreater(s - L.TIP_R, L.HOLE_S[2] + L.HOLE_R + 0.020)
        self.assertLess(s + L.TIP_R, L.HOLE_S[3] - L.HOLE_R - 0.003)  # the pin lands on the web between holes 3 and 4

    def test_offline_playback_follows_the_design(self):
        expect = {"": (4, None), "slide_from_start": (3, None), "release_on_pin_top": (None, None),
                  "past_last": (None, None), "hold_at_last": (4, 1)}
        for variant, (hole, holder) in expect.items():
            plan, rows = _run(variant)
            t, label, pos, arm, q, captured, hold = rows[-1]
            self.assertEqual(None if captured is None else captured + 1, hole, variant)
            self.assertEqual(hold, holder, variant)
            st = states(q)
            r = S.read(st[S.WINDOW])
            self.assertEqual(S.hole_on_pin(r, "A"), hole, variant)
        _, rows = _run("", idle="right")
        self.assertEqual(rows[-1][5], 1)  # untouched: still on hole 2

    def test_no_pad_inside_the_board_pins_or_sash_and_clear_of_them_in_the_demo(self):
        board = T.BOARD_SPEC.rest_state()
        for variant in ("", *T.VARIANTS):
            plan, rows = _run(variant)
            worst_board = math.inf
            for t, label, pos, arm, q, captured, hold in rows[::2]:
                sw = states(q)[S.WINDOW]
                gap = max(float(arm.opening) * rigstate.FULL_GAP, 2 * L.BALL_R if hold is not None else 0.0)
                yaw = T.yaw_of(q) if hold is not None else None
                if yaw is None:
                    continue
                pts = pad_points(arm.tcp, arm.tips, gap, yaw)
                worst_board = min(worst_board, float(board.distance(pts).min()))
                self.assertGreater(float(sw.distance(pts, bodies=(S.SASH, "base")).min()), 0.05, (variant, label))
            self.assertGreater(worst_board, 0.015, variant)  # 15 mm of hand room over the board and from the pins

    def test_pads_straddle_the_ball_not_the_neck(self):
        q = (0.0, 0.0, 0.0)
        sw = states(q)[S.WINDOW]
        d = L.disc_world(q)
        tips = d - np.array([0, 0, 0.004])
        pts = pad_points(tips + np.array([0, 0, 0.013]), tips, 2 * L.BALL_R + 0.0005, T.yaw_of(q))
        self.assertGreater(float(sw.distance(pts, bodies=(S.STAY,)).min()), 0.0)

    def test_offline_checker_predicts_the_scores(self):
        if OFFLINE is None:
            self.skipTest("tools/offline is not next to this task")
        expected = {"": 1.0, "slide_from_start": 0.5, "release_on_pin_top": 0.0, "past_last": 0.0,
                    "hold_at_last": 0.25}
        for variant, want in expected.items():
            with tempfile.TemporaryDirectory() as tmp:
                out = pathlib.Path(tmp) / "report.json"
                env = dict(os.environ, DEMO_VARIANT=variant)
                done = subprocess.run([sys.executable, str(OFFLINE / "check_task.py"), str(HERE), "--json", str(out)],
                                      capture_output=True, text=True, env=env)
                self.assertEqual(done.returncode, 0, done.stdout[-2000:])
                report = json.loads(out.read_text())
            self.assertEqual(report["playback"]["proxy"]["predicted_task_score"], want, variant)
            self.assertEqual(report["playback"]["proxy"]["hook_errors"], {}, variant)
            if variant == "":
                self.assertIsNotNone(report["playback"]["proxy"]["success_held_at"])
                self.assertLessEqual(report["suggested_max_steps"], RUN_LIMIT_STEPS)

    def test_start_hands_are_far_and_end_clear(self):
        sw = states()[S.WINDOW]
        for side in ("left", "right"):
            tcp, rot = K.tcp_pose(K.READY, K.BASES["station"][side])
            arm = rigstate.Arm(tcp, tcp + K.PAD_TIP_PAST_TCP * rot[:, 2], 1.0)
            self.assertGreater(rigstate.zone_gap(arm, S.stay_distance(sw)), 0.200, side)
        plan, rows = _run("")
        t, label, pos, arm, q, captured, hold = rows[-1]
        self.assertGreater(rigstate.zone_gap(arm, S.stay_distance(states(q)[S.WINDOW])), 0.05)
        with self.assertRaises(ValueError):
            T.plans("nonsense")


class Text(unittest.TestCase):
    FIELDS = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
              "reset_notes": 580}

    def test_fields_and_limits(self):
        for k, n in self.FIELDS.items():
            self.assertLessEqual(len(META[k]), n, k)
        self.assertEqual(META["instruction"], T.INSTRUCTION)
        self.assertLess(len(META["instruction"]), 300)
        for word in ("lift", "carry", "slide", "push", "third", "over the", "groove", "past"):  # the goal only
            self.assertNotIn(word, META["instruction"].lower(), word)
        self.assertEqual(META["category"], "Societal relatability")
        self.assertEqual(META["run_limit"], f"{RUN_LIMIT_STEPS} steps (40 s at 20 Hz)")
        scores = META.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = META.get("media")
        self.assertTrue(media is None or re.fullmatch(r"M-[A-Z0-9]+", media), media)
        scoring = META["scoring"]
        for text in ("2.0 mm", "reaches the bar's top face", "30 mm", "20 steps", "1 mm or less", "1.0:", "0.5:",
                     "0.25:", "0:", "glass", "5 N"):
            self.assertIn(text, scoring, text)
        self.assertEqual((S.IN_HOLE_XY, S.PROUD, S.LET_GO, S.STILL_MOVE), (0.002, 0.0, 0.030, 0.001))
        setup = META["setup"]
        for text in ("Samuel Heath P4512-B", "7.2 mm", "64.5, 105.0, 145.5 and 186.0 mm", "13.5 degrees",
                     "30.8 degrees", "22.4 degrees", "3.1 kg", "Run limit 800 steps (40 s)", "x 40.9 cm"):
            self.assertIn(text, setup, text)
        self.assertIn("estimates", META["fidelity"])

    def test_files_are_clean(self):
        files = sorted(HERE.glob("*.py")) + sorted(HERE.glob("tests/*.py")) + [HERE / "task.json", HERE / "README.md"]
        team = re.compile("T" + "-" + r"\d{2}\b")
        email = re.compile(r"[\w.+-]+" + "@" + r"[\w-]+\.[\w.]+")
        for f in files:
            text = f.read_text()
            self.assertNotIn(chr(0x2014), text, f.name)  # no em dash
            self.assertNotIn(chr(0x2013), text, f.name)  # no en dash
            self.assertTrue(text.isascii(), f.name)
            if f.suffix in (".py", ".md"):  # task.json holds each field on one line
                for i, line in enumerate(text.splitlines(), 1):
                    self.assertLessEqual(len(line), 120, f"{f.name}:{i}")
            if f.name in ("task.json", "README.md"):
                self.assertIsNone(team.search(text), f.name)
                self.assertIsNone(email.search(text), f.name)
                for word in ("Claude", "Anthropic", "generated by", "Co-Authored"):
                    self.assertNotIn(word, text, f.name)

    def test_readme(self):
        flat = " ".join(README.split())
        self.assertLessEqual(len(README), 12000)
        for text in ("samuel-heath.com", "p4512-b.igs", "P4512-B", "## Sources", "## Results in simulation",
                     "DEMO_VARIANT=slide_from_start", "DEMO_VARIANT=release_on_pin_top", "DEMO_VARIANT=past_last",
                     "DEMO_VARIANT=hold_at_last", "DEMO_IDLE_ARM", "TASK_TRACE",
                     "python3 -m unittest discover -s tests",
                     "one-handed", "estimate"):
            self.assertIn(text, flat, text)
        for text in ("check_task", "tools/", "offline prediction"):
            self.assertNotIn(text, flat, text)


if __name__ == "__main__":
    unittest.main()
