"""Offline tests for scoring.py, offline_model.py and the hooks in task.py: synthetic motor, straightedge and arm
states (fed through rigstate.offline_feed where the hooks are tested), no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import offline_model  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
from scoring import Tracker  # noqa: E402

HZ = 20.0


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


WIN = L.STILL_STEPS + 1  # control steps that fill the 0.5 s window
FX = -L.FACE_LOCAL[0]  # 0.088: the face centre ahead of the motor's origin along its -x
PLATE_Z = L.TOP + L.PLATE_TOP
READY = {"left": np.array([0.3234, 0.305, L.TOP + 0.2606]), "right": np.array([0.3234, -0.305, L.TOP + 0.2606])}
_LOCAL_BOX = [np.array([x, y, z]) for x in (L.FACE_LOCAL[0], L.BODY_HALF[0]) for y in (-0.035, 0.035)
              for z in (0.0, L.MOTOR_BODY[2])]


def about(axis, deg):
    """Quaternion (xyzw) of a turn of ``deg`` degrees about a body axis ('x', 'y' or 'z')."""
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return np.array([*(math.sin(h) * np.array(v, float)), math.cos(h)])


def qmul(a, b):
    """Quaternion product a * b (xyzw): turn by b, then by a."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def motor_at(origin, quat):
    """An ObjectState-like motor (as mechkit gives it: with its frame's origin; the box round its shapes)."""
    o, q = np.asarray(origin, float), np.asarray(quat, float)
    r = S.rotation(q)
    pts = np.array([o + r @ p for p in _LOCAL_BOX])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(pos=(lo + hi) / 2, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=lo,
                           high=hi, origin=o)


def motor(e=0.0, theta=0.0, dy=0.0, dz=0.0, tilt=None):
    """The motor on its plate with its pulley face ``e`` (m) behind the line and turned ``theta`` degrees, its origin
    ``dy`` off its line and ``dz`` above the plate; ``tilt`` = (body axis, degrees) tips it about its origin."""
    t = math.radians(theta)
    fy = L.MOTOR_Y + dy - FX * math.sin(t)
    ox = L.PLANE_X + e + FX * math.cos(t)
    oy = fy + FX * math.sin(t)
    q = about("z", theta)
    if tilt is not None:
        q = qmul(q, about(*tilt))
    return motor_at((ox, oy, PLATE_Z + dz), q)


def start_motor():
    return motor_at((*L.MOTOR_START[:2], PLATE_Z), about("z", L.MOTOR_YAW0_DEG))


def bar(x=0.190, y=L.BAR_START[1], z=None, yaw=0.0, pitch=0.0):
    """The straightedge standing on its edge, centre at (x, y), its bottom on the table unless ``z`` (centre, above
    the table) says otherwise."""
    zc = L.TOP + (L.BAR_HALF[2] if z is None else z)
    q = qmul(about("z", yaw), about("y", pitch))
    r = S.rotation(q)
    c = np.array([x, y, zc])
    pts = np.array([c + r @ (np.array([sx, sy, sz]) * np.array(L.BAR_HALF))
                    for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=pts.min(axis=0),
                           high=pts.max(axis=0))


def start_bar():
    return bar(L.BAR_START[0], L.BAR_START[1], L.BAR_START[2])


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


def arms(left=None, right=None):
    return {"left": left or arm(READY["left"]), "right": right or arm(READY["right"])}


def settle(tr, m, b, a=None, n=WIN):
    """Feed the same states for ``n`` steps (0.55 s by default: the rest window fills)."""
    a = arms() if a is None else a
    res = None
    for _ in range(n):
        res = tr.update(m, b, a)
    return res


def score_of(m, b=None, a=None):
    tr = Tracker()
    return settle(tr, m, bar() if b is None else b, a)


# -- geometry -------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_start(self):
        m = start_motor()
        f = S.face_centre(m)
        self.assertAlmostEqual(f[0], 0.330 - 0.088 * math.cos(L.MOTOR_YAW0), places=9)
        self.assertAlmostEqual(S.offset(m) * 1000, 12.12, places=1)
        self.assertAlmostEqual(S.heading_deg(m), 3.0, places=9)
        self.assertAlmostEqual(S.misalignment_deg(m), 5.39, places=2)
        self.assertTrue(S.on_plate(m))
        self.assertAlmostEqual(S.y_offset(m), 0.0, places=9)
        self.assertLess(S.tilt_deg(m), 1e-6)
        res = score_of(m, start_bar())
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["failed_gates"], [])
        self.assertEqual(res.details["score_tier"], 0.0)

    def test_aligned(self):
        m = motor_at((L.ALIGNED_X, L.MOTOR_Y, PLATE_Z), about("z", 0.0))
        self.assertAlmostEqual(L.ALIGNED_X, 0.318)
        self.assertAlmostEqual(S.offset(m), 0.0, places=12)
        self.assertEqual(S.misalignment_deg(m), 0.0)
        self.assertEqual(score_of(m).score, 1.0)
        self.assertTrue(score_of(m).success)
        # the aligned body's corners sit inside the plate with room to spare (front edge 11 mm)
        self.assertAlmostEqual(S.plate_margin(m), 0.011, places=6)

    def test_offset_sign_and_alpha(self):
        self.assertAlmostEqual(S.offset(motor(e=0.004)), 0.004, places=12)
        self.assertAlmostEqual(S.offset(motor(e=-0.004)), -0.004, places=12)
        self.assertAlmostEqual(S.misalignment_deg(motor(e=0.004)), S.misalignment_deg(motor(e=-0.004)), places=12)
        self.assertAlmostEqual(S.alpha_deg(0.290 * math.tan(math.radians(0.5))), 0.5, places=12)

    def test_heading_is_signed_and_never_folded(self):
        self.assertAlmostEqual(S.heading_deg(motor(theta=0.4)), 0.4, places=9)
        self.assertAlmostEqual(S.heading_deg(motor(theta=-0.4)), -0.4, places=9)
        self.assertAlmostEqual(S.misalignment_deg(motor(theta=-0.4)), 0.4, places=9)
        turned = motor_at((0.230 - FX, L.MOTOR_Y, PLATE_Z), about("z", 180.0))  # the face on the line, but backwards
        self.assertAlmostEqual(S.offset(turned), 0.0, places=9)
        self.assertGreater(S.misalignment_deg(turned), 179.0)
        self.assertEqual(score_of(turned).score, 0.0)

    def test_tier_boundaries(self):
        e05 = L.CENTRE_DIST * math.tan(math.radians(0.5))  # 2.53 mm alone
        e10 = L.CENTRE_DIST * math.tan(math.radians(1.0))  # 5.06 mm
        e20 = L.CENTRE_DIST * math.tan(math.radians(2.0))  # 10.13 mm
        self.assertAlmostEqual(e05 * 1000, 2.531, places=3)
        self.assertAlmostEqual(e10 * 1000, 5.062, places=3)
        self.assertAlmostEqual(e20 * 1000, 10.127, places=3)
        eps = 1e-6
        for e, want in ((0.0, 1.0), (e05 - eps, 1.0), (e05 + eps, 0.5), (e10 - eps, 0.5), (e10 + eps, 0.25),
                        (e20 - eps, 0.25), (e20 + eps, 0.0), (0.030, 0.0)):
            for sign in (1, -1):
                self.assertEqual(score_of(motor(e=sign * e)).score, want, (sign * e, want))
        for th, want in ((0.499, 1.0), (0.501, 0.5), (0.999, 0.5), (1.001, 0.25), (1.999, 0.25), (2.001, 0.0)):
            for sign in (1, -1):
                self.assertEqual(score_of(motor(theta=sign * th)).score, want, (sign * th, want))
        # angle alone at 0.5 deg is 0.56 mm across the 64 mm pulley
        self.assertAlmostEqual(2 * L.MOTOR_PULLEY_R * math.tan(math.radians(0.5)) * 1000, 0.559, places=3)

    def test_tiers_function(self):
        self.assertEqual([S.tier(v) for v in (0.0, 0.5, 0.5001, 1.0, 1.0001, 2.0, 2.0001, 90.0)],
                         [1.0, 1.0, 0.5, 0.5, 0.25, 0.25, 0.0, 0.0])

    def test_turn_and_offset_add_up(self):
        # theta 0.3 deg and e 1 mm: 0.3 + 0.198 = 0.498 deg (1.0); theta 0.3 and e 1.1 mm: 0.517 (0.5)
        self.assertEqual(score_of(motor(e=0.001, theta=0.3)).score, 1.0)
        self.assertEqual(score_of(motor(e=0.0011, theta=0.3)).score, 0.5)
        # a turn the other way never cancels an offset: |theta| + alpha
        self.assertEqual(score_of(motor(e=0.002, theta=-0.4)).score, 0.5)
        self.assertEqual(score_of(motor(e=-0.002, theta=0.4)).score, 0.5)


# -- the gates --------------------------------------------------------------------------------------------------------
class Gates(unittest.TestCase):
    def test_at_rest_needs_the_window(self):
        tr = Tracker()
        m = motor()
        for i in range(WIN - 1):
            self.assertEqual(tr.update(m, bar(), arms()).score, 0.0, i)
            self.assertIn("at_rest", tr.last.details["failed_gates"])
        self.assertEqual(tr.update(m, bar(), arms()).score, 1.0)

    def test_rest_path_boundary(self):
        for total, want in ((0.0019, 1.0), (0.0021, 0.0)):
            tr = Tracker()
            step = total / L.STILL_STEPS
            for i in range(WIN):  # the face creeps sideways along the line (e stays 0): only the path counts
                res = tr.update(motor(dy=i * step - total / 2), bar(), arms())
            self.assertEqual(res.score, want, total)
            self.assertAlmostEqual(res.details["path_mm"], total * 1000, places=1)

    def test_rest_turn_boundary(self):
        for total, want in ((0.49, 1.0), (0.51, 0.0)):
            tr = Tracker()
            step = total / L.STILL_STEPS
            for i in range(WIN):  # rocking turns about the face centre (theta within +-0.25 deg)
                k = i % 2
                res = tr.update(motor(theta=(k - 0.5) * step), bar(), arms())
            self.assertEqual(res.score, want, total)

    def test_rest_is_judged_by_path_never_by_velocity(self):
        m = motor()
        m.lin_vel = np.array([0.045, -0.03, 0.02])  # the simulator's chatter on a part that stays put
        m.ang_vel = np.array([0.4, 0.3, 0.6])
        self.assertEqual(score_of(m).score, 1.0)

    def test_upright(self):
        for axis in ("x", "y"):
            self.assertEqual(score_of(motor(tilt=(axis, 1.9))).score, 1.0, axis)
            res = score_of(motor(tilt=(axis, 2.1)))
            self.assertEqual(res.score, 0.0, axis)
            self.assertIn("upright", res.details["failed_gates"])
        res = score_of(motor(tilt=("x", 180.0), dz=L.MOTOR_BODY[2]))  # upside down
        self.assertEqual(res.score, 0.0)

    def test_on_plate_outline(self):
        # the front edge of the body is 11 mm inside the plate when aligned; move the plate's limits instead by
        # shifting the motor along y, where the body has 30 mm either side
        side = L.PLATE_Y1 - (L.MOTOR_Y + L.BODY_HALF[1])
        self.assertAlmostEqual(side, 0.030)
        m_in, m_out = motor(dy=side - 0.0001), motor(dy=side + 0.0001)
        self.assertGreaterEqual(S.plate_margin(m_in), 0.0)
        self.assertLess(S.plate_margin(m_out), 0.0)
        # (these also fail the 10 mm line gate; on_plate is checked on its own)
        self.assertTrue(S.on_plate(m_in))
        self.assertFalse(S.on_plate(m_out))
        # the back corners past the plate's back edge: the motor shoved 60 mm back (and well behind the line)
        far = motor(e=L.PLATE_X1 - (L.ALIGNED_X + L.BODY_HALF[0]) + 0.0001)
        self.assertFalse(S.on_plate(far))
        self.assertIn("on_plate", score_of(far).details["failed_gates"])

    def test_on_plate_height(self):
        self.assertEqual(score_of(motor(dz=0.0019)).score, 1.0)
        res = score_of(motor(dz=0.0021))
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["failed_gates"], ["on_plate"])
        res = score_of(motor(dz=-0.0021))  # sunk into the plate (a blow-up)
        self.assertEqual(res.details["failed_gates"], ["on_plate"])

    def test_on_line(self):
        self.assertEqual(score_of(motor(dy=0.0099)).score, 1.0)
        self.assertEqual(score_of(motor(dy=-0.0099)).score, 1.0)
        for dy in (0.0101, -0.0101):
            res = score_of(motor(dy=dy))
            self.assertEqual(res.score, 0.0)
            self.assertEqual(res.details["failed_gates"], ["on_line"])

    def test_hands_off_each_point_each_object(self):
        m, b = motor(), bar()
        top = m.high[2]
        mc = (m.low + m.high) / 2
        self.assertAlmostEqual(L.HANDS_OFF, 0.060)
        for side in ("left", "right"):
            # the pad tips 60 mm above the motor's box: met; 59.9 mm: not (the TCP is 13 mm higher)
            near = arm((mc[0], mc[1], top + L.HANDS_OFF + 0.0001 + 0.013))
            self.assertEqual(score_of(m, b, arms(**{side: near})).score, 1.0, side)
            near = arm((mc[0], mc[1], top + L.HANDS_OFF - 0.0001 + 0.013))
            res = score_of(m, b, arms(**{side: near}))
            self.assertEqual(res.score, 0.0, side)
            self.assertEqual(res.details["failed_gates"], ["hands_off"])
            # beside the straightedge, level with it: 60 mm from its box met, 59.9 not
            bc = np.asarray(b.pos)
            for d, want in ((L.HANDS_OFF + 0.0001, 1.0), (L.HANDS_OFF - 0.0001, 0.0)):
                a = SimpleNamespace(tcp=np.array([b.low[0] - d, bc[1], bc[2]]),
                                    tips=np.array([b.low[0] - d, bc[1], bc[2] - 0.013]), opening=1.0)
                self.assertEqual(score_of(m, b, arms(**{side: a})).score, want, (side, d))
            # only the tips near: still not hands off
            a = SimpleNamespace(tcp=np.array([mc[0], mc[1], top + 0.2]), tips=np.array([mc[0], mc[1], top + 0.03]),
                                opening=1.0)
            self.assertEqual(score_of(m, b, arms(**{side: a})).score, 0.0, side)

    def test_arms_unavailable_count_as_hands_off(self):
        tr = Tracker()
        for _ in range(WIN):
            res = tr.update(motor(), bar(), None)
        self.assertEqual(res.score, 1.0)
        self.assertEqual(tr.arms_unavailable, WIN)
        self.assertFalse(res.details["arms_available"])
        bad = {"left": arm((math.nan, 0, 1)), "right": arm(READY["right"])}
        tr = Tracker()
        self.assertEqual(settle(tr, motor(), bar(), bad).score, 1.0)
        self.assertEqual(tr.arms_unavailable, WIN)
        self.assertEqual(settle(Tracker(), motor(), bar(), {"left": arm(READY["left"])}).score, 1.0)


# -- every listed score, reached by a trial -------------------------------------------------------------------------
class Reachable(unittest.TestCase):
    def trial(self, e, theta):
        """The motor slides from its start to (e, theta) over 2 s with a gripper on it, the gripper lets go and
        rises; then 1 s with nothing moving."""
        tr = Tracker()
        m0 = start_motor()
        r0, o0 = S.motor_pose(m0)
        m1 = motor(e=e, theta=theta)
        _, o1 = S.motor_pose(m1)
        th0 = L.MOTOR_YAW0_DEG
        b = start_bar()
        for i in range(40):
            u = (i + 1) / 40
            o = o0 + u * (o1 - o0)
            m = motor_at(o, about("z", th0 + u * (theta - th0)))
            hand = arm((o[0], o[1], L.TOP + 0.072), 0.66)
            res = tr.update(m, b, arms(left=hand))
            self.assertEqual(res.score, 0.0)  # held: never scores
        for i in range(20):
            hand = arm((o1[0], o1[1], L.TOP + 0.072 + 0.01 * i), 1.0)
            tr.update(m1, b, arms(left=hand))
        return settle(tr, m1, b, arms(), n=20)

    def test_each_tier_is_reachable(self):
        for e, theta, want in ((0.0, 0.0, 1.0), (-0.0015, 0.2, 1.0), (0.004, 0.0, 0.5), (0.0, -0.8, 0.5),
                               (-0.008, 0.0, 0.25), (0.003, 1.2, 0.25), (0.012, 0.0, 0.0), (0.0, 3.0, 0.0)):
            res = self.trial(e, theta)
            self.assertEqual(res.score, want, (e, theta))
        self.assertEqual(set(S.POSSIBLE_SCORES), {0.0, 0.25, 0.5, 1.0})

    def test_random_states_score_only_listed_values(self):
        rnd = random.Random(7)
        seen = set()
        for _ in range(400):
            m = motor(e=rnd.uniform(-0.015, 0.015), theta=rnd.uniform(-3, 3), dy=rnd.uniform(-0.015, 0.015),
                      dz=rnd.uniform(-0.003, 0.003),
                      tilt=(rnd.choice("xy"), rnd.uniform(-3, 3)) if rnd.random() < 0.3 else None)
            a = arms(left=arm((rnd.uniform(0.1, 0.4), rnd.uniform(-0.1, 0.3), L.TOP + rnd.uniform(0.0, 0.3))))
            res = score_of(m, bar(), a)
            self.assertIn(res.score, S.POSSIBLE_SCORES)
            self.assertEqual(res.success, res.score == 1.0)
            seen.add(res.score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})


# -- unintended strategies ------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_holding_the_motor_on_the_line(self):
        m = motor()
        o = S.motor_pose(m)[1]
        hand = arm((o[0], o[1], L.TOP + 0.072), 0.66)  # gripping the body, perfectly still and perfectly in line
        res = score_of(m, bar(), arms(left=hand))
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["failed_gates"], ["hands_off"])
        self.assertFalse(res.success)

    def test_holding_the_straightedge_against_the_motor(self):
        m = motor()
        b = bar(x=L.PLANE_X - L.BAR_HALF[0], y=L.BAR_START[1], z=0.026)  # pressed flat on both faces, still held
        grips = arms(left=arm((b.pos[0], 0.0, L.TOP + 0.029), 0.21), right=arm((b.pos[0], -0.25, L.TOP + 0.029), 0.21))
        res = score_of(m, b, grips)
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["failed_gates"], ["hands_off"])
        # let go and moved away, the same motor scores 1.0, whether the bar stays against the faces or not
        self.assertEqual(score_of(m, b, arms()).score, 1.0)

    def test_pushing_the_motor_past_the_line(self):
        # a bar held at one end pivots on the blower pulley's rim and shoves the motor on
        self.assertEqual(score_of(motor(e=0.004)).score, 0.5)
        self.assertEqual(score_of(motor(e=0.008)).score, 0.25)
        self.assertEqual(score_of(motor(e=0.025)).score, 0.0)

    def test_sliding_the_motor_toward_or_away_from_the_blower(self):
        # moving it along y changes the belt's centre distance: off its line, 0 however well its face lines up
        for dy in (-0.020, 0.020, -0.050):
            res = score_of(motor(dy=dy))
            self.assertEqual(res.score, 0.0, dy)
            self.assertIn("on_line", res.details["failed_gates"])

    def test_turning_the_motor_round(self):
        for th in (90.0, -90.0, 180.0, 45.0):
            self.assertEqual(score_of(motor(theta=th)).score, 0.0, th)

    def test_tipping_or_lifting_the_motor(self):
        on_side = motor(tilt=("x", 90.0), dz=0.035)
        self.assertEqual(score_of(on_side).score, 0.0)
        lifted = motor(dz=0.040)  # held up (or resting on the straightedge) with its face in line
        self.assertEqual(score_of(lifted).score, 0.0)
        nose_down = motor(tilt=("y", 3.0))  # rocked onto the pulley end
        self.assertEqual(score_of(nose_down).score, 0.0)

    def test_moving_the_motor_off_the_plate(self):
        off = motor_at((0.200, 0.140, L.TOP), about("z", 0.0))  # on the table in front of the plate
        res = score_of(off)
        self.assertEqual(res.score, 0.0)
        self.assertIn("on_plate", res.details["failed_gates"])

    def test_still_moving(self):
        tr = Tracker()
        for i in range(30):  # sliding toward the line at 5 mm/s, through it
            res = tr.update(motor(e=-0.003 + 0.00025 * i), bar(), arms())
        self.assertEqual(res.score, 0.0)
        self.assertIn("at_rest", res.details["failed_gates"])

    def test_open_fingers_resting_on_the_motor(self):
        # review exploit: fingers fully open, the TCP and pad tips 45 to 57 mm beside the motor's box (a finger's outer
        # face, 57 mm out from the TCP, against the body), level with it, for 0.55 s: not hands off
        m = motor()
        mc = (m.low + m.high) / 2
        self.assertGreater(L.HANDS_OFF, L.FULL_GAP / 2 + L.FINGER_THICK)  # an open finger reaches 57 mm out
        for d in (0.045, 0.050, 0.057):
            hand = arm((mc[0], m.high[1] + d, L.TOP + 0.060), 1.0)
            res = score_of(m, bar(), arms(left=hand))
            self.assertEqual(res.score, 0.0, d)
            self.assertEqual(res.details["failed_gates"], ["hands_off"], d)
        # just past the gate's 60 mm: met (and no open finger reaches the body from there)
        hand = arm((mc[0], m.high[1] + L.HANDS_OFF + 0.0001, L.TOP + 0.060), 1.0)
        self.assertEqual(score_of(m, bar(), arms(left=hand)).score, 1.0)

    def test_open_fingers_resting_on_the_straightedge(self):
        # review exploit: fingers fully open, the TCP 50 mm in front of the straightedge's box, level with it (an open
        # finger against it), for 0.55 s: not hands off, with either arm
        m, b = motor(), bar()
        for side in ("left", "right"):
            hand = arm((b.low[0] - 0.050, 0.0, L.TOP + 0.035), 1.0)
            res = score_of(m, b, arms(**{side: hand}))
            self.assertEqual(res.score, 0.0, side)
            self.assertEqual(res.details["failed_gates"], ["hands_off"], side)
            hand = arm((b.low[0] - L.HANDS_OFF - 0.0001, 0.0, L.TOP + 0.035), 1.0)
            self.assertEqual(score_of(m, b, arms(**{side: hand})).score, 1.0, side)

    def test_the_straightedge_alone_scores_nothing(self):
        # laying the straightedge on the line (or pushing it there) does nothing for an unmoved motor
        b = bar(x=L.PLANE_X - L.BAR_HALF[0], z=0.026)
        self.assertEqual(score_of(start_motor(), b).score, 0.0)


# -- robustness ------------------------------------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr = Tracker()
        settle(tr, motor(), bar(), arms())
        window = list(tr.window)
        prev = tr.prev
        good = motor()
        bads = []
        for key, val in (("pos", [math.nan, 0, 0]), ("quat_xyzw", [0, 0, 0, 0]), ("quat_xyzw", [math.inf, 0, 0, 1]),
                         ("origin", [0.3, math.nan, 0.8]), ("lin_vel", [math.inf, 0, 0]), ("low", [0, 0]),
                         ("high", None)):
            s = SimpleNamespace(**vars(good))
            setattr(s, key, val)
            bads.append(s)
        bads.append(SimpleNamespace(pos=good.pos))  # missing attributes
        bads.append(None)
        for s in bads:
            res = tr.update(s, bar(), arms())
            self.assertEqual(res.score, 0.0)
            self.assertFalse(res.success)
            self.assertFalse(res.details["valid_state"])
            res = tr.update(motor(), SimpleNamespace(pos=[0, 0, math.nan]), arms())
            self.assertEqual(res.score, 0.0)
        self.assertEqual(tr.bad_steps, 2 * len(bads))
        self.assertEqual(list(tr.window), window)
        self.assertIs(tr.prev, prev)
        self.assertEqual(tr.update(motor(), bar(), arms()).score, 1.0)  # the window survived

    def test_score_is_idempotent(self):
        tr = Tracker()
        settle(tr, motor(e=0.004), bar(), arms())
        before = (tr.steps, list(tr.window), tr.arms_unavailable)
        a = [tr.result(motor(e=0.004), bar(), arms()).score for _ in range(5)]
        self.assertEqual(a, [0.5] * 5)
        self.assertEqual(before, (tr.steps, list(tr.window), tr.arms_unavailable))

    def test_random_states_do_not_crash(self):
        rnd = np.random.default_rng(3)
        tr = Tracker()
        for _ in range(500):
            q = rnd.normal(size=4) * rnd.choice([1e-9, 1.0, 1e6])
            m = motor_at(rnd.uniform(-2, 2, 3), q)
            b = SimpleNamespace(pos=rnd.uniform(-2, 2, 3), quat_xyzw=rnd.normal(size=4), lin_vel=rnd.normal(size=3),
                                ang_vel=rnd.normal(size=3), low=rnd.uniform(-2, 0, 3), high=rnd.uniform(0, 2, 3))
            a = rnd.choice([None, 1, 2])
            a = None if a is None else arms(left=arm(rnd.uniform(-1, 1, 3), float(rnd.uniform(-1, 2))))
            res = tr.update(m, b, a)
            self.assertIn(res.score, S.POSSIBLE_SCORES)
            json.dumps(res.details)  # the trace row serialises

    def test_teleport_back_to_start_resets(self):
        tr = Tracker()
        settle(tr, motor(), bar(), arms(), n=30)
        self.assertEqual(tr.steps, 30)
        tr.update(start_motor(), start_bar(), arms())
        self.assertEqual(tr.steps, 1)  # a fresh trial: the window starts over
        self.assertEqual(len(tr.window), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        tr = Tracker()
        settle(tr, motor(), bar(), arms(), n=30)
        m = motor_at((L.MOTOR_START[0] + 0.02, L.MOTOR_START[1] - 0.02, PLATE_Z), about("z", 7.0))
        b = bar(L.BAR_START[0] - 0.025, L.BAR_START[1] + 0.03, L.BAR_START[2], yaw=4.0)
        tr.update(m, b, arms())
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr = Tracker()
        m1, b1 = motor(), bar()
        m0, b0 = start_motor(), start_bar()
        settle(tr, m1, b1, arms(), n=5)
        o1, o0 = S.motor_pose(m1)[1], S.motor_pose(m0)[1]
        for i in range(1, 21):  # 2 mm a step back to the start
            u = i / 20
            tr.update(motor_at(o1 + u * (o0 - o1), about("z", u * L.MOTOR_YAW0_DEG)),
                      bar(b1.pos[0] + u * (b0.pos[0] - b1.pos[0]), z=b1.pos[2] - L.TOP + u * (b0.pos[2] - b1.pos[2])),
                      arms())
        self.assertEqual(tr.steps, 25)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr = Tracker()
        settle(tr, motor(), bar(), arms(), n=5)
        tr.update(motor(), bar(x=0.30, y=0.0), arms())  # the bar flung 11 cm, nowhere near its start
        self.assertEqual(tr.steps, 6)
        tr.update(start_motor(), bar(x=0.30, y=0.0), arms())  # the motor back at its start, the bar not
        self.assertEqual(tr.steps, 7)

    def test_details_are_numbers(self):
        res = score_of(motor(e=0.001, theta=0.2))
        d = res.details
        self.assertAlmostEqual(d["e_mm"], 1.0)
        self.assertAlmostEqual(d["theta_deg"], 0.2)
        self.assertAlmostEqual(d["m_deg"], 0.2 + math.degrees(math.atan(0.001 / 0.290)), places=3)
        self.assertEqual(d["failed_gates"], [])
        self.assertTrue(d["hands_off"])


# -- the straightedge and the offline model -------------------------------------------------------------------------
class Straightedge(unittest.TestCase):
    def test_face_position(self):
        b = bar(x=0.220, z=0.026)
        self.assertAlmostEqual(S.bar_face_x(b, 0.14), 0.230, places=9)
        self.assertAlmostEqual(S.bar_face_x(b, L.BLOWER_Y), 0.230, places=9)
        self.assertIsNone(S.bar_face_x(b, 0.30))  # past its end
        turned = bar(x=0.220, z=0.026, yaw=180.0)  # either way round
        self.assertAlmostEqual(S.bar_face_x(turned, 0.0), 0.230, places=9)
        yawed = bar(x=0.220, y=0.0, z=0.026, yaw=2.0)  # its +y end swung back toward the arms
        self.assertAlmostEqual(S.bar_face_x(yawed, 0.14), 0.220 + 0.010 / math.cos(math.radians(2.0))
                               - 0.14 * math.tan(math.radians(2.0)), places=9)
        self.assertAlmostEqual(S.bar_yaw_deg(yawed), 2.0, places=9)
        self.assertAlmostEqual(S.bar_pitch_deg(bar(pitch=5.0)), 5.0, places=9)
        self.assertIsNone(S.bar_face_x(bar(yaw=90.0), 0.0))  # lying along x


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub) for mechkit")
class OfflineModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if str(_stub_dir()) not in sys.path:
            sys.path.insert(0, str(_stub_dir()))
        cls.spec = importlib.import_module("task").MOTOR

    def grips(self, x, ys=(0.0, -0.25), opening=12 / 93.9):
        held = {}
        for side, y in zip(("left", "right"), ys):
            held[side] = None if y is None else arm((x, y, L.TOP + 0.029), opening)
        return arms(**{k: v for k, v in held.items() if v is not None})

    def checker_motor(self, x=0.312, yaw_deg=0.0):
        """The motor as the offline checker reports it (its box, carried or dropped)."""
        return motor_at((x, L.MOTOR_Y, PLATE_Z), about("z", yaw_deg))

    def test_holds(self):
        b = bar(x=0.200, z=0.026)
        self.assertTrue(offline_model.holds(arm((0.200, 0.0, L.TOP + 0.029), 0.128), b))
        self.assertTrue(offline_model.holds(arm((0.200, 0.0, L.TOP + 0.029), 0.30), b))
        self.assertFalse(offline_model.holds(arm((0.200, 0.0, L.TOP + 0.029), 0.33), b))  # open over 30 mm
        self.assertFalse(offline_model.holds(arm((0.200, 0.0, L.TOP + 0.090), 0.128), b))  # tips above it
        self.assertFalse(offline_model.holds(arm((0.200, 0.215, L.TOP + 0.029), 0.128), b))  # past its end

    def test_push_modes(self):
        b = bar(x=0.200, z=0.026)
        self.assertEqual(offline_model.push_mode(self.grips(0.200), b), "both sides")
        self.assertEqual(offline_model.push_mode(self.grips(0.200, (0.0, None)), b), "motor side")
        self.assertEqual(offline_model.push_mode(self.grips(0.200, (None, -0.25)), b), "outboard")
        self.assertEqual(offline_model.push_mode(self.grips(0.200, (-0.15, None)), b), "motor side")  # over it
        self.assertIsNone(offline_model.push_mode(arms(), b))
        self.assertIsNone(offline_model.push_mode(None, b))

    def run_press(self, ys, end_x=0.255):
        model = offline_model.MotorModel(self.spec)
        cm = self.checker_motor()
        out = None
        for x in np.linspace(0.205, end_x, 60):  # the bar's face from 0.205 to end_x
            b = bar(x=x - L.BAR_HALF[0], z=0.026)
            out = model.update(cm, b, self.grips(x - L.BAR_HALF[0], ys))
        return model, out

    def test_two_hands_stop_on_the_line(self):
        model, m = self.run_press((0.0, -0.25))
        self.assertAlmostEqual(S.offset(m), 0.0, places=9)
        self.assertAlmostEqual(S.heading_deg(m), 0.0, places=9)
        self.assertGreater(model.pushes, 0)
        self.assertEqual(model.mode, "both sides")
        # drawn back: the motor stays
        for x in np.linspace(0.255, 0.200, 20):
            m = model.update(self.checker_motor(), bar(x=x - 0.01, z=0.026), self.grips(x - 0.01))
        self.assertAlmostEqual(S.offset(m), 0.0, places=9)

    def test_one_hand_overshoots_or_swings_away(self):
        _, m = self.run_press((0.0, None))
        self.assertAlmostEqual(S.offset(m), 0.025, places=9)  # shoved on to the bar's face
        _, m = self.run_press((None, -0.25))
        self.assertAlmostEqual(S.offset(m), -0.006, places=9)  # untouched (6 mm proud, as set down)

    def test_the_checker_carry_wins_and_reset(self):
        model, m = self.run_press((0.0, -0.25))
        self.assertAlmostEqual(S.offset(m), 0.0, places=9)
        moved = self.checker_motor(x=0.300, yaw_deg=1.0)  # a gripper carries it: the checker's pose wins
        m = model.update(moved, bar(), arms())
        self.assertAlmostEqual(S.motor_pose(m)[1][0], 0.300, places=9)
        self.assertAlmostEqual(S.heading_deg(m), 1.0, places=6)
        model.reset()
        self.assertIsNone(model.origin)
        self.assertEqual(model.pushes, 0)
        bad = SimpleNamespace(pos=[math.nan] * 3)
        self.assertIs(model.update(bad, bar(), arms()), bad)


# -- task.py against the offline stand-in ---------------------------------------------------------------------------
@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: the objects, the hooks, the reset hooks, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        cls.mk = mechkit

    def m(self, s, source="sim"):
        return self.mk.state_from_poses(self.task.MOTOR, (*s.origin, *s.quat_xyzw), source=source)

    def feed(self, a):
        rigstate.offline_feed({k: {"tcp": v.tcp, "tips": v.tips, "opening": v.opening} for k, v in a.items()})

    def objs(self, m, b=None, source="sim"):
        return {"motor": self.m(m, source), "straightedge": b or bar()}

    def test_objects(self):
        objs = {o.name: o for o in self.task.OBJECTS}
        self.assertEqual(list(objs), ["blower", "base_plate", "motor", "felt", "straightedge"])
        blower, motor_spec = objs["blower"], objs["motor"]
        self.assertIsNone(blower.mass)
        self.assertEqual(blower.kind, "static")
        self.assertAlmostEqual(motor_spec.mass, 0.15)
        self.assertIsNone(objs["base_plate"].mass)
        self.assertIsNone(objs["felt"].mass)
        self.assertAlmostEqual(objs["straightedge"].mass, 0.18)
        for spec in (motor_spec, objs["straightedge"]):
            self.assertLessEqual(spec.mass, 0.3)
        # the motor's box: the face 88 mm ahead of its origin, its shaft level with the blower's
        lo, hi = motor_spec.local_bounds()
        self.assertAlmostEqual(lo[0], L.FACE_LOCAL[0])
        self.assertAlmostEqual(hi[0], L.BODY_HALF[0])
        self.assertTrue(np.allclose((lo + hi) / 2, L.MOTOR_BOX_CENTRE))
        self.assertAlmostEqual(L.PLATE_TOP + L.MOTOR_PULLEY_C[2], L.SHAFT_Z)
        self.assertAlmostEqual(L.BLOWER_ORIGIN[2] + L.BLOWER_PULLEY_C[2], L.SHAFT_Z)
        # the blower pulley's face is the line; its bottom 7 mm above the table; the motor pulley's 8 mm above the plate
        blo, bhi = blower.world_bounds()
        self.assertAlmostEqual(blo[0], L.PLANE_X)
        self.assertAlmostEqual(L.SHAFT_Z - L.BLOWER_PULLEY_R, 0.007)
        self.assertAlmostEqual(L.MOTOR_PULLEY_C[2] - L.MOTOR_PULLEY_R, 0.008)
        # in line, the motor's pulley stands in front of the plate's front edge, 14 mm clear of it
        self.assertAlmostEqual(L.PLATE_X0 - (L.PLANE_X + L.MOTOR_PULLEY_LEN), 0.014)
        # the press carries the bar's face 30 mm past the line; offline (no pulleys to stop it) it stays 2 mm short
        # of the plate's front edge
        self.assertAlmostEqual(L.PLATE_X0 - (L.PLANE_X + self.task.OVERDRIVE), 0.002)
        # the straightedge spans both pulleys and stands on the felt; its walls are thick enough to simulate
        b = objs["straightedge"]
        self.assertLess(b.pos[1] - b.size[1] / 2, L.OUTER_RIM_Y - 0.06)
        self.assertGreater(b.pos[1] + b.size[1] / 2, L.MOTOR_Y + L.MOTOR_PULLEY_R + 0.02)
        self.assertAlmostEqual(b.pos[2] - b.size[2] / 2, L.TOP + L.FELT_SIZE[2])
        self.assertAlmostEqual(L.CENTRE_DIST, L.MOTOR_Y - L.BLOWER_Y)
        self.assertEqual(self.task.SCORE_OVER, "end")
        self.assertEqual(self.task.SCENE, "station")

    def test_start_layout_hooks(self):
        rigstate.offline_reset()
        self.feed(arms())
        for _ in range(WIN):
            self.assertFalse(self.task.success(self.objs(start_motor(), start_bar())))
        self.assertEqual(self.task.score(self.objs(start_motor(), start_bar())), 0.0)
        self.assertAlmostEqual(self.task._TRACKER.last.details["m_deg"], 5.3933, places=3)

    def test_hooks_read_the_simulator_state(self):
        rigstate.offline_reset()
        self.feed(arms())
        for _ in range(WIN):
            ok = self.task.success(self.objs(motor()))
        self.assertTrue(ok)
        self.assertEqual(self.task.score(self.objs(motor())), 1.0)
        self.assertEqual(self.task.score(self.objs(motor(e=0.004))), 0.5)
        self.assertIsNone(self.task._MODEL.state)  # a simulator state never touches the model

    def test_offline_states_use_the_model(self):
        rigstate.offline_reset()
        self.assertIsNone(self.task._MODEL.origin)
        proud = motor(e=-0.006)
        b0 = bar(x=0.195, z=0.026)
        grips = arms(left=arm((0.195, 0.0, L.TOP + 0.029), 0.128), right=arm((0.195, -0.25, L.TOP + 0.029), 0.128))
        self.feed(grips)
        self.task.success(self.objs(proud, b0, "offline"))
        self.assertAlmostEqual(S.offset(self.task._MODEL.state), -0.006, places=9)
        for x in np.linspace(0.195, 0.245, 30):
            g = arms(left=arm((x, 0.0, L.TOP + 0.029), 0.128), right=arm((x, -0.25, L.TOP + 0.029), 0.128))
            self.feed(g)
            self.task.success(self.objs(proud, bar(x=x, z=0.026), "offline"))
        self.assertAlmostEqual(S.offset(self.task._MODEL.state), 0.0, places=9)
        self.assertEqual(self.task._MODEL.mode, "both sides")
        # score() reads the model without moving it
        n = self.task._MODEL.pushes
        self.task.score(self.objs(proud, bar(x=0.260, z=0.026), "offline"))
        self.assertEqual(self.task._MODEL.pushes, n)
        rigstate.offline_reset()  # rigstate's reset resets the model too
        self.assertIsNone(self.task._MODEL.origin)

    def test_reset_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        self.task.success(self.objs(motor()))
        self.assertEqual(tr.steps, 1)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.task.success(self.objs(motor()))  # arms None right after a reset
        self.assertEqual(tr.arms_unavailable, 1)
        self.feed(arms())
        self.task.success(self.objs(motor()))
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's reset counter: the second, independent signal
        self.task.success(self.objs(motor()))
        self.assertEqual(tr.steps, 1)

    def test_broken_states_score_zero(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        self.feed(arms())
        bad = SimpleNamespace(**vars(self.m(motor())))
        bad.origin = np.array([math.nan, 0.0, 0.0])
        self.assertFalse(self.task.success({"motor": bad, "straightedge": bar()}))
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(self.task.score({"motor": bad, "straightedge": bar()}), 0.0)
        self.assertEqual(self.task.score({"motor": SimpleNamespace(), "straightedge": bar()}), 0.0)

    def test_trace_survives_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            rigstate.offline_reset()
            self.feed(arms())
            self.assertFalse(self.task.success(self.objs(start_motor(), start_bar())))
        finally:
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old
        self.assertFalse((HERE / "no_such_dir").exists())

    def test_demo_switches(self):
        task = self.task
        self.assertEqual(set(task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_plan(self):
        task = self.task
        plan = task.demo()

        def times(wps):
            out, t = [], 0.0
            for w in wps:
                t += w.duration
                out.append((t, w))
            return out

        left, right = times(plan["left"]), times(plan["right"])
        # the bar is handled on one clock: the same x and z targets at the same times from the grip on
        lb = [(round(t, 6), w.pos[0], w.pos[2], w.finger, w.label) for t, w in left if t > task.T_OVER_BAR + 1e-9]
        rb = [(round(t, 6), w.pos[0], w.pos[2], w.finger, w.label) for t, w in right if t > task.T_OVER_BAR + 1e-9]
        self.assertEqual(lb, rb)
        # the press drives both grippers the same 30 mm past the line (OVERDRIVE equal on both arms)
        press = [w for _, w in left + right if w.label.startswith("press")]
        self.assertEqual(len(press), 2)
        self.assertTrue(all(abs(w.pos[0] - (L.PLANE_X - L.BAR_HALF[0] + task.OVERDRIVE)) < 1e-12 for w in press))
        self.assertAlmostEqual(task.OVERDRIVE, 0.030)
        self.assertGreaterEqual(press[0].duration * 0.005, task.PRESS_X - task.START_X - 1e-9)  # <= 5 mm/s average
        # the grips: right outside the blower pulley's outer rim, left between the pulleys, 25 cm apart
        self.assertLess(task.BAR_GRIP_Y["right"] + 0.020, L.OUTER_RIM_Y)
        self.assertGreater(task.BAR_GRIP_Y["left"] - 0.020, L.INNER_RIM_Y)
        self.assertLess(task.BAR_GRIP_Y["left"] + 0.033, L.MOTOR_Y - L.BODY_HALF[1])
        self.assertGreaterEqual(task.BAR_GRIP_Y["left"] - task.BAR_GRIP_Y["right"], 0.20)
        # the motor is set down proud of the line: a straightedge can only push
        self.assertLess(task.PRE_X, L.ALIGNED_X)
        # every open/close takes at least 1.5 s after a settle of at least 0.5 s; every slow descent takes 1.5 s
        for wps in (plan["left"], plan["right"]):
            for a, b in zip(wps, wps[1:]):
                if abs(b.finger - a.finger) > 0.01 and a.pos[2] < L.TOP + 0.1:
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertTrue(a.label.startswith("settle") and a.duration >= 0.5, (a.label, b.label))
                    self.assertEqual(tuple(a.pos), tuple(b.pos))
                if b.label.endswith("slowly"):
                    self.assertGreaterEqual(b.duration, 1.5, b.label)
                    self.assertLessEqual(a.pos[2] - b.pos[2], 0.035, b.label)
        # finger targets 8 mm narrower than the part
        self.assertAlmostEqual(-2 * task.MOTOR_GRIP, L.MOTOR_BODY[1] - 0.008)
        self.assertAlmostEqual(-2 * task.BAR_GRIP, L.BAR_SIZE[0] - 0.008)
        # the hands rise clear (60 mm over the straightedge, with 37 mm to spare) before the plan ends, and the
        # success hold fits in before the scripted policy stops 3 s later
        end = max(left[-1][0], right[-1][0])
        self.assertAlmostEqual(end, 50.9, places=6)
        self.assertGreaterEqual(task.UP_Z - 0.013 - (L.TOP + L.BAR_SIZE[2]), L.HANDS_OFF + 0.035)
        rise = [w for _, w in left if w.label.startswith("straight up")]
        t_clear = next(t for t in np.arange(end - rise[0].duration, end, 0.01)
                       if task.demo_target("left", t)[2] - 0.013 - (L.TOP + L.BAR_SIZE[2]) >= L.HANDS_OFF)
        self.assertLess(t_clear + task.SUCCESS_HOLD_S, end + 3.0 - 1.0)

    def test_demo_target(self):
        task = self.task
        self.assertIsNone(task.demo_target("left", 1.0))  # the first segment starts at the ready pose
        np.testing.assert_allclose(task.demo_target("left", 4.0), task._left()[1].pos)
        # half way through the press in time is half way in distance (smoothstep)
        mid = task.demo_target("right", 30.3 + task.PRESS_S / 2)
        np.testing.assert_allclose(mid, ((0.195 + 0.250) / 2, -0.25, L.TOP + 0.029), atol=1e-9)
        self.assertIsNone(task.demo_target("left", 60.0))
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertIsNone(task.demo_target("right", 35.3))  # an idle arm has no command to compare with
        finally:
            os.environ.pop("DEMO_IDLE_ARM", None)

    def test_text_agrees(self):
        """task.py, task.json and both README commands give the model the same instruction; task.json is
        complete and agrees with the code."""
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))
        self.assertIn("demo: 1.0", meta["scores"])
        self.assertNotIn("media", meta)
        for key in ("title", "category", "keywords", "instruction", "context", "setup", "scoring", "run_limit",
                    "fidelity", "reset_notes"):
            self.assertTrue(meta[key], key)
        steps = int(meta["run_limit"].split()[0].replace(",", ""))
        self.assertIn(f"--max-steps {steps}", readme)
        demo_steps = math.ceil(max(sum(w.duration for w in v) for v in self.task.demo().values()) * HZ)
        self.assertTrue(3 * demo_steps <= steps <= 6 * demo_steps)
        for text in (json.dumps(meta), readme):
            self.assertNotIn("\u2014", text)  # no em-dashes
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for key, n in limits.items():
            self.assertLessEqual(len(meta[key]), n, key)
        # the numbers the text gives
        for needle in ("0.5 degree", "1.0 degree", "2.0 degrees", "290 mm", "60 mm", "10 mm", "Possible scores: 0, "
                       "0.25, 0.5 and 1.0"):
            self.assertIn(needle, meta["scoring"], needle)


def _checker():
    for p in HERE.parents:
        c = p / "tools" / "offline" / "check_task.py"
        if c.is_file():
            return c
    return None


@unittest.skipIf(_checker() is None, "needs our offline checker (tools/offline/check_task.py)")
class OfflineChecker(unittest.TestCase):
    """The whole demo played through our offline checker's kinematics (the motor's pose from offline_model)."""

    def test_demo_predicts_one_and_each_arm_alone_zero(self):
        import subprocess
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("TASK_TRACE", None)
        proc = subprocess.run([sys.executable, str(_checker()), str(HERE), "--one-arm", "--json", "-"],
                              capture_output=True, text=True, env=env, timeout=1200)
        self.assertEqual(proc.returncode, 0, proc.stdout[-3000:] + proc.stderr[-3000:])
        rep = json.loads(proc.stdout)
        proxy = rep["playback"]["proxy"]
        self.assertEqual(proxy["predicted_task_score"], 1.0)
        self.assertFalse(proxy["hook_errors"])
        self.assertLessEqual(proxy["success_held_at"]["step"], 1079)  # the scripted policy stops after step 1,079
        self.assertEqual(rep["one_arm"], {"left_only": 0.0, "right_only": 0.0})
        for side in ("left", "right"):
            self.assertLess(rep["playback"]["arms"][side]["worst_ik_mm"], 0.5)
            self.assertGreater(rep["playback"]["arms"][side]["min_self_mm"], 20.0)
        self.assertGreater(rep["playback"]["min_arm_clearance_m"], 0.04)
        self.assertEqual(proxy["final_hooks"]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
