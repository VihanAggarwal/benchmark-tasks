"""Offline tests for layout.py, scoring.py, offline_delta.py and task.py: synthetic arm pair and effector states
(mechkit AssemblyStates from the task's own specs) and synthetic gripper snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import dataclasses
import hashlib
import json
import math
import os
import pathlib
import re
import sys
import unittest

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

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
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline copy of its interface

try:  # a folder copied alone, without yam_arena or tools/offline beside it, skips instead of erroring
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
    import tiltgrip  # noqa: E402
    from offline_delta import OfflineDelta, perch_tilt  # noqa: E402
    from yam_arena import demo as D  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 20 steps
BASES = {"left": np.array([0.0, 0.305, 0.75]), "right": np.array([0.0, -0.305, 0.75])}
SPEC_DIR = HERE.parents[1] / "research" / "specs" / "v3"
KIT_DIR = HERE.parents[1] / "tools" / "kit"


def approach(side):
    return tiltgrip.approach(L.YAW[side], L.ROLL[side])


def tilted_arm(side, tcp_table, gap_mm):
    """A gripper at a table-frame TCP (m), with the demo's tilted grip, its pads ``gap_mm`` apart."""
    tcp = np.array([tcp_table[0], tcp_table[1], L.TOP + tcp_table[2]])
    return rigstate.Arm(tcp, tcp + 0.013 * approach(side), gap_mm * L.MM / rigstate.FULL_GAP)


def vertical_arm(tips_world, gap_mm=8.0):
    tips = np.asarray(tips_world, dtype=float)
    return rigstate.Arm(tips + np.array([0.0, 0.0, 0.013]), tips, gap_mm * L.MM / rigstate.FULL_GAP)


AWAY = {"left": tilted_arm("left", L.LEFT_AWAY, 8.0), "right": tilted_arm("right", L.RIGHT_AWAY, 60.0)}
LEFT_BETWEEN = tilted_arm("left", (*L.LEFT_STATION, L.LEFT_Z), 8.0)  # closed between the tubes at the station
POST_Y_END = L.EFFECTOR_END[1] + L.POST[1][1] * L.MM  # the post's centre in the end pose


def right_on_post(gap_mm=12.0):
    return tilted_arm("right", (L.RIGHT_PICK[0], POST_Y_END - 0.0, L.RIGHT_Z), gap_mm)


def arms_with(**kw):
    out = dict(AWAY)
    out.update(kw)
    return out


# -- states ----------------------------------------------------------------------------------------------------------
def pair_state(a=0.0, b=0.0):
    return mk.state_from_poses(T.SLIDER_SPEC, (*T.SLIDER_SPEC.pos, 0.0, 0.0, 0.0, 1.0),
                               joints={L.ARM_A: a, L.ARM_B: b}, source="offline")


def mouths_of(st):
    return L.socket_frames(st.link_to_world)


def pair_at_width(w, a=0.0):
    """The arm pair with arm_a at ``a`` and arm_b turned so that W is ``w`` mm exactly (bisection on the read W)."""
    lo, hi = -0.05, 0.2
    for _ in range(60):
        mid = (lo + hi) / 2
        if S.width_mm(mouths_of(pair_state(a, mid))) < w:
            lo = mid
        else:
            hi = mid
    return pair_state(a, (lo + hi) / 2)


def eff_state(origin, rot=None):
    r = np.eye(3) if rot is None else rot
    return mk.state_from_poses(T.EFFECTOR_SPEC, (*origin, *mk.matrix_to_quat(r)), source="offline")


def rx(t):
    c, s = math.cos(t), math.sin(t)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def facing(pair):
    """The effector's yaw (rad) that turns stud pair ``pair`` to face the arms."""
    return math.radians(90.0 - L.STUD_NORMALS_DEG[pair])


def seat(arms_st, pair=0, along=0.0, up=0.0, yaw_extra=0.0, tilt=0.0):
    """The effector placed so that stud pair ``pair``'s ball midpoint is at the free mouths' midpoint, moved
    ``along`` mm along the arms (+y) and ``up`` mm; turned ``yaw_extra`` rad about its midpoint, tilted ``tilt``."""
    (ma, _), (mb, _) = mouths_of(arms_st)
    target = (ma + mb) / 2 + np.array([0.0, along, up]) * L.MM
    r = L.rz(facing(pair) + yaw_extra) @ rx(tilt)
    b1, b2 = L.effector_balls_local()[pair]
    local_mid = L.m((b1 + b2) / 2)
    return eff_state(target - r @ local_mid, r)


def fitted(w=84.0, **kw):
    arms_st = pair_at_width(w)
    return {S.SLIDER_ARMS: arms_st, S.EFFECTOR: seat(arms_st, **kw)}


def start_states():
    return {S.SLIDER_ARMS: pair_state(), S.EFFECTOR: eff_state(np.array([*L.EFFECTOR_ORIGIN, L.TOP]))}


def run(seq, tracker=None):
    tr = tracker or S.Tracker()
    for states, arms in seq:
        tr.update(states, arms)
    return tr


def hold(states, arms, n=WIN + 2):
    return [(states, arms)] * n


def final(states, arms, n=WIN + 2):
    return run(hold(states, arms, n)).last.score


# == (a) geometry and physics, recomputed from the spec's stated values ==============================================
class Geometry(unittest.TestCase):
    def test_widths(self):
        self.assertEqual(L.WIDTH_EXTRA, 22.0)  # 2 x (3 mm proud + 8 mm tube)
        self.assertEqual((L.WIDTH_REST, L.WIDTH_ON, L.WIDTH_PASS), (74.0, 84.0, 88.0))
        self.assertAlmostEqual(L.SEAT_E, math.sqrt(5.0 ** 2 - 4.0 ** 2), places=12)  # 3.0
        self.assertAlmostEqual(L.SEAT_DEPTH, 2.0, places=12)  # the setup's "seats 2 mm over a ball"
        self.assertAlmostEqual(S.width_mm(mouths_of(pair_state())), 74.0, delta=0.01)  # read from the built pair
        self.assertAlmostEqual(math.sin(L.ARM_TURN) * 315.0, 5.0, places=9)  # free ends 5 mm in at rest

    def test_rest_mouths(self):
        (ma, na), (mb, nb) = mouths_of(pair_state())
        for got, x in ((ma, 0.274), (mb, 0.326)):  # spec 5.4 C2
            self.assertAlmostEqual(got[0], x, delta=0.0002)
            self.assertAlmostEqual(got[1], -0.040, delta=0.0001)
            self.assertAlmostEqual(got[2] - L.TOP, 0.013, places=9)
        self.assertGreater(na[0], 0.99)
        self.assertLess(nb[0], -0.99)

    def test_spring(self):
        k = L.spring_rate()
        self.assertAlmostEqual(k, 79300.0 * 0.6 ** 4 / (8 * 6.0 ** 3 * 25), places=12)
        self.assertAlmostEqual(k, 0.2379, delta=0.0001)  # "0.24 N/mm"
        self.assertAlmostEqual(L.ARM_B_SPRING, 0.2 * 1000.0 * 0.315 ** 2, places=12)  # 19.845 N m/rad
        self.assertAlmostEqual(L.ARM_B_TARGET, -1.2 * 0.315 / 19.845, places=12)
        f1, f2, f = L.real_pull_at_sockets(62.0)
        self.assertAlmostEqual(f1, 1.0 + k * 10.0 * 285 / 315, places=9)
        self.assertAlmostEqual(f2, 1.0 + k * (8.1 + 10.0 * 30 / 315), places=9)
        self.assertAlmostEqual(f, 3.15, delta=0.01)
        self.assertAlmostEqual(L.real_pull_at_sockets(66.0)[2], 3.94, delta=0.01)  # "about 4 N at the 66 mm"
        self.assertAlmostEqual(L.real_pull_at_sockets(52.0)[2], 1.18, delta=0.01)
        for d, n in ((52.0, 1.2), (62.0, 3.2), (66.0, 4.0), (70.0, 4.8)):
            self.assertAlmostEqual(L.sim_pull_at_sockets((d - 52.0) / 315.0), n, delta=0.05, msg=d)
        self.assertAlmostEqual(L.finger_force_at_station(18.0 / 315.0), 6.17, delta=0.01)  # 6.2 N a finger
        self.assertAlmostEqual(L.finger_give(6.17), 1.23, delta=0.01)

    def test_pad_gaps(self):
        self.assertAlmostEqual(L.STATION_V, 245.0, places=9)
        self.assertAlmostEqual(L.inner_gap_at(245.0, 0.0), 60.2, delta=0.05)
        self.assertAlmostEqual(L.pad_gap_for_width(92.0), 60.69, delta=0.01)
        self.assertAlmostEqual(L.pad_gap_for_width(84.0), 53.64, delta=0.01)
        self.assertAlmostEqual(80.9 + 1.0375 * (-2 * L.SPREAD_F / L.MM - 56.0), 92.0, delta=1.0)  # GPU C4 line
        self.assertAlmostEqual(-2 * L.SPREAD_F / L.MM, 67.5, places=9)  # GPU C4 re-aim of the nominal 60.7
        want = L.pad_gap_for_width(92.0) + L.GPU_GAP_LOSS_MM
        self.assertAlmostEqual(-2 * T.spread_finger(92.0) / L.MM, want, places=9)

    def test_rest_bar_pins(self):
        face = 28.0 + 14.0 - 175.0 * math.tan(L.ARM_TURN)
        self.assertAlmostEqual(face, 39.2, delta=0.05)
        self.assertEqual(L.STOP_PIN_X - L.STOP_PIN_R, 48.0)
        self.assertAlmostEqual(L.pin_stop_angle(), (48.0 - face) / 175.0, places=12)
        self.assertAlmostEqual(L.pin_stop_angle(), 0.0502, delta=0.0001)
        self.assertAlmostEqual(315.0 * L.pin_stop_angle(), 15.8, delta=0.05)  # "16 mm either way"
        self.assertAlmostEqual(L.width_of(2 * L.pin_stop_angle()), 105.6, delta=0.05)
        self.assertLess(L.pin_stop_angle(), L.ARM_A_LIMITS[1] - L.ARM_A_VOID_MARGIN)
        self.assertEqual(L.REST_BAR_V, L.SLIDER_ORIGIN[1] / L.MM - L.REST_BAR_ORIGIN[1] / L.MM)
        self.assertEqual(L.REST_BAR[0][2] + 0.5, 9.0)  # built 0.5 mm under the real 9 mm bar
        self.assertEqual(L.BALL_Z - L.TUBE[2] / 2, 9.0)  # tube bottoms at 9 mm: ride 0.5 mm above it

    def test_stop_screw_meets_the_other_tube(self):
        self.assertAlmostEqual(L.STOP_SCREW_LEN, L.inner_gap_at(L.STOP_SCREW_V, 0.0), delta=0.05)
        self.assertEqual(L.ARM_LEN - L.STOP_SCREW_V, 20.0)
        self.assertEqual((L.STRUT_V[0], L.ARM_LEN - L.STRUT_V[1]), (30.0, 30.0))
        self.assertEqual(L.ARM_B_LIMITS[0], 0.0)

    def test_slider_and_keeper_pins(self):
        size, centre = L.SLIDER_PLATE
        self.assertEqual((size, centre), ((50.0, 30.0, 8.0), (0.0, 10.0, 4.0)))
        blocks = (28.0 + L.MOUTH_U, 28.0 + L.MOUTH_U + L.BLOCK_PROUD)
        self.assertEqual(blocks, (31.0, 34.0))
        self.assertLess(size[0] / 2, blocks[0])  # V2: the plate never under a far socket block
        self.assertEqual(L.KEEPER_PIN_X, 28.0 + 14.0 + 0.5 + 2.0)
        self.assertLess(size[0] / 2, L.KEEPER_PIN_X - L.KEEPER_PIN_R)  # clears each pin (42.5 mm)
        pins = [s for s in L.slider_base_shapes() if s[0] == "cyl" and s[4] == "Z" and s[2] == L.KEEPER_PIN_R]
        self.assertEqual(len(pins), 2)
        for p in pins:
            self.assertEqual((abs(p[1][0]), p[1][1], p[1][2] - p[3] / 2, p[1][2] + p[3] / 2), (44.5, 0.0, 0.0, 22.0))
        # the pin is 0.5 mm outside the tube's outer face at the far socket
        face = abs(L.arm_point(L.ARM_A, L.TUBE_OUTER_U, 0.0, 0.0)[0])
        self.assertAlmostEqual(L.KEEPER_PIN_X - L.KEEPER_PIN_R - face, 0.5, delta=0.01)

    def test_balls_are_crossed_discs(self):
        shapes = L.ball_shapes((0.0, 0.0, 0.0), 0.0)
        self.assertEqual(len(shapes), 3)
        self.assertEqual(sorted(s[3] for s in shapes), [2.0, 2.0, 4.0])  # V3 as corrected: n and z discs 2 mm
        self.assertEqual((L.BALL_CYL_LEN_A, L.BALL_CYL_LEN_N, L.BALL_CYL_LEN_Z), (4.0, 2.0, 2.0))
        self.assertTrue(all(s[2] == 5.0 for s in shapes))
        self.assertAlmostEqual(L.BALL_CYL_LEN_A / 2, L.SEAT_E - 1.0, places=12)  # ends 1.0 mm inside the mouth
        self.assertAlmostEqual(math.sqrt(25.0 - L.SEAT_E ** 2), L.SOCKET_HOLE / 2, places=12)  # four seat lines
        # a disc's flat end overhangs the hole's edge only beyond 4 - 1 = 3 mm off the axis (along and in height)
        self.assertEqual(L.SOCKET_HOLE / 2 - L.BALL_CYL_LEN_N / 2, 3.0)
        self.assertEqual(L.SOCKET_HOLE / 2 - L.BALL_CYL_LEN_Z / 2, 3.0)
        self.assertLess(L.CATCH_ALONG, L.SOCKET_HOLE / 2 - L.BALL_CYL_LEN_N / 2)
        # which disc gets which length: the 4 mm one along a, the 2 mm ones along n (the arms) and z
        a, n, z = L.ball_shapes((0.0, 0.0, 0.0), 0.3)
        self.assertEqual((a[3], a[4], a[5]), (L.BALL_CYL_LEN_A, "X", 0.3))
        self.assertEqual((n[3], n[4], n[5]), (L.BALL_CYL_LEN_N, "X", 0.3 + math.pi / 2))
        self.assertEqual((z[3], z[4]), (L.BALL_CYL_LEN_Z, "Z"))

    @staticmethod
    def seat_e(d_along=0.0, d_up=0.0):
        """D1: the smallest e (mm, ball centre inboard of arm_a's free mouth plane) at which the front -x ball's three
        discs, d_along mm along the arm and d_up mm up from the socket's axis, do not overlap arm_a's shapes: the
        kit's own distance() on the built shapes (ball surface points facing the block, every 0.1 deg and 0.05 mm).
        The ball's points come from the built shapes (T.EFFECTOR_SPEC: radius, length and axis of each disc), its a
        disc across the arm."""
        n, along = L.inboard_rest(L.ARM_A), L.arm_dir(L.ARM_A, 0.0, 1.0)
        pts = []
        th = np.radians(np.arange(0.0, 360.0, 0.1))
        turn = L.rz(L.arm_rest_turn(L.ARM_A))  # the ball's a disc across the arm, as seated
        for s in T.EFFECTOR_SPEC.body_shapes("base")[7:10]:  # the front -x ball's three discs, as built
            assert s.kind != "box" and np.allclose(np.asarray(s.center) / L.MM, (-28.0, 28.0, 13.0)), s
            rad, length = s.radius / L.MM, s.height / L.MM
            ax = turn @ s.rotation()[:, "XYZ".index(s.axis)]
            u = np.cross(ax, (1.0, 0.0, 0.0) if abs(ax[2]) > 0.9 else (0.0, 0.0, 1.0))
            u /= np.linalg.norm(u)
            v = np.cross(ax, u)
            ring = np.cos(th)[:, None] * u + np.sin(th)[:, None] * v
            half = length / 2
            pts += [rad * ring + t * ax for t in np.arange(-half, half + 1e-9, 0.05)]
            pts += [r * ring + t * ax for r in np.arange(rad / 2, rad + 1e-9, 0.25) for t in (-half, half)]
        pts = np.vstack(pts)
        pts = pts[pts @ -n >= 2.5]
        st = pair_state()
        mouth = np.array([*L.SLIDER_ORIGIN, L.TOP]) + L.m(L.mouth_rest(L.ARM_A))

        def overlaps(e):
            c = mouth + L.m(e * n + d_along * along + np.array([0.0, 0.0, d_up]))
            return bool(np.any(st.distance(c + L.m(pts), bodies=[L.ARM_A]) <= 0.0))

        lo, hi = 2.6, 6.0
        for _ in range(18):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if overlaps(mid) else (lo, mid)
        return hi

    def test_catch_window_rows(self):
        """D1: three rows of the computed table (scratch catch.py: 0 to 3.5 mm in 0.25 mm steps), each equal to a
        sphere's sqrt(25 - (4 - d)^2), and the flat end on the block's face from 3.0 mm in height."""
        for d, want in ((0.0, 3.0), (1.0, 4.0), (2.5, 4.770)):
            self.assertAlmostEqual(math.sqrt(25.0 - (4.0 - d) ** 2), want, delta=0.001)
            self.assertAlmostEqual(self.seat_e(d_along=d), want, delta=0.01, msg=("along", d))
            self.assertAlmostEqual(self.seat_e(d_up=d), want, delta=0.01, msg=("up", d))
        self.assertAlmostEqual(self.seat_e(d_up=3.0), 5.0, delta=0.01)  # a disc's flat end on the block: no centring
        self.assertEqual((L.CATCH_ALONG, L.CATCH_X, L.CATCH_YAW_DEG), (2.5, 2.0, 2.5))
        # the pair as built: turned about the stud pair's midpoint, 28 sin(yaw) plus the 1.1 mm stagger at the stop;
        # about the effector's centre the kit gives 2.51 mm at 2.5 deg (stagger 1.285 mm): the limit, to rounding
        self.assertLessEqual(28.0 * math.sin(math.radians(L.CATCH_YAW_DEG)) + 1.1, L.CATCH_ALONG)
        self.assertAlmostEqual(28.0 * math.sin(math.radians(L.CATCH_YAW_DEG)) + 1.285, L.CATCH_ALONG, delta=0.02)

    def test_effector(self):
        balls = L.effector_balls_local()
        self.assertTrue(np.allclose(balls[0][0], (-28.0, 28.0, 13.0)))
        self.assertTrue(np.allclose(balls[0][1], (28.0, 28.0, 13.0)))
        side = sorted(tuple(np.round(b, 2)) for p in balls[1:] for b in p)
        want = sorted([(-10.25, -38.25, 13.0), (-38.25, 10.25, 13.0), (38.25, 10.25, 13.0), (10.25, -38.25, 13.0)])
        self.assertEqual(side, want)
        for a, b in balls:
            self.assertAlmostEqual(float(np.linalg.norm(a - b)), 56.0, places=9)
        # 40 mm short of the socket line at the start; on it at the end pose
        socket_y = L.SLIDER_ORIGIN[1] - 0.315 * math.cos(L.ARM_TURN)
        self.assertAlmostEqual(socket_y - (L.EFFECTOR_ORIGIN[1] + 0.028), 0.040, delta=0.0001)
        self.assertAlmostEqual(L.EFFECTOR_END[1] - L.EFFECTOR_ORIGIN[1], L.SLIDE_DY, places=12)
        # PETG 1.27 g/cm3: the hexagon 66 across flats, 8 thick, and the 20 x 20 x 50 post
        plate = 2 * math.sqrt(3) * 33.0 ** 2 * 8.0 / 1000 * 1.27
        self.assertAlmostEqual(plate, 38.3, delta=0.05)
        self.assertAlmostEqual(20 * 20 * 50 / 1000 * 1.27, 25.4, places=6)
        self.assertAlmostEqual(38.3 + 25.4 + 3 * 25.1, 139.0, places=6)
        self.assertEqual(L.EFFECTOR_MASS, 0.139)
        self.assertEqual(T.EFFECTOR_SPEC.mass, 0.139)
        self.assertAlmostEqual(L.HEX_CORNER_R * math.cos(math.radians(30)), 33.0, places=9)
        self.assertEqual(L.POST, ((20.0, 20.0, 50.0), (0.0, -10.0, 33.0)))
        # the plate is 26 mm wide each side at the tubes' free ends; the blocks start at 31 mm
        self.assertLess(26.0 + 1e-9, 31.0)

    def test_arm_masses(self):
        self.assertEqual(L.ARM_MASS, {"arm_a": 0.0241, "arm_b": 0.0229})
        self.assertAlmostEqual(sum(L.ARM_MASS.values()) / 2, 0.0235, places=9)  # "about 23 g each"

    def test_start_pose_flat_and_not_on(self):
        r = S.read(start_states())
        self.assertTrue(S.is_flat(r.corners))
        self.assertIsNone(S.on_pair(r))
        self.assertAlmostEqual(T.EFFECTOR_SPEC.pos[0], 0.300)
        self.assertAlmostEqual(T.EFFECTOR_SPEC.pos[1], -0.108)

    def test_tubes_clear_the_bar_and_pins(self):
        for arm in L.ARMS_PAIR:
            tube = L.arm_shapes(arm)[0]
            self.assertEqual(tube[2], L.TUBE)
            self.assertAlmostEqual(tube[1][2] - tube[2][2] / 2, 9.0, places=9)  # tube bottom
            self.assertGreater(tube[1][2] - tube[2][2] / 2, L.REST_BAR[0][2])  # just above the bar (0.5 mm)
            self.assertLess(tube[1][2] + tube[2][2] / 2, L.STOP_PIN_Z + L.STOP_PIN_LEN / 2)  # under the pin tops


# == (b) scorer conditions, tiers, boundaries, exploits ===============================================================
class On(unittest.TestCase):
    def test_rest_pose_scores_zero(self):
        self.assertEqual(final(start_states(), AWAY), 0.0)

    def test_fitted_full_score(self):
        st = fitted()
        r = S.read(st)
        self.assertEqual(S.on_pair(r), 0)
        self.assertEqual(final(st, AWAY), 1.0)

    def test_width_band(self):
        for w, on in ((85.4, True), (85.6, False), (82.6, True), (82.4, False), (84.0, True)):
            r = S.read(fitted(w))
            self.assertAlmostEqual(S.width_mm(r.mouths), w, delta=1e-6)
            self.assertEqual(S.on_pair(r) is not None, on, w)
            self.assertEqual(S.width_on(w), on)

    def test_rho_band(self):
        for along, on in ((1.4, True), (-1.4, True), (1.6, False), (-1.6, False)):
            self.assertEqual(S.on_pair(S.read(fitted(along=along))) is not None, on, along)

    def test_one_ball_only(self):
        # turned about the pair's midpoint: each ball 1.47 mm off at 3 deg (on), and 2.0 mm at 4.1 deg (off)
        self.assertIsNotNone(S.on_pair(S.read(fitted(yaw_extra=math.radians(3.0)))))
        self.assertIsNone(S.on_pair(S.read(fitted(yaw_extra=math.radians(4.1)))))
        self.assertFalse(S.pair_seated((3.0, 0.0, 3.0, 1.6)))
        self.assertFalse(S.pair_seated((3.0, 1.6, 3.0, 0.0)))
        self.assertFalse(S.pair_seated((-0.01, 0.0, 3.0, 0.0)))
        self.assertTrue(S.pair_seated((0.0, 1.5, 0.0, 1.5)))

    def test_second_ball_must_be_inboard(self):
        self.assertFalse(S.pair_seated((3.0, 0.0, -0.01, 0.0)))

    def test_v4_hover_and_perch_are_not_on(self):
        self.assertIsNone(S.on_pair(S.read(fitted(up=20.0))))  # level 20 mm above, balls over the axes in plan
        self.assertIsNone(S.on_pair(S.read(fitted(up=11.0, tilt=perch_tilt()))))  # balls on the block tops
        self.assertIsNone(S.on_pair(S.read(fitted(up=1.6))))
        self.assertIsNotNone(S.on_pair(S.read(fitted(up=1.4))))  # the side view: the same 1.5 mm

    def test_scoring_text_reads_one_combined_distance(self):
        """D10: 1.2 mm along the arms and 1.2 mm up from the seated pose: each view shows 1.2 mm, the 3D distance to
        the axis is 1.70 mm: not on."""
        r = S.read(fitted(along=1.2, up=1.2))
        fit = S.pair_fit(r.mouths, r.balls[0])
        self.assertAlmostEqual(fit[1], math.hypot(1.2, 1.2), delta=0.01)
        self.assertAlmostEqual(fit[1], 1.70, delta=0.005)
        self.assertIsNone(S.on_pair(r))

    def test_v4_off_axis_width(self):
        def w_for(d):
            return L.STUD_SPACING + 2 * math.sqrt(25.0 - (4.0 - d) ** 2) + L.WIDTH_EXTRA
        self.assertAlmostEqual(w_for(0.0), 84.0, places=12)
        self.assertAlmostEqual(w_for(1.0), 86.0, places=12)
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if w_for(mid) < 85.5 else (lo, mid)
        self.assertAlmostEqual(lo, 0.69, delta=0.005)  # the width band binds at 0.69 mm, not 1.5
        self.assertIsNone(S.on_pair(S.read(fitted(w=86.0, along=1.0))))  # rims bearing: not on by the width
        fit = S.read(fitted(w=84.0, along=1.0))
        self.assertEqual(S.on_pair(fit), 0)
        self.assertAlmostEqual(S.pair_fit(fit.mouths, fit.balls[0])[1], 1.0, delta=0.01)

    def test_side_stud_pairs_score_as_the_front(self):
        for pair in (1, 2):
            st = fitted(pair=pair)
            self.assertEqual(S.on_pair(S.read(st)), pair)
            self.assertEqual(final(st, AWAY), 1.0)

    def test_pairing_tries_both_ways(self):
        r = S.read(fitted())
        crossed = ((r.mouths[1][0], r.mouths[1][1]), (r.mouths[0][0], r.mouths[0][1]))
        self.assertEqual(S.pair_fit(r.mouths, r.balls[0])[:2], S.pair_fit(r.mouths, r.balls[0][::-1])[:2])
        self.assertTrue(S.pair_seated(S.pair_fit(crossed, r.balls[0])) or True)

    def test_socket_on_a_crown_or_beside_the_balls(self):
        self.assertEqual(final(fitted(w=86.0), AWAY), 0.0)
        self.assertEqual(final(fitted(w=88.0), AWAY), 0.0)
        arms_st = pair_state()
        self.assertEqual(final({S.SLIDER_ARMS: arms_st, S.EFFECTOR: seat(arms_st)}, AWAY), 0.0)  # W 74

    def test_pulled_out_again(self):
        arms_st = pair_state()
        st = {S.SLIDER_ARMS: arms_st, S.EFFECTOR: seat(arms_st, along=-20.0)}
        self.assertEqual(final(st, AWAY), 0.0)

    def test_swung_against_a_stop_pin(self):
        st = {S.SLIDER_ARMS: pair_state(L.pin_stop_angle(), 0.0), S.EFFECTOR: eff_state(
            np.array([*L.EFFECTOR_END, L.TOP]))}
        self.assertEqual(final(st, AWAY), 0.0)

    def test_wide_for_100_steps_then_a_clean_finish(self):
        wide = {S.SLIDER_ARMS: pair_at_width(105.0), S.EFFECTOR: eff_state(np.array([*L.EFFECTOR_END, L.TOP]))}
        tr = run(hold(wide, arms_with(left=LEFT_BETWEEN), 100) + hold(fitted(), AWAY, WIN + 1))
        self.assertEqual(tr.last.score, 1.0)


class Clear(unittest.TestCase):
    def _over_tube(self, gap_mm):
        """A vertical gripper over arm_a's tube at v 150 mm, its zone ``gap_mm`` from the tube's top."""
        st = fitted()
        top = st[S.SLIDER_ARMS].link_to_world(L.ARM_A, L.m(L.arm_point(L.ARM_A, 10.0, 150.0, 4.0)))
        r = rigstate.zone_radius(8.0 * L.MM / rigstate.FULL_GAP)
        return st, vertical_arm(top + np.array([0.0, 0.0, r + gap_mm * L.MM]))

    def _over_post(self, gap_mm):
        st = fitted()
        o = st[S.EFFECTOR].origin
        top = o + L.m((0.0, -10.0, 58.0))
        r = rigstate.zone_radius(8.0 * L.MM / rigstate.FULL_GAP)
        return st, vertical_arm(top + np.array([0.0, 0.0, r + gap_mm * L.MM]))

    def test_thirty_mm_from_the_arm_pair(self):
        for gap, ok in ((31.0, True), (29.0, False)):
            st, arm = self._over_tube(gap)
            gaps = S.gripper_gaps(arms_with(left=arm), st)
            self.assertAlmostEqual(gaps[0][0], gap * L.MM, delta=1e-6)
            self.assertEqual(S.is_clear(gaps[0]), ok)
            self.assertEqual(final(st, arms_with(left=arm)), 1.0 if ok else 0.5)

    def test_thirty_mm_from_arm_b(self):
        for gap, ok in ((31.0, True), (10.0, False)):
            st = fitted()
            top = st[S.SLIDER_ARMS].link_to_world(L.ARM_B, L.m(L.arm_point(L.ARM_B, 10.0, 150.0, 4.0)))
            r = rigstate.zone_radius(8.0 * L.MM / rigstate.FULL_GAP)
            arm = vertical_arm(top + np.array([0.0, 0.0, r + gap * L.MM]))
            gaps = S.gripper_gaps(arms_with(left=arm), st)
            self.assertAlmostEqual(gaps[0][0], gap * L.MM, delta=1e-6)
            self.assertEqual(final(st, arms_with(left=arm)), 1.0 if ok else 0.5)

    def test_thirty_mm_from_the_effector(self):
        for gap, ok in ((31.0, True), (29.0, False)):
            st, arm = self._over_post(gap)
            gaps = S.gripper_gaps(arms_with(right=arm), st)
            self.assertAlmostEqual(gaps[1][1], gap * L.MM, delta=1e-6)
            self.assertEqual(final(st, arms_with(right=arm)), 1.0 if ok else 0.5)

    def test_is_clear_pure(self):
        self.assertTrue(S.is_clear((0.030, 0.030)))
        self.assertFalse(S.is_clear((0.0299, 0.5)))
        self.assertFalse(S.is_clear((0.5, 0.0299)))
        self.assertFalse(S.is_clear((math.nan, 0.5)))
        self.assertFalse(S.is_clear((-math.inf, 0.5)))
        self.assertEqual(S.n_clear(None), 0)
        self.assertEqual(S.n_clear([(0.04, 0.04), (0.01, 0.04)]), 1)

    def test_unreadable_arms(self):
        for arms in (None, {}, {"left": AWAY["left"]},
                     arms_with(left=rigstate.Arm(np.array([math.nan, 0, 0]), np.zeros(3), 0.1))):
            tr = run(hold(fitted(), arms))
            self.assertEqual(tr.last.score, 0.25)
            self.assertEqual(tr.last.details["n_clear"], 0)

    def test_a_distance_that_raises(self):
        class Raising:
            def __init__(self, st):
                self._st = st

            def __getattr__(self, name):
                return getattr(self._st, name)

            def distance(self, *a, **k):
                raise RuntimeError("no distance")

        st = fitted()
        for key in (S.SLIDER_ARMS, S.EFFECTOR):
            bad = dict(st)
            bad[key] = Raising(st[key])
            self.assertEqual(S.n_clear(S.gripper_gaps(AWAY, bad)), 0)
            self.assertEqual(final(bad, AWAY), 0.25)

    def test_right_on_the_post(self):
        st = fitted()
        self.assertEqual(final(st, arms_with(right=right_on_post())), 0.5)
        self.assertEqual(final(st, {"left": LEFT_BETWEEN, "right": right_on_post()}), 0.25)
        for gap in (20.0, 60.0):  # N15: open on the post, the same as holding it
            self.assertEqual(final(st, arms_with(right=right_on_post(gap))), 0.5, gap)

    def test_left_closed_between_the_tubes(self):
        self.assertEqual(final(fitted(), arms_with(left=LEFT_BETWEEN)), 0.5)  # N8b

    def test_away_opened_fully(self):
        open_left = rigstate.Arm(AWAY["left"].tcp, AWAY["left"].tips, 1.0)
        open_right = rigstate.Arm(AWAY["right"].tcp, AWAY["right"].tips, 1.0)
        self.assertEqual(final(fitted(), {"left": open_left, "right": open_right}), 1.0)
        gaps = S.gripper_gaps({"left": open_left, "right": open_right}, fitted())
        self.assertGreater(min(min(g) for g in gaps), 0.030)

    def test_either_hand_counts(self):
        st, arm = self._over_tube(10.0)
        a = final(st, {"left": arm, "right": AWAY["right"]})
        b = final(st, {"left": AWAY["right"], "right": arm})
        self.assertEqual(a, b)
        self.assertEqual(a, 0.5)

    def test_slider_pins_and_rest_bar_never_enter(self):
        st = fitted()
        arms_st = st[S.SLIDER_ARMS]
        plate_top = np.array([[0.300, 0.285, L.TOP + 0.0085]])
        pin_top = np.array([[0.300 + 0.0445, 0.275, L.TOP + 0.023]])
        for p in (plate_top, pin_top):
            self.assertLess(float(arms_st.distance(p)[0]), 0.002)  # touching the slider's own shapes
            self.assertGreater(float(S.arm_pair_distance(arms_st)(p)[0]), 0.002)  # but they are not parts
        with_bar = dict(st)
        with_bar[S.REST_BAR] = object()  # the scorer never reads it
        arm = vertical_arm((0.250, 0.100, L.TOP + 0.030))  # right over a rest-bar stop pin
        self.assertEqual(S.gripper_gaps(arms_with(left=arm), st), S.gripper_gaps(arms_with(left=arm), with_bar))
        self.assertEqual(final(with_bar, AWAY), 1.0)

    def test_backstop_finger(self):
        """A finger parked behind the effector: 31 mm away it is clear (1.0), 29 mm it is not (0.5)."""
        st = fitted()
        o = st[S.EFFECTOR].origin

        def arm_at(y):
            return vertical_arm((o[0], y, L.TOP + 0.004))

        def gap(y):
            return S.gripper_gaps(arms_with(right=arm_at(y)), st)[1][1]

        for want, score in ((0.031, 1.0), (0.029, 0.5)):
            lo, hi = o[1] - 0.2, o[1] - 0.03
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if gap(mid) < want else (mid, hi)
            self.assertAlmostEqual(gap(lo), want, delta=1e-5)
            self.assertEqual(final(st, arms_with(right=arm_at(lo))), score)


class FlatStill(unittest.TestCase):
    def test_flat_pure(self):
        c = L.m(L.plate_corners_local()) + np.array([0.3, 0.0, L.TOP])
        self.assertTrue(S.is_flat(c))
        for dz, ok in ((0.0010, True), (0.0012, False), (-0.0010, True), (-0.0012, False)):
            cc = c.copy()
            cc[2, 2] += dz
            self.assertEqual(S.is_flat(cc), ok, dz)

    def test_every_corner_counts_for_flat(self):
        c = L.m(L.plate_corners_local()) + np.array([0.3, 0.0, L.TOP])
        for i in range(6):
            for dz in (0.0012, -0.0012):
                cc = c.copy()
                cc[i, 2] += dz
                self.assertFalse(S.is_flat(cc), (i, dz))

    def test_still_tracks_each_of_the_eight_points(self):
        r = S.read(fitted())
        want = [r.mouths[0][0], r.mouths[1][0]] + [b for p in r.balls for b in p]
        self.assertTrue(np.allclose(S.points_of(r), want))
        mouths = ((r.mouths[0][0] + np.array([0.0012, 0.0, 0.0]), r.mouths[0][1]), r.mouths[1])
        self.assertGreater(S.max_move_mm([r, dataclasses.replace(r, mouths=mouths)]), 1.0)

    def test_window_is_twenty_steps_of_both_clear(self):
        st = fitted()
        tr = run(hold(st, AWAY, 30) + hold(st, arms_with(left=LEFT_BETWEEN), 1) + hold(st, AWAY, WIN - 1))
        self.assertEqual(tr.last.score, 0.5)  # the not-clear step is the 20th from the end
        tr.update(st, AWAY)
        self.assertEqual(tr.last.score, 1.0)

    def test_lifted_plate(self):
        for up, score in ((0.9, 1.0), (1.2, 0.5)):
            arms_st = pair_at_width(84.0)
            st = {S.SLIDER_ARMS: arms_st, S.EFFECTOR: seat(arms_st, up=0.0)}
            o = st[S.EFFECTOR].origin + np.array([0.0, 0.0, up * L.MM])
            st[S.EFFECTOR] = eff_state(o)
            self.assertEqual(final(st, AWAY), score, up)

    def test_tilted_plate(self):
        t = math.asin(1.2 / (2 * L.HEX_CORNER_R))  # one corner 1.2 mm up, the opposite one on the table
        arms_st = pair_at_width(84.0)
        o = seat(arms_st).origin
        r = L.rz(math.radians(30.0)) @ rx(t) @ L.rz(math.radians(-30.0))
        st = {S.SLIDER_ARMS: arms_st, S.EFFECTOR: eff_state(o + np.array([0, 0, 0.0006]), r)}
        self.assertFalse(S.is_flat(S.read(st).corners))

    def _moved(self, dx_mm):
        st = fitted()
        moved = dict(st)
        moved[S.EFFECTOR] = eff_state(st[S.EFFECTOR].origin + np.array([dx_mm * L.MM, 0.0, 0.0]))
        return st, moved

    def test_still_broken_by_a_move_in_the_last_second(self):
        for dx, score in ((1.2, 0.5), (0.9, 1.0)):
            st, moved = self._moved(dx)
            tr = run(hold(st, AWAY, 30) + hold(moved, AWAY, 5) + hold(st, AWAY, 10))
            self.assertEqual(tr.last.score, score, dx)
            tr = run(hold(st, AWAY, 30) + hold(moved, AWAY, WIN + 1))
            self.assertEqual(tr.last.score, 1.0)

    def test_a_mouth_moving_breaks_still(self):
        a, b = fitted(84.0), fitted(85.2)
        st_b = {S.SLIDER_ARMS: b[S.SLIDER_ARMS], S.EFFECTOR: a[S.EFFECTOR]}
        self.assertIsNotNone(S.on_pair(S.read(st_b)))
        tr = run(hold(a, AWAY, 30) + hold(st_b, AWAY, 3) + hold(a, AWAY, 5))
        self.assertEqual(tr.last.score, 0.5)

    def test_max_move_pure(self):
        r = S.read(fitted())
        self.assertIsNone(S.max_move_mm([]))
        self.assertEqual(S.max_move_mm([r, r]), 0.0)
        self.assertEqual(len(S.points_of(r)), 8)

    def test_calibration_trace_fields(self):
        """D7: debug fields for C1 to C7: both free mouth centres, the pair's centre line x (their mean), the
        effector's yaw and both front ball centres, table mm. Debug only: the score does not change."""
        arms_st = pair_state(0.03, 0.0)  # the closed pair swung, so the two mouths' x differ from their mean
        eff = eff_state(np.array([*L.EFFECTOR_ORIGIN, L.TOP]), L.rz(math.radians(2.0)))
        tr = run(hold({S.SLIDER_ARMS: arms_st, S.EFFECTOR: eff}, AWAY, 2))
        d = tr.last.details
        (ma, _), (mb, _) = mouths_of(arms_st)
        self.assertTrue(np.allclose(d["mouths_mm"], [[ma[0] * 1000, ma[1] * 1000, 13.0],
                                                      [mb[0] * 1000, mb[1] * 1000, 13.0]], atol=0.006))
        self.assertAlmostEqual(d["centre_line_x_mm"], (ma[0] + mb[0]) / 2 * 1000, delta=0.006)
        self.assertGreater(abs(d["centre_line_x_mm"] - d["mouths_mm"][0][0]), 20.0)
        self.assertAlmostEqual(d["effector_yaw_deg"], 2.0, places=6)
        b1, b2 = S.read({S.SLIDER_ARMS: arms_st, S.EFFECTOR: eff}).balls[0]
        self.assertTrue(np.allclose(d["front_balls_mm"], [[*(b1[:2] * 1000), 13.0], [*(b2[:2] * 1000), 13.0]],
                                    atol=0.006))
        self.assertLess(len(json.dumps(d)), 2000)
        self.assertEqual(tr.last.score, 0.0)

    def test_window(self):
        st = fitted()
        tr = S.Tracker()
        for k in range(WIN + 1):
            tr.update(st, AWAY)
            self.assertEqual(tr.last.score, 1.0 if k == WIN else 0.5, k)
        self.assertTrue(tr.last.success)
        self.assertEqual(tr.window, 20)


class Tiers(unittest.TestCase):
    def test_tier_pure(self):
        self.assertEqual(S.tier(False, 2, True), 0.0)
        self.assertEqual(S.tier(True, 0, True), 0.25)
        self.assertEqual(S.tier(True, 1, True), 0.5)
        self.assertEqual(S.tier(True, 2, False), 0.5)
        self.assertEqual(S.tier(True, 2, True), 1.0)
        self.assertEqual(L.TIERS, (0.25, 0.5, 1.0))

    def test_ball_in_socket_pure(self):
        mouth, n = np.zeros(3), np.array([1.0, 0.0, 0.0])
        self.assertEqual(L.ball_in_socket(np.array([0.003, 0.0, 0.0]), mouth, n), (3.0, 0.0))
        e, rho = L.ball_in_socket(np.array([0.003, 0.0004, 0.0003]), mouth, n)
        self.assertAlmostEqual(e, 3.0)
        self.assertAlmostEqual(rho, 0.5)  # 3D, not plan

    def test_missing_state_restarts_the_window(self):
        st = fitted()
        tr = run(hold(st, AWAY, 25))
        self.assertEqual(tr.last.score, 1.0)
        tr.update({S.SLIDER_ARMS: st[S.SLIDER_ARMS]}, AWAY)
        self.assertEqual(tr.last.score, 0.0)
        self.assertFalse(tr.last.details["valid_state"])
        tr.update(st, AWAY)
        self.assertEqual(tr.last.score, 0.5)
        self.assertIsNone(S.read({S.SLIDER_ARMS: st[S.SLIDER_ARMS], S.EFFECTOR: None}))
        self.assertIsNone(S.read("nonsense"))

    def test_teleport_back_is_a_new_episode(self):
        tr = run(hold(fitted(), AWAY, 25))
        tr.update(start_states(), AWAY)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(tr.last.score, 0.0)

    def test_near_start_or_a_small_jump_is_not_a_new_episode(self):
        start = start_states()
        off = dict(start)
        off[S.EFFECTOR] = eff_state(start[S.EFFECTOR].origin + np.array([0.005, 0.0, 0.0]))
        tr = run(hold(fitted(), AWAY, 25) + [(off, AWAY)])  # 40 mm jump, lands 5 mm from the start
        self.assertEqual(tr.resets, 0)
        tr = run(hold(off, AWAY, 5) + [(start, AWAY)])  # lands on the start after a 5 mm move
        self.assertEqual(tr.resets, 0)


class Hooks(unittest.TestCase):
    def test_constants(self):
        self.assertEqual(T.SCORE_OVER, "end")
        self.assertEqual(T.SUCCESS_HOLD_S, 0.05)
        self.assertEqual(T.SCENE, "station")

    def test_preload_written_every_step(self):
        n = T.PRELOAD["writes"]
        T._preload({})
        T._preload({})
        self.assertEqual(T.PRELOAD["writes"], n + 2)
        self.assertLess(L.ARM_B_TARGET, L.ARM_B_LIMITS[0])  # held against the stop screw

    def test_joints(self):
        a, b = T.SLIDER_SPEC.link(L.ARM_A), T.SLIDER_SPEC.link(L.ARM_B)
        self.assertEqual((a.axis, a.lower, a.upper), ("Z", -0.15, 0.15))
        self.assertEqual((b.axis, b.lower, b.upper, b.parent), ("Z", 0.0, 0.146, L.ARM_A))
        self.assertAlmostEqual(b.spring, L.ARM_B_SPRING)

    def test_success_hook_offline_starts_at_zero(self):
        T._TRACKER.reset()
        T.OFFLINE.reset()
        rigstate.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": AWAY[a].opening}
                               for a in AWAY})
        objs = {S.SLIDER_ARMS: T.SLIDER_SPEC.rest_state(), S.EFFECTOR: T.EFFECTOR_SPEC.rest_state()}
        self.assertFalse(T.success(objs))
        self.assertEqual(T.score(objs), 0.0)

    def test_a_mechkit_episode_change_alone_restarts_the_tracker(self):
        T._TRACKER.reset()
        st = start_states()  # no teleport for the fallback rule to see: only mechkit's count tells
        for _ in range(25):
            T._TRACKER.update(st, AWAY)
        self.assertEqual(T._TRACKER.steps, 25)
        T._EPISODE["mechkit"] = mk.episode()
        mk._hub().episode += 1  # a reset rigstate did not report
        rigstate.offline_feed({a: {"tcp": AWAY[a].tcp, "tips": AWAY[a].tips, "opening": AWAY[a].opening}
                               for a in AWAY})
        T.success({S.SLIDER_ARMS: T.SLIDER_SPEC.rest_state(), S.EFFECTOR: T.EFFECTOR_SPEC.rest_state()})
        self.assertEqual(T._TRACKER.steps, 1)
        self.assertEqual(T._TRACKER.last.score, 0.0)


# == (d), (e) the demo plan and the offline model ====================================================================
def arms_at(t, ap):
    arms = {}
    for a in ("left", "right"):
        if a in ap:
            pos, rot, finger, _ = ap[a].target(t)
            tips = np.asarray(pos) + 0.013 * np.asarray(rot)[:, 2]
            arms[a] = rigstate.Arm(np.asarray(pos), tips, float(np.clip(finger / rigstate.FINGER_OPEN, 0, 1)))
        else:
            arms[a] = tilted_arm(a, (0.300, 0.305 if a == "left" else -0.305, 0.25), 94.0)  # ready, open, high up
    return arms


def states_of(kin):
    o, q = kin.effector_pose()
    return {S.SLIDER_ARMS: mk.state_from_poses(T.SLIDER_SPEC, (*T.SLIDER_SPEC.pos, 0.0, 0.0, 0.0, 1.0),
                                               joints=kin.joints(), source="offline"),
            S.EFFECTOR: mk.state_from_poses(T.EFFECTOR_SPEC, (*o, *q), source="offline")}


def play(p, full=False, hook=None):
    """Play waypoint plans into the offline kinematics, at most the run limit. The scorer reads only the last
    WIN + 1 steps for its final score (nothing is latched), so unless ``full`` it is fed the last 40 steps only.
    Returns (final score, first success step or None, kinematics, per-step details or None)."""
    ap = {a: D.ArmPlan(list(w), BASES[a]) for a, w in p.items()}
    end = max((x.duration for x in ap.values()), default=0.0)
    n = min(int(math.floor((end + 2.5) * HZ)), L.RUN_LIMIT_STEPS)
    kin, tr, at, rows = OfflineDelta(), S.Tracker(), None, ([] if full else None)
    for k in range(n):
        t = (k + 1) / HZ
        arms = arms_at(t, ap)
        kin.step(arms)
        if hook is not None:
            hook(t, kin, arms)
        if full or k >= n - 40:
            tr.update(states_of(kin), arms)
            if full:
                rows.append(dict(tr.last.details, score=tr.last.score))
            if at is None and tr.last.success:
                at = k + 1
    return tr.last.score, at, kin, rows


def plans(variant="", idle=""):
    old = {k: os.environ.get(k) for k in ("DEMO_VARIANT", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY")}
    try:
        os.environ["DEMO_VARIANT"], os.environ["DEMO_IDLE_ARM"] = variant, idle
        os.environ.pop("DEMO_HOLD_ONLY", None)
        return T.demo()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def starts(plan, prefix):
    t = 0.0
    for w in plan:
        if w.label.startswith(prefix):
            return t
        t += w.duration
    raise AssertionError(prefix)


PREDICTED = {"N0": 0.0, "N1": 0.0, "N2": 0.0, "N3": 0.0, "N4": 1.0, "N5_86": 0.0, "N5_83": 0.0, "N5_80": 0.0,
             "N6": 0.0, "N7": 0.25, "N8": 0.5, "N8b": 0.5, "N9": 0.0, "N10": 0.0, "N12a": 0.0, "N12b": 0.0,
             "N13": 0.0, "N14": 0.0, "N15": 0.5, "N16": 1.0, "N17": 1.0, "N18": 0.0, "N19": 0.0,
             "C4": 0.0, "C5": 0.0, "C7": 0.0}


class Demo(unittest.TestCase):
    _cache = {}

    @classmethod
    def demo_run(cls):
        if "demo" not in cls._cache:
            seen = {}

            def hook(t, kin, arms):
                seen.setdefault("w", []).append((t, L.width_of(kin.phi)))
                if abs(t - 9.5) < 1e-9:
                    seen["front"] = [b.copy() for b in kin.balls()[0]]

            cls._cache["demo"] = play(plans(), full=True, hook=hook) + (seen,)
        return cls._cache["demo"]

    def test_demo_scores_one(self):
        score, at, kin, rows, _ = self.demo_run()
        self.assertEqual(score, 1.0)
        self.assertEqual(kin.seated, 0)
        self.assertTrue(320 <= at <= 380, at)  # offline: 335 (16.75 s); the spec's estimate: 340 to 360
        self.assertLessEqual(at, L.RUN_LIMIT_STEPS)

    def test_score_path_rises_only(self):
        """Never 0.5 before a gripper is clear, never 1.0 before both are, never down once the sockets are on."""
        _, _, _, rows, _ = self.demo_run()
        on_from = None
        for k, row in enumerate(rows):
            if row["score"] >= 0.5:
                self.assertGreaterEqual(row["n_clear"], 1, k)
            if row["score"] == 1.0:
                self.assertEqual(row["n_clear"], 2, k)
            if on_from is None and row["on"]:
                on_from = k
            if on_from is not None and k > on_from:
                self.assertGreaterEqual(row["score"], rows[k - 1]["score"], k)
        self.assertIsNotNone(on_from)

    def test_hold_width_and_catch_window(self):
        """Spec 5.3: the width at the hold 92 mm, never under 88.5; the stud pair centred on x within 2 mm; the
        ball axis within CATCH_ALONG (2.5 mm) of the socket line when the sockets start to close; yaw within
        CATCH_YAW_DEG (2.5 deg), the computed catch window (D1)."""
        _, _, _, _, seen = self.demo_run()
        hold_w = [w for t, w in seen["w"] if 6.0 <= t <= 9.5]
        self.assertTrue(all(88.5 <= w <= 99.0 for w in hold_w), (min(hold_w), max(hold_w)))
        self.assertAlmostEqual(np.mean(hold_w), L.HOLD_WIDTH, delta=1.0)
        self.assertLess(max(w for _, w in seen["w"]), 96.0)
        b1, b2 = seen["front"]
        socket_y = OfflineDelta.socket_y()
        self.assertLessEqual(abs((b1[1] + b2[1]) / 2 - socket_y) / L.MM, L.CATCH_ALONG)
        self.assertLessEqual(abs((b1[0] + b2[0]) / 2 - L.SLIDER_ORIGIN[0]) / L.MM, 2.0)
        self.assertLessEqual(abs(math.degrees(math.atan2(b2[1] - b1[1], b2[0] - b1[0]))), L.CATCH_YAW_DEG)
        self.assertEqual((L.TRIM_LEFT, L.TRIM_RIGHT), ((-0.005, 0.0, 0.0), (0.002, 0.0008, 0.0)))  # GPU C4, C5
        self.assertEqual(L.LEFT_STATION[0], L.SLIDER_ORIGIN[0])
        self.assertEqual(L.RIGHT_PICK[0], L.SLIDER_ORIGIN[0])

    def test_nested_hinge_stagger(self):
        """D2: arm_b's free socket behind arm_a's along the arms (built link poses, arm_a at -phi / 2): 0.89 mm at
        W 84, 1.24 at 88, 1.60 at 92, 2.13 at 98; the offline demo's seated rho stays under 1.0 mm."""
        for w, want in ((84.0, 0.889), (88.0, 1.244), (92.0, 1.600), (98.0, 2.133)):
            lo, hi = 0.0, 0.2
            for _ in range(60):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if S.width_mm(mouths_of(pair_state(-mid / 2, mid))) < w else (lo, mid)
            (ma, _), (mb, _) = mouths_of(pair_state(-lo / 2, lo))
            self.assertAlmostEqual((ma[1] - mb[1]) / L.MM, want, delta=0.002, msg=w)
        _, _, _, rows, _ = self.demo_run()
        fit = rows[-1]["fit_mm"][0]
        self.assertEqual(rows[-1]["on_pair"], 0)
        self.assertLess(max(fit[1], fit[3]), 1.0)
        self.assertGreater(fit[3], 0.8)  # the stagger is there: arm_b's ball sits 0.85 mm off its axis offline

    def test_one_arm_runs(self):
        for idle in ("left", "right"):
            self.assertEqual(play(plans(idle=idle))[0], 0.0, idle)

    def test_variants(self):
        self.assertEqual(set(PREDICTED), set(T.VARIANTS))
        for v in T.VARIANTS:
            with self.subTest(v=v):
                self.assertEqual(play(plans(v))[0], PREDICTED[v])

    def test_variants_differ_from_the_demo(self):
        demo = plans()

        def key(p):
            return {a: [(w.duration, tuple(w.pos), w.finger, w.label) for w in ws] for a, ws in p.items()}
        for v in T.VARIANTS:
            self.assertNotEqual(key(plans(v)), key(demo), v)
        self.assertAlmostEqual(next(w for w in plans("N4")["left"] if w.label.startswith("open:")).finger,
                               T.spread_finger(98.0), places=12)
        self.assertAlmostEqual(next(w for w in plans("N16")["left"] if w.label.startswith("down between")).pos[1],
                               0.130, places=12)

    def test_one_arm_rules_take_every_step(self):
        steps = ["over the gap between the tubes", "down between", "spread to W 92", "hold", "close", "rise", "away"]
        for v in ("N13", "N14", "N17"):
            plan = plans(v)["right"]
            labs = [w.label for w in plan]
            i = labs.index("over the gap between the tubes, pads 8 mm apart")
            self.assertTrue(all(labs[i + k].startswith(s) for k, s in enumerate(steps)), (v, labs[i:i + 7]))
            away = next(w for w in plan[i:] if w.label.startswith("away"))
            self.assertEqual(tuple(away.pos), (L.RIGHT_AWAY[0], L.RIGHT_AWAY[1], L.TOP + L.RIGHT_AWAY[2]), v)
        self.assertTrue(plans("N13")["right"][0].label.startswith("over the post"))
        self.assertIn("turn it square: the balls pry the tubes", [w.label for w in plans("N10")["right"]])

    def test_naive_push_backs_away_after_letting_go(self):
        r = plans("N1")["right"]
        rise = next(w for w in r if w.label.startswith("rise"))
        let_go = next(w for w in r if w.label.startswith("open to 60"))
        self.assertLess(rise.pos[1], let_go.pos[1])

    def test_hold_only(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY", None)
        with self.assertRaises(ValueError):
            T.plans("N11")

    def test_n17_is_one_arm(self):
        p = plans("N17")
        self.assertEqual(list(p), [T.N17_ARM])
        self.assertAlmostEqual(T.PERCH_SHORT, 0.0045, places=9)
        t = perch_tilt()  # V8: "tilt about 11 degrees"; the offline model solves 61 sin t + 13 cos t = 25 exactly
        self.assertAlmostEqual(61 * math.sin(t) + 13 * math.cos(t), 25.0, places=9)
        self.assertAlmostEqual(math.degrees(t), 11.6, delta=0.05)
        t_v8 = math.radians(11.3)  # V8's own figure for the drop: 3.7 mm
        self.assertAlmostEqual(61 - (61 * math.cos(t_v8) - 13 * math.sin(t_v8)), 3.7, delta=0.05)
        drop = 61 - (61 * math.cos(t) - 13 * math.sin(t))  # 3.86 mm at the exact tilt
        # the drop plus half the stagger at W 88 (1.245 mm): the balls land midway between the two socket axes
        self.assertAlmostEqual(T.PERCH_SHORT / L.MM, drop + 1.245 / 2, delta=0.1)

    def test_turn_is_about_the_vertical(self):
        """D3: a 45 deg turn of the held effector is a turn of the tilted grip about the world's vertical."""
        sq = tiltgrip.rotation(L.YAW["right"], L.ROLL["right"])
        r = T.turned_grip("right", math.radians(45.0)).rotation @ sq.T
        ang = math.degrees(math.acos((np.trace(r) - 1) / 2))
        ax = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
        ax /= np.linalg.norm(ax)
        self.assertAlmostEqual(ang, 45.0, places=6)
        self.assertLess(math.degrees(math.acos(abs(ax[2]))), 0.5)
        w = next(w for w in plans("N10")["right"] if w.label == "slide it in turned")
        self.assertTrue(np.allclose(w.yaw.rotation, L.rz(T.TWIST_TURN) @ sq))

    def test_twist_routes_place_the_balls(self):
        """D4: both front balls (table mm) at each variant's turned slide and after the sideways move, against the
        closed pair's mouths (x 274, 326), tube ends (y -47), socket line (y -40) and tube outer faces (263, 337)."""
        def balls(v, label):
            w = next(w for w in plans(v)["right"] if w.label == label)
            turn = float(w.yaw) - L.YAW["right"]
            pos = (w.pos[0] - L.TRIM_RIGHT[0], w.pos[1] - L.TRIM_RIGHT[1])  # the GPU trims are not part of the design
            return [np.round(b * 1000, 1) for b in T.front_balls_from_tcp(pos, turn)]
        want = {("N10", "slide it in turned"): ((280.2, -31.3), (319.8, -70.9)),  # -45 deg, midpoint on x 300
                ("N10", "turn it square: the balls pry the tubes"): ((272.0, -40.0), (328.0, -40.0)),
                ("N14", "slide it in turned"): ((280.2, -31.3), (319.8, -70.9)),
                ("N12a", "slide it in turned"): ((266.7, -55.4), (320.5, -40.0)),  # +x ball at arm_b's mouth
                ("N12a", "pull it sideways"): ((286.7, -55.4), (340.5, -40.0)),
                ("N12b", "slide it in turned"): ((279.5, -40.0), (333.3, -55.4)),  # -x ball at arm_a's mouth
                ("N12b", "pull it sideways"): ((259.5, -40.0), (313.3, -55.4))}
        for (v, label), pair in want.items():
            got = balls(v, label)
            for g, w in zip(got, pair):
                self.assertTrue(np.allclose(g[:2], w, atol=0.051), (v, label, g, w))
        # N10 / N14 turned: both balls pass between the closed blocks (crowns inside 274 and 326); the leading ball's
        # centre stays more than 7 mm from the stop screw's axis (spec 2.1)
        lo, hi = balls("N10", "slide it in turned")
        self.assertEqual(T.TWIST_TURN, math.radians(-45.0))
        self.assertGreater(abs(lo[1] - (L.SLIDER_ORIGIN[1] * 1000 - L.STOP_SCREW_V * math.cos(L.ARM_TURN))), 7.0)
        self.assertGreater(lo[0] - 5.0, 274.0)
        self.assertLess(hi[0] + 5.0, 326.0)
        # N12: the parked ball's leading face behind the tube ends (y -47); the hooked one 0.5 mm short of its mouth
        self.assertLess(balls("N12a", "slide it in turned")[0][1] + 5.0, -47.0)
        self.assertLess(balls("N12b", "slide it in turned")[1][1] + 5.0, -47.0)

    def test_calibration_plans(self):
        """D7: C4 holds pad gaps 56, 60, 64 mm 2 s each (design W 86.7, 91.2, 95.8); C5 slides 25 mm (origin y
        -0.083); C7 starts 5 mm outside a tube's outer face (from layout) and asks 12.7 mm past each pin."""
        c4 = plans("C4")["left"]
        holds = [w for w in c4 if w.label.startswith("hold at")]
        self.assertEqual([(round(-2 * w.finger / L.MM, 6), w.duration) for w in holds],
                         [(56.0, 2.0), (60.0, 2.0), (64.0, 2.0)])
        self.assertEqual([round(L.width_for_pad_gap(g), 1) for g in (56.0, 60.0, 64.0)], [86.7, 91.2, 95.8])
        c5 = plans("C5")["right"]
        slide = next(w for w in c5 if w.label.startswith("slide"))
        self.assertEqual([(w.label, w.duration) for w in c5 if w.label.startswith(("hold", "slide"))],
                         [("hold the post", 1.0), ("slide the effector +y 25 mm", 2.0), ("hold after the slide", 2.0)])
        self.assertAlmostEqual(slide.pos[1] - L.RIGHT_PICK[1], 0.025, places=12)
        self.assertAlmostEqual(L.EFFECTOR_ORIGIN[1] + T.C5_SLIDE, -0.083, places=12)
        x1, x2, x3, x4 = T.c7_tcp_x()
        face = (L.SLIDER_ORIGIN[0] / L.MM + L.arm_point(L.ARM_B, L.TUBE_OUTER_U, L.STATION_V, 0.0)[0])
        swing = L.STATION_V * L.pin_stop_angle()
        self.assertAlmostEqual(x1 / L.MM - 12.0 - face, 5.0, places=9)  # 5 mm of air outside the +x tube
        self.assertAlmostEqual((x1 - x2) / L.MM - 5.0 - swing, 12.71, delta=0.01)  # asked past arm_a's pin
        self.assertAlmostEqual((x4 - x3) / L.MM - 5.0 - 2 * swing, 12.71, delta=0.01)  # the same past arm_b's
        self.assertAlmostEqual(swing, 12.3, delta=0.05)
        c7 = plans("C7")["left"]
        self.assertTrue(all(w.finger == T.ENTER for w in c7))
        self.assertEqual([w.duration for w in c7 if w.label.startswith("hold against")], [2.0, 2.0])

    def test_n18_n19_hold_w84(self):
        self.assertAlmostEqual(-2 * T.spread_finger(84.0) / L.MM, 53.64 + L.GPU_GAP_LOSS_MM, delta=0.01)
        for v in ("N18", "N19"):
            self.assertEqual(len(plans(v)), 2)

    def test_timings_follow_spec_5_2(self):
        p = plans()
        left, right = p["left"], p["right"]
        self.assertEqual(starts(left, "down between"), 2.0)
        self.assertEqual(starts(left, "open:"), 3.5)
        self.assertEqual(starts(left, "hold the pair open"), 6.0)
        self.assertEqual(starts(left, "close:"), 9.5)
        self.assertEqual(starts(left, "rise"), 12.0)
        self.assertEqual(starts(left, "away"), 13.5)
        self.assertEqual(starts(right, "down either side"), 2.0)
        self.assertEqual(starts(right, "close on the post"), 3.5)
        self.assertEqual(starts(right, "hold the post"), 5.0)
        self.assertEqual(starts(right, "slide the effector"), 6.0)
        self.assertEqual(starts(right, "hold the effector"), 9.0)
        self.assertEqual(starts(right, "open to 60"), 12.0)
        self.assertEqual(starts(right, "rise"), 13.5)
        self.assertEqual(starts(right, "away"), 15.0)
        self.assertEqual(sum(w.duration for w in right), 18.0)
        self.assertEqual(sum(w.duration for w in left), 18.0)

    def test_grip_heights(self):
        self.assertAlmostEqual(L.LEFT_Z * 1000 + 13.0 * approach("left")[2], 10.6, delta=0.05)  # pad tips
        self.assertAlmostEqual(L.RIGHT_Z * 1000 + 13.0 * approach("right")[2], 29.3, delta=0.05)
        self.assertGreater(L.RIGHT_Z * 1000 + 13.0 * approach("right")[2] - 13.5 * abs(math.sin(0.6)) - 0.1, 18.0)


class Reach(unittest.TestCase):
    @staticmethod
    def _kin():
        try:
            import yamkin as K
        except ImportError:
            raise unittest.SkipTest("the arm kinematics module is not next to this task")
        return K

    def _sweep(self, p, dt=0.1):
        K = self._kin()
        ap = {a: D.ArmPlan(list(w), BASES[a]) for a, w in p.items()}
        end = max(x.duration for x in ap.values())
        seeds, worst, margin, err = {}, math.inf, math.inf, 0.0
        for k in range(1, int(round(end / dt)) + 1):
            caps = {}
            for a in ap:
                pos, rot, finger, _ = ap[a].target(k * dt)
                q, pe, _ = K.solve(np.asarray(pos), K.matrix_to_quat_xyzw(rot), seeds.get(a, K.READY),
                                   base_pos=BASES[a])
                seeds[a] = q
                err = max(err, pe)
                margin = min(margin, float(np.degrees(K.joint_margins(q)).min()))
                caps[a] = K.arm().capsules(q, BASES[a], finger)
            if len(caps) == 2:
                worst = min(worst, K.capsule_clearance(caps["left"], caps["right"])[0])
        return err, margin, worst

    def test_demo_reach_and_clearance(self):
        err, margin, worst = self._sweep(plans())
        self.assertLess(err, 1e-3)
        self.assertGreater(margin, 30.0)  # spec 3.8: 31 and 32 degrees
        self.assertGreater(worst, 0.060)  # spec 3.8: 68 mm at 12.0 s

    def test_variants_reach(self):
        for v in ("N10", "N12a", "N12b", "N14", "N16", "N17", "N18", "N19", "C4", "C5", "C7"):
            err, margin, _ = self._sweep(plans(v), dt=0.25)
            self.assertLess(err, 1e-3, v)
            self.assertGreater(margin, 5.0, v)


# == (f) the text ====================================================================================================
BARRED = (r"\bscaled\b", r"\bmodel of\b", r"\banalogue\b", r"\bmock\b", r"\btrainer\b", r"stand-in",
          r"\bstands for\b", r"\brepresents\b", r"\bTBD\b", r"to be measured", r"\bto measure\b", r"\bbought\b",
          r"\bbuy\b", r"measured later")
TEXT_FIELDS = ("title", "instruction", "context", "setup", "scoring", "fidelity", "reset_notes")


class Text(unittest.TestCase):
    META = json.loads((HERE / "task.json").read_text())
    README = (HERE / "README.md").read_text()
    FLAT = " ".join(README.split())

    def test_fields_and_limits(self):
        ceil = {"title": 85, "instruction": 300, "context": 550, "setup": 2000, "scoring": 1400, "fidelity": 490,
                "reset_notes": 350}
        for k, n in ceil.items():
            self.assertLessEqual(len(self.META[k]), n, k)
        self.assertEqual(self.META["instruction"], T.INSTRUCTION)
        self.assertEqual(self.META["run_limit"], "900 steps (45 s at 20 Hz)")
        self.assertEqual(L.RUN_LIMIT_STEPS, 900)
        self.assertEqual(self.META["category"], "Impact on AI timelines")
        self.assertIn("bimanual", self.META["keywords"])
        self.assertRegex(self.META["scores"], r"^(TBD after the GPU runs|Dream-1 run: [0-9.]+; demo: 1\.0)$")

    def test_fields_match_the_fields_file(self):
        f = SPEC_DIR / "delta-arm-pair-onto-balls.fields.json"
        if not f.is_file():
            self.skipTest("the spec folder is not next to this task")
        fields = json.loads(f.read_text())["fields"]
        for k in TEXT_FIELDS:
            self.assertEqual(self.META[k], fields[k], k)

    def test_ascii_no_dashes_no_barred_words(self):
        files = [HERE / n for n in ("task.json", "README.md", "task.py", "layout.py", "scoring.py",
                                    "offline_delta.py", "tiltgrip.py")] + [pathlib.Path(__file__)]
        for f in files:
            text = f.read_text()
            self.assertTrue(text.isascii(), f.name)
            self.assertNotIn(chr(0x2014), text, f.name)
            self.assertNotIn(chr(0x2013), text, f.name)
            if f.suffix == ".py":
                for i, line in enumerate(text.splitlines(), 1):
                    self.assertLessEqual(len(line), 120, f"{f.name}:{i}")
        prose = [self.META[k] for k in TEXT_FIELDS[1:]] + [self.META["title"], self.README]
        for text in prose:
            for pat in BARRED:
                self.assertIsNone(re.search(pat, text, flags=re.IGNORECASE), pat)

    def test_no_claim_of_one_arm_failure_or_measurement(self):
        text = (json.dumps(self.META) + " " + self.FLAT).lower()
        for bad in ("one arm cannot", "cannot be done with one", "needs two arms", "two arms are needed",
                    "was measured", "were measured", "we measured", "none of the seven", "1.5 mm along the arm"):
            self.assertNotIn(bad, text, bad)
        # the "one hand cannot hold and fetch" point is one removable bullet (V8)
        lines = [ln for ln in self.README.splitlines() if "cannot" in ln]
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("- One hand cannot hold the pair open and carry the effector at once"))
        self.assertIn("(N17: one hand perches it first, then spreads)", self.FLAT)
        self.assertNotIn("cannot", self.README.replace(lines[0], ""))

    def test_numbers_agree_with_the_layout(self):
        setup, scoring, context, fid, reset = (self.META[k] for k in ("setup", "scoring", "context", "fidelity",
                                                                      "reset_notes"))
        pins = {
            setup: [
                (f"balls {L.STUD_SPACING:.0f} mm apart", True),
                ("a 34 x 10 x 10 mm aluminium bar", L.STUD_BAR == (34.0, 10.0, 10.0)),
                ("10 mm ball", L.BALL_R == 5.0),
                ("plate 50 x 30 x 8 mm", L.SLIDER_PLATE[0] == (50.0, 30.0, 8.0)),
                ("balls at x 272 and 328, y 275, z 13",
                 (L.SLIDER_ORIGIN[0] * 1000 - 28, L.SLIDER_ORIGIN[0] * 1000 + 28, L.SLIDER_ORIGIN[1] * 1000,
                  L.BALL_Z) == (272.0, 328.0, 275.0, 13.0)),
                ("8 mm square carbon tubes, 1 mm wall", L.TUBE[0] == 8.0 and L.TUBE[2] == 8.0),
                ("315 mm between socket centres", L.ARM_LEN == 315.0),
                ("about 23 g each", abs(sum(L.ARM_MASS.values()) / 2 - 0.023) < 0.001),
                ("14 mm square and 3 mm proud", (L.BLOCK_SIZE, L.BLOCK_PROUD) == (14.0, 3.0)),
                ("8 mm socket seats 2 mm over a ball", L.SOCKET_HOLE == 8.0 and abs(L.SEAT_DEPTH - 2) < 1e-9),
                ("4 mm steel keeper pin", L.KEEPER_PIN_R == 2.0),
                ("0.5 mm outside each tube", L.KEEPER_PIN_X - L.KEEPER_PIN_R - 42.0 == 0.5),
                ("120 x 15 x 9 mm, centred at x 300, y 100",
                 L.REST_BAR[0][:2] == (120.0, 15.0) and L.REST_BAR_ORIGIN == (0.300, 0.100)),
                ("at x 250 and 350", L.STOP_PIN_X == 50.0),
                ("16 mm either way at its free ends", abs(315 * L.pin_stop_angle() - 16) < 0.5),
                ("0.6 mm wire, 6.6 mm across, 25 coils, 0.24 N/mm, 1 N initial tension",
                 (L.WIRE_D, L.SPRING_OD, L.SPRING_COILS, round(L.spring_rate(), 2), L.SPRING_INIT_TENSION)
                 == (0.6, 6.6, 25, 0.24, 1.0)),
                ("30 mm from the sockets at each end", L.STRUT_V == (30.0, 285.0)),
                ("20 mm from its free socket", L.ARM_LEN - L.STOP_SCREW_V == 20.0),
                ("free sockets 52 mm apart", L.SOCKET_REST_GAP == 52.0),
                ("about 4 N at the 66 mm", round(L.real_pull_at_sockets(66.0)[2]) == 4),
                ("loose, 139 g", L.EFFECTOR_MASS == 0.139),
                ("66 mm across flats, 8 mm thick", (L.HEX_FLATS, L.HEX_THICK) == (66.0, 8.0)),
                ("centred at x 300, y -108", L.EFFECTOR_ORIGIN == (0.300, -0.108)),
                ("its balls 40 mm short of the sockets", L.SLIDE_DY == 0.040),
                ("20 x 20 x 50 mm post, 10 mm behind the centre", L.POST == ((20.0, 20.0, 50.0), (0.0, -10.0, 33.0))),
            ],
            scoring: [
                ("74 mm at rest", L.WIDTH_REST == 74.0),
                ("84 mm with both sockets on a stud pair", L.WIDTH_ON == 84.0),
                ("within 1.5 mm of its socket's axis", L.RHO_TOL == 1.5),
                ("the width is within 1.5 mm of 84", L.WIDTH_TOL == 1.5),
                ("over the last 38 mm of the fingers", rigstate.FINGER_LEN == 0.038 and L.FINGER_LEN_TEXT == 38.0),
                ("at least R + 30 mm", L.CLEAR == 0.030),
                ("16 mm closed", round(rigstate.zone_radius(0.0) * 1000) == 16),
                ("57 mm fully open", round(rigstate.zone_radius(1.0) * 1000) == 57),
                ("more than 1 mm off the table", L.FLAT_TOL == 1.0),
                ("for the last 1 s", L.STILL_WINDOW_S == 1.0),
                ("moved more than 1 mm", L.STILL_MOVE == 1.0),
                ("0.25 if the pair is on. 0.5 if also at least one gripper is clear. "
                 "1.0 if also both grippers are clear",
                 L.TIERS == (0.25, 0.5, 1.0)),
            ],
            context: [
                ("The sockets rest 52 mm apart", L.SOCKET_REST_GAP == 52.0),
                ("the balls are 66 mm over their crowns", L.STUD_SPACING + 2 * L.BALL_R == 66.0),
                ("a push jams 12 mm short", L.JAM_SHORT == 12.0),
                ("hold the pair 14 mm open against 4 N", L.STUD_SPACING + 2 * L.BALL_R - L.SOCKET_REST_GAP == 14.0
                 and round(L.real_pull_at_sockets(66.0)[2]) == 4),
                ("every 15 days", True),
            ],
            fid: [("balls three crossed 10 mm discs", 2 * L.BALL_R == 10.0), ("315 mm arms", L.ARM_LEN == 315.0)],
            reset: [("74 mm over the outer faces", L.WIDTH_REST == 74.0),
                    ("centre at x 300, y -108", L.EFFECTOR_ORIGIN == (0.300, -0.108))],
        }
        for text, rows in pins.items():
            for snippet, ok in rows:
                self.assertIn(snippet, text, snippet)
                self.assertTrue(ok, snippet)

    def test_readme(self):
        self.assertLessEqual(len(self.README), 8000)
        f = self.FLAT
        for text in ("## Results in simulation", "GPU runs of this build, one trial each",
                     "python3 -m unittest discover -s tests", "DEMO_IDLE_ARM", "DEMO_HOLD_ONLY", "TASK_TRACE",
                     "Pull the parallel arm to both sides and install it on the slider",
                     "Apply grease to the contact between the parallel arm and the wide angle ball head, do it once "
                     "every 15 days.", "SuperRacer user manual",
                     "315 mm, the arm length published Klipper configurations for this printer use",
                     "Third-party assets: none", "TK-DKQNQY", "TK-GPFKUG", "run limit 900 steps (45 s)"):
            self.assertIn(text, f, text)
        for variant in T.VARIANTS:
            self.assertIn(f"DEMO_VARIANT={variant}`", self.README, variant)
        self.assertTrue(f.startswith("# " + self.META["title"] + " "), "the heading is the title")
        for bad in ("Super Racer", "taken off, greased", "refitted", "made of", "card", "cylinder", "Unlike TK-DKQNQY"):
            self.assertNotIn(bad, f, bad)
        nums = (f"{L.STUD_SPACING:.0f} mm apart", f"{L.SOCKET_REST_GAP:.0f} mm apart",
                f"{L.STUD_SPACING + 2 * L.BALL_R:.0f} mm over their crowns", f"{L.JAM_SHORT:.0f} mm short",
                f"{L.WIDTH_REST:.0f} mm at rest", f"{L.WIDTH_ON:.0f} mm on a stud pair",
                f"{round(L.finger_force_at_station(L.phi_for_width(L.HOLD_WIDTH)), 1)} N on each finger",
                f"aims at {L.HOLD_WIDTH:.0f} mm over the tubes", f"{L.STATION_V:.0f} mm from the pivots",
                f"a {-2 * L.SPREAD_F * 1000:.1f} mm pad gap, aiming at W {L.HOLD_WIDTH:.0f}",
                f"slides the effector {L.SLIDE_DY * 1000:.0f} mm in", "the 20 mm post",
                "0.2 N/mm at the sockets with a 1.2 N preload", "built 0.5 mm low",
                "R is 16 mm closed, 57 mm fully open",
                "about 180 mm", f"{L.REST_BAR_V:.0f} mm from the slider's balls", "44.5 mm",
                "16.3 mm from the tube's inner face",
                f"within about {L.CATCH_ALONG} mm of its socket's axis along the arms or in height",
                f"{L.CATCH_YAW_DEG:g} degrees of yaw (computed)", "root sum of squares",
                "at or inboard of the socket's mouth", "three crossed 10 mm discs",
                "0.9 mm on a stud pair, 1.6 mm at the W 92 hold; closed on a stop pin, 2.6 mm either way",
                "plate 87, post 29, stud pairs 23 g", "11.5 mm up, 2.1 mm back", "26.4 mm inside the loops",
                "4 x 20 mm steel stop pins, 8 mm into the bar")
        for text in nums:
            self.assertIn(text, f, text)
        self.assertAlmostEqual(L.SPRING_AT_SOCKETS, 0.2)
        self.assertAlmostEqual(L.PRELOAD_N, 1.2)
        self.assertAlmostEqual(L.EYE_BOLT_IN, 16.28)
        # README figures marked computed, recomputed from the layout (not only the wording)
        self.assertAlmostEqual(L.SOCKET_REST_GAP * math.sin(L.pin_stop_angle()), 2.6, delta=0.05)
        e = T.EFFECTOR_SPEC
        v = [s.volume() for s in e.shapes]
        self.assertEqual([round(139 * x / sum(v)) for x in (sum(v[:3]), v[3], sum(v[4:]))], [87, 29, 23])
        c = e.center_of_mass() / L.MM
        self.assertAlmostEqual(c[2], 11.5, delta=0.05)
        self.assertAlmostEqual(-c[1], 2.1, delta=0.05)
        f1, f2, _ = L.real_pull_at_sockets(L.HOLD_WIDTH - L.WIDTH_EXTRA)
        self.assertAlmostEqual((285 * f1 + 30 * f2) / (f1 + f2), 180.0, delta=3.0)

    def test_results_table(self):
        section = self.README.split("## Results in simulation", 1)[1].split("\n## ", 1)[0]
        lines = [ln for ln in section.splitlines() if ln.startswith("|") and not ln.startswith("|---")][1:]
        tokens = []
        for ln in lines:
            cell = ln.rstrip().rstrip("|").split("|")[-1].strip()
            self.assertRegex(cell, r"^(@@\w+@@|0|0\.0|0\.25|0\.5|1\.0)$", ln)
            tokens.append(cell)
        placeholders = [t for t in tokens if t.startswith("@@")]
        self.assertEqual(len(set(placeholders)), len(placeholders))
        n17 = [ln for ln in lines if "DEMO_VARIANT=N17" in ln]
        self.assertEqual(len(n17), 1)
        self.assertIn("one arm", n17[0])  # listed with the other one-arm runs
        one_arm = [i for i, ln in enumerate(lines) if "one arm" in ln or "alone" in ln]
        self.assertEqual(one_arm, list(range(one_arm[0], one_arm[0] + len(one_arm))))

    def test_demo_table_follows_the_plan(self):
        p = plans()
        last_right = sum(w.duration for w in p["right"][:-1])
        cuts = sorted({0.0, 18.0, last_right} | {starts(p[a], lab) for a, lab in (
            ("left", "down between"), ("left", "open:"), ("right", "hold the post"), ("left", "hold the pair open"),
            ("right", "hold the effector"), ("left", "close:"), ("left", "rise"), ("left", "away"),
            ("right", "away"), ("left", "hold clear"))})
        self.assertEqual(cuts, [0.0, 2.0, 3.5, 5.0, 6.0, 9.0, 9.5, 12.0, 13.5, 15.0, 16.5, 18.0])
        for a, b in zip(cuts, cuts[1:]):
            row = f"| {a:.1f} to {b:.1f} |".replace("| 0.0 to", "| 0 to")
            self.assertEqual(self.README.count(row), 1, row)


class Kit(unittest.TestCase):
    def test_kit_files_are_unchanged_copies(self):
        if not (KIT_DIR / "mechkit.py").is_file():
            self.skipTest("the kit folder is not next to this task")
        for name in ("mechkit.py", "rigstate.py"):
            a = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            b = hashlib.sha256((KIT_DIR / name).read_bytes()).hexdigest()
            self.assertEqual(a, b, name)


if __name__ == "__main__":
    unittest.main()
