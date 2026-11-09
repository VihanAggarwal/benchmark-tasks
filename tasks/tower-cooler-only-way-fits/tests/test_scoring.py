"""Offline tests for scoring.py, the layout's geometry and task.py: synthetic cooler states and synthetic arm
snapshots fed through rigstate.offline_feed, no simulator.  python3 -m unittest discover -s tests -v"""

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
    Tracker, airflow_err_deg, base_off, foot_gap, frame_margin, over_board, released, right_way_round, rotation,
    seated, tilt_deg, valid, zone_gap,
)

HZ = 20.0
DT = 1.0 / HZ
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
TIPS = 0.013  # pad tips below the TCP
GRIP_UP = 0.050  # the demo's TCP above the base's bottom face
OPEN, GRIP = 1.0, (L.UPSTREAM + L.DOWNSTREAM - 0.008) / L.FULL_GAP  # rigstate openings: open, closed on fan + fins
CPU_W = L.TOP + L.CPU_TOP
SEAT = np.array([L.CPU_X, L.CPU_Y, CPU_W])  # the base's bottom-face centre, seated
START = np.array(L.START)
READY_L = np.array([0.323, 0.305, L.TOP + 0.26])
READY_R = np.array([0.323, -0.305, L.TOP + 0.26])
CARRY = L.TOP + 0.11
SHAPES = [(L.BASE_CENTER, L.BASE_SIZE), (L.RISER_CENTER, L.RISER_SIZE), (L.FAN_CENTER, L.FAN_SIZE),
          (L.FINS_CENTER, L.FINS_SIZE)]
_SIGNS = np.array(list(itertools.product((-1, 1), repeat=3)), dtype=float)


def quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """(x, y, z, w) of R = Rz(yaw) Ry(pitch) Rx(roll): the cooler turned yaw about the vertical (airflow from +x)
    after pitching (pitch > 0 tips its fins' end down) and rolling about its airflow axis."""
    cy, sy = math.cos(math.radians(yaw_deg) / 2), math.sin(math.radians(yaw_deg) / 2)
    cp, sp = math.cos(math.radians(pitch_deg) / 2), math.sin(math.radians(pitch_deg) / 2)
    cr, sr = math.cos(math.radians(roll_deg) / 2), math.sin(math.radians(roll_deg) / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def cooler(origin=SEAT, yaw=0.0, pitch=0.0, roll=0.0, vel=(0.0, 0.0, 0.0), spin=(0.0, 0.0, 0.0)):
    """A cooler state as mechkit reports it: base-frame origin and orientation, and the world box round its four
    boxes."""
    o = np.asarray(origin, dtype=float)
    q = quat(yaw, pitch, roll)
    r = rotation(q)
    pts = np.vstack([o + (np.asarray(c) + _SIGNS * np.asarray(s) / 2) @ r.T for c, s in SHAPES])
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=(lo + hi) / 2, lin_vel=np.asarray(vel, float),
                           ang_vel=np.asarray(spin, float), low=lo, high=hi)


def snap(left_tcp=READY_L, left_open=OPEN, right_tcp=READY_R, right_open=OPEN):
    """Feed one synthetic arm snapshot through rigstate, as the offline checker does, and return rigstate.arms()."""
    feed = {}
    for side, tcp, opening in (("left", left_tcp, left_open), ("right", right_tcp, right_open)):
        tcp = np.array(tcp, float)
        feed[side] = {"tcp": tcp, "tips": tcp - np.array([0.0, 0.0, TIPS]), "opening": float(opening)}
    rigstate.offline_feed(feed)
    return rigstate.arms()


def grip_tcp(origin, yaw):
    """Where the demo's TCP is for a cooler with its base at ``origin``, airflow at ``yaw`` degrees."""
    a = math.radians(yaw)
    return np.asarray(origin, float) + np.array([L.STACK_MID * math.cos(a), L.STACK_MID * math.sin(a), GRIP_UP])


class Film:
    """A trial as a list of frames: the cooler's base origin and orientation, both TCPs and openings. ``go`` moves
    linearly to new values over n control steps; velocities come from the frame-to-frame motion."""

    KEYS = ("o", "yaw", "pitch", "roll", "L", "Lo", "R", "Ro")

    def __init__(self, **start):
        self.cur = {"o": START, "yaw": 90.0, "pitch": 0.0, "roll": 0.0, "L": READY_L, "Lo": OPEN, "R": READY_R,
                    "Ro": OPEN}
        self.cur.update({k: np.array(v, float) if k in ("o", "L", "R") else v for k, v in start.items()})
        self.frames = [dict(self.cur)]

    def go(self, n, **to):
        a = dict(self.cur)
        b = dict(a)
        b.update({k: np.array(v, float) if k in ("o", "L", "R") else v for k, v in to.items()})
        for i in range(1, n + 1):
            f = i / n
            self.frames.append({k: (1 - f) * np.asarray(a[k], float) + f * np.asarray(b[k], float)
                                if k in ("o", "L", "R") else (1 - f) * a[k] + f * b[k] for k in self.KEYS})
        self.cur = dict(self.frames[-1])
        return self

    def hold(self, n):
        return self.go(n)

    def carry(self, n, o, yaw=None, **arms):
        """Move the held cooler (and the left TCP with it) to base origin ``o``, turning it to ``yaw``."""
        yaw = self.cur["yaw"] if yaw is None else yaw
        return self.go(n, o=o, yaw=yaw, L=grip_tcp(o, yaw), **arms)


def play(film, tracker=None, arms=True, until=None):
    """Feed a film to a tracker (fresh unless given); arms through rigstate.offline_feed (or None)."""
    t = tracker or Tracker(HZ)
    r = None
    prev = None
    for fr in film.frames[:until] if until else film.frames:
        v = np.zeros(3) if prev is None else (fr["o"] - prev["o"]) / DT
        a = snap(fr["L"], fr["Lo"], fr["R"], fr["Ro"]) if arms else None
        r = t.update(cooler(fr["o"], fr["yaw"], fr["pitch"], fr["roll"], v), a)
        prev = fr
    return r, t


def pick(f):
    """The left arm comes down over the cooler on its pad and grips it across fan and fins (about 6 s)."""
    g = grip_tcp(START, 90.0)
    f.go(30, L=g + [0, 0, 0.06]).go(30, L=g).go(30, Lo=GRIP)
    return f


def mount(at=SEAT, yaw=0.0, pitch=0.0, roll=0.0, open_to=OPEN, settle=40, rise=True):
    """The intended trial: pick, lift, carry turning the airflow to ``yaw``, lower the base to ``at`` (finally
    pitched/rolled as given, e.g. resting crooked), open to ``open_to``, rise unless told not to, hold still."""
    f = pick(Film().hold(10))
    above = np.array([*START[:2], CARRY - GRIP_UP])
    f.carry(30, above)
    over = np.array([at[0], at[1], CARRY - GRIP_UP])
    f.carry(50, over, yaw)
    f.carry(20, np.asarray(at, float) + [0, 0, 0.030]).carry(50, np.asarray(at, float))
    if pitch or roll:
        f.go(5, pitch=pitch, roll=roll)
    f.go(30, Lo=open_to)
    if rise:
        f.go(20, L=f.cur["L"] + [0, 0, 0.06])
    return f.hold(settle)


class Geometry(unittest.TestCase):
    """The layout does what the task says: only one way round fits, and the cooler stands on its base."""

    OBSTACLES = None

    @classmethod
    def setUpClass(cls):
        def rect(x, y, size):
            return (x - size[0] / 2, x + size[0] / 2, y - size[1] / 2, y + size[1] / 2)
        cls.OBSTACLES = {**{f"memory_{i + 1}": rect(x, L.CPU_Y, L.MEM_SIZE) for i, x in enumerate(L.MEM_X)},
                         "vrm_heatsink": rect(L.VRM_X, L.VRM_Y, L.VRM_SIZE),
                         "graphics_card": rect(L.GPU_X, L.GPU_Y, L.GPU_SIZE),
                         "io_shield": rect(L.IO_X, L.IO_Y, L.IO_SIZE)}

    @staticmethod
    def stack(yaw_deg, dx, dy):
        """Fan plus fins seen from above (4 x 2), the base's centre at the processor's centre + (dx, dy)."""
        a = math.radians(yaw_deg)
        c, s = math.cos(a), math.sin(a)
        pts = [(-L.UPSTREAM, -L.STACK_W / 2), (L.DOWNSTREAM, -L.STACK_W / 2), (L.DOWNSTREAM, L.STACK_W / 2),
               (-L.UPSTREAM, L.STACK_W / 2)]
        return np.array([(L.CPU_X + dx + c * x - s * y, L.CPU_Y + dy + s * x + c * y) for x, y in pts])

    @staticmethod
    def hits(poly, r):
        """Does a convex polygon overlap the rectangle (x0, x1, y0, y1)? (separating axes)"""
        x0, x1, y0, y1 = r
        rect = np.array([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])
        axes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        for i in range(len(poly)):
            e = poly[(i + 1) % len(poly)] - poly[i]
            axes.append(np.array([-e[1], e[0]]))
        return all(not ((poly @ a).max() <= (rect @ a).min() or (rect @ a).max() <= (poly @ a).min()) for a in axes)

    def fits(self, yaw_deg, dx, dy):
        """Seated with its base there and turned so: the base inside the frame and the stack clear of everything."""
        st = cooler(SEAT + [dx, dy, 0.0], yaw_deg)
        if frame_margin(st) < 0.0:
            return False
        poly = self.stack(yaw_deg, dx, dy)
        return not any(self.hits(poly, r) for r in self.OBSTACLES.values())

    def test_obstacles_are_taller_than_the_seated_stack_foot(self):
        foot = L.CPU_TOP + L.FOOT_UP  # 28 mm
        for top in (L.MEM_TOP, L.VRM_TOP, L.GPU_TOP, L.IO_TOP):
            self.assertGreater(top - foot, 0.0015)
        self.assertAlmostEqual(L.MEM_TOP - foot, 0.006)  # the memory's top is 6 mm above the seated stack's foot
        self.assertGreater(L.CPU_TOP, L.FRAME_TOP)  # the base sits on the lid, not on the frame
        self.assertGreater(L.CPU_HALF, L.BASE_HALF)  # the whole base is on the lid when centred

    def test_only_one_way_round_fits_anywhere_in_the_frame(self):
        shifts = np.arange(-0.006, 0.00601, 0.001)
        feasible = [d for d in range(0, 360) if any(self.fits(d, dx, dy) for dx in shifts for dy in shifts)]
        wrapped = sorted((d + 180) % 360 - 180 for d in feasible)
        self.assertEqual(wrapped, list(range(-15, 16)))  # exactly the scorer's 15 degree airflow window

    def test_the_right_way_fits_wherever_the_base_is_inside_the_frame(self):
        for dx, dy in itertools.product(np.linspace(-0.0049, 0.0049, 11), repeat=2):
            self.assertTrue(self.fits(0.0, dx, dy), (dx, dy))
        poly = self.stack(0.0, 0.0, 0.0)
        mem_face = L.CPU_X - L.MEM_INNER
        self.assertAlmostEqual(poly[:, 0].min() - mem_face, 0.009, places=6)  # 9 mm clear of the memory, centred
        self.assertAlmostEqual(L.VRM_INNER - L.STACK_W / 2, 0.009, places=6)
        self.assertAlmostEqual(L.GPU_INNER - L.STACK_W / 2, 0.009, places=6)

    def test_every_other_way_lands_on_the_memory_by_at_least_5_mm(self):
        for yaw in (90.0, 180.0, 270.0):
            for dx, dy in itertools.product(np.linspace(-0.005, 0.005, 11), repeat=2):
                poly = self.stack(yaw, dx, dy)
                self.assertTrue(self.hits(poly, self.OBSTACLES["memory_1"]), (yaw, dx, dy))
                self.assertGreaterEqual((L.CPU_X - L.MEM_INNER) - poly[:, 0].min(), 0.005 - 1e-9)

    def test_centre_of_mass_inside_the_base(self):
        sys.path.insert(0, str(_stub_dir()))
        import mechkit as mk
        spec = mk.assembly("c", origin=L.START, yaw=L.START_YAW, mass=L.COOLER_MASS,
                           shapes=[mk.box(c, s) for c, s in SHAPES])
        com = spec.center_of_mass()
        self.assertAlmostEqual(com[0], 0.00895, delta=0.0002)
        self.assertAlmostEqual(com[2], 0.0428, delta=0.0003)
        tip = math.degrees(math.atan2(L.BASE_HALF - com[0], com[2]))
        self.assertGreater(tip, 10.0)  # it stands on its base (tips only past about 10.6 degrees)
        lo, hi = spec.local_bounds()
        self.assertTrue(np.allclose(lo, (-L.UPSTREAM, -L.STACK_W / 2, 0.0)))
        self.assertTrue(np.allclose(hi, (L.DOWNSTREAM, L.STACK_W / 2, L.COOLER_H)))

    def test_start_is_on_the_pad_and_off_the_board(self):
        c = cooler(START, 90.0)
        self.assertFalse(over_board(c))
        self.assertAlmostEqual(float(c.low[2]), L.TOP + L.PAD_TOP)
        self.assertTrue(abs(c.low[0] - L.PAD_X) < L.PAD_SIZE[0] / 2 and abs(c.high[0] - L.PAD_X) < L.PAD_SIZE[0] / 2)
        self.assertGreater(float(c.low[1]), L.PAD_Y - L.PAD_SIZE[1] / 2)
        self.assertEqual(round(airflow_err_deg(c)), 90)
        # nothing in the start pose's shadow on the setup screenshot (|y| > 0.45 m with x < 0.20 m)
        self.assertLess(float(c.high[1]), 0.45)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = play(Film().hold(20))
        self.assertEqual((r.score, r.success), (0.0, False))
        self.assertEqual(r.details["zero_reason"], "not_on_the_board")

    def test_full_mount_scores_one(self):
        r, t = play(mount())
        self.assertEqual((r.score, r.success), (1.0, True))
        self.assertTrue(r.details["seated"] and r.details["let_go"] and r.details["at_rest"])
        self.assertEqual(t.arms_unavailable, 0)

    def test_success_appears_only_once_let_go_and_still(self):
        f = mount(open_to=GRIP, rise=False, settle=15)
        r, t = play(f)  # set down and still, still holding: 0.5, no success
        self.assertEqual((r.score, r.success), (0.5, False))
        opening = Film(**{k: v for k, v in f.cur.items()})
        opening.go(30, Lo=OPEN)
        for fr in opening.frames[1:]:
            r = t.update(cooler(fr["o"], fr["yaw"]), snap(fr["L"], fr["Lo"]))
            self.assertEqual(r.success, fr["Lo"] >= L.GRIPPER_OPEN, fr["Lo"])
        self.assertEqual((r.score, r.success), (1.0, True))

    def test_seated_but_still_held_scores_half(self):
        r, _ = play(mount(open_to=GRIP, rise=False))
        self.assertEqual((r.score, r.success), (0.5, False))
        self.assertTrue(r.details["seated"])
        self.assertFalse(r.details["let_go"])

    def test_a_gripper_closed_away_from_the_cooler_has_let_go(self):
        f = mount(open_to=OPEN)
        f.go(20, L=SEAT + [0.0, 0.15, 0.10]).go(10, Lo=0.0).hold(15)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)

    def test_offsets_inside_the_frame_score_one(self):
        for dx, dy in ((0.004, 0.0), (-0.004, 0.0), (0.0, 0.0045), (0.003, -0.003)):
            r, _ = play(mount(at=SEAT + [dx, dy, 0.0]))
            self.assertEqual(r.score, 1.0, (dx, dy))

    def test_base_partly_over_the_frame_scores_half(self):
        for dx, dy in ((0.0055, 0.0), (0.0, -0.006), (0.009, 0.0)):
            r, _ = play(mount(at=SEAT + [dx, dy, 0.0]))
            self.assertEqual(r.score, 0.5, (dx, dy))
            self.assertLess(r.details["frame_margin_mm"], 0.0)

    def test_airflow_tolerance(self):
        r, _ = play(mount(yaw=12.0))
        self.assertEqual(r.score, 1.0)
        r, _ = play(mount(yaw=-20.0))  # beyond 15 degrees it is turned another way round: 0.25, not 0.5
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.details["right_way"])
        r, _ = play(mount(at=SEAT + [0, 0, 0.0025], yaw=-14.0))  # raised, but the right way round
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["right_way"])

    def test_wrong_way_resting_on_the_memory_scores_quarter(self):
        # Turned airflow-toward-the-arms (or sideways), the fins' foot lands on the memory's top 6 mm above the
        # seated height and the cooler leans back onto the lid: tilted, raised, not seated, the wrong way round.
        for yaw, pitch, roll in ((180.0, -9.0, 0.0), (90.0, 0.0, 0.0), (-90.0, 0.0, 9.0)):
            at = SEAT + [0.0, 0.0, 0.004] if not (pitch or roll) else SEAT + [0.001, 0.0, 0.003]
            r, _ = play(mount(at=at, yaw=yaw, pitch=pitch, roll=roll))
            self.assertEqual(r.score, 0.25, (yaw, pitch, roll))
            self.assertFalse(r.details["seated"] or r.details["right_way"])

    def test_right_way_with_the_fan_on_the_memory_scores_half(self):
        # The right way round but set down 12 mm toward the arms: the fan's foot rests on the memory's top and the
        # cooler leans forward onto the lid, about 10 degrees, base raised.
        r, _ = play(mount(at=SEAT + [-0.012, 0.0, 0.004], pitch=10.0))
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["right_way"])
        self.assertFalse(r.details["seated"])

    def test_raised_or_tilted_on_the_lid_scores_half(self):
        r, _ = play(mount(at=SEAT + [0, 0, 0.0025]))
        self.assertEqual(r.score, 0.5)
        r, _ = play(mount(pitch=3.0))
        self.assertEqual(r.score, 0.5)
        r, _ = play(mount(at=SEAT + [0, 0, 0.0015], pitch=1.5))
        self.assertEqual(r.score, 1.0)

    def test_on_the_board_away_from_the_processor_scores_quarter(self):
        for at in ((0.245, 0.070), (0.36, -0.03), (0.30, 0.075)):
            r, _ = play(mount(at=(at[0], at[1], L.TOP + L.BOARD_TOP)))
            self.assertEqual(r.score, 0.25, at)
        r, _ = play(mount(at=(L.CPU_X + 0.019, L.CPU_Y + 0.010, CPU_W + 0.020), pitch=12.0))  # 21.5 mm off
        self.assertEqual(r.score, 0.25)

    def test_near_tier_boundary(self):
        r, _ = play(mount(at=(L.CPU_X + 0.0195, L.CPU_Y, CPU_W + 0.004), pitch=8.0))
        self.assertEqual(r.score, 0.5)
        r, _ = play(mount(at=(L.CPU_X + 0.0205, L.CPU_Y, CPU_W + 0.004), pitch=8.0))
        self.assertEqual(r.score, 0.25)

    def test_tipped_over_scores_zero(self):
        for pitch, roll in ((35.0, 0.0), (90.0, 0.0), (0.0, 90.0), (180.0, 0.0)):
            r, _ = play(mount(at=SEAT + [0.0, 0.0, 0.02], pitch=pitch, roll=roll))
            self.assertEqual(r.score, 0.0, (pitch, roll))
            self.assertEqual(r.details["zero_reason"], "tipped_over")
        r, _ = play(mount(at=SEAT + [0.0, 0.0, 0.005], pitch=29.0))
        self.assertEqual(r.score, 0.5)

    def test_held_in_the_air_scores_zero(self):
        f = pick(Film().hold(10))
        f.carry(40, np.array([*START[:2], CARRY - GRIP_UP])).carry(50, SEAT + [0, 0, 0.03], 0.0).hold(30)
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "still_held"))
        f.carry(20, SEAT + [0, 0, 0.0045]).hold(20)  # held 4.5 mm above the lid: not seated either
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_set_down_on_the_table_scores_zero(self):
        for at in ((0.18, 0.10), (0.30, 0.12), (0.42, 0.0)):
            r, _ = play(mount(at=(at[0], at[1], L.TOP)))
            self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "not_on_the_board"), at)

    def test_still_moving_scores_zero(self):
        f = mount(settle=0)
        f.go(6, o=SEAT + [0.006, 0.0, 0.0])  # sliding at 2 cm/s at the end
        r, _ = play(f)
        self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "still_moving"))
        f.hold(WINDOW + 1)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)  # 6 mm out of the frame's 5 mm

    def test_first_half_second_scores_zero(self):
        t = Tracker(HZ)
        a = snap()
        for i in range(WINDOW):
            r = t.update(cooler(SEAT), a)
            self.assertEqual((r.score, r.details["zero_reason"]), (0.0, "not_yet_seen_0.5_s"))
        self.assertEqual(t.update(cooler(SEAT), a).score, 1.0)

    def test_score_is_idempotent_and_matches_success(self):
        _, t = play(mount())
        c, a = cooler(SEAT + [0.0015, -0.0015, 0.0]), snap()
        r1, r2 = t.result(c, a), t.result(c, a)
        self.assertEqual((r1.score, r1.success), (r2.score, r2.success))
        for s in (0.0, 0.25, 0.5):
            self.assertNotEqual(s, 1.0)
        self.assertTrue(r1.success == (r1.score == 1.0))

    def test_every_outcome_maps_to_one_of_the_tiers(self):
        rng = np.random.default_rng(7)
        seen = set()
        for _ in range(400):
            o = np.array([rng.uniform(0.1, 0.45), rng.uniform(-0.15, 0.3), L.TOP + rng.uniform(-0.02, 0.1)])
            c = cooler(o, rng.uniform(-180, 180), rng.uniform(-60, 60), rng.uniform(-60, 60))
            t = Tracker(HZ)
            opening = float(rng.choice([0.2, 0.5, 1.0]))
            tcp = grip_tcp(o, 0.0) if rng.random() < 0.5 else READY_L
            for _ in range(WINDOW + 1):
                r = t.update(c, snap(tcp, opening))
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            self.assertEqual(r.success, r.score == 1.0)
            seen.add(r.score)
        for _ in range(400):  # near the processor, any heading, a little tilt or height: where 0.5 and 0.25 meet
            o = SEAT + [rng.uniform(-0.025, 0.025), rng.uniform(-0.025, 0.025), rng.uniform(0.0, 0.008)]
            c = cooler(o, rng.uniform(-180, 180), rng.uniform(-12, 12), rng.uniform(-12, 12))
            t = Tracker(HZ)
            for _ in range(WINDOW + 1):
                r = t.update(c, snap(READY_L, 1.0))
            self.assertIn(r.score, (0.25, 0.5, 1.0))
            self.assertEqual(r.score == 0.5, r.details["right_way"] and base_off(c) <= L.NEAR
                             and not r.details["seated"])
            if airflow_err_deg(c) > L.AIRFLOW_TOL_DEG:
                self.assertEqual(r.score, 0.25)  # the wrong way round never earns more than 0.25
            seen.add(r.score)
        self.assertTrue({0.0, 0.25, 0.5} <= seen)


class Exploits(unittest.TestCase):
    def test_holding_it_on_the_lid_never_scores_one(self):
        f = mount(open_to=GRIP, rise=False, settle=200)
        r, _ = play(f)
        self.assertEqual(r.score, 0.5)

    def test_wrong_way_shifted_until_it_clears_the_memory(self):
        # Airflow toward the arms needs the base 10 mm or more toward the exhaust before the fins clear the memory:
        # flat on the lid there, but out of the frame and the wrong way round.
        r, _ = play(mount(at=SEAT + [0.0105, 0.0, 0.0], yaw=180.0))
        self.assertEqual(r.score, 0.25)
        self.assertLess(r.details["frame_margin_mm"], 0.0)

    def test_wrong_way_flat_and_centred_is_not_seated(self):
        # Physically impossible (the fins would be inside the memory), and the airflow rule refuses it anyway.
        for yaw in (180.0, 90.0, -90.0, 45.0):
            r, _ = play(mount(yaw=yaw))
            self.assertEqual(r.score, 0.25, yaw)
            self.assertFalse(r.details["seated"])

    def test_ignoring_which_way_round_earns_at_most_a_quarter(self):
        # Set down on or by the processor at any heading outside the 15 degree window, flat or leaning on the
        # memory: never more than 0.25, however well centred.
        for yaw in (20.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0, -20.0):
            for at, pitch in ((SEAT, 0.0), (SEAT + [0.0, 0.0, 0.004], 8.0), (SEAT + [0.004, -0.003, 0.0], 0.0)):
                r, _ = play(mount(at=at, yaw=yaw, pitch=pitch))
                self.assertEqual(r.score, 0.25, (yaw, pitch))

    def test_cooler_on_top_of_the_memory(self):
        r, _ = play(mount(at=(np.mean(L.MEM_X), L.CPU_Y, L.TOP + L.MEM_TOP)))  # balanced on the modules: 33 mm off
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.details["seated"])

    def test_pushed_off_the_pad_onto_the_table(self):
        f = Film().hold(10).go(20, L=START + [0.0, 0.10, 0.04]).go(30, L=START + [0.0, 0.02, 0.04],
                                                                   o=START + [0.0, -0.06, -L.PAD_TOP]).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 0.0)

    def test_dropped_from_high_lands_scored_by_where_it_ends(self):
        f = pick(Film().hold(10))
        f.carry(40, np.array([*START[:2], CARRY - GRIP_UP])).carry(50, SEAT + [0, 0, 0.04], 0.0)
        f.go(5, Lo=OPEN).go(4, o=SEAT).hold(30)
        r, _ = play(f)
        self.assertEqual(r.score, 1.0)  # a drop that happens to land seated is seated (said in the README)


class Arms(unittest.TestCase):
    def test_arms_unavailable_taken_as_let_go_and_counted(self):
        r, t = play(mount(), arms=False)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(t.arms_unavailable, len(mount().frames))
        self.assertFalse(r.details["arms_available"])

    def test_non_finite_or_missing_arms_count_as_unavailable(self):
        t = Tracker(HZ)
        bad = {"left": SimpleNamespace(tcp=np.full(3, np.nan), tips=np.zeros(3), opening=1.0),
               "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3), opening=1.0)}
        t.update(cooler(SEAT), bad)
        t.update(cooler(SEAT), {"left": bad["right"]})
        t.update(cooler(SEAT), {"left": bad["right"], "right": SimpleNamespace(tcp=np.zeros(3), tips=np.zeros(3),
                                                                               opening=float("inf"))})
        self.assertEqual(t.arms_unavailable, 3)

    def test_either_gripper_holding_counts(self):
        c = cooler(SEAT)
        held = grip_tcp(SEAT, 0.0)
        self.assertFalse(released(snap(left_tcp=held, left_open=GRIP), c))
        self.assertFalse(released(snap(right_tcp=held, right_open=GRIP), c))
        self.assertTrue(released(snap(left_tcp=held, left_open=0.92), c))
        self.assertTrue(released(snap(left_tcp=held + [0, 0, 0.12], left_open=GRIP), c))
        self.assertLess(zone_gap(snap(left_tcp=held, left_open=GRIP)["left"], c.low, c.high), 0.0)

    def test_offline_feed_reaches_the_scorer(self):
        a = snap(left_tcp=(0.1, 0.2, 0.9), left_open=0.25, right_tcp=(0.1, -0.2, 0.9), right_open=0.75)
        self.assertIs(a, rigstate.arms())
        t = Tracker(HZ)
        r = t.update(cooler(START, 90.0), rigstate.arms())
        self.assertEqual(r.details["opening"], {"left": 0.25, "right": 0.75})
        self.assertTrue(r.details["arms_available"])


class Robustness(unittest.TestCase):
    def test_pose_functions(self):
        c = cooler(SEAT + [0.0012, -0.0016, 0.001], yaw=10.0, pitch=1.0)
        self.assertAlmostEqual(foot_gap(c), 0.001)
        self.assertAlmostEqual(base_off(c), 0.002)
        self.assertAlmostEqual(airflow_err_deg(c), 10.0, places=3)
        self.assertAlmostEqual(tilt_deg(c), 1.0, places=6)
        self.assertTrue(seated(c))
        self.assertAlmostEqual(airflow_err_deg(cooler(SEAT, yaw=-170.0)), 170.0, places=3)
        self.assertEqual(airflow_err_deg(cooler(SEAT, pitch=90.0)), 180.0)
        self.assertAlmostEqual(tilt_deg(cooler(SEAT, roll=180.0)), 180.0, places=6)
        self.assertAlmostEqual(frame_margin(cooler(SEAT)), L.FRAME_OPEN_HALF - L.BASE_HALF)
        self.assertTrue(right_way_round(cooler(SEAT, yaw=-14.9)))
        self.assertTrue(right_way_round(cooler(SEAT, yaw=14.9, pitch=20.0)))  # leaning: its heading seen from above
        self.assertFalse(right_way_round(cooler(SEAT, yaw=15.2)) or right_way_round(cooler(SEAT, yaw=180.0)))

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = cooler(SEAT)
        nan.origin = np.full(3, np.nan)
        inf = cooler(SEAT)
        inf.low = np.array([0.0, 0.0, -np.inf])
        zero_q = cooler(SEAT)
        zero_q.quat_xyzw = np.zeros(4)
        no_origin = SimpleNamespace(**{k: v for k, v in vars(cooler(SEAT)).items() if k != "origin"})
        short = cooler(SEAT)
        short.origin = np.zeros(2)
        frames = mount().frames
        for bad in (nan, inf, zero_q, no_origin, short):
            self.assertFalse(valid(bad))
            f = Film()
            f.frames = frames[:-25]
            _, t = play(f)
            before = (len(t.history), t.prev_origin.copy(), t.arms_unavailable)
            a = snap()
            for _ in range(3):
                r = t.update(bad, a)
                self.assertEqual((r.score, r.success), (0.0, False))
                self.assertEqual(r.details["zero_reason"], "invalid_state")
                self.assertEqual(t.result(bad, a).score, 0.0)
            self.assertEqual(len(t.history), before[0])
            self.assertTrue(np.allclose(t.prev_origin, before[1]))
            self.assertEqual(t.bad_steps, 3)
            rest = Film()
            rest.frames = frames[-25:]
            r, _ = play(rest, t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [cooler((5.0, -3.0, -1.0)), cooler((0.3, 0.0, 0.0), roll=120.0), cooler((-0.2, 0.6, L.TOP), pitch=90.0),
               cooler(SEAT, pitch=89.9), cooler((0.3, 0.0, 2.0), vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        for c in odd + odd[::-1]:
            r = t.update(c, snap())
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            self.assertFalse(r.success)
        r = t.update(cooler(START, 90.0), None)
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def mounted(self):
        _, t = play(mount())
        self.assertEqual(t.last.score, 1.0)
        return t

    def test_teleport_back_to_start_resets(self):
        t = self.mounted()
        r = t.update(cooler(START, 90.0), snap())  # back on the pad in one step: a new episode
        self.assertEqual(t.steps, 1)
        self.assertEqual(len(t.history), 1)
        self.assertEqual(r.score, 0.0)
        f = mount()
        f.frames = f.frames[1:]
        r, _ = play(f, t)
        self.assertEqual(r.score, 1.0)

    def test_teleport_to_a_jittered_start_resets(self):
        t = self.mounted()
        t.update(cooler(START + [0.02, -0.025, 0.0], 90.0), snap())
        self.assertEqual(t.steps, 1)

    def test_gradual_return_does_not_reset(self):
        t = self.mounted()
        steps = t.steps
        f = Film(o=SEAT, yaw=0.0, L=READY_L)
        f.go(60, o=SEAT + [0, 0, 0.06]).go(60, o=[*START[:2], START[2] + 0.06], yaw=90.0).go(60, o=START)
        play(f, t)
        self.assertEqual(t.steps, steps + len(f.frames))

    def test_a_jump_elsewhere_is_not_a_reset(self):
        t = self.mounted()
        steps = t.steps
        t.update(cooler((0.25, 0.10, L.TOP), 30.0), snap())  # knocked far, not to the start
        self.assertEqual(t.steps, steps + 1)

    def test_rigstate_reset_hook_resets_the_tracker(self):
        t = self.mounted()
        saved = dict(rigstate._RESET_CALLBACKS)
        rigstate.on_reset(t.reset)
        try:
            ep = rigstate.episode()
            rigstate.offline_reset()  # what rigstate's wrapped reset runs after the embodiment's reset
            self.assertEqual(rigstate.episode(), ep + 1)
            self.assertIsNone(rigstate.arms())
            self.assertEqual((t.steps, len(t.history), t.arms_unavailable), (0, 0, 0))
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

    def objects(self, c=None):
        return {"cooler": c if c is not None else cooler(START, 90.0)}

    def test_objects(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        loose = [o for o in self.task.OBJECTS if getattr(o, "mass", None)]
        self.assertEqual([o.name for o in loose], ["cooler"])
        spec = self.task.COOLER
        self.assertEqual(spec.kind, "rigid")
        self.assertAlmostEqual(spec.mass, L.COOLER_MASS)
        self.assertAlmostEqual(spec.yaw, L.START_YAW)
        st = spec.rest_state()  # the declared spawn pose reads as the start, airflow along +y
        self.assertTrue(np.allclose(st.origin, L.START))
        self.assertTrue(Tracker.at_start(st))
        self.assertAlmostEqual(airflow_err_deg(st), 90.0, places=6)

    def test_reset_hook_and_hooks(self):
        tr = self.task._TRACKER
        rigstate.offline_reset()
        a = snap()
        for _ in range(WINDOW + 1):
            self.task.success(self.objects(cooler(SEAT)))
        self.assertEqual(self.task.score(self.objects(cooler(SEAT))), 1.0)
        self.assertGreater(tr.steps, 0)
        rigstate.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.assertFalse(self.task.success(self.objects()))  # arms None right after a reset: no crash
        self.assertEqual(tr.arms_unavailable, 1)
        snap()
        self.assertFalse(self.task.success(self.objects()))
        self.assertEqual(self.task.score(self.objects()), 0.0)
        del a

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        path = HERE / "tests" / "_trace_test.jsonl"
        try:
            os.environ["TASK_TRACE"] = str(path)
            rigstate.offline_reset()
            snap()
            self.task.success(self.objects())
            row = json.loads(path.read_text().splitlines()[-1])
            for k in ("score", "zero_reason", "seated", "right_way", "let_go", "foot_gap_mm", "airflow_err_deg",
                      "frame_margin_mm", "base_dx_mm", "base_dy_mm", "cooler_origin", "left_tcp"):
                self.assertIn(k, row)
            os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
            self.assertFalse(self.task.success(self.objects()))
        finally:
            if path.exists():
                path.unlink()
            if old is None:
                os.environ.pop("TASK_TRACE", None)
            else:
                os.environ["TASK_TRACE"] = old

    def test_demo_switches_and_geometry(self):
        plan = self.task.demo()
        self.assertEqual(set(plan), {"left"})
        wps = plan["left"]
        self.assertEqual(sum(w.duration for w in wps), 19.3)
        grip = [w for w in wps if "grip the cooler" in w.label][0]
        self.assertAlmostEqual(-2 * grip.finger, L.UPSTREAM + L.DOWNSTREAM - 0.008)
        place = [w for w in wps if "lower the base slowly" in w.label][0]
        self.assertAlmostEqual(place.yaw, -math.pi / 2)
        base = np.array(place.pos) - [L.STACK_MID, 0.0, GRIP_UP]  # airflow +x: the base is behind the grip
        self.assertLess(math.hypot(base[0] - L.CPU_X, base[1] - L.CPU_Y), 0.0025)
        self.assertAlmostEqual(base[2] - CPU_W, 0.001)
        pk = [w for w in wps if "slowly down beside" in w.label][0]
        self.assertTrue(np.allclose(pk.pos, grip_tcp(START, 90.0)))
        self.assertEqual(pk.yaw, 0.0)
        for key, want in (("DEMO_IDLE_ARM", set()), ("DEMO_HOLD_ONLY", set())):
            os.environ[key] = "left" if key == "DEMO_IDLE_ARM" else "1"
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(set(self.task.demo()), {"left"})
        finally:
            os.environ.pop("DEMO_IDLE_ARM", None)


class Text(unittest.TestCase):
    """task.json and the README state what the code does."""

    @classmethod
    def setUpClass(cls):
        cls.spec = json.loads((HERE / "task.json").read_text())
        cls.readme = (HERE / "README.md").read_text()
        src = (HERE / "task.py").read_text()
        m = re.search(r"INSTRUCTION = \(\n(.*?)\n\)", src, re.S)
        cls.instruction = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1)))

    def test_instruction_is_the_same_everywhere(self):
        self.assertEqual(self.spec["instruction"], self.instruction)
        self.assertEqual(self.readme.count(f'--instruction "{self.instruction}"'), 2)

    def test_scoring_text_states_the_thresholds(self):
        s = self.spec["scoring"]
        for phrase in ("within 2 mm", "2 degrees or less", "44 x 44 mm", "within 15 degrees", "within 20 mm",
                       "30 degrees", "2 mm or less in the last 0.5 s", "0.9 or more", "more than 10 mm", "for 1 s",
                       "Possible scores: 0, 0.25, 0.5, 1.0"):
            self.assertIn(phrase, s)
        self.assertEqual(self.task_numbers(), (2.0, 2.0, 15.0, 20.0, 30.0, 0.5, 2.0, 0.9, 10.0))

    @staticmethod
    def task_numbers():
        return (L.SEAT_Z_TOL * 1000, L.SEAT_TILT_DEG, L.AIRFLOW_TOL_DEG, L.NEAR * 1000, L.UPRIGHT_DEG,
                L.STILL_WINDOW_S, L.STILL_TOL * 1000, L.GRIPPER_OPEN, L.LET_GO_GAP * 1000)

    def test_setup_text_states_the_layout(self):
        s = self.spec["setup"]
        for phrase in ("x 30.0 cm, y 0", "38 x 38 mm", "44 x 44 mm", "27 mm", "34 mm", "28 mm", "x 18.0 cm, y +23.5 cm",
                       "37 mm", "18 mm", "74 mm", "0.25 kg", "airflow along +y", "9 mm"):
            self.assertIn(phrase, s)
        self.assertEqual(tuple(round(v, 4) for v in (L.CPU_X, L.CPU_Y, L.PAD_X, L.PAD_Y)), (0.3, 0.0, 0.18, 0.235))


if __name__ == "__main__":
    unittest.main()
