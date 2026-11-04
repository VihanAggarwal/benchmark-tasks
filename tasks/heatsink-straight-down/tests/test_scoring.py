"""Offline tests for scoring.py: synthetic object states, no simulator.  python3 -m unittest discover -s tests -v"""

import math
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    Tracker,
    footprint,
    off_board,
    on_post,
    over_die,
    tilt_deg,
    touching,
    valid,
    yaw_err_deg,
)

HZ = 20.0
DT = 1.0 / HZ
DX, DY = L.DIE_X, L.DIE_Y


def quat(yaw_deg=0.0, tilt_deg_=0.0):
    """Turned yaw about the vertical after tilting tilt about the body's x axis (x, y, z, w)."""
    a, b = math.radians(yaw_deg) / 2, math.radians(tilt_deg_) / 2
    qz = np.array([0.0, 0.0, math.sin(a), math.cos(a)])
    qx = np.array([math.sin(b), 0.0, 0.0, math.cos(b)])
    (x1, y1, z1, w1), (x2, y2, z2, w2) = qz, qx
    return np.array([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2, w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2, w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2])


def rot(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def hs(x, y, low_z, yaw=0.0, tilt=0.0, vel=(0, 0, 0), spin=(0, 0, 0)):
    """A heatsink state: centre (x, y), lowest point at low_z, as the simulator reports it (AABB of the turned box)."""
    q = quat(yaw, tilt)
    half = np.array(L.HS_SIZE) / 2
    corners = np.array([rot(q) @ (half * np.array([sx, sy, sz])) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    c = np.array([x, y, low_z - corners[:, 2].min()])
    return SimpleNamespace(pos=c, quat_xyzw=q, lin_vel=np.array(vel, float), ang_vel=np.array(spin, float),
                           low=c + corners.min(axis=0), high=c + corners.max(axis=0))


START = hs(L.HS_X, L.HS_Y, L.TOP)
SEATED = hs(DX, DY, L.DIE_TOP)
CARRY = hs(DX, DY, L.DIE_TOP + 0.10)


def glide(a, b, n, yaw=(0.0, 0.0)):
    """n states moving from pose a to pose b in equal steps, with the matching velocities (the simulator moves
    things gradually). a, b: (x, y, low_z)."""
    a, b = np.array(a, float), np.array(b, float)
    vel = (b - a) / (n * DT)
    spin = (0.0, 0.0, math.radians(yaw[1] - yaw[0]) / (n * DT))
    out = []
    for i in range(1, n + 1):
        f = i / n
        p = (1 - f) * a + f * b
        out.append(hs(*p, yaw=(1 - f) * yaw[0] + f * yaw[1], vel=vel, spin=spin))
    return out


def still(x, y, low_z, n, **kw):
    return [hs(x, y, low_z, **kw)] * n


def run(states, t=None):
    """Feed the states to a tracker (a fresh one unless given); return the last result and the tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for s in states:
        r = t.update(s)
    return r, t


def approach(x=DX, y=DY, low=L.DIE_TOP + 0.022, yaw=0.0):
    """Pick the heatsink up from its mark, carry it 10 cm up over (x, y) and come down quickly to `low`."""
    top = L.DIE_TOP
    seq = still(L.HS_X, L.HS_Y, L.TOP, 10)
    seq += glide((L.HS_X, L.HS_Y, L.TOP), (L.HS_X, L.HS_Y, top + 0.10), 20)
    seq += glide((L.HS_X, L.HS_Y, top + 0.10), (x, y, top + 0.10), 30, yaw=(0.0, yaw))
    seq += glide((x, y, top + 0.10), (x, y, low), 20, yaw=(yaw, yaw))
    return seq + still(x, y, low, 10, yaw=yaw)


def place(x=DX, y=DY, *, speed=0.0105, yaw=0.0, settle=40):
    """approach(), then lower straight down at `speed` m/s from 22 mm to 1 mm above the die, let go (a 1 mm drop)
    and stand still for `settle` steps."""
    top = L.DIE_TOP
    n_slow = max(1, round(0.021 / speed / DT))
    seq = approach(x, y, top + 0.022, yaw)
    seq += glide((x, y, top + 0.022), (x, y, top + 0.001), n_slow, yaw=(yaw, yaw))
    seq += [hs(x, y, top, yaw=yaw, vel=(0, 0, -0.14))]
    return seq + still(x, y, top, settle, yaw=yaw)


class Geometry(unittest.TestCase):
    def test_touching_band(self):
        self.assertTrue(touching(SEATED))
        self.assertTrue(touching(hs(DX, DY, L.DIE_TOP + 0.0019)))
        self.assertFalse(touching(hs(DX, DY, L.DIE_TOP + 0.0025)))
        self.assertFalse(touching(hs(DX + 0.06, DY, L.BOARD_TOP)))  # on the board beside the die: 3 mm below
        self.assertFalse(touching(hs(DX + 0.05, DY, L.DIE_TOP)))  # at die height but not over it
        tipped_off = hs(DX + 0.030, DY, L.BOARD_TOP, yaw=90.0, tilt=6.0)  # over the die's edge, low side on the board
        self.assertTrue(over_die(tipped_off))
        self.assertFalse(touching(tipped_off))

    def test_footprint_is_the_real_outline(self):
        d = 0.037  # turned 45 deg and moved diagonally: its bounding box covers the die's corner, its outline not
        turned = hs(DX + d, DY + d, L.DIE_TOP, yaw=45.0)
        self.assertLess(turned.low[0], DX + L.DIE_HALF)
        self.assertLess(turned.low[1], DY + L.DIE_HALF)
        self.assertFalse(over_die(turned))
        self.assertTrue(over_die(hs(DX + 0.033, DY + 0.033, L.DIE_TOP, yaw=45.0)))
        self.assertEqual(footprint(SEATED).shape, (4, 2))  # flat: the underside, 4 corners
        self.assertEqual(footprint(hs(DX, DY, L.DIE_TOP, yaw=30.0, tilt=10.0)).shape, (6, 2))  # tilted: a hexagon

    def test_angles(self):
        self.assertAlmostEqual(tilt_deg(quat(30.0, 6.0)), 6.0, places=4)
        self.assertAlmostEqual(yaw_err_deg(quat(3.5)), 3.5, places=4)
        self.assertAlmostEqual(yaw_err_deg(quat(88.0)), 2.0, places=4)  # a square: 88 deg is 2 deg off
        self.assertAlmostEqual(yaw_err_deg(quat(-1.0)), 1.0, places=4)
        self.assertAlmostEqual(yaw_err_deg(quat(45.0)), 45.0, places=4)

    def test_on_post(self):
        self.assertFalse(on_post(SEATED))
        corner_on_post = hs(DX + 0.007, DY + 0.007, L.POST_TOP)  # 7 mm off diagonally: a corner over a post
        self.assertTrue(on_post(corner_on_post))
        self.assertFalse(on_post(hs(DX + 0.007, DY + 0.007, L.POST_TOP + 0.010)))  # in the air above it
        self.assertFalse(on_post(hs(DX + 0.003, DY, L.DIE_TOP)))  # 3 mm off along x: the posts are clear
        rise = 0.055 * math.sin(math.radians(5.0))  # tilted 5 deg, +y side up: its left corners on the posts
        self.assertTrue(on_post(hs(DX + 0.007, DY + 0.007, L.POST_TOP - rise, tilt=5.0)))
        self.assertFalse(on_post(hs(DX + 0.007, DY + 0.007, L.POST_TOP - rise + 0.004, tilt=5.0)))

    def test_off_board(self):
        self.assertTrue(off_board(START))
        self.assertFalse(off_board(SEATED))
        self.assertFalse(off_board(hs(DX + 0.09, DY - 0.07, L.BOARD_TOP)))
        self.assertTrue(off_board(hs(DX + 0.11, DY, L.TOP)))


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, _ = run(still(L.HS_X, L.HS_Y, L.TOP, 10))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_straight_down_slowly_scores_one(self):
        r, t = run(place())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertLess(t.smear, 1e-9)
        self.assertLess(t.setdown, 0.02)

    def test_within_tolerances_scores_one(self):
        r, _ = run(place(DX + 0.0014, DY - 0.0014, yaw=2.5))  # 1.98 mm off, 2.5 deg turned
        self.assertEqual(r.score, 1.0)

    def test_small_slide_scores_075(self):
        seq = place(DX - 0.0015, DY, settle=5) + glide((DX - 0.0015, DY, L.DIE_TOP), (DX + 0.0015, DY, L.DIE_TOP), 10)
        r, t = run(seq + still(DX + 0.0015, DY, L.DIE_TOP, 30))
        self.assertAlmostEqual(t.smear, 0.003, places=6)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)

    def test_fast_set_down_scores_075(self):
        r, t = run(place(speed=0.20))
        self.assertGreater(t.setdown, L.SETDOWN_MAX)
        self.assertEqual(r.score, 0.75)

    def test_off_by_4mm_scores_half(self):
        r, _ = run(place(DX + 0.004, DY))
        self.assertEqual(r.score, 0.5)

    def test_turned_7deg_scores_half(self):
        r, _ = run(place(yaw=7.0))
        self.assertEqual(r.score, 0.5)

    def test_long_slide_into_place_scores_half(self):
        seq = place(DX + 0.004, DY - 0.003, settle=5)  # set down 5 mm off, then slid to the centre
        seq += glide((DX + 0.004, DY - 0.003, L.DIE_TOP), (DX, DY, L.DIE_TOP), 20)
        r, t = run(seq + still(DX, DY, L.DIE_TOP, 30))
        self.assertGreater(t.smear, L.SMEAR_LOOSE - 1e-9)
        self.assertEqual(r.score, 0.5)

    def test_on_die_but_tilted_or_far_off_scores_quarter(self):
        r, _ = run(place(settle=5) + still(DX, DY, L.DIE_TOP, 30, tilt=5.0))
        self.assertEqual(r.score, 0.25)
        r, _ = run(place(DX + 0.010, DY))
        self.assertEqual(r.score, 0.25)
        r, _ = run(place(yaw=20.0))
        self.assertEqual(r.score, 0.25)

    def test_held_above_the_die_scores_zero(self):
        r, t = run(approach(low=L.DIE_TOP + 0.010) + still(DX, DY, L.DIE_TOP + 0.010, 30))
        self.assertFalse(t.touched)
        self.assertEqual(r.score, 0.0)

    def test_left_beside_the_die_on_the_board_scores_zero(self):
        r, _ = run(still(L.HS_X, L.HS_Y, L.TOP, 5) + glide((L.HS_X, L.HS_Y, L.TOP), (DX, DY + 0.05, L.BOARD_TOP), 40)
                   + still(DX, DY + 0.05, L.BOARD_TOP, 30))
        self.assertEqual(r.score, 0.0)

    def test_off_the_board_scores_zero(self):
        r, _ = run(place(settle=5) + glide((DX, DY, L.DIE_TOP), (DX, DY + 0.15, L.TOP), 30)
                   + still(DX, DY + 0.15, L.TOP, 20))
        self.assertTrue(r.details["off_board"])
        self.assertEqual(r.score, 0.0)

    def test_tipped_scores_zero(self):
        leaning = hs(DX, DY, L.DIE_TOP, tilt=35.0)  # still within 2 mm of the die, so it would be 0.25
        self.assertTrue(touching(leaning))
        r, _ = run(place(settle=5) + still(DX, DY, L.DIE_TOP, 20, tilt=35.0))
        self.assertEqual(r.score, 0.0)
        on_side = hs(DX, DY, L.DIE_TOP, tilt=90.0)
        self.assertTrue(touching(on_side))
        r, _ = run(place(settle=5) + [on_side] * 20)
        self.assertEqual(r.score, 0.0)

    def test_resting_on_a_post_scores_zero(self):
        # 7 mm off diagonally: one side up on two posts, the other on the die's far edge (13 deg, low point
        # 1.4 mm below the die top, so touching: 0.25 without the post rule)
        th = math.asin(0.011 / 0.049)
        on_posts = hs(DX + 0.007, DY + 0.007, L.DIE_TOP - 0.006 * math.sin(th), tilt=math.degrees(th))
        self.assertTrue(touching(on_posts))
        self.assertTrue(on_post(on_posts))
        r, _ = run(place(settle=5) + [on_posts] * 30)
        self.assertTrue(r.details["on_post"])
        self.assertEqual(r.score, 0.0)

    def test_success_needs_stillness(self):
        r, _ = run(place(settle=5) + [hs(DX, DY, L.DIE_TOP, vel=(0.0, 0.0, 0.03))] * 3)
        self.assertEqual(r.score, 1.0)
        self.assertFalse(r.success)
        r, _ = run(place(settle=5) + [hs(DX, DY, L.DIE_TOP, spin=(0.0, 0.0, 1.0))] * 3)
        self.assertFalse(r.success)

    def test_score_is_idempotent(self):
        seq = place()
        t = Tracker(HZ)
        run(seq, t)
        a = t.result(seq[-1])
        b = t.result(seq[-1])
        self.assertEqual((a.score, a.details), (b.score, b.details))
        self.assertEqual(t.steps, len(seq))


class Exploits(unittest.TestCase):
    def test_drop_from_height_is_not_a_set_down(self):
        top = L.DIE_TOP
        seq = approach(low=top + 0.06)  # held still 6 cm above the die, then let go
        fall = [hs(DX, DY, top + 0.06 - 0.5 * 9.81 * (k * DT) ** 2, vel=(0, 0, -9.81 * k * DT)) for k in (1, 2)]
        seq += [s for s in fall if s.low[2] > top] + still(DX, DY, top, 30)
        r, t = run(seq)
        self.assertGreater(t.setdown, 0.5)
        self.assertEqual(r.score, 0.75)

    def test_drop_inside_one_step_is_caught(self):
        top = L.DIE_TOP
        seq = approach() + glide((DX, DY, top + 0.022), (DX, DY, top + 0.008), 30)  # slowly to 8 mm, then let go
        seq += still(DX, DY, top + 0.008, 10) + still(DX, DY, top, 30)  # it lands within one control step
        r, t = run(seq)
        self.assertGreater(t.setdown, L.SETDOWN_MAX)
        self.assertEqual(r.score, 0.75)

    def test_fast_then_slow_in_the_window_counts(self):
        top = L.DIE_TOP
        seq = still(L.HS_X, L.HS_Y, L.TOP, 5) + glide((L.HS_X, L.HS_Y, L.TOP), (DX, DY, top + 0.10), 40)
        seq += glide((DX, DY, top + 0.10), (DX, DY, top + 0.004), 3)  # 0.64 m/s into the last 10 mm
        seq += still(DX, DY, top + 0.004, 20) + glide((DX, DY, top + 0.004), (DX, DY, top), 40)
        r, t = run(seq + still(DX, DY, top, 30))
        self.assertGreater(t.setdown, L.SETDOWN_MAX)
        self.assertEqual(r.score, 0.75)

    def test_twisting_square_on_the_paste_is_smear(self):
        seq = place(yaw=8.0, settle=5)  # set down 8 deg turned (0.5), then twisted square while touching
        seq += glide((DX, DY, L.DIE_TOP), (DX, DY, L.DIE_TOP), 20, yaw=(8.0, 0.0))
        r, t = run(seq + still(DX, DY, L.DIE_TOP, 30))
        self.assertGreater(t.smear, L.SMEAR_TIGHT)
        self.assertEqual(r.score, 0.75)
        seq = place(yaw=20.0, settle=5) + glide((DX, DY, L.DIE_TOP), (DX, DY, L.DIE_TOP), 20, yaw=(20.0, 0.0))
        r, _ = run(seq + still(DX, DY, L.DIE_TOP, 30))
        self.assertEqual(r.score, 0.5)

    def test_pushing_it_along_the_board_onto_the_die(self):
        seq = still(L.HS_X, L.HS_Y, L.TOP, 5)
        seq += glide((L.HS_X, L.HS_Y, L.TOP), (DX - 0.05, DY, L.BOARD_TOP), 30)  # onto the board, 5 cm behind
        seq += glide((DX - 0.05, DY, L.BOARD_TOP), (DX - 0.040, DY, L.DIE_TOP), 5)  # up the die's edge
        seq += glide((DX - 0.040, DY, L.DIE_TOP), (DX, DY, L.DIE_TOP), 40)  # pushed across it to the centre
        r, t = run(seq + still(DX, DY, L.DIE_TOP, 30))
        self.assertGreater(t.smear, 0.03)
        self.assertEqual(r.score, 0.5)

    def test_sideways_while_clear_of_the_die_is_not_smear(self):
        top = L.DIE_TOP
        seq = approach(DX + 0.01, DY) + glide((DX + 0.01, DY, top + 0.022), (DX + 0.01, DY, top + 0.0025), 40)
        seq += glide((DX + 0.01, DY, top + 0.0025), (DX, DY, top + 0.0025), 20)  # 2.5 mm up: not touching yet
        seq += glide((DX, DY, top + 0.0025), (DX, DY, top), 30)
        r, t = run(seq + still(DX, DY, top, 30))
        self.assertLess(t.smear, 1e-9)
        self.assertEqual(r.score, 1.0)


    def test_touch_then_lift_and_drop_is_caught(self):
        # Dip gently into the touching band, lift 4 cm, then let it fall: the set-down speed is measured every time
        # it comes to touch, not only the first time.
        top = L.DIE_TOP
        seq = approach() + glide((DX, DY, top + 0.022), (DX, DY, top + 0.0015), 40)
        seq += glide((DX, DY, top + 0.0015), (DX, DY, top + 0.04), 20) + still(DX, DY, top + 0.04, 10)
        for k in range(1, 6):
            z = top + 0.04 - 0.5 * 9.81 * (k * DT) ** 2
            if z <= top:
                break
            seq.append(hs(DX, DY, z, vel=(0, 0, -9.81 * k * DT)))
        r, t = run(seq + still(DX, DY, top, 40))
        self.assertGreater(t.setdown, 0.5)
        self.assertEqual(r.score, 0.75)

    def test_fast_then_slow_on_a_second_approach_counts(self):
        # Touch gently, lift 3 cm, rush back into the last 10 mm and finish slowly: the window is watched on every
        # approach, not only before the first touch.
        top = L.DIE_TOP
        seq = approach() + glide((DX, DY, top + 0.022), (DX, DY, top + 0.0015), 40)
        seq += glide((DX, DY, top + 0.0015), (DX, DY, top + 0.03), 20)
        seq += glide((DX, DY, top + 0.03), (DX, DY, top + 0.004), 2)  # 0.26 m/s into the window
        seq += still(DX, DY, top + 0.004, 10) + glide((DX, DY, top + 0.004), (DX, DY, top), 40)
        r, t = run(seq + still(DX, DY, top, 40))
        self.assertGreater(t.setdown, L.SETDOWN_MAX)
        self.assertEqual(r.score, 0.75)

    def test_a_careful_second_set_down_is_not_penalised(self):
        top = L.DIE_TOP
        seq = approach() + glide((DX, DY, top + 0.022), (DX, DY, top + 0.0015), 40)
        seq += glide((DX, DY, top + 0.0015), (DX, DY, top + 0.008), 20)  # lifted slowly out of the band
        seq += glide((DX, DY, top + 0.008), (DX, DY, top), 40)  # and set down slowly again
        r, t = run(seq + still(DX, DY, top, 40))
        self.assertLess(t.setdown, L.SETDOWN_MAX)
        self.assertEqual(r.score, 1.0)

    def test_slow_creep_counts_in_full(self):
        # 0.02 mm a step (well under the 0.2 mm counting step) for 100 steps: 2 mm, counted.
        seq = place(DX - 0.001, DY, settle=5)
        seq += [hs(DX - 0.001 + 0.00002 * k, DY, L.DIE_TOP, vel=(0.0004, 0, 0)) for k in range(1, 101)]
        r, t = run(seq + still(DX + 0.001, DY, L.DIE_TOP, 30))
        self.assertGreater(t.smear, 0.0018)
        self.assertEqual(r.score, 0.75)

    def test_what_is_left_is_counted_when_it_stops_touching(self):
        # Creep 0.15 mm (under one counting step), then lift it out of the band: the 0.15 mm counts.
        top = L.DIE_TOP
        seq = place(settle=5) + [hs(DX + 0.00005 * k, DY, top) for k in (1, 2, 3)]
        r, t = run(seq + [hs(DX + 0.00015, DY, top + 0.004)])
        self.assertAlmostEqual(t.smear, 0.00015, places=7)

    def test_sliding_while_bobbing_in_and_out_of_touch_counts(self):
        top = L.DIE_TOP
        seq = approach(DX - 0.010, DY) + glide((DX - 0.010, DY, top + 0.022), (DX - 0.010, DY, top + 0.003), 40)
        for k in range(1, 11):  # 1 mm sideways per step, touching every other step
            seq.append(hs(DX - 0.010 + 0.001 * k, DY, top + (0.001 if k % 2 else 0.003), vel=(0.02, 0, 0)))
        seq += glide((DX, DY, top + 0.003), (DX, DY, top), 30)
        r, t = run(seq + still(DX, DY, top, 30))
        self.assertAlmostEqual(t.smear, 0.010, places=6)
        self.assertEqual(r.score, 0.5)


class Robustness(unittest.TestCase):
    def test_jitter_does_not_add_up(self):
        # A seated heatsink shivering +-0.05 mm and +-0.05 deg every step for 10 s (contact jitter) is still 1.0;
        # summing every step's travel would have made it 13 mm of smear.
        rng = np.random.default_rng(0)
        jit = [hs(DX + rng.uniform(-5e-5, 5e-5), DY + rng.uniform(-5e-5, 5e-5), L.DIE_TOP,
                  yaw=rng.uniform(-0.05, 0.05)) for _ in range(200)]
        r, t = run(place(settle=5) + jit + still(DX, DY, L.DIE_TOP, 40))
        self.assertLess(t.smear, L.SMEAR_STEP)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_either_face_down(self):
        r, _ = run(place(settle=5) + still(DX, DY, L.DIE_TOP, 40, tilt=180.0))  # upside down: the same block
        self.assertEqual(r.score, 1.0)
        self.assertAlmostEqual(tilt_deg(quat(0.0, 180.0)), 0.0, places=4)
        self.assertAlmostEqual(tilt_deg(quat(0.0, 145.0)), 35.0, places=4)
        r, _ = run(place(settle=5) + still(DX, DY, L.DIE_TOP, 20, tilt=145.0))
        self.assertEqual(r.score, 0.0)
        flipped_on_post = hs(DX + 0.007, DY + 0.007, L.POST_TOP, tilt=180.0)
        self.assertTrue(on_post(flipped_on_post))

    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        nan = SimpleNamespace(pos=np.full(3, np.nan), quat_xyzw=np.full(4, np.nan), lin_vel=np.zeros(3),
                              ang_vel=np.zeros(3), low=np.full(3, np.nan), high=np.full(3, np.nan))
        inf = SimpleNamespace(pos=np.array([DX, DY, np.inf]), quat_xyzw=np.array([0, 0, 0, 1.0]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3), low=np.array([0, 0, -np.inf]),
                              high=np.zeros(3))
        zero_q = SimpleNamespace(pos=SEATED.pos, quat_xyzw=np.zeros(4), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                 low=SEATED.low, high=SEATED.high)
        for bad in (nan, inf, zero_q):
            self.assertFalse(valid(bad))
            t = Tracker(HZ)
            run(place(settle=5), t)
            before = (t.smear, t.setdown, t.touched)
            r = t.update(bad)
            self.assertEqual((r.score, r.success), (0.0, False))
            self.assertEqual(t.result(bad).score, 0.0)
            self.assertEqual((t.smear, t.setdown, t.touched), before)
            r, _ = run(still(DX, DY, L.DIE_TOP, 50), t)  # the next good states are scored as before
            self.assertEqual(r.score, 1.0)

    def test_states_anywhere_do_not_crash(self):
        odd = [hs(5.0, -3.0, -1.0), hs(DX, DY, 0.0, tilt=120.0, yaw=33.0), hs(-0.2, 0.6, L.TOP, tilt=90.0, yaw=45.0),
               hs(DX, DY, L.DIE_TOP - 0.01, tilt=89.9), hs(DX, DY, 2.0, vel=(5, -5, 30), spin=(40, 0, 0))]
        t = Tracker(HZ)
        for s in odd + odd[::-1]:
            r = t.update(s)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            self.assertTrue(math.isfinite(t.smear) and math.isfinite(t.setdown))
        self.assertEqual(r.score, 0.0)


class Reset(unittest.TestCase):
    def test_teleport_to_a_jittered_start_resets(self):
        # A seeded reset (object_jitter, +-3 cm) puts it back near, not exactly on, its mark: still a new episode.
        _, t = run(place(settle=5) + glide((DX, DY, L.DIE_TOP), (DX + 0.01, DY, L.DIE_TOP), 10))
        t.update(hs(L.HS_X + 0.03, L.HS_Y - 0.03, L.TOP))
        self.assertFalse(t.touched)
        self.assertEqual(t.smear, 0.0)


    def test_teleport_back_to_start_resets(self):
        seq = place(settle=5) + glide((DX, DY, L.DIE_TOP), (DX + 0.01, DY, L.DIE_TOP), 10)
        _, t = run(seq)
        self.assertTrue(t.touched)
        self.assertGreater(t.smear, 0.009)
        r = t.update(START)  # everything jumps back in one step: a new episode
        self.assertFalse(t.touched)
        self.assertEqual(t.smear, 0.0)
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.score, 0.0)
        r, _ = run(place()[1:], t)  # and the next episode is scored on its own
        self.assertEqual(r.score, 1.0)

    def test_carrying_it_back_slowly_does_not_reset(self):
        seq = place(settle=5) + glide((DX, DY, L.DIE_TOP), (DX + 0.01, DY, L.DIE_TOP), 10)
        seq += glide((DX + 0.01, DY, L.DIE_TOP), (DX, DY, L.DIE_TOP + 0.10), 20)
        seq += glide((DX, DY, L.DIE_TOP + 0.10), (L.HS_X, L.HS_Y, L.TOP + 0.10), 30)
        seq += glide((L.HS_X, L.HS_Y, L.TOP + 0.10), (L.HS_X, L.HS_Y, L.TOP + 0.0025), 20)
        seq += glide((L.HS_X, L.HS_Y, L.TOP + 0.0025), (L.HS_X, L.HS_Y, L.TOP), 10)  # 5 mm/s: 'resting' speed
        r, t = run(seq + still(L.HS_X, L.HS_Y, L.TOP, 10))
        self.assertTrue(t.touched)  # back at the start, but it got there gradually: same episode
        self.assertGreater(t.smear, 0.009)
        self.assertEqual(r.score, 0.0)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        seq = place(settle=5) + glide((DX, DY, L.DIE_TOP), (DX + 0.01, DY, L.DIE_TOP), 10)
        _, t = run(seq)
        t.update(hs(DX, DY + 0.15, L.TOP))  # knocked far in one step, but not onto the start mark
        self.assertTrue(t.touched)
        self.assertGreater(t.smear, 0.009)


if __name__ == "__main__":
    unittest.main()
