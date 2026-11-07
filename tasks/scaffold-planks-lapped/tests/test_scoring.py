"""Offline tests for scoring.py (and task.py's hooks and demo): synthetic board states, no simulator.

    python3 -m unittest discover -s tests -v
"""

import importlib
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
import offline_settle  # noqa: E402
from scoring import (  # noqa: E402
    Tracker,
    evaluate,
    heading_err_deg,
    on_board,
    on_putlog_directly,
    overhangs,
    supports,
    tilt_deg,
    valid,
)

HZ = L.CONTROL_HZ
HALF = np.array(L.BOARD_SIZE) / 2
N_STILL = 12  # more than the 0.5 s window (10 steps + the current one)
LEFT, RIGHT = "board_left", "board_right"


# -- building board states ---------------------------------------------------------------------------------------
def rz(deg):
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0.0, 0.0, 1.0]])


def rx(deg):
    a = math.radians(deg)
    return np.array([[1.0, 0.0, 0.0], [0.0, math.cos(a), -math.sin(a)], [0.0, math.sin(a), math.cos(a)]])


def quat_of(r):
    """(x, y, z, w) of a rotation matrix (independent of the code under test)."""
    w = math.sqrt(max(0.0, 1.0 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def box(centre, r, vel=(0.0, 0.0, 0.0)):
    """A board state as the simulator reports it: centre, quaternion, velocities, world bounding box."""
    c = np.asarray(centre, dtype=float)
    corners = np.array([c + r @ (HALF * np.array([i, j, k])) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    return SimpleNamespace(pos=c, quat_xyzw=quat_of(r), lin_vel=np.array(vel, float), ang_vel=np.zeros(3),
                           low=corners.min(axis=0), high=corners.max(axis=0))


def flat(cx, cy, bottom, yaw=90.0, roll=0.0):
    """A board lying flat (or rolled ``roll`` about its length) with its underside at ``bottom``, its long side
    turned ``yaw`` from x (90: along the run)."""
    return box((cx, cy, bottom + HALF[2]), rz(yaw) @ rx(roll))


def start(name):
    return flat(L.BOARD_START_X, L.BOARD_START_Y[name], L.TOP, yaw=0.0)


def lower(cy=L.LOWER_Y, cx=L.RUN_X, yaw=90.0):
    """A board flat on two putlogs (underside on their tops)."""
    return flat(cx, cy, L.PUTLOG_TOP, yaw)


def upper(edge_y, inner_y, cx=L.RUN_X):
    """The lapped board as it settles: its underside resting on the lower board's end edge (at y ``edge_y``, the
    lower board's top) and on the outer edge of the outer putlog of its bay, its inner underside end at y
    ``inner_y``. Its bay is the one on the far side of ``edge_y`` from ``inner_y``."""
    side = 1.0 if edge_y > inner_y else -1.0
    out_edge = side * (abs(L.PUTLOG_Y["putlog_left"]) + L.PUTLOG_HALF_W)
    theta = math.atan2(L.LOWER_TOP - L.PUTLOG_TOP, abs(out_edge - edge_y))
    u = np.array([0.0, side * math.cos(theta), -math.sin(theta)])  # along the board, outward and down
    n = np.array([0.0, side * math.sin(theta), math.cos(theta)])  # its upper face's normal
    z_inner = L.LOWER_TOP + abs(edge_y - inner_y) * math.tan(theta)
    a = np.array([cx, inner_y, z_inner])  # the inner end of its underside's centre line
    centre = a + u * HALF[0] + n * HALF[2]
    return box(centre, np.column_stack([u, np.cross(n, u), n]))


def nominal(left_upper=True, shift_lower=0.0, shift_upper=0.0, upper_x=L.RUN_X):
    """The finished platform: one board flat across one bay, the other lapped over its end across the other."""
    if left_upper:
        lo = lower(L.LOWER_Y + shift_lower)
        edge = L.LOWER_Y + shift_lower + HALF[0]
        return {LEFT: upper(edge, -0.030 + shift_upper, upper_x), RIGHT: lo}
    lo = lower(-L.LOWER_Y + shift_lower)
    edge = -L.LOWER_Y + shift_lower - HALF[0]
    return {LEFT: lo, RIGHT: upper(edge, 0.030 + shift_upper, upper_x)}


def run(seq, t=None):
    """Feed a list of {board: state} dicts to a tracker (a fresh one unless given); last result and tracker."""
    t = Tracker(HZ) if t is None else t
    r = None
    for boards in seq:
        r = t.update(boards)
    return r, t


def held(boards, n=N_STILL):
    return [boards] * n


def glide(a, b, n):
    """n states moving board by board from layout a to layout b along straight lines (centre only)."""
    out = []
    for i in range(1, n + 1):
        f = i / n
        step = {}
        for k in a:
            pa, pb = np.asarray(a[k].pos), np.asarray(b[k].pos)
            d = pb - pa
            s = SimpleNamespace(**vars(b[k] if f > 0.5 else a[k]))
            base = b[k] if f > 0.5 else a[k]
            off = (pa + f * d) - np.asarray(base.pos)
            s.pos, s.low, s.high = base.pos + off, base.low + off, base.high + off
            s.lin_vel = d * HZ / n
            step[k] = s
        out.append(step)
    return out


START = {LEFT: start(LEFT), RIGHT: start(RIGHT)}


# -- geometry ------------------------------------------------------------------------------------------------------
class Geometry(unittest.TestCase):
    def test_flat_board_across_the_right_bay(self):
        b = lower()
        self.assertTrue(on_putlog_directly(b, "putlog_right"))
        self.assertTrue(on_putlog_directly(b, "putlog_middle"))
        self.assertFalse(on_putlog_directly(b, "putlog_left"))
        oh = overhangs(b, ["putlog_right", "putlog_middle"])
        self.assertAlmostEqual(oh["low"], 0.030, places=6)
        self.assertAlmostEqual(oh["high"], 0.030, places=6)
        self.assertAlmostEqual(heading_err_deg(b), 0.0, places=6)

    def test_held_above_a_putlog_is_not_on_it(self):
        self.assertTrue(on_putlog_directly(flat(L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP + 0.0019), "putlog_right"))
        self.assertFalse(on_putlog_directly(flat(L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP + 0.0025), "putlog_right"))

    def test_the_lapped_board(self):
        b = nominal()
        up = b[LEFT]
        self.assertAlmostEqual(tilt_deg(up), math.degrees(math.atan2(0.016, 0.170)), places=4)  # 5.38 deg
        rests, part = on_board(up, b[RIGHT])
        self.assertTrue(rests)
        self.assertLessEqual(part[:, 1].min(), 0.0)  # the overlap covers the middle putlog's centre line
        self.assertGreaterEqual(part[:, 1].max(), 0.0)
        self.assertTrue(on_putlog_directly(up, "putlog_left"))
        self.assertFalse(on_putlog_directly(up, "putlog_middle"))  # it is 18 mm above it, on the other board
        sup = supports(b)
        self.assertEqual(sup[LEFT], {"putlog_left": "direct", "putlog_middle": "through"})
        self.assertEqual(sup[RIGHT], {"putlog_middle": "direct", "putlog_right": "direct"})
        oh = overhangs(up, sup[LEFT])
        self.assertAlmostEqual(oh["low"] + oh["high"], L.BOARD_LEN - 0.190 / math.cos(math.radians(5.38)), places=3)

    def test_offline_settle_matches_the_geometry(self):
        """The offline-only settling model turns a level upper board down onto its putlog as upper() builds it."""
        lo = lower()
        level = flat(L.RUN_X, L.UPPER_Y, L.LOWER_TOP)  # as a checker without physics leaves it
        tipped = offline_settle.tipped(level, lo)
        self.assertIsNotNone(tipped)
        self.assertAlmostEqual(tilt_deg(tipped), 5.38, places=2)
        self.assertTrue(on_putlog_directly(tipped, "putlog_left"))
        self.assertLess(np.linalg.norm(np.asarray(tipped.pos)[:2] - np.array([L.RUN_X, L.UPPER_Y])), 0.001)
        score, _ = evaluate({LEFT: tipped, RIGHT: lo}, {LEFT: True, RIGHT: True})
        self.assertEqual(score, 1.0)
        # Not tipped: the lower board itself, a board whose centre is over the other, boards not touching.
        self.assertIsNone(offline_settle.tipped(lo, tipped))
        self.assertIsNone(offline_settle.tipped(flat(L.RUN_X, L.LOWER_Y + 0.05, L.LOWER_TOP), lo))
        self.assertIsNone(offline_settle.tipped(flat(L.RUN_X, L.UPPER_Y, L.LOWER_TOP + 0.002), lo))
        self.assertIsNone(offline_settle.tipped(flat(L.RUN_X + 0.05, L.UPPER_Y, L.LOWER_TOP), lo))
        # Mirrored: the right board lapped over a left lower board tips the other way.
        lo2 = lower(-L.LOWER_Y)
        t2 = offline_settle.tipped(flat(L.RUN_X, -L.UPPER_Y, L.LOWER_TOP), lo2)
        self.assertTrue(on_putlog_directly(t2, "putlog_right"))
        settler = offline_settle.Settler()
        self.assertIs(settler.state_for(RIGHT, lo), lo)
        self.assertTrue(on_putlog_directly(settler.state_for(LEFT, level), "putlog_left"))

    def test_heading_and_tilt(self):
        self.assertAlmostEqual(heading_err_deg(flat(0.26, 0.0, L.TOP, yaw=0.0)), 90.0, places=6)
        self.assertAlmostEqual(heading_err_deg(flat(0.26, 0.0, L.TOP, yaw=94.0)), 4.0, places=6)
        self.assertAlmostEqual(heading_err_deg(flat(0.26, 0.0, L.TOP, yaw=-87.0)), 3.0, places=6)
        self.assertAlmostEqual(tilt_deg(flat(0.26, 0.0, L.TOP, roll=170.0)), 10.0, places=6)  # upside down, 10 off

    def test_valid(self):
        b = lower()
        self.assertTrue(valid(b))
        for key, bad in (("pos", np.array([np.nan, 0, 0])), ("quat_xyzw", np.zeros(4)),
                         ("lin_vel", np.array([np.inf, 0, 0])), ("low", np.array([0, 0]))):
            s = SimpleNamespace(**vars(b))
            setattr(s, key, bad)
            self.assertEqual(valid(s), key == "low", key)  # a 2-element low is finite and non-empty
        self.assertFalse(valid(SimpleNamespace(pos=b.pos)))


# -- the tiers -----------------------------------------------------------------------------------------------------
class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        r, t = run(held(START))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertTrue(t.at_start(START))
        self.assertEqual(r.details["zero_reason"], "nothing_bears")

    def test_lapped_in_line_scores_one(self):
        r, _ = run(held(nominal()))
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["upper"], LEFT)
        self.assertEqual(r.details["bays_planked"], ["left", "right"])
        # 30 mm of the lower board past the middle centre line, plus the upper board's 29.4 mm (measured along it,
        # on its centre line, which the 5.4 degree slope shifts 0.75 mm toward the middle)
        self.assertAlmostEqual(r.details["lap_mm"], 59.4, delta=0.2)

    def test_either_board_may_be_the_upper_one(self):
        r, _ = run(held(nominal(left_upper=False)))
        self.assertEqual(r.score, 1.0)
        self.assertEqual(r.details["upper"], RIGHT)

    def test_within_the_windows_scores_one(self):
        for sl, su in ((0.007, 0.0), (-0.007, 0.0), (0.0, 0.008), (0.0, -0.008), (0.006, 0.006)):
            r, _ = run(held(nominal(shift_lower=sl, shift_upper=su)))
            self.assertEqual(r.score, 1.0, (sl, su, r.details["boards"]))
        r, _ = run(held(nominal(upper_x=L.RUN_X + 0.009)))  # 9 mm out of line
        self.assertEqual(r.score, 1.0)

    def test_out_of_line_scores_half(self):
        r, _ = run(held(nominal(upper_x=L.RUN_X + 0.012)))  # still on the lower board's edge, 12 mm out of line
        self.assertEqual(r.details["boards"][LEFT]["bears_in"], "left")
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["lapped_in_line"])

    def test_side_by_side_scores_half(self):
        boards = {LEFT: lower(L.UPPER_Y, cx=0.230), RIGHT: lower(L.LOWER_Y, cx=0.290)}  # both flat on the middle
        r, _ = run(held(boards))
        self.assertEqual(r.score, 0.5)
        self.assertIsNone(r.details["upper"])
        self.assertAlmostEqual(r.details["in_line_mm"], 60.0, places=3)

    def test_one_bay_scores_quarter(self):
        r, _ = run(held({LEFT: start(LEFT), RIGHT: lower()}))
        self.assertEqual(r.score, 0.25)
        r, _ = run(held({LEFT: lower(L.UPPER_Y), RIGHT: start(RIGHT)}))
        self.assertEqual(r.score, 0.25)

    def test_both_boards_in_one_bay_score_quarter(self):
        side = {LEFT: lower(L.LOWER_Y, cx=0.230), RIGHT: lower(L.LOWER_Y, cx=0.290)}
        r, _ = run(held(side))
        self.assertEqual(r.score, 0.25)
        stacked = {LEFT: flat(L.RUN_X, L.LOWER_Y, L.LOWER_TOP), RIGHT: lower()}  # one on the other, same bay
        r, _ = run(held(stacked))
        self.assertEqual(r.details["boards"][LEFT]["lies_on"], {"putlog_middle": "through",
                                                                "putlog_right": "through"})
        self.assertEqual(r.details["boards"][LEFT]["bears_in"], "right")
        self.assertEqual(r.score, 0.25)

    def test_not_along_the_run_does_not_bear(self):
        r, _ = run(held({LEFT: start(LEFT), RIGHT: flat(L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP, yaw=96.0)}))
        self.assertEqual(r.score, 0.0)
        r, _ = run(held({LEFT: start(LEFT), RIGHT: flat(L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP, yaw=94.0)}))
        self.assertEqual(r.score, 0.25)

    def test_off_the_putlogs_ends_does_not_bear(self):
        r, _ = run(held({LEFT: start(LEFT), RIGHT: lower(cx=0.299)}))  # outline to x 0.319
        self.assertEqual(r.score, 0.25)
        r, _ = run(held({LEFT: start(LEFT), RIGHT: lower(cx=0.302)}))  # outline to x 0.322
        self.assertEqual(r.score, 0.0)

    def test_rolled_over_8_degrees_does_not_bear(self):
        boards = {LEFT: start(LEFT), RIGHT: flat(L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP, roll=9.0)}
        r, _ = run(held(boards))
        self.assertEqual(r.score, 0.0)

    def test_held_above_scores_zero(self):
        r, _ = run(held({LEFT: start(LEFT), RIGHT: flat(L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP + 0.010)}))
        self.assertEqual(r.score, 0.0)

    def test_still_moving_does_not_bear(self):
        a = {LEFT: start(LEFT), RIGHT: lower(L.LOWER_Y - 0.004)}
        b = {LEFT: start(LEFT), RIGHT: lower(L.LOWER_Y + 0.004)}
        r, t = run(held(a) + glide(a, b, 4))  # slid 8 mm along the run in 0.2 s
        self.assertEqual(r.score, 0.0)
        r, _ = run(held(b, 6), t)
        self.assertEqual(r.score, 0.0)  # 0.3 s after: it has moved 8 mm in the last 0.5 s
        r, _ = run(held(b, 6), t)
        self.assertEqual(r.score, 0.25)

    def test_still_needs_half_a_second_of_the_trial(self):
        r, t = run(held(nominal(), 10))
        self.assertEqual(r.score, 0.0)
        self.assertIsNone(t.moved(LEFT))
        r, _ = run(held(nominal(), 1), t)
        self.assertEqual(r.score, 1.0)

    def test_every_tier_is_reachable(self):
        seen = set()
        for boards in (START, {LEFT: start(LEFT), RIGHT: lower()},
                       {LEFT: lower(L.UPPER_Y, cx=0.230), RIGHT: lower(L.LOWER_Y, cx=0.290)}, nominal()):
            r, _ = run(held(boards))
            seen.add(r.score)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})


# -- the zero rules --------------------------------------------------------------------------------------------------
class TrapEnd(unittest.TestCase):
    def test_lower_board_9mm_off_centre_is_a_trap_end(self):
        r, _ = run(held(nominal(shift_lower=0.009)))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "trap_end")
        self.assertIn(RIGHT, r.details["trap_end"])

    def test_upper_board_slid_out_is_a_trap_end(self):
        r, _ = run(held(nominal(shift_upper=0.010)))  # outer end 40 mm past the left putlog
        self.assertEqual(r.score, 0.0)
        self.assertIn(LEFT, r.details["trap_end"])

    def test_boundaries(self):
        # A flat board in a bay: ends 22/38 mm bears; 19/41 mm leaves a trap end; 38.5/21.5 too.
        r, _ = run(held({LEFT: start(LEFT), RIGHT: lower(L.LOWER_Y - 0.008)}))
        self.assertEqual(r.score, 0.25)
        r, _ = run(held({LEFT: start(LEFT), RIGHT: lower(L.LOWER_Y - 0.0085)}))
        self.assertEqual(r.score, 0.0)
        r, _ = run(held({LEFT: start(LEFT), RIGHT: lower(L.LOWER_Y + 0.011)}))
        self.assertEqual(r.score, 0.0)

    def test_short_end_is_a_trap_end_at_the_other(self):
        r, _ = run(held({LEFT: lower(L.UPPER_Y), RIGHT: lower(L.LOWER_Y + 0.015)}))  # ends 15 / 45 mm
        self.assertEqual(r.score, 0.0)

    def test_board_on_one_putlog(self):
        r, _ = run(held({LEFT: lower(L.UPPER_Y), RIGHT: flat(L.RUN_X, 0.0, L.PUTLOG_TOP)}))  # balanced on the middle
        self.assertEqual(r.details["boards"][RIGHT]["lies_on"], {"putlog_middle": "direct"})
        self.assertEqual(r.score, 0.0)

    def test_lap_off_the_putlog_is_a_trap_end(self):
        """Lapped between putlogs, not over the middle one ((b)(7)): the upper board lies on its outer putlog only,
        so its inner end is a 180 mm cantilever. The lower board is planked correctly (ends 30 mm past its putlogs),
        so the zero comes from the upper board alone."""
        lo = lower()  # its inner end 30 mm past the middle putlog's centre line, at y +0.030
        up = upper(L.LOWER_Y + HALF[0], 0.010)  # resting on that end from y +0.010: the lap misses the centre line
        r, _ = run(held({LEFT: up, RIGHT: lo}))
        self.assertTrue(on_board(up, lo)[0])
        self.assertEqual(r.details["boards"][LEFT]["lies_on"], {"putlog_left": "direct"})
        self.assertEqual(r.details["boards"][RIGHT]["bears_in"], "right")
        self.assertEqual(r.details["trap_end"], [LEFT])
        self.assertGreater(r.details["boards"][LEFT]["overhang_mm"]["low"], 175.0)
        self.assertEqual(r.score, 0.0)

    def test_held_level_on_the_lower_end_is_a_trap_end(self):
        """The upper board held level on the lower board's end (not let go): to the scorer a cantilever."""
        boards = {LEFT: flat(L.RUN_X, L.UPPER_Y, L.LOWER_TOP + 0.001), RIGHT: lower()}
        r, _ = run(held(boards))
        self.assertEqual(r.score, 0.0)
        self.assertIn(LEFT, r.details["trap_end"])

    def test_board_leaning_on_the_table_is_ignored(self):
        """One end on the table, the other on a putlog: not on the scaffold, so no trap end; the other bay counts."""
        theta = math.atan2(0.020, 0.180)
        lean = box((L.RUN_X, 0.12, L.TOP + 0.011), rz(90.0) @ np.array(
            [[math.cos(theta), 0, -math.sin(theta)], [0, 1, 0], [math.sin(theta), 0, math.cos(theta)]]))
        self.assertLess(lean.low[2], L.TOP + L.CLEAR_OF_TABLE)
        r, _ = run(held({LEFT: lean, RIGHT: lower()}))
        self.assertFalse(r.details["boards"][LEFT]["on_scaffold"])
        self.assertEqual(r.score, 0.25)

    def test_board_on_its_edge_is_ignored(self):
        """A board standing on a 250 x 16 mm edge across a bay has no broad face down on the putlogs: ignored."""
        edge = box((L.RUN_X, L.LOWER_Y, L.PUTLOG_TOP + HALF[1]), rz(90.0) @ rx(90.0))
        self.assertAlmostEqual(tilt_deg(edge), 90.0, places=6)
        r, _ = run(held({LEFT: lower(L.UPPER_Y), RIGHT: edge}))
        self.assertFalse(r.details["boards"][RIGHT]["on_scaffold"])
        self.assertEqual(r.score, 0.25)

    def test_board_along_a_putlog_is_ignored(self):
        r, _ = run(held({LEFT: flat(L.PUTLOG_X, 0.190, L.PUTLOG_TOP, yaw=0.0), RIGHT: lower()}))
        self.assertFalse(r.details["boards"][LEFT]["on_scaffold"])
        self.assertEqual(r.score, 0.25)


# -- broken states, resets, idempotence ---------------------------------------------------------------------------
class Robustness(unittest.TestCase):
    def test_nan_state_scores_zero_and_keeps_the_record(self):
        r, t = run(held(nominal()))
        self.assertEqual(r.score, 1.0)
        bad = SimpleNamespace(**vars(nominal()[LEFT]))
        bad.pos = np.array([np.nan, 0.0, 0.0])
        r = t.update({LEFT: bad, RIGHT: nominal()[RIGHT]})
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)
        self.assertEqual(r.details["zero_reason"], "invalid_state")
        self.assertEqual(t.bad_steps, 1)
        self.assertEqual(t.moved(LEFT), 0.0)  # the history was left alone
        r = t.update(nominal())
        self.assertEqual(r.score, 1.0)

    def test_zero_quaternion_inf_and_missing_fields(self):
        b = nominal()
        for key, val in (("quat_xyzw", np.zeros(4)), ("lin_vel", np.array([np.inf, 0, 0])), ("high", None)):
            s = SimpleNamespace(**vars(b[RIGHT]))
            setattr(s, key, val)
            r, _ = run([{LEFT: b[LEFT], RIGHT: s}])
            self.assertEqual(r.score, 0.0, key)
            json.dumps(r.details, allow_nan=False)
        r, _ = run([{LEFT: b[LEFT], RIGHT: SimpleNamespace(pos=b[RIGHT].pos)}])
        self.assertEqual(r.score, 0.0)

    def test_reset_by_teleport_starts_a_new_trial(self):
        r, t = run(held(START) + held(nominal()))
        self.assertEqual(r.score, 1.0)
        self.assertIn("1.0", r.details["first_step"])
        r = t.update(START)  # the simulator puts both boards back in one step
        self.assertEqual(t.steps, 1)
        self.assertEqual(r.details["first_step"], {})
        self.assertIsNone(t.moved(LEFT))

    def test_reset_with_jitter_is_recognised(self):
        _, t = run(held(nominal()))
        jit = {LEFT: flat(L.BOARD_START_X + 0.02, 0.30, L.TOP, yaw=4.0),
               RIGHT: flat(L.BOARD_START_X - 0.02, -0.33, L.TOP, yaw=-3.0)}
        t.update(jit)
        self.assertEqual(t.steps, 1)

    def test_gradual_return_or_jump_elsewhere_is_not_a_reset(self):
        _, t = run(held(START) + held(nominal()))
        n = t.steps
        run(glide(nominal(), START, 40), t)  # carried back gradually: same trial
        self.assertEqual(t.steps, n + 40)
        _, t = run(held(nominal()))
        n = t.steps
        t.update({LEFT: flat(0.40, 0.40, L.TOP, yaw=30.0), RIGHT: lower()})  # a jump, but not onto the marks
        self.assertEqual(t.steps, n + 1)

    def test_result_is_idempotent(self):
        _, t = run(held(nominal()))
        a, b = t.result(nominal()), t.result(nominal())
        self.assertEqual(a.score, b.score)
        self.assertEqual(json.dumps(a.details, sort_keys=True), json.dumps(b.details, sort_keys=True))
        self.assertEqual(t.steps, N_STILL)

    def test_random_states_never_crash(self):
        rng = np.random.default_rng(7)
        for _ in range(300):
            boards = {}
            for n in (LEFT, RIGHT):
                q = rng.normal(size=4)
                r = SimpleNamespace(quat_xyzw=q / np.linalg.norm(q))
                from scoring import rotation
                boards[n] = box(rng.uniform([0.1, -0.4, L.TOP], [0.45, 0.4, L.TOP + 0.1]), rotation(r.quat_xyzw))
            res, _ = run(held(boards, 2))
            self.assertIn(res.score, (0.0, 0.25, 0.5, 1.0))
            json.dumps(res.details, allow_nan=False)


# -- exploits we tried ---------------------------------------------------------------------------------------------
class Exploits(unittest.TestCase):
    def test_planking_one_bay_twice_is_one_bay(self):
        """Both boards across the same (easier, nearer) bay: 0.25, not 0.5."""
        stacked = {LEFT: flat(L.RUN_X, L.LOWER_Y, L.LOWER_TOP), RIGHT: lower()}
        self.assertEqual(run(held(stacked))[0].score, 0.25)

    def test_side_by_side_is_not_a_lap(self):
        """Two boards passing each other over the middle putlog plank both bays but are not one straight run."""
        boards = {LEFT: lower(L.UPPER_Y, cx=0.235), RIGHT: lower(L.LOWER_Y, cx=0.285)}
        self.assertEqual(run(held(boards))[0].score, 0.5)

    def test_long_lap_with_a_trap_end(self):
        """Making the lap long by sliding the upper board in past the window leaves its outer end short and the
        inner end 45 mm past the middle: 0, not 1.0."""
        r, _ = run(held(nominal(shift_upper=-0.016)))
        self.assertEqual(r.score, 0.0)

    def test_upper_board_on_top_of_the_lower_board_only(self):
        """Laying the second board level on the first one, crossing the middle putlog, is a cantilever: 0."""
        boards = {LEFT: flat(L.RUN_X, 0.0, L.LOWER_TOP), RIGHT: lower()}
        r, _ = run(held(boards))
        self.assertEqual(r.score, 0.0)

    def test_pushed_off_the_end_of_the_putlogs(self):
        """Both boards lapped but hanging off the putlogs' ends (x beyond 0.32): neither bears."""
        r, _ = run(held(nominal(upper_x=0.305) | {RIGHT: lower(cx=0.305)}))
        self.assertEqual(r.score, 0.0)


# -- task.py against the offline stand-in for yam_arena ----------------------------------------------------------
def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, trace and demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")

    def test_objects_match_the_layout(self):
        names = [o.name for o in self.task.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(self.task.SCORE_OVER, "end")
        fixed = [o.name for o in self.task.OBJECTS if o.mass is None]
        self.assertEqual(fixed, ["putlog_left", "putlog_middle", "putlog_right"])
        for spec in self.task.BOARDS:
            self.assertTrue(np.allclose(spec.pos, start(spec.name).pos))
            self.assertTrue(callable(getattr(spec, "offline_state", None)))  # attached offline only

    def test_hooks_and_trace(self):
        self.task._TRACKER.reset()
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "trace.jsonl")
            os.environ["TASK_TRACE"] = path
            try:
                self.assertFalse(self.task.success(START))
                self.assertEqual(self.task.score(START), 0.0)
                for _ in range(N_STILL):
                    ok = self.task.success(nominal())
                self.assertTrue(ok)
                self.assertEqual(self.task.score(nominal()), 1.0)
            finally:
                os.environ.pop("TASK_TRACE", None)
            with open(path) as f:
                rows = [json.loads(line) for line in f]
        self.assertEqual(len(rows), 1 + N_STILL)
        for key in ("step", "score", "success", "boards", "bays_planked", "in_line_mm", "lap_mm", "upper",
                    "trap_end", "zero_reason", "moved_mm", "bad_steps", "board_left_pose", "board_right_pose"):
            self.assertIn(key, rows[-1])
        self.assertEqual(rows[-1]["score"], 1.0)

    def test_trace_survives_a_bad_path(self):
        os.environ["TASK_TRACE"] = str(HERE / "no_such_dir" / "trace.jsonl")
        try:
            self.task._TRACKER.reset()
            self.assertFalse(self.task.success(START))
        finally:
            os.environ.pop("TASK_TRACE", None)

    def test_demo_switches(self):
        self.assertEqual(set(self.task.demo()), {"left", "right"})
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(self.task.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_takes_turns_at_the_platform(self):
        """The left arm carries in only after the right arm has let go and gone up; both fit the run limit."""
        plan = self.task.demo()

        def first(wps, label_start):
            t = 0.0
            for w in wps:
                if w.label.startswith(label_start):
                    return t, t + w.duration
                t += w.duration
            raise AssertionError(label_start)

        _, right_up = first(plan["right"], "up")
        left_in, _ = first(plan["left"], "carry")
        self.assertGreaterEqual(left_in, right_up)
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertLess((total + 3.0) * L.CONTROL_HZ, 1200)
        lo = [w for w in plan["right"] if w.label.startswith("slowly set")][0]
        up = [w for w in plan["left"] if w.label.startswith("slowly set")][0]
        self.assertAlmostEqual(lo.pos[2] - self.task.GRIP_UP, L.PUTLOG_TOP + self.task.DROP, places=9)
        self.assertAlmostEqual(up.pos[2] - self.task.GRIP_UP, L.LOWER_TOP + self.task.DROP, places=9)

    def test_task_json_states_the_thresholds(self):
        """The scoring text gives the numbers the scorer uses (not the scores placeholder, which the GPU run fills)."""
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        text = spec["scoring"]
        for phrase in ("19 to 38 mm", "over 38 mm", "within 2 mm", "20 square mm", "45 degrees", "5 mm above the table",
                       "within 5 degrees", "at most 8 degrees", "x 20.0 to 32.0 cm", "2 mm in the last 0.5 s",
                       "within 10 mm", "at least 38 mm", "held for 1 s", "Possible scores: 0, 0.25, 0.5, 1.0"):
            self.assertIn(phrase, text)
        self.assertEqual((L.OVERHANG_MIN, L.OVERHANG_MAX, L.LAP_MIN), (0.019, 0.038, 0.038))
        self.assertEqual((L.REST_TOL, L.IN_LINE_TOL, L.CLEAR_OF_TABLE, L.STILL_TOL), (0.002, 0.010, 0.005, 0.002))
        self.assertEqual((L.HEADING_DEG, L.TILT_MAX_DEG, L.MEASURE_HEADING_DEG), (5.0, 8.0, 45.0))
        self.assertEqual((L.PUTLOG_X0, L.PUTLOG_X1), (0.200, 0.320))
        self.assertEqual(self.task.SUCCESS_HOLD_S, 1.0)
        self.assertIn("1,200 steps", spec["run_limit"])

    def test_instruction_is_the_same_everywhere(self):
        """task.py, task.json and both README commands give the model the same instruction."""
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)
        readme = (HERE / "README.md").read_text()
        given = [line.split('--instruction "', 1)[1].rsplit('"', 1)[0]
                 for line in readme.splitlines() if '--instruction "' in line]
        self.assertEqual(len(given), 2)
        self.assertTrue(all(g == self.task.INSTRUCTION for g in given))


if __name__ == "__main__":
    unittest.main()
