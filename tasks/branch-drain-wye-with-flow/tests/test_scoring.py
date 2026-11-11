"""Offline tests for scoring.py and task.py: synthetic part states and gripper snapshots, no simulator.
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
from scoring import (  # noqa: E402
    PARTS,
    Tracker,
    arms_valid,
    branch,
    branch_err_deg,
    flat_on_bench,
    lowest,
    pipe_line,
    point_distance,
    seat,
    tee_clear,
    valid,
    zone_gap,
    zone_radius,
)

HZ = 20.0
WINDOW = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FULL = 2 * WINDOW + 1  # 1 s of observations


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


def fitting(cx, cy, yaw=0.0, flip=False, up=0.0, pitch=0.0, roll=0.0):
    """A fitting state as mechkit reports it, its run's centre at (cx, cy) and mid-height ``up`` above where a
    fitting lying on the table has it. flip: turned upside down about its run's axis (the fitting frame's y)."""
    q = quat(yaw, pitch + (180.0 if flip else 0.0), roll)
    r = _rot(q)
    c = np.array([cx, cy, L.TOP + L.SEC / 2 + up])
    o = c - r @ np.array(L.RUN_CENTRE)
    return SimpleNamespace(origin=o, quat_xyzw=q, pos=c.copy(), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=c - 0.08, high=c + 0.08)


def pipe(cx, cy, yaw=90.0, up=0.0, pitch=0.0):
    """The branch pipe, its centre at (cx, cy); yaw 90 puts its length along x."""
    c = np.array([cx, cy, L.TOP + L.SEC / 2 + up])
    return SimpleNamespace(pos=c, quat_xyzw=quat(yaw, 0.0, pitch), lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                           low=c - 0.08, high=c + 0.08)


def pipe_on(wye_state, gap=0.005, lateral=0.0, turn=0.0, up=0.0):
    """The pipe laid along the wye's branch: its near end ``gap`` from the opening along the axis, ``lateral``
    across it, turned ``turn`` degrees off the axis about its near end."""
    e, u, th = branch(wye_state)
    n = np.array([u[1], -u[0]])
    near = e[:2] + gap * u + lateral * n
    a = math.radians(th + turn)
    axis = np.array([math.cos(a), math.sin(a)])
    c = near + L.PIPE_HALF_LEN * axis
    return pipe(c[0], c[1], yaw=th + turn - 90.0, up=up)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


FAR = {"left": arm((0.25, 0.30, L.TOP + 0.15), 0.6), "right": arm((0.26, -0.26, L.TOP + 0.15), 0.6)}

WYE_START = fitting(*L.WYE_START)
TEE_START = fitting(*L.TEE_START)
PIPE_START = pipe(*L.PIPE_START)
WYE_OK = fitting(L.DRAIN_X, 0.0, 0.0)  # seated, branch to the front and leaning upstream (theta_b 135)
PIPE_OK = pipe_on(WYE_OK)


def parts(wye=WYE_OK, tee=TEE_START, pipe_=PIPE_OK):
    return {"wye": wye, "tee": tee, "pipe": pipe_}


START = parts(WYE_START, TEE_START, PIPE_START)
DONE = parts()


def feed(tr, seq, arms=FAR):
    r = None
    for p in seq:
        r = tr.update(p, arms)
    return r


def settled(p, arms=FAR, n=FULL):
    tr = Tracker(HZ)
    return tr, feed(tr, [p] * n, arms)


def scored(p, arms=FAR):
    return settled(p, arms)[1].score


def moved(p, name, dxy, n):
    """n observations in which part ``name`` slides by dxy (total), the others as in p."""
    out = []
    base = p[name]
    for i in range(1, n + 1):
        q = dict(p)
        s = SimpleNamespace(**vars(base))
        shift = np.array([dxy[0], dxy[1], 0.0]) * (i / n - 1.0)
        for k in ("pos", "origin"):
            if hasattr(s, k):
                setattr(s, k, np.asarray(getattr(base, k)) + shift)
        q[name] = s
        out.append(q)
    return out


class Geometry(unittest.TestCase):
    def test_branch_directions_of_the_four_flat_orientations(self):
        for yaw, flip, want in ((0.0, False, 135.0), (180.0, False, -45.0), (0.0, True, 45.0),
                                (180.0, True, -135.0)):
            w = fitting(L.DRAIN_X, 0.0, yaw, flip)
            self.assertTrue(seat(w, "wye")["seated"], (yaw, flip))
            _, _, th = branch(w)
            self.assertAlmostEqual(th, want, places=6, msg=(yaw, flip))
        self.assertAlmostEqual(branch_err_deg(-45.0), 180.0)
        self.assertAlmostEqual(branch_err_deg(45.0), 90.0)
        self.assertAlmostEqual(branch_err_deg(-135.0), 90.0)
        self.assertAlmostEqual(branch_err_deg(-225.0), 0.0)
        self.assertAlmostEqual(branch_err_deg(float("nan")), 180.0)

    def test_nominal_opening_and_pipe_target(self):
        e, u, th = branch(WYE_OK)
        np.testing.assert_allclose(e[:2], [0.380 - 0.0530330, 0.0530330], atol=1e-6)
        np.testing.assert_allclose(u, [-math.sqrt(0.5), math.sqrt(0.5)], atol=1e-9)
        line = pipe_line(PIPE_OK, WYE_OK)
        self.assertAlmostEqual(line["gap_mm"], 5.0, places=6)
        self.assertAlmostEqual(line["lateral_mm"], 0.0, places=6)
        self.assertAlmostEqual(line["angle_deg"], 0.0, places=4)
        self.assertAlmostEqual(line["outward_mm"], 75.0, places=6)
        np.testing.assert_allclose(PIPE_OK.pos[:2], e[:2] + 0.075 * u, atol=1e-9)  # 70 mm half-length + 5 mm
        np.testing.assert_allclose(PIPE_OK.pos[:2], [0.2740, 0.1061], atol=1e-4)  # the spec's (0.273, 0.107) at 6 mm

    def test_start_layout(self):
        self.assertFalse(seat(WYE_START, "wye")["seated"])
        self.assertFalse(seat(TEE_START, "tee")["seated"])
        self.assertAlmostEqual(branch(WYE_START)[2], 45.0, places=6)  # starts turned back and upstream
        self.assertTrue(tee_clear(TEE_START)[0])
        self.assertGreater(tee_clear(TEE_START)[1], 0.140)
        self.assertTrue(flat_on_bench(PIPE_START, "pipe"))
        self.assertEqual(scored(START), 0.0)
        self.assertEqual(settled(START)[1].details["zero_reason"], "no_wye_seated")

    def test_every_part_starts_clear_of_the_others_and_the_gap(self):
        self.assertGreater(point_distance(L.GAP_CENTRE + (L.TOP + 0.02,), TEE_START, "tee"), 0.040)
        for a, b in (("wye", "tee"), ("wye", "pipe"), ("tee", "pipe")):
            sa, sb = START[a], START[b]
            # sample the second part's run/centre line and check it is outside the first part
            for s in np.linspace(-0.06, 0.06, 13):
                p = np.asarray(sb.pos) + _rot(sb.quat_xyzw) @ np.array([0.0, s, 0.0])
                self.assertGreater(point_distance(p, sa, a), 0.010, (a, b, s))

    def test_lowest_and_flat(self):
        self.assertAlmostEqual(lowest(WYE_OK, "wye"), 0.0, places=9)
        self.assertAlmostEqual(lowest(fitting(L.DRAIN_X, 0.0, 0.0, True), "wye"), 0.0, places=9)
        self.assertAlmostEqual(lowest(PIPE_OK, "pipe"), 0.0, places=9)
        self.assertFalse(flat_on_bench(pipe(0.27, 0.10, 45.0, up=0.040), "pipe"))  # lying on something 40 mm up
        self.assertFalse(flat_on_bench(pipe(0.27, 0.10, 45.0, pitch=6.0), "pipe"))


class Tiers(unittest.TestCase):
    def test_demo_end_state_scores_one(self):
        tr, r = settled(DONE)
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["line"], 4)

    def test_correct_wye_alone(self):
        self.assertEqual(scored(parts(pipe_=PIPE_START)), 0.5)

    def test_pipe_short_of_the_opening_is_coarse(self):
        for gap, want in ((0.010, 1.0), (0.0105, 0.75), (0.020, 0.75), (0.025, 0.75), (0.0399, 0.75),
                          (0.0405, 0.5), (-0.0029, 1.0), (-0.0031, 0.5)):
            self.assertEqual(scored(parts(pipe_=pipe_on(WYE_OK, gap=gap))), want, gap)

    def test_lateral_and_angle_windows(self):
        for lat, want in ((0.0059, 1.0), (-0.0059, 1.0), (0.0061, 0.75), (0.0149, 0.75), (-0.0151, 0.5)):
            self.assertEqual(scored(parts(pipe_=pipe_on(WYE_OK, lateral=lat))), want, lat)
        for turn, want in ((6.9, 1.0), (-6.9, 1.0), (7.1, 0.75), (-14.9, 0.75), (15.1, 0.5)):
            self.assertEqual(scored(parts(pipe_=pipe_on(WYE_OK, turn=turn))), want, turn)

    def test_pipe_reversed_end_for_end_is_the_same_pipe(self):
        p = PIPE_OK
        flipped = SimpleNamespace(**vars(p))
        flipped.quat_xyzw = quat(225.0)  # the same box, its length the other way round
        self.assertEqual(scored(parts(pipe_=flipped)), 1.0)

    def test_wrong_wye_orientations_score_a_quarter(self):
        for yaw, flip in ((180.0, False), (0.0, True), (180.0, True)):
            w = fitting(L.DRAIN_X, 0.0, yaw, flip)
            # shortcut checks 2 and 3: the pipe laid against the wrong branch changes nothing
            tr, r = settled(parts(wye=w, pipe_=pipe_on(w)))
            self.assertEqual(r.score, 0.25, (yaw, flip))
            self.assertEqual(r.details["zero_reason"], "wye_branch_wrong_way")

    def test_branch_window_edges(self):
        # a seated wye's run is within 5 degrees of the drain, so its branch is within 5 degrees of one of 135, 45,
        # -45 and -135: the 7 degree window only tells the four apart
        for dyaw in (4.9, -4.9):
            w = fitting(L.DRAIN_X, 0.0, dyaw)
            self.assertAlmostEqual(branch_err_deg(branch(w)[2]), 4.9, places=6)
            self.assertEqual(scored(parts(wye=w, pipe_=pipe_on(w))), 1.0)
        for dyaw in (5.1, -5.1, 7.1):
            self.assertFalse(seat(fitting(L.DRAIN_X, 0.0, dyaw), "wye")["seated"])
            self.assertEqual(scored(parts(wye=fitting(L.DRAIN_X, 0.0, dyaw))), 0.0)
        for yaw, flip in ((184.9, False), (4.9, True), (-175.1, True)):
            w = fitting(L.DRAIN_X, 0.0, yaw, flip)
            self.assertGreater(branch_err_deg(branch(w)[2]), 80.0)
            self.assertEqual(scored(parts(wye=w)), 0.25)

    def test_seat_window_edges(self):
        for dx, dy, want in ((0.0059, 0.0, 1.0), (-0.0059, 0.0059, 1.0), (0.0061, 0.0, 0.0), (0.0, -0.0061, 0.0)):
            w = fitting(L.DRAIN_X + dx, dy, 0.0)
            self.assertEqual(scored(parts(wye=w, pipe_=pipe_on(w))), want, (dx, dy))
        # roll tilts the 118 mm run end to end: 2 degrees drops an end 2.1 mm (still within 3 mm of the table)
        for kw, want in (({"pitch": 4.9}, 1.0), ({"roll": 2.0}, 1.0), ({"pitch": 5.1}, 0.0), ({"roll": -5.1}, 0.0),
                         ({"up": 0.0029}, 1.0), ({"up": 0.0031}, 0.0), ({"up": -0.0031}, 0.0)):
            w = fitting(L.DRAIN_X, 0.0, 0.0, **kw)
            p = pipe_on(fitting(L.DRAIN_X, 0.0, 0.0))
            self.assertEqual(scored(parts(wye=w, pipe_=p)), want, kw)

    def test_sanitary_tee_in_the_gap_scores_zero(self):
        # shortcut check 1: the tee pushed straight back into the gap at its start yaw (branch to the back)
        t = fitting(L.DRAIN_X, 0.0, L.TEE_START[2])
        self.assertTrue(seat(t, "tee")["seated"])
        tr, r = settled(parts(wye=WYE_START, tee=t, pipe_=PIPE_START))
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "sanitary_tee_in_the_gap")
        for yaw, flip in ((0.0, False), (0.0, True), (180.0, True)):  # any way round, the pipe at its branch
            t = fitting(L.DRAIN_X, 0.0, yaw, flip)
            self.assertEqual(scored(parts(wye=WYE_START, tee=t, pipe_=PIPE_OK)), 0.0)

    def test_tee_seated_beats_a_correct_looking_wye(self):
        # a correct wye cannot also be in the gap; a wye laid just in front of it with the tee seated is still 0
        t = fitting(L.DRAIN_X, 0.0, 0.0)
        w = fitting(0.30, -0.10, 0.0)
        self.assertEqual(scored(parts(wye=w, tee=t, pipe_=PIPE_OK)), 0.0)

    def test_tee_not_clear_or_not_still_caps_at_three_quarters(self):
        near = fitting(0.320, -0.040, 0.0)  # flat on the bench, 72 mm from the gap's centre: clear
        self.assertTrue(tee_clear(near)[0])
        self.assertEqual(scored(parts(tee=near)), 1.0)
        nearer = fitting(0.330, -0.025, 0.0)  # 56 mm: not clear
        self.assertFalse(tee_clear(nearer)[0])
        self.assertEqual(scored(parts(tee=nearer)), 0.75)
        on_drain = fitting(L.DRAIN_X, -0.15, 0.0, up=L.DRAIN_H)  # lying on the drain's top
        self.assertFalse(tee_clear(on_drain)[0])
        self.assertEqual(scored(parts(tee=on_drain)), 0.75)
        sunk = fitting(0.25, -0.15, 0.0, roll=90.0, up=0.02)  # run upright but 19 mm into the table: not resting
        self.assertEqual(scored(parts(tee=sunk)), 0.75)
        # knocked over on the bench, away from the gap, it is still 'on the bench, off the drain': any face counts
        branch_up = fitting(0.25, -0.15, 0.0, pitch=90.0)  # on the run's side, the branch pointing up
        branch_down = fitting(0.25, -0.15, 0.0, pitch=-90.0, up=L.BRANCH_REACH - L.SEC / 2)  # on the branch's end
        on_end = fitting(0.25, -0.15, 0.0, roll=90.0, up=L.RUN_LEN / 2 - L.SEC / 2)  # standing on a run end
        for t in (branch_up, branch_down, on_end, fitting(0.25, -0.15, 30.0, flip=True)):
            self.assertTrue(tee_clear(t)[0])
            self.assertEqual(scored(parts(tee=t)), 1.0)
        lean = math.radians(20.0)  # propped at 20 degrees, its lowest corner on the table: only the angle fails
        leaning = fitting(0.25, -0.15, 0.0, roll=20.0,
                          up=L.RUN_LEN / 2 * math.sin(lean) + L.SEC / 2 * math.cos(lean) - L.SEC / 2)
        side_on_drain = fitting(L.DRAIN_X, -0.15, 0.0, pitch=90.0, up=L.DRAIN_H)  # on its side on the drain's top
        hovering = fitting(0.25, -0.15, 0.0, up=0.004)  # 4 mm up: resting on something, not the bench
        for t in (leaning, side_on_drain, hovering):
            self.assertFalse(tee_clear(t)[0])
            self.assertEqual(scored(parts(tee=t)), 0.75)
        # the tee moved during the first of the last two windows: not still for 1 s yet
        tr = Tracker(HZ)
        feed(tr, moved(DONE, "tee", (0.02, 0.0), WINDOW))
        r = feed(tr, [DONE] * (WINDOW + 1))
        self.assertEqual(r.score, 0.75)
        r = feed(tr, [DONE] * WINDOW)
        self.assertEqual(r.score, 1.0)

    def test_pipe_not_flat_on_the_bench(self):
        self.assertEqual(scored(parts(pipe_=pipe_on(WYE_OK, up=0.040))), 0.5)  # laid on top of the branch
        self.assertEqual(scored(parts(pipe_=pipe_on(WYE_OK, gap=-0.020, up=0.040))), 0.5)
        tipped = pipe_on(WYE_OK)
        tipped = SimpleNamespace(**vars(tipped))
        tipped.quat_xyzw = quat(45.0, 0.0, 8.0)
        self.assertEqual(scored(parts(pipe_=tipped)), 0.5)

    def test_pipe_pointing_inward_does_not_count(self):
        # its near end at the opening but the pipe lying back over the wye (impossible on the bench, but checked)
        e, u, th = branch(WYE_OK)
        c = e[:2] + 0.005 * u - L.PIPE_HALF_LEN * u
        self.assertLess(pipe_line(pipe(c[0], c[1], th - 90.0), WYE_OK)["out"], L.OUTWARD)
        self.assertEqual(scored(parts(pipe_=pipe(c[0], c[1], th - 90.0))), 0.5)

    def test_every_score_is_one_of_the_five(self):
        rng = random.Random(3)
        seen = set()
        for _ in range(300):
            w = fitting(L.DRAIN_X + rng.uniform(-0.01, 0.01), rng.uniform(-0.01, 0.01), rng.choice((0, 180, 3, -8)),
                        rng.random() < 0.3)
            t = fitting(rng.uniform(0.22, 0.40), rng.uniform(-0.2, 0.0), rng.uniform(-180, 180))
            p = pipe_on(w, gap=rng.uniform(-0.01, 0.05), lateral=rng.uniform(-0.02, 0.02), turn=rng.uniform(-20, 20))
            if rng.random() < 0.15:  # some layouts as the demo leaves them
                w, p = fitting(L.DRAIN_X, 0.0, rng.uniform(-3, 3)), None
                p = pipe_on(w, gap=rng.uniform(0.0, 0.009), lateral=rng.uniform(-0.005, 0.005))
            s = scored(parts(w, t, p))
            self.assertIn(s, (0.0, 0.25, 0.5, 0.75, 1.0))
            seen.add(s)
        self.assertEqual(seen, {0.0, 0.25, 0.5, 0.75, 1.0})


class Counting(unittest.TestCase):
    def test_held_wye_is_not_counted(self):
        # shortcut check 5: the demo's end layout with the right gripper still closed on the wye
        grip = arm((L.DRAIN_X, -0.040, L.TOP + 0.024), 32.0 / 93.9)
        tr, r = settled(DONE, {"left": FAR["left"], "right": grip})
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["zero_reason"], "wye_held_or_moving")
        self.assertLess(zone_gap(grip, WYE_OK, "wye"), 0.0)

    def test_open_gripper_hovering_close_is_not_released(self):
        hover = arm((L.DRAIN_X, -0.040, L.TOP + 0.040 + 0.013 + 0.010 + zone_radius(arm((0, 0, 0)))), 1.0)
        g = zone_gap(hover, WYE_OK, "wye")
        self.assertGreater(g, 0.005)
        self.assertLess(g, L.HANDS_OFF)
        self.assertEqual(scored(DONE, {"left": FAR["left"], "right": hover}), 0.0)
        away = arm((L.DRAIN_X, -0.040, L.TOP + 0.040 + 0.013 + 0.016 + zone_radius(arm((0, 0, 0)))), 1.0)
        self.assertEqual(scored(DONE, {"left": FAR["left"], "right": away}), 1.0)

    def test_released_must_hold_over_the_whole_window(self):
        grip = arm((L.DRAIN_X, -0.040, L.TOP + 0.024), 32.0 / 93.9)
        tr = Tracker(HZ)
        feed(tr, [DONE] * FULL, {"left": FAR["left"], "right": grip})
        r = feed(tr, [DONE] * WINDOW)
        self.assertEqual(r.score, 0.0)  # the gripper was on it 10 steps ago, inside the last 0.5 s
        r = feed(tr, [DONE])
        self.assertEqual(r.score, 1.0)  # clear for the whole window: it had been at rest all along

    def test_current_arms_count_too(self):
        tr, r = settled(DONE)
        grip = arm((L.DRAIN_X, -0.040, L.TOP + 0.024), 32.0 / 93.9)
        self.assertEqual(tr.result(DONE, {"left": FAR["left"], "right": grip}).score, 0.0)
        self.assertEqual(tr.result(DONE, FAR).score, 1.0)

    def test_held_pipe_is_not_counted(self):
        e, u, th = branch(WYE_OK)
        grip = arm((PIPE_OK.pos[0], PIPE_OK.pos[1], L.TOP + 0.024), 32.0 / 93.9)
        self.assertEqual(scored(DONE, {"left": grip, "right": FAR["right"]}), 0.5)

    def test_moving_or_turning_parts_are_not_counted(self):
        tr = Tracker(HZ)
        r = feed(tr, moved(DONE, "wye", (0.0, 0.006), FULL))
        self.assertEqual(r.score, 0.0)  # 2.9 mm in the last 0.5 s
        tr = Tracker(HZ)
        r = feed(tr, moved(DONE, "wye", (0.0, 0.004), FULL))  # 1.9 mm per 0.5 s: settled
        self.assertEqual(r.score, 1.0)
        tr = Tracker(HZ)
        r = feed(tr, moved(DONE, "pipe", (0.006, 0.0), FULL))
        self.assertEqual(r.score, 0.5)
        # turning 1.2 degrees over the last 0.5 s, about the run's centre
        tr = Tracker(HZ)
        seq = []
        for i in range(FULL):
            a = 1.2 * max(0, i - WINDOW) / WINDOW - 1.2
            w = fitting(L.DRAIN_X, 0.0, a)
            seq.append(parts(wye=w, pipe_=PIPE_OK))
        self.assertEqual(feed(tr, seq).score, 0.0)
        self.assertEqual(Tracker(HZ).update(DONE, FAR).score, 0.0)  # no history yet: nothing is counted
        tr = Tracker(HZ)
        self.assertEqual(feed(tr, [DONE] * (WINDOW + 1)).score, 0.75)  # 0.5 s: counted, not still for 1 s
        self.assertEqual(feed(tr, [DONE] * WINDOW).score, 1.0)

    def test_nothing_moved_scores_zero(self):
        self.assertEqual(scored(START), 0.0)


class Exploits(unittest.TestCase):
    def test_pipe_in_the_gap_instead_of_a_fitting(self):
        self.assertEqual(scored(parts(wye=WYE_START, pipe_=pipe(L.DRAIN_X, 0.0, 0.0))), 0.0)

    def test_wye_pushed_in_without_turning(self):
        # slid straight in from its start (theta_b 45, run along x): not along the drain
        self.assertEqual(scored(parts(wye=fitting(L.DRAIN_X, 0.0, -90.0))), 0.0)
        # slid in and turned the wrong way (-90 -> -180): branch to the back, downstream
        w = fitting(L.DRAIN_X, 0.0, -180.0)
        self.assertEqual(scored(parts(wye=w, pipe_=pipe_on(w))), 0.25)

    def test_both_fittings_stacked_in_the_gap(self):
        t = fitting(L.DRAIN_X, 0.0, 180.0, up=L.SEC)  # the tee on top of the seated wye
        self.assertFalse(seat(t, "tee")["seated"])
        self.assertEqual(scored(parts(tee=t)), 0.75)

    def test_tee_laid_along_the_branch_as_the_pipe(self):
        e, u, th = branch(WYE_OK)
        c = e[:2] + 0.065 * u
        t = fitting(c[0], c[1], th - 90.0)
        self.assertEqual(scored(parts(tee=t, pipe_=PIPE_START)), 0.5)

    def test_wye_on_the_drain_or_on_its_side(self):
        self.assertEqual(scored(parts(wye=fitting(L.DRAIN_X, 0.10, 0.0, up=L.DRAIN_H))), 0.0)
        self.assertEqual(scored(parts(wye=fitting(L.DRAIN_X, 0.0, 0.0, roll=90.0, up=0.0))), 0.0)


class Arms(unittest.TestCase):
    def test_unreadable_arms_taken_as_met_and_counted(self):
        tr, r = settled(DONE, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(tr.arms_unavailable, FULL)

    def test_non_finite_arms_count_as_unavailable(self):
        bad = {"left": arm((float("nan"), 0.0, 0.0)), "right": FAR["right"]}
        self.assertFalse(arms_valid(bad))
        self.assertFalse(arms_valid({"left": FAR["left"]}))
        tr, r = settled(DONE, bad)
        self.assertEqual(tr.arms_unavailable, FULL)

    def test_zone_radius(self):
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 1.0)), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 0.0)), math.hypot(0.008, 0.0135))


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, _ = settled(DONE)
        n_hist = len(tr.history)
        bads = []
        w = SimpleNamespace(**vars(WYE_OK))
        w.origin = np.array([float("nan"), 0.0, 0.0])
        bads.append(parts(wye=w))
        w = SimpleNamespace(**vars(WYE_OK))
        w.quat_xyzw = np.zeros(4)
        bads.append(parts(wye=w))
        w = SimpleNamespace(**{k: v for k, v in vars(WYE_OK).items() if k != "origin"})
        bads.append(parts(wye=w))
        p = SimpleNamespace(**vars(PIPE_OK))
        p.pos = np.array([0.0, float("inf"), 0.0])
        bads.append(parts(pipe_=p))
        bads.append({"wye": WYE_OK, "tee": TEE_START})
        bads.append(parts(tee=None))
        for b in bads:
            r = tr.update(b, FAR)
            self.assertEqual(r.score, 0.0)
            self.assertEqual(r.details["zero_reason"], "invalid_state")
            self.assertEqual(len(tr.history), n_hist)
        self.assertEqual(tr.update(DONE, FAR).score, 1.0)  # the record was kept

    def test_unnormalised_quaternion(self):
        w = SimpleNamespace(**vars(WYE_OK))
        w.quat_xyzw = np.asarray(WYE_OK.quat_xyzw) * 3.0
        self.assertTrue(valid(w, True))
        self.assertEqual(scored(parts(wye=w)), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rng = np.random.default_rng(7)
        tr = Tracker(HZ)
        for _ in range(300):
            ps = {}
            for n in PARTS:
                q = rng.normal(size=4)
                c = rng.uniform([0.0, -0.5, 0.6], [0.6, 0.5, 1.0])
                ps[n] = SimpleNamespace(origin=c, pos=c, quat_xyzw=q, lin_vel=np.zeros(3), ang_vel=np.zeros(3),
                                        low=c, high=c)
            r = tr.update(ps, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class Reset(unittest.TestCase):
    def test_teleport_clears_the_record(self):
        tr, r = settled(DONE)
        self.assertEqual(r.score, 1.0)
        r = tr.update(START, FAR)  # every part put back at once: a new episode
        self.assertEqual(tr.jumps, 1)
        self.assertEqual(len(tr.history), 1)
        self.assertEqual(r.score, 0.0)
        r = tr.update(DONE, FAR)  # and straight back again: nothing is counted until it has been seen at rest
        self.assertEqual(r.score, 0.0)
        r = feed(tr, [DONE] * FULL)
        self.assertEqual(r.score, 1.0)

    def test_explicit_reset(self):
        tr, _ = settled(DONE)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.arms_unavailable), (0, 0, 0))
        self.assertEqual(tr.update(DONE, FAR).score, 0.0)

    def test_a_slow_slide_is_not_a_reset(self):
        tr = Tracker(HZ)
        feed(tr, moved(DONE, "pipe", (0.0, 0.07), 20))
        self.assertEqual(tr.jumps, 0)


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, the demo."""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(_stub_dir()))
        cls.task = importlib.import_module("task")
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate

    def sim(self, p):
        """Synthetic states turned into what yam_arena and mechkit report from the simulator."""
        t = self.task
        return {t.NAMES["wye"]: self.mk.state_from_poses(t.WYE, (*p["wye"].origin, *p["wye"].quat_xyzw), source="sim"),
                t.NAMES["tee"]: self.mk.state_from_poses(t.TEE, (*p["tee"].origin, *p["tee"].quat_xyzw), source="sim"),
                t.NAMES["pipe"]: p["pipe"]}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(sorted(o.name for o in t.OBJECTS if o.mass is not None),
                         ["branch_pipe", "sanitary_tee", "wye"])
        for o in t.OBJECTS:
            if o.mass is not None:
                self.assertLessEqual(o.mass, 0.30)
        self.assertEqual(t.WYE.mass, t.TEE.mass)  # the choice is by shape alone
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        dn, up = t.DRAIN
        self.assertAlmostEqual(up.pos[1] - up.size[1] / 2 - (dn.pos[1] + dn.size[1] / 2), 0.134)  # the gap
        self.assertAlmostEqual(L.GAP_HALF * 2 - L.RUN_LEN, 0.016)  # 8 mm clearance at each end
        for b in t.DRAIN:
            self.assertAlmostEqual(b.pos[0], L.DRAIN_X)
            self.assertAlmostEqual(b.pos[2] + b.size[2] / 2, L.TOP + L.DRAIN_H)
        # arrows on the drain's top, each pointing -y: its narrowest part (the tip) at the lowest y
        for i, y in enumerate(L.ARROW_Y):
            pts = [o for o in t.ARROWS if o.name.startswith(f"flow_arrow_{i + 1}_")]
            by_y = sorted(pts, key=lambda o: o.pos[1])  # from the tip (lowest y) back
            self.assertEqual([o.size[0] for o in by_y[:3]], sorted(o.size[0] for o in by_y[:3]))  # head widens
            self.assertEqual(max(by_y, key=lambda o: o.size[1]), by_y[-1])  # the long shaft behind it
            self.assertAlmostEqual(max(o.pos[1] + o.size[1] / 2 for o in pts)
                                   - min(o.pos[1] - o.size[1] / 2 for o in pts), 0.030)
            self.assertTrue(L.GAP_HALF < abs(y) - 0.015 and abs(y) + 0.015 < L.DRAIN_END)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                half = np.abs(_rot(quat(math.degrees(o.yaw)))) @ (np.asarray(o.size) / 2)
                lo, hi = np.asarray(o.pos) - half, np.asarray(o.pos) + half
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)
            self.assertGreaterEqual(lo[0], 0.205, o.name)

    def test_fitting_geometry_matches_the_scorer(self):
        t = self.task
        lo, hi = t.WYE.local_bounds()
        np.testing.assert_allclose(lo, [-(0.055 * math.sqrt(0.5) + 0.02), -0.059, 0.0], atol=1e-9)
        np.testing.assert_allclose(hi, [0.02, 0.059, 0.04], atol=1e-9)
        tip = np.array(L.WYE_TIP[:2])
        last = np.array(L.WYE_BEADS[-1][:2])
        self.assertAlmostEqual(float(np.linalg.norm(tip - last)), L.BEAD_R)  # the last disc's rim is the opening
        self.assertAlmostEqual(float(np.linalg.norm(tip)), L.BRANCH_REACH)
        # consecutive discs 7 mm apart: the branch's sides are smooth to 0.31 mm
        self.assertLess(L.BEAD_R - math.sqrt(L.BEAD_R ** 2 - 0.0035 ** 2), 0.0004)
        lo, hi = t.TEE.local_bounds()
        np.testing.assert_allclose(lo, [-0.075, -0.059, 0.0], atol=1e-9)
        np.testing.assert_allclose(hi, [0.02, 0.059, 0.04], atol=1e-9)
        # the scorer's solid matches mechkit's shapes: points inside the discs are inside, a point between the
        # run's face and the branch root outside the strip is outside
        st = self.mk.state_from_poses(t.WYE, (*WYE_OK.origin, *WYE_OK.quat_xyzw), source="sim")
        for c in L.WYE_BEADS:
            self.assertEqual(point_distance(st.to_world(c), st, "wye"), 0.0)
        self.assertGreater(point_distance(st.to_world((-0.030, -0.030, 0.02)), st, "wye"), 0.010)
        com = t.WYE.center_of_mass()
        self.assertLess(float(np.linalg.norm(com[:2] - t.WYE_GRIP_LOCAL)), 0.060)  # 0.06 N m on the pads at most
        # the wye's pads (27 mm wide, centred 40 mm downstream of the run's centre) stay 17 mm or more from the
        # branch's root, whose downstream edge meets the run's face 8.3 mm from the centre
        root_y = -L.SEC / 2 * math.sqrt(2) + L.SEC / 2
        self.assertAlmostEqual(root_y, -0.0083, places=4)
        self.assertGreater(root_y - (t.WYE_GRIP_LOCAL[1] + L.FINGER_HALF_WIDTH), 0.017)
        self.assertGreater(t.WYE_GRIP_LOCAL[1] - L.FINGER_HALF_WIDTH, -L.RUN_LEN / 2)  # and on the run

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.WYE.offline_box()[0]), quat_xyzw=quat(L.WYE_START[2]),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.WYE.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.WYE_START[0], L.WYE_START[1], L.TOP], atol=1e-12)
        np.testing.assert_allclose(self.task.WYE.offline_box()[1], L.RUN_SIZE)

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        start = self.sim(START)
        self.assertFalse(t.success(start))
        self.assertEqual(t.score(start), 0.0)
        done = self.sim(DONE)
        for _ in range(FULL):
            self.feed_arms(FAR)
            ok = t.success(done)
        self.assertTrue(ok)
        self.assertEqual(t.score(done), 1.0)
        self.assertEqual(t.score(done), 1.0)  # score() does not advance it
        grip = arm((L.DRAIN_X, -0.040, L.TOP + 0.024), 32.0 / 93.9)
        self.feed_arms({"left": FAR["left"], "right": grip})
        self.assertEqual(t.score(done), 0.0)  # read with the arms of the step at hand... and the window
        self.feed_arms(FAR)
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(FAR)
        t.success(done)
        self.assertEqual(tr.steps, 1)
        t.success(done)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(done))
        self.assertEqual(tr.steps, 1)
        self.assertLess(t.distance(done), 1e-9)

    def test_trace_rows_and_a_bad_path(self):
        old = os.environ.get("TASK_TRACE")
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "trace.jsonl"
            os.environ["TASK_TRACE"] = str(path)
            try:
                self.rig.offline_reset()
                self.feed_arms(FAR)
                self.task.success(self.sim(DONE))
                row = json.loads(path.read_text().splitlines()[-1])
                for k in ("step", "score", "success", "line", "zero_reason", "wye", "tee", "pipe", "theta_b_deg",
                          "branch_err_deg", "tip_E", "u", "gap_mm", "lateral_mm", "angle_deg", "outward_mm",
                          "pipe_flat", "pipe_fine", "pipe_coarse", "tee_clear", "tee_dist_mm", "arms_unavailable",
                          "wye_origin", "pipe_pos", "demo_pipe_target", "demo_wye_target", "right_tcp", "openings"):
                    self.assertIn(k, row)
                for k in ("seated", "off_x_mm", "off_y_mm", "run_axis_err_deg", "tilt_deg", "lowest_mm", "counted",
                          "still_1s", "released", "path_mm", "turn_deg"):
                    self.assertIn(k, row["wye"])
                self.assertAlmostEqual(row["theta_b_deg"], 135.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(DONE)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if prev.finger != w.finger:
                    self.assertGreaterEqual(w.duration, 1.5, (side, w.label))
                if w.label == "settle" or w.label.startswith("settle over"):
                    self.assertGreaterEqual(w.duration, 0.5)
            slow = [w for w in wps if w.label.startswith("slowly down")]
            self.assertEqual(len(slow), 2)
            self.assertTrue(all(w.duration >= 1.5 for w in slow))
            self.assertTrue(all(w.pos[2] - 0.013 >= L.TOP + 0.010 for w in wps))  # pad tips 10 mm up or more
        right, left = plan["right"], plan["left"]
        self.assertAlmostEqual(sum(w.duration for w in right), 20.3)
        self.assertAlmostEqual(sum(w.duration for w in left), 31.1)
        # the left arm only moves the pipe toward the drain once the right arm has let go and lifted
        t_left = 0.0
        for w in left:
            t_left += w.duration
            if w.label.startswith("hold it up"):
                break
        self.assertAlmostEqual(t_left, t._right_released_s())
        self.assertAlmostEqual(t._right_released_s(), 18.8)
        # yaws: the wye turns +88.85 (from -90 to -1.15 degrees), the pipe -46.15 (from 90 to 43.85)
        self.assertAlmostEqual(math.degrees(t.WYE_YAW_SET - t.WYE_YAW_PICK), 88.85)
        self.assertAlmostEqual(math.degrees(t.WYE_LAID_YAW), -1.15)
        self.assertAlmostEqual(math.degrees(t.PIPE_YAW_SET - t.PIPE_YAW_PICK), -46.15)
        # every waypoint's yaw within 85 degrees of its point's bearing from the arm's base (joint 6's window)
        for side, wps, by in (("right", right, -0.305), ("left", left, 0.305)):
            for w in wps:
                bearing = math.degrees(math.atan2(w.pos[1] - by, w.pos[0]))
                off = (math.degrees(w.yaw) - bearing + 90.0) % 180.0 - 90.0
                self.assertLess(abs(off), 85.0, (side, w.label))
        # both last horizontal moves run 25 mm along (+x, -y), minus the correct branch axis
        for wps in (right, left):
            i = next(k for k, w in enumerate(wps) if w.label.startswith("last 25 mm"))
            d = np.asarray(wps[i].pos[:2]) - np.asarray(wps[i - 1].pos[:2])
            self.assertAlmostEqual(float(np.linalg.norm(d)), 0.025)
            np.testing.assert_allclose(d / np.linalg.norm(d), [math.sqrt(0.5), -math.sqrt(0.5)], atol=1e-9)
        # the right gripper's pads stay 10 mm or more inside the gap's ends at the place (closing along x)
        self.assertLess(t.WYE_SET[1] + 0.0135 - 0.0, 0.0)
        self.assertGreater(t.WYE_SET[1] - 0.0135, -L.GAP_HALF + 0.010)
        # carried 26 mm over the drain and the tee
        self.assertGreaterEqual(t.HOVER - t.GRIP_UP - (L.TOP + L.DRAIN_H), 0.025)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_demo_outcome_offline_and_with_the_gpu_errors(self):
        """Where the parts end: offline exactly as commanded; on the GPU each part turned +0.6..+1.7 degrees in the
        hand (about its grip point) and landed 2-6 mm short along the last move, independently. Every case is
        inside the fine window with the wye seated and correct."""
        t = self.task

        def rot2(a):
            c, s = math.cos(a), math.sin(a)
            return np.array([[c, -s], [s, c]])

        def outcome(bw, bp, sw, sp):
            d = t.APPROACH_DIR
            tw = np.asarray(t.WYE_SET[:2])
            cw = tw + rot2(math.radians(bw)) @ (t.WYE_AIM - tw) - sw * d
            w = fitting(cw[0], cw[1], math.degrees(t.WYE_LAID_YAW) + bw)
            cp = np.asarray(t.PIPE_SET[:2]) - sp * d
            p = pipe(cp[0], cp[1], L.PIPE_START[2] + math.degrees(t.PIPE_YAW_SET) + bp)
            return w, p, pipe_line(p, w), seat(w, "wye")

        w, p, line, st = outcome(0.0, 0.0, 0.0, 0.0)
        self.assertAlmostEqual(line["gap_mm"], 5.5, places=1)
        self.assertAlmostEqual(line["lateral_mm"], -1.6, delta=0.1)
        self.assertEqual(scored(parts(wye=w, pipe_=p)), 1.0)
        gaps, lats = [], []
        for bw in (0.6, 1.15, 1.7):
            for bp in (0.6, 1.15, 1.7):
                for sw in (0.002, 0.004, 0.006):
                    for sp in (0.002, 0.004, 0.006):
                        w, p, line, st = outcome(bw, bp, sw, sp)
                        self.assertTrue(st["seated"])
                        self.assertLessEqual(branch_err_deg(branch(w)[2]), 1.0)
                        self.assertTrue(-3.0 + 2.0 <= line["gap_mm"] <= 10.0 - 0.5)  # no touch, inside the window
                        self.assertLess(abs(line["lateral_mm"]), 4.0)
                        gaps.append(line["gap_mm"])
                        lats.append(line["lateral_mm"])
        self.assertGreater(min(gaps), 0.5)
        self.assertLess(max(gaps), 9.5)
        # the wye's run stays 4 mm or more off both cut ends of the drain in every case
        w, *_ = outcome(1.7, 1.7, 0.006, 0.002)
        self.assertLess(abs(w.pos[1]) + L.RUN_LEN / 2, L.GAP_HALF - 0.004)

    def test_instruction_withholds_the_answer(self):
        ins = self.task.INSTRUCTION.lower()
        for s in ("arrows", "flows", "front side", "fixture", "plumbing code", "in line with the drain",
                  "at the opening", "off the drain", "let go"):
            self.assertIn(s, ins)
        for s in ("wye", "sanitary", " tee", "upstream", "downstream", "45", "135", "degree"):
            self.assertNotIn(s, ins)
        with open(HERE / "task.json") as f:
            spec = json.load(f)
        self.assertEqual(spec["instruction"], self.task.INSTRUCTION)


if __name__ == "__main__":
    unittest.main()
