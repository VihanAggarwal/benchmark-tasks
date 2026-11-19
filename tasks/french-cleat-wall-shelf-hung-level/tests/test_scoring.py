"""Offline tests for layout.py, scoring.py and task.py: synthetic shelf states (mechkit AssemblyStates from the task's
own spec) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

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
            sys.path.insert(0, str(_p))

try:
    import layout as L  # noqa: E402
    import mechkit as mk  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))
KIT_DIR = HERE.parents[1] / "tools" / "kit"
DEG = math.pi / 180.0


def shelf(x=L.SH_SEAT_X, y=0.0, z=L.SEAT, yaw=0.0, pitch=0.0, roll=0.0):
    """The shelf with its unit frame's origin at (x, y, z above the table), turned yaw, pitch, roll."""
    q = mk.quat_from_ypr(yaw, pitch, roll)
    return mk.state_from_poses(T.SHELF_PART, (x, y, L.TOP + z, *q), source="test")


def crooked(about_y, deg):
    """The seated shelf turned ``deg`` about the x axis through the bevel's middle at world y ``about_y``."""
    phi = deg * DEG
    p = np.array([L.WALL_X - 0.006, about_y, L.SEAT + 0.016])
    o = p + mk.axis_angle_matrix((1.0, 0.0, 0.0), phi) @ (np.array([L.SH_SEAT_X, 0.0, L.SEAT]) - p)
    return shelf(o[0], o[1], o[2], roll=phi)


def along_slope(mm):
    """The seated shelf slid ``mm`` up the bevel, toward the room (its bevel face stays on the wall's plane)."""
    d = mm / 1000.0
    return shelf(L.SH_SEAT_X - d, 0.0, L.SEAT + d)


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.45, L.TOP + 0.25)), "right": arm((0.1, -0.45, L.TOP + 0.25))}


def states(s):
    return {S.SHELF: s}


def run(s, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states(s), arms)
    return res


def score(s, arms=FAR):
    return run(s, arms).score


class Geometry(unittest.TestCase):
    def test_sizes(self):
        self.assertAlmostEqual(L.SH_W, 0.320)
        self.assertAlmostEqual(L.SH_D, 0.100)
        self.assertAlmostEqual(L.SH_H, 0.080)
        self.assertGreaterEqual(L.END_T, 0.016)  # the gripper pinches 16 mm and up
        self.assertAlmostEqual(L.CLEAT_T, 0.012)
        self.assertAlmostEqual(L.SH_MASS, 0.53, places=2)
        self.assertLess(L.SH_MASS * L.G / 2, 3.0)  # each arm's share, under the arms' 5 N

    def test_bevels_are_45_degrees_and_the_same_line(self):
        self.assertAlmostEqual(L.bevel_z(L.WALL_X - L.CLEAT_T) - L.bevel_z(L.WALL_X), 0.012)
        self.assertAlmostEqual(L.bevel_z(L.WALL_X), L.SEAT + 0.010)
        for lx in (0.050, 0.056, 0.062):  # the unit cleat's lower edge at the seat is the wall cleat's top edge
            self.assertAlmostEqual(L.TOP * 0 + L.SEAT + L.u_bevel_z(lx), L.bevel_z(L.SH_SEAT_X + lx), places=9)
        self.assertAlmostEqual(L.SH_SEAT_X + L.UC_BACK_LOCAL, L.WALL_X)
        self.assertAlmostEqual(L.PATCH, 0.0074, places=3)

    def test_wall_cleat_leaves_the_shelf_back_clear(self):
        self.assertLess(L.WALL_X - L.CLEAT_T, L.WALL_X - L.WCLEAT_T)  # the unit cleat's front face is the more forward
        back_panel = L.SH_SEAT_X + L.SH_D / 2
        self.assertAlmostEqual((L.WALL_X - L.WCLEAT_T) - back_panel, 0.001)  # 1 mm behind the back panel

    def _wall_points(self, n=400):
        rng = np.random.default_rng(3)
        pts = []
        for spec in (T.CLEAT, T.WALL):
            for s in spec.shapes:
                if s.kind != "box":
                    continue
                c = np.asarray(s.center) + np.asarray(spec.pos)
                h = np.asarray(s.size) / 2
                pts.append(c + (rng.uniform(-h, h, (n, 3))) @ s.rotation().T)
        return np.vstack(pts)

    def test_seated_shelf_touches_the_cleat_without_overlap(self):
        """No point inside the wall or its cleat is inside the seated shelf; the bevels touch; 0.5 mm shifts do."""
        pts = self._wall_points()
        d = shelf().distance(pts)
        self.assertEqual(int((d == 0).sum()), 0)
        self.assertLess(float(d.min()), 0.0001)
        into_wall = shelf(L.SH_SEAT_X + 0.002).distance(pts)
        self.assertGreater(int((into_wall == 0).sum()), 0)
        down = shelf(z=L.SEAT - 0.002).distance(pts)
        self.assertGreater(int((down == 0).sum()), 0)
        up = shelf(z=L.SEAT + 0.002).distance(pts)
        self.assertEqual(int((up == 0).sum()), 0)

    def test_the_shelf_clears_the_cleat_when_carried_over_it(self):
        """The unit cleat's lowest point is 10 mm over the shelf's underside; the wall cleat's highest is 59 mm up:
        lifted 49 mm or more the shelf passes over it, the demo lifts about 70 mm."""
        pts = self._wall_points(200)
        for z in (0.062, 0.070):
            for x in list(np.arange(0.22, L.SH_SEAT_X, 0.004)) + [L.SH_SEAT_X]:
                self.assertEqual(int((shelf(x, 0.0, z).distance(pts) == 0).sum()), 0, (z, x))
        self.assertGreater(0.070 + 0.010 - L.bevel_z(L.W_FACE_X0), 0.015)

    def test_not_pressed_back_the_shelf_comes_down_in_front(self):
        """Carried 14 mm short of the wall the unit cleat is in front of the wall cleat all the way down."""
        pts = self._wall_points(200)
        for z in np.arange(L.SEAT - 0.004, 0.07, 0.004):
            self.assertEqual(int((shelf(L.SH_SEAT_X - 0.014, 0.0, z).distance(pts) == 0).sum()), 0, z)

    def test_shapes_start_at_rest(self):
        lo, hi = T.SHELF_PART.local_bounds()
        self.assertAlmostEqual(float(lo[2]), 0.0)
        self.assertAlmostEqual(T.SHELF_PART.rest_state().low[2], L.TOP)
        self.assertAlmostEqual(float(hi[0]), L.UC_BACK_LOCAL)

    def test_slab_lies_on_the_bevel_line(self):
        """Every sample just under the unit cleat's bevel face is outside the shelf; just over it, inside."""
        st = shelf(L.SH_START_X, 0.0, 0.0)
        for lx in np.arange(0.0515, 0.0595, 0.001):
            under = st.to_world(np.array([[lx, 0.0, L.u_bevel_z(lx) - 0.0003]]))
            over = st.to_world(np.array([[lx, 0.0, L.u_bevel_z(lx) + 0.0006]]))
            self.assertGreater(float(st.distance(under)[0]), 0.0, lx)
            self.assertEqual(float(st.distance(over)[0]), 0.0, lx)


class Tiers(unittest.TestCase):
    def test_seated(self):
        r = run(shelf())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        d = r.details
        self.assertTrue(d["contact_a"] and d["contact_b"])
        self.assertAlmostEqual(d["engaged_mm"], 300.0, places=1)
        self.assertAlmostEqual(d["under_a_mm"], 40.0, places=1)

    def test_start_scores_zero(self):
        self.assertEqual(score(shelf(L.SH_START_X, 0.0, 0.0)), 0.0)
        self.assertFalse(run(shelf(L.SH_START_X, 0.0, 0.0)).success)

    def test_tier_table(self):
        cases = [
            (shelf(), 1.0),
            (shelf(y=0.015), 1.0),  # 15 mm along the wall: the whole unit cleat still on the wall cleat
            (shelf(y=0.040), 0.5),  # 40 mm along: 280 mm engaged
            (shelf(y=-0.100), 0.5),  # 200 mm engaged
            (shelf(y=0.200), 0.25),  # 120 mm engaged, under half
            (along_slope(2.0), 0.5),  # drawn 2 mm off the wall: past the 1.5 mm
            (along_slope(1.0), 1.0),
            (along_slope(4.0), 0.5),
            (crooked(-0.15, 4.0), 0.25),  # one end on the cleat, the other 4 degrees up
            (crooked(0.15, -4.0), 0.25),
            (shelf(L.SH_START_X, 0.0, 0.0), 0.0),
        ]
        for s, want in cases:
            self.assertEqual(score(s), want, (s.origin, want))

    def test_window_edges(self):
        self.assertEqual(score(along_slope(1.4)), 1.0)
        self.assertEqual(score(along_slope(1.7)), 0.5)
        self.assertEqual(score(shelf(y=0.0200)), 1.0)  # unit cleat ends exactly on the wall cleat's ends
        self.assertEqual(score(shelf(y=0.0210)), 0.5)
        self.assertEqual(score(shelf(y=0.150 + 0.170 - 0.010)), 0.25)  # 10 mm engaged

    def test_levelness_and_plumb_limits(self):
        """Both ends in contact cannot be tilted much; the angles are still read from the pose."""
        r = S.read(states(shelf(roll=0.5 * DEG)))
        self.assertAlmostEqual(S.level_deg(r), 0.5, places=3)
        r = S.read(states(shelf(pitch=-0.7 * DEG)))
        self.assertAlmostEqual(abs(S.plumb_deg(r)), 0.7, places=3)
        pose = S.judge_pose(S.read(states(shelf(pitch=2.0 * DEG))))  # leaning 2 degrees: the faces do not coincide
        self.assertFalse(pose["contact_a"] and pose["contact_b"])
        base = S.judge_pose(S.read(states(shelf())))
        for level, plumb, want in ((0.9, 0.0, 1.0), (1.1, 0.0, 0.5), (0.0, 1.1, 0.5), (2.9, 0.0, 0.5), (3.1, 0.0, 0.25),
                                   (0.0, 3.1, 0.25)):
            j = dict(base, level=level, plumb=plumb)
            self.assertEqual(S.tier(j, True), want, (level, plumb))

    def test_back_face_limits(self):
        base = S.judge_pose(S.read(states(shelf())))
        for back, want in ((0.0, 1.0), (-0.0014, 1.0), (-0.0016, 0.5), (0.0011, 0.5), (-0.0079, 0.5), (-0.0081, 0.25),
                           (0.0031, 0.25)):
            a, b = dict(base["a"], back=back), dict(base["b"], back=back)
            self.assertEqual(S.tier(dict(base, a=a, b=b), True), want, back)

    def test_underside_limit(self):
        base = S.judge_pose(S.read(states(shelf())))
        for under, want in ((0.0405, 1.0), (0.0421, 0.5), (0.0381, 1.0), (0.0379, 0.5)):
            a, b = dict(base["a"], under=under), dict(base["b"], under=under)
            self.assertEqual(S.tier(dict(base, a=a, b=b), True), want, under)

    def test_one_ended_support(self):
        """Contact at one end of the engaged length only is 0.25, whichever end it is."""
        base = S.judge_pose(S.read(states(shelf())))
        for ca, cb in ((True, False), (False, True)):
            self.assertEqual(S.tier(dict(base, contact_a=ca, contact_b=cb), True), 0.25)
        self.assertEqual(S.tier(dict(base, contact_a=False, contact_b=False), True), 0.0)
        self.assertEqual(S.tier(base, False), 0.0)
        pose = S.judge_pose(S.read(states(crooked(-0.15, 4.0))))
        self.assertTrue(pose["contact_a"] != pose["contact_b"])

    def test_contact_needs_flat_faces_not_an_edge(self):
        """The shelf resting on the cleat's front edge, its cleat's lower back corner on the wall cleat's front top
        corner, has bevel probes far from the plane: no contact, 0."""
        d = 0.0074
        for dx in (L.PATCH + 0.001, 0.010, 0.012, 0.020):
            self.assertEqual(score(shelf(L.SH_SEAT_X - dx, 0.0, L.SEAT + dx)), 0.0, dx)
        self.assertEqual(score(shelf(L.SH_SEAT_X - 0.012, 0.0, L.SEAT + 0.001)), 0.0)  # tipped onto the corner
        self.assertEqual(score(shelf(L.SH_SEAT_X - d - 0.001, 0.0, L.SEAT + 0.012, pitch=-8 * DEG)), 0.0)

    def test_slope_overlap_threshold(self):
        """The faces must overlap by 4 mm along the slope: sliding the shelf up the bevel loses contact at about
        4.4 mm."""
        self.assertEqual(score(along_slope(4.2)), 0.5)
        self.assertEqual(score(along_slope(4.8)), 0.0)

    def test_wrong_ways(self):
        self.assertEqual(score(shelf(L.SH_SEAT_X - 0.014, 0.0, 0.0)), 0.0)  # lowered in front of the cleat
        self.assertEqual(score(shelf(L.SH_SEAT_X - 0.014, 0.0, L.SEAT)), 0.0)  # at the seat height, 14 mm out
        self.assertEqual(score(shelf(L.SH_SEAT_X, 0.0, L.SEAT + 0.030)), 0.0)  # hovering 30 mm above the seat
        self.assertEqual(score(shelf(L.SH_SEAT_X, 0.0, 0.0)), 0.0)  # on the table against the wall

    def test_upside_down_and_backwards(self):
        flipped = shelf(L.SH_SEAT_X, 0.0, L.SEAT + 0.080, roll=math.pi)
        self.assertEqual(score(flipped), 0.0)
        back = shelf(yaw=math.pi)
        self.assertEqual(score(back), 0.0)
        self.assertEqual(score(shelf(yaw=math.pi / 2)), 0.0)  # turned across the wall
        self.assertEqual(score(shelf(pitch=math.pi / 2)), 0.0)  # standing on its back edge

    def test_yawed_shelf(self):
        for deg in (1.0, 2.0, 5.0, 20.0):
            self.assertLess(score(shelf(yaw=deg * DEG)), 0.5, deg)
        self.assertEqual(score(shelf(yaw=0.2 * DEG)), 1.0)

    def test_no_bevel_contact_means_no_credit_whatever_the_pose(self):
        """A shelf at the right height and distance but standing off the wall by 6 mm along the horizontal."""
        self.assertEqual(score(shelf(L.SH_SEAT_X - 0.006, 0.0, L.SEAT)), 0.0)

    def test_exploit_shelf_wedged_on_the_casing(self):
        """Pushed along the wall until its end meets the door casing: the casing stops it 110 mm off centre, still 210
        mm of the unit cleat on the wall cleat (0.5, not 1.0); on the table there against the wall: 0."""
        y = L.SIDE_Y0 - L.SH_W / 2
        self.assertEqual(score(shelf(L.SH_SEAT_X, y, 0.0)), 0.0)
        self.assertEqual(score(shelf(L.SH_SEAT_X, y, L.SEAT)), 0.5)
        self.assertAlmostEqual(S.cover(S.read(states(shelf(L.SH_SEAT_X, y, L.SEAT))))[2], 0.210, places=3)

    def test_exploit_shelf_held_up_in_place(self):
        """A perfect pose with a gripper on the shelf scores 0: the score is read hands off."""
        pinch = (L.SH_SEAT_X + L.GRIP_LOCAL_X, L.END_MID, L.TOP + L.SEAT + 0.073)
        hand = {"left": arm(pinch, 0.1), "right": FAR["right"]}
        self.assertEqual(score(shelf(), hand), 0.0)

    def test_exploit_hand_held_just_off_the_shelf(self):
        for gap in (0.020, 0.028):
            y = L.SH_W / 2 + rigstate.zone_radius(1.0) + gap
            hand = {"left": arm((L.SH_SEAT_X, y, L.TOP + L.SEAT + 0.05), 1.0), "right": FAR["right"]}
            self.assertEqual(score(shelf(), hand), 0.0, gap)
        y = L.SH_W / 2 + rigstate.zone_radius(1.0) + 0.034
        hand = {"left": arm((L.SH_SEAT_X, y, L.TOP + L.SEAT + 0.05), 1.0), "right": FAR["right"]}
        self.assertEqual(score(shelf(), hand), 1.0)

    def test_exploit_pressing_down_from_above(self):
        """A closed or open hand resting on the shelf's top to keep it seated: not released."""
        for opening in (0.0, 1.0):
            hand = {"left": arm((L.SH_SEAT_X, 0.0, L.TOP + L.SEAT + L.SH_H + 0.020), opening), "right": FAR["right"]}
            self.assertEqual(score(shelf(), hand), 0.0)

    def test_exploit_shelf_pushed_through_the_wall_does_not_count(self):
        """Inside the wall by 3 mm (a solver blow-up): the back face is 3 mm into it: not 1.0."""
        self.assertLess(score(shelf(L.SH_SEAT_X + 0.003, 0.0, L.SEAT)), 1.0)

    def test_hands_off_hands_on_and_unreadable(self):
        self.assertEqual(score(shelf(), arms=None), 0.0)
        self.assertEqual(score(shelf(), arms={"left": FAR["left"]}), 0.0)
        bad = {"left": arm((float("nan"), 0.4, 0.8)), "right": FAR["right"]}
        self.assertEqual(score(shelf(), bad), 0.0)
        on = {"left": arm((L.SH_SEAT_X, 0.17, L.TOP + 0.09), 0.1), "right": FAR["right"]}
        self.assertEqual(score(shelf(), on), 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        hand = {"left": arm((L.SH_SEAT_X, 0.17, L.TOP + 0.09), 0.1), "right": FAR["right"]}
        for _ in range(WIN + 2):
            tr.update(states(shelf()), hand)
        self.assertEqual(tr.last.score, 0.0)
        for i in range(WIN - 1):
            tr.update(states(shelf()), FAR)
            self.assertEqual(tr.last.score, 0.0, i)
        tr.update(states(shelf()), FAR)
        tr.update(states(shelf()), FAR)
        self.assertEqual(tr.last.score, 1.0)

    def test_moving_shelf(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(shelf(y=0.0002 * i)), FAR)  # 0.2 mm a step: over 1 s that is 4 mm
        self.assertEqual(tr.last.score, 0.0)
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(shelf(roll=0.02 * DEG * i)), FAR)  # 0.4 degrees over the second
        self.assertEqual(tr.last.score, 0.0)

    def test_still_just_under_the_limits(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(states(shelf(y=0.0004 * min(i, WIN) / WIN)), FAR)  # 0.4 mm over the second
        self.assertEqual(tr.last.score, 1.0)

    def test_missing_and_nan_state(self):
        self.assertEqual(S.Tracker().update({}, FAR).score, 0.0)
        self.assertEqual(S.Tracker().update(None, FAR).score, 0.0)
        bad = mk.state_from_poses(T.SHELF_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(states(bad), FAR).score, 0.0)
        inf = mk.state_from_poses(T.SHELF_PART, (L.SH_SEAT_X, float("inf"), L.TOP, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(states(inf), FAR).score, 0.0)
        tr = S.Tracker()
        run(shelf(), tracker=tr)
        self.assertEqual(tr.last.score, 1.0)
        tr.update(states(bad), FAR)  # one bad step: 0, and the second starts again
        self.assertEqual(tr.last.score, 0.0)
        for i in range(WIN):
            tr.update(states(shelf()), FAR)
            self.assertEqual(tr.last.score, 0.0, i)
        tr.update(states(shelf()), FAR)
        self.assertEqual(tr.last.score, 1.0)
        self.assertEqual(tr.bad_steps, 1)

    def test_no_distance_means_not_released(self):
        class NoDistance:
            origin = shelf().origin
            quat_xyzw = shelf().quat_xyzw

        tr = S.Tracker()
        for _ in range(WIN + 3):
            r = tr.update({S.SHELF: NoDistance()}, FAR)
        self.assertEqual(r.score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(shelf(), tracker=tr)
        self.assertEqual(tr.last.score, 1.0)
        tr.update(states(shelf(L.SH_START_X, 0.0, 0.0)), FAR)
        self.assertEqual(tr.resets, 1)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(tr.last.score, 0.0)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(400):
            s = shelf(L.SH_SEAT_X + rng.uniform(-0.03, 0.01), rng.uniform(-0.2, 0.2), rng.uniform(-0.01, 0.1),
                      yaw=rng.uniform(-0.3, 0.3), pitch=rng.uniform(-0.3, 0.3), roll=rng.uniform(-0.3, 0.3))
            r = tr.update(states(s), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))
            self.assertIsInstance(json.dumps(r.details), str)

    def test_every_tier_is_reachable_and_fine_poses_near_the_seat_never_exceed_the_tiers(self):
        seen = set()
        rng = np.random.default_rng(7)
        for _ in range(600):
            d = rng.uniform(0.0, 0.008)
            s = shelf(L.SH_SEAT_X - d + rng.normal(0, 0.0003), rng.uniform(-0.05, 0.05),
                      L.SEAT + d + rng.normal(0, 0.0005), yaw=rng.normal(0, 0.005), roll=rng.normal(0, 0.005))
            seen.add(score(s))
        self.assertTrue({0.0, 0.5, 1.0} <= seen, seen)


class Demo(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["wall", "wall_cleat", "battens", "casing", "shelf"])
        for o in T.OBJECTS[:-1]:
            self.assertIsNone(o.mass)
        self.assertAlmostEqual(T.SHELF_PART.mass, L.SH_MASS)
        self.assertEqual(T.SHELF_PART.pos[0], L.SH_START_X)

    def test_start_pose_rests_on_the_table_clear_of_the_wall(self):
        st = T.SHELF_PART.rest_state()
        self.assertAlmostEqual(st.low[2], L.TOP)
        self.assertLess(st.high[0], L.WALL_X - 0.10)
        self.assertGreater(T.SHELF_PART.mass, 0.0)
        self.assertGreater(abs(st.low[1]), 0.0)
        self.assertGreater(0.45, st.high[1])  # clear of the arms' start pose (|y| > 0.45 m)

    def test_instruction_states_the_goal_only(self):
        self.assertLess(len(T.INSTRUCTION), 300)
        low = T.INSTRUCTION.lower()
        for word in ("lift", "press", "lower", "mm", "step", "tilt", "bevel", "45", "two hands"):
            self.assertNotIn(word, low)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            self.assertEqual(set(plan), {"left", "right"})
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.1505, (v, a, w.label))
                n_grips = sum(1 for x, y in zip(wps, wps[1:]) if abs(x.finger - y.finger) > 0.005)
                self.assertEqual(n_grips, 2)  # close once, open once
            longest = max(sum(w.duration for w in wps) for wps in plan.values())
            self.assertLessEqual(longest * HZ + 40, run_limit / 2)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "skew", {"left", "right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def test_demo_timing_and_half_the_limit(self):
        plan = T.plans("")
        total = max(sum(w.duration for w in wps) for wps in plan.values())
        self.assertAlmostEqual(total, 18.2, places=3)
        released_at = total + 1.0 + 1.0  # hands clear, one second still, one second of success hold
        self.assertLess(released_at * HZ, 450)  # within half of the 900-step limit

    def test_grip_geometry(self):
        self.assertAlmostEqual(-2 * T.GRIP, L.END_T - 0.008)
        self.assertAlmostEqual(-T.OPEN - L.END_T / 2, 0.038, delta=0.0001)  # 38 mm clear round the panel each side
        self.assertGreaterEqual(L.END_T, 0.016)
        # the pads' 27 mm width sits inside the 100 mm panel; pinch 20 mm below the top edge
        self.assertAlmostEqual(T.GRIP_TCP - T.PAD_TIP, L.SH_H - T.TIPS_DOWN)
        self.assertLess(abs(L.GRIP_LOCAL_X) + rigstate.FINGER_HALF_WIDTH, L.SH_D / 2)
        for w in T.plans("")["left"]:
            self.assertAlmostEqual(abs(w.pos[1]), L.END_MID) if w.label != "back off" else None

    def test_pinch_torque_is_within_the_pads(self):
        """The fingers pinch 32 mm in front of the centre of mass: each hand's share of the weight times that lever, a
        third of what two 22 x 27 mm pads at 20 N and a friction of 0.8 hold (about 0.24 N m)."""
        com = float(T.SHELF_PART.total_center_of_mass()[0])
        lever = com - L.GRIP_LOCAL_X
        self.assertAlmostEqual(lever, 0.032, delta=0.003)
        torque = L.SH_MASS * L.G / 2 * lever
        self.assertLess(torque, 0.24 / 2)

    def test_lift_and_clearances(self):
        lift_real = T.LIFT - 0.005  # the arms trail about 5 mm under their 2.6 N share (GPU shim task: 2 mm per N)
        self.assertGreater(lift_real + 0.010 - L.bevel_z(L.W_FACE_X0), 0.015)  # unit cleat 15 mm over the wall cleat
        self.assertGreater(L.SEAT + 0.0, 0.012 * 0 + 0.030)
        self.assertGreater(lift_real, L.SEAT)  # lifted above the seat height: lowered 25 mm or more to seat
        self.assertLess(T.GRIP_TCP + T.LIFT + L.SEAT * 0, 0.15)  # inside the arms' reach with a load
        # the hands, closed on the panels, are 302 mm apart, the housing 28 mm short of the wall at the seat
        self.assertGreater(L.WALL_X - (T.end_x(T.PRESS_X) + 0.040), 0.025)
        # the casing: 110 mm beyond the shelf's end, 28 mm beyond the left gripper's carriage
        self.assertAlmostEqual(L.SIDE_Y0 - L.SH_W / 2, 0.110)

    def test_press_and_lower_commands(self):
        self.assertAlmostEqual(T.SEAT_TCP - T.GRIP_TCP, L.SEAT)
        left = T.plans("")["left"]
        labels = [w.label for w in left]
        self.assertEqual(labels.index("lift the shelf clear of the cleat") + 1, labels.index(
            "carry it back until its cleat meets the wall"))
        carry = left[labels.index("carry it back until its cleat meets the wall")]
        self.assertAlmostEqual(carry.pos[0], L.WALL_X - L.UC_BACK_LOCAL + L.GRIP_LOCAL_X + T.PRESS_X)
        lower = left[labels.index("lower it onto the cleat")]
        self.assertAlmostEqual(lower.pos[2], L.TOP + T.GRIP_TCP + L.SEAT - T.PRESS_Z)
        # the shelf is carried at constant height, then lowered with the hands at the same x and y
        self.assertAlmostEqual(lower.pos[0], carry.pos[0])
        self.assertAlmostEqual(lower.pos[1], carry.pos[1])

    def test_counterfactuals_differ_in_the_intended_way(self):
        base = T.plans("")
        short = T.plans("short")
        i = [w.label for w in base["left"]].index("carry it back until its cleat meets the wall")
        self.assertAlmostEqual(short["left"][i].pos[0] - base["left"][i].pos[0], -0.014)
        skew = T.plans("skew")
        self.assertAlmostEqual(skew["right"][i].pos[0] - base["right"][i].pos[0], -0.012)
        self.assertAlmostEqual(skew["left"][i].pos[0], base["left"][i].pos[0])
        shifted = T.plans("shifted")
        self.assertAlmostEqual(shifted["left"][i].pos[1] - base["left"][i].pos[1], 0.060)
        self.assertAlmostEqual(shifted["right"][i].pos[1] - base["right"][i].pos[1], 0.060)


def arms_at(left, right, lo=0.1, ro=0.1):
    return {"left": arm(left, lo), "right": arm(right, ro)}


class OfflineModel(unittest.TestCase):
    def _play(self, variant="", idle=None):
        """Drive the offline model with the demo's own waypoints (straight lines between them)."""
        plan = T.plans(variant)
        if idle:
            plan.pop(idle)
        m = T.OfflineModel()
        n = max(len(w) for w in plan.values())
        pos = {"left": np.array([0.32, 0.305, L.TOP + 0.26]), "right": np.array([0.32, -0.305, L.TOP + 0.26])}
        fing = {"left": T.OPEN, "right": T.OPEN}
        t = {a: 0.0 for a in plan}
        out = []
        for k in range(n):
            steps = {a: (int(round(plan[a][k].duration * HZ)) if k < len(plan[a]) else 0) for a in plan}
            for i in range(max(steps.values())):
                snap = {}
                for a in ("left", "right"):
                    if a in plan and k < len(plan[a]) and i < steps[a]:
                        w = plan[a][k]
                        f = (i + 1) / steps[a]
                        pos[a] = pos[a] + (np.asarray(w.pos) - pos[a]) / max(1, steps[a] - i)
                        fing[a] = fing[a] + (w.finger - fing[a]) / max(1, steps[a] - i)
                    snap[a] = arm(pos[a], float(np.clip(fing[a] / T.OPEN, 0.0, 1.0)))
                m.step(snap)
                out.append((a, m.o.copy()))
        return m

    def test_demo_seats_the_shelf(self):
        m = self._play()
        self.assertAlmostEqual(float(m.o[0]), L.SH_SEAT_X, places=6)
        self.assertAlmostEqual(float(m.o[2]), L.SEAT, places=6)
        self.assertAlmostEqual(float(m.o[1]), 0.0, places=4)
        r = S.read(m.states())
        self.assertEqual(S.tier(S.judge_pose(r), True), 1.0)

    def test_one_arm_never_seats(self):
        for idle in ("left", "right"):
            m = self._play(idle=idle)
            r = S.read(m.states())
            self.assertEqual(S.tier(S.judge_pose(r), True), 0.0, idle)
            self.assertLess(float(m.o[2]), 0.001)  # it ends on the table

    def test_variants(self):
        want = {"short": 0.0, "skew": 0.25, "shifted": 0.5}
        for v, s in want.items():
            m = self._play(v)
            self.assertEqual(S.tier(S.judge_pose(S.read(m.states())), True), s, v)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()[S.SHELF]
        self.assertTrue(np.allclose(st.origin, T.SHELF_PART.rest_state().origin, atol=1e-6))

    def test_hold_only_stays_put(self):
        m = T.OfflineModel()
        for _ in range(40):
            m.step(FAR)
        self.assertEqual(S.tier(S.judge_pose(S.read(m.states())), True), 0.0)

    def test_a_hand_that_just_passes_does_not_grab(self):
        m = T.OfflineModel()
        for x in np.linspace(0.1, 0.3, 30):
            m.step(arms_at((x, L.END_MID, L.TOP + 0.20), (x, -L.END_MID, L.TOP + 0.20), 0.1, 0.1))
        self.assertEqual(m.grip, {})
        self.assertAlmostEqual(float(m.o[0]), L.SH_START_X)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        for token in ("x 400 mm", "700 mm", "210 mm", "x 389 to 400 mm", "340 mm long", "300 mm long", "320 mm wide",
                      "100 mm deep", "80 mm tall", "18 mm", "6 mm", "0.53 kg", "12 mm", "11 mm", "45 degrees",
                      "x 170 to 270 mm", "40 mm over the table", "y 270 to 300 mm", "Three", "1.5 mm proud", "51 mm",
                      "z 61 mm", "z 50 mm", "22 mm", "70 mm"):
            self.assertIn(token, setup)
        for token in ("1 mm", "4 mm", "300 mm", "150 mm", "1.5 mm", "8 mm", "1 degree", "3 degrees", "42 mm", "30 mm",
                      "0.3 degrees", "Possible scores: 0, 0.25, 0.5, 1.0", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "900 steps (45 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("Rigid", "est", "0.45", "Screws", "680", "eased"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("French cleat", "45 degrees"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)
        self.assertLess(len(d["instruction"]), 300)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "rigstate.py", "mechkit.py",
                  "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("—|–|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f.endswith(".py") else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline_and_has_the_sections(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)
        for head in ("## Why it is hard", "## Scoring", "## Results", "## Demo", "## Simplifications", "## Sources",
                     "## Real build and reset", "## Files", "Third-party assets"):
            self.assertIn(head, text)
        for token in ("inspect-robots run", "Runtime", "Seed", "Bench check"):
            self.assertIn(token, text)

    def test_no_pycache_left(self):
        self.assertFalse(list(HERE.rglob("__pycache__")))


if __name__ == "__main__":
    unittest.main()
