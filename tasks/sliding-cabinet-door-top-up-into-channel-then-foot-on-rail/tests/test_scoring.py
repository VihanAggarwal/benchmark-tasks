"""Offline tests for layout.py, geom.py, scoring.py and task.py: synthetic door states (mechkit AssemblyStates from the
task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
    import geom as G  # noqa: E402
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
D = math.radians
Z_REST = L.FLOOR_TOP  # an upright door's origin height when it stands on the floor


def lowest_foot(lean_deg):
    q = mk.quat_from_ypr(0.0, D(lean_deg), 0.0)
    r = S.rotation(q)
    return float(np.min((S.FOOT_PTS @ r.T)[:, 2]))


def front_at(bx=L.X_F, y=L.FRONT_HUNG_Y, z=None, lean=0.0, yaw=0.0, roll=0.0):
    """The front door with its underside centre at x ``bx`` leaning ``lean`` degrees (top toward +x); ``z`` its
    origin's height over the table top (default: the lowest corner of its underside on the floor)."""
    if z is None:
        z = L.FLOOR_TOP - lowest_foot(lean)
    q = mk.quat_from_ypr(D(yaw), D(lean), D(roll))
    return mk.state_from_poses(T.FRONT_PART, (bx, y, L.TOP + z, *q), source="test")


def hung(**kw):
    return front_at(**kw)


def leaning(lean, **kw):
    """The door standing on the floor with its top centre on the front track's centre line, leaning ``lean`` deg."""
    bx = L.X_F - L.DOOR_H * math.sin(D(lean))
    return front_at(bx=bx, lean=lean, **kw)


def rear_at(dx=0.0, dy=0.0, dz=0.0, tilt=0.0):
    q = mk.quat_from_ypr(0.0, D(tilt), 0.0)
    return mk.state_from_poses(T.REAR_PART, (L.REAR_X + dx, L.REAR_Y + dy, L.TOP + L.FLOOR_TOP + dz, *q),
                               source="test")


def scene(front=None, rear=None):
    return {S.FRONT: front if front is not None else hung(), S.REAR: rear if rear is not None else rear_at()}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, approach=(1.0, 0.0, 0.0), opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp + 0.013 * np.asarray(approach, dtype=float), opening)


# the left hand backed off 150 mm from the door, the right at its ready pose (near the rear door, on purpose)
FAR = {"left": arm((L.X_F - 0.20, 0.14, L.TOP + 0.25)),
       "right": arm((0.323, -0.305, L.TOP + 0.26), (0.83, 0.0, -0.56))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def score_of(front=None, rear=None, arms=FAR):
    return run(scene(front, rear), arms=arms).score


class Geometry(unittest.TestCase):
    def test_sizes(self):
        self.assertAlmostEqual(L.CLEAR_H, 0.270)
        self.assertAlmostEqual(L.DOOR_H - L.CLEAR_H, 0.020)  # taller than the opening: it cannot go in upright
        self.assertAlmostEqual(L.GROOVE_W - L.DOOR_T, 0.016)  # 8 mm each side at the top
        self.assertAlmostEqual((L.FOOT_GROOVE_W - L.RAIL_W) / 2, 0.004)
        self.assertAlmostEqual(L.RAIL_TOP, 0.046)
        self.assertAlmostEqual(L.ROOF, 0.350)
        self.assertAlmostEqual(L.TOP_HUNG, 0.330)
        self.assertAlmostEqual(L.TOP_HUNG - L.TRACK_BOTTOM, 0.020)  # 20 mm up in the groove when hung
        self.assertAlmostEqual(L.FOOT_RAISED - L.RAIL_TOP, 0.014)  # raised to the roof: the foot is 14 mm over the rail
        self.assertAlmostEqual(L.FRONT_GROOVE[0], 0.368)
        self.assertAlmostEqual(L.REAR_GROOVE[1], 0.452)
        self.assertAlmostEqual(L.TRACK_X0, 0.360)
        self.assertAlmostEqual(L.TRACK_X1, 0.460)
        self.assertAlmostEqual(L.X_R - L.X_F, 0.046)
        self.assertAlmostEqual(L.PULL_X, -0.051)
        self.assertAlmostEqual(L.FRONT_START_X, 0.337)

    def test_every_dimension_the_task_text_gives(self):
        d = json.loads((HERE / "task.json").read_text())
        for token in ("x 360 mm", "x 678 mm", "620 mm wide", "z 350 mm", "x 360 to 368, 406 to 414 and 452 to 460",
                      "38 mm wide and 40 mm deep", "x 387 mm", "x 433 mm", "270 mm", "6 mm wide and 6 mm high",
                      "z 46 mm", "y -300 to +20 mm", "22 x 320 x 290 mm", "0.29 kg", "14 mm wide and 8 mm deep",
                      "x 337 mm", "y from -140 to +180 mm", "20 x 20 mm", "140 mm long", "170 mm above the foot",
                      "51 mm in front", "12 x 12 mm", "30 mm", "y -20 to +300 mm", "60 mm above the table top",
                      "14 mm over the rail tops", "20 mm inside the groove"):
            self.assertIn(token, d["setup"], token)

    def test_doors_spawn_where_they_rest(self):
        s = start()
        for n in S.PARTS:
            self.assertTrue(np.all(np.isfinite(s[n].origin)))
        r = S.read(s)
        self.assertAlmostEqual(float(r.pos[S.FRONT][2]), 0.0)  # on the table
        self.assertAlmostEqual(float(r.pos[S.REAR][2]), L.FLOOR_TOP)  # on the floor, rail in its groove
        self.assertTrue(S.rear_kept(r))
        self.assertFalse(S.top_in_groove(r))
        self.assertTrue(S.top_in_groove(r, S.REAR) is False)  # the rear groove is not the front one

    def test_front_door_clear_of_the_arms_at_ready(self):
        # the left hand at the ready pose reaches in to y 190 mm at x 323 to 334: the door spans y -140 to +180
        self.assertLess(L.FRONT_START_Y + L.DOOR_W / 2, 0.19)

    def test_door_cannot_go_in_upright(self):
        # upright, foot 1 mm over the floor's top: the top hits the track's front wall before the foot is over the floor
        hits = [G.clearance((x, L.FLOOR_TOP + 0.001), 0.0) for x in np.arange(0.30, 0.40, 0.002)]
        self.assertLess(min(hits), 0.0)
        # upright and low enough to pass under the track, the foot hits the floor's edge
        self.assertLess(G.clearance((L.CAB_X0 - 0.008, 0.030), 0.0), 0.0)

    def test_leaned_door_goes_in_and_clears_the_walls(self):
        b = G.top_for((L.X_F, L.TRACK_BOTTOM - 0.007), D(10.0))
        self.assertGreater(G.clearance(b, D(10.0)), 0.004)
        for tz, lean in ((0.325, 10.0), (0.338, 7.0), (0.346, 3.5)):
            b = G.top_for((L.X_F, tz), D(lean))
            self.assertGreater(G.clearance(b, D(lean)), 0.003, (tz, lean))

    def test_hung_pose_is_in_contact_only_with_the_floor(self):
        self.assertAlmostEqual(G.clearance((L.X_F, L.FLOOR_TOP), 0.0), 0.0, places=6)
        self.assertGreater(G.clearance((L.X_F, L.FLOOR_TOP + 0.002), 0.0), 0.0015)

    def test_foot_passes_over_the_rail_when_raised(self):
        self.assertGreater(G.clearance((L.X_F, L.FOOT_RAISED), 0.0), -1e-9)
        for dx in (-0.004, 0.0, 0.004):
            self.assertGreaterEqual(G.clearance((L.X_F + dx, L.FOOT_RAISED), 0.0), -1e-9)
        # lowered 4 mm off the rail with the foot 8 mm out of line: a strip lands on the rail's top
        self.assertLess(G.clearance((L.X_F + 0.008, L.FLOOR_TOP + 0.001), 0.0), 0.0)

    def test_pull_pose_maths(self):
        for lean in (0.0, 4.0, 10.0):
            b = np.array([0.31, 0.02])
            tcp = G.tcp_for(b, D(lean))
            self.assertTrue(np.allclose(G.pose_for_tcp(tcp, D(lean)), b))
            a = np.array([math.cos(D(lean)), -math.sin(D(lean))])
            self.assertAlmostEqual(G.lean_from_approach(a), D(lean))
            self.assertAlmostEqual(G.pitch_for(D(lean)), math.pi / 2 - D(lean))

    def test_arm_force_and_grip(self):
        w = L.DOOR_MASS * L.G
        self.assertLess(w, 5.0)  # the 5 N the arm gives
        lever = -L.PULL_X + 0.0  # the door's centre plane is 51 mm behind the bar
        torque = w * (lever + 0.0)
        pad = 0.022  # a pad's length along the approach axis
        n = 5000.0 * ((L.PULL_SIZE[2] - (-2 * T.GRIP)) / 2)  # N per finger, the finger spring
        self.assertGreater(n * pad / 2, 1.5 * torque)  # the pads' couple holds the door's turning weight twice over
        self.assertLessEqual(n, 50.0)
        self.assertGreaterEqual(L.PULL_SIZE[2], 0.016)  # the narrowest grip width


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(scene())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertTrue(r.details["top_in_groove"] and r.details["foot_down"])

    def test_start_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_lean_between_3_and_8_is_half(self):
        for lean in (3.5, 5.0, 7.5):
            self.assertEqual(score_of(leaning(lean)), 0.5, lean)

    def test_lean_under_3_is_full(self):
        # with the rail in the foot groove the top's 8 mm of play allows only about 1.6 degrees of lean before its
        # corner leaves the groove; leaning 1.5 degrees the door is hung (the 3 degree bound binds a door standing on
        # the floor beside the rail, which the rail check also stops)
        self.assertEqual(score_of(front_at(lean=1.5)), 1.0)
        self.assertEqual(score_of(front_at(bx=L.X_F - 0.004, lean=2.5)), 1.0)
        self.assertEqual(score_of(front_at(lean=2.5)), 0.0)  # top corner 2.6 mm through the groove's rear wall

    def test_foot_beside_the_rail_is_half(self):
        # the review's loophole: leaning 2.9 degrees with the foot down on the floor in front of (or behind) the rail,
        # the mid-height point within 7 mm of the line and the top in the groove: not hung, so not 1.0
        for bx, lean in ((L.X_F - 0.014, 2.9), (L.X_F - 0.0142, 2.95), (L.X_F + 0.014, -2.9)):
            f = front_at(bx=bx, lean=lean)
            r = run(scene(f))
            self.assertTrue(r.details["top_in_groove"] and r.details["foot_down"], (bx, lean))
            self.assertLessEqual(abs(r.details["face_offset_mm"]), 7.0, (bx, lean))
            self.assertFalse(r.details["rail_in_groove"], (bx, lean))
            self.assertEqual(r.score, 0.5, (bx, lean))
        # the old helper's pose: the top centre on the line, the foot 12.6 mm in front of it (a strip over the rail)
        self.assertEqual(score_of(leaning(2.5)), 0.5)

    def test_rail_in_groove_bounds(self):
        for dx, want in ((0.0, 1.0), (0.004, 1.0), (-0.004, 1.0), (0.0049, 1.0), (0.0055, 0.5), (-0.0055, 0.5),
                         (0.008, 0.5)):
            self.assertEqual(score_of(front_at(bx=L.X_F + dx)), want, dx)
        self.assertTrue(run(scene()).details["rail_in_groove"])

    def test_door_turned_round_is_quarter(self):
        # hung the wrong way round (pull inside, toward the rear door, where y lets it pass): not 0.5 or 1.0
        f = front_at(yaw=180.0, y=0.149)
        r = run(scene(f))
        self.assertTrue(r.details["top_in_groove"] and r.details["foot_down"] and r.details["rail_in_groove"])
        self.assertFalse(r.details["pull_out"])
        self.assertEqual(r.score, 0.25)
        self.assertTrue(run(scene()).details["pull_out"])

    def test_lean_over_8_with_foot_down_is_quarter_or_less(self):
        # at 9 degrees a door standing on the floor with its top in the groove cannot have its top corners both inside
        self.assertIn(score_of(leaning(9.0)), (0.0, 0.25))

    def test_face_off_is_half(self):
        # a physically impossible but readable pose: upright, foot down, the whole door 8 mm back
        self.assertEqual(score_of(front_at(bx=L.X_F + 0.008)), 0.5)

    def test_face_just_inside(self):
        # the foot 4 mm back (strip against the rail), leaning 1 degree: the centre plane at mid-height 6.5 mm off
        self.assertEqual(score_of(front_at(bx=L.X_F + 0.004, lean=1.0)), 1.0)
        # the foot 5 mm back, leaning 0.9 degrees: 7.3 mm off
        self.assertEqual(score_of(front_at(bx=L.X_F + 0.005, lean=0.9)), 0.5)

    def test_foot_up_on_the_rail_is_quarter(self):
        self.assertEqual(score_of(front_at(z=L.RAIL_TOP)), 0.25)
        self.assertEqual(score_of(front_at(z=L.RAIL_TOP + 0.003)), 0.25)

    def test_foot_at_3mm_edge(self):
        self.assertEqual(score_of(front_at(z=L.FLOOR_TOP + 0.0029)), 1.0)
        self.assertEqual(score_of(front_at(z=L.FLOOR_TOP + 0.0031)), 0.25)

    def test_top_depth_edge(self):
        # raised so the top is 5 mm up (not in) and 7 mm up (in); the foot is up too, so only the depth is read
        up = L.TRACK_BOTTOM + 0.005 - L.DOOR_H
        self.assertEqual(score_of(front_at(z=up)), 0.0)
        up = L.TRACK_BOTTOM + 0.007 - L.DOOR_H
        self.assertEqual(score_of(front_at(z=up)), 0.25)

    def test_door_held_up_to_the_roof_is_quarter(self):
        self.assertEqual(score_of(front_at(z=L.FOOT_RAISED - 0.001)), 0.25)

    def test_door_through_the_roof_is_zero(self):
        self.assertEqual(score_of(front_at(z=L.FOOT_RAISED + 0.01)), 0.0)

    def test_door_on_the_table_scores_zero(self):
        self.assertEqual(score_of(front_at(bx=0.30, z=0.0)), 0.0)

    def test_door_lying_scores_zero(self):
        f = front_at(bx=0.30, y=0.1, z=0.011, lean=90.0)
        self.assertEqual(score_of(f), 0.0)

    def test_door_on_top_of_the_cabinet_scores_zero(self):
        self.assertEqual(score_of(front_at(bx=0.50, z=L.TOP_PANEL_TOP)), 0.0)

    def test_door_leaning_on_the_front_of_the_cabinet_scores_zero(self):
        self.assertEqual(score_of(front_at(bx=0.345, z=0.0, lean=4.0)), 0.0)

    def test_door_in_the_rear_groove_scores_zero(self):
        # exploit: with the rear door out of the way, hang the front door in the REAR track
        self.assertEqual(score_of(front_at(bx=L.X_R), rear_at()), 0.0)
        self.assertEqual(score_of(front_at(bx=L.X_R, y=0.14), rear_at(dy=0.05)), 0.0)

    def test_door_yawed_in_the_track(self):
        # 1 degree about the vertical: the ends swing 2.8 mm at 160 mm, the rail still in the groove at both ends;
        # 2 degrees swings them 5.6 mm: a strip would be over the rail, so not 1.0
        self.assertEqual(score_of(front_at(yaw=1.0)), 1.0)
        self.assertEqual(score_of(front_at(yaw=2.0)), 0.5)
        self.assertIn(score_of(front_at(yaw=10.0)), (0.0, 0.25, 0.5))

    def test_door_slid_along_the_track_is_fine(self):
        self.assertEqual(score_of(front_at(y=0.0)), 1.0)
        self.assertEqual(score_of(front_at(y=0.149)), 1.0)

    def test_door_pushed_into_the_side_panel_is_zero(self):
        self.assertEqual(score_of(front_at(y=0.20)), 0.0)  # its end 50 mm through the side wall: the corners leave

    def test_door_rolled_over_scores_low(self):
        self.assertEqual(score_of(front_at(roll=30.0)), 0.0)


class RearDoor(unittest.TestCase):
    def test_kept_within_tolerance(self):
        self.assertEqual(score_of(rear=rear_at(dy=0.009)), 1.0)
        self.assertEqual(score_of(rear=rear_at(dx=0.004)), 1.0)
        self.assertEqual(score_of(rear=rear_at(tilt=3.0)), 1.0)

    def test_moved_caps_at_quarter(self):
        self.assertEqual(score_of(rear=rear_at(dy=0.012)), 0.25)
        self.assertEqual(score_of(rear=rear_at(dy=0.03)), 0.25)
        self.assertEqual(score_of(rear=rear_at(dx=0.007)), 0.25)
        self.assertEqual(score_of(rear=rear_at(tilt=4.5)), 0.25)

    def test_zero_when_knocked_out(self):
        self.assertEqual(score_of(rear=rear_at(tilt=11.0)), 0.0)
        self.assertEqual(score_of(rear=rear_at(dy=0.07)), 0.0)
        self.assertEqual(score_of(rear=rear_at(dx=0.03)), 0.0)
        self.assertEqual(score_of(rear=rear_at(dz=-0.012)), 0.0)  # fell to z 28 mm

    def test_knocked_out_then_put_back_is_still_zero(self):
        # exploit: shove the rear door out of its track to make room, then restore it: the latch stays
        tr = S.Tracker()
        tr.update(scene(rear=rear_at(tilt=12.0)), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(scene(), tracker=tr).score, 0.0)

    def test_front_hung_over_a_slid_rear_door_is_not_full(self):
        # exploit: slide the rear door 100 mm out of the way, hang the front door, leave both
        self.assertEqual(score_of(rear=rear_at(dy=0.10)), 0.0)

    def test_idle_hand_by_the_rear_door_does_not_block(self):
        near_rear = {"left": FAR["left"], "right": arm((0.35, -0.20, L.TOP + 0.16), (1.0, 0.0, 0.0))}
        self.assertEqual(score_of(arms=near_rear), 1.0)


class ReleaseAndStill(unittest.TestCase):
    def hand_on_the_pull(self, opening=0.13):
        return {"left": arm((L.X_F - 0.051, 0.14, L.TOP + L.FLOOR_TOP + L.PULL_H), (1.0, 0.0, 0.0), opening),
                "right": FAR["right"]}

    def test_hand_on_the_pull(self):
        # exploit: hold the hung door in place
        self.assertEqual(score_of(arms=self.hand_on_the_pull()), 0.0)
        self.assertEqual(score_of(arms=self.hand_on_the_pull(1.0)), 0.0)

    def test_hand_backed_off_as_the_demo_does(self):
        a = {"left": arm((L.X_F - 0.051 - T.RETREAT, 0.14, L.TOP + L.FLOOR_TOP + L.PULL_H), opening=0.64),
             "right": FAR["right"]}
        self.assertEqual(score_of(arms=a), 1.0)

    def test_hand_backed_off_only_a_little(self):
        a = {"left": arm((L.X_F - 0.051 - 0.03, 0.14, L.TOP + L.FLOOR_TOP + L.PULL_H)), "right": FAR["right"]}
        self.assertEqual(score_of(arms=a), 0.0)

    def test_hand_on_the_top_of_the_door_is_not_clear(self):
        a = {"left": arm((L.X_F - 0.02, 0.14, L.TOP + 0.36), (0.0, 0.0, -1.0)), "right": FAR["right"]}
        self.assertEqual(score_of(arms=a), 0.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        run(scene(), arms=self.hand_on_the_pull(), tracker=tr)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        self.assertEqual(run(scene(), tracker=tr, steps=WIN).score, 1.0)

    def test_front_door_moving_five_mm_in_a_second(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(front=front_at(z=Z_REST + 0.00026 * i)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_front_door_moving_under_five_mm(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(front=front_at(z=Z_REST + 0.00008 * i)), FAR)
        self.assertEqual(res.score, 1.0)

    def test_rear_door_moving_blocks_too(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(rear=rear_at(dy=0.0003 * i)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_front_door_turning_over_five_degrees_in_a_second(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(scene(front=front_at(lean=0.3 * i)), FAR)
        self.assertEqual(res.score, 0.0)

    def test_still_needs_a_full_window(self):
        tr = S.Tracker()
        res = run(scene(), tracker=tr, steps=WIN - 2)
        self.assertEqual(res.score, 0.0)
        res = run(scene(), tracker=tr, steps=4)
        self.assertEqual(res.score, 1.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        r = S.Tracker().update({S.FRONT: hung()}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])

    def test_nan_state(self):
        bad = mk.state_from_poses(T.FRONT_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(front=bad), FAR).score, 0.0)
        bad = mk.state_from_poses(T.REAR_PART, (L.REAR_X, L.REAR_Y, float("inf"), 0, 0, 0, 1), source="test")
        self.assertEqual(S.Tracker().update(scene(rear=bad), FAR).score, 0.0)

    def test_nan_in_the_middle_restarts_the_windows(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = mk.state_from_poses(T.FRONT_PART, (float("nan"), 0.0, 0.0, 0, 0, 0, 1), source="test")
        self.assertEqual(tr.update(scene(front=bad), FAR).score, 0.0)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # the still window starts again
        self.assertEqual(run(scene(), tracker=tr, steps=WIN + 1).score, 1.0)

    def test_nan_arm(self):
        bad = {"left": rigstate.Arm(np.array([float("nan")] * 3), np.zeros(3), 1.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=bad).score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(scene(rear=rear_at(tilt=12.0)), tracker=tr)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(5)
        tr = S.Tracker()
        for _ in range(400):
            f = front_at(bx=rng.uniform(0.25, 0.5), y=rng.uniform(-0.3, 0.3), z=rng.uniform(-0.01, 0.4),
                         lean=rng.uniform(-20, 100), yaw=rng.uniform(-20, 20), roll=rng.uniform(-30, 30))
            rr = rear_at(dx=rng.uniform(-0.03, 0.03), dy=rng.uniform(-0.1, 0.1), dz=rng.uniform(-0.02, 0.02),
                         tilt=rng.uniform(-15, 15))
            r = tr.update(scene(f, rr), FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))

    def test_every_tier_is_reachable(self):
        seen = {score_of(leaning(l)) for l in (0.0, 5.0)} | {score_of(front_at(z=L.RAIL_TOP)), score_of(start())}
        self.assertEqual(seen, {0.0, 0.25, 0.5, 1.0})


class OfflineModel(unittest.TestCase):
    """The offline stand-in for the physics (task.OfflineModel): what it does with a hand on the pull."""

    def feed(self, model, tcp, lean_deg, opening):
        a = (math.cos(D(lean_deg)), 0.0, -math.sin(D(lean_deg)))
        model.step({"left": arm(tcp, a, opening), "right": FAR["right"]})

    def play(self, variant="", plan=None, hold=3.0):
        """The plan through the arm's IK and this model, the scorer fed every step: (model, tracker, last result)."""
        if OFFLINE is None:
            self.skipTest("no tools/offline")
        from yam_arena.demo import ArmPlan
        ap = ArmPlan(plan if plan is not None else T.plan(variant), (0.0, 0.305, 0.75))
        import yamkin as K
        m, tr, res = T.OfflineModel(), S.Tracker(), None
        for k in range(int((ap.duration + hold) * HZ)):
            pos, rot, fin, _ = ap.target(k / HZ)
            left = rigstate.Arm(np.asarray(pos, float), np.asarray(pos, float) + 0.013 * rot[:, 2],
                                float(np.clip(fin / -0.04695, 0.0, 1.0)))
            arms = {"left": left, "right": FAR["right"]}
            m.step(arms)
            res = tr.update(m.states(), arms)
        return m, tr, res

    def test_the_demo_hangs_the_door(self):
        m, tr, res = self.play()
        self.assertEqual(res.score, 1.0)
        # the plan presses 4 mm into the roof and, through the measured sag, about 17 mm into the floor (a press
        # down onto the floor is harmless: the door rests on it and the arm only pushes it down)
        self.assertLessEqual(m.max_pen, 0.020)
        self.assertAlmostEqual(float(m.b[1]), L.FLOOR_TOP, places=3)
        self.assertAlmostEqual(float(m.b[0]), L.X_F, places=3)
        self.assertAlmostEqual(m.y, L.FRONT_HUNG_Y, places=3)

    def test_let_go_leaning_3_degrees_is_half(self):
        self.assertEqual(self.play("short")[2].score, 0.5)

    def test_let_go_leaning_10_degrees_out_is_zero(self):
        m, tr, res = self.play("out")
        self.assertEqual(res.score, 0.0)

    def test_left_arm_only_hangs_it_and_right_arm_only_cannot(self):
        self.assertEqual(self.play()[2].score, 1.0)
        self.assertEqual(self.play(plan=[])[2].score, 0.0)

    def test_nothing_scores_zero(self):
        m, tr, res = self.play(plan=[], hold=2.0)
        self.assertEqual(res.score, 0.0)

    def test_open_hand_does_not_carry(self):
        m = T.OfflineModel()
        bar = G.tcp_for(m.b, 0.0)
        tcp = (bar[0], L.FRONT_START_Y, L.TOP + bar[1])
        self.feed(m, tcp, 0.0, 1.0)
        self.assertFalse(m.held)
        self.feed(m, (tcp[0], tcp[1], tcp[2] + 0.1), 0.0, 1.0)
        self.assertAlmostEqual(float(m.b[1]), 0.0)

    def test_hand_off_the_pull_does_not_carry(self):
        m = T.OfflineModel()
        bar = G.tcp_for(m.b, 0.0)
        self.feed(m, (bar[0] - 0.05, L.FRONT_START_Y, L.TOP + bar[1]), 0.0, 0.13)
        self.assertFalse(m.held)

    def test_door_cannot_be_pushed_through_the_track(self):
        m = T.OfflineModel()
        bar = G.tcp_for(m.b, 0.0)
        self.feed(m, (bar[0], L.FRONT_START_Y, L.TOP + bar[1]), 0.0, 0.13)
        # lift upright to 5 cm and drive it in: the top hits the track's front wall
        for x in np.linspace(bar[0], bar[0] + 0.08, 20):
            self.feed(m, (x, L.FRONT_START_Y, L.TOP + bar[1] + 0.05), 0.0, 0.13)
        self.assertLess(float(m.b[0]), L.CAB_X0)
        self.assertGreater(m.max_pen, 0.0)

    def test_released_outside_falls_over(self):
        m = T.OfflineModel()
        bar = G.tcp_for(m.b, 0.0)
        self.feed(m, (bar[0], L.FRONT_START_Y, L.TOP + bar[1]), 0.0, 0.13)
        for k in range(1, 6):
            self.feed(m, (bar[0], L.FRONT_START_Y, L.TOP + bar[1] + 0.01 * k), 5.0 * k / 5, 0.13)
        self.feed(m, (bar[0], L.FRONT_START_Y, L.TOP + bar[1] + 0.05), 5.0, 0.9)
        self.assertTrue(m.lying)
        self.assertEqual(S.Tracker().update({S.FRONT: m.states()[S.FRONT], S.REAR: rear_at()}, FAR).score, 0.0)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-9), n)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["cabinet", "front_door", "rear_door"])
        self.assertIsNone(T.CABINET.mass)  # fixed
        self.assertAlmostEqual(T.FRONT_PART.mass, 0.29)
        self.assertAlmostEqual(T.REAR_PART.mass, 0.29)
        self.assertEqual(T.SCENE, "station")

    def test_pull_only_on_the_front_door(self):
        self.assertGreater(len(T.door_shapes(True)), len(T.door_shapes(False)))
        self.assertEqual(len(T.door_shapes(False)), 3)

    def test_instruction_states_the_goal_only(self):
        t = T.INSTRUCTION.lower()
        for word in ("lean", "tilt", "lift", "swing", "tilted", "pull", "handle"):
            self.assertNotIn(word, t)
        self.assertIn("front track", t)
        self.assertLess(len(T.INSTRUCTION), 300)

    def test_demo_plans(self):
        plans = T.demo()
        self.assertEqual(list(plans), ["left"])  # one arm moves
        self.assertGreater(len(plans["left"]), 15)

    def test_demo_env_switches(self):
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_IDLE_ARM"] = "right"
        try:
            self.assertEqual(list(T.demo()), ["left"])
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]
        with self.assertRaises(ValueError):
            T.plan("nope")

    def test_demo_timing(self):
        dur = sum(w.duration for w in T.plan())
        self.assertLessEqual(dur, 28.0)
        self.assertLessEqual((dur + 3.0) * HZ, 0.5 * 1200)  # within half the run limit, hold included
        for a, b in zip(T.plan(), T.plan()[1:]):
            if abs(a.finger - b.finger) > 0.02:
                self.assertGreaterEqual(b.duration, 1.5)  # a gripper open or close takes 1.5 s

    def test_grip_commands(self):
        self.assertAlmostEqual(-2 * T.GRIP, 0.006)
        self.assertGreater(-2 * T.OPEN, L.PULL_SIZE[2] + 0.02)  # open wide enough to go round the bar
        self.assertGreaterEqual(L.PULL_SIZE[2], 0.016)

    def test_sag_lead_inverts_the_measured_sag(self):
        for a in (0.918, 0.9203, 0.928, 0.942, 0.949, 0.982):
            c = T.lead_z(a)
            self.assertAlmostEqual(c + T.sag_z(c), a, places=6)
            self.assertGreaterEqual(c, a)
        # the GPU's measured pairs (commanded -> reached), within 2 mm
        for c, r in ((0.918, 0.918), (0.930, 0.9203), (0.941, 0.928), (0.963, 0.942), (0.973, 0.949)):
            self.assertAlmostEqual(c + T.sag_z(c), r, delta=0.002)
        pos, lean = T.sagged((0.3, 0.0, 0.918), 0.1)  # the hand at the bar, door on the table: no sag
        self.assertEqual((pos[2], lean), (0.918, 0.1))

    def test_demo_keyframes_clear_the_cabinet(self):
        for name, (tx, tz, lean) in T.KEYS[:-1]:
            b = G.top_for((tx, tz), D(lean))
            self.assertGreater(G.clearance(b, D(lean)), 0.003, name)
        b = G.top_for(T.KEYS[-1][1][:2], 0.0)
        self.assertGreater(G.clearance(b, 0.0), -0.0045)  # pressed 4 mm into the roof: the arm's give

    def test_demo_pose_path_clears_the_walls(self):
        """The door's pose along the plan (from the hand's TCP and approach axis, as the offline model reads it)."""
        if OFFLINE is None:
            self.skipTest("no tools/offline")
        from yam_arena.demo import ArmPlan
        ap = ArmPlan(T.plan(), (0.0, 0.305, 0.75))
        worst = {}
        t = 6.6
        while t < ap.duration:
            pos, rot, fin, label = ap.target(t)
            a = rot[:, 2]
            lean = math.atan2(-a[2], a[0])
            (pos, lean) = T.sagged(tuple(float(v) for v in pos), lean)  # the loaded arm as the GPU measured it
            if fin < -0.002 or label.startswith("close"):
                pass
            if fin > T.GRIP + 1e-6 and fin < T.GRIP + 0.001 or abs(fin - T.GRIP) < 1e-6:
                b = G.pose_for_tcp((pos[0], pos[2] - L.TOP), lean)
                worst[label] = min(worst.get(label, 9.0), G.clearance(b, lean))
            t += 0.05
        for label, c in worst.items():
            if "lean" in label or "foot in" in label or "into the groove" in label:
                self.assertGreater(c, 0.003, label)

    def test_lean_pivots_about_the_foot_and_the_top_enters_the_mouth_footprint(self):
        """GPU run 2: the straight hand path between the slide and the lean swung the foot 10 mm toward the floor
        board (the rear-bottom corner met its front face, x 360, and the door stood upright against it). The lean is
        now stepped in the door's own pose: through the lean legs the door's rear-bottom corner stays 8 mm or more
        in front of the cabinet's face, and from the end of the lean through the rise to the mouth the top edge's
        corners stay inside the front groove's footprint (x 368 to 406) with 5 mm to spare each side."""
        if OFFLINE is None:
            self.skipTest("no tools/offline")
        from yam_arena.demo import ArmPlan
        ap = ArmPlan(T.plan(), (0.0, 0.305, 0.75))
        t, seen_lean, seen_rise = 0.0, 0, 0
        while t < ap.duration:
            pos, rot, fin, label = ap.target(t)
            if label.startswith("lean the door") or label.startswith("settle the lean") or label.startswith("up into"):
                a = rot[:, 2]
                lean = math.atan2(-a[2], a[0]) + math.radians(T.SAG_PITCH)  # the door's lean (hand pitch + sag pitch)
                p, lean = T.sagged((float(pos[0]), float(pos[1]), float(pos[2])), lean - math.radians(T.SAG_PITCH))
                b = G.pose_for_tcp((p[0], p[2] - L.TOP), lean)
                rear_bottom = G.to_world(b, lean, G.HALF_T, 0.0)
                if label.startswith("lean the door"):
                    seen_lean += 1
                    self.assertLess(float(rear_bottom[0]), L.CAB_X0 - 0.008, (label, t))
                fx, rx, tz = G.top_corners_x(b, lean)
                if label.startswith("settle the lean") or (label.startswith("up into") and tz < L.TRACK_BOTTOM + 0.004):
                    seen_rise += 1
                    self.assertGreater(fx, L.FRONT_GROOVE[0] + 0.005, (label, t))
                    self.assertLess(rx, L.FRONT_GROOVE[1] - 0.005, (label, t))
            t += 0.05
        self.assertGreater(seen_lean, 30)
        self.assertGreater(seen_rise, 10)

    def test_demo_reach(self):
        if OFFLINE is None:
            self.skipTest("no tools/offline")
        import yamkin as K
        from yam_arena.demo import ArmPlan
        ap = ArmPlan(T.plan(), (0.0, 0.305, 0.75))
        t, worst, margin = 0.0, 0.0, 9.0
        while t < ap.duration:
            ap.action7(t)
            worst = max(worst, ap.last_ik_mm)
            margin = min(margin, float(np.degrees(np.min(K.joint_margins(ap.q)))))
            t += 0.05
        self.assertLess(worst, 1.0)
        self.assertGreater(margin, 3.0)

    def test_hand_stays_clear_of_the_cabinet(self):
        """The pad tips (13 mm past the TCP along the approach axis) stay 8 mm or more in front of the cabinet's face
        and the track's front wall, wherever the door is."""
        if OFFLINE is None:
            self.skipTest("no tools/offline")
        from yam_arena.demo import ArmPlan
        ap = ArmPlan(T.plan(), (0.0, 0.305, 0.75))
        t = 0.0
        while t < ap.duration:
            pos, rot, fin, label = ap.target(t)
            tip = np.asarray(pos) + 0.013 * rot[:, 2]
            self.assertLess(float(tip[0]), L.CAB_X0 - 0.008, (label, t))
            t += 0.05


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_basics(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["category"], "Societal relatability")
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_scoring_text_matches_the_code(self):
        sc = json.loads((HERE / "task.json").read_text())["scoring"]
        for token in ("6 mm or more above its mouth", "z 316 mm", "z 37 to 43 mm", "7 mm", "x 387 mm",
                      "10 mm or less along y",
                      "5 mm along x", "4 degrees", "30 mm or more", "1 s", "5 mm or turns 5 degrees", "z 30 mm",
                      "10 degrees", "20 mm in x or 60 mm in y", "lean under 3 degrees", "3 to 8 degrees",
                      "Possible scores: 0, 0.25, 0.5, 1.0", "depth gauge", "feeler", "angle gauge", "spacer"):
            self.assertIn(token, sc, token)
        self.assertAlmostEqual(L.TRACK_BOTTOM + L.IN_GROOVE, 0.316)
        self.assertAlmostEqual(L.FLOOR_TOP + L.FOOT_DOWN, 0.043)
        self.assertEqual((L.FACE_TOL, L.TILT_FULL, L.TILT_PART), (0.007, 3.0, 8.0))
        self.assertEqual((L.REAR_Y_TOL, L.REAR_X_TOL, L.REAR_TILT_KEPT, L.REAR_TILT_FELL), (0.010, 0.005, 4.0, 10.0))
        self.assertEqual((L.STILL_MOVE, L.STILL_TURN_DEG, L.LET_GO, L.FALL_Z), (0.005, 5.0, 0.030, 0.030))

    def test_fidelity_and_context(self):
        d = json.loads((HERE / "task.json").read_text())
        for token in ("Rigid bodies", "0.4", "est", "no rollers", "8 mm of play", "4 mm each side of the 6 mm rail"):
            self.assertIn(token, d["fidelity"])
        self.assertAlmostEqual((L.FOOT_GROOVE_W - L.RAIL_W) / 2, 0.004)
        self.assertIn("taller than the opening", d["context"])

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        for f in ("task.json", "README.md", "task.py", "layout.py", "geom.py", "scoring.py", "rigstate.py",
                  "mechkit.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)

    def test_text_rules(self):
        bad = re.compile("\u2014|\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "geom.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f not in ("task.json", "README.md") else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
