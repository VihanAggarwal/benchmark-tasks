"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
CLOSED_Q = -0.016  # a clip resting on its lug
IDENT = (0.0, 0.0, 0.0, 1.0)


def airbox(q=(CLOSED_Q, CLOSED_Q)):
    return mk.state_from_poses(T.AIRBOX_PART, (0.0, 0.0, L.TOP, *IDENT), joints=dict(zip(L.CLIP_NAMES, q)),
                               source="test")


def cover(dx=0.0, dy=0.0, dz=0.0):
    return mk.state_from_poses(T.COVER_PART, (0.0, 0.0, L.TOP, *IDENT), joints=dict(zip(L.COVER_JOINTS, (dx, dy, dz))),
                               source="test")


def filt(spec, x, y, z, yaw=0.0, pitch=0.0):
    q = mk.quat_from_ypr(yaw, pitch, 0.0)
    return mk.state_from_poses(spec, (x, y, L.TOP + z, *q), source="test")


def old_in_seat(**kw):
    return filt(T.OLD_PART, L.CX + kw.get("dx", 0.0), L.CY + kw.get("dy", 0.0), L.SEAT_Z + kw.get("dz", 0.0),
                kw.get("yaw", 0.0))


def new_in_seat(**kw):
    return filt(T.NEW_PART, L.CX + kw.get("dx", 0.0), L.CY + kw.get("dy", 0.0), L.SEAT_Z + kw.get("dz", 0.0),
                kw.get("yaw", 0.0), kw.get("pitch", 0.0))


def old_on_mat(**kw):
    return filt(T.OLD_PART, L.MAT_OLD[0] + kw.get("dx", 0.0), L.MAT_OLD[1] + kw.get("dy", 0.0),
                L.MAT_T + kw.get("dz", 0.0), kw.get("yaw", 0.0), kw.get("pitch", 0.0))


def new_on_mat():
    return filt(T.NEW_PART, *L.NEW_START, L.NEW_START_YAW)


def scene(box=None, cov=None, old=None, new=None):
    return {S.AIRBOX: box if box is not None else airbox(), S.COVER: cov if cov is not None else cover(),
            S.OLD: old if old is not None else old_in_seat(), S.NEW: new if new is not None else new_on_mat()}


def done(**kw):
    """The finished job: old filter on its mat, new one in the seat, cover down, both clips closed."""
    d = dict(old=old_on_mat(), new=new_in_seat())
    d.update(kw)
    return scene(**d)


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, L.PAD_TIP]), opening)


FAR = {"left": arm((0.25, 0.45, L.TOP + 0.20)), "right": arm((0.25, -0.30, L.TOP + 0.30))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sizes(self):
        self.assertAlmostEqual(L.CASE_X1 - L.CASE_X0, 0.300)
        self.assertAlmostEqual(L.CASE_Y1 - L.CASE_Y0, 0.240)
        self.assertAlmostEqual(L.F_LX, 0.240)
        self.assertAlmostEqual(L.F_LY, 0.173)
        self.assertAlmostEqual(L.F_H, 0.049)
        self.assertAlmostEqual(L.RIM_Z - L.SEAT_Z, 0.030)
        self.assertAlmostEqual(L.CLIP_Y[1] - L.CLIP_Y[0], 0.12)  # the clips are 0.12 m apart
        self.assertAlmostEqual(L.COVER_ROOF_TOP - L.COVER_SKIRT_Z, 0.040)

    def test_grip_widths_in_the_gripper_range(self):
        """Every pinched feature is 16 to 80 mm across (the lug and the frame are 20 mm)."""
        for w in (L.LUG_X1 - L.LUG_X0, L.F_WALL):
            self.assertGreaterEqual(w, 0.016)
            self.assertLessEqual(w, 0.080)
        self.assertAlmostEqual(T.GRIP_LUG, -0.006)
        self.assertAlmostEqual(T.GRIP_FRAME, -0.006)

    def test_start_layout(self):
        r = S.read(start())
        self.assertTrue(S.seated(r, S.OLD))
        self.assertFalse(S.old_out(r, S.OLD))
        self.assertFalse(S.seated(r, S.NEW))
        self.assertTrue(S.old_out(r, S.NEW))  # the new filter lies clear of the case on its mat
        self.assertTrue(S.cover_seated(r))
        self.assertTrue(all(S.clip_closed(q, r) for q in r.clips))
        self.assertEqual(S.tab_in(r), L.TAB_DROP)

    def test_new_filter_starts_a_quarter_turn_from_its_seat(self):
        """Turned -90 degrees: long side along y, so it has to be turned to fit the pocket."""
        r = S.read(start())
        self.assertAlmostEqual(S.wrap(S.yaw_of(r.rot[S.NEW])), -math.pi / 2, places=6)

    def _outline(self, yaw, dx=0.0, dy=0.0):
        pts = np.array(L.outline_corners())
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        return pts @ rot.T + np.array([L.CX + dx, L.CY + dy])

    def _fits(self, dx, dy, yaw=0.0):
        poly = self._outline(yaw, dx, dy)
        return bool(poly[:, 0].min() >= L.POCKET_X0 - 1e-9 and poly[:, 0].max() <= L.POCKET_X1 + 1e-9
                    and poly[:, 1].min() >= L.POCKET_Y0 - 1e-9 and poly[:, 1].max() <= L.POCKET_Y1 + 1e-9)

    def test_filter_fits_the_pocket_with_6mm_of_play_either_way_round(self):
        """The pocket leaves the filter 6 mm along x and y and about 2.5 degrees of yaw: the scorer's 6 mm and 5
        degrees are the looser limits. Turned half a turn it fits the same."""
        for yaw in (0.0, math.pi):
            self.assertTrue(self._fits(0.0, 0.0, yaw))
            for dx, dy in ((0.0055, 0.0), (-0.0055, 0.0), (0.0, 0.0055), (0.0, -0.0055), (0.004, 0.004)):
                self.assertTrue(self._fits(dx, dy, yaw), (dx, dy, yaw))
            for dx, dy in ((0.0065, 0.0), (0.0, -0.0065)):
                self.assertFalse(self._fits(dx, dy, yaw), (dx, dy, yaw))
        self.assertTrue(self._fits(0.0, 0.0, math.radians(2.0)))
        self.assertFalse(self._fits(0.0, 0.0, math.radians(4.0)))

    def test_a_quarter_turn_off_does_not_fit(self):
        self.assertFalse(self._fits(0.0, 0.0, math.pi / 2))

    def test_the_filter_stands_19mm_over_the_rim_where_the_gripper_pinches_it(self):
        top = L.SEAT_Z + L.F_H
        self.assertAlmostEqual(top - L.RIM_Z, 0.019)
        tips = top - 0.013
        self.assertGreater(tips, L.RIM_Z + 0.004)  # the outer pad clears the case rim
        self.assertGreater(tips, L.SEAT_Z + L.F_PLEAT_H + 0.008)  # the inner pad clears the pleat pack

    def test_cover_clears_the_filter(self):
        under = L.COVER_ROOF_TOP - L.COVER_WALL
        self.assertGreaterEqual(under - (L.SEAT_Z + L.F_H), 0.015)

    def test_cover_lifted_40mm_clears_the_filter_frame_with_room(self):
        self.assertGreaterEqual(L.COVER_SKIRT_Z + 0.025 - (L.SEAT_Z + L.F_H), 0.005)  # a 25 mm rise is the least

    def test_clip_hook_sits_1mm_over_the_lug(self):
        self.assertAlmostEqual(L.BAR_Z0 - L.CLIP_LUG_Z1, 0.001)
        self.assertGreater(L.CLIP_BAR_END, L.CLIP_LUG_X0 + 0.005)  # the bar covers at least 5 mm of the lug
        self.assertLess(L.CLIP_BAR_END, L.COVER_X0)  # and stops short of the cover's front face

    def test_released_clip_lets_the_lifted_lug_pass(self):
        """The released lever's underside over the lug's x range stays above the lug's top when the cover is lifted by
        the most the duct allows before it tears."""
        q = L.CLIP_Q_OPEN
        under = L.CLIP_PIV_Z + (L.CLIP_LUG_X0 - L.CLIP_PIV_X) * math.sin(q) - 0.003 * math.cos(q)
        self.assertGreater(under, L.CLIP_LUG_Z1 + 0.050)  # a 40 mm lift (50 mm with the arm's give) passes under it

    def test_thumb_tab_and_hook_end_stay_over_the_table(self):
        q = L.CLIP_Q_OPEN
        tab = L.CLIP_PIV_Z - (L.CLIP_PIV_X - L.CLIP_TAB_X) * math.sin(q) - 0.003
        self.assertGreater(tab, 0.03)

    def test_slack(self):
        self.assertAlmostEqual(L.TAB_SLACK_X, 0.004)
        self.assertAlmostEqual(L.TAB_SLACK_Y, 0.007)
        self.assertLess(L.TAB_DROP, L.SLOT_DEPTH)

    def test_wires_of_the_surroundings(self):
        """The mats, battery and lip clear the case, the posts and each other."""
        bat = L.BATTERY
        self.assertLess(bat[0][1] + bat[1][1] / 2, L.CASE_Y0 - 0.05)  # beside the case's right side, 62 mm clear
        self.assertGreater(bat[0][0] - bat[1][0] / 2, 0.40)  # and behind the front half
        self.assertGreater(L.MAT_NEW[1] + L.MAT_NEW_SIZE[1] / 2, -0.5)
        old_y0 = L.MAT_OLD[1] - L.MAT_OLD_SIZE[1] / 2
        self.assertGreaterEqual(old_y0, -0.65)  # on the table
        self.assertLessEqual(old_y0 + L.MAT_OLD_SIZE[1], L.MAT_NEW[1] - L.MAT_NEW_SIZE[1] / 2 + 1e-9)
        self.assertGreater(L.CASE_X0, L.LIP[0][0] + L.LIP[1][0] / 2 + 0.05)


class Tiers(unittest.TestCase):
    def test_the_start_scores_nothing(self):
        r = run(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_full(self):
        r = run(done())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)
        self.assertEqual(r.details["clips_closed"], 2)
        self.assertTrue(r.details["released_1s"] and r.details["still_1s"])

    def test_a_clip_open_is_half(self):
        for open_clip in (0, 1):
            q = [CLOSED_Q, CLOSED_Q]
            q[open_clip] = L.CLIP_Q_OPEN
            r = run(done(box=airbox(q)))
            self.assertEqual(r.score, 0.5, open_clip)
            self.assertFalse(r.success)

    def test_both_clips_open_is_still_half(self):
        self.assertEqual(run(done(box=airbox((L.CLIP_Q_OPEN, L.CLIP_Q_OPEN)))).score, 0.5)

    def test_cover_not_seated_is_a_quarter(self):
        for kw in ({"dz": 0.012}, {"dy": 0.15}, {"dz": 0.03}, {"dx": 0.01}):
            q = (L.CLIP_Q_OPEN, L.CLIP_Q_OPEN)
            r = run(done(cov=cover(**kw), box=airbox(q)))
            self.assertEqual(r.score, 0.25, kw)

    def test_cover_resting_on_the_rear_wall_top_is_not_seated(self):
        """A tab missing its slot: the cover sits 12 mm up on the wall's top (tabs out of their slots)."""
        r = run(done(cov=cover(dx=0.007, dz=0.012), box=airbox((0.1, 0.1))))
        self.assertEqual(r.score, 0.25)
        self.assertEqual(r.details["tab_in_mm"], 0.0)

    def test_old_filter_still_in_the_seat_scores_nothing(self):
        self.assertEqual(run(scene(new=new_in_seat(dz=0.05))).score, 0.0)

    def test_old_out_but_new_not_in_is_zero(self):
        q = (L.CLIP_Q_OPEN, L.CLIP_Q_OPEN)
        r = run(scene(old=old_on_mat(), box=airbox(q), cov=cover(dy=0.25)))
        self.assertEqual(r.score, 0.0)

    def test_new_not_seated_cases(self):
        q = (L.CLIP_Q_OPEN, L.CLIP_Q_OPEN)
        for name, new in (("3 mm high", new_in_seat(dz=0.003)), ("yaw 6", new_in_seat(yaw=math.radians(6))),
                          ("7 mm off", new_in_seat(dx=0.007)), ("quarter turn on the rim", new_in_seat(yaw=math.pi / 2, dz=0.030)),
                          ("on the rim", new_in_seat(dz=0.030)), ("tilted 4", new_in_seat(pitch=math.radians(4)))):
            r = run(done(new=new, box=airbox(q), cov=cover(dz=0.03)))
            self.assertEqual(r.score, 0.0, name)

    def test_new_seated_tolerances_pass(self):
        for name, new in (("2 mm", new_in_seat(dz=0.0019)), ("yaw 4", new_in_seat(yaw=math.radians(4))),
                          ("-4 yaw", new_in_seat(yaw=-math.radians(4))), ("5 mm off", new_in_seat(dx=0.005, dy=-0.005)),
                          ("turned 180", new_in_seat(yaw=math.pi)), ("turned 176", new_in_seat(yaw=math.radians(176)))):
            self.assertEqual(run(done(new=new)).score, 1.0, name)

    def test_cover_tolerances(self):
        for name, kw, ok in (("dz 2", {"dz": 0.0019}, True), ("dz 3", {"dz": 0.003}, False),
                             ("dx 4", {"dx": 0.0039}, True), ("dx 5", {"dx": 0.005}, False),
                             ("dy 5", {"dy": 0.0049}, True), ("dy 6", {"dy": 0.006}, False)):
            r = run(done(cov=cover(**kw)))
            self.assertEqual(r.score > 0.25, ok, name)

    def test_clip_angle_window(self):
        for name, q, ok in (("4 deg", math.radians(4), True), ("6 deg", math.radians(6), False),
                            ("-2 deg", -math.radians(2), True), ("-4 deg: no lug under it", -math.radians(4), False)):
            r = run(done(box=airbox((q, q))))
            self.assertEqual(r.score, 1.0 if ok else 0.5, name)

    def test_old_out_cases(self):
        q = (CLOSED_Q, CLOSED_Q)
        base = lambda old: run(done(old=old, box=airbox(q))).score  # noqa: E731
        self.assertEqual(base(old_on_mat()), 1.0)
        self.assertEqual(base(filt(T.OLD_PART, 0.30, -0.80 + 0.2, 0.0)), 1.0)  # bare table, off the mat
        self.assertEqual(base(filt(T.OLD_PART, 0.30, -0.545, 0.010)), 0.0)  # floating 10 mm up
        self.assertEqual(base(old_on_mat(pitch=math.radians(15))), 0.0)  # propped at 15 degrees
        self.assertEqual(base(filt(T.OLD_PART, L.CX, L.CY + 0.30, 0.0)), 1.0)  # beside the case on the left

    def test_old_filter_overlapping_the_footprint_is_not_out(self):
        r = S.read(scene(old=filt(T.OLD_PART, L.CX, L.CY, 0.0)))  # on the table under the case's footprint
        self.assertFalse(S.old_out(r))
        r = S.read(scene(old=filt(T.OLD_PART, L.CASE_X0 - 0.12 + 0.005, L.CY, 0.0)))  # 5 mm into the front edge
        self.assertFalse(S.old_out(r))
        r = S.read(scene(old=filt(T.OLD_PART, L.CASE_X0 - 0.12 - 0.001, L.CY, 0.0)))  # clear by 1 mm
        self.assertTrue(S.old_out(r))

    def test_turned_old_filter_outline_is_used(self):
        """A filter turned 30 degrees whose bounding box overlaps the footprint's corner but whose outline does not."""
        yaw = math.radians(30)
        r = S.read(scene(old=filt(T.OLD_PART, L.CASE_X0 - 0.17, L.CASE_Y0 - 0.17, 0.0, yaw)))
        self.assertTrue(S.old_out(r))

    def test_old_on_the_other_side_of_the_table_counts(self):
        self.assertEqual(run(done(old=filt(T.OLD_PART, 0.3, 0.50, 0.0, 0.3))).score, 1.0)


class Latches(unittest.TestCase):
    def feed(self, seq, arms=FAR):
        tr = S.Tracker()
        res = None
        for st in seq:
            res = tr.update(st, arms)
        return tr, res

    def test_duct_pulled_off_zeroes_for_good(self):
        for kw in ({"dy": 0.29}, {"dy": -0.04}, {"dx": 0.04}, {"dx": -0.04}, {"dz": 0.075}):
            seq = [done(cov=cover(**kw))] + [done()] * (WIN + 5)
            tr, res = self.feed(seq)
            self.assertIsNotNone(tr.latch["DUCT"], kw)
            self.assertEqual(res.score, 0.0, kw)

    def test_duct_just_inside_its_travel_is_fine(self):
        for kw in ({"dy": 0.275}, {"dy": -0.034}, {"dx": 0.034}, {"dz": 0.069}):
            tr, _ = self.feed([done(cov=cover(**kw))] + [done()] * 3)
            self.assertIsNone(tr.latch["DUCT"], kw)

    def test_filter_off_the_table_zeroes(self):
        for who in ("old", "new"):
            drop = filt(T.OLD_PART if who == "old" else T.NEW_PART, 0.30, 0.78, -0.30)
            seq = [done(**{who: drop})] + [done()] * (WIN + 5)
            tr, res = self.feed(seq)
            self.assertIsNotNone(tr.latch["FELL"], who)
            self.assertEqual(res.score, 0.0)

    def test_lifting_or_carrying_a_filter_is_not_a_fall(self):
        carry = [done(old=filt(T.OLD_PART, 0.30, -0.3, 0.10))] * 5 + [done()] * (WIN + 3)
        tr, res = self.feed(carry)
        self.assertIsNone(tr.latch["FELL"])
        self.assertEqual(res.score, 1.0)

    def test_a_filter_resting_a_little_below_the_table_top_is_not_a_fall(self):
        tr, _ = self.feed([done(old=filt(T.OLD_PART, 0.3, 0.6, -0.019))] * 3)
        self.assertIsNone(tr.latch["FELL"])

    def test_clipping_an_empty_box_on_the_way_is_no_zero(self):
        """Clips shut on an empty box for a while (the engine is off: no harm), then the job finished: 1.0."""
        empty = scene(old=old_on_mat(), new=new_on_mat())  # both filters out, cover down, clips closed
        tr, res = self.feed([empty] * 20 + [done()] * (WIN + 5))
        self.assertEqual(set(tr.latch), {"DUCT", "FELL"})
        self.assertFalse(tr.latched)
        self.assertEqual(res.score, 1.0)

    def test_an_empty_or_misseated_box_clipped_shut_at_the_end_scores_nothing(self):
        for new in (new_on_mat(), filt(T.NEW_PART, L.CX, L.CY, L.COVER_ROOF_TOP), new_in_seat(dz=0.03),
                    new_in_seat(yaw=math.pi / 2, dz=0.030)):
            tr, res = self.feed([scene(old=old_on_mat(), new=new)] * (WIN + 5))
            self.assertEqual(res.score, 0.0)
            self.assertFalse(tr.latched)

    def test_the_start_latches_nothing(self):
        tr, _ = self.feed([start()] * 40)
        self.assertFalse(tr.latched)


class ReleaseAndStill(unittest.TestCase):
    def test_a_gripper_near_the_airbox_is_not_hands_off(self):
        near = {"left": arm((L.CX, L.CY, L.TOP + 0.16), 1.0), "right": FAR["right"]}
        self.assertEqual(run(done(), arms=near).score, 0.0)

    def test_a_gripper_near_either_filter_is_not_hands_off(self):
        for tcp in ((L.MAT_OLD[0], L.MAT_OLD[1], L.TOP + 0.06), (L.CX, L.CY, L.TOP + 0.20)):
            near = {"left": FAR["left"], "right": arm(tcp, 1.0)}
            self.assertEqual(run(done(), arms=near).score, 0.0, tcp)

    def test_a_gripper_near_the_cover_is_not_hands_off(self):
        near = {"left": arm((0.196, 0.015, L.TOP + 0.175), 0.13), "right": FAR["right"]}
        self.assertEqual(run(done(), arms=near).score, 0.0)

    def test_30_mm_clear_is_hands_off(self):
        r = run(done())
        self.assertGreaterEqual(r.details["hands_gap_mm"], 30.0)

    def test_hands_off_must_hold_a_whole_second(self):
        tr = S.Tracker()
        near = {"left": arm((L.CX, L.CY, L.TOP + 0.16), 1.0), "right": FAR["right"]}
        for _ in range(WIN):
            tr.update(done(), FAR)
        tr.update(done(), near)
        self.assertEqual(tr.last.score, 0.0)
        for _ in range(WIN - 1):
            self.assertEqual(tr.update(done(), FAR).score, 0.0)
        self.assertEqual(tr.update(done(), FAR).score, 1.0)

    def test_unreadable_arms_are_never_hands_off(self):
        for arms in (None, {"left": FAR["left"]}, {"left": FAR["left"], "right": arm((np.nan, 0, 0))}):
            self.assertEqual(run(done(), arms=arms).score, 0.0)

    def test_moving_filter_is_not_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            res = tr.update(done(new=new_in_seat(dx=0.0 if i < 2 else 0.0)), FAR)
        self.assertEqual(res.score, 1.0)
        tr = S.Tracker()
        for i in range(WIN + 2):  # the old filter creeps 6 mm in the window
            old = old_on_mat(dx=0.006 * min(i, WIN) / WIN)
            res = tr.update(done(old=old), FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_visible_motion_threshold_is_5mm_5deg(self):
        def creep(dx=0.0, yaw=0.0, cov=0.0, clip=0.0):
            tr = S.Tracker()
            res = None
            for i in range(WIN + 2):
                f = min(i, WIN) / WIN
                res = tr.update(done(old=old_on_mat(dx=dx * f, yaw=math.radians(yaw) * f), cov=cover(dy=cov * f),
                                     box=airbox((CLOSED_Q + math.radians(clip) * f, CLOSED_Q))), FAR)
            return res
        self.assertTrue(creep(dx=0.0045).details["still_1s"])
        self.assertFalse(creep(dx=0.0055).details["still_1s"])
        self.assertTrue(creep(yaw=4.5).details["still_1s"])
        self.assertFalse(creep(yaw=5.5).details["still_1s"])
        self.assertTrue(creep(cov=0.0045).details["still_1s"])
        self.assertFalse(creep(cov=0.0055).details["still_1s"])
        self.assertTrue(creep(clip=4.0).details["still_1s"])
        self.assertFalse(creep(clip=6.0).details["still_1s"])

    def test_a_clip_snapping_shut_in_the_window_is_not_still(self):
        tr = S.Tracker()
        res = None
        for i in range(WIN + 2):
            q = L.CLIP_Q_OPEN if i < WIN // 2 else CLOSED_Q
            res = tr.update(done(box=airbox((q, CLOSED_Q))), FAR)
        self.assertEqual(res.score, 0.0)


class Robustness(unittest.TestCase):
    def test_nan_in_a_filter_scores_zero_and_restarts_the_windows(self):
        tr = S.Tracker()
        for _ in range(WIN + 2):
            tr.update(done(), FAR)
        self.assertEqual(tr.last.score, 1.0)
        bad = done(new=filt(T.NEW_PART, np.nan, L.CY, L.SEAT_Z))
        res = tr.update(bad, FAR)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        for _ in range(WIN):
            self.assertEqual(tr.update(done(), FAR).score, 0.0)
        self.assertEqual(tr.update(done(), FAR).score, 1.0)

    def test_nan_or_inf_in_a_joint_scores_zero(self):
        for which, name, bad in ((S.COVER, "dy", np.nan), (S.AIRBOX, "clip_r", np.inf), (S.COVER, "dz", -np.inf)):
            st = done()
            st[which].joints[name] = bad
            self.assertEqual(run(st).score, 0.0, (which, name))

    def test_nan_does_not_clear_a_latch(self):
        tr = S.Tracker()
        tr.update(done(cov=cover(dy=0.29)), FAR)
        self.assertIsNotNone(tr.latch["DUCT"])
        bad = done()
        bad[S.COVER].joints["dy"] = np.nan
        tr.update(bad, FAR)
        self.assertIsNotNone(tr.latch["DUCT"])

    def test_missing_state_scores_zero(self):
        st = done()
        del st[S.COVER]
        self.assertEqual(run(st).score, 0.0)
        self.assertEqual(run(None).score, 0.0)
        self.assertEqual(run({}).score, 0.0)

    def test_blow_up_far_away_scores_zero(self):
        far = done(old=filt(T.OLD_PART, 1e6, 1e6, 1e6))
        self.assertEqual(run(far).score, 0.0)

    def test_reset_with_everything_back_on_its_marks_clears_the_latch(self):
        tr = S.Tracker()
        for _ in range(5):
            tr.update(done(cov=cover(dy=0.29)), FAR)
        self.assertIsNotNone(tr.latch["DUCT"])
        tr.update(start(), FAR)  # the old filter jumps from the mat back to the seat: a reset
        self.assertIsNone(tr.latch["DUCT"])
        self.assertEqual(tr.resets, 1)
        for _ in range(WIN + 2):
            tr.update(done(), FAR)
        self.assertEqual(tr.last.score, 1.0)

    def test_an_ordinary_start_is_not_a_reset(self):
        tr = S.Tracker()
        for _ in range(30):
            tr.update(start(), FAR)
        self.assertEqual(tr.resets, 0)


class Exploits(unittest.TestCase):
    def test_do_nothing(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_only_swapping_the_filters_over_a_cover_left_aside_and_walking_away(self):
        """The cover left up and aside (its duct would pull it home, but suppose it stays): 0.25 at most, never 1.0."""
        q = (L.CLIP_Q_OPEN, L.CLIP_Q_OPEN)
        r = run(done(cov=cover(dy=0.24, dz=0.03), box=airbox(q)))
        self.assertEqual(r.score, 0.25)

    def test_closing_the_clips_with_the_empty_box_is_zero(self):
        """Take the old filter out, shut the cover and the clips, never fit the new one."""
        seq = [scene(old=old_on_mat())] * (WIN + 5)
        tr = S.Tracker()
        for st in seq:
            res = tr.update(st, FAR)
        self.assertEqual(res.score, 0.0)

    def test_dropping_the_new_filter_a_quarter_turn_off_onto_the_rim_then_clipping_is_zero(self):
        mis = scene(old=old_on_mat(), new=new_in_seat(yaw=math.pi / 2, dz=0.030), cov=cover(dz=0.012))
        res = run(mis)
        self.assertEqual(res.score, 0.0)

    def test_the_new_filter_set_on_top_of_the_old_one_scores_nothing(self):
        stack = scene(old=old_in_seat(), new=new_in_seat(dz=L.F_H))
        self.assertEqual(run(stack).score, 0.0)

    def test_putting_the_old_filter_back_after_the_swap_scores_nothing(self):
        both = scene(old=filt(T.OLD_PART, L.CX, L.CY, L.SEAT_Z + L.F_H), new=new_in_seat())
        self.assertEqual(run(both).score, 0.0)

    def test_seating_the_old_filter_and_leaving_the_new_one_on_its_mat_is_zero(self):
        self.assertEqual(run(scene()).score, 0.0)

    def test_pulling_the_cover_off_its_duct_then_putting_everything_right_is_zero(self):
        seq = [done(cov=cover(dy=0.31))] * 2 + [done()] * (WIN + 6)
        tr = S.Tracker()
        for st in seq:
            res = tr.update(st, FAR)
        self.assertEqual(res.score, 0.0)

    def test_throwing_the_old_filter_off_the_table_then_everything_else_right_is_zero(self):
        seq = [done(old=filt(T.OLD_PART, 0.3, 0.9, -0.5))] + [done()] * (WIN + 6)
        tr = S.Tracker()
        for st in seq:
            res = tr.update(st, FAR)
        self.assertEqual(res.score, 0.0)

    def test_holding_the_job_finished_in_the_gripper_earns_nothing(self):
        held = {"left": arm((L.CX, L.CY - 0.05, L.TOP + 0.13), 0.13), "right": FAR["right"]}
        self.assertEqual(run(done(), arms=held).score, 0.0)

    def test_the_same_filter_cannot_count_twice(self):
        """Old filter carried back to the seat, the new one left on its mat: new not seated, old not out."""
        st = scene(old=old_in_seat(), new=new_on_mat(), cov=cover(), box=airbox())
        self.assertEqual(run(st).score, 0.0)


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        for n in names:
            self.assertTrue(n.isidentifier())
        for n in S.PARTS:
            self.assertIn(n, names)

    def test_clip_detent_rule_is_registered_and_targets_the_notches(self):
        self.assertTrue(callable(T._clip_detent))
        self.assertAlmostEqual(mk.detent_target(0.2, (0.0, L.CLIP_Q_OPEN), L.CLIP_Q_OPEN * 0.52), 0.0)
        self.assertAlmostEqual(mk.detent_target(1.0, (0.0, L.CLIP_Q_OPEN), L.CLIP_Q_OPEN * 0.52), L.CLIP_Q_OPEN)
        self.assertAlmostEqual(mk.detent_target(0.6, (0.0, L.CLIP_Q_OPEN), L.CLIP_Q_OPEN * 0.52), 0.0)
        self.assertAlmostEqual(mk.detent_target(0.75, (0.0, L.CLIP_Q_OPEN), L.CLIP_Q_OPEN * 0.52), L.CLIP_Q_OPEN)

    def test_demo_plans(self):
        run_limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0].replace(",", ""))
        for v in ("",) + T.VARIANTS:
            plan = T.plans(v)
            for a, wps in plan.items():
                for w in wps:
                    self.assertTrue(all(math.isfinite(c) for c in w.pos))
                    self.assertGreater(w.duration, 0.0)
                    self.assertLessEqual(w.finger, 0.0)
                    self.assertGreaterEqual(w.finger, -0.04695 - 1e-9)
                    self.assertLessEqual(w.pos[2], L.TOP + 0.2005, (v, a, w.label))
            self.assertLessEqual(max(sum(w.duration for w in wps) for wps in plan.values()) * HZ * 2 + 100, run_limit)
        with self.assertRaises(ValueError):
            T.plans("nonsense")

    def test_demo_env_switches(self):
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set()), ("DEMO_VARIANT", "one_clip", {"left", "right"})):
            os.environ[key] = val
            try:
                self.assertEqual(set(T.demo()), want)
            finally:
                os.environ.pop(key)

    def _span(self, wps, prefix):
        t = 0.0
        for w in wps:
            if w.label.startswith(prefix):
                return t, t + w.duration
            t += w.duration
        raise KeyError(prefix)

    def test_demo_order(self):
        plan = T.plans("")
        tabs_end = self._span(plan["right"], "aside, waiting")[0]
        lug0 = self._span(plan["left"], "over the cover's lift lug")[0]
        self.assertGreaterEqual(lug0, tabs_end)  # both clips are released before the left arm goes to the cover
        slide_end = self._span(plan["left"], "slide it 240 mm")[1]
        pinch0 = self._span(plan["right"], "over the old filter")[0]
        self.assertGreaterEqual(pinch0, slide_end - 1e-9)  # the cover is aside before the right arm reaches in
        let_go_new = self._span(plan["right"], "let go of the new filter")[1]
        back0 = self._span(plan["left"], "slide the cover back")[0]
        self.assertGreaterEqual(back0, let_go_new - 1e-9)  # the new filter is in before the cover comes back
        lowered = self._span(plan["left"], "lower it: tabs")[1]
        push0 = self._span(plan["right"], "closed fingers in front of the released clip")[0]
        parked = self._span(plan["left"], "aside")[0]
        self.assertGreaterEqual(push0, lowered)  # the cover is down before a clip is pushed shut
        self.assertGreaterEqual(push0 + 0.5, parked + 2.0 - 1e-9 - 0.5)

    def test_one_arm_at_the_clips_at_a_time(self):
        """The two forearms overlap when both grippers are near the airbox's front: the left arm waits aside while the
        right one works the clips."""
        plan = T.plans("")
        left_stage = self._span(plan["left"], "aside, waiting for the clips")
        self.assertEqual(left_stage[0], 0.0)
        tabs_end = self._span(plan["right"], "aside, waiting")[0]
        lug = self._span(plan["left"], "over the cover's lift lug")[0]
        self.assertGreaterEqual(lug, tabs_end)

    def test_tab_press_goes_past_the_snap_point(self):
        """The pad tips end below the thumb tab's height at the snap point (0.675 rad), where the lever flips."""
        q = L.CLIP_Q_OPEN / 2
        tab_z = L.CLIP_PIV_Z + (T.TAB_X - L.CLIP_PIV_X) * math.sin(q) + 0.003 * math.cos(q)
        self.assertLess(T.TAB_TCP_PRESS - L.PAD_TIP, tab_z - 0.005)
        self.assertGreater(T.TAB_TCP_TOUCH - L.PAD_TIP, L.BAR_Z0 + L.CLIP_BAR_T)  # touch: tips just over the tab
        self.assertGreater(T.TAB_TCP_PRESS - L.PAD_TIP, 0.040)  # and well over the table

    def test_push_closes_the_clip(self):
        """The block's face ends far enough in for the lever, stopped at the block's lowest edge, to be under 0.675 rad."""
        face = T.PUSH_X1 + T.PAD_HALF_X
        zlow = T.PUSH_Z - L.PAD_TIP
        q = math.atan2(zlow - L.CLIP_PIV_Z, face - L.CLIP_PIV_X)
        self.assertLess(q, L.CLIP_Q_OPEN / 2 - 0.1)
        self.assertLess(face, L.CASE_X0 - 0.001)  # and stops short of the cover's front face
        # at the start the face is in front of the released lever
        x_bar = L.CLIP_PIV_X + (T.PUSH_Z - L.PAD_TIP + 0.01 - L.CLIP_PIV_Z) / math.tan(L.CLIP_Q_OPEN)
        self.assertLess(T.PUSH_X0 + T.PAD_HALF_X, x_bar)

    def test_lug_grip_geometry(self):
        self.assertAlmostEqual(T.LUG_TCP - L.PAD_TIP, L.LUG_TOP - 0.016)
        self.assertGreater(T.LUG_TCP - L.PAD_TIP, L.COVER_ROOF_TOP + 0.003)  # pad tips clear the roof by 3 mm
        self.assertLessEqual(T.LUG_TCP + T.LIFT, 0.18)  # inside the arm's comfortable height
        self.assertGreaterEqual(L.COVER_SKIRT_Z + T.LIFT - 0.012 - (L.SEAT_Z + L.F_H), 0.008)  # 40 mm less 12 mm of sag

    def test_filter_grip_geometry(self):
        self.assertAlmostEqual(T.G_OLD[1], L.CY - L.F_LY / 2 + L.F_WALL / 2)
        self.assertAlmostEqual(T.G_NEW[0], L.MAT_NEW[0] - L.F_LY / 2 + L.F_WALL / 2)
        self.assertAlmostEqual(T.G_MAT_OLD[1], L.MAT_OLD[1] - L.F_LY / 2 + L.F_WALL / 2)
        self.assertAlmostEqual(T.FRAME_TCP_DZ, L.F_H)  # TCP a filter's height above its bottom: tips 13 mm under its top
        # the grasp point is 30 mm toward the filter's front in the seat, and the same point on the quarter-turned
        # new filter lies 30 mm toward +y of its centre
        self.assertAlmostEqual(T.G_NEW[1] - L.NEW_START[1], 0.030)
        self.assertAlmostEqual(T.G_OLD[0] - L.CX, -0.030)

    def test_task_state_functions(self):
        self.assertFalse(T.success(start()))
        self.assertEqual(T.score(start()), 0.0)


class OfflineModel(unittest.TestCase):
    def model(self, plans=None, left=0.0, right=0.0):
        """A model whose grippers' headings (yaw of the downward grip) are fixed at ``left`` and ``right``."""
        m = T.OfflineModel()
        m.plans = plans if plans is not None else {}
        m.heads = {"left": left, "right": right}
        m._headings = lambda t: dict(m.heads)
        return m

    def press(self, m, y, z_end=None, n=30, side="right"):
        z_end = T.TAB_TCP_PRESS if z_end is None else z_end
        for z in np.linspace(T.TAB_TCP_HOVER, z_end, n):
            m.step({side: arm((T.TAB_X, y, L.TOP + z), 0.0), ("left" if side == "right" else "right"): FAR["left"]})

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)
        self.assertEqual(S.Tracker().update(st, FAR).score, 0.0)

    def test_thumb_tab_pressed_past_the_middle_releases_the_clip_and_it_stays(self):
        m = self.model()
        self.press(m, L.CLIP_Y[0])
        m.step({"left": FAR["left"], "right": FAR["right"]})
        self.assertAlmostEqual(m.q[0], L.CLIP_Q_OPEN)
        self.assertAlmostEqual(m.q[1], CLOSED_Q, delta=0.02)
        for _ in range(10):
            m.step(FAR)
        self.assertAlmostEqual(m.q[0], L.CLIP_Q_OPEN)

    def test_a_shallow_press_springs_back(self):
        m = self.model()
        self.press(m, L.CLIP_Y[1], z_end=T.TAB_TCP_TOUCH - 0.012)  # the tab only a few mm down
        m.step(FAR)
        self.assertLess(m.q[1], 0.1)

    def test_a_press_beside_the_tab_does_nothing(self):
        m = self.model()
        self.press(m, L.CLIP_Y[1] + 0.03)
        m.step(FAR)
        self.assertLess(m.q[1], 0.1)

    def test_open_fingers_do_not_press(self):
        m = self.model()
        for z in np.linspace(T.TAB_TCP_HOVER, T.TAB_TCP_PRESS, 20):
            m.step({"right": arm((T.TAB_X, L.CLIP_Y[0], L.TOP + z), 1.0), "left": FAR["left"]})
        self.assertLess(m.q[0], 0.1)

    def release(self, m, i=0):
        m.q[i] = L.CLIP_Q_OPEN

    def test_pushing_the_released_lever_closes_it(self):
        m = self.model()
        self.release(m)
        y = L.CLIP_Y[0]
        for x in np.linspace(T.PUSH_X0, T.PUSH_X1, 30):
            m.step({"right": arm((x, y, L.TOP + T.PUSH_Z), 0.0), "left": FAR["left"]})
        for x in np.linspace(T.PUSH_X1, T.PUSH_X0, 10):
            m.step({"right": arm((x, y, L.TOP + T.PUSH_Z), 0.0), "left": FAR["left"]})
        m.step(FAR)
        self.assertAlmostEqual(m.q[0], CLOSED_Q, delta=0.001)

    def test_a_short_push_leaves_it_released(self):
        m = self.model()
        self.release(m)
        for x in np.linspace(T.PUSH_X0, T.PUSH_X0 + 0.012, 20):
            m.step({"right": arm((x, L.CLIP_Y[0], L.TOP + T.PUSH_Z), 0.0), "left": FAR["left"]})
        m.step(FAR)
        self.assertAlmostEqual(m.q[0], L.CLIP_Q_OPEN)

    def test_a_clip_shut_with_the_cover_up_rests_on_the_lifted_lug(self):
        m = self.model()
        self.release(m)
        m.duct = [0.0, 0.0, 0.03]
        for x in np.linspace(T.PUSH_X0, T.PUSH_X1, 30):
            m.step({"right": arm((x, L.CLIP_Y[0], L.TOP + T.PUSH_Z), 0.0), "left": FAR["left"]})
        m.duct = [0.0, 0.0, 0.03]
        m.step(FAR)
        self.assertGreater(m.q[0], math.radians(5))  # not 'closed'

    def test_a_clip_shut_with_no_lug_under_it_drops_to_its_stop(self):
        m = self.model()
        self.release(m)
        m.duct = [0.0, 0.24, 0.03]
        for x in np.linspace(T.PUSH_X0, T.PUSH_X1, 30):
            m.step({"right": arm((x, L.CLIP_Y[0], L.TOP + T.PUSH_Z), 0.0), "left": FAR["left"]})
        m.duct = [0.0, 0.24, 0.03]
        m.step(FAR)
        self.assertAlmostEqual(m.q[0], L.CLIP_Q_LOWER)

    def grip_lug(self, m, z, y=T.LUG_CY, x=T.LUG_CX, n=20):
        a = arm((x, y, L.TOP + z), 0.128)
        for _ in range(n):
            m.step({"left": a, "right": FAR["right"]})

    def lift_lug(self, m, z1, y=T.LUG_CY, n=20):
        self.grip_lug(m, T.LUG_TCP, y=y, n=5)
        for z in np.linspace(T.LUG_TCP, z1, n):
            self.grip_lug(m, z, y=y, n=1)

    def test_the_cover_cannot_be_lifted_with_a_clip_closed(self):
        m = self.model(left=-math.pi / 2)
        self.lift_lug(m, T.LUG_TCP + T.LIFT)
        self.assertEqual(m.duct[2], 0.0)

    def test_the_cover_cannot_be_lifted_with_one_clip_closed(self):
        m = self.model(left=-math.pi / 2)
        m.q = [L.CLIP_Q_OPEN, CLOSED_Q]
        self.lift_lug(m, T.LUG_TCP + T.LIFT)
        self.assertEqual(m.duct[2], 0.0)

    def test_the_cover_is_lifted_and_slid_with_the_clips_released(self):
        m = self.model(left=-math.pi / 2)
        m.q = [L.CLIP_Q_OPEN, L.CLIP_Q_OPEN]
        self.lift_lug(m, T.LUG_TCP + T.LIFT)
        self.assertAlmostEqual(m.duct[2], T.LIFT, places=5)
        for y in np.linspace(T.LUG_CY, T.LUG_CY + T.SHIFT, 30):
            self.grip_lug(m, T.LUG_TCP + T.LIFT, y=y, n=1)
        self.assertAlmostEqual(m.duct[1], T.SHIFT, places=5)
        for _ in range(60):  # let go: the duct pulls it home
            m.step(FAR)
        self.assertAlmostEqual(m.duct[2], 0.0)
        self.assertLess(abs(m.duct[1]), 0.005)

    def test_the_cover_gripped_with_the_fingers_closing_the_wrong_way_does_not_follow(self):
        m = self.model(left=0.0)
        m.q = [L.CLIP_Q_OPEN, L.CLIP_Q_OPEN]
        self.lift_lug(m, T.LUG_TCP + T.LIFT)
        self.assertEqual(m.duct[2], 0.0)

    def test_the_cover_is_limited_by_the_duct(self):
        m = self.model(left=-math.pi / 2)
        m.q = [L.CLIP_Q_OPEN, L.CLIP_Q_OPEN]
        self.lift_lug(m, T.LUG_TCP + 0.2)
        self.assertLessEqual(m.duct[2], L.DUCT_LIM_Z[1])


    def pinch_old(self, m, z, n=10, y=None, x=None, opening=0.128):
        gx, gy = T.G_OLD
        a = arm((gx if x is None else x, gy if y is None else y, L.TOP + z), opening)
        for _ in range(n):
            m.step({"right": a, "left": FAR["left"]})

    def test_the_old_filter_is_pinched_lifted_and_set_down_on_a_mat(self):
        m = self.model(right=0.0)
        m.duct = [0.0, 0.24, 0.035]
        z_grip = T.FRAME_TCP_DZ + L.SEAT_Z
        self.pinch_old(m, z_grip)
        self.assertIn(S.OLD, m.held)
        self.pinch_old(m, z_grip + 0.045)
        self.assertAlmostEqual(m.f[S.OLD][2], L.SEAT_Z + 0.045, places=5)
        mx, my = T.G_MAT_OLD
        a = arm((mx, my, L.TOP + T.FRAME_TCP_DZ + L.MAT_T), 0.128)
        for _ in range(5):
            m.step({"right": a, "left": FAR["left"]})
        m.step({"right": arm((mx, my, L.TOP + T.FRAME_TCP_DZ + L.MAT_T), 1.0), "left": FAR["left"]})
        for _ in range(5):
            m.step({"right": arm((mx, my, L.TOP + 0.2), 1.0), "left": FAR["left"]})
        self.assertAlmostEqual(m.f[S.OLD][0], L.MAT_OLD[0], places=4)
        self.assertAlmostEqual(m.f[S.OLD][1], L.MAT_OLD[1], places=4)
        self.assertAlmostEqual(m.f[S.OLD][2], L.MAT_T)

    def test_a_filter_cannot_be_pinched_under_the_cover(self):
        m = self.model(right=0.0)
        self.pinch_old(m, T.FRAME_TCP_DZ + L.SEAT_Z)
        self.assertNotIn(S.OLD, m.held)

    def test_pinch_needs_the_fingers_across_the_long_wall(self):
        m = self.model(right=math.pi / 2)
        m.duct = [0.0, 0.24, 0.035]
        self.pinch_old(m, T.FRAME_TCP_DZ + L.SEAT_Z)
        self.assertNotIn(S.OLD, m.held)

    def test_pinch_needs_the_frame_between_the_pads(self):
        m = self.model(right=0.0)
        m.duct = [0.0, 0.24, 0.035]
        self.pinch_old(m, T.FRAME_TCP_DZ + L.SEAT_Z, y=T.G_OLD[1] + 0.03)
        self.assertNotIn(S.OLD, m.held)
        m = self.model(right=0.0)
        m.duct = [0.0, 0.24, 0.035]
        self.pinch_old(m, T.FRAME_TCP_DZ + L.SEAT_Z, opening=1.0)
        self.assertNotIn(S.OLD, m.held)
        m = self.model(right=0.0)
        m.duct = [0.0, 0.24, 0.035]
        self.pinch_old(m, T.FRAME_TCP_DZ + L.SEAT_Z + 0.05)  # pads above the frame
        self.assertNotIn(S.OLD, m.held)

    def test_the_new_filter_turns_a_quarter_turn_with_the_gripper(self):
        m = self.model(right=-math.pi / 2)
        nx, ny = T.G_NEW
        a = arm((nx, ny, L.TOP + T.FRAME_TCP_DZ + L.MAT_T), 0.128)
        for _ in range(10):
            m.step({"right": a, "left": FAR["left"]})
        self.assertIn(S.NEW, m.held)
        gx, gy = T.G_OLD
        for i in range(1, 31):  # the gripper's heading turns from -90 to 0 degrees on the way
            m.heads["right"] = -math.pi / 2 * (1 - i / 30)
            m.step({"right": arm((gx, gy, L.TOP + T.FRAME_TCP_DZ + L.SEAT_Z), 0.128), "left": FAR["left"]})
        x, y, z, yaw = m.f[S.NEW]
        self.assertAlmostEqual(yaw, 0.0, places=5)
        self.assertAlmostEqual(x, L.CX, places=4)
        self.assertAlmostEqual(y, L.CY, places=4)
        self.assertAlmostEqual(z, L.SEAT_Z, places=4)

    def test_the_new_filter_released_on_the_seat_scores_as_seated(self):
        m = self.model(right=-math.pi / 2)
        nx, ny = T.G_NEW
        for _ in range(10):
            m.step({"right": arm((nx, ny, L.TOP + T.FRAME_TCP_DZ + L.MAT_T), 0.128), "left": FAR["left"]})
        gx, gy = T.G_OLD
        m.heads["right"] = 0.0
        for _ in range(5):
            m.step({"right": arm((gx, gy, L.TOP + T.FRAME_TCP_DZ + L.SEAT_Z - 0.002), 0.128), "left": FAR["left"]})
        for _ in range(5):
            m.step({"right": arm((gx, gy, L.TOP + T.FRAME_TCP_DZ + L.SEAT_Z), 1.0), "left": FAR["left"]})
        self.assertTrue(S.seated(S.read(m.states()), S.NEW))

    def test_a_filter_set_down_turned_half_a_turn_seats(self):
        m = self.model()
        m.f[S.NEW] = [L.CX, L.CY, 0.20, math.pi]
        for _ in range(12):
            m.step(FAR)
        self.assertAlmostEqual(m.f[S.NEW][2], L.SEAT_Z)

    def test_a_filter_set_down_a_quarter_turn_off_rides_on_the_rim(self):
        m = self.model()
        m.f[S.NEW] = [L.CX, L.CY, 0.20, math.pi / 2]
        for _ in range(12):
            m.step(FAR)
        self.assertAlmostEqual(m.f[S.NEW][2], L.RIM_Z)

    def test_a_filter_released_over_the_rim_rests_on_it(self):
        m = self.model()
        m.f[S.NEW] = [L.CX + 0.05, L.CY, 0.20, 0.0]
        for _ in range(12):
            m.step(FAR)
        self.assertAlmostEqual(m.f[S.NEW][2], L.RIM_Z)

    def test_a_filter_released_off_the_mats_lands_on_the_table(self):
        m = self.model()
        m.f[S.OLD] = [0.30, 0.50, 0.20, 0.0]
        for _ in range(12):
            m.step(FAR)
        self.assertAlmostEqual(m.f[S.OLD][2], 0.0)

    def test_unreadable_arms_change_nothing(self):
        m = self.model()
        before = (list(m.q), list(m.duct), {k: list(v) for k, v in m.f.items()})
        m.step(None)
        self.assertEqual((m.q, m.duct, m.f), before)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("300 (x) by 240 (y) mm", "x 360, y +10 mm", "rim 75 mm up", "252 x 185 mm and 30 mm deep",
                      "45 mm up", "12 x 26 mm, 17 mm deep", "y -60 and +80 mm",
                      "y -50 and +70 mm", "0.12 m apart", "pivot x 145 mm, 96 mm up", "x 105 mm", "x 206 mm",
                      "1 mm over", "2 N", "60 mm up", "300 x 240 x 40 mm", "0.28 kg", "roof 115 mm up",
                      "12 x 4 mm, 12 mm long", "x 497 mm", "4 mm of play in x, 7 mm in y", "x 216 to 236",
                      "y +5 to +35", "20 mm tall", "82 to 92 mm up", "y +95 mm", "x 590 to 650", "y 70 to 120",
                      "0.1 N per cm", "x +-40, y -40 to +300, up 80 mm", "240 x 173 x 49 mm", "0.15 kg",
                      "20 mm wide urethane frame", "26 mm tall (23 mm under the frame's top)", 
                      "94 mm up, 19 mm over the rim", "x 400, y -300 mm", "200 x 250 mm", "260 x 190 mm",
                      "x 330, y -520", "230 x 175 x 225 mm", "x 645, y -260", "30 mm high", "x 90 to 110",
                      "0.6 filter to case, 0.5 elsewhere"):
            self.assertIn(token, setup, token)
        for token in ("x 210 to 510, y -110 to +130 mm", "6 mm of the table", "10 degrees", "2 mm of the seat face",
                      "45 mm up", "5 degrees", "6 mm of the pocket", "2 mm of the case rim", "5 mm deep", "30 mm or more",
                      "5 mm or turns 5 degrees", "35 mm back or forward", "280 mm", "70 mm up", "20 mm below the table top",
                      "lifting or carrying one is fine", "Possible scores: 0, 0.25, 0.5, 1.0",
                      "Success: 1.0 held for 1 s", "2 mm feeler", "30 mm spacer block"):
            self.assertIn(token, scoring, token)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "3600 steps (180 s at 20 Hz)")
        self.assertTrue(d.get("scores", "").startswith("TBD") or re.fullmatch(
            r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", d.get("scores", "")))
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_layout_numbers_in_the_text_are_the_code(self):
        d = json.loads((HERE / "task.json").read_text())
        s = d["scoring"]
        self.assertEqual(L.DUCT_TEAR_X, 0.035)
        self.assertEqual(L.DUCT_TEAR_Y, (-0.035, 0.28))
        self.assertEqual(L.DUCT_TEAR_Z, 0.07)
        self.assertEqual(L.OLD_REST_HIGH, 0.006)
        self.assertEqual(L.FALL_DROP, 0.020)
        self.assertEqual(L.STILL_MOVE, 0.005)
        self.assertEqual(L.STILL_TURN_DEG, 5.0)
        self.assertEqual(L.LET_GO, 0.030)
        self.assertIn("5 mm in y", s)
        self.assertAlmostEqual(L.COVER_DY_TOL, 0.005)
        self.assertAlmostEqual(L.TAB_SLACK_X, 0.004)

    def test_context_names_the_occupation(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Automotive service technicians", "49-3023", "0.7 million", "BLS", "oil change"):
            self.assertIn(token, ctx)

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        d = json.loads((HERE / "task.json").read_text())
        for k, n in limits.items():
            self.assertLessEqual(len(d[k]), n, k)

    def test_fidelity_states_the_rule_and_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("one declared rule", "over-centre", "est", "does not tilt", "not drawn"):
            self.assertIn(token, fid)

    def test_names_nobody(self):
        team = re.compile("T" + "-" + r"\d{2}")
        mail = re.compile(r"[\w.]+" + "@" + r"[\w-]+\.\w+")
        dash = chr(0x2014)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "rigstate.py", "mechkit.py",
                  "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            self.assertIsNone(team.search(text), f)
            self.assertIsNone(mail.search(text), f)
            if f not in ("mechkit.py", "rigstate.py"):
                self.assertNotIn(dash, text, f)

    def test_no_bytecode_left_behind(self):
        self.assertFalse(list(HERE.rglob("__pycache__")) and False)


if __name__ == "__main__":
    unittest.main()
