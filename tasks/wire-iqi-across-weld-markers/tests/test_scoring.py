"""Offline tests for scoring.py and task.py: synthetic part states and gripper snapshots, no simulator.
python3 -m unittest discover -s tests -v"""

import importlib
import json
import math
import os
import pathlib
import random
import re
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402
from scoring import (  # noqa: E402
    Tracker,
    bottom_gap,
    footprint,
    marker_tick,
    on_plate,
    readable_sides,
    released_now,
    tilt_deg,
    valid,
    wire_err_deg,
    zone_gap,
    zone_radius,
)

HZ = 20.0
W = int(round(L.STILL_WINDOW_S * HZ))  # 10 steps
FULL = 2 * W + 1  # steps for 'still for 1 s'
PT = L.PLATE_TOP_W
H = {"iqi": L.IQI_H + L.WIRE_T, "marker_0": L.MK_H + L.FIG_T, "marker_1": L.MK_H + L.FIG_T}


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


def raw(origin, q, vel=(0.0, 0.0, 0.0)):
    """A part's state as mechkit reports it (only what the scorer reads)."""
    o = np.asarray(origin, dtype=float)
    return SimpleNamespace(origin=o, quat_xyzw=np.asarray(q, dtype=float), pos=o.copy(),
                           lin_vel=np.asarray(vel, dtype=float), ang_vel=np.zeros(3), low=o - 0.03, high=o + 0.03)


def flat_at(x, y, yaw=0.0, bottom=PT, upside_down=False, name="iqi"):
    """A part lying flat, centred at (x, y) seen from above, its lowest point at ``bottom`` (world z)."""
    if upside_down:
        return raw((x, y, bottom + H[name]), quat(yaw, 0.0, 180.0))
    return raw((x, y, bottom), quat(yaw))


def tilted(x, y, yaw, pitch, name="iqi", bottom=PT):
    """Turned ``pitch`` about its local y, its lowest corner at ``bottom``."""
    s = raw((x, y, bottom), quat(yaw, pitch))
    return raw((x, y, bottom - bottom_gap(s, name) - PT + bottom), quat(yaw, pitch))


def iqi_ok(dx=0.0, y=0.0, yaw=0.0):
    return flat_at(L.WELD_X + dx, y, yaw)


def marker_at(tick, dx=-0.028, dy=0.0, bottom=PT):
    return flat_at(L.WELD_X + dx, L.TICK_Y[tick] + dy, bottom=bottom, name=f"marker_{tick}")


def start(name):
    x, y, yaw = L.IQI_START if name == "iqi" else L.MK_START[name]
    return flat_at(x, y, yaw, bottom=L.TOP, name=name)


def layout(iqi=None, m0=None, m1=None):
    return {"iqi": iqi if iqi is not None else start("iqi"),
            "marker_0": m0 if m0 is not None else start("marker_0"),
            "marker_1": m1 if m1 is not None else start("marker_1")}


def arm(tcp, opening=40.0 / 93.9):
    tcp = np.asarray(tcp, dtype=float)
    return SimpleNamespace(tcp=tcp, tips=tcp - np.array([0.0, 0.0, 0.013]), opening=opening)


FAR = {"left": arm((0.20, 0.30, L.TOP + 0.15)), "right": arm((0.20, -0.30, L.TOP + 0.15))}
DONE = layout(iqi_ok(), marker_at(0), marker_at(1))


def feed(tr, states, arms=FAR):
    r = None
    for st in states:
        r = tr.update(st, arms)
    return r


def settled(st, arms=FAR, n=FULL):
    tr = Tracker(HZ)
    return tr, feed(tr, [st] * n, arms)


def scored(st, arms=FAR, n=FULL):
    return settled(st, arms, n)[1].score


class Geometry(unittest.TestCase):
    def test_start_layout(self):
        st = layout()
        self.assertTrue(Tracker.at_start(st))
        self.assertEqual(scored(st), 0.0)
        for n in L.NAMES:
            self.assertFalse(on_plate(st[n], n))
            self.assertAlmostEqual(bottom_gap(st[n], n), -L.PLATE_T)
        self.assertAlmostEqual(wire_err_deg(st["iqi"]), 90.0)  # starts wires along the weld

    def test_layout_numbers(self):
        self.assertAlmostEqual(L.WELD_X0, 0.353)
        self.assertAlmostEqual(L.WELD_X1, 0.367)
        self.assertEqual(L.TICK_Y, (-0.080, 0.080))
        self.assertAlmostEqual(L.WELD_X - L.TICK_LEN / 2, 0.348)
        # every part starts on the bench, its footprint clear of the plate
        for n in L.NAMES:
            fp = footprint(start(n), n)
            self.assertLess(fp[:, 0].max(), L.PLATE_X0 - 0.020)
        # the wires: six, increasing in width, the thickest 3 mm, all inside the holder
        self.assertEqual(list(L.WIRE_WIDTHS), sorted(L.WIRE_WIDTHS))
        self.assertGreaterEqual(max(L.WIRE_WIDTHS), 0.002)

    def test_measures_at_the_target(self):
        s = iqi_ok(0.002, 0.0019, -1.2)
        self.assertAlmostEqual(wire_err_deg(s), 1.2, places=6)
        self.assertTrue(on_plate(s, "iqi"))
        fp = footprint(s, "iqi")
        self.assertLess(np.abs(fp[:, 1]).max(), 0.0175)
        self.assertEqual(marker_tick(marker_at(0), "marker_0"), 0)
        self.assertEqual(marker_tick(marker_at(1), "marker_1"), 1)
        self.assertEqual(marker_tick(marker_at(1), "marker_0"), 1)  # either marker at either tick

    def test_wires_either_way_round_and_either_face_up(self):
        for s in (iqi_ok(yaw=180.0), iqi_ok(yaw=-179.0), flat_at(L.WELD_X, 0.0, 0.0, upside_down=True),
                  flat_at(L.WELD_X, 0.0, 180.0, upside_down=True)):
            self.assertLess(wire_err_deg(s), 1.0 + 1e-9)
            self.assertTrue(on_plate(s, "iqi"))
            self.assertEqual(scored(layout(s, marker_at(0), marker_at(1))), 1.0)

    def test_finger_zone(self):
        a = arm((0.3, 0.0, 0.9), 1.0)
        self.assertAlmostEqual(zone_radius(a), math.hypot(0.0939 / 2 + 0.008, 0.0135))
        self.assertAlmostEqual(zone_radius(arm((0, 0, 0), 0.0)), math.hypot(0.008, 0.0135))
        # an open gripper whose pad tips are h above the IQI's top, straight above it: the zone starts at the tips
        # the zone runs from the pad tips up: straight above a part, the gap is the tips' height over its top
        top = PT + H["iqi"]
        for h in (0.005, 0.015, 0.040):
            g = zone_gap(arm((L.WELD_X, 0.0, top + h + 0.013), 44.0 / 93.9), iqi_ok(), "iqi")
            self.assertAlmostEqual(g, h, places=6)
        # beside it at the part's height: the gap is lateral, to the pads' outer corners
        op = 44.0 / 93.9
        r = zone_radius(arm((0, 0, 0), op))
        g = zone_gap(arm((L.WELD_X, -L.IQI_W / 2 - r - 0.012, PT + 0.016), op), iqi_ok(), "iqi")
        self.assertAlmostEqual(g, 0.012, delta=0.0005)
        # closed on it, as the demo holds it: overlapping
        self.assertLess(zone_gap(arm((L.WELD_X, 0.0, PT + 0.017), 20.0 / 93.9), iqi_ok(), "iqi"), 0.0)


class Tiers(unittest.TestCase):
    def test_complete_scores_one(self):
        self.assertEqual(scored(DONE), 1.0)
        r = settled(DONE)[1]
        self.assertTrue(r.success)
        self.assertEqual(r.details["ticks_marked"], [0, 1])

    def test_demo_end_pose_scores_one(self):
        """Where the demo lays them (trimmed 3 mm outward), and where they would land 4 mm short instead."""
        t = _task()
        for short in (0.0, 0.004, 0.007):
            parts = {}
            for n in L.NAMES:
                px, py = t.place_point(n)
                bx, by = L.BASES[t.ARM_OF[n]]
                r = math.hypot(px - bx, py - by)
                x, y = px - short * (px - bx) / r, py - short * (py - by) / r
                parts[n] = flat_at(x, y, math.degrees(t.IQI_YAW_SET) if n == "iqi" else 0.0, name=n)
            self.assertEqual(scored(parts), 1.0, short)
        # with the GPU's +0.6 .. +1.7 degree grip bias the IQI ends -0.6 .. +0.5 degrees off x
        for bias in (0.6, 1.7, 6.0):
            parts = dict(DONE)
            parts["iqi"] = iqi_ok(yaw=math.degrees(t.IQI_YAW_SET) + bias)
            self.assertEqual(scored(parts), 1.0)

    def test_markers_on_the_weld_or_beyond_it_count(self):
        on_weld = layout(iqi_ok(), marker_at(0, dx=0.0, bottom=PT + L.TICK_T), marker_at(1, dx=0.0, bottom=PT + L.TICK_T))
        self.assertEqual(scored(on_weld), 1.0)
        far_side = layout(iqi_ok(), marker_at(0, dx=0.030), marker_at(1, dx=0.030))
        self.assertEqual(scored(far_side), 1.0)
        swapped = {"iqi": iqi_ok(), "marker_0": marker_at(1), "marker_1": marker_at(0)}
        self.assertEqual(scored(swapped), 1.0)

    def test_three_quarters(self):
        self.assertEqual(scored(layout(iqi_ok(), marker_at(0), None)), 0.75)  # one marker left on the bench
        self.assertEqual(scored(layout(iqi_ok(), None, marker_at(1))), 0.75)
        both_one_tick = layout(iqi_ok(), marker_at(1, dx=-0.028), marker_at(1, dx=0.028))
        self.assertEqual(scored(both_one_tick), 0.75)
        tr, r = settled(DONE, n=W + 1)  # counted, but not yet still for 1 s
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        self.assertEqual(r.details["tier_reason"], "not_still_for_1_s")

    def test_half(self):
        self.assertEqual(scored(layout(iqi_ok())), 0.5)  # both markers left on the bench
        beside = layout(iqi_ok(), flat_at(L.WELD_X - 0.028, -0.035, name="marker_0"),
                        flat_at(L.WELD_X - 0.028, 0.035, name="marker_1"))
        self.assertEqual(scored(beside), 0.5)  # 45 mm from the ticks

    def test_quarter(self):
        markers = (marker_at(0), marker_at(1))
        along = flat_at(L.WELD_X, 0.0, 90.0)  # wires parallel to the weld
        off_weld = iqi_ok(dx=0.030)  # on the far plate, not over the weld
        past_tick = iqi_ok(y=0.070)  # footprint reaches y 0.084
        diagonal = iqi_ok(yaw=45.0)
        for s in (along, off_weld, past_tick, diagonal):
            self.assertEqual(scored(layout(s, *markers)), 0.25)  # markers do not count without a correct IQI
            self.assertEqual(scored(layout(s)), 0.25)
        self.assertEqual(settled(layout(along, *markers))[1].details["tier_reason"], "iqi_on_plate_but_wrong")

    def test_on_weld_band_means_every_wire_crosses_the_weld(self):
        """At the band's worst case (centre ON_WELD off, wires WIRE_TOL_DEG skewed) every wire still spans the
        whole weld cap, with more than 1 mm to spare; 30 mm off, the wires no longer cross it."""
        def spare(d, deg):
            th = math.radians(deg)
            out = []
            for wy in L.WIRE_Y:
                xs = [d + ex * math.cos(th) - wy * math.sin(th) for ex in (-L.WIRE_LEN / 2, L.WIRE_LEN / 2)]
                out.append(min(-L.WELD_W / 2 - min(xs), max(xs) - L.WELD_W / 2))
            return min(out)
        for d in (L.ON_WELD, -L.ON_WELD):
            for deg in (L.WIRE_TOL_DEG, -L.WIRE_TOL_DEG):
                self.assertGreater(spare(d, deg), 0.001, (d, deg))
        self.assertLess(spare(0.030, 0.0), 0.0)

    def test_band_edges(self):
        e = 1e-4
        for yaw, want in ((L.WIRE_TOL_DEG - 0.01, 1.0), (L.WIRE_TOL_DEG + 0.01, 0.25),
                          (-L.WIRE_TOL_DEG + 0.01, 1.0), (180 - L.WIRE_TOL_DEG - 0.01, 0.25)):
            self.assertEqual(scored(layout(iqi_ok(yaw=yaw), marker_at(0), marker_at(1))), want, yaw)
        for dx, want in ((L.ON_WELD - e, 1.0), (L.ON_WELD + e, 0.25), (-L.ON_WELD + e, 1.0), (-L.ON_WELD - e, 0.25)):
            self.assertEqual(scored(layout(iqi_ok(dx=dx), marker_at(0), marker_at(1))), want, dx)
        y_edge = L.SHOT_HALF - L.IQI_W / 2  # the IQI's side on the tick's centre line
        for y, want in ((y_edge - e, 0.5), (y_edge + e, 0.25), (-y_edge + e, 0.5), (-y_edge - e, 0.25)):
            self.assertEqual(scored(layout(iqi_ok(y=y))), want, y)
        for dy, want in ((L.MARK_ALONG - e, 1.0), (L.MARK_ALONG + e, 0.75), (-L.MARK_ALONG + e, 1.0)):
            self.assertEqual(scored(layout(iqi_ok(), marker_at(0, dy=dy), marker_at(1))), want, dy)
        for dx, want in ((-L.MARK_ACROSS + e, 1.0), (-L.MARK_ACROSS - e, 0.75), (L.MARK_ACROSS - e, 1.0),
                         (L.MARK_ACROSS + e, 0.75)):
            self.assertEqual(scored(layout(iqi_ok(), marker_at(0, dx=dx), marker_at(1))), want, dx)
        for up, want in ((L.SEAT_TOL - e, 1.0), (L.SEAT_TOL + e, 0.75)):
            self.assertEqual(scored(layout(iqi_ok(), marker_at(0, bottom=PT + up), marker_at(1))), want, up)
        for up, want in ((L.SEAT_TOL - e, 0.5), (L.SEAT_TOL + e, 0.0)):
            self.assertEqual(scored(layout(flat_at(L.WELD_X, 0.0, 0.0, bottom=PT + up))), want, up)
        for p, want in ((L.FLAT_DEG - 0.05, 0.5), (L.FLAT_DEG + 0.05, 0.0)):
            self.assertEqual(scored(layout(tilted(L.WELD_X, 0.0, 0.0, p))), want, p)
        for p, want in ((L.FLAT_DEG - 0.05, 1.0), (L.FLAT_DEG + 0.05, 0.75)):
            m = tilted(L.WELD_X - 0.028, L.TICK_Y[0], 0.0, p, name="marker_0")
            self.assertEqual(scored(layout(iqi_ok(), m, marker_at(1))), want, p)

    def test_every_tier_is_reachable_and_each_state_has_one_score(self):
        cases = {1.0: DONE, 0.75: layout(iqi_ok(), marker_at(0)), 0.5: layout(iqi_ok()),
                 0.25: layout(flat_at(L.WELD_X, 0.0, 90.0)), 0.0: layout()}
        for want, st in cases.items():
            r = settled(st)[1]
            self.assertEqual(r.score, want)
            self.assertEqual(r.success, want == 1.0)
        self.assertEqual(sorted(cases), [0.0, 0.25, 0.5, 0.75, 1.0])

    def test_score_is_idempotent(self):
        tr, r = settled(DONE)
        n = tr.steps
        for _ in range(3):
            self.assertEqual(tr.result(DONE, FAR).score, 1.0)
        self.assertEqual(tr.steps, n)


class ZeroConditions(unittest.TestCase):
    def test_nothing_moved(self):
        r = settled(layout())[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["tier_reason"], "iqi_not_on_plate")

    def test_iqi_on_the_table_beside_the_plate(self):  # the film side, not the part
        for x, y in ((0.262, 0.0), (0.460, 0.0), (L.WELD_X, -0.12)):
            s = flat_at(x, y, 0.0, bottom=L.TOP)
            self.assertEqual(scored(layout(s, marker_at(0), marker_at(1))), 0.0)

    def test_iqi_on_its_side_or_end_on_the_weld(self):
        side = raw((L.WELD_X, 0.0, PT + L.IQI_W / 2), quat(0.0, 0.0, 90.0))
        end = raw((L.WELD_X, 0.0, PT + L.IQI_L / 2), quat(0.0, 90.0, 0.0))
        for s in (side, end):
            self.assertAlmostEqual(abs(bottom_gap(s, "iqi")), 0.0, delta=0.001)
            self.assertEqual(scored(layout(s, marker_at(0), marker_at(1))), 0.0)

    def test_iqi_on_a_marker(self):
        m = flat_at(L.WELD_X, 0.0, name="marker_0")
        s = flat_at(L.WELD_X, 0.0, 0.0, bottom=PT + H["marker_0"])
        self.assertEqual(scored(layout(s, m)), 0.0)

    def test_iqi_propped_at_10_degrees(self):
        self.assertEqual(scored(layout(tilted(L.WELD_X, 0.0, 0.0, 10.0), marker_at(0), marker_at(1))), 0.0)

    def test_iqi_still_held(self):
        held = {"left": FAR["left"], "right": arm((L.WELD_X, 0.0, PT + 0.001 + 0.016), 20.0 / 93.9)}
        r = settled(DONE, held)[1]
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["tier_reason"], "iqi_held")

    def test_iqi_still_moving(self):
        tr = Tracker(HZ)
        sts = [layout(iqi_ok(dx=-0.006 + 0.0006 * i), marker_at(0), marker_at(1)) for i in range(FULL)]
        r = feed(tr, sts)  # 0.6 mm per step: 6 mm in 0.5 s
        self.assertEqual(r.score, 0.0)
        self.assertEqual(r.details["tier_reason"], "iqi_not_settled")


class Released(unittest.TestCase):
    def test_hands_off_is_10_mm(self):
        top = PT + H["iqi"]
        op = 44.0 / 93.9
        for h, want in ((0.009, 0.0), (0.011, 1.0)):
            arms = {"left": FAR["left"], "right": arm((L.WELD_X, 0.0, top + h + 0.013), op)}
            self.assertEqual(scored(DONE, arms), want, h)

    def test_left_gripper_counts_too(self):
        arms = {"left": arm((L.WELD_X - 0.028, L.TICK_Y[1], PT + 0.001 + 0.016), 14.0 / 93.9),
                "right": FAR["right"]}
        r = settled(DONE, arms)[1]
        self.assertEqual(r.score, 0.75)  # the left marker is still held: shortcut 5
        self.assertEqual(r.details["ticks_marked"], [0])

    def test_released_for_the_whole_half_second(self):
        tr = Tracker(HZ)
        held = {"left": FAR["left"], "right": arm((L.WELD_X, 0.0, PT + 0.017), 20.0 / 93.9)}
        feed(tr, [DONE] * FULL, held)
        r = feed(tr, [DONE] * W)  # released for 10 steps: the 0.5 s window (11 observations) still holds one held
        self.assertEqual(r.score, 0.0)
        r = feed(tr, [DONE] * 1)
        self.assertEqual(r.score, 0.75)  # counted; still for 1 s needs 10 steps more
        r = feed(tr, [DONE] * W)
        self.assertEqual(r.score, 1.0)

    def test_unreadable_arms_are_skipped_and_counted(self):
        tr, r = settled(DONE, None)
        self.assertEqual(r.score, 1.0)
        self.assertEqual(tr.arms_unavailable, FULL)
        nan = arm((math.nan, 0.0, 0.9))
        held = arm((L.WELD_X, 0.0, PT + 0.017), 20.0 / 93.9)
        self.assertEqual(readable_sides({"left": nan, "right": held}), ("right",))
        self.assertEqual(scored(DONE, {"left": nan, "right": held}), 0.0)  # the readable one is still checked
        self.assertEqual(scored(DONE, {"left": nan, "right": FAR["right"]}), 1.0)
        self.assertTrue(released_now(DONE["iqi"], "iqi", {"left": nan}))


class Stillness(unittest.TestCase):
    def test_needs_half_a_second_of_history(self):
        tr = Tracker(HZ)
        r = feed(tr, [DONE] * W)
        self.assertEqual(r.score, 0.0)

    def test_path_not_reported_velocity(self):
        chatter = layout(raw(iqi_ok().origin, iqi_ok().quat_xyzw, vel=(0.3, -0.2, 0.1)), marker_at(0), marker_at(1))
        self.assertEqual(scored(chatter), 1.0)  # reported velocity chatters; the path is what counts

    def test_slow_creep_under_2_mm_counts(self):
        tr = Tracker(HZ)
        sts = [layout(iqi_ok(dx=0.00015 * i), marker_at(0), marker_at(1)) for i in range(FULL)]
        self.assertEqual(feed(tr, sts).score, 1.0)  # 1.5 mm per 0.5 s

    def test_turning_more_than_a_degree_is_not_settled(self):
        tr = Tracker(HZ)
        sts = [layout(iqi_ok(yaw=0.15 * i), marker_at(0), marker_at(1)) for i in range(FULL)]
        self.assertEqual(feed(tr, sts).score, 0.0)
        tr = Tracker(HZ)
        sts = [layout(iqi_ok(), marker_at(0), flat_at(L.WELD_X - 0.028, 0.08, 0.15 * i, name="marker_1"))
               for i in range(FULL)]
        self.assertEqual(feed(tr, sts).score, 0.75)

    def test_one_second_needs_both_windows(self):
        tr = Tracker(HZ)
        moving = [layout(iqi_ok(), marker_at(0), marker_at(1, dx=-0.040 + 0.0004 * i)) for i in range(W)]
        feed(tr, moving)
        r = feed(tr, [DONE] * (W + 1))
        self.assertEqual(r.score, 0.75)
        r = feed(tr, [DONE] * W)
        self.assertEqual(r.score, 1.0)


class Exploits(unittest.TestCase):
    def test_iqi_carried_straight_and_laid_along_the_weld(self):  # shortcut 1
        self.assertEqual(scored(layout(flat_at(L.WELD_X, 0.0, 90.0), marker_at(0), marker_at(1))), 0.25)

    def test_markers_beside_the_iqi_or_on_the_stripe(self):  # shortcut 2
        beside = layout(iqi_ok(), flat_at(L.WELD_X - 0.028, -0.035, name="marker_0"),
                        flat_at(L.WELD_X - 0.028, 0.035, name="marker_1"))
        self.assertEqual(scored(beside), 0.5)
        stripe = layout(iqi_ok(), marker_at(0, dx=0.0, bottom=PT + L.TICK_T), marker_at(1, dx=0.0, bottom=PT + L.TICK_T))
        self.assertEqual(scored(stripe), 1.0)  # allowed: no code text keeps a marker off the weld

    def test_both_markers_at_one_tick(self):  # shortcut 3
        self.assertEqual(scored(layout(iqi_ok(), marker_at(0, dx=-0.028), marker_at(0, dx=0.028))), 0.75)

    def test_iqi_on_the_table_film_side(self):  # shortcut 4
        self.assertEqual(scored(layout(flat_at(0.262, 0.0, 0.0, bottom=L.TOP), marker_at(0), marker_at(1))), 0.0)

    def test_marker_on_the_table_at_a_tick_is_not_on_the_part(self):
        m = flat_at(0.268, L.TICK_Y[0], name="marker_0", bottom=L.TOP)  # beside the plate, at the tick's y
        self.assertEqual(scored(layout(iqi_ok(), m, marker_at(1))), 0.75)
        m = flat_at(0.268, L.TICK_Y[0], name="marker_0", bottom=PT)  # (as if) floating at the plate's height
        self.assertEqual(scored(layout(iqi_ok(), m, marker_at(1))), 0.75)  # its centre is not over the plate

    def test_marker_stacked_on_the_iqi_or_on_the_other_marker(self):
        on_iqi = flat_at(L.WELD_X, 0.0, name="marker_0", bottom=PT + H["iqi"])
        self.assertEqual(scored(layout(iqi_ok(), on_iqi, marker_at(1))), 0.75)
        stacked = flat_at(L.WELD_X - 0.028, L.TICK_Y[1], name="marker_0", bottom=PT + H["marker_1"])
        self.assertEqual(scored(layout(iqi_ok(), stacked, marker_at(1))), 0.75)

    def test_markers_far_from_the_weld(self):
        far = layout(iqi_ok(), marker_at(0, dx=0.070), marker_at(1, dx=-0.070))  # at the plate's edges
        self.assertEqual(scored(far), 0.5)

    def test_iqi_on_end_over_the_weld_with_markers(self):
        end = raw((L.WELD_X, 0.0, PT + L.IQI_L / 2), quat(0.0, 90.0, 0.0))
        self.assertEqual(scored(layout(end, marker_at(0), marker_at(1))), 0.0)

    def test_full_demo_last_marker_held(self):  # shortcut 5
        arms = {"left": FAR["left"], "right": arm((L.WELD_X - 0.028, L.TICK_Y[0], PT + 0.017), 14.0 / 93.9)}
        self.assertEqual(scored(DONE, arms), 0.75)


class Robustness(unittest.TestCase):
    def test_blow_up_states_score_zero_and_leave_the_tracker_alone(self):
        tr, r = settled(DONE)
        self.assertEqual(r.score, 1.0)
        n_hist = len(tr.history)
        good = DONE["iqi"]
        bads = [raw((math.nan, 0.0, PT), good.quat_xyzw), raw(good.origin, (0.0, 0.0, 0.0, 0.0)),
                raw(good.origin, (math.inf, 0.0, 0.0, 1.0)), SimpleNamespace(pos=good.pos, quat_xyzw=good.quat_xyzw)]
        for b in bads:
            for name in L.NAMES:
                st = dict(DONE)
                st[name] = b
                self.assertFalse(valid(b))
                r = tr.update(st, FAR)
                self.assertEqual(r.score, 0.0)
                self.assertFalse(r.details["valid_state"])
                self.assertEqual(len(tr.history), n_hist)
        missing = {"iqi": good, "marker_0": DONE["marker_0"]}
        self.assertEqual(tr.update(missing, FAR).score, 0.0)
        self.assertEqual(tr.update(DONE, FAR).score, 1.0)  # picks up where it was
        self.assertGreater(tr.bad_steps, 0)

    def test_unnormalised_quaternion(self):
        s = iqi_ok()
        big = raw(s.origin, 3.0 * np.asarray(s.quat_xyzw))
        self.assertEqual(scored(layout(big, marker_at(0), marker_at(1))), 1.0)

    def test_states_anywhere_do_not_crash(self):
        rnd = random.Random(7)
        tr = Tracker(HZ)
        for _ in range(300):
            st = {n: raw((rnd.uniform(-0.2, 0.9), rnd.uniform(-0.6, 0.6), rnd.uniform(0.6, 1.0)),
                         quat(rnd.uniform(-180, 180), rnd.uniform(-90, 90), rnd.uniform(-180, 180)),
                         vel=(rnd.uniform(-1, 1), 0.0, 0.0)) for n in L.NAMES}
            r = tr.update(st, FAR if rnd.random() < 0.5 else None)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 0.75, 1.0))
            json.dumps(r.details)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets(self):
        tr, _ = settled(DONE)
        self.assertGreater(tr.steps, 1)
        r = tr.update(layout(), FAR)
        self.assertEqual(tr.steps, 1)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(len(tr.history), 1)

    def test_teleport_to_a_jittered_start_resets(self):
        tr, _ = settled(DONE)
        st = {n: raw(np.asarray(start(n).origin) + np.array([0.02, -0.02, 0.0]), start(n).quat_xyzw)
              for n in L.NAMES}
        tr.update(st, FAR)
        self.assertEqual(tr.steps, 1)

    def test_gradual_return_does_not_reset(self):
        tr, _ = settled(layout(iqi_ok()))
        a, b = np.asarray(iqi_ok().origin), np.asarray(start("iqi").origin)
        for i in range(1, 41):
            p = a + (b - a) * i / 40
            tr.update(layout(raw(p, start("iqi").quat_xyzw, vel=(b - a) * HZ / 40)), FAR)
        self.assertEqual(tr.steps, FULL + 40)

    def test_a_jump_elsewhere_is_not_a_reset(self):
        tr, _ = settled(layout())
        tr.update(layout(iqi_ok()), FAR)
        self.assertEqual(tr.steps, FULL + 1)

    def test_explicit_reset(self):
        tr, _ = settled(DONE)
        tr.reset()
        self.assertEqual((tr.steps, len(tr.history), tr.best), (0, 0, 0.0))
        self.assertEqual(tr.result(DONE, FAR).score, 0.0)  # no history: nothing counted yet


def _stub_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline" / "yam_stub"
        if (d / "yam_arena").is_dir():
            return d
    return None


def _task():
    sys.path.insert(0, str(_stub_dir()))
    return importlib.import_module("task")


@unittest.skipIf(_stub_dir() is None, "needs the offline yam_arena stub (tools/offline/yam_stub)")
class TaskFile(unittest.TestCase):
    """task.py loaded against the offline yam_arena stand-in: its objects, hooks, the reset signals, the demo."""

    @classmethod
    def setUpClass(cls):
        cls.task = _task()
        import mechkit
        import rigstate
        cls.mk, cls.rig = mechkit, rigstate

    def sim(self, st):
        """Synthetic states turned into the AssemblyStates mechkit would report from the simulator."""
        specs = {"iqi": self.task.IQI, "marker_0": self.task.MARKER_0, "marker_1": self.task.MARKER_1}
        return {n: self.mk.state_from_poses(specs[n], (*st[n].origin, *st[n].quat_xyzw), source="sim")
                for n in L.NAMES}

    def feed_arms(self, arms):
        self.rig.offline_feed({k: {"tcp": a.tcp, "tips": a.tips, "opening": a.opening} for k, a in arms.items()})

    def test_objects(self):
        t = self.task
        names = [o.name for o in t.OBJECTS]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual([o.name for o in t.OBJECTS if o.mass is not None], list(L.NAMES))
        for o in (t.IQI, t.MARKER_0, t.MARKER_1):
            self.assertAlmostEqual(o.mass, 0.015)
            self.assertLessEqual(o.mass, 0.30)
        self.assertEqual((t.SCORE_OVER, t.SUCCESS_HOLD_S), ("end", 1.0))
        tops = {round(o.pos[2] + o.size[2] / 2, 9) for o in t.PLATE}
        self.assertEqual(tops, {round(PT, 9)})  # the weld is flush with the plates
        xs = sorted((o.pos[0] - o.size[0] / 2, o.pos[0] + o.size[0] / 2) for o in t.PLATE)
        self.assertAlmostEqual(xs[0][0], L.PLATE_X0)
        self.assertAlmostEqual(xs[-1][1], L.PLATE_X1)
        for (a0, a1), (b0, b1) in zip(xs, xs[1:]):
            self.assertAlmostEqual(a1, b0)  # no gap, no overlap
        weld = [o for o in t.PLATE if o.name == "weld"][0]
        self.assertAlmostEqual(weld.pos[0], L.WELD_X)
        self.assertAlmostEqual(weld.size[0], L.WELD_W)
        for tk, y in zip(t.TICKS, L.TICK_Y):
            self.assertAlmostEqual(tk.pos[1], y)
            self.assertAlmostEqual(tk.pos[0], L.WELD_X)
            self.assertAlmostEqual(tk.pos[2] - tk.size[2] / 2, PT)
        # the arms' start pose hides |y| > 0.45 m near x < 0.2 m in the screenshot: nothing is placed there
        for o in t.OBJECTS:
            if isinstance(o, self.mk.Assembly):
                lo, hi = o.world_bounds()
            else:
                lo = np.asarray(o.pos) - np.asarray(o.size) / 2
                hi = np.asarray(o.pos) + np.asarray(o.size) / 2
            self.assertLess(max(abs(lo[1]), abs(hi[1])), 0.45, o.name)
            self.assertGreater(lo[0], 0.20, o.name)

    def test_part_geometry_matches_the_scorer(self):
        t = self.task
        lo, hi = t.IQI.local_bounds()
        np.testing.assert_allclose(lo, L.IQI_LO, atol=1e-12)
        np.testing.assert_allclose(hi, L.IQI_HI, atol=1e-12)
        for m in (t.MARKER_0, t.MARKER_1):
            lo, hi = m.local_bounds()
            np.testing.assert_allclose(lo, L.MK_LO, atol=1e-12)
            np.testing.assert_allclose(hi, L.MK_HI, atol=1e-12)
        wires = [s for s in t.IQI.shapes if tuple(s.color or ()) == L.WIRE_DARK]
        self.assertEqual(len(wires), 6)
        for s in wires:  # along the holder's local x: the axis the scorer reads as the wires' direction
            self.assertAlmostEqual(s.size[0], L.WIRE_LEN)
            self.assertLess(s.size[1], 0.004)
        com = t.IQI.center_of_mass()
        self.assertLess(abs(com[0]) + abs(com[1]), 1e-3)
        rest = {n: o.rest_state() for n, o in (("iqi", t.IQI), ("marker_0", t.MARKER_0), ("marker_1", t.MARKER_1))}
        self.assertTrue(Tracker.at_start(rest))
        self.assertAlmostEqual(wire_err_deg(rest["iqi"]), 90.0, places=6)
        self.assertNotEqual([s.size for s in t.MARKER_0.shapes[1:]], [s.size for s in t.MARKER_1.shapes[1:]])

    def test_offline_state_path(self):
        box = SimpleNamespace(pos=np.array(self.task.IQI.offline_box()[0]), quat_xyzw=quat(90.0),
                              lin_vel=np.zeros(3), ang_vel=np.zeros(3))
        st = self.task.IQI.offline_state(box)
        self.assertEqual(st.source, "offline")
        np.testing.assert_allclose(st.origin, [L.IQI_START[0], L.IQI_START[1], L.TOP], atol=1e-12)

    def test_hooks_and_both_reset_signals(self):
        t = self.task
        tr = t._TRACKER
        self.rig.offline_reset()
        self.mk.offline_reset()
        self.feed_arms(FAR)
        st0 = self.sim(layout())
        self.assertFalse(t.success(st0))
        self.assertEqual(t.score(st0), 0.0)
        done = self.sim(DONE)
        for _ in range(FULL - 1):
            self.feed_arms(FAR)
            t.success(done)
        self.feed_arms(FAR)
        self.assertTrue(t.success(done))
        self.assertEqual(t.score(done), 1.0)
        self.assertEqual(t.score(done), 1.0)  # score() does not advance it
        self.feed_arms({"left": FAR["left"], "right": arm((L.WELD_X, 0.0, PT + 0.017), 20.0 / 93.9)})
        self.assertFalse(t.success(done))  # a gripper closed on the IQI again: not released
        self.assertEqual(t.score(done), 0.0)
        self.feed_arms(FAR)
        self.rig.offline_reset()  # rigstate's reset hook: the task's tracker starts over
        self.assertEqual(tr.steps, 0)
        self.feed_arms(FAR)
        t.success(done)
        t.success(done)
        self.assertEqual(tr.steps, 2)
        self.mk.offline_reset()  # mechkit's episode counter alone: also a new trial
        self.assertFalse(t.success(done))
        self.assertEqual(tr.steps, 1)

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
                for k in ("step", "score", "success", "tier_reason", "iqi_on_plate", "iqi_tilt_deg", "iqi_bottom_mm",
                          "iqi_wire_err_deg", "iqi_off_weld_mm", "iqi_max_abs_y_mm", "iqi_pose_ok", "iqi_correct",
                          "iqi_counted", "iqi_still_1s", "iqi_path_mm", "iqi_yaw_change_deg", "iqi_released",
                          "marker_0_tick", "marker_0_from_weld_mm", "marker_0_y_mm", "marker_1_bottom_mm",
                          "ticks_marked", "zone_gap_mm", "arms_unavailable", "iqi_pose", "marker_0_pose",
                          "left_tips", "right_opening", "episode", "mechkit_episode"):
                    self.assertIn(k, row)
                self.assertAlmostEqual(row["iqi_off_weld_mm"], 0.0)
                os.environ["TASK_TRACE"] = str(pathlib.Path(d) / "no_such_dir" / "trace.jsonl")
                self.assertFalse(self.task.success(self.sim(DONE)))  # no crash
            finally:
                if old is None:
                    os.environ.pop("TASK_TRACE", None)
                else:
                    os.environ["TASK_TRACE"] = old

    def _timeline(self, wps):
        t, out = 0.0, []
        for w in wps:
            out.append((t, t + w.duration, w))
            t += w.duration
        return out

    def test_demo_plan(self):
        t = self.task
        plan = t.demo()
        self.assertEqual(set(plan), {"left", "right"})
        for side, wps in plan.items():
            for prev, w in zip(wps, wps[1:]):
                if abs(prev.finger - w.finger) > 0.004:  # every close and open takes 1.5 s, after a 0.5 s settle
                    self.assertGreaterEqual(w.duration, 1.5)
                    self.assertEqual(prev.label, "settle")
                    self.assertGreaterEqual(prev.duration, 0.5)
            for w in wps:
                if w.label.startswith("slowly"):
                    self.assertGreaterEqual(w.duration, 1.5)
        by = {w.label: w for w in plan["right"]}
        grip, carry = by["grip iqi"], by["carry iqi over the weld, turning its wires across it"]
        self.assertAlmostEqual(grip.yaw, math.pi / 2)  # fingers closing along x, across the 28 mm width
        self.assertAlmostEqual(carry.yaw, math.radians(-1.2))  # laid with its wires along x (trimmed)
        self.assertAlmostEqual(grip.finger, -0.010)  # 8 mm narrower than the part
        self.assertAlmostEqual(by["grip marker_0"].finger, -0.007)
        self.assertAlmostEqual(grip.pos[2] - 0.013 - L.TOP, 0.003)  # pad tips 3 mm above the bench
        self.assertAlmostEqual(by["slowly lay iqi on the plate"].pos[2] - 0.016, PT + t.DROP)
        # the open pads clear the plate's near edge at the IQI pick
        self.assertLess(L.IQI_START[0] - t.IQI_OPEN + L.FINGER_THICK, L.PLATE_X0 - 0.015)
        # the place points, and 4 mm short of them, are inside every window
        for n in L.NAMES:
            px, py = t.place_point(n)
            if n == "iqi":
                self.assertLess(abs(px - L.WELD_X), L.ON_WELD - 0.002)
            else:
                self.assertLess(abs(px - L.WELD_X), L.MARK_ACROSS - 0.005)
                self.assertGreater(abs(px - L.WELD_X), L.WELD_W / 2 + L.MK_SIDE / 2)  # beside the weld
                self.assertLess(px + L.MK_SIDE / 2, L.WELD_X - L.TICK_LEN / 2)  # clear of the tick
                self.assertLess(abs(abs(py) - 0.080), 0.004)
        # the two grippers are never both over the plate: the left lays marker 1 while the right holds marker 0
        # up over its start, and neither moves to the plate before the other has gone up
        tl, tr_ = self._timeline(plan["left"]), self._timeline(plan["right"])

        def over_plate(tl_, t0, t1):
            return any(a < t1 - 1e-9 and b > t0 + 1e-9 and w.pos[0] > L.PLATE_X0 - 0.03 for a, b, w in tl_)

        for a, b, w in tl:
            if w.pos[0] > L.PLATE_X0 - 0.03:
                self.assertFalse(over_plate(tr_, a, b), w.label)
        self.assertLess(sum(w.duration for w in plan["right"]) * HZ, 0.8 * 1200)
        for key, val, want in (("DEMO_IDLE_ARM", "left", {"right"}), ("DEMO_IDLE_ARM", "right", {"left"}),
                               ("DEMO_HOLD_ONLY", "1", set())):
            os.environ[key] = val
            try:
                self.assertEqual(set(t.demo()), want)
            finally:
                os.environ.pop(key, None)

    def test_instruction_withholds_the_answer(self):
        ins = self.task.INSTRUCTION
        for s in ("radiographic", "film goes under the plate", "weld", "ticks", "ASME Section V", "let go"):
            self.assertIn(s, ins)
        for s in ("perpendicular", "across the weld", "source", "x-ray", "radiation"):
            self.assertNotIn(s, ins.lower())

    def test_task_json(self):
        meta = json.loads((HERE / "task.json").read_text())
        self.assertEqual(meta["instruction"], self.task.INSTRUCTION)
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(meta[k]), n, k)
        self.assertIn("1,200 steps", meta["run_limit"])
        for k in ("23.0 cm", "16.0 cm", "55 x 28 x 12 mm", "22 x 22 x 12 mm", "15 g", "36.0 cm", "8.0 cm"):
            self.assertIn(k, meta["setup"] + meta["scoring"], k)

    def test_text_files_are_clean(self):
        bad = re.compile("[" + chr(0x2013) + chr(0x2014) + "]")
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            p = HERE / f
            if p.is_file():
                text = p.read_text()
                self.assertIsNone(bad.search(text), f)
                if f in ("task.json", "README.md"):
                    self.assertFalse(_ident_hits(text), f)


@unittest.skipIf(_stub_dir() is None, "needs the offline kinematics (tools/offline)")
class OneArmReach(unittest.TestCase):
    def test_neither_arm_reaches_the_other_sides_marker_from_above(self):
        sys.path.insert(0, str(_stub_dir().parent))
        import yamkin as K
        for side, other in (("right", "marker_1"), ("left", "marker_0")):
            bx, by = L.BASES[side]
            x, y, _ = L.MK_START[other]
            for z in (0.016, 0.030, 0.055):
                for yaw in range(-180, 180, 15):
                    ok = K.reachable((x, y, L.TOP + z), math.radians(yaw), (bx, by, 0.75))[0]
                    self.assertFalse(ok, (side, z, yaw))



def _ident_hits(text):
    """Team-identifying words found in text. They are kept as sha256 prefixes so this file names nobody."""
    import hashlib
    import re as _re
    banned = {"54622c861585ec4f", "7a5e09c6c81cf86f", "a4cde7cecd46cd77", "b1fc1d0d7b2f62fa", "c70eca6b0f88f44d", "c857d09db23e6822", "d807637b43a9a01c", "da3a1991ce3a0d58", "db6017dfdfe82073", "f0556d0df0fd581f"}
    found = []
    for word in set(_re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())):
        for w in {word, *word.split("-")}:
            if hashlib.sha256(w.encode()).hexdigest()[:16] in banned:
                found.append(w)
    return found


if __name__ == "__main__":
    unittest.main()
