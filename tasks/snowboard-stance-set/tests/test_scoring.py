"""Offline tests for scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's own specs)
and synthetic arm snapshots (rigstate.Arm), no simulator.  python3 -m unittest discover -s tests -v"""

import json
import math
import pathlib
import re
import sys
import unittest

import numpy as np

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
            sys.path.insert(0, str(_p))  # mechkit and task.py import yam_arena: the offline stand-in

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402
import rigstate  # noqa: E402
import scoring as S  # noqa: E402
import task as T  # noqa: E402

HZ = 20.0
WIN = int(round(L.STILL_WINDOW_S * HZ))  # 20 steps
SPECS = {o.name: o for o in T.OBJECTS if getattr(o, "_mechkit_assembly", False)}
AWAY = {a: rigstate.Arm(np.array([0.30, s * 0.30, L.TOP + 0.25]), np.array([0.30, s * 0.30, L.TOP + 0.237]), 1.0)
        for a, s in (("left", 1.0), ("right", -1.0))}
MIDDLE = {"binding_nose": (L.LINE_X, 0.200), "binding_tail": (L.LINE_X, -0.200)}


# -- states --------------------------------------------------------------------------------------------------------


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def state(name, origin, r=np.eye(3)):
    return mk.state_from_poses(SPECS[name], (*origin, *mk.matrix_to_quat(r)), source="offline")


def binding(name, xy=None, angle=None, z=L.BOARD_TOP, tilt_deg=0.0, tilt_axis="x"):
    """A binding with its disc centre at ``xy`` and its angle (the baseplate's heading) at ``angle``."""
    xy = MIDDLE[name] if xy is None else xy
    angle = L.PRESET_DEG[name] if angle is None else angle
    r = _rz(math.radians(angle - L.PRESET_DEG[name]))
    if tilt_deg:
        r = r @ (_rx if tilt_axis == "x" else _ry)(math.radians(tilt_deg))
    return state(name, (xy[0], xy[1], z), r)


def screw(name, pos, yaw_deg=0.0, tilt_deg=0.0):
    r = _rz(math.radians(yaw_deg)) @ _ry(math.radians(tilt_deg))
    return state(name, pos, r)


def slot_insert(b_state, k):
    """The insert under slot k of a binding (its world centre on the board's top), or None."""
    over = S.slots_over(*S.frame(b_state))
    i = over[k]
    return None if i is None else S.INSERT_XY[i]


def seated_at(name, b_state, k, gap=0.0, dxy=(0.0, 0.0), yaw_deg=0.0, tilt_deg=0.0):
    """A screw through slot k of a binding into the insert under it, its washer ``gap`` above the disc's top."""
    xy = slot_insert(b_state, k)
    o, r = S.frame(b_state)
    disc_top = float((o + r @ np.array([0.0, 0.0, L.FLOOR_T]))[2])
    return screw(name, (xy[0] + dxy[0], xy[1] + dxy[1], disc_top + gap), yaw_deg, tilt_deg)


def in_holder(name):
    x, y = T.SOCKETS[name]
    return screw(name, (x, y, L.TOP + L.HOLDER_TOP))


def nose_screws():
    return [s for s in S.SCREWS if s.startswith("screw_n")]


def tail_screws():
    return [s for s in S.SCREWS if s.startswith("screw_t")]


def scene(nose=None, tail=None, seat=(), **screws):
    """Every part: the bindings (default: on the middle squares), screws in their holders unless given; ``seat``
    lists bindings whose four screws are seated in their slots."""
    st = {"binding_nose": nose if nose is not None else binding("binding_nose"),
          "binding_tail": tail if tail is not None else binding("binding_tail")}
    for n in S.SCREWS:
        st[n] = in_holder(n)
    for b in seat:
        names = nose_screws() if b == "binding_nose" else tail_screws()
        for k, n in enumerate(names):
            st[n] = seated_at(n, st[b], k)
    st.update(screws)
    for o in T.OBJECTS:
        if o.name not in st and o.name in SPECS:
            st[o.name] = SPECS[o.name].rest_state()
    return st


def start_scene():
    return scene(nose=binding("binding_nose", L.NOSE_START[:2], L.NOSE_START[2]),
                 tail=binding("binding_tail", L.TAIL_START[:2], L.TAIL_START[2]))


def drive_all(fast, st, bindings=S.BINDINGS, steps=80, deg=5.0):
    """Turn every seated screw of the given bindings clockwise ``steps`` x ``deg`` through Fastening (as a gripper
    would), and return the final states (screws at their welded homes, as the welds would hold them)."""
    st = dict(st)
    names = [n for b in bindings for n in (nose_screws() if b == "binding_nose" else tail_screws())]
    yaws = {n: 0.0 for n in names}
    for _ in range(steps):
        for n in names:
            yaws[n] -= deg
            o, _r = S.frame(st[n])
            st[n] = screw(n, o, yaws[n])
        fast.update(st)
    for n in names:
        rule = fast.screws[n]
        if rule.driven:
            st[n] = state(n, rule.home[0], mk.quat_to_matrix(rule.home[1]))
    return st


def run(tracker, fast, st, arms=AWAY, n=WIN + 2):
    res = None
    for _ in range(n):
        fast.update(st)
        res = tracker.update(st, arms, fast)
    return res


def fresh():
    return S.Tracker(), S.Fastening()


# -- tests ---------------------------------------------------------------------------------------------------------


class Geometry(unittest.TestCase):
    def test_slots_line_up_over_the_middle_square_at_the_preset(self):
        for b, half in (("binding_nose", "n"), ("binding_tail", "t")):
            over = S.slots_over(*S.frame(binding(b)))
            got = sorted(tuple(np.round(S.INSERT_XY[i], 3)) for i in over)
            self.assertEqual(got, sorted(L.middle_inserts(half)))

    def test_preset_angle_is_the_baseplate_heading(self):
        self.assertAlmostEqual(S.binding_angle("binding_nose", S.frame(binding("binding_nose"))[1]), 15.0)
        self.assertAlmostEqual(S.binding_angle("binding_tail", S.frame(binding("binding_tail"))[1]), -15.0)

    def test_slots_miss_the_inserts_off_the_presets(self):
        for d in (20.0, 45.0, 60.0):  # the long slots let a disc 10 deg off still take screws: the angle rule stops it
            over = S.slots_over(*S.frame(binding("binding_nose", angle=15.0 + d)))
            self.assertIn(None, over, d)

    def test_slots_line_up_every_quarter_turn_and_only_the_preset_is_aligned(self):
        tr, fa = fresh()
        for q in (90.0, 180.0, 270.0):
            b = binding("binding_nose", angle=15.0 + q)
            self.assertNotIn(None, S.slots_over(*S.frame(b)))
            res = run(tr, fa, scene(nose=b, seat=("binding_tail",)))
            self.assertFalse(res.details["binding_nose"]["aligned"], q)

    def test_alignment_play_boundaries(self):
        # along the board (across the slots): 4.0 mm; across the board (along the slots): 11.0 mm
        for dy, ok in ((0.0040, True), (0.0041, False), (-0.0040, True)):
            over = S.slots_over(*S.frame(binding("binding_nose", (L.LINE_X, 0.200 + dy))))
            self.assertEqual(None not in over, ok, dy)
        for dx, ok in ((0.0110, True), (0.0111, False), (-0.0110, True)):
            over = S.slots_over(*S.frame(binding("binding_nose", (L.LINE_X + dx, 0.200))))
            self.assertEqual(None not in over, ok, dx)

    def test_slot_and_bore_play_are_real_numbers(self):
        self.assertAlmostEqual(S.BORE_PLAY, 0.0015)
        self.assertAlmostEqual(S.SLOT_PLAY[0], 0.0095)
        self.assertAlmostEqual(S.SLOT_PLAY[1], 0.0025)
        self.assertAlmostEqual(L.WASHER_R * 2, 0.018)
        self.assertGreater(2 * L.WASHER_R, L.SLOT_W + 2 * 0.003)  # a washer bears on the disc either side of a slot

    def test_seated_fit(self):
        b = binding("binding_nose")
        f = S.screw_fit(seated_at("screw_n1", b, 0), b)
        self.assertTrue(S.seated(f))
        self.assertEqual(f.slot, 0)
        self.assertAlmostEqual(f.depth, L.STUB_L - L.FLOOR_T, places=6)  # 6.5 mm down the insert
        self.assertTrue(S.seated(S.screw_fit(seated_at("screw_n1", b, 0, gap=0.0010), b)))
        self.assertFalse(S.seated(S.screw_fit(seated_at("screw_n1", b, 0, gap=0.0011), b)))
        self.assertTrue(S.seated(S.screw_fit(seated_at("screw_n1", b, 0, tilt_deg=5.0), b)))
        self.assertFalse(S.seated(S.screw_fit(seated_at("screw_n1", b, 0, tilt_deg=5.2), b)))
        self.assertTrue(S.seated(S.screw_fit(seated_at("screw_n1", b, 0, dxy=(0.0019, 0.0)), b)))
        self.assertFalse(S.seated(S.screw_fit(seated_at("screw_n1", b, 0, dxy=(0.0021, 0.0)), b)))  # out of the bore

    def test_a_screw_in_an_insert_with_no_slot_over_it_is_not_seated(self):
        b = binding("binding_nose", (L.LINE_X, 0.160))  # the inner square
        s = screw("screw_n1", (0.320, 0.260, L.BOARD_TOP + L.FLOOR_T))  # an outer-row insert, beside the disc
        self.assertFalse(S.seated(S.screw_fit(s, b)))

    def test_a_screw_resting_on_the_disc_is_not_seated(self):
        b = binding("binding_nose")
        s = screw("screw_n1", (0.300, 0.200, L.BOARD_TOP + L.FLOOR_T))  # on the disc's centre, no slot
        self.assertFalse(S.seated(S.screw_fit(s, b)))

    def test_finger_zone_is_the_kit_one(self):
        self.assertAlmostEqual(rigstate.zone_radius(1.0), math.hypot(0.0939 / 2 + 0.008, 0.0135))


class Rules(unittest.TestCase):
    def setUp(self):
        self.fast = S.Fastening()
        self.st = scene(seat=("binding_nose",))

    def _turn(self, n, deg_per_step, steps, gap=0.0, start_yaw=0.0):
        """One step at ``start_yaw`` (no turn), then ``steps`` steps of ``deg_per_step`` clockwise."""
        yaw = start_yaw
        self.st[n] = seated_at(n, self.st["binding_nose"], 0, gap=gap, yaw_deg=yaw)
        events = self.fast.update(self.st)
        for _ in range(steps):
            yaw -= deg_per_step
            self.st[n] = seated_at(n, self.st["binding_nose"], 0, gap=gap, yaw_deg=yaw)
            events += self.fast.update(self.st)
        return events, yaw

    def test_one_clockwise_turn_while_seated_drives_the_screw(self):
        events, _ = self._turn("screw_n1", 5.0, 71)  # 355 deg
        self.assertFalse(self.fast.screws["screw_n1"].driven)
        self.assertAlmostEqual(self.fast.screws["screw_n1"].turn, 355.0)
        events, _ = self._turn("screw_n1", 5.0, 1, start_yaw=-355.0)  # 360
        self.assertEqual([e[0] for e in events], ["drive"])
        self.assertTrue(self.fast.screws["screw_n1"].driven)
        pos, quat = events[0][2]
        self.assertAlmostEqual(pos[2], L.BOARD_TOP + L.FLOOR_T, places=9)  # welded with its washer on the disc
        self.assertAlmostEqual(np.linalg.norm(quat), 1.0)

    def test_counter_clockwise_turns_do_not_drive_and_count_back(self):
        self._turn("screw_n1", -5.0, 100)  # 500 deg the wrong way
        self.assertFalse(self.fast.screws["screw_n1"].driven)
        self.assertLess(self.fast.screws["screw_n1"].turn, 0.0)

    def test_lifting_the_screw_out_resets_the_count(self):
        self._turn("screw_n1", 5.0, 60)  # 300 deg
        self._turn("screw_n1", 0.0, 1, gap=0.0031, start_yaw=-300.0)  # the washer 3.1 mm up: disengaged
        self.assertEqual(self.fast.screws["screw_n1"].turn, 0.0)
        self._turn("screw_n1", 5.0, 20, start_yaw=-300.0)
        self.assertFalse(self.fast.screws["screw_n1"].driven)

    def test_engaging_needs_the_washer_within_1_mm_and_it_stays_up_to_3_mm(self):
        self._turn("screw_n1", 5.0, 10, gap=0.0015)
        self.assertIsNone(self.fast.screws["screw_n1"].binding)
        self._turn("screw_n1", 5.0, 1, gap=0.0010)
        self.assertEqual(self.fast.screws["screw_n1"].binding, "binding_nose")
        self._turn("screw_n1", 5.0, 80, gap=0.0029)
        self.assertTrue(self.fast.screws["screw_n1"].driven)

    def test_turning_a_screw_hovering_above_the_slot_counts_nothing(self):
        o, r = S.frame(self.st["binding_nose"])
        xy = slot_insert(self.st["binding_nose"], 0)
        for k in range(100):
            self.st["screw_n1"] = screw("screw_n1", (xy[0], xy[1], L.BOARD_TOP + L.FLOOR_T + 0.006), -5.0 * k)
            self.fast.update(self.st)
        self.assertFalse(self.fast.screws["screw_n1"].driven)
        self.assertIsNone(self.fast.screws["screw_n1"].binding)

    def test_turning_in_the_holder_counts_nothing(self):
        for k in range(100):
            x, y = T.SOCKETS["screw_n1"]
            self.st["screw_n1"] = screw("screw_n1", (x, y, L.TOP + L.HOLDER_TOP), -5.0 * k)
            self.fast.update(self.st)
        self.assertFalse(self.fast.screws["screw_n1"].driven)

    def test_four_driven_screws_clamp_a_flat_binding_once(self):
        st = drive_all(self.fast, self.st, ("binding_nose",))
        self.assertEqual(self.fast.driven_in("binding_nose"), {0, 1, 2, 3})
        self.assertIn("binding_nose", self.fast.clamped)
        kinds = [e[0] for e in self.fast.events_seen]
        self.assertEqual(kinds.count("drive"), 4)
        self.assertEqual(kinds.count("clamp"), 1)
        self.assertEqual(self.fast.update(st), [])

    def test_a_tilted_binding_is_not_clamped(self):
        tilted = binding("binding_nose", tilt_deg=3.5)
        st = scene(nose=tilted, seat=("binding_nose",))
        drive_all(self.fast, st, ("binding_nose",))
        self.assertNotIn("binding_nose", self.fast.clamped)

    def test_a_step_is_counted_once_per_key(self):
        self.st["screw_n1"] = seated_at("screw_n1", self.st["binding_nose"], 0, yaw_deg=0.0)
        self.fast.update(self.st, key=("sim", 1, 1))
        self.st["screw_n1"] = seated_at("screw_n1", self.st["binding_nose"], 0, yaw_deg=-10.0)
        self.fast.update(self.st, key=("sim", 1, 2))
        self.fast.update(self.st, key=("sim", 1, 2))
        self.assertAlmostEqual(self.fast.screws["screw_n1"].turn, 10.0)


class Tiers(unittest.TestCase):
    def test_start_scores_zero(self):
        tr, fa = fresh()
        self.assertEqual(run(tr, fa, start_scene()).score, 0.0)

    def test_every_tier_is_reachable(self):
        cases = []
        tr, fa = fresh()
        cases.append((run(tr, fa, scene(seat=("binding_nose",))).score, 0.25))
        tr, fa = fresh()
        cases.append((run(tr, fa, scene(seat=S.BINDINGS)).score, 0.5))
        tr, fa = fresh()
        st = drive_all(fa, scene(seat=S.BINDINGS))
        cases.append((run(tr, fa, st).score, 1.0))
        tr, fa = fresh()
        outer = scene(nose=binding("binding_nose", (L.LINE_X, 0.240)), tail=binding("binding_tail", (L.LINE_X, -0.160)),
                      seat=S.BINDINGS)
        st = drive_all(fa, outer)
        cases.append((run(tr, fa, st).score, 0.75))
        tr, fa = fresh()
        cases.append((run(tr, fa, start_scene()).score, 0.0))
        self.assertEqual([c[0] for c in cases], [c[1] for c in cases])

    def test_fastened_details(self):
        tr, fa = fresh()
        st = drive_all(fa, scene(seat=S.BINDINGS))
        res = run(tr, fa, st)
        for b in S.BINDINGS:
            d = res.details[b]
            self.assertEqual((d["aligned"], d["seated"], d["driven"], d["clamped"], d["fastened"]),
                             (True, 4, 4, True, True))
        self.assertEqual((res.details["W_mm"], res.details["C_mm"]), (400.0, 0.0))
        self.assertTrue(res.success)

    def test_one_fastened_and_the_other_three_seated_is_a_quarter(self):
        tr, fa = fresh()
        st = drive_all(fa, scene(seat=S.BINDINGS), ("binding_nose",))
        st["screw_t4"] = in_holder("screw_t4")
        self.assertEqual(run(tr, fa, st).score, 0.25)

    def test_both_fastened_with_one_on_an_outer_square_is_three_quarters(self):
        for ny, ty in ((0.240, -0.200), (0.200, -0.240), (0.160, -0.160), (0.240, -0.240)):
            tr, fa = fresh()
            st = drive_all(fa, scene(nose=binding("binding_nose", (L.LINE_X, ny)),
                                     tail=binding("binding_tail", (L.LINE_X, ty)), seat=S.BINDINGS))
            self.assertEqual(run(tr, fa, st).score, 0.75, (ny, ty))

    def test_both_aligned_without_screws_is_zero(self):
        tr, fa = fresh()
        self.assertEqual(run(tr, fa, scene()).score, 0.0)

    def test_three_seated_screws_are_not_enough(self):
        tr, fa = fresh()
        st = scene(seat=("binding_nose",))
        st["screw_n4"] = in_holder("screw_n4")
        self.assertEqual(run(tr, fa, st).score, 0.0)

    def test_scores_are_only_the_listed_ones(self):
        rng = np.random.default_rng(3)
        seen = set()
        for _ in range(40):
            tr, fa = fresh()
            st = scene(seat=[b for b in S.BINDINGS if rng.random() < 0.6])
            for n in S.SCREWS:
                if rng.random() < 0.15:
                    st[n] = in_holder(n)
            if rng.random() < 0.5:
                st = drive_all(fa, st, steps=int(rng.integers(0, 80)))
            seen.add(run(tr, fa, st).score)
        self.assertTrue(seen <= {0.0, 0.25, 0.5, 0.75, 1.0}, seen)


class ZeroConditions(unittest.TestCase):
    def setUp(self):
        self.tr, self.fa = fresh()
        self.done = drive_all(self.fa, scene(seat=S.BINDINGS))

    def test_wrong_angle_or_heel_forward_is_not_aligned(self):
        tr, fa = fresh()
        st = scene(nose=binding("binding_nose", angle=15.0 + 180.0), seat=S.BINDINGS)
        self.assertEqual(run(tr, fa, st).score, 0.25)  # the tail binding only

    def test_swapped_bindings_cannot_line_up(self):
        tr, fa = fresh()
        nose_on_tail = binding("binding_nose", (L.LINE_X, -0.200), 15.0)  # its slots line up there, angle +15
        tail_on_nose = binding("binding_tail", (L.LINE_X, 0.200), -15.0)
        res = run(tr, fa, scene(nose=nose_on_tail, tail=tail_on_nose, seat=S.BINDINGS))
        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.details["binding_nose"]["role"], "back")

    def test_a_binding_held_hovering_is_not_flat(self):
        tr, fa = fresh()
        b = binding("binding_nose", z=L.BOARD_TOP + 0.0016)
        self.assertEqual(run(tr, fa, scene(nose=b, seat=S.BINDINGS)).score, 0.25)

    def test_a_tilted_binding_is_not_aligned(self):
        tr, fa = fresh()
        b = binding("binding_nose", tilt_deg=3.2, tilt_axis="y")
        self.assertEqual(run(tr, fa, scene(nose=b, seat=("binding_tail",))).score, 0.25)
        tr, fa = fresh()
        b = binding("binding_nose", tilt_deg=2.9, tilt_axis="y")
        self.assertTrue(run(tr, fa, scene(nose=b, seat=("binding_tail",))).details["binding_nose"]["aligned"])

    def test_holding_a_part_scores_zero(self):
        for part in ("binding_nose", "screw_t3"):
            tr = S.Tracker()
            o, _ = S.frame(self.done[part])
            tip = o + np.array([0.0, 0.0, 0.035 if part.startswith("binding") else 0.004])
            held = dict(AWAY, left=rigstate.Arm(tip + np.array([0, 0, 0.013]), tip, 0.1))
            self.assertEqual(run(tr, self.fa, self.done, held).score, 0.0, part)

    def test_open_fingers_hovering_near_a_part_are_not_released(self):
        o, _ = S.frame(self.done["screw_n2"])
        tips = o + np.array([0.0, 0.0, L.HEAD_H + L.WASHER_T + 0.025])  # 25 mm over the head, open
        hover = dict(AWAY, left=rigstate.Arm(tips + np.array([0, 0, 0.013]), tips, 1.0))
        self.assertEqual(run(self.tr, self.fa, self.done, hover).score, 0.0)

    def test_released_for_less_than_one_second_is_zero(self):
        o, _ = S.frame(self.done["binding_tail"])
        tip = o + np.array([0.0, 0.0, 0.04])
        held = dict(AWAY, right=rigstate.Arm(tip + np.array([0, 0, 0.013]), tip, 0.1))
        run(self.tr, self.fa, self.done, held, n=WIN + 2)
        self.assertEqual(run(self.tr, self.fa, self.done, AWAY, n=WIN - 1).score, 0.0)
        self.assertEqual(run(self.tr, self.fa, self.done, AWAY, n=1).score, 1.0)

    def test_unreadable_arms_are_never_released(self):
        bad = [None, {}, {"left": AWAY["left"]},
               dict(AWAY, left=rigstate.Arm(np.array([np.nan, 0.3, 1.0]), np.array([0.3, 0.3, 1.0]), 1.0)),
               dict(AWAY, right=rigstate.Arm(AWAY["right"].tcp, AWAY["right"].tips, float("nan")))]
        for arms in bad:
            tr = S.Tracker()
            res = run(tr, self.fa, self.done, arms)
            self.assertEqual(res.score, 0.0, arms)
            self.assertFalse(res.details["released"])

    def test_a_distance_that_raises_is_not_released(self):
        broken = dict(self.done)
        broken["screw_n1"] = mk.state_from_poses(SPECS["screw_n1"], (*S.frame(self.done["screw_n1"])[0], 0, 0, 0, 1))
        object.__setattr__(broken["screw_n1"], "spec", None)  # its distance() now raises
        self.assertEqual(run(S.Tracker(), self.fa, broken).score, 0.0)

    def test_a_moving_part_scores_zero(self):
        tr = S.Tracker()
        run(tr, self.fa, self.done)
        st = dict(self.done)
        x, y = T.SOCKETS["screw_n1"]
        res = None
        for k in range(WIN):  # a stray screw creeping 3 mm over 1 s on the table
            st["screw_n1"] = screw("screw_n1", (0.50, 0.10 + 0.00015 * k, L.TOP))
            res = tr.update(st, AWAY, self.fa)
        self.assertEqual(res.score, 0.0)
        self.assertIn("screw_n1", res.details["moving"])

    def test_a_part_off_the_table_latches_zero(self):
        tr = S.Tracker()
        st = dict(self.done)
        st["screw_n1"] = screw("screw_n1", (0.50, 0.10, L.TOP - 0.03))
        run(tr, self.fa, st, n=2)
        self.assertEqual(tr.lost, "screw_n1")
        self.assertEqual(run(tr, self.fa, self.done).score, 0.0)  # even back in place
        tr.reset()
        self.assertEqual(run(tr, self.fa, self.done).score, 1.0)

    def test_a_dropped_screw_on_the_table_is_not_latched(self):
        tr, fa = fresh()
        st = scene(seat=("binding_nose",))
        st["screw_t2"] = screw("screw_t2", (0.50, -0.10, L.TOP))
        self.assertEqual(run(tr, fa, st).score, 0.25)

    def test_bad_states_score_zero_and_restart_the_windows(self):
        tr = S.Tracker()
        run(tr, self.fa, self.done)
        st = dict(self.done)
        st["binding_nose"] = mk.state_from_poses(SPECS["binding_nose"], (0.3, 0.2, L.BOARD_TOP, 0, 0, 0, 1))
        object.__setattr__(st["binding_nose"], "origin", np.array([np.nan, 0.2, L.BOARD_TOP]))
        self.assertEqual(tr.update(st, AWAY, self.fa).score, 0.0)
        self.assertEqual(run(tr, self.fa, self.done, n=WIN).score, 0.0)
        self.assertEqual(run(tr, self.fa, self.done, n=1).score, 1.0)
        missing = dict(self.done)
        del missing["screw_t1"]
        self.assertEqual(tr.update(missing, AWAY, self.fa).score, 0.0)


class Reset(unittest.TestCase):
    def test_teleport_back_to_start_resets_the_tracker(self):
        tr, fa = fresh()
        st = drive_all(fa, scene(seat=S.BINDINGS))
        run(tr, fa, st)
        tr.update(start_scene(), AWAY, fa)
        self.assertEqual((tr.resets, tr.steps), (1, 1))

    def test_a_gradual_return_is_not_a_reset(self):
        tr, fa = fresh()
        a = np.array(MIDDLE["binding_nose"])
        b = np.array(L.NOSE_START[:2])
        for k in range(41):
            p = a + (b - a) * k / 40
            ang = 15.0 + (L.NOSE_START[2] - 15.0) * k / 40
            st = scene(nose=binding("binding_nose", p, ang), tail=binding("binding_tail", L.TAIL_START[:2], -90.0))
            tr.update(st, AWAY, fa)
        self.assertEqual(tr.resets, 0)

    def test_task_resets_with_mechkit_and_rigstate(self):
        mk.offline_reset()
        rigstate.offline_reset()
        T.FASTENING.screws["screw_n1"].driven = True
        T.FASTENING.clamped["binding_nose"] = ((0, 0, 0), (0, 0, 0, 1))
        mk.offline_reset()  # mechkit's on_reset runs the task's rule reset
        self.assertFalse(T.FASTENING.screws["screw_n1"].driven)
        self.assertEqual(T.FASTENING.clamped, {})
        T._TRACKER.steps = 7
        rigstate.offline_reset()
        self.assertEqual(T._TRACKER.steps, 0)
        for w in (*T.SCREW_WELDS.values(), *T.CLAMP_WELDS.values()):
            self.assertFalse(w.active)


class TaskHooks(unittest.TestCase):
    def setUp(self):
        mk.offline_reset()
        rigstate.offline_reset()

    def _feed(self, arms=AWAY):
        rigstate.offline_feed({a: {"tcp": arms[a].tcp, "tips": arms[a].tips, "opening": arms[a].opening}
                               for a in ("left", "right")})

    def test_start_is_not_success(self):
        self._feed()
        self.assertFalse(T.success(start_scene()))
        self.assertEqual(T.score(start_scene()), 0.0)

    def test_offline_rules_drive_weld_and_score_one(self):
        st = scene(seat=S.BINDINGS)
        names = list(S.SCREWS)
        yaw = 0.0
        ok = False
        for k in range(110):
            yaw -= 5.0 if k < 80 else 0.0
            for n in names:
                rule = T.FASTENING.screws[n]
                if rule.driven:
                    st[n] = state(n, rule.home[0], mk.quat_to_matrix(rule.home[1]))
                else:
                    o, _ = S.frame(st[n])
                    st[n] = screw(n, o, yaw)
            self._feed()
            ok = T.success(st)
        self.assertTrue(ok)
        self.assertEqual(T.score(st), 1.0)
        self.assertTrue(all(w.active for w in T.SCREW_WELDS.values()))
        self.assertTrue(all(w.active for w in T.CLAMP_WELDS.values()))

    def test_unreadable_arms_through_the_task(self):
        st = drive_all(S.Fastening(), scene(seat=S.BINDINGS))
        nan = {a: {"tcp": [float("nan")] * 3, "tips": [0.3, 0.0, 1.0], "opening": 1.0} for a in ("left", "right")}
        for _ in range(WIN + 3):
            rigstate.offline_feed(nan)
            self.assertFalse(T.success(st))
        self.assertEqual(T.score(st), 0.0)


class Offline(unittest.TestCase):
    def test_binding_column_round_trip(self):
        tf = sys.modules["yam_arena.task_file"]
        for spec in (T.BINDING_NOSE, T.BINDING_TAIL):
            c, size, yaw = spec.offline_box()
            st = tf.ObjectState(np.array(c), mk.yaw_quat(yaw), np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3))
            back = spec.offline_state(st)
            self.assertTrue(np.allclose(back.origin, spec.pos))
            self.assertAlmostEqual(S.binding_angle(spec.name, back.rotation()),
                                   L.NOSE_START[2] if spec is T.BINDING_NOSE else L.TAIL_START[2])
            self.assertAlmostEqual(c[2] - size[2] / 2, L.BOARD_TOP)  # the column stands on the board

    def test_screw_column_round_trip_and_seated_height(self):
        tf = sys.modules["yam_arena.task_file"]
        spec = SPECS["screw_n1"]
        c, size, yaw = spec.offline_box()
        self.assertAlmostEqual(c[2] - size[2] / 2, L.TOP + L.HOLDER_TOP - L.FLOOR_T)
        dropped = tf.ObjectState(np.array([0.32, 0.22, L.BOARD_TOP + size[2] / 2]), mk.yaw_quat(0.3), np.zeros(3),
                                 np.zeros(3), np.zeros(3), np.zeros(3))
        back = spec.offline_state(dropped)
        self.assertAlmostEqual(back.origin[2], L.BOARD_TOP + L.FLOOR_T)  # washer on a disc's top
        self.assertTrue(S.seated(S.screw_fit(back, binding("binding_nose"))))


class Clearances(unittest.TestCase):
    """The demo's fingers against the binding's sidewalls and highback while it turns each screw (the pads as
    8 mm x 27 mm x 38 mm boxes, as the finger zone; the gripper's own collision capsules from the offline checker),
    with the binding at its target and 6 mm / 1.5 mm off it (the placement the trims leave)."""

    @staticmethod
    def _walls(b_state):
        o, r = S.frame(b_state)
        p = L.preset_rad(b_state.spec.name)
        out = []
        for sh in T._BASEPLATE[-3:]:  # two sidewalls, the highback
            c = np.array(sh.center)
            cw = o + r @ (_rz(p) @ c)
            out.append((cw, np.array(sh.size), r @ _rz(p)))
        return out

    def _pad_points(self, centre, yaw, gap, tip_z):
        pts = []
        for sign in (-1.0, 1.0):
            for a in np.linspace(gap / 2, gap / 2 + 0.008, 3):
                for b in np.linspace(-0.0135, 0.0135, 5):
                    for h in np.linspace(0.0, 0.038, 5):
                        close = np.array([-math.sin(yaw), math.cos(yaw)])  # yaw 0 closes along world y
                        across = np.array([math.cos(yaw), math.sin(yaw)])
                        p = np.asarray(centre[:2], dtype=float) + sign * a * close + b * across
                        pts.append([p[0], p[1], tip_z + h])
        return np.array(pts)

    def test_pads_clear_the_walls_through_every_stroke(self):
        worst = math.inf
        for side, b in (("left", "binding_nose"), ("right", "binding_tail")):
            sign = 1.0 if b == "binding_nose" else -1.0
            for dx, dy in ((0, 0), (0.006, 0.0015 * sign), (-0.006, -0.0015 * sign), (0.006, -0.0015 * sign)):
                bs = binding(b, (L.LINE_X + dx, sign * 0.200 + dy))
                walls = self._walls(bs)
                for _, xy in T.SCREW_ORDER[side]:
                    c = T.YAW_CENTRE[side]
                    for yaw in np.radians(np.linspace(c - T.STROKE_DEG, c + T.STROKE_DEG, 39)):
                        for tcp in (T.TCP_SEATED, T.TCP_TURN_BACK, T.TCP_RELEASE):
                            pts = self._pad_points(xy, yaw, 2 * -T.OPEN_SCREW, L.TOP + tcp - T.PAD_TIPS)
                            for cw, size, rot in walls:
                                worst = min(worst, float(rigstate.box_distances(pts, cw, size, rot).min()))
        self.assertGreater(worst, 0.008, worst)

    def test_gripper_capsules_clear_the_highback_through_every_stroke(self):
        if OFFLINE is None:
            self.skipTest("tools/offline not found")
        import yamkin as K
        worst = math.inf
        for side, b in (("left", "binding_nose"), ("right", "binding_tail")):
            sign = 1.0 if b == "binding_nose" else -1.0
            base = np.array(K.BASES["station"][side])
            for dx in (0.0, 0.006):
                bs = binding(b, (L.LINE_X + dx, sign * 0.200))
                cw, size, rot = self._walls(bs)[-1]
                for _, xy in T.SCREW_ORDER[side]:
                    seed = None
                    c = T.YAW_CENTRE[side]
                    for yaw in np.radians(np.linspace(c + T.STROKE_DEG, c - T.STROKE_DEG, 11)):
                        ok, q, _, _ = K.reachable((xy[0], xy[1], L.TOP + T.TCP_RELEASE), float(yaw), base,
                                                  seeds=None if seed is None else [seed])
                        self.assertTrue(ok, (side, xy, yaw))
                        seed = q
                        for link, a, e, rad in K.arm().capsules(q, base, T.OPEN_SCREW):
                            if link in K.FINGER_LINKS:
                                continue
                            pts = np.linspace(a, e, 15)
                            worst = min(worst, float(rigstate.box_distances(pts, cw, size, rot).min()) - rad)
        self.assertGreater(worst, 0.003, worst)

    def test_open_pads_clear_the_highback_at_the_grip(self):
        bs = binding("binding_nose", L.NOSE_START[:2], L.NOSE_START[2])
        cw, size, rot = self._walls(bs)[-1]
        x, y = T._hb_point(L.NOSE_START[:2], L.NOSE_START[2])
        pts = self._pad_points((x, y), 0.0, 2 * -T.OPEN_HB, L.TOP + T.TCP_HB - T.PAD_TIPS)
        self.assertGreater(float(rigstate.box_distances(pts, cw, size, rot).min()), 0.019)


class Demo(unittest.TestCase):
    def test_both_arms_and_the_switches(self):
        import os
        self.assertEqual(set(T.demo()), {"left", "right"})
        os.environ["DEMO_IDLE_ARM"] = "left"
        try:
            self.assertEqual(set(T.demo()), {"right"})
        finally:
            del os.environ["DEMO_IDLE_ARM"]
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            del os.environ["DEMO_HOLD_ONLY"]

    def test_no_waypoint_turns_more_than_95_degrees(self):
        for arm, plan in T.demo().items():
            yaw = 0.0
            for w in plan:
                self.assertLessEqual(abs(S.wrap_deg(math.degrees(w.yaw - yaw))), 95.0 + 1e-6, (arm, w.label))
                if w.label.startswith("turn it clockwise") or w.label.startswith("turn back"):
                    self.assertAlmostEqual(abs(S.wrap_deg(math.degrees(w.yaw - yaw))) % 90.0, 0.0, places=6)
                yaw = w.yaw

    def test_each_screw_is_turned_1080_degrees_clockwise_while_gripped(self):
        for arm, plan in T.demo().items():
            total, yaw, strokes = 0.0, None, 0
            for w in plan:
                if yaw is not None and w.finger == T.GRIP_SCREW and w.label.startswith("turn it clockwise"):
                    total += -S.wrap_deg(math.degrees(w.yaw - yaw))
                    strokes += 1
                yaw = w.yaw
            self.assertEqual(T.STROKES, 6)
            self.assertEqual(strokes, 48)  # four screws, six 180 deg strokes of two waypoints each
            self.assertAlmostEqual(total, 4 * 1080.0)

    def test_grip_and_turn_back_heights(self):
        # Gripping a seated head the pad tips are 0.5 mm above its washer (4.5 mm of the 5 mm head held); open and
        # turning back they are 2.5 mm above it, still round the head.
        washer_top = L.BOARD_T + L.FLOOR_T + L.WASHER_T
        self.assertAlmostEqual(T.TCP_SEATED - T.PAD_TIPS - washer_top, 0.0005)
        self.assertAlmostEqual(T.TCP_TURN_BACK - T.PAD_TIPS - washer_top, 0.0025)
        self.assertLess(T.TCP_TURN_BACK - T.PAD_TIPS - washer_top, L.HEAD_H)
        for arm, plan in T.demo().items():
            for w in plan:
                if w.label.startswith("turn it clockwise"):
                    self.assertAlmostEqual(w.pos[2], L.TOP + T.TCP_SEATED, msg=(arm, w.label))
                    self.assertEqual(w.finger, T.GRIP_SCREW)
                elif w.label.startswith("turn back"):
                    self.assertAlmostEqual(w.pos[2], L.TOP + T.TCP_TURN_BACK, msg=(arm, w.label))
                    self.assertEqual(w.finger, T.OPEN_SCREW)

    def test_run_limit_covers_the_demo_with_margin(self):
        # The demo must end, success hold included, with at least 300 steps of the run limit to spare.
        steps = max(sum(w.duration for w in p) for p in T.demo().values()) * HZ
        limit = int(re.match(r"([\d,]+)", json.loads((HERE / "task.json").read_text())["run_limit"]).group(1)
                    .replace(",", ""))
        hold = (T.SUCCESS_HOLD_S + 1.0) * HZ  # the scorer's 1 s windows plus the task's 1 s hold
        self.assertGreaterEqual(limit - (steps + hold), 300)

    def test_trims_are_small(self):
        for dx, dy, dyaw in T.PLACE_TRIM.values():
            self.assertLess(math.hypot(dx, dy), 0.010)
            self.assertLess(abs(dyaw), 3.0)


class Text(unittest.TestCase):
    def test_task_json(self):
        j = json.loads((HERE / "task.json").read_text())
        self.assertEqual(j["instruction"], T.INSTRUCTION)
        self.assertLess(len(j["instruction"]), 200)
        self.assertTrue(j["fidelity"].startswith("Real size."))
        for k, n in (("title", 85), ("instruction", 490), ("context", 950), ("setup", 3300), ("scoring", 3150),
                     ("fidelity", 490), ("reset_notes", 580)):
            self.assertLessEqual(len(j[k]), n, k)

    def test_no_dashes_and_ascii_only(self):
        for name in ("task.json", "README.md", "REVISION.md", "task.py", "scoring.py", "layout.py", "mechkit.py",
                     "rigstate.py", "tests/test_scoring.py"):
            text = (HERE / name).read_text()
            self.assertNotIn(chr(0x2014), text, name)
            self.assertNotIn(chr(0x2013), text, name)
            self.assertTrue(all(ord(ch) < 128 for ch in text), name)

    def test_instruction_gives_no_method(self):
        low = T.INSTRUCTION.lower()
        for word in ("15", "degree", "insert", "slot", "turn", "18", "22", "holder"):
            self.assertNotIn(word, low)

    def test_kit_copies_are_version_2(self):
        self.assertEqual(mk.VERSION, "2")
        self.assertTrue(hasattr(rigstate, "released"))


if __name__ == "__main__":
    unittest.main()
