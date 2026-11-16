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
MM = 0.001


def quat(axis, deg):
    h = math.radians(deg) / 2
    v = {"x": (1, 0, 0), "y": (0, 1, 0), "z": (0, 0, 1)}[axis]
    return tuple(math.sin(h) * np.array(v, float)) + (math.cos(h),)


def qmul(a, b):
    """Hamilton product of two (x, y, z, w) quaternions: a then b is applied as a * b (b first)."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz)


def handset_at(dx=0.0, dy=0.0, dz=0.0, yaw=0.0, pitch=0.0, roll=0.0, q=None):
    """The handset's frame at the seat (over the phone's centre, on the cups' floor) plus (dx, dy, dz), turned ``yaw``
    degrees about the vertical, tipped ``pitch`` about x and ``roll`` about y (degrees, in that order of application)."""
    quat_ = q if q is not None else qmul(quat("z", yaw), qmul(quat("x", pitch), quat("y", roll)))
    o = np.array([L.PHONE_ORIGIN[0] + dx, L.PHONE_ORIGIN[1] + dy, L.SEATED_Z + dz])
    return mk.state_from_poses(T.HANDSET_PART, (*o, *quat_), source="test")


def handset_start():
    return mk.state_from_poses(T.HANDSET_PART, (*L.HANDSET_ORIGIN, *mk.yaw_quat(L.HANDSET_YAW)), source="test")


def handset_about(local_point, anchor_world, yaw=0.0, pitch=0.0):
    """The handset turned ``yaw`` (about z) and tipped ``pitch`` (about x) so that its local point lands on the world
    point ``anchor_world``."""
    q = qmul(quat("z", yaw), quat("x", pitch))
    r = S.rotation(q)
    o = np.array(anchor_world, dtype=float) - r @ np.array(local_point, dtype=float)
    return mk.state_from_poses(T.HANDSET_PART, (*o, *q), source="test")


def phone_at(dx=0.0, dy=0.0, dz=0.0, hook=0.004, q=(0, 0, 0, 1)):
    o = np.array(L.PHONE_ORIGIN) + np.array([dx, dy, dz])
    return mk.state_from_poses(T.PHONE_PART, (*o, *q), joints={L.HOOK: hook}, source="test")


def parts(handset, phone=None):
    return {S.HANDSET: handset, S.PHONE: phone or phone_at()}


def arm(tcp, opening=1.0):
    tcp = np.array(tcp, dtype=float)
    return rigstate.Arm(tcp=tcp, tips=tcp - np.array([0, 0, L.PAD_TIP]), opening=opening)


FAR_LEFT = (0.323, 0.305, L.TOP + 0.26)  # the ready pose
FAR_RIGHT = (0.323, -0.305, L.TOP + 0.26)
CLEAR = {"left": arm(FAR_LEFT), "right": arm(FAR_RIGHT)}
SEAT_TCP = (L.PHONE_ORIGIN[0], L.PHONE_ORIGIN[1] + L.GRIP_Y, L.SEATED_Z + L.GRIP_Z)


def holding(tcp=SEAT_TCP, opening=0.3):
    return {"left": arm(FAR_LEFT), "right": arm(tcp, opening)}


def run(seq, tracker=None):
    """Feed (states, arms) pairs to a tracker; return (last result, tracker)."""
    t = tracker or S.Tracker(HZ)
    r = None
    for st, ar in seq:
        r = t.update(st, ar)
    return r, t


_DEFAULT = object()


def still(states, n=WIN + 5, arms=_DEFAULT):
    return [(states, CLEAR if arms is _DEFAULT else arms)] * n


def full_success_states():
    return parts(handset_at())


class Layout(unittest.TestCase):
    def test_pads_fit_their_own_cups_with_8mm_across_and_6mm_each_end(self):
        for name in ("ear", "mouth"):
            (_, (px, py), _h) = L.PADS[name]
            (_, (cx, cy)) = L.CUPS[name]
            self.assertAlmostEqual((cx - px) / 2, 0.008, places=6, msg=name)
            self.assertAlmostEqual((cy - py) / 2, 0.006, places=6, msg=name)
        # same centres: a pad sits centred in its own cup when the handset's frame is over the cradle's centre
        for name in ("ear", "mouth"):
            self.assertEqual(L.PADS[name][0], L.CUPS[name][0])

    def test_the_mouthpiece_pad_does_not_fit_the_earpiece_cup(self):
        """The handset seats one way round: the longer pad (70 mm) is longer than the earpiece cup (58 mm)."""
        self.assertGreater(L.MOUTH_PAD[1][1], L.EAR_CUP[1][1])
        self.assertGreater(L.MOUTH_PAD[1][0], L.EAR_PAD[1][0])  # and it is wider across too
        # and the earpiece pad, turned end for end, lands 5 mm off the mouthpiece cup's centre while the mouthpiece
        # pad then lands on the earpiece cup's wall: reversed, nothing seats (see Seat.test_reversed)

    def test_geometry_numbers_in_the_text(self):
        self.assertAlmostEqual(L.HANDSET_LEN, 0.197, places=6)
        self.assertAlmostEqual(L.HANDLE_Z0, 0.014)
        self.assertAlmostEqual(L.HANDLE_Z1 - L.HANDLE_Z0, 0.034)
        self.assertAlmostEqual(L.HANDLE_X, 0.034)
        self.assertAlmostEqual(L.FLOOR_Z, 0.045)
        self.assertAlmostEqual(L.HOOK_FREE_TOP, 0.063)
        self.assertEqual(L.PHONE_SIZE, (0.200, 0.240))
        # the handle's underside at the seat is 59 mm up, 4 mm over the top and 4 mm under the plunger's free top
        under = L.FLOOR_Z + L.HANDLE_Z0
        self.assertAlmostEqual(under - L.PHONE_H, 0.004)
        self.assertAlmostEqual(L.HOOK_FREE_TOP - under, 0.004)
        # on the flat top (pads 10 mm higher) it would clear the plunger by 6 mm: the line stays open
        self.assertAlmostEqual((L.PHONE_H + L.HANDLE_Z0) - L.HOOK_FREE_TOP, 0.006)

    def test_hook_switch_spring_is_lighter_than_the_handset(self):
        weight = L.HANDSET_MASS * 9.81
        self.assertLess(L.HOOK_SPRING * L.HOOK_TRAVEL, weight)
        self.assertLess(L.HOOK_SPRING * 0.004, 0.6 * weight)
        self.assertGreater(L.HOOK_TRAVEL, 0.004 + 0.002)  # room to be pressed past the seat

    def test_hook_sits_on_the_flat_top_between_the_cups(self):
        y = L.HOOK_XY[1]
        self.assertGreater(y - L.HOOK_RADIUS, L.MOUTH_CUP[0][1] + L.MOUTH_CUP[1][1] / 2)
        self.assertLess(y + L.HOOK_RADIUS, L.EAR_CUP[0][1] - L.EAR_CUP[1][1] / 2)
        # under the handle (x +-17 mm, y -70..+60 mm in the handset's frame)
        self.assertLess(abs(L.HOOK_XY[0]) + L.HOOK_RADIUS, L.HANDLE_X / 2)
        self.assertTrue(L.HANDLE_Y0 < y < L.HANDLE_Y1)

    def test_reach_and_clearances(self):
        # the carried handset's pad bottoms clear the phone's top by 20 mm or more
        carry_pad = T.CARRY_TCP - L.GRIP_Z - L.TOP
        self.assertGreaterEqual(carry_pad - L.PHONE_H, 0.015)
        # at the seat the pad tips (13 mm under the TCP) stay over the phone's top and the plunger's free top
        tip_seat = T.SET_TCP - L.PAD_TIP - L.TOP
        self.assertGreaterEqual(tip_seat - L.PHONE_H, 0.006)
        # the pad tips at the pick stay off the desk
        self.assertGreaterEqual(T.GRIP_TCP - L.PAD_TIP - L.TOP, 0.015)
        # the hover clears the handle's top by 50 mm or more
        self.assertGreaterEqual(T.HOVER - L.PAD_TIP - (L.TOP + L.HANDLE_Z1), 0.050)
        # the pads (22 mm long) sit inside the handle's height 14..48 mm at the pick and at the seat
        self.assertGreaterEqual(L.GRIP_Z - L.PAD_TIP, L.HANDLE_Z0)
        self.assertLessEqual(L.GRIP_Z + 0.009, L.HANDLE_Z1)

    def test_grip_clears_the_pads_and_the_plunger(self):
        # the fingers (27 mm wide along the handle) at the grip point stay clear of the mouthpiece pad's inner end
        mouth_end = L.MOUTH_PAD[0][1] + L.MOUTH_PAD[1][1] / 2  # -0.032
        self.assertGreaterEqual((L.GRIP_Y - 0.0135) - mouth_end, 0.005)
        self.assertLess(L.GRIP_Y + 0.0135, L.EAR_PAD[0][1] - L.EAR_PAD[1][1] / 2)
        # the gripper opens 94 mm: the handle's 34 mm leaves 30 mm each side; the widest part (pad, 52 mm) fits too
        self.assertLessEqual(L.MOUTH_PAD[1][0], 0.080)
        # across the handle the fingers (x +-13..17 mm) stay 6 mm or more from the plunger's side at the seat
        self.assertGreaterEqual(L.GRIP_W / 2 - L.HOOK_RADIUS, 0.005)

    def test_parts_do_not_overlap_at_the_start(self):
        specs = {s.name: s for s in T.OBJECTS}
        boxes = {n: s.world_bounds() for n, s in specs.items()}

        def xy_overlap(a, b):
            (alo, ahi), (blo, bhi) = boxes[a], boxes[b]
            return bool(np.all(alo[:2] < bhi[:2]) and np.all(blo[:2] < ahi[:2]))

        for a, b in (("handset", "phone"), ("handset", "pen_pot"), ("handset", "monitor"), ("phone", "pen_pot"),
                     ("phone", "monitor"), ("pen_pot", "monitor")):
            self.assertFalse(xy_overlap(a, b), (a, b))
        # at least 25 mm between the handset and the phone's right side (room for a finger and a pad)
        hs_hi = boxes["handset"][1][1]
        ph_lo = boxes["phone"][0][1]
        self.assertGreaterEqual(abs(hs_hi - ph_lo), 0.050)

    def test_start_is_clear_of_the_ready_pose_and_the_arms(self):
        # nothing under the arms' ready poses (TCP 0.26 m up): the handset's pick point is 8.5 cm from the right base's line
        self.assertGreaterEqual(min(p[0] for p in (T.PICK_XY, T.PLACE_XY)), 0.20)
        lo, hi = T.HANDSET_PART.world_bounds()
        self.assertGreaterEqual(float(lo[0]), 0.20)
        self.assertLessEqual(float(hi[0]), 0.45)

    def test_stability(self):
        # the handset on its two pads at the start: its centre of mass is between them, so it stands on them
        com = T.HANDSET_PART.total_center_of_mass()
        self.assertGreater(com[1], L.MOUTH_PAD[0][1] + L.MOUTH_PAD[1][1] / 2 - 0.040)
        self.assertLess(com[1], L.EAR_PAD[0][1] + L.EAR_PAD[1][1] / 2)
        self.assertLess(abs(com[0]), 0.001)
        self.assertAlmostEqual(com[1], L.GRIP_Y, delta=0.003)  # held at (about) its centre of mass
        # tipping about a pad's long edge: the phone's 24 cm wide base is far from tipping
        pcom = T.PHONE_PART.total_center_of_mass()
        self.assertGreater(math.degrees(math.atan2(L.PHONE_SIZE[0] / 2, pcom[2])), 60.0)
        self.assertLess(float(np.hypot(pcom[0], pcom[1])), 0.002)

    def test_masses_and_shapes(self):
        self.assertEqual((T.HANDSET_PART.mass, T.PHONE_PART.mass), (0.17, 1.10))
        self.assertIsNone(T.MONITOR.mass)
        self.assertIsNone(T.PEN_POT.mass)
        self.assertEqual(len(T.HANDSET_PART.shapes), 3)  # two pads and the handle
        lo, hi = T.HANDSET_PART.local_bounds()
        self.assertAlmostEqual(float(lo[2]), 0.0)  # the frame sits on the pad-bottom plane
        self.assertAlmostEqual(float(hi[2]), L.HANDLE_Z1)
        self.assertAlmostEqual(float(hi[1] - lo[1]), L.HANDSET_LEN)
        lo, hi = T.PHONE_PART.local_bounds()
        self.assertAlmostEqual(float(lo[2]), 0.0)
        self.assertAlmostEqual(float(hi[2]), L.HOOK_FREE_TOP)  # the plunger's free top is the highest point
        self.assertEqual([j.name for j in T.PHONE_PART.joints], [L.HOOK])
        j = T.PHONE_PART.joints[0]
        self.assertEqual((j.kind, j.axis, j.upper, j.lower, j.spring), ("prismatic", "-Z", L.HOOK_TRAVEL, 0.0, L.HOOK_SPRING))

    def test_cups_are_cut_in_the_top_plate(self):
        """No shape of the phone stands inside either cup above its floor."""
        for s in T.PHONE_PART.shapes:
            if s.center[2] < L.FLOOR_Z:
                continue
            for name in ("ear", "mouth"):
                (cx, cy), (sx, sy) = L.CUPS[name]
                inside_x = abs(s.center[0] - cx) < sx / 2 - 1e-6 + s.size[0] / 2 and \
                    abs(s.center[0] - cx) + s.size[0] / 2 <= sx / 2 + 1e-6
                inside_y = abs(s.center[1] - cy) + s.size[1] / 2 <= sy / 2 + 1e-6
                self.assertFalse(inside_x and inside_y, f"{s.center} {s.size} stands in the {name} cup")
        # the top plate still covers the flat top between the cups
        flat = [s for s in T.PHONE_PART.shapes if s.center[2] > L.FLOOR_Z and abs(s.center[0]) < 0.001
                and abs(s.center[1] - 0.0085) < 0.001]
        self.assertTrue(flat)


class Seat(unittest.TestCase):
    def read(self, handset, phone=None):
        return S.read(parts(handset, phone))

    def pads(self, handset, phone=None):
        r = self.read(handset, phone)
        return S.pad_reading(r, "ear"), S.pad_reading(r, "mouth"), S.attitude(r), r.hook

    def seated(self, handset, phone=None):
        ear, mouth, a, hook = self.pads(handset, phone)
        return S.seated(ear, mouth, a, hook)

    def test_seated_centre(self):
        ear, mouth, a, _ = self.pads(handset_at())
        self.assertTrue(self.seated(handset_at()))
        self.assertAlmostEqual(ear["over"], -0.006, places=6)  # the tighter (y) side: 6 mm to the wall
        self.assertAlmostEqual(mouth["over"], -0.006, places=6)
        self.assertAlmostEqual(ear["dz_min"], 0.0, places=9)
        self.assertAlmostEqual(a["tilt"], 0.0, places=6)
        self.assertAlmostEqual(ear["centre_off"], 0.0, places=9)

    def test_height_window(self):
        for dz, ok in ((-0.003, False), (-0.002, True), (0.0, True), (0.003, True), (0.0035, False), (0.010, False)):
            self.assertEqual(self.seated(handset_at(dz=dz)), ok, dz)

    def test_on_the_top_is_not_seated(self):
        self.assertFalse(self.seated(handset_at(dz=L.CUP_DEPTH)))

    def test_lateral_offset_limits(self):
        # 8 mm clear each side across (x) and 6 mm at each end (y): 7.5 / 5.5 mm offsets still fit, 8.6 / 6.6 mm do not
        for dx, dy, ok in ((0.0075, 0, True), (-0.0075, 0, True), (0, 0.0055, True), (0, -0.0055, True),
                           (0.0086, 0, False), (-0.0086, 0, False), (0, 0.0066, False), (0, -0.0066, False),
                           (0.0075, 0.0055, True), (0.02, 0, False)):
            self.assertEqual(self.seated(handset_at(dx=dx, dy=dy)), ok, (dx, dy))

    def test_yaw_and_tilt_limits(self):
        self.assertTrue(self.seated(handset_at(yaw=2.0)))
        self.assertTrue(self.seated(handset_at(yaw=-2.0)))
        self.assertFalse(self.seated(handset_at(yaw=8.0)))  # the pads no longer fit and the heading is off
        for pitch in (1.0, 2.0):
            ear, mouth, a, hook = self.pads(handset_at(pitch=pitch))
            self.assertLessEqual(a["tilt"], L.SEAT_TILT_DEG)
        a = S.attitude(self.read(handset_at(pitch=5.0)))
        self.assertAlmostEqual(a["tilt"], 5.0, places=4)
        self.assertGreater(a["tilt"], L.SEAT_TILT_DEG)
        a = S.attitude(self.read(handset_at(yaw=7.0)))
        self.assertAlmostEqual(a["yaw"], 7.0, places=4)

    def test_the_cups_themselves_limit_the_turn_inside_the_6_degree_window(self):
        """Turned about the cradle's centre the pad corners reach the cup walls at about 4 degrees, so the heading rule
        (6 degrees) never decides on its own: it is the stated tolerance for a real trial."""
        ok = [d for d in np.arange(0.0, 10.01, 0.25) if self.seated(handset_at(yaw=float(d)))]
        self.assertGreater(max(ok), 2.0)
        self.assertLess(max(ok), L.SEAT_YAW_DEG)
        self.assertTrue(self.seated(handset_at(yaw=2.0)))
        self.assertFalse(self.seated(handset_at(yaw=6.5)))

    def test_seated_in_a_nudged_phone(self):
        """Everything is judged in the phone's own frame: a phone turned 3 degrees with the handset still in it."""
        ph = phone_at(dx=0.004, q=quat("z", 3.0))
        po = np.array(L.PHONE_ORIGIN) + np.array([0.004, 0.0, 0.0])
        q = quat("z", 3.0)
        ho = po + np.array([0, 0, L.FLOOR_Z])
        h = mk.state_from_poses(T.HANDSET_PART, (*ho, *q), source="test")
        self.assertTrue(self.seated(h, ph))

    def test_reversed_is_not_seated(self):
        """Turned end for end (yaw 180): the earpiece pad is over the mouthpiece cup and the mouthpiece pad over the
        earpiece cup's wall: nothing seats at any offset along y."""
        for dy in np.arange(-0.012, 0.0121, 0.003):
            for dx in (0.0, 0.003, -0.003):
                h = handset_at(dx=dx, dy=float(dy), yaw=180.0)
                ear, mouth, _, _ = self.pads(h)
                self.assertFalse(S.in_cup(ear) and S.in_cup(mouth), (dx, dy))
                self.assertFalse(self.seated(h), (dx, dy))
        # the pads can seat at the far offset where the earpiece pad is in the mouthpiece cup, but each is then in the
        # WRONG cup for the scorer
        h = handset_at(yaw=180.0, dy=0.005)
        ear, mouth, _, _ = self.pads(h)
        self.assertFalse(S.in_cup(ear))
        self.assertFalse(S.in_cup(mouth))

    def test_upside_down_is_not_seated(self):
        self.assertFalse(self.seated(handset_at(roll=180.0, dz=0.05)))

    def test_hook_switch_must_be_down(self):
        for hook, ok in ((0.0, False), (0.001, False), (0.0019, False), (0.002, True), (0.004, True), (0.008, True)):
            self.assertEqual(self.seated(handset_at(), phone_at(hook=hook)), ok, hook)

    def test_missing_hook_is_unreadable(self):
        ph = phone_at()
        object.__setattr__(ph, "joints", {})
        self.assertIsNone(S.read(parts(handset_at(), ph)))


class Rests(unittest.TestCase):
    def read(self, handset, phone=None):
        return S.read(parts(handset, phone))

    def rest(self, handset):
        r = self.read(handset)
        return S.rests(S.pad_reading(r, "ear")), S.rests(S.pad_reading(r, "mouth"))

    def test_on_the_flat_top_and_in_the_cups_rest(self):
        self.assertEqual(self.rest(handset_at()), (True, True))
        self.assertEqual(self.rest(handset_at(dz=L.CUP_DEPTH)), (True, True))
        self.assertEqual(self.rest(handset_at(dz=0.0139)), (True, True))
        self.assertEqual(self.rest(handset_at(dz=0.0145)), (False, False))

    def test_beside_the_phone_or_on_the_desk_does_not(self):
        self.assertEqual(self.rest(handset_start()), (False, False))
        self.assertEqual(self.rest(handset_at(dy=0.25)), (False, False))
        self.assertEqual(self.rest(handset_at(dz=-0.045)), (False, False))  # on the desk through the phone's footprint

    def test_pad_hanging_over_the_phone_edge_does_not(self):
        # shifted 21 mm toward -y: the mouthpiece pad's outer end is 3 mm past the phone's end wall
        r = self.read(handset_at(dy=-0.021, dz=L.CUP_DEPTH))
        self.assertTrue(S.rests(S.pad_reading(r, "ear")))
        mouth = S.pad_reading(r, "mouth")
        self.assertAlmostEqual(mouth["off_top"], 0.003, places=6)
        self.assertFalse(S.rests(mouth))
        # 2 mm inside the wall it rests
        r = self.read(handset_at(dy=-0.0135, dz=L.CUP_DEPTH))
        self.assertTrue(S.rests(S.pad_reading(r, "mouth")))

    def test_pad_readings_json_clean(self):
        r = self.read(handset_at())
        json.dumps(S.pad_reading(r, "ear"))


class Tiers(unittest.TestCase):
    def final(self, states, arms=_DEFAULT, n=WIN + 5):
        r, _ = run(still(states, n, arms))
        return r

    def across_the_top(self):
        """Lying across the phone along x, on the flat top (pads 10 mm up), neither pad in a cup."""
        q = qmul(quat("z", 90.0), (0, 0, 0, 1))
        o = np.array([L.PHONE_ORIGIN[0] - 0.0035, L.PHONE_ORIGIN[1], L.TOP + L.PHONE_H])
        # rotating by +90 about z sends the handset's +y to world -x: the earpiece end is toward -x, inside the top
        return mk.state_from_poses(T.HANDSET_PART, (*o, *q), source="test")

    def one_end_on_the_rim(self):
        """The earpiece pad in its cup, the handset turned 8 degrees about it, the mouthpiece pad up on the cup's rim
        and the top (10 mm higher), the handset tipped 4 degrees."""
        anchor = (L.PHONE_ORIGIN[0], L.PHONE_ORIGIN[1] + 0.072, L.SEATED_Z)
        return handset_about((0.0, 0.072, 0.0), anchor, yaw=8.0, pitch=-math.degrees(math.atan2(0.010, 0.139)))

    def test_full_score(self):
        r = self.final(full_success_states())
        self.assertEqual(r.score, 1.0)
        self.assertTrue(r.success)

    def test_do_nothing_scores_zero(self):
        r = self.final(parts(handset_start()))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_not_in_the_first_second(self):
        """The 1 s windows must fill before anything scores."""
        t = S.Tracker(HZ)
        results = [t.update(*pair) for pair in still(full_success_states(), WIN + 3)]
        self.assertEqual([x.score for x in results[:WIN - 1]], [0.0] * (WIN - 1))
        self.assertEqual(results[WIN].score, 1.0)

    def test_quarter_across_the_top(self):
        st = parts(self.across_the_top(), phone_at(hook=0.0))
        r = self.final(st)
        self.assertEqual(r.score, 0.25)
        self.assertFalse(r.details["ear_in_cup"] or r.details["mouth_in_cup"])
        self.assertTrue(r.details["ear_rests"] and r.details["mouth_rests"])

    def test_half_one_end_in_its_cup_the_other_on_the_rim(self):
        h = self.one_end_on_the_rim()
        st = parts(h, phone_at(hook=0.0))
        r = self.final(st)
        self.assertTrue(r.details["ear_in_cup"], r.details)
        self.assertFalse(r.details["mouth_in_cup"])
        self.assertTrue(r.details["mouth_rests"], r.details)
        self.assertEqual(r.score, 0.5)

    def test_both_pads_in_their_cups_but_turned_4_degrees_too_far_out_of_level_is_a_half(self):
        r = self.final(parts(handset_at(q=quat("x", 4.0))))
        self.assertFalse(r.details["seated"])
        self.assertIn(r.score, (0.5, 0.25, 0.0))

    def test_half_both_in_their_cups_but_the_hook_is_up(self):
        """Both pads in their cups but the switch not down (a phone with the plunger stuck): not 1.0, 0.5."""
        r = self.final(parts(handset_at(), phone_at(hook=0.0)))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.details["line_closed"])

    def test_half_both_in_their_cups_but_turned_8_degrees(self):
        r = self.final(parts(handset_at(yaw=8.0)))
        self.assertIn(r.score, (0.5, 0.25, 0.0))
        self.assertLess(r.score, 1.0)

    def test_wrong_way_round_is_a_quarter_at_best(self):
        h = handset_at(yaw=180.0, dy=0.005)
        r = self.final(parts(h))
        self.assertLessEqual(r.score, 0.25)
        self.assertFalse(r.success)

    def test_handset_on_the_desk_beside_the_phone_is_zero(self):
        for h in (handset_start(), handset_at(dy=-0.15, dz=-0.045)):
            self.assertEqual(self.final(parts(h)).score, 0.0)

    def test_outcomes_are_only_the_four_tiers(self):
        for st in (full_success_states(), parts(handset_start()), parts(self.across_the_top(), phone_at(hook=0.0)),
                   parts(self.one_end_on_the_rim(), phone_at(hook=0.0)), parts(handset_at(yaw=8.0)),
                   parts(handset_at(dz=0.0145)), parts(handset_at(yaw=180.0))):
            self.assertIn(self.final(st).score, (0.0, 0.25, 0.5, 1.0))


class Release(unittest.TestCase):
    def test_hand_on_the_handset_earns_nothing(self):
        r, _ = run(still(full_success_states(), WIN + 5, holding()))
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["released_1s"])

    def test_open_hand_beside_it_earns_nothing(self):
        # an open gripper's zone is 56.6 mm round its 38 mm axis. Beside the phone's +x end wall (x 0.40, 30 mm
        # from the handle's far side too) the phone is the nearest part: the hand releases only 30 mm clear of it
        for dist, expect in ((0.010, 0.0), (0.025, 0.0), (0.040, 1.0)):
            tcp = (L.PHONE_ORIGIN[0] + 0.10 + 0.0566 + dist, L.PHONE_ORIGIN[1], L.TOP + 0.05)
            ar = {"left": arm(FAR_LEFT), "right": arm(tcp, 1.0)}
            r, _ = run(still(full_success_states(), WIN + 5, ar))
            self.assertEqual(r.score, expect, dist)

    def test_hand_on_the_phone_blocks_it_too(self):
        # a gripper resting on the phone's side wall, 100 mm from the handset but touching the phone
        ar = {"left": arm(FAR_LEFT), "right": arm((L.PHONE_ORIGIN[0] - 0.10 - 0.05, L.PHONE_ORIGIN[1] + 0.09,
                                                   L.TOP + 0.04), 1.0)}
        r, _ = run(still(full_success_states(), WIN + 5, ar))
        self.assertEqual(r.score, 0.0)

    def test_unreadable_arms_never_release(self):
        for bad in (None, {"left": arm(FAR_LEFT)}, {"left": arm(FAR_LEFT), "right": arm((math.nan, 0, 0))}):
            r, _ = run(still(full_success_states(), WIN + 5, bad))
            self.assertEqual(r.score, 0.0, bad)

    def test_release_then_the_window_restarts(self):
        seq = still(full_success_states(), 10, holding()) + still(full_success_states(), WIN - 2)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)  # clear for only WIN - 2 steps
        r, _ = run(seq + still(full_success_states(), 3))
        self.assertEqual(r.score, 1.0)


class Stillness(unittest.TestCase):
    def test_a_creeping_handset_is_not_still(self):
        seq = [(parts(handset_at(dx=0.0004 * i)), CLEAR) for i in range(WIN + 5)]  # 0.4 mm a step: 8 mm a second
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["still_1s"])

    def test_a_slow_drift_under_5mm_is_still(self):
        seq = [(parts(handset_at(dx=0.0002 * i)), CLEAR) for i in range(WIN + 5)]  # 0.2 mm a step: 4 mm a second
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)

    def test_a_turning_handset_is_not_still(self):
        seq = [(parts(handset_at(yaw=0.4 * i)), CLEAR) for i in range(WIN + 5)]  # 8 degrees a second
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_the_phone_creeping_is_not_still(self):
        seq = [(parts(handset_at(dy=0.0004 * i), phone_at(dy=0.0004 * i)), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_the_phone_wobbling_is_not_still(self):
        seq = [(parts(handset_at(), phone_at(q=quat("z", 8.0 * (i % 2)))), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)  # 8 degrees back and forth: over the 5 degree limit
        seq = [(parts(handset_at(), phone_at(q=quat("x", 2.0 * (i % 2)))), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)  # 2 degrees is under it

    def test_the_hook_bouncing_is_not_what_stillness_reads(self):
        """The plunger's own jitter is not a part moving; the handset and the phone's frame are what must be still."""
        seq = [(parts(handset_at(), phone_at(hook=0.004 + 0.0004 * (i % 3))), CLEAR) for i in range(WIN + 5)]
        r, _ = run(seq)
        self.assertEqual(r.score, 1.0)


class Latches(unittest.TestCase):
    def test_tilting_it_on_the_way_is_not_a_fault(self):
        """How the handset is tilted while it is carried is not judged: only where it ends up."""
        st = [parts(handset_at(dz=0.05, q=quat("x", 40.0)))] + [full_success_states()] * (WIN + 10)
        r, t = run([(s, CLEAR) for s in st])
        self.assertNotIn("F1", t.latch)
        self.assertEqual(r.score, 1.0)

    def test_handset_tilted_under_30_is_fine(self):
        st = [parts(handset_at(dz=0.05, q=quat("x", 25.0)))] + [full_success_states()] * (WIN + 10)
        r, t = run([(s, CLEAR) for s in st])
        self.assertFalse(t.latched)
        self.assertEqual(r.score, 1.0)

    def test_a_part_off_the_desk_zeroes(self):
        low_handset = handset_at(dz=-0.045 - 0.012)
        r, t = run([(parts(low_handset), CLEAR)] + still(full_success_states()))
        self.assertEqual(t.latch["F2"], 1)
        self.assertEqual(r.score, 0.0)

    def test_the_phone_off_the_desk_zeroes(self):
        r, t = run([(parts(handset_at(), phone_at(dz=-0.012)), CLEAR)] + still(full_success_states()))
        self.assertEqual(t.latch["F2"], 1)
        self.assertEqual(r.score, 0.0)

    def test_a_part_just_below_the_desk_top_is_fine(self):
        r, t = run([(parts(handset_at(), phone_at(dz=-0.009)), CLEAR)] + still(full_success_states()))
        self.assertIsNone(t.latch["F2"])
        self.assertEqual(r.score, 1.0)

    def test_phone_shoved_zeroes_even_if_pushed_back(self):
        r, t = run([(parts(handset_at(dx=0.010), phone_at(dx=0.010)), CLEAR)] + still(full_success_states()))
        self.assertEqual(t.latch["F3"], 1)
        self.assertEqual(r.score, 0.0)

    def test_phone_moved_under_8mm_is_fine(self):
        h = handset_at(dx=0.007)
        r, t = run(still(parts(h, phone_at(dx=0.007))))
        self.assertIsNone(t.latch["F3"])
        self.assertEqual(r.score, 1.0)

    def test_phone_tilted_over_3_degrees_zeroes(self):
        r, t = run([(parts(handset_at(), phone_at(q=quat("y", 3.5))), CLEAR)] + still(full_success_states()))
        self.assertEqual(t.latch["F3"], 1)
        self.assertEqual(r.score, 0.0)


class BadStates(unittest.TestCase):
    def nan_states(self):
        bad = handset_at()
        object.__setattr__(bad, "origin", np.array([math.nan, 0.0, 0.0]))
        return parts(bad)

    def test_nan_scores_zero_and_restarts_the_window(self):
        t = S.Tracker(HZ)
        for pair in still(full_success_states(), WIN + 5):
            t.update(*pair)
        self.assertEqual(t.last.score, 1.0)
        r = t.update(self.nan_states(), CLEAR)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 1)
        for pair in still(full_success_states(), WIN):
            r = t.update(*pair)
        self.assertEqual(r.score, 0.0)  # the 1 s window has to refill
        r = t.update(full_success_states(), CLEAR)
        r = t.update(full_success_states(), CLEAR)
        self.assertEqual(r.score, 1.0)

    def test_nan_quaternion_and_low(self):
        for field_, value in (("quat_xyzw", np.array([math.nan, 0, 0, 1.0])), ("low", np.array([0, 0, math.nan])),
                              ("quat_xyzw", np.zeros(4))):
            bad = handset_at()
            object.__setattr__(bad, field_, value)
            r, t = run([(parts(bad), CLEAR)])
            self.assertEqual(r.score, 0.0)
            self.assertEqual(t.bad_steps, 1, field_)

    def test_nan_or_inf_hook(self):
        for v in (math.nan, math.inf):
            r, t = run([(parts(handset_at(), phone_at(hook=v)), CLEAR)])
            self.assertEqual(r.score, 0.0)
            self.assertEqual(t.bad_steps, 1)

    def test_inf_state(self):
        bad = phone_at()
        object.__setattr__(bad, "origin", np.array([math.inf, 0.0, 0.0]))
        r, t = run([(parts(handset_at(), bad), CLEAR)])
        self.assertEqual(r.score, 0.0)
        self.assertEqual(t.bad_steps, 1)

    def test_missing_part_and_junk(self):
        t = S.Tracker(HZ)
        self.assertEqual(t.update({S.HANDSET: handset_at()}, CLEAR).score, 0.0)
        self.assertEqual(t.update(None, CLEAR).score, 0.0)
        self.assertEqual(t.update({S.HANDSET: None, S.PHONE: None}, CLEAR).score, 0.0)
        self.assertEqual(t.bad_steps, 3)

    def test_nan_latches_nothing(self):
        t = S.Tracker(HZ)
        t.update(self.nan_states(), CLEAR)
        self.assertEqual(t.latch, {"F2": None, "F3": None})

    def test_details_are_json_clean(self):
        r, _ = run(still(full_success_states(), WIN + 2))
        json.dumps(r.details)
        r2, _ = run([(self.nan_states(), CLEAR)])
        json.dumps(r2.details)


class Resets(unittest.TestCase):
    def test_a_new_episode_clears_the_record(self):
        fallen = handset_at(dz=-0.045 - 0.012)
        t = S.Tracker(HZ)
        run(still(parts(fallen), 3), t)
        self.assertEqual(t.latch["F2"], 1)
        t.update(parts(handset_start()), CLEAR)  # the simulator puts every part back on its mark in one step
        self.assertEqual(t.resets, 1)
        self.assertIsNone(t.latch["F2"])

    def test_a_handset_set_back_on_its_mark_by_hand_is_not_a_reset(self):
        """Only if the phone is on its mark too, and a part jumped 8 cm in one step."""
        t = S.Tracker(HZ)
        run(still(parts(handset_at(dz=-0.045 - 0.012)), 3), t)
        t.update(parts(handset_start(), phone_at(dx=0.02)), CLEAR)
        self.assertEqual(t.resets, 0)
        self.assertIsNotNone(t.latch["F2"])

    def test_a_slow_return_is_not_a_reset(self):
        t = S.Tracker(HZ)
        run(still(parts(handset_at(dz=-0.045 - 0.012)), 3), t)
        for i in range(1, 11):
            f = i / 10
            o = (1 - f) * np.array([L.PHONE_ORIGIN[0], L.PHONE_ORIGIN[1], L.SEATED_Z + 0.05]) + f * np.array(L.HANDSET_ORIGIN)
            q = mk.yaw_quat(f * L.HANDSET_YAW)
            t.update(parts(mk.state_from_poses(T.HANDSET_PART, (*o, *q), source="test")), CLEAR)
        self.assertEqual(t.resets, 0)
        self.assertEqual(t.latch["F2"], 1)


class Exploits(unittest.TestCase):
    """Things a robot (or a lazy script) could try for credit it should not get."""

    def test_holding_the_handset_in_the_cups_is_not_a_hang_up(self):
        r, _ = run(still(full_success_states(), 3 * WIN, holding()))
        self.assertEqual(r.score, 0.0)

    def test_dragging_the_phone_to_the_handset_scores_zero(self):
        """Pushing the phone 12 cm toward the handset so that the cups are under it: the phone moves over 8 mm."""
        seq = []
        for i in range(0, 13):
            seq.append((parts(handset_start(), phone_at(dy=-0.01 * i)), CLEAR))
        seq += still(parts(handset_at(dy=-0.12), phone_at(dy=-0.12)), WIN + 5)
        r, t = run(seq)
        self.assertIsNotNone(t.latch["F3"])
        self.assertEqual(r.score, 0.0)

    def test_pressing_the_hook_switch_by_hand_with_the_handset_elsewhere(self):
        """The line closed alone is worth nothing: the handset is not on the phone."""
        r, _ = run(still(parts(handset_start(), phone_at(hook=0.006))))
        self.assertEqual(r.score, 0.0)

    def test_hook_closed_but_the_handset_on_the_desk_in_front_of_the_phone(self):
        r, _ = run(still(parts(handset_at(dx=-0.18, dz=-0.045), phone_at(hook=0.006))))
        self.assertEqual(r.score, 0.0)

    def test_laying_the_handset_across_the_top_with_the_hook_held_down_is_a_quarter(self):
        h = Tiers().across_the_top()
        r, _ = run(still(parts(h, phone_at(hook=0.006))))
        self.assertEqual(r.score, 0.25)

    def test_dropping_the_handset_on_its_back_scores_zero(self):
        r, _ = run(still(parts(handset_at(dz=0.04, roll=180.0)), 2 * WIN))
        self.assertEqual(r.score, 0.0)

    def test_tossing_it_in_from_a_height_that_tips_it_scores_zero(self):
        seq = still(parts(handset_at(dz=0.04, q=quat("y", 45.0))), 3 * WIN)
        r, _ = run(seq)
        self.assertEqual(r.score, 0.0)

    def test_handset_turned_end_for_end_cannot_reach_a_half(self):
        best = 0.0
        for dy in np.arange(-0.02, 0.0201, 0.0025):
            for dx in np.arange(-0.02, 0.0201, 0.005):
                for dz in (0.0, 0.010, 0.012):
                    r, _ = run(still(parts(handset_at(dx=float(dx), dy=float(dy), dz=dz, yaw=180.0)), WIN + 3))
                    best = max(best, r.score)
        self.assertLessEqual(best, 0.25)

    def test_success_needs_the_hold_on_top_of_the_score(self):
        t = S.Tracker(HZ)
        first = None
        for i, pair in enumerate(still(full_success_states(), 3 * WIN)):
            r = t.update(*pair)
            if r.success and first is None:
                first = i
        self.assertEqual(first, WIN)  # not before 1 s of readings, then the task file's 1 s hold on top


class OfflineStandIn(unittest.TestCase):
    def tcp_at(self, o, yaw, dz=None):
        r = np.array(L.yaw_matrix(yaw))
        xy = np.array(o[:2]) + r @ np.array([0.0, L.GRIP_Y])
        return (float(xy[0]), float(xy[1]), float(o[2] + (L.GRIP_Z if dz is None else dz)))

    def test_support_heights(self):
        m = T.OfflineModel()
        o = np.array(L.PHONE_ORIGIN)
        self.assertAlmostEqual(m.support(o, 0.0), L.SEATED_Z)
        self.assertAlmostEqual(m.support(o + np.array([0.005, 0.005, 0.0]), 0.0), L.SEATED_Z)
        self.assertAlmostEqual(m.support(o + np.array([0.012, 0.0, 0.0]), 0.0), L.TOP + L.PHONE_H)  # on the top
        self.assertAlmostEqual(m.support(o + np.array([0.0, 0.0, 0.0]), math.pi), L.TOP + L.PHONE_H)  # end for end
        self.assertAlmostEqual(m.support(np.array([0.30, 0.25, 0.0]), 0.0), L.TOP)  # beside the phone

    def test_hook_follows_the_handle(self):
        m = T.OfflineModel()
        m.o = np.array([*L.PHONE_ORIGIN[:2], L.SEATED_Z])
        m.yaw = 0.0
        self.assertAlmostEqual(m.hook(), 0.004)
        m.o[2] = L.TOP + L.PHONE_H  # on the flat top: the handle is 6 mm over the plunger's free top
        self.assertAlmostEqual(m.hook(), 0.0)
        m.o[2] = L.SEATED_Z - 0.010
        self.assertAlmostEqual(m.hook(), L.HOOK_TRAVEL)  # stopped at the flush position
        m.o[:2] = [0.30, 0.25]
        self.assertAlmostEqual(m.hook(), 0.0)

    def test_grab_carry_turn_release(self):
        m = T.OfflineModel()
        m.plans = {}  # no demo plan: no turn
        hold = arm(self.tcp_at(m.o, m.yaw), 0.3)
        m.step({"left": arm(FAR_LEFT), "right": hold})
        self.assertEqual(m.held[0], "right")
        up = arm((SEAT_TCP[0], SEAT_TCP[1], L.SEATED_Z + L.GRIP_Z + 0.02), 0.3)
        m.step({"left": arm(FAR_LEFT), "right": up})
        want = np.array(m.o[:2]) - np.array(L.yaw_matrix(m.yaw)) @ np.array([0.0, L.GRIP_Y]) + 0 * m.o[:2]
        np.testing.assert_allclose(np.array(up.tcp[:2]) - np.array(L.yaw_matrix(m.yaw)) @ np.array([0.0, L.GRIP_Y]),
                                   m.o[:2], atol=1e-9)
        self.assertAlmostEqual(float(m.o[2]), L.SEATED_Z + 0.02, places=9)

    def test_release_over_the_cups_seats_and_presses_the_hook(self):
        m = T.OfflineModel()
        m.plans = {}
        m.yaw = 0.0
        m.o = np.array([L.PHONE_ORIGIN[0] + 0.003, L.PHONE_ORIGIN[1] - 0.002, L.SEATED_Z + 0.03])
        m.held = ("right", np.array([0.0, L.GRIP_Y]), L.GRIP_Z)
        tcp = self.tcp_at(m.o, 0.0)
        m.step({"left": arm(FAR_LEFT), "right": arm(tcp, 0.3)})
        m.step({"left": arm(FAR_LEFT), "right": arm(tcp, 1.0)})
        self.assertIsNone(m.held)
        self.assertAlmostEqual(float(m.o[2]), L.SEATED_Z)
        st = m.states()
        self.assertAlmostEqual(st[S.PHONE].joints[L.HOOK], 0.004, places=9)
        r = S.read(st)
        ear, mouth = S.pad_reading(r, "ear"), S.pad_reading(r, "mouth")
        self.assertTrue(S.seated(ear, mouth, S.attitude(r), r.hook))

    def test_a_far_or_open_gripper_takes_nothing(self):
        m = T.OfflineModel()
        m.plans = {}
        m.step({"left": arm(FAR_LEFT), "right": arm(tuple(np.array(self.tcp_at(m.o, m.yaw)) + [0.03, 0, 0]), 0.3)})
        self.assertIsNone(m.held)
        m.step({"left": arm(FAR_LEFT), "right": arm(self.tcp_at(m.o, m.yaw), 1.0)})
        self.assertIsNone(m.held)
        m.step({"left": arm(FAR_LEFT), "right": arm(self.tcp_at(m.o, m.yaw, dz=0.06), 0.3)})
        self.assertIsNone(m.held)

    def test_unreadable_arms_change_nothing(self):
        m = T.OfflineModel()
        m.plans = {}
        m.step(None)
        np.testing.assert_allclose(m.o, L.HANDSET_ORIGIN)


class Demo(unittest.TestCase):
    def tearDown(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    def test_demo_plan(self):
        plan = T.demo()
        self.assertEqual(set(plan), {"right"})
        self.assertLess(sum(w.duration for w in plan["right"]) * HZ, 600)  # half the 1,200 step limit

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        os.environ.pop("DEMO_HOLD_ONLY")
        os.environ["DEMO_IDLE_ARM"] = "right"
        self.assertEqual(T.demo(), {})
        os.environ["DEMO_IDLE_ARM"] = "left"
        self.assertEqual(set(T.demo()), {"right"})

    def test_every_gripper_change_takes_at_least_1p5_s(self):
        prev = T.OPEN
        for w in T.right_plan():
            if abs(w.finger - prev) > 0.01:
                self.assertGreaterEqual(w.duration, 1.5, w.label)
            prev = w.finger

    def test_commanded_gap_is_8mm_narrower_than_the_part(self):
        self.assertAlmostEqual(-2 * T.HANDLE_GRIP, L.GRIP_W - 0.008, places=6)

    def test_pad_tips_never_go_below_the_desk(self):
        for w in T.right_plan():
            self.assertGreater(w.pos[2] - L.PAD_TIP, L.TOP + 0.010, w.label)

    def test_the_turn_is_a_quarter_turn_to_the_left(self):
        turn = T.YAW_PLACE - T.YAW_PICK
        self.assertAlmostEqual(math.degrees(turn), 90.0 - 1.2, places=6)
        # the handset's start yaw plus that turn is its seated yaw (within the 1.2 degree trim)
        self.assertAlmostEqual(math.degrees(L.HANDSET_YAW + turn), -1.2, places=6)

    def test_waypoints_in_reach_box(self):
        for w in T.right_plan():
            self.assertTrue(0.10 <= w.pos[0] <= 0.50 and -0.60 <= w.pos[1] <= 0.10, w.label)

    def test_the_carry_height_is_within_the_midline_reach(self):
        # at x 0.30 on the midline the TCP reaches 0.155 m up at most (yam arena reach map): stay 4 cm under
        self.assertLessEqual(T.CARRY_TCP - L.TOP, 0.115)
        self.assertLessEqual(T.HOVER - L.TOP, 0.125)


class Files(unittest.TestCase):
    def test_task_json_matches_the_task(self):
        spec = json.loads((HERE / "task.json").read_text())
        self.assertEqual(spec["instruction"], T.INSTRUCTION)
        for key, limit in {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150,
                           "fidelity": 490, "reset_notes": 580}.items():
            self.assertLessEqual(len(spec[key]), limit, key)
        self.assertEqual(spec["run_limit"], "1,200 steps (1 minute at 20 Hz)")

    def test_no_em_dashes_or_team_names(self):
        for p in list(HERE.glob("*.py")) + list(HERE.glob("*.md")) + list(HERE.glob("*.json")) + list(
                HERE.glob("*.svg")) + list(HERE.glob("tests/*.py")):
            if p.name in ("mechkit.py", "rigstate.py"):
                continue
            text = p.read_text()
            self.assertNotIn(chr(0x2014), text, p.name)
            self.assertNotIn(chr(0x2013), text, p.name)
            self.assertIsNone(re.search("|".join(["T-" + "70", "Vi" + "han", "Aggar" + "wal", "Clau" + "de", "Anthro" + "pic"]), text), p.name)

    def test_kit_files_are_unchanged_copies(self):
        for name in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / name).exists():
                self.assertEqual((HERE / name).read_bytes(), (KIT_DIR / name).read_bytes(), name)


if __name__ == "__main__":
    unittest.main()
