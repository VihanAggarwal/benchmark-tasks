"""Offline tests for layout.py, scoring.py and task.py: synthetic part states (mechkit AssemblyStates from the task's
own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
    python3 -m unittest discover -s tests -v
Every number here is a design value or the offline model's prediction; nothing was measured on the simulator."""

import itertools
import json
import math
import pathlib
import re
import subprocess
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
BASE_O = np.array([L.BASE_ORIGIN[0], L.BASE_ORIGIN[1], L.TOP + L.BASE_ORIGIN[2]])


def tote_state(spec, origin, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    q = mk.quat_from_ypr(math.radians(yaw_deg), math.radians(pitch_deg), math.radians(roll_deg))
    return mk.state_from_poses(spec, (*origin, *q), source="test")


def base_state(dxy=(0.0, 0.0), yaw_deg=0.0, tilt_deg=0.0, dz=0.0):
    o = BASE_O + np.array([dxy[0], dxy[1], dz])
    return tote_state(T.BASE_PART, o, yaw_deg, roll_deg=tilt_deg)


def pick_on_base(dx=0.0, dy=0.0, dz=L.SEAT_DZ, yaw_deg=0.0, roll_deg=0.0, pitch_deg=0.0, base=None):
    """The picked tote ``dx``, ``dy``, ``dz`` from the base tote's origin (world axes; the base tote unturned)."""
    o = (BASE_O if base is None else base) + np.array([dx, dy, dz])
    return tote_state(T.PICK_PART, o, yaw_deg, pitch_deg, roll_deg)


def scene(pick=None, base=None):
    return {S.BASE: base if base is not None else base_state(), S.PICK: pick if pick is not None else pick_on_base()}


def start():
    return {n: T.SPECS[n].rest_state() for n in S.PARTS}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.60, L.TOP + 0.25)), "right": arm((0.1, -0.60, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def penetration(spec_a, st_a, spec_b, st_b):
    """The deepest overlap (m) between any box of ``spec_a`` posed at ``st_a`` and any of ``spec_b`` at ``st_b``
    (separating-axis test on every pair); <= 0 means no pair overlaps."""
    def boxes(spec, st):
        r = st.rotation()
        out = []
        for s in spec.shapes:
            rw = r @ s.rotation()
            out.append((np.asarray(st.origin) + r @ np.asarray(s.center), rw, np.asarray(s.size) / 2))
        return out

    ba, bb = boxes(spec_a, st_a), boxes(spec_b, st_b)
    worst = -1e9
    for (ca, ra, ha), (cb, rb, hb) in itertools.product(ba, bb):
        if np.any(np.abs(ca - cb) > ha.sum() + hb.sum() + 0.01):
            continue
        axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
        axes += [np.cross(ra[:, i], rb[:, j]) for i in range(3) for j in range(3)]
        t = cb - ca
        depth = 1e9
        for ax in axes:
            n = np.linalg.norm(ax)
            if n < 1e-9:
                continue
            ax = ax / n
            pa = float(np.sum(ha * np.abs(ra.T @ ax)))
            pb = float(np.sum(hb * np.abs(rb.T @ ax)))
            depth = min(depth, pa + pb - abs(float(t @ ax)))
        worst = max(worst, depth)
    return worst


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.TOTE_L, 0.400)
        self.assertAlmostEqual(L.TOTE_W, 0.300)
        self.assertAlmostEqual(L.TOTE_H, 0.074)
        self.assertAlmostEqual(L.WALL, 0.003)
        self.assertAlmostEqual(L.FOOT_L, 0.386)
        self.assertAlmostEqual(L.FOOT_W, 0.286)

    def test_nest_numbers(self):
        self.assertAlmostEqual(L.CLEAR, 0.004)  # 4 mm each side between the foot and the lower wall's inner face
        self.assertAlmostEqual(L.SEAT_DZ, 0.065)
        self.assertAlmostEqual(L.RIDE_DZ - L.SEAT_DZ, 0.009)  # riding is 9 mm higher than seated
        self.assertAlmostEqual(L.CAPTURE, 0.010)
        self.assertAlmostEqual(L.FLOOR_L, 0.374)
        self.assertAlmostEqual(L.FLOOR_W, 0.274)

    def test_tote_bounds_and_mass_centre(self):
        lo, hi = T.BASE_PART.local_bounds()
        self.assertTrue(np.allclose(lo, (-0.150, -0.200, 0.0), atol=1e-6), lo)
        self.assertTrue(np.allclose(hi, (0.150, 0.200, L.TOTE_H), atol=1e-6), hi)
        com = T.BASE_PART.center_of_mass()
        self.assertLess(abs(com[0]), 0.002)
        self.assertLess(abs(com[1]), 0.002)
        self.assertLess(com[2], 0.025)  # the goods sit low: the tote hangs level between the two hands

    def test_parts_spawn_where_they_rest(self):
        for n in S.PARTS:
            st = T.SPECS[n].rest_state()
            o = np.asarray(st.origin)
            self.assertAlmostEqual(o[2] - L.TOP, L.DECK_T)  # on the deck
            self.assertAlmostEqual(o[0], L.BAY_X)
        self.assertAlmostEqual(T.BASE_PART.rest_state().origin[1], -L.BAY_DY)
        self.assertAlmostEqual(T.PICK_PART.rest_state().origin[1], +L.BAY_DY)

    def test_totes_stand_clear_of_each_other_and_in_the_frames(self):
        self.assertLess(penetration(T.BASE_PART, start()[S.BASE], T.PICK_PART, start()[S.PICK]), -0.05)
        for n in S.PARTS:
            p = penetration(T.SPECS[n], start()[n], T.CART, T.CART.rest_state())
            self.assertLessEqual(p, 1e-6, f"{n} overlaps the cart at rest ({p * 1000:.1f} mm)")  # it rests on the deck

    def test_locator_frame_is_4_mm_clear(self):
        """Slide a tote 4 mm each way in the bay: no overlap with the frame; 6 mm: the frame stops it."""
        for axis in (0, 1):
            for sign in (-1, 1):
                for d, expect_hit in ((0.0035, False), (0.0065, True)):
                    o = np.array(L.PICK_ORIGIN) + np.array([0.0, 0.0, L.TOP])
                    o[axis] += sign * d
                    p = penetration(T.PICK_PART, tote_state(T.PICK_PART, o), T.CART, T.CART.rest_state())
                    self.assertEqual(p > 0.0005, expect_hit, (axis, sign, d, p))

    def test_ledge_clears_the_frame(self):
        self.assertGreaterEqual(L.LEDGE_Z0 - L.RAIL_H, 0.003 - 1e-9)  # the tote's ledge 3 mm over the rails' top

    def test_seated_pose_does_not_overlap(self):
        p = penetration(T.PICK_PART, pick_on_base(), T.BASE_PART, base_state())
        self.assertLessEqual(p, 0.0003, f"{p * 1000:.2f} mm")

    def test_seated_play_is_4_mm(self):
        """Offset 3.5 mm in x or y: still no overlap; 5 mm: the foot meets the lower wall."""
        for ax in (0, 1):
            for s in (-1, 1):
                d = [0.0, 0.0]
                d[ax] = s * 0.0035
                self.assertLessEqual(penetration(T.PICK_PART, pick_on_base(*d), T.BASE_PART, base_state()), 0.0003)
                d[ax] = s * 0.0055
                self.assertGreater(penetration(T.PICK_PART, pick_on_base(*d), T.BASE_PART, base_state()), 0.0008)

    def test_riding_pose_does_not_overlap(self):
        for dy in (0.012, -0.012):
            p = penetration(T.PICK_PART, pick_on_base(0.0, dy, L.RIDE_DZ), T.BASE_PART, base_state())
            self.assertLessEqual(p, 0.0003, f"dy {dy}: {p * 1000:.2f} mm")

    def test_seated_is_9_mm_below_riding_and_ledge_rests_on_the_walls(self):
        # lowered 2 mm past seated: the ledge would be inside the lower walls' top
        self.assertGreater(penetration(T.PICK_PART, pick_on_base(dz=L.SEAT_DZ - 0.002), T.BASE_PART, base_state()),
                           0.0015)
        # raised 3 mm: clear
        self.assertLess(penetration(T.PICK_PART, pick_on_base(dz=L.SEAT_DZ + 0.003), T.BASE_PART, base_state()), 0.0)

    def test_foot_floor_clears_the_goods_when_seated(self):
        carton_top = L.FLOOR_T + L.CARTON[2]
        self.assertGreater(L.SEAT_DZ - carton_top, 0.015)

    def test_lead_in_takes_10_mm(self):
        """With the foot 9 mm off centre the chamfer (not the wall's top) is what first meets the wall's inner corner:
        the corner's offset from the floor's edge lies within the 6 mm the lead-in covers."""
        inner = L.TOTE_L / 2 - L.WALL  # the lower wall's inner face
        for d in (0.005, 0.009):
            corner_r = inner - d  # in the upper tote's frame
            self.assertGreaterEqual(corner_r, L.FLOOR_L / 2)  # beyond the floor's edge...
            self.assertLessEqual(corner_r, L.FOOT_L / 2)  # ...and within the foot wall's outer face: on the chamfer

    def test_hand_hold_fits_the_pads(self):
        tcp = L.GRIP_Z + 0.002
        self.assertLessEqual(L.BAR_Z0, tcp + 0.009)  # the pad spans 9 mm above the TCP to 13 mm below it
        self.assertGreaterEqual(L.BAR_Z0, tcp - T.PAD_TIP)
        self.assertGreaterEqual(L.BAR_Z1, tcp - T.PAD_TIP)
        self.assertLessEqual(L.BAR_Z1, tcp + 0.009 + 0.003)
        self.assertLess(L.BAR_T, -2 * T.OPEN_G - 0.020)  # 13 mm each side when open
        self.assertLessEqual(-2 * T.GRIP_F, L.BAR_T)  # commanded gap 8 mm under the bar... into its thickness
        self.assertAlmostEqual(-2 * T.GRIP_F, L.BAR_T - 0.008)
        self.assertGreater(L.HH_SLOT_W, 0.100)  # the slot is wider than a hand's 100 mm
        self.assertGreater(L.BAR_Z0 - L.HH_SLOT_Z0, 0.010)

    def test_bar_is_above_the_nest_zone(self):
        """The foot of a seated upper tote reaches down to the lower wall's top less 9 mm: it must stay clear of the
        lower tote's own grip bars (their top is lower than that)."""
        self.assertLess(L.BAR_Z1, L.SEAT_DZ)

    def test_load_per_hand(self):
        self.assertAlmostEqual(L.load_per_hand(), 3.4335, places=3)
        self.assertLess(L.load_per_hand(), 5.0)  # GPU lesson: the arms carry about 5 N each

    def test_lead_in_plates_stay_above_the_table(self):
        for s in T.BASE_PART.shapes:
            lo = np.asarray(s.center) - s.aabb_half(s.rotation()) * 1.0
            # the plates' own faces are inside their box: the lowest corner is the floor's edge at z 0
            c = s.rotation()
            corners = np.array([np.asarray(s.center) + c @ (np.array([sx, sy, sz]) * np.asarray(s.size) / 2)
                                for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            self.assertGreaterEqual(float(corners[:, 2].min()), -1e-9, lo)

    def test_tilt_play(self):
        self.assertAlmostEqual(L.tilt_clearance_deg(), math.degrees(math.atan2(0.004, 0.193)), places=3)
        self.assertLess(L.tilt_clearance_deg(), L.SEAT_TOL_YAW)  # geometry itself keeps a seated tote within 2 deg


class Tiers(unittest.TestCase):
    def test_full(self):
        r = run(scene())
        self.assertEqual(r.score, 1.0, r.details)
        self.assertTrue(r.success)

    def test_tier_table(self):
        near = dict(dx=0.012, dy=0.0, dz=L.RIDE_DZ, yaw=0.0, tilt=0.0)
        sloppy = dict(dx=0.0, dy=0.040, dz=0.060, yaw=0.0, tilt=0.0)
        seated = dict(dx=0.0, dy=0.0, dz=L.SEAT_DZ, yaw=0.0, tilt=0.0)
        beside = dict(dx=0.0, dy=0.46, dz=0.0, yaw=0.0, tilt=0.0)
        self.assertEqual(S.tier(False, True, seated), 1.0)
        self.assertEqual(S.tier(False, True, near), 0.5)
        self.assertEqual(S.tier(False, True, sloppy), 0.25)
        self.assertEqual(S.tier(False, True, beside), 0.0)
        for rel in (seated, near, sloppy):
            self.assertEqual(S.tier(True, True, rel), 0.0)  # latched
            self.assertEqual(S.tier(False, False, rel), 0.0)  # not released and still

    def test_start_scores_zero(self):
        r = run(start())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.success)

    def test_window_edges_height(self):
        self.assertEqual(run(scene(pick_on_base(dz=L.SEAT_DZ + 0.0009))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(dz=L.SEAT_DZ - 0.0009))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(dz=L.SEAT_DZ + 0.0012))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(dz=L.SEAT_DZ - 0.0012))).score, 0.5)

    def test_window_edges_centre(self):
        self.assertEqual(run(scene(pick_on_base(dx=0.0039))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(dy=-0.0039))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(dx=0.0045))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(dy=-0.0045))).score, 0.5)

    def test_window_edges_square_and_level(self):
        self.assertEqual(run(scene(pick_on_base(yaw_deg=1.9))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(yaw_deg=2.2))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(roll_deg=0.9))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(roll_deg=1.2))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(pitch_deg=1.2))).score, 0.5)

    def test_half_tier_edges(self):
        self.assertEqual(run(scene(pick_on_base(0.019, 0.0, L.RIDE_DZ))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(0.021, 0.0, L.RIDE_DZ))).score, 0.25)
        self.assertEqual(run(scene(pick_on_base(0.0, -0.019, L.RIDE_DZ))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(0.0, -0.021, L.RIDE_DZ))).score, 0.25)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, yaw_deg=3.5))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, yaw_deg=4.5))).score, 0.25)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, roll_deg=2.5))).score, 0.5)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, roll_deg=3.5))).score, 0.25)

    def test_riding_on_the_rim_is_half(self):
        r = run(scene(pick_on_base(0.0, 0.012, L.RIDE_DZ)))
        self.assertEqual(r.score, 0.5, r.details)
        self.assertFalse(r.success)

    def test_on_base_otherwise_is_quarter(self):
        for kw in (dict(dx=0.035), dict(dy=-0.05), dict(roll_deg=8.0), dict(dz=0.060), dict(yaw_deg=10.0)):
            kw.setdefault("dz", L.RIDE_DZ)
            self.assertEqual(run(scene(pick_on_base(**kw))).score, 0.25, kw)

    def test_on_base_window_edges(self):
        self.assertEqual(run(scene(pick_on_base(0.058, 0.0, L.RIDE_DZ))).score, 0.25)
        self.assertEqual(run(scene(pick_on_base(0.062, 0.0, L.RIDE_DZ))).score, 0.0)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, 0.092))).score, 0.0)  # hovering 18 mm high
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, 0.054))).score, 0.0)  # sunk: not a pose it can take
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, roll_deg=25.0))).score, 0.0)

    def test_picked_tote_still_in_its_bay_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)
        self.assertEqual(run(scene(pick=T.PICK_PART.rest_state())).score, 0.0)

    def test_tote_turned_180_is_the_same_tote(self):
        self.assertEqual(run(scene(pick_on_base(yaw_deg=180.0))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(yaw_deg=-179.0))).score, 1.0)
        self.assertEqual(run(scene(pick_on_base(yaw_deg=90.0))).score, 0.0)  # crosswise on the rim: not a stack
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, yaw_deg=19.0))).score, 0.25)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, yaw_deg=21.0))).score, 0.0)

    def test_inverted_tote_on_the_base_scores_zero(self):
        """A tote turned over on top of the lower one is not a stack (its tilt is 180 degrees)."""
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, roll_deg=180.0))).score, 0.0)
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, L.RIDE_DZ, pitch_deg=180.0))).score, 0.0)

    def test_tote_on_the_deck_in_the_wrong_place(self):
        self.assertEqual(run(scene(pick_on_base(0.0, 0.46, 0.0))).score, 0.0)  # back in the pick bay
        self.assertEqual(run(scene(pick_on_base(0.0, 0.0, 0.0))).score, 0.0)  # inside the lower one at deck height

    def test_not_held_down_by_arms_far_away(self):
        self.assertEqual(run(scene(), FAR).score, 1.0)


class Latches(unittest.TestCase):
    def test_pick_tote_off_the_deck(self):
        tr = S.Tracker()
        run(scene(pick_on_base(0.0, 0.60, -L.DECK_T - 0.001)), tracker=tr)
        self.assertIsNotNone(tr.latch["F1"])
        self.assertEqual(tr.last.score, 0.0)

    def test_fell_then_put_back_stays_zero(self):
        tr = S.Tracker()
        run(scene(pick_on_base(0.0, 0.60, -L.DECK_T - 0.001)), tracker=tr, steps=3)
        r = run(scene(), tracker=tr)
        self.assertEqual(r.score, 0.0)

    def test_base_tote_off_the_deck(self):
        tr = S.Tracker()
        run(scene(base=base_state(dz=-L.DECK_T - 0.002)), tracker=tr)
        self.assertIsNotNone(tr.latch["F1"])
        self.assertEqual(tr.last.score, 0.0)

    def test_tote_on_its_side(self):
        tr = S.Tracker()
        run(scene(pick_on_base(0.0, 0.46, 0.1, roll_deg=70.0)), tracker=tr)
        self.assertIsNotNone(tr.latch["F1"])
        self.assertEqual(tr.last.score, 0.0)

    def test_a_lifted_tote_is_not_a_fall(self):
        tr = S.Tracker()
        run(scene(pick_on_base(0.0, 0.2, 0.09)), tracker=tr, steps=5)
        self.assertIsNone(tr.latch["F1"])

    def test_base_tote_pushed_out_of_its_frame(self):
        for d in ((0.012, 0.0), (0.0, -0.012), (0.0, 0.012)):
            tr = S.Tracker()
            base = base_state(dxy=d)
            run(scene(pick=pick_on_base(d[0], d[1], base=BASE_O, dz=L.SEAT_DZ, ), base=base), tracker=tr)
            self.assertIsNotNone(tr.latch["F2"], d)
            self.assertEqual(tr.last.score, 0.0, d)

    def test_base_tote_within_its_play_is_fine(self):
        d = (0.004, -0.004)
        r = run(scene(pick=pick_on_base(d[0], d[1]), base=base_state(dxy=d)))
        self.assertEqual(r.score, 1.0, r.details)

    def test_base_tote_turned(self):
        tr = S.Tracker()
        run(scene(base=base_state(yaw_deg=6.0), pick=pick_on_base(yaw_deg=6.0)), tracker=tr)
        self.assertIsNotNone(tr.latch["F2"])
        self.assertEqual(tr.last.score, 0.0)

    def test_base_tote_tilted(self):
        tr = S.Tracker()
        run(scene(base=base_state(tilt_deg=6.0)), tracker=tr)
        self.assertIsNotNone(tr.latch["F2"])

    def test_base_tote_turned_a_little_is_fine(self):
        tr = S.Tracker()
        run(scene(base=base_state(yaw_deg=3.0), pick=pick_on_base(yaw_deg=3.0)), tracker=tr)
        self.assertIsNone(tr.latch["F2"])

    def test_base_tote_knocked_then_put_back_stays_zero(self):
        tr = S.Tracker()
        run(scene(base=base_state(dxy=(0.03, 0.0))), tracker=tr, steps=3)
        r = run(scene(), tracker=tr)
        self.assertEqual(r.score, 0.0)

    def test_the_two_totes_swapped(self):
        """Exploit: stack the tote that stood on the cart onto the picked one. The base tote has moved: 0."""
        tr = S.Tracker()
        swapped = {S.BASE: tote_state(T.BASE_PART, np.array([L.PICK_ORIGIN[0], L.PICK_ORIGIN[1], L.TOP + 0.071])),
                   S.PICK: tote_state(T.PICK_PART, np.array([L.PICK_ORIGIN[0], L.PICK_ORIGIN[1], L.TOP + L.DECK_T]))}
        run(swapped, tracker=tr)
        self.assertIsNotNone(tr.latch["F2"])
        self.assertEqual(tr.last.score, 0.0)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_rim(self):
        rim = (L.BAY_X, L.BASE_Y + L.BAR_Y, L.TOP + L.SEAT_Z + L.GRIP_Z + 0.002)
        arms = {"left": arm(rim, 0.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms).score, 0.0)

    def test_hand_holding_the_seated_tote(self):
        """Exploit: lower it into place and never let go."""
        left = arm((L.BAY_X, L.BASE_Y + L.BAR_Y, L.TOP + L.SEAT_Z + L.GRIP_Z - 0.004), 0.02)
        right = arm((L.BAY_X, L.BASE_Y - L.BAR_Y, L.TOP + L.SEAT_Z + L.GRIP_Z - 0.004), 0.02)
        r = run(scene(), {"left": left, "right": right})
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["released_1s"])

    def test_one_hand_resting_on_it(self):
        right = arm((L.BAY_X, L.BASE_Y - L.BAR_Y - 0.045, L.TOP + 0.13), 0.0)  # closed, 40 mm past the end face
        self.assertEqual(run(scene(), {"left": FAR["left"], "right": right}).score, 0.0)

    def test_open_fingers_near_zero_closed_clear(self):
        near_open = arm((L.BAY_X, L.BASE_Y + L.BAR_Y + 0.085, L.TOP + 0.13), 1.0)
        self.assertEqual(run(scene(), {"left": near_open, "right": FAR["right"]}).score, 0.0)
        clear_closed = arm((L.BAY_X, L.BASE_Y + L.BAR_Y + 0.075, L.TOP + 0.14), 0.0)
        self.assertEqual(run(scene(), {"left": clear_closed, "right": FAR["right"]}).score, 1.0)

    def test_parked_hands_are_clear(self):
        left = arm((L.BAY_X, T.PARK_Y["left"], L.TOP + T.CARRY), 0.0)
        right = arm((L.BAY_X, T.PARK_Y["right"], L.TOP + T.CARRY), 0.0)
        self.assertEqual(run(scene(), {"left": left, "right": right}).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), None).score, 0.0)
        self.assertEqual(run(scene(), {"left": FAR["left"]}).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        for _ in range(WIN - 1):
            tr.update(scene(), FAR)
        near = {"left": arm((L.BAY_X, L.BASE_Y + L.BAR_Y, L.TOP + 0.13), 0.0), "right": FAR["right"]}
        tr.update(scene(), near)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)
        r = None
        for _ in range(WIN + 1):
            r = tr.update(scene(), FAR)
        self.assertEqual(r.score, 1.0)

    def test_moving_tote(self):
        tr = S.Tracker()
        r = None
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(dz=L.SEAT_DZ + 0.00002 * i)), FAR)  # drifting 0.4 mm/s: fine
        self.assertEqual(r.score, 1.0, r.details)
        tr = S.Tracker()
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(dz=L.SEAT_DZ + 0.0003 * math.sin(i * 0.5))), FAR)
        self.assertEqual(r.score, 1.0)
        tr = S.Tracker()
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(dx=0.0001 * i)), FAR)  # 2 mm/s: under 5 mm in the window, not visible
        self.assertEqual(r.score, 1.0, r.details)
        tr = S.Tracker()
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(dz=L.SEAT_DZ + 0.004 * math.sin(i * 0.5))), FAR)  # rocking 8 mm p-p
        self.assertEqual(r.score, 0.0)

    def test_turning_tote(self):
        tr = S.Tracker()
        r = None
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(yaw_deg=0.03 * i)), FAR)  # 0.6 deg/s: under 5 degrees, not visible
        self.assertEqual(r.score, 1.0, r.details)
        tr = S.Tracker()
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(yaw_deg=1.5 * math.sin(i * 0.5))), FAR)  # wobbling 3 deg p-p: fine
        self.assertEqual(r.score, 1.0, r.details)
        tr = S.Tracker()
        for i in range(WIN + 5):
            r = tr.update(scene(pick_on_base(yaw_deg=4.0 * math.sin(i * 0.5))), FAR)  # wobbling 8 deg p-p
        self.assertEqual(r.score, 0.0)

    def test_still_needs_the_full_window(self):
        tr = S.Tracker()
        for i in range(WIN - 1):
            r = tr.update(scene(), FAR)
        self.assertEqual(r.score, 0.0)
        for i in range(3):
            r = tr.update(scene(), FAR)
        self.assertEqual(r.score, 1.0)

    def test_success_needs_the_tier_one_only(self):
        r = run(scene(pick_on_base(0.0, 0.012, L.RIDE_DZ)))
        self.assertEqual(r.score, 0.5)
        self.assertFalse(r.success)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        tr = S.Tracker()
        r = tr.update({S.BASE: base_state()}, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["valid_state"])
        self.assertEqual(S.Tracker().update(None, FAR).score, 0.0)

    def test_nan_state(self):
        tr = S.Tracker()
        run(scene(), tracker=tr, steps=WIN + 3)
        bad = scene()
        o = np.asarray(bad[S.PICK].origin).copy()
        o[2] = float("nan")
        bad[S.PICK] = mk.state_from_poses(T.PICK_PART, (*np.nan_to_num(o), *bad[S.PICK].quat_xyzw), source="test")
        object.__setattr__(bad[S.PICK], "origin", o)
        r = tr.update(bad, FAR)
        self.assertEqual(r.score, 0.0)
        self.assertEqual(tr.bad_steps, 1)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # and the 1 s window starts again

    def test_nan_quaternion_and_inf(self):
        s = scene()
        object.__setattr__(s[S.PICK], "quat_xyzw", np.array([0.0, float("nan"), 0.0, 1.0]))
        self.assertEqual(S.Tracker().update(s, FAR).score, 0.0)
        s = scene()
        object.__setattr__(s[S.BASE], "origin", np.array([0.0, float("inf"), 0.0]))
        self.assertEqual(S.Tracker().update(s, FAR).score, 0.0)

    def test_exploded_state(self):
        s = scene(pick_on_base(0.0, 30.0, 500.0))
        r = run(s)
        self.assertEqual(r.score, 0.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        run(scene(pick_on_base(0.0, 0.60, -L.DECK_T - 0.001)), tracker=tr, steps=3)
        self.assertIsNotNone(tr.latch["F1"])
        tr.update(start(), FAR)
        self.assertIsNone(tr.latch["F1"])
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(400):
            o = BASE_O + rng.normal(0, 0.1, 3)
            q = rng.normal(size=4)
            st = {S.BASE: mk.state_from_poses(T.BASE_PART, (*BASE_O, 0, 0, 0, 1), source="test"),
                  S.PICK: mk.state_from_poses(T.PICK_PART, (*o, *q), source="test")}
            r = tr.update(st, FAR)
            self.assertIn(r.score, (0.0, 0.25, 0.5, 1.0))


class Exploits(unittest.TestCase):
    """Strategies a robot could try with only what is in the scene, none of which may score."""

    def test_hold_the_tote_in_place_without_letting_go(self):
        left = arm((L.BAY_X, L.BASE_Y + L.BAR_Y, L.TOP + L.SEAT_Z + L.GRIP_Z), 0.02)
        right = arm((L.BAY_X, L.BASE_Y - L.BAR_Y, L.TOP + L.SEAT_Z + L.GRIP_Z), 0.02)
        self.assertEqual(run(scene(), {"left": left, "right": right}).score, 0.0)

    def test_shove_the_base_tote_aside_and_stack_on_it(self):
        d = (0.0, -0.020)
        r = run(scene(pick=pick_on_base(*d), base=base_state(dxy=d)))
        self.assertEqual(r.score, 0.0)

    def test_drop_the_tote_onto_the_other_from_height_and_leave_it_tilted(self):
        r = run(scene(pick_on_base(0.0, 0.0, 0.0745, roll_deg=7.0)))
        self.assertEqual(r.score, 0.25)

    def test_stand_the_tote_on_its_end_beside_the_cart(self):
        r = run(scene(pick_on_base(0.0, 0.30, 0.20, roll_deg=90.0)))
        self.assertEqual(r.score, 0.0)

    def test_knock_the_base_tote_off_and_set_the_pick_tote_where_it_stood(self):
        base = base_state(dxy=(0.0, -0.30))
        tr = S.Tracker()
        run(scene(base=base), tracker=tr, steps=3)
        pick_where_base_was = tote_state(T.PICK_PART, BASE_O)
        r = run(scene(pick=pick_where_base_was, base=base), tracker=tr)
        self.assertEqual(r.score, 0.0)

    def test_leave_the_pick_tote_in_its_bay_and_the_base_where_it_is(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_lower_tote_lifted_out_and_dropped_back(self):
        tr = S.Tracker()
        run(scene(base=base_state(dz=0.05)), tracker=tr, steps=3)  # lifted 50 mm, still over its bay: allowed
        self.assertIsNone(tr.latch["F2"])  # (its plan position is unchanged; only a shift off the bay latches)
        run(scene(base=base_state(dxy=(0.0, 0.2))), tracker=tr, steps=2)
        self.assertIsNotNone(tr.latch["F2"])


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["cart", S.BASE, S.PICK])
        for o in T.OBJECTS:
            self.assertTrue(o.name.isidentifier())
        self.assertIsNone(T.CART.mass)
        self.assertEqual(T.BASE_PART.mass, L.TOTE_MASS)

    def test_instruction_states_the_goal_only(self):
        self.assertIn("seated", T.INSTRUCTION)
        self.assertLess(len(T.INSTRUCTION), 300)
        self.assertNotIn("hand hold", T.INSTRUCTION)  # the reach is the robot's to find

    def test_demo_plans(self):
        p = T.demo()
        self.assertEqual(sorted(p), ["left", "right"])
        self.assertEqual(len(p["left"]), len(p["right"]))
        for side in p:
            for w in p[side]:
                self.assertGreater(w.duration, 0.0)
                self.assertGreaterEqual(L.TOP + 0.9, w.pos[2])  # nothing above the reach

    def test_demo_env_switches(self):
        import os
        for env, expect in ((("DEMO_HOLD_ONLY", "1"), {}), (("DEMO_IDLE_ARM", "left"), {"right"}),
                            (("DEMO_IDLE_ARM", "right"), {"left"})):
            old = os.environ.get(env[0])
            os.environ[env[0]] = env[1]
            try:
                self.assertEqual(set(T.demo()), set(expect))
            finally:
                if old is None:
                    del os.environ[env[0]]
                else:
                    os.environ[env[0]] = old

    def test_unknown_variant_is_refused(self):
        with self.assertRaises(ValueError):
            T.plans("sideways")

    def test_demo_timing(self):
        p = T.plans()
        total = sum(w.duration for w in p["left"])
        self.assertAlmostEqual(total, sum(w.duration for w in p["right"]))
        steps = (total + 3.0) * HZ
        self.assertLessEqual(steps, 600)  # half the 1200-step run limit
        for side in p:
            for w in p[side]:
                if "grip" in w.label or "let go" in w.label:
                    self.assertGreaterEqual(w.duration, 1.5)  # the gripper's 0.05 per step limit

    def test_carry_clears_the_lower_walls(self):
        origin = T.CARRY - L.GRIP_Z - 0.002
        self.assertGreaterEqual(origin - (L.DECK_T + L.TOTE_H), 0.015)  # foot underside at least 15 mm over the rim
        origin_lo = T.ABOVE_SEAT - L.GRIP_Z - 0.002
        self.assertGreaterEqual(origin_lo - (L.DECK_T + L.TOTE_H), 0.005)  # still above it before the descent

    def test_press_is_small(self):
        self.assertLessEqual(T.PRESS, 0.005)
        self.assertAlmostEqual(T.GRIP_Z_SEAT, L.SEAT_Z + L.GRIP_Z + 0.002)

    def test_grip_geometry(self):
        self.assertAlmostEqual(T.PICK_Y["left"], L.PICK_Y + 0.195)
        self.assertAlmostEqual(T.PICK_Y["right"], L.PICK_Y - 0.195)
        self.assertAlmostEqual(T.STACK_Y["left"], L.BASE_Y + 0.195)
        self.assertAlmostEqual(T.STACK_Y["right"], L.BASE_Y - 0.195)
        self.assertAlmostEqual(T.GRIP_Z_PICK, 0.054)

    def test_outer_pad_clears_the_other_tote_at_the_pick(self):
        """The right hand's outer pad at the inner bar: 36 mm open, its outer face 7 mm past the pad's inner face."""
        y = T.PICK_Y["right"]
        outer = y + T.OPEN_G - 0.007
        other_end_face = L.BASE_Y + L.TOTE_L / 2
        self.assertGreater(outer - other_end_face, 0.030)

    def test_park_is_clear_of_the_stack(self):
        face_left = L.BASE_Y + L.TOTE_L / 2  # the stack's end face on the left hand's side
        face_right = L.BASE_Y - L.TOTE_L / 2
        self.assertGreater(T.PARK_Y["left"] - face_left, 0.0457 + 0.020)
        self.assertGreater(face_right - T.PARK_Y["right"], 0.0457 + 0.020)

    def test_hands_far_apart(self):
        for k in ("PICK_Y", "STACK_Y"):
            ys = getattr(T, k)
            self.assertAlmostEqual(ys["left"] - ys["right"], 0.390)


class OfflineModel(unittest.TestCase):
    def _hands(self, o, opening=0.02, dz=0.0):
        """Both hands closed on the bars of a tote whose origin is ``o`` (table-top frame)."""
        zc = L.TOP + o[2] + L.GRIP_Z + 0.002 + dz
        return {"left": arm((o[0], o[1] + L.BAR_Y, zc), opening), "right": arm((o[0], o[1] - L.BAR_Y, zc), opening)}

    def test_both_hands_carry_it(self):
        m = T.OfflineModel()
        o = np.array(L.PICK_ORIGIN)
        m.step(self._hands(o))
        self.assertTrue(m.held)
        for z in np.linspace(0.0, 0.08, 10):
            m.step(self._hands(o, dz=z))
        self.assertAlmostEqual(m.pos[2], o[2] + 0.08, places=6)

    def test_one_hand_does_not(self):
        m = T.OfflineModel()
        o = np.array(L.PICK_ORIGIN)
        hands = self._hands(o)
        for side in ("left", "right"):
            alone = {side: hands[side], ("right" if side == "left" else "left"): arm((0.1, 0.5, L.TOP + 0.2), 1.0)}
            m.step(alone)
            self.assertFalse(m.held)

    def test_open_hands_do_not(self):
        m = T.OfflineModel()
        m.step(self._hands(np.array(L.PICK_ORIGIN), opening=1.0))
        self.assertFalse(m.held)

    def _release_at(self, dx, dy):
        m = T.OfflineModel()
        m.step(self._hands(np.array(L.PICK_ORIGIN)))
        o = np.array([L.BASE_ORIGIN[0] + dx, L.BASE_ORIGIN[1] + dy, L.BASE_ORIGIN[2] + L.SEAT_DZ - 0.004])
        self._carry(m, o)
        m.step({"left": FAR["left"], "right": FAR["right"]})
        return m

    def _carry(self, m, o, n=60):
        """Both hands carry the tote from where it is to origin ``o`` in small steps, as a plan would."""
        start = m.pos.copy()
        for f in np.linspace(0.0, 1.0, n + 1)[1:]:
            m.step(self._hands(start + (np.asarray(o) - start) * f))

    def test_release_on_the_nest_seats(self):
        m = self._release_at(0.0, 0.0)
        self.assertAlmostEqual(m.pos[2], L.SEAT_Z)

    def test_release_off_centre_by_less_than_the_lead_in_centres(self):
        for dx, dy in ((0.008, 0.0), (0.0, -0.009), (-0.009, 0.009)):
            m = self._release_at(dx, dy)
            self.assertAlmostEqual(m.pos[2], L.SEAT_Z)
            self.assertLessEqual(abs(m.pos[0] - L.BASE_ORIGIN[0]), L.CLEAR + 1e-9)
            self.assertLessEqual(abs(m.pos[1] - L.BASE_ORIGIN[1]), L.CLEAR + 1e-9)

    def test_release_a_little_further_off_rides(self):
        m = self._release_at(0.0, 0.012)
        self.assertAlmostEqual(m.pos[2], L.BASE_ORIGIN[2] + L.RIDE_DZ)

    def test_release_far_off_lands_on_the_deck(self):
        m = self._release_at(0.0, 0.10)
        self.assertAlmostEqual(m.pos[2], L.DECK_T)
        m = self._release_at(0.0, 1.00)
        self.assertAlmostEqual(m.pos[2], 0.0)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].origin, T.SPECS[n].rest_state().origin, atol=1e-6), n)

    def test_scores_through_the_model(self):
        """The model's states score like the scorer's own: seated and let go is 1.0 after the window."""
        m = T.OfflineModel()
        m.step(self._hands(np.array(L.PICK_ORIGIN)))
        o = np.array([L.BASE_ORIGIN[0], L.BASE_ORIGIN[1], L.BASE_ORIGIN[2] + L.SEAT_DZ - 0.004])
        self._carry(m, o)
        m.step(FAR)
        self.assertEqual(run(m.states()).score, 1.0)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("400 x 300 x 74 mm", "walls 3 mm", "0.70 kg", "200 x 260 x 30 mm", "386 x 286 mm",
                      "45 degree lead-in 6 mm", "9 mm up", "4 mm clear", "65 mm above", "10 mm off centre",
                      "74 mm above", "110 mm wide, 16 mm tall", "10 mm thick, 12 mm tall", "40 to 52 mm",
                      "x 240 mm, y +230 mm", "x 240 mm, y -230 mm", "60 mm apart", "6 mm thick, 410 x 900 mm",
                      "x 80 to 490 mm", "6 mm wide and 6 mm tall", "25 x 25 x 300 mm", "x 470 mm", "3.4 N"):
            self.assertIn(token, setup)
        for token in ("65 mm above", "within 1 mm", "within 4 mm", "2 degrees", "1 degree", "60 mm", "55 to 90 mm",
                      "20 degrees", "30 mm", "5 mm or turns more than 5 degrees", "3 mm above the table",
                      "30 degrees", "10 mm", "5 degrees", "within 20 mm", "within 4 degrees", "3 degrees",
                      "Possible scores: 0, 0.25, 0.5, 1.0", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        scores = d.get("scores", "")  # before the GPU run: empty or TBD; after it: the real Dream-1 run and demo 1.0
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_layout_thresholds_match_task_json(self):
        scoring = json.loads((HERE / "task.json").read_text())["scoring"]
        self.assertAlmostEqual(L.SEAT_TOL_Z, 0.001)
        self.assertAlmostEqual(L.SEAT_TOL_XY, 0.004)
        self.assertAlmostEqual(L.NEAR_XY, 0.020)
        self.assertAlmostEqual(L.MOVED_XY, 0.010)
        self.assertAlmostEqual(L.ON_XY, 0.060)
        self.assertAlmostEqual(L.ON_Z0, 0.055)
        self.assertAlmostEqual(L.ON_Z1, 0.090)
        self.assertAlmostEqual(L.FALL_Z, 0.003)
        self.assertIn("tilt under 20 degrees", scoring)

    def test_fidelity_states_the_estimates(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("Rigid bodies", "est", "lead-in", "carton", "0.30", "0.35"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Stockers and order fillers", "53-7065", "hand laborers and material movers", "53-7062",
                      "BLS", "hand holds"):
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
        bad = re.compile("\u2014|\u2013|co-authored|claude|anthropic|token|password|api[_ ]key", re.IGNORECASE)
        for f in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "tests/test_scoring.py"):
            text = (HERE / f).read_text()
            if not f.startswith("tests/"):
                m = bad.search(text)
                self.assertIsNone(m, f"{f}: {m.group(0) if m else ''}")
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline_and_third_party(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertIn("Third-party assets: none", text)
        self.assertNotIn("TK-", text)

    def test_checker_passes_and_predicts_one(self):
        if OFFLINE is None:
            self.skipTest("tools/offline not found")
        out = subprocess.run([sys.executable, "-B", str(OFFLINE / "check_task.py"), str(HERE), "--require-json",
                              "--strict", "--one-arm"], capture_output=True, text=True, timeout=600)
        self.assertEqual(out.returncode, 0, out.stdout[-2000:])
        self.assertIn("predicted task_score 1.0", out.stdout)
        self.assertIn("'left_only': 0.0, 'right_only': 0.0", out.stdout)


if __name__ == "__main__":
    unittest.main()
