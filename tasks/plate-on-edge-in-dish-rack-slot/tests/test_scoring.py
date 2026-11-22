"""Offline tests for layout.py, scoring.py and task.py: synthetic plate and board states (mechkit AssemblyStates from
the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
P1, P2 = L.PL_NAMES
UP = L.Q_UP


def plate_state(name, origin, rot, vz=0.0):
    pose = (*(np.asarray(origin, dtype=float) + np.array([0.0, 0.0, L.TOP])), *mk.matrix_to_quat(rot))
    return mk.state_from_poses(T.PLATE_PARTS[name], pose, (0, 0, vz, 0, 0, 0), source="test")


def in_slot(name, slot, lean=None, bias=None, y=None, dz=0.0, vz=0.0, dx=0.0):
    """The plate standing on the deck in ``slot``, its lowest point ``bias`` from the slot's middle (default: the
    demo's), leaning ``lean`` degrees toward the robot (default: resting on the near prong)."""
    bias = L.FOOT_BIAS if bias is None else bias
    y = L.PLACE_Y if y is None else y
    foot = L.SLOT_X[slot] + bias + dx
    if lean is None:
        lean = L.lean_in_slot(slot, foot, y)
    o = L.standing_origin(foot, y, lean) + np.array([0.0, 0.0, dz])
    return plate_state(name, o, L.upright_rot(lean), vz)


def flat(name, x, y, support=L.DECK):
    rot = L.upright_rot(90.0)
    p = L.CLOUD0 @ rot.T
    return plate_state(name, (x, y, support - p[:, 2].min()), rot)


def board(q=UP):
    return T.board_state(q, source="test")


def start():
    s = {n: T.PLATE_PARTS[n].rest_state() for n in S.PLATES}
    s[S.BOARD] = board(0.0)
    return s


def both(k1=2, k2=5, q=UP, **kw):
    return {P1: in_slot(P1, k1, **kw), P2: in_slot(P2, k2, **kw), S.BOARD: board(q)}


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.20, 0.34, L.TOP + 0.205)), "right": arm((0.15, -0.25, L.TOP + 0.20))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def score_of(states, arms=FAR):
    return run(states, arms).score


class Geometry(unittest.TestCase):
    def test_real_sizes(self):
        self.assertAlmostEqual(L.PL_D, 0.171)  # 6.75 in
        self.assertAlmostEqual(L.PL_DEPTH, 0.022)  # 0.88 in
        self.assertAlmostEqual(L.PL_MASS, 0.130)  # 4.6 oz
        self.assertAlmostEqual(L.BD_X1 - L.BD_X0, 0.319)
        self.assertAlmostEqual(L.BD_Y1 - L.BD_Y0, 0.400)
        self.assertEqual(L.N_SLOTS, 6)  # 'up to 6 dinner plates'
        self.assertLessEqual(L.DECK + L.PRONG_T, 0.024)  # folded, under the listed 24 mm

    def test_plate_cloud_is_the_dished_shape(self):
        p = L.CLOUD0 @ L.upright_rot(0.0).T
        self.assertAlmostEqual(float(p[:, 2].max() - p[:, 2].min()), L.PL_D, places=6)
        self.assertAlmostEqual(float(p[:, 0].max() - p[:, 0].min()), L.PL_DEPTH, places=6)
        self.assertEqual(len(T.plate_shapes()), 1 + L.N_WALL + L.N_MARLY)

    def test_slots(self):
        for k in range(L.N_SLOTS):
            self.assertAlmostEqual(L.SLOT_HI[k] - L.SLOT_LO[k], 0.032)
        self.assertAlmostEqual(L.PRONG_TOP, 0.115)
        self.assertGreaterEqual(L.BAR_X0, L.BD_X0)
        self.assertLessEqual(L.BAR_X1, L.BD_X1)
        self.assertAlmostEqual(L.HINGE_Y - L.BAR_T / 2 - L.PRONG_L, -0.019)  # folded tips

    def test_start_plates_clear_of_each_other_board_and_upstand(self):
        s = start()
        self.assertAlmostEqual(L.PL_Y[P2] - L.PL_Y[P1] - L.PL_D, 0.014)
        r = S.read(s)
        for n in S.PLATES:
            p = S.points(r, n)
            self.assertAlmostEqual(float(p[:, 2].min()), 0.0, places=9)  # on the counter
            self.assertLessEqual(float(p[p[:, 2] <= L.UP_H, 0].max()), L.UP_FRONT_X + 1e-6)  # in front of the upstand
            self.assertAlmostEqual(S.tilt_deg(r, n), L.LEAN_DEG, places=6)
            over_board = p[(p[:, 1] <= L.BD_Y1) & (p[:, 0] <= L.BD_X1)]
            if len(over_board):
                self.assertGreater(float(over_board[:, 2].min()), L.DECK + 0.010)
        self.assertGreater(float(S.points(r, P2)[:, 1].min() - S.points(r, P1)[:, 1].max()), 0.010)

    def test_plate_colliders_are_the_drawn_shape(self):
        """GPU 22 Nov: a mirrored drawing frame (det -1) gave every turned collider a wrong quaternion and the
        splashback's WALL_T (12 mm) made the sloping wall 3x thick; the plates were thrown flat at spawn. Every
        collider's corners must lie inside the plate's envelope (w 0..22 mm, r <= 86.5 mm) and, at spawn, on or
        over the counter and in front of the upstand."""
        self.assertAlmostEqual(float(np.linalg.det(L.R_DRAWN)), 1.0, places=9)
        self.assertAlmostEqual(L.PL_WALL_T, 0.004)
        corners = []
        for s in T.PLATE_PARTS[P1].body_shapes("base"):
            r = s.rotation()
            if s.kind == "box":
                h = np.asarray(s.size) / 2
                local = np.array([[a, b, c] for a in (-h[0], h[0]) for b in (-h[1], h[1]) for c in (-h[2], h[2])])
            else:
                a = np.linspace(0.0, 2 * math.pi, 64, endpoint=False)
                ring = np.column_stack([s.radius * np.cos(a), s.radius * np.sin(a)])
                local = np.vstack([np.column_stack([ring, np.full(64, z)]) for z in (-s.height / 2, s.height / 2)])
            corners.append(local @ r.T + np.asarray(s.center))
        c = np.vstack(corners)
        uvw = c @ L.R_DRAWN  # back to plate coordinates
        w = uvw[:, 2] + L.W_ORIGIN
        self.assertGreaterEqual(float(w.min()), -1e-6)
        self.assertLessEqual(float(w.max()), L.PL_DEPTH + 1e-6)
        self.assertLessEqual(float(np.hypot(uvw[:, 0], uvw[:, 1]).max()), L.PL_R + 0.001)
        p = c + np.array([L.LEAN_X, 0.0, L.LEAN_Z])
        self.assertGreaterEqual(float(p[:, 2].min()), -1e-6)  # nothing in the counter
        self.assertLessEqual(float(p[p[:, 2] <= L.UP_H, 0].max()), L.UP_FRONT_X + 1e-6)  # nothing in the upstand

    def test_back_rests_on_the_upstand_edge(self):
        r = S.read(start())
        n = S.normal(r, P1)
        e = np.array([L.UP_FRONT_X, L.PL_Y[P1], L.UP_H])
        self.assertAlmostEqual(float((e - r.pos[P1]) @ n), -L.W_ORIGIN, places=9)

    def test_lean_in_slot(self):
        self.assertAlmostEqual(L.SLOT_LEAN_DEG, 14.67, places=1)
        worst = max(L.lean_in_slot(1, L.SLOT_X[1] + b, L.PLACE_Y) for b in np.arange(-0.012, 0.0041, 0.002))
        self.assertLess(worst, 21.0)  # task.json: 'up to 21 degrees'
        self.assertLess(worst, L.PLACED_TILT_DEG - 4.0)
        s = in_slot(P1, 2)
        p = L.plate_points(s.origin - np.array([0, 0, L.TOP]), mk.quat_to_matrix(s.quat_xyzw))
        self.assertLess(L.prong_penetration(p, 2), 1e-4)
        self.assertEqual(L.prong_penetration(p, 3), 0.0)

    def test_hanging_plate_clears_both_prongs(self):
        """At the demo's foot position a vertical plate has 5 mm to each prong along the prong line."""
        o = L.standing_origin(L.SLOT_X[2] + L.FOOT_BIAS, L.PLACE_Y, 0.0)
        p = L._DENSE @ L.upright_rot(0.0).T + o
        m = np.abs(p[:, 1] - L.HINGE_Y) <= L.PRONG_T / 2
        self.assertAlmostEqual(float(p[m, 0].min()) - L.SLOT_LO[2], 0.005, places=3)
        self.assertAlmostEqual(L.SLOT_HI[2] - float(p[m, 0].max()), 0.005, places=3)

    def test_carried_plates_stay_clear_of_the_prong_row(self):
        """Carried along x the plates' outlines are on the far side of the prong row."""
        for n in S.PLATES:
            self.assertGreater(L.PL_Y[n] - L.PL_R, L.HINGE_Y + L.PRONG_T / 2 + 0.090)

    def test_carry_clears_the_deck(self):
        low = T.CARRY_Z - T.PAD_TIP + T.GRIP_DEPTH - T.STAND_H
        self.assertAlmostEqual(low - L.DECK, T.CARRY_CLEAR)
        self.assertGreater(low - L.DECK, 0.007)  # the arms trail 3 to 7 mm under load


class Tiers(unittest.TestCase):
    def test_full(self):
        self.assertEqual(score_of(both()), 1.0)

    def test_every_slot_pair(self):
        for k1 in range(L.N_SLOTS):
            for k2 in range(L.N_SLOTS):
                want = 1.0 if k1 != k2 else 0.5
                self.assertEqual(score_of(both(k1, k2)), want, (k1, k2))

    def test_one_plate_other_on_upstand(self):
        s = start()
        s[P1] = in_slot(P1, 2)
        s[S.BOARD] = board()
        self.assertEqual(score_of(s), 0.5)

    def test_rack_only(self):
        s = start()
        s[S.BOARD] = board()
        self.assertEqual(score_of(s), 0.25)

    def test_one_placed_other_flat(self):
        s = both()
        s[P2] = flat(P2, 0.20, -0.10)
        self.assertEqual(score_of(s), 0.25)

    def test_nothing_is_zero(self):
        self.assertEqual(score_of(start()), 0.0)

    def test_success_flag(self):
        self.assertTrue(run(both()).success)
        self.assertFalse(run(start()).success)


class ZeroConditions(unittest.TestCase):
    def test_rack_down_is_zero(self):
        for q in (0.0, math.radians(60), math.radians(84.5)):
            self.assertEqual(score_of(both(q=q)), 0.0, q)

    def test_rack_at_85_counts(self):
        self.assertEqual(score_of(both(q=math.radians(85.1))), 1.0)

    def test_fall_latches(self):
        tr = S.Tracker()
        tr.update(both(vz=-0.95), FAR)
        self.assertTrue(tr.latched)
        self.assertEqual(run(both(), tracker=tr).score, 0.0)

    def test_slow_drop_does_not_latch(self):
        tr = S.Tracker()
        tr.update(both(vz=-0.85), FAR)
        self.assertFalse(tr.latched)

    def test_off_the_counter(self):
        s = both()
        s[P2] = flat(P2, 0.20, 0.30, support=-0.030)
        self.assertEqual(score_of(s), 0.0)

    def test_hand_on_a_plate(self):
        s = both()
        top = S.points(S.read(s), P1)[:, 2].max() + L.TOP
        x = float(s[P1].origin[0])
        arms = {"left": arm((x, L.PLACE_Y, top + 0.001), 0.0), "right": FAR["right"]}
        self.assertEqual(score_of(s, arms), 0.0)

    def test_hand_holding_the_rack(self):
        p = T.rack_pinch_point(UP)
        arms = {"left": FAR["left"], "right": arm((p[0], p[1], L.TOP + p[2] + 0.010), 0.0)}
        self.assertEqual(score_of(both(), arms), 0.0)

    def test_unreadable_arms(self):
        self.assertEqual(score_of(both(), None), 0.0)

    def test_moving_plate(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(both(dx=0.00011 * i), FAR)  # 2.2 mm in 1 s
        self.assertEqual(tr.last.score, 0.0)

    def test_creeping_plate_under_2mm_is_still(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(both(dx=0.00005 * i), FAR)  # 1.0 mm in 1 s
        self.assertEqual(tr.last.score, 1.0)

    def test_moving_rack(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(both(q=UP - math.radians(0.06 * i)), FAR)  # 1.2 deg in 1 s: tips move 2.1 mm
        self.assertEqual(tr.last.score, 0.0)

    def test_rack_settling_under_2mm_is_still(self):
        tr = S.Tracker()
        for i in range(WIN + 3):
            tr.update(both(q=UP - math.radians(0.02 * i)), FAR)  # 0.4 deg in 1 s: tips move 0.7 mm
        self.assertEqual(tr.last.score, 1.0)

    def test_needs_a_full_second(self):
        self.assertEqual(run(both(), steps=WIN).score, 0.0)
        self.assertEqual(run(both(), steps=WIN + 1).score, 1.0)

    def test_missing_state(self):
        s = both()
        del s[S.BOARD]
        self.assertEqual(score_of(s), 0.0)


class Exploits(unittest.TestCase):
    def test_plate_outside_the_prong_band(self):
        for y in (L.PLACED_Y0 - 0.003, L.PLACED_Y1 + 0.003):
            s = both()
            s[P2] = in_slot(P2, 5, y=y)
            self.assertEqual(score_of(s), 0.5, y)

    def test_plate_on_a_prong_line(self):
        s = both()
        s[P2] = in_slot(P2, 5, lean=0.0, bias=-0.020)  # its bottom straddles prong 5
        self.assertEqual(score_of(s), 0.5)

    def test_plate_lifted_off_the_deck(self):
        s = both()
        s[P2] = in_slot(P2, 5, dz=0.008)
        self.assertEqual(score_of(s), 0.5)

    def test_plate_beside_the_rack(self):
        s = both()
        s[P2] = in_slot(P2, 5, dx=0.060, lean=0.0)  # past the last prong
        self.assertEqual(score_of(s), 0.5)

    def test_plate_too_tilted(self):
        s = both()
        s[P2] = in_slot(P2, 5, lean=27.0)
        self.assertEqual(score_of(s), 0.5)

    def test_plates_flat_on_the_folded_rack(self):
        s = {P1: flat(P1, 0.15, 0.10), P2: flat(P2, 0.27, -0.10), S.BOARD: board(0.0)}
        self.assertEqual(score_of(s), 0.0)


class Readings(unittest.TestCase):
    def test_tilt_and_low(self):
        r = S.read(both())
        for n in S.PLATES:
            d = S.reading_of(r, n)
            self.assertAlmostEqual(d["low"], 0.0, places=9)
            self.assertAlmostEqual(d["tilt"], L.SLOT_LEAN_DEG, places=1)
            self.assertTrue(d["placed"])

    def test_teleport_resets(self):
        tr = S.Tracker()
        run(both(), tracker=tr)
        tr.update(start(), FAR)
        self.assertEqual(tr.resets, 1)


class Demo(unittest.TestCase):
    def setUp(self):
        for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
            os.environ.pop(k, None)

    def test_both_arms(self):
        p = T.demo()
        self.assertEqual(set(p), {"left", "right"})
        total = max(sum(w.duration for w in v) for v in p.values())
        steps = total * HZ
        self.assertLess(steps, 2400 / 2)

    def test_hold_only_and_idle(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        self.assertEqual(T.demo(), {})
        del os.environ["DEMO_HOLD_ONLY"]
        os.environ["DEMO_IDLE_ARM"] = "right"
        self.assertEqual(set(T.demo()), {"left"})

    def test_variants(self):
        self.assertEqual(set(T.plans("rack_only")), {"right"})
        with self.assertRaises(ValueError):
            T.plans("nope")

    def test_rack_lifted_before_plate_one_enters(self):
        right = T.plans()["right"]
        left = T.plans()["left"]
        t = 0.0
        t_lock = None
        for w in right:
            t += w.duration
            if "hold it on the stop" in w.label:
                t_lock = t
            if w.label == "up":
                t_clear = t
        t = 0.0
        for w in left:
            if w.label.startswith("slide it along y"):
                break
            t += w.duration
        self.assertGreater(t, t_clear)
        self.assertGreater(t, t_lock)

    def test_arc_follows_the_hinge(self):
        for deg in T.RACK_ARC_DEG:
            x, y, z = T._rack_point(deg)
            r = math.hypot(y - L.HINGE_Y, z - T.RACK_TCP_OVER - L.HINGE_Z)
            self.assertAlmostEqual(r, T.PRONG_GRIP_R, places=9)

    def test_push_is_inside_the_lock(self):
        self.assertLessEqual(abs(math.radians(T.RACK_PUSH_DEG) - L.Q_UP), L.LOCK_CAPTURE)
        self.assertLessEqual(math.radians(T.RACK_PUSH_DEG), L.Q_STOP + 1e-9)

    def test_prong_fingers_clear_neighbours(self):
        half = -T.PRONG_OPEN + rigstate.FINGER_THICK
        self.assertGreater(L.PRONG_PITCH - L.PRONG_T / 2 - half, 0.007)


class OfflineModel(unittest.TestCase):
    def arms(self, left=None, right=None):
        return {"left": left or FAR["left"], "right": right or FAR["right"]}

    def test_rack_lifts_and_locks(self):
        m = T.OfflineModel()
        x, y, z = T._rack_point(0.0)
        m.step(self.arms(right=arm((x, y, L.TOP + z), 0.0)))
        self.assertEqual(m.rack_arm, "right")
        for deg in (30, 60, 92):
            x, y, z = T._rack_point(deg)
            m.step(self.arms(right=arm((x, y, L.TOP + z), 0.0)))
            self.assertAlmostEqual(math.degrees(m.q), deg, places=6)
        m.step(self.arms(right=arm((x, y, L.TOP + z), 0.5)))
        self.assertAlmostEqual(m.q, L.Q_UP)
        self.assertIsNone(m.rack_arm)

    def test_rack_let_go_short_falls_flat(self):
        m = T.OfflineModel()
        x, y, z = T._rack_point(0.0)
        m.step(self.arms(right=arm((x, y, L.TOP + z), 0.0)))
        x, y, z = T._rack_point(75)
        m.step(self.arms(right=arm((x, y, L.TOP + z), 0.0)))
        m.step(self.arms(right=arm((x, y, L.TOP + z), 0.5)))
        self.assertEqual(m.q, 0.0)

    def test_grip_misses_the_rack_off_the_prong(self):
        m = T.OfflineModel()
        x, y, z = T._rack_point(0.0)
        m.step(self.arms(right=arm((x + 0.020, y, L.TOP + z), 0.0)))
        self.assertIsNone(m.rack_arm)

    def test_plate_dropped_without_the_rack_falls(self):
        m = T.OfflineModel()
        m.step(self.arms(left=arm((T.X_PICK, L.PL_Y[P1], L.TOP + T.GRIP_Z), 0.0)))
        self.assertIn(P1, m.held)
        sx = L.SLOT_X[2] + L.FOOT_BIAS
        for p in ((sx, L.PL_Y[P1], T.CARRY_Z), (sx, L.PLACE_Y, T.CARRY_Z), (sx, L.PLACE_Y, T.SET_Z)):
            m.step(self.arms(left=arm((p[0], p[1], L.TOP + p[2]), 0.0)))
        m.step(self.arms(left=arm((sx, L.PLACE_Y, L.TOP + T.SET_Z), 0.5)))
        self.assertLess(m.vz[P1], -L.FALL_SPEED)

    def test_plate_in_locked_rack_stands(self):
        m = T.OfflineModel()
        m.q = L.Q_UP
        m.step(self.arms(left=arm((T.X_PICK, L.PL_Y[P1], L.TOP + T.GRIP_Z), 0.0)))
        sx = L.SLOT_X[2] + L.FOOT_BIAS
        for p in ((sx, L.PL_Y[P1], T.CARRY_Z), (sx, L.PLACE_Y, T.CARRY_Z), (sx, L.PLACE_Y, T.SET_Z),
                  (sx - T.LEAN_SHIFT, L.PLACE_Y, T.SET_Z)):
            m.step(self.arms(left=arm((p[0], p[1], L.TOP + p[2]), 0.0)))
        m.step(self.arms(left=arm((sx - T.LEAN_SHIFT, L.PLACE_Y, L.TOP + T.SET_Z), 0.5)))
        r = S.read(m.states())
        d = S.reading_of(r, P1)
        self.assertTrue(d["placed"], d)
        self.assertEqual(d["slot"], 2)
        self.assertGreater(m.vz[P1], -L.FALL_SPEED)

    def test_carried_plate_never_below_the_deck(self):
        m = T.OfflineModel()
        m.step(self.arms(left=arm((T.X_PICK, L.PL_Y[P1], L.TOP + T.GRIP_Z), 0.0)))
        sx = L.SLOT_X[2]
        m.step(self.arms(left=arm((sx, L.PLACE_Y, L.TOP + T.SET_Z - 0.030), 0.0)))
        p = L.plate_points(m.pos[P1], m.rot[P1])
        self.assertGreaterEqual(float(p[:, 2].min()), L.DECK - 1e-9)


class Files(unittest.TestCase):
    def setUp(self):
        self.d = json.loads((HERE / "task.json").read_text())

    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = self.d
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Societal relatability")
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "2400 steps (120 s at 20 Hz)")
        for token in ("Joseph Joseph Flip-Up", "319 x 400 mm", "x 26 to 345 mm", "y -200 to +200 mm",
                      "12 mm over the counter", "y +80 mm, 16 mm up", "seven prongs 8 mm square and 95 mm long",
                      "x 90, 130, 170, 210, 250, 290 and 330 mm", "tips at y -19 mm", "115 mm", "2 degrees past",
                      "Within 5 degrees of upright", "90 degrees", "six slots 32 mm wide",
                      "x 110, 150, 190, 230, 270 and 310 mm", "75 mm tall", "x 351 mm", "x 450 mm",
                      "x 100 to 395 mm, y -620 to -240 mm", "x 420, y -430 mm", "Corelle", "6.75 in",
                      "171 mm across, 22 mm deep, 0.130 kg", "12 degrees from vertical", "x 317 mm",
                      "plate 1 centred y +265 mm", "plate 2 centred y +450 mm", "14 mm apart", "94 mm",
                      "5 mm to each prong", "up to 21 degrees", "about 15"):
            self.assertIn(token, setup)
        for token in ("85 degrees", "5 mm", "10 mm", "0.5 mm allowed", "25 degrees", "30 to 70 mm",
                      "y 110 to 150 mm", "40 degrees", "30 mm or more", "moves 2 mm or more", "no dot moving 2 mm",
                      "0.9 m/s", "41 mm",
                      "100 mm", "20 mm below", "0 unless the rack is up", "1.0 both plates", "0.5 one slot",
                      "0.25 no plate placed", "Possible scores: 0, 0.25, 0.5, 1.0", "1 s"):
            self.assertIn(token, scoring)
        self.assertEqual(round(L.FOOT_X * 1000), 317)
        self.assertEqual(round(L.SLOT_LEAN_DEG), 15)
        scores = d.get("scores", "")
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)

    def test_numbers_in_the_scoring_text_are_the_thresholds(self):
        self.assertAlmostEqual(L.RACK_UP_DEG, 85.0)
        self.assertAlmostEqual(L.PLACED_LOW, 0.005)
        self.assertAlmostEqual(L.BOTTOM_BAND, 0.010)
        self.assertAlmostEqual(L.SLOT_TOL, 0.0005)
        self.assertAlmostEqual(L.PLACED_TILT_DEG, 25.0)
        self.assertAlmostEqual(L.PLACED_Y0, 0.110)
        self.assertAlmostEqual(L.PLACED_Y1, 0.150)
        self.assertAlmostEqual(L.UPRIGHT_DEG, 40.0)
        self.assertEqual(round(L.FALL_SPEED ** 2 / (2 * L.G) * 1000), 41)
        self.assertAlmostEqual(L.FALL_SPEED / 30.0, 0.030)
        self.assertAlmostEqual(L.OFF_TABLE, 0.020)
        self.assertAlmostEqual(L.LET_GO, 0.030)
        self.assertAlmostEqual(L.STILL_MOVE, 0.002)
        self.assertAlmostEqual(L.RACK_TIP_R, 0.099)
        self.assertAlmostEqual(math.degrees(L.LOCK_CAPTURE), 5.0)
        self.assertAlmostEqual(math.degrees(L.Q_STOP - L.Q_UP), 2.0)

    def test_fidelity_and_context(self):
        for token in ("Rigid bodies", "dished", "lock is a rule", "est", "fixed", "Side plates"):
            self.assertIn(token, self.d["fidelity"])
        for token in ("washes up", "drain", "folded", "locks", "rim", "prongs"):
            self.assertIn(token, self.d["context"])

    def test_field_limits(self):
        limits = {"title": 85, "instruction": 490, "context": 950, "setup": 3300, "scoring": 3150, "fidelity": 490,
                  "reset_notes": 580}
        for k, n in limits.items():
            self.assertLessEqual(len(self.d[k]), n, k)

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
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                if not (f == "README.md" and line.startswith("|")):  # table rows may run long
                    self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline_and_pending(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("offline prediction", text)
        self.assertTrue("pending" in text or "job 20261122T060000" in text)
        self.assertNotIn("TK-", text)


if __name__ == "__main__":
    unittest.main()
