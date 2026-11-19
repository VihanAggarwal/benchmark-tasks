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


def carton_at(phi=0.0, sign=1.0, dx=0.0, dy=0.0, dz=0.0, joints=None, tilt=0.0, yaw=0.0, flaps="straight"):
    """The carton racked ``phi`` rad toward +y (sign +1) or -y, moved by (dx, dy, dz), its base tilted ``tilt`` rad
    about x and turned ``yaw`` rad about z; its flaps straight out unless ``flaps`` ("start" or a dict, see
    layout.joints_for)."""
    q = mk.quat_from_ypr(yaw, 0.0, tilt)
    return mk.state_from_poses(T.CARTON_PART, (L.CARTON_X + dx, L.CARTON_Y + dy, L.TOP + L.CARTON_ORIGIN_Z + dz, *q),
                               joints=joints if joints is not None else L.joints_for(phi, sign, flaps), source="test")


def flat_state(i, dx=0.0, dy=0.0, dz=0.0, yaw=0.0):
    q = mk.quat_from_ypr(yaw, 0.0, 0.0)
    return mk.state_from_poses(T.FLAT_PARTS[i], (L.PILE_X + dx, L.PILE_Y + dy, L.TOP + L.flat_z(i) + dz, *q),
                               source="test")


def scene(carton=None, flats=None):
    out = {S.CARTON: carton if carton is not None else carton_at(math.pi / 2)}
    for i, n in enumerate(S.FLATS):
        out[n] = flats[i] if flats is not None else flat_state(i)
    return out


def start():
    return scene(carton_at(0.0))


def arm(tcp, opening=1.0):
    tcp = np.asarray(tcp, dtype=float)
    return rigstate.Arm(tcp, tcp - np.array([0.0, 0.0, T.PAD_TIP]), opening)


FAR = {"left": arm((0.1, 0.55, L.TOP + 0.25)), "right": arm((0.1, -0.55, L.TOP + 0.25))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


class Geometry(unittest.TestCase):
    def test_sourced_sizes(self):
        self.assertAlmostEqual(L.LX, 0.1524, places=4)  # 6 in
        self.assertAlmostEqual(L.WA, 0.3048, places=4)  # 12 in
        self.assertAlmostEqual(L.WB, 0.2032, places=4)  # 8 in
        self.assertAlmostEqual(L.CT, 0.0032, places=4)  # 1/8 in C-flute
        self.assertAlmostEqual(L.FLAT_Y, 0.508, places=4)
        self.assertAlmostEqual(L.PT, 0.0222, places=4)
        self.assertAlmostEqual(L.FD, 0.1016, places=4)  # 4 in: half the 8 in width (RSC)
        self.assertAlmostEqual(L.FLAT_X, 0.3556, places=4)  # 6 in plus a 4 in flap at each end
        self.assertAlmostEqual(L.CARTON_MASS, 0.202, places=3)
        self.assertEqual(L.FLAT_MASS, L.CARTON_MASS)

    def test_eight_flaps_on_their_panels(self):
        self.assertEqual(len(L.FLAPS), 8)
        names = [j.name for j in T.CARTON_PART.joints]
        self.assertEqual(names, [*L.CREASES, *L.FLAPS])
        for j in T.CARTON_PART.joints:
            if j.name in L.FLAPS:
                self.assertEqual(j.spring, L.FLAP_SPRING)
                self.assertEqual(j.friction, L.FLAP_FRICTION)
                self.assertEqual(j.parent, {"a": "base", "b": "ab", "c": "bc", "d": "cd"}[j.name[0]])
                self.assertEqual(j.axis, "Y" if j.name[0] in "ac" else "Z")
                self.assertAlmostEqual(abs(j.anchor[0]), L.LX / 2, places=6)
                self.assertAlmostEqual(j.mass, L.MASS_MAJOR if j.name[0] in "ac" else L.MASS_MINOR, places=6)
        major = [L.PANELS[f][1] for f in L.FLAPS if f[0] in "ac"]
        minor = [L.PANELS[f][1] for f in L.FLAPS if f[0] in "bd"]
        for size in major:
            self.assertAlmostEqual(size[0], L.FD, places=6)
            self.assertAlmostEqual(size[1], L.WA, places=6)
        for size in minor:
            self.assertAlmostEqual(size[0], L.FD, places=6)
            self.assertGreater(size[2], L.WB - 1e-6)

    def test_flaps_start_at_their_set(self):
        """At reset the near end's four flaps are out straight (it was emptied through that end); at the far end, its
        tape just slit, the top flap droops 45 degrees, the underneath one stands 30 degrees up off the stack, the -y
        end panel's hangs 30 degrees out past straight and the +y one's is straight. The carton spans x 92 to 448 mm."""
        st = T.CARTON_PART.rest_state()
        self.assertAlmostEqual(float(st.low[0]), 0.0922, places=4)
        self.assertAlmostEqual(float(st.high[0]), 0.4478, places=4)
        r = S.read(scene(st))
        off = {k: round(math.degrees(L.flap_off(k, r.q[k])), 6) for k in L.FLAPS}
        self.assertEqual(off, {"an": 0, "af": -30, "bn": 0, "bf": 0, "cn": 0, "cf": 45, "dn": 0, "df": -30})
        self.assertEqual([k for k in L.FLAPS if not S.flap_flat(k, r.q[k])], ["af", "cf", "df"])
        self.assertEqual(set(L.LOOSE), {"af", "cf", "df"})
        for f in ("an", "bn", "cn", "dn", "bf"):  # straight: in line with its panel
            self.assertTrue(np.allclose(S.set_rot(f), np.eye(3)), f)
        # where the loose ones point (world, the table top at 0)
        tip = {}
        for f in ("af", "cf", "df"):
            lo, hi = st.link_bounds(f)
            tip[f] = (lo - np.array([0, 0, L.TOP]), hi - np.array([0, 0, L.TOP]))
        self.assertAlmostEqual(float(tip["af"][1][2]) - L.PT, L.FD * math.sin(math.radians(30)) + L.CT, delta=0.003)
        self.assertLess(float(tip["cf"][0][2]), L.PT + L.WB - L.FD * math.sin(math.radians(45)) + 0.01)  # droops
        self.assertGreater(float(tip["cf"][0][2]), L.PT + 0.12)  # well clear of the raised bottom flap's tip
        self.assertAlmostEqual(float(tip["df"][0][1]), L.A_YMIN - L.FD * math.sin(math.radians(30)), delta=0.003)
        for f in ("an", "af", "bf", "df"):
            lo, _ = st.link_bounds(f)
            self.assertGreaterEqual(float(lo[2]) - L.TOP, L.PT - 1e-6, f)  # nothing starts inside the stack
        # the stack's flats are as wide, so the underneath panel's flaps are supported
        self.assertGreaterEqual(L.FLAT_X / 2, L.LX / 2 + L.FD - 1e-9)
        self.assertGreater(L.BOARD_X, L.FLAT_X)

    def test_flap_creases_rest_at_their_set_and_hold(self):
        """Each flap's spring rests at its start angle (as drawn) and its travel reaches straight and shut. In every
        crease model (no joint friction) the spring holds a level long flap's weight with the flap well inside the
        yield angle (so no flap takes a set under its own weight, the drooping top flap included), and the most it
        holds before the crease yields is over 1.5 times that weight; a press of well under 1 N yields it."""
        weight = L.MASS_MAJOR * L.G * L.FD / 2  # 0.0085 N m
        self.assertEqual(T.FLAP_MODEL_NAME, "crease_set")
        self.assertEqual(L.FLAP_MODELS[L.FLAP_MODEL_DEFAULT], T.FLAP_MODEL)
        for j in T.CARTON_PART.joints:
            if j.name not in L.FLAPS:
                continue
            s = L.FLAP_SET[j.name]
            self.assertLessEqual(j.lower, -s - math.radians(90) - 1e-9, j.name)
            self.assertGreaterEqual(j.upper, -s + math.radians(90) - 1e-9, j.name)
            self.assertEqual((j.spring, j.damping, j.friction), (L.FLAP_SPRING, L.FLAP_DAMPING, 0.0), j.name)
        for name, m in L.FLAP_MODELS.items():
            if m["yield_deg"] is None:
                continue
            y = math.radians(m["yield_deg"])
            self.assertEqual(m["friction"], 0.0, name)
            self.assertLess(weight / m["spring"], 0.6 * y, name)
            self.assertGreater(m["spring"] * y, 1.5 * weight, name)
            self.assertLess(m["spring"] * y / 0.07, 0.5, name)  # N at a pad 70 mm out
            inertia = L.MASS_MAJOR * L.FD ** 2 / 3
            self.assertGreater(m["damping"], 2 * math.sqrt(m["spring"] * inertia), name)  # overdamped: no ring

    def test_crease_set_follows_a_bent_flap(self):
        """Within the yield angle a flap's set stays (it springs back); bent further, the set moves to the flap. The
        ratchet model moves it only for a flap opened toward straight; the 18 Nov model has no set rule."""
        y = math.radians(L.FLAP_YIELD_DEG)
        m = L.FLAP_MODELS
        self.assertEqual(T.next_set("af", 0.9 * y, 0.0, m["crease_set"]), 0.0)
        self.assertEqual(T.next_set("af", 1.1 * y, 0.0, m["crease_set"]), 1.1 * y)
        self.assertEqual(T.next_set("af", -1.1 * y, 0.0, m["crease_set"]), -1.1 * y)
        # af starts 30 degrees up (set -30 from straight): +joint opens it toward straight, -joint raises it further
        self.assertEqual(T.next_set("af", 0.2, 0.0, m["crease_ratchet"]), 0.2)
        self.assertEqual(T.next_set("af", -0.2, 0.0, m["crease_ratchet"]), 0.0)
        self.assertEqual(T.next_set("af", 0.2, 0.0, m["friction_hold"]), 0.0)
        self.assertEqual(T.next_set("af", float("nan"), 0.1, m["crease_set"]), 0.1)
        s = math.radians(1.5)
        self.assertEqual(T.next_set("bf", 1.1 * s, 0.0, m["crease_stiff"]), 1.1 * s)

    def test_crease_set_rule_sets_the_drive_targets(self):
        """The sim rule (on_step) writes each flap's set as its drive target, clamped to its travel; a reset clears
        the sets (mechkit puts every target back to 0, the start angles)."""
        try:
            q = L.joints_for(math.pi / 2, flaps="start")
            q["af"] = math.radians(25.0)  # pressed 25 degrees toward straight
            q["cf"] = math.radians(1.0)  # within the yield of its set
            T.crease_set_rule({S.CARTON: carton_at(joints=q)})
            targets = mk.rules()["targets"]["carton"]
            self.assertEqual(set(targets), set(L.FLAPS))
            self.assertAlmostEqual(targets["af"], math.radians(25.0))
            self.assertEqual(targets["cf"], 0.0)
            self.assertAlmostEqual(T._SETS["af"], math.radians(25.0))
        finally:
            mk.offline_reset()
        self.assertEqual(T._SETS, {})
        self.assertEqual(mk.rules()["targets"], {})

    def test_loose_flaps_clear_the_stack_while_racking(self):
        """Racked toward +y the -y end panel's flap (out past straight) and the underneath panel's (raised) turn up
        or stay put, so neither digs into the stack; the drooping top flap meets the stack's top only past about 70
        degrees of rack, where the stack folds it out (the offline model's cap), to under 3 degrees once flat."""
        for deg in range(0, 91, 5):
            phi = math.radians(deg)
            st = carton_at(phi, flaps="start")
            for f in ("df", "af", "bf"):
                lo, _ = st.link_bounds(f)
                self.assertGreaterEqual(float(lo[2]) - L.TOP, L.PT - 0.0005, (f, deg))
            lo, _ = st.link_bounds("cf")
            touches = float(lo[2]) - L.TOP < L.PT
            self.assertEqual(touches, deg > 70, deg)
            self.assertEqual(T.top_flap_droop_max(phi) < math.radians(45), deg > 70, deg)
        self.assertLess(math.degrees(T.top_flap_droop_max(math.pi / 2)), 3.0)
        # folding out on the stack does not jam: friction 0.5 at the tip, the flap under 63 degrees from level
        self.assertLess(math.tan(math.radians(45)), 1 / L.CARTON_FRICTION[0])

    def test_racked_with_the_far_flaps_as_found_is_not_flat(self):
        """Racked flat with the stack's top folding the top flap out but the two raised flaps not pressed: not flat
        (the raised flaps stand 50 mm proud), so the robot must fold them out."""
        off = dict(L.FLAP_SET)
        off["cf"] = T.top_flap_droop_max(math.pi / 2)
        res = run(scene(carton_at(math.pi / 2, flaps=off)))
        self.assertFalse(res.details["flat"])
        self.assertGreater(res.details["thickness_mm"], 40.0)
        self.assertEqual(res.score, 0.25)
        for k in ("af", "df"):
            o = dict(off)
            o[k] = 0.0
            self.assertFalse(run(scene(carton_at(math.pi / 2, flaps=o))).details["flat"], k)
        o = dict(off, af=0.0, df=0.0)
        self.assertEqual(run(scene(carton_at(math.pi / 2, flaps=o))).score, 1.0)

    def test_carton_stands_on_the_stack_and_is_203_mm_tall(self):
        st = carton_at(0.0)
        self.assertAlmostEqual(float(st.low[2]) - L.TOP, L.PT - L.CT / 2 - 0.0, delta=0.0017)
        self.assertAlmostEqual(float(st.high[2]) - float(st.low[2]), L.WB + 1.5 * L.CT + 0.0, delta=0.004)
        self.assertGreater(float(st.high[2]) - float(st.low[2]), 0.205)

    def test_stack_flats_rest_on_each_other(self):
        for i in range(L.FLAT_N):
            st = flat_state(i)
            self.assertAlmostEqual(float(st.low[2]) - L.TOP, L.BOARD_T + i * L.FLAT_T, places=6)
        self.assertAlmostEqual(float(flat_state(2).high[2]) - L.TOP, L.PT, places=6)

    def test_fully_racked_is_two_layers_on_the_stack(self):
        r = S.read(scene())
        self.assertAlmostEqual(S.thickness(r), 2 * L.CT, places=5)
        fx, fy = S.footprint(r)
        self.assertAlmostEqual(fx, 0.3556, places=4)  # flaps out straight, as the flats under it
        self.assertAlmostEqual(fy, 0.516, delta=0.002)
        c = S.plan_centre(r)
        self.assertLess(float(np.hypot(c[0] - L.PILE_X, c[1] - L.PILE_Y)), 0.005)
        self.assertAlmostEqual(float(np.min(r.pts[:, 2])), L.PT, delta=0.0017)
        self.assertEqual([round(S.interior(r.q[k])) for k in L.CREASES], [180, 0, 180, 0])
        self.assertEqual([round(math.degrees(L.flap_off(k, r.q[k]))) for k in L.FLAPS], [0] * 8)

    def test_fully_racked_other_way_hangs_off_the_stack(self):
        r = S.read(scene(carton_at(math.pi / 2, sign=-1.0)))
        self.assertTrue(S.is_flat(r))
        self.assertGreater(float(np.hypot(*(S.plan_centre(r) - np.array([L.PILE_X, L.PILE_Y])))), 0.15)

    def test_start_is_standing_not_flat(self):
        r = S.read(start())
        self.assertFalse(S.is_flat(r))
        self.assertGreater(S.thickness(r), 0.2)
        self.assertEqual([round(S.interior(r.q[k])) for k in L.CREASES], [90, 90, 90, 90])

    def test_rack_geometry_is_a_four_bar(self):
        """Racked by phi the end panels stay parallel and the top panel stays level over the underneath one."""
        for phi in (0.3, 0.8, 1.2):
            st = carton_at(phi)
            rb, pb = st.link_pose("ab")
            rd, pd = st.link_pose("cd")
            self.assertAlmostEqual(float(rb[2, 2]), math.cos(phi), places=6)
            self.assertAlmostEqual(float(rd[2, 2]), math.cos(phi), places=6)
            rc, _ = st.link_pose("bc")
            self.assertAlmostEqual(float(rc[2, 2]), 1.0, places=6)

    def test_flaps_lie_in_the_two_layers_when_flat(self):
        """Racked flat, each flap lies in its panel's layer: no flap point is outside the 6.4 mm of the two layers."""
        st = carton_at(math.pi / 2)
        for f in L.FLAPS:
            lo, hi = st.link_bounds(f)
            self.assertGreaterEqual(float(lo[2]) - L.TOP, L.PT - 1e-6, f)
            self.assertLessEqual(float(hi[2]) - L.TOP, L.PT + 2 * L.CT + 1e-6, f)

    def test_thickness_by_phi(self):
        self.assertLess(S.thickness(S.read(scene(carton_at(1.2)))), 0.110)  # over half way
        self.assertGreater(S.thickness(S.read(scene(carton_at(0.9)))), 0.110)


class Tiers(unittest.TestCase):
    def test_full(self):
        res = run(scene())
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.success)

    def test_start_scores_zero(self):
        self.assertEqual(run(start()).score, 0.0)

    def test_tier_table(self):
        self.assertEqual(S.tier(False, True, True, True, True, True, 0.006), 1.0)
        self.assertEqual(S.tier(False, True, True, False, True, True, 0.006), 0.75)
        self.assertEqual(S.tier(False, True, True, False, False, True, 0.006), 0.5)
        self.assertEqual(S.tier(False, True, True, True, True, False, 0.006), 0.5)
        self.assertEqual(S.tier(False, True, False, False, False, True, 0.10), 0.25)
        self.assertEqual(S.tier(False, True, False, False, False, True, 0.20), 0.0)
        self.assertEqual(S.tier(True, True, True, True, True, True, 0.006), 0.0)
        self.assertEqual(S.tier(False, False, True, True, True, True, 0.006), 0.0)

    def test_offset_on_the_stack_is_075(self):
        self.assertEqual(run(scene(carton_at(math.pi / 2, dy=0.09))).score, 0.75)

    def test_flat_on_the_table_beside_the_stack_is_05(self):
        res = run(scene(carton_at(math.pi / 2, dy=0.0, dx=-0.30, dz=-L.PT)))
        self.assertEqual(res.score, 0.5)

    def test_flat_but_high_on_one_side_is_075_or_less(self):
        res = run(scene(carton_at(math.pi / 2, dz=0.012)))  # 12 mm up: past 6, within 20
        self.assertEqual(res.score, 0.75)
        res = run(scene(carton_at(math.pi / 2, dz=0.030)))
        self.assertEqual(res.score, 0.5)

    def test_tilted_carton_is_not_flat(self):
        res = run(scene(carton_at(math.pi / 2, tilt=math.radians(4.0))))  # 516 mm long: 36 mm of rise
        self.assertFalse(res.details["flat"])

    def test_flat_on_a_pushed_stack_is_05(self):
        flats = [flat_state(0), flat_state(1), flat_state(2, dy=0.025)]
        self.assertEqual(run(scene(flats=flats)).score, 0.5)

    def test_wrong_way_racked_is_05(self):
        self.assertEqual(run(scene(carton_at(math.pi / 2, sign=-1.0))).score, 0.5)

    def test_over_half_way_is_025(self):
        self.assertEqual(run(scene(carton_at(1.2))).score, 0.25)

    def test_standing_and_leaning_score_zero(self):
        for phi in (0.0, 0.5, 0.9):
            self.assertEqual(run(scene(carton_at(phi))).score, 0.0, phi)

    def test_one_crease_open_is_not_flat(self):
        j = L.joints_for(math.pi / 2)
        j["bc"] = math.radians(60.0)
        res = run(scene(carton_at(joints=j)))
        self.assertFalse(res.details["flat"])
        self.assertLess(res.score, 0.5)

    def test_crease_window_edges(self):
        for deg, flat in ((14.0, True), (16.0, False)):
            j = L.joints_for(math.pi / 2)
            j["ab"] = -(math.pi / 2 - math.radians(deg))
            r = S.read(scene(carton_at(joints=j)))
            self.assertEqual(all(S.crease_flat(r.q[k]) for k in L.CREASES), flat, deg)

    def test_flap_window_edges(self):
        for deg, flat in ((14.0, True), (16.0, False)):
            j = L.joints_for(math.pi / 2)
            j["cf"] = L.flap_joint("cf", math.radians(deg))
            r = S.read(scene(carton_at(joints=j)))
            self.assertEqual(all(S.flap_flat(k, r.q[k]) for k in L.FLAPS), flat, deg)

    def test_flap_folded_shut_is_not_flat(self):
        """A flap folded in (or standing up) on a racked carton: the flap rule fails and it stands 100 mm proud."""
        for f in ("cn", "df", "an"):
            j = L.joints_for(math.pi / 2)
            j[f] = L.flap_joint(f, math.pi / 2)
            res = run(scene(carton_at(joints=j)))
            self.assertFalse(res.details["flat"], f)
            self.assertLess(res.score, 0.5, f)

    def test_flap_raised_20_degrees_is_not_flat(self):
        j = L.joints_for(math.pi / 2)
        j["bf"] = L.flap_joint("bf", math.radians(20.0))
        r = S.read(scene(carton_at(joints=j)))
        self.assertFalse(S.is_flat(r))
        self.assertGreater(S.thickness(r), L.THICK_MAX)

    def test_flaps_drooping_a_little_are_still_flat(self):
        """A top-layer flap sagging 2 degrees under its own weight onto the stack stays flat."""
        j = L.joints_for(math.pi / 2)
        for f in ("cn", "cf"):
            j[f] = L.flap_joint(f, math.radians(2.0) * (1 if f.endswith("f") else -1))
        r = S.read(scene(carton_at(joints=j)))
        self.assertTrue(all(S.flap_flat(k, r.q[k]) for k in L.FLAPS))
        self.assertLessEqual(S.thickness(r), L.THICK_MAX)

    def test_thickness_and_footprint_edges(self):
        r = S.read(scene())
        self.assertTrue(S.is_flat(r))
        self.assertLessEqual(S.thickness(r), L.THICK_MAX)
        self.assertLess(L.THICK_MAX, 0.016)

    def test_on_the_stack_edges(self):
        for dy, ok in ((0.055, True), (0.065, False)):
            r = S.read(scene(carton_at(math.pi / 2, dy=dy)))
            self.assertEqual(S.on_stack(r), ok, dy)
        for dz, ok in ((0.005, True), (0.008, False)):
            r = S.read(scene(carton_at(math.pi / 2, dz=dz)))
            self.assertEqual(S.on_stack(r), ok, dz)

    def test_stack_kept_edges(self):
        for d, ok in ((0.014, True), (0.016, False)):
            r = S.read(scene(flats=[flat_state(0), flat_state(1), flat_state(2, dx=d)]))
            self.assertEqual(S.stack_kept(S.stack_moves(r)), ok, d)
        r = S.read(scene(flats=[flat_state(0), flat_state(1), flat_state(2, yaw=math.radians(9.0))]))
        self.assertFalse(S.stack_kept(S.stack_moves(r)))
        r = S.read(scene(flats=[flat_state(0), flat_state(1), flat_state(2, dz=0.005)]))
        self.assertFalse(S.stack_kept(S.stack_moves(r)))


class Exploits(unittest.TestCase):
    def test_strip_laid_out_straight_is_not_flat(self):
        """All four corner creases opened to 180 degrees lays the carton out as a 1 m strip: flat in its creases, but
        over the footprint limit, so never 1.0 and never 'flat'."""
        j = {k: -math.pi / 2 for k in L.CREASES}
        res = run(scene(carton_at(joints=j)))
        self.assertFalse(res.details["flat"])
        self.assertLessEqual(res.score, 0.25)

    def test_carton_carried_in_the_hand_scores_zero(self):
        hand = {"left": arm((L.CARTON_X, 0.0, L.TOP + L.PT + 0.02), 0.0), "right": FAR["right"]}
        self.assertEqual(run(scene(), arms=hand).score, 0.0)

    def test_sweeping_the_stack_away_scores_zero(self):
        flats = [flat_state(i, dy=0.06 + 0.01 * i) for i in range(L.FLAT_N)]
        res = run(scene(flats=flats))
        self.assertEqual(res.score, 0.0)
        self.assertIsNotNone(res.details["latch"]["KNOCKED"])

    def test_knock_is_latched(self):
        tr = S.Tracker()
        tr.update(scene(flats=[flat_state(0, dy=0.06), flat_state(1), flat_state(2)]), FAR)
        res = run(scene(), tracker=tr)
        self.assertEqual(res.score, 0.0)

    def test_pile_turned_over_30_degrees_latches(self):
        flats = [flat_state(0, yaw=math.radians(35.0)), flat_state(1), flat_state(2)]
        self.assertEqual(run(scene(flats=flats)).score, 0.0)

    def test_stack_slid_out_from_under_the_carton(self):
        """Pulling the flats out and letting the carton drop to the board: flat, but the stack moved: at best 0.5,
        and 0 once any flat moved 40 mm."""
        flats = [flat_state(i, dy=0.30) for i in range(L.FLAT_N)]
        self.assertEqual(run(scene(carton_at(math.pi / 2, dz=-L.PT + L.BOARD_T), flats=flats)).score, 0.0)


class ReleaseAndStill(unittest.TestCase):
    def test_hand_on_the_carton(self):
        hand = {"left": arm((L.CARTON_X, 0.0, L.TOP + L.PT + 0.04), 1.0), "right": FAR["right"]}
        res = run(scene(), arms=hand)
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["released_1s"])

    def test_hand_clear(self):
        self.assertEqual(run(scene()).score, 1.0)

    def test_parked_hands_are_clear(self):
        parked = {"left": arm((*T.PARK_L[:2], L.TOP + T.PARK_L[2])), "right": arm((*T.PARK_R[:2], L.TOP + T.PARK_R[2]))}
        self.assertEqual(run(scene(), arms=parked).score, 1.0)

    def test_unreadable_arms_never_release(self):
        self.assertEqual(run(scene(), arms=None).score, 0.0)

    def test_release_must_hold_the_whole_second(self):
        tr = S.Tracker()
        hand = {"left": arm((L.CARTON_X, 0.0, L.TOP + L.PT + 0.04), 1.0), "right": FAR["right"]}
        tr.update(scene(), hand)
        res = run(scene(), tracker=tr, steps=WIN - 1)
        self.assertEqual(res.score, 0.0)
        res = run(scene(), tracker=tr, steps=1)
        self.assertEqual(res.score, 1.0)  # the hand step has left the window

    def jitter(self, make, steps=WIN + 3):
        tr = S.Tracker()
        res = None
        for i in range(steps):
            res = tr.update(make(i % 2), FAR)
        return res

    def test_moving_carton(self):
        res = self.jitter(lambda k: scene(carton_at(math.pi / 2, dy=0.006 * k)))
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["still_1s"])

    def test_small_invisible_shake_is_still(self):
        """Under 5 mm and 5 degrees over the last 1 s counts as still (no visible motion)."""
        res = self.jitter(lambda k: scene(carton_at(math.pi / 2, dy=0.004 * k)))
        self.assertTrue(res.details["still_1s"])
        self.assertEqual(res.score, 1.0)

    def test_still_edges(self):
        for d, still in ((0.0049, True), (0.0051, False)):
            res = self.jitter(lambda k: scene(carton_at(math.pi / 2, dx=d * k)))
            self.assertEqual(res.details["still_1s"], still, d)
        self.assertEqual((L.STILL_MOVE, L.STILL_TURN_DEG), (0.005, 5.0))

    def test_turning_carton(self):
        """The flat carton turning 6 degrees on the stack (its ends swing 27 mm): not still."""
        res = self.jitter(lambda k: scene(carton_at(math.pi / 2, yaw=math.radians(6.0) * k)))
        self.assertFalse(res.details["still_1s"])

    def test_flap_still_swinging(self):
        """A flap swinging 6 degrees (its tip 11 mm) over the last 1 s: not still, though every panel is."""
        def make(k):
            j = L.joints_for(math.pi / 2)
            j["af"] = L.flap_joint("af", math.radians(6.0) * k)
            return scene(carton_at(joints=j))
        res = self.jitter(make)
        self.assertFalse(res.details["still_1s"])
        self.assertEqual(res.score, 0.0)

    def test_flap_settling_slightly_is_still(self):
        def make(k):
            j = L.joints_for(math.pi / 2)
            j["af"] = L.flap_joint("af", math.radians(2.0) * k)  # its tip moves 3.5 mm
            return scene(carton_at(joints=j))
        res = self.jitter(make)
        self.assertTrue(res.details["still_1s"])

    def test_creases_still_moving(self):
        def make(k):
            j = L.joints_for(math.pi / 2)
            j["bc"] += math.radians(6.0) * k
            return scene(carton_at(joints=j))
        self.assertEqual(self.jitter(make).score, 0.0)

    def test_still_stack_flat(self):
        res = self.jitter(lambda k: scene(flats=[flat_state(0), flat_state(1), flat_state(2, dy=0.006 * k)]))
        self.assertEqual(res.score, 0.0)


class Robustness(unittest.TestCase):
    def test_missing_state_scores_zero(self):
        for name in S.PARTS:
            s = scene()
            del s[name]
            self.assertEqual(run(s).score, 0.0, name)

    def test_nan_state(self):
        st = carton_at(math.pi / 2)
        bad = mk.state_from_poses(T.CARTON_PART, (float("nan"), 0.0, 0.8, 0.0, 0.0, 0.0, 1.0), source="test")
        res = run(scene(carton=bad))
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.details["valid_state"])
        self.assertIsNotNone(st)

    def test_nan_joint(self):
        j = L.joints_for(math.pi / 2)
        j["cd"] = float("nan")
        self.assertEqual(run(scene(carton_at(joints=j))).score, 0.0)

    def test_inf_flat(self):
        flats = [flat_state(0), flat_state(1), flat_state(2, dy=float("inf"))]
        self.assertEqual(run(scene(flats=flats)).score, 0.0)

    def test_blow_up_restarts_the_windows(self):
        tr = S.Tracker()
        run(scene(), tracker=tr)
        bad = mk.state_from_poses(T.CARTON_PART, (float("nan"), 0.0, 0.8, 0.0, 0.0, 0.0, 1.0), source="test")
        self.assertEqual(tr.update(scene(carton=bad), FAR).score, 0.0)
        self.assertEqual(tr.update(scene(), FAR).score, 0.0)  # the 1 s windows start again
        self.assertEqual(run(scene(), tracker=tr).score, 1.0)

    def test_teleport_back_to_the_start_resets(self):
        tr = S.Tracker()
        tr.update(scene(flats=[flat_state(0, dy=0.06), flat_state(1), flat_state(2)]), FAR)
        self.assertTrue(tr.latched)
        tr.update(start(), FAR)
        self.assertFalse(tr.latched)
        self.assertEqual(tr.resets, 1)

    def test_random_states_never_raise_and_stay_in_the_tiers(self):
        rng = np.random.default_rng(7)
        tr = S.Tracker()
        for _ in range(200):
            j = {k: float(rng.uniform(-1.7, 1.7)) for k in L.CREASES}
            c = carton_at(joints=j, dx=float(rng.normal(0, 0.1)), dy=float(rng.normal(0, 0.2)),
                          dz=float(rng.normal(0, 0.05)), tilt=float(rng.normal(0, 0.2)))
            flats = [flat_state(i, dx=float(rng.normal(0, 0.02)), dy=float(rng.normal(0, 0.02))) for i in range(3)]
            res = tr.update(scene(c, flats), FAR)
            self.assertIn(res.score, (0.0, 0.25, 0.5, 0.75, 1.0))


class TaskFile(unittest.TestCase):
    def test_objects(self):
        names = [o.name for o in T.OBJECTS]
        self.assertEqual(names, ["board", "flat_1", "flat_2", "flat_3", "carton"])
        self.assertIsNone(T.BOARD.mass)
        self.assertEqual(T.SCORE_OVER, "end")
        self.assertEqual(T.SUCCESS_HOLD_S, 1.0)

    def test_instruction_states_the_goal_only(self):
        t = T.INSTRUCTION.lower()
        for word in ("push", "rack", "crease", "press", "left arm", "right arm", "degrees"):
            self.assertNotIn(word, t)
        self.assertLess(len(T.INSTRUCTION), 300)

    def test_demo_plans(self):
        p = T.plans()
        self.assertEqual(set(p), {"left", "right"})
        for side in p:
            self.assertGreater(len(p[side]), 3)

    def test_demo_env_switches(self):
        old = {k: os.environ.pop(k, None) for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT")}
        try:
            self.assertEqual(set(T.demo()), {"left", "right"})
            os.environ["DEMO_IDLE_ARM"] = "left"
            self.assertEqual(set(T.demo()), {"right"})
            os.environ["DEMO_IDLE_ARM"] = "right"
            self.assertEqual(set(T.demo()), {"left"})
            del os.environ["DEMO_IDLE_ARM"]
            os.environ["DEMO_HOLD_ONLY"] = "1"
            self.assertEqual(T.demo(), {})
            del os.environ["DEMO_HOLD_ONLY"]
            os.environ["DEMO_VARIANT"] = "shallow"
            self.assertEqual(set(T.demo()), {"left", "right"})
            os.environ["DEMO_VARIANT"] = "flaps_left"
            labels = [w.label for w in T.demo()["right"]]
            self.assertFalse(any("raised far flaps" in t for t in labels))
            for v in ("crease_set", "crease_stiff", "crease_ratchet", "friction_hold"):  # flap models: the plan
                os.environ["DEMO_VARIANT"] = v
                self.assertEqual(T.demo(), T.plans(""), v)
                self.assertEqual(T.flap_model_name(), v)
            self.assertEqual(T.flap_model_name("shallow"), "crease_set")
            os.environ["DEMO_VARIANT"] = "nonsense"
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            for k in ("DEMO_HOLD_ONLY", "DEMO_IDLE_ARM", "DEMO_VARIANT"):
                os.environ.pop(k, None)
            for k, v in old.items():
                if v is not None:
                    os.environ[k] = v

    def test_demo_timing(self):
        p = T.plans()
        total = max(T._duration(p[s]) for s in p)
        self.assertLess(total, 40.0)
        self.assertLess(total * HZ + 3 * HZ, 1200)  # the demo plus the policy's 3 s hold fits the run limit
        self.assertAlmostEqual(T._duration(p["left"]), T._duration(p["right"]), places=6)  # last presses together

    def test_push_geometry(self):
        """The closed block starts in the air short of the end panel's foot and the push ends before the carton's
        +y end; the end panel is 203 mm tall and the block pushes at 120 mm."""
        self.assertLess(T.PUSH_Y0 + T.PAD_FACE, T.D_FACE_Y)
        self.assertLess(T.PUSH_Y1, L.A_YMAX)
        self.assertLess(T.PUSH_HZ, L.WB)
        self.assertGreater(T.PUSH_PHI, T.GRAVITY_PHI)
        self.assertLess(T.PUSH_PHI, 1.2)
        self.assertGreater(T.PUSH_Y0 - 0.0, -0.40)  # inside the right arm's own column (base y -305, +-100)

    def test_press_geometry(self):
        """The block's pad tips end 1 mm inside the flat carton's top (6.4 mm over the stack's top)."""
        tips = T.PRESS_Z - T.PAD_TIP
        self.assertAlmostEqual(tips, L.PT + 2 * L.CT - 0.001, places=6)
        for y in T.PRESS_R:
            self.assertLessEqual(y, 0.05)  # the right arm reaches only 0.1 m past the midline, and low
        for y in T.PRESS_L:
            self.assertGreaterEqual(y, 0.05)
        self.assertLess(T.PRESS_Z, 0.065)  # the right arm's table-height reach at x 0.3, y 0 to 0.1 past the midline

    def test_flap_press_geometry(self):
        """The first flap press lands 49 mm out from the far flaps' crease (it still catches the underneath panel's
        flap standing 61 degrees), the last two 70 mm out, over the raised far flaps (right: the underneath and -y end
        panels'; left: the +y end panel's), inside each arm's reach; the last presses are both arms' last contact."""
        self.assertAlmostEqual(T.FLAP_PRESS_X - T.X_CREASE_W, 0.049, places=6)
        self.assertAlmostEqual(T.FLAP_FINAL_X - T.X_CREASE_W, 0.070, places=6)
        self.assertGreater(math.degrees(math.acos((T.FLAP_PRESS_X - T.X_CREASE_W) / L.FD)), 60.0)
        for x in (T.FLAP_PRESS_X, T.FLAP_FINAL_X):
            self.assertLess(T.X_CREASE_W + 0.030, x)
            self.assertLess(x + 0.020, T.X_CREASE_W + L.FD)
        for k, y in (("af", T.FLAP_PRESS_Y), ("df", T.FLAP_PRESS_Y), ("bf", T.FLAP_PRESS_YL)):
            lo, hi = T.FLAP_Y[k]
            self.assertLess(lo + 0.010, y, k)
            self.assertLess(y, hi - 0.010, k)
        self.assertLess(T.FLAP_PRESS_Z - T.PAD_TIP, L.PT + 2 * L.CT)
        self.assertGreater(T.FLAP_PRESS_Z - T.PAD_TIP, L.PT + L.CT)
        p = T.plans()
        labels = [w.label for w in p["right"]]
        i = labels.index("straight up, clear")
        self.assertIn("raised far flaps", labels[i + 1])
        self.assertTrue(labels[i + 5].startswith("over y -20"))
        for side in ("left", "right"):
            self.assertIn("the far flaps, last", p[side][-4].label)
            self.assertEqual(p[side][-1].duration, T.FINAL_UP_S)

    def test_hands_apart(self):
        """Sampled every 0.1 s along both plans (straight lines between waypoints), the two TCPs never come within
        0.25 m of each other while both are over the carton."""
        p = T.plans()

        def at(plan, t, start):
            prev, clock = np.asarray(start, dtype=float), 0.0
            for w in plan:
                tgt = np.asarray(w.pos, dtype=float)
                if t <= clock + w.duration:
                    return prev + (tgt - prev) * ((t - clock) / w.duration)
                prev, clock = tgt, clock + w.duration
            return prev

        end = max(T._duration(p[s]) for s in p)
        left0, right0 = (0.32, 0.305, L.TOP + 0.26), (0.32, -0.305, L.TOP + 0.26)
        closest = min(float(np.linalg.norm(at(p["left"], t / 10, left0) - at(p["right"], t / 10, right0)))
                      for t in range(int(end * 10) + 1))
        self.assertGreater(closest, 0.25)


class OfflineModel(unittest.TestCase):
    def feed(self, tcps, opening=0.0, steps=1):
        m = T.OfflineModel()
        for _ in range(steps):
            for tcp in tcps:
                arms = {"left": arm((0.1, 0.5, L.TOP + 0.25)), "right": arm(tcp, opening)}
                rigstate.offline_feed({a: {"tcp": arms[a].tcp, "tips": arms[a].tips, "opening": arms[a].opening}
                                       for a in arms})
                m.step(rigstate.arms())
        return m

    def test_push_leans_the_carton(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.5)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)])
        self.assertAlmostEqual(m.phi, 0.5, places=3)

    def test_weight_finishes_a_deep_push(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.9)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)], steps=60)
        self.assertAlmostEqual(m.phi, math.pi / 2, places=3)

    def test_shallow_push_stays_leaning(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.5)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)], steps=60)
        self.assertAlmostEqual(m.phi, 0.5, places=3)

    def test_open_gripper_does_not_push(self):
        m = self.feed([(L.CARTON_X, -0.15, L.TOP + L.PT + 0.12)], opening=1.0, steps=5)
        self.assertEqual(m.phi, 0.0)

    def test_high_gripper_does_not_push(self):
        m = self.feed([(L.CARTON_X, -0.15, L.TOP + L.PT + 0.25)], steps=5)
        self.assertEqual(m.phi, 0.0)

    def test_off_to_the_side_does_not_push(self):
        m = self.feed([(L.CARTON_X + 0.20, -0.15, L.TOP + L.PT + 0.12)], steps=5)  # past the flaps (177.8 mm)
        self.assertEqual(m.phi, 0.0)

    def test_states_at_rest_match_the_spawn(self):
        m = T.OfflineModel()
        st = m.states()
        for n in S.PARTS:
            self.assertTrue(np.allclose(st[n].pos, T.SPECS[n].rest_state().pos, atol=1e-6), n)
        self.assertEqual(S.Tracker.at_start(S.read(st)), True)

    def flat_model(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.9)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)], steps=60)
        self.assertAlmostEqual(m.phi, math.pi / 2, places=6)
        return m

    def press(self, m, x, y, tips_z, opening=0.0, steps=3):
        tcp = (x, y, tips_z + T.PAD_TIP)
        for _ in range(steps):
            arms = {"left": arm((0.1, 0.5, L.TOP + 0.25)), "right": arm(tcp, opening)}
            rigstate.offline_feed({a: {"tcp": arms[a].tcp, "tips": arms[a].tips, "opening": arms[a].opening}
                                   for a in arms})
            m.step(rigstate.arms())

    def test_stack_folds_the_top_flap_out_as_the_carton_comes_down(self):
        m = T.OfflineModel()
        self.assertAlmostEqual(m.off["cf"], math.radians(45))
        m = self.flat_model()
        self.assertLess(abs(math.degrees(m.off["cf"])), 3.0)
        for k in ("af", "bf", "df"):  # the crease set holds them where they are (no creep)
            self.assertAlmostEqual(m.off[k], L.FLAP_SET[k], places=9)
        self.assertFalse(S.is_flat(S.read(m.states())))

    def test_the_flap_press_folds_both_raised_flaps_out(self):
        m = self.flat_model()
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + T.PRESS_Z - T.PAD_TIP)
        self.assertLess(abs(math.degrees(m.off["af"])), 3.0)
        self.assertLess(abs(math.degrees(m.off["df"])), 1e-6)
        self.assertTrue(S.is_flat(S.read(m.states())))

    def test_a_press_that_misses_the_flaps_does_nothing(self):
        for x, y, opening in ((T.FLAP_PRESS_X, T.FLAP_PRESS_Y, 1.0), (L.CARTON_X, T.FLAP_PRESS_Y, 0.0),
                              (T.FLAP_PRESS_X + 0.05, T.FLAP_PRESS_Y, 0.0), (T.FLAP_PRESS_X, 0.10, 0.0)):
            m = self.flat_model()
            self.press(m, x, y, L.TOP + T.PRESS_Z - T.PAD_TIP, opening)
            self.assertFalse(S.is_flat(S.read(m.states())), (x, y, opening))

    def test_a_high_block_only_half_folds_a_flap(self):
        m = self.flat_model()
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + L.PT + 2 * L.CT + 0.030)
        self.assertGreater(abs(math.degrees(m.off["df"])), 15.0)
        self.assertLess(abs(math.degrees(m.off["df"])), 30.0)

    def test_the_end_panel_flap_cannot_be_pressed_before_the_carton_is_flat(self):
        m = T.OfflineModel()
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + T.PRESS_Z - T.PAD_TIP)
        self.assertEqual(m.off["df"], L.FLAP_SET["df"])

    def test_pressed_flaps_stay_down(self):
        """With the crease set a pressed flap, let go, springs back no more than the yield angle and then stays: a
        high block after the press, and 10 s more, change nothing; nothing creeps."""
        m = self.flat_model()
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + T.PRESS_Z - T.PAD_TIP, steps=10)
        pressed = dict(m.off)
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + L.PT + 0.08, steps=1)
        after = dict(m.off)
        for k in ("af", "df"):
            self.assertLessEqual(abs(after[k] - pressed[k]), math.radians(L.FLAP_YIELD_DEG) + 1e-9, k)
            self.assertLess(abs(math.degrees(after[k])), 5.0, k)
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + L.PT + 0.08, steps=200)
        self.assertEqual(m.off, after)
        self.assertTrue(S.is_flat(S.read(m.states())))

    def test_the_18_nov_model_creeps(self):
        """The 18 Nov flap model (friction_hold, the GPU run's): a press holds by friction, but once flat the lower
        layer's far flaps creep up again by themselves at the GPU trace's rates (2.8 and 1.3 degrees a second); the
        upper layer's stay put."""
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.9)
        m = T.OfflineModel(L.FLAP_MODELS["friction_hold"])
        for _ in range(60):
            rigstate.offline_feed({"left": {"tcp": (0.1, 0.5, L.TOP + 0.25), "tips": (0.1, 0.5, L.TOP + 0.237),
                                            "opening": 1.0},
                                   "right": {"tcp": (L.CARTON_X, y, L.TOP + L.PT + 0.12),
                                             "tips": (L.CARTON_X, y, L.TOP + L.PT + 0.107), "opening": 0.0}})
            m.step(rigstate.arms())
        self.assertAlmostEqual(m.phi, math.pi / 2, places=6)
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + T.PRESS_Z - T.PAD_TIP)
        before = dict(m.off)
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + L.PT + 0.08, steps=20)
        self.assertEqual(m.off["df"], before["df"])
        self.assertEqual(m.off["cf"], before["cf"])
        for k in ("af", "bf"):
            self.assertAlmostEqual(math.degrees(before[k] - m.off[k]), T.CREEP_DEG_S[k], places=6)

    def test_a_pad_beyond_a_steep_flaps_tip_misses_it(self):
        m = self.flat_model()
        m.off["af"] = m.set["af"] = -math.radians(50.0)  # tip 66 mm out: a pad 70 mm out comes down past it
        self.press(m, T.FLAP_FINAL_X, T.FLAP_PRESS_Y, L.TOP + T.FLAP_PRESS_Z - T.PAD_TIP, steps=1)
        self.assertAlmostEqual(math.degrees(m.off["af"]), -50.0, places=6)  # missed: it stays where it was
        self.press(m, T.FLAP_PRESS_X, T.FLAP_PRESS_Y, L.TOP + T.FLAP_PRESS_Z - T.PAD_TIP, steps=1)
        self.assertLess(abs(math.degrees(m.off["af"])), 3.0)

    def test_the_last_presses_leave_too_little_time_under_the_gpu_creep(self):
        """Why the flap model changed: the 12 mm thickness needs the underneath panel's far flap within about 5
        degrees of straight; at the 18 Nov GPU creep it passes that in about 1.7 s after release, and the policy's
        3 s hold alone is longer. The crease set's worst springback (the yield angle) is inside that 5 degrees, and
        an upper-layer flap's (3.2 degrees of room) is too."""
        thick_deg = math.degrees(math.asin((L.THICK_MAX - L.CT) / L.FD))  # 3.2 mm of board plus the tip's rise
        self.assertLess(thick_deg, 6.0)
        self.assertLess(thick_deg / T.CREEP_DEG_S["af"], 3.0)
        upper_deg = math.degrees(math.asin((L.THICK_MAX - 2 * L.CT) / L.FD))
        for name, m in L.FLAP_MODELS.items():
            if m["yield_deg"] is not None:
                self.assertLess(m["yield_deg"], min(thick_deg, upper_deg) - 0.1, name)


    def test_phi_never_decreases(self):
        y = T.D_FACE_Y - T.PAD_FACE + 0.12 * math.tan(0.6)
        m = self.feed([(L.CARTON_X, y, L.TOP + L.PT + 0.12)])
        self.assertAlmostEqual(m.phi, 0.6, places=3)
        self.feed([(L.CARTON_X, y - 0.05, L.TOP + L.PT + 0.12)])
        self.assertAlmostEqual(m.phi, 0.6, places=3)


class Files(unittest.TestCase):
    def test_kit_files_are_byte_identical(self):
        for f in ("mechkit.py", "rigstate.py"):
            if (KIT_DIR / f).is_file():
                self.assertEqual((HERE / f).read_bytes(), (KIT_DIR / f).read_bytes(), f)

    def test_task_json_matches_layout(self):
        d = json.loads((HERE / "task.json").read_text())
        setup, scoring = d["setup"], d["scoring"]
        self.assertEqual(d["category"], "Impact on jobs")
        for token in ("508 (y) x 356 (x) x 6.4 mm", "0.20 kg", "x 270 mm", "540 x 380 x 3 mm", "22.2 mm up",
                      "regular slotted carton (RSC), 12 x 8 x 6 in (305 x 203 x 152 mm)", "203 mm tall",
                      "y -254 to +51 mm", "0.01 N m", "97 degrees", "102 mm deep flap", "eight in all",
                      "lie out straight", "0.3 N m per radian", "more than 3 degrees", "takes a new set",
                      "x 92 to 448 mm",
                      "45 degrees below level", "30 degrees up off the stack", "30 degrees out past straight",
                      "40 to 55, 20 to 40, 20 to 40 and 0 to 10 degrees", "folds its drooping flap out",
                      "glued flat to the underneath panel", "516 mm", "356 mm wide", "6.4 mm thick",
                      "tape has just been slit", "knife step"):
            self.assertIn(token, setup)
        self.assertNotIn("cut off", setup)
        for token in ("15 degrees", "12 mm or less", "540 mm (y) x 380 mm (x)", "60 mm", "6 mm",
                      "120 mm and 20 mm", "15 mm", "4 mm up", "8 degrees", "30 mm", "every flap within 15 degrees",
                      "5 mm or more", "5 degrees or more", "no visible motion", "top video", "side video",
                      "40 mm", "30 degrees", "110 mm tall", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0", "1 s"):
            self.assertIn(token, scoring)
        for gone in ("0.3 degrees", "dial", "1 mm or"):
            self.assertNotIn(gone, scoring)
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["run_limit"], "1200 steps (60 s at 20 Hz)")
        self.assertAlmostEqual(L.PILE_X, 0.270)
        self.assertAlmostEqual(L.A_YMIN, -0.254, places=3)
        self.assertAlmostEqual(L.A_YMAX, 0.0508, places=3)
        self.assertAlmostEqual(L.PT, 0.0222, places=4)
        self.assertEqual(L.THICK_MAX, 0.012)
        self.assertEqual((L.FOOT_Y, L.FOOT_X), (0.540, 0.380))
        self.assertAlmostEqual(L.BOARD_X, 0.380)
        self.assertEqual(round(L.CARTON_MASS, 2), 0.20)
        self.assertEqual(round(L.FLAP_SPRING, 3), 0.3)
        self.assertEqual(L.FLAP_YIELD_DEG, 3.0)
        self.assertEqual(L.FLAP_FRICTION, 0.0)
        self.assertNotIn("friction (est), as a folded crease", setup)
        self.assertEqual({k: round(math.degrees(abs(v))) for k, v in L.FLAP_SET.items() if v},
                         {"cf": 45, "af": 30, "df": 30})
        self.assertEqual(L.FLAP_START_DEG, {"cf": 45.0, "af": 30.0, "df": 30.0})
        self.assertIn("angle gauge", d["reset_notes"])
        self.assertNotIn("opened out straight", d["reset_notes"])
        scores = d.get("scores", "")  # before the GPU run: empty or TBD; after it: the real Dream-1 run and demo 1.0
        self.assertTrue(scores == "" or scores.startswith("TBD")
                        or re.fullmatch(r"Dream-1 run: [^;]+; demo: 1\.0( \(\d+ steps\))?", scores), scores)
        media = d.get("media", "")
        self.assertTrue(media == "" or re.fullmatch(r"M-[A-Z0-9]{6}", media), media)

    def test_fidelity_states_the_simplifications(self):
        fid = json.loads((HERE / "task.json").read_text())["fidelity"]
        for token in ("0.01 N m", "0.3 N m/rad", "3 degrees", "new set", "toward their set", "eight flaps",
                      "never peels", "pass through one another", "est"):
            self.assertIn(token, fid)

    def test_context(self):
        ctx = json.loads((HERE / "task.json").read_text())["context"]
        for token in ("Stockers and order fillers", "53-7065", "BLS", "knife step", "flaps hanging loose"):
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
            for i, line in enumerate(text.splitlines() if f != "task.json" else [], 1):
                self.assertLessEqual(len(line), 120, f"{f}:{i}")

    def test_readme_says_offline(self):
        text = (HERE / "README.md").read_text()
        self.assertIn("Offline prediction", text)
        self.assertIn("Simulator (Isaac Lab-Arena", text)
        self.assertNotIn("TK-", text)
        self.assertIn("Revision (2026-11-18)", text)
        self.assertNotIn("dial indicator", text.split("## Revision")[0])


if __name__ == "__main__":
    unittest.main()
