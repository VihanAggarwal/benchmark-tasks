"""Offline tests for layout.py, parts.py, scoring.py, offline_collar.py and task.py: synthetic part states (mechkit
AssemblyStates from the task's own specs) and synthetic arm snapshots (rigstate.Arm), no simulator.
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
    import offline_collar  # noqa: E402
    import parts as P  # noqa: E402
    import rigstate  # noqa: E402
    import scoring as S  # noqa: E402
    import task as T  # noqa: E402
except ModuleNotFoundError as e:
    if e.name != "yam_arena" and not str(e.name).startswith("yam_arena."):
        raise
    raise unittest.SkipTest("yam_arena is not importable: run inside its environment or next to tools/offline")

HZ = 20.0
WIN = L.WINDOW
KIT_DIR = HERE.parents[1] / "tools" / "kit"
Q0 = (0.0, 0.0, 0.0, 1.0)
B0 = np.array(L.B_ORIGIN, dtype=float)
REST_ROLL = [L.link_angle(L.VERTS[i - 1], L.VERTS[i]) for i in range(1, L.N_LINKS + 1)]


def rx(a):
    return offline_collar.rx(a)


# -- states --------------------------------------------------------------------------------------------------------
def collar_rest(offset=(0.0, 0.0, 0.0)):
    return mk.state_from_poses(T.COLLAR_PART, (*(B0 + np.asarray(offset)), *Q0), source="test")


def male_at(slide=0.0, prongs=0.0, offset=(0.0, 0.0, 0.0)):
    o = B0 + np.asarray(offset, dtype=float) - np.array([0.0, slide, 0.0])
    return mk.state_from_poses(T.MALE_PART, (*o, *Q0), joints={"prong_p": prongs, "prong_n": prongs}, source="test")


def dog_at(offset=(0.0, 0.0, 0.0)):
    o = np.array([L.DOG_ORIGIN[0], L.DOG_ORIGIN[1], L.TOP]) + np.asarray(offset, dtype=float)
    return mk.state_from_poses(T.DOG_PART, (*o, *Q0), source="test")


def laid_out(y0=-0.500, x=L.BX, lift=0.0):
    """The open collar lying straight along +y on the table: female (its mouth at y0 + F_Y0 ... its back), the 13 links,
    then the male, bar first. Returns (collar state, male state)."""
    f_o = np.array([x, y0, L.TOP + 0.0005 + lift])
    poses = {}
    for i in range(1, L.N_LINKS + 1):
        p0 = f_o + np.array([0.0, L.F_Y1 + (i - 1) * L.PITCH, L.END_Z])
        poses[f"w{i}"] = (*p0, *mk.matrix_to_quat(rx(-REST_ROLL[i - 1])))
    bar = f_o + np.array([0.0, L.F_Y1 + L.N_LINKS * L.PITCH, L.END_Z])
    m_o = bar - np.array([0.0, L.M_Y0, L.END_Z])
    col = mk.state_from_poses(T.COLLAR_PART, (*f_o, *Q0), link_poses=poses, source="test")
    male = mk.state_from_poses(T.MALE_PART, (*m_o, *Q0), source="test")
    return col, male


def scene(col=None, male=None, dog=None):
    return {S.COLLAR: col if col is not None else collar_rest(), S.MALE: male if male is not None else male_at(),
            S.DOG: dog if dog is not None else dog_at()}


def done_scene(**kw):
    col, male = laid_out(**kw)
    return scene(col, male)


def arm(tcp, opening=1.0, approach=(0.0, 0.0, -1.0)):
    tcp = np.asarray(tcp, dtype=float)
    a = np.asarray(approach, dtype=float)
    return rigstate.Arm(tcp, tcp + 0.013 * a / np.linalg.norm(a), opening)


FAR = {"left": arm((0.15, 0.40, L.TOP + 0.20)), "right": arm((0.15, -0.30, L.TOP + 0.20))}


def run(states, arms=FAR, steps=WIN + 3, tracker=None):
    tr = tracker or S.Tracker()
    res = None
    for _ in range(steps):
        res = tr.update(states, arms)
    return res


def pad_clearance(tip, yaw, roll, gap):
    """The closest a pad face comes to the neck (m): pads 27 mm wide, 20 mm tall, 8 mm thick, at the commanded gap."""
    import tiltgrip
    r = tiltgrip.rotation(yaw, roll)
    tip = np.asarray(tip, dtype=float)
    best = math.inf
    for u in np.linspace(-rigstate.FINGER_HALF_WIDTH, rigstate.FINGER_HALF_WIDTH, 11):
        for v in np.linspace(0.0, 0.020, 9):
            for w in np.linspace(gap / 2, gap / 2 + rigstate.FINGER_THICK, 5):
                for sg in (1.0, -1.0):
                    p = tip + u * r[:, 1] - v * r[:, 2] + sg * w * r[:, 0]
                    best = min(best, math.hypot(p[1], p[2] - L.NECK_Z) - L.NECK_R)
    return best


# -- replay of the scripted plans through the offline model ----------------------------------------------------------
def replay(plans, extra_s=3.0):
    """Commanded TCPs, pad tips and openings of ``plans`` (missing arms hold the ready pose, open) at 20 Hz, fed to a
    fresh CollarModel and Tracker. Returns (last Result, model)."""
    import yamkin as K
    from yam_arena.demo import ArmPlan
    bases = K.BASES["station"]
    ready = {a: K.tcp_pose(K.READY, bases[a]) for a in ("left", "right")}
    aplans = {a: ArmPlan(p, bases[a]) for a, p in plans.items()}
    dur = max([p.duration for p in aplans.values()] + [1.0])
    model = offline_collar.CollarModel(T.COLLAR_PART, T.MALE_PART, T.DOG_PART)
    tr = S.Tracker()
    res = None
    for i in range(int(round((dur + extra_s) * HZ))):
        t = i / HZ
        arms = {}
        for a in ("left", "right"):
            if a in aplans:
                pos, rot, finger, _ = aplans[a].target(t)
            else:
                (pos, rot), finger = ready[a], K.FINGER_OPEN
            arms[a] = rigstate.Arm(np.asarray(pos), np.asarray(pos) + 0.013 * np.asarray(rot)[:, 2],
                                   float(K.finger_gap(finger) / rigstate.FULL_GAP))
        model.step(arms)
        res = tr.update(model.states(S.PARTS), arms)
    return res, model


# ======================================================================================================================
class Geometry(unittest.TestCase):
    def test_look_only_parts_keep_out_of_the_collars_region(self):
        """Only the skull (to x 265), the neck, the chest, the flank and the first front leg reach into x 258 to 419 mm
        (table frame), where the collar, the buckle and the grippers work; the ears, nose, eye, cheeks, shoulder and
        tail stay out of it."""
        inside = []
        for s in T.DOG_PART.body_shapes("base"):
            c, h = L.DOG_ORIGIN[0] + s.center[0], s.half_extents()[0]
            if c + h > 0.258 and c - h < 0.419:
                inside.append((s.kind, round(float(c - h), 4), round(float(c + h), 4)))
        self.assertEqual(sorted(inside), [("box", 0.395, 0.755), ("cylinder", 0.175, 0.265),
                                          ("cylinder", 0.265, 0.38), ("cylinder", 0.38, 0.77),
                                          ("cylinder", 0.4, 0.46)])

    def test_loop_neck_and_head(self):
        self.assertAlmostEqual(L.LOOP_LEN, 0.381, places=6)
        neck, head = 2 * math.pi * L.NECK_R, 2 * math.pi * L.HEAD_R
        self.assertAlmostEqual(neck, 0.327, delta=0.001)
        self.assertAlmostEqual(head, 0.415, delta=0.001)
        self.assertGreater(L.LOOP_LEN - neck, 0.050)  # the two-finger room
        self.assertGreater(head - L.LOOP_LEN, 0.030)  # the setup check: it cannot come off over the head
        self.assertAlmostEqual(L.NECK_GAP, 0.025)
        self.assertAlmostEqual(L.NECK_TOP, 0.129)

    def test_chain_is_equal_chords_ending_at_the_bar(self):
        v = L.VERTS
        self.assertEqual(len(v), L.N_LINKS + 1)
        for i in range(L.N_LINKS):
            self.assertAlmostEqual(float(np.linalg.norm(v[i + 1] - v[i])), L.PITCH, places=6)
        self.assertLess(float(np.linalg.norm(v[-1] - L.P_MALE)), 1e-6)
        self.assertLess(float(np.linalg.norm(v[0] - L.P_FEMALE)), 1e-9)
        self.assertLess(L.LINK_LEN, L.PITCH)

    def test_buckle_sizes(self):
        self.assertAlmostEqual((L.F_Y1 - L.M_Y0) * 1000, 56.0, places=6)  # 56 mm closed
        self.assertAlmostEqual(2 * L.F_HALF_W, 0.029)
        self.assertAlmostEqual(L.F_H, 0.011)
        self.assertAlmostEqual(2 * L.M_HALF_W, 0.025)
        self.assertAlmostEqual((L.M_Y1 - L.M_Y0), 0.0217, places=6)  # 22 mm showing
        self.assertAlmostEqual((L.TAB_Y[0] + L.TAB_Y[1]) / 2 - L.F_Y0, 0.014)  # tabs 14 mm in from the mouth
        self.assertAlmostEqual(L.TAB_PROUD, 0.0015)
        self.assertAlmostEqual(L.CATCH, 0.0013)
        lo, hi = T.COLLAR_PART.local_bounds()
        self.assertGreater(hi[2] - lo[2], 0.10)

    def test_masses(self):
        self.assertAlmostEqual(P.collar_mass(), 0.0325, delta=0.0005)
        self.assertGreater(P.collar_mass(), 0.010)
        self.assertLess(P.collar_mass(), 0.040)
        self.assertEqual(L.DOG_MASS, 6.8)

    def test_rest_pose_is_clear(self):
        """At rest: the male clear of the female's shapes, the webbing clear of the neck, the buckle 0.5 mm over it."""
        col, male, dog = collar_rest(), male_at(), dog_at()
        mpts = np.vstack([male.link_to_world(b, S.points_of(male.spec, b)) for b in male.spec.body_names])
        self.assertGreater(float(np.min(col.distance(mpts, bodies=("base",)))), 0.0001)
        for b in col.spec.body_names:
            d = float(np.min(dog.distance(col.link_to_world(b, S.points_of(col.spec, b)))))
            self.assertGreater(d, 0.0004 if b == "base" else 0.002, b)
        r = S.read(scene())
        self.assertGreater(float(r.collar_pts[:, 2].min()) - L.TOP, 0.015)

    def test_the_catch(self):
        """Pressed flush, each hook turns 2.2 mm in, past the 1.3 mm rib; it clears it with its tab 0.9 mm in."""
        self.assertAlmostEqual(L.FLUSH_RAD * L.HOOK_LEVER, 0.00219, delta=0.00002)
        self.assertGreater(L.FLUSH_RAD, L.RELEASE_RAD * 1.5)
        self.assertAlmostEqual(L.RELEASE_RAD * L.TAB_LEVER, 0.00089, delta=0.00002)
        self.assertLess(L.FLUSH_RAD, L.PRONG_UPPER)
        hook = (L.HOOK_X1, L.HOOK_Y[0], sum(L.PRONG_Z) / 2)
        for sign, name in ((1.0, "prong_p"), (-1.0, "prong_n")):
            corner = (sign * hook[0], hook[1], hook[2])
            for q, inside in ((0.0, False), (L.FLUSH_RAD, True)):
                x = float(male_at(prongs=q).link_to_world(name, corner)[0] - L.BX)
                if inside:
                    self.assertLess(abs(x), L.RIB_X0 - 0.0005, (name, q))  # the hook is in past the rib's face
                else:
                    self.assertAlmostEqual(abs(x), L.HOOK_X1, places=6)
            tab = male_at(prongs=L.FLUSH_RAD).link_to_world(name, (sign * L.TAB_X1, sum(L.TAB_Y) / 2, 0.0055))
            self.assertAlmostEqual(abs(float(tab[0] - L.BX)), L.F_HALF_W, delta=0.0002)  # flush with the side
        self.assertAlmostEqual(L.PLAY, 0.0003, places=6)
        self.assertLess(L.PLAY, L.UNLATCH_OUT)

    def test_hook_meets_the_rib_when_pulled_unpressed(self):
        """Clicked and unpressed, the male pulled 0.4 mm overlaps the rib (it would hit it); pressed flush it does not."""
        col = collar_rest()
        for q, hits in ((0.0, True), (L.FLUSH_RAD, False)):
            male = male_at(slide=L.PLAY + 0.0001, prongs=q)
            pts = np.vstack([male.link_to_world(n, S.points_of(male.spec, n)) for n in ("prong_p", "prong_n")])
            d = float(np.min(col.distance(pts, bodies=("base",))))
            if hits:
                self.assertLess(d, 1e-6)  # touching or overlapping (distances are not signed)
            else:
                self.assertGreater(d, 0.0002)

    def test_tab_force_and_pinch(self):
        self.assertAlmostEqual(L.PRONG_SPRING * L.FLUSH_RAD / L.TAB_LEVER, L.TAB_FORCE_FLUSH, places=6)
        self.assertAlmostEqual(L.pinch_force(L.PINCH_GAP, 2 * L.F_HALF_W), 5.0, places=6)
        self.assertAlmostEqual(L.pinch_force(L.MALE_GAP, 2 * L.M_HALF_W), 5.0, places=6)
        self.assertGreater(L.pinch_force(L.PINCH_GAP, 2 * L.F_HALF_W), L.TAB_FORCE_FLUSH)
        for gap in (L.OPEN_GAP_FEMALE, L.OPEN_GAP_MALE, L.PINCH_GAP, L.MALE_GAP):
            self.assertGreaterEqual(gap, 0.016)
            self.assertLessEqual(gap, 0.080)
        self.assertGreater(L.OPEN_GAP_FEMALE, 2 * L.TAB_X1 + 0.008)  # opens clear of both tabs
        self.assertGreater(L.OPEN_GAP_MALE, 2 * L.M_HALF_W + 0.008)

    def test_pads_cover_the_tabs_and_clear_the_neck(self):
        x, y, z = T.PINCH
        self.assertLessEqual(y - rigstate.FINGER_HALF_WIDTH, L.TAB_Y[0] + L.BY)
        self.assertGreaterEqual(y + rigstate.FINGER_HALF_WIDTH, L.TAB_Y[1] + L.BY)
        self.assertLess(z - L.BZ0, L.PRONG_Z[0])
        for tip, yaw, roll, gaps in ((T.PINCH, T.YAW_L, L.ROLL_L, (L.OPEN_GAP_FEMALE, L.PINCH_GAP)),
                                     (T.GRIP, T.YAW_R, L.ROLL_R, (L.OPEN_GAP_MALE, L.MALE_GAP))):
            for gap in gaps:
                self.assertGreater(pad_clearance(tip, yaw, roll, gap), 0.002, (tip, gap))
        # the right pads stop short of the female's mouth
        self.assertLess(T.GRIP[1] + rigstate.FINGER_HALF_WIDTH, L.F_Y0 - 0.002)


# ======================================================================================================================
class Tiers(unittest.TestCase):
    def test_start_is_zero(self):
        r = run(scene())
        self.assertEqual(r.score, 0.0)
        self.assertFalse(r.details["unlatched"])

    def test_unlatched_is_quarter(self):
        r = run(scene(male=male_at(slide=0.010, prongs=L.FLUSH_RAD)))
        self.assertEqual(r.score, 0.25)
        self.assertTrue(r.details["unlatched"])
        self.assertFalse(r.details["apart"])
        self.assertEqual(run(scene(male=male_at(slide=0.0029))).score, 0.0)
        self.assertEqual(run(scene(male=male_at(slide=0.0031))).score, 0.25)

    def test_apart_on_the_neck_is_half(self):
        r = run(scene(male=male_at(slide=0.040)))
        self.assertEqual(r.score, 0.5)
        self.assertTrue(r.details["apart"])
        self.assertFalse(r.details["off_dog"])
        self.assertEqual(run(scene(male=male_at(slide=L.OUT_FULL + 0.004))).score, 0.25)  # tips 4 mm out
        self.assertEqual(run(scene(male=male_at(slide=L.OUT_FULL + 0.006))).score, 0.5)

    def test_laid_out_and_let_go_is_one(self):
        st = done_scene()
        tr = S.Tracker()
        for i in range(WIN + 1):  # the first step has no speed yet (not resting), so 21 steps
            r = tr.update(st, FAR)
            self.assertEqual(r.score, 1.0 if i == WIN else 0.75, i)
        self.assertTrue(r.success)
        self.assertGreaterEqual(r.details["dog_gap_mm"], 25.0)

    def test_held_or_touched_is_three_quarters(self):
        col, male = laid_out()
        m_o = male.origin
        hold = dict(FAR, right=arm((m_o[0], m_o[1] - 0.015, m_o[2] + 0.005 + 0.013), opening=L.MALE_GAP / rigstate.FULL_GAP))
        self.assertEqual(run(scene(col, male), hold).score, 0.75)
        near = dict(FAR, right=arm((m_o[0], m_o[1] - 0.015, m_o[2] + 0.040 + 0.013), opening=1.0))
        r = run(scene(col, male), near)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["hands_off"])

    def test_raised_is_three_quarters(self):
        self.assertEqual(run(done_scene(lift=0.028)).score, 1.0)  # the female's top 39.5 mm up
        r = run(done_scene(lift=0.030))
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.details["resting"])

    def test_close_to_the_dog_is_half(self):
        """Laid along the neck's -y side at x 320: the male's end 20 mm from the neck is still on the dog."""
        def y_for(gap):
            lo, hi = -0.50, -0.38
            for _ in range(40):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if S.read(scene(*laid_out(y0=mid))).dog_gap > gap else (lo, mid)
            return lo
        r = run(scene(*laid_out(y0=y_for(0.020))))
        self.assertEqual(r.score, 0.5)
        self.assertAlmostEqual(r.details["dog_gap_mm"], 20.0, delta=0.5)
        self.assertEqual(run(scene(*laid_out(y0=y_for(0.0245)))).score, 0.5)
        self.assertEqual(run(scene(*laid_out(y0=y_for(0.0255)))).score, 1.0)

    def test_moving_is_three_quarters(self):
        tr = S.Tracker()
        for i in range(WIN + 2):
            r = tr.update(done_scene(y0=-0.500 + 0.0004 * i), FAR)  # 8 mm/s
        self.assertEqual(r.score, 0.75)
        tr = S.Tracker()
        for i in range(WIN + 2):
            r = tr.update(done_scene(y0=-0.500 + 0.0002 * i), FAR)  # 4 mm/s
        self.assertEqual(r.score, 1.0)


class Latches(unittest.TestCase):
    def test_below_the_table_latches(self):
        tr = S.Tracker()
        tr.update(done_scene(lift=-0.004), FAR)
        r = run(done_scene(), tracker=tr)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["latch_table"])
        self.assertEqual(run(done_scene(lift=-0.0005)).score, 1.0)

    def test_off_the_table_edge_latches(self):
        tr = S.Tracker()
        tr.update(done_scene(y0=-0.660), FAR)
        self.assertEqual(run(done_scene(), tracker=tr).score, 0.0)

    def test_manikin_dragged_latches(self):
        tr = S.Tracker()
        tr.update(done_scene(), FAR)
        col, male = laid_out()
        r = run(scene(col, male, dog_at((0.0, 0.1005, 0.0))), tracker=tr)
        self.assertEqual(r.score, 0.0)
        self.assertTrue(r.details["latch_dog"])

    def test_manikin_nudged_caps_at_three_quarters(self):
        col, male = laid_out()
        tr = S.Tracker()
        tr.update(done_scene(), FAR)
        r = run(scene(col, male, dog_at((0.0, 0.0205, 0.0))), tracker=tr)
        self.assertEqual(r.score, 0.75)
        self.assertTrue(r.details["dog_nudged"])
        self.assertFalse(r.details["latch_dog"])
        tr = S.Tracker()
        tr.update(done_scene(), FAR)
        r = run(scene(col, male, dog_at((0.060, 0.0, 0.0))), tracker=tr)
        self.assertEqual(r.score, 0.75)
        tr = S.Tracker()
        tr.update(done_scene(), FAR)
        self.assertEqual(run(scene(col, male, dog_at((0.018, 0.0, 0.0))), tracker=tr).score, 1.0)

    def test_nudge_counts_from_any_step(self):
        # nudged mid-trial and put back: still capped (the latch-style "at any step" rule)
        col, male = laid_out()
        tr = S.Tracker()
        tr.update(done_scene(), FAR)
        tr.update(scene(col, male, dog_at((0.0, 0.025, 0.0))), FAR)
        self.assertEqual(run(done_scene(), tracker=tr).score, 0.75)

    def test_dog_turned_is_read_at_the_nose(self):
        a = math.radians(5.0)  # 5 degrees about the vertical through the neck: the nose moves about 20 mm
        q = (0.0, 0.0, math.sin(a / 2), math.cos(a / 2))
        dog = mk.state_from_poses(T.DOG_PART, (L.DOG_ORIGIN[0], 0.0, L.TOP, *q), source="test")
        tr = S.Tracker()
        tr.update(done_scene(), FAR)
        col, male = laid_out()
        r = run(scene(col, male, dog), tracker=tr)
        moved = r.details["dog_moved_mm"]
        self.assertGreater(moved, 15.0)
        self.assertEqual(r.score, 0.75 if moved >= 20.0 else 1.0)

    def test_reset_clears_latches(self):
        tr = S.Tracker()
        tr.update(done_scene(lift=-0.010), FAR)
        tr.reset()
        self.assertEqual(run(done_scene(), tracker=tr).score, 1.0)


class BadStates(unittest.TestCase):
    def test_missing_and_nan(self):
        tr = S.Tracker()
        self.assertEqual(tr.update({}, FAR).score, 0.0)
        self.assertTrue(tr.update(None, FAR).details["bad_state"])
        st = done_scene()
        st[S.DOG] = None
        self.assertIsNone(S.read(st))
        col, male = laid_out()
        bad = mk.state_from_poses(T.MALE_PART, (math.nan, 0.0, L.TOP, *Q0), source="test")
        self.assertIsNone(S.read(scene(col, bad)))

    def test_a_bad_step_restarts_the_window_and_caps(self):
        tr = S.Tracker()
        r = run(done_scene(), tracker=tr)
        self.assertEqual(r.score, 1.0)
        r = tr.update({}, FAR)
        self.assertEqual(r.score, 0.75)
        self.assertFalse(r.success)
        for i in range(WIN - 1):
            r = tr.update(done_scene(), FAR)
            self.assertLess(r.score, 1.0, i)
        r = tr.update(done_scene(), FAR)
        self.assertLess(r.score, 1.0)  # the window holds the bad step for 20 steps after it
        r = tr.update(done_scene(), FAR)
        self.assertEqual(r.score, 1.0)

    def test_unreadable_arms_never_hold_one(self):
        r = run(done_scene(), arms=None)
        self.assertEqual(r.score, 0.75)
        r = run(done_scene(), arms={"left": FAR["left"]})
        self.assertEqual(r.score, 0.75)
        r = run(done_scene(), arms={"left": FAR["left"], "right": arm((math.nan, 0.0, 1.0))})
        self.assertEqual(r.score, 0.75)


class Exploits(unittest.TestCase):
    def test_collar_pulled_round_but_still_buckled(self):
        """The whole closed collar slid 30 mm along the neck's axis (out of plane) is still buckled: 0."""
        r = run(scene(collar_rest((0.030, 0.0, 0.0)), male_at(offset=(0.030, 0.0, 0.0))))
        self.assertEqual(r.score, 0.0)

    def test_male_alone_carried_off_but_still_clipped(self):
        r = run(scene(collar_rest((0.0, -0.3, -0.10)), male_at(offset=(0.0, -0.3, -0.10))))
        self.assertEqual(r.score, 0.0)

    def test_unclipped_and_left_on_the_neck(self):
        self.assertEqual(run(scene(male=male_at(slide=0.060))).score, 0.5)

    def test_whole_collar_lifted_off_and_held_up(self):
        r = run(done_scene(lift=0.150))
        self.assertEqual(r.score, 0.75)


class OfflineModelTests(unittest.TestCase):
    def setUp(self):
        for k in ("DEMO_VARIANT", "DEMO_HOLD_ONLY", "DEMO_IDLE_ARM"):
            os.environ.pop(k, None)

    def pinch_arm(self, gap):
        w = T._w(1.0, T.PINCH, -gap / 2, "test", "left")
        tcp = np.array(w.pos, dtype=float)
        return rigstate.Arm(tcp, tcp + T.PAD_TIP * T.A_L, gap / rigstate.FULL_GAP)

    def grip_arm(self, gap, dy=0.0):
        w = T._w(1.0, (T.GRIP[0], T.GRIP[1] + dy, T.GRIP[2]), -gap / 2, "test", "right")
        tcp = np.array(w.pos, dtype=float)
        return rigstate.Arm(tcp, tcp + T.PAD_TIP * T.A_R, gap / rigstate.FULL_GAP)

    def model(self):
        return offline_collar.CollarModel(T.COLLAR_PART, T.MALE_PART, T.DOG_PART)

    def test_pinch_presses_both_tabs_past_release(self):
        m = self.model()
        m.step({"left": self.pinch_arm(L.PINCH_GAP), "right": FAR["right"]})
        self.assertEqual(m.holds.get("left", (None,))[0], "female")
        self.assertGreaterEqual(min(m.prongs), L.RELEASE_RAD)
        self.assertAlmostEqual(min(m.prongs), L.FLUSH_RAD, places=6)
        self.assertEqual(m.mode, "sliding")

    def test_open_fingers_do_not_press(self):
        m = self.model()
        m.step({"left": self.pinch_arm(L.OPEN_GAP_FEMALE), "right": FAR["right"]})
        self.assertEqual(m.prongs, [0.0, 0.0])
        self.assertEqual(m.mode, "clicked")

    def test_pulling_the_male_alone_slips(self):
        m = self.model()
        for k in range(150):  # 150 mm: the loop's 54 mm of room runs out round the neck
            m.step({"left": FAR["left"], "right": self.grip_arm(L.MALE_GAP, -0.001 * k)})
        self.assertEqual(m.mode, "clicked")
        self.assertGreater(m.slips, 0)
        self.assertLess(m.slide(), L.UNLATCH_OUT)

    def test_pinch_and_pull_comes_apart(self):
        m = self.model()
        for k in range(50):
            m.step({"left": self.pinch_arm(L.PINCH_GAP), "right": self.grip_arm(L.MALE_GAP, -0.001 * k)})
        self.assertEqual(m.mode, "apart")
        self.assertEqual(m.slips, 0)
        r = run(m.states(S.PARTS), FAR)
        self.assertEqual(r.score, 0.5)

    def test_demo_replay_scores_one(self):
        r, m = replay(T.plans())
        self.assertEqual(r.score, 1.0)
        self.assertEqual(m.slips, 0)
        self.assertEqual(m.mode, "apart")

    def test_one_arm_and_counterfactuals(self):
        p = T.plans()
        self.assertEqual(replay({"left": p["left"]})[0].score, 0.0)
        self.assertEqual(replay({"right": p["right"]})[0].score, 0.0)
        self.assertEqual(replay({})[0].score, 0.0)
        self.assertEqual(replay(T.plans("early"))[0].score, 0.0)
        self.assertEqual(replay(T.plans("short"))[0].score, 0.5)

    def test_demo_plans_and_variants(self):
        self.assertEqual(set(T.demo()), {"left", "right"})
        for v in T.VARIANTS:
            os.environ["DEMO_VARIANT"] = v
            try:
                self.assertEqual(set(T.demo()), {"left", "right"})
            finally:
                os.environ.pop("DEMO_VARIANT")
        os.environ["DEMO_VARIANT"] = "nonsense"
        try:
            with self.assertRaises(ValueError):
                T.demo()
        finally:
            os.environ.pop("DEMO_VARIANT")

    def test_hold_only_and_idle_arm(self):
        os.environ["DEMO_HOLD_ONLY"] = "1"
        try:
            self.assertEqual(T.demo(), {})
        finally:
            os.environ.pop("DEMO_HOLD_ONLY")
        for side, other in (("left", "right"), ("right", "left")):
            os.environ["DEMO_IDLE_ARM"] = side
            try:
                d = T.demo()
            finally:
                os.environ.pop("DEMO_IDLE_ARM")
            self.assertNotIn(side, d)
            self.assertIn(other, d)

    def test_timing(self):
        p = T.plans()
        pinched = T._until(p["left"], "settle the pinch")
        self.assertLessEqual(pinched, T._until(p["right"], "wait for the left pinch"))
        pulled = T._until(p["right"], "pull the male")
        self.assertGreater(T._until(p["left"], "hold the tabs in"), pulled + 0.4)
        self.assertGreaterEqual(T._until(p["right"], "wait for the left hand to clear"),
                                T._until(p["left"], "straight up"))
        import yamkin as K
        for plan in p.values():  # every opening or closing of the fingers takes at least 1.5 s
            prev = K.FINGER_OPEN
            for w in plan:
                if abs(w.finger - prev) > 1e-9:
                    self.assertGreaterEqual(w.duration, 1.5, w.label)
                prev = w.finger

    def test_demo_length_is_within_half_the_run_limit(self):
        limit = int(json.loads((HERE / "task.json").read_text())["run_limit"].split()[0])
        steps = max(sum(w.duration for w in plan) for plan in T.demo().values()) * HZ
        self.assertLess(steps + 60, limit / 2)
        self.assertGreater(steps, 400)

    def test_right_wrist_keeps_its_yaw_while_it_holds_the_male(self):
        """GPU run 20261119T034831 (demo 0.5): the right wrist turned 0.35 rad of yaw while it held the male; the
        male's yaw sets the plane of the open strap (its hinges all turn about the female's x), so the female, 300 mm
        behind, was steered 66 mm toward +x, past the neck's end, and jammed on the flank. From the grip to the let-go
        the pads must close along world x at every control step (a roll about x is free: it leaves the plane alone)."""
        import yamkin as K
        from yam_arena.demo import ArmPlan
        plan = T.plans()["right"]
        t0, t1 = T._until(plan, "grip the male's body"), T._until(plan, "let go")
        arm = ArmPlan(list(plan), K.BASES["station"]["right"])
        worst = 0.0
        for k in range(int(t0 * HZ), int(t1 * HZ) + 1):
            rot = arm.target(k / HZ)[1]
            worst = max(worst, math.degrees(math.acos(min(1.0, abs(float(rot[0, 0]))))))
        self.assertLess(worst, 0.1, f"pad axis off world x by {worst:.2f} deg while holding the male")
        # far out the straight-down wrist stays inside its limits
        tip = (T.GRIP[0], T.END_Y, T.DRAW_Z)
        w = T._w(1.0, tip, T.M_GRIP, "test", "drag")
        q, pe, re = K.solve(w.pos, K.matrix_to_quat_xyzw(K.grip_rotation(w.yaw)), None, K.BASES["station"]["right"])
        self.assertLess(pe, 0.001)
        self.assertGreater(float(np.min(K.joint_margins(q))), math.radians(3.0))


class Folder(unittest.TestCase):
    def test_text_has_no_em_dashes_or_identifying_names(self):
        bad = re.compile(chr(0x2014) + "|" + chr(0x2013) + "|T-70|VihanAggarwal|Vihan|Claude|Anthropic|co-author", re.I)
        for name in ("task.json", "README.md", "task.py", "layout.py", "scoring.py", "parts.py", "offline_collar.py",
                     "tiltgrip.py"):
            p = HERE / name
            if p.exists():
                self.assertIsNone(bad.search(p.read_text()), name)

    def test_task_json_matches_the_code(self):
        d = json.loads((HERE / "task.json").read_text())
        self.assertEqual(d["instruction"], T.INSTRUCTION)
        self.assertEqual(d["category"], "Societal relatability")
        self.assertEqual(d["scores"], "Dream-1 run: 0.0; demo: 1.0 (one arm idle: 0.0 left idle, 0.0 right idle)")
        text = d["setup"] + d["scoring"]
        for needle in ("6.8 kg", "(88, 8)", "x 800", "265 to 380", "104 mm across", "327 mm round", "129 mm up",
                       "25 mm above the table", "132 mm across", "415 mm round", "33 g", "381 mm", "x 320",
                       "56 x 29 x 11 mm", "y -4 to +30", "y -26 to -4", "25 mm wide", "14 mm in from the mouth",
                       "1.5 mm proud", "1.3 mm rib", "2.5 N", "0.9 mm in", "25 x 20 mm", "3 mm below", "100 mm or more", "20 mm or more", "US$460",
                       "3 mm or more out", "5 mm or more", "25 mm or more", "40 mm above", "5 mm/s", "30 mm or more",
                       "20 steps", "Possible scores: 0, 0.25, 0.5, 0.75, 1.0"):
            self.assertIn(needle, text, needle)
        self.assertEqual(d["run_limit"].split()[0], "1600")
        self.assertIn("13 rigid 25 x 19 x 2 mm links", d["fidelity"])
        self.assertIn("2.5 N", d["fidelity"])

    def test_numbers_in_the_text_match_layout(self):
        self.assertEqual((L.BELOW_TABLE, L.DOG_NUDGED, L.DOG_MOVED, L.UNLATCH_OUT, L.APART_OUT, L.OFF_DOG, L.REST_HIGH,
                          L.STILL_SPEED, L.HANDS_OFF, L.WINDOW), (0.003, 0.020, 0.100, 0.003, 0.005, 0.025, 0.040, 0.005,
                                                                  0.030, 20))
        self.assertAlmostEqual(L.BX, 0.320)
        self.assertEqual(L.NOSE_PT[:2], (0.088, 0.008))
        self.assertAlmostEqual(L.RUMP_PT[0], 0.800)
        self.assertAlmostEqual(2 * L.HEAD_R, 0.132)
        self.assertAlmostEqual(2 * L.NECK_R, 0.104)
        self.assertEqual((L.NECK_X0, L.NECK_X1), (0.265, 0.380))
        self.assertAlmostEqual(P.collar_mass() * 1000, 33, delta=0.6)

    def test_mechkit_files_are_the_kits(self):
        for n in ("mechkit.py", "rigstate.py"):
            kit = KIT_DIR / n
            if kit.exists():
                self.assertEqual((HERE / n).read_bytes(), kit.read_bytes(), n)


if __name__ == "__main__":
    unittest.main()
