"""The scripted demo against the die's and the block's real shapes (offline, no simulator).

tools/offline/check_task.py sees each mechkit assembly only as the box round it at rest (the die's 170 x 220 x 110 mm
box holds the part's pocket, so it reports the right gripper 'inside' the die whenever it takes the part). This test
replays the demo the same way (the stub's ArmPlan, its IK, 20 Hz) with the stand-in moving the platen and the block,
and measures, every control step, each arm's link capsules and finger pads against every real box: the die's top
layer and back platen, the moving platen's five parts at the current opening, the bed's slab, the block's bar and
handle arm, the part, the stand and the tray. What a gripper holds (the post, the block, the part) is left out for its
own pads. Needs tools/offline (yamkin, scenegeom, the yam_arena stub).  python3 -m unittest discover -s tests -v"""

import importlib
import math
import pathlib
import sys
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


def _offline_dir():
    for p in HERE.parents:
        d = p / "tools" / "offline"
        if (d / "yam_stub" / "yam_arena").is_dir() and (d / "yamkin.py").is_file():
            return d
    return None


PAD_THICK, PAD_HALF_WIDTH, PAD_SPAN = 0.008, 0.010, (-0.025, 0.013)  # as tools/offline/check_task.py models pads
LINK_MIN = 0.0  # m: no link capsule may touch anything
PAD_MIN = -0.001  # m: a pad may graze (contact offset), never cut into anything it does not hold


@unittest.skipIf(_offline_dir() is None, "needs tools/offline (yamkin, scenegeom, yam_stub)")
class DemoClearance(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        off = _offline_dir()
        sys.path.insert(0, str(off / "yam_stub"))
        sys.path.insert(0, str(off))
        cls.K = importlib.import_module("yamkin")
        cls.G = importlib.import_module("scenegeom")
        cls.task = importlib.import_module("task")
        cls.L = importlib.import_module("layout")
        from yam_arena.demo import ArmPlan
        from standin import StandIn

        cls.report = cls._replay(ArmPlan, StandIn)

    @classmethod
    def _boxes(cls, j, block_origin, part_centre):
        """Every real box as (group, scenegeom Solid)."""
        K, G, L, task = cls.K, cls.G, cls.L, cls.task
        out = []

        def add(group, centre, size):
            out.append((group, G.Solid(group, "box", tuple(map(float, centre)), tuple(float(s) / 2 for s in size))))

        o = np.array(L.DIE_ORIGIN)
        for s in task.DIE.shapes:
            add("die", o + s.center, s.size)
        for i, s in enumerate(task.DIE.link("platen").shapes):
            add("post" if i == 4 else "platen", o + s.center - [j, 0, 0], s.size)
        for obj in task.OBJECTS:
            if obj.name in ("die", "block", "part"):
                continue
            add("fixtures", obj.pos, obj.size)
        add("block", np.asarray(block_origin) + L.BAR_CENTER, L.BAR_SIZE)
        add("block", np.asarray(block_origin) + L.ARM_CENTER, L.ARM_SIZE)
        add("part", part_centre, L.PART_SIZE)
        return out

    @classmethod
    def _replay(cls, ArmPlan, StandIn):
        K, G, L, task = cls.K, cls.G, cls.L, cls.task
        import rigstate

        plans = task.demo()
        arms = {a: ArmPlan(plans[a], K.BASES["station"][a]) for a in ("left", "right")}
        end = max(p.duration for p in arms.values())
        hz = 20.0
        standin = StandIn(task.DIE, task.BLOCK, hz)
        part = np.array([L.PART_X, L.PART_Y, L.TOP + L.PART_Z])
        part_held = None  # offset part centre - right TCP while held
        worst = {}  # (arm, kind, group) -> (gap, t, label)
        held_post_open = []
        for k in range(int(math.ceil(end * hz)) + 1):
            t = (k + 1) / hz
            poses, fingers, labels, qs = {}, {}, {}, {}
            for a, plan in arms.items():
                act, label = plan.action7(t)
                q = act[:6]
                fingers[a] = float(act[6]) * K.FINGER_OPEN
                poses[a] = K.tcp_pose(q, K.BASES["station"][a])
                labels[a], qs[a] = label, q
            feed = {a: {"tcp": poses[a][0], "tips": K.pad_tip(*poses[a]),
                        "opening": float(np.clip(fingers[a] / K.FINGER_OPEN, 0, 1))} for a in arms}
            rigstate.offline_feed(feed)
            die_s, block_s = standin.step(rigstate.arms())
            j = float(die_s.joints["platen"])
            r_open = feed["right"]["opening"]
            r_tcp = poses["right"][0]
            if part_held is None and r_open < L.STANDIN_GRIP and np.linalg.norm(r_tcp - (part + [0, 0, 0.012])) < 0.006:
                part_held = part - r_tcp
            elif part_held is not None and r_open >= L.STANDIN_GRIP:
                part_held = None
                part[2] = L.TOP + L.TRAY_FLOOR_Z + L.PART_SIZE[2] / 2 if abs(part[0] - L.TRAY_X) < 0.05 else part[2]
            if part_held is not None:
                part = r_tcp + part_held
            boxes = cls._boxes(j, block_s.origin, part)
            for a in arms:
                p, r = poses[a]
                mine = set()
                if standin.post_holder == a:
                    mine.add("post")
                    held_post_open.append(j)
                if standin.block_holder == a:
                    mine.add("block")
                if a == "right" and part_held is not None:
                    mine.add("part")
                caps = [c for c in K.arm().capsules(qs[a], K.BASES["station"][a], fingers[a])
                        if c[0] not in K.FINGER_LINKS]
                for group, box in boxes:
                    g = min(box.point_sd(pt) - rr for _, c0, c1, rr in caps for pt in G.segment_points(c0, c1, 8))
                    key = (a, "link", group)
                    if key not in worst or g < worst[key][0]:
                        worst[key] = (g, round(t, 2), labels[a])
                gap_eff = K.finger_gap(fingers[a])
                held_width = {"post": L.POST_SIZE[0], "block": L.ARM_SIZE[0], "part": L.PART_SIZE[1]}
                for m in mine:
                    gap_eff = max(gap_eff, held_width[m])
                mid = (PAD_SPAN[0] + PAD_SPAN[1]) / 2
                half = (PAD_THICK / 2, PAD_HALF_WIDTH, (PAD_SPAN[1] - PAD_SPAN[0]) / 2)
                yaw = math.atan2(r[1, 0], r[0, 0])
                for sign in (-1.0, 1.0):
                    c = p + r[:, 0] * sign * (gap_eff / 2 + PAD_THICK / 2) + r[:, 2] * mid
                    pad = G.Solid("pad", "box", tuple(c), half, yaw)
                    for group, box in boxes:
                        if group in mine:
                            continue
                        g = G.gap(pad, box)
                        key = (a, "pad", group)
                        if key not in worst or g < worst[key][0]:
                            worst[key] = (g, round(t, 2), labels[a])
        return {"worst": worst, "held_post_open": held_post_open, "final_part": part.copy(),
                "final_block": np.array(block_s.origin), "final_j": j}

    def test_links_touch_nothing(self):
        bad = {k: v for k, v in self.report["worst"].items() if k[1] == "link" and v[0] < LINK_MIN}
        self.assertEqual(bad, {}, bad)

    def test_pads_cut_into_nothing_they_do_not_hold(self):
        bad = {k: v for k, v in self.report["worst"].items() if k[1] == "pad" and v[0] < PAD_MIN}
        self.assertEqual(bad, {}, bad)

    def test_margins_worth_knowing(self):
        """The tightest real clearances (printed for the GPU check): the right gripper's housing capsule over the
        fixed platen's top edge as it reaches the part, the right pads over the stand as the block is set back, the
        left pads beside the post and over the moving platen."""
        w = self.report["worst"]
        for key in sorted(w, key=lambda k: w[k][0])[:8]:
            print(f"  {key}: {w[key][0] * 1000:.1f} mm at t={w[key][1]} s '{w[key][2]}'")
        self.assertGreater(w[("right", "pad", "die")][0], 0.003)
        self.assertGreater(w[("left", "pad", "platen")][0], 0.003)

    def test_the_stand_in_ends_where_the_task_wants(self):
        L = self.L
        np.testing.assert_allclose(self.report["final_block"], L.BLOCK_ORIGIN, atol=0.002)  # re-gripped 1 mm off
        self.assertLess(self.report["final_j"], 1e-3)
        self.assertLess(abs(self.report["final_part"][0] - L.TRAY_X), 0.002)
        self.assertGreaterEqual(max(self.report["held_post_open"]), 0.072)


if __name__ == "__main__":
    unittest.main()
