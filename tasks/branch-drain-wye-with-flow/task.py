"""On the station table: a plumber's bench mock-up of a horizontal drain (grey 40 mm square-section pipe, fixed)
running across the table along y, 38 cm in front of the arms, with black arrows painted on its top showing that
waste flows toward -y. A 134 mm section has been cut out of it around y 0 to take a horizontal branch drain from a
fixture on the front side (toward the arms). On the bench stand two loose white fittings, a wye (45 degree branch)
and a sanitary tee (90 degree branch with a swept inside corner), and a grey branch pipe lies on the left. Rule
(International Plumbing Code 2021, 706.3 and Table 706.3): a horizontal-to-horizontal junction takes a wye, never a
sanitary tee, installed so the branch guides the waste in the direction of flow: its branch on the fixture side,
leaning upstream. The branch pipe is then laid in line with the wye's branch, its end at the opening.

The fittings are loose mechkit compounds (mechkit.py, next to this file): each one rigid body of a 40 x 118 x 40 mm
run and its branch, so the scorer can read its pose (base frame and orientation). The scorer also reads the arms
(TCP, pad-tip point, opening) through rigstate.py, which wraps the embodiment's step and reset from this file's
import, only to check that each part was let go.

Run it:  inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
Demo:    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...
One arm: DEMO_IDLE_ARM=left (or right) before the demo command leaves that arm at its ready pose.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the fittings are built as compound bodies)
import rigstate  # noqa: E402
from scoring import Tracker, branch, centre  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Success (1.0) must hold 1 s. Every credit already needs the parts it reads released (both grippers' finger zones
# 15 mm or more from them over the last 0.5 s) and settled, and 1.0 needs all three still for 1 s, so the trial
# cannot end while a gripper holds a part.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "A horizontal drain crosses the bench; its painted arrows show which way waste flows. A section is cut out for "
    "a new horizontal branch drain from a fixture on the front side (your side). Of the two white fittings, fit the "
    "one the plumbing code allows here into the gap, in line with the drain and turned the way the code requires. "
    "Lay the grey branch pipe in line with the fitting's branch, its end at the opening. Leave the other fitting on "
    "the bench, off the drain. Let go of everything."
)

NAMES = {"wye": "wye", "tee": "sanitary_tee", "pipe": "branch_pipe"}


def _span_box(name, x0, x1, y0, y1, z0, z1, color, **kw) -> Box:
    """A box given by its extents (x, y in the world; z above the table top)."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + (z0 + z1) / 2), size=(x1 - x0, y1 - y0, z1 - z0),
               color=color, **kw)


X0, X1 = L.DRAIN_X - L.SEC / 2, L.DRAIN_X + L.SEC / 2
DRAIN = [
    _span_box("drain_downstream", X0, X1, -L.DRAIN_END, -L.GAP_HALF, 0.0, L.DRAIN_H, L.PVC_GREY),
    _span_box("drain_upstream", X0, X1, L.GAP_HALF, L.DRAIN_END, 0.0, L.DRAIN_H, L.PVC_GREY),
]
ARROWS = [
    Box(f"flow_arrow_{i + 1}_{j + 1}", pos=(L.DRAIN_X, y + dy, L.TOP + L.DRAIN_H + L.ARROW_T / 2),
        size=(sx, sy, L.ARROW_T), color=L.ARROW_BLACK)
    for i, y in enumerate(L.ARROW_Y) for j, (dy, sx, sy) in enumerate(L.ARROW_PARTS)
]


class Wye(mk.Assembly):
    """The wye exactly as mechkit builds and reads it in the simulator. Only the offline checker calls the two methods
    below: it draws and grips the wye as its run (40 x 118 x 40 mm), the part a gripper holds, instead of the box
    round the run and the branch, and the hooks get the wye's frame rebuilt from that run."""

    def offline_box(self):
        c = np.asarray(L.RUN_CENTRE, dtype=float)
        origin, quat = self.spawn_pose()
        centre_w = origin + mk.quat_to_matrix(quat) @ c
        return tuple(float(v) for v in centre_w), tuple(float(v) for v in L.RUN_SIZE), float(self.yaw)

    def offline_state(self, state):
        quat = np.asarray(state.quat_xyzw, dtype=float)
        origin = np.asarray(state.pos, dtype=float) - mk.quat_to_matrix(quat) @ np.asarray(L.RUN_CENTRE, dtype=float)
        vel = np.concatenate([np.asarray(state.lin_vel, dtype=float), np.asarray(state.ang_vel, dtype=float)])
        return mk.state_from_poses(self, (*origin, *quat), vel, source="offline")


WYE = Wye(name=NAMES["wye"], pos=(L.WYE_START[0], L.WYE_START[1], L.TOP), yaw=math.radians(L.WYE_START[2]),
          mass=L.FITTING_MASS, color=L.PVC_WHITE,
          shapes=(mk.box(L.RUN_CENTRE, L.RUN_SIZE),
                  *(mk.cylinder(c, L.BEAD_R, L.SEC, "Z") for c in L.WYE_BEADS)))
WYE._check()
TEE = mk.assembly(
    NAMES["tee"], origin=(L.TEE_START[0], L.TEE_START[1], L.TOP), yaw=math.radians(L.TEE_START[2]),
    mass=L.FITTING_MASS, color=L.PVC_WHITE,
    shapes=[mk.box(L.RUN_CENTRE, L.RUN_SIZE), mk.box(L.TEE_BRANCH_CENTRE, L.TEE_BRANCH_SIZE),
            *(mk.box(c, s) for c, s in L.TEE_SWEEP)],
)
PIPE = Box(NAMES["pipe"], pos=(L.PIPE_START[0], L.PIPE_START[1], L.TOP + L.PIPE_SIZE[2] / 2), size=L.PIPE_SIZE,
           mass=L.PIPE_MASS, yaw=math.radians(L.PIPE_START[2]), color=L.PVC_GREY)
OBJECTS = [*DRAIN, *ARROWS, WYE, TEE, PIPE]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and a teleport are two more
    signals)."""
    _TRACKER.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _parts(objects) -> dict:
    return {k: objects.get(v) for k, v in NAMES.items()}


def success(objects) -> bool:
    """The wye seated in the gap with its branch to the front, leaning upstream (135 +- 7 degrees), the branch pipe
    flat on the bench in line with the branch (fine), the tee resting on the bench away from the gap, all three let go
    and still for 1 s; held 1 s. This runs first every control step, so it also feeds the tracker (with the arms
    rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    parts = _parts(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(parts, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    try:
        e, u, _ = branch(parts["wye"])
        row.update(source=getattr(parts["wye"], "source", None),
                   wye_origin=parts["wye"].origin, wye_quat=parts["wye"].quat_xyzw,
                   tee_centre=centre(parts["tee"], "tee"), pipe_pos=parts["pipe"].pos,
                   pipe_quat=parts["pipe"].quat_xyzw, pipe_vel=parts["pipe"].lin_vel, wye_vel=parts["wye"].lin_vel,
                   demo_pipe_target=PIPE_SET[:2], demo_wye_target=WYE_SET[:2])
    except (AttributeError, TypeError, ValueError, KeyError):
        row["state_error"] = True
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp, left_tips=arms["left"].tips,
                   right_tips=arms["right"].tips, openings=[arms["left"].opening, arms["right"].opening])
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """First match wins, every part read only once released and settled: the sanitary tee seated in the gap 0; no
    wye seated 0; the wye seated with its branch not to the front and leaning upstream (135 +- 7 degrees) 0.25; the
    correct wye, the pipe on line (fine), the tee resting on the bench more than 60 mm from the gap and all three still
    for 1 s 1.0; the correct wye and the pipe on line (coarse) 0.75; the correct wye alone 0.5. Read from the state
    at hand (the last step)."""
    return _TRACKER.result(_parts(objects), rigstate.arms()).score


def distance(objects) -> float:
    """The wye's run centre to the gap's centre (m): a progress figure for the logs only."""
    try:
        c = centre(objects[NAMES["wye"]], "wye")
        return float(math.hypot(c[0] - L.GAP_CENTRE[0], c[1] - L.GAP_CENTRE[1]))
    except (AttributeError, TypeError, ValueError, KeyError):
        return 1.0


# -- the scripted demo ------------------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.028  # 56 mm: 8 mm either side of a 40 mm face
GRIP = -(L.SEC - 0.008) / 2  # 32 mm commanded across 40 mm
GRIP_UP = 0.024  # the TCP 24 mm above a part's underside: pad tips 11 mm above the table at the picks
DROP = 0.002  # each part is let go with its underside 2 mm above the table
APPROACH = 0.030  # each final descent starts this far above its end height
# Carried with the part's underside 66 mm up: 26 mm over the drain, the tee and the pipe. Not higher: over the gap
# the right arm's joint 4 is 2.5 degrees from its limit at a 12 cm TCP height, 12 degrees at 9 cm (yamkin).
HOVER = L.TOP + 0.09
UP = L.TOP + 0.09
# Yaw (radians) is the waypoint's: 0 closes the fingers along world y. A carried part turns with the gripper.
# In our GPU runs parts carried in a gripper ended turned counter-clockwise from the commanded yaw by about +0.6 to
# +1.7 degrees, so both parts are laid 1.15 degrees clockwise of nominal (YAW_TRIM): at most 0.6 degrees off then.
YAW_TRIM = math.radians(-1.15)
# Parts landed about 4 mm short of the target along the last horizontal move. Both parts make their last move the
# same way, 25 mm along (+x, -y) at a 45 degree heading (APPROACH_DIR = minus the correct branch axis), so a short
# landing moves the wye's opening and the pipe's end the same way along the branch axis and leaves the gap between
# them and the lateral offset almost unchanged. Each is aimed LEAD past its nominal point along that move (half the
# expected shortfall): offline (no shortfall) both land 2 mm long, on the GPU about 2 mm short.
APPROACH_DIR = np.array([L.SQ2, -L.SQ2])
FINAL_MOVE = 0.025
LEAD = 0.002
# The wye: gripped across its run 40 mm downstream of the run's centre (fitting point (0, -0.040)), clear of the
# branch's root (whose downstream edge meets the run's face 8 mm from the centre), picked at yaw 0 where it lies
# along x and laid at yaw +88.85 degrees: it turns +88.85 degrees from its start (-90) to -1.15 degrees. That pair
# keeps the right wrist within 55 degrees of the place's bearing from its base (yaw -91.15 would be 125 off, at
# joint 6's limit).
WYE_GRIP_LOCAL = np.array([0.0, -0.040])
WYE_YAW_PICK = 0.0
WYE_YAW_SET = math.radians(-L.WYE_START[2]) + YAW_TRIM  # +88.85 degrees: the wye from -90 to -1.15
WYE_LAID_YAW = math.radians(L.WYE_START[2]) + (WYE_YAW_SET - WYE_YAW_PICK)


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _wye_grip_world(centre_xy, yaw: float) -> np.ndarray:
    """The TCP (x, y) that holds the wye's grip point for a wye with this run centre and yaw."""
    return np.asarray(centre_xy, dtype=float) + _rot2(yaw) @ WYE_GRIP_LOCAL


WYE_START_GRIP = _wye_grip_world(L.WYE_START[:2], math.radians(L.WYE_START[2]))  # (0.240, -0.210)
WYE_AIM = np.array(L.GAP_CENTRE) + LEAD * APPROACH_DIR  # the run's centre: 2 mm past the gap's centre
WYE_SET = (*_wye_grip_world(WYE_AIM, WYE_LAID_YAW), L.TOP + DROP + GRIP_UP)
WYE_PRE = (*(np.asarray(WYE_SET[:2]) - FINAL_MOVE * APPROACH_DIR), HOVER)
# The pipe: gripped at its middle across its 40 mm section, picked at yaw 0 where it lies along x (pipe yaw 90) and
# laid at yaw -46.15 degrees (pipe yaw 43.85: its length along the branch axis, 133.85 degrees), within 10 degrees of
# the place's bearing from the left base. Its target is set from the nominal seated wye (run centre on the gap's
# centre, yaw 0): opening E = (0.3270, 0.0530), branch axis u at 135 degrees; the pipe's centre goes GAP_AIM + 70 mm
# out along u from E (its near end GAP_AIM from the opening) and LATERAL_AIM across it (along (+x, +y)). LATERAL_AIM
# centres the turn of both parts about their grip points: turned together (the GPU's yaw bias) the pipe's end moves
# across the axis relative to the opening by about 2.5 mm per degree. Modelled outcomes (tests/test_scoring.py,
# test_demo_outcome_offline_and_with_the_gpu_errors): offline (both parts 1.15 degrees clockwise of nominal, no
# shortfall) gap 5.5 mm, lateral -1.6 mm;
# on the GPU (each part turned +0.6..+1.7 degrees and 2-6 mm short, independently) gap 0.6-9.2 mm, lateral
# 0.2-3.7 mm, wye centre within 4.0 mm of the gap's centre: inside the fine window (gap -3..+10, lateral 6 mm).
GAP_AIM = 0.0055
LATERAL_AIM = 0.0014
PIPE_YAW_PICK = 0.0
PIPE_YAW_SET = math.radians(45.0 - L.PIPE_START[2]) + YAW_TRIM  # -46.15 degrees
E_NOMINAL = np.array(L.GAP_CENTRE) + np.array(L.WYE_TIP[:2])
U_NOMINAL = np.array(L.WYE_DIR)
N_NOMINAL = np.array([U_NOMINAL[1], -U_NOMINAL[0]])  # across the axis: (+0.7071, +0.7071)
PIPE_AIM = E_NOMINAL + (L.PIPE_HALF_LEN + GAP_AIM) * U_NOMINAL + LATERAL_AIM * N_NOMINAL + LEAD * APPROACH_DIR
PIPE_SET = (*PIPE_AIM, L.TOP + DROP + GRIP_UP)
PIPE_PRE = (*(PIPE_AIM - FINAL_MOVE * APPROACH_DIR), HOVER)
PIPE_PICK = (L.PIPE_START[0], L.PIPE_START[1], L.TOP + GRIP_UP)
WYE_PICK = (*WYE_START_GRIP, L.TOP + GRIP_UP)
RIGHT_PARK = (0.26, -0.26, L.TOP + 0.15)
LEFT_PARK = (0.25, 0.30, L.TOP + 0.15)


def _pick(p, yaw: float, what: str) -> list[Waypoint]:
    px, py, pz = p
    return [
        Waypoint(2.0, (px, py, HOVER), OPEN, f"over the {what}", yaw),
        Waypoint(1.0, (px, py, pz + APPROACH), OPEN, f"down to just above the {what}", yaw),
        Waypoint(1.5, (px, py, pz), OPEN, f"slowly down beside the {what}", yaw),
        Waypoint(0.5, (px, py, pz), OPEN, "settle", yaw),
        Waypoint(1.5, (px, py, pz), GRIP, f"grip the {what}", yaw),
        Waypoint(1.5, (px, py, HOVER), GRIP, f"lift the {what}", yaw),
    ]


def _place(pre, target, yaw: float, what: str, park) -> list[Waypoint]:
    tx, ty, tz = target
    return [
        Waypoint(2.5, pre, GRIP, f"carry the {what}, turning it", yaw),
        Waypoint(1.5, (tx, ty, HOVER), GRIP, f"last 25 mm toward the target, over it", yaw),
        Waypoint(0.8, (tx, ty, HOVER), GRIP, "settle over the target", yaw),
        Waypoint(1.0, (tx, ty, tz + APPROACH), GRIP, "down to 3 cm above", yaw),
        Waypoint(2.0, (tx, ty, tz), GRIP, f"slowly down, the {what} 2 mm above the table", yaw),
        Waypoint(0.5, (tx, ty, tz), GRIP, "settle", yaw),
        Waypoint(1.5, (tx, ty, tz), OPEN, f"let go of the {what}", yaw),
        Waypoint(1.0, (tx, ty, UP), OPEN, "straight up", yaw),
        Waypoint(1.5, park, OPEN, "aside", yaw),
    ]


def _right() -> list[Waypoint]:
    """Right arm: the wye from its start into the gap, branch to the front and leaning upstream (20.3 s)."""
    return _pick(WYE_PICK, WYE_YAW_PICK, "wye") + _place(WYE_PRE, WYE_SET, WYE_YAW_SET, "wye", RIGHT_PARK)


def _right_released_s() -> float:
    """When the right arm has let go of the wye and lifted clear (the end of 'straight up')."""
    t = 0.0
    for w in _right():
        t += w.duration
        if w.label == "straight up":
            return t
    raise ValueError("no 'straight up' waypoint")


def _left() -> list[Waypoint]:
    """Left arm: picks the pipe at once, waits holding it until the wye is down and the right arm has lifted clear,
    then lays it along the wye's branch (31.1 s)."""
    pick = _pick(PIPE_PICK, PIPE_YAW_PICK, "branch pipe")
    waited = _right_released_s() - sum(w.duration for w in pick)
    px, py, _ = PIPE_PICK
    wait = [Waypoint(waited, (px, py, HOVER), GRIP, "hold it up until the wye is down", PIPE_YAW_PICK)]
    return pick + wait + _place(PIPE_PRE, PIPE_SET, PIPE_YAW_SET, "branch pipe", LEFT_PARK)


def demo():
    """Right arm: the wye; left arm: the branch pipe once the wye is down. DEMO_HOLD_ONLY=1: no plan (both arms hold
    the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"right": _right(), "left": _left()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
