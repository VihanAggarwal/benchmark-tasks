"""On the station table, 30 cm in front of the arm bases, a meter setting is laid out along y: a fixed copper elbow
(40 mm block, with a white mark on top and a short copper supply pipe going into it from the far side) at the left
(+y) end, a fixed brass valve (40 mm block with a red lever along the pipe) at the right (-y) end, and between their
facing sides, 580 mm apart, a channel formed by two low grey aluminium rails 37 mm apart (6 x 8 mm, fixed). Loose
parts lie flat along x: on the right of the table a 250 mm and a 125 mm length of 25 mm square copper-painted pipe;
on the left a 75 mm length and the water meter, a 100 mm brass body (25 mm square) with a blue register on top
carrying a white flow arrow, which points away from the arms. Water flows from the elbow to the valve. Rule: the
meter and the three pipes go into the channel end to end from the elbow with every joint closed (the valve's union
takes up the 30 mm left at its end), the meter's arrow toward the valve with its register up, and at least 10 pipe
diameters (250 mm) of straight pipe between the elbow and the meter and 5 (125 mm) between the meter and the valve.

The meter is a loose mechkit compound (mechkit.py, next to this file): one rigid body made of its body, register and
arrow strips, so the arrow is visible and the scorer can read the meter's frame. The scorer reads both grippers'
finger tips through rigstate.py, which wraps the embodiment's step and reset from this file's import; neither
changes anything the embodiment does.

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

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import layout as L  # noqa: E402
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the meter is built as one compound body)
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with both grippers let go: each one's finger line (its pad tips to 38 mm back up the gripper) 65 mm or
# more from every part is one of the 1.0 conditions, so the hold can only start once the last gripper has opened and
# moved clear of the run.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Lay the water meter and the three pipe lengths end to end in the channel from the elbow to the valve, the first "
    "one against the elbow and every joint closed; the valve's union takes up the gap left at its end. Water flows "
    "from the elbow to the valve: point the meter's arrow toward the valve with its blue register on top, and leave "
    "at least 10 pipe diameters (250 mm) of straight pipe between the elbow and the meter and 5 (125 mm) between "
    "the meter and the valve. Let go and move clear."
)


def _fixed(name: str, centre, size, color) -> Box:
    """A fixed box; ``centre`` is (x, y, height of its centre above the table top)."""
    return Box(name, pos=(centre[0], centre[1], L.TOP + centre[2]), size=size, color=color)


FIXED = [
    _fixed("elbow", L.ELBOW_C, L.ELBOW_SIZE, L.COPPER),
    _fixed("inlet_mark", L.INLET_TILE_C, L.INLET_TILE_SIZE, L.WHITE),
    _fixed("supply_pipe", L.SUPPLY_C, L.SUPPLY_SIZE, L.COPPER),
    _fixed("valve", L.VALVE_C, L.VALVE_SIZE, L.BRASS),
    _fixed("valve_lever", L.LEVER_C, L.LEVER_SIZE, L.LEVER_RED),
    _fixed("rail_near", L.RAIL_NEAR_C, L.RAIL_SIZE, L.RAIL_GREY),
    _fixed("rail_far", L.RAIL_FAR_C, L.RAIL_SIZE, L.RAIL_GREY),
]
PIPES = [
    Box(name, pos=(x, y, L.TOP + L.HALF_W), size=(length, L.PIPE_W, L.PIPE_W), mass=mass, color=L.COPPER)
    for name, (x, y, length, mass) in L.PIPE_START.items()
]
METER = mk.assembly(
    "meter",
    origin=(L.METER_START[0], L.METER_START[1], L.TOP + L.HALF_W),
    mass=L.METER_MASS,
    color=L.BRASS,
    shapes=[
        mk.box(*L.BODY),
        mk.box(*L.REGISTER, color=L.REGISTER_BLUE),
        *(mk.box(c, s, color=L.WHITE) for c, s in L.ARROW),
    ],
)
OBJECTS = [*FIXED, *PIPES, METER]

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
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
    return {n: objects.get(n) for n in L.PARTS}


def success(objects) -> bool:
    """All four parts laid in the channel, butted from the elbow (every joint 5 mm or less), the meter's arrow toward
    the valve with its register up, at least 245 mm of straight pipe before the meter and 120 mm after it, both
    grippers read with their finger lines (pad tips to 38 mm back up) 65 mm or more from every part, and no part's
    end moving more than 2 mm in 0.5 s; held 1 s.
    This runs first every control step, so it also feeds the tracker (with the arms rigstate read for this step)."""
    arms = rigstate.arms()
    result = _TRACKER.update(_parts(objects), arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    try:
        m = objects["meter"]
        row.update(meter_source=getattr(m, "source", None), meter_origin=getattr(m, "origin", None),
                   meter_quat=m.quat_xyzw, **{f"{n}_pos": objects[n].pos for n in L.PIPES})
    except (AttributeError, KeyError, TypeError):
        row["state_error"] = True
    try:
        row.update(left_tips=arms["left"].tips, right_tips=arms["right"].tips,
                   opening={s: round(float(arms[s].opening), 3) for s in ("left", "right")})
    except (AttributeError, KeyError, TypeError, ValueError):
        row["rigstate_error"] = rigstate.error() or "arms not readable"
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 unless both grippers have let go (each one read and its finger line 65 mm or more from every part; a gripper
    that cannot be read has not) and all four parts lie in the
    channel (flat, register up, along it within 5 degrees, centred within 8 mm, on the table within 3 mm) butted
    from the elbow (every joint between -2 and +5 mm; the gap at the valve is not checked). Then 0.25 with the
    meter's arrow toward the elbow; 1.0 with it toward the valve and at least 245 mm of straight pipe from the
    elbow to the meter and 120 mm from the meter to the valve; 0.5 otherwise. Read from the state at hand."""
    return _TRACKER.result(_parts(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Order from the elbow: pipe_75 (left arm), pipe_250 (right), meter (left), pipe_125 (right): 75 mm + 250 mm = 325 mm
# of straight pipe before the meter and 125 mm + the 30 mm union gap = 155 mm after it. Every part is picked lying
# along x with the fingers closing along y (yaw 0), turned a quarter turn in the hand (left -pi/2, right +pi/2: the
# signs that reach the midline), set down 12 mm short of its upstream neighbour with its bottom 2 mm above the table,
# slid toward the elbow while still gripped to 10 mm past contact (pressing the parts already laid against the fixed
# elbow), eased back to 1.5 mm past contact, let go and left straight up.
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
GRIP = -(L.PIPE_W - 0.008) / 2  # 17 mm commanded on the 25 mm parts: about 20 N per finger
YAW = {"left": -math.pi / 2, "right": math.pi / 2}
PICK_Z = L.TOP + 0.025  # pad tips 12 mm above the table: the top 13 mm of a part's 25 mm side held
SET_Z = L.TOP + 0.027  # a held part's bottom 2 mm above the table; pad tips 6 mm above the rails' tops
ENTRY_Z = L.TOP + 0.055  # the last 28 mm are lowered slowly
HOVER_Z = L.TOP + 0.10  # over a part before picking it; the carry height near the midline
LIFT_Z = L.TOP + 0.12  # parts are turned and wait up here (a held part's bottom 95 mm up, clear of the valve)
UP_Z = L.TOP + 0.075  # after letting go: pad tips 37 mm above a pipe's top, 22 mm above the meter's outline top
#   (the release rule's 65 mm is met only on the way to the park pose: 123 mm left, 138 mm right)
PARK_Z = L.TOP + 0.14
GAP0 = 0.012  # set down this far short of contact (covers a few mm of grip offset along the run)
OVERSHOOT = 0.010  # slid this far past contact while gripped (the arm trails 3-7 mm under the load; tune 6-14 mm)
RELAX = 0.0015  # eased back to this far past contact before opening, so the arm's spring does not kick the part
# Per-arm (x, y) trim of every set-down point, for the GPU: read <part>'s centre_off_mm and joints_mm in TASK_TRACE.
PLACE_TRIM = {"left": (0.0, 0.0), "right": (0.0, 0.0)}

# Where each part's grip lands along the run when it is butted (contact), from the order above: the 75 against the
# elbow face (y 0.290), centred at 0.2525; the 250 from 0.215 to -0.035, gripped 30 mm from its centre toward the
# arms, which the right arm's quarter turn puts 30 mm on its -y side (y 0.060); the meter from -0.035 to -0.135,
# gripped 25 mm from its centre toward its inlet (+y) end (y -0.060); the 125 from -0.135 to -0.260, centred at
# -0.1975.
GRIP_250_OFF = 0.030  # the 250 is picked this far toward the arms from its centre
CONTACT = {
    "pipe_75": L.ELBOW_FACE_Y - L.LENGTH["pipe_75"] / 2,
    "pipe_250": L.ELBOW_FACE_Y - L.LENGTH["pipe_75"] - L.LENGTH["pipe_250"] / 2 - GRIP_250_OFF,
    "meter": L.ELBOW_FACE_Y - L.LENGTH["pipe_75"] - L.LENGTH["pipe_250"] - (L.METER_LEN / 2 + L.METER_GRIP_LOCAL),
    "pipe_125": L.ELBOW_FACE_Y - L.LENGTH["pipe_75"] - L.LENGTH["pipe_250"] - L.METER_LEN - L.LENGTH["pipe_125"] / 2,
}


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _end_of(wps, label: str) -> float:
    """When the first waypoint whose label starts with ``label`` ends."""
    t = 0.0
    for w in wps:
        t += w.duration
        if w.label.startswith(label):
            return t
    raise KeyError(label)


def _wait(wps, until: float, label: str) -> None:
    """Hold the last pose until ``until`` (s on the shared clock)."""
    t = _span(wps)
    if until > t + 1e-9:
        last = wps[-1]
        wps.append(Waypoint(until - t, last.pos, last.finger, label, last.yaw))


def _pick(part: str, x: float, y: float, yaw_after: float, inboard=None, travel: float = 2.0) -> list[Waypoint]:
    """Down onto a part lying along x, grip it across its 25 mm (yaw 0), lift it, optionally move it inboard, and
    turn it a quarter turn so it lies along y."""
    wps = [
        Waypoint(travel, (x, y, HOVER_Z), OPEN, f"over {part}", 0.0),
        Waypoint(1.0, (x, y, PICK_Z + 0.03), OPEN, f"down to {part}", 0.0),
        Waypoint(1.5, (x, y, PICK_Z), OPEN, "slowly down", 0.0),
        Waypoint(0.5, (x, y, PICK_Z), OPEN, "settle", 0.0),
        Waypoint(1.5, (x, y, PICK_Z), GRIP, f"grip {part}", 0.0),
        Waypoint(1.0, (x, y, LIFT_Z), GRIP, "lift", 0.0),
    ]
    if inboard is not None:
        x, y = inboard
        wps.append(Waypoint(1.5, (x, y, LIFT_Z), GRIP, "move inboard to turn", 0.0))
    wps.append(Waypoint(2.0, (x, y, LIFT_Z), GRIP, f"turn {part} along the run", yaw_after))
    return wps


def _place(arm: str, part: str, via_y: float | None = None, carry: float = 1.8) -> list[Waypoint]:
    """Carry the part over the channel, lower it 12 mm short of contact, slide it toward the elbow to 10 mm past
    contact, ease back to 1.5 mm past, let go, go straight up. ``via_y``: carry to there first and come down
    diagonally (the reach near the midline is lower than at the table)."""
    yaw = YAW[arm]
    tx, ty = PLACE_TRIM[arm]
    x, c = L.RUN_X + tx, CONTACT[part] + ty
    set_y = c - GAP0
    return [
        Waypoint(carry, (x, set_y if via_y is None else via_y + ty, HOVER_Z), GRIP, f"carry {part}", yaw),
        Waypoint(1.0, (x, set_y, ENTRY_Z), GRIP, "down" if via_y is None else "down diagonally", yaw),
        Waypoint(1.5, (x, set_y, SET_Z), GRIP, "slowly down into the channel", yaw),
        Waypoint(0.5, (x, set_y, SET_Z), GRIP, "settle", yaw),
        Waypoint(1.5, (x, c + OVERSHOOT, SET_Z), GRIP, f"slide {part} to butt", yaw),
        Waypoint(0.5, (x, c + OVERSHOOT, SET_Z), GRIP, "press", yaw),
        Waypoint(0.5, (x, c + RELAX, SET_Z), GRIP, "ease off", yaw),
        Waypoint(1.5, (x, c + RELAX, SET_Z), OPEN, f"let go of {part}", yaw),
        Waypoint(1.0, (x, c + RELAX, UP_Z), OPEN, "straight up", yaw),
    ]


# Where the short pipes and the meter are turned: joint 6 keeps 12.7 degrees from its limit there (it would have none
# where the short pipes lie, at x 0.15, y +-0.42; 5.9 at the meter's pick point). The 250 is turned where it is picked
# (12.4 degrees).
TURN_AT = {"left": (0.24, 0.38), "right": (0.24, -0.38)}


def plans() -> dict:
    """Both arms on one clock, the waits computed from the other arm's plan."""
    x75, y75 = L.PIPE_START["pipe_75"][:2]
    left = _pick("pipe_75", x75, y75, YAW["left"], inboard=TURN_AT["left"])  # a turn where it lies pins joint 6
    left += _place("left", "pipe_75")
    t_75_open = _end_of(left, "let go of pipe_75")  # the left fingers are open beside the 75; it goes up now

    x250, y250 = L.PIPE_START["pipe_250"][:2]
    right = _pick("pipe_250", x250 - GRIP_250_OFF, y250, YAW["right"])  # 30 mm toward the arms from its centre
    _wait(right, t_75_open, "wait for the 75")
    right += _place("right", "pipe_250", via_y=0.02, carry=2.0)
    t_250_clear = _span(right)  # the right gripper is up off the 250

    # The meter is picked 25 mm behind its centre and turned clockwise seen from above (yaw 0 -> -pi/2): its arrow
    # goes from +x to -y (toward the valve) and its gripped inlet half leads (+y, toward the elbow).
    left += _pick("meter", L.METER_START[0] + L.METER_GRIP_LOCAL, L.METER_START[1], YAW["left"],
                  inboard=TURN_AT["left"])
    _wait(left, t_250_clear, "wait for the 250")
    left += _place("left", "meter", via_y=-0.03, carry=2.2)
    t_meter_up = _span(left)  # the left gripper is up off the meter
    left.append(Waypoint(2.0, (0.24, 0.34, PARK_Z), OPEN, "park", YAW["left"]))

    x125, y125 = L.PIPE_START["pipe_125"][:2]
    right += _pick("pipe_125", x125, y125, YAW["right"], inboard=TURN_AT["right"])
    _wait(right, t_meter_up, "wait for the meter")
    right += _place("right", "pipe_125")
    right.append(Waypoint(2.0, (0.24, -0.34, PARK_Z), OPEN, "park", YAW["right"]))
    return {"left": left, "right": right}


def demo():
    """Both arms. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right:
    that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = plans()
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
