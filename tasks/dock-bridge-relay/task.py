"""On the station table: a fixed grey loading dock (180 mm along x, 260 mm deep, 40 mm high) on the right arm's side
and a fixed brown truck bed (39 mm high) on the left arm's side, facing each other across an 80 mm gap on the
midline (y -40 .. +40 mm), 23 to 41 cm in front of the arm bases. Each has two seats (rebates 50 mm wide, 45 mm deep,
20.5 mm below its top) at its gap edge, one pair at x 26-31 cm and one at x 34-39 cm. Two loose aluminium runner
planks (40 x 160 x 20 mm, 60 g) lie on the table left of the truck bed; a loose blue cart (a 100 x 100 x 30 mm
block, 0.21 kg, too wide to grip) stands on the dock; two yellow marks on the bed show where it goes. Rule: lay both
runners into their seats across the gap first, then push the cart over them onto the bed, never lifting it.

Only the left arm reaches the runners and only the right arm reaches the cart at the start; the right arm can push
the cart only part of the way onto the bed, so the left arm takes the push over once it is past the gap.

The scorer reads the arms (TCP, finger tips, gripper opening) through rigstate.py, which wraps the embodiment's
step and reset from this file's import; it changes nothing the embodiment does. offline_push.py is used only by our
offline checker (no simulator), which cannot push; the simulator never calls it.

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
import offline_push  # noqa: E402
import rigstate  # noqa: E402
from scoring import Tracker  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Still for 1 s with the cart on its mark, both runners seated and both grippers' finger zones 30 mm clear.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "Lay the two grey runner planks across the gap between the loading dock and the truck bed, each into its pair "
    "of seats on the dock edge and the truck-bed edge. Then push the blue cart from the dock over the runners onto "
    "the truck bed, between the two yellow marks. Never lift the cart: slide it along the surfaces. Leave both "
    "runners in their seats and let go of everything."
)


def _block(name: str, x0: float, x1: float, y0: float, y1: float, top: float, color) -> Box:
    """A fixed box standing on the table, spanning x0..x1, y0..y1, from the table up to ``top``."""
    return Box(name, pos=((x0 + x1) / 2, (y0 + y1) / 2, L.TOP + top / 2), size=(x1 - x0, y1 - y0, top), color=color)


def _side(prefix: str, sign: float, top: float, y_end: float, color) -> list[Box]:
    """The dock (sign -1) or the truck bed (+1): a main block, two full-height cheeks outside the seats, two seat
    floors and the low rib between them. No seam runs across the cart's path at the cart's level."""
    g, s = sign * L.GAP_Y, sign * L.SEAT_Y
    lo, hi = sorted((g, s))
    m0, m1 = sorted((s, y_end))
    (a0, a1), (b0, b1) = L.SEATS["a"], L.SEATS["b"]
    return [
        _block(f"{prefix}_main", L.X0, L.X1, m0, m1, top, color),
        _block(f"{prefix}_cheek_near", *L.CHEEKS[0], lo, hi, top, color),
        _block(f"{prefix}_cheek_far", *L.CHEEKS[1], lo, hi, top, color),
        _block(f"{prefix}_seat_a", a0, a1, lo, hi, L.SEAT_FLOOR, color),
        _block(f"{prefix}_seat_b", b0, b1, lo, hi, L.SEAT_FLOOR, color),
        _block(f"{prefix}_rib", *L.RIB_X, lo, hi, L.RIB_TOP, color),
    ]


DOCK = _side("dock", -1.0, L.DOCK_TOP, L.DOCK_Y_END, L.CONCRETE)
BED = _side("bed", 1.0, L.BED_TOP, L.BED_Y_END, L.DECK_BROWN)
MARKS = [Box(name, pos=(x, y, L.TOP + L.BED_TOP + L.MARK_SIZE[2] / 2), size=L.MARK_SIZE, color=L.YELLOW)
         for name, (x, y) in L.MARKS.items()]
RUNNER_A, RUNNER_B = (
    Box(name, pos=(x, y, L.TOP + L.RUNNER_HALF[2]), size=L.RUNNER_SIZE, mass=L.RUNNER_MASS, color=L.ALUMINIUM)
    for name, (x, y) in L.RUNNERS.items()
)
CART = Box("cart", pos=(L.CART_X, L.CART_Y, L.TOP + L.DOCK_TOP + L.CART_HALF[2]), size=L.CART_SIZE,
           mass=L.CART_MASS, color=L.CART_BLUE)
OBJECTS = [*DOCK, *BED, *MARKS, RUNNER_A, RUNNER_B, CART]  # the runners before the cart (offline_push below)

# Offline checker only (tools/offline/check_task.py): it carries the runners but cannot push the cart, so the
# cart's offline state comes from a small push model. The simulator never calls these hooks.
_PUSH = offline_push.CartPush((RUNNER_A.name, RUNNER_B.name))
_PUSH.attach(CART, (RUNNER_A, RUNNER_B))

_TRACKER = Tracker()


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (the scorer also spots a teleport back to the start)."""
    _TRACKER.reset()
    _PUSH.reset()


def _trace(row: dict) -> None:
    """With TASK_TRACE=<file>, append one JSON line per control step: what the scorer saw and decided."""
    path = os.environ.get("TASK_TRACE")
    if path:
        try:
            with open(path, "a") as f:
                f.write(json.dumps(row, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")
        except OSError:  # a debugging aid only: never let it stop a trial
            pass


def _three(objects) -> tuple:
    return objects.get("cart"), objects.get("runner_a"), objects.get("runner_b")


def success(objects) -> bool:
    """Cart on the truck bed, its centre within 15 mm of its mark and square within 5 degrees, both runners seated,
    both grippers' finger zones 30 mm or more from the cart and the runners, no rule ever broken; held still 1 s.
    This runs first every control step, so it also feeds the tracker (with the arms rigstate read for this
    step)."""
    cart, runner_a, runner_b = _three(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(cart, runner_a, runner_b, arms)
    row = {"episode": rigstate.episode(), "step": _TRACKER.steps, "score": result.score, "success": result.success,
           **result.details}
    for name, s in (("cart", cart), ("runner_a", runner_a), ("runner_b", runner_b)):
        if s is not None:
            row[f"{name}_pos"], row[f"{name}_quat"] = getattr(s, "pos", None), getattr(s, "quat_xyzw", None)
    if arms is not None:
        row.update(left_tcp=arms["left"].tcp, right_tcp=arms["right"].tcp,
                   left_tips=arms["left"].tips, right_tips=arms["right"].tips)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if, at any step, part of the cart came over the gap before both runners were seated, the cart dropped below
    35 mm or rose above 45 mm, tilted over 10 degrees, a runner moved over 10 mm while the cart was on it, or
    anything left the table. Else 1.0 cart on the bed within 15 mm of its
    mark and square within 5 degrees, both runners seated, both grippers 30 mm clear; 0.75 cart on the bed within
    40 mm, both runners seated; 0.5 both runners seated and the cart's centre past y +40 mm; 0.25 both runners
    seated; 0 otherwise."""
    return _TRACKER.result(*_three(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
OPEN = -0.04695
CLOSED = 0.0  # pushing: the fingers closed together, a block 16 mm deep along y and 27 mm wide along x
RUNNER_GRIP = -(L.RUNNER_SIZE[0] - 0.008) / 2  # across the runner's 40 mm width: 32 mm commanded
RUNNER_RELEASE = -(L.RUNNER_SIZE[0] + 0.024) / 2  # 64 mm: let go without the pads reaching the next seat's runner
YAW_A = -math.pi / 2  # the left arm grips each runner across its width, the fingers closing along x
YAW_B_PICK = math.pi / 2  # runner B lies 15 cm outboard of the left base, where only +90 deg reaches; it is turned
#                           to -90 deg (the same grip of a symmetric plank) in two quarter turns while carried
YAW_PUSH = 0.0  # both arms push with the fingers closing along y: the leading finger's outer face on the cart
GRIP_UP = 0.017  # the TCP this far above a runner's bottom: pad tips 4 mm above it, pads on 16 of its 20 mm
RUNNER_PICK_Z = L.TOP + GRIP_UP
RUNNER_CARRY_Z = L.TOP + 0.075  # the runner's bottom 58 mm up: 18 mm over the dock and the bed
RUNNER_ENTRY_Z = L.TOP + L.SEAT_FLOOR + 0.026 + GRIP_UP  # its bottom 5.5 mm over the dock top, over the seats
RUNNER_PLACE_Z = L.TOP + L.SEAT_FLOOR + 0.001 + GRIP_UP  # let go 1 mm above the seat floors
HOVER_Z = L.TOP + 0.100
PUSH_Z = L.TOP + 0.060  # pad tips 47 mm up: 7 mm over the dock and 8 over the bed; on the cart's rear face 7-30 mm up
FINGER_FRONT = L.PAD_THICK  # the closed fingers' outer face is this far ahead of the TCP along y (where they push)
# The arms trail a push. Without payload compensation each joint is a spring (80/80/80/10/10/10 N m/rad); the
# compliance model (J K^-1 J^T at the fingers, at these poses) puts a pushing gripper 9-13 mm behind its command
# under the cart's sliding friction (2.06 N; 2.5 N to start it), and pitches it about 2.5 deg with the tips trailing,
# so the fingers bear on the cart's top rear edge: 30 mm up, where even the breakaway force (friction 1.2) cannot tip
# a 100 mm long block (it would need 42 mm). Each push is commanded past its goal by the modelled trail at its end;
# the cart's final place is set by the left arm's push alone and lands within 15 mm of the mark for any trail from
# 0 to 25 mm. On the GPU read cart_rear_y_mm against the TCPs (rear_minus_tcp_mm: 8 mm while in contact) in
# TASK_TRACE and adjust TRAIL_R / TRAIL_L.
TRAIL_R = 0.0125  # right arm, at the end of its push (y +0.005): 12.6 mm modelled
TRAIL_L = 0.010  # left arm, at the end of its push (y +0.16): 10.3 mm modelled
CART_REAR0 = L.CART_Y - L.CART_HALF[1]  # -0.250
R_START_Y = CART_REAR0 - FINGER_FRONT - 0.004  # the right fingers come down 4 mm behind the cart
HANDOVER_REAR = 0.000  # the right arm pushes the cart's rear face to y 0 (its centre to +0.050: over the bed side)
R_END_Y = HANDOVER_REAR - FINGER_FRONT + TRAIL_R  # +0.0045: the right arm reaches y +0.075 here
L_START_Y = -0.040  # the left fingers come down 32 mm behind the cart's rear (20 mm if the right arm trailed 25 mm)
L_ENTRY_Z = L.TOP + 0.080  # behind the cart (pad tips 3 mm under its top); on the way the tips pass 9 mm over it
L_END_Y = L.TARGET_Y - L.CART_HALF[1] - FINGER_FRONT + TRAIL_L  # +0.162
BACK_OFF = 0.040  # after each push the fingers back away from the cart before rising
L_PARK = (0.230, 0.200, L.TOP + 0.150)  # the left arm waits here, out of the cart's path, while the right pushes
L_END = (0.220, 0.110, L.TOP + 0.150)  # where the left arm ends, moving away from the cart (never over it): its
#                                       finger zone 40 mm from the cart and 51 mm from the runners, open
R_PARK = (0.200, -0.250, L.TOP + 0.150)


def _span(wps) -> float:
    return sum(w.duration for w in wps)


def _seat(x: float, yaw: float, release: float) -> list[Waypoint]:
    """Lower a carried runner into the seats at x, let go of it, rise."""
    return [
        Waypoint(1.0, (x, 0.0, RUNNER_ENTRY_Z), RUNNER_GRIP, "down to just above the seats", yaw),
        Waypoint(0.5, (x, 0.0, RUNNER_ENTRY_Z), RUNNER_GRIP, "settle over the seats", yaw),
        Waypoint(2.0, (x, 0.0, RUNNER_PLACE_Z), RUNNER_GRIP, "slowly down into the seats", yaw),
        Waypoint(0.5, (x, 0.0, RUNNER_PLACE_Z), RUNNER_GRIP, "settle", yaw),
        Waypoint(1.5, (x, 0.0, RUNNER_PLACE_Z), release, "let go of the runner", yaw),
        Waypoint(0.5, (x, 0.0, RUNNER_PLACE_Z), release, "settle", yaw),
        Waypoint(1.5, (x, 0.0, RUNNER_CARRY_Z), release, "straight up", yaw),
    ]


def _pick(x: float, y: float, yaw: float, travel: float, finger_in: float) -> list[Waypoint]:
    """Go over a runner lying on the table, come down round it and grip it across its width, lift it."""
    return [
        Waypoint(travel, (x, y, HOVER_Z), finger_in, "over the runner", yaw),
        Waypoint(1.5, (x, y, RUNNER_PICK_Z + 0.030), OPEN, "down to just above it", yaw),
        Waypoint(1.5, (x, y, RUNNER_PICK_Z), OPEN, "slowly down round its sides", yaw),
        Waypoint(0.5, (x, y, RUNNER_PICK_Z), OPEN, "settle", yaw),
        Waypoint(1.5, (x, y, RUNNER_PICK_Z), RUNNER_GRIP, "grip the runner", yaw),
        Waypoint(0.5, (x, y, RUNNER_PICK_Z), RUNNER_GRIP, "settle the grip", yaw),
        Waypoint(1.5, (x, y, RUNNER_CARRY_Z), RUNNER_GRIP, "lift", yaw),
    ]


def _left_runners() -> list[Waypoint]:
    """Left arm: runner B (the far one) into the seats at x 0.365, then runner A into those at x 0.285."""
    bx, by = L.RUNNERS["runner_b"]
    ax, ay = L.RUNNERS["runner_a"]
    sb, sa = L.SEATED_X["b"], L.SEATED_X["a"]
    return (
        _pick(bx, by, YAW_B_PICK, 3.0, OPEN)
        + [Waypoint(2.5, (0.260, 0.240, RUNNER_CARRY_Z), RUNNER_GRIP, "carry it over the bed, a quarter turn", 0.0),
           Waypoint(2.5, (sb, 0.0, RUNNER_CARRY_Z), RUNNER_GRIP, "over its seats, another quarter turn", YAW_A)]
        + _seat(sb, YAW_A, RUNNER_RELEASE)
        + _pick(ax, ay, YAW_A, 2.5, OPEN)
        + [Waypoint(3.0, (sa, 0.0, RUNNER_CARRY_Z), RUNNER_GRIP, "carry it over its seats", YAW_A)]
        + _seat(sa, YAW_A, RUNNER_RELEASE)
        + [Waypoint(3.0, L_PARK, CLOSED, "out of the cart's way, fingers closed for pushing", YAW_PUSH)]
    )


def _right_push() -> list[Waypoint]:
    x, z = L.CART_X, PUSH_Z
    return [
        Waypoint(8.0, (x, R_END_Y, z), CLOSED, "push the cart over the runners until it is past the gap", YAW_PUSH),
        Waypoint(0.5, (x, R_END_Y, z), CLOSED, "settle", YAW_PUSH),
        Waypoint(1.0, (x, R_END_Y - BACK_OFF, z), CLOSED, "back away from the cart", YAW_PUSH),
        Waypoint(1.5, (x, R_END_Y - BACK_OFF, L.TOP + 0.120), CLOSED, "up", YAW_PUSH),
    ]


def _right() -> list[Waypoint]:
    """Right arm: fingers closed, down behind the cart; wait for both runners to be seated and the left arm parked;
    push the cart along the dock and over the runners until it is past the gap (its centre at y +0.05: the arm
    reaches no farther); back off, rise and park out of the way, open."""
    x, z = L.CART_X, PUSH_Z
    ready = [
        Waypoint(3.0, (x, R_START_Y, L.TOP + 0.100), CLOSED, "over the dock behind the cart, fingers closing",
                 YAW_PUSH),
        Waypoint(1.5, (x, R_START_Y, z), CLOSED, "down behind the cart's rear face", YAW_PUSH),
    ]
    wait = _span(_left_runners()) - _span(ready)
    return ready + [Waypoint(wait, (x, R_START_Y, z), CLOSED, "wait for the runners to be seated", YAW_PUSH)] \
        + _right_push() + [
            Waypoint(2.5, R_PARK, CLOSED, "park out of the left arm's way", YAW_PUSH),
            Waypoint(1.5, R_PARK, OPEN, "open", YAW_PUSH),
    ]


def _left() -> list[Waypoint]:
    """Left arm: lay both runners into their seats, park out of the cart's way while the right arm pushes, then
    come down behind the cart once the right arm has risen, push it the rest of the way onto its mark, back off,
    rise, move clear, open."""
    x, z = L.CART_X, PUSH_Z
    lay = _left_runners()
    # The right arm starts its push as this arm reaches its park, and its park move once the push is done.
    out = lay + [Waypoint(_span(_right_push()), L_PARK, CLOSED, "wait while the right arm pushes", YAW_PUSH)]
    return out + [
        Waypoint(3.0, (x, L_START_Y, L_ENTRY_Z), CLOSED, "over the cart to behind its rear face", YAW_PUSH),
        Waypoint(1.5, (x, L_START_Y, z), CLOSED, "down behind the cart", YAW_PUSH),
        Waypoint(0.5, (x, L_START_Y, z), CLOSED, "settle", YAW_PUSH),
        Waypoint(7.0, (x, L_END_Y, z), CLOSED, "push the cart onto its mark on the truck bed", YAW_PUSH),
        Waypoint(0.5, (x, L_END_Y, z), CLOSED, "settle", YAW_PUSH),
        Waypoint(1.0, (x, L_END_Y - BACK_OFF, z), CLOSED, "back away from the cart", YAW_PUSH),
        Waypoint(1.5, (x, L_END_Y - BACK_OFF, L.TOP + 0.110), CLOSED, "up", YAW_PUSH),
        Waypoint(2.5, L_END, CLOSED, "clear of the cart and the runners", YAW_PUSH),
        Waypoint(1.5, L_END, OPEN, "open", YAW_PUSH),
    ]


def demo():
    """Both arms on one clock. DEMO_HOLD_ONLY=1: no plan (both arms hold the ready pose, a settling check).
    DEMO_IDLE_ARM=left|right: that arm's plan is dropped (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
