"""On the station table: a bench mock-up of a radiography room at about 40% scale. On the midline lies an x-ray table
(a dark grey slab, 260 mm along x by 160 mm) with a pale yellow light field (210 x 130 mm) and a black central-ray
cross painted on it. Along its -y edge stands a cream strip for the x-ray tube's housing, its far end (+x) marked "-"
(a black plate: the cathode), its near end "+" (a red plate: the anode). On a white side cart on the right lies a
right-femur phantom (180 mm long, 40 mm wide, stepped from 24 mm tall at the knee end to 48 mm at the rounded hip
end, 0.22 kg) crosswise, along y; on a grey tray on the left lie two side markers (24 x 24 x 16 mm, 30 g): red R and
blue L. Rule (AP femur and the anode heel effect): lay the phantom flat along the tube's cathode-anode axis with its
thick hip end toward the cathode, centred on the cross and wholly inside the light field, put the R marker in the
field clear of the bone (the L marker stays out), and let go of everything.

The phantom is a loose mechkit compound (mechkit.py, next to this file): one rigid body of three boxes, an upright
cylinder (the femoral head) and two thin painted condyles, so its taper and its rounded hip end are real and the
scorer can read its pose. The scorer also reads the arms (TCP, pad-tip point, opening) through rigstate.py, which
wraps the embodiment's step and reset from this file's import; neither changes anything else the embodiment does.

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
import mechkit as mk  # noqa: E402  (importing it hooks yam_arena so the phantom is built as a compound body)
import rigstate  # noqa: E402
from scoring import Tracker, centre_offset, heading_deg, rotation  # noqa: E402
from yam_arena.demo import Waypoint  # noqa: E402
from yam_arena.task_file import Box  # noqa: E402

rigstate.install()  # reads both arms before every control step (False offline: the checker feeds it instead)

SCENE = "station"
SCORE_OVER = "end"
# Every tier needs the phantom let go (both grippers' finger zones 5 mm or more from it) and still over the last 1 s,
# and full credit needs the same of the R marker; held 1 s more, so the trial cannot end while a gripper is on either.
SUCCESS_HOLD_S = 1.0

INSTRUCTION = (
    "AP right femur. The cream strip along the x-ray table's edge stands for the tube housing; its two ends are "
    "marked - and +. Lay the femur phantom on the table for an AP projection, set the way a radiographer positions "
    "it for this tube, centred on the black cross and wholly inside the yellow light field. Then put the correct "
    "side marker (red R, blue L) inside the light field without covering the bone, and let go of everything."
)


def _span(name, x, y, z0, z1, color, **kw) -> Box:
    """A box given by its extents (x, y in the world; z above the station table top)."""
    return Box(name, pos=((x[0] + x[1]) / 2, (y[0] + y[1]) / 2, L.TOP + (z0 + z1) / 2),
               size=(x[1] - x[0], y[1] - y[0], z1 - z0), color=color, **kw)


def _around(cx, cy, hx, hy):
    return (cx - hx, cx + hx), (cy - hy, cy + hy)


TABLE = _span("xray_table", L.TABLE_X, L.TABLE_Y, 0.0, L.TABLE_H, L.TABLE_GREY)
_CX, _CY = L.CROSS
_Z0, _Z1 = L.TABLE_H, L.FIELD_TOP
_HL, _HW = L.CROSS_LEN / 2, L.CROSS_W / 2
# The cross's lines and the light field are one flat 0.5 mm coat of paint: the lines are boxes flush with the field's
# top, and the field is laid in eight pieces round them, so nothing overlaps and nothing stands proud (the phantom
# lies on one flat surface). The line along x is whole; the line along y is in two halves either side of it.
CROSS = [
    _span("cross_x", (_CX - _HL, _CX + _HL), (_CY - _HW, _CY + _HW), _Z0, _Z1, L.CROSS_BLACK),
    _span("cross_y_left", (_CX - _HW, _CX + _HW), (_CY + _HW, _CY + _HL), _Z0, _Z1, L.CROSS_BLACK),
    _span("cross_y_right", (_CX - _HW, _CX + _HW), (_CY - _HL, _CY - _HW), _Z0, _Z1, L.CROSS_BLACK),
]
_FX, _FY = L.FIELD_X, L.FIELD_Y
FIELD = [
    _span(f"light_field_{i + 1}", x, y, _Z0, _Z1, L.FIELD_YELLOW)
    for i, (x, y) in enumerate((
        ((_FX[0], _CX - _HL), _FY),  # the near part, before the cross
        ((_CX + _HL, _FX[1]), _FY),  # the far part, beyond it
        ((_CX - _HL, _CX + _HL), (_CY + _HL, _FY[1])),  # beside the cross, left
        ((_CX - _HL, _CX + _HL), (_FY[0], _CY - _HL)),  # beside the cross, right
        ((_CX - _HL, _CX - _HW), (_CY + _HW, _CY + _HL)),  # the four quarters between the lines
        ((_CX + _HW, _CX + _HL), (_CY + _HW, _CY + _HL)),
        ((_CX - _HL, _CX - _HW), (_CY - _HL, _CY - _HW)),
        ((_CX + _HW, _CX + _HL), (_CY - _HL, _CY - _HW)),
    ))
]
STRIP = _span("tube_housing", L.STRIP_X, L.STRIP_Y, 0.0, L.STRIP_H, L.STRIP_CREAM)
_SY = (L.STRIP_Y[0] + L.STRIP_Y[1]) / 2
_P0, _P1 = L.STRIP_H, L.STRIP_H + L.PLATE_H
_S1 = _P1 + L.SIGN_H
CATHODE_X = (L.CATHODE_PLATE_X[0] + L.CATHODE_PLATE_X[1]) / 2
ANODE_X = (L.ANODE_PLATE_X[0] + L.ANODE_PLATE_X[1]) / 2
PLATES = [
    _span("cathode_plate", L.CATHODE_PLATE_X, L.STRIP_Y, _P0, _P1, L.CATHODE_BLACK),
    _span("cathode_minus", *_around(CATHODE_X, _SY, L.SIGN_LONG / 2, L.SIGN_W / 2), _P1, _S1, L.SIGN_WHITE),
    _span("anode_plate", L.ANODE_PLATE_X, L.STRIP_Y, _P0, _P1, L.ANODE_RED),
    _span("anode_plus_x", *_around(ANODE_X, _SY, L.SIGN_LONG / 2, L.SIGN_W / 2), _P1, _S1, L.SIGN_WHITE),
    _span("anode_plus_left", (ANODE_X - L.SIGN_W / 2, ANODE_X + L.SIGN_W / 2),
          (_SY + L.SIGN_W / 2, _SY + L.SIGN_SHORT / 2), _P1, _S1, L.SIGN_WHITE),
    _span("anode_plus_right", (ANODE_X - L.SIGN_W / 2, ANODE_X + L.SIGN_W / 2),
          (_SY - L.SIGN_SHORT / 2, _SY - L.SIGN_W / 2), _P1, _S1, L.SIGN_WHITE),
]
CART = _span("side_cart", L.CART_X, L.CART_Y, 0.0, L.CART_H, L.CART_WHITE)
TRAY = _span("marker_tray", L.TRAY_X, L.TRAY_Y, 0.0, L.TRAY_H, L.TRAY_GREY)

PHANTOM = mk.assembly(
    "femur_phantom",
    origin=(L.PH_START[0], L.PH_START[1], L.TOP + L.CART_H),
    yaw=L.PH_START_YAW,
    mass=L.PH_MASS,
    color=L.BONE_WHITE,
    shapes=[
        *(mk.box(c, s) for c, s in L.PH_BOXES),
        mk.cylinder(L.PH_HEAD[0], L.PH_HEAD[1], L.PH_HEAD[2], axis="Z"),
        *(mk.box(c, L.CONDYLE_SIZE, color=L.CONDYLE_GREY) for c in L.CONDYLE_CENTERS),
    ],
)


def _marker(name: str, xy, color) -> Box:
    return Box(name, pos=(xy[0], xy[1], L.TOP + L.TRAY_H + L.MARKER_SIZE[2] / 2), size=L.MARKER_SIZE,
               mass=L.MARKER_MASS, color=color)


R_MARKER = _marker("r_marker", L.R_START, L.MARKER_RED)
L_MARKER = _marker("l_marker", L.L_START, L.MARKER_BLUE)
OBJECTS = [TABLE, *FIELD, *CROSS, STRIP, *PLATES, CART, TRAY, PHANTOM, R_MARKER, L_MARKER]

_TRACKER = Tracker()
_EPISODE = {"mechkit": None}


@rigstate.on_reset
def _new_episode() -> None:
    """rigstate saw the embodiment reset: a new trial (mechkit's reset counter and a teleport back to the start are
    two more signals)."""
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


def _parts(objects):
    return objects["femur_phantom"], objects["r_marker"], objects["l_marker"]


def success(objects) -> bool:
    """1.0 now: the phantom let go and still for 1 s, flat on the table (anterior face up) along the tube axis within
    10 degrees, hip end toward the cathode, wholly in the light field, centre within 15 mm of the cross; the R marker
    let go, still, flat on the table, its centre in the field and 5 mm or more clear of the bone; the L marker's
    centre outside the field; the phantom never on the floor; held 1 s. This runs first every control step, so it
    also feeds the tracker (with the arms rigstate read for this step)."""
    if _EPISODE["mechkit"] != mk.episode():  # mechkit counts resets too: a new trial
        if _EPISODE["mechkit"] is not None:
            _TRACKER.reset()
        _EPISODE["mechkit"] = mk.episode()
    ph, rm, lm = _parts(objects)
    arms = rigstate.arms()
    result = _TRACKER.update(ph, rm, lm, arms)
    row = {"episode": rigstate.episode(), "mechkit_episode": mk.episode(), "step": _TRACKER.steps,
           "score": result.score, "success": result.success, **result.details}
    try:
        off = centre_offset(ph)
        row.update(source=getattr(ph, "source", None), ph_origin=ph.origin, ph_quat=ph.quat_xyzw,
                   ph_vel=ph.lin_vel, ph_yaw_deg=round(heading_deg(rotation(ph.quat_xyzw)), 2),
                   ph_off_x_mm=round(float(off[0]) * 1000, 2), ph_off_y_mm=round(float(off[1]) * 1000, 2),
                   r_pos=rm.pos, r_quat=rm.quat_xyzw, l_pos=lm.pos)
    except (AttributeError, TypeError, ValueError):
        row["state_error"] = True
    if isinstance(arms, dict):
        for side in ("left", "right"):
            arm = arms.get(side)
            row[f"{side}_tcp"] = getattr(arm, "tcp", None)
            row[f"{side}_tips"] = getattr(arm, "tips", None)
            row[f"{side}_opening"] = getattr(arm, "opening", None)
    else:
        row["rigstate_error"] = rigstate.error()
    _trace(row)
    return result.success


def score(objects) -> float:
    """0 if the phantom ever touched the floor, or is not let go and still, not lying flat on the table anterior face
    up (lowest point within 3 mm of the field's top, tilted under 5 degrees), not within 10 degrees of the tube axis
    or not wholly inside the light field. Otherwise 0.25 with its hip end toward the anode; with it toward the
    cathode 0.5 if its centre is more than 15 mm from the cross, 0.75 if centred but the marker condition fails,
    1.0 if centred and the R marker lies let go and still, flat in the field and 5 mm or more clear of the bone, with
    the L marker out of the field. Read from the state at hand (the last step)."""
    return _TRACKER.result(*_parts(objects), rigstate.arms()).score


# -- the scripted demo ---------------------------------------------------------------------------------------
# Finger targets: the joint value is minus half the commanded gap; command 8 mm narrower than the part.
PH_OPEN = -0.030  # 60 mm: 10 mm either side of the 40 mm phantom
PH_GRIP = -(L.PH_W - 0.008) / 2  # 32 mm commanded across its 40 mm
M_OPEN = -0.020  # 40 mm: 8 mm either side of the 24 mm marker
M_GRIP = -(L.MARKER_SIZE[0] - 0.008) / 2  # 16 mm commanded across its 24 mm
# The right arm grips the phantom across its 40 mm width on the 36 mm tall shaft, right over its centre of mass
# (16 mm from mid-length toward the hip, so it hangs level), and turns it a quarter turn clockwise in the air: from
# along y (hip toward +y) to along x (hip toward +x, the cathode). Picked at gripper yaw +90 deg (fingers closing
# along x), laid at yaw 0 (closing along y). The other answer, hip toward the anode, is the same grip at yaw -90 deg
# turned the other way to the same yaw 0: both are inside joint 6's range here, so the turn itself gives nothing away.
YAW_PICK = math.pi / 2
# Trim of the place yaw, for the GPU run: in our runs parts carried in a gripper ended +0.6 to +1.7 degrees
# counter-clockwise of the commanded yaw, so the phantom is laid at -1 degree (10 degrees are allowed either way).
# If ph_yaw_deg in TASK_TRACE at rest is more than 2 degrees from 0, change YAW_TRIM by minus that.
YAW_TRIM = math.radians(-1.0)
YAW_PLACE = 0.0 + YAW_TRIM
GRIP_UP = 0.024  # the TCP 24 mm above the phantom's underside: pad tips 11 mm above it, the pads on 11-36 mm
DROP = 0.001  # the phantom and the marker are let go with their undersides 1 mm above the field
APPROACH = 0.030  # each final descent starts this far above its end height
PH_HOVER = L.TOP + 0.12  # over the cart: the carried phantom's underside 96 mm up
PH_HOVER_PLACE = L.TOP + 0.11  # over the table (joint 4 margin): its underside 86 mm up, 56 mm over the strip
_COM_START = (L.PH_START[0] + L.PH_COM_X * math.cos(L.PH_START_YAW),
              L.PH_START[1] + L.PH_COM_X * math.sin(L.PH_START_YAW))
PH_PICK = (_COM_START[0], _COM_START[1], L.TOP + L.CART_H + GRIP_UP)
# The target: the phantom's centre on the cross, along x, hip toward +x, so its centre of mass (under the grip) is
# 16 mm beyond the cross. Trim of the place point, for the GPU run: carried parts landed about 4 mm short along the
# carry (here mostly +y), so the place point leads by 4 mm in y (the centre may be 15 mm off). If ph_off_x_mm /
# ph_off_y_mm in TASK_TRACE at rest are more than 5 mm from 0, move PH_TRIM by minus the offset.
PH_TRIM = (0.0, 0.004)
PH_PLACE = (L.CROSS[0] + L.PH_COM_X + PH_TRIM[0], L.CROSS[1] + PH_TRIM[1], L.TOP + L.FIELD_TOP + DROP + GRIP_UP)
PH_PARK = (0.20, -0.25, L.TOP + 0.16)

# The left arm grips the R marker across two faces, fingers closing along x (yaw -90 deg: the fingers stay clear of
# the L marker beside it along y), and lays it beside the bone on the +y side (the right femur's lateral side, with
# the patient's head toward the cathode end), about 15 mm clear of the phantom and inside the field.
M_YAW = -math.pi / 2
M_GRIP_UP = 0.017  # TCP 17 mm above the marker's underside: pad tips 4 mm above it, the pads on its 4-16 mm
M_HOVER = L.TOP + 0.10
M_PICK = (L.R_START[0], L.R_START[1], L.TOP + L.TRAY_H + M_GRIP_UP)
MARKER_AT = (0.270, 0.050)  # its centre: its outline 18 mm clear of the bone's, 3 mm inside the field's +y edge
M_TRIM = (0.0, -0.003)  # carried 23 cm toward -y: aimed 3 mm further, so a short landing ends near MARKER_AT
M_PLACE = (MARKER_AT[0] + M_TRIM[0], MARKER_AT[1] + M_TRIM[1], L.TOP + L.FIELD_TOP + DROP + M_GRIP_UP)
M_PARK = (0.20, 0.25, L.TOP + 0.15)
WAIT_MARGIN = 0.5  # the left arm leaves the tray this long after the right arm's gripper is back up


def _right() -> list[Waypoint]:
    """Right arm: grip the phantom over its centre of mass, lift it, carry it over the table while turning it a
    quarter turn clockwise, lower it slowly onto the light field, let go, go up and aside."""
    px, py, pz = PH_PICK
    qx, qy, qz = PH_PLACE
    return [
        Waypoint(2.0, (px, py, PH_HOVER), PH_OPEN, "over the phantom", YAW_PICK),
        Waypoint(1.0, (px, py, pz + APPROACH), PH_OPEN, "down to just above it", YAW_PICK),
        Waypoint(1.5, (px, py, pz), PH_OPEN, "slowly down beside it", YAW_PICK),
        Waypoint(0.5, (px, py, pz), PH_OPEN, "settle", YAW_PICK),
        Waypoint(1.5, (px, py, pz), PH_GRIP, "grip the phantom", YAW_PICK),
        Waypoint(0.3, (px, py, pz), PH_GRIP, "settle the grip", YAW_PICK),
        Waypoint(1.5, (px, py, PH_HOVER), PH_GRIP, "lift it", YAW_PICK),
        Waypoint(3.0, (qx, qy, PH_HOVER_PLACE), PH_GRIP, "carry it over the table, hip end turning to the cathode",
                 YAW_PLACE),
        Waypoint(0.8, (qx, qy, PH_HOVER_PLACE), PH_GRIP, "settle over the table", YAW_PLACE),
        Waypoint(1.0, (qx, qy, qz + APPROACH), PH_GRIP, "down to 3 cm above", YAW_PLACE),
        Waypoint(2.0, (qx, qy, qz), PH_GRIP, "slowly onto the light field", YAW_PLACE),
        Waypoint(0.5, (qx, qy, qz), PH_GRIP, "settle", YAW_PLACE),
        Waypoint(1.5, (qx, qy, qz), PH_OPEN, "let go of it", YAW_PLACE),
        Waypoint(0.3, (qx, qy, qz), PH_OPEN, "settle", YAW_PLACE),
        Waypoint(1.0, (qx, qy, L.TOP + 0.12), PH_OPEN, "straight up", YAW_PLACE),
        Waypoint(1.5, PH_PARK, PH_OPEN, "aside", 0.0),
    ]


def _right_up_at() -> float:
    """When the right arm's gripper is back up over the placed phantom (s from the start)."""
    t = 0.0
    for w in _right():
        t += w.duration
        if w.label == "straight up":
            return t
    raise AssertionError("no 'straight up' waypoint")


def _left() -> list[Waypoint]:
    """Left arm: grip the R marker on the tray and lift it; wait until the right arm has let go of the phantom and
    gone up; carry the marker beside the bone, lower it slowly onto the field, let go, go up and aside."""
    px, py, pz = M_PICK
    qx, qy, qz = M_PLACE
    before = [
        Waypoint(2.0, (px, py, M_HOVER), M_OPEN, "over the R marker", M_YAW),
        Waypoint(1.0, (px, py, pz + APPROACH), M_OPEN, "down to just above it", M_YAW),
        Waypoint(1.5, (px, py, pz), M_OPEN, "slowly down round it", M_YAW),
        Waypoint(0.5, (px, py, pz), M_OPEN, "settle", M_YAW),
        Waypoint(1.5, (px, py, pz), M_GRIP, "grip the R marker", M_YAW),
        Waypoint(0.3, (px, py, pz), M_GRIP, "settle the grip", M_YAW),
        Waypoint(1.5, (px, py, M_HOVER), M_GRIP, "lift it", M_YAW),
    ]
    wait = max(0.5, _right_up_at() + WAIT_MARGIN - sum(w.duration for w in before))
    return before + [
        Waypoint(wait, (px, py, M_HOVER), M_GRIP, "wait for the phantom to be let go", M_YAW),
        Waypoint(2.0, (qx, qy, M_HOVER), M_GRIP, "carry it beside the bone", M_YAW),
        Waypoint(0.5, (qx, qy, M_HOVER), M_GRIP, "settle over its place", M_YAW),
        Waypoint(1.0, (qx, qy, qz + APPROACH), M_GRIP, "down to 3 cm above", M_YAW),
        Waypoint(1.5, (qx, qy, qz), M_GRIP, "slowly onto the light field", M_YAW),
        Waypoint(0.5, (qx, qy, qz), M_GRIP, "settle", M_YAW),
        Waypoint(1.5, (qx, qy, qz), M_OPEN, "let go of it", M_YAW),
        Waypoint(0.3, (qx, qy, qz), M_OPEN, "settle", M_YAW),
        Waypoint(1.0, (qx, qy, L.TOP + 0.12), M_OPEN, "straight up", M_YAW),
        Waypoint(1.5, M_PARK, M_OPEN, "aside", M_YAW),
    ]


def demo():
    """Both arms on one clock: the right arm lays the phantom, the left arm then lays the R marker. DEMO_HOLD_ONLY=1:
    no plan (both arms hold the ready pose, a settling check). DEMO_IDLE_ARM=left|right: that arm's plan is dropped
    (the one-arm check)."""
    if os.environ.get("DEMO_HOLD_ONLY") == "1":
        return {}
    plan = {"left": _left(), "right": _right()}
    plan.pop(os.environ.get("DEMO_IDLE_ARM", ""), None)
    return plan
