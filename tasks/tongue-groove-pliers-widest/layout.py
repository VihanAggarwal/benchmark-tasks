"""Every dimension and threshold of the task, in one place, and the pliers' geometry as plain functions.

Channellock 420 straight-jaw tongue-and-groove pliers, 9.5 in. Maker's page: 241.3 mm long, 1.5 in capacity,
1.12 in jaws, 0.47 in joint, 1.57 in handle span, 0.87 lb (394.6 g); retailers list 5 positions. Every other size
below (the 6 mm groove pitch, the part sizes) and the release angle, open stop and slide force are our estimates of
that tool (see the README, Simplifications).

How the real tool is built (US 7,017,458 B2, its description of the usual design; Wikipedia, "Tongue-and-groove
pliers"): the SLOTTED half carries the channel, the arc grooves and the upper, outer jaw; the other half (here the
RIVET HALF) carries the pivot rivet, the tongue and the lower, inner jaw. With the rivet at the channel's end nearest
the jaws the jaws are at their narrowest (groove 1); sliding the rivet back along the channel, toward the handles,
widens them (groove 5).

The plan view is laid out from a photograph of the maker's 12 in pliers of the same pattern lying flat with the rivet
half on top (Wikimedia Commons, "Channellock tongue-and-groove pliers.jpg", CC BY-SA 3.0; proportions only), with the
handles toward the viewer and the slotted half's handle on the left:
- the two halves cross at the rivet: the slotted half's handle is the left one and its jaw the upper right one; the
  rivet half's handle is the right one and its jaw the lower left one;
- the slot leans about 31 deg off the handle line, its handle end toward the slotted half's own handle (32 here);
- the outer jaw's face is square to the slot, so a slide of one pitch opens the jaws one pitch; the jaws stand ahead
  and to the left of the rivet at about 57 deg to the handle line (58 here; Wikipedia: 45 to 60 deg);
- closed in groove 1 the handles lie about parallel on either side of the rivet (here the slotted handle's centre
  line 18 mm from the rivet axis and the other 9 mm on the far side, 40 mm across their outer edges, the maker's
  handle span); sliding the rivet to groove 5 carries the rivet half 12.7 mm toward the slotted handle and 20.4 mm
  back, so in groove 5 the closed handles are 27 mm across and the rivet half's handle stands 10 mm proud.

World frame (station): x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at
x 0, y +-0.305 and the bench (the station's table top) is at z 0.7475.

Frames, in millimetres unless named:
- the SLOT frame is mechkit's base frame of the assembly: origin on the rivet axis in groove 1, on the bench top;
  x' along the slot toward its jaw end (world heading -32 deg), y' 90 deg counter-clockwise from it, z up. The slot
  runs from the origin along -x' (world heading 148 deg, toward the robot and its left): the slider ``slip`` reads 0
  in groove 1 and 24 mm in groove 5 and moves the rivet along -x' (mechkit axis "-X"). The hinge ``spread`` turns
  about +z through the rivet.
- the SLOTTED frame: x along the slotted half's handle line, toward the handle end, turned 212 deg from the slot
  frame (world heading 180 deg: straight back toward the arms); y 90 deg counter-clockwise from it.
- the RIVET-HALF frame: origin on the rivet, x along the rivet half's handle line, turned 212 + sigma deg from the
  slot frame, where sigma is the handle spread (the angle from the slotted handle to the rivet half's handle,
  counter-clockwise from above; 0 with the handles parallel, the closed stop). The pliers are drawn (mechkit's joint
  zero) at the 30 deg start in groove 1, so the hinge reads sigma - 30 deg.

Heights (z over the bench, mm): the head plate, the slotted half's arm to its handle and the jaw shoulder 0 to 5;
the ridges and the tongue rib 5 to 6; the lap plate and the rivet half's bridges 6 to 10; the rivet's cap 10 to 12;
the handles and jaws 0 to 12. No box of one half meets a box of the other half, and the rivet meets no box of the
slotted half, at any spread from 0 to 85 deg and any slide from 0 to 24 mm (``interpenetrations``), except the rib
and the ridges, which are the tongue and the grooves.
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475  # the station bench top
MM = 0.001
CONTROL_HZ = 20.0

# -- the catalogue -------------------------------------------------------------------------------------------------
CAT_LENGTH_MM = 241.3  # 9.5 in
CAT_CAPACITY_MM = 38.1  # 1.5 in
CAT_JAW_MM = 28.4  # 1.12 in
CAT_JOINT_MM = 11.9  # 0.47 in
CAT_HANDLE_SPAN_MM = 39.9  # 1.57 in
CAT_MASS = 0.3946  # 0.87 lb
POSITIONS = 5

# -- materials -----------------------------------------------------------------------------------------------------
STEEL = 7.85  # g/cm3
PVC_DIP = 1.25  # g/cm3: the handles' plastisol dip
FRICTION = 0.4  # pliers to bench (steel and PVC on oak), static and dynamic

# -- the start pose ------------------------------------------------------------------------------------------------
B1 = (0.380, 0.000)  # m, world: the rivet in groove 1 (the slot frame's origin)
SLOT_YAW_DEG = -32.0  # world heading of the slot frame's x' (along the slot, toward its jaw end)
SLOT_YAW = math.radians(SLOT_YAW_DEG)
SLIDE_HEADING_DEG = SLOT_YAW_DEG + 180.0  # 148: the rivet's way from groove 1 to groove 5 (toward the handles)
SLOTTED_TURN_DEG = 212.0  # slotted frame from the slot frame (world heading 180): the slot leans 32 deg off it
SIGMA0_DEG = 30.0  # the handles start 30 deg apart
ORIGIN = (B1[0], B1[1], TOP)

# -- the joint -----------------------------------------------------------------------------------------------------
PITCH = 0.006  # m between grooves
GROOVES = 5
SLIDE_TRAVEL = PITCH * (GROOVES - 1)  # 0.024 m: groove 1 at 0, groove 5 at 24 mm
SLIDE_AXIS = "-X"  # mechkit axis of the slider: the slide moves the rivet along -x' (toward the handles)
SLIDE_DIR = np.array([-1.0, 0.0])  # the same, in the slot frame's plane
SLIDE_FRICTION = 1.5  # the slide's joint friction setting (mechkit.py: an effort in N in Isaac Sim 5+)
LOCK_STIFFNESS = 10000.0  # N/m: the tongue held toward its groove's centre (10 N moves it 1 mm)
LOCK_DAMPING = 60.0  # N s/m: about 0.7 of critical for the rivet and rivet half (0.18 kg)
# Free mode (72 deg and above): the target is set each control step where the rivet would be at the step's end if it
# kept 0.9 of its last speed. The target is written once per 50 ms step and this spring settles in 25 ms, so a
# target at the rivet itself would hold it like a 500 N s/m damper (k x 50 ms; mechkit's own note: a followed
# target creeps at force / spring per step); leading by the last step's travel lets a pushed rivet keep moving.
FOLLOW_GAIN = 0.9  # the lead as a fraction of the rivet's travel over the last control step: at 1.0 a pushed rivet
# coasted to the slot's end by itself (GPU, 13 Nov: a 30 mm push ran the full 24 mm); 0.9 leaves a 50 N s/m drag
# (GPU, 13 Nov: a 24 mm push stopped 15.5 mm along the slot)
FOLLOW_LEAD_MAX = 0.001  # m: the lead is clipped to 1 mm (20 mm/s)
RELEASE_DEG = 72.0  # at and above this spread the tongue is clear of the ridge ends: the joint slides
STOP_DEG = 85.0  # open stop
CLOSED_DEG = 0.0  # closed stop: the jaws meet in groove 1 and the handles lie parallel, 40 mm across
HINGE_FRICTION = 0.02  # N m
HINGE_DAMPING = 0.002  # N m s/rad
HINGE_LOWER = math.radians(CLOSED_DEG - SIGMA0_DEG)  # -0.524 rad
HINGE_UPPER = math.radians(STOP_DEG - SIGMA0_DEG)  # 0.960 rad

# -- masses (kg): the three bodies, from masses_g() below (the parts' sizes and materials) -------------------------
SLOTTED_MASS = 0.2074
RIVET_MASS = 0.0052
RIVET_HALF_MASS = 0.1787
TOTAL_MASS = SLOTTED_MASS + RIVET_MASS + RIVET_HALF_MASS  # 0.3913

# -- the shapes (mm): (centre (x, y, z), size (x, y, z), yaw deg) --------------------------------------------------
# Handles: a 10.4 x 8 mm steel bar in a PVC dip, 13 x 12 mm over the dip.
HANDLE_CORE_SECTION = (10.4, 8.0)
HANDLE_SECTION = (13.0, 12.0)

# Slotted half. The handle and its arm are given in the SLOTTED frame, the rest in the slot frame.
SLOTTED_OFF = 18.0  # the slotted handle's centre line off its frame's x axis (the rivet axis in groove 1)
RIVET_OFF = 9.0  # the rivet half's handle centre line off the rivet axis, on the other side
HANDLE_U = (58.0, 184.5)  # the slotted handle's full-height part along its line (the arm reaches in to the head)
SLOTTED_HANDLE = ((121.25, -SLOTTED_OFF, 6.0), (126.5, 13.0, 12.0), 0.0)
SLOTTED_ARM = ((44.0, -SLOTTED_OFF, 2.5), (28.0, 13.0, 5.0), 0.0)  # z 0 to 5, from the head plate out to the handle
HEAD_X = (-36.0, 24.0)  # head plate along the slot
HEAD_W = 26.0
HEAD_T = 5.0
SLOT_X = (-28.2, 4.2)  # the slot (8.4 mm wide): the rivet at 0 in groove 1, at -24 in groove 5
SLOT_W = 8.4
SHOULDER = ((26.5, 3.5, 2.5), (5.0, 19.0, 5.0), 0.0)  # z 0 to 5: from the head plate to the outer jaw
OUTER_JAW = ((34.5, 23.0, 6.0), (42.0, 11.0, 12.0), 90.0)  # the upper, outer jaw: its face square to the slot
RIDGE_CENTRES = (3.0, -3.0, -9.0, -15.0, -21.0, -27.0)  # x' of each ridge arc's centre (grooves between them)
RIDGE_R = 18.0
RIDGE_W, RIDGE_H = 1.5, 1.0  # ridges 1.5 mm wide, 1 mm high, z 5 to 6
RIDGE_PSI = (-40.0, 26.95)  # deg about the slot's forward direction (+x'), positive toward y' > 0; 26.95 puts the
# ridges' ends just clear of the rib at 72 deg: 42 + asin(18.75 sin 26.95 / 17) = 71.99
RIDGE_SPLIT = 14.5  # deg: 18 sin 14.5 = 4.5, so the split chords' ends stay clear of the 8 mm pin as it passes

# Rivet (the slider's link), in the slot frame at groove 1.
PIN_D, PIN_H = 8.0, 10.0  # z 0 to 10
CAP_D, CAP_H = 10.0, 2.0  # the rivet's head, z 10 to 12: the 12 mm joint
JOINT_STACK = (HEAD_T, RIDGE_H, 4.0, CAP_H)  # plate, ridges and tongue, lap plate, cap: 12 mm

# Rivet half, given in the RIVET-HALF frame (turned 212 + sigma deg into the slot frame), except the parts given in the
# slot frame at the closed stop in groove 1 (marked; ``_closed_to_rivet_half`` turns them into the rivet-half frame).
RIVET_HANDLE_R = (44.0, 174.0)  # the handle starts 44 mm out, clear of the head plate; a bridge joins it to the lap
RIVET_HANDLE = ((109.0, RIVET_OFF, 6.0), (130.0, 13.0, 12.0), 0.0)
HANDLE_BRIDGE = ((28.0, RIVET_OFF, 8.0), (32.0, 13.0, 4.0), 0.0)  # z 6 to 10: lap plate to handle
LAP_PLATE = ((-4.0, 4.0, 8.0), (32.0, 32.0, 4.0), 0.0)  # z 6 to 10, over the head plate
NECK_CLOSED = ((17.5, 19.0, 8.0), (20.0, 12.0, 4.0), 58.0)  # z 6 to 10, slot frame at the closed stop: lap to jaw
INNER_JAW_CLOSED = ((23.5, 30.0, 6.0), (28.0, 11.0, 12.0), 90.0)  # the lower, inner jaw, slot frame closed
RIB_ARC = (318.0 - SLOTTED_TURN_DEG, 354.0 - SLOTTED_TURN_DEG)  # deg, rivet-half frame: 106 to 142
_RIB_MID = 18.0 * math.cos(math.radians(9.0))  # a chord of 18 deg on the 18 mm arc sits 17.78 mm out
RIB = tuple(((round(_RIB_MID * math.cos(math.radians(a)), 3), round(_RIB_MID * math.sin(math.radians(a)), 3), 5.5),
             (2.0, 5.63, 1.0), a) for a in (RIB_ARC[0] + 9.0, RIB_ARC[0] + 27.0))  # the tongue: two chords
RIB_R = (17.0, 19.0)  # mm, inner and outer radius

# -- grips (demo) --------------------------------------------------------------------------------------------------
GRIP_U = 0.162  # m out along the rivet half's handle line from the rivet (12 mm inside the handle's end)
GRIP_U_SLOTTED = 0.170  # m out along the slotted handle's line (14.5 mm inside its end)
SLOTTED_OFFSET = SLOTTED_OFF * MM  # m
RIVET_OFFSET = RIVET_OFF * MM  # m
SLOTTED_HEADING_DEG = SLOT_YAW_DEG + SLOTTED_TURN_DEG  # 180

# -- the scorer's thresholds -----------------------------------------------------------------------------------------
GROOVE_BAND = 0.0015  # m: in groove n within 1.5 mm of its centre along the slot (a quarter pitch)
LATCH_SLIDE = 0.003  # m: the 0.2 milestone, slid half a pitch out of groove 1 with the joint open
FLAT_TILT_DEG = 3.0
FLAT_LOW = 0.002  # m: each half's lowest point within 2 mm of the bench top
OFF_BENCH = 0.050  # m: any part's lowest point this far below the bench top at the end: 0
LET_GO = 0.025  # m: every finger point at least 25 mm from the pliers
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001  # m
STILL_TURN_DEG = 1.0
EPS = 1e-9
# Fallback reset detection (rigstate's and mechkit's reset hooks come first): the pliers back on their start pose in
# groove 1 at 30 deg with a jump no trial makes in one control step.
RESET_START_TOL = 0.005
RESET_JUMP = 0.050
RESET_SLIDE_JUMP = 0.003
RESET_TURN_DEG = 20.0

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
BLUE = (0.10, 0.30, 0.75)  # the handles' dip
STEEL_GREY = (0.55, 0.56, 0.58)
DARK_STEEL = (0.30, 0.31, 0.33)


# == geometry ======================================================================================================


def rot2(deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])


def slot_dir() -> np.ndarray:
    """World unit vector of the rivet's way along the slot from groove 1 to groove 5 (heading 148 deg)."""
    a = math.radians(SLIDE_HEADING_DEG)
    return np.array([math.cos(a), math.sin(a)])


def rivet_xy(q: float) -> np.ndarray:
    """Slot-frame position (m) of the rivet at slide ``q`` (m)."""
    return float(q) * SLIDE_DIR


def groove_centre(n: int) -> float:
    """Slide position (m) of groove n (1..5)."""
    return round(PITCH * (n - 1), 9)


def groove_of(q: float) -> int | None:
    """The groove (1..5) the rivet is in: its centre within GROOVE_BAND of that groove's centre, else None."""
    for n in range(1, GROOVES + 1):
        if abs(float(q) - groove_centre(n)) <= GROOVE_BAND + EPS:
            return n
    return None


def nearest_groove_target(q: float) -> float:
    """The groove centre the tongue drops into when the handles close below the release angle: the nearest one, and
    from exactly half way between two the one nearer the jaws."""
    k = min(GROOVES - 1, max(0, int(math.floor(float(q) / PITCH + 0.5 - 1e-9))))
    return round(PITCH * k, 9)


def sigma_deg(q_hinge: float) -> float:
    """Handle spread (deg) from the hinge reading (rad, 0 as drawn at 30 deg)."""
    return SIGMA0_DEG + math.degrees(float(q_hinge))


def hinge_of(sigma: float) -> float:
    return math.radians(float(sigma) - SIGMA0_DEG)


def locked(sigma: float) -> bool:
    return float(sigma) < RELEASE_DEG


def to_world_xy(local_xy, origin=ORIGIN, yaw=SLOT_YAW) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    p = np.asarray(local_xy, dtype=float)
    return np.array([origin[0] + c * p[0] - s * p[1], origin[1] + s * p[0] + c * p[1]])


def to_slot_xy(world_xy, origin=ORIGIN, yaw=SLOT_YAW) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    d = np.asarray(world_xy, dtype=float)[:2] - np.asarray(origin, dtype=float)[:2]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


def rivet_grip(sigma: float, q: float = 0.0, u: float = GRIP_U, origin=ORIGIN) -> np.ndarray:
    """World xy of the point ``u`` m out along the rivet half's handle centre line, rivet at slide ``q``."""
    local = rivet_xy(q) + rot2(SLOTTED_TURN_DEG + sigma) @ np.array([u, RIVET_OFFSET])
    return to_world_xy(local, origin)


def slotted_grip(u: float = GRIP_U_SLOTTED, origin=ORIGIN) -> np.ndarray:
    """World xy of the point ``u`` m out along the slotted half's handle centre line."""
    return to_world_xy(rot2(SLOTTED_TURN_DEG) @ np.array([u, -SLOTTED_OFFSET]), origin)


def rivet_grip_yaw_deg(sigma: float) -> float:
    """Gripper yaw (deg) whose fingers close across the rivet half's handle (yam_arena: yaw 0 closes along y)."""
    return SLOTTED_HEADING_DEG + sigma - 180.0


def slotted_grip_yaw_deg() -> float:
    return SLOTTED_HEADING_DEG - 180.0


def _shape_in_slot(centre, size, yaw, turn):
    c = rot2(turn) @ np.array(centre[:2])
    return (float(c[0]), float(c[1]), float(centre[2])), tuple(float(v) for v in size), float(yaw + turn)


def _closed_to_rivet_half(centre, size, yaw):
    """A rivet-half box given in the slot frame at the closed stop in groove 1, in the rivet-half frame."""
    return _shape_in_slot(centre, size, yaw, -(SLOTTED_TURN_DEG + CLOSED_DEG))


NECK = _closed_to_rivet_half(*NECK_CLOSED)
INNER_JAW = _closed_to_rivet_half(*INNER_JAW_CLOSED)


def ridge_chords() -> list[tuple]:
    """The 15 ridge chords (slot frame): (centre, size, yaw deg). Each arc of radius 18 mm about a ridge centre on
    the slot axis runs from psi -40 to +26.95 deg about the slot's forward direction (+x', positive toward y' > 0): a
    closing chord (-40 to -14.5), an opening chord (14.5 to 26.95) and, for the three front arcs over solid plate, a
    middle chord (-14.5 to 14.5); the other three are split by the slot."""
    out = []
    for k, xc in enumerate(RIDGE_CENTRES):
        spans = [(RIDGE_PSI[0], -RIDGE_SPLIT), (RIDGE_SPLIT, RIDGE_PSI[1])]
        if k < 3:
            spans.insert(1, (-RIDGE_SPLIT, RIDGE_SPLIT))
        for a, b in spans:
            pa = np.array([xc, 0.0]) + RIDGE_R * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
            pb = np.array([xc, 0.0]) + RIDGE_R * np.array([math.cos(math.radians(b)), math.sin(math.radians(b))])
            mid, d = (pa + pb) / 2, pb - pa
            out.append(((float(mid[0]), float(mid[1]), 5.0 + RIDGE_H / 2), (float(np.linalg.norm(d)), RIDGE_W, RIDGE_H),
                        math.degrees(math.atan2(d[1], d[0]))))
    return out


def slotted_shapes() -> list[tuple]:
    """Every box of the slotted half in the slot frame (mm): (name, centre, size, yaw deg). The head plate is four
    bars round the slot."""
    out = [("handle", *_shape_in_slot(*SLOTTED_HANDLE, SLOTTED_TURN_DEG)),
           ("arm", *_shape_in_slot(*SLOTTED_ARM, SLOTTED_TURN_DEG))]
    x0, x1 = HEAD_X
    s0, s1 = SLOT_X
    hw, sw, z = HEAD_W / 2, SLOT_W / 2, HEAD_T / 2
    out += [
        ("head_rear", ((x0 + s0) / 2, 0.0, z), (s0 - x0, HEAD_W, HEAD_T), 0.0),
        ("head_front", ((s1 + x1) / 2, 0.0, z), (x1 - s1, HEAD_W, HEAD_T), 0.0),
        ("head_left", ((s0 + s1) / 2, (sw + hw) / 2, z), (s1 - s0, hw - sw, HEAD_T), 0.0),
        ("head_right", ((s0 + s1) / 2, -(sw + hw) / 2, z), (s1 - s0, hw - sw, HEAD_T), 0.0),
        ("shoulder", *SHOULDER),
    ]
    out += [(f"ridge{i}", *r) for i, r in enumerate(ridge_chords())]
    out.append(("jaw", *OUTER_JAW))
    return out


def rivet_half_local_shapes() -> list[tuple]:
    """Every box of the rivet half in its own frame (mm): (name, centre, size, yaw deg)."""
    out = [("handle", *RIVET_HANDLE), ("bridge", *HANDLE_BRIDGE), ("lap_plate", *LAP_PLATE), ("neck", *NECK),
           ("jaw", *INNER_JAW)]
    out += [(f"rib{i}", *r) for i, r in enumerate(RIB)]
    return out


def rivet_half_shapes(sigma: float = SIGMA0_DEG, q: float = 0.0) -> list[tuple]:
    """Every box of the rivet half in the slot frame (mm) at spread ``sigma`` with the rivet at slide ``q`` (m)."""
    turn = SLOTTED_TURN_DEG + sigma
    shift = rivet_xy(q) / MM
    out = []
    for name, c, s, y in rivet_half_local_shapes():
        c2, s2, y2 = _shape_in_slot(c, s, y, turn)
        out.append((name, (c2[0] + float(shift[0]), c2[1] + float(shift[1]), c2[2]), s2, y2))
    return out


def box_volume(size) -> float:
    return float(size[0] * size[1] * size[2]) / 1000.0  # cm3 from mm


def handle_mass(length_mm: float) -> float:
    """g: a handle's steel core plus the PVC dip around it."""
    core = length_mm * HANDLE_CORE_SECTION[0] * HANDLE_CORE_SECTION[1] / 1000.0
    outer = length_mm * HANDLE_SECTION[0] * HANDLE_SECTION[1] / 1000.0
    return core * STEEL + (outer - core) * PVC_DIP


def _steel(shapes, names) -> float:
    return sum(box_volume(s) for n, _, s, _ in shapes if n.startswith(names)) * STEEL


def masses_g() -> dict:
    """Mass of every part (g) from its size and material; the sums are the three bodies' masses."""
    sl, rh = slotted_shapes(), rivet_half_local_shapes()
    pin = math.pi * (PIN_D / 2) ** 2 * PIN_H / 1000.0 * STEEL
    cap = math.pi * (CAP_D / 2) ** 2 * CAP_H / 1000.0 * STEEL
    parts = {
        "slotted handle": handle_mass(SLOTTED_HANDLE[1][0]), "slotted arm": _steel(sl, ("arm",)),
        "head plate": _steel(sl, ("head_",)), "shoulder": _steel(sl, ("shoulder",)), "ridges": _steel(sl, ("ridge",)),
        "outer jaw": _steel(sl, ("jaw",)),
        "pin": pin, "cap": cap,
        "rivet handle": handle_mass(RIVET_HANDLE[1][0]), "handle bridge": _steel(rh, ("bridge",)),
        "lap plate": _steel(rh, ("lap_plate",)), "neck": _steel(rh, ("neck",)), "inner jaw": _steel(rh, ("jaw",)),
        "rib": _steel(rh, ("rib",)),
    }
    parts["slotted half"] = sum(parts[k] for k in ("slotted handle", "slotted arm", "head plate", "shoulder", "ridges",
                                                   "outer jaw"))
    parts["rivet"] = parts["pin"] + parts["cap"]
    parts["rivet half"] = sum(parts[k] for k in ("rivet handle", "handle bridge", "lap plate", "neck", "inner jaw",
                                                 "rib"))
    parts["total"] = parts["slotted half"] + parts["rivet"] + parts["rivet half"]
    return parts


def _box_corners_2d(centre, size, yaw):
    u = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    v = np.array([-u[1], u[0]])
    c = np.asarray(centre[:2], dtype=float)
    return [c + a * size[0] / 2 * u + b * size[1] / 2 * v for a in (-1, 1) for b in (-1, 1)]


def jaw_gap_mm(n: int, sigma: float = CLOSED_DEG) -> float:
    """Gap (mm) between the two jaw faces in groove n at spread ``sigma`` (at the closed stop they are parallel),
    measured square to the outer jaw's face, which is square to the slot."""
    oj = OUTER_JAW
    ij = [s for s in rivet_half_shapes(sigma, groove_centre(n)) if s[0] == "jaw"][0]
    normal = np.array([math.cos(math.radians(oj[2] - 90)), math.sin(math.radians(oj[2] - 90))])
    return abs(float((np.array(ij[1][:2]) - np.array(oj[0][:2])) @ normal)) - (oj[1][1] + ij[2][1]) / 2


def handle_ends_across_mm(sigma: float = CLOSED_DEG, q: float = 0.0) -> float:
    """Across the two handles' outer edges (mm), square to the slotted handle, at the rivet half's handle end, at
    spread ``sigma`` with the rivet at slide ``q`` (m)."""
    end = rivet_xy(q) / MM + rot2(SLOTTED_TURN_DEG + sigma) @ np.array([RIVET_HANDLE_R[1], RIVET_OFF])
    side = rot2(SLOTTED_TURN_DEG) @ np.array([0.0, 1.0])
    return float(end @ side) + SLOTTED_OFF + HANDLE_SECTION[0]


def overall_length_mm() -> float:
    """From the slotted handle's end to the farthest jaw corner, at the closed stop in groove 1."""
    pts = []
    for _, c, size, yaw in slotted_shapes():
        pts += _box_corners_2d(c, size, yaw)
    for _, c, size, yaw in rivet_half_shapes(CLOSED_DEG):
        pts += _box_corners_2d(c, size, yaw)
    pts = np.array(pts)
    axis = rot2(SLOTTED_TURN_DEG) @ np.array([1.0, 0.0])
    along = pts @ axis
    return float(along.max() - along.min())


def _boxes_meet(a, b, tol: float = 1e-6) -> bool:
    """True when two boxes (centre, size, yaw; z from the centre and height) overlap by more than ``tol`` mm in
    height and in plan (separating axes of the two rectangles)."""
    (ca, sa, ya), (cb, sb, yb) = a, b
    if min(ca[2] + sa[2] / 2, cb[2] + sb[2] / 2) - max(ca[2] - sa[2] / 2, cb[2] - sb[2] / 2) <= tol:
        return False
    pa, pb = np.array(_box_corners_2d(ca, sa, ya)), np.array(_box_corners_2d(cb, sb, yb))
    for yaw in (ya, ya + 90.0, yb, yb + 90.0):
        axis = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
        ra, rb = pa @ axis, pb @ axis
        if min(ra.max(), rb.max()) - max(ra.min(), rb.min()) <= tol:
            return False
    return True


def _post_meets_box(axis_xy, radius: float, z0: float, z1: float, box, tol: float = 1e-6) -> bool:
    """True when an upright round post (axis at ``axis_xy``, mm, from z0 to z1) overlaps a box (centre, size, yaw)
    by more than ``tol`` mm in height and in plan."""
    c, s, yaw = box
    if min(c[2] + s[2] / 2, z1) - max(c[2] - s[2] / 2, z0) <= tol:
        return False
    u = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
    d = np.asarray(axis_xy, dtype=float)[:2] - np.asarray(c[:2], dtype=float)
    lx = abs(float(d @ u)) - s[0] / 2
    ly = abs(float(u[0] * d[1] - u[1] * d[0])) - s[1] / 2
    gap = math.hypot(max(lx, 0.0), max(ly, 0.0)) if (lx > 0.0 or ly > 0.0) else max(lx, ly)
    return gap < radius - tol


def interpenetrations(sigma: float, q: float) -> list[tuple[str, str]]:
    """Pairs (slotted box, rivet-half box) that meet at spread ``sigma`` (deg) and slide ``q`` (m), and pairs
    (slotted box, "pin" or "cap") where the rivet meets the slotted half. The ridges and the rib are left out: they
    are the groove and the tongue."""
    out = []
    axis = rivet_xy(q) / MM
    for na, ca, sa, ya in slotted_shapes():
        for nb, cb, sb, yb in rivet_half_shapes(sigma, q):
            if na.startswith("ridge") and nb.startswith("rib"):
                continue
            if _boxes_meet((ca, sa, ya), (cb, sb, yb)):
                out.append((na, nb))
        for nb, radius, z0, z1 in (("pin", PIN_D / 2, 0.0, PIN_H), ("cap", CAP_D / 2, PIN_H, PIN_H + CAP_H)):
            if _post_meets_box(axis, radius, z0, z1, (ca, sa, ya)):
                out.append((na, nb))
    return out


def rib_band(sigma: float) -> tuple[float, float]:
    """Across-slot band (y', mm) the tongue rib covers at spread ``sigma`` (arc model, radii 17 to 19 mm)."""
    lo, hi = math.inf, -math.inf
    for a in np.linspace(RIB_ARC[0], RIB_ARC[1], 361):
        ang = math.radians(SLOTTED_TURN_DEG + sigma + a)
        for r in RIB_R:
            y = r * math.sin(ang)
            lo, hi = min(lo, y), max(hi, y)
    return lo, hi


def ridge_band() -> tuple[float, float]:
    """Across-slot band (y', mm) the ridges cover (arc model, radii 17.25 to 18.75 mm)."""
    lo, hi = math.inf, -math.inf
    for psi in np.linspace(RIDGE_PSI[0], RIDGE_PSI[1], 671):
        ang = math.radians(psi)
        for r in (RIDGE_R - RIDGE_W / 2, RIDGE_R + RIDGE_W / 2):
            y = r * math.sin(ang)
            lo, hi = min(lo, y), max(hi, y)
    return lo, hi


def rib_ridge_overlap_mm(sigma: float) -> float:
    """How far the rib's across-slot band overlaps the ridges' band (mm; 0 or less: the rib can pass the ridges)."""
    a0, a1 = rib_band(sigma)
    b0, b1 = ridge_band()
    return min(a1, b1) - max(a0, b0)
