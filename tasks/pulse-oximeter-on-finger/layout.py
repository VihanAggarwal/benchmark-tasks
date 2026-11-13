"""Every dimension and threshold of the task, in one place (metres, kilograms, radians unless a name says _DEG or _MM),
station frame: x forward from the line through the two arm bases, y to the left, z up; the station's table top is at
z 0.7475 and every height below is above it.

The scene: a fixed manikin right hand on its little-finger edge, index finger outstretched toward the robot on the
midline, and a loose fingertip pulse oximeter (a Contec CMS50D: 57 x 31 x 32 mm, about 50 g with two AAA cells, the
maker's figures) lying on its side 61.5 mm short of the fingertip.

The oximeter's frame (the assembly's base frame, the battery shell's): origin at the unit's centre, X along its length
toward the lips, Y across the two shells toward the display shell, Z along the pivot. It lies with Z up. ``u`` is the
distance from the back end: X = u - 28.5 mm. The display shell is the link ``jaw`` on one hinge about Z through the
pivot (u 20); a positive angle swings its front toward +Y (the mouth opens) and its tail toward the battery shell's
tail. The two tails touch at 28.07 degrees, so the joint's upper limit, 28 degrees, is the open stop.

The centre line: the line through the pivot midway between the two shells, i.e. the base frame's X turned by half the
pivot angle. Each shell is turned a = q / 2 from it. All the "half gap" formulas below are distances from it.
"""

import math

TOP = 0.7475  # station table top (world z)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
MM = 0.001

# -- the manikin hand (fixed) --------------------------------------------------------------------------------------
FINGER_D = 0.015  # index finger diameter
FINGER_Z = 0.075  # its axis above the table
FINGER_Y = 0.0
TIP_X = 0.270  # the rounded tip's end face (the fingertip point P)
KNUCKLE_X = 0.342
TIP_STEPS = ((0.270, 0.272, 0.010), (0.272, 0.274, 0.013))  # (x0, x1, diameter): two steps on a 7.5 mm ball end
FULL_SECTION_FROM_TIP = 0.004  # the full 15 mm section starts this far from the tip
P = (TIP_X, FINGER_Y, FINGER_Z)  # the fingertip point, table frame (z above the table)
SKIN = (0.90, 0.72, 0.60)
HAND_FRICTION = (0.8, 0.7)

# -- the oximeter (loose articulation) -------------------------------------------------------------------------------
UNIT_LEN = 0.057
UNIT_W = 0.031  # along Z (the pivot), the maker's width
UNIT_H = 0.032  # across the two shells (Y), the maker's height
UNIT_MASS = 0.050
JAW_MASS = 0.014  # display shell (our estimate)
BASE_MASS = UNIT_MASS - JAW_MASS  # battery shell with its cells, 0.036
U0 = 0.0285  # X = u - U0
PIVOT_U = 0.020
PIVOT_X = PIVOT_U - U0  # -0.0085
PIVOT_U_MM = 20.0
SPRING = 0.15  # N m / rad, linear from shut
DAMPING = 0.01  # N m s / rad
OPEN_LIMIT_DEG = 28.0
OPEN_LIMIT = math.radians(OPEN_LIMIT_DEG)  # 0.4887
UNIT_FRICTION = (1.0, 0.8)
BLUE = (0.10, 0.35, 0.80)
DARK = (0.06, 0.08, 0.12)  # display window
RUBBER = (0.04, 0.04, 0.04)
CUSHION = (0.60, 0.62, 0.64)
START = (0.180, 0.000, 0.0155)  # base frame origin (table frame), yaw 0: lies on its side, pivot upright

# Shapes, in mm in (u0, u1), (Y0, Y1), (Z0, Z1) of the base frame at rest, and a colour.
BATTERY_SHELL = [
    ("back_plate", (0, 57), (-16, -13), (-15.5, 15.5), BLUE),
    ("tail_block", (0, 18), (-13, -5), (-15.5, 15.5), BLUE),
    ("grip_pad", (0, 10), (-18.5, -16), (-10, 10), RUBBER),
    ("wall_top", (21, 57), (-13, 13), (11, 15.5), BLUE),
    ("wall_bottom", (21, 57), (-13, 13), (-15.5, -11), BLUE),
    ("stop", (21, 26), (-13, 2), (-11, 11), BLUE),
]
CUSHION_STEPS_U = ((32, 37), (37, 42), (42, 47), (47, 52), (52, 57))
CUSHION_FACES = (5.86, 5.15, 4.43, 3.72, 3.00)  # each step's inner face, mm from the shell's inner datum
DISPLAY_SHELL = [
    ("front_plate_tail", (0, 20), (13, 16), (-15.5, 15.5), BLUE),
    ("window", (20, 50), (13, 16), (-15.5, 15.5), DARK),
    ("front_plate_nose", (50, 57), (13, 16), (-15.5, 15.5), BLUE),
    ("tail_block", (0, 18), (5, 13), (-15.5, 15.5), BLUE),
    ("grip_pad", (0, 10), (16, 18.5), (-10, 10), RUBBER),
    ("stop", (21, 26), (-2, 13), (-10.5, 10.5), BLUE),
]
JAW_CUSHION_Z = (-10.5, 10.5)
BASE_CUSHION_Z = (-11, 11)
STOP_U = 26.0  # the stop's face: 6 mm ahead of the pivot
STOP_CORNER = (6.0, 2.0)  # the battery shell's stop corner that leads, mm (along, across) from the pivot
LIP_U = 57.0
LIP_FACE = 3.00  # mm
INNER_U = 32.0
INNER_FACE = 5.86
GRIP_PAD_EDGE = (10.0, 18.5)  # the grip pad's front edge: u, mm out from the centre plane
TAIL_GAP = 0.010
GRIP_PAD = (0.010, 0.020, 0.0025)  # added rubber pad on each tail: long, wide, thick

# -- the gripper and the finger drive ---------------------------------------------------------------------------------
FULL_GAP = 0.0939  # pads' gap fully open
FINGER_DRIVE = 5.0  # N per mm of each finger's position error (yam_arena: 5000 N/m)
FINGER_CAP = 50.0  # N
PAD_WIDTH_TCP = 0.027  # a pad's width where it meets the tail (TCP level and above)

# -- scoring thresholds ------------------------------------------------------------------------------------------------
CHAMBER_MIN_MM = 4.0  # in the chamber: depth 4 mm or more ...
CHAMBER_MAX_MM = 32.5  # ... and not past the stop (31.0 mm shut, plus contact slack)
CENTRE_TOL_MM = 5.0  # the fingertip's centre within 5 mm of the centre line (v and w)
OVER_DEPTH_MM = 10.0  # over the finger: depth 10 mm or more ...
ALIGN_DEG = 20.0  # ... the unit's length within 20 degrees of the finger's ...
NAIL_SIDE_DEG = 45.0  # ... and its display shell on the nail side (+y) within 45 degrees
OPEN_DEG = 22.0  # jaws open: pivot at 22 degrees or more
SLIDE_SLACK_MM = 1.5  # slid on open: greatest shut depth at most 1.5 mm past the greatest open depth (closing adds 1.1)
HOLD_DEG = 16.3  # the cushions parallel and 15 mm apart: how they sit on this finger
HOLD_BAND_DEG = 1.8  # gripping: within 1.8 degrees of it (a 1 mm feeler stopped within 5 mm of the lips)
HOME_MM = 28.0  # 1.0 needs depth 28 mm or more (the fingertip within 2 mm of the stop)
LET_GO = 0.030  # every finger zone 30 mm or more from the oximeter
TABLE_CLEAR = 0.020  # no part of the oximeter within 20 mm of the table
STILL_MM = 2.0  # through the last 1 s: moved under 2 mm ...
STILL_DEG = 2.0  # ... and turned under 2 degrees
WINDOW_S = 1.0  # the last 1 s (20 steps)
LATCH_S = 0.5  # the 0.5 and 0.25 records: 0.5 s (10 steps)
EPS = 1e-9
EPS_DEG = 1e-6
# Fallback reset detection: the unit back at its start pose (within 3 mm, 2 deg, jaws within 2 deg of shut) in one
# step, after a jump of 30 mm or more.
RESET_START_TOL = 0.003
RESET_START_DEG = 2.0
RESET_JUMP = 0.030


# -- geometry (a = half the pivot angle, radians; results in mm) ------------------------------------------------------


def half_gap_lips(a: float) -> float:
    """Half the mouth: the lip (u 57) from the centre line."""
    return LIP_FACE * math.cos(a) + (LIP_U - PIVOT_U_MM) * math.sin(a)


def half_gap_inner(a: float) -> float:
    """Half the gap at the cushions' inner end (u 32)."""
    return INNER_FACE * math.cos(a) + (INNER_U - PIVOT_U_MM) * math.sin(a)


def lip_line(a: float) -> float:
    """How far the line joining the two lips is ahead of the pivot along the centre line."""
    return (LIP_U - PIVOT_U_MM) * math.cos(a) - LIP_FACE * math.sin(a)


def stop_line(a: float) -> float:
    """How far the stop's leading corners are ahead of the pivot along the centre line."""
    return STOP_CORNER[0] * math.cos(a) + STOP_CORNER[1] * math.sin(a)


def depth_on_stop(a: float) -> float:
    """Depth (fingertip past the lips) with the fingertip against the stop."""
    return lip_line(a) - stop_line(a)


def front_width(a: float) -> float:
    """Across the outside of the two shells at the front end (u 57, 16 mm out): what a video frame shows."""
    return 2 * (16.0 * math.cos(a) + (LIP_U - PIVOT_U_MM) * math.sin(a))


def pads_across(a: float) -> float:
    """Across the two grip pads' front edges (where a flat gripper pad bears)."""
    u, y = GRIP_PAD_EDGE
    return 2 * (y * math.cos(a) - (PIVOT_U_MM - u) * math.sin(a))


def pad_lever(a: float) -> float:
    """The grip pad's front edge's lever arm about the pivot (mm)."""
    u, y = GRIP_PAD_EDGE
    return (PIVOT_U_MM - u) * math.cos(a) + y * math.sin(a)


def pad_force(a: float) -> float:
    """Newtons on each grip pad that hold the jaws at pivot angle 2a against the spring."""
    return SPRING * 2 * a / (pad_lever(a) * MM)


def cushion_force(q_hold: float = math.radians(HOLD_DEG)) -> float:
    """Newtons with which each cushion presses on the finger (contacts centred 22 mm from the pivot)."""
    return SPRING * q_hold / 0.022


def cushion_half_gap(p: float, a: float) -> float | None:
    """Half the gap between the cushions at a point p mm ahead of the pivot along the centre line (None outside
    the cushions, u 32 to 57). The pivot-side edges of the five steps are collinear; between them a step's face is
    flat, so the narrowest point of each step is its pivot-side edge (the steps fall away toward the lips)."""
    best = None
    for (u0, u1), face in zip(CUSHION_STEPS_U, CUSHION_FACES):
        x0, x1 = u0 - PIVOT_U_MM, u1 - PIVOT_U_MM
        # along the centre line a point of the face at shell coordinate x is at x cos a + face sin a
        lo, hi = x0 * math.cos(a) + face * math.sin(a), x1 * math.cos(a) + face * math.sin(a)
        if lo - 1e-9 <= p <= hi + 1e-9:
            x = (p - face * math.sin(a)) / math.cos(a)
            h = face * math.cos(a) + x * math.sin(a)
            best = h if best is None else min(best, h)
    return best


def finger_radius(d: float) -> float:
    """The finger's radius (mm) d mm from its tip (d >= 0)."""
    if d < 0:
        return 0.0
    if d < 2.0:
        return 5.0
    if d < 4.0:
        return 6.5
    return 7.5


def deepest_s(a: float, step: float = 0.05) -> float:
    """The smallest s (fingertip ahead of the pivot along the centre line, mm) a centred finger can reach with the
    jaws held at 2a: blocked by the stop, or by the first point where the finger is wider than the cushions'
    (or the lips') gap. Pure geometry, used offline and by the tests."""
    s_stop = stop_line(a)
    lip = lip_line(a)
    s = lip
    best = lip
    while s >= s_stop - 1e-9:
        ok = True
        p = max(s, INNER_U - PIVOT_U_MM - 1.0)
        while p <= lip + 1e-9:
            h = cushion_half_gap(p, a)
            if h is None:
                h = half_gap_lips(a) if p >= lip - 1e-6 else math.inf
            if finger_radius(p - s) > h + 1e-6:
                ok = False
                break
            p += 0.1
        if not ok:
            break
        best = s
        s -= step
    return max(best, s_stop)


def finger_force_gap(gap_cmd_mm: float) -> float:
    """The pivot angle (rad) a gripper commanded to ``gap_cmd_mm`` holds the clip at, its pads on the grip pads'
    front edges: the finger drive's force (5 N per mm of each finger's error, 50 N cap) against the spring through
    the pad lever; the open stop at 28 degrees."""
    lo, hi = 0.0, OPEN_LIMIT / 2
    if min(FINGER_CAP, FINGER_DRIVE * (pads_across(hi) - gap_cmd_mm) / 2) >= pad_force(hi):
        return OPEN_LIMIT
    if pads_across(0.0) <= gap_cmd_mm:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        f = min(FINGER_CAP, FINGER_DRIVE * (pads_across(mid) - gap_cmd_mm) / 2)
        if f > pad_force(mid):
            lo = mid
        else:
            hi = mid
    return 2 * lo
