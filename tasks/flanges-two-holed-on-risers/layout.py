"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A scaled pipe-rack fit-up bench. Two vertical riser stubs (pipe ends standing up out of the floor) are fixed on the
table, a small one on the left (+y) and a large one on the right (-y). Around each, the plant's two principal
centrelines are painted on the floor in white: the north-south line along x (plant north is +x, marked by a white
square at the line's north end) and the east-west line along y, both through the stub's axis. Two loose weld-neck
flanges lie neck down on the table nearer the arms: a 4-hole one on the left, an 8-hole one on the right, each with
its bolt holes (painted black on its face) ON the centrelines' directions. Each flange is bored for one stub: its
neck's outside diameter is that stub's.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the riser stubs (fixed): upright cylinders, black steel ----------------------------------------------------------
STUB_H = 0.060  # 60 mm tall: the stub end (where a flange's neck rests) is 60 mm above the table
STUBS = {  # name: (axis x, axis y, outside radius)
    "stub_small": (0.330, 0.150, 0.018),  # 36 mm OD
    "stub_large": (0.330, -0.150, 0.025),  # 50 mm OD
}
STUB_END_W = TOP + STUB_H

# -- the painted plant centrelines (fixed, white): around each stub, four strips from its outside surface out to
#    LINE_REACH from its axis (the stub stands where the two lines cross), plus a north mark at the N-S line's +x end
LINE_W = 0.004  # 4 mm wide (a 3 mm line is about 1 px in the setup screenshot)
LINE_T = 0.0005  # 0.5 mm thick paint
LINE_REACH = 0.080  # each arm of a cross ends 80 mm from the stub's axis: 45 mm beyond the larger flange's rim
NORTH_MARK = (0.010, 0.010)  # a 10 x 10 mm white square just past the north (+x) end of each N-S line

# -- the two weld-neck flanges: each one loose rigid body (mechkit compound) -------------------------------------
# Base frame: origin on the flange's axis at the bottom of its neck; local +z up the axis (neck down, face up),
# local +x through the centre of bolt hole 1. A flange = a neck (cylinder, the stub's OD) under a disc (the flange
# proper); on the disc's top face, a dark bore disc in the middle and the bolt holes, painted black, equally spaced
# on the bolt circle starting at local +x. Aluminium model flanges (solid steel ones this size would be 0.35-0.52 kg).
NECK_H = 0.010  # neck 10 mm tall
DISC_T = 0.012  # flange disc 12 mm thick: its underside 10 mm, its face 22 mm above the neck's bottom
PAINT_T = 0.0005  # paint discs 0.5 mm thick on the face
FACE_Z = NECK_H + DISC_T  # 0.022: the face (top of the disc) above the neck's bottom
TOP_Z = FACE_Z + PAINT_T  # 0.0225: the flange's highest point above the neck's bottom
FLANGES = {
    # name: holes, disc radius, neck radius (= its stub's), hole radius, bolt-circle radius, bore radius, mass (kg),
    #       the stub it is bored for, start (x, y), start yaw of hole 1 (rad), colour key
    "flange_4": dict(holes=4, disc_r=0.030, neck_r=0.018, hole_r=0.004, bc_r=0.023, bore_r=0.013, mass=0.12,
                     stub="stub_small", start=(0.210, 0.130), yaw=0.0),
    "flange_8": dict(holes=8, disc_r=0.035, neck_r=0.025, hole_r=0.0035, bc_r=0.028, bore_r=0.020, mass=0.18,
                     stub="stub_large", start=(0.210, -0.130), yaw=0.0),
}
ALUMINIUM_DENSITY = 2700.0  # kg/m3: the masses above are the solid aluminium bodies' (checked in the tests)


def pitch_deg(name: str) -> float:
    """The angle between neighbouring bolt holes (90 for 4 holes, 45 for 8)."""
    return 360.0 / FLANGES[name]["holes"]


def two_hole_deg(name: str) -> float:
    """Where the holes straddle a centreline: half the pitch (45 for 4 holes, 22.5 for 8)."""
    return pitch_deg(name) / 2.0


# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
CENTRE_TOL = 0.006  # 'centred': the flange's axis within 6 mm of its stub's axis, seen from above
SEAT_TOL = 0.003  # 'on the stub end': the neck's bottom within 3 mm of the stub end's height
TILT_MAX_DEG = 5.0  # its axis within 5 degrees of vertical, face up
TWO_HOLE_TOL_DEG = {"flange_4": 8.0, "flange_8": 7.0}  # hole pattern within this of straddling (45 +- 8, 22.5 +- 7)
SEATED_CREDIT = 0.25  # per seated flange
TWO_HOLED_CREDIT = 0.25  # per seated flange that is also two-holed
STILL_WINDOW_S = 1.0  # 'still': over the last 1 s neither its axis point nor its hole-1 point moved more than ...
STILL_TOL = 0.002  # ... 2 mm (positions, not the reported velocity, which chatters while a part is gripped)
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a flange moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)

# -- 'let go' (rigstate reports each gripper's pad-tip point and opening, not its yaw) -------------------------------
# With the fingers FULL_GAP x opening apart, a finger (FINGER_THICK thick outside the gap and 2 x FINGER_HALF_WIDTH
# wide across it, as measured on yam_arena's finger collision meshes) can touch a flange only if the flange's axis is
# within release_reach() of the pad-tip point horizontally. A gripper whose pad-tip point is lower than RELEASE_ABOVE
# over the flange's face while within that reach may be holding, pressing or propping it: the flange is not seated.
FULL_GAP = 0.0939  # the pads' gap fully open (finger joints at -0.04695)
FINGER_THICK = 0.010
FINGER_HALF_WIDTH = 0.020
RELEASE_MARGIN = 0.005
RELEASE_ABOVE = 0.010


def release_reach(opening: float, disc_r: float) -> float:
    """The farthest a gripper's fingers reach horizontally from its pad-tip point, over every yaw, plus the flange's
    radius and the margin (m). ``opening`` is rigstate's 0 (closed) .. 1 (fully open)."""
    half_gap = FULL_GAP * min(max(float(opening), 0.0), 1.0) / 2
    return math.hypot(half_gap + FINGER_THICK, FINGER_HALF_WIDTH) + disc_r + RELEASE_MARGIN


# -- colours (sRGB) --------------------------------------------------------------------------------------------------
STUB_BLACK = (0.12, 0.12, 0.13)  # black steel pipe
LINE_WHITE = (0.96, 0.96, 0.94)
FLANGE_GREY = (0.66, 0.67, 0.69)  # steel-grey paint
HOLE_BLACK = (0.03, 0.03, 0.03)
BORE_DARK = (0.20, 0.20, 0.22)
