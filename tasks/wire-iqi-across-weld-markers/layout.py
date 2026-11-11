"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A butt-welded steel plate lies flat on the table on the midline, set up for a radiographic test shot: the film goes
under the plate, so its top face is the source side. The weld runs along y across the plate's middle (centreline
x 0.360) and two yellow ticks across it at y -0.080 and +0.080 mark the ends of the length this shot covers. ASME
Section V, Article 2: the wire IQI goes on the source side, on the weld, with the length of its wires perpendicular
to the length of the weld (T-277.1), and a location marker goes on the part's source side at each end of the area
covered (T-275, T-275.1(a)(1) for flat components). The weld runs along y, so the wires must run along x.
"""

import math

TOP = 0.7475  # station table top (world z)

# -- the welded plate (fixed): two steel plates and the weld between them, all 10 mm thick, their tops flush --------
PLATE_X0, PLATE_X1 = 0.280, 0.440  # 160 mm along x
PLATE_HALF_Y = 0.100  # 200 mm along y
PLATE_T = 0.010
PLATE_TOP_W = TOP + PLATE_T
WELD_X = 0.360  # the weld's centreline (along y)
WELD_W = 0.014  # the ground-flush weld cap, 14 mm wide
WELD_X0, WELD_X1 = WELD_X - WELD_W / 2, WELD_X + WELD_W / 2  # 0.353 .. 0.367

# -- the shot-length ticks (fixed): yellow painted strips across the weld, 0.5 mm proud -----------------------------
TICK_Y = (-0.080, 0.080)  # their centre lines: the ends of the length this shot covers (160 mm)
TICK_LEN = 0.024  # along x, centred on the weld, 5 mm past each edge of the cap: x 0.348 .. 0.372
TICK_W = 0.004  # along y (4 mm, not 2, so the 960 x 528 setup screenshot shows them; measured to their centre)
TICK_T = 0.0005

# -- the wire IQI: one loose rigid body (mechkit compound). Base frame: origin at the middle of its bottom face, local
#    +x along its wires (its 55 mm length), local +z up through the face the wires are painted on --------------------
IQI_L, IQI_W, IQI_H = 0.055, 0.028, 0.012  # a clear blue plastic holder with the wire pack sealed in it
IQI_MASS = 0.015
WIRE_LEN = 0.050  # six dark wires along local x, painted on the top face, of increasing width
WIRE_WIDTHS = (0.0010, 0.0013, 0.0016, 0.0020, 0.0025, 0.0030)
WIRE_Y = (-0.010, -0.006, -0.002, 0.002, 0.006, 0.010)  # their centre lines across the holder (4 mm pitch)
WIRE_T = 0.0005
IQI_LO = (-IQI_L / 2, -IQI_W / 2, 0.0)  # its bounding box in its own frame (holder and wires)
IQI_HI = (IQI_L / 2, IQI_W / 2, IQI_H + WIRE_T)
IQI_START = (0.230, 0.000, 90.0)  # (x, y, yaw deg): on the bench in front of the plate, wires along y (the weld)

# -- the two location markers: loose rigid bodies (mechkit compounds), a lead-figure holder with a white figure on top
MK_SIDE, MK_H = 0.022, 0.012
MK_MASS = 0.015
FIG_T = 0.0005  # the white figures, 0.5 mm, on the top face; the top of a figure is toward local +x
FIG_STROKE = 0.0025
FIG_H, FIG_W = 0.014, 0.010  # the "0": a ring 14 mm tall (local x) and 10 mm wide (local y)
MK_LO = (-MK_SIDE / 2, -MK_SIDE / 2, 0.0)
MK_HI = (MK_SIDE / 2, MK_SIDE / 2, MK_H + FIG_T)
MARKERS = ("marker_0", "marker_1")
MK_START = {  # (x, y, yaw deg): on the bench, one each side of the plate, beside the arm that fetches it
    "marker_0": (0.240, -0.160, 0.0),
    "marker_1": (0.240, 0.160, 0.0),
}
NAMES = ("iqi", *MARKERS)

# -- rule thresholds (scoring.py) ----------------------------------------------------------------------------------
FLAT_DEG = 5.0  # 'flat': its local z within 5 degrees of vertical (either face up)
SEAT_TOL = 0.003  # 'on the plate's top': its lowest point within 3 mm of the plate's top (not on another part)
WIRE_TOL_DEG = 8.0  # IQI: its wires (local x) within 8 degrees of x, i.e. of perpendicular to the weld
ON_WELD = 0.015  # IQI: its centre within 15 mm of the weld centreline (x): lying on the weld, every one of its 50 mm
#                  wires crosses the whole 14 mm cap (1.3 mm to spare past the far edge even at 15 mm off and 8 deg)
SHOT_HALF = 0.080  # IQI: every corner of its footprint between the ticks' centre lines (|y| <= 0.080)
MARK_ALONG = 0.010  # marker: its centre within 10 mm of a tick's centre line, along the weld (y) ...
MARK_ACROSS = 0.035  # ... and within 35 mm of the weld centreline (x): on or beside the weld
HANDS_OFF = 0.010  # 'released': every readable gripper's finger zone at least 10 mm from the part ...
STILL_WINDOW_S = 0.5  # ... over the last 0.5 s, in which the part's centre moved under 2 mm in all and its yaw
STILL_PATH = 0.002  # under 1 degree ('settled'; counted = released and settled). 'Still for 1 s': both of the
STILL_YAW_DEG = 1.0  # last two 0.5 s windows settled, released throughout.
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # a part landing on its starting mark having moved this much further in one control step than
#                    its last velocity and gravity could carry it was teleported: a new episode began
START_TOL = 0.045  # ... 'on its starting mark': within this of it (covers a seeded reset's +-3 cm jitter)

# -- the grippers (the released rule; rigstate reports the TCP, the point between the pad tips and the opening) ------
FULL_GAP = 0.0939  # the pads' gap fully open
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the fingers are about 27 mm wide across the closing axis
FINGER_LEN = 0.038  # the finger zone runs from the pad tips this far back up the gripper

# -- where the demo puts them (scoring windows in brackets) ---------------------------------------------------------
IQI_TARGET = (WELD_X, 0.000)  # centre on the weld centreline (+-15 mm), mid-shot (footprint |y| <= 14 mm of 80)
IQI_TARGET_YAW = 0.0  # wires along x (+-8 degrees)
MK_TARGET = {  # on the arms' side of the weld, 28 mm from its centreline (+-35 mm), on the ticks' lines (+-10 mm);
    "marker_0": (0.332, -0.080),  # the footprint (x 0.321 .. 0.343) ends 5 mm short of the tick (x 0.348)
    "marker_1": (0.332, 0.080),
}
BASES = {"left": (0.0, 0.305), "right": (0.0, -0.305)}

# -- colours (sRGB) ------------------------------------------------------------------------------------------------
STEEL = (0.55, 0.56, 0.58)
WELD_DARK = (0.27, 0.27, 0.29)
TICK_YELLOW = (0.98, 0.84, 0.10)
IQI_BLUE = (0.62, 0.80, 0.95)  # a clear blue plastic holder
WIRE_DARK = (0.06, 0.06, 0.07)
LEAD_GREY = (0.40, 0.41, 0.43)
FIG_WHITE = (0.97, 0.97, 0.97)


def start_yaw(name: str) -> float:
    start = IQI_START if name == "iqi" else MK_START[name]
    return math.radians(start[2])


assert len(WIRE_WIDTHS) == len(WIRE_Y) == 6
assert max(abs(y) + w / 2 for y, w in zip(WIRE_Y, WIRE_WIDTHS)) < IQI_W / 2
assert WIRE_LEN < IQI_L
