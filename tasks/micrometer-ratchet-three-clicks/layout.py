"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up; the arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W, and
heights named h_* are above the work plate's top (the anvil's face), PL above the table.

A bench micrometer stand (a steel base with a lapped work plate flush with the anvil's face, two guide rails, a rear
post with a clamp) holds an outside micrometer upright, anvil down, its C-frame open toward the robot. The spindle
(0.5 mm per thimble turn) starts open to 20.30 mm. A 20.000 mm gauge block stands on the plate in the rails' channel,
55 mm in front of the faces. The job: slide the block between the faces, run the spindle down, finish on the ratchet
knob until it clicks, lock the lever.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)

# -- where the micrometer stands --------------------------------------------------------------------------------------
X_A = 0.240  # the spindle's axis (x)
Y_A = 0.000  # and (y): on the midline, where both arms reach
PL = 0.030  # the work plate's top, and the anvil's face: 30 mm above the table (the base is 30 mm tall)

# -- the measuring screw ----------------------------------------------------------------------------------------------
PITCH = 0.0005  # 0.5 mm of spindle travel per thimble turn (a 0 to 25 mm outside micrometer with 0.01 mm graduation)
G0 = 0.0203  # the gap between the faces as found: 20.30 mm
T_BLOCK = 0.0200  # the gauge block's thickness between its measuring faces
ADV_C = G0 - T_BLOCK  # 0.30 mm of spindle travel from as found to touching the block
THETA_C = 2 * math.pi * ADV_C / PITCH  # 3.7699 rad = 216 degrees of thimble turn from as found to touching the block
RAD_PER_M = 2 * math.pi / PITCH
FACE_R = 0.003175  # both measuring faces are 6.35 mm in diameter (carbide)
TURNS_MAX = 2 * math.pi - 0.001  # a hinge's travel is at most one turn (mechkit)

# -- the stand and the frame (fixed): boxes and cylinders as (centre x, y, height-above-table) ------------------------
BASE_X0, BASE_X1 = -0.090, 0.095  # 185 mm long (x, relative to the axis), 90 mm across (y), 30 mm tall
BASE_HY = 0.045
POST_X0, POST_X1, POST_HY, POST_H = -0.0705, -0.0505, 0.018, 0.045  # rear post: 20 x 36 mm, 45 mm over the plate
CLAMP_X0, CLAMP_X1, CLAMP_HY = -0.0505, -0.0405, 0.016  # clamp jaw pressed on the frame's column
CLAMP_Z0, CLAMP_Z1 = 0.006, 0.026
SCREW_R, SCREW_L, SCREW_X = 0.007, 0.014, -0.0775  # the clamp screw's knob behind the post (axis x)
COL_X0, COL_X1, COL_HY = -0.0405, -0.0205, 0.013  # the frame's column (the C's back): 20 x 26 mm
FRAME_H = 0.029  # the frame's upper arm ends 29 mm over the plate
ARM_Z0 = 0.0233  # the upper arm's underside (3 mm over the spindle's face as found)
ARM_X1, ARM_HY, BOSS_R = 0.0, 0.010, 0.0085
SLEEVE_R, SLEEVE_TOP = 0.0045, 0.0545  # the barrel (fixed sleeve) up to under the ratchet
PAD_X0, PAD_X1, PAD_T = -0.0385, -0.0225, 0.002  # the frame's black insulating pads on both flanks of the column
RAIL_X0, RAIL_X1 = COL_X1, BASE_X1  # the guide rails run from the column's face to the plate's front edge
RAIL_H, RAIL_W, RAIL_GAP = 0.0025, 0.004, 0.0105  # 2.5 mm tall, 4 mm wide, channel 10.5 mm between them
THROAT = -COL_X1  # 20.5 mm: the column's front face is behind the axis by this much

# -- the moving controls (links) --------------------------------------------------------------------------------------
SP_MASS = 0.010
SP_K, SP_D = 5000.0, 10.0  # the drive spring N/m and damping N s/m that carry the spindle to the screw's advance
SP_SAG = SP_MASS * 9.81 / SP_K  # the drive's sag under the spindle's own weight (19.6 um), taken out of the target
SP_LO, SP_HI = -0.0003, 0.0006
THIM_R, THIM_Z0, THIM_Z1 = 0.010, PL + 0.0350, PL + 0.0545  # thimble: 20 mm across, 19.5 mm long
THIM_MASS, THIM_K, THIM_D = 0.03, 1.0, 0.02
THIM_LO, THIM_HI = -0.9, TURNS_MAX
RAT_R, RAT_Z0, RAT_Z1 = 0.009, PL + 0.0548, PL + 0.0668  # ratchet knob: 18 mm across, 12 mm long
RAT_MASS, RAT_K, RAT_D = 0.015, 0.05, 0.002  # a light spring: the knob slips freely; the thimble follows by rule
RAT_LO, RAT_HI = -0.3, TURNS_MAX
LEV_Z0, LEV_Z1 = PL + 0.0295, PL + 0.0345  # the lock lever's slab: 5 mm thick
LEV_L, LEV_W = 0.030, 0.008  # 30 x 8 mm, pointing +y as found
COLLAR_R, COLLAR_Z0, COLLAR_Z1 = 0.0095, PL + 0.029, PL + 0.0345
LEV_Q_LOCK = math.pi / 2  # flipped 90 degrees clockwise (seen from above): pointing toward the robot (+x)
LEV_LO, LEV_HI = -0.1, LEV_Q_LOCK + 0.06
LEV_MASS, LEV_K, LEV_D = 0.01, 0.03, 0.005
LEV_SNAP_ON, LEV_SNAP_OFF, LEV_PRELOAD = 0.60, 0.25, 0.04  # the cam's two-notch rule (a detent), radians
LOCK_Q = 1.40  # "locked": the lever turned 80 degrees or more
LEV_TIP_FORCE = 0.5  # N at the tip to flip it (est): K x (0.25 + 0.04) / 0.030 = 0.29 N at the snap-off, 0.5 N peak

# -- the gauge block (loose) ------------------------------------------------------------------------------------------
# Grade 0 steel gauge block, 20 mm, the rectangular 35 x 9 mm cross-section of the standard series. Steel, 7850 kg/m3.
BL_X, BL_Y, BL_Z = 0.035, 0.009, T_BLOCK  # long (x), across (y), between the measuring faces (z, up)
BLOCK_MASS = round(7850.0 * BL_X * BL_Y * BL_Z, 4)  # 49.5 g
BL_START_DX = 0.055  # as found, its centre this far in front of the axis
BL_X0 = X_A + BL_START_DX
BL_Z0 = PL + BL_Z / 2  # standing on its 35 x 9 face on the plate
FRICTION = (0.20, 0.15)  # steel on steel, lightly oiled (est), the block, plate, rails and frame
BLOCK_STEEL = (0.74, 0.76, 0.80)

# -- colours ----------------------------------------------------------------------------------------------------------
STAND_GREY = (0.36, 0.38, 0.42)
FRAME_BLUE = (0.16, 0.30, 0.52)
CARBIDE = (0.10, 0.10, 0.12)
BLACK = (0.05, 0.05, 0.05)
SILVER = (0.78, 0.79, 0.82)
KNOB_BLACK = (0.12, 0.12, 0.14)
LEVER_RED = (0.80, 0.12, 0.10)
MARK_WHITE = (0.95, 0.95, 0.92)

# -- rule thresholds (scoring.py) -------------------------------------------------------------------------------------
GAP_TOL = 0.00003  # 1.0 and 0.5: the faces within 0.03 mm of 20.000
GAP_NEAR = 0.00020  # 0.25: within 0.20 mm
CONTACT_TOL = 0.00002  # the faces count as touching the block when 0.02 mm or less from it
CLICK_DEG = 15.0  # one click per 15 degrees of ratchet slip after the faces touch
CLICKS_FULL = 3
CLICK_SEAT_DEG = 2.0  # clicks count only with the thimble within 2 degrees of its touch turn (the faces seated)
CLICK_LEVER_MAX = LEV_SNAP_ON  # and the lock lever off: under 34 degrees (a locked spindle's ratchet slips untouched)
SLIP_EPS = 0.01  # the rule's 'faces touching': the thimble within 0.6 degree of the touch turn
SQUEEZE_DEG = 10.0  # the thimble turned this far past touching: about 14 N at 1 N per um of closing (est)
BETWEEN_DX = 0.0143  # the block's centre within 14.3 mm of the axis along x: each 6.35 mm face wholly on its 35 mm side
BETWEEN_DY = 0.0013  # within 1.3 mm across: each face wholly on its 9 mm side
BETWEEN_DZ = 0.0015  # standing on the plate (centre 10 mm up) within 1.5 mm
BETWEEN_TILT_DEG = 5.0  # its measuring faces within 5 degrees of square to the spindle
FELL_DZ = -0.005  # latched: the block's centre more than 5 mm below the plate's top (off the plate)
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.0005  # the block moves under 0.5 mm over the last second
STILL_TURN_DEG = 1.0  # and turns under 1 degree; the thimble, ratchet and lever turn under 1 degree
STILL_SPINDLE = 0.000005  # the spindle moves under 5 um
LET_GO = 0.030  # both grippers' finger zones this far from the block and the controls
CONTROLS = ("spindle", "thimble", "ratchet", "lever")
CONTROL_HZ = 20.0
RESET_JUMP = 0.02
START_TOL = 0.003
EPS = 1e-9

SPINDLE, THIMBLE, RATCHET, LEVER = CONTROLS


def adv_of(theta: float) -> float:
    """The spindle's travel (m, + toward the anvil) for a thimble turned ``theta`` rad from as found."""
    return PITCH * theta / (2 * math.pi)


def gap_of(q: float) -> float:
    """The faces' gap (m) with the spindle ``q`` m down from as found."""
    return G0 - q


def theta_of_gap(gap: float) -> float:
    return RAD_PER_M * (G0 - gap)


def _box(mk, cx, cy, cz, sx, sy, sz, color):
    return mk.box((cx, cy, cz), (sx, sy, sz), color)


def base_shapes(mk):
    """The fixed shapes in the micrometer's frame (origin on the axis at the table top): the stand, the frame, the
    sleeve."""
    s = []
    s.append(_box(mk, (BASE_X0 + BASE_X1) / 2, 0.0, PL / 2, BASE_X1 - BASE_X0, 2 * BASE_HY, PL, STAND_GREY))
    s.append(mk.cylinder((0.0, 0.0, PL - 0.003), FACE_R, 0.006, "Z", CARBIDE))  # the anvil's insert, flush
    for sy in (-1, 1):  # the guide rails
        s.append(_box(mk, (RAIL_X0 + RAIL_X1) / 2, sy * (RAIL_GAP / 2 + RAIL_W / 2), PL + RAIL_H / 2,
                      RAIL_X1 - RAIL_X0, RAIL_W, RAIL_H, SILVER))
    s.append(_box(mk, (POST_X0 + POST_X1) / 2, 0.0, PL + POST_H / 2, POST_X1 - POST_X0, 2 * POST_HY, POST_H,
                  STAND_GREY))
    s.append(_box(mk, (CLAMP_X0 + CLAMP_X1) / 2, 0.0, PL + (CLAMP_Z0 + CLAMP_Z1) / 2, CLAMP_X1 - CLAMP_X0,
                  2 * CLAMP_HY, CLAMP_Z1 - CLAMP_Z0, STAND_GREY))
    s.append(mk.cylinder((SCREW_X, 0.0, PL + 0.016), SCREW_R, SCREW_L, "X", KNOB_BLACK))
    s.append(_box(mk, (COL_X0 + COL_X1) / 2, 0.0, PL + FRAME_H / 2, COL_X1 - COL_X0, 2 * COL_HY, FRAME_H, FRAME_BLUE))
    for sy in (-1, 1):  # the insulating pads
        s.append(_box(mk, (PAD_X0 + PAD_X1) / 2, sy * (COL_HY + PAD_T / 2), PL + 0.0135, PAD_X1 - PAD_X0, PAD_T,
                      0.021, BLACK))
    s.append(_box(mk, (COL_X0 + ARM_X1) / 2, 0.0, PL + (ARM_Z0 + FRAME_H) / 2, ARM_X1 - COL_X0, 2 * ARM_HY,
                  FRAME_H - ARM_Z0, FRAME_BLUE))
    s.append(mk.cylinder((0.0, 0.0, PL + (ARM_Z0 + FRAME_H) / 2), BOSS_R, FRAME_H - ARM_Z0, "Z", FRAME_BLUE))
    s.append(mk.cylinder((0.0, 0.0, PL + (FRAME_H + SLEEVE_TOP) / 2), SLEEVE_R, SLEEVE_TOP - FRAME_H, "Z", SILVER))
    return s


def spindle_shapes(mk):
    """The spindle bar: its carbide face as found 20.30 mm over the plate, up into the frame's boss."""
    z0, z1 = PL + G0, PL + FRAME_H
    return [mk.cylinder((0.0, 0.0, (z0 + z1) / 2), FACE_R, z1 - z0, "Z", CARBIDE)]


def thimble_shapes(mk):
    zc, h = (THIM_Z0 + THIM_Z1) / 2, THIM_Z1 - THIM_Z0
    return [mk.cylinder((0.0, 0.0, zc), THIM_R, h, "Z", SILVER),
            _box(mk, THIM_R + 0.0005, 0.0, zc, 0.001, 0.0016, h, BLACK)]  # an index line on the rim


def ratchet_shapes(mk):
    zc, h = (RAT_Z0 + RAT_Z1) / 2, RAT_Z1 - RAT_Z0
    return [mk.cylinder((0.0, 0.0, zc), RAT_R, h, "Z", KNOB_BLACK),
            _box(mk, RAT_R + 0.0005, 0.0, zc, 0.001, 0.0016, h, MARK_WHITE)]


def lever_shapes(mk):
    zc = (LEV_Z0 + LEV_Z1) / 2
    return [mk.cylinder((0.0, 0.0, (COLLAR_Z0 + COLLAR_Z1) / 2), COLLAR_R, COLLAR_Z1 - COLLAR_Z0, "Z", STAND_GREY),
            _box(mk, 0.0, LEV_L / 2, zc, LEV_W, LEV_L, LEV_Z1 - LEV_Z0, LEVER_RED)]


def block_pose_rel(u_x: float = BL_START_DX, y: float = 0.0):
    """The block's centre relative to the axis and the plate's top: (dx, dy, dz)."""
    return (u_x, y, BL_Z / 2)
