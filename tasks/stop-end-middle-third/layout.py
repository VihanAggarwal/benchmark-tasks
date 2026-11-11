"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say
'world'.

A 1:15 model of one bay of beam formwork: two girders run along x, each an open trough between two plywood side
forms with a concrete column at each end (column centres 300 mm apart: a 4.5 m span). A cross beam's trough runs
along y between the two girders and frames into each 105 mm from the near column's centre. A loose plywood stop-end
(bulkhead) stands on the table beside each girder. Rule (ACI 318 construction joints): a stop-end goes across each
girder, wholly inside the middle third of its span and at least two times the cross beam's width clear of the cross
beam's face. Only the band x 0.295-0.340 meets both; the obvious mid-span spot (x 0.290) is inside the middle third
but 22.5 mm from the beam.
"""

import math

TOP = 0.7475  # station table top (world z)
SCALE = 15  # 1:15

# -- the columns (fixed, concrete) -------------------------------------------------------------------------------
COL_SIZE = (0.040, 0.040, 0.040)
COL_X = (0.140, 0.440)  # column centres along each girder: a 300 mm span (4.5 m)
GIRDER_Y = 0.200  # each girder's centre line is at y +-0.200
SPAN = COL_X[1] - COL_X[0]
COL_FACE_X = (COL_X[0] + COL_SIZE[0] / 2, COL_X[1] - COL_SIZE[0] / 2)  # 0.160 and 0.420: the trough's two ends

# -- the girder forms (fixed, plywood) -----------------------------------------------------------------------------
FORM_T = 0.010  # every side form's thickness
FORM_H = 0.025  # every side form's height
TROUGH_HALF = 0.025  # a girder trough is 50 mm wide inside (0.75 m)
TROUGH_IN = GIRDER_Y - TROUGH_HALF  # 0.175: the inside face of the inner form (toward the midline)
TROUGH_OUT = GIRDER_Y + TROUGH_HALF  # 0.225: the inside face of the outer form
FORM_X0, FORM_X1 = COL_FACE_X  # the side forms run between the columns' faces

# -- the cross beam (its two side forms; fixed, plywood) -----------------------------------------------------------
BEAM_W = 0.020  # the cross beam's width (0.30 m)
BEAM_NEAR_FACE = 0.235  # its side faces (the inside faces of its two forms), along x
BEAM_FAR_FACE = BEAM_NEAR_FACE + BEAM_W  # 0.255
BEAM_FORM_Y = TROUGH_IN - FORM_T  # 0.165: its forms run between the girders' inner forms, y -0.165 .. +0.165

# The fixed boxes: (name, centre x, centre y, size x, size y). Left girder first; the right girder is its mirror.
def _forms() -> list[tuple[str, float, float, float, float]]:
    out = []
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        out.append((f"form_{side}_outer", (FORM_X0 + FORM_X1) / 2, sgn * (TROUGH_OUT + FORM_T / 2),
                    FORM_X1 - FORM_X0, FORM_T))
        # the inner form is in two pieces: the cross beam's trough enters the girder between them
        out.append((f"form_{side}_inner_near", (FORM_X0 + BEAM_NEAR_FACE) / 2, sgn * (TROUGH_IN - FORM_T / 2),
                    BEAM_NEAR_FACE - FORM_X0, FORM_T))
        out.append((f"form_{side}_inner_far", (BEAM_FAR_FACE + FORM_X1) / 2, sgn * (TROUGH_IN - FORM_T / 2),
                    FORM_X1 - BEAM_FAR_FACE, FORM_T))
    out.append(("beam_form_near", BEAM_NEAR_FACE - FORM_T / 2, 0.0, FORM_T, 2 * BEAM_FORM_Y))
    out.append(("beam_form_far", BEAM_FAR_FACE + FORM_T / 2, 0.0, FORM_T, 2 * BEAM_FORM_Y))
    return out


FORMS = _forms()
COLUMNS = [(f"column_{end}_{side}", x, sgn * GIRDER_Y)
           for end, x in (("near", COL_X[0]), ("far", COL_X[1])) for side, sgn in (("left", 1.0), ("right", -1.0))]

# -- the stop-ends (loose, plywood, faces painted yellow) ---------------------------------------------------------
# Size along its own axes: x its thickness, y its length (across the trough when placed), z its height.
STOP_SIZE = (0.025, 0.040, 0.060)
STOP_HALF = tuple(v / 2 for v in STOP_SIZE)
STOP_MASS = 0.030  # 60 cm3 of plywood at 0.5 g/cm3
STOP_NAMES = ("stop_end_left", "stop_end_right")
START_YAW_DEG = 30.0  # each starts with its length turned 30 degrees from y (left +30, right -30)
STOP_START = {  # (x, y, yaw in rad): standing upright on the table beside each girder
    "stop_end_left": (0.200, 0.320, math.radians(START_YAW_DEG)),
    "stop_end_right": (0.200, -0.320, -math.radians(START_YAW_DEG)),
}

# -- the rules (scoring.py) ----------------------------------------------------------------------------------------
MID_LO = COL_X[0] + SPAN / 3  # 0.240: the middle third of the span
MID_HI = COL_X[0] + 2 * SPAN / 3  # 0.340
OFFSET = 2 * BEAM_W  # 0.040: a joint in a girder at least twice the cross beam's width from its face
BAND_LO = BEAM_FAR_FACE + OFFSET  # 0.295: the band that meets both rules is x 0.295 .. 0.340
UPRIGHT_DEG = 10.0  # standing: its 60 mm side within this of vertical (either end down)
ON_FLOOR = 0.005  # standing: its lowest point at most this above the trough floor (the table)
SQUARE_DEG = 10.0  # standing: its 40 mm length, seen from above, within this of square across the trough (y)
TROUGH_TOL = 0.002  # in the trough: its outline may reach this far past a face (contact)
HANDS_OFF = 0.020  # standing: both grippers' finger zones at least this far from it (let go)
FULL, PART = 0.5, 0.25  # per girder: in the band / in the middle third but too near the cross beam
EPS = 1e-6  # m: limits are compared with a micrometre of slack
STILL_TOL = 0.002  # success: neither end of either stop-end moved more than this ...
STILL_WINDOW_S = 0.5  # ... over the last 0.5 s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a stop-end moving this far in one control step was teleported: a new trial began ...
START_TOL = 0.045  # ... if both are then within this of their starts (covers a seeded reset's jitter)

# -- the grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open) --
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap over its lowest 38 mm
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there
FINGER_LEN = 0.038  # the finger zone runs from the tips this far back up the gripper

# Colours (sRGB).
CONCRETE = (0.62, 0.62, 0.60)
FORM_PLY = (0.36, 0.17, 0.10)  # film-faced formwork plywood, dark red-brown: unlike the oak-coloured table
STOP_YELLOW = (0.95, 0.80, 0.10)
