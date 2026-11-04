"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The machine: a loose painted-steel box standing on the table on the midline, 30 cm in front of the arm bases.
MACHINE_SIZE = (0.070, 0.090, 0.060)  # along x, along y, height
MACHINE_MASS = 0.30
MACHINE_X, MACHINE_Y = 0.30, 0.0

# The skid: a loose hardwood plank lying flat on the table to the machine's right, its long side along y.
SKID_SIZE = (0.060, 0.440, 0.012)  # along x, along y, thickness
SKID_MASS = 0.12
SKID_X, SKID_Y = 0.30, -0.27  # it spans y -0.49 to -0.05: its left end stops 5 mm short of the machine
SKID_TOP = TOP + SKID_SIZE[2]

# Rule thresholds (scoring.py).
ON_SKID_Z_TOL = 0.002  # 'on the skid': the machine's underside (at its centre) within this of the skid's top,
ON_SKID_TILT_DEG = 3.0  # the machine tilted less than this,
ON_SKID_COVER = 0.75  # and at least this share of its footprint over the skid
UNDER_COVER = 0.50  # 'the skid lies under it': at least this share of the machine's footprint over the skid,
UNDER_Z_TOL = 0.003  # and the machine's underside (at its centre) not more than this below the skid's top
SKID_FLAT_Z_TOL = 0.003  # the skid 'lies flat on the table': its underside within this of the table top
SKID_FLAT_TILT_DEG = 3.0  # and tilted less than this
TIP_DEG = 30.0  # the machine tilted more than this at any moment: 0
OFF_TABLE_DROP = 0.010  # the machine's lowest point this far below the table top: it fell off the table
DRIFT_FULL, LIFT_FULL = 0.010, 0.030  # 1.0: the machine's centre never more than 10 mm from its start
#                                       horizontally, no part of its base ever more than 30 mm above the table
DRIFT_PART, LIFT_PART = 0.030, 0.050  # 0.75: raise within 50 mm (and drift within 30 mm)
DRIFT_MAX = DRIFT_PART  # drift beyond this at any moment (dragged out of place, or carried over to the skid): 0
STILL_SPEED, STILL_SPIN = 0.01, 0.5  # 'still': under 1 cm/s and 0.5 rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object that lands on its starting mark having moved this much further in one control step
#                    than its last velocity and gravity could carry it was teleported: a new episode began

# Colours (sRGB).
GREEN = (0.30, 0.45, 0.30)
WOOD = (0.72, 0.55, 0.35)
