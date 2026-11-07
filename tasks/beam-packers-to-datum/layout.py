"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.
"""

TOP = 0.7475  # station table top

# Two fixed concrete piers (bearings) of unequal height, one in front of each arm, 360 mm apart centre to centre.
PIER_X = 0.290
PIER_Y = 0.180  # centres at y +0.180 (left) and -0.180 (right)
PIER_SIDE = 0.060  # 60 x 60 mm top
PIER_HALF = PIER_SIDE / 2
PIER_H = {"left": 0.020, "right": 0.030}  # tops 20 mm and 30 mm above the table
SIDES = ("left", "right")
SIGN = {"left": 1.0, "right": -1.0}

# The datum: the level both bearings must be brought up to, shown by a red-capped post behind each pier.
DATUM = 0.070  # 70 mm above the table
POST_X = 0.355  # 25 mm behind the pier's far face (x 0.320)
POST_Y = 0.230  # 10 mm outboard of the pier's outer face (y 0.210); inner faces 440 mm apart: the beam cannot span them
POST_SIDE = 0.020
CAP_H = 0.010  # red top band
POST_BODY_H = DATUM - CAP_H  # grey part, 60 mm

# Six loose beech packers, 50 x 50 mm, colour-coded by thickness; one set of three in front of each arm.
PACKER_SIDE = 0.050
PACKER_HALF = PACKER_SIDE / 2
PACKERS = {  # thickness (m), mass (kg: beech, 0.72 g/cm^3), colour name
    20: (0.020, 0.036, "yellow"),
    30: (0.030, 0.054, "blue"),
    40: (0.040, 0.072, "green"),
}
SET_Y = 0.300  # each set in a row along x at y +0.300 / -0.300, in front of its arm's base
SET_X = {20: 0.160, 30: 0.220, 40: 0.280}  # 10 mm gaps between packers
# The one correct packing: shortfalls 50 mm (left) and 40 mm (right); from {20, 30, 40} only 20 + 30 makes 50
# and only 40 (or 20 + 20 from both sets) makes 40.
SOLUTION = {"left": (20, 30), "right": (40,)}


def packer_name(t: int, side: str) -> str:
    return f"packer_{t}_{side}"


PACKER_NAMES = tuple(packer_name(t, s) for s in SIDES for t in PACKERS)

# The beam: a loose spruce beam lying flat on the table on the midline, its length along y.
BEAM_SIZE = (0.040, 0.420, 0.030)  # 40 mm wide (x), 420 mm long (y), 30 mm thick (z)
BEAM_LEN = BEAM_SIZE[1]
BEAM_MASS = 0.227  # spruce, 0.45 g/cm^3
BEAM_X, BEAM_Y = 0.150, 0.0
# Seated centred on the packing, the beam's ends (y +-0.210) lie over the outer edge of each top packer
# (y 0.155 .. 0.205): 50 mm of bearing at each end; it can sit up to 25 mm off-centre along y and keep 30 mm.

LOOSE = PACKER_NAMES + ("beam",)

# Rule thresholds (scoring.py).
ON_PIER_MARGIN = 0.015  # a packer is 'on' a pier: centre over the pier's top or within 15 mm of it ...
ON_PIER_MIN_LOW = 0.010  # ... and its lowest point more than 10 mm above the table
FLAT_DEG = 3.0  # 'lies flat': its thickness axis within 3 deg of vertical
EDGE_DEG = 45.0  # 'on edge or end': its thickness axis more than 45 deg from vertical (0 if on a pier)
OVERHANG = 0.005  # its outline seen from above within the pier's top extended by 5 mm on every side
REST_TOL = 0.002  # underside within 2 mm of the pier top / of the packer below it, measured at its centre
DATUM_TOL = 0.003  # a pier is packed when its packing's top is within 3 mm of the datum
BEAM_FACE_DEG = 5.0  # the beam lies on a long face: the axis through one pair of long faces within 5 deg of vertical
BEAM_LEVEL_DEG = 1.25  # level along its length; a beam resting across both piers out of level by more: 0
# Why 1.25: packers come in 10 mm steps, so a wrong pier is at least 10 mm off. A beam resting across a 10 mm
# step has its two contacts at most its 420 mm length apart: at least 1.36 deg (1.59 with everything centred).
# Two packed piers differ by at most 6 mm and the inner edge of a counted top packer is at least 145 mm from the
# midline, so the contacts are at least 290 mm apart: at most 1.19 deg. (At 1.5 deg a released beam across a
# 10 mm step could score 0 or 0.25 depending on where the packers sat on their piers.)
BEAM_REST_TOL = 0.002  # resting: its underside within 2 mm of a support's top where they meet
BEAM_SINK_TOL = 0.003  # (a support may appear up to 3 mm into the beam: contact penetration)
BEARING_MIN = 0.030  # at least 30 mm of the beam's length over each top packer, its centre line crossing it
# (a centre line crossing both 50 mm packers, 360 mm apart, is within 9.2 deg of y: no separate heading limit)
FACE_SAMPLES = 9  # support faces are sampled 9 x 9 (edges included) to find where the beam meets them
STILL_WINDOW_S = 0.5  # 'still': moved under 3 mm and turned under 2 deg in the last 0.5 s (positions, not the
STILL_MOVE = 0.003  # reported velocities, which chatter while an object is gripped)
STILL_TURN_DEG = 2.0
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # an object moving this far in one control step was teleported: a new episode began ...
START_TOL = 0.045  # ... if every loose object is then within this of its starting place (seeded jitter +-3 cm)

# Colours (sRGB).
CONCRETE = (0.62, 0.62, 0.60)
POST_GREY = (0.45, 0.45, 0.47)
CAP_RED = (0.80, 0.10, 0.08)
PACKER_COLOURS = {"yellow": (0.95, 0.80, 0.15), "blue": (0.15, 0.35, 0.80), "green": (0.20, 0.60, 0.25)}
SPRUCE = (0.85, 0.72, 0.52)
