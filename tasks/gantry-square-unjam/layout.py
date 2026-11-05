"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

import math

TOP = 0.7475  # station table top

# Two guide rails: fixed steel bars along x, one in front of each arm, their inner faces 521 mm apart.
RAIL_X0, RAIL_X1 = 0.170, 0.500  # the rails run from x 0.170 (near the arms) to x 0.500
RAIL_LEN = RAIL_X1 - RAIL_X0  # 0.330
RAIL_X = (RAIL_X0 + RAIL_X1) / 2
RAIL_W, RAIL_H = 0.012, 0.040  # across (y) and tall
RAIL_INNER_Y = 0.2605  # inner faces at y +-0.2605: a 521 mm gap
RAIL_Y = RAIL_INNER_Y + RAIL_W / 2  # rail centres at y +-0.2665

# Retaining lips: fixed strips along the rails' inner faces at the top, overhanging the beam's ends by 10 mm
# and 4 mm above its top, so the beam cannot be lifted out from between the rails.
LIP_W, LIP_H = 0.010, 0.006
LIP_Y = RAIL_INNER_Y - LIP_W / 2  # y +-0.2555: the lips cover y 0.2505 .. 0.2605
LIP_Z0 = 0.034  # lips' undersides above the table (the beam is 30 mm tall: 4 mm below them)

# The gantry beam: a loose hollow aluminium extrusion, 40 mm deep (x), 520 mm long (y), 30 mm tall, lying on the
# table between the rails at the back, out of square: its left end is 8 mm nearer the arms than its right end.
BEAM_SIZE = (0.040, 0.520, 0.030)
BEAM_HALF = tuple(s / 2 for s in BEAM_SIZE)
BEAM_MASS = 0.25  # a 40 x 30 mm hollow extrusion, about 0.5 kg per metre (a solid bar would weigh 1.7 kg)
BEAM_X0 = 0.444  # starting centre, on the midline
BEAM_LEN = BEAM_SIZE[1]
END_Y = BEAM_LEN / 2  # end centres 260 mm either side of the beam's centre, along its length
START_LEAD = 0.008  # the right end starts 8 mm farther from the arms than the left end
BEAM_YAW0 = math.asin(START_LEAD / BEAM_LEN)  # +0.0154 rad (+0.88 deg) about the vertical
# With the beam square the ends have 0.5 mm of play to each rail; yawed it touches both rails at diagonal corners
# once 260 cos(yaw) + 20 sin(yaw) mm reaches 260.5 mm: at 1.8 deg, a lead of 16.3 mm. At the start (0.88 deg) the
# ends are 0.22 mm from the rails.

# Yellow target marks: thin fixed tabs on the table just outside the rails, level with where the ends must stop.
TARGET_X = 0.220
MARK_SIZE = (0.030, 0.020, 0.001)
MARK_Y = 0.285  # y +-0.285: 2.5 mm outside the rails' outer faces (y +-0.2725)

# Rule thresholds (scoring.py).
LEAD_MAX = 0.012  # one end leading the other by more than this at any step: racked, 0
SQUARE_LEAD = 0.002  # 'squared': the first step with the lead at or under this
SQUARE_BY = 0.025  # 'squared first' (1.0 and 0.75): squared before the centre had moved this far toward the marks
POP_RISE = 0.005  # the beam's lowest point this far above the table: out from under the lips (they allow 4 mm)
POP_TILT_DEG = 5.0  # ... or tilted more than this: popped up
RAIL_SIDE_TOL = 0.010  # the beam's centre this far off the midline sideways: out of the rails
# Both end centres within this of x 0.220 (1.0). The arms trail a pull: the compliance model (J K^-1 J^T, joints
# 80/80/80/10/10/10 N m/rad) puts each gripper about 9 mm behind its command under its half of the beam's sliding
# friction (1.2 N), so an open-loop policy cannot place the ends to 1-2 mm; 5 mm keeps 'level with the marks'
# (30 mm long) tight while the score rests on the lead.
END_TIGHT = 0.005
END_LOOSE = 0.010  # (0.75)
CENTRE_LOOSE = 0.015  # the beam's centre within this of x 0.220 (0.5)
FINAL_LEAD = 0.002  # 1.0: the lead at the end at most this
AFTER_SQUARE_TIGHT = 0.004  # 1.0: the largest lead after squaring
AFTER_SQUARE_LOOSE = 0.007  # 0.75
TRAVEL_SPAN = BEAM_X0 - TARGET_X  # 0.224: the whole move
TRAVEL_PART = 0.30  # 0.25: squared and moved at least this share of the way
HANDS_OFF = 0.030  # 1.0: each gripper's finger zone at least this far from the beam
# 'By both ends together': a gripper is at an end when its finger-tip point is within AT_BEAM of the beam and at
# least END_ZONE from the beam's middle along its length, toward that end. The beam's centre may come at most
# UNHELD_MAX (net, toward the marks) closer while there is not a gripper at each end; more is 0. Tilted 10 deg,
# either arm reaches the beam's middle, where a pull turns nothing: this keeps the move the two-handed one asked for.
AT_BEAM = 0.010
END_ZONE = 0.130  # half of the beam's half-length (260 mm): the outer half of each half
UNHELD_MAX = 0.020
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # the beam moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... onto its starting place: within this of it (covers a seeded reset's +-3 cm jitter)

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.008  # finger pad thickness, outside the gap
FINGER_HALF_WIDTH = 0.0135  # the pads are about 27 mm wide (across the closing axis) at the TCP's level
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper

# Colours (sRGB).
STEEL = (0.30, 0.30, 0.32)
ALUMINIUM = (0.75, 0.76, 0.78)
YELLOW = (0.95, 0.80, 0.10)
