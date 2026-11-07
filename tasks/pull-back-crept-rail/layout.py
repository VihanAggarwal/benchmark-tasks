"""Every dimension, position, mass, colour and threshold of the task, in one place (metres, kilograms, radians
unless a name says otherwise; the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left of their midpoint, z up. The arm
bases stand at x 0, y +-0.305; the station's table top is at z 0.7475. Heights in comments are above the table top.
Every box is square to the table (yaw 0).
"""

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# (1) Sleepers: four fixed dark-brown hardwood bars across the track (along x), carrying both rails.
SLEEPER_SIZE = (0.260, 0.025, 0.018)
SLEEPER_X = 0.290  # they span x 0.160 .. 0.420
SLEEPER_YS = (0.250, 0.080, -0.080, -0.250)  # outer pair under the fixed rails, middle pair under the middle length
SLEEPER_TOP = SLEEPER_SIZE[2]  # 18 mm up
SLEEPER_NEAR_X = SLEEPER_X - SLEEPER_SIZE[0] / 2  # 0.160: the sleepers' near ends

# (2) Rails: 20 mm wide (x), 25 mm tall bars standing on the sleepers.
RAIL_W, RAIL_H = 0.020, 0.025
RAIL_Z = SLEEPER_TOP + RAIL_H / 2  # 0.0305: centre height of every rail
RAIL_X = 0.220  # the near rail (the one the arms work on)
FAR_RAIL_X = 0.360  # the far rail, set correctly, for reference

# The near rail's two fixed lengths, their inner end faces 330 mm apart.
FIX_END_Y = 0.165  # inner end faces at y +0.165 and -0.165
FIXED_LEN = 0.235  # each spans |y| 0.165 .. 0.400
FIXED_Y = FIX_END_Y + FIXED_LEN / 2  # 0.2825

# (3) The middle (closure) length: a loose aluminium box section (4 mm walls) between the fixed lengths.
MID_LEN = 0.310
HALF_LEN = MID_LEN / 2  # 0.155
MID_SIZE = (RAIL_W, MID_LEN, RAIL_H)
MID_MASS = 0.25  # (20 x 25 - 12 x 17 mm) x 310 mm of aluminium at 2.7 g/cm3: 0.248 kg
MID_X0, MID_Y0 = RAIL_X, 0.010  # crept 10 mm to the left: ends at y +0.165 (on the fixed end) and -0.145
GAP_TOTAL = 2 * FIX_END_Y - MID_LEN  # 0.020: the two joints always add up to 20 mm while it is square
GAP_TARGET = GAP_TOTAL / 2  # 0.010: the expansion gap asked for at each joint
START_LEFT_GAP = FIX_END_Y - (MID_Y0 + HALF_LEN)  # 0.000: the left joint shut
START_RIGHT_GAP = (MID_Y0 - HALF_LEN) + FIX_END_Y  # 0.020: the right joint wide open (5 mm each side of the shim)

# (4) Rail seat shoulders: fixed dark-grey painted aluminium blocks on the two middle sleepers, either side of the
# middle length.
SHOULDER_SIZE = (0.006, 0.025, 0.010)
SHOULDER_Z = SLEEPER_TOP + SHOULDER_SIZE[2] / 2  # 0.023
SEAT_PLAY = 0.005  # each side of the 20 mm rail: inner faces at x 0.205 and 0.235
SHOULDER_XS = (RAIL_X - RAIL_W / 2 - SEAT_PLAY - SHOULDER_SIZE[0] / 2,  # 0.202
               RAIL_X + RAIL_W / 2 + SEAT_PLAY + SHOULDER_SIZE[0] / 2)  # 0.238
SEAT_YS = (0.080, -0.080)  # the two rail seats (the middle sleepers' centre lines)

# (5) The far rail: fixed aluminium bars of the same section, two 235 mm lengths and a 310 mm middle length with
# correct 10 mm joints at |y| 0.160.
FAR_MID_LEN = MID_LEN

# (6) The expansion shim: a loose yellow-painted steel plate standing on its 60 x 10 mm edge in the cess, in front
# of the sleepers' near ends, its 60 mm side across the track (x), 10 mm along it (y), 45 mm tall.
SHIM_SIZE = (0.060, 0.010, 0.045)
SHIM_MASS = 0.21  # 27 cm3 of steel at 7.85 g/cm3: 0.212 kg
SHIM_X0, SHIM_Y0 = 0.120, -0.160  # it spans x 0.090 .. 0.150
SHIM_THICK = SHIM_SIZE[1]  # 0.010

# The near rail's fixed right length (the fixed rail end the shim is stood against), as a box: centre and size.
FIXED_RIGHT_POS = (RAIL_X, -FIXED_Y, TOP + RAIL_Z)
FIXED_RIGHT_SIZE = (RAIL_W, FIXED_LEN, RAIL_H)
RAIL_LINE_X = (RAIL_X - RAIL_W / 2, RAIL_X + RAIL_W / 2)  # 0.210 .. 0.230: the near rail's width

# Rule thresholds (scoring.py).
SEAT_TOL = 0.006  # 'seated': the middle length's centre line within this of x 0.220 at both seats
SEAT_Z_TOL = 0.002  # ... its lowest corner within this of the sleeper tops (16 .. 20 mm up)
TILT_MAX_DEG = 3.0  # ... its 25 mm (height) axis within this of vertical, either way up
SHIM_TABLE_TOL = 0.003  # 'put away': the shim's lowest corner within this of the table top ...
CESS_X = SLEEPER_NEAR_X  # ... and every corner at x below 0.160 (nearer the arms than the sleepers' near ends)
TRACK_X = (SLEEPER_NEAR_X, SLEEPER_X + SLEEPER_SIZE[0] / 2)  # 'on the track': a shim corner at x 0.160 .. 0.420,
TRACK_Y = 0.420  # ... |y| below this (the rails end at 0.400) ...
TRACK_Z = SLEEPER_TOP + RAIL_H + 0.002  # ... and lower than 0.045 above the table (2 mm above the rail tops)
GAUGE_TOUCH = 0.001  # 'gauged': the shim, standing in the right joint, within this of both rail ends' faces
HANDS_OFF = 0.060  # 'hands off': each gripper's pad-tip point at least this far from the middle length and the shim
STILL_STEPS = 10  # 'still' is judged over the last 0.5 s (10 control steps) ...
STILL_PATH = 0.002  # ... each part's centre moved along a path shorter than this
BAND1 = (0.0085, 0.0115)  # 1.0: both gaps 8.5 .. 11.5 mm
BAND2 = (0.006, 0.014)  # 0.5: both gaps 6 .. 14 mm
BAND3_MIN = 0.003  # 0.25: both gaps at least 3 mm
RESET_JUMP = 0.004  # a part moving this much further in one step (in plan) than its velocity allows was put back ...
START_TOL = 0.004  # ... onto its starting place: within this of it

# The grippers (yam_arena).
PAD_TIPS = 0.013  # the pad tips are 13 mm below the TCP
FULL_GAP = 0.0939  # pads apart fully open

# Colours (sRGB).
SLEEPER_BROWN = (0.33, 0.22, 0.13)
RAIL_GREY = (0.46, 0.47, 0.49)
SHOULDER_GREY = (0.20, 0.20, 0.22)
SHIM_YELLOW = (0.95, 0.80, 0.10)
