"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475.
"""

TOP = 0.7475  # station table top

# The rule: ISO 13855's minimum distance from a light curtain's detection plane to the hazard, for a curtain of
# resolution d <= 40 mm: S = K T + C, K = 2000 mm/s (valid while S <= 500 mm), C = 8 (d - 14) mm.
K_MM_S = 2000.0  # approach speed of a hand (mm/s)
STOP_T = 0.06  # overall stopping time of the press plus the curtain (s)
RES_D = 20.0  # the curtain's resolution (detection capability, mm)
S_MM = K_MM_S * STOP_T + 8.0 * (RES_D - 14.0)  # 120 + 48 = 168 mm
S_MIN = S_MM / 1000.0  # 0.168: closer than this is unsafe (score 0)
S_SLACK = 0.020  # "or up to 20 mm farther"
S_MAX = S_MIN + S_SLACK  # 0.188

# The press (a mock): a fixed block beyond the opening, its front face at x 0.446.
PRESS_X0, PRESS_X1 = 0.446, 0.526
PRESS_HALF_Y = 0.150
PRESS_H = 0.140

# The opening in front of the press: a 2 mm floor plate between the two side guards, from x 0.200 to the press.
OPEN_X0 = 0.200
GUARD_INNER_Y = 0.140  # the guards' inner faces at y +-0.140: a 280 mm wide opening
GUARD_T = 0.010  # the guards are 10 mm thick: outer faces at y +-0.150
GUARD_OUTER_Y = GUARD_INNER_Y + GUARD_T
GUARD_H = 0.100
FLOOR_T = 0.002
FLOOR_TOP = TOP + FLOOR_T

# The die line: yellow tape on the floor plate along the press's front, marking the die's edge (the hazard).
# Distances are measured from its near edge (the edge toward the arms), at x 0.440.
DIE_X = 0.440
DIE_TAPE_W = 0.006  # x 0.440 .. 0.446
DIE_TAPE_T = 0.001

# The two light-curtain posts (emitter and receiver): loose plain boxes standing for the housings. Local axes:
# x through the two broad faces (the lens window is a broad face: both are alike, so either counts), y across
# the broad faces, z up the post's length.
POST_SIZE = (0.036, 0.056, 0.100)  # depth (lens normal), width, height
POST_HALF = tuple(s / 2 for s in POST_SIZE)
POST_MASS = 0.15  # a hollow aluminium extrusion with end caps and optics
MAT_SIZE = (0.080, 0.080, 0.003)  # a blue rubber staging mat under each post
MAT_X, MAT_Y = 0.160, 0.220  # mats at (0.160, +-0.220), outside the guards and in front of their ends
POST_START = {"left": (MAT_X, MAT_Y), "right": (MAT_X, -MAT_Y)}  # broad faces toward +-x (lens toward the press)
POST_START_YAW = 0.0

# Where the demo stands them: beam 178 mm from the die line (the middle of the 168-188 mm window), each post's
# outer face 5 mm from its guard.
TARGET_D = (S_MIN + S_MAX) / 2  # 0.178
TARGET_X = DIE_X - TARGET_D  # 0.262
TARGET_GAP = 0.005
TARGET_Y = GUARD_INNER_Y - TARGET_GAP - POST_HALF[0]  # 0.117 (turned: its 36 mm depth runs along y)

# Rule thresholds (scoring.py).
GAP_MAX = 0.010  # 'against its guard': the post's outer face within this of the guard's inner face
GAP_PEN = 0.002  # ... and not more than this into it (contact penetration); deeper means it is not beside it
TILT_MAX_DEG = 5.0  # 'upright': the post's long axis within this of vertical
FLOOR_TOL = 0.003  # 'on the floor': the post's lowest point within this of the floor plate's top
FACING_MAX_DEG = 5.0  # 'facing': the lens axis within this of the line between the two posts' centres
STILL_WINDOW_S = 0.5  # 'at rest': no tracked point of a post moved more than STILL_TOL over the last 0.5 s
STILL_TOL = 0.002
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
RESET_JUMP = 0.08  # a post moving this far in one control step was teleported: a new episode began
START_TOL = 0.045  # ... onto its starting mark: within this of it (covers a seeded reset's +-3 cm jitter)
EPS = 1e-9  # float slack on the window's edges (0.440 - 0.272 is not exactly 0.168 in binary)

# Colours (sRGB).
PRESS_GREEN = (0.33, 0.43, 0.38)
GUARD_GREY = (0.30, 0.30, 0.32)
FLOOR_GREY = (0.62, 0.62, 0.64)
TAPE_YELLOW = (0.95, 0.80, 0.10)
POST_BLACK = (0.08, 0.08, 0.09)
MAT_BLUE = (0.20, 0.30, 0.55)
