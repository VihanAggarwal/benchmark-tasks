"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless named *_W.

A Logitech K120 wired keyboard (maker: 450 x 155 x 23.5 mm, 550 g, full-size US layout) lies on the desk with its
front edge toward the robot, long side along y, Left Ctrl at the robot's left (+y) end and the number pad at the
right (-y) end. Behind it stands the PC's monitor showing the Windows 11 lock screen. Keys sit on the standard
19.05 mm (0.75 in) pitch; key positions below are given in key units (u) from the left edge of the main block and in
rows from the front (row 1 = the Ctrl / space row). Estimates are named in task.json's fidelity.
"""

import math

MM = 0.001
TOP = 0.7475
G = 9.81
U = 0.01905  # key pitch

# -- the keyboard (loose, base body) ---------------------------------------------------------------------------------
KB_L, KB_D = 0.450, 0.155  # length along y, depth along x (maker)
KB_MASS = 0.55  # maker
KB_X, KB_Y = 0.330, 0.010  # body frame origin (centre of its footprint) in the world
FOOT_H = 0.004  # rubber feet
CASE_Z0, CASE_Z1 = FOOT_H, 0.016  # the case under the keys
CAP_H = 0.007  # keycap height above the case; cap tops 23 mm up
CAP_TOP = CASE_Z1 + CAP_H
CAP_GAP = 0.001  # 18 mm caps on 19.05 mm pitch (1.05 mm between caps)
MAIN_LEFT_Y = 15.0 * U / 2 + 7.5 * U / 2  # body frame y of the main block's left edge: the 22.5 u key field centred
ROW1_X = -KB_D / 2 + 0.012 + U / 2  # body frame x of row 1 (front row) centres
FKEY_GAP = 0.5 * U  # the function-key row stands half a key behind row 5
KB_BLACK = (0.10, 0.10, 0.11)
CAP_BLACK = (0.16, 0.16, 0.17)
LEGEND_WHITE = (0.92, 0.92, 0.92)
FEET_RUBBER = (0.05, 0.05, 0.05)
KB_FRICTION = (0.8, 0.7)  # rubber feet on desk laminate (est)


def key_xy(col_u: float, row: float) -> tuple:
    """Body frame (x, y) of a key's centre from its centre column (u from the main block's left edge) and row."""
    return (ROW1_X + (row - 1.0) * U, MAIN_LEFT_Y - col_u * U)


# Modelled keys (sprung caps on sliders): name -> (centre column u, row, width u, height u). Every key near the
# targets is modelled, so a stray press registers.
KEYS = {
    "lctrl": (0.625, 1, 1.25, 1), "lwin": (1.875, 1, 1.25, 1), "lalt": (3.125, 1, 1.25, 1),
    "space": (6.875, 1, 6.25, 1), "ralt": (10.625, 1, 1.25, 1), "rwin": (11.875, 1, 1.25, 1),
    "menu": (13.125, 1, 1.25, 1), "rctrl": (14.375, 1, 1.25, 1), "lshift": (1.125, 2, 2.25, 1),
    "insert": (15.75, 5, 1, 1), "home": (16.75, 5, 1, 1), "pgup": (17.75, 5, 1, 1),
    "delete": (15.75, 4, 1, 1), "end": (16.75, 4, 1, 1), "pgdn": (17.75, 4, 1, 1),
    "numlock": (19.0, 5, 1, 1),
    "np7": (19.0, 4, 1, 1), "np8": (20.0, 4, 1, 1), "np9": (21.0, 4, 1, 1),
    "np4": (19.0, 3, 1, 1), "np5": (20.0, 3, 1, 1), "np6": (21.0, 3, 1, 1),
    "np1": (19.0, 2, 1, 1), "np2": (20.0, 2, 1, 1), "np3": (21.0, 2, 1, 1),
    "np0": (19.5, 1, 2, 1), "npdot": (21.0, 1, 1, 1), "npenter": (22.0, 1.5, 1, 2),
}
DIGIT_KEYS = {"np0": "0", "np1": "1", "np2": "2", "np3": "3", "np4": "4", "np5": "5", "np6": "6", "np7": "7",
              "np8": "8", "np9": "9"}
CTRL_KEYS = ("lctrl", "rctrl")
ALT_KEYS = ("lalt", "ralt")
WIN_KEYS = ("lwin", "rwin")

# The rest of the board as fixed caps on the case (visual and collision): (centre column u, row, width u).
STATIC_ROWS = {
    2: [(3.25, 1)] + [(2.25 + 0.5 + i, 1) for i in range(1, 10)] + [(13.125, 2.75)],
    3: [(0.875, 1.75)] + [(1.75 + 0.5 + i, 1) for i in range(11)] + [(13.875, 2.25)],
    4: [(0.75, 1.5)] + [(1.5 + 0.5 + i, 1) for i in range(12)] + [(14.25, 1.5)],
    5: [(0.5 + i, 1) for i in range(13)] + [(14.0, 2.0)],
}
STATIC_NAV = [(16.75, 2), (15.75, 1), (16.75, 1), (17.75, 1)]  # arrow keys
STATIC_NUMPAD = [(20.0, 5), (21.0, 5), (22.0, 5)]  # / * -
NP_PLUS = (22.0, 3.5)  # the tall + key, rows 3 to 4
FKEYS = [0.5] + [2.0 + i for i in range(4)] + [6.5 + i for i in range(4)] + [11.0 + i for i in range(4)] + \
        [15.75, 16.75, 17.75]

# -- key travel (membrane dome on a slider; est, measure on the unit) ------------------------------------------------
KEY_TRAVEL = 0.0028  # to the bottom stop (the slider's joint limit; distributor datasheet: 2.8 mm)
KEY_ACT = 0.0020  # registers past 2.0 mm
KEY_SPRING = 200.0  # N/m
KEY_PRELOAD = 0.0015  # the target sits 1.5 mm above the rest stop: 0.30 N at rest, 0.70 N at actuation
KEY_MASS = 0.003
KEY_DAMPING = 0.05
ACT_FORCE = KEY_SPRING * (KEY_PRELOAD + KEY_ACT)  # 0.70 N

# -- the monitor (fixed) and its screen states --------------------------------------------------------------------
MON_X, MON_Y = 0.560, 0.010  # stand base centre
MON_W, MON_T, MON_H = 0.530, 0.020, 0.310  # panel (Dell P2422H class, 24 in)
MON_Z0 = 0.110  # panel's lower edge
MON_FACE_X = MON_X - 0.040  # the screen's face (toward the robot)
SCREEN_LOCK = (0.08, 0.20, 0.42)
SCREEN_PROMPT = (0.12, 0.30, 0.52)
SCREEN_DESKTOP = (0.15, 0.55, 0.60)
WHITE = (0.95, 0.95, 0.95)
FRAME_BLACK = (0.06, 0.06, 0.06)
PLATE_SHOW = 0.010  # a state plate slides this far out of the panel (toward -x) to show

PIN = "2580"  # the account's PIN (on the sticky note on the monitor's bezel)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
CONTROL_HZ = 20.0
HOLD_PAIR_S = 0.5  # 0.25: Ctrl and Alt both down together this long
LIFT_LATCH = 0.005  # or lifted more than 5 mm
TILT_LATCH_DEG = 5.0
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.001
STILL_TURN_DEG = 0.3
LET_GO = 0.030  # grippers 30 mm or more from the keyboard
RESET_JUMP = 0.01
EPS = 1e-9


def key_world(name: str) -> tuple:
    """World (x, y) of a modelled key's centre (keyboard at its start pose)."""
    col, row, _, _ = KEYS[name]
    x, y = key_xy(col, row)
    return (KB_X + x, KB_Y + y)


def key_half(name: str) -> tuple:
    """Half extents (x, y) of a modelled key's cap top."""
    _, _, w, h = KEYS[name]
    return ((h * U - CAP_GAP) / 2, (w * U - CAP_GAP) / 2)


def dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
