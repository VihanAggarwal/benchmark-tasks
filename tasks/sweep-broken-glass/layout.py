"""Every dimension of the scene, in one place. Pure Python.

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is at TOP.

Sources (README.md): OXO Good Grips Compact Dustpan & Brush Set, item 1334280 (dustpan 186 x 137 x 40 mm; the set
0.45 lb). The pieces of the broken glass are stand-ins cut from 3 mm clear acrylic to the sizes below (a drinking
glass's wall is about 3 mm thick). The dustpan's and brush's section, the brush head's size and the masses are
estimates (task.json "fidelity").
"""

from __future__ import annotations

MM = 0.001
TOP = 0.7475
CONTROL_HZ = 20.0

# -- the dustpan (pan frame: origin on the table under the middle of its lip, x back toward the robot, y left) -------
# It lies on the table with its mouth facing +x (away from the robot) and its back toward the robot.
PAN_W = 186.0  # mm across (y)
PAN_D = 137.0  # mm from lip to back (x in the pan frame, toward the robot)
PAN_H = 40.0  # mm: the back wall's height
PAN_WALL = 3.0
PAN_FLOOR = 2.5  # mm floor thickness
LIP_LEN = 18.0  # mm: the thin lip ramps from 0.5 mm at its edge up to the floor's top over this length
LIP_EDGE = 0.5  # mm
SIDE_H = 22.0  # mm side walls (they taper on the real pan; a constant height here)
M_PAN = 0.120  # kg (est.; the set weighs 0.45 lb with the brush)
PAN_COLOUR = (0.12, 0.12, 0.13)  # black
PAN_FRICTION = (0.5, 0.4)
PAN_LIP_X = 0.270  # world x of the lip's edge (the mouth faces +x)
PAN_Y = 0.000

# -- the brush (brush frame: origin at the middle of the bristle tips' footprint, the head's length along y, the
# handle continuing the head's line toward -y, as on a hand brush; it sweeps sideways, along x) ----------------------
HEAD_W = 140.0  # mm: the head's length (y)
HEAD_D = 28.0  # mm through the head (x)
BRISTLE_H = 22.0  # mm of bristles below the head's back
BACK_H = 12.0  # mm head back (stock)
HANDLE_L = 70.0  # mm handle beyond the head's end
HANDLE_W = 16.0  # mm across (x)
HANDLE_T = 18.0  # mm thick (z)
M_BRUSH = 0.085  # kg (est.)
BRUSH_COLOUR = (0.12, 0.12, 0.13)
BRISTLE_COLOUR = (0.85, 0.85, 0.85)
BRUSH_FRICTION = (0.35, 0.3)  # nylon bristle tips on the table (est.)
BRUSH_START = (0.200, -0.190)  # world x, y of the bristle footprint's middle: standing on its bristles, right of the pan

# -- the pieces (stand-ins): name, (x, y) world centre, (length, width) mm, yaw deg ----------------------------------
SHARD_T = 3.0  # mm clear acrylic
SHARDS = (
    ("shard_a", (0.312, 0.034), (32.0, 18.0), 20.0),
    ("shard_b", (0.330, -0.012), (26.0, 14.0), -35.0),
    ("shard_c", (0.348, 0.050), (18.0, 12.0), 60.0),
    ("shard_d", (0.356, 0.006), (22.0, 16.0), 5.0),
    ("shard_e", (0.322, -0.048), (15.0, 10.0), 80.0),
    ("shard_f", (0.372, -0.030), (28.0, 12.0), -60.0),
    ("shard_g", (0.300, -0.010), (12.0, 9.0), 45.0),
    ("shard_h", (0.380, 0.028), (20.0, 11.0), -15.0),
)
ACRYLIC_DENSITY = 1180.0  # kg/m3
SHARD_COLOUR = (0.78, 0.86, 0.90)  # clear acrylic, tinted so it shows on camera
SHARD_FRICTION = (0.45, 0.35)  # acrylic on the table (est.)

# -- scoring -----------------------------------------------------------------------------------------------------------
IN_PAN_Z = 0.012  # m: a piece's centre at most 12 mm above the pan's floor top, inside its walls (resting in it)
PAN_FLAT_DEG = 5.0  # at the end the pan rests flat on the table
PAN_ON_TABLE = 0.004  # m: its lip edge within 4 mm of the table top
TOUCH = 0.0  # m: a gripper's finger zone reaching a piece (rigstate.hands_gap <= 0) counts as touching it
OFF_TABLE_Z = 0.020  # m: a piece more than 20 mm below the table top has fallen off it
LET_GO = 0.030
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.002
EPS = 1e-9
RESET_START_TOL = 0.004
RESET_JUMP = 0.05


def shard_mass(size_mm) -> float:
    return ACRYLIC_DENSITY * size_mm[0] * size_mm[1] * SHARD_T * 1e-9
