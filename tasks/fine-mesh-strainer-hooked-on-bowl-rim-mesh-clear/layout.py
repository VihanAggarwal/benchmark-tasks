"""Every dimension of the task, in one place (metres, kilograms, the station's world frame), real size.

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at x 0,
y +-0.305; the station's table top is at z 0.7475. Heights named *_B are in the bowl's frame (z up from the bottom of
its foot); heights without a suffix are above the table top unless said otherwise.

On a counter (the table) stand a stainless mixing bowl on a non-slip mat, a stock pot and a backsplash wall. The bowl
holds about half a litre of strained stock. A fine-mesh strainer lies mesh down beside the bowl. It is picked up by
its handle and hung across the bowl: the hook over the right rim, the handle on the left rim, the mesh in the bowl
clear of the stock, then let go.

Strainer frame ("local"): origin at the centre of the ring on the plane z 0 that the hook strap's and the handle's
undersides lie in (the plane that rests on the rim); local x toward the handle, y across it, z up. Placed with yaw
+90 degrees, local x is the world's +y.

Figures are sourced (README) or estimates (est, said in task.json's fidelity): see the comments.
"""

import math

MM = 0.001
TOP = 0.7475  # station table top (world z)
G = 9.81

# -- the non-slip mat (fixed): a damp cloth or silicone mat under the bowl ------------------------------------------
MAT_X, MAT_Y = 0.30, 0.0
MAT_SIZE = (0.26, 0.26, 0.004)  # est
MAT_COLOR = (0.30, 0.34, 0.40)
MAT_FRICTION = (0.9, 0.8)  # silicone on steel (est)

# -- the bowl (loose): stainless mixing bowl, 220 mm across the rim, 105 mm deep, rolled rim bead 8 mm ----------------
BOWL_X, BOWL_Y = 0.30, 0.0  # on the mat, centred
BOWL_Z0 = TOP + MAT_SIZE[2]  # the bottom of its foot
BOWL_H = 0.105  # foot to the bead's top
BOWL_FLOOR = 0.002  # floor thickness
BOWL_WALL = 0.002  # wall thickness (a real one is about 1 mm: thin plates glitch)
R_FLOOR_IN = 0.070  # the wall's inner radius at the floor
R_RIM_IN = 0.100  # the wall's inner radius at the rim: the opening is 200 mm across inside
BEAD_OUT = 0.110  # the bead's outer radius: 220 mm across the rim
BEAD_H = 0.008  # the rolled bead 8 mm
BEAD_TOP_B = BOWL_H  # 105 mm: the rim's top in the bowl's frame
RIM_R = (R_RIM_IN + BEAD_OUT) / 2  # 0.105: the bead's centre line
WALL_SLOPE = (R_RIM_IN - R_FLOOR_IN) / (BOWL_H - BOWL_FLOOR)  # 0.2913: the wall flares outward going up
WALL_SEGMENTS = 24
BEAD_SEGMENTS = 24
STOCK_DEPTH = 0.030  # 0.5 L of strained stock, drawn as a solid disc
STOCK_TOP_B = BOWL_FLOOR + STOCK_DEPTH  # 32 mm above the foot
STOCK_R = 0.0775
BOWL_MASS = 0.30 + 0.52  # the bowl 0.30 kg (3 qt class, est) + 0.5 L of stock at 1 kg/L
BOWL_STEEL = (0.74, 0.76, 0.80)
STOCK_COLOR = (0.62, 0.42, 0.18)
STEEL_FRICTION = (0.40, 0.35)  # stainless on stainless (est)
BOWL_FRICTION = (0.40, 0.35)

# -- the strainer (loose): hook strainer, 130 mm ring, 55 mm deep ------------------------------------------------------
RING_R = 0.065  # 130 mm across
RING_WALL = 0.0015
RING_DEPTH = 0.047  # the ring's wall below the rim plane (the mesh floor's underside)
RING_TOP = 0.010  # the ring's wall rises to the handle's top: 57 mm tall in all
MESH_T = 0.002  # the mesh drawn as a 2 mm solid floor (a real 0.5 mm mesh is a screen)
RING_SEGMENTS = 16
STRAP_T = 0.0015  # the hook strap and the handle's web
HK_IN = -0.090  # the hook's inner leg's outer face (local x): inside the bowl's wall
HK_OUT = -0.120  # the hook's outer leg's inner face: outside the bead. The gap between them is 30 mm
HK_GAP = HK_IN - HK_OUT
HK_END = HK_OUT - STRAP_T  # -0.1215: the hook's outermost face
HK_W = 0.018  # the strap is 18 mm wide
HK_LEG_OUT = 0.020  # the outer leg hangs 20 mm below the strap's underside
HK_LEG_IN = 0.012  # the inner leg 12 mm
HD_X0 = 0.0625  # the handle starts at the ring's wall
HD_LEN = 0.130
HD_X1 = HD_X0 + HD_LEN  # 0.1925: the handle's end
HD_W = 0.018  # a channel 18 mm wide
HD_FLANGE_H = 0.010  # with 10 mm flanges (a folded stainless handle); the flanges' bottoms are in the rest plane
HD_FLANGE_T = 0.0015
STRAINER_LEN = HD_X1 - HK_END  # 0.314 m: from the hook's outside to the handle's end
STRAINER_MASS = 0.12  # est (a 5 in hook strainer weighs 120 to 200 g: a light one, so the pinch carries less)
STRAINER_STEEL = (0.80, 0.82, 0.85)
STRAINER_FRICTION = (0.40, 0.35)

# where the strainer lies at the start: mesh down on the table, long axis along y, handle toward +y
ST_X, ST_Y = 0.30, 0.252  # the ring's centre
ST_YAW = math.pi / 2
ST_Z = TOP + RING_DEPTH  # the origin's height when the mesh floor lies on the table
ST_HANDLE_END_Y = ST_Y + HD_X1  # 0.4445: inside |y| 0.45, clear of the left arm's start pose

GRIP_X = 0.082  # where the pads close on the handle (local x): next to the ring, 71 mm from the centre of mass, so the
# grip carries half the twisting load of a grip at the handle's end (0.12 kg x 71 mm = 0.08 N m); hung on the bowl the pads are inside the opening

# and where it is meant to end: the hook over the right (-y) rim, the handle on the left (+y) rim
HOOK_AIM_S = 0.002  # the demo aims the ring's centre this far past the bowl's axis toward the handle (m)
BEAD_TOP_W = BOWL_Z0 + BEAD_TOP_B  # 0.8565: the rim's top in the world

# -- the scene around it (fixed) -----------------------------------------------------------------------------------
WALL_X = 0.565  # a backsplash wall behind the counter: its front face
WALL_SIZE = (0.020, 0.90, 0.30)
WALL_COLOR = (0.88, 0.88, 0.84)
POT_X, POT_Y = 0.42, -0.30  # the stock pot the stock came from
POT_R, POT_H = 0.095, 0.20
POT_COLOR = (0.55, 0.57, 0.60)
SCENE_FRICTION = (0.5, 0.45)

# -- rule thresholds (scoring.py) ------------------------------------------------------------------------------------
HOOK_GAP_UP = 0.006  # 'on the rim': the strap's underside within 6 mm above the bead's top ...
HOOK_GAP_DOWN = 0.002  # ... and no more than 2 mm into it
HANDLE_OVER = 0.005  # the handle's underside crosses the bead within 5 mm of its centre line
RING_CENTRE_MAX = 0.030  # 'wholly inside the bowl's outline': the ring's rim 5 mm inside the rim's inner edge
AXIS_THROUGH = 0.020  # the strainer's long axis passes within 20 mm of the bowl's axis (top view)
LEVEL_DEG = 10.0  # the long axis within 10 degrees of horizontal
MESH_ABOVE_STOCK = 0.015  # 'mesh clear': the mesh's underside 15 mm or more above the stock
MESH_BELOW_RIM = 0.020  # and at least 20 mm below the rim's top
RESTS_TILT_DEG = 25.0  # 0.5: a strainer resting across the rims, not hooked, within 25 degrees of horizontal
RIGHT_RIM_DEG = 45.0  # 'the right rim': the hook's middle within 45 degrees of straight right (-y) of the bowl's axis
IN_BOWL_CENTRE = 0.035  # (not scored) the ring's centre over the bowl's floor: a strainer sitting in the stock scores 0
BOWL_MOVED = 0.030  # final state: the bowl's foot within this of its mark (top view), else 0
BOWL_TILT_DEG = 10.0  # final state: the bowl tilted no more than this, else 0
FALL_DROP = 0.020  # latched: either part's lowest point further than this below the table top
STILL_WINDOW_S = 1.0
STILL_MOVE = 0.005
STILL_TURN_DEG = 5.0
LET_GO = 0.030
CONTROL_HZ = 20.0
RESET_JUMP = 0.05
START_TOL = 0.01
EPS = 1e-9
