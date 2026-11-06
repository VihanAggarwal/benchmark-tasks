"""Every dimension and threshold of the task, in one place (metres, kilograms, seconds, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say TOP.

The washer rack is a fixed floor plate with four low walls and two fixed holders filling its far corners (native
fixed boxes: stand-ins for the fittings a real basket has fixed in it). The feed hub and the spray arm are
one fixed mechkit assembly: the hub is its base and the spray arm a link on a revolute joint about the vertical
through HUB (its angle, positive anticlockwise seen from above, is how far the arm has turned; the red cap marks the
end that points forward at the start). The two jars and the two bowls are loose native cylinders.
"""

import math

TOP = 0.7475  # station table top
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20

# -- the washer rack (fixed): a floor plate and four walls standing on its edges ---------------------------------
RACK_XY = (0.270, 0.0)  # the floor plate's centre
FLOOR_SIZE = (0.312, 0.312, 0.004)
FLOOR_T = FLOOR_SIZE[2]
FLOOR_TOP = TOP + FLOOR_T  # the rack floor's top (world z)
WALL_T, WALL_H = 0.006, 0.020  # the walls stand on the floor plate: z 0.004 to 0.024 above the table
INNER_X = (0.120, 0.420)  # the walls' inner faces (world x) ...
INNER_Y = (-0.150, 0.150)  # ... and (world y): 30 x 30 cm inside
WALL_TOP = FLOOR_T + WALL_H  # 0.024
# Two fixed holders fill the far corners: boxes 85 x 85 mm, 50 mm tall, standing on the floor plate against the far
# wall and a side wall (x 0.335 to 0.420, |y| 0.065 to 0.150). With them there, the only places a jar can stand
# clear of the arm's sweep are the two near corners: an upright jar anywhere in the rack's far part (its side
# against a holder, a wall or nothing) has its centre at most 131.2 mm from the hub, 9.3 mm short of CLEAR_R.
HOLDER = (0.085, 0.085, 0.050)
HOLDER_XY = {"holder_left": (INNER_X[1] - HOLDER[0] / 2, INNER_Y[1] - HOLDER[1] / 2),
             "holder_right": (INNER_X[1] - HOLDER[0] / 2, INNER_Y[0] + HOLDER[1] / 2)}  # (0.3775, +-0.1075)
HOLDER_TOP = FLOOR_T + HOLDER[2]  # 0.054 above the table

# -- the feed hub and the spray arm (one fixed mechkit assembly; base frame origin on the table top at HUB) -------
HUB = (0.270, 0.0)  # the arm's vertical axis (world x, y): the rack's centre
HUB_R = 0.015  # the feed hub: 30 mm across ...
HUB_TOP = 0.084  # ... standing on the floor plate (z 0.004 to 0.084)
BAR = (0.220, 0.020, 0.020)  # the spray arm: 220 mm long (along x at the start), 20 mm wide, 20 mm tall
BAR_Z = 0.095  # its centre's height: underside 0.085 (1 mm over the hub), top 0.105
BAR_LOW = BAR_Z - BAR[2] / 2  # 0.085
BAR_TOP = BAR_Z + BAR[2] / 2  # 0.105
CAP = (0.020, 0.020, 0.004)  # a red cap on top of the forward end ...
CAP_X = 0.100  # ... centred this far out along the arm
CAP_TOP = BAR_TOP + CAP[2]  # 0.109
R_TIP = math.hypot(BAR[0] / 2, BAR[1] / 2)  # 0.11045: the circle the arm's end corners (and the cap's) sweep
ARM_MASS = 0.080  # kg (a hollow 20 x 20 mm aluminium square tube 220 mm long with its plastic cap)
ARM_DAMPING = 0.005  # N m s/rad: light drag, no spring (with the link's 3.3e-4 kg m^2 it coasts 0.065 s)
ARM_LOWER, ARM_UPPER = -6.2, 6.2  # rad: the axle's stops, 355 degrees either way (inside the simulator's +-2 pi)

# -- the load (loose) ----------------------------------------------------------------------------------------------
JAR_R, JAR_H, JAR_MASS = 0.025, 0.120, 0.100  # white polypropylene jars: 50 mm across, 120 mm tall, closed
BOWL_R, BOWL_H, BOWL_MASS = 0.035, 0.040, 0.060  # silver-grey stainless bowls, upside down: 70 mm across, 40 mm tall
JARS = ("jar_left", "jar_right")
BOWLS = ("bowl_left", "bowl_right")
ITEMS = JARS + BOWLS
SIZE = {"jar_left": (JAR_R, JAR_H), "jar_right": (JAR_R, JAR_H),
        "bowl_left": (BOWL_R, BOWL_H), "bowl_right": (BOWL_R, BOWL_H)}  # name -> (radius, height)
START = {  # the items' centres at the start (world): standing on the table beside the rack
    "jar_left": (0.180, 0.280, TOP + JAR_H / 2),
    "jar_right": (0.180, -0.280, TOP + JAR_H / 2),
    "bowl_left": (0.400, 0.300, TOP + BOWL_H / 2),
    "bowl_right": (0.400, -0.300, TOP + BOWL_H / 2),
}

# -- where the demo puts them (the score does not depend on where, only on the rules below) -----------------------
JAR_PLACE = {"jar_left": (0.155, 0.115), "jar_right": (0.155, -0.115)}  # the rack's near corners: 162.6 mm from
#                                         the hub (22.1 mm more than CLEAR_R), 10 mm from both walls
BOWL_PLACE = {"bowl_left": (0.265, 0.105), "bowl_right": (0.265, -0.105)}  # under the arm's sweep, beside the hub
#                                         (105.1 mm out), 10 mm from the side wall, 35 mm from the holders (an open
#                                         far finger's outer face 13 mm from them)
PUSH_R = 0.072  # the demo pushes the arm round with a closed gripper's fingers on this circle about the hub

# -- scoring (scoring.py) -------------------------------------------------------------------------------------------
LOAD_TILT_DEG = 10.0  # 'loaded': the axis within this of vertical (either end up) ...
LOAD_BOTTOM = 0.005  # ... its lowest point within this of the rack floor's top ...
#                        (and its centre, seen from above, inside INNER_X x INNER_Y)
CLEAR_MARGIN = 0.005  # 'clear': a jar's side at least this far outside the 11.05 cm sweep circle, i.e. ...
CLEAR_R = 0.1405  # ... its centre at least this far from the hub's axis (11.05 + 2.5 + 0.5 cm) ...
CLEAR_SIDE = CLEAR_R - JAR_R  # ... and its side at least 0.1155 from it at every height the arm and its cap pass
CLEAR_Z = (BAR_LOW, CAP_TOP)  # (0.085 to 0.109 above the table; the jar's cross-section there taken as a circle of
#                               radius JAR_R / cos(tilt)): this only bites for a jar leaning toward the hub
STRETCH_MOVE = 0.010  # a load stretch ends when an item's centre is this far from where it was when it started
#                        (or on a step on which a gripper is on an item: see HELD_MARGIN)
TURN_NEED_DEG = 180.0  # the clean half turn: this much turn within one stretch (either way)
HELD_MARGIN = 0.020  # a gripper is ON an item when its fingers come this close to the item's side (horizontally)
#                       with its pad-tip point below the item's top. The fingers are taken as a disc round the
#                       pad-tip point (the point midway between the finger tips) reaching a finger's outer corner:
#                       hypot(half the gap + FINGER_THICK, FINGER_HALF_WIDTH), 16.8 mm closed, 58.5 mm fully open
FULL_GAP = 0.0939  # m between the pads at opening 1.0
FINGER_THICK = 0.010  # a finger's thickness outside the gap
FINGER_HALF_WIDTH = 0.0135  # half a finger's width across the closing direction, at the tips
STILL_PATH = 0.005  # success: every item's path over the last STILL_WINDOW_S under this
STILL_WINDOW_S = 0.5
STILL_STEPS = int(round(STILL_WINDOW_S * CONTROL_HZ))  # 10 control steps
SUCCESS_HOLD_S = 1.0  # the trial ends once success has held this long

# Reset fallback (rigstate's and mechkit's reset hooks normally come first): in one control step some item jumps
# more than RESET_JUMP and afterwards every item is within START_TOL of its start, with the arm back along x.
RESET_JUMP = 0.020
START_TOL = 0.005
START_ARM_TOL_DEG = 5.0

# -- the offline stand-in for the arm's joint (offline_model.py; the simulator never uses it) -----------------------
PUSH_Z = (0.045, 0.105)  # a gripper's pad-tip point this high above the table ...
PUSH_RADII = (0.020, 0.110)  # ... and this far from the hub's axis pushes the arm ...
PUSH_HALF = 0.024  # ... keeping its line this far (bar half-width 10 mm + finger half 14 mm) from the tips point

# Colours (sRGB).
RACK_GREY = (0.35, 0.36, 0.38)  # the stainless rack, dark
HOLDER_GREY = (0.50, 0.51, 0.53)
HUB_GREY = (0.70, 0.71, 0.72)
ARM_GREY = (0.75, 0.76, 0.78)
RED = (0.85, 0.12, 0.10)
WHITE = (0.95, 0.95, 0.94)
SILVER = (0.82, 0.83, 0.85)
