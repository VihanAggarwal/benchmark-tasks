"""Every dimension of the task, in one place (metres, kilograms, the station's world frame).

World frame: x forward from the line through the two arm bases, y to the left, z up. The arm bases stand at
x 0, y +-0.305; the station's table top is at z 0.7475. Heights below are above the table top unless they say
'world'.

A loading dock (on the right arm's side, y < 0) and a truck bed (on the left arm's side, y > 0) face each other
across an 80 mm gap at y -0.040 .. +0.040. Each has two seats (rebates) at its gap edge, one pair at x 0.260-0.310
and one at x 0.340-0.390, separated by a low rib. A runner plank seated in a dock seat and the facing bed seat
bridges the gap. The cart starts on the dock and must end on the truck bed between two yellow marks.

The surfaces the cart slides over step down 0.5 mm at every seam it crosses in the direction it is pushed (dock
40.0 mm, runners 39.5 mm, bed 39.0 mm), so its leading edge never meets a vertical face. The blocks under the
cart's path have no seams across the path: the dock's and the bed's main blocks span the full width.
"""

TOP = 0.7475  # station table top (world z)

# -- the dock (fixed, concrete) and the truck bed (fixed, timber deck) ------------------------------------------
X0, X1 = 0.230, 0.410  # both span x 0.230 .. 0.410
GAP_Y = 0.040  # the gap: y -0.040 .. +0.040
SEAT_DEPTH = 0.045  # the seats reach this far back from the gap edge: to y -0.085 / +0.085
SEAT_Y = GAP_Y + SEAT_DEPTH  # 0.085
DOCK_Y_END = -0.300  # the dock's back edge
BED_Y_END = 0.300  # the bed's far edge
DOCK_TOP = 0.0400
BED_TOP = 0.0390
SEAT_FLOOR = 0.0195  # the seats' floors (both sides)
RIB_TOP = 0.0295  # the low rib between a side's two seats: 10 mm under a seated runner's top, never touches the cart
SEATS = {"a": (0.260, 0.310), "b": (0.340, 0.390)}  # each pair of seats, x from .. to
RIB_X = (0.310, 0.340)
CHEEKS = ((X0, 0.260), (0.390, X1))  # full-height cheeks outside the seats (outside the cart's path)

# Seat rectangles (x0, x1, y0, y1) on each side.
DOCK_SEAT = {k: (x0, x1, -SEAT_Y, -GAP_Y) for k, (x0, x1) in SEATS.items()}
BED_SEAT = {k: (x0, x1, GAP_Y, SEAT_Y) for k, (x0, x1) in SEATS.items()}

# Surfaces at the cart's level (x0, x1, y0, y1, top): what it can slide on or rest on.
CART_SURFACES = (
    (X0, X1, DOCK_Y_END, -SEAT_Y, DOCK_TOP),  # dock main block
    (*CHEEKS[0], -SEAT_Y, -GAP_Y, DOCK_TOP),
    (*CHEEKS[1], -SEAT_Y, -GAP_Y, DOCK_TOP),
    (X0, X1, SEAT_Y, BED_Y_END, BED_TOP),  # bed main block
    (*CHEEKS[0], GAP_Y, SEAT_Y, BED_TOP),
    (*CHEEKS[1], GAP_Y, SEAT_Y, BED_TOP),
)

# -- the runner planks (loose) ---------------------------------------------------------------------------------
RUNNER_SIZE = (0.040, 0.160, 0.020)  # across (x), long (y), thick: a 40 x 20 mm aluminium tube, 1.2 mm wall
RUNNER_HALF = tuple(s / 2 for s in RUNNER_SIZE)
RUNNER_MASS = 0.06  # 22 cm3 of aluminium (a solid bar this size would weigh 0.35 kg)
RUNNER_TOP_SEATED = SEAT_FLOOR + RUNNER_SIZE[2]  # 0.0395: half a millimetre under the dock, over the bed
# Lying on the table left of the truck bed, along y, end to end; both more than 60 mm from the bed's x 0.230 edge.
RUNNERS = {"runner_a": (0.150, 0.290), "runner_b": (0.160, 0.460)}  # centres: A y 0.210..0.370, B y 0.380..0.540
SEATED_X = {"a": 0.285, "b": 0.365}  # a runner centred in each pair of seats

# -- the cart (loose) --------------------------------------------------------------------------------------------
CART_SIZE = (0.100, 0.100, 0.030)  # too wide for the gripper (94 mm fully open); low, so a push cannot tip it
CART_HALF = tuple(s / 2 for s in CART_SIZE)
CART_MASS = 0.21  # a solid painted hardwood block, about 0.7 g/cm3
CART_X, CART_Y = 0.325, -0.200  # on the dock, square to it: x 0.275..0.375, y -0.250..-0.150
TARGET_X, TARGET_Y = 0.325, 0.210  # the cart's centre on the truck bed, between the marks

# Yellow marks on the bed top beside the cart's path (fixed, 0.5 mm thick).
MARK_SIZE = (0.025, 0.120, 0.0005)
MARKS = {"mark_near": (0.2425, TARGET_Y), "mark_far": (0.3975, TARGET_Y)}  # at the bed's x edges: x 0.230..0.255, 0.385..0.410

# -- rule thresholds (scoring.py) --------------------------------------------------------------------------------
CART_DROP = 0.035  # the cart's lowest point below this: it dropped into the gap (or off the dock)
CART_LIFT = 0.045  # ... above this: it was lifted, not pushed
CART_TIP_DEG = 10.0  # ... tilted more than this: it tipped
RUNNER_SHIFT_MAX = 0.010  # a runner's end moving more than this while the cart is on it: it slid out under the load
OFF_TABLE = 0.010  # anything's lowest point this far below the table top: it left the table
SEAT_Z_TOL = 0.002  # 'seated': the runner's bottom within this of the seat floors at both ends,
SEAT_FLAT_DEG = 5.0  # lying flat (broad faces within this of level),
SEAT_XY_TOL = 0.001  # each end's corners inside a seat (to this) and the facing seat on the other side
BED_Z_TOL = 0.002  # 'on the bed': the cart's lowest point within this of the bed top, its footprint on the bed
CENTRE_TIGHT = 0.015  # 1.0: the cart's centre within this of the target
CENTRE_LOOSE = 0.040  # 0.75
YAW_TIGHT_DEG = 5.0  # 1.0: the cart square to the bed within this
HANDOVER_Y = GAP_Y  # 0.5: the cart's centre at or past y +0.040 (over the truck-bed side of the gap)
OVER_GAP_TOL = 0.001  # the cart is 'over the gap' when its outline reaches more than this into y -0.040 .. +0.040
ON_RUNNER_Z = 0.003  # the cart counts as on a runner when its lowest point is within this of the runner's top
ON_RUNNER_OVERLAP = 0.005  # ... and their outlines overlap by at least this along x and along y
HANDS_OFF = 0.030  # 1.0: each gripper's finger zone at least this far from the cart and both runners
STILL_SPEED = 0.01  # m/s
STILL_SPIN = 0.5  # rad/s
CONTROL_HZ = 20.0  # the program's runs use -E control_hz=20
GRAVITY = 9.81
RESET_JUMP = 0.08  # an object moving this much further in one step than its velocity allows was teleported
START_TOL = 0.045  # ... and a new trial began if every loose object is then within this of its start

# The grippers (yam_arena): pads 94 mm apart fully open; rigstate reports the opening as 0 (closed) .. 1 (open).
FULL_GAP = 0.0939
FINGER_THICK = 0.010  # finger thickness outside the gap, lowest 38 mm (6.4 mm at the tips, 10 mm at 38 mm up)
FINGER_HALF_WIDTH = 0.020  # half the finger's width across the closing axis there (27 mm at the tips, 40 mm higher)
FINGER_LEN = 0.038  # the fingers' zone runs from the tips this far back up the gripper
# The demo's pushes and the offline push model use the fingers where they bear on the cart (their lowest 25 mm):
PAD_THICK = 0.008  # finger thickness there, outside the gap (6.4 mm at the tips)
PAD_HALF_WIDTH = 0.0135  # half the finger's width there (27 mm at the tips)

# Colours (sRGB).
CONCRETE = (0.52, 0.52, 0.50)
DECK_BROWN = (0.35, 0.25, 0.18)
ALUMINIUM = (0.80, 0.81, 0.83)
CART_BLUE = (0.15, 0.30, 0.70)
YELLOW = (0.95, 0.80, 0.10)
