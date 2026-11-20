"""Every dimension of the scene, in one place. Pure Python (numpy for the geometry helpers).

World frame (yam_arena's): x forward from the line through the arm bases, y to the robot's left, z up, metres. The
station table's top is TOP. The bulb's frame: origin at its centre; x across it (the squeeze), y along it (its tail and
air-release valve toward -y), z up.

Sources (README.md): a standard manual sphygmomanometer (adult cuff, aneroid gauge 0 to 300 mmHg, latex inflation
bulb with a thumbscrew air-release valve) and the deflation rate in blood pressure guidelines (2 to 3 mmHg per second).
The bulb's stiffness, the air volumes and the valve's flow are estimates (task.json "fidelity").
"""

from __future__ import annotations

import math

import numpy as np

TOP = 0.7475
CONTROL_HZ = 20.0
DT = 1.0 / CONTROL_HZ

# -- the training arm, cuff and gauge (fixed) ---------------------------------------------------------------------------
ARM_X = 0.500  # the training arm lies along y at this x
UPPER_ARM = dict(y=0.115, r=0.045, l=0.300)  # centre y, radius, length
FOREARM = dict(y=-0.130, r=0.035, l=0.220)
HAND = dict(y=-0.285, size=(0.080, 0.090, 0.030))
CUFF = dict(y=0.130, r=0.056, l=0.140)  # an adult cuff (bladder 12 x 22 cm) wrapped on the upper arm
GAUGE = dict(x=0.430, y=0.130, z=0.125, r=0.045)  # a large-dial clinic gauge (90 mm): its centre, facing the robot
SKIN = (0.86, 0.69, 0.58)
CUFF_COLOUR = (0.10, 0.18, 0.45)
DIAL = (0.95, 0.95, 0.93)
DARK = (0.08, 0.08, 0.08)
NEEDLE_COLOUR = (0.80, 0.10, 0.10)
NEEDLE_ZERO = math.radians(-135.0)  # the needle at 0 mmHg points down-left; it turns clockwise (seen from the robot)
NEEDLE_SPAN = math.radians(270.0)  # 0 to 300 mmHg

# -- the inflation bulb (loose) ------------------------------------------------------------------------------------------
BULB_L = 0.100  # body along y
BULB_W = 0.060  # across x, at rest: two halves 30 mm thick
BULB_H = 0.055
SQUEEZE_TRAVEL = 0.020  # the halves close 20 mm when squeezed flat
SQUEEZE_SPRING = 1000.0  # N/m: 20 N squeezed flat (est.)
VALVE_STEM_Y = -(BULB_L / 2 + 0.024)  # the valve body's end, bulb frame
SCREW_Y = VALVE_STEM_Y - 0.013  # the knurled thumbscrew's centre, bulb frame (10 mm stem between)
SCREW_R = 0.005  # the hub
SCREW_T = 0.006
WING = (0.008, 0.006, 0.018)  # the flat-sided wing across the hub: 8 mm thick (x), 6 mm deep (y), 18 mm across (z),
# its flats facing x while the valve is a quarter turn open
VALVE_DAMPING = 0.01  # N m s/rad: the valve turns smoothly and stays where it is set (it is balanced about its axis;
# PhysX joint friction scales with the force carried through the joint and locked the valve while the bulb was held)
M_BULB = 0.045
BULB_FRICTION = (1.4, 1.2)  # latex and the knurled thumbscrew (est.)
RUBBER = (0.10, 0.10, 0.10)
CHROME = (0.78, 0.79, 0.80)
BULB_START = np.array([0.300, 0.100, TOP + BULB_H / 2 + 0.002])  # lying on the table (spawned 2 mm up)
BULB_REST = BULB_START - np.array([0.0, 0.0, 0.002])

# The valve's angle: theta = 0 shut, increasing as it opens (counterclockwise seen from the tail end, i.e. turning the
# thumbscrew about +y). It starts a quarter turn open; the joint is 0 there.
VALVE_START = math.pi / 2
VALVE_MAX = math.pi  # half a turn: wide open

# -- the air (a rule, scoring.Cuff) -------------------------------------------------------------------------------------
PUMP_PER_MM = 0.9  # mmHg per mm of squeeze stroke at 0 mmHg (est.: 15 to 17 strokes reach 180)
PUMP_CEILING = 400.0  # mmHg: what a bulb could reach; each stroke adds less as the cuff fills
PUMP_DEADBAND = 0.0005  # m: squeeze changes smaller than this pump nothing (a bulb merely held)
LEAK = 0.198  # valve flow, mmHg/s = LEAK * f(theta) * sqrt(P) (an orifice's flow): about 57 deg gives 2.5 at 160 and
# 93 deg gives 2.5 at 60
LEAK_KNEE = 2.0  # rad: past this the valve's seat clears and the flow climbs steeply (wide open empties the cuff fast)
LEAK_STEEP = 30.0


def valve_flow(theta: float) -> float:
    """f(theta): the valve's opening as a flow factor (rad-like units), linear up to the knee, then steep."""
    t = max(float(theta), 0.0)
    return t + (LEAK_STEEP * (t - LEAK_KNEE) ** 2 if t > LEAK_KNEE else 0.0)

# -- scoring ---------------------------------------------------------------------------------------------------------------
PEAK_RANGE = (170.0, 200.0)
PEAK_LOOSE = (160.0, 220.0)
DESCENT = (160.0, 60.0)  # the peak must reach 160; the rate is judged from just below the peak down to 60
DESCENT_START_DROP = 2.0  # mmHg below the peak: the descent has begun
RATE_ON = (1.5, 3.5)  # mmHg per 1 s window: 2 to 3, read to +/-0.5 on the video
RATE_GROSS = (1.0, 5.0)  # a pause or a dump
EMPTY_BELOW = 20.0
EMPTY_WITHIN_S = 10.0
EPS = 1e-9


def bulb_point(local, origin=BULB_REST) -> np.ndarray:
    """A point given in the bulb's frame (the bulb lying as it starts), in the world."""
    return np.asarray(origin, dtype=float) + np.asarray(local, dtype=float)
