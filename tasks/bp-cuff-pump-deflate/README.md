# Pump a blood-pressure cuff to 180 mmHg, then let it down at 2 to 3 mmHg a second

An adult training arm lies across the table with a blood-pressure cuff wrapped on its upper arm and the cuff's aneroid
gauge facing the robot. The inflation bulb lies in front of the robot's left arm. Its air-release valve, a wing
thumbscrew on the bulb's tail, starts a quarter turn open. The robot has to:

1. take the bulb and shut the valve;
2. pump the cuff to 170 to 200 mmHg;
3. open the valve just enough that the needle falls 2 to 3 mmHg each second, from just below the peak all the way down to 60;
4. open the valve wide to empty the cuff.

## Why it matters

A manual blood-pressure reading is only as good as its deflation. Guidelines call for 2 to 3 mmHg a second: faster
misses the true systolic and diastolic sounds, and slower leaves the arm congested. Nurses and medical assistants do it
many times a day. One hand works the bulb; a thumb works a small valve screw. Because the valve's flow grows with the
pressure behind it, the screw has to be opened a little further as the pressure drops, to keep the needle falling at
the same pace. The physical difficulty is precise, sustained position control of a small valve, done one way only
(backing it off and re-opening makes the needle stall and jump), together with a real squeeze, repeated, to pump the
cuff up.

## Sources

- A standard manual sphygmomanometer: adult cuff (bladder about 12 x 22 cm), aneroid gauge 0 to 300 mmHg, latex
  inflation bulb with a thumbscrew air-release valve.
- The deflation rate of 2 to 3 mmHg per second, from blood-pressure measurement guidelines (e.g. the American Heart
  Association's recommendations for blood pressure measurement in humans).

Every other figure is an estimate (task.json "fidelity"): the bulb's stiffness (20 N to squeeze flat over 20 mm), the
air volumes (how much each squeeze adds) and the valve's flow. No third-party meshes or scans: everything is built
from boxes and cylinders in task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Training arm | along y at x 0.500, upper arm 90 mm across from y -0.035 to +0.265; forearm and hand toward -y (fixed) |
| Cuff | 140 mm wide, 112 mm across, on the upper arm at y +0.130 (fixed) |
| Gauge | a large-dial clinic gauge, 90 mm dial facing the robot, centre (0.430, 0.130, 0.125 up); 0 to 300 mmHg over 270 degrees, ticks every 10 mmHg |
| Bulb | 100 x 60 x 55 mm, lying at (0.300, 0.100), long axis along y, tail and valve toward -y; loose |
| Valve | wing thumbscrew (18 mm across, 8 mm thick) 37 mm beyond the bulb's tail; opens counterclockwise seen from the tail; starts a quarter turn open, the wing upright |

The cuff, arm and gauge are one fixed mechkit mechanism. Only the gauge's needle moves, on a hinge driven to show
the pressure. The bulb is a loose mechanism:

- one half slides toward the other on a 20 N spring over 20 mm;
- the thumbscrew turns the valve on a damped hinge, from shut to half a turn open.

The joint is damped rather than given joint friction, because PhysX's joint friction scales with the force carried
through the joint and locked the valve while the bulb was held.

## The air (the one rule; scoring.Cuff)

Fed the bulb's two joints every control step, it keeps the cuff's pressure P (mmHg):

- **Squeeze.** A squeeze adds 0.9 mmHg per mm of stroke at 0 mmHg, scaled by (1 - P/400). It counts only past a
  0.5 mm deadband, so a bulb merely held pumps nothing. From 4 to 20 mm, 18 squeezes reach about 178 mmHg.
- **Valve.** The valve lets air out at 0.198 x f(theta) x sqrt(P) mmHg per second, where theta is the opening from
  shut in radians. f is linear up to 115 degrees and steep past it: about 57 degrees gives 2.5 mmHg/s at 160, 93
  degrees gives 2.5 at 60, and wide open empties the cuff from 60 in a few seconds.

The robot does the real physical work behind the rule: real squeezes against the bulb's spring, and turning the
thumbscrew itself.

## Scoring (scoring.py)

- **Peak:** the highest pressure at any time.
- **Descent:** from the first step the pressure is 2 mmHg below its peak (pumping higher starts it again) to its first
  fall through 60, in consecutive 1 s windows, plus one more window ending at the fall through 60. A window is on rate at a fall of 1.5 to 3.5 mmHg (2 to 3, read to +/-0.5), and gross below 1 (a pause or
  a rise) or above 5 (a dump).
- **Emptied:** below 20 mmHg within 10 s of passing 60.
- **1.0:** peak 170 to 200, at least 90% of the windows on rate, none gross, emptied (success, held 1 s).
- **0.5:** peak 160 to 220, at least 60% on rate, none falling more than 8.
- **0.25:** peak 160 to 220 otherwise.
- **0:** anything else, including a peak over 220.

Real trial: a calibrated digital pressure logger on a T in the cuff's line (0.1 mmHg, 10 readings a second; for
example a handheld digital manometer with data logging and a 3-way luer or barbed tube T) gives the pressure for
scoring; the robot works from the dial, as a nurse does. Lay the tube's slack to the left of the bulb, clear of the
valve.

## The demo (both arms)

Open loop, nothing read from the simulator.

1. **Hold the bulb.** The left hand takes the bulb in a straight-down grip, its pads on the two halves, and holds it
   up in front of the cuff.
2. **Shut the valve.** The right hand comes in level along the bulb's axis from the robot's right and pinches the wing
   on its flats. It rolls the wrist to turn the valve shut, then lets go: each squeeze shifts the bulb a little in the
   left hand's symmetric grip.
3. **Pump.** The left hand squeezes 18 times, 0.7 s each.
4. **Let it down.** The right hand pinches the wing again and turns it, in one direction only, to openings worked out
   from the air rule so that the needle falls 2.5 mmHg a second, from the peak down: one setting a second, a little further open as the
   pressure falls. The wrist gives under the pinch, and the valve ends about 25 degrees short of the hand's turn
   (measured on the simulator), so every opening is commanded 25 degrees further. Below 60 it opens the valve wide.

Counterfactual variants (DEMO_VARIANT, never the submitted demo):

- `dump`: opens the valve far at once after pumping.
- `overpump`: squeezes 22 times.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/bp-cuff-pump-deflate
INSTR="Take the inflation bulb, shut its valve, and pump the cuff up to between 170 and 200 mmHg on the gauge. Then open the valve just enough that the needle falls 2 to 3 mmHg each second, all the way down to 60, and then open it wide to empty the cuff."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/bp-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 2400 --store-frames --log-dir ~/yam-eval/bp-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step (pressure, valve opening, squeeze). No seeds: the layout is
fixed (`object_jitter=0`). Offline tests (no simulator):
`PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 2 minutes for the scripted demo (15 s to start Isaac Sim, then 1,676 steps) and
about 9 minutes for a Dream-1 run of 2,400 steps (the policy is queried over the network each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0: peak 178 mmHg; all 51 one-second windows from just below the peak to 60 fell 2.25 to 2.42 mmHg; emptied | 1,657 (success) |
| Dream-1 | 0.0: it turned the valve wide open and never pumped (peak 0.6 mmHg) | 2,400 (the limit) |
| `dump` | 0.25: half the windows fell faster than 3.5 mmHg | 1,245 |
| `overpump` | 0.5: peak 205 mmHg, over 200; the descent itself was on rate | 2,090 |

The demo scored the same on two runs (the simulator repeats a run exactly).
