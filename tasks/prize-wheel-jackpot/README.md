# Flick a tabletop prize wheel so it coasts a full turn and stops on JACKPOT

A 20 cm tabletop prize wheel stands upright facing the robot, on a stand clamped to the table. It has 12 segments, a
peg standing out of the rim at every boundary, and a pointer at 12 o'clock. The gold JACKPOT segment starts at 9
o'clock, a quarter turn counterclockwise of the pointer as the robot sees it. The robot has to spin the wheel
clockwise by a peg so that it coasts at least one full turn after the hand leaves it and stops with the pointer on
JACKPOT: a turn and a quarter, or that plus whole turns.

## Why it matters

Spin-to-win wheels are a fixture of fairs, school events, shops and trade shows. Landing a chosen segment is a
physical judgement:

- **Coast.** The wheel coasts and slows on its bearing after it leaves the hand.
- **Flick.** The flick has to give it just the right speed. Too soft and it stops short; too hard and it runs past.
- **Exit.** The hand has to get out of the way of the pegs coming round.

The skill is dosing an impulse by the speed of a stroke, then letting go and leaving the result to physics. That is
different from moving a part to a target and holding it there.

## Sources

Tabletop prize wheels of 8 to 12 inches with 12 segments, boundary pegs and a clicker pointer are sold widely for
events and classrooms. Bought wheels differ in their clicker and bearing, which decide how far a flick carries, so the
real trial uses a wheel built to the simulation instead:

- **Wheel:** a 200 mm disc of 12 mm birch plywood with a 150 mm x 2 mm aluminium disc screwed to its back (about
  320 g), 12 printed segments (JACKPOT gold), and a 10 mm square hardwood peg standing 20 mm out of the rim at each
  boundary. It turns on an 8 mm steel axle in two 608 ball bearings in a plywood stand clamped to the table.
- **No clicker:** the pointer hangs from the stand's top arm just clear of the pegs, as in the simulation.
- **Drag:** a 20 x 10 mm N42 disc magnet on an adjusting screw faces the aluminium disc. This eddy-current brake gives
  drag proportional to speed, the same kind of drag as the simulation (0.00117 N m s/rad).
- **Build dimensions** (mm; x toward the robot, z up from the table; the wheel's centre at height 130):
  - Base: 18 mm plywood, 200 (x) x 260 (y), clamped to the table with two G-clamps at its back edge.
  - Two uprights: 18 mm plywood, 60 wide x 175 tall, screwed to the base 40 apart (one each side of the wheel), each with
    a 22 mm hole 7 deep at height 130 holding a 608 bearing (22 x 8 x 7), pressed in.
  - Axle: 8 mm silver-steel rod 80 long through both bearings, held by two 8 mm shaft collars outside the uprights.
  - Wheel: the 200 mm plywood disc and the 150 mm aluminium disc are screwed together with four M3 screws on a 100 mm
    circle and fixed to the axle by an 8 mm flanged shaft hub (M3 grub screw), centred between the uprights, the
    plywood face toward the robot.
  - Pointer: a 3 mm acrylic strip 15 wide hung from a top arm (18 mm plywood bridging the uprights at height 250), its
    tip 3 mm in front of the rim at 12 o'clock and 5 mm clear of the pegs' path.
  - Brake: the magnet glued to the end of an M6 screw running in a threaded insert in the rear upright's extension, its
    face toward the aluminium disc at 60 mm from the axle, the gap adjustable from 1 to 10 mm (start at 4) and locked
    with an M6 nut.
- **Calibration and reset** (before each set of trials): spin the wheel by hand and check in 60 fps video that its
  speed halves in 0.85 to 0.95 s, which is a coast of about 1.28 turns from 1 turn per second (the simulated wheel:
  0.89 s). If not, move the magnet and check again, then lock the gap with its nut. Turn the wheel back to its start. No third-party meshes: everything is built from boxes and cylinders in
task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Wheel | 200 mm across, 12 mm thick, 300 g; centre (0.360, 0.000, 0.130 up); axle along x; front face at x 0.354 |
| Segments | 12, centred every 30 degrees; JACKPOT (gold) centred at 9 o'clock at the start |
| Pegs | 10 mm square, 20 mm out of the rim's edge, on every segment boundary |
| Pointer | hangs from the stand's top arm in front of the rim at 12 o'clock |
| Stand | clamped; base plate behind the wheel, a post and an arm over the top |

The stand and wheel are one fixed mechkit mechanism: the wheel is a spindle with viscous drag, 0.00117 N m s/rad.
Nothing is moved or latched by code.

## Scoring (scoring.py)

The wheel's turn is unwrapped step by step, clockwise as the robot sees it.

- **Touch:** a gripper's TCP or pad tips within 40 mm in front of the face and within the rim plus 35 mm. The coast
  is counted from the last step with a touch.
- **At rest:** turning slower than 0.02 rad/s (about 1 degree a second) for 1 s.
- **Segment:** the one at the pointer at rest.
- **Tiers:**
  - 1.0: JACKPOT after a coast of at least one full turn (success, held 1 s).
  - 0.5: a neighbour of JACKPOT after a full-turn coast.
  - 0.25: any segment after a full-turn coast.
  - 0: less than a full turn of coast (walked round, or stopped by hand), or not at rest.

Real trial: two synchronized 60 fps videos, one facing the wheel (its turn counted from a tape mark on the rim) and
one side-on at its height. The last touch is the last frame in which any part of a gripper contacts the wheel or a
peg; the simulation's distance zone stands in for that contact. Record the wheel's make and model with the trials.

## The demo (right arm)

Open loop, nothing read from the simulator.

1. **Set up.** Fingers closed, the hand comes down over the rim peg just below 3 o'clock, its finger tips on the peg's
   upper side.
2. **Flick.** It drives the peg down along the rim for 35 mm, then peels off outward and on for another 35 mm, all in
   0.25 s. The peel-off clears the pegs coming round. The first version of the stroke ended with the fingers inside
   the pegs' circle, and the next peg slammed into them.
3. **Exit.** It moves out to the right.

The wheel leaves the hand at about 5.5 rad/s and coasts a turn and a fifth.

Counterfactual variants (DEMO_VARIANT, never the submitted demo):

- `soft`: the same stroke 1.6 times slower.
- `walked`: pushes the peg slowly round a quarter turn and lets it stop.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/prize-wheel-jackpot
INSTR="Spin the prize wheel clockwise by one of its pegs so that it turns at least once after you let go and comes to rest with the pointer on the gold JACKPOT segment."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/wheel-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/wheel-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step (turn, speed, touch, coast, segment). No seeds: the layout
is fixed (`object_jitter=0`). Offline tests (no simulator):
`PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 30 s for the scripted demo (15 s to start Isaac Sim, then 256 steps: success at
step 237 and its 1 s hold) and about 5 minutes for a Dream-1 run of 1,200 steps (the policy is queried over the network
each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0: it left the hand at 5.7 rad/s, coasted 419.5 degrees and stopped 442 degrees from the start, the pointer on JACKPOT (its centre is at 450) | 237 (success) |
| Dream-1 | 0.0: it never touched the wheel | 1,200 (the limit) |
| `soft` | 0: it coasted 345 degrees, short of a full turn | 424 |
| `walked` | 0: the peg pushed round 35 degrees, no coast | 500 |

The demo scored the same on two runs (the simulator repeats a run exactly).
