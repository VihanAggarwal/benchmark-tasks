# Grind five turns of pepper over a plate with a twist mill, then stand the mill back up

A 12 cm wooden twist pepper mill stands on the table to the robot's left; an empty dinner plate lies in front of the
right arm. The robot has to grind pepper over the middle of the plate, five full turns of the mill's head, holding
the mill clear above the plate, and then stand the mill back up on the table, off the plate, and let go.

## Why this is everyday

Seasoning food at the table with a twist mill is something nearly everyone has done and recognizes on sight. The
hand's problem is real and bimanual: the head grinds only when it turns against the body, and the burrs resist, so
one hand must hold the body while the other twists the head; the mill has to be held over the food, not set on it,
because the pepper falls out of the base; and a wrist twist covers well under a full turn, so each turn takes a
regrip. The YAM's wrist turns about 225 degrees at a stretch, so a turn of the head takes two strokes.

## Sources

- Mill: Peugeot Paris u'Select 12 cm, natural beech (Peugeot Saveurs product range,
  https://uk.peugeot-saveurs.com/en/paris-u-select-manual-pepper-mill-chocolate-finish-beech-wood-12-cm.html).
  Retailer listings give 12 cm tall, 5.9 cm across and 0.19 kg (Marks & Spencer,
  https://www.marksandspencer.com/paris-u-select-wooden-pepper-mill/p/hbp23094498; Butch,
  https://www.butch.de/en/new-pfeffermuehle-8006.html). It grinds when the head is turned clockwise seen from above.
- Plate: IKEA OFTAST plate 25 cm, art. 302.589.13 (https://www.ikea.com/gb/en/p/oftast-plate-white-30258913/):
  25 cm across, 2 cm high, 0.36 kg, tempered opal glass.

Every figure not listed above is an estimate (task.json "fidelity"): the mill's profile (stacked cylinders from
product photos), the split of its mass between body and head, the peppercorn load (about 25 g) and the grinding
resistance (0.15 N m of joint friction; no published figure).

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Mill (base centre) at the start | x 0.205, y +0.165, on the table |
| Mill profile (mill frame, mm) | foot r 29.5 (0-12), lower body r 26 (12-52), waist r 22 (52-72), shoulder r 26 (72-79); head: crown r 29 (80-110), knob r 11 (110-120) |
| Head joint | the mill's own axis, free (no limits), 0.15 N m friction, 0.005 N m s/rad damping |
| Plate centre | x 0.225, y -0.060; 250 mm across, 20 mm high |

The mill is one mechkit articulation: the body (loose, 170 g with the peppercorns) and the head (45 g) on a spindle
joint. mechkit's hinges are limited to plus or minus one turn, so this copy of mechkit adds `spindle()`, a revolute
joint written without limit attributes (UsdPhysics reads that as free); the scorer unwraps its angle step by step.
Nothing is latched or moved by code.

## Scoring (scoring.py)

Head turns are counted relative to the body, clockwise seen from above; turning back counts for nothing and undoes
nothing. A turn counts only on steps where the mill is over the plate: its base's centre within 80 mm (in plan) of
the plate's centre, at least 5 mm above the plate's top, and the mill within 30 degrees of upright.

- 0.15 per whole turn so counted, at most five (0.75).
- Plus 0.25 (1.0) when, after five, the mill stands upright (within 10 degrees) on the table with its axis at least
  160 mm from the plate's centre, both grippers 30 mm or more from it, for the last 20 steps, and still (2 mm).
- 0 for the whole run if at any moment the mill tips past 60 degrees, goes off the table, or the plate is pushed 30
  mm or tipped 10 degrees.

In a real trial: a dot of paint on the crown and a pencil line on the shoulder below it; count dot-past-line turns
on the video while the base is over the plate's middle and clear of it (the top camera; the plate's outline gives
the 80 mm circle), then check that the mill stands upright off the plate.

## The demo (both arms)

Open loop: timed waypoints from the known start layout.

1. Left: the straight-down grip pitched 70 degrees about world y reaches in from the robot's side and closes on the
   lower body 49 mm above the base, the whole mill's centre of mass (held lower, the top-heavy mill rolls over in the
   pads). It lifts the mill 65 mm, carries it over the plate's centre, lowers it until the base is 25 mm above the
   plate's top, and holds it there.
2. Right: straight down, the pads either side of the crown, their tips 4 mm below it. Twenty strokes: close, turn
   110 degrees clockwise (yaw +55 to -55 degrees, in two steps), open, turn back. The two fingers then sweep two 110
   degree arcs either side of the robot and never the 70 degree gap toward the robot, where the left gripper is.
   2200 degrees commanded; the pads slip about 8 degrees a stroke, so the head turns about 2020 degrees (5.6 turns).
3. Right: up off the crown and clear. Left: carries the mill back to its start spot, stands it down, lets go, backs
   out 50 mm along its approach and lifts 180 mm, clear of the mill.

Each finger is driven to its target on its own. Told to close right up, both saturate at their effort limit on the
part and nothing centres it between them, so the part drifts along the closing axis (in the first runs the mill slid
21 mm off centre in the left grip and the right fingers then missed the crown). The demo closes each finger to 3 mm
inside the part instead, so the pads squeeze at about 15 N each and their position springs keep the part centred,
as the real gripper's single rack does.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `on_table` grinds with the mill standing on its
start spot (0); `three_turns` makes 11 strokes, about three turns (0.45); `leave_on_plate` sets the mill down on the plate
(0.75).

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/grind-pepper-over-plate
INSTR="Grind pepper over the middle of the plate: five full turns of the mill's head. Then stand the mill back up on the table, off the plate."
# the scripted demo
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 2000 --store-frames --log-dir ~/yam-eval/pepper-demo
# Dream-1
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 2000 --store-frames --log-dir ~/yam-eval/pepper-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step (what the scorer saw and decided). No seeds: the layout is
fixed (`object_jitter=0`). Offline tests (no simulator; numpy and the pure-Python yam_arena package):
`PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 4 minutes for the scripted demo (16 s to start Isaac Sim, then 1,614 steps to
success) and about 6 to 9 minutes for a Dream-1 run of 2,000 steps (the policy is queried over the network each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0 (2025 degrees ground over the plate, 5 turns; stood back up, let go) | 1,614 (success) |
| Dream-1 | 0.0 (no turn ground; the mill was not lifted) | 2,000 (the limit) |
| Variant `on_table` | 0 (the head turned about 5.6 times, none of it over the plate) | |
| Variant `three_turns` | 0.45 (1114 degrees: 3 turns) | |
| Variant `leave_on_plate` | 0.75 (5 turns; left on the plate) | |

The demo scored the same, to the tenth of a degree, on two runs.
