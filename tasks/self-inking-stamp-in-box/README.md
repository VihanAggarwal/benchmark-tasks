# Stamp RECEIVED inside the box on a delivery note with a self-inking stamp

A self-inking RECEIVED stamp stands on the table to the robot's right. A delivery note lies in front of the robot,
with a GOODS RECEIVED box printed near its lower right corner. The robot has to pick up the stamp, set it down on the
box square and the right way up, push its handle all the way down so it prints, let it spring back, and stand it
back where it stood.

## Why it matters

Stamping paperwork is one of the most familiar office acts. Most people have done it at a receiving desk, post room,
library or school office. A self-inking stamp only prints when its handle is pushed all the way down against its
spring: that flips the die off the ink pad inside and onto the paper. Pressed short, it prints nothing. Set down off
the mark or crooked, it prints across the box's lines. Turned round, it prints upside down. Done well, it takes
three things together: carrying the stamp square, placing its 47 x 18 mm die inside a 60 x 30 mm box, and pushing
down the full 22 mm against the spring without rocking the base.

## Sources

- Stamp: Trodat Printy 4912, imprint 47 x 18 mm (maker's catalogue, e.g. https://www.trodat.net, Printy 4912).
- Delivery note: A4 paper, 297 x 210 mm.

Every figure not listed above is an estimate (task.json "fidelity"): the stamp's outer size (66 x 30 mm, 58 mm tall),
mass (65 g), handle travel (22 mm) and spring (about 12 N at the bottom), the 3 mm card backing, the printed box and
text, and every friction. No third-party meshes or scans are used: every object is built from boxes in task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Stamp | standing upright, base centred x 0.200, y -0.180; long side along y; front label toward the robot (-x) |
| Delivery note | x 0.1715 to 0.4685, y -0.105 to +0.105, 3 mm thick |
| GOODS RECEIVED box (inside) | centre x 0.250, y -0.060; 30 mm along x by 60 mm along y |

The stamp is a loose mechkit mechanism: the base (60 x 24 x 26 mm) whose sole prints, and the handle (a cap and four
skirt walls) on a sprung slider that goes 22 mm down over it. The sheet is a loose compound with the box, heading and
text lines printed on it. The imprint (a 0.3 mm red print with eight letters) waits on the floor out of view.

What is a rule, not physics: the die's flip and the ink. When the handle bottoms out with the base flat on the sheet,
the print is recorded where the die's footprint lies. Once the stamp has been lifted 15 mm clear, the red imprint is
placed there and held on the sheet (a mechkit weld, rewritten every control step). The robot still has to do the real
work: carry the stamp square, set it flat on the mark, and push the handle the full travel against the spring.

## Scoring (scoring.py)

- Print: the handle within 2 mm of the bottom of its travel while the base lies flat on the sheet (within 1.5 mm
  of its surface, tilted 3 degrees or less), held 0.2 s. The handle has to rise above 10 mm before another press can
  print.
- 1.0: exactly one print, the whole imprint inside the box, within 5 degrees of square and the right way up, the sheet
  moved 5 mm or less during the press, and the stamp standing upright (5 degrees) within 30 mm of its spot, both
  grippers 30 mm or more from it, still (2 mm) for the last second.
- 0.75: that print, the stamp not back or still held.
- 0.5: one print the right way up, at least half inside the box, within 15 degrees of square.
- 0.25: one print on the sheet, the right way up.
- 0: no print, two or more prints, an upside-down print, and the whole run if the stamp tips past 45 degrees.

In a real trial: the inked imprint against the box on a photo of the sheet (corners inside the lines; angle against
the box's edge); pencil marks at the sheet's corners and at the stamp's spot for movement; the press and the return
from the trial video. A short press leaves no imprint, so "printed" is read off the paper itself.

## The demo (right arm)

Open loop, the right arm only; the left stays parked. The press is straight down, so the sheet does not need holding,
and the task does not claim to need two arms.

A straight-down grip on the handle's cap, the pads closing on its two long sides. The stamp is lifted 60 mm, carried
over the box, lowered square to it (the front label toward the robot), and set down on the paper. The hand then pushes
the handle down its full 22 mm travel, aiming 25 mm past the bottom: the arm gives under the spring's 12 N, and the
pad tips always stay 3 mm or more above the paper. It holds the handle down 0.8 s, lets it spring back, lifts the
stamp off, and stands it back on its spot.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `half_press` pushes the handle only half way, so
nothing prints (0); `skewed` sets the stamp down 12 degrees off square (0.5); `not_back` prints well but keeps the
stamp in hand (0.75).

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/self-inking-stamp-in-box
INSTR="Stamp RECEIVED inside the box marked on the delivery note, the right way up, then put the stamp back where it stood."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1000 --store-frames --log-dir ~/yam-eval/stamp-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1000 --store-frames --log-dir ~/yam-eval/stamp-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step. No seeds: the layout is fixed (`object_jitter=0`).
Offline tests (no simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 40 s for the scripted demo (15 s to start Isaac Sim, then 422 steps to success)
and about 4 minutes for a Dream-1 run of 1,000 steps (the policy is queried over the network each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0 (one print, its centre 0.5 mm and 0.6 mm from the box's centre, 1.9 degrees off square, the right way up; sheet moved 0.01 mm; stamp back on its spot) | 422 (success) |
| Dream-1 | 0.0 (the stamp was not moved; no print) | 1,000 (the limit) |
| `half_press` | 0 (handle 5.7 mm down: no print) | 510 |
| `skewed` | 0.5 (print 13.9 degrees off square) | 510 |
| `not_back` | 0.75 | 410 |

The demo scored the same, to the hundredth of a millimetre, on two runs.
