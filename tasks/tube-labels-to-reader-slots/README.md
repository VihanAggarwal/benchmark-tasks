# Load blood tubes into an analyser sample rack, each barcode facing its reader slot

Chemistry and haematology analysers in hospital and research laboratories take sample tubes in racks, loaded by hand
tube by tube. Beside each rack position is a slot through which the analyser's barcode reader reads the tube's label;
makers' instructions require the barcode to show through that slot (Beckman Coulter UniCel DxC sample processing job
aid: "Ensure that bar code label is visible through the rack slot"; Abbott CELL-DYN 3200 operator's manual: racks set
"so that the Bar Code Reader can read the rack label and the tube labels through the rack slots"). A tube racked with
its label turned away is a no-read, and its result waits for an operator.

Here a dark grey five-position analyser sample rack is held in place on the midline, its reader slots in the wall
nearest the arms, and two bench tube racks in front of the arms hold four capped blood tubes whose white labels all
face sideways: two to the left and two to the right. The rack's positions are 28 mm apart, so neighbouring 16 mm tubes
stand only 12 mm apart: a gripper setting a tube down beside one already standing has room for its 8 mm pads only in
front of and behind the tube (fingers across the rack), not along the row. That fixes the grip's direction at the
place, and since the label must end facing the slot, it fixes the grip at the pick too: each tube has to be gripped
so that one quarter turn of the wrist, made the right way (half of the possible quarter turns leave the label facing
the back wall), leaves both the fingers across the rack and the label in the slot. Then the tube is lowered into a
24 mm position between standing tubes without knocking them, and let go. Only the tubes' final places and label
bearings, and that the grippers have let go, are scored.

What is sourced and what is ours: the slot rule (the two manuals above); five-position racks (Roche's cobas 6000 analyser
series takes 5-position racks); 16 mm tubes need no rack adapter on the Beckman Coulter AU480/680 (13.6 to 16 mm,
AU480/680 Racks and Sample Containers job aid, 2015). The rack's exact dimensions (24 mm positions, 28 mm pitch, 40 mm
walls, 18 mm slots) are our model, not taken from a maker's drawing; the tube's 20 g is our estimate (tube and closure
about 9 g plus a 10 mL fill).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four tubes' states, with a small tracker for their last half second |
| `mechkit.py` | The compound-body helper the tubes are built with (each tube, closure and label is one rigid body) |
| `rigstate.py` | Both grippers' positions and openings for the scorer's let-go rule (fed by the offline checker too) |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the let-go rule, the exploits we tried and the demo plan |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the racks are
`yam_arena`'s fixed boxes, and `mechkit.py` (next to `task.py`, imported by it) builds each tube as one loose rigid
body from two cylinders and a box when Isaac Sim starts. The standard `task_score` scorer reads `score()`.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the four blood tubes from the two bench racks into the analyser's sample rack, one tube per position. Stand each tube upright with its white barcode label facing out through its position's reader slot, in the rack wall nearest you, and let go of it. A label turned away from its slot cannot be read." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/tube-labels-to-reader-slots/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the four blood tubes from the two bench racks into the analyser's sample rack, one tube per position. Stand each tube upright with its white barcode label facing out through its position's reader slot, in the rack wall nearest you, and let go of it. A label turned away from its slot cannot be read." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/tube-labels-to-reader-slots/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: per tube `pos` (its position, 1 to 5, or none), `racked`,
`readable`, `why` (the reason it earns nothing), `bearing_err_deg`, `tilt_deg`, `bottom_mm`, `at_rest`, `released`,
`moved_mm`,
and every tube's `origins`, `quats` and `label_faces`. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four tubes' poses (each tube's base
frame: the centre of its bottom, its axis, and the direction its label faces) and both grippers' positions and
openings (`rigstate.py`), once per control step (20 Hz), and keeps each tube's last half second of positions: its
bottom, its closure's top and the middle of its label's face, so a tube still being lowered or spun in place is not
yet at rest. Positions are numbered P1 to P5 from the robot's left; which tube goes where does not matter and one
position stays empty. The label's bearing is its outward normal seen from above, compared with straight out through
the slot (-x); a label pointing straight up or down (a tube lying on its side) has no bearing and does not count.
A label counts as readable only when both of its vertical edges (the face's middle plus or minus 5 mm along the face, in y) lie inside the position's 18 mm slot, so no post hides part of the barcode: a centred tube may turn up to about 30 degrees, a tube 4 mm off centre only if its label is square to the slot (fails closed otherwise: `label_behind_slot_post`).
A tube counts only once it has been let go: each gripper's finger zone (rigstate's: every point the pads can occupy
round the gripper's axis at their present opening, from the pad-tip point 38 mm back toward the tool point; 18 mm in
radius closed, 29 mm at the demo's 36 mm release, 57 mm fully open) at least 20 mm from the tube (its body, closure
and label). The rule fails closed: if the arms cannot be read on a step, no tube counts on that step. A new episode is
recognised from mechkit's reset counter, or when the simulator puts all four tubes back on their bench-rack marks
(within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number, a
zero quaternion or a tube missing its base frame scores 0 for that step and leaves the scorer's record as it was.

| Outcome, per tube | Earns |
|---|---|
| Racked (bottom's centre inside a position's 24 x 24 mm opening and within 5 mm of the table, upright within 5 degrees, alone in that position), at rest, let go (both finger zones 20 mm or more away), label within 30 degrees of facing out through the slot and whole inside the 18 mm slot | 0.25 |
| Racked but its label turned away: sideways (not turned) or toward the back wall (turned the wrong way) | 0 |
| Racked, label facing out but partly behind a slot post (tube off centre and turned) | 0 |
| Racked but a gripper still holds it or is within 20 mm of it | 0 |
| Sharing a position with another tube (both earn nothing) | 0 |
| In a bench rack, elsewhere on the table, lying, upside down, leaning more than 5 degrees (a tube resting against a rack wall leans more than 8.6 degrees, the least angle at which it tips; in a 24 mm position it can lean at most 11.3), above the floor, dropped | 0 |
| Still moving (bottom, closure top or label face moved more than 2 mm in the last 0.5 s) | 0 |

The score is the sum: 0, 0.25, 0.5, 0.75 or 1.0, every one reachable (one to four readable tubes; the tests build
each). Success is all four readable (1.0), held for 1 s.

Unintended strategies earn nothing for the tubes concerned, as the tests show: racking the tubes without turning
them leaves every label sideways (0); turning them all the same way puts two labels toward the back wall (0.5); a
tube held still on a position's floor by a gripper, label in the slot, earns nothing until the gripper has let go and
moved 20 mm clear; a tube propped against the slotted wall at 11 degrees, its label in the slot, or held 6 mm above
the floor, earns nothing; a tube stood in front of a slot outside the rack, or turned to face the arms while still in
its bench-rack cell, earns nothing; two tubes in one position cancel each other. Tubes cannot be slid into the rack:
its walls are 40 mm tall.

## One arm alone

Each arm loads its own side's two tubes; nothing in the task needs two hands at once. Both offline and in simulation
(L40S, one trial each, job `20261122T010150-00` on the 24 mm positions at 28 mm pitch rack) the scripted demo scored
0.5 with `DEMO_IDLE_ARM=left` and 0.5 with `DEMO_IDLE_ARM=right`: the working arm's two tubes racked and readable, the
far side's two still in their bench rack, because the one-arm demo only loads its own side. Straight down, neither arm
reaches the other side's tubes: with the tool point at the 70 mm grip height, the left arm reaches y -15.0 cm at x 20 cm
and y -12.0 cm at x 26 cm, 50 and 80 mm short of the far tubes' axes at y -20 cm (inverse kinematics of the arm's URDF
within its joint limits). Tilted 15 to 20 degrees toward its own base, though, its tool point does reach the far tubes'
grip points, so a single arm using tilted grips could in principle rack all four and score up to 1.0 (not tried in
simulation). The task does not claim to need both arms; the two arms simply share the work, each loading its own side,
and alternate at the analyser rack. (All GPU scores are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 51.0 s (1020 control steps). Each arm racks its inner position first, then its outer one:
the left arm P2 then P1, the right arm P4 then P5 (P3 stays empty), so a tube is never carried past one already
standing in the rack. Only one gripper is over the analyser rack at a time: an arm holds its picked tube over its
bench-rack cell until the other has risen from the rack (and 1.0 s more). For each tube: over it with the fingers
open to 44 mm, down to 30 mm above the grip (1.0 s), the last 30 mm slowly (1.5 s), settle, grip across the 16 mm
tube (fingers commanded to 8 mm, over 1.5 s) with the tool point 70 mm above its bottom, so the pads hold bare tube
from 57 to 79 mm up, 7 mm above the label's top and 3 mm below the closure; lift it out (1.5 s) to carry with its
bottom 60 mm up (20 mm over the rack's walls); carry it over its position while the wrist turns it a quarter turn
(2.5 s), settle, lower until its bottom is 47 mm up (1.0 s), then slowly stand it in the position, let go with its
bottom 2 mm above the floor (2.5 s), settle, open the fingers to 36 mm (1.5 s), rise, and at the end move aside.
The grip's yaw (0 closes the fingers along y) changes by exactly a quarter turn between pick and place. Near the
midline the left arm needs negative yaws and the right arm positive ones (joint 6 turns +-120 degrees), so a tube
whose label starts on the arm's outer side (`tube_1`, label to the left, for the left arm; `tube_3`, label to the
right, for the right arm) is picked with the grip at 120 degrees and set down at 30 into an inner position (P2, P4)
while both its neighbours are still empty, and one whose label starts toward the midline (`tube_2`, `tube_4`) is
picked at 0 and set down at 90, fingers across the rack, into an outer position (P1, P5) beside the tube already
standing: its pads open toward the front and back walls (their tips 19 mm above them) and reach 10 mm to each side
of the tube's axis, 10 mm short of the standing neighbour, whose side is 20 mm away. Every release opens only 10 mm
past the tube on each side.

The let-go rule means success can start only once the last gripper has moved aside (rising straight up leaves its
pad tips 17 mm over the closure, still inside the 20 mm clearance), so the usual 1 s hold is enough and the trial ends
with the fingers clear. On the GPU (job `20261122T010150-00` on the current 24 mm positions at 28 mm pitch rack and
let-go scorer), the scripted demo reached and held 1.0 in 1,016 steps (50.8 s). The run limit is 1,800 steps (90 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: each tube back upright in its bench-rack cell,
label facing as stated in `setup` (x 20, y +20 cm: left; x 26, y +20 cm: right; x 20, y -20 cm: right; x 26, y -20 cm:
left); the racks are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 51 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

The analyser's own five-position sample rack, if its positions take 16 mm tubes, held in place by a clamp or a rack
tray; otherwise glue 6 mm and 4 mm acrylic or printed PLA strips to a base plate to this model: five 24 x 24 mm
positions in a row at 28 mm pitch, 4 mm dividers inside 6 mm outer walls, all 40 mm tall, with an 18 mm slot cut in
the front wall of each position from 6 mm up to the top. Two bench racks: blocks with two 30 x 30 mm cells each,
20 mm deep, 60 mm apart, clamped to the bench. Tubes: four 10 mL K2EDTA blood collection tubes, 16 x 100 mm with
lavender closures (BD Vacutainer or equivalent), empty and filled with red-dyed water to 20 g, never real samples,
closures pushed fully home and taped, the racks on a drip tray so a leak cannot reach the rig, each with a 10 mm wide
white printed barcode label from 10 to 50 mm up and a black line drawn across the top of its closure in line with the
middle of the label, so a label's bearing can be read from above even when the label itself is hidden under the
closure. A photo from above at the end shows the closure lines and that no finger is within 2 cm of a tube; a photo
from in front shows each barcode whole in its slot (a post covering part of it scores 0); a 60-degree sector template laid on a closure, pointed straight out
through the slot, settles a line near the 30-degree limit.

## Results in simulation

Measured on the current 24 mm positions at 28 mm pitch rack and let-go scorer with the commands above on an L40S
(Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each, job `20261122T010150-00`):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written, 1,016 steps) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`, 1,081 steps) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`, 871 steps) | 0.5 |
| Dream-1 (1,800 steps) | 0.0 |
