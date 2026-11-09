# Leave a walking frame ready before he stands: square in front, open side toward him

Nursing assistants, personal care aides and physical therapist aides get residents up from their chairs many times
a day. A resident who walks with a frame is taught to stand by pushing up on the chair's arms, not by pulling on
the frame (which can tip toward him), and then to take the frame's handgrips. So the frame is left ready before he
stands: on its four feet directly in front of his chair, square to it, its open side toward him so that he stands
up inside it, and a short step from his toes. The common errors are a frame left beside the chair, out of his
reach, or turned round with its crossbar toward him.

Here a care-home lounge at 1:8 scale stands on the table: a green armchair across the far side, facing the arms;
Mr Okafor, a rigid seated manikin, in it, his forearms on the chair's arms and his slippers on the floor; and his
walking frame (a U-shaped aluminium frame, open at the back) standing side-on to his left front, out of his reach.
The score is what a care supervisor checks on a top photo with a protractor and a rule: the frame standing on its
four feet, square to his chair with its open side toward him, centred on him, its back legs 10 to 50 mm (8 to 40 cm
at scale) in front of his toes. Each of the three conditions has a loose band that earns partial credit. This is a
one-arm task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the walking frame's pose, with a small tracker of its last half second |
| `mechkit.py` | Helper that builds the walking frame as one rigid body of seventeen coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets, the demo plan and its carry path (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and sphere (the armchair and the manikin), the standard `task_score` scorer, and one
helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no open
side. `mechkit.py` (imported by `task.py`) builds the walking frame from seventeen boxes (four rubber ferrules, four
legs, two top rails each in three pieces with a black handgrip in the middle, two lower side rails and a crossbar)
as one loose rigid body through a hook on `yam_arena.scene`, and reports its pose (the frame origin at the middle
of its footprint, on the floor, and its orientation) to the scorer. It changes nothing else the embodiment does,
and it stops the run with an error if the frame was not built as a compound body. The technique (a
`task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (right arm; the left arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mr Okafor is about to stand up from his armchair and walk with his walking frame. Leave the frame ready for him: standing on its four feet directly in front of him and centred on him, square to his chair, with its open side toward him and its back legs 10 to 50 mm (8 to 40 cm at this 1:8 scale) in front of his toes." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/walking-frame-sit-to-stand/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mr Okafor is about to stand up from his armchair and walk with his walking frame. Leave the frame ready for him: standing on its four feet directly in front of him and centred on him, square to his chair, with its open side toward him and its back legs 10 to 50 mm (8 to 40 cm at this 1:8 scale) in front of his toes." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/walking-frame-sit-to-stand/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`zero_reason`, `set_down`, `on_feet`, `still`, `square`,
`centred`, `toe_gap` (each `full`, `loose` or `out`), `n_loose`, `turn_deg`, `off_centre_mm`, `toe_gap_mm`,
`centre_mm`, `feet_mm`, `tilt_deg`, `path_mm`, and the frame's raw origin, quaternion, lowest point and velocity).
Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at
every reset (mechkit's reset counter, or the simulator putting the frame back on its mark in one step); the run
rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the frame's pose at the last step.
Its **footprint** is the 56 x 70 mm rectangle round the outer faces of its four legs; its **front** is the
crossbar side, and its back is open.

Seen from above (x up the page, away from the arms; y to the left):

```
          y +4.4          y 0          y -4.4
        +------------- backrest -------------+     x 47.0
        | arm |       torso, seat      | arm |     the armchair, x 39.5-45.5, facing the arms
        +-----+        thighs          +-----+
                    [shins, slippers]              x 35.0: his toe line (d is measured from it)

                  |                   |            back legs (open side toward him): d 10-50 mm,
                  |  the frame goes   |            so at x 30.0-34.0
                  |  here: square,    |            centred: footprint centre within 20 mm of y 0
                  |  centred on him   |
                  +---- crossbar -----+                      +--------+
                                                        open           | crossbar   the frame starts here:
                                                             +--------+            side-on, x 30.0, y -17.0

       left arm base (y +30.5)          x 0          right arm base (y -30.5)
```

**Turn**: the angle seen from above between the frame's forward direction (open side to crossbar) and -x (toward
the arms): 0 = square with its open side toward him, 90 = side-on (as it starts), 180 = crossbar toward him.
**Off-centre**: the footprint centre's distance from his centre line (y 0). **Toe gap d**: from his toe line
(x 35.0 cm) to the footprint corner nearest it, along x; with the frame square that is its back legs; negative
means a leg past the line, round his feet. **Set down**: each of its four feet has its lowest point within 3 mm of
the floor (so it stands on all four, not tipped onto two and not lifted more than 3 mm) and it is still (the footprint centre's
path, with turns counted at the corners, under 5 mm over the last 0.5 s: positions, not the simulator's
velocities, which chatter while a part is gripped).

| Condition | Full band | Loose band | Outside the loose band |
|---|---|---|---|
| Square | turn 15 degrees or less | over 15, up to 30 degrees | side-on, crossbar toward him, anything over 30 degrees: 0 |
| Centred | off-centre 20 mm or less | over 20, up to 50 mm | beside the chair, over 50 mm: 0 |
| Distance | d 10 to 50 mm (8-40 cm at scale) | 0 to under 10, or over 50 to 100 mm | a leg past his toe line, or over 100 mm away: 0 |

| Outcome | Score |
|---|---|
| Not set down (lifted, tipped, on its side, upside down, still moving), or any condition outside its loose band | 0 |
| All three conditions in their full bands | 1.0 |
| One condition only in its loose band | 0.5 |
| Two conditions only in their loose bands | 0.25 |
| All three only in their loose bands | 0 |

Band edges count as inside. Examples: square, centred and 30 mm from his toes 1.0; the same but turned 20
degrees, or 35 mm off centre, or 70 mm or 5 mm from his toes, 0.5; turned 20 degrees and 35 mm off centre, or 35
mm off centre and 70 mm away, 0.25; all three 0; side-on or crossbar toward him anywhere 0; 60 mm off centre 0;
120 mm away 0; round his feet 0. The start scores 0 (side-on, 170 mm off centre). The trial ends early once 1.0
has held for 3 s. A simulator state with a non-finite number or a zero quaternion scores 0 for that step and leaves
the tracker as it was.

Unintended strategies earn nothing more than they should, as the tests show: the classic error, the frame at the
right place but turned crossbar toward him, is 0; so is the frame left side-on beside his chair or square but
beside him, pushed in round his feet (its open back lets its back legs pass either side of his slippers), standing
with two legs on his slippers, left out in the room 150 mm away, or set on its side or upside down at the right
place. A frame standing level where it scores more than 0 never touches him or the chair (tested over 300 random
placements), so the scorer does not need to see contacts. A frame held still 2 mm above the floor at the right
place counts as set down (the scorer sees only the frame; `task.json` fidelity says so).

## One arm alone

This task needs one arm, and the demo uses one: the right arm picks the frame up at his left front, turns it
square on the way and sets it down in front of him, while the left arm holds its ready pose. In simulation on the
GPU (one trial each, see Results in simulation), `DEMO_IDLE_ARM=left` is therefore the demo itself and scored 1.0,
and `DEMO_IDLE_ARM=right` leaves nothing to move and scored 0.0 (the frame stayed side-on where it started), as
the offline kinematic check predicted. The right arm alone does the whole task.

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips, every yaw from -90
to +90 degrees in 15 degree steps, tool point 25 to 127 mm up): the left arm reaches no point over the frame's
footprint where it starts (an 8 x 6 grid over x 26.5-33.5, y -19.8 to -14.2 cm), so it cannot pick it up there.
Low down (tool point 25 mm up) it reaches to about y -16 cm just in front of the frame (x 20-22 cm), close enough
for an open fingertip to touch the frame's near side, so a left arm that pushes or drags the frame toward the
midline and then picks it up is not ruled out. We make no claim that the task needs both arms: its difficulty is
reading the rule (which side of the frame is open, where his toes are, how far a short step is) and setting the
frame down square.

## Demo and run limit

The scripted demo takes 21.1 s (422 control steps). The right arm comes down over the side-on frame with its
fingers closing along x, slowly over the last 20 mm, to a tool point 107 mm up (pad tips 94 mm up, 4 mm below the
handgrips' underside); grips it across its two black handgrips (70 mm outside, commanded 62 mm, closing over
1.5 s), 2 mm forward of the middle of its footprint, at its centre of mass along its depth (the centre of mass is
61 mm up, hanging 41 mm below the grip); lifts it 20 mm; carries it in front of him over 3.5 s while turning the
grip 90 degrees, which turns the frame from side-on to square; lowers it 18 mm over 2 s; settles for 0.6 s; opens
over 1.5 s with its feet aimed 2 mm above the floor; rises straight up to 16 mm over the frame's top; and parks.
The plan sets the frame's footprint centre at x 29.2 cm, y 0, square, its back legs at x 32.0 cm: d 30 mm, in the
middle of the 10-50 mm band, and 20 mm inside the off-centre and 15 degrees inside the turn limits. In the GPU run
it ended at x 29.0 cm, y -0.2 cm, turned 1.1 degrees, d 31 mm, all four feet on the floor.

Offline (kinematic check): worst inverse-kinematics error 0.08 mm; smallest joint margin 9.9 degrees (joint 4);
the two arms' links at least 10.2 cm apart (capsule proxy), tool points at least 33.2 cm apart; pad tips never
lower than 94 mm above the table. On the way, the lifted frame's footprint stays at least 28 mm from his slippers
(seen from above; its feet are 20 mm up and the slippers 9 mm tall), 51 mm from his shins and 64 mm from the chair
(tested). Success has held for its 3 s at step 364; the scripted policy stops at step 483. The run limit is 1,200
steps (1 minute), which leaves a policy room for a slower carry or a regrip.

In the GPU run the frame's feet came within the 3 mm floor tolerance near the end of the slow descent (step 301,
1.6-2.8 mm up), while the gripper still held it, and reached the floor as the fingers opened. The 3 s success hold
(rather than the usual 1 s) is longer than the settle and the opening together (2.1 s), so the trial ended at step
360 with the fingers open and the frame standing on its own.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the walking frame on its four feet with
its footprint centre at x 30.0 cm, y -17.0 cm, side-on: its crossbar toward -y (the robot's right), its open side
toward +y. The armchair and manikin are fixed. Nothing is consumed.

## Runtime per trial

On an L40S once Isaac Sim is running: about 1 minute for the scripted demo (it ends at step 360, 18 s of simulated
time) and about 3 minutes for a full 1,200-step Dream-1 trial (1 minute of simulated time). The first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 1:8 armchair (wooden blocks covered in green fabric, the sizes in `task.json`) and a seated manikin of the sizes
given, 3D-printed in PLA and painted, fixed to the table with his slippers on the floor; and a 1:8 walking frame made from
8 mm square aluminium tube (70 mm across, 56 mm deep, 106 mm tall, open at the back, about 60 g), with rubber feet
and black foam sleeves on the middle of its top rails. Mark his toe line (x 35.0 cm) and his centre line (y 0) on
the floor. Every part is light and nothing is hot or sharp, so trials are safe. Score from a top photo: a
protractor against a line along y for the turn, a rule square to the toe line for d, and a rule from the centre
line for the off-centre distance.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

The scripted demo reached 1.0 at step 301 and ended at step 360 after holding it for 3 s. Settling with no plan
(160 steps), the frame stayed on its mark on its four feet. Dream-1 ran the full 1,200 steps and left the frame
side-on where it started (moved 1.7 mm, turned 87 degrees from square), so it scored 0.
