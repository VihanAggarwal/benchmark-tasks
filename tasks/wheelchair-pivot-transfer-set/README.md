# Set a wheelchair for a pivot transfer: her strong side, 30 to 50 degrees to the bed

Nursing assistants, home health aides and personal care aides move residents from bed to wheelchair many times a
day. For a resident who can stand but has one weak side, as after a stroke, the usual method is the stand-pivot
transfer, and the chair is set up before she stands: beside the bed on her stronger side, right next to her, so that
she turns on her strong leg; close to the bed; and angled to it (commonly taught as 30 to 45 degrees; full credit
here for 30 to 50), facing the foot of the bed, so that she turns less than a quarter turn before she sits. The
common error is the mirror image: the chair parked on her weak side, where she would have to pivot on the weak leg.

Here a care-home room at 1:10 scale stands on the table: a bed across the far side, its head end to the robot's
left (marked by a pillow); Mrs Hale, a rigid seated manikin, on its near edge facing the arms, her left forearm in a
blue sling; and her wheelchair past the foot of the bed on her weak side. She faces the robot, so her strong right
side is the robot's left: the policy has to work that out from the instruction. The score is what a nurse educator
checks with a protractor and a tape on a top photo: the chair beside the bed next to her on her strong side (its
nearest corner at most 40 mm, 0.4 m at scale, along the bed past her), within 20 mm of the bed (0.2 m at scale),
its front toward the foot of the bed and turned out 30 to 50 degrees.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the wheelchair's pose, with a small tracker of its last half second |
| `mechkit.py` | Helper that builds the wheelchair as one rigid body of twelve coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets, the demo plan and its carry path (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and sphere (the bed, pillow and manikin), the standard `task_score` scorer, and one
helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no front or
back. `mechkit.py` (imported by `task.py`) builds the wheelchair from twelve boxes (two rear wheels, seat, backrest,
two side rails, two front legs, two castors, two footplates) as one loose rigid body through a hook on
`yam_arena.scene`, and reports its pose (the frame origin at the middle of its footprint, and its orientation) to
the scorer. It changes nothing else the embodiment does, and it stops the run with an error if the chair was not
built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook
patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner) follows the one first
shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms, a relay): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mrs Hale is sitting on the edge of her bed, about to move to her wheelchair; her left side is weak after a stroke. Set the wheelchair up for a stand-pivot transfer: beside the bed next to her on her strong side (its nearest corner at most 40 mm along the bed past her), within 20 mm of the bed's side (0.4 m and 0.2 m at this 1:10 scale), its front toward the foot of the bed, angled 30 to 50 degrees out from the bed's side, so she can stand, turn on her strong leg and sit down." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/wheelchair-pivot-transfer-set/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mrs Hale is sitting on the edge of her bed, about to move to her wheelchair; her left side is weak after a stroke. Set the wheelchair up for a stand-pivot transfer: beside the bed next to her on her strong side (its nearest corner at most 40 mm along the bed past her), within 20 mm of the bed's side (0.4 m and 0.2 m at this 1:10 scale), its front toward the foot of the bed, angled 30 to 50 degrees out from the bed's side, so she can stand, turn on her strong leg and sit down." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/wheelchair-pivot-transfer-set/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`zero_reason`, `set_down`, `upright`, `on_floor`, `still`,
`closeness`, `angle_credit`, `alpha_deg`, `gap_mm`, `min_corner_y_mm`, `centre_mm`, `tilt_deg`, `lowest_mm`,
`path_mm`, and the chair's raw origin, quaternion, lowest point and velocity). Videos: `inspect-robots video
<log>.json`. The scorer keeps one trial's record per process and starts it over at every reset (mechkit's reset
counter, or the simulator putting the chair back on its mark in one step); the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the chair's pose at the last step.
Its **footprint** is the 95 x 63 mm rectangle under its wheels and castors; its **front** is the footplate end.

Seen from above (x up the page, away from the arms; y to the left, so the head end is on the left):

```
     y 24.0      y 17.0   y 13.0   y 10.0          y 0
     head end       :        :        :            foot end
   +--[pillow]------:--------:--------:-------------+   x 49.0
   |                :       BED       :             |
   +----------------:--------:---[her shins]--------+   x 40.0: the bed's side face (g is measured to it)
                    :        :    Mrs Hale,
     the chair goes here:    :    facing the arms               the chair starts here
     every corner left of    :                                  (x 30.0, y -20.0), past
     y 13.0 (her strong      :                                  the foot end, on her
     side), the nearest one  :                                  weak side
     right of y 17.0         :

        left arm base (y +30.5)        x 0         right arm base (y -30.5)

   alpha 0:  front points right (to the foot end), parallel to the bed
   alpha 40: front points right and down (toward the arms), 40 degrees off the bed's side
   alpha 90: front points down (toward the arms), square to the bed
```

The angle **alpha** goes from the bed's side pointing to its foot end (-y) to the chair's front direction, positive
turned out toward the room (-x): 0 = parallel to the bed facing its foot, 90 = square to the bed facing the room,
negative = turned in toward the bed, over 90 = facing toward the head end. The **gap g** runs from the bed's near
side face (x 40.0 cm) to the footprint corner nearest it, along x. **Set down**: tilted 10 degrees or less, lowest
point within 3 mm of the table, and still (the footprint centre's path, with turns counted at the corners, under
5 mm over the last 0.5 s: positions, not the simulator's velocities, which chatter while a part is gripped).

| Outcome | Score |
|---|---|
| Not set down (tipped, lifted, still moving, on the bed or on her) | 0 |
| Any footprint corner at y below 13.0 cm (her strong-side line, just past her right arm): on her weak side, in front of her or across her. The start scores 0 | 0 |
| Its nearest corner more than 40 mm along the bed past that line (y over 17.0 cm): too far from her (0.4 m at scale) | 0 |
| A corner more than 3 mm past the line of the bed's side face (into the bed) | 0 |
| alpha below 0 (turned in toward the bed) or above 90 degrees (facing the head end) | 0 |
| g over 80 mm (0.8 m at scale) | 0 |
| Otherwise: closeness (0.5 if g <= 20 mm, else 0) + angle (0.5 if alpha is 30-50 degrees; 0.25 if 15 to under 30 or over 50 to 65; else 0) | 0, 0.25, 0.5, 0.75, 1.0 |

Examples, all beside her on her strong side: 10 mm from the bed at 40 degrees 1.0; at 20 or 60 degrees 0.75;
parallel (5 degrees) or square (85 degrees) 0.5; 40 degrees but 50 mm off 0.5; 20 degrees and 50 mm off 0.25;
parallel and 50 mm off 0. On her weak side, or more than 40 mm along the bed past her, at any angle and any
distance: 0. The 40 mm limit also keeps the whole chair beside the bed: for any angle from 0 to 90 degrees its
centre is then at most 57 mm further along (y 22.7 cm), short of the bed's head end (y 24.0 cm). The trial ends
early once 1.0 has held for 3 s. A simulator state with a non-finite number or a zero quaternion scores 0 for that
step and leaves the tracker as it was.

Unintended strategies earn nothing more than they should, as the tests show: the mirror image (the same 40 degrees
on her weak side, front toward the head end) is 0; so is the chair at any heading in front of her or across her
feet, parked backwards beside the bed (front toward the head end), parked at a good angle close to the bed but up by
the head end, out of her reach (its nearest corner more than 40 mm past her), tucked past the head end of the bed
close to the line of its side, left out in the room 200 mm from the bed, set on its side or upside down at the right
place, or resting on the bed. A chair held still 2 mm above the floor at the right place counts as set down (the
scorer sees only the chair; `task.json` fidelity says so).

## One arm alone

The scripted demo is a relay, because of reach: the chair starts 0.5 m across from the left arm's base and its place
beside the bed is 0.5 m across from the right arm's. In simulation on the GPU (one trial each) the scripted demo
scores 0 with either arm idle, as the offline kinematic check (`--one-arm`) predicted: with the left arm idle, the
right arm leaves the chair on the midline in front of the foot of the bed (footprint centre x 24.6 cm, y -0.2 cm,
its nearest corner at y -3.4 cm: 0, her weak side of the strong-side line); with the right arm idle, the chair never
moves from its start (0).

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips, every yaw in 15
degree steps, tool point 25 to 80 mm up): the left arm reaches no point over the chair's footprint where it starts,
so it cannot pick it up there. The right arm picks it up easily but reaches no point of the chair once it is within
20 mm of the bed on her strong side, at any angle from 0 to 90 degrees; it reaches anything only with the chair
about 50 mm from the bed (at 60 to 90 degrees) or 80 mm (at 0 to 50 degrees), and then only the chair's front
corner, 25 mm above the floor. So with straight-down grasps alone the right arm can earn at most 0.5 (the angle
credit, the chair about 80 mm from the bed, pinched by its front corner; holding it across its wheels, as the demo
does, it scores 0), and the left arm 0. We do not claim one arm cannot do better: an arm that pushes the chair with
its open fingers could slide it the last few centimetres toward the bed (the right arm) or drag it within reach
first (the left arm), and leaned grips are not ruled out. Both arms are used naturally: the right arm brings the
chair round in front of her and the left arm sets it beside the bed.

## Demo and run limit

The scripted demo takes 43.1 s (862 control steps). The right arm grips the chair across its two rear wheels (63 mm
outside, commanded 55 mm, closing over 1.5 s), 10 mm behind the middle of its footprint (6 mm from its centre of
mass), with the tool point 48 mm up (pad tips 35 mm up, on the wheels from 35 to 60 mm), lifts it 30 mm, carries it
to the midline turning it to face the arms, and sets it down there (footprint centre x 24.75 cm, y 0; this half-way
state scores 0, on her weak side of the line), then parks. The left arm then picks it up there the same way, carries
it along y 68 mm in front of her slippers, turns it out to 40 degrees, brings it in, lowers the last 28 mm over 2 s
and lets go by opening to 75 mm (the far pad 17 mm short of the bed's side): footprint centre x 33.5 cm, y 20.5 cm,
alpha 40 degrees, g 10 mm, its nearest corner at y 14.8 cm, 18 mm on her strong side of the line and 22 mm inside
the 40 mm limit. On the way, the lifted chair's footprint stays at least 45 mm from her slippers and shins and at
least 10 mm from the bed's side (tested).

Offline (kinematic check): worst inverse-kinematics error 0.09 mm; smallest joint margin 12.2 degrees (joint 4);
arm links at least 11.1 cm apart (capsule proxy), tool points at least 30.6 cm apart; pad tips never lower than 35
mm above the table. Success has held for its 3 s at step 805; the scripted policy stops at step 923. The run limit
is 2,400 steps (2 minutes), which leaves a policy room for a slower relay or a regrip.

On the GPU run the arm sagged a few millimetres under the 120 g chair, so the chair reached the floor slightly
before the slow descent ended and 1.0 held from about step 740 (37 s); the left gripper let go (38.1 to 39.6 s) and
started up before the 3 s hold ended the trial at step 800 (40.0 s). Final pose, from `TASK_TRACE`: footprint centre
x 33.45 cm, y 20.47 cm (aim 33.5, 20.5), alpha 38.6 degrees (aim 40; band 30 to 50), g 11.2 mm (aim 10; at most
20), nearest corner at y 14.79 cm (aim 14.8; band 13.0 to 17.0), tilt 0, still.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the wheelchair on its wheels with its
footprint centre at x 30.0 cm, y -20.0 cm, its front pointing toward the arms and to the left (toward -x and +y, 45
degrees from each). The bed, pillow and manikin are fixed. Nothing is consumed.

## Runtime per trial

About 1.5 to 2.5 minutes on an L40S once Isaac Sim is running (the demo's plan is 43 s of simulated time; its GPU
trial ended at 40 s, when success had held for 3 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 1:10 doll's bed (a painted wooden block 240 x 90 x 48 mm) and pillow, clamped to the table; a seated doll or a
printed manikin of the sizes in `task.json`, fixed to the bed's edge, with a blue sling on her left forearm; and a
1:10 model wheelchair about 95 x 63 mm and 85 mm tall, about 120 g (a printed or kit model; weight it if it is
lighter), with plain plates for rear wheels so a gripper can hold it across them. Lock its wheels or glue them, so it
stands where it is set. Every part is light and nothing is hot or sharp, so trials are safe. Score from a top photo:
a protractor laid along the bed's side for alpha, a rule square to the bed's side for g, and lines on the floor at
y 13.0 cm (her strong-side line) and y 17.0 cm (40 mm past it) for the side.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

The demo ended at step 800 with the chair at alpha 38.6 degrees, 11.2 mm from the bed, beside her on her strong
side. With either arm idle the chair ends on her weak side (left arm idle: on the midline; right arm idle: it never
moves). Dream-1 ran the full 2,400 steps: it slid the chair about 9 cm toward the midline without lifting it, and
left it on her weak side (footprint centre x 27.6 cm, y -11.3 cm): 0.
