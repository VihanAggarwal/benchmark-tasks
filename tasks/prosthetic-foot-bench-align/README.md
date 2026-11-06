# Bench-align a prosthetic foot: 30 mm ahead of the laser line, toes out 5 degrees

Set a right prosthetic foot on an alignment jig with the middle of its length 30 mm in front of the jig's laser
reference line, its toes turned out 5 degrees and its heel raised on a block: the first step of bench alignment.
Every new below-knee prosthesis is bench-aligned in a jig before the first fitting, by prosthetists (O*NET
29-2091.00) and prosthetic technicians (51-9082.00), following the foot makers' instructions: the heel raised on a
block to the shoe's effective heel height (heel height minus forefoot sole thickness) plus 5 mm, the middle of the
foot about 30 mm in front of the reference line the jig's laser projects, the toes turned out about 5 degrees; the
socket is then built up over the laser line. A foot set too far back throws the knee into flexion, too far forward
pushes it back, and a wrong toe-out loads it sideways. Here a grey jig plate with an inlaid red laser line (across,
x 27.0 cm) and a dark centre line (along, y 0) is fixed on the midline, an orange 20 mm heel block lies in front of
the left arm, and a right prosthetic foot (a 220 mm flat-soled shell with a black adapter block on top) lies in
front of the right arm. The block goes flat on the plate under where the heel will be; the foot goes on, heel on the
block and toe on the plate (it rests pitched about 7 degrees, toe down), its middle 30 mm in front of the laser
line, its heel on the centre line and its toes turned 5 degrees toward the robot's right (-y, the outside of a right
foot); then both grippers let go and move clear. Here the shoe's effective heel height is 15 mm, so the block is
20 mm tall. The score is the end state, measured from the poses as a ruler and a photo would measure it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the foot's, the block's and the grippers' states, with a small tracker for stillness and resets |
| `mechkit.py` | Builds the foot (one loose rigid body of three boxes) and reports its pose to the scorer (copied unchanged from our kit) |
| `rigstate.py` | Reads both grippers (tool point, pad-tip point) for the scorer each control step (copied unchanged from our kit) |
| `standin.py` | Offline only: how the foot comes to rest on the block and the plate, so the offline check can score the demo (never used in the simulator) |
| `tests/test_scoring.py` | Offline tests of every tier and zero condition at its boundary, stillness, resets, the stand-in, the hooks and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed and loose boxes, the standard `task_score` scorer, and helper modules next to `task.py`.

`yam_arena` has no compound objects and does not pass the grippers' state to a task file's hooks. `mechkit.py`
(imported by `task.py`) builds the foot from USD prims through a hook on `yam_arena.scene` and reports its pose;
`rigstate.py` wraps the embodiment's step and reset once, at import, to take a snapshot of both arms before each
control step and to start a new trial at each reset. Neither changes anything else the embodiment does. If the arms
cannot be read, `rigstate` prints why once; nothing then counts as let go on those steps, and the trace counts them
(`arms_unavailable`). `mechkit` stops the run with an error if the foot was not built as an assembly.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Bench-align the right prosthetic foot on the grey jig plate. First put the orange heel block flat on the plate where the heel will rest. Then set the foot on the plate, sole down, with its heel on the block and its toe on the plate: the middle of the foot's length 30 mm in front of the red laser line, the heel on the dark centre line, and the toe end turned out about 5 degrees toward your right. Then let go and move both grippers clear of the foot and the block." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/prosthetic-foot-bench-align/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Bench-align the right prosthetic foot on the grey jig plate. First put the orange heel block flat on the plate where the heel will rest. Then set the foot on the plate, sole down, with its heel on the block and its toe on the plate: the middle of the foot's length 30 mm in front of the red laser line, the heel on the dark centre line, and the toe end turned out about 5 degrees toward your right. Then let go and move both grippers clear of the foot and the block." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/prosthetic-foot-bench-align/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (the settle check: nothing may move); `TASK_TRACE=trace.jsonl`
writes one JSON line per control step with what the scorer saw and decided: the gate flags (`g1_sole_over_plate`,
`g2_sole_down`, `g3_toes_forward`, `g4_heel_on_line`, `g5_released`, `g6_still`), the checks (`a_forward`,
`b_toe_out`, `c_heel_on_block`), `ap_mm`, `toe_out_deg`, `heel_y_mm`, `heel_h_mm`, `toe_t_mm`, `foot_tilt_deg`,
`block_along_mm`, `block_side_mm`, `block_front_mm`, `block_tilt_deg`, `block_bottom_mm`, `block_shift_mm` (the
block's centre from where the demo sets it down), `grip_clear_mm`, the three 0.5 s paths, both tool points and
openings.
`MECHKIT_TRACE=mech.jsonl` adds the foot's raw state per step. Videos: `inspect-robots video <log>.json`. The scorer
keeps one trial's record per process and starts it over at every reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step the scorer reads the foot's pose, the
block's pose and each gripper's tool point and pad-tip point (13 mm past it). From the foot's pose, never its
bounding box: H and T are the middles of the bottom edges of the heel end and the toe end, M is halfway between
them, and "heel to toe" is the direction from H to T seen from above.

- **Gate** (all needed, else 0): all four sole corners over the plate seen from above; sole down (the foot's up
  axis within 15 degrees of vertical); heel to toe within 20 degrees of +x; H within 15 mm of the centre line
  sideways; let go and moved clear, as the instruction asks (each arm's tool point and pad-tip point 20 mm or more
  from every box of the foot and from the block; arms that cannot be read count as not let go); still (H, T and
  the block's centre each moved less than 2 mm, as a path, over the last 0.5 s).
- **(A) Forward**: M 25 to 35 mm in front of the laser line's centre (x 27.0 cm).
- **(B) Toe-out**: heel to toe turned 2 to 8 degrees from +x toward -y, the robot's right. With a ruler: the y of H
  minus the y of T is 3.5 to 14.1 % of the x of T minus the x of H (7.6 to 30.6 mm over the pitched foot's 218 mm).
- **(C) Heel on the block**: the block flat on the plate (its 20 mm axis within 10 degrees of vertical, either face
  down, its lowest point within 3 mm of the plate's top); its centre within 35 mm of the line through H and T
  sideways and no more than 30 mm behind H along that line; none of its corners, seen from above, more than 90 mm
  in front of H along that line (4 mm behind the foot's centre of mass, 94 mm from H, so a foot resting heel on the
  block and toe on the plate stays there by itself; for a block square to the foot this is its centre at most 60 mm
  in front of H, for one turned 90 degrees at most 55 mm); H 18 to 40 mm above the plate's top; T within 4 mm of
  it.

| Outcome | Score |
|---|---|
| Gate not met: a sole corner off the plate, the foot tipped over 15 degrees (on its side, upside down), heel to toe over 20 degrees off +x (backwards, across), H over 15 mm off the centre line, a tool or pad-tip point within 20 mm of the foot or the block (still held, or opened but not moved clear), or H, T or the block moved 2 mm or more over the last 0.5 s | 0 |
| Gate met, and A, B and C | 1.0 |
| Gate met, and two of A, B and C (for example the foot aligned but no block under it) | 0.75 |
| Gate met, and one of A, B and C | 0.5 |
| Gate met, none of A, B and C | 0.25 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0; every end state gets exactly one. The trial ends early once 1.0 has held
for 1 s (which needs the grippers let go); otherwise the end state at the run limit is scored. A new trial is
recognised from rigstate's and mechkit's reset hooks or, as a fallback, when the simulator puts the foot and the
block back on their starting places in one step. A simulator state with a non-finite number, a zero quaternion or a
missing field scores 0 for that step and leaves the record of the trial as it was. The block is the same either
60 x 70 mm face down, so its tilt is folded; the foot is not, so upside down and backwards are 180 degrees off.

Unintended strategies earn nothing, or only what the text says, as the tests show: holding the foot in the right
pose is 0; pushing the foot along the table to the plate leaves it against the plate's side, not on it, 0; the foot
on its side or backwards at the right spot is 0; the block under the toe (the foot then rests heel down, toe up) or
under the midfoot (the foot lies flat on it, toe in the air) gives no (C); the block stood on end under the heel
tips the foot past 15 degrees, 0; the block where the foot belongs and the foot left where it lay is 0; toed in, or
square, gives no (B).

Letting go means moving clear, and the instruction says so: a gripper opened wide round the adapter but not lifted
away (its pad-tip point within 20 mm of the adapter) has not finished the task, and the gate gives 0 until it is
20 mm clear. The scorer reads only each gripper's tool point and pad-tip point (`rigstate` gives no heading), so a
wide-open finger lying on the foot while both points are 20 mm clear is not seen: an aligned foot with a finger
resting on its forefoot, the points 40 mm from the shell, scores 1.0 (a test shows it). It cannot hold a foot where
it would not stay: with every corner of the block at most 90 mm in front of H, 4 mm behind the centre of mass, a
foot heel on the block and toe on the plate stays so by itself. A block turned on the plate so that it reaches past
the centre of mass leaves the foot lying flat on it, toe in the air; pinned toe-down by a finger, that pose still
gets no (C), since the block's corners are then more than 90 mm in front of H. A finger could only matter by
propping the heel above the block, a deliberate hold beside the foot.

A real trial is scored the same way: mark H, T and M on the foot's sides and square them down to the plate, measure
M from the laser line, H and T from the centre line and the block's front corners from H along the foot with a
ruler, the heel's height above the plate with a ruler, and take a photo from above. A block whose front edge is
more than 90 mm in front of H gets no (C) even where the foot would still balance toe-down on it (to about 94 mm,
the centre of mass); the 4 mm keep that choice clear of the tipping point.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) sets the foot down at 30 mm and 5 degrees with no block under it: it
drops flat onto the plate (on the GPU M 30.0 mm, toe-out 4.3 degrees, the block still where it started), (A) and
(B) but not (C): 0.75. `DEMO_IDLE_ARM=right` (the left arm alone) puts the block on the plate and the foot never
moves (H still 380 mm off the centre line): 0. Both are GPU results (see 'Results in simulation' below) and match
the offline predictions.

The layout gives each arm one part, but this task does not need two arms, and no claim is made that it does.
Straight down, the left gripper reaches toward -y only to y -0.12 to -0.13 at the adapter's grip height (the
adapter's centre is at y -0.380, the shell's near side at -0.345); tilted toward -y it reaches -0.30 to -0.32 at
40 degrees, -0.34 to -0.36 at 50 degrees and -0.38 to -0.39 at 60 degrees (x 0.25 to 0.275, pad tips 36 mm up), so
a left gripper tilted about 60 degrees could just take the adapter, and one arm could then do the whole task. The
right gripper is the mirror image: straight down it reaches toward +y only to y +0.14 to +0.16 at the block's grip
height (pad tips 8 mm up; the block spans y +0.365 to +0.435), tilted toward +y +0.32 to +0.34 at 40 degrees,
+0.36 to +0.38 at 50 degrees and +0.39 to +0.41 at 60 degrees (x 0.22 to 0.25), so a right gripper tilted about 60
degrees could just pinch the block's near half. (Pose-only inverse kinematics with our offline arm model, no
collisions; the scripted demo grips straight down only.) With both parts on one side, one arm would score 1.0.
Raking the block over with the carried foot is conceivable but contrived.

## Demo and run limit

The scripted demo takes 33.0 s (660 control steps). Grips close 8 mm narrower than the part over 1.5 s; the last
2-3 cm of every descent take 1.5-2 s.

1. The left arm (fingers closing along y) grips the heel block across its 70 mm with the pad tips 8 mm above its
   bottom, lifts it, carries it over the plate and sets it down 1 mm above the plate centred at x 22.0 cm, y 0
   (it spans x 19.0 to 25.0 cm, under where the heel will be), lets go, goes up and parks at x 25 cm, y +30 cm, 15 cm
   up (20.5 s).
2. Meanwhile the right arm (fingers closing along y, across the foot) grips the adapter's two sides with the pad tips
   36 mm above the sole (8 mm above the rear shell's top; 22 mm of the 30 mm adapter held) and lifts the foot level,
   the sole 71 mm up, and holds it there until the left arm has parked.
3. From 20.5 s the right arm carries the foot over the plate, turning it 5 degrees (gripper yaw 0 to -5 degrees),
   lowers it level until the sole is 7 mm above the block's top, with the adapter over (x 27.8 cm, y -0.6 cm): H
   on the centre line, M 30 mm in front of the laser line plus the 3.0 mm trim below. It opens: the heel drops onto
   the block, the foot tips toe-down about the block's front edge until the toe end is on the plate (7.2 degrees;
   H about 27.4 mm above the plate; M moves 0.4 mm back). The gripper goes straight up and parks; both arms are
   then clear of the foot and the block.

With the fingers along y the pivot at release runs along the pad faces, so neither pad pushes the foot back along x;
the centre of mass is 9 mm from the adapter's centre toward the toe, a 0.023 N m pitching moment that the pads'
friction (20 N each) holds easily. The arm's controller does not carry the foot's weight: the static estimate
(J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad) puts the gripper (-2.2, -1.9, -6.6) mm off at the place and the foot
0.9 degrees toe down. The place point is trimmed by (+3.0, +1.0) mm: offline (no sag) M ends 32.6 mm in front of
the line and H 1.0 mm left of the centre line; with the estimated sag about 30.4 mm and -0.9 mm; with anything from
no sag to twice the estimate, M lands 28.2 to 32.6 mm, at least 2.4 mm inside the 25 to 35 mm band. The fingers
open with the sole 7 mm above the block's top, more than the estimated 6.6 mm vertical sag (the toe-down sag lifts
the sole over the block's front edge 0.4 mm), so by the estimate the sole is still 0.4 to 0.8 mm clear of the
block when it is let go, and the arm should not press the foot onto the block first (`block_shift_mm` shows whether
it did). Offline, success has held for its 1 s at
step 608 (30.4 s), with the right gripper rising (G5 is met from step 589, when its pad-tip point is 20 mm above the
adapter). On the GPU the demo scored 1.0 and ended at step 609: M 30.6 mm in front of the line, toe-out 3.7
degrees, H 3.4 mm right of the centre line and 27.0 mm above the plate, T 0.3 mm above it, the foot pitched 7.0 degrees, the
block 1.1 mm from where it was set down. The tool points stay at least 30.8 cm apart and the arms' collision capsules at least 11.2 cm apart. The
run limit is 2,400 steps (2 minutes, 3.6 times the demo), with `--max-steps 2400`.

What to read in `TASK_TRACE` on the first GPU runs: `ap_mm` and `heel_y_mm` after the release (32.6 and 1.0
offline, about 30.4 and -0.9 with the estimated sag; if `ap_mm` ends more than 3 mm from 30, move `PLACE_TRIM[0]` in
`task.py` by the difference); `foot_tilt_deg` while the foot is carried (0 to 2) and after it settles (about 7.2);
`heel_h_mm` (about 27.4) and `toe_t_mm` (about 0); `block_front_mm` (59 offline; (C) needs 90 or less);
`block_shift_mm` and `block_tilt_deg` before and after the foot lands (the block should not move more than a few
mm); `grip_clear_mm` while the right gripper rises; `still` and the three paths once everything rests.

The plate is seven fixed boxes with flush tops (the inlaid lines). PhysX can catch a part that slides across a seam
between two of them. The demo sets both parts straight down, so it is not affected, but a Dream-1 run that pushes
the foot or the block across the plate may see it stop short at a seam (x 26.8 or 27.2 cm, y -0.2 or +0.2 cm): look
for that in the video and in the trace's `heel` and `toe` points before reading a low score as the policy's alone.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the heel block flat on the table centred at
x 25.0 cm, y +40.0 cm, sides parallel to x and y; the foot sole down, toes along +x, its long axis on y -38.0 cm with
the heel end at x 19.0 cm. The plate is fixed. Nothing is consumed.

## Runtime per trial

Measured on an L40S once Isaac Sim is running: about 1.5 minutes for the scripted demo (33 s of simulated time)
and about 5 minutes for a Dream-1 run to the 2,400-step limit (2 minutes of simulated time; both ran at about 8 to 9
control steps a second, slower than real time). The first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 300 x 200 mm plate of 6 mm aluminium or painted MDF clamped to the bench, with a 4 mm red line across it and a
4 mm dark grey line along it (paint or thin tape); a 20 mm polyurethane or wooden block, 60 x 70 mm; and a right
foot stand-in of about 250 g: a 3D-printed shell with a flat 220 x 70 mm sole (rear 120 mm 28 mm tall, front 100
mm 16 mm tall) and a black 3D-printed plastic block, 46 mm square and 30 mm tall, on top 85 mm from the heel end,
or a real prosthetic foot with a pyramid adapter. Everything is unpowered and light; a trial is safe. Mark H and
T on the foot's sides at sole level.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.75 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

The Dream-1 run hit the 2,400-step limit: by the end it had pushed the foot a few millimetres, turning it about 1
degree where it lay, and had not moved the heel block, so the gate failed (no sole corner over the plate): 0.
