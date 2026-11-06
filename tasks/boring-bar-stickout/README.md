# Set a boring bar in the lathe tool block: tip to the chuck, 45-64 mm out

Before boring a hole on a lathe, the machinist lays the boring bar in the tool block and sets its overhang: the
shortest stick-out that still reaches the bottom of the bore with a few millimetres to spare, and never more than
about four times the shank size for a steel bar, because chatter and deflection grow steeply with overhang (a long
bar leaves a rough, oversize bore; a short one lets the block hit the part before the bore is finished). The bar
must also face the chuck with its cutting insert up. Here a bench-scale lathe mock stands on the table: a chuck
holding an aluminium workpiece with the bore's mouth marked on its face, and a tool block with a 26 mm channel
facing it. A 16 mm square boring bar lies on a mat beside it, pointing the wrong way (forward). The robot has to
turn it a quarter turn the right way, lay it in the channel and set the tip 45 to 64 mm past the block's face
(the 40 mm bore depth plus 5 mm, and four times the 16 mm shank), clear of the part. The score is what a setter
reads with a steel rule: which way the bar points, whether it lies in the channel, and its stick-out. This is a
single-arm task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the bar's pose, with a small tracker of its last half second |
| `mechkit.py` | Helper that builds the bar as one rigid body of three boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, the exploits we tried, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and one helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain box has no visible tip end.
`mechkit.py` (imported by `task.py`) builds the boring bar from three boxes (shank, head, insert) as one loose rigid
body through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the scorer. It
changes nothing else the embodiment does, and it stops the run with an error if the bar was not built as a compound
body. The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching
`yam_arena.scene` from the task file's import, pxr authoring in a custom spawner) follows the one first shown in the
program repository's `tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (right arm; the left arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the boring bar in the tool block to bore the 40 mm deep hole: lay it in the block's channel, gold insert up and its tip toward the chuck, sticking out past the block's face far enough to reach the full depth with 5 mm to spare (at least 45 mm) but no more than four times its 16 mm shank (64 mm). Keep the tip off the part." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/boring-bar-stickout/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the boring bar in the tool block to bore the 40 mm deep hole: lay it in the block's channel, gold insert up and its tip toward the chuck, sticking out past the block's face far enough to reach the full depth with 5 mm to spare (at least 45 mm) but no more than four times its 16 mm shank (64 mm). Keep the tip off the part." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/boring-bar-stickout/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`s_mm`, `tip_toward_chuck`, `tilt_deg`, `heading_err_deg`,
`over_block_mm`, `centre_off_mm` and `underside_mm` at both ends of the block, `at_rest`, `moved_mm`, `tier`,
`zero_reason`, the bar's origin, tip and back end). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the bar's pose, once per control
step (20 Hz). From the pose of the bar's base frame (the middle of its bottom face, x toward the tip, z up through
the insert) it computes the centres of the tip face and the back face. The stick-out is the tip face's distance past
the block's chuck-side face along the spindle axis (y), as a steel rule laid along the channel reads it. "In the
channel" is checked where the bar's centre line crosses the two ends of the part of it that lies over the block
(the block's back face or the bar's back end, and the block's face or the tip): there it must be within 6 mm of the
channel's centre sideways (a 16 mm bar in the 26 mm channel has 5 mm of play either way; 1 mm is allowed for the
solver) and its underside within 3 mm of the floor. Tilt is the angle of the insert's axis from the vertical, so a
bar pitched, rolled, on its side or upside down counts as tilted; the square bar fits the channel on any face, but
only insert up can cut. Every score needs the bar at rest: over the last 0.5 s neither end moved more than 2 mm,
measured on positions (the simulator's reported velocities chatter while a part is gripped). A new episode is
recognised when mechkit counts a reset, or when the simulator puts the bar back on its starting mark (within
4.5 cm of it, which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number,
a zero quaternion or no base-frame origin scores 0 for that step and leaves the scorer's record as it was.

| Outcome (bar at rest at the end) | Score |
|---|---|
| In the channel, insert up, tip toward the chuck, 45 <= s <= 64 mm (success once held 2.5 s) | 1.0 |
| The same, 38 <= s < 45 mm (less than the 5 mm to spare, at most 2 mm short of the 40 mm depth) or 64 < s <= 72 mm (past the four-times rule) | 0.5 |
| The same, s < 38 mm (more than 2 mm short of the 40 mm depth; includes a tip inside the block) | 0.25 |
| Tip pointing away from the chuck; not in the channel (on a wall, held up, tilted, on its side, skewed); s > 72 mm (tip within 3 mm of the part, or touching it); still moving at the end | 0 |

Every tier is a position the bar rests in by itself: lying in the channel it balances on the block face's edge at
a stick-out of 74.8 mm and tips backward off the block's back edge below -5.2 mm, so every stick-out from -5 to
72 mm is a stable rest, and each maps to exactly one tier. Unintended strategies earn nothing, as the tests show:
the bar laid backwards with its back end 45-64 mm out (0); laid on its side or upside down in the channel at a
band stick-out (0); propped on a wall's top and the floor (0); laid across the block on the walls (0); held in the
air over the channel at a band stick-out (0); the tip pushed against the workpiece (0); resting across the block's
front edge, tip dipped toward the table (0). A bar slid along the channel into the band or dropped in from just
above it is scored by where it comes to rest, which is the real setter's criterion. The scorer cannot see the
gripper, so a bar held still in place is scored where it lies (see `fidelity`).

## One arm alone

This task needs one arm, and the demo uses one: the right arm grips the bar 35 mm back from its tip, lifts it,
turns it a quarter turn to the left while carrying it over the channel, lays it in and lets go, while the left arm
holds its ready pose. `DEMO_IDLE_ARM=left` is therefore the same run as the demo (1.0 on the GPU);
`DEMO_IDLE_ARM=right` leaves nothing to move (0 on the GPU: the bar stays on the mat). That 0 says nothing about the
left arm's ability: gripping straight down, the left arm reaches the back of the bar where it starts (x 12-20 cm
at y -17 cm, though not the demo's grip point 23.5 cm out) and the place pose over the channel (kinematics only, not
run), so the left arm alone should also be able to do the task (it would have to grip nearer the tip, after setting
the bar down once, for its fingers to come down clear of the block). No claim is made that the task needs both arms: its difficulty is reading which end
is the tip, turning it the right way and setting the overhang to the band. (The GPU results are in 'Results in
simulation' below.)

## Demo and run limit

The scripted demo takes 18.8 s (376 control steps): the right arm comes down beside the bar on the mat (pad tips
4 mm above it), grips it across its 16 mm (fingers commanded 8 mm narrower, over 1.5 s) 35 mm back from the tip,
lifts it to 10 cm (gripper height), carries it over the channel in 2.5 s while turning the grip from yaw 0 to
+90 degrees (the tip swings from forward to left, toward the chuck), comes down to 3 cm above the channel, lowers the
last 3 cm over 2 s, waits 0.5 s, opens over 1.5 s with the bar's underside aimed 2 mm above the floor, goes straight
up and moves aside. It aims at a stick-out of 55 mm (10 mm inside the band's lower end and 9 mm inside its upper
end), with the grip 20 mm past the block's face, so the fingers come down clear of the block. The arm's controller
does not carry the bar's weight: the compliance model (J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad, yamkin) puts
the gripper about 7 mm low and (-1.3, +0.5) mm off at the place pose, well inside the channel's 5 mm of play and the
band's margins, so no trim is applied (`PLACE_TRIM` in `task.py` stays at zero). With the sag the bar reaches the
floor before the slow descent ends; success then needs 0.5 s at rest and 2.5 s held. On the GPU the bar was on the
floor at about 14.0 s (step 280, 0.3 s before the slow descent ended), at rest from step 285, and the trial ended
with success at step 334 (16.7 s), with the fingers fully open and the hand going up: stick-out 54.4 mm, centre line
0.9-1.1 mm off the channel's centre, tilt 0.0 degrees, at most 1.8 degrees of tilt in the hand during the carry.
Offline (no sag) the bar drops 2 mm at the release and success holds for its 2.5 s at step 364. The bar's centre of
mass is 40 mm behind the grip, about 0.12 N m on the pads. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the bar back on the black mat centred at
x 19.5 cm, y -17.0 cm, lying along x with its tip forward (+x) and the insert up; everything else is fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 17 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 50 x 80 x 40 mm steel (or aluminium) block with two 12 x 12 mm strips fixed along its top edges for the tool
block, leaving a 26 mm channel; a 100 x 30 x 96 mm block as the chuck with a 50 x 40 x 50 mm aluminium block fixed
to its face, 75 mm from the tool block's face along the channel's line, with a 22 mm square of black tape on it
centred on the channel's centre line, 48 mm above the bench (the height of a seated bar's centre line); a black
rubber mat; and a 16 x 16 x 150 mm steel bar (0.30 kg) with its last 15 mm painted black and a 6 x 6 x 3 mm
gold-coloured block glued on top at that end, flush with the end, as the insert (a dummy with no cutting edge).
Clamp everything but the bar to the bench; nothing is powered. Score with a steel rule from the tool block's face
to the bar's tip, a photo from above and a look from the side (the bar flat on the channel floor).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
