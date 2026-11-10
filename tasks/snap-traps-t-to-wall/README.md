# Set mouse snap traps at the droppings: square to the wall, trigger end at the wall

Pest control workers and building maintenance workers trap mice where droppings show they run (O*NET 37-2021.00 Pest
Control Workers, 49-9071.00 Maintenance and Repair Workers, General). Mice keep to walls as they travel, so the CDC's
rodent-control guidance says to set a snap trap with its bait (trigger) end next to the wall, the trap making a "T"
with it, and the US National Park Service's guide to snap traps shows a single trap set with its trigger against the
wall: a mouse running along the wall then meets the trigger from either direction. A single trap laid parallel to the
wall meets mice from one side only (a parallel set uses two traps side by side, triggers outward), and one with its
trigger end out in the room is run past. The traps here lie in mid-floor facing either way, so each trap's trigger
end has to be found and turned to the wall.

Here everything is at real size. A light grey vinyl floor (340 x 500 mm) runs between two white skirting boards along
x, the wall above them cut away; along each skirting are two dark brown droppings patches, 5 to 15 mm from its face.
Four standard wooden mouse snap traps (98 x 45 mm, unarmed: the bail has sprung and lies over the yellow trigger
pedal) lie in the middle of the floor, parallel to the walls, the nearer two with their pedal ends toward the robot
and the farther two away from it. The robot has to set one trap at each patch: square to the wall, its pedal end
toward the wall and no more than 8 mm from the skirting, at the patch. So each trap needs a quarter turn, and the two
traps on a side need opposite turns. The score is 0.25 for each patch with a trap set at it, read from how the traps
lie at the end, as a ruler and a set square would read them. The natural split is one wall each: each arm sets the
two traps on its side.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four traps' states, with a small tracker for stillness and the trace |
| `mechkit.py` | Compound bodies for `yam_arena` task files (each trap is one rigid body of six boxes), hooked from the task file's import; unchanged from our kit |
| `tests/test_scoring.py` | Offline tests of every scoring tier and threshold, every zero condition, the exploits we tried, resets, broken states, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the fixed parts are
`yam_arena`'s own task-file boxes, the traps are built by `mechkit.py` (next to `task.py`, imported by it) and the
score comes from the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mice are running along both walls of this passage; the dark brown droppings mark their runs. Set the four snap traps (they are not armed) the way pest controllers do: one trap at each droppings patch, square to the wall, with its trigger end (the yellow pedal) toward the wall and no more than 8 mm from the skirting board." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/snap-traps-t-to-wall/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mice are running along both walls of this passage; the dark brown droppings mark their runs. Set the four snap traps (they are not armed) the way pest controllers do: one trap at each droppings patch, square to the wall, with its trigger end (the yellow pedal) toward the wall and no more than 8 mm from the skirting board." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/snap-traps-t-to-wall/dream
```

Run one trial per process (the default). The score depends only on where the traps lie; the small tracker in the
task module keeps each trap's path over the last 0.5 s (for stillness), and starts afresh when mechkit counts a reset
or when the simulator puts every trap back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided: `patches_set`, `zero` (why the score is 0, if it is), for each patch
whether it is `set` and by which `traps`, and for each trap `set_at`, `placed`, `at_rest`, `base_down`, `tilt_deg`,
`on_floor`, `low_mm`, `pedal_heading_deg`, `faces` (the wall its pedal end points nearer to), `square_err_deg`,
`end_gap_mm`, `nearest_patch`, `along_mm` and `origin_mm`, and the raw poses. mechkit prints, after the first reset,
how Isaac Lab built each trap (one loose rigid body of 25 g). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the four traps' poses, once per control
step (20 Hz), and scores the state at hand: the score is read at the last step. Each trap's frame (the centre of its
base's underside and its orientation) gives every point of it, from its known shape; the floor, skirting boards and
patches are fixed, so their places come from `layout.py`. The traps are alike: any trap may be set at any patch.

A trap is still when neither its centre nor the midpoint of its pedal end moved more than 2 mm over the last 0.5 s
(positions, not the simulator's reported velocity, which chatters while a part is gripped). It is placed when it is
still and lies base down on the floor: its base tilted at most 10 degrees from level, pedal side up, and its lowest
point within 3 mm of the floor's top. A placed trap is set at a patch when (a) seen from above, the direction from its
plain end to its pedal end is within 15 degrees of straight at the patch's wall; (b) its point nearest that wall (a
corner of the pedal end when it is turned) is no more than 8 mm from the skirting's face (touching counts, and so does
up to 3 mm of contact overlap reported by the simulator); and (c) the midpoint of its pedal end is within 25 mm of the
patch's centre, measured along the wall.

| Outcome | Score |
|---|---|
| No patch with a trap set at it | 0 |
| For each droppings patch with at least one trap set at it | 0.25 |

So the score is 0.25 x the number of patches with a trap set at them: 0, 0.25, 0.5, 0.75 or 1.0, and every tier is
reached by an ordinary partial job. A patch counts once however many traps are set at it, and no trap can be within
25 mm of two patches (a wall's two patches are 90 and 110 mm apart). Every outcome, with numbers: all four set 1.0;
three set and the fourth 12 mm off the wall, turned 20 degrees, 30 mm along the wall from its patch or left in
mid-floor 0.75; one wall's two set and the other wall's two laid parallel to the skirting, or set with their pedal
ends out in the room (plain end at the wall), 0.5; two traps set side by side at one patch and the other two left
where they started 0.25; the start (all four in mid-floor, parallel to the walls) 0. A trap upside down, on its side,
standing on an end, leaning on the skirting, lying on another trap, held more than 3 mm up or moving earns its patch
nothing. The trial ends early (success) once 1.0 has held for 2.5 s; otherwise it is scored as it stands at the run
limit. A state with a non-finite number, a zero quaternion, no frame origin or a missing trap (a simulator blow-up)
scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: pushing the traps flat against the skirting lengthwise
(the common mistake) 0; a trap square to the wall with its plain end touching it 0; standing a trap on its pedal end
against the skirting, or leaning it on the skirting, 0; two traps stacked at one patch count once; a trap thrown off
the table changes nothing. A turned trap is judged by its nearest corner, so turning it cannot hide a gap. The scorer
cannot see the grippers, so a trap held still within 3 mm of the floor counts as lying there (see `fidelity`); 3.5 mm
up it does not.

## One arm alone

The demo splits the work by wall: each arm sets the two traps on its own side at the two patches on its own wall.
With one arm idle the other still sets its own two: on the GPU the scripted demo scored 0.5 with the left arm idle
and 0.5 with the right arm idle (the working arm's two traps set, the idle arm's two left where they started), as
the offline kinematic check predicted. The task is not scored on using both arms: one arm working only its own wall
earns 0.5.

The other wall is out of a straight-down grip's reach. At the height that grips a trap, our IK model puts the left
arm's tool point at most at y -14.0 cm at x 26 cm, -12.8 cm at x 28 cm, -11.5 cm at x 30 cm and -9.3 cm at x 33 cm,
and short of the midline from x 42 cm on, while a trap set at either right-wall patch lies wholly beyond y -13.5 cm:
in every pose the rule credits (turned up to 15 degrees, up to 8 mm off the wall, up to 25 mm along it) no corner of
it is within that reach (the closest misses by about 6 mm, and a corner is no grip). The right arm is the mirror
image, and the left wall's nearer patch is 1 cm farther forward than the right wall's, harder still. We do not claim
one arm cannot do more: a leaned grip, or shoving traps across the floor with another trap held as a pusher, might
set one at the far wall's nearer patch, a move the scripted demo does not try. The ordinary one-arm result is 0.5.

## Demo and run limit

The scripted demo takes 32.6 s (652 control steps), both arms on one clock, each working only on its own side of the
midline, so the two tool points never come closer than 26.0 cm along y (offline: the arms' collision capsules at
least 8.9 cm apart). Each arm, at the same time as the other, for each of its two traps:

1. comes down round the trap at its middle, fingers open to 70 mm across the 45 mm base (12.5 mm to spare each side),
   the pad tips 3 mm above the floor, and grips it (commanded to 37 mm) on 7 mm of the base's sides, clear of the
   steel spring block and bail on top;
2. lifts it 67 mm (its underside 7 mm over the skirting's top) and carries it to its patch, turning it a quarter turn
   on the way, whichever way brings the pedal end round to the wall: the left arm turns its nearer trap -90 degrees
   and its farther one +90, the right arm +90 and -90 (the wrist's joint 6 keeps at least 15.9 degrees from its
   limit; the turn sweeps the trap round the tool point with its corners staying at least 4 mm inside the skirting's
   face);
3. sets it down square to the wall, its pedal end aimed 4 mm from the skirting's face (the middle of the 0-8 mm band)
   and its pedal end's midpoint level with the patch's centre (25 mm to spare either way along the wall); the trap's
   pedal end rests on the 0.5 mm patch;
4. lets go and goes straight up; after the second trap it moves up and out to the side.

Every gripper open or close takes 1.5 s, every final descent is slow (the last 30 mm in 1.5 s) and every trap is let
go 2 mm above the floor after a 0.3 s settle. The traps (25 g) barely load the arms. `PLACE_TRIM` in `task.py` moves
a trap's place point if the GPU run shows it landing off by a steady amount (compare `end_gap_mm` with 4 and
`along_mm` with 0 in the trace); all trims are zero, and on the GPU the four traps landed 2.8 to 4.5 mm from the
skirting's face, 0.5 to 1.7 mm from their patches' centres along the wall and square to within 1.3 degrees. Offline
the smallest joint margins are 15.5 degrees (joint 4, lifting the farther trap) and 15.9 degrees (joint 6, at the
far patches' +90 and -90 degree turns), the worst IK error 0.08 mm and the pad tips stay at least 3 mm above the
floor. A trap counts as set while still gripped 2 mm above the floor, so success must hold for 2.5 s, longer than
the settle and release after the last set-down (0.3 + 1.5 s) plus the 0.5 s stillness window: the trial cannot end
with a trap still in a closed gripper. Offline, success holds at step 621; the plan runs to step 652 and the
scripted policy stops at step 713. On the GPU success held at step 621 too, and the trial ended there. The run limit
is 1,200 steps (60 s), which leaves Dream-1 nearly twice the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lay the four traps back base down in the middle
of the floor, long sides along x, centred at (x 30.0, y +13.0), (42.0, +13.0), (30.0, -13.0) and (42.0, -13.0) cm, the
two at x 30.0 cm with their pedal ends toward the robot and the two at x 42.0 cm with their pedal ends away from it.
The traps are interchangeable and stay unarmed; the floor, skirting boards and droppings patches are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 31 s of simulated time, when
success has held for 2.5 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 340 x 500 mm sheet of light grey vinyl flooring, about 2 mm thick, fixed to the table 14 cm in front of the arm
bases (x 14.0 to 48.0 cm, y -25.0 to +25.0 cm). Two skirting boards: 340 mm lengths of white-painted MDF, 62 mm tall
and 15 mm thick, fixed to the table against the vinyl's long edges, faces at y +25.0 and -25.0 cm. Four droppings
patches: 20 x 10 mm dark brown stickers (or paint) on the vinyl, centred 10 mm from a skirting's face, at x 34.0 and
43.0 cm on the left and x 33.0 and 44.0 cm on the right. Four standard wooden mouse snap traps (98 x 45 mm pine base,
yellow plastic trigger pedal), about 25 g each, unarmed: the bail sprung over the pedal end. Positions in
`task.json`'s `setup`.

Trials are safe: the traps stay unarmed throughout (nothing is set to snap, and the task never asks for it), the
heaviest part is a 25 g trap, nothing is hot, powered or under load, and nobody needs to reach into the robot's space
until the trial has ended. For extra margin the bail's spring can be removed or the bail tied down.

Scoring a real trial, from the floor as it stands: for each patch, check that a trap lies base down at it, then
measure with a ruler from the skirting's face to the trap's nearest point (8 mm or less), hold a set square against
the skirting to check the trap is square to it (within 15 degrees: 26 mm of offset over its 98 mm length) with its
yellow pedal at the wall end, and measure along the skirting from the patch's centre to the middle of the trap's
pedal end (25 mm or less).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
