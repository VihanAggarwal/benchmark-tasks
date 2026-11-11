# Stand each girder's stop-end in the middle third of its span, clear of the beam

When a concrete floor is poured in more than one go, formwork carpenters (O*NET 47-2031) fix a stop-end (a
bulkhead) across the beam forms where the day's pour will stop, and the concrete finishers (47-2051) pour up to it,
leaving a construction joint. Where that joint may go is set by the building code: ACI 318 (26.5.6.2) places
construction joints in floors within the middle third of the spans of slabs, beams and girders, where shear is
lowest, and offsets a joint in a girder at least two times the width of any intersecting beam, measured from that
beam's face, because beside a beam that frames in, the girder carries that beam's load. Here a 1:15 model of one bay
of beam formwork stands on the table: two girders run along x, each an open trough 50 mm wide between dark plywood
side forms 25 mm tall, with a grey concrete column at each end (column centres 300 mm apart: a 4.5 m span), and a
cross beam's trough, 20 mm wide (0.30 m), runs between them and frames into each girder at x 23.5 to 25.5 cm. A loose
yellow plywood stop-end (40 x 25 x 60 mm) stands on the table beside each girder, turned 30 degrees. Each must be
stood upright across its girder, square to it and on the trough's floor, with all of it inside the middle third of
the span (x 24.0 to 34.0 cm) and at least 40 mm clear of the cross beam's face, then let go. The two rules leave one
band, x 29.5 to 34.0 cm, on the cross beam's far side; the obvious mid-span spot (centred at x 29.0 cm) is in the
middle third but only 22.5 mm from the beam, and scores half. The score is what a site engineer checks with a tape
from the column centres and the cross beam's faces, and a top photo. What is measured is not placing a part at a
mark: nothing marks the band. The policy has to read the span from the columns and the offset from the cross beam's
width, and combine two rules that pull in different directions. Both arms are used: each sets the stop-end in front
of it (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two stop-ends' poses and the grippers' state, with a small tracker of the last half second |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and limit, every zero condition, the exploits we tried, resets, NaN states, the gripper hook, the demo plan and the offline checker on this folder (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

A stop-end counts only once it stands on its own (both grippers clear of it), and `yam_arena` does not pass the
grippers' state to a task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset
once, at import, to take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects'
frame) before each control step and to start a new trial at each reset. It reads only state the simulator already
has and changes nothing the embodiment does. If it cannot read the arms it prints why once; the let-go condition is
then taken as met on those steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Today's pour stops in this bay. Stand one yellow stop-end upright across each of the two girder forms, square to it and on its floor, with all of it inside the middle third of the girder's span (between the column centres) and at least twice the cross beam's width (40 mm) clear of the cross beam's side. Then let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/stop-end-middle-third/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Today's pour stops in this bay. Stand one yellow stop-end upright across each of the two girder forms, square to it and on its floor, with all of it inside the middle third of the girder's span (between the column centres) and at least twice the cross beam's width (40 mm) clear of the cross beam's side. Then let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/stop-end-middle-third/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`girder_left`, `girder_right`, `still`, `moved_mm`,
`first_full_step`, `arms_unavailable`, and for each stop-end `girder`, `score`, `why`, `stands`, `x_mm`, `y_mm`,
`xmin_mm`, `xmax_mm`, `beam_clear_mm`, `middle_third`, `in_trough`, `low_mm`, `tilt_deg`, `square_err_deg`, `let_go`,
`left_zone_mm`, `right_zone_mm`; the stop-ends' poses, both TCPs and openings). Videos: `inspect-robots video
<log>.json`. The scorer keeps one trial's record per process and starts it over at every reset; the run rate is
assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 30 s: one test plays the whole demo
and both one-arm variants through the offline checker).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the two stop-ends' poses and both grippers'
points once per control step (20 Hz) and scores the end state. A stop-end's outline is the shadow of its eight
corners seen from above, turned and tilted as it is. It is over a girder when its centre lies between that girder's
column faces (x 16.0 to 42.0 cm) and side forms' inside faces (y 17.5 to 22.5 cm either side), at any height.

| Each stop-end over a girder, when it stands across it (upright, on the floor, square, inside the trough, let go) | Score |
|---|---|
| Its outline wholly within x 29.5 to 34.0 cm: middle third, and 40 mm or more from the cross beam's face | 0.5 |
| Its outline wholly within the middle third, x 24.0 to 34.0 cm, but less than 40 mm from the cross beam (this includes one standing partly in front of the cross beam's mouth, its outline starting at x 24.0 to 25.5 cm) | 0.25 |
| Partly or wholly outside the middle third (near a column, across the whole of the cross beam's mouth, past the far third point) | 0 |
| Not standing across it: lying, leaning more than 10 degrees, lowest point more than 5 mm up, turned more than 10 degrees from square, astride a form, or a gripper's finger zone within 20 mm | 0 |

Each girder scores the worst of the stop-ends over it, and 0 if none is (a stop-end left on the table, dropped in
the cross beam's trough, on a column). The score is the sum of the two girders: 0, 0.25, 0.5, 0.75 or 1.0, each
reachable: both stop-ends in the band 1.0; one in the band and one at mid-span 0.75; both at mid-span 0.5, or one in
the band and the other left on the table; one at mid-span alone 0.25; nothing standing 0.

A stop-end's place is its whole outline, not its centre: a stop-end is a board, and the joint forms against its
face, so all of it must be in the band. Square, it is 25 mm thick along x in a 45 mm band, so its centre has 20 mm of
play (x 30.75 to 32.75 cm). Every limit is inclusive. A stop-end is the same either end down. The scorer cannot see
which way a gripper is turned, so each gripper's fingers count as a round zone about its axis, out to the fingers'
outer corners (a finger is 10 mm thick and 40 mm wide: 60 mm fully open, 22 mm closed), from the finger tips to 38 mm
back up the gripper. The score is read at the last step; the trial ends early once 1.0 has held for 1 s with neither
end of either stop-end having moved more than 2 mm over the last 0.5 s. A new trial is recognised from rigstate's
reset hook, or, as a fallback, when the simulator puts a stop-end back in a single step and both are then on their
starting marks (within 4.5 cm). A simulator state with a non-finite number, or a missing stop-end, scores 0 for that
step and leaves the record of the trial as it was.

Unintended strategies earn nothing more than they should, as the tests show: both stop-ends put in one girder score
at most that girder's 0.25 (two 25 mm boards do not both fit in the 45 mm band, and the worse one counts); a second
stop-end lying in a girder beside a good one zeroes that girder (it would be cast into the beam); a stop-end
standing along the trough (its 40 mm length along x, which fits the band) is not square and scores 0; one held in
the band, or with an open gripper still round it, scores 0; one lying, lifted, on a form's top, astride the inner
form or in the cross beam's trough scores 0. Only the end state counts: a stop-end set at mid-span and then slid
into the band scores 0.5, as on site.

## One arm alone

With straight-down grips each arm reaches only its own girder and its own stop-end. With its tool point 52 to 80 mm
above the table (the grip and place heights) the left arm misses the right girder's place points (x 30.5 to 33.0 cm
on y -20.0 cm) by 50 mm or more, and the nearest side of the right stop-end where it starts (x 20.0 cm, y -32.0 cm)
by 90 mm or more (inverse kinematics of the arm's URDF with joint limits, grip headings 0, +-30, +-45, +-60, +-90,
+-120 and 180 degrees); the right arm is its mirror image. So with one arm idle the other sets only
its own girder's stop-end: 0.5 either way (on the GPU the scripted demo scored 0.5 with `DEMO_IDLE_ARM=left` and 0.5
with `right`: the working arm's stop-end in the band, the idle arm's still on the table; see the results below). That is a limit of where the girders are, not of the act: standing a stop-end in a
form does not in itself need two hands, and we do not claim it does. A tilted wrist reaching across has not been
tried.

## Demo and run limit

The scripted demo takes 18.7 s (374 control steps); both arms work at once, each on its own girder, mirror images
of each other, their tool points never less than 40 cm apart. Every grip closes over 1.5 s with the fingers
commanded 8 mm narrower than the 25 mm stop-end, every release opens over 1.5 s, and the set-down pauses with the
stop-end's bottom 10 mm above the forms' tops, then enters the trough slowly.

- Each arm opens to 45 mm, comes down over its stop-end turned 30 degrees to match it (left -60, right +60 degrees of
  grip heading), the tool point 52 mm above the table (pad tips 39 mm up, on the stop-end's top 21 mm), and closes
  across its 25 mm thickness.
- It lifts until the stop-end's bottom is 78 mm up (53 mm over the forms, 38 mm over the columns), carries it over
  the girder at (31.75, +-20.0) cm turning it square (grip heading -90 / +90 degrees: the fingers close along x, the
  40 mm length lies across the trough), comes down until its bottom is 10 mm above the forms' tops (35 mm up), pauses
  0.8 s, and lowers it over 3 s into the trough to 0.5 mm above its floor.
  Its outline then spans x 30.5 to 33.0 cm, 10 mm inside either end of the band, and y 18.0 to 22.0 cm, 5 mm from
  each side form. The pad tips stay 14 mm above the forms' tops.
- It opens to 45 mm (10 mm clear of each face), rises straight up to 15 cm (pad tips 77 mm over the stop-end's top,
  the finger zone 38 mm clear), and parks over its own side.

The load is 0.03 kg, so the arms' sag under it is about a millimetre (the program's GPU runs show 3 to 7 mm only
under 1 to 3 N); the band leaves 10 mm either way and the trough 5 mm either side. Offline, success has held for its
1 s at step 355; on the GPU 1.0 was first reached at step 337 and the run ended at step 356. The scripted policy
would stop at step 435. The run limit is 1,200 steps (1 minute).

Our offline kinematic check (NumPy, no physics; not part of this folder): worst IK error 0.09 mm, smallest joint
margin 15.9 degrees (joint 4), tool points at least 40 cm apart, the arms' collision capsules at least 25 cm apart,
pad tips never lower than 39 mm above the table.

Things to look at in the GPU runs: in the settle run nothing moves (both stop-ends stand on a 40 x 25 mm end). After
each release, the stop-end's `xmin_mm` near 305 and `xmax_mm` near 330 (if either is more than 5 mm off, move
`PLACE_X` in `task.py` by the difference), `y_mm` within 3 mm of +-200 (5 mm to each side form; a stop-end that
catches a form's top on the way in shows as `low_mm` or `tilt_deg` off), `tilt_deg` and `square_err_deg` under 2,
`low_mm` near 0, `in_trough` true; `let_go` true once the gripper has risen; `arms_unavailable` 0 (otherwise let go
silently passes: do not submit).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two stop-ends back upright on the table,
the left one centred at x 20.0 cm, y +32.0 cm with its 40 mm length turned 30 degrees anticlockwise from y (seen
from above), the right one at x 20.0 cm, y -32.0 cm turned 30 degrees clockwise. The columns and forms are fixed.
Nothing in the start layout lies where the arms' start pose hides the table from the top camera.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo is about 17 s of simulated time until success
ends it; a full Dream-1 run is 1 minute); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Four 40 mm concrete-grey blocks (or painted wood) for the columns. Strips of 10 mm film-faced plywood ripped 25 mm
tall, screwed or clamped to the table: for each girder one 260 mm strip on the outside and two on the inside (75 and
165 mm, 20 mm apart at the cross beam's mouth), 50 mm apart inside; two 330 mm strips across between the girders,
20 mm apart inside, for the cross beam. Two plywood blocks 40 x 25 x 60 mm (about 30 g), faces painted yellow, for
the stop-ends. To score: a tape from the column centres (the middle third is the 100 mm in the middle of the 300 mm
span) and from the cross beam's far face (40 mm), a square against a side form, and a top photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
