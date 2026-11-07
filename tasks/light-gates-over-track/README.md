# Set two light gates over an air track, straddling it, 200 mm apart

School and college science technicians set up the apparatus before each practical. For the speed and acceleration
practicals, two light gates (photogates) go over an air track or runway: each straddles the track so that the card
on the glider cuts its beam, square to the track, at the spacing the worksheet gives, with the glider's run left
clear. A gate left resting on the track, or standing too close to it, stops the glider; gates at the wrong spacing
give every group the wrong speed. Here a 440 mm air track is fixed along the table 28 cm in front of the arm bases,
with a buffer at each end and a glider parked (air off) at the left end, a black card standing on it. Two loose
light gates, each a free-standing inverted U on two feet, stand on the table, one beside each arm, turned along the
track's direction. Each must be lifted, turned square in the air and lowered so that its legs straddle the 30 mm
track (19 mm to spare on each side), with both feet on the table, in front of the glider; the two gates are scored
on their spacing, 200 mm centre to centre. Anything of a gate in the glider's path scores 0.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two gates' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each light gate as one rigid body of seven boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (track, buffers, glider), the standard `task_score` scorer, and one helper module next
to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a light gate is an inverted U: its two legs
must stand on the table either side of the track with the crossbar passing over it. `mechkit.py` (imported by
`task.py`) builds each gate from seven boxes (two feet, two legs, the crossbar and a red sensor window on each leg)
as one loose rigid body through a hook on `yam_arena.scene`, and reports its pose (base-frame origin, midway between
the feet on their undersides, and orientation) to the scorer. It changes nothing else the embodiment does, and it
stops the run with an error if a gate was not built as a compound body. The technique (a `task_file.Box` subclass
the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's import, pxr authoring in a
custom spawner) follows the one first shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an
independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the two light gates over the air track for the speed run: each one straddling the track with both feet on the bench and nothing of it in the glider's path, square to the track, in front of the parked glider, the two gates 200 mm apart, centre to centre." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/light-gates-over-track/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the two light gates over the air track for the speed run: each one straddling the track with both feet on the bench and nothing of it in the glider's path, square to the track, in front of the parked glider, the two gates 200 mm apart, centre to centre." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/light-gates-over-track/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `blocked`, `n_set`, `spacing_mm` and, per gate,
`set`, `at_rest`, `standing`, `straddling`, `square_err_deg`, `in_section`, `in_path`, `centre_mm`,
`off_centre_mm`, `path_gap_mm`, `foot_low_mm`, `foot_high_mm`, plus each gate's raw origin, quaternion and velocity).
Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at
every reset (mechkit counts resets); the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the two gates' poses, once per
control step (20 Hz), and keeps the last half second of where each gate's feet were, to tell whether it is at rest
(from positions, not the reported velocity, which chatters while a part is gripped). A gate's centre is the point
midway between its feet. The glider's path is the space it sweeps on its run: within 22 mm of the track's centre
line across the track (the glider is 41 mm wide), from the table top up to 57 mm (its card's top is 55 mm), between
the buffers. Whether a gate blocks the run is tested shape by shape: each of its seven boxes, as the gate actually
stands (turned, tilted, lying on its side), against that space, touching counting as inside; the crossbar, 74 mm up
on a standing gate, passes over it. A gate is set when it is at rest with every corner of both feet within 3 mm of
the table top, one foot on each side of the track, the crossbar within 10 degrees of square to the track, its centre
in the timing section (y -15.0 to +13.0 cm: in front of the parked glider, the crossbar at least 9 mm clear of the
glider's front; and far enough from the far buffer that the glider, stopped against it, has carried its card right
through the beam, the card's back edge 10 mm past the windows) and nothing of it in the path. Standing square, a gate's feet clear the
path with its centre up to 12 mm either side of the track's centre line; turned 10 degrees, up to 8 mm.

| Outcome | Score |
|---|---|
| Any part of either gate in the glider's path (resting on the track, lying across it, standing too close to it) | 0 |
| Both gates set, centres 190 to 210 mm apart along the track | 1.0 |
| Both set, centres 180 to under 190 or over 210 to 220 mm apart | 0.75 |
| Both set, centres under 180 or over 220 mm apart | 0.5 |
| One gate set | 0.25 |
| Anything else (no gate set: still at its start, beside the track, held up, turned, straddling the glider, a buffer or the last 7 cm before the far buffer) | 0 |

The trial ends early once 1.0 has held for 2.5 s (`SUCCESS_HOLD_S`), otherwise it is scored as it stands at the
run limit. A new trial is recognised from mechkit's reset counter or, as a fallback, when both gates are put back on
their starting marks (within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a
non-finite number or a zero quaternion scores 0 for that step and leaves the record of the trial as it was. The two
gates are alike: either may end nearer the glider, and a gate turned end for end is the same gate.

Unintended strategies earn nothing they should not, as the tests show: lowering a gate without turning it square lands
its feet on the track's top (0); two gates square and 200 mm apart beside the track instead of over it score 0; a gate
leaning with one foot on the track is in the path (0); a gate set astride the parked glider, astride the far buffer or
within the last 7 cm before it (where the stopped glider's card would sit in its beam) is not set; gates held 5 mm
above the table are not set; an upside-down gate is not set; a gate creeping or turning on the spot is not at rest.
Sliding a gate onto the track from its end (the 68 mm between its feet passes over the 40 mm buffer) and then along it
is a real way of setting it and is scored like any other. The scorer cannot see the grippers, so a gate held still
standing on the table counts as set (see `fidelity`): in the simulated demo the second gate counts as set while
still gripped, and the trial ends 0.75 s after that gripper starts to open.

## One arm alone

In simulation (L40S, one trial each, see 'Results in simulation'), the scripted demo scores 0.25 with the left arm
idle and 0.25 with the right arm idle, as the offline kinematic check predicted: the working arm sets the gate on its
own side (centre at x 277.9 mm, y +101.6 or -101.4 mm, within 1.0 degree of square) and the other gate stays at its start,
unturned, so only one gate is set.

Reach (inverse kinematics of the arm's URDF with its link collision proxy, at the gate's grip height): gripping
straight down, the left arm's tool point reaches at most y -14.3 cm at x 20 cm, 7 cm short of the right gate's
near end (y -21.4 cm), and the right arm is its mirror image. Leaning its gripper toward the far side, though, an arm
reaches y -21.3 cm at 15 degrees and y -23.5 cm at 20 degrees: the end of the other gate's crossbar. So one arm
alone could in principle pick the far gate by the end of its crossbar with a leaned grip, carry it in and set both
gates in turn (straight down at x 28 cm it reaches 9.75 cm past the midline, enough for a 190 to 210 mm spacing with
the gates shifted toward it); the scripted one-arm runs do not try that, and we do not claim that one arm cannot
score 1.0. Both arms are used naturally: each sets the gate on its own side, and they take turns over the track.

## Demo and run limit

The scripted demo takes 31.6 s (632 control steps). Each arm grips its gate by the middle of its crossbar with the
fingers closing across the crossbar's 20 mm depth (along x, the left arm at yaw -90 degrees and the right at +90;
commanded 12 mm, closing over 1.5 s), the tool point 82 mm above the gate's feet (pad tips 69 mm up, 5 mm below the
crossbar's underside). It lifts the gate until its feet are 58 mm up, carries it straight along y to x 20.0 cm,
100 mm from the midline, turning it a quarter turn on the way (yaw to 0, the crossbar now square to the track), so
it never passes over the glider or a buffer, then 80 mm across over the track (feet 33 mm above the track's top). It
lowers the last 30 mm over 2 s to 2 mm above the table, centred on the track at y +10.0 cm (left) or -10.0 cm
(right), opens over 1.5 s with the fingers along y (the pads 13.5 mm either side of the gate's centre, well inside
its legs), rises and goes back over its starting place. The clock: left 0 to 21.1 s (gate let go at 18.1 s); the
right arm waits over its gate for 10.5 s, grips it at 15.5 to 17.0 s and carries it in front of the track from
18.8 s, once the left hand has let go, and across only from 21.3 s, once the left hand is back: both place points
are on the track's line 200 mm apart, the least spacing for two grippers at one x. The right gate is let go at 28.6 s.

Offline (kinematic check): worst inverse-kinematics error 0.086 mm; smallest joint margin 15.7 degrees (joint 4 of
each arm, at its final pose back over the starting place); arm links at least 14.8 cm apart (capsule proxy), tool points at least 31.9 cm
apart; pad tips never lower than 69 mm above the table. Replaying the carry with the gate's seven boxes, every box
stays at least 28 mm from the glider and the buffers; set down, the feet are 19 mm from the track's sides. Success
has held for its 2.5 s at step 619; the scripted policy stops at step 693. The run limit is 1,200 steps (60 s).

In simulation (L40S): the arm sags a few millimetres under the 80 g gate, so each gate's feet reach the table, and the
gate counts as set, before the slow descent ends and before its gripper opens. The left gate was set from about step 330,
the right from about step 537, when the score became 1.0; 1.0 held for 2.5 s and the trial ended at step 587 (29.35 s), the
right gripper 0.75 s into its 1.5 s opening. Final centres (x, y): (277.9, +101.6) and (277.9, -101.5) mm, 203.1 mm
apart; both within 1.0 degree of square; the nearest part of each gate 9.6 to 9.8 mm clear of the glider's path; every
foot corner on the table (within 0.1 mm). `TASK_TRACE` shows these as `centre_mm`, `square_err_deg`, `path_gap_mm` and
`foot_high_mm`; `PLACE_TRIM` in `task.py` moves the place points if a different simulator lands a gate off its aim.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the left gate back on its feet centred at
x 20.0 cm, y +26.0 cm and the right gate at x 20.0 cm, y -26.0 cm, both with the crossbar along y. The track,
buffers and glider are fixed. Nothing is consumed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 32 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 440 mm length of 30 x 25 mm aluminium box section (or a real air track, with the scoring distances scaled to it)
screwed or clamped to the table along y, centred 28 cm in front of the arm bases, a rubber block at each end and a
parked glider (a 41 mm wide plastic channel over the track with a 25 mm card on top) taped in place at the left end.
For the gates, either two school light gates on feet of the stated size, or two inverted U frames of 3 mm plywood or
printed plastic (92 mm across, 84 mm tall, 68 mm between the feet, about 80 g) with red tape for the sensor windows.
Score from a top photo over a rule (centres, angles, spacing), a 3 mm feeler gauge under each foot, and a card
template cut to the glider's path (44 mm wide, 57 mm tall, notched to sit over the track) slid from buffer to
buffer. The gates weigh 80 g, so trials are safe.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
