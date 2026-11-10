# Set a three-point bend fixture to a 16:1 span and centre the specimen on it

Three-point bend (flexural) tests of plastics and composites (ASTM D790, ISO 178) are run every day by
materials-testing technicians in moulding, compounding and composites quality labs. Before each new specimen
thickness the technician resets the fixture: the two supports go 16 times the specimen's depth apart (16 +- 1 times is
allowed), symmetric about the loading nose, and the bar is laid flat across both rollers, square to them and centred
so that both ends overhang equally. A wrong or off-centre span, or an off-centre or skewed bar, invalidates the
flexural strength and modulus. Here a bend fixture stands on the table, still set for the last specimen (a 4 mm bar,
rollers 64 mm apart): a steel base with a red centre line where the loading nose comes down and a scale of white ticks
every 10 mm, and two loose blue support blocks, each with a steel roller on top. The next specimen, a 10 mm thick
glass-filled nylon bar, lies on a mat. The robot has to work out the span (16 x 10 = 160 mm), move both supports out
to 80 mm either side of the red line, lay the bar across the rollers centred and square, and let go of everything.
The score is what a technician reads with a steel rule, a protractor and a look from above. One arm can do all of it
(see 'One arm alone'); the demo uses both, each moving the support on its own side.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the specimen's and supports' poses and the grippers' state, with a small tracker of the last half second |
| `mechkit.py` | Helper that builds each support block as one rigid body of a block and a lying roller, and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, the exploits we tried, unreadable and non-finite states, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and a loose box (the specimen), the standard `task_score` scorer, and two helper
modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, upright cylinders or spheres, and a support needs a roller lying along
the top of its block. `mechkit.py` (imported by `task.py`) builds each support from a box and a lying cylinder, as one
loose rigid body, through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to
the scorer. It stops the run with an error if a support was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read the arms it prints why once; the let-go condition is
then taken as met on those steps, and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the bend fixture for this 10 mm thick specimen: move the two support blocks so their rollers are 16 times its thickness apart, centred on the red centre line, then lay the specimen flat across the rollers, centred and square to them. Let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/flexure-span-sixteen-to-one/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the bend fixture for this 10 mm thick specimen: move the two support blocks so their rollers are 16 times its thickness apart, centred on the red centre line, then lay the specimen flat across the rollers, centred and square to them. Let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/flexure-span-sixteen-to-one/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`tier`, `span_tier`, `sp_tier`, `zero_reason`, `span_mm`,
`span_error_mm`, `centre_off_mm`, `roller_y_mm`, `roller_x_mm`, each support's `tilt_deg`, `roller_heading_deg`,
`height_mm` and `inside_base`, `sp_tilt_deg`, `sp_heading_err_deg`, and at each roller `square_err_deg`,
`along_roller_mm`, `in_from_end_mm` and `underside_mm`, `sp_offset_mm`, `at_rest`, `moved_mm`, `hands_off`,
`zone_gap_mm`, `arms_unavailable`, the specimen's centre, orientation and ends, both supports' origins and
orientations and roller centres, both TCPs, pad-tip points and openings). Videos: `inspect-robots video <log>.json`.
The scorer keeps one trial's record per process and starts it over at every reset (rigstate's and mechkit's reset
counters, and the specimen or a support put back on its starting mark in a single step), so run one trial per
process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the specimen's pose, both supports' poses
and both grippers once per control step (20 Hz). From each support's base frame (the middle of its underside; x along
the roller, z up) it computes the middle of its roller's axis. A support is "standing" when it is upright within 5
degrees, the middle of its underside is within 2 mm of the base's top and inside the base's outline, and its roller is
within 5 degrees of x (parallel to the red line). The span s is the distance along y between the two roller axes (at their middles);
the offset e is how far the point midway between them is from the red line (on a real rig, half the difference of the
two rule readings from the line to each roller's axis). For the specimen the scorer takes the centre line of whichever
20 x 210 mm face is down and finds where, seen from above, it passes over each roller's axis: there its underside must
be within 2 mm of the roller's top, its whole 20 mm width must be on the roller's 50 mm length, and the roller must be
at least 5 mm in from the bar's end; the bar must lie flat (its 10 mm thickness within 5 degrees of vertical, so a bar
on its edge, 20 mm deep, does not count), within 10 degrees of square to each roller, with one roller on each side of
its middle. The specimen's offset c is the distance along the bar from its middle to the point midway between the two
crossings, which is half the difference of its two overhangs past the roller axes: what a technician measures with a
rule at each end. Its square error is the larger of its two angles from square to a roller.

Every score needs everything at rest (over the last 0.5 s no end of the specimen and no roller axis moved more than
2 mm, measured on positions: the simulator's reported velocities chatter while a part is gripped) and let go: each
gripper's finger zone at least 20 mm from the specimen and from both supports. rigstate gives no gripper heading, so a
gripper's fingers are taken to be anywhere round its axis: the zone is everything within reach of a pad's outer corner
(half the current finger gap plus the 8 mm pad thickness out, 13.5 mm across; 57 mm when fully open) around the 38 mm
stretch back from the point between the pad tips. A new trial is recognised at rigstate's or mechkit's reset, or when
the simulator puts the specimen back on its starting mark (within 4.5 cm of it, which covers a seeded reset's jitter)
or a support back on its starting mark (within 15 mm) in a single step. A simulator state with a non-finite number, a
zero quaternion or a missing field scores 0 for that step and leaves the scorer's record as it was.

| Outcome (at rest, let go, at the end; both supports standing, the specimen on both rollers) | Score |
|---|---|
| Span within 5 mm of 160 mm and its middle within 5 mm of the red line; specimen within 5 mm of the span's middle and square within 3 degrees (success once held 1 s) | 1.0 |
| Span as above; specimen within 10 mm but not both within 5 mm and square within 3 degrees | 0.75 |
| Span within 10 mm of 160 mm (16 +- 1 times the depth) and its middle within 8 mm, but not as above; specimen within 5 mm and square within 3 degrees | 0.75 |
| Span within 5 mm and middle within 5 mm; specimen more than 10 mm off the span's middle | 0.5 |
| Span only within 10 mm / 8 mm; specimen only within 10 mm | 0.5 |
| Span only within 10 mm / 8 mm; specimen more than 10 mm off | 0.25 |
| Span more than 10 mm off 160 mm or its middle more than 8 mm off the line (including the starting 64 mm); a support tipped, upside down, off the base, on the table, stacked or with its roller turned more than 5 degrees; the specimen on its edge, crosswise, more than 10 degrees out of square, held up or down on the base, partly off a roller's end, or with a roller under 5 mm from its end; still moving; a gripper on or within 20 mm of any of the three parts | 0 |

So the span and the specimen each earn up to half, and each has a tight and a wide limit; the wide span limit is the
standards' own 16 +- 1 times the depth. Every tier is a set-up the bar rests on by itself, its middle between the rollers (on a 160 mm span it
still crosses each roller at least 5 mm in from its end with c up to 20 mm). Unintended strategies earn nothing,
as the tests show: the bar laid centred across the supports left at 64 mm (0); held in place on the rollers by the
gripper (0); pressed down by an open gripper's pad (0); laid on its edge across the rollers (0); laid on the base
between the supports (0); propped from a roller down to the base (0); supports laid on their sides, or turned with
their rollers along y (0); supports and bar set on the table beside the base (0); one support stacked on the other
(0). Either face of the bar may be up, either end may point either way, and a support may face either way: they are
the same parts.

## One arm alone

The demo uses both arms: each moves the support on its own side (gripped across the block, straight down), and the
left arm also lays the specimen. With one arm idle the scripted demo scores 0.0 either way on the GPU (one trial each,
as the offline kinematic check predicted): `DEMO_IDLE_ARM=right` leaves the right support at 32 mm, so the left arm
lays the bar across a 113 mm span whose middle is 25 mm off the red line (0.0); `DEMO_IDLE_ARM=left` leaves the left
support at 32 mm and the specimen on its mat (0.0). That is the
script, not a property of the task: the left arm alone reaches both supports where they stand and where they go
(yamkin: straight down, yaw 0, at (0.31, -0.032) and (0.31, -0.080) m at working heights, clear) and the specimen's
middle, so one left arm moving both supports and then the bar can score 1.0. The right arm reaches both supports too,
but of the specimen only its near end: at the bar's grip height, straight down, it reaches no further left than
about y 17.4 cm at the bar's x (yamkin), so it reaches the bar only within about 50 mm of its near end and misses its
middle by about 56 mm; alone it would have to pick the bar 56 mm or more off its middle (or shift it first) and lay it
with that grip; that is not excluded either, so the right arm alone may also score. No claim
is made that the task needs both arms: its difficulty is working out the span from the specimen's thickness, setting both supports symmetric
about the red line to a few millimetres, and laying a 210 mm bar centred and square across them. (All scores are in
'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 40.3 s (806 control steps). The left arm comes down round the left support (fingers along y,
pad tips 6 mm above the base), grips it across its 30 mm block (commanded 8 mm narrower, over 1.5 s), lifts it 41 mm
off the base, moves it out to 79.5 mm from the red line, sets it down over 2 s with its underside aimed 1 mm above the
base, lets go, rises and goes to the specimen. The right arm waits aside until the left gripper has risen from the
fixture (14.5 s), then does the same with the right support. Meanwhile the left arm grips the specimen at its middle
across its 20 mm width (fingers along x, pad tips 3 mm above the mat), lifts it and holds it up over the mat until the
right gripper has risen from its support (29 s), carries it over the middle of the span with its underside 104 mm up,
lowers it over 2 s until its underside is aimed 1 mm above the rollers' tops, lets go (opening to 40 mm), rises and
moves aside. Two grippers are never over the fixture at the same time (closest approach of the two arms' collision
capsules 11 cm, yamkin). The arm's controller does not carry a load's weight: the compliance model (J K^-1 J^T of the
weight, joints 80/80/80/10/10/10 N m/rad, yamkin) puts a gripper holding a 0.13 kg support at its place pose
(-1.4, +-1.0, -3.0) mm off (the 1.0 mm outward, away from the red line) and the left gripper holding the 0.057 kg bar
at (-0.4, +0.4, -1.4) mm off over the span. Each place point is trimmed by about half the sideways part (the supports
0.5 mm inward, the bar 0.2 mm toward -y; `SUP_TRIM`, `SP_TRIM_Y` in `task.py`); the vertical part only makes each
part touch down a few millimetres before its slow descent ends. Offline (no sag) the span is 159 mm, the bar drops
1 mm onto the rollers at the release and success holds for its 1 s at step 793, once the hand has risen 20 mm clear.
On the GPU the supports came to rest 81.1 and 81.9 mm from the red line (1.6 and 2.4 mm further out than aimed), so
the span was 163.0 mm (3 mm long, inside the 5 mm limit) with its middle 0.4 mm off the line; the bar lay 0.5 mm off
the span's middle and within 0.8 degrees of square, and success held at step 793. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two support blocks back on the base
centred at x 31.0 cm, y +3.2 and -3.2 cm, upright with their rollers along x (64 mm apart); the specimen back on the
black mat centred at x 19.0 cm, y +23.0 cm, lying flat with its length along y; everything else is fixed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo reaches success after about 40 s of simulated time and stops by 43 s); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 90 x 300 x 10 mm steel plate as the base, with a 2 mm red tape or paint line across its middle and a painted or
taped scale of short marks every 10 mm along its near edge (longer at 50 and 100 mm); two 50 x 30 x 25 mm aluminium
blocks, each with a 10 mm steel dowel pin, 50 mm long, glued along a shallow groove in its top; a 40 x 240 x 3 mm black
rubber mat; and a 210 x 20 x 10 mm glass-filled nylon bar (any 10 mm thick plastic bar of that size will do). Clamp
the plate and the mat to the bench; nothing is powered. Score with a steel rule between the dowels' axes and from the
red line to each, a rule at each end of the bar for its two overhangs, a protractor or square, and a photo from above
showing both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
