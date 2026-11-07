# Set ripple-tank barriers for a two-slit demo: two centred slits, sealed at the walls

School and college science technicians set up the demonstrations teachers use, before the lesson, and the ripple
tank is the standard demonstration of wave interference: a vibrating straight dipper sends plane waves across
shallow water toward a barrier with two slits, and a lamp above projects the pattern onto a screen below. For a
clean two-slit pattern the barrier pieces go in one straight line across the tank, parallel to the dipper (so both
slits are driven in step), closed against both side walls (so no wave leaks round the ends), leaving two slits about
one wavelength wide, a few wavelengths apart and centred. Here a small dry bench tank (200 mm wide inside) is fixed
on the table with its dipper and beach, and three black aluminium barrier bars stand beside it: two 70 mm long, one
in front of each arm, and one 30 mm long. Their lengths add up to 170 mm, so in a line across the tank they leave
30 mm of openings: closed against both walls with the short bar in the middle, two 15 mm slits whose centres are
45 mm apart. The score is what the technician checks with a rule before switching the dipper on: the line, both
slits' widths, their spacing and centring, and the gaps at the walls.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three bars' states, with a small tracker for stillness and resets |
| `tests/test_scoring.py` | Offline tests of every tier and combination, every zero condition, the limits, resets, bad states, the exploits we tried, the task file and the offline checker (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the ripple tank for the two-slit demonstration: stand the three black barrier bars in one straight line across the tank, parallel to the white dipper and 120 mm beyond it (centre line to centre line), closed against both side walls, so that they leave two slits about 15 mm wide, 45 mm apart centre to centre, centred on the tank." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/ripple-tank-double-slit/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the ripple tank for the two-slit demonstration: stand the three black barrier bars in one straight line across the tank, parallel to the white dipper and 120 mm beyond it (centre line to centre line), closed against both side walls, so that they leave two slits about 15 mm wide, 45 mm apart centre to centre, centred on the tank." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/ripple-tank-double-slit/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`line`, `standing`, `parallel`, `at_distance`,
`inline_spread_mm`, `order`, `slits_mm`, `slit_centres_mm`, `slit_sep_mm`, `slit_mid_mm`, `wall_gaps_mm`,
`slits_ok`, `centres_ok`, `walls_ok`, `still`, `moved_mm`, and per bar `stands`, `low_mm`, `tilt_deg`,
`axis_err_deg`, `inside`, `beyond_dipper_mm`, `x_mm`, `y_mm`, `ymin_mm`, `ymax_mm`, plus each bar's pose).
Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process; a new trial is
recognised when the simulator puts all three bars back on their marks in one step.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the three bars' poses, once per
control step (20 Hz), and judges the end state. A bar's outline is the shadow of its eight corners seen from above
(turned and tilted as it is, not its bounding box). A bar stands when its lowest point is within 2 mm of the tank
floor's top, its 25 mm side within 10 degrees of vertical (either way up: the bar is a plain box) and its outline
inside the tank's inside faces, with 3 mm allowed for contact with a wall. The three form the line when all stand,
each one's length is within 5 degrees of across the tank (either end either way), their centres are within 6 mm of
each other along x and each centre is 110 to 130 mm beyond the dipper's centre line. Ordered across the tank by
their centres, the bars leave four openings measured along y between outlines: the left wall gap, slit 1, slit 2 and
the right wall gap (bars that touch, or whose outlines overlap, leave a slit of 0).

| Outcome | Score |
|---|---|
| No line: a bar missing, on the table or outside the tank, lying on a side or an end, more than 2 mm off the floor, tilted over 10 degrees, turned over 5 degrees, centres more than 6 mm apart along x, or the line outside 110-130 mm from the dipper | 0 |
| The line, nothing else | 0.25 |
| + (S) both slits 10 to 20 mm wide | + 0.25 |
| + (C) the slits' centres 40 to 50 mm apart and the point midway between them within 5 mm of the centre line | + 0.25 |
| + (W) both wall gaps 5 mm or less | + 0.25 |

All eight combinations of S, C and W occur (the tests build each one), so every score from 0.25 to 1.0 occurs
(0.5 and 0.75 in three ways each). With the bars in the right order (long, short, long), C follows from S: two slits of
10 mm or more leave at most 10 mm for the wall gaps, so their centres are at least 40 mm apart and their midpoint
within 5 mm of the centre line. What C adds is the arrangement: a short bar at a wall and a long bar in the middle
can leave two good 15 mm slits and closed walls, but the slits' centres are then 85 mm apart and 20 mm off the
centre line, so the pattern falls to one side (0.75). Likewise the openings always add up to 30 mm, so a slit wider
than 20 mm leaves the other under 10 mm: the 20 mm limit is the real-world window and is stated, but S is never lost
on it alone. Examples (all in the tests): long bars 1 mm from the walls and the short bar centred, 1.0 (slits 14 mm,
centres 44 mm apart); bar A 6 mm off its wall and the short bar 3 mm right of centre, 0.75 (no W); the long bars
against the walls and the short bar 6 mm left of centre, 0.75 (no S: slits 9 and 21 mm); bar A 6 mm off its wall
and the short bar 4 mm left of centre, 0.5 (slits 5 and 19 mm); both long bars 10 mm off their walls, 0.25.

The score is read at the last step. The trial ends early once the 1.0 conditions have held for 2.5 s with neither
end of any bar moving more than 2 mm in the last 0.5 s (positions, not the simulator's velocities); otherwise it is
scored as it stands at the run limit. A simulator state with a non-finite number, a zero quaternion or a missing bar
scores 0 for that step and leaves the scorer's record of the bars as it was.

Unintended strategies earn no more than they should, as the tests show: the short bar at a wall instead of in the
middle, 0.75 at most; propping a long bar's end on the wall's top so its shadow closes the wall gap, 0 (its outline
reaches over the wall); turning a long bar 8 degrees so it spans more of the tank, 0; a staggered line (the short bar
10 mm in front), 0; bars laid flat, standing on an end, stacked, or one left out of the tank, 0; pushing all three
together against one wall, 0.25 at most. The scorer cannot see the grippers, so a bar held still within 2 mm of the
floor counts as standing (`fidelity` says so); in the scripted demo the 2.5 s of stillness is longer than the time
from the last bar touching the floor until the fingers are open, so the trial ends after they have let go.

## One arm alone

The demo uses both arms: the left arm lays bar A against the left wall and later the short bar in the middle, the
right arm lays bar B against the right wall. With one arm idle, the other still sets its own bars, but no line forms
without the third bar, and on the GPU each arm alone scored 0: with the left arm idle, the right arm stood bar B
against the right wall and bars A and C stayed on the table; with the right arm idle, the left arm stood bar A and
the short bar on the line and bar B stayed on the table. That zero comes from the layout, not from the act.
Gripping straight down, the left arm reaches bar A's and the short bar's starting places and every place on the
line, but not bar B's start, 54.5 cm across from its base (43 mm short in our inverse kinematics of the arm, with
joint limits); the right arm reaches only bar B (the short bar's start is 6 mm out of its reach, bar A's 43 mm). A single arm able to reach all three bars, or one that first pushed the far bar within
reach, could do the whole set-up: the task does not need two arms, and we do not claim it does. The real
difficulty is placing three loose bars in one line to within a few millimetres, with the slits read in millimetres.
(The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 39.4 s (788 control steps), both arms on one clock, one gripper in the tank at a time. Every
grip is across a bar's 20 mm thickness, fingers closing along x (yaw -90 degrees left, +90 right) and commanded 8 mm
narrower (12 mm), opened to 36 mm round the bar, with the tool point 20 mm above the bar's bottom (pad tips 7 mm
above it), so the pads stay in front of and behind the line, never in a slit or at a wall. Every open or close
takes 1.5 s and every final descent is slow. In order:

1. The left arm picks bar A, carries it with its bottom 80 mm up (over the 15 mm walls), sets it down on the line
   (x 30.0 cm) 6 mm in from the left wall, lowers it slowly to 4 mm above the floor (8 mm below the wall's top),
   slides it out (1.5 s) toward a point 1.5 mm past the wall's inside face, so that the wall's face stops it, lowers
   it the last 3 mm, lets go 1 mm above the floor and rises. Setting it down 6 mm in keeps it clear of the wall's top
   whatever the arm's sideways error; driving it past the face registers it against the wall (gap 0) unless the arm
   falls more than 6.5 mm short, and an arm that is spot on only presses it on the wall (about 0.5 N).
2. Meanwhile the right arm waits over bar B, picks it as the left rises, and lays it against the right wall the same
   way, while the left arm picks the short bar and holds it over its own side.
3. Once the right arm has risen out of the tank, the left arm lowers the short bar straight onto the floor at the
   centre of the line, lets go, rises and parks; the right arm parks over its own side.

Against both walls the slits are 15 mm wide and 45 mm apart centre to centre, and the short bar may land up to 5 mm
off the centre line and still score 1.0 (the tests check these margins). Offline there are no contacts, so the
checker draws each long bar 1.5 mm into its wall, which the scorer reads as touching it (slits 16.5 mm). Offline
(no physics), success has held for its 2.5 s at step 750; the scripted policy stops at step 849. On the GPU the
three bars formed the line at step 692 and the trial ended at step 751, with wall gaps of 0.06 and 0 mm, slits of
14.6 mm each, their centres 44.8 mm apart and their midpoint 0.05 mm off the centre line. The tool points
stay at least 28 cm apart and the arms' collision capsules at least 10 cm; the smallest joint margin is 16.5 degrees
(joint 4). The run limit is 1,800 steps (90 s), more than twice the demo's time.

What to watch on the GPU (`TASK_TRACE`): after each long bar is let go, `wall_gaps_mm` and its `y_mm`. It should end
against its wall (gap 0). If a wall gap is over 2 mm the arm fell short: move that bar's `TRIM` in `task.py` toward
its wall by the gap (the wall stops a generous trim). If the short bar lands more than 3 mm off the centre line, trim
it back by its offset. Each bar's `low_mm` (3.0 on the floor), `tilt_deg` and `axis_err_deg` show whether it stands
square; a long bar's `tilt_deg` over 3 degrees after the push would mean the wall turned it in the grip (lower
`PUSH_PAST` in `layout.py`). In the GPU run no trim was needed: both long bars ended against their walls, the short
bar 0.07 mm off the centre line, and every bar stood square (`tilt_deg` 0, `axis_err_deg` under 1 degree).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the three bars back on the table on a long
narrow face, lengths along y: bar A centred at x 22.0 cm, y +24.0 cm; bar B at x 22.0 cm, y -24.0 cm; the short
bar C at x 14.0 cm, y +20.0 cm. The tank, dipper and beach are fixed. Nothing on the table lies where the arms' start
pose covers it in the top camera's first frame.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 39 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Any bench ripple tank with a straight dipper, or a tray with a flat floor 260 x 200 mm inside and walls 15 mm tall;
a 10 x 12 mm bar laid across it 3 cm from the near end as the dipper and a strip of foam along the far end as the
beach. The barrier bars are 20 x 25 mm aluminium bar stock (or hardwood) cut to 70, 70 and 30 mm and painted black.
Score with a steel rule laid along the line (slits and wall gaps along y, the line's distance from the dipper) and a
photo from above for the line and the bars' angles. Trials are run dry, with the dipper unpowered and no lamp:
nothing near the arms is wet or live (for the lesson itself, 5 to 8 mm of water goes in afterwards and the bars stay
where they stand).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

A settling run with no plan (160 steps) left every bar on its mark. With either arm idle no line forms, so both
one-arm runs score 0 (see 'One arm alone'). In the Dream-1 trial no bar reached the tank: by the run limit bar A
had been nudged about 2 mm and turned about 3 degrees on the table, and the other two had not moved.
