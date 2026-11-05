# Lay three reinforcing bars on the bolsters at 100 mm centres with 20 mm end cover

Rebar workers lay a slab's bars at the spacing and concrete cover the drawings call for, and the spacing and cover
are checked with a tape before the pour. Here a small slab is formed on the table (four fixed plywood boards), with
two grey plastic bolsters (bar supports) inside it, and three rust-brown bars lie along x on timber racks outside
it: two on the left arm's side, one on the right arm's. Each bar must be picked from its rack, turned a quarter
turn, and laid across both bolsters, so that the three end up parallel, 100 mm apart centre to centre where they
cross each bolster, with at least 20 mm between every bar end and the side forms. The score is read from where the
bars lie at the end. Each arm lays the bars on its own side: gripping from above, neither reaches the other side's
bars, so one arm gripping from above lays at most its own side's bars (the left two, 0.5; the right one, 0.25).
Whether a side grip could do more is discussed under "One arm alone".

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three bars' states, with a small tracker for the trace |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |

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
  --instruction "Lay the three rust-brown bars across the two grey bolsters, parallel and 10 cm apart centre to centre, with at least 2 cm between each bar end and the wooden side forms." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rebar-mat-spacing/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the three rust-brown bars across the two grey bolsters, parallel and 10 cm apart centre to centre, with at least 2 cm between each bar end and the wooden side forms." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/rebar-mat-spacing/dream
```

Run one trial per process (the default). The score depends only on where the bars lie; the small tracker in the
task module keeps diagnostics for the trace and recognises a new episode in the same process only when the
simulator puts every bar back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`mat_order`, `gaps_mm` at each bolster, `spacing_dev_mm`,
`cover_mm`, `laid`, `on_bolsters`, `heading_err_deg`, the first step each score was reached, and each bar's
centre, tilt, heading error, end covers and speed). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the three bars' poses and velocities,
once per control step (20 Hz), and scores the state at hand: the score is read at the last step. The forms and
bolsters are fixed, so their faces come from `layout.py`.

A bar rests on a bolster when, seen from above, its underside (its real, turned and tilted outline, not its
bounding box) overlaps the bolster's top by at least half of a square crossing of their widths (16 x 12 mm, so
96 square mm), and at the centre of that overlap its underside is within 3 mm of the bolster's top. It is on the
bolsters when it is flat (tilted under 5 degrees) and rests on both, and laid when it is also within 3 degrees of
y. Its end cover is the smaller distance from its outline to the two side forms' inner faces. The mat is the bars
on the bolsters, in order along x, and the spacing of two neighbouring laid bars is read as a tape laid along each
bolster reads it: the distance along x between their centre lines where they cross that bolster's centre line.
Both readings count, so two bars whose centres are 100 mm apart but which are not parallel do not pass. The bar's
section is square, so it may lie on any of its four long faces and either way round.

| Outcome | Score |
|---|---|
| Any bar off the table, standing on an end or an edge (tilted over 30 degrees), or on or across a form board | 0 |
| All three laid, end cover 20 mm or more, both spacings 100 +- 5 mm at both bolsters; held still for 3 s | 1.0 |
| All three laid, end cover 10 mm or more, both spacings 100 +- 12 mm at both bolsters, not 1.0 | 0.75 |
| Two neighbouring bars in the mat laid with 20 mm cover and 100 +- 5 mm apart at both bolsters | 0.5 |
| At least one bar laid with 20 mm cover | 0.25 |
| Anything else (no bar laid: along x, turned more than 3 degrees, on one bolster, piled up) | 0 |

The trial ends early (success) once the 1.0 conditions have held with all three bars still (under 1 cm/s and
0.5 rad/s) for 3 s, longer than the settle and release after the last set-down (0.5 + 1.5 s), so it cannot end
while the last bar still hangs in a closed gripper; otherwise it is scored as it stands at the run limit. Any bar
may go in any place and the mat may lie anywhere along the bolsters. A state with a non-finite number or a zero
quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

'On or across a form' means the bar's outline seen from above overlaps a form board while its lowest point is no
more than 20 mm above the board's top: a bar sticking out of the slab's edge. A 400 mm bar cannot lie along x
inside the 320 mm long slab, so a bar laid along a bolster always ends on the front and back forms (0). A bar left
on its rack, or anywhere else on the table outside the forms, is simply not laid: it does not zero the trial,
which is what lets a partial mat (two bars, one bar) earn its partial score.

Unintended strategies earn nothing extra, as the tests show: bars along x (on the forms: 0); bars piled on a laid
bar (only the bottom one is on the bolsters: 0.25); a mat turned 12 degrees with its bars 100 mm apart square to
each other (not along y: 0); a middle bar turned 2.5 degrees with its centre exactly on the mark (the tape at
each bolster reads 100 -+ 5.2 mm: 0.75); a bar dropped with one end on the table and the other over a bolster
(not on both, and across the form: 0); a good pair with a third bar on the bolsters between them (not neighbours:
0.25). The scorer cannot see the grippers, so a bar held still within 3 mm of both bolsters counts as resting on
them (see `fidelity`); held 3.5 mm up it does not.

## One arm alone

The demo splits the work by reach: the left arm lays its two bars (x 26 and 16 cm), the right arm its one
(x 36 cm). `DEMO_IDLE_ARM=right` (the left arm alone) lays two neighbouring bars 100 mm apart with 30 mm of cover
and leaves the right bar on its rack: 0.5. `DEMO_IDLE_ARM=left` (the right arm alone) lays one bar: 0.25. Gripping
from above, neither arm reaches the other side's bars: the straight-down envelope ends about 10 cm past the midline,
and the other side's bars lie 30 to 38 cm past it. We do not claim one arm is strictly unable to finish: tipped
30 to 70 degrees sideways, an arm's TCP can reach the other side's bars along much of their length (position and
orientation IK with the arm's collision capsules clear of the table and its own base), so a policy able to grip a
16 mm bar from the side, near the table, could in principle fetch the third bar itself. The scripted demo grips
from above only. (Offline kinematic predictions; the GPU runs will be recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 45.4 s (908 control steps). Each arm grips a bar across its 16 mm width at mid-height,
7 cm from its centre toward the arm bases (fingers open to 50 mm, commanded to 8 mm to close, about 20 N per
finger; the pad tips 15 mm above the table, beside the bar and clear of the racks), lifts it 7 cm (the bar's
underside 92 mm up, clear of the forms, racks and laid bars), turns it a quarter turn over its rack (left arm yaw
0 to -90 degrees, right arm 0 to +90), carries it over its place and sets it down along y, centred on y 0: its
ends 30 mm from the side forms, the grip 7 cm to the arm's own side of the midline. Every final descent is a
separate 1.5 s move over the last 3 cm, each bar is let go with its underside aimed 2 mm above the bolsters after a
0.5 s settle, and every gripper open or close takes 1.5 s. The order keeps one bar at a time over the mat:

1. The right arm lays its bar at x 36 cm (the far place, done first so the left arm never reaches over a laid bar)
   and moves back to its own side.
2. The left arm, waiting over its inner bar, grips and lifts it as the right arm lets go, and lays it at x 26 cm.
3. The left arm lays its outer bar at x 16 cm and moves back to its side.

The TCPs stay at least 39.9 cm apart and the arms' collision capsules at least 25.5 cm (offline). The smallest
joint margins are 10.4 degrees (left joint 6, turning the outer bar over its rack) and 17.1 degrees (right joint
4, carrying at the far place). The payload is not gravity-compensated: by our stiffness estimate (J K^-1 J^T,
the arm's joint springs), the 0.10 kg bar held 7 cm off centre makes the arm sag about 2 to 3 mm at the place
(the bar's centre 2.8 to 3.9 mm), 1.1 to 1.5 mm toward the arm bases and 2.1 to 2.8 mm toward the arm's own side,
and tilts it about 0.7 degrees. All three bars shift about the same along x, so the spacing changes by less than
0.5 mm; the cover drops to about 27 mm at one end. Each bar meets the bolsters about at the end of its descent,
pressing on them at most a few mm of spring, so success may start counting while the gripper still holds the last
bar; the 3 s hold covers the settle and release. In the offline proxy success is held at step 900; the scripted
policy stops at step 969. The run limit is 2,400 steps (120 s), which leaves Dream-1 more than twice the demo's
time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: each bar back on its side's two racks,
lying flat along x and centred at x 28.0 cm (two on the left at y +30.0 and +38.0 cm, one on the right at
y -32.0 cm); the forms, bolsters and racks are fixed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 45 s of simulated time,
when success has held for 3 s; a run to the limit is 120 s); the first start of Isaac Sim on a new instance takes
several minutes more.

## Real-world build

Four 10 mm plywood strips 25 mm tall screwed into a rectangle 320 x 460 mm inside, two grey 12 x 20 mm plastic (or
grey-painted hardwood) strips 280 mm long glued inside it 240 mm apart as the bolsters, and four 20 x 20 mm timber
blocks glued down outside it as the racks: two 140 mm long on the left, two 80 mm long on the right, each pair
26 cm apart centre to centre along x (positions in `task.json`'s `setup`). The bars are 16 x 16 mm square
aluminium tubes with a 1.6 mm wall, 400 mm long (about 100 g each), painted rust brown; real 16 mm steel bars would
weigh 0.8 kg each, more than the arms should carry.

Scoring a real trial: lay a tape along each bolster from the front form and read where each bar's centre line
crosses it (the middle of its two edges). The spacing is the difference between neighbouring bars' readings on the
same tape; a bar is within 3 degrees of y when its readings on the two tapes (240 mm apart) differ by 12.6 mm or
less. Read the end cover with a rule from each side form's inner face to the nearer corner of each bar end. A bar
rests on a bolster when at least half its width is on the bolster and a 3 mm feeler gauge does not pass under it
there.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
