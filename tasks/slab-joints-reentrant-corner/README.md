# Lay joint strips in an L-shaped slab so every panel is a near-square rectangle

Cement masons and concrete finishers (O*NET 47-2051) lay out contraction (control) joints so that a ground slab cracks
along them and nowhere else. Industry guidance, such as the Portland Cement Association's, keeps panels as square as
possible (the long side no more than about 1.5 times the short side), spaces joints no more than about 24 to 36 times
the slab's thickness apart, and runs a joint out of every re-entrant (inside) corner, because a slab cracks diagonally
from an inside corner. Pre-formed joint strips (crack inducers) are laid on the sub-base before the pour. Here a 1:25
model of a 100 mm slab ready to pour stands on the table: an L-shaped form of dark plywood boards on a thin stone
sub-base, a long leg 120 x 460 mm (3 x 11.5 m) and a return leg 120 x 100 mm (3 x 2.5 m) on its left end toward the
arms, and four loose dark grey joint strips (18 mm wide, 12 mm thick) beside it, two on each side: three 112 mm long
and one 83 mm. The strips must be laid flat, square to the forms, with both ends against a form or another strip, so
that every panel they mark out is a rectangle no longer than 132 mm (3.3 m) and no more than 1.5 times as long as it
is wide. Nothing marks where they go. The rules alone place them: the panel round the inside corner is L-shaped unless
a joint runs out of the corner, so one strip must run from it across the long leg; the return leg and the long leg's
end beside it make a 240 mm panel, so the short strip must cross between the legs, from that strip to the left form
(91 mm, where it fits with 4 mm at each end; a long strip does not fit there, and the short one falls 18.5 mm short of
the forms across the long leg); and the remaining 360 mm of the long leg needs two strips across it, each panel 80 to
132 mm. The score is 0.2 per good panel, what a site engineer checks with a tape from the forms and a top photo. Both
arms are used: each lays the two strips on its own side (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four strips' poses and the grippers' state (which strips count, the lines they draw, the panels), with a small tracker of the last half second |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and limit, every zero condition, the exploits we tried, resets, NaN states, the gripper hook, the demo plan and the offline checker on this folder (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

A strip counts only once it is laid (both grippers clear of it), and `yam_arena` does not pass the grippers' state
to a task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import,
to take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each
control step and to start a new trial at each reset. It reads only state the simulator already has and changes
nothing the embodiment does. If it cannot read the arms it prints why once; the let-go condition is then taken as
met on those steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the four grey joint strips flat on the sub-base inside the L-shaped slab form, each running square to the forms with both ends against a form or another strip, so that the joints divide the slab into panels that are all rectangles no longer than 132 mm and no more than 1.5 times as long as they are wide. Then let go of them." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/slab-joints-reentrant-corner/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the four grey joint strips flat on the sub-base inside the L-shaped slab form, each running square to the forms with both ends against a form or another strip, so that the joints divide the slab into panels that are all rectangles no longer than 132 mm and no more than 1.5 times as long as they are wide. Then let go of them." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/slab-joints-reentrant-corner/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`good_panels`, `panels` with each one's extent in mm and
whether it is a rectangle and good, `lines_mm`, `still`, `moved_mm`, `first_full_step`, `arms_unavailable`, and for
each strip `counts`, `why`, `x_mm`, `y_mm`, `axis`, `turn_deg`, `tilt_deg`, `low_mm`, `inside`, `let_go`,
`left_zone_mm`, `right_zone_mm`, `line_mm`, `line_raw_mm`, `end_gaps_mm`, `ends_meet`; the strips' poses, both TCPs
and openings). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts
it over at every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about a minute: one test plays the whole
demo and both one-arm variants through the offline checker).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four strips' poses and both grippers'
points once per control step (20 Hz) and scores the end state in three steps.

1. **Which strips count.** A strip counts as a joint when it lies flat on an 18 mm face (its 12 mm thickness within
   5 degrees of vertical), its lowest point at most 3 mm above the sub-base, its length within 5 degrees of x or y,
   its outline seen from above inside the slab (within 2 mm), let go (both grippers' finger zones 20 mm or more from
   it), and each of its ends within 8 mm of a form's inside face (measured straight along the strip) or of another
   counted strip's outline lying beyond that end (up to 2 mm into it allowed, for contact). Strips that fail are
   dropped and the rest checked again, so a strip that only meets a dropped one is dropped too. Two strips whose ends
   meet at a corner, each within 8 mm of the other, hold each other.
2. **The lines.** Each counted strip draws its centre line, carried across its end gaps to the form face it meets
   or to the centre line of the strip it meets (to that strip's end if the two lie end to end). A centre line within
   6 mm of a line through the inside corner (y +12.0 cm for a strip along x, x 26.0 cm along y) is taken to lie on
   it: a strip laid "out of the corner" a few millimetres off still closes it.
3. **The panels.** The lines and the forms' inside faces divide the slab into panels (exactly, on the grid of every
   line's coordinates). A panel is good when it is a rectangle (no inside corner), its longer side is at most
   132 mm, and its longer side is at most 1.5 times its shorter side.

| Outcome | Good panels | Score |
|---|---|---|
| Nothing laid, or nothing that counts (the whole slab is one L-shaped panel) | 0 | 0 |
| One strip across the long leg 120 mm from the right end | 1 | 0.2 |
| A strip out of the inside corner and the short strip between the legs (the return leg, the long leg's end beside it) | 2 | 0.4 |
| Those two and one strip across on y 0 (the last 240 mm is too long) | 3 | 0.6 |
| Strips across on y +12, 0 and -12 cm, none between the legs (the return leg and the long leg's end make one 240 mm panel) | 3 | 0.6 |
| All four, one cross strip 30 mm off (one panel 150 mm long) | 4 | 0.8 |
| Strips across on y +12, 0 and -12 cm and the short strip between the legs: three 120 mm squares, two 120 x 100 mm | 5 | 1.0 |

Score = 0.2 x the number of good panels. Four strips (no two can cross flat on the sub-base) make at most five
panels, so the score is at most 1.0. Every limit is inclusive. The score is read at the last step; the trial ends
early once 1.0 has held for 1 s with no strip's end having moved more than 2 mm over the last 0.5 s. A new trial is
recognised from rigstate's reset hook, or, as a fallback, when the simulator puts the strips back in a single step
and all four are then on their starting marks (within 4.5 cm). A simulator state with a non-finite number, or a
missing strip, scores 0 for that step and leaves the record of the trial as it was.

How much room there is: a 112 mm strip across the long leg has 4 mm of play at each end, and the short strip between
the legs 4 mm at each end, so anywhere it fits it is within 8 mm at both ends. Across the long leg, with the corner
strip on y +12 cm, the strip near y 0 may lie anywhere from y -1.2 to +1.2 cm and the one near y -12 cm from -13.2 to
-10.8 cm (each panel 108 to 132 mm). The corner strip may lie up to 6 mm above the corner's line (snapped) or up to 8
mm below it (the short strip, from it to the left form, then runs out of the corner); the short strip may lie from x
24.9 to 27.2 cm, as long as its end is within 8 mm of the corner strip and both panels it separates stay 132 mm or
shorter. The tests check each of these limits on both sides.

Unintended strategies earn nothing they should not, as the tests show: a strip stacked on another, on a 12 mm edge,
standing on end, lying on a form, held in the air, or still held (or with an open gripper round it) divides nothing; a
strip along a form either does not count (its end points into the open return leg) or cuts off a sliver that is not a
good panel; the short strip cannot span the long leg (18.5 mm from a form at each end at best) or the return leg (8.5
mm), so laying all four across the long leg scores at most 0.6; the short strip between the legs with no corner strip
has nothing within 8 mm of its lower end and divides nothing; a long strip (112 mm) does not fit between the legs (100
mm): it lies over the forms, outside the slab. Any two strips, wherever they lie, make at most two good panels (three
good panels cover at most 3 x 132^2 = 52,272 mm2 of the slab's 67,200 mm2), and three strips across the long leg with
none between the legs at most three (both checked on random layouts).

## One arm alone

Each arm lays the two strips on its own side, and with one arm idle the other lays only those: 0.4 either way (the
scripted demo with `DEMO_IDLE_ARM=left` and `right` scored 0.4 and 0.4 in simulation on the GPU, as the offline
check predicted). Two strips can never
make more than two good panels, as above, so 0.4 is the most one arm can score with its own strips. The other side's
strips start beyond its straight-down reach: with the tool point 13 to 20 mm above the table the left arm misses
every grip point on the right strips (x 17.4 to 28.6 cm, y -33.1 to -40.9 cm) by 60 mm or more, and the right arm
misses the left strips by 42 mm or more (inverse kinematics of the arm's URDF with joint limits, grip headings every
15 degrees). With the gripper tipped 45 to 80 degrees from vertical the tool point can reach them (in the same
model, clear of the table), so an arm that dragged the other side's strips closer first could in principle score
more; that has not been tried, and we do not claim the task needs two arms. Laying a strip does not in itself need
two hands; the task uses both because the strips start on both sides.

## Demo and run limit

The scripted demo takes 41.0 s (820 control steps). Every grip closes over 1.5 s with the fingers commanded 8 mm
narrower than the 18 mm strip (10 mm), every release opens over 1.5 s to 40 mm, and every set-down slows for its last
20 mm, letting go with the strip's underside 0.5 mm above the sub-base.

- Each arm opens to 40 mm (11 mm each side of a strip), comes down over a strip with its fingers across the strip's
  width, the tool point 16 mm above the table (pad tips 3 mm up, gripping its lower 3 to 12 mm), closes, and lifts it
  until its underside is 74 mm up (59 mm over the forms).
- 0 to 15.4 s, both arms at once, their tool points 24 cm apart: the left arm lays its 112 mm strip out of the inside
  corner across the long leg (centred at x 32.0 cm, y +12.0 cm) and the right arm lays its inner strip across the long
  leg 120 mm from the right end (x 32.0, y -12.0 cm). Each ends 4 mm from each form.
- 15.4 to 30.8 s: the right arm lays its outer strip across the long leg on the midline (x 32.0, y 0), while the left
  arm picks the short strip and holds it 9 cm up over its start.
- 30.8 to 41.0 s: the right arm parks over its own side; the left arm carries the short strip over the line between
  the legs, turning it along y (grip heading -90 degrees), and lays it centred at x 26.0, y +17.45 cm, 4 mm from the
  corner strip and 4 mm from the left form; it rises and parks over its own side.

Each place point is 4 mm from what the strip butts against on either side, half the 8 mm allowed. The load is 0.033 kg
at most, so the arms' sag under it is well under a millimetre (the program's GPU runs show 3 to 7 mm only under 1 to 3
N). Offline, success has held for its 1 s at step 802; the scripted policy would stop at step 881. The run limit is
1,600 steps (80 s), about twice the demo.

Our offline kinematic check (NumPy, no physics; not part of this folder): worst IK error 0.08 mm, smallest joint
margin 14.7 degrees (joint 4), tool points at least 24 cm apart, the arms' collision capsules at least 6.9 cm apart
(elbows, while both lay their first strips), pad tips never lower than 3 mm above the table.

Things to look at in the GPU runs: in the settle run nothing moves (the strips lie flat). After each release, the
strip's `counts` true, `end_gaps_mm` near 4 and 4 (if one end is more than 3 mm off, move that place point in
`task.py` by the difference), `tilt_deg` and `turn_deg` under 2, `low_mm` near 1 (the sub-base's top); for the short
strip `ends_meet` the left long strip and a form; at the end `good_panels` 5; `arms_unavailable` 0 (otherwise let go
silently passes: do not submit).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the four strips back flat on the table, each
on an 18 mm face with its length along x: a 112 mm strip at x 23.0 cm, y +34.0 cm and the 83 mm strip at x 23.0 cm,
y +40.0 cm; the other two 112 mm strips at x 23.0 cm, y -34.0 cm and x 23.0 cm, y -40.0 cm. The forms and the
sub-base are fixed. Nothing in the start layout lies where the arms' start pose hides the table from the top camera.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is about 41 s of simulated time until success ends
it; a full Dream-1 run is 80 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Strips of 10 mm film-faced plywood ripped 15 mm tall, screwed or clamped to the table to the six positions in `setup`
(inside faces: a long leg 120 x 460 mm and a return leg 120 x 100 mm). A sheet of light grey 1 mm card or fine
abrasive paper cut to the L as the sub-base. Four rigid dark grey PVC bars, 18 x 12 mm, cut to 112, 112, 112 and 83 mm
(33 g, 33 g, 33 g and 25 g). To score: a tape from the forms' inside faces to each strip's centre line and ends, a
square against a form, and a top photo; then measure each panel between the lines and the forms.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.4 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.4 |
| Dream-1 | 0.0 |
