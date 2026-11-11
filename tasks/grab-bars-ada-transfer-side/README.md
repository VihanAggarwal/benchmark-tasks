# Fit an accessible toilet's two grab bars where ADA 604.5 puts them

Carpenters, maintenance workers and building inspectors fit and check grab bars in accessible toilet rooms to the
2010 ADA Standards for Accessible Design. Section 604.5 sets the bars from the toilet, not by eye: the side-wall bar
is at least 42 in long, starts at most 12 in from the rear wall and reaches at least 54 in out from it (604.5.1); the
rear-wall bar is at least 36 in long and runs from the toilet's centreline at least 12 in toward the side wall and at
least 24 in toward the open side, where a wheelchair user transfers (604.5.2). Two common errors are the shorter bar on
the side wall and a rear bar centred on the toilet, which leaves the transfer side short. Here a floor plan of the room
is laid out on the table at 1 in = 4 mm: two grey kerbs are the rear and side walls, the toilet is painted on the floor
with its centreline 17 in from the side wall (604.2 allows 16 to 18 in), and two aluminium grab bars (48 in and 40 in
at scale) lie in the room, each parallel to the wrong wall. The robot has to work out which bar goes on which wall
(only the 48 in bar can meet the side-wall rule) and where the rear bar must sit (off centre, toward the open side),
lay each bar along its wall about 1-1/2 in off the kerb and let go. The score is what an inspector reads with a tape.
This is a single-arm task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the code's numbers |
| `scoring.py` | The scorer: plain functions of the bars' poses and the grippers' state, with a small tracker of each bar's last second |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every condition and limit, the shortcuts and exploits we tried, broken states, resets, the demo plan and the text (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses `yam_arena`'s own boxes and
an upright cylinder (the painted bowl), the standard `task_score` scorer, and one helper module next to `task.py`.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks. `rigstate.py`
(imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of both arms (tool
point, the point between the pad tips, finger opening, in the objects' frame) before each control step and to start a
new trial at each reset. If it cannot read the arms it prints why once; the let-go condition is then taken as met on
those steps, and the trace counts them (`arms_unavailable`). It reads only state the simulator already has and changes
nothing else the embodiment does. The technique (wrapping the embodiment's step and reset from a task file) follows
the one first shown in the program repository's `tasks/bearing-press`; this is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (right arm; the left arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This floor plan of an accessible toilet room is drawn at 1 in = 4 mm. The grey kerbs are the rear wall behind the toilet and the side wall beside it; the white shapes are the toilet's bowl and tank. Fit the two grab bars (40 in and 48 in long), one on each wall, where the 2010 ADA Standards require them for this toilet. Lay each bar flat on the floor along its wall, about 1-1/2 in off the kerb, then let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/grab-bars-ada-transfer-side/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This floor plan of an accessible toilet room is drawn at 1 in = 4 mm. The grey kerbs are the rear wall behind the toilet and the side wall beside it; the white shapes are the toilet's bowl and tank. Fit the two grab bars (40 in and 48 in long), one on each wall, where the 2010 ADA Standards require them for this toilet. Lay each bar flat on the floor along its wall, about 1-1/2 in off the kerb, then let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/grab-bars-ada-transfer-side/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`score`, `tier_reason`, `side_wall` and `rear_wall` with `has`
and `ok`, `all_still_1s`, and for each bar `centre_mm`, `heading_deg`, `flat_err_deg`, `low_mm`, `ends_mm`, `wall`,
`gap_mm`, `compliant`, `why_not`, the code readings (`rear_end_from_rear_wall_mm`, `front_end_from_rear_wall_mm` on
the side wall; `wall_side_past_cl_mm`, `open_side_past_cl_mm` on the rear wall), `settled`, `released`, `counted`,
`still_1s`, `path_mm`, `turn_deg`, `hands_off_now`, `zone_gap_mm`; and the right TCP, both pad-tip points and both
openings). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it
over at every reset (rigstate's reset counter, and a teleport of both bars back to their starting marks), so run one
trial per process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads both bars' poses and both grippers once per
control step (20 Hz). From each bar's pose (centre, orientation; its length is its local x axis) it works out its
heading seen from above, how far its nearest long face is from facing down, its lowest corner and its two end points
(its centre plus and minus half its length along its axis).

A bar is along the side wall when it lies flat on the floor (a long face within 5 degrees of facing down, its lowest
corner within 3 mm of the table top), its length is within 5 degrees of x, its wall-side face is 0 to 15 mm off the
side kerb's face (y -31.5 cm) at its midpoint (a bar pressed against the kerb, up to 1 mm in, counts; 609.3's
1-1/2 in is 6 mm) and both its ends are beside the kerb (x 20.0 to 46.0 cm). Along the rear wall is the same with
y, the rear kerb's face (x 45.5 cm) and ends between y -32.0 and +1.0 cm. A bar lying on a kerb or on the other bar is
not on the floor; a bar parallel to one wall cannot be along the other.

A bar counts only once it is let go and has settled: over the last 0.5 s both grippers' finger zones stayed at least
10 mm from it at every control step, its centre moved under 2 mm in all, and its heading changed under 1 degree
(measured on positions: the simulator's reported velocities chatter while a part is gripped). rigstate gives no
gripper heading, so a gripper's fingers are taken to be anywhere round its axis: its zone is everything within reach
of a pad's outer corner (half the current finger gap plus the 8 mm pad thickness out, 13.5 mm across; 57 mm when fully
open) around the 38 mm stretch back from the point between its pad tips. Still for 1 s means settled in each of the
last two 0.5 s windows.

Each wall is judged by the code, whichever bar is on it (the scorer knows each bar's length from which bar it is):

| Wall | Compliant when a counted bar along it | In table coordinates |
|---|---|---|
| Side (604.5.1) | is at least 42 in (168 mm) long, its rear end at most 12 in (48 mm) from the rear wall's face, its front end at least 54 in (216 mm) from it | rear end x >= 40.7 cm, front end x <= 23.9 cm; for the 192 mm bar, centre x 31.1 to 33.5 cm |
| Rear (604.5.2) | is at least 36 in (144 mm) long, its side-wall end at least 12 in (48 mm) past the centreline, its other end at least 24 in (96 mm) past it toward the open side | ends at y <= -29.5 cm and y >= -15.1 cm; for the 160 mm bar, centre y -23.1 to -21.5 cm |

| Outcome (first match wins) | Score |
|---|---|
| Both walls compliant and both bars still for 1 s (success once held 1 s more) | 1.0 |
| One wall compliant and a counted bar along the other (not compliant, or the 1.0 layout not yet still for 1 s) | 0.75 |
| One wall compliant, nothing counted along the other | 0.5 |
| A counted bar along a wall, neither wall compliant | 0.25 |
| No counted bar along either wall (nothing moved; bars held, moving, tilted, stacked, crosswise, on a kerb, over 15 mm off a kerb or past its end) | 0 |

Every tier is reachable and each final state matches one line: the demo layout 1.0; a compliant side bar with the rear
bar centred on the toilet (stopped by the side kerb, its open end reaches only 23 in past the centreline) 0.75; the
bars swapped 0.75 (the 48 in bar really does meet the rear-wall rule; the 40 in bar is under 42 in); a compliant side
bar with the 40 in bar left where it started 0.5; both bars along their walls, the side bar's rear end more than 12 in
from the rear wall and the rear bar centred, 0.25; nothing moved 0. The tests also try the shortcuts and exploits we
could think of: the side bar pushed back beside the bowl (front end only 48 in out: 0.75 with a compliant rear bar,
0.25 without); both bars along the side wall (0.5); a bar laid on top of another or propped on a kerb (does not
count); the bars held in place, or pressed by an open gripper (do not count); both bars crosswise in the corner (0).
No latch is needed: sliding a bar along its wall is a correction, and nothing here is a violation a later state could
hide. A new trial is recognised at rigstate's reset or when the simulator puts both bars back on their starting marks
(within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number or a
zero quaternion scores 0 for that step and leaves the scorer's record as it was.

## One arm alone

This task needs one arm, and the demo uses one: the right arm, whose base is beside both walls, fits the 48 in bar to
the side wall and then the 40 in bar to the rear wall, while the left arm holds its ready pose. `DEMO_IDLE_ARM=left`
is therefore the same run as the demo (the offline kinematic check predicts 1.0); `DEMO_IDLE_ARM=right` leaves
nothing to move (the bars stay where they start: 0). Gripping straight down, the left arm reaches neither wall where
a bar must lie: the side wall is about 0.6 m to the side of its base, and along the rear wall (x 43 to 45 cm) our
inverse kinematics of the arm's URDF (joint limits, solved from the ready pose; tool point 21 to 30 mm up, headings
every 10 degrees) finds no straight-down grip at y +1.5 cm or lower, while a bar along the rear kerb lies wholly at
y +1 cm or lower. A policy might still push a bar there with a tilted gripper, so no claim is made
that the left arm alone cannot score. No claim is made that the task needs both arms: its difficulty is reading the
code's asymmetric rule from the plan, choosing the right bar for each wall and setting the rear bar off centre.

## Demo and run limit

The scripted demo takes 34.5 s (690 control steps). The right arm comes down over the 48 in bar's middle (pad tips
8 mm above the floor), grips it across its 20 mm width at yaw +90 degrees (fingers commanded 8 mm narrower, over
1.5 s), lifts it to 10 cm (tool point), carries it beside the side wall in 2.5 s while turning it a quarter turn to
lie along x, lowers it over 2.5 s (the last 30 mm slowly) until its underside is 2 mm above the floor, waits 0.5 s,
opens over 1.5 s and goes straight up. It then does the same with the 40 in bar, gripped at yaw 0 and turned a
quarter turn to lie along y beside the rear wall, and moves back by its base. Open, the fingers are 34 mm apart (7 mm
either side of a bar); the wall-side finger passes 6 mm over the 4 mm kerb. Aims: the side bar's centre at (32.1,
-29.7) cm, its wall-side face 8 mm off the kerb (window 0 to 15 mm), its rear end 38 mm from the rear wall (window 24
to 48 mm) and its front end 230 mm out (54 in is 216 mm); the rear bar's centre at (43.7, -22.3) cm, 8 mm off the
kerb, its side-wall end 12 mm off the side wall (window 4 to 20 mm) and its open end 104 mm past the centreline (24 in
is 96 mm). The windows are 15 to 24 mm wide. The wall gaps are aimed 2 mm wider than 609.3's 1-1/2 in (6 mm) so that
the trimmed place points below, which move each bar toward its kerb, still keep every corner of a bar at least 2.5 mm
clear of the kerb's face.

Two corrections from our earlier simulator runs are built in. Parts carried in a gripper ended turned counter-clockwise
of the commanded heading by about 0.6 to 1.7 degrees, so each place heading is set 1.15 degrees clockwise (`YAW_BIAS`
in `task.py`); a 1.7 degree error moves a bar's end by under 3 mm. Parts also landed about 4 mm short of the place
point along the carry, so each place point is moved 4 mm further along its carry (`PLACE_TRIM`); offline, with no
shortfall, the bars land on the trimmed points, which are inside the windows too. The demo keeps at least 6 mm between
the side bar's rear end and the rear bar's front face whichever way each lands, so the two never meet at the corner.
Offline the kinematic check plays the demo through the arm's inverse kinematics (worst error 0.08 mm, joint 4 margin
14 degrees, joint 6 margin 41 degrees), drops each bar 2 mm (the rear bar onto the 0.5 mm painted tank), and success
holds for its 1 s at step 681 (34.05 s), with the hand well clear. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the 48 in bar back flat on the floor centred at
x 24.0 cm, y -12.0 cm with its length along y, and the 40 in bar centred at x 35.0 cm, y -6.0 cm with its length
along x; either long face may be down and either end either way. The kerbs and the painted toilet are fixed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 35 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two strips of wood 10 mm wide and 4 mm tall, painted grey, glued to the bench at a right angle as the kerbs (330 mm
for the rear wall, 250 mm for the side wall, butting against the rear strip's face); the toilet as two shapes cut from
white vinyl sheet (a 40 x 72 mm rectangle against the rear strip, its centre line 68 mm from the side strip's face, and
a 72 mm disc touching its front edge); and two lengths of 20 x 20 mm aluminium square tube with a 1.5 mm wall and
capped ends, 192 mm (58 g) and 160 mm (48 g) long. Nothing moves but the bars, nothing is powered, and the bars are
light and blunt, so trials are safe. Score with a tape from each kerb's face to the bars' faces and ends, and a photo
from above showing both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
