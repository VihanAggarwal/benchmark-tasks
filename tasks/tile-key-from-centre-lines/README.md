# Set out each room's first floor tile so no border cut is under half a tile

Tile and stone setters (O*NET 47-2044.00) set a floor out from the room's centre lines: two chalk lines through the
middle of the room, then a dry run of tiles out to the walls. If the run would leave thin slivers at the walls, they
move the start by half a tile, putting a joint on the line instead of a tile's centre (or the other way round), so
that the cut tiles at opposite walls are equal and at least half a tile wide. Slivers look bad, break when cut and
come loose. Each direction is decided separately from the room's size across that line, so a room that needs a
joint on one line can need a tile centred on the other. Here a 1:5 model of two rooms stands on the table, one in
front of each arm: white walls 25 mm tall, two blue chalk lines through the middle of each room, and one loose
60 mm terracotta tile in each room (300 mm tiles with 10 mm joints at 1:5: a 62 mm module). The left room is
278 mm front to back and 340 mm side to side; the right room is turned the other way, 340 by 278 mm. In each room
the tile must be laid flat at the crossing of the lines, square to the walls, and for each line either centred on it
or with a joint on it, whichever leaves both border cuts across that line at least 30 mm. The two rooms need
opposite answers. The score is what a setter checks with a ruler and a photo; each room scores 0.5 only when both of
its directions are right. The demo uses both arms, each laying the tile in front of it (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the set-out arithmetic |
| `scoring.py` | The scorer: plain functions of the two tiles' poses, with a small tracker of their last half second |
| `tests/test_scoring.py` | Offline tests of the cut arithmetic, every tier, every zero condition, the exploits we tried, resets, NaN states, the hooks and the demo plan (no simulator needed) |

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
  --instruction "Each room has its two centre lines chalked in blue. In each room, lay the room's loose tile flat at the crossing of the centre lines as the first tile of the floor, square to the walls, and for each line either centred on it or with a joint on it, whichever leaves the cut tiles at both walls across that line at least half a tile wide. Tiles are 60 mm with 2 mm joints. The left room is 278 mm front to back and 340 mm wide; the right room is 340 mm front to back and 278 mm wide." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tile-key-from-centre-lines/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Each room has its two centre lines chalked in blue. In each room, lay the room's loose tile flat at the crossing of the centre lines as the first tile of the floor, square to the walls, and for each line either centred on it or with a joint on it, whichever leaves the cut tiles at both walls across that line at least half a tile wide. Tiles are 60 mm with 2 mm joints. The left room is 278 mm front to back and 340 mm wide; the right room is 340 mm front to back and 278 mm wide." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tile-key-from-centre-lines/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (per room: `tiles_at_crossing`, `tile`,
`off_front_to_back_line_mm`, `off_side_to_side_line_mm`, both pairs of cuts in mm, `front_to_back_good`,
`side_to_side_good`, `tilt_deg`, `square_err_deg`, `lowest_mm`, `still`, `zero_reason`; then `rooms_right`,
`moved_mm`, `first_step`, `bad_steps` and each tile's pose and speed). Videos: `inspect-robots video <log>.json`. The
scorer keeps one trial's record per process and starts it over when the simulator puts both tiles back on their
starting marks, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py` and the arithmetic is in `layout.py`'s docstring. Across a
centre line with D of room across it, a tile whose centre is u from the line has a grid of 60 mm tiles and 2 mm
joints that, extended to the two walls parallel to the line, leaves border cuts of (D/2 - u - 32 mm) and
(D/2 + u - 32 mm), each reduced modulo 62 mm: the gap from the last whole tile to the wall, less one joint. A
direction is right when both are at least 30 mm (half a tile). The two cuts always add up to either 28 mm or 90 mm,
so both reach 30 mm only when they add up to 90. That gives two windows:

| Room across the line | Direction right when | Centred (u = 0) | Joint on the line (u = 31 mm) |
|---|---|---|---|
| 278 mm | the tile's centre is within 15 mm of the line | cuts 45 / 45 mm: right | cuts 14 / 14 mm: slivers |
| 340 mm | the tile's centre is 16 mm or more from the line (either side) | cuts 14 / 14 mm: slivers | cuts 45 / 45 mm: right |

So the left room (340 mm across its front-to-back line, 278 mm across its side-to-side line) needs a joint on its
front-to-back line and a tile centred on its side-to-side line: the tile's centre at x 28.9 cm, y +27.1 or +20.9 cm.
The right room needs the opposite: centre at y -20.9 cm, x 25.9 or 32.1 cm. Around the demo's aims each direction
has 15 mm of window either way, except 14 mm outward from a joint line (the 45 mm crossing limit below). Offsets are
measured to a chalk line's centre.

A tile is at a room's crossing when its centre is within 45 mm of both of the room's lines (that limit also keeps
out the same grid one module further out, whose cuts are the same). It is laid when it is flat (its 60 x 60 mm
faces within 5 degrees of level, either face up), square (edges within 5 degrees of the walls), on the floor (its
lowest point within 2 mm of the table top) and still (none of the four corners of a face moved more than 2 mm
in the last 0.5 s; a gripped tile's reported velocity chatters, so the path is used).

| Outcome for one room | That room's score |
|---|---|
| Exactly one tile at its crossing, laid, both directions right | 0.5 |
| No tile at its crossing (left where it started; centre more than 45 mm from either line) | 0 |
| Both tiles at its crossing | 0 |
| The tile there tilted more than 5 degrees or on an edge, turned more than 5 degrees (e.g. laid diagonally), not within 2 mm of the floor (held up, or fallen off the table), or still moving | 0 |
| The tile there laid, but set out wrong in either direction | 0 |

The score is the sum of the two rooms: 0, 0.5 or 1.0. Either tile may be used in either room. The score is read at
the last step; the trial ends early, at 1.0, once 1.0 has held for 2 s (a 1.0 state at the run limit counts even if
it has held less). A new trial is recognised when the simulator puts both tiles back on their starting marks (within
4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number scores 0
for that step and leaves the record of the trial as it was.

Why no credit per direction: a room set out wrong in either direction gets slivers along two walls and is re-laid,
and per-direction credit would pay 0.5 for centring both tiles on their crossings without reading either room.
Unintended strategies earn nothing more than they should, as the tests show: both tiles centred on the crossings 0;
a joint on every line 0; the left room's answer copied into the right room 0.5 (only the left room right); both
tiles at one crossing, one each way (a hedge), 0; a tile laid on the right grid one module further out 0; a tile
right but turned 10 degrees 0 for that room; a tile stacked on the other at a crossing 0; a tile held flat 1 mm
above the right spot counts as laid there (the scorer reads only the tiles; the instruction asks for the tile to be
laid, and the run ends after 2 s of it).

## One arm alone

The scripted one-arm runs (`DEMO_IDLE_ARM=left`, `DEMO_IDLE_ARM=right`) lay only the working arm's own room and score
0.5 each in simulation (left arm idle 0.5, right arm idle 0.5; see "Results in simulation"), as the offline
kinematic check predicted. The limit
comes from where the rooms are, not from the act, which one hand can do; the task makes no two-arm claim. Straight
down (the scripted grips), neither arm reaches the other room's tile or spots: the left arm reaches no further right
than y -17 cm at x 20 cm (the right room's tile starts at y -27.5 cm, its nearest edge at -23.5 cm), and only -15 to
-9 cm over x 24.5 to 33.5 cm, where the right room's good spots need the centre at y -19.4 to -22.4 cm; the right
arm is its mirror image (the left room's tile starts at y +33.5 cm, and its good spots need the centre at y +19.5 cm
or more, at x 27.4 to 30.4 cm, where the right arm reaches +13 cm at most). These figures come from inverse
kinematics of the arm's URDF with joint limits and a collision proxy of its own base and the table, over 16 grip
headings. Leaned 20 to 40 degrees from vertical, though, our kinematic model has the pad tips of either arm reaching
the other room's tile and its good spots, so one arm with leaned grips could in principle lay both tiles and score
1.0.

## Demo and run limit

The scripted demo takes 17.3 s (346 control steps), both arms at once, each in its own room (the tool points stay at
least 48 cm apart). Each arm comes down over its tile with the fingers open to 79 mm and turned with the tile (left
+20 degrees, right -25 degrees), the last 30 mm slowly, to the tool point 16 mm above the table (pad tips 3 mm up,
on the tile's edges 3 to 10 mm up); grips it across two opposite edges (fingers commanded 52 mm, over 1.5 s); lifts
it until the tool point is 85 mm up (the tile's underside 69 mm up, 44 mm over the walls); carries it over its spot
turning it square (yaw 0); comes down, the last 30 mm over 1.5 s, to the tile's underside 2 mm above the floor; lets
go over 1.5 s; and goes straight up, then back to 15 cm above the table at x 20 cm, y +-40 cm (over its room's near
outer corner), clear of the laid tile, so the top camera sees both tiles. Left room: centre aimed at x 28.9, y +27.1
cm (centred on the side-to-side line, a joint on the front-to-back line). Right room: x 25.9, y -20.9 cm (a joint on
the side-to-side line, centred on the front-to-back line). Open, the pads stay at least 24 mm from every wall. Laid
across a chalk line, a tile rests on it (0.5 mm proud, 4 mm wide) and sits about 1 degree off level; a tile with a
joint on a line overlaps that line's edge by 1 mm. Because a tile held 2 mm above the floor already counts as on it,
success must hold 2 s (not the usual 1 s), so a trial ends after the fingers are fully open. Offline, success has
held for its 2 s at step 305; the scripted policy stops at step 407. The run limit is 1,200 steps (60 s).

To retune on the GPU, read each room's `off_front_to_back_line_mm` and `off_side_to_side_line_mm` in `TASK_TRACE`
(aims: left room +31 and 0, right room 0 and -31) and move `PLACE_TRIM` in `task.py` by minus the offset.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the left room's tile flat at x 20.5 cm,
y +33.5 cm, turned 20 degrees anticlockwise from the walls (seen from above); the right room's tile flat at
x 19.5 cm, y -27.5 cm, turned 25 degrees clockwise. The walls and chalk lines are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 17 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two rectangles of 10 x 25 mm wooden strip, painted white, fixed to the bench top with the inside sizes and positions
in `task.json` (278 x 340 mm and 340 x 278 mm, inner walls 12 cm apart); a blue plastic strip 4 mm wide and 0.5 mm
thick (or 4 mm blue tape) glued wall to wall through the middle of each room both ways; and two 60 x 60 mm squares
cut from a 10 mm unglazed terracotta quarry tile (about 80 g each). Score with a steel rule: in each room, the
distance from each chalk line (its centre) to the tile's two edges either side of it gives the centre's offset u;
then read the table above (within 15 mm of a 278 mm line, 16 to 45 mm from a 340 mm line), and check squareness and
flatness on a top photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
