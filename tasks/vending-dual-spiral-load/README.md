# Make two vending spirals one wide selection and load it from the front pitch

When a planogram brings in a product too wide for one spiral, the vending technician makes a double-wide
(dual-spiral) selection: they lift out the plastic divider between two neighbouring spirals and load the wide
product across both coils, one product per pitch, from the front. The front pitch must be filled so that the first
vend drops a product, and a product left resting on a coil loop or on the divider jams the selection (the customer
pays and nothing drops, or two products drop). This is work done by coin, vending and amusement machine servicers and
repairers (O*NET 49-2091.00).

Here a spiral tray with two neighbouring spirals is fixed in front of the left arm. Each coil is modelled where its
loops meet the shelf: five black 12 mm steel ribs per spiral, 62 mm apart and lined up across both spirals, which
leaves four 54 mm pitches. A white divider stands in the 24 mm slot between the spirals; a grey parts mat lies on the
midline; four wide product cartons (140 x 30 x 70 mm) stand on the table to the left of the tray. The robot must take
the divider out onto the mat and then stand one carton in each pitch, across both spirals, on the shelf floor
between the ribs. The score is a front-first run of seated pitches plus the divider on the mat, and any carton left
riding a rib, the divider, a wall or another carton zeroes it. This is a one-arm task (see "One arm alone").

Nearest tasks on the board, and why this one differs. No board task mentions vending, spiral merchandisers, coils or
pitches (keyword search of the 668 board tasks on 2026-11-06; the two that mention planograms, TK-RNWKBY and TK-CVSB8W, line up
cans on a shop shelf). The closest in kind are TK-RMZJR4 (stand four drug cartons upright in a rack's pockets, one per
pocket, look-alikes apart) and TK-BTYFXV (shelve three books upright in a row from an end wall, then a bookend): both
place items into fixed pockets or along a flat shelf, with nothing to remove first. Here the divider must come out
before a 140 mm product can stand across two lanes, each box must go down between ribs 12 mm tall with only 12 mm of
play either side, and the score is a front-first run of seated pitches, with a box left riding a rib counted as a jam.
TK-DA3R5M (change over a conveyor's guide rails for a new carton) slides two rails to a measured gap and places one
test carton, with no fill order and nothing that can jam. TK-7AHCHJ (lift two floor tiles to a pad,
stand cones in the holes) is remove-then-place but scored per hole, with no fill order, no run and no riding failure.
TK-WXD2T8 (DIMMs, one per channel, first slots first) is a slot-choice rule with nothing to remove and no riding.
TK-RNF7BR (four blanks into a gravity magazine) is a vertical stack in a channel; TK-LR2YJR (blades and blanking
panels to a work order) a slot assignment by work order; TK-JRTJVD (pull a file, stand an out-card in its place)
swaps one item for another in a shelf row; TK-35JE3X (clear a line before setting out the next reel) a latched
precedence cap; TK-F545PQ (fill seedling-tray gaps) fills cells in no order with no changeover. TK-384FU2 (insert
a carton divider, pilot) holds a carton and inserts a divider into its slots to a set depth; here the divider is taken
out and the score is the loaded state. LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench and RoboDojo have no
vending or spiral-loading task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the divider's and the boxes' states and the grippers, with a small tracker for stillness and resets |
| `rigstate.py` | Reads both grippers (tool point, pad-tip point, opening) for the scorer each control step |
| `trace_trim.py` | Reads one GPU demo run's `TASK_TRACE` file and prints where each box stood in its pitch, how it tilted, and the `PLACE_TRIM` to set (a tuning aid; no simulator needed) |
| `tests/test_scoring.py` | Offline tests of every score, every zero condition and its boundary, held and moving objects, the exploits we tried, bad states, resets, the task file's hooks, the trace reader and a kinematic replay of the demo (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed and loose boxes) and the standard `task_score` scorer. `rigstate.py` sits next to `task.py`
and wraps the embodiment's step and reset from the task file's import so the scorer can read both grippers; it
changes nothing the embodiment does.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm; the right arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Make the two spirals one wide selection: lift the divider out from between them and set it on the grey mat. Then load the four wide boxes into the selection: one box in each of the four pitches, standing upright on the shelf across both spirals, between the coil loops and not on them. The front pitch (nearest you) must hold a box, and no pitch may be left empty in front of a filled one." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/vending-dual-spiral-load/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Make the two spirals one wide selection: lift the divider out from between them and set it on the grey mat. Then load the four wide boxes into the selection: one box in each of the four pitches, standing upright on the shelf across both spirals, between the coil loops and not on them. The front pitch (nearest you) must hold a box, and no pitch may be left empty in front of a filled one." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/vending-dual-spiral-load/dream
```

Run one trial per process (the default). The scorer starts a new trial when the embodiment resets (through
`rigstate`'s reset hook or, failing that, when every object jumps back onto its starting mark in one step).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided: `score`, `n`, `seated_pitches`, `divider_on_mat`, `zeros`, and for the
divider and each box `inside`, `held`, `at_rest`, `path_mm`, `low_mm`, `tilt_deg`, `pos` (boxes also `pitch`, `jam`,
`yaw_off_deg`, `lane_off_mm`, `x_pitch`), the grippers' `opening`, tool points and pad tips, `arms_unavailable` and
`bad_steps`. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the end state at the last control step
(`SCORE_OVER = "end"`), and the trial ends early once 1.0 has held for 1 s (`SUCCESS_HOLD_S = 1`). Every control step
(20 Hz) the scorer reads the divider's and the four boxes' poses and both grippers. There are no whole-trial rules:
the scorer keeps only each object's last 0.5 s of positions (for "at rest") and counts unreadable steps.

Terms. **At rest**: the object's centre has moved less than 2 mm (path length) over the last 0.5 s (10 control
steps); this is a path of positions, never a velocity, because a gripped or settling object's reported velocity
chatters at 10-45 mm/s while it stays put. **Held**: a gripper less than half open (pads under 47 mm apart) has its
pad-tip point inside the object's outline grown by 15 mm on every side; free means not held. **Tilt**: the angle of a
box's 70 mm axis from vertical, either way up. **Inside the selection**: some part of the object's outline seen from
above (the shadow of its real, turned and tilted box) lies more than 1 mm inside x 16.0 to 42.0 cm, y +5.0 to +23.0 cm,
the shelf between the walls' inner faces, at any height (the 1 mm keeps a box standing on the table against the
tray's front or back face outside). **Seated in pitch k**: a box free and at rest, its lowest point 19 to 23 mm above the
table top (the shelf floor is at 20 mm; the 1 mm below it allows for a resting contact that reads slightly into its
support; a box riding a 12 mm rib is 32 mm up), tilted at most 8 degrees, its centre inside pitch k's
54 mm along x and within 20 mm of the selection's centreline (y +14.0), and its 140 mm side within 10 degrees of the
y axis. Two boxes cannot share a pitch: 30 + 30 mm is more than 54 mm. **n**: the pitches seated in an unbroken run
from pitch 1 (pitch 1, then 2, 3, 4); n is 0 if pitch 1 is not seated. **Divider on the mat**: free and at rest, its
centre over the mat's top (x 15.5 to 41.5, y -5.0 to +1.0 cm) and its lowest point 4 to 10 mm above the table top
(the mat's top is at 5 mm: a divider tipped off the mat's edge onto the bare table is not on it), lying or standing.

Zero rules (score 0 whatever else holds): **(Z1) jammed product**: any box that is free, at rest and inside the
selection but not seated (riding a rib, the divider, a wall or another box; tilted or leaning; lying on a side;
standing on end; not turned across the lane); **(Z2) divider not out**: any part of the divider inside the
selection, held or not, at any height. Otherwise the score is 0.2 if the divider is on the mat, plus 0.2 x n.

| Outcome | Score |
|---|---|
| Starting layout (divider in its slot) | 0 (Z2) |
| Any part of the divider still over the selection: in the slot, pulled partway out, held up over the tray, lying across the walls, or on the mat but turned so it reaches the shelf | 0 (Z2) |
| Any box free and at rest inside the selection but not seated: on a rib, on another box, on the divider or a wall, tilted over 8 degrees, lying down, on end, not turned across the lane, two in one pitch | 0 (Z1) |
| Divider on the mat, no box seated | 0.2 |
| Divider on the mat, pitch 1 seated | 0.4 |
| Divider on the mat, pitches 1-2 seated | 0.6 |
| Divider on the mat, pitches 1-3 seated | 0.8 |
| Divider on the mat, all four pitches seated (the trial ends after 1 s of this) | 1.0 |
| All four seated, the divider dropped on the table off the mat (and off the selection) | 0.8 |
| Pitches 2-4 seated, pitch 1 empty | 0.2 with the divider on the mat, otherwise 0 |
| Pitches 1, 2 and 4 seated (3 empty), divider on the mat | 0.6 |
| A box still held, or still moving, at the end | neither counts nor zeroes |
| The divider still held, or still moving, at the end | not on the mat (and Z2 still applies over the selection) |

Possible scores: 0, 0.2, 0.4, 0.6, 0.8, 1.0. Every outcome maps to exactly one of them.

Unintended strategies earn nothing, as the tests show: laying a box on top of a seated box (70 mm above the floor) or
across a wall top is a jam (Z1 has no height band); the divider pushed partway out of the front of the slot, or held
in the air over the tray while the boxes go in, still has part of its outline over the selection (Z2, at any height);
the divider dropped anywhere but the mat earns no 0.2, so 1.0 is out of reach; a box or the divider still held at the
end does not count (seated and on-mat both need free), and parking a closed gripper on a box riding a rib saves the
trial from Z1 only to what leaving that box off the tray gives; boxes not turned (140 mm along x) ride the ribs; boxes
lying on their 140 x 70 face (70 mm along x, more than a pitch) or standing on end are tilted or riding; two boxes in
one pitch cannot both be seated; the 4 mm of floor behind the last rib and the flush front rib leave nowhere else to
stand a box; filling pitches 2-4 and leaving pitch 1 empty adds nothing, as the first vend would drop nothing.
Pushing a box into a pitch is not possible: the boxes start on the table and the shelf is a 20 mm step up, and
tipping one in leaves it lying down (0). A box dropped from height that lands seated is a real loaded product and
counts. The order of loading does not matter to the score: the instruction states an end state (the front pitch
holds a box and no pitch is left empty in front of a filled one), and the score reads only that. Grippers that
cannot be read on a step hold nothing (counted in `arms_unavailable`); a state with a non-finite number or a zero
quaternion scores 0 for that step and leaves the tracker unchanged.

## One arm alone

This is a one-arm task, and the demo uses one arm: the left arm takes the divider out to the mat and loads all four
boxes while the right arm holds its ready pose. `DEMO_IDLE_ARM=right` is therefore the same run as the demo: it
scored 1.0 on the GPU (as offline), with all four pitches seated and the divider on the mat. `DEMO_IDLE_ARM=left`
leaves nothing to move: it scored 0.0 on the GPU (as offline: the divider stays in its slot, Z2). So the left arm
alone reaches a full score. Everything except the mat is in front of the left arm. From our offline reach check (not
run on the GPU): straight down (the scripted grip), the right arm reaches only the divider's front end (x 17 to 22 cm),
the front pitch and the mat, and none of the boxes (their nearest ends are 57.5 cm to the left of its base), so with a
straight-down grip the right arm alone scores at most 0.2 (the divider on the mat). Tilted about 55 to 85 degrees from
vertical, its tool point can reach the near ends of the boxes (offline IK), so we do not claim the right arm is
strictly unable to load one. No claim is made that the task needs both arms: its
difficulty is the changeover and the placement (keeping each carton square across the lane and upright, setting it
down between ribs 62 mm apart without riding them, and filling from the front).

## Demo and run limit

The scripted demo (left arm only) takes 89.2 s (1,784 control steps); success has held for its 1 s at step 1,768,
just before the plan ends, and the trial ends there (offline and on the GPU run alike).

1. Divider (15.2 s, yaw 0, the fingers closing along y across its 20 mm), gripped at x 25.7 cm, the middle of
   pitch 2 and 28 mm nearer the arms than its centre, so that no rib lies under the pads (their 20 mm along x,
   x 24.7 to 26.7 cm, is 17 mm from the rib faces either side; the off-centre torque is 0.05 N m): above the grip
   point at 12.0 cm, fully open; down to 2 cm above the grip, then the last 2 cm over 1.5 s to a tool-point height
   of 5.8 cm (pad tips 4.5 cm up: 15 mm below the divider's top and 25 mm above the pitch's floor); settle 0.6 s;
   close to a 12 mm command (20 - 8 mm) over 1.5 s; lift straight up to 12.0 cm (its bottom 8.2 cm up, clear of the
   5 cm walls); carry 2 s along y over the mat (the tool point stays at x 25.7, so the divider's centre comes over
   the mat's centre); down to 2 cm above, the last 2 cm over 1 s to 4.6 cm (its bottom 3 mm above the mat); settle;
   open over 1.5 s; up.
2. Boxes, front pitch first (18.0 s each, 18.5 s for the first; yaw -90 degrees, the fingers closing along x across a
   box's 30 mm). Supply x to pitch x: 18.0 to 19.5, 26.0 to 25.7, 34.0 to 31.9, 42.0 to 38.1 cm. For each box: to
   above it at 12.8 cm with the fingers going to a 50 mm gap on the way (10 mm clear of the box each side, 30 mm
   from the next box in the line); settle 0.5 s; down to 9.0 cm, then over 1.5 s to 3.8 cm (pad tips 25 mm above
   the box's bottom: a mid-height grip); settle 0.6 s; close to a 22 mm command (30 - 8 mm) over 1.5 s; lift to
   12.8 cm (the box's bottom 9.0 cm up: above the 7 cm supply boxes and the 5 cm walls, and level with the tops of
   placed boxes, which it never passes over); along x to the pitch's x in 1 s, then along y over the pitch in 2.5 s; settle 0.6 s; down to 8.6 cm (its bottom 16 mm above the rib tops), then the last 25 mm over 1 s
   to 6.1 cm (its bottom 3 mm above the shelf floor); settle 0.6 s; open to a 46 mm gap over 1.5 s (8 mm each side;
   the fingers' outer faces stay 14 mm from a centred neighbour box); straight up to 12.8 cm (pad tips 11.5 cm up,
   above the 9 cm box tops). After the last box the fingers open fully.
3. Offline (our kinematic check): IK error under 0.1 mm everywhere; the smallest joint margin is 0.20 rad (joint4,
   lifting the last box at x 42 cm), joint6 0.33 rad (gripping the first box); nothing in the arm's way (no pad, link or carried-object
   interference), pad tips never lower than 25 mm above the table.

On the GPU run the demo needed no trim (`PLACE_TRIM` 0.0, `RELEASE_S` 1.5 s, `RELEASE_DROP` 3 mm): each box ended
seated, upright (tilt 0 degrees), its bottom on the shelf floor and its centre within 2 mm of its pitch's centre
along x and 1 mm of the centreline; the divider ended on the mat, its bottom 5 mm up. Any box left riding a rib zeroes
the trial, so if a different build places boxes off-centre or tips them, tune the place point this way:

1. Run the demo once with `TASK_TRACE` set and read the trace with `python3 trace_trim.py <trace.jsonl>`. For each
   box it prints the grip offset along x while carried, where the box stood in its pitch just before the pads
   opened, its largest tilt before the lift (the pads closing on a 30 x 70 mm footprint, which tips at about 23
   degrees) and after the release, and how it ended; then the `PLACE_TRIM` that cancels the median place error
   (pads have been seen about 8 mm off the commanded point in other tasks; a pitch leaves 12 mm of play either side
   of a centred box). Set `PLACE_TRIM` in `task.py`, then run the scored demo.
2. If a box tips as the pads open, open more slowly (`RELEASE_S` 2.0 instead of 1.5) or let go lower
   (`RELEASE_DROP` 1 to 2 mm instead of 3; a box that does not drop at all then already has its bottom 21 to 22 mm
   above the table, inside the 19 to 23 mm seated band).
3. That the divider lifts out of the 24 mm slot without catching a rib (2 mm either side) and stands on the mat
   (the trace's `divider.low_mm` about 5).

The run limit is 2,400 steps (2 minutes at 20 Hz).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the divider back in the slot, centred at
x 28.5, y +14.0 cm; stand the four boxes back on the table at y +34.0 and x 18.0, 26.0, 34.0 and 42.0 cm, 140 mm
side along y. The tray, walls, ribs and mat are fixed. Nothing is consumed.

## Runtime per trial

Measured on an L40S, one trial per process including Isaac Sim's start (about 15 s): about 4 minutes for the
scripted demo (it ends after about 88 s of simulated time) and about 5.5 minutes for the Dream-1 run to the 2,400-step
limit (120 s of simulated time). The first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 260 x 200 x 20 mm steel or aluminium plate (or 20 mm plywood painted light grey) clamped to the bench as the
tray, two 260 x 10 x 30 mm bars screwed along its sides 180 mm apart inside, and ten 8 x 78 x 12 mm black steel (or
painted hardwood) strips glued across the floor as the coil ribs: five per side at 62 mm centres, the first flush with
the front edge, leaving a 24 mm slot down the middle. The divider is a 230 x 20 x 40 mm white HDPE bar (175 g); the
mat a 260 x 60 x 5 mm grey rubber sheet. The products are four rigid cartons 140 x 30 x 70 mm (stiffened with a card
insert) of about 90 g. A real spiral machine's coils, motors and tray latches are not needed for the loading
skill; the ribs stand where a coil's loops cross the shelf.

Scoring a real trial: from a top photo and a ruler read each box's position in its pitch and its yaw, which pitches
are filled and whether the divider rests on the mat; slide a 3 mm shim under each box's edges to check it stands on
the shelf floor (a box on a 12 mm rib is obvious); check tilt with a small level or a side photo; read the gripper
openings at the end from the logged joint states.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |
