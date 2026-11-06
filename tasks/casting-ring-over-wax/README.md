# Seat the casting ring that leaves 5-8 mm over a wax crown, never touching the wax

In a dental laboratory, metal crowns are cast by the lost-wax method. The technician stands the sprued wax pattern
on a rubber crucible former and sets a casting ring over it, chosen so that about 6 mm of investment will cover the
pattern at the ring's open end: much less and the mould end cracks or the metal breaks through, much more and the
mould's gases cannot escape (back-pressure porosity). The ring goes straight down without touching the pattern,
because inlay wax dents under the lightest touch, and is seated on the former so the investment cannot leak.

Here a black rubber former (an 88 x 88 x 10 mm block with a 36 mm centring step, 6 mm tall) is fixed on the table
in front of the left arm, with a red wax sprue and a blue wax crown on the step; the crown's top is 36 mm above the
table. Three grey steel casting rings (square tubes, 60 x 60 mm outside, 4 mm walls) stand on the table further left,
24, 32 and 40 mm tall and otherwise alike. Stood on the former's base, only the 32 mm ring's top is 5 to 8 mm above
the crown's top (6 mm; the others give -2 mm and +14 mm). The core of the task is the contact-free placement: a
sleeve lowered straight down round a fragile part it then hides, with the scorer judging the ring's real walls and
the gripper's fingers against the wax at every step. Choosing the ring by the cover it will leave comes first.

How it differs from tasks already on the board: picking the pin that fits a hole (TK-TPC59R) chooses a part by its
fit and pushes it in; bracing a crate with the dunnage block that fits the gap (TK-HVH57C) chooses a block by its
thickness and drops it in; standing sample tubes in a water bath with their tops in a height band (TK-7ZRDUM)
chooses each tube's support so its top ends in the band. In all three the chosen part goes into an open space. Here
the chosen part is a sleeve that must pass round another, fragile part without touching it, then seat round a
centring step; a ring that gives the right cover but brushes the wax scores 0. The board has no dental-laboratory,
casting or investing task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the rings' and the grippers' state, with a tracker for the whole trial |
| `mechkit.py` | Builds each ring (four walls) as one loose rigid body and reports its state |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step (the program kit's file, unchanged) |
| `gripturn.py` | Reads each gripper's turn about its axis (the pose of its `gripper` link) at the same moment, for the finger test |
| `offline_model.py` | Offline checks only: where a released ring comes to rest (the simulator never calls it; see below) |
| `tests/test_scoring.py` | Offline tests of every tier and rule with its boundary, resets, blow-ups, the hooks, gripturn against a stand-in embodiment, the offline model, the exploits we tried, and the demo and its tier variants through our offline checker (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py`, `rigstate.py` and
`gripturn.py` sit next to `task.py` and hook `yam_arena` from the task file's import (each ring is built as one
compound rigid body, since a native box cannot be a hollow tube; both arms, and each gripper's `gripper` link, are read
before each scoring call); they change nothing the embodiment does, and the task uses the standard `task_score`
scorer. The arms are read before the embodiment steps the physics, one control step older than the rings: the wax is
fixed, so this only delays a touch's detection by one step, and the 88 mm 'held' gap and the 2.5 s success hold absorb
it at the release. If the arms cannot be read on a step (rigstate says why once on stderr), the gripper touch rule is
skipped and every ring counts as held on that step; if that is more than 1% of the steps the trial scores 0. If a
gripper's turn cannot be read (gripturn says why once on stderr, or its reading does not put the tool point within
3 mm and the axis within 3 degrees of rigstate's, in either quaternion order), that gripper's fingers are judged by
the turn-free zone on that step, which is never more lenient.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm; the right arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Three steel casting rings stand on the left of the table. Pick the one whose top edge will be 5 to 8 mm above the top of the blue wax crown once the ring stands on the black former. Lower it straight down over the crown and stand it on the former around the raised step, then let go. Leave the other two rings off the former. Never let a ring or a gripper touch the wax crown or its red sprue." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/casting-ring-over-wax/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Three steel casting rings stand on the left of the table. Pick the one whose top edge will be 5 to 8 mm above the top of the blue wax crown once the ring stands on the black former. Lower it straight down over the crown and stand it on the former around the raised step, then let go. Leave the other two rings off the former. Never let a ring or a gripper touch the wax crown or its red sprue." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/casting-ring-over-wax/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module and starts a new one
when the embodiment resets (through `rigstate`'s and `mechkit`'s reset hooks, or, failing both, when a ring jumps
back onto its start spot in one step, further than its velocity allows, with the other two on theirs). The run rate
is assumed to be 20 Hz (`-E control_hz=20`).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check: nothing may move); `TASK_TRACE=trace.jsonl` writes one JSON
line per control step with what the scorer saw and decided: `tier`, `touched` and `touch` (first step and cause),
`rings_on_former`, `arms_unavailable`, `arms_turn_unknown` (steps on which each gripper's turn was not known), per ring
(`rings.ring_c`, ...) `pos`, `tilt_deg`, `low_corner_mm` and `high_corner_mm` (its lower end's outer corners above the
table), `axis_offset_mm`, `wall_wax_gap_mm` and `min_wall_wax_gap_mm`, `encloses`, `on_base`, `on_former`, `held`,
`still`, `seated`; per gripper (`arms.left`, ...) `gap_mm`, `zone_margin_mm`, `min_zone_margin_mm` and `turn_known`;
both tool points and finger tips, each known closing axis (`left_closing_axis`, ...) and `gripturn_error` if the links
could not be read. `MECHKIT_TRACE=mechkit.jsonl` writes the rings' poses as mechkit reads them. Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 2 minutes; the last tests run our
offline checker on the folder when it is present).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Ring A, B and C in the text are `ring_a` (40 mm), `ring_b`
(24 mm) and `ring_c` (32 mm) in the code. The rings are open at both ends, so either end may be down: tilt is folded
(0 to 90 degrees) and the "lower end" is whichever end is lower.

Touching the wax, at any moment, ends the trial's chances (0, whatever follows), in either of two ways:

- **a ring's wall**: each of the four 4 mm walls, as a box posed as the ring is, within 0.5 mm of the crown or of the
  4 x 4 mm square post standing for the round 4 mm sprue (a separating-axis test against them grown by 0.5 mm);
- **a gripper's fingers**: a point of the crown's or the sprue's surface (sampled every 2 mm or finer) inside a
  finger. Each finger is a box 10 mm thick outside the gap, 40 mm wide and 70 mm long from the tips up to the finger
  carriage, posed as the gripper is (its tip point and axis from `rigstate`, its turn from `gripturn`) and grown 1 mm
  on every side. That is at least as big as the real finger (10 mm by 40 mm 19 to 38 mm above the tips, 6.4 mm by
  27 mm at the tips). On a step where a gripper's turn cannot be read, its boxes are swept round its axis, covering
  every turn: from half the gap less 1 mm to sqrt((half the gap + 11 mm)^2 + (21 mm)^2) from the axis, from 1 mm below
  the tips to 71 mm above them. The space between the fingers is never in a finger, so a ring held across its
  outside, or open fingers, can pass over the wax as real fingers would. The gripper's body above the fingers and the
  rest of the arm are not judged: straight down, the body stays at least 34 mm above the crown's top while the finger
  tips are above the table.

| Term | Meaning |
|---|---|
| Over the wax | Tilted 10 degrees or less; every corner of the crown and of the sprue post inside the 52 x 52 mm opening, in the ring's own frame; the crown's top above the ring's lower end |
| Held | A gripper's fingers under 88 mm apart, its tip point below the ring's top and within 45 mm of the ring's axis horizontally. A grip across the flats stops at 60 mm, one across the corners at 84.9 mm (60 x (cos t + sin t) at a turn t off the flats); the corners are 42.4 mm from the axis |
| Still | The ring's centre moved 2 mm or less in total over the last 0.5 s (10 control steps; positions, never velocities) |
| Seated | Over the wax, all four outer corners of its lower end 9 to 12 mm above the table (the base's top is 10 mm up, the step's 16 mm; within 3 mm of each other over the 60 mm sides, so also tilted under 3 degrees), not held, still |
| On the former | Its centre within 45 mm of the former's axis, seen from above, or a point of its lower end's outer edge (sampled every 5 mm) over the base's 88 x 88 mm outline and 5 mm or more above the table: standing, leaning or perched on the base, the step or another ring there, or held over the base |

| Outcome | Score |
|---|---|
| The wax touched at any moment (a ring wall within 0.5 mm, or a point of it in a finger) | 0 |
| At the end, two or more rings on the former | 0 |
| The grippers unreadable on more than 1% of the steps | 0 |
| Otherwise, ring C (32 mm) seated | 1.0 |
| Ring C over the wax but not seated (still held, hovering, crooked, on the step's edge, moving) | 0.5 |
| Ring A (40 mm) or ring B (24 mm) seated | 0.25 |
| Anything else (no ring over the wax, a wrong ring over it but not seated, nothing done) | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. Each outcome is one row: the zero rules come first, then ring C seated, then
ring C over the wax, then a wrong ring seated (ring C cannot then be on the former as well, or the two-rings rule
applies). The score is read at the last step; the trial ends early once 1.0 has held for 2.5 s. A simulator state with
a non-finite number, a zero quaternion or a missing ring scores 0 for that step and leaves the trial's record as it
was. The 2.5 s hold is longer than the time from "no longer held" (fingers past 88 mm, about 1.2 s into the 1.5 s
release) to the gripper being fully open, so the trial ends with the gripper open and moving away.

Unintended strategies earn nothing or the stated partial score, as the tests show: pushing ring C along the table and
across the former's base to the axis sweeps a wall through the crown (0); setting it down off centre on the crown and
then seating it properly stays 0 (the touch is latched); seating ring B and standing ring C on it puts two rings on the
former (0), and so does leaving ring B leaning with one wall on the base's margin beside a seated ring C; holding ring
C seated, or hovering it 2 mm over the base, is 0.5, whether gripped across its flats, 15 degrees off them (73.5 mm),
across its corners (84.9 mm) or pinched at one wall's end (tip point 41 mm from the axis); an open gripper lowered
beside the wax (turn unknown) touches it at some turn (0), and so does a closed one poking it or the idle arm reaching
in; a ring leaning on the crown is 0; an upside-down ring C seated is 1.0 (the tube is the same either way up); seating
ring B first, putting it back and then seating ring C is 1.0. With the turn known, pinching ring C by a single wall,
seating it and opening the fingers 16 mm is 1.0 for the x walls and the -y wall (the inner finger stays 5 to 10 mm
clear of the crown) and 0 for the +y wall (opening 16 mm moves the inner finger 2 mm into the crown; opening 8 mm
keeps it 2 mm clear, 1.0); a closed gripper with its axis 20 mm from the crown's face is clear when its fingers close
toward the crown (they reach 11 mm out) and a touch when turned a quarter (their 40 mm width reaches 21 mm out).

## One arm alone

The task needs one arm, and the demo uses one: the left arm picks ring C, carries it over the former and seats it,
while the right arm holds its ready pose. On the GPU, `DEMO_IDLE_ARM=right` (the left arm alone, the same run as the
demo) scored 1.0, and `DEMO_IDLE_ARM=left` (the right arm alone, which has nothing to do) scored 0.0. So one arm, the
left, completes the task. The right arm cannot reach any ring with a
straight-down grip: at x 20, 29 and 39 cm it reaches only y +16, +12 and +3 cm (yamkin straight-down IK), while the
rings stand at y +33, +32 and +29 cm, 17 to 26 cm beyond at their centres (14 to 23 cm to their nearest walls). A
grip tilted toward the left adds 10 to 16 cm, so at best the right arm might pinch ring A's near wall (14 cm beyond),
and ring A is a wrong ring (0.25 at most, and only if it could stand it level on the former); ring C is out of its
reach. No claim is made that the task needs both arms: its difficulty is choosing the ring by the cover it will leave
and setting it down straight round the wax. (The reach figures are offline kinematics; the GPU scores are in
'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 21.8 s (436 control steps), left arm only, yaw 0 throughout (fingers closing along y): over
ring C open (3.0 s), down to 6 cm (1.0 s) and slowly to the grip (1.5 s, TCP 3 mm below the ring's top, so the pads
hold the top 16 mm of its walls), settle 0.5 s, close to 52 mm across its 60 mm outside (1.5 s, about 20 N per finger),
lift to 12 cm (1.5 s; the ring's bottom about 9 cm up), carry over the former (2.5 s), down until its bottom is 10 mm
over the crown's top (1.5 s), settle 0.5 s, lower it round the wax over 2.5 s (35 mm, peak 21 mm/s) until its bottom
is 1 mm above the base, settle 0.8 s, open (1.5 s), straight up (1.5 s) and away toward its own side, 16 cm up
(2.0 s). Centred on the former's axis the ring's inner walls clear the crown by 14 mm (+y) to 26 mm and the step by
8 mm on every side, so the ring may land up to 8 mm off the axis along x or y and still stand on the base (8 mm off
toward -y its +y wall still clears the crown by 6 mm). The fingers close along y, so the grip centres the ring in y;
along x (the fingers' width) nothing centres it, and a pad offset at the pick carries straight into x at the seat.
The arm's controller does not carry the ring's 2.3 N: the static estimate of the sag at the seat is 2.3 mm in -x,
1.8 mm in +y and 5.2 mm down, with under 1 degree of tilt. The ring therefore rests on the base about 4 mm before the
slow descent ends (harmless; the base takes the weight), and the place point is moved by half the sideways sag
(+1.2 mm in x, -0.9 mm in y), so the ring lands within about 1.5 mm of the axis whether the arm sags as estimated or
not at all. On the GPU, read `rings.ring_c.axis_offset_mm`, `low_corner_mm` and `high_corner_mm` in `TASK_TRACE` at
the release: if the ring lands more than 6 mm off the axis, tune only `PLACE_TRIM` (and `SEAT_GAP`). Offline, success
has held for its 2.5 s at step 408 (20.4 s). On the GPU the trial ended at step 409 with ring C seated 1.7 mm off the
axis, level, its walls never closer than 14.1 mm to the wax. The run limit is 1,600 steps (80 s), 3.7 times the demo.

Our offline checker (no physics) carries each ring as its bounding box, a solid 60 x 60 mm block, so it reports
"carried ring_c runs into wax_crown / sprue / former_step" during the descent: the box, not the tube, does (the
scorer, which judges the real walls, measures 14 mm of clearance), and on release it drops that box onto the sprue
("centre not over its support", the ring listed as ending 26 mm up). `offline_model.py` is used instead for where a
released ring comes to rest: straight down onto the highest thing under its four walls (here the base), never onto
what lies inside its opening. These warnings are expected; the GPU run is the real check.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the three rings back upright on their start
spots, sides parallel to x and y (ring A at x 20.0 cm, y +33.0 cm; ring B at x 29.0 cm, y +32.0 cm; ring C at
x 39.0 cm, y +29.0 cm). The former, sprue and crown are fixed.

## Runtime per trial

Measured on an L40S, one trial per process including about 13 s of Isaac Sim start-up: about 1 minute for the
scripted demo (409 steps) and about 3.3 minutes for the Dream-1 trial (all 1,600 steps, 80 s of simulated time). The
first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Three lengths of 60 x 60 x 4 mm stainless or mild steel square tube, cut to 24, 32 and 40 mm and deburred (about
0.17, 0.23 and 0.28 kg). An 88 x 88 x 10 mm black rubber block clamped to the bench, with a 36 mm rubber disc, 6 mm
thick, glued at its centre. A 4 mm red wax sprue rod, 10 mm long, pressed upright onto the disc 6 mm off centre (to
the left), with a 10 x 12 x 10 mm block of blue inlay wax on it. All unpowered; nothing hot. Film from above and from
two sides; check the wax for dents after each trial, read the cover with a straightedge across the ring's top and a
depth gauge down to the crown, slide a 2 mm feeler gauge round the ring's lower edge, and count the rings standing
on, leaning on or held over the former.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |

The Dream-1 trial ran all 1,600 steps without moving any ring or touching the wax (score 0).
