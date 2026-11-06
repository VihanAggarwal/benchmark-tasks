# Hand-feed a pinned seam past the needle, pulling each pin just before it gets there

In free-motion quilting, darning or embroidery the machine's feed dogs are dropped and the operator moves the work,
often held in a frame or hoop, under a fixed needle. Pins hold the layers in register and are taken out one by one
just ahead of the needle: not earlier, or the layers shift; not later, because sewing over a pin can bend or break the
needle and throw fragments, as machine manuals and sewing guides commonly warn. Stopping the work to pull a pin is
normal practice. Here a cream frame plate (two plies of a seam, laminated, held in a sliding frame) lies on a long
grey machine bed and slides on a rail toward a fixed needle head on the right. Three red pins stand in it on its blue
seam line near the needle; a black handle stands on its far left end. One gripper must move the plate past the needle
by its handle; the other must grip each pin and pull it out while the pin is in a window 30 to 70 mm before the
needle line (a yellow band on the bed) and put it in a black dish. No pin may ever reach the needle, be carried under
the needle head or be knocked loose, and the plate may move only while a gripper holds its handle. The scorer watches
every control step: the plate's travel, where each pin is when it leaves the plate and whether a gripper held it then,
and how close each pin ever came to the needle.

The two jobs are far apart on purpose. The handle starts 38 cm left of the midline, out of the right arm's reach;
the pull window and the dish lie 29 cm or more right of it, out of the left arm's; both hold with the gripper
straight down or tilted up to 40 degrees (numbers in "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the plate's travel, the pins' states and the grippers' state, with a tracker for the whole trial |
| `mechkit.py` | Builds the bed and the frame plate as one fixed articulation with one prismatic joint (the rail) and reads its travel for the scorer each control step |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_model.py` | Offline checks only: a small kinematic model of the plate and the pins (the simulator never calls it; see "Offline checks") |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition and its boundary, resets, the gripper hook, the offline model, the exploits we tried, and the whole demo through our offline checker (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py` and `rigstate.py` sit next
to `task.py` and hook `yam_arena` from the task file's import (the work is built as an articulation; both arms are
read before each scoring call); they read only state the simulator already has, change nothing the embodiment does,
and the task uses the standard `task_score` scorer. If `rigstate` cannot read the arms it prints why once; on those
steps the rules about holding and letting go are taken as met, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sew the seam: move the cream plate at least 16 cm toward the needle by its black handle only. Grip each red pin and pull it out of the plate while it is beside the yellow bands before the needle, not earlier and not later, and put it in the black dish; never knock a pin loose. No pin may reach the needle or be carried under the needle head, and the plate may move only while a gripper holds its handle. Finish with both grippers open or clear of the handle and the pins." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/sew-past-pins/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Sew the seam: move the cream plate at least 16 cm toward the needle by its black handle only. Grip each red pin and pull it out of the plate while it is beside the yellow bands before the needle, not earlier and not later, and put it in the black dish; never knock a pin loose. No pin may reach the needle or be carried under the needle head, and the plate may move only while a gripper holds its handle. Finish with both grippers open or clear of the handle and the pins." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/sew-past-pins/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module and starts a new one
when the embodiment resets (through `rigstate`'s and `mechkit`'s reset hooks, or, failing both, when the plate and
all three pins jump back to their starting places in one step).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`feed_mm`, `feed_max_mm`, `handle_held_by`, `unheld_travel_mm`,
`pin_d_mm`, `in_work`, `pulls` with the gripper and each pin's `d_mm` when it left the plate, `min_d_mm`,
`pulled_in_time`, `min_needle_gap_mm`, `in_dish`, `pin_low_mm`, `pin_tilt_deg`, `zeros`, `zero_at` and
`zero_reasons`, `opening`, `grippers_let_go`, `arms_unavailable`, both tool points and finger tips, `source`);
`MECHKIT_TRACE=mechkit.jsonl` the plate's joint. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 2 minutes; the last test runs our
offline checker on the folder when it is present).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step the scorer reads the feed (how far the
plate has slid along its rail, mechkit's joint reading), the three pins, and each gripper's finger-tip point and
opening. A pin's d is its centre's distance before the needle line along the seam: the pins start at d 55, 105 and
155 mm and the feed takes d down. A pin is in the plate while its centre is over the plate and its lowest point less
than 3 mm above the plate's top (standing in its pocket it is 6 mm below the top; lying on the plate, on it). A
gripper holds a pin when it is less open than 0.30 (fingers under 28 mm apart) with its finger-tip point within 4 mm
of the pin; closed fingers that push a pin from the side keep that point at least 6 mm from it (they are 6.4 mm thick
at the tips), and fingers left open beside it are too open. A pin is pulled when it leaves the plate, lifted out of
its pocket or slid off the plate's side, while a gripper holds it; only its first pull counts. A pull is in time if
d was then 30 to 70 mm (the yellow band) and the pin never came closer than 30 mm to the needle line while in the
plate, at any time in the trial (feeding the plate back, or standing a pin back in its pocket and letting it ride on,
does not make a late pin timely). A gripper holds the handle when it is less open than 0.6 (fingers under 56 mm apart)
with its finger-tip point over the handle's footprint grown by 5 mm and below the handle's top: its fingers are round
the handle, not hovering over it. It has let go when it is open 0.9 or more, or its finger-tip point is more than
20 mm from the handle and every pin. A pin is in the dish when its centre is over the dish's 140 x 70 mm floor, its
lowest point below the 15 mm rim and it is still (under 1 cm/s and 0.5 rad/s). Fed means the furthest the plate has
slid.

Four rules hold for the whole trial; broken at any moment they score 0, whatever follows:

- Strike: any part of a pin comes within 6 mm of the needle line, or crosses it, while the pin is under the needle
  head (its centre between the posts' outer faces, within 72 mm of the seam line, and its lowest point below the bar's
  top, 90 mm above the table). For an upright pin that is its centre within 15 mm of the line; the needle itself is
  11 mm from the centre of a pin touching it. A pin riding in the plate into the needle, and a pin carried past the
  needle under the bar, both count; one carried over the bar or round a post does not.
- Knocked loose: a pin leaves the plate while no gripper holds it. On a machine a loose pin on the bed can be fed
  under the foot or into the bobbin area.
- The plate moves more than 5 mm in all (counted in 0.2 mm steps, both ways) while no gripper holds its handle.
- A pin falls off the table (its lowest point 10 mm or more below the table's top; a pin held in the air beyond the
  table's edge has not fallen).

| Outcome | Score |
|---|---|
| A strike, a pin knocked loose, the plate moved more than 5 mm unheld, or a pin off the table, at any moment | 0 |
| Fed 160 mm or more (past where pin 3 started), all three pins pulled in time and in the dish, both grippers let go | 1.0 |
| Fed 160 mm, all three pulled in time | 0.75 |
| Two pins pulled in time | 0.5 |
| One pin pulled in time and the plate fed 45 mm or more (pin 1 starts 55 mm before the needle line and is struck past 40 mm) | 0.25 |
| Anything else | 0 |

The ladder rewards the pulls in time, the skill the task is about; the last of the feed and tidying the pins away
come after them. The score is read at the last step; the trial ends early once the 1.0 conditions have held with the
pins still and the plate still (under 5 mm/s) for 1 s. A simulator state with a non-finite number, a zero quaternion
or a missing field scores 0 for that step and leaves the record of the trial as it was. A pin standing on either end
is upright.

Unintended strategies earn nothing or little, as the tests show: swiping the pins off the plate onto the bed or the
table, with fingers open beside them or closed against them, knocks them loose (0), and so does swiping only pin 1
and feeding on; pulling every pin at the start (pins 2 and 3 at 100 and 150 mm) and then feeding scores 0.25, only
pin 1 being in time; dragging the plate by a pin, pushing it anywhere but at its handle, or holding a closed gripper
over the handle while something else moves the plate moves it unheld (0); pins knocked over onto the plate stay in it
and ride into the needle (0); carrying a pulled pin along the seam past the needle under the bar is a strike (0), and
so is jumping it across the line between two steps; letting a pin come 20 mm from the needle line and feeding the
plate back before pulling it is late, and so is a token lift in time, the pin stood back in its pocket and pulled
again 16 mm from the line (0.5 at most); gripping a pin early and lifting it slowly while the plate feeds on is judged
where it comes out of its pocket, still early; holding the last pin in the dish instead of letting go scores 0.75. A
pin gripped and pulled only 8 mm up, then slid sideways off the plate to the dish, the way a real pin comes out, is a
proper pull (1.0).

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone): it grips the handle and feeds, but nobody pulls pin 1, which is struck
when the feed passes 40 mm (its centre 15 mm from the needle line) and then stops against the needle: 0.
`DEMO_IDLE_ARM=left` (the right arm alone): it pulls pin 1 in time and puts it in the dish, but nobody feeds the
plate: 0. (Offline predictions through the model below; the GPU runs will be recorded in `task.json`'s `scores`.)

Reach is what splits the roles (our offline kinematics: position and orientation IK, tilts of 15 to 70
degrees from straight down toward azimuths every 15 to 30 degrees and yaws every 30 degrees, joint limits, the arm's
collision capsules clear of the table and its own base; the arms are mirror images). On the plate's line, x 40 cm, at
grip height (tool point 30 to 60 mm above the table), the left arm's tool point reaches only about the midline
straight down (y -1.5 cm), y -24.8 cm tilted 40 degrees, y -29.2 cm at 50 and y -32.4 cm at 60. A pull in time happens
between y -29 and -33 cm and pin 1 starts at y -30.5 cm, so up to 40 degrees the left arm cannot pull any pin in time
(4 cm short) and cannot take pin 1 at all; nor does it reach the dish (its floor's nearest corner is 69 cm from the
left arm's base). Holding the handle at its start needs a finger-tip point over the handle's footprint grown by 5 mm,
y +36 cm or more, below its top; the right arm's finger-tip point gets no closer to that zone than 108 mm tilted 40
degrees, 61 mm at 50 and 25 mm at 60 (4 mm at 70, which the 5 mm the plate may creep unheld could close). Neither arm
can do the job alone with the gripper straight down or tilted up to 40 degrees. We do not claim more than that for the
left arm: tilted 50 degrees it reaches the window's near end and over the dish, at 60 degrees pin 1, so a policy able
to grip and carry pins with its gripper tilted 60 degrees or more from vertical, working in turns, might do the whole
job with the left arm alone (kinematics and the capsules' table clearance only: whether so steep a gripper could close
on a pin standing in the plate, beside the bed, is not checked). The scripted demo grips from above only.

The nearest board tasks differ in kind. Lifting a cable so a cart can pass (TK-FAMWR5): one arm advances a load while
the other clears an obstacle from its path, 0 on contact; the obstacle is cleared once and held, with no timing. Here
the obstacles ride in the work itself and each must come out inside a two-sided window (not before 70 mm, not after
30 mm) at a pace the robot sets, the work moving only between removals. Conveyor picking (TK-8NV6QX, biscuit packs off
a belt before a red line; TK-LJQEDJ, sorting blocks off a moving conveyor; RoboDojo's conveyor picking): the belt
drives itself and only the upper bound counts. Removing transport keys before moving a carrier (TK-VEY6WZ): every
restraint comes off first, then the carrier moves; here the restraints come off one at a time while the work moves.
Holding back a tote queue while pulling a tote (TK-DHZYJZ) is a static hold, and hand-guiding a carriage while the
other arm holds an enabling plunger in a band is a permission held throughout, not a timed removal from the moving
work.

## Demo and run limit

The scripted demo takes 48.9 s (978 control steps). Both grippers close their fingers along x (left yaw -90 degrees,
right +90), across the handle and the pins, so the open fingers never reach toward the needle head along the seam.
The left arm comes over the handle, lowers its open fingers round it (the last 3 cm over 1.5 s), settles, closes to
22 mm on the 30 mm handle over 1.5 s with its finger tips 10 mm above the plate, and holds (from 7.6 s). The right arm
has meanwhile come down round pin 1 and gripped it, and holds it in its pocket until 0.5 s after the handle is held
(real fingers lag their command), then pulls it 55 mm straight up. Each time a pin is clear, the left arm moves the
plate by the handle: to 50 mm (2.5 s), to 100 mm (2.5 s), and after the third pin to 178 mm (3.0 s), 18 mm past what
the seam needs and 2 mm short of the rail's end; then it settles, opens over 1.5 s and rises. The right arm takes
every pin at the same place, d 55 mm (y -30.5 cm, 25 mm inside the window's near end and 15 mm inside its far end): it
lowers its open fingers round the pin (the last 3 cm over 1.5 s), settles, closes to 10 mm on the 18 mm pin over
1.5 s, pulls it 55 mm straight up, carries it along x to the x of its place in the dish (every turn and every crossing
of the needle line happen with the tool point 4 cm or more outside the needle head's posts), turns the fingers to
close along y on the way over the dish, lowers the pin to 2 mm above the dish's floor with the finger tips 7 mm above
the rim, settles, opens over 1.5 s and rises. The pins stand 45 mm apart in the dish, so the fingers opening along y
stay 16 mm clear of a neighbour. Each pin is out of the plate before the plate moves, and the plate is still again
well before the next pin is gripped. The tool points stay more than 55 cm apart. Offline, success has held for its 1 s
at step 975; the scripted policy stops at step 1,039. The run limit is 3,000 steps (150 s at 20 Hz), 3.1 times the
demo: three timed pulls, each waiting on a feed.

## Offline checks

Our NumPy-only offline checker (not part of this folder) has no physics: it sees the work as mechkit's assembly at
rest, its rail always at 0, and carries a loose object only where the pads close across it at its starting place. So
when the work's state says it came from that checker (mechkit marks such states `source` "offline"; the simulator's
say "sim"), `task.py` (`is_offline_checker_state`, `_offline_checker_states`) feeds the scorer the plate's travel and
the pins from `offline_model.py` instead: the plate follows a gripper closed onto the handle (its finger-tip point in
the handle, opening 0.36 or less), clamped to the rail and stopped where a pin riding in it would touch the needle
(the grip then slips); a pin rides in its pocket until a gripper closes on it centred within 8 mm with its pads over
the pin's height, moves with that tool point, and when let go stands on whatever is under it (the dish's floor, a
pocket or the plate's top, then riding again, the bed, the bar, the table or the floor). Nothing else moves anything:
pushing the plate or dragging it by a pin moves nothing in the model, so an arm that does not do the job moves
nothing. It models no lag, sag, slip in the hand, rattle in the pockets, tipping or knocking; the GPU runs decide
those. The checker's own warnings that remain are about that approximation: the pins and the needle overlap the
work's bounding box, and the pins look 16 mm above the table (they stand in the plate's pockets); its own proxy drops
pins 2 and 3 to the table when it grasps pin 1 and never sees them pulled.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: slide the plate back to the start of its rail
(handle centred at y +38.0 cm), stand the three pins upright in their pockets (pin 1 nearest the needle) and empty
the dish. Nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes for the demo on an L40S once Isaac Sim is running (49 s of simulated time). A trial that runs
to the 3,000-step limit takes about 4 minutes of simulation, plus the policy's own time (a Dream-1 trial of that length
takes about 8 to 11 minutes); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

All of it is unpowered: there is no motor and the needle is a blunt rod. The bed is a 120 x 970 mm aluminium or
plywood board, 12 mm thick, with a captive linear slide under the plate (a ball-bearing drawer slide, or a channel of
two wooden side guides that the plate's edges run in) and an end stop at each end of the plate's 180 mm travel, and a
steel rule taped along its near edge; the red mark and the yellow bands are tape. The plate is 10 mm PVC foam board,
70 x 760 mm, with three 28 mm square pockets 6 mm deep routed on a drawn seam line and a 30 x 30 x 40 mm black plastic
block glued on as the handle. The pins are 18 mm aluminium rod cut to 40 mm and painted red (about 27 g each). The
needle head is two 10 x 10 mm posts and a bar, with a 4 mm blunt steel rod (or a dowel) hanging to 8 mm above the
plate; the dish is any 140 x 70 mm tray with 15 mm sides. Film from above and from the side: read the feed and each
pin's d at the moment it leaves the plate on the rule and against the marks, and check from the side view that each
pin is gripped as it comes out and that no pin passes under the needle head below the bar.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
