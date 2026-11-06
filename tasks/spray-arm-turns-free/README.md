# Load a washer rack so its spray arm turns freely, then turn the arm half a turn

Sterile processing technicians load washer-disinfector racks before instruments are cleaned. The rack's rotating
spray arm cleans only while it turns: an item standing in the arm's sweep stops it, and that load comes out with soil
left on it and has to be run again. So short items may go under the arm, tall ones only where its ends cannot reach,
and the technician turns the arm by hand to check that it moves. Here a washer rack (a floor plate with low walls, 30
x 30 cm inside) is fixed on the table with a feed hub at its centre and a 22 cm spray arm turning on the hub, its
underside 8.5 cm above the table and a red cap on one end; two fixed holders, 5 cm tall, fill the rack's far corners.
Two white jars, 12 cm tall, and two steel bowls, 4 cm tall and upside down, stand beside the rack, one of each on each
side. Every item must end standing on the rack floor, each jar with its side 5 mm or more outside the 11.05 cm circle
the arm's ends sweep (its centre at least 14.05 cm from the hub's axis), and the arm must be turned at least half a
turn, so that its two ends between them pass over the whole circle, with the load let go of and left as it stands.
With the holders in the far corners, the only places a jar can stand clear of the sweep are the two near corners, so
the bowls belong under the arm. The end state is measured with a ruler; the turn with a top video of the red cap.

Nearest tasks on the board, and why this one differs: TK-23H4KG (clear the swing path of a fire door) is also about the
area a moving part sweeps, but there nothing moves: the robot works out a quarter circle on the floor and takes the
items in it out, leaving the rest alone. Here the robot loads items round the swept area by height: tall ones must stand
outside the circle, where the rack has room only in its two near corners, so the short ones belong under the arm (a bowl
in each near corner leaves a jar nowhere clear: 0.25 at best); then it proves the clearance by turning the part itself,
with the load let go of and undisturbed. TK-JB93HQ (load two tubes into a centrifuge so it can spin) is about where
loads go so a rotor can run, but its rule is balance (opposite pockets) and the rotor is never turned; here the rule is
swept clearance and the arm's turn is part of the score. TK-YG6R3L (pilot: fit toy car wheels, then check they turn)
checks a turning part after assembly, not a load around it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the rack and its holders, the spray assembly, the jars and bowls, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the items' poses, the spray arm's angle and the grippers, with a tracker for the whole trial |
| `mechkit.py` | Builds the feed hub and the spray arm as one fixed articulation with a revolute joint about the vertical, and reads the arm's link pose for the scorer each control step |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_model.py` | Offline checks only: a small kinematic model of a gripper pushing the spray arm round (the simulator never calls it; see "Offline checks") |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition and its boundary, the exploits we tried, resets, bad states, the offline model, the hooks and a kinematic replay of the whole demo |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py` and `rigstate.py` sit next to
`task.py` and hook `yam_arena` from the task file's import (the spray assembly is built as a mechkit articulation; both
arms are read before each scoring call); they change nothing the embodiment does, and the task uses the standard
`task_score` scorer. The rack, the jars and the bowls are `yam_arena`'s own fixed boxes and loose cylinders (the holders
are fixed boxes too).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the two white jars and the two steel bowls onto the washer rack, each standing upright on the rack floor inside its walls, so the spray arm can turn freely: anything taller than the arm's underside must stand at least 5 mm outside the circle the arm's ends sweep. Then, without moving or touching the load, turn the spray arm round at least half a turn, so its two ends between them pass over the whole circle. Finish with both grippers at least 2 cm clear of the items." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/spray-arm-turns-free/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the two white jars and the two steel bowls onto the washer rack, each standing upright on the rack floor inside its walls, so the spray arm can turn freely: anything taller than the arm's underside must stand at least 5 mm outside the circle the arm's ends sweep. Then, without moving or touching the load, turn the spray arm round at least half a turn, so its two ends between them pass over the whole circle. Finish with both grippers at least 2 cm clear of the items." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/spray-arm-turns-free/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module and starts a new one
when the embodiment resets (through `rigstate`'s and `mechkit`'s reset hooks, or, failing both, when the load jumps
back onto its start spots in one step with the arm back along x).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`loaded`, `jar_clear`, `jar_hub_mm`, `jar_side_mm`, `tilt_deg`, `bottom_mm`,
`held`, `turn_deg`, `arm_yaw_deg`, `arm_joint`, `stretch_start`, `stretch_turn_deg`, `best_turn_deg`, `stretches`,
`max_drift_mm`, `latched`, `latch_step`, `still`, `path_mm`, `arms_unavailable`, the items' centres and the finger
tips); `MECHKIT_TRACE=mechkit.jsonl` the spray arm's joint and link pose. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step (20 Hz) the scorer reads the four items'
poses, the spray arm's angle (the direction of its red end, from the arm link's pose, counted on continuously by adding
each step's change, so 200 degrees reads 200, not -160) and both grippers' finger-tip points.

An item is **loaded** when its centre, seen from above, is inside the rack's inner walls (x 12.0 to 42.0 cm, y -15.0
to +15.0 cm), its axis is within 10 degrees of vertical (either end up: the bowls go in upside down) and its lowest
point is within 5 mm of the rack floor's top. A jar is **clear** when its centre is at least 14.05 cm from the hub's
axis and its side is at least 11.55 cm from that axis at every height the arm and its cap pass (8.5 to 10.9 cm above
the table; the jar's cross-section there taken as a circle of radius 2.5 cm / cos(tilt)), so its side is 5 mm or more
outside the 11.05 cm circle the arm's ends sweep even if it leans toward the hub. A gripper is **on an item** when its
fingers come within 2 cm of the item's side, horizontally, with its finger tips below the item's top; the fingers are
taken as a disc round the point midway between the finger tips, out to a finger's outer corner (half the finger gap
plus 1 cm out, 1.35 cm across: 1.68 cm with the fingers shut, 5.85 cm fully open), whichever way the gripper is
turned. A **load stretch** runs while all four items are loaded, both jars clear and no gripper on an item, and ends
on a step on which that is no longer so or any item's centre is more than 10 mm from where it was when the stretch
began (a new one can start on the same step). A stretch has a **clean half turn** once the arm's angle has been 180
degrees or more from its angle at the stretch's start, either way. Full marks need the clean half turn in the stretch
still running at the end: the load let go of and left as it stood while the arm turned, as it then goes through the
wash.

| Outcome at the end | Score |
|---|---|
| An item not loaded (on the table, lying, stacked, on a holder, propped on a wall, held up), or a gripper on an item (its fingers within 2 cm of the item's side, its finger tips below the item's top) | 0 |
| Every item loaded, a jar not clear (its centre less than 14.05 cm from the hub's axis, or its side less than 11.55 cm from it at the arm's height) | 0.25 |
| Every item loaded, both jars clear, the stretch running at the end has no clean half turn | 0.5 |
| Every item loaded, both jars clear, the stretch running at the end has a clean half turn | 1.0 |

Possible scores: 0, 0.25, 0.5, 1.0; every outcome falls in exactly one row. The trial ends early once 1.0 has held for 1
s with every item still (its path over the last 0.5 s under 5 mm); otherwise it is scored at the run limit. Grippers
that cannot be read on a step are taken from the last step they could be read (never read: no gripper is on an item;
`arms_unavailable` in the trace counts such steps, and it should stay 0). A state with a non-finite number or a zero
quaternion scores 0 for that step and leaves the record as it was.

Unintended strategies earn what the table says and no more, as the tests show: turning the arm before the load is in
place counts for nothing (0.5 once loaded); so does turning it while the last item is still in a gripper, even set 4 mm
above the floor where it already counts as standing (0.5 once it is let go of and the gripper has left), or with a
gripper resting on any item (the stretch starts again once the hand is off); turning it back and forth never adds up
past the largest swing (150 degrees each way: 0.5); a jar left in the sweep stops the arm or is tipped by it (about 0.27
N at the arm's height tips it, 1.2 N would slide it), and while it stands there no turn counts, so a jar shoved clear by
the arm only starts the count from where it came clear; nudging the load more than 10 mm mid-turn restarts the count,
and so does a slow creep or dragging the load round with the arm (the 10 mm is from where it stood when the stretch
began, not per step); a jar laid on its side would pass under the arm but is not standing (0); a jar on a bowl, on a
holder, on a wall or leaning on one is not on the floor (0); a bowl in each near corner leaves no place where a jar is
clear (the best is 136.8 mm from the hub, 3.7 mm short: 0.25), and the far corners are the holders' (an upright jar
anywhere in the rack's far part is at most 131.2 mm from the hub, 9.3 mm short); both jars fit in one near corner only
pressed against its walls, with under 3 mm to spare each, leaving the other corner to a bowl, which then scores 1.0 like
any other clear loading; the bowl in the corner and the jar beside the hub leaves the jar in the sweep (0.25); a jar
centred 14.06 cm out but leaning 9.9 degrees toward the hub reaches 1.7 mm into the sweep at the arm's height and is not
clear (0.25), though once the grippers have let go nothing on the table can hold a jar at such a lean (a jar tilted less
than 22.6 degrees rights itself unless something props up its raised edge, and nothing on the rack floor stands lower
than 2 cm); a gripper still holding a jar, or an open gripper with one finger against a jar's side, scores 0; turning
the arm, then moving a jar into the sweep, scores 0.25; and turning the arm, then moving any item more than 10 mm, even
to another allowed place, scores 0.5 until the arm is turned half a turn again (1.0 then, counted in the new stretch).

## One arm alone

On the GPU, `DEMO_IDLE_ARM=left` and `DEMO_IDLE_ARM=right` both scored 0.0, as predicted offline: with the right arm idle, the left arm loads its own side and turns the spray arm 260 degrees, but the right jar and bowl stay on the table, and an item not loaded scores 0.

`DEMO_IDLE_ARM=left` and `DEMO_IDLE_ARM=right` both predict 0 offline: the idle arm's jar and bowl stay on the table
beside the rack, and an item not loaded scores 0. (The GPU results are in 'Results in simulation' below.)

The half turn itself is one-arm work: either arm reaches the whole push circle round the hub, and the demo's turning
is done by the left arm alone. Two arms are needed only because the items start on both sides of the rack, beyond one
arm's straight-down reach. With its gripper pointing straight down an arm misses the far side's bowl (at x 40.0 cm,
30.0 cm past the midline) by 181 mm and the far jar by 88 mm. Tilted grips reach further: about 30 degrees of tilt
reaches the far jar, but the far bowl is still 73 mm out of reach at 30 degrees and 36 mm at 45 degrees (24 mm with a
grasp 25 mm off its centre), and pushing it toward the rack would need a finger on its far side. A gripper tilted
about 60 degrees does reach the top of the far bowl's near half; pressing down on it and dragging it by friction might
bring it within reach. We have not tried that, so a single arm scoring 1.0 with steeply tilted grips is not ruled out;
no claim is made that the task needs both arms.

## Demo and run limit

The scripted demo takes 55.6 s (1,112 control steps). Both arms grip with their fingers closing along x (left yaw -90
degrees, right +90 degrees), 8 mm narrower than the part, over 1.5 s after a 0.6 s settle, and lower the last 28 mm of
every descent in 1.5 s. First the left arm sets its jar in the rack's near left corner (centre at x 15.5, y +11.5 cm:
162.6 mm from the hub's axis, 22.1 mm more than the 14.05 cm needed, 10 mm from both walls) while the right arm sets
its bowl under the arm's sweep beside the hub (x 26.5, y -10.5 cm: 10.5 cm from the hub's axis, 10 mm from the side
wall, the open far finger 13 mm from the holder); then they swap: the left bowl at x 26.5, y +10.5 cm and the right
jar in the near right corner, so the two arms are never low in the rack at the same x. Jars are gripped with the
finger tips 12 mm above their middle and set down 1 mm above the floor; bowls at their middle, carried with their
bottom 7.2 cm up (1.8 cm over the holders), and set 1 mm above the floor. The right arm then parks over its base (t =
33.2 s). The left arm shuts its fingers into a paddle on the way to a 72 mm circle round the hub, comes down on it at
phi 135 degrees (anticlockwise from forward), its finger tips 75 mm above the table (10 mm below the arm's underside,
so the fingers cover the arm's height, and 3.1 cm over the bowls), and from t = 37.7 s pushes the arm's rear end
round, 15 degrees every 0.4 s, to phi 240; it backs off, rises with its tips 13 mm over the red cap, goes back round
over the arm to phi 40 and pushes the red end round from there to phi 240. Offline the arm turns 258 degrees in all
(the joint stops at 355); the load stretch starts at step 617 (t = 30.9 s, once both grippers are clear of the last
jar and bowl), it has its clean half turn at step 999 (t = 50 s, in the second push) and success has held for its 1 s
at step 1,018. The scripted policy stops at step 1,173. The run limit is 3,600 steps (3 minutes), 3.2 times the demo;
the commands above pass `--max-steps 3600`.

What to check first in `TASK_TRACE` on the GPU:

1. `turn_deg` must rise smoothly through 180 degrees to about 255 to 265 (`arm_yaw_deg` wraps at 180; `arm_joint`,
   in radians, should match `turn_deg`). If it stalls near 180, the revolute joint is being wrapped: report it.
2. Settle (`DEMO_HOLD_ONLY=1`): `turn_deg` stays within 1 degree of 0 and every `tilt_deg` stays under 1.
3. The pushes: `turn_deg` should lead the left finger tips' angle about the hub by 15 to 25 degrees while pushing and
   coast a few degrees after each 15 degree step (the arm's drag is light). If the fingers land on the arm instead of
   beside it, start push 1 at a smaller phi (`PUSH1` in `task.py`); if the push misses the arm's height, tune `PUSH_Z`
   between 0.085 and 0.095.
4. The jars' set-down in the corners: `tilt_deg` under 2 and `bottom_mm` about 0 after release, `jar_hub_mm` about
   163 (140.5 needed) and `jar_side_mm` about 138 (115.5 needed).
5. The bowls' grasp (12 mm of clearance per side when open), their carry over the holders and set-down beside the hub
   (the open far finger 13 mm from the holder).
6. Through the pushes: `max_drift_mm` under 10, `held` empty, `arms_unavailable` 0 (unreadable arms count as clear,
   so a non-zero count means the `held` rule was not checked on those steps), `stretch_start` about step 617, after
   both grippers have left the last items and before the first push, and unchanged to the end, `latch_step` in the
   second push. If the light arm link is unstable under finger contact, raise `ARM_MASS` to 0.12 kg in `layout.py`.

## Offline checks

Without a simulator the spray arm's joint never moves, so when the hooks get a spray state whose `source` is
"offline" (only a kinematic replay says so), `task.py` feeds the scorer the arm's angle from `offline_model.py`
instead of the link pose: a gripper whose finger-tip point is 45 to 105 mm above the table and 20 to 110 mm from the
hub's axis keeps the arm's line at least atan(24 mm / r) from its tips' angle about the axis, on the side it came
from, so it pushes the arm ahead of its fingers; nothing else turns it, and it stops at the axle's stops. The model
has no coasting, drag or slip. Through it the demo predicts 1.0 and either arm alone 0.

The replay's warnings for this task: the left gripper "closes on nothing" at t = 35.5 s (it shuts its fingers into a
paddle for pushing, on purpose); the demo runs more steps than the walkthrough's 600 (pass `--max-steps 3600`); and
`task.json` has no media ID until the videos are uploaded.
The spray assembly is checked as its bounding box at rest (approximate), so its contacts with the pushing fingers are
not checked offline. The two arms' collision capsules stay 6.6 cm apart at their closest (their forearms, while the
left arm sets its jar in the near corner and the right arm its bowl beside the hub) and the tool points 25.0 cm apart.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the jars upright at x 18.0 cm, y +28.0
and -28.0 cm and the bowls upside down at x 40.0 cm, y +30.0 and -30.0 cm, on the table, and turn the spray arm back
along x with its red end forward. The rack, its holders and the hub are fixed; nothing is consumed.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is about 51 to 56 s of simulated time); the
first start of Isaac Sim on a new instance takes several minutes more. A Dream-1 trial at the full 3,600 steps takes
about 9 to 13 minutes.

## Real-world build

A 4 mm plywood tray 31.2 cm square with 6 x 20 mm battens standing on its edges (30 x 30 cm inside), screwed to the
bench; two 8.5 cm square wooden blocks 5 cm tall glued in its far corners, against the battens; a 3 cm dowel hub 8 cm
tall glued at its centre (its top 8.4 cm above the bench), with an M6 bolt standing up out of its top as the axle and a
nylon washer on it; a 22 cm length of hollow 20 x 20 mm aluminium square tube (about 80 g), drilled through its middle
and turning on the bolt with its underside 8.5 cm above the bench (a nut or a sleeve under it), a red plastic cap or
red tape on one end; two closed 5 cm plastic jars 12 cm tall weighted to 100 g; and two 7 cm stainless gallipots about
4 cm tall (60 g), set upside down. Draw the 11.05 cm circle the tube's ends sweep and circles of 11.55 and 14.05 cm
round the bolt on the tray's floor. Nothing is powered and nothing sharp is used; the arm turns by hand with a light
push.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
