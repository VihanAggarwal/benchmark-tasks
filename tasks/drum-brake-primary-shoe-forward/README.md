# Fit drum brake shoes: short-lined primary forward, long-lined secondary rearward

Automotive service technicians reline drum brakes routinely, and fitting the shoes the right way round is part of
ASE A5 (Brakes) certification. On a duo-servo (Bendix-type) drum brake the primary shoe, with the shorter lining,
goes toward the front of the car and the secondary, with the longer lining, toward the rear: in forward braking the
primary is dragged round into the secondary, which does most of the braking. Left and right brakes are mirror
images, so with both backing plates the same way up "toward the front" is on opposite sides of the two plates, and
a fitter who copies the first brake onto the second gets it backwards. Here a half-size bench model of both rear
brakes of one car lies on the table, shoe side up with the anchor posts on the far side, a white arrow on each plate
(and a larger one on the bench beyond it) pointing toward the front of the car: toward +y on the left brake, toward
-y on the right. Four loose shoes lie in a row in front of them; the two primaries' painted linings run over 90
degrees of the arc, the two secondaries' over 130. The robot has to tell the shoes apart by their linings, put a
primary on each plate's seat toward its arrow and a secondary on the other seat, lay them flat and let go. The
instruction names the brakes as duo-servo but does not say which shoe goes where. The score is what a fitter reads
with a steel rule, a protractor and a look from above.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four shoes' poses and the grippers' state, with a small tracker of the shoes' last second |
| `mechkit.py` | Helper that builds each shoe as one loose rigid body (a block, twelve discs and its lining dots) and each backing plate as one fixed body with its painted seats and arrows, and reports their poses (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, the point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, all 24 ways of filling the four seats, the shortcuts we tried, unreadable and non-finite states, resets, the full shoe's clearances and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed cylinders and boxes, the standard `task_score` scorer, and two helper modules next to
`task.py`.

`yam_arena`'s objects are single boxes, upright cylinders or spheres, and none of them is a curved shoe.
`mechkit.py` (imported by `task.py`) builds each shoe from a 20 x 16 x 15 mm block at mid-arc (the part a gripper
closes on) and twelve upright 20 mm discs along its 46 mm radius centre line, with its lining as a row of thin dark
brown discs on its top, as one loose rigid body, and each backing plate with its painted seats and arrows as one
fixed body, through a hook on `yam_arena.scene`. It reports each shoe's base frame (its mid-arc point and heading)
to the scorer, and stops the run with an error if an object was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read an arm it prints why once; the let-go condition then
skips that arm on those steps, and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms, each fitting its own side's plate): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "These are the two rear drum brakes from one car, laid shoe side up on the bench with each anchor post on the far side. The white arrow on each backing plate points toward the front of the car from that brake. They are duo-servo brakes. Fit one primary and one secondary shoe to each backing plate, each shoe lying flat on a painted seat between the anchor post and the adjuster block, the way they go on this kind of brake, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drum-brake-primary-shoe-forward/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "These are the two rear drum brakes from one car, laid shoe side up on the bench with each anchor post on the far side. The white arrow on each backing plate points toward the front of the car from that brake. They are duo-servo brakes. Fit one primary and one secondary shoe to each backing plate, each shoe lying flat on a painted seat between the anchor post and the adjuster block, the way they go on this kind of brake, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drum-brake-primary-shoe-forward/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`score`, `line` (which line of the decision list matched),
`plate_fine`, `plate_coarse`, `seats_fine`, `seats_coarse`, `n_seated`, `counted`, `all_still_1s`, `first_step`,
`bad_steps`, `arms_unavailable`, both pad-tip points and openings, and for each shoe `mid_mm`, `yaw_deg`,
`tilt_deg`, `bottom_mm`, `held`, `counted`, `still_1s`, `nearest_seat`, `off_seat_mm`, `yaw_err_deg`, `path_mm` and
`yaw_range_deg`). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and
starts it over at every reset (rigstate's and mechkit's reset counters, and a teleport of every shoe back to its
starting mark), so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four shoes' poses and both grippers once
per control step (20 Hz). A shoe's base frame is its mid-arc point (the middle of its 46 mm radius centre line, on
its bottom face) with x pointing radially outward through the lined, convex side and z up through the lining; its
heading is that x axis seen from above. Each plate has two seats, one each side of the line from the anchor post to
the adjuster block: the seat's target is the mid-arc point 46 mm from the plate's centre straight toward +y or -y,
the heading pointing the same way, which puts the shoe's centre of curvature on the plate's centre and its ends
10.8 mm clear of the post and 7.8 mm clear of the block. The seat on the arrow's side is the front seat: on the left
plate the +y one, on the right plate the -y one.

A shoe counts only once it is let go and settled. Let go: at every step of the last 0.5 s, no gripper's pad-tip
point was lower than 10 mm above the shoe's top while within that gripper's finger reach of the shoe's outline seen
from above. rigstate gives no gripper heading, so a finger may be on any side of that point: the reach is
hypot(half the finger gap + 10 mm, 20 mm) + 5 mm, 65 mm fully open and 27 mm closed. Settled: over the last 0.5 s
the mid-arc point's path is under 2 mm and the heading stays within a 1 degree range (paths, not the simulator's
reported velocities, which chatter while a part is gripped). Seated (fine): counted, flat lining up (up axis within
5 degrees of vertical: a shoe upside down is 180 degrees off), its lowest point within 3 mm of the plate's top (not
on a post, on the adjuster or on another shoe), its mid-arc point within 8 mm and its heading within 8 degrees of
the seat's. Seated (coarse): 14 mm and 15 degrees. A shoe turned concave side out is 180 degrees off: never seated.
A seat holds a shoe only when exactly one shoe is seated in it. A plate is correct (at a level) when its front seat
holds a primary and its rear seat a secondary, both seated at that level. Either primary may go on either plate.

| Outcome (first match wins, read at the last step) | Score |
|---|---|
| Both plates correct (fine), all four shoes still for 1 s (settled in each of the last two 0.5 s windows); success once held 1 s | 1.0 |
| Both plates correct (coarse): a shoe 8 to 14 mm or 8 to 15 degrees off, or the 1.0 layout not yet still for 1 s | 0.75 |
| Exactly one plate correct (coarse); the other unfinished, empty, or with its shoes swapped (the mirror error) | 0.5 |
| Two or more shoes seated (coarse) anywhere, no plate correct (for example both brakes fitted primary rearward) | 0.25 |
| Fewer than two shoes seated | 0 |

The mirror error (one brake copied onto the other) is ranked with an unfinished brake, not above it, because it is
the mistake the rule exists to prevent. Nothing here needs a latch: trying a shoe in a seat and moving it again is
normal bench work. A new trial is recognised at rigstate's or mechkit's reset, or when the simulator puts every shoe
back on its starting mark (within 4.5 cm of it, which covers a seeded reset's jitter) in a single step. A simulator
state with a non-finite number, a zero quaternion or no base-frame origin scores 0 for that step and leaves the
scorer's record as it was.

Shortcuts and their scores, all in the tests: taking each shoe straight forward to the nearer seat (every primary
lands on a rear seat) 0.25; copying one plate's layout onto the other 0.5; both brakes primary rearward 0.25; every
one of the 24 ways of filling the four seats scores 1.0 (the 4 right ones), 0.5 (one plate right, 8) or 0.25 (12),
nothing else; a shoe concave side out, upside down, on its edge, tilted 6 degrees, 3.5 mm up (on a post or another
shoe) or stacked on the other shoe is not seated; holding the primaries in their seats leaves only the two
secondaries counted (0.25), holding them with nothing else fitted scores 0; the full fit with the last shoe still
gripped (or the gripper open but not lifted) scores 0.5 until the gripper rises clear, then 1.0 after a second;
nothing moved scores 0; 300 random states score only 0, 0.25, 0.5, 0.75 or 1.0.

## One arm alone

The demo uses both arms at once, each fitting its own side's plate with its own side's two shoes. With
`DEMO_IDLE_ARM=left` or `right` the other arm fits its own plate correctly and the far plate stays bare: the offline
check predicted 0.5 each way, and the GPU one-arm runs scored 0.5 each way. Gripping straight down (yamkin, 25 headings), neither
arm reaches either seat of the other plate, nor the other side's outer (secondary) shoe; each does reach the other
side's inner primary. A tilted gripper might reach further, so no claim is made that one arm alone cannot score
more. No claim is made that the task needs two arms: its difficulty is telling the primary from the secondary by
the lining's length, knowing that the primary goes toward the front, and reading the front from each plate's arrow,
which points opposite ways on the two mirror-image brakes.

## Demo and run limit

The scripted demo takes 34.9 s (698 control steps). Both arms run on one clock. Each comes down round a shoe (pad
tips 4 mm above the bench, the tool point 3 mm inside the shoe's mid-arc point, fingers open to 50 mm: 12 mm clear of
the outer face and 9 mm clear of the concave side at the fingers' corners), grips it at its mid-arc block across its
width (fingers commanded to 12 mm, over 1.5 s), lifts it to 10 cm (tool point), carries it over its seat in 2.5 s while turning it a quarter turn,
lowers it, the last 20 mm over 1.5 s, until its underside is 1 mm above the seat's paint, waits 0.5 s, opens over
1.5 s and goes straight up. Every seat's radial direction is along y, so every shoe is laid at gripper yaw 0 and
picked at the yaw that turns it onto its seat: yaw -90 degrees for a +y seat, +90 for a -y seat. The row is ordered
secondary, primary, primary, secondary from -y so that each arm's inner pick needs the yaw its wrist reaches there
(the right arm +90 degrees, the left arm -90; the other sign at an inner pick runs joint 6 into its 120 degree
limit). The right arm takes its inner shoe (the primary) to the right plate's front seat while the left arm takes
its outer shoe (the secondary) to the left plate's rear seat; then the right arm takes its secondary to the rear
seat and the left arm its primary to the front seat, so the two grippers stay at least 0.34 m apart in y. Both arms
then park over their own side of the bench, clear of the plates.

Two trims are pre-set from our earlier simulator runs. Carried parts ended turned counter-clockwise of the commanded
heading by about 0.6 to 1.7 degrees, so every place yaw is set 1.15 degrees clockwise (`YAW_TRIM`); offline, with no
such turn, each shoe is laid 1.15 degrees turned, inside the 8 degree window. Parts landed about 4 mm short along the
carry, so each place point is led 3 mm on along the carry's direction (`PLACE_LEAD`). Closing across the shoe, the
outer finger meets the flat block face 10 mm out from the mid-arc point but the inner finger meets the concave side
first at its edges, about 16 mm in, so the jaws centre on a point about 3 mm inside the mid-arc point. The tool point
therefore comes down 3 mm inside the mid-arc point (`GRIP_INSET`: both fingers meet the shoe together and do not push
it along the bench), the shoe rides 3 mm outward of the tool point in the hand, and each place point is set 3 mm
inside its seat's mid-arc point, toward the plate's centre. Expected in the simulator: each shoe lands within about
4 mm of its seat (3 mm on along the carry, less 3 to 7 mm of lag). Offline, where the checker centres the grip block
between the fingers, each shoe lands 1.3 to 2.2 mm from its seat. The offline checker sees each shoe as its grip block (the whole shoe's box would put its centre 12 mm inside
the arc), so the tests check the whole shoe's clearances separately: open fingers at least 9 mm from it at every
pick and place (7 mm with the shoe centred as the checker has it) and 30 mm from the other shoe on the plate; a fitted shoe 10.8 mm from its post, 7.8 mm from its
adjuster and 25 mm from its neighbour. Offline success holds for its 1 s at step 684 (34.2 s), after both grippers
have risen clear of the last two shoes. GPU watch (`TASK_TRACE`): after each "let go", each shoe's `off_seat_mm`
should be under about 4 mm and `yaw_err_deg` within 2 degrees (if not, move that arm's `PLACE_TRIM` by minus the
offset, or `YAW_TRIM` by minus the error); `tilt_deg` should stay under 2 degrees in the hand (if a shoe tips in the
grip, grip 2 mm nearer the centre of curvature). The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the four shoes back flat, lining up, convex
side toward the plates (+x), the middle of each arc's outer edge at x 26.0 cm (mid-arc point at x 25.0 cm): a
secondary centred at y -33.0 cm, a primary at -11.0 cm, a primary at +11.0 cm and a secondary at +33.0 cm. The plates,
anchor posts, adjuster blocks, painted seats and arrows are fixed; nothing is consumed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 35 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 120 mm steel discs, 6 mm thick, screwed to the bench 34 cm apart centre to centre as the backing plates; on each
a 10 mm steel pin 8 mm tall, 46 mm from the centre on the far side (the anchor), and a 12 x 14 x 8 mm brass block
46 mm from the centre on the near side (the adjuster). Paint two filled shoe footprints in yellow on each plate, one either
side of the line from pin to block, and a white arrow marked FRONT on each plate's centre and on the bench beyond it,
toward the left on the left plate and toward the right on the right plate. Print four shoes solid in PETG (about 41 g each): a
20 mm wide, 15 mm tall curved bar with a 46 mm centre-line radius, rounded ends, 140 degrees end to end; paint them
steel grey, with a 6 mm wide band 1 mm high printed along the outer edge of the top and painted dark brown, over the middle 90 degrees on two shoes
(the primaries) and the middle 130 degrees on the other two (the secondaries). Nothing moves or is powered. Score
with a steel rule and a protractor (or a top photo with a grid overlay) once both grippers are lifted clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
