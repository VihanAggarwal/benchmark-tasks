# Lay a Japanese set meal for the guest opposite: rice left, soup right, fish head left

Japanese set meals (teishoku) are laid on the tray to a standard layout, given in the washoku guidance of Japan's
Ministry of Agriculture, Forestry and Fisheries and in table-manner and school food-education material. Seen by the
diner: rice front left, soup front right (to the right of the rice), the main dish back right (beyond the soup) and
the side dish back left (beyond the rice), the ichiju-nisai form; a whole fish lies with its head to the diner's left
and its belly toward the diner. Restaurant servers, ryokan inn staff and hospital and care-home meal-service workers
lay trays this way at every meal. The tray is laid for the guest, who sits opposite and faces the server, so the
layout has to be read from the guest's side: laid from the server's own side, every dish lands in the opposite
place and the fish faces the wrong way.

Here a dark red lacquer tray stands on the table and an indigo cushion beyond its far edge shows where the guest
sits, facing the robot. The four dishes, at about two thirds of real size, wait on two serving boards: a cream-white
rice bowl and a small blue side-dish bowl on the right, a black lacquer soup bowl and a white plate with a grilled
fish on the left. The instruction names the dishes and says the tray is for the guest opposite, but not where any
dish goes or which way the fish lies: that is what the task tests. The score is read from where the dishes stand at
the end, as a server or the guest would check the tray at a glance (a ruler and a top photo): 0.25 for each dish in
its place.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four dishes' poses (and the grippers' pad-tip points and openings, for 'let go'), with a small tracker for stillness, the tip latch and the trace |
| `mechkit.py` | Helper that builds each dish as one rigid body of several shapes (a bowl with a disc of food on top, the plate with its fish) and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the let-go rule, the tip latch, the exploits we tried, resets, broken states, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour each, and a plain box shows no fish
head. `mechkit.py` (imported by `task.py`) builds each dish as one loose rigid body through a hook on
`yam_arena.scene`: a bowl is a cylinder with a thin disc of food on top (the soup bowl also a red band at its rim);
the fish plate is a slab with a rigid fish on it (body, darker head and tail, a white belly strip along one edge and
a dark dorsal ridge along the other), so the fish's direction can be seen from above. It reports each dish's pose
(base-frame origin, the middle of its bottom face, and orientation) to the scorer, and stops the run with an error if
a dish was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read an arm it prints why once; the let-go rule then skips
that arm on those steps (the other is still checked), and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this teishoku tray for the guest who sits on the indigo cushion across the table, facing you. The white bowl is rice, the black lacquer bowl is miso soup, the small blue bowl is the side dish, and the plate holds a whole grilled fish. Put all four on the tray the way a Japanese set meal is laid for the guest, then let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/washoku-tray-diner-opposite/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this teishoku tray for the guest who sits on the indigo cushion across the table, facing you. The white bowl is rice, the black lacquer bowl is miso soup, the small blue bowl is the side dish, and the plate holds a whole grilled fish. Put all four on the tray the way a Japanese set meal is laid for the guest, then let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/washoku-tray-diner-opposite/dream
```

Run one trial per process (the default). The scorer keeps one trial's record (each dish's path over the last second
and the tip latch) and starts it over at every reset (rigstate's and mechkit's reset counters, and every dish
teleported back onto its starting mark in one step); the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: for each dish under `dishes`, its `centre_mm` (the middle of
its base), the `quadrant` it is in seen from the guest, `inset_mm` (how far inside its own quadrant), `overhang_mm`
(past the tray's edge), `rest_mm` (lowest point above the tray top), `tilt_deg`, `held`, `path_mm` (over the last
second), `still`, `overlaps`, `tipped_latched`, `points` and `zero_reason`, and for the fish plate `head_err_deg`,
`belly_err_deg` and `head_heading_deg` (the head's direction seen from above, counter-clockwise from +x; the goal is
-90); then `tipped_at_step`, `first_step` (when each score was first reached), `bad_steps`, `arms_readable`,
`arms_unavailable`, each arm's `left_tips`/`right_tips` and `left_opening`/`right_opening` (or `rigstate_error`).
Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four dishes' poses once per control
step (20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the
finger tips) and how far it is open, and scores the state at hand: the score is read at the last step.

The guest sits beyond the tray's far edge facing the robot (facing -x), so the guest's left is -y (the robot's
right) and the guest's front is the tray's far half. The tray's centre lines, x 33.0 cm and y 0, split it into four
quadrants seen from the guest. A dish's centre is the middle of its base; its footprint is a bowl's base circle or
the plate's 130 x 60 mm outline, seen from above.

| Dish | Its quadrant (guest's view) | Where that is on the table |
|---|---|---|
| Rice bowl | front left | x above 33.0 cm, y below 0 |
| Soup bowl | front right | x above 33.0 cm, y above 0 |
| Fish plate | back right, the fish's head to the guest's left, belly toward the guest | x below 33.0 cm, y above 0; head toward -y, belly toward +x |
| Side-dish bowl | back left | x below 33.0 cm, y below 0 |

A dish is placed when every one of these holds:

| Condition | Number |
|---|---|
| Centre in its quadrant | at least 10 mm from both centre lines (a dish astride a line is in neither) |
| Footprint on the tray | sticks out past the tray's edge by at most 5 mm |
| Standing on the tray | lowest point within 3 mm of the tray top (not on another dish, not held up) |
| Upright | tilted at most 10 degrees |
| Let go | no gripper's pad-tip point less than 10 mm above the dish's top while within that gripper's finger reach of its footprint, seen from above: hypot(half the finger gap + 10 mm, 20 mm) + 5 mm over every yaw, 65 mm fully open, 52 mm closed on the rice bowl |
| Still | its centre moved under 3 mm in all over the last 1 s (a path, not the reported velocity, which chatters while a part is gripped) |
| Fish plate only | the fish's head within 15 degrees of -y (the guest's left) and its belly within 15 degrees of +x (toward the guest), seen from above |

| Points | When |
|---|---|
| 0.25 per dish | placed |
| 0 for both | two dishes whose footprints cross by more than 1 mm (one on, leaning on or pushed under another) |
| 0 for the rest of the trial | a dish that tilted past 45 degrees at any step (tipped, its food spilled), even if set upright again |

Score = the sum over the four dishes: 0, 0.25, 0.5, 0.75 or 1.0. Every outcome, with numbers: the full layout for the
guest (the demo's places, each 55 mm inside one centre line and 75 mm from the other), 1.0; the same with the fish's
head to the guest's right (+y) or the plate lying front to back, 0.75; the soup tipped on the way and set upright in
its place, 0.75; rice and side dish swapped front to back (both on the guest's left, wrong rows), 0.5; rice and soup
swapped left to right, 0.5; the rice bowl pushed back against the side dish so their footprints cross, 0.5; one arm's
two dishes left on their board, 0.5; only the fish right, 0.25; the layout laid from the robot's own side (rice near
on the robot's left, soup near on its right, fish far on its right with its head to the robot's left, side dish far
on its left), 0, every dish in the opposite quadrant. A dish still held, astride a centre line, hanging off the tray,
on the table or on another dish earns 0 itself. The trial ends early (success) once 1.0 has held for 2.5 s;
otherwise it is scored as it stands at the run limit. A state with a non-finite number, a zero quaternion or a dish
missing (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

The plate cannot be laid as a mirror image: upside down it is tipped, so with its head to the guest's left its belly
always faces the guest; both are checked all the same. Pushing a dish across the tray instead of carrying it reaches
the same end state and scores the same.

Unintended strategies earn nothing extra, as the tests show: the layout laid from the robot's side (0); its mirror
images, left and right or front and back swapped (0); every dish crowded at the middle of the tray astride the centre
lines (0); a bowl set on the fish plate or on another bowl (both 0); dishes held in place by the grippers at the end
(0 for them); the plate stood on its edge and laid back (latched, 0); 400 random layouts score only 0, 0.25, 0.5 or
0.75.

## One arm alone

The demo splits the work by reach: the right (-y) arm lays the rice and the side dish (the guest's left half), the
left (+y) arm the fish and the soup (the guest's right half). Gripping straight down, at any heading (tried every 30
degrees), the left arm cannot reach the rice or side dish where they start (x 33.0 and 21.5 cm, y -30.0 cm) and the
right arm cannot reach the soup or fish plate (x 20.5 and 33.0 cm, y +30.0 cm), by our IK model of the arm
(`tests/test_scoring.py`, `OneArmReach`); neither arm reaches the front quadrant on the other side straight down
either. So with either arm idle the scripted demo lays two dishes: predicted 0.5 offline with the left arm idle and
0.5 with the right arm idle. Two arms are not needed for the skill being tested; the split follows reach. We do not
claim one arm is strictly unable to do more: with the wrist tilted from vertical the model's tool point does reach
the other side's dishes, so a policy able to grip or push them from the side might. (GPU results will be added under
'Results in simulation'.)

## Demo and run limit

The scripted demo takes 50.4 s (1,008 control steps). Each arm lays its own side's two dishes, and only one arm works
over the tray at a time (two wrists over the tray at once came within 1 cm of each other in our kinematic model):
before each pick an arm waits over the dish until the other arm has left the tray. In order:

1. The right arm sets the rice bowl front left (target x 38.5 cm, y -7.5 cm). Meanwhile the left arm waits over the
   fish plate.
2. The left arm grips the plate across its 60 mm width at its middle, by its two long edges under the fish, and
   turns it a quarter turn clockwise (grip yaw 0 to -91 degrees) while carrying it, so the fish's head swings from
   +x to -y (the guest's left) and its belly from +y to +x (toward the guest); it sets it back right (x 27.5 cm,
   y +7.5 cm). Meanwhile the right arm waits over the side dish.
3. The right arm sets the side-dish bowl back left (x 27.5 cm, y -7.5 cm) and parks.
4. The left arm sets the soup bowl front right (x 38.5 cm, y +7.5 cm) and parks.

Bowls are gripped straight down with the fingers closing along y, opened fully (94 mm: 14 to 19 mm either side of
a bowl) and commanded 8 mm narrower than the bowl (58, 56 and 47 mm), the pad tips 7 to 14 mm above the bowl's
bottom; the plate is commanded to 52 mm across its 60 mm, the pad tips 3 mm above its underside and the pads 12 mm
clear of the fish either side. Every gripper open or close takes 1.5 s, every final descent (onto a dish and onto
the tray) is slow (20 mm over 1.5 s) after a 0.8 s settle over the place, and each dish is let go 1 mm above the tray
after a 0.5 s settle. Dishes travel at a tool-point height of 9.5 cm, their bottoms 68 to 79 mm up, at least 12 mm
over any dish's top. Each place point leads its target by 4 mm along the way the dish travels (parts carried in a
gripper landed about 4 mm short in our earlier GPU runs), and the plate is laid 1 degree clockwise of -90 (carried
parts ended 0.6 to 1.7 degrees counter-clockwise of the commanded yaw); `PLACE_TRIM` and `YAW_TRIM` in `task.py`
adjust them if the GPU trace shows a steady offset. The margins are wide: each dish's centre is 55 mm inside its
quadrant against the 10 mm needed, every footprint is 10 mm or more inside the tray's edge (plus the 5 mm overhang
allowed), and the fish's 15 degree window dwarfs the expected 1 to 2 degree turn.

Offline (kinematic check, `tools/offline/check_task.py`): worst IK error 0.1 mm; the smallest joint margin 13.7
degrees (joint 4, over the front places at travel height), joint 6 at least 69 degrees from its limits; the arms'
collision capsules at least 18.1 cm apart and the tool points at least 37.5 cm; pad tips at least 5 mm above the
table. The last dish (the soup) is let go at 46.6 s and counts once the open gripper has risen with its pad tips 10
mm over the bowl's top; success then holds for its 2.5 s at step 1,021 (51.1 s); the scripted policy would stop at
step 1,069. The run limit is 2,400 steps (120 s), more than twice the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the rice bowl back upright on the right
serving board centred at x 33.0 cm, y -30.0 cm and the side-dish bowl at x 21.5 cm, y -30.0 cm; the soup bowl upright
on the left board centred at x 20.5 cm, y +30.0 cm; lay the fish plate on the left board centred at x 33.0 cm, y +30.0
cm, its length along x, the fish's head toward +x and its belly toward +y. The tray, cushion and boards are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 51 s of simulated time, when
success has held for 2.5 s; a run to the limit is 120 s); the first start of Isaac Sim on a new instance takes
several minutes more.

## Real-world build

A 220 x 300 mm lacquered wooden meal tray (a flat board 6 mm thick, or a real zen tray with its rim lowered), clamped
to the table on the midline 22 cm in front of the arm bases; a 120 x 200 mm indigo cloth or card beyond its far edge
for the guest's cushion; two 245 x 110 mm light wood boards (2 mm veneer or card) at the sides. Dishes (light replicas, so nothing
spills and a dropped dish does not shatter): a straight-sided 66 mm cream porcelain bowl 44 mm tall, filled level with
foam under a white felt or clay disc for the rice, 80 g in all; a 55 mm blue bowl 34 mm tall, filled the same way under
a green disc, 50 g; a 64 mm black lacquered wooden bowl 45 mm tall with a 4 mm red rim band on top, filled under a
brown disc, 70 g (add or remove filling to match each mass); and a 130 x 60 mm rectangular plate 12 mm thick with
straight edges (white melamine, or a white lacquered wooden slab) with a food-replica fish about 110 x 35 mm glued
along its middle (a restaurant display fish, or painted resin or wood: dark head at one end, white belly edge along
one side), 120 g with the plate. Positions are in
`task.json`'s `setup`.

Scoring a real trial: at the end, before the arms move away, take a top photo to check the let-go rule (no gripper's
pad tips lower than 10 mm over a dish's top within finger reach of it); then, once they have withdrawn, take a top
photo with a ruler on the tray (or measure): mark the tray's two centre lines; check each dish's centre is at least 10 mm into its quadrant, nothing sticks out past the
tray's edge by more than 5 mm, every dish stands upright on the tray with no dish over another seen from above, and the fish's
head points to the guest's left within 15 degrees (a protractor on the photo). Note any dish that tipped over during
the trial.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
