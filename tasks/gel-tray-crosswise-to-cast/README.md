# Set a gel tray up to cast: crosswise so its end gaskets seal, comb in an end slot

Biological and laboratory technicians cast agarose gels every day for DNA and RNA electrophoresis. In gasket-type
horizontal gel boxes the gel is cast in the tank itself: the UV-clear tray, open at both ends, is turned crosswise
on the platform so the rubber gaskets on its two open ends press against the tank's side walls and seal them, and
the comb is stood in the slot near one end, so the wells sit at one end of the gel with its whole length to run.
Once the gel has set, the tray is turned lengthwise (open ends toward the electrodes) to run it. A tray left
lengthwise leaks the molten agarose out of both open ends; a comb in the middle slot halves the run length. Here a
clear gel tank with a grey casting platform and its two electrodes stands on the table; the tray lies lengthwise on
the bench on the left and the comb stands in a white comb stand on the right. The robot has to turn the tray a
quarter turn on the way in, set it down between the side walls, stand the comb in a slot near one end and let go of
both. The score is what a technician sees in a photo from above and checks with a rule.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the tray's and the comb's poses and the grippers' state, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds the tray and the comb as rigid bodies of several boxes each, and reports their poses (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `offline_nest.py` | Used only by an offline kinematic check (no physics): never called in the simulator |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, the exploits we tried, unreadable and non-finite states, resets, the offline helper and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain box has no side walls, notches or
teeth. `mechkit.py` (imported by `task.py`) builds the tray from 13 boxes (a floor, two side walls each with three
notches in its top, and a black gasket on each open end) and the comb from two (its bar and the plate standing in for
its teeth), each as one loose rigid body, through a hook on `yam_arena.scene`, and reports their poses (base-frame
origin and orientation) to the scorer. It stops the run with an error if either part was not built as a compound
body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read the arms it prints why once; the hands-clear
conditions are then taken as met on those steps, and the trace counts them (`arms_unavailable`).

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
  --instruction "Set the gel box up to cast a gel. Put the gel tray on the grey platform in the tank turned crosswise, so its two open ends, which carry the black gaskets, face the tank's two side walls and are sealed by them (not toward the electrodes). Then stand the comb in the comb slot nearest one end of the tray, across the tray with its bar down in the notches of both side walls. Let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/gel-tray-crosswise-to-cast/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set the gel box up to cast a gel. Put the gel tray on the grey platform in the tank turned crosswise, so its two open ends, which carry the black gaskets, face the tank's two side walls and are sealed by them (not toward the electrodes). Then stand the comb in the comb slot nearest one end of the tray, across the tray with its bar down in the notches of both side walls. Let go of both." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/gel-tray-crosswise-to-cast/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`zero_reason`, `tray_set`, `tray_pose_ok`, `tray_at_rest`,
`tray_moved_mm`, `tray_tilt_deg`, `cross_err_deg`, `corner_gap_mm`, `gasket_gap_mm`, `tray_centre_mm`,
`tray_overhang_mm`, `comb_slot`,
`comb_slot_pose`, `comb_reason`, `comb_at_rest`, `comb_moved_mm`, `comb_seat_x_mm`, `comb_seat_z_mm`,
`comb_mid_y_mm`, `comb_heading_deg`, `comb_tilt_deg`, `clear_of_tray`, `clear_of_comb`, `zone_gap_mm`,
`arms_unavailable`, both parts' origins and orientations, both TCPs, pad-tip points and openings). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every reset
(rigstate's and mechkit's reset counters, and a teleport of the tray back to its starting mark), so run one trial per
process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (the tests that load `task.py` need an
offline `yam_arena` stand-in and are skipped without it).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the tray's and the comb's poses and both
grippers once per control step (20 Hz). The tray's base frame is the middle of its underside (x along its length,
gasket to gasket; z up out of its open side); the comb's is the middle of its bar's underside (y along the bar; z up,
teeth below).

"Set to cast" is read from the tray alone: open side up and tilted 3 degrees or less, every corner of its underside
within 2 mm of the platform's top height and over the platform (within 55 mm of y 0: a tray hanging past the
platform's end is not on it, even if level), its length within 12 degrees of x (the line between the side walls), and
each gasketed end's face within 10 mm of the side wall it faces. The tank's side walls are 94 mm apart and the tray is
84 mm from gasket to gasket, so a crosswise tray down between the walls always has its two end gaps adding up to
10 mm (5 mm each when centred: the mock-up's rigid gaskets need that play for a robot to set the tray in); the 10 mm
limit says the tray is down between the walls, not perched on one. Between the walls the tray can turn at most about
10 degrees from crosswise, so every tray down between them passes the 12 degree limit; turned lengthwise (64 mm
across) it fits on the platform too, within about 25 degrees of lengthwise, and scores 0. Its two ends are alike (a
gasket on each, slots placed symmetrically), so either way round is the same tray.

"Seated" is read from the comb in the tray's frame: upright relative to the tray (within 5 degrees, teeth down), its
bar within 10 degrees of the tray's width, its middle within 6 mm of the tray's centre line (the 44 mm teeth between
the side walls, 54 mm apart inside), and, where the bar crosses each side wall, its underside within 1.5 mm of the
notch bottoms (13 mm above the tray's underside) and within 6 mm of the same slot's centre along the tray. The 12 mm
bar has 5 mm of play a side in the 22 mm notch (so a bar down in a notch always reads within 6 mm of its centre and
within about 9 degrees of the tray's width), and the slots are 25 mm apart; a bar resting on the walls' tops beside or
across a notch is 5 mm too high, and a comb rolled along its length by 4 degrees has one end 2 mm high: neither is seated.

Every score needs the tray at rest (no corner of its underside moved more than 2 mm in the last 0.5 s, measured on
positions: the simulator's reported velocities chatter while a part is gripped) and both grippers clear of it: each
gripper's finger zone at least 20 mm from the tray's box. The comb counts as seated only at rest (neither end of its
bar moved more than 2 mm in 0.5 s) with both grippers 20 mm clear of it too. rigstate gives no gripper heading, so a
gripper's fingers are taken to be anywhere round its axis: the zone is everything within reach of a pad's outer
corner (half the current finger gap plus the 8 mm pad thickness out, 13.5 mm across; 57 mm when fully open) around
the 38 mm stretch back from the point between the pad tips. A gripper holding the comb in a slot is within 20 mm of
the tray, so that scores 0, not 0.5: the score is read with both hands off the set-up. A new trial is recognised at
rigstate's or mechkit's reset, or when the simulator puts the tray back on its starting mark (within 4.5 cm of it,
which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number, a zero quaternion
or no base-frame origin, for either part, scores 0 for that step and leaves the scorer's record as it was.

| Outcome (at the end) | Score |
|---|---|
| Tray set to cast, at rest, both grippers clear; comb seated in an end slot (either end), at rest, both grippers clear of it (success once held 1 s) | 1.0 |
| Tray set to cast, at rest, both grippers clear; comb seated so in the middle slot | 0.75 |
| Tray set to cast, at rest, both grippers clear; comb not seated so (in its stand, held away from the tray, on the bench, lying in the tray, resting on the walls' tops, or in a slot but still moving or with a gripper within 20 mm of it) | 0.5 |
| Tray not set to cast (lengthwise with its gaskets toward the electrodes, on the bench, propped on a wall top, tilted, upside down, lifted off the platform, hanging past the platform's end), still moving, or a gripper within 20 mm of it (holding the tray, or holding the comb in it) | 0 |

Unintended strategies earn less, as the tests show: a lengthwise tray in the tank with the comb seated in an end slot
(0); the tray held down between the walls by a gripper (0); the comb held in its end slot (0); the comb laid on the
walls' tops over the end slot (0.5); the comb seated in the tray while the tray is still on the bench (0); the tray
upside down between the walls with the comb on it (0); the comb pushed down onto the tray's floor (0.5); unreadable
arms do not help a lengthwise tray (0).

## One arm alone

The demo uses both arms, one part each: the left arm grips the tray across its width where it lies lengthwise, lifts
it, turns the grip a quarter turn while carrying it over the tank and sets it down crosswise between the side walls;
the right arm lifts the comb out of its stand by the ends of its bar, waits aside, and once the left arm has let go
and moved off, stands the comb in the slot nearer the arms. The split comes from reach, not from a two-hand rule:
gripping straight down, the right arm cannot reach the tray where it starts (y +27 cm) and the left arm cannot reach
the comb (y -27 cm). Measured in simulation (L40S, 'Results in simulation' below): the left arm alone
(`DEMO_IDLE_ARM=right`) set the tray crosswise between the side walls and left the comb in its stand, 0.5; the right
arm alone (`DEMO_IDLE_ARM=left`) could only stand the comb on the platform of an empty tank (the tray stayed on the
bench), 0. A policy might still nudge a part toward the other arm with a tilted gripper, so no claim is made that one
arm cannot finish the task; what it tests is the casting set-up itself: the tray turned crosswise and set down between
the walls, and the comb in the right slot.

## Demo and run limit

The scripted demo takes 29.3 s (586 control steps). Left arm: it comes down round the tray's side walls (pad tips
5 mm above the table), grips it across its 64 mm width (fingers commanded 8 mm narrower, over 1.5 s), lifts it to
13 cm (gripper height), carries it over the tank at 11 cm in 2.5 s while turning the grip from yaw -90 to 0 degrees
(the tray swings from lengthwise to crosswise; the fingers, now closing along y on its side walls, are 33.5 mm clear
of the tank's side walls), comes down to just above the walls, lowers it the last 35 mm between the side walls over 2 s
to 0.5 mm above the platform, waits 0.5 s, opens over 1.5 s, goes straight up and moves aside. Right arm, at the same
time: it comes down round the ends of the comb's bar, grips the bar's 72 mm length (64 mm commanded), lifts the comb
straight out of its stand and waits aside at (20, -24) cm; at 19.0 s, when the left arm has let go and gone up, it
carries the comb over the slot 25 mm nearer the arms than the tray's middle (x 29.5 cm), comes down to 25 mm above
its seat, lowers it the last 25 mm into the notches over 2 s to 0.5 mm above their bottoms (its pad tips stay 3.5 mm
above the tray's walls, its fingers 8.5 mm from the tank's near side wall), waits 0.5 s, opens over 1.5 s, goes
straight up and moves aside. The arms' controller does not carry a part's weight: the compliance model (J K^-1 J^T of
the part's weight, joints 80/80/80/10/10/10 N m/rad, yamkin) puts the left gripper about (-0.4, +0.4, -1.2) mm off
under the 38 g tray and the right one about (-0.2, -0.2, -0.6) mm off under the 20 g comb at their place poses, far
inside the 5 mm of play at each end of the tray and each side of the comb's bar, so the place points are not
trimmed. The comb's fit stacks on the tray's: where the tray comes to rest along x moves the slot with it. Offline (no sag) both
parts are let go 0.5 mm above where they rest and success holds for its 1 s at step 588, once the right hand is 20 mm clear of
the comb. In simulation (L40S) success also held at step 588: the tray crosswise within 0.2 degrees with 3.0 and
7.0 mm gasket gaps, the comb's bar in the end slot nearer the arms, 1.8 mm from that slot's centre and down on the
notch bottoms. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the gel tray back on the table open side up
and lengthwise (its length along y), centred at x 18.0 cm, y +27.0 cm, and the comb back in its stand with its bar
along y and its teeth in the gap between the plates; the tank, platform, electrodes and stand are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 30 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A gel tank is a 106 x 240 mm open acrylic box 45 mm tall (6 mm walls, 4 mm floor) with a 94 x 110 x 12 mm acrylic
block glued in its middle as the platform and two 80 mm black and red bars on the floor near its ends as the
electrodes. The tray: a 80 x 64 x 4 mm acrylic floor with two 80 x 5 x 18 mm acrylic side walls glued along its long
edges, three 22 mm wide, 5 mm deep notches cut in each wall's top (at the middle and 25 mm either side), and a
2 mm thick, 14 mm tall strip of black rubber glued across each open end (about 38 g in all). The comb: a 72 x 12 x
18 mm black acrylic bar with a 44 x 3 x 7 mm plate glued under its middle (20 g). The stand: two 50 x 5 x 18 mm
white plastic plates glued to a base 6 mm apart. A commercial gasketed casting tray and comb of similar size can be
used instead if the tank is made to fit them with at most 5 mm of play at each end. Clamp the tank and stand to the
bench; nothing is powered and no agarose is poured. Score with a photo from above and a steel rule (gasket gaps,
which slot), with both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
