# Set two weld-neck flanges on their riser stubs, two-holed to the plant centrelines

Pipefitters set every flange on its pipe before it is tacked, and the flange is "two-holed": its bolt holes straddle
the plant's principal centrelines and never sit on them (unless the drawing asks for it), so that it mates with
valves, instruments and the matching flange however the far end is laid out. On a vertical riser those are the plant
north-south and east-west lines (Wermac, flange bolt hole orientation; the same convention as ASME B16.5 flanged
fittings). Bolt counts are multiples of four and smaller pipe carries fewer bolts (Class 150: NPS 2 has 4 bolts,
NPS 4 has 8), so what two-holing means depends on the flange: with equally spaced holes, a two-holed 4-hole flange
has its pattern 45 degrees off the lines and an 8-hole flange 22.5 degrees off. Here two black steel riser stubs
(36 and 50 mm across) stand on the table with the plant centrelines painted in white around each, and two loose
weld-neck flanges lie neck down nearer the arms: a 4-hole one bored for the small stub and an 8-hole one bored for
the large stub, both with their holes on the lines, as an untrained hand would set them. Each must go on the stub
it fits, neck down on the stub end, centred and two-holed, and be let go. The instruction says "two-hole it" and
which marks are the bolt holes; what that means for 4 and for 8 holes is the knowledge tested, and is not
described. The score is what a fitter checks with a rule and a square on the lines: 0.25 per flange seated on its
own stub and 0.25 more per seated flange that is two-holed.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the flanges' poses and the grippers, with a small tracker of the last second |
| `mechkit.py` | Helper that builds each flange as one rigid body (neck, disc and painted face) and reports its pose |
| `rigstate.py` | Helper that reads both grippers' pad-tip points and openings for the 'let go' test |
| `tests/test_scoring.py` | Offline tests of every scoring tier and window edge, every zero condition, the exploits we tried, resets, invalid states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the stubs and painted
lines are `yam_arena`'s own fixed cylinders and boxes, the scorer is the standard `task_score`, and two helper
modules sit next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain disc shows no bolt
holes and no neck. `mechkit.py` (imported by `task.py`) builds each flange from cylinders (a neck under a disc, and
thin black discs on its face for the bore and the bolt holes) as one loose rigid body through a hook on
`yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the scorer. `rigstate.py` wraps the
embodiment's step and reset to read each gripper's pad-tip point and opening before every scoring call, only to check
that each flange was let go. Neither changes anything else the embodiment does, and mechkit stops the run with an
error if a flange was not built as a compound body. The techniques (a `task_file.Box` subclass the loader accepts, a
`sys.meta_path` hook patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner,
wrapping the embodiment's step and reset) follow those first shown in the program repository's `tasks/bearing-press`;
both helpers are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit-up on the two riser stubs: set each weld-neck flange on the stub it is bored for, neck down on the stub end and centred, and two-hole it to the plant centrelines painted on the floor around the stub. The small black circles on each flange's face are its bolt holes. Then let go of both flanges and move the grippers clear." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/flanges-two-holed-on-risers/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit-up on the two riser stubs: set each weld-neck flange on the stub it is bored for, neck down on the stub end and centred, and two-hole it to the plant centrelines painted on the floor around the stub. The small black circles on each flange's face are its bolt holes. Then let go of both flanges and move the grippers clear." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/flanges-two-holed-on-risers/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `zero_reason`, `moved_mm`, `seated_count`,
`two_holed_count`, `readable_arms`, `arms_unavailable`; per flange `centre_mm` (its axis from its own stub's axis),
`seat_gap_mm` (its neck's bottom above the stub end), `tilt_deg`, `pattern_yaw_deg` (hole 1's heading from +x),
`straddle_off_deg` (0 when exactly two-holed), `on_own_stub`, `on_other_stub`, `nearest_stub`, `held_by`,
`at_rest`, `seated`, `two_holed`; each flange's origin, orientation and hole-1 point; both pad-tip points and
openings). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it
over at every reset (rigstate's and mechkit's reset signals, or both flanges teleported back to their starting
marks), so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads both flanges' poses and both grippers once
per control step (20 Hz). Each flange's base frame sits on its axis at the bottom of its neck, local z up the axis
and local x through bolt hole 1. A flange is seated when it is on the stub it is bored for (the 4-hole flange on the
36 mm stub, the 8-hole flange on the 50 mm stub), its axis within 6 mm of the stub's axis seen from above, the centre
of its neck's bottom within 3 mm of the stub end's height, face up and within 5 degrees of level, still (over the last
1 s neither its neck's bottom centre nor its hole-1 centre moved more than 2 mm, measured on positions: the
simulator's reported velocities chatter while a part is gripped), and let go. rigstate gives no gripper heading, so a
gripper counts as possibly holding, pressing or propping a flange when its pad-tip point is lower than 10 mm above
the flange's highest point (its painted face when level) while the flange's axis is within a finger's reach of it at any yaw (half the finger gap plus 10 mm
out and 20 mm across, plus the flange's radius and 5 mm; about 95 mm for the 4-hole flange and 100 mm for the 8-hole one, fully open). An arm that cannot be read is
skipped and counted (`arms_unavailable`); the other arm is still checked.

A seated flange is two-holed when its pattern straddles the centrelines: psi, the heading of the line from its axis
to any hole seen from above (0 along the north-south line), taken modulo the hole pitch (90 degrees for 4 holes, 45
for 8), is within 8 degrees of 45 for the 4-hole flange (37 to 53) and within 7 degrees of 22.5 for the 8-hole flange
(15.5 to 29.5). The two lines are a quarter turn apart and the patterns repeat every pitch, so any equivalent heading
counts; a hole on a line (psi modulo the pitch near 0) does not.

| Outcome (s seated, t of them two-holed) | Score |
|---|---|
| s 2, t 2: both flanges on their own stubs, centred, two-holed, still and let go (success once held 1 s) | 1.0 |
| s 2, t 1: one pattern off (e.g. both turned 45 degrees: the 8-hole flange's holes are then back on the lines) | 0.75 |
| s 2, t 0 (both seated with their holes on the lines, as they start); or s 1, t 1 | 0.5 |
| s 1, t 0 | 0.25 |
| s 0: nothing moved; flanges swapped onto the wrong stubs; on the table; face down or on a rim; off centre; still held or moving | 0 |

Score = 0.25 s + 0.25 t, so every outcome maps to exactly one score; each tier is reached in the tests by a concrete
layout, and 500 random layouts are checked against a second scorer written from the task's text. Unintended
strategies earn nothing more than they should, as the tests show: the flanges set with their holes on the lines (the
neat-looking answer, 0.5); the 4-hole answer copied onto the 8-hole flange (0.75) or the 8-hole answer onto the
4-hole one (0.75); the flanges swapped onto each other's stubs, turned correctly (0); one flange still in the gripper
(only the other counts); an open gripper hovering on a flange (that flange 0); a flange stacked on the other flange,
pushed against its stub on the table, leaning on its stub end, face down or on its rim (0 for that flange); a flange
turning in place (not still). Pushing cannot seat a flange: the stub ends are 60 mm above the table. A new trial is
recognised at rigstate's or mechkit's reset, or when the simulator puts both flanges back on their starting marks
(within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number, a
zero quaternion or no base-frame origin scores 0 for that step and leaves the scorer's record as it was.

## One arm alone

Each flange starts on its own arm's side and goes to the stub on that side, and the demo uses one arm per flange, at
the same time. The scripted one-arm runs (`DEMO_IDLE_ARM=left`, `DEMO_IDLE_ARM=right`) are predicted offline to score
0.5 each: the working arm's flange seated and two-holed, the idle arm's flange left on the table (GPU results to
follow). Gripping straight down, neither arm reaches the far stub: at the stub's place height the left arm reaches no
point over the 50 mm stub at any yaw, and the right arm none over the 36 mm stub (yamkin, joint limits). Each arm does
reach the other arm's flange where it starts, straight down. Leaned 30 to 45 degrees toward the far side, though, our
kinematics put the TCP over the far stub (no collision or grasp check), so one arm with a steeply leaned grip could
in principle seat both flanges: the task does not claim to need two arms. Its natural division of work is one arm per
flange, both at once.

## Demo and run limit

The scripted demo takes 19.0 s (380 control steps), both arms at once. Each arm comes down around its flange with the
fingers fully open (94 mm), grips it across the disc's rim (fingers commanded 8 mm narrower than the disc, over 1.5
s; the tool point 20 mm above the neck's bottom, pad tips 7 mm above the table, 3 mm under the disc and outside the
neck), lifts it to carry height (neck's bottom 100 mm up, 40 mm over the stub ends) and carries it over its stub,
turning the grip on the way in two equal halves. The left arm turns the 4-hole flange's pattern 45 degrees clockwise
(its pattern repeats every 90 degrees, and this keeps the wrist within 22 degrees of its bearing to the stub); the
right arm turns the 8-hole flange's pattern 22.5 degrees counter-clockwise. Both are commanded 1 degree clockwise of
the answer (`YAW_TRIM`), since in our earlier simulator runs gripped parts ended turned 0.6 to 1.7 degrees
counter-clockwise of the commanded yaw. Each place point leads its stub's axis by 4 mm along the carry (`LEAD`),
since carried parts landed about 4 mm short there: a flange landing anywhere from 0 to 9 mm short is within the 6 mm
centring window. Each arm settles 0.8 s over its stub, lowers the flange the last 3 cm over 2 s to aim its neck 1 mm
above the stub end, settles 0.5 s, opens over 1.5 s, goes straight up (pad tips about 65 mm above the face) and aside. The
gripping pads are 30 and 35 mm from the flange's axis, outside the stubs (18 and 25 mm). Offline: worst IK error
0.083 mm, joint limits at least 10.9 degrees away (joint 4), the tool points at least 26 cm apart and the arms' links
at least 6.7 cm apart; each flange drops 1 mm onto its stub end, 4 mm off its axis, its pattern 1 degree short of
the answer; success has held for its 1 s at step 354 and the scripted policy stops at step 441. If a flange lands
off on the GPU, read `centre_mm` and `straddle_off_deg` in `TASK_TRACE` and adjust `TRIM` or `YAW_TRIM` in
`task.py`. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the 4-hole flange back neck down on the table
centred at x 21.0 cm, y +13.0 cm, and the 8-hole flange at x 21.0 cm, y -13.0 cm, each with one bolt hole pointing
along +x (plant north), so all their holes lie on the centrelines' directions. The stubs and painted lines are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 19 s of simulated time); the first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

Two short pieces of black steel pipe (or turned bar) 36 and 50 mm across and 60 mm tall, glued upright to the bench
30 cm apart; around each, two strips of 4 mm white tape crossing at its axis, along and across the bench, 80 mm out
each way, with a small white square at the far end of the lengthwise strip for plant north. Two aluminium model
flanges, turned or 3D-printed and weighted: a 60 mm disc 12 mm thick on a 36 mm neck 10 mm tall (120 g) with four
8 mm black dots on a 46 mm circle, and a 70 mm disc on a 50 mm neck (180 g) with eight 7 mm dots on a 56 mm circle,
each with a dark bore disc painted in the middle of its face. Score with a rule from each stub's axis, a square or
protractor laid on the white lines, and a photo from above showing both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
