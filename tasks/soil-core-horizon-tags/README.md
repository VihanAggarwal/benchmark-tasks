# Tag every horizon boundary along an opened soil core

Before a soil core is sampled, someone logs it. The core is split open and laid in a core tray, and the person
logging it finds each boundary between horizons (layers that differ in colour and texture), marks it with a tag,
nail or pen tick, and records its depth from the ground surface. Here an opened 560 mm core lies fixed in a flat
black tray, painted in five horizons (A, E, Bt, BC, C) with four boundaries at 46, 173, 314 and 434 mm deep, and
four white depth tags stand on the table, a pair at each end of the tray. The task is to stand one tag upright on
the tray beside the core at each boundary, let go and lift the grippers clear. Each boundary with a tag within
10 mm of it scores 0.25. The boundaries are irregular, so they have to be seen: tags at the points dividing the
core or the tray into 2 to 6 equal parts, or every 10 or 20 cm from either end of the core, score 0, and no blind
spacing we tried scores more than 0.5 (see 'Blind layouts' below).

## The practice

- **The work.** A soil-survey core taken with a hydraulic probe (or a direct-push core in an environmental site
  assessment) is split open and laid in a core tray. The person logging it finds each boundary between horizons,
  marks it with a tag, nail or tick, and records its depth from the ground surface.
- **Why it matters.** The boring log, the choice of sample intervals (a contaminant or lab sample comes from within
  one layer, never across a boundary) and the site interpretation all depend on these marks.
- **Who does it.** Soil scientists, environmental science and protection technicians and geoscientists (O*NET
  19-1013, 19-4042, 19-2042), following the USDA NRCS Field Book for Describing and Sampling Soils and ASTM D5434
  (field logging of subsurface explorations).
- **How it is checked.** A supervisor checks a logged core with a ruler from each mark to the colour change, and
  that is how this task is scored. Depths are recorded to the nearest centimetre, which matches the 10 mm tolerance.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the tags' states and the grippers' pad-tip points and openings, with a small tracker |
| `rigstate.py` | Reads both grippers (tool point, pad-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, blind layouts and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, loose cylinders) and the standard `task_score` scorer.

The release rule needs the grippers' state, which `yam_arena` does not pass to a task file's hooks. `rigstate.py`
(imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of both arms
(tool point, pad-tip point, finger opening, in the objects' frame) before each control step and to start a new
trial at each reset. It reads only state the simulator already has and changes nothing the embodiment does. If it
cannot read the arms it prints why once; the release rule is then taken as met on those steps, and the trace
counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This soil core lies open in the black tray. Stand one white tag upright on the tray right next to the core at each place where one soil layer meets the next, one tag per boundary. Then let go and lift the grippers clear." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/soil-core-horizon-tags/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This soil core lies open in the black tray. Stand one white tag upright on the tray right next to the core at each place where one soil layer meets the next, one tag per boundary. Then let go and lift the grippers clear." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/soil-core-horizon-tags/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: for each tag `x_mm`, `y_mm`, `tilt_deg`, `bottom_gap_mm`,
`upright`, `on_tray`, `side`, `path_mm`, `at_rest`, `held`, `tip_xy_mm`, `nearest`, `off_mm` and `placed`, plus
`tagged`, `arms_unavailable`, each gripper's finger reach `reach_mm` and both pad-tip points and openings. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step (20 Hz) the scorer reads the four tags
and, for both grippers, the pad-tip point (midway between the finger tips) and how far it is open. A tag is
**placed** when it is (a) upright, its axis within 10 degrees of vertical (either end down); (b) on the tray, its
lowest point within 3 mm of the tray's top, so not on the core, on another tag or on the table; (c) still, its
centre's path over the last 0.5 s (10 control steps) under 3 mm (the path, never the velocity, which chatters at
10-45 mm/s while a part is gripped); (d) beside the core, its centre outside the core and at most 45 mm from the
nearer long edge (x 0.340-0.385 m or 0.435-0.480 m; a tag pressed against the core, centre 10 mm from the edge,
counts); and (e) let go and clear: no gripper's pad-tip point lower than 10 mm above its top (56 mm above the table
for a tag on the tray) while the tag's axis is within that gripper's finger reach horizontally. `rigstate.py` does
not report a gripper's yaw, so the reach is the widest over every yaw at the gripper's current opening: hypot(half
the finger gap + 10 mm finger thickness, 20 mm half finger width) + 10 mm tag radius + 5 mm, which is 75 mm fully
open, 43 mm closed on a tag and 37 mm fully closed. An open finger touching or propping a tag from any side
therefore counts as holding it, and the neighbouring boundaries (120 mm or more apart) are always out of reach of a
gripper working at one of them. A boundary is **tagged** when a placed tag's centre is within 10 mm of it along the
core. The boundaries are at least 120 mm apart, so one tag can tag at most one, and each counts once however many
tags are near it.

| Outcome | Score |
|---|---|
| All four boundaries tagged (held for 1 s, which ends the trial) | 1.0 |
| Three tagged (for example one tag 15 mm off its boundary) | 0.75 |
| Two tagged (for example only one arm's two tags placed) | 0.5 |
| One tagged (a single correct tag, or two tags at the same boundary) | 0.25 |
| None: tags at their start, lying down, tilted over 10 degrees, stacked, on the core, on the table, still held, or more than 10 mm from every boundary | 0 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0. Without a 1 s success the score is the last step's. A new trial is
recognised from rigstate's reset hook, or, as a fallback, when the simulator puts all four tags back on their
marks in a single step. A step on which any tag's state has a non-finite number or a zero quaternion scores 0 and
leaves the tracker as it was. The core and the tray are fixed, so nothing can be moved to bring a boundary to a
tag; there are no whole-trial rules.

Blind layouts (tests `BlindLayouts`), measured in mm down the core from its top end, with the tray running from
-30 (its top end) to 690 (its bottom end) on the same scale. A rule with more points than there are tags is
followed with any four consecutive points; "with error" means each tag up to 3 mm off its rule's point.

- **These rules score 0.** For the core and for the tray: the points dividing it into 2 to 6 equal parts, the
  centres of 1 to 4 equal parts, and 2 to 5 points spread from end to end; a tag every 10 or 20 cm from either end
  of the core or from the tray's bottom end; and the ends. Every one of these points is at least 13 mm from every
  boundary (the closest: 160 mm, the 10 and 20 cm grids from the core's bottom, against B2 at 173), so they score
  0 even with an error under 3 mm.
- **These tag one boundary (0.25).** The centres of 5 equal parts of the core (56, 168, 280, 392, 504: 168 is 5 mm
  from B2, and 56 is exactly 10 mm from B1, on the tolerance, so with error 0.5) and of the tray (42, 186, 330,
  474, 618: 42 is 4 mm from B1 and 186 is 13 mm from B2, so 0.5 with an error of 3 mm); a tag every 10 or 20 cm
  from the tray's top end (-30, 70, 170, 270, ...: 170 is 3 mm from B2 and every other point is 24 mm or more from
  a boundary, so 0.25 even with error).
- **Denser rules reach at most two boundaries (0.5).** The centres of 6 to 12 equal parts, the points dividing it
  into 7 to 12 parts and 6 to 12 points end to end, of the core or the tray, and a tag every 5 cm from any end of
  either. They have more points than there are tags, and choosing which four to use would take seeing the
  boundaries. Any four consecutive points come within 13 mm of at most two boundaries (so at most 0.5 even with
  error), and even all of a rule's points together come within 10 mm of at most two.
- **Why the boundaries are where they are.** A first layout (65, 171, 318 and 483 mm deep in a 660 mm tray) let
  four tags at the centres of four equal parts of the core (70, 210, 350, 490 mm) tag B1 and B4 for 0.5. The
  boundaries were moved and the tray lengthened until no rule of the first list comes within 13 mm of any of them.
  B2 at 173 mm is the only whole-millimetre depth 120 mm or more from both B1 and B3 that keeps that margin (160
  and 186.7 mm are both rule points), which is why the 5-part centre at 168 mm and the tray's 10 cm grid at 170 mm
  reach it. Moving B1 shallower to clear the 5-part centre at 56 mm (anywhere from 13 to 45 mm) lets the points of
  some denser rule reach three boundaries, so B1 stays at 46 mm.

Other unintended strategies and what they earn, as the tests show:

| Strategy | Result |
|---|---|
| Hold a tag at a boundary, or press it down from above | that tag counts only once every gripper is out of reach of it (75 mm open, 43 mm closed on it) or 10 mm or more above its top |
| Prop a tag with one open finger (pad-tip point 37 mm off its axis, inside face touching; or 67 mm, outside face) | held: that tag does not count |
| Stand a tag on the core, stack one on another, or lay it down | that tag counts nowhere (the lower of two stacked tags still counts if it is placed) |
| Two tags at one boundary, or one tag between two | the boundary counts once; one tag can never cover two (120 mm apart against 10 mm) |
| Tags at the boundary depths but out at the tray's front edge (centres 55 mm from the core) | 0 |
| A tag dropped from a few centimetres that lands upright at its boundary | counts once it is still: it is still a correct mark |
| A tag leaning on the core at 8 degrees / 12 degrees | counts / does not count |
| A slow creep under 3 mm per 0.5 s, or a still tag whose reported velocity chatters | counts; a slide of 2 mm per step counts only once it has stopped |
| A gripper that cannot be read | rule (e) taken as met, counted in `arms_unavailable` |

## One arm alone

At most 0.5. The tags start in a pair at each end of the tray, as on a long logging bench with a box of tags kept
at each end, so each arm can reach only its own two: they start 0.40 m from the midline, against a reach of about
0.10 m past it, even with tilted grips. With a straight-down grip the left arm can reach B1, B2 and B3 (B3
only low down, not at the 10 cm travel height) and the right arm only B3 and B4 (B2 is 0.11 m past the midline at
x 0.36, out of its reach). So the second arm is needed for the other two tags (and for B4 or B1), not for any
coordination: in the demo the two arms work at the same time, each on its own half of the core. In simulation the
scripted demo with the left arm idle scored 0.5 (the right arm tagged B3 and B4) and with the right arm idle 0.5
(the left arm tagged B1 and B2), as the offline kinematic check predicted. Those runs play each arm's own half of
the demo; the 'at most' comes from the reach limits above. The task does not claim more: the two halves need no
coordination, only the second arm's reach.

## Seeing the boundaries

At the program's 320 x 176 camera size the top camera sees about 2 pixels per centimetre (on the stored frames of
the simulation run, the Bt and BC horizons, 261 mm together, span about 49 pixels), so the 10 mm tolerance
is only about 2 pixels there and the 46 mm A horizon about 9. The wrist camera is the one to place by: with the
gripper over the core at or below the 10 cm carry height it is roughly 10 to 20 cm from the core, so with a
D405-like field of view (about 87 degrees across 320 pixels) it sees about 0.6 to 1.2 mm per pixel (the 10 mm
tolerance is 8 to 17 pixels), and each colour change can be located to well under 1 cm by sight. That matches
the field practice of logging boundary depths to the nearest centimetre. (An estimate from the camera's
specification, not a measurement: the simulation run stored no wrist-camera frames.) The five colours differ
clearly (dark brown, light grey, reddish brown, yellowish brown, olive grey), and the white tags stand out against
the black tray.

## Demo and run limit

The scripted demo (`demo()`) grips straight down with yaw 0, so the fingers close along y (along the core) and
open away from the tag without touching the core. Each move takes 15 s: over the tag at a TCP height of 10 cm
(2.5 s), down to 25 mm, pad tips 12 mm above the table (1.5 s), settle 0.5 s, close to 12 mm (8 mm narrower than the
tag) over 1.5 s, lift to 10 cm (1.5 s), carry over the place point (2.5 s), lower slowly to 32 mm, the tag's
bottom 1 mm above the tray (2.0 s), settle 0.5 s, open fully over 1.5 s, and straight up to 10 cm (1.0 s), where
the pad tips are 87 mm above the table, clear of the release rule. The place point is x 0.360 m: the tag's centre
25 mm in front of the core's near edge and the 27 mm wide pads 11 mm clear of it. The left arm stands its first tag
at B1 (0-15 s) and its second at B2 (15-30 s); the right arm, at the same time, its first at B3 and its second at
B4; each then holds 10 cm above its last tag. Whenever both grippers are at the same x they are at least 0.26 m
apart in y, so neither waits; no carry passes over the core, and a carried tag clears the tags already standing
(tops 46 mm above the table) by 29 mm. Travel is at 10 cm, not 12 cm: at 12 cm the right arm's hover over B3, near
the midline, would leave joint4 only 0.12 rad from its limit (0.22 rad at 10 cm).

Offline (kinematic proxy, no physics) every tag drops 1 mm onto the tray and success has held its 1 s at step 610
(30.5 s); the plan is 600 steps and the scripted policy would stop at step 661. IK errors stay under 0.1 mm; the
smallest joint margin is joint4 of the right arm, 0.22 rad, over B3. The run limit is 1,800 steps (90 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the four tags back upright on the
table at (0.200, +0.400), (0.280, +0.400), (0.200, -0.400) and (0.280, -0.400) m; the tray and core are fixed.
Nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 31 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A flat 720 x 180 mm black plastic board (6 mm) as the core tray, taped to the bench; a 560 x 50 x 30 mm wooden or
printed bar painted in the five bands at the stated depths (matte paint, sharp edges between colours) and taped
along the tray's middle, 30 mm from its left end; and four 40 mm lengths of 20 mm white nylon rod (about 15 g
each) as tags. Lay a ruler along the tray beside the core for the top photo; after the arms withdraw, measure from
each standing tag's centre to the nearest colour change.

## Fidelity

Horizons are crisp colour changes between rigid painted blocks; real boundaries can be gradual or wavy, and are
also judged by texture, structure and feel, which the simulator does not show. The core is a flat-topped bar, not
a half-round core in a cut liner. Tags stand on the tray beside the core; on a real core, nails or flags are pushed
into the soil, or the liner is ticked. Depths are not written down; only the marks are scored. One friction value
applies to everything. The simulator reads the grippers through `rigstate.py` only to check that each tag has
been let go.

## Nearest board tasks and how this differs

- TK-YTGSTJ (mark a cable into equal thirds along its route): two markers at computed fractions of a cable's
  length. Here there are four irregular positions that must be seen, and even spreads of the core score 0.
- TK-TEB4TV (continue a harness standoff run at the same pitch): continues a visible pitch. A soil core has none.
- TK-PDUF4Z (split a lamp cord apart to the white band): a different act that stops at one band.
- TK-BFDQWD and TK-Q6MY8Z (tag the gauge in its red zone; tag the stuck valve): one tag on one discrete item chosen
  by reading. Here tags go at positions along one continuous object, judged in millimetres from each colour change.
- TK-MKC8DD (contain a suspect lot back to the last good check): the boundary is already tagged and the act is
  moving parts into a bin.
- No board task, and none in LIBERO, RoboCasa, SimplerEnv, BEHAVIOR, DataCenterBench or RoboDojo, finds and marks
  several visible boundaries along one object. Board searches for soil, horizon, stratigraphy, core tray, boring
  log and lithology return nothing relevant.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

Every run's trace shows `arms_unavailable` = 0, so the release rule (e) was checked on every step. Settling (no
plan played, 160 steps): no tag moved or tilted. Demo: all four boundaries were tagged from about step 596 and the
trial ended as a success at step 611 (30.6 s); every tag stood upright within 0.6 mm of its boundary along the
core, its centre 25 mm in front of the core's near edge, and no tag tilted more than 1.8 degrees on the way.
Dream-1 ran the full 1,800 steps and moved no tag.
