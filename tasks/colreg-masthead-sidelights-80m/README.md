# Site an 80 m ship's masthead lights and sidelights where COLREG Annex I requires

Marine electricians and naval architects (O*NET 47-2111.00, 17-2121.00) site a new ship's navigation lights to the
International Regulations for Preventing Collisions at Sea (COLREG 1972; in US law 33 CFR parts 83 and 84), and a
surveyor checks the result on the deck drawing with a tape. For a power-driven vessel the plan rules are:

- Rule 23(a): two white masthead lights, "a second masthead light abaft of and higher than the forward one", which
  a vessel of less than 50 m need not show; an 80 m ship must show both. Rule 21(a): masthead lights are "placed
  over the fore and aft centreline".
- Annex I section 3(a): "the horizontal distance between them shall not be less than one-half of the length of the
  vessel but need not be more than 100 metres. The forward light shall be placed not more than one-quarter of the
  length of the vessel from the stem."
- Annex I section 3(b): "On a power-driven vessel of 20 metres or more in length the sidelights shall not be placed
  in front of the forward masthead lights. They shall be placed at or near the side of the vessel." Rule 21(b): green
  to starboard, red to port. IMO MSC.1/Circ.1260/Rev.1 (the unified interpretation) reads "near the side" as "not
  more than 10 per cent of the breadth of the vessel inboard from the side, up to a maximum of 1 metre".

Here the ship is a 1:100 deck model of an 80 m, 13 m beam cargo ship: a grey deck plate 800 x 130 mm lying along
y in front of both arms, its pointed bow toward +y (the left arm), its square transom toward -y, a white centreline
along it and the white sternlight already fitted at the stern. Two white masthead light units (upright cylinders)
and a red and a green sidelight unit (cubes) stand on the table in front of it. The robot has to stand all four on
the deck where the rules put them and let go. Nothing on the deck marks a place and the instruction gives no
distance, side or order: the spacing has to come from the ship's length (a quarter length is 200 mm, half a length
400 mm), the side band from the beam (1 m, the IMO cap, is 10 mm), and which side is port from which way the bow
points (facing the bow with z up, starboard is +x, the side away from the arms). Each correct light earns 0.25.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the rule's arithmetic checked at import |
| `scoring.py` | The scorer: plain functions of the four units' states (and the arms' pad-tip points and openings, for 'let go'), with a small tracker for stillness and the trace |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does); the same file as in our other tasks |
| `tests/test_scoring.py` | Offline tests of the arithmetic, every tier and band edge, the gate, every zero condition, the let-go rule, broken states, resets, the shortcuts we tried, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes and cylinders) and the standard `task_score` scorer;
`rigstate.py`, in this folder, reads the arms only to check that each unit was let go.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This is a 1:100 deck model of a new 80 m power-driven cargo ship. Its bow (the pointed end) points toward the left arm and its stern toward the right arm; the white sternlight is already fitted. Stand the two white masthead light units and the red and green sidelight units upright on the deck where COLREG 1972, the international rules of the road at sea, requires them for this ship, then let go. Only where each light stands on the deck is judged, not heights, screens or arcs." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/colreg-masthead-sidelights-80m/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This is a 1:100 deck model of a new 80 m power-driven cargo ship. Its bow (the pointed end) points toward the left arm and its stern toward the right arm; the white sternlight is already fitted. Stand the two white masthead light units and the red and green sidelight units upright on the deck where COLREG 1972, the international rules of the road at sea, requires them for this ship, then let go. Only where each light stands on the deck is judged, not heights, screens or arcs." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/colreg-masthead-sidelights-80m/dream
```

Run one trial per process (the default). The score depends only on where the units stand, whether each is still
and whether a gripper may still be touching one; the small tracker in the task module keeps each unit's path over
the last second (for stillness) and diagnostics for the trace, and starts a new episode when rigstate sees the
embodiment reset or the simulator puts every unit back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: `score`, `F` and `A` (which white unit is which), `gate`,
`F_ok`, `A_ok`, `red_ok`, `green_ok`, `F_from_stem_mm`, `separation_mm`, `paths_mm` (each unit's path in the last
two 0.5 s windows), `first_step` (the first step each score was reached), `best_score`, `bad_steps`, each arm's
`left_tips`/`right_tips` and `left_opening`/`right_opening`, `arms_readable` and `arms_unavailable` (or
`rigstate_error`), and for each unit under `units`: `x_mm`, `y_mm`, `off_centreline_mm`, `tilt_deg`, `bottom_mm`,
`on_deck`, `port_gap_mm`, `stbd_gap_mm`, `held`, `still` and `stands`. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four units' poses once per control step
(20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the finger
tips) and how far it is open, and scores the state at hand: the score is read at the last step. Positions are read
in plan, as a tape on the deck or a top photo with a scale reads them once the arms have withdrawn: a masthead unit
by the centre of its base, a sidelight by its footprint.

A unit stands when all of these hold:

- upright: its axis (a cube: the face normal nearest the vertical, so a cube on any face is upright) within 10
  degrees of vertical;
- on the deck: its lowest point (worked out from its pose) within 3 mm of the deck top, and its whole footprint,
  seen from above, inside the deck outline (the straight-sided outline in `task.json`'s setup);
- let go: no gripper's pad-tip point is less than 10 mm above its top while within that gripper's finger reach of
  its centre. The gripper's yaw is not read, so the reach is the widest over every yaw: hypot(half the finger gap +
  10 mm, 20 mm) + the unit's own reach (10 mm for a cylinder, 14.1 mm for a cube) + 5 mm, which is 75.4 / 79.5 mm
  fully open and 43.3 / 47.4 mm closed on a unit. Each gripper is read on its own; one that cannot be read is
  skipped and counted in the trace's `arms_unavailable`;
- still: its centre's path under 1.5 mm in each of the last two 0.5 s windows (read from the path, not from the
  reported velocity, which chatters while a part is gripped).

F is the standing white unit nearer the bow (larger y), A the other standing white unit (either white unit may be
either light). Each part is worth 0.25:

| Part | Correct when | Points |
|---|---|---|
| F | within 6 mm of the centreline (x 35.0 cm) and at most 200 mm (20 m, a quarter length) from the stem: y at least +20.0 cm | 0.25 |
| Gate | A, red and green are judged only when F stands within 6 mm of the centreline (whether or not it is far enough forward); otherwise they score 0 | |
| A | within 6 mm of the centreline and at least 400 mm (40 m, half a length) abaft F | 0.25 |
| Red | its centre on the port half (x under 35.0 cm), its footprint at most 10 mm (1 m) inboard of the port side (and on the deck), its centre not nearer the bow than F's | 0.25 |
| Green | the same on the starboard half (x over 35.0 cm), at the starboard side | 0.25 |

Score = the sum: 0, 0.25, 0.5, 0.75 or 1.0, and every state has one score. Every outcome, with numbers: the demo's
F at y +28.0 cm and A at -18.0 cm on the centreline (460 mm apart), red at x 30.0 and green at x 40.0 cm (each 5
mm inboard, the middle of the band) at y -8.0 cm, 1.0; the small-craft habit of the sidelights at the bow ahead of
the masthead light (y +32.0 cm with F at +28.0) and no second masthead light, 0.25; mastheads 30 m (300 mm) apart,
the rest right, 0.75; F 225 mm (22.5 m) from the stem, the rest right, 0.75; a single masthead light at midships
(y 0) with the other left on the table and the sidelights at y -8.0 cm, 0.5 (F is too far aft and there is no A,
but the sidelights abaft it still count); colours swapped (red to starboard), 0.5; sidelights midway between the
centreline and the sides (x 31.75 / 38.25 cm, 22.5 mm inboard), 0.5; F 10 mm off the centreline, 0 in all (the
gate); nothing moved, 0. A unit lying on its side, stacked on another, perched on the sternlight, overhanging the
deck edge, on the table beside the hull, or still held does not stand: a masthead unit that does not stand cannot be
F or A, and a sidelight that does not stand earns nothing.

The forward limit for F is set by the deck itself: a 20 mm unit fits on the centreline up to y +37.3 cm, where the
bow's tapered edges come within 10 mm of its axis. The trial ends early once 1.0 has held for 2 s (`SUCCESS_HOLD_S`); otherwise it
is scored as it stands at the run limit. Every credit needs the unit let go and still for 1 s, and the 1 s plus the
2 s hold is longer than the 0.5 s settle and 1.5 s release after each set-down, so even on a step where the arms
could not be read the trial cannot end with the fingers closed on a unit. A state with a non-finite number or a
zero quaternion (a simulator blow-up) scores 0 for that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: the coloured cubes stood on the centreline as masthead
lights (the mastheads are the white units: 0); both white units together near the bow (A fails: 0.75); a unit on the
sternlight or overhanging the side to sit right at the lamp position (it does not stand); sidelights on the table
against the hull (0.5); a red cube at the starboard side (red fails whatever the band); an open gripper left low
round a placed unit (every unit within its reach counts as held); 400 random layouts score only 0, 0.25, 0.5, 0.75
or 1.0, and never credit A or a sidelight with the gate closed.

## One arm alone

The demo splits the work by reach: the left arm (+y, toward the bow) sets the forward masthead light F, the right
arm sets the after masthead light A and both sidelights abreast at y -8.0 cm, where a cargo ship's bridge wings
would be. With one arm idle, the scripted demo scores (offline prediction, confirmed by the GPU one-arm runs):

- left arm idle: 0.0. The right arm sets A, red and green, but the other white unit stays on the table, so the
  unit at y -18.0 cm becomes F: it is too far aft, there is no A, and both sidelights are now ahead of it.
- right arm idle: 0.25. F alone.

What one arm could do in principle, by our IK model of the arm (`tests/test_scoring.py`, `OneArmReach`), gripping
straight down at any heading: the right arm cannot reach the centreline anywhere in the forward quarter (it reaches
the centreline only to about y +7 cm, and F needs y +20.0 cm or more), so the right arm alone cannot earn F; its
ceiling is 0.75 (F on the centreline further aft, A half a length abaft it, both sidelights abaft F). The left arm
reaches every unit's start and the centreline from about y -7 cm to the bow, so alone it could score 1.0, but only
by setting F within a few millimetres of the deck's forward limit (y +37.2 cm) and A just inside its own reach (y
-3.1 cm), with the sidelights at y +10 cm: a tight layout at the edge of its reach. We do not claim the task needs two
arms; the two-arm plan is the natural one.

## Demo and run limit

The scripted demo takes 53.2 s (1,064 control steps), both arms on one clock. Each unit is gripped straight down at
yaw 0 (fingers closing along y, so they are never outboard of the deck's sides), opened to 40 mm (10 mm either side
of a 20 mm unit; neighbours on the table stand 60 mm apart) and commanded to 12 mm to close, with the tool point
17 mm above the unit's bottom so the pad tips stay 4 mm above the table. Every gripper open or close takes 1.5 s,
every final descent is slow (20 mm over 1.2 or 1.5 s), each unit is carried with its bottom 53 mm above the table (23
mm over a standing unit's top, 28 mm over the sternlight), its last 40 mm of travel runs along y only (so the
3 to 7 mm an arm trails under load lands it short in y, where every margin is large, not in x), and it is let go
1 mm above the deck after a 0.5 s settle, at a commanded yaw of -1.15 degrees (gripped parts end 0.6 to 1.7 degrees
counter-clockwise of the command). In order:

1. The left arm sets the white unit from x 23.0, y +15.0 cm as F at x 35.0, y +28.0 cm (120 mm, 12 m, from the
   stem: 80 mm inside the forward quarter) and parks back toward its base; at the same time the right arm sets the
   white unit from y -9.0 cm as A at x 35.0, y -18.0 cm (460 mm, 46 m, abaft F: 60 mm over half a length).
2. The right arm sets the red cube from y -3.0 cm at x 30.0, y -8.0 cm (its outboard face 5 mm, 0.5 m, inboard of
   the port side: the middle of the 10 mm band).
3. The right arm sets the green cube from y +3.0 cm (the left arm has parked) at x 40.0, y -8.0 cm (5 mm inboard
   of the starboard side), and parks.

Offline (`tools/offline/check_task.py`): worst IK error 0.09 mm, smallest joint margin 12.1 degrees (joint 4, right arm), the
two tool points never closer than 24.0 cm and the arms' collision capsules never closer than 5.5 cm. The
kinematic proxy predicts 1.0, with success held at step 1,065 (53.3 s); the scripted policy would stop at step
1,125. Every unit landing 4 mm short along y still scores 1.0 (a test). `PLACE_TRIM` in `task.py` moves an arm's
place points if the GPU trace shows its units landing off by a steady amount in x (the sidelights have 5 mm either
way, the masthead units 6 mm). The run limit is 2,400 steps (120 s), more than twice the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the white masthead units back upright on
the table centred at x 23.0 cm, y +15.0 cm and y -9.0 cm, the green cube at x 23.0 cm, y +3.0 cm and the red cube at
x 23.0 cm, y -3.0 cm, square to the table's edges; the deck plate, centreline and sternlight are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 53 s of simulated time, when
success has held for 2 s; a run to the limit is 120 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 10 mm grey PVC (or painted plywood) plate cut to the deck outline: 800 mm long, 130 mm wide, square at the stern,
parallel sides for 640 mm, then tapering straight in to a point over the last 160 mm, fixed to the table 28.5 cm in
front of the arm bases with its length along y and the point toward the left arm; a 2 mm white line painted or
taped along its middle; a 12 mm white plastic dowel 15 mm long glued upright on the line 13 mm from the transom (the
sternlight). Two 20 mm lengths of 20 mm white-anodised (or white-painted) aluminium round bar, and two 20 mm
aluminium cubes, one red and one green. Positions in `task.json`'s `setup`.

Scoring a real trial: once the arms have withdrawn, take a top photo with a ruler on the deck for scale, or measure
directly. With a tape along the centreline from the stem, read the forward white unit's base centre (200 mm or less
from the stem, within 6 mm of the line) and the distance between the two white units (400 mm or more, the after one
within 6 mm of the line); with a rule square to the side, read each sidelight cube's gap from its own side's edge
(10 mm or less, cube not overhanging) and check it is no nearer the bow than the forward white unit. Check every
unit stands upright on the deck.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
