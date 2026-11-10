# Load a Charpy specimen notch away from the striker, centred and square on the anvils

Charpy impact tests (ASTM E23, ISO 148-1) are run every day by materials-testing technicians in weld-procedure
qualification, steel-mill and pressure-vessel quality labs. The notched test bar is laid on the machine's two
supports, square against its two anvils, with the notch centred between the anvils and facing away from the striker:
the pendulum's striker hits the plain back face at mid-span and the bar breaks through the notch. A bar loaded notch
toward the striker, off centre or standing off an anvil gives an invalid absorbed-energy value and a retest. The bar
is placed with tongs, and they are withdrawn before the pendulum is released. Here a 2:1 bench mock-up of a tester's
specimen area stands on the table (a cast-iron base, two steel supports, two steel anvils with an 80 mm gap between
them, and a red strip marking the striker's side), and a 2:1 aluminium Charpy bar lies on a mat beside it with its
notch to one side. The robot has to turn it a quarter turn the right way (which way is right depends only on where
the striker is), lay it on both supports against both anvils with the notch centred, let go and move clear. The score
is what a technician reads with a steel rule, a feeler gauge and a look from above. This is a single-arm task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the specimen's pose and the grippers' state, with a small tracker of the bar's last half second |
| `mechkit.py` | Helper that builds the specimen as one rigid body of three boxes and a tick, and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, the exploits we tried, unreadable and non-finite states, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain box has no notch. `mechkit.py`
(imported by `task.py`) builds the specimen from three boxes (a full-length back part and two notch-side pieces
leaving the 4 x 4 mm slot) and a thin black tick, as one loose rigid body, through a hook on `yam_arena.scene`, and
reports its pose (base-frame origin and orientation) to the scorer. It stops the run with an error if the specimen
was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read the arms it prints why once; the let-go condition is
then taken as met on those steps, and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (right arm; the left arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the Charpy specimen into the impact tester: lay it across the two supports, square against both anvils, with its notch facing away from the striker (its side is marked by the red strip) and centred between the anvils. Then let go of it and move clear of it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/charpy-notch-away-centred/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Load the Charpy specimen into the impact tester: lay it across the two supports, square against both anvils, with its notch facing away from the striker (its side is marked by the red strip) and centred between the anvils. Then let go of it and move clear of it." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/charpy-notch-away-centred/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`tier`, `zero_reason`, `tilt_deg`, `heading_err_deg`,
`notch_angle_deg`, `notch_dir`, `offset_mm`, `gap_left_mm`, `gap_right_mm`, `onto_mm`, `end_x_mm` and
`underside_mm` at both ends, `on_supports`, `at_rest`, `moved_mm`, `hands_off`, `zone_gap_mm`, `arms_unavailable`,
`notch_y_mm`, `notch_yaw_deg` (the notched face's heading seen from above, counter-clockwise from +x), the
specimen's origin, orientation and ends, both TCPs, pad-tip points and openings). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every reset
(rigstate's and mechkit's reset counters, and a teleport back to the starting mark), so run one trial per process or
rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the specimen's pose and both grippers once
per control step (20 Hz). From the pose of the bar's base frame (the middle of its bottom face; x along the bar,
-y out of the notched face, z up through the tick) it computes the centres of its two end faces, the centre of the
notch's mouth and the notched face's line at mid-height. "On the supports" is checked at each end: at least 3 mm
past its support's inner edge (40 mm from the midplane), its centre line over the support (x 300-330 mm) and its
underside within 2 mm of the support's top; the notch's 20 mm length must also be within 5 degrees of vertical
(the bar is the same either way up, so either face may be on top) and the bar's length within 10 degrees of the
line between the anvils. "Notch away" is the notched face's outward normal within 30 degrees of +x, toward the
anvils and away from the red strip. The offset d is the notch's distance along y from the midplane between the
anvils: half the difference of two rule readings, one from each anvil's inner edge to the notch. The anvil gap is,
for each anvil, the widest gap between its face and the bar's notched face over the stretch of the bar in front of
it, at mid-height: where a 1.5 mm feeler gauge would enter. A bar touching one anvil and turned about it stands off
the other: turned 1 degree it is 1.9 mm off at the far end, and loses the last 0.25.

Every score needs the bar at rest (over the last 0.5 s neither end moved more than 2 mm, measured on positions: the
simulator's reported velocities chatter while a part is gripped) and let go: each gripper's finger zone at least
20 mm from the bar. rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round its
axis: the zone is everything within reach of a pad's outer corner (half the current finger gap plus the 8 mm pad
thickness out, 13.5 mm across; 57 mm when fully open) around the 38 mm stretch back from the point between the pad
tips. A new trial is recognised at rigstate's or mechkit's reset, or when the simulator puts the bar back on its
starting mark (within 4.5 cm of it, which covers a seeded reset's jitter) in a single step. A simulator state with a
non-finite number, a zero quaternion or no base-frame origin scores 0 for that step and leaves the scorer's record
as it was.

| Outcome (at rest, let go, at the end) | Score |
|---|---|
| On the supports, notch away, d <= 2 mm, both anvil gaps <= 1.5 mm (success once held 1 s) | 1.0 |
| On the supports, notch away, d <= 2 mm, an anvil gap over 1.5 mm (centred, standing off an anvil) | 0.75 |
| On the supports, notch away, 2 < d <= 8 mm, both anvil gaps <= 1.5 mm (against both anvils, off centre) | 0.75 |
| On the supports, notch away, 2 < d <= 8 mm, an anvil gap over 1.5 mm | 0.5 |
| Notch toward the striker (or any way but away); d over 8 mm; not on the supports (held, propped, on its side, standing on an end, on the anvils' tops, down on the base or table, crosswise); still moving; a gripper on or within 20 mm of it | 0 |

Every tier is a position the bar rests in by itself: 8 mm off centre each end still reaches 7 mm onto its support,
and a bar up to 10 mm in front of the anvils still lies wholly on the 30 mm deep supports. Unintended strategies
earn nothing, as the tests show: the bar centred against the anvils but notch toward the striker (0); held in place
against the anvils by the gripper (0); pressed against them by an open gripper's pad from behind (0); laid on its
side, or standing on an end between the anvils (0); laid across the anvils' tops (0); dropped into the gap onto the
base (0); propped between a support and the base (0). A bar turned upside down (tick down) with its notch away is
the same bar loaded the right way and scores the same.

## One arm alone

This task needs one arm, and the demo uses one: the right arm grips the bar at its middle across the notched and back
faces, lifts it, turns it a quarter turn to the left while carrying it to the tester (the notch swings from -y to +x,
toward the anvils), lowers it beside the anvils, moves it against them, lowers it onto the supports, lets go and moves
aside, while the left arm holds its ready pose. `DEMO_IDLE_ARM=left` is therefore the same run as the demo, and on
the GPU it scored 1.0; `DEMO_IDLE_ARM=right` leaves nothing to move and scored 0.0 (the bar stays on the mat). So the
right arm alone completes the task. Gripping straight down, the left arm reaches no part of the bar where it starts
(yamkin: its nearest miss is about 10 mm, near the bar's back end), so it cannot pick it up there; a policy might
still nudge the bar within the left arm's reach with a tilted gripper, so no claim is made that the left arm alone
cannot score. No claim is made that the task needs both arms: its difficulty is reading which way the notch must face
from where the striker is, turning the bar the right way, centring it and seating it against both anvils.

## Demo and run limit

The scripted demo takes 22.3 s (446 control steps). The right arm comes down beside the bar on the mat (pad tips 4 mm
above it), grips it across its 20 mm at its middle (fingers commanded 8 mm narrower, over 1.5 s), lifts it to 12 cm
(gripper height), carries it over the supports in 2.5 s while turning the grip from yaw 0 to +88.8 degrees (see
below), comes down beside the anvils with the notched face 4 mm in front of them and the bar's underside 8 mm above
the supports, moves it 6.5 mm toward the anvils (aimed 2.5 mm past their faces, so the anvils stop it), lowers it the
last 7 mm onto the supports over 1.5 s while pressed against them, waits 0.5 s, opens over 1.5 s, goes straight up
and moves aside. Lowering beside the anvils first means the bar can never land on an anvil's top; pressing it against
them lets the anvils, not the arm's accuracy, close the gap: any arm error from 4 mm short of the anvils upward still
ends in contact. The arm's controller does not carry the bar's weight: the compliance model (J K^-1 J^T of the 0.118
kg bar's weight, joints 80/80/80/10/10/10 N m/rad, yamkin) puts the gripper about (-0.9, -0.9, -3.5) mm off at the
place pose, so pressed 1.6 mm past contact the arm pushes the bar against the anvils with about 0.25 N, and the bar
meets the supports before the slow descent ends. The place point is trimmed 0.4 mm along y (about half the modelled
sideways sag), so the notch lands within about 0.5 mm of the midplane with or without the sag (`PLACE_TRIM` in
`task.py`). Squareness is the tight part: the bar takes the heading of the flat pads that hold it, and a 1.5 mm
feeler over its 110 mm length allows only 0.78 degrees of skew, which pressing at its middle cannot take out. In our
earlier simulator runs, parts carried in a gripper ended turned counter-clockwise from the commanded heading by about
0.6 to 1.7 degrees (in a matching pick and place with the right arm, 1.5 degrees in the hand), so the demo lays the
bar at a heading 1.2 degrees clockwise of square (`YAW_TRIM` in `task.py`): any turn from 0.45 to 1.95 degrees then
leaves the far end within the feeler. Offline, with no such turn, the bar is laid 1.2 degrees turned (far gap 2.3 mm,
which would score 0.75). Offline the bar drops 1 mm at the release and success holds for its 1 s at step 448, once
the hand has moved 20 mm clear. On the GPU the bar came to rest turned 0.45 degrees clockwise of square (anvil gaps
0.35 and 1.1 mm), its notch 0.4 mm from the midplane, and success held for its 1 s at step 448 (score 1.0). The run
limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the specimen back on the black mat centred
at x 20.0 cm, y -22.0 cm, lying flat along x with the notched face toward -y (away from the midline) and the tick up;
everything else is fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 23 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 140 x 220 x 30 mm cast-iron or steel plate as the base; two 30 x 35 x 10 mm hardened-steel (or any steel) blocks
as the supports and two 20 x 35 x 40 mm blocks as the anvils, fixed right behind them with their faces flush with
the supports' far edges and their inner edges 80 mm apart; a 25 x 8 mm strip of red paint or tape on the base in
front of the gap; a 140 x 50 x 3 mm black rubber mat; and a 110 x 20 x 20 mm aluminium bar (118 g) with a 4 mm wide,
4 mm deep slot milled across the middle of one long face, the full 20 mm height, and a small black paint mark on
its top face behind the slot. Clamp everything but the bar to the bench; nothing moves or is powered. Score with a
steel rule from each anvil's inner edge to the notch, a 1.5 mm feeler gauge at each anvil and a photo from above
showing both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
