# Set up a weld radiograph: IQI wires across the weld, lead markers at both end ticks

Industrial radiographers set up every weld shot before the exposure, and the code fixes where the reference objects
go. ASME Boiler and Pressure Vessel Code Section V, Article 2: the image quality indicator (IQI) goes on the source
side of the part, and a wire-type IQI is placed on the weld so that the length of its wires is perpendicular to the
length of the weld (T-277.1); location markers go on the part, not on the film holder, so that their images show the
coverage at each end of the area of interest, on the source side for flat components (T-275, T-275.1(a)(1)). The
wires' images prove the shot's sensitivity across the weld, and the markers tie the film to the weld so a defect can
be found again; a film without a valid IQI or markers is rejected and the weld shot again.

Here a butt-welded steel plate (160 x 200 x 10 mm) lies on the table on the midline, with the film taken to be
under it, so its top face is the source side. The weld runs along y across its middle and two yellow ticks across
the weld mark the ends of the 160 mm this shot covers. On the bench in front of the plate lies a wire IQI (a clear
blue holder with six dark wires along its length) with its wires along the weld, so carrying it straight across lays
it the wrong way; beside it, one on each side, lie the two lead location markers "0" and "1". The instruction names
the code but not the placements: the robot has to lay the IQI on the plate's top on the weld with its wires across
it, inside the shot length, and a marker on the plate's top at each tick, then let go. The score is
what a radiographer reads from a top photo with a rule.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three parts' poses and the grippers' state, with a small tracker of each part's last second |
| `mechkit.py` | Helper that builds the IQI and the markers as rigid bodies of a holder and thin painted wires or figure, and reports their poses (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and band edge, every zero condition, the released rule, stillness, the exploits we tried, broken states, resets, the task file, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, which carry one colour. `mechkit.py` (imported by
`task.py`) builds the IQI from its holder and six thin dark strips (the wires) on its top, and each marker from its
block and a thin white figure, as one loose rigid body each, through a hook on `yam_arena.scene`, and reports each
part's pose (base-frame origin and orientation) to the scorer, so the wires show in the camera which way they run.
It stops the run with an error if a part was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. An arm it cannot read is skipped by the released rule on that step (the
other is still checked) and the trace counts such steps (`arms_unavailable`).

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
  --instruction "This butt-welded plate is being set up for its radiographic test shot. The film goes under the plate, so the top of the plate is the side the shot is taken from. The painted stripe is the weld, and the two painted ticks across it mark the ends of the length this shot covers. Place the wire-type image quality indicator and the two lead location markers (0 and 1) where ASME Section V requires them for this shot, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/wire-iqi-across-weld-markers/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This butt-welded plate is being set up for its radiographic test shot. The film goes under the plate, so the top of the plate is the side the shot is taken from. The painted stripe is the weld, and the two painted ticks across it mark the ends of the length this shot covers. Place the wire-type image quality indicator and the two lead location markers (0 and 1) where ASME Section V requires them for this shot, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/wire-iqi-across-weld-markers/dream
```

Run one trial per process (the default). Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves
that arm at its ready pose (the one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check);
`TASK_TRACE=trace.jsonl` writes one JSON line per control step with what the scorer saw and decided: `score`,
`tier_reason`; for the IQI `iqi_on_plate`, `iqi_tilt_deg`, `iqi_bottom_mm` (its lowest point over the plate's top),
`iqi_wire_err_deg`, `iqi_off_weld_mm`, `iqi_max_abs_y_mm`, `iqi_pose_ok`, `iqi_correct`; for each marker
`marker_*_tick`, `marker_*_from_weld_mm`, `marker_*_y_mm`, `marker_*_bottom_mm`, `marker_*_tilt_deg`; for each part
`*_released`, `*_path_mm`, `*_yaw_change_deg`, `*_counted`, `*_still_1s` and its pose; `ticks_marked`; each
gripper's `zone_gap_mm` to each part, `left_tips`/`right_tips`, `left_opening`/`right_opening`, `arms_readable` and
`arms_unavailable`. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three parts' poses once per control
step (20 Hz), and from the embodiment each gripper's tool point, the point between its pad tips and how far it is
open, and scores the state at hand: the score is read at the last step.

A part is released at a step when each readable gripper's finger zone is at least 10 mm from its bounding box. The
gripper's yaw is not read, so the zone is a cylinder round the gripper's axis from the point between the pad tips
38 mm back up the gripper, out to a pad's outer corner over every yaw (hypot(half the finger gap + 8 mm, 13.5 mm):
33 mm with the fingers open 44 mm): a gripper closed on a part, or open with its pads beside it, holds it; one whose
pad tips are 10 mm or more above the part's top has let go. A part is settled when its centre moved under 2 mm in all
over the last 0.5 s and its heading under 1 degree (read from the path, not from the simulator's reported velocity,
which chatters while a part is gripped); it counts when it was released at every step of that 0.5 s and settled.
Still for 1 s: settled in each of the last two 0.5 s windows and released throughout the last second.

A part lies on the plate when it is flat (its top face within 5 degrees of level, either face up), its centre is
over the plate's outline and its lowest point is within 3 mm of the plate's top, so a part on the table (the film
side), on another part, on its side or held up does not. The IQI is correct when it counts, lies on the plate, its
wires are within 8 degrees of x (perpendicular to the weld, either way round), its centre is within 15 mm of the weld
centreline (x 36.0 cm: it lies on the weld and every one of its six 50 mm wires crosses the whole 14 mm cap, with
at least 1.3 mm of wire past the far edge even 15 mm off and 8 degrees skewed) and every corner of
its footprint is between the ticks' centre lines (|y| at most 8.0 cm). A tick has a correct marker when a marker
counts, lies on the plate, and its centre is within 10 mm of that tick's centre line along the weld and within
35 mm of the weld centreline. Either marker may be at either tick, and two markers at one tick count once. Where a
marker sits across the weld line is not scored (on the weld or beside it on either side): no code text found
requires it.

| Line (first match wins) | Condition | Score |
|---|---|---|
| 1 | IQI correct, both ticks with a correct marker, all three parts still for 1 s | 1.0 |
| 2 | IQI correct, at least one tick with a correct marker (or the line 1 layout not yet still for 1 s) | 0.75 |
| 3 | IQI correct, no correct marker | 0.5 |
| 4 | IQI counted and lying on the plate, not correct (wires along the weld or more than 8 degrees off x, centre more than 15 mm off the weld centreline, footprint past a tick) | 0.25 |
| 5 | Anything else: the IQI on the bench, on the table beside the plate, on its side or end, on a marker, still held or moving | 0 |

Markers are scored only when the IQI is correct: a shot without a valid IQI is rejected whatever its markers show.
Every final state matches one line, and every line is reachable: the demo, 1.0; one marker left on the bench, or
both at one tick, 0.75; both markers left on the bench, or both laid beside the IQI 45 mm from the ticks, 0.5; the IQI
carried straight across and laid along the weld (with or without markers), 0.25; nothing moved, 0. No latch is
needed: placing and adjusting the parts is set-up, and nothing is a violation a later state could hide. The trial
ends early once line 1 has held for 1 s; otherwise it is scored as it stands at the run limit. A state with a
non-finite number, a zero quaternion or a part not reported as an assembly scores 0 for that step and leaves the
tracker as it was.

Unintended strategies earn nothing extra, as the tests show: the IQI laid along the weld with both markers at the
ticks (0.25); the IQI on the table beside the plate (0); the IQI standing on its side or end over the weld (0); the
IQI on a marker (0); a marker on the IQI or on the other marker (not on the plate's top: that tick is unmarked);
markers on the table at the ticks' y beside the plate (not on the part: 0.5 with the IQI right); markers at the
plate's edges, 70 mm from the weld (0.5); the full layout with the last marker still gripped (0.75).

## One arm alone

The demo splits the work: the right arm lays the IQI and marker 0, the left arm marker 1. The offline kinematic proxy
predicts 0.75 with the left arm idle (the IQI and marker 0 placed, marker 1 left on the bench) and 0 with the right
arm idle (marker 1 placed, the IQI left on the bench). Gripping straight down, neither arm reaches the other side's
marker where it starts (x 24.0 cm, y -16.0 or +16.0 cm, 46.5 cm across from the far base; no heading in 15 degree
steps reaches it in our IK model, `tests/test_scoring.py`, `OneArmReach`), but each reaches the IQI and both ticks.
We make no claim that two arms are needed: one arm could score 1.0 by pushing or sliding the far marker into reach,
or by gripping it with the wrist leaned. The GPU one-arm scores will be reported in "Results in simulation".

## Demo and run limit

The scripted demo takes 42.0 s (840 control steps), both arms on one clock. Each part is gripped straight down across
two faces with the pad tips 3 mm above its underside, the gap commanded 8 mm narrower than the part (IQI: opened to
44 mm, closed to 20 mm across its 28 mm width; markers: 40 and 14 mm across 22 mm). Every open and close takes 1.5 s
after a 0.5 s settle, every final descent is slow (30 mm over 1.5 s), each part is let go 1 mm above the plate's top,
and parts travel with their undersides 39 mm above the table (29 mm over the plate).

1. Right arm, 0 to 16.5 s: grips the IQI at yaw 90 degrees (fingers closing along x), carries it over the weld
   while turning a quarter turn clockwise, and lays it with its wires along x, centred at x 36.2 cm, y 0.2 cm.
2. Left arm, 16.5 to 32.5 s (it waits high over its side of the bench until the right arm has left the plate):
   lays marker 1 beside the weld at the +y tick (x 33.5 cm, y +7.8 cm), on the arms' side of the weld.
3. Right arm, 18.5 to 42 s: picks marker 0, holds it up over its start until the left gripper has risen from the
   plate, then lays it at the -y tick (x 33.5 cm, y -7.8 cm) and goes back over its side of the bench.

Place points are moved 3 mm outward along the line from the placing arm's base, against the about 4 mm short
landings seen in our GPU runs, and the IQI is laid at -1.2 degrees, against the +0.6 to +1.7 degree turn parts take
in the grippers; with or without those errors every part is well inside its window (the IQI's centre 2.3 mm from the
weld centreline of 15 mm allowed, its wires 1.2 degrees off of 8; the markers 25.5 mm from the centreline of 35 mm,
1.7 mm from the tick lines of 10, their footprints 2.5 mm short of the ticks). `PLACE_TRIM`, `LAND_SHORT` and
`YAW_TRIM` in `task.py` adjust them if the GPU trace shows a steady offset. Offline the grippers stay at least
25.6 cm apart and the arms' collision capsules 9.0 cm; the two grippers are never over the plate at the same time.
The smallest joint margin is 15.5 degrees (joint 4); the worst IK error 0.1 mm. Success (line 1 held 1 s) is reached
at step 842 offline, after the right arm has gone back; the scripted policy would stop at step 901. The run limit is
1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lay the IQI back flat on the table centred at
x 23.0 cm, y 0, wires along y and up; marker 0 flat at x 24.0 cm, y -16.0 cm and marker 1 at x 24.0 cm,
y +16.0 cm, square to the table edges, figures up with their tops toward +x. The plate, weld and ticks are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 43 s of simulated time; a run
to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 80 x 200 x 10 mm grey steel plates (or one 160 x 200 x 10 mm plate) fixed to the table side by side, x 28.0 to
44.0 cm, y -10.0 to +10.0 cm, with a 14 mm dark grey strip along y across the middle (x 35.3 to 36.7 cm) standing
for the ground-flush weld cap; two 24 x 4 mm yellow tape ticks across it at y -8.0 and +8.0 cm. The wire IQI: a
55 x 28 x 12 mm clear blue plastic block of about 15 g with six dark lines along its length on top (1.0 to 3.0 mm
wide, 50 mm long, at a 4 mm pitch), or a real wire IQI glued on such a block. Two 22 x 22 x 12 mm grey plastic blocks
of about 15 g with a white or lead "0" and "1" on top. Positions in `task.json`'s `setup`.

Scoring a real trial: take a top photo with a rule on the plate. Check the IQI lies flat on the plate's top with its
wires within 8 degrees of square to the weld, its middle within 15 mm of the weld's centreline and its whole footprint
between the ticks; check each tick has a marker flat on the plate's top within 10 mm of it along the weld and within
35 mm of the weld's centreline, after the arms have let go and withdrawn.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.75 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
