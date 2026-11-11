# Join a branch drain with a wye, not a sanitary tee, its branch entering with the flow

Plumbers join new branch drains to existing horizontal drains every day. The International Plumbing Code (IPC 2021,
Section 706.3) requires that "fittings shall be installed to guide sewage and waste in the direction of flow", and
its Table 706.3 (fittings for change in direction) allows a sanitary tee only for a horizontal-to-vertical change:
its vertical-to-horizontal and horizontal-to-horizontal cells are blank at every pipe size, while a wye (or a
combination wye and eighth bend) is allowed for every change. So where a horizontal branch joins a horizontal drain
the fitting must be a wye, and it guides the waste with the flow only when its 45 degree branch enters pointing
downstream: seen from above, the branch's outward axis (from the drain toward the fixture) points upstream, at 135
degrees to the flow. A sanitary tee laid flat, or a wye turned against the flow, sends waste across or against the
flow, leaves solids behind and clogs; an inspector fails it. (Footnote a of the table lets a quarter bend or short
sweep turn horizontal-to-horizontal on a fixture drain of 2 in or less; neither fitting is in this scene, and the
footnote does not apply to a sanitary tee.)

Here a level drain of grey 40 mm square-section pipe crosses the table, black arrows on its top showing that waste
flows toward -y, and a 134 mm section has been cut out of it around y 0 for a new horizontal branch from a fixture
on the front side (toward the arms). Two white fittings stand on the bench, a wye and a sanitary tee, both with the
same 118 mm run (so both fit the gap) and the same mass; they differ only in their branch: 45 degrees and rounded,
or 90 degrees and square with a swept inside corner. The robot has to pick the fitting the code allows from its
shape, seat it in the gap in line with the drain, turn it so its branch faces the fixture and leans upstream (one of
the wye's four flat orientations), lay the grey branch pipe in line with the branch, its end at the opening, leave
the tee on the bench and let go. This is the dry fit a plumber makes before cementing; the score is what they read
with a rule, a protractor and a look from above.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three loose parts' poses and the grippers' state, with a small tracker of each part's last second |
| `mechkit.py` | Helper that builds each fitting as one rigid body of its run and branch, and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and window edge, every zero condition, the shortcut checks and exploits we tried, unreadable and non-finite states, resets, the demo plan and its modelled outcome (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses `yam_arena`'s own boxes,
the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain box has no branch. `mechkit.py`
(imported by `task.py`) builds each fitting as one loose rigid body through a hook on `yam_arena.scene` and reports
its pose (base-frame origin and orientation) to the scorer; it stops the run with an error if a fitting was not built
as a compound body. Its shapes are axis-aligned in the fitting's own frame, so the wye's 45 degree branch is a row
of six overlapping upright round discs (40 mm across and tall, 7 mm apart) along the branch axis: a smooth 40 mm wide
branch with a rounded end. The tee's branch is a 40 x 40 x 55 mm box and its sweep three small steps in one inside
corner. The branch pipe is a plain loose box.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks. `rigstate.py`
(imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of both arms (tool
point, the point between the pad tips, finger opening, in the objects' frame) before each control step and to start
a new trial at each reset. If it cannot read the arms it prints why once; the let-go condition is then taken as met
on those steps, and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique (a
`task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (right arm: the wye; left arm: the branch pipe): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python3 -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/branch-drain-wye-with-flow/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python3 -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/branch-drain-wye-with-flow/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: `line` (which line of the decision list matched) and
`zero_reason`; for `wye` and `tee`: `seated`, `off_x_mm`, `off_y_mm`, `run_axis_err_deg`, `tilt_deg`, `lowest_mm`,
`counted`, `still_1s`, `released`, `path_mm`, `turn_deg`; for `pipe` the same stillness fields; `theta_b_deg`,
`branch_err_deg`, `tip_E`, `u`, `gap_mm`, `lateral_mm`, `angle_deg`, `outward_mm`, `pipe_flat`, `pipe_tilt_deg`,
`pipe_lowest_mm`, `pipe_fine`, `pipe_coarse`, `tee_clear`, `tee_dist_mm`, `arms_unavailable`, `jumps`; the wye's
origin and orientation, the tee's run centre, the pipe's pose and both parts' velocities, the demo's two targets,
both TCPs, pad-tip points and openings. Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's
record per process and starts it over at every reset (rigstate's and mechkit's reset counters, and any part moving
8 cm or more in one step), so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three loose parts' poses and both
grippers once per control step (20 Hz). Nothing is latched: dry-fitting the wrong fitting and swapping it is normal
practice and leaves no trace, and nothing in the task is a violation a later state could hide, so the score is the
line the final state matches.

A part is *counted* when it is released (both grippers' finger zones at least 15 mm from it at every step of the
last 0.5 s and at the step at hand) and settled (over the last 0.5 s its tracked centre moved under 2 mm, measured
as path length on positions because the simulator's reported velocities chatter while a part is gripped, and it
turned under 1 degree, summed step by step). rigstate gives no gripper heading, so a gripper's fingers are taken to
be anywhere round its axis: the zone is everything within reach of a pad's outer corner (half the current finger
gap plus the 8 mm pad thickness out, 13.5 mm across; 57 mm when fully open) around the 38 mm stretch back from the
point between the pad tips. *Still for 1 s* is settled in each of the last two 0.5 s windows.

A fitting is *seated* when it is counted, flat (its up axis within 5 degrees of vertical, either way up, and its
run's lowest corner within 3 mm of the table top), its run within 5 degrees of the drain's axis and the run's centre
within 6 mm of the gap's centre (x 380 mm, y 0) in x and in y. The wye's branch direction theta_b is the yaw of its
branch's outward axis computed from its pose, so an upside-down wye is read correctly: 135 degrees is the code's
orientation (front, leaning upstream); the other three flat orientations read 45 (back, upstream), -45 (back,
downstream) and -135 (front, downstream). 45 and -135 are the wye turned over, -45 is it upright and turned end for
end. A seated wye's run is within 5 degrees of the drain, so its branch is within 5 degrees of one of the four; the 7
degree window tells them apart. The opening E is the centre of
the branch's end, 75 mm out from the run's centre line along the branch axis u. The branch pipe is *on line* when it
is counted, flat on the bench (its length within 5 degrees of horizontal and its lowest corner within 3 mm of the
table top: a pipe resting on a fitting never counts), lies outward of E (its centre at least 50 mm beyond E along
u), its length is within 7 degrees of u and the centre of its end nearer E is within 6 mm of the branch axis and -3
to +10 mm beyond E along it, 3 mm of overlap allowed for contact (fine); 15 degrees, 15 mm and -3 to +40 mm
(coarse). The tee is *off
the drain* when it rests on the bench on any of its faces (upright, turned over, on its side or on its end: one
of its axes within 5 degrees of vertical and its lowest corner within 3 mm of the table top, so a tee on the drain,
on another part or propped at an angle never counts) with its run's centre more than 60 mm from the gap's centre.

| Final state (first match wins) | Score |
|---|---|
| The sanitary tee seated in the gap, any way round (IPC Table 706.3 does not allow it horizontal-to-horizontal) | 0 |
| No wye seated: the gap empty, the wye held, moving, tilted, crosswise or off centre by more than 6 mm | 0 |
| The wye seated with its branch to the back or pointing downstream (theta_b 45, -45 or -135) | 0.25 |
| The wye seated and correct, the pipe on line (fine), the tee off the drain, all three still for 1 s (success once held 1 s) | 1.0 |
| The wye seated and correct, the pipe on line (coarse); includes the 1.0 layout with the tee on the drain, near the gap or not yet still | 0.75 |
| The wye seated and correct, the pipe anywhere else (on the bench, short of the opening, off the axis, resting on the wye, held) | 0.5 |

Every tier is reachable and every final state matches exactly one line: the start state and a seated tee give 0, a
wye dropped in at the wrong yaw 0.25, a correct wye alone 0.5, the pipe left 20 to 25 mm short of the opening 0.75,
the demo 1.0. Unintended strategies earn nothing beyond what the rule allows, as the tests show: the tee pushed into
the gap at its start yaw or any other way round (0); the pipe laid in the gap instead of a fitting (0); the wye slid
in from its start without turning (crosswise, 0) or turned the wrong way (0.25, the pipe against its branch changes
nothing); the wye upside down with its branch to the front (0.25); the wye held in place by a gripper at the end,
or an open gripper hovering within 15 mm of it (0); the pipe laid on top of the wye's branch (0.5); the tee laid
along the branch in place of the pipe (0.5); the tee stacked on the seated wye (0.75 at most).

## One arm alone

The demo uses both arms, one part each: the right arm seats the wye, and once it has let go and lifted clear the
left arm lays the branch pipe. `DEMO_IDLE_ARM=left` (right arm only) seats the wye correctly and leaves the pipe on
the bench: offline prediction 0.5. `DEMO_IDLE_ARM=right` (left arm only) lays the pipe where the branch would be but
seats no fitting: offline prediction 0. Gripping straight down, neither arm reaches both start areas: the wye starts
51.5 cm to the side of the left arm's base (y -21 cm), and the pipe starts 66.5 cm to the side of the right arm's
(y +36 cm), far past the 10 cm beyond the midline either arm reaches. A policy might still push parts within the
other arm's reach or use a leaned gripper, so no claim is made that one arm cannot score 1.0. The task does not need
two arms as a skill: its difficulty is choosing the fitting by the code, reading the flow, turning the wye the one
right way of four and lining the pipe up with its branch; one arm only lacks the reach in this layout. GPU one-arm
runs: 0.5 with the left arm idle and 0 with the right arm idle, as predicted.

## Demo and run limit

The scripted demo takes 31.1 s (622 control steps). Right arm (20.3 s): it comes down beside the wye where it lies
with its run along x (pad tips 11 mm above the table), grips the run across its 40 mm (fingers commanded 32 mm, over
1.5 s) 40 mm downstream of the run's centre, clear of the branch's root (the pads stay 21 mm from it; the wye's
centre of mass is 59 mm from the grip, about 0.06 N m on the pads), lifts it to 9 cm (gripper height; the wye's
underside 26 mm over the drain and the tee), carries it toward the gap while turning the grip from yaw 0 to +88.85
degrees (the wye turns from -90 to -1.15 degrees: run along y, branch to the front and leaning upstream), makes its
last move 25 mm along (+x, -y), settles 0.8 s, lowers it into the gap until its underside is 2 mm above the table
(the last 3 cm over 2 s), waits 0.5 s, opens over 1.5 s, goes straight up and moves aside. That turn keeps the wrist
within 55 degrees of the place's bearing from the base; the other way (yaw -91.15) would be 125 degrees off, at joint
6's limit. The carry height is 9 cm, not 12: over the gap the right arm's joint 4 is 2.5 degrees from its limit at a
12 cm TCP height and 12 degrees at 9 cm (our kinematics). Left arm (31.1 s): it grips the pipe at its middle across
its 40 mm section at once, holds it up until the right arm has let go and lifted (18.8 s), carries it toward the
branch while turning the grip by -46.15 degrees (the pipe's length from x to 133.85 degrees), makes its last move 25
mm along (+x, -y), lowers it the same way, lets go and moves aside.

The pipe's target is set from the nominal seated wye (run centre on the gap's centre, yaw 0): opening E at (327.0,
53.0) mm, branch axis at 135 degrees; the pipe's centre goes 75.5 mm out from E along the axis (its near end 5.5 mm
from the opening) and 1.4 mm across it. In our earlier GPU runs, parts carried in a gripper ended turned
counter-clockwise from the commanded heading by about +0.6 to +1.7 degrees and landed about 4 mm short of the target
along the last move. So both parts are laid 1.15 degrees clockwise of nominal (`YAW_TRIM`), both are aimed 2 mm past
their nominal points (`LEAD`, half the shortfall), and both make their last 25 mm move the same way, along minus the
branch axis: a short landing then moves the wye's opening and the pipe's end the same way along the axis and leaves
the gap between them and the lateral offset almost unchanged. Offline (no turn, no shortfall) the wye lands 2 mm
long, at (381.4, -1.4) mm, and the pipe's end is 5.5 mm from the opening and 1.6 mm off the axis. Modelled with each
part independently turned +0.6 to +1.7 degrees about its grip point and 2 to 6 mm short (in the tests), the
wye ends within 4.0 mm of the gap's centre and at least 4 mm from both cut ends, its branch within 0.6 degrees of
135, and the pipe's end 0.6 to 9.2 mm from the opening and 0.2 to 3.7 mm off the axis: inside the fine window in
every case. Offline the success holds for its 1 s at step 631. The run limit is 1,200 steps (60 s).

GPU watch (`TASK_TRACE`): after the wye's release, `wye.off_x_mm`/`off_y_mm` about -2 to 0 and +0 to +2 and
`theta_b_deg` about 135; after the pipe's release `gap_mm` about 3 to 8 and `lateral_mm` about 0 to 3. If the wye
lands more than 3 mm off, move `LEAD` or the wye's aim by minus the offset; if `lateral_mm` is beyond +-3, move
`LATERAL_AIM` by minus it; if `gap_mm` is under 2 or over 8, move `GAP_AIM`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the wye back flat with its run along x
centred at x 28.0 cm, y -21.0 cm, its branch on the +y side leaning toward +x; the sanitary tee flat with its run
along y centred at x 24.0 cm, y -4.0 cm, its branch toward +x and its sweep on the branch's +y side; the branch
pipe flat along x centred at x 28.0 cm, y +36.0 cm. The drain and the arrows are fixed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 32 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 213 mm lengths of 40 x 40 mm square-section grey PVC (or 1-1/2 in DWV pipe in cradles) clamped to the bench end
to end along one line with a 134 mm gap between their cut ends, and four black arrows painted on their tops, all
pointing the same way. A white 1-1/2 in wye and sanitary tee with the same run length (118 mm here) and mass, and a
140 mm length of the same grey pipe. Nothing is cemented, powered or under pressure. Score with a steel rule (the
run's centre from the gap's centre, the pipe's end from the branch's opening and from its axis), a protractor (the
branch's direction against the arrows, the pipe's line against the branch) and a photo from above showing both
grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
