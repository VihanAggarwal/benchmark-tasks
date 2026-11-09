# Mount a tower CPU cooler the one way round its offset fins clear the tall memory

Computer repairers and PC builders fitting a tower CPU cooler check its clearance to tall memory modules, the VRM
heatsink and the graphics card, and turn it until it fits. A cooler with a square mounting pattern can go on four
ways round, and on some the fin stack is offset from the base so that it clears the memory only one way, normally
with the fan blowing toward the rear exhaust. Forced on another way, the cooler sits on the memory, off the
processor. Here a bench model of a motherboard at about half scale lies flat on the table with its rear I/O edge
away from the arms: the processor in its retention frame, two tall memory modules on the near side, a VRM heatsink
on the left, a graphics card on the right and the I/O shield at the back. The cooler stands on a foam pad beside
the board, its airflow a quarter turn from the one way it fits. It must be carried to the processor, turned so its
fan blows toward the I/O shield, set down with its base flat on the lid inside the frame, and let go. The score is
read from where the cooler ends: a top photo with the frame as the gauge, and a feeler under the base.

The scene makes the rule physical. The cooler's fan and fins reach 37 mm from its base's centre on the outlet side,
18 mm on the fan side and 37 mm to either side, and their lowest edge is 6 mm below the memory's top when the base
sits on the lid. With the airflow toward the I/O shield they clear the memory, the VRM heatsink and the card by 9 mm
each (at least 4 mm wherever the base is inside the frame). A quarter or half turn from that puts the fan or the fins
at least 5 mm onto the memory, wherever the base is inside the frame, so the base cannot reach the lid.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the cooler's pose and the grippers' state, with a short memory for "at rest" |
| `mechkit.py` | Helper that builds the cooler as one loose rigid body of four boxes and reports its pose (see Requirements) |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests: the layout's geometry, every tier, every zero condition, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes, the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a plain box has no offset fin stack.
`mechkit.py` (imported by `task.py`) builds the cooler from four boxes (base, heat-pipe block, fan, fins) as one
loose rigid body through a hook on `yam_arena.scene`, and reports its pose (base-frame origin and orientation) to the
scorer. It stops the run with an error if the cooler was not built as a compound body. The score also needs to know
whether the robot has let go of the cooler, and `yam_arena` does not pass the grippers' state to a task file's hooks:
`rigstate.py` wraps the embodiment's step and reset once, at import, to take a snapshot of both arms (tool point,
finger-tip point, finger opening, in the objects' frame) before each control step and to start a new trial at each
reset. Both read only state the simulator already has and change nothing the embodiment does. The technique (a
`task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, wrapping the embodiment's step) follows the one first shown in the program repository's
`tasks/bearing-press`; both helpers are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm; the right arm holds its ready pose): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the tower cooler on the processor: set it down so its copper base sits flat on the processor inside the black retention frame, then let go. On this board it fits only one way round." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tower-cooler-only-way-fits/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mount the tower cooler on the processor: set it down so its copper base sits flat on the processor inside the black retention frame, then let go. On this board it fits only one way round." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/tower-cooler-only-way-fits/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `zero_reason`, `seated`, `let_go`, `at_rest`,
`moved_mm`, `foot_gap_mm`, `tilt_deg`, `airflow_err_deg`, `frame_margin_mm`, `base_off_mm`, `base_dx_mm`,
`base_dy_mm`, `over_board`, `lowest_mm`, `opening`, `left_zone_mm`, `right_zone_mm`, `arms_unavailable`,
`bad_steps`, the cooler's origin, orientation, velocity and stack top, both TCPs and tip points). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the state at the last step. The
cooler's pose is its base frame: the origin at the centre of the base's bottom face, local +x along the airflow (fan
to fins), local +z up. "At rest" means the base centre and the centre of the stack's top have each moved 2 mm or
less in the last 0.5 s (positions, not the reported velocity, which chatters while a part is gripped). "Let go"
means each gripper is open 0.9 or more, or its finger zone (every point that its fingers could occupy, whatever its
heading) is more than 10 mm from the cooler's bounding box.

| Outcome (cooler at rest) | Score |
|---|---|
| Seated (base centre within 2 mm of the lid, tilted 2 degrees or less, all four bottom corners inside the frame's 44 x 44 mm opening, airflow within 15 degrees of +x) and both grippers let go | 1.0 |
| Seated but still held | 0.5 |
| Let go, upright (within 30 degrees), the right way round (airflow within 15 degrees of +x), base centre within 20 mm of the processor's centre, not seated: tilted, raised, or partly over the frame | 0.5 |
| Let go, upright, base centre over the board, but turned another way round (for instance resting on the memory) or more than 20 mm from the processor's centre | 0.25 |
| Anything else: still moving, held and not seated, tipped over, still on the pad, on the table | 0 |

The frame's opening leaves the base 5 mm of play along x and y when it is square (a centred base turned more than
about 21 degrees no longer fits inside it). For seating, the airflow window is not an extra rule on top of the
physics: the test `test_only_one_way_round_fits_anywhere_in_the_frame` sweeps every heading in 1 degree steps and
every base position inside the frame, and the cooler clears the memory, VRM heatsink, card and shield exactly for
headings within 15 degrees of +x. For 0.5 the same window separates a near miss the right way round from a cooler
turned another way, which rests on the memory and earns 0.25. The trial ends early once the 1.0 conditions have held for 1 s (they need the gripper open, so a
trial never ends with the cooler still held); otherwise it is scored as it stands at the run limit. A new trial is
recognised from rigstate's reset hook, or, as a fallback, when the simulator puts the cooler back on the pad (within
4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with a non-finite number scores 0
for that step and leaves the record as it was. On a step where the grippers cannot be read, "let go" is taken as met
and the step is counted (`arms_unavailable`).

Every tier is reachable and the tests reach each: 1.0 by the demo's path; 0.5 by holding it seated, or by setting
it down the right way round but 6 mm off centre (partly over the frame), raised 2.5 mm, tilted 3 degrees, or 12 mm
toward the arms so the fan's foot rests on the memory; 0.25 by setting it down turned a quarter or half turn or 20
degrees off (it rests on the memory), or on the board away from the processor (or balanced on the memory, 33 mm
off); 0 by leaving it on the pad, putting it on the table, tipping it over, holding it in the air, or while it
still moves. Unintended strategies earn nothing more than they should: ignoring which way round it goes earns 0.25
unless the turn happens to be the right one; holding the cooler on the lid and never letting go scores 0.5, not
1.0; turned the wrong way, it clears the memory only with its base 10 mm or more off centre, out of the frame
(0.25); a wrong-way cooler forced flat and centred (impossible in the simulator, the fan or fins would be inside
the memory) is still 0.25 by the airflow rule; pushing it off the pad onto the table is 0. A cooler dropped from a
few centimetres that happens to land seated counts as seated: the end state is what is scored.

## One arm alone

The task needs one arm, and the demo uses only the left one. On the GPU the left arm alone scored 1.0
(`DEMO_IDLE_ARM=right`: the same run as the demo, success held at the same step) and `DEMO_IDLE_ARM=left` scored
0.0: the demo gives the right arm nothing to do, so the cooler stays on the pad. That 0 shows only that the demo is
left-armed. Whether the right arm alone could do it some other way is not measured: pointing straight down (headings
0, 45 and 90 degrees) its tool point reaches no further left than y 0.197 m anywhere over x 0.12 to 0.24 m, at heights
of 2 to 12 cm (inverse kinematics of the arm's URDF with joint limits), while the cooler's nearest face is at y 0.217 m
and a grip across its fan and fins has to be at y 0.2445 m; it reaches the grip point only tilted about 50 degrees
from vertical, which we have not tried. The task makes no claim that it needs two arms: it is about choosing the turn
that fits and setting the part down flat.

## Demo and run limit

The scripted demo takes 19.3 s (386 control steps), left arm only. It comes down over the cooler with the fingers
along the airflow (yaw 0), the tool point 50 mm above the base's bottom (pad tips 37 mm up, 17 mm above the stack's
lowest edge), over the middle of fan plus fins (9.5 mm downstream of the base's centre); the last 30 mm take 1.5 s.
It grips across fan and fins (55 mm, commanded 47 mm, over 1.5 s), lifts to 11 cm, and carries the cooler over the
processor in 2.5 s while turning the wrist a quarter turn (yaw 0 to -90 degrees: the airflow turns from +y to +x).
Across the airflow the cooler is symmetric, so it hangs square in the grip, and its centre of mass (43 mm up) is
level with the pads. It goes down to 3 cm above in 1 s, then lowers the last 30 mm over 2.5 s (12 mm/s) to 1 mm above
the lid, waits 0.5 s, opens over 1.5 s, rises and moves aside. Carried at 11 cm, the base is 26 mm over the memory
and 30 mm over the VRM heatsink; going down (with the trim below), the fan clears the memory by 10.5 mm and the
stack's sides clear the VRM heatsink by 10.5 mm and the card by 7.5 mm. On the GPU (and offline), success has held
for its 1 s at step 347, which ends the trial; without the early end the scripted policy stops at step 447. The run limit is 1,200 steps (60 s). Closest approach of the arms: TCPs
33.9 cm, link capsules 13.2 cm; smallest joint margin 13 degrees (joint 4, over the processor at 11 cm; at 12 cm it
would be 10.7).

The place point is trimmed by (+1.5, -1.5) mm (`PLACE_TRIM` in `task.py`) for the arm's sideways sag under the
0.25 kg cooler: in an earlier GPU run, a 0.20 kg block set down by the left arm at the same point landed (-1.8,
+1.6) mm off its aim. Offline (no sag) the base lands 2.1 mm off centre, inside the frame's 5 mm. On the GPU it
landed 2.3 mm off centre (x -2.3 mm, y +0.5 mm; its corners 2.4 mm inside the opening), flat on the lid (gap 0 mm,
tilt 0 degrees) with the airflow 1.2 degrees off +x; in the sampled trace it never tilted more than 1.9 degrees.
To retune, read `base_dx_mm`, `base_dy_mm`, `foot_gap_mm`, `tilt_deg` and `airflow_err_deg` in `TASK_TRACE` after
the release and move `PLACE_TRIM` by minus the offset.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. With `object_jitter` the cooler, the one loose part,
moves up to 3 cm on the pad. Reset: put the cooler back standing on its base on the pink pad, the base centred at
x 18.0 cm, y +23.5 cm, its airflow along +y (fan toward the board); everything else is fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 19 s of simulated time); the first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

A 160 x 170 mm piece of green FR-4 (or a scrap motherboard's corner) as the board; a 38 x 38 mm aluminium square
4 mm thick glued on as the processor lid, with a 3 mm black frame round a 44 x 44 mm opening; two 4 x 70 x 30 mm
aluminium strips standing on edge as the memory, an aluminium block as the VRM heatsink, a 120 x 20 x 50 mm block
as the card and a strip as the I/O shield, all glued down where `task.json` says. The cooler is a 3D-printed or
milled model of the four boxes (base, heat-pipe block, fan, fins) weighted to 0.25 kg, with no sharp fin edges.
Check a trial with a photo from above (all four base corners inside the frame's opening, the airflow within 15
degrees of the line toward the I/O shield) and feeler gauges at the base's edges (the base's centre within 2 mm of
the lid, its edges level to within 1.2 mm, which is 2 degrees across the 34 mm base).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |

The settling run (`DEMO_HOLD_ONLY=1`, 160 steps) left the cooler standing on the pad where it started: no drift, no
tilt. Dream-1 left the cooler on the pad (0.0).
