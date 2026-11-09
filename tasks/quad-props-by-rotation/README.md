# Fit each quadcopter motor with a propeller whose leading edges lead the way it turns

On a quadcopter, neighbouring motors turn opposite ways: two take clockwise (CW) propellers and two
counter-clockwise (CCW) ones. A propeller's hand is read from its blades: the leading edge, the higher and thicker
edge of each pitched blade, must lead in the motor's direction of turn, and no turn changes a CW propeller into a
CCW one. A propeller of the wrong hand, or one fitted upside down, pushes air the wrong way and the drone flips over
at take-off, so drone and robotics technicians check the hands at every propeller change. Here a 5-inch quad frame
without propellers is fixed on the midline, a red arrow beside each motor shows which way it turns (front left and
rear right clockwise, front right and rear left counter-clockwise), and six propellers lie on two foam mats, each
blade's leading edge marked by a raised white strip. Each mat holds one propeller of each hand for its own side's
two motors and a spare of the hand that the other side's rear motor must not have. The score is what the
technician checks before the first flight: which motors carry a propeller standing upright and centred on them, and
whether each is of that motor's hand.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the propellers' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each propeller as one rigid body of five shapes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets, the task file and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and cylinders, the standard `task_score` scorer, and one helper module next to
`task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a propeller's hand is in its
shape. `mechkit.py` (imported by `task.py`) builds each propeller from a cylinder (the grey hub), two boxes (the
black blades) and two thin boxes (the white leading-edge strips) as one loose rigid body through a hook on
`yam_arena.scene`, and reports its pose (base-frame origin, the centre of the hub's bottom face, and orientation) to
the scorer. It changes nothing else the embodiment does, and it stops the run with an error if a propeller was not
built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook
patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner) follows the one first
shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit a propeller on each of the drone's four motors, standing upright and centred on the motor's top (hub up, as they lie on the mats). The red arrow beside each motor shows which way that motor turns: use a propeller whose leading edges, the raised white strips along one side of each blade, lead in that direction. Two of the six propellers are spares." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/quad-props-by-rotation/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Fit a propeller on each of the drone's four motors, standing upright and centred on the motor's top (hub up, as they lie on the mats). The red arrow beside each motor shows which way that motor turns: use a propeller whose leading edges, the raised white strips along one side of each blade, lead in that direction. Two of the six propellers are spares." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/quad-props-by-rotation/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `zero_reason`, `motors_fitted`, `fitted` (motor ->
propellers), `wrong`, `upside_down`, and per propeller in `props`: `hand`, `motor`, `status` (`fitted`,
`wrong_hand`, `upside_down`, `still_moving`, `not_on_a_motor`), `nearest`, `centre_off_mm`, `seat_mm`, `tilt_deg`,
`at_rest`, `moved_mm`; then every propeller's `origins` and `quats` and the state's `source`). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset (mechkit's reset counter, or the simulator putting every propeller back on its mark in one step); the run
rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from where the six propellers are at the
last step. A propeller **stands on** a motor when its hub's axis is within 8 degrees of vertical, the centre of the
hub's bottom face is within 6 mm of the motor's axis seen from above, and that face is within 3 mm of the motor's
top. It stands on it **upside down** when its hub's top face meets the same 6 mm and 3 mm (blades uppermost). A
propeller tilted more than 8 degrees, off-centre by more than 6 mm, resting on another propeller, lying on the frame
or on a mat, held higher or off the table stands on no motor. A propeller's hand is fixed by its shape: seen from
above with the blades along x, a CW propeller has the white strip of its +x blade on the -y side and that of its -x
blade on the +y side, the edges that lead when it turns clockwise, and a CCW propeller is the mirror image; no turn
about the vertical changes one into the other. A motor is **fitted** when a propeller of its own hand stands
upright on it and is at rest: the centres of its hub's bottom and top faces and both blade tips each stayed within
2 mm of where they are now over the last 0.5 s (positions, not the simulator's velocities, which chatter while a part
is gripped; the tips catch a propeller turning in place).

| Outcome | Score |
|---|---|
| Any propeller of the wrong hand stands on a motor, or any propeller stands on a motor upside down (at rest or not): that drone flips at take-off | 0 |
| Otherwise, for each motor fitted with a propeller of its own hand | + 0.25 |
| No motor fitted (propellers on the mats, lying elsewhere, held above a motor, off-centre or tilted) | 0 |
| One motor fitted | 0.25 |
| Two motors fitted | 0.5 |
| Three motors fitted, the fourth empty | 0.75 |
| All four fitted | 1.0 |

Any propeller of the right hand counts, from either mat: the three CW propellers are the same part, and so are the
three CCW ones. The spares may stay on the mats or go anywhere off the motors. Only the propellers' poses are read,
whether or not a gripper still holds one (see `fidelity`). The trial ends early once 1.0 has held for 2.5 s (not the
default 1 s, so that it ends after the last propellers have been let go); otherwise it is scored as it stands at the
run limit. A simulator state with a non-finite number, a zero quaternion or a propeller not built as a compound scores
0 for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing more than they should, as the tests show: turning a propeller about the vertical
never changes its hand (any yaw counts the same); flipping a CCW propeller upside down, so that seen from below it
looks like a CW one, scores 0 on any motor; a propeller held just above its motor (more than 3 mm), leaning on it,
off its axis or lying on an arm counts for nothing; a wrong-hand propeller set on top of a fitted one stands on no
motor and adds nothing; two propellers on one motor count once; a propeller turning in place is not at rest; a
single arm's spare carried to the far side's rear motor (the only far motor it reaches holding a hub straight down)
scores 0.

## One arm alone

In simulation (one trial each, see 'Results in simulation'), the scripted demo with one arm idle scores 0.5 with
either arm idle, as our offline kinematic check predicted. With the right arm idle (`DEMO_IDLE_ARM=right`) the left
arm fits its CW propeller on the front left motor and its CCW one on the rear left: 0.5. With the left arm idle the
right arm fits the rear right (CW) and front right (CCW) motors: 0.5. In both runs the other side's motors stay
empty and the spares stay on their mats.

Reach (inverse kinematics of the arm's URDF with its link collision proxy): gripping a hub straight down, the left
arm reaches its own mat, both left motors and the rear right motor (x 22 cm, y -10 cm), but not the front right motor
(33 mm short at the place height) or the right mat (its nearest propeller, at y -23.0 cm, 35 mm short); the right arm
is its mirror image. Its own mat holds one CW and one CCW propeller for its own two motors and a spare of the hand
that the reachable far rear motor must not have (left: a spare CCW, while the rear right motor needs CW). So a single
arm holding each propeller by its hub, straight down, can fit at most two motors, 0.5, and putting its spare on the
far rear motor scores 0. Other grips reach further. Holding a propeller straight down by the end of a blade (18 mm
wide), 55 to 60 mm out from the hub's axis, an arm can kinematically set the hub on the far front motor, and its
spare is of that motor's hand: 0.75 in principle. Leaning its gripper 15 to 30 degrees toward the far side, an arm
also reaches the far front motor (20 degrees) and the far mat's two nearer propellers (15 and 30 degrees; one of each
hand), so a single arm could in principle fetch the missing hand and fit all four. The scripted one-arm runs try
neither, and we do not claim that one arm cannot score 1.0. Both arms are used naturally: each fits the two motors on
its own side from its own mat, at the same time.

## Demo and run limit

The scripted demo takes 36.5 s (730 control steps). Each arm grips a propeller's hub straight down with the fingers
closing along y (yaw 0) across the 22 mm hub, so the blades, which lie along x, pass between the fingers' planes;
the fingers open to 46 mm on the way down (their outer faces 13 mm from the next hub on the mat), close to a
commanded 14 mm over 1.5 s with the tool point 26 mm above the hub's bottom (pad tips 13 mm up the hub, 6 mm above
the strips' tops), lift to 12 cm (the hub's bottom 94 mm up, 22 mm over a fitted propeller's hub), carry it over the
motor, come down to 30 mm above the place point and lower the last 30 mm over 2 s, so the hub's bottom ends 2 mm
above the motor's top; after 0.5 s still they let go over 1.5 s (opening to 40 mm) and go straight up. The left arm
fits the front left motor while the right arm fits the rear right one (tool points 256 mm apart), then the left
fits the rear left while the right fits the front right; both then move aside. Every propeller keeps its blades along
x, so the two left propellers' tips end 34 mm apart.

Offline (kinematic check): every waypoint reachable straight down, worst inverse-kinematics error under 0.1 mm;
smallest joint margin 11.7 degrees (joint 4, both arms); closest tool points 25.6 cm and arm links at least 8.5 cm
apart (capsule proxy), at the second pair of places; pad tips never lower than 16 mm above the table. Success has
held for its 2.5 s at step 711 offline; in simulation the demo's trial ended with 1.0 at step 709. The scripted
policy stops at step 791. The run limit is 1,800 steps (90 s). The
propellers weigh 20 g, so the arms' sag under them is negligible (well under 1 mm); the 6 mm centring band and 3 mm
seat band leave room for the release.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the six propellers back hub up on the mats,
blades along x, hubs at x 20.0 cm: left mat y +23.0 cm CW, +28.5 cm CCW, +34.0 cm CCW; right mat y -23.0 cm CCW,
-28.5 cm CW, -34.0 cm CW. The frame, motors, arrows and mats are fixed. Nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 36.5 s of simulated time, a full Dream-1
trial 90 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A frame cut from 8 mm black plywood or carbon sheet (four 16 mm arms on the diagonals of a 160 x 200 mm rectangle and
a 70 x 70 x 12 mm centre plate), screwed to the table centred 30 cm in front of the arm bases, with four flat-topped
28 mm aluminium rods 30 mm tall standing on the arms' ends (no shaft: the printed hubs have no bore); four red arrows
cut from 5 mm wide, 1 mm red plastic strip, stuck to the table round the motors' outer sides; two 140 x 180 x 3 mm
grey foam mats. Six propellers printed in glass-filled nylon (or PLA) to the stated shape (22 mm hub 34 mm tall,
126 mm span, 18 mm blades with a 6 x 3 mm raised white strip on each leading edge), about 20 g each, three of each
hand. Nothing is powered, so trials are safe. Score from a top
photo: which propeller stands on which motor, on which side of each blade its white strip lies, and a rule across
the motor's top for the centring.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
