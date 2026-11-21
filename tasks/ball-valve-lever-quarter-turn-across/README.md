# Shut a cold-supply ball valve a quarter turn to its stop, leaving the hot one open

A utility alcove stands on the station table, open above and in front: a 1/2 in plywood back panel (the wall) 400 mm
in front of the arm bases, 876 mm wide, with a side panel at each end, and through it two 1/2 in copper supply stubs,
70 mm above the floor and 203 mm (8 in) apart. Each stub carries a 1/2 in lead-free brass full-port ball valve
with sweat ends, modelled on the Apollo 94ALF-203-01A, with a 130 mm flat lever and a round grip, both levers open
(lying along the pipe, pointing at the robot). The left valve has a blue grip: the cold supply. The right one has a
red grip: the hot supply. The robot must shut the cold valve, swinging its lever a quarter turn clockwise (seen from
above) to its stop so it stands across the pipe, leave the hot valve open, and let go. The instruction names neither
the direction nor the angle.

Instruction: "Shut off the cold supply at its blue-gripped ball valve, leaving the red-gripped hot valve open, and
let go of it."

Category: Societal relatability. Shutting off one supply line is a household chore: before a washing machine or
laundry tub fill hose, a faucet or a toilet fill valve is changed, or when a hose bursts, someone turns the
quarter-turn ball valve on that line, as on a water heater or a house's main shutoff; a plumber does the same before
any repair on a live line. Here only the cold line of a laundry tub is shut and the hot line stays in service.

## Why it is hard

1. **Which valve.** Two identical valves sit 203 mm apart; only the grip colour says which is the cold supply. Closing
   the hot one, or both, scores 0 (the hot lever may not turn more than 10 degrees at any time).
2. **A firm, steady push through 90 degrees.** The valve turns against 0.20 N m of friction: 2.0 N at the middle of the
   grip, with no spring to help and nothing holding the lever once the hand lets go. The arms' limit is about 5 N and
   they trail a command by about 2 mm per newton, so a push that is a few millimetres off the lever's arc stalls or
   slips past it.
3. **The whole quarter turn.** A lever left at 60 degrees still passes water: the tiers are 15 / 45 / 88 degrees and
   only a lever against its stop scores 1.0.
4. **Let go.** The score is read with both grippers 30 mm clear of the lever and valve body and the lever still for
   1 s: a hand left pressing on the stop earns nothing.
5. **Close quarters.** The valves stand 130 mm in front of the back panel, the pipe and clips pass 5 to 16 mm under
   the lever, and the other valve's lever is 203 mm away. The gripper has the space above and in front, and the
   lever sweeps a quarter circle 130 mm across that it must follow without catching the clips or the stop plate.

## Scoring

| Condition | Rule |
|---|---|
| A | the cold lever's angle from open, seen from above, clockwise positive: 0 along the pipe, 90 against the stop |
| Hot untouched | the hot lever never turns more than 10 degrees from open (latched) |
| Frame still | the cabinet, a stub or a valve body never moves more than 5 mm (latched) |
| Hands off | last 1 s: both finger zones 30 mm or more from the cold lever and valve body on every step |
| Still | over the last 1 s A changes by under 2 degrees |

| Outcome | Score |
|---|---|
| A from 88 degrees up to the stop (read up to 95), hands off and still, hot untouched | 1.0 |
| A from 45 up to 88 | 0.5 |
| A from 15 up to 45 | 0.25 |
| A under 15 | 0 |
| Not hands off or not still, hot lever turned over 10 degrees, frame moved 5 mm, A past 95 | 0 |

Every angle from 0 to 95 degrees falls in exactly one tier (a test sweeps them). The valve cannot be moved by the
robot in the simulation (the frame is fixed); the 5 mm check is for the real trial.

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| A (15 / 45 / 88 degrees) | protractor card under the cold stem, read from the camera straight above it |
| Hot lever (10 degrees) | the same, with a second camera and card over the hot stem, every frame |
| Frame moved (5 mm) | the overhead video; a mm scale by the pencil marks on the panel and stubs, every frame |
| Hands off (30 mm) | every frame of the last second, overhead and side views; 30 mm in either view passes |
| Still (2 degrees) | A over the last second's 30 frames on the card (2 degrees is about 6 mm at the marks) |
| Optional, not scored | pressurised line: the cold stub's open end stops dripping within 10 s, the hot one runs on |

The cameras: three, at 30 fps for the whole run. One straight above each stem axis, 600 mm up, over a printed
protractor card lying on the table, centred on that axis, with 1 degree marks from 160 to 220 mm radius labelled
0, 15, 45, 88 and 90. Seen from a point on the axis, a lever's centre line through the axis keeps its true angle,
so the lever's height above the card does not bias the reading. The third camera is level with the levers on the +y
side, square to the pipes, with a 5 mm grid board beyond the valves. A view never shows a gap larger than the real
one, so a 30 mm gap in either view proves 30 mm in space.

Success: 1.0 held for 1 s. Doing nothing scores 0.

## Results

GPU runs: L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 600-step limit, one trial each; rows marked not run
were not run on the GPU. The offline column (`tools/offline/check_task.py` with this task's offline stand-in for the
physics, `task.OfflineModel`) is an offline prediction, not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 260) | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the right arm has no plan | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the demo is the left arm's | 1.0 | 1.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=short` | the sweep ends near 30 degrees | 0.25 | not run |
| `DEMO_VARIANT=half` | the sweep ends near 60 degrees | 0.5 | not run |
| `DEMO_VARIANT=hold_on` | the whole sweep, no let-go | 0.0 | not run |

One arm is enough here: the demo uses the left arm and the right one stays at its ready pose, so the task does not
claim to need two. The right arm reaches the hot valve, so the other mistake (shutting the hot one) is a real option
for either arm.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator.

| Time (s) | Left gripper (fingers closed into a block) |
|---|---|
| 0 to 2.0 | over the cold valve, beside the grip on its trailing side (8 mm short of it), closing |
| 2.0 to 5.0 | down to the lever's height: pad tips 2 mm under the lever's centre line |
| 5.0 to 6.0 | 16 mm toward the lever: the block face ends 8 mm inside the grip |
| 6.0 to 9.5 | seven moves round the stem at the grip's middle radius (yaw turning with it), 4 degrees past the stop |
| 9.5 to 10.6 | settle against the stop, back off 33 mm |
| 10.6 to 13.1 | straight up, aside (closed) |

The arm trails about 2 mm per newton and the valve needs 2.0 N, so the block is commanded 8 mm into the grip: the
arm's own give turns that into about 3.5 N, and the stop takes the overtravel. Offline: worst IK 0.08 mm; left
joint 4 margin 6 degrees (at the hover); arm-arm clearance 19.9 cm; the block's path stays more than 150 mm from the
panels and the hot valve. Offline prediction: success held from step 260, inside half the 600-step limit.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 600
```

Runtime: a 600-step trial takes about 1.5 to 3 minutes of wall-clock on the L40S including start-up (the demo itself
is 262 steps, 13 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every
run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: both levers' angles, the
frame's movement, the let-go and stillness readings, both TCPs, pad tips and openings.

## Simplifications

- Against the Apollo 94ALF-203-01A (maker's sheet SS1186): the body's 52 mm (2.05 in) end to end and the 5/8 in
  sockets for 1/2 in copper match. Its stock handle (3.54 in, 90 mm, white grip, top 1.84 in, 47 mm, over the pipe's
  axis) is shorter and higher than the modelled lever, so the real build cuts it down and bolts on an extension bar
  (Real build below) that matches the model from 63 mm out to the tip: 130 mm from the stem, 97 mm up, grip middle
  at 100 mm. Within 63 mm of the stem the real bar rises to the stock handle's stub (about 120 mm up); the model's
  bar is straight at 97 mm there. The hand works only at the grip. No stock 1/2 in sweat ball valve with a 130 mm
  lever was found (handles listed at distributors run about 80 to 90 mm), hence the extension.
- Rigid bodies. Each lever is a hinge on a fixed stem with 0.20 N m joint friction and stops at 0 and 90 degrees
  (the stop plate's tabs are drawn but the joint limit is the stop). The ball, seat, packing and water are not
  modelled. The vinyl grip is a cylinder on the bar. The tabs, escutcheons and clips are fixed boxes.
- 0.20 N m is an estimate for a cleaned, lubricated valve on a drained line; many real 1/2 in valves are stiffer. It
  was chosen so the arms (about 5 N at most) can supply it with margin. A real valve's torque must be measured and
  set (reset notes); if it is much higher the task needs a longer lever or a different grip.
- The cabinet, stubs and valve bodies are fixed, so they cannot be torn out in the simulation; the 5 mm check is a
  real-trial check.
- Friction 0.5 static / 0.4 dynamic on every fixed and lever surface, the vinyl grip included (est); the gripper
  pads are the station's 1.2 / 1.0.

## Sources

- Apollo 94ALF-A series submittal sheet SS1186 (11/21), 94ALF-203-01A, 1/2 in solder ends: B 2.05 in end to end,
  C 1.84 in pipe axis to top, D 3.54 in handle, E 0.63 in socket; steel plated handle with PVC grip; 600 CWP.
  Distributor listings (Nov 2026), part 94ALF-203-01A (also 94ALF20301A): about US$11 to 31 each, e.g.
  https://www.supplyhouse.com/Apollo-Valves-94ALF20301A-1-2-Sweat-Full-Port-Ball-Valve-Lead-Free and
  https://www.zoro.com/apollo-12-sweat-brass-ball-valve-inline-94alf-203-01a/i/G6177726/
- 1/2 in copper tube, 5/8 in OD: ASTM B88 (type L).
- Hot and cold stubs 8 in apart: common under-sink practice (est).
- Quarter-turn valves shut clockwise seen from the handle.

Estimates (stated in fidelity): the valve's torque, the body size, the stub spacing, frictions.

## Real build and reset

A 1/2 in plywood panel on a frame 400 mm from a mark on the bench; two 1/2 in copper stubs soldered into 1/2 in
female fittings behind the panel at 70 mm height, 203 mm apart; two Apollo 94ALF-203-01A ball valves soldered on, stem
axes 130 mm in front of the panel; four plastic saddle clips on wood standoff posts; two 18 mm plywood side panels.

Each lever, to match the model:

1. Cut the stock handle off 40 mm from the stem axis (hacksaw, file the end). Keep its nut, hub and the valve's own
   stops at 0 and 90 degrees. Drill two 4.5 mm holes in the stub at 25 and 35 mm from the stem axis.
2. Cut 120 mm of 6061 aluminium flat bar, 18 x 6 mm. Bend it down 23 mm with two 45 degree bends so that the part
   that will lie 40 to 63 mm from the stem is the slope. Drill it to match the stub and bolt it on top of the stub
   with two M4 bolts and nuts, its start 20 mm from the stem.
3. Check with a steel rule: from 63 mm out to the tip, 130 mm from the stem axis, the bar's centre line is 97 mm
   above the table (27 mm over the pipe's axis). Adjust the bends if not.
4. Grip: wind blue (cold) or red (hot) vinyl electrical tape over the outer 60 mm of the bar until it is 21 mm round
   (calipers). Its middle is 100 mm from the stem.
5. Weigh the bar, bolts and tape: about 50 g.

Reset: both levers open, along the pipe, pointing at the robot; each bar's centre line still 97 mm up; the panel not
shifted against the pencil marks; each valve's turning force read with a spring scale at the middle of the
grip before each session (target 2.0 N, 0.20 N m) and recorded with the trial; a new valve is stiffer, so it is
worked in on a drained line or replaced.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; the cabinet and levers, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the cabinet as an articulation with two hinges; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
