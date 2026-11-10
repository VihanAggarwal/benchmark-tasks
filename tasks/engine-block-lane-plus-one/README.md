# Block a crash lane: engine angled upstream across two lanes, ambulance downstream

Firefighters and EMTs work highway crashes a few metres from moving traffic, and responders are struck at incident
scenes every year. US traffic incident management training (the FHWA's National TIM Responder Training; NFPA 1091
covers traffic control incident management personnel) teaches the first fire apparatus to "block": it parks
upstream of the crash, angled across the involved lane plus one more ("lane + 1"), leaving a buffer before the work
area, with its pump panel on the downstream side, where the pump operator is shielded by the apparatus. Ambulances
park downstream of the crash, inside the protected area, so patients are loaded through the rear doors inside that
protected space. Buffer distances differ between department SOPs; this task states its own: 4 to 10 m.

Here a 1:80 one-way, two-lane highway runs across the table, traffic coming from the robot's left. A blue car has
crashed in the near lane (lane 1), so lane + 1 is both lanes. A red fire engine waits upstream in the far lane,
heading downstream with its silver pump panel toward the median; a white ambulance waits downstream on the near
shoulder, facing the wrong way (rear doors away from the car). The score is what a company officer checks with a
glance and a tape, and what a top photo shows: the engine angled 30 to 60 degrees across both lanes, its pump-panel
side toward the car, 50 to 125 mm (4 to 10 m) short of it; the ambulance straight in a lane, rear doors toward the
car, 50 to 125 mm beyond it; neither touching the car or the other vehicle.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two vehicles' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each vehicle as one rigid body of coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets, the demo plan, its carry paths and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (the road strips and lines, the lane arrows, the crashed car), the standard
`task_score` scorer, and one helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no front,
back or pump-panel side. `mechkit.py` (imported by `task.py`) builds the fire engine from five boxes (body,
windscreen, light bar, pump panel and the silver patch above it) and the ambulance from six (body in three bands
with its red stripe, rear doors, windscreen, light bar), each as one loose rigid body through a hook on
`yam_arena.scene`, and reports its pose (the frame origin at the middle of the vehicle's base, and its orientation)
to the scorer. It changes nothing else the embodiment does, and it stops the run with an error if a vehicle was not
built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook
patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner) follows the one first
shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms at once, one vehicle each): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A crashed car is stopped in the near lane of this one-way, two-lane highway (1:80 scale; traffic comes from your left, as the arrows show). Park the fire engine upstream of the car, angled 30 to 60 degrees to the lanes so it blocks both lanes, its silver pump-panel side facing the car, 50 to 125 mm (4 to 10 m) short of it. Park the ambulance downstream, straight in either lane, rear doors toward the car, 50 to 125 mm beyond it. Neither may touch the car or the other vehicle." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/engine-block-lane-plus-one/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "A crashed car is stopped in the near lane of this one-way, two-lane highway (1:80 scale; traffic comes from your left, as the arrows show). Park the fire engine upstream of the car, angled 30 to 60 degrees to the lanes so it blocks both lanes, its silver pump-panel side facing the car, 50 to 125 mm (4 to 10 m) short of it. Park the ambulance downstream, straight in either lane, rear doors toward the car, 50 to 125 mm beyond it. Neither may touch the car or the other vehicle." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/engine-block-lane-plus-one/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `success`, `zero_reason`, `valid_state`,
`to_car_mm`, `between_mm`; for each vehicle `credit`, `why`, `tilt_deg`, `lowest_mm`, `path_mm`, `origin_mm`; for
the engine `buffer_mm`, `x_lo_mm`, `x_hi_mm`, `crash_lane_cover_mm`, `angle_deg`, `panel_y`; for the ambulance
`gap_mm`, `centre_x_mm`, `front_off_deg`; then the raw `origins`, `quats`, `low_z` and `vels`). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset (mechkit's reset counter, or the simulator putting both vehicles back on their marks in one step); the run
rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the vehicles' poses at the last
step. A vehicle's **outline** is its footprint seen from above (the fire engine 125 x 32 mm with its pump panel, the
ambulance 88 x 30 mm; for a tilted vehicle, the shadow of its whole box). Distances along the road are measured in
y from the car's ends: its upstream end at y -0.05 cm, its downstream end at y -5.95 cm.

Seen from above (x up the page, away from the arms; y to the left; traffic flows to the right, toward -y):

```
  y +45 (upstream)                 y 0                                 y -45 (downstream)
  ==================== grass median, x 29.5 to 36.0 ==============================================
  - far shoulder - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
  ===== yellow edge line, x 27.75 =================================================================
  <-- lane 2   [engine] start,                      .                                      <--
               centre x 25.55, y +33.0,             .
  - - - - lane line, x 23.25 - - -  /==\  - - - - - . - - - - - - - - - - - - - - - - - - - - - -
                                   /    \  engine   . [car]  y -0.05 .. -5.95   [ambulance]
  <-- lane 1                       \    /  angled   .  x 21.1          place: centre x 21.0,   <--
                                    \==/  45 deg    .                  y -19.0, front to -y
  ===== white edge line, x 18.75 ==================================================================
  - near shoulder - - - - - - - - - - - - - - - - - - - - - - - - - - -  [ambulance] start, x 17.0,
  ===== grass verge, x 10.0 to 15.0 ===================================      y -37.0, front to +y

          left arm base (y +30.5)                x 0                right arm base (y -30.5)
```

**Parked**: tilted 10 degrees or less, its lowest point within 3 mm of the road surface, its middle (of its
overall length, on its centreline) over the road (x 15.0 to 29.5 cm), and still (moved under 5 mm over the last 0.5
s, turns counted at its corners: positions, not the simulator's velocities, which chatter while a part is gripped).

| Outcome | Score |
|---|---|
| Either vehicle on its side or roof (tilted more than 45 degrees) | 0 for the trial |
| Either vehicle's outline within 2 mm of the car's outline or of the other vehicle's (seen from above) | 0 for the trial |
| Fire engine: parked; whole outline upstream of the car, buffer (its most downstream point to the car's upstream end) 50 to 125 mm; long axis 30 to 60 degrees to the lanes; outline spanning both lanes (x 19.05 cm or less to x 27.45 cm or more); pump-panel side facing downstream (its outward direction toward -y) | 0.5 |
| ... parked, upstream with a buffer of 250 mm or less, outline covering at least 28 mm of lane 1's 45 mm width, but not all of the above | 0.25 |
| ... anything else (left in lane 2, downstream of the car, more than 250 mm back, lifted, moving or leaning) | 0 |
| Ambulance: parked; whole outline downstream of the car, gap (the car's downstream end to its most upstream point) 50 to 125 mm; its middle in lane 1 or 2 (x 18.75 to 27.75 cm); front within 15 degrees of downstream (rear doors toward the car) | 0.5 |
| ... parked, downstream with a gap of 250 mm or less, its middle in lane 1 or 2, but not all of the above | 0.25 |
| ... anything else (left on the shoulder, upstream, more than 250 mm on, lifted, moving or leaning) | 0 |
| Total: the engine's credit plus the ambulance's | 0, 0.25, 0.5, 0.75, 1.0 |

Examples: the start 0; both set right 1.0; engine right, ambulance in the lane nose to the car 0.75; ambulance right
and the engine square across the road (75 degrees), or angled right but with its panel toward the traffic, or straight
in lane 1 behind the car (a lane block, not lane + 1) 0.75; engine right, ambulance left on the shoulder 0.5; both
0.25 cases 0.5; one 0.25 case, the other vehicle not moved 0.25; either vehicle only shoved sideways into a lane from
its start 0 (both start more than 250 mm from the car: the engine's buffer is 268 mm, the ambulance's gap 266.5 mm, so
each has to be brought along the road as well to earn anything); either vehicle pushed against the car 0. The trial
ends early once 1.0 has held for 3 s. A simulator state with a non-finite number or a zero quaternion scores 0 for
that step and leaves the tracker as it was.

Why the bands: angled 30 degrees to the lanes the engine's outline spans 90 mm across the road, and at 60 degrees 124
mm; both lanes less 3 mm at the outer edges is 84 mm, so the whole angle band can block both lanes, and only an
angled engine can: straight in a lane it covers 32 mm, and square across (beyond 60 degrees) is outside the band. Of
the four 45-degree headings two put the pump panel (on the engine's left) downstream, toward the car; either is
accepted. A straight engine in lane 1 covers 32 mm of its 45 mm and earns the partial credit of a single-lane block.

Unintended strategies earn nothing more than they should, as the tests show: the engine stood on its end or rolled
over, or either vehicle on top of the other or of the car, zeroes the trial; the engine pushed against the car to
fix the buffer zeroes it; both vehicles swapped (the ambulance angled upstream, the engine straight downstream) earn
nothing; the ambulance angled across the lanes downstream earns only 0.25; the engine parked angled downstream of
the car, or alongside it (even 1 mm past its upstream end), earns 0; a vehicle held still 2 mm above the road counts
as parked (the scorer sees only the vehicles; `task.json` fidelity says so).

## One arm alone

On the GPU the scripted demo with one arm idle scored 0.5 either way, as the offline kinematic check (`--one-arm`)
predicted: with the left arm idle the ambulance was set right (0.5) and the fire engine stayed in lane 2, 268 mm back
(0); with the right arm idle the engine was set right (0.5) and the ambulance stayed on the shoulder, 266.5 mm on
(0). One arm alone, picking and placing, earns half the score; the full score needs both vehicles placed.

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips at tool-point heights
13 to 60 mm, every yaw in 15 degree steps within joint 6's +-120 degrees, over a 5 x 3 grid of points on each
vehicle's outline): neither arm reaches any point of the other vehicle where it starts. The engine's nearest point
is 57 cm to the left of the right arm's base and the ambulance's 63 cm to the right of the left arm's; straight down
(tool point 26 mm up, yaws every 30 degrees), the right arm reaches y +13 to +15 cm over the far lane, about 12 cm
short of the engine, and the left arm y -17 to -18.5 cm over the near shoulder, about 14 cm short of the ambulance.
So with pick and place one arm alone earns 0.5, and the two arms work at the same time, each on the vehicle on its
own side. We have not tried a single arm using its own vehicle as a tool to shove the other one into place before
setting its own (that could add the other vehicle's 0.25 at most), or a tilted grip, and do not claim either is
impossible.

## Demo and run limit

The scripted demo takes 21.6 s (432 control steps), both arms at once. Each arm goes over its vehicle with the
fingers open (94 mm), comes down beside it, takes the last 30 mm over 1.5 s and grips across the vehicle's width
(commanded 8 mm narrower than the part, closing over 1.5 s): the left arm grips the engine 10 mm behind its middle
at yaw +90 degrees, its 27 mm pads 11.5 mm clear of the pump panel, the tool point 25 mm above the engine's base
(pads on the body from 12 mm up to its roof, the gripper 17 mm over the light bar); the right arm grips the
ambulance at its middle at yaw -90 degrees, 24 mm up. Each lifts its vehicle 30 mm and carries it over 4 s in two
segments, turning the wrist: the engine turns 45 degrees clockwise seen from above (heading -90 to -135 degrees,
yaw +90 to +45), the ambulance half a turn (yaw -90 through 0 to +90; two quarter turns, so each segment's
interpolation is unambiguous). Then it settles 0.5 s, lowers the last 28 mm over 2 s, settles, opens with the
vehicle 2 mm above the road, rises straight up and parks. Joint 6 stays between about -82 and +69 degrees.

The places: the engine with its middle at x 23.2 cm, y +14.25 cm, heading -135 degrees (front toward the near
shoulder and downstream, pump panel facing x+ and y-): buffer 87.2 mm (37 mm inside each end of the 50-125
mm band), outline from x 17.68 to 28.79 cm (13.7 and 13.4 mm beyond the 19.05 and 27.45 cm marks), angle 45 degrees
(15 degrees inside each end). The ambulance at x 21.0 cm (lane 1's centre), y -19.0 cm, front toward -y: gap 86.5
mm (36.5 mm inside the band's near end, 38.5 mm inside its far end). The engine's outline stays 88.9 mm and the
ambulance's 86.5 mm from the car; on the way, both stay more than 80 mm from the car and more than 20 cm from each
other (tested). On the GPU the demo left the engine with a buffer of 86.2 mm, at 44.3 degrees, its outline from x
17.60 to 28.62 cm, and the ambulance with a gap of 86.4 mm, its middle at x 20.93 cm, its front 0.5 degrees off
downstream; in the sampled trace neither vehicle tilted more than 0.7 degrees or came within 86 mm of the car, and the
trial ended at step 380 once 1.0 had held for 3 s.

Offline (kinematic check): worst inverse-kinematics error 0.10 mm; smallest joint margin 15.5 degrees (joint 4);
arm links at least 16.9 cm apart (capsule proxy), tool points at least 34.1 cm apart; pad tips never lower than 12
mm above the table. Success has held for its 3 s at step 379; the scripted policy stops at step 493. The run limit
is 1,200 steps (1 minute), which leaves a policy room for slower carries, a set-down to regrip for a bigger turn, or
a second try.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the fire engine in lane 2 with its
middle at x 25.55 cm, y +33.0 cm, heading downstream (front toward -y, pump panel toward the median), and the
ambulance on the near shoulder with its middle at x 17.0 cm, y -37.0 cm, front toward +y (rear doors away from the
car). The road, lines, arrows and crashed car are fixed. Nothing is consumed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 22 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 90 x 26 cm board laid on the table from x 10.0 to 36.0 cm: 5 cm of green felt or artificial turf for the verge, 14.5
cm of dark grey card for the road with 3 mm white and yellow tape lines centred at x 18.75 and 27.75 cm and 37.5 mm
white dashes every 15 cm centred at x 23.25 cm, 6.5 cm of turf for the median, and four white tape arrows. The crashed
car is a 22 x 56 x 18 mm blue block with a 18 x 30 x 9 mm cabin, glued down turned 10 degrees. The fire engine is a
123 x 31 x 40 mm red-painted wood block with a 27 x 10 x 2 mm black windscreen on its front (125 mm overall), a 24 x 6
x 3 mm white strip for the light bar, a 15 x 28 x 1 mm aluminium plate on its left side 15 to 30 mm forward of its
middle for the pump panel and a 15 x 10 mm piece of aluminium tape on the roof edge above it, weighted to 80 g in all;
the ambulance an 85 x 30 x 35 mm white block with a 4 mm red tape stripe 20 mm up, a 26 x 26 x 1 mm dark grey door
panel on the back and a 26 x 10 x 2 mm black windscreen on the front (88 mm overall) and a 24 x 6 x 3 mm red roof
strip at the front, 50 g in all (1:80 die-cast models of the same size work too). Everything is light and blunt, so
trials are safe. Score from a top photo: a tape along the lane line from the car's ends for the buffer and the gap, a
protractor against the lane line for the engine's angle, a ruler across the lanes for the coverage, and the panel
patch and light bars for the sides and fronts.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
