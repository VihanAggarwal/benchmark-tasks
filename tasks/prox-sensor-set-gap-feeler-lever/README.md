# Set a proximity sensor's gap to 1.0 mm with a feeler leaf, then lock its cam clamp

A replacement Turck BI2-M12-AP6X-H1141 inductive sensor (M12 x 1 barrel, 52 mm, 31 g, rated switching distance 2 mm) has
just been slid into its open clamp, its face 12.0 mm short of a steel trip dog parked at its end stop. The clamp is a split
aluminium block (our own part) closed by a KIPP K0006.1501406X20 adjustable cam lever (A 70.4 mm, red), stood up open. An 8 in (203 mm) feeler leaf, 1.00 mm thick and 12.7 mm wide, in a handle, lies on its edge in a rack tray to
the right. The robot must set the sensing gap to 1.00 mm (half the rated distance) using the leaf, shut the clamp lever,
and put the leaf back in its tray, with both hands off. The instruction names neither the pinch nor the order.

## Why it is hard

1. **The gap is set by contact, not by position.** The arms trail their commands by 3 to 7 mm, so a face placed "about
   1 mm" from the dog misses by several mm. The leaf is the gauge: it must stand upright between the face and the dog
   while the sensor is pushed up to it, so the face stops on its 1.00 mm. The scorer checks the moment the lever comes
   shut: the leaf must be standing upright in the gap then, on both faces to within 0.25 mm. A gap that is right without that (set
   by eye, or the leaf taken out before the lock, or loose in a wider gap) is capped at 0.5.
2. **The leaf goes in from the side.** It is a 1 mm blade in a 12 mm gap that is open at the sides: the blade, 203 mm
   long, is slid in along y until its tip is over the face's round and the dog's width, with 8 mm to the face and 3 mm to
   the dog. The hand on its handle stays 20 cm from the hand at the sensor's rear.
3. **The lock is one way.** Shut with the face further out and it stays there; shut with the face against the dog and the
   first machine cycle drives the dog into it (a latched zero). The cam lever must be pressed through a 70 degree arc
   while the leaf still stands in the gap, without knocking it: a closed gripper pressing the tip down follows the tip as
   it moves out (four waypoints).
4. **The leaf comes out against a lock.** With the lever shut the sensor holds where it stood against the leaf; the leaf
   is drawn out sideways (a residual squeeze of 0.1 mm is 0.2 N), lifted and put back in its tray.
5. **Hands off.** The score is read with both grippers 30 mm clear and every part still for 1 s.

## Scoring

| Condition | Rule |
|---|---|
| Gap g | the dog's face minus the sensor's face: 12.0 mm less the slide |
| Lever shut | within 5 degrees of its stop |
| Good gap | 1.00 mm within 0.25 mm |
| In range | 0.50 to 1.62 mm (the maker's secured operating distance, 0.81 x Sn) |
| Locked on the leaf | on the step the lever last came shut (into its last 5 degrees after being 20 degrees or more open): the blade upright and square (10 degrees) between the face and the dog, over 6 mm or more of the face's 12 mm round, each flat face within 0.25 mm of the face it meets, and g within 0.25 mm of the leaf's 1.00 mm |
| Leaf out | every point of the leaf 30 mm or more from the gap's centre line (top view) |
| In tray | every point over the tray's 60 x 270 mm footprint, lowest point within 10 mm of the tray's floor |
| Hands off | last 1 s: finger zones 30 mm or more from the sensor, the lever and the leaf; slide under 0.2 mm, lever under 1 degree, leaf under 1 mm and 0.3 degrees |
| Crash (latched) | the lever shut with g under 0.30 mm at any step |

| Outcome | Score |
|---|---|
| Lever shut, good gap, locked on the leaf, leaf out and in the tray | 1.0 |
| The same with the leaf out but not in the tray | 0.75 |
| Lever shut, good gap, leaf still in the gap or not locked on it; or lever shut, gap in range (not good), leaf out | 0.5 |
| Lever shut, gap 0.30 to 4.0 mm otherwise; or good gap with the lever open | 0.25 |
| Anything else, a crash, or not hands off | 0 |

Bench check for each condition on a real build (the sim reads joints and poses; a person reads these):

| Condition | Bench method |
|---|---|
| Good gap (0.25 mm) | go and no-go leaves for the 1.00 +- 0.25 mm window: a 0.75 mm leaf enters, a 1.25 mm leaf does not |
| Lock holds | a 3 N pull on the sensor moves it under 0.05 mm on a real clamp (the sim's lock is 2 N per mm, declared) |
| Lever shut | protractor, 5 degrees |
| Locked on the leaf | video: the leaf stands square in the gap, held up by the drag of both faces after the hand lets go, as the lever comes down |
| Leaf out, in tray | a 30 mm spacer block and a look from above; a rule from the tray's floor (10 mm) |
| Hands off | a 30 mm spacer block against each hand; 30 fps video over the last 1 s; a dial indicator on the sensor body (0.2 mm) |
| Crash | a 0.30 mm leaf cannot enter the shut gap |

Success: 1.0 held for 1 s. Doing nothing scores 0.

## Results

GPU runs (L40S, Isaac Lab-Arena, object jitter 0, 20 Hz control, one trial each) next to the offline predictions
(`tools/offline/check_task.py` with this task's stand-in for the physics, `task.OfflineModel`, not a measurement).
The DEMO_VARIANT rows were not run on the GPU.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Scripted demo | below | 1.0 (held from step 1143) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | pushes the sensor, no leaf | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | leaf and lever, no push | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | not run |
| `DEMO_VARIANT=short` | pushed 9.0 mm: 3.0 mm gap, lever shut | 0.25 | not run |
| `DEMO_VARIANT=crash` | no leaf, pushed to the dog, lever shut | 0.0 (latched) | not run |
| `DEMO_VARIANT=leaf_stays` | leaf left standing in the gap | 0.5 | not run |
| `DEMO_VARIANT=leaf_table` | leaf let go on bare table | 0.75 | not run |

With either arm idle the demo scores 0 on the GPU: one hand pushes the sensor while the other places the leaf and shuts
the lever.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The push length is computed from
the layout (12.0 mm gap, 1.00 mm leaf), not read from any sensor.

| Time (s) | Left gripper (fingers close along x) | Right gripper |
|---|---|---|
| 0 to 9.5 | waits aside | open over the leaf's handle in its tray, down, grip, lift 14 mm, carry across to the gap's x, slide the blade in 127 mm, lower it to the table |
| 9.5 to 17.0 | over the air behind the sensor, closes into a block, down to the connector's height | hold the leaf upright |
| 17.0 to 18.4 | push 15 mm at 30 mm/s: the sensor goes 11.0 mm, the leaf is pushed to the dog and pinched | hold |
| 18.4 to 22.4 | hold the push | let go (1.5 s), up and back, close into a block |
| 22.4 to 25.9 | back off 45 mm, up, aside | wait, then (from 23.9) round behind the sensor's rear, clear of the open lever |
| 25.9 to 36 | parked | across and in over the clamp, inboard of the open lever's face (with the arm's measured 31 mm shortfall toward +y commanded away), sweep the closed block out onto the face and press it down through 70, 50, 32, 15, 0 degrees, a last press on the shut lever, hold, up |
| 36 to 56 | parked | over the leaf's handle at the gap, grip, pull the blade out sideways, lift, carry, lower, let go, aside |

Offline: worst IK 0.10 mm; left joint 4 margin 5.3 degrees (the 150 mm hover), right 13.6 degrees; the TCPs at least 19 cm
apart while both hands are low, the arms' capsules 1.8 cm apart at the pinch. The demo is 1122 steps (56.1 s) of the
2400-step limit.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: a 2400-step trial takes about 4 to 7 minutes of wall-clock on the L40S including start-up (the demo itself is
1122 steps, 56 s of simulated time; estimate, not measured). Seed: nothing is random; `object_jitter=0` fixes the start
layout, so every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the gap, the lever, the
leaf's pose, the lock-on-the-leaf and crash latches, both TCPs and openings, and a `hist` of the last 3 s of those numbers.

## Simplifications

- Rigid bodies. The sensor and the lever are links of one mechanism (a fixed clamp, a slide and a hinge); links do not
  collide with the block, so the barrel passes through the drawn bore on the joint alone.
- Two declared rules (`task._rig_rule`, a mechkit `on_step` callback). (1) The slide is a position drive, 2 N per mm,
  whose target follows the sensor each step while the lever is open (a light drag, about 0.2 N at 5 mm/s) and freezes at the
  position it had when the lever came within 5 degrees of its stop (the clamp). A real clamp is far stiffer: a 3 N pull
  moves the sim's sensor 1.5 mm, a real one under 0.05 mm. (2) The cam lever has two notches (open and shut) that the
  drive snaps to within 0.35 rad: the cam's over-centre feel, as a detent. The cam's clamping force and the cap's flex
  are not modelled: the lock is the slide's drive freezing.
- The clamp block is our own part, fully dimensioned in `layout.py` and the setup: 6061 aluminium, 22 mm along the bore x
  45 mm across x 25 mm tall; a 12.2 mm bore along x centred 12.5 mm up, split at the axis plane into a base and a 12.5 mm
  cap; the cap pivots on an M6 x 30 socket screw (left just loose) on the +y side and is pulled down on the -y side by the
  KIPP cam lever's M6 x 20 stud, threaded into the base, its thrust piece set so the cam clamps as the lever comes flat;
  the base takes two M5 screws into the machine bed. The sim draws the bore as a 12.4 mm square and the cap and base as one
  fixed body.
- The sensor's two jam nuts are left off (the clamp holds the barrel); the M12 connector is a 14 mm round, 12 mm long
  block (est); no cable.
- The dog (24 x 24 x 4 mm St37) and the 40 x 40 x 34 mm carriage are estimates; the carriage is fixed at its end stop, not
  on a spring slide, and no machine cycle is run: the crash is read from the lever and the gap.
- The leaf: 1.00 mm x 12.7 mm (the width of Starrett 667 series feeler stock, 1/2 in) x 203 mm, steel; its handle (22 x 40 x
  24 mm, ABS) and the 30 g mass are estimates; a real set's leaf would be a stock thickness and a common length. Friction 0.6 / 0.5.

## Sources

- Industrial machinery mechanics, machinery maintenance workers and millwrights, about 538,300 US jobs in 2024: BLS Occupational
  Outlook Handbook,
  https://www.bls.gov/ooh/installation-maintenance-and-repair/industrial-machinery-mechanics-and-maintenance-workers-and-millwrights.htm
- Sensor: Turck BI2-M12-AP6X-H1141, ID 46065 (M12 x 1, flush, 2 mm, secured operating distance 0.81 x Sn, 52 mm, 31 g),
  https://www.turck.us/en/product/46065 and its datasheet https://www.turck.us/datasheet/_us/edb_46065_eng_us.pdf
- Cam lever: KIPP K0006.1501406X20, cam lever adjustable, size 1, M6 x 20, A 70.4 mm, B 21.5 mm, aluminium red RAL 3003
  (KIPP K0006 datasheet,
  https://www.aftfasteners.com/content/K0006-Datasheet-17079-Cam-levers-aluminum-adjustable-with-external-thread-plastic-thrust-washer-and-steel-or-stainless-steel-stud-enUS.pdf,
  and distributor listings of the K0006 size 1 family)
- Feeler stock 1/2 in wide, 12 in long strips: Starrett 667 series (retailer listing),
  https://dillonsupply.com/Product/starrett-667-1
- The 1.00 mm target is half the rated distance (the setup sheet's choice here): the datasheet's secured operating distance
  is 0.81 x Sn, so 0.5 x Sn leaves margin for wear and slide wobble.

Estimates (stated in fidelity): the dog and carriage, the leaf's length and handle, the connector's size, friction, the
cam lever's handle section (12 x 6 mm) and mass (20 g), the tray (60 x 270 x 8 mm). The clamp block is our own design.

## Real build and reset

A sensor in the split block above, closed by the KIPP cam lever, on a steel plate; a stop block or a second plate as the dog (a 24 mm square of steel); a 1.00 mm feeler
leaf (or a 1 mm shim) 12.7 mm wide, 200 mm long, with a handle; a tray. Reset: the cam lever up, the sensor slid back until a
12.0 mm spacer fits between its face and the dog; the leaf in its tray. A real feeler leaf should be checked for a kink between
trials.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the drive rules, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies and a mechanism; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
