# Lift a dish drainer's folded plate rack until it locks, then stand two plates in it

Anyone who washes up by hand ends by standing the clean plates on edge in a drainer's plate rack so they drain. Here
the drainer is a Joseph Joseph Flip-Up draining board (319 x 400 mm), stored as it usually is with its plate rack
folded flat on the deck. Two washed Corelle 6.75 in side plates lean on the sink's back upstand beside it, because the
rack was still folded. The robot must lift the rack until it locks upright, then stand each plate on edge in a
different slot between the prongs, and let go. The instruction names no arm, grip, order or slot.

## Why it is hard

1. **The rack has to be raised first, and only a locked rack holds.** The rack is a hinge bar with seven prongs that
   lies flat. Pinched at a prong's tip, it swings up about the bar on a 87 mm arc; within 5 degrees of upright its lock
   holds it at 90 degrees, anywhere else it sinks flat again (a hand let go at 75 degrees leaves it flat). A plate set
   against an unlocked rack falls flat with it. The hand must follow the hinge's arc, not a straight line, and get off
   the prong without pulling it back down.
2. **A plate can only be gripped by its top rim.** The pads pinch the 14 mm flat rim from above, so the plate hangs
   about 160 mm below the hand with its 22 mm deep dished body behind the pinch. It is lifted straight up off the
   upstand, carried beside the rack 14 mm over the deck, slid in along its own plane between two prongs (5 mm to spare
   on each side), set on the deck, tipped toward the robot about its foot while still held, and let go onto the near
   prong.
3. **Order and two arms.** In the demo the right arm raises the rack (the folded prong tips point toward it) and the
   left arm moves the plates (they stand in its own column, beyond the right arm's reach). The left hand has to wait,
   plate held up, until the rack is locked and the right hand is clear before it slides in.
4. **Hands off, still.** The score is read with both grippers 30 mm clear of both plates and the rack, everything
   still for 1 s: a plate held in a slot or a hand on the rack earns nothing, and a plate moving down at 0.9 m/s
   (41 mm of free fall) scores 0.

## Scoring

| Condition | Rule |
|---|---|
| Rack up | 85 degrees or more from flat |
| Placed | rack up, and all of the next four rows |
| Low | lowest point within 5 mm of the deck |
| Between prongs | every point within 10 mm of its lowest point between one slot's two prongs (0.5 mm allowed) |
| Tilt | plane within 25 degrees of vertical |
| Across the prongs | centre line 30 to 70 mm beyond the prong row (y 110 to 150 mm) |
| Upright | plane within 40 degrees of vertical (standing or leaning, not lying down) |
| Hands off | last 1 s: finger zones 30 mm or more from both plates and the rack; no point of either plate or of the rack moves 2 mm or more (a plate's top rim and the prong tips move most) |
| Fell (latched) | downward speed 0.9 m/s at a step (41 mm of free fall; any fall of 100 mm or more is caught) |
| Off the counter (latched) | a plate's lowest point went 20 mm below the counter |

| Outcome | Score |
|---|---|
| Both plates placed, in different slots | 1.0 |
| One slot holds a placed plate (or both are placed in the same slot) and every plate is upright | 0.5 |
| No plate placed and every plate upright (the rack raised and locked, nothing in it); or one placed and the other lying down | 0.25 |
| Anything else, the rack not up, a fall, or not hands off | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. The path, the plate order and which arm does what are not scored. Success: 1.0
held for 1 s. Doing nothing scores 0 (the rack is folded).

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Rack up (85 deg) | phone level on a prong; the lock engaged |
| Low (5 mm) | 5 mm feeler gauge under the plate's lowest point |
| Between prongs (0.5 mm) | by eye from above against the two prongs |
| Tilt (25 / 40 deg) | phone level on the plate's face |
| Across the prongs (30 to 70 mm) | ruler from the prong row to the plate's centre |
| Hands off (30 mm, 1 s) | 30 mm spacer block against each hand; 30 fps video of the last 1 s |
| Still (2 mm, 1 s) | a sticker dot on each plate's top rim and on both end prongs' tips, filmed at 30 fps from the front and the side beside a mm rule: no dot moves 2 mm over the last 1 s |
| Fall (0.9 m/s) | 30 fps video against a rule: a plate moving down 30 mm or more between two frames |
| Off the counter (20 mm) | by eye |

## Results

The geometry, the demo and the scorer changed on 19 Nov 2026 (redesign: folding rack with a lock, real dished
plates, the draining board, sink and tap). GPU figures below are from 22 Nov job 20261122T060000 (Isaac Lab, YAM
arena) of this version, after a fix to how the plate colliders were built (the first runs of this version spawned
the plates flat); earlier GPU runs were of a different design and are not reported here.

The offline prediction column comes from `tools/offline/check_task.py` with this task's offline stand-in for the
physics (`task.OfflineModel`), not from a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (settle) | both arms hold still | 0.0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 1044) | 1.0 (from step 1036 of 2400) |
| `DEMO_IDLE_ARM=right` (left alone) | plates carried, rack never raised: plates fall flat | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the rack raised and locked, nothing in it | 0.25 | 0.25 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=one_plate` | rack raised, only plate 1 placed | 0.5 | not run |
| `DEMO_VARIANT=rack_only` | rack raised, no plate moved | 0.25 | not run |
| `DEMO_VARIANT=unlocked` | rack let go at 75 degrees: it falls flat | 0.0 | not run |

The task claims two arms: neither arm alone scores over 0.25, offline or on the GPU (0.0 and 0.25).

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator.

| Time (s) | Right gripper (fingers close along x, across a prong) | Left gripper (fingers close along x, across a plate) |
|---|---|---|
| 0 to 6 | over the folded rack's middle prong tip, down round it, close | over plate 1's rim, down round it, close: the pads stand it up |
| 6 to 12.1 | swing the rack up along the hinge's arc to 92 degrees (on the stop) | lift plate 1 straight up 26 mm (to 9 s), then hold it up |
| 12.1 to 17.6 | let go (lock holds at 90), back 50 mm along -y, up, aside | wait, holding plate 1 up |
| 15 to 27.5 | parked | carry along x beside slot 2, slide in along y, lower to the deck, draw back 20 mm (it tips), let go, up |
| 27.5 to 49 | parked | the same for plate 2 into slot 5 |
| 49 to 51.5 | parked | aside, clear of both plates and the rack |

The demo is 1030 steps (51.5 s) and the policy stops at step 1091; the run limit is 2400 steps (120 s at 20 Hz).
Offline: worst IK 0.10 mm; minimum joint margins left joint 4 2.5 degrees, left joint 6 4.4 degrees, right joint 4
1.6 degrees; arm-to-arm capsule clearance 3.6 cm while plate 1 waits.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 2400
```

Seed: nothing is random; `object_jitter=0` fixes the start layout. For Dream-1 use `--policy dream`.
`TASK_TRACE=trace.jsonl` writes one line per control step (each plate's slot, lowest point, tilt, y, vertical speed,
the rack's angle, both TCPs and openings).

## Simplifications

- Rigid bodies. Each plate is the real dished profile built from boxes: a flat well 110 mm across, a sloping wall and
  a flat rim 14 mm wide, 4 mm walls (profile estimated from the listed 171 mm diameter and 22 mm depth).
- The rack's lock is a rule on its hinge: within 5 degrees of upright a stiff joint target holds it at 90 degrees,
  anywhere else the target sits just below the rack's angle so it sinks flat, as the real rack falls without its
  catch. Prong size (8 mm square, 95 mm long), pitch (40 mm), the deck height (12 mm) and the hinge height are
  estimates: the maker lists the board's outer size and that the rack holds up to 6 plates, not its slot sizes.
- The draining board is fixed (the real one stands on non-slip feet). The upstand, splashback, sink rim and tap are
  fixed; the sink bowl is a dark patch inside its rim.
- Side plates, not dinner plates: a dinner plate hanging from its rim would need the hand higher than the left arm's
  reach allows.
- Friction 0.50 / 0.40 on the plates (est); plates dry.

## Sources and estimates

| Figure | Value | Source |
|---|---|---|
| Draining board | 319 x 400 mm, 24 mm folded; rack lifts, locks, holds up to 6 dinner plates | Joseph Joseph Flip-Up draining board (85139), maker's and retailers' listings |
| Plate | 171 mm (6.75 in) across, 22 mm (0.88 in) deep | Corelle Livingware Winter Frost White bread and butter plate, retailer listings |
| Plate mass | 0.130 kg (4.6 oz) | Corelle listing |
| Plate profile (well, wall, rim, 4 mm walls) | est | from the two listed sizes |
| Prongs, pitch, deck and hinge heights, lock capture (5 deg), stop (2 deg past) | est | chosen to fit the listed size and 6 slots |
| Upstand 75 mm, splashback, sink rim, tap 250 mm | est | typical kitchen fittings |
| Friction | est | |

No third-party meshes, scans or scenes.

## Real build and reset

A Joseph Joseph Flip-Up draining board, two Corelle 6.75 in side plates, a 75 mm steel strip for the upstand, a
phone level, a 5 mm feeler gauge, a ruler, a mm rule and sticker dots for the stillness video. The prong size, pitch
and hinge height in `layout.py` are estimates: measure the real board's and set them to match before a real trial.
Reset: fold the rack flat on the deck, prongs toward the sink (-y); lean
each plate back on the upstand's top front edge, 12 degrees from vertical, face toward the robot, its lowest point at
x 317 mm: plate 1 centred y +265 mm, plate 2 centred y +450 mm. Wipe the plates dry; replace a chipped plate.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; the parts, the lock rule, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound bodies and the hinged rack; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
