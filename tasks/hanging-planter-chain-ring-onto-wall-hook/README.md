# Hang a planter by its ring on a wall bracket's hook, clear of the table, and let go

A 4 in galvanized pail planter (100 mm across the rim, 80 mm tall, 170 g of sand, 0.25 kg) stands on the table. Its 4 mm
wire bail is hinged on two ears at the rim and a 2 in zinc ring (40 mm hole) is linked through the bail's apex, so the
pot swings from the ring on two hinges. 430 mm in front of the arm bases a plywood wall panel stands on the table with
a steel plant bracket on its face: a plate, an 8 mm round bar projecting 150 mm toward the robot, and an upturned lip at
its free end. The robot has to take the planter by its ring, hang the ring on the bar behind the lip so the pot
hangs clear of the table, and let go. The instruction names neither the lip nor the wait for the swing to die.

## Why it is hard

1. **The ring has to go over the lip.** The bar is 8 mm round and the lip's top stands 10 mm above the bar's top, so
   bar and lip together are 18 mm tall inside a 40 mm hole. The ring's centre has to be held within a 22 mm window
   (4 mm over the bar's axis is near the middle: 12 mm clear of the bar's underside, 10 mm clear of the lip's top), moved in
   along the bar, then let down 20 mm so its upper wire rests on the bar's top. Hung, the lip is above the hole, so
   the ring cannot come back out by itself.
2. **A swinging load.** Lifted by its ring the pot (0.25 kg) hangs below the hand and swings about the bail's ears
   (fore and aft) and about the ring's link (sideways). Moving in before the swing has died knocks the ring on the lip
   or the pot on the panel.
3. **The room is tight.** The pot's bottom must end 20 mm or more above the table, the bar is only 190 mm up (a
   hung pot's base is 27.5 mm up: 7.5 mm to spare) and the panel stands 430 mm out: the pot's rim must stay 3 mm
   short of it.
4. **Hands off.** The score is read with both grippers 30 mm clear and the pot and ring still for 1 s: a held planter
   earns nothing, and a pot still swinging earns half.

## Scoring

| Condition | Rule |
|---|---|
| Ring on the bar | hole axis within 60 degrees of the bar's (x); the bar's axis crosses the ring's plane within 20 mm of its centre; that crossing between x 287 mm (just behind the lip) and 419 mm; ring centre at least 8 mm below the bar's axis (hung: 16 mm) |
| Clear | pot's lowest point 20 mm or more above the table top; rim 3 mm or more short of the wall panel (x 427 mm) |
| Plumb | pot's axis within 10 degrees of vertical |
| Hands off | last 1 s: both finger zones 30 mm or more from every part of the planter |
| Still | last 1 s: the pot's base centre and the ring's centre each move under 5 mm from where they were at the second's start; the pot and the ring each turn under 5 degrees |
| Fall (latched) | the pot's lowest point more than 20 mm below the table top at any time |

| Outcome | Score |
|---|---|
| Ring on the bar, clear, plumb, hands off, still | 1.0 |
| Ring on the bar, clear, hands off, but leaning or still swinging | 0.5 |
| Ring on the bar, hands off, but the pot within 20 mm of the table or 3 mm of the wall | 0.25 |
| Anything else, or not hands off, or the pot ever fell | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0 (the ring is on its own bail).

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Ring on the bar | side photo with a rule: the bar through the hole, the ring behind the lip (287 mm or more from the line through the two arm bases), the ring's centre 8 mm or more under the bar's axis |
| Clear | steel rule from the table to the pot's lowest edge (20 mm); a 3 mm spacer between the rim and the panel |
| Plumb | front and side photos against a card ruled with a 10 mm grid and level lines: the rim's line within 10 degrees of level, measured with a protractor on the photo (nothing is added to the hanging planter) |
| Still | 30 fps video of the last second against the same card: the pot's base and the ring move under 5 mm, and the rim's line turns under 5 degrees between the second's first and last frames and any frame between |
| Hands off | a 30 mm spacer block against each hand, the same video |
| Fall (latched) | the pot on the floor, or 20 mm below the table top by eye and a rule |

The pot's tilt and its origin are linked: the pot turns about its ears 76 mm above its base, so 4.5 degrees of swing move
its base 6 mm. The 5 mm limit is reached before the 5 degree one; both are kept so the check is the same on a bench.

## Results

Run on an L40S (Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 1000-step limit), one trial each (GPU job of 17 Nov,
after the hand's entry was moved clear of the bracket). The offline prediction column (`tools/offline/check_task.py`, with this task's offline stand-in for the physics,
`task.OfflineModel`) is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run) |
| Scripted demo | below | 1.0 (held from step 488) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the whole demo | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | does nothing | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=front` | the ring let down 60 mm short of the lip and let go: the planter falls | 0.0 | not run |
| `DEMO_VARIANT=hold` | the whole demo but the fingers never open | 0.0 | not run |

The task does not claim two arms: one hand carries the planter, as a person would. The right arm waits at its ready pose.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The left gripper closes along y
on the ring's two vertical flats (42 mm commanded across 50 mm), so the grip is on the ring and the bail is not touched.

| Time (s) | Left gripper |
|---|---|
| 0 to 3 | from the ready pose down to x 225 mm in the arm's own column (y 305 mm), then across to y 100 mm, short of the bar's lip (x 270 mm) so the open pads never touch the bracket |
| 3 to 8.5 | over the ring at x 250, y 100 mm, down beside its flats (pad tips 8 mm below the ring's centre), close |
| 8.5 to 10.5 | settle, lift 47.5 mm (the pot's base clears the table) |
| 10.5 to 14 | carry 100 mm along y to the hook's line (y 200 mm) in front of the lip (ring at x 250 mm), wait for the swing |
| 14 to 17 | move the ring 60 mm in along x over the lip to x 310 mm (the ring's centre 4 mm over the bar's axis), wait |
| 17 to 19.5 | let the ring down 24 mm (20 mm to the bar's top, 4 mm past it, the arm trails), settle: the bar takes the weight |
| 19.5 to 23.5 | open (1.5 s), back off 140 mm along x with the ring between the open pads, hold |

Offline: worst IK 0.09 mm; left joint 4 margin 4.5 degrees (at the carry height); the arms' capsules stay 21 cm apart;
the arm at its ready pose clears the bar and the panel by more than 5 cm. Success is predicted held from step 488 of 1000 (the demo is under half the limit).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Hang the planter on the wall hook by its ring so it hangs clear of the table, and let go of it." \
  --scorer task_score --max-steps 1000
```

Runtime: a 1000-step trial takes about 1.5 to 3 minutes of wall-clock on the L40S including start-up (the demo itself is
470 steps, 23.5 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run starts
from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the ring's and pot's poses
in millimetres, the bar crossing, the pot's lowest point, tilt and the let-go and stillness readings, both joints and
their speeds, both TCPs and openings.

## Simplifications

- Rigid bodies. The pot is a solid cylinder with a rim (a real pail is hollow and holds 170 g of sand), the ring an
  octagon of eight 5 mm segments, the lip a straight post, the bar a round cylinder.
- No chain. A ring with three chains cannot be one simulated articulation (the pot, the chains and the ring close a loop), so
  the hanger is a rigid wire bail on the rim's ears and a ring linked at its apex: the pot swings about two axes from the
  ring, as it would from a short chain. Each hinge has a light spring toward upright (0.03 and 0.01 N m/rad) and damping
  (0.015 and 0.006 N m s/rad): the springs stand in for the pinched ears and the wire binding that keep a real bail and ring standing up;
  they are small against the pot's own weight (about 0.09 N m/rad).
- The simulator's default friction (1.2 static, 1.0 dynamic) on every part.
- The bracket's tip is a straight post, not a curl; the plate's screws are not modelled (the plate and panel are fixed).
- No software rule touches the planter. The scorer reads its pose; the arms are read through `rigstate.py` for the let-go check.

## Sources

None needed: all dimensions are ordinary hardware sizes (4 in pail planter, 2 in welded ring, 8 mm bracket bar, 20 mm
plywood), stated in `task.json`. Estimates (said in fidelity): the pot's height (80 mm), the bail's rise (48 mm), the
ear height (76 mm), the lip's height (10 mm), the springs and damping, frictions.

## Real build and reset

Store-bought parts (any brand), each with a check before use:

| Part | What to buy | Acceptance check |
|---|---|---|
| Planter | a 4 in (100 mm) galvanized steel pail with a wire bail handle, sold as a small hanging bucket planter | rim 95 to 105 mm across, 75 to 85 mm tall; bail wire 3 to 5 mm; ear axis 70 to 80 mm above the base |
| Load | 170 g of dry play sand in a zip bag, set in the pail | pail, sand and bail 0.25 kg plus or minus 10 g on a kitchen scale |
| Ring | a 2 in welded steel ring, 3/16 in (5 mm) wire, zinc plated | hole 38 to 42 mm, outside 48 to 52 mm, about 22 g |
| Ring's link | the ring passed over the bail's top bar and bound to it with two turns of 1 mm soft iron wire, so it stands upright on the apex and rocks a little sideways (the sim's light spring) | ring stands upright on its own with the pail on the bench; its centre 140 to 153 mm above the bench |
| Bail | the pail's own bail; if it falls over, pinch each ear a little with pliers until it stands upright by friction | bail stands upright on its own and still swings freely under the pail's weight |
| Bracket | a 6 in (150 mm) steel plant bracket with an 8 mm round bar and an upturned tip, wall plate about 50 x 80 mm | bar 7 to 9 mm; tip's top 8 to 12 mm above the bar's top; bar level within 2 degrees |
| Wall | a 20 mm plywood panel, 500 x 320 mm, clamped upright to the bench with two corner brackets, the bracket screwed to its face with two wood screws | face 430 mm in front of the arm bases; bar's axis 190 mm above the bench at y 200 mm |

Tools: a steel rule, a 3 mm and a 30 mm spacer, a card ruled with a 10 mm grid and level lines stood behind the planter, a phone camera (30 fps video) and an on-screen protractor.
Reset (about 1 minute): the pail on its mark (x 250 mm, y 100 mm, taped outline), the bail upright, the ring upright on
the apex with its hole facing the wall; refill any spilled sand to 0.25 kg; check the bracket is tight.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the planter, panel and bracket as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
