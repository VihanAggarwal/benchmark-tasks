# Clamp a loose right-angle pine joint with a trigger bar clamp

Two loose pieces of pine, cut from 1x2 furring (19 x 38 mm), lie flat on the bench as a person leaves them after a rough
dry fit of a small frame's corner: a 300 mm rail B along y and a 100 mm stub A along x, A's end 4 mm off B's face, its
outer face 8 mm short of flush with B's end and 2.5 degrees out of square. An IRWIN QUICK-GRIP Mini one-handed bar clamp,
6 in, hangs open on its side over the corner, its pads clear of both boards. The robot must square and flush the corner,
squeeze the trigger until the jaw bites with one pad on the rail's back face and the other on the stub's free end, and
let go of everything with the corner still flush and square. The instruction does not say how.

## Why it is hard

1. **Squaring and flushing a loose dry fit.** The stub is 8 mm and 2.5 degrees out. The demo's left hand pinches it across
   its 38 mm width: the two pads centre it on the flush line and square it to the bench's x axis, then press its end
   against the rail. The stub then has to stay put while the jaw comes up to it.
2. **The clamp is a loose, 0.28 kg part hung over the corner.** Its pads straddle the corner without touching either board
   (3 mm clear behind the rail, 9 mm beyond the stub), the bar passing 16 mm outside the stub's outer face and 8 mm from
   the left gripper's pad. The right hand takes its pistol grip (pads on the handle's far face and on the trigger's far
   face, 64 mm apart) and works the trigger without shifting the clamp off the corner. Its outer side lies 2.5 mm off a bench stop (a pine strip
   fixed to the bench, x 20.0 to 35.0 cm, 2 cm wide, 1.9 cm tall, y -5.85 to -3.85 cm) that touches neither board: GPU run
   2026-11-21 showed that each ratchet step pulls the base toward the hand-held jaw and turned it about the hand (21 mm and
   6 degrees), which the stop takes. (An earlier version had the hand drag
   the clamp 70 mm into place; on the GPU the clamp turned 10.5 degrees about the pinch and lagged the hand by 43 mm, so
   it starts over the corner now.)
3. **The ratchet.** Each full squeeze of the trigger (18 mm of travel at the grip, 4 N spring) moves the jaw one 3 mm step
   toward the fixed jaw and the pawl keeps it there. The trigger must come back under 30 percent before the next step
   counts, so holding it down gives one step, and half squeezes give none. There are 12 mm of slack (3 mm behind the rail,
   9 mm beyond the stub): four squeezes only take it up (the pad just touches, no force); the fifth gives 60 N, the sixth
   120 N. The hand follows the handle as the jaw advances.
4. **The force is a quantity, and a trap sits on the clamp.** The clamp force is 20 N per mm of the ratchet's pull past the
   jaw's position (est); under 40 N is not clamped. The release tab on the head (a 1 N m/rad return spring, about 9 N at its tip), if pressed past 60% while the trigger is
   not squeezed, frees the pawl
   and the force goes. Both pads must be on the boards (within 1.5 mm of the rail's back face and the stub's free end,
   over at least 15 mm of each board's width): the clamp has to be placed, not just tightened.
5. **Hands off.** The score is read with both grippers 30 mm clear of the stub, the rail and the clamp and everything
   still for 1 s: a joint held by a hand does not get the full score.

### Why a hand holds the corner square (no jig)

A corner clamp or a bench corner jig would square a frame, and people use them. For a small frame, a drawer box or a repair,
with a 6 in bar clamp, the corner is dry-fitted, checked with a try square, held square by one hand and clamped by the
other: the IRWIN QUICK-GRIP is sold as "one-handed" so that the second hand is free to hold the work. This task has no
corner fence and no corner block (only a bench stop for the clamp's outer side): the corner is squared and held by the robot's hand, and the clamp's own pad does the rest. The
pinch squares the stub because the gripper's pads are parallel plates (the stub's faces are made parallel to them) and centre
it on the gripper's axis; the left idle arm therefore leaves the corner 8 mm short of flush, which the scorer reads.

## Two arms

The left hand squares the stub and holds it until the clamp has taken up the slack and bitten; the right hand works the
trigger. With the right arm idle there is no clamp force (offline prediction 0.0); with the left arm
idle the clamp is tightened on the rough dry fit, which is 8 mm short of flush (0.25). A single hand cannot pinch the
stub and, at the same time, hold a clamp's grip and work its trigger.

## Scoring

Measured in the rail's frame (x across the rail, y along it from its end face):

| Quantity | Meaning | Tight | Loose |
|---|---|---|---|
| gap | the stub's end face to the rail's front face | 1 mm | 3 mm |
| flush | the stub's outer face at the joint in from the rail's end face (either sign) | 1.5 mm | 3 mm |
| square | the stub's far end off the perpendicular, mm over its 100 mm | 2.5 mm (1.4 degrees) | 5 mm (2.9 degrees) |

The stub must be on the rail's front side with both boards flat (tilt under 3 degrees, tops within 3 mm).

| Condition | Rule |
|---|---|
| On the work | the fixed pad's face within 1.5 mm of the rail's back face; the moving pad's face within 1.5 mm of the stub's free end; both pads over at least 15 mm of the stub's width and of the rail's; the clamp flat (under 5 degrees) |
| Clamped | on the work and clamp force of 40 N or more (20 N per mm of ratchet pull past the jaw's position), the release tab under 60% of its turn |
| Hands off | last 1 s: both finger zones 30 mm or more from the stub, the rail and the clamp |
| Still | last 1 s: nothing moved over 1 mm or turned over 1 degree |
| Fell | latched: any part's lowest point more than 20 mm below the table top |

| Outcome (the first rule that holds sets the score) | Score |
|---|---|
| A part fell off the table, or not clamped | 0 |
| Clamped, the joint outside the loose limits | 0.25 |
| Clamped, inside loose, outside tight | 0.5 |
| Clamped, tight, a hand within 3 cm or something moved in the last 1 s | 0.75 |
| Clamped, tight, hands off, still for 1 s | 1.0 |

Doing nothing scores 0; tightening the clamp in the air, off the boards, or leaving the clamp at touch scores 0.
Success: 1.0 held for 1 s. The simulator cannot lift the corner off the bench; on a real build that last check is part of the
bench method below.

### Bench method for each condition on a real build

| Condition | Bench method |
|---|---|
| gap (1 / 3 mm) | a feeler gauge between the stub's end and the rail's face |
| flush (1.5 / 3 mm) | a straightedge laid along the stub's outer face and the rail's end face, a feeler under it |
| square (2.5 / 5 mm) | a try square with its stock on the rail's face: a feeler of that thickness must not fit between its blade and the stub's far end, 100 mm out |
| boards flat | a spirit level on each; a 3 mm feeler between the tops |
| On the work | a 1.5 mm feeler at each pad face; a ruler for the 15 mm cover |
| Clamp force (40 N) | calibrate once on the same pine stock: clamp a dry-fit stub and rail pair of this stock across a luggage or bathroom scale (the pads on the boards, the boards on its platform), so the joint's own compliance is in the count, and note the squeezes past first contact that read 40 N or more (about one, est 60 N per squeeze). In a scored trial the pads must meet the 1.5 mm contact limits, so the force is not measured there: count the squeezes past first contact from video (the pawl clicks) |
| Gentle lift | after the hands are off, lift the corner by the clamp's bar 5 cm and set it down: the boards must not shift (not scored in the simulator) |
| Hands off (30 mm, 1 s, still) | a 30 mm spacer against each hand; 30 fps video of the last 1 s: no motion against a taped mark |
| Fall (20 mm) | by eye against a rule |

## Sources and estimates

| Item | Figure | Source |
|---|---|---|
| IRWIN QUICK-GRIP One-Handed Mini Bar Clamp, 6 in (maker's catalogue item number to be confirmed on the pack: believed 1964747), about USD 12 | 6 in (about 150 mm) opening, 2 in throat | the maker's listing (opening is in the name; throat est: measure the one used) |
| clamp mass 0.28 kg, jaws 26 mm tall, bar 20 x 8 mm, pistol grip 22 mm x 110 mm, trigger 14 mm thick | est | weigh and measure the one used |
| 1x2 pine furring, 8 ft, about USD 5, cut to 300 mm and 100 mm | actual 3/4 x 1-1/2 in (19.05 x 38.1 mm); pine about 500 kg/m3, so 0.108 kg and 0.036 kg | dressed lumber sizes (src); density est |
| Trigger: 18 mm of travel at the grip, 4 N | est | a spring scale on the trigger at the grip |
| Ratchet step 3 mm per full squeeze, 20 N per mm of pull past contact | est | a ruler on the jaw per squeeze; the scale between pad and stub |
| Carpenters, SOC 47-2031, over 700,000 jobs | BLS Occupational Outlook Handbook | cited |

The jaw-advance per squeeze and the force per squeeze are not published in a form this task can cite: measure both on the
clamp used (README bench table, the one-off calibration) and, if they differ, set `STEP` and `K_JAW` in `layout.py`. No third-party meshes or
assets: every shape is a box built in `parts.py`.

## Results

GPU column: this version's full run on the L40S (21 Nov; settle, demo, both one-arm runs, Dream-1). The offline prediction column is `tools/offline/check_task.py` and the same
demo stepped through this task's offline stand-in for the physics (`offline_clamp.ClampModel`: the pinch squares and flushes
the stub, the clamp stays over the corner, the pad gap turns the trigger, the ratchet steps the jaw, the pad pushes
the stub to the rail and the rail to the fixed pad). It is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle) |
| Scripted demo | below | 1.0 (success held from step 680) | 1.0 |
| One arm: `DEMO_IDLE_ARM=right` (left alone) | no clamp is moved or squeezed: no force | 0.0 | 0.0 |
| One arm: `DEMO_IDLE_ARM=left` (right alone) | clamped on the rough dry fit: 8 mm short of flush | 0.25 | 0.25 |
| `DEMO_VARIANT=four` | four squeezes: the pad just touches, no force | 0.0 | not run |
| `DEMO_VARIANT=five` | five squeezes: 60 N | 1.0 | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

The demo (about 36 s, 731 steps; run limit 1,500): the left arm goes over the stub (fingers 52 mm apart), down either side,
closes (centres and squares it) and presses it 4.5 mm in; the right arm goes over the clamp's grip (fingers 72 mm apart),
waits until 8 s, squeezes six times, the hand following the handle 3 mm per step for four steps; it opens, lifts and
parks; the left arm lets go after it (opening only to 46 mm, so its outer pad stays clear of the bar) and parks.

Run it:

    inspect-robots run --embodiment yam_arena -E task=task.py --scorer task_score -E object_jitter=0 ...
    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 ...

`DEMO_HOLD_ONLY=1` for the settle check, `DEMO_IDLE_ARM=left|right` for one arm, `TASK_TRACE=<file>` for one JSON line
per control step. A trial takes about 36 s of simulated time.

## Files

`task.py` (scene, ratchet rule, hooks, scripted demo), `layout.py` (every number), `parts.py` (shapes), `scoring.py`
(the Tracker), `offline_clamp.py` (offline stand-in), `mechkit.py` and `rigstate.py` (copied unchanged from `tools/kit`),
`tests/test_scoring.py`.

## Watch on the first GPU run

- Settle (`DEMO_HOLD_ONLY=1`): the clamp on its side must stay still and open (jaw 0), the boards must not creep.
- The right pads closing from 72 mm round the grip press the trigger to its stop: the first ratchet step must come at
  squeeze 1 (about 9.5 s); between squeezes the open pads (70 mm) must let the trigger return under 30 percent (trace
  `trigger_rad` under 0.07).
- The clamp must not move: the trace row `clamp_pose` (x mm, y mm, yaw deg) should stay at about [222, -26, 0] (jaw
  closing aside) from the first step to the last. `fixed_gap_mm` 3 and `moving_gap_mm` about 9 before the squeezes,
  `cover_a_mm` and `cover_b_mm` about 25.
- `ratchet.pull` rises 3 mm per squeeze; `jaw_mm` follows until 12 mm, then stops while `force_n` reaches 60 and 120;
  `on_the_work` turns true at squeeze 4 or 5 (the jaw pushes the stub to the rail and the rail to the fixed pad).
- The left pinch must centre the stub on y 0.019 and square it (trace `flush_mm` under 1.5, `square_mm` under 2.5) before
  the right hand starts squeezing (8 s); its outer pad passes 9 mm from the bar closing and 4 mm opening.
- Squeezing must not turn or shift the clamp (a pinch holds no torque; the clamp is never carried or lifted).
