# Seat a baby bottle in a twin warmer's free well beside a warming one, then start it

A two-well bottle warmer fixture (110 x 320 x 60 mm, cream, two 72 mm wells 50 mm deep, one push button per
well on its top, at the warmer's end; built as in "Real build and reset") stands on a nursery counter against a tiled splashback. A child's 260 mL feeding bottle
(a 62 mm across, 135 mm tall stand-in, pink collar) already stands warming in the +y well. A second child's bottle (blue collar)
waits on a bottle tray at (x 270, y -250), to the right of the warmer. The robot takes it from the tray, stands it in the
free well, pushes that well's own button, and leaves everything else as it was. The instruction names the job, not
the method: "Put the bottle from the tray into the warmer's free well and start that well. Leave the bottle
already warming undisturbed."

## Why it is hard

1. **Tight, tall, close.** The well is only 5 mm wider each side than the bottle, 50 mm deep, and the other bottle is
   48 mm away. The open gripper's pads are 94 mm apart and 8 mm thick, so they cannot go into the well beside the
   bottle: it has to be carried gripped by its upper body (pads 90 to 128 mm up its side) and lowered straight down.
2. **The neighbour is watched for the whole trial.** A bottle knocked 3 mm, nudged up 2 mm, tilted 3 degrees or whose
   own button is touched caps the score at 0.5, even if it is put back. Knocked out of its well or over: 0.
3. **The natural last step is part of the job.** The well is started by pushing its own button down (3 mm of 5 mm of
   travel) after the bottle is seated. As on the real warmer, one push turns the well on and the next turns it off; a
   push while the bottle is not seated (a dry start) caps the score at 0.5.
4. **Hands off.** The score is read with both grippers 30 mm clear of both bottles and the warmer and everything
   still for 1 s.

## Scoring

| Condition | Rule |
|---|---|
| Fell (latched 0) | either bottle's lowest point more than 50 mm below the counter's top |
| Spilled (latched 0) | either bottle's axis tilts more than 30 degrees |
| Knocked out (latched 0) | the neighbour's centre more than 12 mm off its well's axis, or its base more than 30 mm up |
| Disturbed (latched, caps) | the neighbour's centre ever 3 mm from where it stood, base ever 2 mm up, axis ever over 3 degrees off vertical, or its own button ever pushed 1 mm |
| Push | the free well's button 3 mm or more in; counted again only after it is back out to 1 mm |
| Dry start (latched, caps) | a push while the loose bottle is not seated |
| Started | an odd number of pushes: the well is on |
| Seated | centre within 6 mm of the free well's axis, base within 2 mm of the floor's top, axis within 3 degrees |
| Resting | not seated: base on the warmer's top (lowest point within 3 mm of it, centre over the outline), or leaning in the free well (centre within 36 mm of its axis, lowest point between 2 mm under the floor's top and the warmer's top) |
| Hands off | last 1 s: finger zones 30 mm or more from both bottles and the warmer; each bottle moves under 5 mm and turns under 5 degrees (visible motion); a button is not read for stillness, its state is the pushes counted above |

| Outcome | Score |
|---|---|
| Seated, started, hands off | 1.0 |
| Seated, started, not yet hands off | 0.75 |
| Seated, not started | 0.5 |
| Resting on the warmer or leaning in the free well, hands off and still (not held hovering) | 0.25 |
| Anything else, or any 0 latch | 0 |
| Neighbour disturbed, or a dry start | at most 0.5 if started, else at most 0.25 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0.

The still limits were relaxed from 1 mm, 0.3 degrees and 0.5 mm of button motion to the visible-motion rule above (5 mm,
5 degrees, no button reading), because the old limits cannot be checked in a 30 fps video. The GPU demo (run
20261116T132033) scored 1.0 under the stricter limits, so the relaxed ones cannot lower it; its GPU score stands.

Bench method for each threshold on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Fall 50 mm | rule from the counter to the bottle's base |
| Spill 30 degrees, axis 3 degrees | angle gauges |
| Knocked out | by eye against the well; 12 mm with a rule from the well's centre mark |
| Seated centre 6 mm | rule from the well wall to the bottle on opposite sides along two perpendicular lines; half the difference of each opposite pair is the centre offset (at most 6 mm) |
| Base 2 mm | from above: a rule from the warmer's top face to a line drawn round the bottle at its fully seated height (within 2 mm) |
| Neighbour 3 mm, 2 mm, 3 degrees | pencil ring round its base, a rule, an angle gauge; a top-down 30 fps video against a grid card, and a side-view 30 fps video with a tape mark on the neighbour 2 mm above the well rim, for any move or lift in between |
| Button 3 mm, 1 mm | a rule against its travel; the video for the order (bottle seated before the push) |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s: no bottle visibly moves or turns against a taped mark on the counter (5 mm, 5 degrees) |

## Results

GPU runs: L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, one trial each; rows marked not run were not
run on the GPU. The settle run (nothing commanded) scored 0.0. The offline prediction column
(`tools/offline/check_task.py`,
with this task's offline stand-in for the physics, `task.OfflineModel`: it carries the bottle with a gripper closed on
it, drops it onto the well's floor, the warmer's top or the tray, and pushes a button by a closed pad block; no
contacts) is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0 | 0.0 |
| Scripted demo | below | 1.0 (held from step 588; demo 560 steps) | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the same demo: it uses only the right arm | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | nothing moves; the tray is out of the left arm's reach | 0 | 0.0 |
| `DEMO_VARIANT=no_start` | bottle seated, button never pushed | 0.5 | not run |
| `DEMO_VARIANT=press_first` | button pushed first (a dry start), then the bottle seated | 0.5 | not run |
| `DEMO_VARIANT=short_press` | the pad tips stop with the button 2 mm down | 0.5 | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

The task is one-armed by design and does not claim to need two arms: the tray, the free well and its button all lie
in the right arm's reach. This matches how the act is usually done at home: one-handed, often with a baby held on the
other arm.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The right gripper points
straight down, fingers closing along x.

| Time (s) | Right gripper |
|---|---|
| 0 to 6.5 | over the bottle (pad tips 19 mm over its cap), down round it, grip it 103 mm over its base |
| 6.5 to 11 | lift 70 mm (base 10 mm over the warmer's top), carry over the free well |
| 11 to 16 | down to 25 mm over the floor, slowly into the well, settle, let go |
| 16 to 19 | straight up between the pads, aside over the tray, closing the fingers into a block |
| 19 to 23 | closed block over the free well's button (x 230, y -143), pad tips 30 mm over it, settle, down to 10 mm over it |
| 23 to 25.5 | slowly down, the pad tips commanded 9 mm under the button's rest top (3 mm under the warmer's top, so the button goes to its 5 mm stop); hold 1 s |
| 25.5 to 28 | up off the button, up to 140 mm over the warmer's top |

Offline: worst IK 0.08 mm; least joint margin 8.4 degrees (joint 4, at the carry height); lowest pad tips 57 mm
(on the button). The gripper opens or closes over at least 1.5 s each time. The scripted policy stops at 621
steps; the run limit is 1500 steps (75 s).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Put the bottle from the tray into the warmer's free well and start that well. \
Leave the bottle already warming undisturbed." --scorer task_score --max-steps 1500
```

Runtime: a trial takes about 2 to 4 minutes of wall-clock on the L40S including start-up (the demo itself is 560
steps, 28 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so no seed is
needed. For Dream-1 use `--policy dream`. `DEMO_HOLD_ONLY=1` is the settle check, `DEMO_IDLE_ARM` (left or right) drops
one arm's plan, `DEMO_VARIANT` runs a counterfactual, `TASK_TRACE=trace.jsonl` writes one line per control step (both
bottles' poses and tilts, the seat offset, the buttons' travel, the latches, the hands' gap, both TCPs and openings).

## What to watch on the GPU

- The grip: 0.20 kg held by friction on a 62 mm body with 54 mm commanded; `bottle_tilt_deg` near 0 through the carry;
  the cap's top is 32 mm over the TCP, just under the gripper body.
- The seat: `bottle_off_well_mm` under 5, `bottle_low_mm` at 10 after the let-go; `neighbour_moved_mm` stays 0 and
  `disturbed.D` stays null (the fingers pass 55 mm from the neighbour's wall in y).
- The button is pushed from above by the closed pad block (as the conveyor task's caps were on the GPU): `btn_b_mm` reaching 3 or more (the stop is 5), then back to 0 after the lift, `started_step` set after the seat; spring and damping
  (250 N/m, 2 N s/m) are estimates, as is the sprung slider's behaviour at the stop.
- `tight_fit` on the warmer's 16 wall boxes per well: no jitter of the seated bottle.

## Simplifications

- Rigid bodies. The wells are 16 turned boxes each (72 mm across the flats), not round; the bottle is a solid body
  (body, collar, cap hood) without a nipple, the milk a fixed mass. Nothing is heated or lit.
- The bottle (62 x 135 mm) is smaller than sold 260 mL bottles: the Philips Avent Natural Response 260 mL
  (SCY903/01) is listed at 7.10 x 7.10 x 16.6 cm. The 5 mm each-side clearance is what
  matters: a real build with a larger bottle keeps it by boring each well 10 mm wider than the bottle used.
- The warmer is a generic fixture, not a sold model; its buttons sit on its top, one near each end.
- The warmer is fixed (a real one has non-slip feet; tape it down in a trial). The splashback is a 200 mm upstand.
- The button is a sprung slider (est): 250 N/m, 5 mm travel. It only moves; no software rule.
- Friction is yam_arena's default for every body.

## Sources

- Estimates (not measured; measure your own): the bottle (62 mm across, 135 mm with its cap, 0.20 kg with
  a 150 mL feed; empty, about 50 g), the warmer's size, the wells' 72 mm, the buttons.
- Sold 260 mL bottle size for comparison: Philips Avent Natural Response 260 mL, SKU SCY903-01, product dimensions
  7.10 x 7.10 x 16.6 cm (https://www.mumzworld.com/bh-en/philips-avent-natural-3-0-feeding-bottle-260-ml).

## Real build and reset

Category: Societal relatability. Warming a baby's bottle in a two-well warmer while the other bottle stays put is a
familiar parenting task, usually done one-handed; no occupational count is claimed. The warmer here is an unheated fixture, so no sold appliance
or wiring is needed. Everything below is a design to build and check, not something we measured.

**Warmer fixture** (110 x 320 x 60 mm, cream paint, taped to the counter 5 mm in front of the splashback):

| Part | Spec |
|---|---|
| Base sheet | 1 plywood sheet, 10 x 110 x 320 mm |
| Block | 2 plywood or MDF sheets, 25 x 110 x 320 mm, glued, then glued on the base sheet (60 mm in all) |
| Wells | 2 through bores in the block with a 72 mm hole saw, axes 110 mm apart (y +55, y -55), centred in x; the base sheet is the 10 mm floor; 50 mm deep |
| Button bores | 2 bores 19 mm across, 30 mm deep from the top, at x 230 (30 mm toward the front from the centre), y +143 and y -143 |
| Plungers | 2 hardwood dowels 18 mm across, 24 mm long, each with an 8 mm long (along the axis), 3.2 mm wide, 4 mm deep slot cut in one side, facing the front; painted green |
| Stop pins | 2 steel pins 3 mm across, 20 mm long, through a 3 mm hole drilled in from the block's front face (x 205) into the bore, through the slot |
| Springs | 2 steel compression springs, outer diameter 15 to 17 mm, free length 12 mm, solid height 7 mm or less, about 0.25 N/mm |

Assembly: glue the sheets and cut the bores; drop each spring into its bore, then the plunger with its slot facing the
pin hole; drive in the pin so it sits in the slot's lower end at rest (the plunger is held 6 mm proud) and the spring
slightly compressed; the slot's upper end stops the plunger at 5 mm of travel. Plunger and pin are not glued.

Acceptance checks (calipers, force gauge or a kitchen scale pressing the plunger): well bore 72 plus or minus 1 mm,
depth 50 plus or minus 1, centres 110 plus or minus 1, walls 19 mm or more; plunger proud 6 plus or minus 0.5 mm, travel
5 plus or minus 0.5 mm, force at the stop 0.75 to 2 N (pick the spring to fit), springs fully back when released and does
not stick. The button needs no wiring: a push is the plunger 3 mm or more in, read with a rule. A sold two-well warmer
whose wells and buttons pass the same checks may be used instead, unplugged.

**Other parts**: a tray 100 x 100 x 12 mm (wall 4, floor 4) at (x 270, y -250); two 260 mL class feeding bottles (check
with calipers: 62 mm across the body plus or minus 2, 135 mm tall with the cap plus or minus 5, one with a blue and one
with a pink collar), each with 150 mL of water; a 200 mm tall tile or board splashback 20 mm thick, 5 mm behind the warmer.

Reset: neighbour's bottle centred on the +y well's floor, the other on the tray at (x 270, y -250), both buttons out
(proud 6 mm); wipe up spills.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; the parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
