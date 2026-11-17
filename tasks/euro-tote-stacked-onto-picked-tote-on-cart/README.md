# Carry a picked tote by its end hand holds and seat it on the tote on a pick cart

A pick cart's bed stands on the table: a 6 mm steel deck with a locator frame round each of two tote bays and a rear
frame with a push handle. In each bay stands a 400 x 300 x 74 mm stacking tote (Euronorm bin, polypropylene, 0.70 kg
with the picked goods in it), the two totes 60 mm apart end to end. The robot must lift the tote in the left bay out
of its locator frame by its two end hand holds (one hand on each), carry it across 460 mm, set it on the tote in the
right bay so its foot nests inside the lower tote's walls and its ledge rests on their top, and let go. The
instruction names neither the hand holds nor the nest.

The steps are the ones a picker does: lift the full tote out of its bay (first step), carry it level, find the lower
tote's opening and set it down without dragging it, open both hands and clear the stack (last step).

## Why it is hard

1. **It takes both hands.** The long sides are smooth: the only place to hold a tote is the hand hold in each end wall
   (a 110 x 16 mm slot under a 10 x 12 mm grip bar). With the tote 400 mm long the hands are 390 mm apart on one clock;
   each carries 3.4 N. One hand alone leaves it hanging by one end.
2. **The cart limits the reach.** Each hand holds an end bar on its own side of the cart's centre line. The picked
   tote's inner bar is 34 cm across in y from the right arm's base (42 cm in plan), over the midline, where the
   straight-down reach has little height to spare (joint 4 is within 9 degrees of its limit at the 14 cm carry
      height). The lift out of the bay has to clear the lower tote's 80 mm rim (74 mm tote on a 6 mm deck) 60 mm away
   on one side and the locator
   frame on the other, and the carry has to stay level over the rim.
3. **The foot has to find the nest.** The 386 x 286 mm foot has 4 mm clear on each side inside the lower tote's walls,
   with a 6 mm lead-in that takes up to 10 mm of offset. Seated, the ledge rests on the walls' top, 65 mm above the
   lower tote's foot. Set down 10 to 13 mm off centre, the foot misses the opening and the tote rides on the wall tops,
   9 mm higher (74 mm) and unstable; further off, it slides down beside the lower tote.
4. **Hands off.** The score is read with both grippers 30 mm clear and both totes still for 1 s: a tote held in place
   earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| Seated | origin 65 mm above the base tote's (1 mm); centre within 4 mm in x, y; axis within 2 deg; tilt under 1 deg |
| Near | on the base tote, 62 to 77 mm above it, within 20 mm of its centre, 4 degrees, tilt under 3 degrees |
| On | on the base tote: centre within 60 mm, 55 to 90 mm above it, axis within 20 deg, tilt under 20 degrees |
| Hands off | last 1 s: finger zones 30 mm or more from both totes; each tote moves under 5 mm, turns under 5 deg |
| Fell (latched) | a tote's foot underside under 3 mm above the table (off the deck), or tilted over 30 degrees |
| Disturbed (latched) | the base tote over 10 mm from where it stood, or turned or tilted over 5 degrees |

| Outcome | Score |
|---|---|
| Seated, hands off, still | 1.0 |
| Near but not seated (riding on the rim, or seated sloppily) | 0.5 |
| On the base tote otherwise | 0.25 |
| Anything else, a latch fired, or not hands off and still | 0 |

Success: 1.0 held for 1 s. Doing nothing scores 0. A tote is the same turned 180 degrees, so a long axis is compared
modulo 180 degrees. The two totes are not interchangeable: the base tote must stay where it stood.

Bench check for each condition on a real build (the sim reads poses; a person reads these). The 65 mm seated height is
this tote's rim minus its 9 mm ledge height: measure it once on your own pair (rim to rim, seated) and use that
figure:

| Condition | Bench method |
|---|---|
| Seated height (1 mm) | rule at each corner, lower rim to upper rim (65 mm); the ledge sits on the lower rim |
| Centre (4 mm), square (2 degrees) | rule across both end walls at each end; protractor or square on the long edges |
| Tilt (1 degree) | spirit level on the upper rim, or the rim gap (1 degree is 7 mm over 400 mm) |
| Near / on (20 mm, 60 mm, 4, 3 and 20 degrees, axis 20 degrees) | rule and protractor from the lower tote's edges |
| Fell (3 mm) | rule from the table to the foot underside; by eye for 30 degrees |
| Base tote undisturbed (10 mm, 5 degrees) | pencil marks at its corners and a rule |
| Hands off (30 mm), still (5 mm, 5 deg) | 30 mm spacer by each hand; 30 fps video of the last 1 s, taped mark |

## Results

GPU runs: L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 1200-step limit, one trial each; rows marked not
run were not run on the GPU. The GPU scores below are for the current build (74 mm totes, 65 mm seated height,
current scorer), measured on 17 Nov 2026: the demo held full score from step 523 of its run. The settle run (nothing
commanded) scored 0.0.
The offline prediction column
(`tools/offline/check_task.py`, with this task's offline stand-in for the physics, `task.OfflineModel`) is not a
measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run: nothing moved) |
| Scripted demo | below | 1.0 (held from step 521) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the left hand takes its bar alone | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the right hand takes its bar alone | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=offset_x` | set down 12 mm off in x (past the 10 mm lead-in) | 0.5 (modelled as riding) | not run |
| `DEMO_VARIANT=offset_y` | set down 12 mm off in y | 0.5 (modelled as riding) | not run |
| `DEMO_VARIANT=hold` | lowered into place, never let go | 0.0 | not run |

With one arm idle the offline model predicts 0.0 either way: a tote lifted by one end hangs from it, and the model does
not carry it. On the GPU both one-arm runs also scored 0.0. The offset variants are only a model of where a
mis-set foot ends up: in the simulator a foot 12 mm off may tip into the opening, jam on the goods, or fall beside the
lower tote, and the scorer reads whichever pose it reaches.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both hands work on one clock,
the same moves mirrored: the left hand on the +y end bar, the right hand on the -y end bar, yaw 0 (the pads close
along y, across the end wall). Finger targets: 36 mm open (13 mm each side of the 10 mm bar, the outer pad clear of
the neighbouring tote), 2 mm commanded to grip (about 20 N a finger). TCP heights are above the table top.

| Time (s) | Both hands |
|---|---|
| 0 to 2.5 | over the end bar at 120 mm, fingers 36 mm open |
| 2.5 to 5.5 | down round the bar (slowly the last 20 mm) to 54 mm: pad tips 41 mm up |
| 5.5 to 7.5 | settle, grip the bar |
| 7.5 to 9.5 | lift the tote out of its bay to 145 mm (its foot 17 mm over the lower tote's rim) |
| 9.5 to 14.5 | carry it 460 mm across in y, the hands 390 mm apart throughout |
| 14.5 to 19.5 | settle, down to 136 mm (8 mm over the rim), then slowly to 115 mm (4 mm past seated) |
| 19.5 to 21.0 | settle, then open the fingers over 1.5 s |
| 21.0 to 22.5 | straight up to 175 mm (tips 17 mm over the rim), the hands sliding 90 mm toward the bases |
| 22.5 to 25.0 | aside over the rim, fingers open (left to y +75 mm, right to y -510 mm) |
| 25.0 to 26.5 | close the fingers, clear of both totes |

Two real moves: the lift-and-carry, and the lowering into the nest. Offline: worst IK 0.09 mm; joint 4 margin 8.1
degrees at the tightest point (each hand over the stack's inner bar); TCPs 39 cm apart throughout; the two arms'
capsules at least 18 cm apart; the arm links (the fingers on the bars aside) at least 1.9 cm clear of the totes and
the cart (exact shapes against the arms' capsule proxy; in that proxy the outer pad at the inner bar is 11 mm from
the lower tote's end wall, and its fingers are rounder than the real ones). The demo is 530 steps (26.5 s); success
holds from step 521, and the scripted policy stops at 591, inside half of the 1200-step limit.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: a 1200-step trial takes about 2 to 4 minutes of wall-clock on the L40S including start-up (the demo itself is
530 steps, 26.5 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run
starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the picked tote's height
error from seated, its offset from the lower tote, its tilt, the lower tote's displacement, both TCPs and openings,
and a `hist` of the last 3 s of those numbers in every row.

## Simplifications

- Rigid bodies. A tote is a compound of 26 boxes: floor, the foot ring with its 45 degree lead-in, the ledge, 3 mm
  walls, the end walls with their slot and grip bar, and the carton of goods. Ribs, feet and lugs of a real tote are
  left out. The picked goods are a rigid carton (200 x 260 x 30 mm) fixed to the floor, so the 0.70 kg is one rigid
  body whose centre of mass sits 18 mm up; real goods shift.
- The cart is a fixed bed: a real one rolls and is braked.
- Friction 0.30 / 0.27 polypropylene on polypropylene (est), 0.35 / 0.30 on the steel deck (est). The pads' grip is
  the gripper's own.
- No software rule moves or holds anything. Every effect is the bodies' own contact.
- Sizes are nominal. A real tote varies by about 1 mm; the 4 mm clearance and 10 mm lead-in absorb that.

## Sources

- Stockers and order fillers (SOC 53-7065, about 2.8 million US jobs) and hand laborers and material movers (SOC
  53-7062, about 3.0 million): BLS Occupational Outlook Handbook figures, as listed in the program's occupation table.
- The tote's format: the Euronorm stacking container, 400 x 300 mm (the half size of the 600 x 400 mm modular system),
  low model 74 mm. Published figures of a common product of this type, Transoplast Euronorm stacking bin 400 x 300 x
  H 74 mm, article 9227.V00.522 (transoplast.com datasheet): external 400 x 300 x 74 mm, polypropylene, 0.48 kg,
  closed handgrips on the short sides, 10 kg rated load, 250 kg static stacking load, internal 355 x 256 x 64 mm.
  The model matches the external size, the weight, the material and the hand grips on the short (end) walls. A taller
  bin of the same family (WERIT Euronorm 400 x 300 x 320 mm, art. 330-000000095, two handle openings, 1.9 kg) shows the
  same footprint and handle layout.
- Not published by any maker we found: the stacking foot (386 x 286 x 12 mm, 45 degree lead-in 6 mm), the ledge
  height (9 mm), the hand-hold opening (110 x 16 mm) and the grip bar (10 x 12 mm). These are specified by this task.
  The published internal size (355 x 256 mm, measured in a tapered, ribbed cavity) is smaller than the model's
  straight 3 mm walls give (394 x 294 mm at the rim); the model keeps its own walls so that the foot's 4 mm clearance
  is stated and measurable.

Estimates (stated in fidelity): the picked goods' mass (0.22 kg; the tote's 0.48 kg is published), frictions, the foot,
ledge and lead-in sizes, the hand-hold slot and bar sizes, the cart's frame sizes.

## Real build and reset

The overall tote is a stock part; its stacking features are fabricated to this specification, because no maker
publishes them. Stock part: two Euronorm stacking bins 400 x 300 x 74 mm, polypropylene, closed hand grips on the short
sides (for example the Transoplast 9227.V00.522 above), straight-walled if you can get them. Both totes get the same
features, so any pair nests. All sizes are measured from the foot's underside plane (z 0) with the tote on its feet:

- Foot ring: 386 x 286 mm outside, 12 mm tall, 3 mm wall, closed corners: 3 mm PP or ABS strip, solvent or hot-air
  welded to the floor, centred within 1 mm.
- Lead-in: 45 degrees, 6 mm deep and 6 mm wide on the ring's lower outer edge: a filed or printed chamfer.
- Ledge: the rim ring's underside 9 mm above z 0, flat, at least 10 mm wide all round: the bin's own outer wall step,
  or a 3 mm plate shim; measure it.
- Walls: straight, 3 mm, rim at 74 mm; inside the rim 394 x 294 mm (the foot sits 4 mm clear); check with a rule
  across the rim.
- Hand hold: in each short wall a slot 110 mm wide and 16 mm tall, 24 to 40 mm up, centred, under a grip bar 10 mm
  thick and 12 mm tall, 40 to 52 mm up: cut the opening to size, glue in a printed or cut bar.
- Goods: 0.22 kg: one 200 x 260 x 30 mm carton fixed to the floor; the loaded tote 0.70 kg within 0.03 kg (weigh it).

Build route when the stock bin's taper or ribs stop the nest (its published inside is 355 x 256 mm, so its walls are
thicker than the 3 mm modelled): cut the tote from 3 mm PP sheet and weld it, to this cut list (z up from the foot's
underside; the same parts as `tote_shapes()` in `task.py`).

- Floor: 374 x 274 mm at z 0 to 3 mm, with the 6 mm 45 degree lead-in plates round its edge.
- Foot wall: 386 x 286 mm outside, 3 mm thick, z 6 to 12 mm.
- Ledge: a flat ring from the foot wall out to the outer face, 10 mm wide, z 9 to 12 mm.
- Side walls (the 400 mm ones): 400 x 3 mm, z 9 to 74 mm, at x +-148.5 mm.
- End walls (the 300 mm ones): 300 mm wide, 3 mm thick, solid z 9 to 24 mm, with the 110 x 16 mm slot at z 24 to 40
  mm, a 10 mm thick, 12 mm tall grip bar at z 40 to 52 mm, and wall again z 52 to 74 mm.

Go/no-go gauge before any trial: lower the first tote by hand onto the second; the foot must drop in with 4 mm (3 to 5
mm) visible on all four sides, and the seated rim-to-rim must read 65 mm (64 to 66). If the stock bin's own taper or
ribs stop that, make the walls from the cut pieces the table gives (or print the whole tote from the sizes in
`layout.py`, which lists every dimension). If your seated height differs from 65 mm, the task is not reproduced.

Also needed: a cart, or a steel plate, with a locator frame (6 x 6 mm strip, 4 mm clear of the foot) round two bays
60 mm apart end to end; a steel rule, a spirit level. Reset: the base tote in the right bay and the picked tote in the
left bay, each on its foot inside the locator frame, rims up, long sides along y, goods on each floor. Nothing is
consumed.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the rest geometry; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
