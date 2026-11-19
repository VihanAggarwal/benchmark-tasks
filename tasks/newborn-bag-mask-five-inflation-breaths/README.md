# Give a newborn manikin five inflation breaths with an infant bag and mask

A newborn training manikin lies on its back on the midline, crown toward the robot, its face up 90 mm over the
table. In front of its crown a tray holds an infant self-inflating bag (280 mL, 40 cmH2O relief valve, 360 degree
swivel) with a size-1 round newborn mask (64 mm across, 37 g) on the swivel, mask down, the bag lying level toward
-y. The robot must give the five inflation breaths that start newborn life support: put the mask on the face over
nose and mouth, give five breaths of 2 to 3 s that each raise the chest, keep off the eyes, and finish with the mask
still held on the face and the bag let go (the hand that squeezed is now free, as for the next check).

## Why it needs two hands

1. **The seal needs a press.** The mask seals only while every part of its cushion is pressed on and the cushion's
   force is 2 to 10 N. The mask and valve weigh 0.66 N, so a mask left on the face leaks every breath (in the real
   kit the acceptance check "mask resting under its own weight: no rise" makes this a fact of the parts, not a rule).
2. **The bag hangs off to one side, on a soft neck.** The bag's middle is 148 mm from the mask's axis, on a neck
   that bends freely +-0.5 rad, and the valve turns freely on its swivel. Nothing holds the bag level except a hand.
3. **One hand cannot do both.** Held by the valve elbow alone, the mask can be pressed on, but the bag droops to its
   stop: the neck kinks (over 0.40 rad) and no breath goes through; and nothing squeezes it. Held by the bag alone,
   the squeezing hand must also press the mask on through the neck. With the neck free, moments about its hinge
   bound that press: the most a hand on the bag can put on the cushion with every pad still pressed is
   `N <= W y / (a - h mu)` (the mask's and valve's weight moment over the pads' nearest resultant arm less friction's),
   1.11 N at mu 0.6, 1.24 N at 0.7, 1.59 N at 0.9: always under the 2 N seal minimum (tests `ArmGive`). Any more
   and the mask tips about the hinge and lifts its near pads.
4. **Timing.** Each breath must stay at 4 mm or more of rise for 2.0 to 3.0 s: squeeze, hold, ease. A quick
   squeeze-and-release (under 2 s) gives no good breath; holding over 3 s latches "too long".

## Scoring

| Condition | Rule |
|---|---|
| Sealed | mask axis within 8 mm (top view) of the target point, tilt 6 degrees or less, cushion within 0.5 mm of the face, every pad pressed 0.3 mm or more and none at its 10 mm stop, cushion force 2 to 10 N |
| Kinked | the bag's neck bent over 0.40 rad |
| Rise (declared air rule) | sealed and not kinked: each further mm of squeeze raises the chest 0.6 mm; easing 1 mm off the peak lets it fall (0.3 s); unsealed or kinked: it falls (0.2 s) and squeezing does nothing; capped at 12 mm |
| Inflation | the rise at 4 mm or more (dips over 1 mm join), counted when it falls under 1 mm; good if it lasted 2.0 to 3.0 s |
| Too long (latched) | an inflation over 3.0 s |
| Over-inflated (latched) | a rise over 9 mm |
| Dropped (latched) | after the mask has been lifted 10 mm off the tray, any part of the bag and mask within 2 mm of the table |
| Eyes (latched) | the bag and mask within 1 mm, or a gripper's finger zone within 2 mm, of an eye zone (7.5 mm ball) |
| Lifted (latched) | after the first good inflation, unsealed for over 1.0 s at a time |
| At rest | every step of the last 1.0 s: sealed, squeeze under 2 mm, every finger zone 5 mm or more from the bag, rise under 1 mm |

Score: 0 if eyes or no good inflation; else 0.25 if over-inflated or dropped; else 0.5 if too long; else 1.0 for
exactly 5 inflations, all good, at rest and never lifted; 0.75 for 5 or more good otherwise; 0.5 for 3 or 4; 0.25
for 1 or 2. Success: 1.0 held for 1 s. A reset back on the tray's marks clears everything.

## The mechanism

- **Manikin (fixed).** Skull as lying cylinder slices on an ellipse (crown x 164, chin x 289 mm), a level face seat
  84 mm across at z 90 mm under the mask, nose and mouth bumps (under the mask's cavity even 8 mm off centre, at the
  pads' stop), two eye discs at x 206, y +-21 mm. Body with a chest plate on a stiff sprung slide (2000 N/m); the air
  rule drives its target every control step (`mk.on_step`), so the chest visibly rises and falls.
- **Bag and mask (one loose mechkit articulation, 147 g).** The mask's rigid shell; the cushion as 8 pads on vertical
  sprung slides (150 N/m each, 10 mm travel); the valve (elbow and housing) on a free swivel; the bag on a free
  neck hinge (+-0.5 rad); two squeeze panels on the bag's sides on sprung slides (400 N/m each, 12 mm travel).
- **Air (declared rule, scoring.Lung).** Seal from the pads' compressions, the tilt and the place; squeeze from the
  panels' travel. The scorer runs the same rule offline.

## Demo

| Phase | Left (mask hand) | Right (bag hand) |
|---|---|---|
| Pick up | straight down over the valve elbow, pinches it (19 mm) | rolled 40 degrees, comes in along its own axis over the bag, closes on the panels (6.5 mm squeeze, to carry) |
| Carry | lifts and carries the mask to over the face, both hands together | carries the bag level beside it |
| Seat | down onto the face, then 19 mm further (the arm's give sets the force) | level at the neck's height |
| Breaths x5 | holds the mask on | squeeze to 16.5 mm (0.6 s), hold 2.0 s, ease (0.6 s), wait 1.0 s |
| End | keeps holding the mask | opens, backs out along its axis, parks aside |

The plan runs 36.7 s (734 steps); the scripted policy stops at 795 steps; run limit 1300 steps. Each breath is
2.45 s at 4 mm or more in the offline model (rise 6.0 mm). Why a rolled right grip: straight-down grips 148 mm apart
put the two wrists' links into each other; the roll gives 5.7 cm of arm-to-arm clearance.

**Press band (quasi-static, cushion 1200 N/m in series with the arm's give, 19 mm over-command, mask and valve
weight included):**

| Arm give at the TCP (N/m) | 100 | 150 | 200 | 300 | 400 | 600 |
|---|---|---|---|---|---|---|
| Cushion force (N) | 2.36 | 3.12 | 3.82 | 5.09 | 6.19 | 8.04 |
| After the bag is let go and hangs (N) | 3.08 | 3.82 | 4.49 | 5.71 | 6.78 | 8.56 |

GPU traces of other tasks put the arms' give at about 150 to 250 N/m; a joint-stiffness model of this pose gives
about 390 N/m vertical. Every value is inside 2 to 10 N. The squeeze is set by the fingers' gap and does not depend
on where they close on the bag (both panels stay in contact with the fingers up to 8 mm off centre; tests
`ArmGive`). The gripper drives each finger at 5000 N/m to its target, capped at 50 N: each panel takes about 3.3 N at
the breath's peak (an earlier draft's "5 N gripper limit" was wrong).

## Results

GPU run 1 on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each, beside the offline
prediction (`check_task --require-json --strict --one-arm` with this task's offline stand-in for the physics).

| Run | Offline prediction | GPU score and what happened |
|---|---|---|
| Nothing moves (settle) | 0.0 | 0.0: the bag and mask stayed put on the tray (mask moved under 0.1 mm, neck 0.001 rad) |
| Scripted demo | 1.0 (held from step 723) | 1.0 at step 712: sealed from step 208, cushion force 4.1 to 7.6 N (band 2 to 10 N), mask tilt at most 1.3 degrees and 2.4 mm off centre; 5 good inflations of 2.45 s each, 16.4 mm of squeeze and 6.0 mm of rise per breath; neck at 0.10 rad during the breaths; after the bag is let go it hangs at its neck stop and the mask stays sealed (7.6 N) for the last 1 s |
| Left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 | 0.0: the right hand alone could put at most 1.1 N on the cushion and the mask slid 40 mm off the face; no breath counted |
| Right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 | 0.0: the mask was sealed (7.6 N) but nothing squeezed the bag and the bag hung kinked at its stop; no breath |
| Dream-1, full 1300-step limit | not predicted | 0.0: the bag and mask were nudged 3 mm on the tray but never lifted |

Offline counterfactuals (offline check only): `four` breaths 0.5, `short` 1.2 s holds 0.0, `light` (stops 3 mm short
of pressing) 0.0, `keep` (never lets go of the bag) 0.75.

## Simplifications

- The face under the mask is a rigid level seat, not a shaped soft face; the cushion's softness is in its 8 pads.
- Air, leak and lungs are a declared rule read from the pads and the squeeze; no air is simulated. The 2 to 10 N seal
  band, the 0.6 mm per mm gain and the 9 mm over-inflation line are estimates; the setup's acceptance check calibrates
  the real kit (4 to 9 mm of rise for a 10 mm squeeze at a sealing press).
- The mask is held through its valve elbow (a pinch over the mask's top) rather than the hand's two-point top hold
  with the jaw lifted; the force goes down the mask's axis the same way.
- The bag is a rigid ellipsoid; only its middle (the panels) squeezes.

## GPU checks (run 1)

1. Settle: the 3 g pads, the 5 g panels and the free swivel and neck stayed still on the tray.
2. Press force: 4.1 to 7.6 N while sealed, inside the 2 to 10 N band (offline estimate was 3 to 6 N).
3. The mask's tilt stayed under 1.3 degrees and its centre within 2.4 mm of the target; every pad was pressed.
4. The finger squeeze reached 16.4 mm on the panels and the rise was 6.0 mm on each of the five breaths.
5. When the right let go, the bag swung down to its neck stop (0.5 rad) and the mask stayed sealed.
6. No collisions between the arms or with the manikin showed in the frames.

## Sources and measurements

- Laerdal Newborn Mask user guide (size 1: 64 mm outside diameter, 37 g; size 0: 49 mm, 17 g):
  https://shop.laerdalglobalhealth.com/wp-content/uploads/User-guide-Newborn-Mask-Rev-B.pdf
- Intersurgical 7150000, BVM resuscitator, infant, 280 mL bag with 40 cmH2O pressure relief valve, 360 degree swivel:
  https://ca.intersurgical.com/content/files/72303/114610301
- NeoNatalie newborn manikin (Laerdal Global Health, chest rises with correct bag and mask ventilation):
  https://shop.laerdalglobalhealth.com/product/neonatalie/ ; https://laerdal.com/us/item/104-10002
- NeoNatalie Resuscitator (bag with size 0 and 1 masks): https://shop.laerdalglobalhealth.com/product/neonatalie-resuscitator/
- Each year about 10 million babies need help to start breathing, about 6 million basic resuscitation (usually bag
  and mask): Wall et al. 2009, Int J Gynaecol Obstet 107 Suppl 1, https://pmc.ncbi.nlm.nih.gov/articles/PMC2875104
- Everything else (manikin dimensions, cushion and bag stiffness, the air rule's constants) is an estimate, marked
  (est) in `task.json` and `layout.py`.

## Real build and reset

Manikin on its back on the midline, body fixed, head on a zeroed kitchen scale let into the table so the face is at
the height given; tray fixed in front of the crown; the bag and mask in the tray's painted outline. Real trial
measures: rise from a 30 fps side video against a 1 mm scale (marked at 1, 4 and 9 mm); the seal at every frame from the head's scale
(2 to 10 N is 204 to 1020 g over the head's weight) and the top video (rim on the face, within 8 mm of the mark); eyes from a chalked cushion rim; the rest from top and side video. Reset: mask lifted
straight off, the kit laid back in the outline, chest down, bag round again, chalk wiped and renewed.

## Files

| File | What |
|---|---|
| `task.json` | the task's text |
| `task.py` | parts, the air rule driving the chest, hooks, the offline stand-in, the demo and its variants |
| `scoring.py` | the seal, the air rule, the breath count, the latches and the score |
| `layout.py`, `parts.py` | every dimension; the shapes |
| `mechkit.py`, `rigstate.py`, `tiltgrip.py` | the shared kit: articulated parts, arm reading, rolled grips |
| `tests/test_scoring.py` | geometry, seal, air rule, tiers, latches, bad states, exploits, offline demo, arm give, clearances |
