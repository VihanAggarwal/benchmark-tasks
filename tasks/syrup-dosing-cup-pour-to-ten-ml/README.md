# Pour a 10 mL dose of syrup into its dosing cup, held level, and put both down

A 4 fl oz (118 mL) children's liquid-medicine bottle, already open, stands on the table with 118 mL of thick syrup in
it. A clear, see-through, round 30 mL dosing cup, graduated 5 to 30 mL, stands on the white platform of a pocket
scale. The robot must pour 10 mL into the cup (the cup's 10 mL line),
holding the cup up and level while it pours, then set the cup back on the scale and the bottle back on its mark, with
nothing spilled. The instruction names neither the tilt nor how long to pour.

Why the bottle starts open: the carer has already shaken it and pressed and turned its child-resistant cap off (the cap
is put away, not on the table); the step tested is measuring the dose. The cap goes back on after the child has taken
the dose from the cup, which is after this trial.

## Revision (2026-11-19, look)

Review feedback: "The dosing cup should have obvious markings. Better to make it transparent like a measuring cup".
This revision answers it with the cup's look and the task text: the cup now reads as a clear graduated
measuring cup with bold markings. The physics the hand works against, the masses, the scorer, its thresholds, the
syrup ledger and the demo are unchanged; the cup's colliders are split as described below.

- **The wall is clear because it is not rendered (not translucent).** Each of the cup's 24 flat wall panels is two
  colliders with the same radial span and width: the lower part, from the floor up to 35.4 mm, is a collider that is
  drawn not at all (the fingers still close on it, at 19 mm), and the top 2.5 mm (35.5 to 38.0 mm) is the cup's
  rolled lip, drawn opaque in a pale cool grey-blue. Nothing in the scene has an opacity below 1. Through the cup the
  camera sees the white platform under it; the lip draws a crisp pale ellipse, the rim of a clear cup. The GPU log
  prints one line per assembly with hidden shapes: `mechkit: dosing_cup 24 shapes drawn invisible (colliders kept)`.
  If a renderer ignored the hide, the lower panels would draw in the platform's white: a white cup with crisp rings,
  never a grey disc.
- **Tinted base.** The 2 mm floor is opaque and pale clear-plastic blue: the cup's base disc on the white platform. It
  also hides the empty syrup disc parked inside it, so an empty cup shows no syrup.
- **Printed numerals.** 10, 20 and 30 (seven-segment digits, 7 mm tall, 4 mm wide, 1 mm strokes; 10 red, 20 and 30
  near black) stand beside their lines on the two panels facing each arm, in reading order for a person outside the
  cup on that side. The rings leave out those four panels, inside and outside, so no line strikes through or shows
  behind a numeral.
- **Bolder rings, thicker syrup edge.** The 20 and 30 mL rings are now 4 mm tall (19.5 to 23.5 mm and 29.3 to 33.3
  mm over the base); the 10 mL line stays red and 6 mm tall, centred on 10 mL, the 9.0 to 11.0 mL window its middle
  third. The golden syrup disc is 1.8 mm thick (was 0.9 mm), so its edge shows through the clear wall. The 5, 15 and
  25 mL ticks (near black, 1.5 mm) are outside on three panels on each arm's side, clear of the numerals.
- **More pixels.** The setup screenshot is rendered at 2880 x 1584 (was 960 x 528) and the demo's cameras at 640 x
  352 (was 320 x 176), so the rings, the lip and the syrup surface are several pixels tall.

Each ring is drawn once round the cup, on one face of each panel: outside on the six panels of the top camera's half
that the hand does not grip and that carry no numeral, inside on the other fourteen (the far half seen over the near
lip, and the gripped near panels seen through the clear wall). Drawn on both faces, a ring showed twice and the
stacked dark lines read as a grey band.

What the top camera sees in the 2880 x 1584 screenshot (about 1.8 px/mm across the cup, about 0.8 px per mm of wall
height, the camera looking down from behind the arms at about 64 degrees):

| Feature | Size | Pixels |
|---|---|---|
| White platform | 100 x 100 mm | about 180 x 162 |
| Rim (the pale lip), an ellipse | 40 mm across | about 72 x 65 |
| Inside of the cup (base, then syrup) | 36 mm | about 65 x 58 |
| 10 mL ring, red | 6 mm tall | about 4.8 tall |
| 20 and 30 mL rings, near black | 4 mm tall, 9.8 mm apart | about 3.2 tall, 7.8 apart |
| Golden syrup disc at 10 mL | 35 mm across, 1.8 mm thick, top in the middle of the red band | about 63 x 57 |
| Numerals 10, 20, 30 | 7 mm tall, on the sides facing the arms | nearly edge-on from the top camera; the wrist cameras' view |
| Ticks at 5, 15, 25 mL | 1.5 mm, on each arm's side | about 1.2 tall where seen |

The left hand comes in from the +y side and holds the cup there while the syrup goes in, so its wrist camera looks
at the cup's +y side (numerals, rings and the rising golden surface) from close by: that video is the close view of
the graduations and of the syrup reaching the red band.

## Why it is hard

1. **The stream is fast and cannot be taken back.** The 22 mm neck passes about 3 mL/s (an estimate: see the
   fidelity note), so the whole 9.0 to 11.0 mL window is two thirds of a second of stream, and over 15 mL in the cup
   scores 0 for good. The bottle has to be tipped past about 65 degrees before anything comes out, and the stop
   is a tilt back, not a switch.
2. **The lip has to be over the cup.** The stream falls straight from the lip's lowest point and lands only within
   14 mm of the cup's axis: one hand must hold the 40 mm cup still and level (it weighs 8 g) while the other tips the
   170 g bottle over it. A stream that misses is spilled syrup.
3. **Reading the line.** Many carers hold the cup up at eye level in one hand, level, and pour with the other, so they
   can watch the line as the syrup rises; this trial tests that two-handed way. Stream that goes in while the cup is
   lower than 20 mm over the pan or tilted over 5 degrees counts as unread stream; more than 0.15 s of it caps the
   score at 0.5. The scale's pan is where the cup stands and where the trial checks the dose.
4. **A clean finish.** The cup back on the pan, the bottle upright on its mark, both grippers 30 mm clear, both
   parts still for 1 s.

## Scoring

| Quantity | Rule |
|---|---|
| Dose | syrup in the cup (real trial: cup content on a 0.1 g scale / 1.31 g/mL) |
| Spilled | left the bottle and did not land in the cup, or was in a cup tipped over 45 degrees |
| Unread | seconds of stream into the cup while it was under 20 mm over the pan or tilted over 5 degrees |
| Placed | cup on the pan (15 mm, 3 mm, 3 degrees), bottle on its mark (30 mm, 3 mm, 5 degrees), grippers 30 mm clear |
| Zero, latched | more than 15 mL ever in the cup; cup tipped over 45 degrees; more than 10 mL spilled |

| Outcome | Score |
|---|---|
| 9.0 to 11.0 mL, spilled 0.5 mL or less, unread 0.15 s or less, placed | 1.0 |
| Same dose, any of that not met (but spilled 3 mL or less, unread 0.15 s or less) | 0.75 |
| Same dose with more than 0.15 s unread or more than 3 mL spilled | 0.5 |
| 8.0 to 12.0 mL (other doses) | 0.5 |
| At least 4 mL | 0.25 |
| Below 4 mL, or a latched zero | 0 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0. Success: 1.0 with both parts moving under 1 cm/s and 0.5 rad/s, held 1 s.
Doing nothing scores 0 (a pour with the lip over the cup is the only way to get syrup in it).

Bench method for each threshold on a real build:

| Threshold | Bench method |
|---|---|
| Dose 9.0 to 11.0 mL (8.0 to 12.0; 4; 15) | tare the empty cup on a 0.1 g scale; the content in grams over 1.31 g/mL: 11.8 to 14.4 g (10.5 to 15.7; 5.2; 19.7 g) |
| Spilled 0.5 mL (3 mL, 10 mL) | the bottle's weight before minus after, minus the cup's gain: 0.66 g (3.9 g, 13.1 g); the scale reads to 0.1 g |
| Cup tipped over 45 degrees | a 45 degree angle gauge on the 30 fps video; the content is weighed again afterwards |
| Unread 0.15 s | 30 fps video: the frames with syrup falling into the cup while its base is under 20 mm over the pan (a 20 mm spacer block beside it) or its wall is over 5 degrees off an angle gauge; more than 4 such frames (0.15 s) is too much |
| Cup placed | a ruler on the pan from its centre (15 mm), a 3 mm feeler under the cup, a 3 degree angle gauge |
| Bottle placed | a ruler from the mark (30 mm), a 3 mm feeler, a 5 degree angle gauge |
| Grippers clear | a 30 mm spacer block against each hand |
| Still, 1 s | the 30 fps video of the last second: nothing moves more than 1 cm in 1 s |

## Results

GPU runs (L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 1,200-step limit, one trial each) are in the
GPU column. The offline prediction column is `tools/offline/check_task.py` with this task's offline stand-in for the
physics (`task.OfflineModel`, which rebuilds the bottle's tilt from the right hand's pose) and the real scorer; it is
not a measurement. Rows marked not run have no GPU number. Both columns are for the current scene and demo (the
hidden wall, drawn lip and numerals); the GPU log printed the hide line for the cup's 24 wall panels. Volumes and
step numbers are from the GPU run's per-step trace.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, settle) | both arms hold still | 0.0 | 0.0 (160 steps; bottle and cup did not move) |
| Scripted demo | below | 1.0 (10.08 mL, held from step 442) | 1.0 (10.29 mL, no spill or unread; success step 426) |
| `DEMO_IDLE_ARM=right` (left alone) | lifts the cup, sets it down: no pour | 0.0 | 0.0 (0 mL poured) |
| `DEMO_IDLE_ARM=left` (right alone) | pours into the cup left on the pan | 0.5 (unread syrup caps it) | 0.5 (10.29 mL, 3.6 s unread) |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 (1,200 steps; 0 mL) |
| `DEMO_POUR_HOLD=1.5` | 8.1 mL | 0.5 | not run |
| `DEMO_POUR_HOLD=3.0` | 12.4 mL | 0.25 | not run |

With the left hand idle the cup stays on the pan: the dose is right but the syrup went in with the cup not held up
and level, so it is capped at 0.5; with the right hand idle nothing is poured. Only both hands together score 1.0.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The hold at full tilt is
the dose: offline the ledger takes 2.9 mL per second of hold plus 3.8 mL during the tilt in and out, so 2.2 s makes
10.1 mL (1.9 to 2.5 s keep it inside 9.0 to 11.0); the GPU run gave 10.29 mL. Gripper moves are 1.5 s each.

| Time (s) | Left gripper (fingers across the cup's +-x panels, side grip) | Right gripper (fingers across the bottle, side grip) |
|---|---|---|
| 0 to 5.2 | beside the cup, in, close on its wall | beside the bottle, in, close on its body |
| 5.2 to 6.4 | lift the cup 32 mm and 6 mm toward +y, keep it level | lift the bottle straight up (1.4 s) |
| 6.6 to 8.2 | hold | over to the cup, upright (1.6 s) |
| 8.2 to 10.7 | hold | tilt to 76 degrees about the grip point (2.5 s) |
| 10.7 to 12.9 | hold | hold the pour (2.2 s) |
| 12.9 to 15.4 | hold | tilt back upright (2.5 s) |
| 15.4 to 17.2 | set the cup on the scale (1.2 s), open | bottle back over its mark, lower, open (4.5 s) |
| to 21.4 | withdraw | withdraw up and back |

Offline: worst IK 0.09 mm; the smallest joint margin is 6.1 degrees (right wrist at the full tilt) and 11.6 degrees
(left); the arms' capsules come within 2.9 cm of each other while the bottle tilts back upright; the TCPs stay at
least 11.5 cm apart.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: the demo is 441 steps on the GPU (22 s simulated). On the L40S, including start-up, the demo took about
2 minutes of wall-clock and the full 1,200-step Dream-1 trial about 5 minutes. Seed: nothing is random;
`object_jitter=0` fixes the start layout, so no seed is needed. `DEMO_POUR_HOLD=<s>` changes the hold, `DEMO_HOLD_ONLY=1` and `DEMO_IDLE_ARM=left|right`
are the settle and one-arm checks, `TASK_TRACE=<file>` writes one JSON line per control step (the ledger, both
parts' poses, both TCPs and openings).

For Dream-1 use `--policy dream`.

## Simplifications

- Rigid bodies, no liquid. The syrup is a volume ledger computed from the bottle's and the cup's real poses each
  step (`scoring.py`): the free surface is the horizontal plane that has the remaining volume below it in the
  bottle's bore (a 45 mm round body, then a 22 mm neck); syrup leaves over the lip's lowest point at a laminar film
  rate `g w h^3 / (3 nu)` (w the chord, h the depth of the surface over the lip, nu = 1.15e-4 m^2/s for 150 mPa s and
  1.31 g/mL), never more than 3.0 mL/s (what the neck passes: an estimate); it falls straight down from the lip and
  lands in the cup if that point is within 14 mm of the cup's axis (4 mm in from its wall), above its rim, and the cup
  is tilted 30 degrees or less. A cup tipped past 45 degrees pours out what it holds. No delay, drip tail, cling to
  the lip, splash or foam is modelled, and no stream is drawn: the stream stops because the surface drops under the
  lip as the bottle tilts back. A real thick syrup keeps running for a moment and leaves a drip; the real trial's
  thresholds are by weight, so this shows up as measured spill, not as a different rule.
- The syrup and the graduations are drawn, and the graduations and numerals are drawn bolder than real print so the
  board's cameras resolve them (a real cup's line is 1 to 2 mm). The cup's wall is drawn clear by not rendering it:
  below its 2.5 mm rolled lip each of the 24 panels is a collider that is not drawn; the lip is drawn opaque pale
  grey-blue and the 2 mm base opaque pale clear-plastic blue. The graduations are strips at the volume over the cup's
  10.24 cm^2 inside: rings at 10 mL (red, 6 mm, centred 11.8 mm over its base), 20 and 30 mL (near black, 4 mm), each
  drawn once round the cup, outside on six panels on the top camera's side (0.6 mm proud) and inside on fourteen
  others (0.3 mm proud), with a gap at the four panels that carry the numerals 10, 20 and 30 (7 mm, seven-segment,
  0.6 mm proud); and near black 1.5 mm ticks at 5, 15 and 25 mL outside on three panels on each arm's side. A thin
  opaque golden disc (1.8 mm thick) inside the cup is moved by a slider, every control step, so that its top is at the
  ledger's level (`task.syrup_target`); with the cup empty it is parked inside the solid base. (Simple Syrup NF is
  colourless; the disc is drawn golden so the cameras see it.) It is the one declared rule that moves a part, it is
  display only, and the scorer reads poses, not the disc. The disc is a link of the cup's articulation, which never collides with its own body, and it stays 0.1 mm
  clear of the inside marks.
- The cup's round wall is 24 flat panels (a real cup is smooth and slightly tapered); its inside area is 1.2 percent
  less than a 36 mm circle's. The score is by volume, so a real cup is read by the weight of its content.
- The bottle is two cylinders (49 mm body, 28 mm neck) with a plain white label and a dark disc for its opening; a real
  bottle has a shoulder and a ribbed, threaded neck. Mass 170 g (118 mL x 1.31 g/mL of syrup plus 16 g of bottle is
  171 g, rounded).
- Friction is the simulator's, the same for every part (static 1.2, dynamic 1.0). Viscosity 150 mPa s and the neck's
  3.0 mL/s are estimates; the real trial uses Simple Syrup NF at 20 C (see "Real build and reset") and times its own
  flow with the bottle held at 76 degrees over a measuring cylinder.

## Sources and estimates

- Bottle: a 4 fl oz (118 mL) amber HDPE children's liquid-medicine bottle with a child-resistant push-and-turn cap, the
  usual retail size (retail listings of 4 fl oz children's pain and fever liquids; stock 4 oz amber pharmacy bottles);
  outer sizes of the model are estimates to be measured on the bench.
- Dosing cup: the 30 mL (1 fl oz) clear polypropylene graduated medicine cup packed with liquid medicines, round,
  see-through, with printed mL graduations in 5 mL steps; 40 mm by 38 mm is the design size: measure the cup that comes
  with the bottle.
- Syrup: Simple Syrup NF (the USP-NF Syrup: 85 g of sucrose per 100 mL of syrup, about 1.31 g/mL), a pharmacy
  compounding vehicle sold sealed in 16 fl oz bottles, shelf-stable until opened and kept closed between sessions.
- Estimates (also in the fidelity note): viscosity 150 mPa s, the neck's 3.0 mL/s, the pan's size (100 x 100 mm, 14 mm
  high, as a small pocket scale; drawn white, any colour works on the bench).

## Real build and reset

A 4 fl oz medicine bottle with its own child-resistant cap (or a 4 oz amber bottle of that size with a push-and-turn
cap); a 30 mL dosing cup; a pocket scale reading to 0.1 g with its pan about 100 mm across; a 20 mm spacer block, angle
gauges for 3, 5 and 45 degrees, a ruler, feeler gauges; a probe thermometer, a 100 mL measuring cylinder, a shallow
drip tray, thin nitrile finger sleeves, cling film.

**Syrup (bought, sealed, shelf-stable).** Simple Syrup NF from a pharmacy supplier, used as bought, no mixing: keep it
closed at 20 C (+/- 1 C, probe thermometer; the room is held at 20 +/- 2 C). Check by weight: 100 mL in the measuring
cylinder weighs 131 g (+/- 1 g), i.e. 1.31 g/mL. Viscosity is about 150 mPa s at 20 C and rises steeply as it cools
(roughly 10 percent per degree), so keep to the temperature. Do not reuse syrup that has been poured into the cup or
touched anything. Flow check at the start of each session: hold the filled bottle at 76 degrees over a measuring
cylinder; it must pass 2.5 to 3.5 mL/s (the model's 3.0 mL/s); if slower, check that it is at 20 C, then stir in 10 g of
water per kg of syrup and check again; if faster, use a new sealed bottle.

**Protecting the rig.** Fit thin disposable nitrile sleeves over both grippers' pads (they grip the bottle and the cup,
which get sticky) and lay cling film over the scale's pan, tare the scale with the film and empty cup on it, and run
everything inside a shallow drip tray larger than the pan and the bottle's mark, so spills never reach the scale
or the pads.

**Reset and cleanup.** Before each trial: the bottle upright on its mark with 118 mL (155 g) of 20 C syrup, the empty
cup in the middle of the pan, the lip and the bottle's body wiped. After each trial: wipe spills with a damp cloth (the
syrup dissolves in water), then dry; replace the finger sleeves and the film if they are wet or tacky; empty and rinse
the drip tray in warm water; rinse and dry the cup between trials and check that the scale still tares to zero. Dried
syrup is hard to remove, so do not let it stand.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, offline stand-in |
| `layout.py`, `scoring.py` | Every dimension and rule threshold; the syrup ledger and the scorer |
| `pourplan.py`, `tiltgrip.py` | The demo's geometry; tilted grips for the scripted demo |
| `mechkit.py`, `rigstate.py` | Build the parts (bottle, cup) as compound rigid bodies (this copy adds a per-shape hidden look, a collider that is not drawn, and a per-shape look opacity, default 1.0, which this task does not use); read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
