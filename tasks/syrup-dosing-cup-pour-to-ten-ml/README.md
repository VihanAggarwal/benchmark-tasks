# Pour a 10 mL dose of syrup into its dosing cup, held level, and put both down

A 4 fl oz (118 mL) children's liquid-medicine bottle, cap off, stands on the table with 118 mL of thick syrup in it.
A 30 mL dosing cup stands on the pan of a pocket scale. The robot must pour 10 mL into the cup (the cup's 10 mL
line), holding the cup up and level while it pours, then set the cup back on the scale and the bottle back on its mark,
with nothing spilled. The instruction names neither the tilt nor how long to pour.

## Why it is hard

1. **The stream is fast and cannot be taken back.** The 22 mm neck passes about 3 mL/s (an estimate: see the
   fidelity note), so the whole 9.0 to 11.0 mL window is two thirds of a second of stream, and over 15 mL in the cup
   scores 0 for good. The bottle has to be tipped past about 65 degrees before anything comes out, and the stop
   is a tilt back, not a switch.
2. **The lip has to be over the cup.** The stream falls straight from the lip's lowest point and lands only inside the
   cup's 32 mm opening: one hand must hold the 36 mm cup still and level (it weighs 8 g) while the other
   tips the 170 g bottle over it. A stream that misses is spilled syrup.
3. **Reading the line.** Stream that goes in while the cup is lower than 20 mm over the pan or tilted over 5 degrees
   counts as unread stream (the label's rule is to hold the cup level and read the line); more than 0.15 s of it caps
   the score at 0.5.
4. **A clean finish.** The cup back on the pan, the bottle upright on its mark, both grippers 30 mm clear, both
   parts still for 1 s.

## Scoring

| Quantity | Rule |
|---|---|
| Dose | syrup in the cup (real trial: cup content on a 0.1 g scale / 1.30 g/mL) |
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
| Dose 9.0 to 11.0 mL (8.0 to 12.0; 4; 15) | tare the empty cup on a 0.1 g scale; the content in grams over 1.30 g/mL: 11.7 to 14.3 g (10.4 to 15.6; 5.2; 19.5 g) |
| Spilled 0.5 mL (3 mL, 10 mL) | the bottle's weight before minus after, minus the cup's gain: 0.65 g (3.9 g, 13 g); the scale reads to 0.1 g |
| Cup tipped over 45 degrees | a 45 degree angle gauge on the 30 fps video; the content is weighed again afterwards |
| Unread 0.15 s | 30 fps video: the frames with syrup falling into the cup while its base is under 20 mm over the pan (a 20 mm spacer block beside it) or its wall is over 5 degrees off an angle gauge; more than 4 such frames (0.15 s) is too much |
| Cup placed | a ruler on the pan from its centre (15 mm), a 3 mm feeler under the cup, a 3 degree angle gauge |
| Bottle placed | a ruler from the mark (30 mm), a 3 mm feeler, a 5 degree angle gauge |
| Grippers clear | a 30 mm spacer block against each hand |
| Still, 1 s | the 30 fps video of the last second: nothing moves more than 1 cm in 1 s |

## Results

GPU runs (L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 1,200-step limit, one trial each) are in the
GPU column. The offline prediction column is `tools/offline/check_task.py` with this task's offline stand-in for the physics
(`task.OfflineModel`, which rebuilds the bottle's tilt from the right hand's pose) and the real scorer; it is not a
measurement. Rows marked not run have no GPU number.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, settle) | both arms hold still | 0.0 | 0.0 (parts stayed put) |
| Scripted demo | below: 10.08 mL | 1.0 (held from step 442) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | lifts the cup, sets it down: no pour | 0.0 | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | pours with the cup on the pan, not held | 0.5 (unread syrup caps it) | 0.5 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_POUR_HOLD=1.5` | 8.1 mL | 0.5 | not run |
| `DEMO_POUR_HOLD=3.0` | 12.4 mL | 0.25 | not run |

With the left hand idle the cup stays on the pan: the dose is right but the syrup went in with the cup not held up
and level, so it is capped at 0.5; with the right hand idle nothing is poured. Only both hands together score 1.0.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The hold at full tilt is
the dose: offline the ledger takes 2.9 mL per second of hold plus 3.8 mL during the tilt in and out, so 2.2 s makes
10.1 mL (1.9 to 2.5 s keep it inside 9.0 to 11.0). Gripper moves are 1.5 s each.

| Time (s) | Left gripper (fingers across the cup, side grip) | Right gripper (fingers across the bottle, side grip) |
|---|---|---|
| 0 to 5.2 | beside the cup, in, close on its wall | beside the bottle, in, close on its body |
| 5.2 to 6.4 | lift the cup 45 mm and keep it level | lift the bottle straight up (1.4 s) |
| 6.6 to 8.2 | hold | over to the cup, upright (1.6 s) |
| 8.2 to 10.7 | hold | tilt to 76 degrees about the grip point (2.5 s) |
| 10.7 to 12.9 | hold | hold the pour (2.2 s) |
| 12.9 to 15.4 | hold | tilt back upright (2.5 s) |
| 15.4 to 17.2 | set the cup on the scale (1.2 s), open | bottle back over its mark, lower, open (4.5 s) |
| to 21.4 | withdraw | withdraw up and back |

Offline: worst IK 0.09 mm; the smallest joint margin is 6.1 degrees (right wrist at the full tilt) and 11.6 degrees
(left); the arms' capsules come within 2.1 cm of each other while the bottle tilts back upright; the TCPs stay at
least 10.4 cm apart.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: the demo is about 430 steps (21 s simulated); a 1,200-step trial takes about 2 to 4 minutes of wall-clock on
the L40S including start-up (an estimate). Seed: nothing is random; `object_jitter=0` fixes the start
layout, so no seed is needed. `DEMO_POUR_HOLD=<s>` changes the hold, `DEMO_HOLD_ONLY=1` and `DEMO_IDLE_ARM=left|right`
are the settle and one-arm checks, `TASK_TRACE=<file>` writes one JSON line per control step (the ledger, both
parts' poses, both TCPs and openings).

For Dream-1 use `--policy dream`.

## Simplifications

- Rigid bodies, no liquid. The syrup is a volume ledger computed from the bottle's and the cup's real poses each
  step (`scoring.py`): the free surface is the horizontal plane that has the remaining volume below it in the
  bottle's bore (a 45 mm round body, then a 22 mm neck); syrup leaves over the lip's lowest point at a laminar film
  rate `g w h^3 / (3 nu)` (w the chord, h the depth of the surface over the lip, nu = 1.15e-4 m^2/s for 150 mPa s and
  1.30 g/mL), never more than 3.0 mL/s (what the neck passes: an estimate); it falls straight down from the lip and
  lands in the cup if that point is within the cup's opening, 4 mm in from the walls, above its rim, and the cup is
  tilted 30 degrees or less. A cup tipped past 45 degrees pours out what it holds. No delay, drip tail, cling to the
  lip, splash or foam is modelled: the stream stops because the surface drops under the lip as the bottle tilts back.
  A real thick syrup keeps running for a moment and leaves a drip; the real trial's thresholds are by weight, so this
  shows up as measured spill, not as a different rule.
- The syrup and the printed line are drawn: the 10 mL line is a dark 2 mm strip on the cup's front and back faces
  (11.8 mm over its base, 0.6 mm proud), and a thin amber plate inside the cup is moved by a slider, every control
  step, to the ledger's level (`task.syrup_target`: the volume over the cup's 32 x 32 mm floor). It is the one
  declared rule that moves a part, it is display only, and the scorer reads poses, not the plate.
- The cup is square (36 mm outside, 32 mm inside) where a real dosing cup is round about 36 mm across. The score
  is by volume, so a round cup is read by the weight of its content; only the height of the 10 mL line differs.
- The bottle is two cylinders (49 mm body, 28 mm neck); a real bottle has a shoulder and a ribbed neck. Mass 170 g
  (118 mL x 1.30 g/mL of syrup plus 16 g of bottle, rounded).
- Friction is the simulator's, the same for every part (static 1.2, dynamic 1.0). Viscosity 150 mPa s, density
  1.30 g/mL and the neck's 3.0 mL/s are estimates; the real trial uses the sugar syrup recipe in "Real build and reset" and times its own flow with the bottle held at 76 degrees over a measuring cylinder.

## Sources and estimates

- Bottle: a 4 fl oz (118 mL) HDPE children's liquid-medicine bottle, the usual retail size (retail listings of
  4 fl oz children's pain and fever liquids); outer size of the model is an estimate to be measured on the bench.
- Dosing cup: the 30 mL (1 fl oz) plastic cup packed with liquid medicines, graduated in 5 mL steps; 36 mm by 38 mm is
  the design size: measure the cup that comes with the bottle.
- Estimates (also in the fidelity note): viscosity 150 mPa s, density 1.30 g/mL, the neck's 3.0 mL/s, the cup's
  corner radius, the pan's size (100 x 100 mm, 14 mm high, as a small pocket scale).

## Real build and reset

A 4 fl oz medicine bottle (or a bottle of that size, its cap off); a 30 mL dosing cup; a pocket scale reading to 0.1 g
with its pan about 100 mm across; a 20 mm spacer block, angle gauges for 3, 5 and 45 degrees, a ruler, feeler gauges;
a probe thermometer, a 100 mL measuring cylinder, a shallow drip tray, thin nitrile finger sleeves, cling film.

**Syrup (standard, repeatable, a few minutes to make).** Plain sugar syrup from supermarket granulated sugar and tap
water: 640 g sugar into 360 g water, warmed and stirred until clear, cooled in a sealed jar to 20 C (+/- 1 C, probe
thermometer; the room is held at 20 +/- 2 C). Check by weight: 100 mL in the measuring cylinder weighs 130 g
(+/- 1 g), i.e. 1.30 g/mL; correct by adding a little water or sugar. Viscosity is about 150 mPa s at 20 C and rises
steeply as it cools (roughly 10 percent per degree), so keep to the temperature. Use within 8 h of cooling, or keep
sealed in a fridge for up to a week and bring back to 20 C, stirring gently so no foam forms. Do not reuse syrup that
has been poured into the cup or touched anything. Flow check at the start of each session: hold the filled bottle at 76
degrees over a measuring cylinder; it must pass 2.5 to 3.5 mL/s (the model's 3.0 mL/s); if slower add 10 g of water
per kg of syrup, if faster add 10 g of sugar per kg, then check again.

**Protecting the rig.** Fit thin disposable nitrile sleeves over both grippers' pads (they grip the bottle and the cup,
which get sticky) and lay cling film over the scale's pan, tare the scale with the film and empty cup on it, and run
everything inside a shallow drip tray larger than the pan and the bottle's mark, so spills never reach the scale
or the pads.

**Reset and cleanup.** Before each trial: the bottle upright on its mark with 118 mL of 20 C syrup (weigh it: 170 g), the
empty cup in the middle of the pan, the lip and the bottle's body wiped. After each trial: wipe spills with a damp cloth
(sugar syrup dissolves in water), then dry; replace the finger sleeves and the film if they are wet or tacky; empty
and rinse the drip tray in warm water; rinse and dry the cup between trials and check that the scale still tares
to zero. Dried syrup is hard to remove, so do not let it stand.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, offline stand-in |
| `layout.py`, `scoring.py` | Every dimension and rule threshold; the syrup ledger and the scorer |
| `pourplan.py`, `tiltgrip.py` | The demo's geometry; tilted grips for the scripted demo |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
