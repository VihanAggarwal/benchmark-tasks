# Clamp two Mini-PROTEAN TGX gels in the electrode assembly: short plates in, arms shut

A Bio-Rad Mini-PROTEAN Tetra electrode assembly (the clamping frame of the 2-gel cell for precast gels, cat. 1658005)
is fixed on the table with both green clamping arms open. Two Mini-PROTEAN TGX precast gel cassettes (cat. 4561033)
stand in wet-storage rack blocks beside it, each a quarter turn from how it goes in. The robot must set one cassette on
each green gasket with its short plate facing in, press it flat, stand it in the gel support so the short plate's top
sits just under the notch, and swing each green arm over its cassette until the clip catches; then let go.

## Why this task (and why it is not the gel-tray task)

A gel-tray task (casting tray into an agarose gel box) was not accepted in review. This task answers the points such a
review raises, head-on:

- **Importance.** SDS-PAGE, alone or as the first half of a Western blot, is run daily in molecular biology,
  biochemistry, pharma QC and clinical research labs, and the Mini-PROTEAN Tetra is among the most common mini vertical
  cells. Loading the electrode assembly is the step where most failed runs start: a cassette short plate out, not
  pressed flat or not seated leaves the inner buffer chamber leaking, and the gel runs dry or the samples are lost.
  The category is **Other (lab procedure)**: we do not claim it for Impact on jobs.
- **Not a single press or pick-and-place.** Each cassette must be picked, turned a quarter turn the right way round
  (short plate inward; the plates differ by 10 mm in height and 0.6 mm in thickness), placed against the gasket, seated
  on the support, and then held flat while its arm is swung round 90 degrees over a cam and into its clip. Four
  sub-goals per side, two sides, eight checkable conditions.
- **A held, two-handed clamp.** The manual (Assembling the Mini-PROTEAN Tetra Cell) has the user grip the cassettes
  against the core with one hand while closing the arms with the other, and its Important Note warns against locking
  the arms before the cassettes are aligned at the notches. In the simulation, as on the bench, a cassette let go in its
  support leans out about 12 degrees onto the lip (pencil-in-a-cup geometry); its arm, which runs low over the
  cassette's foot, then meets it about 7 degrees short of shut and cannot reach the clip. One hand must squeeze the
  cassettes flat on their gaskets while the other swings the arm shut.
- **The real product's geometry, so the full-score state seals.** The cassette is the TGX cassette's published size,
  100 x 80 x 4.6 mm, with a 1 mm gel. The full score needs the cassette's face within 1.5 mm of the gasket over its
  whole width, under 3 degrees of tilt, the short plate inward, its top within 0 to 4 mm of the notch, and the arm
  within 1.5 degrees of shut: the state in which a real assembly holds buffer. Dimensions we could not source are
  marked (est) in task.json and below.
- **No stand-ins for the defining mechanism.** No weld, drive or spring closes an arm or seats a cassette. The arm swings
  freely; the cam resists its last 12 degrees and the clip catches it, both only with a cassette flat on the gasket
  (one declared rule, see Fidelity). The racks are plain blocks with a slot, standing in for the gel's storage pouch.

## Sources

- Bio-Rad, Mini-PROTEAN Tetra Cell instruction manual, bulletin 10007296, section "Assembling the Mini-PROTEAN Tetra
  Cell" (steps a to d and the Important Note): https://www.bio-rad.com/webroot/web/pdf/lsr/literature/10007296.pdf
- Bio-Rad, Mini-PROTEAN Tetra vertical electrophoresis cell for mini precast gels, 2-gel, cat. 1658005 (includes the
  electrode assembly): https://www.bio-rad.com/en-us/sku/1658005-mini-protean-tetra-vertical-electrophoresis-cell-for-mini-precast-gels-2-gel?ID=1658005
  (also sold to schools, for example Flinn Scientific FB2420, a few hundred dollars).
- Bio-Rad, Mini-PROTEAN TGX precast gels, 10%, 10-well, 30 uL, cat. 4561033: cassette 10 x 8 cm x 0.46 cm, gel
  8.6 x 6.7 cm, 1.0 mm thick.

## Real build and estimates

| Item | Value | Source |
|---|---|---|
| Cassette | 100 x 80 x 4.6 mm, 1.0 mm gel | Bio-Rad TGX |
| Long plate / short plate with gel | 2.0 / 2.6 mm | est |
| Short plate height | 70 mm (10 mm under the long plate's top) | est |
| Cassette mass, friction | 30 g, 0.6 | est |
| Gasket faces | 18 mm either side of the core's centre | est |
| Gel support | groove 6 mm wide, floor 10 mm up, 3 mm lip to 16 mm | est |
| Notch | 82 mm up (a seated short plate's top 2 mm under it) | est |
| Arm | 195 mm from pin to handle end, 8 x 12 mm, 40 g, 18 to 30 mm up | est |
| Cam and clip | 0.1 N m, last 12 degrees; clip 2 degrees short of shut | est |
| Rack blocks | 30 x 120 x 20 mm, slot 6 x 104 x 15 mm | stand-in for the storage pouch |

A real trial uses the cell's own electrode assembly (fixed to the table with tape or a clamp on its base) and two TGX
cassettes kept wet in running buffer, or two spent cassettes with their gels still in them.

## Scoring

Per gasket, the best cassette on that side:

| Condition | Rule | Bench check |
|---|---|---|
| Against | face toward the centre within 1.5 mm of the gasket at four points on the gasket legs (46 mm either side of its middle, 15 and 60 mm up), tilt under 3 degrees, width within 3 degrees of the gasket's line, middle within 6 mm of the frame's along x | a 1.5 mm feeler gauge does not enter; a 3 degree wedge |
| Short plate in | the short plate faces the centre | by eye |
| Seated | against, short plate in, short plate top 0 to 4 mm under the notch | a rule from the notch |
| Shut | arm within 1.5 degrees of shut | clip clicked, arm flat on the cassette (about 5 mm at the handle) |

| Credit per side | |
|---|---|
| Against, short plate in | 0.15 |
| Seated | +0.15 |
| Arm shut over that cassette | +0.20 |
| Cassette short plate out | 0 for that side, arm included |

Score: the sum over both sides (1.0: both seated short plate in, both arms shut). It counts only with both grippers
20 mm or more from both cassettes and both arms over the last 1 s and nothing moving (2 mm, 2 degrees, arm 2 degrees);
otherwise 0. Latched 0 only if a cassette falls off onto the floor (its lowest point 20 mm under the table top).
Success: 1.0 held for 1 s. Doing nothing scores 0. A cassette pressed in but left to lean out earns 0 for its side
(tilt over 3 degrees). Seated cassettes with the arms open would earn 0.6, but only while a hand holds them flat, and a
held state never counts, so in practice the arms must be shut.

## The demo

Left arm: picks gel A from its rack at its middle (pads across its thickness), turns it a quarter turn on the way over,
lowers it outboard of the +y gasket, moves it in against the gasket, slides it down into the support, lets go (it leans
out) and goes aside. Right arm: the same with gel B on the -y gasket. Left arm: straddles both cassettes near their far
ends with its fingers open across the pair and closes to 4 mm inside the pair's thickness, squeezing both flat on their
gaskets. Right arm: with its fingers closed, comes down behind the -y arm's handle 180 mm from its pin and pushes it
round, slowly over the cam, to shut; the clip catches. The left hand opens (the +y cassette leans out again) and goes
aside; the right hand straddles and squeezes; the left arm pushes the +y arm shut; the right hand lets go; both park.

## Results

GPU physics evaluation (Isaac Lab, L40S), job 20261123T035336. The offline checker's predictions are kept for comparison.

| Run | Offline prediction | GPU score |
|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, settle) | 0.0 | 0.0 (nothing moves) |
| Scripted demo | 1.0 | 1.0 (success at step 1441 of 1800) |
| `DEMO_IDLE_ARM=right` (left alone) | 0.0 | 0.5 (one cassette clamped; the other never seated) |
| `DEMO_IDLE_ARM=left` (right alone) | 0.0 | 0.5 (one cassette clamped; the other never seated) |
| Dream-1 (one trial, full run limit) | not predicted | 0.0 |

With one arm idle, the working hand clamps the cassette on its own side, but no hand is left to hold the second
cassette upright against its gasket while the other arm is closed, so the score stops at 0.5.

The four gap points were moved after the GPU run from 35 mm to 46 mm either side of the middle, so that they lie on the gasket legs where a feeler gauge can reach. On the GPU run the cassettes lay within 0.1 degree of the gasket plane with gaps under 0.3 mm, which moves the gaps at the new points by under 0.03 mm (computed), so the result stands.

## Fidelity

Rigid bodies: the gaskets do not compress and the gel is part of the short plate. The cam and clip are one declared
rule on each arm's hinge spring target: free, the target follows the arm (no drag); in the last 12 degrees with a
cassette flat on its gasket it is set 0.05 rad behind the arm (0.1 N m opening); 2 degrees short of shut it catches and
holds the arm 0.05 rad past shut (0.1 N m pressing the cassette); with no cassette it never catches. The 12 degree lean
comes from the groove and lip geometry. Estimates are marked (est).
