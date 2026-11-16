# Pack six eggs point-down from an egg flat into a half-dozen carton and close its lid

Six USDA large eggs (57 g, 57 x 44 mm) stand pointed end down in a 30-pocket moulded egg flat on the left. In the middle
a half-dozen (2 x 3) moulded-pulp carton sits in a fixed pack-out cradle, its lid hinged along the side toward the arms
and standing open at 110 degrees against its hinge stop. The robot must stand each egg pointed end down in a carton
cell without dropping or knocking it hard (a cracked egg scores 0), close the lid, and let go.

## Why it is hard

1. **Fragile, upright, close fit.** An egg is 44 mm wide in a cell 52 mm across at the rim and 20 mm at the floor, so it
   must be lowered onto its pointed end within a few millimetres of the cell's centre and left standing, blunt end 28 mm
   over the rim. An egg let go high, tipped or wedged on the rim does not count and stops the lid.
2. **Neighbours.** The cells are 52 mm apart: two seated eggs leave 8 mm between their widest rings. The pads grip the
   egg's blunt-end shoulder (a 38 mm wide disc, 43 to 47.5 mm up from the point), where neighbours are 14 mm apart, and
   open only to 45 mm to let go, so a pad never lands on a seated egg.
3. **The lid.** Open at 110 degrees it leans back toward the arms. A closed gripper's pad has to push its outer face
   past vertical along its arc (the lid is 111 mm deep, 10 g); the lid then falls shut under its own weight. Pushed
   only to 100 degrees it swings back open (0.75).
4. **Two hands.** The left arm moves the eggs (each arm reaches the table 0.10 to 0.50 m in front of its base, about 10 cm
   past the midline at low height); the right arm closes the lid once the left has parked.
5. **Hands off.** Scored with both grippers 30 mm clear and nothing visibly moving for 1 s (under 5 mm and 5 degrees).

## Scoring

| Condition | Rule |
|---|---|
| Seated | egg's centre within 12 mm of a cell centre (top view), within 6 mm of the seated height (blunt end no more than 6 mm high), long axis within 25 degrees of vertical, pointed end down; one egg per cell |
| Lid closed | free edge within 5 mm of its closed place (lid joint within 0.045 rad of closed) |
| Cracked (latched) | an egg ever moving at 0.8 m/s or more (a fall of about 33 mm or a throw), or lost off the table (centre 10 mm under the table top) |
| Hands off | last 1 s: finger zones 30 mm or more from the carton, lid and eggs; every egg moves no more than 5 mm and turns no more than 5 deg; the lid turns no more than 5 deg (motion a person can see on video) |

| Outcome | Score |
|---|---|
| Six seated, lid closed, hands off | 1.0 |
| Six seated, lid not closed | 0.75 |
| Four or five seated | 0.5 |
| One to three seated | 0.25 |
| Any egg cracked, none seated, or not hands off | 0 |

Possible scores 0, 0.25, 0.5, 0.75, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0.

Bench check for each condition on a real build:

| Condition | Bench method |
|---|---|
| In a cell | look from above; 12 mm mark on a card at each cell centre |
| Seated height | straightedge across the carton rim, ruler: blunt end no more than 34.5 mm over the rim (a seated egg stands 28.5 mm over it; measure yours and set the figure from it) |
| Pointed end down, 25 degrees | 25 degree protractor or card template held against each egg's long axis |
| Lid closed | 5 mm feeler gauge at the lid's free edge, all along it |
| Cracked | candle each egg, or look at the shell and the carton for a crack or a leak; a drop of 60 mm or more onto a hard surface counts as cracked |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s showing nothing move (5 mm mark on the carton, 5 degree card); the eggs inside the closed carton are judged by the carton and lid, then lift the lid and check them |

The sim's crack rule (0.8 m/s, about a 33 mm fall) is a stand-in for the real shell: the real figure is the egg that
cracks, found by candling.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. All waypoints are straight-down
reachable (worst IK under 0.1 mm). The left arm, per egg: open over the egg, down beside its blunt end (pad tips 48.5 mm up,
16.5 mm over the pocket rim), close to a 33 mm command, lift to 135 mm, carry over its cell, lower until seated, open to
45 mm, straight up. The right arm waits out of the way, turns its closed fingers across x, sets its pad tip 3 mm into the
lid's outer face 90 mm from the hinge and pushes along the lid's arc from 110 to 50 degrees, backs off and waits for the
lid to shut.

| Time (s) | Left gripper | Right gripper |
|---|---|---|
| 0 to 96 | six eggs, one after another (15.6 s each), then parks | out of the way |
| 96 to 114 | parked | to the lid, push it from 110 to 50 degrees, back off |
| 114 to 125 | parked | out of the way; the lid shuts under its weight |

Demo about 124 s (2480 steps, 69 percent of the limit); the arms stay 13 cm or more apart (capsules).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 3600
```

Runtime: a trial takes about 4 to 6 minutes of wall-clock on the L40S including start-up (estimate; the demo is about
2480 steps). A Dream-1 trial at the full limit takes longer. Seed: nothing is random; `object_jitter=0` fixes the start
layout. For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the eggs' heights,
positions and speeds, the lid's angle, the let-go gap, and a 3 s `hist`.

## Simplifications

- Each egg is a stack of twelve discs turned on its axis (44 x 57 mm, 57 g, blunt end larger and tinted), not a smooth
  ovoid. The pulp cells are four 2 mm slabs each (a square funnel); pulp and shell are rigid. A real carton flexes and
  its lid has a tab and a tuck; here the lid is a hinge with a stop and it counts as closed within 5 mm of its closed place.
- The carton is fixed in its cradle (a real one rests in a pack-out cradle with a back stop and does not move); the cradle
  is four 6 mm rails.
- A crack is a declared rule read from an egg's speed: shell strength and grip force are not simulated, and the sim's pads
  squeeze at about 12 N a finger (5 mm under the shoulder's width), below a real shell's crush load (est about 35 N).
- The flat is partly emptied: only six of its 30 pockets hold an egg, in two columns three pockets apart, so no egg
  stands beside another along the gripper's closing axis.
- Sizes (carton 108 x 160 mm, cell 52/20/32 mm, flat pitch 51 mm), friction (0.45 shell, 0.50 pulp), the lid's 10 g and
  its hinge damping (0.004 N m s/rad) are estimates from typical cartons; measure your own.

## Sources

- Packing eggs into a carton is a familiar home and farm-stall job; no occupation count is claimed.
- USDA large egg: 2 oz (57 g) average, about 57 mm long x 44 mm wide (USDA egg grading standards; the dimensions are the
  commonly quoted averages).
- Half-dozen moulded-pulp carton and 30-count flat: typical catalogue sizes (est).

## Eggs: storage, handling, stand-ins

- Keep the eggs at 4 C in their own carton; take them out 30 to 60 min before a trial so they are dry and at room
  temperature (20 to 22 C), and use them the same day. Cold eggs sweat and slip in the pads.
- Discard any egg with a hairline crack before the trial and replace any egg that cracks or is dropped with a fresh one.
- Cleanup: wipe a break up at once with paper towel, wipe the pads, flat and carton with a damp cloth, and swap a soiled
  carton or flat.
- Rig-safe option: shell-on hard-boiled eggs (12 min, cooled, dried) leave nothing wet on the grippers and show a crack; ceramic
  or wooden eggs of the same size (57 g) are a poorer stand-in because they cannot crack.

## Real build and reset

The real trial needs a half-dozen pulp carton, a 30-pocket flat and six fresh large eggs; a cradle (four rails or a
carton jig) keeps the carton still. Reset: stand six fresh eggs (stored and handled as above) pointed end down in the flat's pockets listed, empty the
carton, open its lid to 110 degrees and seat the carton in its cradle. Discard any egg that cracked or was dropped.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound bodies with a hinge; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.

## Results

GPU runs (L40S, Isaac Lab-Arena, object jitter 0, 20 Hz control, one trial each) next to the offline prediction
(`tools/offline/check_task.py`, a kinematic check, not a measurement).

| Run | Offline prediction | GPU score |
|---|---|---|
| Scripted demo (`demo()` as written) | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=right` (left arm alone) | 0.75 | 0.75 |
| `DEMO_IDLE_ARM=left` (right arm alone) | 0.0 | 0.0 |
| Dream-1 (`--policy dream`, full limit) | not predicted | 0.0 |
| Lid pushed only to 100 degrees | 0.75 | not run |
| An egg let go 70 mm over its cell | 0.0 | not run |
| Five eggs | 0.5 | not run |

The GPU demo's 1.0 was run under the stricter 1 mm and 0.3 degree stillness check (last-second motion 0.02 mm and
0.12 degrees); the scorer now uses the visible 5 mm and 5 degrees, which cannot lower it. The GPU settle run reads 0.0
because nothing is packed, as expected.
