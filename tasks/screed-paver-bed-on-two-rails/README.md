# Screed a paver sand bed flat: a board riding on two steel rails

The bedding course of a two-paver repair, 200 mm (x) by 400 mm (y), sits inside an edging of three fixed steel flats
(the near side is open toward the robot). Two fixed steel rails along x, 25 x 25 mm, stand for the bed thickness: their
tops are 25 mm up. The sand is raked deeper (36 mm) with a mound (45 mm) and a hollow (19 mm). A loose aluminium
screed board (25 x 50 mm tube, 400 mm long, on edge, 0.24 kg) lies on two rest blocks in front of the bed. The robot
fetches the board, lays it across both rails at the far end, pulls it 175 mm toward itself with both hands in step so
that it cuts the sand down to the rail tops and fills the hollow from the heap it carries, lifts it clear, sets it back
on its blocks and leaves the bed untouched.

## Why it is hard

1. **The board must ride both rails for the whole pull.** The scorer takes the board's measured pose every control
   step: sole height, tilt and skew decide what is cut and filled. A board 3 mm off a rail is not "bridged"; one end
   3 mm higher than the other leaves the outermost cells (past each rail) 4 mm high: off the 3 mm tier.
2. **The sand cannot be seen to be right from the pose alone.** The hollow is only filled if the board stays within
   3 mm of the rails while it passes (the heap is carried and dropped into it). A board lifted mid-bed drops its heap
   as a hump; a board pushed the wrong way piles it against the far wall.
3. **The natural first and last steps are part of the job.** The board is fetched from its trestle (rest blocks), laid on
   the rails without touching the 45 mm mound with its sole (carried 20 mm over it), and put back on the blocks at the
   end, hands 30 mm clear, nothing pressed into the finished sand.
4. **The edging limits the work.** The board's ends pass 15 mm from the side walls and its start is 15 mm from the far
   wall; the arms reach over the 60 mm walls to hold the board's ends.

## Scoring

The scorer is `scoring.py`; it reads the board's pose and both grippers once per control step. Every cell of the bed
counts (112 sand cells), so a high spot or a divot anywhere is caught at every tier.

| Condition | Rule |
|---|---|
| Bridged pull | the board's underside within 3 mm of both rail tops (where its centre line crosses them) on every step of a run, its centre travelling at least 150 mm along x in that run |
| Cut | at least 84 of the 112 sand cells (three quarters) within 6 mm of 25 mm |
| Deviation | the largest `abs(depth - 25 mm)` over all 112 sand cells |
| Stowed | underside within 3 mm of both block tops, centre within 10 mm of the blocks' centre line, turned under 3 deg about z, tilted under 3 deg |
| Hands off | last 1 s: both finger zones 30 mm or more from the board, the rails and the bed |
| Still | last 1 s: the board moves under 1 mm and turns under 0.3 deg |
| Pressed | after the board has passed a cell, a pad tip, or any part of a board tilted more than 30 deg (on its side or flat: it cuts nothing), came within 2 mm of its sand or into it |
| Fall (latched) | the board's lowest point 20 mm under the table top (knocked off the table) |

| Outcome | Score |
|---|---|
| Bridged pull, deviation 3 mm or less, stowed, hands off and still | 1.0 |
| Bridged pull, deviation 4 mm or less, stowed | 0.75 |
| Bridged pull, deviation 6 mm or less | 0.5 |
| Bridged pull and cut, otherwise | 0.25 |
| No bridged pull, or not cut, or the fall latch | 0 |
| Pressed | caps the score at 0.5 |

There is no zero gate for rework: the real harm is rework, not injury. Doing nothing scores 0 (no bridged pull, and only 23
cells start within 6 mm: the column under the board's start and the hollow). One high cell left anywhere drops the
score to 0.25, not 0; a board that tips over on the table is not a fall (it is not stowed, and it presses any finished
sand it lies on). Success: 1.0 held for 1 s on top of the scorer's own 1 s of let-go and
stillness.

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| Bridged pull | a side video along each rail, level with its top, a 3 mm gauge block on the rail for scale and a tape with 150 mm marked: the gap under the sole stays below the block on every frame of the run |
| Cut (84 cells) | the deviation readings below: count the cells within 6 mm |
| Deviation (3, 4, 6 mm) | the same at the 112 cell centres (25 mm grid, rail columns skipped), depth = 60 mm minus the gap; or the straightedge on both rail tops at each 25 mm step along x and a 3, 4 or 6 mm feeler along its length (it must not fit) |
| Stowed | 3 mm feeler under the board at each block; a square for 3 deg; a rule for the 10 mm |
| Hands off | 30 mm spacer against each hand; 30 fps video of the last 1 s |
| Still | dial indicator on the board's top; a protractor/level for 0.3 deg |
| Pressed | look for footprints or a tool mark in the finished sand |
| Fall | by eye and a rule |

## Results

GPU runs (L40S, Isaac Lab-Arena, `object_jitter=0`, 20 Hz control, 2,400-step limit, one trial each) next to the
offline predictions (`tools/offline/check_task.py` with this task's stand-in for contact; not a measurement).

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Scripted demo | below | 1.0 (held from step 886 of 921) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the left hand pulls one end | not predictable offline | 0.0 |
| `DEMO_IDLE_ARM=left` (right alone) | the right hand pulls one end | not predictable offline | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | not run |
| `DEMO_VARIANT=short` | pulls only 80 mm | 0.0 | not run |
| `DEMO_VARIANT=high` | pulls 5 mm over the rails | 0.0 | not run |
| `DEMO_VARIANT=nostow` | leaves the board on the bed | 0.5 | not run |

With either arm idle the GPU run scores 0: one hand on one end of the 400 mm board cannot keep it riding both rails.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both arms run the same x and z
on one clock and hold the board's two ends at y +-180 mm; the left hand turns -90 deg, the right +90 deg, so the fingers
close along x across the board's 25 mm.

| Time (s) | Both grippers |
|---|---|
| 0 to 7 | over the board's ends, down beside it (slowly the last 3 cm), grip it across its 25 mm |
| 7 to 8.5 | lift it 10 mm off the blocks |
| 8.5 to 14.5 | carry it over the bed (sole 65 mm up: 20 mm over the mound) to the rails' far end |
| 14.5 to 19 | lower it onto the rails (the last 7 mm slowly, commanded 2 mm under the rail tops: a light press) |
| 19 to 33.5 | pull 175 mm at 12.5 mm/s, both ends in step |
| 33.5 to 37.5 | lift it clear, carry it back over the near edge to the blocks |
| 37.5 to 43 | lower it onto the blocks, let go, step each hand out along the board's axis to y +-285 mm |

Offline: worst IK 0.1 mm; smallest joint margin 16 deg (joint 4); TCPs at least 36 cm apart; arm capsules at least
17 cm apart; pad tips never below 56 mm above the table. The demo takes 860 steps (43 s), 36% of the run limit's 2,400.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 950
```

Runtime: a 950-step trial should take about 2 to 4 minutes of wall-clock on the L40S including start-up (an estimate from
similar tasks, to be measured; the demo is 860 steps, 43 s of simulated time). Seed: nothing is random;
`object_jitter=0` fixes the start layout, so every run starts from the same poses and no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: the score, the pull so
far, the sand's largest and smallest depth, the number of cells off by more than 3 mm, the board's pose and both TCPs
and openings, and every half second the depth of all 128 cells.

## Simplifications

- The sand is a height model kept by the scorer, not particles; a 20 mm slab is drawn under the board's sole and the
  model's heights are not drawn. Per control step (interpolated in 4 mm steps), for every cell under the board's sole,
  the sole's height at the cell comes from the board's measured pose (tilt and skew count); sand above it is cut and
  joins a heap carried in that cell's y column; a lower cell is filled from the heap while the sole is within 3 mm of the
  rail tops; a column lifted more than 3 mm above them drops its heap on the cell just ahead; heap past the near edge
  falls into the spoil; the volume is conserved. The rule never moves the board; the hands do.
- No pulling resistance from the sand is modelled (the board's drag is friction on the rails only).
- The hollow aluminium tube is a solid box of the same mass and outline. The rails and the edging are rigid and fixed.
- Friction is the simulator default (1.2 / 1.0 for every object). Masses from density and size.
- Cell and sand figures are estimates (play sand 1.5 g/mL, 25 mm bedding thickness is the usual figure).

## Sources

- Construction laborers, SOC 47-2061, and landscaping and groundskeeping workers, SOC 37-3011: BLS Occupational
  Outlook Handbook, https://www.bls.gov/ooh/ (construction laborers and helpers; landscaping and groundskeeping workers).
- Bedding course about 1 in (25 mm), screeded on rails and the rails removed afterwards: Interlocking Concrete Pavement
  Institute, Tech Spec 2, Construction of Interlocking Concrete Pavements, https://www.icpi.org.
- 1 in square steel tube for rails and a straight aluminium tube as a screed board are common trade stock (supplier
  listings); the dimensions here are the task's own.

Estimates (stated in fidelity): the sand's density, the board's wall thickness and mass, the rest blocks' size, the
start depths of the sand.

## Real build and reset

Two 25 x 25 mm steel square tubes 215 mm long; three 25 x 60 mm steel flats for the edging; a 25 x 50 mm aluminium tube
400 mm long with 1.5 mm wall; two 40 x 30 x 55 mm steel blocks; play sand; a straightedge, a depth gauge, feeler gauges
and a steel rule. Reset: rake the sand to 36 mm, build the mound and press the hollow with a depth gauge on a
straightedge across the side walls, strike the column under the board's start and the 15 mm strips beside the walls to
25 mm with the board on the rails, set the board on its blocks, centred at x 150 mm. Use damp play sand and rake it level
between runs.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the offline stand-in for contact, `success()`, `score()`, `demo()`, variants |
| `layout.py`, `scoring.py` | Every dimension and the start depths; the sand model and the scorer |
| `rigstate.py` | Reads both grippers each step (the let-go and press checks) |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
