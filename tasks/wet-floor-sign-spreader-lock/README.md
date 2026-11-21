# Spread a wet-floor sign to its hinge stop with both hands and stand it at the spill

Janitors and building cleaners (O*NET 37-2011.00, 2,432,600 US jobs in 2025) "Clean spills and other hazards with
appropriate equipment" (BLS duties); the wet-floor sign goes out open at the spill first. A two-leaf A-frame sign
modelled on the Rubbermaid Commercial FG611277YEL (0.55 kg here, leaves 280 x 610 mm, printed warning on both faces)
stands folded where it was set down, at the edge of a 300 x 200 mm spill on a corridor floor. Its ridge runs across the
corridor (along x), so its faces look along it (along y). The robot must draw the leaves apart, one hand on each leaf's
grab handle, until the top hinge reaches its moulded stop (25 degrees between the leaves, feet 264 mm apart), then let
go: the sign stands by itself, square to the spill, its +y foot 40 to 140 mm from the spill's edge, no foot pad on it,
and its middle within the spill's length. The instruction does not say how far the stop is or where to stand it.

## Why it is hard

1. **Open all the way, blind.** The hinge stops the leaves, so over-pulling does no harm, but stopping short leaves a
   narrow base (a sign that is knocked or blown over). The hands cannot see the stop: they must draw both handles along
   their arcs (each leaf swings about the hinge, 610 mm up) until the leaves stop moving.
2. **Nothing holds the sign down, so both hands pull at once.** The sign is free on its pads. Drawn by one leaf, it
   slides toward the hand instead of opening: on the GPU, with either arm idle, the same demo opened the hinge only
   8.7 to 9.2 degrees (feet 93 to 98 mm apart, against 264 mm at the stop), the other foot slid 22 to 25 mm, and it
   scored 0.25. Drawn by both leaves outward, the two pulls cancel and the sign opens in place, symmetric about its
   hinge line; its +y foot then ends 86 mm from the spill (GPU). One hand could still open the sign in several moves and
   drag it back into place; the scorer reads the end state only, so that would also score 1.0.
3. **Hands off.** The score is read with both grippers 50 mm clear and the sign still for 2 s.

## Scoring

| Condition | Rule |
|---|---|
| Open | distance between the two foot lines 244 to 270 mm (the stop gives 264 mm; 244 mm is 1.9 degrees of hinge short) |
| Lean | each leaf 9 to 16 degrees from vertical |
| Placed | nearer foot line 40 to 140 mm from the spill's near edge (along y); neither foot pad over the spill (each pad tested on its own, top view) |
| Beside | the hinge's middle within the spill's 300 mm length (x 200 to 500 mm) |
| Square | ridge (hinge line) within 10 degrees of the spill's long edge |
| Stands | both foot lines within 10 mm of the floor; last 2 s: foot lines and hinge move under 3 mm, spread changes under 2 mm, leaves turn under 1 degree |
| Hands off | last 2 s: finger zones 50 mm or more from every part of the sign |

| Outcome | Score |
|---|---|
| All conditions | 1.0 |
| All but hands off | 0.75 |
| Stands and open, ridge within 30 degrees, no pad on the spill, otherwise (lean, placement band, beside, ridge 10 to 30 degrees) | 0.5 |
| Stands, spread at least 70 mm, otherwise (not open to the stop, ridge over 30 degrees, pad on the spill) | 0.25 |
| A leaf over 45 degrees from vertical, not standing, feet off the floor, or spread under 70 mm (do nothing) | 0 |

Bench check for each condition on a real build:

| Condition | Bench method |
|---|---|
| Open (244 to 270 mm) | tape between the foot lines |
| Lean (9 to 16 degrees) | digital inclinometer on each leaf's face |
| Placed (40 to 140 mm) | rule from the nearer foot line to the spill edge; look from above for a pad on the spill |
| Beside (300 mm) | tape along the spill's edge to the point below the ridge's middle |
| Square (10 degrees) | 10 degree protractor between the ridge and the spill's long edge |
| Stands | 10 mm spacer under each foot; 2 s by the clock with a pencil mark at each foot and a 3 mm gauge; tape for the 2 mm spread change |
| Hands off (50 mm) | 50 mm spacer block against each hand |
| Zero gate (45 degrees) | inclinometer at 45 degrees on each leaf |

Success: 1.0 held for 1 s on top of the scorer's 2 s window. Doing nothing scores 0. The score is read from the end
state only.

## Results

GPU runs: L40S, Isaac Lab-Arena, object jitter 0, 20 Hz control, one trial each;
rows marked not run were not run on the GPU. The offline prediction column comes from `tools/offline/check_task.py`
with this task's stand-in for the physics (`task.OfflineModel`); it is not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, settle) | both arms hold still, 160 steps | 0.0 | 0.0 (sign moved 0 mm) |
| Scripted demo | below | 1.0 (held from step 478) | 1.0 (hinge 25.0 degrees, feet 264 mm apart, near foot 86 mm from the spill) |
| `DEMO_IDLE_ARM=right` (left alone) | left hand draws leaf B only | 0.25 | 0.25 (hinge 8.7 degrees, feet 93 mm apart) |
| `DEMO_IDLE_ARM=left` (right alone) | right hand draws leaf A only | 0.25 | 0.25 (hinge 9.2 degrees, feet 98 mm apart) |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=short` | each leaf drawn 60 percent of the way | 0.25 | not run |
| `DEMO_VARIANT=held` | the right way, hands keep hold | 0.75 | not run |

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both hands work at once, each
on its own side: the right hand on leaf A (-y), the left on leaf B (+y). Each hand comes straight down over its
leaf's grab handle with the fingers open 60 mm along x and closes across it (12 mm commanded on 20 mm).

| Time (s) | Each hand |
|---|---|
| 0 to 2 | over its grab handle (TCP 155 mm up, gripper turned to close along x) |
| 2 to 5 | down round the handle to 75 mm |
| 5 to 7.6 | settle, close (1.5 s), settle |
| 7.6 to 15.6 | draw the leaf out along its arc in four moves to 1 degree past the stop (the stop takes it up) |
| 15.6 to 16.6 | hold against the stop |
| 16.6 to 21.6 | let go (1.5 s), settle, straight up, aside to x 200 mm, y +-400 mm |

432 steps (21.6 s). Closest TCPs 24 cm; closest arm capsules 71 mm; the wrists stay 12 mm or more clear of the leaves
(capsule model against the leaves' real boxes, `tests/`); worst IK 0.08 mm; smallest joint margin 10 degrees.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1000
```

Runtime: a 1,000-step trial takes about 2 to 3 minutes of wall-clock on the L40S including start-up (the demo itself is
432 steps, 22 s of simulated time). Seed: nothing is random; `object_jitter=0` fixes the start layout.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: spread, leaf angles,
near gap, ridge yaw, beside, hands gap, stillness and release.

## Simplifications

- Rigid leaves on a revolute hinge with a hard end stop (no snap detent); no software rule acts on the sign.
- The grab handle on each leaf is a pull handle sized for a top-down gripper, not a feature of the stock FG611277YEL
  (whose only handle is its top carry handle, not modelled): a solid fin
  standing 140 mm out from the face, so the closed gripper's wrist (about 52 mm round) clears the 610 mm leaf. On a
  stock sign a person pulls the leaf's edge; any top-down grip within about 100 mm of the leaf puts the wrist into it
  (capsule model, `tests/`). A real build fits the same fin to each leaf. Mass 0.55 kg (the stock sign ships at 1.95 lb), pad friction 0.6 and
  hinge friction 0.02 N m are estimates. Drawing a leaf takes about 2 N at the hand (quasi-static estimate in `tests/`).
- Carrying the folded sign to the spill is not modelled: it starts just set down at the spill's edge. The print is a
  black panel 200 x 150 mm.
- The station table stands for the floor; a 300 mm block stands for the corridor's side wall.

## Sources

- Janitors and building cleaners, SOC 37-2011, 2,432,600 jobs in 2025: BLS Occupational Outlook Handbook,
  https://www.bls.gov/ooh/building-and-grounds-cleaning/janitors-and-building-cleaners.htm
- Janitors' duty "Clean spills and other hazards with appropriate equipment": the same BLS page.
- O*NET 37-2011.00 Janitors and Cleaners, Except Maids and Housekeeping Cleaners: https://www.onetonline.org/link/summary/37-2011.00
- Rubbermaid Commercial FG611277YEL, 11 x 12 x 25 in, 1.95 lb shipping weight, with handle, 14.99 USD:
  https://www.webstaurantstore.com/rubbermaid-fg611277yel-25-yellow-double-sided-wet-floor-sign-caution-wet-floor/690611277YEL.html
  The simulated sign keeps its 11 in width; leaf length (610 mm), open depth (264 mm at the feet) and mass are estimates.

## Real build and reset

A Rubbermaid FG611277YEL sign with a grab fin fitted low on each leaf (or one built from two 280 x 610 mm sheets, a
hinge with a stop at 25 degrees and two grab fins), a 300 x 200 mm blue mat, a tape, an inclinometer, a protractor and two
spacers. Reset: fold the sign, ridge along x, its middle at x 330 mm, y +10 mm, leaf A facing -y; mat at x 200 to
500 mm, y 232 to 432 mm.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension and the sign's kinematics; the scorer |
| `mechkit.py`, `rigstate.py` | Build the sign as an articulation; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
