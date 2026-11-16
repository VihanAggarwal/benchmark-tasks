# Close a rigid telescoping-lid box over its product without wedging the lid

A rigid two-piece setup box stands on the table: an open base (70 x 114 x 40 mm, 2 mm greyboard) with a pocket laser
distance meter in its hard case lying in it, and beside it the matching telescoping lid standing top up. The lid's inside
is 2.5 mm larger than the base's outside on every side. The robot must pick the lid up, bring it over the base, lower it
level and in line so that it telescopes down over the base's walls until its top panel rests on the base's rim, leave
the box where it was packed with the product inside, and let go.

## Why it is hard

1. **A few millimetres of play.** The lid goes over the base with 2.5 mm to spare each side. Out of line by more
   than that, it lands on the base's rim and rests there (0.25); yawed or tilted by a couple of degrees it wedges half
   on (0.5). With these sizes the lid's inside clears the base's corners only up to about 2.5 degrees of yaw
   (`layout.lid_yaw_ok_limit_deg`), so the 1.5 degree gate sits inside the geometric limit.
2. **A thin-walled lid in a tight grip.** The pads must close on the lid's two long skirt walls (79 mm across, 94 mm
   open: 7.5 mm a side) with the pad tips 6 mm above the table. Where the lid ends up in the hand is where the grasp
   left it, so the arm has to put the lid, not the hand, over the base within the 2.5 mm of play.
3. **Do not move the box.** The base is a 50 g box on a table (friction 0.45): a lid pressed on out of line pushes it
   around. Over 20 mm, off the table or tilted over 2 degrees scores 0.
4. **Hands off.** Both grippers 30 mm clear and nothing moving for 1 s.

Only one arm is needed: the lid, the base and the product are all inside the right arm's reach. The scripted demo uses
the right arm alone; the left arm holds the ready pose. The task is not claimed to need two arms (see Results).

## Scoring

| Condition | Rule |
|---|---|
| Seated | the lid's top outer face within 1.5 mm of 42 mm over the table |
| Square | the lid's yaw within 1.5 degrees of the base's (modulo 180) and its top face tilted under 1.0 degree |
| Centred | the lid's centre within 3.5 mm of the base's (top view) |
| Engaged | the lid upright (tilt under 10 degrees), centre within 8 mm of the base's, skirt's lowest point 10 mm or more below the base's rim |
| On the base | the lid upright within 30 degrees, centre within 25 mm of the base's, lowest point 4 mm or more over the table |
| Box still | the base within 20 mm of its start (top view), within 3 mm of the table, tilted under 2 degrees |
| Hands off | last 1 s: finger zones 30 mm or more from the base, lid and product; each part moves under 1 mm, turns under 0.3 deg |
| Fell (latched) | the lid or base more than 10 mm below the table top, or the product's centre outside the base's inside |

| Outcome | Score |
|---|---|
| Seated, square and centred | 1.0 |
| Engaged otherwise | 0.5 |
| On the base otherwise | 0.25 |
| Anything else, fell, box moved, or not hands off and still | 0 |

Possible scores 0, 0.25, 0.5, 1.0. Success: 1.0 held for 1 s. Doing nothing scores 0. A lid upside down on the base, or
standing on the table, scores 0.

Bench check for each condition on a real build:

| Condition | Bench method |
|---|---|
| Seated | calipers or a steel rule from the table to the lid's top face, 1.5 mm of 42 mm (a real box: 2 x the board plus the base's height) |
| Square | a try-square against the lid's side and the base's, and a phone level on the lid's top (1.0 degree) |
| Centred | a 0.5 mm feeler gauge in the gap on each side: both sides within 3.5 mm of each other's centring |
| Engaged | a rule against the skirt's rim, 10 mm below the base's rim |
| On the base | by eye and a rule: lid resting on the rim within 25 mm of centred |
| Box still | a pencil outline of the base before, a 20 mm rule after, a 3 mm feeler under it, 2 degree phone level on its top |
| Hands off | a 30 mm spacer against each hand, 30 fps video over the last 1 s, a 1 mm mark on the table |
| Fell | a 10 mm drop by eye; the product seen inside the base |
| Dent (real trial only) | a wall dented or creased more than 2 mm, a 2 mm feeler: score 0 |

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. All waypoints are
straight-down reachable (worst IK 0.1 mm, 0.01 deg offline). The right arm only:

| Time (s) | What |
|---|---|
| 0 to 7.0 | over the lid, down round it (pad tips 6 mm over the table), close the pads on the long skirt walls (71 mm commanded, 8 mm under the lid) |
| 7.0 to 9.0 | lift 50 mm (rim 10 mm over the base's rim) |
| 9.0 to 13.5 | carry over the base, settle |
| 13.5 to 15.5 | down to 3 mm over the base's rim, settle |
| 15.5 to 19.0 | lower the lid over the base's walls in 3 s (the rim goes from 43 mm to 6 mm), 1 mm past the seat, settle |
| 19.0 to 23.5 | open the pads (1.5 s), straight up, aside |

Demo 23.5 s (470 steps, 39 percent of the 1200 step limit); the scripted policy stops 3 s after the plan (531 steps).
`X_TRIM` and `Y_TRIM` in `task.py` shift where the lid is put over the base: the pads are not centred exactly on the
commanded point (about 8 mm measured in other tasks), so they are the calibration to tune on the GPU.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

Runtime: a trial takes about 3 to 5 minutes of wall-clock on the L40S including start-up (estimate; the demo is 470
steps). Seed: nothing is random; `object_jitter=0` fixes the start layout. For Dream-1 use `--policy dream`.
`TASK_TRACE=trace.jsonl` writes one line per control step: the lid's top height, offset, yaw gap and tilt, the booleans
for each condition and a 3 s `hist`. `DEMO_HOLD_ONLY=1` is the settling check (nothing moves), `DEMO_IDLE_ARM=left|right`
the one-arm check, `DEMO_VARIANT=off_x|off_y` puts the lid over the base 5 mm out of line (counterfactual, never the
submitted demo).

## Simplifications

- Rigid bodies. The base and lid are compounds of 2 mm boxes (floor and four walls; panel and four skirt walls); nothing
  flexes, dents or tears. A real lid's skirt can bow and a real trial adds a dent check. There is no software rule: the
  task is rigid bodies and friction (mechkit compounds).
- The 2.5 mm play, the 0.45 friction, the box sizes and the masses (50 g each for base and lid, 120 g for the product) are
  estimates: measure the real box. A real telescoping lid's fit varies with the board and wrap.
- The product is a solid block of the instrument's outer size and mass; the case's rounded corners, foam and the paper
  wrap are not modelled.
- The pads close on a 2 mm wall at 20 N a side; a real grip on a flexible skirt is gentler.

## Sources

- Packers and packagers, hand, SOC 53-7064, about 0.6 million US jobs: BLS Occupational Outlook Handbook, material
  moving occupations (hand packers and packagers).
- Two-piece rigid ("setup") boxes with a telescoping lid: standard packaging form; the sizes here are our estimates for a
  small electronics box (base 70 x 114 x 40 mm outside, 2 mm board) and the pocket laser distance meter's case
  (55 x 105 x 28 mm, 0.12 kg is est from catalogue figures for this class of instrument).
- Gripper and reach limits: the program's station figures.

## Real build and reset

The real trial needs one two-piece rigid box (base 70 x 114 x 40 mm, 2 mm board, lid 2.5 mm larger inside on each side;
measure the real one with calipers and feeler gauge and use its sizes), a pocket instrument about 55 x 105 x 28 mm of
about 0.12 kg, and a table with pencil marks for the base's footprint and the lid's start. Reset: put the instrument in
the base (long side along y), base on its marks, lid standing top up on its mark. Replace a box whose walls are dented or
creased more than 2 mm.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `scoring.py` | Every dimension; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.

## Results

GPU runs (L40S, Isaac Lab-Arena, object jitter 0, 20 Hz control, one trial each) next to the offline prediction
(`tools/offline/check_task.py --require-json --strict --one-arm`, a kinematic check with no physics, not a measurement).

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Scripted demo (`demo()` as written) | right arm closes the box | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=left` (right arm alone) | the same | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=right` (left arm alone) | nothing moves | 0.0 | 0.0 |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |
| `DEMO_VARIANT=off_x` / `off_y` | lid 5 mm out of line | 0.25 | not run |
| `DEMO_HOLD_ONLY=1` | nothing moves | 0.0 | not run |

One arm is enough (measured on the GPU), so the task does not claim to need two.
