# Return a drained houseplant to its saucer on the window sill, beside another plant

After watering at the kitchen sink a houseplant drains in the sink drainer. It then goes back to the window sill behind the worktop (a deep sill flush with the worktop, as in many kitchens): lifted
out, carried over the sill tray's edge and the saucer's rim, set down in its drip saucer, and let go, so the last
drips fall in the saucer and not on the wood. Here a 3 inch square nursery pot (0.20 kg, black, dry potting mix and an artificial succulent) stands in a drainer near the right arm. On the sill, in a black tray with two bays, the saucer waits in
the right bay and another plant (artificial; its own saucer and pot, a dowel stem and a 160 mm canopy of artificial leaves, 0.30 kg) stands in
the left bay with its canopy reaching over the right bay's inner edge. Behind the tray is the window: glass in a
white frame. The robot carries the pot over, seats its foot in the saucer's well (7 mm clear each side), lets go and
takes its hand away. The saucer must stay in its bay and the other plant must not be knocked.

## Why it is hard

The distinct difficulty is the seat, not the pick-up: a 58 mm foot into a 72 mm well, square to it, in a loose saucer that must stay in its bay, with a top-heavy neighbour 14 mm away that must stay untouched. Standing a pot upright on a bare surface is much easier.

1. **The fit.** The pot's foot is 58 mm; the well is 72 mm. The pot is square and so is the well: it must also go
   in square (sides within 10 degrees). A pot set down on the rim does not seat.
2. **Getting there.** The pot is lifted over the drainer's 15 mm edge and carried over the saucer's rim and the tray's
   edge (15 mm over the table) before it is lowered.
3. **The crowded sill.** The other plant's canopy hangs over the right bay's inner edge, 150 mm up. The gripper's
   housing and the pot must stay clear of it (close the fingers along x, so the 65 mm housing stays 19 mm away).
4. **Hands off.** The score is read with both grippers 30 mm clear and all three parts still for 1 s: a held pot
   earns nothing.

## Scoring

| Condition | Rule |
|---|---|
| Seated | the pot's four foot corners inside the saucer's 72 mm well (1 mm allowed) and 2 mm below to 3 mm above its floor; axis within 5 degrees of the saucer's; sides within 10 degrees of the saucer's; axis within 8 mm of the saucer's |
| In the bay | saucer tilted under 3 degrees; its foot's four corners inside the 102 mm bay square (1 mm allowed); within 2 mm of the tray floor's height (not on a stop) |
| Other plant undisturbed | never more than 3 mm from its place in any direction (a lift included), never tilted more than 3 degrees (latched) |
| Hands off | last 1 s: both finger zones 30 mm or more from the pot, the saucer and the other plant; each moves under 5 mm and turns under 5 degrees (visible motion; the GPU demo ran at 0.0 mm and 0.0 degrees) |
| Faults (latched) | pot axis more than 30 degrees from vertical; other plant more than 30 degrees; any of the three more than 10 mm below the table top |

| Outcome | Score |
|---|---|
| Seated, in the bay, other plant undisturbed, hands off | 1.0 |
| Seated, saucer still on the tray, but the saucer out of its bay or the other plant moved or tilted (not toppled) | 0.5 |
| Seated, but the saucer's centre outside the tray's raised edge (carried off the sill) | 0 |
| Not seated: pot upright (within 10 degrees), its foot's centre over the tray's outline and at most 20 mm above the table, other plant undisturbed | 0.25 |
| Anything else, a fault, or not hands off | 0 |

Doing nothing scores 0 (the pot is outside the tray). Success: 1.0 held for 1 s.

Bench check for each condition on a real build:

| Condition | Bench method |
|---|---|
| Foot height (2 mm / 3 mm) | feeler gauges under the foot at the corners |
| Axis (5 and 10 degrees) | phone level on the pot's side; try-square for the sides |
| Axis offset (8 mm) | rule across the saucer's rim each way |
| Saucer in bay | 102 mm square template and a ruler; 2 mm feeler under the foot |
| Other plant (3 mm / 3 degrees) | 3 mm spacer block against its saucer; protractor |
| Hands off | 30 mm spacer against each hand; video of the last 1 s (nothing visibly moves) |
| Tip / fall | protractor and a ruler; by eye |

## Demo, run limit and runtime

The right arm alone does it: down round the pot's middle layer (pads 22 mm in a 24 mm layer, fingers commanded to 57
mm across 65 mm), lift 32 mm, carry 21 cm over the right bay, lower 2 mm past the well floor, let go (1.5 s), up,
away. 21 s (420 steps); success predicted offline at step 443; the policy stops at 481. The run limit is 1,200 steps
(1 minute). The left arm stays at its ready pose. One arm is enough: `DEMO_IDLE_ARM=left` is the same demo (offline
prediction 1.0, GPU 1.0); `DEMO_IDLE_ARM=right` scores 0 (nothing moves; GPU 0.0). This task does not claim to need two arms.

Switches: `DEMO_HOLD_ONLY=1` (settle check), `DEMO_IDLE_ARM=left|right`, `TASK_TRACE=trace.jsonl` (one JSON line per
control step: foot height over the well floor, pot off-centre, other plant's move, saucer's offset from the bay).
Trial runtime on an L40S once Isaac Sim is running: about 1 to 1.5 minutes; the first start on a new instance takes
several minutes more. Layout is fixed (`-E object_jitter=0`): no seed.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The watered houseplant has drained in the sink drainer. Carry it to the window sill and set its pot down in its saucer in the tray, then let go. Leave the saucer in its bay and the other plant where it is." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/plant-pot-saucer/demo
```

Use `--policy dream` for Dream-1. One trial per process. Offline checks, no simulator:
`python3 -m unittest discover -s tests -v`, then `python3 tools/offline/check_task.py tasks/<name> --require-json --strict`.

## Simplifications and what to read on the GPU

Rigid bodies; stepped taper (3 layers) and a two-step saucer; the pot is solid; no water; one rigid disc for the
canopy; friction 0.5 on all plastics (estimate). The offline checker's carry-and-drop stand-in (`task.py`,
`OfflineModel`) predicted 1.0 before the GPU runs, which are in Results in simulation below. Things to read in `TASK_TRACE`:
`foot_dz_max_mm` (0 to 3 at the end), `pot_centre_off_mm` (under 7 for a foot that fits), `nb_move_mm` (under 3 all
trial), and that the arm's lag leaves the foot within the well's 7 mm. The demo, not the scoring, is tuned on the GPU.

## Real build and sources

Plants are artificial so every trial is the same: the pot holds dry potting mix and sand (0.20 kg in all) under a plastic or foam succulent rosette pushed into a card disc that covers the mix so nothing spills on the gripper pads; the second plant is a matching pot and saucer with sand to 0.30 kg, an 8 mm dowel and a 160 mm canopy of artificial leaves or a foam disc. A 3 inch square nursery pot and a matching square saucer, a tray with a raised edge and two bays marked with 3 mm
stops (or tape), a pane in a frame, or a board standing in for it, with the frame's face 38 mm behind the tray's back edge, a second plant. Retail 3 inch square
pots range from about 69 mm to 84 mm wide and 78 to 89 mm tall (listings on groworganic.com, printables.com's 76 mm
model, retail multipacks); our 78 mm rim, 58 mm foot, 78 mm height are inside that range and are design values: measure
your pot and saucer and keep the 7 mm well clearance. The saucer, stepped taper, canopy and tray sizes are estimates.
No third-party assets or meshes: every part is a box or cylinder.

## Files

`task.json` (submission text), `task.py` (objects, hooks, demo, offline stand-in), `layout.py` (every dimension),
`scoring.py` (the scorer), `mechkit.py` and `rigstate.py` (the program's compound-body and arm-reading helpers,
unchanged), `tests/test_scoring.py` (84 offline tests: every tier, boundary, latch, NaN, reset and exploit).

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control, 1,200-step limit), one trial each. The settle run
(`DEMO_HOLD_ONLY=1`, nothing commanded) scored 0.0.

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
