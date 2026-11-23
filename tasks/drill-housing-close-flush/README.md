# Close a BLACK+DECKER LDX120C drill's clamshell over its seated motor and switch

Every cordless drill, hair dryer, glue gun and similar product with a moulded clamshell housing is closed by hand at
the end of its assembly line: the motor-gearbox unit and the trigger switch go into one half, then the other half is
brought down over them and the two halves must meet all round before the screws go in. Both parts straddle the
parting plane, so the second half's own ribs come down over them. A part that is not fully in its cradle holds the
halves apart, and driving the screws then cracks the screw bosses or jams the gears. The fit is blind: once the
second half is over the housing, its ribs and its locating tongue are hidden under its own shell.

Here the open left half of a compact 20 V drill/driver's housing (the BLACK+DECKER LDX120C 20V MAX 3/8 in drill/driver,
https://www.blackanddecker.com/product/ldx120c/20v-max-cordless-drilldriver, is the product a real trial uses; see
"Sizes and sources") is fixed in a three-block nest. The motor-gearbox unit lies on a kitting tray on the
left, the trigger switch stands on the right, and the right half (the cover) lies open side down in front of the nest.
The robot lays the unit in its two stepped cradle ribs with the spindle through the nose, sets the switch in its
pocket with the trigger through its slot, then carries the right half over with both hands, one across the barrel's
nose and one across the battery foot, and lowers it onto the left half. **Closing the right half over a part that
is not seated scores 0.**

**Category: Impact on jobs.** Team assemblers (O*NET 51-2092, counted in a group of 1,403,500 US workers) "work as
part of a team having responsibility for assembling an entire product or component of a product" and rotate through
its stations. On cordless-tool and small-appliance lines one of those stations closes the moulded two-part housing over
the installed motor and switch, the last hand step before the screws; O*NET lists for electrical and electronic
equipment assemblers (51-2022) the task "Assemble electrical or electronic systems or support structures and install
components, units, subassemblies, wiring, or assembly casings". The order is set by the parts, not by a rule: the
unit and the switch go in first because the other half's ribs come down over both. Sources:
https://www.onetonline.org/link/summary/51-2092.00 and https://www.onetonline.org/link/summary/51-2022.00.

## What makes it hard

- **Two parts that straddle the parting plane.** Seated, the unit's axis lies on the parting plane (its top 18 mm
  above it) and the switch's top is 11 mm above it. The cover's ribs come down to 1.5 mm above each part, so a part
  riding 3 mm high (on a rib step or an axial stop) holds the cover up. Turned end for end, the unit does not go
  in (its spindle meets the rear wall, which has no notch, and its can meets the front stop); turned round, the
  switch's trigger meets the pocket's back rib and the handle's back wall.
- **A blind locating tongue.** The cover's 3 mm tongue hangs 2 mm inside the left half's walls. Within 1.8 mm of
  its place the tongue goes in; 3.5 mm off, it lands on the left half's rim (or a rib on the unit) and the cover
  rests on it, open by 3 mm or more, which the rim-gap rule catches.
- **A level carry and a straight drop.** The cover is a T-shaped shell of 110 g with a battery foot; it must arrive
  level, in line within about 2 mm and within about 0.7 degrees of yaw (what the tongue's play allows) before it is
  lowered. Held at one end it tips: with the right arm idle, the left arm's grip at the nose alone left it on the
  housing 3 mm open. The scripted demo carries it with two hands (one across
  the barrel's nose, one across the foot behind the handle, its centre of mass 9 mm off the line between them). The
  scorer does not care how it gets there.
- **Physical, not counted.** No rule says which part goes first or where: the cover's ribs and tongue decide whether
  it closes. The scorer only reads where everything ended up.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, context, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place, each size marked MODEL or published |
| `scoring.py` | The scorer: plain NumPy functions of the parts', the cover's and the arms' states, with a small tracker |
| `mechkit.py` | A copy of the program kit's compound-body helper (version 1 features only): the unit, switch and cover |
| `rigstate.py` | Reads both arms' TCP, pad tips and opening for the scorer (wraps the embodiment's step and reset; changes nothing it does) |
| `offline_model.py` | Offline checks only (no simulator): moves the three parts by their own shapes (grasp, carry, straight drop onto the fixed boxes), so the offline checker scores the demo without bounding-box artefacts; unused in the simulator |
| `tests/test_scoring.py` | Offline tests: the parts fit seated and do not fit turned round, the cover's ribs land on a raised part, the tongue is the first contact of a misplaced cover, every scoring tier and threshold, the held rule, the at-rest window, the reset and the demo's numbers (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses `yam_arena`'s task-file
objects (fixed boxes) and `mechkit.py` compound bodies, and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms)
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close up the drill housing. Lay the motor and gearbox in the open orange half, spindle through the nose, and set the trigger switch in its pocket with the trigger through its slot. Then close the other half down onto it so the two halves meet all round." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/drill-housing-close-flush/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close up the drill housing. Lay the motor and gearbox in the open orange half, spindle through the nose, and set the trigger switch in its pocket with the trigger through its slot. Then close the other half down onto it so the two halves meet all round." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/drill-housing-close-flush/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`motor_seated`, `switch_seated`, `cover_shut`,
`cover_on_housing`, `closed_over_unseated`, each part's `_held_by`, `_at_rest` and `_path_mm`, the unit's
`motor_dx_mm`, `_dy_mm`, `_dz_mm`, `motor_axis_deg`, the switch's `switch_dxy_mm`, `_dz_mm`, `_tilt_deg`, `_yaw_deg`,
the cover's `rim_gaps_mm` and `corner_steps_mm` at its six corners, the grippers' `opening`, `arms_available` and
`arms_unavailable`). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads every loose part's pose once per control
step (20 Hz) and both grippers' pad-tip points and openings through `rigstate.py`. It keeps only each part's last
0.5 s of positions (for "at rest") and when a new episode starts. There is no whole-trial rule.

- **At rest**: the part's frame moved along a path shorter than 3 mm over the last 0.5 s (10 control steps).
- **Held**: a gripper less than 90 % open has its pad-tip point within 15 mm of the part's surface and no farther
  from it than from any other loose part (a tie holds both). A held part earns nothing and triggers no rule. On a
  step where the arms cannot be read, every part counts as held (it fails closed: that step scores 0 and is never a
  success) and the step is counted in `arms_unavailable`.
- **Unit seated**: not held, at rest, its can/gearbox joint within 4 mm of its place along the axis (the stops allow
  3.5 mm each way), 2 mm across and 1.5 mm in height, its axis within 3 degrees of the barrel's, spindle toward the
  nose.
- **Switch seated**: not held, at rest, body centre within 3 mm of its seat seen from above and 1.5 mm in height (on
  its standoff), its body's vertical axis within 5 degrees of vertical, trigger within 8 degrees of +x (through its slot).
- **Shut**: the cover not held, at rest, and at each of its six outline corners (the barrel's four, the battery
  foot's two far ones) its rim within 1.5 mm of the parting plane and within 4 mm of the left half's matching corner
  seen from above.
- **On the housing**: the cover not held and its frame point (on the barrel's axis) inside the housing's outline.

| Outcome (first match) | Score |
|---|---|
| The cover is on the housing while a part that is not held has its frame inside the outline and is not seated | 0 |
| Unit and switch seated, the halves shut | 1.0 |
| Both seated, the cover on the housing and at rest, but not shut (tongue on the rim, turned, shifted) | 0.7 |
| Both seated, the cover anywhere else | 0.5 |
| Only the unit seated | 0.3 |
| Only the switch seated | 0.2 |
| Nothing seated | 0 |

The trial ends with success once 1.0 has held for 1 s (`SUCCESS_HOLD_S = 1.0`); otherwise `task_score` is read
at the run limit with the same rules.

**Real trial.** A 1.5 mm feeler gauge must not enter the parting line at any of the six corners, and a steel rule
across each corner reads the step. Then the right half is lifted off: a photo shows the spindle in the nose notch and
the trigger in its slot, a depth gauge from a straightedge on the rim reads the unit's top 18 mm and the switch's
11 mm above the parting plane (within 1.5 mm), and a rule reads the unit's end gaps to its stops.

## One arm alone

The unit starts 26 cm left of the midline and the switch 26 cm right, each out of the other arm's reach. Measured on
the GPU (job of 2026-11-19, the same demo with one arm left at its ready pose): the left arm alone scored 0.3 (the
unit seated; its one-handed attempt at the right half left it on the housing 3 mm open), the right arm alone 0.2
(the switch seated). The offline checker predicted the same (0.3 and 0.2).

## Demo and run limit

The left arm lays the unit in its ribs (gripped over its centre of mass, the pad tips 1.5 mm above its axis,
released 1.5 mm above its seat) and moves aside; the right arm picks the switch, waits until the left arm is clear,
and sets it in its pocket. Then the right arm goes to the battery foot's rear end and the left to the barrel's nose
(the right first, so the arms never move past each other), both grip, lift the cover level to 60 mm, carry it
15.5 cm along x, lower it to 12 mm above the rim, then slowly to 0.5 mm, settle, let go fully open and rise 5.5 cm. Every
set-down ends with a slow last stretch and a settle before the release (for the unit: a 1 s settle 2 cm up, then 3 s
for the last 2 cm). The demo plan lasts 49.4 s (988 steps at 20 Hz); the run limit is 2,400 steps (2 minutes).

The demo's place points carry trims measured in the first GPU run (2026-11-19, demo 0.0): carrying the 315 g unit
the left arm sat 2.5 to 3.4 mm short in x and 1.3 to 2.1 mm off in y, and the grip turned the unit 1.5 degrees about
the vertical, so the unit came down on the rear stop and its spindle on the nose wall beside the notch, 6.5 mm high,
and the right half was then closed over it (score 0). The unit's place point is moved 2.7 mm forward and 1.5 mm
toward -y and the wrist turned back 1.5 degrees; the two arms carrying the right half turned it 1.0 degree with their
grips, which leaves its tongue about 0.1 mm of its 2 mm play, so their points over the housing are moved by the
measured offsets (left +1.8, -1.0 mm; right -0.9, +1.2 mm). Only the demo changed; no size, mass or rule.

Offline kinematic check (the program kit's offline checker): worst IK error 0.1 mm, smallest joint margin 15 degrees,
the TCPs never closer than 19.2 cm and the arms' collision capsules never closer than 2.9 cm (at the release
pose). Its own proxy sees each part as a bounding box (the unit's box lands on the nose wall, since its spindle
passes through the notch), so offline the task moves the parts by their own shapes instead (`offline_model.py`:
grasp, carry, straight drop onto the fixed boxes; no sliding or tipping; near a trimmed place point a carried part
is offset by the arm error the GPU trace measured there, minus the trim). With it the checker predicts 1.0 (success
held from step 981), 0.3 with the left arm alone and 0.2 with the right alone. This is kinematics only; the GPU run
decides.

## Sizes and sources

| Item | Size | Source |
|---|---|---|
| Motor can | 36 x 57 mm | MODEL at the common 550 motor size (sellers list 35.8 x 57 mm; to be measured on the drill) |
| Gearbox, spindle, rear boss | 36 x 45 mm, 10 x 24 mm, 12 x 5 mm | MODEL: estimated, to be measured on the drill |
| Unit mass | 315 g | MODEL: to be weighed |
| Housing halves | barrel 131 x 56 mm, handle 46 x 100 mm, battery foot 120 x 30 mm, 28 mm deep each | MODEL |
| Rib, stop, pocket and slot positions | see `layout.py` | MODEL |
| Trigger switch | body 30 x 24 x 22 mm, trigger 15 x 10 x 12 mm, 40 g | MODEL |
| Right half (cover) | 110 g | MODEL (its boxes: 102 cm3, about 107 g of ABS) |

The real trial needs none of these numbers: it uses the drill's own parts, which fit each other by design. The
nest blocks are set against the real left half's outline (1 mm off), and the depth-gauge references are read once at
setup with each part seated by hand; a trial passes when its readings are within 1.5 mm of them. The table is the
simulation's model of the same drill.

## Real-world build

One BLACK+DECKER LDX120C 20 V drill/driver, taken apart once: screws out, the chuck off, the battery set aside, and
the motor's lead unhooked from the switch only (its repair guides disconnect it the same way to lift the switch out).
The lead stays on the motor, its free end taped along the can, so a real trial still has to lay the unit without
trapping the lead under a rib or the rim; a trapped lead holds the parting line open and the feeler gauge enters.
The simulation leaves the lead out (fidelity note). Its own two housing halves, motor-gearbox unit and trigger switch
are reused for every trial; nothing is consumed. The left half sits in a nest of three 10 mm aluminium blocks
screwed to the bench 1 mm off its outline (at the nose, along the barrel's top and at the foot's end). The unit rests
on a two-rail kitting tray (two 6 mm aluminium bars, inner edges 16 mm apart). Reset: lift the right half off, lift
the switch and the unit out and put each back at its start (`reset_notes` in `task.json`).

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each, GPU job of 2026-11-19 02:11 UTC
on the current design (cylindrical unit straddling the halves, stepped cradle ribs, trigger slot, lipped right half
with its own ribs, battery foot, two-handed close, with the arm-error trims described above):

| Run | `task_score` |
|---|---|
| Settle (no plan, 160 steps) | nothing moved |
| Scripted demo | 1.0 (success from step 981 of 2,400; rim gaps 0 mm, largest corner step 1.05 mm; `arms_unavailable` 0) |
| Left arm alone (right idle) | 0.3 |
| Right arm alone (left idle) | 0.2 |
| Dream-1 | 0.0 (it did not grasp any part; nothing seated) |

An earlier run of this design (2026-11-19 00:22 UTC, before the trims) scored 0: the unit landed on its rear stop and
the right half was closed over it, which the zero rule caught. Results of the design before 2026-11-18 do not carry
over.
