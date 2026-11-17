# Change an engine air filter with the airbox cover held up on its duct, then clip shut

On a car with a plastic airbox (here the air cleaner of a 2014 to 2019 Toyota Corolla 1.8 L, full size) the air filter
is changed with two clips and a flexible intake duct in the way. Both clips start closed. The cover stays on its
intake duct, which pulls it back toward its seat, so one hand has to keep it up and out of the way while the other takes
the old filter out and sets the new one flat in its seat. Then the cover goes down with its two rear tabs in their
slots and both clips are closed. The filter is a plain rectangle that fits its pocket with 6 mm of play, either way
round; set down crooked or a quarter turn off, its seal edge rests on the rim and leaves a gap, and dust goes straight
past it into the engine. The scorer reads the end state, with hands off, and gives 0 if the duct is pulled off its
stub or a filter falls off the table. Shutting a clip on an empty box on the way is no zero (the engine is off),
but the end state scores 0 unless the new filter is seated.

## Why it is hard

1. **The first and last steps are part of the job.** The clips start closed: a closed clip's hook lies over the cover's
   catch lug and stops the cover rising. Each is released by pressing its thumb tab down past the middle and closed by
   pushing the released lever back over the middle onto the lug, after the cover is down.
2. **The cover cannot be put aside.** It hangs on a flexible duct (a spring of about 0.1 N per cm). One gripper has to
   hold it up about 40 mm and 240 mm aside, a load of 3 to 4 N with its weight, the whole time the other gripper works
   under it, and the duct has a travel (35 mm back or forward, 35 mm to the right, 280 mm to the left, 70 mm up) past
   which it is pulled off its stub (a zero).
3. **The seat is a close fit.** The pocket leaves the filter 6 mm of play and about 2.5 degrees of yaw. The new filter
   lies a quarter turn from its place, so it has to be turned as it is carried over and set down square and centred,
   or its edge rests on the rim and it is not seated.
4. **The cover has to go back by its tabs.** Lowered in its seat the tabs go into 12 x 26 mm slots (4 mm of play along
   x, 7 mm along y); a tab that misses leaves the cover 12 mm up and the clips cannot close.
5. **Surroundings.** A group 35 battery stands behind the right side of the box and the radiator support's top edge runs along the
   front, so the filter leaves over the right side and the clips are reached from the front.
6. **Hands off.** Both grippers must be 30 mm clear and every part still for 1 s.

The ability tested is not "press into a snap" or "hold one part still": it is holding a part that is pulled back by a
flexible duct while the other hand does close-fit, blind-ish seating under it, in an engine bay's real clearances, with the
job's own first and last steps.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the parts, the clips' detent rule, `success()`, `score()`, the scripted `demo()`, the offline stand-in for the physics |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the states, with a tracker for the latches, stillness and hands off |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies with joints; read both grippers each step (unchanged copies of the kit) |
| `tests/test_scoring.py` | Offline tests of every tier, zero condition, stillness, NaN states and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses `yam_arena`'s task-file
objects plus the compound bodies of `mechkit.py` (next to `task.py`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the instruction in task.json>" \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/air-filter/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the instruction in task.json>" \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/air-filter/dream
```

Run one trial per process. The scorer keeps the trial's record in the task module and recognises a new episode in the
same process when the simulator puts the filters, the cover and the clips back on their marks in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm check);
`DEMO_HOLD_ONLY=1` plays no plan (a settling check); `DEMO_VARIANT=one_clip` (a counterfactual, never the submitted
demo) leaves the left clip open; `TASK_TRACE=trace.jsonl` writes one JSON line per control step with what the scorer saw
(`old_out`, `new_seated`, `cover_*_mm`, `clip_q_deg`, `latch`, `hands_gap_mm`, ...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`, and from the repository root
`python3 tools/offline/check_task.py tasks/<this folder> --require-json --strict --one-arm`.

Runtime per trial: the scripted demo is 76 s of simulated time (1,526 control steps; the policy stops at step 1,587),
about 3 to 5 minutes of wall clock on the L40S including Isaac Sim's start-up (the first start on a new instance takes
several minutes more); a run to the 3,600-step limit is about 6 to 9 minutes. Seed: nothing is random;
`-E object_jitter=0` fixes the start layout, so every run starts from the same poses and no seed is needed.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the poses of the two filters, the cover's
three duct joints and the two clips' angles, once per control step (20 Hz), and the grippers for the let-go check.

| Condition | Rule |
|---|---|
| Old out | the old filter's whole outline (top view) clear of the case's footprint (x 210 to 510, y -110 to +130 mm), its lowest point within 6 mm of the table (a mat is 2 mm), tilted under 10 degrees |
| New seated | all four bottom corners within 2 mm of the seat face (45 mm up), long side along the pocket, either way round (yaw within 5 degrees of 0 or 180), centre within 6 mm of the pocket's centre, tilt under 3 degrees |
| Cover seated | rim within 2 mm of the case rim (dz), dx within 4 mm and dy within 5 mm: both tabs at least 5 mm into their slots |
| Clipped | each clip's lever within 5 degrees of closed and not past it (lower than -3 degrees means no lug under the hook), the cover seated |
| Hands off | last 1 s: both finger zones 30 mm or more from the airbox, cover and both filters; no filter moved 5 mm or turned 5 degrees, the cover moved under 5 mm, no clip turned 5 degrees |
| DUCT (latched) | the cover more than 35 mm back or forward, more than 35 mm to the right or 280 mm to the left of its seat, or more than 70 mm up |
| FELL (latched) | a filter's lowest point more than 20 mm below the table top (off the table); lifting or carrying one is fine |

| Outcome | Score |
|---|---|
| A latch fired, or not hands off | 0 |
| Old out, new seated, cover seated, both clips closed | 1.0 |
| The same with a clip open | 0.5 |
| Old out and new seated, cover not seated | 0.25 |
| Anything else (doing nothing, the old filter still in, the new one not seated) | 0 |

Success: 1.0 held for 1 s (on top of the scorer's own 1 s of hands off and stillness). Doing nothing scores 0. A trial
without success is scored as it stands at the run limit.

Bench method for a real trial (the simulator reads poses; a person reads these):

| Threshold | Bench method |
|---|---|
| Old out (footprint, 6 mm, 10 degrees) | by eye against a tape outline of the case; a 6 mm feeler under it; a 10 degree protractor |
| New seated (2 mm, 5 degrees, 6 mm, tilt) | after the trial, with the cover lifted: a 2 mm feeler at the four corners of the frame; a 5 degree protractor on the long edge; a rule from the pocket wall for the 6 mm |
| Cover seated (2 mm, 4 mm, 5 mm, tab depth) | a 2 mm feeler round the rim; a torch at the rear slots (the tabs visibly in, at least 5 mm); a tape mark across cover and case for the 4 and 5 mm |
| Clipped (5 degrees, on the lug) | a protractor on each lever; a light tug on the cover (it does not lift) |
| Hands off (30 mm, 5 mm, 5 degrees) | a 30 mm spacer block against each hand; 30 fps video of the last second against a 5 mm grid card |
| DUCT (35 mm, 35 mm, 280 mm, 70 mm) | tape flags on the cover's rear edge and a ruler fixed at each limit; the video; the stub looked at afterwards |
| FELL (20 mm) | by eye: the filter fell off the bench edge |

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone) scores 0: its gripper on the cover's lug cannot lift it, because both clips
are still closed. `DEMO_IDLE_ARM=left` (the right arm alone) releases both clips, then cannot reach the old filter under
the cover, which stays down: 0. `DEMO_HOLD_ONLY=1` scores 0. `DEMO_VARIANT=one_clip` (the left clip never shut) scores 0.5.
(Offline kinematic predictions from this task's stand-in for the physics, `task.OfflineModel`; the GPU runs are
recorded in `task.json`'s `scores` once done.)

## Demo and run limit

Open loop: timed waypoints from the known start layout, nothing read from the simulator. One arm is near the front of
the airbox at a time (two forearms there overlap by 8 cm in the checker's capsule model); the left arm waits aside.

| Time (s) | Left gripper | Right gripper |
|---|---|---|
| 0 to 8.3 | waits aside | closed fingers on each clip's thumb tab in turn, press it down 30 mm past the middle: both clips release |
| 10.5 to 21.6 | over the lift lug, grip it across its 20 mm (12 mm commanded), lift the cover 40 mm (commanded), slide it 240 mm toward the duct | aside, waiting |
| 21.6 to 36.4 | holds the cover aside | pinch the old filter's right frame wall (20 mm, 12 mm commanded), lift it 45 mm, carry it over the new filter's mat to the old mat, set it down, let go |
| 36.4 to 53.0 | holds | pinch the new filter's front frame wall, lift it, carry it over the pocket turning it a quarter turn, lower it onto the seat, let go |
| 53.0 to 62.7 | slides the cover back, lowers it onto its tabs, lets go, goes aside | backs away |
| 62.2 to 76.3 | aside | closed fingers push each released lever's upper end back over the middle (onto the lug), back off |

The demo takes 76.3 s (1,526 steps); the scripted policy stops at step 1,587. Offline: worst IK 0.08 mm; smallest joint
margin 7.4 degrees (right joint 4) and 8.4 (left joint 4); the arms' capsules stay at least 10 cm apart; TCPs at least
28 cm apart. Every gripper open or close takes at least 1.5 s; each final descent is a separate slow move; every grip is
preceded by a short settle. The cover is lifted 40 mm commanded: the arm sags 4 to 8 mm under its weight and the duct's
pull, so the skirt ends 28 to 36 mm up, clearing the filter's frame (19 mm over the rim) and the released clips'
levers. `TASK_TRACE` reports `cover_dz_mm`: keep it between 25 and 60 mm while the cover is aside. The run limit is 3,600
steps (180 s at 20 Hz), 2.3 times the demo: Dream-1 and the real trial need more time than a script.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both clips closed over the cover's lugs with the
cover down; the old filter in the seat; the new filter on its mat at x 400, y -300 mm, long side along y; the old-filter mat clear; both grippers open in their ready poses.
Nothing is consumed.

## Real-world build and calibration

Parts: one complete air cleaner assembly (case, cover, both clips) of a 2014 to 2019 Toyota Corolla 1.8 L, the one the
FRAM CA10171 is listed for, its part number read off the case label and logged, on a bench stand, the matching FRAM CA10171 panel filters (two), a 300 mm length of flexible rubber hose sized to the cover's
outlet, clamped to it and looped back to a fixed stub at the throttle body's place, a group 35 battery case (or a box
230 x 175 x 225 mm), a 30 mm high bar for the radiator support's top edge, two bench mats, a rule, a 2 mm feeler gauge,
a 30 mm spacer block, a luggage scale.

The real trial uses the real parts; the simulation's numbers are nominal estimates (marked `est` in `task.json`).
Acceptance checks before the first trial, each with a pass range (an assembly that passes them all is a valid build):

| Check | Tool | Pass |
|---|---|---|
| The filter seats flat either way round | the filter itself, a 2 mm feeler | flat (feeler does not pass) both ways round; a quarter turn off it rests on the rim |
| The duct's give | luggage scale on the lift lug, the cover 40 mm up and 240 mm toward the stub | 1 to 4 N, the hose still on its stub |
| The clips | luggage scale at the thumb tab and at the lever end | each releases and closes for under 15 N |
| Gripper access | the robot's gripper, before the first trial | the seated filter's frame end takes a 20 mm pinch 13 mm deep without touching the case; the lift lug takes a 20 mm pinch |
| Weights | kitchen scale | cover about 0.28 kg, filter about 0.15 kg |

Measured sizes, forces and weights replace the `est` figures in `layout.py` (the case and cover sizes, the filter's
size, frame width and pleat height, the slots' play, the clip force, the duct's pull). The clips are
simulated as rocker levers (the real ones are spring clips: the same press, flip and shut, a lever past an over-centre
point).

## Simplifications

- Rigid bodies, built from boxes and cylinders. The pleat pack is a rigid block. The case is one fixed body on its
  stand; the rubber grommets, the hose clamps, the mass airflow sensor's plug and the wiring on the real cover are not
  drawn.
- The duct is not drawn: it is three sprung sliders (x, y, z) under the cover, 0.1 N per cm each, so the cover only
  slides, it does not tilt. A real cover tilts a little about its rear tabs as its front lug is lifted; the job is the
  same (lift the tabs clear, slide aside, hold, bring back onto the tabs), and the scoring reads only where it ends up.
- The clips are hinged levers with one declared rule: an over-centre detent (`_clip_detent`), a spring toward the
  nearer of two notches, closed and released, 2 N at the thumb tab at the snap point (est).
- The filter's frame stands 19 mm over the case rim where the gripper pinches it, and the pleat pack is 23 mm lower
  than the frame's top, so the gripper's two pads can reach it from above (the gripper-access acceptance check).
- Frictions 0.6 (filter) and 0.5 (case, cover): est.

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

## Sources

- Automotive service technicians and mechanics, SOC 49-3023, about 0.7 million US jobs: BLS Occupational Outlook
  Handbook, https://www.bls.gov/ooh/installation-maintenance-and-repair/automotive-service-technicians-and-mechanics.htm
- The Corolla air cleaner (Toyota 17700 series) and the FRAM CA10171 filter are named from retailer parts listings; every
  size above is an estimate until measured on a real unit (README, Real-world build and calibration).

Third-party assets: none. Every part is a box or cylinder built in `task.py` from the numbers in `layout.py`; no
meshes, scans or scenes are used.
