# Shut spring kitchen tongs and draw out the pull tab that locks them

## What the robot does

A pair of spring kitchen tongs with a hinge pull-tab lock (sized after the OXO Good Grips 9-inch tongs) lies open on its side on
the worktop, tips toward the robot. One gripper squeezes the two steel arms shut and holds them; the other pinches
the black tab at the hinge end and draws it out 12 mm to its stop; then both grippers let go and the lock holds the
tongs shut on the worktop. The start screenshot (top camera, first frame) is from the first GPU run.

## Why it is hard

1. **The lock is gated by a force held somewhere else.** The tab's lock block passes fully under the free arm's heel
   lugs only while the arms are within 2.86 degrees of shut, and the tab reaches the 11 mm that scores only within
   3.0 degrees (tips 9.2 mm apart or less). Shut, the spring gives 0.40 N m: 2.3 N at the tips, 4.4 N at 90 mm from
   the pin, where the demo holds them.
2. **Two contacts far apart.** From the ring's centre to the hold 90 mm from the pin is 131.5 mm with the tab in
   and 143.5 mm with it out; to the tips 217.5 mm. A gripper opens 94 mm and does one thing at a time: pinch the
   arms or pull the ring.
3. **Letting go undoes it at once.** Released unlocked, the free arm (54 g) reaches its 27.5 degree stop in about
   0.1 s. An arm needs over a second to travel the 13 cm to the tab, so "squeeze, let go, pull" cannot work.
4. **The tool is loose and light.** Weight 1.32 N; worktop friction at 0.6 is 0.79 N, less than the tab's 1 N of
   friction. A tab pulled without a hold tows the tongs from the start; the tab moves 3.8 mm at most, where the
   open arms block it. The hold has two jobs: shut the arms and anchor the tool.
5. **A hold that is released early loses the lock.** While held, locked and unlocked tongs differ only by 12 mm of
   tab. Unlocked tongs are back at 84 mm a tenth of a second after the hold opens, and the score is then the
   squeeze's 0.25 and nothing more.
6. **The pull ends on a hard stop with both grippers on one rigid tool.** After 12 mm the tab stops dead; a pull
   that keeps pushing hard drags the 135 g tool through the holding pads or the ring through the pulling pads.
   The ring is 8 mm thick and stands on edge 3 mm above the worktop, 5 mm behind the hinge end of the arms.

Knowing all of this in advance removes none of it: the instruction states the method in full.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the tongs, the spring preload, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, and the lock's geometry as functions (tab travel against hinge angle, where let-go arms come to rest) |
| `scoring.py` | The scorer (`Tracker`: squeeze milestone, lock, let-go and stillness windows, tiers) as plain NumPy functions |
| `offline_kinematics.py` | How the tongs move without a simulator (offline checker and tests only; never used in the simulator) |
| `mechkit.py` | Our mechanism kit, version 2, a byte-for-byte copy of `tools/kit/mechkit.py` |
| `rigstate.py` | A copy of `tools/kit/rigstate.py`: reads both grippers each control step; its fail-closed `released` is the scorer's let-go check |
| `tests/test_scoring.py` | Offline tests: the lock geometry, every tier, near misses, exploits, fail-closed arms, resets, the demo and its clearances, the text |

## Objects

One object, nothing fixed, nothing else on the worktop.

| Item | Value | Source |
|---|---|---|
| Size reference | OXO Good Grips 9-inch locking tongs, no. 28481, stainless steel, black non-slip grips (size and mass only; this task does not claim that product's lock direction) | [OXO product page](https://www.oxo.com/9-tongs-1276.html) |
| Length, tab in | 230 mm (9 in) | catalogue (America's Test Kitchen review) |
| Mass | 135 g (4 3/4 oz) | catalogue (America's Test Kitchen review) |
| Pull tab at the hinge end, doubles as a hanging hole | yes | OXO product page ("large pull-tab for locking") |
| Lock: a sliding plate between the arms, drawn rearward to hold them closed, pushed forward to free them (pull-to-lock, as in this task) | yes | patent [US 7,448,660](https://patents.google.com/patent/US7448660B2/en) (Helen of Troy, OXO's owner) |
| The hinge pull tab is the most popular tongs lock; the cook closes the tongs with one hand and pulls the tab with the other | yes | patent [US 11,974,682](https://patents.google.com/patent/US11974682B2/en), background |
| Arm section and length | 1 mm stainless channel 20 x 6 mm, 200 mm long, head 30 mm wide | our estimate |
| Masses of the parts | pin arm with spring 61 g, other arm 54 g, tab with plate about 20 g | our estimate (sum is the catalogue 135 g) |
| Pin position | 24 mm from the arms' back ends, 176 mm from the tips | our estimate |
| Spring | 27.5 degrees open (84 mm between the tips' inner faces), 0.20 N m open to 0.40 N m shut | our estimate |
| Tab | ring 24 x 25 x 8 mm with a 12 x 13 mm hole; 12 mm of travel against about 1 N; tab gap 5 mm in, 17 mm out | our estimate |
| Lock rest | arms spring back 3 degrees onto the lock (tips 9 mm apart) | our estimate |

Masses recompute: an arm's sheet is 200 x 32 x 1 mm of stainless steel, 6.4 cm3 x 7.9 g/cm3 = 50.6 g, plus a 3.4 g
grip (and 7 g of ears, pin and spring on the pin arm); the tab is a steel block 18 x 10 x 10.6 mm (15.0 g) and a
nylon ring and stem (3.9 cm3 x 1.15 = 4.5 g), 19.5 g. 61 + 54 + 20 = 135 g.

## How the lock is built

| | Product | Simulation |
|---|---|---|
| keeper | a sliding plate between the arms, moved by the pull tab (patent US 7,448,660) | one steel block on the tab, 18 x 10 x 10.6 mm |
| what it bears on | the heel ends of the arms behind the pin | two lugs on the free arm's heel, 20 mm behind the pin |
| locked state | plate drawn back: arms held closed | block drawn 12 mm back under the lugs: arms held 2.9 degrees from shut |
| free state | plate pushed forward | block forward of the lugs |
| while the arms are open | not claimed | the lugs are in the block's path: the tab stops 3.8 to 6.3 mm out (arms 4 degrees or more open) |
| direction | pull to lock, push to unlock (patent US 7,448,660; some OXO models lock the other way) | pull to lock, push to unlock |

Three bodies on one loose articulation (`mechkit`), all sizes in the tongs' frame (u along the pin arm toward the
tips, w across the arms, v along the pin, which is upright because the tongs lie on their side):

- **Pin arm** (the root, 61 g): bar u -24 to 146, 20 x 6 mm; head u 146 to 176, 30 x 30 x 14 mm; two hinge ears.
- **Free arm** (54 g) on a **hinge** about the pin: 0 (open, as drawn) to 0.480 rad (shut, heads meeting). A joint
  spring of 0.417 N m/rad pulls toward a target 0.480 rad beyond the open stop (`mk.set_targets`, set after every
  reset and at every step): 0.20 N m at the open stop, 0.40 N m shut. Damping 0.03 N m s/rad.
- **Tab** (20 g) on a **slider** along the pin arm, 0 to 12 mm, hard stops, 1 N of joint friction, damping
  2 N s/m: the lock block (u -8 to 2 tab in), a stem and the ring (u -54 to -29) as four bars round its hole.
- **The lock is contact only.** Tab out, the block lies at u -20 to -10 with its top 1 mm under the lugs' undersides
  (6 mm of overlap along u). Let go, the free arm opens until the lugs sit on the block's back top edge: 2.86 degrees,
  tips 8.8 mm apart, 0.379 N m / 0.020 m = 19 N on the block across the slider's axis. The tab cannot be pushed back
  by it (the lug face slopes 2.86 degrees: 0.95 N along the tab against 1 N of joint friction and 0.6 x 19 N of
  contact friction). No weld, bond, drive rule or `on_step` logic holds the arms shut.
- **Self-collision.** The block (on the tab) and the lugs (on the free arm) are sibling links of one articulation;
  mechkit writes articulations with self-collision off, so `task.py` turns it on for this assembly only, through the
  plan that mechkit's own author step reads (`_usd_plan_self_collisions`; `mechkit.py` itself is unchanged). PhysX
  still filters each parent and child pair. With the tab in, the rest of the tab stays 1.3 mm or more from the free
  arm at every hinge angle (a test sweeps it); over the parked block the open heel clears it by 1.62 mm vertically
  and 1.43 mm square to the arm.
- **Tongs settings**: one friction 0.6 for every surface (between steel 0.4 and the grips 0.8); tight-fit contact
  settings (0.5 mm contact offset, 32 solver iterations, angular damping 1).

Side view of the lock at the hinge end (tab drawn out; lugs on the free arm above, block on the tab below):

```
          free arm bar (w 22..28)
   ======================================  <- pivot pin at u 0, w 19
   | lug |   lug underside w 17.6
   +-----+   1 mm clear when shut
   [ block u -20..-10, top w 16.6 ]
   ---------- pin arm bar (w 0..6) ----------
```

How far the tab can be drawn against how far the arms are open (layout.travel_limit, tested):

| Arms open by | 27.5 deg | 20 | 10 | 5 | 4 | 3.5 | 3.0 | 2.86 |
|---|---|---|---|---|---|---|---|---|
| Tab travel possible | 3.8 mm | 4.7 | 5.5 | 5.8 | 6.3 | 8.3 | 11.0 | 12.0 |

The tab is loose whenever the arms are squeezed fully shut: the lugs are then 1 mm off the block and only the
slider's 1 N holds the 20 g tab (the same is expected of the product). In the demo that is the time between the left
gripper letting go of the ring and the right one letting go of the arms. On the GPU the loose tab still crept
outward at about 0.15 mm/s while the arms were held shut (0.72 to 1.46 mm in the demo, 2.8 mm in the left-idle run),
so the slider's friction does not hold it fully still; the demo pulls it to its stop before the arms are let go. A tab left 5.9
to 11 mm out leaves 0 to 5 mm of block under the lugs: the simulated arms come to rest on the block's edge at 4.1 to
3.0 degrees (tips 12.6 to 9.2 mm). That is not scored as locked (the tab is not within 1 mm of its stop), so no
score depends on what a part-drawn tab does. A tab under 5.9 mm meets the lugs with its back face instead, is pushed back in to 3.8 mm and the arms spring fully open.

The simulated centre of mass (from the colliders, locked) is about 75 mm from the pin toward the tips; a sheet-metal
pair's is about 56 to 66 mm, because the heads here are solid blocks. Nothing carries the tongs, so this only
changes how they settle.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`, its thresholds `layout.py`.

| Condition | Rule (scorer) | Real-bench check |
|---|---|---|
| Tip gap | 2 x 176 mm x sin(angle from shut / 2): 84 mm open, 9 mm on the lock | ruler across the tips' inner faces |
| Tab gap | 5 mm + tab travel: 5 mm in, 17 mm at the stop | ruler from the arms' hinge end to the ring |
| Squeezed shut | tip gap at most 14 mm on 20 consecutive steps (1 s) at some point; latched until the next trial | video |
| Lock engaged, at the end | tab gap at least 16 mm (within 1 mm of the stop) and tip gap at most 14 mm | ruler, two readings |
| Let go | on every step of the last 1 s, both grippers' finger zones at least 30 mm from every shape of the tongs; unreadable arms or a distance that cannot be computed count as not let go | video, last second |
| Still | over the last 1 s the root body's path under 2 mm and its turn under 2 degrees; tip gap and tab gap each the same to 1 mm | video; two photos 1 s apart with a ruler in view |
| On the worktop | lowest point within 1 mm of the worktop | side photo |
| Not fallen | no part more than 10 mm below the worktop level | video or photo |

| Outcome | Score |
|---|---|
| Lock engaged, let go, still and on the worktop (the trial ends once this has held 1 s) | 1.0 |
| Lock engaged and not fallen, but a gripper still within 3 cm, or the tongs still moving, held up or lying on something | 0.5 |
| Not engaged at the end, not fallen, squeezed shut at some point | 0.25 |
| Anything else: doing nothing, a tab pulled with the arms open, tongs on the floor (locked or not), any step whose tongs state is missing or not finite | 0 |

Why 0.5 can be earned with a gripper still on the tongs: the reading (tab within 1 mm of its stop) cannot be
produced by holding, because the tab passes 11 mm only when the block is under the lugs. It is the "latched, hands
still on" stage of the mechanism; 1.0 needs the lock left to hold by itself. Nothing zeroes the whole trial: no step
here can hurt a person or the tool. The tab cannot be out 11 mm unless the arms are within 3 degrees of shut, so the
second reading (tip gap) adds nothing in simulation; it is kept because it is a second ruler reading of a real trial.

Exploits and near misses the tests check (`Tiers`, `FailClosed`, `Resets`): nothing (0); tab pulled with the arms
open and the tongs towed (0); arms pinched shut for 0.9 or 0.95 s, then released (0); for 1 s or more (0.25);
held shut to the end with the tab in (0.25); pressed shut by a closed gripper (0.25); tab 6 to 10.9 mm out with the
arms on the block's edge (0.25); locked and still pinched (0.5); locked, fingertip zone 29 mm away (0.5) and 31 mm
(1.0); let go 19 steps before the end (0.5); sliding 3 mm/s or turning 3 degrees/s (0.5); held 1.5 mm, 5 mm or 5 cm
up (0.5), 0.9 mm (1.0); tilted onto something (0.5); partly over the worktop's edge, resting on it (1.0); 10.1 mm
below the worktop or on the floor (0); tab gap 15.9 mm (0.25), 16.0 mm (1.0); tip gap 14.1 mm with the tab at its
stop (0.25), 14.0 mm (1.0); tip gap drifting 1.1 mm in the last second (0.5), 0.9 mm (1.0); tab swinging 1.1 mm
(0.5); locked, then the tab pushed back in (0.25); unreadable arms in seven forms (0.5 with the tongs locked); a
distance that raises (0.5); a missing or non-finite state (0, and the windows start again); a state without joint
readings (0). A new trial is recognised from rigstate's and mechkit's reset hooks, and as a fallback when the tongs
are put back on their start pose, open and tab in, with a jump no trial makes in one step (a natural release at the
start pose is not one: tested).

## Counterfactuals

Run each with the demo command below and the environment variable shown. Predicted offline (kinematic proxy, no
physics); measured on the first GPU run (job 20261112T230210, one trial each) where it was run there, otherwise
offline only. The demonstrated method uses two arms; no field claims
that two are necessary (a one-gripper route, for example hooking the ring behind the worktop's edge, is conceivable
on a real bench).

| Run | Switch | What it does | Predicted | Measured |
|---|---|---|---|---|
| tab only | `DEMO_VARIANT=tab_only` | the left gripper pinches the ring where it lies and makes the full pull (27 mm commanded); no hold | 0 (the tongs are towed; the tab stays in) | offline only |
| squeeze, let go, then pull | `DEMO_VARIANT=squeeze_release_pull` | the right squeezes for 1 s and lets go, the left then pulls | 0.25 | offline only |
| right arm only | `DEMO_IDLE_ARM=left` | the full demo with the left arm idle: squeeze, hold, let go | 0.25 | 0.25 (squeezed and held shut; on release the arms opened to 18 degrees on the tab, which had crept 3 mm out) |
| left arm only | `DEMO_IDLE_ARM=right` | the full demo with the right arm idle: the ring pulled with nothing holding the arms | 0 | 0 (tab stopped at 3.1 mm, tongs towed 2 cm along +x, arms never left 27.5 degrees) |
| tab drawn part way | `DEMO_VARIANT=tab_8mm` | the demo with the pull commanded to 8 mm only (the trace's `tab_mm` gives where the tab stopped) | 0.25 (the arms rest on the block's edge in simulation) | offline only |
| lock, never let go | `DEMO_VARIANT=no_let_go` | the demo with the right gripper never opening | 0.5 | offline only |
| do nothing | `DEMO_HOLD_ONLY=1` | both arms hold the ready pose (also the settle check) | 0 | 0 (the settle run, 160 steps: nothing moved) |

## Demo

Open loop: timed waypoints from the known start layout; it reads nothing from the simulator. Both grippers point
straight down with their fingers closing along y, across the tongs (which lie along x, tips toward the robot).

| Time (s) | Right gripper (the hold) | Left gripper (the tab) |
|---|---|---|
| 0 to 2.5 | open to 94 mm, over the arms 90 mm from the pin (x 0.100, y 0), pad tips 60 mm up | open to 40 mm, to a waiting point clear of the right arm (x 0.26, y 0.14, pad tips 120 mm up) |
| 2.5 to 4.5 | down to pad tips 6 mm up over 1.5 s (the open arms span 66 to 80 mm under the pads: 7 mm or more a side), settle | wait |
| 4.5 to 6.5 | close to a commanded 24 mm over 2 s: the arms meet at 28 mm | over the ring's held position (x 0.240, y -0.006), pad tips 60 mm up |
| 6.5 to 9.5 | hold (the arms settle shut along x) | wait |
| 9.5 to 11.5 | hold | down to pad tips 4 mm up over 1.5 s, one pad either side of the 8 mm ring (16 mm a side), settle |
| 11.5 to 13.5 | hold | close on the ring over 1.5 s (commanded 5 mm), settle |
| 13.5 to 17.5 | hold | draw the tab out along +x over 3 s (27 mm commanded: 12 mm of travel and 15 mm for the arms' give, see below), dwell 1 s |
| 17.5 to 20.5 | hold | open to 40 mm over 1.5 s, straight up to pad tips 120 mm |
| 20.5 to 23.5 | open to 60 mm over 1.5 s (the arms spring onto the lock), straight up to pad tips 80 mm | home |
| 23.5 to 26.5 | home | wait at home |

The order keeps the arms apart: two straight-down YAM grippers 13 to 14 cm apart along the tongs stay clear of each
other only with the tongs pointing at the robot (the hold gripper near its own base line, the tab gripper 14 cm
further out), the hold gripper going down first and rising last. Offline the arms' collision capsules stay 27 mm or
more apart (closest while both are down), and every link but the fingers stays 28 mm or more from the tongs; with the
tongs lying across the robot (along y) the wrists' capsules overlap by 62 mm, which is why the layout is this way
round. Smallest joint margins 12.3 degrees (left joint 4) and 20.9 degrees (right joint 4).

Why the pull is commanded 27 mm for a 12 mm stroke: the YAM wrists are soft (joints 4 to 6 at 10 N m/rad, 1 to 3 at
80). At these poses the pad tips give about 6.4 mm per newton along x on the left arm and 5.0 mm on the right one
(from the arm's Jacobian; a test recomputes it), so the tab's 1 N of friction alone takes up about 11 mm: the left
pads lag and the right pads, with the tongs in them, follow the pull. A pull commanded to 12 or 13.5 mm would leave
the tab short of the 11 mm that locks. Past the stop the remaining travel only presses the tab on its stop with 1 to
3 N, against about 9 N that the ring pinch holds and 12 N that the hold on the arms holds. When the left gripper
opens, the right one draws the tongs back by the same few millimetres, and the tab (at its stop, under nothing but
its own friction) goes with them.

Offline (`check_task.py --require-json --strict --one-arm`): predicted 1.0, success held from step 526 (the plans
end at step 530; the scripted policy stops at step 591); the right arm alone 0.25, the left arm alone 0. Run limit
900 steps (45 s): 1.7 times the demo.

Trace checks on the first GPU demo (`TASK_TRACE`, all logged state, none used by the demo): `hinge_from_shut_deg`
read 0.0 from the end of the squeeze (step 127) until the right gripper opened, through the pull and the left
gripper's release; the tongs' root moved 5.1 mm along +x with the pull (x 194.8 to 199.9 mm) and came back to 198.2 mm
when the left gripper opened; `tab_mm` reached 12.0 (its stop) by step 301 (15.0 s) and stayed there to the end; after
the right gripper opened the hinge settled at 2.91 degrees (tip gap 8.94 mm) and stayed there. Details under Results.

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock the kitchen tongs shut. Squeeze the two steel arms together, pull the black tab at the hinge end straight out until it stops, then take both grippers away. The lock must then hold the tongs shut on the worktop." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/lock-kitchen-tongs-pull-tab/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock the kitchen tongs shut. Squeeze the two steel arms together, pull the black tab at the hinge end straight out until it stops, then take both grippers away. The lock must then hold the tongs shut on the worktop." \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/lock-kitchen-tongs-pull-tab/dream
```

`TASK_TRACE=trace.jsonl` writes one JSON line per control step: `score`, `hinge_from_shut_deg`, `tab_mm`,
`tip_gap_mm`, `tab_gap_mm`, `shut`, `squeezed`, `squeezed_at`, `engaged`, `released`, `let_go_1s`, `still_1s`,
`on_top`, `not_fallen`, `low_mm`, `origin`, `heading_deg`, `motion` (path, turn and gap spreads over the last 1 s),
`zone_mm` (the nearer finger zone to the tongs), `arms_readable`, the TCPs and openings. `MECHKIT_TRACE=<file>` adds
mechkit's own per-step record. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`, and from the repository root
`python3 tools/offline/check_task.py tasks/lock-kitchen-tongs-pull-tab --require-json --strict --one-arm`.

## Dream-1

One trial at the full run limit (900 steps): 0.0. Dream-1 lowered both grippers onto the open arms 3 to 5 cm in
front of the pin and closed them without squeezing the arms together (the hinge stayed at 27.5 degrees and the tab
moved 0.25 mm at most); around 21 s it turned the tongs about 35 degrees and slid them about 5 cm, then lifted, came
back down to the same place and was still there at the limit.

## Real build and reset

Real bench: one pair of 9-inch spring tongs (230 mm, about 135 g) whose hinge pull tab locks when pulled out, unmodified, on a laminate worktop. Nothing
is made and nothing is used up. Reset in three moves: squeeze the arms shut and push the black tab in until it stops;
let the arms spring open; lay the tongs on their side (hinge pin upright) with the pin at x 19.0, y +0.5 cm and the
tips toward the robot, the arm that the tab's stem lies along at y -4.2 cm (the ring's centre is then at x 23.3,
y +0.4 cm). Nothing is powered, sharp or hot.

## Simplifications

From `fidelity`, plus:

- The hinge spring is a joint spring with a preload target; the tab is a slider with joint friction; the lock plate is
  a block under two heel lugs that holds by contact (no code rule).
- Each channel arm is a bar of its outline, 20 x 6 mm; each scalloped head is a 30 x 30 x 14 mm block (nothing here
  uses the heads), so the simulated balance point is about 15 mm nearer the tips than a sheet-metal pair's. The grips
  are not separate shapes. Arm flex is not simulated.
- One friction, 0.6, for every surface of the tongs.
- The catalogue gives the length and the mass; every other size, mass, force and the lock's travel are our
  estimates of this pair.

Third-party assets: none.

## Offline tests

`tests/test_scoring.py` (60 tests): the lock's geometry (tab travel against hinge angle, the 2.86 degree lock and its
8.8 mm tip gap, the block-edge rests, the heel clearance, the tab clear of the free arm at every angle, masses and
spring figures); the kit (byte-identical copies, self-collision on for this assembly only, joint limits within
2 pi, the preload set every step and after reset); every tier, every exploit and near miss above, fail-closed arms
and distances, broken states and resets; the offline kinematics playing the demo (1.0, the tab at its stop, the arms
on the lock) and every counterfactual (0, 0.25, 0, 0.25, 0.25, 0.5, and no plan); the demo's structure (opens and
closes of 1.5 s or more, never while moving, an end hold at home, within the run limit), the pull's slack against
both arms' give at the pull poses, its ordering, the arms'
clearances from each other and from the tongs along the whole demo; the text (field lengths, no dashes or barred
words, numbers in the fields matching `layout.py`).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
