# Open a child-proofed drawer: pull to the catch, press the latch down, slide it open

**Instruction:** Open the child-proofed drawer at least 100 mm and let go.

A stock low chest stands on the table, screwed down, its drawer shut: a MUJI stacking chest, half, 1 drawer, oak
veneer (JAN 4550512381105, 370 x 280 x 185 mm). The drawer is child-proofed with a Safety 1st cabinet and drawer latch
(model 48390), fitted as the maker's guide says: the latch screwed to the back of the drawer front at the top, its
catch screwed under the chest's top 1 in from the front edge. Pulled by its handle (an IKEA BAGGANAS 143 mm bar,
screwed on), the drawer opens until the back of its front is 38 mm out of the carcass (the guide's "about 1.5 in")
and stops with the latch's hook on the catch. To open it, a finger goes into that gap and presses the latch arm down
until the hook passes under the catch; then the drawer slides open. The task ends with the drawer 100 mm or more
open, the latch and catch undamaged and both hands off.

Every parent of a toddler, and every visitor to their home, does this many times a day; a low chest like this one is
exactly the drawer a crawling child reaches. The latch is the defining resistance and it is physical in the
simulation: a sprung arm on the drawer whose hook's face meets the catch's lip. There is no rule that releases the
drawer.

## Why it is hard

- The latch is out of sight inside the drawer and is reached only through the 35 to 38 mm gap that the first pull
  opens, between the back of the drawer front and the carcass's front edge: closed pads, about 12 mm thick at the
  tip, go down into it 5 mm clear of the carcass's edge.
- Pulling harder never works. The hook's face is square to the lip, and the hook pivots about a line above the
  contact, so a pull turns the hook further up into the catch (the self-locking that makes the product work).
- The press must be made with the hook's face off the lip (the drawer eased back 3 mm) and held while the drawer
  moves out, at least until the hook's face has passed under the lip; let go before that and the hook springs back
  up behind the lip and the drawer is caught again.
- The press is light (about 2 N) and small (6 mm at the hook); the arm springs back as soon as it is let go. Near the
  root, where the gap gives access, the lever is short, so the pads must go well past the point of contact.
- The last step is natural: the drawer left open, hands off.

## Scoring

Read at the last step from the chest's joints (drawer travel, latch angle), the latch hook's pose in the chest's
frame, the fixed catch and both grippers (`scoring.py`, a `Tracker` fed every control step).

| Condition | Definition |
|---|---|
| Opening | how far the back of the drawer front stands out from the carcass's front edge: the slider's travel less the inset front's 15 mm |
| Forced (zero, permanent) | at any step the hook's box more than 2 mm into the catch's lip along both x and z; or the hook passing from behind the lip to in front of it within one step while standing more than 2 mm up into the lip's height (tunnelling). In the simulation this is the solver letting the hook through; in a real trial it is a cracked or bent latch or catch |
| Released | on every step of the last 1 s, both grippers' finger zones 30 mm or more from the chest (carcass, drawer, handle, latch) and the catch |
| Still | over the last 1 s the drawer moved under 5 mm |

| Score | When |
|---|---|
| 0 | forced at any step, or the drawer open less than 60 mm at the end (the hook not yet 10 mm past the catch) |
| 1.0 | open 100 mm or more (a hand goes in), released and still |
| 0.5 | open 60 to 100 mm, released and still |
| 0.25 | open 60 mm or more, but a gripper within 30 mm or the drawer still moving |

Success (1.0) must hold 1 s (`SUCCESS_HOLD_S`), on top of the scorer's own 1 s windows. A missing or non-finite state
scores 0 for that step and restarts the 1 s windows. A jump of 50 mm or more back to shut (zero travel) with the latch at
rest is read as an environment reset and clears the forced flag (a trial never moves the drawer 50 mm in 50 ms).

Real-trial checks (simple tools):

| Check | How |
|---|---|
| Opening (60 and 100 mm) | steel rule on the drawer's side, zero at the carcass's front edge |
| Latch and catch undamaged | look for cracks, white stress marks or bending; both screws of each still tight. Any found: 0 |
| Chest not moved | it is screwed to the table; a shift means the screws were pulled out: 0 (real trial only; in the simulation the chest is fixed and this is not read) |
| Hands off, still | side video at 30 fps over the last second with a 5 mm scale on the drawer side; a 30 mm block against the nearest finger in the final frame if in doubt |

## Results

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control, run limit 1,000 steps), one trial each, GPU
run 4 (17 Nov, this version: the stock MUJI chest):

| Run | What it does | `task_score` | What the trace shows |
|---|---|---|---|
| Nothing (settling, 160 steps) | both arms hold still | 0.0 | drawer shut (opening -15 mm, its rest), latch at its 1.04 degree rest sag |
| Scripted demo | below | 1.0 | success from step 505, held to step 521: latch pressed to 10.5 degrees, drawer open 134 mm, hands 70 mm clear |
| `DEMO_IDLE_ARM=left` | only the right hand: it pulls | 0.0 | the drawer stops on the catch at a 38 mm gap; latch never pressed |
| `DEMO_IDLE_ARM=right` | only the left hand: it presses the latch | 0.0 | the drawer stays shut |
| Dream-1 | `--policy dream`, full 1,000 steps | 0.0 | the drawer never opened |

History. GPU run 2 scored 1.0 on the previous version of this task, a custom plywood unit with the same latch, catch,
handle, scoring and demo plan. Run 3, the first on the MUJI chest, scored 0.0: the right hand came straight down from
its ready pose and its rear pad landed on the chest's top (25 mm higher than the old unit's), so it never gripped the
bar. The demo now brings the right hand in from in front of the chest and grips the bar 42 mm right of centre (run 4
above); the offline stand-in now treats a pad pressed into the chest as resting there, so it also scores run 3's demo
0.0.

The offline counterfactuals below are `tools/offline/check_task.py` with this task's stand-in for the physics
(`task.OfflineModel`, offline only): predictions, not measurements, kept to show the scorer's tiers.

| Variant | What it does | Offline prediction |
|---|---|---|
| `DEMO_VARIANT=no_press` | the right hand pulls 150 mm without the press: the hook holds the drawer at a 38 mm gap | 0.0 |
| `DEMO_VARIANT=half_open` | pressed and pulled to an 80 mm gap only | 0.5 |
| `DEMO_VARIANT=hold_on` | pulled 150 mm, the right hand keeps hold of the handle | 0.25 |

One or two arms: the demo uses two (one pulls, one presses), as most people do it with one hand on the handle and a
finger of the other in the gap. The task does not claim that two are needed. One arm could do it in principle: pull
to the stop, let go, reach into the gap with closed pads, press the arm down near its root and push the drawer out
from behind its front. The one-arm runs above are the two-arm demo with one hand idle, so they score 0 because each
hand does only half of the job; they are not evidence that one arm cannot do it.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The right pads close along
x on the bar (yaw +90 degrees); the left pads, closed, are about 12 mm thick along x in the gap (yaw -90 degrees).
Drawer positions below are gaps (the back of the front past the carcass's edge); the slider's travel is 15 mm more.

| Time (s) | Left gripper (the press) | Right gripper (the handle) |
|---|---|---|
| 0 to 6.7 | pads closed, wait at (270, 200) mm, 195 mm up, clear of the right arm | to 35 mm in front of the bar, 190 mm up, the pads closing to 22 mm on the way (0 to 2.0); over the bar 42 mm right of centre, 180 mm up (to 3.2); down with the pads straddling it, grip it |
| 6.7 to 9.7 | wait | pull to a 42 mm gap (the drawer stops at 38 mm on the catch), hold 0.5 s, ease back to a 35 mm gap (the hook's face 3 mm off the lip) |
| 9.7 to 14.7 | over the latch (x 289, y 140 mm: the pads' centre 24 mm behind the back of the front, 5 mm clear of the carcass's edge), down into the gap, press the arm with the pad tips commanded 12 mm below its top, hold 1 s | hold at a 35 mm gap |
| 14.7 to 16.7 | keep pressing while moving out with the drawer to a 65 mm gap at 15 mm/s (the hook passes under the lip from 38 to 50 mm) | pull to a 65 mm gap at 15 mm/s |
| 16.7 to 19.7 | settle 0.5 s, straight up (the latch springs back, its hook now in front of the catch), aside | hold 1.5 s while the left hand leaves, then pull to 150 mm of travel (a 135 mm gap) at 30 mm/s (18.2 to 20.5 s) |
| 19.7 to 24.5 | parked | settle, open the pads to 22 mm, straight up, aside |

The press depth allows for the arm's give. The arms hold their joints with PD springs and give about 2 mm per newton
downward at the press pose, and the tilting latch arm meets the pads' edge nearest its pivot, 13 mm from it. Commanded
12 mm into the arm, the pads come to rest at about 0.2 rad (0.17 rad at 2.5 mm/N, 0.11 rad at 4 mm/N; 0.084 rad
clears the hook); a stiffer arm only meets the latch's stop at 0.25 rad. Pressing near the pivot is deliberate: the
first GPU run pressed in the middle of the gap, 5 mm from the pivot, with the tips 3 mm into the arm, and reached only
2.5 degrees (4.8 clears the hook). The through-pull is slow (15 mm/s) so that the pulling hand lags the pressing one
by little and the pads stay where they press. All of this is unchanged from GPU run 2 on the earlier unit, which scored 1.0, as did run 4 on this chest: the pads meet
the latch at the same place relative to the drawer front, only 25 mm higher.

The right hand never passes over the chest. Over the bar it cannot get high enough for that: straight down there the
arm reaches no higher than 195 mm, and the pad tips hang 13 mm under the TCP, so a hand coming from its ready pose
over the chest lands on the 185 mm top (GPU run 3). It first goes to 35 mm in front of the bar, 190 mm up, closing
its pads to straddle on the way, and only then over the bar at 180 mm, its tips 18 mm in front of and 5 mm below the
top's edge. It grips 42 mm right of centre: the pads' rear corners stay 4.5 mm clear of the handle's leg (64 mm) and
the forearms stay apart. The left hand approaches 195 mm up over the latch, 20 mm under its reach limit there, its
pads 5 mm in front of the carcass's edge.

Offline: worst IK 0.09 mm; smallest joint margins 5.1 degrees (left joint 4) and 4.4 degrees (right joint 4); the
forearms (link capsules) pass 27 mm apart at t 16.7 s and the TCPs stay 194 mm or more apart (grip points 182 mm
apart along y). No pad comes within 1 mm of the chest, drawer, handle or catch except where it should touch. The
demo is 491 steps (24.5 s), under half of the 1,000-step limit; the scripted policy stops after 551.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Open the child-proofed drawer at least 100 mm and let go." \
  --scorer task_score --max-steps 1000 --store-frames
```

For Dream-1 use `--policy dream`. Nothing is random (`object_jitter=0`), so no seed is needed.
`TASK_TRACE=trace.jsonl` writes one line per control step: opening, latch angle, the hook's front and top in the
chest's frame, the hook's overlap with the lip, the let-go and stillness flags, both TCPs and openings, and the last
3 s of (step, opening, latch angle). `DEMO_HOLD_ONLY=1` is the settling check, `DEMO_IDLE_ARM=left|right` the
one-arm check, `DEMO_VARIANT=<name>` the counterfactuals above (never the submitted demo).

Offline checks:

```bash
cd tasks/drawer-child-safety-catch-pressed-and-pulled-open && python3 -B -m unittest discover -s tests
python3 -B tools/offline/check_task.py tasks/drawer-child-safety-catch-pressed-and-pulled-open --require-json --strict --one-arm
```

## How it differs from tasks already on the board

- **Open a drawer held shut by a bar (accepted).** There a locking bar runs through two sleeves on the cabinet's
  front corners: the job is to slide a visible, loose obstacle out and then pull, one after the other. Here the
  lock is a sprung latch inside the drawer, out of sight, reachable only through the gap the first pull makes; the
  press must be held at the same time as the drawer moves, and pulling harder makes it hold tighter. The object is
  the household child-safety latch, not a cabinet locking bar.
- **Take a drawer off its slides by pressing two release levers while pulling (accepted).** The closest skill
  (press a release while pulling), but a different job and setting: removing a drawer from metal ball-bearing
  slides, with the levers on the slides beside the drawer, the drawer carried out and set down. Here the drawer is
  only opened in normal use, through a child latch reached in a 35 mm gap, and left open.

## Simplifications

- The latch's arm is a flexing plastic strip; it is modelled as a rigid arm on a pin 4 mm above the latch's foot
  with a torsion spring (0.30 N m/rad, est). The pin sits above the hook's contact patch, as the flexing root does
  on the real part, so a pull cannot cam the hook out.
- The real hook has a ramp on its back, so a drawer pushed shut clicks past the catch and re-latches. The modelled
  hook is a square block: closing would stop on the catch's front. Closing is not part of the task (see reset).
- The chest and the catch are fixed (screwed down). The catch is its own fixed part because links of one mechkit
  assembly never collide with each other and the hook must meet the catch.
- Drawer drag (wood on wood) is joint damping (60 N s/m, about 1.8 N at the demo's 30 mm/s), not Coulomb friction; joint friction
  has no visible effect on the GPU (found on an earlier mechkit task).
- The latch and catch are simple boxes at the listed size; the catch's real profile (a hooked lip on a screw plate)
  is a 4 mm plate and a 6 mm lip.
- The chest's panels and the drawer's box are plain boxes at the published outside and drawer-inside sizes, with
  estimated thicknesses (below). The real drawer has no stop and comes out past about 250 mm; the simulated slider
  ends at 200 mm of travel, beyond the demo's 150 mm.

## Sources and estimates

Sourced:

| Figure | Source |
|---|---|
| The chest: MUJI stacking chest, half, 1 drawer, oak veneer, JAN 4550512381105; outside W37 x D28 x H18.5 cm; drawer inside W33.5 x D24.5 x H15.5 cm; particle board, MDF, white oak veneer; drawer load 3 kg; A$79.95 | MUJI Australia, https://muji.com.au/products/stacking-chest-half-1-drawer-ke276a2a |
| Same chest (also listed as "Oak Cube Drawer Unit, 1 Tier Half"): 37 x 28 x 18.5 cm, drawer inside W33.5 x D24.5 x H15.5 cm, 5.00 kg, EUR 44.95 | MUJI Europe, https://www.muji.eu/products/oak-cube-drawer-unit-1-tier-half-10462 |
| Same chest in Japan (product 12381105): W37 x D28 x H18.5 cm, about 5 kg, 3,490 yen | https://www.muji.com/jp/ja/store/cmdty/detail/4550512381105 and https://mirumu.jp/en/products/muji-4550512381105 |
| Latch fitted to the back of the drawer front at the top; catch "1 in from inside edge of cabinet frame"; "When drawer is opened 1.5 in the latch will engage catch. Press down on latch with finger to open." | Safety 1st Cabinet and Drawer Latches instructions (models 48444, 48390, HS204), https://safetystore.iu.edu/content/Product%20Instructions/Safety%201st/SF%20Cab%20%26%20Drawer%20Latches%20instructions.pdf |
| Latch 0.83 x 1.18 x 3.27 in (21 x 30 x 83 mm), Safety 1st cabinet and drawer latch, 14 pack, UPC 052181483905 (48390) | Target listing, https://www.target.com/p/safety-1st-cabinet-and-drawer-latch-14pk-white/-/A-92118216 |
| IKEA BAGGANAS handle 143 mm, stainless steel (703.384.18): length 143, width 9, depth 31 mm | https://www.ikea.com/se/en/p/bagganaes-handle-stainless-steel-70338418/ |
| Same handle: hole spacing 128 mm, drilled hole 5 mm | https://www.ikea.com/ch/en/p/bagganaes-handle-stainless-steel-70338418 |

Estimates (est), each with the bench measurement that replaces it on the bought chest and latch. MUJI publishes the
outside and the drawer's inside only; the panel figures below are chosen to fit those two sets of sizes.

| Figure (est) | Value | Bench method |
|---|---|---|
| Chest panels | top and bottom 12 mm, sides 9 mm, back 4 mm | calipers on the bought chest's edges |
| Drawer front | inset, flush with the carcass, 346 x 157 x 15 mm; 3 mm side and 2 mm top and bottom gaps | calipers and a rule with the drawer shut; if the front overlaps the carcass instead, the gap figures still hold (they are taken from the back of the front) and only the 15 mm offset changes |
| Drawer box | sides and back 6 mm, bottom 4 mm; sides 159 mm tall, riding on the carcass bottom | calipers with the drawer out |
| Drawer mass | 1.4 kg empty | kitchen scale |
| Drawer drag | 60 N s/m, about 1.8 N at 30 mm/s (wood on wood) | push-pull gauge (or a 10 N spring scale) on the handle, pulling the empty drawer steadily (no source for the value: est) |
| Latch arm thickness; hook height and length | 4 mm; hook to 14 mm above the arm's underside, 6 mm long | calipers on a bought 48390 latch |
| Catch: plate, lip, width | 4 mm plate, 6 mm lip hanging 10 mm, 25 mm wide | calipers on the catch from the same pack |
| Hook bite behind the lip | 6 mm | fit both as the guide says, pull the drawer to the stop, feeler gauges between the hook's top and the lip's lower edge while pressing the arm until it slips past |
| Latch spring | 0.30 N m/rad: about 2 N at 12.5 mm from the root draws the hook clear | 10 N spring scale (or a kitchen scale under a pin) on the arm 12.5 mm from the foot; read the force at which the drawer, pulled lightly, slips past the catch |
| Arm stop | 0.25 rad (about 18 mm at the hook) | press the arm until it bottoms; rule at the hook |
| Latch mass | 12 g | kitchen scale |
| Handle bar section along x, leg radius | 9 mm, 4 mm | calipers on the handle |
| Thresholds 60 and 100 mm, 30 mm, 5 mm, 1 s | rule choices | steel rule, 30 mm block, 30 fps video (table above) |

## Fitting and reset

Buy the chest, a 48390 pack and a BAGGANAS 143 mm handle. Drill two 5 mm holes 128 mm apart through the drawer
front, centred, 31 mm below its top edge, and screw the handle on. Fit the latch and catch as the guide says: latch
foot on the back of the front at the top, centred 140 mm left of the drawer's centre (seen from the front); catch
under the chest's top 25 mm in from its front edge, aligned with the hook. Screw the chest to the table through its
bottom panel (two screws, drawer out) with its front edge at x 300 mm, centred on the arms' midline. Check with a
rule that the drawer stops with 35 to 40 mm between the back of its front and the carcass's edge.

Reset: press the latch arm down through the gap and push the drawer shut (a real latch's ramp clicks past the catch
by itself); pull once to check that it stops with about 38 mm behind its front, then push it shut again. Between
trials check the latch and catch for cracks or bending and that all screws are tight. Nothing is consumed.

## Files

| File | What |
|---|---|
| `layout.py` | every dimension (published or est), the hook geometry (clear angle, press angle) and the thresholds |
| `parts.py` | the chest, drawer, handle, latch and catch as mechkit shapes |
| `scoring.py` | the scorer (`Tracker`), plain functions of the part states and the arms |
| `task.py` | the scene (one fixed articulation plus the fixed catch), the offline stand-in, the hooks and the demo |
| `mechkit.py`, `rigstate.py` | the shared mechanism kit and arm reader, unchanged |
| `tests/test_scoring.py` | geometry, tiers, zero conditions, bad states, exploits, the offline demo and one-arm runs |
