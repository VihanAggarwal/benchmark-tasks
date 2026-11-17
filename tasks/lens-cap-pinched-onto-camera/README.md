# Put a centre-pinch lens cap back on a DSLR's kit lens and stand the camera down

An entry-level DSLR, reference unit Canon EOS 2000D (Rebel T7) with its kit lens, the EF-S 18-55mm f/3.5-5.6 IS II,
stands on its base on the table with the lens pointing sideways. Its front cap, the Canon E-58 II supplied with that
lens, lies face up beside it. The job is the one done before the camera goes back in its bag: put the cap back on so
that it latches, and leave the camera standing. The instruction does not say how the cap works, that the camera has
to be lifted and held, or that the cap must be flat on the lens before it is let go.

The E-58 II is a centre-pinch cap. Two finger tabs on its front each work one latch tooth on its back; pinching the
tabs pulls both teeth in (2.5 mm, against a 2 N spring each, est) so the cap passes the lip of the lens's 58 mm filter
thread. Let go with the cap flat on the lens, the teeth spring out behind the lip and the cap holds; let go short of
the face, the teeth spring out against the lip's bore, the cap is not on and comes off.

The scorer reads only the end state, not how the cap was held. As on the real cap, the teeth have lead-in ramps, so a
cap pushed on without pinching meets the lip with its ramps and must force both latches in against their springs and
the ramps' friction. Pushing the cap home that way has not been tried in the simulation; the demo pinches.

## Why it is hard

1. **The lens points sideways, 44 mm up.** The cap has to be turned to face the lens and pressed on along the lens's
   axis. A YAM gripper turned onto its side cannot get its TCP that low (in the offline kinematics its joint 4 runs
   out of range below about 55 mm above the table at the lens's place), so the camera must be lifted by one hand and
   held while the other presents the cap: the way a person holds the camera in one hand and caps it with the other.
2. **Pinch, carry, turn, press.** The natural grip on the 12 g cap is the pinch on its tabs, which also frees the
   latches. It must hold through the lift, the 90 degree turn of the wrist and the push, and the cap must arrive
   square (the offline model's lead-in: about 4.5 mm across the tabs, 3 mm the other way, 4 degrees; est) and go
   fully home before the tabs are let go.
3. **A 0.675 kg camera held by its top.** The holding hand grips the pentaprism hump, lifts the camera and must
   keep it still against the push; then set it back on its base without tipping it. The arm bends under the
   camera's weight (in the simulator the lifted camera hangs about 21 mm lower than the hand's target and turns
   about 4 degrees nose down, see Demo), so where the lens ends up is not where the hand was sent.
4. **Hands off.** The score is read with both grippers 30 mm clear and both parts still for 1 s. A cap still pinched
   by the hand has both latches in and does not count.

## Scoring

From the poses of the camera and the cap, the travel of the two latch slides and both grippers:

| Condition | Simulation | Real trial |
|---|---|---|
| Cap home | seat face within 1 mm of the lens's front face (not more than 0.5 mm past it), centre within 2 mm of the lens's axis, axes within 3 degrees; turn about its own axis free | a 1 mm feeler does not enter at four points 90 degrees apart; lens rim showing beyond the cap differs by no more than 4 mm between opposite sides |
| Latched | home, both latches out (each within 0.4 mm of rest) | hold one tab in, turn the camera lens down over a towel for 3 s: the cap stays on; repeat with the other tab |
| Camera down | up axis within 10 degrees of vertical, lowest point within 5 mm of the table | phone level on the top plate; a look along the table |
| Fell (latched) | any part's lowest point more than 30 mm below the table top | a part on the floor |
| Let go and still | last 1 s: finger zones 30 mm or more from both parts; neither moves more than 5 mm or turns more than 5 degrees | video |

Score, held 1 s at the end: **1.0** cap latched and camera down; **0.5** cap latched, camera not down (on its side or
back); **0.25** cap home with a latch not out, camera down; **0** if Fell, if not let go and still, if the cap is not
home, or anything else. A latch out with the cap home is behind the lip: the unit tests check that a free latch
overlaps the lip anywhere from 0.5 mm off the face outward, so it cannot read "out" unless the cap is home.

## Results

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each, 17 Nov 2026 (job
20261117T150758). The offline column is `tools/offline/check_task.py` with this task's offline stand-in for the
physics (`task.OfflineModel`): a prediction, not a measurement.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`), 160 steps | both arms hold still | 0.0 | 0.0 (camera and cap did not move) |
| Scripted demo | below | 1.0 (held from step 527) | 1.0 (success from step 511 to the end at step 530; both latches out behind the lip, cap 0.04 mm from home and 0.08 degrees off the axis, camera level on its base) |
| `DEMO_IDLE_ARM=left` | only the right arm: it presents the cap where the lifted lens would be, about 57 mm above the standing camera's lens axis; its rim meets the top of the barrel and, let go, it falls | 0.0 | 0.0 (the cap ended resting on top of the barrel, not on the lens) |
| `DEMO_IDLE_ARM=right` | only the left arm: lifts and sets down the camera; the cap stays on the table | 0.0 | 0.0 (cap still on the table) |
| Dream-1 | `--policy dream`, full 1,200-step limit | not predicted | 0.0 (neither part moved) |
| `DEMO_VARIANT=short_push` | the cap let go 2 mm short of the lens's face | 0.0 | not run on the GPU |

The first plan (earlier on 17 Nov, job 20261117T102929) scored 0.0: the camera's weight bent the left arm, the lens hung 28 mm
low and the cap was let go in the air. The section below explains how the demo now allows for that.

The task needs both arms: the cap hand cannot reach the lens's axis with the camera on the table, and the camera
hand has nothing to put on the lens.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both grippers close along
x. The right hand is turned about world x with `tiltgrip.py` (a waypoint yaw that carries a full rotation): flat for
the pick, 90 degrees for the press (approach +y, the cap's front toward -y). The left hand is leaned 0.4 rad about
world x for the whole demo, its wrist toward its own base: straight down, its joint 4 runs out of range with the
camera lifted.

| Time (s) | Left gripper (camera) | Right gripper (cap) |
|---|---|---|
| 0 to 7 | over the hump (y +62 mm), down, close 50 mm across the 58 mm hump | over the cap, down (pad tips 2 mm over its plate), pinch the tabs to a 27 mm gap: both latches in |
| 7 to 9 | lift: the hand 80 mm, turning 4.7 degrees nose up (the camera hangs level, lens axis about 101 mm up) | lift the cap to 140 mm |
| 8.5 to 12.5 | hold | turn onto its side and bring the cap 30 mm in front of the lens, on its axis, settle |
| 12.5 to 15.7 | hold | push along the axis at 15 mm/s to 3.5 mm past the face, settle, ease off to 1.5 mm |
| 15.7 to 18.7 | hold | open (the latches spring out behind the lip), straight back 45 mm |
| 18.7 to 25.2 | set the camera down on its base (the hand turning back), let go, up, aside | aside, upright again |

Where the lens is while it is held. The arms are joint-PD position control that compensates gravity for their own
links only (joints 1 to 3 at 80 Nm/rad, 4 to 6 at 10 Nm/rad), so the camera's weight bends the left arm. The first
GPU run (hand not turned) measured it: the left TCP 21 mm below and 8 mm behind its target, the camera turned 4.2
degrees nose down (and about 3 degrees about y, 1 degree about z), the lens's face at (336, -18, 96) mm instead of
(340, -30, 124). The cap, presented where that plan expected the lens, stopped 26 mm above the axis and 8 mm short of
the face, and fell when let go. The demo now turns the left hand 4.7 degrees nose up as it lifts, and the right hand
aims at the lens where it is predicted to hang: the measured bend plus the change a joint-stiffness model of the arm
predicts for the turned hand (`task.HANG_*`, `task.held_lens`: face about (336.1, -20.3, 101.1) mm, axis level and
1.3 degrees off -y, the hand turned to match), less the right hand's own error measured in that run (1.2 mm in x,
1.0 mm in z). Offline: worst IK 0.09 mm; smallest joint margins 7.3 degrees (right joint 5, hand on its side), 9.8
degrees (right joint 4) and 10.6 degrees (left joint 6); demo 505 steps (25.2 s), the scripted policy stops after
565, under half of the 1,200-step limit.

Clashes (link capsules of `tools/offline/yamkin.py`, parts placed by the offline model): arm to arm 86 mm at the
closest (right pad vs left link 4, while both close, t 6.1 s); TCP to TCP 122 mm (during the press); the right
gripper body stays 100 mm or more from the lens. The right pads' capsules reach the lens's front (0.3 mm overlap) at
the end of the push, where the plan commands the pads 1 mm off the face: the capsules are larger than the pads, the
pads sit inside the cap's 62 mm plate, and the cap stops the hand first.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "Put the lens cap back on the camera and leave the camera standing on the table." \
  --scorer task_score --max-steps 1200 --store-frames
```

Runtime: the 1,200-step Dream-1 trial took about 4 minutes of wall-clock on an L40S including start-up (17 Nov). Seed: nothing is random; `object_jitter=0` fixes the start layout, so no seed is needed.

For Dream-1 use `--policy dream`. `TASK_TRACE=trace.jsonl` writes one line per control step: both parts' poses, both
latches' travel, the cap's position on the lens's axis, the conditions, both TCPs and openings.
`DEMO_HOLD_ONLY=1` is the settling check, `DEMO_IDLE_ARM=left|right` the one-arm check, `DEMO_VARIANT=short_push`
the counterfactual above (never the submitted demo).

## Simplifications

- Rigid bodies. The camera is one body: boxes for the body, hump, grip and eyecup, a cylinder for the barrel; the
  lens's front lip (56 mm across inside, 1.5 mm deep) and the groove behind it (60 mm across, 7.5 mm deep) are
  24-sided rings of boxes. The filter thread is not modelled: the lip stands for its crests, the groove for its
  root. Both are deeper than on the real lens: a 58 x 0.75 mm filter thread has its crests about 57.2 mm across and
  its root 58 mm (ISO metric fine thread), against 56 and 60 mm here, so the simulated latches bite 1.4 mm behind the
  lip where the real ones catch in a thread about 0.4 mm deep. Measure the cap's latch travel and the lens's thread
  before the first trial (table below).
- The cap is one articulation (mechkit): a plate with two fixed centring lugs, and two sprung slider links, each a
  tab on the front and a tooth on the back. A tooth is a ramp (crest 1.6 mm outside the lip) and a flat block whose
  top face is the catch, 0.3 mm behind the lip's rear face when the cap is home and overlapping it by 1.4 mm. A
  latch that is out has 0.4 mm to the groove's wall, so the cap cannot slide across far enough for a catch to lose
  its bite. Latching is joints, springs and shapes; no software rule. The links do not collide with their own plate.
- The lens is at 18 mm (retracted); the zoom and focus rings, the hood bayonet and the strap are not modelled.
- Estimates, to measure on the bought units: the cap's size, tabs, latch travel and spring, the lens's front lip
  and groove, the height of the lens axis above the base, the hump's size, the mass split between body and lens,
  friction (camera 0.6, cap 0.45).

## Sources

- Camera body: Canon EOS 2000D specifications, 129.0 x 101.3 x 77.6 mm:
  https://www.canon.co.uk/cameras/eos-2000d/specifications/ ; weight 475 g with battery and card:
  https://www.imaging-resource.com/PRODS/canon-t7/canon-t7A.HTM and
  https://petapixel.com/pixelmeter/canon-eos-rebel-t7-2000d/
- Lens: Canon EF-S 18-55mm f/3.5-5.6 IS II specifications, 68.5 x 70 mm, 200 g, 58 mm filter, cap E-58 II:
  https://www.canon.co.uk/lenses/ef-s-18-55mm-f-3-5-5-6-is-ii-lens/specification.html and
  https://asia.canon/en/business/ef-s-18-55mm-f35-56-is-ii/specification
- Cap: Canon E-58 II 58 mm centre-pinch lens cap (Canon 5673B001), retail listing:
  https://www.bhphotovideo.com/c/product/899348-REG/canon_5673b001_lens_cap_for_58mm.html . Its own dimensions are not
  published; every cap figure here is (est).
- No third-party assets, meshes or scans: every shape is a box or a cylinder.

## Real build and reset

A Canon EOS 2000D (or Rebel T7) with the EF-S 18-55mm IS II at 18 mm, battery and card in, strap tucked under the
body, no hood; its E-58 II cap. Reset: pinch the tabs and take the cap off; lay it face up at x 250 mm, y -130 mm;
stand the camera on its base with the lens toward the right arm, lens axis at x 340 mm, the lens's front face at
y -30 mm (tape marks at the base's corners). Nothing is consumed; between trials check that both tabs spring back.

Measure before the first trial (calipers, 0-10 N spring scale, kitchen scale); the measured values replace the
estimates above, and a unit that fails a check is not used:

| Check | Pass |
|---|---|
| Camera with lens, battery and card (scale) | 0.62 to 0.72 kg |
| Lens axis above the table, camera on its base (calipers) | 40 to 48 mm |
| Across the two tabs, out and pinched (calipers) | out 45 mm or less, pinched at least 4 mm narrower |
| Pinch force to free the cap (spring scale on one tab, the other against a stop) | under 10 N |
| Hump across x (calipers) | 50 to 70 mm |
| Cap on, one tab held in, pull on the opposite rim | stays on at 3 N |
| Lens front: across the filter thread's crests, and each latch tooth's reach past it (calipers) | record both; they replace the 56 mm lip and the 1.4 mm bite |

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, `success()`, `score()`, `demo()`, the variant, offline model |
| `layout.py`, `parts.py`, `scoring.py` | Every dimension; the shapes of each part; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as compound rigid bodies and sprung sliders; read both grippers each step |
| `tiltgrip.py` | Turned grips for the scripted demo (a waypoint yaw carrying a full rotation) |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |
