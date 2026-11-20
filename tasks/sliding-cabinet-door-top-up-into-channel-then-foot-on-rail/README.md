# Re-hang a sliding cabinet door: top up into its channel, then foot onto the rail

A low media-style cabinet (620 mm opening, 270 mm clear height) has a double top track (two grooves 38 mm wide and
40 mm deep) and two low rails on its floor. The rear door is already hung. The loose front door, a 22 mm honeycomb
panel 320 x 290 mm with a bar pull on its face, stands on the table in front of the cabinet. The door is 20 mm taller
than the opening, so it cannot go in upright: its top is leaned in under the front groove, lifted up into it, its
foot is swung in over the floor's edge while the top stays in the groove, and it is lowered onto the rail (the rail in
the door's foot groove). Then both hands let go. The instruction names none of these steps.

This is the order track makers give. Knape & Vogt's specification sheet for its plastic tracks and guides for sliding
wood doors (upper guide 2418, lower track 2420 for 3/4 in doors) says: "Height of door should be 7/16" (1.10cm) shorter
than distance between top of guide and bottom of track" and "Place top of door in guide and swing bottom of door into
track." Furniture makers who build the track into the cabinet itself cut the upper grooves considerably deeper than the
lower ones, so that a door can be lifted off the lower track (Seth Janofsky, "Sliding Doors for Furniture", Fine
Woodworking No. 172, 2004). This cabinet is built that way, not with a bought track kit: the door is 20 mm shorter than the
roof-to-floor distance (290 against 310 mm), and its top sits 20 mm up in a 40 mm groove.

## Why it is hard

1. **Order and clearance.** Upright, the top hits the track's front wall before the foot is over the floor (a test
   checks it). Leaned 10 degrees with its top 7 mm under the groove's mouth, the door has 4 to 5 mm to spare each side.
   It must then go up and turn upright together: at 10 degrees and full depth the front face would touch the wall.
2. **The foot window.** With the top at the roof the foot is 14 mm over the rail tops. Lifted less, the foot's strips
   catch the rail or the floor's edge; swung in too low, the foot catches the floor's edge.
3. **The landing.** The foot groove is 14 mm wide over a 6 mm rail: 4 mm each side. Set down more than 4 mm off the
   line, a strip lands on the rail's top and the door stands tilted or perched on it; set down 14 mm or more off, the foot
   stands on the floor beside the rail, which looks hung from above but is not (0.5 at most).
4. **Hands off, rear door kept.** The score is read with the front door let go and both doors still; shoving the rear
   door out of its track to make room is penalised.

## Scoring

| Condition | Rule |
|---|---|
| In the groove | all four corners of the front door's top edge between the front groove's walls (+-2 mm), at z 316 mm or higher (6 mm or more above the mouth), none above the roof (350 mm, +2 mm), none beyond the opening's ends (+-2 mm) |
| Foot down | every corner of the underside (both foot strips) within 3 mm of the floor top (z 37 to 43 mm); a strip resting on the rail top reads 6 mm up and fails |
| Rail in the groove | at both ends of the door the foot groove's centre line within 5 mm of the rail's centre line (x 387 mm): the strips stop it at 4 mm, and a foot standing on the floor beside the rail is 14 mm or more off |
| Pull out | the door's outward normal (its pull side) within 60 degrees of facing the robot: a door turned round fails |
| Centre plane | the door's centre plane (11 mm behind its front face) at mid-height within 7 mm of the front track's centre line (x 387 mm) |
| Lean | angle of the door's up axis from the vertical |
| Inside the opening | the door's ends between the side panels (+-2 mm) |
| Rear door kept | moved 10 mm or less along y, 5 mm along x; leans under 4 degrees |
| Hands off | last 1 s: both grippers' finger zones 30 mm or more from the front door |
| Still | last 1 s: each door moves under 5 mm and turns under 5 degrees (visible motion) |
| Rear door knocked out (latched) | lowest underside corner under z 30 mm, leaning over 10 degrees, or moved over 20 mm in x or 60 mm in y |

| Outcome | Score |
|---|---|
| In the groove, foot down, rail in the groove, pull out, centre plane within 7 mm, lean under 3 degrees, inside the opening, rear door kept, hands off and still | 1.0 |
| In the groove, foot down, pull out, lean under 8 degrees but not 1.0 (lean 3 to 8 degrees, the foot beside the rail, centre plane over 7 mm off, or an end outside the opening), rear door kept, hands off and still | 0.5 |
| In the groove otherwise (foot up on the rail top, leaning out, or the door turned round with its pull inside), or any higher tier with the rear door moved | 0.25 |
| Anything else; not hands off and still; rear door knocked out; a state that is not finite | 0 |

Bench check for each condition on a real build (the sim reads poses; a person reads these):

| Condition | Bench method |
|---|---|
| In the groove (316 mm) | depth gauge from the groove's mouth to the door's top edge (6 mm), or a pencil line on the door's edge 6 mm below its top that must be hidden |
| Foot down (3 mm) | 3 mm feeler gauge slid along the underside: it must not enter at either end; a 6 mm rail top is visible if the foot sits on it |
| Rail in the groove (5 mm) | rule at both ends of the door from the rail's front face to the door's front face at the floor: 3 to 13 mm (8 mm centred); a foot beside the rail reads 22 mm or more, or the rail shows in front of the door |
| Pull out | by eye: the pull faces the robot |
| Centre plane (7 mm) | rule from the track's front edge (x 360 mm) to the door's front face at mid-height: 16 +-7 mm |
| Lean (3 and 8 degrees) | 3 degree and 8 degree angle gauges against the door's face |
| Rear door kept | rule along the rear door's edge against a pencil mark (10 mm); 4 degree gauge |
| Hands off (30 mm) | 30 mm spacer against each hand; 30 fps video of the last 1 s |
| Still (5 mm, 5 degrees) | video of the last 1 s against a rule and a 5 degree gauge |
| Rear door knocked out | by eye: out of its track or leaning past a 10 degree gauge |

Unintended strategies earn nothing (each has a test): holding the hung door in the hand (hands on: 0); hanging the front
door in the REAR groove (face 46 mm off the front line: top not in the front groove: 0); sliding the rear door out of the
way (moved over 10 mm: 0.25 at most; over 60 mm or tilted over 10 degrees: latched 0); laying the door on top of the
cabinet or leaning it on the front (not in the groove: 0); standing the foot on the floor beside the rail with the top in
the groove and the door leaning under 3 degrees (rail not in the foot groove: 0.5); hanging the door turned round, pull
inside (0.25); the idle hand's ready pose near the rear door is not counted
against the hands-off check (only the front door is).

## Files

| File | What it is |
|---|---|
| `task.json` | The text: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The parts, `success()`, `score()`, `demo()`, counterfactual variants, the offline stand-in for the physics |
| `layout.py` | Every dimension and threshold |
| `geom.py` | Plane geometry (x-z) of door and cabinet: clearance by separating axes, pull and hand maths |
| `scoring.py` | The scorer: plain functions of both doors' states and the arms, with a tracker for the 1 s windows and the latch |
| `mechkit.py`, `rigstate.py` | Build the cabinet and doors as compound rigid bodies; read both grippers each step (copied unchanged from the program's kit) |
| `tiltgrip.py` | Pitched grips for the demo (wraps `yam_arena.demo.grip_rotation`) |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests -v` |

Third-party assets: none (every part is a box of the task's own; mechkit, rigstate and tiltgrip are our own code).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (left arm): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Hang the loose white door, still facing the robot, in the cabinet's front track and let go of it, leaving the rear door where it is." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/sliding-door/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 \
  --instruction "Hang the loose white door, still facing the robot, in the cabinet's front track and let go of it, leaving the rear door where it is." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/sliding-door/dream
```

Run one trial per process (the default). Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` drops that
arm's plan; `DEMO_HOLD_ONLY=1` plays no plan (a settling check: nothing may move); `DEMO_VARIANT=short|out` plays a
counterfactual (never the submitted demo); `TASK_TRACE=trace.jsonl` writes one JSON line per control step with what the
scorer saw (`top_min_z_mm`, `foot_height_mm`, `tilt_deg`, `face_offset_mm`, `rear_*`, `hands_gap_mm`, both TCPs and
openings, and a `hist` of the last 3 s of the key numbers). Videos: `inspect-robots video <log>.json`.

Runtime per trial: about 1.5 to 3 minutes of wall-clock on an L40S once Isaac Sim is running (the demo itself is about
27 s of simulated time, 599 policy steps); the first start of Isaac Sim on a new instance takes several minutes more.
Seed: nothing is random; `object_jitter=0` fixes the start layout, so every run starts from the same poses and no seed
is needed.

## Results

The offline prediction column comes from `tools/offline/check_task.py` with this task's offline stand-in for the
physics (`task.OfflineModel`). It is a kinematic prediction, not a measurement. The GPU column is the physics
simulator (Isaac Lab, 20 Hz control, GPU job of 17 Nov); rows marked not run have no GPU number.

| Run | What it does | Offline prediction | GPU score |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`) | both arms hold still | 0.0 | 0.0 (settle run) |
| Scripted demo | below | 1.0 (held from step 536) | 1.0 |
| `DEMO_IDLE_ARM=right` (left alone) | the same plan | 1.0 | 1.0 |
| `DEMO_IDLE_ARM=left` (right alone) | no plan | 0.0 | 0.0 |
| `DEMO_VARIANT=short` | let go leaning 3.5 degrees, foot 16 mm over the floor and 17 mm in front of the rail; it drops onto the floor in front of the rail and stays leaning 3.5 degrees (foot 2 mm up, face 9 mm off): in the groove and foot down, not 1.0 | 0.5 | not run |
| `DEMO_VARIANT=out` | let go leaning 10 degrees, top just in the groove | 0.0 (falls out) | not run |
| Dream-1 | `--policy dream`, full limit | not predicted | 0.0 |

The door ends closed: hung over y -20 to +300 mm, it overlaps the rear door by 40 mm, as a fitted pair of sliding doors does.

One hand does the whole job in this build: the door is light (0.29 kg), the pull gives one grip in the middle of its
width, and the right arm is not needed (`DEMO_IDLE_ARM=right` scores the same, `DEMO_IDLE_ARM=left` scores 0). The task
does not claim to need two arms. Both arms are free for a policy that wants to use them.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. The door's pose is computed
from its geometry (`geom.py`): keyframes of its top-edge centre (x, z, lean), turned into the hand's pose at the bar.

| Time (s) | Left gripper (pads above and below the 20 mm bar) |
|---|---|
| 0 to 4.5 | hand level, to 60 mm in front of the bar; in to the bar; settle |
| 4.5 to 8.0 | close on the bar (6 mm commanded, about 35 N a pad); lift the door 12 mm off the table (as reached) |
| 8.0 to 9.8 | slide the lifted door 120 mm along y under the groove, level (nothing moves in x) |
| 9.8 to 13.4 | lean it about its foot in four legs to 10 degrees (the foot stays at x 337; top 7 mm under the mouth), settle |
| 13.4 to 17.9 | up into the groove at 10 degrees, then up and swing the foot in: 7, then 3.5 degrees |
| 17.9 to 19.9 | upright, top 4 mm past the roof (as reached; the roof stops it); settle |
| 19.9 to 22.4 | lower to 4 mm over the floor (the unloaded arm's sag puts the foot on it); settle |
| 22.4 to 26.9 | open (1.5 s), back off 130 mm, up 50 mm |

The commanded heights lead the loaded arm's sag. The first GPU run measured the left arm carrying the door by its bar:
past a commanded hand height of 0.930 m it reached 9.7 mm plus 0.36 mm per mm lower than commanded (0.941 reached 0.928,
0.963 reached 0.942, 0.982 reached 0.949), and the door's lean read 2.4 to 4.5 degrees over the hand's pitch. While the hand
carries the door in the air the demo therefore commands the height that the arm reaches as planned (`task.lead_z`) and a
hand pitch 3 degrees under the door's lean; the offline stand-in applies the same measured sag (`task.sagged`) to the
door it carries, so the offline numbers below are for the arm as measured, not for an ideal one. The lean is stepped in the door's own pose (GPU run 2 showed that the straight hand path from the lift to the lean swung
the foot 10 mm toward the floor board and the door stood against the cabinet's face), and the y slide comes first. The keys through the
rise and swing are 7 mm higher than first planned (rise 332 mm, swing 342 and 346 mm): the foot's strips clear the
floor's edge by 7 mm, not 4.

Offline: worst IK 0.09 mm; smallest joint margin 11 degrees (joint 5); peak joint speed 1.7 rad/s; the right arm stays at
its ready pose, 21 cm from the left arm's capsules. The pad tips stay 8 mm or more in front of the cabinet (a test). The
door's pose along the plan keeps 3.4 mm or more from the cabinet and the rear door through the lean, rise and swing (the
separating-axis bound, a lower bound), and presses 4 mm into the roof and 4 mm into the floor by design.

The demo is 26.9 s (538 steps) and the scripted policy holds 3 s more (599 steps in all): under half the 1,200-step limit.
Every gripper open and close takes 1.5 s.

## Simplifications

- Rigid bodies. Plain panels with a bar pull on the outer (front) door, which nothing slides past; the rear door is not
  handled and has no pull. No rollers, door stops, soft-close or lift-out catches. The foot groove is
  two 4 mm strips; the track is three 8 mm walls; the rails are boxes. Friction 0.4 on every surface (est).
- The track is shop-built in the cabinet, as furniture makers build sliding doors (deep top grooves), not a bought kit.
  Its play (8 mm each side of the 22 mm door at the top, 4 mm each side of the rail at the foot) is a choice, looser than
  a closely fitted groove.
- Honeycomb is a solid box of the same mass (0.29 kg, about 3.1 kg/m2, est). The cabinet is solid.
- The bar pull is a convenience of the real door too (a pull is how a door is lifted); it stands 30 mm off the face so
  that the pads fit.

## Sources

The install order: Knape & Vogt, "Plastic Tracks and Guides for Sliding Wood Doors" specification sheet, section 5
Installation (https://xcart.wurthlac.com/skin/common_files/files/specs/KVP2419-TAN-48-SpecSheet.pdf).
Deep upper grooves so the door lifts off the lower track: Seth Janofsky, "Sliding Doors for Furniture", Fine Woodworking
No. 172, September/October 2004 (https://www.finewoodworking.com/2004/10/01/sliding-doors-for-furniture).

Estimates (stated in fidelity): the doors' 22 mm thickness and 0.29 kg, the grooves' and rails' sizes and play, friction.
Store-bought stock for a real build: 22 mm honeycomb door board, 20 x 20 mm aluminium square tube, 8 x 40 mm and 6 x 6 mm
timber or aluminium strip, 16 mm and 18 mm melamine chipboard. No third-party meshes or assets.

## Real build and reset

Cabinet: 16 mm melamine chipboard, 620 mm opening, 318 mm deep, 40 mm plinth under the floor board; three 8 x 40 mm strips
620 mm long glued and screwed under the 18 mm top board at x 360 to 368, 406 to 414 and 452 to 460 mm (grooves 38 mm);
two 6 x 6 mm strips on the floor, centred on x 387 and 433 mm. Doors: two 22 x 320 x 290 mm honeycomb panels, 0.29 kg
each (weigh; trim or add mass), a 14 x 8 mm groove along the foot (route it or glue two 4 x 8 mm strips); on the front
door a 140 mm length of 20 x 20 mm aluminium tube on two 12 x 12 x 30 mm posts 100 mm apart, 170 mm above the foot, centred. Reset:
hang the rear door; stand the front door at x 337 mm, y -140 to +180 mm, pull facing the robot; check the grooves and
rails are clean. Honeycomb doors dent at the foot: swap a dented one.
