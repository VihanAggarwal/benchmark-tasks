# Double-pot two plants in a planter on risers, each pot rim 10-20 mm below its rim

What is measured is a height against a datum: with a straightedge laid across the planter's rim, a ruler down to each
grow pot's rim must read 10 to 20 mm. Neither pot gets there by itself or on just any support. The two pots differ in
height (35 and 46 mm) and the risers come in three heights (20, 30 and 42 mm), so the robot has to judge two heights
together, the pot's and its support's, against a rim it cannot rest anything on, and the answer differs between the
pots: pot A needs the 30 mm riser (15 mm below the rim), pot B the 20 mm one (14 mm). The wrong riser leaves the rim
too low (pot A on 20 mm: 25 mm down) or too high (pot B on 42 mm: 8 mm above the rim, so the grow pot shows), and once
the pot is on, the riser is hidden under it. There is no fixed shelf or rack to choose, as in tasks that set each item
on the floor or a rack by its height: the robot builds each support itself from three loose risers, laid out in a
different order on each side, before the pot goes on.

This is how interior plantscape technicians and garden-centre display staff double-pot plants: the plant stays in
its plastic grow pot, which is dropped into a decorative planter on a riser (a foam puck, an upturned pot or a riser
ring), so the grow pot's rim sits a little below the planter's rim, where top-dressing hides it and there is room to
water, and its base stands clear of the planter floor, where drainage water collects. Set too high, the grow pot
shows; set too low, the plant looks sunk and is hard to water. Double potting lets a technician swap a tired plant
in minutes without repotting it. Full size, the grow pot's rim is usually set a few centimetres below the planter's;
with 7 cm pots this task scales that to 10-20 mm. The closest O*NET occupation is 37-3011.00 Landscaping and
Groundskeeping Workers, whose tasks include planting foliage and flowering plants and watering plants with watering
cans (sample title: landscape technician); garden-centre and florist display staff do the same.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain NumPy functions of the pots', risers' and arms' states, with a small tracker |
| `rigstate.py` | Reads both arms' TCP, finger tips and opening for the scorer (wraps the embodiment's step and reset; changes nothing it does) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition and stated bound, the stillness window, the reset, the exploits we tried and the demo (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (fixed boxes for the planter, fixed and loose cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Double-pot the two potted plants in the grey trough planter. For each plant, stand one of its white risers on a grey disc on the planter floor, then set the plant's pot on top of that riser so the pot's rim ends 10 to 20 mm below the planter's rim. The risers differ in height: choose the one that gives that gap. Leave both pots standing upright, one on each disc, then let go and move both grippers away from the pots." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/planter-risers-rim-line/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Double-pot the two potted plants in the grey trough planter. For each plant, stand one of its white risers on a grey disc on the planter floor, then set the plant's pot on top of that riser so the pot's rim ends 10 to 20 mm below the planter's rim. The risers differ in height: choose the one that gives that gap. Leave both pots standing upright, one on each disc, then let go and move both grippers away from the pots." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/planter-risers-rim-line/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check). `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: for each pot (`pot_a_...`, `pot_b_...`) `set`, `staged`,
`rim_gap_mm`, `tilt_deg`, `disc`, `disc_off_mm`, `riser` (the riser that carries it, or null), `near_riser`,
`riser_step_mm`, `riser_axis_off_mm`, `riser_low_mm`, `riser_tilt_deg`, `bottom_mm`, `path_mm`, `still`, `tcp_mm`,
`let_go`; then `arms_unavailable`, `bad_steps`, every pot's and riser's position and orientation, both TCPs and both
openings. Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads every loose part's pose once per control
step (20 Hz) and works out every face and rim from the pose and the parts' known sizes. It keeps only two things from
the past: each pot's last 0.5 s of positions, for 'still', and when a new episode starts. Pots and risers are closed
cylinders, the same either end up: tilt is the axis's angle from vertical folded to 0-90 degrees, the 'top face' is
whichever end is higher.

- **Rim gap**: 86 mm (the planter's rim) minus the height of the highest point of the pot's top rim, which is its top
  face's centre plus 35 mm x sin(tilt). Positive is below the rim.
- **Set** (all of): (a) upright, the axis within 3 degrees of vertical; (b) the pot's centre within 20 mm of a disc's
  centre seen from above (the 70 mm pot's outline at most 10 mm past the 90 mm disc); (c) on a riser: the centre of
  the pot's bottom face within 1.5 mm (up or down) of the centre of a riser's top face, the two within 35 mm of each
  other seen from above; (d) that riser upright within 2 degrees with the centre of its bottom face 4 to 7.5 mm above
  the table, which is the planter floor's top (5 mm) or a disc's (6 mm; 7 mm for a riser tilted 2 degrees on its
  edge): a riser on another riser is 25 mm or more up, one on a pot more, one on the table at 0; (e) still: the pot's
  centre moved along a path shorter than 3 mm over the last 0.5 s (10 control steps; the reported velocity is never
  used, since a gripped or resting part's velocity chatters at 10-45 mm/s while it stays put); (f) let go and moved
  away: both grippers' TCPs (the tool point between the pads) at least 60 mm from the pot's centre. No grip reaches
  that far: a straight-down grip pinching only the top 6 mm of pot B has its TCP 30 mm from the pot's centre (23 - 6
  + 13 mm), closed fingers pressing on its top 36 mm, and any grip at all, tilted ones included, less than 55 mm (the
  farthest point of pot B's rim is 42 mm from its centre, the TCP 13 mm past the pad tips). An open gripper left
  hovering closer does not count either: pad tips 20 mm over pot A's top (5 mm over the planter's rim) put the TCP
  50.5 mm from its centre, which is why the instruction says to move both grippers away. Pad tips 30 mm over a pot's
  top are far enough; the demo lifts to 14 cm, 86 mm from pot A's centre and 91 mm from pot B's.
- **Staged**: set, with a rim gap of 10 to 20 mm.
- **Bounds**: every bound stated as inclusive ('within', 'at least', '4 to 7.5 mm', '10 to 20 mm') is compared with a
  tolerance of 1e-9, so a value exactly on it lands in the tier the table gives; 'shorter than' and 'closer than'
  stay strict.
- **Arms**: read through `rigstate.py`; on a step where they cannot be read, 'let go' is taken as met and the step is
  counted in `arms_unavailable` (a policy cannot cause that; the GPU runs check that it stays 0).

| Outcome | Score |
|---|---|
| Both pots staged | 1.0 |
| Exactly one pot staged (for example one arm alone, or one pot on the wrong riser) | 0.5 |
| Neither pot staged, at least one set: a pot on the wrong riser (rim too high or too low) | 0.25 |
| Anything else: nothing done; a pot on the floor or a disc with no riser, on two stacked risers, on a riser lying on its side, tilted more than 2 degrees or standing on the table; tipped more than 3 degrees; still held, or a gripper not moved away (a TCP closer than 60 mm); still moving; more than 20 mm off its disc; outside the planter; left on the table | 0 |

Possible scores: 0, 0.25, 0.5, 1.0. Every outcome falls in exactly one row. The trial ends with success once the 1.0
conditions have held for 1 s (`SUCCESS_HOLD_S = 1.0`; 'let go' is part of them, so the hold is with both grippers
away); otherwise `task_score` is read from the last step at the run limit with the same rules (`SCORE_OVER = "end"`).
There is no whole-trial rule.

Rim gaps for reference, riser on a disc (riser on the bare floor: 1 mm more):

| Support | Pot A (35 mm) | Pot B (46 mm) |
|---|---|---|
| none (on the disc) | 45 | 34 |
| 20 mm riser | 25 | **14** |
| 30 mm riser | **15** | 4 |
| 42 mm riser | 3 | -8 (above the rim) |
| two stacked risers (not 'set') | 5 or less | -6 or less |

The limits in (a), (c) and (d) are set so that no wrong riser can put a pot's rim in the band, however they stack: with
the riser's bottom anywhere from 4 to 7.5 mm up and tilted up to 2 degrees, the pot up to 1.5 mm above or below its top
and tilted up to 3 degrees (which raises its highest rim point by 1.8 mm, 35 mm x sin 3 degrees), pot A on the 20 mm
riser stays at least 20.2 mm down, pot B on the 30 mm riser at most 7.5 mm, pot A on the 42 mm riser at most 6.5 mm,
while pot A on the 30 mm riser stays in the band at every corner (10.2 to 18.5 mm). Pot B on the 20 mm riser spans 9.2
to 17.5 mm; it drops below 10 mm only near the far corner, its riser propped to about 2 degrees and the pot propped to
about 3 degrees on its edge (or lifted clear), where its rim really is less than 10 mm down. No set pot can end 18.6 to
20.2 mm down. Real placements sit well inside: a riser straddling a disc's edge tilts 0.95 degrees and changes the gap
by 0.07 mm; one tilted 2 degrees on a prop leaves pot A on the right riser 12.8 mm down and on the 20 mm one 22.8 mm.
(Earlier, looser limits of 5 degrees, 3 mm and 3 to 9 mm let a pot on a propped wrong riser into the band; the tests
keep those cases.) Two risers of one height side by side under a pot (the instruction says one) give the same rim gap
and count as one: nothing is gained.

Unintended strategies earn nothing extra, as the tests show: holding a pot at the right height instead of setting it
down scores 0 for that pot (not let go), and so does holding a riser under it or leaving an open gripper over it; a pot
on a riser that stands on the table outside the planter, on two stacked risers, on a riser on its side or straight on a
disc scores 0 for that pot; pushing a pot over the wall onto the floor scores 0; tilting a pot on the 42 mm riser cannot
lower its rim; and the wrong pot on a riser is scored on its merits (pot B on a 30 mm riser: set, rim 4 mm down, 0.25).
Pushing, dragging or dropping a part into place ends in a true end state and is scored as it ends: nothing in the scene
can be broken or modified, and the planter and discs are fixed. A missing object, a non-finite number or a zero
quaternion scores 0 for that step and leaves the tracker as it was. A new episode is recognised from rigstate's reset
hook, and as a fallback when every pot and riser is back upright on its starting place (within 4.5 cm, which covers a
seeded reset's jitter) and at least one of them jumped more than 8 cm in a single step.

## One arm alone

The task is bimanual by reach: pot A and the left risers are reachable only by the left arm, pot B and the right
risers only by the right arm. Our kinematics find no straight-down IK for either arm at the other arm's pot (x 20, y
+38.5 or -38.5 cm), at 12 yaws 30 degrees apart and TCP heights 21 to 100 mm (position errors 88 mm or more), nor at
the other arm's risers (heights 19 to 100 mm; errors 9 mm or more). Measured in simulation (the same demo with one
arm left at its ready pose, one trial each; the offline kinematic check predicted the same):

- left arm alone (`DEMO_IDLE_ARM=right`): **0.5**, pot A staged on the 30 mm riser (rim 15.0 mm down), pot B left on
  the table;
- right arm alone (`DEMO_IDLE_ARM=left`): **0.5**, pot B staged on the 20 mm riser (rim 14.0 mm down), pot A left on
  the table.

A tilted grip would not change that: tilting adds 0.10 to 0.16 m to the straight-down reach of about 0.10 m past the
midline, so it might reach the other arm's risers (26 cm past it), which earn nothing on their own, but not the
other pot, whose nearer edge is 35 cm past it. One arm could still push or drag the far pot within its reach and
then do the whole task alone; we claim only that it cannot be done by gripping the parts where they start. The two
arms work at the same time, each on its own side, and take turns inside the planter; they do not need to cooperate.

## Demo and run limit

Both arms run the same 27-waypoint schedule on one clock, mirrored in y, except that the right arm first waits 10 s over
its riser (28 waypoints), so that it enters the planter only after the left arm has left it: 49.3 s (986 control steps;
without the wait both wrists are low over the discs together with 3.8 cm between the arms, and a wait of 2 to 8 s still
leaves 3.8 to 4.0 cm; with 10 s, 20.5 cm). Fingers close along x (left arm yaw -pi/2, right +pi/2), toward the planter's
160 mm inside length. The left arm uses the 30 mm riser (x 27, y +26 cm) and pot A, the right arm the 20 mm riser (x 37,
y -26 cm) and pot B. Finger targets: 80 mm open beside a riser (10 mm clear each side, 20 mm clear of its neighbours at
the 100 mm pitch), 52 mm on a riser's 60 mm, 76 mm to let a riser go inside the planter, 86 mm open at a pot, 62 mm on a
pot's 70 mm. Every close and release takes 1.5 s after a 0.6 to 0.8 s settle, and the last 30 mm of every descent takes
1.5 s (2 s when a pot is set down).

- Riser: over it at 13 cm (2.0 s; the right arm then waits 10 s), down to 19 mm (pad tips 6 mm above the table), grip,
  lift to 13 cm (its bottom 11.1 cm up, 25 mm over the rim), carry it over the disc, down into the planter to TCP 30 mm,
  its bottom 5 mm above the disc (the finger carriages, 57 mm above the TCP, stay just above the rim and 9 mm or more
  inside the walls), open to 76 mm: it drops 5 mm onto the disc. Then up to 14 cm.
- Pot: over it at 14 cm, down to 21 mm (tips 8 mm above the table), grip, lift to 14 cm (its bottom 11.9 cm up, 33 mm
  over the rim), carry it over the disc, down to TCP 60 mm (pot A) or 50 mm (pot B), its bottom 3 mm above the riser's
  top, open to 86 mm: it drops 3 mm onto the riser. Straight up to 14 cm and park at x 20, y +-25 cm.

The offline kinematic check (no simulator): worst IK error 0.088 mm / 0.013 deg; smallest joint margins 8.2 deg
(joint 6 of both arms, at the pots, 0.14 rad) and 17.7 deg (joint 4); peak joint speed 1.18 rad/s (joint 6);
arm-to-arm capsule clearance 20.5 cm and TCPs at least 36.7 cm apart (at t 39.6 s, the right arm going down into the
planter with its pot, the left one parked); lowest pad tips 6 mm above the table; no pad, carried part or link runs
into anything. The kinematic proxy drops each riser 5 mm onto its disc and each pot 3.1 mm onto its riser and predicts
1.0: each gripper is let go and away once its TCP is 60 mm from its pot's centre on the way straight up, and success
(held 1 s) comes at t 47.7 s (step 954), while the right arm moves to park; the GPU demo also ended with success at
step 954 (rim gaps 15.0 and 14.0 mm, both risers 6.0 mm up on their discs). The scripted policy stops at step 1,047;
our commands pass `--max-steps 3600`. The run limit is 3,600 steps (180 s): about 3.7 times the demo, room for a model
to fix a wrong riser.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lift the pots and risers out of the planter
and stand each upright on its tape outline: pot A at x 20.0, y +38.5 cm, pot B at x 20.0, y -38.5 cm, the left risers
(2.0, 3.0, 4.2 cm tall) at x 17, 27, 37 and y +26.0 cm, the right risers (3.0, 4.2, 2.0 cm) at x 17, 27, 37 and
y -26.0 cm. The planter and its discs stay fixed. Nothing is consumed.

## Runtime per trial

About 1 to 4 minutes on an L40S once Isaac Sim is running (the demo is 49 s of simulated time, a full-length Dream-1
trial 180 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Planter: a fibreglass trough of the stated size, or a box of 8 mm plywood (176 x 356 mm outside, 86 mm tall, 5 mm
floor), painted charcoal grey and screwed or taped to the bench. Discs: two 90 mm circles of light-grey vinyl floor
tile, 1 mm thick, glued to the floor at their centres. Pots: two 70 mm straight-sided plastic pots filled with damp
potting mix and capped, weighed to 100 g (35 mm tall) and 130 g (46 mm tall), or turned wooden cylinders of the same
size and mass. Risers: six 60 mm pucks cut from white high-density urethane sign foam (about 0.25 g/cm3), 20, 30 and
42 mm tall, two of each. Everything is light, unpowered and soft-edged; nothing is hot, sharp or pinching. Scoring a
real trial, before touching anything: check that each pot is upright with a small level; measure each pot's centre to
its disc's centre on the photo from above; lay a straightedge across the planter's rim and read the gap down to the
highest point of each pot's rim with a steel rule; then lift each pot out and check that it sat on one upright riser
standing on the floor or its disc.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
