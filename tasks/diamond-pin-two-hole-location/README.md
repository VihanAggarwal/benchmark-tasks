# Locate a bracket on one round pin and one diamond pin, flats facing the round pin

Machining, drilling, welding and inspection fixtures, and the nests and pallets of robot cells, locate a part by
two of its holes: one round locating pin, which fixes the part's position, and one diamond (relieved) pin, ground
with two flats so that it only stops the part turning about the round pin. The diamond's flats must face the round
pin, so that the difference between the part's hole spacing and the fixture's pin spacing, which is never exactly
zero, is taken up along the line between the holes. Two round pins, or a diamond turned a quarter turn,
over-constrain the part: it jams or will not go down over both. Fitting the pins the right way round is part of
building or changing over a fixture, done by tool and die makers and machinists.

Here a grey fixture plate on the table has two square sockets 100 mm apart. Two round pins stand on a tray on the
right, the diamond pin on a tray on the left with its head turned the wrong way (its long side along the line
between the sockets), and a blue acetal bracket with two holes 124 mm apart lies in front of the fixture. The robot
has to stand one round pin in one socket and the diamond in the other, turned so its flats face the round pin, and
lay the bracket flat on the fixture over both pins. The clearances are exaggerated so the over-constraint shows at
robot accuracy: the parts' own shapes refuse a wrong set-up, because over two round pins, or over a diamond turned
the wrong way, the bracket cannot come down and rests on the pin heads. The score is what a setter checks with a look
from above, a square and a feeler gauge. Each arm fetches the pin on its own side, so both arms are used naturally;
two arms are not strictly required (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three pins' and the bracket's poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds the diamond pin and the bracket each as one rigid body of several shapes and reports their poses (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, the fit and seat tolerances, the exploits we tried, the bracket's way down from the parts' shapes, resets, invalid states and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and loose cylinders, the standard `task_score` scorer, and one helper module next to
`task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres: a diamond pin (a round shank under a flat-sided
head) and a bracket with two through-holes are not. `mechkit.py` (imported by `task.py`) builds each of them as one
loose rigid body from several shapes (the pin from a cylinder and a box, the bracket from six boxes) through a hook
on `yam_arena.scene`, and reports their poses (base-frame origin and orientation) to the scorer. It changes nothing
else the embodiment does, and it stops the run with an error if either part was not built as a compound body. The
technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the
task file's import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Locate the blue bracket on the fixture by its two holes: stand one round pin in one socket and the diamond pin in the other, turned so its two flat sides face the round pin, then lay the bracket flat on the fixture over both pins." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/diamond-pin-two-hole-location/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Locate the blue bracket on the fixture by its two holes: stand one round pin in one socket and the diamond pin in the other, turned so its two flat sides face the round pin, then lay the bracket flat on the fixture over both pins." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/diamond-pin-two-hole-location/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`tier`, `zero_reason`, `fitted` (socket per pin),
`located_set`, `diamond_turn_err_deg`, `bracket_seated`, `bracket_flat`, `bracket_over_fitted_pins`,
`bracket_x_mm` and `bracket_y_mm` (its centre from the fixture's centre), `bracket_tilt_deg`, `bracket_corner_mm`,
`pin_hole`, `pin_hole_offset_mm`, `socket`, `tilt_deg`, `floor_mm`, `centre_off_mm`, `moved_mm`, `off_table`, the
diamond's and bracket's poses, the pins' bottom centres, `source`). Videos: `inspect-robots video <log>.json`. The
scorer keeps one trial's record per process and starts it over at every reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the three pins' and the bracket's
poses, once per control step (20 Hz), and keeps each part's last half second of positions to tell a part at rest
from one still moving (two points per part: a pin's bottom and top centres, the bracket's two hole centres; at rest
means neither moved more than 2 mm in the last 0.5 s). The score is read at the last step; the trial ends early once
1.0 has held for 2.5 s.

- A pin is **fitted** in a socket when it is upright within 5 degrees (a round pin either end down, the diamond
  shank down), the centre of its bottom face is inside the socket's 55 x 55 mm opening seen from above and within
  3 mm of the table (the socket's floor), and it is at rest. A pin on the plate's top, on its tray or lying down is
  not fitted.
- The diamond is **turned right** when, seen from above, the long side of its head is within 15 degrees of square to
  the line between the sockets: its two flats face the other socket.
- A **located set** is a round pin fitted in one socket and the diamond fitted in the other, turned right (either way
  round: round pin in B and diamond in A, or the mirror image).
- The bracket is **seated** when it lies flat at the plate's height (boss up, level within 2 degrees, every corner of
  its underside within 2 mm of the plate's top), every fitted pin's axis passes through the bracket inside a hole's
  64 x 64 mm outline (a different hole for each pin), and it is at rest.

| Outcome | Score |
|---|---|
| Any pin or the bracket off the table (centre outside its outline, or lowest point more than 10 mm below its top) | 0 |
| No pin fitted | 0 |
| A pin fitted but no located set: one pin only, two round pins, or the diamond not turned right (a bracket seated over such pins stays here) | 0.25 |
| A located set, the bracket not seated (on the table, held up, on the pin heads, over one pin only, or still moving) | 0.5 |
| A located set and the bracket seated over both pins | 1.0 |

There is no 0.75: with a located set the bracket either comes down over both pins or it does not. The spare round
pin may be anywhere on the table (on its tray or not). A new episode is recognised from mechkit's reset counter, or,
as a fallback, when the simulator puts every part back on its starting mark (within 4.5 cm, which covers a seeded
reset's jitter) with at least one of them jumping more than 8 cm in a single step. A simulator state with a
non-finite number, a zero quaternion or a compound without its base frame scores 0 for that step and leaves the
record of the trial as it was.

The geometry behind the rule (tested in `tests/test_scoring.py` from the parts' shapes, independently of the
scorer): the bracket's holes are 24 mm farther apart than the sockets. Over a round pin a hole has 8 mm of clearance
all round; over the diamond turned right, 28 mm along the line and 8 mm across it, because its head is a 48 x 8 mm
bar and its round shank is exactly as tall as the plate is thick, so the bracket meets only the head. With a round
pin in B and the diamond in A the bracket comes down with its centre 4 to 16 mm to the left of the fixture's centre
(at least 5 mm of that range remains wherever the pins stand in their 3.5 mm of play). A second round pin, or the
diamond turned a quarter turn, allows only -20 to -4 mm from socket A's side while the round pin in B needs 4 to 20
mm: the bracket misses by 8 mm with the pins centred, and by at least 1 mm wherever they stand, so it rests on the pin
heads. With the pins centred, a diamond turned up to 15 degrees still lets the bracket down (the window's upper
edge comes down from 16 mm to 9.9 mm; the 15-degree tolerance); from about 32 degrees it no longer does; in between
a seated bracket scores 0.25, as for any set that is not located. A turn error narrows the window further when the
pins stand off-centre in their play (at 15 degrees, with both pins pushed the wrong way, it closes), so the demo sets
the diamond square, and the scoring does not depend on how wide the window is.

Unintended strategies earn no more than they should, as the tests show: the bracket held still just above the pins,
laid on the plate beside them, or turned across the fixture over one pin is 0.5; pins stood on the plate's top next
to the sockets are not fitted (0); pins leaned in their sockets so that the bracket could pass over two round
pins are no longer upright (0); a diamond laid on its side in a socket is not fitted; a bracket
seated over two round pins or a wrongly turned diamond, should the physics ever allow it, scores 0.25. The scorer
cannot see the grippers, so a bracket held still in its seated pose counts as seated (holding it cannot bring it down
over a wrong set); see `scoring` in `task.json`.

## One arm alone

In simulation (L40S, see 'Results in simulation') the scripted demo scores 0.25 with either arm idle, as the
offline kinematic check (`--one-arm`) predicted. With the right arm idle (`DEMO_IDLE_ARM=right`) the left arm stands
the diamond in socket A, turned right, and lays the bracket flat over it, but no round pin is fitted: one pin, no
located set, 0.25 (the bracket seated over the diamond alone does not raise it). With the left arm idle the right
arm stands one round pin in socket B, and the diamond and the bracket stay where they started: 0.25. (A second round
pin in socket A would still score 0.25: there would be no diamond, and the bracket cannot come down over two round
pins.)

Reach (inverse kinematics of the arm's URDF with its link collision proxy): gripping straight down at a pin's grip
height, the left arm reaches at most y -15 cm at x 22 cm, 58 mm short of the nearer round pin's edge (y -21.1 cm);
the right arm reaches at most y +14.5 cm at x 23.5 cm, 81 mm short of the diamond's edge (y +22.6 cm). Both arms reach
both sockets and the bracket. Leaning its gripper 20 to 30 degrees toward the far side, though, an arm does reach the
other tray's pins (and the leaned pose is clear of its own base and the table). So one arm alone could in principle
fetch the other pin with a leaned grip and then do everything in turn; the scripted one-arm runs do not try that, and
we do not claim that one arm cannot score 1.0. Both arms are used naturally: each fetches the pin on its own side.

## Demo and run limit

The scripted demo takes 37.6 s (752 control steps), the arms taking turns at the fixture because its sockets are
only 100 mm apart. Every close and open takes 1.5 s; fingers are commanded 8 mm narrower than the part; every final
descent is slow (at most 25 mm/s on average).

- Left arm, diamond pin (0 to 16 s): grips the head by its two 8 mm ends (yaw 0, fingers along y, pad tips 25 mm
  above the pin's bottom, 10 mm up the head), lifts it to 12 cm, carries it over socket A while turning the hand a
  quarter turn (so the head's long side ends along x and its flats face socket B), lowers it the last 35 mm over 2 s
  to 2 mm above the socket's floor, lets go and goes straight up.
- Right arm, round pin (0 to 26.5 s): grips the nearer round pin across its 48 mm (yaw +90, fingers along x), lifts
  it, waits above its tray until the left arm has left socket A, then sets it in socket B the same way and moves
  aside.
- Left arm, bracket (16 to 37.6 s): grips the bracket's 30 mm boss (yaw -90, fingers along x, pad tips 4 mm above
  the bracket's top face; the open fingers come down outside its 88 mm plate), lifts it so its underside is 28 mm
  above the pin heads, waits until the right arm has moved aside, carries it over the fixture with its centre aimed
  10 mm to the left of the fixture's centre (the middle of its 4 to 16 mm window), comes down to 10 mm above the pin
  heads, lowers it the last 49 mm over 3 s onto the plate (underside aimed 1 mm above it), lets go and moves aside.

Offline (kinematic check): worst inverse-kinematics error 0.09 mm; smallest joint margin 14.7 degrees (joint 4);
arm links at least 4.7 cm apart (capsule proxy), tool points at least 23 cm apart; pad tips never lower than 14 mm
above the table; the carried bracket at least 5 cm from both arms' links. Success has held for its 2.5 s at step 708;
the scripted policy stops at step 813. The run limit is 1,200 steps (60 s).

On the L40S the score reached 1.0 at step 676 and the trial ended as a success at step 724 (36.2 s). At the end the
round pin stood in socket B and the diamond in socket A, each about 1 mm from its socket's centre and upright, the
diamond's head 1 degree off square to the line between the sockets, and the bracket flat on the plate with its centre
11.6 mm to the left of the fixture's centre (window 4 to 16 mm) and 0.8 mm nearer the robot.

For the GPU run: each arm sags a few millimetres toward its base under a load, so the place points carry small trims
(`TRIM` in `task.py`, about half the sag measured in earlier runs, scaled to each part's mass). The bracket comes to
rest on the plate slightly before its slow descent ends; the 2.5 s success hold (rather than 1 s) lets the gripper
open before the trial ends. Read `centre_off_mm` (each pin from its socket's centre; 3.5 mm of play), `floor_mm`,
`tilt_deg`, `diamond_turn_err_deg`, `bracket_x_mm` and `bracket_y_mm` (aim 0 and 10; window across +-8, along 4
to 16) in `TASK_TRACE`; if a part lands off its aim, move its place point by minus the offset with `TRIM`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the two round pins upright on the right tray
at x 22.0 cm, y -23.5 cm and x 22.0 cm, y -29.5 cm; the diamond pin shank down on the left tray at x 23.5 cm,
y +25.0 cm, its head's long side along y; the bracket flat, boss up, centred at x 13.5 cm, y +11.0 cm, its long side
along y. Both sockets empty. The fixture plate and the trays are fixed. Nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 38 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A fixture plate 100 x 250 x 15 mm, made of five pieces of 15 mm steel or aluminium flat bar screwed to a baseboard
(or a 15 mm plate with two 55 mm square pockets cut through it), leaving two 55 mm square sockets 100 mm apart,
clamped 30 cm in front of the arm bases. Two round pins: 48 mm thin-walled steel tube, 55 mm long, capped at both
ends (about 100 g each). The diamond pin: a 48 mm steel disc 15 mm thick with a 48 x 8 x 40 mm blackened steel bar
fixed upright across it (about 80 g; hollow or aluminium parts keep it light). The bracket: 10 mm acetal sheet cut to
88 x 212 mm with two 64 mm square holes on its centre line 124 mm apart and a 30 x 40 x 25 mm acetal block glued at
its centre (about 190 g). Two 3 mm rubber mats as trays. Score from a photo from above (which pin is in which socket,
and the head's direction against the line between the sockets, with a square or protractor for the 15 degrees), a
ruler (each pin's top 55 mm above the table, within 3 mm) and a 2 mm feeler gauge under the bracket's corners. The
parts are light and nothing is powered, so trials are safe.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
