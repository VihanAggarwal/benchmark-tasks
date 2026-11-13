# Sharpen a new pencil in a crank sharpener and lay it back in its rest

A desk crank sharpener only cuts while two things happen at once: the pencil's end is pressed onto the cutters and
the crank turns. The robot has to do both with two arms. The left gripper takes a new, unsharpened pencil off its
pen rest, slides it coaxially into the sharpener's bore and keeps it pressed in while the cut draws it 24 mm deeper
(1.5 mm a turn over 16 turns); the right gripper carries the crank knob round a 90 mm vertical circle against the
cutting drag. Once the pencil stop ends the cut, the left gripper pulls the pencil straight out, lays it back in the
two V notches of its rest and lets go. Everything is real size: an X-ACTO KS 1031 manual sharpener screwed to a pine
2x4 offcut, a Dixon Ticonderoga My First pencil without eraser (33036), a two-notch pine pen rest.

## Why it is hard

- **Coaxial insertion.** The 10.3 mm pencil goes into a bore 12.3 mm across flats (1.0 mm a side; the real
  7/16 in, 11.1 mm hole gives 0.4 mm) through two narrow ring sections whose outer ends are 25 mm apart: the
  pencil's axis has to be within about 1 mm and 4.6 degrees ((12.3 - 10.3) / 25 rad) of the bore's, 118 mm above the
  table, from a top-down grip 103 mm behind the end that goes in.
- **A press that follows the cut.** The cutting front only cuts while the pencil's end is within 1 mm of it and
  pushes its feeler in at least 2 mm (2 N), and the crank turns forward: 1.5 mm a turn, 24 mm in 16 turns. As wood
  comes off, the front moves away from the pencil, so the press fades unless the pencil follows it. A pencil held
  still by a stiff grip stops cutting once its press falls below 2 mm: from the demo's 4 N press the cut stops at
  2.0 mm, and even 50 N (feeler bottomed, the stiff slide behind it taking 50 N per millimetre) stops it at 3.9 mm,
  below the first tier at 6 mm (unit tests `StillPencil`). Pushing without the crank cuts nothing.
- **A continuous vertical orbit against the cut at the same time.** The other gripper carries the knob round a
  90 mm circle in the x-z plane, 283 mm a turn and 4.8 m for the demo's 17 turns, while the knob spins on its pin in
  the fingers. While the pencil is pressed and the point unfinished the crank takes 0.2 N m (0.05 bearing, 0.15
  cutting drag), 4.4 N tangential at the knob, which the gripper must push round the whole circle, straight up and
  straight down included. The arm is compliant: a static calculation from the YAM's joint stiffness (80 N m/rad shoulder and
  elbow, 10 wrist) gives about 160 N/m at the TCP along x and 400 to 550 N/m along z, so a knob commanded on its own
  circle runs 20 to 50 degrees of crank behind the command. Where the knob is in front of the axis going down and back (about
  150 degrees) a command on the knob's own circle can turn the crank with at most 0.235 N m, barely over the 0.2 N m
  it takes.
- **Then a second precise placement.** The pencil has to come straight back out of the bore and go down into two
  90 degree notches 90 mm apart, its axis within 3 mm sideways and 2 mm in height at both, and be let go.
- **Two hands, by reach.** The knob (y -121.5 mm) is in the right arm's column, 427 mm from the left base in y; the
  pen rest (y 220 mm) is in the left arm's column, 525 mm from the right base. One arm cannot both feed and crank.
- **Nothing is chosen.** The layout is the same every trial and the instruction states the method; the difficulty
  is the coordinated physical motion, not a decision.

## Scene

Positions in mm from the midpoint between the arm bases: x forward, y to the robot's left, z up from the table top.

| Object | Real part | Size, mass, material | Start |
|---|---|---|---|
| Sharpener | X-ACTO KS 1031 manual pencil sharpener (listed 4.75 x 2.75 x 4.25 in, 10.4 oz, dual hardened helical cutters, a hardened pencil stop that ends the cut once the point is made, an 8-hole guide dial) | outer envelope 121 x 70 x 108 mm, 295 g, steel and black plastic; fixed | body centred on x 300, y -20 to +40.5 along the bore; bore axis along y at x 300, z 118 |
| Mounting block | pine 2x4 offcut, two #8 x 3/4 in wood screws, one C-clamp | 38 x 89 x 150 mm, 254 g (0.51 L at 0.5 g/cm3); fixed | centre (300, -20), lying flat |
| Crank | the KS's own crank | steel hub 16 mm across, steel arm 45 mm from axis to knob pin (55 g), black polypropylene knob 20 mm across, 50 mm long (14 g), spinning freely on its pin | knob straight up, its centre at (300, -121.5, 163) |
| Pencil | Dixon Ticonderoga My First pencil, no eraser, part 33036, sold unsharpened | round, 10.3 mm (13/32 in) across, 166 mm (6.52 in) long, incense cedar with a graphite core, yellow lacquer, about 9 g (13.8 cm3: cedar about 4.4 g, graphite about 4.2 g) | lying along y in the rest, centre (220, 220, 30.3) |
| Pen rest | any desk pen rest with two 90 degree V notches; here pine | base strip 30 x 130 x 8 mm, two V blocks 30 x 16 x 22 mm with a 90 degree notch 7 mm deep at y 175 and 265; 25 g (50.8 cm3); taped down | centred at (220, 220) |

Product sources: the KS 1031's dimensions and weight (retail listings: 4.75 x 2.75 x 4.25 in, 10.4 oz; 12.07 x 6.99
x 10.8 cm), "dual hardened helical cutters" (maker's copy), the "specially hardened pencil stop which automatically
prevents further cutting after point is made" (dealer listing for the KS), its dial sizes (15/32, 7/16, 3/8, 11/32,
5/16, 9/32, 1/4, 7/32 in). The pencil: 13/32 in across (maker's and Office Depot pages), 6.52 in long for the
eraserless 36-count (Target). Crank radius, knob size, bore height, turns to a point (16), and the bearing and
cutting torques are our estimates for the KS 1031, and so are the bore's layout inside the body (the ring and
cutting-front positions) and the 24 mm cone length; `fidelity` says so.

## How the sharpener is built

One fixed mechkit assembly (`task.py` `SHARPENER`), its base frame on the table at (300, 0), not turned:

| Body | Joint | What it is | Limits, drive |
|---|---|---|---|
| base | fixed | the offcut, the body solid round a 19 x 19 mm passage along the bore axis, the crank end wall, the dial ring (y 34.5 to 37.5, an octagon of eight turned boxes, 12.3 mm across flats) behind a 45 degree lead-in 3 mm deep, the cutter carriage's entry ring (y 12.5 to 16.5) behind a 1.5 mm lead-in | tight-fit contacts (0.5 mm offset, 32 solver iterations), friction 0.4 / 0.3 |
| `slide` | slider along -y | the cut depth: a 4 x 4 mm rod | 0 to 26 mm; spring 50,000 N/m, damping 200 N s/m; target set every step to the cut depth D |
| `front` (on `slide`) | slider along -y | the cutting front: a 14 mm disc, face at y 10.5 at rest, on a sprung feeler | 0 to 5 mm (bottoms on the slide); spring 1,000 N/m, damping 20 N s/m |
| `crank` | hinge about +y through (300, -85.5, 118) | hub and arm | +-6 rad (rewritten by whole turns, below); friction 0.05 N m; drive spring 2.0 N m/rad, damping 0.02 N m s/rad (0.9 of critical for its 6.0e-5 kg m2) whose target the drag rule sets |
| `knob` (on `crank`) | hinge about +y through the knob pin | the knob, free on its pin | +-6 rad (rewritten by whole turns); friction 0.002 N m |

**The cut rule** (`scoring.Cutter`, run by a mechkit `on_step` callback at the start of every control step, before
the physics). The pencil is *pressed* when its front end has gone through both rings (past y 12.5), is within
1.5 mm of the bore axis and within 1 mm of the disc's face, and pushes the feeler in at least 2 mm. The crank's angle
is integrated unwrapped; its *gain* in a step is the forward travel past the furthest angle reached within the last
turn (that furthest angle is never more than one turn ahead, so a long backward run is forgiven after one turn, as
on a real KS whose two cutters sweep the whole cone each turn). While pressed, D grows by 1.5 mm per turn of gain to
24 mm (the pencil stop) and never shrinks; the slide's drive target follows D. The front can never run ahead of the
pencil: as the slide moves in, the feeler relaxes, and the pencil gets at most 5 mm (plus 1 mm per 50 N) past the
uncut wood. Real counterpart: the KS's helical cutters, which shave only a pressed pencil end, so the pencil advances
only as wood comes off.

**The cutting drag.** While pressed, turning forward and D below 24 mm, the crank meets a forward-only friction of
0.15 N m, full above 0.5 rad/s and proportional below. It is a drive target on the crank's 2.0 N m/rad spring:
`target = q + w DT / 2 + (damping w - drag) / K`, which at steady speed averages a resisting torque of exactly the
drag over each 50 ms control step (the drive's damping cancelled), and with drag 0 leaves the free crank with only
its 0.05 N m bearing friction. A crank let go mid-cut springs back by at most drag / K = 0.075 rad (4.3 degrees)
before the next step's target follows it: a simulator behaviour, as a real friction never turns a crank backward.
The value: a 24 mm point removes 2/3 x pi x 5.15^2 x 24 = 1,333 mm3 of cedar and graphite; at 10 to 20 J/cm3 for
thin shavings of a soft wood (our engineering estimate) that is 13 to 27 J over 16 turns (100.5 rad), 0.13 to
0.27 N m; 0.15 N m sits at the low end, where the compliant arm still carries the knob over the top.

**Whole-turn rewrite.** The kit's hinges are limited to +-2 pi and a point takes 16 turns. So every control step,
once the crank's or the knob's angle passes +-pi, the rule rewrites that joint's angle by a whole turn (Isaac Lab's
joint-position writer, velocity unchanged, `task._write_joint_positions`); a whole turn leaves every part in the
same pose, so no contact changes, and the cut rule's unwrapped angle counts the short way across it (unit test
`test_wrap_counts_the_short_way`). The crank's drag target is computed from the rewritten angle and stays within
+-3.5 rad. If no writer works, the run prints `[sharpener] ERROR` once and the crank stops at its 6 rad limit.

**Deviations, each in `fidelity`.** The 7/16 in (11.1 mm) round hole is a 12.3 mm octagonal bore with lead-ins, for
the solver's fit range (the kit builds bores from boxes). The cutters are the disc, feeler and slide above; the
simulated pencil keeps its square end, and the cut shows as the pencil going 24 mm deeper into the bore (its back
end 24 mm closer to the dial face) while the crank turns against the drag. The knob's 50 mm length is our estimate
of the KS knob; at that length the gripper's pads, about 27 mm wide, grip it more than 20 mm from the crank arm.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`, its thresholds `layout.py`. No zero is latched over the
trial: nothing in it does lasting harm (the sharpener is screwed down, the stop prevents over-sharpening, turning
backward is harmless). Doing nothing scores 0.

| Tier | Condition (simulator) | Real-trial check |
|---|---|---|
| 0.25 | D reached 6 mm (four pressed forward turns) | ruler: a cut cone at least 6 mm long at the pencil's end, the flat at its tip at most 7.8 mm across (a 6 mm cut leaves 10.3 x 18 / 24 = 7.7 mm) |
| 0.5 | D reached 12 mm | ruler: a cone at least 12 mm long, the flat no wider than half the pencil (5.2 mm = 10.3 x 12 / 24) |
| 0.75 | D reached 24 mm: the point is complete | ruler and photo: a cone 24 mm long, the graphite brought to a point, no flat |
| 1.0 | D is 24 mm and, at the end: the whole pencil at least 10 mm clear of the sharpener's outer box; its axis at both V blocks within 3 mm sideways and 2 mm in height of the notch seat; released on every step of the last 1 s (`rigstate.released`, every finger zone of both grippers 30 mm or more from the pencil; unreadable arms count as holding it); moved 2 mm or less over that 1 s | side photo: the pencil down in both V notches; video: both grippers 3 cm clear and nothing moving for the last second |

The trial ends once 1.0 has held for 1 s, otherwise at the run limit; the score is the latest step's. The top tier is
judged on the current state, so a pencil laid back and then picked up again loses it.

The press and the turns are simulator thresholds. A real KS cuts at any press, faster as the press rises, so a real
trial counts no turns and reads no force: it reads the cone with a ruler, and each tier's cone length is the D it
stands on.

Exploits the tests check: pushing hard and never cranking (0); cranking with the pencil on its rest (0, and no drag:
the free crank tells the robot nothing is cut); cranking backward while pressing (0); rocking the crank (only net
forward travel counts); a pencil held still by a stiff grip, at 4 N or 50 N (0); a feeler pushed 1.9 mm, an end
1.1 mm off the front, 1.6 mm off the axis, or not yet through the carriage ring (no cut); a pencil tilted and jammed
in the bore (no cut); the pencil held hovering over its rest, held in its notches, left in the sharpener, 9 mm from
the sharpener, one end on the table, 2.1 mm high or 3.1 mm sideways in the notches, fingers resting on it, a finger
zone 29 mm away, let go less than 1 s ago, still rolling (0.75 each); unreadable arms in five forms (never
released); a broken or missing state (0 for that step, the 1 s windows restart). A pencil dropped into the notches
from a few centimetres is allowed (harmless for 9 g). A new trial is recognised from rigstate's and mechkit's reset
hooks, or, as a fallback, when the pencil jumps back onto its start pose in one step (the cut restarts too).

## Counterfactuals and one arm

The two one-arm runs and the full demo were measured on the GPU (marked "GPU"). The other variants were not run
on the GPU; their scores come from the offline replay and the force model below and are marked "offline".

| Run | What it does | Score |
|---|---|---|
| Do nothing | | 0 |
| Left arm alone (`DEMO_IDLE_ARM=right`) | inserts and presses; nobody cranks | 0.0 (GPU) |
| Right arm alone (`DEMO_IDLE_ARM=left`) | cranks 17 turns; the pencil stays on its rest | 0.0 (GPU) |
| Crank backward (`backward`) | the demo with the orbit reversed | 0 (offline) |
| Let go in the bore, then crank (`let_go`) | the left arm inserts, opens and rises; the right arm cranks | 0 (offline): in the simulator the feeler pushes a released pencil back off the disc (2 to 5 N against 0.04 N of friction for 9 g); on a real KS a released pencil stays put and gains at most a fraction of a turn's cut |
| A stiff hold (unit-test model) | the pencil held still while the crank turns 17 times | 0 (offline; D 2.0 mm at 4 N, 3.9 mm at 50 N) |
| Press, then a fixed command (`still`) | the left arm keeps its first press command and does not follow | about 0.5 (offline): the compliant arm acts as a spring, so the push fades as the cut advances and stops below 2 N, near D 14 mm at 200 N/m |
| Three turns backward first, then 17 forward (unit test) | | 1.0 (offline): one turn of backlog, covered by the 17th turn |
| No lay-back (`no_layback`) | the pencil held in the sharpener to the end | 0.75 (offline) |
| Laid on the table 60 mm short of the rest (`table`) | | 0.75 (offline) |
| Full demo | | 1.0 (GPU), success first held at step 1067 of 1,200 |

Why the demo works with compliant arms. The press: the static calculation puts about 185 to 200 N/m at the left TCP
along the bore, in series with the feeler's 1,000 N/m, so the demo commands the pencil's end 26 mm past the front's
free face and moves that command 1.5 mm per commanded turn. The feeler then reads about 4.3 mm (2.8 mm at 120 N/m); a
stiffer arm bottoms it, which also counts as pressed (unit test `test_press_with_a_compliant_arm`). The crank: an arm
of joint springs (the YAM's 80 and 10 N m/rad) pulls the knob round only as far as its command runs ahead. On the
knob's own 45 mm circle the weakest point (the knob in front, going down and back) gives at most 0.235 N m against
the 0.2 N m load, and a 20 % softer arm, or a bearing twice as stiff, stalls the crank there in a step-by-step model
of the orbit. The demo therefore commands a circle 15 mm larger: the weakest point then gives 0.37 N m (0.29 N m
with a 20 % softer arm; unit test `test_compliant_arm_carries_the_crank`), the crank lags at most about 35 degrees,
and the bearing takes the radial pull (about 2 N while cutting, at most about 6 N with the crank free, held by the
pads' friction). The knob stays on its 45 mm circle; only the command is larger.

## Demo and run limit

`demo()` is open loop: a fixed schedule of TCP poses and finger targets that reads no simulator state. Both grips are
straight down with the fingers closing along x; every open and close takes 1.5 s.

- Left arm (pencil), 53.2 s: over the pencil 20 mm behind its middle (between the V blocks), down, close (a 2.3 mm
  command on the 10.3 mm pencil, about 20 N a finger), lift to the bore's height, onto the bore line with the end
  19.5 mm off the dial face, slowly in through both rings (2 s), press (1 s), then one waypoint per control step
  through the orbit, the end's command 26 mm past the free face and 1.5 mm deeper per commanded turn for 16 turns;
  hold, pull straight out (2 s), carry back over the rest, lower to 2 mm above the notches, open, rise to 125 mm
  (the open finger zone 50 mm above the pencil), hold.
- Right arm (crank), 46.5 s: over the knob, down either side of it, close (a 12 mm command on the 20 mm knob, 10 mm
  outboard of its middle), wait for the press (to 11.4 s), then 17 forward turns, one waypoint per control step,
  1.8 s a turn at full speed (knob 0.157 m/s) with 1 s linear speed ramps at both ends (31.6 s), hold, open, rise.
  The orbit is commanded on a circle 15 mm outside the knob's (60 mm radius, blended in over the first ramp and out
  over the last, so it starts and ends on the knob at the top); see below.
- Offline (`tools/offline/check_task.py --require-json --strict --one-arm`): predicted 1.0, success held at step
  1084 of the 1,200-step run limit; each arm alone 0; worst IK 0.1 mm; the smallest joint margin 7.4 degrees
  (right joint 4, over the knob); the arms' collision capsules 23 mm apart at the closest (both arms' link4, at 43 s,
  the end of the orbit: the knob back at the top and the pencil deepest).
- Dream-1: one trial at the full 1,200-step run limit scored 0.0. It never brought the pencil to the bore (its end
  finished 119 mm off the bore line) and never turned the crank, so the cut depth stayed 0 mm.

## Real build and reset

Shopping list: one X-ACTO KS 1031 manual sharpener; Dixon Ticonderoga My First pencils without eraser (33036); a
pine 2x4 offcut 150 mm long, two #8 x 3/4 in wood screws and one C-clamp; a two-notch pen rest (or two pine blocks
with a 90 degree V-groove on a strip), tape. Screw the sharpener to the offcut through its base, clamp the offcut to
the table so the crank clears it, turn the dial to the 7/16 in hole.

Reset: each real trial uses up a pencil, so start every trial with a new, unsharpened pencil (166 mm, both ends
square) and never re-use a sharpened one. Check the dial is at its 7/16 in hole. Empty the shavings drawer every ten
trials. Lay the pencil in both notches, centred between them, and turn the crank knob straight up. In the simulator
the cutting slide, feeler, crank and pencil return to their start poses.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details |
| `task.py` | The `yam_arena` task file: the sharpener, pencil and rest, the cut and drag rules with the whole-turn rewrite (`on_step`), the offline reconstruction of the joints, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The cut rule (`Cutter`), the drag target (`crank_target`) and the scorer (`Tracker`: tiers, clearance, notches, release and stillness windows) as plain NumPy functions |
| `mechkit.py` | Our mechanism kit, version 2, a copy of `tools/kit/mechkit.py`: compound bodies (turned boxes for the octagonal rings, lead-ins and V notches), sliders and hinges with springs, tight-fit contacts, per-step drive targets and `on_step` |
| `rigstate.py` | A copy of `tools/kit/rigstate.py`: reads both grippers each control step; its fail-closed `released` is the scorer's release check |
| `tests/test_scoring.py` | Offline tests: geometry and masses, the cut rule and its thresholds, the still-pencil force model, the drag, every tier and its boundaries, the top tier's conditions broken one at a time, exploits, unreadable arms, resets, the offline joint reconstruction, the demo's structure and timing, the text |

Third-party assets: none. Every part is built from boxes and cylinders in `task.py` and `mechkit.py`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`, and from the repository root
`python3 tools/offline/check_task.py tasks/crank-sharpen-a-pencil --require-json --strict --one-arm`. Offline, the
checker carries the pencil rigidly and cannot turn a joint, so `success()` rebuilds the sharpener's joints from what
it plays (`task.OfflineSharpener`: the crank angle from the right TCP while that gripper is closed on the knob, the
feeler from the pencil's end); in the simulator the joints are read from the physics and the reconstruction is never used.

## Requirements and running

The Isaac Lab-Arena environment from the program's Brev launchable (Isaac Lab-Arena 0.3.1, Isaac Sim 6.1) with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install.

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python3 -c "import json; print(json.load(open('task.json'))['instruction'])")" \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/crank-sharpen-a-pencil/demo
```

Environment switches: `DEMO_IDLE_ARM=left|right` (one arm), `DEMO_HOLD_ONLY=1` (no plan: a settling check),
`DEMO_VARIANT=still|backward|let_go|no_layback|table` (the counterfactuals above). `TASK_TRACE=trace.jsonl` writes
one JSON line per control step: `D_mm`, `tier`, `score`, `clear_mm`, `on_rest`, `notch_mm`, `released`,
`released_1s`, `path_1s_mm`, `still_1s`, `zone_mm`, `arms_readable`, and under `cut` the rule's `pressed`,
`on_axis`, `off_axis_mm`, `end_y_mm`, `touch_mm`, `slide_mm`, `feeler_mm`, `crank_rad`, `crank_w`, `turns`,
`gain_rad`, `turns_cut`, `drag_nm`; under `wrap` the joint writer used and its count. `MECHKIT_TRACE=<file>` adds
mechkit's own per-step record.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
