# Splice two guardrail panels at a post, lapped in the direction of traffic

Highway maintenance crews replace W-beam guardrail panels after crashes. On a strong-post run the panels are spliced
at a post, and every splice is lapped in the direction of traffic: the panel that traffic reaches first lies on the
road side of the next one, so a vehicle sliding along the barrier passes over the joint. Lapped the other way, the
next panel's end faces oncoming traffic and can snag or spear the vehicle. Here a 1:8 bench model of the run stands
on the table: three dark brown posts in a row along the far edge of a strip of road, whose white arrows show that
traffic runs from the robot's right to its left. Two grey rail panels (pine battens) stand on edge outside the posts,
one on each side. One panel must stand against the left and middle posts (downstream, D) and the other against the
right post (upstream, U), their ends overlapping 3 to 6 cm, centred on the middle post, with U's end on the road
side of D's end; then both grippers let go and lift clear. The same splice lapped the other way scores 0, so what
is tested is knowing the lap rule and building the end state it asks for. The layout is fixed, so traffic always
runs the same way here: the arrows (60 mm long with 10 mm strokes, about 11 by 2 pixels in the 320 x 176 top
camera image) state the direction, but the task does not test reading a marking that changes from trial to trial. Each arm
reaches only the panel on its own side (see "One arm alone").

It differs from lapping boards over supports (scaffold planks, for example), where the boards lie flat across the
supports, one on top of the other, and the rule is how far each end runs past its support. Here the panels stand
on edge against the faces of the posts, side by side across the barrier's thickness, no end-overhang rule is
checked, and what is scored is which panel's end faces the road: the same overlap with the order swapped scores 0.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, context, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two panels' poses and the grippers' state, with a small tracker for the last half second |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step (a copy of a shared helper, unchanged) |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition and its boundary, resets, the hooks, the demo and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

Full score needs both grippers clear of both panels at the end, and `yam_arena` does not pass the grippers' state
to a task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at
import, to take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before
each control step and to start a new trial at each reset. It reads only state the simulator already has and changes
nothing the embodiment does. If it cannot read the arms it prints why once; neither gripper then counts as released
on those steps, so they cannot score 1.0 (a hand still on a panel must never end a trial early), and the trace
counts them (`arms_unavailable`). A GPU trial with `arms_unavailable` above 0 is not a valid trial.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Splice the two grey guardrail panels at the middle post. Stand one on edge against the left and middle posts and the other on edge against the right post, their ends overlapping 3 to 6 cm, centred on the middle post. Lap them in the direction of traffic, which runs the way the white arrows point: at the splice, the end of the panel that traffic reaches first must lie on the road side of (nearer you than) the other panel's end. Then let go of both panels and lift both grippers clear." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/guardrail-lap-with-traffic/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Splice the two grey guardrail panels at the middle post. Stand one on edge against the left and middle posts and the other on edge against the right post, their ends overlapping 3 to 6 cm, centred on the middle post. Lap them in the direction of traffic, which runs the way the white arrows point: at the splice, the end of the panel that traffic reaches first must lie on the road side of (nearer you than) the other panel's end. Then let go of both panels and lift both grippers clear." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/guardrail-lap-with-traffic/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: `score`, `tier`, `D`, `U`, `overlap_mm`, `mid_mm`,
`lap_gap_min_mm`, `lap_gap_max_mm` (D's front edge minus U's back edge over the readings), `wrong_gap_min_mm`,
`wrong_gap_max_mm` (U's front edge minus D's back edge), `wrong_reach_y_mm` (how far left U's end must reach for a
wrong lap), `right_post_gap_mm`, `u_turn_deg`, `lapped`, `wrong_lap`, `splice_ok`, `released`,
`left_finger_clear_mm`, `right_finger_clear_mm`, `arms_unavailable`, and for each panel `standing`, `still`,
`moved_mm`, `turned_deg`, `tilt_deg`, `turn_deg`, `low_mm`, `gap_left_post_mm`, `gap_middle_post_mm`,
`right_end_y_mm`, `left_end_y_mm`, `in_downstream_place`, plus both panels' poses and both tool points and finger
openings. Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it
over at every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the end state, as an inspector
reads a splice from a top photo, a tape and a feeler gauge. The scorer reads the two panels' poses and the two
grippers' tool points, finger-tip points and openings once per control step (20 Hz) and keeps each panel's last
half second of poses, to tell a panel at rest from one still moving; it does not use the simulator's velocities,
which chatter while a body is gripped or resting.

A panel's outline is its box seen from above (its real, turned and tilted shape); at a given y its front edge is the
outline's smallest x there (toward the road) and its back edge the largest. A panel is *standing* when its 45 mm
axis is within 8 degrees of vertical (either long edge down), its lowest point within 3 mm of the table, and over
the last 0.5 s its centre has stayed within 2 mm of where it is and it has turned less than 2 degrees. The
*downstream place* is a standing panel whose back edge, at y +24.0 cm (the left post) and at y 0 (the middle post),
is 0-8 mm in front of the posts' faces (x 31.0 cm), or up to 3 mm past them (contact: at the posts' own y a panel
cannot really be behind their faces), with its right end (the smallest y of its outline) at y 0 to -5.5 cm: it
reaches at least the middle post's centre and runs at most 5.5 cm past it. That panel is D and the other U: the
panels are interchangeable. The *shared stretch* runs from D's right end to U's left end; its length is the overlap
(negative if they do not meet). It is read every 2 mm from 5 mm inside each end, so that the end face of a turned
panel is never read as its long face. *Lapped on the road side*: U standing, its length within 15 degrees of the y
axis, an overlap of 10-100 mm, and at every reading D's front edge lies from 1 mm in front of to 15 mm behind U's
back edge (the lap gap, -1 to +15 mm: U's end in front of D's). A *wrong lap* (U's end tucked behind D's): U's left
end at most 2 mm short of D's right end or of the middle post's right side (y -1.25 cm), whichever is farther right
(with D's end short of that side, the post is all U's end can tuck in against), U's lowest point within 5 mm of the
table (standing, leaning or held there), and U's front edge from 1 mm in front of to 15 mm behind D's back edge: at
every reading of a stretch 10 mm or longer, otherwise once, U's front edge 5 mm inside its left end against D's back
edge 5 mm inside its right end. Behind means farther from the road. *Released*: each gripper's fingers at least
10 mm from both panels' boxes. rigstate gives no gripper heading, so the fingers are taken to be anywhere round the
gripper's axis: a solid cylinder from the point between the finger tips 38 mm back up the gripper, of radius
hypot(half the finger gap + 10 mm, 20 mm), out to a finger's outer corner (60.4 mm wide open, 27.6 mm closed on
a panel's 18 mm). A finger left touching a panel is never missed, while a gripper lifted straight up off a panel is clear once
its finger tips are 10 mm above the panel's top. (The cylinder's surface is sampled every 5 degrees round and every
1 to 2.5 mm, which reads the clearance at most 0.3 mm long.) Unreadable grippers are not released.

| Outcome | Score |
|---|---|
| No panel standing in the downstream place (nothing done; a panel lying, leaning, held up or still moving; a single panel against the right-hand posts; a panel ending short of the middle post's centre; the upstream panel set first flat across the middle post) | 0 |
| A wrong lap: U's end tucked behind D's end at the splice | 0 |
| U lapped on the road side, overlap 30-60 mm, the stretch's middle within 15 mm of y 0, U's back edge at the right post (y -24.0 cm) 1 mm past to 12 mm in front of its face, both grippers released; held 1 s | 1.0 |
| U lapped on the road side but 1.0 not met (overlap 10-30 or 60-100 mm, off centre, off the right post, a gripper still on a panel or its fingers within 10 mm of one, the grippers unreadable) | 0.5 |
| D in place, U neither lapped on the road side nor a wrong lap (U untouched, butted end to end, short of the splice, more than 15 mm out in front of D, lying, leaning or lifted, doubled over D) | 0.25 |

Possible scores: 0, 0.25, 0.5, 1.0; the rows are checked top-down, so every outcome takes exactly one. A wrong lap
and a road-side lap can never both hold: at any reading the two gaps add up to minus both panels' thicknesses
(-36 mm or less), so they cannot both lie within -1 to +15 mm. The demo's splice, as planned with D flush against the
posts, reads: overlap 45 mm, middle at -2.5 mm, lap gap 2.7-5.6 mm, U 2.5 mm off the right post and turned 4.7
degrees (offline, where D is carried 2 mm into the posts, the lap gap reads 4.7-7.6 mm). On the GPU it read: D 4.4 mm
off the left post and 1.0 mm off the middle post, overlap 44.2 mm, middle at -2.9 mm, lap gap 5.8-8.8 mm, U 2.2 mm
off the right post and turned 5.8 degrees. The half-score example (`tests`): U parallel to D,
7 mm in front of it and 25 mm off the right post. Two conditions follow from the others and so are not checked
separately: U's centre lies right of the middle post (an overlap of at most 100 mm with D's right end at y 0 or less
puts it at y -4.7 cm or less), and a U that meets the 1.0 lap and right-post readings is turned at most about 10
degrees (a search found 10.1 with D standing 8 mm off the posts, 7.6 with D flush).

Unintended strategies earn what the text says, as the tests show: the upstream panel tucked behind the downstream
one's end (lapped against traffic) is 0, held, leaning or standing, down to a 2 mm gap between the ends, and so is
U's end jammed against the middle post's side behind D's line when D's end stops short of that side; putting the
upstream panel up first, flat against the right and middle posts, leaves the other no place against the middle post
(0); holding a panel in place instead of letting go, opening the fingers without lifting, or leaving an open finger
against a panel is 0.5 at most; a panel held up in the air, lying flat in front of the splice, stood on top of D,
butted end to end, set 20 mm out in the lane or laid wholly in front of D is not lapped (0.25); a D that stops short
of the middle post's centre is not in place (0); knocking D over after the splice drops the score to 0. A splice
right in every other way scores 1.0 wherever D's end lies from the middle post's centre to 45 mm right of it (the
farthest a 60 mm overlap centred within 15 mm allows), on the post's face or past its right side. A
panel cannot be past the posts' faces at the posts' own y without passing through them, so the 3 mm allowance there
gives nothing away. A simulator state with a non-finite number, a zero quaternion or a missing panel scores 0 for
that step and leaves the scorer's record as it was. A new trial is recognised from rigstate's reset hook or, as a
fallback, when the simulator puts every panel back on its start (within 4.5 cm, which covers a seeded reset's
jitter) with at least one of them jumping more than 8 cm in one step.

## One arm alone

Each arm reaches only the panel on its own side, so a full splice needs both: the left arm sets D, the right arm sets U.
Straight down at grip height, at any heading, the left arm reaches no farther right than y -18 cm at x 16 cm, -14 cm
at x 25 cm and -11 cm at x 30 cm (inverse kinematics of the arm's URDF with joint limits and a collision proxy); the
right panel's near face is at y -39.1 cm, 21 cm beyond. A tilted grip adds 10-16 cm, which still leaves the tool
point at least 5 cm short of that face: the open far finger might touch it, but cannot grip or pull it. The right
arm is the mirror image. The demo with one arm idle, predicted offline (kinematic check) and measured on the GPU (one
trial each, see "Results in simulation"):

| Run | Predicted | GPU |
|---|---|---|
| `DEMO_IDLE_ARM=right` (left arm alone: D in place, the right panel untouched) | 0.25 | 0.25 |
| `DEMO_IDLE_ARM=left` (right arm alone: U at its place, no D) | 0 | 0 |

The right arm alone could instead set its own panel in the downstream place (a straight-down grip at its centre,
x 30.3 cm, y +9.5 to +10 cm, is reachable): 0.25, the most one arm can earn by reaching. A single arm could only get
the far panel by raking it within reach with the near one held by an end (about 28 cm more reach), which the scorer
does not forbid; dragged on edge, the far panel would most likely fall flat and then have to be stood back on edge.
We have not tried it.

## Demo and run limit

The scripted demo takes 33.8 s (676 control steps). Each arm comes down over its own panel with the fingers open
(the last 28 mm over 1.5 s), grips it across its 18 mm thickness at yaw 0 (fingers commanded 8 mm narrower, over 1.5
s; tool point 32 mm above the table, pad tips 19 mm), lifts it to 12 cm and carries it, turning it a quarter turn on
the way, with its bottom 88 mm up, 28 mm over the posts' tops. The left arm sets D first: over its place at x 29 cm,
y +12.5 cm (yaw -90 degrees), down to 5 cm, 11 mm in front of the posts and below their tops, then over 2 s back to
x 30.3 cm and down to 40 mm (the panel's bottom 8 mm up, about 5 mm with the arm's sag), which drives D's back face
2 mm past the posts' faces so the compliant arm presses it flush against both, then straight down over 1.5 s to
33.5 mm, so the panel never moves sideways with its bottom on the table. It lets go 1.5 mm above the table (the arm
sags about 3 mm under the panel), goes straight up to 10 cm and parks. The right arm meanwhile waits with U at x
22 cm, y -13 cm, 13 cm up (yaw +94.7 degrees: turned with the panel), 7 cm in front of its place, until the left
hand is up at 23.2 s; then it comes down to x 28.4 cm and 5 cm (U passes below D's top only when it is about 3 cm in
front of it), over 1.5 s in to x 28.94 cm, y -13.02 cm and 40 mm, and straight down over 1.5 s to 33.5 mm: U's
back-left corner (x 28.6 cm, y +2.0 cm) 6 mm in front of D's front face, its back face 1.5 mm off the right post's
right side. It lets go, goes straight up and parks; once both grippers' fingers are 10 mm clear of the panels (the
right one's finger tips 10 mm over U's top, about half-way up its lift), success counts. The tool points stay at
least 26.5 cm apart and the arms' links 9.5 cm apart (capsule proxy). Offline, success has held for its 1 s at step
641 (32.05 s); on the GPU the trial ended at step 642 (32.1 s) with 1.0 held. The scripted policy stops at step 737. The run limit is 2,400 steps (2 minutes), 3.6 times the demo.

What to watch on the GPU, in `TASK_TRACE`: `arms_unavailable` must stay 0 (a trial with an unreadable step is not
valid, and cannot score 1.0); the standing battens (18 x 45 mm, tipping at 21.8 degrees) in a `DEMO_HOLD_ONLY=1`
settle run; D pressed into the posts (`gap_left_post_mm`, `gap_middle_post_mm` near 0, `right_end_y_mm` -25,
`turn_deg` near 0: the batten must not turn in the grip); each panel's `turn_deg` and `tilt_deg` as it comes
straight down the last 6.5 mm (they must not change as it touches the table; if a panel catches the table on the
way in, raise `SLIDE_Z`); the U place, where the lap gap reads 2.7-5.6 mm with D flush, 2.3-6.0 mm at the stretch's
very ends (`lap_gap_min_mm`, `lap_gap_max_mm`, `overlap_mm`, `right_post_gap_mm`, `u_turn_deg`): if the pads sit
8 mm toward the posts, U presses on D and the right post (contact, -1 mm, is allowed); 8 mm away, the lap gap
reaches 13.6 mm (limit 15) and the post gap 10.5 mm (limit 12), so tune `U_X` from the first trace. The fingers open
fully before lifting (34.5 mm clear of each face), and `released` should turn true during the second lift
(`right_finger_clear_mm` 10 or more).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: both panels back on edge at their starts,
lengths pointing away from the robot, centred at x 31.0 cm and y +40.0 and -40.0 cm. The posts, road, edge line
and arrows are fixed. Nothing is consumed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running for the scripted demo (34 s of simulated time), and about
5 minutes for a Dream-1 trial run to the 2,400-step limit; the first start of Isaac Sim on a new instance takes
several minutes more.

## Real-world build

Three 25 x 25 x 60 mm blocks of pine or MDF painted dark brown, taped to the bench in a row 240 mm apart with their
front faces on one line; a 130 x 700 mm board (hardboard) painted dark grey in front of them as the road, with a
6 mm strip of white tape as the edge line and three 60 mm white tape arrows (10 mm tape) pointing to the robot's
left; and two 300 x 18 x 45 mm pine battens (about 0.12 kg each) painted light grey as the panels. Safe and
unpowered. Score a trial from a top photo of the splice (which end is nearer the road), a tape (the overlap along
the barrier and D's end past the middle post) and a feeler gauge or ruler (the gaps at the posts, the lap gap, and
that both grippers end at least 1 cm clear of both panels).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |

The demo ended early with 1.0 held for 1 s (step 642). Dream-1 ran to the 2,400-step limit without setting either
panel in place: it nudged the right panel 7 mm and left both standing at their starts. A settle run (no plan, 160
steps) left both panels where they started. Both grippers were read on every step of every run
(`arms_unavailable` 0).
