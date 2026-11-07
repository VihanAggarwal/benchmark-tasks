# Lay three floorboards tight together, leaving a 10-20 mm expansion gap at each wall

Wood and laminate floors are laid with an expansion gap at every wall, commonly 10 to 15 mm and later hidden by the
skirting board, because wood swells with the humidity: a floor laid tight against a wall buckles or lifts when it
swells. The boards themselves go tight together, closed up with a tapping block or a pull bar, so that no joint
stays open. Carpenters (O*NET 47-2031, whose work includes hardwood floors) and floor fitters do both at once: each
new board is set down short of the last and pushed home against it without driving the floor into the far wall, and
each stays centred between the side walls. Here a bay walled on three sides by grey plywood walls 40 mm tall (330 mm
wide, 200 mm deep, open toward the arms) stands on the table, and three loose pine floorboards stained dark brown
(300 x 50 x 16 mm) lie outside it, two in front of the left arm and one in front of the right arm. The boards must
go side by side across the bay, starting along the far wall, tight against each other (every joint 3 mm or less),
with a gap of 10 to 20 mm to both side walls and to the far wall. A 300 mm board in the 330 mm bay is in the window
at both side walls only within 5 mm of centred. Any board left within 3 mm of a wall's inside face (or touching it,
leaning on it or lying on its top) scores 0: that is the defect the gap is there to prevent. The score is what a
floor layer checks with a rule, feeler gauges and a top photo. What is measured is not placing pieces at marks or at
a set spacing, nor only setting them next to each other on a floor (as in household floor-laying activities such as
BEHAVIOR's): it is every joint closed to 3 mm or less (in practice by pushing each board home against the last) while
a gap of a set size stays open at three walls. Both arms are used: each lays the boards in front of it (see "One arm
alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three boards' poses and the grippers' state, with a small tracker of the last half second |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_push.py` | Offline checks only: a small model of a gripper pushing the boards (the simulator never calls it; see below) |
| `tests/test_scoring.py` | Offline tests of every tier and boundary, the wall rule, the exploits we tried, resets, NaN states, the gripper hook, the push model, the demo plan and the offline checker on this folder (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The 1.0 tier needs both grippers clear of the boards at the end (a floor is judged with nothing holding it), and
`yam_arena` does not pass the grippers' state to a task file's hooks. `rigstate.py` (imported by `task.py`) wraps
the embodiment's step and reset once, at import, to take a snapshot of both arms (tool point, finger-tip point,
finger opening, in the objects' frame) before each control step and to start a new trial at each reset. It reads
only state the simulator already has and changes nothing the embodiment does. If it cannot read the arms it prints
why once; the hands-off condition is then taken as met on those steps, and the trace counts them
(`arms_unavailable`).

`offline_push.py` is not used in the simulator. Our offline checker (NumPy kinematics, no physics) carries what a
gripper closes on but cannot push, so `task.py` gives each board's spec an `offline_state` hook (a plain attribute
the simulator never reads) through which the checker gets the boards' poses from a small quasi-static push model:
closed fingers move a board along x out of their way, the arm trails its command by 6 mm per newton of load (the
stiffness estimate below), each board slides on 1.1 N, so a board pushed into its neighbour stops against it unless
the push is hard enough to slide both, and the far wall stops everything.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the three brown floorboards flat in the walled bay, side by side across it, starting along the far wall. Leave a gap of 10 to 20 mm between the boards and each of the three walls, and push the boards tight against each other." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/floorboards-expansion-gap/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the three brown floorboards flat in the walled bay, side by side across it, starting along the far wall. Leave a gap of 10 to 20 mm between the boards and each of the three walls, and push the boards tight against each other." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/floorboards-expansion-gap/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`n_laid`, `laid`, `pinned`, `far_gap_mm`, `joints_mm`,
`hands_off`, `left_zone_mm`, `right_zone_mm`, `still`, `moved_mm`, `first_all_laid_step`, `first_full_step`, and for
each board `laid`, `low_mm`, `tilt_deg`, `axis_err_deg`, `in_bay`, `gap_left_mm`, `gap_right_mm`, `wall_mm`, `x_mm`,
`y_mm`; the boards' poses, both TCPs and openings). Videos: `inspect-robots video <log>.json`. The scorer keeps one
trial's record per process and starts it over at every reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 1.5 minutes: one test plays the
whole demo and both one-arm variants through the offline checker).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three boards' poses and both grippers'
points once per control step (20 Hz) and scores the end state. A board's outline is the shadow of its eight corners
seen from above, turned and tilted as it is; the walls' inside faces are the lines y +16.5 cm and y -16.5 cm (x 16.0
to 36.0 cm) and x 36.0 cm (y -16.5 to +16.5 cm).

| Outcome | Score |
|---|---|
| Any board pinned: its lowest point at most 42 mm above the table and its outline within 3 mm of a wall's inside face or across it | 0 |
| All three boards laid; far gap 10 to 20 mm; both joints 3 mm or less on both measuring lines; both grippers' finger zones 30 mm or more from every board | 1.0 |
| All three laid, but a joint over 3 mm, the far gap outside 10 to 20 mm, or a gripper's finger zone under 30 mm from a board | 0.75 |
| Two boards laid | 0.5 |
| One board laid | 0.25 |
| None laid | 0 |

A board is laid when its broad faces are tilted at most 3 degrees, its lowest point is at most 2 mm above the table,
its length seen from above is within 3 degrees of y, its outline lies inside the bay (x 16.0 to 36.0 cm), and its
end gaps to both side walls are 10 to 20 mm. Joints are read between neighbouring laid boards (in order of distance
from the far wall) on the lines y +14.0 cm and y -14.0 cm, 10 mm in from a centred board's ends, so a board turned
on its neighbour opens one end of the joint and is caught (a 1-degree turn opens it about 4.9 mm at one line). Every
limit is inclusive. A board's two broad faces are alike, so either may be down. The scorer cannot see which way a
gripper is turned, so each gripper's fingers count as a round zone about its axis, out to the fingers' outer corners
(a finger is 10 mm thick and 40 mm wide: 60 mm fully open, 22 mm closed), from the finger tips to 38 mm back up the
gripper. The score is read at the last step; the trial ends early once the 1.0 conditions have held for 1 s with
neither end of any board having moved more than 2 mm over the last 0.5 s. A new trial is recognised from rigstate's
reset hook, or, as a fallback, when the simulator puts a board back in a single step and all three are then on their
starting marks (within 4.5 cm, which covers a seeded reset's jitter). A simulator state with a non-finite number, or
a missing board, scores 0 for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing more than they should, as the tests show: driving the floor against the far wall
scores 0 (pinned), and so does a board pushed against a side wall, a board leaning on a wall or lying on a wall's
top, whether or not it is laid; the floor slid to the open edge (far gap 50 mm) scores 0.75; a floor shifted 6 mm
sideways lays nothing (9 and 21 mm end gaps, 0); a board left on top of the others, standing on its edge or turned
more than 3 degrees is not laid; a board turned 1 degree on its neighbour opens the joint (0.75); a gripper still
pressing a board loses the tier (0.75). A board carried over the walls more than 42 mm up, or one against a wall's
outside face (outside the bay), is not pinned. Only the end state counts: a floor that touched the far wall and was
drawn back to 15 mm scores 1.0, as a floor layer's would.

## One arm alone

With straight-down grips each arm reaches only the boards in front of it. The right board's inner side is at y -21.0
cm; at grip height the left arm's tool point reaches no further right than y -20.0 cm at x 10 cm, -17.0 cm at x 20
cm and -14.5 cm at x 25 cm (inverse kinematics of the arm's URDF with joint limits, every 30 degrees of grip
heading), so it cannot grip the right board across its width (its centre line is at y -23.5 cm); at best its fingers
reach the board's inner side near its near end, from where they could only push it away. The right arm is its mirror
image and cannot reach the left boards. So with one arm idle the other lays only its own side's boards: the left arm
alone two (0.5), the right arm alone one (0.25). That is a limit of where the boards start, not of the act: laying
and pushing boards does not in itself need two hands, and we do not claim it does. A tilted wrist reaching across
has not been tried.

The demo's one-arm runs on the GPU scored as the offline check predicted: `DEMO_IDLE_ARM=right` 0.5 (the left arm
alone laid board a along the far wall, 15.3 mm from it, and pushed board b tight against it, joint 0.0 mm; board c
stayed on its start) and `DEMO_IDLE_ARM=left` 0.25 (the right arm alone laid board c and pushed it to where board b
would be; with no board there it stopped 1.2 mm further on than in the two-arm demo, still laid; boards a and b
stayed on their starts). Neither arm alone reached 1.0.

## Demo and run limit

The scripted demo takes 65.8 s (1,316 control steps). Every grip closes over 1.5 s with the fingers commanded 8 mm
narrower than the 50 mm board, every release opens over 1.5 s, and every set-down slows for its last 20 mm.

- Left arm, board a: picks it where it lies along x, across its width at its middle (yaw 0, fingers 60 mm open, pad
  tips 3 mm above the table), lifts it until its underside is 69 mm up (29 mm over the walls), carries it over the
  bay turning it a quarter turn (yaw -90 degrees) to lie across it, comes down to 2 cm above the floor and lowers it
  over 2 s to 1 mm above, centred at x 31.8 cm, y 0: its far face 17 mm from the far wall. It opens to 56 mm (3 mm
  each side, so the far finger stays 6 mm from the wall) and rises.
- Left arm, board b: the same, set down at x 25.0 cm, 18 mm short of board a (so the far finger, 11 mm past the
  board once open, stays 7 mm from it). It rises 43 mm, closes its fingers in the air, comes down behind board b's
  near face (8 mm clear, pad tips 5 mm above the table) and pushes along x over 3 s, 34 mm in all, the closed fingers'
  face commanded to 8 mm past where board b meets board a. It settles 1 s, backs off 20 mm, rises and parks over its
  own side (44.2 s).
- Right arm: picks board c the same way while the left arm is pushing, holds it 12 cm up over its start until the
  left arm has parked, then carries it over the bay turning it the other way (yaw +90 degrees), sets it down at
  x 19.6 cm, 22 mm short of where board b should be, and pushes it tight against board b the same way (38 mm). It
  backs off, rises and parks over its own side.

The arms trail a push: the joints are position-controlled springs (80/80/80/10/10/10 N m/rad, no payload
compensation), and the compliance model (J K^-1 J^T at the fingers, at the push poses) puts a pushing gripper about
6 mm behind its command per newton along x (5.8 to 6.4). A board slides on about 1.1 N (0.11 kg, friction 1.0; 1.3 N
to start it), so a gripper pushing one board trails about 6.5 mm and pushing two about 13 mm. Each push is commanded
8 mm past the point where the board meets its neighbour: enough for the board to reach it (the push stops when the
spring's pull falls to the board's friction), not enough to slide the neighbour too (twice the friction). The window
is wide: the joint closes to 3 mm or less for any compliance up to about 10 mm/N, and the far gap stays 10 mm or
more for any compliance down to about 2 mm/N (a stiff arm drives the floor on a few mm, which is why board a is set
17 mm from the wall). On the GPU, read `joints_mm`, `far_gap_mm` and each board's `x_mm` in `TASK_TRACE` at the end
of each push and tune `OVERSHOOT` in `task.py` if a joint stays open (larger) or board a moves more than 5 mm
(smaller).

The fingers push 5 to 16 mm up the board's 16 mm face; the compliance model pitches a pushing gripper slightly, tips
trailing, so the fingers bear near the board's top edge, but a 50 mm wide board on friction 1.2 tips only when
pushed more than 20.8 mm above its base, and once it meets its neighbour the neighbour holds its front face.
Offline, success has held for its 1 s at step 1287 (it starts once the right gripper's closed finger zone is 30 mm
clear of board c, during its rise); the scripted policy would stop at step 1377. The run limit is 2,400 steps (2
minutes).

Our offline kinematic check (NumPy, no physics; not part of this folder): worst IK error 0.09 mm, smallest joint-4
margin 16.5 degrees and joint-6 margin 31.3 degrees, tool points at least 23.5 cm apart, the arms' collision
capsules at least 5.7 cm apart (while the left arm backs away from board b and the right arm settles over board c),
pad tips never lower than 3 mm above the table.

Things to look at in the GPU runs: in the settle run nothing moves; board a's `far_gap_mm` near 17 after it is let
go and every board's `gap_left_mm` and `gap_right_mm` near 15 (a sideways offset of an arm under the 0.11 kg load
would show here; the window is +-5 mm); `joints_mm` [0-3] after the first push and [0-3, 0-3] after the second;
board a's `x_mm` moving less than 5 mm during the pushes; each board's `tilt_deg` under 1 (a board riding up on its
neighbour would show); `axis_err_deg` under 1; `arms_unavailable` 0 (otherwise hands-off silently passes: do not
submit).

What the GPU demo run showed: in the settle run nothing moved. Board a, let go, lay 15.3 mm from the far wall (1.7 mm
nearer than planned) with end gaps of 14.1 and 15.3 mm. Board b was set down 18.5 mm short of board a and pushed
home (joint 0.0 mm) while board a moved 0.1 mm; board c was set down 23.3 mm short of board b and pushed home (joint
0.0 mm), with no board moving further. At the end every board was flat (tilt 0.0 degrees), turned 0.76 degrees from
y, 14.0 to 15.3 mm from each side wall, the far gap 15.3 mm, `arms_unavailable` 0. The 1.0 conditions held from step
1269 (once the right gripper had risen clear) and the trial ended at step 1288.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the three boards back flat on the table with
their length along x, centred at x 25.0 cm and y +23.5 cm (board a), x 25.0 cm and y +31.0 cm (board b) and x 25.0
cm and y -23.5 cm (board c). The walls are fixed. Nothing in the start layout lies where the arms' start pose hides
the table from the top camera.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is about 65 s of simulated time until success
ends it; a full Dream-1 run is 2 minutes); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

Three lengths of 20 mm plywood ripped 40 mm tall and painted grey, screwed or clamped to the table: two 200 mm long
with their inside faces 330 mm apart, and one 370 mm long across their far ends. Three pine boards cut 300 x 50 x 16
mm (about 0.11 kg each) with square edges, lightly sanded and stained dark brown. Mark the lines y +14.0 cm and y
-14.0 cm on the far wall's top and on a card at the open edge for the joint readings. To score: feeler gauges (or a
steel rule) for each joint on both lines and for the gaps at both ends of every board and at the far wall, and a top
photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The settle run (no plan, 160 steps) left every board on its start (0, nothing laid). Dream-1 ran the full 2,400
steps: it picked up board b and left it lying on top of board a outside the bay; no board was laid (0).
