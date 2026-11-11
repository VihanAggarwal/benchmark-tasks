# Set out an exam room: every desk facing the front, 1.25 m apart centre to centre

School site staff, exams officers and invigilators in the UK set rooms out for written exams to the Joint Council for
Qualifications' Instructions for Conducting Examinations (JCQ ICE): every candidate facing the same direction, and at
least 1.25 m between the centres of neighbouring desks in every direction, so that no candidate can read another's
script. Rooms are normally turned to face the front, where the clock and the exam times are displayed for every
candidate to see; this task asks for that. The same classrooms are used for group or seminar work, so turning a room
round is routine before every mock and exam series.

Here a 1:10 classroom stands on the table: a grey floor and, at its far end, the front wall (cut down to 60 mm so the
camera sees the room) with a whiteboard and the exam clock on it. Four single desks, each with a blue chair fixed
behind it, stand as a seminar left them: two rows facing each other across the middle of the room, each row's two
desks 100 mm (1 m) apart. Every desk has to be turned a quarter turn, the right way round (a quarter turn the other
way leaves its candidate facing the back of the room), and the four spread so no two centres are closer than 125 mm.
The score is what an exams officer checks with a glance and a tape, and what a top photo shows: which way each chair
faces, and the distance between the middles of the desk tops.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four desks' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each desk unit as one rigid body of coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every reason a desk is not set, the exploits we tried, resets, the demo plan, its carry paths and pad clearances, and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (the floor, the wall, the whiteboard and the clock), the standard `task_score` scorer,
and one helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no front or
back. `mechkit.py` (imported by `task.py`) builds each desk unit from eight boxes (the desk top, its two side frames,
the rail, the chair's seat, its two side frames and its back) as one loose rigid body through a hook on
`yam_arena.scene`, and reports its pose (the frame origin on the floor under the middle of the desk top, and its
orientation) to the scorer. It changes nothing else the embodiment does, and it stops the run with an error if a desk
was not built as a compound body. The technique (a `task_file.Box` subclass the loader accepts, a `sys.meta_path`
hook patching `yam_arena.scene` from the task file's import, pxr authoring in a custom spawner) follows the one first
shown in the program repository's `tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms at once, two desks each): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this classroom out for a written exam (1:10 scale). Move and turn the four desks, each with its chair fixed behind it, so that every candidate would face the front of the room (the whiteboard and the clock), each desk square to the front wall within 15 degrees, and no two desks are closer than 125 mm (1.25 m) centre to centre in any direction, measured between the middles of the desk tops. Leave every desk standing upright on the grey floor." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/exam-room-desks-jcq-spacing/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this classroom out for a written exam (1:10 scale). Move and turn the four desks, each with its chair fixed behind it, so that every candidate would face the front of the room (the whiteboard and the clock), each desk square to the front wall within 15 degrees, and no two desks are closer than 125 mm (1.25 m) centre to centre in any direction, measured between the middles of the desk tops. Leave every desk standing upright on the grey floor." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/exam-room-desks-jcq-spacing/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `success`, `zero_reason`, `valid_state`, `credited`
(the desks that count); for each desk `credited`, `why`, `set`, `tilt_deg`, `lowest_mm`, `path_mm`, `origin_mm`,
`centre_mm` (the middle of its top, seen from above), `heading_deg`, `off_front_deg`, `faces_front`, `nearest_mm`,
`nearest_desk`, `spaced`; then the raw `origins`, `quats`, `low_z` and `vels`). Videos: `inspect-robots video
<log>.json`. The scorer keeps one trial's record per process and starts it over at every reset (mechkit's reset
counter, or the simulator putting all four desks back on their marks in one step); the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the desks' poses at the last step.
A desk **faces** from its chair toward its desk, seen from above (the way its candidate would sit); its **centre** is
the middle of its desk top.

Seen from above (x up the page, toward the front wall; y to the left), the start and the demo's places:

```
                    front wall (face x 46.0), whiteboard y -15 .. +15, clock at y +20
      +------------------------------------------------------------------------+
      |  [2] place x 39.0, y +15.0 (^)            [4] place x 39.0, y -15.0 (^) |
      |  [2] start x 36.0, y +20.0 (>)            [4] start x 36.0, y -20.0 (<) |
      |                                                                        |
      |  [1] start x 26.0, y +20.0 (>)            [3] start x 26.0, y -20.0 (<) |
      |  [1] place x 23.0, y +15.0 (^)            [3] place x 23.0, y -15.0 (^) |
      +---------------------------- grey floor, x 15.0 .. 46.0, y -28 .. +28 ---+
         left arm base (y +30.5)                       right arm base (y -30.5)

  (>) faces -y (toward the right of the room), (<) faces +y, (^) faces +x: the front
```

**Set**: tilted 10 degrees or less, its lowest point within 3 mm of the floor, its centre over the grey floor (seen
from above; the edge counts), and still (moved under 5 mm over the last 0.5 s, turns counted at its farthest corner:
positions, not the simulator's velocities, which chatter while a part is gripped). **Faces the front**: it faces
within 15 degrees of square to the front wall (+x, toward the whiteboard and the clock). **Spaced**: its centre is at
least 125 mm (1.25 m), seen from above, from the centre of every other desk, set or not.

| Outcome | Score |
|---|---|
| Each desk that is set, faces the front and is spaced | 0.25 |
| Score: the sum over the four desks | 0, 0.25, 0.5, 0.75, 1.0 |
| No desk set, facing the front and spaced (the start, all sideways, a tight block, all tipped) | 0 |

Examples: the start (all four facing sideways, each row's two desks 100 mm apart) 0; a 2 x 2 grid all facing the
front, 160 mm apart front to back and 300 mm across, 1.0; the same with one desk turned round to face the back, or left
facing sideways, 0.75; two desks facing the front but 100 mm apart (neither counts) and the other two right 0.5; one
starting row's two desks done and the other row untouched 0.5; one desk right and the rest as they started 0.25 (its nearest unmoved neighbour
is 139 mm away); all four facing the front in a tight block 100 mm apart 0; spread out but all facing a side wall 0.
The desks are alike, so any desk may go anywhere, and any layout that keeps the rule scores 1.0 (four in a line
across the room 130 mm apart, a diamond, two rows). The trial ends early once 1.0 has held for 3 s. A simulator state
with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it was.

Why the numbers: 1.25 m centre to centre in every direction and every candidate facing the same way are the JCQ
seating rule, and facing the front (the clock and the exam times) is the usual way rooms meet it; 125 mm is 1.25 m at
1:10. The 15 degree band is this task's reading of "facing the front" (a desk turned
more than that is visibly askew in a row). The rows start 100 mm apart, inside the rule, and the room is large enough
for any 2 x 2 grid with a margin: the demo leaves 35 mm over the rule front to back.

Unintended strategies earn nothing more than they should, as the tests show: spreading the desks without turning them
leaves them facing sideways (0); a desk turned round the wrong way faces the back (not credited); a desk knocked over,
lifted or stacked on another is not set, and it still counts as a neighbour, so tipping a desk over does not clear its
place for the others; a desk pushed off the grey floor earns nothing; a desk held still 2 mm above the floor counts as
set (the scorer sees only the desks; `task.json` fidelity says so).

## One arm alone

Offline (the kinematic check with `--one-arm`), the scripted demo scores 0.5 with either arm idle: the idle arm's two
desks stay where they started, facing sideways (not credited), and the other arm turns and spreads its own two (0.5).
On the GPU (Isaac Lab-Arena, one trial each) it also scored 0.5 with either arm idle: the working arm's two desks
were credited and the idle arm's two were left facing sideways at their starting places (see 'Results in simulation').

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips at tool-point heights
13 to 90 mm, every yaw in 15 degree steps within joint 6's +-120 degrees, over a 5 x 5 grid of points on each desk's
outline where it starts: 8,500 poses): neither arm reaches any point of the other row's desks. At x 23 to 29 cm,
where the other row's desks stand, the left arm reaches no further than y -15.5 cm at table height (-12.5 cm at 90 mm
up), and the nearest point of desk 3 is at y -17.75 cm; the right arm, mirrored, is the same. So by pick and place one
arm alone has two desks and earns at most 0.5. The two arms work at the same time, each on the desks on its own side.
We have not tried a single arm nudging or turning a far desk with an open finger (the open fingers reach 55 mm past the
tool point, which is enough to touch the far row's nearest edges), pushing it, or a tilted grip, and do not claim those
are impossible.

## Demo and run limit

The scripted demo takes 41.8 s (836 control steps), both arms at once. Each arm moves the desk nearer the arms first,
to the place nearer the arms, then the other to the place near the front wall. For each desk the arm goes over it
with the fingers 76 mm open, comes down beside its side frames (8 mm clear each side of the 60 mm desk, 24 mm from the
neighbouring desk's frame), takes the last 30 mm over 1.5 s and grips across the two side frames (commanded 52 mm,
closing over 1.5 s) 10 mm behind the middle of the desk, toward the chair (the centre of mass is 13.7 mm behind the
middle), the tool point 60 mm above the table: the 20 mm pads hold the frames' outer faces and the desk top's edges
from 47 mm up to the top (73 mm), within the frames' 45 mm depth; above the pads, the finger carriages and the
gripper housing (from 57 mm above the tool point) pass 30 mm over the chair backs, the tallest part of any desk. It
lifts the desk 20 mm and carries it over 4 s in two segments, turning the wrist a quarter turn: the left arm turns
desks 1 and 2 anticlockwise (facing -y to facing +x; yaw -90 through -45 to 0 degrees), the right arm turns desks 3
and 4 clockwise (facing +y to +x; yaw +90 through +45 to 0). Then it settles 0.5 s, lowers the last 18 mm over 2 s,
settles, opens to 76 mm with the desk 2 mm above the floor and rises straight up; it turns the wrist back in two
segments on the way to the second desk, 20 mm over the chair backs. Joint 6 stays at least 44.8 degrees from its
limits.

The places: desks 1 and 2 at x 23.0 and 39.0 cm, y +15.0 cm; desks 3 and 4 at x 23.0 and 39.0 cm, y -15.0 cm; all
facing the front. That is 160 mm front to back (35 mm over the rule), 300 mm across and 340 mm on the diagonals; the
chairs of the near desks end 25.5 mm inside the floor's near edge and the far desks' fronts 47.5 mm short of the
wall. On the way, every carried desk stays more than 35 mm from every other desk (39 mm at the closest, desk 1
passing desk 2 early in its turn), and the pads stay more than 20 mm from every other desk (24 mm at the closest) (tested).

Offline (kinematic check): worst inverse-kinematics error 0.09 mm; smallest joint margin 15.2 degrees (joint 4);
arm links at least 11.9 cm apart (capsule proxy), tool points at least 30.0 cm apart; pad tips never lower than 47 mm
above the table. Success has held for its 3 s at step 778; the scripted policy stops at step 897. The run limit is
1,200 steps (1 minute), which leaves a policy room for slower carries, a set-down to regrip, or a second try.

For the GPU run: read each desk's `centre_mm` (aims (230, +-150) and (390, +-150)), `off_front_deg` (aim 0) and
`nearest_mm` (aim 160) in `TASK_TRACE`. The arm sags a few millimetres under each 80 g desk, so a desk touches the
floor slightly before its slow descent ends; the 3 s success hold lets the last grippers open before the trial ends.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the four desk units upright on the grey
floor as they started: desk 1 centred at x 26.0 cm, y +20.0 cm and desk 2 at x 36.0 cm, y +20.0 cm, both facing -y
(chairs toward +y); desk 3 at x 26.0 cm, y -20.0 cm and desk 4 at x 36.0 cm, y -20.0 cm, both facing +y (chairs toward
-y). A desk's centre is the middle of its top. The floor, wall, whiteboard and clock are fixed. Nothing is consumed.

## Runtime per trial

About 1.5 to 2 minutes on an L40S once Isaac Sim is running (the demo is 42 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 310 x 560 mm mid-grey vinyl sheet 1 mm thick laid on the table from x 15.0 to 46.0 cm, y -28.0 to +28.0 cm; an
off-white painted MDF strip 10 x 60 mm in section and 560 mm long glued down along its far edge for the front wall,
with a 300 x 40 mm white card (the whiteboard) and a 30 mm black square with a white face (the clock) on its face. Four
1:10 desk-and-chair models, about 80 g each: a 60 x 45 x 6 mm beech-effect top on two 6 mm grey side panels (66 mm
tall), a steel or wooden rail between the panels' back ends, and a blue chair (seat 36 x 32 mm at 46 mm, back to 86
mm, on two thin side frames) glued to it; 1:10 doll's-house school furniture works if a chair is fixed behind each
desk. Only the front wall is built (cut down to 60 mm, as in the simulation); the sheet's other three edges mark the
room. Mark the middle of each desk top with a pencil dot. Everything is light and blunt, so trials are safe. Score once
nothing moves, from a top photo: a tape between the dots for the spacing, and a protractor against the line of the
front wall for the facing (the chairs show which way each desk faces); a 3 mm feeler gauge must not pass under a set
desk.

Simplifications (`task.json` fidelity): each chair is fixed behind its desk, as in combination desk-and-chair units, so
the unit shows which way its candidate faces (loose chairs are more usual); there is no candidate, paper or bag; only
the front wall is modelled; the scorer sees only the desks, so a desk held still within 3 mm of the floor counts as
set.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
