# Float-mount a print on a hidden riser, centred on the backing board

What is measured is a set-back. With the print laid on, a 10 mm card strip must slide its full width under every
edge of the print without meeting the riser (the riser is then at least 10 mm in from every edge, so it cannot be
seen), and the print must sit centred and square on the backing board. As the riser lies on the table, its 70 mm
length runs across the way the print will lie, and it is exactly as long as the print is wide: laid as it lies, its
ends come flush with the print's long edges and show. It passes the card-strip test only if the right arm turns it a
quarter turn on the way and lays it lengthwise. So the score turns on a measured margin that the robot can only
reach by re-orienting the support, not on one part being stacked on or centred on another (the print is centred on
the fixed board, not on the riser; the riser has 10 mm of play either way).

This is how picture framers float-mount prints, especially ones with deckled or signed edges: the print is raised
off the backing board on a hidden riser (a stack of mat board or foam board) so that all four of its edges show and
cast a shadow. The riser, cut smaller than the print, is set back from every edge by at least its own height (10 mm
here), or it shows when the frame is seen from the side; and the print is centred on the backing with even margins.
The order (riser first, then print) is not scored as a rule: the end state forces it, since a print laid on the bare
board cannot float and a riser laid on the print is not on the board. Here a white backing board is fixed on the
midline, a black 70 x 30 x 10 mm riser lies on the table on the right and a cream 110 x 70 x 10 mm print on the left.
The scorer checks the end state the way a framer does: a 10 mm card strip slid under each edge (the inset), a steel
rule for the margins and the float height, and a square for the turn.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain NumPy functions of the parts' and the arms' states, with a small tracker |
| `rigstate.py` | Reads both arms' TCP, finger tips and opening for the scorer (wraps the embodiment's step and reset; changes nothing it does) |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero condition, every stated bound, the finger zone, the stillness window, the reset, the exploits we tried and the demo's timing (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (one fixed and two loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Float-mount the print on the white backing board. First lay the black riser flat on the board, then lay the print face up on top of the riser, centred on the board with its long side along the board's long side. The riser must end at least 1 cm in from every edge of the print, so that it cannot be seen. Then let go and move both grippers away." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/float-mount-hidden-riser/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Float-mount the print on the white backing board. First lay the black riser flat on the board, then lay the print face up on top of the riser, centred on the board with its long side along the board's long side. The riser must end at least 1 cm in from every edge of the print, so that it cannot be seen. Then let go and move both grippers away." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/float-mount-hidden-riser/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`riser_set`, `riser_low_mm`, `riser_tilt_deg`, `riser_yaw_deg`,
`riser_on_board`, `riser_path_mm`, `floating`, `print_low_mm`, `print_tilt_deg`, `print_path_mm`, `inset_mm`,
`off_centre_mm`, `turned_deg`, `clear`, `left_finger_gap_mm`, `right_finger_gap_mm`, `arms_unavailable`, both parts'
poses and both TCPs). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads both parts' poses once per control step
(20 Hz) and works out every outline from the pose and the parts' known sizes. It keeps only two things from the
past: each part's last 0.5 s of poses, for 'at rest', and when a new episode starts.

- **At rest**: over the last 0.5 s (10 control steps) the part's centre moved along a path shorter than 5 mm and
  it turned less than 3 degrees (summed step by step). The reported velocity is never used: a gripped part's
  velocity chatters at 10-45 mm/s while it stays put, so a velocity test would never call a held part still.
- **Riser set**: flat (its 10 mm axis within 10 degrees of vertical, either broad face down), its lowest point 3 to
  7 mm above the table, every corner of it seen from above inside the board's 180 x 130 mm outline, and at rest.
  Over the board the 3-7 mm band can only be the board's top (5 mm up): anything under the riser (the print) lifts
  it to 15 mm or more, and on the bare table it is at 0.
- **Print floating**: face up (its face normal within 3 degrees of straight up), at rest, its lowest point 13 to
  17 mm above the table (on the riser's top, 15 mm up), and the riser's centre inside its outline seen from above.
  In simulation the stand-in print is the same cream on every face; a top-down grip cannot turn it over, and a real
  print shows its face.
- **Inset**: in the print's own frame, the smallest signed distance from any corner of the riser's outline to any
  edge of the print's outline: positive when every corner is inside, negative when one sticks out. It is taken at
  the riser's corners, its extreme points, so no part of the riser can stick out without a corner doing so.
  Nominal 20 mm (a 70 x 30 riser lengthwise under a 110 x 70 print, both centred), which leaves +-10 mm of play;
  the riser laid across the print gives 0.
- **Off centre**: plan distance from the print's centre to the board's centre (nominal margins 35 mm along x and
  30 mm along y on each side). **Turned**: the angle between the print's 110 mm sides and the board's 180 mm sides,
  folded to 0-90 degrees: half a turn counts as square (the stand-in print has no top or bottom edge), a quarter
  turn is 90.
- **Clear**: each gripper's finger zone is at least 30 mm from the print and from the riser (their real, turned
  outlines). The zone is a flat-ended cylinder about the gripper's axis, from the point between its finger tips
  38 mm back up the gripper, with the radius from that axis to a pad's outer corner at the opening it has, so both
  pads are inside it however the gripper is turned. Its lower end is the tips' plane, because there is nothing of
  the fingers below their tips: the 30 mm is measured from the pads in every direction, so finger tips 30 mm
  straight above the print are clear whether the gripper is open or closed, and an open gripper beside the print
  needs about 87 mm from its axis to the print's side (57 mm to a pad's outer corner plus 30). The arms are read
  through `rigstate.py`; on a step where they cannot be read, 'clear' is taken as met and the step is counted in
  `arms_unavailable` (a policy cannot cause that; the GPU runs check that it stays 0).
- **Bounds**: every bound stated as inclusive ('or more', 'or less', 'within', '3 to 7 mm') is compared with a
  tolerance of 1e-9 (metres or degrees), so a value exactly on it (6.0 mm off centre, turned 3.0 or 8.0 degrees,
  12.0 mm off centre, inset 10.0 mm, 30.0 mm clear) scores the tier the table gives it; bounds stated as strict
  ('shorter than', 'less than', 'more than') stay strict.
- **Off the table**: a loose part's lowest point more than 10 mm below the table top.

| Outcome (checked in this order) | Score |
|---|---|
| A part off the table, or the riser not set: nothing done; riser left on the table, on top of the print, on its edge or end, hanging past the board's edge, or still moving; the scripted demo with the left arm alone | 0 |
| Riser set, print not floating: the scripted demo with the right arm alone; print on the bare board, held above the riser, tipped with an edge on the board, face down, or so far off that the riser's centre is not under it | 0.25 |
| Riser set, print floating, inset 10 mm or more, off centre 6 mm or less, turned 3 degrees or less, both grippers clear | 1.0 |
| Riser set, print floating, inset 10 mm or more, off centre 12 mm or less, turned 8 degrees or less, but not 1.0 (for example 9 mm off centre, turned 5 degrees, a gripper still holding a part or its pads within 30 mm of one) | 0.75 |
| Riser set, print floating, anything else: the riser laid as it lies, across the print (its ends flush with the print's long edges: it shows); the print turned a quarter turn to cover an unturned riser; more than 12 mm off centre; turned more than 8 degrees; inset under 10 mm | 0.5 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0. Every outcome falls in exactly one row. The trial ends with success once
the 1.0 conditions have held for 1 s (`SUCCESS_HOLD_S = 1.0`); otherwise `task_score` is read from the last step at
the run limit with the same rules (`SCORE_OVER = "end"`, no hold needed). There is no whole-trial rule.

Unintended strategies earn less, as the tests show: the riser laid on top of the print, or the print laid on a riser
that sits on the table, scores 0 (riser not set); a riser stood on its end under the print scores 0 (not flat); the
print held still at the right height by the gripper scores 0.75 at most (1.0 needs the grippers clear), while
letting go, lifting the open gripper straight up until its tips are 3 cm above the print's face and stopping there
scores 1.0 (the written rule); turning the print instead of the riser, so that it covers a riser laid as it lies,
scores 0.5 (turned), and shifting the print sideways cannot hide that riser either (its 70 mm length equals the
print's width: 0.5); a print pushed half onto the riser so that it tips scores 0.25. Pushing, dragging or dropping a
part into place ends in a true end state and is scored on its merits: nothing in the scene can be broken or
modified, and the board is fixed. A simulator state with a non-finite number or a zero quaternion scores 0 for that
step and leaves the tracker as it was. A new episode is recognised from rigstate's reset hook, and as a fallback
when both parts jump back onto their starting places (within 4.5 cm, which covers a seeded reset's jitter) in a
single step.

## One arm alone

The two parts start on opposite sides of the table, each beyond the other arm's reach with a straight-down grip:
at x 24 cm the right arm's straight-down reach near the table ends at about y +15 cm and the left arm's at about
y -15 cm, and the print's nearer edge is at y +32.5 cm, the riser's at y -32.5 cm. Measured in simulation on
the GPU (the scripted demo with one arm left at its ready pose, one trial each), the same as `check_task.py
--one-arm` (kinematic proxy) predicted:

- right arm alone (`DEMO_IDLE_ARM=left`): **0.25**, the riser set on the board (lengthwise, flat, 5 mm up) and
  the print left where it started on the table;
- left arm alone (`DEMO_IDLE_ARM=right`): **0**, the riser cannot be reached and stays where it started, so the
  print could only go on the bare board (the demo's left arm lays it there, 0.4 mm off centre), which does not
  count as floating, and the riser is not set: 0.

With straight-down grips both arms are needed, and only because of where the parts start. Each arm does one pick and place, and the right arm also
turns its part a quarter turn on the way; the arms meet at the board in turn, not together. Our kinematics show that
a gripper tilted 40 degrees or more toward the other side can bring its finger tips to the other part's nearer edge,
but not grip it there: the print can be gripped only across its 70 mm (its 110 mm is wider than the 94 mm open
gripper), and with the gripper tilted 40 degrees about x its fingers, closing along y 70 mm apart, sit 45 mm apart
in height, so the far one would have to go below the table; the riser, gripped across its 30 mm with the fingers
closing along x, stays out of the left arm's reach even tilted (its tips reach y -31 cm at most that way, the
riser's nearer end is at -32.5 cm). An arm could still try to press on the far part and drag it within reach, and
then do the whole task alone; we make no claim that the task cannot be done with one arm, only that it cannot be
done with straight-down grips. The left arm could also set the print down in the shared zone for the right arm to
finish, which is real two-arm cooperation.

## Demo and run limit

Both arms run on one clock; the scripted demo takes 30.7 s (614 control steps). Finger targets are 8 mm narrower
than the part (riser 22 mm across its 30 mm, print 62 mm across its 70 mm), every close and release takes 1.5 s
after a 0.5 s settle, both parts are gripped with the pad tips 2 mm above their bottom (8 mm of the 10 mm side
held), carries are 10 cm above the table, and each place ends with the last 2 cm lowered over 2 s.

- Right arm (riser): over the riser with the fingers closing along x (yaw +pi/2, across its 30 mm; 2.8 s), down,
  grip, lift, then carry it over the board's centre while turning a quarter turn to yaw 0 (3.5 s), so the riser's
  70 mm length now runs along x; down to 2 cm above the board, the last 2 cm slowly, let go with its underside
  1 mm above the board, straight up, and park at x 24 cm, y -30 cm, 12 cm up (parked at t 20.0 s). Yaw -pi/2 at
  the board is out of the right arm's reach (joint 6), so it picks at +pi/2 and turns to 0. Joint 6 also makes up
  for joint 1's swing, so both of these moves are slowed to keep it under 1.2 rad/s.
- Left arm (print), at the same time on its own side: over the print with the fingers closing along y (yaw 0,
  across its 70 mm), down, grip, lift, and hold it at a wait point (x 22 cm, y +24 cm, 10 cm up) until the right
  arm has parked; then carry it over the board's centre, down to 2 cm above the riser, the last 2 cm slowly, let go
  with its underside 1 mm above the riser's top (the pads are then 39 mm either side of the centre line, outside
  the riser's 15 mm), straight up, and park at x 24 cm, y +30 cm, 12 cm up.

Offline (`tools/offline/check_task.py --require-json --strict --one-arm`): worst IK error 0.088 mm / 0.014 deg;
smallest joint margins 17.1 deg (right joint 6) and 17.4 deg (both joints 4); peak joint speed 1.15 rad/s (right
joint 6, during the quarter turn); arm-to-arm capsule clearance 7.9 cm and TCPs at least 25.3 cm apart (both at
t 11.9 s, the right arm coming down over the board while the left waits); lowest pad tips 2 mm above the table.
The kinematic proxy drops the riser 1.0 mm onto the board and the print 1.0 mm onto the riser and predicts 1.0:
the left gripper is clear once its tips are 30 mm above the print's face on the way straight up (t 28.2 s), and
success (the 1.0 conditions held for 1 s) comes at t 29.2 s (step 584), while the left arm moves to park; the
scripted policy stops at step 675 (our commands pass `--max-steps 1800`; success comes before step 600). A variant
of the demo that leaves the riser unturned predicts 0.5. The run limit is 1,800 steps (90 s): about three times the
demo, enough for a model to make one correction.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the riser back flat on the table centred at
x 24.0 cm, y -36.0 cm with its 70 mm length along y, and the print back face up centred at x 24.0 cm, y +36.0 cm
with its 110 mm sides along x; the backing board is fixed. Nothing is consumed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo is 31 s of simulated time, a full-length
Dream-1 trial 90 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 180 x 130 mm piece of 5 mm white acid-free foam board on a stiff base, taped to the bench at the board's place.
The riser: seven layers of black 4-ply (1.4 mm) mat board glued into a 70 x 30 x 10 mm block (about 15 g at 0.7
g/cm3). The print: any print dry-mounted on a 110 x 70 mm piece of 10 mm gatorboard, face up (about 22 g: 77 cm3 of
gatorboard at about 0.25 g/cm3, plus the print and the dry-mount tissue). Scoring a real trial, before touching
anything: slide a 10 mm wide card strip under each edge of the print along its whole length (it must go in its full
width everywhere without meeting the riser: inset 10 mm or more); read the four margins with a steel rule (off
centre combines half the difference of the front and back margins with half the difference of the left and right
ones) and the turn with a small square against the board's long edge; read the float height at an edge with a rule
(the print's underside 15 mm up, 10 mm above the board). In a real trial the grippers count as clear only when they
are parked away from the parts: each gripper's centre line about 9 cm or more from the parts' sides, or its finger
tips 3 cm or more straight above the print's face (the simulated zone holds both pads however the gripper is turned,
so beside a part it is stricter than a ruler on the real pads would be). Then lift the print off and check that the
riser lies flat with its whole outline on the board.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

The scripted demo ended in success at step 585 (the 1.0 conditions held for 1 s): the riser lay flat on the board about 1 degree off lengthwise, set back 18.2 mm from the print's nearest edge; the print was 0.4 mm off centre, turned 0.1 degree, its underside 15.0 mm up. Its score went 0, then 0.25 once the riser was set, 0.75 while the left gripper still held the print on the riser, and 1.0 once that gripper's pads were 30 mm or more from both parts on its way up. The settle run (no plan) moved neither part. Dream-1 ran the full 1,800 steps and left both parts where they started: 0.
