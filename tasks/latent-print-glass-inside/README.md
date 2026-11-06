# Move a fingerprinted glass into an evidence box, touching only its inside

A drinking glass found at a scene may carry latent fingerprints on its outside. Crime-scene investigators and
evidence technicians lift it without touching that surface: gloved fingers spread inside it, never gripped round its
sides; it is set on a clean examination sheet to be documented, then moved on so that nothing touches its outside.
Here a pale blue square acrylic glass (74 x 74 mm, 80 mm tall, a 62 x 62 mm opening) stands on a yellow start mark
in front of the left arm. The left gripper must carry it to a white exam sheet on the midline and the right gripper
from there into a brown evidence box in front of the right arm, each gripping it only from the inside: the fingers go
in through the mouth nearly closed, then open until their outer faces press the inner walls (an expanding internal
grasp). The scorer watches the whole trial: no finger may be on the glass's outside or within 1 cm of it below the
rim, its outside may touch nothing (not the box's walls), it may be set down only on its mark, the sheet or the box
floor, and it may not be tipped or dropped. The arms work in turn by reach: only the left arm reaches the glass's
start, only the right arm the box, and the exam sheet is the one place both reach. The task is the handling for latent
prints only: the box is the transport box the glass is set into; fixing it there (a dowel or ties) and swabbing the
lip for DNA (which fingers inside it would touch) are later steps and out of scope.

How it differs from tasks already on the board: unplugging a cable by its plug body (TK-KJ48WA) is a where-to-grip
rule on a cable, checked only while pulling; squeezing a handle-less box between both grippers (TK-2D287M) is an
unusual grasp, but an external one with no contact rule; changing a dressing with one clean and one dirty hand
(TK-ZL98EE) sets contact rules by arm, not by surface of one item; carrying cord-linked devices (TK-J7Q44Q) forbids
pulling a cord. None asks for an internal grasp, and none keeps one surface of one object untouched for a whole
trial. The board has no forensic, fingerprint or evidence-handling task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the finger measurements the grip is sized from |
| `scoring.py` | The scorer: plain functions of the glass's state and the grippers' state, with a tracker for the whole trial |
| `mechkit.py` | Builds the glass (a floor and four walls) as one loose rigid body and reports its state |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_model.py` | Offline checks only: where an internal grip would carry the glass (the simulator never calls it; see below) |
| `tests/test_scoring.py` | Offline tests of every tier, every rule and its boundary, resets, the hooks and the arm readings, the offline model, the exploits we and an adversarial review tried, a randomised check of the finger rule against the real finger blocks, and the whole demo through our offline checker (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `mechkit.py` and `rigstate.py` sit next
to `task.py` and hook `yam_arena` from the task file's import (the glass is built as one compound rigid body, since a
native box cannot be open at the top; both arms are read before each scoring call); they change nothing the
embodiment does, and the task uses the standard `task_score` scorer. `rigstate` takes its snapshot of the arms
before the embodiment steps the physics, one control step older than the objects the hooks are given, so `task.py`
reads both arms again at the same moment as the glass, with `rigstate`'s own reader on the env `mechkit` keeps (both
files unchanged). If that read fails it says why once on stderr and uses `rigstate`'s snapshot, carried one step on
with the glass for a gripper inside it; if `rigstate` cannot hook the embodiment at all it says so at import. On a step
where the arms cannot be read at all (it says so once) the finger and drop rules are not applied and the trace counts
the step (`arms_unavailable`); if that is more than 1% of the steps the trial scores 0: the finger rule could not be
judged.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The glass is evidence with fingerprints on its outside, so touch it only on the inside: put a gripper's fingers into it nearly closed, then open them against its inner walls. With the left gripper move it from its yellow mark onto the white exam sheet, then with the right gripper into the brown evidence box, standing on the box floor, and let go. Never let its outside touch anything, and set it down only on its mark, the sheet or the box floor." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/latent-print-glass-inside/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The glass is evidence with fingerprints on its outside, so touch it only on the inside: put a gripper's fingers into it nearly closed, then open them against its inner walls. With the left gripper move it from its yellow mark onto the white exam sheet, then with the right gripper into the brown evidence box, standing on the box floor, and let go. Never let its outside touch anything, and set it down only on its mark, the sheet or the box floor." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/latent-print-glass-inside/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module and starts a new one
when the embodiment resets (through `rigstate`'s and `mechkit`'s reset hooks, or, failing both, when the glass jumps
back onto its start mark in one step, further than its velocity allows). The run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`breaches`, `first_breach_step`, `in_box`, `on_sheet`, `examined`,
`lifted`, `tilt_deg`, `low_mm`, `box_gap_mm`, `min_box_gap_mm`, `max_fall_mps`, `fingers` per gripper (`above`,
`away`, `inside` or `outside`), `holding`, `let_go`, `opening`, `arms_read` (`after_step`, `before_step` or none),
`arms_unavailable`, `arms_blind`, the glass's position and velocity, both tool points and finger tips, ...);
`MECHKIT_TRACE=mechkit.jsonl` the glass's pose as mechkit reads it.
Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about 2 minutes; the last tests run our
offline checker on the folder when it is present).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Shapes are judged in the glass's own frame (its real box,
turned or tilted as it is) and, for outlines, seen from above.

The finger rule needs the fingers' real size. `yam_arena`'s finger collision meshes (as measured for the program's
earlier finger-zone tasks) are 6.4 mm thick outside the gap between the pads and 27 mm wide at their tips, and 10 mm
thick and 40 mm wide 19 to 38 mm above them; the finger carriages are 70 mm above the tips. So each finger is a block
10 mm thick (outside the gap) and 40 mm wide, along the gripper's axis as it is tilted (the scorer reads the tool point
and the finger-tip point, so it knows the axis). It cannot see the gripper's turn about that axis, so it tries every
turn from 0 to 175 degrees in 5-degree steps (the pair looks the same turned half a turn). In the glass's own frame,
a gripper is fine when:

- no part of its fingers' lowest 38 mm can reach below the rim, at any turn (not judged); or
- at every turn, all of its fingers' lowest 38 mm is at least 10 mm from the glass: outside a box 10 mm bigger than
  the glass on every side and 10 mm under its bottom, up to the rim (tested on the round zone the fingers sweep about
  the axis, so it holds whatever the real turn); or
- both fingers, wherever they are below the rim, are in the opening: within 5 mm of the inner walls (fingers pressing
  them reach them; the walls are 6 mm thick) and above the inner floor, each finger counted up to its carriage, since
  a finger can only be in the glass through its mouth. If the finger-tip point is in the glass, one turn that fits is
  enough (the walls hold fingers that came in through the mouth to such a turn); otherwise the fingers must fit at
  every turn. For a tilted gripper this is also asked of the fingers as far above the rim as they could swing down
  (the zone's radius about the axis times the sine of the tilt), so that no turn hides a finger reaching down over
  the outside.

Otherwise a finger is on the outside of the glass, or within 1 cm of it below the rim: 0 for the trial. The gripper's
body and the arm above the fingers are not judged, nor fingers resting on the rim's top. A randomised test checks the
rule against the real finger blocks at their true turn: wherever a real finger is on the glass's outside or within
1 cm of it below the rim, the rule is broken (over 3,600 such poses, tipped 0 to 90 degrees, any opening, on upright,
tilted and turned glasses: no misses). Legitimate internal grips keep a margin: at the contact gap a gripper tipped up
to 8 degrees in any direction (16 along the glass's sides) or with its tip point 7 mm off the glass's axis still counts
as inside (the wrists sag under 1 degree).

Seven rules hold for the whole trial; broken at any moment they score 0, whatever follows:

| Rule (any moment) | Score |
|---|---|
| (1) A gripper breaks the finger rule: a finger on the glass's outside, or within 1 cm of it below the rim | 0 |
| (2) The glass within 2 mm of a box wall, seen from above, while its lowest point is under 2 mm above the wall's top | 0 |
| (3) The glass tilted more than 10 degrees | 0 |
| (4) Its lowest point under 4 mm above the table top while its outline is not wholly on the start mark, the sheet or the box floor | 0 |
| (5) It moves down faster than 0.45 m/s (a 1 cm fall; over the box's walls 1.04 m/s) without a gripper's finger-tip point inside it moving down with it (within 0.15 m/s) | 0 |
| (6) It leaves the table (lowest point 10 mm under the table top, or its centre beyond the table's edge) | 0 |
| (7) The grippers cannot be read on more than 1% of the steps (on such a step rules 1 and 5 are skipped and 'let go' is taken as met) | 0 |

Otherwise the score is read at the end. 'Standing on' a surface means tilted under 5 degrees, its outline wholly on
it, its lowest point within 3 mm of its top, and still (under 1 cm/s and 0.5 rad/s). A gripper has let go when its
finger-tip point is 20 mm or more above the rim or 60 mm or more from the glass's axis.

| Outcome | Score |
|---|---|
| Standing in the box (on its 105 x 105 mm floor), both grippers let go, and it stood on the exam sheet earlier | 1.0 |
| Standing in the box, but a gripper not let go, or it never stood on the exam sheet | 0.75 |
| Standing on the exam sheet | 0.5 |
| Lifted at some moment (its lowest point 20 mm or more above the table top), none of the above | 0.25 |
| Anything else | 0 |

The trial ends early once the 1.0 conditions have held for 1 s. A simulator state with a non-finite number or a zero
quaternion scores 0 for that step and leaves the record of the trial as it was. The glass is square, so its turn about
its own axis matters nowhere; it has one right way up (upside down is a 180-degree tilt).

Unintended strategies earn nothing or little, as the tests show. Grips on the outside break rule (1) at once: an
ordinary top-down grip across the glass (also when played through our offline checker, whose own grasp proxy does
carry the glass that way); a side grip with the gripper lying (nearly) on its side, its axis just over the rim and its
fingers, now upright, squeezing the top of the outer walls (each arm in turn, sheet and box: 0); pinching the rim with
one finger inside and one outside. So do pushing it with closed fingers or with the back of an open finger (the glass
pushed 15 mm along its mark: 0), and a tipped open gripper whose low finger swings down beside the outer wall while
its tip point is still over the rim. Fingers opened wide before they go in land on the rim and hold nothing.
Dragging it across the table, even when gripped from inside, breaks rule (4), as does setting it on the bare table,
half off the sheet or off its mark; dropping it into the box breaks rule (5), tipping it in breaks rule (3), and
setting it on a box wall's top or brushing a wall breaks rule (2); holding it over the box or hanging inside it, not
standing on the floor, is 0.25; skipping the exam sheet caps it at 0.75. A few millimetres of wrist sag or of pads off
centre (the glass up to 8 mm off the box's centre, the gripper 7 mm off the glass's axis) still score 1.0.

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone) moves the glass to the exam sheet: 0.5. `DEMO_IDLE_ARM=left` (the right
arm alone) comes down over the empty exam sheet and nothing moves: 0. The work is split by reach. Straight down, an
arm's reach near the table ends about 13 cm past the midline at x 26 cm, and the start mark and the box are 38 cm
from it, 7.5 cm outside each arm's own base line. Tipped, an arm reaches further, so we checked that too, with
position-and-orientation IK (yamkin; any tilt direction and any turn about the gripper's axis; within 1 mm and
0.5 degrees; the arm's collision capsules clear of the table and its own base). The left arm's tool point reaches
the height at which it would hold the glass standing on the box floor (51 mm up) only tipped 50 degrees, and then only
at the floor's nearest corner (the glass's centre at x 24.5 cm, y -36.5 cm; at the box's centre not even at 50); to
carry the glass over the box's near wall (108 mm up there) it must be tipped 55 degrees. The right arm and the start
mark are the mirror image. An internal grasp cannot be made with a gripper tipped that far: tipped 50 degrees across
the closing axis the 40 mm fingers no longer fit the 62 mm opening at all, tipped along it they fit only 27 mm deep or
less, and the finger carriages (48 mm out from the gripper's axis, 70 mm above the tips when upright) come down to 8
mm above the tip point, inside the glass's height. So we expect neither arm to be able to finish alone, also for a
policy that tilts its gripper. (Offline kinematic predictions; the GPU runs will be recorded in `task.json`'s
`scores`.)

## Demo and run limit

The scripted demo takes 38.4 s (768 control steps), the arms in turn on one clock, yaw 0 throughout (fingers closing
along y; the glass is square). Every grip: over the glass with the fingers 10 mm apart (16 mm clear of each inner
wall along the closing axis, 11 mm across it), down until the finger tips are 10 mm above the rim, then 52 mm more
over 2 s (tips 42 mm below the rim, 32 mm above the inner floor), settle 0.6 s, open the fingers over 1.5 s to a
commanded 50 mm (their outer faces meet the walls at 42 mm: 4 mm more per finger presses each wall with about 20 N,
against the glass's 1.8 N weight), settle 0.6 s, lift. With the tips 42 mm in, the fingers' 10 mm-thick band (19 to
38 mm above the tips) is 4 to 23 mm inside the glass and only 4 mm of the finger roots above it; the finger carriages
(70 mm above the tips) stay 28 mm above the rim and the gripper's body (83 mm) 41 mm above it. Every set-down: down
until the glass's bottom is 15 mm above the surface, settle, the rest over 1.5 s, settle 0.6 s, close the fingers to
10 mm over 1.5 s, settle, straight up to 14 cm (84 mm at the sheet, 87.5 mm at the box).

The left arm grips the glass on its mark, lifts it until its bottom is 75 mm up, carries it 38 cm to the exam sheet
(2.5 s), sets it down there, lets go, rises out of it and moves back toward its own side. The right arm waits 24 cm
to the right of the sheet meanwhile (the tool points stay 24 cm apart and the arms' collision capsules 6.7 cm), sets
off 0.5 s after the left arm has risen out of the glass, grips it on the sheet, lifts it until its bottom is 85 mm
up (30 mm over the box's walls), carries it 38 cm over the box, lowers it onto the floor in the box's middle (15.5 mm
clear of every wall), lets go and rises. The wrists sag under the glass's weight (yamkin, joints 4 to 6 at
10 N m/rad, the payload not compensated): about 4.7 mm down at the sheet and 2.6 mm at the box, 1.5 to 2 mm
sideways, under 1 degree of tilt; so the glass is commanded to 3 mm over the sheet and 1.5 mm over the box floor and
comes to rest on them before the fingers close (even a 3 mm fall would reach only 0.24 m/s, against rule (5)'s
0.45). Offline, success has held for its 1 s at step 776; the scripted policy stops at step 829. The run limit is
2,400 steps (120 s), about 3 times the demo: two internal grips, two carries and two set-downs, one arm after the
other.

On the GPU, read `low_mm` and `max_fall_mps` in `TASK_TRACE` around each release (the glass should rest on its surface
before the fingers close) and tune only `SET_GAP_SHEET` / `SET_GAP_BOX` in `task.py`; if the glass shifts sideways in
the hand, read `box_gap_mm` (15.5 nominal, rule (2) at 2) and `fingers` (`inside` while gripping). `arms_read` should
say `after_step` on every step; then, during a carry, `left_tips - glass_pos` (or `right_tips`) stays constant, while
a late reading would lag by the glass's velocity times one control step (0.05 s: 11 mm at the demo's peak 0.23 m/s).

## Offline checks

Our NumPy-only checker (`tools/offline/check_task.py`, not part of the submission) has no physics, and its own grasp
proxy carries an object only when the pads close across it, so it never moves the glass for an internal grip. It
hands `task.py`'s hooks the glass as mechkit's assembly at its start, marked `source` "offline", and feeds `rigstate`
the arms from its kinematics. When `task.py` sees that mark, it takes the glass's state (and the arms as the physics
would leave them) from `offline_model.py` instead, and the same scorer runs on it (the checker feeds the arms for
the very step it scores, so no reading is late there). The model: a gripper has entered the glass when its tip point
crosses below the rim with its fingers fitting through the mouth (no allowance into the walls); an entered gripper 25 mm
or more in, opened to the 42 mm contact gap, grips it; the glass then keeps its offset from that gripper's tips,
upright, cannot go below what is under it, and the reported opening stays at 42 mm (the walls stop the fingers); when
the fingers close below 40 mm it lets go and the glass falls freely onto what is under it. An arm that never enters
the glass moves nothing, and if the checker's own proxy carries the glass by its outside the checker's state is used.
It does not model lag, slip, wrist sag, the glass shifting as the fingers open, or contacts of any kind: those are for
the GPU runs. Through it the demo predicts 1.0 and each arm alone 0.5 and 0, and the checker's own proxy moves nothing.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the glass back upright on its yellow start mark,
centred at x 26.0 cm, y +38.0 cm, sides parallel to x and y; the mark, the exam sheet and the box are fixed.

## Runtime per trial

About 1.5 to 2 minutes for the demo on an L40S once Isaac Sim is running (38 s of simulated time), and up to about 3
minutes for a policy that runs to the 2,400-step limit; the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A square glass box of 6 mm clear acrylic (pale blue tinted), 74 x 74 mm outside and 80 mm tall with a 6 mm floor
(about 180 g), glued from five pieces; it does not break like a real drinking glass. A 110 x 110 mm yellow card taped
to the bench as the start mark, a 140 x 140 mm white card as the exam sheet, and a brown cardboard box with 55 mm walls
round a 105 x 105 mm floor, taped down, with a 2 cm scale bar on its side. Before each trial dust the glass's outside
with fingerprint powder or chalk, so that any touch shows. Film from two sides at right angles (one view alone cannot
judge distance along its own line of sight) and from above; judge the finger rule frame by frame from the side videos
(below the rim each finger inside the glass or at least 1 cm from it) and the powder, the other rules from the same
videos, and the end state with a ruler. Fixing the glass in its box and the DNA swab of its lip are not part of
the trial.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
