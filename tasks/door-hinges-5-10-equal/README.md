# Set a steel door's hinges 5 in below the head, 10 in above the floor, middle centred

Carpenters and door hardware installers (O*NET 47-2031.00) hang hollow metal (steel) doors with their hinges at
fixed locations, called "5, 10 and equal": the top hinge 5 in from the frame's head rabbet (the underside of the
frame head) to the top of the hinge, the bottom hinge 10 in from the finished floor to the bottom of the hinge, and
any intermediate hinge centred between them. Allegion's Steelcraft knowledge base gives this as the NAAMM/HMMA
hollow metal hinge location, and Ives, the hinge maker, gives the same three locations in its hang-the-door guide
("Top hinge 5" from jamb rabbet to top of barrel", "Bottom hinge 10" from bottom edge of barrel to finished floor",
"Third hinge centered between top and bottom hinges"). Steel door makers' own stock locations (SDI 129 tables,
measured to hinge centrelines) differ by maker, which is why the instruction names NAAMM/HMMA.

Here a steel door stands on edge on the table in front of both arms, hinge edge up, along y, its top end (the head)
at +y. Lengths along the door are to scale, 1 in = 5 mm: the 80 in door is 400 mm, a 4-1/2 in hinge is 22.5 mm and
the 3/4 in undercut is 3.75 mm. A red HEAD line on the table at the door's top end stands for the frame's head
rabbet; a blue FLOOR line 3.75 mm past its bottom end stands for the finished floor. Three brass hinge leaves lie on
a parts tray in front of the door. The robot has to set them on the hinge edge at the three locations, flat,
lengthwise and centred across the edge, and let go. Nothing on the door is marked: the two end locations are
measured from different references (the head and the floor, not the door's own ends) by different amounts, so a
symmetric or evenly spaced layout is wrong at one end or both, and the middle location follows from wherever the top
and bottom hinges actually are. The instruction names the standard but not its numbers. The score is read from
where the leaves lie at the end, as a ruler along the edge would read it.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the rule's arithmetic |
| `scoring.py` | The scorer: plain functions of the three leaves' states (and the arms' pad-tip points and openings, for 'released'), with a small tracker for stillness, release and the trace |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does); the same file as in our other tasks |
| `tests/test_scoring.py` | Offline tests of every tier and window edge, every zero condition, the release rule, stillness, broken states, resets, the exploits we tried, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer; `rigstate.py`, in
this folder, reads the arms only to check that each leaf was let go.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This steel door stands on edge, hinge edge up. The red HEAD line is the frame's head rabbet, where the top of the door hangs; the blue FLOOR line is the finished floor below its bottom end. Lengths along the door are to scale: 1 in = 5 mm. Set the three hinge leaves on the hinge edge at the NAAMM/HMMA hinge locations for a hollow metal door, each lying flat, lengthwise along the edge and centred across it, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/door-hinges-5-10-equal/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This steel door stands on edge, hinge edge up. The red HEAD line is the frame's head rabbet, where the top of the door hangs; the blue FLOOR line is the finished floor below its bottom end. Lengths along the door are to scale: 1 in = 5 mm. Set the three hinge leaves on the hinge edge at the NAAMM/HMMA hinge locations for a hollow metal door, each lying flat, lengthwise along the edge and centred across it, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/door-hinges-5-10-equal/dream
```

Run one trial per process (the default). The score depends only on where the leaves are and whether a gripper has
held one in the last 0.5 s; the small tracker in the task module keeps each leaf's centre, heading and held flag
over the last 1 s (for 'settled', 'still for 1 s' and 'released') and diagnostics for the trace, and starts a new
episode when rigstate sees the embodiment reset or the simulator puts every leaf back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`tier`, `k`, `on_edge`, `top_ok`, `middle_ok`,
`bottom_ok`, `top_leaf`/`middle_leaf`/`bottom_leaf`, `y_m_mm`, `top_err_mm`, `bottom_err_mm`, `middle_err_mm`,
`all_still_1s`, the first step each score was reached, `arms_readable`, `arms_unavailable` or `rigstate_error`,
each arm's `left_tips`/`right_tips` and `left_opening`/`right_opening`; per leaf in `leaves` its `released`,
`held_now`, `settled`, `still_1s`, `spread_mm` and `turn_deg`, and in `hinge_leaf_*_pose` its centre, `tilt_deg`,
`along_err_deg`, `off_centreline_mm`, `bottom_above_edge_mm`, `resting_on_edge` and `held`). Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three leaves' poses once per control
step (20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the
finger tips) and how far it is open, and scores the state at hand: the score is read at the last step. Positions
along the door are leaf centres' y, as a ruler laid along the edge from a line's door-side edge reads them (a leaf's
centre is its top end minus 11.25 mm).

A leaf is on the edge when all of these hold:

| Condition | Threshold |
|---|---|
| Released | in a real trial, no finger has touched it in the last 0.5 s; in simulation, no gripper's pad-tip point within the fingers' reach of the leaf's centre (over every yaw: hypot(half the gap + 10 mm, 20 mm) + 13.8 mm + 5 mm, 79 mm fully open, 56.5 mm at the demo's 44 mm opening) and less than 10 mm above its top, at any control step of the last 0.5 s |
| Settled | its centre moved at most 2 mm and its 22.5 mm side turned at most 1 degree over the last 0.5 s (from its path: a gripped part's reported velocity chatters) |
| Flat | its big faces within 5 degrees of level (either face up) |
| Lengthwise | its 22.5 mm side within 5 degrees of the door's length (either way round) |
| Resting on the edge | its lowest point within 3 mm of the edge top (80 mm above the table) |
| Centred across | its centre within 8 mm of the edge's centreline (x 30.0 cm), so all of it is on the 36 mm edge, even turned or tilted 5 degrees |
| On the door | no part of it past either end of the door (centre within 188.75 mm of the door's middle) |

Locations, each within 5 mm (1 in) of the leaf's centre:

| Location | Rule | Centre (y) |
|---|---|---|
| Top | its top 25 mm (5 in) below the HEAD line's door-side edge (y +200 mm) | +163.75 mm |
| Bottom | its bottom 50 mm (10 in) above the FLOOR line's door-side edge (y -203.75 mm) | -142.5 mm |
| Middle | centred between the top and bottom leaves | their midpoint as placed (nominally +10.625 mm) |

With three leaves on the edge, the highest (largest y) is the top leaf, the lowest the bottom leaf, and the third is
the middle leaf, judged against the midpoint of the top and bottom leaves as placed, right or wrong: the middle
follows the hinges actually hung. With fewer than three on the edge, each leaf is judged against the nominal
windows (the middle's centred on +10.625 mm); the windows do not overlap, so each leaf counts for at most one location.
k is the number of locations right (0 to 3).

| Tier (first match wins) | Score |
|---|---|
| Three leaves on the edge, k = 3, all three still for 1 s (settled in each of the last two 0.5 s windows) | 1.0 |
| Three leaves on the edge with the top and bottom both right (the middle off centre, or the 1.0 layout not yet still for 1 s) | 0.75 |
| k = 2 otherwise (an end wrong with the other end and the middle right, or only two leaves on the edge) | 0.5 |
| k = 1 | 0.25 |
| Otherwise (k = 0, or no leaf on the edge) | 0 |

Each final state gives one leaf count, one k and one top/bottom result, so every outcome maps to one score (the
tests check every combination). The two end locations carry the standard (different references, different
offsets), so 0.75 needs both ends right; a layout with an end wrong scores at most 0.5 even if its middle is
centred between what it hung. Every tier is reachable, with numbers: the correct layout, 1.0; the middle leaf 10
mm off, 0.75; top and bottom right with the third leaf left on the tray, 0.5; the bottom copied from the top
(symmetric, centre y -163.75 mm, 21.25 mm or 4.25 in too low) with the middle centred, 0.5; the top and the middle
hung by the left arm alone, 0.5; leaves at the quarter points (y +100, 0, -100 mm), 0.25 (only the middle is
centred between them); the bottom alone hung by the right arm, 0.25; nothing moved, 0. No latch is needed: a leaf
slid along the edge to a better place is a legitimate correction. Any leaf may go at any location. The trial ends
early once the 1.0 state has held for 2 s (`SUCCESS_HOLD_S`); otherwise it is scored as it stands at the run limit.
A leaf must be still for 1 s and then hold the 1.0 state for 2 s, longer than the settle and the opening after
the last leaf's slow descent (0.6 + 1.5 s), so even if the arms could not be read the trial cannot end while a leaf
is still in a closed gripper. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for
that step and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: evenly spaced leaves (0.25); a symmetric layout (0.5);
measuring 5 in from the floor as well as from the head (0.5); a leaf crosswise at a right location (counts for
nothing); a leaf still gripped on its spot at the end (counts for nothing: with both end leaves gripped, 0.25);
two leaves stacked at one location (the upper one is not on the edge); a leaf on the table past the door's end or
beside the door (nothing); a leaf overhanging a face of the door, tilted or turned (nothing). Measuring the bottom
from the door's own end instead of the floor puts it 3.75 mm (3/4 in) high, inside the 1 in window: the undercut is
there to make the scene true to the trade, not as a trap, and it is stated in the setup.

## One arm alone

The demo splits the work by end: the left arm hangs the top leaf and then the middle one, the right arm hangs the
bottom leaf. Gripping straight down, at any heading (tried every 22.5 degrees) and at the release height or up to
50 mm above it, the left arm cannot reach the bottom location (x 30.0 cm, y -14.25 cm) and the right arm cannot reach
the top location (y +16.375 cm): at x 30 cm, at the height where a leaf is let go, the left arm reaches no lower than
about y -7 cm and the right arm no higher than about y +7 cm (`tests/test_scoring.py`, `OneArmReach`, our IK model of
the arm). One arm can therefore hang at most its own end and the middle: with only two leaves on the edge the score
is 0.5 at most, and with three on the edge its far end is wrong, which also caps it at 0.5. Offline, the scripted
demo with the right arm idle hangs the top and middle leaves and the third stays on the tray (predicted 0.5); with
the left arm idle it hangs the bottom leaf only (predicted 0.25). GPU results: 'Results in simulation' below. We do
not claim one arm is strictly unable to do more: a policy that pushes a leaf along the edge with another part, or
grips with the wrist strongly tilted, is outside our model.

## Demo and run limit

The scripted demo takes 34.8 s (696 control steps), both arms on one clock. Each leaf is gripped straight down with
the fingers closing along y, across its 22.5 mm length, so the fingers sit off its two ends and, at the door, over
the edge rather than straddling the door's faces: open to 44 mm (10.75 mm off each end) and commanded to 14.5 mm to
close (8 mm narrower than the leaf), with the tool point 16 mm above the leaf's underside so the pads hold 7 of its
10 mm and their tips stay 3 mm above the tray. Every open or close takes 1.5 s, every final descent (onto a leaf and
onto the edge) is slow (20 mm over 1.5 s), each leaf is let go 1.5 mm above the edge after a 0.6 s settle (pad tips
4.5 mm above the edge), and leaves travel with their undersides 24 mm over the edge. Each leaf is laid at a yaw of
-1.15 degrees (gripped parts end 0.6 to 1.7 degrees anticlockwise of the command), and each place point is moved 4 mm
on along its line of travel (parts land about 4 mm short): the aims are within 4 mm of the window centres, with 4 mm
to spare across the edge and 3.5 mm along it either way. In order:

1. The left arm carries leaf A from the tray (x 20.0, y +13.0 cm) to the top location (aim x 30.38, y +16.50 cm);
   at the same time the right arm carries leaf C (x 20.0, y -13.0 cm) to the bottom location (aim x 30.40,
   y -14.30 cm). The grippers stay at least 26 cm apart and the arms' collision capsules at least 6.8 cm (offline).
2. The right arm parks beyond the door's bottom end (x 22, y -25 cm, 12 cm up). The left arm carries leaf B (x 20.0,
   y +5.0 cm) to the nominal middle (aim x 30.37, y +0.92 cm), then parks beyond the door's top end (x 22, y +25
   cm). The middle is judged against where the top and bottom leaves actually lie, so a steady landing error shared
   by the three leaves cancels.

The smallest joint margin is 13.2 degrees (joint 4, left arm, over the middle location at travel height); the
worst IK error 0.1 mm. Offline (kinematic proxy) success is held at step 708 (35.4 s), just after the left arm parks;
the scripted policy would stop at step 757. `PLACE_TRIM` in `task.py` moves a location's place point if the GPU run
shows a leaf landing off by a steady amount. The run limit is 1,200 steps (60 s), 1.7 times the demo's time.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: lay the three leaves back flat on the tray,
their 22.5 mm sides along y, centred at x 20.0 cm and y +13.0, +5.0 and -13.0 cm (any leaf on any spot); the door,
the two lines and the tray are fixed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo ends after about 36 s of simulated time, when success
has held for 2 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

A grey-painted wooden block 400 x 36 x 80 mm screwed to the table on one long face, its centreline 30 cm in front of
the arm bases and centred between them (the door, hinge edge up). A red vinyl strip 80 x 6 mm glued to the table
across its top end, its edge flush with the block's end (the HEAD line), and a blue one across the bottom end with a
3.75 mm gap to the block (the FLOOR line). A dark grey plastic plate 60 x 320 x 3 mm fixed in front of it (the tray).
Three solid brass blocks 22.5 x 16 x 10 mm, about 31 g each (the hinge leaves), cut from bar stock. Positions in
`task.json`'s `setup`.

Scoring a real trial: once the arms have let go and withdrawn, lay a ruler along the edge from the HEAD line's
door-side edge and read each leaf's top and bottom ends (its centre is halfway); check that each lies flat on the
edge, lengthwise, with no part over a face of the door; the top leaf's top end should be 25 mm (20 to 30 mm) below
the HEAD line, the bottom leaf's bottom end 50 mm (45 to 55 mm) above the FLOOR line, the middle leaf's
centre halfway between the other two centres, each within 5 mm.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
