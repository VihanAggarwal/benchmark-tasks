# Set PCB support pins under a pick-and-place board, clear of underside parts

At every product changeover on a surface-mount (SMT) electronics line, the operator sets support pins on the
pick-and-place machine's support plate, under where the next board will stop between the conveyor rails. Spread
one to each region of the board, the pins stop it flexing under the placement head; but they may stand only where
the board's underside has no components, because a pin under an underside part cracks or dislodges it when the
board is clamped, and a pin at a rail fouls the edge clamp. Here a grey support plate stands between two low blue
conveyor rails; on it a white outline of the coming board (120 x 160 mm) is split into quarters, and red areas mark
where the board has parts underneath, leaving one clear spot in each quarter, in a different place each time. Six
steel pins stand in two black trays, three on each side. One pin must end standing upright in each quarter, 5 mm
inside the outline and 3 mm clear of red, let go with the grippers moved away; any pin on red or at a rail
scores 0. Finding the four clear spots among the red areas, and setting each pin down straight inside a window
only a few millimetres larger than the pin, is the task.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place (the red areas, the quarters, the pins, the trays) |
| `scoring.py` | The scorer: plain functions of the six pins' states and the grippers' state, with a small tracker |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, the windows, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, loose cylinders) and the standard `task_score` scorer.

A pin still in a gripper must not count as set, and `yam_arena` does not pass the grippers' state to a task file's
hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a
snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. It reads only state the simulator already has and changes nothing the
embodiment does. If it cannot read the arms it prints why once; every pin is then taken as let go on those steps,
and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the board supports for the next job: stand one support pin upright in each quarter of the white board outline (the thin white lines split it into quarters), the whole pin at least 5 mm inside the outline and at least 3 mm clear of every red area, where the board has parts on its underside that a pin would damage. Keep every pin at least 5 mm from the blue conveyor rails. Take the pins from the black trays, let go of them and move both grippers away from the pins." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/smt-support-pins/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set up the board supports for the next job: stand one support pin upright in each quarter of the white board outline (the thin white lines split it into quarters), the whole pin at least 5 mm inside the outline and at least 3 mm clear of every red area, where the board has parts on its underside that a pin would damage. Keep every pin at least 5 mm from the blue conveyor rails. Take the pins from the black trays, let go of them and move both grippers away from the pins." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/smt-support-pins/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`supported`, `supporters`, `touching_red`, `at_rail`,
`still`, and for each pin `quarter`, `supports`, `upright`, `on_plate`, `released`, `tilt_deg`, `bottom_mm`,
`inside_mm`, `red_mm`, `red_nearest`, `rail_mm`; each gripper's finger-zone gap to each pin, `zones_mm`; the
openings; `arms_unavailable`; the pins' positions and both TCPs). Videos: `inspect-robots video <log>.json`. The
scorer keeps one trial's record per process and starts it over at every reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Once per control step the scorer reads the six pins' poses
and both grippers' state, and decides each pin's place from its real, possibly tilted shape: its footprint is its
shadow seen from above, the convex hull of its two end circles (drawn as 48-sided polygons round the circles, so
never smaller than the pin). A pin supports a quarter when it stands upright (axis within 10 degrees of vertical,
either end down) with its lowest point within 3 mm of the plate's top, let go, its whole footprint at least 5 mm
inside the board's edge (the middle of the white outline line) and at least 3 mm from every red area; it supports
the quarter holding its centre, so a pin over a quarter line counts once, for one quarter. Two rules zero the
score whatever else is right, for any pin, spares included: a footprint overlapping or touching a red area while
the pin's lowest point is within 5 mm of the plate (standing on it, lying across it, leaning over it or held just
above it: that part would be crushed), and a footprint within 5 mm of a rail's while the pin's lowest point is at
most 5 mm above the rail's top (on, against or beside a rail: the edge clamp would hit it). The pins are told
apart by nothing but where they stand: any pin can support any quarter.

| Outcome | Score |
|---|---|
| Any pin touching a red area or at a rail | 0 |
| A supporting pin in each of the four quarters | 1.0 |
| Three quarters supported | 0.75 |
| Two quarters supported | 0.5 |
| One quarter supported | 0.25 |
| Anything else (no quarter supported) | 0 |

The windows: the red areas leave, in each quarter, one patch of centres where a pin counts, a rectangle 14 to 16.5
mm on a side (the pin is 20 mm across), and the tests show by a 0.5 mm grid search that there is no other spot on
the board with the margins. The patches are in a different place in each quarter: at the board's centre corner (far
left), at the board's near-left corner (near left), in the middle of the quarter, ringed by red (near right) and in
the middle of the board's right edge (far right). A gripper's 'finger zone' is every point within a radius of the 38
mm segment running back up the gripper from the point between its finger tips, out to a finger's outer corner (half
the gap plus the 8 mm finger, and 13.5 mm across: 57 mm fully open, 31 mm with the fingers 40 mm apart); the scorer
cannot see which way the gripper is turned, so the zone is round. It is round at the finger tips too: an open
gripper hovering less than that radius over a pin (57 mm fully open) still holds it, which is why the instruction
asks for both grippers to be moved away. A pin either zone reaches is still held and supports nothing. The score is
read at the last step; the trial ends early once 1.0 has held for 1 s with a supporting pin in each quarter still (2
mm or less of travel over the last 0.5 s and under 5 cm/s: grippers' reported velocities chatter, so the path
decides). A new trial is recognised from rigstate's reset hook or, as a fallback, when the simulator puts every pin
back on its tray mark (within 4.5 cm, which covers a seeded reset's jitter) in a single step. A simulator state with
a non-finite number, a zero quaternion or a missing pin scores 0 for that step and leaves the record as it was.

Unintended strategies earn nothing more than they should, as the tests show: a pin left in the gripper, or with an
open gripper hovering beside or over it, does not count (0.75 with the other three set); a pin on its side in a
window, stacked on another pin, standing on the table under the plate's level or outside the outline counts for
nothing; a second pin in a quarter adds nothing; a pin straddling a quarter line counts only for the quarter holding
its centre; a pin leaning 10 degrees over a red area from inside a window scores 0, and so does a spare pin dropped
on a red area or near a rail, even with all four quarters set. Standing a pin upside down is allowed (the pins are
plain posts).

## One arm alone

The demo uses both arms, each setting two pins from its own tray: the left arm near left and far left, the right arm
near right and far right, one gripper over the board at a time. Gripping straight down, each arm reaches all four
windows but only its own tray: the other tray, 23 cm beyond the midline, is out of its reach (inverse kinematics of
the arm's URDF with joint limits; tilted grips were not checked). So, gripping straight down, one arm alone has only
its own three pins and can support at most three quarters, 0.75 (a tilted reach into the other tray, or a pin pushed
over from it, might give it a fourth; neither was tried). In simulation each scripted one-arm run scored 0.5, as the
offline kinematic check predicted, since that arm's plan sets only its own two pins (`DEMO_IDLE_ARM=left`: the right
arm set near right and far right; `DEMO_IDLE_ARM=right`: the left arm set near left and far left; no pin touched red
or came near a rail). The 0.75 ceiling follows only from where the trays stand; no claim is made that the job itself
needs two hands: the difficulty of the task is reading the layout and setting each pin precisely inside its window,
not coordination between the arms. (The full results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 44.9 s (898 control steps); the scripted policy stops at step 959. Each pick: over the pin
in its tray, down to 2 cm above the grip height over 1.5 s, the last 2 cm over 1 s, a 0.5 s settle, the fingers
closed across the pin's 20 mm (commanded 8 mm narrower, over 1.5 s) with the tool point 32 mm above the pin's
bottom, and a straight lift to 12 cm above the table (the carried pin's bottom 40 mm over the tops of the pins in
the trays and 38 mm over those already set on the plate). Each place: over the middle of the quarter's window, down
to 2 cm above the place height, a 0.8 s settle, the last 2 cm over 2 s to where the pin's bottom would be 1 mm above
the plate, the fingers opened to 40 mm over 1.5 s (well clear of anything next to the pin), and straight up. The
fingers close along y throughout, so they never point at a rail or at a neighbouring pin in the tray. The arms take
turns over the board: left near left, right near right, left far left, right far right, each fetching its next pin
while the other is over the board and coming over only once the other has let go and lifted; then both park beside
their trays. The arms are joint springs and the controller does not carry a payload's weight: scaled from the sag
measured in simulation for a 200 g part held the same way (-1.9, +-1.9, -5.6 mm), a 110 g pin sags about 1 mm toward
the arms, 1 mm toward the arm's own side and 3 mm down, well inside the windows (7 to 8.3 mm each way round the
place points), so no trim is applied; the pin touches the plate about 2 mm before the slow descent ends. The tests
check that the place points score 1.0 with that sag and with 4 mm off in any direction. Offline (no sag), success
has held for its 1 s at about step 869. In simulation the four pins stood upright (tilt 0.0 degrees) within 2 mm of
their place points, 8.5 mm or more clear of red and 11 mm or more inside the board's edge, and success held its 1 s
at step 869, which ended the trial early. The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the six pins back upright in the
trays at x 23.0, 19.0 and 15.0 cm, three at y +23.0 cm and three at y -23.0 cm; the plate, rails, lines, red areas
and trays are fixed.

## Runtime per trial

On an L40S once Isaac Sim is running, about 2.5 minutes for the scripted demo (870 control steps) and about 3
minutes for a Dream-1 trial at the full 1,200 steps; the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 124 x 220 mm steel plate (5 mm, or any flat grey plate) between two 6 x 20 mm aluminium bars 240 mm long, 1 mm
outside its long edges; the board outline and quarter lines as 3 mm white PVC strips and the red areas as red PVC
sheet, 1 mm thick, cut to the coordinates in `task.json` and fixed on the plate (a print at 1:1 on film works too:
'on the plate' allows 3 mm); two 140 x 40 mm black trays; and six 20 mm steel rods cut 45 mm long (about 110 g
each), faced square so they stand. A real magnetic-base support pin works too if it is 20 mm across at its widest. A
photo from above with a millimetre scale on the plate gives each footprint against the lines, red areas and rails; a
square held against each pin shows its tilt.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The scripted demo set all four quarters and held success for 1 s at step 869. Each one-arm run set its arm's two
quarters (0.5). Dream-1 ran to the 1,200-step limit without moving any pin from its tray (0.0).
