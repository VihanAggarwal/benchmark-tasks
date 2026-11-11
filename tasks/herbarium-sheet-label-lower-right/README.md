# Mount a plant on a herbarium sheet: roots down, label bottom right, det slip above it

Herbarium technicians and trained volunteers mount pressed plant specimens on standard archival sheets (11.5 x 16.5
in) before the sheets are glued, filed and imaged for collection databases. Where each item goes is set by written
mounting guides. The Australian National Herbarium's guidelines say: "Place the label in the bottom right hand
corner, a few mm from the edges of the sheet"; "Any determinavit/confirmavit slips should go just above the label";
mount the plant "to represent the natural habit of the plant, i.e. roots mostly at the bottom"; "The plant material
should not cover the CANB barcode number or the label"; the barcode goes in the top right corner. The Florida Museum
Herbarium's mounting guide agrees: "Labels are placed in the bottom right corner, unless unfeasible due to the shape
of the specimen"; "Annotation slips with name changes go above or to the left of the label"; roots or the lower stem
at the bottom, flowers toward the top; barcode preferably top right.

Here a sheet at 1:1.5 (194.7 x 279.4 mm) lies portrait on the bench with its foot nearest the arms and its barcode
already stuck on. Beside it lie a pressed specimen (a board with the plant printed on it, roots at one end, flowers
at the other), its collection label and its determinavit slip. The instruction names the items and says to lay them
out "the way a herbarium mounts them", but gives no corner, edge, order or orientation: the robot has to know the
convention, turn the plant so its roots point to the foot (read from the print), keep it clear of the label, the
slip and the barcode, and set the label and the slip a few millimetres in from the right corner's edges. The score is
what a curator reads with a steel rule and a top photo.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the three items' poses and the grippers' state, with a small tracker of the last second |
| `mechkit.py` | Helper that builds each item (with its print) as one rigid body and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and limit, every zero condition, the spec's shortcut checks, the exploits we tried, non-finite states, resets, the task file and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses `yam_arena`'s own fixed
boxes for the sheet, its outline and the barcode, the standard `task_score` scorer, and two helper modules next to
`task.py`.

`yam_arena`'s loose objects are single plain boxes, and a plain box cannot carry a printed plant or printed text.
`mechkit.py` (imported by `task.py`) builds each loose item from its main box and thin printed boxes on its top (the
root cluster, stem, leaves and flowers on the specimen board; the text lines on the label and the slip) as one
loose rigid body, through a hook on `yam_arena.scene`, and reports each item's pose (the middle of its bottom face and
its orientation) to the scorer. It stops the run with an error if an item was not built as a compound body.

The rule includes letting go, and `yam_arena` does not pass the grippers' state to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, the point between the pad tips, finger opening, in the objects' frame) before each control
step and to start a new trial at each reset. If it cannot read the arms it prints why once; the let-go condition is
then taken as met on those steps, and the trace counts them (`arms_unavailable`).

Both helpers read only state the simulator already has and change nothing else the embodiment does. The technique
(a `task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task
file's import, wrapping the embodiment's step and reset) follows the one first shown in the program repository's
`tasks/bearing-press`; both modules are independent implementations.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This is a herbarium mounting sheet (the standard 11.5 x 16.5 in sheet, shown at 1:1.5 scale) lying portrait on the bench. Its foot, the bottom edge, is the edge nearest the arms, and the sheet is read upright from there. The accession barcode is already stuck on. Lay out this pressed specimen, its collection label and its determination slip on the sheet the way a herbarium mounts them, ready for gluing, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/herbarium-sheet-label-lower-right/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This is a herbarium mounting sheet (the standard 11.5 x 16.5 in sheet, shown at 1:1.5 scale) lying portrait on the bench. Its foot, the bottom edge, is the edge nearest the arms, and the sheet is read upright from there. The accession barcode is already stuck on. Lay out this pressed specimen, its collection label and its determination slip on the sheet the way a herbarium mounts them, ready for gluing, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/herbarium-sheet-label-lower-right/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided: `score`, `success`, `correct` (the items that count), and per
item `zero_reason`, `tilt_deg`, `lowest_mm`, `still`, `let_go` and `extent_mm` (footprint min/max x and y); for the
label `square_err_deg`, `upright_err_deg`, `right_edge_in_mm`, `foot_in_mm`; for the slip `upright_err_deg`,
`gap_above_mm`, `align_above_mm`, `gap_left_mm`, `align_left_mm`, `overlaps_label`, `on_sheet`; for the specimen `root_err_deg`, `heading_deg`,
`margin_mm`, `covers`; plus `path_mm` (each item's corner paths in the last two 0.5 s windows), `zone_gap_mm`,
`first_step` (the first step each score was reached), `arms_unavailable`, `bad_steps`, each item's origin, heading,
tilt and speed, both TCPs and openings. Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's
record per process and starts it over at every reset (rigstate's and mechkit's reset counters, and a teleport of
every item back to its starting mark), so run one trial per process or rely on the reset; the run rate is assumed to
be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the three items' poses and both grippers
once per control step (20 Hz). Directions are the sheet's, read from its foot (a reader at the foot looks along +x):
its right edge is the -y edge (y -97.35 mm), "above" is toward the head (+x), "left" toward +y. Each item's footprint
is the outline of its main body seen from above (the specimen's is its 200 x 70 mm board, the label's its 47 x 67 mm
tile); a rule laid along the sheet's edges reads the footprint's extents.

An item counts only when it is laid: let go (each gripper's finger zone at least 20 mm from it), printed face up and
within 5 degrees of flat, lying on the table or sheet (its lowest corner within 3 mm of the table top: an item resting
on another item, propped or held up does not count) and still (none of its four top corners travelled more than 1.5
mm in either of the last two 0.5 s windows: measured on positions, as the simulator's reported velocities chatter
while a part is gripped). rigstate gives no gripper heading, so a gripper's fingers are taken to be anywhere round
its axis: the zone is a solid cylinder round the gripper's axis, flat at the point between the pad tips (nothing of
the fingers reaches below them) and running 38 mm back up the gripper, its radius reaching a pad's outer corner (half
the current finger gap plus the 8 mm pad thickness out, 13.5 mm across; 57 mm when fully open). So a gripper whose
pad tips are 20 mm or more above an item has let go of it, however wide it is open. A reading exactly on a limit counts as inside it (to 1e-9 m), so no rounding decides a tier.

- Label (0.4): square and reading upright from the foot (long side within 5 degrees of the sheet's width, its top
  text line toward the head: the instruction says the sheet is read upright from the foot, so a label turned end for
  end, upside down, does not count); right edge 2-12 mm inside the sheet's right edge and bottom edge 2-12 mm inside
  the foot ("a few mm from the edges": 3-18 mm full size).
- Slip (0.2, judged only when the label scores): square and reading upright, as for the label; wholly on the
  sheet; not overlapping the label; and directly above it (its bottom edge 0-10 mm above the label's top edge, the
  two right edges within 10 mm) or directly left of it (its right edge 0-10 mm left of the label's left edge, the two
  bottom edges within 10 mm). ANBG says just above; the Florida Museum allows above or to the left; both count.
- Specimen (0.4): whole footprint at least 2 mm inside every sheet edge; root end within 15 degrees of pointing
  straight at the foot; footprint overlapping neither the label, the slip nor the barcode sticker.

| Outcome (each item laid as above, at the end) | Score |
|---|---|
| Label, slip and specimen all correct (success once held 1 s; the run then ends) | 1.0 |
| Label and specimen correct; slip wrong, missing, overlapping the label, held or moving | 0.8 |
| Label and slip correct; specimen wrong (roots up or across, off the sheet, on the label, slip or barcode, held, on its edge) | 0.6 |
| Label alone correct, or specimen alone correct (a correct-looking slip with the label wrong earns nothing) | 0.4 |
| Neither label nor specimen correct (nothing moved; the label in another corner and the plant upside down) | 0 |

The wrong layouts a non-curator would pick all score 0 for the label: bottom centre (its right edge about 50 mm
outside the 2-12 mm band), bottom left (about 110 mm outside), top right (its bottom edge over 200 mm outside the
band); the label or the slip turned end for end reads upside down; and the plant laid roots toward the head is 180
degrees off. Unintended strategies earn nothing extra, as the
tests show: the label and slip swapped (only the plant counts, 0.4); the slip stacked on the label (not on the
table: as if missing, 0.8); the plant laid over the label and slip (not on the table, and it covers them: the plant earns nothing, 0.6); the label
laid in the corner on top of the board (not on the table: 0); everything held in place by the grippers, or pressed
there by an open pad (not let go); the board stood on its long edge on the sheet (not flat); items knocked off the
table (0). Arms that cannot be read are counted and cannot add score that the layout does not earn. A simulator state
with a non-finite number, a zero quaternion or no base-frame origin scores 0 for that step and leaves the scorer's
record as it was.

## One arm alone

The demo splits the work by side: the left arm lays the specimen, and the right arm lays the slip and then the label.
Every item starts in front of the arm that lays it: the label at 25.0 cm and the slip at 35.0 cm forward, both
22.0 cm right of the midline; the specimen at 30.0 cm forward, 25.0 cm left. Gripping straight down, the left arm
reaches no further right than about y -16 cm at x 20 cm and -10 cm at x 30 cm (reach maps from our IK model of the
arm, `tools/offline/reach_maps_station.txt`), so neither tile (whose nearest edges are at y -18.7 and -19.0 cm) is in
its reach; the right arm, mirrored, reaches no further left than about y +10 to +13 cm where the board lies, and the
board's nearest edge is at y +15.0 cm. So the offline kinematic check predicts 0.6 with the left arm idle (label and
slip laid, the plant still beside the sheet) and 0.4 with the right arm idle (the plant laid, label and slip still
beside the sheet). We do not claim one arm is strictly unable to do it all: with a tilted wrist, or by pushing an item
toward itself, a policy might reach the other side's items. The scripted demo grips from above only. (GPU results:
'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 45.0 s (900 control steps), both arms on one clock; the scripted policy stops after step
961. Every gripper close or open takes 1.5 s without moving, every final descent onto an item or onto the sheet is
slow (the last 30 mm over 1.5 s), there is a 0.5 s settle before each close and each release, and each item is let go
with its underside 1 mm above the sheet. Pad tips stay 3 mm above an item's underside when gripping it.

1. Left arm, specimen: comes down over the board's middle at yaw -90 degrees (fingers closing along x, across the
   70 mm board), opens to 86 mm, closes to a commanded 62 mm, lifts the board to 104 mm (underside), turns it a
   quarter turn anticlockwise in place over its start (the 200 mm board sweeps a 106 mm radius there, far from the
   right arm), so the roots point to the foot, carries it over the sheet and lays it centred at x 35.5 cm, y +4.5 cm
   (footprint x 25.5-45.5 cm, y +1.0 to +8.0 cm: 50 mm above the foot, 29 mm below the head, 17 mm in from the left
   edge, 33 mm clear of the label, and 50 mm clear of the barcode in y, ending 4 mm short of it along x), lets go,
   goes straight up and moves aside.
2. Right arm, slip: at the same time grips the slip across its 20 mm width (yaw 90, open 36 mm, commanded 12 mm),
   lifts it and holds it up over its start until the left gripper has risen from the plant (19.3 s), then lays it
   6 mm above where the label will go, flush with the label's right edge (x 26.5-28.5 cm, y -9.035 to -3.035 cm).
   The slip goes first so the label's fingers can close along its 67 mm length, on its two short ends: no pad ever
   comes down between the slip and the label (the label's 27 mm wide fingers stay 15 mm short of the slip).
3. Right arm, label: grips it across its 67 mm length (yaw 0, open 81 mm, commanded 59 mm), lays it 7 mm in from the
   sheet's right edge and the foot (the middle of both 2-12 mm bands), lets go, goes up and moves aside.

The grippers stay at least 26 cm apart and the arms' collision capsules at least 10 cm (offline); the smallest joint
margin is 9.7 degrees (joint 4, left arm, at the specimen's place). In our earlier simulator runs, heavier parts
landed about 4 mm short of the place point along the carry (the arm trails under load), light parts like these
(10-60 g) landed 0.3 to 1.8 mm off aim, and gripped parts ended turned 0.6 to 1.7 degrees counter-clockwise of the
commanded heading. So every place point is aimed 1 mm past its target along the carry and every place heading is
1.15 degrees clockwise of square. Offline, with no trailing, the label lands 7.5 mm in from the right edge and 6.2 mm
from the foot, the slip 4.4 mm above it and aligned within 0.2 mm, the plant 1 mm toward the midline of its target.
A landing 4 mm short leaves the label's right edge 3.6 mm in, and one 1.8 mm past leaves it 9.3 mm in, both inside
the 2-12 mm band. `PLACE_TRIM` in `task.py` moves a place point if the GPU run shows a steady offset. Offline,
success has held for its 1 s at step 877 (43.85 s), once the right gripper has risen clear of the label.
The run limit is 1,200 steps (60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the specimen back flat, plant up, centred at
x 30.0 cm, y +25.0 cm, its long axis along y with the roots toward +y; the label centred at x 25.0 cm, y -22.0 cm
and the slip at x 35.0 cm, y -22.0 cm, both printed face up with their long sides along y, on the bare table off the
sheet. The sheet, its outline and the barcode are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 45 s of simulated time, a run
to the limit is 60 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 194.7 x 279.4 mm sheet of white archival card (or a full-size 11.5 x 16.5 in herbarium sheet, scaling every
distance by 1.5) taped portrait to the bench 20.5 cm in front of the arm bases, centred between them, with a strip of
2 mm grey tape round it; a 20 x 50 mm white barcode sticker in its top right corner, 5.4 mm below the head and
7.35 mm in from the right edge. A 200 x 70 x 10 mm basswood board (about 60 g) with a plant printed or drawn on its
top: a dark brown root cluster at one end, green stem and leaves, yellow flowers at the other. A 47 x 67 x 12 mm and
a 20 x 60 x 12 mm pine block (about 20 g and 8 g), white and cream, with a printed label and a printed "Det." slip
glued on top, each reading upright from the foot at the start. Positions in `task.json`'s `setup`. Score a real trial with a steel rule from the sheet's right edge
and foot to the label, from the label to the slip and from every sheet edge to the board, a protractor or set square
on a top photo for the headings, and a check that nothing rests on anything else and both grippers are clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.6 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.4 |
| Dream-1 | 0.0 |
