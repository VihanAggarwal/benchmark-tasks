# X-ray a femur phantom AP: hip end toward the cathode, centred, R marker in the field

Radiographers position long bones for the anode heel effect: the x-ray beam is noticeably stronger at the tube's
cathode end than at its anode end, so the thick end of a long part goes under the cathode to even out the density
along the image (Bontrager and Lampignano, Textbook of Radiographic Positioning and Related Anatomy). For an AP femur
the hip end goes toward the cathode, the femur lies along the tube's cathode-anode axis, centred to the central ray
inside the collimated light field, and an anatomical side marker (R for a right limb) goes inside the field but off
the anatomy, so it shows on the image without hiding anything. A femur set the wrong way round gives a pale hip and
a dark knee; a missing or wrong marker can mean a repeat. Here a bench mock-up of a radiography room at about 40%
scale stands on the table: an x-ray table with a painted light field and central-ray cross, a cream strip beside it
standing for the tube's housing with its cathode ("-") and anode ("+") ends marked, a right-femur phantom lying
crosswise on a side cart, and a red R and a blue L marker on a tray. The phantom must be turned a quarter turn onto
the table, and only the heel-effect rule says which way; the instruction names the strip's two marks but not which
end of the phantom goes where, nor which marker to use. The score is what a radiographer checks with a ruler and a
look from above.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the phantom's, the markers' and the grippers' state, with a small tracker of the last second and the floor latch |
| `mechkit.py` | Helper that builds the phantom as one rigid body of three boxes, a cylinder and two painted condyles, and reports its pose (see Requirements) |
| `rigstate.py` | Helper that reads both grippers (tool point, point between the pad tips, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier and edge, every zero condition, the shortcuts we tried, unreadable and non-finite states, resets and the demo plan (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses `yam_arena`'s own fixed
and loose boxes, the standard `task_score` scorer, and two helper modules next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a single box has no thick end. `mechkit.py`
(imported by `task.py`) builds the phantom from three boxes (a 24 mm tall knee section, a 36 mm shaft, a 48 mm hip
section), an upright cylinder (the round femoral head capping the hip end) and two thin grey condyle marks, as one
loose rigid body, through a hook on `yam_arena.scene`, and reports its pose (frame origin and orientation) to the
scorer. It stops the run with an error if the phantom was not built as a compound body.

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
# scripted demo (right arm lays the phantom, then the left arm lays the R marker): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "AP right femur. The cream strip along the x-ray table's edge stands for the tube housing; its two ends are marked - and +. Lay the femur phantom on the table for an AP projection, set the way a radiographer positions it for this tube, centred on the black cross and wholly inside the yellow light field. Then put the correct side marker (red R, blue L) inside the light field without covering the bone, and let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/femur-phantom-anode-heel/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "AP right femur. The cream strip along the x-ray table's edge stands for the tube housing; its two ends are marked - and +. Lay the femur phantom on the table for an AP projection, set the way a radiographer positions it for this tube, centred on the black cross and wholly inside the yellow light field. Then put the correct side marker (red R, blue L) inside the light field without covering the bone, and let go of everything." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/femur-phantom-anode-heel/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`tier`, `zero_reason`, `tilt_deg`, `lowest_mm` (the phantom's
lowest point above the field's top), `flat`, `align_err_deg`, `heading_deg`, `hip_to_cathode`, `field_margin_mm`,
`centre_off_mm`, `centre_dist_mm`, `released`, `zone_gap_mm`, `still`, `path_mm`, `turn_deg`, `dropped_at`,
`marker_ok`, `marker_reason`, `r_released`, `r_zone_gap_mm`, `r_still`, `r_path_mm`, `r_turn_deg`, `r_tilt_deg`,
`r_flat`, `r_in_field`, `r_clear_mm`, `l_in_field`, `arms_unavailable`, `bad_steps`, the phantom's origin,
orientation, heading `ph_yaw_deg` and centre offsets `ph_off_x_mm` / `ph_off_y_mm`, both markers' positions, both
TCPs, pad-tip points and openings). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record
per process and starts it over at every reset (rigstate's and mechkit's reset counters, and a teleport of the
phantom back to its starting mark), so run one trial per process or rely on the reset; the run rate is assumed to be
20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the phantom's pose, both markers' poses and
both grippers once per control step (20 Hz). The phantom's frame has its origin at the middle of its bottom face,
half-way along its 180 mm length (its centre), local x from the knee end to the hip end and local z up out of its
anterior (stepped) face. "Flat on the table" is its lowest point within 3 mm of the light field's top with its
anterior face up, tilted under 5 degrees: upside down (posterior face up, a PA set-up), on its side, standing on an
end or propped on a marker it is not. "Aligned" is its length within 10 degrees of the table's long axis (x), which is
the tube's cathode-anode axis. "In the field" is its whole outline seen from above (each box section's eight
corners, and the head's two end circles) inside the 210 x 130 mm light field. "Centred" is its centre within 15 mm of
the cross. The R marker counts when it is let go and still, lying flat on the table (tilted under 10 degrees, either
face up, its lowest point within 3 mm of the field's top), its centre inside the light field and its outline at least
5 mm clear of the phantom's outline seen from above; the L marker's centre must not be inside the field.

Every score needs the phantom let go and still. Let go: each gripper's finger zone at least 5 mm from it. rigstate
gives no gripper heading, so a gripper's fingers are taken to be anywhere round its axis: the zone is everything
within reach of a pad's outer corner (half the current finger gap plus the 8 mm pad thickness out, 13.5 mm across;
57 mm when fully open) around the 38 mm stretch back from the point between the pad tips. Still: its centre's path
under 1.5 mm in each of the last two half seconds and its heading within 2 degrees over the last second, measured on
positions (the simulator's reported velocities chatter while a part is gripped). If any point of the phantom is ever
lower than 5 mm above the station table top (the floor here: below the tops of the cart, the x-ray table and the
tray), the trial scores 0 from then on, whatever happens next. A new trial is recognised at rigstate's or mechkit's
reset, or when the simulator puts the phantom back on its starting mark (within 4.5 cm of it, which covers a seeded
reset's jitter) in a single step. A simulator state with a non-finite number, a zero quaternion or no frame origin
scores 0 for that step and leaves the scorer's record as it was.

| Outcome (phantom let go and still, never on the floor) | Score |
|---|---|
| Flat, aligned, in the field, hip toward the cathode, centred, marker OK (success once held 1 s) | 1.0 |
| Flat, aligned, in the field, hip toward the cathode, centred; marker condition fails (no R marker in the field, R on the bone, under 5 mm from it, tilted, held or moving, or the L marker in the field) | 0.75 |
| Flat, aligned, in the field, hip toward the cathode, centre more than 15 mm from the cross | 0.5 |
| Flat, aligned, in the field, hip toward the anode (any centring, any markers) | 0.25 |
| Not flat on the table, not aligned, not wholly in the field; held or touched; still moving; ever on the floor | 0 |

Every tier is reachable and each end state maps to one score. The 210 mm field leaves the 180 mm phantom only 15 mm
of play along its length, so 0.5 always needs some offset across the table (up to 45 mm). Unintended strategies earn nothing
more than their end state deserves, as the tests show: the phantom turned the other way, centred, with the R marker
placed (0.25); the R marker laid on the bone (0.75); the L marker used instead, or both markers put in the field as a
hedge (0.75); the R marker still in the gripper (0.75); the phantom 30 mm off the cross across the table (0.5); left
crosswise or diagonal on the table (0); laid upside down, on its side or stood on its hip end (0); propped on a
marker or laid on both (0); half off the field (0); held in place by the gripper (0); dropped on the floor and then
placed perfectly (0). Either side of the bone is accepted for the marker; the demo puts it on the lateral side.

## One arm alone

The demo uses both arms because of reach, not because the skill needs two hands: the right arm lays the phantom and
the left arm lays the R marker. Gripping straight down, our kinematics find no left-arm pose at any point of the
phantom on its cart (y -23 to -41 cm; nearest miss about 16 mm) and no right-arm pose at any point of either marker
on the tray (y +28.8 to +35.2 cm; nearest miss about 41 mm), so the offline kinematic check predicts 0.75 with the left arm idle (the phantom is
set right but no marker is placed) and 0 with the right arm idle (the phantom never reaches the table). These are
offline predictions until the GPU runs. A policy might still reach farther with a tilted gripper, or slide a part
within the other arm's reach, so no claim is made that one arm cannot score 1.0. The task's difficulty is reading
which end of the phantom goes toward which end of the tube, turning it the right way, centring it in the field and
choosing and placing the correct marker.

## Demo and run limit

The scripted demo takes 28.7 s (574 control steps). The right arm comes down beside the phantom on the cart over its
centre of mass (16 mm from mid-length toward the hip end, so it hangs level; pad tips 11 mm above the cart), grips it
across its 40 mm width on the 36 mm shaft (fingers commanded 8 mm narrower, over 1.5 s), lifts it, carries it over the
table in 3 s while turning it a quarter turn clockwise (gripper yaw +90 to -1 degree: the hip end swings from +y to
+x, toward the cathode), settles, lowers it slowly over 2 s until its underside is 1 mm above the field, waits
0.5 s, opens over 1.5 s, goes straight up and moves aside. Meanwhile the left arm has gripped the R marker on the tray
(fingers closing along x, clear of the L marker beside it) and lifted it; once the right gripper is back up it
carries the marker beside the bone on the +y side (the right femur's lateral side, with the patient's head toward the
cathode end), lowers it slowly to 1 mm above the field, lets go, goes up and moves aside. The other answer, hip toward
the anode, is the same grip taken at gripper yaw -90 instead of +90 and laid at the same yaw 0: both are well inside
the right arm's reach and joint 6's range (a test checks this), so the turn the arm can make gives nothing away. The
place points lead by 4 mm (phantom) and 3 mm (marker) along the carry, and the phantom is laid 1 degree clockwise,
for the short landings and counter-clockwise grip bias seen in our earlier simulator runs (`PH_TRIM`, `M_TRIM`,
`YAW_TRIM` in `task.py`); the windows are 15 mm and 10 degrees for the phantom and 5 mm clearances for the marker
(about 11 mm at the demo's trimmed aims). Offline the phantom lands centred 4.3 mm off the cross, the marker 11 mm clear of it, and
success holds for its 1 s at step 556 (27.8 s), once the left gripper has risen clear. The run limit is 1,200 steps
(60 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the phantom back on the side cart, lying flat
along y, centred at x 27.0 cm, y -32.0 cm, hip end toward +y and anterior face up; the R marker on the tray at x
25.0 cm, y +30.0 cm and the L marker at x 25.0 cm, y +34.0 cm, lying flat; everything else is fixed. The light field
and the cross are one flat 0.5 mm coat of paint (the field is laid in eight pieces round the cross's three line
pieces, all flush), so the phantom lies on one flat surface.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is about 29 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 260 x 160 x 10 mm dark board as the x-ray table, with a 210 x 130 mm rectangle of pale yellow paint and a 60 mm
black cross (2 mm lines) at its centre; a 260 x 20 x 30 mm cream bar along one long edge with a black "-" plate at
the far end and a red "+" plate at the near end (40 x 20 mm each); a 80 x 220 x 10 mm white board as the cart and a
50 x 100 x 10 mm grey board as the tray. The phantom: a 180 mm long, 40 mm wide bone-white block stepped 24 / 36 /
48 mm tall over 60 / 70 / 30 mm, with a 40 mm round cap centred on the tall end's face (it sticks out 20 mm,
making the 180 mm length) and two grey marks on the low end, 0.22 kg (a foam-cored plastic shell); two plain
24 x 24 x 16 mm plastic blocks, 30 g, one red (the R marker) and one blue (the L marker). Fix
everything but the phantom and the markers to the bench; nothing moves or is powered. Score with a ruler from the
cross to the phantom's centre and a photo from above showing the outlines and both grippers clear.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.75 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
