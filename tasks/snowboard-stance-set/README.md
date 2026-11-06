# Set a rental snowboard's bindings to a 40 cm duck stance at 15 degrees

Before a rental snowboard goes out, the technician sets its stance: both bindings on the board's centre line, the
stance width from the shop chart (here 40 cm, a child's width on a 110 cm board) centred on the board's reference
mark, and the binding angles for a beginner's duck stance, the front toes turned 15 degrees toward the nose and the
back toes 15 degrees toward the tail. Here a fixed child's board top lies across the table, nose to the left, heel
edge toward the robot: white strips with a black centre line, a small red cross at its middle and two grey insert
grids marked 20 cm either side of it. The two bindings lie parked flat on the board's two ends, toes toward the
ends. Each arm takes the binding on its own side by its ankle-strap bar, lifts it, carries it inward while turning it
75 degrees, sets it down over its grid and lifts clear. The score is the end state: each binding flat with its disc
over an insert grid and centred on the line, turned to its angle within one 3-degree step, still and let go with the
grippers lifted clear, and the two discs 40 cm apart and centred on the cross, as a technician checks with a tape and
the disc's angle scale before screwing down.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two bindings' poses and the grippers' finger zones, with a 0.5 s pose window for stillness |
| `mechkit.py` | Builds each binding as one loose compound rigid body (five shapes) and reads its pose |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier and boundary, the rank rule, resets, bad states, the offline proxy, clearances and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install.

`yam_arena` task files declare single boxes, cylinders and spheres, so `mechkit.py` (imported by `task.py`) adds
the bindings: each is one loose rigid body made of a baseplate, a disc, a toe tick, a highback and an ankle-strap
bar, with the default surface friction. It hooks the scene builder and the object-state reader from the task file's
import; it changes nothing else. The rule that both grippers have let go needs the grippers' state, which
`yam_arena` does not pass to a task file's hooks, so `rigstate.py` wraps the embodiment's step and reset once, at
import, to take a snapshot of both arms before each control step and to start a new trial at each reset. Both read
only state the simulator already has. If the arms cannot be read on a step, no binding counts as let go on it, and
the trace says why (`rigstate_error`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this rental snowboard's bindings in a duck stance. Move each binding from its end of the board and set it down flat over a grey insert grid, highback on the board edge nearest you and disc centred on the black centre line, so the two discs are 40 cm apart and centred on the red cross. Turn each binding so its toe end points 15 degrees off straight across the board: the left binding toward the left end, the right one toward the right end. Then let go of both and lift clear." \
  --scorer task_score --max-steps 1500 --store-frames --log-dir ~/yam-eval/snowboard-stance-set/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set this rental snowboard's bindings in a duck stance. Move each binding from its end of the board and set it down flat over a grey insert grid, highback on the board edge nearest you and disc centred on the black centre line, so the two discs are 40 cm apart and centred on the red cross. Turn each binding so its toe end points 15 degrees off straight across the board: the left binding toward the left end, the right one toward the right end. Then let go of both and lift clear." \
  --scorer task_score --max-steps 1500 --store-frames --log-dir ~/yam-eval/snowboard-stance-set/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check). `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: for each binding its `role` (front or back), `origin`,
`heading_deg`, `angle_err_deg`, `tilt_deg`, `z_above_mm` (its centre above the board's top), `x_off_mm` (off the
centre line), `path_mm` and `turn_deg` (over the last 0.5 s), `tip_mm` (the nearer finger-tip point to any of its
parts), `zone_mm` (the nearer finger zone to any of its parts: 30 mm needed) and its `flat`, `over_grid`, `centred`,
`angled`, `still`, `let_go` and `set` flags; then `n_set`, `W_mm`, `C_mm`, `tier`, both TCPs, both finger-tip points
and both openings. `MECHKIT_TRACE=<file>` adds mechkit's own per-step record.
Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at
every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. A binding's centre is its disc's centre, on the underside
of its baseplate (the frame mechkit reports); its angle is the heading of its heel-to-toe axis seen from above,
from +x (straight across the board), positive toward the nose. A binding is **set** when it is flat (its centre
within 3 mm of the board's top, its base tilted 5 degrees or less; upside down is 180), over an insert grid (disc
centre |y| 140 to 260 mm, either grid), centred (disc centre within 10 mm of the centre line), at its angle (within
3 degrees), still (its centre's path 2 mm or less and its summed turn 1 degree or less over the last 0.5 s, 10
control steps; never judged by the reported velocity) and let go with the grippers lifted clear: each gripper's
finger zone 30 mm or more from every part of it (baseplate, disc as its bounding box, tick, highback, strap bar).
rigstate gives no gripper heading, so the zone takes the fingers to be anywhere round the gripper's axis: it is
every point within the finger radius of the 38 mm of that axis from the point midway between the finger tips back
up toward the wrist, and the radius reaches a pad's outer corner (half the finger gap plus the pad's 8 mm, and
13.5 mm across): 57 mm fully open, 45 mm at a 70 mm gap, 16 mm closed. A gripper straight above a binding
therefore needs its tip point 87, 75 or 46 mm from it, and a finger resting on a binding, open or closed, is not
let go. The binding further left (larger y, nearer the nose) is the front binding, target +15 degrees; the other
the back binding, target -15. They are identical, so which one ends where does not matter. W is the distance along
the board between the two disc centres, C the y of their midpoint.

| Outcome | Score |
|---|---|
| Both set, W 390 to 410 mm, C within 10 mm of the cross | 1.0 |
| Both set, W 380 to 420 mm, C within 20 mm, not 1.0 | 0.75 |
| Both set, W or C further off (for example discs at +-24 cm: W 480 mm; or at +24 and -16 cm: C 40 mm) | 0.5 |
| Exactly one set (for example one placed, the other left where it lay) | 0.25 |
| Neither set (not moved or only turned where they lie, off the line, wrong angle, tilted, off the board, moving, held, touched or not lifted clear) | 0 |

The trial ends early once 1.0 has held for 1 s; otherwise the score is the end state at the run limit. Each
listed score is reached by a state sequence in the tests: one binding set gives 0.25, both carried to the grids'
outer rows (W 480 mm) 0.5, discs 385 mm apart 0.75, the demo's end state 1.0. A boundary value counts as inside
(390.0 and 410.0 mm, 140.0 and 260.0 mm, 3.0 degrees, 10.0 mm, 5.0 degrees, 3.0 mm, 30.0 mm, 2.0 mm).

Unintended strategies earn what the text says, as the tests show: holding both bindings in place at the end, or
resting the fingers on one (closed, or open with one pad on the highback's top or a baseplate's edge, the tip point
37 to 45 mm away), is not let go (0, or 0.25 for the other one alone); opening the grippers where they let go and
stopping, or lifting them 20 mm, or only until the tips are 30 mm over the strap bar, is not lifted clear (0);
turning both in place on the board's ends gives 0, as they are not over the insert grids, and placing one while
turning the other where it lies gives 0.25; a binding dragged into place without lifting is scored by where it ends,
like a lifted one; a binding knocked off the board, lying on the other one, upside down or on its heel is not set;
both toes toward the nose sets only the front one (0.25) and both turned in sets neither (0); discs exactly 400 mm
apart but 15 mm off the cross give 0.75. A binding still creeping (more than 2 mm in 0.5 s) is not set. A new trial is recognised from
rigstate's and mechkit's reset hooks, or, as a fallback, when both bindings are put back on their parked poses
(within 5 mm and 2 degrees) in a single step with one of them jumping 50 mm or turning 20 degrees to get there; a
gradual return or a jump elsewhere is not a reset. A binding state with a non-finite number, a zero quaternion or a
missing field scores 0 for that step and leaves the scorer's window as it was.

## One arm alone

Measured on the GPU (one trial each, as predicted offline): `DEMO_IDLE_ARM=right` (the left arm alone) set the nose
binding (disc 2.5 mm off the centre line, 1.1 degrees off its angle) and left the tail binding untouched where it was
parked: 0.25. `DEMO_IDLE_ARM=left` (the right arm alone) mirrored it, setting the tail binding: 0.25. With both arms
holding still (`DEMO_HOLD_ONLY=1`, the settle run) neither binding moved: 0.

With the grips the demo uses, one arm cannot reach the other binding. Straight down, the left arm reaches toward
-y only to y -0.150 (x 0.24, table height), -0.115 (x 0.30) and -0.065 (x 0.36); a grip tilted 15 to 40 degrees adds
0.10 to 0.16 m, so about -0.31 at best. The tail binding's nearest part, its highback, is at y -0.345 to -0.355, and
its strap bar at y -0.430 to -0.460. Carrying its own binding to the far stance spot with a tilted grip (y -0.20
needs about 0.10 m of tilt gain) leaves it nearer the nose than the other binding, so it is then the front binding
at the wrong angle: 0.

Two one-arm routes remain that we cannot rule out, so we state them. First, a joint-space search with the offline
checker's collision model (all links above the table, clear of the arm's own base and of the idle arm at its ready
pose) finds the left arm laid out flat sideways across the table, shoulder turned about 75 degrees, wrist pointing
toward -y, with its fingers 3 to 10 cm above the table reaching y -0.46 at x 0.24 to 0.36: behind the tail binding's
highback (y -0.355) and up to its strap bar. Second, the held binding could serve as a hook if it were tilted toe-down over the
other's 80 mm highback, which faces the middle (a flat-held binding hits that highback). Either way one arm would
push or pull the far binding about 25 cm toward the middle, then turn it 75 degrees and centre it within 10 mm and
3 degrees without gripping it, at full stretch, and could in principle score up to 1.0. We have not tried either
and do not claim the task needs two arms; the demo, like a technician, sets one binding with each hand.

## Demo and run limit

The scripted demo takes 21.4 s (428 control steps); both arms run the same timeline, mirrored. Each comes over its
own binding's strap bar with the fingers 70 mm apart (20 mm from each face of the 30 mm bar, 47 mm from the
highback), lowers the last 3 cm over 1.5 s to the pad tips 40 mm above the table (12 mm above the disc, 25 mm up
the bar's sides), closes over 1.5 s to 22 mm (8 mm narrower than the bar), lifts it to the gripper 12 cm above the
table (the binding's underside 67 mm above the board), carries it inward over 4 s while turning it 75 degrees (the
left arm's yaw from 0 to -75 degrees, the binding's angle from +90 to +15; the right arm mirrored), comes down to 3 cm
above its spot, lowers the last 3 cm over 2 s to the underside 2 mm above the board, settles 0.8 s, opens back to
70 mm over 1.5 s and goes straight up to 17 cm (finger tips 92 mm above the strap bar, so the finger zone, 45 mm
round the axis at that opening, is 47 mm from the bar: 30 mm needed). The strap bar sits over the disc's centre, so
the TCP goes straight to the disc target, (0.300, +-0.200). The gripper body stays 24 mm from the strap bar's top and
at least 27 mm from the highback (the right arm, whose housing sits to one side of the closing axis, toward its
binding's highback; 45 mm for the left arm); the two TCPs stay at least 40 cm apart. Offline, success has held for
its 1 s at step 438 (the zone clears the bar 1.5 s into the final 2 s rise); the scripted policy stops at step 489.
On the GPU the trial ended early at step 439 with 1.0 (W 402.6 mm, C -0.3 mm). The run limit is 1,500 steps
(75 s), 3.5 times the demo.

The offline checker sees each binding only as its strap column (the bar's 30 x 130 mm footprint from the
underside to the bar's top, `Binding.offline_box`), so its interference checks do not see the baseplate, the disc or
the highback. `tests/test_scoring.py` (class `Clearances`) repeats those checks with the checker's own pad boxes and
link capsules against every part: the open pads are 20 mm from the bar's faces and 47 mm from the highback, and the
gripper body 27 mm (right arm) and 45 mm (left arm) from the highback, at the pick and at the place. The checker's warning that the dropped column
lands on an insert dot with its centre not over it is an artifact of that proxy: the real 200 x 120 mm baseplate
covers six of the eight dots of its grid and rests on them, 0.5 mm above the board.

Demo knobs, in `task.py` only: `PLACE_TRIM` (per arm dx, dy, dyaw) to cancel the pads' offset from the commanded
point (about 8 mm along the closing axis, which at the place is 15 degrees off x: about 7.7 mm in x) and the arms'
trail under the 0.25 kg load (3 to 7 mm), and `GRIP_TIP`. Both trims are zero: on the GPU the untrimmed demo left
each disc 2.2 to 2.5 mm off the centre line, within 2 mm of its y target and 1.1 degrees off its angle, well inside
the 10 mm and 3 degree tolerances. If a rebuild drifts, read `x_off_mm`, `origin` and `angle_err_deg` in
`TASK_TRACE` at the end and set dx to minus `x_off`, dy to minus the y error and dyaw to minus `angle_err`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: each binding back on its parked spot on its
own end of the board, toe toward that end and 5 mm from it, disc on the centre line (x 30.0 cm, y +-44.5 cm). The
board, the cross and the insert dots are fixed. Nothing is consumed.

## Runtime per trial

On an L40S a whole run, Isaac Sim's start included, took about 1.3 minutes for the scripted demo and about 3.5
minutes for a full-length Dream-1 trial (1,500 steps; a trial is at most 75 s of simulated time, the demo about 22 s);
the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 1100 x 240 x 10 mm plywood or HDPE plank painted white, clamped to the table, with a 4 mm black vinyl tape centre
line, a red vinyl tape cross (two 30 x 4 mm strips) at its middle and 8 mm grey vinyl dot stickers at the 16 grid
points. A real
110 cm child's board can be used if it is shimmed flat. Two binding mock-ups, 3D-printed in PLA or glued up from
plywood to the listed parts and sizes and weighted to 0.25 kg each, with a white tick on the disc. Real child's
bindings (about 0.5 kg, straps zip-tied up) are an option within the YAM's payload, but they change the mass. Score
with a tape measure, a protractor against the board's edge (or an overhead photo) and a 3 mm feeler gauge. The strap
bar covers each disc's centre, so take the centre midway between the disc rim's heel-most and toe-most points (or
mark the baseplate's sides at the disc's centre); flat means the baseplate rests on the board and the 3 mm feeler
gauge does not pass under the middle of either long side of it. Nothing is powered, sharp or hot.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |

The Dream-1 trial ran to the 1,500-step limit with neither binding set: the nose binding was not moved, and the tail
binding was pushed about 2.5 cm toward the middle without being turned (still at -90 degrees).
