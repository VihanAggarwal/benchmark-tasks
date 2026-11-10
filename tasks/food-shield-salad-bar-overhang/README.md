# Put the food shield back over the salad bar, its front lip 18-30 cm past the food

Food preparation workers and dining room and cafeteria attendants set up and break down self-service salad bars and
buffets. The FDA Food Code (3-306.11) requires food on display to be protected by counter, service-line or salad-bar
food guards, display cases or other effective means. A portable food shield (sneeze guard) protects the food only if
its glass and front lip stand between the customers' faces and the food: the lip must reach out past the customers'
edge of the pans, but pushed too far, customers can no longer reach the back of the pans under it. Staff pull the
shield back to refill or wipe the pans and must set it back. Here a salad bar is built at 1:4 on the table: five
fixed 1/3-size pans of food in a row along y, their customer-side edge 34 cm in front of the arm bases, and a tray
slide on the customers' side. A loose portable shield (320 mm long, an acrylic top glass on two end panels, a front
lip, a hollow stainless end cap over each panel; 0.25 kg) stands on the staff side, pulled back for refilling and left
30 mm toward the left end of the bar. It must be put back standing on its two end panels astride the row, square,
its lip 45 to 75 mm (18 to 30 cm full size) past the pans' customer-side edge at both ends of the row, and centred
along the bar. The 18 to 30 cm band is the task's stated rule, not a figure quoted from a code.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the shield's pose, with a small tracker of its last half second |
| `mechkit.py` | Helper that builds the shield as one rigid body of six boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, blow-up states, resets, the demo plan and a replay of the carry against the pans (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes (pans, food, tray slide), the standard `task_score` scorer, and one helper module next
to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres, and a food shield is a frame: two end panels
must stand on the counter either side of the pans with the glass and lip passing over them. `mechkit.py` (imported
by `task.py`) builds the shield from six boxes (top glass, two end panels, front lip, two end caps) as one loose
rigid body through a hook on `yam_arena.scene`, and reports its pose (base-frame origin, at the centre of its
footprint on the panels' undersides, and orientation) to the scorer. It changes nothing else the embodiment does,
and it stops the run with an error if the shield was not built as a compound body. The technique (a
`task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms carry the shield together): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the portable food shield back in service over the salad bar: standing on its two end panels, centred along the row of pans, with its front lip (the customers' side, by the tray slide) 18 to 30 cm (45 to 75 mm at this 1:4 scale) beyond the customers' edge of the food pans. Do not rest it on the pans." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/food-shield-salad-bar-overhang/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Put the portable food shield back in service over the salad bar: standing on its two end panels, centred along the row of pans, with its front lip (the customers' side, by the tray slide) 18 to 30 cm (45 to 75 mm at this 1:4 scale) beyond the customers' edge of the food pans. Do not rest it on the pans." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/food-shield-salad-bar-overhang/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `at_rest`, `standing`, `square`, `spans_row`,
`overhang_left_mm`, `overhang_right_mm`, `overhang_credit`, `centred`, `centre_mm`, `turn_deg`, `tilt_deg`,
`panel_low_mm`, `panel_high_mm`, plus the shield's raw origin, quaternion and velocity). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset (mechkit counts resets); the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the shield's pose, once per control
step (20 Hz), and keeps the last half second of where its two panels' feet were, to tell whether it is at rest (from
positions, not the reported velocity, which chatters while a part is gripped). The overhang is read as a tape would
read it at each end of the row of pans (y +11.4 and -11.4 cm): along x, from the pans' customer-side edge (x 34.0 cm)
to the lip's outer face at its mid-height, positive past the pans. Both readings must be in a band, so a shield
turned a few degrees reads differently at the two ends (about 4 mm per degree) and turned more than 7.5 degrees it
cannot have both inside the 30 mm full band. The shield's centre is the point midway between its end panels.

The shield counts only when it is standing on its end panels (every corner of both panels' undersides within 3 mm
of the counter, tilted 5 degrees or less), square (its lip facing the customers within 10 degrees, seen from above),
spanning the row (its lip runs over both ends of the row) and at rest (neither panel's foot moved more than 3 mm in
the last 0.5 s).

| Outcome | Score |
|---|---|
| Not standing on its end panels (a panel on a pan, on the food or on the tray slide; held up; tipped), turned more than 10 degrees (or lip to the staff side), not spanning the row, or not at rest | 0 |
| Both overhang readings 45 to 75 mm, centre within 15 mm of the row's centre along the bar | 1.0 |
| Both readings 45 to 75 mm, centre further off | 0.75 |
| Both readings 25 to 90 mm (not both 45 to 75), centred within 15 mm | 0.5 |
| Both readings 25 to 90 mm (not both 45 to 75), centre further off | 0.25 |
| Any reading under 25 mm (the food not covered from the customers' side, including the start, 130 mm short) or over 90 mm (customers cannot reach the back of the pans; the tray slide stops the lip at 94 mm) | 0 |

Band edges count as inside. The trial ends early once 1.0 has held for 3 s (`SUCCESS_HOLD_S`), otherwise it is
scored as it stands at the run limit. A new trial is recognised from mechkit's reset counter or, as a fallback,
when the shield is put back on its starting mark (within 4.5 cm, standing, square) in a single step. A simulator
state with a non-finite number or a zero quaternion scores 0 for that step and leaves the record of the trial as it
was.

Unintended strategies earn nothing they should not, as the tests show: a shield set with one panel on a pan (tilted
only 4.5 degrees) scores 0; held 5 mm or 30 mm above the counter, 0; pushed hard against the tray slide (94 mm), 0;
propped with its panels on the slide, 0; set the right distance out but turned lip to the staff side, turned along
the bar or slid along the bar past the end of the row, 0; turned 11 degrees with both readings inside 25 to 90 mm, 0;
centred but short, 0 (centring earns only with the overhang). Sliding the shield along the counter instead of
carrying it is a real way of putting it back and is scored like any other. The scorer cannot see the grippers, so a
shield held still, standing in place, counts (see `fidelity`): in the simulated demo it counts while still gripped,
and the 3 s hold ends with both grippers open.

## One arm alone

One arm alone can do this task: in simulation the scripted demo with either arm idle still scores 1.0 (left arm
idle: overhangs 45.1 and 54.6 mm, centre 4.6 mm off, turned 2.4 degrees; right arm idle: 57.1 and 55.3 mm, centre
4.5 mm off, turned 0.4 degrees; see 'Results in simulation'). Carried by one end cap, the light shield still ends
standing square, inside the band and centred. We make no two-arm claim. Carrying the shield level by both end caps is the natural way a person
puts it back, and that is what the scripted demo does, but the task does not need it.

## Demo and run limit

The scripted demo takes 22.1 s (442 control steps). Both arms move together on one clock, mirrored. Each grips the
end cap on its own side with the fingers closing along x across the cap's 30 mm depth (the left arm at yaw -90
degrees, the right at +90; commanded 22 mm, closing over 1.5 s), the tool point 112 mm above the panels' feet (pad
tips 5 mm above the glass, 20 mm of the 25 mm cap between the pads); the two grips are 296 mm apart along the bar.
They lift the shield 30 mm together, carry it forward 190 mm over 5 s and 30 mm to the right at the same time (so it
ends centred), settle, lower the last 15 mm over 2 s to 2 mm above the counter at x 36.0 cm (lip 60 mm past the
pans: 15 mm inside either end of the full band), settle, open together over 1.5 s, rise and move back to the side.
While carried, the shield's end panels pass beside the row of pans (at least 21 mm from any pan) and its lip 70 mm
above the food (a replay of its six boxes along the planned path); set down, the lip is 34 mm short of the tray
slide.

Offline (kinematic check): worst inverse-kinematics error 0.094 mm; smallest joint margin 12.6 degrees (joint 4 of
each arm, carrying over the place, tool point 142 mm up at x 36 cm); arm links at least 10.5 cm apart (capsule
proxy), tool points at least 29.6 cm apart; pad tips never lower than 99 mm above the counter. Success has held for
its 3 s at step 400; the scripted policy stops at step 503. The run limit is 1,200 steps (60 s).

For the GPU run: the arms sag a few millimetres under the 0.25 kg shield (about 1.2 N each), so its panels should
reach the counter before the slow descent ends; the scorer then counts the shield as set 0.5 s later, while it is
still gripped, and the 3 s hold ends after both grippers have opened (2.0 s after the descent ends). Read
`overhang_left_mm`, `overhang_right_mm` (aim 60, band 45 to 75) and `centre_mm` (aim y 0, within 15) in
`TASK_TRACE`; `PLACE_TRIM` in `task.py` moves the place point by minus any offset. `panel_high_mm` should stay under
3 after the set-down. The checker's interference warnings about the pans are against the shield's bounding box (80
x 320 x 119 mm, which overlaps the pans' last 20 mm during the last 23 mm of the set-down); the real panels stand 42 mm
outside the row.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the shield back on its end panels on the
staff side, square, front lip toward the customers, centred at x 17.0 cm, y +3.0 cm (lip at x 21.0 cm). The pans,
food and tray slide are fixed. Nothing is consumed.

## Runtime per trial

About 1 minute on an L40S once Isaac Sim is running (the demo is 22 s of simulated time); the first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

At 1:4: five 81 x 44 mm blocks 22 mm tall (or small stainless containers on a 22 mm riser) with 3 mm coloured
food cards on top, side by side 2 mm apart and taped down, their customer-side edge 34 cm in front of the arm bases;
a 10 x 10 mm rail 320 mm long taped down 94 mm in front of them. The shield: a 320 x 80 x 4 mm clear acrylic sheet
glued on two 80 x 90 x 4 mm acrylic end panels, a 312 x 25 x 4 mm acrylic strip under its front edge, and two
hollow 30 x 24 x 25 mm caps (bent 0.5 mm stainless sheet, about 13 g each, or printed) on top at the ends, flush
with them (about 0.25 kg in all: the acrylic is 228 g; a commercial mini sneeze guard of similar size works with the
bands scaled). Score with a tape across the bar at each end of the row (overhang), a tape along
the bar (centring), a 3 mm feeler gauge under each panel and a top photo for the angle. The shield is light acrylic,
so trials are safe.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |
