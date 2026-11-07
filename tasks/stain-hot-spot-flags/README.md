# Flag the darkest spot of each soil stain and one clean background point

In a contaminated-site investigation the field team marks every sample location with a pin flag before sampling
starts. Under a judgmental (biased) sampling design (EPA QA/G-5S; ASTM E1903, Phase II environmental site
assessment), each visibly stained area is sampled at its most impacted spot, the darkest part of the stain, not
its middle, and a background (reference) sample is taken from the same ground well away from any stain, to give
the unaffected level (EPA 540-R-01-003). A flag on a stain's pale edge dilutes the result, and a "background" point
near a stain is not background. The work is done by environmental science and protection technicians and
environmental scientists (O*NET 19-4042, 19-2041).

Here a light grey site sheet covers the middle of the table, with three soil stains on it. Each stain has three
tones: a pale tan halo of two lobes, a larger mid-brown streak, and a small near-black core at one end of the
streak, set off the stain's middle like a drip point and its run-off. Four orange pin flags stand beside the
sheet, two on each side. The robot must stand one flag on the darkest spot (the core) of each stain, and one
wholly on clean grey sheet at least 15 cm from the edge of every stain, pale halo included, then let go. The score
is the end state: where each standing, released flag's base is, measured with a ruler or a top photo. Unlike tasks
that place markers at listed or measured positions, each target here is the most impacted 2-D sub-area of a stain,
read from its tone beside a larger mid-tone streak that acts as a decoy, and the background point is not a marked
spot: it is any clean point that keeps an exclusion distance from every stain.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the flags' states and the grippers' pad-tip points, with a small tracker |
| `rigstate.py` | Reads both arms' pad-tip points and openings each control step (used only for the let-go rule); copied unchanged from our kit |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes and loose cylinders) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mark the sampling points on the grey site sheet: stand one orange flag upright on the darkest spot of each stain, and one flag wholly on clean grey sheet at least 15 cm from the edge of every stain, pale halo included. Then let go." \
  --scorer task_score --max-steps 600 --store-frames --log-dir ~/yam-eval/stain-hot-spot-flags/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Mark the sampling points on the grey site sheet: stand one orange flag upright on the darkest spot of each stain, and one flag wholly on clean grey sheet at least 15 cm from the edge of every stain, pale halo included. Then let go." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/stain-hot-spot-flags/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (per flag: `base_x_mm`, `base_y_mm`, `tilt_deg`,
`bottom_mm`, `path_mm`, `upright`, `on_ground`, `at_rest`, `held`, `tip_xy_mm`, `tip_above_mm`, `planted`, `spot`,
`stain_dist_mm`; and `items`, `reach_mm`, `arms_readable`, `arms_unavailable`, `arms_unavailable_by_side`, the
grippers' pad-tip points and openings). Videos:
`inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. A flag's base is the centre of its lower end (either end
may be down). A flag is **planted** when (a) its axis is within 10 degrees of vertical, (b) its lowest point is
within 5 mm of the table top (on the table or on the 1 to 3 mm sheets; not on another flag, not held up), (c) its
base has moved less than 2 mm along its path over the last 0.5 s (10 control steps; the scorer never reads
velocity, which chatters while a part is gripped), and (d) it is let go and clear: no gripper's pad-tip point is
lower than 10 mm above the flag's top while the flag's axis is within that gripper's finger reach horizontally.
rigstate gives no gripper yaw, so the reach is the widest over every yaw at the gripper's current opening,
hypot(half the finger gap + 10 mm, 20 mm) + 12.5 mm flag radius + 5 mm: 78 mm fully open, 58 mm at the demo's
50 mm opening, 48 mm closed on a flag. A finger touching, pressing or propping a flag from any side therefore counts
as holding it. Each gripper is read on its own: if one cannot be read on a step, (d) skips that gripper but still
checks the other, so a flag the readable gripper holds is still held. Such steps are counted in the trace
(`arms_unavailable`, and per arm in `arms_unavailable_by_side`); a GPU run counts only if its trace shows
`arms_unavailable` = 0.

- **Hot spot**: 0.25 for each stain with a planted flag whose base is inside its 50 x 50 mm core at least 5 mm in
  from all four edges (a 40 x 40 mm target). Each stain counts once.
- **Background**: 0.25, once, for a planted flag whose base is on the site sheet at least 15 mm inside its edges
  (the whole flag on the sheet, its 12.5 mm radius with 2.5 mm to spare) and at least 150 mm from every stain's
  outline, pale halo included (the nearest point of either halo lobe). The only such ground is
  one pocket by the sheet's near edge in the middle, within x 9.5 to 15.5 cm and y -8.4 to +4.6 cm. No point
  between two stains qualifies (the gaps are 45 and 75 mm).

The score is 0.25 times the number of the four items met. One flag can meet at most one item: a core lies inside
its halo, so it is never background, and the cores are far apart.

| Outcome | Score |
|---|---|
| A planted flag on each of the three cores and one in the background pocket | 1.0 |
| Any three of the four items (for example the three cores and no background) | 0.75 |
| Any two (one arm's two flags, for example) | 0.5 |
| Any one (for example the background alone) | 0.25 |
| None: flags at their starts, on a halo or streak off the core target, between stains, off the sheet, lying down, tilted over 10 degrees, on another flag, still held or still moving | 0 |

Nothing is penalised, and how a flag reaches its spot is not scored (a flag dropped or slid onto a core counts
once it stands still there). The trial ends early once all four items have stayed met for 1 s
(`SUCCESS_HOLD_S = 1.0`, `SCORE_OVER = "end"`); otherwise it is scored as it stands at the run limit. A new episode
is recognised when the simulator puts every flag back on its start mark in one step (and when rigstate sees the
embodiment reset). A step on which a flag's state has a non-finite number or a zero quaternion scores 0 and leaves
the tracker as it was.

Unintended strategies the tests cover, and what they earn:

- **Aiming at the middle of each stain.** The core is off-centre: each stain's larger lobe's centre lies 25 mm
  outside its target, the streak's centre 40 to 45 mm, the stain's area centroid 40 to 47 mm and its bounding-box
  centre 50 to 60 mm. Flags at any of these earn nothing for that stain, and so does a flag 20 mm off the larger
  lobe's centre in any direction. The streak is larger than the core and nearer the middle, so "the brown part" is
  not the answer: the darkest tone is.
- **Nudging a flag to earn the background.** The flags start on the bare table, 77.5 mm off the sheet's side
  edges and more than 330 mm from the pocket, so no start position and no short push earns anything.
- **Holding a flag on a spot** (closed, open round it, a finger propping it from the side, or pressing on its top):
  not let go, nothing for that flag. The three targets and the background pocket are at least 110 mm apart, more
  than any gripper's 78 mm reach, so a gripper still low over one spot never blocks a flag on another.
- **Stacking**: a flag stood on another flag is not on the ground; only the bottom one can count.
- **Spraying all four flags over one stain**: that stain counts once (0.25).
- **A flag lying across a core**: not upright, nothing.
- **Sliding or creeping flags**: counted only after 0.5 s with under 2 mm of path; a back-and-forth shiver adds up
  as path.
- Velocity chatter on a still flag is ignored (stillness is read from positions only).

## One arm alone

The flags are kept in two bundles, one at each side of the site, as a field team leaves a bundle of pin flags at
each side of a large site. Each arm can reach only the two flags on its own side: the other arm's base is 0.705 m
away in y. So one arm alone can meet at most two items and score at most 0.5. In simulation the scripted demo
scored 0.5 with either arm idle, as the offline kinematic check predicted: with the right arm idle the left arm
marked S1 and the background; with the left arm idle the right arm marked S2 and S3 (see 'Results in simulation'
below). The task needs two arms only because of where the flags are, not because of the marking itself: the two
arms work in parallel, each on its own side, and need no coordination beyond keeping apart. Both arms reach the
background pocket, and the left arm also reaches the S2 core close to the table.

## Demo and run limit

The scripted demo runs both arms in parallel for 26.4 s (528 control steps), yaw 0 throughout (fingers closing
along y). Each flag takes a 13.2 s cycle: over the flag with the tool point 10 cm above the table (1.8 s); down to
4 cm (1.2 s) and settle (0.4 s); close the fingers to 17 mm, 8 mm narrower than the flag, over 1.5 s; lift to
10 cm (1.2 s) and carry at 10 cm (2.2 s); down to 2 cm above the place height (0.8 s), the last 2 cm slowly
(1.2 s), settle (0.4 s); open to 50 mm over 1.5 s with the flag's bottom 1 mm above the surface; straight up to
10 cm (1.0 s). The left arm takes its far flag (x 29.0 cm) to the S1 core and then its near flag (x 20.0 cm) to
the background point at x 11.5 cm, y +0.5 cm (177 mm from S1, 190 mm from S2, 208 mm from S3, and at least 20 mm
inside the pocket); the right arm takes its far flag to the S2 core and then its near flag to the S3 core. Every
place point is a core's centre, 20 mm inside its target. This order keeps the arms about 8 cm apart (capsule proxy);
marking the background first would bring them within 5 cm. It also keeps every carry clear of the flags already
standing: a carried flag's bottom is 60 mm up, 1 to 3 mm below a standing flag's top, so no carry path crosses a
placed flag. The arms then hold 10 cm above the last two flags.

Offline (kinematic proxy, no physics), success holds for its 1 s at step 541 and the scripted policy stops after
589 steps, so the demo fits `--max-steps 600`; in simulation the trial ended on success at step 542. The run limit for a model is 1,800 steps (90 s).

## Seeing the stains

The station's top camera sees the table at about 1.7 to 2.0 px per cm at 320 x 176, so a 50 mm core is about 9 to
10 px across and each streak about 14 x 8 px. In relative luminance the near-black core is about 9 times darker than
the brown streak beside it, the streak about 3 times darker than the tan halo, and the halo about 1.6 times darker
and clearly browner than the grey sheet; the core is the only near-black area in each stain. The wrist cameras
see the stains much larger on the way down. In the top camera's first frame the right arm's soft shadow falls
across S3's inboard side, streak and core; the core is still the darkest area there, and the shadow moves with the
arm. Field screening (a PID or XRF meter) is not modelled: visual staining
is the only cue, as in the first walk-over of a site.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the four flags back upright on the
table at x 20.0 cm, y +40.0 cm; x 29.0 cm, y +40.0 cm; x 20.0 cm, y -40.0 cm; and x 29.0 cm, y -40.0 cm. The site
sheet and the stains are fixed; nothing is consumed.

## Runtime per trial

About 1.5 minutes on an L40S for the demo (26 s of simulated time), and about 4 minutes for a full 1,800-step
Dream-1 run (both measured including Isaac Sim's start-up); the first start of Isaac Sim on a new instance takes several minutes
more.

## Real-world build

A 380 x 620 mm sheet of light grey vinyl (or a grey floor tile offcut) taped to the bench; three stains cut from
1 mm self-adhesive vinyl in pale tan, mid brown and near black, laid as listed in `task.json` (or printed on one
sheet of matte paper and laminated); four orange plastic rods 25 mm across and 60 mm tall, cut from 25 mm
polypropylene rod (900 to 910 kg/m3, so 27 g each), standing beside the sheet. Mark
the four 40 x 40 mm targets and the background pocket on a transparent overlay for scoring from a top photo.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

Every trace shows `arms_unavailable` = 0 (both arms read on every step). The demo ended on success at step 542 with
all four flags upright (tilt 0 degrees), released and still; the background flag stood at x 11.5 cm, y +0.6 cm,
177 mm from the nearest halo. In the 160-step settling run nothing moved. In the Dream-1 trial the arms moved and
closed their grippers, but no flag was picked up: all four stayed at their starts until the run limit.
