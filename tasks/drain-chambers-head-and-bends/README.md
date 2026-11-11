# Set four inspection chambers on a new drain where Approved Document H requires access

Pipelayers and plumbers (O*NET 47-2151.00, 47-2152.00) set the access points on a new below-ground drain from the
drawings and the Building Regulations, and Building Control checks them before the trench is backfilled. Approved
Document H (England), Part H1, Section 2 (2015 edition, paragraph 2.49) asks for access on or near the head of each
drain run, at a bend and at a change of gradient, at a change of pipe size, and at a junction unless each run can
be cleared from an access point. Table 13 limits the spacing (22 m from the start of a drain to an inspection
chamber, 45 m between inspection chambers). Straight lengths between access points need no chamber: they are
rodded from the chambers at their ends. A chamber in the wrong place leaves a length that cannot be cleared, and
putting it right means digging the run up again.

Here a plan at 1:50 (1 m = 20 mm) is fixed to the table. A new foul drain, drawn in dark blue, starts at its free
end (the head of the run), runs as a thin 10 mm line (100 mm pipe), widens to a thick 20 mm line (150 mm pipe),
turns twice through 90 degrees and ends at the existing chamber at the site boundary. It has no junctions and one
fall (it steps up to 150 mm to suit the existing chamber), and its lengths (7.5, 8, 7 and 12.5 m) are all within
Table 13, so the rule asks for access at exactly four points: the head, the change from 100 to 150 mm pipe, and the
two bends. None of them is marked. Four loose chamber bases stand on the plan off the route. The robot has to read
the plan and stand one chamber, upright and let go, on each of those four points; a chamber on a straight length,
beside the line or against the existing chamber does nothing. Each point covered scores 0.25. The natural shortcuts
score partial credit only: chambers at the corners and the ends of the run miss the size change (0.75), and
chambers spaced evenly along the run cover only the head (0.25).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place, with the arithmetic the scene rests on |
| `scoring.py` | The scorer: plain functions of the four chambers' states (and the arms' pad-tip points and openings, for 'let go'), with a small tracker for 'settled' and 'still' and the trace |
| `rigstate.py` | Reads both arms' pad-tip points and gripper openings before every control step, by wrapping the embodiment's step and reset from the task file's import (changes nothing the embodiment does); the same file as in our other tasks |
| `tests/test_scoring.py` | Offline tests of every tier, the 15 mm cover radius, every zero condition, settling and stillness, the let-go rule, the shortcut checks, resets, broken states, the demo plan and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes and cylinders) and the standard `task_score` scorer;
`rigstate.py`, in this folder, reads the arms only to check that each chamber was let go.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This plan, drawn at 1:50 (1 m = 20 mm), shows a new below-ground foul drain. It runs from its free end, the head of the run, to the existing chamber at the boundary; it is laid at one fall and has no junctions. The thin line is 100 mm pipe, the thick line 150 mm pipe. Set the four inspection chamber bases on the route where Approved Document H requires access points on this run, each one upright and centred on the line, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drain-chambers-head-and-bends/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "This plan, drawn at 1:50 (1 m = 20 mm), shows a new below-ground foul drain. It runs from its free end, the head of the run, to the existing chamber at the boundary; it is laid at one fall and has no junctions. The thin line is 100 mm pipe, the thick line 150 mm pipe. Set the four inspection chamber bases on the route where Approved Document H requires access points on this run, each one upright and centred on the line, then let go of them." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/drain-chambers-head-and-bends/dream
```

Run one trial per process (the default). The score depends on where the chambers stand, whether a gripper still
holds one and each chamber's path over the last second; the small tracker in the task module keeps that path and
diagnostics for the trace, and starts a new episode when rigstate sees the embodiment reset or the simulator puts
every chamber back on its mark in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`n_covered`, `covered` (point -> chamber), `missing`, `counted`,
`held`, `not_standing`, `not_settled`, `all_still_1s`, `nearest` (each chamber's nearest point and distance), the
first step each score was reached; each arm's `left_tips`/`right_tips` and `left_opening`/`right_opening`,
`arms_readable`, `reach_mm` and `arms_unavailable`, or `rigstate_error`; and each chamber's centre, tilt, bottom
above the sheet, nearest point, `off_mm` and `off_xy_mm` from it, path over the last 0.5 s, `held` and
`standing`). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads the four chambers' poses once per control
step (20 Hz), and from the embodiment each gripper's pad-tip point (13 mm past its tool point, midway between the
finger tips) and how far it is open, and scores the state at hand: the score is read at the last step. Distances
are between a chamber's centre and a point, seen from above, as a ruler from the drawn line's end, width change
and corners, or a top photo with a scale, reads them once the arms have withdrawn.

The four points (table frame, cm): the head H at x 37.0, y -26.0; the change from 100 to 150 mm pipe P at x 37.0,
y -11.0; the first bend B1 at x 37.0, y +5.0; the second bend B2 at x 23.0, y +5.0. A chamber is counted when it
is tilted at most 10 degrees from upright (either end up), its lowest point (worked out from its pose) is within
3 mm of the sheet top, it is let go, and it is settled (its centre moved under 2 mm in all over the last 0.5 s). Let
go: no gripper's pad-tip point is less than 10 mm above its top while within that gripper's finger reach of its
axis. The gripper's yaw is not read, so the reach is the widest over every yaw, hypot(half the finger gap + 10 mm,
20 mm) + 30 mm + 5 mm: 95 mm fully open, 80 mm closed on a chamber, 57 mm closed on nothing. Each gripper is read
on its own: on a step where one cannot be read, the let-go rule skips it but still checks the other (such steps
are counted in the trace's `arms_unavailable`). A counted chamber covers a point when its centre is within 15 mm
(0.75 m at 1:50) of it. The points are at least 140 mm apart, so a chamber covers at most one point, and two 60 mm
chambers cannot both stand within 15 mm of one point.

| n = points covered | Score |
|---|---|
| 0 | 0 |
| 1 | 0.25 |
| 2 | 0.5 |
| 3 | 0.75 |
| 4, every chamber also settled in the 0.5 s before the last 0.5 s (still for 1 s) | 1.0 |
| 4, not yet still for 1 s | 0.75 |

Every outcome, with numbers: chambers at H, P, B1 and B2 (any chamber at any point), 1.0; the head and both bends
with the fourth on a straight (say midway between H and P), against the existing chamber, still gripped, or lying
on its side, 0.75; only the two bends, or only H and P, 0.5; chambers every 175 mm along the 700 mm route from the
head, 0.25 (the head is covered, the others miss P, B1 and B2 by 25, 40 and 75 mm); nothing moved, 0. So every
score 0, 0.25, 0.5, 0.75 and 1.0 can be reached. A chamber more than 15 mm from every point, tipped more than 10
degrees, stacked on another (its bottom 40 mm up), held up, still gripped or still moving covers nothing. The
trial ends early (success) once 1.0 has held for 2 s; otherwise it is scored as it stands at the run limit. A
chamber counts only once let go; the 2 s hold is also longer than the 1.5 s release after the last settle, so even
on a step where the arms could not be read the trial cannot end while a chamber is still in a closed gripper.
Stillness is read from each chamber's path, not from the simulator's reported velocity, which chatters while a
part is gripped. A state with a non-finite number or a zero quaternion (a simulator blow-up) scores 0 for that step
and leaves the tracker as it was.

Unintended strategies earn nothing extra, as the tests show: chambers at the corners and both ends of the run
(0.75: the size change is missed), at the corners and the size change but 60 mm short of the head (0.75), spaced
evenly (0.25), stacked two high at one point (counted once), held on a point by a gripper (not counted), laid on
their sides on the points (0). The plan, the drawn lines and the existing chamber are fixed, so nothing on the
plan can be moved or destroyed.

## One arm alone

The demo splits the work by side: the right arm sets the two chambers on its side at the head and the size change,
the left arm the two on its side at the two bends. With one arm idle the other still covers its own two points:
expected 0.5 each (offline proxy: 0.5 with the left arm idle, 0.5 with the right arm idle; GPU results below once
run). Gripping straight down at any heading (tried every 30 degrees), the left arm cannot reach the right side's
chambers (x 23.0, y -25.0 cm and x 31.0, y -19.0 cm), nor the head or the size change, and the right arm cannot
reach the left side's chambers (x 34.0, y +13.0 cm and x 35.0, y +26.0 cm), by our IK model of the arm
(`tests/test_scoring.py`, `OneArmReach`). The right arm can reach the two bends, so it could set one of its own
two chambers at a bend instead, still 0.5. We claim nothing about the task needing two arms: a policy able to grip
or push chambers with the wrist tilted might reach further.

## Demo and run limit

The scripted demo takes 39.8 s (796 control steps), both arms on one clock. Each chamber is gripped straight down
across its 60 mm diameter with the fingers closing along y, opened to 80 mm (10 mm either side) and commanded to
52 mm to close (8 mm narrower than the part), with the tool point 24 mm above the chamber's bottom so the pads hold
its lower 11 to 40 mm and their tips stay 11 mm above the sheet. Every gripper open or close takes 1.5 s after a
0.5 s settle, every final descent (onto a chamber and onto the plan) is slow (20 mm over 1.2 or 1.5 s), each
chamber is let go 2 mm above the sheet, and chambers travel with their bottoms 61 mm up (21 mm over any standing
chamber and the existing one). Each place point is aimed 4 mm beyond its access point along the carry, because
parts land a few millimetres short under load; the 15 mm radius leaves 11 mm to spare. In order:

1. The right arm sets the chamber at x 23.0, y -25.0 cm on the head (x 37.0, y -26.0 cm); at the same time the
   left arm sets the chamber at x 34.0, y +13.0 cm on the first bend (x 37.0, y +5.0 cm), 31 cm away.
2. The right arm sets the chamber at x 31.0, y -19.0 cm on the size change (x 37.0, y -11.0 cm) and parks. The
   left arm waits 7 s over the chamber at x 35.0, y +26.0 cm, then sets it on the second bend (x 23.0, y +5.0 cm)
   once the right arm has left the size change (both at once, their elbows would pass 38 mm apart).
3. The left arm parks over its side, clear of the route.

The grippers stay at least 27.4 cm apart and the arms' collision capsules at least 10.5 cm (offline). The smallest
joint margin is 15.8 degrees (joint 4, left arm, over the first bend at travel height). A 100 g chamber barely
loads an arm. A chamber set on a drawn line (0.5 mm proud) stands level on it, or rests with its rim on the sheet
up to about 1.5 degrees off upright, well inside the 10 degrees allowed. `PLACE_TRIM` in `task.py` moves an arm's
place points if the GPU run shows them landing off by a steady amount. Offline, the last chamber counts once the
open left gripper's pad tips have risen 10 mm over its top and success is held at step 797 (39.9 s); the scripted
policy would stop at step 857. The run limit is 1,200 steps (60 s), half as long again as the demo.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the four chamber bases back upright on
the plan sheet centred at x 23.0, y -25.0 cm; x 31.0, y -19.0 cm; x 34.0, y +13.0 cm; and x 35.0, y +26.0 cm (any
base on any spot). The sheet, the drawn lines, the boundary line and the existing chamber are fixed.

## Runtime per trial

About 1 to 2 minutes on an L40S once Isaac Sim is running (the demo ends after about 40 s of simulated time, when
success has held for 2 s; a run to the limit is 60 s); the first start of Isaac Sim on a new instance takes several
minutes more.

## Real-world build

A 250 x 660 mm pale cream rigid plastic sheet 1 mm thick taped to the table 17 cm in front of the arm bases (x
17.0 to 42.0 cm, y -31.0 to +35.0 cm). The drain drawn on it in dark blue vinyl tape 0.5 mm thick: a 10 mm strip
from the head at x 37.0, y -26.0 cm along x 37.0 cm to y -11.0 cm, then a 20 mm strip on to y +5.0 cm, along y +5.0
cm to x 23.0 cm and along x 23.0 cm to y +26.5 cm. A red tape strip 3 mm wide across the sheet at y +34.0 cm (the
site boundary). The existing chamber: a 70 mm dark grey concrete or painted cylinder 40 mm tall glued at x 23.0, y
+30.0 cm, a 56 mm light grey plastic disc 1 mm thick on top. Four 60 mm black plastic cylinders 40 mm tall weighted to about 100 g
(short lengths of pipe with a plug, or turned plastic) for the chamber bases. Positions in `task.json`'s `setup`.

Scoring a real trial: after the arms have let go and withdrawn, check each chamber stands upright on the sheet,
then measure from the end of the thin line, from where the line widens, and from the two corners of the line to
the nearest chamber's centre (a chamber covers that point within 15 mm), or take a top photo with a ruler on the
sheet for scale. Count the points covered and score 0.25 each (1.0 for all four).

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
