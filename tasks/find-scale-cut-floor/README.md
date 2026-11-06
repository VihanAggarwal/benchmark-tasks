# Lay the photo scale beside a find on the floor of its cut

Before an excavated find is lifted, it is photographed where it lies. Tools and hands come out of the shot, and a
scale goes beside the find, parallel to it, flat on the same surface the find lies on, and not touching it. The
find itself is never moved: where it lies is part of the record. The level matters. A scale left on the ground
surface above a sunken cut is nearer the camera than the find, so it reads too large: about 4% here, where the cut
is 40 mm deep and the station camera about 0.94 m up, and about 25% in a 0.5 m grave cut shot from a 2 m pole. So a
scale laid neatly on the bank beside the find earns no scale credit here, and a find nudged more than 5 mm scores 0.

On the table, four light-brown blocks 40 mm tall (the ground surface) make a square frame round a 180 x 180 mm cut
whose floor is the table top. A cream find lies on the cut floor, turned 20 degrees. A grey trowel lies parallel to
it, 18 mm from its -y side, on the very strip where the scale belongs: the 20 mm scale cannot fit between them, and
beyond the trowel it would be too far from the find. An orange tool tray stands 36 cm right of the midline (only the
right arm reaches it) and a dark-grey photo-kit box with a white 10 cm scale bar 33 cm left (only the left arm
reaches it). The scorer reads the end state as a tape measure would (the scale parallel within 5 degrees, its centre
33-63 mm from the find's centre line, which for a parallel scale is an edge gap of 10-40 mm, centred along the find
within 25 mm, at floor level; the trowel in the tray; both grippers open and out of the shot) and watches the find
over the whole trial (moved over 5 mm, turned or tilted over 3 degrees, or lifted over 3 mm: 0).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the find's, trowel's and scale's states and the grippers', with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, bad states and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The 'hands clear' rule needs the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset (it also counts the episodes, and the scorer starts over only when that count
changes). It reads only state the simulator already has and changes nothing the embodiment does. If it cannot read
the arms it prints why once; the hands then count as not clear (no credit), and the trace counts those steps
(`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the find ready for its record photograph. Put the trowel back in the orange tool tray, then lay the 10 cm scale flat on the floor of the cut beside the find, centred alongside it, parallel to it and 1 to 4 cm from it. Leave the find exactly where it lies. When you are done, let go and move both arms back, clear of the cut and out of the photo." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/find-scale-cut-floor/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Get the find ready for its record photograph. Put the trowel back in the orange tool tray, then lay the 10 cm scale flat on the floor of the cut beside the find, centred alongside it, parallel to it and 1 to 4 cm from it. Leave the find exactly where it lies. When you are done, let go and move both arms back, clear of the cut and out of the photo." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/find-scale-cut-floor/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`disturbed`, `disturbed_why`, `find_move_mm`,
`find_turn_deg`, `find_max_move_mm`, `find_max_turn_deg`, `find_max_tilt_deg`, `find_max_lift_mm`, `scale_on_floor`,
`scale_bottom_mm`, `scale_tilt_deg`, `scale_yaw_deg`, `a_deg`, `g_mm`, `s_mm`, `overlap`, `tight`, `rough`, `stowed`,
`trowel_bottom_over_tray_floor_mm`, `trowel_tilt_deg`, `scale_rests_on_find`, `trowel_rests_on_find`,
`hands_clear`, `opening`, `tcp_in_photo`, `still`, `arms_unavailable`, the objects' poses and both TCPs). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step the scorer reads the three loose
objects' poses and both grippers (tool point and opening), and keeps the find's pose at the first control step, the
first time the find was disturbed, and the last 0.5 s of positions (for stillness). In the find frame, u is along
the find's long axis and n square to it, seen from above.

- **Disturbed** (latched over the whole trial): the find's centre more than 5 mm (horizontally) from its reference,
  its heading turned more than 3 degrees, tilted more than 3 degrees, or its bottom more than 3 mm above the floor.
- **Rests on the find** (at the end): the scale's or the trowel's outline seen from above overlaps the find's, and
  that object's bottom is less than 21 mm above the cut floor (the find's 16 mm top plus 5 mm).
- **On the floor**: the scale's bottom within 3 mm of the cut floor, its 15 mm axis within 5 degrees of vertical
  (either 100 x 20 mm face down: a real build bands both), its whole outline inside the cut opening (2 mm slack for
  contact).
- **a**: the angle between the scale's and the find's long axes, folded to 0-90 degrees. **g**: the distance from
  the scale's centre to the find's long centre line, along n, minus 23 mm (half the find's 26 mm plus half the
  scale's 20 mm: for a parallel scale, the edge-to-edge gap). **s**: from the find's centre to the scale's, along u.
- **Tight**: on the floor, a <= 5 deg, g 10-40 mm, s <= 25 mm, outlines not overlapping. **Rough**: on the floor,
  a <= 12 deg, g 5-50 mm, s <= 45 mm, not overlapping (tight is also rough).
- **Stowed**: the trowel's centre inside the tray's inner outline (x 16-34 cm, y -42.5 to -29.5 cm), its bottom
  within 5 mm of the tray floor (10 mm above the table). A trowel leaning with its lowest point on the floor counts.
- **Hands clear** (at the end): both grippers open 0.5 or more (pads 47 mm or more apart; no loose part is over 40 mm
  across, so nothing can be held), and neither TCP, seen from above, at x 5 cm or more with y -21 to +21 cm: over
  the frame, within 60 mm of it, or past its far side. The 60 mm is because a fully open gripper's pads reach about
  56 mm from its TCP (half the 94 mm gap, the 8 mm pad, half its 20 mm width), so no pad is over the frame; the zone
  runs on past the far side because a TCP out there (the left arm reaches about x 48 cm, y +12 cm at table height)
  has its upper arm over the far bank. Grippers that cannot be read are not clear. The offline collision proxy
  (capsules fatter than the real links) puts no part of either arm over the cut for any reachable straight-down
  pose with the TCP outside the zone (the closest comes 27 mm short of it); its fattest finger and wrist capsules
  can still reach about 3 cm in over the banks' outer edges, never the cut.

| Outcome | Score |
|---|---|
| The find disturbed at any step, anything resting on the find, or the hands not clear | 0 |
| Scale tight and trowel stowed | 1.0 |
| Scale rough (not tight) and trowel stowed | 0.75 |
| Scale tight or rough, trowel not stowed (still in the cut, on a bank, beside the tray, on its rim) | 0.5 |
| Trowel stowed, scale neither (still in the kit, on the trowel, on the bank above the find, too far, crooked) | 0.25 |
| Anything else (nothing placed, or the scale neither tight nor rough with the trowel not stowed) | 0 |

Every outcome maps to exactly one score. The score is read at the last step (`SCORE_OVER = "end"`); the trial ends
early once the 1.0 conditions, hands clear included, have held for 1 s (`SUCCESS_HOLD_S = 1.0`) with the find, the
scale and the trowel each within 3 mm of every position it had over the last 0.5 s (positions, never velocities: a
gripped part's reported velocity chatters). A new trial starts only at a reset: rigstate's reset hook, and the
episode number `task.py` passes with every step. Within one episode nothing clears the record, so a find lifted
and put back stays disturbed even if every part is then returned to its marks. (Only `scoring.py` used on its own,
with no episode number, falls back to spotting a new trial when all three loose objects teleport back within 2 mm
and 2 degrees of their starting poses with one of them jumping more than 20 mm in a single control step.) A
simulator state with a non-finite number or a zero quaternion scores 0 for that step and leaves the record of the
trial as it was.

Why the trowel matters, and the exploits the tests cover:

- **Squeezing the scale in beside the trowel.** The trowel lies parallel to the find, 18 mm from it, on the scale's
  strip; the 20 mm scale cannot fit, and beyond the trowel it is 58 mm or more from the find, past the 50 mm rough
  limit. A search over g 0-70 mm, s -60 to +60 mm and a -12 to +12 degrees of every pose a real scale could take on
  that side (inside the cut, overlapping neither the find nor the trowel) finds no tight or rough pose while the
  trowel lies where it starts; with the trowel gone it finds hundreds. A scale beyond the trowel scores 0.
- **The cramped far side.** On the find's +y side the floor strip is only 12-43 mm wide. A search of the scorer
  itself, with its 2 mm cut slack (n in 0.5 mm, s in 1 mm, a in 0.5 degree steps), finds about 2,500 cramped rough
  poses, all toward the near end: g 5-16 mm, |s| 19-39 mm, a -12 to +2.5 degrees (most turned toward the bank).
  Without the slack (the poses a real scale could take) they narrow to g 5-13.5 mm, |s| 24-37 mm, a -12 to -0.5
  degrees. None is tight: with the trowel left in the cut such a pose is 0.5, the same as any placement that leaves
  the trowel out of the tray.
  This follows from the geometry, not from an order rule: a tight scale needs the trowel moved, and full credit
  needs it in the tray (tools out of the shot).
- **Holding instead of placing.** A scale held still at floor level, or the trowel held on the tray floor, leaves a
  gripper closed (and, for the scale, a TCP over the cut): 0. An open gripper left over the cut is 0 too.
- **Parking a hand at the edge of the shot.** An open gripper with its TCP 31 mm off the near bank (x 7.99 cm, 20 mm
  up) would hang its pads over the bank: inside the 60 mm zone, 0. A TCP past the frame's far side, its arm reaching
  over the frame, is 0 too.
- **Making room by moving the find.** Lifting it out of the way and putting it back latches 0 (its bottom rose more
  than 3 mm); shoving it 10 mm latches 0. A nudge under 5 mm and 3 degrees is tolerated and buys nothing. Putting
  the find, the trowel and the scale all back on their marks (the scale dropped 25 mm into the kit, like a reset)
  and starting again does not clear the latch: only a new episode does.
- **Using the find as a rest.** A scale laid on the find, leaning on it, or a trowel left across it: 0.
- **The wrong level.** A scale on the trowel in the cut is 18 mm up, not on the floor (0, or 0.25 with the scale on
  the stowed trowel in the tray). A scale on the bank beside the find, at the right gap but on the ground surface,
  scores 0.25 with the trowel stowed. A scale standing on edge or tipped more than 5 degrees is not on the floor.
- **Moving the trowel without stowing it.** Pushing it aside on the cut floor or lifting it onto a bank, then laying
  the scale tight, is allowed and pays 0.5; parked on the tray's rim (20 mm above its floor) it is not stowed.
- **Dropping the scale.** A dropped scale that lands in place is a legitimate placement; one that knocks the find
  7 mm scores 0.

## One arm alone

The tool tray is 0.36 m right of the midline and only the right arm reaches it; the photo-kit box is 0.33 m left and
only the left arm reaches it. The demo uses both arms at once: the right arm takes the trowel by its handle and
stows it while the left fetches the scale, then the left lays the scale where the trowel was. The two arms are needed
because of this reach split, not because the act itself needs two hands. Measured in simulation (one trial each,
as the offline kinematic proxy predicted): the right arm alone (`DEMO_IDLE_ARM=left`) stows the trowel and leaves
the scale in the kit, 0.25; the left arm alone running its own plan (`DEMO_IDLE_ARM=right`) lays the scale on the
trowel, 18 mm above the cut floor, 0. The find did not move in either run. A left arm that lifted the trowel onto a
bank (or pushed it aside) and then laid the scale tight would score 0.5 (a scorer test, not a simulated run). Neither
arm alone reaches 0.75 or 1.0: at x 0.25 the tray's inner edge is 0.65 m from the left arm's base and the kit's
0.64 m from the right arm's, beyond reach even with grips tilted 15-40 degrees. (The full table is in 'Results in
simulation' below.)

## Demo and run limit

The scripted demo takes 25.3 s (506 control steps), both arms on one clock. Fingers are commanded 8 mm narrower than
the part and every open and close takes 1.5 s, with a settle before each close and release. The right arm comes
over the trowel's handle end (48 mm from its centre, toward the robot), opens to 52 mm, goes down (pad tips 2 mm
above the floor; the find-side pad passes about 20 mm beyond the find's end), closes to 32 mm on the 40 mm trowel, lifts
it to 12 cm, carries it over the tool tray, lowers it onto the tray floor (its centre at the tray's centre; pads
13 mm from the walls), opens to 52 mm, goes up and moves back to (20, -25) cm, fully open, by 17.6 s. Meanwhile the
left arm opens to 34 mm over the scale, goes down, closes to 12 mm on its 20 mm width, lifts it 10 cm and waits
beside the cut until the trowel is over the tray (12.0 s). It then crosses the midline at 10 cm (joint 4 keeps
0.34 rad of margin over the scale's place; 0.25 rad at 12 cm), turns the scale parallel to the find over its place (g 30 mm, s 0,
a 0), comes down to 25 mm above the floor, lowers the last 25 mm over 2.0 s, opens to 34 mm only (the find-side pad
stays 15 mm from the find), lifts straight up to 10 cm, opens fully and moves back to (25, +25) cm. Offline the IK
error stays under 0.09 mm, every joint keeps at least 19 degrees of margin, the TCPs stay at least 249 mm apart and
the arms' collision capsules 81 mm apart; the hands are clear from step 496 (the left TCP leaves the zone during its
retreat), success has held for its 1 s at step 515, and the scripted policy would stop at step 567 (plan plus its
3 s hold). The run limit is 1,200 steps (60 s at 20 Hz).

Gripped 48 mm from its centre, the 90 g trowel puts about 0.04 N m on a friction grip. In simulation it did not
pivot: `trowel_tilt_deg` stayed at or below about 2 degrees while it was carried and lowered, it ended flat on the
tray floor, and the find did not move (0 mm, 0 degrees). The simulated demo's trace (sampled every 6 steps) has the
trowel stowed by step 265, the scale tight by step 355, the hands clear by step 499 and success at step 516. (Offline, the trowel's
closest approach to the near bank on the lift is 11.6 mm and the pad's to the tray's right wall at the release
13.2 mm.) If a real grip lets it pivot, gripping it 38 mm from its centre still keeps the find-side pad about 10 mm
beyond the find's end.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the find back on its corner tape marks
(centred at x 26.5 cm, y +5.0 cm, long axis at 20 degrees), the trowel back on its outline with its handle end toward
the robot, and the scale back flat in the kit box along x; the frame, the tool tray and the kit box are fixed.
Nothing is consumed.

## Runtime per trial

About 1.5 minutes on an L40S once Isaac Sim is running (the demo is 26 s of simulated time); the first start of
Isaac Sim on a new instance takes several minutes more.

## Real-world build

Four pine blocks 40 mm tall, painted light brown, screwed to a board as a 300 x 300 mm frame round a 180 x 180 mm
opening (or clamped to the bench), the bench top being the cut floor; a 90 x 26 x 16 mm PLA print, cream, of about
35 g, as the find; a 140 x 40 x 18 mm HDPE block of about 90 g as the trowel (mark its handle end); an open orange
plastic tray 180 x 130 mm inside with 30 mm walls and a dark-grey tray 140 x 90 mm inside with 15 mm walls, both
clamped; and a 100 x 20 x 15 mm hardwood bar of about 25 g, painted white, as the scale, with 1 cm black bands and
a centre mark painted on both 100 x 20 mm faces (the scorer accepts either face down, so whichever face is up must
read in the photo). Mark the find's long centre line and its centre on its top, and put tape marks at its four
corners on the floor. To score: the trial video shows any lift, tilt or turn of the find during the trial, and the
tape marks give its final displacement and turn (against 5 mm and 3 degrees). In a top photo, measure with a steel
rule square from the scale's centre mark to the find's centre line and subtract 23 mm: that is g as the scorer
computes it (from the scale's centre, not its edge), and it equals the edge-to-edge gap when the scale is parallel.
s is the distance along the find's centre line from its centre mark to the foot of that square line. The photo or a
protractor gives a, a straightedge laid across the banks shows the scale's top 25 +/- 3 mm below it (on the floor),
a rule gives the trowel's place in the tray. For 'hands clear', tape lines on the bench along the zone's edge: one
across at x 5 cm from y -21 to +21 cm, and one along x at y +21 cm and at y -21 cm from x 5 cm to beyond the far bank.
The last video frame, taken from above, then shows each gripper's tool point against them (outside the taped zone)
and whether both grippers are open at least 47 mm.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
