# Pinch a fingertip pulse oximeter open and clip it onto a manikin's finger

## What the robot does

A fingertip pulse oximeter lies shut on the table in front of a manikin hand whose index finger points at the robot.
The robot pinches the oximeter's tail behind the pivot to open its jaws, lifts it by that pinch, slides it over the
fingertip with the jaws held open until the fingertip meets the stop inside, then opens the gripper and moves away,
leaving the oximeter held on the finger by its own spring. One arm does the whole task; either arm can.

## Physics: why a grasp round the body cannot put it on

The numbers follow from the geometry in `layout.py` (the tests recompute each one):

| Pivot angle | Mouth (lips) | Cushions' inner end | Outside of the front end | Across the grip pads | Force per grip pad | Lips past a fingertip on the stop |
|---|---|---|---|---|---|---|
| 0 deg, shut | 6.0 mm | 11.7 mm | 32.0 mm | 37.0 mm | 0 | 31.0 mm |
| 14.1 deg, mouth as wide as the finger | 15.0 mm | 14.6 mm | 40.8 mm | 34.3 mm | 3.0 N | 30.2 mm |
| 16.3 deg, on the 15 mm finger | 16.4 mm | 15.0 mm, cushions parallel | 42.2 mm | 33.8 mm | 3.4 N | 30.0 mm |
| 22 deg, the scored "jaws open" line | 20.0 mm | 16.1 mm | 45.5 mm | 32.5 mm | 4.3 N | 29.5 mm |
| 28 deg, tails touching | 23.7 mm | 17.2 mm | 49.0 mm | 31.1 mm | 5.2 N | 28.9 mm |

1. **The obvious grasp shuts the clip.** The unit's centre is 8.5 mm ahead of the pivot. A grasp there, or anywhere
   ahead of the pivot, presses both shells together: the mouth stays 6 mm and the finger is 15 mm. Pushing the shut
   unit at the fingertip only butts its lips against it.
2. **The grasp that opens it must land on the tail.** The pinch has to bear on the two 10 x 20 mm grip pads at the
   back end. A gripper pad is 27 mm wide where it meets them, wider than the 20 mm tail, so it must overhang the back
   end: when its front edge is ahead of the pivot it bears on the body too and the jaws stop short (2.9 mm ahead:
   22 deg; 5 mm: 18.9 deg; 7.5 mm: 16.3 deg, no room left; over the pivot, 13.5 mm ahead: 12.1 deg, mouth 13.8 mm,
   no entry). For jaws that count as open the pad centre must lie between 10.5 mm behind the back end and 9.4 mm
   ahead of it: a 20 mm window at the far end of a 57 mm part whose centre, the natural target, is 19 mm outside it.
3. **The same grasp carries the unit.** Opened, the unit hangs from two upright line contacts (the grip pads' front
   edges on the flat gripper pads), about 23 mm behind its centre of mass, with the lips 49 mm ahead of those lines:
   a 2 deg swing of the unit in the pinch moves its mouth 1.7 mm.
4. **The fit narrows as it goes in.** 4.4 mm per side at the lips, 1.1 mm per side at the cushions' inner end,
   3.5 mm per side between the side walls, over 29 mm of travel past the lips.
5. **Letting go is part of the task.** The gripper pads leave the grip pads when the gap across them reaches 33.8 mm;
   from then on the unit sinks 3.5 mm (the play between its side walls and the finger) until its upper side wall rests
   on the finger, which carries its 0.49 N, and two cushions at about 1.9 N each keep it from sliding off along the
   finger (friction at 0.8 gives about 3.0 N, a margin of about 6). A unit released at 10 to 28 mm of depth still
   holds on but earns 0.75; one pushed fully on and released but not clipped on (moving, or not gripping) earns 0.5;
   one still held earns 0.25 at most, however far on it is.

What is not hard: regulating the squeeze. Fully open is a hard stop (the tails touch), so any firm squeeze holds the
jaws there and no force has an upper limit to respect. There is a floor: 4.3 N on each grip pad for jaws that count
as open, against 3.4 N at which the cushions only just pass the finger. With the simulated finger drive (5 N per mm
of each finger's position error, 50 N cap) a commanded gap of 29.0 mm or less opens the clip fully, 30.8 mm gives
22 deg, 32.4 mm gives the 16.3 deg at which the cushions just pass the finger, and 33.0 mm gives a mouth as wide as
the finger; the demo commands 25 mm (15 N per finger, 5.2 N of it against the spring).

A fully explicit instruction does not remove points 1 to 5: the instruction states the method.

## Scene and dimensions

| Part | Size and mass | Source |
|---|---|---|
| Oximeter outside | Contec CMS50D, 57 x 31 x 32 mm, about 50 g with two AAA cells | maker's manual |
| Method | open the clip, put the finger between the cushions, let it close; nail and light emitter on the same side | maker's manual |
| Pivot 20 mm from the back end; tail gap 10 mm; chamber 22 mm wide and 31 mm deep to a stop 6 mm ahead of the pivot; cushions 6 mm apart at the lips and 11.7 mm at their inner end, shut; display shell 14 g, battery shell with cells 36 g; spring 0.15 N m/rad from shut | | our estimate |
| Grip pads, 10 x 20 x 2.5 mm black rubber, one on each tail | added so a flat gripper pad bears on the tail | our addition |
| Manikin right hand and forearm, rigid; index finger 15 mm across (adult index fingertips are 14 to 17 mm), 72 mm long, rounded tip | | our part, printed to this table |

Station frame: x forward from the line through the arm bases, y to the left, heights above the table top. The
finger lies along x at y 0, its axis 75 mm up, tip at x 0.270, knuckle at x 0.342, nail toward +y; the rest of the
hand is beyond x 0.342 (fist 95 x 60 x 85 mm, forearm 58 mm thick to x 0.617). The oximeter lies on its side (pivot
upright), centre at (0.180, 0.000), 15.5 mm up, display shell toward +y, lips at x 0.2085, 61.5 mm short of the
fingertip. Fingertip steps: 10 mm across for the last 2 mm, 13 mm for the 2 mm behind that (a 7.5 mm ball end).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`, its thresholds `layout.py`. Readings, every control step,
in the centre-line frame (through the pivot, midway between the shells):

- **Depth**: how far the fingertip is past the line joining the lips. On the stop: 31.0 mm shut, 30.0 mm on the
  finger (16.3 deg), 28.9 mm fully open.
- **In the chamber**: depth 4 mm to 32.5 mm and the fingertip's centre within 5 mm of the centre line.
  **Over the finger**: in the chamber at depth 10 mm or more, length within 20 deg of the finger, display shell
  within 45 deg of the nail side. **Jaws open**: pivot at 22 deg or more.
- **Slid on open**: since the fingertip last came in past the lips, its greatest depth with the jaws not open is at
  most 1.5 mm more than its greatest depth with them open. Closing on the stop adds 30.0 - 28.9 = 1.1 mm by itself
  (the offline replay of the demo shows 0.9 mm), so the slack is that plus 0.4 mm of margin: the fingertip may not be
  pushed further in by spreading the cushions.
- **Gripping**: over the finger, pivot within 1.8 deg of 16.3 deg. **Let go**: on every step of the last 1 s, both
  grippers' finger zones 30 mm or more from the oximeter's exact shapes (`rigstate.released`; the zone is every point
  within half the finger gap plus 8 mm, and 13.5 mm across, of the 38 mm of the gripper's axis above the pad tips).
  An unreadable gripper, or a distance that cannot be computed, is not let go. **Clipped on**: gripping and let go,
  nothing of the oximeter within 20 mm of the table, and through the last 1 s gripping on every step, moved under
  2 mm and turned under 2 deg.

| Score | Condition | Real-trial reading |
|---|---|---|
| 1.0 (held 1 s, ends the trial) | clipped on, depth 28 mm or more, slid on open | the second pen ring is out of sight inside the unit in the photo from above; a 1 mm feeler is stopped within 5 mm by each cushion; grippers open, 3 cm clear and nothing moving for 1 s in the video; the front end 45.5 mm or more across (ruler on the frame, the unit is 57 mm long) in every frame in which the unit travels onto the finger, bar the last 1.5 mm |
| 0.75 | clipped on (depth 10 mm or more), short of 28 mm or not slid on open | the 10 mm ring is out of sight; the feeler, the grippers and the stillness as above |
| 0.5 | pushed home at some time (over the finger at depth 28 mm or more for 0.5 s), and now over the finger at depth 10 mm or more with every gripper 30 mm clear for 1 s, but not clipped on (moving, not gripping, or touching the table) | the second ring was out of sight in two video frames 0.5 s apart; the unit is on the finger, grippers 3 cm clear for 1 s |
| 0.25 | at some time jaws open for 0.5 s with all of the oximeter 20 mm or more above the table (a minor first stage, earned while still held) | two video frames 0.5 s apart with the unit clear of the table and its front end 45.5 mm or more across (32 mm shut) |
| 0 | anything else; doing nothing | |

The pen rings: the first is 10 mm from the fingertip; the second is drawn where the lips reach when the unit is
pushed fully on by hand and let go, less 2 mm toward the tip (28 mm with the dimensions above). Seen from above, the
battery shell's upper side wall covers the finger; its front edge is skewed 8.1 deg and covers 30.1 to 32.2 mm of a
finger on the stop, so a ring goes out of sight 0.1 mm after the lips pass it. Held fully open on the stop the lips
are 28.9 mm past the tip, so the second ring is out of sight only within 0.9 mm of the stop: "pushed home" means on
the stop, not near it.

Both records (pushed home, opened in the air) last until a reset, but only the second is credited by itself; 0.5 needs
the release as well. There is no latched zero: bumping the fixed hand, a second try or
a re-grip costs nothing.

Held and hovering states:

| State at the end | Score | Why it is not more |
|---|---|---|
| Held open in the air, anywhere | 0.25 | the clip was operated; nothing is on the finger |
| Held open round the fingertip, depth 10 to 28 mm | 0.25 | 4.4 to 1.1 mm clear of the finger all round |
| Held open on the stop, however long, then withdrawn with the unit | 0.25 | held credit stops at the first stage; every higher tier needs the release |
| Clipped on, gripper open but within 30 mm, or arms unreadable | 0.25 if it was opened in the air, else 0 | not let go |
| Pushed home, let go, on the finger but moving, or not gripping | 0.5 | released, short of clipped on |
| Clipped on and let go, short of the stop | 0.75 | holds by its spring; the sensor is short of the nail bed |
| Clipped on and let go at the stop, jaws not kept open on the way | 0.75 | the end state of 1.0 by a way the maker's method and the instruction rule out |

Other attempts and what they earn (each in the tests):

| Attempt | Score |
|---|---|
| Do nothing | 0 |
| Grasp round the body and push against the fingertip | 0 |
| Grasp over the pivot (jaws near 12 deg) | 0 |
| Lay the unit on the fist or across the finger | 0 |
| Pinch it open on the table without lifting | 0 |
| Show open jaws at the lips, relax the pinch, push the unit home | 0.75 if let go, 0.25 if held |
| Let the fingertip's rounded end into a part-open mouth while still squeezing, then go on with the jaws open | 1.0 |
| Clip it on with the display shell on the pad side, or crosswise | 0, or 0.25 if it was opened in the air |
| Clip on at the stop and let go, then knock it off before the 1 s is complete | 0.25 |
| Clip on at the stop, let go, unit still swinging | 0.5 |

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 1.0 |
| Dream-1 | 0.0 |

## Demo

Open loop: fixed waypoints from the setup text; it reads no simulator state. One arm on the midline, yaw 0 (fingers
closing along y). 23.5 s, 470 steps; run limit 900.

| # | Until (s) | TCP (x, y, z) m above the table | Commanded gap | What happens |
|---|---|---|---|---|
| 1 | 2.0 | (0.1510, 0, 0.120) | 60 mm | over the tail; pad centre 0.5 mm behind the back end, mid-window |
| 2 | 3.0 | (0.1510, 0, 0.046) | 60 mm | down to 3 cm above the pick; the pad tips are still above the unit |
| 3 | 4.5 | (0.1510, 0, 0.016) | 60 mm | slowly down the last 3 cm; pad tips 3 mm above the table, 11.5 mm from each grip pad |
| 4 | 5.0 | same | 60 mm | settle before the pinch |
| 5 | 6.5 | same | 25 mm | the pads meet the grip pads at 37 mm; the jaws open to 28 deg at 31.1 mm; the unit slides 4.2 mm forward |
| 6 | 7.0 | same | 25 mm | settle |
| 7 | 8.5 | (0.1510, 0, 0.0755) | 25 mm | lift 59.5 mm: the unit's centre at the finger's height |
| 8 | 10.5 | (0.1995, 0, 0.0755) | 25 mm | forward: the lips 10.6 mm short of the fingertip |
| 9 | 11.5 | same | 25 mm | settle 1 s: the wrist catches up before the slide |
| 10 | 16.0 | (0.2445, 0, 0.0755) | 25 mm | slide 45 mm at 10 mm/s, jaws at 28 deg; the fingertip passes the lips at TCP x 0.2101 and meets the stop at 0.2390; the last 5.5 mm load the soft wrist |
| 11 | 16.5 | same | 25 mm | settle on the stop |
| 12 | 18.0 | same | 70 mm | open: the cushions close on the finger; the pads leave the grip pads at 33.8 mm; the wrist's load takes up the 1.7 mm the pivot would draw back |
| 13 | 18.5 | same | 70 mm | settle |
| 14 | 20.0 | (0.2445, 0, 0.160) | 70 mm | straight up; the pads pass 18 mm clear of the grip pads on each side |
| 15 | 21.5 | (0.160, 0, 0.180) | 70 mm | back over the start: every finger zone 30 mm clear of the unit from about 21.0 s |
| 16 | 23.5 | same | 70 mm | hold: let go and still for 1 s, then 1.0 held for 1 s |

The demo obeys every instruction clause: the jaws are at 28 deg from the lips to the stop (slid on open), it ends
within 2 mm of the stop and it moves the gripper away.

Environment switches: `DEMO_IDLE_ARM=right` runs the plan with the left arm (`left` keeps the right arm's plan);
`DEMO_HOLD_ONLY=1` plays no plan (the settle check); `DEMO_VARIANT` selects a row of the table above;
`TASK_TRACE=trace.jsonl` writes one JSON line per control step with what the scorer saw and decided: `q_deg`,
`depth_mm`, `s_mm`, `v_mm`, `w_mm`, `align_deg`, `nail_deg`, `low_mm`, `in_chamber`, `over_finger`, `jaws_open`,
`gripping`, `d_open_mm`, `d_shut_mm`, `slid_on_open`, `released`, `let_go_1s`, `moved_mm`, `turned_deg`, `still_1s`,
`gripping_1s`, `clipped_on`, `opened_in_air`, `pushed_home`, `zone_mm`, `arms_readable`, `origin`, the arms' TCPs and
openings, then `tier` and `score`.

What to watch on the first GPU run, and the fallback for each (one change per run):

| Risk | Sign in the trace | Fallback |
|---|---|---|
| The pads sit off the commanded point along x | `q_deg` under 28 after step 6, or no opening | shift the pick x by the traced offset; the window is 9.9 mm either way |
| The pads slide on the grip pads instead of the unit sliding on the table | `origin` x after step 6 differs from the plan (pivot 24.7 mm ahead of the TCP) | set the slide end to the traced stop contact plus 5.5 mm |
| The pinch slips or chatters | `q_deg` wanders below 25 during the lift | pad friction 1.2 |
| The gripper pad bears on the body ahead of the pivot | `q_deg` stops between 17 and 24 | move the pick 3 mm back (TCP x 0.148) |
| The mouth is off the finger's line across the shells (wrist lag, pinch off-centre) | `v_mm` more than 3 mm at step 9 (the lips clear the 15 mm section by 4.4 mm a side) | shift the slide's y by the traced `v_mm`; slide at 5 mm/s |
| The unit pitches nose-down in the pinch | `w_mm` more than 3 mm at step 9 | add the measured drop to the carry height |
| The finger jams short of the stop | `depth_mm` 24 to 28 at step 11 | push 8 mm past contact instead of 5.5 |
| Letting go draws the unit off the stop | `depth_mm` under 29 after step 13 | push 8 mm past contact; open over 2.5 s |
| The 14 g jaw jitters | `q_deg` noise above 0.5 deg at rest | damping 0.02 |
| The jaw drifts open on the table (no preload) | `q_deg` above 1 after the settle run | joint friction 0.002 N m |

## Real-world build and reset

A Contec CMS50D with a 10 x 20 x 2.5 mm self-adhesive rubber pad on the end of each tail, laid on its side in a
tape outline. The manikin hand and forearm printed in skin-tone PLA to the table above (one piece or glued pieces;
the fingertip may be a true 7.5 mm ball end, which the two steps follow) and screwed to the table through the heel
and the forearm on marked positions; it is rigid, like the simulated one, so its 15 mm finger is the same part in
both. Two pen rings round the finger: one 10 mm from the tip, the other 2 mm nearer the tip than the lips reach when
the unit is pushed fully on by hand and let go. Score from the video, one photo from above and a 1 mm feeler gauge.
Nothing is powered, sharp or hot.

Reset: pinch the tail and take the unit off the finger; lay it in its tape outline, lips toward the hand, display
shell toward +y, jaws shut; check the hand on its marks and both rings visible; arms at READY, grippers open.

## Simplified, and what each simplification changes

- A linear spring without preload: the first millimetres of the squeeze are lighter than on a real unit.
- Hard stepped cushions and square lips: in the simulation a shut or part-shut unit cannot be pushed over the
  fingertip at all, while a real unit's soft rounded lips can be forced apart by a fingertip if a shell is free to
  move. The scoring does not rest on that difference: full score needs the jaws at 22 deg or more for the whole way
  in, which only a pinch gives, so a forced-on unit earns 0.75 at most in a real trial as well.
- The fingertip is rounded in two steps.
- The inside dimensions, mass split and spring rate are our estimates of the CMS50D (see the table above); the
  real-trial readings do not depend on them (rings drawn from the unit itself, the feeler, the width half-way between
  the unit's own clipped-on and tails-together widths).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the manikin and the oximeter, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, and the clip's geometry (mouth, pad lever, spring force, entry depth) |
| `scoring.py` | The scorer (`read`: the centre-line readings; `Tracker`: the slide record, the two records, the release and stillness windows, the tiers), NumPy only |
| `offline_clip.py` | The clip's kinematics for the offline checker and the tests only (never used with a simulator) |
| `mechkit.py` | Our mechanism kit, version 2, a copy of `tools/kit/mechkit.py`: the fixed hand, the oximeter articulation with its sprung hinge, tight-fit contact settings |
| `rigstate.py` | A copy of `tools/kit/rigstate.py`: reads both grippers each control step; its fail-closed `released` is the scorer's release check |
| `tests/test_scoring.py` | Offline tests: geometry, every tier, near misses, held and hovering states, the exploit list, unreadable arms, resets, the demo and its variants through `offline_clip.py`, the text |

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "$(python3 -c 'import json; print(json.load(open("task.json"))["instruction"])')" \
  --scorer task_score --max-steps 900 --store-frames --log-dir ~/yam-eval/pulse-oximeter-on-finger/demo
```

Replace `--policy yam_scripted` with `--policy dream` for Dream-1.

## Tests

`python3 -m unittest discover -s tests -v` (NumPy only), and from the repository root
`python3 tools/offline/check_task.py tasks/pulse-oximeter-on-finger --require-json --strict --one-arm`. Near misses
tested on both sides: depth 27.9 / 28.1, 9.9 / 10.1, 3.9 / 4.1 and 32.5 / 32.6 mm; hold band 14.4 / 14.6 and
18.0 / 18.2 deg; jaws open 21.9 / 22.1 deg; slide margin 1.4 / 1.6 mm; centre offset 4.9 / 5.1 mm across and along
the pivot; finger zone 29 / 31 mm; lowest point 19 / 21 mm; 9 / 10 steps for each record; alignment 19 / 21 deg;
nail side 44 / 46 deg; movement 1.9 / 2.1 mm; turn 1.9 / 2.1 deg; gripping lost, or the arms unreadable, on one step
of the last second.

Third-party assets: none. Primitives only; no meshes, fonts or textures.
