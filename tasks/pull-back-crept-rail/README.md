# Pull a crept rail back so both joints are at the 10 mm expansion gap

What is measured is two coupled gaps, set on a held gauge. A loose 310 mm middle length of rail lies between two
fixed rail ends 330 mm apart, so its two joints always add up to 20 mm: opening one closes the other. It has crept 10
mm to the left, so its left joint is shut (0 mm) and its right joint 20 mm (wide open). Both must end at the 10 mm
expansion gap, read at each joint's narrowest point as a taper gauge reads it, with the middle length still between
its seat shoulders, the gauge shim taken out of the track and put back on the table in front of it, and both grippers
clear. The way the trade does it is the way the instruction asks: stand the 10 mm shim in the open joint against the
fixed rail end and hold it there, slide the middle length onto it, let go of the rail, and lift the shim out. Full
marks need the gauge used: on some step the shim must stand in the right joint within 1 mm of both rail ends while
that joint is 8.5 to 11.5 mm. With both arms, one gripper holds the gauge while the other draws the rail (one arm
can do it too; see 'One arm alone'). Joints set by eye, without the gauge, score at most 0.5.

This is rail-creep correction on jointed track (O*NET 47-4061.00, Rail-Track Laying and Maintenance Equipment
Operators, and track labourers). Traffic and braking make rails creep along the track, shutting some joints and
opening others; a shut joint cannot take up summer expansion and the track can buckle, and an open one batters the
rail ends and breaks joint bars. With the fastenings slackened, a shim as thick as the rail-temperature table gives
is held in the joint, the rail is drawn up to it and the shim is taken out (an 18 m rail changes length by about
0.21 mm per deg C, so rail laid at -5 deg C needs about 10 mm). Here the table carries a bench-scale length of track
on four sleepers: a far rail set correctly (10 mm joints, for reference) and the near rail with the crept middle
length. Nearest tasks on the board, and how this differs: setting a valve lash with a feeler gauge under a screw
that is turned down onto it (both score the gauge's use; here a free part slides in its seats onto a gauge held in a
joint, two coupled gaps are both read, and the gauge is drawn out from between two rail ends and taken off the
track); mounting a power supply with a cooling gap to its neighbour, and squeezing a loose row of crates onto a
centre mark (both set by eye, with no gauge, and neither has a coupled pair of gaps); sliding a tile into a tight gap.
No railway task is on the board.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, reset notes, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain NumPy functions of the two loose parts' and the arms' states, with a small tracker |
| `rigstate.py` | Reads both arms' TCP, finger tips and opening for the scorer (wraps the embodiment's step and reset; changes nothing it does) |
| `tests/test_scoring.py` | Offline tests of every tier and every zero condition with its bounds, the full left-gap sweep, stillness, resets, robustness, the exploits we tried, the demo's switches and timing, and that the instruction is the same in all three files (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only `yam_arena`'s own
task-file objects (13 fixed boxes and 2 loose ones) and the standard `task_score` scorer.

The score needs both grippers clear of both parts for 1.0, and `yam_arena` does not pass the grippers' state to a
task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to
take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each
control step and to start a new trial at each reset. It reads only state the simulator already has and changes
nothing the embodiment does. If it cannot read the arms it prints why once; 'hands off' is then taken as met on
those steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The middle length of the near rail has crept: its left joint is shut and its right joint is wide open. Set both joints to the 10 mm expansion gap. Stand the yellow 10 mm shim upright in the right joint against the fixed rail end and hold it there, slide the middle length to the right until it bears on the shim, let go of the rail, then lift the shim out, put it back on the table where it started, in front of the sleeper ends, and move both grippers away." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/pull-back-crept-rail/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The middle length of the near rail has crept: its left joint is shut and its right joint is wide open. Set both joints to the 10 mm expansion gap. Stand the yellow 10 mm shim upright in the right joint against the fixed rail end and hold it there, slide the middle length to the right until it bears on the shim, let go of the rail, then lift the shim out, put it back on the table where it started, in front of the sleeper ends, and move both grippers away." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/pull-back-crept-rail/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check: nothing may move); `TASK_TRACE=trace.jsonl` writes one
JSON line per control step with what the scorer saw and decided (`tier`, `left_gap_mm`, `right_gap_mm`, `rail_x_mm`,
`rail_y_mm`, `rail_z_mm`, `rail_low_mm`, `rail_yaw_deg`, `rail_tilt_deg`, `rail_x_at_left_seat_mm`,
`rail_x_at_right_seat_mm`, `seated`, `shim_x_mm`, `shim_y_mm`, `shim_low_mm`, `shim_tilt_deg`, `shim_far_x_mm`,
`shim_away`, `shim_on_track`, `shim_in_joint`, `shim_to_fixed_end_mm` and `shim_to_rail_mm` (worked out only
while the shim is in the right joint), `gauged`, `gauged_step`, the four tip-point distances `left_tip_to_rail_mm`
... `right_tip_to_shim_mm`, `hands_off`,
`rail_path_mm`, `shim_path_mm`, `still`, `opening`, `arms_unavailable`, both parts' poses and velocities and both
TCPs). Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over
at every reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads both loose parts' poses once per control
step (20 Hz) and works out every corner from the pose and the parts' known sizes. It keeps three things from the
past: one whole-trial record ('gauged', below), each part's last 0.5 s of positions, for 'still', and when a new
episode starts. Everything else is the end state.

- **Gaps**: the left gap is 165 mm minus the largest y of any of the middle length's eight corners, the right gap
  the smallest y of any corner plus 165 mm (the fixed rail ends' faces are at y +165 and -165 mm). That is each
  joint's narrowest point, where a taper gauge stops. Square, the two add up to 20 mm; turned as far as its
  shoulders allow (3.1 degrees) they add up to 19.4 mm, so turning the rail never widens a gap.
- **Seated**: the middle length's lowest corner is 16 to 20 mm above the table (the sleeper tops are at 18 mm), its
  25 mm axis is within 3 degrees of vertical either way up (so not on its side and not tipped with an end up), and
  its centre line crosses y +8.0 cm and -8.0 cm (the two seats) within 6 mm of x 22.0 cm. The shoulders allow 5 mm
  either way (4.3 mm at the seats when it is turned 3.1 degrees), so between its shoulders it is always seated;
  riding up on a shoulder (it is then pitched 3.6 degrees), lifted, tipped or knocked out of line it is not.
- **Shim in the right joint**: its lowest corner within 3 mm of the table top (standing on it, or held just above
  it), part of it at x 21.0 to 23.0 cm (across the rail's line), and its centre between y -16.5 cm (the fixed rail
  end's face) and the middle length's right end.
- **On the track**: some corner of the shim at x 16.0 to 42.0 cm (over the sleepers), |y| below 42.0 cm (the rails
  end at 40.0 cm) and lower than 45 mm above the table, 2 mm above the rail tops: in a joint, on a rail or a
  sleeper, between the rails, or held down among them. Lifted 2 mm clear of the rail tops it is off the track.
- **Put away**: its lowest corner within 3 mm of the table top and every corner at x below 16.0 cm, nearer the arms
  than the sleepers' near ends, standing or lying. It starts there.
- **Hands off**: each gripper's finger-tip point (the point between its finger tips) at least 60 mm from the middle
  length and from the shim (their real, turned outlines). Fully open, a finger pad's outer corner is about 55 mm
  from that point, so at 60 mm neither gripper's pads can be on either part. Only that point is read: a policy that
  turned a gripper on its side could rest a palm or wrist on the rail with the tip point 60 mm away. That cannot
  move a gap, which is what is scored, and the scripted demo grips straight down.
- **Still**: over the last 0.5 s (10 control steps) the centres of the middle length and of the shim each moved
  along a path shorter than 2 mm. The reported velocity is never used: a part held still reports 10-45 mm/s of
  chatter. A part is never still on less than 0.5 s of history.
- **Gauged** (the one whole-trial record): on some control step the shim was in the right joint, within 1 mm of both
  the fixed rail end and the middle length (the gaps between their real, turned outlines, worked out exactly from
  the boxes' edges), while the right gap was 8.5 to 11.5 mm. That is the joint set on the gauge: the shim stands
  square across the joint and both rail ends bear on it. The two contacts and the band stop the shortcuts the tests
  try: a shim stood against the middle length's end instead of the fixed rail end touches only one end, and a 10 mm
  shim turned about 20 degrees, or leaning about 17, touches both ends of the 20 mm starting joint without the rail
  moving, but that joint is then 20 mm, outside the band. Reset clears the record; an unreadable step leaves it as
  it was. A real trial reads it from the video.
- **Bounds**: every bound stated as inclusive ('to', 'within', 'at least') is compared with a tolerance of 1e-9 m,
  so a value exactly on it lands in the tier the table gives it; 'under', 'below', 'lower than' and 'shorter than'
  stay strict.

| Outcome (checked in this order) | Score |
|---|---|
| The middle length not seated (out of its seats, up on a shoulder, tipped, on its side, held up), or either gap under 3 mm (nothing done, slid under 3 mm, or slid more than 17 mm so the right joint closed up) | 0 |
| The shim in the right joint, held or let go (the run limit reached with the rail drawn onto the held shim, or the gauge left in: a joint with the gauge in it cannot close) | 0.25 |
| The shim anywhere else on the track: on a rail or a sleeper, between the rails, in the left joint, its far end poking in between the sleepers, held down among the rails | 0 |
| Both gaps 8.5 to 11.5 mm, the shim put away, both grippers' tip points 60 mm or more from both parts, both parts still, and 'gauged' set during the trial | 1.0 |
| Both gaps 6 to 14 mm, but not 1.0 (a gripper still on the rail or the shim; the shim held in the air, on the table behind the track or dropped off the table; a part still moving; the joints set by eye without the gauge; a gap of 6 to 8.5 or 11.5 to 14 mm) | 0.5 |
| Anything else: both gaps at least 3 mm, one under 6 mm (square, the shut joint opened only 3 to 6 mm, or the rail slid 14 to 17 mm and the open joint shut too far) | 0.25 |

Possible scores: 0, 0.25, 0.5, 1.0. Every outcome falls in exactly one row. Seated and square, a slide of s mm
leaves gaps of s and 20 - s mm; with the shim put away, hands off, both parts still and 'gauged' set, as the left gap
goes from 0 to 20 mm the score is 0, 0.25, 0.5, 1.0, 0.5, 0.25, 0, changing at a left gap of 3, 6, 8.5, 11.5, 14
and 17 mm; without 'gauged' it is 0, 0.25, 0.5, 0.25, 0, changing at 3, 6, 14 and 17 mm; the tests sweep both in
0.01 mm steps. The start (0 mm and 20 mm) scores 0 and `success()` is false there. The score follows the procedure:
nothing done 0; the rail drawn onto the held shim 0.25; the shim lifted out, the joints set, a gripper still on a
part 0.5; both grippers clear 1.0. The trial ends with success once the 1.0 conditions have held for 1 s
(`SUCCESS_HOLD_S = 1.0`); otherwise `task_score` is read from the last step at the run limit with the same rules
(`SCORE_OVER = "end"`).

Unintended strategies earn no more than the table says, as the tests show: setting both joints by eye at 10 mm
without the shim scores 0.5; wedging the shim on a slant across the wide joint, or standing it against the middle
length's end, then setting the joints by eye, scores 0.5; holding the rail in place at 10 mm instead of letting go
scores 0.5; leaving the shim in the right joint scores 0.25 whatever the gaps; dropping it anywhere else on the track
scores 0; dragging the rail until the right joint shuts scores 0; a rail left lifted out of its seats, or knocked off
them, scores 0; the shim still held over the table in front of the track scores 0.5 at any height (the gripper is on
it), as does the shim put down behind the track or dropped off the table; turning the rail against its shoulders
cannot widen a gap. Lifting the rail out and laying it back between its shoulders in the right place is scored on
its end state, like any other way of moving it: nothing in the scene can be broken or modified, and every fixed part
stays put. A simulator state with a missing field, a non-finite number or a zero quaternion scores 0 for that step
and leaves the tracker as it was. A new episode is recognised from rigstate's reset hook, and as a fallback when both
loose parts are back within 4 mm of their starting places and at least one of them got there by a jump of more than 4
mm (in plan) beyond what its velocity allows in one step (the demo puts the shim back on its own place, so at a reset
only the rail jumps, 10 mm); either way the 'gauged' record is cleared.

## One arm alone

Measured in the simulator (the scripted demo with the other arm left at its ready pose, one trial each; the
offline kinematic proxy had predicted 0 and 0.5):

- right arm alone (`DEMO_IDLE_ARM=left`): **0**. The shim goes into the right joint and back out to the table, the
  rail never moves, and the left joint stays shut (0 mm).
- left arm alone (`DEMO_IDLE_ARM=right`): **0.25**. The shim never leaves the table, so 'gauged' is never set, and
  with no shim in the joint nothing stops the rail: the slide, commanded 27 mm, carried it about 15 mm, and it
  ended with gaps of 15.1 mm (left) and 4.6 mm (right), one under 6 mm. Even with the rail stopped at 10 mm this
  half of the demo could not score more than 0.5 (no gauge).

One arm can do this task; we do not claim otherwise. Straight down, the right arm reaches the shim, the open joint
and the middle length anywhere from y 0 to -13 cm (offline kinematics with a collision proxy), so a capable policy
can stand the shim in the joint against the fixed rail end and let go (it stands on its 10 mm edge; it tips at 12.5
degrees), draw the rail onto it, let go, and lift the shim out: 1.0. The left arm reaches the shim where it starts,
but straight down it reaches the open joint (x 21 to 23 cm, y -16 cm) only out to x 21 cm, and there only with its
TCP within about 3 cm of the table; tilted 10 degrees or more toward the right, it reaches the joint's centre line
at every height from the table to 10 cm, so a tilted grip lets it do the whole task too. Skipping the shim and
sliding the rail 10 mm by eye earns at most 0.5: the 1.0 band is plus or minus 1.5 mm, the arm trails a push by
several millimetres under the rail's friction, and the trade uses a gauge for that reason. With both arms the gauge
is held while the rail is drawn onto it, as a track gang works, and it never stands unsupported.

## Demo and run limit

Both arms run on one clock; the scripted demo takes 46.3 s (926 control steps). Finger targets are 8 mm narrower
than the part (52 mm across the shim's 60 mm, 12 mm across the rail's 20 mm), every close and release takes 1.5 s
after a 0.5-0.6 s settle, every final descent is slow (1.5-2 s), and both grippers close their fingers along x,
across the track (left yaw -90 degrees, right +90).

- Right arm (shim): over the shim, down, grip it at its 60 mm ends with the TCP 30 mm up (pad tips 17 mm up), lift
  to 10 cm, carry it over the middle of the open right joint (y -15.5 cm), down until its underside is 2 mm above
  the rail tops, then slowly into the joint (5 mm clear of each rail end) until its underside is 1.5 mm above the
  table (17.6 s), and move it 5 mm right, to y -16.0 cm, so that its face is on the fixed rail end's face (in place
  at 19.7 s). It holds the shim there until the left gripper has let go of the rail and risen clear (31.2 s), lifts
  it straight out until its underside is 2 mm above the rail tops (off the track), carries it back to where it
  started, sets it down on its edge, lets go, goes up and parks at x 15 cm, y -25 cm, 14 cm up.
- Left arm (rail): over the middle length at y +13.5 cm (30 mm in from its left end, outboard of the left middle
  sleeper and its shoulders), down, grip it across its width with the TCP 40 mm up (pad tips 27 mm up, 16 mm of pad
  on the rail), done 0.5 s after the shim is in place. Then it slides the rail right: commanded 27 mm over 4 s with
  the grip 2 mm higher, held 1 s (the rail on the shim, the shim on the fixed rail end: 'gauged' is set here), then
  eased back over 2 s to 10 mm (where the rail stops on the shim) and its starting height, let go, straight up to 10
  cm (31.2 s), and parked at x 20 cm, y +22 cm, 14 cm up.

Why the slide is commanded past the shim: the YAM's joints are position-controlled springs, and the compliance model
(J K^-1 J^T, joints 80/80/80/10/10/10 N m/rad) gives about 5 mm of trail per newton along y at this grip. Pushing
the rail (0.25 kg; 2.45 N of sliding friction, 2.9 N to start it) toward the right arm also presses the left gripper
down on it (about 0.5 N per newton of push with the rail held), adding friction; lifting the grip 2 mm while it
pushes cancels that. Commanded 27 mm, 17 mm past the shim, the rail reaches the shim for any trail up to 17 mm and
then presses on it, backed by the fixed rail end, with about 3-6 N (a one-dimensional friction model of the slide,
with arm stiffnesses of 0.18 to 0.6 N/mm along y, with and without the press-down, stops the rail on the shim every
time; without the 2 mm lift the softer arms never start it); easing the gripper back to 10 mm leaves no load on the
rail when it is let go, so nothing drags it back or pushes it on, and the shim is drawn out only after the left
gripper has let go (a shim pulled while the rail is still pressed lets the rail jump into the gap). The order of the
two arms is set by waypoint durations computed from the other arm's plan.

Offline (a kinematic check of the scripted demo): worst IK error 0.099 mm / 0.012
deg; smallest joint margins 14.4 deg (right joint 4) and 18.6 deg (left joint 4); peak joint speed 0.71 rad/s;
arm-to-arm capsule clearance 8.2 cm and TCPs at least 26.8 cm apart (both at t 24.1 s, the left arm at the end of
its commanded slide while the right holds the shim); lowest pad tips 17 mm (right, in the joint) and 27 mm (left)
above the table. The kinematic proxy carries the rail through the shim and into the fixed rail end during the
over-command (its warnings about that are expected: the proxy has no contacts), ends with the rail 10 mm along (gaps
10.0 and 10.0 mm) and the shim back on the table, and predicts 1.0: success (the 1.0 conditions held for 1 s) comes
at step 921 (46.05 s), once the right gripper has moved 60 mm clear of the shim on its way to park; the scripted
policy stops at step 987. It also notes that the rail's centre is not over the one sleeper it names as its support:
the rail rests on both middle sleepers, its centre between them.

What to check in `TASK_TRACE` on the GPU: `shim_low_mm` at most 3 (1.5 commanded) from 17.6 s, with `shim_in_joint`
true (43 mm or more means the shim landed on a rail end: move `JOINT_Y` in `task.py`, demo only); `gauged` true by
about 25 s, with `shim_to_fixed_end_mm` and `shim_to_rail_mm` both 1 or less while the rail is held on the shim (if
the shim is held short of the fixed rail end, the rail's push should close it up); `rail_y_mm` 0 within about 0.5
mm from then on, through the release (both gaps 10 mm). To retune the slide, read `rail_y_mm`, `rail_x_mm`,
`left_gap_mm` and `right_gap_mm` and adjust `SLIDE_CMD`, `LIFT_BIAS` and `EASE_TO` in `task.py`.

The run limit is 3,000 steps (150 s), about 3.2 times the demo: enough for a model to set the shim, draw the rail
and correct a gap once.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Run with no object jitter: the middle length sits
between its shoulders with 5 mm of play and against the left fixed rail end, and a random shift would start it
inside one of them. Reset: the middle length back on the two middle sleepers between its shoulders, its centre 10 mm
left of the track's centre (its left end against the fixed rail end, left joint shut; right joint 20 mm, checked
with feeler gauges); the shim back standing on its 60 x 10 mm edge on the table centred at x 12.0 cm, y -16.0 cm,
its 60 mm side across the track. Everything else is fixed. Nothing is consumed.

## Runtime per trial

About 1.5 to 3 minutes on an L40S once Isaac Sim is running (the demo is 46 s of simulated time, a full-length
Dream-1 trial 150 s); the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Four 260 x 25 x 18 mm hardwood strips (painted dark brown) screwed across a baseboard at 8 and 25 cm either side of
the centre line as sleepers. Rails: 20 x 25 mm aluminium box section (4 mm walls) painted steel grey; screw two 235
mm lengths and one 310 mm length to the sleepers along the far line with 10 mm joints, and two 235 mm lengths along
the near line with their inner ends 330 mm apart; the near 310 mm middle length stays loose (about 0.25 kg). Four 6
x 25 x 10 mm aluminium blocks, painted dark grey, screwed to the two middle sleepers either side of the middle
length, 5 mm clear of it each side, are the rail seat shoulders. The shim is a 60 x 10 x 45 mm piece of steel flat
bar (about 0.21 kg), deburred and painted yellow, stood on its 60 x 10 mm edge in front of the track. The middle
length slides on the bare sleepers (with less friction than the simulation's 1.0). Nothing is powered and nothing is
sharp. To score a real trial: a taper gauge in each joint at its narrowest point (1.0 needs 8.5 to 11.5 mm in both);
a top photo showing the middle length between its shoulders on both middle sleepers, the shim on the table in front
of the sleeper ends and both grippers at least 6 cm from both parts; and, from the video, whether the shim stood in
the right joint against the fixed rail end with the middle length's end on it ('gauged').

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |

Scripted demo: the shim was in the right joint against the fixed rail end and 'gauged' was set at step 464 (23.2 s);
the rail was let go with gaps of 9.4 mm (left) and 10.2 mm (right), and the trial ended with success at step 922
(46.1 s). Settling (no plan, 160 steps): nothing moved. Dream-1 ran the full 3,000 steps: it moved the middle length
0.5 mm (left gap 0.3 mm, right gap 19.3 mm) and knocked the shim over: it ended lying flat on the table, its
centre about 25 mm left of where it stood, still in front of the sleeper ends; score 0 (the left joint still shut).
The one-arm runs are described under 'One arm alone'.
