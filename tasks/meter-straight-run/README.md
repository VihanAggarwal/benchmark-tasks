# Set a water meter with 10 pipe diameters of straight pipe before it and 5 after

Velocity-type water meters (single-jet, Woltmann) read correctly only when the water reaching them flows straight.
ISO 4064-1 and OIML R 49 give each meter a flow-profile sensitivity class marked on it, for example U10 D5: at least
10 nominal diameters of straight pipe before the meter and 5 after it, with no elbow, valve or other fitting inside
those lengths (butted straights of the same bore count as straight pipe). Closer than that, the swirl from an elbow or a valve shifts the reading and the customer is billed
wrongly for years; a meter fitted backwards reads backwards or not at all. Plumbers and meter installers therefore
lay a meter set out from pre-cut straights so that the meter falls in the right place, arrow downstream, register
up. Here the run is 25 mm pipe between a fixed copper elbow (the supply) and a fixed brass valve, 580 mm apart, in a
channel between two low rails. Three straights (250, 125 and 75 mm) and the meter (100 mm) must be butted end to end
from the elbow, the valve's union taking up the 30 mm left at its end. Only some orders leave 250 mm (10 D) before
the meter and 125 mm (5 D) after it: the 250 mm straight must go upstream of the meter and the 125 mm one
downstream. The parts start on both sides of the table, each pair out of the other arm's grasp, so the scripted demo
uses both arms; one arm alone would have to rake the far pair within reach (see 'One arm alone').

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the parts' poses and the grippers' finger tips, with a small tracker |
| `mechkit.py` | Builds the meter as one rigid body made of several boxes (body, register, arrow), and reports its frame |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every zero condition, resets, the demo and the exploits we tried |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. The elbow, valve, rails and the three
straights are `yam_arena`'s own task-file boxes. The meter needs a visible arrow and register, so it is a compound of
seven boxes made into one loose rigid body by `mechkit.py`, which hooks `yam_arena` from the task file's import (it
changes no `yam_arena` file and adds no joints). The release rule needs the grippers' finger tips and tool points,
which `yam_arena` does not pass to a task file's hooks: `rigstate.py` wraps the embodiment's step and reset once, at
import, to take a snapshot of both arms before each control step and to start a new trial at each reset. It reads only
state the simulator already has. If it cannot read the arms it prints why once; a gripper that cannot be read has not
let go, so the score is 0 on those steps (reason `gripper_not_readable`), and the trace counts them
(`arms_unavailable`, 0 in a good run).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the water meter and the three pipe lengths end to end in the channel from the elbow to the valve, the first one against the elbow and every joint closed; the valve's union takes up the gap left at its end. Water flows from the elbow to the valve: point the meter's arrow toward the valve with its blue register on top, and leave at least 10 pipe diameters (250 mm) of straight pipe between the elbow and the meter and 5 (125 mm) between the meter and the valve. Let go and move clear." \
  --scorer task_score --max-steps 4000 --store-frames --log-dir ~/yam-eval/meter-straight-run/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the water meter and the three pipe lengths end to end in the channel from the elbow to the valve, the first one against the elbow and every joint closed; the valve's union takes up the gap left at its end. Water flows from the elbow to the valve: point the meter's arrow toward the valve with its blue register on top, and leave at least 10 pipe diameters (250 mm) of straight pipe between the elbow and the meter and 5 (125 mm) between the meter and the valve. Let go and move clear." \
  --scorer task_score --max-steps 4000 --store-frames --log-dir ~/yam-eval/meter-straight-run/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per control
step with what the scorer saw and decided (`score`, `reason`, `parts` with each part's `laid`, `why`, `flat_deg`,
`heading_deg`, `centre_off_mm`, `bottom_mm` and end positions, `order`, `joints_mm`, `valve_gap_mm`, `U_mm`, `D_mm`,
`arrow`, `tips_clear_mm` (each gripper's finger-line clearance, null if unreadable), `released`, `path_mm`, `still`,
`arms_unavailable`, the meter's origin and `meter_source`, both finger-tip points and openings). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every reset;
the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the end state. A part is laid when it
lies flat (a straight on any of its four long faces, within 5 degrees; the meter with its register on top, within 5
degrees), along the run within 5 degrees, its centre within 8 mm of the channel's centre line (x 30.0 cm) and its
lowest point within 3 mm of the table (so not on a rail or on another part). Its ends are its two end-face centres,
and the parts are taken in order from the elbow. The elbow joint runs from the elbow's face to the first part's near
end; each part-to-part joint from one part's far end to the next part's near end. A joint is closed between -2 mm and
+5 mm: -2 mm allows for contact penetration in the simulation, and anything below it (two parts overlapping along the
run, which only a part lying on or across another can do) is not a joint. The gap at the valve is not checked: the
union takes up 0 to 30 mm. U is the straight run from the elbow's face to the meter's inlet face; D from its outlet
face to the valve's face. A gripper has let go when it can be read and its finger line, from its finger-tip point 38
mm back up the gripper (through its tool point), is 65 mm or more from every part's outline box (the meter's 40 mm
tall with its register and arrow). An open finger reaches 60.4 mm out from that line (the outer corner of its 10 mm
thick, 40 mm wide section, 57 mm out and 20 mm across), so no finger of a let-go gripper touches a part, however the
gripper is tilted; for a gripper pointing down, the line's lowest point, the finger-tip point, decides. Each gripper
is judged on its own, and one that cannot be read on a step has not let go.

| Outcome (in this order) | Score |
|---|---|
| A gripper's finger line within 65 mm of a part, or a gripper that cannot be read | 0 |
| The run not closed: fewer than four parts laid (one tilted, on a rail, on another part, off the line, across it, the meter on its side or upside down), or a joint over 5 mm or under -2 mm | 0 |
| Closed, the meter's arrow toward the elbow | 0.25 |
| Closed, the arrow toward the valve, U at least 245 mm and D at least 120 mm | 1.0 |
| Closed, the arrow toward the valve, U under 245 mm or D under 120 mm (they cannot both be short) | 0.5 |

From the elbow, exactly four orders score 1.0: 75-250-meter-125 (U 325 mm, D 155 mm with the joints at 0),
250-75-meter-125 (325, 155), 250-meter-75-125 and 250-meter-125-75 (250, 230). With every joint anywhere in -2 to +5
mm, these give U 246-260 or 319-340 mm and D 220-234 or 140-161 mm (D is always 480 mm less U). 250-125-meter-75 puts
the meter 375 mm from the elbow and leaves D at most 111 mm (0.5); any order with the 250 after the meter leaves U at
most 215 mm (0.5). The 5 mm margins below 250 and 125 mm cover a tape's reading and the -2 mm the joint rule allows
each joint: every 1.0 order passes at its worst, and no other order comes within 9 mm of either line. The trial ends
early once the 1.0 conditions have held for 1 s with no part's end-face centre having travelled more than 2 mm in all
over the last 0.5 s (a path, so a part that shivers in place adds up its shiver; reported velocities, which chatter on
resting parts, are not used); otherwise it is scored as it stands at the run limit. A new trial is recognised from
rigstate's reset hook or, as a fallback, when some part jumps further in one step than its velocity allows and every
part is then back at its start (within 4.5 cm, which covers a seeded reset's jitter). A part state with a non-finite
number or a zero quaternion scores 0 for that step and leaves the record as it was.

Unintended strategies earn nothing more than the text says, as the tests show: laying a part on the table along the
line past the elbow or past the valve opens a joint at least as wide as the 40 mm block between it and the rest of the
run (0); the trap order 250-125-meter-75 with its joints opened to stretch D to 125 mm opens a joint past 5 mm (0);
setting the meter on top of the 250 to save length lifts it off the table (0); holding the last part in place instead
of letting go is 0; so is resting an open finger on a part or pushing a part's end with the back of an open finger,
with the point between the finger tips 30 to 54 mm away, or a gripper lying on its side with its fingers beside a part
and its tips pointing away, and so is a step on which one gripper cannot be read while the other is still on a part; a
short part cocked 9 degrees in the 37 mm channel is not along the run (0); butting the run against the valve instead
of the elbow leaves a 30 mm elbow joint (0). Pushing the parts along the table into the channel instead of lifting
them is allowed: only the end state counts, as a fitter's tape and feeler gauge would see it.

## One arm alone

In simulation on the GPU (see 'Results in simulation'), the scripted demo scores 0.0 with either arm idle, as the
offline kinematic check (`--one-arm`) predicted. With the left arm idle (`DEMO_IDLE_ARM=left`) the right arm lays the
250 and the 125 in the channel and the 75 and the meter stay where they start (0: the 75 is not along the run). With
the right arm idle (`DEMO_IDLE_ARM=right`) the left arm lays the 75 against the elbow (a 0.3 mm joint) and the meter
in the channel with a 244 mm gap where the 250 belongs, and the 250 and the 125 stay where they start (0: the 250 is
not along the run).
Lifting and carrying as the demo does, one arm can lay only its own side's two parts, and the run needs all four.
Reach across the midline (inverse kinematics of the arm's URDF with its link collision proxy; the point between the
finger tips 12 mm above the table, the gripper tilted toward the other side, fingers closing along y), at the far
parts' near ends (x 8.75 to 23.5 cm): gripping straight down an arm reaches 15-20 cm past the midline; tilted 15
degrees 22-27 cm, 25 degrees 27-32 cm, 40 degrees 33.5-38 cm. An arm reaches farthest across near the bases, where
the short pipes lie. The other side's parts start with their nearest edges 39.75 cm (the meter, the 250) and 40.75 cm
(the 75, the 125) from the midline: at least 27 mm beyond the 40-degree reach at their x (the 125 27 mm, the 75 29
mm, the meter 52 mm, the 250 62 mm). Tilted 50 degrees, which only works with the fingers opened little wider than a
part (otherwise the lower finger is in the table), the finger tips would reach 5 to 11 mm over the near edge of the
other side's short pipe at its end nearest the bases, still short of its centre line, and stay 12 mm (the meter) and
24 mm (the 250) short of the long parts. Pushing a far part toward yourself would mean getting behind it. The only
one-arm route we can think of is to use the 250 mm straight as a steeply tilted rake to drag the far parts within
reach; the scripted demo does not try it, so we do not claim one arm cannot score.

## Demo and run limit

The scripted demo lays the run in the order 75-250-meter-125 (U 325 mm, D 155 mm) and takes 53.8 s (1,076 control
steps). Every part is picked lying along x with the fingers closing across its 25 mm (commanded 17 mm, about 20 N per
finger, over 1.5 s), the tool point 25 mm above the table; lifted to 12 cm and turned a quarter turn in the hand (left
arm to yaw -90 degrees, right arm to +90, the signs that reach the midline); carried at 10 cm; lowered into the
channel 12 mm short of its upstream neighbour (the last 28 mm over 1.5 s) with its bottom 2 mm above the table; slid
toward the elbow while still gripped, over 1.5 s, to 10 mm past contact, pressing the parts already laid against the
fixed elbow; held 0.5 s, eased back to 1.5 mm past contact, let go over 1.5 s and left straight up to 7.5 cm (finger
tips 37 mm above the pipes, 22 mm above the meter's arrow), then parked with the finger lines 123 mm (left) and 135 mm
(right) from the nearest part on the GPU (123 and 138 mm offline), past the 65 mm the release rule needs. The left arm
sets the 75 (picked where it lies, near the left base) and then the meter, each lifted and moved inboard to x 24 cm, y
38 cm before its turn (where the 75 lies the turn would pin joint 6; there it keeps 12.7 degrees, at the meter's pick
point 5.9). The meter is gripped 25 mm from its centre toward its inlet end: the 27 mm wide pads end 11.5 mm short of
that end, more than their 8 mm offset, so neither catches the 250 while the meter is slid to butt, and they close on
the 25 mm body's sides, 4.5 mm outside the 16 mm register. It is turned clockwise seen from above, so its arrow ends
toward the valve and its gripped inlet half leads. The right arm sets the 250 (gripped 30 mm from its centre toward
the arms, so it lands 30 mm nearer the right arm; turned where it was picked, joint 6 12.4 degrees from its limit) and
then the 125 (picked near the right base, turned at x 24 cm, y -38 cm). Each arm starts toward the channel only once
the other has opened its fingers there and is going up, and the right arm's last carry waits until the left is fully
up (the TCPs stay 30 cm apart, the arm links 14.9 cm, capsule proxy); the 250 and the meter come down diagonally from
3 cm nearer their own side, where the reach near the midline is lower. Offline, success has held for its 1 s at step
1,073 (53.65 s); the scripted policy would stop at step 1,137. On the GPU the trial ended at step 1,074 with 1.0 held:
order 75-250-meter-125, U 325.3 mm, D 154.7 mm, joints 0.1 to 0.3 mm, every part within 1.6 mm of the centre line and
1.3 degrees of the run, the meter's arrow toward the valve. The run limit is 4,000 steps (200 s at 20 Hz), 3.7 times
the demo.

Trims for the GPU, in `task.py`: `OVERSHOOT` (10 mm, how far past contact a part is pressed; the arm trails 3-7 mm
under such a load), `RELAX` (1.5 mm), `GAP0` (12 mm, the set-down gap) and `PLACE_TRIM` per arm (x, y). Read
`joints_mm`, each part's `centre_off_mm` and `heading_deg`, `U_mm`, `D_mm` and `tips_clear_mm` in `TASK_TRACE`, and
check that `arms_unavailable` stays 0.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the 250 mm straight back flat along x
centred at x 36.0 cm, y -41.0 cm; the 125 mm at x 15.0, y -42.0; the 75 mm at x 15.0, y +42.0; the meter's body
centred at x 26.0, y +41.0, register up, arrow pointing away from the arms. The elbow, supply pipe, valve and rails
are fixed. Nothing is consumed. Every loose part starts inside 43.25 cm of the midline, clear of the arms' start
pose, so all four are in the setup screenshot (the first top-camera frame); only the 250's far 8 cm is behind the
right gripper in that view. Each short pipe lies 22.5 mm short of its side's long part, nearer the arm bases.

## Runtime per trial

On an L40S, measured per process (including Isaac Sim's start on a warm instance): the demo about 2.7 minutes
(54 s of simulated time); the Dream-1 run to the 4,000-step limit (200 s) about 8 to 9 minutes. The first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

Two 580 mm aluminium strips 6 x 8 mm glued to the bench 37 mm apart; a 40 mm wooden or metal block painted copper at
one end with a 60 mm stub of 25 mm square bar against its far side (the elbow and its supply), and a 40 mm brass or
painted block with a red lever strip on top at the other (the valve), their facing sides 580 mm apart, across the
channel's ends. The straights are 25 x 25 mm square aluminium tube (2 mm wall) cut to 250, 125 and 75 mm and painted
copper. The meter is a 100 x 25 x 25 mm hollow brass box with 2 mm walls (about 170 g; a solid aluminium block weighs
about the same) with a blue painted-aluminium block 40 x 16 x 12 mm (about 20 g) glued on top and white plastic arrow
strips on it: about 0.19 kg in all, close to the model's 0.20 kg. Nothing is powered, hot or sharp. Score with a tape
from the elbow's face to the meter's inlet face and from its outlet face to the valve, a 5 mm feeler gauge at every
joint, and a photo from above.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

A settling run (no arm moves, 160 steps) scores 0.0 with no part moved. The demo ended at step 1,074 with 1.0 held
(see 'Demo and run limit'). Left arm idle: the right arm laid the 250 and the 125; the 75 and the meter stayed where
they start (0: the 75 is not along the run). Right arm idle: the left arm laid the 75 against the elbow and the meter
244 mm beyond the 75's end; the 250 and the 125 stayed where they start (0: the 250 is not along the run). Dream-1
ran to the 4,000-step limit and was not a refusal: its left gripper hovered over the meter, opening and closing, for
most of the run without moving it; near the end (from about step 3,700) its right arm picked up the 250 mm straight
and set it down along the run near y -10 cm on top of the near rail (8 mm up, 20 mm off the centre line), so it is
not laid, and the 75, the 125 and the meter never moved (0).
