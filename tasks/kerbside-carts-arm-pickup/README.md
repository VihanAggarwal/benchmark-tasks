# Set two carts out for an automated refuse truck: lids to the street, 3 ft clear

Refuse and recyclable material collectors drive automated side-loader trucks: an arm reaches out from the truck,
grabs each wheeled cart, lifts it and tips it into the hopper. The arm can only grab a cart that stands at the kerb
with its lid opening toward the street (handle and wheels toward the house) and with about 3 feet (0.9 m) of space
around it: from the next cart, mailboxes, poles, hydrants and parked cars. Cities and haulers print this rule on
collection calendars and mould an arrow into the lids that points to the street. A cart set out wrongly is skipped,
or the driver steps down and re-sets it by hand; walk-up services for residents who cannot move their own carts set
them out the same way. This hauler wants carts on the verge, never on the kerb or in the road, where they block
traffic.

Here a residential street at 1:10 scale stands on the table: the table top is the verge, a raised kerb runs across
the far side with the road beyond it, a mailbox on its post stands on the midline, a fire hydrant to the left, a car
is parked at the kerb to the right and the house's driveway crosses the verge on the far left. Two carts (green and
blue lids) stand near the house, side-on, their fronts toward each other, as they were wheeled out of the garage.
The score is what a collector sees from the cab and what a top photo with a tape shows: each cart at the kerb (its
front 0 to 30 mm behind the kerb face, 0 to 0.3 m at scale), its lid arrow toward the street within 25 degrees, off
the driveway and 91 mm (3 ft at scale) clear of the other cart and of everything the truck's arm could hit. Between
the mailbox and the hydrant there is room for one cart, and between the mailbox and the car for one more.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two carts' poses, with a small tracker of their last half second |
| `mechkit.py` | Helper that builds each cart as one rigid body of nine coloured boxes and reports its pose (see Requirements) |
| `tests/test_scoring.py` | Offline tests of every scoring tier and band edge, every zero condition, the exploits we tried, resets, the demo plan, its carry paths and the one-arm reach (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses
`yam_arena`'s own fixed boxes and cylinders (the kerb, road, driveway, mailbox, hydrant and car), the standard
`task_score` scorer, and one helper module next to `task.py`.

`yam_arena`'s loose objects are single boxes, cylinders or spheres of one colour, and a plain block has no front or
back. `mechkit.py` (imported by `task.py`) builds each cart from nine boxes (body, lid, handle bar, two wheels and a
four-part white lid arrow) as one loose rigid body through a hook on `yam_arena.scene`, and reports its pose (the
frame origin at the middle of its body's base, and its orientation) to the scorer. It changes nothing else the
embodiment does, and it stops the run with an error if a cart was not built as a compound body. The technique (a
`task_file.Box` subclass the loader accepts, a `sys.meta_path` hook patching `yam_arena.scene` from the task file's
import, pxr authoring in a custom spawner) follows the one first shown in the program repository's
`tasks/bearing-press`; `mechkit.py` is an independent implementation.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms at once, one cart each): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set both wheeled carts out for the automated refuse truck. Stand each cart on the verge with its lid opening toward the street (the white arrow, square to the kerb within 25 degrees; handle and wheels toward the house), its front 0 to 30 mm (0 to 0.3 m at this 1:10 scale) behind the kerb and not on it. Keep each cart at least 91 mm (3 feet) from the other cart, the mailbox and its post, the hydrant and the parked car, and off the driveway." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/kerbside-carts-arm-pickup/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Set both wheeled carts out for the automated refuse truck. Stand each cart on the verge with its lid opening toward the street (the white arrow, square to the kerb within 25 degrees; handle and wheels toward the house), its front 0 to 30 mm (0 to 0.3 m at this 1:10 scale) behind the kerb and not on it. Keep each cart at least 91 mm (3 feet) from the other cart, the mailbox and its post, the hydrant and the parked car, and off the driveway." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/kerbside-carts-arm-pickup/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`score`, `success`, `zero_reason`, `valid_state`, and for
each cart `credit`, `why`, `standing`, `facing_err_deg`, `setback_mm`, `on_driveway`, `clear_mm`, `nearest`,
`origin_mm`, `tilt_deg`, `lowest_mm`, `path_mm`; then the carts' raw `origins`, `quats`, `low_z` and `vels`).
Videos: `inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at
every reset (mechkit's reset counter, or the simulator putting both carts back on their marks in one step); the run
rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The score is read from the carts' poses at the last step.
A cart's **footprint** is its 79 x 58 mm outline seen from above (handle and wheels included; for a tilted cart, the
shadow of its whole box); its **front** is the lid's opening side, where the white arrow points.

Seen from above (x up the page, toward the street; y to the left):

```
   y +64        y +36   y +30              y 0                  y -30            y -64
   . . . . . . road (x 40 to 60) . . . . . . . . . . . . . . . . . . . . . . . . . . . .
   [driveway] ====kerb, face at x 39.0====================================[ parked car ]
   [x 24-40,]       (H) hydrant              [M] mailbox on its post       [x 41-59,   ]
   [ dropped ]           x 36.5                   x 35-39                  [ y -64..-30]
   [  kerb   ]   |  one clear place  |                 |  one clear place  |
                 |  centre y +12.8   |                 |  centre y -12.8   |
                 |     to +16.8      |                 |     to -18.6      |

                     [green lid]                            [blue lid]
                   centre x 22, y +13                    centre x 22, y -13
                   front toward -y  -->              <--  front toward +y      (start)

          left arm base (y +30.5)                x 0                right arm base (y -30.5)
```

**Standing**: tilted 10 degrees or less, its lowest point within 3 mm of the verge, and still (the centre of its
base moved under 5 mm over the last 0.5 s, turns counted at its corners: positions, not the simulator's velocities,
which chatter while a part is gripped). **Facing the street**: its front within 25 degrees of +x. **At the kerb**:
the footprint's most forward point 0 to 30 mm behind the kerb face (x 36.0 to 39.0 cm; up to 2 mm past it allowed
for contact). **On the driveway**: any part of the footprint over the driveway sheet. **Clear**: the footprint at
least 91 mm, seen from above, from the other cart's footprint, the mailbox and its post (the mailbox's 40 x 16 mm
outline covers the post), the hydrant's 24 mm circle and the car's 180 x 340 mm outline.

| Outcome | Score |
|---|---|
| Either cart knocked over (tilted more than 45 degrees) | 0 for the trial |
| Any part of either cart's footprint more than 2 mm past the kerb face's line (on the kerb, in the road, or past the line where it crosses the driveway) | 0 for the trial |
| Otherwise, per cart: standing, off the driveway, facing the street, at the kerb and clear | 0.5 |
| ... all of those but not clear (within 91 mm of the other cart, the mailbox, the hydrant or the car) | 0.25 |
| ... anything else (left at the house, side-on or with its handle to the street, more than 30 mm back, on the driveway, lifted, moving or leaning) | 0 |
| Total: the sum of the two carts | 0, 0.25, 0.5, 0.75, 1.0 |

Examples: the start 0; one cart set out between the mailbox and the hydrant, the other left at the house, 0.5;
both set out, one within 91 mm of the mailbox, 0.75; both clear 1.0; both at the kerb in the stretch left of the
mailbox (too close to each other) 0.25 + 0.25 = 0.5; a cart with its handle to the street 0 for that cart; a cart in
the road 0 for the trial. Either cart may go on either side. The trial ends early once 1.0 has held for 3 s. A
simulator state with a non-finite number or a zero quaternion scores 0 for that step and leaves the tracker as it
was.

Where they fit: for a cart square to the kerb with its front 12 mm behind it, the only clear places are with the
centre of its body at y +12.8 to +16.8 cm (91 mm from the mailbox and from the hydrant) or at y -18.6 to -12.8 cm
(from the mailbox and from the car's corner), and each holds one cart (a second needs 149 mm). Past the hydrant the
verge is the driveway, and in front of the car a cart is at most 50 mm from it.

Unintended strategies earn nothing more than they should, as the tests show: both carts in one stretch earn 0.25
each (0.5); a cart stood on its front with the lid toward the street, upside down or on its back zeroes the trial; a
cart leaning on the kerb (its top over it) zeroes the trial; a cart stood on the other cart earns nothing and the
lower one only 0.25; a cart beside the car, against the post or the hydrant, or on the driveway earns 0.25, 0.25 or
0; both carts set 100 mm back from the kerb, clear of everything, earn 0. A cart held still 2 mm above the verge at
the right place counts as standing (the scorer sees only the carts; `task.json` fidelity says so).

## One arm alone

In simulation (the commands above, one trial each) the scripted demo scores 0.5 with the left arm idle and 0.5 with
the right arm idle, as the offline kinematic check (`--one-arm`) predicted: the idle arm's cart stays at the house,
side-on (0), and the other is set out clear of everything (0.5).

Reach (inverse kinematics of the arm's URDF with its link collision proxy, straight-down grips, every yaw in 15
degree steps within joint 6's +-120 degrees): with the tool point 68 to 78 mm up (low enough for the pads to hold
the body, high enough for the gripper's body to clear the lid), neither arm reaches any grip on a cart standing in
the clear place on the other arm's side, at any set-back from 0 to 30 mm and turned up to 25 degrees: 11 cm past the
midline an arm's reach ends at x 0.27 m, and a grip on such a cart needs x 0.29 m or more. Each arm can pick up the
other arm's cart where it starts, though. So with pick and place one arm alone can still earn 0.75: its own cart set
out (0.5) and the other cart stood at the kerb just past the mailbox on the far side (its centre about 4.5 cm past
the midline: standing, facing the street, at the kerb, but within 91 mm of the mailbox, 0.25), 137 mm from the first
cart; the arm reaches that place (tool point at x 0.31 to 0.34 m, 4 to 5 cm past the midline, 78 mm up; both arms
checked). The demo does not try it. Full marks with one arm would need the second cart pushed some 80 mm along the
kerb, beyond the arm's straight-down grasping reach, or a tilted grip (we checked straight-down grips only); we have
not tried either and do not claim it is impossible. Both arms are used naturally: each sets out the cart on its own
side, at the same time.

## Demo and run limit

The scripted demo takes 21.1 s (422 control steps), both arms at once. Each arm goes over its cart with the fingers
open (94 mm) and the grip turned to -90 degrees (left) or +90 degrees (right), comes down beside the body, takes the
last 30 mm over 1.5 s to the tool point 78 mm up, and grips across the 58 mm body at its middle (0.7 mm in front of
its centre of mass; commanded 50 mm, closing over 1.5 s), the pads on the body and lid from 65 mm up to the lid's
top. It lifts the cart 25 mm, carries it to the kerb over 3.5 s turning the wrist to 0 degrees (the cart turns a
quarter turn, from side-on to facing the street), lowers the last 23 mm over 2 s, settles, opens to 94 mm with the
cart 2 mm above the verge, rises straight up and parks. The places: body centres at x 34.15 cm, y +15.0 cm and
y -15.0 cm, fronts 12 mm behind the kerb face; clearances 113 mm to the mailbox, 109 mm to the hydrant, 125 mm to the
car's corner and 242 mm between the carts, at least 18 mm more than 91 mm everywhere. On the way, the lifted carts
stay at least 12 mm behind the kerb face, at least 97 mm from the mailbox, the hydrant and the car, and at least 180
mm apart (tested).

Offline (kinematic check): worst inverse-kinematics error 0.09 mm; smallest joint margin 15.1 degrees (joint 4);
arm links at least 7.3 cm apart (capsule proxy), tool points at least 26.0 cm apart; pad tips never lower than 65 mm
above the table. Success has held for its 3 s at step 367; the scripted policy stops at step 483. The run limit is
1,200 steps (1 minute), which leaves a policy room for slower carries, a set-down to turn a cart, or a regrip.

In simulation (L40S, one trial) the score reached 1.0 at about step 310 and the trial ended at step 369, after the
3 s hold. At the end the green-lidded cart stood with its front 11.5 mm behind the kerb face, turned 0.6 degrees,
108 mm from the hydrant, and the blue-lidded cart 13.2 mm behind, turned 0.9 degrees, 112 mm from the mailbox; both
upright and still on the verge. `TASK_TRACE` shows these values (`setback_mm`, `facing_err_deg`, `clear_mm`,
`origin_mm`) at every step.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand the green-lidded cart with the centre
of its body at x 22.0 cm, y +13.0 cm, its front toward -y, and the blue-lidded cart at x 22.0 cm, y -13.0 cm, its
front toward +y (fronts toward each other, 187 mm apart), both upright. The kerb, road, driveway, mailbox, hydrant
and car are fixed. Nothing is consumed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is 21 s of simulated time); the first start
of Isaac Sim on a new instance takes several minutes more.

## Real-world build

Two 1:10 model carts, 73 x 58 x 95 mm bodies with a 6 mm lid (one green, one blue) and a white arrow on it, a 6 mm
handle bar and two wheel blocks at the back (printed, or cut from foam board on a block), weighted to 120 g; a 10 x
12 mm strip of wood for the kerb, dark card for the road, pale card for the driveway, a 16 mm dowel 105 mm tall with
a 40 x 16 x 20 mm black block on top for the mailbox, a 24 mm red dowel 55 mm tall with an 18 x 10 mm cap for the
hydrant, and a 180 x 340 x 80 mm block with a 160 x 200 x 60 mm cabin block on it for the car, all taped or clamped
down at the positions in `task.json`. Every part is light and blunt, and nothing is hot or sharp, so trials are
safe. Score from a top photo: a tape square to the kerb for the set-back, a tape between outlines for the clearances
(or a 91 mm card spacer), a protractor on the lid arrow for the facing.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |
