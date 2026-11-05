# Bridge the dock gap with two runner planks, then push the cart onto the truck bed

Between a loading dock and a trailer there is a gap that a cart crosses only on a dock board or plate laid and
secured first: a cart run at an open gap drops a wheel and tips, and an unsecured plate slides out under the load.
Here a grey dock on the right arm's side and a brown truck bed on the left arm's side face each other across an
80 mm gap. Two loose aluminium runner planks lie on the table beyond the bed, and a blue cart (a 100 x 100 x 30 mm
block, too wide for the gripper) stands on the dock. The runners must go into their seats (rebates) on both edges
of the gap, the cart must be pushed over them onto the bed between two yellow marks, never lifted, and both
grippers must end clear of everything. The scorer watches the whole trial, not only where the cart ends up: a cart
pushed at an open gap drops into it, a cart that comes over the gap before both runners are seated (even held up),
a lifted cart and a runner sliding more than 10 mm under the cart all score 0. The layout splits the work: with the
grippers pointing down, only the left arm reaches the runners, only the right arm reaches the cart's rear face at
the start, and the right arm can push the cart only just past the gap, so the left arm takes the push over. Tilted
far over, at the edge of its reach, an arm gets at more of the other's side, so the relay is the natural way, not
the only one (see "One arm alone").

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the cart's, the runners' and the grippers' states, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `offline_push.py` | Offline checks only: a small model of a gripper pushing the cart (the simulator never calls it; see below) |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook, the push model and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed and loose boxes) and the standard `task_score` scorer.

The 1.0 tier needs both grippers clear of the cart and the runners at the end, and `yam_arena` does not pass the
grippers' state to a task file's hooks. `rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset
once, at import, to take a snapshot of both arms (tool point, finger-tip point, finger opening, in the objects'
frame) before each control step and to start a new trial at each reset. It reads only state the simulator already
has and changes nothing the embodiment does. If it cannot read the arms it prints why once; the hands-off condition
is then taken as met on those steps, and the trace counts them (`arms_unavailable`).

`offline_push.py` is not used in the simulator. Our offline checker (NumPy kinematics, no physics) carries what a
gripper closes on but cannot push, so `task.py` gives the cart's spec an `offline_state` hook (a plain attribute the
simulator never reads) through which the checker gets the cart's pose from a small quasi-static push model: a
closed gripper's fingers move the cart along y out of their way, and the cart stays up only while its centre lies
over the dock, the seated runners or the bed, otherwise it falls into the gap.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted -P hold_s=3 \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the two grey runner planks across the gap between the loading dock and the truck bed, each into its pair of seats on the dock edge and the truck-bed edge. Then push the blue cart from the dock over the runners onto the truck bed, between the two yellow marks. Never lift the cart: slide it along the surfaces. Leave both runners in their seats and let go of everything." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dock-bridge-relay/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lay the two grey runner planks across the gap between the loading dock and the truck bed, each into its pair of seats on the dock edge and the truck-bed edge. Then push the blue cart from the dock over the runners onto the truck bed, between the two yellow marks. Never lift the cart: slide it along the surfaces. Leave both runners in their seats and let go of everything." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/dock-bridge-relay/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`seat_a`, `seat_b`, `both_seated_step`, `cart_low_mm`,
`min_cart_low_mm`, `max_cart_low_mm`, `cart_tilt_deg`, `max_cart_tilt_deg`, `cart_yaw_err_deg`, `cart_x_off_mm`,
`cart_y_mm`, `cart_rear_y_mm`, `centre_off_mm`, `on_bed`, `cart_over_gap`, `handover_step`, `cart_on_runner`,
`runner_shift_mm`, `max_runner_shift_mm`, `runner_low_mm`, `rear_minus_tcp_mm` (the cart's rear face minus each
TCP's y: 8 mm while the closed fingers push), `left_zone_mm`, `right_zone_mm`, `opening`, `hands_off`, `arms_unavailable`, `breaches`,
`first_breach_step`, the three objects' poses, both TCPs and tip points). Videos: `inspect-robots video <log>.json`.
The scorer keeps one trial's record per process and starts it over at every reset, so run one trial per process
or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v` (about four minutes: one test plays the
whole demo and both one-arm variants through the offline checker).

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. A runner is seated when it lies flat, the corners of one end
are inside a dock seat and those of the other inside the facing bed seat of the same pair, and its underside is on
the seat floors (within 2 mm) at both ends; the two runners must be in different pairs. The cart is on the bed when
its whole outline is over the bed top, clear of the seats, with its underside on it (within 2 mm). Six things
score 0 the moment they happen, whatever follows: any part of the cart more than 1 mm over the gap (y -4.0 to
+4.0 cm) while the runners are not both seated (the path used before it is built, even if the grippers hold the
cart up over the gap), the cart's lowest point below 35 mm above the table (it dropped
into the gap; on the dock it is at 40 mm, on the runners 39.5, on the bed 39) or above 45 mm (lifted), the cart
tilted more than 10 degrees, a runner's end moving more than 10 mm while the cart is on that runner (measured from
where it was when the cart came onto it), or the cart or a runner falling off the table.

| Outcome | Score |
|---|---|
| The cart over the gap before both runners are seated; a dropped, lifted or tipped cart, a runner slid under it, or anything off the table, at any step | 0 |
| Both runners seated; the cart on the bed, its centre within 15 mm of its mark (x 32.5 cm, y +21.0 cm), square within 5 degrees; both grippers' finger zones 30 mm or more from the cart and the runners | 1.0 |
| Both runners seated; the cart on the bed, its centre within 40 mm of its mark (turned, or a gripper still at it) | 0.75 |
| Both runners seated; the cart's centre at y +4.0 cm or beyond (over the bed's side of the gap) | 0.5 |
| Both runners seated | 0.25 |
| Anything else | 0 |

The scorer cannot see which way a gripper is turned, so each gripper's fingers count as a round zone about its
axis, out to the fingers' outer corners (a finger is 10 mm thick and 40 mm wide over its lowest 38 mm: 60 mm fully
open, 22 mm closed), from the finger tips to 38 mm back up the gripper. A gripper closed again well clear of everything is clear. The score is read at the last step; the trial
ends early once the 1.0 conditions have held with the cart and both runners still (under 1 cm/s and 0.5 rad/s) for
1 s. A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts an object
back in a single step and all three are then at their starting places (within 4.5 cm, which covers a seeded
reset's jitter). A simulator state with a non-finite number, or a missing object, scores 0 for that step and
leaves the record of the trial as it was. The cart and the runners are plain boxes, so either broad face may be
down.

Unintended strategies earn nothing more than they should, as the tests show: pushing the cart at the open gap drops
it (0, and laying the runners afterwards does not undo it); carrying it across raised 10 mm is 0 (lifted); holding
it up over the open gap, or over one seated runner, at the dock's height and laying the runners once it is across is
0 (open gap); standing the runners on edge in the gap, flush with the dock, lets the cart across but leaves them out
of their seats, and seating them afterwards does not help (0, open gap); dragging the cart across on a runner that
slides with it is 0; a runner knocked out of its seats after the cart has crossed loses the tier (0); a cart pushed
only partway onto the bed scores 0.5; laying the runners alone scores 0.25. Adjusting a runner in its seats before
the cart comes onto it is allowed.

## One arm alone

With the grippers pointing straight down each arm can do only its own part. Straight down at the runners' grip
height, at any heading, the right arm reaches no point beyond y +20.0 cm at x 10 cm, +19.0 cm at x 15 cm and
+17.5 cm at x 18 cm; the nearer runner's end lies at y +21.0 cm and the other runner beyond it (inverse kinematics
of the arm's URDF with joint limits and a collision proxy, 40 random starts). At the push height the left arm reaches
no point beyond y -7.5 cm at x 32.5 cm (-10 cm at the table at x 30 cm), while the cart's front face starts at
y -15.0 cm and its rear face, where a push toward the bed must bear, at y -25.0 cm. So with the left arm idle
(`DEMO_IDLE_ARM=left`), the right arm can only push the cart at an open gap: 0 (it drops in, and it came over the
gap before the runners were seated). With the right arm idle (`DEMO_IDLE_ARM=right`), the left arm lays both
runners and never reaches the cart: 0.25. The offline check predicts exactly these (0 and 0.25).

Tilted far over, an arm reaches further, and we do not claim that one arm alone could never do more. Searching any
gripper orientation (the arm's links and the closed fingers kept clear of the table, the dock, the bed, the cart and
the runners), the right arm reaches runner A's near end tilted 10 degrees and its middle tilted 30, runner B's near
end only tilted 50 to 60 degrees with 2 mm to spare, and the cart's rear face with the cart's centre at about
y +21 cm (its mark) tilted 30 to 60 degrees with under 1 mm to spare; the left arm reaches behind the cart's rear
face at the start only with its gripper almost horizontal (80 degrees), reaching round the cart's far side, with
4 mm to spare. So a single arm working at the edge of its reach, steeply tilted throughout, might lay both runners
and work the cart across; the scripted one-arm runs show what an arm does with its own side's moves.

The relay is the natural way for the last leg too. Pushing from behind with the gripper straight down, the right
arm's tool point reaches y +7.5 cm at x 32.5 cm (+9.5 cm at x 30 cm), which brings the cart's centre to about y +13
cm: past the gap (0.5) but not onto the bed, which needs its centre at y +13.5 cm or more, nor near its mark at y
+21 cm. The left arm can reach the cart's rear face only once the cart's centre is past about y -2 cm (its rear face
at y -7 cm), so it takes over at the gap.

## Demo and run limit

The scripted demo takes 75.5 s. The left arm grips runner B (the far one) at its middle across its 40 mm width
(fingers commanded 8 mm narrower, over 1.5 s), the tool point 17 mm above the table (pad tips 4 mm), at yaw +90
degrees: 15 cm outboard of the left base only that heading reaches. It lifts it 58 mm clear, carries it over the bed
turning it to yaw -90 in two quarter turns (the same grip of a symmetric plank; a half turn in one move would leave
the interpolation's direction undefined), lowers it to 5.5 mm above the dock and then over 2 s the last 25 mm into
the far seats (x 36.5 cm), and opens to 64 mm, which keeps the pads 20 mm from the next seat's runner. Runner A goes
into the near seats (x 28.5 cm) the same way at yaw -90. Meanwhile the right arm closes its fingers and comes down
4 mm behind the cart's rear face, the tool point 60 mm above the table (pad tips 47 mm, 7 mm over the dock). Once
both runners are in and the left arm has parked out of the way with its fingers closed, the right arm pushes the
cart 26.7 cm along y over 8 s (peak 50 mm/s) until its rear face is at y 0 (its centre over the bed's side of the
gap), backs off 40 mm, rises and parks. The left arm then comes over the cart to 32 mm behind its rear face, down to
the same push height, pushes it 20 cm over 7 s onto its mark, backs off 40 mm, rises, moves away (never back over
the cart) and opens. Both pushes are made with the fingers closed at yaw 0, at x 32.5 cm, so the 27 mm wide fingers
pass between the runners and 7 mm above everything. The TCPs stay at least 23.5 cm apart and the arm links (capsule
proxy) at least 6.4 cm. Offline, success has held for its 1 s at step 1442 (it starts once the left gripper's closed
finger zone is 30 mm clear of the cart, during its rise); the scripted policy would stop at step 1571 (with
`-P hold_s=3`). The run limit is 2,400 steps (2 minutes).

The arms trail a push: the joints are position-controlled springs (80/80/80/10/10/10 N m/rad, no payload
compensation), and the compliance model (J K^-1 J^T at the fingers) puts a pushing gripper 9 to 13 mm behind its
command under the cart's sliding friction (2.06 N; 2.5 N to start it) and pitches it about 2.5 degrees with the tips
trailing, so the fingers bear on the cart's top rear edge, 30 mm up. The cart is 30 mm tall so that even the
breakaway force there cannot tip it (a 100 mm long block on friction 1.2 tips only when pushed 42 mm or more above
its base; a 50 mm tall cart would). Each push is commanded past its goal by the modelled trail at its end: 12.5 mm
for the right arm, 10 mm for the left. The cart's final place is set by the left arm's push alone and lands within
15 mm of the mark for any trail from 0 to 25 mm; the left arm comes down 32 mm behind the expected rear face, 20 mm
clear even if the right arm trailed 25 mm. On the GPU, read `cart_rear_y_mm` and `rear_minus_tcp_mm` in
`TASK_TRACE` at the end of each push and tune `TRAIL_R` and `TRAIL_L` in `task.py` if they are off.

Things to look at in the GPU runs: in the settle run nothing moves; the runners enter their seats (5 mm of play at
each side and end) and `seat_a`/`seat_b` read "a"/"b"; `min_cart_low_mm` stays at 38 or more over the seams (the
path steps down 0.5 mm at each one so a sliding block never meets a vertical edge; a lower value means PhysX caught
an edge); `max_cart_tilt_deg` stays small; `cart_x_off_mm` stays within about 10 mm and `cart_yaw_err_deg` under 5
(a single 27 mm finger face pushes the cart; if it turns, push 5 mm lower or with the fingers opened to 20 mm);
`max_runner_shift_mm` stays under 5 (the cart drags each runner toward the bed seat's end, 5 mm away);
`arms_unavailable` is 0 (otherwise hands-off silently passes: do not submit).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. The dock's and the bed's main blocks span the cart's
whole path in x, so no seam crosses the path at the cart's level; the seats, the rib and the cheeks are separate
fixed blocks at the gap edges. Reset: both runners back flat on the table along y, end to end, centred at x 15.0 cm,
y +29.0 cm and x 16.0 cm, y +46.0 cm, and the cart back on the dock square to it, centred at x 32.5 cm, y -20.0 cm;
the dock, the bed, the seats and the marks are fixed.

## Runtime per trial

About 2 to 3 minutes on an L40S once Isaac Sim is running (the demo is 72 s of simulated time until success ends it);
the first start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 40 mm and a 39 mm thick board (the dock and the truck bed, 180 x 260 mm each) screwed to a baseboard 80 mm apart,
each with two 50 mm wide, 45 mm deep rebates cut 20.5 mm deep at the facing edges, leaving a 30 mm rib between them
(cut down to 29.5 mm) and full-height cheeks outside; two 160 mm lengths of 40 x 20 mm aluminium rectangular tube
(about 60 g each) as the runners; a 100 x 100 x 30 mm hardwood block painted blue (about 0.21 kg) as the cart; a
rubber mat on the board tops so the block drags as in the simulation; two strips of yellow tape on the bed; 10 mm
scale marks along the seats; and a card with lines 35 mm and 45 mm above the table standing beside the gap, out of
the arms' way. Film from the side to read the drop, lift and tilt rules, and from above for the runners and the
end state.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
