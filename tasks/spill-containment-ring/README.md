# Ring a spill with absorbent socks, never carrying a sock or fingers over it

A liquid spill is first contained by "diking": responders ring it with absorbent socks laid end to end, working from
the outside in, never step or kneel in the spill, and never reach a gloved hand or carry a sock over it. Here a dark
disc on the midline stands in for a chemical spill, and two yellow spill-kit trays, one in front of each arm, each
hold two white absorbent socks (rigid bars 200 x 30 x 30 mm). The four socks must end flat on the table round the
spill, close to its edge, with no gap in the ring wider than 10 mm. The scorer also watches the whole trial, not
only its end: no sock and no gripper's fingers may ever come over the spill or within 5 mm of its edge, seen from
above, at any height. Working right at the edge of a forbidden zone without crossing it is the point of the task.
(The rule is about the socks and the fingers, the parts that handle the absorbent; the gripper's body and the arm
above them are not judged.)

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the four socks' states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, stillness, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (a fixed cylinder, fixed and loose boxes) and the standard `task_score` scorer.

The scorer's gripper rule needs the grippers' state, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; the gripper part of the rule is then not applied on those
steps, and the trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Ring the dark spill with the four white absorbent socks from the yellow trays: lay them flat on the table end to end round it, close to its edge, leaving no gap wider than 1 cm. Never let a sock or a gripper's fingers come over the spill or within 5 mm of its edge, even up in the air." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/spill-containment-ring/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Ring the dark spill with the four white absorbent socks from the yellow trays: lay them flat on the table end to end round it, close to its edge, leaving no gap wider than 1 cm. Never let a sock or a gripper's fingers come over the spill or within 5 mm of its edge, even up in the air." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/spill-containment-ring/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `placed`, `ring_closed`,
`ring_joints_mm`, `group_span_deg`, `sock_clear_mm`, `min_sock_clear_mm`, `finger_clear_mm`, `min_finger_clear_mm`,
`sock_tilt_deg`, `sock_low_mm`, `opening`, `grippers_let_go`, `arms_unavailable`, both TCPs, ...). Videos:
`inspect-robots video <log>.json`. The scorer keeps one trial's record per process and starts it over at every
reset, so run one trial per process or rely on the reset; the run rate is assumed to be 20 Hz
(`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Shapes are judged from above. A sock's outline is the
outline of its real box seen from above, turned or tilted as it is (not its world-aligned bounding box, which for a
200 mm sock carried at 45 degrees is far larger than the sock). The keep-out circle is the spill's edge expanded by
5 mm: radius 65 mm. The scorer cannot see which way a gripper is turned, so each gripper's fingers count as a round
zone about its axis, out to the fingers' outer corners (60 mm fully open, 40 mm with the fingers 50 mm apart, 32 mm
closed on a sock), from the finger tips to 38 mm back up the gripper. Those radii use the fingers' size 19 to
38 mm above their tips on `yam_arena`'s finger collision meshes (10 mm thick outside the gap and 40 mm wide; at the
tips they are 6.4 mm and 27 mm), so the zone holds the lowest 38 mm of both fingers at any heading. Higher up the
fingers widen into their roots and carriages inside the gripper's body, and the body and the arm are wider still:
none of that is judged. In the demo the gripper's body and the fingers' roots come up to 6 mm inside the keep-out
circle (to about the spill's edge) 8 to 17 cm above the table, and each arm's wrist passes over the spill about
25 cm up while it works on the side of the ring nearer its base (measured offline on the same collision meshes).

One rule holds for the whole trial; broken at any moment it scores 0, whatever follows: no sock's outline and no
finger zone may come inside the keep-out circle, at any height. Carrying a sock over the spill, reaching across it,
or opening the fingers wide next to it all count.

A sock is placed when it lies flat on the table (one of its four long faces down within 5 degrees, its lowest point
within 3 mm of the table top: not on a tray or on another sock) with its outline within 60 mm of the spill's edge.
The gap between two socks is the shortest distance between their outlines. Placed socks close a ring when all of
them can be taken in a cycle, each within the joint limit of the next, whose loop (each sock's centre, the nearest
points of each joint, the next sock's centre, ...) goes round the spill's centre; every segment of that loop lies
inside a sock or crosses a joint, so any way out from the spill crosses a sock or a joint no wider than the limit.
This needs no fixed places or order: any sock may go anywhere, in any ring shape. With all four placed, all four
must be in the ring: three socks can only just close one (joints of about 9, 9 and 24 mm, each sock within 1 mm of
the keep-out circle), and with the fourth lying elsewhere that scores 0.5, not 0.75.

| Outcome | Score |
|---|---|
| A sock or a finger zone came inside the keep-out circle at any moment | 0 |
| All four socks placed and together closing a ring round the spill, every joint 10 mm or less, both grippers let go | 1.0 |
| All four placed and together closing a ring with every joint 25 mm or less, at most one over 10 mm (or the 1.0 ring with a gripper not let go) | 0.75 |
| Three or more placed socks, linked into one group by gaps of 25 mm or less, covering at least 250 degrees round the spill's centre (three sides of the square cover about 277) | 0.5 |
| At least two socks placed | 0.25 |
| Anything else | 0 |

A gripper has let go when it is open (opening 0.9 or more, fingers at least 84.5 mm apart) or its finger zone is
more than 10 mm from every sock, so a gripper closed again once it is clear changes nothing, while one still closed
round a sock in the ring scores 0.75.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with all four socks
still (under 1 cm/s and 0.5 rad/s) for 1 s. A new trial is recognised from rigstate's reset hook, or, as a
fallback, when the simulator puts every sock back on its starting place (within 4.5 cm, which covers a seeded
reset's jitter) in a single step that moved one more than 5 cm further than its velocity allows. A simulator state
with a non-finite number scores 0 for that step and leaves the record of the trial as it was. The socks have a
square section, so any of their four long faces may be down.

Unintended strategies earn nothing or little, as the tests show: one arm reaching straight across the spill to the
other side breaks the rule (0); carrying a sock over the spill at any height, sliding one across it, or dropping one
on it and pulling it back out breaks it (0); gripping a sock level with the spill's centre instead of toward its
corner end brings the fingers inside (0); opening the fingers fully at the ring reaches inside (0); a straight row
of socks along one side (0.25) or two doubled walls left and right with all four placed (0.25) is not a ring; three
socks covering only two sides, one of them doubled, are not three-quarters of one (0.25); nor are two corners, each
a joined pair, apart from each other by joints over 25 mm (0.25); three socks closing a ring with the fourth
elsewhere are not a ring of four (0.5); socks stacked on others, stood on end, or with one end propped on a
neighbour are not placed; a closed pinwheel moved off the spill does not go round it. Each sock up to 2 mm off its
place on its own, or all of them 5 mm off the same way (arm sag), still scores 1.0.

## One arm alone

The demo splits the work by reach: each arm lays the two socks from its own tray. `DEMO_IDLE_ARM=left` (the right
arm alone) lays the right side and the front of the ring and leaves the left tray full: 0.25. `DEMO_IDLE_ARM=right`
(the left arm alone) lays the left side and the back: 0.25. Two sides of the ring are not three-quarters of it
(they cover about 187 degrees). Gripping from above, neither arm reaches the other tray: its straight-down reach
ends about 18 cm past the midline near the table, and straddling the nearest sock on the other tray needs the
gripper's axis 20.5 cm past it. We do not claim one arm is strictly unable to finish: tipped 60 to 80 degrees
sideways, an arm's tool point can reach over the other tray (position and orientation IK, with the arm's collision
capsules clear of the table and its own base), so a policy able to grip or drag a sock with a sideways-pointing
gripper near the table could in principle fetch the other tray's socks itself. The scripted demo grips from above
only. Both arms working at once is also simply faster: four placements, two at a time. (Offline kinematic
predictions; the GPU runs will be recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 26.8 s (536 control steps), both arms in parallel on one clock. The ring is a pinwheel: each
sock's end stops 5 mm short of the inner side of the next sock, and its other end is flush with the next sock's
outer side, so the four close a square with 5 mm joints and no corner gaps; the socks' inner sides are 87.5 mm from
the spill's centre (22.5 mm outside the keep-out circle). Each arm first lays the side of the ring on its own side
of the spill, then the end of the ring nearer its own corner. Every sock is gripped across its 30 mm width (fingers
commanded 8 mm narrower, over 1.2 s) with the finger tips 6 mm above its bottom, 40 mm from its centre toward the
end that sits at a corner of the ring; there the gripper is 117.5 mm from the spill's centre, and the finger zone,
with the fingers opened only to 50 mm near the ring, stays 12 mm outside the keep-out circle (fully open it would
reach 8 mm inside it). Each arm lifts its sock until its bottom is 6 cm above the table, turns the second one a
quarter turn about the grip while still over its tray, carries it across, comes down to 2 cm above the table,
settles, lowers the last 2 cm over 1.5 s, lets go by opening to 50 mm with the sock 2 mm above the table, and goes 8
cm straight up, which takes its finger zone 17 mm clear of the sock (let go). The TCPs stay at least 23.5 cm apart.
Offline, success has held for its 1 s at step 552. The run limit is 2,400 steps (120 s): four socks to carry and lay
precisely, two at a time at best.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the four socks back flat on their trays,
length along x (sock 1 at x 29.75 cm, y +23.5 cm; sock 2 at x 21.75 cm, y +30.5 cm; sock 3 at x 26.25 cm,
y -23.5 cm; sock 4 at x 42.25 cm, y -30.5 cm); the spill and trays are fixed.

## Runtime per trial

About 1 to 1.5 minutes for the demo on an L40S once Isaac Sim is running (27 s of simulated time), and up to
about 3 minutes for a policy that runs to the 2,400-step limit; the first start of Isaac Sim on a new instance
takes several minutes more.

## Real-world build

A 120 mm disc of dark vinyl or painted 1 mm board taped to the bench as the spill, with a circle drawn 5 mm outside
its edge (radius 65 mm) for judging the rule from top video; two yellow plastic boards, 310 x 130 mm and 390 x 130
mm and 8 mm thick, as the spill-kit trays; and four 200 x 30 x 30 mm absorbent socks: white fabric sleeves packed
firmly with polypropylene absorbent to about 50 g, or stiff foam bars of that size and weight wrapped in white
cloth. Film from above and measure the joints with a 10 mm and a 25 mm gauge block. From above the fingers are
hidden under the gripper's body, so judge the finger zones from the robot's logged joint states (the same tool
point, finger-tip point and opening the scorer reads, by forward kinematics) and the socks from the top video.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.25 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.25 |
| Dream-1 | 0.0 |
