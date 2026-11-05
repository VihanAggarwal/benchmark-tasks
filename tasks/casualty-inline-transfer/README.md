# Move a casualty manikin onto a stretcher, keeping its head in line with its body

When a spinal injury is suspected, rescuers move the head, neck and trunk as one unit (manual in-line stabilisation):
one rescuer controls the head, everyone lifts on a count, and the head is never allowed to drop or tip back while the
body moves. Here a simplified casualty manikin (about 0.4 scale in length, narrower so that a gripper spans it) lies
on its back on a grey floor mat: a rigid trunk and a rigid head joined by one free pin hinge across the body (the
neck, +-40 degrees, a light return spring). It must go onto a blue stretcher 30 mm tall, with its head on the white
head zone. The scorer reads the neck's hinge angle every control step and scores the worst bend over the whole move,
not only the final pose; it also checks that the head and the trunk go up and come down together, that neither body's
underside goes more than 8 cm above the table, and that no gripper holds the manikin by the neck. The task needs both
arms: one grip lifts one body and the other hangs from the free neck (the head drops, or the neck bends back while
the trunk drags).

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the manikin's state and the arms', with a small tracker for the whole trial |
| `mechkit.py` | Builds the manikin as a floating articulation (two bodies, one revolute joint) and reports its joint and link poses to the scorer |
| `rigstate.py` | Reads both arms (tool point, finger tips, opening) for the scorer before every control step |
| `tests/test_scoring.py` | Offline tests of every scoring tier, every zero rule, the reset hooks and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the `yam_arena`
embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install. `yam_arena`'s task files can only
declare single primitives, so `task.py` imports `mechkit.py` (next to it), which hooks `yam_arena.scene` when Isaac
Sim imports it and spawns the manikin from USD prims it writes: a loose articulation whose base body is the trunk
and whose one link, the head with its neck bar, turns on a revolute joint with a spring and damping (Isaac Lab
implicit actuator). `rigstate.py` wraps the embodiment's step and reset to read the arms. Neither file changes
`yam_arena` or what the embodiment does; both are loaded only from this folder.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# settle: nothing may move (both arms hold the ready pose)
DEMO_HOLD_ONLY=1 MECHKIT_PRINT_EVERY=20 inspect-robots run --embodiment yam_arena --policy yam_scripted -P hold_s=8 \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the manikin from the grey floor mat onto the blue stretcher, on its back with its head on the white head zone. Its neck is hinged: keep the head in line with the body the whole time (neck bent no more than 6 degrees), lift and lower the head and the body together, never raise its underside more than 8 cm above the table, and never hold it by the neck. Then let go of it with both grippers." \
  --scorer task_score --max-steps 200 --store-frames --log-dir ~/yam-eval/casualty-inline-transfer/settle

# scripted demo (both arms): expected 1.0
TASK_TRACE=trace.jsonl MECHKIT_TRACE=mechkit.jsonl inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the manikin from the grey floor mat onto the blue stretcher, on its back with its head on the white head zone. Its neck is hinged: keep the head in line with the body the whole time (neck bent no more than 6 degrees), lift and lower the head and the body together, never raise its underside more than 8 cm above the table, and never hold it by the neck. Then let go of it with both grippers." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/casualty-inline-transfer/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Move the manikin from the grey floor mat onto the blue stretcher, on its back with its head on the white head zone. Its neck is hinged: keep the head in line with the body the whole time (neck bent no more than 6 degrees), lift and lower the head and the body together, never raise its underside more than 8 cm above the table, and never hold it by the neck. Then let go of it with both grippers." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/casualty-inline-transfer/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module; it starts a new
record on every embodiment reset (from `rigstate`'s and `mechkit`'s reset hooks) and also when the simulator puts
the manikin back on its mark on the mat in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the one-arm
check); `DEMO_HOLD_ONLY=1` plays no plan (the settling check); `TASK_TRACE=trace.jsonl` writes one JSON line per
control step with what the scorer saw and decided (`neck_deg`, `worst_neck_deg`, `lift_trunk_mm`, `lift_head_mm`,
`under_trunk`, `on_stretcher`, `head_off_mm`, `left_clear`, `at_neck`, `breach`, the TCPs and grippers, ...);
`MECHKIT_TRACE=mechkit.jsonl` writes the manikin's joint, velocities and link poses per step, and
`MECHKIT_PRINT_EVERY=20` a summary line to stderr. After the first reset `mechkit` prints what Isaac Lab built
(bodies, joint, limits, stiffness, damping, masses). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Once per control step (20 Hz) the scorer reads the manikin's
trunk pose, its neck angle and the head link's pose (from `mechkit`) and the arms (from `rigstate`). Every height
and footprint is of a body's own boxes (the trunk; the head with its neck bar), never of the whole manikin's
bounding box. It keeps four facts about the trial: the worst neck bend so far, the first rule broken (if any),
whether both bodies were ever up 20 mm together with the neck within 10 degrees, and how often the arms could not be
read. A body is "lifted" when its lowest point is more than 5 mm above the surface under the middle of its outline
seen from above (stretcher top, mat or table): a body reaching half over the stretcher's edge while its middle is
still over the table is counted from the table. A state with NaN or inf in it, or a zero quaternion, scores 0 for
that step and changes nothing on the record.

| Outcome | Score |
|---|---|
| At any moment: neck bent over 20 deg; a body's lowest point over 80 mm above the table; one body more than 15 mm up while the other is not lifted; a tool point at the neck (within 25 mm along, 50 mm across, 60 mm above the neck bar's centre) while either body is lifted; a body's lowest point dropping faster than 0.25 m/s; a body more than 10 mm below the table top | 0 |
| On the stretcher, worst bend 6 deg or less, neck within 3 deg now, both grippers let go (open and finger tips 30 mm or more from the manikin) | 1.0 |
| On the stretcher, worst bend 10 deg or less, neck within 5 deg now | 0.75 |
| On the stretcher, worst bend 15 deg or less | 0.5 |
| On the stretcher with a larger worst bend; or not on it, but both bodies were once 20 mm up together with the worst bend then 10 deg or less | 0.25 |
| Anything else | 0 |

"On the stretcher": both bodies wholly over its top (every corner seen from above, to 1 mm), each lowest point
between 3 mm below and 4 mm above it, the trunk flat on its back (the face that was down at the start is down, tilted
10 deg or less; face down does not count), and the head block's centre within 25 mm of the head zone's centre (lying
wholly on the 110 mm wide top, the trunk is then within about 9 deg of the stretcher's axis, head toward the head
zone). The trial ends early once the 1.0 conditions have held with the manikin still for 1 s; both grippers must have
let go throughout that second, so it cannot end while a gripper still holds the manikin. Otherwise the score is read
as things stand at the run limit (stillness is not needed).

Unintended strategies earn nothing, as the tests show: lifting the trunk alone lets the head swing down on the free
neck (about 32 deg with its crown dragging on the mat at a 50 mm lift, 40 deg at the stop) and lifting the head alone
tilts the trunk about its far end; either breaks the "together" rule once the lifted body is 15 mm up (0). Lifting
the head first by just enough to clear the stretcher bends the neck only about 7 degrees, which is why the "together"
rule exists: without it one arm could move the head and then the trunk. Gripping the neck bar, the one part between
the bodies, is caught by the held-by-the-neck rule as soon as either body is lifted (0). Dropping the manikin onto
the stretcher from more than about 15 mm is a fall (0); a release from lower is not told apart from a set-down at 20
Hz. Sliding it across without lifting cannot climb the stretcher's 30 mm side (0). Ending with a straight neck does
not hide a bend on the way: a 12 degree bend during the carry caps the score at 0.5. A manikin put on the stretcher
end for end has its head off the head zone, and one put down face down is not on its back: neither is on the
stretcher.

## One arm alone

The task needs both arms. Each gripper can hold one body, and the neck between them is a free hinge.
`DEMO_IDLE_ARM=left` (the right arm alone lifts the trunk) lets the head swing down and drag on the mat: the trunk
is 15 mm up while the head is not lifted at about 9.7 s, and the neck passes 20 deg at about a 33 mm lift; it scores
0. `DEMO_IDLE_ARM=right` (the left arm alone lifts the head) tilts the trunk about its far end, which stays on the
mat: the head is 15 mm up alone at about 9.7 s, and it scores 0. These are the physical predictions; the offline
kinematic check cannot model the hinge (it carries the manikin as one rigid box with its joint at 0), so it predicts
1.0 for each arm alone. The GPU runs are recorded in `task.json`'s `scores` and below.

## Demo and run limit

Both arms run the same 21.0 s timeline (420 control steps), so the two bodies go up, across and down together. The
left arm holds the head across its 60 mm (fingers commanded to 52 mm), 15 mm from its centre toward the crown (the
fingers' 27 mm width ends 6.5 mm short of the crown); the right arm holds the trunk across its 70 mm (commanded to 62
mm), 30 mm from its centre toward the hips. Both close along x (yaw -90 deg left, +90 deg right) with the pad tips 12
mm above the table, so the open fingers stand in front of and behind the manikin. The grips are 0.230 m apart at one
x: the arms spend 8 s at x 0.20 (down, grip, lift), where the wrists' (link4) collision capsules come within 6 mm of
each other at 0.205 m apart (the head's and the trunk's centres plus 20 mm) and 23 mm at 0.220 m; at 0.230 m they
stay 34 mm apart, and 64 mm over the stretcher. Each gripper closes over 1.5 s after a 0.5 s settle; both lift 50 mm
over 2.0 s (undersides 23 mm over the stretcher's top, 27 mm under the 80 mm cap), carry 165 mm straight out along x
over 3.0 s, come down onto the stretcher over 2.0 s (undersides aimed 1 mm above its top), settle, open over 1.5 s,
settle 1 s and rise to 14 cm above the table over 2.0 s, until the finger tips are 53 mm over the head and 57 mm over
the trunk. The offline check reaches success at step 426; the scripted policy stops at step 481 (3 s hold). The run
limit is 1,200 steps (60 s).

The arms' controllers do not carry a payload's weight, and the right arm holds the heavier body (0.22 kg against 0.07
kg), so it is commanded 2 mm higher once loaded (`SAG_TRIM_R` in `task.py`) to keep the bodies level. A quasi-static
estimate from the arms' joint stiffnesses (80 N m/rad at the shoulder and elbow, 10 N m/rad at the wrist) has each
TCP sagging about 1.6 mm per newton at x 0.20 and 2.7 mm per newton over the stretcher, so the trunk's arm sags 2.5
to 4.4 mm more than the head's and the trim leaves under 2.5 mm. With about 9 N m/rad of stiffness about x at each
TCP, that mismatch and the off-centre grips (0.065 N m on the trunk, 0.013 N m on the head) bend the neck under 1 deg
during the carry; the 1.0 limit is 6 deg.

## First runs on the GPU

This is the first task to move a floating articulation (the kit's smoke test built one and left it alone), so run
the settle check, then `DEMO_IDLE_ARM=right` for a one-arm lift, before the demo. In `TASK_TRACE` / `MECHKIT_TRACE`:
in the settle run the neck stays within 0.01 rad of 0 and the trunk's origin moves under 1 mm; in the demo,
`worst_neck_deg` stays under 6 and `lift_trunk_mm` and `lift_head_mm` stay within a few mm of each other during the
carry. Tuning knobs, demo only: `SAG_TRIM_R` (equalise the two lifts), the grip height `GRIP_Z`, and, if the hinge
jitters, `NECK_DAMPING` 0.02 or `solver_iterations=32` on the assembly (then update `setup`'s damping).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the manikin flat on its back on the grey mat
along y, trunk centred at x 20.0 cm, y -8.5 cm, head toward +y, neck straight; the mat and stretcher are fixed.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 24 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A simplified manikin from two closed plastic boxes: a 70 x 240 x 40 mm trunk of 0.22 kg and a 60 x 70 x 44 mm head of
0.06 kg on a 24 x 30 x 20 mm neck bar (0.07 kg together), joined by a pin hinge along the trunk's width at its
mid-height, with a soft return spring (a light torsion spring or an elastic band) and stops at +-40 degrees;
skin-coloured paint, the neck darker, BACK written on the trunk's underside. A printed protractor on the side of the
neck bar shows the hinge angle on video. A 120 x 380 x 3 mm grey rubber mat and a 110 x 400 x 30 mm board painted
blue with a white band 70 mm wide 30 mm from one end as the stretcher, both taped down. Stand a card with a
millimetre scale upright behind the mat and the stretcher to read heights (5, 15, 20 and 80 mm above the surface
under a body) on a side video; the 0.25 m/s fall speed is read from the video's frame rate.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
