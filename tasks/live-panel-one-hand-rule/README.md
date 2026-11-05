# Change a fuse in a live panel with one gripper while the other stays parked

Electricians who must reach near exposed energised parts, for example to change a fuse in a panel that cannot be
switched off, keep to the "one-hand rule": one hand works inside the enclosure and the other is kept well away, so
a fault current cannot pass from hand to hand across the chest. Here a live panel (a dark plate with a yellow
border) stands in front of the left arm, with a red fuse in a porcelain socket between two copper busbars. The green
spare stands in a blue cup in front of the right arm, a white hand-off pad lies on the midline outside the panel, a
black discard tray stands in front of the left arm, and each arm has a grey parking square in front of its base.
The first gripper to go inside the panel becomes the panel hand: the other may never go inside, and whenever the
panel hand is inside, the other must wait in its own parking square. Only the right arm reaches the spare and only
the left arm reaches the socket, so the second arm is needed, but only outside the panel and only at the right
moments. The scorer watches both grippers and both fuses at every control step, not only the end state.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the two fuses' states and the grippers' state, with a tracker for the whole trial |
| `rigstate.py` | Reads both grippers (tool point, finger-tip point, opening) for the scorer each control step |
| `tests/test_scoring.py` | Offline tests of every tier, every rule, resets, the gripper hook and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes, loose cylinders) and the standard `task_score` scorer.

The one-hand rule needs the grippers' positions, which `yam_arena` does not pass to a task file's hooks.
`rigstate.py` (imported by `task.py`) wraps the embodiment's step and reset once, at import, to take a snapshot of
both arms (tool point, finger-tip point, finger opening, in the objects' frame) before each control step and to
start a new trial at each reset. It reads only state the simulator already has and changes nothing the embodiment
does. If it cannot read the arms it prints why once; the gripper rules are then not applied on those steps, and the
trace counts them (`arms_unavailable`).

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The panel inside the yellow outline is live. Replace the red fuse in its socket with the green spare from the blue cup, and put the red fuse in the black tray; the spare may be set down on the white pad. Only one gripper may ever go inside the yellow outline, and while it is inside, the other gripper must wait in its own grey parking square, less than 10 cm above it." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/live-panel-one-hand-rule/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "The panel inside the yellow outline is live. Replace the red fuse in its socket with the green spare from the blue cup, and put the red fuse in the black tray; the spare may be set down on the white pad. Only one gripper may ever go inside the yellow outline, and while it is inside, the other gripper must wait in its own grey parking square, less than 10 cm above it." \
  --scorer task_score --max-steps 1600 --store-frames --log-dir ~/yam-eval/live-panel-one-hand-rule/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`breaches`, `first_breach_step`, `panel_hand`,
`first_entry_step`, `left_in_panel`, `right_in_panel`, `left_parked`, `right_parked`, `min_right_to_panel_mm`,
`min_park_margin_mm`, `green_seated`, `green_socket_off_mm`, `green_low_mm`, `green_tilt_deg`, `red_in_tray`,
`red_out_step`, `arms_unavailable`, both fuses, both TCPs and tip points, ...). Videos: `inspect-robots video
<log>.json`. The scorer keeps one trial's record per process and starts it over at every reset, so run one trial
per process or rely on the reset; the run rate is assumed to be 20 Hz (`-E control_hz=20`).

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads each gripper's tool point, the point
between its finger tips (13 mm past the tool point) and its opening, but not its heading, so the fingers are taken
to fill a round zone about the gripper's axis: a flat-ended cylinder from the tip point to 38 mm back up the
gripper, out to a finger's outer corner (57 mm from the axis fully open, 23 mm closed on a fuse). A gripper is
inside the panel when any part of that zone is over the panel's 160 x 160 mm outline and less than 15 cm above the
table top, so open fingers hanging over the yellow border count even when the gripper's axis is outside it; the
first gripper inside is the panel hand. A gripper is parked when its tool point is over its own 100 x 100 mm parking
square and less than 10 cm above the table top. Three rules hold for the whole trial; one broken at any moment
scores 0 whatever follows:

1. once the panel hand is known, the other gripper never goes inside the panel (both going in on the same control
   step breaks it too);
2. on every control step on which the panel hand is inside the panel, the other gripper is parked;
3. neither fuse leaves the table (its lowest point more than 10 mm below the table top).

| Outcome | Score |
|---|---|
| Any rule broken at any moment | 0 |
| Green fuse seated, red fuse in the discard tray, neither gripper inside the panel | 1.0 |
| Green fuse seated, red fuse clear of the panel, but not in the tray or a gripper still inside the panel | 0.75 |
| Red fuse out of the socket, green fuse inside the panel (not seated, or seated with the red fuse still in the panel) | 0.5 |
| Red fuse in the discard tray | 0.25 |
| Anything else (red fuse still in the socket, or set aside outside the tray with the spare not in the panel) | 0 |

Seated: the fuse's axis within 10 degrees of vertical (either end down: the fuses are plain cylinders), its lowest
point within 3 mm of the socket's floor and its centre within 8 mm of the socket's centre (anywhere in the 30 mm
opening). In the tray: centre over the tray's 80 x 80 mm floor, lowest point within 10 mm of the table (standing
or lying). Clear of the panel: no part of the fuse's outline seen from above over the panel's outline.

The score is read at the last step; the trial ends early once the 1.0 conditions have held with both fuses still
for 1 s. A new trial is recognised from rigstate's reset hook, or, as a fallback, when the simulator puts both fuses
back in their starting places (within 4.5 cm) in a single step. A simulator state with a non-finite number scores 0
for that step and leaves the record of the trial as it was.

Unintended strategies earn nothing, as the tests show: the right gripper reaching into a corner of the panel at any
time, even after the job is done, or only its open fingers reaching over the border, breaks rule 1 (0); the right
arm holding the spare just outside the panel's edge, ready for the left to take, while the left works inside breaks
rule 2 (0); the left arm going in while the right arm is still at its ready pose, hovers 13 cm above its square,
waits low somewhere else, or lifts out of its square for a moment while the left is inside breaks rule 2 (0); a
gripper that goes in first, even briefly, becomes the panel hand, so the other can no longer go in (0); sweeping the
red fuse off the table scores 0. The rule binds only while the panel hand is inside: the right arm may leave its
square while the left arm is outside the panel (that trial still scores 1.0), and a gripper whose finger tips stay
above the 15 cm line is not inside.

## One arm alone

`DEMO_IDLE_ARM=left` (the right arm alone) stages the spare on the hand-off pad and parks; nobody touches the red
fuse: 0. `DEMO_IDLE_ARM=right` (the left arm alone) goes into the panel while the idle right arm is at its ready
pose, not in its parking square, which breaks rule 2 at once: 0. With a straight-down grip the right arm cannot
reach the socket or the tray (the socket is 15 cm past the midline; the right arm's reach ends about 10 cm past it)
and the left arm cannot reach the spare's cup (35 cm past the midline), so neither arm can finish alone: with the
right arm parked first, the left arm alone can at most put the red fuse in the tray (0.25, tested). A strongly
tilted gripper reaches further, but the one-hand rule still binds: whichever arm works inside the panel, the other
must be parked meanwhile. (The GPU results are in 'Results in simulation' below.)

## Demo and run limit

The scripted demo takes 51.9 s (1,038 control steps). Every grip closes the fingers along y over 1.5 s, commanded 8
mm narrower than the 20 mm fuse, with the finger tips 33 mm above the fuse's bottom (its top 17 mm held between the
pads). The right arm comes down over the blue cup (finger tips 13 mm above its rim), grips the spare, lifts it,
carries it with its bottom 74 mm above the table to the hand-off pad, lowers the last 2 cm over 1.5 s, lets go 2 mm
above the pad, goes up and moves to its parking square, where it stays with its tool point 5 cm above the square
(inside its parking zone from 17.8 s, fully down at 18.5 s). Meanwhile the left arm waits 14 cm up over the discard
tray, outside the panel. One second after the right arm is down, the left arm heads into the panel (inside from 20.1
s), comes down beside the red fuse with its finger tips 8 mm above the socket's 25 mm rim, grips it, pulls it
straight up, carries it out of the panel to the tray and lets it drop the last 5 mm (finger tips 13 mm above the
tray's rim). It then takes the spare off the pad, carries it back into the panel over the socket, comes down to 15
mm above the rim, lowers it over 2 s into the socket (5 mm of clearance on every side) and lets go 2 mm above the
floor with its finger tips 10 mm above the rim, then goes up and back over the tray, out of the panel (51.5 s). The
right gripper's finger zone comes no closer than 2.4 cm to the panel's live zone (fully open, over the hand-off
pad), and it is parked, with 5 cm to spare on every side of its zone, on every step the left gripper is inside. The
TCPs stay at least 30.2 cm apart and the arms' collision capsules at least 9.2 cm apart. Offline, success has held
for its 1 s at step 1,049 (the scripted policy stops at step 1,099). The run limit is 1,600 steps (80 s).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the red fuse back upright and centred in the
socket and the green fuse back upright and centred in the blue cup; everything else is fixed.

## Runtime per trial

About 2 minutes on an L40S once Isaac Sim is running (the demo is 51 s of simulated time); the first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

Nothing is wired or powered. A 160 x 160 mm, 3 mm dark grey painted plywood plate fixed to the bench, bordered with
12 mm wide, 2 mm thick yellow plastic strip; a 30 x 30 mm square socket, 25 mm tall, at its centre (a porcelain fuse
base, or one glued up from 8 mm ivory plastic), with two 40 x 12 x 6 mm copper bar offcuts in line with it as
busbars; two dummy cartridge fuses, 20 mm across and 50 mm long, weighted to 30 g (ceramic or dowel tube with brass
caps), one painted red and one green; a blue plastic cup with the same 30 mm square opening, 20 mm tall; a white 70
x 70 mm, 3 mm plastic sheet as the hand-off pad; a black plastic tray with 25 mm walls round an 80 x 80 mm floor;
and two grey 100 x 100 mm, 2 mm rubber mats in front of the arm bases, each placed as in `task.json`'s `setup`. Mark
15 cm on a post at one corner of the panel and 10 cm on a post at each parking square, and film from above and from
the side to judge the rules.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | not scored: the Dream-1 API declined the instruction ("Replacing a fuse in a live panel presents an electrical hazard"), so the trial errored before the policy acted and the video shows the arms at rest |
