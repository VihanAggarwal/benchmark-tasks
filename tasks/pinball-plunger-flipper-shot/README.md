# Play tabletop pinball: launch the ball, then flip it back up into the top lanes

A tabletop pinball machine stands in front of the robot, held to the table. The steel ball rests in the plunger lane.
The robot has to pull the spring plunger back and let it go, so the ball runs up the lane and round the top. Then,
about two and a half seconds later, it has to press a flipper button at the moment the ball rolls onto that flipper,
so the ball is struck back up the playfield into the top lanes instead of draining between the flippers.

## Why it matters

Pinball is a familiar arcade and bar game, and anyone watching a clip sees at once whether a shot was made. Even one
shot takes two kinds of physical control that the board does not test together:

- a launch whose strength is set by how far a spring plunger is drawn back: too little and the ball rolls back down
  the lane, too much and it rattles round the top;
- a strike timed to a moving ball. The flipper only sends the ball back up if it is raised while the ball is on it,
  within a window of about a tenth of a second.

The hand also has to get from the plunger to the button in the time the ball takes to come down.

## Sources

- Machine: the EastPoint Sports Tabletop Arcade Pinball (UPC 840281714605; Target DPCI 091-14-7737,
  https://www.target.com/p/arcade-pinball/-/A-94628600): 19 x 13 x 6 in assembled, 1.8 lb, supplied with 13 balls. It
  has no electrics ("This game is not electronic"): a spring plunger launches the ball, and each of the two flipper
  buttons works its flipper through a lever.

Every figure not listed above is an estimate (task.json "fidelity"): the playfield's layout (lane, top guides, inlane
guides, flipper positions), the flippers' size and 56-degree throw, the plunger's 45 mm travel and 300 N/m spring, the
buttons' 6 mm travel, the 19 mm steel ball, and every friction. No third-party meshes or scans are used: every object
is built from boxes and cylinders in task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Cabinet | x 0.200 to 0.683, y -0.165 to +0.165, held to the table |
| Playfield | 420 x 306 mm, from x 0.250, rising 6 degrees away from the robot |
| Plunger lane | along the right wall (y -0.140), 330 mm long; curved guide at the top right |
| Flippers | pivots 75 mm up the playfield at y +/-0.068; 60 mm long, tips 28 degrees down, 30 mm drain gap; 56-degree throw |
| Plunger knob | x 0.154 (at rest), y -0.140, 48 mm up; drawn back toward the robot |
| Flipper buttons | on the front deck's top, x 0.225, y +/-0.143; 6 mm travel: the flipper's angle follows the button |
| Top lanes' line | 330 mm up the playfield |
| Ball | 19 mm steel, in the plunger lane against the plunger |

The machine is a fixed mechkit mechanism: the plunger slides on a spring, each button slides down on a spring, and
each flipper turns on a hinge. One rule stands in for each flipper's lever: the flipper is driven (through a stiff
spring) toward the angle its button's travel gives, rest with the button up and fully raised at 6 mm, so it swings as
fast and as far as the button is pressed. The ball is a loose
sphere. Nothing else is moved or latched by code.

## Scoring (scoring.py)

Ball positions are read along the playfield's slope (u, from its front edge) and across it.

- Launched: the ball has left the plunger lane for the playfield.
- Flipped: after the ball has left the plunger lane, it is struck by a flipper: it moves up the slope faster than
  0.3 m/s while that flipper is raised more than half way, having been within that flipper's reach (82 mm of its
  pivot) in the last 0.2 s, and within 0.3 s after that flipper fired (rose past half way). A fast launch with a
  flipper up, or a bounce off a flipper held up, does not count.
- Top: after a flipper shot, the ball reaches the top lanes' line, 330 mm up.
- Drained: the ball falls to within 30 mm of the bottom (between or past the flippers); nothing after that counts.
- 1.0: a flipper shot into the top lanes. 0.5: a flipper shot that falls short. 0.25: launched, never flipped.
- 0: never launched, and the whole run if a gripper reaches over the playfield inside the cabinet's walls.

In a real trial: a phone held overhead, looking straight down on the playfield at 60 fps. The top lanes' line is
printed on the playfield and a tape line marks the drain line, so the strike, the ball's direction and the crossings
are read off the video frame by frame.

## The demo (right arm)

Open loop, the right arm only; the left stays parked. A straight-down grip on the plunger's knob, the pads either side
of it. The hand draws the plunger back 42 mm (the arm gives a little against the spring), and then opens: the ball
runs up the lane to the top and rolls back down the right side, where the inlane guide sends it across the right
flipper. Meanwhile the same hand, closed, moves over the right flipper button. It presses the button all the way
down in 0.1 s, 70% down at 9.2 s, the moment the ball reaches the flipper, timed from the launch on the simulator. It
holds the button 0.6 s and lets it up. The flipper strikes the ball back up the playfield into the top lanes.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `no_flip` launches and never presses a button (the
ball drains, 0.25), and `early_flip` presses and releases 0.9 s too early.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/pinball-plunger-flipper-shot
INSTR="Play the pinball machine: launch the ball with the plunger, then flip it back up into the top lanes with a flipper."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 800 --store-frames --log-dir ~/yam-eval/pinball-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 800 --store-frames --log-dir ~/yam-eval/pinball-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step. No seeds: the layout is fixed (`object_jitter=0`).
Offline tests (no simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 30 s for the scripted demo (15 s to start Isaac Sim, then 193 steps to success)
and about 4 minutes for a Dream-1 run of 800 steps (the policy is queried over the network each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0 (launched; the right flipper struck the ball at 0.7 m/s up the slope; it reached 385 mm, past the top lanes' line) | 193 (success) |
| Dream-1 | 0.0 (the ball was never launched) | 800 (the limit) |
| `no_flip` | 0.25 (launched, drained) | 390 |
| `early_flip` | 0.25 (the flipper was down again when the ball arrived; drained) | 374 |

The demo scored the same on two runs.
