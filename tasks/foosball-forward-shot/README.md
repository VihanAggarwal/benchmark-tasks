# Score a table football goal from your three-man rod without spinning the rod

A tabletop foosball table stands in front of the robot, its long side toward it, goals at the left and right ends.
The robot plays red, whose handles are on its side, and attacks the right-hand goal. The ball rests just in front of
the red three-man forward rod, in the lane between the blue goalie's foot and the right-hand goal post. The robot
has to slide the forward rod so its centre player's foot is behind the ball, then strike with a short, quick turn of
the handle, scoring without spinning the rod and without a hand in the playfield.

## Why it matters

Table football is played in bars, offices, student unions and homes, and anyone watching a clip knows at once what a
shot is and why it is hard. The ball sits under a row of players the robot cannot reach directly. All it has is a
handle 30 cm away from the foot that does the work, through a rod that both slides and turns. A good shot means
lining a player up with the ball to within a few millimetres by sliding the rod, then turning the handle quickly
enough to drive the ball 26 cm straight through a gap 17 mm wide past the goalie. Spinning the rod, turning it more
than a full turn, is against the rules everywhere the game is played, and so is putting a hand into the playfield.

## Sources

- Table: Best Choice Products 40 in tabletop foosball table: 40 x 22.5 x 6 in without legs, 23.5 lb, two balls
  (https://bestchoiceproducts.com/products/tabletop-foosball-table-arcade-table-soccer-w-2-balls-40in).

Every figure not listed above is an estimate (task.json "fidelity"): the cabinet's inside sizes, the rods' positions
(127 mm apart), heights and the 1/2 in diameter, the players' shape, the goal mouth (130 x 45 mm), the 36 mm ball and
every friction. No third-party meshes or scans are used: every object is built from boxes and cylinders in task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Cabinet | x 0.200 to 0.772, y -0.508 to +0.508, 152 mm tall; playfield 30 mm up |
| Goals | mouths 130 mm wide (along x), centred at x 0.486, in the end walls at y +/-0.493 |
| Red forward rod | y -0.1905; handle x 0.100 to 0.200; players at x 0.316, 0.486, 0.656; slides +/-60 mm |
| Blue defence and goalie | y -0.3175 (players at x 0.386 and 0.586) and y -0.4445 (player at x 0.486) |
| Ball | x 0.524, y -0.2365 |

The table is a fixed mechkit mechanism. The red forward rod slides along x on a joint with 3 N of bushing friction
between +/-60 mm bumpers, and turns freely about x, its players hanging under their own weight. Every other rod is
held still by a stop. The ball is a loose sphere. Nothing is moved or latched by code.

## Scoring (scoring.py)

- Goal: the ball wholly past the right-hand goal line, inside the goal mouth (latched).
- On target: the ball reached the right-hand end wall within 100 mm of the goal mouth (latched).
- 1.0: a goal, and both grippers 30 mm or more from the forward rod's handle for the last second.
- 0.75: a goal, the handle still held. 0.5: on target without a goal. 0.25: the ball moved 150 mm or more toward the
  right-hand goal.
- 0: anything else, and the whole run if the forward rod turns more than 360 degrees one way from its start (a spin),
  a gripper goes over the playfield below the cabinet's top or within 5 mm of the ball (a hand in the game), or the
  ball goes into the left-hand goal (an own goal).

In a real trial: the ball in the goal's catch box; tape marks on the end wall 100 mm either side of the mouth and on
the playfield 150 mm toward the goal; the rod's turns and the hands from the trial video.

## The demo (right arm)

Open loop, the right arm only. A straight-down grip on the red forward rod's handle, near its outer end (85 mm out
from the cabinet, so the hand clears the cabinet's wall after the slide), the pads closing on it from either side.
The hand slides the rod 38 mm away from the robot (it travels about 31 mm against the bushing friction, enough to put
the centre player's 24 mm foot behind the ball), winds the rod back 25 degrees, then turns it 45 degrees the other way
in 0.5 s, so the foot swings down through the ball and drives it straight toward the goal. The hand turns about the
rod's axis (the TCP sits 8 mm above it and follows the small circle that point makes). It turns the rod back to
hanging, lets go and lifts away.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `no_align` strikes without sliding the rod first
(the foot misses the ball), and `hold_on` scores but keeps hold of the handle (0.75).

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/foosball-forward-shot
INSTR="The ball is at your centre forward's feet. Shoot it into the goal on your right with your three-man rod, without spinning the rod."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 600 --store-frames --log-dir ~/yam-eval/foosball-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 600 --store-frames --log-dir ~/yam-eval/foosball-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step. No seeds: the layout is fixed (`object_jitter=0`).
Offline tests (no simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 30 s for the scripted demo (15 s to start Isaac Sim, then 273 steps to success)
and about 3 minutes for a Dream-1 run of 600 steps (the policy is queried over the network each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0 (the ball struck straight, 310 mm into the right-hand goal; no spin, no hand in the game; handle let go) | 273 (success) |
| Dream-1 | 0.0 (its right gripper reached into the playfield: a hand in the game; the ball never moved) | 600 (the limit) |
| `no_align` | 0 (the foot misses the ball) | 360 |
| `hold_on` | 0.75 (a goal, the handle still held) | 304 |

The demo scored the same on two runs.
