# Bounce a ping-pong ball on a table tennis paddle ten times in a row, 10 to 25 cm high

A table tennis paddle lies red face up on a foam block in front of the robot, its handle pointing to the robot's
right. A ping-pong ball sits in a small cup to the left. The robot has to pick up the paddle, drop the ball onto it from
no higher than 15 cm, and keep the ball bouncing on the red face ten times in a row, the ball's bottom rising 10 to
25 cm above the face each time.

## Why it matters

Bouncing a ball on a paddle is the warm-up every table tennis player does and a familiar party challenge; anyone
watching counts the bounces. It needs a kind of physical control the board does not test: a steady rhythm against a
moving ball.

- **Energy.** Each touch loses some of the bounce (here 28% of the height). Held still, the paddle lets the bounces
  die away: under the highest allowed drop, a still paddle gives at most one bounce over 10 cm. Every tap has to add
  back what the bounce loses, by meeting the falling ball with the face moving gently upward, once per bounce.
- **Level.** A face tilted by a degree or two at a tap sends the ball sideways; a few such taps walk it off the edge.
  On this robot the wrist flexes as the paddle moves up and down, so the hand also has to hold the face level against
  its own motion.

## Sources

- Paddle: a standard recreational table tennis bat (blade about 158 x 150 mm, 6 mm with rubbers, handle about
  100 mm, 160 to 190 g).
- Ball: ITTF 40+ (40 mm, 2.7 g).

Every other figure is an estimate (task.json "fidelity"): the ball's bounce off the rubber (restitution 0.85), the
frictions, the paddle's mass split. No third-party meshes or scans: everything is built from boxes and cylinders in
task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Foam block | 110 x 110 x 100 mm, centred x 0.360, y 0 (fixed) |
| Paddle | blade 154 mm round, 6 mm thick, red face up, centred on the block; handle 100 x 26 x 22 mm toward -y (centre y -0.115); 180 g, loose |
| Ball | 40 mm, 2.7 g, loose, in a 30 x 30 x 22 mm cup (3 mm walls) at x 0.300, y +0.150 |

The paddle is one loose rigid body (mechkit). Its surface's restitution is combined with the ball's by "max" (added
to mechkit for this task: `assembly(restitution=...)`), so the ball bounces off the face and nothing else. Nothing is
moved or latched by code.

## Scoring (scoring.py)

The ball is read in the paddle's frame: its bottom's height above the red face, and its centre's distance from the
blade's centre in the face's plane. A touch is where, between two control steps, the ball's velocity along the face's
normal (relative to the paddle) turns from falling to rising, its own velocity along the normal jumps up by more than
0.5 m/s (a contact's impulse; gravity alone takes 0.49 m/s off it per step), and its bottom is within 50 mm of the face
at one of the two steps (at 20 Hz, a ball meeting the face at 2 m/s is at most that far at the nearer step) and
not below the face at either.

- **Face touch:** a touch with the ball's centre over the blade, the paddle held up (its blade at least 15 mm above
  the block).
- **Run:** starts at the first face touch after a gripper lets go of the ball. It ends at a flight outside the band
  below (that touch can start the next run), a gripper within 30 mm of the ball, or a touch anywhere else (table,
  block, blade edge, handle). After those two, only a new drop from a gripper starts a run.
- **Bounce:** a flight between two face touches in a run in which the ball's bottom rises 100 to 250 mm above the face
  and no gripper (TCP or pad tips) comes within 30 mm of the ball. The upper limit matters: after a higher flight a
  still paddle could add several passive bounces.
- **The drop:** from the last step a gripper has the ball (within 10 mm of it) to the first face touch. If the ball's
  bottom is more than 150 mm above the face at any step of it, the let-go point included (so lowering the paddle after
  letting go counts too), the bounces after it do not count until a hand has the ball again.
- 1.0: 10 bounces in a row (success, held 1 s). 0.5: 6 to 9. 0.25: 4 or 5. 0: fewer. The best run counts. A still
  paddle under the highest allowed drop gives at most one bounce, and after the highest allowed bounce at most two
  more, so passive rebounds alone never reach a tier.

Real trial: two synchronized 60 fps videos. A side-on camera level with the paddle, with a steel ruler fixed upright
behind it marked every 10 mm (100, 150 and 250 mm in red), gives each touch and the ball's height above the face's position at that moment. A phone
overhead, looking straight down, shows the ball's centre over the blade at each touch and the grippers' clearance from
the ball. Both are read frame by frame.

## The demo (both arms)

Open loop, nothing read from the simulator.

- **Right hand: paddle.** It comes in level along the handle from the robot's right, with one pad above the handle
  and one below, and closes on the handle's flats. A straight-down grip across the handle let the blade droop about
  13 degrees: the pads barely resist a twist about their own normal. The pinch on the flats holds it level. The hand
  lifts the paddle 42 mm off its block.
- **Left hand: ball.** It takes the ball from its cup, leans in from behind the blade (pitched 60 degrees) and lets
  it go 120 mm above the centre of the face. Then it moves straight back and to the left.
- **The rhythm.** The right hand moves the paddle straight up and down 12 mm each way, once per bounce (0.35 s,
  which makes each bounce about 15 cm high). The first stroke's bottom is timed 35 ms after the release. The wrist
  flexes with the stroke, tipping the face about 2.7 degrees toward +x at the bottom and toward -x at the top, so the
  commanded face is tipped 1.3 degrees the other way at each end. A constant 0.63-degree trim cancels the held
  paddle's lean about x.

The rhythm was tuned on the simulator, which repeats a run exactly. Nearby settings give 5 to 9 bounces before
the ball wanders off.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `still` drops the ball from 120 mm onto a paddle
held still, and `high_drop` lets it go 250 mm above a still paddle.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/paddle-bounce-ping-pong
INSTR="Pick up the paddle, drop the ping-pong ball onto it from no more than 15 cm above the face, and keep it bouncing on the red face ten times in a row, each bounce 10 to 25 cm high."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/paddle-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/paddle-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step. No seeds: the layout is fixed (`object_jitter=0`).
Offline tests (no simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 35 s for the scripted demo (15 s to start Isaac Sim, then 275 steps: success at
step 256 and its 1 s hold) and about 5 minutes for a Dream-1 run of 1,200 steps (the policy is queried over the network
each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0: ten bounces in a row, each rising 121 to 210 mm above the face; the ball stayed within 70 mm of the blade's centre | 256 (success) |
| Dream-1 | 0.0: it reached the paddle and the ball but lifted neither (the paddle moved 3 mm) | 1,200 (the limit) |
| `still` | 0: dropped from 120 mm onto a paddle held still, no bounce rose 100 mm | 427 |
| `high_drop` | 0: let go 250 mm up over a still paddle; the drop is too high, so its rebounds do not count | 427 |

The demo scored the same on two runs (the simulator repeats a run exactly).
