# Hit an incoming air hockey puck into the far goal with a held mallet

A tabletop air hockey game stands on the table pointing away from the robot. Seven seconds into the trial a practice
kicker at the far end sends the puck down the left of the table toward the robot. The robot holds the mallet by its
handle and has to hit the moving puck back into the far goal, keeping the mallet on its own half.

## Why it matters

Air hockey is one of the most familiar arcade and family games, and hitting a moving puck is a compact test of three
things at once:

- **Timing.** The puck arrives at about 0.8 m/s; the mallet has to be moving through the meeting point as it gets there.
- **Aim.** The puck comes down 80 mm left of centre, so knocking it straight back sends it into the end rail beside the
  goal. The strike has to be angled across the puck's path.
- **Tool use.** The puck is struck with a held tool, not with the hand.

## Sources

Battery tabletop air hockey games are about 50 to 75 cm long and 30 to 40 cm wide, with rails, a goal slot at each end,
pucks of 38 to 50 mm and mallets of 50 to 60 mm (for example the Relaxdays 10022514: 56 x 31 cm, a 40 mm puck, 55 mm
pushers). The simulated game is 600 x 320 mm with a 50 mm puck and a 60 mm mallet. The puck's speed and the mallet's rebound are estimates (task.json "fidelity"). The kicker
stands in for a spring launcher (a person serving the puck). No third-party meshes: everything is built from boxes and
cylinders in task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Playfield | x 0.220 to 0.820, y -0.160 to 0.160, top 0.060 above the table; friction 0.02; halfway line at x 0.520 |
| Rails | 20 mm high, 15 mm thick; a 90 mm goal slot centred in each end rail, a white box behind each |
| Puck | 50 mm, 6 mm thick, 10 g; starts at (0.740, 0.080) |
| Mallet | 60 mm disc 12 mm thick, handle 24 mm x 35 mm, 50 g; starts at (0.260, 0.000), in front of the robot's goal and 25 mm clear of the puck's line |
| Kicker | 20 x 50 x 14 mm block 15 mm behind the puck; at 7 s its face moves 40 mm at 0.55 m/s |

The game is one fixed mechkit mechanism; the kicker is a driven slider and the only thing moved by code. The puck
leaves the kicker at 0.86 m/s and crosses the halfway line at 0.82 m/s.

## Scoring (scoring.py)

- **Hit:** between two control steps the puck's velocity along the table turns from toward the robot to away from it
  (0.05 m/s or more, so a puck that just stops against the mallet or the rail is not a hit),
  with the mallet's rim within 30 mm of the puck's (at 20 Hz the two are a few cm apart either side of the contact)
  and the mallet itself moving at 0.2 m/s or more over the
  0.1 s up to the hit (a strike: a mallet held still in the puck's way is not a hit). The mallet has to be held (a
  gripper's TCP within 20 mm of its axis at the handle's height, the fingers closed past half open: about 0.26 on the
  24 mm handle), and the puck's edge at least 10 mm from the near end rail (not a rebound off the rail).
- **Tiers:** 1.0 after a hit the puck goes into the far goal (success, held 1 s); 0.5 after a hit it crosses the
  halfway line moving away; 0.25 a hit; 0 otherwise.
- **Fouls (score 0):** any part of the mallet past the halfway line; a gripper's TCP or pad tips within 15 mm of the
  puck's edge (and 30 mm of its height); the puck in the robot's own goal.

The halfway-line rule is the game's own and also rules out carrying the puck down the table to the goal.

Real trial: a battery tabletop game of about this size (for example a Relaxdays 10022514, 56 x 31 cm; record its model
and sizes), an overhead slow-motion video at 240 fps (a phone's slow-motion mode) with the halfway line and the goals
marked, the same rules read from its frames, and a spring launcher (a sled on a rail drawn back 40 mm to a latch, released at 7 s by a hobby servo on a timer) whose spring is checked before each set
of trials: the puck takes 0.22 to 0.25 s between two marks 200 mm apart.

## The demo (left arm)

Open loop, nothing read from the simulator.

1. **Grip.** The hand comes down over the mallet's handle and closes on it, holding the mallet 2 mm down onto the
   table so it doesn't tip up as it strikes.
2. **Ready.** It slides the mallet to a spot behind where the puck will be met (x 0.36, the puck's line), on a line
   8 degrees right of straight up the table.
3. **Strike.** At 7.42 s it drives the mallet 100 mm along that line in 0.2 s, meeting the puck mid-stroke. The puck
   leaves at 1.3 m/s, angled across to the goal.

The aim and the timing were worked out on the simulator. In tuning, aims of 5 and 11 degrees sent the puck into the end
rail either side of the goal.

Counterfactual variants (DEMO_VARIANT, never the submitted demo):

- `block`: the mallet held still in the puck's way.
- `late`: the same strike 0.3 s late.
- `idle`: the hand waits over the mallet.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/air-hockey-return-into-goal
INSTR="Play air hockey: hold the mallet by its handle and hit the puck that slides toward you so that it goes into the far goal. Keep the mallet on your half and your hands off the puck."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 600 --store-frames --log-dir ~/yam-eval/hockey-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 600 --store-frames --log-dir ~/yam-eval/hockey-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step (puck, its velocity, mallet, lift, flags). No seeds: the
layout is fixed (`object_jitter=0`). Offline tests (no simulator):
`PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 25 s for the scripted demo (15 s to start Isaac Sim, then 178 steps: success at
step 159 and its 1 s hold) and about 3 minutes for a Dream-1 run of 600 steps (the policy is queried over the network
each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0: the mallet met the puck at 0.36 m/s (step 151), the puck left at 1.3 m/s and went into the far goal | 159 (success) |
| Dream-1 | 0.0: it never hit the puck, which stopped against the near end rail | 600 (the limit) |
| `block` | 0: no hit (mallet still); the puck bounced off it into the far end rail beside the goal | 332 |
| `late` | 0: no hit; the puck ran into the mallet after the stroke had stopped | 339 |
| `idle` | 0: the puck passed the mallet and stopped against the near end rail | 400 |

The demo scored the same on two runs (the simulator repeats a run exactly).
