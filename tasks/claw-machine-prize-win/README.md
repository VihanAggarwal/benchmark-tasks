# Win a prize on a desktop claw machine: steer the claw with the joystick, then drop it

A desktop claw machine is clamped to the table in front of the robot. Its control deck faces the robot, with an 8-way
bat-top joystick on the right and a drop button on the left. Three boxed prizes stand in its pit, and the prize chute
is the pit's front-left corner. One credit is loaded. The robot has to:

1. steer the claw over a prize with the joystick;
2. press the button to drop it;
3. win the prize into the chute, without reaching into the cabinet.

## Why it matters

Claw machines are everywhere: arcades, malls, restaurants. Winning means placing a claw you don't hold:

- **Stick.** The joystick only chooses which way the claw travels, at a fixed 50 mm/s.
- **Judgement.** The player has to judge from outside the cabinet where the claw is over the prize.
- **Timing.** The stick must be let go at the right moment in each direction, before the 30 s timer runs out.

The physical skill is operating a spring-centred joystick precisely: tilting it past its switch, holding it there for
a judged time, and letting it spring back to stop the claw. That is remote positioning through a velocity interface, a
different kind of control from moving an object by hand.

## Sources

- Machine for real trials: a tabletop claw machine of the Contixo CM1 type (retail listings, e.g. Brookstone and Kohl's:
  one illuminated 360-degree joystick and a grab button, USB or 4 x AA powered, prizes up to 3.5 in, a side door for
  refilling, about $50 to $70). It is used as sold, clamped to the table, loaded with three 50 mm boxed prizes on the
  stated spots.
- Simulated machine: its sizes, the claw's 50 mm/s, the drop cycle, the 30 s timer, the 12 mm grab radius and the
  joystick's 8-degree switch angle are estimates for that type. The score depends only on where the claw closes and
  whether a prize is won, not on those values. No third-party meshes: everything is built from boxes
and cylinders in task.py.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Cabinet | x 0.330 to 0.590, y -0.130 to +0.130; pit floor 0.100 up; gantry housing 0.420 to 0.450 up; clamped |
| Control deck | x 0.220 to 0.330, 0.070 high |
| Joystick | pivot (0.260, -0.065) on the deck; bat top 24 mm across, 40 mm long, its middle 0.135 up; +/-22 degrees, switch at 8 |
| Button | (0.260, +0.065), 32 mm across, 5 mm travel, fires at 3 mm |
| Chute | the floor's opening at x 0.350 to 0.420, y +0.040 to +0.110 |
| Prizes | 50 mm boxes, 30 g, at (0.500, -0.050) red, (0.530, +0.040) blue, (0.450, -0.010) yellow |
| Claw | starts over the chute at (0.385, +0.075); travel x 0.360 to 0.560, y -0.105 to +0.105 |

The machine is one fixed mechkit mechanism:

- the joystick tilts on two sprung hinges;
- the button slides down on a spring;
- the claw rides a gantry: two sliders and a drop slider, driven to the program's positions.

## The machine's program (the one rule; scoring.Program)

- **Play.** While a direction's switch is closed (the stick tilted past 8 degrees that way), the claw travels that way
  at 50 mm/s, both axes at once on a diagonal. The 30 s timer starts at the stick's first move.
- **Drop.** The button pressed past 3 mm, or the timer run out, starts the drop:
  1. the claw drops at 80 mm/s to 4 mm above the prizes' tops;
  2. it closes, for 0.6 s;
  3. if its axis is within 12 mm of a prize's centre, it holds that prize (a mechkit weld carries the prize under it);
  4. it rises, travels back over the chute at 50 mm/s and opens (the weld lets go).
- **One credit, one drop.**

The robot does the real physical work, tilting and holding the stick and pressing the button. The rule stands in only
for the machine's motors and claw.

## Scoring (scoring.py)

- **Won:** a prize has fallen through the chute (inside the base, below the pit floor).
- **Near:** the claw closed with its axis within 25 mm of a prize's centre.
- **Hand in the cabinet (any time):** a gripper's TCP or pad tips inside the cabinet above the pit floor.
- **Tiers:**
  - 1.0: won (success, held 1 s).
  - 0.5: near, not won.
  - 0.25: the claw moved at least 50 mm from its start before it dropped.
  - 0: anything else, and every trial with a hand in the cabinet.

Real trial: an overhead video for the claw's position when it closes. A prize won drops through the chute's opening,
which is in plain view from above.

## The demo (both arms)

Open loop, nothing read from the simulator.

1. **Cage.** The right hand cages the bat top between its open pads (40 mm apart round the 24 mm handle). It pushes
   the bat with one pad, so the stick tilts freely. A pinch would hold the bat against tilting.
2. **Forward.** Pads before and behind the bat, the hand pushes the stick forward (about 14 degrees) for 1.55 s, then
   eases back. The stick springs to centre and the claw stops.
3. **Right.** The hand turns a quarter turn, pads either side, and pushes the stick right for 2.10 s. It pushes 12 mm
   further than forward, because the arm gives more that way.
4. **Drop.** The left hand presses the drop button.

The switch also stays closed for part of each push in and out (about 0.75 s forward, 0.4 s right; measured on the
simulator), so the holds put the claw over the red prize at (170, -50) mm in the machine's frame.

Counterfactual variants (DEMO_VARIANT, never the submitted demo):

- `short`: holds forward 1.2 s less, so the claw closes about 40 mm short.
- `no_drop`: never presses the button, so the 30 s timer drops the claw where it stopped.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/claw-machine-prize-win
INSTR="Play the claw machine: steer the claw over a prize with the joystick, press the button to drop it, and win a prize into the chute. You have one credit; don't reach into the machine."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/claw-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1200 --store-frames --log-dir ~/yam-eval/claw-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step (stick angles, claw position and state, prizes). No seeds:
the layout is fixed (`object_jitter=0`). Offline tests (no simulator):
`PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial on an L40S: about 40 s for the scripted demo (15 s to start Isaac Sim, then 449 steps: success at
step 430 and its 1 s hold) and about 5 minutes for a Dream-1 run of 1,200 steps (the policy is queried over the network
each step).

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0: the claw closed at (170.0, -50.0) mm, on the red prize's centre; it carried it back and dropped it down the chute | 430 (success) |
| Dream-1 | 0.0: it nudged the stick past its switch once (the claw moved 10 mm) and never pressed the button | 1,200 (the limit) |
| `short` | 0.25: the claw closed at (127.5, -50.0) mm, 42.5 mm short of the red prize and 41 mm from the yellow one: a real attempt, nothing near | 840 |
| `no_drop` | 0.5 at the run's end: the 30 s timer dropped the claw where it stopped; it closed on the prize and was carrying it back when the variant's plan ended | 830 |

The demo scored the same on two runs (the simulator repeats a run exactly).
