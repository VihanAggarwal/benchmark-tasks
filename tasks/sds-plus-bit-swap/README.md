# Swap a rotary hammer's SDS-plus bit: sleeve held back to draw the old one, the new one twisted in until it locks

A Bosch 11224VSRC SDS-plus rotary hammer lies loose on the bench with a 6 mm bit locked in its chuck; a 10 mm bit lies
beside it. The robot is asked to swap them. The old bit comes out only while the chuck's locking sleeve is held back,
so one hand holds the sleeve while the other draws the bit. The new bit goes in only with its two drive slots on the
chuck's hidden keys, so it has to be twisted while it is pushed, until it drops onto them and locks.

## Why it matters

Changing a drill bit is one of the most familiar jobs in a workshop or a home: drill one hole to hang a shelf, swap to
a bigger bit for the next. SDS-plus hammer drills are the standard tool for drilling masonry, at home and on site
(construction laborers, O*NET 47-2061.00: "Operate jackhammers or drills to break up concrete or pavement"). The
physical work:

- **Two hands at once.** Drawing the old bit needs the sleeve held 8 mm back against its spring the whole time.
- **A hidden alignment.** The new bit's slots must meet keys inside the chuck that can't be seen; the bit stops 12 mm
  in until it is turned onto them.
- **A loose tool.** The hammer lies free on the bench; pushing and twisting the bit in must not shove it away.

## Sources

- Bosch 11224VSRC 7/8" SDS-plus Bulldog rotary hammer: 15 in long, 7.9 lb (Bosch catalogue).
- Bosch SDS-plus instructions (11224VSRC operating instructions, p. 7): "Insert accessory into the tool holder through
  the dust shield, while twisting and pushing inward until it locks automatically into place." "Pull outward on the
  accessory to be certain it is locked into the tool holder." "To remove an accessory, pull and hold locking sleeve
  backward and pull bit forward."
- SDS-plus bits: a 10 mm shank 40 mm long with two open drive slots (6 x 210 mm and 10 x 160 mm are stock sizes).

The chuck's, sleeve's and dust cap's sizes, the sleeve spring and the frictions are estimates from photographs
(task.json "fidelity"). No third-party meshes: everything is built from boxes and cylinders in task.py.

## Geometry (world, m; bench top z 0.7475)

The hammer's frame has its origin on the bench under the dust cap's face, at (0.300, 0.000), its +y along the chuck's
axis into the tool (pointing away from the robot, world +x), its +x the tool's top (world -y).

| Part | Where (hammer frame) |
|---|---|
| Housing | x -0.060 to 0.034, y 0.085 to 0.381 (15 in), z 0 to 0.076, rounded top edges; gear-case nose a 54 mm cylinder, y 0.040 to 0.085 |
| Pistol grip | x -0.190 to -0.060, y 0.300 to 0.345; trigger on its front |
| Chuck axis | z 0.038 |
| Dust cap | a cone, y -0.012 to 0.005, 24 mm across at its mouth closing to 10.8 mm (round a 10 mm shank) |
| Tool holder | a bore 12 mm across (1 mm play), y 0.004 to 0.040 (home: shank end at 0.040) |
| Drive keys | at 12 mm deep: a stand-in gate across the bore that clears only for a shank with its slots within 8 degrees |
| Locking sleeve | 45 mm across, y 0.008 to 0.040; slides 8 mm toward the housing, spring 600 N/m (4.8 N full) |
| 6 mm bit | 6 x 210 mm (10 mm shank 40 mm, 6 mm body 170 mm), 35 g; locked, slots on the keys |
| 10 mm bit | 10 x 160 mm, 80 g; on the bench along x, centre (0.200, -0.110) world, slots 45 degrees off the keys |

Three rules (task.py), the rest is contact physics:

- **Locking balls** (welds: the bit's pose is rewritten on the hammer every control step while locked): a locked bit is
  freed while the sleeve is 6 mm or more back; a bit pushed home (within 2 mm of 40 mm deep) with its slots within 8
  degrees of the keys locks once the sleeve is let go, its slots set onto the keys.
- **Drive keys:** the gate closes across the bore at 12 mm whenever a shank arriving there has its slots more than 8
  degrees off the keys, and stays clear otherwise (and whenever a shank is already past it).

## Scoring (scoring.py)

Everything is measured in the hammer's frame.

- **Old out:** the 6 mm bit's shank end at least 12 mm in front of the dust cap, or 20 mm off the chuck's axis.
  **Released:** no gripper within 30 mm of it.
- **Started:** the 10 mm bit's shank end within 3 mm of the axis and the bit within 6 degrees of it (the bore's play
  lets a part-inserted bit lean about 4), its shank at least 10 mm in.
  **Locked:** pushed home with its slots on the keys and the sleeve let go (under 6 mm back): the locking balls hold
  it.
- **Tiers:** 1.0 old out and released, new bit locked, both grippers 30 mm clear of the hammer and both bits, all
  still for 1 s (success); 0.75 locked; 0.5 started; 0.25 old out and released; 0 otherwise.
- **Foul (0 for the trial):** the hammer off the bench.

A gripper is the segment from its TCP to its pad tips, swollen by half its pads' current gap plus 8 mm. Real trial:
overhead and side videos, and a 20 N spring-scale pull on the new bit (Bosch: "pull outward ... to be certain it is
locked").

## The demo (both arms)

Open loop, nothing read from the simulator.

1. **Old bit.** The left hand's closed fingertips go down just in front of the locking sleeve, above the dust cap, and
   push its front rim back toward the housing (8 mm) and hold it. The right hand pinches the 6 mm bit 70 mm from its
   tip, draws it 100 mm straight out (rising 3 mm as it goes, against the arm's sag), slides it 50 mm aside and puts it
   down. The sleeve is let go.
2. **New bit.** The left hand presses the gear case's nose down onto the bench. The right picks the 10 mm bit up at its
   middle (its centre of mass, so it does not pivot between the pads), carries it onto the chuck's axis and pushes it
   into the dust cap up to the keys. There it turns the bit 45 degrees about its own axis (the wrist rolled about the
   bit), the slots find the keys, and it pushes the bit 25 mm in. It lets go, takes the bit again near its tip (8 mm
   lower: part-way in, the bit leans in the bore's play) and pushes it home; it locks. Both hands let go and move out to
   the sides.

The sleeve push, the regrip height and the rise on the pull were worked out on the simulator.

Counterfactual variants (DEMO_VARIANT, never the submitted demo):

- `no_twist`: the new bit pushed straight in without the twist.
- `no_sleeve`: the sleeve rested on but not pushed back.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/sds-plus-bit-swap
INSTR="Swap the drill bit in the rotary hammer for the 10 mm bit on the bench."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1800 --store-frames --log-dir ~/yam-eval/sds-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 1800 --store-frames --log-dir ~/yam-eval/sds-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step (sleeve, keys, both bits' poses in the hammer frame, the new
bit's depth, tilt and slot angle, locks, gaps). No seeds: the layout is fixed (`object_jitter=0`). Offline tests (no
simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

## Results

| Run | Score | Steps |
|---|---|---|
| Scripted demo | 1.0: sleeve held 8 mm back, 6 mm bit drawn out and put down; 10 mm bit stopped at the keys, turned 45 degrees (slots within 2 degrees of the keys), pushed home and locked; hands clear | 629 (success) |
| Dream-1 | 0.0: the 6 mm bit was never drawn out | 1,800 (the limit) |
| `no_twist` | 0.25: the 10 mm bit stopped at the keys, 10.7 mm in, and went no further | 706 |
| `no_sleeve` | 0: the 6 mm bit stayed locked; the pull slid off it | 706 |

The demo scored the same on two runs (the simulator repeats a run exactly). Runtime per trial on an L40S: about 50 s
for the scripted demo (15 s to start Isaac Sim, then 648 steps) and about 8 minutes for a Dream-1 run of 1,800 steps
(the policy is queried over the network each step).
