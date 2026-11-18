# Sweep up a broken glass with a dustpan and brush, holding the pan flat at the table

A drinking glass broke on the table: eight pieces lie in front of the robot. A dustpan lies with its lip toward them
and its brush stands to the right. The robot has to sweep every piece into the dustpan with the brush, without
touching any piece, then put the brush down and leave the dustpan flat on the table.

## Why this is everyday

Sweeping up a broken glass with a dustpan and brush is a familiar household chore, and the pieces are sharp: they are
swept up, never picked up by hand. It takes both hands working together: one holds the dustpan flat and still with
its lip on the table, the other brushes the pieces toward it and over the lip without flicking them aside, and a
piece left at the lip has to be gone back for.

## Sources

- Dustpan and brush: OXO Good Grips Compact Dustpan & Brush Set, item 1334280 (dustpan 186 x 137 x 40 mm, e.g.
  https://www.mobilitysmart.co.uk/oxo-compact-dustpan-brush-set.html; the set 7.35 x 5.4 x 1.6 in, 0.45 lb,
  https://www.webstaurantstore.com/gg-compact-dustpan-brush-set/2971334280.html).
- The pieces: stand-ins cut from 3 mm clear acrylic (a drinking glass's wall is about 3 mm thick), so real trials
  are safe; their sizes and layout are in task.json "setup" and layout.py.

Every figure not listed above is an estimate (task.json "fidelity"): the dustpan's section (2.5 mm floor, an 18 mm
lip ramp from 0.5 mm, 22 mm side walls, a 40 mm back wall), the brush (a 140 x 28 mm head, 22 mm bristles, a 70 mm
handle continuing the head's line), and the masses.

## Geometry (world, m; table top z 0.7475)

| Part | Where |
|---|---|
| Dustpan | lip edge along x 0.270, centred on y 0; back wall at x 0.133; mouth facing +x |
| Brush | bristle footprint centre at x 0.200, y -0.190; head along y; handle toward -y |
| Pieces | 8, between x 0.30 and 0.38 and y -0.05 and +0.05 |

The dustpan and the brush are loose compound rigid bodies (mechkit); the pieces are loose boxes. Nothing is moved or
latched by code.

## Scoring (scoring.py)

- In the pan: a piece's centre inside the pan's walls and at most 12 mm above its floor (in the pan's own frame).
- 1.0: all 8 in, the pan flat (5 degrees, its lip within 4 mm of the table), both grippers 30 mm or more from the pan
  and the brush for the last 20 steps, and nothing moving (2 mm over the last second).
- 0.5: 6 or 7 in, or all 8 but not yet set down, let go or still. 0.25: 4 or 5 in. 0: fewer.
- 0 for the whole run if at any moment a gripper touches a piece, or a piece falls off the table. In simulation
  "touches" is fail-closed: a finger zone (half the finger gap plus 8 mm round the pads, wherever the fingers point)
  reaching a piece, or arms that cannot be read.

In a real trial: count the pieces in the pan on a top-down photo; touching is judged on the video.

## The demo (both arms)

Open loop.

1. Left: pinches the dustpan's back wall from above (the pads across the 3 mm wall, their tips 12 mm below its top)
   and holds the pan where it lies.
2. Right: takes the brush by its handle (straight down, the pads across it), lifts it, sets the bristles down 160 mm
   beyond the lip and sweeps toward the robot at 40 mm/s; the brush's leading face stops 2 mm short of the lip's
   edge, then the brush rises 3 mm while it crosses the lip (as bristles bend over a lip) and pushes on onto the
   pan's floor. Twice. Then it stands the brush back where it was and lets go.
3. Left: lets go of the pan and lifts away.

Both hands let go by opening the pads only to a 24 mm gap: opened fully, the left hand's far finger would stand
47 mm out from the back wall, right over the pieces in the pan.

Counterfactual variants (DEMO_VARIANT, never the submitted demo): `one_stroke` sweeps once; `no_hold` leaves the
pan unheld; `keep_brush` never puts the brush down.

## Run it

On a VM set up with the program's Brev launchable (Isaac Lab-Arena 0.3.1 and the yam_arena embodiment):

```bash
source /data/IsaacLab-Arena/yam.env
cd tasks/sweep-broken-glass
INSTR="A glass broke on the table. Sweep all the pieces into the dustpan with the brush without touching any of them, then put the brush down and leave the dustpan flat on the table."
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 800 --store-frames --log-dir ~/yam-eval/sweep-demo
inspect-robots run --embodiment yam_arena --policy dream -E task=$PWD/task.py -E object_jitter=0 \
  -E control_hz=20 -E cam_width=320 -E cam_height=176 --instruction "$INSTR" --scorer task_score \
  --max-steps 800 --store-frames --log-dir ~/yam-eval/sweep-dream
```

`TASK_TRACE=<file>` writes one JSON line per control step. No seeds: the layout is fixed (`object_jitter=0`).
Offline tests (no simulator): `PYTHONPATH=<yam-isaac-lab-arena checkout> python3 -m unittest discover -s tests -v`.

Runtime per trial: RESULTS_PENDING
