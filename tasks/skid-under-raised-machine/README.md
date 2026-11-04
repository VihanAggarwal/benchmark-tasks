# Slide a skid under a machine while it is raised no more than 30 mm

Millwrights and machinery movers raise a machine only as far as needed to get skates or a skid under it, and
keep it from shifting while it is up. Here a small green "machine" (a 0.30 kg box) stands on the table and a long
wooden skid lies beside it. One arm raises the machine about 2 cm and holds it where it stands while the other
slides the skid under it along the table; then the machine is set down on the skid. The scorer watches the whole
trial: how high any part of the machine's underside went and how far its centre moved. The task needs both arms:
a single arm can either hold the machine up or slide the skid, not both; sliding the skid without raising the
machine shoves the machine along the table, and carrying the machine over to the skid instead takes it out of its
30 mm outline, which scores 0.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores, media ID |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the machine's and skid's states, with a small tracker for the whole trial |
| `tests/test_scoring.py` | Offline tests of every scoring tier, edge case and the exploits we tried (no simulator needed) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (loose boxes) and the standard `task_score` scorer.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Raise the green machine no more than 3 cm, slide the wooden skid under it, and set the machine down on the skid. Don't drag the machine out of place." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/skid-under-machine/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Raise the green machine no more than 3 cm, slide the wooden skid under it, and set the machine down on the skid. Don't drag the machine out of place." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/skid-under-machine/dream
```

Run one trial per process (the default). The scorer keeps the trial's record in the task module, and recognises a
new episode in the same process only when the simulator puts both objects back on their marks in one step.

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided (`max_lift_mm`, `max_drift_mm`, `gap_to_skid_mm`, `cover`,
...). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. The scorer reads only the machine's and the skid's poses
and velocities, once per control step (20 Hz), and keeps three facts about the trial: the machine's largest
horizontal drift from its starting place, the greatest height any point of its underside reached above the table
(a tilted machine counts by its raised edge), and its largest tilt. Footprint overlap is computed from the two
boxes' actual outlines (turned skids included), not their bounding boxes. A skid turned over is still a flat plank
(either broad face up). A new episode is recognised when the simulator puts both objects back on their starting
marks, still, in a single step, with one of them having moved 8 cm further than its last velocity and a free fall
could carry it in that step (so a machine dropped straight back onto its mark is not mistaken for a reset). An observation with NaN or inf in it scores 0 for that step and
changes nothing on the record.

| Outcome | Score |
|---|---|
| The machine tilted more than 30 degrees, or its drift went over 30 mm, at any moment; or it ends off the table | 0 |
| Machine on the skid; drift never over 10 mm, raise never over 30 mm | 1.0 |
| Machine on the skid; raise up to 50 mm, but not within the 1.0 limits | 0.75 |
| Machine on the skid; raised more than 50 mm | 0.5 |
| Skid under the machine (at least 50 % of its footprint), machine not on it (held up, only partly on it, or leaning on the skid's end) | 0.25 |
| Anything else (machine back on the table, skid beside it or on top of it) | 0 |

The trial ends early (success) once the 1.0 conditions have held with both objects still (under 1 cm/s and
0.5 rad/s) for 3 s, longer than letting go of the machine and lifting the hand clear, so it cannot end while the
machine still hangs in a closed gripper just above the skid. Otherwise it is scored as it stands at the run limit.

'On the skid': the skid flat on the table, the machine's underside within 2 mm of the skid's top, the machine
tilted under 3 degrees, at least 75 % of its footprint over the skid (the skid is 60 mm wide, the machine 70 mm,
so a centred skid covers 86 %).

Unintended strategies earn nothing: shoving the skid against the machine without raising it pushes the machine
along the table, out of its 30 mm outline (0); carrying the machine over to the skid where it lies, set on it or
held over it, takes it out of the outline however low it is carried (0), and so does pushing the skid back with the
machine on it afterwards (0, the drift is the largest over the trial); inside the outline one arm can bring at most
24 % of the machine's footprint over the skid where it lies, short of the 50 % that 'under' needs. Tilting the
machine instead of raising it level counts by its raised edge, so a tilt cannot hide a high raise (tipped 20
degrees onto one of its 70 mm bottom edges, the opposite edge is 31 mm up). Holding the skid up off the table with
the machine on it is not "on the skid" (0), and a machine still held more than 2 mm above the skid has not been set
down (0.25).

## One arm alone

`DEMO_IDLE_ARM=right` (the left arm alone raises the machine, holds it, and sets it back down where the skid
should have been) scores 0: the machine ends on the table. `DEMO_IDLE_ARM=left` (the right arm alone slides the
skid) scores 0: the skid's end runs into the machine, which still stands on the table, and shoves it about 21 cm
along, out of its 30 mm outline (the kinematic proxy cannot push: it reports the skid running into the machine and
scores the overlap as 0). A single arm cannot hold the machine up and slide the skid at the same time, and the
one-arm alternative, carrying the machine over to the skid, scores 0 (see above). (Offline kinematic predictions;
the GPU runs are recorded in `task.json`'s `scores`.)

## Demo and run limit

The scripted demo takes 21.5 s (430 control steps). Every final descent is a separate 1.5 s move over the last
3 cm. The left arm grips the machine across its 70 mm length with the pad tips 15 mm above the table (fingers
commanded to 62 mm) and raises it about 2 cm. The right arm, which has gripped the skid's right end across its
60 mm width with the pad tips 4 mm above the table (fingers commanded to 52 mm), slides it 22 cm along y at table
height, so that its centre ends 5 cm right of the machine's and it runs under the whole 90 mm of the machine with
12 cm to spare beyond it; it lets go and backs off up and outward. The arms' grip points are then 25 cm apart.
The left arm then lowers the machine onto the skid and lets go, and lifts its hand clear. Every gripper open or
close takes 1.5 s. Success (both objects still for 3 s) is reached at about step 454 in the offline proxy; the
scripted policy stops at step 491. The run limit is 1,200 steps (60 s).

The raise is commanded as 28 mm: the arm's shoulder and elbow are position-controlled springs without gravity
compensation for a payload, and they sag about 8.4 mm (and 3.5 mm sideways) under the 0.30 kg machine, so the
machine should end up about 20 mm up: its lowest corner 6 mm over the skid's 12 mm top, and still under 30 mm if
the arm sags less. `TASK_TRACE` reports the actual `max_lift_mm` and `max_drift_mm`.

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: the machine back on its mark centred at
x 30.0 cm, y 0; the skid back flat on the table centred at x 30.0 cm, y -27.0 cm, long side along y.

## Runtime per trial

About 1 to 1.5 minutes on an L40S once Isaac Sim is running (the demo is about 23 s of simulated time); the first
start of Isaac Sim on a new instance takes several minutes more.

## Real-world build

A 70 x 90 x 60 mm steel or aluminium block (or a box weighted to 0.30 kg), painted green, and a 60 x 440 x 12 mm
hardwood plank as the skid. Stand a card upright 10 cm behind the machine (on the side away from the arms) with
marks at 30 mm and 50 mm above the bench so the raise can be read on a side video, and tape outlines 10 mm and
30 mm outside the machine's starting footprint for the drift.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |
