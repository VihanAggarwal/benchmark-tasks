# Key two lock plugs alike: every key pin flush with the shear line

Rekeying pin-tumbler cylinders is the most common job a locksmith does (new owners, lost keys, tenant changes;
O*NET 49-9094.00 Locksmiths and Safe Repairers), and a very common request is to have the front and back doors keyed
alike, so one key opens both. The locksmith takes each plug out of its shell with the new key in it and drops into
every chamber the key pin whose size number matches the key's cut there. The check is physical: every key pin top
must sit flush with the plug's surface, the shear line; a pin that stands proud or sits low by one step stops the
plug from turning.

Here two grey training plugs (enlarged models of plugs out of their shells) are fixed in front of the two arms,
each with four open-topped 34 x 34 mm chambers whose light silver floors stand for the new key's cuts (code
3-1-4-2: chambers 40, 28, 46 and 34 mm deep). A pinning kit on the midline holds ten gold key pins, 24 mm across and
28 to 52 mm long (size 1 to size 5, two of each). The robot must put the code's pin into each of the eight chambers
so that all eight tops lie in one plane, the plugs' top surface, with nothing else in or over a chamber. Each arm
keys the plug in front of it, and the two arms take turns at the one shared kit.

## Files

| File | What it is |
|---|---|
| `task.json` | The submission's details: instruction, setup, scoring, run limit, fidelity, scores |
| `task.py` | The `yam_arena` task file: the objects, `success()`, `score()` and the scripted `demo()` |
| `layout.py` | Every dimension, position, mass, colour and threshold, in one place |
| `scoring.py` | The scorer: plain functions of the pins' states and the grippers' tool points, with a small tracker |
| `rigstate.py` | Reads both grippers' tool points each control step for the let-go rule (unchanged shared helper) |
| `tests/test_scoring.py` | Offline tests of every score, every rule and its edge, and the exploits we tried (no simulator) |

## Requirements

The Isaac Lab-Arena environment from the program's Brev launchable (simulation walkthrough), with the
`yam_arena` embodiment, `inspect-robots` and the Dream-1 client. Nothing else to install: the task uses only
`yam_arena`'s own task-file objects (fixed boxes and loose cylinders) and the standard `task_score` scorer.
`rigstate.py` wraps the embodiment's step and reset from the task file's import to read the two grippers' tool
points; it changes nothing the embodiment does.

## Run

From this folder, after `source <ARENA_DIR>/yam.env`:

```bash
# scripted demo (both arms, taking turns at the kit): expected 1.0
inspect-robots run --embodiment yam_arena --policy yam_scripted \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Key both lock plugs alike to the code 3-1-4-2, chamber 1 being the one nearest the key's bow. Into each chamber drop one key pin of the size its code number gives, taken from the pin kit (size 1, the shortest, is in the kit's nearest row; size 5, the longest, is in its farthest), so that every pin stands on the key and ends flush with the top of its plug, with nothing else in or over the chamber." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/key-two-plugs-alike/demo

# Dream-1
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=task.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Key both lock plugs alike to the code 3-1-4-2, chamber 1 being the one nearest the key's bow. Into each chamber drop one key pin of the size its code number gives, taken from the pin kit (size 1, the shortest, is in the kit's nearest row; size 5, the longest, is in its farthest), so that every pin stands on the key and ends flush with the top of its plug, with nothing else in or over the chamber." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/key-two-plugs-alike/dream
```

Environment switches for the demo: `DEMO_IDLE_ARM=left` or `right` leaves that arm at its ready pose (the
one-arm check); `DEMO_HOLD_ONLY=1` plays no plan (a settling check); `TASK_TRACE=trace.jsonl` writes one JSON line
per control step with what the scorer saw and decided: `k`, the chambers pinned and, per chamber (`L1`..`L4`,
`R1`..`R4`), the pin in it, `tilt_deg`, `top_mm` (its top-face centre above the shear line), `path_mm` (its path in
the last 0.5 s), `net_mm` and `spread_mm` (its net displacement over the same 0.5 s and its largest distance from
the window's mean position: trace only, never scored), `tcp_mm` (the nearer gripper's tool point), `others_over`,
each check and why it failed; also both tool points and openings and every pin's position and lean
(`pin_pos`, `pin_tilt_deg`, so a pin caught on a wall top shows too). Videos: `inspect-robots video <log>.json`.

Offline checks, no simulator: `python3 -m unittest discover -s tests -v`.

## Scoring

See `scoring` in `task.json`; the code is `scoring.py`. Every control step (20 Hz) the scorer reads the ten pins'
poses and the two grippers' tool points (TCP, the point 13 mm above the finger tips). A chamber is **pinned** when
all five hold:

- (a) a key pin stands in it: the pin's centre, seen from above, is inside the chamber's 34 x 34 mm opening, and its
  axis is at most 8 degrees from vertical (either end up: a pin is the same both ways);
- (b) the centre of its top face is within 2.5 mm of the plug's top surface (the shear line), above or below;
- (c) it is still: its centre's path over the last 0.5 s is under 2 mm (from positions; a gripped part's reported
  velocity chatters in the simulator, so velocities are never used);
- (d) it is let go: neither gripper's TCP is within 50 mm of its top-face centre. At a step when the grippers cannot
  be read, no chamber counts as let go (no credit without evidence; counted in the trace's `arms_unavailable`);
- (e) nothing else is over it: no part of any other pin, seen from above at any height (its real outline, the
  shadow of the turned cylinder), lies over that chamber's opening.

k is the number of pinned chambers over both plugs, and the score is k/8, read at the last step. The trial ends
early once k = 8 (both cylinders keyed alike) has held for 1 s.

| Outcome (at the last step) | Score |
|---|---|
| All eight chambers pinned (success; the trial ends once this has held 1 s) | 1.0 |
| k chambers pinned, k = 0 to 7 | k/8: 0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875 |
| A chamber empty, with a wrong pin, two pins, its pin tilted over 8 degrees, still moving, still held, or anything over it | that chamber adds 0 |
| Pins in the kit, on the table, or on a plug top away from every chamber opening | add and subtract nothing |
| A missing or non-finite pin state (simulator blow-up) | 0 for that step; the tracker keeps its record |

Every outcome maps to exactly one k, so to exactly one score. Only the code's pin can count in a chamber:

- sizes differ by 6 mm, so a pin one size off stands 6 mm proud or sits 6 mm low, against the 2.5 mm band;
- any two pins stacked stand at least 10 mm proud (and each is "over" the other);
- only size 1 (28 mm) fits lying in a chamber with clearance (size 2 is exactly 34 mm, the chamber's width); either
  is then at least 4 mm low and the 8 degree rule rejects it;
- the 34 mm chamber caps any pin's lean at 13 to 27 degrees (its side meets the far wall, or that wall's top edge),
  so no standing pin can fall flat in it; a pin left leaning more than 8 degrees (for example with its foot caught
  against a chamber wall) does not count; at 8 degrees or less its top moves by at most 1.4 mm, so a correct pin
  still reads flush and a wrong one stays 4.6 mm or more off. (A crooked pin need not right itself: standing on one
  bottom-rim point with its opposite bottom corner against a wall, it can right itself only if its rim slides, and
  at the simulator's friction it stays, from 0.5 degrees up to the cap: 26.5, 19.5, 15.5 and 13.0 degrees for
  sizes 1 to 4. The tests check these leans in 0.5 degree steps, leaning away from each of the four walls.)

Unintended strategies earn nothing, as the tests show: a wrong pin held up so its top is flush is caught by the
let-go rule (the pads, 20 mm out from the pin's axis, cannot enter a 34 mm chamber, so a held pin is gripped above
the shear line, within 50 mm of its top); the last pin still gripped cannot complete k = 8 (and a gripper 19 mm over
one chamber is also within 50 mm of its neighbour's pin, 44 mm away); a spare stacked on a flush pin, standing on a
wall top, lying across the plug top over a wall (its centre in no chamber, its outline over two), or held above a
chamber stops that chamber counting; pins dropped and still falling or rocking count only once still; a pin left
leaning with its foot caught against a chamber wall counts only at 8 degrees or less, where only the code's pin
reads flush; wrong codes (the code reversed, sizes in order, all size 5) score 0; random poses never score outside
the nine values. Pushing does nothing: the plugs, the floors and the kit are fixed. Taking pins from the other
arm's kit column or putting a pin in upside down is harmless and counts, because the plugs are identical and a pin
is the same either end up.

## One arm alone

Two arms are not needed to key one plug, and the task does not claim they are. Each arm keys the cylinder in front
of it, and the arms take turns at the one shared pin kit on the midline: this is parallel work on two cylinders plus
turn-taking at the kit, not a two-hand hold. Neither arm can reach the other plug: straight-down IK fails everywhere
over it (x 20 to 36.6 cm, y -21.5 to -24 cm for the left arm and +21.5 to +24 cm for the right, TCP 7 to 13 cm above
the table, yaw 0, +-1.0 and +-1.57 rad). So one arm alone keys only its own plug: k = 4, score 0.5. In simulation the scripted demo with the other arm idle scored
exactly that: 0.5 with the left arm idle (the right arm keyed its own plug) and 0.5 with the right arm idle (see
'Results in simulation' below), as the offline check had predicted.

## Demo and run limit

Both arms on one clock, yaw 0 throughout (the fingers close along y). "Open" is a 40 mm gap everywhere (the gripper
is never fully opened near the kit) and "closed" is commanded to 16 mm, 8 mm narrower than the 24 mm pin, on its top
12 mm. Each arm takes its chambers in order 1 to 4, from its own kit column (y +3.0 cm for the left arm, -3.0 cm for
the right): kit rows 3, 1, 4 and 2. Each pin takes 14.5 s: travel at TCP 13 cm over the kit cell (2.0 s; 3.0 s for
the first move), down to TCP = pin length + 1 mm (1.5 s), settle 0.5 s, close (1.5 s), lift to 13 cm (1.5 s; the
carried pin's bottom then clears the plug top by 23 mm or more), carry over the chamber's centre (2.5 s), settle
0.5 s, down to TCP 7.9 cm (1.5 s: the pin's top at 78 mm, its bottom 10 to 28 mm inside the chamber with 5 mm
clearance a side, the pad tips 6 mm above the plug top), settle 0.5 s, open (1.5 s: the pin drops 18 mm onto its
floor and its top comes to 60 mm, the shear line) and straight up to 13 cm (1.0 s). The right arm moves over its own
plug and waits there until 7.5 s (its ready pose is not reachable with a straight-down grip), then runs the mirror
image of the left arm's sequence. So one arm is always at its plug while the other works in the kit: the left arm
is low in the kit at 3.0-8.0, 17.5-22.5, 32.0-37.0 and 46.5-51.5 s, the right at 10.5-15.5, 25.0-30.0, 39.5-44.5 and
54.0-59.0 s. The offline check keeps the two arms' collision capsules at least 4.5 cm apart (TCPs 23 cm or more).
After its last rise each arm holds at TCP 13 cm, 70 mm above the pin top (beyond the 50 mm let-go radius). The left
arm ends at 59.0 s and the right at 66.5 s; offline, k = 8 holds for its 1 s at step 1,341 (67.05 s); in simulation the
trial ended at step 1,342 (67.1 s) with k = 8. The run limit is 2,400 steps (2 minutes at 20 Hz).

## Layout, seeds and reset

The layout is fixed (`-E object_jitter=0`): there are no seeds. Reset: stand each pin back upright at the centre of
its kit cell, row n = size n (size 1 in the row nearest the robot), in both columns. The plugs, keys and kit are
fixed; nothing is consumed. The scorer starts over on every episode reset (rigstate's reset hook; it also spots the
pins being put back on their marks in one step).

## Runtime per trial

On an L40S once Isaac Sim is running: about 4 minutes for the scripted demo (it ends at step 1,342, 67 s of simulated
time, once k = 8 has held for 1 s) and 6 to 9 minutes for a trial at the full 2,400 steps (2 minutes of simulated time). The first start of Isaac
Sim on a new instance takes several minutes more.

## Real-world build

- Plugs: 3D-printed or machined aluminium, as one piece or from the listed blocks (two 186 x 8 x 60 mm side walls,
  five 10 x 34 x 60 mm cross walls), on a base board, with the four 34 x 34 mm floor blocks glued in at 20, 32, 14
  and 26 mm tall. The floors and the 12 x 30 x 30 mm bow can be printed and painted silver.
- Pins: turned from 24 mm 6061 aluminium bar to 28, 34, 40, 46 and 52 mm (two of each) and anodised or painted gold;
  each weighs 34 to 64 g.
- Kit: a printed grid of 4 mm walls, 6 mm tall, with ten 42 x 56 mm open cells, taped to the bench.
- Scoring each chamber: lay a straightedge across the plug top over the chamber. The pin is flush if a 2.5 mm feeler
  gauge does not pass between the straightedge and the pin top, and the pin top does not lift the straightedge
  2.5 mm or more. Judge tilt by eye against a square; stillness, let-go (50 mm) and anything over a chamber from the
  video's last second and a top photo.

## How it differs from nearby tasks

The board (668 tasks) has nothing on rekeying, pin tumblers or keying alike (searched rekey, pin tumbler, shear
line, key pin, keyed alike, locksmith, lock cylinder, bitting, pinning). The check needs no code sheet: a
straightedge across each plug shows every wrong pin 4.6 mm or more proud or low (6 mm when upright), so the task is
producing a measured coplanar surface (eight pin tops in the plugs' top plane), not following a lookup rule. The
nearest:

- TK-TPC59R ('Pick the pin that fits the hole and seat it'): one pin chosen by diameter as a go/no-go (only one
  enters), scored on seating depth in one hole. Here every pin enters every chamber; eight pins are chosen by length
  from a numeric bitting code with spares, and the outcome is that eight tops across two cylinders lie in one plane,
  the shear line, checkable with a straightedge without knowing the code.
- TK-F49W29 (keycaps onto switches by a colour card): colour to position, seated on a switch, no height outcome.
- TK-8YB397 (plug a manifold's open ports): identical plugs; the rule is which port to leave open.
- TK-XE2VEC (a fuse by rating) and TK-HVH57C (a dunnage block by width to remove play): one item each, other trades
  and outcomes. TK-F545PQ and TK-7AHCHJ: identical items, no selection. TK-MYZWLH (turn a key in a reverse cam
  lock): a different act. TK-RNWKBY: cans ordered by height on marks.

## Results in simulation

Measured with the commands above on an L40S (Isaac Lab-Arena 0.3.1, `-E object_jitter=0 -E control_hz=20`, one trial each):

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 0.5 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.5 |
| Dream-1 | 0.0 |

The scripted demo reached k = 8 and the trial ended at step 1,342, well inside the 2,400-step limit; every pin ended
upright (0 degrees) with its top at the shear line and no movement at rest. With either arm idle the other arm keyed
its own plug (k = 4). The Dream-1 trial ran the full 2,400 steps without moving a pin (k = 0).
