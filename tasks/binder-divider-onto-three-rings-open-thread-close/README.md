# Thread a plastic divider onto all three rings of a binder and close the rings

An Avery Durable View binder with 1 in Slant Rings (item 17011, black, about US$9) lies open and flat on the table,
its spine along y. A loose Avery plastic divider (from an 8-tab set, about US$6; US letter, 279.4 x 215.9 mm) stands
in a slanted desk letter sorter (about US$15 est) on the binder's front cover. The robot must open the rings, put the
divider's three holes onto the three rings, close the rings with the divider on them and take its hands away.

Instruction (task.json and `task.py` `INSTRUCTION`, identical): "Open the binder's rings, put the loose plastic
divider onto all three rings, close the rings with the divider on them, and take your hands off everything."

Status: the revised demo scored 1.0 on the GPU (21 Nov 2026); the first GPU run had scored 0.3 (see "GPU results").

## The act

1. The right hand presses the ring mechanism's thumb lever down: the three near half rings turn open together (the
   tips of each ring 23 mm apart at the stop) and a detent holds them there.
2. Both hands pinch the divider's far edge, one 120 mm either side of its middle, draw it up out of the sorter and
   carry it over the rings tilted 44 degrees, punched edge first.
3. Holes in line with the open half rings' tips, the hands move the divider along the wires' curve: the three holes
   go 12 mm onto the rings at once.
4. Still holding it, the hands carry the divider about 17 mm toward the rings' fixed halves, the way the threaded
   wire points go as the rings shut: the holes' edges drive the three near half rings shut together (the detent takes
   them home under 0.25 rad).
5. The hands open: the divider leans back onto the sorter about the shut rings. Hands away.

## Why it is hard

1. **Three holes at once.** The holes meet three rings 108 mm apart: all three or none. A divider that is a few
   millimetres off along its edge, or turned a few degrees about the normal, has a hole on a tip while another misses.
2. **A pinch holds no torque about its closing axis.** One pinch on a 279 mm sheet lets it swing about the pads until
   its centre of mass hangs under them; the hole row then turns away from the rings. Two pinches along the edge hold
   the sheet's plane: this is why both arms are needed (one arm: 0.0).
3. **Order and state.** The rings must be open before the divider goes on and shut after; a divider hooked on the
   tips alone falls off; shut rings stop it entirely.
4. **Tilted approach.** The holes meet the open half rings along the wire's tangent, 46 degrees above horizontal at
   the tip, so the hands carry the sheet tilted (both wrists pitched 0.80 rad about y) and the gripper housings stay
   clear of the sorter and each other.

## Scoring

| Condition | Rule (sim) | Bench method (real trial) |
|---|---|---|
| Open | gate 0.55 rad or more open | a 16 mm spacer block passes between the middle ring's tips |
| Shut | gate 0.10 rad or less from home | a 3 mm drill bit does not pass between the tips |
| On a ring | that ring's wire crosses the divider's mid-plane inside its hole; rings open: at least 8 mm along the wire from its tip | look along the ring: it passes through the hole; the divider cannot be lifted off the open ring without sliding it 8 mm along the wire (mark the wire 8 mm from its tip) |
| Still | divider frame under 5 mm/s, gate under 0.1 rad/s | 30 fps video of the last 1 s, no visible motion |
| Hands off | both finger zones 30 mm or more from the divider | a 30 mm spacer block against each gripper |
| Latch | any part of the divider more than 3 mm below the table top, or past the table's edge, at any step | by eye against a rule |

| Outcome | Score |
|---|---|
| Latch fired, or rings shut with the divider not on all three | 0 |
| Rings open, divider not on all three | 0.3 |
| Divider on all three rings, but not (shut, still, hands off) on every step of the last 1 s | 0.7 |
| Divider on all three, rings shut, still, hands off on every one of the last 20 steps (1 s) | 1.0 |

Possible scores: 0, 0.3, 0.7, 1.0. Success: 1.0 held for 1 s. Nothing reads the lever or the order of moves, only
poses, so any way of getting the holes round the wires and the rings shut scores the same.

## Model

* The binder, covers, spine, plate, the three fixed half rings and the sorter are fixed boxes (4 mm square wire in
  seven boxes per half ring: a half-ellipse 26 mm wide and 27 mm high, not a D shape).
* The three near half rings and the thumb lever are one rigid link ("gate") on a hinge along y at the halves' feet,
  with a spring to a rule-based two-notch detent (shut, open 0.80 rad): `task._detent` sets the drive target each step.
* The divider is a loose 2 mm rigid plate with three 8 mm square holes (real: 0.2 mm flexible, round holes about
  6.4 mm), 0.05 kg est, friction 0.35 / 0.30.
* All dimensions marked est in task.json are design values; measure the real parts as the setup says. The thumb lever and
  tip gaps are design values the real binder must meet: the setup lists acceptance checks and a fallback (a steel tab).

## Run

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 -E control_hz=20 ...

Run limit 2400 steps (120 s); the demo is 1070 steps (53.5 s). Settle check: `DEMO_HOLD_ONLY=1`. One arm:
`DEMO_IDLE_ARM=left` or `right`. Counterfactuals: `DEMO_VARIANT=open` (rings opened, then stop: 0.3) and `hang`
(divider let go in the air in front of the tips: 0.3). Trace: `TASK_TRACE=<file>`. Seed: `object_jitter=0`, nothing
random.

## Offline checks (this folder)

    python3 -m unittest discover -s tests -q
    python3 tools/offline/check_task.py tasks/binder-divider-onto-three-rings-open-thread-close --require-json --strict --one-arm

The offline model (`offline_binder.py`) is a kinematic stand-in: the lever presses the gate open by pad height, a
closed pad pushes it shut as far as the wire's surface allows, a divider pinched by one or two closed grippers follows
the hand (one hand: swings under its pad; two hands: it sags by the carry sag measured on the GPU), a held divider
whose holes' edges reach the movable wires drives the gate shut, let go on the rings it slides down the half rings. No
friction, inertia, contact forces or collision of the divider with the wires.

## GPU results

| Run | Result |
|---|---|
| settle (`DEMO_HOLD_ONLY=1`), 2026-11-21 | divider still in the sorter; the shut gate sags from 0.03 to 0.08 rad under the lever's weight against the weak spring (shut is 0.10 or less) |
| demo, first version, 2026-11-21 | 0.3: rings opened (gate 0.849), but the holes struck the wire tips: the divider rode 6 mm low in the pinch and the straight in-line path cut the wires' curve, so the tips met the sheet 3 to 9 mm off the holes' centres. Let go, it fell beside the rings; the left pad then came down on it instead of the wire and the rings stayed open |
| demo, revised (carry trim, path along the wires' curve, shut by the held divider) | 1.0 |
| one arm, left idle | 0.3 |
| one arm, right idle | 0.0 |
| Dream-1 | 0.0 |

## GPU risks (what to read first)

* **Pinching a 2 mm sheet.** The grippers close to a gap under the 16 mm minimum on a thin plate (finger target 0,
  both fingers saturate at 50 N with the sheet between the pads): the sheet may slip or be shoved to one pad.
* **Pad axis vs the sheet at the pick.** The sheet leans 44 degrees; the pads' closing axis is 46 degrees above
  horizontal, so they close nearly square on it, but a few degrees either way turns it in the hands.
* **Threading is open loop.** The holes have 2 mm to spare around the 4 mm wire and the arms trail 3 to 7 mm under
  load; the hole row must be within that of the three rings along y and along the wire's tangent.
* **Shutting by the held divider.** The holes' edges drive the wires (20 to 40 mm of hole travel along v per rad of
  gate lag); the hands aim 0.15 rad past shut so a lagging gate is driven to its stop. A pad push is not possible
  once the divider is on: it covers every ring from y -140 to +140 mm.
* **Shut margin.** Left alone the shut gate rests at about 0.08 rad (the lever outweighs the half rings about the
  hinge; spring 0.05 N m/rad), close to the 0.10 rad limit. The divider resting on the movable halves presses them
  shut; a divider that pulled them open by 0.02 rad would cost the 1.0.
* **Let go on shut rings.** The divider leans back onto the sorter about the rings; the open lower finger may carry
  it for a moment while the hand draws back along the sheet's plane.
* **Hands near the sorter.** The wrists pitched 0.80 rad and the housings near the sorter's lip: capsule proxy shows
  contact with the sorter's bounding box only, which is larger than the sorter.
* **The thumb lever** needs 2 to 4 N from a closed pad; the hinge's spring is 0.05 N m/rad (it left the gate at
  0.849 rad on the GPU, past the 0.80 rad stop; the threading is aimed at that angle).
* mechkit tight-fit parts and the detent rule are version 2 features (not GPU-verified).
