# Change a mini centrifuge to its strip rotor: turn it onto the pin, click it on, load

A Corning LSE 6770 mini microcentrifuge (white body, smoked lid, 174 x 148 x 122 mm, 1.1 kg) stands loose on the table, unplugged, its
lid shut and latched, the 8-place tube rotor snapped on its shaft. The 4-place strip rotor lies hub up beside it, a
quarter turn off the shaft's key, and two equally filled 8-tube PCR strips stand in a rack. The robot must change the
rotor and load the strips the way a technician does: press the lid's tab, lift the tube rotor off and leave it on the
bench, turn the strip rotor to the key so it drops over the shaft, push it until it clicks, put the two strips in
opposite slots, and shut the lid until it latches.

## Why it is hard

1. **Into position.** The maker's manual says to place the new rotor "into position on the motor shaft and snap in";
   the coupling's shape is not published. In the sim it is a cross pin (an estimate, checked on the real unit). Turned more than 10 degrees off the key
   (two keyed angles, half a turn apart) the strip rotor rests on the pin 8 mm proud and will not go on. It must be
   turned a quarter turn on the way.
2. **A snap that needs a push.** On the key the rotor drops only onto the snap's two sprung noses, about 4.8 mm proud. Let
   go there it looks fitted but is loose; it needs a straight push of about 3 N to click down onto its seat. The tube
   rotor comes off only by pulling past the same snap (the manual's way is a thumb on top of the shaft and two
   fingers lifting under the rotor; here one hand pinches the hub and the unit's own 1.1 kg holds it down).
3. **Order.** The lid opens only by its tab; the strips go in only once the strip rotor is on; and the lid must latch
   last: latching on a loose rotor, a single strip or an uneven load scores 0 (the real unit would start spinning), and
   once the lid has been opened and latched shut, pressing the tab to free it again scores 0 (on the real unit that
   shut started a run).
4. **Two hands on one clock.** One hand presses the tab and must get off it for the lid to pop up while the other
   waits out of the lid's sweep; one hand clears the chamber before the other comes in with a strip; the lid is
   swung shut from behind, not slammed.
5. **Small, light parts.** The strips are 72 x 7 x 21 mm and 1.66 g, pinched across their 7 mm cap rows and dropped into 10 mm slots; the
   rotors are 30 to 40 g and are pinched by a 22 mm hub.

## Scoring

The scorer is `scoring.py` (plain functions, NumPy only). It reads the poses of the unit, the lid, the tab, the two
snap plungers, the rotors and the strips, and both grippers' finger zones (rigstate) every control step.

| Condition | Rule |
|---|---|
| Latch | the scorer's own copy of the catch (`layout.Latch`): shut and latched at the start; the tab pressed 2 mm frees it; it catches again once the lid has opened 3 degrees and is back within 0.6 degrees of shut with the tab up |
| Latched | caught and the lid within 1 degree of shut |
| Seated | the strip rotor's hub within 1 mm of the shaft's axis, its underside within 0.5 mm of its seat, tilted under 3 degrees, both plunger noses back out within 0.2 mm (the click) |
| On shaft | the hub within 2.5 mm of the axis, its underside under 10 mm over its seat, tilted under 8 degrees (on the pin or on the noses) |
| Strip in | its middle within 2.5 mm of a slot's middle line and 4 mm along it, its underside within 2 mm of the slot floor, tilted under 10 degrees, turned under 15 degrees from the slot |
| Balanced | both strips in the two slots of one opposite pair (the inner two, 19 mm each side of the hub, or the outer two, 31 mm); either strip in either slot |
| Stored | the tube rotor's lowest point within 4 mm of the table top, tilted under 10 degrees, 5 mm or more clear of the unit's footprint |
| Hands off | over the last 1 s (20 steps), every finger zone 30 mm or more from the unit, the rotors and the strips |

| Outcome | Score |
|---|---|
| At any step: a rotor or strip off the table (20 mm under its top), the unit tilted over 10 degrees (tipped), the lid forced over 2 degrees off its stop while caught, or, once the lid has been opened and latched shut, the tab pressed far enough to free it again | 0 |
| At the last step: hands off failed, or latched with the strip rotor not seated or the load not balanced | 0 |
| Seated, balanced, latched, stored | 1.0 |
| Seated, balanced, stored, lid not latched | 0.75 |
| Seated, stored, not balanced, lid not latched | 0.5 |
| Stored, the strip rotor on the shaft but not seated, lid not latched | 0.25 |
| Anything else | 0 |

Success: 1.0 held for 1 s. Score over the run: the last step. The real-trial checks for each condition are in
task.json (scoring).

Ways round the rules we tried, each with a test in `tests/test_scoring.py`: latching on the loose (unclicked) rotor;
one strip, or two strips in non-opposite slots; latching first and opening again to fix the load; prying the lid up
while it is caught; leaving the tube rotor on top of the unit or in the chamber; teleport-style resets; a hand resting
on the lid at the end.

## The mechanism (simulated)

- **The unit** is one loose articulation (`mechkit.py`): the body with its chamber, deck, front panel, shaft, collar
  and cross pin; the lid on a hinge at the top back edge (spring 0.15 N m/rad, damping 0.10 N m s/rad, back stop at 75
  degrees); the tab on a 3 mm slider on the lid's front lip (1500 N/m); two sprung plungers in the shaft (0.9 mm travel,
  1300 N/m each) whose 30-degree noses are the snap.
- **The key** is contact only: a 16 x 3 x 3 mm pin through the shaft, 5 to 8 mm over the seat, and two slots under
  opposite walls of each hub. The pin passes a slot within about 10 degrees of the key (1.5 mm of play).
- **The snap** is contact only: keyed, the hub's bore lands on the noses' upper faces 4.8 mm proud (4.5 to
  4.7 mm in the GPU run, the plungers giving under the rotor's weight); pushed, the faces drive the plungers in and the
  noses spring out into the groove inside the hub. About 3 N in all to push on or pull off (est).
- **The slots** are contact only: four 75 x 10 mm slots, 17 mm deep, in the strip rotor's top.
- **The catch is the one declared rule** (`task.py` `_lid_rule`): every step `layout.Latch` is fed the lid's angle and
  the tab's travel and sets the lid spring's target: -2 rad while latched (0.30 N m holding it shut), +2 rad once freed
  (the pop-up spring: 0.30 N m shut, 0.10 N m at the stop, so the lid swings up in about 0.7 s and stays). The scorer
  runs its own copy of the same state machine.

## Results

GPU: every figure below is from the full run of this version on 2026-11-19 (job 20261119T063532), read from its
per-step trace (`TASK_TRACE`). The setup screenshot and the videos are of this version, from the same run. The offline
column is `tools/offline/check_task.py` with this task's offline stand-in for the physics (`task.OfflineModel`, offline
only): a prediction, not a measurement.

| Run | What it does | Offline prediction | GPU score (this version) |
|---|---|---|---|
| Nothing (`DEMO_HOLD_ONLY=1`, 160 steps) | both arms hold still | 0.0 | 0.0: nothing moved (lid latched at 0 degrees, tube rotor seated, plungers 0 mm, unit moved 0 mm) |
| Scripted demo | below | 1.0 (1.0 held from step 1890) | 1.0, held for 1 s at step 1890 (1890 of 2400 steps), where the run stopped |
| `DEMO_IDLE_ARM=left` | the right arm opens the lid, lowers the strips onto the tube rotor still on the shaft and shuts the lid: latched with no strip rotor on | 0.0 | 0.0: as predicted; the strips rest on the tube rotor, which never left the shaft, and the lid latched over them |
| `DEMO_IDLE_ARM=right` | the lid stays latched; the left hand meets the shut lid, cannot take the tube rotor off and leaves the strip rotor on the lid | 0.0 | 0.0: the lid stayed latched (tab never pressed); carrying the strip rotor in toward the shaft with the lid still shut, the left hand pushed the unit 85 mm (score 0 from step 652); the strip rotor ended on the table |
| Dream-1 | `--policy dream`, full 2400-step limit | not predicted | 0.0: the arms moved but never pressed the tab; the lid stayed latched and nothing in the scene moved |
| `DEMO_VARIANT=nopress` | the strip rotor let go on the noses, never pushed: the lid latches on a loose rotor | 0.0 | not run |
| `DEMO_VARIANT=offkey` | the strip rotor lowered a quarter turn off its key: it stays on the pin, 8 mm proud | 0.0 | not run |
| `DEMO_VARIANT=onestrip` | one strip loaded: the lid latches on an unbalanced load | 0.0 | not run |
| `DEMO_VARIANT=openlid` | everything but the lid | 0.75 | not run |

The demo on the GPU (the trace samples about every 23 steps, then every step at the end): the tab reached 2.08 mm
and freed the catch by step 93 (4.7 s); the lid was at 14 degrees at step 116 and at its 75 degree stop by step 139
(7.0 s), where it stayed until the right hand swung it shut. The tube rotor left the shaft between steps 277 and 300
and was 64 mm up by step 323 (16 s) and lay on the table, its lowest point 0 mm over the table top, by step 438. The strip rotor
landed on the noses on the key by step 737 (37 s; 4.5 to 4.7 mm proud, 0.3 mm off the axis) and clicked onto its
seat by step 875 (44 s). One strip was in a slot by step 1105 (55 s) and the other in the opposite slot by step 1473
(74 s): the inner pair, slots [2, 1] in the trace. Letting go of the first strip nudged the rotor 0.3 mm up for a
moment (step 1059); it was back on its seat by step 1082. The lid was swung shut after step 1634 and was latched by
step 1795 (90 s): one opening, one catch, never freed again. At the last step (1890): lid 0 degrees and latched, highest
opening 75 degrees; strip rotor 0.0 mm over its seat, 0.24 mm off the axis, tilted 0.01 degrees, plungers 0.0 and
-0.03 mm (clicked); both strips in opposite slots; tube rotor flat on the table 207 mm from the shaft; unit moved 0 mm
and tilted 0 degrees over the whole run; hands off for the last 1 s.

Two arms are needed: the strip rotor lies on the robot's left (y +200 mm) and the rack on its right (y -200 mm), and
neither arm reaches both (offline IK: the right gripper misses the strip rotor by 40 mm or more, the left misses the
rack's strips by 32 mm or more). The two hands also share the work in time: one opens the lid and fetches the strips
while the other changes the rotor. With either arm idle the offline model predicts 0, and the GPU run scored 0.0 with
either arm idle.

## Demo

Open loop: timed waypoints from the known start layout, nothing read from the simulator. Both wrists point straight
down. The plan runs 94.8 s (1896 steps) of the 120 s limit. With the lid at its stop its underside leans over the
chamber's back half (about 87 mm behind the shaft at the deck, 59 mm at 210 mm up) and its side skirts hang 23 mm
further forward at y +-85. Every move over the open chamber is made with the wrist at yaw 0 or turned half a turn
(fingers closing along y), never turned a quarter turn (the 102 mm gripper housing then reaches into the lid); the
fingers open only 36 mm over a hub and 22 mm over a strip, never wide (open, the finger carriages reach the skirts).

| Time (s) | Left gripper |
|---|---|
| 0 to 9.5 | fingers 36 mm apart, wait out of the lid's sweep at x 120, y +220 |
| 9.5 to 14.5 | in past the lid's side skirt, over the tube rotor's hub, down, pinch it (20 mm commanded on the 22 mm hub: about 5 N a finger) |
| 14.5 to 23.2 | lift it 165 mm straight off, past the snap; forward out of the chamber, over the bench at x 100, y +200, lower to 3 mm over the table, let go, up |
| 23.2 to 29.4 | over the strip rotor's hub with the wrist turned a quarter turn, down, pinch it, lift it |
| 29.4 to 33.9 | turning the wrist back a quarter turn (the rotor turns onto the key) in front of the open lid, in past the lid's side skirt, over the shaft |
| 33.9 to 38.4 | lower it until the hub is 0.7 mm over the noses, settle, let go (it drops onto the noses), up |
| 38.4 to 43.9 | close the fingers into a block, down onto the hub, push 10 mm past flush (the snap's travel and the arm's give), hold: it clicks on |
| 43.9 to 48.6 | up, forward away from the lid, back to its own side, aside |

| Time (s) | Right gripper |
|---|---|
| 0 to 5.5 | fingers closed into a block over the tab's front 11 mm (3 mm clear of the lid itself), wrist a quarter turn; press 10 mm under the tab's top: the catch lets go |
| 5.5 to 6.1 | back off 60 mm forward and 70 mm up in 0.6 s: the lid springs to its 75-degree stop behind the block |
| 6.1 to 45.9 | aside, wait; over the first strip's back end in the rack, down either side of its cap row, pinch it across the row (5 mm commanded on the 7 mm row: about 5 N a finger, 30 mm behind its middle), lift it out |
| 45.9 to 49.4 | once the left hand has left: up toward the unit's front while turning it end for end (half a turn in two quarter turns), across in front of the unit to line up with slot 1, 155 mm over the table |
| 49.4 to 54.7 | in to 10 mm short of the slot's middle, down to 4 mm over the rotor's top, along the last 10 mm, down into the slot, settle, let go (22 mm): it drops onto the slot floor |
| 54.7 to 58.6 | up just clear of its caps, straight up, forward and out to the side, the wrist turned back |
| 58.6 to 77.8 | the second strip, into slot 2 (the opposite slot) the same way |
| 77.8 to 82.3 | fingers closed into a block; beside the unit behind the open lid, in, onto the lid's back 90 mm in front of the hinge |
| 82.3 to 89.3 | swing the lid shut in ten 0.7 s steps, the block commanded 6 mm into the lid's face |
| 89.3 to 94.8 | press 10 mm past shut (the block commanded into the shut lid): the catch clicks; hold; up; aside |

Offline (`tools/offline/check_task.py`): worst IK 0.09 mm; minimum joint margin 2.4 degrees (right joint 4, putting
the block on the lid's back; right joint 6 keeps 6.6 degrees with the strip turned end for end); closest TCPs 221 mm
apart; the arms' collision capsules 15 mm apart at closest (39 s, link 4 to link 4: the left hand over the shaft,
the right going to the rack). Gripper housings (the yamkin capsules of the gripper and both finger carriages, not
the pads), with the lid at the angle the offline model gives it at each step: left hand 11.1 mm from the lid at
closest (in past its side skirt) and 39 mm from the unit's body; right hand 12.6 mm from the lid (picking the
second strip, under the lid's raised corner) and 23 mm from the body; the two housings 84 mm apart
(`test_housings_keep_10_mm_from_the_lid_unit_and_each_other` checks 10 mm over the whole plan). The left arm's link
4 passes 3 mm under the open lid's back corner while it pinches the tube rotor (on the GPU it touched nothing: the
lid stayed at its 75 degree stop and the unit did not move).

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 2400
```

Seed: nothing is random; `object_jitter=0` fixes the start layout. `TASK_TRACE=trace.jsonl` writes one line per
control step: lid angle, tab travel, the catch's state, each rotor's rise over its seat and offset from the axis, the
plungers, which slot each strip is in, stored, hands off.

## Simplifications

- Rigid bodies. The round chamber is a 16-sided wall; the rotors are built from flat plates (the tube rotor's disc a
  16-sided plate, 96 mm across its flats, with round tube holes drawn on it). The real quick-release
  shaft's internals are not published: the cross pin and the two sprung plungers stand for its coupling and snap,
  built to behave as the operator's session checks describe (goes on only in position, snaps in with a click).
- Looks: every shape in this build is also a collider, so the looks were changed by colour only, never by shape (the
  GPU run's physics stays valid). The body is drawn as a white box (the real one has rounded corners); the lid, a
  smoked see-through moulding on the real unit, is drawn opaque in a dark smoked tint with a darker rim and tab
  (fractional opacity is not proven on the simulator); the strips are drawn white; the strip rotor is drawn mid grey
  and the tube rotor dark grey (their real colours were not checked).
- The lid's catch is a declared state machine driving the hinge spring's target, not a contact latch.
- The water is mass only (sealed under closed caps); each strip is one rigid body.
- No motor, no spin and no lid switch; the unit is unplugged.

## Sources and measurements

- The maker's manual for this model: "Model No. 6770 Corning LSE Mini Microcentrifuge Instruction Manual",
  https://www.corning.com/catalog/cls/documents/equipment-manuals/CLS-EQ-123-EN.pdf (read through its copy at
  https://www.manualslib.com/manual/2289896/Corning-Lse-6770.html , pages 2 to 4). Its rotor change, word for word:
  "The microcentrifuge is supplied with two interchangeable quick release rotors. No tools are required to exchange the
  rotors or attach them to the motor shaft. The rotors can be easily exchanged with one hand by way of the quick
  release system. Removal of the rotors is done by placing your thumb on top of the motor shaft and lifting up with two
  fingers underneath the rotor. Place the other rotor into position on the motor shaft and snap in." The lid: "To begin
  a run, simply close the lid of the microcentrifuge." and "Fast Stop: Press down and release the tab on the front of
  the lid. This will release the lid lock, allowing the lid to open". Size and mass: 6.8 x 5.8 x 4.8 in. (17.4 x 14.8 x
  12.2 cm), 2.43 lbs./1.1 kg. The two rotors: one for 1.5/2.0 mL tubes (eight places), one for four 8 x 0.2 mL strips.
- The product page: Corning LSE Mini Microcentrifuge, cat. 6770 ("Quick release rotor system eliminates the need for
  tools to exchange rotors", "Dual lid switch provides safe operation"; up to 8 microtubes or 4 PCR strips), list price
  321.78 USD, spare rotors 6770-RT 52.12 USD and 6770-RTS 36.62 USD (Nov 2026),
  https://ecatalog.corning.com/life-sciences/b2b/US/en/p/6770 ; a US distributor lists the unit at 323.70 USD, in
  stock, https://www.labdepotinc.com/p-59829-corning-lse-mini-microcentrifuge
- The strips: Axygen PCR-0208-AF-C, 0.2 mL 8-strip tubes with attached flat caps,
  https://ecatalog.corning.com/life-sciences/b2b/US/en/p/PCR-0208-C

The maker publishes no drawing of the shaft, the hub or the snap, no forces, no lid opening angle and no rotor masses.
The manual says the rotor is placed "into position" and snapped in, and that the tab releases the lid lock; it does not
say how many positions there are or whether the lid springs open. Those figures in the sim are estimates (listed in
task.json fidelity). A lab does not have to reproduce them: the operator's once-per-session checks in the setup ask only
for the behaviour the task depends on (the rotor goes on only in position, snaps in with a felt click, drives the
shaft; the tab frees the lid and it rises and stays open). The sim's values:

| Estimate | Sim value | What the operator checks on the real unit |
|---|---|---|
| Key pin height, play | rests 8 mm proud off the key; drops within about 10 degrees, two keyed angles | (a) out of position it does not go on; turned slowly it drops only at its keyed position(s) |
| Snap push-on and pull-off | about 3 N each | (b) a firm straight push snaps it on with a felt click; it lifts off by hand; clicked hub-top height marked |
| Hub | 22 mm square, 16 mm tall, 9 mm bore | (c) seated, turning it turns the shaft |
| Lid stop, opening time | 75 degrees, about 0.7 s | (d) the tab frees the lid; it rises and stays open |
| Tab release | 2 mm at about 3 N | (d) |
| Rotor masses | 40 g tube, 30 g strip | not needed |
| Strip rotor and slots | 80 x 76 x 20 mm; slots 75 x 10 mm, 17 deep, 19 and 31 mm off the hub | not needed (the real rotor as sold) |
| Strip mass | 1.66 g with 8 x 20 uL water | (e) both strips the same within 0.05 g (0.01 g scale) |

## Real build and reset

A Corning LSE 6770 with both rotors (about 320 USD list, both rotors included), a 96-place 0.2 mL rack, two Axygen
8-strips with 20 uL of water in each tube under closed caps, tape for the unit's feet, a steel rule, a 0.01 g scale.
Reset: press the tab and open the lid; lift the strips into rack columns 4 and 9, each along x; lift the strip rotor
off (thumb on top of the shaft, two fingers under the rotor, as the manual says) and lay it hub up at x 250, y +200, a
quarter turn off the key; push the tube rotor on, turning it
until it drops, and press until it clicks; shut the lid until it latches; feet in their marks. Nothing is used up.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The text; parts, the catch rule, `success()`, `score()`, `demo()`, variants, offline model |
| `layout.py`, `parts.py`, `scoring.py` | Every dimension and the catch's state machine; the parts as shapes; the scorer |
| `mechkit.py`, `rigstate.py` | Build the parts as articulations and compound rigid bodies; read both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.
