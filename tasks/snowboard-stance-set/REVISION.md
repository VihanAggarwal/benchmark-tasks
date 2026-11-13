# Revision for review TK-CRLXDF (PR #857)

Reviewer: "Revise to avoid this being a simple pick and place task. Include mounting-hole alignment and fastening."

## The review, sentence by sentence

**"Revise to avoid this being a simple pick and place task."** The binding is no longer set down on a mark. It has
to land flat with its four slots over four real bores (each insert's centre within 4.0 mm of its slot's centre
across the slot) and its angle within 3 degrees, and then be screwed down with eight loose screws: each one picked
by its 12 mm head, fed with its 5 mm shank through a 10 mm slot into an 8 mm bore, and turned a full turn clockwise
in regrips while it stays seated. Carrying the binding is still a pick and a placement; what the earlier version
lacked, the holes and the screws, is now the larger part of the trial (four screws per arm, 46.0 s each, against
20.8 s for the binding).

**"Include mounting-hole alignment ..."** Answered physically. The 16 grey dot stickers are now M6 inserts on the
standard 4 x 4 pattern: 8 mm bores, 9 mm deep, in the board. Each binding has a slotted 4 x 4 disc (four 10 x 24 mm
slots on a 40 mm square) preset in its baseplate, +15 degrees on the nose binding and -15 on the tail binding. The
slots sit over four inserts only with the disc square to the board (within about 10 degrees by the scorer's window;
the angle rule asks for 3), and that puts the baseplate at its preset angle. A binding turned heel forward or
swapped with the other cannot be aligned.

**"... and fastening."** What is physical: each screw is a real-size M6 binding screw with its washer (12 mm head,
18 mm washer, 5 mm shank); the robot must pick it, feed it through the slot into the bore, and turn it 360 degrees
clockwise while its washer stays within 3 mm of the disc and it stays upright; lifting it or tilting it past
10 degrees restarts the count, and only contact friction and the screw's damping resist the turn. What is a rule,
stated in `fidelity`: there is no thread (smooth bores and shank cores), so the screw slides down to its seat where a
real one would thread in about 6.5 turns; after the 360 degree turn the screw is welded where it stands (driven),
and a binding whose four screws are driven is welded flat (clamped), each by a pose rewrite every control step. The
360 degrees match the real build's last turn: at the M6 pitch of 1.0 mm, a real screw seated with its washer within
1 mm of the disc needs one more full turn to bring the washer down. A real trial reads it that way (a 1 mm feeler
gauge, then a paint witness mark for the last turn, and the binding not sliding by hand).

## What changed, in detail

1. **Real inserts**, as above: in two rows across the board (x 28 and 32 cm) and four rows along each half
   (|y| 14, 18, 22 and 26 cm).
2. **Slotted 4 x 4 discs, preset and fixed in the baseplates**, as above. The 40 cm stance centred on the cross means
   both discs go on the middle insert squares.
3. **Eight screws** standing in two holders. A screw is engaged once its shank is through a slot, 5 mm or more down
   an insert, upright within 10 degrees, washer within 1 mm of the disc; its clockwise turn is summed while engaged.
4. **Real-size binding.** 230 x 130 mm, with an 8 mm floor, 40 mm sidewalls and a 100 mm highback; about 0.35 kg (it
   was 0.25), carried by its highback. The rigid ankle-strap bar is gone (no straps). Still simplified, and stated in
   `fidelity`: the highback is a flat 100 mm plate, lower than a real one (a taller one would put the carry grip above
   the arms' reach); the slots leave the 5 mm shank 2.5 mm of play a side (real slots fit the screw closely) to take
   up the arm's placement error; the inserts are smooth bores and the shanks smooth cores (no threads).
5. **Scorer rewritten.**
   - Every score needs a physical state: slots over the inserts with the binding flat, then screws seated, then
     screws driven, then the binding clamped.
   - Release is checked from finger geometry on every step of the last 1 s, and it fails closed: an unreadable
     gripper, or a part whose distance cannot be computed, never counts as released.
   - Every part must be still for 1 s. Nothing held or hovering earns credit.
   - One latched zero: a part that falls off the table.
   - Tiers: 1.0 both bindings fastened on the middle squares; 0.75 both fastened elsewhere; 0.5 both aligned with
     four seated screws; 0.25 one aligned with four seated screws.
   - A real-trial reading for every threshold: slots by eye, a 1 mm feeler gauge, a witness mark on each head, the
     binding not sliding by hand, a protractor against the board's centre line, a tape for the stance.
6. **Instruction states the goal only (140 characters):** "Mount this rental snowboard's bindings for a 40 cm duck
   stance centred on the red cross, heels on the edge nearest you, and screw both down."
7. **Demo.** Open-loop, both arms at once, mirrored. Each arm carries its binding by the highback to the middle
   square, turning it 75 degrees, then fetches its four screws one at a time: it lowers each through its slot into
   the insert until the washer is 0.5 mm above the disc, lets go, grips the head again 0.5 mm above the washer and turns it clockwise in six
   180 degree strokes, rising 2 mm open to turn back between them (1,080 degrees commanded for the 360 the rule
   needs, the rest a margin for a stroke lost when the pads pop off the round head). The demo takes about 210 s. The run limit is now 6,000 steps.

## Status

- Offline checks pass: the unit tests and `check_task.py --require-json --strict --one-arm` (predicted 1.0, and 0.25
  with each arm alone).
- GPU run 1: settle 0.0 with nothing moving; demo 0.5 (both bindings aligned, all eight screws seated, seven driven:
  one screw reached 278 of 360 degrees because the pads slipped on its head). The demo's turning was raised from 450
  to 720 degrees per screw in response.
- GPU run 2: settle 0.0; demo 0.5 again (both aligned, all eight seated, seven driven: `screw_t4` reached 353 of
  360 degrees after its count restarted part way through its second stroke; another screw lost a stroke when the pads
  popped up off its head). The demo now grips the seated head 1 mm lower (4.5 of its 5 mm held), rises 2 mm open
  before each turn-back so the fingers do not drag the washer, holds each regrip 0.3 s, and turns six strokes
  (1,080 degrees) instead of four. Demo only: the scoring, its thresholds and the task text are unchanged.
- GPU run 3 (job 20261112T230200): demo 1.0 (all eight screws driven and both bindings clamped; success at step 4,221 of 6,000), 0.25 with either arm idle, Dream-1 0.0. These scores are in `task.json`.
