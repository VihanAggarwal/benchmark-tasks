# Weigh a rock on a triple beam balance: riders in order, poise slid to zero

An OHAUS 750-S0 class mechanical balance (610 g, 0.1 g) stands fixed on the table. The empty balance reads zero. A
granite specimen lies in front of it. The robot sets the rock on the pan, moves the 100 g rider and then the 10 g
rider along their notched beams, and slides the smooth 0 to 10 g poise until the pointer rests on the red zero mark.
The instruction does not give the weight. The answer is read off the instrument: 200 g + 30 g + 6.4 g = 236.4 g.

Category: Societal relatability (the school-science weighing most adults did firsthand). One arm does all of it; the task does not claim two arms.

## Why it is hard

1. **A quantity along a path with a swinging, damped beam.** The beam turns on a knife edge, is held by a self-righting
   spring (0.5 g at the pan per degree with the rock on the pan) and a damper (about 3 s to settle). With the rock on the pan and a wrong weight
   the beam sits on its +-8 degree stop; only the last 40 mm of poise travel bring the pointer off the stop, and it follows
   the poise with a lag, so the poise has to be crept (0.45 g per second in the demo).
2. **The notched riders** are drawn into a notch and need about 2 N to push out; they must be placed largest first, and
   a rider pushed one notch too far swings the beam the other way.
3. **Narrow windows.** The poise's whole travel is 100 mm for 10 g, so 0.5 g is 5 mm; the pointer's 1 degree is one scale
   mark (4.6 mm at the pointer tip).
4. **Nothing to copy.** The reading is a property of the instrument; a memorised pose is not the answer for another rock.

## Scoring (see task.json)

Reading = 100 x (100 g rider's notch) + 10 x (10 g rider's notch) + poise in mm / 10. Tiers: 0 (fell, specimen held or
moving, or not on the pan); 0.25 on the pan; 0.5 both riders in their notches (200 g and 30 g); 0.75 reading within 2 g,
pointer within 3 degrees, beam at rest 1 s; 1.0 reading within 0.5 g, pointer within 1 degree, at rest, both hands 30 mm
or more from the beam, riders and poise. A beam on its stop scores at most 0.5.

Bench check (real trial): video of the pointer against the scale and the three beam scales. The pointer rests within one
scale mark of the red zero; the riders sit in the 200 g and 30 g notches; the poise reads 6.4 g within 0.5 g on its
0.1 g graduations; the rock's reference mass (weigh it on a reference scale and trim it by grinding to 236.4 g within 0.1 g).

## Running it

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py --scorer task_score -E object_jitter=0 -E control_hz=20 --max-steps 1800

About 81 s of motion (1,628 steps); run limit 2,000 steps. No random seed: `object_jitter=0`, one fixed layout.
`DEMO_HOLD_ONLY=1` settles (nothing moves; the empty beam must hang at zero), `DEMO_IDLE_ARM=left|right` drops an arm's
plan, `DEMO_VARIANT=no_poise|no_riders|wrong_notch` runs the counterfactuals, `TASK_TRACE=<file>` writes one JSON line
per step (beam angle, reading, poise).

## Demo (open loop, right arm only; the left arm parks)

1. Grip the rock across its 48 mm (commanded 38 mm), carry it 190 mm up over the pan, set it 3 mm over the plate, let go,
   rise; wait 5 s while the beam drops onto its stop.
2. Closed fingers (the closed pad block is 16 mm along y, 20 mm along x) come down 12 mm left of the 100 g rider, push it
   at 10 mm/s to 2 mm short of notch 2 (the detent pulls it home), rise. Pushing a weight at a height over the knife
   edge turns the beam pan side up; the 10 g rider's push out of its zero notch takes the beam to its +8 degree stop at
   once (GPU run 2), so that push is planned for +8: it runs on to where the rider would stand 1 mm past notch 3 with no
   load (the arm trails its command by up to 10 mm while it builds the 2 N that pulls the rider out of notch 2), the pad
   tip 17.5 mm over the axis line at the end, then draws back 15 mm before rising, so the beam's return to -8 cannot
   carry the rider on under a pad that is still down.
3. The same beside the poise: 24 mm at 20 mm/s, then creep it to 65.5 mm at 4.5 mm/s, wait 3 s, rise (the beam comes off
   the push and settles on zero), park with the fingers closed.

## Results

| Run | What it did | Score |
|---|---|---|
| Offline prediction (`check_task.py`, `OfflineModel` in task.py: a kinematic and lever model, not a measurement) | all steps; success held from step 1442 of 1661 | 1.0 |
| Offline, left arm alone / right arm alone | | 0.0 / 1.0 |
| Settle (GPU run 1, 2026-11-21, first build) | the empty beam ran off zero onto its stop in 7 s (inverted pendulum) | 0.0 |
| Scripted demo (GPU run 1, first build) | see below | 0.25 |
| Scripted demo (GPU run 2, 2026-11-21) | rock, 100 g rider and poise right; the 10 g rider stayed in notch 2 (see below) | 0.25 |
| Settle (GPU run 3, 2026-11-21, this build) | nothing commanded; the empty beam stayed at zero, nothing drifted | 0.0 |
| Scripted demo (GPU run 3) | rock on the pan, riders in notches 2 and 3, poise at 6.4 g, pointer on zero; 1.0 held from step 1441 | 1.0 |
| Left arm idle (GPU run 3) | the demo is right arm only, so this is the same run; held from step 1441 | 1.0 |
| Right arm idle (GPU run 3) | nothing is touched | 0.0 |
| Dream-1 (GPU run 3, 2000 steps) | | 0.0 |

Offline counterfactuals: `no_poise` 0.5, `no_riders` 0.25, `wrong_notch` 0.25 (see the tests).

## Sources and estimates

Sourced: capacity 610 g, 0.1 g readability, beams 0 to 10 g (front), 0 to 500 g by 100 g (middle), 0 to 100 g by 10 g
(rear), fixed 6 in (15.2 cm) flat pan (Fisher Scientific, Carolina, Rice Lake and the maker's data sheet for the 750-S0).
Estimated (check on the unit before a real trial): the balance's overall size and beam heights; knife edge to pan axis
(0.120 m); rider and poise masses (set by the lever ratio: 100 g per 40 mm for the 300 g rider, 10 g per 15 mm for the 80 g
rider, 10 g over 100 mm for the 12 g poise); notch spacing; the net self-righting (0.0337 N m/rad with the rock on the
pan; the joint's spring is 0.269 N m/rad because the drawn pan, weights and rock sit above the knife edge), damper
(0.045 N m s/rad), poise friction (0.30 N) and the notch pull (about 2 N).

## GPU run 1 (first build): why it scored 0.25

- The beam was an inverted pendulum: the pan plate, the riders, the poise and the rock sit above the knife edge, and
  their weight (0.096 N m/rad without the rock, 0.235 with it) beat the 0.0337 N m/rad spring. The empty beam ran off zero
  at 1.16 per second, the growth those centres of mass predict (net -0.062 N m/rad). Now the spring is 0.0337 plus that
  term (`parts.gravity_stiffness`).
- The rock lay 7 mm toward the knife edge in the beam's frame (it was set with the beam already on its stop): 14 g light
  on a rigid pan. A rule now stands for the Roberval linkage (`roberval_target` in task.py).
- The 10 g rider's push turned the beam onto its +8 degree stop (push force at a height over the edge, plus the 14 g);
  the pad, aimed for -8, passed over the rider at 31 mm (notch 2, not 3) and later over the poise at 48 mm. Pad heights
  now hold on either stop, and the poise is as tall as the riders.
- The soft detent (200 N/m) left the 100 g rider 3 mm off its notch (77 mm, then 82 mm), 7.5 g at the pan. The detent
  is now a stiff spring with its pull capped at 2 N.

## GPU run 2: why it scored 0.25

- The beam held its stops and the 100 g rider reached its notch; the poise ended at 65.6 mm.
- Pushing the 10 g rider out of its zero notch turned the beam to +8 at once (only 36 g was left on the pan side). On
  +8 the rider followed the hand, but the push's end, planned for -8, stood for 36.6 mm on +8: the rider stalled at
  30.6 mm with the hand 5.7 mm behind its target pushing about 1.1 N, short of the 2 N that pulls it out of notch 2.
- Now that push is planned for +8 and runs on further (above), and the poise's creep ends 1.5 mm further (near balance
  the pad meets the poise lower, 2 to 4 mm less poise for the same hand).

## Files

`layout.py` (every number), `parts.py` (the balance and rock as mechkit assemblies; the tare weight's place is solved so
the empty beam balances), `scoring.py` (Tracker), `task.py` (scene, rule-based detents, offline model, demo),
`mechkit.py` and `rigstate.py` (copies of `tools/kit`), `tests/test_scoring.py`.
Tests: from the repository root, `PYTHONPATH=tools/offline/yam_stub:tools/offline python3 -m unittest discover -s tasks/triple-beam-balance-riders-set-pointer-on-zero/tests -q`.
