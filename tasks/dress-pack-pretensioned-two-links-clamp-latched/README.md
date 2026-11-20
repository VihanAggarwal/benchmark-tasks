# Pretension a cobot's dress-pack hose two links and latch its protector at the exit

A parked cobot forearm lies across the table with its dress pack on top: an igus triflex R TRE.40 cable hose (43 mm
across, 13.9 mm links) coming out of the exit end of an igus triflex RSE(C) TR.RSE.40 linear retraction unit
toward the wrist, its moving end not yet fixed. igus's RSE assembly instructions say to pretension the system by
pulling the triflex forward two links and installing a protector at the exit of the moving end. The protector, an
igus TR.40.30 quick-release protector (55 mm across, 27 mm long), sits open round the hose at the exit. The robot
must pull the hose out against the unit's elastic band, close and latch the protector on it at the exit, and let
go: the protector, wider than the exit, then rests on the exit face and keeps the hose two links (27.8 mm) out.

## Why it needs two hands

1. **A sprung degree of freedom.** The unit's band pulls the hose back in (about 4 N at two links; an estimate,
   igus give no force). Let go of the hose and it runs back into the unit.
2. **The other hand closes the protector.** The protector is about 230 mm from where the hose is held, and its lid
   is pushed over along an arc against its own spring. One hand holds the hose out; the other pushes the lid shut.
3. **Form fit and contact, not a rule, hold the hose.** A rib under the lid fits the hose's waists. Over a waist the
   lid shuts; over a crest the rib stops it about 8 deg open. Shut, the rib carries the protector with the hose,
   and the protector stops on the exit face: let go at 33.8 mm, the hose runs back 6 mm and stays 27.8 mm out.
   Only the quick-lock is a rule: within 2.5 deg of shut the lid is held shut.

## Scoring

| Condition | Rule (sim) | Bench method (real trial) |
|---|---|---|
| Pull | the hose's travel out of the unit; the mark is two links, 27.8 mm | caliper from the exit face to the band's wrist-side edge (8 mm at the start, 35.8 mm at the mark) |
| Shut | the lid within 2.5 deg of shut | the quick-lock's click, a 2.4 mm feeler under the lid's free edge |
| Near / close | pull within 7 mm (half a link) / 5 mm of the mark | the caliper |
| Hands off | both finger zones 40 mm or more from the hose and the protector | video |
| Still | the hose moving under 2 mm/s | no visible motion |

Tiers: 0.25 (near at some time), 0.5 (shut while near at some time), 0.75 (shut and close at the end), 1.0 (shut,
close, hands off and still for the last 1 s). Doing nothing scores 0, the lid cannot shut without a pull, and
setting the hose one or three links out scores 0.

## Real trial parts

igus TRE.40 triflex R e-chain (as TRE.RSE.40.058 for the RSE), igus TR.RSE.40 RSE(C) linear retraction unit, igus
TR.40.30 quick-release protector, on any cobot forearm or a 63 mm tube; the cradle (a 71 mm wide floor from the exit 61 mm toward the wrist, 0.5 mm under the protector, with a 6 x 12 mm guide 1.5 mm outside each side, printed or bent sheet); a spring scale to read the band's pull.

## Results

| Run | Offline prediction | GPU |
|---|---|---|
| Scripted demo (both arms) | 1.0 | 1.0 |
| One arm (left idle: nobody closes the lid) | 0.25 | 0.25 |
| One arm (right idle: nobody pulls; the lid stops on a crest) | 0.0 | 0.0 |
| Counterfactual `onelink` (shut a link short) | 0.0 | not run |
| Counterfactual `early` (let go before the lid is shut) | 0.25 | not run |
| Settle (`DEMO_HOLD_ONLY=1`) | 0.0, nothing moves | 0.0, nothing moved |
| Dream-1 | | 0.0 |

The offline numbers come from a kinematic stand-in (no friction or contact forces).

### GPU notes

Two scripted demo runs on 20 Nov scored 0.25. In both, the right hand drew the hose out and the left pushed the lid
over and pressed it down, but the lid stopped on its rib on a crest (9.0 and 8.6 deg open). The hose was 25.2 and
31.3 mm out while the lid was pressed. A waist is under the rib only from 31.8 to 35.8 mm, so the quick-lock never
caught. When the right hand let go, the hose ran back and the lid sprang open. Commanded 42.6 and 53.8 mm, the hose
came out 1.64 to 1.72 times less, because the arm gives under the band's pull and the hose creeps in the pinch. One
open-loop stop inside a 4 mm window is not reliable, so the demo no longer tries it. The right hand draws the hose to
40.75 mm, where a crest is centred under the rib (3.45 mm of margin either way). The left presses the lid down onto
that crest. Still pressing, the right hand eases the hose back over 4 s. As the waist passes under the rib, the lid
drops in and latches. The waist's wall then carries the protector onto the exit face. The end command (about 24 mm
on the GPU) is below the mark, so the protector on the exit face, not the hand, stops the hose at 27.8 mm.

## Running it

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
        -E control_hz=20 --scorer task_score --max-steps 1000

`DEMO_HOLD_ONLY=1` (settle), `DEMO_IDLE_ARM=left|right` (one arm), `DEMO_VARIANT=onelink|early`
(counterfactuals), `TASK_TRACE=<file>` (one JSON line per step). Tests: `python3 -m unittest discover -s tests`.

## Files

`task.py` (scene, hooks, quick-lock rule, demo), `layout.py` (every number), `parts.py` (shapes), `scoring.py`
(tracker), `offline_dresspack.py` (offline stand-in), `mechkit.py` and `rigstate.py` (shared kit, unchanged).
