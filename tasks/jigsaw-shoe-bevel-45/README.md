# Set a cordless jigsaw's shoe to a 45-degree bevel and lock it

A DeWalt DCS331 20V MAX jigsaw (bare tool: battery out, no blade, 5.4 lb) lies on its left side on a rubber bench
mat, front away from the robot, its shoe square and clamped. The robot sets the shoe to a 45 degree bevel, clamps
it and lets go. The instruction says only that.

GPU results (2026-11-14, L40S): the scripted demo scores 1.0 (see Results in simulation). The first GPU run had scored
it 0.0: a closed-finger push on the lever's tab wedged, so the shoe stayed clamped. The demo now pinches the tab and
swings the lever along its arc instead.

Revised 2026-11-15 after review (see "Revision (2026-11-15)" at the end): moving the saw on the bench is no longer
penalised; only dropping it, pushing it off the bench or letting it fall over latches a zero. The saw's figures were
re-checked against the maker's manual, datasheet and service parts list.

## Why it is hard

1. **The lever unlocks the shoe, but the shoe still does not tilt.** The maker's manual: "pull the keyless bevel
   lever to the side" to unlock, then "slide the shoe forward to release it from the 0 degree positive stop". With
   the lever out and the shoe not slid, a push on its edge meets the stop and the shoe stays square.
2. **Detents on the way.** The shoe tilts through detents at 15 and 30 degrees before 45 (the end of travel). A push
   that stops at the first click leaves a 15 degree bevel.
3. **It is not set until it is clamped.** The lever must be pushed back under the body; an unclamped shoe moves
   under a light push. The score is read with both hands at least 5 cm away for 2 s.
4. **The hands.** Everything is done with straight-down grippers: a pinch on the lever's 6 mm finger tab, which lies
   in a 22 mm gap beside the motor housing, carried along the lever's arc; a grip on the shoe's 10 mm edge where it
   stands clear in front of the gear housing; a sideways push on that edge; the same pinch to swing the lever back.
   Every working point keeps 15 mm or more from body parts it does not work on, except the fingers astride the
   flush tab, 7 mm from the housing (all the gap allows).

## Scoring

Conditions (real trial in brackets):

- Bevel: the shoe's angle to square, either way (protractor between shoe and body side).
- Clamped: the lever within 10 degrees of flush under the body (then a 5 N push on the shoe's top edge, spring
  scale, turns it under 1 degree; unclamped it moves 10 degrees or more).
- Hands off: on every step of the last 2 s every part of both grippers' fingers 5 cm or more from the saw (video).
- Still: over the last 2 s the bevel changes 0.5 degree or less, the lever 2 degrees or less, the body 2 mm or less.
- Unlocked off the stop: at some point the lever out (10 degrees or more) while the shoe is 6 mm or more forward of
  its 0 degree stop (a pencil mark on the shoe against the gear housing, video).
- Latched zero, only for clearly unsafe outcomes: the saw dropped or off the bench (any part 5 cm or more below the
  bench top, its body 5 cm or more below where it lay, or its blade line, the shoe's pivot, past the bench's edge),
  or fallen over (the side that faced up turned 60 degrees or more from vertical: onto its handle, its shoe or an
  end; protractor on the side video). Sliding or turning the saw on the bench is not penalised: a saw moved 40 mm and
  turned 20 degrees with its shoe locked at 45 scores 1.0.

| Outcome | Score |
|---|---|
| Hands off, still, bevel 45 +-2 degrees, clamped | 1.0 |
| Hands off, still, bevel 45 +-2 degrees, not clamped | 0.75 |
| Hands off, still, bevel 28 degrees or more (the 30 detent or past it), not 45 | 0.5 |
| Unlocked off the stop at some point, none of the above (held, at 15, touched) | 0.25 |
| Anything else, or after a latched zero | 0 |

Success: 1.0 held for 1 s. Doing nothing scores 0.

## Offline predictions (no physics)

| Run | What it does | Predicted |
|---|---|---|
| `DEMO_HOLD_ONLY=1` | both arms hold still | 0.0 |
| Scripted demo | both arms, below | 1.0 (held from step 1,055 of 1,200) |
| `DEMO_IDLE_ARM=left` | the right hand alone | 1.0 offline (see below) |
| `DEMO_IDLE_ARM=right` | the left hand only rests on the handle | 0.0 |
| `DEMO_VARIANT=no_unlock` | lever never swung out: pull and push meet a clamped shoe | 0.0 |
| `DEMO_VARIANT=no_slide` | lever out, no slide: the push meets the 0 degree stop; lever back | 0.0 |
| `DEMO_VARIANT=stop_at_15` | the push stops in the 15 degree detent; lever back | 0.25 |
| `DEMO_VARIANT=lever_left_out` | shoe at 45, lever left out | 0.75 |

One arm: the right hand makes every move; the left only steadies the saw on its handle while the right pulls and
pushes along and across it. On the GPU the right hand alone also scored 1.0 (the 2.45 kg saw on rubber did not
skid). The task does not claim that two arms are needed.

## Demo

Open loop, timed waypoints from the known layout; both grippers straight down, fingers closing along y (wrist yaw
0). A closed gripper (1 mm commanded) acts as one block 20 mm along x and 17 mm along y. Five deliberate moves:

- Right, 0 to 11.25 s, move 1: open 20 mm, down astride the lever's finger tab (pad tips level with the housing's
  top, 7 mm from the tab and from the housing either side), pinches the tab (2 mm commanded) and carries it up
  along its arc to 50 degrees in five 10 degree legs (the fingers close along the lever's hinge axis, so the tab
  turns between the pads); lets go: past 25 degrees the lever goes on out to 60; up.
- Left, 11.25 to 15.25 s, move 2: from home, rests its closed fingers on the top handle's upper side, 1.5 mm in,
  and stays there until 33.25 s (while the right pulls and pushes along and across the saw), then goes home.
- Right, 11.25 to 22.75 s, move 3: open 60 mm, down astride the shoe's edge 40 mm ahead of the gear housing, closes
  on the 10 mm edge (2 mm commanded), pulls 12 mm forward (the shoe stops at the slot's end, 10 mm; 6 mm takes it
  off the stop), lets go, up.
- Right, 22.75 to 33.25 s, move 4: closed, down on the housing side of the shoe's upper rib, pushes 34.5 mm
  outward: the shoe tilts past 15 and 30 to 45 (the end of travel), the block 3.3 mm past the 45 degree contact;
  backs off, up. It then waits over the shoe until the left is home (36.75 s).
- Right, 36.75 to 48.5 s, move 5: back along the saw above the open lever, open 20 mm down astride its tab at 60
  degrees, pinches the same spot and carries it back down its arc to 5 degrees: under 10 the shoe clamps at 45;
  lets go (under 25 the lever falls flush); up, home at 51 s.

Success is held from about 52.75 s (2 s hands off and still, then 1 s). Smallest clearance offline: the open
fingers astride the flush tab 7 mm from the housing (15 mm or more once the lever is 15 degrees out), the pad
block to the body 16.3 mm while astride the shoe's edge; arm capsules 2.7 cm; grippers 20.8 cm apart.

Why not a push: the first GPU run pushed the tab's rear face forward with closed fingers. That face is 7 to 22.5 mm
above the pivot and 57 mm behind it, so the push turns the lever only while the pad friction is under 0.4; above
that it wedges. On the GPU the arm stalled 21 mm short and the lever turned 1 degree.

```bash
inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
  -E control_hz=20 --instruction "<the instruction>" --scorer task_score --max-steps 1200
```

`TASK_TRACE=trace.jsonl` writes one line per control step: the three joints, the rule's mode and targets, the
tier's inputs, the body's offset from its marks and tilt, the shoe's top front corner, the lever tab's top and both
TCPs in the saw's frame, the gripper openings, and where the pinched spot of the tab and the gripped spot of the
shoe's edge are from the right pad tips (about (0, 0, 5) and (0, 0, 4) while held; a growing offset is a slip).

## Simplifications

- **One software rule** (`shoerule.py`) stands for the stop, the detents and the clamp: it writes drive targets on
  the three joints every control step. Clamped (lever under 10 degrees): the slide and bevel are held where they
  were (10,000 N/m and 20 N m/rad: 5 N on the edge turns the shoe 0.6 degree). Unclamped: the slide follows the
  hand (its target led by 0.9 of its last step's travel, as in our GPU-tested pliers task); the bevel is held at 0
  until the shoe is 6 mm forward, then follows the same way with a detent pull of at most 0.03 N m within 3
  degrees of 0, 15, 30 and 45 either way. The lever is an over-centre cam: under 25 degrees it falls flush, above
  it rests at 60, each pull at most 0.05 N m.
- Rigid boxes. The shoe, its carriage and the lever are links of one articulation and do not collide with the body;
  their shapes are laid out so they do not overlap it in the demo's range.
- The real DCS331's lever is a bent rod ending in a round knob (manual, Fig. A and Fig. G); here it is a flat bar
  with a flat finger tab. Its place and swing match (between shoe and motor housing, handle end at the rear,
  swinging to the side in the shoe's plane); its exact size, height, travel and knob position are estimates (see
  Revision). Which side it swings to is not settled by the manual's figures (the rod shows from both sides). The
  layout assumes the saw's right (up, as it lies on its left side). If it swings to the left on the real saw, the
  real trial lays the saw on its right side and mirrors the layout.

## Sources

- DeWalt DCS331 instruction manual, bevel cutting: "pull the keyless bevel lever (f) to the side"; "Slide the shoe
  (g) forward to release it from the 0 degree positive stop position"; detents at 15, 30 and 45 degrees either way;
  "push the keyless bevel lever back under the body of the jigsaw":
  https://www.manualslib.com/manual/1084350/Dewalt-Dcs331.html?page=16
- DeWalt DCS331B product page (all-metal keyless shoe bevel with detents at 0, 15, 30 and 45 degrees):
  https://www.dewalt.com/product/dcs331b/en-us-20v-max-cordless-jig-saw-tool-only
- DeWalt DCS331 manual (types 3 to 10, from the maker's service site): technical data "Weight (without battery pack)
  2.3 kg", bevel 0 to 45 degrees left and right; "aluminum shoe"; Fig. A (the saw from its left, the bevel lever 6
  with its knob) and Fig. G (unlocking, sliding):
  https://service.dewalt.co.uk/i/DEWALT/GLOBALBOM/XJ/DCS331/3/Instruction_Manual/EN/DCS331_T3-T10_GB_XE.pdf
- DeWalt DCS331 type 3 service parts list: item 85 shoe 581263-00, item 86 lever 493248-00 and lever lock
  495746-00, item 89 clamp 496996-00, items 90 to 92 bolts and washer, item 88 plunger; no dimensions or forces:
  https://service.dewalt.co.uk/BD/IE/en-IE/Part/PartsForProduct?productId=36805 (product page with the weight,
  2.3 kg: https://service.dewalt.co.uk/BD/IE/en-IE/Product/ProductDetail?id=36805)
- DeWalt DCS331N-XJ datasheet (the same saw, EU bare unit): length 305 mm, height 215 mm, weight 2.3 kg, stroke
  26 mm, bevel 0 to 45 degrees:
  https://assets.cef.co.uk/downloads/pdg/dewalt_dcs331n-xj_datasheet/dewalt_dcs331n-xj_datasheet.pdf
- Tool weight 5.4 lb, 3.88 in across (US retailer listing): https://usatoolsinc.com/20v-max-jig-saw-tool-only-dcs331b
- Length 11.25 in, 3.9 in (US retailer listings, others give 10.5 in), for example:
  https://www.truevalue.com/product/max-cordless-jig-saw-20-volt-tool-only/
- Carpenters, SOC 47-2031, 889,700 jobs in 2025: https://www.bls.gov/ooh/construction-and-extraction/carpenters.htm

Sourced and used: 5.4 lb (2.45 kg; the maker's EU figure is 2.3 kg, 6 percent less, and the US figure is kept since
the task names the US 20V MAX tool); 215 mm from shoe to handle top (the model's handle top is at 215 mm); 99 mm
across; an aluminium shoe; the detents, stop, unlock and lock wording. Length: 286 mm (US) against 305 mm (EU sheet);
the model keeps 286 mm, and the extra length would only lengthen the rear block that nothing touches.

Estimates (no published figure; not measured on a saw): the shoe's width (90 mm), plate and ribs and its mass (250 g:
the modelled plate and ribs in aluminium weigh 220 g, the real casting has bosses); the lever's size, height, travel
and the tab standing for its knob; the 6 mm slide off the stop and the 10 mm slot; the handle's shape; every force of
the rule. Scaled from the manual's Fig. A by the 215 mm height (a perspective line drawing, so only to about 10 mm):
the shoe about 170 mm long (as modelled), the knob about 15 mm long and 8 mm across at the shoe's rear end. The
real trial measures them on the saw first.

## Real build and reset

The saw, battery out and no blade; a rubber mat taped down with marks for the saw's outline; a protractor, a ruler,
a 10 N spring scale, top and side video. Reset: pull the lever out, tilt the shoe back to 0, slide it back into the
0 degree stop, push the lever back under the body and check the shoe will not tilt; lay the saw on its marks.

## Files

| File | What it is |
|---|---|
| `task.json`, `task.py` | The task text; the saw, the rule's hook, `success()`, `score()`, `demo()` and its variants |
| `layout.py`, `shoerule.py`, `scoring.py` | Every dimension; the stop, detent and clamp rule; the scorer |
| `offline_saw.py` | Offline only: moves the joints from the commanded arm motion (the checker cannot) |
| `mechkit.py`, `rigstate.py` | Unchanged copies of the kit: the saw as an articulation; both grippers each step |
| `tests/` | Offline tests: `python3 -m unittest discover -s tests` |

Third-party assets: none.

## Results in simulation

Measured on an L40S (Isaac Lab-Arena, object jitter 0, 20 Hz control), one trial each:

| Run | `task_score` |
|---|---|
| Scripted demo (`demo()` as written) | 1.0 |
| Scripted demo, left arm idle (`DEMO_IDLE_ARM=left`) | 1.0 |
| Scripted demo, right arm idle (`DEMO_IDLE_ARM=right`) | 0.0 |
| Dream-1 | 0.0 |

## Revision (2026-11-15)

The review came back "Not accepted (revisable)". Each point and what changed:

**Scoring: "A 30 mm translation or 15-degree reorientation permanently scores zero even when the saw is safely set
to the requested angle."** Agreed; removed. `scoring.py` no longer latches a zero for the saw body moving 30 mm or
turning 15 degrees, or for any other move on the bench: a saw slid or turned while it is worked, then left with its
shoe locked at 45, scores 1.0 like one that never moved. The 2 s "still" window (body 2 mm or less) still applies,
so the score is read on a saw at rest. Tests: `test_safe_repositioning_is_not_penalised` (slid 40 mm, turned 20
degrees, tilted 20 degrees, and combinations: 1.0) and `test_moved_during_the_trial_then_locked_scores_one` (pushed
40 mm and turned 20 degrees while unlocked, then locked at 45: 1.0, nothing latched).

**"Retain a penalty for dropping the saw or other clearly unsafe outcomes."** Kept, and only those latch a zero for
the rest of the trial: the saw dropped (any part 5 cm or more below the bench top, or its body 5 cm or more below
where it lay), pushed off the bench (its blade line past the bench top's edge), or fallen over onto its handle, its
shoe (blade side) or an end (the side that faced up turned 60 degrees or more from vertical; lying on its handle or
shoe it would be near 90, so 60 is well past any tilt a hand sets down on purpose and short of the fall). Tests:
`test_pushed_off_the_bench_is_a_latched_zero` (0 latched, even if put back locked at 45),
`test_dropped_is_a_latched_zero`, `test_fallen_over_is_a_latched_zero` (60 and 90 degrees either way; 59 is not).
The rest of the ladder (1.0, 0.75, 0.5, 0.25, 0) is unchanged. The `scoring` field and the Scoring section above say
the same.

**Flag: "Verify the estimated lever geometry, access and forces against the specified real jigsaw."** Re-checked
against the maker's DCS331 manual (types 3 to 10, Fig. A and Fig. G), its service parts list and the maker's EU
datasheet (Sources). What they settle:

- Agrees with the model: the bevel lever lies in the gap between the shoe and the motor housing, its handle end is
  at the rear and it swings to the side in the plane of the shoe to unlock and back under the body to lock (Fig. G);
  the saw is 215 mm from shoe to handle top (datasheet), as modelled; scaled by that height, Fig. A shows a shoe
  about 170 mm long, as modelled. The parts list has a separate lever (493248-00) with a lever lock and a clamp,
  consistent with a cam lever clamping the shoe.
- Corrected: the shoe is aluminium (manual: "aluminum shoe"), not steel. Its mass stays 250 g (the modelled plate and
  ribs weigh 220 g in aluminium; the real casting has bosses). The text and the shoe's colour name say aluminium;
  no size or mass changed.
- Differences that stay estimates, said so in `fidelity`: the real lever is a bent rod ending in a round knob, about
  15 mm long and 8 mm across, at the shoe's rear end (Fig. A, a perspective line drawing, so only to about 10 mm);
  the model has a flat bar with a 10 x 6 mm tab about 35 mm further back. Fig. G does not show the travel in a
  measurable view; 60 degrees stays an estimate. Which side the knob swings to is not settled by the figures (the
  rod shows from both sides of the saw); the layout keeps the saw's right and the real trial mirrors it if needed.
  None of these were changed to suit the robot; they are left as built until the real saw is measured.
- Forces: the manual, the datasheet and the parts list give none. The rule's figures (detent 0.03 N m, lever cam
  0.05 N m, clamp 5 N for 0.6 degree) stay estimates; the real trial measures, with the 10 N spring scale, the pull
  on the knob to unlock, the push to slide the shoe off its stop, the push through each detent and the 5 N clamp
  check before the first trial.
- Length and weight: the maker's EU datasheet gives 305 mm and 2.3 kg, the US listings 11.25 in (286 mm) and 5.4 lb
  (2.45 kg). The model keeps the US figures (the task names the US 20V MAX tool) and the README lists both.

The demo is unchanged (no test needed a change), so the GPU scores above (demo 1.0, left arm idle 1.0, right arm
idle 0.0, Dream-1 0.0) stand for the same moves; a GPU re-run of the revised scorer is queued to confirm them.
