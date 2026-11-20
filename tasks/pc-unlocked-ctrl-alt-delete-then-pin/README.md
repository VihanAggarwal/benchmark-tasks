# Sign in to a workstation's test account with Ctrl+Alt+Delete and its PIN

An office PC sits locked at the Windows 11 lock screen. Its policy "Interactive logon: Do not require CTRL+ALT+DEL"
is Disabled, so the lock screen asks for Ctrl+Alt+Delete before the PIN box appears (Microsoft: this ensures users
are communicating by means of a trusted path when entering their passwords). The PIN, 2580, is on a sticky note on
the monitor's bezel. The keyboard is a Logitech K120 (maker: 450 x 155 x 23.5 mm, 550 g, full-size US layout) lying
loose on its rubber feet in front of the monitor.

Windows takes the sequence only if Delete goes down while a Ctrl and an Alt are already held and still held. On a full-size board no
two of the three are adjacent: Left Ctrl and Left Alt are 47.6 mm apart with the Windows key between them, and Delete is 29 cm away in the navigation cluster. So one hand holds the pair, straddling
the Windows key, while the other strikes Delete; then the PIN is typed key by key on the number pad and Enter pressed.

## Why it is hard

1. **A held chord across a key that must stay up.** The left hand's two fingertips must sit on Ctrl and Alt and press
   both past 2.0 mm (about 0.70 N each) and keep them there, with the Windows cap between its fingers untouched.
2. **Order and timing between the hands.** Delete must go down after both are held and while they still are; Delete
   first, all three at the same instant, or letting either go as Delete goes down, does nothing. Only both hands working together reach the PIN prompt.
3. **Precise single-key strikes.** Five 18 mm caps on a 19.05 mm pitch, each pressed past 2.0 mm and released, without
   brushing a neighbour (a stray key, or a wrong digit, caps the score at 0.5).
4. **A loose keyboard.** The 550 g keyboard slides and tips under a careless press; sliding is harmless, but lifted 5 mm or tilted
   5 degrees scores 0.

## Scoring

| Condition | Rule (sim) | Real trial |
|---|---|---|
| Key down | a modelled cap 2.0 mm or more in | the PC itself, plus an inline USB keystroke logger (KeyGrabber USB) timestamping every key-down and key-up |
| Upset (latched 0) | the keyboard lifted over 5 mm or tilted over 5 degrees (pulled off the desk) | a rule at its corners on the video; tilt by eye |
| PIN prompt | Delete down while a Ctrl and an Alt (either side) were both already down and are still down | the monitor shows the PIN box |
| PIN | number pad digits after the prompt, 2, 5, 8, 0 and nothing else | dots in the box, the logger |
| Desktop | Enter after the right PIN | the monitor shows the desktop |
| Hands off | every key up, both grippers 30 mm or more from the keyboard and it moves under 1 mm, over the last 1 s | the video and a 30 mm spacer block |

| Outcome | Score |
|---|---|
| Upset | 0 |
| Desktop shown, hands off | 1.0 |
| Prompt and the right PIN typed (or desktop shown but not hands off) | 0.75 |
| PIN prompt shown; or anything higher with a stray key or a wrong-PIN Enter | 0.5 |
| A Ctrl and an Alt held together for 0.5 s | 0.25 |
| Otherwise | 0 |

Possible scores: 0, 0.25, 0.5, 0.75, 1.0. Success: 1.0 held for 1 s. A stray key is any other modelled key (a Windows key included) that goes
down outside the act: before the prompt any key but Ctrl, Alt and Delete; after it any key but the digits and Enter.

| Bypass | What stops it |
|---|---|
| Press Ctrl, Alt and Delete at once with one sweep | the same instant does not count; brushing the Windows key between Ctrl and Alt caps 0.5 |
| Use Right Ctrl and Right Alt (also split by Windows and Menu) | allowed, as on a real PC: still needs a second hand on Delete |
| Mash the number pad | any wrong digit or extra key caps 0.5 |
| Leave a finger resting on a key | not hands off: 0.75 |
| Drag the keyboard to bring Delete near | Delete stays 29 cm along the board from the held pair: still two hands; 1.0 needs it at rest |

## Model

`task.py` builds the keyboard as one loose mechkit articulation (case on four rubber feet, 550 g) with every key of
the bottom row, the navigation cluster and the number pad as its own cap on a vertical slider (28 keys): a linear
spring (200 N/m) toward a drive target 1.5 mm above the rest stop gives 0.30 N at rest and 0.70 N at the 2.0 mm
actuation point, 2.8 mm travel to the stop (a distributor datasheet for the K120 gives 2.8 mm; actuation point and force
are estimates). The other keys are fixed caps, so a finger can rest on them but they
never register. Caps are pressed only by contact. White seven-segment legends are drawn on the keys in play (Ctrl, Alt,
the Windows logo, Del, End, the digits, Ent), on the lock screen (Ctrl+Alt+Del) and on the sticky note (2580). The monitor is fixed; the PC's response (`scoring.PC`) is a rule
run every control step (`_rules`, a mechkit on_step callback) that slides coloured plates out of the panel: the lock
screen, the PIN prompt, one dot per digit, the desktop. The scorer reads the key joints, the keyboard's pose and the
grippers. Key positions follow the standard 19.05 mm pitch.

Positions (world, mm): Left Ctrl (274, 212), Windows (274, 189), Left Alt (274, 165), Delete (331, -76), number pad
2 (293, -157), 5 (312, -157), 8 (331, -157), 0 (274, -147), Enter (284, -195).

## Demo

Open loop, both arms. The left arm, yawed to close along y, opens its fingers to 41.6 mm between the pads (fingertip
centres on the Ctrl and Alt centres, the Windows cap clear of both inner faces by over 8 mm), lowers until the pad
tips are 3 mm into both caps and holds. The right arm, closed and yawed to close along x, strikes Delete while the
pair is held, then the left arm lets go and moves aside while the right types 2, 5, 8, 0 on the number pad and
Enter, each a press to 3 mm, a 0.3 s hold and a lift. Then aside. 466 steps of plan; the closest the two TCPs come is
27 cm and the arm capsules stay 10 cm apart.

## Results

| Run | Offline (no physics) | GPU |
|---|---|---|
| Demo | 1.0 (success held at step 491) | pending |
| One arm idle: right idle (left holds the pair only) | 0.25 | pending |
| One arm idle: left idle (right alone) | 0.0 | pending |
| Settle (DEMO_HOLD_ONLY=1) | 0 | pending |
| Variant wrong_pin (2, 5, 8, 9) | 0.5 | pending |
| Variant no_enter | 0.75 | pending |
| Variant delete_first | 0.25 | pending |
| Dream-1 | | 0.0 |

Offline numbers come from the task's own stand-in (`OfflineModel`: fingertip footprints push caps down by their
overlap depth). On the GPU the first run commanded the pads only 3 mm under the cap tops and no key moved, so `PRESS` is now
8 mm, as on other GPU-proven button tasks: the 2.8 mm bottom stop (the slider's joint limit) takes the key's share and
the arm's give takes the rest. `TASK_TRACE` logs every key's travel each step.

## Run

    inspect-robots run --embodiment yam_arena --policy yam_scripted -E task=task.py -E object_jitter=0 \
        -E control_hz=20 --scorer task_score --max-steps 1200

`DEMO_HOLD_ONLY=1` (settle), `DEMO_IDLE_ARM=left|right` (one arm), `DEMO_VARIANT=wrong_pin|no_enter|delete_first`
(counterfactuals), `TASK_TRACE=<file>` (one JSON line per step). Seed: none used (open loop, `object_jitter=0`).

Tests: `python3 -m unittest discover -s tests` (layout, PC rule and tiers, latches, offline model, demo).
