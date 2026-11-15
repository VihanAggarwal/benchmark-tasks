"""Kinematics of the window for use ONLY where there is no simulator: tools/offline/check_task.py (which by itself can
carry and drop a gripped box, nothing else) and the tests. The simulator never takes a result from here: there the
sash hinge, the swivel and the lift pin are PhysX joints of one articulation, the stay's holes are rings of boxes
and the pins tapers of boxes, and nothing is moved by code.

What this file assumes is what the design claims, so the offline prediction is the design's prediction, not a
measurement. It moves the window from the arms' COMMANDED pad-tip points and openings (rigstate's offline feed):

- A gripper holds the ball once its pads close to 16 mm or less with the ball's centre within 6 mm of the point 4 mm
  above the pad tips; it lets go when they open past 18 mm. While held, the ball follows that point: the three joints
  are solved for it (the linkage is rigid: two turns in plan and the lift).
- Pins: where a pin is under the bar, the bar cannot go lower than resting on it: on the pin's top with the groove
  over it (the web's underside at the pin's top), on a leg (3.75 mm higher) when the pin is off the groove, and in a
  hole when the hole's centre is within 1.1 mm of the pin (the tip's radius 2.5 mm in a 3.6 mm hole): then the pin
  is in the hole, the hole centred on it and the sash and swivel locked by it, the bar no lower than the seat.
- A bar not held falls to the highest support (a hole's seat, a pin's top, a leg on a pin, or the board).
- A held bar whose hole is on the pin cannot be dragged along: the joints in plan stay; the grip slips (lets go) when
  the hand is 12 mm or more from the ball in plan.
"""

from __future__ import annotations

import math

import numpy as np

import layout as L
import rigstate

SUBSTEPS = 6
HOLD_GAP = 2 * L.BALL_R + 0.002
LET_GO_GAP = 2 * L.BALL_R + 0.004
GRAB_RADIUS = 0.006
GRIP_UP = 0.004  # the ball's centre above the pad tips when held
CAPTURE = L.HOLE_R - L.TIP_R  # 1.1 mm
SLIP = 0.012
LEG_UP = L.BAR_D - L.WEB_T


def floor_for(s: float, n: float) -> tuple[str, float | None, int | None]:
    """What a pin at (s along the bar, n across) holds the bar's web underside above: ('hole', seat, k),
    ('top', pin top, None), ('leg', pin top + 3.75 mm, None) or ('none', None, None)."""
    for k, hs in enumerate(L.HOLE_S):
        if math.hypot(s - hs, n) <= CAPTURE:
            return "hole", L.PIN_SEAT_H, k
    if L.GROOVE_S0 <= s <= L.GROOVE_S1 and abs(n) <= L.GROOVE_W / 2 - L.TIP_R:
        return "top", L.PIN_TOP_H, None
    if L.BAR_S0 - L.TIP_R <= s <= L.BAR_S1 + L.TIP_R and abs(n) <= L.BAR_W / 2 + L.TIP_R:
        return "leg", L.PIN_TOP_H + LEG_UP, None
    if L.BAR_S1 < s <= L.TAIL_END + L.TIP_R and abs(n) <= L.BALL_R + L.TIP_R:
        return "top", L.PIN_TOP_H, None  # the neck and ball ride higher than the web line: counted as the top
    return "none", None, None


def _clamp(q):
    return (float(np.clip(q[0], L.SASH_LOWER, L.SASH_UPPER)), float(np.clip(q[1], L.SWIVEL_LOWER, L.SWIVEL_UPPER)),
            float(np.clip(q[2], L.LIFT_LOWER, L.LIFT_UPPER)))


def lift_where(q, pin: str, h: float) -> float:
    """The lift angle at which the web's underside over pin ``pin`` is at height ``h`` (sash, swivel as in q)."""
    lo, hi = L.LIFT_LOWER, L.LIFT_UPPER
    for _ in range(50):
        mid = (lo + hi) / 2
        if L.pin_in_bar((q[0], q[1], mid), pin)[2] < h:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def board_lift(q) -> float:
    """The lift angle at which the bar's lowest point (the legs' end) meets the board."""
    s_end = L.GROOVE_S1 - L.S_LIFT
    h0 = L.LEG_BOT_H - L.LIFT_H
    lo, hi = L.LIFT_LOWER, 0.0
    for _ in range(50):
        mid = (lo + hi) / 2
        z = L.LIFT_H + math.sin(mid) * s_end + math.cos(mid) * h0
        lo, hi = (lo, mid) if z > 0 else (mid, hi)
    return hi


def ik(q, target, iters: int = 30):
    """Joints that put the ball's centre at ``target`` (world), from q (damped least squares)."""
    q = np.array(q, dtype=float)
    for _ in range(iters):
        e = np.asarray(target, dtype=float) - L.disc_world(q)
        if float(np.linalg.norm(e)) < 1e-6:
            break
        jac = np.zeros((3, 3))
        for i in range(3):
            d = np.zeros(3)
            d[i] = 1e-6
            jac[:, i] = (L.disc_world(q + d) - L.disc_world(q)) / 1e-6
        dq = np.linalg.solve(jac.T @ jac + 1e-8 * np.eye(3), jac.T @ e)
        q = np.array(_clamp(q + dq))
    return tuple(float(v) for v in q)


class OfflineStay:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.q = (0.0, 0.0, 0.0)  # as drawn: on hole 2
        self.holder = None  # index of the arm holding the ball
        self.captured = 1  # the hole (0-based) that pin A is in: hole 2 at the start
        self.last = None
        self.events: list = []

    def _supports(self, q):
        """The highest support of the bar over each pin at the plan pose of q: (lift angle, kind, pin, hole)."""
        best = (board_lift(q), "board", None, None)
        for pin in L.PINS:
            s, n, _ = L.pin_in_bar(q, pin)
            kind, h, k = floor_for(s, n)
            if kind == "none":
                continue
            lift = lift_where(q, pin, h)
            if lift > best[0]:
                best = (lift, kind, pin, k)
        return best

    def _centre_on(self, k: int, lift: float):
        rho = L.S_LIFT + (L.HOLE_S[k] - L.S_LIFT) * math.cos(lift) - (L.WEB_BOT_H - L.LIFT_H) * math.sin(lift)
        js, jw, _ = L.q_for(rho, lift=lift)
        return (js, jw, lift)

    def _held(self, target):
        q = self.q
        if self.captured is not None:
            k = self.captured
            lo, hi = L.LIFT_LOWER, L.LIFT_UPPER  # the lift that puts the ball at the hand's height
            for _ in range(50):
                mid = (lo + hi) / 2
                lo, hi = (mid, hi) if L.disc_world(self._centre_on(k, mid))[2] < target[2] else (lo, mid)
            lift = (lo + hi) / 2
            seat = lift_where(self._centre_on(k, lift), "A", L.PIN_SEAT_H)
            lift = max(lift, seat)
            qn = self._centre_on(k, lift)
            if L.pin_in_bar(qn, "A")[2] > L.PIN_TOP_H:  # lifted clear of the pin: free again
                self.captured = None
                self.events.append(("lifted_off", k + 1))
            else:
                d = L.disc_world(qn)
                if math.hypot(d[0] - target[0], d[1] - target[1]) >= SLIP:
                    self.holder = None
                    self.events.append(("slip", k + 1))
                self.q = qn
                return
        qn = ik(q, target)
        s, n, h = L.pin_in_bar(qn, "A")
        kind, _, k = floor_for(s, n)
        if kind == "hole" and h < L.PIN_TOP_H:  # the hand takes the web below the pin's top over a hole: in it goes
            self.captured = k
            self.events.append(("in_hole", k + 1))
            lift = max(qn[2], lift_where(self._centre_on(k, qn[2]), "A", L.PIN_SEAT_H))
            self.q = self._centre_on(k, lift)
            return
        lift, _, _, _ = self._supports(qn)
        self.q = (qn[0], qn[1], max(qn[2], lift))

    def _fall(self):
        q = self.q
        if self.captured is not None:
            self.q = self._centre_on(self.captured, lift_where(self._centre_on(self.captured, q[2]), "A",
                                                               L.PIN_SEAT_H))
            return
        lift, kind, pin, k = self._supports(q)
        if kind == "hole" and pin == "A":
            self.captured = k
            self.events.append(("dropped_in", k + 1))
            self.q = self._centre_on(k, lift)
            return
        self.q = (q[0], q[1], max(lift, L.LIFT_LOWER))

    def step(self, arms) -> None:
        read = rigstate.read_arms(arms)
        if read is None:
            self.last = None
            return
        now = [(np.asarray(t, dtype=float), np.asarray(p, dtype=float), float(o)) for t, p, o in read]
        before = self.last if self.last is not None else now
        self.last = now
        for j in range(1, SUBSTEPS + 1):
            f = j / SUBSTEPS
            hands = []
            for (t0, p0, o0), (t1, p1, o1) in zip(before, now):
                tcp, tips, op = t0 + f * (t1 - t0), p0 + f * (p1 - p0), o0 + f * (o1 - o0)
                u = tcp - tips
                nu = float(np.linalg.norm(u))
                u = u / nu if nu > 1e-9 else np.array([0.0, 0.0, 1.0])
                hands.append((tips + GRIP_UP * u, float(np.clip(op, 0.0, 1.0)) * rigstate.FULL_GAP))
            disc = L.disc_world(self.q)
            if self.holder is not None and hands[self.holder][1] > LET_GO_GAP:
                self.holder = None
                self.events.append(("let_go", None))
            if self.holder is None:
                for i, (g, gap) in enumerate(hands):
                    if gap <= HOLD_GAP and float(np.linalg.norm(g - disc)) <= GRAB_RADIUS:
                        self.holder = i
                        self.events.append(("grip", i))
                        break
            if self.holder is not None:
                self._held(hands[self.holder][0])
            else:
                self._fall()

    def joints(self) -> dict:
        return {"sash": self.q[0], "swivel": self.q[1], "stay": self.q[2]}
