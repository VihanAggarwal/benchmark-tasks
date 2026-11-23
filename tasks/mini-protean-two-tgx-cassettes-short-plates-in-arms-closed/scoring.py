"""The scorer, as plain functions of the two cassettes', the electrode assembly's and the arms' state. NumPy only.

``Tracker.update(states, arms)`` -> Result(score, success, details), fed one observation per control step. It reads
the mechkit AssemblyStates of the two cassettes (base frame at the middle of the bottom edge; short plate on the
local -y side) and of the electrode assembly (fixed; joints ``arm_pos`` on the +y side, ``arm_neg`` on the -y side,
0 = open, Q_CLOSED = shut over a seated cassette). The arms are read only for the let-go check.

Per side (+y, -y), whichever cassette is there:
    AGAINST   upright (tilt <= 3 deg), its plane within 3 deg of the gasket's, its middle within 6 mm of the frame's
              middle along x, and its inner face within 1.5 mm of the gasket face at four points (both ends, low and
              high): it lies flat against both legs of the gasket
    INWARD    its short plate faces the frame's centre
    SEATED    AGAINST and INWARD and the short plate's top 0 to 4 mm under the gasket's notch
    SHUT      the side's green arm within 1.5 deg of shut (latched: an unlatched arm over a cassette is pushed back
              open by its cam)
    credit    0.15 AGAINST and INWARD; + 0.15 SEATED; + 0.20 SHUT over an AGAINST and INWARD cassette. An arm shut
              over a reversed cassette (short plate out) earns nothing for it or for the arm.
    0         at any step a cassette's lowest point more than 20 mm under the table top (off the bench: latched)
    0         not released and still: over the last 1 s both finger zones 20 mm or more from both cassettes and both
              arms, the cassettes moved at most 2 mm and turned at most 2 deg, the arms turned at most 2 deg
    score     the sum over both sides (0 .. 1.0, steps of 0.05); success: 1.0
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

GEL_A, GEL_B, FRAME = "gel_a", "gel_b", "electrode_assembly"
GELS = (GEL_A, GEL_B)
PARTS = (GEL_A, GEL_B, FRAME)
ARM_JOINT = {"pos": "arm_pos", "neg": "arm_neg"}
SIDES = ("pos", "neg")
# corners of the cassette's box (local), for the lowest point
CORNERS = np.array([(sx * L.CS_W / 2, sy * L.CS_T / 2, z) for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, L.CS_H)])
FACE_PTS = [(sx * 0.046, z) for sx in (-1, 1) for z in (0.015, 0.060)]  # on the inner face: (local x, local z)


def rotation(quat_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def tilt_deg(r: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def turn_deg(r0: np.ndarray, r1: np.ndarray) -> float:
    c = (float(np.trace(r0.T @ r1)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


@dataclass(frozen=True)
class Gel:
    o: np.ndarray  # origin (bottom-edge middle), table top at z 0
    r: np.ndarray

    def world(self, p) -> np.ndarray:
        return self.o + self.r @ np.asarray(p, dtype=float)

    @property
    def low(self) -> float:
        return float(np.min((self.o + CORNERS @ self.r.T)[:, 2]))


def gel_from_state(s) -> Gel | None:
    try:
        o = np.asarray(s.origin, dtype=float).ravel()
        r = rotation(s.quat_xyzw)
    except (AttributeError, TypeError, ValueError):
        return None
    if o.size != 3 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
        return None
    return Gel(o - np.array([0.0, 0.0, L.TOP]), r)


def fit(g: Gel, side: str) -> dict:
    """How cassette ``g`` sits against the ``side`` gasket."""
    s = L.arm_sign(side)
    n_short = -g.r[:, 1]  # the short plate's outward normal
    inward = bool(n_short[1] * (-s) > 0.0)
    plane = math.degrees(math.asin(min(1.0, abs(float(g.r[1, 0])))))  # its width off the gasket's line (x)
    face_y = -L.CS_T / 2 if inward else L.CS_T / 2  # the face toward the centre
    gaps = [float(s * g.world((x, face_y, z))[1] - L.GY) for x, z in FACE_PTS]
    outer = [float(s * g.world((x, -face_y, z))[1] - (L.GY + L.CS_T)) for x, z in FACE_PTS]
    mid = g.world((0.0, 0.0, L.CS_H / 2))
    short_top = float(g.world((0.0, -L.CS_T / 2 + L.CS_TS / 2, L.CS_HS))[2])
    return {"inward": inward, "tilt": tilt_deg(g.r), "plane": plane, "gaps": gaps, "outer": outer,
            "x_err": float(mid[0] - L.FX), "side_ok": bool(s * float(mid[1]) > 0.0),
            "below_notch": L.NOTCH_Z - short_top}


def against(f: dict, tol: float = L.AGAINST_TOL) -> bool:
    return bool(f["side_ok"] and f["tilt"] <= L.TILT_MAX_DEG + L.EPS and f["plane"] <= L.YAW_MAX_DEG + L.EPS
                and abs(f["x_err"]) <= L.X_TOL + L.EPS and all(abs(v) <= tol + L.EPS for v in f["gaps"]))


def seated(f: dict) -> bool:
    return bool(against(f) and f["inward"] and -L.EPS <= f["below_notch"] <= L.NOTCH_LOW + L.EPS)


def present(f: dict) -> bool:
    """For the clip rule: a cassette (either way round) pressed flat on the gasket, its outer face within 1.5 mm of
    the shut arm's line, so the cam has something of the right thickness to ride over."""
    return bool(f["side_ok"] and f["tilt"] <= L.TILT_MAX_DEG and f["plane"] <= L.YAW_MAX_DEG
                and abs(f["x_err"]) <= L.X_TOL and all(abs(v) <= L.PRESENT_TOL for v in f["outer"]))


def arm_shut(q: float) -> bool:
    return abs(q - L.Q_CLOSED) <= math.radians(L.ARM_WIN_DEG) + L.EPS


def side_credit(gels: dict, q: float, side: str) -> tuple[float, dict]:
    """The best credit any cassette earns on ``side`` (and what that cassette shows)."""
    cands = []
    for name, g in gels.items():
        f = fit(g, side)
        c = 0.0
        if against(f) and f["inward"]:
            c = L.CREDIT_AGAINST + (L.CREDIT_SEATED if seated(f) else 0.0) + (L.CREDIT_ARM if arm_shut(q) else 0.0)
        cands.append((c, f["side_ok"], name, f))
    c, _, name, f = max(cands, key=lambda t: (t[0], t[1]))
    info = {"gel": name, "against": against(f), "inward": f["inward"], "seated": seated(f),
            "gap_mm": [round(v * 1000, 2) for v in f["gaps"]], "tilt_deg": round(f["tilt"], 2),
            "plane_deg": round(f["plane"], 2), "x_err_mm": round(f["x_err"] * 1000, 2),
            "below_notch_mm": round(f["below_notch"] * 1000, 2), "arm_deg": round(math.degrees(q), 2),
            "arm_shut": arm_shut(q), "credit": round(c, 3)}
    return c, info


@dataclass(frozen=True)
class Reading:
    gels: dict  # name -> Gel
    q: dict  # side -> arm angle (rad)


def read(states) -> Reading | None:
    if not isinstance(states, dict):
        return None
    gels = {}
    for n in GELS:
        s = states.get(n)
        g = gel_from_state(s) if s is not None else None
        if g is None:
            return None
        gels[n] = g
    fr = states.get(FRAME)
    try:
        q = {side: float(fr.joints[ARM_JOINT[side]]) for side in SIDES}
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in q.values()):
        return None
    return Reading(gels, q)


def raw_score(r: Reading) -> tuple[float, dict]:
    total, sides = 0.0, {}
    for side in SIDES:
        c, info = side_credit(r.gels, r.q[side], side)
        total += c
        sides[side] = info
    return round(total, 6), sides


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


class Tracker:
    """The drop latch, the last 1 s of readings (stillness) and of the let-go check, and the reset fallback."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))
        self.resets = 0
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.dropped = None
        self.hist: deque = deque(maxlen=self.window + 1)
        self.released: deque = deque(maxlen=self.window)
        self.prev: Reading | None = None
        self.last = Result(0.0, False, {})

    @staticmethod
    def at_start(r: Reading) -> bool:
        a, b = r.gels[GEL_A].o, r.gels[GEL_B].o
        return bool(np.linalg.norm(a - np.array(L.A_START[:3])) <= L.START_TOL
                    and np.linalg.norm(b - np.array(L.B_START[:3])) <= L.START_TOL)

    def teleported(self, prev: Reading, now: Reading) -> bool:
        jump = max(float(np.linalg.norm(now.gels[n].o - prev.gels[n].o)) for n in GELS)
        return bool(self.at_start(now) and jump >= L.RESET_JUMP)

    def update(self, states, arms) -> Result:
        self.steps += 1
        r = read(states)
        if r is None:
            self.bad_steps += 1
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps,
                                            "dropped": self.dropped})
            return self.last
        if self.prev is not None and self.teleported(self.prev, r):
            self.reset()
            self.steps = 1
            self.resets += 1
        self.hist.append(r)
        self.prev = r
        if self.dropped is None and any(g.low < -L.FLOOR_DROP for g in r.gels.values()):
            self.dropped = self.steps
        try:
            fns = [states[n].distance for n in GELS]
            frame = states[FRAME]

            def dist(points):
                d = [f(points) for f in fns]
                d.append(frame.distance(points, bodies=list(ARM_JOINT.values())))
                return np.min(d, axis=0)
        except Exception:  # noqa: BLE001 - no distance: not released
            dist = None
        ok, gap = False, None
        if dist is not None:
            try:
                ok = bool(rigstate.released(arms, dist, L.LET_GO))
                gap = rigstate.hands_gap(arms, dist)
            except Exception:  # noqa: BLE001
                ok = False
        self.released.append(ok)
        self.last = self.judge(r, gap)
        return self.last

    def motion(self) -> dict | None:
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        first = h[0]
        move = turn = arm = 0.0
        for x in h[1:]:
            for n in GELS:
                move = max(move, float(np.linalg.norm(x.gels[n].o - first.gels[n].o)))
                turn = max(turn, turn_deg(first.gels[n].r, x.gels[n].r))
            for side in SIDES:
                arm = max(arm, abs(math.degrees(x.q[side] - first.q[side])))
        return {"move": move, "turn": turn, "arm": arm}

    def judge(self, r: Reading, gap) -> Result:
        raw, sides = raw_score(r)
        released = len(self.released) == self.window and all(self.released)
        m = self.motion()
        still = bool(m is not None and m["move"] <= L.STILL_MOVE + L.EPS and m["turn"] <= L.STILL_TURN_DEG + L.EPS
                     and m["arm"] <= L.STILL_ARM_DEG + L.EPS)
        level = raw if (released and still and self.dropped is None) else 0.0
        details = {
            "valid_state": True, "score": level, "raw_score": raw, "dropped": self.dropped,
            "pos": sides["pos"], "neg": sides["neg"],
            "released_now": bool(self.released[-1]), "released_1s": bool(released), "still_1s": still,
            "hands_gap_mm": None if gap is None or not math.isfinite(gap) else round(gap * 1000, 1),
            "gel_low_mm": {n: round(g.low * 1000, 1) for n, g in r.gels.items()},
            "motion": None if m is None else {k: round(v * (1000 if k == "move" else 1), 3) for k, v in m.items()},
        }
        return Result(level, bool(level >= 1.0 - 1e-6), details)
