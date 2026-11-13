"""The fastening rules and the scorer, as plain functions of the parts' states and the arms' state. NumPy only.

Two pieces, both fed one observation per control step:

``Fastening.update(states)``: the thread and clamp rules. A screw is ENGAGED in an insert when its stub has gone
through one of a binding's slots into that insert (the stub's tip 5 mm or more down it), the screw is upright within
10 deg and its washer is within 1 mm of the disc's top; it stays engaged until the washer is 3 mm up, the stub leaves
that insert or slot, or it tilts past 10 deg. While engaged, its turn about its axis is summed (clockwise seen from
above is positive, counter-clockwise subtracts); disengaging sets the sum back to 0. At one full turn (360 deg) the
screw is DRIVEN: ``update`` returns it, and task.py welds it to the board where it stands (upright, washer on the
disc's top). When all four slots of a binding hold driven screws and the binding lies flat, it is CLAMPED: task.py
welds it flat on the board. Neither is ever undone within a trial. In the simulator task.py runs this from a
mechkit ``on_step`` rule (before the physics); offline it runs from ``success()``.

``Tracker.update(states, arms, fastening)`` -> Result(score, success, details): the end state, judged every step.
Per binding: ALIGNED (flat: underside within 1.5 mm of the board's top and tilted 3 deg or less; its angle within
3 deg of its target; a 5 mm stub could pass each of its four slots into the insert under it), SEATED n (screws seated
through its slots: upright within 5 deg, washer within 1 mm of the disc, stub 5 mm down an insert), DRIVEN n and
CLAMPED. Release: both grippers' finger zones 30 mm or more from every part of both bindings and of all eight screws
on every step of the last 1 s; an unreadable arm or a distance that cannot be computed is NOT released (rigstate's
fail-closed rule). Still: every binding and screw moved 2 mm or less and turned 1 deg or less over that 1 s.

    1.0  both fastened (aligned, 4 driven, clamped) on the middle squares (discs 400 mm apart, centred on the cross)
    0.75 both fastened, on other squares (one on each half)
    0.5  both aligned, each with 4 screws seated
    0.25 one aligned with 4 screws seated
    0    otherwise, or not released, or anything moving; and from the step a binding or a screw falls off the table
         (below the table top by 20 mm) for the rest of the trial

The front binding is the one nearer the nose (larger y), target +15 deg; the other is the back binding, -15 deg.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

import layout as L
import rigstate

BINDINGS = ("binding_nose", "binding_tail")
SCREWS = tuple(f"screw_{h}{k}" for h in ("n", "t") for k in range(1, 5))
PARTS = BINDINGS + SCREWS
STARTS = {"binding_nose": L.NOSE_START, "binding_tail": L.TAIL_START}
INSERT_XY = np.array([(x, y) for _, x, y in L.INSERTS])
INSERT_NAMES = tuple(n for n, _, _ in L.INSERTS)
SLOT_XY = np.array(L.SLOT_CENTRES)
# Free play of a 5 mm stub: in a bore (1.5 mm a side) and in a slot (7.5 mm along, 2.5 mm across)
BORE_PLAY = L.INSERT_HOLE / 2 - L.STUB_R
SLOT_PLAY = (L.SLOT_LEN / 2 - L.STUB_R, L.SLOT_W / 2 - L.STUB_R)
# 'Aligned': a stub at some point of each slot could go straight down into the insert under it, so each insert's
# centre lies within (slot play + bore play) of its slot's centre: 11.0 mm along the slot, 4.0 mm across it
ALIGN_PLAY = (SLOT_PLAY[0] + BORE_PLAY, SLOT_PLAY[1] + BORE_PLAY)


# -- geometry ------------------------------------------------------------------------------------------------------


def rotation(quat_xyzw) -> np.ndarray:
    """3 x 3 rotation matrix of a quaternion (x, y, z, w); its columns are the body's axes in the world."""
    x, y, z, w = (float(v) for v in quat_xyzw)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def wrap_deg(a: float) -> float:
    """An angle in degrees wrapped to (-180, 180]."""
    a = math.fmod(float(a) + 180.0, 360.0)
    if a <= 0.0:
        a += 360.0
    return a - 180.0


def frame(s) -> tuple[np.ndarray, np.ndarray]:
    """(origin, rotation) of a part's frame: an AssemblyState's ``origin``, else its ``pos``."""
    o = getattr(s, "origin", None)
    return np.asarray(s.pos if o is None else o, dtype=float).reshape(3), rotation(s.quat_xyzw)


def valid(s) -> bool:
    """The part's origin and orientation present, every number finite and the quaternion not degenerate."""
    try:
        o = getattr(s, "origin", None)
        o = np.asarray(s.pos if o is None else o, dtype=float).ravel()
        q = np.asarray(getattr(s, "quat_xyzw"), dtype=float).ravel()
    except (AttributeError, TypeError, ValueError):
        return False
    if o.size != 3 or q.size != 4 or not (np.all(np.isfinite(o)) and np.all(np.isfinite(q))):
        return False
    return bool(np.linalg.norm(q) > 0.5)


def heading_deg(r: np.ndarray) -> float:
    """Heading of a frame's x axis seen from above, degrees in (-180, 180], 0 = +x, positive toward +y."""
    return wrap_deg(math.degrees(math.atan2(r[1, 0], r[0, 0])))


def tilt_deg(r: np.ndarray) -> float:
    """Angle between a frame's z axis and the vertical, degrees 0..180."""
    return math.degrees(math.acos(max(-1.0, min(1.0, float(r[2, 2])))))


def binding_angle(name: str, r: np.ndarray) -> float:
    """A binding's angle: the heading of its baseplate's heel-to-toe axis, its disc frame's heading plus its
    preset."""
    return wrap_deg(heading_deg(r) + L.PRESET_DEG[name])


def nearest_insert(xy) -> tuple[int, float, float]:
    """(index, dx, dy) of the insert nearest a world point (x, y)."""
    d = INSERT_XY - np.asarray(xy, dtype=float)[:2]
    i = int(np.argmin(np.einsum("ij,ij->i", d, d)))
    return i, float(-d[i, 0]), float(-d[i, 1])


def slots_over(origin, r) -> list[int | None]:
    """For each of a binding's four slots, the insert a 5 mm stub could go into through it (None: no insert in
    reach): the insert's centre within ALIGN_PLAY of the slot's centre, measured in the disc's frame."""
    out = []
    for sx, sy in SLOT_XY:
        c = origin + r @ np.array([sx, sy, L.FLOOR_T / 2])
        i, _, _ = nearest_insert(c)
        local = r.T @ (np.array([*INSERT_XY[i], c[2]]) - origin)
        ok = abs(local[0] - sx) <= ALIGN_PLAY[0] + L.EPS and abs(local[1] - sy) <= ALIGN_PLAY[1] + L.EPS
        out.append(i if ok else None)
    return out


@dataclass(frozen=True)
class Fit:
    """Where one screw is relative to one binding and the board."""
    slot: int | None  # the binding's slot its stub passes through (None: not in a slot)
    insert: int | None  # the insert its stub is in (None: not in a bore)
    gap: float  # its washer's underside above the disc's top (m; negative: below it)
    tilt: float  # deg
    depth: float  # its stub's tip below the board's top (m)


def screw_fit(screw, binding) -> Fit:
    """Measure one screw against one binding (both states)."""
    so, sr = frame(screw)
    bo, br = frame(binding)
    axis = sr[:, 2]
    tilt = tilt_deg(sr)
    local = br.T @ (so - bo)
    a_l = br.T @ axis
    gap = float(local[2]) - L.FLOOR_T
    slot = None
    if abs(a_l[2]) > 0.5:
        mid = local + a_l * ((L.FLOOR_T / 2 - local[2]) / a_l[2])  # where the axis crosses the disc's mid-plane
        for k, (sx, sy) in enumerate(SLOT_XY):
            if (abs(mid[0] - sx) <= SLOT_PLAY[0] + L.FIT_SLACK + L.EPS
                    and abs(mid[1] - sy) <= SLOT_PLAY[1] + L.FIT_SLACK + L.EPS):
                slot = k
                break
    insert = None
    depth = -math.inf
    if axis[2] > 0.5:
        at_top = so + axis * ((L.BOARD_TOP - so[2]) / axis[2])  # where the axis crosses the board's top
        i, dx, dy = nearest_insert(at_top)
        if max(abs(dx), abs(dy)) <= BORE_PLAY + L.FIT_SLACK + L.EPS:
            insert = i
        depth = L.BOARD_TOP - float(so[2] - L.STUB_L * axis[2])
    return Fit(slot, insert, gap, tilt, depth)


def seated(fit: Fit) -> bool:
    """End state: through a slot into an insert, 5 mm down it, upright within 5 deg, washer within 1 mm of the
    disc's top."""
    return bool(fit.slot is not None and fit.insert is not None and fit.depth >= L.STUB_DEPTH_MIN - L.EPS
                and fit.tilt <= L.SCREW_TILT_DEG + L.EPS_DEG and abs(fit.gap) <= L.SEAT_GAP + L.EPS)


# -- the thread and clamp rules ------------------------------------------------------------------------------------


@dataclass
class ScrewRule:
    binding: str | None = None  # engaged through this binding's slot ...
    slot: int | None = None
    insert: int | None = None  # ... into this insert
    turn: float = 0.0  # summed clockwise turn while engaged (deg)
    yaw: float | None = None  # last heading seen
    driven: bool = False
    home: tuple | None = None  # the welded pose: (position, quat_xyzw)


class Fastening:
    """Thread engagement, turn counting, driving and clamping (see the module docstring). ``update`` returns the
    events of this step: [("drive", screw, (pos, quat)), ("clamp", binding, (pos, quat))]."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.screws = {n: ScrewRule() for n in SCREWS}
        self.clamped: dict[str, tuple] = {}  # binding -> its welded pose
        self.key = None
        self.events_seen: list = []

    def driven_in(self, binding: str) -> set[int]:
        return {r.slot for r in self.screws.values() if r.driven and r.binding == binding}

    def update(self, states: dict, key=None) -> list:
        """One step. ``key`` (episode, step): a second call with the same key does nothing (the sim's on_step and
        the offline call from success() never both count a step)."""
        if key is not None and key == self.key:
            return []
        self.key = key
        events = []
        binds = {b: states.get(b) for b in BINDINGS}
        for name, rule in self.screws.items():
            s = states.get(name)
            if rule.driven or s is None or not valid(s):
                continue
            _, sr = frame(s)
            yaw = heading_deg(sr)
            d = 0.0 if rule.yaw is None else wrap_deg(yaw - rule.yaw)
            rule.yaw = yaw
            fits = {b: screw_fit(s, bs) for b, bs in binds.items() if bs is not None and valid(bs)}
            if rule.binding is not None:  # engaged: does it stay?
                f = fits.get(rule.binding)
                stays = (f is not None and f.insert == rule.insert and f.slot == rule.slot
                         and f.gap <= L.DISENGAGE_GAP + L.EPS and f.tilt <= L.ENGAGE_TILT_DEG + L.EPS_DEG)
                if stays:
                    rule.turn += -d  # clockwise from above: the heading decreases
                else:
                    rule.binding = rule.slot = rule.insert = None
                    rule.turn = 0.0
            if rule.binding is None:
                for b, f in fits.items():
                    if (f.slot is not None and f.insert is not None and f.depth >= L.STUB_DEPTH_MIN - L.EPS
                            and f.gap <= L.ENGAGE_GAP + L.EPS and f.tilt <= L.ENGAGE_TILT_DEG + L.EPS_DEG):
                        rule.binding, rule.slot, rule.insert, rule.turn = b, f.slot, f.insert, 0.0
                        break
            if rule.binding is not None and rule.turn >= L.DRIVE_TURN_DEG - L.EPS_DEG:
                rule.driven = True
                rule.home = self._home(s, binds[rule.binding])
                events.append(("drive", name, rule.home))
        for b, bs in binds.items():
            if b in self.clamped or bs is None or not valid(bs):
                continue
            if len(self.driven_in(b)) == 4:
                o, r = frame(bs)
                if abs(float(o[2]) - L.BOARD_TOP) <= L.FLAT_Z_TOL + L.EPS and tilt_deg(r) <= L.FLAT_TILT_DEG + 1e-6:
                    yaw = math.atan2(r[1, 0], r[0, 0])
                    self.clamped[b] = ((float(o[0]), float(o[1]), L.BOARD_TOP),
                                       (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))
                    events.append(("clamp", b, self.clamped[b]))
        self.events_seen.extend(events)
        return events

    @staticmethod
    def _home(screw, binding) -> tuple:
        """The pose a driven screw is welded at: upright on its axis where it stands, its washer on the disc's top,
        its heading kept."""
        so, sr = frame(screw)
        bo, br = frame(binding)
        local = br.T @ (so - bo)
        top = bo + br @ np.array([local[0], local[1], L.FLOOR_T])
        yaw = math.atan2(sr[1, 0], sr[0, 0])
        return ((float(so[0]), float(so[1]), float(top[2])), (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)))

    def status(self) -> dict:
        return {n: {"binding": r.binding, "slot": r.slot,
                    "insert": None if r.insert is None else INSERT_NAMES[r.insert],
                    "turn_deg": round(r.turn, 1), "driven": r.driven} for n, r in self.screws.items()}


# -- release and the scorer ----------------------------------------------------------------------------------------


def parts_distance(states: dict):
    """A distance function (world points -> distance to the nearest part of any binding or screw) for rigstate's
    ``released``: the parts' own exact distance (mechkit ``AssemblyState.distance``). A part whose state carries no
    distance raises, which rigstate counts as NOT released."""
    fns = [states[n].distance for n in PARTS]

    def dist(points):
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        best = np.full(len(pts), np.inf)
        for f in fns:
            best = np.minimum(best, np.asarray(f(pts), dtype=float).reshape(-1))
        return best

    return dist


@dataclass
class Result:
    score: float
    success: bool
    details: dict = field(default_factory=dict)


def tier(per: dict, square_of: dict) -> float:
    """The score from each binding's flags (``aligned``, ``seated``, ``fastened``) and the square each one is on
    ((half, centre y) or None)."""
    fastened = [b for b in BINDINGS if per[b]["fastened"]]
    full = [b for b in BINDINGS if per[b]["aligned"] and per[b]["seated"] == 4]
    if len(fastened) == 2:
        front = max(BINDINGS, key=lambda b: square_of[b][1])
        back = BINDINGS[1] if front == BINDINGS[0] else BINDINGS[0]
        if square_of[front][0] != "n" or square_of[back][0] != "t":
            return 0.5  # both fastened on one half: cannot happen with these parts; no stance, so no fastening tier
        middle = (abs(square_of[front][1] - L.MIDDLE_SQUARE_Y) <= L.EPS
                  and abs(square_of[back][1] + L.MIDDLE_SQUARE_Y) <= L.EPS)
        return 1.0 if middle else 0.75
    if len(full) == 2:
        return 0.5
    if len(full) == 1:
        return 0.25
    return 0.0


def _mm(v) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v) * 1000, 2)


class Tracker:
    """The last 1 s of every part's pose (stillness) and of the release check; the lost-part latch."""

    def __init__(self, control_hz: float = L.CONTROL_HZ) -> None:
        self.window = max(1, int(round(L.STILL_WINDOW_S * control_hz)))  # 20 steps at 20 Hz
        self.resets = 0  # fallback resets seen (teleports back to the start), kept across resets
        self.reset()

    def reset(self) -> None:
        self.steps = 0
        self.bad_steps = 0
        self.hist: deque = deque(maxlen=self.window + 1)  # {part: (origin, heading)} per valid step
        self.released: deque = deque(maxlen=self.window)  # released flag per step
        self.prev = None
        self.lost: str | None = None  # the first part that fell off the table (latched)
        self.last = Result(0.0, False, {})

    # -- reset fallback ----------------------------------------------------------------------------------------
    @staticmethod
    def at_start(sample) -> bool:
        for name in BINDINGS:
            origin, heading = sample[name]
            x, y, a = STARTS[name]
            if float(np.linalg.norm(origin - np.array([x, y, L.BOARD_TOP]))) > L.RESET_START_TOL:
                return False
            if abs(wrap_deg(heading - a)) > L.RESET_START_TURN_DEG:
                return False
        return True

    @classmethod
    def teleported(cls, prev, now) -> bool:
        """The simulator put both bindings back on their start poses in one step: a new episode."""
        if not cls.at_start(now):
            return False
        return any(float(np.linalg.norm(now[n][0] - prev[n][0])) > L.RESET_JUMP
                   or abs(wrap_deg(now[n][1] - prev[n][1])) > L.RESET_TURN_DEG for n in BINDINGS)

    # -- per step ------------------------------------------------------------------------------------------------
    def update(self, states: dict, arms, fastening: Fastening) -> Result:
        """One control step: record the poses and the release check, then judge."""
        self.steps += 1
        if not all(n in states and valid(states[n]) for n in PARTS):
            self.bad_steps += 1  # a blow-up or a missing part: 0, and the 1 s windows start again
            self.hist.clear()
            self.released.clear()
            self.prev = None
            self.last = Result(0.0, False, {"valid_state": False, "bad_steps": self.bad_steps})
            return self.last
        sample = {}
        for n in PARTS:
            o, r = frame(states[n])
            sample[n] = (o, binding_angle(n, r) if n in BINDINGS else heading_deg(r))
        if self.prev is not None and self.teleported(self.prev, sample):
            self.reset()
            self.steps = 1
            self.resets += 1
        for n in PARTS:
            if self.lost is None and float(sample[n][0][2]) < L.LOST_Z:
                self.lost = n
        self.hist.append(sample)
        self.prev = sample
        try:
            dist = parts_distance(states)
        except Exception:  # noqa: BLE001 - no distance: rigstate's rule, not released
            dist = None
        self.released.append(dist is not None and rigstate.released(arms, dist, L.LET_GO))
        self.last = self.judge(states, arms, fastening)
        return self.last

    def motion(self, name: str) -> tuple[float, float] | None:
        """(path of the origin in m, summed heading change in deg) over the last 1 s, or None before 1 s of valid
        states has been seen."""
        if len(self.hist) < self.window + 1:
            return None
        h = list(self.hist)
        path = sum(float(np.linalg.norm(b[name][0] - a[name][0])) for a, b in zip(h, h[1:]))
        turn = sum(abs(wrap_deg(b[name][1] - a[name][1])) for a, b in zip(h, h[1:]))
        return path, turn

    def judge(self, states: dict, arms, fastening: Fastening) -> Result:
        poses = {n: frame(states[n]) for n in BINDINGS}
        front = max(BINDINGS, key=lambda n: (float(poses[n][0][1]), n == "binding_nose"))
        targets = {front: L.FRONT_TARGET_DEG, (BINDINGS[1] if front == BINDINGS[0] else BINDINGS[0]):
                   L.BACK_TARGET_DEG}
        fits = {s: {b: screw_fit(states[s], states[b]) for b in BINDINGS} for s in SCREWS}
        per, square_of = {}, {}
        for b in BINDINGS:
            o, r = poses[b]
            angle = binding_angle(b, r)
            err = wrap_deg(angle - targets[b])
            z_above = float(o[2] - L.BOARD_TOP)
            tilt = tilt_deg(r)
            flat = abs(z_above) <= L.FLAT_Z_TOL + L.EPS and tilt <= L.FLAT_TILT_DEG + L.EPS_DEG
            angled = abs(err) <= L.ANGLE_TOL_DEG + L.EPS_DEG
            over = slots_over(o, r)
            square = None
            if all(i is not None for i in over) and len(set(over)) == 4:
                ys = [float(INSERT_XY[i, 1]) for i in over]
                xs = {float(INSERT_XY[i, 0]) for i in over}
                if len(xs) == 2 and max(ys) - min(ys) <= L.INSERT_PITCH + L.EPS:
                    square = ("n" if ys[0] > 0 else "t", round(sum(ys) / 4, 4))
            aligned = bool(flat and angled and square is not None)
            seat_slots = {fits[s][b].slot for s in SCREWS if seated(fits[s][b])}
            driven = {fits[s][b].slot for s in SCREWS
                      if fastening.screws[s].driven and fastening.screws[s].binding == b and seated(fits[s][b])}
            clamped = b in fastening.clamped
            fastened = aligned and clamped and len(driven) == 4
            square_of[b] = square
            per[b] = {"role": "front" if b == front else "back", "origin": [round(float(v), 4) for v in o],
                      "angle_deg": round(angle, 2), "angle_err_deg": round(err, 2), "tilt_deg": round(tilt, 2),
                      "z_above_mm": _mm(z_above), "flat": bool(flat), "angled": bool(angled),
                      "slots_over": [None if i is None else INSERT_NAMES[i] for i in over],
                      "square": square, "aligned": aligned, "seated": len(seat_slots), "driven": len(driven),
                      "clamped": clamped, "fastened": bool(fastened)}
        still = True
        moving = []
        for n in PARTS:
            m = self.motion(n)
            if m is None or m[0] > L.STILL_PATH + L.EPS or m[1] > L.STILL_TURN_DEG + L.EPS_DEG:
                still = False
                moving.append(n)
        released_now = bool(self.released and self.released[-1])
        released_window = len(self.released) == self.window and all(self.released)
        level = tier(per, square_of)
        score = level if (released_window and still and self.lost is None) else 0.0
        gap = None
        try:
            gap = rigstate.hands_gap(arms, parts_distance(states))
        except Exception:  # noqa: BLE001 - trace only
            gap = None
        w = c = None
        if all(square_of[b] is not None for b in BINDINGS):
            ys = sorted((square_of[b][1] for b in BINDINGS), reverse=True)
            w, c = ys[0] - ys[1], (ys[0] + ys[1]) / 2
        details = {"valid_state": True, "tier": level, "score": score, "released": released_now,
                   "released_1s": released_window, "still_1s": still, "moving": moving[:4], "lost": self.lost,
                   "zone_mm": _mm(gap), "arms_readable": rigstate.readable(arms), "W_mm": _mm(w), "C_mm": _mm(c),
                   **per}
        return Result(score, bool(score == 1.0), details)
