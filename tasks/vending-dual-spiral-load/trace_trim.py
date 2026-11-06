"""Read the trace of one GPU demo run and say how to tune the demo's place point and release (task.py).

    python3 trace_trim.py ~/yam-eval/vending-dual-spiral-load/demo/trace.jsonl

TASK_TRACE=<file> makes task.py write one JSON row per control step. This reads the last episode in the file and
prints, for each box: its centre's x offset from the left tool point while carried (the grip offset), where it
stood in its pitch along x just before the pads opened (the place error: the mean of its last 0.5 s with the
gripper still closed on it), its largest tilt before it was lifted (the pads closing) and from the release to the
end, and how it ended (pitch, tilt, lowest point, jam). Then the PLACE_TRIM that cancels the median place error,
and for the divider its largest tilt while held (lifting it out of the slot) and its lowest point at the end (about
5 mm on the mat). No simulator needed; NumPy is not used.
"""

from __future__ import annotations

import json
import pathlib
import re
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import layout as L  # noqa: E402

# The left gripper's measured opening (pad gap / 93.9 mm) under which it is still closed on a box: the pads stop on
# the box's 30 mm (0.32) under the 22 mm command, and the 46 mm release reads 0.49; 0.36 is 2 mm off each face.
CLOSED = 0.36
CARRIED_MM = 60.0  # a held box whose lowest point is this high is being carried, clear of everything
WINDOW = int(round(L.STILL_S * L.CONTROL_HZ))  # 10 control steps: 0.5 s
SPREAD_MM = 6.0  # place errors spread wider than this need more than one trim


def current_trim(task_py: pathlib.Path = HERE / "task.py") -> float:
    """PLACE_TRIM as task.py sets it now (0 if it cannot be read)."""
    try:
        m = re.search(r"^PLACE_TRIM = ([-+0-9.eE]+)", task_py.read_text(), re.M)
        return float(m.group(1)) if m else 0.0
    except (OSError, ValueError):
        return 0.0


def load(path) -> list[dict]:
    """The rows of the last episode in a trace file that carry a valid state."""
    rows = []
    for line in pathlib.Path(path).read_text().splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("valid_state") and isinstance(r.get("boxes"), dict):
            rows.append(r)
    if rows:
        last = rows[-1].get("episode")
        rows = [r for r in rows if r.get("episode") == last]
    return rows


def _opening(r) -> float:
    o = r.get("opening")
    return float(o["left"]) if isinstance(o, dict) and "left" in o else 1.0


def report(rows: list[dict], trim: float = 0.0) -> dict:
    """Per-box and divider findings, and the suggested PLACE_TRIM (m), as a dict."""
    out = {"current_trim_mm": round(trim * 1000, 1), "boxes": {}, "suggested_trim_mm": None, "warnings": []}
    errors = []
    for k, name in enumerate(L.BOX_NAMES):
        b = [r["boxes"][name] for r in rows]
        held = [i for i, s in enumerate(b) if s["held"]]
        if not held:
            out["boxes"][name] = {"held": False}
            out["warnings"].append(f"{name} was never held")
            continue
        carried = [i for i in held if (b[i]["low_mm"] or 0.0) >= CARRIED_MM and rows[i].get("left_tcp")]
        grip = (statistics.median(b[i]["pos"][0] - rows[i]["left_tcp"][0] for i in carried) * 1000
                if carried else None)
        lifted = carried[0] if carried else held[0]
        closed = [i for i in held if i > lifted and _opening(rows[i]) < CLOSED]
        place = closed[-WINDOW:]
        err = (statistics.mean(b[i]["pos"][0] for i in place) - L.PITCH_X[k]) * 1000 if place else None
        if err is not None:
            errors.append(err)
        released = (place[-1] + 1) if place else held[-1]
        end = b[-1]
        out["boxes"][name] = {
            "held": True,
            "grip_offset_x_mm": None if grip is None else round(grip, 1) + 0.0,
            "place_error_x_mm": None if err is None else round(err, 1) + 0.0,
            "tilt_before_lift_deg": max(s["tilt_deg"] for s in b[:lifted + 1]),
            "tilt_after_release_deg": max(s["tilt_deg"] for s in b[released:]) if released < len(b) else None,
            "end": {"pitch": end["pitch"], "tilt_deg": end["tilt_deg"], "low_mm": end["low_mm"], "jam": end["jam"]},
        }
        if err is not None and abs(err) > 12.0:
            out["warnings"].append(f"{name}: placed {err:+.1f} mm off its pitch's centre (12 mm of play)")
    if errors:
        med = statistics.median(errors)
        out["suggested_trim_mm"] = round((trim * 1000 - med) * 2) / 2 + 0.0  # to 0.5 mm
        if max(errors) - min(errors) > SPREAD_MM:
            out["warnings"].append(f"place errors spread {max(errors) - min(errors):.1f} mm: one trim will not centre "
                                   "every box")
    d = [r["divider"] for r in rows]
    dh = [s["tilt_deg"] for s in d if s["held"]]
    out["divider"] = {"tilt_while_held_deg": max(dh) if dh else None, "end_low_mm": d[-1]["low_mm"] if d else None,
                      "on_mat": rows[-1].get("divider_on_mat") if rows else None}
    out["end"] = {"score": rows[-1].get("score"), "n": rows[-1].get("n"), "steps": len(rows)} if rows else None
    return out


def main(argv) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    rows = load(argv[1])
    if not rows:
        print("no valid rows in", argv[1])
        return 1
    rep = report(rows, current_trim())
    print(json.dumps(rep, indent=1))
    if rep["suggested_trim_mm"] is not None:
        print(f"set PLACE_TRIM = {rep['suggested_trim_mm'] / 1000:.4f} in task.py (now {rep['current_trim_mm']} mm)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
