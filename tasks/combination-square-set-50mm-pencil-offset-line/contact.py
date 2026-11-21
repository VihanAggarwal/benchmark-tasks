"""The PhysX contact force between the pencil and the board, for the scorer (simulator only).

Copied from tasks/knife-pull-through-sharpener-coarse-then-fine/contact.py after that task's read fix (names are set
by install()). The first copy here was the knife task's first version, and this task's first GPU run (21 Nov) lost
every read the same way: RuntimeError('Invalid indexing in slice: 1') on every step, so no ink could ever count.

How: when mechkit writes the pencil's and the board's prims, this module also applies PhysxContactReportAPI
(threshold 0) to both base bodies, and turns PhysX contact processing on (the carb setting
``/physics/disableContactProcessing`` False). At the first scored step it builds a rigid-contact view on the
first body (the pencil) filtered against the second (the board) (``physics_sim_view.create_rigid_contact_view``) and
from then on reads ``get_contact_force_matrix(dt=...)`` once per control step: the summed contact force of the last
physics substep, pencil on board, in newtons. If that read raises, the same view's ``get_contact_data`` is summed
instead (a second read of the same PhysX normal forces). Buffers go through ``to_numpy`` (Isaac Lab 3 ``.torch``,
torch, warp ``.numpy()``): the knife task's first GPU run (19 Nov) lost every read to ``np.asarray`` on a warp array,
and its second run (job 20261119T061906) read numbers on every step with this code. The pattern is the one
research/sim_capabilities.md describes (a PhysX rigid-contact view between chosen bodies); this is an independent
implementation.

Fail closed: if the view cannot be built or neither read works (or a read matches no body pair), ``force()``
returns None and the scorer counts no ink in the simulator (the trace says why: ``contact_error``).
Offline (no simulator) nothing here runs; the scorer then uses its geometric rule only, and says so.

The force is between the two whole bodies (any part of the pencil on any part of the board). The scorer asks for
it only on steps where the pose check already puts the lead's tip on the board's top face; the only part of the
pencil that reaches the board is the lead.
"""

from __future__ import annotations

import sys

import numpy as np

import mechkit as mk

PHYSICS_DT = 1.0 / 240.0  # yam_arena's PhysX step
_STATE: dict = {"view": None, "error": None, "built": False, "episode": None, "installed": False}
NAMES: tuple[str, str] = ("pencil", "board")


def _loud(msg: str) -> None:
    print(f"[contact] {msg}", file=sys.stderr, flush=True)


def install(first: str, second: str) -> bool:
    """Wrap mechkit's prim writer so the two bodies get contact reports. Returns False offline."""
    global NAMES
    NAMES = (first, second)
    impl = getattr(mk._hub(), "impl", None)
    if impl is None or not hasattr(impl, "author"):
        return False
    original = getattr(impl.author, "_contact_original", impl.author)

    def author(stage, prim_path, spec, translation=None, orientation=None):
        prim = original(stage, prim_path, spec, translation, orientation)
        if getattr(spec, "name", None) in NAMES:
            try:
                from pxr import PhysxSchema

                body = stage.GetPrimAtPath(f"{prim_path}/base")
                api = PhysxSchema.PhysxContactReportAPI.Apply(body)
                api.CreateThresholdAttr().Set(0.0)
            except Exception as exc:  # noqa: BLE001 - reported, and the scorer then fails closed
                _STATE["install_error"] = f"contact report API not applied: {exc!r}"
                _loud("ERROR " + _STATE["install_error"])
        return prim

    author._contact_original = original
    impl.author = author
    try:
        import carb
    except ImportError:  # offline: no Kit, nothing to enable
        carb = None
    try:
        if carb is not None:
            carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception as exc:  # noqa: BLE001
        _STATE["install_error"] = f"contact processing not enabled: {exc!r}"
        _loud("ERROR " + _STATE["install_error"])
    _STATE["installed"] = True
    return True


def _sim_view():
    """The PhysX tensor sim view: Isaac Lab's SimulationContext (what the first GPU run used, and it built), else
    Isaac Lab 3's PhysxManager (what the accepted cut-taut-cord and peel-the-cucumber tasks use)."""
    errors = []
    try:
        import isaaclab.sim as sim_utils

        view = getattr(sim_utils.SimulationContext.instance(), "physics_sim_view", None)
        if view is not None:
            return view
        errors.append("SimulationContext.physics_sim_view is None")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"SimulationContext: {exc!r}")
    try:
        from isaaclab_physx.physics import PhysxManager

        view = PhysxManager.get_physics_sim_view()
        if view is not None:
            return view
        errors.append("PhysxManager.get_physics_sim_view() is None")
    except Exception as exc:  # noqa: BLE001
        errors.append(f"PhysxManager: {exc!r}")
    raise RuntimeError("; ".join(errors))


def _build():
    _STATE["built"] = True
    env = mk.env()
    if env is None:
        _STATE["error"] = "no simulator env"
        return None
    try:  # Isaac Lab may have switched contact processing off while it configured the scene: switch it on again
        import carb

        carb.settings.get_settings().set_bool("/physics/disableContactProcessing", False)
    except Exception:  # noqa: BLE001 - offline or no Kit; the read below says whether contacts arrive
        pass
    try:
        view = _sim_view().create_rigid_contact_view(
            mk._body_path(NAMES[0], "base"), filter_patterns=[mk._body_path(NAMES[1], "base")],
            max_contact_data_count=64)
        _STATE["view"] = view
        _loud(f"rigid-contact view built: {mk._body_path(NAMES[0], 'base')} vs {mk._body_path(NAMES[1], 'base')}")
        return view
    except Exception as exc:  # noqa: BLE001
        _STATE["error"] = f"rigid-contact view not built: {exc!r}"
        _loud("ERROR " + _STATE["error"])
        return None


def to_numpy(value) -> np.ndarray:
    """A float64 numpy array from whatever the PhysX tensor API hands back in this build.

    The first GPU runs (knife task 19 Nov, this task 21 Nov) failed here: ``np.asarray`` on the returned buffer
    raised ``RuntimeError('Invalid indexing in slice: 1')``, which is warp's ``array.__getitem__`` refusing numpy's
    item-by-item fallback (a warp array on the GPU has no ``.cpu()`` and no numpy array interface). The order below
    is the one the accepted tasks use (tasks/cut-taut-cord/cordkit.py ``_arr``, peel-the-cucumber ``_arr``,
    mechkit ``_np``): an Isaac Lab 3 buffer's ``.torch``, a torch tensor (``.detach().cpu().numpy()``), anything
    with ``.numpy()`` (a warp array copies itself to the host), and only then ``np.asarray``."""
    t = getattr(value, "torch", None)
    if t is not None and not callable(t):
        value = t
    if hasattr(value, "detach"):
        return np.asarray(value.detach().cpu().numpy(), dtype=np.float64)
    numpy_fn = getattr(value, "numpy", None)
    if callable(numpy_fn):
        return np.asarray(numpy_fn(), dtype=np.float64)
    if isinstance(value, np.ndarray):
        return value.astype(np.float64)
    if isinstance(value, (list, tuple, float, int)):
        return np.asarray(value, dtype=np.float64)
    raise TypeError(f"cannot convert {type(value).__name__} to numpy")


def _describe(value) -> str:
    return f"{type(value).__module__}.{type(value).__name__} shape {getattr(value, 'shape', '?')}"


def _from_matrix(view) -> float:
    """Newtons from ``get_contact_force_matrix``: (sensors, filters, 3), summed over the pencil-on-board pairs."""
    raw = view.get_contact_force_matrix(dt=PHYSICS_DT)
    if not _STATE.get("described"):
        _STATE["described"] = True
        _loud(f"force matrix returned {_describe(raw)}")
    m = to_numpy(raw)
    if m.size == 0 or m.size % 3:
        raise RuntimeError(f"force matrix has {m.size} numbers: the view matched no pencil/board body pair")
    return float(np.linalg.norm(m.reshape(-1, 3).sum(axis=0)))


def _from_contact_data(view) -> float:
    """Newtons from ``get_contact_data`` (the fallback): every contact point's normal force times its normal,
    summed (the same normal forces the matrix sums; tasks/peel-the-cucumber reads it this way)."""
    forces, _points, normals, _seps, counts, starts = view.get_contact_data(dt=PHYSICS_DT)
    f, n = to_numpy(forces).reshape(-1), to_numpy(normals).reshape(-1, 3)
    c, s = to_numpy(counts).reshape(-1).astype(int), to_numpy(starts).reshape(-1).astype(int)
    if c.size == 0:
        raise RuntimeError("contact data has no sensor/filter pair: the view matched no body pair")
    total = np.zeros(3)
    for k, i0 in zip(c, s):
        if k < 0 or i0 < 0 or i0 + k > f.size or i0 + k > len(n):
            raise RuntimeError(f"contact data range {i0}+{k} outside {f.size} points")
        total += (f[i0:i0 + k, None] * n[i0:i0 + k]).sum(axis=0)
    return float(np.linalg.norm(total))


READERS = (("force matrix", _from_matrix), ("contact data", _from_contact_data))


def force() -> float | None:
    """Newtons, pencil on board, at the last physics substep; None when it cannot be read (fail closed).

    Two independent reads of the same PhysX view: the force matrix, then (if that raises) the per-point contact
    data. Each failure is printed once and kept in ``error()``; a read that comes back empty (no body pair matched)
    is a failure too, never a silent 0 N."""
    view = _STATE["view"] if _STATE["built"] else _build()
    if view is None:
        return None
    problems = []
    for name, reader in READERS:
        try:
            f = reader(view)
        except Exception as exc:  # noqa: BLE001
            problems.append(f"{name}: {exc!r}")
            continue
        if not np.isfinite(f):
            problems.append(f"{name}: not finite ({f})")
            continue
        if _STATE.get("reader") != name:
            _STATE["reader"] = name
            _loud(f"contact force read through the {name}")
        _STATE["error"] = None if name == READERS[0][0] else "force matrix failed, contact data used: " + problems[0]
        return f
    msg = "contact force not read: " + "; ".join(problems)
    if _STATE.get("read_error") != msg:
        _STATE["read_error"] = msg
        _loud("ERROR " + msg)
    _STATE["error"] = msg
    return None


def error() -> str | None:
    """What went wrong (install or read), for the trace; None when the last read worked and install was clean."""
    parts = [m for m in (_STATE.get("install_error"), _STATE["error"]) if m]
    return "; ".join(parts) or None
