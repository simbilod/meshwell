"""Mesh-stage structured-sweep kernel: discovery + pairing validation.

Discovers the synthetic ``__sweep|<name>|<side>|<i>`` (dim 2) and
``__sweepsrc|<name>`` (dim 1) physical groups the CAD stage
(``meshwell.structured.sweep_cad``) writes into the XAO, validates that
the groups found in the loaded model pair 1:1 with the
``StructuredSweepResolutionSpec`` entries the caller supplies, and
builds the hook that runs immediately before ``generate(2)``.

``_stamp_all`` fills band faces with an exact tensor-product mesh
(triangle-pairs or quads) and sets ``Mesh.MeshOnlyEmpty`` so gmsh fills
only the surrounding unstructured surfaces conformally.
``discover_sweeps()`` returns an empty dict when the model has no sweep
groups, so the whole path is a no-op for non-sweep meshes.
"""
from __future__ import annotations

import contextlib
import logging
from collections import defaultdict
from collections.abc import Callable

import gmsh
import numpy as np

from meshwell.resolution import (
    StructuredExtrusionResolutionSpec,
    StructuredSweepResolutionSpec,
    resolve_normal_offsets,
)
from meshwell.structured._zmath import signed_axis
from meshwell.structured.exceptions import (
    SweepPairingError,
    SweepSeamMismatchError,
    SweepSplitCoordinateError,
)

logger = logging.getLogger(__name__)

FACE_PREFIX = "__sweep|"
SRC_PREFIX = "__sweepsrc|"


def discover_sweeps() -> dict[str, dict]:
    """Scan the loaded gmsh model's physical groups for sweep synthetics.

    Returns:
        ``{name: {"faces": {(side, i): face_tag}, "src_curves": set[int]}}``
        keyed by ``StructuredSweep.name``. Empty dict when no ``__sweep``
        groups are present in the model.

    """
    faces: dict[str, dict[tuple[str, int], int]] = defaultdict(dict)
    srcs: dict[str, set[int]] = defaultdict(set)
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if dim == 2 and gname.startswith(FACE_PREFIX):
            _, name, side, idx = gname.split("|")
            for tag in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag):
                faces[name][(side, int(idx))] = int(tag)
        elif dim == 1 and gname.startswith(SRC_PREFIX):
            _, name = gname.split("|")
            srcs[name].update(
                int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)
            )
    return {
        name: {"faces": f, "src_curves": srcs.get(name, set())}
        for name, f in faces.items()
    }


def _is_sweep_spec(spec: object) -> bool:
    """True for specs that pair with a sweep (not a wedge/cohort extrusion)."""
    return isinstance(spec, StructuredSweepResolutionSpec) and not isinstance(
        spec, StructuredExtrusionResolutionSpec
    )


def validate_sweep_pairing(
    discovered: dict[str, dict], resolution_specs: dict
) -> dict[str, StructuredSweepResolutionSpec]:
    """Hard-error on unpaired sweep groups or unpaired sweep specs.

    Args:
        discovered: Output of :func:`discover_sweeps`.
        resolution_specs: The ``resolution_specs`` mapping passed to
            ``mesh()``/``generate_mesh()``. Entries that are
            ``StructuredExtrusionResolutionSpec`` instances are ignored --
            those pair with cohorts, not sweeps.

    Returns:
        ``{sweep_name: StructuredSweepResolutionSpec}`` for every
        discovered sweep.

    Raises:
        SweepPairingError: A discovered ``__sweep`` group has no matching
            spec, or a sweep spec has no matching ``__sweep`` group.

    """
    specs_by_name: dict[str, StructuredSweepResolutionSpec] = {}
    for key, specs in (resolution_specs or {}).items():
        for spec in specs:
            if _is_sweep_spec(spec):
                specs_by_name[key] = spec

    missing = set(discovered) - set(specs_by_name)
    if missing:
        raise SweepPairingError(
            f"Sweep group(s) {sorted(missing)} present in CAD but no "
            "StructuredSweepResolutionSpec provided under that name."
        )
    orphans = set(specs_by_name) - set(discovered)
    if orphans:
        raise SweepPairingError(
            f"StructuredSweepResolutionSpec(s) {sorted(orphans)} have no "
            "matching __sweep group in the loaded CAD."
        )
    return specs_by_name


def strip_sweep_groups() -> None:
    """Remove ``__sweep*`` bookkeeping groups before the .msh is written."""
    to_remove: list[tuple[int, int]] = []
    names: list[str] = []
    for dim, gtag in gmsh.model.getPhysicalGroups():
        gname = gmsh.model.getPhysicalName(dim, gtag)
        if gname.startswith("__sweep"):
            to_remove.append((dim, gtag))
            names.append(gname)
    if not to_remove:
        return
    gmsh.model.removePhysicalGroups(to_remove)
    for gname in names:
        with contextlib.suppress(Exception):
            gmsh.model.removePhysicalName(gname)


def make_sweep_pre_2d_hook(
    discovered: dict[str, dict],
    specs_by_name: dict[str, StructuredSweepResolutionSpec],
    point_tolerance: float,
) -> Callable[[], None]:
    """Build the pre-``generate(2)`` hook: stamp every sweep, then strip bookkeeping.

    Args:
        discovered: Output of :func:`discover_sweeps`.
        specs_by_name: Output of :func:`validate_sweep_pairing`.
        point_tolerance: Geometric snap tolerance used by the stamping kernel.

    Returns:
        A zero-argument callable suitable for ``pre_2d_hook``.

    """

    def _hook() -> None:
        _stamp_all(discovered, specs_by_name, point_tolerance)
        strip_sweep_groups()

    return _hook


def _stamp_all(
    discovered: dict[str, dict],
    specs_by_name: dict[str, StructuredSweepResolutionSpec],
    point_tolerance: float,
) -> None:
    """Stamp every discovered sweep face with an exact tensor-product mesh.

    For each sweep a single source frame ``(origin, t_hat, n_hat)`` is
    built from the union of its source-curve endpoints; every band face
    is then hand-stamped with a ``(t, n)`` tensor grid (triangle-pairs or
    quads). ``Mesh.MeshOnlyEmpty`` is set so the subsequent ``generate``
    fills only the surrounding unstructured surfaces, conformally to the
    frozen boundary-curve nodes.

    Args:
        discovered: Output of :func:`discover_sweeps`.
        specs_by_name: Output of :func:`validate_sweep_pairing`.
        point_tolerance: Geometric snap tolerance for node matching.
    """
    logger.info("sweep stamping: %d sweep(s) discovered", len(discovered))
    gmsh.option.setNumber("Mesh.MeshOnlyEmpty", 1)

    # gmsh only preserves a hand-stamped surface through ``generate(2)``
    # (under ``MeshOnlyEmpty``) if the boundary-curve *point* nodes were
    # created by the mesher itself -- hand-adding them with ``addNodes(0,
    # ...)`` makes the mesher discard the whole surface. So we first pin
    # every sweep boundary curve to a 2-node transfinite line and run a
    # ``generate(1)`` to materialise the corner point nodes; ``_stamp_curve``
    # then clears each curve and re-lays the exact tangential grid on it,
    # reusing those mesher-owned point nodes.
    sweep_curves: set[int] = set()
    for groups in discovered.values():
        for ftag in groups["faces"].values():
            for _d, ct in gmsh.model.getBoundary(
                [(2, ftag)], oriented=False, recursive=False
            ):
                sweep_curves.add(abs(ct))
    for ctag in sweep_curves:
        gmsh.model.mesh.setTransfiniteCurve(ctag, 2)
    gmsh.model.mesh.generate(1)

    stamped_curves: set[int] = set()
    for name, groups in discovered.items():
        spec = specs_by_name[name]
        frame = _sweep_frame(groups["src_curves"])
        # Global tangential extent of the whole sweep (all faces of all
        # sides), computed once: distinguishes clip/attachment ends (at the
        # extent, auto-inserted) from interior BOP splits (must be grid
        # members).
        all_t_ends = []
        for ftag in groups["faces"].values():
            for _d, ct in gmsh.model.getBoundary(
                [(2, ftag)], oriented=False, recursive=False
            ):
                for _dd, pt in gmsh.model.getBoundary([(1, abs(ct))], oriented=False):
                    xy = np.array(gmsh.model.getValue(0, abs(pt), [])[:2])
                    all_t_ends.append(float((xy - frame[0]) @ frame[1]))
        groups["_extent"] = (min(all_t_ends), max(all_t_ends))
        for (side, _idx), face_tag in sorted(groups["faces"].items()):
            _stamp_face(
                face_tag, side, spec, frame, groups, stamped_curves, point_tolerance
            )
    # The outer ``generate(2)`` (which runs a 1D pass first) fills every
    # remaining empty curve and unstructured surface; stamped sweep faces
    # are non-empty and are skipped under ``MeshOnlyEmpty``.


def _sweep_frame(src_curve_tags):
    """Build ``(origin, t_hat, n_hat)`` from the source-curve endpoints.

    ``t_hat`` is the principal axis of the endpoint cloud; ``origin`` is
    the endpoint with minimum tangential projection; ``n_hat`` is ``t_hat``
    rotated +90 deg. The per-face sign of the normal is resolved in
    :func:`_stamp_face`.
    """
    pts = []
    for ctag in src_curve_tags:
        for _dim, ptag in gmsh.model.getBoundary([(1, ctag)], oriented=False):
            xyz = gmsh.model.getValue(0, abs(ptag), [])
            pts.append(np.array(xyz[:2]))
    pts = np.array(pts)
    # Signed endpoint-to-endpoint direction (not the bounding-box diagonal,
    # which y-reflects a negative-slope source) so the reconstructed frame
    # matches the real imprinted rectangle at any orientation.
    d = signed_axis(pts)
    t_hat = d / np.linalg.norm(d)
    proj = pts @ t_hat
    origin = pts[int(np.argmin(proj))]
    n_hat = np.array([-t_hat[1], t_hat[0]])
    return origin, t_hat, n_hat


def _cross2(a, b):
    """Scalar z-component of the 2D cross product ``a x b``."""
    return float(a[0]) * float(b[1]) - float(a[1]) * float(b[0])


def _round_key(xy, tol):
    """Snap ``xy`` to an integer key for coordinate-based node matching."""
    q = max(tol, 1e-12)
    return (round(float(xy[0]) / q), round(float(xy[1]) / q))


def _curve_endpoints(ctag):
    """Return the two boundary-point coordinates of curve ``ctag`` (xy)."""
    out = []
    for _d, ptag in gmsh.model.getBoundary([(1, ctag)], oriented=False):
        out.append(np.array(gmsh.model.getValue(0, abs(ptag), [])[:2]))
    return out


def _boundary_node_lookup(curves, tol):
    """Map ``_round_key`` -> node tag for every node on the given curves."""
    lookup = {}
    for ctag in curves:
        tags, coords, _ = gmsh.model.mesh.getNodes(1, ctag, includeBoundary=True)
        for i, t in enumerate(tags):
            lookup[_round_key(coords[3 * i : 3 * i + 2], tol)] = int(t)
    return lookup


def _grid_points_on_segment(grid_xy, e0, e1, tol):
    """Ordered grid points lying on segment ``e0``-``e1`` (endpoints incl.)."""
    e0 = np.asarray(e0)
    seg = np.asarray(e1) - e0
    length = np.linalg.norm(seg)
    d = seg / length
    flat = grid_xy.reshape(-1, 2)
    on = []
    for p in flat:
        v = p - e0
        s = float(v @ d)
        if -tol <= s <= length + tol and abs(_cross2(d, v)) <= 10 * tol:
            on.append((s, p))
    on.sort(key=lambda x: x[0])
    return [p for _s, p in on]


def _stamp_curve(ctag, pts, tol):
    """Re-lay curve ``ctag`` with ordered nodes + line elements at ``pts``.

    ``pts`` is the ordered list of grid points on the curve (endpoints
    included). The curve's provisional 2-node transfinite mesh (from the
    prelude ``generate(1)``) is cleared and replaced; the two endpoint
    *point* nodes -- owned by the mesher -- are reused so neighbouring
    entities stay conformal and ``generate(2)`` preserves the surface.

    Interior nodes are added with parametric coordinates: gmsh discards a
    hand-stamped surface if its 1D boundary nodes lack them.
    """
    if len(pts) < 2:
        return
    # Reuse the mesher-created endpoint point nodes.
    end_nodes = {}
    for _d, ptag in gmsh.model.getBoundary([(1, ctag)], oriented=False):
        xyz = gmsh.model.getValue(0, abs(ptag), [])
        tags, _c, _p = gmsh.model.mesh.getNodes(0, abs(ptag))
        end_nodes[_round_key(xyz[:2], tol)] = int(tags[0])
    # Parametric mapping (straight curve): param is affine in arc length.
    (pmin,), (pmax,) = gmsh.model.getParametrizationBounds(1, ctag)
    base = np.array(gmsh.model.getValue(1, ctag, [pmin])[:2])
    far = np.array(gmsh.model.getValue(1, ctag, [pmax])[:2])
    span = far - base
    length2 = float(span @ span)

    def _param(p):
        frac = float((np.asarray(p) - base) @ span) / length2 if length2 else 0.0
        return pmin + frac * (pmax - pmin)

    gmsh.model.mesh.clear([(1, ctag)])
    seq = []
    interior_tags, interior_coords, interior_params = [], [], []
    next_tag = gmsh.model.mesh.getMaxNodeTag() + 1
    for p in pts:
        key = _round_key(p, tol)
        if key in end_nodes:
            # Relocate the reused corner point node onto the snapped grid so
            # every entity sharing it (this curve, adjacent edges, neighbour
            # surfaces meshed later) sees the same nominal coordinate.
            gmsh.model.mesh.setNode(
                end_nodes[key], [float(p[0]), float(p[1]), 0.0], []
            )
            seq.append(end_nodes[key])
        else:
            seq.append(next_tag)
            interior_tags.append(next_tag)
            interior_coords += [float(p[0]), float(p[1]), 0.0]
            interior_params.append(_param(p))
            next_tag += 1
    if interior_tags:
        gmsh.model.mesh.addNodes(
            1, ctag, interior_tags, interior_coords, interior_params
        )
    conn = []
    for a, b in zip(seq[:-1], seq[1:]):
        conn += [int(a), int(b)]
    gmsh.model.mesh.addElementsByType(ctag, 1, [], conn)


def _tangential_coords(tangential, edge_ts, groups, face_tag, tol):
    """Tangential grid coordinates for one face.

    The global sweep extent ``[T0, T1]`` (cached on ``groups["_extent"]``)
    anchors the grid. ``edge_ts`` is the sorted list of every distinct
    boundary-curve endpoint tangential coordinate of the face: its extremes
    span the face, and any value strictly inside ``(T0, T1)`` marks a BOP
    split (e.g. a ridge corner imprinted on the seam) that must fall on a
    grid coordinate (else :class:`SweepSplitCoordinateError`). Endpoints at
    the global extent are clip/attachment ends and are auto-inserted.

    Args:
        tangential: ``None`` (rejected in 2D), a scalar target size, or an
            explicit array of offsets measured from ``T0``.
        edge_ts: Sorted distinct boundary-curve endpoint tangential coords.
        groups: The sweep's discovery dict (carries ``_extent``).
        face_tag: Surface tag, for error reporting.
        tol: Membership tolerance.

    Returns:
        Sorted list of tangential coordinates spanning ``[t0f, t1f]``.
    """
    T0, T1 = groups["_extent"]
    t0f, t1f = edge_ts[0], edge_ts[-1]

    if tangential is None:
        raise SweepPairingError(
            "StructuredSweepResolutionSpec.tangential is required in 2D."
        )
    if isinstance(tangential, (int, float)):
        h = float(tangential)
        k0 = int(np.floor((t0f - T0) / h + 0.5))
        candidates = [T0 + k * h for k in range(k0, int((T1 - T0) / h) + 2)]
        inside = [t for t in candidates if t0f - tol <= t <= t1f + tol]
    else:
        arr = [T0 + t for t in np.asarray(tangential, dtype=float)]
        inside = [t for t in arr if t0f - tol <= t <= t1f + tol]
    # Every boundary-curve endpoint must coincide with a grid coordinate.
    # Endpoints at the global sweep extent are clip/attachment ends and are
    # auto-inserted; endpoints interior to the sweep are BOP splits that the
    # caller's grid must align with, else the stamped seam is non-conformal.
    for end in edge_ts:
        if any(abs(t - end) <= tol for t in inside):
            continue
        if abs(end - T0) <= tol or abs(end - T1) <= tol:
            inside.append(end)  # clip/attachment end: auto-insert
        else:
            raise SweepSplitCoordinateError(end, face_tag)
    ts = sorted(set(round(t, 12) for t in inside))
    return [t for t in ts if t0f - tol <= t <= t1f + tol]


def _stamp_face(face_tag, side, spec, frame, groups, stamped_curves, tol):
    """Stamp one band face with a ``(t, n)`` tensor grid of nodes + cells.

    Freezes boundary-curve nodes (shared with neighbours), adds interior
    nodes, and emits triangle-pairs (default) or quads. Already-stamped
    shared curves are checked for seam agreement
    (:class:`SweepSeamMismatchError`).
    """
    origin, t_hat, n_hat = frame
    curves = sorted(
        {
            abs(t)
            for _d, t in gmsh.model.getBoundary(
                [(2, face_tag)], oriented=False, recursive=False
            )
        }
    )

    def _tn(xy):
        v = np.asarray(xy[:2]) - origin
        return float(v @ t_hat), float(v @ n_hat)

    all_tn = [_tn(p) for c in curves for p in _curve_endpoints(c)]
    edge_ts = sorted({round(t, 12) for t, _ in all_tn})
    t0f, t1f = edge_ts[0], edge_ts[-1]
    n_vals = sorted({round(n, 12) for _, n in all_tn})
    n_lo, n_hi = n_vals[0], n_vals[-1]
    src_here = [c for c in curves if c in groups["src_curves"]]
    if src_here:
        n_src = _tn(_curve_endpoints(src_here[0])[0])[1]
    else:
        n_src = n_lo if abs(n_lo) < abs(n_hi) else n_hi
    n_far = n_hi if abs(n_src - n_lo) <= abs(n_src - n_hi) else n_lo
    thickness = abs(n_far - n_src)
    sign = 1.0 if n_far > n_src else -1.0

    offsets = resolve_normal_offsets(
        spec.normal[side] if isinstance(spec.normal, dict) else spec.normal,
        thickness,
        tol,
    )
    ts = _tangential_coords(spec.tangential, edge_ts, groups, face_tag, tol)

    grid_xy = np.array(
        [
            [origin + t * t_hat + (n_src + sign * off) * n_hat for off in offsets]
            for t in ts
        ]
    )
    # Snap the grid onto the ``point_tolerance`` lattice: the CAD stage
    # perturbs polygon entities (~1e-5) for robust booleans, which would
    # otherwise leave the stamped nodes off their nominal coordinates.
    # Reused corner nodes are relocated to match (see ``_stamp_curve``).
    q = max(tol, 1e-12)
    grid_xy = np.round(grid_xy / q) * q

    # 1) freeze boundary curves (once each; shared with neighbours).
    for ctag in curves:
        e0, e1 = _curve_endpoints(ctag)
        pts = _grid_points_on_segment(grid_xy, e0, e1, tol)
        if ctag in stamped_curves:
            existing = _boundary_node_lookup([ctag], tol)
            if any(_round_key(p, tol) not in existing for p in pts):
                raise SweepSeamMismatchError(ctag)
            continue
        stamped_curves.add(ctag)
        _stamp_curve(ctag, pts, tol)

    # 2) interior nodes (grid points not already on a boundary curve).
    node_tag = gmsh.model.mesh.getMaxNodeTag() + 1
    tag_grid = np.zeros(grid_xy.shape[:2], dtype=np.int64)
    interior_tags, interior_coords = [], []
    boundary_lookup = _boundary_node_lookup(curves, tol)
    for it in range(len(ts)):
        for ik in range(len(offsets)):
            key = _round_key(grid_xy[it, ik], tol)
            if key in boundary_lookup:
                tag_grid[it, ik] = boundary_lookup[key]
            else:
                tag_grid[it, ik] = node_tag
                interior_tags.append(node_tag)
                interior_coords += [
                    float(grid_xy[it, ik, 0]),
                    float(grid_xy[it, ik, 1]),
                    0.0,
                ]
                node_tag += 1
    if interior_tags:
        gmsh.model.mesh.addNodes(2, face_tag, interior_tags, interior_coords)

    # 3) elements (CCW-enforced quads, split into triangle-pairs by default).
    tris, quads = [], []
    for it in range(len(ts) - 1):
        for ik in range(len(offsets) - 1):
            q = [
                int(tag_grid[it, ik]),
                int(tag_grid[it + 1, ik]),
                int(tag_grid[it + 1, ik + 1]),
                int(tag_grid[it, ik + 1]),
            ]
            a, b, c = grid_xy[it, ik], grid_xy[it + 1, ik], grid_xy[it + 1, ik + 1]
            if _cross2(b - a, c - b) < 0:
                q = q[::-1]
            if spec.element_type == "quad":
                quads += q
            else:
                tris += [q[0], q[1], q[2], q[0], q[2], q[3]]
    if quads:
        gmsh.model.mesh.addElementsByType(face_tag, 3, [], quads)
    if tris:
        gmsh.model.mesh.addElementsByType(face_tag, 2, [], tris)
