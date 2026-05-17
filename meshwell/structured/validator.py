"""Conformality validator for structured-polyprism meshes.

Runs in the live gmsh session immediately after
``apply_structured_mesh``. Reports topological and geometric
conformality failures between the structured wedge/hex slabs and the
surrounding tet regions.

Public API:

- :class:`Issue` — one validation finding (severity + check name + message + entities).
- :class:`ValidationResult` — collected errors + warnings + report formatter.
- :func:`validate_structured_mesh` — entry point. See its docstring.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any, Literal

from meshwell.structured.builder import _map_phantom_laterals_to_gmsh

if TYPE_CHECKING:
    from meshwell.structured.spec import (
        PhantomMap,
        StructuredMeshPlan,
        StructuredPlan,
    )


Severity = Literal["error", "warning"]


@dataclass(frozen=True)
class Issue:
    """One validation finding.

    ``entities`` is a tuple of (kind, tag) or (kind, key) tuples that
    localize the issue. Examples:
    - ``(("face", 42),)`` — a single gmsh face tag.
    - ``(("node", 17), ("node", 19))`` — a pair of node tags.
    - ``(("slab_piece", (1, 0)),)`` — a (slab_index, piece_index) key.
    """

    severity: Severity
    check: str
    message: str
    entities: tuple = ()


@dataclass(frozen=True)
class ValidationResult:
    """Collected validator output.

    ``__bool__`` is True iff ``errors`` is empty (warnings are not
    failures). Use ``assert result, result.format_report()`` in tests.
    """

    errors: tuple[Issue, ...]
    warnings: tuple[Issue, ...]

    def __bool__(self) -> bool:
        """Return True iff no errors are present (warnings are not failures)."""
        return not self.errors

    def format_report(self) -> str:
        """Return a human-readable multi-line report grouped by check name."""
        if not self.errors and not self.warnings:
            return "Structured-mesh validation: no issues."
        lines: list[str] = ["Structured-mesh validation report"]
        if self.errors:
            lines.append("")
            lines.append(f"ERRORS ({len(self.errors)})")
            for check, group in _group_by_check(self.errors).items():
                lines.append(f"  [{check}]")
                for issue in group:
                    lines.append(f"    - {issue.message}")
                    if issue.entities:
                        lines.append(f"      entities: {issue.entities}")
        if self.warnings:
            lines.append("")
            lines.append(f"WARNINGS ({len(self.warnings)})")
            for check, group in _group_by_check(self.warnings).items():
                lines.append(f"  [{check}]")
                for issue in group:
                    lines.append(f"    - {issue.message}")
                    if issue.entities:
                        lines.append(f"      entities: {issue.entities}")
        return "\n".join(lines)


def _group_by_check(issues: tuple[Issue, ...]) -> dict[str, list[Issue]]:
    out: dict[str, list[Issue]] = defaultdict(list)
    for issue in issues:
        out[issue.check].append(issue)
    return dict(out)


_DEFAULT_TOL_FALLBACK = 1e-9
_QUALITY_WARNING_THRESHOLD = 1e-3


def _resolve_tol(tol: float | None) -> float:
    """Pick a coordinate tolerance for geometric checks.

    If ``tol`` is given, returns it unchanged. Otherwise derives from
    the minimum edge length of the current gmsh mesh:
    ``min_edge_length * 1e-6``. Falls back to ``1e-9`` when no elements
    exist in the model yet.
    """
    if tol is not None:
        return float(tol)

    import gmsh

    try:
        elem_types, elem_tags, _ = gmsh.model.mesh.getElements(3)
        if len(elem_types) == 0:
            return _DEFAULT_TOL_FALLBACK
        # Concatenate all 3D element tags across types.
        all_tags = [t for tags in elem_tags for t in tags]
        if not all_tags:
            return _DEFAULT_TOL_FALLBACK
        min_edges = gmsh.model.mesh.getElementQualities(all_tags, "minEdge")
        if len(min_edges) == 0:
            return _DEFAULT_TOL_FALLBACK
        min_edge = float(min(min_edges))
        if min_edge <= 0:
            return _DEFAULT_TOL_FALLBACK
        return min_edge * 1e-6
    except (AttributeError, RuntimeError, ValueError):
        return _DEFAULT_TOL_FALLBACK


def _check_near_duplicate_nodes(tol: float) -> tuple[list[Issue], list[Issue]]:
    """Detect exact + near-duplicate node coordinates.

    Returns ``(errors, warnings)``. Exact duplicates are reported as
    errors (indicates ``removeDuplicateNodes`` was skipped or too tight).
    Near-duplicates within ``tol`` but with non-zero offset are warnings.
    """
    import gmsh
    import numpy as np

    node_tags_arr, node_coords_flat, _ = gmsh.model.mesh.getNodes()
    if len(node_tags_arr) == 0:
        return [], []

    node_tags = np.asarray(node_tags_arr, dtype=np.int64)
    coords = np.asarray(node_coords_flat, dtype=float).reshape(-1, 3)
    n = coords.shape[0]
    if n < 2:
        return [], []

    # Spatial-hash bin = tol. Pairs within sqrt(3)*tol may straddle bins,
    # so check the 27 neighbouring bins around each.
    bin_size = max(tol, 1e-15)
    bins = np.floor(coords / bin_size).astype(np.int64)

    bucket: dict[tuple[int, int, int], list[int]] = {}
    for i in range(n):
        key = (int(bins[i, 0]), int(bins[i, 1]), int(bins[i, 2]))
        bucket.setdefault(key, []).append(i)

    errors: list[Issue] = []
    warnings: list[Issue] = []

    for i in range(n):
        bi = (int(bins[i, 0]), int(bins[i, 1]), int(bins[i, 2]))
        for dx, dy, dz in product((-1, 0, 1), repeat=3):
            key = (bi[0] + dx, bi[1] + dy, bi[2] + dz)
            nbrs = bucket.get(key)
            if not nbrs:
                continue
            for j in nbrs:
                if j <= i:
                    continue
                d2 = float(np.sum((coords[i] - coords[j]) ** 2))
                if d2 > tol * tol:
                    continue
                pair = (int(node_tags[i]), int(node_tags[j]))
                if d2 == 0.0:
                    errors.append(
                        Issue(
                            severity="error",
                            check="near_duplicate_nodes",
                            message=(
                                f"Exact duplicate node coords at "
                                f"({coords[i, 0]:.6g}, {coords[i, 1]:.6g}, "
                                f"{coords[i, 2]:.6g}); removeDuplicateNodes "
                                f"may have been skipped."
                            ),
                            entities=(("node", pair[0]), ("node", pair[1])),
                        )
                    )
                else:
                    warnings.append(
                        Issue(
                            severity="warning",
                            check="near_duplicate_nodes",
                            message=(
                                f"Near-duplicate node pair near "
                                f"({coords[i, 0]:.6g}, {coords[i, 1]:.6g}, "
                                f"{coords[i, 2]:.6g}): distance "
                                f"{d2 ** 0.5:.3e} < tol {tol:.3e}."
                            ),
                            entities=(("node", pair[0]), ("node", pair[1])),
                        )
                    )

    return errors, warnings


def _check_element_quality() -> tuple[list[Issue], list[Issue]]:
    """Flag negative-Jacobian and near-degenerate elements.

    Uses ``gmsh.model.mesh.getElementQualities(..., "minSICN")``. The
    SICN (Scaled Inverse Condition Number) metric is in [0, 1] for
    valid elements and negative for tangled elements.

    - minSICN <= 0                              → error (tangled / inverted element).
    - minSICN < ``_QUALITY_WARNING_THRESHOLD``  → warning (near-degenerate).
    """
    import gmsh

    errors: list[Issue] = []
    warnings: list[Issue] = []

    elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3)
    if len(elem_types) == 0:
        return errors, warnings

    tag_to_type: dict[int, int] = {}
    for et, tags in zip(elem_types, elem_tags_per_type):
        for t in tags:
            tag_to_type[int(t)] = int(et)

    all_tags = list(tag_to_type.keys())
    if not all_tags:
        return errors, warnings

    sicn = gmsh.model.mesh.getElementQualities(all_tags, "minSICN")
    for tag, q in zip(all_tags, sicn):
        et = tag_to_type[int(tag)]
        if q <= 0:
            errors.append(
                Issue(
                    severity="error",
                    check="element_quality",
                    message=f"Element {tag} (type {et}) has minSICN={q:.3e} (tangled/inverted).",
                    entities=(("element", int(tag)),),
                )
            )
        elif q < _QUALITY_WARNING_THRESHOLD:
            warnings.append(
                Issue(
                    severity="warning",
                    check="element_quality",
                    message=f"Element {tag} (type {et}) has minSICN={q:.3e} (near-degenerate).",
                    entities=(("element", int(tag)),),
                )
            )
    return errors, warnings


def _check_plan_mesh_consistency(
    plan: "StructuredPlan",
    mesh_plan: "StructuredMeshPlan",
    vol_tags: list[int],
) -> list[Issue]:
    """Each plan piece has a vol_tag; each vol_tag holds n_layers x N elements."""
    import gmsh

    issues: list[Issue] = []

    expected_pieces = sum(len(s.face_partition) for s in plan.slabs)
    if len(vol_tags) != expected_pieces:
        issues.append(
            Issue(
                severity="error",
                check="plan_mesh_consistency",
                message=(
                    f"vol_tag count {len(vol_tags)} does not match expected "
                    f"piece count {expected_pieces} from the plan."
                ),
                entities=(("vol_tag_list", tuple(vol_tags)),),
            )
        )
        return issues

    # vol_tags is flat in (slab, piece) lexicographic order, matching the
    # iteration order in apply_structured_mesh. A future refactor that
    # reorders vol_tags must update this walk to keep slab/piece
    # attribution correct in error messages.
    cursor = 0
    for slab_idx, slab in enumerate(plan.slabs):
        n_layers = int(mesh_plan.n_layers[slab_idx])
        for piece_idx in range(len(slab.face_partition)):
            vol = vol_tags[cursor]
            cursor += 1

            _elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, vol)
            total = sum(len(t) for t in elem_tags_per_type)

            if total == 0:
                issues.append(
                    Issue(
                        severity="error",
                        check="plan_mesh_consistency",
                        message=(
                            f"Slab {slab_idx} piece {piece_idx} (vol_tag {vol}) "
                            f"has zero elements; expected multiple of n_layers={n_layers}."
                        ),
                        entities=(
                            ("slab_piece", (slab_idx, piece_idx)),
                            ("vol_tag", vol),
                        ),
                    )
                )
                continue

            if total % n_layers != 0:
                issues.append(
                    Issue(
                        severity="error",
                        check="plan_mesh_consistency",
                        message=(
                            f"Slab {slab_idx} piece {piece_idx} (vol_tag {vol}) "
                            f"has {total} elements, not a multiple of n_layers={n_layers}."
                        ),
                        entities=(
                            ("slab_piece", (slab_idx, piece_idx)),
                            ("vol_tag", vol),
                        ),
                    )
                )

    return issues


def _check_internal_seams_unmeshed(
    phantom_map: "PhantomMap",
    occ_entities: list[Any],
) -> list[Issue]:
    """Faces shared between two pieces of the same slab must carry no 2D elements.

    Mirrors detection logic in meshwell.structured.builder.apply_structured_mesh.
    Reports any face that the builder should have cleared but didn't.

    Two code paths for resolving phantom_map.output_laterals values:
    (a) Tests pass int gmsh tags directly in output_laterals values.
    (b) Real pipeline passes TopoDS_Face values; use the lateral map.
    """
    import gmsh

    from meshwell.structured.spec import LateralKey  # local import, avoids cycle

    issues: list[Issue] = []

    # Resolve lateral keys → gmsh face tags once.
    lateral_to_gmsh: dict[Any, list[int]] = {}
    # Classify all values across all keys: all int (test path) vs all non-int
    # (real pipeline). A mix indicates caller bug; refuse silently degrading.
    all_values = [v for vals in phantom_map.output_laterals.values() for v in vals]
    int_count = sum(1 for v in all_values if isinstance(v, int))
    if all_values and int_count not in (0, len(all_values)):
        raise ValueError(
            "phantom_map.output_laterals contains a mix of int gmsh tags "
            "and non-int values; expected all-int (test path) or all-OCC "
            "(real pipeline). Mixed maps would silently miss seam checks."
        )

    if int_count == len(all_values) and all_values:
        lateral_to_gmsh = {
            k: [int(v) for v in vals] for k, vals in phantom_map.output_laterals.items()
        }
    elif phantom_map.output_laterals:
        lateral_to_gmsh = _map_phantom_laterals_to_gmsh(phantom_map, occ_entities)

    face_tag_to_keys: dict[int, list[LateralKey]] = {}
    for key, tags in lateral_to_gmsh.items():
        for tag in tags:
            face_tag_to_keys.setdefault(int(tag), []).append(key)

    for face_tag, keys in face_tag_to_keys.items():
        if len(keys) < 2:
            continue
        slabs = {k.slab_index for k in keys}
        pieces = {k.piece_index for k in keys}
        if len(slabs) != 1 or len(pieces) < 2:
            continue
        _elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(2, face_tag)
        total_2d = sum(len(t) for t in elem_tags_per_type)
        if total_2d > 0:
            issues.append(
                Issue(
                    severity="error",
                    check="internal_seam_unmeshed",
                    message=(
                        f"Internal seam face {face_tag} carries {total_2d} "
                        f"2D elements; should have been cleared by the builder."
                    ),
                    entities=(
                        ("face", face_tag),
                        (
                            "seam_keys",
                            tuple(
                                sorted(
                                    (k.slab_index, k.piece_index, k.outer_edge_index)
                                    for k in keys
                                )
                            ),
                        ),
                    ),
                )
            )

    return issues


def _check_watertight(vol_tags: list[int]) -> list[Issue]:
    """For each volume, every face appears in 1 or 2 elements of that volume.

    Scans triangular faces (face_type=3) and quad faces (face_type=4) via
    ``gmsh.model.mesh.getElementFaceNodes``. That API returns an empty array
    (not an exception) when an element type has no faces of the requested
    kind, so no try/except is needed.

    A face shared by 3+ elements within the same volume indicates geometric
    overlap — this would normally be impossible in a valid structured mesh but
    can occur if vol_tags are mis-assigned or elements are accidentally
    duplicated.
    """
    import gmsh
    import numpy as np

    issues: list[Issue] = []

    for vol in vol_tags:
        elem_types, _elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, vol)
        if len(elem_types) == 0:
            continue

        face_count: dict[frozenset[int], int] = {}
        for et in elem_types:
            for face_type in (3, 4):
                face_nodes = gmsh.model.mesh.getElementFaceNodes(et, face_type, vol)
                if len(face_nodes) == 0:
                    continue
                arr = np.asarray(face_nodes, dtype=np.int64).reshape(-1, face_type)
                for row in arr:
                    key = frozenset(int(x) for x in row)
                    face_count[key] = face_count.get(key, 0) + 1

        for face_key, count in face_count.items():
            if count > 2:
                issues.append(
                    Issue(
                        severity="error",
                        check="watertight",
                        message=(
                            f"vol_tag {vol}: face shared by {count} elements "
                            f"(should be 1 boundary or 2 internal)."
                        ),
                        entities=(
                            ("vol_tag", vol),
                            ("face_nodes", tuple(sorted(face_key))),
                        ),
                    )
                )

    return issues


# Gmsh element type codes used here.
_ELEM_TET = 4  # 4-node tet
_ELEM_HEX = 5  # 8-node hex
_ELEM_PRISM = 6  # 6-node prism
_STRUCTURED_TYPES = (_ELEM_HEX, _ELEM_PRISM)


def _check_prism_tet_interface(vol_tags: list[int]) -> list[Issue]:
    """Check that structured/tet face matching is conformal.

    Faces shared between a structured volume (wedge/hex) and a tet
    volume must be matched 1:1 (triangle face) or 2 triangles on the
    tet side spanning the same 4 nodes (quad face).
    """
    import gmsh
    import numpy as np

    if not vol_tags:
        return []

    issues: list[Issue] = []

    # Classify each volume by its element types.
    vol_kind: dict[int, str] = {}
    for vol in vol_tags:
        elem_types, _, _ = gmsh.model.mesh.getElements(3, vol)
        types = {int(t) for t in elem_types}
        if len(types) == 0:
            continue
        if types & set(_STRUCTURED_TYPES):
            vol_kind[vol] = "structured"
        elif _ELEM_TET in types:
            vol_kind[vol] = "tet"
        else:
            vol_kind[vol] = "other"

    # Collect all faces per volume: structured (tri + quad) vs. tet (tri).
    # Values are occurrence counts; the downstream checks only need to know
    # whether a face appears once (boundary) or more (internal/shared).
    tri_faces_by_vol: dict[int, dict[frozenset[int], int]] = {}
    quad_faces_by_vol: dict[int, dict[frozenset[int], int]] = {}

    for vol in vol_kind:
        elem_types, elem_tags_per_type, _ = gmsh.model.mesh.getElements(3, vol)
        tri_map: dict[frozenset[int], int] = {}
        quad_map: dict[frozenset[int], int] = {}
        for et, _ in zip(elem_types, elem_tags_per_type):
            tri = gmsh.model.mesh.getElementFaceNodes(int(et), 3, vol)
            tri_arr = np.asarray(tri, dtype=np.int64).reshape(-1, 3)
            for row in tri_arr:
                key = frozenset(int(x) for x in row)
                tri_map[key] = tri_map.get(key, 0) + 1
            quad = gmsh.model.mesh.getElementFaceNodes(int(et), 4, vol)
            quad_arr = np.asarray(quad, dtype=np.int64).reshape(-1, 4)
            for row in quad_arr:
                key = frozenset(int(x) for x in row)
                quad_map[key] = quad_map.get(key, 0) + 1
        tri_faces_by_vol[vol] = tri_map
        quad_faces_by_vol[vol] = quad_map

    # For each structured volume, find faces that bound it (count == 1 in
    # its own face map) -- those that don't match a counterpart on a tet
    # volume are non-conformal.
    structured_vols = [v for v, k in vol_kind.items() if k == "structured"]
    tet_vols = [v for v, k in vol_kind.items() if k == "tet"]

    # tet_node_set is the union of all nodes referenced by triangular
    # faces of each tet volume. We use it as the "is on the prism/tet
    # interface" gate. Assumes every tet volume that participates in
    # the interface has boundary triangles — true for any closed tet
    # region adjacent to a structured slab.
    tet_node_set: dict[int, set[int]] = {}
    for tv in tet_vols:
        nodes_used: set[int] = set()
        for fk in tri_faces_by_vol[tv]:
            nodes_used.update(fk)
        tet_node_set[tv] = nodes_used

    # Inverted index: for each tet vol, map each node tag to the set of
    # triangle face keys that contain it. Lets the per-quad covering loop
    # skip tri faces that don't touch any of the quad's 4 nodes.
    tet_node_to_tris: dict[int, dict[int, set[frozenset[int]]]] = {}
    for tv in tet_vols:
        per_node: dict[int, set[frozenset[int]]] = {}
        for tri_key in tri_faces_by_vol[tv]:
            for n in tri_key:
                per_node.setdefault(n, set()).add(tri_key)
        tet_node_to_tris[tv] = per_node

    for sv in structured_vols:
        # If no tet volumes exist, prism/tet conformality checks are vacuous.
        if not tet_vols:
            continue

        # Triangle faces.
        for face_key, count in tri_faces_by_vol[sv].items():
            if count != 1:
                continue  # Internal to the structured volume.
            matched = any(face_key in tri_faces_by_vol[tv] for tv in tet_vols)
            if not matched:
                referenced = any(face_key.issubset(tet_node_set[tv]) for tv in tet_vols)
                if referenced:
                    issues.append(
                        Issue(
                            severity="error",
                            check="prism_tet_interface",
                            message=(
                                f"Structured vol {sv} triangle face has no exact "
                                f"match in any tet volume (possible T-junction)."
                            ),
                            entities=(
                                ("vol_tag", sv),
                                ("face_nodes", tuple(sorted(face_key))),
                            ),
                        )
                    )

        # Quad faces.
        for face_key, count in quad_faces_by_vol[sv].items():
            if count != 1:
                continue
            covering_tris: list[frozenset[int]] = []
            for tv in tet_vols:
                # Only scan tri keys that share at least one node with the quad.
                candidates: set[frozenset[int]] = set()
                for n in face_key:
                    candidates.update(tet_node_to_tris[tv].get(n, set()))
                covering_tris.extend(
                    tri_key for tri_key in candidates if tri_key.issubset(face_key)
                )
            if len(covering_tris) == 0:
                touches_tet = any(
                    face_key.issubset(tet_node_set[tv]) for tv in tet_vols
                )
                if touches_tet:
                    issues.append(
                        Issue(
                            severity="error",
                            check="prism_tet_interface",
                            message=(
                                f"Structured vol {sv} quad face has 0 covering tet "
                                f"triangles (T-junction or missing element)."
                            ),
                            entities=(
                                ("vol_tag", sv),
                                ("face_nodes", tuple(sorted(face_key))),
                            ),
                        )
                    )
            elif len(covering_tris) != 2:
                issues.append(
                    Issue(
                        severity="error",
                        check="prism_tet_interface",
                        message=(
                            f"Structured vol {sv} quad face has "
                            f"{len(covering_tris)} covering tet triangles "
                            f"(expected 2 for clean quad split)."
                        ),
                        entities=(
                            ("vol_tag", sv),
                            ("face_nodes", tuple(sorted(face_key))),
                        ),
                    )
                )
            else:
                union = covering_tris[0] | covering_tris[1]
                if union != face_key:
                    issues.append(
                        Issue(
                            severity="error",
                            check="prism_tet_interface",
                            message=(
                                f"Structured vol {sv} quad face: two covering tet "
                                f"triangles don't span the full quad (extra Steiner node)."
                            ),
                            entities=(
                                ("vol_tag", sv),
                                ("face_nodes", tuple(sorted(face_key))),
                                (
                                    "tet_tris",
                                    tuple(tuple(sorted(t)) for t in covering_tris),
                                ),
                            ),
                        )
                    )

    return issues


def validate_structured_mesh(
    plan: "StructuredPlan",
    mesh_plan: "StructuredMeshPlan",
    phantom_map: "PhantomMap",
    occ_entities: list[Any],
    vol_tags: list[int],
    *,
    tol: float | None = None,
    include_quality: bool = False,
) -> ValidationResult:
    """Validate the live-gmsh-session mesh against the builder's plan.

    Must be called while a gmsh model is initialized and meshed (i.e.
    after ``meshwell.structured.builder.apply_structured_mesh``). Reads
    from gmsh's in-memory model; writes nothing.

    Args:
        plan: the StructuredPlan from the planner.
        mesh_plan: the StructuredMeshPlan from ``resolve_mesh_plan``.
        phantom_map: the PhantomMap built by Phase-2 phantom stage.
        occ_entities: the OCC entity list used by the builder
            (needed for face/edge gmsh-tag lookup via the existing
            ``_map_phantom_*`` helpers).
        vol_tags: the list of per-piece 3D entity tags returned by
            ``apply_structured_mesh``.
        tol: absolute coordinate tolerance for geometric checks. If
            None, derived from the minimum edge length reported by
            ``gmsh.model.mesh.getElementQualities(..., "minEdge")``.
        include_quality: if True, additionally run a quality check via
            ``gmsh.model.mesh.getElementQualities``. Off by default;
            the validator's focus is conformality, not quality.

    Returns:
        ValidationResult with collected errors / warnings. ``bool(result)``
        is True iff ``errors`` is empty.

    Raises:
        RuntimeError: if no gmsh model is initialized.
    """
    resolved_tol = _resolve_tol(tol)

    errors: list[Issue] = []
    warnings: list[Issue] = []

    dup_errors, dup_warnings = _check_near_duplicate_nodes(resolved_tol)
    errors.extend(dup_errors)
    warnings.extend(dup_warnings)

    if include_quality:
        q_errors, q_warnings = _check_element_quality()
        errors.extend(q_errors)
        warnings.extend(q_warnings)

    errors.extend(_check_plan_mesh_consistency(plan, mesh_plan, vol_tags))
    errors.extend(_check_watertight(vol_tags))
    errors.extend(_check_internal_seams_unmeshed(phantom_map, occ_entities))
    errors.extend(_check_prism_tet_interface(vol_tags))

    return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))
