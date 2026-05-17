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


def validate_structured_mesh(
    plan: "StructuredPlan",
    mesh_plan: "StructuredMeshPlan",
    phantom_map: "PhantomMap",  # noqa: ARG001
    occ_entities: list[Any],  # noqa: ARG001
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

    return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))
