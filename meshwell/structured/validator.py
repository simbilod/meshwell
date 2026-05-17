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
        if not elem_types:
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


def validate_structured_mesh(
    plan: "StructuredPlan",  # noqa: ARG001
    mesh_plan: "StructuredMeshPlan",  # noqa: ARG001
    phantom_map: "PhantomMap",  # noqa: ARG001
    occ_entities: list[Any],  # noqa: ARG001
    vol_tags: list[int],  # noqa: ARG001
    *,
    tol: float | None = None,  # noqa: ARG001
    include_quality: bool = False,  # noqa: ARG001
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
    errors: list[Issue] = []
    warnings: list[Issue] = []
    # Checks are added one at a time in subsequent tasks.
    return ValidationResult(errors=tuple(errors), warnings=tuple(warnings))
