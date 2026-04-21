"""Post-fragment shape healing for OCC entities.

Primary tool: ``ShapeUpgrade_UnifySameDomain``. This is the same class
that ``BRepAlgoAPI_BuilderAlgo::SimplifyResult`` uses for its post-BOP
cleanup in OCCT itself -- it unifies adjacent coplanar faces and
collinear edges into single sub-shapes, resolves the split-face
artifacts BOPAlgo sometimes leaves behind, and preserves solid/shell
structure (so it won't silently fuse two meshwell entities).

Secondary, opt-in tool: ``ShapeFix_Shape`` for broader cleanup (tiny
edges, tolerance mismatches, disoriented wires). We leave it off by
default because on some inputs ``BRep_Tool::Pnt`` raises on degenerate
seam/apex vertices that ``ShapeFix_Shape`` tries to evaluate.

Tag preservation:
    ``ShapeUpgrade_UnifySameDomain`` exposes ``History()`` -- a
    ``BRepTools_History`` recording Modified/Generated/Removed for every
    input sub-shape. We walk each entity's owned shapes, query the
    history, and rewrite the shape list in place. ``physical_name``,
    ``keep``, ``mesh_order`` are untouched; the XAO writer re-enumerates
    sub-TShapes at write time so tags follow the new identities
    automatically.

Ordering inside ``CAD_OCC.process_entities``:

    instantiate -> _fragment_all -> canonicalize_topology -> heal_shapes
    -> validate_fragment

Healing runs last among the TShape-mutating passes so the
``OCCGeometryCache``'s stored TShapes are irrelevant at this point.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING

from OCP.BRep import BRep_Builder
from OCP.ShapeUpgrade import ShapeUpgrade_UnifySameDomain
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE, TopAbs_VERTEX
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS_Compound
from OCP.TopTools import TopTools_ShapeMapHasher

if TYPE_CHECKING:
    from meshwell.cad_occ import OCCLabeledEntity

_logger = logging.getLogger(__name__)
_HASHER = TopTools_ShapeMapHasher()


def _cross_entity_shared_shapes(entities) -> list:
    """Return TopoDS_Shapes whose TShape appears in >1 entity.

    Shared sub-shapes (interface edges/faces between neighbouring
    entities, shared vertices on meeting corners) carry the meshwell
    interface tagging -- if ``UnifySameDomain`` replaces them with a
    fresh TShape, the XAO writer can't detect the interface anymore
    and groups like ``A___B`` disappear. We feed this list into the
    unifier's ``KeepShape`` to pin them across the Build.
    """
    owners: dict[int, set[int]] = defaultdict(set)
    shape_by_hash: dict[int, object] = {}
    for idx, ent in enumerate(entities):
        seen: set[int] = set()
        for top_shape in ent.shapes:
            if top_shape is None or top_shape.IsNull():
                continue
            for kind in (TopAbs_VERTEX, TopAbs_EDGE, TopAbs_FACE):
                exp = TopExp_Explorer(top_shape, kind)
                while exp.More():
                    sub = exp.Current()
                    h = _HASHER(sub)
                    if h not in seen:
                        seen.add(h)
                        owners[h].add(idx)
                        shape_by_hash.setdefault(h, sub)
                    exp.Next()
    return [shape_by_hash[h] for h, idxs in owners.items() if len(idxs) > 1]


def _build_compound(shapes) -> TopoDS_Compound:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        if shape is not None and not shape.IsNull():
            builder.Add(compound, shape)
    return compound


def _rewrite_through_history(entities, history) -> None:
    """Rewrite each entity's ``shapes`` via a ``BRepTools_History`` query.

    For every owned shape:
      - if ``history.IsRemoved(s)``, drop it
      - else if ``history.Modified(s)`` is non-empty, replace with that list
      - else keep the original (unchanged by the pass)
    """
    for ent in entities:
        rewritten: list = []
        for shape in ent.shapes:
            if history.IsRemoved(shape):
                continue
            modified = history.Modified(shape)
            try:
                modified_list = list(modified)
            except TypeError:
                # OCP exposes NCollection_List with an iterator protocol,
                # but some builds bind it without __iter__; fall back to
                # treating the return as "no modifications found".
                modified_list = []
            if modified_list:
                rewritten.extend(
                    s for s in modified_list if s is not None and not s.IsNull()
                )
            elif shape is not None and not shape.IsNull():
                rewritten.append(shape)
        ent.shapes = rewritten


def heal_shapes(
    entities: list[OCCLabeledEntity],
    point_tolerance: float = 1e-3,
    angular_tolerance: float = 1e-6,
    unify_edges: bool = True,
    unify_faces: bool = True,
    concat_bsplines: bool = False,
    allow_internal_edges: bool = False,
    run_shape_fix: bool = False,
    shape_fix_max_tolerance_multiplier: float = 100.0,
) -> list[OCCLabeledEntity]:
    """Heal every entity's shapes in place.

    Args:
        entities: Entities returned by ``CAD_OCC._fragment_all`` (and
            optionally post-``canonicalize_topology``).
        point_tolerance: Linear tolerance for the unifier -- the
            ``SimplifyResult`` pattern in OCCT passes the BOP fuzzy value
            here, so matching ``cad_occ``'s ``point_tolerance`` is the
            conservative choice.
        angular_tolerance: Maximum angle (radians) between adjacent face
            normals or edge tangents for them to be considered
            same-domain. Default ``1e-6``.
        unify_edges: Merge collinear edges into a single edge.
        unify_faces: Merge coplanar/cosurface faces into a single face.
        concat_bsplines: Concatenate C1-continuous BSpline / Bezier edges.
        allow_internal_edges: Permit creating INTERNAL edges inside merged
            faces for non-manifold inputs. Default off.
        run_shape_fix: If ``True``, chain a ``ShapeFix_Shape`` pass after
            the unifier to clean up residual tiny edges / tolerance
            mismatches. Off by default because some inputs raise
            ``BRep_Tool::Pnt`` on degenerate seam vertices.
        shape_fix_max_tolerance_multiplier: Upper bound for the optional
            ``ShapeFix_Shape`` is this multiplier times ``point_tolerance``.

    Returns:
        The same ``entities`` list; each entity's ``shapes`` is updated
        in place with its unified / repaired counterparts where healing
        succeeded. On failure the entity's shapes are left unchanged and
        a warning is logged.
    """
    if not entities:
        return entities

    # Heal the full compound so BOPAlgo split-face artifacts at shared
    # boundaries get a consistent repair across entities, and pin every
    # cross-entity shared sub-shape so the unifier doesn't re-TShape
    # them -- that would silently drop interface physical groups like
    # ``A___B`` from the XAO writer's output.
    compound = _build_compound(shape for ent in entities for shape in ent.shapes)
    unifier = ShapeUpgrade_UnifySameDomain(
        compound, unify_edges, unify_faces, concat_bsplines
    )
    unifier.SetLinearTolerance(point_tolerance)
    unifier.SetAngularTolerance(angular_tolerance)
    unifier.SetSafeInputMode(False)
    unifier.AllowInternalEdges(allow_internal_edges)
    for shared in _cross_entity_shared_shapes(entities):
        unifier.KeepShape(shared)

    try:
        unifier.Build()
    except Exception as exc:
        _logger.warning("heal_shapes: UnifySameDomain failed: %s", exc)
        if run_shape_fix:
            _shape_fix_fallback(
                entities, point_tolerance, shape_fix_max_tolerance_multiplier
            )
        return entities

    _rewrite_through_history(entities, unifier.History())

    if run_shape_fix:
        _shape_fix_fallback(
            entities, point_tolerance, shape_fix_max_tolerance_multiplier
        )

    return entities


def _shape_fix_fallback(
    entities,
    point_tolerance: float,
    max_tolerance_multiplier: float,
) -> None:
    """Optional ``ShapeFix_Shape`` cleanup pass after the unifier.

    Guarded against ``BRep_Tool::Pnt`` crashes on degenerate vertices:
    on any exception we fall back to per-entity fixing, and an entity
    whose fixer also raises is left unchanged with a warning logged.
    """
    from OCP.ShapeFix import ShapeFix_Shape

    max_tolerance = point_tolerance * max_tolerance_multiplier

    def _fix(shape):
        fixer = ShapeFix_Shape()
        fixer.Init(shape)
        fixer.SetPrecision(point_tolerance)
        fixer.SetMaxTolerance(max_tolerance)
        try:
            fixer.Perform()
        except Exception as exc:
            _logger.warning("heal_shapes: ShapeFix_Shape failed: %s", exc)
            return None
        return fixer

    compound = _build_compound(shape for ent in entities for shape in ent.shapes)
    fixer = _fix(compound)
    if fixer is not None:
        ctx = fixer.Context()
        for ent in entities:
            ent.shapes = [
                applied
                for applied in (ctx.Apply(s) for s in ent.shapes)
                if applied is not None and not applied.IsNull()
            ]
        return

    _logger.warning(
        "heal_shapes: ShapeFix_Shape compound failed; falling back per-entity"
    )
    for ent in entities:
        ent_compound = _build_compound(ent.shapes)
        ent_fixer = _fix(ent_compound)
        if ent_fixer is None:
            _logger.warning(
                "heal_shapes: skipping %s (per-entity ShapeFix raised)",
                ent.physical_name,
            )
            continue
        ctx = ent_fixer.Context()
        ent.shapes = [
            applied
            for applied in (ctx.Apply(s) for s in ent.shapes)
            if applied is not None and not applied.IsNull()
        ]
