"""Post-fragment shape healing via OCC's ShapeFix_Shape.

``ShapeFix_Shape`` runs the ``ShapeFix_Wire/Face/Shell/Solid`` sub-fixers
and cleans up the small-geometry artifacts BOPAlgo sometimes leaves in
the post-fragment compound: tiny edges, wires with out-of-order or
degenerate segments, faces whose outer wire doesn't close under the
face surface's tolerance, shells with inconsistent face orientations,
and vertex/edge/face tolerance mismatches. These are exactly the inputs
that reach tetgen as "almost coincident" facets or segments piercing
triangulations.

Healing mints fresh TShapes for every repaired sub-shape. To preserve
physical-tag attribution, we drive healing off a compound that contains
every entity's owned shapes, then use the fixer's ``ShapeBuild_ReShape``
context to rewrite each entity's ``shapes`` list in place. Untouched
sub-shapes pass through the context unchanged; repaired ones come out
with their new TShape. ``OCCLabeledEntity.physical_name``, ``keep``, and
``mesh_order`` are untouched; the XAO writer re-enumerates sub-TShapes
at write time so tags follow the new identities automatically.

Call order inside ``CAD_OCC.process_entities``:

    instantiate -> _fragment_all -> canonicalize_topology -> heal_shapes
    -> validate_fragment

Healing is last among the TShape-mutating passes so the
``OCCGeometryCache``'s stored TShapes are irrelevant at this point --
downstream consumers (XAO writer) only care about the current state of
each entity's ``shapes`` list.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from OCP.BRep import BRep_Builder
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopoDS import TopoDS_Compound

if TYPE_CHECKING:
    from meshwell.cad_occ import OCCLabeledEntity

_logger = logging.getLogger(__name__)


def _build_compound(shapes) -> TopoDS_Compound:
    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for shape in shapes:
        if shape is not None and not shape.IsNull():
            builder.Add(compound, shape)
    return compound


def _perform_fix(
    shape, point_tolerance: float, max_tolerance: float
) -> ShapeFix_Shape | None:
    """Run ``ShapeFix_Shape`` on ``shape``; return the fixer on success.

    Returns ``None`` when the fixer raises (e.g. a degenerate vertex
    without a ``gp_Pnt`` trips ``BRep_Tool::Pnt`` on some inputs). The
    caller is expected to fall back to a narrower scope or skip healing
    for that shape.
    """
    fixer = ShapeFix_Shape()
    fixer.Init(shape)
    fixer.SetPrecision(point_tolerance)
    fixer.SetMaxTolerance(max_tolerance)
    try:
        fixer.Perform()
    except Exception as exc:
        _logger.warning("ShapeFix_Shape.Perform failed: %s", exc)
        return None
    return fixer


def heal_shapes(
    entities: list[OCCLabeledEntity],
    point_tolerance: float = 1e-3,
    max_tolerance_multiplier: float = 100.0,
) -> list[OCCLabeledEntity]:
    """Heal every entity's shapes in place via ``ShapeFix_Shape``.

    Healing is best-effort: if the fixer raises (for example on a
    degenerate vertex without a ``gp_Pnt``), we first retry
    per-entity so entities with clean geometry still benefit; if that
    also fails, the entity's shapes are left untouched and a warning
    is logged.

    Args:
        entities: Entities returned by ``CAD_OCC._fragment_all`` (and
            optionally post-``canonicalize_topology``).
        point_tolerance: Lower bound for ``ShapeFix_Shape.SetPrecision``.
            Usually matches the ``cad_occ`` value.
        max_tolerance_multiplier: Upper bound for the fixer is
            ``max_tolerance_multiplier * point_tolerance``. Larger values
            let ShapeFix repair wider gaps at the cost of potentially
            fusing features you didn't want fused. ``100`` is the OCC
            example default.

    Returns:
        The same ``entities`` list; each entity's ``shapes`` is updated
        in place with its repaired counterparts where healing succeeded.
    """
    if not entities:
        return entities

    max_tolerance = point_tolerance * max_tolerance_multiplier

    # First try: fix the full compound so shared boundaries get a
    # consistent repair across entities.
    compound = _build_compound(shape for ent in entities for shape in ent.shapes)
    fixer = _perform_fix(compound, point_tolerance, max_tolerance)
    if fixer is not None:
        context = fixer.Context()
        for ent in entities:
            ent.shapes = [
                applied
                for applied in (context.Apply(s) for s in ent.shapes)
                if applied is not None and not applied.IsNull()
            ]
        return entities

    # Fallback: per-entity healing. A single bad entity won't block
    # healing for the rest; an entity whose fixer also raises is
    # left unchanged.
    _logger.warning("heal_shapes: compound healing failed; falling back to per-entity")
    for ent in entities:
        ent_compound = _build_compound(ent.shapes)
        ent_fixer = _perform_fix(ent_compound, point_tolerance, max_tolerance)
        if ent_fixer is None:
            _logger.warning(
                "heal_shapes: skipping %s (per-entity healing raised)",
                ent.physical_name,
            )
            continue
        ctx = ent_fixer.Context()
        ent.shapes = [
            applied
            for applied in (ctx.Apply(s) for s in ent.shapes)
            if applied is not None and not applied.IsNull()
        ]
    return entities
