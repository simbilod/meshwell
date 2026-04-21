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

from typing import TYPE_CHECKING

from OCP.BRep import BRep_Builder
from OCP.ShapeFix import ShapeFix_Shape
from OCP.TopoDS import TopoDS_Compound

if TYPE_CHECKING:
    from meshwell.cad_occ import OCCLabeledEntity


def heal_shapes(
    entities: list[OCCLabeledEntity],
    point_tolerance: float = 1e-3,
    max_tolerance_multiplier: float = 100.0,
) -> list[OCCLabeledEntity]:
    """Heal every entity's shapes in place via ``ShapeFix_Shape``.

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
        in place with its repaired counterparts. Entities whose shapes
        are entirely removed by the fixer end up with an empty
        ``shapes`` list -- the XAO writer will emit no geometry for
        them but the physical-name record survives.
    """
    if not entities:
        return entities

    builder = BRep_Builder()
    compound = TopoDS_Compound()
    builder.MakeCompound(compound)
    for ent in entities:
        for shape in ent.shapes:
            builder.Add(compound, shape)

    fixer = ShapeFix_Shape()
    fixer.Init(compound)
    fixer.SetPrecision(point_tolerance)
    fixer.SetMaxTolerance(point_tolerance * max_tolerance_multiplier)
    fixer.Perform()

    context = fixer.Context()

    for ent in entities:
        repaired: list = []
        for shape in ent.shapes:
            new_shape = context.Apply(shape)
            if new_shape is None or new_shape.IsNull():
                # ShapeFix removed this piece entirely (degenerate).
                continue
            repaired.append(new_shape)
        ent.shapes = repaired

    return entities
