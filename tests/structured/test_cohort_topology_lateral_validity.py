"""Regression: cohort-path lateral-adjacent solids must produce valid XAO.

The Phase 2 cohort topology builder originally keyed `vertical_edges` by
`(slab_index, corner_id)`. When two slabs at the same z-interval shared
a lateral face (deduped by zinterval), the lateral face carried one slab's
vertical edges while the OTHER slab's shell used different TopoDS_Edge
objects, leaving its shell open. The resulting XAO could not be opened
by gmsh (silent failure).

This test exercises the lateral-adjacency case end-to-end through gmsh.open
to ensure the dedup fix holds. Uses straight squares only (no arcs) so it
doesn't depend on the separate arc-handling code paths.
"""

from __future__ import annotations

import pytest
import shapely

from meshwell.cad_common import prepare_entities
from meshwell.cad_occ import cad_occ
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.structured import (
    StructuredExtrusionResolutionSpec,
    build_phantom_shapes,
    build_plan,
)
from meshwell.structured import phantom as phantom_mod
from meshwell.structured.phantom import _group_phantom_solids_by_entity


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _poly(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


@pytest.fixture
def cohort_topology_on():
    prior = phantom_mod._USE_COHORT_TOPOLOGY
    phantom_mod._USE_COHORT_TOPOLOGY = True
    try:
        yield
    finally:
        phantom_mod._USE_COHORT_TOPOLOGY = prior


def test_cohort_lateral_adjacent_xao_opens_in_gmsh(
    cohort_topology_on,  # noqa: ARG001  pytest fixture
    tmp_path,
):
    """Two laterally-adjacent structured cubes produce an XAO gmsh can open."""
    pytest.importorskip("gmsh")
    import gmsh

    entities = [
        _poly("D", 0, 1, 1, x0=0, y0=0, x1=1, y1=1),
        _poly("E", 0, 1, 2, x0=1, y0=0, x1=2, y1=1),
    ]
    prepare_entities(entities, perturbation=1e-5, resolve_snap=1e-3)
    plan = build_plan(entities)
    phantom_result = build_phantom_shapes(plan)
    overrides = _group_phantom_solids_by_entity(plan, phantom_result)
    occ_entities = cad_occ(entities, entity_shape_overrides=overrides)

    xao_path = tmp_path / "cohort_lateral.xao"
    write_xao(occ_entities, xao_path)

    if gmsh.isInitialized():
        gmsh.finalize()
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        gmsh.open(str(xao_path))
    finally:
        gmsh.finalize()
