"""A structured cohort no longer needs unstructured neighbours wrapping it.

The old ``validate_cohort_wrapping`` check required every cohort outer
z-plane to be fully covered by adjacent unstructured PolyPrisms, raising
``CohortNotWrappedError`` otherwise. That guard was v1 conservatism, not a
correctness precondition: an exposed cohort face is a legitimate exterior
boundary. These end-to-end tests pin down that cohorts mesh correctly with
full, partial, and no surrounding cladding.
"""
from __future__ import annotations

import meshio
from shapely.geometry import box

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _core():
    # 4x4 wedge-meshed slab, z in [0, 1], 3 layers through the thickness.
    return PolyPrism(
        box(0, 0, 4, 4),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="core",
        structured=True,
    )


def _mesh(entities, tmp_path):
    out = tmp_path / "x.msh"
    generate_mesh(
        entities,
        dim=3,
        output_mesh=out,
        default_characteristic_length=1.0,
        resolution_specs={"core": [StructuredExtrusionResolutionSpec(n_layers=3)]},
    )
    m = meshio.read(out)
    wedges = sum(cb.data.shape[0] for cb in m.cells if cb.type == "wedge")
    tets = sum(cb.data.shape[0] for cb in m.cells if cb.type == "tetra")
    groups = {k for k in m.cell_sets if "___" in k}
    return wedges, tets, groups


def test_cohort_fully_wrapped(tmp_path):
    """Cladding above and below fully covering the core: both caps interface."""
    below = PolyPrism(box(-2, -2, 6, 6), {-1.0: 0.0, 0.0: 0.0}, physical_name="below")
    above = PolyPrism(box(-2, -2, 6, 6), {1.0: 0.0, 2.0: 0.0}, physical_name="above")
    wedges, tets, groups = _mesh([_core(), below, above], tmp_path)
    assert wedges > 0
    assert tets > 0
    # Both caps are conformal interfaces with their neighbours.
    assert "core___below" in groups
    assert "core___above" in groups


def test_cohort_half_overlapping_cladding(tmp_path):
    """A partly-covering neighbour splits a cap into interface + exterior tiles."""
    below = PolyPrism(box(-2, -2, 6, 6), {-1.0: 0.0, 0.0: 0.0}, physical_name="below")
    # 'above' covers x in [-2, 2], i.e. only the left half of the core's
    # [0, 4] footprint at z=1.
    above = PolyPrism(box(-2, -2, 2, 6), {1.0: 0.0, 2.0: 0.0}, physical_name="above")
    wedges, tets, groups = _mesh([_core(), below, above], tmp_path)
    assert wedges > 0
    assert tets > 0
    # The covered part of the top cap is an interface...
    assert "core___above" in groups
    # ...and the uncovered part is an exterior face.
    assert "core___None" in groups


def test_cohort_no_cladding(tmp_path):
    """A fully exposed cohort meshes into wedges with only exterior faces."""
    wedges, tets, groups = _mesh([_core()], tmp_path)
    assert wedges > 0
    # No unstructured neighbours -> no tetrahedra, no interface groups.
    assert tets == 0
    assert groups == {"core___None"}
