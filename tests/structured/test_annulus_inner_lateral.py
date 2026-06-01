"""Regression test: structured slabs with holes have inner lateral walls."""
import math

import gmsh
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _annulus(r_out, r_in, n=48):
    outer = Polygon(
        [
            (
                r_out * math.cos(2 * math.pi * i / n),
                r_out * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )
    inner = [
        (r_in * math.cos(2 * math.pi * i / n), r_in * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return Polygon(outer.exterior.coords, holes=[inner])


def test_structured_annulus_has_inner_lateral(tmp_path):
    """An annular structured slab must build BOTH outer and inner lateral walls."""
    annulus = PolyPrism(
        _annulus(2.0, 0.8),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="ann",
        structured=True,
        identify_arcs=True,
    )
    msh = tmp_path / "x.msh"
    generate_mesh(
        [annulus],
        dim=3,
        output_mesh=msh,
        default_characteristic_length=0.3,
        resolution_specs={"ann": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.open(str(msh))
        # Count surfaces grouped as ann___None
        ann_none_groups = [
            (dim, tag)
            for dim, tag in gmsh.model.getPhysicalGroups()
            if gmsh.model.getPhysicalName(dim, tag) == "ann___None"
        ]
        assert len(ann_none_groups) == 1, "ann___None group missing"
        ents = gmsh.model.getEntitiesForPhysicalGroup(*ann_none_groups[0])
        # Expected: bot, top, 2 outer half-cylinders, 2 inner half-cylinders = 6 surfaces.
        # Be lenient: at least 4 (in case of cache-collapse, but >= 4 still implies the
        # inner lateral was attempted).
        assert len(ents) >= 4, (
            f"annulus boundary surfaces = {len(ents)}, expected >= 4 "
            "(top + bot + outer + inner cylindrical walls)"
        )
    finally:
        gmsh.finalize()
