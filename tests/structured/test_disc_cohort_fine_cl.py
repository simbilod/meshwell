"""Disc-cohort PLC regression at fine characteristic length.

Pre-simplification, this scene failed with
``PLC Error: A segment and a facet intersect at point`` at cl <= 0.4
because the CohortNeighbourUnstructured custom shell built cylindrical
laterals via BRepBuilderAPI_MakeFace which wrapped Geom_CylindricalSurface
in Geom_RectangularTrimmedSurface. gmsh's XAO/BREP reader sees the wrap
as ``Unknown`` instead of ``Cylinder`` and refines the mesh to a state
that violates its PLC checker. Routing the cladding through plain
PolyPrism (which uses BRepPrimAPI_MakePrism and keeps Geom_CylindricalSurface
unwrapped) fixes the failure.

Spike evidence: 2026-06-05 investigation in
docs/superpowers/specs/2026-06-01-cohort-topology-investigations.md.
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec


def _disc(radius=5.0, n=48):
    return Polygon(
        [
            (
                radius * math.cos(2 * math.pi * i / n),
                radius * math.sin(2 * math.pi * i / n),
            )
            for i in range(n)
        ]
    )


def test_disc_cohort_meshes_at_fine_cl(tmp_path):
    """Disc-cohort + cap + base meshes at cl=0.2 without PLC error."""
    disc = PolyPrism(
        _disc(),
        {0.0: 0.0, 1.0: 0.0},
        physical_name="disc",
        structured=True,
        identify_arcs=True,
    )
    base = PolyPrism(
        _disc(),
        {-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
        identify_arcs=True,
    )
    cap = PolyPrism(
        _disc(),
        {1.0: 0.0, 2.0: 0.0},
        physical_name="cap",
        identify_arcs=True,
    )
    out = tmp_path / "x.msh"
    generate_mesh(
        [disc, base, cap],
        dim=3,
        output_mesh=out,
        default_characteristic_length=0.2,
        resolution_specs={"disc": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
    assert out.exists()
