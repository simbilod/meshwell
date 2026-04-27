"""Shared pytest fixtures for meshwell tests.

Centralizes scenes that recur across test files: unit-square polysurfaces,
standard extruded prisms, and tiled-prism arrangements. Tests can request
these by parameter name and avoid re-constructing the same scene inline.
"""
from __future__ import annotations

import pytest
import shapely

from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

# --- 2D scenes -----------------------------------------------------------


@pytest.fixture
def unit_square_polysurface() -> PolySurface:
    """Single 1x1 PolySurface at the origin, mesh_order=1, name='A'."""
    return PolySurface(
        polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        physical_name="A",
        mesh_order=1,
    )


@pytest.fixture
def overlapping_unit_squares() -> list[PolySurface]:
    """Two unit squares overlapping on x in [0.5, 1]: A wins (mesh_order=1)."""
    return [
        PolySurface(
            polygons=shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
            physical_name="A",
            mesh_order=1,
        ),
        PolySurface(
            polygons=shapely.Polygon([(0.5, 0), (1.5, 0), (1.5, 1), (0.5, 1)]),
            physical_name="B",
            mesh_order=2,
        ),
    ]


# --- 3D scenes (prisms) --------------------------------------------------


@pytest.fixture
def standard_prism_buffers() -> dict[float, float]:
    """Vertical extrusion 0->1 with no lateral buffering at either height."""
    return {0.0: 0.0, 1.0: 0.0}


@pytest.fixture
def unit_box_prism(standard_prism_buffers) -> PolyPrism:
    """5x5x1 PolyPrism at the origin, mesh_order=1, name='A'."""
    return PolyPrism(
        polygons=shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
        buffers=standard_prism_buffers,
        physical_name="A",
        mesh_order=1,
    )


@pytest.fixture
def two_abutting_prisms(standard_prism_buffers) -> list[PolyPrism]:
    """Prisms A and B sharing an interface at x=5; A is winner (mesh_order=1)."""
    return [
        PolyPrism(
            polygons=shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),
            buffers=standard_prism_buffers,
            physical_name="A",
            mesh_order=1,
        ),
        PolyPrism(
            polygons=shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
            buffers=standard_prism_buffers,
            physical_name="B",
            mesh_order=2,
        ),
    ]


@pytest.fixture
def three_abutting_prisms(standard_prism_buffers) -> list[PolyPrism]:
    """Prisms A, B, C in a row sharing interfaces at x=2 and x=5."""
    return [
        PolyPrism(
            polygons=shapely.Polygon([(0, 0), (2, 0), (2, 5), (0, 5)]),
            buffers=standard_prism_buffers,
            physical_name="A",
            mesh_order=1,
        ),
        PolyPrism(
            polygons=shapely.Polygon([(2, 0), (5, 0), (5, 5), (2, 5)]),
            buffers=standard_prism_buffers,
            physical_name="B",
            mesh_order=2,
        ),
        PolyPrism(
            polygons=shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)]),
            buffers=standard_prism_buffers,
            physical_name="C",
            mesh_order=3,
        ),
    ]


# --- Cross-backend pipeline fixture --------------------------------------


@pytest.fixture(params=["gmsh", "occ"])
def cad_pipeline(request, tmp_path):
    """Run the full CAD+mesh pipeline through one backend.

    Tests requesting this fixture run twice (once per backend) and receive
    a callable ``run(entities, dim=3, **mesh_kwargs) -> meshio.Mesh``. The
    callable abstracts the in-memory (cad_gmsh) vs xao-roundtrip (cad_occ)
    glue. ``run.backend`` is the active backend name ("gmsh" or "occ").
    """
    import meshio  # noqa: F401 — confirm meshio is importable at fixture time

    backend = request.param
    msh_path = tmp_path / "out.msh"

    if backend == "gmsh":
        from meshwell.cad_gmsh import cad_gmsh
        from meshwell.mesh import mesh

        def run(entities, dim=3, default_characteristic_length=1.0, **mesh_kwargs):
            _, mm = cad_gmsh(entities)
            return mesh(
                dim=dim,
                default_characteristic_length=default_characteristic_length,
                model=mm,
                output_file=str(msh_path),
                n_threads=1,
                **mesh_kwargs,
            )

    else:
        from meshwell.cad_occ import cad_occ
        from meshwell.mesh import mesh
        from meshwell.occ_xao_writer import write_xao

        def run(entities, dim=3, default_characteristic_length=1.0, **mesh_kwargs):
            labeled = cad_occ(entities)
            xao_path = tmp_path / "out.xao"
            write_xao(labeled, str(xao_path))
            return mesh(
                dim=dim,
                default_characteristic_length=default_characteristic_length,
                input_file=str(xao_path),
                output_file=str(msh_path),
                n_threads=1,
                **mesh_kwargs,
            )

    run.backend = backend
    return run
