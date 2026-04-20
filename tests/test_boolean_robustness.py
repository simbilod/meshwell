import numpy as np
import shapely
from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_overlapping_boxes_robustness():
    """Verify that overlapping boxes with specific mesh_order resolve correctly.

    This scenario replicates the conditions where coordinate snapping previously
    caused incorrect vertex connections.
    """
    width = 1
    height = 1

    # Core: priority 0
    core = shapely.geometry.box(-width / 2, -0.2, +width / 2, height)
    # Cladding: priority 1
    cladding = shapely.geometry.box(-width * 2, 0, width * 2, height * 3)
    # Buried Oxide: priority 2
    buried_oxide = shapely.geometry.box(-width * 2, -height * 2, width * 2, 0)

    core_surface = PolySurface(polygons=core, physical_name="core", mesh_order=0)
    cladding_surface = PolySurface(
        polygons=cladding, physical_name="cladding", mesh_order=1
    )
    buried_oxide_surface = PolySurface(
        polygons=buried_oxide, physical_name="buried_oxide", mesh_order=2
    )

    entities = [core_surface, cladding_surface, buried_oxide_surface]

    # Generate mesh using the unified OCC pipeline
    m = generate_mesh(
        entities=entities,
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )

    # Check that we have the expected physical groups
    # 'core___cladding', 'core___buried_oxide', 'cladding___buried_oxide' should exist
    physical_names = list(m.field_data.keys())
    assert "core" in physical_names
    assert "cladding" in physical_names
    assert "buried_oxide" in physical_names
    assert "core___cladding" in physical_names
    assert "core___buried_oxide" in physical_names
    assert "cladding___buried_oxide" in physical_names

    # Ensure no node connection artifacts. Exact node count is non-deterministic
    # under gmsh's meshing; pin a reasonable envelope around the OCC baseline.
    assert 120 <= len(m.points) <= 150


def _rounded_rect(w, h, r, n_arc=8):
    hw, hh = w / 2, h / 2
    specs = [
        ((hw - r, hh - r), 0.0),
        ((-hw + r, hh - r), np.pi / 2),
        ((-hw + r, -hh + r), np.pi),
        ((hw - r, -hh + r), 3 * np.pi / 2),
    ]
    coords = []
    for (cx, cy), a0 in specs:
        coords.extend(
            (cx + r * np.cos(a), cy + r * np.sin(a))
            for a in np.linspace(a0, a0 + np.pi / 2, n_arc + 1)
        )
    return Polygon(coords)


def test_rounded_rect_in_rect_shapely_diff_meshes_3d():
    """3D mesh succeeds for rounded-rect-in-rect built via shapely.difference.

    Exercises the full hardening pipeline end-to-end: pointwise precision
    snapping, seam-duplicate stripping, and canonical seam rotation must all
    cooperate so BOPAlgo produces clean shared arc boundaries and gmsh can
    tetrahedralize the resulting volume.
    """
    inner = _rounded_rect(w=4.0, h=3.0, r=0.6, n_arc=8)
    outer_rect = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    outer_with_cutout = outer_rect.difference(inner)

    entities = [
        PolyPrism(
            polygons=outer_with_cutout,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="outer",
            mesh_order=2,
            identify_arcs=True,
        ),
        PolyPrism(
            polygons=inner,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="inner",
            mesh_order=1,
            identify_arcs=True,
        ),
    ]

    m = generate_mesh(
        entities=entities,
        dim=3,
        backend="occ",
        default_characteristic_length=0.3,
        n_threads=1,
    )

    physical_names = list(m.field_data.keys())
    assert "outer" in physical_names
    assert "inner" in physical_names
    assert any(n in physical_names for n in ("outer___inner", "inner___outer"))

    tet_counts = [len(cb.data) for cb in m.cells if cb.type == "tetra"]
    assert tet_counts, "no tetra cell block produced"
    assert sum(tet_counts) > 0, "no tetrahedra generated"
    assert len(m.points) > 0


if __name__ == "__main__":
    test_overlapping_boxes_robustness()
    test_rounded_rect_in_rect_shapely_diff_meshes_3d()
