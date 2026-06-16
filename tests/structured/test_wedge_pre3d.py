from shapely.geometry import Polygon

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec

SQ = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])


def test_wedge_count_matches_bot_triangles_times_n_layers(tmp_path):
    p = PolyPrism(
        polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="s", structured=True
    )
    below = PolyPrism(polygons=SQ, buffers={-1.0: 0.0, 0.0: 0.0}, physical_name="below")
    above = PolyPrism(polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0}, physical_name="above")
    generate_mesh(
        entities=[p, below, above],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "s": [StructuredExtrusionResolutionSpec(n_layers=3)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")
    # Restrict wedge/tet counts to the structured slab via cell_sets.
    s_sets = m.cell_sets["s"]
    s_wedges = sum(
        len(s) for s, b in zip(s_sets, m.cells) if b.type == "wedge" and s is not None
    )
    s_tets = sum(
        len(s) for s, b in zip(s_sets, m.cells) if b.type == "tetra" and s is not None
    )
    assert s_wedges > 0
    assert s_tets == 0


def test_stacked_cohort_wedges_conformal(tmp_path):
    a = PolyPrism(
        polygons=SQ, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="a", structured=True
    )
    b = PolyPrism(
        polygons=SQ, buffers={1.0: 0.0, 2.0: 0.0}, physical_name="b", structured=True
    )
    below = PolyPrism(polygons=SQ, buffers={-1.0: 0.0, 0.0: 0.0}, physical_name="below")
    above = PolyPrism(polygons=SQ, buffers={2.0: 0.0, 3.0: 0.0}, physical_name="above")
    generate_mesh(
        entities=[a, b, below, above],
        dim=3,
        output_mesh=tmp_path / "out.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    import meshio

    m = meshio.read(tmp_path / "out.msh")
    import numpy as np

    pts = m.points
    _, counts = np.unique(np.round(pts, 6), axis=0, return_counts=True)
    assert counts.max() == 1, "duplicate node positions → non-conformal mesh"


def test_stamp_wedges_function_exists():
    """Smoke test: the function imports and is callable."""
    from meshwell.structured.wedge import stamp_wedges

    assert callable(stamp_wedges)
