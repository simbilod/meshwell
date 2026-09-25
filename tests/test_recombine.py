"""RecombineResolutionSpec: triangle -> quad recombination (and hexes in cohorts)."""

import numpy as np
import pytest
import shapely

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.resolution import (
    ConstantInField,
    RecombinedInterfaceError,
    RecombineResolutionSpec,
    StructuredExtrusionResolutionSpec,
)
from meshwell.utils import deserialize


def _counts(mesh) -> dict[str, int]:
    out: dict[str, int] = {}
    for cb in mesh.cells:
        out[cb.type] = out.get(cb.type, 0) + cb.data.shape[0]
    return out


def test_2d_surface_all_quads(tmp_path):
    mesh = generate_mesh(
        entities=[PolySurface(polygons=shapely.box(0, 0, 2, 1), physical_name="s")],
        dim=2,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.1,
        resolution_specs={"s": [RecombineResolutionSpec()]},
    )
    c = _counts(mesh)
    assert c.get("quad", 0) > 0
    assert c.get("triangle", 0) == 0


def test_2d_min_mass_filters_surfaces(tmp_path):
    big = PolySurface(polygons=shapely.box(0, 0, 2, 2), physical_name="s", mesh_order=1)
    small = PolySurface(
        polygons=shapely.box(3, 0, 3.5, 0.5), physical_name="s", mesh_order=1
    )
    mesh = generate_mesh(
        entities=[big, small],
        dim=2,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.1,
        resolution_specs={"s": [RecombineResolutionSpec(min_mass=1.0)]},
    )
    c = _counts(mesh)
    assert c.get("quad", 0) > 0  # big square recombined
    assert c.get("triangle", 0) > 0  # small square left alone


def test_3d_outer_boundary_quads_get_pyramids(tmp_path):
    box = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1), buffers={0: 0, 1: 0}, physical_name="b"
    )
    mesh = generate_mesh(
        entities=[box],
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.25,
        resolution_specs={"b": [RecombineResolutionSpec()]},
    )
    c = _counts(mesh)
    assert c.get("quad", 0) > 0
    assert c.get("pyramid", 0) > 0
    assert c.get("tetra", 0) > 0


def test_3d_shared_quad_face_with_tets_raises(tmp_path):
    b1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1), buffers={0: 0, 1: 0}, physical_name="b1"
    )
    b2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1), buffers={0: 0, 1: 0}, physical_name="b2"
    )
    with pytest.raises(RecombinedInterfaceError, match="shared by two unstructured"):
        generate_mesh(
            entities=[b1, b2],
            dim=3,
            output_mesh=tmp_path / "m.msh",
            default_characteristic_length=0.25,
            resolution_specs={"b1": [RecombineResolutionSpec()]},
        )


def test_3d_boundary_group_avoids_interface(tmp_path):
    """Targeting the b1___None boundary group recombines only outer faces."""
    b1 = PolyPrism(
        polygons=shapely.box(0, 0, 1, 1), buffers={0: 0, 1: 0}, physical_name="b1"
    )
    b2 = PolyPrism(
        polygons=shapely.box(1, 0, 2, 1), buffers={0: 0, 1: 0}, physical_name="b2"
    )
    mesh = generate_mesh(
        entities=[b1, b2],
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.25,
        resolution_specs={"b1___None": [RecombineResolutionSpec()]},
    )
    c = _counts(mesh)
    assert c.get("quad", 0) > 0
    assert c.get("pyramid", 0) > 0


def _stacked_cohort(extra=()):
    sq = shapely.box(0, 0, 2, 2)
    a = PolyPrism(
        polygons=sq, buffers={0.0: 0.0, 1.0: 0.0}, physical_name="a", structured=True
    )
    b = PolyPrism(
        polygons=sq, buffers={1.0: 0.0, 2.0: 0.0}, physical_name="b", structured=True
    )
    return [a, b, *extra]


@pytest.mark.parametrize("target", ["a", "b"])
def test_cohort_column_all_hexes(tmp_path, target):
    """Spec on either slab recombines the column seed face: hexes throughout."""
    specs = {
        "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
        "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
    }
    specs[target].append(RecombineResolutionSpec())
    mesh = generate_mesh(
        _stacked_cohort(),
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.5,
        resolution_specs=specs,
    )
    c = _counts(mesh)
    assert c.get("hexahedron", 0) > 0
    assert c.get("wedge", 0) == 0
    assert c.get("tetra", 0) == 0


def test_cohort_without_spec_stays_wedges(tmp_path):
    mesh = generate_mesh(
        _stacked_cohort(),
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )
    c = _counts(mesh)
    assert c.get("wedge", 0) > 0
    assert c.get("hexahedron", 0) == 0


_FACES = {
    "tetra": [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
    "pyramid": [(0, 1, 2, 3), (0, 1, 4), (1, 2, 4), (2, 3, 4), (3, 0, 4)],
    "wedge": [(0, 1, 2), (3, 4, 5), (0, 1, 4, 3), (1, 2, 5, 4), (2, 0, 3, 5)],
    "hexahedron": [
        (0, 1, 2, 3),
        (4, 5, 6, 7),
        (0, 1, 5, 4),
        (1, 2, 6, 5),
        (2, 3, 7, 6),
        (3, 0, 4, 7),
    ],
}


def _assert_conformal_box(mesh, box_area: float) -> None:
    """Every face is shared by <= 2 cells; unshared faces tile the outer box."""
    faces: dict[frozenset, list] = {}
    for cb in mesh.cells:
        for local in _FACES.get(cb.type, []):
            for cell in cb.data:
                f = [int(cell[i]) for i in local]
                faces.setdefault(frozenset(f), []).append(f)
    assert max(len(v) for v in faces.values()) <= 2
    pts = np.asarray(mesh.points)
    area = 0.0
    for occ in faces.values():
        if len(occ) == 1:
            f = occ[0]
            for k in range(1, len(f) - 1):  # fan triangulation
                area += 0.5 * np.linalg.norm(
                    np.cross(pts[f[k]] - pts[f[0]], pts[f[k + 1]] - pts[f[0]])
                )
    assert area == pytest.approx(box_area, rel=1e-6)


def test_cohort_recombined_column_on_tets_gets_pyramids(tmp_path):
    """Recombined column over an unstructured base: pyramids on the tet side."""
    base = PolyPrism(
        polygons=shapely.box(0, 0, 2, 2),
        buffers={-1.0: 0.0, 0.0: 0.0},
        physical_name="base",
    )
    mesh = generate_mesh(
        _stacked_cohort(extra=(base,)),
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.5,
        resolution_specs={
            "a": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "b": [
                StructuredExtrusionResolutionSpec(n_layers=2),
                RecombineResolutionSpec(),
            ],
        },
    )
    c = _counts(mesh)
    assert c.get("hexahedron", 0) > 0
    assert c.get("wedge", 0) == 0
    assert c.get("tetra", 0) > 0
    assert c.get("pyramid", 0) == 16  # one per quad of the 4x4 interface face
    _assert_conformal_box(mesh, box_area=2 * (2 * 2) + 4 * (2 * 3))


def test_cohort_lateral_face_next_to_tets_gets_pyramids(tmp_path):
    """Cohort lateral (quad) face shared with an unstructured prism."""
    a = PolyPrism(
        polygons=shapely.box(0, 0, 2, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="a",
        structured=True,
    )
    u = PolyPrism(
        polygons=shapely.box(2, 0, 4, 4),
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="u",
    )
    mesh = generate_mesh(
        [a, u],
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=0.5,
        resolution_specs={"a": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )
    c = _counts(mesh)
    assert c.get("wedge", 0) > 0
    assert c.get("tetra", 0) > 0
    assert c.get("pyramid", 0) == 16  # 8 x 2 quads on the shared 4 x 1 face
    _assert_conformal_box(mesh, box_area=2 * (4 * 4) + 2 * (4 * 1) + 2 * (4 * 1))


def _hybrid_stack():
    """Cohorts + recombined film + recombined outer cladding faces, next to tets."""
    film_shape = shapely.union(shapely.box(0, 0, 6, 1.5), shapely.box(0, 0, 1.5, 4))
    film = PolyPrism(
        polygons=film_shape,
        buffers={0.0: 0.0, 0.4: 0.0},
        physical_name="film",
        structured=True,
    )
    pad = PolyPrism(
        polygons=shapely.box(3, 0, 6, 1.5),
        buffers={0.4: 0.0, 1.0: 0.0},
        physical_name="pad",
        structured=True,
    )
    wg = PolyPrism(
        polygons=shapely.box(3, 2.5, 7, 3.2),
        buffers={0.0: 0.0, 0.4: 0.0},
        physical_name="wg",
        structured=True,
    )
    domain = shapely.box(-1, -1, 8, 5)
    entities = [
        film,
        pad,
        wg,
        PolyPrism(
            polygons=domain, buffers={-1.0: 0.0, 0.0: 0.0}, physical_name="substrate"
        ),
        PolyPrism(
            polygons=domain.difference(film_shape).difference(wg.polygons),
            buffers={0.0: 0.0, 0.4: 0.0},
            physical_name="cladding",
        ),
        PolyPrism(
            polygons=domain.difference(pad.polygons),
            buffers={0.4: 0.0, 1.0: 0.0},
            physical_name="cladding",
        ),
        PolyPrism(
            polygons=domain, buffers={1.0: 0.0, 2.0: 0.0}, physical_name="cladding"
        ),
    ]
    specs = {
        "film": [
            ConstantInField(resolution=0.25, apply_to="volumes"),
            StructuredExtrusionResolutionSpec(n_layers=2),
            RecombineResolutionSpec(),
        ],
        "pad": [
            ConstantInField(resolution=0.25, apply_to="volumes"),
            StructuredExtrusionResolutionSpec(n_layers=3),
        ],
        "wg": [
            ConstantInField(resolution=0.2, apply_to="volumes"),
            StructuredExtrusionResolutionSpec(n_layers=2),
        ],
        "cladding___None": [RecombineResolutionSpec()],
    }
    return entities, specs


@pytest.mark.parametrize("h", [0.3, 0.4])
def test_hybrid_pyramids_do_not_fold(tmp_path, h):
    """Regression: gmsh's default pyramid smoother folded the tet mesh here.

    ``process_mesh`` meshes with ``Mesh.OptimizePyramids=1`` and runs
    ``validate_mesh_topology`` (which checks folds) whenever pyramids are built.
    """
    import gmsh

    entities, specs = _hybrid_stack()
    mesh = generate_mesh(
        entities,
        dim=3,
        output_mesh=tmp_path / "m.msh",
        default_characteristic_length=h,
        resolution_specs=specs,
    )
    c = _counts(mesh)
    assert c.get("pyramid", 0) > 0
    assert c.get("hexahedron", 0) > 0
    assert c.get("tetra", 0) > 0
    if gmsh.isInitialized():
        assert gmsh.option.getNumber("Mesh.OptimizePyramids") == 0


def test_serialization_roundtrip():
    spec = RecombineResolutionSpec(angle=30.0, mesh_algorithm_2d=None, min_mass=0.5)
    d = spec.to_dict()
    assert d["resolution_type"] == "RecombineResolutionSpec"
    back = deserialize(d)
    assert isinstance(back, RecombineResolutionSpec)
    assert back == spec


def test_refine_is_identity():
    spec = RecombineResolutionSpec(angle=30.0)
    assert spec.refine(0.5) == spec
