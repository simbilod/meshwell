"""Node-dedup hardening: absolute tolerance, collapse detection, topology validation."""
from __future__ import annotations

import gmsh
import numpy as np
import pytest
from shapely.geometry import Polygon, box

from meshwell.orchestrator import generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.resolution import StructuredExtrusionResolutionSpec
from meshwell.structured.exceptions import (
    DegenerateElementsAfterDedupError,
    InvalidMeshTopologyError,
)
from meshwell.structured.wedge import (
    _check_no_degenerate_elements,
    _remove_duplicate_nodes_tight,
    validate_mesh_topology,
)


@pytest.fixture
def gmsh_session():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("t")
    yield
    gmsh.finalize()


def _add_point_nodes(coords: list[tuple[float, float, float]]) -> None:
    """One discrete point entity + point element per coordinate."""
    for i, p in enumerate(coords, start=1):
        e = gmsh.model.addDiscreteEntity(0)
        gmsh.model.mesh.addNodes(0, e, [i], list(p))
        gmsh.model.mesh.addElementsByType(e, 15, [i], [i])


def _n_nodes() -> int:
    return len(gmsh.model.mesh.getNodes()[0])


@pytest.mark.parametrize("model_size", [1.0, 100.0, 3000.0])
def test_dedup_keeps_one_dbu_gap_regardless_of_model_size(gmsh_session, model_size):
    """A 1 nm gap (== point_tolerance) survives dedup even on mm-scale models."""
    _add_point_nodes([(0, 0, 0), (1e-3, 0, 0), (model_size, model_size, 0)])
    _remove_duplicate_nodes_tight(point_tolerance=1e-3)
    assert _n_nodes() == 3


@pytest.mark.parametrize("gap", [1e-10, 7.1e-6, 3.5e-5])
@pytest.mark.parametrize("model_size", [1.0, 3000.0])
def test_dedup_merges_perturbation_scale_duplicates(gmsh_session, model_size, gap):
    """Unshared duplicates separated by float noise or the 1e-5 perturbation still merge."""
    _add_point_nodes([(0, 0, 0), (gap, 0, 0), (model_size, model_size, 0)])
    _remove_duplicate_nodes_tight(point_tolerance=1e-3)
    assert _n_nodes() == 2


def test_dedup_never_merges_nodes_sharing_an_element(gmsh_session):
    """Two close nodes of the same element are real geometry and are kept distinct."""
    v = gmsh.model.addDiscreteEntity(3)
    coords = [0, 0, 0, 1, 0, 0, 0, 1, 0, 1e-5, 0, 1e-6]  # node 4 ~ node 1
    gmsh.model.mesh.addNodes(3, v, [1, 2, 3, 4], coords)
    gmsh.model.mesh.addElementsByType(v, 4, [1], [1, 2, 3, 4])
    _remove_duplicate_nodes_tight(point_tolerance=1e-3)
    assert _n_nodes() == 4


def test_degenerate_element_check_raises(gmsh_session):
    """A collapsed element (repeated node) is reported, not silently stripped."""
    v = gmsh.model.addDiscreteEntity(3)
    gmsh.model.mesh.addNodes(3, v, [1, 2, 3], [0, 0, 0, 1, 0, 0, 0, 1, 0])
    gmsh.model.mesh.addElementsByType(v, 4, [1], [1, 2, 3, 3])
    with pytest.raises(DegenerateElementsAfterDedupError):
        _check_no_degenerate_elements()


def _mesh_box() -> None:
    gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
    gmsh.model.occ.synchronize()
    gmsh.option.setNumber("Mesh.MeshSizeMax", 0.5)
    gmsh.model.mesh.generate(3)


def test_validate_mesh_topology_accepts_valid_mesh(gmsh_session):
    _mesh_box()
    validate_mesh_topology()


def test_validate_mesh_topology_detects_face_shared_by_three(gmsh_session):
    _mesh_box()
    _, etags, enodes = gmsh.model.mesh.getElements(3, 1)
    first = [int(n) for n in np.asarray(enodes[0])[:4]]
    new_tag = int(max(etags[0])) + 1000
    gmsh.model.mesh.addElementsByType(1, 4, [new_tag], first)  # duplicate tet
    with pytest.raises(InvalidMeshTopologyError, match="more than two"):
        validate_mesh_topology()


def test_validate_mesh_topology_detects_orphan_surface_element(gmsh_session):
    _mesh_box()
    s = gmsh.model.getBoundary([(3, 1)], oriented=False)[0][1]
    _, _, enodes = gmsh.model.mesh.getElements(3, 1)
    tet = [int(n) for n in np.asarray(enodes[0])[:4]]
    # Triangle through a tet's interior (nodes 0,1 + its 4th node rotated) that is
    # not a face of any tet: pick three nodes from two different tets.
    _, _, snodes = gmsh.model.mesh.getElements(2, abs(s))
    tri = [int(n) for n in np.asarray(snodes[0])[:2]] + [tet[3]]
    if len(set(tri)) < 3:
        tri[2] = tet[2]
    gmsh.model.mesh.addElementsByType(abs(s), 2, [10_000_000], tri)
    with pytest.raises(InvalidMeshTopologyError, match="not matching any volume face"):
        validate_mesh_topology()


def test_one_dbu_jog_in_mm_domain_meshes_without_collapse(tmp_path):
    """Regression: 2 nm jog in a ~3 mm domain (structured cohort on unstructured slab).

    With a relative Geometry.Tolerance of 1e-6, gmsh's effective merge distance
    was ~3 nm here, collapsing elements built across the stub and producing
    faces shared by three volume elements.
    """
    length = 3000.0
    domain = box(0, 0, length, 10)
    # Strip with a 2 nm vertical stub before a 45-degree chamfer (1-dbu grid artefact).
    strip = Polygon(
        [
            (500, 4.0),
            (1500, 4.0),
            (1500, 4.002),
            (1500.8, 4.802),
            (1500.8, 6.0),
            (500, 6.0),
        ]
    )
    bg = domain.difference(strip)
    entities = [
        PolyPrism(strip, {1.0: 0.0, 2.0: 0.0}, physical_name="metal", structured=True),
        PolyPrism(bg, {1.0: 0.0, 2.0: 0.0}, physical_name="bg", structured=True),
        PolyPrism(domain, {0.0: 0.0, 1.0: 0.0}, physical_name="substrate"),
    ]
    generate_mesh(
        entities,
        dim=3,
        output_mesh=tmp_path / "jog.msh",
        default_characteristic_length=50.0,
        resolution_specs={
            "metal": [StructuredExtrusionResolutionSpec(n_layers=1)],
            "bg": [StructuredExtrusionResolutionSpec(n_layers=1)],
        },
    )
    # post_3d hook already ran validate_mesh_topology(); re-check on final model
    # and confirm both stub endpoints survive as distinct nodes on each z-plane.
    validate_mesh_topology()
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    xyz = xyz.reshape(-1, 3)
    for z in (1.0, 2.0):
        for y in (4.0, 4.002):
            d = np.linalg.norm(xyz - np.array([1500.0, y, z]), axis=1)
            assert d.min() < 1e-6, f"missing node at (1500, {y}, {z})"
