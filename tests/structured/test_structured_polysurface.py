"""Tests for StructuredPolySurface (horizontal) and InterfaceTag (vertical) on structured cohorts."""
from __future__ import annotations

import gmsh
import pytest
from shapely.geometry import LineString, Polygon

from meshwell.interface_tag import InterfaceTag
from meshwell.mesh import mesh
from meshwell.orchestrator import cad, generate_mesh
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import StructuredPolySurface
from meshwell.resolution import StructuredExtrusionResolutionSpec
from meshwell.structured.exceptions import StructuredZStackError
from meshwell.utils import deserialize


def _get_physical_group_map() -> dict[tuple[int, str], list[int]]:
    """Return {(dim, name): [entity_tag, ...]} from the active gmsh model."""
    out: dict[tuple[int, str], list[int]] = {}
    for dim, gtag in gmsh.model.getPhysicalGroups():
        name = gmsh.model.getPhysicalName(dim, gtag)
        tags = [int(t) for t in gmsh.model.getEntitiesForPhysicalGroup(dim, gtag)]
        out[(dim, name)] = tags
    return out


def _get_entity_mesh_nodes(dim: int, tags: list[int]) -> set[int]:
    """Collect all mesh node tags belonging to the given gmsh entities."""
    nodes: set[int] = set()
    for t in tags:
        _etypes, _etags, enodes = gmsh.model.mesh.getElements(dim, t)
        for arr in enodes:
            nodes.update(int(n) for n in arr)
    return nodes


def test_horizontal_structured_polysurface_interior_interface(tmp_path):
    """StructuredPolySurface at an interior z-plane tags shared 3D wedge faces conformally."""
    box_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    port_poly = Polygon([(0, 4), (2, 4), (2, 6), (0, 6)])

    slab_bot = PolyPrism(
        polygons=box_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab_bot",
        structured=True,
    )
    slab_top = PolyPrism(
        polygons=box_poly,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="slab_top",
        structured=True,
    )
    port = StructuredPolySurface(
        polygons=port_poly,
        z=1.0,
        physical_name="lumped_port_1",
    )

    generate_mesh(
        [slab_bot, slab_top, port],
        dim=3,
        output_mesh=tmp_path / "interior_port.msh",
        default_characteristic_length=2.0,
        resolution_specs={
            "slab_bot": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "slab_top": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )

    groups = _get_physical_group_map()
    assert (2, "lumped_port_1") in groups
    port_faces = groups[(2, "lumped_port_1")]
    assert len(port_faces) >= 1

    # Every tagged face must be bound to both 3D volumes (0 floating 2D entities).
    for ftag in port_faces:
        upward_vols, _downward = gmsh.model.getAdjacencies(2, ftag)
        assert len(upward_vols) == 2, f"Expected 2 adjacent volumes for face {ftag}, got {upward_vols}"

    # Every 2D mesh node on the port must exist in the 3D volume elements (MFEM STable3D invariant).
    vol_tags = groups[(3, "slab_bot")] + groups[(3, "slab_top")]
    vol_nodes = _get_entity_mesh_nodes(3, vol_tags)
    port_nodes = _get_entity_mesh_nodes(2, port_faces)
    assert port_nodes, "Port should have 2D elements"
    assert port_nodes.issubset(vol_nodes), "All 2D port nodes must belong to 3D volume elements"

    # Natural solid-to-solid interface is preserved, while Solid___Port groups are not emitted.
    assert (2, "slab_bot___slab_top") in groups or (2, "slab_top___slab_bot") in groups
    assert (2, "slab_bot___lumped_port_1") not in groups
    assert (2, "slab_top___lumped_port_1") not in groups


def test_horizontal_structured_polysurface_exterior_subtracts_from_none(tmp_path):
    """StructuredPolySurface on an exterior cohort z-plane subtracts from Solid___None."""
    box_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    port_poly = Polygon([(2, 2), (8, 2), (8, 8), (2, 8)])

    slab = PolyPrism(
        polygons=box_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab",
        structured=True,
    )
    port = StructuredPolySurface(
        polygons=port_poly,
        z=1.0,
        physical_name="top_patch",
    )

    generate_mesh(
        [slab, port],
        dim=3,
        output_mesh=tmp_path / "exterior_port.msh",
        default_characteristic_length=2.0,
        resolution_specs={"slab": [StructuredExtrusionResolutionSpec(n_layers=1)]},
    )

    groups = _get_physical_group_map()
    assert (2, "top_patch") in groups
    assert (2, "slab___None") in groups
    patch_faces = set(groups[(2, "top_patch")])
    none_faces = set(groups[(2, "slab___None")])
    assert patch_faces.isdisjoint(none_faces), "Tagged exterior face must be subtracted from slab___None"
    for ftag in patch_faces:
        upward_vols, _ = gmsh.model.getAdjacencies(2, ftag)
        assert len(upward_vols) == 1


def test_vertical_interfacetag_chord_and_interior_dangling(tmp_path):
    """InterfaceTag works for both boundary-to-boundary chords and interior dangling segments."""
    box_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    slab_bot = PolyPrism(
        polygons=box_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab_bot",
        structured=True,
    )
    slab_top = PolyPrism(
        polygons=box_poly,
        buffers={1.0: 0.0, 2.0: 0.0},
        physical_name="slab_top",
        structured=True,
    )

    # Interior vertical sheet from boundary (0, 5) terminating strictly inside at (2, 5),
    # spanning both z-intervals [0, 1] and [1, 2].
    v_port = InterfaceTag(
        linestrings=LineString([(0.0, 5.0), (2.0, 5.0)]),
        zmin=0.0,
        zmax=2.0,
        physical_name="vertical_lumped_port",
        structured=True,
    )

    generate_mesh(
        [slab_bot, slab_top, v_port],
        dim=3,
        output_mesh=tmp_path / "vertical_interior_port.msh",
        default_characteristic_length=2.0,
        resolution_specs={
            "slab_bot": [StructuredExtrusionResolutionSpec(n_layers=2)],
            "slab_top": [StructuredExtrusionResolutionSpec(n_layers=2)],
        },
    )

    groups = _get_physical_group_map()
    assert (2, "vertical_lumped_port") in groups
    v_faces = groups[(2, "vertical_lumped_port")]
    # Spans 2 z-intervals ([0, 1] and [1, 2]) -> 2 lateral faces
    assert len(v_faces) == 2
    for ftag in v_faces:
        upward_vols, _ = gmsh.model.getAdjacencies(2, ftag)
        assert len(upward_vols) == 2, f"Interior vertical port face {ftag} must bound 2 sub-solids"

    vol_tags = groups[(3, "slab_bot")] + groups[(3, "slab_top")]
    vol_nodes = _get_entity_mesh_nodes(3, vol_tags)
    port_nodes = _get_entity_mesh_nodes(2, v_faces)
    assert port_nodes and port_nodes.issubset(vol_nodes)


def test_structured_polysurface_zstack_error():
    """StructuredPolySurface and InterfaceTag(structured=True) raise StructuredZStackError on unaligned z."""
    box_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    slab = PolyPrism(
        polygons=box_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab",
        structured=True,
    )

    # Mid-slab z = 0.5
    bad_h_mid = StructuredPolySurface(
        polygons=Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        z=0.5,
        physical_name="bad_mid",
    )
    with pytest.raises(StructuredZStackError):
        cad([slab, bad_h_mid])

    # Outside z = 5.0
    bad_h_out = StructuredPolySurface(
        polygons=Polygon([(2, 2), (4, 2), (4, 4), (2, 4)]),
        z=5.0,
        physical_name="bad_out",
    )
    with pytest.raises(StructuredZStackError):
        cad([slab, bad_h_out])

    # Unaligned vertical InterfaceTag(structured=True)
    bad_v = InterfaceTag(
        linestrings=LineString([(0, 0), (5, 0)]),
        zmin=0.0,
        zmax=0.5,
        physical_name="bad_v",
        structured=True,
    )
    with pytest.raises(StructuredZStackError):
        cad([slab, bad_v])


def test_structured_polysurface_mesh_order_priority(tmp_path):
    """Overlapping StructuredPolySurfaces resolve ownership by mesh_order."""
    box_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    slab = PolyPrism(
        polygons=box_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab",
        structured=True,
    )
    p_outer = StructuredPolySurface(
        polygons=Polygon([(1, 1), (9, 1), (9, 9), (1, 9)]),
        z=1.0,
        physical_name="outer_tag",
        mesh_order=2,
    )
    p_inner = StructuredPolySurface(
        polygons=Polygon([(3, 3), (7, 3), (7, 7), (3, 7)]),
        z=1.0,
        physical_name="inner_tag",
        mesh_order=1,
    )

    generate_mesh(
        [slab, p_outer, p_inner],
        dim=3,
        output_mesh=tmp_path / "prio.msh",
        default_characteristic_length=2.0,
        resolution_specs={"slab": [StructuredExtrusionResolutionSpec(n_layers=1)]},
    )

    groups = _get_physical_group_map()
    inner_faces = set(groups[(2, "inner_tag")])
    outer_faces = set(groups[(2, "outer_tag")])
    assert inner_faces and outer_faces
    assert inner_faces.isdisjoint(outer_faces)


def test_serialization_and_decoupled_cad_mesh(tmp_path):
    """StructuredPolySurface and InterfaceTag round-trip via to_dict/deserialize and decoupled cad -> mesh."""
    box_poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
    slab = PolyPrism(
        polygons=box_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="slab",
        structured=True,
    )
    h_port = StructuredPolySurface(
        polygons=Polygon([(0, 2), (2, 2), (2, 4), (0, 4)]),
        z=1.0,
        physical_name="h_port",
    )
    v_port = InterfaceTag(
        linestrings=LineString([(0, 0), (5, 0)]),
        zmin=0.0,
        zmax=1.0,
        physical_name="v_port",
        structured=True,
    )

    serialized = [e.to_dict() for e in (slab, h_port, v_port)]
    reconstructed = deserialize(serialized)
    assert isinstance(reconstructed[1], StructuredPolySurface)
    assert isinstance(reconstructed[2], InterfaceTag)
    assert reconstructed[2].structured is True

    xao_path = tmp_path / "scene.xao"
    msh_path = tmp_path / "scene.msh"
    cad(reconstructed, output_file=xao_path)
    mesh(
        dim=3,
        input_file=xao_path,
        output_file=msh_path,
        default_characteristic_length=2.0,
        resolution_specs={"slab": [StructuredExtrusionResolutionSpec(n_layers=2)]},
    )

    gmsh.initialize()
    try:
        gmsh.open(str(msh_path))
        groups = _get_physical_group_map()
        assert (2, "h_port") in groups
        assert (2, "v_port") in groups
    finally:
        gmsh.finalize()
