from __future__ import annotations

from pathlib import Path

import gmsh
import pytest
import shapely

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface
from meshwell.utils import compare_gmsh_files


def test_occ_polysurface(tmp_path):
    """Test PolySurface with OCC backend."""
    polygon1 = shapely.Polygon(
        [[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]],
        holes=([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [0.5, 0.5]],),
    )
    polygon2 = shapely.Polygon([[-1, -1], [-2, -1], [-2, -2], [-1, -2], [-1, -1]])
    polygon = shapely.MultiPolygon([polygon1, polygon2])

    polysurface_obj = PolySurface(polygons=polygon, physical_name="polysurface")

    # 1. Process via OCC
    occ_entities = cad_occ(entities_list=[polysurface_obj])
    assert len(occ_entities) == 1
    assert occ_entities[0].dim == 2

    # 2. Serialize to XAO
    output_xao = tmp_path / "test_polysurface_occ.xao"
    write_xao(occ_entities, output_xao)

    assert output_xao.exists()

    # 3. Mesh from XAO
    output_msh = tmp_path / "test_polysurface_occ.msh"
    mesh(
        input_file=output_xao,
        output_file=output_msh,
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    assert output_msh.exists()
    compare_gmsh_files(output_msh)


def test_occ_composite_3D():
    """Test composite 3D CAD with OCC backend, mirroring test_composite_cad_3D."""
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeSphere
    from OCP.gp import gp_Pnt

    # Create a prism
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    buffers = {0.0: 0.0, 1.0: 0.0}
    prism_obj = PolyPrism(
        polygons=polygon, buffers=buffers, physical_name="prism", mesh_order=1
    )

    # Create an OCC_entity box
    box_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeBox(
            gp_Pnt(1, 1, 0.5), 2.0, 2.0, 1.0
        ).Shape(),
        physical_name="box",
        mesh_order=2,
        additive=False,
    )

    # Create an OCC_entity sphere
    sphere_obj = OCC_entity(
        occ_function=lambda: BRepPrimAPI_MakeSphere(gp_Pnt(0, 0, 0.5), 0.75).Shape(),
        physical_name="sphere",
        mesh_order=2,
        additive=False,
    )

    entities = [prism_obj, sphere_obj, box_obj]

    # Process
    occ_entities = cad_occ(entities_list=entities)

    # Check that we have results
    assert len(occ_entities) > 0

    # Bridge and Tag
    # This will initialize gmsh and tag everything
    from meshwell.model import ModelManager

    mm = ModelManager()
    mm.load_occ_entities(occ_entities)

    # Verify physical names in gmsh model
    groups = gmsh.model.getPhysicalGroups()
    names = [gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups]
    assert "prism" in names
    assert "box" in names
    assert "sphere" in names

    mm.finalize()


def test_occ_buffered_prism(tmp_path):
    """Test PolyPrism with buffering in OCC backend."""
    polygon = shapely.Polygon([[0, 0], [2, 0], [2, 2], [0, 2], [0, 0]])
    # Non-zero buffers force the _create_occ_volume path
    buffers = {0.0: 0.0, 0.5: 0.2, 1.0: 0.0}
    prism_obj = PolyPrism(
        polygons=polygon, buffers=buffers, physical_name="buffered_prism"
    )

    occ_entities = cad_occ(entities_list=[prism_obj])
    assert len(occ_entities) == 1
    assert occ_entities[0].dim == 3

    output_xao = tmp_path / "test_buffered_prism_occ.xao"
    write_xao(occ_entities, output_xao)
    assert output_xao.exists()


def test_occ_many_polyprism_stress_with_arcs_and_coincidences():
    """Stress the OCC all-fragment pipeline with many PolyPrism entities.

    Exercises:
      - arc fitting (circles via shapely buffer with identify_arcs=True),
      - arc-on-arc overlaps (two circles intersecting each other),
      - arc-on-polygon overlaps (square carving into a disk),
      - exactly coinciding shapes (same polygon, different mesh_order — the
        lower-priority instance must be fully absorbed and produce no tags),
      - buffered (non-extrude) prism interleaved with extruded ones,
      - annulus (polygon with a hole) with arcs,
      - varied z-ranges so interfaces emerge in all dimensions.
    """
    import numpy as np
    from shapely.geometry import Point, Polygon

    from meshwell.model import ModelManager

    def circle(cx, cy, r, segs=48):
        return Point(cx, cy).buffer(r, quad_segs=segs)

    def reg_polygon(cx, cy, r, n):
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return Polygon([(cx + r * np.cos(a), cy + r * np.sin(a)) for a in ang])

    # Polygon reused by two coinciding prisms.
    coincident_poly = circle(3, 0, 1.0)

    entities = [
        # Background slab (lowest priority).
        PolyPrism(
            polygons=Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="slab",
            mesh_order=20,
        ),
        # Two arc-fitted disks with an arc-on-arc overlap.
        PolyPrism(
            polygons=circle(-2.0, 0.0, 1.5),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="disk_a",
            mesh_order=5,
            identify_arcs=True,
        ),
        PolyPrism(
            polygons=circle(-0.5, 0.0, 1.5),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="disk_b",
            mesh_order=6,
            identify_arcs=True,
        ),
        # Exactly coinciding pair: hi wins everywhere, lo is fully absorbed.
        PolyPrism(
            polygons=coincident_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="coincident_hi",
            mesh_order=1,
            identify_arcs=True,
        ),
        PolyPrism(
            polygons=coincident_poly,
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="coincident_lo",
            mesh_order=10,
            identify_arcs=True,
        ),
        # Square cutting into disk_b — arc-on-polygon interface.
        PolyPrism(
            polygons=Polygon([(-1.0, 1.0), (2.0, 1.0), (2.0, 3.0), (-1.0, 3.0)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="square_over_disk",
            mesh_order=3,
        ),
        # Annulus (polygon with hole), taller than the slab.
        PolyPrism(
            polygons=circle(1.0, -2.0, 1.2).difference(circle(1.0, -2.0, 0.6)),
            buffers={0.0: 0.0, 2.0: 0.0},
            physical_name="ring",
            mesh_order=4,
            identify_arcs=True,
        ),
        # Buffered (non-extrude) tapered prism.
        PolyPrism(
            polygons=Polygon([(3.0, 3.0), (4.5, 3.0), (4.5, 4.5), (3.0, 4.5)]),
            buffers={0.0: 0.0, 0.5: 0.2, 1.0: 0.0},
            physical_name="pyramid",
            mesh_order=7,
        ),
        # Thin tall column poking above everything.
        PolyPrism(
            polygons=Polygon([(-4.0, 3.0), (-3.5, 3.0), (-3.5, 3.5), (-4.0, 3.5)]),
            buffers={0.0: 0.0, 3.0: 0.0},
            physical_name="column",
            mesh_order=2,
        ),
        # Hexagon.
        PolyPrism(
            polygons=reg_polygon(-3.0, -3.0, 0.8, 6),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="hex",
            mesh_order=8,
        ),
        # Triangle.
        PolyPrism(
            polygons=Polygon([(2.0, -4.0), (4.0, -4.0), (3.0, -2.0)]),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="triangle",
            mesh_order=11,
        ),
        # 24-sided polygon sitting inside disk_a — carves an arc-bounded
        # inset because disk_a's curved boundary is preserved by arc fitting.
        PolyPrism(
            polygons=reg_polygon(-2.0, 0.0, 0.6, 24),
            buffers={0.0: 0.0, 1.0: 0.0},
            physical_name="inset",
            mesh_order=1,
        ),
    ]

    occ_entities = cad_occ(entities_list=entities)
    assert len(occ_entities) == len(entities)

    mm = ModelManager(filename="test_occ_stress")
    try:
        mm.load_occ_entities(occ_entities)

        groups = gmsh.model.getPhysicalGroups()
        group_names = {gmsh.model.getPhysicalName(dim, tag) for dim, tag in groups}

        expected = {
            "slab",
            "disk_a",
            "disk_b",
            "coincident_hi",
            "square_over_disk",
            "ring",
            "pyramid",
            "column",
            "hex",
            "triangle",
            "inset",
        }
        missing = expected - group_names
        assert not missing, f"missing physical groups: {missing}"

        # Exactly-coincident absorbed entity must have no surviving tags.
        assert "coincident_lo" not in group_names

        # Fragmentation should populate the model at every topological dim.
        for d in (0, 1, 2, 3):
            assert len(gmsh.model.getEntities(d)) > 0, f"no entities at dim {d}"

        # At least one interface group at dim 2 (volumes share faces).
        dim2_names = [gmsh.model.getPhysicalName(d, t) for d, t in groups if d == 2]
        assert any("___" in n for n in dim2_names), dim2_names
    finally:
        mm.finalize()


def _physical_names_in_xao(xao_path: Path) -> set[str]:
    """Open xao_path in gmsh and return the set of physical names."""
    gmsh.initialize()
    try:
        gmsh.open(str(xao_path))
        gmsh.model.occ.synchronize()
        return {
            gmsh.model.getPhysicalName(d, t) for d, t in gmsh.model.getPhysicalGroups()
        }
    finally:
        gmsh.finalize()


def test_cad_occ_two_abutting_prisms_share_interface(tmp_path):
    """OCC backend mirror of cad_gmsh's adjacent-prisms-share-interface test."""
    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 2.0: 0.0}
    labeled = cad_occ(
        [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
        ]
    )
    xao = tmp_path / "two_prisms.xao"
    write_xao(labeled, xao)
    names = _physical_names_in_xao(xao)
    assert {"A", "B"} <= names
    assert "A___B" in names or "B___A" in names
    assert "A___None" in names
    assert "B___None" in names


def test_cad_occ_interface_tag_resolves_to_winning_boundary(tmp_path):
    """OCC backend mirror of the InterfaceTag e2e test.

    Two abutting prisms + one InterfaceTag at their shared face. The
    InterfaceTag must produce an `iface` physical group, and A must
    not be internally split by the InterfaceTag panel.
    """
    from shapely.geometry import LineString

    from meshwell.interface_tag import InterfaceTag

    A = shapely.Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
    B = shapely.Polygon([(5, 0), (10, 0), (10, 5), (5, 5)])
    buffers = {0.0: 0.0, 1.0: 0.0}
    labeled = cad_occ(
        [
            PolyPrism(polygons=A, buffers=buffers, physical_name="A", mesh_order=1),
            PolyPrism(polygons=B, buffers=buffers, physical_name="B", mesh_order=2),
            InterfaceTag(
                linestrings=LineString([(5, 0), (5, 5)]),
                zmin=0.0,
                zmax=1.0,
                physical_name="iface",
                mesh_order=3,
            ),
        ]
    )
    xao = tmp_path / "iface.xao"
    write_xao(labeled, xao)
    names = _physical_names_in_xao(xao)
    assert {"A", "B", "iface"} <= names


def test_cad_occ_perturbation_below_point_tolerance():
    """Default perturbation in cad_occ is 1e-5; bound shift stays below point_tolerance."""
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib

    point_tol = 1e-3
    poly = shapely.Polygon([(0, 0), (2, 0), (2, 1), (0, 1)])
    labeled = cad_occ(
        [PolySurface(polygons=poly, physical_name="A", mesh_order=1)],
        point_tolerance=point_tol,
    )
    assert len(labeled) == 1
    bbox = Bnd_Box()
    for shape in labeled[0].shapes:
        BRepBndLib.Add_s(shape, bbox)
    xmin, ymin, _, xmax, ymax, _ = bbox.Get()
    assert abs(xmin - 0.0) < point_tol, xmin
    assert abs(ymin - 0.0) < point_tol, ymin
    assert abs(xmax - 2.0) < point_tol, xmax
    assert abs(ymax - 1.0) < point_tol, ymax


if __name__ == "__main__":
    pytest.main([__file__])
