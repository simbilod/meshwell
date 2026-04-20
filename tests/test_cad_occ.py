from __future__ import annotations

from pathlib import Path

import pytest
import shapely

import gmsh
from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_entity import OCC_entity
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface


def test_occ_polysurface():
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
    output_xao = Path("test_polysurface_occ.xao")
    write_xao(occ_entities, output_xao)

    assert output_xao.exists()

    # 3. Mesh from XAO
    output_msh = Path("test_polysurface_occ.msh")
    mesh(
        input_file=output_xao,
        output_file=output_msh,
        dim=2,
        default_characteristic_length=0.5,
        n_threads=1,
    )
    assert output_msh.exists()


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


def test_occ_buffered_prism():
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

    output_xao = Path("test_buffered_prism_occ.xao")
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


def test_occ_rounded_rect_inside_rect_with_cutout_shares_arcs():
    """An inner rounded rectangle fits exactly into a rounded cutout in a bigger rectangle.

    It must share its curved side faces with the outer prism. Verifies that:
      - fragmentation preserves topology sharing across arc-fitted boundaries,
      - the interface physical group owns cylindrical (arc-swept) faces, not
        duplicate coincident ones.
    """
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.model import ModelManager

    def rounded_rect_coords(w, h, r, n_arc=8):
        """CCW coords of a rounded rectangle: w x h with corner radius r.

        Each corner is sampled with n_arc+1 points so the arc identifier has
        a clean run of collinear-on-circle points to fit, and the straight
        edges between corners are a single segment — avoids the noisy vertex
        sequences that shapely.buffer(... join_style='round') produces.
        """
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
        return coords

    inner_coords = rounded_rect_coords(w=4.0, h=3.0, r=0.6, n_arc=8)
    rounded = Polygon(inner_coords)

    # Big rectangle with a rounded-rect hole. Hole uses the inner's EXACT
    # coords, reversed for proper orientation — so BOPAlgo sees identical
    # vertex sequences on both sides and BREP sub-shape sharing actually
    # kicks in. The whole point of the test.
    outer_poly = (
        Polygon(  # noqa: S604 — shapely Polygon(shell=...) kwarg, not subprocess
            shell=[(-5.0, -5.0), (5.0, -5.0), (5.0, 5.0), (-5.0, 5.0)],
            holes=[inner_coords[::-1]],
        )
    )

    outer = PolyPrism(
        polygons=outer_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="outer",
        mesh_order=5,
        identify_arcs=True,
    )
    inner = PolyPrism(
        polygons=rounded,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="inner",
        mesh_order=3,
        identify_arcs=True,
    )

    occ_entities = cad_occ(entities_list=[outer, inner])
    assert len(occ_entities) == 2

    mm = ModelManager(filename="test_rounded_rect_in_cutout")
    try:
        mm.load_occ_entities(occ_entities)

        groups = gmsh.model.getPhysicalGroups()
        by_name = {gmsh.model.getPhysicalName(d, t): (d, t) for d, t in groups}
        assert {"outer", "inner"}.issubset(by_name)

        # Locate the interface group and check BREP sharing.
        interface_name = next(
            n for n in by_name if n in {"outer___inner", "inner___outer"}
        )
        dim, tag = by_name[interface_name]
        assert dim == 2, (interface_name, dim)

        iface_surfaces = gmsh.model.getEntitiesForPhysicalGroup(2, tag)
        # 4 planar sides + 4 cylindrical (arc-swept) corner faces = 8.
        # If BREP sharing had failed, pieces would double to 16.
        assert 4 <= len(iface_surfaces) <= 12, iface_surfaces

        surface_types = [gmsh.model.getType(2, s) for s in iface_surfaces]
        # Cylindrical faces are the tell-tale sign that the arc fitting
        # survived the BREP round-trip into GMSH.
        assert any("ylind" in t for t in surface_types), surface_types

        # Confirm the curved sides are bounded by Circle edges (shared arcs
        # at the top and bottom of each cylindrical face).
        circle_edges = [
            e for e in gmsh.model.getEntities(1) if gmsh.model.getType(*e) == "Circle"
        ]
        assert circle_edges, "no Circle edges — arcs were not preserved"

        # Exactly two volumes survived (outer + inner), sharing interior faces.
        assert len(gmsh.model.getEntities(3)) == 2
    finally:
        mm.finalize()


def test_occ_shapely_difference_rounded_rect_shares_all_arcs():
    """Outer polygon from shapely.difference must share all arcs and surfaces with the inner.

    This is the regression gate for the three hardening measures applied to
    entity ingestion: shapely.set_precision snapping, consecutive-duplicate
    stripping, and canonical seam rotation. Without them, shapely's diffed
    outer ring has a duplicated seam vertex (producing a sliver inner piece)
    and a rotated coord order (breaking arc sharing).
    """
    import numpy as np
    from shapely.geometry import Polygon

    from meshwell.model import ModelManager

    def rounded_rect_coords(w, h, r, n_arc=8):
        hw, hh = w / 2, h / 2
        specs = [
            ((hw - r, hh - r), 0.0),
            ((-hw + r, hh - r), np.pi / 2),
            ((-hw + r, -hh + r), np.pi),
            ((hw - r, -hh + r), 3 * np.pi / 2),
        ]
        out = []
        for (cx, cy), a0 in specs:
            out.extend(
                (cx + r * np.cos(a), cy + r * np.sin(a))
                for a in np.linspace(a0, a0 + np.pi / 2, n_arc + 1)
            )
        return out

    inner_coords = rounded_rect_coords(w=4.0, h=3.0, r=0.6, n_arc=8)
    inner_poly = Polygon(inner_coords)
    bigger = Polygon([(-5, -5), (5, -5), (5, 5), (-5, 5)])
    outer_diffed = bigger.difference(inner_poly)

    outer_prism = PolyPrism(
        polygons=outer_diffed,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="outer",
        mesh_order=1,
        identify_arcs=True,
    )
    inner_prism = PolyPrism(
        polygons=inner_poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="inner",
        mesh_order=2,
        identify_arcs=True,
    )
    occ_ents = cad_occ([outer_prism, inner_prism])
    by_name = {e.physical_name[0]: e for e in occ_ents}

    # Inner must be a single piece — the duplicate-seam sliver is gone.
    assert len(by_name["inner"].shapes) == 1

    mm = ModelManager(filename="test_shapely_diff_sharing")
    try:
        mm.load_occ_entities(occ_ents)
        groups = gmsh.model.getPhysicalGroups(2)
        named = {gmsh.model.getPhysicalName(d, t): t for d, t in groups}
        interface_name = next(
            n for n in named if n in {"outer___inner", "inner___outer"}
        )
        tag = named[interface_name]
        faces = gmsh.model.getEntitiesForPhysicalGroup(2, tag)
        # Lateral wall of the inner prism: 4 cylinders + 4 planes.
        types = sorted(gmsh.model.getType(2, f) for f in faces)
        n_cyl = sum(1 for t in types if "ylind" in t)
        n_plane = sum(1 for t in types if "lane" in t)
        assert n_cyl == 4, types
        assert n_plane == 4, types
    finally:
        mm.finalize()


if __name__ == "__main__":
    pytest.main([__file__])
