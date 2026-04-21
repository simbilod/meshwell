"""Cross-entity face-TShape sharing via the OCC geometry cache.

These tests pin the property that two entities with geometrically identical
outer wires receive the same ``TopoDS_Face`` TShape when they share an
``OCCGeometryCache``. Without the cache, every entity mints its own face
TShape, which is what leads ``BOPAlgo_Builder`` to emit near-coincident
duplicate facets at shared boundaries (surfacing as dihedral-0 or PLC
boundary-recovery errors downstream).

The scenes use arcs and low-height extrusions on purpose: those are the
cases where drift between independently-built TShapes is most pronounced,
and where a BOPAlgo-only pipeline most often leaves unfused duplicates.
"""
from __future__ import annotations

import numpy as np
import shapely
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopTools import TopTools_ShapeMapHasher

from meshwell.cad_occ import cad_occ
from meshwell.mesh import mesh
from meshwell.occ_geometry_cache import OCCGeometryCache
from meshwell.occ_xao_writer import write_xao
from meshwell.polyprism import PolyPrism
from meshwell.polysurface import PolySurface

_HASHER = TopTools_ShapeMapHasher()


def _tshape_hashes(shape, kind) -> list[int]:
    out: list[int] = []
    exp = TopExp_Explorer(shape, kind)
    while exp.More():
        out.append(_HASHER(exp.Current()))
        exp.Next()
    return out


def _arc_polygon(n: int = 30, r: float = 1.0) -> shapely.Polygon:
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return shapely.Polygon([(r * np.cos(t), r * np.sin(t)) for t in theta])


def _keyhole_polygon() -> shapely.Polygon:
    """A 10x6 rectangle with a circular bite on one side — mixes arcs+lines."""
    rect = shapely.Polygon([(0, 0), (10, 0), (10, 6), (0, 6)])
    bite = shapely.Polygon(
        [(5 + np.cos(t), 6 + np.sin(t)) for t in np.linspace(0, -np.pi, 20)]
    )
    return rect.difference(bite)


def test_polysurface_arc_face_shared_across_entities():
    """Two PolySurfaces on the same arc polygon share their face TShape."""
    poly = _arc_polygon()
    cache = OCCGeometryCache(point_tolerance=1e-3)

    a = PolySurface(polygons=poly, physical_name="a", identify_arcs=True)
    b = PolySurface(polygons=poly, physical_name="b", identify_arcs=True)

    fa = _tshape_hashes(a.instanciate_occ(occ_cache=cache), TopAbs_FACE)
    fb = _tshape_hashes(b.instanciate_occ(occ_cache=cache), TopAbs_FACE)

    assert fa == fb, "identical arc polygons must produce one cached face TShape"


def test_polysurface_keyhole_face_shared_across_entities():
    """Face sharing holds for mixed arc + straight-line outer boundaries."""
    poly = _keyhole_polygon()
    cache = OCCGeometryCache(point_tolerance=1e-3)

    a = PolySurface(polygons=poly, physical_name="a", identify_arcs=True)
    b = PolySurface(polygons=poly, physical_name="b", identify_arcs=True)

    fa = _tshape_hashes(a.instanciate_occ(occ_cache=cache), TopAbs_FACE)
    fb = _tshape_hashes(b.instanciate_occ(occ_cache=cache), TopAbs_FACE)
    assert fa == fb


def test_polyprism_low_height_bottom_face_shared():
    """Two arc-based prisms with a short extrusion share the bottom face TShape.

    ``BRepPrimAPI_MakePrism`` mints fresh top and side TShapes per call, so
    only the bottom face is expected to match across entities — but that
    match is what BOPAlgo needs to anchor SameDomain fusion of the rest.
    """
    poly = _arc_polygon()
    cache = OCCGeometryCache(point_tolerance=1e-3)

    # "Low height" — 1 % of the polygon's radius.
    buffers = {0.0: 0.0, 0.01: 0.0}
    a = PolyPrism(polygons=poly, buffers=buffers, physical_name="a", identify_arcs=True)
    b = PolyPrism(polygons=poly, buffers=buffers, physical_name="b", identify_arcs=True)

    fa = set(_tshape_hashes(a.instanciate_occ(occ_cache=cache), TopAbs_FACE))
    fb = set(_tshape_hashes(b.instanciate_occ(occ_cache=cache), TopAbs_FACE))

    assert fa & fb, "at least the bottom face must be cached + shared"


def test_adjacent_polyprisms_share_edges_via_cache():
    """Two side-by-side prisms with arc boundaries share their interface edges.

    The shared edge runs vertically between the two prisms; both wires
    reference the same cached edge even though one traverses it FORWARD
    and the other REVERSED. Sharing is TShape-level (orientation-agnostic
    hash).
    """
    # Two keyhole polygons stitched along x=10.
    left = _keyhole_polygon()
    right = shapely.affinity.translate(left, xoff=10)

    cache = OCCGeometryCache(point_tolerance=1e-3)
    buffers = {0.0: 0.0, 0.02: 0.0}  # low-height extrude

    a = PolyPrism(
        polygons=left,
        buffers=buffers,
        physical_name="left",
        identify_arcs=True,
    )
    b = PolyPrism(
        polygons=right,
        buffers=buffers,
        physical_name="right",
        identify_arcs=True,
    )
    ea = set(_tshape_hashes(a.instanciate_occ(occ_cache=cache), TopAbs_EDGE))
    eb = set(_tshape_hashes(b.instanciate_occ(occ_cache=cache), TopAbs_EDGE))

    assert ea & eb, "adjacent prisms must share the interface edges"


def test_cad_occ_end_to_end_mesh_complex_arcs_low_height(tmp_path):
    """End-to-end: fragment + mesh a scene of arc prisms with small dz.

    Regression against a class of ``could not recover boundary mesh``
    failures: prior to the cache, two keyhole prisms stitched at an arc
    face produced near-coincident but not-identical facets. With face and
    edge caching, BOPAlgo fuses them cleanly and tetgen succeeds.
    """
    left = _keyhole_polygon()
    right = shapely.affinity.translate(left, xoff=10)

    buffers = {0.0: 0.0, 0.05: 0.0}  # ratio ~0.005
    a = PolyPrism(
        polygons=left,
        buffers=buffers,
        physical_name="left",
        identify_arcs=True,
        mesh_order=1,
    )
    b = PolyPrism(
        polygons=right,
        buffers=buffers,
        physical_name="right",
        identify_arcs=True,
        mesh_order=2,
    )

    xao = tmp_path / "complex.xao"
    write_xao(cad_occ([a, b]), xao)

    m = mesh(
        dim=3,
        input_file=xao,
        output_file=tmp_path / "complex.msh",
        n_threads=1,
        default_characteristic_length=1.0,
        verbosity=0,
    )
    # meshio returns a mesh with tet blocks; the assertion fails loudly if
    # tetgen couldn't recover the boundary.
    assert any(block.type == "tetra" and len(block.data) > 0 for block in m.cells)
