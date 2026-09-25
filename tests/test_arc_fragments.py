"""Regression tests for fragmented arcs on grid-snapped rings.

``PolyPrism`` snaps coordinates to ``point_tolerance`` (1 nm by default).
With ``arc_tolerance`` equal to the grid, the resulting radial noise split
one physical arc into several greedy fragments on visibly different
circles; ``canonicalize_ring_segments`` then joined nearly coincident
circles far from the shared vertex and the 3-point emitter built wrong-way
(~360 degree) arcs -- a gmsh segfault, ``CohortShellModifiedError`` or
"Impossible to mesh periodic surface". Fixed by the 2x-grid default
``arc_tolerance`` plus the junction guard in ``canonicalize_ring_segments``.
"""

import numpy as np
import pytest
import shapely

from meshwell.cad_settings import DEFAULT_ARC_TOLERANCE, DEFAULT_POINT_TOLERANCE
from meshwell.geometry_entity import (
    DecompositionSegment,
    _decompose_vertices_3d,
    canonicalize_ring_segments,
)

POINT_TOL = DEFAULT_POINT_TOLERANCE
ARC_TOL = DEFAULT_ARC_TOLERANCE


def _sector(r0=4.0, r1=4.5, a0=45.0, a1=75.0, n=64) -> shapely.Polygon:
    t = np.radians(np.linspace(a0, a1, n))
    outer = np.c_[r1 * np.cos(t), r1 * np.sin(t)]
    inner = np.c_[r0 * np.cos(t), r0 * np.sin(t)][::-1]
    return shapely.Polygon(np.vstack([outer, inner]))


def _snap(geom):
    # Same snapping PolyPrism applies in its constructor.
    return shapely.set_precision(geom, grid_size=POINT_TOL)


def _decompose(ring_coords):
    verts = [(x, y, 0.0) for x, y in ring_coords]
    return _decompose_vertices_3d(
        verts,
        point_tolerance=POINT_TOL,
        identify_arcs=True,
        min_arc_points=5,
        arc_tolerance=ARC_TOL,
    )


def _arc_span_deg(seg: DecompositionSegment) -> float:
    p = np.array(seg.points)
    ang = np.unwrap(np.arctan2(p[:, 1] - seg.center[1], p[:, 0] - seg.center[0]))
    return float(np.degrees(abs(ang[-1] - ang[0])))


def _assert_two_sector_arcs(segments, radii=(4.0, 4.5), span=30.0):
    arcs = [s for s in segments if s.is_arc]
    assert len(arcs) == 2, [(s.radius, len(s.points)) for s in arcs]
    assert sorted(s.radius for s in arcs) == pytest.approx(sorted(radii), abs=5e-3)
    for s in arcs:
        assert _arc_span_deg(s) == pytest.approx(span, abs=0.1)


@pytest.mark.parametrize("reverse", [False, True])
def test_snapped_sector_exterior_is_two_arcs(reverse):
    coords = list(_snap(_sector()).exterior.coords)
    if reverse:
        coords = coords[::-1]
    _assert_two_sector_arcs(_decompose(coords))


@pytest.mark.parametrize("reverse", [False, True])
def test_snapped_sector_hole_is_two_arcs(reverse):
    holed = _snap(shapely.box(-1, -1, 5, 5).difference(_sector()))
    assert len(holed.interiors) == 1
    coords = list(holed.interiors[0].coords)
    if reverse:
        coords = coords[::-1]
    _assert_two_sector_arcs(_decompose(coords))


def test_snapped_bend_with_tangent_arm():
    # Sector plus a straight arm leaving tangentially from the 75 deg end:
    # line->arc junctions are tangent, the arcs must stay whole.
    t = np.radians(75.0)
    tangent = np.array([-np.sin(t), np.cos(t)])
    p_in = 4.0 * np.array([np.cos(t), np.sin(t)])
    p_out = 4.5 * np.array([np.cos(t), np.sin(t)])
    arm = shapely.Polygon([p_in, p_out, p_out + 2.0 * tangent, p_in + 2.0 * tangent])
    bend = _snap(shapely.union(_sector(), arm).buffer(0))
    arcs = [s for s in _decompose(list(bend.exterior.coords)) if s.is_arc]
    # Inner arc (no tangent neighbour on the noisy side) is one arc. The
    # outer arc may keep a short end fragment -- the greedy scan can fold
    # the straight arm plus a few outer samples into one huge-radius arc
    # (within arc_tolerance) -- so the carved geometry must still build and
    # mesh (junction guard).
    inner = [s for s in arcs if abs(s.radius - 4.0) < 0.3]
    assert len(inner) == 1
    assert inner[0].radius == pytest.approx(4.0, abs=5e-3)
    assert 29.9 < _arc_span_deg(inner[0]) <= 30.1

    from meshwell.orchestrator import generate_mesh

    dom = shapely.box(-3, -1, 5, 6)
    mesh = generate_mesh(
        entities=[_prism(dom.difference(bend), 0.0, 0.4, "clad")],
        dim=3,
        output_mesh=None,
        default_characteristic_length=0.5,
        identify_arcs=True,
    )
    assert "tetra" in _cell_types(mesh)


def test_distinct_tangent_arcs_are_not_merged():
    # S-bend: two arcs of different circles meeting tangentially must stay
    # two arcs even at the looser default tolerance.
    t1 = np.radians(np.linspace(-90.0, 0.0, 40))
    a1 = np.c_[3.0 * np.cos(t1), 3.0 + 3.0 * np.sin(t1)]
    t2 = np.radians(np.linspace(180.0, 90.0, 40))[1:]
    a2 = np.c_[6.0 + 3.0 * np.cos(t2), 3.0 + 3.0 * np.sin(t2)]
    path = np.vstack([a1, a2])
    verts = [(x, y, 0.0) for x, y in np.round(path, 3)]
    segs = _decompose_vertices_3d(verts, POINT_TOL, True, 5, ARC_TOL)
    arcs = [s for s in segs if s.is_arc]
    assert len(arcs) == 2
    assert [s.radius for s in arcs] == pytest.approx([3.0, 3.0], abs=5e-3)
    assert arcs[0].center[:2] != pytest.approx(arcs[1].center[:2], abs=0.1)


def test_canonicalize_junction_stays_near_vertex_for_near_coincident_circles():
    # Two fragments of the same physical arc on circles that differ by
    # 2e-2 in radius: their exact intersection is far from the shared
    # vertex, the guard must keep the junction local.
    z = 0.0
    ang = np.radians(np.linspace(45.0, 75.0, 31))
    pts = [(4.0 * np.cos(a), 4.0 * np.sin(a), z) for a in ang]
    s1 = DecompositionSegment(
        points=pts[:16], is_arc=True, center=(0.0, 0.0, z), radius=4.0
    )
    s2 = DecompositionSegment(
        points=pts[15:], is_arc=True, center=(0.02, 0.0, z), radius=4.021
    )
    shared = pts[15]
    out = canonicalize_ring_segments(
        [s1, s2], None, eps=0.0, match_tolerance=2e-3, slack=1e-3
    )
    j = out[0].points[-1]
    assert out[1].points[0] == j
    assert np.hypot(j[0] - shared[0], j[1] - shared[1]) < 0.02


def _prism(poly, z0, z1, name, structured=False, mesh_order=None):
    from meshwell.polyprism import PolyPrism

    return PolyPrism(
        polygons=poly,
        buffers={z0: 0.0, z1: 0.0},
        physical_name=name,
        structured=structured,
        mesh_order=mesh_order,
    )


def _cell_types(mesh):
    return {c.type for c in mesh.cells}


def test_box_minus_sector_meshes_with_arcs():
    from meshwell.orchestrator import generate_mesh

    dom = shapely.box(-1, -1, 5, 5)
    mesh = generate_mesh(
        entities=[_prism(dom.difference(_sector()), 0.0, 0.4, "clad")],
        dim=3,
        output_mesh=None,
        default_characteristic_length=0.5,
        identify_arcs=True,
    )
    assert "tetra" in _cell_types(mesh)


def test_structured_sector_with_carved_cladding_meshes():
    from meshwell.orchestrator import generate_mesh
    from meshwell.resolution import StructuredExtrusionResolutionSpec

    arc = _sector()
    dom = shapely.box(-1, -1, 5, 5)
    entities = [
        _prism(arc, 0.0, 0.4, "arc", structured=True, mesh_order=1),
        _prism(dom, -1.0, 0.0, "substrate", mesh_order=5),
        _prism(dom, 0.4, 1.0, "clad_top", mesh_order=5),
        _prism(dom.difference(arc), 0.0, 0.4, "clad", mesh_order=5),
    ]
    mesh = generate_mesh(
        entities=entities,
        dim=3,
        output_mesh=None,
        default_characteristic_length=0.5,
        resolution_specs={"arc": [StructuredExtrusionResolutionSpec(n_layers=2)]},
        identify_arcs=True,
    )
    assert {"wedge", "tetra"} <= _cell_types(mesh)
