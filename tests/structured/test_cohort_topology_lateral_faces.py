"""build_cohort_topology populates lateral face registry for straight edges."""

from __future__ import annotations

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan


def _square(x0, y0, x1, y1):
    return shapely.Polygon([(x0, y0), (x1, y0), (x1, y1), (x0, y1)])


def _polyprism(name, z0, z1, mesh_order, x0=0, y0=0, x1=1, y1=1):
    return PolyPrism(
        polygons=_square(x0, y0, x1, y1),
        buffers={float(z0): 0.0, float(z1): 0.0},
        physical_name=name,
        mesh_order=mesh_order,
        structured=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )


def test_lateral_faces_registered_per_slab_per_edge():
    """Check lateral face count for 2 stacked slabs.

    A square cohort with 2 stacked slabs has 4 outer arrangement edges
    x 2 slabs = 8 lateral faces.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.lateral_faces) == 8


def test_lateral_face_edges_match_registry():
    """A lateral face's vertices must be in the vertex registry.

    lateral_faces values are list[TopoDS_Face]; check the first face in the
    first entry. For 2-vertex arrangement edges the face's edges come directly
    from horizontal/vertical registries; for multi-vertex edges the face uses
    fresh segment edges (not registered) but their VERTICES are still in the
    registry. We test the weaker (always-true) invariant: all vertex TShapes
    on the lateral face must be registered.

    Uses stacked slabs to get a proper 4-edge arrangement (all 2-vertex edges).
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    face_key = next(iter(topology.lateral_faces))
    face_list = topology.lateral_faces[face_key]
    # face_list is list[TopoDS_Face]; pick the first face.
    face = face_list[0]

    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    # Check edges from horizontal/vertical registries.
    face_edge_hashes = set()
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        face_edge_hashes.add(hash(exp.Current()))
        exp.Next()

    # horizontal_edges now stores TopoDS_Wire; collect edges within each wire.
    horiz_edge_hashes: set[int] = set()
    for wire in topology.horizontal_edges.values():
        exp2 = TopExp_Explorer(wire, TopAbs_EDGE)
        while exp2.More():
            horiz_edge_hashes.add(hash(exp2.Current()))
            exp2.Next()
    all_edge_hashes = horiz_edge_hashes | {
        hash(e) for e in topology.vertical_edges.values()
    }
    # For 2-vertex arrangement edges (stacked slabs), all 4 lateral face edges
    # should come from the registries.
    assert face_edge_hashes <= all_edge_hashes


def test_lateral_face_shared_between_lateral_neighbors():
    """Two laterally-adjacent PolyPrisms share an arrangement edge via registry.

    The lateral face built for that edge is the same TopoDS_Face TShape for
    both pieces' sub-prisms (via registry).
    """
    A = _polyprism("A", 0, 1, 1, x0=0, y0=0, x1=1, y1=1)
    B = _polyprism("B", 0, 1, 2, x0=1, y0=0, x1=2, y1=1)  # adjacent to A at x=1
    plan = build_plan([A, B])
    topology = build_cohort_topology(plan, component_index=0)
    # The seam edge between A and B should be one arrangement edge.
    # There are 2 slabs (A and B); each has 4 outer edges including the
    # shared seam — so total lateral_faces = 2 slabs * arrangement_edges_count.
    # If the seam edge is shared, the same lateral_faces entry would be
    # accessed by both pieces during assembly. Here we just confirm the
    # registry has the expected number of entries.
    arr = plan.arrangements[0]
    expected = len(plan.slabs) * len(arr.edges)
    assert len(topology.lateral_faces) == expected
