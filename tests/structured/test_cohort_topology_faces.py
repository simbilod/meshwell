"""build_cohort_topology populates horizontal face registry."""

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


def test_horizontal_faces_registered_per_piece_per_z_plane_single_entity():
    """A single PolyPrism with one z-interval has 1 piece x 2 z-planes = 2 horizontal faces."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
    topology = build_cohort_topology(plan, component_index=0)
    assert len(topology.horizontal_faces) == 2


def test_two_stacked_entities_share_horizontal_face_at_common_z():
    """Two stacked PolyPrisms with the same XY footprint share the face at z=1.

    The horizontal_faces registry is keyed by (z, piece_fingerprint).
    Two slabs with identical face_partition_edges at the same z-plane produce
    the same fingerprint -> same TopoDS_Face TShape (face sharing invariant).

    A(z=0..1) and B(z=1..2) both have piece_fingerprint equal (same arrangement).
    Unique z-planes: {0.0, 1.0, 2.0} -> 3 horizontal face entries total.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    # 3 unique (z, fingerprint) combinations: (0.0, fp), (1.0, fp), (2.0, fp).
    # The face at z=1 is shared between A's top and B's bottom.
    assert len(topology.horizontal_faces) == 3


def test_horizontal_face_outer_wire_uses_registry_edges():
    """The face's outer wire edges must come from topology.horizontal_edges.

    horizontal_edges now stores TopoDS_Wire objects; we collect all edges
    within those wires to build the reference hash set.

    Uses two stacked same-footprint PolyPrisms so the arrangement produces
    4 proper 2-vertex segment edges (one per polygon side). Single isolated
    polygons produce a degenerate one-edge arrangement that requires a
    closing edge not in the registry.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    face_key = next(iter(topology.horizontal_faces))
    face = topology.horizontal_faces[face_key]

    from OCP.TopAbs import TopAbs_EDGE
    from OCP.TopExp import TopExp_Explorer

    face_edge_hashes = set()
    exp = TopExp_Explorer(face, TopAbs_EDGE)
    while exp.More():
        face_edge_hashes.add(hash(exp.Current()))
        exp.Next()

    # horizontal_edges stores TopoDS_Wire; collect edges within each wire.
    registry_edge_hashes: set[int] = set()
    for wire in topology.horizontal_edges.values():
        exp2 = TopExp_Explorer(wire, TopAbs_EDGE)
        while exp2.More():
            registry_edge_hashes.add(hash(exp2.Current()))
            exp2.Next()
    assert (
        face_edge_hashes <= registry_edge_hashes
    ), "Horizontal face's outer wire references edges not in the registry."
