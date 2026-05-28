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


def test_two_entities_same_cohort_get_separate_horizontal_face_entries():
    """Two PolyPrisms in the same cohort get separate face entries.

    Different source_index -> different piece_ids -> separate face entries
    even at shared z-planes.
    """
    plan = build_plan([_polyprism("A", 0, 1, 1), _polyprism("B", 1, 2, 2)])
    topology = build_cohort_topology(plan, component_index=0)
    # 2 entities x 1 piece x 2 z-planes per entity = 4 faces total.
    # (At the shared z=1 plane, A and B contribute separately.)
    assert len(topology.horizontal_faces) == 4


def test_horizontal_face_outer_wire_uses_registry_edges():
    """The face's outer wire edges must come from topology.horizontal_edges."""
    plan = build_plan([_polyprism("A", 0, 1, 1)])
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

    registry_edge_hashes = {hash(e) for e in topology.horizontal_edges.values()}
    assert (
        face_edge_hashes <= registry_edge_hashes
    ), "Horizontal face's outer wire references edges not in the registry."
