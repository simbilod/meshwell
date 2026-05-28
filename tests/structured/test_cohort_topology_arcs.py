"""Arc support in build_cohort_topology — horizontal edges only (Task 7)."""

from __future__ import annotations

import math

import shapely

from meshwell.polyprism import PolyPrism
from meshwell.structured import StructuredExtrusionResolutionSpec
from meshwell.structured.cohort_topology import build_cohort_topology
from meshwell.structured.plan import build_plan


def _circle(cx, cy, r, n=32):
    """N-gon approximation of a circle.

    PolyPrism's identify_arcs=True detects it as arcs.
    """
    pts = [
        (cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    return shapely.Polygon(pts)


def test_arc_horizontal_edge_is_built_when_arrangement_edge_has_circle():
    """A circular PolyPrism with identify_arcs=True produces arc arrangement edges.

    The topology builder's horizontal_edges entries for arc edges should have
    arc geometry, not be missing.
    """
    poly = _circle(0, 0, 1, n=32)
    A = PolyPrism(
        polygons=poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        identify_arcs=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A])
    topology = build_cohort_topology(plan, component_index=0)

    arc_arr_edges = [e for e in plan.arrangements[0].edges if e.circle is not None]
    assert arc_arr_edges, "Test setup: expected arc edges in arrangement"

    # Every arc arrangement edge must have a registered horizontal_edge at z=0.
    for arr_edge in arc_arr_edges:
        assert (
            0.0,
            arr_edge.edge_id,
        ) in topology.horizontal_edges, f"Arc arrangement edge {arr_edge.edge_id} missing from horizontal_edges at z=0"


def test_arc_horizontal_edge_has_circle_geometry():
    """The registered arc wire must contain an edge of GeomAbs_Circle type.

    horizontal_edges now stores TopoDS_Wire; extract the first edge to check
    its geometry type.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Curve
    from OCP.BRepTools import BRepTools_WireExplorer
    from OCP.GeomAbs import GeomAbs_Circle

    poly = _circle(0, 0, 1, n=32)
    A = PolyPrism(
        polygons=poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        identify_arcs=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A])
    topology = build_cohort_topology(plan, component_index=0)
    arc_arr_edges = [e for e in plan.arrangements[0].edges if e.circle is not None]

    for arr_edge in arc_arr_edges:
        wire = topology.horizontal_edges[(0.0, arr_edge.edge_id)]
        # Extract the first (and only) edge from the arc wire.
        exp = BRepTools_WireExplorer(wire)
        assert exp.More(), "Arc horizontal wire is empty"
        edge = exp.Current()
        adaptor = BRepAdaptor_Curve(edge)
        assert adaptor.GetType() == GeomAbs_Circle, (
            f"Arc arrangement edge produced non-Circle horizontal edge "
            f"(got type {adaptor.GetType()})"
        )


def test_arc_lateral_face_is_cylindrical():
    """A circular cohort's lateral face for an arc edge is cylindrical.

    lateral_faces now stores list[TopoDS_Face]; the first (and only) face for
    an arc edge should be cylindrical.
    """
    from OCP.BRepAdaptor import BRepAdaptor_Surface
    from OCP.GeomAbs import GeomAbs_Cylinder

    poly = _circle(0, 0, 1, n=32)
    A = PolyPrism(
        polygons=poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        identify_arcs=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A])
    topology = build_cohort_topology(plan, component_index=0)
    arc_arr_edges = [e for e in plan.arrangements[0].edges if e.circle is not None]
    assert arc_arr_edges, "Test setup: expected arc edges"

    for arr_edge in arc_arr_edges:
        face_list = topology.lateral_faces[(0, arr_edge.edge_id)]  # slab_index=0
        assert isinstance(face_list, list), "Expected list of faces"
        assert face_list, "Expected non-empty list of faces"
        face = face_list[0]
        adaptor = BRepAdaptor_Surface(face)
        assert (
            adaptor.GetType() == GeomAbs_Cylinder
        ), f"Arc lateral face is not cylindrical (got {adaptor.GetType()})"


def test_arc_horizontal_edge_endpoints_are_registry_vertices():
    """After Task 8's vertex snap, arc horizontal edges use registry vertices."""
    from OCP.TopAbs import TopAbs_VERTEX
    from OCP.TopExp import TopExp_Explorer

    poly = _circle(0, 0, 1, n=32)
    A = PolyPrism(
        polygons=poly,
        buffers={0.0: 0.0, 1.0: 0.0},
        physical_name="A",
        mesh_order=1,
        structured=True,
        identify_arcs=True,
        resolutions=[StructuredExtrusionResolutionSpec(n_layers=[2])],
    )
    plan = build_plan([A])
    topology = build_cohort_topology(plan, component_index=0)
    arc_arr_edges = [e for e in plan.arrangements[0].edges if e.circle is not None]
    assert arc_arr_edges

    registry_vertex_hashes = {hash(v) for v in topology.vertices.values()}
    arc_edge = topology.horizontal_edges[(0.0, arc_arr_edges[0].edge_id)]
    endpoints = set()
    exp = TopExp_Explorer(arc_edge, TopAbs_VERTEX)
    while exp.More():
        endpoints.add(hash(exp.Current()))
        exp.Next()
    assert endpoints <= registry_vertex_hashes, (
        "Arc horizontal edge's endpoints are NOT in the vertex registry. "
        "Vertex snap not applied or applied incorrectly."
    )
