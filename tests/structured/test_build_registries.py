import numpy as np

from meshwell.structured.build import EdgeRegistry, VertexRegistry


def test_vertex_registry_dedups_within_tolerance():
    reg = VertexRegistry(point_tolerance=1e-3)
    v1 = reg.get_or_create(0.0, 0.0, 0.0)
    v2 = reg.get_or_create(1e-4, 0.0, 0.0)  # within tol → same vertex
    v3 = reg.get_or_create(1.0, 0.0, 0.0)
    assert v1 is v2
    assert v1 is not v3
    assert len(reg) == 2


def test_edge_registry_dedups_xy_edges_at_z():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vreg, point_tolerance=1e-3)
    coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0), (0.0, 0.0)]
    e1 = ereg.polyline_xy(coords, z=0.0, identify_arcs=False)
    e2 = ereg.polyline_xy(coords, z=0.0, identify_arcs=False)
    # Same path at same z → same edge sequence.
    assert e1 == e2


def test_edge_registry_arc_detected():
    vreg = VertexRegistry(point_tolerance=1e-3)
    ereg = EdgeRegistry(vreg, point_tolerance=1e-3)
    # 16-pt sampling of a unit circle.
    n = 16
    coords = [(np.cos(a), np.sin(a)) for a in np.linspace(0, 2 * np.pi, n + 1)]
    edges = ereg.polyline_xy(
        coords,
        z=0.0,
        identify_arcs=True,
        min_arc_points=4,
        arc_tolerance=1e-2,
    )
    # All edges should be arc edges (single full-circle case decomposes
    # to one or more arc segments, depending on seam logic).
    # We at least require: total edge count < n (some were merged into arcs).
    assert len(edges) < n
